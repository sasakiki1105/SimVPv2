#!/usr/bin/env python3
"""Use already-saved scalar moments to refine the B15/B25 closure diagnosis.

The current stitched PIC files contain electron drift velocity components and
one scalar temperature. They do not contain a pressure tensor. This script
tests whether those available moments reduce analogue-future ambiguity before
spending another PIC run on tensor diagnostics.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze_radaz_b15_b25_predictive_state_ambiguity as ambiguity
import analyze_radaz_magnetic_sweep_rom as magnetic_rom


ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT.parent
OUTPUT = (
    RESEARCH
    / "research_results"
    / "2D_RadAz"
    / "SimVPv2"
    / "workdirs"
    / "analyze_radaz_b15_b25_pre_repic_moments"
)

CASES = ("B15", "B25")
FIELDS = ("electron_ud", "electron_vd", "electron_wd", "electron_Temp")
RADIAL_BANDS = 8
MAX_MODE = 30
PCA_MAX_COMPONENTS = 20
PCA_VARIANCE_TARGET = 0.95
HORIZONS_US = ambiguity.HORIZONS_US
FOCAL_HORIZON_US = ambiguity.PRIMARY_HORIZON_US
STATE_ORDER = (
    "L",
    "L+U",
    "L+Tiso",
    "L+U+Tiso",
    "L+organisation",
    "L+organisation+U+Tiso",
)
COLORS = {
    "L": "#4d4d4d",
    "L+U": "#0072b2",
    "L+Tiso": "#e69f00",
    "L+U+Tiso": "#009e73",
    "L+organisation": "#cc79a7",
    "L+organisation+U+Tiso": "#d55e00",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def case_b(case_name: str) -> int:
    return int(case_name[1:])


def extract_moment_fourier(case_name: str, output: Path) -> Path:
    cache = output / f"{case_name}_available_electron_moments.h5"
    if cache.is_file():
        return cache
    source_path = magnetic_rom.fields_path(case_b(case_name))
    with h5py.File(source_path, "r") as source:
        time_us_all = np.asarray(source["axes/time_s"], dtype=np.float64) * 1.0e6
        frames = int(np.searchsorted(time_us_all, ambiguity.COMMON_END_US + 1.0e-9))
        frames = min(frames, len(time_us_all))
        time_us = time_us_all[:frames]
        band_edges = np.linspace(0, 257, RADIAL_BANDS + 1, dtype=int)
        coefficient = np.empty(
            (frames, len(FIELDS), RADIAL_BANDS, MAX_MODE + 1),
            dtype=np.complex64,
        )
        for start in range(0, frames, 16):
            stop = min(start + 16, frames)
            for field_index, field in enumerate(FIELDS):
                raw = np.asarray(
                    source[f"fields/{field}"][start:stop, :257, :256],
                    dtype=np.float32,
                )
                for band in range(RADIAL_BANDS):
                    radial_mean = np.mean(
                        raw[:, band_edges[band] : band_edges[band + 1], :], axis=1
                    )
                    coefficient[start:stop, field_index, band] = (
                        np.fft.rfft(radial_mean, axis=-1)[..., : MAX_MODE + 1] / 256.0
                    )
            if stop == frames or stop % 400 == 0:
                print(f"[MOMENT] {case_name}: {stop}/{frames}", flush=True)
    packed = np.concatenate(
        (coefficient[..., :1].real, coefficient[..., 1:].real, coefficient[..., 1:].imag),
        axis=-1,
    ).astype(np.float32)
    features = packed.reshape(frames, len(FIELDS), -1)
    with h5py.File(cache, "w") as target:
        target.create_dataset("time_us", data=time_us)
        target.create_dataset("features", data=features, compression="gzip", compression_opts=4)
        target.create_dataset("fields", data=np.asarray(FIELDS, dtype="S"))
        target.create_dataset("radial_band_edges", data=band_edges)
        target.attrs["source_h5"] = str(source_path)
        target.attrs["radial_bands"] = RADIAL_BANDS
        target.attrs["max_mode"] = MAX_MODE
        target.attrs["packing"] = "Re(n=0), Re(n=1..N), Im(n=1..N)"
    return cache


def nearest_sample(source_time: np.ndarray, values: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(source_time, target_time)
    indices = np.clip(indices, 1, len(source_time) - 1)
    left = indices - 1
    choose_left = np.abs(target_time - source_time[left]) <= np.abs(
        source_time[indices] - target_time
    )
    indices = np.where(choose_left, left, indices)
    return values[indices]


def fit_group_pca(
    values: np.ndarray,
    fit: np.ndarray,
    max_components: int = PCA_MAX_COMPONENTS,
) -> tuple[np.ndarray, dict]:
    centered = values[fit] - np.mean(values[fit], axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(centered * centered)))
    scaled = values / max(scale, 1.0e-12)
    count = min(max_components, int(np.count_nonzero(fit)) - 1, scaled.shape[1])
    pca = PCA(n_components=count, svd_solver="randomized", random_state=42)
    pca.fit(scaled[fit])
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    selected = min(int(np.searchsorted(cumulative, PCA_VARIANCE_TARGET) + 1), count)
    coordinates = pca.transform(scaled)[:, :selected]
    coordinates = ambiguity.standardize(coordinates, fit)
    return coordinates, {
        "input_dimensions": int(values.shape[1]),
        "computed_components": int(count),
        "selected_components": int(selected),
        "variance_capture": float(cumulative[selected - 1]),
        "global_rms_scale": scale,
    }


def build_case(case_name: str, output: Path) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    time_us, latent, old_states, old_meta = ambiguity.build_states(case_name, 1.5, 0.15, 30)
    cache = extract_moment_fourier(case_name, output)
    with h5py.File(cache, "r") as source:
        moment_time = np.asarray(source["time_us"], dtype=np.float64)
        features = np.asarray(source["features"], dtype=np.float64)
    features = nearest_sample(moment_time, features, time_us)
    fit = (time_us >= ambiguity.COMMON_START_US - 1.0e-9) & (
        time_us < ambiguity.REPRESENTATION_END_US - 1.0e-9
    )
    # Balance the three velocity components before their joint PCA.
    balanced = features.copy()
    channel_scales = []
    for channel in range(len(FIELDS)):
        local = features[fit, channel]
        local = local - np.mean(local, axis=0, keepdims=True)
        channel_scale = max(float(np.sqrt(np.mean(local * local))), 1.0e-12)
        balanced[:, channel] /= channel_scale
        channel_scales.append(channel_scale)
    u_raw = balanced[:, :3].reshape(len(time_us), -1)
    t_raw = balanced[:, 3:].reshape(len(time_us), -1)
    ut_raw = balanced.reshape(len(time_us), -1)
    u, u_meta = fit_group_pca(u_raw, fit)
    temperature, t_meta = fit_group_pca(t_raw, fit)
    ut, ut_meta = fit_group_pca(ut_raw, fit)
    organisation = old_states["L+organisation"][:, latent.shape[1] :]
    states = {
        "L": latent,
        "L+U": np.concatenate((latent, u), axis=1),
        "L+Tiso": np.concatenate((latent, temperature), axis=1),
        "L+U+Tiso": np.concatenate((latent, ut), axis=1),
        "L+organisation": old_states["L+organisation"],
        "L+organisation+U+Tiso": np.concatenate((latent, organisation, ut), axis=1),
    }
    groups = {
        "U": u,
        "Tiso": temperature,
        "U+Tiso": ut,
        "organisation": organisation,
    }
    metadata = {
        "latent_dimensions": int(latent.shape[1]),
        "state_dimensions": {key: int(value.shape[1]) for key, value in states.items()},
        "available_fields": FIELDS,
        "channel_rms_scales": dict(zip(FIELDS, channel_scales)),
        "U_PCA": u_meta,
        "Tiso_PCA": t_meta,
        "U_Tiso_PCA": ut_meta,
        "prior_state_metadata": old_meta,
    }
    return time_us, latent, states, groups, metadata


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["case"], row["state"], row["horizon_us"])].append(row)
    output = []
    for (case_name, state_name, horizon), local in grouped.items():
        analog_error = np.asarray([row["analog_error"] for row in local])
        persistence = np.asarray([row["persistence_error"] for row in local])
        output.append(
            {
                "case": case_name,
                "state": state_name,
                "horizon_us": horizon,
                "queries": len(local),
                "mean_ambiguity_normalized": float(
                    np.mean([row["ambiguity_normalized"] for row in local])
                ),
                "median_ambiguity_normalized": float(
                    np.median([row["ambiguity_normalized"] for row in local])
                ),
                "mean_analog_error_normalized": float(
                    np.mean([row["analog_error_normalized"] for row in local])
                ),
                "analog_skill_vs_zero_increment": float(
                    1.0 - np.mean(analog_error) / max(np.mean(persistence), 1.0e-30)
                ),
                "median_neighbor_radius_normalized": float(
                    np.median([row["neighbor_radius_normalized"] for row in local])
                ),
            }
        )
    return sorted(
        output,
        key=lambda row: (row["case"], STATE_ORDER.index(row["state"]), row["horizon_us"]),
    )


def residual_rank_correlation(
    predictor: np.ndarray,
    target: np.ndarray,
    controls: np.ndarray,
) -> float:
    y = rankdata(target)
    x = rankdata(predictor)
    design = np.column_stack(
        [np.ones(len(target))] + [rankdata(controls[:, column]) for column in range(controls.shape[1])]
    )
    y_residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    x_residual = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    return float(np.corrcoef(x_residual, y_residual)[0, 1])


def pair_associations(
    case_name: str,
    time_us: np.ndarray,
    groups: dict[str, np.ndarray],
    pairs: list[dict],
) -> tuple[list[dict], list[dict]]:
    local = [
        row
        for row in pairs
        if row["case"] == case_name
        and row["state"] == "L"
        and math.isclose(row["horizon_us"], FOCAL_HORIZON_US)
    ]
    time_to_index = {round(float(value), 8): index for index, value in enumerate(time_us)}
    detail = []
    for row in local:
        query = time_to_index[round(float(row["query_time_us"]), 8)]
        neighbor = time_to_index[round(float(row["neighbor_time_us"]), 8)]
        item = dict(row)
        item["time_separation_us"] = abs(float(row["query_time_us"]) - float(row["neighbor_time_us"]))
        for name, values in groups.items():
            difference = values[query] - values[neighbor]
            item[f"{name}_distance"] = float(np.sqrt(np.mean(difference * difference)))
        detail.append(item)
    current = np.asarray([row["current_distance_normalized"] for row in detail])
    future = np.asarray([row["future_latent_distance_normalized"] for row in detail])
    controls = np.column_stack(
        (
            current,
            [row["time_separation_us"] for row in detail],
            [row["query_time_us"] for row in detail],
            [row["neighbor_time_us"] for row in detail],
        )
    )
    summary = []
    close = current <= np.median(current)
    low_cut = np.quantile(future[close], 0.25)
    high_cut = np.quantile(future[close], 0.75)
    for name in groups:
        values = np.asarray([row[f"{name}_distance"] for row in detail])
        raw = float(spearmanr(values, future).statistic)
        partial = residual_rank_correlation(values, future, controls)
        coherent = values[close & (future <= low_cut)]
        divergent = values[close & (future >= high_cut)]
        summary.append(
            {
                "case": case_name,
                "group": name,
                "pairs": len(values),
                "raw_spearman_group_distance_vs_future_divergence": raw,
                "partial_rank_correlation_controlling_L_distance_and_times": partial,
                "close_pair_current_distance_cut": float(np.median(current)),
                "coherent_future_q25": float(low_cut),
                "divergent_future_q75": float(high_cut),
                "coherent_pairs": len(coherent),
                "divergent_pairs": len(divergent),
                "median_group_distance_coherent": float(np.median(coherent)),
                "median_group_distance_divergent": float(np.median(divergent)),
                "divergent_over_coherent_median_distance": float(
                    np.median(divergent) / max(np.median(coherent), 1.0e-30)
                ),
            }
        )
    return summary, detail


def component_ambiguity(
    case_name: str,
    time_us: np.ndarray,
    latent: np.ndarray,
) -> list[dict]:
    state = latent
    base = np.flatnonzero(
        (time_us >= ambiguity.COMMON_START_US - 1.0e-9)
        & (time_us <= ambiguity.COMMON_END_US + 1.0e-9)
    )[:: ambiguity.ANALOG_STEP_FRAMES]
    horizon = int(round(FOCAL_HORIZON_US / ambiguity.DT_US))
    valid = base[(base + horizon < len(time_us))]
    valid = valid[time_us[valid + horizon] <= ambiguity.COMMON_END_US + 1.0e-9]
    component_variance = np.zeros(latent.shape[1], dtype=np.float64)
    count = 0
    for query in valid:
        candidates = valid[
            np.abs(time_us[valid] - time_us[query]) >= ambiguity.PRIMARY_THEILER_US - 1.0e-9
        ]
        distance = np.sqrt(np.mean((state[candidates] - state[query]) ** 2, axis=1))
        neighbors = candidates[np.argsort(distance, kind="stable")[: ambiguity.PRIMARY_K]]
        increments = latent[neighbors + horizon] - latent[neighbors]
        component_variance += np.mean(
            (increments - np.mean(increments, axis=0, keepdims=True)) ** 2, axis=0
        )
        count += 1
    component_variance /= max(count, 1)
    total = max(float(np.sum(component_variance)), 1.0e-30)
    order = np.argsort(component_variance)[::-1]
    rank_by_component = np.empty_like(order)
    rank_by_component[order] = np.arange(1, len(order) + 1)
    return [
        {
            "case": case_name,
            "latent_pc": component + 1,
            "ambiguity_variance": float(component_variance[component]),
            "fraction_of_total_ambiguity": float(component_variance[component] / total),
            "ambiguity_rank": int(rank_by_component[component]),
        }
        for component in range(latent.shape[1])
    ]


def plot_horizons(summary: list[dict], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 5.3), constrained_layout=True, sharey=True)
    for axis, case_name in zip(axes, CASES):
        for state_name in STATE_ORDER:
            local = [
                row for row in summary if row["case"] == case_name and row["state"] == state_name
            ]
            axis.plot(
                [row["horizon_us"] for row in local],
                [row["mean_ambiguity_normalized"] for row in local],
                marker="o",
                linewidth=1.8,
                color=COLORS[state_name],
                label=state_name,
            )
        axis.set_title(case_name)
        axis.set_xlabel("future horizon [us]")
        axis.set_ylim(-0.05, 0.9)
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right", fontsize=7)
    axes[0].set_ylabel("normalized future-increment ambiguity")
    figure.suptitle("Available-moment ablation before pressure-tensor PIC")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_time_localization(query_rows: list[dict], path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(13.0, 8.2), constrained_layout=True, sharex=True)
    selected_states = ("L", "L+U+Tiso", "L+organisation+U+Tiso")
    for axis, case_name in zip(axes, CASES):
        for state_name in selected_states:
            local = [
                row
                for row in query_rows
                if row["case"] == case_name
                and row["state"] == state_name
                and math.isclose(row["horizon_us"], FOCAL_HORIZON_US)
            ]
            x = np.asarray([row["time_us"] for row in local])
            y = np.asarray([row["ambiguity_normalized"] for row in local])
            width = 7
            smooth = np.convolve(y, np.ones(width) / width, mode="same")
            axis.plot(x, smooth, linewidth=2.0, color=COLORS[state_name], label=state_name)
        axis.set_title(case_name)
        axis.set_ylabel("local ambiguity")
        axis.grid(alpha=0.25)
        lower, upper = axis.get_ylim()
        axis.set_ylim(min(-0.05, lower), upper)
        axis.legend(loc="lower right", fontsize=8)
    axes[-1].set_xlabel("current time [us]")
    figure.suptitle(f"Time-localized ambiguity at {FOCAL_HORIZON_US:g} us horizon")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_components(rows: list[dict], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), constrained_layout=True)
    for axis, case_name in zip(axes, CASES):
        local = sorted(
            (row for row in rows if row["case"] == case_name),
            key=lambda row: row["ambiguity_rank"],
        )
        axis.bar(
            np.arange(1, len(local) + 1),
            [row["fraction_of_total_ambiguity"] for row in local],
            color="#d55e00" if case_name == "B25" else "#0072b2",
        )
        axis.set_xticks(np.arange(1, len(local) + 1))
        axis.set_xticklabels([f"PC{row['latent_pc']}" for row in local], rotation=60)
        axis.set_title(case_name)
        axis.set_xlabel("latent PC, sorted by ambiguity contribution")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("fraction of total analogue ambiguity")
    figure.suptitle(f"Latent directions responsible for {FOCAL_HORIZON_US:g} us ambiguity")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_pair_association(rows: list[dict], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), constrained_layout=True, sharey=True)
    groups = ("U", "Tiso", "U+Tiso", "organisation")
    for axis, case_name in zip(axes, CASES):
        local = {row["group"]: row for row in rows if row["case"] == case_name}
        values = [local[group]["partial_rank_correlation_controlling_L_distance_and_times"] for group in groups]
        axis.bar(np.arange(len(groups)), values, color=("#0072b2", "#e69f00", "#009e73", "#cc79a7"))
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(np.arange(len(groups)))
        axis.set_xticklabels(groups, rotation=25)
        axis.set_title(case_name)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("partial rank correlation with future divergence")
    figure.suptitle("Do omitted current variables explain divergence among latent analogues?")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_query: list[dict] = []
    all_pairs: list[dict] = []
    all_fixed_query: list[dict] = []
    pair_summary: list[dict] = []
    pair_detail: list[dict] = []
    component_rows: list[dict] = []
    metadata = {}
    for case_name in CASES:
        print(f"[CASE] {case_name}", flush=True)
        time_us, latent, states, groups, case_meta = build_case(case_name, OUTPUT)
        metadata[case_name] = case_meta
        for state_name, state in states.items():
            query, pairs = ambiguity.evaluate_ambiguity(
                case_name,
                time_us,
                latent,
                state_name,
                state,
                ambiguity.COMMON_START_US,
                ambiguity.COMMON_END_US,
                HORIZONS_US,
                (ambiguity.PRIMARY_THEILER_US,),
                (ambiguity.PRIMARY_K,),
                state_name == "L",
            )
            all_query.extend(query)
            all_pairs.extend(pairs)
        fixed, fixed_meta = ambiguity.fixed_dimension_states(
            time_us, states, latent.shape[1]
        )
        metadata[case_name]["fixed_dimension"] = fixed_meta
        for state_name, state in fixed.items():
            query, _ = ambiguity.evaluate_ambiguity(
                case_name,
                time_us,
                latent,
                state_name,
                state,
                ambiguity.COMMON_START_US,
                ambiguity.COMMON_END_US,
                HORIZONS_US,
                (ambiguity.PRIMARY_THEILER_US,),
                (ambiguity.PRIMARY_K,),
                False,
            )
            all_fixed_query.extend(query)
        local_summary, local_detail = pair_associations(
            case_name, time_us, groups, all_pairs
        )
        pair_summary.extend(local_summary)
        pair_detail.extend(local_detail)
        component_rows.extend(component_ambiguity(case_name, time_us, latent))

    summary = summarize(all_query)
    fixed_summary = summarize(all_fixed_query)
    write_csv(OUTPUT / "moment_ablation_summary.csv", summary)
    write_csv(OUTPUT / "moment_ablation_query_level.csv", all_query)
    write_csv(OUTPUT / "moment_ablation_fixed_dimension_summary.csv", fixed_summary)
    write_csv(OUTPUT / "moment_pair_association_summary.csv", pair_summary)
    write_csv(OUTPUT / "moment_pair_association_detail.csv", pair_detail)
    write_csv(OUTPUT / "latent_pc_ambiguity.csv", component_rows)
    plot_horizons(summary, OUTPUT / "moment_ablation_ambiguity_by_horizon.png")
    plot_time_localization(all_query, OUTPUT / "ambiguity_localization_time.png")
    plot_components(component_rows, OUTPUT / "latent_pc_ambiguity_contributions.png")
    plot_pair_association(pair_summary, OUTPUT / "omitted_moment_pair_association.png")

    focal = {}
    for case_name in CASES:
        baseline = next(
            row
            for row in summary
            if row["case"] == case_name
            and row["state"] == "L"
            and math.isclose(row["horizon_us"], FOCAL_HORIZON_US)
        )
        focal[case_name] = {}
        for state_name in STATE_ORDER:
            row = next(
                row
                for row in summary
                if row["case"] == case_name
                and row["state"] == state_name
                and math.isclose(row["horizon_us"], FOCAL_HORIZON_US)
            )
            focal[case_name][state_name] = {
                "ambiguity": row["mean_ambiguity_normalized"],
                "analog_error": row["mean_analog_error_normalized"],
                "skill": row["analog_skill_vs_zero_increment"],
                "ambiguity_reduction_vs_L": 1.0
                - row["mean_ambiguity_normalized"]
                / max(baseline["mean_ambiguity_normalized"], 1.0e-30),
            }
    report = {
        "status": "PASS",
        "question": "Do already-saved electron drift velocities and scalar temperature explain B25 predictive ambiguity?",
        "protocol": {
            "interval_us": [ambiguity.COMMON_START_US, ambiguity.COMMON_END_US],
            "reference_end_us": ambiguity.REPRESENTATION_END_US,
            "horizons_us": HORIZONS_US,
            "theiler_us": ambiguity.PRIMARY_THEILER_US,
            "neighbors": ambiguity.PRIMARY_K,
            "radial_bands": RADIAL_BANDS,
            "max_azimuthal_mode": MAX_MODE,
        },
        "metadata": metadata,
        "focal_1p2us": focal,
        "pair_associations": pair_summary,
        "guardrail": (
            "A reduction would support added observability from saved scalar moments. "
            "A null result does not prove that pressure anisotropy is irrelevant because "
            "the saved temperature is only the trace of the central second moment."
        ),
    }
    (OUTPUT / "analysis_summary.json").write_text(
        json.dumps(json_safe(report), indent=2), encoding="utf-8"
    )
    readme = [
        "# B15/B25 pre-rePIC moment diagnostic",
        "",
        "This analysis uses only existing stitched PIC outputs. No PIC or SimVP model was rerun.",
        "",
        "Available electron fields: `ud`, `vd`, `wd`, and scalar `Temp`.",
        "The scalar temperature is not a pressure tensor and contains no anisotropy or off-diagonal stress.",
        "",
        "Primary diagnostic: Theiler=1.0 us, k=10 analogues, 1.2 us future horizon.",
        "See `analysis_summary.json` and the CSV files for numerical results.",
        "",
        "The current PEPAPIC JSON `moment_tensor` item is not implemented by the exporter; a future tensor run requires code changes.",
    ]
    (OUTPUT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"[DONE] {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
