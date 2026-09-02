#!/usr/bin/env python3
"""Diagnose predictive-state ambiguity in the B15 and B25 reduced states.

The analysis is nonparametric. For each current reduced state, temporally
separated nearest analogues are found and their future latent increments are
compared. A large conditional spread means that the tested coordinates do not
identify a unique future at the available resolution. It is evidence against
predictive closure of that coarse state, not proof that the full PIC dynamics
are intrinsically stochastic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze_radaz_b25_state_augmentation_ablation as ablation
import analyze_radaz_b25_temporal_switching_rom as switching


ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT.parent
OUTPUT = (
    RESEARCH
    / "research_results"
    / "2D_RadAz"
    / "SimVPv2"
    / "workdirs"
    / "analyze_radaz_b15_b25_predictive_state_ambiguity"
)
ROLLING_DIR = (
    RESEARCH
    / "research_results"
    / "2D_RadAz"
    / "SimVPv2"
    / "workdirs"
    / "analyze_radaz_b25_rolling_window_rom_0to40us"
)
ABLATION_DIR = (
    RESEARCH
    / "research_results"
    / "2D_RadAz"
    / "SimVPv2"
    / "workdirs"
    / "analyze_radaz_b25_state_augmentation_ablation_0to40us"
)

DT_US = 0.015
REPRESENTATION_START_US = 12.0
REPRESENTATION_END_US = 24.0
COMMON_START_US = 13.5
COMMON_END_US = 29.75
ANALOG_STEP_FRAMES = 10
HORIZONS_US = (0.15, 0.30, 0.60, 1.20, 2.40)
THEILER_US = (0.50, 1.00, 2.60)
K_VALUES = (5, 10, 20)
PRIMARY_THEILER_US = 1.0
PRIMARY_K = 10
PRIMARY_HORIZON_US = 1.20
HISTORY_LAGS = (10, 20, 40, 80)
HISTORY_COMPONENTS = 9
MODE_BANDS = switching.MODE_BANDS
STATE_ORDER = ("L", "L+A+P", "L+history", "L+organisation")
CASE_COLORS = {"B15": "#0072b2", "B25": "#d55e00"}
STATE_COLORS = {
    "L": "#4d4d4d",
    "L+A+P": "#0072b2",
    "L+history": "#009e73",
    "L+organisation": "#cc79a7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--bico-width-us", type=float, default=1.50)
    parser.add_argument("--bico-step-us", type=float, default=0.15)
    parser.add_argument("--bico-max-mode", type=int, default=30)
    return parser.parse_args()


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


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


def standardize(values: np.ndarray, fit: np.ndarray) -> np.ndarray:
    mean = np.nanmean(values[fit], axis=0)
    scale = np.nanstd(values[fit], axis=0)
    scale = np.where(scale > 1.0e-10, scale, 1.0)
    return (values - mean) / scale


def interpolate(source_time: np.ndarray, values: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.interp(target_time, source_time, values[:, column]) for column in range(values.shape[1])],
        axis=1,
    )


def latest_sample(source_time: np.ndarray, values: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(source_time, target_time, side="right") - 1
    indices = np.clip(indices, 0, len(source_time) - 1)
    return values[indices]


def physical_groups(case_name: str, target_time: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    time_us, channels, coefficient = switching.unpack_fourier(switching.fourier_path(case_name))
    electron = coefficient[:, channels.index("electron_den")]
    phi = coefficient[:, channels.index("phi")]
    mode = np.arange(phi.shape[-1], dtype=np.float64)
    ey = -1j * mode[None, None, :] * phi
    amplitudes = []
    phases = []
    transports = []
    field_tesla = 0.015 if case_name == "B15" else 0.025
    for lower, upper in MODE_BANDS.values():
        upper = min(upper, phi.shape[-1] - 1)
        selected = slice(lower, upper + 1)
        amplitudes.append(np.sqrt(np.mean(np.abs(ey[..., selected]) ** 2, axis=(1, 2))))
        cross = np.sum(electron[..., selected] * np.conj(ey[..., selected]), axis=(1, 2))
        phases.append(np.angle(cross))
        transports.append(-2.0 * np.real(cross) / field_tesla)
    amplitude = np.log(np.maximum(np.stack(amplitudes, axis=1), 1.0e-30))
    phase = np.stack(phases, axis=1)
    ap = np.concatenate((amplitude, np.cos(phase), np.sin(phase)), axis=1)
    transport = np.stack(transports, axis=1)
    return (
        {
            "AP": interpolate(time_us, ap, target_time),
            "T_raw": interpolate(time_us, transport, target_time),
        },
        {
            "fourier_time_us": time_us,
            "channels": channels,
            "coefficient": coefficient,
        },
    )


def bicoherence_summaries(
    fourier: dict,
    end_us: float,
    width_us: float,
    step_us: float,
    max_mode: int,
) -> tuple[np.ndarray, np.ndarray]:
    time_us = fourier["fourier_time_us"]
    channels = fourier["channels"]
    coefficient = fourier["coefficient"]
    max_mode = min(max_mode, coefficient.shape[-1] - 1)
    triads = ablation.triad_indices(max_mode)
    selected_channels = (channels.index("phi"), channels.index("electron_den"))
    end_times = np.arange(REPRESENTATION_START_US, end_us + 1.0e-9, step_us)
    summaries = []
    for end in end_times:
        local_time = (time_us > end - width_us - 1.0e-9) & (time_us <= end + 1.0e-9)
        channel_summary = []
        for channel in selected_channels:
            local = coefficient[local_time, channel, :, : max_mode + 1].astype(np.complex128)
            local -= np.mean(local, axis=0, keepdims=True)
            values = ablation.spatial_bicoherence(local, triads)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            entropy = ablation.normalized_entropy(values)
            probability = values / max(float(np.sum(values)), 1.0e-30)
            channel_summary.extend(
                (
                    entropy,
                    math.exp(entropy * math.log(max(len(values), 2))),
                    float(np.sum(np.sort(probability)[-10:])),
                    float(np.max(values)),
                )
            )
        summaries.append(channel_summary)
    return end_times, np.asarray(summaries, dtype=np.float64)


def history_coordinates(latent: np.ndarray, fit: np.ndarray) -> tuple[np.ndarray, dict]:
    delayed = []
    for lag in HISTORY_LAGS:
        block = np.full_like(latent, np.nan)
        block[lag:] = latent[:-lag]
        delayed.append(block)
    raw = np.concatenate(delayed, axis=1)
    valid_fit = fit & np.all(np.isfinite(raw), axis=1)
    count = min(HISTORY_COMPONENTS, int(np.count_nonzero(valid_fit)) - 1, raw.shape[1])
    pca = PCA(n_components=count, svd_solver="randomized", random_state=42)
    pca.fit(raw[valid_fit])
    transformed = np.full((len(raw), count), np.nan, dtype=np.float64)
    valid = np.all(np.isfinite(raw), axis=1)
    transformed[valid] = pca.transform(raw[valid])
    transformed = standardize(transformed, valid_fit)
    return transformed, {
        "lags_frames": HISTORY_LAGS,
        "components": count,
        "variance_capture": float(np.sum(pca.explained_variance_ratio_)),
    }


def build_states(
    case_name: str,
    bico_width_us: float,
    bico_step_us: float,
    bico_max_mode: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict]:
    case = switching.load_case(case_name)
    representation = switching.fit_representation(
        case, REPRESENTATION_START_US, REPRESENTATION_END_US
    )
    time_us = representation.time_us
    latent = representation.latent
    fit = (time_us >= COMMON_START_US - 1.0e-9) & (time_us < REPRESENTATION_END_US - 1.0e-9)
    groups, fourier = physical_groups(case_name, time_us)
    ap = standardize(groups["AP"], fit)
    transport_scale = np.nanmedian(np.abs(groups["T_raw"][fit]), axis=0)
    transport_scale = np.where(transport_scale > 1.0e-30, transport_scale, 1.0)
    transport = np.arcsinh(groups["T_raw"] / transport_scale)
    transport = standardize(transport, fit)
    network_time, network = bicoherence_summaries(
        fourier,
        float(time_us[-1]),
        bico_width_us,
        bico_step_us,
        bico_max_mode,
    )
    network = latest_sample(network_time, network, time_us)
    network = standardize(network, fit)
    history, history_meta = history_coordinates(latent, fit)
    states = {
        "L": latent,
        "L+A+P": np.concatenate((latent, ap), axis=1),
        "L+history": np.concatenate((latent, history), axis=1),
        "L+organisation": np.concatenate((latent, ap, transport, network), axis=1),
    }
    metadata = {
        "latent_dimensions": int(latent.shape[1]),
        "state_dimensions": {name: int(values.shape[1]) for name, values in states.items()},
        "history": history_meta,
        "organisation": {
            "AP_dimensions": int(ap.shape[1]),
            "transport_dimensions": int(transport.shape[1]),
            "bicoherence_summary_dimensions": int(network.shape[1]),
            "bicoherence_width_us": bico_width_us,
            "bicoherence_step_us": bico_step_us,
        },
    }
    return time_us, latent, states, metadata


def fixed_dimension_states(
    time_us: np.ndarray,
    states: dict[str, np.ndarray],
    target_dimensions: int,
) -> tuple[dict[str, np.ndarray], dict]:
    fit_time = (time_us >= COMMON_START_US - 1.0e-9) & (
        time_us < REPRESENTATION_END_US - 1.0e-9
    )
    output = {}
    metadata = {}
    for state_name, values in states.items():
        if values.shape[1] <= target_dimensions:
            output[state_name] = values.copy()
            metadata[state_name] = {
                "input_dimensions": int(values.shape[1]),
                "output_dimensions": int(values.shape[1]),
                "variance_capture": 1.0,
            }
            continue
        valid = np.all(np.isfinite(values), axis=1)
        fit = fit_time & valid
        pca = PCA(
            n_components=target_dimensions,
            svd_solver="randomized",
            random_state=42,
        )
        pca.fit(values[fit])
        transformed = np.full((len(values), target_dimensions), np.nan)
        transformed[valid] = pca.transform(values[valid])
        transformed = standardize(transformed, fit)
        output[state_name] = transformed
        metadata[state_name] = {
            "input_dimensions": int(values.shape[1]),
            "output_dimensions": target_dimensions,
            "variance_capture": float(np.sum(pca.explained_variance_ratio_)),
        }
    return output, metadata


def pairwise_distance_scale(values: np.ndarray) -> float:
    difference = values[:, None, :] - values[None, :, :]
    distance = np.sqrt(np.mean(difference * difference, axis=2))
    upper = distance[np.triu_indices(len(values), k=1)]
    finite = upper[np.isfinite(upper) & (upper > 0.0)]
    return float(np.median(finite)) if len(finite) else 1.0


def evaluate_ambiguity(
    case_name: str,
    time_us: np.ndarray,
    latent: np.ndarray,
    state_name: str,
    state: np.ndarray,
    start_us: float,
    end_us: float,
    horizons_us: tuple[float, ...],
    theiler_values: tuple[float, ...],
    k_values: tuple[int, ...],
    save_primary_pairs: bool,
    current_end_us: float | None = None,
) -> tuple[list[dict], list[dict]]:
    current_end = end_us if current_end_us is None else current_end_us
    base = np.flatnonzero(
        (time_us >= start_us - 1.0e-9)
        & (time_us <= current_end + 1.0e-9)
        & np.all(np.isfinite(state), axis=1)
    )
    base = base[::ANALOG_STEP_FRAMES]
    state_scale = pairwise_distance_scale(state[base])
    rows: list[dict] = []
    pair_rows: list[dict] = []
    for horizon_us in horizons_us:
        horizon = int(round(horizon_us / DT_US))
        valid = base[base + horizon < len(time_us)]
        valid = valid[time_us[valid + horizon] <= end_us + 1.0e-9]
        increments = latent[valid + horizon] - latent[valid]
        centered = increments - np.mean(increments, axis=0, keepdims=True)
        increment_variance = float(np.mean(centered * centered))
        increment_variance = max(increment_variance, 1.0e-30)
        future_scale = pairwise_distance_scale(latent[valid + horizon])
        for theiler_us in theiler_values:
            for k in k_values:
                for query_position, query in enumerate(valid):
                    candidates = valid[np.abs(time_us[valid] - time_us[query]) >= theiler_us - 1.0e-9]
                    if len(candidates) < k:
                        continue
                    difference = state[candidates] - state[query]
                    distances = np.sqrt(np.mean(difference * difference, axis=1))
                    order = np.argsort(distances, kind="stable")[:k]
                    neighbors = candidates[order]
                    neighbor_distances = distances[order]
                    neighbor_increments = latent[neighbors + horizon] - latent[neighbors]
                    mean_increment = np.mean(neighbor_increments, axis=0)
                    ambiguity = float(np.mean((neighbor_increments - mean_increment) ** 2))
                    analog_error = float(np.mean((increments[query_position] - mean_increment) ** 2))
                    persistence_error = float(np.mean(increments[query_position] ** 2))
                    row = {
                        "case": case_name,
                        "state": state_name,
                        "time_us": float(time_us[query]),
                        "horizon_us": horizon_us,
                        "theiler_us": theiler_us,
                        "k": k,
                        "neighbor_radius": float(neighbor_distances[-1]),
                        "neighbor_radius_normalized": float(neighbor_distances[-1] / state_scale),
                        "ambiguity": ambiguity,
                        "ambiguity_normalized": ambiguity / increment_variance,
                        "analog_error": analog_error,
                        "analog_error_normalized": analog_error / increment_variance,
                        "persistence_error": persistence_error,
                        "persistence_error_normalized": persistence_error / increment_variance,
                    }
                    rows.append(row)
                    if (
                        save_primary_pairs
                        and math.isclose(theiler_us, PRIMARY_THEILER_US)
                        and k == PRIMARY_K
                    ):
                        for neighbor, current_distance in zip(neighbors, neighbor_distances):
                            future_distance = float(
                                np.sqrt(np.mean((latent[neighbor + horizon] - latent[query + horizon]) ** 2))
                            )
                            pair_rows.append(
                                {
                                    "case": case_name,
                                    "state": state_name,
                                    "query_time_us": float(time_us[query]),
                                    "neighbor_time_us": float(time_us[neighbor]),
                                    "horizon_us": horizon_us,
                                    "current_distance_normalized": float(current_distance / state_scale),
                                    "future_latent_distance_normalized": float(future_distance / future_scale),
                                }
                            )
    return rows, pair_rows


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["case"], row["state"], row["horizon_us"], row["theiler_us"], row["k"])
        groups[key].append(row)
    summary = []
    for key, local in groups.items():
        ambiguity = np.asarray([row["ambiguity_normalized"] for row in local])
        error = np.asarray([row["analog_error"] for row in local])
        persistence = np.asarray([row["persistence_error"] for row in local])
        summary.append(
            {
                "case": key[0],
                "state": key[1],
                "horizon_us": key[2],
                "theiler_us": key[3],
                "k": key[4],
                "queries": len(local),
                "median_neighbor_radius_normalized": float(
                    np.median([row["neighbor_radius_normalized"] for row in local])
                ),
                "mean_ambiguity_normalized": float(np.mean(ambiguity)),
                "median_ambiguity_normalized": float(np.median(ambiguity)),
                "q25_ambiguity_normalized": float(np.quantile(ambiguity, 0.25)),
                "q75_ambiguity_normalized": float(np.quantile(ambiguity, 0.75)),
                "mean_analog_error_normalized": float(
                    np.mean([row["analog_error_normalized"] for row in local])
                ),
                "median_analog_error_normalized": float(
                    np.median([row["analog_error_normalized"] for row in local])
                ),
                "analog_skill_vs_zero_increment": float(1.0 - np.mean(error) / max(np.mean(persistence), 1.0e-30)),
            }
        )
    return sorted(summary, key=lambda row: (row["case"], STATE_ORDER.index(row["state"]), row["horizon_us"], row["theiler_us"], row["k"]))


def distance_matched_summary(pair_rows: list[dict]) -> list[dict]:
    primary = [row for row in pair_rows if math.isclose(row["horizon_us"], PRIMARY_HORIZON_US)]
    values = np.asarray([row["current_distance_normalized"] for row in primary])
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, 7)))
    output = []
    if len(edges) < 3:
        return output
    for case_name in ("B15", "B25"):
        for state_name in STATE_ORDER:
            local = [row for row in primary if row["case"] == case_name and row["state"] == state_name]
            for index in range(len(edges) - 1):
                selected = [
                    row
                    for row in local
                    if edges[index] <= row["current_distance_normalized"]
                    and (
                        row["current_distance_normalized"] < edges[index + 1]
                        or (index == len(edges) - 2 and row["current_distance_normalized"] <= edges[index + 1])
                    )
                ]
                if len(selected) < 10:
                    continue
                output.append(
                    {
                        "case": case_name,
                        "state": state_name,
                        "horizon_us": PRIMARY_HORIZON_US,
                        "distance_bin": index + 1,
                        "distance_lower": float(edges[index]),
                        "distance_upper": float(edges[index + 1]),
                        "pairs": len(selected),
                        "median_current_distance_normalized": float(
                            np.median([row["current_distance_normalized"] for row in selected])
                        ),
                        "median_future_latent_distance_normalized": float(
                            np.median([row["future_latent_distance_normalized"] for row in selected])
                        ),
                    }
                )
    return output


def primary_row(summary: list[dict], case_name: str, state_name: str, horizon_us: float) -> dict:
    return next(
        row
        for row in summary
        if row["case"] == case_name
        and row["state"] == state_name
        and math.isclose(row["horizon_us"], horizon_us)
        and math.isclose(row["theiler_us"], PRIMARY_THEILER_US)
        and row["k"] == PRIMARY_K
    )


def rolling_link(
    time_us: np.ndarray,
    latent: np.ndarray,
    states: dict[str, np.ndarray],
) -> list[dict]:
    metrics = read_csv(ABLATION_DIR / "ablation_metrics.csv")
    output = []
    for fit_start, fit_end, test_end in ablation.WINDOWS:
        window = f"{fit_start:g}-{fit_end:g}_to_{test_end:g}us"
        for state_name, metric_state in (("L", "L"), ("L+A+P", "L+A+P")):
            query, _ = evaluate_ambiguity(
                "B25",
                time_us,
                latent,
                state_name,
                states[state_name],
                fit_start,
                fit_end + 0.60,
                (0.60,),
                (PRIMARY_THEILER_US,),
                (PRIMARY_K,),
                False,
                current_end_us=fit_end,
            )
            local = query
            metric = next(row for row in metrics if row["window"] == window and row["state"] == metric_state)
            output.append(
                {
                    "window": window,
                    "fit_start_us": fit_start,
                    "fit_end_us": fit_end,
                    "test_end_us": test_end,
                    "state": state_name,
                    "ambiguity_horizon_us": 0.60,
                    "mean_ambiguity_normalized": float(np.mean([row["ambiguity_normalized"] for row in local])),
                    "mean_analog_error_normalized": float(np.mean([row["analog_error_normalized"] for row in local])),
                    "rom_latent_skill_vs_persistence": float(metric["latent_skill_vs_persistence"]),
                    "rom_latent_correlation": float(metric["latent_correlation"]),
                }
            )
    for state_name in ("L", "L+A+P"):
        local = [row for row in output if row["state"] == state_name]
        ambiguity = np.asarray([row["mean_ambiguity_normalized"] for row in local])
        skill = np.asarray([row["rom_latent_skill_vs_persistence"] for row in local])
        if np.std(ambiguity) > 0.0 and np.std(skill) > 0.0:
            correlation = float(np.corrcoef(ambiguity, skill)[0, 1])
        else:
            correlation = float("nan")
        ambiguity_rank = np.argsort(np.argsort(ambiguity, kind="stable"), kind="stable")
        skill_rank = np.argsort(np.argsort(skill, kind="stable"), kind="stable")
        if np.std(ambiguity_rank) > 0.0 and np.std(skill_rank) > 0.0:
            spearman = float(np.corrcoef(ambiguity_rank, skill_rank)[0, 1])
        else:
            spearman = float("nan")
        for row in local:
            row["ambiguity_skill_pearson_across_six_overlapping_windows"] = correlation
            row["ambiguity_skill_spearman_across_six_overlapping_windows"] = spearman
    return output


def plot_horizon(summary: list[dict], key: str, ylabel: str, path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.6), constrained_layout=True, sharex=True)
    for axis, state_name in zip(axes.ravel(), STATE_ORDER):
        for case_name in ("B15", "B25"):
            local = [
                row
                for row in summary
                if row["case"] == case_name
                and row["state"] == state_name
                and math.isclose(row["theiler_us"], PRIMARY_THEILER_US)
                and row["k"] == PRIMARY_K
            ]
            axis.plot(
                [row["horizon_us"] for row in local],
                [row[key] for row in local],
                marker="o",
                linewidth=2.0,
                color=CASE_COLORS[case_name],
                label=case_name,
            )
        axis.set_title(state_name)
        axis.set_xlabel("future horizon [us]")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(loc="upper left")
    figure.suptitle(f"Predictive-state analogue diagnostic (Theiler={PRIMARY_THEILER_US:g} us, k={PRIMARY_K})", fontsize=14)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_distance_matched(rows: list[dict], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True, sharey=True)
    for axis, case_name in zip(axes, ("B15", "B25")):
        for state_name in STATE_ORDER:
            local = [row for row in rows if row["case"] == case_name and row["state"] == state_name]
            if not local:
                continue
            axis.plot(
                [row["median_current_distance_normalized"] for row in local],
                [row["median_future_latent_distance_normalized"] for row in local],
                marker="o",
                linewidth=1.8,
                color=STATE_COLORS[state_name],
                label=state_name,
            )
        axis.set_title(case_name)
        axis.set_xlabel("current state distance / random-pair median")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("future latent distance / random-pair median")
    figure.suptitle(f"Distance-matched analogue divergence at {PRIMARY_HORIZON_US:g} us", fontsize=14)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_rolling(rows: list[dict], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), constrained_layout=True)
    for axis, state_name in zip(axes, ("L", "L+A+P")):
        local = [row for row in rows if row["state"] == state_name]
        axis.scatter(
            [row["mean_ambiguity_normalized"] for row in local],
            [row["rom_latent_skill_vs_persistence"] for row in local],
            color=STATE_COLORS[state_name],
            s=55,
        )
        for row in local:
            axis.annotate(row["window"].split("_to_")[0], (row["mean_ambiguity_normalized"], row["rom_latent_skill_vs_persistence"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        correlation = local[0]["ambiguity_skill_pearson_across_six_overlapping_windows"]
        spearman = local[0]["ambiguity_skill_spearman_across_six_overlapping_windows"]
        axis.set_title(f"{state_name}: Pearson={correlation:.3f}, Spearman={spearman:.3f}")
        axis.set_xlabel("fit-window ambiguity at 0.6 us")
        axis.set_ylabel("6 us ROM skill vs persistence")
        axis.set_yscale("symlog", linthresh=1.0)
        axis.grid(alpha=0.25)
    figure.suptitle("B25 local ambiguity versus rolling autonomous ROM skill", fontsize=14)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def build_interpretation(
    summary: list[dict], fixed_summary: list[dict], rolling: list[dict]
) -> dict:
    horizons = {}
    for horizon in HORIZONS_US:
        b15 = primary_row(summary, "B15", "L", horizon)
        b25 = primary_row(summary, "B25", "L", horizon)
        horizons[str(horizon)] = {
            "B15_L_ambiguity": b15["mean_ambiguity_normalized"],
            "B25_L_ambiguity": b25["mean_ambiguity_normalized"],
            "B25_over_B15": b25["mean_ambiguity_normalized"] / max(b15["mean_ambiguity_normalized"], 1.0e-30),
        }
    focal = {}
    baseline = primary_row(summary, "B25", "L", PRIMARY_HORIZON_US)
    for state_name in STATE_ORDER:
        row = primary_row(summary, "B25", state_name, PRIMARY_HORIZON_US)
        focal[state_name] = {
            "mean_ambiguity_normalized": row["mean_ambiguity_normalized"],
            "mean_analog_error_normalized": row["mean_analog_error_normalized"],
            "analog_skill_vs_zero_increment": row["analog_skill_vs_zero_increment"],
            "ambiguity_reduction_vs_L": 1.0 - row["mean_ambiguity_normalized"] / max(baseline["mean_ambiguity_normalized"], 1.0e-30),
        }
    correlations = {}
    for state_name in ("L", "L+A+P"):
        row = next(row for row in rolling if row["state"] == state_name)
        correlations[state_name] = {
            "pearson": row["ambiguity_skill_pearson_across_six_overlapping_windows"],
            "spearman": row["ambiguity_skill_spearman_across_six_overlapping_windows"],
        }
    fixed_baseline = primary_row(
        fixed_summary, "B25", "L", PRIMARY_HORIZON_US
    )
    fixed_dimension = {}
    for state_name in STATE_ORDER:
        row = primary_row(fixed_summary, "B25", state_name, PRIMARY_HORIZON_US)
        fixed_dimension[state_name] = {
            "mean_ambiguity_normalized": row["mean_ambiguity_normalized"],
            "mean_analog_error_normalized": row["mean_analog_error_normalized"],
            "ambiguity_reduction_vs_L": 1.0
            - row["mean_ambiguity_normalized"]
            / max(fixed_baseline["mean_ambiguity_normalized"], 1.0e-30),
        }
    return {
        "primary_configuration": {
            "theiler_us": PRIMARY_THEILER_US,
            "k": PRIMARY_K,
            "focal_horizon_us": PRIMARY_HORIZON_US,
        },
        "L_case_comparison_by_horizon": horizons,
        "B25_state_ablation_at_1p2us": focal,
        "B25_fixed_dimension_sensitivity_at_1p2us": fixed_dimension,
        "rolling_ambiguity_skill_correlation": correlations,
        "guardrail": (
            "High ambiguity rejects predictive sufficiency only for the tested coarse coordinates, metric, "
            "sampling density, and finite trajectory. It does not prove intrinsic stochasticity or identify "
            "a unique hidden kinetic variable."
        ),
    }


def write_readme(
    path: Path,
    metadata: dict,
    interpretation: dict,
    summary: list[dict],
    fixed_summary: list[dict],
) -> None:
    lines = [
        "# B15/B25 predictive-state ambiguity",
        "",
        "This analysis asks whether nearby reduced states have reproducibly similar futures.",
        "It uses temporally separated nearest analogues and compares future increments of",
        "the frozen SimVP encoder latent state.",
        "",
        "## Protocol",
        "",
        f"- Common interval: {COMMON_START_US:g}-{COMMON_END_US:g} us.",
        f"- Analogue grid: every {ANALOG_STEP_FRAMES} frames ({ANALOG_STEP_FRAMES * DT_US:g} us).",
        f"- Horizons: {', '.join(f'{value:g}' for value in HORIZONS_US)} us.",
        f"- Theiler exclusion widths: {', '.join(f'{value:g}' for value in THEILER_US)} us.",
        f"- Neighbor counts: {', '.join(str(value) for value in K_VALUES)}.",
        "- PCA, scaling, history compression, and organization scaling use only the",
        f"  {COMMON_START_US:g}-{REPRESENTATION_END_US:g} us reference interval.",
        "- The structural analogue diagnostic may use nonlocal states occurring before",
        "  or after the query. It is not presented as a deployable causal forecast.",
        "",
        "## State definitions",
        "",
        "- `L`: frozen SimVP encoder PCA coordinates.",
        "- `L+A+P`: L plus long/MTSI/ECDI amplitudes and density-Ey cross-phase.",
        "- `L+history`: L plus nine PCA coordinates from causal L delays at",
        "  0.15, 0.30, 0.60, and 1.20 us.",
        "- `L+organisation`: L+A+P plus modal transport and causal trailing-window",
        "  bicoherence concentration summaries.",
        "- A separate sensitivity analysis compresses every augmented state to the",
        "  same dimension as L using reference-interval PCA. This checks whether raw",
        "  augmentation failed only because nearest neighbors degrade in higher dimension.",
        "",
        "## Primary result at 1.2 us",
        "",
        "| case | state | ambiguity/global increment variance | analogue error | skill vs zero increment |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for case_name in ("B15", "B25"):
        for state_name in STATE_ORDER:
            row = primary_row(summary, case_name, state_name, PRIMARY_HORIZON_US)
            lines.append(
                f"| {case_name} | {state_name} | {row['mean_ambiguity_normalized']:.3f} | "
                f"{row['mean_analog_error_normalized']:.3f} | {row['analog_skill_vs_zero_increment']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Fixed-dimension sensitivity at 1.2 us",
            "",
            "| case | state | ambiguity/global increment variance | analogue error |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for case_name in ("B15", "B25"):
        for state_name in STATE_ORDER:
            row = primary_row(fixed_summary, case_name, state_name, PRIMARY_HORIZON_US)
            lines.append(
                f"| {case_name} | {state_name} | {row['mean_ambiguity_normalized']:.3f} | "
                f"{row['mean_analog_error_normalized']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- A small neighbor spread is insufficient by itself: the neighbor-mean analogue",
            "  error is reported to detect a compact but biased future distribution.",
            "- Theiler exclusion suppresses trivial temporal neighbors and is varied through",
            "  2.6 us, slightly longer than the previously measured B15 period near 2.565 us.",
            "- Current distances are divided by each state's random-pair median before the",
            "  distance-matched comparison. This reduces, but cannot eliminate, metric and",
            "  dimensionality effects.",
            "- B15 and B25 are each one PIC trajectory. Queries and rolling windows overlap,",
            "  so the reported correlations are descriptive and not independent-sample tests.",
            "- High ambiguity means the tested coarse state is not predictively sufficient at",
            "  the available resolution. It does not prove that the full PIC state is stochastic;",
            "  finite data, PIC noise, an unsuitable metric, nonstationarity, or omitted state",
            "  variables can all produce the same symptom.",
            "",
            "## Primary methodological sources",
            "",
            "- Lorenz (1969), naturally occurring analogues:",
            "  https://doi.org/10.1175/1520-0469(1969)26%3C636:APARBN%3E2.0.CO;2",
            "- Theiler (1986), temporal-neighbor exclusion for autocorrelated time series:",
            "  https://doi.org/10.1103/PhysRevA.34.2427",
            "- Shalizi and Crutchfield (2001), predictive sufficiency of causal states:",
            "  https://arxiv.org/abs/cond-mat/9907176",
            "- Zhao and Giannakis (2016), ensemble analog forecasting and delay coordinates:",
            "  https://arxiv.org/abs/1412.3831",
            "",
            "## Outputs",
            "",
            "- `query_level_ambiguity.csv`: every query-level diagnostic.",
            "- `ambiguity_summary.csv`: case/state/horizon/Theiler/k aggregation.",
            "- `fixed_dimension_ambiguity_summary.csv`: equal-dimensional metric control.",
            "- `primary_analogue_pairs.csv`: distance-matched pair diagnostics.",
            "- `distance_matched_summary.csv`: pooled current-distance bins.",
            "- `rolling_ambiguity_vs_rom_skill.csv`: six overlapping B25 windows.",
            "- `analysis_summary.json`: protocol, metadata, and primary interpretation values.",
            "- Four PNG summaries.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    case_data = {}
    all_query_rows: list[dict] = []
    all_pair_rows: list[dict] = []
    all_fixed_query_rows: list[dict] = []
    metadata = {}
    for case_name in ("B15", "B25"):
        print(f"[PREP] {case_name} states", flush=True)
        time_us, latent, states, local_metadata = build_states(
            case_name,
            args.bico_width_us,
            args.bico_step_us,
            args.bico_max_mode,
        )
        case_data[case_name] = (time_us, latent, states)
        metadata[case_name] = local_metadata
        fixed_states, fixed_metadata = fixed_dimension_states(
            time_us, states, latent.shape[1]
        )
        metadata[case_name]["fixed_dimension_sensitivity"] = fixed_metadata
        for state_name in STATE_ORDER:
            print(f"[KNN] {case_name} {state_name}", flush=True)
            query_rows, pair_rows = evaluate_ambiguity(
                case_name,
                time_us,
                latent,
                state_name,
                states[state_name],
                COMMON_START_US,
                COMMON_END_US,
                HORIZONS_US,
                THEILER_US,
                K_VALUES,
                True,
            )
            all_query_rows.extend(query_rows)
            all_pair_rows.extend(pair_rows)
            print(f"[KNN-FIXED-DIM] {case_name} {state_name}", flush=True)
            fixed_rows, _ = evaluate_ambiguity(
                case_name,
                time_us,
                latent,
                state_name,
                fixed_states[state_name],
                COMMON_START_US,
                COMMON_END_US,
                HORIZONS_US,
                THEILER_US,
                K_VALUES,
                False,
            )
            all_fixed_query_rows.extend(fixed_rows)
    summary = summarize(all_query_rows)
    fixed_summary = summarize(all_fixed_query_rows)
    distance_summary = distance_matched_summary(all_pair_rows)
    b25_time, b25_latent, b25_states = case_data["B25"]
    rolling = rolling_link(b25_time, b25_latent, b25_states)
    interpretation = build_interpretation(summary, fixed_summary, rolling)

    write_csv(args.output / "query_level_ambiguity.csv", all_query_rows)
    write_csv(args.output / "ambiguity_summary.csv", summary)
    write_csv(args.output / "fixed_dimension_query_level_ambiguity.csv", all_fixed_query_rows)
    write_csv(args.output / "fixed_dimension_ambiguity_summary.csv", fixed_summary)
    write_csv(args.output / "primary_analogue_pairs.csv", all_pair_rows)
    write_csv(args.output / "distance_matched_summary.csv", distance_summary)
    write_csv(args.output / "rolling_ambiguity_vs_rom_skill.csv", rolling)
    plot_horizon(
        summary,
        "mean_ambiguity_normalized",
        "future-increment ambiguity / global variance",
        args.output / "ambiguity_by_horizon.png",
    )
    plot_horizon(
        summary,
        "mean_analog_error_normalized",
        "analogue error / global variance",
        args.output / "analogue_error_by_horizon.png",
    )
    plot_horizon(
        fixed_summary,
        "mean_ambiguity_normalized",
        "fixed-dimension ambiguity / global variance",
        args.output / "fixed_dimension_ambiguity_by_horizon.png",
    )
    plot_distance_matched(distance_summary, args.output / "distance_matched_future_divergence.png")
    plot_rolling(rolling, args.output / "rolling_ambiguity_vs_rom_skill.png")
    payload = {
        "status": "PASS",
        "protocol": {
            "common_interval_us": [COMMON_START_US, COMMON_END_US],
            "reference_interval_us": [COMMON_START_US, REPRESENTATION_END_US],
            "dt_us": DT_US,
            "analogue_step_frames": ANALOG_STEP_FRAMES,
            "horizons_us": HORIZONS_US,
            "theiler_us": THEILER_US,
            "k_values": K_VALUES,
            "state_order": STATE_ORDER,
        },
        "metadata": metadata,
        "interpretation": interpretation,
        "counts": {
            "query_rows": len(all_query_rows),
            "fixed_dimension_query_rows": len(all_fixed_query_rows),
            "primary_pair_rows": len(all_pair_rows),
            "summary_rows": len(summary),
            "distance_matched_rows": len(distance_summary),
            "rolling_rows": len(rolling),
        },
    }
    (args.output / "analysis_summary.json").write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    write_readme(
        args.output / "README.md",
        metadata,
        interpretation,
        summary,
        fixed_summary,
    )
    print(json.dumps(json_safe(interpretation), indent=2, ensure_ascii=True), flush=True)
    print(f"[DONE] {args.output}", flush=True)


if __name__ == "__main__":
    main()
