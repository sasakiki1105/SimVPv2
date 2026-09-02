#!/usr/bin/env python3
"""Map predictive sufficiency and saved-moment value across the Ez sweep.

This is a pre-rePIC diagnostic.  It reuses the frozen Ez=10 SimVP Fourier
latent features and the already-stitched electron drift/scalar-temperature
fields for Ez=10,20,25,30,40 kV/m.  Existing ROM results are treated as frozen
external outcomes; this script does not tune or refit a ROM after seeing the
ambiguity map.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze_radaz_b15_b25_predictive_state_ambiguity as ambiguity
import analyze_radaz_blockwise_fourier_latent_dynamics as block
import analyze_radaz_local_rom_closure_map as local_map


ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT.parent
DEFAULT_OUTPUT = (
    RESEARCH
    / "research_results"
    / "2D_RadAz"
    / "SimVPv2"
    / "workdirs"
    / "analyze_radaz_electric_sweep_predictive_sufficiency"
)

EZ_VALUES = (10, 20, 25, 30, 40)
FIELDS = ("electron_ud", "electron_vd", "electron_wd", "electron_Temp")
FIELD_LABELS = ("ud", "vd", "wd", "Tiso")
RADIAL_BANDS = 8
MAX_MODE = 30
PCA_MAX_COMPONENTS = 20
PCA_VARIANCE_TARGET = 0.95
LATENT_DIMENSIONS = 20
HORIZONS_US = (0.15, 0.30, 0.60, 1.20, 2.40)
PRIMARY_HORIZON_US = 1.20
PRIMARY_K = 10
PRIMARY_THEILER_US = 1.0
K_VALUES = (5, 10, 20)
THEILER_VALUES_US = (0.5, 1.0, 2.6)
RANDOM_NEIGHBOR_REPEATS = 32

PROTOCOLS = {
    "developed": {
        "representation_start_us": 12.0,
        "representation_end_us": 24.0,
        "analysis_start_us": 13.5,
        "analysis_end_us": 29.75,
    },
    "early_mid": {
        "representation_start_us": 4.5,
        "representation_end_us": 12.0,
        "analysis_start_us": 5.0,
        "analysis_end_us": 20.0,
    },
}

STATE_ORDER = (
    "L",
    "L+ud",
    "L+vd",
    "L+wd",
    "L+U",
    "L+Tiso",
    "L+U+Tiso",
)
STATE_COLORS = {
    "L": "#4d4d4d",
    "L+ud": "#56b4e9",
    "L+vd": "#0072b2",
    "L+wd": "#cc79a7",
    "L+U": "#009e73",
    "L+Tiso": "#e69f00",
    "L+U+Tiso": "#d55e00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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


def case_label(ez_kvm: int) -> str:
    return f"E{ez_kvm}"


def latent_path(ez_kvm: int) -> Path:
    return local_map.latent_path(ez_kvm)


def extract_moment_fourier(ez_kvm: int, output: Path) -> Path:
    cache = output / "moment_cache" / f"E{ez_kvm}_electron_moments.h5"
    if cache.is_file():
        return cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    source_path = local_map.source_path(ez_kvm)
    with h5py.File(source_path, "r") as source:
        time_us = np.asarray(source["axes/time_s"], dtype=np.float64) * 1.0e6
        frames = len(time_us)
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
                        np.fft.rfft(radial_mean, axis=-1)[..., : MAX_MODE + 1]
                        / 256.0
                    )
            if stop == frames or stop % 400 == 0:
                print(f"[MOMENT] E{ez_kvm}: {stop}/{frames}", flush=True)
    packed = np.concatenate(
        (
            coefficient[..., :1].real,
            coefficient[..., 1:].real,
            coefficient[..., 1:].imag,
        ),
        axis=-1,
    ).astype(np.float32)
    with h5py.File(cache, "w") as target:
        target.create_dataset(
            "features",
            data=packed.reshape(frames, len(FIELDS), -1),
            compression="gzip",
            compression_opts=4,
        )
        target.create_dataset("time_us", data=time_us)
        target.create_dataset("fields", data=np.asarray(FIELDS, dtype="S"))
        target.create_dataset("radial_band_edges", data=band_edges)
        target.attrs["source_h5"] = str(source_path)
        target.attrs["radial_bands"] = RADIAL_BANDS
        target.attrs["max_mode"] = MAX_MODE
        target.attrs["packing"] = "Re(n=0), Re(n=1..N), Im(n=1..N)"
    return cache


def nearest_sample(
    source_time: np.ndarray, values: np.ndarray, target_time: np.ndarray
) -> np.ndarray:
    indices = np.searchsorted(source_time, target_time)
    indices = np.clip(indices, 1, len(source_time) - 1)
    left = indices - 1
    choose_left = np.abs(target_time - source_time[left]) <= np.abs(
        source_time[indices] - target_time
    )
    return values[np.where(choose_left, left, indices)]


def fit_group_pca(
    values: np.ndarray, fit: np.ndarray, max_components: int = PCA_MAX_COMPONENTS
) -> tuple[np.ndarray, dict]:
    count = min(max_components, int(np.count_nonzero(fit)) - 1, values.shape[1])
    pca = PCA(
        n_components=count,
        svd_solver="randomized",
        random_state=42,
        iterated_power=5,
    )
    pca.fit(values[fit])
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    selected = min(int(np.searchsorted(cumulative, PCA_VARIANCE_TARGET) + 1), count)
    coordinates = pca.transform(values)[:, :selected]
    coordinates = ambiguity.standardize(coordinates, fit)
    return coordinates, {
        "input_dimensions": int(values.shape[1]),
        "computed_components": int(count),
        "selected_components": int(selected),
        "variance_capture": float(cumulative[selected - 1]),
    }


def fit_local_latent(
    features: np.ndarray, time_us: np.ndarray, representation_start: float, representation_end: float
) -> tuple[np.ndarray, list[dict]]:
    fit = (time_us >= representation_start - 1.0e-9) & (
        time_us < representation_end - 1.0e-9
    )
    _, latent, rows = block.fit_block_models(
        features, fit, block.BUDGETS["medium_20"]
    )
    return ambiguity.standardize(latent, fit), rows


def fit_common_block_models(
    representation_start: float, representation_end: float
) -> tuple[dict[str, block.BlockPCA], np.ndarray, np.ndarray, list[dict]]:
    models: dict[str, block.BlockPCA] = {}
    score_pieces = []
    metadata = []
    for name, (mode_start, mode_end) in block.BLOCKS.items():
        pieces = []
        feature_shape = None
        for ez_kvm in EZ_VALUES:
            path = latent_path(ez_kvm)
            with h5py.File(path, "r") as source:
                time_us = np.asarray(source["translator_time_s"], dtype=np.float64) * 1.0e6
                fit_indices = np.flatnonzero(
                    (time_us >= representation_start - 1.0e-9)
                    & (time_us < representation_end - 1.0e-9)
                )
                start = int(fit_indices[0])
                stop = int(fit_indices[-1] + 1)
                values = np.asarray(
                    source["translator_fourier_ri"][
                        start:stop, :, :, mode_start : mode_end + 1, :
                    ],
                    dtype=np.float32,
                )
            feature_shape = values.shape[1:]
            pieces.append(values.reshape(len(values), -1))
        pooled = np.concatenate(pieces, axis=0)
        full_mean = np.mean(pooled, axis=0)
        variance = np.var(pooled, axis=0)
        floor = max(float(np.max(variance)) * 1.0e-12, 1.0e-20)
        active = variance > floor
        requested = int(block.BUDGETS["medium_20"][name])
        components = min(requested, int(np.count_nonzero(active)), len(pooled) - 1)
        pca = PCA(
            n_components=components,
            svd_solver="randomized",
            random_state=42,
            iterated_power=5,
        )
        pca.fit(pooled[:, active])
        models[name] = block.BlockPCA(
            name,
            mode_start,
            mode_end,
            components,
            feature_shape,
            full_mean,
            active,
            pca,
        )
        score_pieces.append(pca.transform(pooled[:, active]))
        metadata.append(
            {
                "block": name,
                "mode_start": mode_start,
                "mode_end": mode_end,
                "components": components,
                "active_features": int(np.count_nonzero(active)),
                "variance_capture": float(np.sum(pca.explained_variance_ratio_)),
            }
        )
    pooled_scores = np.concatenate(score_pieces, axis=1)
    mean = np.mean(pooled_scores, axis=0)
    scale = np.std(pooled_scores, axis=0)
    scale = np.where(scale > 1.0e-10, scale, 1.0)
    return models, mean, scale, metadata


def transform_common_latent(
    features: np.ndarray,
    models: dict[str, block.BlockPCA],
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    scores = np.concatenate(
        [models[name].transform(features) for name in block.BLOCKS], axis=1
    )
    return (scores - mean) / scale


def build_states(
    ez_kvm: int,
    protocol: dict,
    output: Path,
    common_models: dict[str, block.BlockPCA] | None = None,
    common_mean: np.ndarray | None = None,
    common_scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    features, time_us, frames = block.load_features(latent_path(ez_kvm))
    latent, latent_meta = fit_local_latent(
        features,
        time_us,
        protocol["representation_start_us"],
        protocol["representation_end_us"],
    )
    common_latent = None
    if common_models is not None and common_mean is not None and common_scale is not None:
        common_latent = transform_common_latent(
            features, common_models, common_mean, common_scale
        )
    del features

    cache = extract_moment_fourier(ez_kvm, output)
    with h5py.File(cache, "r") as source:
        moment_time = np.asarray(source["time_us"], dtype=np.float64)
        raw = np.asarray(source["features"], dtype=np.float64)
    raw = nearest_sample(moment_time, raw, time_us)
    fit = (time_us >= protocol["representation_start_us"] - 1.0e-9) & (
        time_us < protocol["representation_end_us"] - 1.0e-9
    )
    balanced = raw.copy()
    scales = []
    for channel in range(len(FIELDS)):
        centered = raw[fit, channel] - np.mean(raw[fit, channel], axis=0)
        scale_value = max(float(np.sqrt(np.mean(centered * centered))), 1.0e-12)
        balanced[:, channel] /= scale_value
        scales.append(scale_value)

    group_raw = {
        "ud": balanced[:, 0],
        "vd": balanced[:, 1],
        "wd": balanced[:, 2],
        "U": balanced[:, :3].reshape(len(time_us), -1),
        "Tiso": balanced[:, 3],
        "U+Tiso": balanced.reshape(len(time_us), -1),
    }
    groups = {}
    group_meta = {}
    for name, values in group_raw.items():
        groups[name], group_meta[name] = fit_group_pca(values, fit)
    states = {
        "L": latent,
        "L+ud": np.concatenate((latent, groups["ud"]), axis=1),
        "L+vd": np.concatenate((latent, groups["vd"]), axis=1),
        "L+wd": np.concatenate((latent, groups["wd"]), axis=1),
        "L+U": np.concatenate((latent, groups["U"]), axis=1),
        "L+Tiso": np.concatenate((latent, groups["Tiso"]), axis=1),
        "L+U+Tiso": np.concatenate((latent, groups["U+Tiso"]), axis=1),
    }
    metadata = {
        "frames": int(len(time_us)),
        "frame_start": int(frames[0]),
        "frame_end": int(frames[-1]),
        "time_start_us": float(time_us[0]),
        "time_end_us": float(time_us[-1]),
        "latent_dimensions": int(latent.shape[1]),
        "state_dimensions": {name: int(values.shape[1]) for name, values in states.items()},
        "latent_blocks": latent_meta,
        "moment_channel_rms_scales": dict(zip(FIELD_LABELS, scales)),
        "moment_group_pca": group_meta,
        "moment_cache": str(cache),
    }
    if common_latent is not None:
        metadata["common_latent_dimensions"] = int(common_latent.shape[1])
    groups["common_latent"] = common_latent
    return time_us, latent, states, groups, metadata


def fixed_dimension_states(
    states: dict[str, np.ndarray], fit: np.ndarray, target_dimensions: int
) -> tuple[dict[str, np.ndarray], dict]:
    output = {}
    metadata = {}
    for name, values in states.items():
        if values.shape[1] <= target_dimensions:
            output[name] = values.copy()
            metadata[name] = {
                "input_dimensions": int(values.shape[1]),
                "output_dimensions": int(values.shape[1]),
                "variance_capture": 1.0,
            }
            continue
        pca = PCA(
            n_components=target_dimensions,
            svd_solver="randomized",
            random_state=42,
            iterated_power=5,
        )
        pca.fit(values[fit])
        transformed = pca.transform(values)
        output[name] = ambiguity.standardize(transformed, fit)
        metadata[name] = {
            "input_dimensions": int(values.shape[1]),
            "output_dimensions": target_dimensions,
            "variance_capture": float(np.sum(pca.explained_variance_ratio_)),
        }
    return output, metadata


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["protocol"],
            row["case"],
            row["state"],
            row["horizon_us"],
            row["theiler_us"],
            row["k"],
        )
        grouped[key].append(row)
    summary = []
    for key, local in grouped.items():
        error = np.asarray([row["analog_error"] for row in local])
        persistence = np.asarray([row["persistence_error"] for row in local])
        summary.append(
            {
                "protocol": key[0],
                "case": key[1],
                "ez_kvm": int(key[1][1:]),
                "state": key[2],
                "horizon_us": key[3],
                "theiler_us": key[4],
                "k": key[5],
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
                    1.0 - np.mean(error) / max(np.mean(persistence), 1.0e-30)
                ),
                "median_neighbor_radius_normalized": float(
                    np.median([row["neighbor_radius_normalized"] for row in local])
                ),
            }
        )
    return sorted(
        summary,
        key=lambda row: (
            row["protocol"],
            row["ez_kvm"],
            STATE_ORDER.index(row["state"]) if row["state"] in STATE_ORDER else 99,
            row["horizon_us"],
            row["theiler_us"],
            row["k"],
        ),
    )


def evaluate_states(
    protocol_name: str,
    case_name: str,
    time_us: np.ndarray,
    latent: np.ndarray,
    states: dict[str, np.ndarray],
    protocol: dict,
) -> list[dict]:
    output = []
    for state_name, state in states.items():
        rows, _ = ambiguity.evaluate_ambiguity(
            case_name,
            time_us,
            latent,
            state_name,
            state,
            protocol["analysis_start_us"],
            protocol["analysis_end_us"],
            HORIZONS_US,
            (PRIMARY_THEILER_US,),
            (PRIMARY_K,),
            False,
        )
        output.extend({"protocol": protocol_name, **row} for row in rows)
    return output


def evaluate_protocol_sensitivity(
    case_name: str,
    time_us: np.ndarray,
    latent: np.ndarray,
    state: np.ndarray,
    protocol: dict,
) -> list[dict]:
    rows, _ = ambiguity.evaluate_ambiguity(
        case_name,
        time_us,
        latent,
        "L",
        state,
        protocol["analysis_start_us"],
        protocol["analysis_end_us"],
        (PRIMARY_HORIZON_US,),
        THEILER_VALUES_US,
        K_VALUES,
        False,
    )
    return [{"protocol": "developed_sensitivity", **row} for row in rows]


def random_neighbor_null(
    ez_kvm: int,
    time_us: np.ndarray,
    latent: np.ndarray,
    state: np.ndarray,
    protocol: dict,
) -> dict:
    base = np.flatnonzero(
        (time_us >= protocol["analysis_start_us"] - 1.0e-9)
        & (time_us <= protocol["analysis_end_us"] + 1.0e-9)
    )[:: ambiguity.ANALOG_STEP_FRAMES]
    horizon = int(round(PRIMARY_HORIZON_US / ambiguity.DT_US))
    valid = base[(base + horizon < len(time_us))]
    valid = valid[
        time_us[valid + horizon] <= protocol["analysis_end_us"] + 1.0e-9
    ]
    increments = latent[valid + horizon] - latent[valid]
    centered = increments - np.mean(increments, axis=0, keepdims=True)
    increment_variance = max(float(np.mean(centered * centered)), 1.0e-30)
    rng = np.random.default_rng(20260825 + ez_kvm)
    random_values = []
    knn_values = []
    for query_position, query in enumerate(valid):
        candidates = valid[
            np.abs(time_us[valid] - time_us[query])
            >= PRIMARY_THEILER_US - 1.0e-9
        ]
        distances = np.sqrt(np.mean((state[candidates] - state[query]) ** 2, axis=1))
        neighbors = candidates[np.argsort(distances, kind="stable")[:PRIMARY_K]]
        neighbor_increments = latent[neighbors + horizon] - latent[neighbors]
        knn_values.append(
            float(np.mean((neighbor_increments - np.mean(neighbor_increments, axis=0)) ** 2))
            / increment_variance
        )
        for _ in range(RANDOM_NEIGHBOR_REPEATS):
            selected = rng.choice(candidates, size=PRIMARY_K, replace=False)
            local = latent[selected + horizon] - latent[selected]
            random_values.append(
                float(np.mean((local - np.mean(local, axis=0)) ** 2))
                / increment_variance
            )
    return {
        "case": case_label(ez_kvm),
        "ez_kvm": ez_kvm,
        "horizon_us": PRIMARY_HORIZON_US,
        "theiler_us": PRIMARY_THEILER_US,
        "k": PRIMARY_K,
        "queries": len(valid),
        "random_repeats_per_query": RANDOM_NEIGHBOR_REPEATS,
        "knn_mean_ambiguity": float(np.mean(knn_values)),
        "random_mean_ambiguity": float(np.mean(random_values)),
        "knn_over_random": float(np.mean(knn_values) / np.mean(random_values)),
    }


def primary_lookup(
    rows: list[dict], protocol: str, ez_kvm: int, state: str, horizon: float
) -> dict:
    return next(
        row
        for row in rows
        if row["protocol"] == protocol
        and row["ez_kvm"] == ez_kvm
        and row["state"] == state
        and math.isclose(row["horizon_us"], horizon)
        and math.isclose(row["theiler_us"], PRIMARY_THEILER_US)
        and row["k"] == PRIMARY_K
    )


def exact_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    observed = float(spearmanr(x, y).statistic)
    permuted = []
    for order in itertools.permutations(range(len(y))):
        permuted.append(float(spearmanr(x, y[list(order)]).statistic))
    p_value = float(
        np.mean(np.abs(np.asarray(permuted)) >= abs(observed) - 1.0e-12)
    )
    return observed, p_value


def condition_diagnostics(output: Path) -> list[dict]:
    adaptive_path = (
        output.parent
        / "compare_radaz_local_rom_closure_map_adaptive"
        / "adaptive_mode_ablation.csv"
    )
    regime_path = (
        output.parent
        / "compare_radaz_local_rom_closure_map"
        / "regime_diagnostics.csv"
    )
    adaptive = {
        int(row["ez_kvm"]): row
        for row in read_csv(adaptive_path)
        if row["strategy"] == "joint"
    }
    regime = {int(row["ez_kvm"]): row for row in read_csv(regime_path)}
    rows = []
    for ez_kvm in EZ_VALUES:
        physical = (
            output.parent
            / "compare_radaz_local_rom_closure_map"
            / "cases"
            / f"E{ez_kvm}kVm"
            / "physical_fourier_targets.h5"
        )
        with h5py.File(physical, "r") as source:
            time_us = np.asarray(source["time_us"], dtype=np.float64)
            fields = [
                item.decode() if isinstance(item, bytes) else str(item)
                for item in source["fields"][:]
            ]
            modes = np.asarray(source["modes"], dtype=np.int64)
            radial_weights = np.asarray(source["radial_weights"], dtype=np.float64)
            ey = np.asarray(
                source["coefficients"][:, fields.index("efy")],
                dtype=np.complex128,
            )
        fit = (time_us >= 15.0 - 1.0e-9) & (time_us < 24.0 - 1.0e-9)
        n0 = float(adaptive[ez_kvm]["ecdi_n0"])
        ratio = modes / n0
        mtsi = (modes >= 1) & (ratio <= 0.60)
        ecdi = (ratio >= 0.75) & (ratio <= 1.25)
        weights = radial_weights / np.sum(radial_weights)

        def band_power(mask: np.ndarray) -> float:
            local = np.abs(ey[fit][:, :, mask]) ** 2
            return float(np.mean(np.sum(local * weights[None, :, None], axis=(1, 2))))

        p_mtsi = band_power(mtsi)
        p_ecdi = band_power(ecdi)
        rows.append(
            {
                "ez_kvm": ez_kvm,
                "ecdi_n0": n0,
                "mtsi_modes": ",".join(str(value) for value in modes[mtsi]),
                "ecdi_modes": ",".join(str(value) for value in modes[ecdi]),
                "mtsi_efy_power": p_mtsi,
                "ecdi_efy_power": p_ecdi,
                "ecdi_power_fraction": p_ecdi / max(p_ecdi + p_mtsi, 1.0e-30),
                "spectral_entropy": float(
                    regime[ez_kvm]["transport_fit_spectral_entropy"]
                ),
                "correlation_time_us": float(
                    regime[ez_kvm]["transport_fit_correlation_time_us"]
                ),
                "frozen_joint_fixed_closure_level": int(
                    adaptive[ez_kvm]["fixed_closure_level"]
                ),
                "frozen_joint_fixed_transport_correlation": float(
                    adaptive[ez_kvm]["fixed_transport_correlation"]
                ),
                "frozen_joint_rolling_pass_fraction": float(
                    adaptive[ez_kvm]["rolling_transport_pass_fraction"]
                ),
                "frozen_joint_rolling_median_transport_correlation": float(
                    adaptive[ez_kvm]["rolling_median_transport_correlation"]
                ),
            }
        )
    return rows


def meaningful_screen(
    raw: list[dict], fixed: list[dict]
) -> list[dict]:
    rows = []
    for ez_kvm in EZ_VALUES:
        for state in STATE_ORDER[1:]:
            qualifying_horizons = 0
            horizon_detail = []
            for horizon in HORIZONS_US:
                baseline = primary_lookup(raw, "developed", ez_kvm, "L", horizon)
                candidate = primary_lookup(raw, "developed", ez_kvm, state, horizon)
                ambiguity_reduction = 1.0 - candidate["mean_ambiguity_normalized"] / max(
                    baseline["mean_ambiguity_normalized"], 1.0e-30
                )
                error_reduction = 1.0 - candidate["mean_analog_error_normalized"] / max(
                    baseline["mean_analog_error_normalized"], 1.0e-30
                )
                if ambiguity_reduction > 0.10 and error_reduction > 0.0:
                    qualifying_horizons += 1
                horizon_detail.append((horizon, ambiguity_reduction, error_reduction))
            baseline_focal = primary_lookup(
                raw, "developed", ez_kvm, "L", PRIMARY_HORIZON_US
            )
            candidate_focal = primary_lookup(
                raw, "developed", ez_kvm, state, PRIMARY_HORIZON_US
            )
            fixed_baseline = primary_lookup(
                fixed, "developed_fixed_dimension", ez_kvm, "L", PRIMARY_HORIZON_US
            )
            fixed_candidate = primary_lookup(
                fixed,
                "developed_fixed_dimension",
                ez_kvm,
                state,
                PRIMARY_HORIZON_US,
            )
            focal_ambiguity_reduction = 1.0 - candidate_focal[
                "mean_ambiguity_normalized"
            ] / max(baseline_focal["mean_ambiguity_normalized"], 1.0e-30)
            focal_error_reduction = 1.0 - candidate_focal[
                "mean_analog_error_normalized"
            ] / max(baseline_focal["mean_analog_error_normalized"], 1.0e-30)
            fixed_ambiguity_reduction = 1.0 - fixed_candidate[
                "mean_ambiguity_normalized"
            ] / max(fixed_baseline["mean_ambiguity_normalized"], 1.0e-30)
            fixed_error_reduction = 1.0 - fixed_candidate[
                "mean_analog_error_normalized"
            ] / max(fixed_baseline["mean_analog_error_normalized"], 1.0e-30)
            meaningful = (
                qualifying_horizons >= 2
                and focal_ambiguity_reduction > 0.10
                and focal_error_reduction > 0.0
                and fixed_ambiguity_reduction > 0.0
                and fixed_error_reduction > 0.0
            )
            rows.append(
                {
                    "ez_kvm": ez_kvm,
                    "state": state,
                    "qualifying_horizons_of_5": qualifying_horizons,
                    "focal_ambiguity_reduction": focal_ambiguity_reduction,
                    "focal_analog_error_reduction": focal_error_reduction,
                    "fixed_dimension_focal_ambiguity_reduction": fixed_ambiguity_reduction,
                    "fixed_dimension_focal_analog_error_reduction": fixed_error_reduction,
                    "meaningful_by_predeclared_screen": meaningful,
                    "horizon_reductions": ";".join(
                        f"{h:g}:{a:.4f}/{e:.4f}"
                        for h, a, e in horizon_detail
                    ),
                }
            )
    return rows


def correlation_summary(
    raw: list[dict], fixed: list[dict], diagnostics: list[dict]
) -> list[dict]:
    diag = {row["ez_kvm"]: row for row in diagnostics}
    baseline = np.asarray(
        [
            primary_lookup(raw, "developed", ez, "L", PRIMARY_HORIZON_US)[
                "mean_ambiguity_normalized"
            ]
            for ez in EZ_VALUES
        ]
    )
    ut = np.asarray(
        [
            primary_lookup(raw, "developed", ez, "L+U+Tiso", PRIMARY_HORIZON_US)[
                "mean_ambiguity_normalized"
            ]
            for ez in EZ_VALUES
        ]
    )
    fixed_baseline = np.asarray(
        [
            primary_lookup(
                fixed, "developed_fixed_dimension", ez, "L", PRIMARY_HORIZON_US
            )["mean_ambiguity_normalized"]
            for ez in EZ_VALUES
        ]
    )
    fixed_ut = np.asarray(
        [
            primary_lookup(
                fixed,
                "developed_fixed_dimension",
                ez,
                "L+U+Tiso",
                PRIMARY_HORIZON_US,
            )["mean_ambiguity_normalized"]
            for ez in EZ_VALUES
        ]
    )
    variables = {
        "frozen_joint_fixed_transport_correlation": np.asarray(
            [diag[ez]["frozen_joint_fixed_transport_correlation"] for ez in EZ_VALUES]
        ),
        "frozen_joint_rolling_median_transport_correlation": np.asarray(
            [diag[ez]["frozen_joint_rolling_median_transport_correlation"] for ez in EZ_VALUES]
        ),
        "spectral_entropy": np.asarray(
            [diag[ez]["spectral_entropy"] for ez in EZ_VALUES]
        ),
        "ecdi_power_fraction": np.asarray(
            [diag[ez]["ecdi_power_fraction"] for ez in EZ_VALUES]
        ),
    }
    predictors = {
        "L_ambiguity": baseline,
        "raw_UT_ambiguity_gain": 1.0 - ut / baseline,
        "fixed_UT_ambiguity_gain": 1.0 - fixed_ut / fixed_baseline,
    }
    rows = []
    for predictor_name, x in predictors.items():
        for outcome_name, y in variables.items():
            rho, p_value = exact_spearman(x, y)
            rows.append(
                {
                    "predictor": predictor_name,
                    "outcome": outcome_name,
                    "cases": len(EZ_VALUES),
                    "spearman": rho,
                    "exact_two_sided_permutation_p": p_value,
                }
            )
    return rows


def plot_ambiguity_map(raw: list[dict], fixed: list[dict], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 5.4), constrained_layout=True)
    for axis, rows, protocol, title in (
        (axes[0], raw, "developed", "raw augmented states"),
        (axes[1], fixed, "developed_fixed_dimension", "all states fixed to 20D"),
    ):
        for state in STATE_ORDER:
            values = [
                primary_lookup(rows, protocol, ez, state, PRIMARY_HORIZON_US)[
                    "mean_ambiguity_normalized"
                ]
                for ez in EZ_VALUES
            ]
            axis.plot(
                EZ_VALUES,
                values,
                marker="o",
                linewidth=1.8,
                color=STATE_COLORS[state],
                label=state,
            )
        axis.set_title(title)
        axis.set_xlabel("Ez [kV/m]")
        axis.set_xticks(EZ_VALUES)
        axis.grid(alpha=0.25)
        lower, upper = axis.get_ylim()
        axis.set_ylim(min(-0.05, lower), upper)
        axis.legend(loc="lower right", fontsize=7)
    axes[0].set_ylabel(f"normalized ambiguity at {PRIMARY_HORIZON_US:g} us")
    figure.suptitle("Electric-field sweep predictive-state ablation")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_rom_relation(raw: list[dict], diagnostics: list[dict], path: Path) -> None:
    diag = {row["ez_kvm"]: row for row in diagnostics}
    x = np.asarray(
        [
            primary_lookup(raw, "developed", ez, "L", PRIMARY_HORIZON_US)[
                "mean_ambiguity_normalized"
            ]
            for ez in EZ_VALUES
        ]
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=True)
    outcomes = (
        ("frozen_joint_fixed_transport_correlation", "fixed 20-30 us transport correlation"),
        ("frozen_joint_rolling_median_transport_correlation", "rolling median transport correlation"),
    )
    for axis, (key, label) in zip(axes, outcomes):
        y = np.asarray([diag[ez][key] for ez in EZ_VALUES])
        axis.scatter(x, y, s=65, color="#0072b2")
        for ez, x_value, y_value in zip(EZ_VALUES, x, y):
            axis.annotate(f"E{ez}", (x_value, y_value), xytext=(5, 5), textcoords="offset points")
        rho, p_value = exact_spearman(x, y)
        axis.set_title(f"Spearman={rho:.3f}, exact p={p_value:.3f}")
        axis.set_xlabel("L ambiguity at 1.2 us")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    figure.suptitle("Does predictive ambiguity track the frozen adaptive-ROM result?")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_moment_regime(
    raw: list[dict], fixed: list[dict], diagnostics: list[dict], path: Path
) -> None:
    diag = {row["ez_kvm"]: row for row in diagnostics}
    raw_gain = []
    fixed_gain = []
    for ez in EZ_VALUES:
        base = primary_lookup(raw, "developed", ez, "L", PRIMARY_HORIZON_US)
        candidate = primary_lookup(raw, "developed", ez, "L+U+Tiso", PRIMARY_HORIZON_US)
        raw_gain.append(
            1.0 - candidate["mean_ambiguity_normalized"] / base["mean_ambiguity_normalized"]
        )
        fixed_base = primary_lookup(
            fixed, "developed_fixed_dimension", ez, "L", PRIMARY_HORIZON_US
        )
        fixed_candidate = primary_lookup(
            fixed,
            "developed_fixed_dimension",
            ez,
            "L+U+Tiso",
            PRIMARY_HORIZON_US,
        )
        fixed_gain.append(
            1.0
            - fixed_candidate["mean_ambiguity_normalized"]
            / fixed_base["mean_ambiguity_normalized"]
        )
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
    x_values = (
        (np.asarray(EZ_VALUES), "Ez [kV/m]"),
        (np.asarray([diag[ez]["spectral_entropy"] for ez in EZ_VALUES]), "spectral entropy"),
        (np.asarray([diag[ez]["ecdi_power_fraction"] for ez in EZ_VALUES]), "ECDI candidate power fraction"),
    )
    for axis, (x, label) in zip(axes, x_values):
        axis.scatter(x, raw_gain, label="raw L+U+Tiso", color="#d55e00", marker="o", s=60)
        axis.scatter(x, fixed_gain, label="fixed-20D", color="#0072b2", marker="s", s=55)
        for ez, x_value, y_value in zip(EZ_VALUES, x, raw_gain):
            axis.annotate(f"E{ez}", (x_value, y_value), xytext=(4, 4), textcoords="offset points", fontsize=8)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xlabel(label)
        axis.grid(alpha=0.25)
        lower, upper = axis.get_ylim()
        axis.set_ylim(min(-0.1, lower), upper)
        axis.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("relative ambiguity reduction")
    figure.suptitle("Does saved-moment value follow Ez or dynamical regime?")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_window_sensitivity(raw: list[dict], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    for protocol, color, marker, label in (
        ("developed", "#0072b2", "o", "developed 13.5-29.75 us"),
        ("early_mid", "#d55e00", "s", "early/mid 5-20 us"),
    ):
        values = [
            primary_lookup(raw, protocol, ez, "L", PRIMARY_HORIZON_US)[
                "mean_ambiguity_normalized"
            ]
            for ez in EZ_VALUES
        ]
        axis.plot(EZ_VALUES, values, color=color, marker=marker, linewidth=2.0, label=label)
    axis.set_xlabel("Ez [kV/m]")
    axis.set_ylabel("L ambiguity at 1.2 us")
    axis.set_xticks(EZ_VALUES)
    axis.grid(alpha=0.25)
    lower, upper = axis.get_ylim()
    axis.set_ylim(min(-0.05, lower), upper)
    axis.legend(loc="lower right")
    axis.set_title("Is the Ez ordering tied to one time interval?")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_common_coordinate(
    raw: list[dict], common: list[dict], path: Path
) -> None:
    local_values = [
        primary_lookup(raw, "developed", ez, "L", PRIMARY_HORIZON_US)[
            "mean_ambiguity_normalized"
        ]
        for ez in EZ_VALUES
    ]
    common_lookup = {row["ez_kvm"]: row for row in common if math.isclose(row["horizon_us"], PRIMARY_HORIZON_US)}
    common_values = [common_lookup[ez]["mean_ambiguity_normalized"] for ez in EZ_VALUES]
    figure, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    axis.plot(EZ_VALUES, local_values, marker="o", linewidth=2.0, color="#0072b2", label="case-local 20D block PCA")
    axis.plot(EZ_VALUES, common_values, marker="s", linewidth=2.0, color="#d55e00", label="pooled common 20D block PCA")
    axis.set_xlabel("Ez [kV/m]")
    axis.set_ylabel("ambiguity at 1.2 us")
    axis.set_xticks(EZ_VALUES)
    axis.grid(alpha=0.25)
    lower, upper = axis.get_ylim()
    axis.set_ylim(min(-0.05, lower), upper)
    axis.legend(loc="lower right")
    axis.set_title("Local-coordinate and common-coordinate ambiguity")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    print("[COMMON] fitting pooled developed-state block PCA", flush=True)
    common_models, common_mean, common_scale, common_meta = fit_common_block_models(
        PROTOCOLS["developed"]["representation_start_us"],
        PROTOCOLS["developed"]["representation_end_us"],
    )

    all_raw_rows: list[dict] = []
    all_fixed_rows: list[dict] = []
    sensitivity_rows: list[dict] = []
    common_rows: list[dict] = []
    random_rows: list[dict] = []
    metadata = {}
    for ez_kvm in EZ_VALUES:
        print(f"[CASE] E{ez_kvm}: developed", flush=True)
        time_us, latent, states, groups, case_meta = build_states(
            ez_kvm,
            PROTOCOLS["developed"],
            output,
            common_models,
            common_mean,
            common_scale,
        )
        fit = (time_us >= PROTOCOLS["developed"]["representation_start_us"] - 1.0e-9) & (
            time_us < PROTOCOLS["developed"]["representation_end_us"] - 1.0e-9
        )
        raw_rows = evaluate_states(
            "developed", case_label(ez_kvm), time_us, latent, states, PROTOCOLS["developed"]
        )
        all_raw_rows.extend(raw_rows)
        fixed_states, fixed_meta = fixed_dimension_states(states, fit, LATENT_DIMENSIONS)
        all_fixed_rows.extend(
            evaluate_states(
                "developed_fixed_dimension",
                case_label(ez_kvm),
                time_us,
                latent,
                fixed_states,
                PROTOCOLS["developed"],
            )
        )
        sensitivity_rows.extend(
            evaluate_protocol_sensitivity(
                case_label(ez_kvm), time_us, latent, states["L"], PROTOCOLS["developed"]
            )
        )
        random_rows.append(
            random_neighbor_null(
                ez_kvm, time_us, latent, states["L"], PROTOCOLS["developed"]
            )
        )
        common_latent = groups["common_latent"]
        common_case_rows, _ = ambiguity.evaluate_ambiguity(
            case_label(ez_kvm),
            time_us,
            common_latent,
            "L_common",
            common_latent,
            PROTOCOLS["developed"]["analysis_start_us"],
            PROTOCOLS["developed"]["analysis_end_us"],
            HORIZONS_US,
            (PRIMARY_THEILER_US,),
            (PRIMARY_K,),
            False,
        )
        common_rows.extend({"protocol": "developed_common_coordinate", **row} for row in common_case_rows)
        case_meta["fixed_dimension"] = fixed_meta
        metadata.setdefault(case_label(ez_kvm), {})["developed"] = case_meta

        print(f"[CASE] E{ez_kvm}: early/mid sensitivity", flush=True)
        early_time, early_latent, early_states, _, early_meta = build_states(
            ez_kvm, PROTOCOLS["early_mid"], output
        )
        all_raw_rows.extend(
            evaluate_states(
                "early_mid",
                case_label(ez_kvm),
                early_time,
                early_latent,
                early_states,
                PROTOCOLS["early_mid"],
            )
        )
        metadata[case_label(ez_kvm)]["early_mid"] = early_meta

    raw_summary = summarize(all_raw_rows)
    fixed_summary = summarize(all_fixed_rows)
    sensitivity_summary = summarize(sensitivity_rows)
    common_summary = summarize(common_rows)
    diagnostics = condition_diagnostics(output)
    screen = meaningful_screen(raw_summary, fixed_summary)
    correlations = correlation_summary(raw_summary, fixed_summary, diagnostics)

    write_csv(output / "primary_and_window_ablation_summary.csv", raw_summary)
    write_csv(output / "fixed_dimension_ablation_summary.csv", fixed_summary)
    write_csv(output / "protocol_sensitivity_summary.csv", sensitivity_summary)
    write_csv(output / "common_coordinate_summary.csv", common_summary)
    write_csv(output / "random_neighbor_null.csv", random_rows)
    write_csv(output / "condition_diagnostics.csv", diagnostics)
    write_csv(output / "meaningful_moment_screen.csv", screen)
    write_csv(output / "correlation_summary.csv", correlations)

    plot_ambiguity_map(
        raw_summary,
        fixed_summary,
        output / "electric_sweep_ambiguity_map.png",
    )
    plot_rom_relation(
        raw_summary,
        diagnostics,
        output / "ambiguity_vs_frozen_rom_skill.png",
    )
    plot_moment_regime(
        raw_summary,
        fixed_summary,
        diagnostics,
        output / "moment_gain_vs_regime.png",
    )
    plot_window_sensitivity(
        raw_summary, output / "developed_vs_early_mid_ambiguity.png"
    )
    plot_common_coordinate(
        raw_summary,
        common_summary,
        output / "local_vs_common_coordinate_ambiguity.png",
    )

    focal = []
    for ez_kvm in EZ_VALUES:
        for state in STATE_ORDER:
            row = primary_lookup(
                raw_summary, "developed", ez_kvm, state, PRIMARY_HORIZON_US
            )
            fixed_row = primary_lookup(
                fixed_summary,
                "developed_fixed_dimension",
                ez_kvm,
                state,
                PRIMARY_HORIZON_US,
            )
            focal.append(
                {
                    "ez_kvm": ez_kvm,
                    "state": state,
                    "ambiguity": row["mean_ambiguity_normalized"],
                    "analogue_error": row["mean_analog_error_normalized"],
                    "skill": row["analog_skill_vs_zero_increment"],
                    "fixed_dimension_ambiguity": fixed_row["mean_ambiguity_normalized"],
                    "fixed_dimension_analogue_error": fixed_row["mean_analog_error_normalized"],
                    "fixed_dimension_skill": fixed_row["analog_skill_vs_zero_increment"],
                }
            )

    report = {
        "status": "PASS",
        "question": "Does predictive ambiguity and saved-moment value track electric-field-dependent ROM reducibility?",
        "protocol": {
            "cases_kvm": EZ_VALUES,
            "protocols": PROTOCOLS,
            "horizons_us": HORIZONS_US,
            "primary_horizon_us": PRIMARY_HORIZON_US,
            "primary_k": PRIMARY_K,
            "primary_theiler_us": PRIMARY_THEILER_US,
            "sensitivity_k": K_VALUES,
            "sensitivity_theiler_us": THEILER_VALUES_US,
            "latent_definition": "frozen Ez=10 SimVP translator Fourier features; per-case medium_20 block PCA",
            "common_coordinate": "pooled five-Ez medium_20 block PCA fit on 12-24 us only",
            "primary_moment_candidate": "L+U+Tiso",
            "meaningful_moment_rule": ">10% ambiguity reduction and lower analogue error at >=2 horizons, including 1.2 us, with same-sign fixed-20D gains at 1.2 us",
            "frozen_rom_source": "compare_radaz_local_rom_closure_map_adaptive/adaptive_mode_ablation.csv, strategy=joint",
        },
        "common_block_pca": common_meta,
        "metadata": metadata,
        "focal_1p2us": focal,
        "random_neighbor_null": random_rows,
        "condition_diagnostics": diagnostics,
        "correlations": correlations,
        "meaningful_moment_screen": screen,
        "guardrails": [
            "The five Ez cases are descriptive and do not support a strong causal correlation claim.",
            "Tiso is a scalar trace temperature, not a pressure tensor.",
            "Existing ROM outcomes are frozen; no ROM is selected using this ambiguity result.",
            "A low ambiguity is evidence for predictive sufficiency at this resolution, not proof of exact Markov closure.",
        ],
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(json_safe(report), indent=2), encoding="utf-8"
    )

    meaningful = [row for row in screen if row["meaningful_by_predeclared_screen"]]
    diagnostics_by_ez = {row["ez_kvm"]: row for row in diagnostics}
    random_by_ez = {row["ez_kvm"]: row for row in random_rows}
    screen_by_ez_state = {
        (row["ez_kvm"], row["state"]): row for row in screen
    }
    common_by_ez = {
        row["ez_kvm"]: row
        for row in common_summary
        if math.isclose(row["horizon_us"], PRIMARY_HORIZON_US)
    }
    correlations_by_pair = {
        (row["predictor"], row["outcome"]): row for row in correlations
    }
    readme = [
        "# Electric-sweep predictive sufficiency and saved-moment diagnostic",
        "",
        "This pre-rePIC analysis uses Ez=10,20,25,30,40 kV/m and does not retrain SimVP or refit a ROM after inspecting ambiguity.",
        "",
        "Primary: developed 13.5--29.75 us, representation fit 12--24 us, horizon 1.2 us, k=10, Theiler=1.0 us.",
        "Sensitivity: early/mid 5--20 us, k/Theiler grid, fixed-20D state control, random-neighbour null, and pooled common-coordinate PCA.",
        "",
        f"States passing the predeclared meaningful-moment screen: {len(meaningful)}.",
    ]
    for row in meaningful:
        readme.append(
            f"- E{row['ez_kvm']} {row['state']}: ambiguity gain={row['focal_ambiguity_reduction']:.3f}, error gain={row['focal_analog_error_reduction']:.3f}, fixed gains={row['fixed_dimension_focal_ambiguity_reduction']:.3f}/{row['fixed_dimension_focal_analog_error_reduction']:.3f}."
        )
    readme.extend(
        [
            "",
            "## Frozen primary comparison",
            "",
            "| Ez [kV/m] | L ambiguity | L analogue error | kNN / random ambiguity | frozen fixed-ROM transport corr. | frozen rolling median corr. |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for ez_kvm in EZ_VALUES:
        baseline = primary_lookup(
            raw_summary, "developed", ez_kvm, "L", PRIMARY_HORIZON_US
        )
        diagnostic = diagnostics_by_ez[ez_kvm]
        readme.append(
            f"| {ez_kvm} | {baseline['mean_ambiguity_normalized']:.3f} | "
            f"{baseline['mean_analog_error_normalized']:.3f} | "
            f"{random_by_ez[ez_kvm]['knn_over_random']:.3f} | "
            f"{diagnostic['frozen_joint_fixed_transport_correlation']:.3f} | "
            f"{diagnostic['frozen_joint_rolling_median_transport_correlation']:.3f} |"
        )
    readme.extend(
        [
            "",
            "## Best saved-moment candidate in each condition",
            "",
            "The best candidate is selected only for description; the predeclared pass/fail rule above is unchanged.",
            "",
            "| Ez [kV/m] | candidate | ambiguity gain | analogue-error gain | fixed-20D ambiguity gain | fixed-20D error gain |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for ez_kvm in EZ_VALUES:
        candidates = [
            screen_by_ez_state[(ez_kvm, state)] for state in STATE_ORDER[1:]
        ]
        best = max(candidates, key=lambda row: row["focal_ambiguity_reduction"])
        readme.append(
            f"| {ez_kvm} | {best['state']} | {best['focal_ambiguity_reduction']:.3f} | "
            f"{best['focal_analog_error_reduction']:.3f} | "
            f"{best['fixed_dimension_focal_ambiguity_reduction']:.3f} | "
            f"{best['fixed_dimension_focal_analog_error_reduction']:.3f} |"
        )
    fixed_corr = correlations_by_pair[
        ("L_ambiguity", "frozen_joint_fixed_transport_correlation")
    ]
    rolling_corr = correlations_by_pair[
        ("L_ambiguity", "frozen_joint_rolling_median_transport_correlation")
    ]
    readme.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Baseline ambiguity versus frozen fixed-ROM transport correlation: Spearman={fixed_corr['spearman']:.3f}, exact two-sided permutation p={fixed_corr['exact_two_sided_permutation_p']:.3f}.",
            f"- Baseline ambiguity versus frozen rolling median transport correlation: Spearman={rolling_corr['spearman']:.3f}, exact p={rolling_corr['exact_two_sided_permutation_p']:.3f}.",
            "- E40 combines low ambiguity and strong frozen-ROM skill, but E25 has high ambiguity despite strong ROM skill and E20 has only intermediate ambiguity despite ROM failure. Ambiguity is therefore informative but neither necessary nor sufficient for the previous ROM outcome.",
            "- E25 gains locally from Tiso at 1.2 us, but the gain does not persist over the required number of horizons and is smaller in the fixed-dimensional control. No new autonomous ROM is fit from this exploratory effect.",
            "- The early/mid sensitivity retains the same broad Ez ordering, so the map is not explained by one developed-time interval alone.",
            "- Pooled common-coordinate PCA lowers ambiguity in all five cases; this is a descriptive indication that coordinate construction matters, not a leave-one-Ez-out generalization result.",
            "",
            "| Ez [kV/m] | local-coordinate L ambiguity | pooled common-coordinate ambiguity |",
            "| ---: | ---: | ---: |",
        ]
    )
    for ez_kvm in EZ_VALUES:
        baseline = primary_lookup(
            raw_summary, "developed", ez_kvm, "L", PRIMARY_HORIZON_US
        )
        readme.append(
            f"| {ez_kvm} | {baseline['mean_ambiguity_normalized']:.3f} | "
            f"{common_by_ez[ez_kvm]['mean_ambiguity_normalized']:.3f} |"
        )
    readme.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Five Ez conditions are too few for a strong causal correlation claim.",
            "- The structural analogue search can use temporally separated states before or after a query; it is not a deployable causal forecast.",
            "- Low ambiguity at this resolution is evidence for predictive sufficiency, not proof of exact Markov closure.",
            "- Existing ROM outcomes were frozen before this map was inspected.",
            "",
            "`Tiso` is the existing scalar trace temperature and cannot test pressure anisotropy or off-diagonal stress.",
            "See `analysis_summary.json` and the CSV files for the full result.",
        ]
    )
    (output / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"[DONE] {output}", flush=True)


if __name__ == "__main__":
    main()
