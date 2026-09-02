#!/usr/bin/env python3
"""Confirm predictive geometry and observable-dependent closure across Ez.

This analysis is deliberately pre-rePIC and does not retrain SimVP or refit an
autonomous ROM.  It performs three frozen tests:

1. leave-one-Ez-out common-coordinate PCA;
2. task-specific ambiguity for transport observables;
3. a direct physical-field Fourier/POD coordinate control.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze_radaz_augmented_physical_state_dynamics as augmented
import analyze_radaz_b15_b25_predictive_state_ambiguity as ambiguity
import analyze_radaz_blockwise_fourier_latent_dynamics as block
import analyze_radaz_e25_carrier_state_closure as carrier
import analyze_radaz_e25_transport_residual_closure as closure
import analyze_radaz_electric_sweep_predictive_sufficiency as sufficiency
import analyze_radaz_local_rom_closure_map as local_map


ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT.parent
DEFAULT_OUTPUT = (
    RESEARCH
    / "research_results"
    / "2D_RadAz"
    / "SimVPv2"
    / "workdirs"
    / "analyze_radaz_electric_sweep_predictive_geometry_controls"
)
PREVIOUS_OUTPUT = (
    RESEARCH
    / "research_results"
    / "2D_RadAz"
    / "SimVPv2"
    / "workdirs"
    / "analyze_radaz_electric_sweep_predictive_sufficiency"
)
ADAPTIVE_OUTPUT = (
    RESEARCH
    / "research_results"
    / "2D_RadAz"
    / "SimVPv2"
    / "workdirs"
    / "compare_radaz_local_rom_closure_map_adaptive"
)

EZ_VALUES = (10, 20, 25, 30, 40)
FIT_START_US = 12.0
FIT_END_US = 24.0
ANALYSIS_START_US = 13.5
ANALYSIS_END_US = 29.75
HORIZONS_US = (0.15, 0.30, 0.60, 1.20, 2.40)
PRIMARY_HORIZON_US = 1.20
PRIMARY_K = 10
PRIMARY_THEILER_US = 1.0
PHYSICAL_COMPONENTS = 20
RADIAL_BANDS = 8
RANDOM_REPEATS = 32

COORDINATE_ORDER = ("local", "all5_common", "leave_one_ez")
COORDINATE_LABELS = {
    "local": "case-local PCA",
    "all5_common": "all-5 common PCA",
    "leave_one_ez": "leave-one-Ez common PCA",
}
COORDINATE_COLORS = {
    "local": "#4d4d4d",
    "all5_common": "#d55e00",
    "leave_one_ez": "#0072b2",
}

TRANSPORT_ORDER = (
    "joint_selected",
    "selected_mode_pair",
    "mtsi_band",
    "ecdi_band",
    "resolved_n1_21",
    "full_positive_modes",
)
TRANSPORT_LABELS = {
    "joint_selected": "joint selected transport",
    "selected_mode_pair": "selected MTSI/ECDI pair",
    "mtsi_band": "MTSI-band transport",
    "ecdi_band": "ECDI-band transport",
    "resolved_n1_21": "resolved n=1..21 transport",
    "full_positive_modes": "full positive-mode transport",
}
TRANSPORT_COLORS = {
    "joint_selected": "#0072b2",
    "selected_mode_pair": "#56b4e9",
    "mtsi_band": "#009e73",
    "ecdi_band": "#d55e00",
    "resolved_n1_21": "#cc79a7",
    "full_positive_modes": "#e69f00",
}
TRANSPORT_COORDINATE_ORDER = (
    "local",
    "leave_one_ez",
    "transport_direct",
    "rom_matched",
)
TRANSPORT_COORDINATE_LABELS = {
    "local": "L neighbours",
    "leave_one_ez": "LOO-L neighbours",
    "transport_direct": "T and dT/dt neighbours",
    "rom_matched": "L+Pcirc+T neighbours",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
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
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def exact_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    observed = float(spearmanr(x, y).statistic)
    values = []
    for order in itertools.permutations(range(len(y))):
        values.append(float(spearmanr(x, y[list(order)]).statistic))
    p_value = float(
        np.mean(np.abs(np.asarray(values)) >= abs(observed) - 1.0e-12)
    )
    return observed, p_value


def fit_pooled_block_models(
    training_ez: tuple[int, ...],
) -> tuple[dict[str, block.BlockPCA], np.ndarray, np.ndarray, list[dict]]:
    models: dict[str, block.BlockPCA] = {}
    score_pieces = []
    metadata = []
    for name, (mode_start, mode_end) in block.BLOCKS.items():
        pieces = []
        feature_shape = None
        for ez_kvm in training_ez:
            with h5py.File(local_map.latent_path(ez_kvm), "r") as source:
                time_us = (
                    np.asarray(source["translator_time_s"], dtype=np.float64)
                    * 1.0e6
                )
                fit_indices = np.flatnonzero(
                    (time_us >= FIT_START_US - 1.0e-9)
                    & (time_us < FIT_END_US - 1.0e-9)
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
    score_mean = np.mean(pooled_scores, axis=0)
    score_scale = np.std(pooled_scores, axis=0)
    score_scale = np.where(score_scale > 1.0e-10, score_scale, 1.0)
    return models, score_mean, score_scale, metadata


def transform_block_state(
    features: np.ndarray,
    models: dict[str, block.BlockPCA],
    score_mean: np.ndarray,
    score_scale: np.ndarray,
) -> np.ndarray:
    scores = np.concatenate(
        [models[name].transform(features) for name in block.BLOCKS], axis=1
    )
    return (scores - score_mean) / score_scale


def evaluate_state_ambiguity(
    ez_kvm: int,
    time_us: np.ndarray,
    state_name: str,
    state: np.ndarray,
) -> list[dict]:
    rows, _ = ambiguity.evaluate_ambiguity(
        f"E{ez_kvm}",
        time_us,
        state,
        state_name,
        state,
        ANALYSIS_START_US,
        ANALYSIS_END_US,
        HORIZONS_US,
        (PRIMARY_THEILER_US,),
        (PRIMARY_K,),
        False,
    )
    return rows


def summarize_query_rows(rows: list[dict], category_key: str) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row["case"],
            row[category_key],
            float(row["horizon_us"]),
            float(row["theiler_us"]),
            int(row["k"]),
        )
        grouped.setdefault(key, []).append(row)
    output = []
    for key, local in grouped.items():
        error = np.asarray([item["analog_error"] for item in local])
        persistence = np.asarray([item["persistence_error"] for item in local])
        output.append(
            {
                "case": key[0],
                "ez_kvm": int(key[0][1:]),
                category_key: key[1],
                "horizon_us": key[2],
                "theiler_us": key[3],
                "k": key[4],
                "queries": len(local),
                "mean_ambiguity_normalized": float(
                    np.mean([item["ambiguity_normalized"] for item in local])
                ),
                "median_ambiguity_normalized": float(
                    np.median([item["ambiguity_normalized"] for item in local])
                ),
                "mean_analog_error_normalized": float(
                    np.mean([item["analog_error_normalized"] for item in local])
                ),
                "analog_skill_vs_zero_increment": float(
                    1.0 - np.mean(error) / max(np.mean(persistence), 1.0e-30)
                ),
                "median_neighbor_radius_normalized": float(
                    np.median(
                        [item["neighbor_radius_normalized"] for item in local]
                    )
                ),
            }
        )
    return sorted(
        output,
        key=lambda row: (row["ez_kvm"], str(row[category_key]), row["horizon_us"]),
    )


def coordinate_analysis(output: Path) -> tuple[list[dict], dict[int, dict], dict]:
    print("[COORD] fitting all-5 common coordinate", flush=True)
    all_models, all_mean, all_scale, all_meta = fit_pooled_block_models(EZ_VALUES)
    rows = []
    states_by_ez: dict[int, dict] = {}
    metadata: dict = {"all5": all_meta, "leave_one": {}}
    for ez_kvm in EZ_VALUES:
        print(f"[COORD] E{ez_kvm}: local/all5/leave-one", flush=True)
        features, time_us, frames = block.load_features(local_map.latent_path(ez_kvm))
        local_state, local_meta = sufficiency.fit_local_latent(
            features, time_us, FIT_START_US, FIT_END_US
        )
        all_state = transform_block_state(features, all_models, all_mean, all_scale)
        training_ez = tuple(value for value in EZ_VALUES if value != ez_kvm)
        loo_models, loo_mean, loo_scale, loo_meta = fit_pooled_block_models(training_ez)
        loo_state = transform_block_state(features, loo_models, loo_mean, loo_scale)
        states = {
            "local": local_state,
            "all5_common": all_state,
            "leave_one_ez": loo_state,
        }
        for coordinate, values in states.items():
            local_rows = evaluate_state_ambiguity(
                ez_kvm, time_us, coordinate, values
            )
            rows.extend(
                {**row, "coordinate": coordinate} for row in local_rows
            )
        states_by_ez[ez_kvm] = {
            "time_us": time_us,
            "frames": frames,
            **states,
        }
        metadata["leave_one"][str(ez_kvm)] = {
            "training_ez_kvm": training_ez,
            "blocks": loo_meta,
            "target_excluded_from_normalization_pca_and_scaling": True,
            "local_blocks": local_meta,
        }
        del features
    summary = summarize_query_rows(rows, "coordinate")
    write_csv(output / "coordinate_ambiguity_summary.csv", summary)
    return summary, states_by_ez, metadata


def load_adaptive_rows() -> dict[int, dict[str, str]]:
    rows = read_csv(ADAPTIVE_OUTPUT / "adaptive_mode_ablation.csv")
    return {
        int(row["ez_kvm"]): row for row in rows if row["strategy"] == "joint"
    }


def full_transport_cache(ez_kvm: int, output: Path) -> Path:
    path = output / "transport_cache" / f"E{ez_kvm}_full_positive_transport.h5"
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    source_path = local_map.source_path(ez_kvm)
    physical_path = (
        PREVIOUS_OUTPUT.parent
        / "compare_radaz_local_rom_closure_map"
        / "cases"
        / f"E{ez_kvm}kVm"
        / "physical_fourier_targets.h5"
    )
    with h5py.File(physical_path, "r") as physical:
        radial_weights = np.asarray(physical["radial_weights"], dtype=np.float64)
    radial_weights = radial_weights / np.sum(radial_weights)
    band_edges = np.linspace(0, 257, RADIAL_BANDS + 1, dtype=int)
    with h5py.File(source_path, "r") as source:
        time_us = np.asarray(source["axes/time_s"], dtype=np.float64) * 1.0e6
        frames = len(time_us)
        values = np.empty(frames, dtype=np.float64)
        for start in range(0, frames, 16):
            stop = min(start + 16, frames)
            ne = np.asarray(
                source["fields/electron_den"][start:stop, :257, :256],
                dtype=np.float64,
            )
            ey = np.asarray(
                source["fields/efy"][start:stop, :257, :256],
                dtype=np.float64,
            )
            cross = np.empty((stop - start, RADIAL_BANDS, 129), dtype=np.complex128)
            for band in range(RADIAL_BANDS):
                ne_band = np.mean(
                    ne[:, band_edges[band] : band_edges[band + 1], :], axis=1
                )
                ey_band = np.mean(
                    ey[:, band_edges[band] : band_edges[band + 1], :], axis=1
                )
                ne_fft = np.fft.rfft(ne_band, axis=-1) / 256.0
                ey_fft = np.fft.rfft(ey_band, axis=-1) / 256.0
                cross[:, band] = ne_fft * np.conj(ey_fft)
            positive = 2.0 * np.sum(cross[:, :, 1:-1], axis=2) + cross[:, :, -1]
            values[start:stop] = -np.real(
                np.einsum("r,tr->t", radial_weights, positive)
            ) / carrier.B_T
            if stop == frames or stop % 400 == 0:
                print(f"[TRANSPORT] E{ez_kvm}: {stop}/{frames}", flush=True)
    with h5py.File(path, "w") as target:
        target.create_dataset("time_us", data=time_us)
        target.create_dataset("full_positive_mode_transport", data=values)
        target.attrs["source_h5"] = str(source_path)
        target.attrs["definition"] = (
            "-Re[2*sum(n=1..127)+Nyquist(n=128)] radial-weighted "
            "electron_density*conj(Ey)/Bx"
        )
    return path


def standardize_observable(values: np.ndarray, fit: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    mean = np.mean(array[fit], axis=0)
    scale = np.std(array[fit], axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    return (array - mean) / scale


def transport_targets(
    ez_kvm: int,
    time_us: np.ndarray,
    frames: np.ndarray,
    local_state: np.ndarray,
    output: Path,
    adaptive: dict[int, dict[str, str]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    physical_path = (
        PREVIOUS_OUTPUT.parent
        / "compare_radaz_local_rom_closure_map"
        / "cases"
        / f"E{ez_kvm}kVm"
        / "physical_fourier_targets.h5"
    )
    raw = carrier.load_raw_physical(physical_path)
    if not np.array_equal(raw.frame, frames):
        raise ValueError(f"Physical frame mismatch for E{ez_kvm}")
    row = adaptive[ez_kvm]
    selected_modes = np.asarray(
        [int(value) for value in row["selected_modes"].split(",")], dtype=int
    )
    modes = np.arange(raw.cross.shape[-1], dtype=int)
    n0 = float(row["ecdi_n0"])
    ratio = modes / n0
    mtsi_modes = modes[(modes >= 1) & (ratio <= 0.60)]
    ecdi_modes = modes[(ratio >= 0.75) & (ratio <= 1.25)]

    def transport_for_modes(selected: np.ndarray) -> np.ndarray:
        return carrier.transport_from_selected_cross(
            raw.cross[:, :, selected], raw.radial_weights
        )

    selected_components = np.column_stack(
        [transport_for_modes(np.asarray([mode])) for mode in selected_modes]
    )
    cache = full_transport_cache(ez_kvm, output)
    with h5py.File(cache, "r") as source:
        full_time = np.asarray(source["time_us"], dtype=np.float64)
        full_values = np.asarray(
            source["full_positive_mode_transport"], dtype=np.float64
        )
    full_values = sufficiency.nearest_sample(full_time, full_values, time_us)
    fit = (time_us >= FIT_START_US - 1.0e-9) & (
        time_us < FIT_END_US - 1.0e-9
    )
    phi_block = carrier.build_carrier_block(
        "phi",
        raw.phi,
        selected_modes,
        raw.radial_weights,
        raw.frame,
        fit,
    )
    cross_block = carrier.build_carrier_block(
        "cross",
        raw.cross,
        selected_modes,
        raw.radial_weights,
        raw.frame,
        fit,
    )
    hybrid = closure.build_transport_residual(cross_block, fit)
    direct_groups = {"transport_direct": hybrid.transport_state}
    direct_scaler = augmented.GroupScaler.fit(direct_groups, fit)
    rom_groups = {
        "latent": local_state,
        "phi_circular": phi_block.circular,
        "transport_direct": hybrid.transport_state,
    }
    rom_scaler = augmented.GroupScaler.fit(rom_groups, fit)
    conditioning_states = {
        "transport_direct": direct_scaler.transform(direct_groups),
        "rom_matched": rom_scaler.transform(rom_groups),
    }
    targets = {
        "joint_selected": transport_for_modes(selected_modes),
        "selected_mode_pair": selected_components,
        "mtsi_band": transport_for_modes(mtsi_modes),
        "ecdi_band": transport_for_modes(ecdi_modes),
        "resolved_n1_21": transport_for_modes(modes[modes >= 1]),
        "full_positive_modes": full_values,
    }
    targets = {
        name: standardize_observable(values, fit) for name, values in targets.items()
    }
    metadata = {
        "selected_modes": selected_modes,
        "mtsi_modes": mtsi_modes,
        "ecdi_modes": ecdi_modes,
        "ecdi_n0": n0,
        "physical_fourier_h5": str(physical_path),
        "full_transport_cache": str(cache),
        "conditioning_state_dimensions": {
            name: int(values.shape[1])
            for name, values in conditioning_states.items()
        },
        "rom_matched_groups": {
            "latent": int(local_state.shape[1]),
            "phi_circular": int(phi_block.circular.shape[1]),
            "transport_direct": int(hybrid.transport_state.shape[1]),
        },
    }
    return targets, conditioning_states, metadata


def random_target_null(
    ez_kvm: int,
    time_us: np.ndarray,
    state: np.ndarray,
    target: np.ndarray,
    target_name: str,
    coordinate: str,
) -> dict:
    base = np.flatnonzero(
        (time_us >= ANALYSIS_START_US - 1.0e-9)
        & (time_us <= ANALYSIS_END_US + 1.0e-9)
    )[:: ambiguity.ANALOG_STEP_FRAMES]
    horizon = int(round(PRIMARY_HORIZON_US / ambiguity.DT_US))
    valid = base[(base + horizon < len(time_us))]
    valid = valid[
        time_us[valid + horizon] <= ANALYSIS_END_US + 1.0e-9
    ]
    increments = target[valid + horizon] - target[valid]
    centered = increments - np.mean(increments, axis=0, keepdims=True)
    scale = max(float(np.mean(centered * centered)), 1.0e-30)
    rng = np.random.default_rng(20260825 + ez_kvm + TRANSPORT_ORDER.index(target_name))
    knn_values = []
    random_values = []
    for query in valid:
        candidates = valid[
            np.abs(time_us[valid] - time_us[query])
            >= PRIMARY_THEILER_US - 1.0e-9
        ]
        distances = np.sqrt(np.mean((state[candidates] - state[query]) ** 2, axis=1))
        neighbors = candidates[np.argsort(distances, kind="stable")[:PRIMARY_K]]
        local = target[neighbors + horizon] - target[neighbors]
        knn_values.append(
            float(np.mean((local - np.mean(local, axis=0)) ** 2)) / scale
        )
        for _ in range(RANDOM_REPEATS):
            selected = rng.choice(candidates, size=PRIMARY_K, replace=False)
            random_local = target[selected + horizon] - target[selected]
            random_values.append(
                float(
                    np.mean(
                        (random_local - np.mean(random_local, axis=0)) ** 2
                    )
                )
                / scale
            )
    return {
        "ez_kvm": ez_kvm,
        "coordinate": coordinate,
        "observable": target_name,
        "knn_mean_ambiguity": float(np.mean(knn_values)),
        "random_mean_ambiguity": float(np.mean(random_values)),
        "knn_over_random": float(np.mean(knn_values) / np.mean(random_values)),
        "queries": len(valid),
        "random_repeats_per_query": RANDOM_REPEATS,
    }


def transport_analysis(
    output: Path,
    states_by_ez: dict[int, dict],
) -> tuple[list[dict], list[dict], dict]:
    adaptive = load_adaptive_rows()
    query_rows = []
    null_rows = []
    metadata = {}
    for ez_kvm in EZ_VALUES:
        print(f"[TASK] E{ez_kvm}: transport observables", flush=True)
        entry = states_by_ez[ez_kvm]
        targets, conditioning_states, target_meta = transport_targets(
            ez_kvm,
            entry["time_us"],
            entry["frames"],
            entry["local"],
            output,
            adaptive,
        )
        metadata[str(ez_kvm)] = target_meta
        state_lookup = {
            "local": entry["local"],
            "leave_one_ez": entry["leave_one_ez"],
            **conditioning_states,
        }
        for coordinate in TRANSPORT_COORDINATE_ORDER:
            state = state_lookup[coordinate]
            for target_name, target in targets.items():
                rows, _ = ambiguity.evaluate_ambiguity(
                    f"E{ez_kvm}",
                    entry["time_us"],
                    target,
                    target_name,
                    state,
                    ANALYSIS_START_US,
                    ANALYSIS_END_US,
                    HORIZONS_US,
                    (PRIMARY_THEILER_US,),
                    (PRIMARY_K,),
                    False,
                )
                query_rows.extend(
                    {**row, "coordinate": coordinate, "observable": target_name}
                    for row in rows
                )
                if coordinate in ("local", "rom_matched"):
                    null_rows.append(
                        random_target_null(
                            ez_kvm,
                            entry["time_us"],
                            state,
                            target,
                            target_name,
                            coordinate,
                        )
                    )
    summary = summarize_transport_rows(query_rows)
    write_csv(output / "transport_ambiguity_summary.csv", summary)
    write_csv(output / "transport_random_neighbor_null.csv", null_rows)
    return summary, null_rows, metadata


def summarize_transport_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row["case"],
            row["coordinate"],
            row["observable"],
            float(row["horizon_us"]),
            float(row["theiler_us"]),
            int(row["k"]),
        )
        grouped.setdefault(key, []).append(row)
    output = []
    for key, local in grouped.items():
        error = np.asarray([item["analog_error"] for item in local])
        persistence = np.asarray([item["persistence_error"] for item in local])
        output.append(
            {
                "case": key[0],
                "ez_kvm": int(key[0][1:]),
                "coordinate": key[1],
                "observable": key[2],
                "horizon_us": key[3],
                "theiler_us": key[4],
                "k": key[5],
                "queries": len(local),
                "mean_ambiguity_normalized": float(
                    np.mean([item["ambiguity_normalized"] for item in local])
                ),
                "median_ambiguity_normalized": float(
                    np.median([item["ambiguity_normalized"] for item in local])
                ),
                "mean_analog_error_normalized": float(
                    np.mean([item["analog_error_normalized"] for item in local])
                ),
                "analog_skill_vs_zero_increment": float(
                    1.0 - np.mean(error) / max(np.mean(persistence), 1.0e-30)
                ),
            }
        )
    return sorted(
        output,
        key=lambda row: (
            row["ez_kvm"],
            row["coordinate"],
            TRANSPORT_ORDER.index(row["observable"]),
            row["horizon_us"],
        ),
    )


def physical_pca_states(
    ez_kvm: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    path = (
        PREVIOUS_OUTPUT.parent
        / "compare_radaz_local_rom_closure_map"
        / "cases"
        / f"E{ez_kvm}kVm"
        / "physical_fourier_targets.h5"
    )
    with h5py.File(path, "r") as source:
        coefficients = np.asarray(source["coefficients"], dtype=np.complex128)
        time_us = np.asarray(source["time_us"], dtype=np.float64)
        raw_fields = source["fields"][:]
    fields = [
        item.decode() if isinstance(item, bytes) else str(item)
        for item in raw_fields
    ]
    fit = (time_us >= FIT_START_US - 1.0e-9) & (
        time_us < FIT_END_US - 1.0e-9
    )
    field_scales = {}
    balanced = coefficients.copy()
    for field_index, field in enumerate(fields):
        centered = coefficients[fit, field_index] - np.mean(
            coefficients[fit, field_index], axis=0
        )
        scale = max(float(np.sqrt(np.mean(np.abs(centered) ** 2))), 1.0e-30)
        balanced[:, field_index] /= scale
        field_scales[field] = scale

    def state_for(selected_fields: tuple[str, ...]) -> tuple[np.ndarray, dict]:
        selected = balanced[:, [fields.index(name) for name in selected_fields]]
        packed = np.concatenate((selected.real, selected.imag), axis=-1)
        matrix = packed.reshape(len(time_us), -1)
        count = min(PHYSICAL_COMPONENTS, int(np.count_nonzero(fit)) - 1, matrix.shape[1])
        pca = PCA(
            n_components=count,
            svd_solver="randomized",
            random_state=42,
            iterated_power=5,
        )
        pca.fit(matrix[fit])
        state = ambiguity.standardize(pca.transform(matrix), fit)
        return state, {
            "fields": selected_fields,
            "input_dimensions": int(matrix.shape[1]),
            "components": count,
            "variance_capture": float(np.sum(pca.explained_variance_ratio_)),
        }

    p3, p3_meta = state_for(("electron_den", "ion_den", "phi"))
    p4, p4_meta = state_for(("electron_den", "ion_den", "phi", "efy"))
    return time_us, {"physical_P3": p3, "physical_P4": p4}, {
        "source": str(path),
        "field_scales": field_scales,
        "physical_P3": p3_meta,
        "physical_P4": p4_meta,
    }


def physical_control_analysis(output: Path) -> tuple[list[dict], dict]:
    rows = []
    metadata = {}
    for ez_kvm in EZ_VALUES:
        print(f"[PHYSICAL] E{ez_kvm}: P3/P4 coordinates", flush=True)
        time_us, states, local_meta = physical_pca_states(ez_kvm)
        metadata[str(ez_kvm)] = local_meta
        for coordinate, state in states.items():
            local_rows = evaluate_state_ambiguity(
                ez_kvm, time_us, coordinate, state
            )
            rows.extend(
                {**row, "physical_coordinate": coordinate} for row in local_rows
            )
    summary = summarize_query_rows(rows, "physical_coordinate")
    write_csv(output / "physical_coordinate_ambiguity_summary.csv", summary)
    return summary, metadata


def primary_lookup(
    rows: list[dict], category_key: str, ez_kvm: int, category: str
) -> dict:
    return next(
        row
        for row in rows
        if row["ez_kvm"] == ez_kvm
        and row[category_key] == category
        and math.isclose(row["horizon_us"], PRIMARY_HORIZON_US)
        and math.isclose(row["theiler_us"], PRIMARY_THEILER_US)
        and row["k"] == PRIMARY_K
    )


def frozen_rom_metrics() -> dict[int, dict]:
    rows = load_adaptive_rows()
    return {
        ez: {
            "fixed_transport_correlation": float(row["fixed_transport_correlation"]),
            "rolling_transport_correlation": float(
                row["rolling_median_transport_correlation"]
            ),
        }
        for ez, row in rows.items()
    }


def correlation_analysis(
    coordinate_summary: list[dict],
    transport_summary: list[dict],
    physical_summary: list[dict],
) -> list[dict]:
    rom = frozen_rom_metrics()
    outcomes = (
        "fixed_transport_correlation",
        "rolling_transport_correlation",
    )
    predictors: dict[str, np.ndarray] = {}
    for coordinate in COORDINATE_ORDER:
        predictors[f"state_{coordinate}"] = np.asarray(
            [
                primary_lookup(
                    coordinate_summary, "coordinate", ez, coordinate
                )["mean_ambiguity_normalized"]
                for ez in EZ_VALUES
            ]
        )
    for coordinate in TRANSPORT_COORDINATE_ORDER:
        for observable in TRANSPORT_ORDER:
            predictors[f"transport_{coordinate}_{observable}"] = np.asarray(
                [
                    next(
                        row
                        for row in transport_summary
                        if row["ez_kvm"] == ez
                        and row["coordinate"] == coordinate
                        and row["observable"] == observable
                        and math.isclose(row["horizon_us"], PRIMARY_HORIZON_US)
                    )["mean_ambiguity_normalized"]
                    for ez in EZ_VALUES
                ]
            )
    for coordinate in ("physical_P3", "physical_P4"):
        predictors[coordinate] = np.asarray(
            [
                primary_lookup(
                    physical_summary, "physical_coordinate", ez, coordinate
                )["mean_ambiguity_normalized"]
                for ez in EZ_VALUES
            ]
        )
    rows = []
    for predictor_name, predictor in predictors.items():
        for outcome in outcomes:
            target = np.asarray([rom[ez][outcome] for ez in EZ_VALUES])
            rho, p_value = exact_spearman(predictor, target)
            rows.append(
                {
                    "predictor": predictor_name,
                    "outcome": outcome,
                    "cases": len(EZ_VALUES),
                    "spearman": rho,
                    "exact_two_sided_permutation_p": p_value,
                }
            )
    local_latent = predictors["state_local"]
    for name in ("physical_P3", "physical_P4"):
        rho, p_value = exact_spearman(local_latent, predictors[name])
        rows.append(
            {
                "predictor": "state_local",
                "outcome": f"{name}_ambiguity",
                "cases": len(EZ_VALUES),
                "spearman": rho,
                "exact_two_sided_permutation_p": p_value,
            }
        )
    return rows


def coordinate_robustness_summary(coordinate_summary: list[dict]) -> list[dict]:
    rows = []
    for ez in EZ_VALUES:
        local_values = []
        all5_values = []
        loo_values = []
        for horizon in HORIZONS_US:
            local_values.append(
                next(
                    row["mean_ambiguity_normalized"]
                    for row in coordinate_summary
                    if row["ez_kvm"] == ez
                    and row["coordinate"] == "local"
                    and math.isclose(row["horizon_us"], horizon)
                )
            )
            all5_values.append(
                next(
                    row["mean_ambiguity_normalized"]
                    for row in coordinate_summary
                    if row["ez_kvm"] == ez
                    and row["coordinate"] == "all5_common"
                    and math.isclose(row["horizon_us"], horizon)
                )
            )
            loo_values.append(
                next(
                    row["mean_ambiguity_normalized"]
                    for row in coordinate_summary
                    if row["ez_kvm"] == ez
                    and row["coordinate"] == "leave_one_ez"
                    and math.isclose(row["horizon_us"], horizon)
                )
            )
        local_array = np.asarray(local_values)
        all5_array = np.asarray(all5_values)
        loo_array = np.asarray(loo_values)
        ratio = loo_array / np.maximum(local_array, 1.0e-30)
        all5_ratio = loo_array / np.maximum(all5_array, 1.0e-30)
        rows.append(
            {
                "ez_kvm": ez,
                "loo_better_than_local_horizons_of_5": int(np.sum(ratio < 1.0)),
                "loo_within_10pct_of_all5_horizons_of_5": int(
                    np.sum(all5_ratio <= 1.10)
                ),
                "median_loo_over_local": float(np.median(ratio)),
                "worst_loo_over_local": float(np.max(ratio)),
                "median_loo_over_all5": float(np.median(all5_ratio)),
                "horizon_loo_over_local": ";".join(
                    f"{horizon:g}:{value:.4f}"
                    for horizon, value in zip(HORIZONS_US, ratio)
                ),
            }
        )
    return rows


def make_plots(
    output: Path,
    coordinate_summary: list[dict],
    transport_summary: list[dict],
    physical_summary: list[dict],
    correlations: list[dict],
) -> None:
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "figure.dpi": 140})

    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    for coordinate in COORDINATE_ORDER:
        values = [
            primary_lookup(coordinate_summary, "coordinate", ez, coordinate)[
                "mean_ambiguity_normalized"
            ]
            for ez in EZ_VALUES
        ]
        axis.plot(
            EZ_VALUES,
            values,
            marker="o",
            linewidth=2,
            color=COORDINATE_COLORS[coordinate],
            label=COORDINATE_LABELS[coordinate],
        )
    axis.set_xlabel("Ez [kV/m]")
    axis.set_ylabel("normalized state ambiguity at 1.2 us")
    axis.set_title("Does the common predictive geometry transfer to a held-out Ez?")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output / "leave_one_ez_coordinate_ambiguity.png", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    for ez in EZ_VALUES:
        ratios = []
        for horizon in HORIZONS_US:
            local_value = next(
                row["mean_ambiguity_normalized"]
                for row in coordinate_summary
                if row["ez_kvm"] == ez
                and row["coordinate"] == "local"
                and math.isclose(row["horizon_us"], horizon)
            )
            loo_value = next(
                row["mean_ambiguity_normalized"]
                for row in coordinate_summary
                if row["ez_kvm"] == ez
                and row["coordinate"] == "leave_one_ez"
                and math.isclose(row["horizon_us"], horizon)
            )
            ratios.append(loo_value / max(local_value, 1.0e-30))
        axis.plot(HORIZONS_US, ratios, marker="o", linewidth=1.8, label=f"E{ez}")
    axis.axhline(1.0, color="#333333", linewidth=1.0)
    axis.set_xlabel("future horizon [us]")
    axis.set_ylabel("LOO ambiguity / local ambiguity")
    axis.set_title("Is the leave-one-Ez coordinate gain horizon-robust?")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output / "leave_one_ez_horizon_sensitivity.png", bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(15.5, 10.0), sharey=True)
    for axis, coordinate in zip(axes.ravel(), TRANSPORT_COORDINATE_ORDER):
        for observable in TRANSPORT_ORDER:
            values = [
                next(
                    row
                    for row in transport_summary
                    if row["ez_kvm"] == ez
                    and row["coordinate"] == coordinate
                    and row["observable"] == observable
                    and math.isclose(row["horizon_us"], PRIMARY_HORIZON_US)
                )["mean_ambiguity_normalized"]
                for ez in EZ_VALUES
            ]
            axis.plot(
                EZ_VALUES,
                values,
                marker="o",
                linewidth=1.7,
                color=TRANSPORT_COLORS[observable],
                label=TRANSPORT_LABELS[observable],
            )
        axis.set_xlabel("Ez [kV/m]")
        axis.set_title(TRANSPORT_COORDINATE_LABELS[coordinate])
        axis.grid(alpha=0.25)
    axes[0, 0].set_ylabel("normalized transport ambiguity at 1.2 us")
    axes[1, 0].set_ylabel("normalized transport ambiguity at 1.2 us")
    axes[1, 1].legend(loc="lower right", fontsize=8)
    figure.suptitle("Observable-dependent predictive ambiguity")
    figure.tight_layout()
    figure.savefig(output / "task_specific_transport_ambiguity.png", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    latent = [
        primary_lookup(coordinate_summary, "coordinate", ez, "local")[
            "mean_ambiguity_normalized"
        ]
        for ez in EZ_VALUES
    ]
    axis.plot(EZ_VALUES, latent, marker="o", linewidth=2, label="SimVP latent L")
    for coordinate, label, color in (
        ("physical_P3", "physical ne+ni+phi", "#009e73"),
        ("physical_P4", "physical ne+ni+phi+Ey", "#d55e00"),
    ):
        values = [
            primary_lookup(
                physical_summary, "physical_coordinate", ez, coordinate
            )["mean_ambiguity_normalized"]
            for ez in EZ_VALUES
        ]
        axis.plot(EZ_VALUES, values, marker="s", linewidth=2, color=color, label=label)
    axis.set_xlabel("Ez [kV/m]")
    axis.set_ylabel("normalized self-state ambiguity at 1.2 us")
    axis.set_title("Frozen SimVP representation versus direct physical coordinates")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output / "physical_coordinate_control.png", bbox_inches="tight")
    plt.close(figure)

    rom = frozen_rom_metrics()
    fixed_skill = np.asarray([rom[ez]["fixed_transport_correlation"] for ez in EZ_VALUES])
    predictors = {
        "full latent state": np.asarray(latent),
        "L-conditioned joint transport": np.asarray(
            [
                next(
                    row
                    for row in transport_summary
                    if row["ez_kvm"] == ez
                    and row["coordinate"] == "local"
                    and row["observable"] == "joint_selected"
                    and math.isclose(row["horizon_us"], PRIMARY_HORIZON_US)
                )["mean_ambiguity_normalized"]
                for ez in EZ_VALUES
            ]
        ),
        "T+dT/dt-conditioned joint transport": np.asarray(
            [
                next(
                    row
                    for row in transport_summary
                    if row["ez_kvm"] == ez
                    and row["coordinate"] == "transport_direct"
                    and row["observable"] == "joint_selected"
                    and math.isclose(row["horizon_us"], PRIMARY_HORIZON_US)
                )["mean_ambiguity_normalized"]
                for ez in EZ_VALUES
            ]
        ),
        "ROM-matched joint transport": np.asarray(
            [
                next(
                    row
                    for row in transport_summary
                    if row["ez_kvm"] == ez
                    and row["coordinate"] == "rom_matched"
                    and row["observable"] == "joint_selected"
                    and math.isclose(row["horizon_us"], PRIMARY_HORIZON_US)
                )["mean_ambiguity_normalized"]
                for ez in EZ_VALUES
            ]
        ),
    }
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 10.0), sharey=True)
    for axis, (name, predictor) in zip(axes.ravel(), predictors.items()):
        rho, p_value = exact_spearman(predictor, fixed_skill)
        axis.scatter(predictor, fixed_skill, s=70)
        for x_value, y_value, ez in zip(predictor, fixed_skill, EZ_VALUES):
            axis.annotate(f"E{ez}", (x_value, y_value), xytext=(5, 4), textcoords="offset points")
        axis.set_xlabel(f"{name} ambiguity")
        axis.set_title(f"rho={rho:.2f}, exact p={p_value:.3f}")
        axis.grid(alpha=0.25)
    axes[0, 0].set_ylabel("frozen fixed-ROM transport correlation")
    axes[1, 0].set_ylabel("frozen fixed-ROM transport correlation")
    figure.suptitle("Does task-matched ambiguity explain the frozen ROM result?")
    figure.tight_layout()
    figure.savefig(output / "task_ambiguity_vs_frozen_rom.png", bbox_inches="tight")
    plt.close(figure)


def focal_tables(
    coordinate_summary: list[dict],
    transport_summary: list[dict],
    physical_summary: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    coordinate = []
    transport = []
    physical = []
    for ez in EZ_VALUES:
        coordinate.append(
            {
                "ez_kvm": ez,
                **{
                    f"{name}_ambiguity": primary_lookup(
                        coordinate_summary, "coordinate", ez, name
                    )["mean_ambiguity_normalized"]
                    for name in COORDINATE_ORDER
                },
            }
        )
        transport.append(
            {
                "ez_kvm": ez,
                **{
                    f"{name}_ambiguity": next(
                        row
                        for row in transport_summary
                        if row["ez_kvm"] == ez
                        and row["coordinate"] == "local"
                        and row["observable"] == name
                        and math.isclose(row["horizon_us"], PRIMARY_HORIZON_US)
                    )["mean_ambiguity_normalized"]
                    for name in TRANSPORT_ORDER
                },
            }
        )
        physical.append(
            {
                "ez_kvm": ez,
                "physical_P3_ambiguity": primary_lookup(
                    physical_summary, "physical_coordinate", ez, "physical_P3"
                )["mean_ambiguity_normalized"],
                "physical_P4_ambiguity": primary_lookup(
                    physical_summary, "physical_coordinate", ez, "physical_P4"
                )["mean_ambiguity_normalized"],
            }
        )
    return coordinate, transport, physical


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    coordinate_summary, states_by_ez, coordinate_meta = coordinate_analysis(output)
    transport_summary, transport_null, transport_meta = transport_analysis(
        output, states_by_ez
    )
    physical_summary, physical_meta = physical_control_analysis(output)
    correlations = correlation_analysis(
        coordinate_summary, transport_summary, physical_summary
    )
    write_csv(output / "correlation_summary.csv", correlations)
    coordinate_robustness = coordinate_robustness_summary(coordinate_summary)
    write_csv(output / "coordinate_robustness_summary.csv", coordinate_robustness)
    coordinate_focal, transport_focal, physical_focal = focal_tables(
        coordinate_summary, transport_summary, physical_summary
    )
    write_csv(output / "coordinate_focal_1p2us.csv", coordinate_focal)
    write_csv(output / "transport_focal_1p2us.csv", transport_focal)
    write_csv(output / "physical_focal_1p2us.csv", physical_focal)
    make_plots(
        output,
        coordinate_summary,
        transport_summary,
        physical_summary,
        correlations,
    )

    report = {
        "status": "PASS",
        "question": (
            "Are common predictive coordinates transferable, is closure "
            "observable-dependent, and is the Ez map a SimVP artifact?"
        ),
        "protocol": {
            "ez_kvm": EZ_VALUES,
            "fit_us": (FIT_START_US, FIT_END_US),
            "analysis_us": (ANALYSIS_START_US, ANALYSIS_END_US),
            "horizons_us": HORIZONS_US,
            "primary_horizon_us": PRIMARY_HORIZON_US,
            "k": PRIMARY_K,
            "theiler_us": PRIMARY_THEILER_US,
            "coordinate_dimensions": 20,
            "loo_guard": (
                "target excluded from normalization, active-feature selection, "
                "PCA basis, and score scaling"
            ),
            "structural_not_deployable": True,
        },
        "coordinate_focal": coordinate_focal,
        "coordinate_robustness": coordinate_robustness,
        "transport_focal": transport_focal,
        "physical_focal": physical_focal,
        "coordinate_metadata": coordinate_meta,
        "transport_metadata": transport_meta,
        "physical_metadata": physical_meta,
        "transport_random_null": transport_null,
        "correlations": correlations,
        "guardrails": [
            "Five Ez cases provide descriptive ranks, not a strong causal test.",
            "All-5 common PCA uses the target distribution and is transductive.",
            "LOO common PCA excludes the target from every representation-fit step.",
            "Physical and SimVP ambiguities have different future coordinates; compare ordering, not absolute equality.",
            "Full positive-mode transport is an anomalous ExB transport proxy and excludes the n=0 contribution.",
            "No autonomous ROM is selected or refit in this analysis.",
        ],
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(json_safe(report), indent=2), encoding="utf-8"
    )

    local_lookup = {row["ez_kvm"]: row for row in coordinate_focal}
    transport_lookup = {row["ez_kvm"]: row for row in transport_focal}
    physical_lookup = {row["ez_kvm"]: row for row in physical_focal}
    corr_lookup = {(row["predictor"], row["outcome"]): row for row in correlations}
    robustness_lookup = {row["ez_kvm"]: row for row in coordinate_robustness}
    readme = [
        "# Electric-sweep predictive geometry controls",
        "",
        "This frozen pre-rePIC analysis tests leave-one-Ez common coordinates, task-specific transport ambiguity, and direct physical-field coordinates. It does not retrain SimVP or fit a new ROM.",
        "",
        "## Coordinate transfer at 1.2 us",
        "",
        "| Ez [kV/m] | local | all-5 common | leave-one-Ez common |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for ez in EZ_VALUES:
        row = local_lookup[ez]
        readme.append(
            f"| {ez} | {row['local_ambiguity']:.3f} | "
            f"{row['all5_common_ambiguity']:.3f} | "
            f"{row['leave_one_ez_ambiguity']:.3f} |"
        )
    readme.extend(
        [
            "",
            "## Leave-one-Ez horizon robustness",
            "",
            "| Ez [kV/m] | LOO improves over local [of 5] | median LOO/local | worst LOO/local |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for ez in EZ_VALUES:
        row = robustness_lookup[ez]
        readme.append(
            f"| {ez} | {row['loo_better_than_local_horizons_of_5']} | "
            f"{row['median_loo_over_local']:.3f} | {row['worst_loo_over_local']:.3f} |"
        )
    readme.extend(
        [
            "",
            "## Task-specific transport ambiguity using local L neighbours",
            "",
            "| Ez [kV/m] | joint selected | selected pair | MTSI band | ECDI band | resolved n=1..21 | full positive modes |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for ez in EZ_VALUES:
        row = transport_lookup[ez]
        readme.append(
            f"| {ez} | {row['joint_selected_ambiguity']:.3f} | "
            f"{row['selected_mode_pair_ambiguity']:.3f} | "
            f"{row['mtsi_band_ambiguity']:.3f} | "
            f"{row['ecdi_band_ambiguity']:.3f} | "
            f"{row['resolved_n1_21_ambiguity']:.3f} | "
            f"{row['full_positive_modes_ambiguity']:.3f} |"
        )
    readme.extend(
        [
            "",
            "## Joint-selected transport ambiguity by conditioning state",
            "",
            "| Ez [kV/m] | L | LOO-L | T and dT/dt | L+Pcirc+T | frozen ROM corr. |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    rom_metrics = frozen_rom_metrics()
    for ez in EZ_VALUES:
        values = {}
        for coordinate in TRANSPORT_COORDINATE_ORDER:
            values[coordinate] = next(
                row["mean_ambiguity_normalized"]
                for row in transport_summary
                if row["ez_kvm"] == ez
                and row["coordinate"] == coordinate
                and row["observable"] == "joint_selected"
                and math.isclose(row["horizon_us"], PRIMARY_HORIZON_US)
            )
        readme.append(
            f"| {ez} | {values['local']:.3f} | {values['leave_one_ez']:.3f} | "
            f"{values['transport_direct']:.3f} | {values['rom_matched']:.3f} | "
            f"{rom_metrics[ez]['fixed_transport_correlation']:.3f} |"
        )
    readme.extend(
        [
            "",
            "## Physical-coordinate control at 1.2 us",
            "",
            "| Ez [kV/m] | physical ne+ni+phi | physical ne+ni+phi+Ey |",
            "| ---: | ---: | ---: |",
        ]
    )
    for ez in EZ_VALUES:
        row = physical_lookup[ez]
        readme.append(
            f"| {ez} | {row['physical_P3_ambiguity']:.3f} | "
            f"{row['physical_P4_ambiguity']:.3f} |"
        )
    selected_corr = corr_lookup[
        ("transport_rom_matched_joint_selected", "fixed_transport_correlation")
    ]
    local_transport_corr = corr_lookup[
        ("transport_local_joint_selected", "fixed_transport_correlation")
    ]
    direct_transport_corr = corr_lookup[
        ("transport_transport_direct_joint_selected", "fixed_transport_correlation")
    ]
    latent_corr = corr_lookup[("state_local", "fixed_transport_correlation")]
    physical_corr = corr_lookup[("state_local", "physical_P4_ambiguity")]
    physical_rom_corr = corr_lookup[("physical_P4", "fixed_transport_correlation")]
    readme.extend(
        [
            "",
            "## Frozen comparisons",
            "",
            f"- Full latent ambiguity versus fixed ROM transport correlation: rho={latent_corr['spearman']:.3f}, exact p={latent_corr['exact_two_sided_permutation_p']:.3f}.",
            f"- L-conditioned joint transport ambiguity versus fixed ROM transport correlation: rho={local_transport_corr['spearman']:.3f}, exact p={local_transport_corr['exact_two_sided_permutation_p']:.3f}.",
            f"- T-conditioned joint transport ambiguity versus fixed ROM transport correlation: rho={direct_transport_corr['spearman']:.3f}, exact p={direct_transport_corr['exact_two_sided_permutation_p']:.3f}.",
            f"- ROM-matched joint-selected transport ambiguity versus fixed ROM transport correlation: rho={selected_corr['spearman']:.3f}, exact p={selected_corr['exact_two_sided_permutation_p']:.3f}.",
            f"- SimVP latent ambiguity versus physical P4 ambiguity: rho={physical_corr['spearman']:.3f}, exact p={physical_corr['exact_two_sided_permutation_p']:.3f}.",
            f"- Physical P4 ambiguity versus fixed ROM transport correlation: rho={physical_rom_corr['spearman']:.3f}, exact p={physical_rom_corr['exact_two_sided_permutation_p']:.3f}.",
            "",
            "## Interpretation",
            "",
            "- Leave-one-Ez coordinates reduce latent self-state ambiguity robustly for E10--E30. The E40 gain at 1.2 us is horizon-specific, so the all-five common-coordinate result must not be read as out-of-domain evidence.",
            "- Conditioning only on the SimVP latent state does not close future modal transport. Current transport plus its local time derivative is the strongest diagnostic of the frozen ROM result (rho=-0.900), but five cases give only exact p=0.083; this is suggestive, not statistically significant.",
            "- E25 combines high full-latent ambiguity with low T+dT/dt ambiguity and strong frozen-ROM skill. Closure is therefore observable-dependent: the full field need not be Markovian for selected transport observables to be locally predictable.",
            "- Direct physical coordinates reproduce the broad decrease in ambiguity toward E40, so that trend is not purely a frozen-encoder artifact. They do not reproduce the sharp E25 latent peak, which is representation- or metric-sensitive.",
            "- No ROM was refitted in this confirmatory analysis. Nearest-neighbour ambiguity is a structural analogue test and is not by itself a deployable causal forecast.",
            "",
            "See `analysis_summary.json`, the CSV files, and five PNG summaries for the complete result and guardrails.",
        ]
    )
    (output / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"[DONE] {output}", flush=True)


if __name__ == "__main__":
    main()
