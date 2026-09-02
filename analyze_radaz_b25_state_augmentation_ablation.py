#!/usr/bin/env python3
"""Test whether causal physical coordinates restore B25 latent ROM closure.

The analysis reuses the frozen B20-trained encoder and the existing B25 PIC
diagnostics.  Every augmented coordinate is part of the autonomous ROM state:
holdout physics is used only as evaluation truth and is never injected during
rollout.  Bicoherence features are recomputed from trailing windows so their
timestamp is the window end, not its center.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze_radaz_hankel_havok as hankel


ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT.parent
ROLLING_DIR = (
    RESEARCH
    / "research_results"
    / "2D_RadAz"
    / "SimVPv2"
    / "workdirs"
    / "analyze_radaz_b25_rolling_window_rom_0to40us"
)
PHYSICS_DIR = (
    RESEARCH
    / "research_results"
    / "2D_RadAz"
    / "PEPAPIC"
    / "2D_Landmark"
    / "analysis_results"
    / "compare_magnetic_field_sweep_three_component_B25_B30mT_E10kVm"
    / "B25mT"
)
OUTPUT = (
    RESEARCH
    / "research_results"
    / "2D_RadAz"
    / "SimVPv2"
    / "workdirs"
    / "analyze_radaz_b25_state_augmentation_ablation_0to40us"
)

DT_US = 0.015
WINDOWS = (
    (20.0, 24.0, 30.0),
    (22.0, 26.0, 32.0),
    (24.0, 28.0, 34.0),
    (26.0, 30.0, 36.0),
    (28.0, 32.0, 38.0),
    (30.0, 34.0, 40.0),
)
DELAYS = (10, 20, 40, 80, 120, 160)
RANKS = (8, 15, 20, 30, 40)
COMPONENTS = ("long", "mtsi", "ecdi")
STATE_ORDER = (
    "L",
    "L+A",
    "L+P",
    "L+T",
    "L+B",
    "L+A+P",
    "L+A+P+T",
    "L+A+P+T+B",
)
STATE_GROUPS = {
    "L": (),
    "L+A": ("A",),
    "L+P": ("P",),
    "L+T": ("T",),
    "L+B": ("B",),
    "L+A+P": ("A", "P"),
    "L+A+P+T": ("A", "P", "T"),
    "L+A+P+T+B": ("A", "P", "T", "B"),
}


@dataclass
class NetworkState:
    end_time_us: np.ndarray
    vectors: np.ndarray
    summaries: np.ndarray
    triads: np.ndarray


@dataclass
class Factorization:
    delay: int
    state_dimensions: int
    delay_mean: np.ndarray
    right: np.ndarray
    centered: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--bico-width-us", type=float, default=1.50)
    parser.add_argument("--bico-step-us", type=float, default=0.30)
    parser.add_argument("--bico-max-mode", type=int, default=30)
    parser.add_argument("--network-pcs", type=int, default=6)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    known = set(fieldnames)
    for row in rows[1:]:
        for key in row:
            if key not in known:
                fieldnames.append(key)
                known.add(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_encoder() -> tuple[np.ndarray, np.ndarray]:
    path = ROLLING_DIR / "B25_latent_0to40us.h5"
    with h5py.File(path, "r") as source:
        time_us = np.asarray(source["encoder_time_s"], dtype=np.float64) * 1.0e6
        values = np.asarray(source["encoder_pooled"], dtype=np.float32)
    return time_us, values.reshape(len(values), -1).astype(np.float64)


def load_fourier() -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    path = ROLLING_DIR / "B25_physical_fourier_0to40us.h5"
    with h5py.File(path, "r") as source:
        time_us = np.asarray(source["time_us"], dtype=np.float64)
        channels = tuple(item.decode("utf-8") for item in source["channels"][:])
        raw = np.asarray(source["features"], dtype=np.float32)
        bands = len(source["radial_band_edges"]) - 1
    packed = raw.reshape(len(raw), len(channels), bands, -1)
    max_mode = (packed.shape[-1] - 1) // 2
    coefficient = np.zeros(
        (len(raw), len(channels), bands, max_mode + 1), dtype=np.complex64
    )
    coefficient[..., 0] = packed[..., 0]
    coefficient[..., 1:] = (
        packed[..., 1 : max_mode + 1]
        + 1j * packed[..., max_mode + 1 : 2 * max_mode + 1]
    )
    return time_us, channels, coefficient


def triad_indices(max_mode: int) -> np.ndarray:
    return np.asarray(
        [
            (left, right, left + right)
            for left in range(1, max_mode + 1)
            for right in range(left, max_mode + 1)
            if left + right <= max_mode
        ],
        dtype=np.int64,
    )


def spatial_bicoherence(coefficient: np.ndarray, triads: np.ndarray) -> np.ndarray:
    left, right, output = triads.T
    product = coefficient[..., left] * coefficient[..., right]
    numerator = np.abs(
        np.mean(product * np.conj(coefficient[..., output]), axis=(0, 1))
    ) ** 2
    denominator = np.mean(np.abs(product) ** 2, axis=(0, 1)) * np.mean(
        np.abs(coefficient[..., output]) ** 2, axis=(0, 1)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 1.0e-30,
    )


def normalized_entropy(values: np.ndarray) -> float:
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    probability = values / max(float(np.sum(values)), 1.0e-30)
    nonzero = probability > 0.0
    if len(probability) <= 1:
        return 0.0
    return float(
        -np.sum(probability[nonzero] * np.log(probability[nonzero]))
        / math.log(len(probability))
    )


def build_causal_network(
    width_us: float, step_us: float, max_mode: int
) -> NetworkState:
    time_us, channels, coefficient = load_fourier()
    triads = triad_indices(max_mode)
    selected_channels = (channels.index("phi"), channels.index("electron_den"))
    end_times = np.arange(20.0, 40.0 + 1.0e-9, step_us)
    vectors = []
    summaries = []
    for end in end_times:
        mask = (time_us > end - width_us - 1.0e-9) & (time_us <= end + 1.0e-9)
        channel_vectors = []
        channel_summaries = []
        for channel in selected_channels:
            local = coefficient[mask, channel, :, : max_mode + 1].astype(
                np.complex128
            )
            local -= np.mean(local, axis=0, keepdims=True)
            bicoherence = spatial_bicoherence(local, triads)
            bicoherence = np.nan_to_num(bicoherence, nan=0.0, posinf=0.0, neginf=0.0)
            entropy = normalized_entropy(bicoherence)
            probability = bicoherence / max(float(np.sum(bicoherence)), 1.0e-30)
            channel_vectors.append(bicoherence)
            channel_summaries.extend(
                (
                    entropy,
                    math.exp(entropy * math.log(len(bicoherence))),
                    float(np.sum(np.sort(probability)[-10:])),
                    float(np.max(bicoherence)),
                )
            )
        vectors.append(np.concatenate(channel_vectors))
        summaries.append(channel_summaries)
    return NetworkState(
        end_time_us=end_times,
        vectors=np.asarray(vectors, dtype=np.float64),
        summaries=np.asarray(summaries, dtype=np.float64),
        triads=triads,
    )


def interpolate_columns(
    source_time: np.ndarray, values: np.ndarray, target_time: np.ndarray
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.stack(
        [np.interp(target_time, source_time, values[:, column]) for column in range(values.shape[1])],
        axis=1,
    )


def load_physics_groups(target_time: np.ndarray) -> dict[str, np.ndarray]:
    rows = read_csv(PHYSICS_DIR / "three_component_time_series.csv")
    source_time = np.asarray([float(row["time_us"]) for row in rows])
    amplitude = np.stack(
        [
            np.asarray([float(row[f"{component}_ey_amplitude"]) for row in rows])
            for component in COMPONENTS
        ],
        axis=1,
    )
    phase = np.radians(
        np.stack(
            [
                np.asarray([float(row[f"{component}_cross_phase_deg"]) for row in rows])
                for component in COMPONENTS
            ],
            axis=1,
        )
    )
    transport = np.stack(
        [
            np.asarray([float(row[f"{component}_exb_transport"]) for row in rows])
            for component in COMPONENTS
        ],
        axis=1,
    )
    return {
        "A": interpolate_columns(source_time, np.log(np.maximum(amplitude, 1.0e-30)), target_time),
        "P": interpolate_columns(
            source_time,
            np.concatenate((np.cos(phase), np.sin(phase)), axis=1),
            target_time,
        ),
        "T_raw": interpolate_columns(source_time, transport, target_time),
    }


def latest_sample(
    source_time: np.ndarray, values: np.ndarray, target_time: np.ndarray
) -> np.ndarray:
    indices = np.searchsorted(source_time, target_time, side="right") - 1
    indices = np.clip(indices, 0, len(source_time) - 1)
    return values[indices]


def build_network_group(
    network: NetworkState,
    target_time: np.ndarray,
    fit_start: float,
    validation_start: float,
    components: int,
) -> tuple[np.ndarray, int, float]:
    fit = (network.end_time_us >= fit_start - 1.0e-9) & (
        network.end_time_us < validation_start - 1.0e-9
    )
    count = min(components, int(np.count_nonzero(fit)) - 1, network.vectors.shape[1])
    if count < 1:
        raise RuntimeError("Not enough causal bicoherence windows for network PCA")
    pca = PCA(n_components=count, svd_solver="randomized", random_state=42)
    pca.fit(network.vectors[fit])
    network_scores = pca.transform(network.vectors)
    combined = np.concatenate((network_scores, network.summaries), axis=1)
    return (
        latest_sample(network.end_time_us, combined, target_time),
        count,
        float(np.sum(pca.explained_variance_ratio_)),
    )


def transform_transport(values: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    scale = np.median(np.abs(values[fit_mask]), axis=0)
    fallback = np.mean(np.abs(values[fit_mask]), axis=0)
    scale = np.where(scale > 1.0e-30, scale, fallback)
    scale = np.where(scale > 1.0e-30, scale, 1.0)
    return np.arcsinh(values / scale)


def standardize(values: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    mean = np.mean(values[fit_mask], axis=0)
    scale = np.std(values[fit_mask], axis=0)
    scale = np.where(scale > 1.0e-10, scale, 1.0)
    return (values - mean) / scale


def latent_state(
    values: np.ndarray, fit_mask: np.ndarray
) -> tuple[np.ndarray, int, float]:
    maximum = min(40, int(np.count_nonzero(fit_mask)) - 1, values.shape[1])
    pca = PCA(n_components=maximum, svd_solver="randomized", random_state=42)
    scores = pca.fit_transform(values[fit_mask])
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n95 = int(np.searchsorted(cumulative, 0.95) + 1)
    count = min(n95, 20)
    all_scores = pca.transform(values)[:, :count]
    return (
        standardize(all_scores, fit_mask),
        n95,
        float(np.sum(pca.explained_variance_ratio_[:count])),
    )


def delay_factorization(
    states: np.ndarray, delay: int, maximum_rank: int
) -> Factorization:
    delay_vectors = hankel.make_delay_vectors(states, delay)
    delay_mean = np.mean(delay_vectors, axis=0)
    centered = delay_vectors - delay_mean
    count = min(maximum_rank, min(centered.shape) - 1)
    if count < 1:
        raise ValueError("Insufficient delay vectors")
    _, _, right = randomized_svd(
        centered,
        n_components=count,
        n_iter=5,
        random_state=42,
    )
    return Factorization(delay, states.shape[1], delay_mean, right, centered)


def model_from_factorization(
    factorization: Factorization, rank: int
) -> hankel.HankelModel:
    if rank > len(factorization.right):
        raise ValueError("rank exceeds cached randomized factorization")
    basis = factorization.right[:rank].T
    coordinates = factorization.centered @ basis
    left = coordinates[:-1].T
    right = coordinates[1:].T
    matrix = right @ np.linalg.pinv(left, rcond=1.0e-10)
    return hankel.HankelModel(
        delay=factorization.delay,
        rank=rank,
        state_dimensions=factorization.state_dimensions,
        delay_mean=factorization.delay_mean,
        basis=basis,
        matrix=matrix,
        eigenvalues=np.linalg.eigvals(matrix),
        singular_values=np.empty(0, dtype=np.float64),
    )


def radius_normalized_model(model: hankel.HankelModel) -> hankel.HankelModel:
    """Return a fit-only control with spectral radius at most one."""
    radius = float(np.max(np.abs(model.eigenvalues)))
    matrix = model.matrix / max(radius, 1.0)
    return hankel.HankelModel(
        delay=model.delay,
        rank=model.rank,
        state_dimensions=model.state_dimensions,
        delay_mean=model.delay_mean,
        basis=model.basis,
        matrix=matrix,
        eigenvalues=np.linalg.eigvals(matrix),
        singular_values=model.singular_values,
    )


def safe_correlation(truth: np.ndarray, prediction: np.ndarray) -> float:
    left = truth.ravel()
    right = prediction.ravel()
    finite = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(finite) < 3:
        return float("nan")
    if np.std(left[finite]) < 1.0e-12 or np.std(right[finite]) < 1.0e-12:
        return float("nan")
    return float(np.corrcoef(left[finite], right[finite])[0, 1])


def subset_metrics(
    truth: np.ndarray, prediction: np.ndarray, persistence: np.ndarray
) -> dict[str, float]:
    if not np.all(np.isfinite(prediction)):
        return {
            "mse": float("inf"),
            "rmse": float("inf"),
            "skill_persistence": float("-inf"),
            "skill_mean": float("-inf"),
            "correlation": safe_correlation(truth, prediction),
            "std_ratio": float("inf"),
        }
    mse = float(np.mean((prediction - truth) ** 2))
    persistence_mse = float(np.mean((persistence - truth) ** 2))
    mean_mse = float(np.mean(truth**2))
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "skill_persistence": float(1.0 - mse / max(persistence_mse, 1.0e-30)),
        "skill_mean": float(1.0 - mse / max(mean_mse, 1.0e-30)),
        "correlation": safe_correlation(truth, prediction),
        "std_ratio": float(np.std(prediction) / max(np.std(truth), 1.0e-30)),
    }


def select_hyperparameters(
    states: np.ndarray,
    time_us: np.ndarray,
    fit_start: float,
    fit_end: float,
    latent_dimensions: int,
) -> tuple[dict, dict, list[dict]]:
    validation_start = fit_end - 1.0
    subtrain_mask = (time_us >= fit_start) & (time_us < validation_start)
    validation_mask = (time_us >= validation_start) & (time_us < fit_end)
    subtrain = states[subtrain_mask]
    validation = states[validation_mask]
    persistence = np.repeat(subtrain[-1][None, :], len(validation), axis=0)
    trials = []
    for delay in DELAYS:
        try:
            factorization = delay_factorization(subtrain, delay, max(RANKS))
        except (ValueError, np.linalg.LinAlgError):
            continue
        for rank in RANKS:
            try:
                model = model_from_factorization(factorization, rank)
                prediction = hankel.rollout_hankel(model, subtrain, len(validation))
                metrics = subset_metrics(
                    validation[:, :latent_dimensions],
                    prediction[:, :latent_dimensions],
                    persistence[:, :latent_dimensions],
                )
                stabilized_prediction = hankel.rollout_hankel(
                    radius_normalized_model(model), subtrain, len(validation)
                )
                stabilized_metrics = subset_metrics(
                    validation[:, :latent_dimensions],
                    stabilized_prediction[:, :latent_dimensions],
                    persistence[:, :latent_dimensions],
                )
                radius = float(np.max(np.abs(model.eigenvalues)))
            except (ValueError, np.linalg.LinAlgError):
                metrics = {"mse": float("inf"), "skill_persistence": float("-inf"), "correlation": float("nan")}
                stabilized_metrics = metrics
                radius = float("nan")
            trials.append(
                {
                    "delay": delay,
                    "rank": rank,
                    "validation_latent_mse": metrics["mse"],
                    "validation_latent_skill_vs_persistence": metrics["skill_persistence"],
                    "validation_latent_correlation": metrics["correlation"],
                    "validation_stabilized_latent_mse": stabilized_metrics["mse"],
                    "validation_stabilized_latent_skill_vs_persistence": stabilized_metrics[
                        "skill_persistence"
                    ],
                    "validation_stabilized_latent_correlation": stabilized_metrics[
                        "correlation"
                    ],
                    "validation_spectral_radius": radius,
                }
            )
    finite = [row for row in trials if np.isfinite(row["validation_latent_mse"])]
    if not finite:
        raise RuntimeError("No finite Hankel DMD candidate")
    selected = min(
        finite,
        key=lambda row: (row["validation_latent_mse"], row["delay"], row["rank"]),
    )
    stable_finite = [
        row
        for row in trials
        if np.isfinite(row["validation_stabilized_latent_mse"])
    ]
    selected_stabilized = min(
        stable_finite,
        key=lambda row: (
            row["validation_stabilized_latent_mse"],
            row["delay"],
            row["rank"],
        ),
    )
    return selected, selected_stabilized, trials


def evaluate_window(
    time_us: np.ndarray,
    encoder: np.ndarray,
    physics: dict[str, np.ndarray],
    network: NetworkState,
    window: tuple[float, float, float],
    network_pcs: int,
) -> tuple[list[dict], dict[str, dict], list[dict]]:
    fit_start, fit_end, test_end = window
    validation_start = fit_end - 1.0
    fit_mask = (time_us >= fit_start) & (time_us < fit_end)
    test_mask = (time_us >= fit_end) & (time_us <= test_end + 1.0e-9)
    latent, latent_n95, latent_capture = latent_state(encoder, fit_mask)
    groups = {
        "A": standardize(physics["A"], fit_mask),
        "P": standardize(physics["P"], fit_mask),
        "T": standardize(transform_transport(physics["T_raw"], fit_mask), fit_mask),
    }
    network_values, network_count, network_capture = build_network_group(
        network, time_us, fit_start, validation_start, network_pcs
    )
    groups["B"] = standardize(network_values, fit_mask)

    metrics_rows = []
    predictions = {}
    trial_rows = []
    for state_name in STATE_ORDER:
        arrays = [latent]
        slices = {"L": slice(0, latent.shape[1])}
        offset = latent.shape[1]
        for group_name in STATE_GROUPS[state_name]:
            arrays.append(groups[group_name])
            slices[group_name] = slice(offset, offset + groups[group_name].shape[1])
            offset += groups[group_name].shape[1]
        states = np.concatenate(arrays, axis=1)
        selected, selected_stabilized, trials = select_hyperparameters(
            states, time_us, fit_start, fit_end, latent.shape[1]
        )
        for row in trials:
            trial_rows.append(
                {
                    "window": f"{fit_start:g}-{fit_end:g}_to_{test_end:g}us",
                    "state": state_name,
                    **row,
                    "selected": int(
                        row["delay"] == selected["delay"]
                        and row["rank"] == selected["rank"]
                    ),
                    "selected_stabilized": int(
                        row["delay"] == selected_stabilized["delay"]
                        and row["rank"] == selected_stabilized["rank"]
                    ),
                }
            )
        fit = states[fit_mask]
        truth = states[test_mask]
        persistence = np.repeat(fit[-1][None, :], len(truth), axis=0)
        factorization = delay_factorization(fit, int(selected["delay"]), int(selected["rank"]))
        model = model_from_factorization(factorization, int(selected["rank"]))
        prediction = hankel.rollout_hankel(model, fit, len(truth))
        stabilized_factorization = delay_factorization(
            fit,
            int(selected_stabilized["delay"]),
            int(selected_stabilized["rank"]),
        )
        stabilized_model = radius_normalized_model(
            model_from_factorization(
                stabilized_factorization, int(selected_stabilized["rank"])
            )
        )
        stabilized_prediction = hankel.rollout_hankel(
            stabilized_model, fit, len(truth)
        )
        latent_metrics = subset_metrics(
            truth[:, slices["L"]], prediction[:, slices["L"]], persistence[:, slices["L"]]
        )
        stabilized_latent_metrics = subset_metrics(
            truth[:, slices["L"]],
            stabilized_prediction[:, slices["L"]],
            persistence[:, slices["L"]],
        )
        joint_metrics = subset_metrics(truth, prediction, persistence)
        frame_rmse = np.sqrt(
            np.mean((prediction[:, slices["L"]] - truth[:, slices["L"]]) ** 2, axis=1)
        )
        contiguous = 0
        for value in np.isfinite(frame_rmse) & (frame_rmse < 1.0):
            if not value:
                break
            contiguous += 1
        result = {
            "window": f"{fit_start:g}-{fit_end:g}_to_{test_end:g}us",
            "fit_start_us": fit_start,
            "fit_end_us": fit_end,
            "test_end_us": test_end,
            "state": state_name,
            "state_dimensions": states.shape[1],
            "latent_dimensions": latent.shape[1],
            "latent_pcs95": latent_n95,
            "latent_variance_capture": latent_capture,
            "network_pcs": network_count,
            "network_variance_capture": network_capture,
            "delay": int(selected["delay"]),
            "rank": int(selected["rank"]),
            "validation_latent_mse": selected["validation_latent_mse"],
            "validation_latent_skill_vs_persistence": selected[
                "validation_latent_skill_vs_persistence"
            ],
            "validation_latent_correlation": selected["validation_latent_correlation"],
            "stabilized_delay": int(selected_stabilized["delay"]),
            "stabilized_rank": int(selected_stabilized["rank"]),
            "validation_stabilized_latent_mse": selected_stabilized[
                "validation_stabilized_latent_mse"
            ],
            "validation_stabilized_latent_skill_vs_persistence": selected_stabilized[
                "validation_stabilized_latent_skill_vs_persistence"
            ],
            "validation_stabilized_latent_correlation": selected_stabilized[
                "validation_stabilized_latent_correlation"
            ],
            "spectral_radius": float(np.max(np.abs(model.eigenvalues))),
            "latent_mse": latent_metrics["mse"],
            "latent_rmse": latent_metrics["rmse"],
            "latent_skill_vs_persistence": latent_metrics["skill_persistence"],
            "latent_skill_vs_training_mean": latent_metrics["skill_mean"],
            "latent_correlation": latent_metrics["correlation"],
            "latent_std_ratio": latent_metrics["std_ratio"],
            "stabilized_latent_skill_vs_persistence": stabilized_latent_metrics[
                "skill_persistence"
            ],
            "stabilized_latent_skill_vs_training_mean": stabilized_latent_metrics[
                "skill_mean"
            ],
            "stabilized_latent_correlation": stabilized_latent_metrics[
                "correlation"
            ],
            "stabilized_latent_std_ratio": stabilized_latent_metrics["std_ratio"],
            "joint_skill_vs_persistence": joint_metrics["skill_persistence"],
            "joint_correlation": joint_metrics["correlation"],
            "contiguous_latent_rmse_lt_1_us": contiguous * DT_US,
        }
        for group_name, group_slice in slices.items():
            if group_name == "L":
                continue
            group_metrics = subset_metrics(
                truth[:, group_slice], prediction[:, group_slice], persistence[:, group_slice]
            )
            result[f"{group_name}_skill_vs_persistence"] = group_metrics[
                "skill_persistence"
            ]
            result[f"{group_name}_correlation"] = group_metrics["correlation"]
        metrics_rows.append(result)
        predictions[state_name] = {
            "time_us": time_us[test_mask],
            "truth_latent": truth[:, slices["L"]],
            "prediction_latent": prediction[:, slices["L"]],
            "stabilized_prediction_latent": stabilized_prediction[:, slices["L"]],
            "persistence_latent": persistence[:, slices["L"]],
        }
        print(
            f"[ABLATE] {result['window']} {state_name:<13} "
            f"delay={result['delay']:>2} rank={result['rank']:>2} "
            f"skill={result['latent_skill_vs_persistence']:.3f} "
            f"corr={result['latent_correlation']:.3f}",
            flush=True,
        )
    metadata = {
        "window": f"{fit_start:g}-{fit_end:g}_to_{test_end:g}us",
        "network_pcs": network_count,
        "network_variance_capture": network_capture,
    }
    return metrics_rows, predictions, trial_rows


def finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    return float(np.mean(array[finite])) if np.any(finite) else float("nan")


def summarize(metrics: list[dict]) -> list[dict]:
    baseline = {row["window"]: row for row in metrics if row["state"] == "L"}
    rows = []
    for state in STATE_ORDER:
        selected = [row for row in metrics if row["state"] == state]
        improvements = [
            row["latent_skill_vs_persistence"]
            - baseline[row["window"]]["latent_skill_vs_persistence"]
            for row in selected
        ]
        rows.append(
            {
                "state": state,
                "windows": len(selected),
                "mean_latent_skill_vs_persistence": finite_mean(
                    [row["latent_skill_vs_persistence"] for row in selected]
                ),
                "median_latent_skill_vs_persistence": float(
                    np.nanmedian([row["latent_skill_vs_persistence"] for row in selected])
                ),
                "mean_latent_correlation": finite_mean(
                    [row["latent_correlation"] for row in selected]
                ),
                "median_stabilized_latent_skill_vs_persistence": float(
                    np.nanmedian(
                        [
                            row["stabilized_latent_skill_vs_persistence"]
                            for row in selected
                        ]
                    )
                ),
                "mean_stabilized_latent_correlation": finite_mean(
                    [row["stabilized_latent_correlation"] for row in selected]
                ),
                "mean_skill_improvement_over_L": finite_mean(improvements),
                "windows_improved_over_L": int(np.sum(np.asarray(improvements) > 0.0)),
                "windows_positive_skill": int(
                    np.sum(
                        np.asarray(
                            [row["latent_skill_vs_persistence"] for row in selected]
                        )
                        > 0.0
                    )
                ),
                "windows_positive_stabilized_skill": int(
                    np.sum(
                        np.asarray(
                            [
                                row["stabilized_latent_skill_vs_persistence"]
                                for row in selected
                            ]
                        )
                        > 0.0
                    )
                ),
                "windows_positive_skill_and_correlation": int(
                    np.sum(
                        [
                            row["latent_skill_vs_persistence"] > 0.0
                            and row["latent_correlation"] > 0.0
                            for row in selected
                        ]
                    )
                ),
                "early_window_skill": selected[0]["latent_skill_vs_persistence"],
                "late_window_skill": selected[-1]["latent_skill_vs_persistence"],
            }
        )
    return rows


def display_value(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    if value < -9.0:
        return "<-9"
    return f"{value:.2f}"


def plot_heatmaps(metrics: list[dict], path: Path) -> None:
    windows = [row["window"] for row in metrics if row["state"] == "L"]
    skill = np.asarray(
        [
            [
                next(
                    row["latent_skill_vs_persistence"]
                    for row in metrics
                    if row["state"] == state and row["window"] == window
                )
                for window in windows
            ]
            for state in STATE_ORDER
        ]
    )
    correlation = np.asarray(
        [
            [
                next(
                    row["latent_correlation"]
                    for row in metrics
                    if row["state"] == state and row["window"] == window
                )
                for window in windows
            ]
            for state in STATE_ORDER
        ]
    )
    stabilized_skill = np.asarray(
        [
            [
                next(
                    row["stabilized_latent_skill_vs_persistence"]
                    for row in metrics
                    if row["state"] == state and row["window"] == window
                )
                for window in windows
            ]
            for state in STATE_ORDER
        ]
    )
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), constrained_layout=True)
    for axis, values, title in (
        (axes[0], skill, "Latent skill versus persistence"),
        (
            axes[1],
            stabilized_skill,
            "Fit-only spectral-radius-normalized skill (diagnostic control)",
        ),
        (axes[2], correlation, "Latent trajectory correlation"),
    ):
        image = axis.imshow(np.clip(values, -1.0, 1.0), cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
        axis.set_yticks(range(len(STATE_ORDER)), STATE_ORDER)
        axis.set_xticks(range(len(windows)), [item.replace("_to_", " -> ").replace("us", "") for item in windows])
        axis.set_title(title)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                axis.text(column, row, display_value(values[row, column]), ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    axes[2].set_xlabel("Fit window -> autonomous holdout [us]")
    fig.suptitle("B25 causal state-augmentation ablation", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_summary(summary: list[dict], path: Path) -> None:
    states = [row["state"] for row in summary]
    x = np.arange(len(states))
    median_skill = np.clip(
        np.nan_to_num(
            [row["median_latent_skill_vs_persistence"] for row in summary],
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ),
        -1.0,
        1.0,
    )
    mean_correlation = np.clip(
        np.nan_to_num(
            [row["mean_latent_correlation"] for row in summary],
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ),
        -1.0,
        1.0,
    )
    stabilized_skill = np.clip(
        np.nan_to_num(
            [
                row["median_stabilized_latent_skill_vs_persistence"]
                for row in summary
            ],
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ),
        -1.0,
        1.0,
    )
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    axes[0].bar(x - 0.24, median_skill, width=0.24, label="median raw skill")
    axes[0].bar(x, stabilized_skill, width=0.24, label="median stabilized skill")
    axes[0].bar(x + 0.24, mean_correlation, width=0.24, label="mean correlation")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Summary across six windows")
    axes[0].set_xticks(x, states, rotation=25, ha="right")
    axes[0].legend(loc="lower right")
    axes[1].bar(
        x - 0.18,
        [row["windows_improved_over_L"] for row in summary],
        width=0.36,
        color="#2979a8",
        label="raw state improves over L",
    )
    axes[1].bar(
        x + 0.18,
        [row["windows_positive_stabilized_skill"] for row in summary],
        width=0.36,
        color="#e67e22",
        label="stabilized state has positive skill",
    )
    axes[1].set_ylim(0.0, 6.5)
    axes[1].set_ylabel("Rolling windows out of six")
    axes[1].legend(loc="lower right")
    axes[1].set_xticks(x, states, rotation=25, ha="right")
    axes[1].set_xlabel("Autonomous state")
    fig.suptitle(
        "Does a causal physical state restore B25 latent closure?\n"
        "Display range clipped to [-1, 1]; exact values are in CSV",
        fontsize=14,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_rollouts(all_predictions: dict, path: Path) -> None:
    selected_windows = ("20-24_to_30us", "24-28_to_34us", "30-34_to_40us")
    full_state = "L+A+P+T+B"
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    for axis, window in zip(axes, selected_windows):
        baseline = all_predictions[window]["L"]
        full = all_predictions[window][full_state]
        axis.plot(baseline["time_us"], baseline["truth_latent"][:, 0], color="black", linewidth=1.8, label="truth latent PC1")
        axis.plot(baseline["time_us"], baseline["prediction_latent"][:, 0], color="#d35400", linewidth=1.2, label="L")
        axis.plot(full["time_us"], full["prediction_latent"][:, 0], color="#1f77b4", linewidth=1.2, label=full_state)
        axis.plot(baseline["time_us"], baseline["stabilized_prediction_latent"][:, 0], color="#2e8b57", linewidth=1.1, linestyle=":", label="L stabilized")
        axis.plot(full["time_us"], full["stabilized_prediction_latent"][:, 0], color="#7b2cbf", linewidth=1.1, linestyle=":", label=f"{full_state} stabilized")
        axis.plot(baseline["time_us"], baseline["persistence_latent"][:, 0], color="#777777", linewidth=1.0, linestyle="--", label="persistence")
        axis.set_title(window.replace("_to_", " -> ").replace("us", " us"))
        axis.set_ylabel("standardized PC1")
        axis.legend(loc="lower right", ncol=3, fontsize=8)
    axes[-1].set_xlabel("Time [us]")
    fig.suptitle("B25 autonomous latent rollout: baseline versus full causal augmentation", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_readme(output: Path, summary: list[dict], metrics: list[dict]) -> None:
    lines = [
        "# B25 causal state-augmentation ablation",
        "",
        "This analysis asks whether the B25 latent ROM fails because a physically",
        "important coordinate is missing, or because the local operator itself drifts.",
        "The B20-trained SimVP encoder is frozen. No PIC or SimVP retraining is used.",
        "",
        "## State definitions",
        "",
        "- `L`: encoder latent PCA state.",
        "- `A`: log Ey amplitudes for long-wave, MTSI, and ECDI candidate bands.",
        "- `P`: sine/cosine representation of the three density-Ey cross-phases.",
        "- `T`: asinh-scaled mode-resolved ExB transport proxies.",
        "- `B`: six causal PCA coordinates plus concentration summaries of the full",
        "  phi/electron-density spatial bicoherence networks.",
        "",
        "All added quantities are forecast as part of the autonomous state. Holdout",
        "truth is never injected. Bicoherence at time t uses only [t-1.5 us, t] and",
        "network PCA is fitted before the one-microsecond validation interval.",
        "Hyperparameters are selected by validation latent MSE, not joint-state MSE.",
        "The spectral-radius-normalized forecast is a numerical stability control",
        "whose delay/rank are also selected using validation only. It is not the primary",
        "ROM and does not prove physical closure.",
        "",
        "## Six-window summary",
        "",
        "| state | median skill | stabilized median | mean corr | improved windows | positive raw/stabilized | early skill | late skill |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {state} | {median_latent_skill_vs_persistence:.3f} | "
            "{median_stabilized_latent_skill_vs_persistence:.3f} | "
            "{mean_latent_correlation:.3f} | {windows_improved_over_L}/6 | "
            "{windows_positive_skill}/6 / {windows_positive_stabilized_skill}/6 | "
            "{early_window_skill:.3f} | {late_window_skill:.3f} |".format(**row)
        )
    full = next(row for row in summary if row["state"] == "L+A+P+T+B")
    baseline = next(row for row in summary if row["state"] == "L")
    focal = next(
        row
        for row in metrics
        if row["window"] == "26-30_to_36us" and row["state"] == "L+A+P"
    )
    long_delay_count = sum(int(row["delay"]) > 80 for row in metrics)
    stable_long_delay_count = sum(
        int(row["stabilized_delay"]) > 80 for row in metrics
    )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            f"The full augmentation changes median latent skill from {baseline['median_latent_skill_vs_persistence']:.3f} to {full['median_latent_skill_vs_persistence']:.3f} and improves {full['windows_improved_over_L']}/6 rolling windows.",
            "A gain in transport or bicoherence prediction alone is not counted as latent",
            "closure recovery. The primary endpoint is the future encoder-latent state.",
            f"Extending the Hankel memory search to 2.4 us selected a delay above 1.2 us in only {long_delay_count}/{len(metrics)} raw fits and {stable_long_delay_count}/{len(metrics)} stabilized fits; longer memory did not restore the early-window trajectory.",
            f"The selective `L+A+P` state did contain useful local information in 26-30 -> 36 us (skill {focal['latent_skill_vs_persistence']:.3f}, correlation {focal['latent_correlation']:.3f}), but the same state failed in earlier windows.",
            "The evidence therefore favors a time-dependent or switching local operator",
            "over one universally missing coordinate. The stability control removes many",
            "divergences, but its weak early correlations show that damping toward the",
            "mean is not equivalent to trajectory closure.",
            "These six holdouts overlap and are not six independent experiments; results",
            "are mechanistic/exploratory rather than a confirmatory significance test.",
            "",
            "## Outputs",
            "",
            "- `ablation_metrics.csv`: holdout metrics for every state and window.",
            "- `hyperparameter_trials.csv`: validation-only delay/rank search.",
            "- `ablation_summary.csv`: state-level aggregation.",
            "- `causal_bicoherence_network.h5`: trailing network state and triads.",
            "- `ablation_rollouts.h5`: latent truth and autonomous predictions.",
            "- `state_ablation_heatmaps.png`, `state_ablation_summary.png`,",
            "  `state_ablation_selected_rollouts.png`.",
        ]
    )
    output.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    time_us, encoder = load_encoder()
    physics = load_physics_groups(time_us)
    print("[PREP] causal bicoherence network", flush=True)
    network = build_causal_network(
        args.bico_width_us, args.bico_step_us, args.bico_max_mode
    )
    with h5py.File(args.output / "causal_bicoherence_network.h5", "w") as output:
        output.create_dataset("end_time_us", data=network.end_time_us)
        output.create_dataset("vectors", data=network.vectors, compression="gzip", compression_opts=4)
        output.create_dataset("summaries", data=network.summaries)
        output.create_dataset("triads", data=network.triads)
        output.attrs["window_width_us"] = args.bico_width_us
        output.attrs["step_us"] = args.bico_step_us
        output.attrs["channels"] = "phi,electron_den"

    metrics = []
    trials = []
    all_predictions = {}
    window_metadata = []
    for window in WINDOWS:
        rows, predictions, local_trials = evaluate_window(
            time_us, encoder, physics, network, window, args.network_pcs
        )
        metrics.extend(rows)
        trials.extend(local_trials)
        all_predictions[rows[0]["window"]] = predictions
        window_metadata.append(
            {
                "window": rows[0]["window"],
                "network_pcs": rows[0]["network_pcs"],
                "network_variance_capture": rows[0]["network_variance_capture"],
            }
        )
    summary = summarize(metrics)
    write_csv(args.output / "ablation_metrics.csv", metrics)
    write_csv(args.output / "hyperparameter_trials.csv", trials)
    write_csv(args.output / "ablation_summary.csv", summary)
    with h5py.File(args.output / "ablation_rollouts.h5", "w") as output:
        for window, states in all_predictions.items():
            for state, values in states.items():
                group = output.require_group(f"{window}/{state.replace('+', '_plus_')}")
                for key, value in values.items():
                    group.create_dataset(key, data=value, compression="gzip", compression_opts=4)
    plot_heatmaps(metrics, args.output / "state_ablation_heatmaps.png")
    plot_summary(summary, args.output / "state_ablation_summary.png")
    plot_rollouts(all_predictions, args.output / "state_ablation_selected_rollouts.png")
    write_readme(args.output, summary, metrics)
    payload = {
        "status": "PASS",
        "protocol": {
            "case": "B25mT_E10kVm",
            "windows": WINDOWS,
            "validation_us": 1.0,
            "holdout_us": 6.0,
            "delays": DELAYS,
            "ranks": RANKS,
            "bicoherence_window_us": args.bico_width_us,
            "bicoherence_step_us": args.bico_step_us,
            "bicoherence_channels": ["phi", "electron_den"],
            "holdout_state_injection": False,
        },
        "window_metadata": window_metadata,
        "summary": summary,
        "metrics": metrics,
    }
    (args.output / "analysis_summary.json").write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"[PASS] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
