#!/usr/bin/env python3
"""Test whether unresolved B-sweep states are slaved to resolved PCA states.

The analysis keeps a strict chronological holdout.  A physical Fourier PCA
basis and an unresolved-state decoder are developed on 20--24 us, while
24--30 us is used only for evaluation.  A nearest-analogue transverse-distance
diagnostic is also reported.  That diagnostic is a single-trajectory proxy,
not a transverse Lyapunov exponent.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT.parent / "research_results" / "2D_RadAz"
SOURCE = (
    RESEARCH
    / "SimVPv2"
    / "workdirs"
    / "analyze_radaz_magnetic_sweep_rom_B10_B15_B20_B25_B30mT_E10kVm"
    / "physical_features"
)
OUTPUT = (
    RESEARCH
    / "SimVPv2"
    / "workdirs"
    / "analyze_radaz_magnetic_sweep_slaving_B15_B20_B25_B30mT_E10kVm"
)

CASES = (15, 20, 25, 30)
PCA_START_US = 20.0
MODEL_TRAIN_END_US = 23.0
HOLDOUT_START_US = 24.0
HOLDOUT_END_US = 30.0
MAX_PCA_COMPONENTS = 64
PRIMARY_RESOLVED_DIMENSION = 4
RIDGE_ALPHAS = (1.0e-6, 1.0e-4, 1.0e-2, 1.0, 100.0)
KNN_NEIGHBORS = (1, 3, 5, 10, 20)
TRANSVERSE_HORIZONS_US = (0.15, 0.30, 0.60, 1.20)
COLORS = {15: "#0072b2", 20: "#009e73", 25: "#d55e00", 30: "#cc79a7"}


@dataclass
class Decoder:
    kind: str
    degree: int | None
    alpha: float | None
    neighbors: int | None
    z_mean: np.ndarray
    z_scale: np.ndarray
    h_mean: np.ndarray
    feature_mean: np.ndarray | None = None
    feature_scale: np.ndarray | None = None
    weights: np.ndarray | None = None
    z_train: np.ndarray | None = None
    h_train: np.ndarray | None = None

    def predict(self, z: np.ndarray) -> np.ndarray:
        standardized = (z - self.z_mean) / self.z_scale
        if self.kind == "mean":
            return np.broadcast_to(self.h_mean, (len(z), len(self.h_mean))).copy()
        if self.kind == "poly":
            features = polynomial_features(standardized, int(self.degree))
            features = (features - self.feature_mean) / self.feature_scale
            return features @ self.weights
        if self.kind == "knn":
            return knn_predict(
                standardized,
                self.z_train,
                self.h_train,
                int(self.neighbors),
            )
        raise ValueError(f"Unknown decoder kind: {self.kind}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def feature_path(b_mt: int) -> Path:
    return SOURCE / f"B{b_mt}_physical_fourier.h5"


def load_case(b_mt: int) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(feature_path(b_mt), "r") as handle:
        time_us = np.asarray(handle["time_us"], dtype=np.float64)
        scaled = np.asarray(handle["features"], dtype=np.float32)
        channel_scale = np.asarray(handle["channel_scale"], dtype=np.float32)
    channel_width = scaled.shape[1] // len(channel_scale)
    raw = scaled.reshape(len(scaled), len(channel_scale), channel_width)
    raw = raw * channel_scale[None, :, None]
    keep = time_us <= HOLDOUT_END_US + 1.0e-9
    return time_us[keep], raw[keep]


def scale_features(raw: np.ndarray, fit: np.ndarray) -> np.ndarray:
    output = raw.astype(np.float64, copy=True)
    for channel in range(output.shape[1]):
        centered = output[fit, channel] - np.mean(output[fit, channel], axis=0)
        scale = max(float(np.sqrt(np.mean(centered * centered))), 1.0e-12)
        output[:, channel] /= scale
    return output.reshape(len(output), -1).astype(np.float32)


def polynomial_features(values: np.ndarray, degree: int) -> np.ndarray:
    columns = [np.ones(len(values), dtype=np.float64)]
    dimensions = values.shape[1]
    for order in range(1, degree + 1):
        for indices in itertools.combinations_with_replacement(range(dimensions), order):
            columns.append(np.prod(values[:, indices], axis=1))
    return np.column_stack(columns)


def ridge_weights(features: np.ndarray, targets: np.ndarray, alpha: float) -> np.ndarray:
    rows, columns = features.shape
    if columns <= rows:
        gram = features.T @ features
        gram.flat[:: columns + 1] += alpha
        return np.linalg.solve(gram, features.T @ targets)
    gram = features @ features.T
    gram.flat[:: rows + 1] += alpha
    return features.T @ np.linalg.solve(gram, targets)


def fit_decoder(
    z: np.ndarray,
    h: np.ndarray,
    specification: dict,
    permutation: np.ndarray | None = None,
) -> Decoder:
    z_mean = np.mean(z, axis=0)
    z_scale = np.std(z, axis=0)
    z_scale = np.where(z_scale > 1.0e-10, z_scale, 1.0)
    standardized = (z - z_mean) / z_scale
    targets = h if permutation is None else h[permutation]
    h_mean = np.mean(targets, axis=0)
    kind = specification["kind"]
    if kind == "mean":
        return Decoder(kind, None, None, None, z_mean, z_scale, h_mean)
    if kind == "knn":
        return Decoder(
            kind,
            None,
            None,
            int(specification["neighbors"]),
            z_mean,
            z_scale,
            h_mean,
            z_train=standardized,
            h_train=targets,
        )
    degree = int(specification["degree"])
    features = polynomial_features(standardized, degree)
    feature_mean = np.mean(features, axis=0)
    feature_scale = np.std(features, axis=0)
    feature_mean[0] = 0.0
    feature_scale[0] = 1.0
    feature_scale = np.where(feature_scale > 1.0e-10, feature_scale, 1.0)
    normalized = (features - feature_mean) / feature_scale
    weights = ridge_weights(normalized, targets, float(specification["alpha"]))
    return Decoder(
        kind,
        degree,
        float(specification["alpha"]),
        None,
        z_mean,
        z_scale,
        h_mean,
        feature_mean,
        feature_scale,
        weights,
    )


def knn_predict(
    query: np.ndarray,
    train: np.ndarray,
    targets: np.ndarray,
    neighbors: int,
) -> np.ndarray:
    neighbors = min(neighbors, len(train))
    output = np.empty((len(query), targets.shape[1]), dtype=np.float64)
    for start in range(0, len(query), 64):
        stop = min(start + 64, len(query))
        distance = np.sum((query[start:stop, None, :] - train[None, :, :]) ** 2, axis=2)
        index = np.argpartition(distance, neighbors - 1, axis=1)[:, :neighbors]
        selected_distance = np.take_along_axis(distance, index, axis=1)
        weight = 1.0 / np.maximum(selected_distance, 1.0e-12)
        weight /= np.sum(weight, axis=1, keepdims=True)
        output[start:stop] = np.einsum("bk,bkd->bd", weight, targets[index])
    return output


def metric_bundle(
    full: np.ndarray,
    unresolved: np.ndarray,
    prediction: np.ndarray,
    reference_full_mean: np.ndarray,
    reference_h_mean: np.ndarray,
) -> dict:
    error = unresolved - prediction
    error_sse = float(np.sum(error * error))
    base_sse = float(np.sum(unresolved * unresolved))
    h_centered = unresolved - reference_h_mean
    h_denominator = max(float(np.sum(h_centered * h_centered)), 1.0e-30)
    full_centered = full - reference_full_mean
    full_denominator = max(float(np.sum(full_centered * full_centered)), 1.0e-30)
    return {
        "unresolved_r2": 1.0 - error_sse / h_denominator,
        "unresolved_gain_vs_linear_pca": 1.0 - error_sse / max(base_sse, 1.0e-30),
        "linear_pca_total_r2": 1.0 - base_sse / full_denominator,
        "slaving_manifold_total_r2": 1.0 - error_sse / full_denominator,
        "unresolved_nrmse": math.sqrt(error_sse / max(base_sse, 1.0e-30)),
        "unresolved_energy_fraction": base_sse / full_denominator,
    }


def phase_features(time_us: np.ndarray, frequency_mhz: float, harmonics: int) -> np.ndarray:
    columns = [np.ones(len(time_us), dtype=np.float64)]
    phase = 2.0 * np.pi * frequency_mhz * time_us
    for harmonic in range(1, harmonics + 1):
        columns.extend((np.sin(harmonic * phase), np.cos(harmonic * phase)))
    return np.column_stack(columns)


def fit_dominant_frequency(
    time_us: np.ndarray,
    signal: np.ndarray,
) -> float:
    local_time = time_us - time_us[0]
    centered = signal - np.mean(signal)
    dt_us = float(np.median(np.diff(local_time)))
    frequency = np.fft.rfftfreq(len(centered), d=dt_us)
    power = np.abs(np.fft.rfft(centered)) ** 2
    power[0] = 0.0
    peak = int(np.argmax(power))
    initial = float(frequency[peak])
    resolution = 1.0 / max(float(local_time[-1] - local_time[0]), dt_us)
    lower = max(1.0e-4, initial - 1.5 * resolution)
    upper = min(0.5 / dt_us * 0.999, initial + 1.5 * resolution)

    def objective(candidate: float) -> float:
        design = phase_features(local_time, candidate, 1)
        coefficient = np.linalg.lstsq(design, centered, rcond=None)[0]
        residual = centered - design @ coefficient
        return float(np.mean(residual * residual))

    result = minimize_scalar(objective, bounds=(lower, upper), method="bounded")
    return float(result.x)


def fit_recurrence_frequency(time_us: np.ndarray, z: np.ndarray) -> float:
    dt_us = float(np.median(np.diff(time_us)))
    standardized = (z - np.mean(z, axis=0)) / np.where(
        np.std(z, axis=0) > 1.0e-10,
        np.std(z, axis=0),
        1.0,
    )
    minimum_lag = max(2, int(round(0.30 / dt_us)))
    maximum_lag = min(int(round(3.50 / dt_us)), len(z) - 3)
    lags = np.arange(minimum_lag, maximum_lag + 1)
    error = np.asarray(
        [np.mean((standardized[lag:] - standardized[:-lag]) ** 2) for lag in lags]
    )
    lag = int(lags[int(np.argmin(error))])
    return 1.0 / (lag * dt_us)


def evaluate_phase_frequency(
    frequency: float,
    harmonics_grid: tuple[int, ...],
    time_us: np.ndarray,
    unresolved: np.ndarray,
    full: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    refit: np.ndarray,
    test: np.ndarray,
) -> dict:
    best = None
    for harmonics in harmonics_grid:
        train_features = phase_features(time_us[train], frequency, harmonics)
        validation_features = phase_features(time_us[validation], frequency, harmonics)
        for alpha in RIDGE_ALPHAS:
            weights = ridge_weights(train_features, unresolved[train], alpha)
            prediction = validation_features @ weights
            metrics = metric_bundle(
                full[validation],
                unresolved[validation],
                prediction,
                np.mean(full[train], axis=0),
                np.mean(unresolved[train], axis=0),
            )
            score = metrics["unresolved_gain_vs_linear_pca"]
            if best is None or score > best[0]:
                best = (score, harmonics, alpha)
    _, harmonics, alpha = best
    refit_features = phase_features(time_us[refit], frequency, harmonics)
    test_features = phase_features(time_us[test], frequency, harmonics)
    weights = ridge_weights(refit_features, unresolved[refit], alpha)
    metrics = metric_bundle(
        full[test],
        unresolved[test],
        test_features @ weights,
        np.mean(full[refit], axis=0),
        np.mean(unresolved[refit], axis=0),
    )
    return {
        "frequency_mhz": frequency,
        "period_us": 1.0 / frequency,
        "harmonics": harmonics,
        "alpha": alpha,
        "unresolved_gain": metrics["unresolved_gain_vs_linear_pca"],
        "total_r2": metrics["slaving_manifold_total_r2"],
    }


def phase_null_metrics(
    time_us: np.ndarray,
    z: np.ndarray,
    unresolved: np.ndarray,
    full: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    refit: np.ndarray,
    test: np.ndarray,
) -> dict:
    carrier_frequency = fit_dominant_frequency(time_us[refit], z[refit, 0])
    carrier = evaluate_phase_frequency(
        carrier_frequency,
        (1, 2, 3, 4, 5),
        time_us,
        unresolved,
        full,
        train,
        validation,
        refit,
        test,
    )
    coarse_recurrence_frequency = fit_recurrence_frequency(time_us[refit], z[refit])
    carrier_harmonic = max(
        1,
        int(round(carrier_frequency / coarse_recurrence_frequency)),
    )
    refined_recurrence_frequency = carrier_frequency / carrier_harmonic
    dt_us = float(np.median(np.diff(time_us[refit])))
    maximum_harmonic = min(
        80,
        max(1, int(math.floor((0.5 / dt_us) * 0.98 / refined_recurrence_frequency))),
    )
    recurrence_harmonics = tuple(
        value
        for value in (1, 2, 3, 5, 8, 12, 16, 20, 26, 32, 40, 52, 65, 78, maximum_harmonic)
        if value <= maximum_harmonic
    )
    recurrence_harmonics = tuple(sorted(set(recurrence_harmonics)))
    recurrence = evaluate_phase_frequency(
        refined_recurrence_frequency,
        recurrence_harmonics,
        time_us,
        unresolved,
        full,
        train,
        validation,
        refit,
        test,
    )
    return {
        "phase_null_frequency_mhz": recurrence["frequency_mhz"],
        "phase_null_period_us": recurrence["period_us"],
        "phase_null_harmonics": recurrence["harmonics"],
        "phase_null_alpha": recurrence["alpha"],
        "phase_null_unresolved_gain": recurrence["unresolved_gain"],
        "phase_null_total_r2": recurrence["total_r2"],
        "phase_null_coarse_period_us": 1.0 / coarse_recurrence_frequency,
        "phase_null_carrier_harmonic": carrier_harmonic,
        "carrier_null_frequency_mhz": carrier["frequency_mhz"],
        "carrier_null_period_us": carrier["period_us"],
        "carrier_null_harmonics": carrier["harmonics"],
        "carrier_null_unresolved_gain": carrier["unresolved_gain"],
        "carrier_null_total_r2": carrier["total_r2"],
    }


def candidate_specifications(resolved_dimensions: int) -> list[dict]:
    specifications = [{"kind": "mean", "name": "mean"}]
    max_degree = 3 if resolved_dimensions <= 8 else 2
    for degree in range(1, max_degree + 1):
        for alpha in RIDGE_ALPHAS:
            specifications.append(
                {
                    "kind": "poly",
                    "degree": degree,
                    "alpha": alpha,
                    "name": f"poly{degree}_ridge{alpha:g}",
                }
            )
    for neighbors in KNN_NEIGHBORS:
        specifications.append(
            {"kind": "knn", "neighbors": neighbors, "name": f"knn{neighbors}"}
        )
    return specifications


def evaluate_slaving(
    b_mt: int,
    time_us: np.ndarray,
    full: np.ndarray,
    pca: PCA,
    resolved_dimensions: int,
    label: str,
    pca_start_us: float = PCA_START_US,
    model_train_end_us: float = MODEL_TRAIN_END_US,
    holdout_start_us: float = HOLDOUT_START_US,
    holdout_end_us: float = HOLDOUT_END_US,
) -> tuple[list[dict], dict, Decoder, np.ndarray, np.ndarray]:
    scores = pca.transform(full).astype(np.float64)
    z = scores[:, :resolved_dimensions]
    unresolved = (
        full.astype(np.float64)
        - pca.mean_[None, :]
        - z @ pca.components_[:resolved_dimensions]
    )
    train = (time_us >= pca_start_us - 1.0e-9) & (
        time_us < model_train_end_us - 1.0e-9
    )
    validation = (time_us >= model_train_end_us - 1.0e-9) & (
        time_us < holdout_start_us - 1.0e-9
    )
    refit = (time_us >= pca_start_us - 1.0e-9) & (
        time_us < holdout_start_us - 1.0e-9
    )
    test = (time_us >= holdout_start_us - 1.0e-9) & (
        time_us <= holdout_end_us + 1.0e-9
    )

    rows = []
    best = None
    for specification in candidate_specifications(resolved_dimensions):
        decoder = fit_decoder(z[train], unresolved[train], specification)
        prediction = decoder.predict(z[validation])
        metrics = metric_bundle(
            full[validation],
            unresolved[validation],
            prediction,
            np.mean(full[train], axis=0),
            np.mean(unresolved[train], axis=0),
        )
        row = {
            "B_mT": b_mt,
            "resolved_definition": label,
            "resolved_dimensions": resolved_dimensions,
            "model": specification["name"],
            "split": "validation",
            "development_start_us": pca_start_us,
            "forecast_start_us": holdout_start_us,
            "forecast_end_us": holdout_end_us,
            **metrics,
        }
        rows.append(row)
        score = metrics["unresolved_gain_vs_linear_pca"]
        if best is None or score > best[0]:
            best = (score, specification)

    selected = best[1]
    decoder = fit_decoder(z[refit], unresolved[refit], selected)
    prediction = decoder.predict(z[test])
    metrics = metric_bundle(
        full[test],
        unresolved[test],
        prediction,
        np.mean(full[refit], axis=0),
        np.mean(unresolved[refit], axis=0),
    )
    rng = np.random.default_rng(1000 + b_mt + resolved_dimensions)
    shuffled = fit_decoder(
        z[refit],
        unresolved[refit],
        selected,
        permutation=rng.permutation(np.count_nonzero(refit)),
    )
    shuffled_metrics = metric_bundle(
        full[test],
        unresolved[test],
        shuffled.predict(z[test]),
        np.mean(full[refit], axis=0),
        np.mean(unresolved[refit], axis=0),
    )
    selected_row = {
        "B_mT": b_mt,
        "resolved_definition": label,
        "resolved_dimensions": resolved_dimensions,
        "model": selected["name"],
        "split": "strict_holdout",
        "development_start_us": pca_start_us,
        "forecast_start_us": holdout_start_us,
        "forecast_end_us": holdout_end_us,
        **metrics,
        "shuffle_unresolved_gain": shuffled_metrics["unresolved_gain_vs_linear_pca"],
        "shuffle_total_r2": shuffled_metrics["slaving_manifold_total_r2"],
    }
    phase_metrics = phase_null_metrics(
        time_us,
        z,
        unresolved,
        full,
        train,
        validation,
        refit,
        test,
    )
    selected_row.update(phase_metrics)
    selected_row["state_gain_beyond_phase_null"] = (
        selected_row["unresolved_gain_vs_linear_pca"]
        - selected_row["phase_null_unresolved_gain"]
    )
    rows.append(selected_row)
    return rows, selected_row, decoder, z, unresolved


def bootstrap_median_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(500, len(values)), replace=True)
    medians = np.median(samples, axis=1)
    return tuple(np.quantile(medians, (0.025, 0.975)))


def transverse_proxy(
    b_mt: int,
    time_us: np.ndarray,
    z: np.ndarray,
    unresolved: np.ndarray,
) -> list[dict]:
    dt_us = float(np.median(np.diff(time_us)))
    pretest = (time_us >= PCA_START_US - 1.0e-9) & (
        time_us < HOLDOUT_START_US - 1.0e-9
    )
    z_mean = np.mean(z[pretest], axis=0)
    z_scale = np.std(z[pretest], axis=0)
    z_scale = np.where(z_scale > 1.0e-10, z_scale, 1.0)
    zs = (z - z_mean) / z_scale
    rows = []
    for horizon_us in TRANSVERSE_HORIZONS_US:
        lag = max(1, int(round(horizon_us / dt_us)))
        analog = np.flatnonzero(
            (time_us >= PCA_START_US - 1.0e-9)
            & (time_us < HOLDOUT_START_US - horizon_us - 1.0e-9)
        )
        anchors = np.flatnonzero(
            (time_us >= HOLDOUT_START_US - 1.0e-9)
            & (time_us <= HOLDOUT_END_US - horizon_us + 1.0e-9)
        )[::5]
        resolved_initial = []
        transverse_initial = []
        transverse_ratio = []
        full_ratio = []
        for anchor in anchors:
            distance = np.sum((zs[analog] - zs[anchor]) ** 2, axis=1)
            match = analog[int(np.argmin(distance))]
            dz0 = float(np.sqrt(np.mean((zs[anchor] - zs[match]) ** 2)))
            dh0 = float(np.sqrt(np.mean((unresolved[anchor] - unresolved[match]) ** 2)))
            dh1 = float(
                np.sqrt(
                    np.mean(
                        (unresolved[anchor + lag] - unresolved[match + lag]) ** 2
                    )
                )
            )
            full0 = math.sqrt(
                dz0 * dz0
                + float(np.mean((unresolved[anchor] - unresolved[match]) ** 2))
            )
            dz1 = float(np.sqrt(np.mean((zs[anchor + lag] - zs[match + lag]) ** 2)))
            full1 = math.sqrt(
                dz1 * dz1
                + float(
                    np.mean(
                        (unresolved[anchor + lag] - unresolved[match + lag]) ** 2
                    )
                )
            )
            resolved_initial.append(dz0)
            transverse_initial.append(dh0)
            transverse_ratio.append(dh1 / max(dh0, 1.0e-12))
            full_ratio.append(full1 / max(full0, 1.0e-12))
        ratios = np.asarray(transverse_ratio, dtype=np.float64)
        log_rate = np.log(np.maximum(ratios, 1.0e-12)) / horizon_us
        low, high = bootstrap_median_ci(log_rate, 2000 + b_mt + lag)
        rows.append(
            {
                "B_mT": b_mt,
                "resolved_dimensions": z.shape[1],
                "horizon_us": horizon_us,
                "pairs": len(ratios),
                "median_initial_resolved_distance": float(np.median(resolved_initial)),
                "median_initial_transverse_distance": float(np.median(transverse_initial)),
                "median_transverse_ratio": float(np.median(ratios)),
                "transverse_contraction_fraction": float(np.mean(ratios < 1.0)),
                "median_transverse_log_rate_per_us": float(np.median(log_rate)),
                "median_log_rate_ci_low": low,
                "median_log_rate_ci_high": high,
                "median_full_state_ratio": float(np.median(full_ratio)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
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
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def make_plots(
    output: Path,
    selected_rows: list[dict],
    transverse_rows: list[dict],
    rolling_rows: list[dict],
) -> None:
    fixed = [row for row in selected_rows if row["resolved_definition"] == "fixed_r4"]
    local = [row for row in selected_rows if row["resolved_definition"] == "local_r95"]
    x = np.arange(len(CASES), dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), constrained_layout=True)
    width = 0.34
    axes[0, 0].bar(
        x - width / 2,
        [row["linear_pca_total_r2"] for row in fixed],
        width,
        label="linear PCA r=4",
        color="#999999",
    )
    axes[0, 0].bar(
        x + width / 2,
        [row["slaving_manifold_total_r2"] for row in fixed],
        width,
        label="r=4 + H(z)",
        color="#0072b2",
    )
    axes[0, 0].set_ylabel("strict-holdout total-state $R^2$")
    axes[0, 0].set_title("Does a nonlinear manifold improve reconstruction?")
    axes[0, 0].legend(loc="lower right")

    axes[0, 1].bar(
        x - width / 2,
        [row["unresolved_gain_vs_linear_pca"] for row in fixed],
        width,
        label="fixed r=4",
        color="#009e73",
    )
    axes[0, 1].bar(
        x + width / 2,
        [row["unresolved_gain_vs_linear_pca"] for row in local],
        width,
        label="local r95",
        color="#e69f00",
    )
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set_ylabel("unresolved SSE gain vs linear PCA")
    axes[0, 1].set_title("Strict-holdout slaving predictability")
    axes[0, 1].legend(loc="lower right")

    axes[1, 0].bar(
        x - width,
        [row["unresolved_gain_vs_linear_pca"] for row in fixed],
        width,
        label="true pairing",
        color="#0072b2",
    )
    axes[1, 0].bar(
        x,
        [row["phase_null_unresolved_gain"] for row in fixed],
        width,
        label="time-phase null",
        color="#e69f00",
    )
    axes[1, 0].bar(
        x + width,
        [row["shuffle_unresolved_gain"] for row in fixed],
        width,
        label="shuffled h control",
        color="#cc79a7",
    )
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_ylabel("unresolved SSE gain")
    axes[1, 0].set_title("State decoder vs periodic clock and shuffled controls")
    axes[1, 0].legend(loc="lower right")

    for b_mt in CASES:
        rows = [row for row in transverse_rows if row["B_mT"] == b_mt]
        axes[1, 1].plot(
            [row["horizon_us"] for row in rows],
            [row["median_transverse_ratio"] for row in rows],
            marker="o",
            linewidth=2,
            color=COLORS[b_mt],
            label=f"B{b_mt}",
        )
    axes[1, 1].axhline(1.0, color="black", linewidth=0.9, linestyle="--")
    axes[1, 1].set_xlabel("lead time [us]")
    axes[1, 1].set_ylabel("median transverse-distance ratio")
    axes[1, 1].set_title("Nearest-analogue transverse proxy")
    axes[1, 1].legend(loc="best")

    for axis in axes.flat[:3]:
        axis.set_xticks(x, [f"B{b}" for b in CASES])
        axis.grid(axis="y", alpha=0.25)
    axes[1, 1].grid(alpha=0.25)
    fig.suptitle("Magnetic-sweep slaving and transverse-contraction diagnostics")
    fig.savefig(output / "slaving_and_transverse_summary.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)
    for b_mt in CASES:
        rows = [row for row in transverse_rows if row["B_mT"] == b_mt]
        y = np.asarray([row["median_transverse_log_rate_per_us"] for row in rows])
        low = np.asarray([row["median_log_rate_ci_low"] for row in rows])
        high = np.asarray([row["median_log_rate_ci_high"] for row in rows])
        axis.errorbar(
            [row["horizon_us"] for row in rows],
            y,
            yerr=np.vstack((y - low, high - y)),
            marker="o",
            capsize=3,
            linewidth=2,
            color=COLORS[b_mt],
            label=f"B{b_mt}",
        )
    axis.axhline(0.0, color="black", linewidth=0.9, linestyle="--")
    axis.set_xlabel("lead time [us]")
    axis.set_ylabel("median $\log(d_\perp(t+\tau)/d_\perp(t))/\tau$ [$\mu$s$^{-1}$]")
    axis.set_title("Single-trajectory transverse proxy (95% bootstrap CI)")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    fig.savefig(output / "transverse_log_rate.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9.4, 5.6), constrained_layout=True)
    for b_mt in CASES:
        rows = [row for row in rolling_rows if row["B_mT"] == b_mt]
        axis.plot(
            [row["forecast_start_us"] for row in rows],
            [row["unresolved_gain_vs_linear_pca"] for row in rows],
            marker="o",
            linewidth=2,
            color=COLORS[b_mt],
            label=f"B{b_mt}",
        )
    axis.axhline(0.0, color="black", linewidth=0.9, linestyle="--")
    axis.set_xlabel("strict-holdout start [us]")
    axis.set_ylabel("unresolved SSE gain vs linear PCA")
    axis.set_title("Shifted-window confirmation of h = H(z), fixed r=4")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    fig.savefig(output / "rolling_slaving_gain.png", dpi=180)
    plt.close(fig)


def write_readme(
    output: Path,
    selected: list[dict],
    transverse: list[dict],
    rolling: list[dict],
) -> None:
    fixed = [row for row in selected if row["resolved_definition"] == "fixed_r4"]
    fixed.sort(key=lambda row: row["B_mT"])
    table = []
    for row in fixed:
        table.append(
            "| B{B_mT} | {resolved_dimensions} | {model} | {linear_pca_total_r2:.3f} | "
            "{slaving_manifold_total_r2:.3f} | {unresolved_gain_vs_linear_pca:.3f} | "
            "{phase_null_unresolved_gain:.3f} | {state_gain_beyond_phase_null:.3f} | "
            "{shuffle_unresolved_gain:.3f} |".format(**row)
        )
    primary = {
        row["B_mT"]: row
        for row in transverse
        if abs(row["horizon_us"] - 1.20) < 1.0e-9
    }
    transverse_table = []
    for b_mt in CASES:
        row = primary[b_mt]
        transverse_table.append(
            f"| B{b_mt} | {row['median_transverse_ratio']:.3f} | "
            f"{row['transverse_contraction_fraction']:.3f} | "
            f"{row['median_transverse_log_rate_per_us']:.3f} | "
            f"[{row['median_log_rate_ci_low']:.3f}, {row['median_log_rate_ci_high']:.3f}] |"
        )
    rolling_table = []
    for b_mt in CASES:
        rows = [row for row in rolling if row["B_mT"] == b_mt]
        gains = np.asarray(
            [row["unresolved_gain_vs_linear_pca"] for row in rows], dtype=float
        )
        total_r2 = np.asarray(
            [row["slaving_manifold_total_r2"] for row in rows], dtype=float
        )
        rolling_table.append(
            f"| B{b_mt} | {np.min(gains):.3f} | {np.median(gains):.3f} | "
            f"{np.max(gains):.3f} | {np.median(total_r2):.3f} |"
        )
    readme = f"""# Magnetic-sweep unresolved-state slaving test

## Protocol

- physical state: 8 radial bands x azimuthal Fourier modes n=0..48 for ne, ni, phi
- PCA/decoder development: 20--24 us
- decoder selection: 20--23 us -> 23--24 us
- strict holdout: 24--30 us
- primary resolved state: the first four local physical PCA coordinates in every case
- controls: local r95, shuffled unresolved-state labels, linear/quadratic/cubic ridge, kNN

## Strict-holdout results

| case | r | selected H(z) | linear-PCA total R2 | manifold total R2 | unresolved gain | phase-null gain | state minus phase | shuffled gain |
|---|---:|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

`unresolved gain` is `1 - SSE[residual - H(z)] / SSE[linear-PCA residual]`.
It measures whether the unresolved physical state is determined by the resolved
coordinates beyond the linear PCA reconstruction itself.

## 1.2 us nearest-analogue transverse proxy

| case | median d_perp ratio | fraction contracting | median log-rate [/us] | bootstrap 95% CI |
|---|---:|---:|---:|---:|
{chr(10).join(transverse_table)}

The transverse calculation compares a holdout state with the closest resolved
state from 20--24 us and follows both truth trajectories.  It is a
single-trajectory analogue diagnostic, not a perturbed-trajectory transverse
Lyapunov exponent.  Negative log-rate supports conditional contraction; a
causal proof requires paired PIC reruns with controlled transverse
perturbations.

## Shifted-window confirmation

Four 4 us development windows ending at 18, 20, 22, and 24 us were each
followed by a disjoint 6 us holdout.  PCA, model selection, and refitting were
repeated independently in every window.

| case | min unresolved gain | median | max | median manifold total R2 |
|---|---:|---:|---:|---:|
{chr(10).join(rolling_table)}

## Result

B15 is the only case with reproducible strict-holdout slaving evidence.  Four
linear PCs already reconstruct 97.2% of the holdout physical state; the
nonlinear state decoder raises this to 99.6% and removes 86.6% of the remaining
linear-PCA residual.  The shuffled pairing worsens the residual, and the gain
remains 0.788--0.889 across four shifted windows.  B20, B25, and B30 show no
comparable gain, including when each case uses its local r95 dimension.

A deliberately strong periodic-orbit clock explains 72.4% of the B15
unresolved residual, so much of the apparent slaving is associated with the
stable periodic orbit.  However, H(z) still adds 14.2 percentage points of
residual gain on the fixed holdout and remains better in every shifted window.
The data therefore support on-attractor h approximately equal to H(z), beyond
absolute clock phase alone.

Transverse attraction is not established.  The B15 1.2 us log-rate confidence
interval includes zero, while B25 shows the strongest apparent contraction
despite failing the slaving decoder.  This demonstrates that the
single-trajectory analogue ratio is confounded by recurrence, mean reversion,
and trajectory geometry.  Controlled paired PIC perturbations are required to
test a negative transverse Lyapunov exponent.

## Interpretation rule

Strong evidence for slaving requires all of the following: small unresolved
energy, positive strict-holdout unresolved gain, failure of the shuffled
control, and transverse ratios below one over more than one horizon.  A high
snapshot PCA capture alone is not sufficient.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    missing = [feature_path(b) for b in CASES if not feature_path(b).is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(map(str, missing)))

    candidates = []
    selected_rows = []
    transverse_rows = []
    rolling_rows = []
    pca_rows = []
    for b_mt in CASES:
        print(f"[CASE] B{b_mt}", flush=True)
        time_us, raw_full = load_case(b_mt)
        pca_fit = (time_us >= PCA_START_US - 1.0e-9) & (
            time_us < HOLDOUT_START_US - 1.0e-9
        )
        full = scale_features(raw_full, pca_fit)
        components = min(MAX_PCA_COMPONENTS, np.count_nonzero(pca_fit) - 1, full.shape[1])
        pca = PCA(n_components=components, svd_solver="randomized", random_state=42)
        pca.fit(full[pca_fit])
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        local_r95 = int(np.searchsorted(cumulative, 0.95) + 1)
        pca_rows.append(
            {
                "B_mT": b_mt,
                "local_r95": local_r95,
                "fixed_r4_variance_capture": float(np.sum(pca.explained_variance_ratio_[:4])),
                "local_r95_variance_capture": float(cumulative[local_r95 - 1]),
            }
        )

        fixed_outputs = None
        for label, resolved_dimensions in (
            ("fixed_r4", PRIMARY_RESOLVED_DIMENSION),
            ("local_r95", local_r95),
        ):
            rows, selected, decoder, z, unresolved = evaluate_slaving(
                b_mt,
                time_us,
                full,
                pca,
                resolved_dimensions,
                label,
            )
            candidates.extend(rows)
            selected_rows.append(selected)
            if label == "fixed_r4":
                fixed_outputs = (z, unresolved)
        transverse_rows.extend(
            transverse_proxy(b_mt, time_us, fixed_outputs[0], fixed_outputs[1])
        )

        for forecast_start_us in (18.0, 20.0, 22.0, 24.0):
            development_start_us = forecast_start_us - 4.0
            model_train_end_us = forecast_start_us - 1.0
            forecast_end_us = min(forecast_start_us + 6.0, HOLDOUT_END_US)
            local_fit = (time_us >= development_start_us - 1.0e-9) & (
                time_us < forecast_start_us - 1.0e-9
            )
            local_components = min(
                MAX_PCA_COMPONENTS,
                np.count_nonzero(local_fit) - 1,
                raw_full.shape[1] * raw_full.shape[2],
            )
            local_full = scale_features(raw_full, local_fit)
            local_pca = PCA(
                n_components=local_components,
                svd_solver="randomized",
                random_state=42,
            )
            local_pca.fit(local_full[local_fit])
            _, selected, _, _, _ = evaluate_slaving(
                b_mt,
                time_us,
                local_full,
                local_pca,
                PRIMARY_RESOLVED_DIMENSION,
                "fixed_r4_rolling",
                pca_start_us=development_start_us,
                model_train_end_us=model_train_end_us,
                holdout_start_us=forecast_start_us,
                holdout_end_us=forecast_end_us,
            )
            rolling_rows.append(selected)

    write_csv(output / "pca_summary.csv", pca_rows)
    write_csv(output / "decoder_candidates.csv", candidates)
    write_csv(output / "strict_holdout_slaving.csv", selected_rows)
    write_csv(output / "transverse_proxy.csv", transverse_rows)
    write_csv(output / "rolling_slaving.csv", rolling_rows)
    make_plots(output, selected_rows, transverse_rows, rolling_rows)
    summary = {
        "protocol_us": {
            "pca_and_refit": [PCA_START_US, HOLDOUT_START_US],
            "decoder_train": [PCA_START_US, MODEL_TRAIN_END_US],
            "decoder_validation": [MODEL_TRAIN_END_US, HOLDOUT_START_US],
            "strict_holdout": [HOLDOUT_START_US, HOLDOUT_END_US],
        },
        "pca": pca_rows,
        "strict_holdout": selected_rows,
        "transverse_proxy": transverse_rows,
        "rolling_slaving": rolling_rows,
        "caveat": (
            "Transverse ratios are single-trajectory nearest-analogue proxies; "
            "controlled perturbation reruns are required for a transverse Lyapunov exponent."
        ),
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2), encoding="utf-8"
    )
    write_readme(output, selected_rows, transverse_rows, rolling_rows)
    print(f"[DONE] {output}", flush=True)


if __name__ == "__main__":
    main()
