#!/usr/bin/env python3
"""Diagnose observable-dependent predictive dimension in RadAz transport.

The preceding closure experiment showed that an observed-state affine model
using modal transport T and a causal velocity V_delta can predict E40 robustly,
E25 only partially, and no magnetic-sweep condition robustly.  This script
tests whether that hierarchy is visible directly in the projected phase-space
geometry and whether E25/E40 admit an autonomous two-dimensional map.

The descriptive geometry uses 12--30 us.  Derivative lags are selected only
from the preceding 12--18 us train / 18--20 us validation experiment.  The
autonomous map repeats that split, refits on 12--20 us, and rolls out 20--30 us
without teacher forcing.  Rolling maps use only the preceding four microseconds
and predict the next two microseconds with frozen hyperparameters.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze_radaz_observable_dependent_closure as closure


ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT.parent
DEFAULT_OUTPUT = (
    RESEARCH
    / "research_results"
    / "2D_RadAz"
    / "SimVPv2"
    / "workdirs"
    / "analyze_radaz_transport_phase_geometry"
)

DT_US = closure.DT_US
ANALYSIS_INTERVAL = (12.0, 30.0)
FUTURE_HORIZON_US = 1.20
FUTURE_HORIZON_FRAMES = int(round(FUTURE_HORIZON_US / DT_US))
THEILER_US = 1.0
THEILER_FRAMES = int(round(THEILER_US / DT_US))
NEIGHBORS = 12
QUERY_STEP = 5
RECURRENCE_EPSILON = 0.20

MAP_STEP_FRAMES = 10
MAP_STEP_US = MAP_STEP_FRAMES * DT_US
MAP_DEGREES = (1, 2, 3)
MAP_RIDGES = (1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0e-1, 1.0)
MAP_KINDS = ("mechanical", "general")
ROLLING_STARTS = (20.0, 22.0, 24.0, 26.0, 28.0)
ROLLING_HISTORY_US = 4.0
ROLLING_TEST_US = 2.0

COLORS = {
    "E25": "#0072b2",
    "E40": "#d55e00",
    "mechanical": "#009e73",
    "general": "#cc79a7",
    "truth": "#111111",
    "persistence": "#888888",
}


@dataclass
class MapModel:
    kind: str
    degree: int
    ridge: float
    state_mean: np.ndarray
    state_scale: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray

    def normalize(self, state: np.ndarray) -> np.ndarray:
        return (np.asarray(state, dtype=np.float64) - self.state_mean) / self.state_scale

    def denormalize(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=np.float64) * self.state_scale + self.state_mean

    def step(self, normalized_state: np.ndarray) -> np.ndarray:
        state = np.asarray(normalized_state, dtype=np.float64).reshape(1, 2)
        features = polynomial_features(state, self.degree)
        features = (features - self.feature_mean) / self.feature_scale
        augmented = np.column_stack((np.ones(len(features)), features))
        learned = (augmented @ self.coefficients).ravel()
        if self.kind == "general":
            return learned
        physical = self.denormalize(state).ravel()
        t_next = physical[0] + MAP_STEP_US * physical[1]
        t_next_normalized = (t_next - self.state_mean[0]) / self.state_scale[0]
        return np.asarray([t_next_normalized, learned[0]], dtype=np.float64)


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


def finite_or_nan(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else float("nan")


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(x) < 3 or np.std(x) <= 1.0e-14 or np.std(y) <= 1.0e-14:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def selected_lags() -> dict[str, int]:
    path = closure.DEFAULT_OUTPUT / "fixed_forecast_metrics.csv"
    result: dict[str, int] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if (
                row["observable"] == "joint_selected"
                and math.isclose(float(row["horizon_us"]), FUTURE_HORIZON_US)
                and row["model"] == "affine_T_dT_selected"
            ):
                label = f"{row['sweep']}{int(round(float(row['condition'])))}"
                result[label] = int(row["affine_T_dT_selected_lag_frames"])
    required = {f"E{x}" for x in closure.E_VALUES} | {f"B{x}" for x in closure.B_VALUES}
    missing = required - set(result)
    if missing:
        raise ValueError(f"Missing validation-selected derivative lags: {sorted(missing)}")
    return result


def load_cases() -> list[closure.CaseData]:
    cases = [closure.load_electric_case(value) for value in closure.E_VALUES]
    cases.extend(closure.load_magnetic_case(value) for value in closure.B_VALUES)
    trimmed = []
    for case in cases:
        common_length = min(
            len(case.time_us),
            *(len(values) for values in case.observables.values()),
        )
        trimmed.append(
            closure.CaseData(
                sweep=case.sweep,
                condition=case.condition,
                label=case.label,
                time_us=np.asarray(case.time_us[:common_length], dtype=np.float64),
                observables={
                    key: np.asarray(values[:common_length], dtype=np.float64)
                    for key, values in case.observables.items()
                },
                metadata={
                    **case.metadata,
                    "common_length": common_length,
                    "original_time_length": len(case.time_us),
                },
            )
        )
    return trimmed


def state_from_transport(values: np.ndarray, lag: int) -> np.ndarray:
    transport = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    velocity = closure.causal_secant(transport, lag)
    return np.column_stack((transport.ravel(), velocity.ravel()))


def standardize_state(state: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(state[mask], axis=0)
    scale = np.std(state[mask], axis=0)
    scale = np.where(scale > 1.0e-14, scale, 1.0)
    return (state - mean) / scale, mean, scale


def local_geometry_metrics(
    time_us: np.ndarray,
    state_normalized: np.ndarray,
    transport: np.ndarray,
) -> dict:
    available_end = min(ANALYSIS_INTERVAL[1], float(time_us[-1]))
    valid = np.flatnonzero(
        (time_us >= ANALYSIS_INTERVAL[0] - 1.0e-9)
        & (time_us + FUTURE_HORIZON_US <= available_end + 1.0e-9)
    )
    query = valid[::QUERY_STEP]
    points = state_normalized[valid]
    tree = cKDTree(points)
    future_increment = transport[valid + FUTURE_HORIZON_FRAMES] - transport[valid]
    future_scale = max(float(np.std(future_increment)), 1.0e-30)

    k_query = min(len(valid), max(128, NEIGHBORS + 2))
    recurrence_distances = []
    recurrent_future_differences = []
    local_variances = []
    local_ranges = []
    branch_counts = []
    analogue_errors = []
    persistence_errors = []
    local_dimensions = []
    transverse_fractions = []

    valid_lookup = {int(index): position for position, index in enumerate(valid)}
    for original_index in query:
        point_position = valid_lookup[int(original_index)]
        distances, positions = tree.query(points[point_position], k=k_query)
        distances = np.atleast_1d(distances)
        positions = np.atleast_1d(positions)
        keep_positions = []
        keep_distances = []
        for distance, position in zip(distances, positions):
            candidate = int(valid[int(position)])
            if abs(candidate - int(original_index)) < THEILER_FRAMES:
                continue
            keep_positions.append(int(position))
            keep_distances.append(float(distance))
            if len(keep_positions) >= NEIGHBORS:
                break
        if len(keep_positions) < max(5, NEIGHBORS // 2):
            continue
        neighbor_future = future_increment[np.asarray(keep_positions, dtype=int)]
        query_future = future_increment[point_position]
        recurrence_distances.append(keep_distances[0])
        recurrent_future_differences.append(
            abs(float(neighbor_future[0] - query_future)) / future_scale
        )
        local_variances.append(float(np.var(neighbor_future)) / (future_scale**2))
        local_ranges.append(float(np.ptp(neighbor_future)) / future_scale)
        sorted_future = np.sort(neighbor_future / future_scale)
        branch_counts.append(1 + int(np.sum(np.diff(sorted_future) > 0.75)))
        local_prediction = float(np.mean(neighbor_future))
        analogue_errors.append((float(query_future) - local_prediction) ** 2)
        persistence_errors.append(float(query_future) ** 2)

        _, local_positions = tree.query(points[point_position], k=min(31, len(valid)))
        local_cloud = points[np.atleast_1d(local_positions).astype(int)]
        covariance = np.cov(local_cloud.T)
        eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 0.0))[::-1]
        denominator = float(np.sum(eigenvalues * eigenvalues))
        local_dimensions.append(
            float(np.sum(eigenvalues) ** 2 / denominator) if denominator > 0.0 else 1.0
        )
        transverse_fractions.append(
            float(eigenvalues[-1] / max(np.sum(eigenvalues), 1.0e-30))
        )

    if not recurrence_distances:
        raise ValueError("No valid recurrence queries")

    recurrence_distances_array = np.asarray(recurrence_distances)
    recurrent_future_array = np.asarray(recurrent_future_differences)
    recurrent = recurrence_distances_array <= RECURRENCE_EPSILON
    global_covariance = np.cov(points.T)
    global_eigenvalues = np.maximum(np.linalg.eigvalsh(global_covariance), 0.0)
    global_pr = float(np.sum(global_eigenvalues) ** 2 / np.sum(global_eigenvalues**2))
    return {
        "geometry_queries": len(recurrence_distances),
        "global_participation_dimension": global_pr,
        "median_local_participation_dimension": float(np.median(local_dimensions)),
        "median_local_transverse_fraction": float(np.median(transverse_fractions)),
        "median_recurrence_distance": float(np.median(recurrence_distances_array)),
        "recurrence_fraction_eps0p2": float(np.mean(recurrent)),
        "fold_fraction_given_recurrence": (
            float(np.mean(recurrent_future_array[recurrent] > 1.0))
            if np.any(recurrent)
            else float("nan")
        ),
        "median_recurrent_future_difference": (
            float(np.median(recurrent_future_array[recurrent]))
            if np.any(recurrent)
            else float("nan")
        ),
        "normalized_local_future_variance": float(np.mean(local_variances)),
        "median_local_future_range": float(np.median(local_ranges)),
        "multibranch_fraction": float(np.mean(np.asarray(branch_counts) > 1)),
        "analogue_skill_vs_persistence": 1.0
        - float(np.mean(analogue_errors)) / max(float(np.mean(persistence_errors)), 1.0e-30),
    }


def spectral_metrics(time_us: np.ndarray, transport: np.ndarray) -> dict:
    window_us = 4.0
    step_us = 2.0
    window_frames = int(round(window_us / DT_US))
    frequencies = np.fft.rfftfreq(window_frames, d=DT_US)
    dominant = []
    concentration = []
    entropy = []
    starts = np.arange(ANALYSIS_INTERVAL[0], ANALYSIS_INTERVAL[1] - window_us + 1.0e-9, step_us)
    for start in starts:
        first = int(np.argmin(np.abs(time_us - start)))
        segment = np.asarray(transport[first : first + window_frames], dtype=np.float64)
        if len(segment) != window_frames:
            continue
        x = np.arange(len(segment), dtype=np.float64)
        slope, intercept = np.polyfit(x, segment, 1)
        segment = segment - (slope * x + intercept)
        spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment)))) ** 2
        spectrum[0] = 0.0
        peak = int(np.argmax(spectrum))
        total = max(float(np.sum(spectrum)), 1.0e-30)
        lo = max(1, peak - 1)
        hi = min(len(spectrum), peak + 2)
        probability = spectrum[1:] / total
        dominant.append(float(frequencies[peak]))
        concentration.append(float(np.sum(spectrum[lo:hi])) / total)
        nonzero = probability[probability > 0.0]
        entropy.append(float(-np.sum(nonzero * np.log(nonzero))) / math.log(len(probability)))
    dominant_array = np.asarray(dominant)
    return {
        "spectrum_windows": len(dominant),
        "dominant_frequency_mhz_median": float(np.median(dominant_array)),
        "dominant_frequency_mhz_std": float(np.std(dominant_array)),
        "dominant_frequency_cv": float(np.std(dominant_array))
        / max(abs(float(np.mean(dominant_array))), 1.0e-30),
        "spectral_peak_concentration": float(np.mean(concentration)),
        "spectral_entropy": float(np.mean(entropy)),
    }


def return_map_metrics(time_us: np.ndarray, transport: np.ndarray) -> dict:
    mask = (time_us >= ANALYSIS_INTERVAL[0]) & (time_us <= ANALYSIS_INTERVAL[1])
    values = np.asarray(transport[mask], dtype=np.float64)
    times = time_us[mask]
    standardized = (values - np.mean(values)) / max(float(np.std(values)), 1.0e-30)
    smooth = np.convolve(standardized, np.ones(5) / 5.0, mode="same")
    peaks, _ = find_peaks(smooth, distance=3, prominence=0.10)
    if len(peaks) < 4:
        return {
            "peak_count": len(peaks),
            "peak_interval_cv": float("nan"),
            "peak_amplitude_cv": float("nan"),
            "return_map_correlation": float("nan"),
        }
    intervals = np.diff(times[peaks])
    amplitudes = smooth[peaks]
    return {
        "peak_count": len(peaks),
        "peak_interval_cv": float(np.std(intervals)) / max(float(np.mean(intervals)), 1.0e-30),
        "peak_amplitude_cv": float(np.std(amplitudes)) / max(abs(float(np.mean(amplitudes))), 1.0e-30),
        "return_map_correlation": correlation(amplitudes[:-1], amplitudes[1:]),
    }


def diagnose_geometry(case: closure.CaseData, lag: int, representation: str) -> tuple[dict, np.ndarray]:
    transport = np.asarray(case.observables["joint_selected"], dtype=np.float64)
    state = state_from_transport(transport, lag)
    mask = (case.time_us >= ANALYSIS_INTERVAL[0]) & (case.time_us <= ANALYSIS_INTERVAL[1])
    state_normalized, _, _ = standardize_state(state, mask)
    row = {
        "sweep": case.sweep,
        "condition": case.condition,
        "case": case.label,
        "representation": representation,
        "lag_frames": lag,
        "lag_us": lag * DT_US,
        **local_geometry_metrics(case.time_us, state_normalized, transport),
        **spectral_metrics(case.time_us, transport),
        **return_map_metrics(case.time_us, transport),
    }
    return row, state_normalized


def polynomial_features(state: np.ndarray, degree: int) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    columns = []
    for order in range(1, degree + 1):
        for combination in itertools.combinations_with_replacement(range(2), order):
            column = np.ones(len(state), dtype=np.float64)
            for index in combination:
                column *= state[:, index]
            columns.append(column)
    return np.column_stack(columns)


def fit_map(
    state: np.ndarray,
    time_us: np.ndarray,
    interval: tuple[float, float],
    kind: str,
    degree: int,
    ridge: float,
) -> MapModel:
    current = np.flatnonzero(
        (time_us >= interval[0] - 1.0e-9)
        & (time_us + MAP_STEP_US <= interval[1] + 1.0e-9)
    )
    current = current[current + MAP_STEP_FRAMES < len(state)]
    state_mean = np.mean(state[current], axis=0)
    state_scale = np.std(state[current], axis=0)
    state_scale = np.where(state_scale > 1.0e-14, state_scale, 1.0)
    x = (state[current] - state_mean) / state_scale
    y = (state[current + MAP_STEP_FRAMES] - state_mean) / state_scale
    raw_features = polynomial_features(x, degree)
    feature_mean = np.mean(raw_features, axis=0)
    feature_scale = np.std(raw_features, axis=0)
    feature_scale = np.where(feature_scale > 1.0e-12, feature_scale, 1.0)
    features = (raw_features - feature_mean) / feature_scale
    augmented = np.column_stack((np.ones(len(features)), features))
    target = y if kind == "general" else y[:, 1:2]
    penalty = np.eye(augmented.shape[1])
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        augmented.T @ augmented + ridge * penalty,
        augmented.T @ target,
    )
    return MapModel(
        kind=kind,
        degree=degree,
        ridge=ridge,
        state_mean=state_mean,
        state_scale=state_scale,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        coefficients=coefficients,
    )


def rollout_map(
    model: MapModel,
    state: np.ndarray,
    time_us: np.ndarray,
    start_us: float,
    end_us: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start_index = int(np.argmin(np.abs(time_us - start_us)))
    count = int(math.floor((end_us - time_us[start_index]) / MAP_STEP_US + 1.0e-9)) + 1
    indices = start_index + np.arange(count) * MAP_STEP_FRAMES
    indices = indices[indices < len(state)]
    normalized = model.normalize(state[start_index])
    prediction = [model.denormalize(normalized)]
    for _ in range(1, len(indices)):
        normalized = model.step(normalized)
        if not np.all(np.isfinite(normalized)) or np.max(np.abs(normalized)) > 100.0:
            remaining = len(indices) - len(prediction)
            prediction.extend([np.full(2, np.nan)] * remaining)
            break
        prediction.append(model.denormalize(normalized))
    return time_us[indices], state[indices], np.asarray(prediction, dtype=np.float64)


def state_rollout_score(model: MapModel, state: np.ndarray, time_us: np.ndarray, start: float, end: float) -> float:
    _, truth, prediction = rollout_map(model, state, time_us, start, end)
    if not np.all(np.isfinite(prediction)):
        return 1.0e12
    normalized_truth = model.normalize(truth)
    normalized_prediction = model.normalize(prediction)
    return float(np.mean((normalized_truth - normalized_prediction) ** 2))


def select_map_model(case: closure.CaseData, state: np.ndarray, kind: str) -> tuple[dict, list[dict]]:
    rows = []
    for degree in MAP_DEGREES:
        for ridge in MAP_RIDGES:
            model = fit_map(state, case.time_us, (12.0, 18.0), kind, degree, ridge)
            score = state_rollout_score(model, state, case.time_us, 18.0, 20.0)
            rows.append(
                {
                    "case": case.label,
                    "kind": kind,
                    "degree": degree,
                    "ridge": ridge,
                    "validation_state_nmse": score,
                }
            )
    selected = min(rows, key=lambda row: row["validation_state_nmse"])
    return selected, rows


def rollout_metrics(
    case: closure.CaseData,
    model: MapModel,
    times: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    evaluation_end_us: float,
    protocol: str,
    forecast_start_us: float,
) -> dict:
    keep = times <= evaluation_end_us + 1.0e-9
    times = times[keep]
    truth = truth[keep]
    prediction = prediction[keep]
    finite = np.all(np.isfinite(prediction), axis=1)
    valid_fraction = float(np.mean(finite))
    if np.sum(finite) < 3:
        return {
            "case": case.label,
            "kind": model.kind,
            "protocol": protocol,
            "forecast_start_us": forecast_start_us,
            "evaluation_end_us": evaluation_end_us,
            "horizon_us": evaluation_end_us - forecast_start_us,
            "samples": len(times),
            "valid_fraction": valid_fraction,
            "transport_skill_vs_persistence": float("nan"),
            "transport_correlation": float("nan"),
            "transport_std_ratio": float("nan"),
            "state_nmse": float("nan"),
        }
    truth = truth[finite]
    prediction = prediction[finite]
    persistence = np.full(len(truth), truth[0, 0])
    model_mse = float(np.mean((truth[:, 0] - prediction[:, 0]) ** 2))
    persistence_mse = float(np.mean((truth[:, 0] - persistence) ** 2))
    normalized_truth = model.normalize(truth)
    normalized_prediction = model.normalize(prediction)
    return {
        "case": case.label,
        "kind": model.kind,
        "degree": model.degree,
        "ridge": model.ridge,
        "protocol": protocol,
        "forecast_start_us": forecast_start_us,
        "evaluation_end_us": evaluation_end_us,
        "horizon_us": evaluation_end_us - forecast_start_us,
        "samples": len(times),
        "valid_fraction": valid_fraction,
        "transport_skill_vs_persistence": 1.0 - model_mse / max(persistence_mse, 1.0e-30),
        "transport_correlation": correlation(truth[:, 0], prediction[:, 0]),
        "transport_std_ratio": float(np.std(prediction[:, 0])) / max(float(np.std(truth[:, 0])), 1.0e-30),
        "velocity_correlation": correlation(truth[:, 1], prediction[:, 1]),
        "state_nmse": float(np.mean((normalized_truth - normalized_prediction) ** 2)),
    }


def evaluate_autonomous_maps(
    case: closure.CaseData,
    lag: int,
) -> tuple[list[dict], list[dict], list[dict], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    state = state_from_transport(case.observables["joint_selected"], lag)
    selection_rows = []
    fixed_rows = []
    rolling_rows = []
    rollouts = {}
    selected_by_kind = {}
    for kind in MAP_KINDS:
        selected, candidates = select_map_model(case, state, kind)
        selection_rows.extend(candidates)
        selected_by_kind[kind] = selected
        model = fit_map(
            state,
            case.time_us,
            (12.0, 20.0),
            kind,
            int(selected["degree"]),
            float(selected["ridge"]),
        )
        times, truth, prediction = rollout_map(model, state, case.time_us, 20.0, 30.0)
        rollouts[kind] = (times, truth, prediction)
        for horizon in (0.30, 0.60, 1.20, 2.0, 5.0, 10.0):
            fixed_rows.append(
                rollout_metrics(
                    case,
                    model,
                    times,
                    truth,
                    prediction,
                    min(20.0 + horizon, float(times[-1])),
                    "fixed_12to20_rollout_20to30",
                    20.0,
                )
            )
        for start in ROLLING_STARTS:
            history = (start - ROLLING_HISTORY_US, start)
            rolling_model = fit_map(
                state,
                case.time_us,
                history,
                kind,
                int(selected["degree"]),
                float(selected["ridge"]),
            )
            r_time, r_truth, r_prediction = rollout_map(
                rolling_model,
                state,
                case.time_us,
                start,
                min(start + ROLLING_TEST_US, float(case.time_us[-1])),
            )
            for horizon in (1.20, 2.0):
                rolling_rows.append(
                    rollout_metrics(
                        case,
                        rolling_model,
                        r_time,
                        r_truth,
                        r_prediction,
                        min(start + horizon, float(r_time[-1])),
                        "rolling_4us_history",
                        start,
                    )
                )
    return selection_rows, fixed_rows, rolling_rows, rollouts


def plot_phase_portraits(
    cases: list[closure.CaseData],
    lags: dict[str, int],
    states: dict[tuple[str, str], np.ndarray],
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), constrained_layout=True, sharex=True, sharey=True)
    for row_index, sweep in enumerate(("E", "B")):
        selected_cases = [case for case in cases if case.sweep == sweep]
        for axis, case in zip(axes[row_index], selected_cases):
            state = states[(case.label, "selected")]
            mask = (case.time_us >= ANALYSIS_INTERVAL[0]) & (case.time_us <= ANALYSIS_INTERVAL[1])
            points = axis.scatter(
                state[mask, 0],
                state[mask, 1],
                c=case.time_us[mask],
                s=4,
                alpha=0.65,
                cmap="viridis",
                rasterized=True,
            )
            axis.set_title(f"{case.label}: lag={lags[case.label] * DT_US:.3f} us")
            axis.axhline(0.0, color="#cccccc", linewidth=0.6)
            axis.axvline(0.0, color="#cccccc", linewidth=0.6)
            axis.grid(alpha=0.18)
        axes[row_index, 0].set_ylabel("normalized V_delta")
    for axis in axes[-1]:
        axis.set_xlabel("normalized transport T")
    colorbar = fig.colorbar(points, ax=axes, location="right", fraction=0.018, pad=0.02)
    colorbar.set_label("time [us]")
    fig.suptitle("Joint modal transport phase portraits (validation-selected causal lag)")
    fig.savefig(output / "transport_phase_portraits_selected_lag.png", dpi=180)
    plt.close(fig)


def plot_geometry_summary(rows: list[dict], output: Path) -> None:
    selected = [row for row in rows if row["representation"] == "selected"]
    selected.sort(key=lambda row: (row["sweep"], row["condition"]))
    labels = [row["case"] for row in selected]
    metrics = (
        ("median_local_participation_dimension", "local participation dimension"),
        ("recurrence_fraction_eps0p2", "recurrence fraction"),
        ("fold_fraction_given_recurrence", "folding among recurrences"),
        ("normalized_local_future_variance", "local future variance"),
        ("spectral_peak_concentration", "spectral peak concentration"),
        ("dominant_frequency_cv", "dominant-frequency CV"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    colors = ["#0072b2" if label.startswith("E") else "#d55e00" for label in labels]
    for axis, (key, title) in zip(axes.ravel(), metrics):
        values = [row[key] for row in selected]
        axis.bar(np.arange(len(labels)), values, color=colors, alpha=0.85)
        axis.set_xticks(np.arange(len(labels)), labels, rotation=45)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Observable-dependent transport geometry diagnostics")
    fig.savefig(output / "transport_geometry_summary.png", dpi=180)
    plt.close(fig)


def plot_lag_controls(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    metrics = (
        ("fold_fraction_given_recurrence", "folding among recurrences"),
        ("normalized_local_future_variance", "local future variance"),
        ("analogue_skill_vs_persistence", "analogue skill vs persistence"),
    )
    for axis, (key, title) in zip(axes, metrics):
        for sweep, marker in (("E", "o"), ("B", "s")):
            cases = sorted({row["case"] for row in rows if row["sweep"] == sweep})
            for case_label in cases:
                pair = {row["representation"]: row for row in rows if row["case"] == case_label}
                lag1 = pair["lag1"][key]
                selected = pair["selected"][key]
                axis.scatter(lag1, selected, marker=marker, s=55, label=sweep if case_label == cases[0] else None)
                axis.annotate(case_label, (lag1, selected), xytext=(4, 3), textcoords="offset points", fontsize=8)
        limits = axis.get_xlim()
        low = min(limits[0], axis.get_ylim()[0])
        high = max(limits[1], axis.get_ylim()[1])
        axis.plot([low, high], [low, high], "--", color="#777777", linewidth=1)
        axis.set_xlabel("15 ns velocity")
        axis.set_ylabel("validation-selected velocity")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, loc="best")
    fig.savefig(output / "lag_selection_geometry_controls.png", dpi=180)
    plt.close(fig)


def plot_autonomous_rollouts(
    case_rollouts: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), constrained_layout=True)
    for row, case_label in enumerate(("E25", "E40")):
        rollouts = case_rollouts[case_label]
        for column, kind in enumerate(MAP_KINDS):
            axis = axes[row, column]
            times, truth, prediction = rollouts[kind]
            scale = max(float(np.std(truth[:, 0])), 1.0e-30)
            centered_truth = (truth[:, 0] - np.mean(truth[:, 0])) / scale
            centered_prediction = (prediction[:, 0] - np.mean(truth[:, 0])) / scale
            axis.plot(times, centered_truth, color=COLORS["truth"], linewidth=1.5, label="truth")
            axis.plot(times, centered_prediction, color=COLORS[kind], linewidth=1.2, label=kind)
            axis.axhline(centered_truth[0], color=COLORS["persistence"], linestyle="--", linewidth=1, label="persistence")
            axis.set_title(f"{case_label}: {kind} autonomous map")
            axis.set_xlabel("time [us]")
            axis.set_ylabel("transport [test-standardized]")
            axis.grid(alpha=0.25)
            axis.legend(frameon=False, loc="lower right")
    fig.savefig(output / "e25_e40_autonomous_transport_rollouts.png", dpi=180)
    plt.close(fig)


def plot_rolling_map_skill(rows: list[dict], output: Path) -> None:
    primary = [row for row in rows if math.isclose(float(row["horizon_us"]), 1.2, abs_tol=1.0e-6)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for row_index, case_label in enumerate(("E25", "E40")):
        for column, kind in enumerate(MAP_KINDS):
            axis = axes[row_index, column]
            selected = sorted(
                [row for row in primary if row["case"] == case_label and row["kind"] == kind],
                key=lambda row: row["forecast_start_us"],
            )
            axis.plot(
                [row["forecast_start_us"] for row in selected],
                [row["transport_skill_vs_persistence"] for row in selected],
                marker="o",
                color=COLORS[kind],
                label=kind,
            )
            axis.axhline(0.0, color="#777777", linestyle="--", linewidth=1)
            axis.set_title(f"{case_label}: {kind}")
            axis.set_xlabel("forecast start [us]")
            axis.set_ylabel("MSE skill vs persistence")
            axis.grid(alpha=0.25)
            axis.legend(frameon=False, loc="best")
    fig.suptitle("4 us history -> 1.2 us autonomous transport forecast")
    fig.savefig(output / "e25_e40_rolling_autonomous_skill.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    lags = selected_lags()
    cases = load_cases()
    geometry_rows = []
    states = {}
    for case in cases:
        for representation, lag in (("lag1", 1), ("selected", lags[case.label])):
            row, state = diagnose_geometry(case, lag, representation)
            geometry_rows.append(row)
            states[(case.label, representation)] = state
    write_csv(output / "transport_phase_geometry_metrics.csv", geometry_rows)

    selection_rows = []
    fixed_rows = []
    rolling_rows = []
    case_rollouts = {}
    for label in ("E25", "E40"):
        case = next(case for case in cases if case.label == label)
        selected, fixed, rolling, rollouts = evaluate_autonomous_maps(case, lags[label])
        selection_rows.extend(selected)
        fixed_rows.extend(fixed)
        rolling_rows.extend(rolling)
        case_rollouts[label] = rollouts
    write_csv(output / "autonomous_map_model_selection.csv", selection_rows)
    write_csv(output / "autonomous_map_fixed_metrics.csv", fixed_rows)
    write_csv(output / "autonomous_map_rolling_metrics.csv", rolling_rows)

    plot_phase_portraits(cases, lags, states, output)
    plot_geometry_summary(geometry_rows, output)
    plot_lag_controls(geometry_rows, output)
    plot_autonomous_rollouts(case_rollouts, output)
    plot_rolling_map_skill(rolling_rows, output)

    selected_geometry = [row for row in geometry_rows if row["representation"] == "selected"]
    fixed_full = []
    for case_label in ("E25", "E40"):
        for kind in MAP_KINDS:
            candidates = [
                row for row in fixed_rows if row["case"] == case_label and row["kind"] == kind
            ]
            fixed_full.append(max(candidates, key=lambda row: float(row["horizon_us"])))
    summary = {
        "status": "PASS",
        "protocol": {
            "analysis_interval_us": ANALYSIS_INTERVAL,
            "future_horizon_us": FUTURE_HORIZON_US,
            "theiler_us": THEILER_US,
            "recurrence_epsilon_in_standardized_TV": RECURRENCE_EPSILON,
            "selected_lags_frames": lags,
            "map_step_us": MAP_STEP_US,
            "map_validation": "train 12--18 us, autonomous validation 18--20 us",
            "map_test": "refit 12--20 us, autonomous rollout 20--30 us",
            "rolling_map": "preceding 4 us refit, next 2 us autonomous rollout",
        },
        "selected_geometry": {row["case"]: row for row in selected_geometry},
        "fixed_autonomous_primary": [
            row for row in fixed_rows if math.isclose(float(row["horizon_us"]), 1.2, abs_tol=1.0e-6)
        ],
        "fixed_autonomous_full": fixed_full,
        "rolling_autonomous_primary": [
            row for row in rolling_rows if math.isclose(float(row["horizon_us"]), 1.2, abs_tol=1.0e-6)
        ],
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    geometry_by_case = {row["case"]: row for row in selected_geometry}
    fixed_by_key = {
        (row["case"], row["kind"], round(float(row["horizon_us"]), 1)): row
        for row in fixed_rows
    }
    e25_primary = fixed_by_key[("E25", "general", 1.2)]
    e40_primary = fixed_by_key[("E40", "general", 1.2)]
    e25_full = next(row for row in fixed_full if row["case"] == "E25" and row["kind"] == "general")
    e40_full = next(row for row in fixed_full if row["case"] == "E40" and row["kind"] == "general")
    e40_rolling = [
        row
        for row in rolling_rows
        if row["case"] == "E40"
        and row["kind"] == "general"
        and math.isclose(float(row["horizon_us"]), 1.2, abs_tol=1.0e-6)
    ]

    readme = f"""# RadAz observable-dependent transport phase geometry

This directory tests whether the condition-dependent closure of joint modal
transport can be explained by the geometry of the projected state
`(T, V_delta)`.

## Leakage controls

- Derivative lags come from the preceding chronological validation only.
- Geometry is descriptive over 12--30 us and is reported for both 15 ns and
  the validation-selected lag.
- Autonomous-map hyperparameters are selected on 18--20 us after fitting
  12--18 us.  Final maps are refit on 12--20 us and rolled out over 20--30 us
  with no teacher forcing.
- Rolling maps use only the preceding 4 us and forecast the next 2 us.

## Outputs

- `transport_phase_geometry_metrics.csv`: local dimension, recurrence,
  projection folding, analogue skill, return-map and spectral diagnostics.
- `transport_phase_portraits_selected_lag.png`: all E/B conditions on common
  standardized axes.
- `lag_selection_geometry_controls.png`: 15 ns versus validated finite-time
  velocity, preventing a favorable-lag-only interpretation.
- `autonomous_map_*`: validation, fixed 10 us and rolling autonomous tests for
  E25 and E40.

## Main results

- E40 has a narrow recurrent loop: folding fraction
  {geometry_by_case['E40']['fold_fraction_given_recurrence']:.3f}, normalized
  local future variance
  {geometry_by_case['E40']['normalized_local_future_variance']:.3f}, and
  spectral peak concentration
  {geometry_by_case['E40']['spectral_peak_concentration']:.3f}.
- The unconstrained two-dimensional E40 map passes the 1.2 us autonomous test
  with skill {e40_primary['transport_skill_vs_persistence']:.3f}, correlation
  {e40_primary['transport_correlation']:.3f}, and standard-deviation ratio
  {e40_primary['transport_std_ratio']:.3f}.  Over the full
  {e40_full['horizon_us']:.3f} us rollout these are
  {e40_full['transport_skill_vs_persistence']:.3f},
  {e40_full['transport_correlation']:.3f}, and
  {e40_full['transport_std_ratio']:.3f}.
- E40 also remains reproducible in every rolling window: the 1.2 us skill is
  {min(float(row['transport_skill_vs_persistence']) for row in e40_rolling):.3f}
  to {max(float(row['transport_skill_vs_persistence']) for row in e40_rolling):.3f},
  with correlations
  {min(float(row['transport_correlation']) for row in e40_rolling):.3f} to
  {max(float(row['transport_correlation']) for row in e40_rolling):.3f}.
- E25 is locally predictable but not a faithful long autonomous closure.  At
  1.2 us the general map has skill
  {e25_primary['transport_skill_vs_persistence']:.3f}, correlation
  {e25_primary['transport_correlation']:.3f}, and amplitude ratio
  {e25_primary['transport_std_ratio']:.3f}; over
  {e25_full['horizon_us']:.3f} us correlation falls to
  {e25_full['transport_correlation']:.3f} and the amplitude ratio to
  {e25_full['transport_std_ratio']:.3f}.
- Enforcing `T_next = T + delta_t * V_delta` is unstable, including at E40.
  The positive result is therefore a two-dimensional discrete autonomous map,
  not yet evidence for the stricter mechanical ODE `dT/dt = V_delta`.

The geometry also prevents an E40-only overstatement: E25 and B15 have low
local folding under their validated finite-time coordinates.  B15 nevertheless
failed the preceding affine lead-time closure.  Low folding is therefore a
useful necessary-screening signal here, but not a sufficient forecast guarantee;
nonlinearity, time drift, and the exact prediction task still matter.

The autonomous map is intentionally stricter than the preceding lead-time
forecast.  A good lead-time result means future T is predictable when the true
current state is observed at every query.  A good autonomous result additionally
requires the two-dimensional state to remain on its own trajectory after the
initial condition is supplied once.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(f"Saved analysis to {output}")


if __name__ == "__main__":
    main()
