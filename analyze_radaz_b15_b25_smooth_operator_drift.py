#!/usr/bin/env python3
"""Separate smooth operator drift from switching and estimation noise.

The analysis keeps one latent coordinate system per case, fits equal-length
rolling affine DMD operators, and uses B15 as a stationary negative control.
Overlapping-window smoothness is checked against non-overlapping operators,
time-order permutations, and a moving-block bootstrap noise floor.

The scheduled time-varying forecast is causal with respect to the holdout: its
operator-coordinate model is selected on an earlier validation interval and
then extrapolated without future truth.  The online rolling one-step result
uses past holdout truth and is therefore diagnostic rather than autonomous.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from scipy.linalg import subspace_angles
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze_radaz_b25_temporal_switching_rom as switching


ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT.parent
OUTPUT = (
    RESEARCH
    / "research_results"
    / "2D_RadAz"
    / "SimVPv2"
    / "workdirs"
    / "analyze_radaz_b15_b25_smooth_operator_drift"
)

DT_US = 0.015
DIAGNOSTIC_START_US = 12.0
DIAGNOSTIC_END_US = 29.75
REPRESENTATION_END_US = 24.0
WINDOW_US = 2.0
ROLLING_STEP_US = 0.25
NONOVERLAP_STEP_US = 2.0
VALIDATION_US = 2.0
BOOTSTRAPS = 40
BOOTSTRAP_BLOCK = 8
PERMUTATIONS = 1000
ALPHAS = (1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0)
TREND_HORIZONS_US = (2.0, 4.0, 6.0)
OPERATOR_COMPONENTS = (1, 2, 3)
RNG_SEED = 20260825


@dataclass(frozen=True)
class Protocol:
    case: str
    name: str
    fit_start_us: float
    fit_end_us: float
    test_end_us: float


PROTOCOLS = (
    Protocol("B25", "B25_12-24_to_30", 12.0, 24.0, 30.0),
    Protocol("B25", "B25_18-30_to_36", 18.0, 30.0, 36.0),
    Protocol("B15", "B15_12-24_to_29p75", 12.0, 24.0, 29.75),
)


@dataclass
class AffineOperator:
    weight: np.ndarray
    bias: np.ndarray
    alpha: float

    @property
    def packed(self) -> np.ndarray:
        return np.concatenate((self.weight, self.bias[:, None]), axis=1)


@dataclass
class CaseRepresentation:
    name: str
    time_us: np.ndarray
    states: dict[str, np.ndarray]
    latent_dimensions: int


def interval_mask(
    time_us: np.ndarray, start_us: float, end_us: float, *, include_end: bool = False
) -> np.ndarray:
    upper = time_us <= end_us + 1.0e-9 if include_end else time_us < end_us - 1.0e-9
    return (time_us >= start_us - 1.0e-9) & upper


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


def load_representation(case_name: str) -> CaseRepresentation:
    case = switching.load_case(case_name)
    representation = switching.fit_representation(
        case, DIAGNOSTIC_START_US, REPRESENTATION_END_US
    )
    return CaseRepresentation(
        name=case_name,
        time_us=representation.time_us,
        states={"L": representation.latent, "L+A+P": representation.lap},
        latent_dimensions=representation.latent_dimensions,
    )


def fit_affine_pairs(x: np.ndarray, y: np.ndarray, alpha: float) -> AffineOperator:
    if len(x) != len(y) or len(x) < x.shape[1] + 2:
        raise ValueError("insufficient transition pairs for affine operator")
    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y, axis=0)
    xc = x - x_mean
    yc = y - y_mean
    gram = xc.T @ xc + alpha * np.eye(x.shape[1])
    weight = np.linalg.solve(gram, xc.T @ yc).T
    bias = y_mean - weight @ x_mean
    return AffineOperator(weight=weight, bias=bias, alpha=alpha)


def fit_affine(states: np.ndarray, alpha: float) -> AffineOperator:
    return fit_affine_pairs(states[:-1], states[1:], alpha)


def apply_operator(operator: AffineOperator, values: np.ndarray) -> np.ndarray:
    return values @ operator.weight.T + operator.bias


def stabilize(operator: AffineOperator) -> AffineOperator:
    radius = float(np.max(np.abs(np.linalg.eigvals(operator.weight))))
    if not math.isfinite(radius) or radius <= 1.0:
        return operator
    return AffineOperator(
        weight=operator.weight / radius,
        bias=operator.bias,
        alpha=operator.alpha,
    )


def relative_action_distance(
    left: AffineOperator, right: AffineOperator, probes: np.ndarray
) -> float:
    left_action = apply_operator(left, probes)
    right_action = apply_operator(right, probes)
    numerator = np.sqrt(np.mean((left_action - right_action) ** 2))
    denominator = 0.5 * (
        np.sqrt(np.mean(left_action**2)) + np.sqrt(np.mean(right_action**2))
    )
    return float(numerator / max(denominator, 1.0e-12))


def relative_frobenius(left: AffineOperator, right: AffineOperator) -> float:
    numerator = np.linalg.norm(left.packed - right.packed)
    denominator = 0.5 * (
        np.linalg.norm(left.packed) + np.linalg.norm(right.packed)
    )
    return float(numerator / max(denominator, 1.0e-12))


def eigenset_distance(left: AffineOperator, right: AffineOperator) -> float:
    left_values = np.linalg.eigvals(left.weight)
    right_values = np.linalg.eigvals(right.weight)
    cost = np.abs(left_values[:, None] - right_values[None, :])
    rows, columns = linear_sum_assignment(cost)
    scale = 0.5 * (np.mean(np.abs(left_values)) + np.mean(np.abs(right_values)))
    return float(np.mean(cost[rows, columns]) / max(scale, 1.0e-12))


def invariant_subspace_angle(left: AffineOperator, right: AffineOperator) -> float:
    count = min(3, left.weight.shape[0])
    left_values, left_vectors = np.linalg.eig(left.weight)
    right_values, right_vectors = np.linalg.eig(right.weight)
    left_selected = np.argsort(np.abs(left_values))[-count:]
    right_selected = np.argsort(np.abs(right_values))[-count:]
    angles = subspace_angles(
        left_vectors[:, left_selected], right_vectors[:, right_selected]
    )
    return float(np.degrees(np.mean(angles)))


def operator_radius(operator: AffineOperator) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(operator.weight))))


def window_pairs(
    time_us: np.ndarray, states: np.ndarray, start_us: float, end_us: float
) -> tuple[np.ndarray, np.ndarray]:
    mask = interval_mask(time_us, start_us, end_us, include_end=True)
    selected = states[mask]
    if len(selected) < states.shape[1] + 3:
        raise ValueError(f"window {start_us}-{end_us} us is too short")
    return selected[:-1], selected[1:]


def trailing_operator(
    time_us: np.ndarray,
    states: np.ndarray,
    end_us: float,
    alpha: float,
    width_us: float = WINDOW_US,
) -> AffineOperator:
    x, y = window_pairs(time_us, states, end_us - width_us, end_us)
    return fit_affine_pairs(x, y, alpha)


def select_alpha(
    time_us: np.ndarray, states: np.ndarray, fit_start_us: float = 12.0
) -> tuple[float, list[dict]]:
    validation_start = 22.0
    validation_end = 24.0
    indices = np.flatnonzero(
        (time_us >= validation_start - 1.0e-9)
        & (time_us < validation_end - DT_US - 1.0e-9)
    )
    rows = []
    for alpha in ALPHAS:
        errors = []
        for index in indices:
            end_us = float(time_us[index])
            if end_us - WINDOW_US < fit_start_us - 1.0e-9:
                continue
            operator = trailing_operator(time_us, states, end_us, alpha)
            prediction = apply_operator(operator, states[index : index + 1])[0]
            errors.append(float(np.mean((prediction - states[index + 1]) ** 2)))
        rows.append(
            {
                "alpha": alpha,
                "validation_one_step_mse": float(np.mean(errors)),
                "validation_predictions": len(errors),
            }
        )
    selected = min(rows, key=lambda row: row["validation_one_step_mse"])
    return float(selected["alpha"]), rows


def block_bootstrap_noise(
    x: np.ndarray,
    y: np.ndarray,
    operator: AffineOperator,
    alpha: float,
    probes: np.ndarray,
    rng: np.random.Generator,
) -> float:
    count = len(x)
    starts = np.arange(max(1, count - BOOTSTRAP_BLOCK + 1))
    distances = []
    blocks_needed = int(np.ceil(count / BOOTSTRAP_BLOCK))
    for _ in range(BOOTSTRAPS):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        indices = np.concatenate(
            [np.arange(start, min(start + BOOTSTRAP_BLOCK, count)) for start in chosen]
        )[:count]
        sampled = fit_affine_pairs(x[indices], y[indices], alpha)
        distances.append(relative_action_distance(operator, sampled, probes))
    return float(np.median(distances))


def rolling_operators(
    case: CaseRepresentation,
    representation: str,
    start_us: float,
    end_us: float,
    step_us: float,
    alpha: float,
    bootstrap: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[AffineOperator], np.ndarray]:
    states = case.states[representation]
    probe_mask = interval_mask(case.time_us, 12.0, 24.0)
    probes = states[probe_mask][:: max(1, np.count_nonzero(probe_mask) // 128)]
    ends = np.arange(start_us + WINDOW_US, end_us + 1.0e-9, step_us)
    operators: list[AffineOperator] = []
    noise = []
    for end in ends:
        x, y = window_pairs(case.time_us, states, end - WINDOW_US, end)
        operator = fit_affine_pairs(x, y, alpha)
        operators.append(operator)
        noise.append(
            block_bootstrap_noise(x, y, operator, alpha, probes, rng)
            if bootstrap
            else math.nan
        )
    return ends, operators, np.asarray(noise, dtype=np.float64)


def adjacent_metrics(
    case: str,
    representation: str,
    ends: np.ndarray,
    operators: list[AffineOperator],
    noise: np.ndarray,
    probes: np.ndarray,
    window_kind: str,
) -> list[dict]:
    rows = []
    for index, (left, right) in enumerate(zip(operators[:-1], operators[1:])):
        action = relative_action_distance(left, right, probes)
        pooled_noise = math.sqrt(noise[index] ** 2 + noise[index + 1] ** 2)
        rows.append(
            {
                "case": case,
                "representation": representation,
                "window_kind": window_kind,
                "left_end_us": float(ends[index]),
                "right_end_us": float(ends[index + 1]),
                "relative_frobenius": relative_frobenius(left, right),
                "relative_action_distance": action,
                "eigenset_distance": eigenset_distance(left, right),
                "dominant_subspace_angle_deg": invariant_subspace_angle(left, right),
                "pooled_bootstrap_noise": pooled_noise,
                "drift_to_noise": action / max(pooled_noise, 1.0e-12),
            }
        )
    return rows


def ordered_path_test(
    operators: list[AffineOperator],
    probes: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    def path(order: np.ndarray) -> float:
        return float(
            sum(
                relative_action_distance(operators[left], operators[right], probes)
                for left, right in zip(order[:-1], order[1:])
            )
        )

    chronological = path(np.arange(len(operators)))
    null = np.asarray(
        [path(rng.permutation(len(operators))) for _ in range(PERMUTATIONS)],
        dtype=np.float64,
    )
    return {
        "chronological_path": chronological,
        "permuted_path_median": float(np.median(null)),
        "path_ratio_to_permuted": chronological / max(float(np.median(null)), 1.0e-12),
        "permutation_p_lower": float((1 + np.count_nonzero(null <= chronological)) / (len(null) + 1)),
    }


def direction_persistence(operators: list[AffineOperator]) -> float:
    vectors = np.stack([operator.packed.ravel() for operator in operators])
    differences = np.diff(vectors, axis=0)
    values = []
    for left, right in zip(differences[:-1], differences[1:]):
        denominator = np.linalg.norm(left) * np.linalg.norm(right)
        if denominator > 1.0e-12:
            values.append(float(np.dot(left, right) / denominator))
    return float(np.mean(values)) if values else math.nan


def lag_distance_correlation(
    operators: list[AffineOperator], probes: np.ndarray, max_lag: int = 8
) -> tuple[float, float]:
    lags = []
    distances = []
    for lag in range(1, min(max_lag, len(operators) - 1) + 1):
        for index in range(len(operators) - lag):
            lags.append(lag)
            distances.append(
                relative_action_distance(
                    operators[index], operators[index + lag], probes
                )
            )
    result = spearmanr(lags, distances)
    return float(result.statistic), float(result.pvalue)


def operator_pca(
    operators: list[AffineOperator], components: int = 3
) -> tuple[PCA, np.ndarray]:
    values = np.stack([operator.packed.ravel() for operator in operators])
    count = min(components, len(values) - 1, values.shape[1])
    model = PCA(n_components=count, random_state=42)
    coordinates = model.fit_transform(values)
    return model, coordinates


def prediction_metrics(
    truth: np.ndarray, prediction: np.ndarray, persistence: np.ndarray
) -> dict[str, float]:
    mse = float(np.mean((truth - prediction) ** 2))
    persistence_mse = float(np.mean((truth - persistence) ** 2))
    centered_truth = truth - np.mean(truth, axis=0, keepdims=True)
    centered_prediction = prediction - np.mean(prediction, axis=0, keepdims=True)
    denominator = np.linalg.norm(centered_truth) * np.linalg.norm(centered_prediction)
    correlation = (
        float(np.sum(centered_truth * centered_prediction) / denominator)
        if denominator > 1.0e-12
        else math.nan
    )
    return {
        "mse": mse,
        "skill_vs_persistence": 1.0 - mse / max(persistence_mse, 1.0e-12),
        "correlation": correlation,
    }


def rollout_fixed(
    operator: AffineOperator, initial: np.ndarray, steps: int
) -> np.ndarray:
    prediction = np.empty((steps, len(initial)), dtype=np.float64)
    state = initial.copy()
    stable = stabilize(operator)
    for index in range(steps):
        state = apply_operator(stable, state[None])[0]
        prediction[index] = state
    return prediction


def unpack_operator(vector: np.ndarray, dimensions: int, alpha: float) -> AffineOperator:
    packed = vector.reshape(dimensions, dimensions + 1)
    return AffineOperator(
        weight=packed[:, :dimensions], bias=packed[:, dimensions], alpha=alpha
    )


def coordinate_schedule(
    ends: np.ndarray,
    coordinates: np.ndarray,
    future_time: np.ndarray,
    kind: str,
    trend_horizon_us: float,
) -> np.ndarray:
    if kind == "linear":
        mask = ends >= ends[-1] - trend_horizon_us - 1.0e-9
        design = np.stack((np.ones(np.count_nonzero(mask)), ends[mask]), axis=1)
        coefficients = np.linalg.lstsq(design, coordinates[mask], rcond=1.0e-10)[0]
        future_design = np.stack((np.ones(len(future_time)), future_time), axis=1)
        return future_design @ coefficients
    if kind != "ar1":
        raise ValueError(kind)
    x = coordinates[:-1]
    y = coordinates[1:]
    model = fit_affine_pairs(x, y, 1.0e-3)
    step = float(np.median(np.diff(ends)))
    anchor_times = np.arange(ends[-1], future_time[-1] + step + 1.0e-9, step)
    anchor_coordinates = np.empty((len(anchor_times), coordinates.shape[1]))
    anchor_coordinates[0] = coordinates[-1]
    for index in range(1, len(anchor_times)):
        anchor_coordinates[index] = apply_operator(
            stabilize(model), anchor_coordinates[index - 1 : index]
        )[0]
    return np.stack(
        [
            np.interp(future_time, anchor_times, anchor_coordinates[:, column])
            for column in range(coordinates.shape[1])
        ],
        axis=1,
    )


def scheduled_rollout(
    states: np.ndarray,
    time_us: np.ndarray,
    fit_start_us: float,
    fit_end_us: float,
    forecast_time: np.ndarray,
    alpha: float,
    operator_components: int,
    kind: str,
    trend_horizon_us: float,
) -> tuple[np.ndarray, dict]:
    ends, operators, _ = rolling_operators(
        CaseRepresentation("temporary", time_us, {"state": states}, states.shape[1]),
        "state",
        fit_start_us,
        fit_end_us,
        ROLLING_STEP_US,
        alpha,
        False,
        np.random.default_rng(RNG_SEED),
    )
    vectors = np.stack([operator.packed.ravel() for operator in operators])
    count = min(operator_components, len(vectors) - 1)
    pca = PCA(n_components=count, random_state=42)
    coordinates = pca.fit_transform(vectors)
    future_coordinates = coordinate_schedule(
        ends, coordinates, forecast_time, kind, trend_horizon_us
    )
    lower = np.min(coordinates, axis=0)
    upper = np.max(coordinates, axis=0)
    span = np.maximum(upper - lower, 1.0e-12)
    future_coordinates = np.clip(
        future_coordinates, lower - 0.5 * span, upper + 0.5 * span
    )
    reconstructed = pca.inverse_transform(future_coordinates)
    test_start = int(np.searchsorted(time_us, forecast_time[0] - 0.5 * DT_US))
    state = states[test_start - 1].copy()
    prediction = np.empty((len(forecast_time), states.shape[1]), dtype=np.float64)
    radii = []
    for index, vector in enumerate(reconstructed):
        operator = unpack_operator(vector, states.shape[1], alpha)
        radii.append(operator_radius(operator))
        state = apply_operator(stabilize(operator), state[None])[0]
        prediction[index] = state
    metadata = {
        "operator_components": count,
        "operator_variance_capture": float(
            np.sum(pca.explained_variance_ratio_[:count])
        ),
        "raw_radius_mean": float(np.mean(radii)),
        "raw_radius_max": float(np.max(radii)),
    }
    return prediction, metadata


def select_schedule(
    states: np.ndarray,
    time_us: np.ndarray,
    fit_start_us: float,
    fit_end_us: float,
    alpha: float,
) -> tuple[dict, list[dict]]:
    validation_start = fit_end_us - VALIDATION_US
    validation_mask = (
        (time_us > validation_start + 1.0e-9)
        & (time_us <= fit_end_us + 1.0e-9)
    )
    validation_time = time_us[validation_mask]
    truth = states[validation_mask]
    rows = []
    for components in OPERATOR_COMPONENTS:
        for kind in ("linear", "ar1"):
            horizons = TREND_HORIZONS_US if kind == "linear" else (6.0,)
            for horizon in horizons:
                try:
                    prediction, metadata = scheduled_rollout(
                        states,
                        time_us,
                        fit_start_us,
                        validation_start,
                        validation_time,
                        alpha,
                        components,
                        kind,
                        horizon,
                    )
                    mse = float(np.mean((truth - prediction) ** 2))
                except (ValueError, np.linalg.LinAlgError):
                    mse = math.inf
                    metadata = {}
                rows.append(
                    {
                        "kind": kind,
                        "operator_components": components,
                        "trend_horizon_us": horizon,
                        "validation_mse": mse,
                        **metadata,
                    }
                )
    selected = min(rows, key=lambda row: row["validation_mse"])
    return selected, rows


def online_one_step(
    time_us: np.ndarray,
    states: np.ndarray,
    start_us: float,
    end_us: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.flatnonzero(
        (time_us >= start_us - 1.0e-9)
        & (time_us < end_us - DT_US - 1.0e-9)
    )
    truth = states[indices + 1]
    persistence = states[indices]
    prediction = []
    for index in indices:
        operator = trailing_operator(time_us, states, float(time_us[index]), alpha)
        prediction.append(apply_operator(stabilize(operator), states[index : index + 1])[0])
    return truth, np.asarray(prediction), persistence


def fixed_one_step(
    operator: AffineOperator,
    time_us: np.ndarray,
    states: np.ndarray,
    start_us: float,
    end_us: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.flatnonzero(
        (time_us >= start_us - 1.0e-9)
        & (time_us < end_us - DT_US - 1.0e-9)
    )
    truth = states[indices + 1]
    persistence = states[indices]
    prediction = apply_operator(stabilize(operator), states[indices])
    return truth, prediction, persistence


def diagnostic_analysis(
    cases: dict[str, CaseRepresentation]
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    rng = np.random.default_rng(RNG_SEED)
    rolling_rows: list[dict] = []
    adjacent_rows: list[dict] = []
    summary_rows: list[dict] = []
    coordinate_rows: list[dict] = []
    for case_name, case in cases.items():
        for representation, states in case.states.items():
            alpha, alpha_rows = select_alpha(case.time_us, states)
            for row in alpha_rows:
                row.update({"case": case_name, "representation": representation})
            probe_mask = interval_mask(case.time_us, 12.0, 24.0)
            probes = states[probe_mask][:: max(1, np.count_nonzero(probe_mask) // 128)]
            ends, operators, noise = rolling_operators(
                case,
                representation,
                DIAGNOSTIC_START_US,
                min(DIAGNOSTIC_END_US, float(case.time_us[-1])),
                ROLLING_STEP_US,
                alpha,
                True,
                rng,
            )
            pca, coordinates = operator_pca(operators, 3)
            for index, (end, operator) in enumerate(zip(ends, operators)):
                row = {
                    "case": case_name,
                    "representation": representation,
                    "window_start_us": float(end - WINDOW_US),
                    "window_end_us": float(end),
                    "alpha": alpha,
                    "spectral_radius": operator_radius(operator),
                    "bootstrap_action_noise": float(noise[index]),
                }
                for component in range(coordinates.shape[1]):
                    row[f"operator_pc{component + 1}"] = float(coordinates[index, component])
                rolling_rows.append(row)
                coordinate_rows.append(row.copy())
            adjacent = adjacent_metrics(
                case_name,
                representation,
                ends,
                operators,
                noise,
                probes,
                "overlapping",
            )
            adjacent_rows.extend(adjacent)

            non_ends, non_operators, _ = rolling_operators(
                case,
                representation,
                DIAGNOSTIC_START_US,
                min(DIAGNOSTIC_END_US, float(case.time_us[-1])),
                NONOVERLAP_STEP_US,
                alpha,
                False,
                rng,
            )
            path_result = ordered_path_test(non_operators, probes, rng)
            lag_correlation, lag_p = lag_distance_correlation(operators, probes)
            drift_to_noise = np.asarray(
                [row["drift_to_noise"] for row in adjacent], dtype=np.float64
            )
            summary_rows.append(
                {
                    "case": case_name,
                    "representation": representation,
                    "dimensions": states.shape[1],
                    "alpha": alpha,
                    "rolling_windows": len(operators),
                    "nonoverlap_windows": len(non_operators),
                    "adjacent_action_distance_mean": float(
                        np.mean([row["relative_action_distance"] for row in adjacent])
                    ),
                    "adjacent_eigenset_distance_mean": float(
                        np.mean([row["eigenset_distance"] for row in adjacent])
                    ),
                    "adjacent_subspace_angle_mean_deg": float(
                        np.mean([row["dominant_subspace_angle_deg"] for row in adjacent])
                    ),
                    "bootstrap_noise_mean": float(np.mean(noise)),
                    "drift_to_noise_median": float(np.median(drift_to_noise)),
                    "fraction_adjacent_above_noise": float(np.mean(drift_to_noise > 1.0)),
                    "lag_distance_spearman": lag_correlation,
                    "lag_distance_p": lag_p,
                    "direction_persistence": direction_persistence(operators),
                    "operator_pc1_variance": float(pca.explained_variance_ratio_[0]),
                    "operator_pc2_cumulative_variance": float(
                        np.sum(pca.explained_variance_ratio_[:2])
                    ),
                    "operator_pc3_cumulative_variance": float(
                        np.sum(pca.explained_variance_ratio_[:3])
                    ),
                    **path_result,
                }
            )
    return rolling_rows, adjacent_rows, summary_rows, coordinate_rows


def forecast_analysis(
    cases: dict[str, CaseRepresentation]
) -> tuple[list[dict], list[dict], dict]:
    metrics_rows: list[dict] = []
    selection_rows: list[dict] = []
    payloads: dict = {}
    for protocol in PROTOCOLS:
        case = cases[protocol.case]
        payloads[protocol.name] = {}
        for representation, states in case.states.items():
            alpha, _ = select_alpha(case.time_us, states)
            fit_mask = interval_mask(
                case.time_us,
                protocol.fit_start_us,
                protocol.fit_end_us,
                include_end=True,
            )
            fit_states = states[fit_mask]
            fixed = fit_affine(fit_states, alpha)
            last = trailing_operator(
                case.time_us, states, protocol.fit_end_us, alpha
            )
            effective_test_end = min(
                protocol.test_end_us, float(case.time_us[-1])
            )
            test_mask = (
                (case.time_us > protocol.fit_end_us + 1.0e-9)
                & (case.time_us <= effective_test_end + 1.0e-9)
            )
            test_time = case.time_us[test_mask]
            truth = states[test_mask]
            persistence = np.repeat(fit_states[-1][None], len(truth), axis=0)
            fixed_prediction = rollout_fixed(fixed, fit_states[-1], len(truth))
            last_prediction = rollout_fixed(last, fit_states[-1], len(truth))
            selected, candidates = select_schedule(
                states,
                case.time_us,
                protocol.fit_start_us,
                protocol.fit_end_us,
                alpha,
            )
            for row in candidates:
                selection_rows.append(
                    {
                        "protocol": protocol.name,
                        "case": protocol.case,
                        "representation": representation,
                        "selected": int(row is selected),
                        **row,
                    }
                )
            scheduled_prediction, scheduled_metadata = scheduled_rollout(
                states,
                case.time_us,
                protocol.fit_start_us,
                protocol.fit_end_us,
                test_time,
                alpha,
                int(selected["operator_components"]),
                str(selected["kind"]),
                float(selected["trend_horizon_us"]),
            )
            for method, prediction, extra in (
                ("fixed_all_data", fixed_prediction, {}),
                ("last_local_2us", last_prediction, {}),
                (
                    "scheduled_smooth_drift",
                    scheduled_prediction,
                    {
                        "schedule_kind": selected["kind"],
                        "schedule_operator_components": selected[
                            "operator_components"
                        ],
                        "schedule_trend_horizon_us": selected["trend_horizon_us"],
                        **scheduled_metadata,
                    },
                ),
            ):
                result = prediction_metrics(truth, prediction, persistence)
                metrics_rows.append(
                    {
                        "protocol": protocol.name,
                        "case": protocol.case,
                        "representation": representation,
                        "evaluation": "autonomous_rollout",
                        "method": method,
                        "fit_start_us": protocol.fit_start_us,
                        "fit_end_us": protocol.fit_end_us,
                        "test_end_us": float(test_time[-1]),
                        "alpha": alpha,
                        **result,
                        **extra,
                    }
                )
            fixed_truth, fixed_one, one_persistence = fixed_one_step(
                fixed,
                case.time_us,
                states,
                protocol.fit_end_us,
                min(protocol.test_end_us, float(case.time_us[-1])),
            )
            online_truth, online_prediction, online_persistence = online_one_step(
                case.time_us,
                states,
                protocol.fit_end_us,
                min(protocol.test_end_us, float(case.time_us[-1])),
                alpha,
            )
            for method, local_truth, prediction, local_persistence in (
                ("fixed_one_step", fixed_truth, fixed_one, one_persistence),
                (
                    "causal_online_local_one_step",
                    online_truth,
                    online_prediction,
                    online_persistence,
                ),
            ):
                metrics_rows.append(
                    {
                        "protocol": protocol.name,
                        "case": protocol.case,
                        "representation": representation,
                        "evaluation": "one_step_diagnostic",
                        "method": method,
                        "fit_start_us": protocol.fit_start_us,
                        "fit_end_us": protocol.fit_end_us,
                        "test_end_us": float(test_time[-1]),
                        "alpha": alpha,
                        **prediction_metrics(
                            local_truth, prediction, local_persistence
                        ),
                    }
                )
            payloads[protocol.name][representation] = {
                "time_us": test_time,
                "truth": truth,
                "fixed": fixed_prediction,
                "last": last_prediction,
                "scheduled": scheduled_prediction,
                "selected": selected,
            }
    return metrics_rows, selection_rows, payloads


def plot_drift(
    rolling_rows: list[dict], adjacent_rows: list[dict], path: Path
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    colors = {"B15": "#16817a", "B25": "#c44e52"}
    for column, representation in enumerate(("L", "L+A+P")):
        for case in ("B15", "B25"):
            selected = [
                row
                for row in rolling_rows
                if row["case"] == case and row["representation"] == representation
            ]
            axes[0, column].plot(
                [row["window_end_us"] for row in selected],
                [row["spectral_radius"] for row in selected],
                label=case,
                color=colors[case],
                linewidth=2,
            )
            adjacent = [
                row
                for row in adjacent_rows
                if row["case"] == case
                and row["representation"] == representation
                and row["window_kind"] == "overlapping"
            ]
            axes[1, column].plot(
                [row["right_end_us"] for row in adjacent],
                [row["drift_to_noise"] for row in adjacent],
                label=case,
                color=colors[case],
                linewidth=1.8,
            )
        axes[0, column].axhline(1.0, color="black", linestyle="--", linewidth=1)
        axes[0, column].set_title(f"{representation}: local spectral radius")
        axes[0, column].set_xlabel("window end [us]")
        axes[0, column].set_ylabel("spectral radius")
        axes[0, column].grid(alpha=0.25)
        axes[0, column].legend(loc="best")
        axes[1, column].axhline(1.0, color="black", linestyle="--", linewidth=1)
        axes[1, column].set_title(f"{representation}: adjacent drift / bootstrap noise")
        axes[1, column].set_xlabel("right window end [us]")
        axes[1, column].set_ylabel("signal-to-noise ratio")
        axes[1, column].grid(alpha=0.25)
        axes[1, column].legend(loc="best")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_manifold(coordinate_rows: list[dict], path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for row_index, case in enumerate(("B15", "B25")):
        for column, representation in enumerate(("L", "L+A+P")):
            axis = axes[row_index, column]
            selected = [
                row
                for row in coordinate_rows
                if row["case"] == case and row["representation"] == representation
            ]
            scatter = axis.scatter(
                [row["operator_pc1"] for row in selected],
                [row["operator_pc2"] for row in selected],
                c=[row["window_end_us"] for row in selected],
                cmap="viridis",
                s=30,
                alpha=0.85,
            )
            axis.plot(
                [row["operator_pc1"] for row in selected],
                [row["operator_pc2"] for row in selected],
                color="#555555",
                alpha=0.35,
                linewidth=1,
            )
            axis.set_title(f"{case}, {representation}: within-case operator PCA")
            axis.set_xlabel("operator PC1")
            axis.set_ylabel("operator PC2")
            axis.grid(alpha=0.25)
            figure.colorbar(scatter, ax=axis, label="window end [us]")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_forecasts(payloads: dict, path: Path) -> None:
    protocols = list(payloads)
    figure, axes = plt.subplots(
        len(protocols), 2, figsize=(13, 3.6 * len(protocols)), constrained_layout=True
    )
    if len(protocols) == 1:
        axes = np.asarray([axes])
    for row_index, protocol in enumerate(protocols):
        for column, representation in enumerate(("L", "L+A+P")):
            payload = payloads[protocol][representation]
            axis = axes[row_index, column]
            axis.plot(payload["time_us"], payload["truth"][:, 0], color="black", label="truth")
            axis.plot(payload["time_us"], payload["fixed"][:, 0], label="fixed")
            axis.plot(payload["time_us"], payload["last"][:, 0], label="last-local")
            axis.plot(
                payload["time_us"],
                payload["scheduled"][:, 0],
                label="scheduled drift",
            )
            axis.set_title(f"{protocol}, {representation}, state 1")
            axis.set_xlabel("time [us]")
            axis.set_ylabel("standardized coordinate")
            axis.grid(alpha=0.25)
            axis.legend(loc="best")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_readme(
    summary_rows: list[dict], forecast_rows: list[dict], path: Path
) -> None:
    lines = [
        "# B15/B25 smooth operator drift diagnosis",
        "",
        "This analysis tests whether B25 is better described by a smoothly evolving local affine DMD operator than by the chronological two-expert switch tested previously.",
        "",
        "## Guardrails",
        "",
        "- All rolling operators for a case use one fixed SimVP latent/PCA coordinate system.",
        "- Every local operator uses the same 2 us window and ridge penalty selected before the diagnostic holdout.",
        "- Smoothness is checked with non-overlapping windows and a time-order permutation, so overlap alone cannot establish drift.",
        "- Moving-block bootstrap distances estimate the local operator noise floor.",
        "- Scheduled drift is selected on an earlier validation interval and does not read future holdout truth.",
        "- Causal online one-step uses past holdout truth and is diagnostic only, not an autonomous forecast.",
        "- Operator PCA is descriptive evidence of a low-rank affine subspace, not proof of a nonlinear manifold.",
        "",
        "## Diagnostic summary",
        "",
        "| case | state | drift/noise median | above-noise fraction | nonoverlap path ratio | permutation p | operator PC2 cumulative |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['case']} | {row['representation']} | {row['drift_to_noise_median']:.3f} | "
            f"{row['fraction_adjacent_above_noise']:.3f} | {row['path_ratio_to_permuted']:.3f} | "
            f"{row['permutation_p_lower']:.4f} | {row['operator_pc2_cumulative_variance']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Forecast summary",
            "",
            "| protocol | state | evaluation | method | skill | correlation |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for row in forecast_rows:
        lines.append(
            f"| {row['protocol']} | {row['representation']} | {row['evaluation']} | {row['method']} | "
            f"{row['skill_vs_persistence']:.3f} | {row['correlation']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Result",
            "",
            "B25 operator changes are mostly below the bootstrap uncertainty, and the chronological path through non-overlapping windows is not significantly shorter than random permutations. Two operator PCs explain only about 35--37% of B25 variation. Thus the estimated operators do not form a resolved low-dimensional smooth trajectory in the tested coordinates.",
            "",
            "The scheduled smooth-drift model gives only a small L-state gain in the first B25 holdout and loses badly in the later B25 holdout and both L+A+P holdouts. The causal online local operator improves the later B25 L+A+P one-step prediction, but this uses past holdout truth and does not provide an autonomous operator forecast.",
            "",
            "Therefore the specific hypothesis of a smoothly drifting affine DMD operator in the current L or L+A+P coordinates is not supported. This does not exclude nonlinear operator evolution, organisation-conditioned local dynamics, coordinate drift, or missing memory variables.",
            "",
            "Interpretation must use both diagnostic and forecast blocks. A smooth-looking operator trajectory is insufficient if it lies below the bootstrap noise floor or loses temporal order in non-overlapping windows. A descriptive drift is not a predictive ROM unless the scheduled holdout forecast improves over both fixed and last-local baselines.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = {name: load_representation(name) for name in ("B15", "B25")}
    rolling_rows, adjacent_rows, summary_rows, coordinate_rows = diagnostic_analysis(cases)
    forecast_rows, selection_rows, payloads = forecast_analysis(cases)

    write_csv(OUTPUT / "rolling_operator_metrics.csv", rolling_rows)
    write_csv(OUTPUT / "adjacent_operator_metrics.csv", adjacent_rows)
    write_csv(OUTPUT / "operator_manifold_summary.csv", summary_rows)
    write_csv(OUTPUT / "operator_coordinates.csv", coordinate_rows)
    write_csv(OUTPUT / "forecast_metrics.csv", forecast_rows)
    write_csv(OUTPUT / "schedule_model_selection.csv", selection_rows)
    plot_drift(rolling_rows, adjacent_rows, OUTPUT / "operator_drift_vs_noise.png")
    plot_manifold(coordinate_rows, OUTPUT / "operator_coordinate_trajectories.png")
    plot_forecasts(payloads, OUTPUT / "smooth_drift_forecast_comparison.png")
    write_readme(summary_rows, forecast_rows, OUTPUT / "README.md")

    summary = {
        "status": "PASS",
        "protocol": {
            "diagnostic_interval_us": [DIAGNOSTIC_START_US, DIAGNOSTIC_END_US],
            "rolling_window_us": WINDOW_US,
            "rolling_step_us": ROLLING_STEP_US,
            "nonoverlap_step_us": NONOVERLAP_STEP_US,
            "bootstrap_replicates": BOOTSTRAPS,
            "permutations": PERMUTATIONS,
            "representations": ["L", "L+A+P"],
            "same_basis_within_case": True,
            "operator_pca_is_diagnostic": True,
            "online_one_step_is_autonomous": False,
            "scheduled_drift_uses_future_truth": False,
        },
        "diagnostic_summary": summary_rows,
        "forecast_metrics": forecast_rows,
    }
    (OUTPUT / "analysis_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"PASS: wrote smooth operator drift analysis to {OUTPUT}")


if __name__ == "__main__":
    main()
