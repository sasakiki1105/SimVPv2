#!/usr/bin/env python3
"""Test observable-dependent local closure in the RadAz E and B sweeps.

The analysis converts the preceding nearest-neighbour result into an actual
lead-time forecast without refitting a large latent ROM.  For each modal
transport observable it compares persistence, constant-velocity
extrapolation, an affine T-only model, and affine (T, dT/dt) models.

All derivatives are causal.  Ridge strength and the optional derivative lag
are selected on a chronological validation interval.  The fixed protocol is
12--18 us train, 18--20 us validation, and 20--30 us test.  Five rolling
protocols use four microseconds of history and two microseconds of future.
The forecasts are observed-state lead-time forecasts, not autonomous rollouts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze_radaz_b25_temporal_switching_rom as magnetic
import analyze_radaz_e25_carrier_state_closure as carrier
import analyze_radaz_electric_sweep_predictive_geometry_controls as geometry
import analyze_radaz_local_rom_closure_map as local_map


ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT.parent
DEFAULT_OUTPUT = (
    RESEARCH
    / "research_results"
    / "2D_RadAz"
    / "SimVPv2"
    / "workdirs"
    / "analyze_radaz_observable_dependent_closure"
)

DT_US = 0.015
E_VALUES = (10, 20, 25, 30, 40)
B_VALUES = (10, 15, 20, 25, 30)
HORIZONS_US = (0.30, 0.60, 1.20)
PRIMARY_HORIZON_US = 1.20
DERIVATIVE_LAGS = (1, 10, 20)
RIDGES = (0.0, 1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0e-1, 1.0)

FIXED_TRAIN = (12.0, 18.0)
FIXED_VALIDATION = (18.0, 20.0)
FIXED_TEST = (20.0, 30.0)
ROLLING_FORECAST_STARTS = (20.0, 22.0, 24.0, 26.0, 28.0)
ROLLING_HISTORY_US = 4.0
# The primary lead time is 1.2 us, so the chronological validation tail must
# be longer than that horizon to contain a nonempty set of forecast pairs.
ROLLING_VALIDATION_US = 1.5
ROLLING_TEST_US = 2.0

ANALOG_STEP_FRAMES = 10
ANALOG_K = 10
ANALOG_THEILER_US = 1.0
CORRELATION_MIN = 0.50
STD_RATIO_MIN = 0.50
STD_RATIO_MAX = 1.50

MODEL_ORDER = (
    "persistence",
    "constant_velocity_lag1",
    "constant_velocity_selected",
    "affine_T",
    "affine_T_dT_lag1",
    "affine_T_dT_selected",
)
MODEL_LABELS = {
    "persistence": "persistence",
    "constant_velocity_lag1": "constant velocity (15 ns)",
    "constant_velocity_selected": "constant velocity (validated lag)",
    "affine_T": "affine T",
    "affine_T_dT_lag1": "affine T+dT/dt (15 ns)",
    "affine_T_dT_selected": "affine T+dT/dt (validated lag)",
}
MODEL_COLORS = {
    "persistence": "#777777",
    "constant_velocity_lag1": "#56b4e9",
    "constant_velocity_selected": "#0072b2",
    "affine_T": "#e69f00",
    "affine_T_dT_lag1": "#cc79a7",
    "affine_T_dT_selected": "#009e73",
}

E_OBSERVABLES = ("joint_selected", "mtsi_band", "ecdi_band")
B_OBSERVABLES = ("joint_selected", "long_band", "mtsi_band", "ecdi_band")
OBSERVABLE_LABELS = {
    "joint_selected": "joint transport",
    "long_band": "long-wave transport",
    "mtsi_band": "MTSI-band transport",
    "ecdi_band": "ECDI-band transport",
}


@dataclass
class CaseData:
    sweep: str
    condition: float
    label: str
    time_us: np.ndarray
    observables: dict[str, np.ndarray]
    metadata: dict


@dataclass
class AffineModel:
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    coefficients: np.ndarray

    def predict(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        normalized = (matrix - self.x_mean) / self.x_scale
        augmented = np.column_stack((np.ones(len(normalized)), normalized))
        result = augmented @ self.coefficients
        return result * self.y_scale + self.y_mean


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


def ensure_column(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[:, None] if array.ndim == 1 else array


def causal_secant(values: np.ndarray, lag: int) -> np.ndarray:
    array = ensure_column(values)
    result = np.empty_like(array)
    result[lag:] = (array[lag:] - array[:-lag]) / (lag * DT_US)
    result[:lag] = result[lag]
    return result


def interval_indices(
    time_us: np.ndarray,
    start_us: float,
    end_us: float,
    horizon_frames: int,
) -> np.ndarray:
    indices = np.flatnonzero(
        (time_us >= start_us - 1.0e-9)
        & (time_us <= end_us + 1.0e-9)
    )
    indices = indices[indices + horizon_frames < len(time_us)]
    return indices[
        time_us[indices + horizon_frames] <= end_us + 1.0e-9
    ]


def current_indices(
    time_us: np.ndarray,
    start_us: float,
    end_us: float,
    horizon_frames: int,
) -> np.ndarray:
    indices = np.flatnonzero(
        (time_us >= start_us - 1.0e-9)
        & (time_us <= end_us + 1.0e-9)
    )
    indices = indices[indices + horizon_frames < len(time_us)]
    return indices[
        time_us[indices + horizon_frames] <= end_us + 1.0e-9
    ]


def fit_affine(
    x: np.ndarray,
    y: np.ndarray,
    ridge: float,
) -> AffineModel:
    x = ensure_column(x)
    y = ensure_column(y)
    x_mean = np.mean(x, axis=0)
    x_scale = np.std(x, axis=0)
    x_scale = np.where(x_scale > 1.0e-12, x_scale, 1.0)
    y_mean = np.mean(y, axis=0)
    y_scale = np.std(y, axis=0)
    y_scale = np.where(y_scale > 1.0e-12, y_scale, 1.0)
    normalized_x = (x - x_mean) / x_scale
    normalized_y = (y - y_mean) / y_scale
    augmented = np.column_stack((np.ones(len(normalized_x)), normalized_x))
    penalty = np.eye(augmented.shape[1])
    penalty[0, 0] = 0.0
    gram = augmented.T @ augmented + ridge * penalty
    rhs = augmented.T @ normalized_y
    try:
        coefficients = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    return AffineModel(
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
        coefficients=coefficients,
    )


def mse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((ensure_column(truth) - ensure_column(prediction)) ** 2))


def correlation(truth: np.ndarray, prediction: np.ndarray) -> float:
    x = ensure_column(truth).ravel()
    y = ensure_column(prediction).ravel()
    if np.std(x) <= 1.0e-14 or np.std(y) <= 1.0e-14:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    persistence: np.ndarray,
    history_mean: np.ndarray,
) -> dict:
    truth = ensure_column(truth)
    prediction = ensure_column(prediction)
    persistence = ensure_column(persistence)
    history_mean = ensure_column(history_mean)
    current_mse = mse(truth, prediction)
    persistence_mse = mse(truth, persistence)
    mean_mse = mse(truth, np.broadcast_to(history_mean, truth.shape))
    truth_std = max(float(np.std(truth)), 1.0e-30)
    prediction_std = float(np.std(prediction))
    return {
        "mse": current_mse,
        "mae": float(np.mean(np.abs(truth - prediction))),
        "nrmse_by_truth_std": math.sqrt(current_mse) / truth_std,
        "correlation": correlation(truth, prediction),
        "skill_vs_persistence": 1.0 - current_mse / max(persistence_mse, 1.0e-30),
        "skill_vs_history_mean": 1.0 - current_mse / max(mean_mse, 1.0e-30),
        "std_ratio": prediction_std / truth_std,
        "normalized_bias": float(np.mean(prediction - truth)) / truth_std,
        "persistence_mse": persistence_mse,
        "history_mean_mse": mean_mse,
    }


def design(values: np.ndarray, derivative: np.ndarray | None) -> np.ndarray:
    values = ensure_column(values)
    if derivative is None:
        return values
    return np.column_stack((values, ensure_column(derivative)))


def validation_choice(
    values: np.ndarray,
    derivatives: dict[int, np.ndarray],
    time_us: np.ndarray,
    horizon_frames: int,
    train_start: float,
    train_end: float,
    validation_start: float,
    validation_end: float,
    use_derivative: bool,
    lags: tuple[int, ...],
) -> tuple[int, float, float]:
    train = interval_indices(time_us, train_start, train_end, horizon_frames)
    validation = interval_indices(
        time_us, validation_start, validation_end, horizon_frames
    )
    if len(train) < 20 or len(validation) < 10:
        raise ValueError("Insufficient chronological train/validation pairs")
    best: tuple[float, int, float] | None = None
    for lag in lags:
        derivative = derivatives[lag] if use_derivative else None
        x_train = design(values[train], None if derivative is None else derivative[train])
        x_validation = design(
            values[validation],
            None if derivative is None else derivative[validation],
        )
        y_train = values[train + horizon_frames]
        y_validation = values[validation + horizon_frames]
        for ridge in RIDGES:
            model = fit_affine(x_train, y_train, ridge)
            score = mse(y_validation, model.predict(x_validation))
            candidate = (score, lag, ridge)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise RuntimeError("Validation search did not produce a model")
    return best[1], best[2], best[0]


def velocity_lag_choice(
    values: np.ndarray,
    derivatives: dict[int, np.ndarray],
    time_us: np.ndarray,
    horizon_us: float,
    horizon_frames: int,
    validation_start: float,
    validation_end: float,
) -> tuple[int, float]:
    validation = interval_indices(
        time_us, validation_start, validation_end, horizon_frames
    )
    best: tuple[float, int] | None = None
    for lag in DERIVATIVE_LAGS:
        prediction = values[validation] + horizon_us * derivatives[lag][validation]
        score = mse(values[validation + horizon_frames], prediction)
        candidate = (score, lag)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise RuntimeError("Velocity-lag validation failed")
    return best[1], best[0]


def protocol_predictions(
    values: np.ndarray,
    time_us: np.ndarray,
    horizon_us: float,
    train_start: float,
    train_end: float,
    validation_start: float,
    validation_end: float,
    test_start: float,
    test_end: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict]:
    values = ensure_column(values)
    horizon_frames = int(round(horizon_us / DT_US))
    derivatives = {
        lag: causal_secant(values, lag) for lag in DERIVATIVE_LAGS
    }
    t_lag, t_ridge, t_validation_mse = validation_choice(
        values,
        derivatives,
        time_us,
        horizon_frames,
        train_start,
        train_end,
        validation_start,
        validation_end,
        False,
        (1,),
    )
    del t_lag
    td_lag1, td_ridge1, td_validation_mse1 = validation_choice(
        values,
        derivatives,
        time_us,
        horizon_frames,
        train_start,
        train_end,
        validation_start,
        validation_end,
        True,
        (1,),
    )
    td_lag, td_ridge, td_validation_mse = validation_choice(
        values,
        derivatives,
        time_us,
        horizon_frames,
        train_start,
        train_end,
        validation_start,
        validation_end,
        True,
        DERIVATIVE_LAGS,
    )
    velocity_lag, velocity_validation_mse = velocity_lag_choice(
        values,
        derivatives,
        time_us,
        horizon_us,
        horizon_frames,
        validation_start,
        validation_end,
    )

    pretest = interval_indices(
        time_us, train_start, validation_end, horizon_frames
    )
    test = current_indices(time_us, test_start, test_end, horizon_frames)
    if len(pretest) < 20 or len(test) < 10:
        raise ValueError("Insufficient refit/test pairs")

    t_model = fit_affine(
        design(values[pretest], None),
        values[pretest + horizon_frames],
        t_ridge,
    )
    td_model1 = fit_affine(
        design(values[pretest], derivatives[td_lag1][pretest]),
        values[pretest + horizon_frames],
        td_ridge1,
    )
    td_model = fit_affine(
        design(values[pretest], derivatives[td_lag][pretest]),
        values[pretest + horizon_frames],
        td_ridge,
    )
    predictions = {
        "persistence": values[test].copy(),
        "constant_velocity_lag1": values[test]
        + horizon_us * derivatives[1][test],
        "constant_velocity_selected": values[test]
        + horizon_us * derivatives[velocity_lag][test],
        "affine_T": t_model.predict(design(values[test], None)),
        "affine_T_dT_lag1": td_model1.predict(
            design(values[test], derivatives[td_lag1][test])
        ),
        "affine_T_dT_selected": td_model.predict(
            design(values[test], derivatives[td_lag][test])
        ),
    }
    metadata = {
        "horizon_frames": horizon_frames,
        "affine_T_ridge": t_ridge,
        "affine_T_validation_mse": t_validation_mse,
        "affine_T_dT_lag1_frames": td_lag1,
        "affine_T_dT_lag1_ridge": td_ridge1,
        "affine_T_dT_lag1_validation_mse": td_validation_mse1,
        "affine_T_dT_selected_lag_frames": td_lag,
        "affine_T_dT_selected_lag_us": td_lag * DT_US,
        "affine_T_dT_selected_ridge": td_ridge,
        "affine_T_dT_selected_validation_mse": td_validation_mse,
        "constant_velocity_selected_lag_frames": velocity_lag,
        "constant_velocity_selected_lag_us": velocity_lag * DT_US,
        "constant_velocity_validation_mse": velocity_validation_mse,
        "refit_pairs": len(pretest),
        "test_pairs": len(test),
    }
    return test, values[test + horizon_frames], predictions, metadata


def task_ambiguity(
    values: np.ndarray,
    time_us: np.ndarray,
    history_start: float,
    history_end: float,
    horizon_us: float,
    derivative_lag: int,
) -> dict:
    values = ensure_column(values)
    horizon_frames = int(round(horizon_us / DT_US))
    derivative = causal_secant(values, derivative_lag)
    base = interval_indices(
        time_us, history_start, history_end, horizon_frames
    )[::ANALOG_STEP_FRAMES]
    if len(base) <= ANALOG_K:
        raise ValueError("Too few analogue samples")
    state = np.column_stack((values, derivative))
    mean = np.mean(state[base], axis=0)
    scale = np.std(state[base], axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    state = (state - mean) / scale
    increments = values[base + horizon_frames] - values[base]
    centered = increments - np.mean(increments, axis=0)
    increment_variance = max(float(np.mean(centered * centered)), 1.0e-30)
    ambiguity_values = []
    analogue_errors = []
    persistence_errors = []
    radii = []
    for position, query in enumerate(base):
        candidates = base[
            np.abs(time_us[base] - time_us[query])
            >= ANALOG_THEILER_US - 1.0e-9
        ]
        if len(candidates) < ANALOG_K:
            continue
        distance = np.sqrt(np.mean((state[candidates] - state[query]) ** 2, axis=1))
        order = np.argsort(distance, kind="stable")[:ANALOG_K]
        neighbors = candidates[order]
        local = values[neighbors + horizon_frames] - values[neighbors]
        mean_increment = np.mean(local, axis=0)
        ambiguity_values.append(float(np.mean((local - mean_increment) ** 2)))
        analogue_errors.append(float(np.mean((increments[position] - mean_increment) ** 2)))
        persistence_errors.append(float(np.mean(increments[position] ** 2)))
        radii.append(float(distance[order[-1]]))
    if not ambiguity_values:
        raise ValueError("No valid analogue queries")
    return {
        "queries": len(ambiguity_values),
        "ambiguity_normalized": float(np.mean(ambiguity_values))
        / increment_variance,
        "analogue_error_normalized": float(np.mean(analogue_errors))
        / increment_variance,
        "analogue_skill_vs_persistence": 1.0
        - float(np.mean(analogue_errors))
        / max(float(np.mean(persistence_errors)), 1.0e-30),
        "median_neighbor_radius": float(np.median(radii)),
    }


def load_electric_case(ez_kvm: int) -> CaseData:
    physical_path = (
        geometry.PREVIOUS_OUTPUT.parent
        / "compare_radaz_local_rom_closure_map"
        / "cases"
        / f"E{ez_kvm}kVm"
        / "physical_fourier_targets.h5"
    )
    raw = carrier.load_raw_physical(physical_path)
    row = geometry.load_adaptive_rows()[ez_kvm]
    selected_modes = np.asarray(
        [int(value) for value in row["selected_modes"].split(",")], dtype=int
    )
    modes = np.arange(raw.cross.shape[-1], dtype=int)
    n0 = float(row["ecdi_n0"])
    ratio = modes / n0
    mtsi_modes = modes[(modes >= 1) & (ratio <= 0.60)]
    ecdi_modes = modes[(ratio >= 0.75) & (ratio <= 1.25)]

    def transport(selected: np.ndarray) -> np.ndarray:
        return carrier.transport_from_selected_cross(
            raw.cross[:, :, selected], raw.radial_weights
        )

    observables = {
        "joint_selected": transport(selected_modes),
        "mtsi_band": transport(mtsi_modes),
        "ecdi_band": transport(ecdi_modes),
    }
    return CaseData(
        sweep="E",
        condition=float(ez_kvm),
        label=f"E{ez_kvm}",
        time_us=raw.time_us,
        observables=observables,
        metadata={
            "source": str(physical_path),
            "selected_modes": selected_modes,
            "mtsi_modes": mtsi_modes,
            "ecdi_modes": ecdi_modes,
            "ecdi_n0": n0,
            "transport_definition": "physical density-Ey modal transport",
        },
    )


def load_magnetic_case(b_mt: int) -> CaseData:
    source_path = magnetic.fourier_path(f"B{b_mt}")
    time_us, channels, coefficient = magnetic.unpack_fourier(source_path)
    electron = coefficient[:, channels.index("electron_den")]
    phi = coefficient[:, channels.index("phi")]
    mode = np.arange(phi.shape[-1], dtype=np.float64)
    ey = -1j * mode[None, None, :] * phi
    cross = electron * np.conj(ey)

    def band_transport(lower: int, upper: int) -> np.ndarray:
        upper = min(upper, cross.shape[-1] - 1)
        selected = cross[:, :, lower : upper + 1]
        return -2.0 * np.real(np.sum(selected, axis=(1, 2))) / (b_mt * 1.0e-3)

    long_transport = band_transport(*magnetic.MODE_BANDS["long"])
    mtsi_transport = band_transport(*magnetic.MODE_BANDS["mtsi"])
    ecdi_transport = band_transport(*magnetic.MODE_BANDS["ecdi"])
    observables = {
        "joint_selected": mtsi_transport + ecdi_transport,
        "long_band": long_transport,
        "mtsi_band": mtsi_transport,
        "ecdi_band": ecdi_transport,
    }
    return CaseData(
        sweep="B",
        condition=float(b_mt),
        label=f"B{b_mt}",
        time_us=time_us,
        observables=observables,
        metadata={
            "source": str(source_path),
            "mode_bands": magnetic.MODE_BANDS,
            "transport_definition": (
                "B20-normalized density-Ey Fourier transport proxy; "
                "compare dynamics within each case, not absolute E/B magnitudes"
            ),
        },
    )


def evaluate_fixed_case(
    case: CaseData,
) -> tuple[list[dict], list[dict], dict]:
    metric_rows = []
    prediction_rows = []
    protocol_meta = {}
    for observable, values in case.observables.items():
        for horizon_us in HORIZONS_US:
            test, truth, predictions, metadata = protocol_predictions(
                values,
                case.time_us,
                horizon_us,
                FIXED_TRAIN[0],
                FIXED_TRAIN[1],
                FIXED_VALIDATION[0],
                FIXED_VALIDATION[1],
                FIXED_TEST[0],
                min(FIXED_TEST[1], float(case.time_us[-1])),
            )
            history = (case.time_us >= FIXED_TRAIN[0] - 1.0e-9) & (
                case.time_us <= FIXED_VALIDATION[1] + 1.0e-9
            )
            history_mean = np.mean(ensure_column(values)[history], axis=0)
            persistence = predictions["persistence"]
            for model_name in MODEL_ORDER:
                row = {
                    "sweep": case.sweep,
                    "condition": case.condition,
                    "case": case.label,
                    "observable": observable,
                    "horizon_us": horizon_us,
                    "model": model_name,
                    "test_start_us": float(case.time_us[test[0]]),
                    "test_last_current_us": float(case.time_us[test[-1]]),
                    "test_samples": len(test),
                    **metadata,
                    **metrics(
                        truth,
                        predictions[model_name],
                        persistence,
                        history_mean,
                    ),
                }
                metric_rows.append(row)
            protocol_meta[f"{observable}_{horizon_us:g}us"] = metadata
            if math.isclose(horizon_us, PRIMARY_HORIZON_US) and observable == "joint_selected":
                for position, index in enumerate(test):
                    row = {
                        "sweep": case.sweep,
                        "condition": case.condition,
                        "case": case.label,
                        "observable": observable,
                        "horizon_us": horizon_us,
                        "current_time_us": float(case.time_us[index]),
                        "target_time_us": float(
                            case.time_us[index + int(round(horizon_us / DT_US))]
                        ),
                        "truth": float(truth[position, 0]),
                    }
                    for model_name in MODEL_ORDER:
                        row[model_name] = float(predictions[model_name][position, 0])
                    prediction_rows.append(row)
    return metric_rows, prediction_rows, protocol_meta


def evaluate_rolling_case(case: CaseData) -> list[dict]:
    rows = []
    for observable, values in case.observables.items():
        for forecast_start in ROLLING_FORECAST_STARTS:
            test_end = min(
                forecast_start + ROLLING_TEST_US,
                float(case.time_us[-1]),
            )
            history_start = forecast_start - ROLLING_HISTORY_US
            train_end = forecast_start - ROLLING_VALIDATION_US
            test, truth, predictions, metadata = protocol_predictions(
                values,
                case.time_us,
                PRIMARY_HORIZON_US,
                history_start,
                train_end,
                train_end,
                forecast_start,
                forecast_start,
                test_end,
            )
            ambiguity_lag1 = task_ambiguity(
                values,
                case.time_us,
                history_start,
                forecast_start,
                PRIMARY_HORIZON_US,
                1,
            )
            selected_lag = int(
                metadata["affine_T_dT_selected_lag_frames"]
            )
            ambiguity_selected = task_ambiguity(
                values,
                case.time_us,
                history_start,
                forecast_start,
                PRIMARY_HORIZON_US,
                selected_lag,
            )
            history = (case.time_us >= history_start - 1.0e-9) & (
                case.time_us <= forecast_start + 1.0e-9
            )
            history_mean = np.mean(ensure_column(values)[history], axis=0)
            persistence = predictions["persistence"]
            row = {
                "sweep": case.sweep,
                "condition": case.condition,
                "case": case.label,
                "observable": observable,
                "forecast_start_us": forecast_start,
                "history_start_us": history_start,
                "train_end_us": train_end,
                "test_end_us": test_end,
                "horizon_us": PRIMARY_HORIZON_US,
                **metadata,
            }
            for key, value in ambiguity_lag1.items():
                row[f"lag1_{key}"] = value
            for key, value in ambiguity_selected.items():
                row[f"selected_lag_{key}"] = value
            for model_name in MODEL_ORDER:
                current = metrics(
                    truth,
                    predictions[model_name],
                    persistence,
                    history_mean,
                )
                for key, value in current.items():
                    row[f"{model_name}_{key}"] = value
            rows.append(row)
    return rows


def safe_spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3 or np.std(x) <= 1.0e-14 or np.std(y) <= 1.0e-14:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def rolling_correlations(rows: list[dict]) -> list[dict]:
    output = []
    groups: list[tuple[str, str | None]] = [("all", None)]
    groups.extend((sweep, None) for sweep in ("E", "B"))
    groups.extend((sweep, observable) for sweep, observables in (("E", E_OBSERVABLES), ("B", B_OBSERVABLES)) for observable in observables)
    for sweep, observable in groups:
        selected = [
            row
            for row in rows
            if (sweep == "all" or row["sweep"] == sweep)
            and (observable is None or row["observable"] == observable)
        ]
        for subset, starts in (
            ("all_overlapping_windows", None),
            ("nonoverlapping_20_24_28us", {20.0, 24.0, 28.0}),
        ):
            local = selected if starts is None else [
                row for row in selected if row["forecast_start_us"] in starts
            ]
            for ambiguity_key, model_name in (
                ("lag1_ambiguity_normalized", "affine_T_dT_lag1"),
                (
                    "selected_lag_ambiguity_normalized",
                    "affine_T_dT_selected",
                ),
            ):
                x = [row[ambiguity_key] for row in local]
                y = [row[f"{model_name}_skill_vs_persistence"] for row in local]
                output.append(
                    {
                        "sweep": sweep,
                        "observable": observable or "all",
                        "window_subset": subset,
                        "model": model_name,
                        "ambiguity_definition": ambiguity_key,
                        "samples": len(local),
                        "spearman_ambiguity_vs_skill": safe_spearman(x, y),
                        "note": "descriptive only; windows and cases are not iid",
                    }
                )
    return output


def focal_fixed(rows: list[dict]) -> list[dict]:
    selected = [
        row for row in rows if math.isclose(row["horizon_us"], PRIMARY_HORIZON_US)
    ]
    output = []
    keys = sorted({(row["sweep"], row["condition"], row["case"], row["observable"]) for row in selected})
    for sweep, condition, case, observable in keys:
        local = {
            row["model"]: row
            for row in selected
            if row["sweep"] == sweep
            and row["condition"] == condition
            and row["observable"] == observable
        }
        output.append(
            {
                "sweep": sweep,
                "condition": condition,
                "case": case,
                "observable": observable,
                "horizon_us": PRIMARY_HORIZON_US,
                "T_skill_vs_persistence": local["affine_T"]["skill_vs_persistence"],
                "T_correlation": local["affine_T"]["correlation"],
                "T_dT_lag1_skill_vs_persistence": local["affine_T_dT_lag1"]["skill_vs_persistence"],
                "T_dT_lag1_correlation": local["affine_T_dT_lag1"]["correlation"],
                "T_dT_selected_skill_vs_persistence": local["affine_T_dT_selected"]["skill_vs_persistence"],
                "T_dT_selected_correlation": local["affine_T_dT_selected"]["correlation"],
                "T_dT_selected_std_ratio": local["affine_T_dT_selected"]["std_ratio"],
                "gain_selected_T_dT_vs_T": local["affine_T_dT_selected"]["skill_vs_persistence"]
                - local["affine_T"]["skill_vs_persistence"],
                "selected_derivative_lag_us": local["affine_T_dT_selected"]["affine_T_dT_selected_lag_us"],
                "constant_velocity_selected_skill": local["constant_velocity_selected"]["skill_vs_persistence"],
                "fixed_closure_pass": bool(
                    local["affine_T_dT_selected"]["skill_vs_persistence"] > 0.0
                    and local["affine_T_dT_selected"]["correlation"]
                    >= CORRELATION_MIN
                    and STD_RATIO_MIN
                    <= local["affine_T_dT_selected"]["std_ratio"]
                    <= STD_RATIO_MAX
                ),
            }
        )
    return output


def closure_map(
    focal: list[dict], rolling: list[dict]
) -> list[dict]:
    output = []
    for row in focal:
        local = [
            item
            for item in rolling
            if item["sweep"] == row["sweep"]
            and item["condition"] == row["condition"]
            and item["observable"] == row["observable"]
        ]
        output.append(
            {
                **row,
                "rolling_windows": len(local),
                "rolling_median_lag1_ambiguity": float(
                    np.median([
                        item["lag1_ambiguity_normalized"] for item in local
                    ])
                ),
                "rolling_median_selected_lag_ambiguity": float(
                    np.median([
                        item["selected_lag_ambiguity_normalized"]
                        for item in local
                    ])
                ),
                "rolling_median_T_skill": float(
                    np.median([item["affine_T_skill_vs_persistence"] for item in local])
                ),
                "rolling_median_T_dT_selected_skill": float(
                    np.median([
                        item["affine_T_dT_selected_skill_vs_persistence"]
                        for item in local
                    ])
                ),
                "rolling_median_T_dT_selected_correlation": float(
                    np.nanmedian([
                        item["affine_T_dT_selected_correlation"]
                        for item in local
                    ])
                ),
                "rolling_positive_skill_windows": int(
                    np.sum([
                        item["affine_T_dT_selected_skill_vs_persistence"] > 0.0
                        for item in local
                    ])
                ),
                "rolling_closure_pass_windows": int(
                    np.sum([
                        item["affine_T_dT_selected_skill_vs_persistence"] > 0.0
                        and item["affine_T_dT_selected_correlation"]
                        >= CORRELATION_MIN
                        and STD_RATIO_MIN
                        <= item["affine_T_dT_selected_std_ratio"]
                        <= STD_RATIO_MAX
                        for item in local
                    ])
                ),
                "rolling_closure_pass_fraction": float(
                    np.mean([
                        item["affine_T_dT_selected_skill_vs_persistence"] > 0.0
                        and item["affine_T_dT_selected_correlation"]
                        >= CORRELATION_MIN
                        and STD_RATIO_MIN
                        <= item["affine_T_dT_selected_std_ratio"]
                        <= STD_RATIO_MAX
                        for item in local
                    ])
                ),
                "rolling_median_gain_T_dT_vs_T": float(
                    np.median([
                        item["affine_T_dT_selected_skill_vs_persistence"]
                        - item["affine_T_skill_vs_persistence"]
                        for item in local
                    ])
                ),
            }
        )
    return output


def metric_lookup(
    rows: list[dict], sweep: str, condition: float, observable: str, model: str
) -> dict:
    return next(
        row
        for row in rows
        if row["sweep"] == sweep
        and row["condition"] == condition
        and row["observable"] == observable
        and row["model"] == model
        and math.isclose(row["horizon_us"], PRIMARY_HORIZON_US)
    )


def plot_fixed_heatmaps(rows: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    models = ("constant_velocity_selected", "affine_T", "affine_T_dT_lag1", "affine_T_dT_selected")
    for axis, sweep, conditions in zip(axes, ("E", "B"), (E_VALUES, B_VALUES)):
        matrix = np.asarray([
            [metric_lookup(rows, sweep, value, "joint_selected", model)["skill_vs_persistence"] for model in models]
            for value in conditions
        ])
        image = axis.imshow(matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
        axis.set_xticks(range(len(models)), [MODEL_LABELS[model] for model in models], rotation=28, ha="right")
        axis.set_yticks(range(len(conditions)), [f"{value} kV/m" if sweep == "E" else f"{value} mT" for value in conditions])
        axis.set_title(f"{sweep} sweep: joint transport, 1.2 us")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(column_index, row_index, f"{matrix[row_index, column_index]:.2f}", ha="center", va="center", fontsize=8)
        figure.colorbar(image, ax=axis, shrink=0.82, label="MSE skill vs persistence (color clipped to [-1, 1])")
    figure.tight_layout()
    figure.savefig(output / "fixed_minimal_predictor_skill_heatmaps.png", bbox_inches="tight")
    plt.close(figure)


def plot_rolling_relation(rows: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
    for axis, sweep, conditions in zip(axes, ("E", "B"), (E_VALUES, B_VALUES)):
        colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(conditions)))
        for color, condition in zip(colors, conditions):
            local = [
                row
                for row in rows
                if row["sweep"] == sweep
                and row["condition"] == condition
                and row["observable"] == "joint_selected"
            ]
            axis.scatter(
                [row["selected_lag_ambiguity_normalized"] for row in local],
                [row["affine_T_dT_selected_skill_vs_persistence"] for row in local],
                s=45,
                color=color,
                label=f"{condition:g}",
                alpha=0.85,
            )
        all_local = [row for row in rows if row["sweep"] == sweep and row["observable"] == "joint_selected"]
        rho = safe_spearman(
            [row["selected_lag_ambiguity_normalized"] for row in all_local],
            [row["affine_T_dT_selected_skill_vs_persistence"] for row in all_local],
        )
        axis.axhline(0.0, color="#555555", linewidth=1)
        axis.set_xlabel("history-only A_J(T,dT/dt)")
        axis.set_title(f"{sweep} sweep joint transport, rho={rho:.2f}")
        axis.grid(alpha=0.25)
        axis.set_ylim(-2.0, 1.2)
        axis.text(
            0.02,
            0.03,
            "display clipped below -2; CSV keeps exact values",
            transform=axis.transAxes,
            fontsize=8,
            color="#555555",
        )
        axis.legend(title="kV/m" if sweep == "E" else "mT", loc="lower right", ncol=2)
    axes[0].set_ylabel("next-window affine T+dT/dt skill vs persistence")
    figure.tight_layout()
    figure.savefig(output / "rolling_ambiguity_vs_forecast_skill.png", bbox_inches="tight")
    plt.close(figure)


def plot_closure_map(rows: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    for axis, sweep, conditions, observables in zip(
        axes,
        ("E", "B"),
        (E_VALUES, B_VALUES),
        (E_OBSERVABLES, B_OBSERVABLES),
    ):
        matrix = np.asarray([
            [
                next(
                    row["rolling_closure_pass_fraction"]
                    for row in rows
                    if row["sweep"] == sweep
                    and row["condition"] == condition
                    and row["observable"] == observable
                )
                for observable in observables
            ]
            for condition in conditions
        ])
        image = axis.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
        axis.set_xticks(range(len(observables)), [OBSERVABLE_LABELS[item] for item in observables], rotation=25, ha="right")
        axis.set_yticks(range(len(conditions)), [f"{value} kV/m" if sweep == "E" else f"{value} mT" for value in conditions])
        axis.set_title(f"{sweep} sweep observable-dependent closure")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(column_index, row_index, f"{matrix[row_index, column_index]:.2f}", ha="center", va="center", fontsize=8)
        figure.colorbar(image, ax=axis, shrink=0.82, label="rolling closure-pass fraction")
    figure.tight_layout()
    figure.savefig(output / "observable_dependent_closure_map.png", bbox_inches="tight")
    plt.close(figure)


def plot_fixed_timeseries(rows: list[dict], output: Path) -> None:
    selected_cases = (("E", 20.0), ("E", 25.0), ("E", 40.0), ("B", 15.0), ("B", 25.0), ("B", 30.0))
    figure, axes = plt.subplots(3, 2, figsize=(13.5, 10.0), sharex=False)
    for axis, (sweep, condition) in zip(axes.ravel(), selected_cases):
        local = [row for row in rows if row["sweep"] == sweep and row["condition"] == condition]
        time = np.asarray([row["target_time_us"] for row in local])
        truth = np.asarray([row["truth"] for row in local])
        center = np.mean(truth)
        scale = max(float(np.std(truth)), 1.0e-30)
        axis.plot(time, (truth - center) / scale, color="#111111", linewidth=1.5, label="truth")
        axis.plot(time, (np.asarray([row["persistence"] for row in local]) - center) / scale, color=MODEL_COLORS["persistence"], linewidth=1.0, label="persistence")
        axis.plot(time, (np.asarray([row["affine_T_dT_selected"] for row in local]) - center) / scale, color=MODEL_COLORS["affine_T_dT_selected"], linewidth=1.1, label="T+dT/dt")
        axis.set_title(f"{sweep}{condition:g}, joint transport")
        axis.set_xlabel("target time [us]")
        axis.set_ylabel("test-standardized transport")
        axis.grid(alpha=0.2)
        axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output / "fixed_joint_transport_forecasts.png", bbox_inches="tight")
    plt.close(figure)


def summarize_readme(
    output: Path,
    closure_rows: list[dict],
    correlations: list[dict],
) -> None:
    lines = [
        "# Observable-dependent closure in RadAz E and B sweeps",
        "",
        "This analysis turns the preceding nearest-neighbour diagnostic into a chronological observed-state lead-time forecast. It does not retrain SimVP and it does not fit a large latent ROM.",
        "",
        "## Frozen protocol",
        "",
        "- Fixed: 12--18 us train, 18--20 us validation, refit on 12--20 us, test on 20--30 us.",
        "- Rolling: five four-us histories ending at 20, 22, 24, 26, and 28 us; the final 1.5 us selects hyperparameters; the next two us are held out.",
        "- Horizons: 0.30, 0.60, and 1.20 us; the closure map uses 1.20 us.",
        "- Derivatives are backward-only. The exact previous ambiguity state uses 15 ns; a validation-only choice among 15, 150, and 300 ns is also reported.",
        "- These are lead-time forecasts supplied with the observed current T and causal derivative at each query, not autonomous 10-us rollouts.",
        "- Rolling windows overlap. Their rank correlations are descriptive and receive no iid p-value.",
        "",
        "## Fixed 1.2-us joint-transport result",
        "",
        "| sweep | condition | affine T skill/corr | affine T+dT selected skill/corr/std | gain from dT | selected lag [us] | pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in closure_rows:
        if row["observable"] != "joint_selected":
            continue
        lines.append(
            f"| {row['sweep']} | {row['condition']:g} | "
            f"{row['T_skill_vs_persistence']:.3f}/{row['T_correlation']:.3f} | "
            f"{row['T_dT_selected_skill_vs_persistence']:.3f}/{row['T_dT_selected_correlation']:.3f}/{row['T_dT_selected_std_ratio']:.3f} | "
            f"{row['gain_selected_T_dT_vs_T']:+.3f} | "
            f"{row['selected_derivative_lag_us']:.3f} | "
            f"{int(row['fixed_closure_pass'])} |"
        )
    lines.extend(
        [
            "",
            "## Rolling 1.2-us joint-transport result",
            "",
            "| sweep | condition | median A(lag1/selected) | median T skill | median T+dT skill/corr | skill-positive | closure pass | median dT gain |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in closure_rows:
        if row["observable"] != "joint_selected":
            continue
        lines.append(
            f"| {row['sweep']} | {row['condition']:g} | "
            f"{row['rolling_median_lag1_ambiguity']:.3f}/{row['rolling_median_selected_lag_ambiguity']:.3f} | "
            f"{row['rolling_median_T_skill']:.3f} | "
            f"{row['rolling_median_T_dT_selected_skill']:.3f}/"
            f"{row['rolling_median_T_dT_selected_correlation']:.3f} | "
            f"{row['rolling_positive_skill_windows']}/5 | "
            f"{row['rolling_closure_pass_windows']}/5 | "
            f"{row['rolling_median_gain_T_dT_vs_T']:+.3f} |"
        )
    lines.extend(["", "## Descriptive ambiguity-skill association", ""])
    for row in correlations:
        if row["observable"] == "joint_selected" and row["window_subset"] == "all_overlapping_windows":
            lines.append(
                f"- {row['sweep']} sweep / {row['model']}: rho={row['spearman_ambiguity_vs_skill']:.3f} over {row['samples']} overlapping windows."
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The exact 15-ns derivative from the preceding ambiguity test adds almost no fixed E25 skill over T alone. E25 passes the fixed closure rule only after validation selects a 150-ns backward secant. The result therefore supports a finite-time velocity coordinate, not yet an instantaneous derivative closure.",
            "- E40 is the robust positive control: the fixed 15-ns T+dT/dt model passes, and the validation-selected rolling model passes all five windows for both joint and ECDI-band transport.",
            "- E25 is partial and observable-dependent: fixed joint and ECDI transport pass, but rolling joint transport passes only one of five windows while ECDI transport passes three of five.",
            "- E20 often beats persistence in MSE while correlation and variance collapse. This is mean-regression, not waveform closure. E30 is horizon/window dependent: short-horizon shape prediction can be good, but the 1.2-us fixed forecast does not beat its unusually strong persistence baseline.",
            "- No B-sweep condition passes joint-transport closure in the fixed test or in any of the five rolling windows. Occasional long-wave or MTSI window passes do not form a condition-wide closure. B15 full-state reducibility and scalar modal-transport reducibility are therefore distinct statements.",
            "- Rolling ambiguity has essentially no monotone association with next-window forecast skill inside either sweep. The earlier five-case rho=-0.90 was a useful screening signal but does not survive this within-condition confirmatory test as a general skill predictor.",
            "",
            "Interpret the closure map by observable and state definition. Positive skill means the minimal model beats persistence; it does not establish full-state Markov closure. See the CSV files and four PNG summaries for all observables, horizons, baselines, and validation-selected lags.",
        ]
    )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for ez_kvm in E_VALUES:
        print(f"[LOAD] E{ez_kvm}", flush=True)
        cases.append(load_electric_case(ez_kvm))
    for b_mt in B_VALUES:
        print(f"[LOAD] B{b_mt}", flush=True)
        cases.append(load_magnetic_case(b_mt))

    fixed_rows = []
    prediction_rows = []
    rolling_rows = []
    provenance = {}
    for case in cases:
        print(f"[FIXED] {case.label}", flush=True)
        local_fixed, local_predictions, protocol_meta = evaluate_fixed_case(case)
        fixed_rows.extend(local_fixed)
        prediction_rows.extend(local_predictions)
        print(f"[ROLLING] {case.label}", flush=True)
        rolling_rows.extend(evaluate_rolling_case(case))
        provenance[case.label] = {
            **case.metadata,
            "time_start_us": float(case.time_us[0]),
            "time_end_us": float(case.time_us[-1]),
            "frames": len(case.time_us),
            "fixed_protocol": protocol_meta,
        }

    focal = focal_fixed(fixed_rows)
    closure_rows = closure_map(focal, rolling_rows)
    correlations = rolling_correlations(rolling_rows)
    write_csv(output / "fixed_forecast_metrics.csv", fixed_rows)
    write_csv(output / "fixed_primary_predictions.csv", prediction_rows)
    write_csv(output / "rolling_window_metrics.csv", rolling_rows)
    write_csv(output / "rolling_ambiguity_skill_correlations.csv", correlations)
    write_csv(output / "observable_dependent_closure_map.csv", closure_rows)

    plot_fixed_heatmaps(fixed_rows, output)
    plot_rolling_relation(rolling_rows, output)
    plot_closure_map(closure_rows, output)
    plot_fixed_timeseries(prediction_rows, output)
    summarize_readme(output, closure_rows, correlations)

    summary = {
        "status": "PASS",
        "purpose": "minimal task-specific forecast and observable-dependent closure map",
        "protocol": {
            "fixed_train_us": FIXED_TRAIN,
            "fixed_validation_us": FIXED_VALIDATION,
            "fixed_test_us": FIXED_TEST,
            "rolling_forecast_starts_us": ROLLING_FORECAST_STARTS,
            "rolling_history_us": ROLLING_HISTORY_US,
            "rolling_validation_us": ROLLING_VALIDATION_US,
            "rolling_test_us": ROLLING_TEST_US,
            "horizons_us": HORIZONS_US,
            "derivative_lags_frames": DERIVATIVE_LAGS,
            "derivative_lags_us": [lag * DT_US for lag in DERIVATIVE_LAGS],
            "ridges": RIDGES,
            "analogue_k": ANALOG_K,
            "analogue_theiler_us": ANALOG_THEILER_US,
            "observed_state_lead_time_forecast_not_autonomous_rollout": True,
            "rolling_statistics_descriptive_not_iid": True,
            "closure_thresholds": {
                "skill_vs_persistence_min_exclusive": 0.0,
                "correlation_min": CORRELATION_MIN,
                "std_ratio_min": STD_RATIO_MIN,
                "std_ratio_max": STD_RATIO_MAX,
            },
        },
        "provenance": provenance,
        "closure_map": closure_rows,
        "rolling_correlations": correlations,
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2), encoding="utf-8"
    )
    print(f"[DONE] {output}", flush=True)


if __name__ == "__main__":
    main()
