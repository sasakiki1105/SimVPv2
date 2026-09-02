#!/usr/bin/env python3
"""Test whether the E40 transport closure is only a periodic-orbit fit.

The previous E40 two-dimensional map was advanced every 150 ns, whereas the
native PIC diagnostic interval is 15 ns.  The dominant E40 transport frequency
is close to 21 MHz and therefore aliases to about 1 MHz on the 150 ns map grid.
This script separates native and stroboscopic prediction and compares:

* a constant-frequency sinusoid fitted without test data,
* an initial-state phase/amplitude-reset sinusoid,
* native 15 ns AR(2),
* linear, quadratic, and cubic (T, V_delta) autonomous maps at 15 ns,
* the same degree-controlled maps at the previous 150 ns map interval.

All ridge choices use 12--18 us for fitting and 18--20 us for autonomous
validation.  Final models are refit on 12--20 us and tested after 20 us.
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
from scipy.optimize import minimize_scalar
from scipy.signal import hilbert

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze_radaz_observable_dependent_closure as closure
import analyze_radaz_transport_phase_geometry as geometry


ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT.parent
DEFAULT_OUTPUT = (
    RESEARCH
    / "research_results"
    / "2D_RadAz"
    / "SimVPv2"
    / "workdirs"
    / "analyze_radaz_e40_periodic_null"
)

DT_US = closure.DT_US
TRAIN = (12.0, 18.0)
VALIDATION = (18.0, 20.0)
REFIT = (12.0, 20.0)
TEST_START_US = 20.0
TEST_END_US = 30.0
NATIVE_STEP = 1
STROBE_STEP = 10
RIDGES = (1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0e-1, 1.0)
DEGREES = (1, 2, 3)
ROLLING_STARTS = (20.0, 22.0, 24.0, 26.0, 28.0)
ROLLING_HISTORY_US = 4.0
ROLLING_HORIZON_US = 1.2

COLORS = {
    "sinusoid_absolute_native": "#e69f00",
    "sinusoid_phase_reset_native": "#56b4e9",
    "AR2_native": "#0072b2",
    "linear2D_native": "#009e73",
    "quadratic2D_native": "#cc79a7",
    "cubic2D_native": "#d55e00",
    "linear2D_strobe": "#009e73",
    "quadratic2D_strobe": "#cc79a7",
    "cubic2D_strobe": "#d55e00",
}


@dataclass
class OscillatorModel:
    frequency_mhz: float
    coefficients: np.ndarray
    time_origin_us: float

    def predict(self, time_us: np.ndarray) -> np.ndarray:
        phase = 2.0 * np.pi * self.frequency_mhz * (
            np.asarray(time_us, dtype=np.float64) - self.time_origin_us
        )
        return (
            self.coefficients[0]
            + self.coefficients[1] * np.cos(phase)
            + self.coefficients[2] * np.sin(phase)
        )

    @property
    def mean(self) -> float:
        return float(self.coefficients[0])


@dataclass
class AR2Model:
    mean: float
    scale: float
    coefficients: np.ndarray
    ridge: float

    def rollout(self, previous: float, current: float, count: int) -> np.ndarray:
        previous_z = (float(previous) - self.mean) / self.scale
        current_z = (float(current) - self.mean) / self.scale
        result = [current_z]
        for _ in range(1, count):
            next_z = (
                self.coefficients[0]
                + self.coefficients[1] * current_z
                + self.coefficients[2] * previous_z
            )
            if not math.isfinite(float(next_z)) or abs(float(next_z)) > 1.0e6:
                result.extend([float("nan")] * (count - len(result)))
                break
            result.append(float(next_z))
            previous_z, current_z = current_z, float(next_z)
        return np.asarray(result, dtype=np.float64) * self.scale + self.mean


@dataclass
class DelayMap:
    degree: int
    ridge: float
    step_frames: int
    state_mean: np.ndarray
    state_scale: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray

    def normalize(self, state: np.ndarray) -> np.ndarray:
        return (np.asarray(state, dtype=np.float64) - self.state_mean) / self.state_scale

    def denormalize(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=np.float64) * self.state_scale + self.state_mean

    def step(self, state_z: np.ndarray) -> np.ndarray:
        features = polynomial_features(np.asarray(state_z).reshape(1, 2), self.degree)
        features = (features - self.feature_mean) / self.feature_scale
        augmented = np.column_stack((np.ones(1), features))
        return (augmented @ self.coefficients).ravel()

    def rollout(self, initial_state: np.ndarray, count: int) -> np.ndarray:
        state_z = self.normalize(initial_state)
        result = [self.denormalize(state_z)]
        for _ in range(1, count):
            state_z = self.step(state_z)
            if not np.all(np.isfinite(state_z)) or np.max(np.abs(state_z)) > 1.0e4:
                result.extend([np.full(2, np.nan)] * (count - len(result)))
                break
            result.append(self.denormalize(state_z))
        return np.asarray(result, dtype=np.float64)


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


def load_e40() -> closure.CaseData:
    return next(case for case in geometry.load_cases() if case.label == "E40")


def interval_indices(time_us: np.ndarray, interval: tuple[float, float]) -> np.ndarray:
    return np.flatnonzero(
        (time_us >= interval[0] - 1.0e-9)
        & (time_us <= interval[1] + 1.0e-9)
    )


def sample_indices(
    time_us: np.ndarray,
    start_us: float,
    end_us: float,
    step_frames: int,
) -> np.ndarray:
    start = int(np.argmin(np.abs(time_us - start_us)))
    end = min(float(end_us), float(time_us[-1]))
    count = int(math.floor((end - float(time_us[start])) / (step_frames * DT_US) + 1.0e-9)) + 1
    indices = start + np.arange(count, dtype=int) * step_frames
    return indices[indices < len(time_us)]


def oscillator_design(time_us: np.ndarray, frequency_mhz: float, origin_us: float) -> np.ndarray:
    phase = 2.0 * np.pi * frequency_mhz * (time_us - origin_us)
    return np.column_stack((np.ones(len(time_us)), np.cos(phase), np.sin(phase)))


def fit_oscillator(
    time_us: np.ndarray,
    transport: np.ndarray,
    interval: tuple[float, float],
) -> OscillatorModel:
    indices = interval_indices(time_us, interval)
    t = time_us[indices]
    y = transport[indices]
    origin = float(t[0])
    centered = y - np.mean(y)
    frequencies = np.fft.rfftfreq(len(y), d=DT_US)
    spectrum = np.abs(np.fft.rfft(centered * np.hanning(len(y)))) ** 2
    spectrum[(frequencies < 0.1) | (frequencies >= 0.98 / (2.0 * DT_US))] = 0.0
    peak_frequency = float(frequencies[int(np.argmax(spectrum))])
    frequency_bin = 1.0 / max(float(t[-1] - t[0]), DT_US)

    def residual(frequency: float) -> float:
        design = oscillator_design(t, frequency, origin)
        coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
        return float(np.mean((y - design @ coefficients) ** 2))

    lower = max(0.05, peak_frequency - 2.0 * frequency_bin)
    upper = min(0.98 / (2.0 * DT_US), peak_frequency + 2.0 * frequency_bin)
    optimized = minimize_scalar(residual, bounds=(lower, upper), method="bounded")
    frequency = float(optimized.x)
    design = oscillator_design(t, frequency, origin)
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    return OscillatorModel(frequency, coefficients, origin)


def phase_reset_oscillator_prediction(
    model: OscillatorModel,
    times: np.ndarray,
    initial_transport: float,
    initial_velocity: float,
) -> np.ndarray:
    omega = 2.0 * np.pi * model.frequency_mhz
    displacement = float(initial_transport) - model.mean
    quadrature = -float(initial_velocity) / max(omega, 1.0e-30)
    amplitude = max(math.hypot(displacement, quadrature), 1.0e-30)
    phase0 = math.atan2(quadrature / amplitude, displacement / amplitude)
    return model.mean + amplitude * np.cos(
        phase0 + omega * (np.asarray(times) - float(times[0]))
    )


def fit_ar2(
    time_us: np.ndarray,
    transport: np.ndarray,
    interval: tuple[float, float],
    ridge: float,
) -> AR2Model:
    target = np.flatnonzero(
        (time_us >= interval[0] - 1.0e-9)
        & (time_us <= interval[1] + 1.0e-9)
    )
    target = target[target >= 2]
    mean = float(np.mean(transport[target]))
    scale = max(float(np.std(transport[target])), 1.0e-30)
    normalized = (transport - mean) / scale
    design = np.column_stack(
        (np.ones(len(target)), normalized[target - 1], normalized[target - 2])
    )
    y = normalized[target]
    penalty = np.eye(3)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + ridge * penalty,
        design.T @ y,
    )
    return AR2Model(mean, scale, coefficients, ridge)


def rollout_ar2(
    model: AR2Model,
    transport: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    start = int(indices[0])
    full_count = int(indices[-1] - start) + 1
    full = model.rollout(transport[start - 1], transport[start], full_count)
    return full[indices - start]


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


def fit_delay_map(
    state: np.ndarray,
    time_us: np.ndarray,
    interval: tuple[float, float],
    degree: int,
    ridge: float,
    step_frames: int,
) -> DelayMap:
    current = np.flatnonzero(
        (time_us >= interval[0] - 1.0e-9)
        & (time_us + step_frames * DT_US <= interval[1] + 1.0e-9)
    )
    current = current[current + step_frames < len(state)]
    mean = np.mean(state[current], axis=0)
    scale = np.std(state[current], axis=0)
    scale = np.where(scale > 1.0e-14, scale, 1.0)
    x = (state[current] - mean) / scale
    y = (state[current + step_frames] - mean) / scale
    raw = polynomial_features(x, degree)
    feature_mean = np.mean(raw, axis=0)
    feature_scale = np.std(raw, axis=0)
    feature_scale = np.where(feature_scale > 1.0e-12, feature_scale, 1.0)
    features = (raw - feature_mean) / feature_scale
    design = np.column_stack((np.ones(len(features)), features))
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + ridge * penalty,
        design.T @ y,
    )
    return DelayMap(
        degree,
        ridge,
        step_frames,
        mean,
        scale,
        feature_mean,
        feature_scale,
        coefficients,
    )


def rollout_delay_map(
    model: DelayMap,
    state: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    prediction = model.rollout(state[int(indices[0])], len(indices))
    return prediction[:, 0]


def validation_mse(truth: np.ndarray, prediction: np.ndarray) -> float:
    if len(truth) != len(prediction) or not np.all(np.isfinite(prediction)):
        return 1.0e30
    scale = max(float(np.std(truth)), 1.0e-30)
    return float(np.mean(((truth - prediction) / scale) ** 2))


def select_ar2_ridge(case: closure.CaseData, transport: np.ndarray) -> tuple[float, list[dict]]:
    indices = sample_indices(case.time_us, VALIDATION[0], VALIDATION[1], NATIVE_STEP)
    truth = transport[indices]
    rows = []
    for ridge in RIDGES:
        model = fit_ar2(case.time_us, transport, TRAIN, ridge)
        prediction = rollout_ar2(model, transport, indices)
        rows.append(
            {
                "model": "AR2_native",
                "ridge": ridge,
                "validation_nmse": validation_mse(truth, prediction),
            }
        )
    selected = min(rows, key=lambda row: row["validation_nmse"])
    return float(selected["ridge"]), rows


def select_map_ridge(
    case: closure.CaseData,
    state: np.ndarray,
    degree: int,
    step_frames: int,
) -> tuple[float, list[dict]]:
    indices = sample_indices(case.time_us, VALIDATION[0], VALIDATION[1], step_frames)
    truth = state[indices, 0]
    grid = "native" if step_frames == NATIVE_STEP else "strobe"
    rows = []
    for ridge in RIDGES:
        model = fit_delay_map(state, case.time_us, TRAIN, degree, ridge, step_frames)
        prediction = rollout_delay_map(model, state, indices)
        rows.append(
            {
                "model": f"degree{degree}_2D_{grid}",
                "degree": degree,
                "step_frames": step_frames,
                "ridge": ridge,
                "validation_nmse": validation_mse(truth, prediction),
            }
        )
    selected = min(rows, key=lambda row: row["validation_nmse"])
    return float(selected["ridge"]), rows


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) <= 1.0e-14 or np.std(y) <= 1.0e-14:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def phase_amplitude_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    dt_us: float,
) -> dict:
    if not np.all(np.isfinite(prediction)) or len(truth) < 12:
        return {
            "phase_rmse_rad": float("nan"),
            "final_phase_drift_rad": float("nan"),
            "instantaneous_frequency_mae_mhz": float("nan"),
            "envelope_correlation": float("nan"),
            "envelope_nrmse_by_mean": float("nan"),
            "envelope_mean_ratio": float("nan"),
        }
    truth_centered = truth - np.mean(truth)
    prediction_centered = prediction - np.mean(prediction)
    truth_analytic = hilbert(truth_centered)
    prediction_analytic = hilbert(prediction_centered)
    truth_phase = np.unwrap(np.angle(truth_analytic))
    prediction_phase = np.unwrap(np.angle(prediction_analytic))
    phase_error = prediction_phase - truth_phase
    phase_error -= phase_error[0]
    truth_frequency = np.gradient(truth_phase, dt_us) / (2.0 * np.pi)
    prediction_frequency = np.gradient(prediction_phase, dt_us) / (2.0 * np.pi)
    truth_envelope = np.abs(truth_analytic)
    prediction_envelope = np.abs(prediction_analytic)
    edge = max(2, min(10, len(truth) // 10))
    interior = slice(edge, len(truth) - edge)
    return {
        "phase_rmse_rad": float(np.sqrt(np.mean(phase_error[interior] ** 2))),
        "final_phase_drift_rad": float(phase_error[-edge - 1]),
        "instantaneous_frequency_mae_mhz": float(
            np.mean(np.abs(prediction_frequency[interior] - truth_frequency[interior]))
        ),
        "envelope_correlation": correlation(
            truth_envelope[interior], prediction_envelope[interior]
        ),
        "envelope_nrmse_by_mean": float(
            np.sqrt(np.mean((prediction_envelope[interior] - truth_envelope[interior]) ** 2))
            / max(float(np.mean(truth_envelope[interior])), 1.0e-30)
        ),
        "envelope_mean_ratio": float(np.mean(prediction_envelope[interior]))
        / max(float(np.mean(truth_envelope[interior])), 1.0e-30),
    }


def forecast_metrics(
    model_name: str,
    grid: str,
    times: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict:
    finite = np.isfinite(prediction)
    valid_fraction = float(np.mean(finite))
    if np.sum(finite) < 3:
        return {
            "model": model_name,
            "grid": grid,
            "samples": len(truth),
            "valid_fraction": valid_fraction,
            "skill_vs_persistence": float("nan"),
            "correlation": float("nan"),
            "std_ratio": float("nan"),
            **phase_amplitude_metrics(truth, prediction, float(np.median(np.diff(times)))),
        }
    truth_valid = truth[finite]
    prediction_valid = prediction[finite]
    persistence = np.full(len(truth_valid), truth_valid[0])
    model_mse = float(np.mean((truth_valid - prediction_valid) ** 2))
    persistence_mse = float(np.mean((truth_valid - persistence) ** 2))
    mean_baseline_mse = float(np.mean((truth_valid - np.mean(truth_valid)) ** 2))
    return {
        "model": model_name,
        "grid": grid,
        "samples": len(truth),
        "valid_fraction": valid_fraction,
        "start_time_us": float(times[0]),
        "end_time_us": float(times[-1]),
        "skill_vs_persistence": 1.0 - model_mse / max(persistence_mse, 1.0e-30),
        "variance_explained_vs_mean": 1.0
        - model_mse / max(mean_baseline_mse, 1.0e-30),
        "correlation": correlation(truth_valid, prediction_valid),
        "std_ratio": float(np.std(prediction_valid)) / max(float(np.std(truth_valid)), 1.0e-30),
        "mse": model_mse,
        **phase_amplitude_metrics(
            truth_valid,
            prediction_valid,
            float(np.median(np.diff(times[finite]))),
        ),
    }


def dominant_frequency(values: np.ndarray, dt_us: float) -> tuple[float, float]:
    centered = values - np.mean(values)
    spectrum = np.abs(np.fft.rfft(centered * np.hanning(len(centered)))) ** 2
    frequencies = np.fft.rfftfreq(len(centered), d=dt_us)
    spectrum[0] = 0.0
    peak = int(np.argmax(spectrum))
    concentration = float(np.sum(spectrum[max(1, peak - 1) : peak + 2])) / max(
        float(np.sum(spectrum)), 1.0e-30
    )
    return float(frequencies[peak]), concentration


def fit_all_models(case: closure.CaseData) -> tuple[list[dict], dict, dict, dict]:
    transport = np.asarray(case.observables["joint_selected"], dtype=np.float64)
    state = geometry.state_from_transport(transport, 1)
    selection_rows = []

    ar2_ridge, rows = select_ar2_ridge(case, transport)
    selection_rows.extend(rows)
    selected_map_ridges = {}
    for step_frames in (NATIVE_STEP, STROBE_STEP):
        for degree in DEGREES:
            ridge, rows = select_map_ridge(case, state, degree, step_frames)
            selection_rows.extend(rows)
            selected_map_ridges[(step_frames, degree)] = ridge

    oscillator = fit_oscillator(case.time_us, transport, REFIT)
    ar2 = fit_ar2(case.time_us, transport, REFIT, ar2_ridge)
    maps = {
        (step, degree): fit_delay_map(
            state,
            case.time_us,
            REFIT,
            degree,
            selected_map_ridges[(step, degree)],
            step,
        )
        for step in (NATIVE_STEP, STROBE_STEP)
        for degree in DEGREES
    }

    predictions = {}
    for step, grid in ((NATIVE_STEP, "native"), (STROBE_STEP, "strobe")):
        indices = sample_indices(case.time_us, TEST_START_US, TEST_END_US, step)
        times = case.time_us[indices]
        truth = transport[indices]
        absolute = oscillator.predict(times)
        reset = phase_reset_oscillator_prediction(
            oscillator,
            times,
            transport[indices[0]],
            state[indices[0], 1],
        )
        ar_prediction = rollout_ar2(ar2, transport, indices)
        predictions[(grid, "truth")] = (times, truth)
        predictions[(grid, "sinusoid_absolute_native")] = absolute
        predictions[(grid, "sinusoid_phase_reset_native")] = reset
        predictions[(grid, "AR2_native")] = ar_prediction
        for degree, label in zip(DEGREES, ("linear2D", "quadratic2D", "cubic2D")):
            predictions[(grid, f"{label}_{grid}")] = rollout_delay_map(
                maps[(step, degree)], state, indices
            )

    fitted = {
        "oscillator": oscillator,
        "ar2": ar2,
        "maps": maps,
        "selected_map_ridges": selected_map_ridges,
        "transport": transport,
        "state": state,
    }
    return selection_rows, predictions, fitted, selected_map_ridges


def evaluate_fixed(predictions: dict) -> list[dict]:
    rows = []
    for grid in ("native", "strobe"):
        times, truth = predictions[(grid, "truth")]
        model_names = [
            "sinusoid_absolute_native",
            "sinusoid_phase_reset_native",
            "AR2_native",
            f"linear2D_{grid}",
            f"quadratic2D_{grid}",
            f"cubic2D_{grid}",
        ]
        for model_name in model_names:
            rows.append(
                forecast_metrics(
                    model_name,
                    grid,
                    times,
                    truth,
                    predictions[(grid, model_name)],
                )
            )
    return rows


def evaluate_rolling(case: closure.CaseData, fitted: dict) -> list[dict]:
    transport = fitted["transport"]
    state = fitted["state"]
    ar2_ridge = fitted["ar2"].ridge
    map_ridges = fitted["selected_map_ridges"]
    rows = []
    for start in ROLLING_STARTS:
        history = (start - ROLLING_HISTORY_US, start)
        oscillator = fit_oscillator(case.time_us, transport, history)
        ar2 = fit_ar2(case.time_us, transport, history, ar2_ridge)
        for step, grid in ((NATIVE_STEP, "native"), (STROBE_STEP, "strobe")):
            indices = sample_indices(case.time_us, start, start + ROLLING_HORIZON_US, step)
            times = case.time_us[indices]
            truth = transport[indices]
            model_predictions = {
                "sinusoid_phase_reset_native": phase_reset_oscillator_prediction(
                    oscillator,
                    times,
                    transport[indices[0]],
                    state[indices[0], 1],
                ),
                "AR2_native": rollout_ar2(ar2, transport, indices),
            }
            for degree, label in zip(DEGREES, ("linear2D", "quadratic2D", "cubic2D")):
                model = fit_delay_map(
                    state,
                    case.time_us,
                    history,
                    degree,
                    map_ridges[(step, degree)],
                    step,
                )
                model_predictions[f"{label}_{grid}"] = rollout_delay_map(model, state, indices)
            for model_name, prediction in model_predictions.items():
                row = forecast_metrics(model_name, grid, times, truth, prediction)
                row["forecast_start_us"] = start
                row["history_start_us"] = history[0]
                row["horizon_us"] = ROLLING_HORIZON_US
                rows.append(row)
    return rows


def periodicity_diagnostics(case: closure.CaseData, fitted: dict) -> dict:
    transport = fitted["transport"]
    oscillator = fitted["oscillator"]
    train_indices = interval_indices(case.time_us, REFIT)
    test_indices = sample_indices(case.time_us, TEST_START_US, TEST_END_US, NATIVE_STEP)
    train_frequency, train_concentration = dominant_frequency(transport[train_indices], DT_US)
    test_frequency, test_concentration = dominant_frequency(transport[test_indices], DT_US)
    sample_frequency = 1.0 / (STROBE_STEP * DT_US)
    alias_frequency = abs(
        ((oscillator.frequency_mhz + 0.5 * sample_frequency) % sample_frequency)
        - 0.5 * sample_frequency
    )
    truth = transport[test_indices]
    analytic = hilbert(truth - np.mean(truth))
    phase = np.unwrap(np.angle(analytic))
    frequency = np.gradient(phase, DT_US) / (2.0 * np.pi)
    envelope = np.abs(analytic)
    edge = 10
    interior = slice(edge, len(truth) - edge)
    return {
        "native_dt_ns": DT_US * 1000.0,
        "strobe_dt_ns": STROBE_STEP * DT_US * 1000.0,
        "oscillator_fit_frequency_mhz": oscillator.frequency_mhz,
        "oscillator_period_ns": 1000.0 / oscillator.frequency_mhz,
        "native_frames_per_period": 1.0 / (oscillator.frequency_mhz * DT_US),
        "strobe_sampling_frequency_mhz": sample_frequency,
        "expected_strobe_alias_frequency_mhz": alias_frequency,
        "refit_fft_frequency_mhz": train_frequency,
        "refit_fft_peak_concentration": train_concentration,
        "test_fft_frequency_mhz": test_frequency,
        "test_fft_peak_concentration": test_concentration,
        "test_instantaneous_frequency_mean_mhz": float(np.mean(frequency[interior])),
        "test_instantaneous_frequency_cv": float(np.std(frequency[interior]))
        / max(abs(float(np.mean(frequency[interior]))), 1.0e-30),
        "test_envelope_cv": float(np.std(envelope[interior]))
        / max(float(np.mean(envelope[interior])), 1.0e-30),
    }


def plot_native_rollouts(predictions: dict, output: Path) -> None:
    times, truth = predictions[("native", "truth")]
    names = (
        "sinusoid_phase_reset_native",
        "AR2_native",
        "linear2D_native",
        "quadratic2D_native",
        "cubic2D_native",
    )
    scale = max(float(np.std(truth)), 1.0e-30)
    mean = float(np.mean(truth))
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), constrained_layout=True)
    zoom = times <= times[0] + 0.8
    axes[0].plot(times[zoom], (truth[zoom] - mean) / scale, color="#111111", linewidth=1.7, label="truth")
    for name in names:
        prediction = predictions[("native", name)]
        axes[0].plot(times[zoom], (prediction[zoom] - mean) / scale, linewidth=1.1, color=COLORS[name], label=name)
    axes[0].set_title("Native 15 ns waveform: first 0.8 us of holdout")
    axes[0].set_ylabel("transport [standardized]")
    axes[0].legend(frameon=False, ncol=3, loc="lower right")
    axes[0].grid(alpha=0.25)

    for name in names:
        prediction = predictions[("native", name)]
        finite = np.isfinite(prediction)
        error = np.full(len(prediction), np.nan)
        error[finite] = np.abs(prediction[finite] - truth[finite]) / scale
        axes[1].plot(times, error, linewidth=1.0, color=COLORS[name], label=name)
    axes[1].set_title("Absolute native-grid error over the complete holdout")
    axes[1].set_xlabel("time [us]")
    axes[1].set_ylabel("absolute error [truth std]")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.25)
    fig.savefig(output / "e40_native_periodic_null_rollouts.png", dpi=180)
    plt.close(fig)


def plot_strobe_rollouts(predictions: dict, output: Path) -> None:
    times, truth = predictions[("strobe", "truth")]
    names = (
        "sinusoid_phase_reset_native",
        "AR2_native",
        "linear2D_strobe",
        "quadratic2D_strobe",
        "cubic2D_strobe",
    )
    scale = max(float(np.std(truth)), 1.0e-30)
    mean = float(np.mean(truth))
    fig, axis = plt.subplots(figsize=(16, 5.5), constrained_layout=True)
    axis.plot(times, (truth - mean) / scale, color="#111111", linewidth=1.8, label="truth")
    for name in names:
        prediction = predictions[("strobe", name)]
        axis.plot(times, (prediction - mean) / scale, linewidth=1.15, color=COLORS[name], label=name)
    axis.set_title("E40 holdout on the previous 150 ns stroboscopic map grid")
    axis.set_xlabel("time [us]")
    axis.set_ylabel("transport [standardized]")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=3, loc="lower right")
    fig.savefig(output / "e40_stroboscopic_periodic_null_rollouts.png", dpi=180)
    plt.close(fig)


def plot_metric_comparison(rows: list[dict], output: Path) -> None:
    native = [row for row in rows if row["grid"] == "native"]
    order = [
        "sinusoid_phase_reset_native",
        "AR2_native",
        "linear2D_native",
        "quadratic2D_native",
        "cubic2D_native",
    ]
    by_name = {row["model"]: row for row in native}
    labels = ["sinusoid", "AR(2)", "linear 2D", "quadratic 2D", "cubic 2D"]
    metrics = (
        ("skill_vs_persistence", "MSE skill vs persistence"),
        ("correlation", "waveform correlation"),
        ("std_ratio", "standard-deviation ratio"),
        ("phase_rmse_rad", "phase RMSE [rad]"),
        ("instantaneous_frequency_mae_mhz", "instantaneous-frequency MAE [MHz]"),
        ("envelope_nrmse_by_mean", "envelope NRMSE / mean"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(17, 8), constrained_layout=True)
    colors = [COLORS[name] for name in order]
    for axis, (metric, title) in zip(axes.ravel(), metrics):
        values = [float(by_name[name][metric]) for name in order]
        axis.bar(np.arange(len(order)), values, color=colors)
        axis.set_xticks(np.arange(len(order)), labels, rotation=35, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("E40 native-grid periodic-null controls")
    fig.savefig(output / "e40_periodic_null_metric_comparison.png", dpi=180)
    plt.close(fig)


def plot_alias_spectrum(case: closure.CaseData, diagnostics: dict, output: Path) -> None:
    transport = np.asarray(case.observables["joint_selected"], dtype=np.float64)
    indices = sample_indices(case.time_us, TEST_START_US, TEST_END_US, NATIVE_STEP)
    truth = transport[indices] - np.mean(transport[indices])
    native_frequency = np.fft.rfftfreq(len(truth), d=DT_US)
    native_power = np.abs(np.fft.rfft(truth * np.hanning(len(truth)))) ** 2
    strobe = truth[::STROBE_STEP]
    strobe_frequency = np.fft.rfftfreq(len(strobe), d=STROBE_STEP * DT_US)
    strobe_power = np.abs(np.fft.rfft(strobe * np.hanning(len(strobe)))) ** 2
    native_power /= max(float(np.max(native_power)), 1.0e-30)
    strobe_power /= max(float(np.max(strobe_power)), 1.0e-30)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    axes[0].plot(native_frequency, native_power, color="#0072b2")
    axes[0].axvline(diagnostics["oscillator_fit_frequency_mhz"], color="#d55e00", linestyle="--", label="fitted oscillator")
    axes[0].set_xlim(0.0, 1.0 / (2.0 * DT_US))
    axes[0].set_title("Native 15 ns spectrum")
    axes[0].set_xlabel("frequency [MHz]")
    axes[0].set_ylabel("normalized power")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)
    axes[1].plot(strobe_frequency, strobe_power, color="#009e73")
    axes[1].axvline(diagnostics["expected_strobe_alias_frequency_mhz"], color="#d55e00", linestyle="--", label="expected alias")
    axes[1].set_title("150 ns stroboscopic spectrum")
    axes[1].set_xlabel("frequency [MHz]")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)
    fig.savefig(output / "e40_native_vs_stroboscopic_spectrum.png", dpi=180)
    plt.close(fig)


def plot_rolling(rows: list[dict], output: Path) -> None:
    models = (
        "sinusoid_phase_reset_native",
        "AR2_native",
        "linear2D_native",
        "quadratic2D_native",
        "cubic2D_native",
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    for axis, metric, title in (
        (axes[0], "skill_vs_persistence", "rolling 1.2 us skill"),
        (axes[1], "correlation", "rolling 1.2 us correlation"),
    ):
        for model in models:
            selected = sorted(
                [row for row in rows if row["grid"] == "native" and row["model"] == model],
                key=lambda row: row["forecast_start_us"],
            )
            axis.plot(
                [row["forecast_start_us"] for row in selected],
                [row[metric] for row in selected],
                marker="o",
                color=COLORS[model],
                label=model,
            )
        axis.axhline(0.0, color="#777777", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("forecast start [us]")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("MSE skill vs persistence")
    axes[1].set_ylabel("correlation")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(output / "e40_periodic_null_rolling.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    case = load_e40()

    selection_rows, predictions, fitted, selected_map_ridges = fit_all_models(case)
    fixed_rows = evaluate_fixed(predictions)
    rolling_rows = evaluate_rolling(case, fitted)
    diagnostics = periodicity_diagnostics(case, fitted)

    write_csv(output / "model_selection.csv", selection_rows)
    write_csv(output / "fixed_holdout_metrics.csv", fixed_rows)
    write_csv(output / "rolling_metrics.csv", rolling_rows)
    write_csv(output / "periodicity_diagnostics.csv", [diagnostics])

    plot_native_rollouts(predictions, output)
    plot_strobe_rollouts(predictions, output)
    plot_metric_comparison(fixed_rows, output)
    plot_alias_spectrum(case, diagnostics, output)
    plot_rolling(rolling_rows, output)

    fixed_by_key = {(row["grid"], row["model"]): row for row in fixed_rows}
    summary = {
        "status": "PASS",
        "protocol": {
            "train_us": TRAIN,
            "validation_us": VALIDATION,
            "refit_us": REFIT,
            "test_start_us": TEST_START_US,
            "test_available_end_us": float(case.time_us[-1]),
            "native_step_ns": NATIVE_STEP * DT_US * 1000.0,
            "strobe_step_ns": STROBE_STEP * DT_US * 1000.0,
            "ridge_selection": "autonomous validation only",
        },
        "periodicity": diagnostics,
        "selected_ar2_ridge": fitted["ar2"].ridge,
        "selected_map_ridges": {
            f"step{step}_degree{degree}": ridge
            for (step, degree), ridge in selected_map_ridges.items()
        },
        "fixed_holdout": fixed_rows,
        "rolling": rolling_rows,
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    sinusoid_absolute = fixed_by_key[("native", "sinusoid_absolute_native")]
    sinusoid = fixed_by_key[("native", "sinusoid_phase_reset_native")]
    ar2 = fixed_by_key[("native", "AR2_native")]
    linear_native = fixed_by_key[("native", "linear2D_native")]
    cubic_native = fixed_by_key[("native", "cubic2D_native")]
    linear_strobe = fixed_by_key[("strobe", "linear2D_strobe")]
    cubic_strobe = fixed_by_key[("strobe", "cubic2D_strobe")]
    sinusoid_strobe = fixed_by_key[("strobe", "sinusoid_absolute_native")]
    strobe_residual_reduction = 1.0 - float(cubic_strobe["mse"]) / max(
        float(sinusoid_strobe["mse"]), 1.0e-30
    )
    readme = f"""# E40 periodic-orbit null controls

## Critical sampling fact

The native E40 joint-transport peak is {diagnostics['oscillator_fit_frequency_mhz']:.4f} MHz
(period {diagnostics['oscillator_period_ns']:.2f} ns), giving only
{diagnostics['native_frames_per_period']:.2f} native 15 ns frames per period.
The preceding 150 ns map samples at {diagnostics['strobe_sampling_frequency_mhz']:.4f} MHz,
so this mode aliases to {diagnostics['expected_strobe_alias_frequency_mhz']:.4f} MHz.
Native and stroboscopic closure must therefore be reported separately.

## Native 15 ns holdout

| model | skill | variance explained | correlation | std ratio | phase RMSE [rad] | envelope NRMSE |
|---|---:|---:|---:|---:|---:|---:|
| absolute-phase sinusoid | {sinusoid_absolute['skill_vs_persistence']:.3f} | {sinusoid_absolute['variance_explained_vs_mean']:.3f} | {sinusoid_absolute['correlation']:.3f} | {sinusoid_absolute['std_ratio']:.3f} | {sinusoid_absolute['phase_rmse_rad']:.3f} | {sinusoid_absolute['envelope_nrmse_by_mean']:.3f} |
| initial-state sinusoid | {sinusoid['skill_vs_persistence']:.3f} | {sinusoid['variance_explained_vs_mean']:.3f} | {sinusoid['correlation']:.3f} | {sinusoid['std_ratio']:.3f} | {sinusoid['phase_rmse_rad']:.3f} | {sinusoid['envelope_nrmse_by_mean']:.3f} |
| AR(2) | {ar2['skill_vs_persistence']:.3f} | {ar2['variance_explained_vs_mean']:.3f} | {ar2['correlation']:.3f} | {ar2['std_ratio']:.3f} | {ar2['phase_rmse_rad']:.3f} | {ar2['envelope_nrmse_by_mean']:.3f} |
| linear 2D | {linear_native['skill_vs_persistence']:.3f} | {linear_native['variance_explained_vs_mean']:.3f} | {linear_native['correlation']:.3f} | {linear_native['std_ratio']:.3f} | {linear_native['phase_rmse_rad']:.3f} | {linear_native['envelope_nrmse_by_mean']:.3f} |
| cubic 2D | {cubic_native['skill_vs_persistence']:.3f} | {cubic_native['variance_explained_vs_mean']:.3f} | {cubic_native['correlation']:.3f} | {cubic_native['std_ratio']:.3f} | {cubic_native['phase_rmse_rad']:.3f} | {cubic_native['envelope_nrmse_by_mean']:.3f} |

## Previous 150 ns stroboscopic grid

The linear 2D map has skill/correlation/std ratio
{linear_strobe['skill_vs_persistence']:.3f}/
{linear_strobe['correlation']:.3f}/
{linear_strobe['std_ratio']:.3f}; the cubic map has
{cubic_strobe['skill_vs_persistence']:.3f}/
{cubic_strobe['correlation']:.3f}/
{cubic_strobe['std_ratio']:.3f}.
The cubic map reduces the remaining stroboscopic MSE by
{strobe_residual_reduction:.1%} relative to the absolute-phase sinusoid.

## Interpretation

The constant-frequency null is not rejected.  On the native grid a sinusoid
fixed from 12--20 us explains {sinusoid_absolute['variance_explained_vs_mean']:.1%}
of holdout variance and outperforms the native cubic map.  E40 transport is
therefore primarily a stable one-frequency orbit at the present output
resolution.  The cubic stroboscopic map captures a modest residual beyond that
orbit, but this residual is measured after severe 150 ns aliasing and cannot by
itself establish a general nonlinear plasma-transport closure.

The comparison determines whether cubic nonlinearity carries holdout value
beyond a stationary oscillator or linear two-delay recurrence.  Positive skill
alone is not enough: phase drift, amplitude ratio, and envelope modulation are
reported together.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(f"Saved analysis to {output}")


if __name__ == "__main__":
    main()
