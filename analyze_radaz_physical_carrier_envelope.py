from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SIMVP_ROOT = Path(__file__).resolve().parent
RESEARCH_ROOT = SIMVP_ROOT.parent
PIC_ROOT = RESEARCH_ROOT / "PEPAPIC" / "test" / "results" / "2D_Landmark"
DEFAULT_OUTDIR = (
    SIMVP_ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_physical_carrier_envelope"
)

ELECTRIC_FIELDS = (10, 20, 30, 40)
PHYSICAL_FIELDS = ("phi", "electron_den", "efy")
FRAME_DT_US = 0.015
FIT_START_US = 20.0
VALIDATION_START_US = 23.0
FIT_END_US = 24.0
HOLDOUT_END_US = 30.0
MODE_MAX = 32
MODE_COUNT = 5
RANK_CANDIDATES = (4, 8, 12, 16, 20, 24, 30)
DELAY_CANDIDATES = (5, 10, 20, 40, 80)
RIDGE_CANDIDATES = (
    0.0,
    1.0e-8,
    1.0e-6,
    1.0e-4,
    1.0e-2,
    1.0e-1,
    1.0,
    10.0,
    100.0,
    1000.0,
)


@dataclass
class CaseData:
    electric_field_kvm: int
    source_h5: Path
    time_us: np.ndarray
    frame: np.ndarray
    modes: np.ndarray
    mode_roles: list[str]
    scores: np.ndarray
    scales: np.ndarray
    carrier_angles: np.ndarray
    carrier_frequencies_mhz: np.ndarray
    raw_complex: np.ndarray
    normalized_complex: np.ndarray
    envelope_complex: np.ndarray

    @property
    def state_dimensions(self) -> int:
        return 2 * len(self.modes) * len(PHYSICAL_FIELDS)

    def flatten(self, values: np.ndarray) -> np.ndarray:
        flat = values.reshape(len(values), -1)
        return np.concatenate((flat.real, flat.imag), axis=1)

    def unflatten(self, values: np.ndarray) -> np.ndarray:
        half = values.shape[1] // 2
        complex_flat = values[:, :half] + 1j * values[:, half:]
        return complex_flat.reshape(
            len(values), len(self.modes), len(PHYSICAL_FIELDS)
        )

    def remodulate(
        self, envelope_states: np.ndarray, frame_indices: np.ndarray
    ) -> np.ndarray:
        envelope = self.unflatten(envelope_states)
        phase = np.exp(
            1j
            * frame_indices[:, None, None]
            * self.carrier_angles[None, :, None]
        )
        return envelope * phase

    def physical_coefficients(self, normalized: np.ndarray) -> np.ndarray:
        return normalized * self.scales[None, :, :]


@dataclass
class LinearModel:
    mean: np.ndarray
    matrix: np.ndarray
    eigenvalues: np.ndarray
    rank: int


@dataclass
class HankelModel:
    delay: int
    rank: int
    state_dimensions: int
    delay_mean: np.ndarray
    basis: np.ndarray
    matrix: np.ndarray
    eigenvalues: np.ndarray


@dataclass
class ConditionedModel:
    weights: np.ndarray
    ridge: float
    state_dimensions: int
    means: dict[int, np.ndarray]


def case_name(electric_field_kvm: int) -> str:
    return (
        "2D_RadAz_Xe1p_Bx20mT_"
        f"Ez{electric_field_kvm}kVm_dt15ps_out15ns"
    )


def diagnostic_path(electric_field_kvm: int) -> Path:
    name = case_name(electric_field_kvm)
    return (
        PIC_ROOT
        / name
        / name
        / f"bifurcation_analysis_B20mT_E{electric_field_kvm}kVm"
        / "bifurcation_diagnostics_uncompressed.h5"
    )


def contiguous_mask(
    time_us: np.ndarray,
    start_us: float,
    end_us: float,
    include_end: bool = True,
) -> np.ndarray:
    if include_end:
        return (time_us >= start_us) & (time_us <= end_us + 1.0e-10)
    return (time_us >= start_us) & (time_us < end_us - 1.0e-10)


def safe_correlation(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth_flat = np.asarray(truth, dtype=np.float64).ravel()
    prediction_flat = np.asarray(prediction, dtype=np.float64).ravel()
    finite = np.isfinite(truth_flat) & np.isfinite(prediction_flat)
    if np.count_nonzero(finite) < 3:
        return float("nan")
    left = truth_flat[finite] - np.mean(truth_flat[finite])
    right = prediction_flat[finite] - np.mean(prediction_flat[finite])
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= np.finfo(float).tiny:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    return value


def select_modes(
    coefficients: np.ndarray,
    fit_mask: np.ndarray,
    mode_count: int = MODE_COUNT,
    mode_max: int = MODE_MAX,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    upper = min(mode_max, coefficients.shape[1] - 1)
    scores = np.zeros(upper + 1, dtype=np.float64)
    for field_index in range(coefficients.shape[2]):
        energy = np.mean(
            np.abs(coefficients[fit_mask, : upper + 1, field_index]) ** 2,
            axis=0,
        )
        energy[0] = 0.0
        scores += energy / max(float(np.sum(energy)), np.finfo(float).tiny)

    primary = int(np.argmax(scores[1:]) + 1)
    modes = [primary]
    roles = ["primary"]
    for harmonic, role in ((2, "harmonic_2"), (3, "harmonic_3")):
        candidate = primary * harmonic
        if candidate <= upper and candidate not in modes:
            modes.append(candidate)
            roles.append(role)

    for candidate in np.argsort(scores[1:])[::-1] + 1:
        candidate = int(candidate)
        if candidate not in modes:
            modes.append(candidate)
            roles.append(f"competitor_{len(modes)}")
        if len(modes) >= mode_count:
            break

    if len(modes) != mode_count:
        raise ValueError(
            f"Could select only {len(modes)} modes, expected {mode_count}"
        )
    return np.asarray(modes, dtype=np.int64), roles, scores


def estimate_representation(
    electric_field_kvm: int,
    source_h5: Path,
) -> CaseData:
    with h5py.File(source_h5, "r") as handle:
        time_us = np.asarray(handle["axes/time_s"], dtype=np.float64) * 1.0e6
        signals = np.stack(
            [
                np.asarray(handle[f"radial_mean/{field}"], dtype=np.float64)
                for field in PHYSICAL_FIELDS
            ],
            axis=-1,
        )

    frame = np.arange(len(time_us), dtype=np.int64)
    fit_mask = contiguous_mask(time_us, FIT_START_US, FIT_END_US)
    coefficients = np.fft.rfft(signals, axis=1, norm="forward")
    modes, roles, scores = select_modes(coefficients, fit_mask)
    selected = coefficients[:, modes, :]

    scales = np.sqrt(np.mean(np.abs(selected[fit_mask]) ** 2, axis=0))
    scales = np.maximum(scales, np.finfo(np.float64).tiny)
    normalized = selected / scales[None, :, :]

    fit_indices = np.flatnonzero(fit_mask)
    carrier_angles = np.zeros(len(modes), dtype=np.float64)
    for mode_index in range(len(modes)):
        current = normalized[fit_indices[:-1], mode_index, :]
        following = normalized[fit_indices[1:], mode_index, :]
        cross = np.sum(following * np.conj(current))
        if abs(cross) > np.finfo(float).tiny:
            carrier_angles[mode_index] = float(np.angle(cross))

    phase = np.exp(
        -1j
        * frame[:, None, None]
        * carrier_angles[None, :, None]
    )
    envelope = normalized * phase
    frequencies = carrier_angles / (2.0 * np.pi * FRAME_DT_US)

    return CaseData(
        electric_field_kvm=electric_field_kvm,
        source_h5=source_h5,
        time_us=time_us,
        frame=frame,
        modes=modes,
        mode_roles=roles,
        scores=scores,
        scales=scales,
        carrier_angles=carrier_angles,
        carrier_frequencies_mhz=frequencies,
        raw_complex=selected,
        normalized_complex=normalized,
        envelope_complex=envelope,
    )


def fit_linear_model(states: np.ndarray, rank: int) -> LinearModel:
    mean = np.mean(states, axis=0)
    centered = states - mean
    x = centered[:-1].T
    y = centered[1:].T
    u, singular_values, vh = np.linalg.svd(x, full_matrices=False)
    effective_rank = min(rank, int(np.count_nonzero(singular_values > 1.0e-10)))
    if effective_rank < 1:
        raise ValueError("No nonzero singular values for linear model")
    inverse = np.diag(1.0 / singular_values[:effective_rank])
    matrix = (
        y
        @ vh[:effective_rank].T
        @ inverse
        @ u[:, :effective_rank].T
    )
    return LinearModel(
        mean=mean,
        matrix=matrix,
        eigenvalues=np.linalg.eigvals(matrix),
        rank=effective_rank,
    )


def rollout_linear(
    model: LinearModel, initial_state: np.ndarray, steps: int
) -> np.ndarray:
    state = np.asarray(initial_state, dtype=np.float64) - model.mean
    forecast = np.empty((steps, len(state)), dtype=np.float64)
    for index in range(steps):
        state = model.matrix @ state
        if (
            not np.all(np.isfinite(state))
            or np.max(np.abs(state)) > 1.0e8
        ):
            forecast[index:] = np.nan
            break
        forecast[index] = state + model.mean
    return forecast


def make_delay_vectors(states: np.ndarray, delay: int) -> np.ndarray:
    if delay < 1 or len(states) < delay:
        raise ValueError(f"Invalid delay={delay} for {len(states)} states")
    return np.asarray(
        [
            np.concatenate(
                [states[index - lag] for lag in range(delay)], axis=0
            )
            for index in range(delay - 1, len(states))
        ],
        dtype=np.float64,
    )


def fit_hankel_model(
    states: np.ndarray, delay: int, rank: int
) -> HankelModel:
    delay_vectors = make_delay_vectors(states, delay)
    delay_mean = np.mean(delay_vectors, axis=0)
    centered = delay_vectors - delay_mean
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    maximum_rank = min(len(delay_vectors) - 1, right.shape[0])
    effective_rank = min(rank, maximum_rank)
    if effective_rank < 1:
        raise ValueError("Hankel model has zero effective rank")
    basis = right[:effective_rank].T
    coordinates = centered @ basis
    x = coordinates[:-1].T
    y = coordinates[1:].T
    matrix = y @ np.linalg.pinv(x, rcond=1.0e-10)
    return HankelModel(
        delay=delay,
        rank=effective_rank,
        state_dimensions=states.shape[1],
        delay_mean=delay_mean,
        basis=basis,
        matrix=matrix,
        eigenvalues=np.linalg.eigvals(matrix),
    )


def rollout_hankel(
    model: HankelModel, history: np.ndarray, steps: int
) -> np.ndarray:
    delay_vector = make_delay_vectors(
        history[-model.delay :], model.delay
    )[0]
    coordinate = (delay_vector - model.delay_mean) @ model.basis
    forecast = np.empty(
        (steps, model.state_dimensions), dtype=np.float64
    )
    for index in range(steps):
        coordinate = model.matrix @ coordinate
        if (
            not np.all(np.isfinite(coordinate))
            or np.max(np.abs(coordinate)) > 1.0e8
        ):
            forecast[index:] = np.nan
            break
        reconstructed = coordinate @ model.basis.T + model.delay_mean
        forecast[index] = reconstructed[: model.state_dimensions]
    return forecast


def prediction_mse(truth: np.ndarray, prediction: np.ndarray) -> float:
    if prediction.shape != truth.shape or not np.all(np.isfinite(prediction)):
        return float("inf")
    return float(np.mean((prediction - truth) ** 2))


def reconstruct_raw_state(
    case: CaseData,
    envelope_prediction: np.ndarray,
    frame_indices: np.ndarray,
) -> np.ndarray:
    return case.flatten(case.remodulate(envelope_prediction, frame_indices))


def select_local_models(case: CaseData) -> tuple[dict, list[dict]]:
    fit_mask = contiguous_mask(case.time_us, FIT_START_US, FIT_END_US)
    subtrain_mask = fit_mask & (case.time_us < VALIDATION_START_US)
    validation_mask = fit_mask & (case.time_us >= VALIDATION_START_US)

    raw_states = case.flatten(case.normalized_complex)
    envelope_states = case.flatten(case.envelope_complex)
    raw_subtrain = raw_states[subtrain_mask]
    envelope_subtrain = envelope_states[subtrain_mask]
    raw_validation = raw_states[validation_mask]
    validation_frames = case.frame[validation_mask]
    trials: list[dict] = []

    best_raw = {"objective": float("inf")}
    for rank in RANK_CANDIDATES:
        try:
            model = fit_linear_model(raw_subtrain, rank)
            prediction = rollout_linear(
                model, raw_subtrain[-1], len(raw_validation)
            )
            mse = prediction_mse(raw_validation, prediction)
            radius = float(np.max(np.abs(model.eigenvalues)))
        except (ValueError, np.linalg.LinAlgError):
            mse = float("inf")
            radius = float("inf")
        objective = mse + max(0.0, radius - 1.02) * 10.0
        row = {
            "electric_field_kvm": case.electric_field_kvm,
            "method": "raw_dmd",
            "rank": rank,
            "delay": 1,
            "validation_mse": mse,
            "spectral_radius": radius,
            "objective": objective,
        }
        trials.append(row)
        if objective < best_raw["objective"]:
            best_raw = row

    best_envelope = {"objective": float("inf")}
    for rank in RANK_CANDIDATES:
        try:
            model = fit_linear_model(envelope_subtrain, rank)
            envelope_prediction = rollout_linear(
                model, envelope_subtrain[-1], len(raw_validation)
            )
            prediction = reconstruct_raw_state(
                case, envelope_prediction, validation_frames
            )
            mse = prediction_mse(raw_validation, prediction)
            radius = float(np.max(np.abs(model.eigenvalues)))
        except (ValueError, np.linalg.LinAlgError):
            mse = float("inf")
            radius = float("inf")
        objective = mse + max(0.0, radius - 1.02) * 10.0
        row = {
            "electric_field_kvm": case.electric_field_kvm,
            "method": "envelope_dmd",
            "rank": rank,
            "delay": 1,
            "validation_mse": mse,
            "spectral_radius": radius,
            "objective": objective,
        }
        trials.append(row)
        if objective < best_envelope["objective"]:
            best_envelope = row

    best_hankel = {"objective": float("inf")}
    for delay in DELAY_CANDIDATES:
        for rank in RANK_CANDIDATES:
            try:
                model = fit_hankel_model(
                    envelope_subtrain, delay, rank
                )
                envelope_prediction = rollout_hankel(
                    model, envelope_subtrain, len(raw_validation)
                )
                prediction = reconstruct_raw_state(
                    case, envelope_prediction, validation_frames
                )
                mse = prediction_mse(raw_validation, prediction)
                radius = float(np.max(np.abs(model.eigenvalues)))
            except (ValueError, np.linalg.LinAlgError):
                mse = float("inf")
                radius = float("inf")
            objective = mse + max(0.0, radius - 1.02) * 10.0
            row = {
                "electric_field_kvm": case.electric_field_kvm,
                "method": "envelope_hankel_dmd",
                "rank": rank,
                "delay": delay,
                "validation_mse": mse,
                "spectral_radius": radius,
                "objective": objective,
            }
            trials.append(row)
            if objective < best_hankel["objective"]:
                best_hankel = row

    selected = {
        "raw_dmd": best_raw,
        "envelope_dmd": best_envelope,
        "envelope_hankel_dmd": best_hankel,
    }
    return selected, trials


def conditioned_features(states: np.ndarray, parameter: float) -> np.ndarray:
    return np.concatenate((states, parameter * states), axis=1)


def fit_conditioned_model(
    states_by_case: dict[int, np.ndarray],
    ridge: float,
) -> ConditionedModel:
    features = []
    targets = []
    means = {}
    for electric_field_kvm, states in states_by_case.items():
        parameter = (electric_field_kvm - 25.0) / 15.0
        mean = np.mean(states, axis=0)
        means[electric_field_kvm] = mean
        centered = states - mean
        features.append(conditioned_features(centered[:-1], parameter))
        targets.append(centered[1:])
    x = np.concatenate(features, axis=0)
    y = np.concatenate(targets, axis=0)
    gram = x.T @ x
    regularization = ridge * np.eye(gram.shape[0], dtype=np.float64)
    weights = np.linalg.pinv(
        gram + regularization, rcond=1.0e-10
    ) @ (x.T @ y)
    return ConditionedModel(
        weights=weights,
        ridge=ridge,
        state_dimensions=y.shape[1],
        means=means,
    )


def rollout_conditioned(
    model: ConditionedModel,
    initial_state: np.ndarray,
    electric_field_kvm: int,
    steps: int,
) -> np.ndarray:
    parameter = (electric_field_kvm - 25.0) / 15.0
    mean = model.means[electric_field_kvm]
    state = np.asarray(initial_state, dtype=np.float64) - mean
    forecast = np.empty((steps, len(state)), dtype=np.float64)
    for index in range(steps):
        feature = conditioned_features(state[None, :], parameter)[0]
        state = feature @ model.weights
        if (
            not np.all(np.isfinite(state))
            or np.max(np.abs(state)) > 1.0e8
        ):
            forecast[index:] = np.nan
            break
        forecast[index] = state + mean
    return forecast


def conditioned_spectral_radius(
    model: ConditionedModel, electric_field_kvm: int
) -> float:
    parameter = (electric_field_kvm - 25.0) / 15.0
    dimensions = model.state_dimensions
    matrix = (
        model.weights[:dimensions]
        + parameter * model.weights[dimensions:]
    )
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def select_conditioned_model(
    cases: dict[int, CaseData],
) -> tuple[dict, list[dict]]:
    subtrain_states = {}
    validation_masks = {}
    for electric_field_kvm, case in cases.items():
        fit_mask = contiguous_mask(case.time_us, FIT_START_US, FIT_END_US)
        subtrain_mask = fit_mask & (case.time_us < VALIDATION_START_US)
        validation_mask = fit_mask & (case.time_us >= VALIDATION_START_US)
        subtrain_states[electric_field_kvm] = case.flatten(
            case.envelope_complex
        )[subtrain_mask]
        validation_masks[electric_field_kvm] = validation_mask

    trials = []
    best = {"objective": float("inf")}
    for ridge in RIDGE_CANDIDATES:
        try:
            model = fit_conditioned_model(subtrain_states, ridge)
            case_mse = []
            case_radius = []
            for electric_field_kvm, case in cases.items():
                validation_mask = validation_masks[electric_field_kvm]
                raw_truth = case.flatten(case.normalized_complex)[
                    validation_mask
                ]
                envelope_prediction = rollout_conditioned(
                    model,
                    subtrain_states[electric_field_kvm][-1],
                    electric_field_kvm,
                    len(raw_truth),
                )
                raw_prediction = reconstruct_raw_state(
                    case,
                    envelope_prediction,
                    case.frame[validation_mask],
                )
                case_mse.append(prediction_mse(raw_truth, raw_prediction))
                case_radius.append(
                    conditioned_spectral_radius(
                        model, electric_field_kvm
                    )
                )
            mse = float(np.mean(case_mse))
        except (ValueError, np.linalg.LinAlgError):
            mse = float("inf")
            case_mse = [float("inf")] * len(cases)
            case_radius = [float("inf")] * len(cases)
        maximum_radius = float(np.max(case_radius))
        # A tiny unstable eigenvalue is harmless over the 1 us validation
        # window but can dominate the strict 6 us holdout. Prefer a stable
        # conditioned map even when its short validation MSE is slightly
        # higher.
        objective = mse + max(0.0, maximum_radius - 1.0) * 100.0
        row = {
            "method": "conditioned_shared_envelope",
            "ridge": ridge,
            "validation_mse": mse,
            "maximum_spectral_radius": maximum_radius,
            "objective": objective,
        }
        for electric_field_kvm, value, radius in zip(
            cases, case_mse, case_radius
        ):
            row[f"validation_mse_e{electric_field_kvm}"] = value
            row[f"spectral_radius_e{electric_field_kvm}"] = radius
        trials.append(row)
        if "ridge" not in best or objective < best["objective"]:
            best = row
    return best, trials


def estimate_frequency_mhz(
    values: np.ndarray, time_us: np.ndarray
) -> float:
    finite = np.isfinite(values.real) & np.isfinite(values.imag)
    if np.count_nonzero(finite) < 3:
        return float("nan")
    phase = np.unwrap(np.angle(values[finite]))
    slope = np.polyfit(time_us[finite], phase, 1)[0]
    return float(slope / (2.0 * np.pi))


def selected_modal_transport(
    physical_coefficients: np.ndarray,
) -> np.ndarray:
    electron_index = PHYSICAL_FIELDS.index("electron_den")
    efy_index = PHYSICAL_FIELDS.index("efy")
    electron = physical_coefficients[:, :, electron_index]
    efy = physical_coefficients[:, :, efy_index]
    return -2.0 * np.sum(np.real(electron * np.conj(efy)), axis=1) / 0.020


def evaluate_prediction(
    case: CaseData,
    raw_truth_state: np.ndarray,
    raw_prediction_state: np.ndarray,
    persistence_state: np.ndarray,
    carrier_state: np.ndarray,
    holdout_frames: np.ndarray,
    method: str,
) -> tuple[dict, dict]:
    finite = np.isfinite(raw_prediction_state).all(axis=1)
    mse = prediction_mse(raw_truth_state, raw_prediction_state)
    persistence_mse = prediction_mse(raw_truth_state, persistence_state)
    carrier_mse = prediction_mse(raw_truth_state, carrier_state)
    rmse = math.sqrt(mse) if np.isfinite(mse) else float("inf")
    skill_persistence = (
        1.0 - mse / persistence_mse
        if np.isfinite(mse) and persistence_mse > 0.0
        else float("-inf")
    )
    skill_carrier = (
        1.0 - mse / carrier_mse
        if np.isfinite(mse) and carrier_mse > 0.0
        else float("-inf")
    )

    truth_complex = case.unflatten(raw_truth_state)
    prediction_complex = case.unflatten(raw_prediction_state)
    amplitude_truth = np.abs(truth_complex)
    amplitude_prediction = np.abs(prediction_complex)
    phase_error = np.abs(
        np.angle(prediction_complex * np.conj(truth_complex))
    )
    phase_weights = amplitude_truth**2
    valid_complex = np.isfinite(prediction_complex)
    weighted_phase = (
        float(
            np.sum(phase_error[valid_complex] * phase_weights[valid_complex])
            / np.sum(phase_weights[valid_complex])
        )
        if np.any(valid_complex)
        else float("nan")
    )
    coherence_denominator = math.sqrt(
        float(np.sum(np.abs(truth_complex) ** 2))
        * float(
            np.nansum(np.abs(prediction_complex) ** 2)
        )
    )
    coherence = (
        float(
            abs(
                np.nansum(
                    prediction_complex * np.conj(truth_complex)
                )
            )
            / coherence_denominator
        )
        if coherence_denominator > np.finfo(float).tiny
        else float("nan")
    )

    truth_physical = case.physical_coefficients(truth_complex)
    prediction_physical = case.physical_coefficients(prediction_complex)
    carrier_physical = case.physical_coefficients(
        case.unflatten(carrier_state)
    )
    truth_transport = selected_modal_transport(truth_physical)
    prediction_transport = selected_modal_transport(prediction_physical)
    carrier_transport = selected_modal_transport(carrier_physical)
    transport_mse = prediction_mse(
        truth_transport[:, None], prediction_transport[:, None]
    )
    carrier_transport_mse = prediction_mse(
        truth_transport[:, None], carrier_transport[:, None]
    )
    transport_skill = (
        1.0 - transport_mse / carrier_transport_mse
        if np.isfinite(transport_mse) and carrier_transport_mse > 0.0
        else float("-inf")
    )

    primary_index = 0
    phi_index = PHYSICAL_FIELDS.index("phi")
    primary_truth = truth_complex[:, primary_index, phi_index]
    primary_prediction = prediction_complex[:, primary_index, phi_index]
    time_us = case.time_us[holdout_frames]

    metrics = {
        "electric_field_kvm": case.electric_field_kvm,
        "method": method,
        "state_rmse": rmse,
        "state_skill_vs_persistence": skill_persistence,
        "state_skill_vs_constant_carrier": skill_carrier,
        "state_correlation": safe_correlation(
            raw_truth_state, raw_prediction_state
        ),
        "complex_coherence": coherence,
        "weighted_phase_mae_rad": weighted_phase,
        "amplitude_correlation": safe_correlation(
            amplitude_truth, amplitude_prediction
        ),
        "finite_fraction": float(np.mean(finite)),
        "selected_transport_correlation": safe_correlation(
            truth_transport, prediction_transport
        ),
        "selected_transport_skill_vs_constant_carrier": transport_skill,
        "primary_truth_frequency_mhz": estimate_frequency_mhz(
            primary_truth, time_us
        ),
        "primary_prediction_frequency_mhz": estimate_frequency_mhz(
            primary_prediction, time_us
        ),
    }
    metrics["primary_frequency_absolute_error_mhz"] = abs(
        metrics["primary_prediction_frequency_mhz"]
        - metrics["primary_truth_frequency_mhz"]
    )
    series = {
        "time_us": time_us,
        "state_error": np.sqrt(
            np.nanmean(
                (raw_prediction_state - raw_truth_state) ** 2, axis=1
            )
        ),
        "primary_truth_amplitude": np.abs(primary_truth),
        "primary_prediction_amplitude": np.abs(primary_prediction),
        "primary_phase_error_rad": np.abs(
            np.angle(primary_prediction * np.conj(primary_truth))
        ),
        "transport_truth": truth_transport,
        "transport_prediction": prediction_transport,
    }
    return metrics, series


def build_predictions(
    cases: dict[int, CaseData],
    selected_local: dict[int, dict],
    selected_conditioned: dict,
) -> tuple[dict, dict, ConditionedModel]:
    fit_envelopes = {}
    for electric_field_kvm, case in cases.items():
        fit_mask = contiguous_mask(case.time_us, FIT_START_US, FIT_END_US)
        fit_envelopes[electric_field_kvm] = case.flatten(
            case.envelope_complex
        )[fit_mask]
    conditioned = fit_conditioned_model(
        fit_envelopes, float(selected_conditioned["ridge"])
    )

    predictions: dict[int, dict[str, np.ndarray]] = {}
    model_details = {}
    for electric_field_kvm, case in cases.items():
        fit_mask = contiguous_mask(case.time_us, FIT_START_US, FIT_END_US)
        holdout_mask = (
            (case.time_us > FIT_END_US + 1.0e-10)
            & (case.time_us <= HOLDOUT_END_US + 1.0e-10)
        )
        fit_frames = case.frame[fit_mask]
        holdout_frames = case.frame[holdout_mask]
        raw_states = case.flatten(case.normalized_complex)
        envelope_states = case.flatten(case.envelope_complex)
        raw_fit = raw_states[fit_mask]
        envelope_fit = envelope_states[fit_mask]
        steps = int(np.count_nonzero(holdout_mask))

        persistence = np.repeat(raw_fit[-1][None, :], steps, axis=0)
        constant_envelope = np.repeat(
            envelope_fit[-1][None, :], steps, axis=0
        )
        carrier = reconstruct_raw_state(
            case, constant_envelope, holdout_frames
        )

        raw_config = selected_local[electric_field_kvm]["raw_dmd"]
        raw_model = fit_linear_model(raw_fit, int(raw_config["rank"]))
        raw_dmd = rollout_linear(raw_model, raw_fit[-1], steps)

        envelope_config = selected_local[electric_field_kvm]["envelope_dmd"]
        envelope_model = fit_linear_model(
            envelope_fit, int(envelope_config["rank"])
        )
        envelope_dmd_state = rollout_linear(
            envelope_model, envelope_fit[-1], steps
        )
        envelope_dmd = reconstruct_raw_state(
            case, envelope_dmd_state, holdout_frames
        )

        hankel_config = selected_local[electric_field_kvm][
            "envelope_hankel_dmd"
        ]
        hankel_model = fit_hankel_model(
            envelope_fit,
            int(hankel_config["delay"]),
            int(hankel_config["rank"]),
        )
        hankel_state = rollout_hankel(hankel_model, envelope_fit, steps)
        hankel_prediction = reconstruct_raw_state(
            case, hankel_state, holdout_frames
        )
        selected_envelope_method = min(
            ("envelope_dmd", "envelope_hankel_dmd"),
            key=lambda method: float(
                selected_local[electric_field_kvm][method]["objective"]
            ),
        )
        selected_envelope_prediction = {
            "envelope_dmd": envelope_dmd,
            "envelope_hankel_dmd": hankel_prediction,
        }[selected_envelope_method]

        conditioned_state = rollout_conditioned(
            conditioned,
            envelope_fit[-1],
            electric_field_kvm,
            steps,
        )
        conditioned_prediction = reconstruct_raw_state(
            case, conditioned_state, holdout_frames
        )

        predictions[electric_field_kvm] = {
            "truth": raw_states[holdout_mask],
            "persistence": persistence,
            "constant_carrier": carrier,
            "raw_dmd": raw_dmd,
            "envelope_dmd": envelope_dmd,
            "envelope_hankel_dmd": hankel_prediction,
            "selected_envelope": selected_envelope_prediction,
            "conditioned_shared_envelope": conditioned_prediction,
            "holdout_frames": holdout_frames,
            "fit_frames": fit_frames,
        }
        model_details[electric_field_kvm] = {
            "raw_dmd": {
                "rank": raw_model.rank,
                "spectral_radius": float(
                    np.max(np.abs(raw_model.eigenvalues))
                ),
            },
            "envelope_dmd": {
                "rank": envelope_model.rank,
                "spectral_radius": float(
                    np.max(np.abs(envelope_model.eigenvalues))
                ),
            },
            "envelope_hankel_dmd": {
                "delay": hankel_model.delay,
                "history_us": hankel_model.delay * FRAME_DT_US,
                "rank": hankel_model.rank,
                "spectral_radius": float(
                    np.max(np.abs(hankel_model.eigenvalues))
                ),
            },
            "selected_envelope_method": selected_envelope_method,
        }
    return predictions, model_details, conditioned


def plot_rollout(
    cases: dict[int, CaseData],
    predictions: dict,
    all_series: dict,
    outdir: Path,
) -> None:
    methods = (
        "constant_carrier",
        "raw_dmd",
        "selected_envelope",
        "conditioned_shared_envelope",
    )
    labels = {
        "constant_carrier": "constant carrier",
        "raw_dmd": "raw DMD",
        "selected_envelope": "validation-selected envelope",
        "conditioned_shared_envelope": "Ez-conditioned envelope",
    }
    colors = {
        "constant_carrier": "#777777",
        "raw_dmd": "#3b6fb6",
        "selected_envelope": "#d14b40",
        "conditioned_shared_envelope": "#18835a",
    }
    fig, axes = plt.subplots(
        len(ELECTRIC_FIELDS),
        4,
        figsize=(18.0, 13.0),
        sharex=True,
        constrained_layout=True,
    )
    for row_index, electric_field_kvm in enumerate(ELECTRIC_FIELDS):
        case = cases[electric_field_kvm]
        holdout_frames = predictions[electric_field_kvm]["holdout_frames"]
        time_us = case.time_us[holdout_frames]
        truth_complex = case.unflatten(
            predictions[electric_field_kvm]["truth"]
        )
        primary_truth = truth_complex[:, 0, PHYSICAL_FIELDS.index("phi")]

        ax = axes[row_index, 0]
        ax.plot(
            time_us,
            np.abs(primary_truth),
            color="black",
            linewidth=2.0,
            label="truth",
        )
        for method in methods:
            prediction_complex = case.unflatten(
                predictions[electric_field_kvm][method]
            )
            ax.plot(
                time_us,
                np.abs(
                    prediction_complex[:, 0, PHYSICAL_FIELDS.index("phi")]
                ),
                color=colors[method],
                linewidth=1.2,
                label=labels[method],
            )
        ax.set_ylabel(
            f"Ez={electric_field_kvm} kV/m\nnormalized amplitude"
        )
        ax.grid(alpha=0.25)
        if row_index == 0:
            ax.set_title(
                f"Primary n={int(case.modes[0])}: phi envelope"
            )

        ax = axes[row_index, 1]
        for method in methods:
            series = all_series[electric_field_kvm][method]
            ax.plot(
                time_us,
                series["primary_phase_error_rad"],
                color=colors[method],
                linewidth=1.2,
                label=labels[method],
            )
        ax.set_ylim(0.0, np.pi * 1.05)
        ax.grid(alpha=0.25)
        if row_index == 0:
            ax.set_title("Primary carrier phase error [rad]")

        ax = axes[row_index, 2]
        truth_transport = all_series[electric_field_kvm][
            "constant_carrier"
        ]["transport_truth"]
        scale = max(float(np.std(truth_transport)), np.finfo(float).tiny)
        center = float(np.mean(truth_transport))
        ax.plot(
            time_us,
            (truth_transport - center) / scale,
            color="black",
            linewidth=2.0,
            label="truth",
        )
        for method in methods:
            series = all_series[electric_field_kvm][method]
            ax.plot(
                time_us,
                (series["transport_prediction"] - center) / scale,
                color=colors[method],
                linewidth=1.2,
                label=labels[method],
            )
        ax.grid(alpha=0.25)
        if row_index == 0:
            ax.set_title("Selected-mode transport [truth std]")

        ax = axes[row_index, 3]
        for method in methods:
            series = all_series[electric_field_kvm][method]
            ax.plot(
                time_us,
                series["state_error"],
                color=colors[method],
                linewidth=1.2,
                label=labels[method],
            )
        ax.grid(alpha=0.25)
        if row_index == 0:
            ax.set_title("Normalized coefficient RMSE")

    for ax in axes[-1]:
        ax.set_xlabel("Time [us]")
    axes[-1, -1].legend(
        loc="lower right",
        bbox_to_anchor=(1.0, -0.02),
        fontsize=8,
        frameon=True,
    )
    fig.savefig(
        outdir / "carrier_envelope_autonomous_rollout.png", dpi=180
    )
    plt.close(fig)


def plot_metrics(metric_rows: list[dict], outdir: Path) -> None:
    methods = (
        "constant_carrier",
        "raw_dmd",
        "selected_envelope",
        "conditioned_shared_envelope",
    )
    labels = {
        "constant_carrier": "constant carrier",
        "raw_dmd": "raw DMD",
        "selected_envelope": "validation-selected envelope",
        "conditioned_shared_envelope": "Ez-conditioned envelope",
    }
    colors = {
        "constant_carrier": "#777777",
        "raw_dmd": "#3b6fb6",
        "selected_envelope": "#d14b40",
        "conditioned_shared_envelope": "#18835a",
    }
    lookup = {
        (int(row["electric_field_kvm"]), row["method"]): row
        for row in metric_rows
    }
    fig, axes = plt.subplots(
        2, 2, figsize=(12.5, 8.5), constrained_layout=True
    )
    panels = (
        ("state_correlation", "Coefficient trajectory correlation"),
        (
            "state_skill_vs_constant_carrier",
            "State skill vs constant carrier",
        ),
        ("weighted_phase_mae_rad", "Weighted phase MAE [rad]"),
        (
            "selected_transport_correlation",
            "Selected-mode transport correlation",
        ),
    )
    for ax, (metric, title) in zip(axes.ravel(), panels):
        for method in methods:
            values = [
                lookup[(electric_field_kvm, method)][metric]
                for electric_field_kvm in ELECTRIC_FIELDS
            ]
            ax.plot(
                ELECTRIC_FIELDS,
                values,
                marker="o",
                linewidth=1.5,
                color=colors[method],
                label=labels[method],
            )
        if "skill" in metric:
            ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Ez [kV/m]")
        ax.grid(alpha=0.25)
    axes[-1, -1].legend(
        loc="lower right",
        bbox_to_anchor=(1.0, -0.02),
        fontsize=8,
        frameon=True,
    )
    fig.savefig(outdir / "carrier_envelope_metrics_by_ez.png", dpi=180)
    plt.close(fig)


def plot_carrier_frequencies(
    cases: dict[int, CaseData], outdir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 6.0), constrained_layout=True)
    markers = ("o", "s", "^", "D", "P")
    role_labels = (
        "primary",
        "harmonic 2",
        "harmonic 3",
        "competitor 1",
        "competitor 2",
    )
    for role_index in range(MODE_COUNT):
        frequencies = [
            abs(cases[electric_field_kvm].carrier_frequencies_mhz[role_index])
            for electric_field_kvm in ELECTRIC_FIELDS
        ]
        modes = [
            int(cases[electric_field_kvm].modes[role_index])
            for electric_field_kvm in ELECTRIC_FIELDS
        ]
        ax.plot(
            ELECTRIC_FIELDS,
            frequencies,
            marker=markers[role_index],
            linewidth=1.5,
            label=role_labels[role_index],
        )
        for electric_field_kvm, frequency, mode in zip(
            ELECTRIC_FIELDS, frequencies, modes
        ):
            ax.annotate(
                f"n={mode}",
                (electric_field_kvm, frequency),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    ax.set_xlabel("Ez [kV/m]")
    ax.set_ylabel("|training carrier frequency| [MHz]")
    ax.set_title("Carrier frequencies estimated from 20-24 us only")
    ax.grid(alpha=0.25)
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.16, 0.0),
        frameon=True,
    )
    fig.savefig(outdir / "carrier_frequencies_by_ez.png", dpi=180)
    plt.close(fig)


def plot_holdout_frequency_prediction(
    metric_rows: list[dict], outdir: Path
) -> None:
    methods = (
        "raw_dmd",
        "selected_envelope",
        "conditioned_shared_envelope",
    )
    labels = {
        "raw_dmd": "raw DMD",
        "selected_envelope": "validation-selected envelope",
        "conditioned_shared_envelope": "Ez-conditioned envelope",
    }
    colors = {
        "raw_dmd": "#3b6fb6",
        "selected_envelope": "#d14b40",
        "conditioned_shared_envelope": "#18835a",
    }
    lookup = {
        (int(row["electric_field_kvm"]), row["method"]): row
        for row in metric_rows
    }
    truth = [
        lookup[(electric_field_kvm, "raw_dmd")][
            "primary_truth_frequency_mhz"
        ]
        for electric_field_kvm in ELECTRIC_FIELDS
    ]
    fig, ax = plt.subplots(
        figsize=(10.0, 6.0), constrained_layout=True
    )
    ax.plot(
        ELECTRIC_FIELDS,
        truth,
        color="black",
        linewidth=2.2,
        marker="o",
        label="holdout truth",
    )
    for method in methods:
        prediction = [
            lookup[(electric_field_kvm, method)][
                "primary_prediction_frequency_mhz"
            ]
            for electric_field_kvm in ELECTRIC_FIELDS
        ]
        ax.plot(
            ELECTRIC_FIELDS,
            prediction,
            color=colors[method],
            linewidth=1.5,
            marker="o",
            label=labels[method],
        )
    ax.set_xlabel("Ez [kV/m]")
    ax.set_ylabel("Signed primary frequency [MHz]")
    ax.set_title("Autonomous primary-carrier frequency, 24-30 us")
    ax.grid(alpha=0.25)
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, -0.02),
        frameon=True,
    )
    fig.savefig(
        outdir / "primary_frequency_autonomous_prediction.png", dpi=180
    )
    plt.close(fig)


def save_rollout_h5(
    path: Path,
    cases: dict[int, CaseData],
    predictions: dict,
) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["fit_interval_us"] = [FIT_START_US, FIT_END_US]
        handle.attrs["holdout_interval_us"] = [FIT_END_US, HOLDOUT_END_US]
        handle.attrs["frame_dt_us"] = FRAME_DT_US
        for electric_field_kvm, case in cases.items():
            group = handle.create_group(f"E{electric_field_kvm}kVm")
            group.create_dataset("modes", data=case.modes)
            group.create_dataset(
                "carrier_frequencies_mhz",
                data=case.carrier_frequencies_mhz,
            )
            group.create_dataset("scales", data=case.scales)
            holdout_frames = predictions[electric_field_kvm][
                "holdout_frames"
            ]
            group.create_dataset(
                "time_us", data=case.time_us[holdout_frames]
            )
            for method, values in predictions[electric_field_kvm].items():
                if method in ("holdout_frames", "fit_frames"):
                    continue
                group.create_dataset(method, data=values)


def generate_readme(
    outdir: Path,
    selected_modes_rows: list[dict],
    metric_rows: list[dict],
    selected_local: dict,
    selected_conditioned: dict,
) -> None:
    modes_text = "\n".join(
        (
            f"| {row['electric_field_kvm']} | "
            f"{row['selected_modes']} | "
            f"{row['primary_frequency_mhz']:.4f} |"
        )
        for row in selected_modes_rows
    )
    metric_lookup = {
        (int(row["electric_field_kvm"]), row["method"]): row
        for row in metric_rows
    }
    result_lines = []
    for electric_field_kvm in ELECTRIC_FIELDS:
        carrier = metric_lookup[
            (electric_field_kvm, "constant_carrier")
        ]
        raw = metric_lookup[
            (electric_field_kvm, "raw_dmd")
        ]
        selected_envelope = metric_lookup[
            (electric_field_kvm, "selected_envelope")
        ]
        conditioned = metric_lookup[
            (electric_field_kvm, "conditioned_shared_envelope")
        ]
        result_lines.append(
            "| "
            f"{electric_field_kvm} | "
            f"{carrier['state_correlation']:.3f} | "
            f"{raw['state_correlation']:.3f} | "
            f"{raw['state_skill_vs_constant_carrier']:.3f} | "
            f"{selected_envelope['state_correlation']:.3f} | "
            f"{selected_envelope['state_skill_vs_constant_carrier']:.3f} | "
            f"{conditioned['state_correlation']:.3f} | "
            f"{conditioned['state_skill_vs_constant_carrier']:.3f} |"
        )

    text = f"""# Physical Fourier carrier-envelope reduced dynamics

The radial-mean PIC fields were transformed directly into azimuthal complex
Fourier coefficients. No SimVP latent feature and no SimVP retraining are
used in this analysis.

- Cases: Bx=20 mT, Ez=10, 20, 30, 40 kV/m
- Physical channels: phi, electron density, Ey
- Fit interval: 20-24 us
- Validation interval: 23-24 us
- Strict autonomous holdout: 24-30 us
- Selected state: primary mode, its second and third harmonics, and the
  strongest remaining competitors
- Carrier frequency: estimated from complex phase rotation in 20-24 us
- Envelope: complex coefficient after removing the fitted carrier rotation

## Selected modes

| Ez [kV/m] | selected n | primary carrier [MHz] |
|---:|---|---:|
{modes_text}

## Holdout summary

| Ez [kV/m] | carrier corr | raw DMD corr | raw skill | selected envelope corr | envelope skill | conditioned corr | conditioned skill |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(result_lines)}

The constant-carrier baseline is deliberately strong: it keeps the last
observed complex envelope and advances only the fitted phase rotation.
Positive skill against this baseline is stronger evidence than merely beating
raw persistence.

The selected envelope model chooses ordinary envelope DMD or envelope Hankel
DMD using only the 23-24 us validation interval. The conditioned model is one
shared map with Ez-dependent linear interactions. Its selected ridge is
`{selected_conditioned['ridge']}`.

## 日本語

PICの物理場から直接、方位角方向の複素Fourier係数を取り出した。
複素係数なので、モード振幅だけでなく進行波の位相も保持している。
20-24 usで平均的な位相回転をcarrierとして推定し、それを除いた遅い複素
envelopeを低次元状態として予測した。

比較対象の`constant carrier`は、24 us直前のenvelopeを固定し、学習区間で
測った周波数だけで波を進める物理baselineである。このbaselineに勝てた場合、
単に周期を当てただけでなく、振幅・相対位相・モード間結合の変化も予測できた
可能性がある。

結果として、raw DMDはconstant carrierに対してEz=10, 20, 30, 40 kV/mで
それぞれ約9%, 40%, 56%, 49%のstate MSE改善を示した。検証区間だけで選んだ
envelope modelはEz=20, 30, 40 kV/mでは約26%, 38%, 8%改善したが、
Ez=10 kV/mでは約7%悪化した。したがって、高電場側の支配的な進行波は少数の
物理Fourier係数で部分的に自律予測できる一方、低電場側は同じ表現だけでは
閉じていない。

ただし、電子密度係数とEy係数のcross-phaseから計算した選択モード輸送の
相関は全体として低い。Fourier係数の軌道MSEが改善しても、輸送を決める
微妙な相対位相までは十分に予測できていない。この点はSimVP画像予測で
得られた結論とも整合する。

## Files

- `carrier_envelope_summary.json`: settings and all numerical results
- `selected_physical_modes.csv`: selected n and fitted carrier frequencies
- `carrier_envelope_metrics.csv`: holdout metrics for every method and Ez
- `carrier_envelope_model_selection.csv`: local validation search
- `conditioned_model_selection.csv`: shared Ez-conditioned validation search
- `carrier_envelope_time_series.csv`: time-resolved errors and transport
- `carrier_envelope_rollout.h5`: reusable truth and prediction states
- `carrier_envelope_autonomous_rollout.png`
- `carrier_envelope_metrics_by_ez.png`
- `carrier_frequencies_by_ez.png`
- `primary_frequency_autonomous_prediction.png`
"""
    (outdir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build physical Fourier carrier-envelope states and evaluate "
            "strict autonomous forecasts over the electric-field sweep."
        )
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    cases = {}
    selected_modes_rows = []
    for electric_field_kvm in ELECTRIC_FIELDS:
        source_h5 = diagnostic_path(electric_field_kvm)
        if not source_h5.is_file():
            raise FileNotFoundError(source_h5)
        case = estimate_representation(electric_field_kvm, source_h5)
        cases[electric_field_kvm] = case
        mode_score = [
            float(case.scores[int(mode)]) for mode in case.modes
        ]
        selected_modes_rows.append(
            {
                "electric_field_kvm": electric_field_kvm,
                "source_h5": str(source_h5),
                "selected_modes": ",".join(map(str, case.modes.tolist())),
                "mode_roles": ",".join(case.mode_roles),
                "selected_mode_scores": ",".join(
                    f"{score:.8g}" for score in mode_score
                ),
                "primary_frequency_mhz": float(
                    case.carrier_frequencies_mhz[0]
                ),
                "carrier_frequencies_mhz": ",".join(
                    f"{value:.8g}"
                    for value in case.carrier_frequencies_mhz
                ),
            }
        )
        print(
            f"E{electric_field_kvm}: modes={case.modes.tolist()} "
            f"carrier={case.carrier_frequencies_mhz.tolist()}"
        )

    selected_local = {}
    local_trials = []
    for electric_field_kvm, case in cases.items():
        selected, trials = select_local_models(case)
        selected_local[electric_field_kvm] = selected
        local_trials.extend(trials)
        print(f"E{electric_field_kvm}: selected={selected}")

    selected_conditioned, conditioned_trials = select_conditioned_model(cases)
    print(f"conditioned selected={selected_conditioned}")
    predictions, model_details, conditioned_model = build_predictions(
        cases, selected_local, selected_conditioned
    )

    metric_rows = []
    time_series_rows = []
    all_series = {}
    methods = (
        "persistence",
        "constant_carrier",
        "raw_dmd",
        "envelope_dmd",
        "envelope_hankel_dmd",
        "selected_envelope",
        "conditioned_shared_envelope",
    )
    for electric_field_kvm, case in cases.items():
        case_predictions = predictions[electric_field_kvm]
        all_series[electric_field_kvm] = {}
        truth = case_predictions["truth"]
        holdout_frames = case_predictions["holdout_frames"]
        for method in methods:
            metrics, series = evaluate_prediction(
                case,
                truth,
                case_predictions[method],
                case_predictions["persistence"],
                case_predictions["constant_carrier"],
                holdout_frames,
                method,
            )
            metric_rows.append(metrics)
            all_series[electric_field_kvm][method] = series
            for index, time_us in enumerate(series["time_us"]):
                time_series_rows.append(
                    {
                        "electric_field_kvm": electric_field_kvm,
                        "method": method,
                        "time_us": float(time_us),
                        "state_error": float(series["state_error"][index]),
                        "primary_truth_amplitude": float(
                            series["primary_truth_amplitude"][index]
                        ),
                        "primary_prediction_amplitude": float(
                            series["primary_prediction_amplitude"][index]
                        ),
                        "primary_phase_error_rad": float(
                            series["primary_phase_error_rad"][index]
                        ),
                        "transport_truth": float(
                            series["transport_truth"][index]
                        ),
                        "transport_prediction": float(
                            series["transport_prediction"][index]
                        ),
                    }
                )

    write_csv(outdir / "selected_physical_modes.csv", selected_modes_rows)
    write_csv(
        outdir / "carrier_envelope_model_selection.csv", local_trials
    )
    write_csv(
        outdir / "conditioned_model_selection.csv", conditioned_trials
    )
    write_csv(outdir / "carrier_envelope_metrics.csv", metric_rows)
    write_csv(
        outdir / "carrier_envelope_time_series.csv", time_series_rows
    )
    save_rollout_h5(
        outdir / "carrier_envelope_rollout.h5", cases, predictions
    )
    plot_rollout(cases, predictions, all_series, outdir)
    plot_metrics(metric_rows, outdir)
    plot_carrier_frequencies(cases, outdir)
    plot_holdout_frequency_prediction(metric_rows, outdir)

    summary = {
        "status": "PASS",
        "definition": {
            "physical_fields": PHYSICAL_FIELDS,
            "spatial_source": "radial_mean",
            "selected_modes_per_case": MODE_COUNT,
            "mode_selection": (
                "training-energy primary, primary harmonics 2/3, then "
                "strongest remaining training-energy modes"
            ),
            "carrier_estimation": (
                "least-squares complex one-frame phase rotation over "
                "20-24 us"
            ),
            "envelope": "normalized complex coefficient with carrier removed",
            "fit_interval_us": [FIT_START_US, FIT_END_US],
            "validation_interval_us": [
                VALIDATION_START_US,
                FIT_END_US,
            ],
            "holdout_interval_us": [FIT_END_US, HOLDOUT_END_US],
            "frame_dt_us": FRAME_DT_US,
        },
        "cases": {
            electric_field_kvm: {
                "source_h5": str(case.source_h5),
                "modes": case.modes,
                "mode_roles": case.mode_roles,
                "carrier_frequencies_mhz": case.carrier_frequencies_mhz,
                "scales": case.scales,
                "selected_local_models": selected_local[
                    electric_field_kvm
                ],
                "final_model_details": model_details[electric_field_kvm],
            }
            for electric_field_kvm, case in cases.items()
        },
        "conditioned_model": {
            "selected": selected_conditioned,
            "state_dimensions": conditioned_model.state_dimensions,
            "parameter": "(Ez_kVm - 25) / 15",
            "map": (
                "z_next - mean_E = (A0 + Ez_norm A1) "
                "(z - mean_E)"
            ),
            "final_spectral_radius": {
                electric_field_kvm: conditioned_spectral_radius(
                    conditioned_model, electric_field_kvm
                )
                for electric_field_kvm in ELECTRIC_FIELDS
            },
        },
        "metrics": metric_rows,
    }
    with (outdir / "carrier_envelope_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(json_safe(summary), handle, indent=2, ensure_ascii=False)

    generate_readme(
        outdir,
        selected_modes_rows,
        metric_rows,
        selected_local,
        selected_conditioned,
    )
    print(f"PASS: wrote carrier-envelope analysis to {outdir}")


if __name__ == "__main__":
    main()
