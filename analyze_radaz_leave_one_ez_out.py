from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_radaz_cross_phase_transport_ablation as observable_analysis
import analyze_radaz_kinetic_moment_ablation as kinetic
import analyze_radaz_physical_carrier_envelope as base
import analyze_radaz_radial_band_fourier_ablation as radial


OUTPUT_DIR = (
    base.SIMVP_ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_leave_one_ez_out"
)
RADIAL_BANDS = 4
COMMON_MODES = np.arange(1, 31, dtype=np.int64)
VARIANTS = ("base_observables", "plus_axial_current")
METHODS = ("shared_blind", "shared_ez_conditioned")
RANK_CANDIDATES = (8, 12, 16, 24, 32, 40, 48, 60, 80, 100)
RIDGE_CANDIDATES = (1.0e-8, 1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1)
MAGNETIC_FIELD_T = 0.020


@dataclass
class RawCase:
    electric_field_kvm: int
    time_us: np.ndarray
    frame: np.ndarray
    radial_weights: np.ndarray
    fourier: np.ndarray
    cross_phase: np.ndarray
    transport: np.ndarray
    axial_current: np.ndarray


@dataclass
class FoldScales:
    fourier: np.ndarray
    transport: np.ndarray
    axial_current: np.ndarray


@dataclass
class StateCase:
    raw: RawCase
    variant: str
    states: np.ndarray
    slices: dict[str, slice | None]
    scales: FoldScales


@dataclass
class SharedModel:
    mean: np.ndarray
    basis: np.ndarray
    weights: np.ndarray
    conditioned: bool
    ridge: float
    rank: int


def complex_to_real(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(len(values), -1)
    return np.concatenate((flat.real, flat.imag), axis=1)


def real_to_complex(
    values: np.ndarray,
    shape: tuple[int, ...],
) -> np.ndarray:
    half = values.shape[1] // 2
    flat = values[:, :half] + 1j * values[:, half:]
    return flat.reshape((len(values),) + shape)


def rms_scale(arrays: list[np.ndarray]) -> np.ndarray:
    total = None
    count = 0
    for array in arrays:
        energy = np.sum(np.abs(array) ** 2, axis=0)
        total = energy if total is None else total + energy
        count += len(array)
    if total is None or count == 0:
        raise ValueError("Cannot compute a scale from an empty collection")
    scale = np.sqrt(total / count)
    return np.maximum(scale, np.finfo(np.float64).tiny)


def extract_raw_case(electric_field_kvm: int) -> RawCase:
    source_h5 = radial.analysis_fields_path(electric_field_kvm)
    time_us, frame, signals, _, radial_weights = (
        radial.extract_band_signals(source_h5, (RADIAL_BANDS,))
    )
    coefficients = np.fft.rfft(
        signals[RADIAL_BANDS], axis=2, norm="forward"
    )[:, :, COMMON_MODES, :]
    physical_cross = (
        coefficients[..., base.PHYSICAL_FIELDS.index("electron_den")]
        * np.conj(
            coefficients[..., base.PHYSICAL_FIELDS.index("efy")]
        )
    )
    cross_magnitude = np.abs(physical_cross)
    cross_phase = np.divide(
        physical_cross,
        cross_magnitude,
        out=np.zeros_like(physical_cross),
        where=cross_magnitude > np.finfo(float).tiny,
    )
    transport = -2.0 * np.real(physical_cross) / MAGNETIC_FIELD_T

    _, current = kinetic.extract_kinetic_signals(source_h5, time_us)
    axial_current = np.fft.rfft(
        current[..., 2], axis=2, norm="forward"
    )[:, :, COMMON_MODES]
    return RawCase(
        electric_field_kvm=electric_field_kvm,
        time_us=time_us,
        frame=frame,
        radial_weights=radial_weights[RADIAL_BANDS],
        fourier=coefficients,
        cross_phase=cross_phase,
        transport=transport,
        axial_current=axial_current,
    )


def interval_mask(
    time_us: np.ndarray,
    start_us: float,
    end_us: float,
    include_start: bool = True,
) -> np.ndarray:
    if include_start:
        lower = time_us >= start_us - 1.0e-10
    else:
        lower = time_us > start_us + 1.0e-10
    return lower & (time_us <= end_us + 1.0e-10)


def compute_fold_scales(
    cases: dict[int, RawCase],
    training_fields: tuple[int, ...],
    start_us: float,
    end_us: float,
) -> FoldScales:
    fourier = []
    transport = []
    current = []
    for electric_field_kvm in training_fields:
        case = cases[electric_field_kvm]
        mask = interval_mask(case.time_us, start_us, end_us)
        fourier.append(case.fourier[mask])
        transport.append(case.transport[mask])
        current.append(case.axial_current[mask])
    return FoldScales(
        fourier=rms_scale(fourier),
        transport=rms_scale(transport),
        axial_current=rms_scale(current),
    )


def build_state_case(
    raw: RawCase,
    scales: FoldScales,
    variant: str,
) -> StateCase:
    blocks: list[np.ndarray] = []
    slices: dict[str, slice | None] = {
        "fourier": None,
        "cross_phase_real": None,
        "cross_phase_imag": None,
        "transport": None,
        "axial_current": None,
    }
    offset = 0

    normalized_fourier = raw.fourier / scales.fourier[None, ...]
    values = complex_to_real(normalized_fourier)
    slices["fourier"] = slice(offset, offset + values.shape[1])
    blocks.append(values)
    offset += values.shape[1]

    phase_flat = raw.cross_phase.reshape(len(raw.time_us), -1)
    width = phase_flat.shape[1]
    slices["cross_phase_real"] = slice(offset, offset + width)
    blocks.append(phase_flat.real)
    offset += width
    slices["cross_phase_imag"] = slice(offset, offset + width)
    blocks.append(phase_flat.imag)
    offset += width

    values = (
        raw.transport / scales.transport[None, ...]
    ).reshape(len(raw.time_us), -1)
    slices["transport"] = slice(offset, offset + values.shape[1])
    blocks.append(values)
    offset += values.shape[1]

    if variant == "plus_axial_current":
        normalized_current = (
            raw.axial_current / scales.axial_current[None, ...]
        )
        values = complex_to_real(normalized_current)
        slices["axial_current"] = slice(
            offset, offset + values.shape[1]
        )
        blocks.append(values)
    elif variant != "base_observables":
        raise ValueError(f"Unknown variant: {variant}")

    return StateCase(
        raw=raw,
        variant=variant,
        states=np.concatenate(blocks, axis=1),
        slices=slices,
        scales=scales,
    )


def parameter_value(electric_field_kvm: int) -> float:
    return (electric_field_kvm - 25.0) / 15.0


def transition_features(
    latent: np.ndarray,
    electric_field_kvm: int,
    conditioned: bool,
) -> np.ndarray:
    ones = np.ones((len(latent), 1), dtype=np.float64)
    if not conditioned:
        return np.concatenate((latent, ones), axis=1)
    parameter = parameter_value(electric_field_kvm)
    return np.concatenate(
        (
            latent,
            parameter * latent,
            ones,
            parameter * ones,
        ),
        axis=1,
    )


def fit_shared_model(
    states_by_case: dict[int, np.ndarray],
    rank: int,
    ridge: float,
    conditioned: bool,
) -> SharedModel:
    stacked = np.concatenate(list(states_by_case.values()), axis=0)
    mean = np.mean(stacked, axis=0)
    centered = stacked - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    effective_rank = min(rank, len(vt))
    basis = vt[:effective_rank].T

    features = []
    targets = []
    for electric_field_kvm, states in states_by_case.items():
        latent = (states - mean) @ basis
        features.append(
            transition_features(
                latent[:-1], electric_field_kvm, conditioned
            )
        )
        targets.append(latent[1:])
    x = np.concatenate(features, axis=0)
    y = np.concatenate(targets, axis=0)
    gram = x.T @ x
    ridge_scale = max(
        float(np.mean(np.diag(gram))), np.finfo(float).tiny
    )
    regularization = ridge * ridge_scale * np.eye(
        gram.shape[0], dtype=np.float64
    )
    weights = np.linalg.pinv(
        gram + regularization, rcond=1.0e-10
    ) @ (x.T @ y)
    return SharedModel(
        mean=mean,
        basis=basis,
        weights=weights,
        conditioned=conditioned,
        ridge=ridge,
        rank=effective_rank,
    )


def effective_matrix(
    model: SharedModel, electric_field_kvm: int
) -> np.ndarray:
    rank = model.rank
    if not model.conditioned:
        return model.weights[:rank]
    parameter = parameter_value(electric_field_kvm)
    return (
        model.weights[:rank]
        + parameter * model.weights[rank : 2 * rank]
    )


def spectral_radius(
    model: SharedModel, electric_field_kvm: int
) -> float:
    return float(
        np.max(
            np.abs(
                np.linalg.eigvals(
                    effective_matrix(model, electric_field_kvm)
                )
            )
        )
    )


def rollout_shared(
    model: SharedModel,
    initial_state: np.ndarray,
    electric_field_kvm: int,
    steps: int,
) -> np.ndarray:
    latent = (initial_state - model.mean) @ model.basis
    prediction = np.empty(
        (steps, len(model.mean)), dtype=np.float64
    )
    for index in range(steps):
        feature = transition_features(
            latent[None, :],
            electric_field_kvm,
            model.conditioned,
        )[0]
        latent = feature @ model.weights
        state = model.mean + latent @ model.basis.T
        if (
            not np.all(np.isfinite(state))
            or np.max(np.abs(state)) > 1.0e8
        ):
            prediction[index:] = np.nan
            break
        prediction[index] = state
    return prediction


def prediction_mse(
    truth: np.ndarray, prediction: np.ndarray
) -> float:
    finite = np.isfinite(truth) & np.isfinite(prediction)
    if not np.any(finite):
        return float("inf")
    return float(np.mean((truth[finite] - prediction[finite]) ** 2))


def select_model(
    state_cases: dict[int, StateCase],
    training_fields: tuple[int, ...],
    conditioned: bool,
) -> tuple[dict, list[dict]]:
    subtrain_by_case = {}
    validation_by_case = {}
    initial_by_case = {}
    for electric_field_kvm in training_fields:
        case = state_cases[electric_field_kvm]
        subtrain_mask = interval_mask(
            case.raw.time_us,
            base.FIT_START_US,
            base.VALIDATION_START_US,
            include_start=True,
        ) & (case.raw.time_us < base.VALIDATION_START_US - 1.0e-10)
        validation_mask = interval_mask(
            case.raw.time_us,
            base.VALIDATION_START_US,
            base.FIT_END_US,
            include_start=True,
        )
        subtrain_by_case[electric_field_kvm] = case.states[
            subtrain_mask
        ]
        validation_by_case[electric_field_kvm] = case.states[
            validation_mask
        ]
        initial_by_case[electric_field_kvm] = case.states[
            np.flatnonzero(subtrain_mask)[-1]
        ]

    trials = []
    best = {"objective": float("inf")}
    for rank in RANK_CANDIDATES:
        for ridge in RIDGE_CANDIDATES:
            try:
                model = fit_shared_model(
                    subtrain_by_case,
                    rank,
                    ridge,
                    conditioned,
                )
                case_mse = []
                for electric_field_kvm in training_fields:
                    truth = validation_by_case[electric_field_kvm]
                    prediction = rollout_shared(
                        model,
                        initial_by_case[electric_field_kvm],
                        electric_field_kvm,
                        len(truth),
                    )
                    case_mse.append(prediction_mse(truth, prediction))
                validation_mse = float(np.mean(case_mse))
                radii = [
                    spectral_radius(model, electric_field_kvm)
                    for electric_field_kvm in base.ELECTRIC_FIELDS
                ]
                maximum_radius = float(np.max(radii))
                objective = validation_mse + max(
                    0.0, maximum_radius - 1.002
                ) * 100.0
            except (ValueError, np.linalg.LinAlgError):
                validation_mse = float("inf")
                maximum_radius = float("inf")
                objective = float("inf")
                radii = [float("inf")] * len(base.ELECTRIC_FIELDS)
                case_mse = [float("inf")] * len(training_fields)
            row = {
                "requested_rank": rank,
                "ridge": ridge,
                "conditioned": conditioned,
                "validation_mse": validation_mse,
                "maximum_spectral_radius": maximum_radius,
                "objective": objective,
            }
            for electric_field_kvm, value in zip(
                training_fields, case_mse
            ):
                row[f"validation_mse_e{electric_field_kvm}"] = value
            for electric_field_kvm, value in zip(
                base.ELECTRIC_FIELDS, radii
            ):
                row[f"spectral_radius_e{electric_field_kvm}"] = value
            trials.append(row)
            if objective < best["objective"]:
                best = row
    return best, trials


def integrate_transport(
    raw: RawCase, modal_transport: np.ndarray
) -> np.ndarray:
    per_band = np.sum(modal_transport, axis=2)
    return np.einsum(
        "b,tb->t", raw.radial_weights, per_band, optimize=True
    )


def safe_skill(
    truth: np.ndarray,
    prediction: np.ndarray,
    persistence: np.ndarray,
) -> float:
    mse = prediction_mse(truth, prediction)
    baseline_mse = prediction_mse(truth, persistence)
    if not np.isfinite(mse) or baseline_mse <= 0.0:
        return float("-inf")
    return 1.0 - mse / baseline_mse


def evaluate_prediction(
    case: StateCase,
    truth: np.ndarray,
    prediction: np.ndarray,
    persistence: np.ndarray,
    time_us: np.ndarray,
) -> tuple[dict, dict]:
    fourier_slice = case.slices["fourier"]
    phase_real_slice = case.slices["cross_phase_real"]
    phase_imag_slice = case.slices["cross_phase_imag"]
    transport_slice = case.slices["transport"]
    assert fourier_slice is not None
    assert phase_real_slice is not None
    assert phase_imag_slice is not None
    assert transport_slice is not None

    fourier_shape = case.raw.fourier.shape[1:]
    truth_fourier = real_to_complex(
        truth[:, fourier_slice], fourier_shape
    )
    prediction_fourier = real_to_complex(
        prediction[:, fourier_slice], fourier_shape
    )
    persistence_fourier = real_to_complex(
        persistence[:, fourier_slice], fourier_shape
    )
    truth_physical = (
        truth_fourier * case.scales.fourier[None, ...]
    )
    prediction_physical = (
        prediction_fourier * case.scales.fourier[None, ...]
    )
    truth_cross = (
        truth_physical[..., base.PHYSICAL_FIELDS.index("electron_den")]
        * np.conj(
            truth_physical[..., base.PHYSICAL_FIELDS.index("efy")]
        )
    )
    phase_weights = np.abs(truth_cross)
    truth_phase = np.divide(
        truth_cross,
        phase_weights,
        out=np.zeros_like(truth_cross),
        where=phase_weights > np.finfo(float).tiny,
    )
    prediction_phase = (
        prediction[:, phase_real_slice]
        + 1j * prediction[:, phase_imag_slice]
    ).reshape(truth_phase.shape)
    prediction_phase_magnitude = np.abs(prediction_phase)
    prediction_phase = np.divide(
        prediction_phase,
        prediction_phase_magnitude,
        out=np.zeros_like(prediction_phase),
        where=prediction_phase_magnitude > np.finfo(float).tiny,
    )
    phase_mae = observable_analysis.weighted_phase_mae(
        truth_phase, prediction_phase, phase_weights
    )

    transport_shape = case.raw.transport.shape[1:]
    truth_modal_transport = (
        truth[:, transport_slice].reshape(
            (len(truth),) + transport_shape
        )
        * case.scales.transport[None, ...]
    )
    prediction_modal_transport = (
        prediction[:, transport_slice].reshape(
            (len(prediction),) + transport_shape
        )
        * case.scales.transport[None, ...]
    )
    persistence_modal_transport = (
        persistence[:, transport_slice].reshape(
            (len(persistence),) + transport_shape
        )
        * case.scales.transport[None, ...]
    )
    truth_transport = integrate_transport(
        case.raw, truth_modal_transport
    )
    prediction_transport = integrate_transport(
        case.raw, prediction_modal_transport
    )
    persistence_transport = integrate_transport(
        case.raw, persistence_modal_transport
    )
    derived_prediction_transport = integrate_transport(
        case.raw,
        -2.0
        * np.real(
            prediction_physical[
                ..., base.PHYSICAL_FIELDS.index("electron_den")
            ]
            * np.conj(
                prediction_physical[
                    ..., base.PHYSICAL_FIELDS.index("efy")
                ]
            )
        )
        / MAGNETIC_FIELD_T,
    )

    current_correlation = float("nan")
    current_skill = float("nan")
    current_slice = case.slices["axial_current"]
    if current_slice is not None:
        current_shape = case.raw.axial_current.shape[1:]
        truth_current = real_to_complex(
            truth[:, current_slice], current_shape
        )
        prediction_current = real_to_complex(
            prediction[:, current_slice], current_shape
        )
        persistence_current = real_to_complex(
            persistence[:, current_slice], current_shape
        )
        current_correlation = base.safe_correlation(
            complex_to_real(truth_current),
            complex_to_real(prediction_current),
        )
        current_skill = safe_skill(
            complex_to_real(truth_current),
            complex_to_real(prediction_current),
            complex_to_real(persistence_current),
        )

    metrics = {
        "full_state_correlation": base.safe_correlation(
            truth, prediction
        ),
        "full_state_skill_vs_persistence": safe_skill(
            truth, prediction, persistence
        ),
        "fourier_state_correlation": base.safe_correlation(
            truth[:, fourier_slice], prediction[:, fourier_slice]
        ),
        "fourier_state_skill_vs_persistence": safe_skill(
            truth[:, fourier_slice],
            prediction[:, fourier_slice],
            persistence[:, fourier_slice],
        ),
        "cross_phase_mae_rad": phase_mae,
        "transport_correlation": base.safe_correlation(
            truth_transport, prediction_transport
        ),
        "transport_skill_vs_persistence": safe_skill(
            truth_transport[:, None],
            prediction_transport[:, None],
            persistence_transport[:, None],
        ),
        "transport_consistency_correlation": base.safe_correlation(
            derived_prediction_transport, prediction_transport
        ),
        "axial_current_correlation": current_correlation,
        "axial_current_skill_vs_persistence": current_skill,
        "finite_fraction": float(
            np.mean(np.isfinite(prediction).all(axis=1))
        ),
    }
    series = {
        "time_us": time_us,
        "transport_truth": truth_transport,
        "transport_prediction": prediction_transport,
        "transport_persistence": persistence_transport,
        "transport_derived_prediction": derived_prediction_transport,
    }
    return metrics, series


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metrics(rows: list[dict], outdir: Path) -> None:
    metric_specs = (
        ("fourier_state_correlation", "Fourier-state correlation"),
        (
            "fourier_state_skill_vs_persistence",
            "Fourier-state skill vs persistence",
        ),
        ("cross_phase_mae_rad", "Cross-phase MAE [rad]"),
        ("transport_correlation", "Modal-transport correlation"),
        (
            "transport_skill_vs_persistence",
            "Transport skill vs persistence",
        ),
        ("axial_current_correlation", "Axial-current correlation"),
    )
    labels = {
        ("base_observables", "shared_blind"): "base, blind",
        (
            "base_observables",
            "shared_ez_conditioned",
        ): "base, Ez-conditioned",
        ("plus_axial_current", "shared_blind"): "+ Jz, blind",
        (
            "plus_axial_current",
            "shared_ez_conditioned",
        ): "+ Jz, Ez-conditioned",
    }
    markers = ("o", "s", "^", "D")
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    x = np.asarray(base.ELECTRIC_FIELDS, dtype=float)
    for axis, (metric, title) in zip(axes.ravel(), metric_specs):
        for marker, key in zip(markers, labels):
            variant, method = key
            values = []
            for electric_field_kvm in base.ELECTRIC_FIELDS:
                row = next(
                    item
                    for item in rows
                    if item["heldout_electric_field_kvm"]
                    == electric_field_kvm
                    and item["variant"] == variant
                    and item["method"] == method
                )
                values.append(float(row[metric]))
            display_values = np.asarray(values, dtype=float)
            display_floor = None
            if metric == "fourier_state_skill_vs_persistence":
                display_floor = -1.0
            elif metric == "transport_skill_vs_persistence":
                display_floor = -100.0
            if display_floor is not None:
                display_values = np.maximum(
                    display_values, display_floor
                )
            axis.plot(
                x,
                display_values,
                marker=marker,
                linewidth=1.8,
                label=labels[key],
            )
            if display_floor is not None:
                below = np.asarray(values) < display_floor
                if np.any(below):
                    axis.scatter(
                        x[below],
                        np.full(np.count_nonzero(below), display_floor),
                        marker="v",
                        color=axis.lines[-1].get_color(),
                        s=35,
                        zorder=4,
                    )
        axis.axhline(0.0, color="0.55", linewidth=0.8)
        if metric == "fourier_state_skill_vs_persistence":
            axis.set_ylim(-1.08, 1.0)
            axis.text(
                0.01,
                0.02,
                "downward triangles: value < -1",
                transform=axis.transAxes,
                fontsize=8,
                va="bottom",
            )
        elif metric == "transport_skill_vs_persistence":
            axis.set_yscale("symlog", linthresh=1.0, linscale=1.0)
            axis.set_ylim(-120.0, 1.0)
            axis.text(
                0.01,
                0.02,
                "downward triangles: value < -100",
                transform=axis.transAxes,
                fontsize=8,
                va="bottom",
            )
        axis.set_title(title)
        axis.set_xlabel("Held-out Ez [kV/m]")
        axis.grid(alpha=0.25)
    axes[1, 2].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Strict leave-one-Ez-out autonomous prediction, 24-30 us"
    )
    fig.tight_layout()
    fig.savefig(
        outdir / "leave_one_ez_out_metrics.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_transport(
    series: dict[tuple[int, str, str], dict],
    outdir: Path,
) -> None:
    fig, axes = plt.subplots(
        len(base.ELECTRIC_FIELDS), 1, figsize=(13, 11), sharex=True
    )
    colors = {
        "base_observables": "#3366aa",
        "plus_axial_current": "#cc5500",
    }
    for axis, electric_field_kvm in zip(
        axes, base.ELECTRIC_FIELDS
    ):
        reference = series[
            (
                electric_field_kvm,
                "base_observables",
                "shared_ez_conditioned",
            )
        ]
        axis.plot(
            reference["time_us"],
            reference["transport_truth"],
            color="black",
            linewidth=1.6,
            label="PIC truth",
        )
        axis.plot(
            reference["time_us"],
            reference["transport_persistence"],
            color="0.55",
            linestyle="--",
            linewidth=1.2,
            label="persistence",
        )
        for variant in VARIANTS:
            current = series[
                (
                    electric_field_kvm,
                    variant,
                    "shared_ez_conditioned",
                )
            ]
            axis.plot(
                current["time_us"],
                current["transport_prediction"],
                color=colors[variant],
                linewidth=1.1,
                label=(
                    "conditioned base"
                    if variant == "base_observables"
                    else "conditioned + Jz"
                ),
            )
        truth_and_persistence = np.concatenate(
            (
                reference["transport_truth"],
                reference["transport_persistence"],
            )
        )
        finite_reference = truth_and_persistence[
            np.isfinite(truth_and_persistence)
        ]
        if len(finite_reference):
            lower = float(np.min(finite_reference))
            upper = float(np.max(finite_reference))
            span = max(upper - lower, abs(upper), abs(lower), 1.0)
            axis.set_ylim(lower - 0.12 * span, upper + 0.12 * span)
        offscale = []
        for variant in VARIANTS:
            values = series[
                (
                    electric_field_kvm,
                    variant,
                    "shared_ez_conditioned",
                )
            ]["transport_prediction"]
            finite = values[np.isfinite(values)]
            finite_fraction = len(finite) / len(values)
            outside = (
                len(finite)
                and (
                    np.min(finite) < axis.get_ylim()[0]
                    or np.max(finite) > axis.get_ylim()[1]
                )
            )
            if finite_fraction < 1.0 or outside:
                label = "base" if variant == "base_observables" else "+Jz"
                offscale.append(
                    f"{label}: off-scale, finite={finite_fraction:.1%}"
                )
        if offscale:
            axis.text(
                0.01,
                0.97,
                "\n".join(offscale),
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "0.75",
                    "alpha": 0.9,
                },
            )
        axis.set_ylabel(f"Ez={electric_field_kvm}\ntransport")
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right", fontsize=8)
    axes[-1].set_xlabel("Time [us]")
    fig.suptitle("Leave-one-Ez-out modal-transport rollouts")
    fig.tight_layout()
    fig.savefig(
        outdir / "leave_one_ez_out_transport_rollouts.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_h5(
    outdir: Path,
    metrics: list[dict],
    series: dict[tuple[int, str, str], dict],
) -> None:
    with h5py.File(
        outdir / "leave_one_ez_out_rollouts.h5", "w"
    ) as handle:
        handle.attrs["heldout_interval_us"] = [
            base.FIT_END_US,
            base.HOLDOUT_END_US,
        ]
        handle.attrs["common_modes"] = COMMON_MODES
        for row in metrics:
            key = (
                int(row["heldout_electric_field_kvm"]),
                str(row["variant"]),
                str(row["method"]),
            )
            group = handle.create_group(
                f"E{key[0]}/{key[1]}/{key[2]}"
            )
            for name, values in series[key].items():
                group.create_dataset(
                    name, data=np.asarray(values), compression="gzip"
                )
            for name, value in row.items():
                if isinstance(value, (str, bool)):
                    group.attrs[name] = value
                elif np.isscalar(value):
                    group.attrs[name] = value


def generate_readme(outdir: Path, rows: list[dict]) -> None:
    lines = [
        "# Leave-one-Ez-out low-dimensional dynamics",
        "",
        "## 日本語",
        "",
        "Bx=20 mTの4条件から1条件を完全に遷移則の学習対象外とし、",
        "残り3条件だけで共通POD基底と低次元遷移則を同定した。",
        "未学習条件から使うのは24 usの単一初期状態だけであり、",
        "正規化scale、POD基底、遷移則は残り3条件だけから作った。",
        "24-30 usは真値再入力なしで自律予測した。",
        "",
        "- 共通方位角モード: m=1..30（holdout依存の選択なし）",
        "- radial bands: 4",
        "- 状態: base、base + axial electron current Jz",
        "- 比較: Ezを教えないshared blind、Ezとの双線形項を持つconditioned",
        "- 主評価: Fourier状態、cross-phase、modal transport、Jz",
        "",
        "ここでzero-shotとは「除外条件の遷移ペアを学習しない」という意味である。",
        "除外条件から使うのは24 usの単一初期状態だけであるため、",
        "完全なcold-start予測ではなくwarm-start zero-shot dynamicsである。",
        "",
        "## English",
        "",
        "A shared POD transition model is trained on three Ez cases and",
        "evaluated on the excluded Ez without using its transition pairs.",
        "Only the excluded case state at 24 us initializes its forecast;",
        "normalization, POD, and dynamics use the other three cases.",
        "",
        "## Selected models",
        "",
        "| held-out Ez | state | method | rank | ridge | Fourier corr | transport corr |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {heldout_electric_field_kvm} | {variant} | {method} | "
            "{selected_rank} | {selected_ridge:.1e} | "
            "{fourier_state_correlation:.3f} | "
            "{transport_correlation:.3f} |".format(**row)
        )
    (outdir / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict leave-one-Ez-out reduced-order evaluation."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    raw_cases = {}
    for electric_field_kvm in base.ELECTRIC_FIELDS:
        raw_cases[electric_field_kvm] = extract_raw_case(
            electric_field_kvm
        )
        print(f"E{electric_field_kvm}: extracted common-mode state")

    metric_rows = []
    selection_rows = []
    rollout_series = {}
    for heldout in base.ELECTRIC_FIELDS:
        training_fields = tuple(
            value
            for value in base.ELECTRIC_FIELDS
            if value != heldout
        )
        selection_scales = compute_fold_scales(
            raw_cases,
            training_fields,
            base.FIT_START_US,
            base.VALIDATION_START_US,
        )
        final_scales = compute_fold_scales(
            raw_cases,
            training_fields,
            base.FIT_START_US,
            base.FIT_END_US,
        )
        for variant in VARIANTS:
            selection_cases = {
                electric_field_kvm: build_state_case(
                    raw_cases[electric_field_kvm],
                    selection_scales,
                    variant,
                )
                for electric_field_kvm in base.ELECTRIC_FIELDS
            }
            final_cases = {
                electric_field_kvm: build_state_case(
                    raw_cases[electric_field_kvm],
                    final_scales,
                    variant,
                )
                for electric_field_kvm in base.ELECTRIC_FIELDS
            }
            for method in METHODS:
                conditioned = method == "shared_ez_conditioned"
                selected, trials = select_model(
                    selection_cases,
                    training_fields,
                    conditioned,
                )
                for row in trials:
                    selection_rows.append(
                        {
                            "heldout_electric_field_kvm": heldout,
                            "training_fields_kvm": ",".join(
                                map(str, training_fields)
                            ),
                            "variant": variant,
                            "method": method,
                            **row,
                        }
                    )

                fit_states = {}
                for electric_field_kvm in training_fields:
                    current = final_cases[electric_field_kvm]
                    fit_mask = interval_mask(
                        current.raw.time_us,
                        base.FIT_START_US,
                        base.FIT_END_US,
                    )
                    fit_states[electric_field_kvm] = current.states[
                        fit_mask
                    ]
                model = fit_shared_model(
                    fit_states,
                    int(selected["requested_rank"]),
                    float(selected["ridge"]),
                    conditioned,
                )
                heldout_case = final_cases[heldout]
                fit_mask = interval_mask(
                    heldout_case.raw.time_us,
                    base.FIT_START_US,
                    base.FIT_END_US,
                )
                holdout_mask = interval_mask(
                    heldout_case.raw.time_us,
                    base.FIT_END_US,
                    base.HOLDOUT_END_US,
                    include_start=False,
                )
                truth = heldout_case.states[holdout_mask]
                initial = heldout_case.states[
                    np.flatnonzero(fit_mask)[-1]
                ]
                prediction = rollout_shared(
                    model, initial, heldout, len(truth)
                )
                persistence = np.repeat(
                    initial[None, :], len(truth), axis=0
                )
                metrics, series = evaluate_prediction(
                    heldout_case,
                    truth,
                    prediction,
                    persistence,
                    heldout_case.raw.time_us[holdout_mask],
                )
                row = {
                    "heldout_electric_field_kvm": heldout,
                    "training_fields_kvm": ",".join(
                        map(str, training_fields)
                    ),
                    "variant": variant,
                    "method": method,
                    "state_dimensions": heldout_case.states.shape[1],
                    "common_mode_min": int(COMMON_MODES[0]),
                    "common_mode_max": int(COMMON_MODES[-1]),
                    "selected_rank": model.rank,
                    "selected_ridge": model.ridge,
                    "validation_mse": selected["validation_mse"],
                    "validation_maximum_spectral_radius": selected[
                        "maximum_spectral_radius"
                    ],
                    "heldout_spectral_radius": spectral_radius(
                        model, heldout
                    ),
                    **metrics,
                }
                metric_rows.append(row)
                rollout_series[(heldout, variant, method)] = series
                print(
                    f"leave E{heldout} {variant} {method}: "
                    f"rank={model.rank} "
                    f"Fourier corr={metrics['fourier_state_correlation']:.3f} "
                    f"transport corr={metrics['transport_correlation']:.3f}"
                )

    write_csv(outdir / "leave_one_ez_out_metrics.csv", metric_rows)
    write_csv(
        outdir / "leave_one_ez_out_model_selection.csv",
        selection_rows,
    )
    plot_metrics(metric_rows, outdir)
    plot_transport(rollout_series, outdir)
    save_h5(outdir, metric_rows, rollout_series)
    generate_readme(outdir, metric_rows)
    summary = {
        "status": "PASS",
        "definition": {
            "electric_fields_kvm": base.ELECTRIC_FIELDS,
            "common_modes": COMMON_MODES,
            "radial_bands": RADIAL_BANDS,
            "variants": VARIANTS,
            "methods": METHODS,
            "training_interval_us": [
                base.FIT_START_US,
                base.FIT_END_US,
            ],
            "validation_interval_us": [
                base.VALIDATION_START_US,
                base.FIT_END_US,
            ],
            "holdout_interval_us": [
                base.FIT_END_US,
                base.HOLDOUT_END_US,
            ],
            "zero_shot_definition": (
                "No held-out transition pairs enter POD or dynamics fitting; "
                "only the held-out state at 24 us initializes the forecast. "
                "Normalization, POD, and dynamics use the other cases."
            ),
        },
        "metrics": metric_rows,
    }
    (outdir / "leave_one_ez_out_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2),
        encoding="utf-8",
    )
    print(f"PASS: wrote leave-one-Ez-out analysis to {outdir}")


if __name__ == "__main__":
    main()
