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

import analyze_radaz_physical_carrier_envelope as base


DEFAULT_OUTDIR = (
    base.SIMVP_ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_radial_band_fourier_ablation"
)
RADIAL_BAND_COUNTS = (1, 2, 4, 8)
RADIAL_MIN_M = 0.09e-2
RADIAL_MAX_M = 1.19e-2
RANK_CANDIDATES = (
    4,
    8,
    12,
    16,
    20,
    24,
    30,
    40,
    60,
    80,
    100,
    120,
    160,
    200,
)
METHODS = (
    "persistence",
    "constant_carrier",
    "raw_dmd",
    "envelope_dmd",
    "selected_dmd",
)


@dataclass
class BandCaseData:
    electric_field_kvm: int
    source_h5: Path
    time_us: np.ndarray
    frame: np.ndarray
    band_count: int
    radial_edges_m: np.ndarray
    radial_weights: np.ndarray
    modes: np.ndarray
    mode_roles: list[str]
    scales: np.ndarray
    global_scales: np.ndarray
    carrier_angles: np.ndarray
    carrier_frequencies_mhz: np.ndarray
    raw_complex: np.ndarray
    normalized_complex: np.ndarray
    envelope_complex: np.ndarray

    @property
    def state_dimensions(self) -> int:
        return (
            2
            * self.band_count
            * len(self.modes)
            * len(base.PHYSICAL_FIELDS)
        )

    def flatten(self, values: np.ndarray) -> np.ndarray:
        flat = values.reshape(len(values), -1)
        return np.concatenate((flat.real, flat.imag), axis=1)

    def unflatten(self, values: np.ndarray) -> np.ndarray:
        half = values.shape[1] // 2
        complex_flat = values[:, :half] + 1j * values[:, half:]
        return complex_flat.reshape(
            len(values),
            self.band_count,
            len(self.modes),
            len(base.PHYSICAL_FIELDS),
        )

    def remodulate(
        self,
        envelope_states: np.ndarray,
        frame_indices: np.ndarray,
    ) -> np.ndarray:
        envelope = self.unflatten(envelope_states)
        phase = np.exp(
            1j
            * frame_indices[:, None, None, None]
            * self.carrier_angles[None, None, :, None]
        )
        return envelope * phase

    def physical_coefficients(self, normalized: np.ndarray) -> np.ndarray:
        return normalized * self.scales[None, :, :, :]

    def collapse_physical(self, physical: np.ndarray) -> np.ndarray:
        return np.einsum(
            "b,tbmf->tmf",
            self.radial_weights,
            physical,
            optimize=True,
        )

    def collapse_normalized(self, normalized: np.ndarray) -> np.ndarray:
        physical = self.physical_coefficients(normalized)
        collapsed = self.collapse_physical(physical)
        return collapsed / self.global_scales[None, :, :]

    def collapse_state(self, state: np.ndarray) -> np.ndarray:
        collapsed = self.collapse_normalized(self.unflatten(state))
        flat = collapsed.reshape(len(collapsed), -1)
        return np.concatenate((flat.real, flat.imag), axis=1)


def analysis_fields_path(electric_field_kvm: int) -> Path:
    name = base.case_name(electric_field_kvm)
    return base.PIC_ROOT / name / name / "analysis_fields_uncompressed.h5"


def radial_masks(
    x_m: np.ndarray,
    band_count: int,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    selected = np.flatnonzero(
        (x_m >= RADIAL_MIN_M - 1.0e-15)
        & (x_m <= RADIAL_MAX_M + 1.0e-15)
    )
    if len(selected) < band_count:
        raise ValueError(
            f"Only {len(selected)} radial nodes for {band_count} bands"
        )
    edges = np.linspace(RADIAL_MIN_M, RADIAL_MAX_M, band_count + 1)
    masks = []
    for index in range(band_count):
        if index + 1 == band_count:
            mask = selected[
                (x_m[selected] >= edges[index] - 1.0e-15)
                & (x_m[selected] <= edges[index + 1] + 1.0e-15)
            ]
        else:
            mask = selected[
                (x_m[selected] >= edges[index] - 1.0e-15)
                & (x_m[selected] < edges[index + 1] - 1.0e-15)
            ]
        if len(mask) < 2:
            raise ValueError(
                f"Radial band {index} has only {len(mask)} nodes"
            )
        masks.append(mask)
    counts = np.asarray([len(mask) for mask in masks], dtype=np.float64)
    return edges, masks, counts / np.sum(counts)


def extract_band_signals(
    source_h5: Path,
    band_counts: tuple[int, ...],
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
]:
    with h5py.File(source_h5, "r") as handle:
        all_time_us = (
            np.asarray(handle["axes/time_s"], dtype=np.float64) * 1.0e6
        )
        all_frames = np.asarray(handle["axes/frame_id"], dtype=np.int64)
        x_m = np.asarray(handle["axes/x_m"], dtype=np.float64)
        y_m = np.asarray(handle["axes/y_m"], dtype=np.float64)
        keep = (
            (all_time_us >= base.FIT_START_US - 1.0e-10)
            & (all_time_us <= base.HOLDOUT_END_US + 1.0e-10)
        )
        indices = np.flatnonzero(keep)
        if len(indices) < 3 or not np.all(np.diff(indices) == 1):
            raise ValueError("Requested time interval is not contiguous")
        start = int(indices[0])
        stop = int(indices[-1]) + 1
        time_us = all_time_us[start:stop]
        frame = all_frames[start:stop]
        ny = len(y_m) - 1

        edges_by_count = {}
        masks_by_count = {}
        weights_by_count = {}
        signals_by_count = {}
        for band_count in band_counts:
            edges, masks, weights = radial_masks(x_m, band_count)
            edges_by_count[band_count] = edges
            masks_by_count[band_count] = masks
            weights_by_count[band_count] = weights
            signals_by_count[band_count] = np.empty(
                (
                    len(time_us),
                    band_count,
                    ny,
                    len(base.PHYSICAL_FIELDS),
                ),
                dtype=np.float64,
            )

        _, complete_masks, _ = radial_masks(x_m, 1)
        complete_radial_mask = complete_masks[0]
        radial_start = int(complete_radial_mask[0])
        radial_stop = int(complete_radial_mask[-1]) + 1
        chunk_size = 64
        for field_index, field in enumerate(base.PHYSICAL_FIELDS):
            dataset = handle[f"fields/{field}"]
            for chunk_start in range(start, stop, chunk_size):
                chunk_stop = min(stop, chunk_start + chunk_size)
                local = slice(chunk_start - start, chunk_stop - start)
                array = np.asarray(
                    dataset[
                        chunk_start:chunk_stop,
                        radial_start:radial_stop,
                        :ny,
                    ],
                    dtype=np.float64,
                )
                for band_count in band_counts:
                    for band_index, mask in enumerate(
                        masks_by_count[band_count]
                    ):
                        first = int(mask[0]) - radial_start
                        last = int(mask[-1]) - radial_start + 1
                        signals_by_count[band_count][
                            local,
                            band_index,
                            :,
                            field_index,
                        ] = np.mean(array[:, first:last, :], axis=1)

    return (
        time_us,
        frame,
        signals_by_count,
        edges_by_count,
        weights_by_count,
    )


def build_band_case(
    electric_field_kvm: int,
    source_h5: Path,
    time_us: np.ndarray,
    frame: np.ndarray,
    signals: np.ndarray,
    radial_edges_m: np.ndarray,
    radial_weights: np.ndarray,
    modes: np.ndarray,
    mode_roles: list[str],
) -> BandCaseData:
    coefficients = np.fft.rfft(signals, axis=2, norm="forward")
    selected = coefficients[:, :, modes, :]
    fit_mask = base.contiguous_mask(
        time_us, base.FIT_START_US, base.FIT_END_US
    )

    scales = np.sqrt(
        np.mean(np.abs(selected[fit_mask]) ** 2, axis=0)
    )
    scales = np.maximum(scales, np.finfo(np.float64).tiny)
    normalized = selected / scales[None, :, :, :]

    fit_indices = np.flatnonzero(fit_mask)
    carrier_angles = np.zeros(len(modes), dtype=np.float64)
    for mode_index in range(len(modes)):
        current = normalized[
            fit_indices[:-1], :, mode_index, :
        ]
        following = normalized[
            fit_indices[1:], :, mode_index, :
        ]
        cross = np.sum(following * np.conj(current))
        if abs(cross) > np.finfo(float).tiny:
            carrier_angles[mode_index] = float(np.angle(cross))

    phase = np.exp(
        -1j
        * frame[:, None, None, None]
        * carrier_angles[None, None, :, None]
    )
    envelope = normalized * phase
    collapsed = np.einsum(
        "b,tbmf->tmf",
        radial_weights,
        selected,
        optimize=True,
    )
    global_scales = np.sqrt(
        np.mean(np.abs(collapsed[fit_mask]) ** 2, axis=0)
    )
    global_scales = np.maximum(
        global_scales, np.finfo(np.float64).tiny
    )

    return BandCaseData(
        electric_field_kvm=electric_field_kvm,
        source_h5=source_h5,
        time_us=time_us,
        frame=frame,
        band_count=signals.shape[1],
        radial_edges_m=radial_edges_m,
        radial_weights=radial_weights,
        modes=modes,
        mode_roles=mode_roles,
        scales=scales,
        global_scales=global_scales,
        carrier_angles=carrier_angles,
        carrier_frequencies_mhz=(
            carrier_angles / (2.0 * np.pi * base.FRAME_DT_US)
        ),
        raw_complex=selected,
        normalized_complex=normalized,
        envelope_complex=envelope,
    )


def fit_rank_family(
    states: np.ndarray,
    ranks: tuple[int, ...],
) -> dict[int, base.LinearModel]:
    mean = np.mean(states, axis=0)
    centered = states - mean
    x = centered[:-1].T
    y = centered[1:].T
    u, singular_values, vh = np.linalg.svd(x, full_matrices=False)
    available = int(np.count_nonzero(singular_values > 1.0e-10))
    models = {}
    for requested_rank in ranks:
        effective_rank = min(requested_rank, available)
        if effective_rank < 1:
            continue
        inverse = 1.0 / singular_values[:effective_rank]
        matrix = (
            y
            @ (vh[:effective_rank].T * inverse[None, :])
            @ u[:, :effective_rank].T
        )
        models[requested_rank] = base.LinearModel(
            mean=mean,
            matrix=matrix,
            eigenvalues=np.linalg.eigvals(matrix),
            rank=effective_rank,
        )
    return models


def reconstruct_raw_state(
    case: BandCaseData,
    envelope_prediction: np.ndarray,
    frame_indices: np.ndarray,
) -> np.ndarray:
    return case.flatten(
        case.remodulate(envelope_prediction, frame_indices)
    )


def select_models(
    case: BandCaseData,
) -> tuple[dict[str, dict], list[dict]]:
    fit_mask = base.contiguous_mask(
        case.time_us, base.FIT_START_US, base.FIT_END_US
    )
    subtrain_mask = fit_mask & (
        case.time_us < base.VALIDATION_START_US
    )
    validation_mask = fit_mask & (
        case.time_us >= base.VALIDATION_START_US
    )
    raw_states = case.flatten(case.normalized_complex)
    envelope_states = case.flatten(case.envelope_complex)
    raw_subtrain = raw_states[subtrain_mask]
    envelope_subtrain = envelope_states[subtrain_mask]
    raw_validation = raw_states[validation_mask]
    validation_frames = case.frame[validation_mask]

    rank_candidates = tuple(
        rank
        for rank in RANK_CANDIDATES
        if rank <= min(case.state_dimensions, len(raw_subtrain) - 1)
    )
    if not rank_candidates:
        raise ValueError("No valid rank candidate")

    families = {
        "raw_dmd": fit_rank_family(raw_subtrain, rank_candidates),
        "envelope_dmd": fit_rank_family(
            envelope_subtrain, rank_candidates
        ),
    }
    selected = {}
    trials = []
    for method, models in families.items():
        initial = (
            raw_subtrain[-1]
            if method == "raw_dmd"
            else envelope_subtrain[-1]
        )
        best = {"objective": float("inf")}
        for requested_rank, model in models.items():
            prediction = base.rollout_linear(
                model, initial, len(raw_validation)
            )
            if method == "envelope_dmd":
                prediction = reconstruct_raw_state(
                    case, prediction, validation_frames
                )
            mse = base.prediction_mse(raw_validation, prediction)
            radius = float(np.max(np.abs(model.eigenvalues)))
            objective = mse + max(0.0, radius - 1.02) * 10.0
            row = {
                "electric_field_kvm": case.electric_field_kvm,
                "radial_bands": case.band_count,
                "state_dimensions": case.state_dimensions,
                "method": method,
                "requested_rank": requested_rank,
                "effective_rank": model.rank,
                "validation_mse": mse,
                "spectral_radius": radius,
                "objective": objective,
            }
            trials.append(row)
            if objective < best["objective"]:
                best = row
        selected[method] = best
    selected["selected_dmd_method"] = min(
        ("raw_dmd", "envelope_dmd"),
        key=lambda method: float(selected[method]["objective"]),
    )
    return selected, trials


def build_predictions(
    case: BandCaseData,
    selected: dict[str, dict],
) -> tuple[dict[str, np.ndarray], dict]:
    fit_mask = base.contiguous_mask(
        case.time_us, base.FIT_START_US, base.FIT_END_US
    )
    holdout_mask = (
        (case.time_us > base.FIT_END_US + 1.0e-10)
        & (case.time_us <= base.HOLDOUT_END_US + 1.0e-10)
    )
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

    raw_model = base.fit_linear_model(
        raw_fit, int(selected["raw_dmd"]["requested_rank"])
    )
    raw_prediction = base.rollout_linear(
        raw_model, raw_fit[-1], steps
    )
    envelope_model = base.fit_linear_model(
        envelope_fit,
        int(selected["envelope_dmd"]["requested_rank"]),
    )
    envelope_state = base.rollout_linear(
        envelope_model, envelope_fit[-1], steps
    )
    envelope_prediction = reconstruct_raw_state(
        case, envelope_state, holdout_frames
    )
    selected_method = selected["selected_dmd_method"]
    selected_prediction = {
        "raw_dmd": raw_prediction,
        "envelope_dmd": envelope_prediction,
    }[selected_method]

    predictions = {
        "truth": raw_states[holdout_mask],
        "persistence": persistence,
        "constant_carrier": carrier,
        "raw_dmd": raw_prediction,
        "envelope_dmd": envelope_prediction,
        "selected_dmd": selected_prediction,
        "holdout_frames": holdout_frames,
    }
    details = {
        "selected_dmd_method": selected_method,
        "raw_dmd_rank": raw_model.rank,
        "raw_dmd_spectral_radius": float(
            np.max(np.abs(raw_model.eigenvalues))
        ),
        "envelope_dmd_rank": envelope_model.rank,
        "envelope_dmd_spectral_radius": float(
            np.max(np.abs(envelope_model.eigenvalues))
        ),
    }
    return predictions, details


def modal_transport(physical: np.ndarray) -> np.ndarray:
    electron_index = base.PHYSICAL_FIELDS.index("electron_den")
    efy_index = base.PHYSICAL_FIELDS.index("efy")
    electron = physical[..., electron_index]
    efy = physical[..., efy_index]
    per_band = -2.0 * np.sum(
        np.real(electron * np.conj(efy)), axis=2
    ) / 0.020
    return per_band


def weighted_cross_phase_mae(
    truth_physical: np.ndarray,
    prediction_physical: np.ndarray,
) -> float:
    electron_index = base.PHYSICAL_FIELDS.index("electron_den")
    efy_index = base.PHYSICAL_FIELDS.index("efy")
    truth_cross = (
        truth_physical[..., electron_index]
        * np.conj(truth_physical[..., efy_index])
    )
    prediction_cross = (
        prediction_physical[..., electron_index]
        * np.conj(prediction_physical[..., efy_index])
    )
    error = np.abs(
        np.angle(prediction_cross * np.conj(truth_cross))
    )
    weights = np.abs(truth_cross)
    finite = np.isfinite(error) & np.isfinite(weights)
    denominator = float(np.sum(weights[finite]))
    if denominator <= np.finfo(float).tiny:
        return float("nan")
    return float(np.sum(error[finite] * weights[finite]) / denominator)


def complex_coherence(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> float:
    finite = np.isfinite(prediction.real) & np.isfinite(prediction.imag)
    if not np.any(finite):
        return float("nan")
    denominator = math.sqrt(
        float(np.sum(np.abs(truth[finite]) ** 2))
        * float(np.sum(np.abs(prediction[finite]) ** 2))
    )
    if denominator <= np.finfo(float).tiny:
        return float("nan")
    return float(
        abs(np.sum(prediction[finite] * np.conj(truth[finite])))
        / denominator
    )


def evaluate_prediction(
    case: BandCaseData,
    predictions: dict[str, np.ndarray],
    method: str,
) -> tuple[dict, dict]:
    truth_state = predictions["truth"]
    prediction_state = predictions[method]
    carrier_state = predictions["constant_carrier"]
    truth_complex = case.unflatten(truth_state)
    prediction_complex = case.unflatten(prediction_state)
    carrier_complex = case.unflatten(carrier_state)

    mse = base.prediction_mse(truth_state, prediction_state)
    carrier_mse = base.prediction_mse(truth_state, carrier_state)
    skill = (
        1.0 - mse / carrier_mse
        if np.isfinite(mse) and carrier_mse > 0.0
        else float("-inf")
    )

    truth_collapsed = case.collapse_normalized(truth_complex)
    prediction_collapsed = case.collapse_normalized(
        prediction_complex
    )
    carrier_collapsed = case.collapse_normalized(carrier_complex)
    truth_collapsed_state = case.collapse_state(truth_state)
    prediction_collapsed_state = case.collapse_state(prediction_state)
    carrier_collapsed_state = case.collapse_state(carrier_state)
    collapsed_mse = base.prediction_mse(
        truth_collapsed_state, prediction_collapsed_state
    )
    collapsed_carrier_mse = base.prediction_mse(
        truth_collapsed_state, carrier_collapsed_state
    )
    collapsed_skill = (
        1.0 - collapsed_mse / collapsed_carrier_mse
        if np.isfinite(collapsed_mse) and collapsed_carrier_mse > 0.0
        else float("-inf")
    )

    truth_physical = case.physical_coefficients(truth_complex)
    prediction_physical = case.physical_coefficients(
        prediction_complex
    )
    carrier_physical = case.physical_coefficients(carrier_complex)
    truth_transport_bands = modal_transport(truth_physical)
    prediction_transport_bands = modal_transport(prediction_physical)
    carrier_transport_bands = modal_transport(carrier_physical)
    truth_transport = np.einsum(
        "b,tb->t",
        case.radial_weights,
        truth_transport_bands,
    )
    prediction_transport = np.einsum(
        "b,tb->t",
        case.radial_weights,
        prediction_transport_bands,
    )
    carrier_transport = np.einsum(
        "b,tb->t",
        case.radial_weights,
        carrier_transport_bands,
    )
    transport_mse = base.prediction_mse(
        truth_transport[:, None], prediction_transport[:, None]
    )
    transport_carrier_mse = base.prediction_mse(
        truth_transport[:, None], carrier_transport[:, None]
    )
    transport_skill = (
        1.0 - transport_mse / transport_carrier_mse
        if np.isfinite(transport_mse) and transport_carrier_mse > 0.0
        else float("-inf")
    )

    phi_index = base.PHYSICAL_FIELDS.index("phi")
    primary_truth = truth_collapsed[:, 0, phi_index]
    primary_prediction = prediction_collapsed[:, 0, phi_index]
    holdout_frames = predictions["holdout_frames"]
    time_us = case.time_us[
        np.searchsorted(case.frame, holdout_frames)
    ]
    truth_frequency = base.estimate_frequency_mhz(
        primary_truth, time_us
    )
    prediction_frequency = base.estimate_frequency_mhz(
        primary_prediction, time_us
    )

    metrics = {
        "electric_field_kvm": case.electric_field_kvm,
        "radial_bands": case.band_count,
        "state_dimensions": case.state_dimensions,
        "method": method,
        "state_mse": mse,
        "state_skill_vs_constant_carrier": skill,
        "state_correlation": base.safe_correlation(
            truth_state, prediction_state
        ),
        "complex_coherence": complex_coherence(
            truth_complex, prediction_complex
        ),
        "collapsed_state_mse": collapsed_mse,
        "collapsed_state_skill_vs_constant_carrier": collapsed_skill,
        "collapsed_state_correlation": base.safe_correlation(
            truth_collapsed_state, prediction_collapsed_state
        ),
        "cross_phase_mae_rad": weighted_cross_phase_mae(
            truth_physical, prediction_physical
        ),
        "transport_correlation": base.safe_correlation(
            truth_transport, prediction_transport
        ),
        "transport_skill_vs_constant_carrier": transport_skill,
        "primary_truth_frequency_mhz": truth_frequency,
        "primary_prediction_frequency_mhz": prediction_frequency,
        "primary_frequency_absolute_error_mhz": abs(
            prediction_frequency - truth_frequency
        ),
        "finite_fraction": float(
            np.mean(np.isfinite(prediction_state).all(axis=1))
        ),
    }
    series = {
        "time_us": time_us,
        "primary_truth_amplitude": np.abs(primary_truth),
        "primary_prediction_amplitude": np.abs(primary_prediction),
        "primary_phase_error_rad": np.abs(
            np.angle(primary_prediction * np.conj(primary_truth))
        ),
        "transport_truth": truth_transport,
        "transport_prediction": prediction_transport,
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


def plot_metrics(metric_rows: list[dict], outdir: Path) -> None:
    selected = [
        row for row in metric_rows if row["method"] == "selected_dmd"
    ]
    specifications = (
        ("state_correlation", "Radial-state correlation"),
        (
            "state_skill_vs_constant_carrier",
            "Radial-state skill vs carrier",
        ),
        (
            "collapsed_state_correlation",
            "Collapsed-state correlation",
        ),
        ("cross_phase_mae_rad", "Cross-phase MAE [rad]"),
        ("transport_correlation", "Modal-transport correlation"),
        (
            "primary_frequency_absolute_error_mhz",
            "Primary frequency error [MHz]",
        ),
    )
    colors = {
        10: "#2455a4",
        20: "#2a9d5b",
        30: "#d18f00",
        40: "#c43c39",
    }
    fig, axes = plt.subplots(
        2, 3, figsize=(14.0, 8.4), constrained_layout=True
    )
    for ax, (metric, title) in zip(axes.ravel(), specifications):
        for electric_field_kvm in base.ELECTRIC_FIELDS:
            rows = sorted(
                (
                    row
                    for row in selected
                    if row["electric_field_kvm"] == electric_field_kvm
                ),
                key=lambda row: row["radial_bands"],
            )
            ax.plot(
                [row["radial_bands"] for row in rows],
                [row[metric] for row in rows],
                marker="o",
                linewidth=1.8,
                color=colors[electric_field_kvm],
                label=f"Ez={electric_field_kvm} kV/m",
            )
        ax.axhline(0.0, color="#999999", linewidth=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(RADIAL_BAND_COUNTS)
        ax.set_xticklabels([str(value) for value in RADIAL_BAND_COUNTS])
        ax.set_xlabel("Number of radial bands")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Radial-band Fourier-state ablation, autonomous 24-30 us"
    )
    fig.savefig(
        outdir / "radial_band_ablation_metrics.png", dpi=180
    )
    plt.close(fig)


def plot_rollouts(
    cases: dict[tuple[int, int], BandCaseData],
    predictions: dict[tuple[int, int], dict[str, np.ndarray]],
    series: dict[tuple[int, int], dict[str, dict]],
    outdir: Path,
) -> None:
    colors = {1: "#777777", 4: "#2455a4", 8: "#c43c39"}
    fig, axes = plt.subplots(
        len(base.ELECTRIC_FIELDS),
        2,
        figsize=(13.0, 12.0),
        constrained_layout=True,
    )
    for row_index, electric_field_kvm in enumerate(base.ELECTRIC_FIELDS):
        truth_series = series[(electric_field_kvm, 1)]["selected_dmd"]
        axes[row_index, 0].plot(
            truth_series["time_us"],
            truth_series["primary_truth_amplitude"],
            color="#111111",
            linewidth=2.2,
            label="PIC truth",
        )
        axes[row_index, 1].plot(
            truth_series["time_us"],
            truth_series["transport_truth"],
            color="#111111",
            linewidth=2.2,
            label="PIC truth",
        )
        for band_count in (1, 4, 8):
            current = series[
                (electric_field_kvm, band_count)
            ]["selected_dmd"]
            axes[row_index, 0].plot(
                current["time_us"],
                current["primary_prediction_amplitude"],
                color=colors[band_count],
                linewidth=1.4,
                label=f"{band_count} radial band(s)",
            )
            axes[row_index, 1].plot(
                current["time_us"],
                current["transport_prediction"],
                color=colors[band_count],
                linewidth=1.4,
                label=f"{band_count} radial band(s)",
            )
        axes[row_index, 0].set_ylabel(
            f"Ez={electric_field_kvm}\nprimary |c|"
        )
        axes[row_index, 1].set_ylabel(
            f"Ez={electric_field_kvm}\nmodal transport"
        )
        axes[row_index, 0].grid(alpha=0.25)
        axes[row_index, 1].grid(alpha=0.25)
    axes[0, 0].set_title("Collapsed primary-mode amplitude")
    axes[0, 1].set_title("Radially integrated selected-mode transport")
    axes[-1, 0].set_xlabel("Time [us]")
    axes[-1, 1].set_xlabel("Time [us]")
    axes[0, 0].legend(loc="lower right", fontsize=8)
    axes[0, 1].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Validation-selected DMD, strict autonomous holdout"
    )
    fig.savefig(
        outdir / "radial_band_autonomous_rollouts.png", dpi=180
    )
    plt.close(fig)


def save_reduced_h5(
    path: Path,
    cases: dict[tuple[int, int], BandCaseData],
    predictions: dict[tuple[int, int], dict[str, np.ndarray]],
) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["fit_interval_us"] = [
            base.FIT_START_US,
            base.FIT_END_US,
        ]
        handle.attrs["holdout_interval_us"] = [
            base.FIT_END_US,
            base.HOLDOUT_END_US,
        ]
        for (electric_field_kvm, band_count), case in cases.items():
            group = handle.create_group(
                f"E{electric_field_kvm}/bands_{band_count}"
            )
            group.create_dataset("modes", data=case.modes)
            group.create_dataset(
                "radial_edges_m", data=case.radial_edges_m
            )
            group.create_dataset(
                "radial_weights", data=case.radial_weights
            )
            group.create_dataset(
                "carrier_frequencies_mhz",
                data=case.carrier_frequencies_mhz,
            )
            current = predictions[(electric_field_kvm, band_count)]
            holdout_frames = current["holdout_frames"]
            indices = np.searchsorted(case.frame, holdout_frames)
            group.create_dataset(
                "time_us", data=case.time_us[indices]
            )
            for method in ("truth",) + METHODS:
                group.create_dataset(
                    method,
                    data=current[method],
                    compression="gzip",
                    compression_opts=1,
                )


def generate_readme(
    outdir: Path,
    metric_rows: list[dict],
    selected_models: dict[tuple[int, int], dict],
) -> None:
    selected_rows = [
        row for row in metric_rows if row["method"] == "selected_dmd"
    ]
    summary_lines = []
    for electric_field_kvm in base.ELECTRIC_FIELDS:
        rows = [
            row
            for row in selected_rows
            if row["electric_field_kvm"] == electric_field_kvm
        ]
        best_state = max(rows, key=lambda row: row["state_correlation"])
        best_transport = max(
            rows, key=lambda row: row["transport_correlation"]
        )
        summary_lines.append(
            "| "
            f"{electric_field_kvm} | "
            f"{best_state['radial_bands']} "
            f"({best_state['state_correlation']:.3f}) | "
            f"{best_transport['radial_bands']} "
            f"({best_transport['transport_correlation']:.3f}) |"
        )

    text = f"""# Radial-band physical Fourier-state ablation

## 日本語

物理Fourier低次元状態へradial局在情報を追加すると、24--30 usの
厳密な自律予測が改善するかを調べた解析です。Bx=20 mT、
Ez=10/20/30/40 kV/mの各ケースで、20--23 usをsubtrain、
23--24 usをvalidation、20--24 usを最終同定、24--30 usを未学習
holdoutとしました。予測途中でPIC真値は再入力していません。

radial解析範囲0.9--11.9 mmを1/2/4/8個の等幅帯域へ分けました。
方位角モードは既存のradial平均解析と同じ5モードを使い、変更したのは
radial解像度だけです。各状態次元は30/60/120/240です。

今回は状態設計のablationを分離するため、Hankelや追加のcross-phase
状態はまだ使用していません。raw DMDとcarrierを除いたenvelope DMDを
23--24 usだけで選択しています。

| Ez [kV/m] | best state bands (corr) | best transport bands (corr) |
|---:|---:|---:|
{chr(10).join(summary_lines)}

state correlationの改善だけでは完全閉包を意味しません。cross-phaseと
modal transportも同時に改善したかを確認してください。帯域数増加による
次元増加だけでvalidationへ過適合する可能性もあるため、1-bandを必ず
基準にします。

## English

This is a controlled ablation of radial localization in the physical Fourier
state. The same five azimuthal modes and the same temporal split are used for
1, 2, 4, and 8 equal-width radial bands. Only raw DMD and envelope DMD are
tested at this stage; delay coordinates and explicit cross-phase states are
reserved for subsequent ablations.

## Files

- `radial_band_ablation_metrics.csv`: every holdout metric
- `radial_band_model_selection.csv`: validation rank search
- `radial_band_selected_models.csv`: selected model per case/band count
- `radial_band_ablation_metrics.png`: metric trends versus radial bands
- `radial_band_autonomous_rollouts.png`: primary amplitude and transport
- `radial_band_reduced_rollouts.h5`: reduced truth and predictions
- `radial_band_ablation_summary.json`: settings and numerical summary
"""
    (outdir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether radial-band Fourier states improve strict "
            "autonomous reduced-order forecasts."
        )
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
    )
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    cases: dict[tuple[int, int], BandCaseData] = {}
    selected_models = {}
    predictions = {}
    model_details = {}
    metric_rows = []
    trial_rows = []
    selected_rows = []
    all_series = {}

    for electric_field_kvm in base.ELECTRIC_FIELDS:
        source_h5 = analysis_fields_path(electric_field_kvm)
        if not source_h5.is_file():
            raise FileNotFoundError(source_h5)
        global_case = base.estimate_representation(
            electric_field_kvm,
            base.diagnostic_path(electric_field_kvm),
        )
        (
            time_us,
            frame,
            signals_by_count,
            edges_by_count,
            weights_by_count,
        ) = extract_band_signals(source_h5, RADIAL_BAND_COUNTS)
        print(
            f"E{electric_field_kvm}: extracted "
            f"{len(time_us)} frames from {source_h5}",
            flush=True,
        )

        for band_count in RADIAL_BAND_COUNTS:
            case = build_band_case(
                electric_field_kvm=electric_field_kvm,
                source_h5=source_h5,
                time_us=time_us,
                frame=frame,
                signals=signals_by_count[band_count],
                radial_edges_m=edges_by_count[band_count],
                radial_weights=weights_by_count[band_count],
                modes=global_case.modes.copy(),
                mode_roles=list(global_case.mode_roles),
            )
            key = (electric_field_kvm, band_count)
            cases[key] = case
            selected, trials = select_models(case)
            selected_models[key] = selected
            trial_rows.extend(trials)
            case_predictions, details = build_predictions(case, selected)
            predictions[key] = case_predictions
            model_details[key] = details
            selected_rows.append(
                {
                    "electric_field_kvm": electric_field_kvm,
                    "radial_bands": band_count,
                    "state_dimensions": case.state_dimensions,
                    "selected_method": selected[
                        "selected_dmd_method"
                    ],
                    "raw_rank": selected["raw_dmd"][
                        "requested_rank"
                    ],
                    "raw_validation_mse": selected["raw_dmd"][
                        "validation_mse"
                    ],
                    "envelope_rank": selected["envelope_dmd"][
                        "requested_rank"
                    ],
                    "envelope_validation_mse": selected[
                        "envelope_dmd"
                    ]["validation_mse"],
                    "selected_modes": ",".join(
                        map(str, case.modes.tolist())
                    ),
                }
            )
            all_series[key] = {}
            for method in METHODS:
                metrics, series = evaluate_prediction(
                    case, case_predictions, method
                )
                metric_rows.append(metrics)
                all_series[key][method] = series
            print(
                f"E{electric_field_kvm} bands={band_count} "
                f"dims={case.state_dimensions} "
                f"selected={selected['selected_dmd_method']}",
                flush=True,
            )

    write_csv(outdir / "radial_band_ablation_metrics.csv", metric_rows)
    write_csv(outdir / "radial_band_model_selection.csv", trial_rows)
    write_csv(outdir / "radial_band_selected_models.csv", selected_rows)
    plot_metrics(metric_rows, outdir)
    plot_rollouts(cases, predictions, all_series, outdir)
    save_reduced_h5(
        outdir / "radial_band_reduced_rollouts.h5",
        cases,
        predictions,
    )

    summary = {
        "status": "PASS",
        "definition": {
            "electric_fields_kvm": base.ELECTRIC_FIELDS,
            "magnetic_field_t": 0.020,
            "physical_fields": base.PHYSICAL_FIELDS,
            "radial_interval_m": [RADIAL_MIN_M, RADIAL_MAX_M],
            "radial_band_counts": RADIAL_BAND_COUNTS,
            "mode_policy": (
                "same five modes selected by the existing radial-mean "
                "training-energy analysis"
            ),
            "fit_interval_us": [
                base.FIT_START_US,
                base.FIT_END_US,
            ],
            "subtrain_interval_us": [
                base.FIT_START_US,
                base.VALIDATION_START_US,
            ],
            "validation_interval_us": [
                base.VALIDATION_START_US,
                base.FIT_END_US,
            ],
            "holdout_interval_us": [
                base.FIT_END_US,
                base.HOLDOUT_END_US,
            ],
            "rank_candidates": RANK_CANDIDATES,
            "methods": METHODS,
        },
        "selected_models": {
            f"E{electric_field_kvm}_bands{band_count}": {
                "selection": selected_models[
                    (electric_field_kvm, band_count)
                ],
                "final": model_details[
                    (electric_field_kvm, band_count)
                ],
                "modes": cases[
                    (electric_field_kvm, band_count)
                ].modes,
                "carrier_frequencies_mhz": cases[
                    (electric_field_kvm, band_count)
                ].carrier_frequencies_mhz,
            }
            for electric_field_kvm in base.ELECTRIC_FIELDS
            for band_count in RADIAL_BAND_COUNTS
        },
        "metrics": metric_rows,
    }
    with (outdir / "radial_band_ablation_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            json_safe(summary),
            handle,
            indent=2,
            ensure_ascii=False,
        )
    generate_readme(outdir, metric_rows, selected_models)
    print(f"PASS: wrote radial-band ablation to {outdir}")


if __name__ == "__main__":
    main()
