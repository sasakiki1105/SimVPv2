"""Evaluate carrier-rate physical states for local E25 ROM closure.

The physical MTSI Fourier coefficients are compressed in the radial direction
with fit-only complex POD.  Two equivalent state descriptions are compared:

* Cartesian envelope: real/imaginary parts after removing a fitted carrier.
* Carrier-rate: log amplitude, residual phase, causal growth rate, and phase
  velocity (or cross-phase rate for the density--Ey cross spectrum).

All representation fitting, mode selection, scaling, and Hankel model
selection exclude the six-microsecond autonomous forecast interval.
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_radaz_augmented_physical_state_dynamics as augmented
import analyze_radaz_blockwise_fourier_latent_dynamics as block
import analyze_radaz_coupling_and_rolling_validation as rolling
import analyze_radaz_reduced_dynamics as reduced


ROOT = Path(__file__).resolve().parent
DEFAULT_PHYSICAL = augmented.DEFAULT_PHYSICAL
DEFAULT_FEATURES = augmented.CHECKPOINTS["data_only"]["features"]
DEFAULT_OUTPUT = (
    ROOT / "workdirs" / "compare_radaz_e25_carrier_state_closure"
)
FRAME_DT_US = 0.015
DOMAIN_LENGTH_Y_M = 1.28e-2
B_T = 0.020
MTSI_MODES = np.arange(1, 7, dtype=np.int64)
SELECTED_MODE_COUNT = 2
RADIAL_COMPONENTS = 2
RATE_FEATURES = (
    "log_amplitude",
    "residual_phase",
    "growth_rate",
    "phase_velocity_or_cross_phase_rate",
)
CIRCULAR_FEATURES = (
    "log_amplitude",
    "cos_residual_phase",
    "sin_residual_phase",
    "growth_rate",
    "phase_velocity_or_cross_phase_rate",
)

SYSTEMS = {
    "phi_cartesian_only": ("phi_cartesian",),
    "phi_carrier_only": ("phi_carrier",),
    "phi_circular_only": ("phi_circular",),
    "latent_phi_cartesian": ("latent", "phi_cartesian"),
    "latent_phi_carrier": ("latent", "phi_carrier"),
    "latent_phi_circular": ("latent", "phi_circular"),
    "phi_cross_carrier_only": ("phi_carrier", "cross_carrier"),
    "latent_phi_cross_carrier": (
        "latent",
        "phi_carrier",
        "cross_carrier",
    ),
    "phi_cross_circular_only": ("phi_circular", "cross_circular"),
    "latent_phi_cross_circular": (
        "latent",
        "phi_circular",
        "cross_circular",
    ),
}
SYSTEM_LABELS = {
    "phi_cartesian_only": "Pxy",
    "phi_carrier_only": "Pcar",
    "phi_circular_only": "Pcirc",
    "latent_phi_cartesian": "L+Pxy",
    "latent_phi_carrier": "L+Pcar",
    "latent_phi_circular": "L+Pcirc",
    "phi_cross_carrier_only": "Pcar+Xcar",
    "latent_phi_cross_carrier": "L+Pcar+Xcar",
    "phi_cross_circular_only": "Pcirc+Xcirc",
    "latent_phi_cross_circular": "L+Pcirc+Xcirc",
}
PHYSICAL_BASELINES = {
    "latent_phi_cartesian": "phi_cartesian_only",
    "latent_phi_carrier": "phi_carrier_only",
    "latent_phi_circular": "phi_circular_only",
    "latent_phi_cross_carrier": "phi_cross_carrier_only",
    "latent_phi_cross_circular": "phi_cross_circular_only",
}
WINDOWS = rolling.WINDOWS
METHODS = rolling.METHODS


@dataclass
class RawPhysical:
    time_us: np.ndarray
    frame: np.ndarray
    radial_weights: np.ndarray
    phi: np.ndarray
    cross: np.ndarray


@dataclass
class CarrierBlock:
    name: str
    modes: np.ndarray
    radial_weights: np.ndarray
    bases: np.ndarray
    scales: np.ndarray
    carrier_angles: np.ndarray
    phase_references: np.ndarray
    explained_variance: np.ndarray
    original: np.ndarray
    cartesian: np.ndarray
    rate: np.ndarray
    circular: np.ndarray
    frame: np.ndarray

    @property
    def carrier_count(self) -> int:
        return int(len(self.modes) * self.bases.shape[1])

    def decode_cartesian(
        self, values: np.ndarray, frames: np.ndarray
    ) -> np.ndarray:
        shaped = values.reshape(
            len(values), len(self.modes), self.bases.shape[1], 2
        )
        envelope = shaped[..., 0] + 1j * shaped[..., 1]
        normalized = envelope * np.exp(
            1j * frames[:, None, None] * self.carrier_angles[None]
        )
        return self._scores_to_coefficients(normalized * self.scales[None])

    def decode_rate(
        self, values: np.ndarray, frames: np.ndarray
    ) -> np.ndarray:
        shaped = values.reshape(
            len(values), len(self.modes), self.bases.shape[1], 4
        )
        log_amplitude = shaped[..., 0]
        residual_phase = shaped[..., 1] + self.phase_references[None]
        finite = np.all(np.isfinite(shaped), axis=(1, 2, 3))
        amplitude = np.exp(np.clip(log_amplitude, -30.0, 30.0))
        normalized = amplitude * np.exp(
            1j
            * (
                residual_phase
                + frames[:, None, None] * self.carrier_angles[None]
            )
        )
        normalized[~finite] = np.nan + 1j * np.nan
        return self._scores_to_coefficients(normalized * self.scales[None])

    def decode_circular(
        self, values: np.ndarray, frames: np.ndarray
    ) -> np.ndarray:
        shaped = values.reshape(
            len(values), len(self.modes), self.bases.shape[1], 5
        )
        log_amplitude = shaped[..., 0]
        residual_phase = (
            np.arctan2(shaped[..., 2], shaped[..., 1])
            + self.phase_references[None]
        )
        finite = np.all(np.isfinite(shaped), axis=(1, 2, 3))
        amplitude = np.exp(np.clip(log_amplitude, -30.0, 30.0))
        normalized = amplitude * np.exp(
            1j
            * (
                residual_phase
                + frames[:, None, None] * self.carrier_angles[None]
            )
        )
        normalized[~finite] = np.nan + 1j * np.nan
        return self._scores_to_coefficients(normalized * self.scales[None])

    def constant_carrier(
        self, fit_mask: np.ndarray, frames: np.ndarray
    ) -> np.ndarray:
        shaped = self.cartesian.reshape(
            len(self.cartesian), len(self.modes), self.bases.shape[1], 2
        )
        envelope = shaped[..., 0] + 1j * shaped[..., 1]
        last = envelope[np.flatnonzero(fit_mask)[-1]]
        normalized = last[None] * np.exp(
            1j * frames[:, None, None] * self.carrier_angles[None]
        )
        return self._scores_to_coefficients(normalized * self.scales[None])

    def _scores_to_coefficients(self, scores: np.ndarray) -> np.ndarray:
        result = np.empty(
            (len(scores), len(self.radial_weights), len(self.modes)),
            dtype=np.complex128,
        )
        sqrt_weights = np.sqrt(self.radial_weights)
        for mode_index in range(len(self.modes)):
            weighted = scores[:, mode_index] @ self.bases[mode_index]
            result[:, :, mode_index] = weighted / sqrt_weights[None]
        return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_median(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")


def finite_min(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.min(array)) if array.size else float("nan")


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_raw_physical(path: Path) -> RawPhysical:
    with h5py.File(path, "r") as handle:
        coefficients = np.asarray(handle["coefficients"], dtype=np.complex128)
        fields_raw = np.asarray(handle["fields"])
        time_us = np.asarray(handle["time_us"], dtype=np.float64)
        frame = np.asarray(handle["frame"], dtype=np.int64)
        radial_weights = np.asarray(handle["radial_weights"], dtype=np.float64)
    fields = [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in fields_raw
    ]
    phi = coefficients[:, fields.index("phi"), :, :]
    electron = coefficients[:, fields.index("electron_den"), :, :]
    efy = coefficients[:, fields.index("efy"), :, :]
    cross = electron * np.conj(efy)
    if not (
        np.all(np.isfinite(phi))
        and np.all(np.isfinite(cross))
        and np.all(radial_weights > 0.0)
    ):
        raise ValueError("Physical Fourier data contain invalid values")
    radial_weights = radial_weights / np.sum(radial_weights)
    return RawPhysical(time_us, frame, radial_weights, phi, cross)


def causal_difference(values: np.ndarray, dt_us: float) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    result[1:] = np.diff(values, axis=0) / dt_us
    result[0] = result[1]
    return result


def anchor_basis_phase(basis: np.ndarray) -> np.ndarray:
    anchored = basis.copy()
    for index in range(len(anchored)):
        pivot = int(np.argmax(np.abs(anchored[index])))
        factor = np.exp(-1j * np.angle(anchored[index, pivot]))
        anchored[index] *= factor
    return anchored


def select_modes(
    phi: np.ndarray,
    radial_weights: np.ndarray,
    fit_mask: np.ndarray,
) -> np.ndarray:
    energy = np.einsum(
        "r,trm->m", radial_weights, np.abs(phi[fit_mask]) ** 2
    )
    selected = np.argsort(energy[MTSI_MODES])[::-1][:SELECTED_MODE_COUNT]
    return MTSI_MODES[selected]


def build_carrier_block(
    name: str,
    values: np.ndarray,
    modes: np.ndarray,
    radial_weights: np.ndarray,
    frame: np.ndarray,
    fit_mask: np.ndarray,
) -> CarrierBlock:
    selected = values[:, :, modes]
    mode_count = len(modes)
    bases = np.empty(
        (mode_count, RADIAL_COMPONENTS, len(radial_weights)),
        dtype=np.complex128,
    )
    scales = np.empty((mode_count, RADIAL_COMPONENTS), dtype=np.float64)
    carrier_angles = np.empty_like(scales)
    references = np.empty_like(scales)
    explained = np.empty_like(scales)
    cartesian = np.empty(
        (len(values), mode_count, RADIAL_COMPONENTS, 2), dtype=np.float64
    )
    rate = np.empty(
        (len(values), mode_count, RADIAL_COMPONENTS, 4), dtype=np.float64
    )
    circular = np.empty(
        (len(values), mode_count, RADIAL_COMPONENTS, 5), dtype=np.float64
    )
    sqrt_weights = np.sqrt(radial_weights)
    fit_indices = np.flatnonzero(fit_mask)
    if len(fit_indices) < RADIAL_COMPONENTS + 2:
        raise ValueError("Insufficient fit samples for radial POD")

    for mode_index in range(mode_count):
        weighted = selected[:, :, mode_index] * sqrt_weights[None]
        _, singular_values, vh = np.linalg.svd(
            weighted[fit_mask], full_matrices=False
        )
        basis = anchor_basis_phase(vh[:RADIAL_COMPONENTS])
        bases[mode_index] = basis
        variance = singular_values**2
        explained[mode_index] = (
            variance[:RADIAL_COMPONENTS]
            / max(float(np.sum(variance)), np.finfo(float).tiny)
        )
        scores = weighted @ basis.conj().T
        scale = np.sqrt(np.mean(np.abs(scores[fit_mask]) ** 2, axis=0))
        scale = np.maximum(scale, np.finfo(float).tiny)
        scales[mode_index] = scale
        normalized = scores / scale[None]

        current = normalized[fit_indices[:-1]]
        following = normalized[fit_indices[1:]]
        cross_step = np.sum(following * np.conj(current), axis=0)
        angles = np.angle(cross_step)
        carrier_angles[mode_index] = angles
        envelope = normalized * np.exp(
            -1j * frame[:, None] * angles[None]
        )
        cartesian[:, mode_index, :, 0] = envelope.real
        cartesian[:, mode_index, :, 1] = envelope.imag

        amplitude = np.maximum(np.abs(normalized), 1.0e-10)
        log_amplitude = np.log(amplitude)
        residual_phase = np.unwrap(np.angle(envelope), axis=0)
        reference = residual_phase[fit_indices[0]].copy()
        references[mode_index] = reference
        residual_phase -= reference[None]
        growth = causal_difference(log_amplitude, FRAME_DT_US)
        residual_rate = causal_difference(residual_phase, FRAME_DT_US)
        if name == "phi":
            ky = 2.0 * np.pi * float(modes[mode_index]) / DOMAIN_LENGTH_Y_M
            total_rate = angles[None] / FRAME_DT_US + residual_rate
            fourth = -(total_rate * 1.0e6) / ky
        else:
            fourth = residual_rate
        rate[:, mode_index, :, 0] = log_amplitude
        rate[:, mode_index, :, 1] = residual_phase
        rate[:, mode_index, :, 2] = growth
        rate[:, mode_index, :, 3] = fourth
        circular[:, mode_index, :, 0] = log_amplitude
        circular[:, mode_index, :, 1] = np.cos(residual_phase)
        circular[:, mode_index, :, 2] = np.sin(residual_phase)
        circular[:, mode_index, :, 3] = growth
        circular[:, mode_index, :, 4] = fourth

    return CarrierBlock(
        name=name,
        modes=modes,
        radial_weights=radial_weights,
        bases=bases,
        scales=scales,
        carrier_angles=carrier_angles,
        phase_references=references,
        explained_variance=explained,
        original=selected,
        cartesian=cartesian.reshape(len(values), -1),
        rate=rate.reshape(len(values), -1),
        circular=circular.reshape(len(values), -1),
        frame=frame,
    )


def group_sources(
    latent: np.ndarray,
    phi_block: CarrierBlock,
    cross_block: CarrierBlock,
) -> dict[str, np.ndarray]:
    return {
        "latent": latent,
        "phi_cartesian": phi_block.cartesian,
        "phi_carrier": phi_block.rate,
        "phi_circular": phi_block.circular,
        "cross_carrier": cross_block.rate,
        "cross_circular": cross_block.circular,
    }


def groups_for_system(
    system: str, sources: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    return {name: sources[name] for name in SYSTEMS[system]}


def state_metrics(
    standardized: np.ndarray,
    prediction: np.ndarray,
    fit_mask: np.ndarray,
    forecast_mask: np.ndarray,
    time_us: np.ndarray,
) -> dict[str, float]:
    truth = standardized[forecast_mask]
    fit = standardized[fit_mask]
    persistence = np.repeat(fit[-1:], len(truth), axis=0)
    metrics, _ = reduced.evaluate_prediction(
        truth, prediction, persistence, time_us[forecast_mask]
    )
    return metrics


def mse_skill(
    truth: np.ndarray, prediction: np.ndarray, baseline: np.ndarray
) -> float:
    if not np.all(np.isfinite(prediction)):
        return float("-inf")
    mse = float(np.mean(np.abs(prediction - truth) ** 2))
    baseline_mse = float(np.mean(np.abs(baseline - truth) ** 2))
    return (
        1.0 - mse / baseline_mse
        if baseline_mse > np.finfo(float).tiny
        else float("nan")
    )


def coefficient_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    persistence: np.ndarray,
    carrier_baseline: np.ndarray,
    radial_weights: np.ndarray,
) -> dict[str, float]:
    truth_amplitude = np.abs(truth)
    prediction_amplitude = np.abs(prediction)
    real_scale = float(np.std(np.real(truth), ddof=1))
    return {
        "coefficient_correlation": augmented.correlation(truth, prediction),
        "coefficient_nrmse": augmented.nrmse(truth, prediction),
        "amplitude_correlation": augmented.correlation(
            truth_amplitude, prediction_amplitude
        ),
        "amplitude_ratio": float(
            np.mean(prediction_amplitude)
            / max(float(np.mean(truth_amplitude)), np.finfo(float).tiny)
        ),
        "normalized_real_bias": float(
            np.mean(np.real(prediction) - np.real(truth))
            / max(real_scale, np.finfo(float).tiny)
        ),
        "coefficient_skill_vs_persistence": mse_skill(
            truth, prediction, persistence
        ),
        "coefficient_skill_vs_constant_carrier": mse_skill(
            truth, prediction, carrier_baseline
        ),
        "weighted_phase_mae_rad": augmented.weighted_phase_mae(
            truth, prediction, radial_weights
        ),
    }


def phi_envelope(values: np.ndarray, radial_weights: np.ndarray) -> np.ndarray:
    return np.sqrt(
        np.einsum("r,trm->t", radial_weights, np.abs(values) ** 2)
    )


def transport_from_selected_cross(
    values: np.ndarray, radial_weights: np.ndarray
) -> np.ndarray:
    global_cross = np.einsum("r,trm->t", radial_weights, values)
    return -2.0 * np.real(global_cross) / B_T


def physical_metrics(
    system: str,
    method: str,
    prediction_groups: dict[str, np.ndarray],
    phi_block: CarrierBlock,
    cross_block: CarrierBlock,
    fit_mask: np.ndarray,
    forecast_mask: np.ndarray,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    rows: list[dict] = []
    traces: dict[str, np.ndarray] = {}
    forecast_frames = phi_block.frame[forecast_mask]
    fit_last = np.flatnonzero(fit_mask)[-1]
    phi_group = None
    phi_format = None
    if "phi_circular" in prediction_groups:
        phi_group = prediction_groups["phi_circular"]
        phi_format = "circular_carrier_rate"
        phi_prediction = phi_block.decode_circular(
            phi_group, forecast_frames
        )
    elif "phi_carrier" in prediction_groups:
        phi_group = prediction_groups["phi_carrier"]
        phi_format = "carrier_rate"
        phi_prediction = phi_block.decode_rate(phi_group, forecast_frames)
    elif "phi_cartesian" in prediction_groups:
        phi_group = prediction_groups["phi_cartesian"]
        phi_format = "cartesian_envelope"
        phi_prediction = phi_block.decode_cartesian(
            phi_group, forecast_frames
        )
    else:
        return rows, traces

    phi_truth = phi_block.original[forecast_mask]
    phi_persistence = np.repeat(
        phi_block.original[fit_last : fit_last + 1], len(phi_truth), axis=0
    )
    phi_carrier = phi_block.constant_carrier(fit_mask, forecast_frames)
    rows.append(
        {
            "system": system,
            "method": method,
            "quantity": "selected_phi_coefficients",
            "state_format": phi_format,
            **coefficient_metrics(
                phi_truth,
                phi_prediction,
                phi_persistence,
                phi_carrier,
                phi_block.radial_weights,
            ),
        }
    )
    truth_envelope = phi_envelope(phi_truth, phi_block.radial_weights)
    prediction_envelope = phi_envelope(
        phi_prediction, phi_block.radial_weights
    )
    persistence_envelope = np.repeat(
        phi_envelope(
            phi_block.original[fit_last : fit_last + 1],
            phi_block.radial_weights,
        ),
        len(phi_truth),
    )
    envelope_metrics = augmented.scalar_metrics(
        truth_envelope, prediction_envelope, persistence_envelope
    )
    rows.append(
        {
            "system": system,
            "method": method,
            "quantity": "selected_phi_envelope",
            "state_format": phi_format,
            **envelope_metrics,
        }
    )
    traces["phi_envelope_truth"] = truth_envelope
    traces["phi_envelope_prediction"] = prediction_envelope

    if (
        "cross_carrier" not in prediction_groups
        and "cross_circular" not in prediction_groups
    ):
        return rows, traces
    if "cross_circular" in prediction_groups:
        cross_prediction = cross_block.decode_circular(
            prediction_groups["cross_circular"], forecast_frames
        )
        cross_format = "circular_carrier_rate"
    else:
        cross_prediction = cross_block.decode_rate(
            prediction_groups["cross_carrier"], forecast_frames
        )
        cross_format = "carrier_rate"
    cross_truth = cross_block.original[forecast_mask]
    cross_persistence = np.repeat(
        cross_block.original[fit_last : fit_last + 1],
        len(cross_truth),
        axis=0,
    )
    cross_carrier = cross_block.constant_carrier(fit_mask, forecast_frames)
    rows.append(
        {
            "system": system,
            "method": method,
            "quantity": "selected_cross_spectrum",
            "state_format": cross_format,
            **coefficient_metrics(
                cross_truth,
                cross_prediction,
                cross_persistence,
                cross_carrier,
                cross_block.radial_weights,
            ),
        }
    )
    truth_transport = transport_from_selected_cross(
        cross_truth, cross_block.radial_weights
    )
    prediction_transport = transport_from_selected_cross(
        cross_prediction, cross_block.radial_weights
    )
    persistence_transport = np.repeat(
        transport_from_selected_cross(
            cross_block.original[fit_last : fit_last + 1],
            cross_block.radial_weights,
        ),
        len(cross_truth),
    )
    carrier_transport = transport_from_selected_cross(
        cross_carrier, cross_block.radial_weights
    )
    transport_metrics = augmented.scalar_metrics(
        truth_transport, prediction_transport, persistence_transport
    )
    transport_scale = float(np.std(truth_transport, ddof=1))
    rows.append(
        {
            "system": system,
            "method": method,
            "quantity": "selected_modal_transport",
            "state_format": cross_format,
            **transport_metrics,
            "skill_vs_constant_carrier": mse_skill(
                truth_transport, prediction_transport, carrier_transport
            ),
            "mean_ratio": (
                float(np.mean(prediction_transport) / np.mean(truth_transport))
                if abs(float(np.mean(truth_transport)))
                > np.finfo(float).tiny
                else float("nan")
            ),
            "normalized_bias": float(
                np.mean(prediction_transport - truth_transport)
                / max(transport_scale, np.finfo(float).tiny)
            ),
        }
    )
    traces["transport_truth"] = truth_transport
    traces["transport_prediction"] = prediction_transport
    return rows, traces


def analyze(
    raw: RawPhysical,
    features: np.ndarray,
    delays: list[int],
    ranks: list[int],
) -> dict[str, list[dict]]:
    output = {
        "selections": [],
        "candidates": [],
        "state": [],
        "physical": [],
        "representations": [],
        "traces": [],
    }
    budget = block.BUDGETS["medium_20"]
    for window, (fit_start, fit_end, forecast_end) in WINDOWS.items():
        validation_start = fit_end - 1.0
        subtrain_mask = (raw.time_us >= fit_start) & (
            raw.time_us < validation_start
        )
        fit_mask = (raw.time_us >= fit_start) & (raw.time_us < fit_end)
        forecast_mask = (raw.time_us >= fit_end) & (
            raw.time_us <= forecast_end
        )
        selected_sub = select_modes(raw.phi, raw.radial_weights, subtrain_mask)
        selected_final = select_modes(raw.phi, raw.radial_weights, fit_mask)
        sub_phi = build_carrier_block(
            "phi",
            raw.phi,
            selected_sub,
            raw.radial_weights,
            raw.frame,
            subtrain_mask,
        )
        sub_cross = build_carrier_block(
            "cross",
            raw.cross,
            selected_sub,
            raw.radial_weights,
            raw.frame,
            subtrain_mask,
        )
        phi_block = build_carrier_block(
            "phi",
            raw.phi,
            selected_final,
            raw.radial_weights,
            raw.frame,
            fit_mask,
        )
        cross_block = build_carrier_block(
            "cross",
            raw.cross,
            selected_final,
            raw.radial_weights,
            raw.frame,
            fit_mask,
        )
        _, latent_sub, _ = block.fit_block_models(
            features, subtrain_mask, budget
        )
        _, latent_final, pca_rows = block.fit_block_models(
            features, fit_mask, budget
        )
        sub_sources = group_sources(latent_sub, sub_phi, sub_cross)
        final_sources = group_sources(latent_final, phi_block, cross_block)

        for carrier_name, carrier in (("phi", phi_block), ("cross", cross_block)):
            for mode_index, mode in enumerate(carrier.modes):
                for component in range(RADIAL_COMPONENTS):
                    output["representations"].append(
                        {
                            "window": window,
                            "carrier": carrier_name,
                            "mode": int(mode),
                            "radial_component": component + 1,
                            "explained_variance": float(
                                carrier.explained_variance[
                                    mode_index, component
                                ]
                            ),
                            "carrier_frequency_mhz": float(
                                carrier.carrier_angles[
                                    mode_index, component
                                ]
                                / (2.0 * np.pi * FRAME_DT_US)
                            ),
                        }
                    )

        for system in SYSTEMS:
            sub_groups = groups_for_system(system, sub_sources)
            sub_scaler = augmented.GroupScaler.fit(
                sub_groups, subtrain_mask
            )
            sub_standardized = sub_scaler.transform(sub_groups)
            selected, candidates = rolling.dynamic_search_hankel(
                sub_standardized,
                raw.time_us,
                fit_start,
                fit_end,
                delays,
                ranks,
            )
            for row in candidates:
                output["candidates"].append(
                    {
                        "window": window,
                        "system": system,
                        "selected": (
                            row["delay"] == selected["delay"]
                            and row["rank"] == selected["rank"]
                        ),
                        **row,
                    }
                )
            groups = groups_for_system(system, final_sources)
            scaler = augmented.GroupScaler.fit(groups, fit_mask)
            standardized = scaler.transform(groups)
            predictions = augmented.fit_and_forecast(
                standardized,
                fit_mask,
                int(np.count_nonzero(forecast_mask)),
                int(selected["delay"]),
                int(selected["rank"]),
            )
            state_dimension = int(sum(value.shape[1] for value in groups.values()))
            output["selections"].append(
                {
                    "window": window,
                    "system": system,
                    "groups": "+".join(SYSTEMS[system]),
                    "state_dimension": state_dimension,
                    "latent_dimension": int(latent_final.shape[1]),
                    "latent_source_dimension": int(
                        sum(row["total_features"] for row in pca_rows)
                    ),
                    "selected_modes": ",".join(map(str, selected_final)),
                    **selected,
                }
            )
            for method in METHODS:
                state_row = {
                    "window": window,
                    "system": system,
                    "method": method,
                    **state_metrics(
                        standardized,
                        predictions[method],
                        fit_mask,
                        forecast_mask,
                        raw.time_us,
                    ),
                }
                output["state"].append(state_row)
                prediction_groups = scaler.inverse(predictions[method])
                metric_rows, traces = physical_metrics(
                    system,
                    method,
                    prediction_groups,
                    phi_block,
                    cross_block,
                    fit_mask,
                    forecast_mask,
                )
                for row in metric_rows:
                    row["window"] = window
                    output["physical"].append(row)
                if window == "fit20_24_forecast24_30":
                    forecast_times = raw.time_us[forecast_mask]
                    for index, time_us in enumerate(forecast_times):
                        trace_row = {
                            "window": window,
                            "system": system,
                            "method": method,
                            "time_us": float(time_us),
                        }
                        for name, values in traces.items():
                            trace_row[name] = float(values[index])
                        output["traces"].append(trace_row)
        print(
            f"PASS {window}: selected MTSI modes "
            f"{selected_final.tolist()}",
            flush=True,
        )
    return output


def summarize(results: dict[str, list[dict]]) -> list[dict]:
    rows: list[dict] = []
    physical_lookup = {}
    for row in results["physical"]:
        physical_lookup.setdefault(
            (row["system"], row["method"], row["quantity"]), []
        ).append(row)
    for system in SYSTEMS:
        for method in METHODS:
            state = [
                row
                for row in results["state"]
                if row["system"] == system and row["method"] == method
            ]
            selection = [
                row for row in results["selections"] if row["system"] == system
            ]
            summary = {
                "system": system,
                "label": SYSTEM_LABELS[system],
                "method": method,
                "groups": "+".join(SYSTEMS[system]),
                "state_dimension": int(selection[0]["state_dimension"]),
                "median_delay": finite_median(row["delay"] for row in selection),
                "median_rank": finite_median(row["rank"] for row in selection),
                "median_state_skill": finite_median(
                    row["skill_vs_persistence"] for row in state
                ),
                "min_state_skill": finite_min(
                    row["skill_vs_persistence"] for row in state
                ),
                "positive_state_skill_windows": sum(
                    row["skill_vs_persistence"] > 0.0 for row in state
                ),
                "median_state_correlation": finite_median(
                    row["flattened_correlation"] for row in state
                ),
            }
            for quantity in (
                "selected_phi_coefficients",
                "selected_phi_envelope",
                "selected_cross_spectrum",
                "selected_modal_transport",
            ):
                values = physical_lookup.get((system, method, quantity), [])
                if not values:
                    continue
                prefix = quantity
                for metric in (
                    "correlation",
                    "temporal_anomaly_correlation",
                    "skill_vs_persistence",
                    "coefficient_correlation",
                    "coefficient_skill_vs_persistence",
                    "coefficient_skill_vs_constant_carrier",
                    "weighted_phase_mae_rad",
                    "skill_vs_constant_carrier",
                    "amplitude_correlation",
                    "amplitude_ratio",
                    "normalized_real_bias",
                    "mean_ratio",
                    "normalized_bias",
                ):
                    available = [row[metric] for row in values if metric in row]
                    if available:
                        summary[f"{prefix}_median_{metric}"] = finite_median(
                            available
                        )
                skill_key = (
                    "coefficient_skill_vs_persistence"
                    if quantity.endswith("coefficients")
                    or quantity == "selected_cross_spectrum"
                    else "skill_vs_persistence"
                )
                summary[f"{prefix}_positive_skill_windows"] = sum(
                    row.get(skill_key, float("-inf")) > 0.0 for row in values
                )
            rows.append(summary)

    lookup = {(row["system"], row["method"]): row for row in rows}
    for row in rows:
        baseline_name = PHYSICAL_BASELINES.get(row["system"])
        if baseline_name is not None:
            baseline = lookup[(baseline_name, row["method"])]
            row["physical_only_baseline"] = baseline_name
            for metric in (
                "median_state_skill",
                "selected_phi_coefficients_median_coefficient_correlation",
                "selected_phi_coefficients_median_coefficient_skill_vs_constant_carrier",
                "selected_phi_envelope_median_temporal_anomaly_correlation",
                "selected_modal_transport_median_temporal_anomaly_correlation",
            ):
                if metric in row and metric in baseline:
                    row[f"{metric}_gain_vs_physical_only"] = (
                        row[metric] - baseline[metric]
                    )
    return rows


def plot_state_skill(path: Path, state_rows: list[dict]) -> None:
    labels = list(WINDOWS)
    x = np.arange(len(labels))
    figure, axes = plt.subplots(
        1, 2, figsize=(14.0, 5.4), sharey=True, constrained_layout=True
    )
    colors = plt.get_cmap("tab10")
    for axis, method in zip(axes, METHODS):
        for index, system in enumerate(SYSTEMS):
            selected = {
                row["window"]: row
                for row in state_rows
                if row["system"] == system and row["method"] == method
            }
            values = [selected[label]["skill_vs_persistence"] for label in labels]
            shown = np.maximum(values, -1.0)
            axis.plot(
                x,
                shown,
                marker="o",
                linewidth=1.7,
                color=colors(index),
                label=SYSTEM_LABELS[system],
            )
            for position, (raw, display) in enumerate(zip(values, shown)):
                if raw < -1.0:
                    axis.annotate(
                        f"{raw:.1f}",
                        (position, display),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center",
                        fontsize=7,
                        color=colors(index),
                    )
        axis.axhline(0.0, color="#777777", linewidth=1.0)
        axis.set_ylim(-1.05, 1.0)
        axis.set_xticks(x)
        axis.set_xticklabels(["12-16 to 22", "16-20 to 26", "20-24 to 30"])
        axis.set_xlabel("fit interval to forecast end [us]")
        axis.set_title(method.replace("_", " "))
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right", fontsize=8, framealpha=0.9)
    axes[0].set_ylabel("joint-state skill vs persistence")
    figure.suptitle("E25 carrier-state closure robustness")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_physical_summary(path: Path, summary_rows: list[dict]) -> None:
    keys = [(row["system"], row["method"]) for row in summary_rows]
    labels = [f"{SYSTEM_LABELS[s]}\n{m.replace('_', ' ')}" for s, m in keys]
    phi_corr = [
        row.get(
            "selected_phi_envelope_median_temporal_anomaly_correlation",
            np.nan,
        )
        for row in summary_rows
    ]
    phi_phase = [
        row.get(
            "selected_phi_coefficients_median_weighted_phase_mae_rad",
            np.nan,
        )
        for row in summary_rows
    ]
    transport_corr = [
        row.get(
            "selected_modal_transport_median_temporal_anomaly_correlation",
            np.nan,
        )
        for row in summary_rows
    ]
    state_skill = [row["median_state_skill"] for row in summary_rows]
    figure, axes = plt.subplots(2, 2, figsize=(20.0, 10.0), constrained_layout=True)
    series = (
        (state_skill, "Median joint-state skill", (-1.0, 1.0)),
        (phi_corr, "MTSI phi-envelope correlation", (-1.0, 1.0)),
        (phi_phase, "MTSI coefficient phase MAE [rad]", (0.0, np.pi)),
        (transport_corr, "Selected-mode transport correlation", (-1.0, 1.0)),
    )
    x = np.arange(len(labels))
    for axis, (values, title, limits) in zip(axes.ravel(), series):
        finite_values = np.asarray(values, dtype=np.float64)
        colors = ["#4c78a8" if np.isfinite(value) else "#d9d9d9" for value in finite_values]
        axis.bar(x, np.nan_to_num(finite_values, nan=0.0), color=colors)
        axis.set_ylim(*limits)
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=42, ha="right", fontsize=7)
        axis.grid(axis="y", alpha=0.25)
        axis.axhline(0.0, color="#777777", linewidth=0.8)
    figure.suptitle("E25 carrier-state physical observable summary")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_last_rollout(path: Path, traces: list[dict], summary: list[dict]) -> None:
    candidates = [
        row
        for row in summary
        if row["system"] == "latent_phi_cross_circular"
    ]
    best = max(candidates, key=lambda row: row["median_state_skill"])
    selected = [
        row
        for row in traces
        if row["system"] == best["system"] and row["method"] == best["method"]
    ]
    selected.sort(key=lambda row: row["time_us"])
    time = np.asarray([row["time_us"] for row in selected])
    figure, axes = plt.subplots(2, 1, figsize=(11.5, 7.0), constrained_layout=True)
    for axis, truth_key, prediction_key, ylabel in (
        (
            axes[0],
            "phi_envelope_truth",
            "phi_envelope_prediction",
            "selected MTSI phi envelope",
        ),
        (
            axes[1],
            "transport_truth",
            "transport_prediction",
            "selected modal transport",
        ),
    ):
        axis.plot(time, [row[truth_key] for row in selected], color="#000000", label="PIC truth")
        axis.plot(time, [row[prediction_key] for row in selected], color="#d55e00", label="autonomous ROM")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right")
    axes[1].set_xlabel("time [us]")
    figure.suptitle(
        f"E25 20-24 to 24-30 us: {SYSTEM_LABELS[best['system']]} / "
        f"{best['method'].replace('_', ' ')}"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_readme(path: Path, summary: list[dict], representations: list[dict]) -> None:
    ranked = sorted(
        summary, key=lambda row: row["median_state_skill"], reverse=True
    )
    lines = [
        "# E25 physical carrier-state closure",
        "",
        "## Purpose",
        "",
        "This experiment tests whether a phase-aware physical MTSI state is easier to evolve autonomously than the same complex POD coefficients represented only by real and imaginary parts. SimVP is frozen; no neural-network retraining is performed.",
        "",
        "## Leakage control",
        "",
        "For each rolling window, MTSI mode selection, radial POD, carrier frequency, normalization, delay, and rank are fitted without using the six-microsecond forecast truth. Delay/rank use only the final one microsecond inside the fit interval for validation.",
        "",
        "## State definitions",
        "",
        "- `Pxy`: two energetic MTSI modes times two radial POD components, stored as carrier-removed real/imaginary coefficients (8 values).",
        "- `Pcar`: the same four carriers stored as log amplitude, unwrapped residual phase, causal log-amplitude growth rate, and physical azimuthal phase velocity (16 values).",
        "- `Pcirc`: the same carriers stored as log amplitude, cosine/sine residual phase, causal growth rate, and phase velocity (20 values). The cosine/sine pair preserves the circular topology of phase.",
        "- `Xcar`: density-Ey cross-spectrum for the same modes/POD budget, stored as log amplitude, cross-phase, growth rate, and cross-phase rate (16 values).",
        "- `Xcirc`: the circular-phase equivalent of Xcar (20 values).",
        "- `L`: the frozen data-only SimVP blockwise latent state (20 values).",
        "",
        "The derivative variables are backward differences. Therefore the state at the forecast boundary does not use the first forecast truth frame.",
        "",
        "## Rolling summary",
        "",
        "| method | state | dim | state skill median/min | positive windows | phi envelope corr | phi phase MAE [rad] | transport corr |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked:
        lines.append(
            f"| {row['method']} | {row['label']} | {row['state_dimension']} | "
            f"{row['median_state_skill']:.3f}/{row['min_state_skill']:.3f} | "
            f"{row['positive_state_skill_windows']}/3 | "
            f"{row.get('selected_phi_envelope_median_temporal_anomaly_correlation', float('nan')):.3f} | "
            f"{row.get('selected_phi_coefficients_median_weighted_phase_mae_rad', float('nan')):.3f} | "
            f"{row.get('selected_modal_transport_median_temporal_anomaly_correlation', float('nan')):.3f} |"
        )

    lookup = {(row["system"], row["method"]): row for row in summary}
    carrier_findings = []
    for method in METHODS:
        cart = lookup[("phi_cartesian_only", method)]
        carrier = lookup[("phi_carrier_only", method)]
        circular = lookup[("phi_circular_only", method)]
        latent_cart = lookup[("latent_phi_cartesian", method)]
        latent_carrier = lookup[("latent_phi_carrier", method)]
        latent_circular = lookup[("latent_phi_circular", method)]
        cross = lookup[("phi_cross_carrier_only", method)]
        latent_cross = lookup[("latent_phi_cross_carrier", method)]
        circular_cross = lookup[("phi_cross_circular_only", method)]
        latent_circular_cross = lookup[
            ("latent_phi_cross_circular", method)
        ]
        carrier_findings.extend(
            [
                (
                    f"- `{method}` physical-only: carrier-rate changes the "
                    "phi-envelope correlation from "
                    f"{cart['selected_phi_envelope_median_temporal_anomaly_correlation']:.3f} "
                    "to "
                    f"{carrier['selected_phi_envelope_median_temporal_anomaly_correlation']:.3f}, "
                    "but phase MAE changes from "
                    f"{cart['selected_phi_coefficients_median_weighted_phase_mae_rad']:.3f} "
                    "to "
                    f"{carrier['selected_phi_coefficients_median_weighted_phase_mae_rad']:.3f} rad."
                ),
                (
                    f"- `{method}` latent-coupled: carrier-rate changes median "
                    "joint-state skill from "
                    f"{latent_cart['median_state_skill']:.3f} to "
                    f"{latent_carrier['median_state_skill']:.3f}, while phi "
                    "phase MAE changes from "
                    f"{latent_cart['selected_phi_coefficients_median_weighted_phase_mae_rad']:.3f} "
                    "to "
                    f"{latent_carrier['selected_phi_coefficients_median_weighted_phase_mae_rad']:.3f} rad."
                ),
                (
                    f"- `{method}` circular phase: physical-only phi "
                    "envelope correlation/phase MAE are "
                    f"{circular['selected_phi_envelope_median_temporal_anomaly_correlation']:.3f}/"
                    f"{circular['selected_phi_coefficients_median_weighted_phase_mae_rad']:.3f} rad; "
                    "with latent coupling they are "
                    f"{latent_circular['selected_phi_envelope_median_temporal_anomaly_correlation']:.3f}/"
                    f"{latent_circular['selected_phi_coefficients_median_weighted_phase_mae_rad']:.3f} rad."
                ),
                (
                    f"- `{method}` cross carrier: physical-only transport "
                    "correlation/skill are "
                    f"{cross['selected_modal_transport_median_temporal_anomaly_correlation']:.3f}/"
                    f"{cross['selected_modal_transport_median_skill_vs_persistence']:.3f}; "
                    "with latent coupling they are "
                    f"{latent_cross['selected_modal_transport_median_temporal_anomaly_correlation']:.3f}/"
                    f"{latent_cross['selected_modal_transport_median_skill_vs_persistence']:.3f}."
                ),
                (
                    f"- `{method}` circular cross carrier: physical-only "
                    "transport correlation/skill are "
                    f"{circular_cross['selected_modal_transport_median_temporal_anomaly_correlation']:.3f}/"
                    f"{circular_cross['selected_modal_transport_median_skill_vs_persistence']:.3f}; "
                    "with latent coupling they are "
                    f"{latent_circular_cross['selected_modal_transport_median_temporal_anomaly_correlation']:.3f}/"
                    f"{latent_circular_cross['selected_modal_transport_median_skill_vs_persistence']:.3f}. "
                    "The corresponding normalized transport biases are "
                    f"{circular_cross['selected_modal_transport_median_normalized_bias']:.3f} "
                    "and "
                    f"{latent_circular_cross['selected_modal_transport_median_normalized_bias']:.3f}."
                ),
            ]
        )

    selected_modes = sorted(
        {
            (
                row["window"],
                row["carrier"],
                int(row["mode"]),
            )
            for row in representations
        }
    )
    mode_text = ", ".join(
        f"{window}:{carrier}:n{mode}"
        for window, carrier, mode in selected_modes
        if carrier == "phi"
    )
    lines.extend(
        [
            "",
            "## Interpretation rules",
            "",
            "1. Compare `Pcar` and `Pcirc` with `Pxy`, and their latent-coupled counterparts. This isolates phase representation from the choice of Fourier/POD information.",
            "2. Compare each latent-coupled state with its physical-only counterpart. A positive gain means the SimVP latent history contributes predictive information beyond the physical carrier history.",
            "3. `L+Pcirc+Xcirc` is the strongest physical-closure test here because it includes circular mode phase and the density-Ey relation needed for modal transport.",
            "4. Success is local to E25. It does not establish transfer across electric field or a universal plasma closure.",
            "",
            "## Main result",
            "",
            *carrier_findings,
            "",
            "The numerical rows above distinguish amplitude-envelope closure from full complex phase and transport closure. Positive joint-state skill alone is not treated as proof: phase error and transport skill must also improve.",
            "",
            f"Fit-only selected modes: {mode_text}.",
            "",
            "## Files",
            "",
            "- `summary.csv`: three-window aggregate by state and method.",
            "- `state_metrics_by_window.csv`: autonomous joint-state metrics.",
            "- `physical_metrics_by_window.csv`: coefficient, envelope, phase, cross-spectrum, and transport metrics.",
            "- `representation_by_window.csv`: selected modes, POD variance, and carrier frequencies.",
            "- `model_selections.csv` and `validation_candidates.csv`: fit-only hyperparameter selection.",
            "- `last_window_time_series.csv`: reusable 24-30 us traces.",
            "- `state_skill_by_window.png`, `physical_summary.png`, and `best_carrier_rollout.png`: visual summaries.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical", type=Path, default=DEFAULT_PHYSICAL)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--delays", default="40,80,120")
    parser.add_argument("--ranks", default="8,12,20,30,40")
    args = parser.parse_args()
    delays = [int(value) for value in args.delays.split(",")]
    ranks = [int(value) for value in args.ranks.split(",")]
    args.output.mkdir(parents=True, exist_ok=True)

    raw = load_raw_physical(args.physical)
    features, time_us, frames = block.load_features(args.features)
    if not np.allclose(raw.time_us, time_us, atol=1.0e-9):
        raise ValueError("Physical and latent time axes do not match")
    if not np.array_equal(raw.frame, frames):
        raise ValueError("Physical and latent frame axes do not match")

    results = analyze(raw, features, delays, ranks)
    summary = summarize(results)
    write_csv(args.output / "summary.csv", summary)
    write_csv(args.output / "state_metrics_by_window.csv", results["state"])
    write_csv(args.output / "physical_metrics_by_window.csv", results["physical"])
    write_csv(args.output / "representation_by_window.csv", results["representations"])
    write_csv(args.output / "model_selections.csv", results["selections"])
    write_csv(args.output / "validation_candidates.csv", results["candidates"])
    write_csv(args.output / "last_window_time_series.csv", results["traces"])
    plot_state_skill(args.output / "state_skill_by_window.png", results["state"])
    plot_physical_summary(args.output / "physical_summary.png", summary)
    plot_last_rollout(args.output / "best_carrier_rollout.png", results["traces"], summary)
    write_readme(args.output / "README.md", summary, results["representations"])

    payload = {
        "status": "PASS",
        "physical_source": str(args.physical.resolve()),
        "latent_source": str(args.features.resolve()),
        "windows": WINDOWS,
        "delays": delays,
        "ranks": ranks,
        "selected_mode_count": SELECTED_MODE_COUNT,
        "radial_components": RADIAL_COMPONENTS,
        "rate_features": RATE_FEATURES,
        "circular_features": CIRCULAR_FEATURES,
        "forecast_truth_used_as_input": False,
        "summary": summary,
    }
    with (args.output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, ensure_ascii=False)
    print(f"PASS: wrote E25 carrier-state closure analysis to {args.output}")


if __name__ == "__main__":
    main()
