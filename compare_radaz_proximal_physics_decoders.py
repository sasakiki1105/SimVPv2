#!/usr/bin/env python3
"""Compare proximal physics decoders, including hard spectral protection.

Three cases are evaluated with the frozen E25-trained G8-SimVP:

* E25 stationary held-out 29--30 us (development reference), and
* E25 -> E22.5 transition 30.360--34.950 us (early relaxation), and
* E25 -> E22.5 transition 35.355--39.945 us (late relaxation).

Both operational decoders use a 1% truth-free displacement cap.  The phi-only
decoder is unchanged from evaluate_radaz_g8_proximal_physics_decoder.py.  The
joint decoder minimizes a per-frame, prediction-normalized correction norm and
distributes Poisson correction between phi and charge difference ni-ne while
preserving ne+ni.  Four predeclared hard spectral variants additionally remove
selected azimuthal Fourier components from the phi correction, so the corresponding
raw SimVP modes are exactly carried through the decoder.  PIC truth is used only
after every rule has been applied.
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
import torch
from scipy.fft import dst, fft, idst, ifft

from analyze_radaz_g2_stability_reconstruction import (
    E_CHARGE,
    EPS0,
    TINY,
    field_metrics,
    json_safe,
    modal_analysis,
    modal_transport_analysis,
    periodic_physics_metrics,
)
from evaluate_radaz_g8_proximal_physics_decoder import (
    compact_phi_modal_metrics,
    electric_field_displacement,
    electric_field_truth_metrics,
    physical_fields,
    poisson_residual,
    scalar_stats,
    solve_periodic_poisson_correction,
)
from openstl.models.simvp_model import SimVP_Model
from train_radaz_g2_residual_superresolution import (
    DEFAULT_H5,
    make_grid_interpolated,
    segment_starts,
)
import train_radaz_electric_history_hidden_band_envelope_rom as envelope_rom


ROOT = Path(__file__).resolve().parent
DEFAULT_G8 = ROOT / "workdirs" / "radaz_e25_g8_simvp_residual_sr_sync10_20to30us"
DEFAULT_TRANSITION_H5 = (
    ROOT
    / "workdirs"
    / "radaz_e25_to_e22p5_primary"
    / "radaz_3ch_e25targetnorm_native257x256_pad260x256.h5"
)
DEFAULT_OUTPUT = ROOT / "workdirs" / "radaz_proximal_decoder_joint_comparison"
CHANNELS = ("electron_den", "ion_den", "phi")
AZIMUTHAL_BANDS = {
    "n0": (0, 0),
    "mtsi_n1_6": (1, 6),
    "transition_n7_8": (7, 8),
    "ecdi_n9_21": (9, 21),
    "high_n22_nyquist": (22, None),
}
SPECTRAL_PROTECTIONS = {
    "positive_protect_mtsi_n1_6": (1, 6),
    "positive_protect_transition_n7_8": (7, 8),
    "positive_protect_ecdi_n9_21": (9, 21),
    "positive_protect_all_n1_21": (1, 21),
}
SOFT_GATE_DECODER = "positive_softgate_ecdi_n9_21"
SOFT_GATE_Z_START = 3.0
SOFT_GATE_Z_FULL = 5.0


@dataclass(frozen=True)
class Case:
    label: str
    h5: Path
    start_us: float
    end_us: float
    all_frames: bool
    radial_size: int
    evaluation_start_us: float | None = None
    evaluation_end_us: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g8-workdir", type=Path, default=DEFAULT_G8)
    parser.add_argument("--stationary-h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--transition-h5", type=Path, default=DEFAULT_TRANSITION_H5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budget", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([json_safe(row) for row in rows])


def predict_unique_frames(
    fine: np.ndarray,
    baseline: np.ndarray,
    starts: np.ndarray,
    checkpoint: dict,
    device: torch.device,
    radial_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    model = SimVP_Model(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    length = int(checkpoint["model_kwargs"]["in_shape"][0])
    scale = torch.as_tensor(
        checkpoint["residual_rms"], dtype=torch.float32, device=device
    ).view(1, 1, 3, 1, 1)
    first = int(starts[0])
    final = int(starts[-1]) + length
    sums = np.zeros((final - first, 3, radial_size, 256), dtype=np.float64)
    square_sums = np.zeros_like(sums)
    counts = np.zeros(final - first, dtype=np.int64)
    with torch.inference_mode():
        for start in starts:
            inputs = torch.from_numpy(
                baseline[int(start) : int(start) + length]
            ).unsqueeze(0).to(device)
            with torch.amp.autocast(
                device_type="cuda", enabled=device.type == "cuda"
            ):
                outputs = inputs + model(inputs) * scale
            values = outputs[0, :, :, :radial_size].float().cpu().numpy()
            local = int(start) - first
            sums[local : local + length] += values
            square_sums[local : local + length] += values**2
            counts[local : local + length] += 1
    covered = counts > 0
    indices = np.arange(first, final, dtype=np.int64)[covered]
    prediction = (
        sums[covered] / counts[covered, None, None, None]
    ).astype(np.float32)
    overlap_variance = np.maximum(
        square_sums[covered] / counts[covered, None, None, None]
        - prediction.astype(np.float64) ** 2,
        0.0,
    )
    consistency = {
        name: {
            "overlap_prediction_std": float(
                np.sqrt(np.mean(overlap_variance[:, channel]))
            ),
            "overlap_prediction_std_over_truth_std": float(
                np.sqrt(np.mean(overlap_variance[:, channel]))
                / max(float(np.std(fine[indices, channel, :radial_size])), TINY)
            ),
            "minimum_predictions_per_frame": int(np.min(counts[covered])),
            "maximum_predictions_per_frame": int(np.max(counts[covered])),
        }
        for channel, name in enumerate(CHANNELS)
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (
        prediction,
        fine[indices, :, :radial_size].copy(),
        baseline[indices, :, :radial_size].copy(),
        indices,
        consistency,
    )


def spatial_fluctuation_rms(values: np.ndarray) -> np.ndarray:
    centered = values - np.mean(values, axis=(1, 2), keepdims=True)
    return np.sqrt(np.mean(centered**2, axis=(1, 2)))


def solve_weighted_joint_correction(
    residual: np.ndarray,
    phi_scale: np.ndarray,
    ne_scale: np.ndarray,
    ni_scale: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimum normalized-L2 exact correction in phi and q=ni-ne.

    The radial potential correction has zero Dirichlet boundaries.  Charge
    correction is applied only where the discrete Poisson residual is defined.
    ne+ni is preserved by applying -delta_q/2 to ne and +delta_q/2 to ni.
    """
    frame_count, radial_interior, azimuth_count = residual.shape
    radial_modes = np.arange(1, radial_interior + 1, dtype=np.float64)
    lambda_x = (
        -4.0
        * np.sin(math.pi * radial_modes / (2.0 * (radial_interior + 1))) ** 2
        / dx**2
    )
    azimuth_modes = np.arange(azimuth_count, dtype=np.float64)
    lambda_y = (
        -4.0 * np.sin(math.pi * azimuth_modes / azimuth_count) ** 2 / dy**2
    )
    operator_eigenvalues = lambda_x[:, None] + lambda_y[None, :]
    charge_scale = 2.0 / np.sqrt(
        1.0 / np.maximum(ne_scale, TINY) ** 2
        + 1.0 / np.maximum(ni_scale, TINY) ** 2
    )
    coupling = E_CHARGE / EPS0
    delta_phi = np.zeros(
        (frame_count, radial_interior + 2, azimuth_count), dtype=np.float64
    )
    delta_q = np.zeros_like(delta_phi)
    for frame in range(frame_count):
        residual_hat = fft(
            dst(residual[frame], type=1, axis=0, norm="ortho"),
            axis=1,
            norm="ortho",
        )
        denominator = (
            phi_scale[frame] ** 2 * operator_eigenvalues**2
            + charge_scale[frame] ** 2 * coupling**2
        )
        multiplier = residual_hat / np.maximum(denominator, TINY)
        delta_phi_hat = (
            -phi_scale[frame] ** 2 * operator_eigenvalues * multiplier
        )
        delta_q_hat = -charge_scale[frame] ** 2 * coupling * multiplier
        delta_phi[frame, 1:-1] = idst(
            ifft(delta_phi_hat, axis=1, norm="ortho").real,
            type=1,
            axis=0,
            norm="ortho",
        )
        delta_q[frame, 1:-1] = idst(
            ifft(delta_q_hat, axis=1, norm="ortho").real,
            type=1,
            axis=0,
            norm="ortho",
        )
    return delta_phi, delta_q


def normalized_candidate(
    prediction: np.ndarray,
    physical: np.ndarray,
    norm_low: np.ndarray,
    norm_high: np.ndarray,
    delta_phi: np.ndarray | None = None,
    delta_q: np.ndarray | None = None,
    weight: np.ndarray | None = None,
) -> np.ndarray:
    if weight is None:
        weight = np.ones(len(prediction), dtype=np.float64)
    corrected = physical.copy()
    if delta_phi is not None:
        corrected[:, 2] += weight[:, None, None] * delta_phi
    if delta_q is not None:
        applied_q = weight[:, None, None] * delta_q
        corrected[:, 0] -= 0.5 * applied_q
        corrected[:, 1] += 0.5 * applied_q
    return (
        (corrected - norm_low[None, :, None, None])
        / (norm_high - norm_low)[None, :, None, None]
    ).astype(np.float32)


def correction_ratios(
    delta_phi: np.ndarray,
    delta_q: np.ndarray,
    weight: np.ndarray,
    phi_scale: np.ndarray,
    ne_scale: np.ndarray,
    ni_scale: np.ndarray,
) -> dict[str, np.ndarray]:
    applied_phi = weight[:, None, None] * delta_phi
    applied_q = weight[:, None, None] * delta_q
    return {
        "phi": np.sqrt(np.mean(applied_phi**2, axis=(1, 2)))
        / np.maximum(phi_scale, TINY),
        "electron_den": np.sqrt(np.mean((0.5 * applied_q) ** 2, axis=(1, 2)))
        / np.maximum(ne_scale, TINY),
        "ion_den": np.sqrt(np.mean((0.5 * applied_q) ** 2, axis=(1, 2)))
        / np.maximum(ni_scale, TINY),
    }


def maximum_step_inside_rms_ball(
    current: np.ndarray,
    direction: np.ndarray,
    radius: np.ndarray,
    active_frames: np.ndarray,
) -> np.ndarray:
    """Largest alpha in [0, 1] with RMS(current + alpha*direction) <= radius."""
    alpha = np.zeros(len(current), dtype=np.float64)
    for frame in np.flatnonzero(active_frames):
        current_frame = current[frame]
        direction_frame = direction[frame]
        quadratic = float(np.mean(direction_frame**2))
        if quadratic <= TINY:
            continue
        linear = 2.0 * float(np.mean(current_frame * direction_frame))
        constant = float(np.mean(current_frame**2) - radius[frame] ** 2)
        # The incoming joint correction is already inside the ball.  Clamp a
        # tiny positive round-off violation so alpha=0 remains feasible.
        constant = min(constant, 0.0)
        discriminant = max(linear * linear - 4.0 * quadratic * constant, 0.0)
        upper_root = (-linear + math.sqrt(discriminant)) / (2.0 * quadratic)
        alpha[frame] = min(max(upper_root, 0.0), 1.0)
    return alpha


def positivity_aware_joint_correction(
    ne_raw: np.ndarray,
    ni_raw: np.ndarray,
    phi_raw: np.ndarray,
    joint_delta_phi: np.ndarray,
    joint_delta_q: np.ndarray,
    joint_weight: np.ndarray,
    phi_scale: np.ndarray,
    ne_scale: np.ndarray,
    ni_scale: np.ndarray,
    budget: float,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Enforce density positivity and reproject residual within the phi budget.

    The active-set density projection preserves ne+ni wherever that is
    feasible.  Only frames whose density correction hits the positivity bound
    receive a second Poisson correction, and that correction is restricted to
    the unused part/direction of the original phi RMS ball.
    """
    applied_phi = joint_weight[:, None, None] * joint_delta_phi
    applied_q = joint_weight[:, None, None] * joint_delta_q
    # A tiny positive floor survives the normalized float32 round-trip while
    # remaining negligible compared with the predicted density fluctuation.
    density_floor = 1.0e-6 * np.minimum(ne_scale, ni_scale)
    lower = 2.0 * (
        density_floor[:, None, None] - ni_raw
    )
    upper = 2.0 * (
        ne_raw - density_floor[:, None, None]
    )
    infeasible = lower > upper
    if np.any(infeasible):
        raise RuntimeError(
            "Cannot preserve ne+ni while enforcing positivity in "
            f"{int(np.count_nonzero(infeasible))} cells"
        )
    projected_q = np.minimum(np.maximum(applied_q, lower), upper)
    adjusted = projected_q != applied_q
    active_frames = np.any(adjusted, axis=(1, 2))

    ne_projected = ne_raw - 0.5 * projected_q
    ni_projected = ni_raw + 0.5 * projected_q
    phi_projected = phi_raw + applied_phi
    projected_residual, _ = poisson_residual(
        ne_projected, ni_projected, phi_projected, dx, dy
    )
    phi_repair_direction = solve_periodic_poisson_correction(
        projected_residual, dx, dy
    )
    phi_repair_direction[~active_frames] = 0.0
    phi_repair_alpha = maximum_step_inside_rms_ball(
        applied_phi,
        phi_repair_direction,
        budget * phi_scale,
        active_frames,
    )
    projected_phi = (
        applied_phi
        + phi_repair_alpha[:, None, None] * phi_repair_direction
    )
    metadata: dict[str, object] = {
        "positivity_active_frame_fraction": float(np.mean(active_frames)),
        "positivity_adjusted_cell_fraction": float(np.mean(adjusted)),
        "positivity_floor_over_density_fluctuation": 1.0e-6,
        "phi_residual_repair_alpha": scalar_stats(phi_repair_alpha[active_frames])
        if np.any(active_frames)
        else scalar_stats(np.zeros(1, dtype=np.float64)),
        "density_sum_preservation_max_abs_m3": float(
            np.max(np.abs((ne_projected + ni_projected) - (ne_raw + ni_raw)))
        ),
    }
    return projected_phi, projected_q, metadata


def protect_phi_correction_modes(
    delta_phi: np.ndarray, first_mode: int, last_mode: int
) -> np.ndarray:
    """Remove a fixed azimuthal-mode band from the phi correction.

    Filtering the correction, rather than the corrected field, exactly preserves
    the raw SimVP carrier in the protected modes and cannot increase its RMS norm.
    """
    modes = np.fft.rfft(delta_phi, axis=-1)
    upper = min(last_mode + 1, modes.shape[-1])
    modes[..., first_mode:upper] = 0.0
    return np.fft.irfft(modes, n=delta_phi.shape[-1], axis=-1)


def soft_protect_phi_correction_modes(
    delta_phi: np.ndarray,
    first_mode: int,
    last_mode: int,
    protection_strength: np.ndarray,
) -> np.ndarray:
    """Attenuate a phi-correction band by a per-frame truth-free gate."""
    if protection_strength.shape != (len(delta_phi),):
        raise ValueError("protection_strength must contain one value per frame")
    modes = np.fft.rfft(delta_phi, axis=-1)
    upper = min(last_mode + 1, modes.shape[-1])
    modes[..., first_mode:upper] *= (
        1.0 - protection_strength[:, None, None]
    )
    return np.fft.irfft(modes, n=delta_phi.shape[-1], axis=-1)


def fit_log_poisson_ood_reference(
    relative_residual: np.ndarray,
) -> dict[str, float]:
    """Fit a robust log-residual reference without looking at PIC truth."""
    log_values = np.log(np.maximum(relative_residual, TINY))
    center = float(np.median(log_values))
    mad = float(np.median(np.abs(log_values - center)))
    robust_scale = max(1.4826 * mad, 1.0e-6)
    return {
        "log_median": center,
        "log_mad_sigma": robust_scale,
        "z_start": SOFT_GATE_Z_START,
        "z_full": SOFT_GATE_Z_FULL,
        "calibration_frame_count": int(len(relative_residual)),
    }


def poisson_ood_soft_gate(
    relative_residual: np.ndarray, reference: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothstep gate: off through 3 robust sigma and fully on at 5."""
    z_score = (
        np.log(np.maximum(relative_residual, TINY)) - reference["log_median"]
    ) / max(reference["log_mad_sigma"], TINY)
    interval = max(reference["z_full"] - reference["z_start"], TINY)
    normalized = np.clip((z_score - reference["z_start"]) / interval, 0.0, 1.0)
    strength = normalized**2 * (3.0 - 2.0 * normalized)
    return strength, z_score


def one_sided_mode_weights(azimuth_count: int) -> np.ndarray:
    """Parseval weights for a real-valued rFFT."""
    count = azimuth_count // 2 + 1
    weights = np.full(count, 2.0, dtype=np.float64)
    weights[0] = 1.0
    if azimuth_count % 2 == 0:
        weights[-1] = 1.0
    return weights


def band_mode_slice(
    band: tuple[int, int | None], available_modes: int
) -> slice:
    first, last = band
    stop = available_modes if last is None else min(last + 1, available_modes)
    return slice(min(first, available_modes), stop)


def bandwise_poisson_residual(
    physical: np.ndarray, dx: float, dy: float
) -> dict[str, np.ndarray]:
    """Per-frame relative Poisson residual split by azimuthal mode band."""
    residual, source = poisson_residual(
        physical[:, 0], physical[:, 1], physical[:, 2], dx, dy
    )
    residual_modes = np.fft.rfft(residual, axis=-1, norm="forward")
    source_modes = np.fft.rfft(source, axis=-1, norm="forward")
    weights = one_sided_mode_weights(residual.shape[-1])
    metrics: dict[str, np.ndarray] = {}
    for name, band in AZIMUTHAL_BANDS.items():
        selected = band_mode_slice(band, residual_modes.shape[-1])
        band_weights = weights[selected][None, None, :]
        residual_energy = np.sum(
            band_weights * np.abs(residual_modes[..., selected]) ** 2,
            axis=(1, 2),
        )
        source_energy = np.sum(
            band_weights * np.abs(source_modes[..., selected]) ** 2,
            axis=(1, 2),
        )
        metrics[name] = np.sqrt(
            residual_energy / np.maximum(source_energy, TINY)
        )
    return metrics


def phi_band_change_relative_to_raw(
    candidate_phi: np.ndarray, raw_phi: np.ndarray
) -> dict[str, float]:
    """Global Fourier-band displacement of phi relative to raw SimVP."""
    candidate_modes = np.fft.rfft(candidate_phi, axis=-1, norm="forward")
    raw_modes = np.fft.rfft(raw_phi, axis=-1, norm="forward")
    weights = one_sided_mode_weights(raw_phi.shape[-1])
    changes: dict[str, float] = {}
    for name, band in AZIMUTHAL_BANDS.items():
        selected = band_mode_slice(band, raw_modes.shape[-1])
        band_weights = weights[selected][None, None, :]
        numerator = float(
            np.sum(
                band_weights
                * np.abs(candidate_modes[..., selected] - raw_modes[..., selected])
                ** 2
            )
        )
        denominator = float(
            np.sum(band_weights * np.abs(raw_modes[..., selected]) ** 2)
        )
        changes[name] = math.sqrt(numerator / max(denominator, TINY))
    return changes


def hidden_band_coefficients(physical: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reordered = physical[:, (2, 0, 1)]
    groups = np.array_split(np.arange(physical.shape[2], dtype=np.int64), 8)
    coefficients = np.empty((len(physical), 3, 8, 5), dtype=np.complex128)
    for field in range(3):
        for radial, group in enumerate(groups):
            radial_mean = np.mean(reordered[:, field, group, :], axis=1)
            coefficients[:, field, radial] = np.fft.rfft(
                radial_mean, axis=-1, norm="forward"
            )[:, 17:22]
    weights = np.asarray([len(group) for group in groups], dtype=np.float64)
    weights /= np.sum(weights)
    return coefficients, weights


def evaluate_candidate(
    label: str,
    candidate: np.ndarray,
    truth: np.ndarray,
    raw: np.ndarray,
    time_s: np.ndarray,
    norm_low: np.ndarray,
    norm_high: np.ndarray,
    dx: float,
    dy: float,
    correction: dict[str, np.ndarray],
    applied_weight: np.ndarray,
) -> tuple[dict, dict]:
    truth_diag = periodic_physics_metrics(truth, norm_low, norm_high, dx, dy)
    raw_diag = periodic_physics_metrics(raw, norm_low, norm_high, dx, dy)
    candidate_diag = periodic_physics_metrics(
        candidate, norm_low, norm_high, dx, dy
    )
    field = {
        name: field_metrics(truth[:, channel], candidate[:, channel])
        for channel, name in enumerate(CHANNELS)
    }
    e_metrics = electric_field_truth_metrics(candidate_diag, truth_diag)
    _, stability, frequency, _ = modal_analysis(truth, raw, candidate, time_s)
    transport, _ = modal_transport_analysis(
        truth, raw, candidate, norm_low, norm_high, dy
    )
    truth_phys = physical_fields(truth, norm_low, norm_high)
    raw_phys = physical_fields(raw, norm_low, norm_high)
    candidate_phys = physical_fields(candidate, norm_low, norm_high)
    truth_band_poisson = bandwise_poisson_residual(truth_phys, dx, dy)
    raw_band_poisson = bandwise_poisson_residual(raw_phys, dx, dy)
    candidate_band_poisson = bandwise_poisson_residual(candidate_phys, dx, dy)
    phi_band_change = phi_band_change_relative_to_raw(
        candidate_phys[:, 2], raw_phys[:, 2]
    )
    truth_hidden, radial_weights = hidden_band_coefficients(truth_phys)
    candidate_hidden, _ = hidden_band_coefficients(candidate_phys)
    hidden = envelope_rom.envelope_metrics(
        truth_hidden,
        np.abs(candidate_hidden),
        candidate_hidden,
        radial_weights,
        time_s * 1.0e6,
    )
    modal = compact_phi_modal_metrics(truth, candidate)
    raw_poisson = float(np.median(raw_diag["relative_poisson_residual"]))
    truth_poisson = float(np.median(truth_diag["relative_poisson_residual"]))
    row = {
        "decoder": label,
        "weight_median": float(np.median(applied_weight)),
        "phi_correction_ratio_median": float(np.median(correction["phi"])),
        "ne_correction_ratio_median": float(
            np.median(correction["electron_den"])
        ),
        "ni_correction_ratio_median": float(np.median(correction["ion_den"])),
        "poisson_residual_median": float(
            np.median(candidate_diag["relative_poisson_residual"])
        ),
        "poisson_residual_ratio_to_raw": float(
            np.median(candidate_diag["relative_poisson_residual"])
            / max(raw_poisson, TINY)
        ),
        "poisson_residual_ratio_to_truth": float(
            np.median(candidate_diag["relative_poisson_residual"])
            / max(truth_poisson, TINY)
        ),
        "poisson_balance_corr_median": float(
            np.median(candidate_diag["poisson_balance_corr"])
        ),
        "quasineutral_imbalance_median": float(
            np.median(candidate_diag["quasineutral_imbalance"])
        ),
        "ne_relative_l2": field["electron_den"]["relative_l2"],
        "ni_relative_l2": field["ion_den"]["relative_l2"],
        "phi_relative_l2": field["phi"]["relative_l2"],
        "electric_field_relative_l2": e_metrics["relative_l2"],
        "electric_field_energy_ratio": e_metrics["energy_ratio"],
        "electric_field_displacement_to_raw": electric_field_displacement(
            candidate_diag, raw_diag
        ),
        "phi_dominant_mode": modal["candidate_dominant_mode"],
        "phi_dominant_mode_time_agreement": modal[
            "dominant_mode_time_agreement"
        ],
        "phi_mtsi_power_ratio": modal["mtsi_n1_6_mean_power_ratio"],
        "phi_ecdi_power_ratio": modal["ecdi_n9_21_mean_power_ratio"],
        "phi_ecdi_over_mtsi_ratio_to_truth": modal[
            "ecdi_over_mtsi_ratio_to_truth"
        ],
        "transport_mtsi_relative_l2": transport["mtsi_candidate_n1_6"][
            "model"
        ]["relative_l2"],
        "transport_ecdi_relative_l2": transport["ecdi_candidate_n9_21"][
            "model"
        ]["relative_l2"],
        "transport_ecdi_time_correlation": transport["ecdi_candidate_n9_21"][
            "model"
        ]["time_correlation"],
        "hidden_n17_21_amplitude_relative_l2": hidden["amplitude_relative_l2"],
        "hidden_n17_21_mean_power_ratio": hidden["mean_power_ratio"],
        "hidden_n17_21_power_series_relative_l2": hidden[
            "power_series_relative_l2"
        ],
        "hidden_n17_21_power_time_correlation": hidden[
            "power_time_correlation"
        ],
        "hidden_n17_21_complex_relative_l2": hidden["complex_relative_l2"],
        "hidden_n17_21_complex_coherence": hidden["complex_coherence"],
        "negative_ne_fraction": float(np.mean(candidate_phys[:, 0] < 0.0)),
        "negative_ni_fraction": float(np.mean(candidate_phys[:, 1] < 0.0)),
    }
    for band in AZIMUTHAL_BANDS:
        candidate_median = float(np.median(candidate_band_poisson[band]))
        raw_median = float(np.median(raw_band_poisson[band]))
        truth_median = float(np.median(truth_band_poisson[band]))
        row[f"poisson_{band}_residual_median"] = candidate_median
        row[f"poisson_{band}_residual_ratio_to_raw"] = (
            candidate_median / max(raw_median, TINY)
        )
        row[f"poisson_{band}_residual_ratio_to_truth"] = (
            candidate_median / max(truth_median, TINY)
        )
        row[f"phi_{band}_change_relative_to_raw"] = phi_band_change[band]
    detail = {
        "truth_free": {
            "applied_weight": scalar_stats(applied_weight),
            "correction_ratio_to_prediction_fluctuation": {
                name: scalar_stats(values) for name, values in correction.items()
            },
            "relative_poisson_residual": scalar_stats(
                candidate_diag["relative_poisson_residual"]
            ),
            "poisson_residual_median_ratio_to_raw": row[
                "poisson_residual_ratio_to_raw"
            ],
            "poisson_balance_correlation": scalar_stats(
                candidate_diag["poisson_balance_corr"]
            ),
            "bandwise_relative_poisson_residual": {
                band: scalar_stats(values)
                for band, values in candidate_band_poisson.items()
            },
            "phi_band_change_relative_to_raw": phi_band_change,
        },
        "truth_evaluation_only": {
            "field_metrics": field,
            "electric_field": e_metrics,
            "phi_modal": modal,
            "hidden_n17_21": hidden,
            "modal_transport_proxy_ne_ey": transport,
            "stability_and_band_metrics": stability,
            "phase_frequency": frequency,
        },
    }
    return row, detail


def evaluate_case(
    case: Case,
    checkpoint: dict,
    device: torch.device,
    budget: float,
    ood_reference: dict[str, float] | None = None,
    candidate_labels: tuple[str, ...] | None = None,
) -> tuple[list[dict], dict, dict[str, float]]:
    with h5py.File(case.h5, "r") as handle:
        all_time_s = np.asarray(handle["time_s"], dtype=np.float64)
        selected = np.flatnonzero(
            (all_time_s * 1.0e6 >= case.start_us - 1.0e-9)
            & (all_time_s * 1.0e6 <= case.end_us + 1.0e-9)
        )
        fine = np.asarray(
            handle["data_tchw"][int(selected[0]) : int(selected[-1]) + 1],
            dtype=np.float32,
        )
        times_s = all_time_s[selected]
        norm_low = np.asarray(handle["norm_low"], dtype=np.float64)
        norm_high = np.asarray(handle["norm_high"], dtype=np.float64)
        x_m = np.asarray(handle["x_m"], dtype=np.float64)[: case.radial_size]
        y_m = np.asarray(handle["y_m"], dtype=np.float64)
    baseline_all = make_grid_interpolated(fine, 8)
    evaluation_begin = 0 if case.all_frames else int(math.floor(0.9 * len(fine)))
    length = int(checkpoint["model_kwargs"]["in_shape"][0])
    hop = int(checkpoint["metadata"]["window_hop"])
    starts = segment_starts(evaluation_begin, len(fine), length, hop)
    prediction, truth, _, indices, consistency = predict_unique_frames(
        fine, baseline_all, starts, checkpoint, device, case.radial_size
    )
    eval_times_s = times_s[indices]
    keep = np.ones(len(eval_times_s), dtype=bool)
    if case.evaluation_start_us is not None:
        keep &= eval_times_s * 1.0e6 >= case.evaluation_start_us - 1.0e-9
    if case.evaluation_end_us is not None:
        keep &= eval_times_s * 1.0e6 <= case.evaluation_end_us + 1.0e-9
    prediction = prediction[keep]
    truth = truth[keep]
    eval_times_s = eval_times_s[keep]
    dx = float(np.median(np.diff(x_m)))
    dy = float(np.median(np.diff(y_m)))
    raw_phys = physical_fields(prediction, norm_low, norm_high)
    ne_pred, ni_pred, phi_pred = raw_phys[:, 0], raw_phys[:, 1], raw_phys[:, 2]
    residual, source = poisson_residual(ne_pred, ni_pred, phi_pred, dx, dy)
    raw_relative_poisson = np.sqrt(np.mean(residual**2, axis=(1, 2))) / np.maximum(
        np.sqrt(np.mean(source**2, axis=(1, 2))), TINY
    )
    reference_fitted_here = ood_reference is None
    resolved_ood_reference = (
        fit_log_poisson_ood_reference(raw_relative_poisson)
        if reference_fitted_here
        else dict(ood_reference)
    )
    soft_gate_strength, soft_gate_z = poisson_ood_soft_gate(
        raw_relative_poisson, resolved_ood_reference
    )
    scales = {
        "electron_den": spatial_fluctuation_rms(ne_pred),
        "ion_den": spatial_fluctuation_rms(ni_pred),
        "phi": spatial_fluctuation_rms(phi_pred),
    }

    phi_delta = solve_periodic_poisson_correction(residual, dx, dy)
    zero_q = np.zeros_like(phi_delta)
    phi_full_ratios = correction_ratios(
        phi_delta,
        zero_q,
        np.ones(len(prediction)),
        scales["phi"],
        scales["electron_den"],
        scales["ion_den"],
    )
    phi_weight = np.minimum(
        1.0, budget / np.maximum(phi_full_ratios["phi"], TINY)
    )

    joint_phi_delta, joint_q_delta = solve_weighted_joint_correction(
        residual,
        scales["phi"],
        scales["electron_den"],
        scales["ion_den"],
        dx,
        dy,
    )
    joint_full_ratios = correction_ratios(
        joint_phi_delta,
        joint_q_delta,
        np.ones(len(prediction)),
        scales["phi"],
        scales["electron_den"],
        scales["ion_den"],
    )
    joint_max_ratio = np.maximum.reduce(list(joint_full_ratios.values()))
    joint_weight = np.minimum(1.0, budget / np.maximum(joint_max_ratio, TINY))

    positivity_phi_delta, positivity_q_delta, positivity_metadata = (
        positivity_aware_joint_correction(
            ne_pred,
            ni_pred,
            phi_pred,
            joint_phi_delta,
            joint_q_delta,
            joint_weight,
            scales["phi"],
            scales["electron_den"],
            scales["ion_den"],
            budget,
            dx,
            dy,
        )
    )
    unit_weight = np.ones(len(prediction), dtype=np.float64)
    positivity_ratios = correction_ratios(
        positivity_phi_delta,
        positivity_q_delta,
        unit_weight,
        scales["phi"],
        scales["electron_den"],
        scales["ion_den"],
    )
    positivity_metadata["channel_budget_satisfied"] = {
        name: bool(float(np.max(values)) <= budget * (1.0 + 1.0e-9))
        for name, values in positivity_ratios.items()
    }
    positivity_metadata["channel_correction_ratio_max"] = {
        name: float(np.max(values)) for name, values in positivity_ratios.items()
    }

    candidates = {
        "raw_simvp": (
            prediction,
            {name: np.zeros(len(prediction)) for name in scales},
            np.zeros(len(prediction)),
            {},
        ),
        "phi_only_1pct": (
            normalized_candidate(
                prediction,
                raw_phys,
                norm_low,
                norm_high,
                delta_phi=phi_delta,
                weight=phi_weight,
            ),
            correction_ratios(
                phi_delta,
                zero_q,
                phi_weight,
                scales["phi"],
                scales["electron_den"],
                scales["ion_den"],
            ),
            phi_weight,
            {},
        ),
        "weighted_joint_1pct": (
            normalized_candidate(
                prediction,
                raw_phys,
                norm_low,
                norm_high,
                delta_phi=joint_phi_delta,
                delta_q=joint_q_delta,
                weight=joint_weight,
            ),
            correction_ratios(
                joint_phi_delta,
                joint_q_delta,
                joint_weight,
                scales["phi"],
                scales["electron_den"],
                scales["ion_den"],
            ),
            joint_weight,
            {},
        ),
        "positivity_joint_1pct": (
            normalized_candidate(
                prediction,
                raw_phys,
                norm_low,
                norm_high,
                delta_phi=positivity_phi_delta,
                delta_q=positivity_q_delta,
            ),
            positivity_ratios,
            joint_weight,
            positivity_metadata,
        ),
        "phi_only_exact": (
            normalized_candidate(
                prediction,
                raw_phys,
                norm_low,
                norm_high,
                delta_phi=phi_delta,
            ),
            phi_full_ratios,
            unit_weight,
            {},
        ),
        "weighted_joint_exact": (
            normalized_candidate(
                prediction,
                raw_phys,
                norm_low,
                norm_high,
                delta_phi=joint_phi_delta,
                delta_q=joint_q_delta,
            ),
            joint_full_ratios,
            unit_weight,
            {},
        ),
    }
    for label, (first_mode, last_mode) in SPECTRAL_PROTECTIONS.items():
        protected_phi_delta = protect_phi_correction_modes(
            positivity_phi_delta, first_mode, last_mode
        )
        protected_ratios = correction_ratios(
            protected_phi_delta,
            positivity_q_delta,
            unit_weight,
            scales["phi"],
            scales["electron_den"],
            scales["ion_den"],
        )
        protected_metadata = {
            **positivity_metadata,
            "protected_phi_modes": f"n={first_mode}-{last_mode}",
            "spectral_protection_rule": (
                "hard zero of the selected azimuthal Fourier coefficients "
                "of the phi correction; density correction unchanged"
            ),
            "channel_budget_satisfied": {
                name: bool(float(np.max(values)) <= budget * (1.0 + 1.0e-9))
                for name, values in protected_ratios.items()
            },
            "channel_correction_ratio_max": {
                name: float(np.max(values))
                for name, values in protected_ratios.items()
            },
        }
        candidates[label] = (
            normalized_candidate(
                prediction,
                raw_phys,
                norm_low,
                norm_high,
                delta_phi=protected_phi_delta,
                delta_q=positivity_q_delta,
            ),
            protected_ratios,
            joint_weight,
            protected_metadata,
        )

    soft_phi_delta = soft_protect_phi_correction_modes(
        positivity_phi_delta, 9, 21, soft_gate_strength
    )
    soft_ratios = correction_ratios(
        soft_phi_delta,
        positivity_q_delta,
        unit_weight,
        scales["phi"],
        scales["electron_den"],
        scales["ion_den"],
    )
    soft_metadata = {
        **positivity_metadata,
        "protected_phi_modes": "soft n=9-21",
        "spectral_protection_rule": (
            "per-frame smoothstep attenuation of the n=9-21 phi correction "
            "from the raw-prediction Poisson OOD score; density correction unchanged"
        ),
        "soft_gate_reference_fitted_in_this_case": reference_fitted_here,
        "soft_gate_reference": resolved_ood_reference,
        "soft_gate_ood_z": scalar_stats(soft_gate_z),
        "soft_gate_protection_strength": scalar_stats(soft_gate_strength),
        "soft_gate_active_frame_fraction": float(
            np.mean(soft_gate_strength > 0.0)
        ),
        "soft_gate_full_frame_fraction": float(
            np.mean(soft_gate_strength >= 1.0 - 1.0e-12)
        ),
        "channel_budget_satisfied": {
            name: bool(float(np.max(values)) <= budget * (1.0 + 1.0e-9))
            for name, values in soft_ratios.items()
        },
        "channel_correction_ratio_max": {
            name: float(np.max(values)) for name, values in soft_ratios.items()
        },
    }
    candidates[SOFT_GATE_DECODER] = (
        normalized_candidate(
            prediction,
            raw_phys,
            norm_low,
            norm_high,
            delta_phi=soft_phi_delta,
            delta_q=positivity_q_delta,
        ),
        soft_ratios,
        joint_weight,
        soft_metadata,
    )

    evaluation_candidates = candidates
    if candidate_labels is not None:
        missing = set(candidate_labels) - set(candidates)
        if missing:
            raise KeyError(f"Unknown candidate labels: {sorted(missing)}")
        evaluation_candidates = {
            label: candidates[label] for label in candidate_labels
        }

    rows = []
    details = {}
    for label, (
        candidate,
        ratios,
        applied_weight,
        candidate_metadata,
    ) in evaluation_candidates.items():
        row, detail = evaluate_candidate(
            label,
            candidate,
            truth,
            prediction,
            eval_times_s,
            norm_low,
            norm_high,
            dx,
            dy,
            ratios,
            applied_weight,
        )
        row["case"] = case.label
        row.update(candidate_metadata)
        detail["truth_free"].update(candidate_metadata)
        rows.append(row)
        details[label] = detail
        print(
            f"[{case.label}/{label}] Pois/raw={row['poisson_residual_ratio_to_raw']:.5g} "
            f"ne={row['ne_relative_l2']:.5g} ni={row['ni_relative_l2']:.5g} "
            f"phi={row['phi_relative_l2']:.5g} E={row['electric_field_relative_l2']:.5g}",
            flush=True,
        )

    joint_exact_phys = physical_fields(
        candidates["weighted_joint_exact"][0], norm_low, norm_high
    )
    joint_exact_residual, _ = poisson_residual(
        joint_exact_phys[:, 0],
        joint_exact_phys[:, 1],
        joint_exact_phys[:, 2],
        dx,
        dy,
    )
    raw_residual_rms = np.sqrt(np.mean(residual**2, axis=(1, 2)))
    joint_exact_rms = np.sqrt(np.mean(joint_exact_residual**2, axis=(1, 2)))
    case_summary = {
        "case": case.label,
        "source_h5": str(case.h5.resolve()),
        "evaluation_time_us": [
            float(eval_times_s[0] * 1.0e6),
            float(eval_times_s[-1] * 1.0e6),
        ],
        "frames": int(len(eval_times_s)),
        "radial_size": case.radial_size,
        "overlap_consistency_before_time_trim": consistency,
        "prediction_scales": {name: scalar_stats(value) for name, value in scales.items()},
        "solver_checks": {
            "joint_exact_float32_residual_ratio_to_raw": scalar_stats(
                joint_exact_rms / np.maximum(raw_residual_rms, TINY)
            ),
            "joint_phi_boundary_max_abs_V": float(
                np.max(np.abs(joint_phi_delta[:, (0, -1), :]))
            ),
            "joint_charge_boundary_max_abs_m3": float(
                np.max(np.abs(joint_q_delta[:, (0, -1), :]))
            ),
        },
        "soft_gate": {
            "reference_fitted_in_this_case": reference_fitted_here,
            "reference": resolved_ood_reference,
            "raw_relative_poisson_residual": scalar_stats(
                raw_relative_poisson
            ),
            "ood_z": scalar_stats(soft_gate_z),
            "protection_strength": scalar_stats(soft_gate_strength),
        },
        "decoders": details,
    }
    return rows, case_summary, resolved_ood_reference


def plot_case_comparison(path: Path, rows: list[dict]) -> None:
    operational = [
        row
        for row in rows
        if row["decoder"]
        in (
            "raw_simvp",
            "phi_only_1pct",
            "weighted_joint_1pct",
            "positivity_joint_1pct",
        )
    ]
    cases = list(dict.fromkeys(row["case"] for row in operational))
    decoders = (
        "raw_simvp",
        "phi_only_1pct",
        "weighted_joint_1pct",
        "positivity_joint_1pct",
    )
    labels = ("raw", "phi-only 1%", "joint 1%", "positive joint 1%")
    metrics = (
        ("poisson_residual_ratio_to_raw", "Poisson residual / raw"),
        ("phi_relative_l2", "phi relative L2"),
        ("electric_field_relative_l2", "electric-field relative L2"),
        ("hidden_n17_21_complex_relative_l2", "n17--21 complex relative L2"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    width = 0.19
    x = np.arange(len(cases), dtype=np.float64)
    lookup = {(row["case"], row["decoder"]): row for row in operational}
    for axis, (metric, title) in zip(axes.ravel(), metrics):
        for index, (decoder, label) in enumerate(zip(decoders, labels)):
            values = [lookup[(case, decoder)][metric] for case in cases]
            axis.bar(x + (index - 1.5) * width, values, width, label=label)
        axis.set_xticks(x, cases)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_spectral_protection_comparison(path: Path, rows: list[dict]) -> None:
    decoders = (
        "positivity_joint_1pct",
        *SPECTRAL_PROTECTIONS,
        SOFT_GATE_DECODER,
    )
    labels = (
        "positive joint",
        "protect n1-6",
        "protect n7-8",
        "protect n9-21",
        "protect n1-21",
        "soft-gate n9-21",
    )
    selected_rows = [row for row in rows if row["decoder"] in decoders]
    cases = list(dict.fromkeys(row["case"] for row in selected_rows))
    lookup = {(row["case"], row["decoder"]): row for row in selected_rows}
    metrics = (
        ("poisson_residual_ratio_to_raw", "full Poisson residual / raw"),
        (
            "poisson_ecdi_n9_21_residual_ratio_to_raw",
            "ECDI-band Poisson residual / raw",
        ),
        ("electric_field_relative_l2", "electric-field relative L2"),
        ("phi_ecdi_power_ratio", "phi ECDI power / truth"),
        ("transport_ecdi_relative_l2", "ECDI transport relative L2"),
        ("hidden_n17_21_complex_relative_l2", "n17--21 complex relative L2"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    width = 0.125
    x = np.arange(len(cases), dtype=np.float64)
    for axis, (metric, title) in zip(axes.ravel(), metrics):
        for index, (decoder, label) in enumerate(zip(decoders, labels)):
            values = [lookup[(case, decoder)][metric] for case in cases]
            axis.bar(x + (index - 2.5) * width, values, width, label=label)
        axis.set_xticks(x, cases, rotation=10)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_soft_gate_summary(path: Path, summaries: dict[str, dict]) -> None:
    cases = list(summaries)
    x = np.arange(len(cases), dtype=np.float64)
    z_stats = [summaries[case]["soft_gate"]["ood_z"] for case in cases]
    gate_stats = [
        summaries[case]["soft_gate"]["protection_strength"] for case in cases
    ]
    z_median = np.asarray([item["median"] for item in z_stats])
    z_min = np.asarray([item["min"] for item in z_stats])
    z_max = np.asarray([item["max"] for item in z_stats])
    gate_mean = np.asarray([item["mean"] for item in gate_stats])
    gate_median = np.asarray([item["median"] for item in gate_stats])
    gate_max = np.asarray([item["max"] for item in gate_stats])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].errorbar(
        x,
        z_median,
        yerr=np.vstack((z_median - z_min, z_max - z_median)),
        fmt="o",
        capsize=5,
        label="median and range",
    )
    axes[0].axhline(SOFT_GATE_Z_START, color="tab:orange", ls="--", label="gate starts")
    axes[0].axhline(SOFT_GATE_Z_FULL, color="tab:red", ls="--", label="full protection")
    axes[0].set_title("raw-Poisson OOD z score")
    axes[0].set_xticks(x, cases, rotation=10)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)

    width = 0.26
    axes[1].bar(x - width, gate_mean, width, label="mean")
    axes[1].bar(x, gate_median, width, label="median")
    axes[1].bar(x + width, gate_max, width, label="max")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("ECDI protection strength")
    axes[1].set_xticks(x, cases, rotation=10)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    requested = args.device
    if requested.startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"
    device = torch.device(requested)
    checkpoint_path = args.g8_workdir / "checkpoint_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("metadata", {}).get("grid_factor", -1)) != 8:
        raise ValueError("checkpoint is not the frozen G8 model")
    cases = (
        Case("E25_stationary", args.stationary_h5, 20.0, 30.0, False, 256),
        Case(
            "E25_to_E22p5_30to35us",
            args.transition_h5,
            30.0,
            35.0,
            True,
            257,
            evaluation_start_us=30.360,
            evaluation_end_us=34.950,
        ),
        Case(
            "E25_to_E22p5_35to40us",
            args.transition_h5,
            35.0,
            40.0,
            True,
            257,
            evaluation_start_us=35.355,
            evaluation_end_us=39.945,
        ),
    )
    all_rows: list[dict] = []
    summaries = {}
    ood_reference: dict[str, float] | None = None
    for case in cases:
        rows, case_summary, case_reference = evaluate_case(
            case,
            checkpoint,
            device,
            args.budget,
            ood_reference=ood_reference,
        )
        if ood_reference is None:
            ood_reference = case_reference
        all_rows.extend(rows)
        summaries[case.label] = case_summary
    operational = {
        case.label: {
            row["decoder"]: row
            for row in all_rows
            if row["case"] == case.label
            and row["decoder"]
            in (
                "raw_simvp",
                "phi_only_1pct",
                "weighted_joint_1pct",
                "positivity_joint_1pct",
            )
        }
        for case in cases
    }
    spectral = {
        case.label: {
            row["decoder"]: row
            for row in all_rows
            if row["case"] == case.label
            and row["decoder"]
            in (
                "positivity_joint_1pct",
                *SPECTRAL_PROTECTIONS,
                SOFT_GATE_DECODER,
            )
        }
        for case in cases
    }
    summary = {
        "description": (
            "Fixed 1% phi-only, prediction-weighted joint, positivity-aware joint, "
            "predeclared hard spectral-protection, and truth-free soft-gated Poisson decoders"
        ),
        "checkpoint": str(checkpoint_path.resolve()),
        "budget": args.budget,
        "selection_policy": {
            "truth_used_for_decoder_or_weight_selection": False,
            "phi_only": "same per-frame 1% phi fluctuation RMS cap as Section 34",
            "weighted_joint": (
                "minimum prediction-normalized L2 exact direction in phi and ni-ne; "
                "ne+ni preserved; common scale caps every channel at 1% of its predicted fluctuation RMS"
            ),
            "positivity_joint": (
                "active-set projection of weighted_joint onto ne,ni>=1e-6 of the smaller predicted "
                "density fluctuation scale, followed by Poisson phi repair inside the original 1% phi RMS ball"
            ),
            "spectral_protection": (
                "remove the phi-correction coefficients in one predeclared band: "
                "MTSI n=1-6, transition n=7-8, ECDI n=9-21, or all n=1-21; "
                "the positivity-aware density correction is unchanged"
            ),
            "soft_gate": (
                "fit median and MAD of log raw relative Poisson residual on E25 stationary; "
                "smoothstep ECDI protection is zero through z=3 and full at z=5; "
                "PIC truth is not used for calibration or gating"
            ),
            "prediction_scales": "computed independently per frame from each raw G8-SimVP prediction",
        },
        "soft_gate_ood_reference": ood_reference,
        "cases": summaries,
        "operational_comparison": operational,
        "spectral_protection_comparison": spectral,
        "claim_boundary": (
            "The transition intervals are post-confirmatory retrospective transfer tests of rules fixed on E25. "
            "They are not independently run native coarse-PIC tests."
        ),
    }
    write_csv(args.output_dir / "decoder_comparison.csv", all_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_case_comparison(args.output_dir / "decoder_comparison.png", all_rows)
    plot_spectral_protection_comparison(
        args.output_dir / "spectral_protection_comparison.png", all_rows
    )
    plot_soft_gate_summary(
        args.output_dir / "soft_gate_diagnostics.png", summaries
    )
    readme = f"""# G8 proximal decoder comparison

This directory compares the unchanged {100*args.budget:g}% phi-only proximal
decoder with a weighted joint phi/charge-difference projection.  Both use the
same frozen E25-trained G8-SimVP and truth-free, per-frame displacement rules.

The E25 stationary held-out interval is a development reference.  The two
E25->E22.5 intervals compare early and late relaxation under the same frozen
decoder.  PIC truth is used only for the metrics in `decoder_comparison.csv`
and was not used to select the joint weights or correction cap.

The joint projection preserves ne+ni and radial boundary values.  It uses
predicted channel fluctuation scales to allocate low/high spatial-frequency
Poisson residual between charge difference and potential.  Electric fields are
derived from phi; they are not independently predicted channels.

The positivity-aware variant projects active density cells onto a small
positive floor while preserving ne+ni, then reduces the newly introduced
Poisson residual using only the remaining feasible part of the original 1%
phi-correction ball.

Four hard spectral variants start from that positivity-aware correction and
remove its phi coefficients in one predeclared azimuthal band: MTSI n=1--6,
transition n=7--8, ECDI n=9--21, or all instability modes n=1--21.  Thus the
raw G8-SimVP phi carrier is retained exactly in the protected band.  Density
positivity and the original channel budgets remain enforced; the density
correction itself is not spectrally filtered.

The soft-gated ECDI variant calibrates a robust log relative-Poisson-residual
reference from raw E25 stationary predictions only.  It leaves n=9--21 phi
corrections unchanged through three robust sigma of that reference and smoothly
attenuates them to zero between z=3 and z=5.  No PIC truth is used by this gate.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False), flush=True)
    print(f"[saved] {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
