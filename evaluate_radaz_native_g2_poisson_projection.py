#!/usr/bin/env python3
"""Evaluate fine-grid Poisson projection of the E20 native-G2 reconstruction.

This is a post-processing experiment.  It never updates the trained SimVPv2
checkpoint.  The primary comparison is among native interpolation, native
densities plus Poisson projection, the current three-channel SimVP result, and
SimVP densities plus Poisson projection.  Fine densities plus projection are
used only to validate the discrete solver.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_banded


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent / "PEPAPIC" / "test" / "results" / "2D_Landmark"
FINE_CASE = "2D_RadAz_Xe1p_Bx20mT_Ez20kVm_dt15ps_out15ns"
FINE_ROOT = RESULTS / FINE_CASE / FINE_CASE
DEFAULT_FINE = FINE_ROOT / "analysis_fields_uncompressed.h5"
COARSE_CASE = "2D_RadAz_Xe1p_Bx20mT_Ez20kVm_G2coarse_0to30us_dt15ps_out15ns"
DEFAULT_COARSE = (
    ROOT.parent
    / "research_results"
    / "2D_RadAz"
    / "PEPAPIC"
    / "2D_Landmark"
    / COARSE_CASE
    / COARSE_CASE
    / "analysis_fields_uncompressed_20to30us.h5"
)
DEFAULT_RECONSTRUCTION = (
    ROOT
    / "workdirs"
    / "radaz_e20_native_g2_reconstruction"
    / "native_g2_reconstructed_3ch_20to30us.h5"
)
DEFAULT_OUTPUT = ROOT / "workdirs" / "radaz_e20_native_g2_poisson_projection"
CHANNELS = ("electron_den", "ion_den", "phi")
EPS0 = 8.8541878128e-12
E_CHARGE = 1.602176634e-19
TINY = np.finfo(np.float64).tiny

METHOD_LABELS = {
    "native_interpolation": "native interpolation",
    "native_density_poisson": "native density + Poisson",
    "simvp": "three-channel SimVP",
    "simvp_density_poisson": "SimVP density + Poisson",
}
METHOD_COLORS = {
    "native_interpolation": "#6b7280",
    "native_density_poisson": "#2563eb",
    "simvp": "#f59e0b",
    "simvp_density_poisson": "#16a34a",
}
BANDS = {
    "n1_6": slice(1, 7),
    "n7_8": slice(7, 9),
    "n9_21": slice(9, 22),
    "n1_32": slice(1, 33),
}

PROTOCOL = {
    "status": "fixed_before_projection_metrics",
    "scope": (
        "No-retraining post-processing test of a discrete fine-grid Poisson "
        "projection for the E20/B20 native-G2 reconstruction."
    ),
    "methods": {
        "A_native_interpolation": "Interpolated native-G2 ne, ni, and phi.",
        "B_native_density_poisson": (
            "Interpolated native-G2 ne and ni with phi replaced by the fine-grid "
            "discrete Poisson solution; projection-only control."
        ),
        "C_simvp": "Current data-only residual SimVP ne, ni, and phi.",
        "D_simvp_density_poisson": (
            "Current SimVP ne and ni with phi replaced by the fine-grid discrete "
            "Poisson solution."
        ),
        "solver_validation": (
            "Fine ne and ni projected to phi and compared with fine phi; never a "
            "candidate method."
        ),
    },
    "operator": {
        "equation": "laplacian(phi) + e*(ni-ne)/eps0 = 0",
        "radial": (
            "second-order centered difference; Dirichlet phi=0 at x=0 and "
            "x=12.8 mm.  The stored 256-point core ends at x=12.75 mm, so the "
            "known outer boundary is appended to the solve and omitted again."
        ),
        "azimuthal": "second-order centered difference with 256-point periodic FFT",
        "tuning": "No filtering, blending, clipping, fitted scale, or target-dependent tuning.",
    },
    "paired_truth_warning": (
        "Fine and native-G2 runs are independent particle realizations.  Primary "
        "comparisons use distributions, profiles, spectra, and transport distributions, "
        "not frame-wise pixel error or phase coherence."
    ),
    "primary_gates_for_simvp_density_poisson_vs_native_interpolation": {
        "field": (
            "Both phi quantile distance and phi radial-profile relative L2 must "
            "not exceed native interpolation."
        ),
        "mode": (
            "Both phi and Ey log10 power RMSE over n=1..32 must not exceed native "
            "interpolation."
        ),
        "transport": (
            "Both ECDI-band n=9..21 and total n=1..32 transport quantile distances "
            "must not exceed native interpolation."
        ),
        "poisson": "Median relative residual must not exceed 10 times the fine-truth floor.",
        "overall": "All four gates must pass; otherwise report the trade-off explicitly.",
    },
    "secondary_diagnostics_not_used_to_change_gates": (
        "Distribution, radial profile, and modal power of the Poisson source ni-ne "
        "are reported to identify whether separately corrected densities create an "
        "incorrect charge-imbalance field.  Raw fine and raw native-G2 fields are also "
        "checked with their own grid spacings to distinguish solver consistency from "
        "interpolation-induced fine-grid inconsistency."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction-h5", type=Path, default=DEFAULT_RECONSTRUCTION)
    parser.add_argument("--coarse-h5", type=Path, default=DEFAULT_COARSE)
    parser.add_argument("--fine-h5", type=Path, default=DEFAULT_FINE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocol-only", action="store_true")
    return parser.parse_args()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def selected_indices(time_s: np.ndarray, start_us: float, end_us: float) -> np.ndarray:
    selected = np.flatnonzero(
        (time_s * 1.0e6 >= start_us - 1.0e-9)
        & (time_s * 1.0e6 <= end_us + 1.0e-9)
    )
    if not len(selected) or not np.all(np.diff(selected) == 1):
        raise ValueError("Requested time interval is empty or non-contiguous")
    return selected


def interpolation_stencil(source: np.ndarray, target: np.ndarray, periodic: bool):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if periodic:
        spacing = float(np.median(np.diff(source)))
        period = spacing * len(source)
        coordinate = np.mod(target - source[0], period) / spacing
        left_raw = np.floor(coordinate).astype(np.int64)
        alpha = coordinate - left_raw
        return (
            np.mod(left_raw, len(source)),
            np.mod(left_raw + 1, len(source)),
            alpha.astype(np.float32),
        )
    right = np.searchsorted(source, target, side="right")
    right = np.clip(right, 1, len(source) - 1)
    left = right - 1
    alpha = (target - source[left]) / (source[right] - source[left])
    below, above = target <= source[0], target >= source[-1]
    left[below] = right[below] = 0
    alpha[below] = 0.0
    left[above] = right[above] = len(source) - 1
    alpha[above] = 0.0
    return left, right, alpha.astype(np.float32)


def interpolate_native_to_model(
    coarse: np.ndarray,
    coarse_x: np.ndarray,
    coarse_y_unique: np.ndarray,
    fine_x: np.ndarray,
    fine_y: np.ndarray,
) -> np.ndarray:
    if coarse.ndim != 4 or coarse.shape[1:] != (3, 129, 128):
        raise ValueError(f"Expected native coarse shape (T,3,129,128), got {coarse.shape}")
    r0, r1, ra = interpolation_stencil(coarse_x, fine_x[:256], periodic=False)
    a0, a1, aa = interpolation_stencil(coarse_y_unique, fine_y[:256], periodic=True)
    radial = (
        coarse[:, :, r0, :] * (1.0 - ra)[None, None, :, None]
        + coarse[:, :, r1, :] * ra[None, None, :, None]
    )
    core = (
        radial[:, :, :, a0] * (1.0 - aa)[None, None, None, :]
        + radial[:, :, :, a1] * aa[None, None, None, :]
    )
    output = np.empty((len(coarse), 3, 260, 256), dtype=np.float32)
    output[:, :, :256] = core
    output[:, :, 256:] = core[:, :, -1:, :]
    return output


def load_native_baseline(
    path: Path,
    start_us: float,
    end_us: float,
    fine_x: np.ndarray,
    fine_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        time_s = np.asarray(handle["axes/time_s"], dtype=np.float64)
        selected = selected_indices(time_s, start_us, end_us)
        x = np.asarray(handle["axes/x_m"], dtype=np.float64)
        y = np.asarray(handle["axes/y_m"], dtype=np.float64)
        if (len(x), len(y)) != (129, 129):
            raise ValueError(f"Expected native G2 stitched grid 129x129, got {len(x)}x{len(y)}")
        fields = np.empty((len(selected), 3, 129, 128), dtype=np.float32)
        for channel, name in enumerate(CHANNELS):
            fields[:, channel] = np.asarray(
                handle[f"fields/{name}"][int(selected[0]) : int(selected[-1]) + 1, :, :128],
                dtype=np.float32,
            )
    baseline = interpolate_native_to_model(fields, x, y[:128], fine_x, fine_y)
    return baseline, time_s[selected], selected


def load_fine_reference(
    path: Path, start_us: float, end_us: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        time_s = np.asarray(handle["axes/time_s"], dtype=np.float64)
        selected = selected_indices(time_s, start_us, end_us)
        fine = np.empty((len(selected), 3, 256, 256), dtype=np.float32)
        for channel, name in enumerate(CHANNELS):
            fine[:, channel] = np.asarray(
                handle[f"fields/{name}"][int(selected[0]) : int(selected[-1]) + 1, :256, :256],
                dtype=np.float32,
            )
        x = np.asarray(handle["axes/x_m"], dtype=np.float64)
        y = np.asarray(handle["axes/y_m"], dtype=np.float64)
    return fine, time_s[selected], x, y


def sample_flat(values: np.ndarray, maximum: int = 1_500_000) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).ravel()
    stride = max(1, int(math.ceil(len(flat) / maximum)))
    return np.asarray(flat[::stride], dtype=np.float64)


def distribution_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    ref = sample_flat(reference)
    cand = sample_flat(candidate)
    quantiles = np.linspace(0.01, 0.99, 99)
    reference_quantiles = np.quantile(ref, quantiles)
    candidate_quantiles = np.quantile(cand, quantiles)
    reference_std = max(float(np.std(ref)), TINY)
    return {
        "reference_mean": float(np.mean(ref)),
        "candidate_mean": float(np.mean(cand)),
        "mean_shift_over_reference_std": float(abs(np.mean(cand) - np.mean(ref)) / reference_std),
        "reference_std": float(np.std(ref)),
        "candidate_std": float(np.std(cand)),
        "std_ratio": float(np.std(cand) / reference_std),
        "rms_ratio": float(
            np.sqrt(np.mean(cand**2)) / max(float(np.sqrt(np.mean(ref**2))), TINY)
        ),
        "quantile_rmse_over_reference_std": float(
            np.sqrt(np.mean((candidate_quantiles - reference_quantiles) ** 2)) / reference_std
        ),
        "q01_ratio_or_shift": float((candidate_quantiles[0] - reference_quantiles[0]) / reference_std),
        "q50_shift_over_reference_std": float((candidate_quantiles[49] - reference_quantiles[49]) / reference_std),
        "q99_ratio_or_shift": float((candidate_quantiles[-1] - reference_quantiles[-1]) / reference_std),
    }


def centered_correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).ravel().copy()
    b = np.asarray(right, dtype=np.float64).ravel().copy()
    a -= np.mean(a)
    b -= np.mean(b)
    return float(np.dot(a, b) / max(math.sqrt(float(np.dot(a, a) * np.dot(b, b))), TINY))


def profile_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    return {
        "relative_l2": float(np.linalg.norm(cand - ref) / max(np.linalg.norm(ref), TINY)),
        "nrmse_over_profile_std": float(
            np.sqrt(np.mean((cand - ref) ** 2)) / max(np.std(ref), TINY)
        ),
        "correlation": centered_correlation(ref, cand),
    }


def aligned_indices(source_time: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(source_time, target_time)
    if np.any(indices >= len(source_time)) or not np.allclose(
        source_time[indices], target_time, atol=1.0e-14, rtol=0.0
    ):
        raise ValueError("Could not align the PIC frames to reconstruction times")
    return indices


def poisson_project_phi(
    ne: np.ndarray,
    ni: np.ndarray,
    dx: float,
    dy: float,
    label: str,
) -> np.ndarray:
    """Solve the exact discrete operator used by the residual diagnostic.

    Input/output contain x indices 0..255.  x=0 is a Dirichlet boundary, x=256
    is the omitted outer Dirichlet boundary, and x=1..255 are solved unknowns.
    """
    if ne.shape != ni.shape or ne.ndim != 3 or ne.shape[1:] != (256, 256):
        raise ValueError(f"Expected density arrays (T,256,256), got {ne.shape} and {ni.shape}")
    nt, nx_saved, ny = ne.shape
    nx_interior = nx_saved - 1
    source = E_CHARGE * (ni[:, 1:].astype(np.float64) - ne[:, 1:].astype(np.float64)) / EPS0
    source_hat = np.fft.rfft(source, axis=-1)
    del source
    solution_hat = np.empty_like(source_hat)
    radial_offdiag = 1.0 / dx**2
    mode_index = np.arange(source_hat.shape[-1], dtype=np.float64)
    azimuth_eigenvalue = -4.0 * np.sin(np.pi * mode_index / ny) ** 2 / dy**2
    for mode, eigenvalue in enumerate(azimuth_eigenvalue):
        banded = np.zeros((3, nx_interior), dtype=np.float64)
        banded[0, 1:] = radial_offdiag
        banded[1, :] = -2.0 / dx**2 + eigenvalue
        banded[2, :-1] = radial_offdiag
        rhs = -source_hat[:, :, mode].T
        solution_hat[:, :, mode] = solve_banded(
            (1, 1), banded, rhs, overwrite_ab=True, overwrite_b=False, check_finite=False
        ).T
        if mode in {0, 32, 64, 96, 128}:
            print(f"[project:{label}] mode {mode}/{source_hat.shape[-1] - 1}", flush=True)
    interior = np.fft.irfft(solution_hat, n=ny, axis=-1)
    projected = np.zeros((nt, nx_saved, ny), dtype=np.float64)
    projected[:, 1:] = interior
    return projected


def scalar_modal_power(values: np.ndarray, maximum_mode: int = 64, chunk: int = 16) -> np.ndarray:
    total = np.zeros(maximum_mode + 1, dtype=np.float64)
    samples = 0
    for start in range(0, len(values), chunk):
        block = np.asarray(values[start : start + chunk], dtype=np.float64)
        coeff = np.fft.rfft(block, axis=-1, norm="forward")[..., : maximum_mode + 1]
        total += np.sum(np.abs(coeff) ** 2, axis=(0, 1))
        samples += block.shape[0] * block.shape[1]
    return total / samples


def ey_modal_power(phi: np.ndarray, dy: float, maximum_mode: int = 64, chunk: int = 16) -> np.ndarray:
    total = np.zeros(maximum_mode + 1, dtype=np.float64)
    samples = 0
    for start in range(0, len(phi), chunk):
        block = np.asarray(phi[start : start + chunk], dtype=np.float64)
        ey = -(np.roll(block, -1, axis=-1) - np.roll(block, 1, axis=-1)) / (2.0 * dy)
        coeff = np.fft.rfft(ey, axis=-1, norm="forward")[..., : maximum_mode + 1]
        total += np.sum(np.abs(coeff) ** 2, axis=(0, 1))
        samples += block.shape[0] * block.shape[1]
    return total / samples


def modal_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    modes = np.arange(1, 33)
    floor = max(float(np.max(reference[1:33])) * 1.0e-14, TINY)
    error = np.log10(np.maximum(candidate[1:33], floor)) - np.log10(
        np.maximum(reference[1:33], floor)
    )
    result = {
        "log10_power_rmse_n1_32": float(np.sqrt(np.mean(error**2))),
        "dominant_mode_reference_n1_32": int(modes[np.argmax(reference[1:33])]),
        "dominant_mode_candidate_n1_32": int(modes[np.argmax(candidate[1:33])]),
        "bands": {},
    }
    for band, selection in BANDS.items():
        result["bands"][band] = {
            "candidate_to_reference_power": float(
                np.sum(candidate[selection]) / max(float(np.sum(reference[selection])), TINY)
            )
        }
    return result


def modal_transport_components(ne: np.ndarray, phi: np.ndarray, dy: float, chunk: int = 16) -> dict:
    output = {name: [] for name in BANDS}
    for start in range(0, len(phi), chunk):
        ne_block = np.asarray(ne[start : start + chunk], dtype=np.float64)
        phi_block = np.asarray(phi[start : start + chunk], dtype=np.float64)
        ey = -(np.roll(phi_block, -1, axis=-1) - np.roll(phi_block, 1, axis=-1)) / (2.0 * dy)
        ne_mode = np.fft.rfft(ne_block, axis=-1, norm="forward")
        ey_mode = np.fft.rfft(ey, axis=-1, norm="forward")
        flux = 2.0 * np.mean(np.real(ne_mode * np.conj(ey_mode)), axis=1)
        for name, selection in BANDS.items():
            output[name].append(np.sum(flux[:, selection], axis=1))
    return {name: np.concatenate(parts) for name, parts in output.items()}


def poisson_metrics_components(
    ne: np.ndarray, ni: np.ndarray, phi: np.ndarray, dx: float, dy: float, chunk: int = 16
) -> dict:
    relative = []
    correlation = []
    for start in range(0, len(phi), chunk):
        ne_block = np.asarray(ne[start : start + chunk], dtype=np.float64)
        ni_block = np.asarray(ni[start : start + chunk], dtype=np.float64)
        phi_block = np.asarray(phi[start : start + chunk], dtype=np.float64)
        radial = (phi_block[:, 2:] - 2.0 * phi_block[:, 1:-1] + phi_block[:, :-2]) / dx**2
        azimuth = (
            np.roll(phi_block[:, 1:-1], -1, axis=-1)
            - 2.0 * phi_block[:, 1:-1]
            + np.roll(phi_block[:, 1:-1], 1, axis=-1)
        ) / dy**2
        laplacian = radial + azimuth
        source = E_CHARGE * (ni_block[:, 1:-1] - ne_block[:, 1:-1]) / EPS0
        residual = laplacian + source
        relative.extend(
            np.sqrt(np.mean(residual**2, axis=(1, 2)))
            / np.maximum(np.sqrt(np.mean(source**2, axis=(1, 2))), TINY)
        )
        left = laplacian.reshape(len(phi_block), -1)
        right = (-source).reshape(len(phi_block), -1)
        left -= np.mean(left, axis=1, keepdims=True)
        right -= np.mean(right, axis=1, keepdims=True)
        correlation.extend(
            np.sum(left * right, axis=1)
            / np.maximum(np.sqrt(np.sum(left**2, axis=1) * np.sum(right**2, axis=1)), TINY)
        )
    return {
        "relative_poisson_residual_mean": float(np.mean(relative)),
        "relative_poisson_residual_median": float(np.median(relative)),
        "poisson_balance_correlation_mean": float(np.mean(correlation)),
        "poisson_balance_correlation_median": float(np.median(correlation)),
    }


def own_grid_poisson_metrics(
    path: Path, target_time_s: np.ndarray, chunk: int = 16
) -> dict:
    relative = []
    correlation = []
    radial_boundary_max = 0.0
    periodic_endpoint_max = 0.0
    with h5py.File(path, "r") as handle:
        source_time = np.asarray(handle["axes/time_s"], dtype=np.float64)
        indices = aligned_indices(source_time, target_time_s)
        x_m = np.asarray(handle["axes/x_m"], dtype=np.float64)
        y_m = np.asarray(handle["axes/y_m"], dtype=np.float64)
        dx = float(np.median(np.diff(x_m)))
        dy = float(np.median(np.diff(y_m)))
        if not np.isclose(y_m[-1] - y_m[0], dy * (len(y_m) - 1), atol=1.0e-14):
            raise ValueError(f"Unexpected azimuthal coordinate in {path}")
        unique_y = len(y_m) - 1
        for start in range(0, len(indices), chunk):
            block_indices = indices[start : start + chunk]
            if not np.all(np.diff(block_indices) == 1):
                raise ValueError("Own-grid Poisson diagnostic requires contiguous frames")
            selection = slice(int(block_indices[0]), int(block_indices[-1]) + 1)
            ne = np.asarray(handle["fields/electron_den"][selection, :, :unique_y], dtype=np.float64)
            ni = np.asarray(handle["fields/ion_den"][selection, :, :unique_y], dtype=np.float64)
            phi = np.asarray(handle["fields/phi"][selection, :, :unique_y], dtype=np.float64)
            radial = (phi[:, 2:] - 2.0 * phi[:, 1:-1] + phi[:, :-2]) / dx**2
            azimuth = (
                np.roll(phi[:, 1:-1], -1, axis=-1)
                - 2.0 * phi[:, 1:-1]
                + np.roll(phi[:, 1:-1], 1, axis=-1)
            ) / dy**2
            laplacian = radial + azimuth
            source = E_CHARGE * (ni[:, 1:-1] - ne[:, 1:-1]) / EPS0
            residual = laplacian + source
            relative.extend(
                np.sqrt(np.mean(residual**2, axis=(1, 2)))
                / np.maximum(np.sqrt(np.mean(source**2, axis=(1, 2))), TINY)
            )
            left = laplacian.reshape(len(phi), -1)
            right = (-source).reshape(len(phi), -1)
            left -= np.mean(left, axis=1, keepdims=True)
            right -= np.mean(right, axis=1, keepdims=True)
            correlation.extend(
                np.sum(left * right, axis=1)
                / np.maximum(np.sqrt(np.sum(left**2, axis=1) * np.sum(right**2, axis=1)), TINY)
            )
            radial_boundary_max = max(
                radial_boundary_max,
                float(np.max(np.abs(phi[:, (0, -1), :]))),
            )
            full_phi = np.asarray(handle["fields/phi"][selection, :, :], dtype=np.float64)
            periodic_endpoint_max = max(
                periodic_endpoint_max,
                float(np.max(np.abs(full_phi[:, :, -1] - full_phi[:, :, 0]))),
            )
    return {
        "path": str(path.resolve()),
        "grid_nodes_with_duplicate_endpoint": [int(len(x_m)), int(len(y_m))],
        "unique_azimuthal_nodes": int(unique_y),
        "dx_m": dx,
        "dy_m": dy,
        "relative_poisson_residual_mean": float(np.mean(relative)),
        "relative_poisson_residual_median": float(np.median(relative)),
        "poisson_balance_correlation_mean": float(np.mean(correlation)),
        "poisson_balance_correlation_median": float(np.median(correlation)),
        "radial_dirichlet_boundary_max_abs_phi_v": radial_boundary_max,
        "periodic_endpoint_max_abs_mismatch_phi_v": periodic_endpoint_max,
    }


def sampled_correlation(left: np.ndarray, right: np.ndarray, maximum: int = 2_000_000) -> float:
    stride = max(1, int(math.ceil(left.size / maximum)))
    a = np.asarray(left, dtype=np.float64).ravel()[::stride].copy()
    b = np.asarray(right, dtype=np.float64).ravel()[::stride].copy()
    a -= np.mean(a)
    b -= np.mean(b)
    return float(np.dot(a, b) / max(math.sqrt(float(np.dot(a, a) * np.dot(b, b))), TINY))


def evaluate_method(
    reference: tuple[np.ndarray, np.ndarray, np.ndarray],
    candidate: tuple[np.ndarray, np.ndarray, np.ndarray],
    reference_powers: dict[str, np.ndarray],
    reference_transport: dict[str, np.ndarray],
    dx: float,
    dy: float,
) -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
    ref_ne, ref_ni, ref_phi = reference
    ne, ni, phi = candidate
    fields = {"electron_den": ne, "ion_den": ni, "phi": phi}
    ref_fields = {"electron_den": ref_ne, "ion_den": ref_ni, "phi": ref_phi}
    distributions = {}
    profiles = {}
    powers = {}
    modes = {}
    for name in CHANNELS:
        distributions[name] = distribution_metrics(ref_fields[name], fields[name])
        ref_profile = np.mean(ref_fields[name], axis=(0, 2), dtype=np.float64)
        candidate_profile = np.mean(fields[name], axis=(0, 2), dtype=np.float64)
        profiles[name] = profile_metrics(ref_profile, candidate_profile)
        powers[name] = scalar_modal_power(fields[name])
        modes[name] = modal_metrics(reference_powers[name], powers[name])
    reference_charge_source = ref_ni - ref_ne
    charge_source = ni - ne
    powers["charge_source"] = scalar_modal_power(charge_source)
    charge_source_metrics = {
        "distribution": distribution_metrics(reference_charge_source, charge_source),
        "radial_profile": profile_metrics(
            np.mean(reference_charge_source, axis=(0, 2), dtype=np.float64),
            np.mean(charge_source, axis=(0, 2), dtype=np.float64),
        ),
        "modal_power": modal_metrics(reference_powers["charge_source"], powers["charge_source"]),
    }
    powers["ey"] = ey_modal_power(phi, dy)
    modes["ey"] = modal_metrics(reference_powers["ey"], powers["ey"])
    transport_series = modal_transport_components(ne, phi, dy)
    transports = {
        band: distribution_metrics(reference_transport[band], transport_series[band])
        for band in BANDS
    }
    result = {
        "field_distribution": distributions,
        "radial_profile": profiles,
        "modal_power": modes,
        "poisson_source_ni_minus_ne": charge_source_metrics,
        "transport_distribution": transports,
        "poisson": poisson_metrics_components(ne, ni, phi, dx, dy),
    }
    return result, powers, transport_series


def save_projection_h5(
    path: Path,
    time_s: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    native_phi: np.ndarray,
    simvp_phi: np.ndarray,
) -> None:
    with h5py.File(path, "w") as handle:
        kwargs = {"chunks": (1, 256, 256), "compression": "gzip", "compression_opts": 4}
        handle.create_dataset("native_density_poisson_phi", data=native_phi, **kwargs)
        handle.create_dataset("simvp_density_poisson_phi", data=simvp_phi, **kwargs)
        handle.create_dataset("time_s", data=time_s)
        handle.create_dataset("x_m", data=x_m)
        handle.create_dataset("y_m", data=y_m)
        handle.attrs["units"] = "phi in V"
        handle.attrs["equation"] = "laplacian(phi) + e*(ni-ne)/eps0 = 0"
        handle.attrs["radial_boundary"] = "Dirichlet phi=0 at x=0 and x=12.8 mm"
        handle.attrs["azimuthal_boundary"] = "periodic"
        handle.attrs["retraining"] = False


def make_spectrum_plot(
    path: Path, powers: dict[str, dict[str, np.ndarray]], reference_powers: dict[str, np.ndarray]
) -> None:
    modes = np.arange(1, 65)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    for axis, field, title in zip(
        axes,
        ("phi", "ey", "charge_source"),
        (r"$\phi$ power", r"$E_y$ power", r"$(n_i-n_e)$ power"),
    ):
        axis.axhline(1.0, color="black", linewidth=1.0, label="fine reference")
        for method in METHOD_LABELS:
            ratio = powers[method][field][1:65] / np.maximum(reference_powers[field][1:65], TINY)
            axis.semilogy(modes, ratio, label=METHOD_LABELS[method], color=METHOD_COLORS[method])
        axis.axvspan(9, 21, color="#e5e7eb", alpha=0.5, label="ECDI n=9--21" if field == "phi" else None)
        axis.set_xlabel("azimuthal mode n")
        axis.set_ylabel("candidate / fine power")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_metric_plot(path: Path, results: dict) -> None:
    metric_specs = [
        ("phi quantile", lambda r: r["field_distribution"]["phi"]["quantile_rmse_over_reference_std"]),
        ("phi profile", lambda r: r["radial_profile"]["phi"]["relative_l2"]),
        ("phi spectrum", lambda r: r["modal_power"]["phi"]["log10_power_rmse_n1_32"]),
        ("Ey spectrum", lambda r: r["modal_power"]["ey"]["log10_power_rmse_n1_32"]),
        (
            "charge spectrum",
            lambda r: r["poisson_source_ni_minus_ne"]["modal_power"]["log10_power_rmse_n1_32"],
        ),
        ("ECDI transport", lambda r: r["transport_distribution"]["n9_21"]["quantile_rmse_over_reference_std"]),
        ("total transport", lambda r: r["transport_distribution"]["n1_32"]["quantile_rmse_over_reference_std"]),
    ]
    methods = list(METHOD_LABELS)
    x = np.arange(len(metric_specs))
    width = 0.19
    fig, axis = plt.subplots(figsize=(13, 5), constrained_layout=True)
    baseline = np.array([getter(results["native_interpolation"]) for _, getter in metric_specs])
    for index, method in enumerate(methods):
        values = np.array([getter(results[method]) for _, getter in metric_specs])
        axis.bar(
            x + (index - 1.5) * width,
            values / np.maximum(baseline, TINY),
            width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
        )
    axis.axhline(1.0, color="black", linewidth=1.0)
    axis.set_xticks(x, [name for name, _ in metric_specs], rotation=20, ha="right")
    axis.set_ylabel("error / native-interpolation error")
    axis.set_title("Native-G2 Poisson projection: lower is better")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_snapshot_plot(
    path: Path,
    time_us: float,
    fine_phi: np.ndarray,
    native_phi: np.ndarray,
    native_projected: np.ndarray,
    simvp_phi: np.ndarray,
    simvp_projected: np.ndarray,
) -> None:
    fields = (fine_phi, native_phi, native_projected, simvp_phi, simvp_projected)
    titles = (
        "fine (independent)",
        "native interpolation",
        "native density + Poisson",
        "three-channel SimVP",
        "SimVP density + Poisson",
    )
    low, high = np.quantile(np.concatenate([item.ravel() for item in fields]), [0.01, 0.99])
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.7), constrained_layout=True)
    image = None
    for axis, field, title in zip(axes, fields, titles):
        image = axis.imshow(field, origin="lower", aspect="auto", cmap="coolwarm", vmin=low, vmax=high)
        axis.set_title(title, fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.colorbar(image, ax=axes, label=r"$\phi$ [V]", shrink=0.85)
    fig.suptitle(f"Independent-realization snapshot at {time_us:.3f} us (visual only)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output_dir / "poisson_projection_protocol.json"
    protocol_path.write_text(json.dumps(PROTOCOL, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[protocol] {protocol_path}", flush=True)
    if args.protocol_only:
        return

    with h5py.File(args.reconstruction_h5, "r") as handle:
        prediction = np.asarray(handle["data_tchw"], dtype=np.float32)
        time_s = np.asarray(handle["time_s"], dtype=np.float64)
        x_saved = np.asarray(handle["x_m"], dtype=np.float64)
        y_saved = np.asarray(handle["y_m"], dtype=np.float64)
    if prediction.shape != (len(time_s), 3, 256, 256):
        raise ValueError(f"Unexpected reconstruction shape {prediction.shape}")
    dx = float(np.median(np.diff(x_saved)))
    dy = float(np.median(np.diff(y_saved)))
    expected_outer = float(x_saved[0] + 256 * dx)
    if not np.isclose(expected_outer, 1.28e-2, atol=1.0e-14):
        raise ValueError(f"Unexpected inferred outer radial boundary {expected_outer}")

    fine_all, fine_time, fine_x, fine_y = load_fine_reference(
        args.fine_h5, float(time_s[0] * 1.0e6 - 1.0e-6), float(time_s[-1] * 1.0e6 + 1.0e-6)
    )
    fine_indices = aligned_indices(fine_time, time_s)
    fine = fine_all[fine_indices]
    del fine_all
    baseline_all, baseline_time, _ = load_native_baseline(
        args.coarse_h5,
        float(time_s[0] * 1.0e6 - 1.0e-6),
        float(time_s[-1] * 1.0e6 + 1.0e-6),
        fine_x,
        fine_y,
    )
    baseline_indices = aligned_indices(baseline_time, time_s)
    baseline = np.asarray(baseline_all[baseline_indices, :, :256], dtype=np.float32)
    del baseline_all

    if not np.allclose(x_saved, fine_x[:256]) or not np.allclose(y_saved, fine_y[:256]):
        raise ValueError("Reconstruction and fine coordinate axes differ")
    print(f"[loaded] {len(time_s)} frames, {time_s[0]*1e6:.3f}--{time_s[-1]*1e6:.3f} us", flush=True)

    own_grid_physics = {
        "raw_fine_on_fine_grid": own_grid_poisson_metrics(args.fine_h5, time_s),
        "raw_native_g2_on_native_grid": own_grid_poisson_metrics(args.coarse_h5, time_s),
    }
    (args.output_dir / "own_grid_poisson_diagnostics.json").write_text(
        json.dumps(json_safe(own_grid_physics), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        "[own-grid] fine="
        f"{own_grid_physics['raw_fine_on_fine_grid']['relative_poisson_residual_median']:.6g}, "
        "native="
        f"{own_grid_physics['raw_native_g2_on_native_grid']['relative_poisson_residual_median']:.6g}",
        flush=True,
    )

    fine_projected_phi = poisson_project_phi(fine[:, 0], fine[:, 1], dx, dy, "fine-validation")
    solver_validation = {
        "role": "operator validation only; not a candidate reconstruction",
        "phi_relative_l2": float(
            np.linalg.norm(fine_projected_phi - fine[:, 2])
            / max(float(np.linalg.norm(fine[:, 2])), TINY)
        ),
        "phi_sampled_correlation": sampled_correlation(fine_projected_phi, fine[:, 2]),
        "phi_distribution": distribution_metrics(fine[:, 2], fine_projected_phi),
        "phi_radial_profile": profile_metrics(
            np.mean(fine[:, 2], axis=(0, 2), dtype=np.float64),
            np.mean(fine_projected_phi, axis=(0, 2), dtype=np.float64),
        ),
        "projected_poisson": poisson_metrics_components(
            fine[:, 0], fine[:, 1], fine_projected_phi, dx, dy
        ),
        "saved_fine_poisson": poisson_metrics_components(fine[:, 0], fine[:, 1], fine[:, 2], dx, dy),
    }
    del fine_projected_phi
    (args.output_dir / "poisson_solver_validation.json").write_text(
        json.dumps(json_safe(solver_validation), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        "[validate] fine-density phi relative L2="
        f"{solver_validation['phi_relative_l2']:.6g}, corr="
        f"{solver_validation['phi_sampled_correlation']:.9f}",
        flush=True,
    )

    projection_h5 = args.output_dir / "native_g2_poisson_projected_phi_20to30us.h5"
    if projection_h5.exists():
        with h5py.File(projection_h5, "r") as handle:
            saved_time = np.asarray(handle["time_s"], dtype=np.float64)
            if not np.array_equal(saved_time, time_s):
                raise ValueError("Existing projected H5 has a different time axis")
            native_projected_phi = np.asarray(
                handle["native_density_poisson_phi"], dtype=np.float64
            )
            simvp_projected_phi = np.asarray(
                handle["simvp_density_poisson_phi"], dtype=np.float64
            )
        print(f"[reuse] {projection_h5}", flush=True)
    else:
        native_projected_phi = poisson_project_phi(
            baseline[:, 0], baseline[:, 1], dx, dy, "native-density"
        )
        simvp_projected_phi = poisson_project_phi(
            prediction[:, 0], prediction[:, 1], dx, dy, "simvp-density"
        )
        save_projection_h5(
            projection_h5, time_s, x_saved, y_saved, native_projected_phi, simvp_projected_phi
        )

    reference = (fine[:, 0], fine[:, 1], fine[:, 2])
    reference_powers = {name: scalar_modal_power(fine[:, index]) for index, name in enumerate(CHANNELS)}
    reference_powers["ey"] = ey_modal_power(fine[:, 2], dy)
    reference_powers["charge_source"] = scalar_modal_power(fine[:, 1] - fine[:, 0])
    reference_transport = modal_transport_components(fine[:, 0], fine[:, 2], dy)
    candidates = {
        "native_interpolation": (baseline[:, 0], baseline[:, 1], baseline[:, 2]),
        "native_density_poisson": (baseline[:, 0], baseline[:, 1], native_projected_phi),
        "simvp": (prediction[:, 0], prediction[:, 1], prediction[:, 2]),
        "simvp_density_poisson": (prediction[:, 0], prediction[:, 1], simvp_projected_phi),
    }
    results = {}
    powers = {}
    transports = {}
    for method, candidate in candidates.items():
        print(f"[evaluate] {method}", flush=True)
        results[method], powers[method], transports[method] = evaluate_method(
            reference, candidate, reference_powers, reference_transport, dx, dy
        )

    native = results["native_interpolation"]
    projected = results["simvp_density_poisson"]
    current = results["simvp"]
    gates = {
        "field": bool(
            projected["field_distribution"]["phi"]["quantile_rmse_over_reference_std"]
            <= native["field_distribution"]["phi"]["quantile_rmse_over_reference_std"]
            and projected["radial_profile"]["phi"]["relative_l2"]
            <= native["radial_profile"]["phi"]["relative_l2"]
        ),
        "mode": bool(
            projected["modal_power"]["phi"]["log10_power_rmse_n1_32"]
            <= native["modal_power"]["phi"]["log10_power_rmse_n1_32"]
            and projected["modal_power"]["ey"]["log10_power_rmse_n1_32"]
            <= native["modal_power"]["ey"]["log10_power_rmse_n1_32"]
        ),
        "transport": bool(
            projected["transport_distribution"]["n9_21"]["quantile_rmse_over_reference_std"]
            <= native["transport_distribution"]["n9_21"]["quantile_rmse_over_reference_std"]
            and projected["transport_distribution"]["n1_32"]["quantile_rmse_over_reference_std"]
            <= native["transport_distribution"]["n1_32"]["quantile_rmse_over_reference_std"]
        ),
        "poisson": bool(
            projected["poisson"]["relative_poisson_residual_median"]
            <= 10.0 * solver_validation["saved_fine_poisson"]["relative_poisson_residual_median"]
        ),
    }
    gates["overall"] = bool(all(gates.values()))
    tradeoff = {
        "poisson_residual_ratio_projected_to_current_simvp": float(
            projected["poisson"]["relative_poisson_residual_median"]
            / max(current["poisson"]["relative_poisson_residual_median"], TINY)
        ),
        "ecdi_transport_distance_ratio_projected_to_current_simvp": float(
            projected["transport_distribution"]["n9_21"]["quantile_rmse_over_reference_std"]
            / max(
                current["transport_distribution"]["n9_21"]["quantile_rmse_over_reference_std"],
                TINY,
            )
        ),
        "total_transport_distance_ratio_projected_to_current_simvp": float(
            projected["transport_distribution"]["n1_32"]["quantile_rmse_over_reference_std"]
            / max(
                current["transport_distribution"]["n1_32"]["quantile_rmse_over_reference_std"],
                TINY,
            )
        ),
        "phi_mode_error_ratio_projected_to_current_simvp": float(
            projected["modal_power"]["phi"]["log10_power_rmse_n1_32"]
            / max(current["modal_power"]["phi"]["log10_power_rmse_n1_32"], TINY)
        ),
    }

    metric_rows = []
    transport_rows = []
    mode_rows = []
    for method in METHOD_LABELS:
        item = results[method]
        metric_rows.append(
            {
                "method": method,
                "phi_quantile_distance": item["field_distribution"]["phi"]["quantile_rmse_over_reference_std"],
                "phi_radial_profile_relative_l2": item["radial_profile"]["phi"]["relative_l2"],
                "phi_log10_power_rmse_n1_32": item["modal_power"]["phi"]["log10_power_rmse_n1_32"],
                "ey_log10_power_rmse_n1_32": item["modal_power"]["ey"]["log10_power_rmse_n1_32"],
                "charge_source_quantile_distance": item["poisson_source_ni_minus_ne"]["distribution"]["quantile_rmse_over_reference_std"],
                "charge_source_radial_profile_relative_l2": item["poisson_source_ni_minus_ne"]["radial_profile"]["relative_l2"],
                "charge_source_log10_power_rmse_n1_32": item["poisson_source_ni_minus_ne"]["modal_power"]["log10_power_rmse_n1_32"],
                "poisson_relative_residual_median": item["poisson"]["relative_poisson_residual_median"],
                "poisson_balance_correlation_median": item["poisson"]["poisson_balance_correlation_median"],
            }
        )
        for band in BANDS:
            transport_rows.append(
                {
                    "method": method,
                    "band": band,
                    **item["transport_distribution"][band],
                }
            )
        for field in ("electron_den", "ion_den", "phi", "ey", "charge_source"):
            for mode in range(1, 65):
                mode_rows.append(
                    {
                        "method": method,
                        "field": field,
                        "mode": mode,
                        "fine_power": float(reference_powers[field][mode]),
                        "candidate_power": float(powers[method][field][mode]),
                        "candidate_to_fine": float(
                            powers[method][field][mode] / max(reference_powers[field][mode], TINY)
                        ),
                    }
                )
    write_csv(args.output_dir / "poisson_projection_method_metrics.csv", metric_rows)
    write_csv(args.output_dir / "poisson_projection_transport_metrics.csv", transport_rows)
    write_csv(args.output_dir / "poisson_projection_modal_power.csv", mode_rows)

    make_spectrum_plot(
        args.output_dir / "poisson_projection_phi_ey_spectra.png", powers, reference_powers
    )
    make_metric_plot(args.output_dir / "poisson_projection_error_ratios.png", results)
    index = len(time_s) // 2
    make_snapshot_plot(
        args.output_dir / "poisson_projection_snapshot.png",
        float(time_s[index] * 1.0e6),
        fine[index, 2],
        baseline[index, 2],
        native_projected_phi[index],
        prediction[index, 2],
        simvp_projected_phi[index],
    )

    summary = {
        "status": "complete",
        "retraining_performed": False,
        "protocol": PROTOCOL,
        "inputs": {
            "reconstruction_h5": str(args.reconstruction_h5.resolve()),
            "coarse_h5": str(args.coarse_h5.resolve()),
            "fine_h5": str(args.fine_h5.resolve()),
            "time_us": [float(time_s[0] * 1.0e6), float(time_s[-1] * 1.0e6)],
            "frames": int(len(time_s)),
            "dx_m": dx,
            "dy_m": dy,
        },
        "solver_validation": solver_validation,
        "own_grid_poisson_diagnostics": own_grid_physics,
        "methods": results,
        "primary_gates": gates,
        "tradeoff_vs_current_simvp": tradeoff,
        "projection_h5": str(projection_h5.resolve()),
    }
    summary_path = args.output_dir / "poisson_projection_summary.json"
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    readme = f"""# E20/B20 native-G2 Poisson projection

This post-processing experiment used the existing trained SimVP checkpoint and
existing native-G2 reconstruction; no model retraining was performed.

Primary gates for `SimVP density + Poisson` versus native interpolation:

- field: `{gates['field']}`
- mode: `{gates['mode']}`
- transport: `{gates['transport']}`
- Poisson: `{gates['poisson']}`
- overall: `{gates['overall']}`

The fine and native-G2 PIC runs are independent particle realizations.  The
fine-density projection is solver validation only.  See
`poisson_projection_summary.json` for the complete metrics and fixed protocol.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"primary_gates": gates, "tradeoff": tradeoff}, indent=2), flush=True)
    print(f"[saved] {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
