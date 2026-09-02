"""Measure information loss from synthetic spatial grid coarsening.

Spatial reductions are named by grid-coarsening factor, never by ``stride``:
G2 means 2x2 block averaging, Gr2 means radial-only factor two, and Gtheta2
means azimuth-only factor two.  The experiment uses no trained model and no
new PIC trajectory; it measures an optimistic recoverability ceiling from an
existing fine PIC trajectory.
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


ROOT = Path(__file__).resolve().parent
CASE = "2D_RadAz_Xe1p_Bx20mT_Ez25kVm_dt15ps_out15ns"
DEFAULT_H5 = (
    ROOT.parent
    / "PEPAPIC"
    / "test"
    / "results"
    / "2D_Landmark"
    / CASE
    / CASE
    / "analysis_fields_uncompressed.h5"
)
DEFAULT_OUTPUT = ROOT / "workdirs" / "analyze_radaz_e25_grid_coarsening_information_loss"
FIELDS = ("electron_den", "ion_den", "phi", "efx", "efy")
CONFIGURATIONS = (
    ("G2", 2, 2),
    ("G4", 4, 4),
    ("G8", 8, 8),
    ("Gr2", 2, 1),
    ("Gr4", 4, 1),
    ("Gr8", 8, 1),
    ("Gtheta2", 1, 2),
    ("Gtheta4", 1, 4),
    ("Gtheta8", 1, 8),
)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


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


@dataclass
class ScalarAccumulator:
    count: int = 0
    truth_sum: float = 0.0
    reconstruction_sum: float = 0.0
    truth_square_sum: float = 0.0
    reconstruction_square_sum: float = 0.0
    cross_sum: float = 0.0
    error_square_sum: float = 0.0

    def update(self, truth: np.ndarray, reconstruction: np.ndarray) -> None:
        truth64 = np.asarray(truth, dtype=np.float64)
        reconstruction64 = np.asarray(reconstruction, dtype=np.float64)
        difference = reconstruction64 - truth64
        self.count += int(truth64.size)
        self.truth_sum += float(np.sum(truth64))
        self.reconstruction_sum += float(np.sum(reconstruction64))
        self.truth_square_sum += float(np.sum(truth64 * truth64))
        self.reconstruction_square_sum += float(
            np.sum(reconstruction64 * reconstruction64)
        )
        self.cross_sum += float(np.sum(truth64 * reconstruction64))
        self.error_square_sum += float(np.sum(difference * difference))

    def metrics(self) -> dict[str, float]:
        tiny = np.finfo(float).tiny
        truth_mean = self.truth_sum / self.count
        reconstruction_mean = self.reconstruction_sum / self.count
        truth_variance = max(
            self.truth_square_sum / self.count - truth_mean * truth_mean, 0.0
        )
        reconstruction_variance = max(
            self.reconstruction_square_sum / self.count
            - reconstruction_mean * reconstruction_mean,
            0.0,
        )
        covariance = (
            self.cross_sum / self.count - truth_mean * reconstruction_mean
        )
        rmse = math.sqrt(self.error_square_sum / self.count)
        return {
            "rmse": rmse,
            "nrmse_std": rmse / max(math.sqrt(truth_variance), tiny),
            "relative_l2": math.sqrt(
                self.error_square_sum / max(self.truth_square_sum, tiny)
            ),
            "correlation": covariance
            / max(math.sqrt(truth_variance * reconstruction_variance), tiny),
            "rms_ratio": math.sqrt(
                self.reconstruction_square_sum / max(self.truth_square_sum, tiny)
            ),
            "mean_bias_over_truth_rms": (reconstruction_mean - truth_mean)
            / max(math.sqrt(self.truth_square_sum / self.count), tiny),
        }


@dataclass
class ModeAccumulator:
    truth_power: np.ndarray
    reconstruction_power: np.ndarray
    error_power: np.ndarray
    cross: np.ndarray

    @classmethod
    def create(cls, modes: int) -> "ModeAccumulator":
        zeros = np.zeros(modes, dtype=np.float64)
        return cls(
            truth_power=zeros.copy(),
            reconstruction_power=zeros.copy(),
            error_power=zeros.copy(),
            cross=np.zeros(modes, dtype=np.complex128),
        )

    def update(self, truth: np.ndarray, reconstruction: np.ndarray) -> None:
        difference = reconstruction - truth
        axes = tuple(range(truth.ndim - 1))
        self.truth_power += np.sum(np.abs(truth) ** 2, axis=axes)
        self.reconstruction_power += np.sum(
            np.abs(reconstruction) ** 2, axis=axes
        )
        self.error_power += np.sum(np.abs(difference) ** 2, axis=axes)
        self.cross += np.sum(reconstruction * np.conj(truth), axis=axes)

    def rows(self, label: str, field: str) -> list[dict]:
        rows = []
        tiny = np.finfo(float).tiny
        for mode in range(len(self.truth_power)):
            denominator = max(
                math.sqrt(
                    self.truth_power[mode] * self.reconstruction_power[mode]
                ),
                tiny,
            )
            power_ratio = self.reconstruction_power[mode] / max(
                self.truth_power[mode], tiny
            )
            coherence = min(abs(self.cross[mode]) / denominator, 1.0)
            rows.append(
                {
                    "configuration": label,
                    "field": field,
                    "mode": mode,
                    "truth_power": self.truth_power[mode],
                    "reconstruction_power": self.reconstruction_power[mode],
                    "power_ratio": power_ratio,
                    "amplitude_ratio": math.sqrt(power_ratio),
                    "coherent_power_ratio": power_ratio * coherence * coherence,
                    "coherent_amplitude_ratio": math.sqrt(power_ratio) * coherence,
                    "relative_complex_l2": math.sqrt(
                        self.error_power[mode]
                        / max(self.truth_power[mode], tiny)
                    ),
                    "complex_coherence": coherence,
                    "mean_phase_bias_rad": float(np.angle(self.cross[mode])),
                }
            )
        return rows


def coarsen(values: np.ndarray, radial_factor: int, azimuth_factor: int) -> np.ndarray:
    frames, radial, azimuth = values.shape
    if radial % radial_factor or azimuth % azimuth_factor:
        raise ValueError("Grid is not divisible by the coarsening factors")
    return values.reshape(
        frames,
        radial // radial_factor,
        radial_factor,
        azimuth // azimuth_factor,
        azimuth_factor,
    ).mean(axis=(2, 4))


def nonperiodic_linear_weights(size: int, factor: int) -> np.ndarray:
    coarse_size = size // factor
    centers = np.arange(coarse_size, dtype=np.float64) * factor + (factor - 1) / 2
    positions = np.arange(size, dtype=np.float64)
    weights = np.zeros((size, coarse_size), dtype=np.float64)
    if coarse_size == 1:
        weights[:, 0] = 1.0
        return weights
    right = np.searchsorted(centers, positions, side="right")
    right = np.clip(right, 1, coarse_size - 1)
    left = right - 1
    alpha = (positions - centers[left]) / (centers[right] - centers[left])
    below = positions <= centers[0]
    above = positions >= centers[-1]
    alpha = np.clip(alpha, 0.0, 1.0)
    weights[np.arange(size), left] = 1.0 - alpha
    weights[np.arange(size), right] += alpha
    weights[below] = 0.0
    weights[below, 0] = 1.0
    weights[above] = 0.0
    weights[above, -1] = 1.0
    return weights


def periodic_linear_weights(size: int, factor: int) -> np.ndarray:
    coarse_size = size // factor
    positions = np.arange(size, dtype=np.float64)
    first_center = (factor - 1) / 2
    coordinate = (positions - first_center) / factor
    left_unwrapped = np.floor(coordinate).astype(np.int64)
    alpha = coordinate - left_unwrapped
    left = np.mod(left_unwrapped, coarse_size)
    right = np.mod(left_unwrapped + 1, coarse_size)
    weights = np.zeros((size, coarse_size), dtype=np.float64)
    weights[np.arange(size), left] += 1.0 - alpha
    weights[np.arange(size), right] += alpha
    return weights


def reconstruct(
    coarse: np.ndarray,
    radial_weights: np.ndarray,
    azimuth_weights: np.ndarray,
) -> np.ndarray:
    # Linear interpolation has at most two nonzero weights per fine point.
    # Applying the dense matrices is needlessly O(N^3), so extract their
    # sparse two-point stencils and apply only those neighbours.
    radial_indices = np.argpartition(radial_weights, -2, axis=1)[:, -2:]
    radial_coefficients = np.take_along_axis(
        radial_weights, radial_indices, axis=1
    )
    radial_interpolated = (
        coarse[:, radial_indices[:, 0], :]
        * radial_coefficients[None, :, 0, None]
        + coarse[:, radial_indices[:, 1], :]
        * radial_coefficients[None, :, 1, None]
    )
    azimuth_indices = np.argpartition(azimuth_weights, -2, axis=1)[:, -2:]
    azimuth_coefficients = np.take_along_axis(
        azimuth_weights, azimuth_indices, axis=1
    )
    return (
        radial_interpolated[:, :, azimuth_indices[:, 0]]
        * azimuth_coefficients[None, None, :, 0]
        + radial_interpolated[:, :, azimuth_indices[:, 1]]
        * azimuth_coefficients[None, None, :, 1]
    ).astype(np.float32, copy=False)


def field_fourier(values: np.ndarray, maximum_mode: int) -> np.ndarray:
    return np.fft.rfft(values, axis=-1, norm="forward")[..., : maximum_mode + 1]


def configuration_kind(radial_factor: int, azimuth_factor: int) -> str:
    if radial_factor == azimuth_factor:
        return "isotropic"
    if radial_factor > 1:
        return "radial_only"
    return "azimuth_only"


def plot_field_metrics(path: Path, rows: list[dict]) -> None:
    configurations = [item[0] for item in CONFIGURATIONS]
    matrix = np.asarray(
        [
            [
                next(
                    row["nrmse_std"]
                    for row in rows
                    if row["configuration"] == configuration
                    and row["field"] == field
                )
                for field in FIELDS
            ]
            for configuration in configurations
        ]
    )
    fig, axis = plt.subplots(figsize=(8.0, 5.5))
    image = axis.imshow(matrix, aspect="auto", cmap="magma")
    axis.set_xticks(np.arange(len(FIELDS)), FIELDS, rotation=25, ha="right")
    axis.set_yticks(np.arange(len(configurations)), configurations)
    axis.set_title("Interpolation information loss: NRMSE/std")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.3f}",
                color="white" if matrix[row, column] > 0.35 else "black",
                ha="center",
                va="center",
                fontsize=8,
            )
    fig.colorbar(image, ax=axis, label="NRMSE / truth std")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_phi_modes(path: Path, rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), sharex=True)
    styles = {
        "isotropic": "-",
        "radial_only": "--",
        "azimuth_only": ":",
    }
    for label, radial_factor, azimuth_factor in CONFIGURATIONS:
        selected = [
            row
            for row in rows
            if row["configuration"] == label
            and row["field"] == "phi"
            and 1 <= row["mode"] <= 21
        ]
        modes = np.asarray([row["mode"] for row in selected])
        amplitude = np.asarray(
            [row["coherent_amplitude_ratio"] for row in selected]
        )
        coherence = np.asarray([row["complex_coherence"] for row in selected])
        style = styles[configuration_kind(radial_factor, azimuth_factor)]
        axes[0].plot(modes, amplitude, style, marker=".", label=label)
        axes[1].plot(modes, coherence, style, marker=".", label=label)
    axes[0].axhline(1.0, color="black", lw=0.8)
    axes[0].set_ylabel("phi coherent amplitude retention")
    axes[1].set_ylabel("phi complex coherence")
    for axis in axes:
        axis.set_xlabel("azimuthal mode n")
        axis.set_xticks([1, 2, 4, 7, 10, 14, 18, 21])
        axis.grid(alpha=0.25)
    axes[1].legend(ncol=3, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_summary(path: Path, rows: list[dict]) -> None:
    labels = [row["configuration"] for row in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5))
    quantities = (
        ("phi_n2_amplitude_ratio", "phi n=2 amplitude retention", 1.0),
        ("phi_n7_amplitude_ratio", "phi n=7 amplitude retention", 1.0),
        (
            "phi_ecdi_n9_21_coherent_power_ratio",
            "phi coherent ECDI n=9--21 power retention",
            1.0,
        ),
        (
            "field_gradient_residual_ratio_reconstruction_to_truth",
            "field-gradient residual / truth residual",
            1.0,
        ),
    )
    for axis, (key, title, reference) in zip(axes.flat, quantities):
        values = [row[key] for row in rows]
        colors = [
            {"isotropic": "tab:purple", "radial_only": "tab:blue", "azimuth_only": "tab:orange"}[
                row["kind"]
            ]
            for row in rows
        ]
        axis.bar(x, values, color=colors)
        axis.axhline(reference, color="black", lw=0.8)
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_snapshot(
    path: Path,
    truth: np.ndarray,
    reconstructions: dict[str, np.ndarray],
    time_us: float,
) -> None:
    labels = ("G2", "G4", "G8")
    low = float(np.quantile(truth, 0.01))
    high = float(np.quantile(truth, 0.99))
    scale = max(
        max(
            float(np.quantile(np.abs(reconstructions[label] - truth), 0.99))
            for label in labels
        ),
        1.0e-30,
    )
    fig, axes = plt.subplots(len(labels), 3, figsize=(11.0, 9.0))
    for row, label in enumerate(labels):
        reconstruction = reconstructions[label]
        error = reconstruction - truth
        axes[row, 0].imshow(truth, origin="lower", aspect="auto", vmin=low, vmax=high)
        axes[row, 1].imshow(
            reconstruction, origin="lower", aspect="auto", vmin=low, vmax=high
        )
        image = axes[row, 2].imshow(
            error,
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
            vmin=-scale,
            vmax=scale,
        )
        axes[row, 0].set_ylabel(label)
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    axes[0, 0].set_title(f"fine phi, t={time_us:.3f} us")
    axes[0, 1].set_title("interpolated coarse")
    axes[0, 2].set_title("error")
    fig.colorbar(image, ax=axes[:, 2], shrink=0.75)
    fig.subplots_adjust(left=0.06, right=0.91, bottom=0.05, top=0.93, wspace=0.08, hspace=0.12)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-us", type=float, default=20.0)
    parser.add_argument("--end-us", type=float, default=30.0)
    parser.add_argument("--chunk-frames", type=int, default=16)
    parser.add_argument("--maximum-mode", type=int, default=21)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scalar = {
        label: {field: ScalarAccumulator() for field in FIELDS}
        for label, _, _ in CONFIGURATIONS
    }
    modes = {
        label: {
            field: ModeAccumulator.create(args.maximum_mode + 1) for field in FIELDS
        }
        for label, _, _ in CONFIGURATIONS
    }
    radial_truth = {field: np.zeros(256, dtype=np.float64) for field in FIELDS}
    radial_reconstruction = {
        label: {field: np.zeros(256, dtype=np.float64) for field in FIELDS}
        for label, _, _ in CONFIGURATIONS
    }
    physics = {
        label: {
            "truth_residual_power": 0.0,
            "reconstruction_residual_power": 0.0,
            "truth_ey_power": 0.0,
            "reconstruction_ey_power": 0.0,
        }
        for label, _, _ in CONFIGURATIONS
    }
    weight_cache = {
        (radial_factor, azimuth_factor): (
            nonperiodic_linear_weights(256, radial_factor),
            periodic_linear_weights(256, azimuth_factor),
        )
        for _, radial_factor, azimuth_factor in CONFIGURATIONS
    }
    snapshot_truth = None
    snapshot_reconstructions: dict[str, np.ndarray] = {}
    snapshot_time_us = None
    processed_frames = 0

    with h5py.File(args.h5, "r") as handle:
        if handle.attrs.get("axis_order", "") != "time,x,y":
            raise ValueError("Expected consolidated time,x,y field layout")
        times_us = np.asarray(handle["axes/time_s"], dtype=np.float64) * 1.0e6
        indices = np.flatnonzero(
            (times_us >= args.start_us - 1.0e-10)
            & (times_us <= args.end_us + 1.0e-10)
        )
        if len(indices) < 3 or not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            raise ValueError("Selected analysis frames must be a nonempty contiguous interval")
        period_m = float(handle["axes/y_m"][-1] - handle["axes/y_m"][0])
        wave_numbers = (
            2.0 * np.pi * np.arange(args.maximum_mode + 1, dtype=np.float64) / period_m
        )
        snapshot_index = int(indices[len(indices) // 2])

        for start in range(int(indices[0]), int(indices[-1]) + 1, args.chunk_frames):
            stop = min(start + args.chunk_frames, int(indices[-1]) + 1)
            truth_fields = {
                field: np.asarray(handle[f"fields/{field}"][start:stop, :256, :256], dtype=np.float32)
                for field in FIELDS
            }
            for field, truth in truth_fields.items():
                radial_truth[field] += np.sum(truth, axis=(0, 2), dtype=np.float64)
            for label, radial_factor, azimuth_factor in CONFIGURATIONS:
                radial_weights, azimuth_weights = weight_cache[
                    (radial_factor, azimuth_factor)
                ]
                reconstructions = {}
                fourier_truth = {}
                fourier_reconstruction = {}
                for field, truth in truth_fields.items():
                    coarse = coarsen(truth, radial_factor, azimuth_factor)
                    reconstruction = reconstruct(
                        coarse, radial_weights, azimuth_weights
                    )
                    reconstructions[field] = reconstruction
                    scalar[label][field].update(truth, reconstruction)
                    radial_reconstruction[label][field] += np.sum(
                        reconstruction, axis=(0, 2), dtype=np.float64
                    )
                    truth_fourier = field_fourier(truth, args.maximum_mode)
                    reconstruction_fourier = field_fourier(
                        reconstruction, args.maximum_mode
                    )
                    fourier_truth[field] = truth_fourier
                    fourier_reconstruction[field] = reconstruction_fourier
                    modes[label][field].update(
                        truth_fourier, reconstruction_fourier
                    )
                truth_residual = (
                    fourier_truth["efy"]
                    + 1j * wave_numbers[None, None, :] * fourier_truth["phi"]
                )[..., 1:]
                reconstruction_residual = (
                    fourier_reconstruction["efy"]
                    + 1j
                    * wave_numbers[None, None, :]
                    * fourier_reconstruction["phi"]
                )[..., 1:]
                physics[label]["truth_residual_power"] += float(
                    np.sum(np.abs(truth_residual) ** 2)
                )
                physics[label]["reconstruction_residual_power"] += float(
                    np.sum(np.abs(reconstruction_residual) ** 2)
                )
                physics[label]["truth_ey_power"] += float(
                    np.sum(np.abs(fourier_truth["efy"][..., 1:]) ** 2)
                )
                physics[label]["reconstruction_ey_power"] += float(
                    np.sum(np.abs(fourier_reconstruction["efy"][..., 1:]) ** 2)
                )
                if start <= snapshot_index < stop and label in ("G2", "G4", "G8"):
                    local = snapshot_index - start
                    snapshot_truth = truth_fields["phi"][local].copy()
                    snapshot_reconstructions[label] = reconstructions["phi"][local].copy()
                    snapshot_time_us = float(times_us[snapshot_index])
            processed_frames += stop - start
            print(
                f"[PROGRESS] {processed_frames}/{len(indices)} frames",
                flush=True,
            )

    field_rows = []
    mode_rows = []
    profile_rows = []
    summary_rows = []
    samples_per_radius = processed_frames * 256
    for label, radial_factor, azimuth_factor in CONFIGURATIONS:
        kind = configuration_kind(radial_factor, azimuth_factor)
        for field in FIELDS:
            field_rows.append(
                {
                    "configuration": label,
                    "kind": kind,
                    "radial_factor": radial_factor,
                    "azimuth_factor": azimuth_factor,
                    "coarse_radial_cells": 256 // radial_factor,
                    "coarse_azimuth_cells": 256 // azimuth_factor,
                    "field": field,
                    **scalar[label][field].metrics(),
                }
            )
            mode_rows.extend(modes[label][field].rows(label, field))
            truth_profile = radial_truth[field] / samples_per_radius
            reconstruction_profile = (
                radial_reconstruction[label][field] / samples_per_radius
            )
            profile_rows.append(
                {
                    "configuration": label,
                    "field": field,
                    "relative_profile_l2": float(
                        np.linalg.norm(reconstruction_profile - truth_profile)
                        / max(np.linalg.norm(truth_profile), np.finfo(float).tiny)
                    ),
                    "profile_correlation": float(
                        np.corrcoef(truth_profile, reconstruction_profile)[0, 1]
                    ),
                }
            )
        phi_mode = modes[label]["phi"]
        truth_ecdi = float(np.sum(phi_mode.truth_power[9:22]))
        reconstruction_ecdi = float(
            np.sum(phi_mode.reconstruction_power[9:22])
        )
        ecdi_denominator = np.sqrt(
            phi_mode.truth_power[9:22]
            * phi_mode.reconstruction_power[9:22]
        )
        ecdi_coherence = np.minimum(
            np.abs(phi_mode.cross[9:22])
            / np.maximum(ecdi_denominator, np.finfo(float).tiny),
            1.0,
        )
        coherent_reconstruction_ecdi = float(
            np.sum(
                phi_mode.reconstruction_power[9:22]
                * ecdi_coherence
                * ecdi_coherence
            )
        )
        truth_residual_over_ey = math.sqrt(
            physics[label]["truth_residual_power"]
            / max(physics[label]["truth_ey_power"], np.finfo(float).tiny)
        )
        reconstruction_residual_over_ey = math.sqrt(
            physics[label]["reconstruction_residual_power"]
            / max(
                physics[label]["reconstruction_ey_power"],
                np.finfo(float).tiny,
            )
        )
        summary_rows.append(
            {
                "configuration": label,
                "kind": kind,
                "radial_factor": radial_factor,
                "azimuth_factor": azimuth_factor,
                "coarse_radial_cells": 256 // radial_factor,
                "coarse_azimuth_cells": 256 // azimuth_factor,
                "coarse_azimuth_nyquist_mode": (256 // azimuth_factor) // 2,
                "cell_fraction": 1.0 / (radial_factor * azimuth_factor),
                "phi_n2_amplitude_ratio": math.sqrt(
                    phi_mode.reconstruction_power[2]
                    / max(phi_mode.truth_power[2], np.finfo(float).tiny)
                ),
                "phi_n2_complex_coherence": min(
                    abs(phi_mode.cross[2])
                    / max(
                        math.sqrt(
                            phi_mode.truth_power[2]
                            * phi_mode.reconstruction_power[2]
                        ),
                        np.finfo(float).tiny,
                    ),
                    1.0,
                ),
                "phi_n7_amplitude_ratio": math.sqrt(
                    phi_mode.reconstruction_power[7]
                    / max(phi_mode.truth_power[7], np.finfo(float).tiny)
                ),
                "phi_n7_complex_coherence": min(
                    abs(phi_mode.cross[7])
                    / max(
                        math.sqrt(
                            phi_mode.truth_power[7]
                            * phi_mode.reconstruction_power[7]
                        ),
                        np.finfo(float).tiny,
                    ),
                    1.0,
                ),
                "phi_ecdi_n9_21_power_ratio": reconstruction_ecdi
                / max(truth_ecdi, np.finfo(float).tiny),
                "phi_ecdi_n9_21_coherent_power_ratio": coherent_reconstruction_ecdi
                / max(truth_ecdi, np.finfo(float).tiny),
                "phi_ecdi_n9_21_relative_complex_l2": math.sqrt(
                    float(np.sum(phi_mode.error_power[9:22]))
                    / max(truth_ecdi, np.finfo(float).tiny)
                ),
                "truth_field_gradient_residual_over_ey": truth_residual_over_ey,
                "reconstruction_field_gradient_residual_over_ey": reconstruction_residual_over_ey,
                "field_gradient_residual_ratio_reconstruction_to_truth": reconstruction_residual_over_ey
                / max(truth_residual_over_ey, np.finfo(float).tiny),
            }
        )

    write_csv(args.output / "field_information_loss.csv", field_rows)
    write_csv(args.output / "mode_information_loss.csv", mode_rows)
    write_csv(args.output / "radial_profile_information_loss.csv", profile_rows)
    write_csv(args.output / "configuration_summary.csv", summary_rows)
    plot_field_metrics(args.output / "field_nrmse_heatmap.png", field_rows)
    plot_phi_modes(args.output / "phi_mode_retention.png", mode_rows)
    plot_summary(args.output / "configuration_summary.png", summary_rows)
    if snapshot_truth is not None and len(snapshot_reconstructions) == 3:
        save_snapshot(
            args.output / "phi_reconstruction_snapshot.png",
            snapshot_truth,
            snapshot_reconstructions,
            float(snapshot_time_us),
        )
    result = {
        "status": "PASS",
        "experiment": "synthetic_grid_coarsening_information_loss_no_training",
        "new_pic_run_used": False,
        "native_coarse_claim": False,
        "spatial_naming": {
            "G2": "2x2 cell-block mean",
            "Gr2": "radial-only factor two",
            "Gtheta2": "azimuth-only factor two",
            "temporal_stride_reserved_for": "input/output frame spacing only",
        },
        "source_h5": str(args.h5.resolve()),
        "time_us": [args.start_us, args.end_us],
        "frames": processed_frames,
        "fine_unique_grid": [256, 256],
        "fields": list(FIELDS),
        "maximum_mode": args.maximum_mode,
        "interpolation": {
            "radial": "linear on block centers with constant edge extension",
            "azimuth": "periodic linear on block centers",
        },
        "configurations": summary_rows,
        "limitations": [
            "The coarse inputs are projections of fine PIC, not native coarse-solver trajectories.",
            "No low-PPC dynamics, coarse-grid numerical dispersion, or coarse-dt dynamics are represented.",
            "The results are an optimistic upper bound for later multi-fidelity reconstruction.",
        ],
    }
    (args.output / "information_loss_summary.json").write_text(
        json.dumps(json_safe(result), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(json_safe(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
