#!/usr/bin/env python3
"""Evaluate whether the E25 G2 residual model reconstructs saturated instability.

The analysis is restricted to the held-out final 10% of the 20--30 us
interval.  Overlapping synchronous SimVP outputs are averaged per physical
frame before field, modal, temporal-frequency, and physics diagnostics are
computed.  G denotes grid-coarsening factor, not temporal frame stride.
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
import torch

from openstl.models.simvp_model import SimVP_Model
from train_radaz_g2_residual_superresolution import (
    DEFAULT_H5,
    make_coarse_size_interpolated,
    make_grid_interpolated,
    segment_starts,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_WORKDIR = (
    ROOT / "workdirs" / "radaz_e25_g2_simvp_residual_sr_sync10_20to30us"
)
CHANNELS = ("electron_den", "ion_den", "phi")
DISPLAY_NAMES = (r"$n_e$", r"$n_i$", r"$\phi$")
EPS0 = 8.8541878128e-12
E_CHARGE = 1.602176634e-19
TINY = np.finfo(np.float64).tiny


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start-us", type=float, default=20.0)
    parser.add_argument("--end-us", type=float, default=30.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


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


def centered_correlation(a: np.ndarray, b: np.ndarray) -> float:
    left = np.array(a, dtype=np.float64, copy=True).ravel()
    right = np.array(b, dtype=np.float64, copy=True).ravel()
    left -= np.mean(left)
    right -= np.mean(right)
    denominator = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
    return float(np.dot(left, right) / max(denominator, TINY))


def field_metrics(truth: np.ndarray, reconstruction: np.ndarray) -> dict[str, float]:
    difference = np.asarray(reconstruction, dtype=np.float64) - np.asarray(
        truth, dtype=np.float64
    )
    truth64 = np.asarray(truth, dtype=np.float64)
    mse = float(np.mean(difference * difference))
    return {
        "mse": mse,
        "nrmse_std": math.sqrt(mse / max(float(np.var(truth64)), TINY)),
        "relative_l2": math.sqrt(
            float(np.sum(difference * difference))
            / max(float(np.sum(truth64 * truth64)), TINY)
        ),
        "correlation": centered_correlation(truth64, reconstruction),
    }


def predict_unique_frames(
    fine: np.ndarray,
    baseline: np.ndarray,
    starts: np.ndarray,
    checkpoint: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    model = SimVP_Model(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    sequence_length = int(checkpoint["model_kwargs"]["in_shape"][0])
    scale = torch.as_tensor(
        checkpoint["residual_rms"], device=device, dtype=torch.float32
    ).view(1, 1, 3, 1, 1)
    first = int(starts[0])
    final = int(starts[-1]) + sequence_length
    sums = np.zeros((final - first, 3, 256, 256), dtype=np.float64)
    square_sums = np.zeros_like(sums)
    counts = np.zeros(final - first, dtype=np.int64)
    with torch.inference_mode():
        for start in starts:
            input_tensor = torch.from_numpy(
                baseline[int(start) : int(start) + sequence_length]
            ).unsqueeze(0).to(device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                output = input_tensor + model(input_tensor) * scale
            output_np = output[0, :, :, :256].float().cpu().numpy()
            local = int(start) - first
            sums[local : local + sequence_length] += output_np
            square_sums[local : local + sequence_length] += output_np**2
            counts[local : local + sequence_length] += 1
    covered = counts > 0
    prediction = (sums[covered] / counts[covered, None, None, None]).astype(
        np.float32
    )
    overlap_variance = np.maximum(
        square_sums[covered] / counts[covered, None, None, None]
        - prediction.astype(np.float64) ** 2,
        0.0,
    )
    indices = np.arange(first, final, dtype=np.int64)[covered]
    consistency = {}
    for channel, name in enumerate(CHANNELS):
        consistency[name] = {
            "overlap_prediction_std": float(
                np.sqrt(np.mean(overlap_variance[:, channel]))
            ),
            "overlap_prediction_std_over_truth_std": float(
                np.sqrt(np.mean(overlap_variance[:, channel]))
                / max(float(np.std(fine[indices, channel, :256])), TINY)
            ),
            "minimum_predictions_per_frame": int(np.min(counts[covered])),
            "maximum_predictions_per_frame": int(np.max(counts[covered])),
        }
    return (
        prediction,
        fine[indices, :, :256].copy(),
        baseline[indices, :, :256].copy(),
        indices,
        consistency,
    )


def complex_mode_metrics(
    truth: np.ndarray, candidate: np.ndarray
) -> dict[str, float]:
    truth_power = float(np.vdot(truth, truth).real)
    candidate_power = float(np.vdot(candidate, candidate).real)
    cross = np.vdot(candidate, truth)
    return {
        "amplitude_ratio": math.sqrt(candidate_power / max(truth_power, TINY)),
        "coherence": float(
            abs(cross) / max(math.sqrt(candidate_power * truth_power), TINY)
        ),
        "relative_error": math.sqrt(
            float(np.vdot(candidate - truth, candidate - truth).real)
            / max(truth_power, TINY)
        ),
    }


def band_metrics(
    truth_power: np.ndarray, candidate_power: np.ndarray, modes: slice
) -> dict[str, float]:
    truth_series = np.sum(truth_power[:, modes], axis=1)
    candidate_series = np.sum(candidate_power[:, modes], axis=1)
    return {
        "mean_power_ratio": float(
            np.mean(candidate_series) / max(float(np.mean(truth_series)), TINY)
        ),
        "time_correlation": centered_correlation(truth_series, candidate_series),
        "time_series_relative_l2": math.sqrt(
            float(np.sum((candidate_series - truth_series) ** 2))
            / max(float(np.sum(truth_series**2)), TINY)
        ),
    }


def phase_frequency(
    coefficients: np.ndarray, time_s: np.ndarray
) -> tuple[float, float]:
    phase = np.unwrap(np.angle(coefficients))
    centered_time = time_s - np.mean(time_s)
    design = np.column_stack([centered_time, np.ones_like(centered_time)])
    slope, offset = np.linalg.lstsq(design, phase, rcond=None)[0]
    fitted = slope * centered_time + offset
    residual = phase - fitted
    total = float(np.sum((phase - np.mean(phase)) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / max(total, TINY)
    return float(slope / (2.0 * np.pi)), r_squared


def modal_analysis(
    truth: np.ndarray,
    baseline: np.ndarray,
    prediction: np.ndarray,
    time_s: np.ndarray,
) -> tuple[list[dict], dict, dict, dict]:
    transforms = {
        "truth": np.fft.rfft(truth, axis=-1, norm="forward")[..., :33],
        "baseline": np.fft.rfft(baseline, axis=-1, norm="forward")[..., :33],
        "model": np.fft.rfft(prediction, axis=-1, norm="forward")[..., :33],
    }
    powers = {
        source: np.mean(np.abs(values) ** 2, axis=2)
        for source, values in transforms.items()
    }
    rows: list[dict] = []
    stability: dict = {}
    frequency: dict = {}
    bands = {
        "mtsi_candidate_n1_6": slice(1, 7),
        "transition_n7_8": slice(7, 9),
        "ecdi_candidate_n9_21": slice(9, 22),
    }
    for channel, name in enumerate(CHANNELS):
        for mode in range(1, 33):
            truth_mode = transforms["truth"][:, channel, :, mode]
            row = {"field": name, "mode": mode}
            truth_amplitude = np.sqrt(powers["truth"][:, channel, mode])
            for source in ("baseline", "model"):
                metrics = complex_mode_metrics(
                    truth_mode, transforms[source][:, channel, :, mode]
                )
                row.update({f"{source}_{key}": value for key, value in metrics.items()})
                candidate_amplitude = np.sqrt(powers[source][:, channel, mode])
                row[f"{source}_amplitude_time_correlation"] = centered_correlation(
                    truth_amplitude, candidate_amplitude
                )
            rows.append(row)

        truth_mean_power = np.mean(powers["truth"][:, channel, 1:22], axis=0)
        baseline_mean_power = np.mean(powers["baseline"][:, channel, 1:22], axis=0)
        model_mean_power = np.mean(powers["model"][:, channel, 1:22], axis=0)
        truth_dominant = int(np.argmax(truth_mean_power) + 1)
        baseline_dominant = int(np.argmax(baseline_mean_power) + 1)
        model_dominant = int(np.argmax(model_mean_power) + 1)
        truth_dominant_time = np.argmax(powers["truth"][:, channel, 1:22], axis=1) + 1
        model_dominant_time = np.argmax(powers["model"][:, channel, 1:22], axis=1) + 1
        baseline_dominant_time = np.argmax(powers["baseline"][:, channel, 1:22], axis=1) + 1
        item = {
            "truth_dominant_mode_mean_spectrum": truth_dominant,
            "baseline_dominant_mode_mean_spectrum": baseline_dominant,
            "model_dominant_mode_mean_spectrum": model_dominant,
            "model_dominant_mode_time_agreement": float(
                np.mean(model_dominant_time == truth_dominant_time)
            ),
            "baseline_dominant_mode_time_agreement": float(
                np.mean(baseline_dominant_time == truth_dominant_time)
            ),
            "bands": {},
        }
        for band_name, mode_slice in bands.items():
            item["bands"][band_name] = {
                source: band_metrics(
                    powers["truth"][:, channel], powers[source][:, channel], mode_slice
                )
                for source in ("baseline", "model")
            }
        truth_ecdi = np.sum(
            powers["truth"][:, channel, 9:22], axis=1, dtype=np.float64
        )
        truth_mtsi = np.sum(
            powers["truth"][:, channel, 1:7], axis=1, dtype=np.float64
        )
        truth_floor = max(float(np.mean(truth_mtsi)) * 1.0e-12, TINY)
        truth_ratio = truth_ecdi / np.maximum(truth_mtsi, truth_floor)
        truth_ratio_of_means = float(np.mean(truth_ecdi)) / max(
            float(np.mean(truth_mtsi)), TINY
        )
        item["ecdi_n9_21_over_mtsi_n1_6"] = {}
        for source in ("baseline", "model"):
            source_ecdi = np.sum(
                powers[source][:, channel, 9:22], axis=1, dtype=np.float64
            )
            source_mtsi = np.sum(
                powers[source][:, channel, 1:7], axis=1, dtype=np.float64
            )
            source_floor = max(float(np.mean(source_mtsi)) * 1.0e-12, TINY)
            source_ratio = source_ecdi / np.maximum(source_mtsi, source_floor)
            source_ratio_of_means = float(np.mean(source_ecdi)) / max(
                float(np.mean(source_mtsi)), TINY
            )
            item["ecdi_n9_21_over_mtsi_n1_6"][source] = {
                "ratio_of_mean_powers_to_truth": source_ratio_of_means
                / max(truth_ratio_of_means, TINY),
                "time_correlation": centered_correlation(truth_ratio, source_ratio),
            }
        stability[name] = item

        frequency[name] = {}
        for mode in (2, 7, truth_dominant):
            mode_key = f"n{mode}"
            if mode_key in frequency[name]:
                continue
            radial_power = np.mean(
                np.abs(transforms["truth"][:, channel, :, mode]) ** 2, axis=0
            )
            radial_index = int(np.argmax(radial_power))
            truth_series = transforms["truth"][:, channel, radial_index, mode]
            truth_frequency, truth_r2 = phase_frequency(truth_series, time_s)
            frequency[name][mode_key] = {
                "radial_index": radial_index,
                "truth_phase_frequency_hz": truth_frequency,
                "truth_phase_fit_r_squared": truth_r2,
            }
            for source in ("baseline", "model"):
                series = transforms[source][:, channel, radial_index, mode]
                source_frequency, source_r2 = phase_frequency(series, time_s)
                frequency[name][mode_key].update(
                    {
                        f"{source}_phase_frequency_hz": source_frequency,
                        f"{source}_phase_fit_r_squared": source_r2,
                        f"{source}_frequency_error_hz": source_frequency
                        - truth_frequency,
                        f"{source}_temporal_complex_coherence": complex_mode_metrics(
                            truth_series, series
                        )["coherence"],
                    }
                )
    return rows, stability, frequency, powers


def series_metrics(truth: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    truth64 = np.asarray(truth, dtype=np.float64)
    candidate64 = np.asarray(candidate, dtype=np.float64)
    difference = candidate64 - truth64
    truth_rms = math.sqrt(float(np.mean(truth64**2)))
    return {
        "truth_mean": float(np.mean(truth64)),
        "candidate_mean": float(np.mean(candidate64)),
        "mean_ratio": float(np.mean(candidate64))
        / (
            float(np.mean(truth64))
            if abs(float(np.mean(truth64))) > TINY
            else TINY
        ),
        "relative_l2": math.sqrt(
            float(np.sum(difference**2)) / max(float(np.sum(truth64**2)), TINY)
        ),
        "nrmse_truth_std": math.sqrt(float(np.mean(difference**2)))
        / max(float(np.std(truth64)), TINY),
        "time_correlation": centered_correlation(truth64, candidate64),
        "truth_rms": truth_rms,
    }


def modal_transport_analysis(
    truth: np.ndarray,
    baseline: np.ndarray,
    prediction: np.ndarray,
    norm_low: np.ndarray,
    norm_high: np.ndarray,
    dy: float,
) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    transport_modes: dict[str, np.ndarray] = {}
    for source, values in (
        ("truth", truth),
        ("baseline", baseline),
        ("model", prediction),
    ):
        physical = (
            values.astype(np.float64)
            * (norm_high - norm_low)[None, :, None, None]
            + norm_low[None, :, None, None]
        )
        ne = physical[:, 0]
        phi = physical[:, 2]
        ey = -(
            np.roll(phi, -1, axis=-1) - np.roll(phi, 1, axis=-1)
        ) / (2.0 * dy)
        ne_modes = np.fft.rfft(ne, axis=-1, norm="forward")
        ey_modes = np.fft.rfft(ey, axis=-1, norm="forward")
        # For positive Fourier modes, the real-space azimuthal mean of ne*Ey
        # is twice the real cross spectrum. B is constant, so omitting 1/B
        # leaves all relative reconstruction diagnostics unchanged.
        transport_modes[source] = 2.0 * np.mean(
            np.real(ne_modes * np.conj(ey_modes)), axis=1
        )
    band_slices = {
        "mtsi_candidate_n1_6": slice(1, 7),
        "transition_n7_8": slice(7, 9),
        "ecdi_candidate_n9_21": slice(9, 22),
        "resolved_n1_32": slice(1, 33),
    }
    series = {
        source: {
            band: np.sum(values[:, modes], axis=1)
            for band, modes in band_slices.items()
        }
        for source, values in transport_modes.items()
    }
    summary = {}
    for band in band_slices:
        summary[band] = {
            source: series_metrics(series["truth"][band], series[source][band])
            for source in ("baseline", "model")
        }
    return summary, series


def periodic_physics_metrics(
    values: np.ndarray, norm_low: np.ndarray, norm_high: np.ndarray, dx: float, dy: float
) -> dict[str, np.ndarray]:
    physical = values.astype(np.float64) * (norm_high - norm_low)[None, :, None, None] + norm_low[None, :, None, None]
    ne, ni, phi = physical[:, 0], physical[:, 1], physical[:, 2]
    radial_second = (
        phi[:, 2:, :] - 2.0 * phi[:, 1:-1, :] + phi[:, :-2, :]
    ) / dx**2
    azimuth_second = (
        np.roll(phi[:, 1:-1, :], -1, axis=-1)
        - 2.0 * phi[:, 1:-1, :]
        + np.roll(phi[:, 1:-1, :], 1, axis=-1)
    ) / dy**2
    laplacian = radial_second + azimuth_second
    source = E_CHARGE * (ni[:, 1:-1] - ne[:, 1:-1]) / EPS0
    residual = laplacian + source
    residual_rms = np.sqrt(np.mean(residual**2, axis=(1, 2)))
    source_rms = np.sqrt(np.mean(source**2, axis=(1, 2)))
    lap0 = laplacian.reshape(len(laplacian), -1)
    rhs0 = (-source).reshape(len(source), -1)
    lap0 -= np.mean(lap0, axis=1, keepdims=True)
    rhs0 -= np.mean(rhs0, axis=1, keepdims=True)
    balance_corr = np.sum(lap0 * rhs0, axis=1) / np.maximum(
        np.sqrt(np.sum(lap0**2, axis=1) * np.sum(rhs0**2, axis=1)), TINY
    )
    ex = -np.gradient(phi, dx, axis=1, edge_order=2)
    ey = -(
        np.roll(phi, -1, axis=-1) - np.roll(phi, 1, axis=-1)
    ) / (2.0 * dy)
    imbalance = np.sqrt(np.mean((ni - ne) ** 2, axis=(1, 2))) / np.maximum(
        np.sqrt(np.mean((0.5 * (ni + ne)) ** 2, axis=(1, 2))), TINY
    )
    return {
        "relative_poisson_residual": residual_rms / np.maximum(source_rms, TINY),
        "poisson_balance_corr": balance_corr,
        "quasineutral_imbalance": imbalance,
        "ex": ex,
        "ey": ey,
    }


def summarize_physics(
    truth: np.ndarray,
    baseline: np.ndarray,
    prediction: np.ndarray,
    norm_low: np.ndarray,
    norm_high: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    diagnostics = {
        source: periodic_physics_metrics(values, norm_low, norm_high, dx, dy)
        for source, values in (
            ("truth", truth),
            ("baseline", baseline),
            ("model", prediction),
        )
    }
    summary = {}
    truth_e_power = np.mean(
        diagnostics["truth"]["ex"] ** 2 + diagnostics["truth"]["ey"] ** 2
    )
    for source in ("truth", "baseline", "model"):
        item = {}
        for metric in (
            "relative_poisson_residual",
            "poisson_balance_corr",
            "quasineutral_imbalance",
        ):
            values = diagnostics[source][metric]
            item[metric] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
            }
        ex_error = diagnostics[source]["ex"] - diagnostics["truth"]["ex"]
        ey_error = diagnostics[source]["ey"] - diagnostics["truth"]["ey"]
        item["electric_field_relative_l2"] = math.sqrt(
            float(np.sum(ex_error**2 + ey_error**2))
            / max(
                float(
                    np.sum(
                        diagnostics["truth"]["ex"] ** 2
                        + diagnostics["truth"]["ey"] ** 2
                    )
                ),
                TINY,
            )
        )
        item["electric_field_energy_ratio"] = float(
            np.mean(diagnostics[source]["ex"] ** 2 + diagnostics[source]["ey"] ** 2)
            / max(float(truth_e_power), TINY)
        )
        summary[source] = item
    for source in ("baseline", "model"):
        summary[source]["poisson_residual_median_ratio_to_truth"] = float(
            np.median(diagnostics[source]["relative_poisson_residual"])
            / max(float(np.median(diagnostics["truth"]["relative_poisson_residual"])), TINY)
        )
        summary[source]["quasineutral_median_ratio_to_truth"] = float(
            np.median(diagnostics[source]["quasineutral_imbalance"])
            / max(float(np.median(diagnostics["truth"]["quasineutral_imbalance"])), TINY)
        )
    return summary, diagnostics


def plot_snapshot(
    path: Path,
    truth: np.ndarray,
    baseline: np.ndarray,
    prediction: np.ndarray,
    index: int,
    time_us: float,
    grid_label: str,
) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(14, 9), constrained_layout=True)
    sources = (truth, baseline, prediction)
    titles = (
        "PIC truth",
        f"{grid_label} interpolation",
        "SimVP residual",
        "model error",
    )
    for channel, name in enumerate(DISPLAY_NAMES):
        low, high = np.percentile(truth[index, channel], [1, 99])
        for column, source in enumerate(sources):
            image = axes[channel, column].imshow(
                source[index, channel], origin="lower", aspect="auto", vmin=low, vmax=high
            )
            fig.colorbar(image, ax=axes[channel, column], shrink=0.72)
        error = prediction[index, channel] - truth[index, channel]
        limit = max(float(np.percentile(np.abs(error), 99)), TINY)
        image = axes[channel, 3].imshow(
            error, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit
        )
        fig.colorbar(image, ax=axes[channel, 3], shrink=0.72)
        axes[channel, 0].set_ylabel(name)
        for column in range(4):
            axes[channel, column].set_xlabel("azimuth index")
    for column, title in enumerate(titles):
        axes[0, column].set_title(title)
    fig.suptitle(
        f"Held-out {grid_label} reconstruction at {time_us:.3f} us (normalized fields)"
    )
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_mode_spectra(path: Path, powers: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    modes = np.arange(1, 33)
    for channel, axis in enumerate(axes):
        for source, style in (("truth", "-"), ("baseline", "--"), ("model", "-.")):
            mean_power = np.mean(powers[source][:, channel, 1:33], axis=0)
            axis.semilogy(modes, np.maximum(mean_power, TINY), style, label=source)
        axis.axvspan(9, 21, alpha=0.1, color="tab:red", label="n=9--21" if channel == 0 else None)
        axis.set_title(DISPLAY_NAMES[channel])
        axis.set_xlabel("azimuthal mode n")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("mean modal power")
    axes[0].legend(fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_mode_time_series(path: Path, powers: dict, time_us: np.ndarray) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(14, 9), constrained_layout=True, sharex=True)
    selections = (("n=2", slice(2, 3)), ("n=7", slice(7, 8)), ("n=9--21", slice(9, 22)))
    for channel in range(3):
        for column, (label, mode_slice) in enumerate(selections):
            axis = axes[channel, column]
            truth_series = np.sum(powers["truth"][:, channel, mode_slice], axis=1)
            scale = max(float(np.mean(truth_series)), TINY)
            for source, style in (("truth", "-"), ("baseline", "--"), ("model", "-.")):
                series = np.sum(powers[source][:, channel, mode_slice], axis=1) / scale
                axis.plot(time_us, series, style, label=source, linewidth=1.2)
            if channel == 0:
                axis.set_title(label)
            if column == 0:
                axis.set_ylabel(f"{DISPLAY_NAMES[channel]}\npower / truth mean")
            if channel == 2:
                axis.set_xlabel("time [us]")
            axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_physics(path: Path, diagnostics: dict, time_us: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    metrics = (
        ("relative_poisson_residual", "relative Poisson residual"),
        ("poisson_balance_corr", "Poisson balance correlation"),
        ("quasineutral_imbalance", "quasineutral imbalance"),
    )
    for axis, (key, title) in zip(axes, metrics):
        for source, style in (("truth", "-"), ("baseline", "--"), ("model", "-.")):
            axis.plot(time_us, diagnostics[source][key], style, label=source)
        axis.set_title(title)
        axis.set_xlabel("time [us]")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_modal_transport(path: Path, series: dict, time_us: np.ndarray) -> None:
    bands = (
        ("mtsi_candidate_n1_6", "MTSI candidate n=1--6"),
        ("ecdi_candidate_n9_21", "ECDI candidate n=9--21"),
        ("resolved_n1_32", "resolved total n=1--32"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for axis, (band, title) in zip(axes, bands):
        truth = series["truth"][band]
        scale = max(math.sqrt(float(np.mean(truth**2))), TINY)
        for source, style in (("truth", "-"), ("baseline", "--"), ("model", "-.")):
            axis.plot(time_us, series[source][band] / scale, style, label=source)
        axis.set_title(title)
        axis.set_xlabel("time [us]")
        axis.set_ylabel(r"$\langle n_e E_y\rangle$ / truth RMS")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.workdir / "stability_reconstruction_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    checkpoint_path = args.workdir / "checkpoint_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metadata = checkpoint.get("metadata", {})
    coarse_size = int(metadata.get("coarse_size", 256 // int(metadata.get("grid_factor", 2))))
    grid_factor = float(metadata.get("effective_grid_factor", metadata.get("grid_factor", 2)))
    grid_label = str(metadata.get("grid_label", f"G{grid_factor:g}"))
    with h5py.File(args.h5, "r") as handle:
        times_s_all = np.asarray(handle["time_s"], dtype=np.float64)
        selected = np.flatnonzero(
            (times_s_all * 1.0e6 >= args.start_us - 1.0e-9)
            & (times_s_all * 1.0e6 <= args.end_us + 1.0e-9)
        )
        fine = np.asarray(
            handle["data_tchw"][int(selected[0]) : int(selected[-1]) + 1],
            dtype=np.float32,
        )
        times_s = times_s_all[selected]
        norm_low = np.asarray(handle["norm_low"], dtype=np.float64)
        norm_high = np.asarray(handle["norm_high"], dtype=np.float64)
        x_m = np.asarray(handle["x_m"], dtype=np.float64)
        y_m = np.asarray(handle["y_m"], dtype=np.float64)
    baseline_all = make_coarse_size_interpolated(fine, coarse_size) if 256 % coarse_size else make_grid_interpolated(fine, 256 // coarse_size)
    frame_count = len(fine)
    val_end = int(math.floor(0.9 * frame_count))
    sequence_length = int(checkpoint["model_kwargs"]["in_shape"][0])
    hop = int(checkpoint["metadata"]["window_hop"])
    starts = segment_starts(val_end, frame_count, sequence_length, hop)
    prediction, truth, baseline, local_indices, consistency = predict_unique_frames(
        fine, baseline_all, starts, checkpoint, device
    )
    evaluation_times_s = times_s[local_indices]
    evaluation_times_us = evaluation_times_s * 1.0e6
    print(
        f"[prediction] {len(starts)} windows -> {len(local_indices)} unique held-out frames "
        f"({evaluation_times_us[0]:.3f}--{evaluation_times_us[-1]:.3f} us)",
        flush=True,
    )

    field_rows = []
    for channel, name in enumerate(CHANNELS):
        baseline_metrics = field_metrics(truth[:, channel], baseline[:, channel])
        model_metrics = field_metrics(truth[:, channel], prediction[:, channel])
        field_rows.append(
            {
                "field": name,
                **{f"baseline_{key}": value for key, value in baseline_metrics.items()},
                **{f"model_{key}": value for key, value in model_metrics.items()},
                "model_skill_over_interpolation": 1.0
                - model_metrics["mse"] / baseline_metrics["mse"],
            }
        )

    mode_rows, stability, frequency, powers = modal_analysis(
        truth, baseline, prediction, evaluation_times_s
    )
    dx = float(np.median(np.diff(x_m)))
    dy = float(np.median(np.diff(y_m)))
    physics, physics_time = summarize_physics(
        truth, baseline, prediction, norm_low, norm_high, dx, dy
    )
    modal_transport, modal_transport_time = modal_transport_analysis(
        truth, baseline, prediction, norm_low, norm_high, dy
    )

    write_csv(output_dir / "field_metrics.csv", field_rows)
    write_csv(output_dir / "azimuthal_mode_metrics.csv", mode_rows)
    summary = {
        "description": (
            f"Held-out saturated-instability reconstruction diagnostics for "
            f"E25 {grid_label} synchronous residual SimVP"
        ),
        "grid_factor": grid_factor,
        "grid_label": grid_label,
        "interpretation_scope": {
            "can_test": "same-case held-out spatial reconstruction and saturated modal dynamics",
            "cannot_test": "linear growth rate from initial perturbations or transfer to a newly run coarse PIC case",
        },
        "checkpoint": str(checkpoint_path.resolve()),
        "source_h5": str(args.h5.resolve()),
        "held_out_time_us": [float(evaluation_times_us[0]), float(evaluation_times_us[-1])],
        "held_out_unique_frames": len(local_indices),
        "time_step_ns": float(np.median(np.diff(evaluation_times_s)) * 1.0e9),
        "temporal_frequency_resolution_hz": float(
            1.0 / (len(evaluation_times_s) * np.median(np.diff(evaluation_times_s)))
        ),
        "overlap_consistency": consistency,
        "field_metrics": {row["field"]: row for row in field_rows},
        "saturated_stability": stability,
        "phase_frequency": frequency,
        "modal_transport_proxy_ne_ey": modal_transport,
        "physics": physics,
    }
    (output_dir / "stability_reconstruction_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    snapshot_index = len(evaluation_times_us) // 2
    plot_snapshot(
        output_dir / "heldout_snapshot_comparison.png",
        truth,
        baseline,
        prediction,
        snapshot_index,
        float(evaluation_times_us[snapshot_index]),
        grid_label,
    )
    plot_mode_spectra(output_dir / "azimuthal_mode_spectra.png", powers)
    plot_mode_time_series(
        output_dir / "azimuthal_mode_power_time_series.png", powers, evaluation_times_us
    )
    plot_physics(
        output_dir / "physics_consistency_time_series.png",
        physics_time,
        evaluation_times_us,
    )
    plot_modal_transport(
        output_dir / "modal_transport_proxy_time_series.png",
        modal_transport_time,
        evaluation_times_us,
    )
    readme = f"""# E25 {grid_label} saturated-instability reconstruction analysis

This analysis averages overlapping synchronous SimVP outputs into unique
physical frames and evaluates only the held-out final segment
({evaluation_times_us[0]:.3f}--{evaluation_times_us[-1]:.3f} us).

It compares PIC truth, G2 linear interpolation, and learned residual
reconstruction using field errors, azimuthal modes, modal power histories,
phase-derived frequencies, MTSI/ECDI modal electron-transport proxies,
electric fields, periodic Poisson residuals, and quasineutral imbalance.

The interval is already saturated.  These results do not measure the linear
growth rate from an initially small perturbation and do not establish transfer
to a separately run coarse-PIC system.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False), flush=True)
    print(f"[saved] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
