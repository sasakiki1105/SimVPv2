#!/usr/bin/env python3
"""Apply a synthetic-G2 residual model to an independently run native G2 PIC.

The synthetic training pairs and the native coarse PIC are different particle
realizations.  There is therefore no frame-wise fine truth for the native run.
This script deliberately evaluates distributional field statistics, radial
profiles, azimuthal spectra, modal transport, and discrete Poisson consistency
against an independent fine E20 reference instead of reporting pixel MSE.
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
from train_radaz_g2_residual_superresolution import segment_starts


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent / "PEPAPIC" / "test" / "results" / "2D_Landmark"
FINE_CASE = "2D_RadAz_Xe1p_Bx20mT_Ez20kVm_dt15ps_out15ns"
FINE_ROOT = RESULTS / FINE_CASE / FINE_CASE
DEFAULT_FINE = FINE_ROOT / "analysis_fields_uncompressed.h5"
DEFAULT_NORM = (
    FINE_ROOT
    / "SimVPv2_inputs"
    / "radaz_3ch_localnorm_trainfixed_margin20_native257x256_pad260x256.h5"
)
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
DEFAULT_WORKDIR = ROOT / "workdirs" / "radaz_e20_g2_simvp_residual_sr_sync10_20to30us"
DEFAULT_OUTPUT = ROOT / "workdirs" / "radaz_e20_native_g2_reconstruction"
CHANNELS = ("electron_den", "ion_den", "phi")
DISPLAY = (r"$n_e$", r"$n_i$", r"$\phi$")
EPS0 = 8.8541878128e-12
E_CHARGE = 1.602176634e-19
TINY = np.finfo(np.float64).tiny


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-h5", type=Path, default=DEFAULT_COARSE)
    parser.add_argument("--fine-h5", type=Path, default=DEFAULT_FINE)
    parser.add_argument("--normalization-h5", type=Path, default=DEFAULT_NORM)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-us", type=float, default=20.0)
    parser.add_argument("--end-us", type=float, default=30.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window-hop", type=int, default=5)
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
    """Interpolate native 128-cell PIC nodes to the 260x256 model grid."""
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


def predict_native(
    baseline: np.ndarray,
    checkpoint: dict,
    device: torch.device,
    hop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = SimVP_Model(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    length = int(checkpoint["model_kwargs"]["in_shape"][0])
    starts = segment_starts(0, len(baseline), length, hop)
    scale = torch.as_tensor(
        checkpoint["residual_rms"], dtype=torch.float32, device=device
    ).view(1, 1, 3, 1, 1)
    final = int(starts[-1]) + length
    sums = np.zeros((final, 3, 260, 256), dtype=np.float32)
    counts = np.zeros(final, dtype=np.int16)
    with torch.inference_mode():
        for window_index, start in enumerate(starts, start=1):
            tensor = torch.from_numpy(baseline[int(start) : int(start) + length]).unsqueeze(0).to(device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                output = tensor + model(tensor) * scale
            values = output[0].float().cpu().numpy()
            sums[int(start) : int(start) + length] += values
            counts[int(start) : int(start) + length] += 1
            if window_index == 1 or window_index == len(starts) or window_index % 25 == 0:
                print(f"[predict] {window_index}/{len(starts)} windows", flush=True)
    covered = counts > 0
    prediction = sums[covered] / counts[covered, None, None, None]
    return prediction, np.flatnonzero(covered), counts[covered]


def sample_flat(values: np.ndarray, maximum: int = 1_500_000) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).ravel()
    stride = max(1, int(math.ceil(len(flat) / maximum)))
    return np.asarray(flat[::stride], dtype=np.float64)


def distribution_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    ref = sample_flat(reference)
    cand = sample_flat(candidate)
    quantiles = np.linspace(0.01, 0.99, 99)
    rq = np.quantile(ref, quantiles)
    cq = np.quantile(cand, quantiles)
    ref_std = max(float(np.std(ref)), TINY)
    return {
        "reference_mean": float(np.mean(ref)),
        "candidate_mean": float(np.mean(cand)),
        "mean_shift_over_reference_std": float(abs(np.mean(cand) - np.mean(ref)) / ref_std),
        "reference_std": float(np.std(ref)),
        "candidate_std": float(np.std(cand)),
        "std_ratio": float(np.std(cand) / ref_std),
        "rms_ratio": float(np.sqrt(np.mean(cand**2)) / max(float(np.sqrt(np.mean(ref**2))), TINY)),
        "quantile_rmse_over_reference_std": float(np.sqrt(np.mean((cq - rq) ** 2)) / ref_std),
        "q01_ratio_or_shift": float((cq[0] - rq[0]) / ref_std),
        "q50_shift_over_reference_std": float((cq[49] - rq[49]) / ref_std),
        "q99_ratio_or_shift": float((cq[-1] - rq[-1]) / ref_std),
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
        "nrmse_over_profile_std": float(np.sqrt(np.mean((cand - ref) ** 2)) / max(np.std(ref), TINY)),
        "correlation": centered_correlation(ref, cand),
    }


def modal_power(values: np.ndarray, maximum_mode: int = 64, chunk: int = 16) -> np.ndarray:
    total = np.zeros((3, maximum_mode + 1), dtype=np.float64)
    samples = 0
    for start in range(0, len(values), chunk):
        block = np.asarray(values[start : start + chunk], dtype=np.float32)
        coeff = np.fft.rfft(block, axis=-1, norm="forward")[..., : maximum_mode + 1]
        total += np.sum(np.abs(coeff) ** 2, axis=(0, 2))
        samples += block.shape[0] * block.shape[2]
    return total / samples


def modal_summary(reference: np.ndarray, candidate: np.ndarray) -> dict:
    modes = np.arange(1, 33)
    bands = {"n1_6": slice(1, 7), "n7_8": slice(7, 9), "n9_21": slice(9, 22)}
    result = {}
    for channel, name in enumerate(CHANNELS):
        # Density and potential powers differ by many physical orders of
        # magnitude.  A channel-global floor would erase all phi differences.
        floor = max(float(np.max(reference[channel, 1:33])) * 1.0e-14, TINY)
        log_error = np.log10(np.maximum(candidate[channel, 1:33], floor)) - np.log10(
            np.maximum(reference[channel, 1:33], floor)
        )
        result[name] = {
            "log10_power_rmse_n1_32": float(np.sqrt(np.mean(log_error**2))),
            "dominant_mode_reference_n1_32": int(modes[np.argmax(reference[channel, 1:33])]),
            "dominant_mode_candidate_n1_32": int(modes[np.argmax(candidate[channel, 1:33])]),
            "bands": {
                band: {
                    "candidate_to_reference_power": float(
                        np.sum(candidate[channel, selection])
                        / max(float(np.sum(reference[channel, selection])), TINY)
                    )
                }
                for band, selection in bands.items()
            },
        }
    return result


def modal_transport(values: np.ndarray, dy: float, chunk: int = 16) -> dict[str, np.ndarray]:
    outputs = {name: [] for name in ("n1_6", "n7_8", "n9_21", "n1_32")}
    bands = {"n1_6": slice(1, 7), "n7_8": slice(7, 9), "n9_21": slice(9, 22), "n1_32": slice(1, 33)}
    for start in range(0, len(values), chunk):
        block = np.asarray(values[start : start + chunk], dtype=np.float64)
        ne, phi = block[:, 0], block[:, 2]
        ey = -(np.roll(phi, -1, axis=-1) - np.roll(phi, 1, axis=-1)) / (2.0 * dy)
        ne_mode = np.fft.rfft(ne, axis=-1, norm="forward")
        ey_mode = np.fft.rfft(ey, axis=-1, norm="forward")
        flux = 2.0 * np.mean(np.real(ne_mode * np.conj(ey_mode)), axis=1)
        for name, selection in bands.items():
            outputs[name].append(np.sum(flux[:, selection], axis=1))
    return {name: np.concatenate(parts) for name, parts in outputs.items()}


def poisson_summary(values: np.ndarray, dx: float, dy: float, chunk: int = 16) -> dict[str, float]:
    relative = []
    correlation = []
    for start in range(0, len(values), chunk):
        block = np.asarray(values[start : start + chunk], dtype=np.float64)
        ne, ni, phi = block[:, 0], block[:, 1], block[:, 2]
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
        a = laplacian.reshape(len(block), -1)
        b = (-source).reshape(len(block), -1)
        a -= np.mean(a, axis=1, keepdims=True)
        b -= np.mean(b, axis=1, keepdims=True)
        correlation.extend(
            np.sum(a * b, axis=1)
            / np.maximum(np.sqrt(np.sum(a**2, axis=1) * np.sum(b**2, axis=1)), TINY)
        )
    return {
        "relative_poisson_residual_mean": float(np.mean(relative)),
        "relative_poisson_residual_median": float(np.median(relative)),
        "poisson_balance_correlation_mean": float(np.mean(correlation)),
        "poisson_balance_correlation_median": float(np.median(correlation)),
    }


def save_reconstruction(
    path: Path,
    prediction: np.ndarray,
    time_s: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    args: argparse.Namespace,
    norm_low: np.ndarray,
    norm_high: np.ndarray,
) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "data_tchw",
            data=np.asarray(prediction, dtype=np.float32),
            chunks=(1, 3, 256, 256),
            compression="gzip",
            compression_opts=4,
        )
        handle.create_dataset("time_s", data=time_s)
        handle.create_dataset("x_m", data=x_m[:256])
        handle.create_dataset("y_m", data=y_m[:256])
        handle.create_dataset("props", data=np.asarray(CHANNELS, dtype="S"))
        handle.create_dataset("norm_low", data=norm_low)
        handle.create_dataset("norm_high", data=norm_high)
        handle.attrs["units"] = "physical SI fields: density m^-3, phi V"
        handle.attrs["task"] = "same-time native G2 to fine-grid residual reconstruction"
        handle.attrs["paired_fine_truth_available"] = False
        handle.attrs["checkpoint"] = str((args.workdir / "checkpoint_best.pth").resolve())
        handle.attrs["source_native_coarse_h5"] = str(args.coarse_h5.resolve())


def make_plots(
    output: Path,
    fine: np.ndarray,
    baseline: np.ndarray,
    prediction: np.ndarray,
    powers: dict[str, np.ndarray],
    profiles: dict[str, np.ndarray],
) -> None:
    modes = np.arange(1, 33)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for channel, axis in enumerate(axes):
        for source, style in (("fine", "-"), ("native_g2", "--"), ("reconstructed", "-.")):
            axis.semilogy(modes, np.maximum(powers[source][channel, 1:33], TINY), style, label=source)
        axis.axvspan(9, 21, color="tab:red", alpha=0.1)
        axis.set_title(DISPLAY[channel])
        axis.set_xlabel("azimuthal mode n")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("mean modal power")
    axes[0].legend(fontsize=8)
    fig.savefig(output / "azimuthal_mode_spectra.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    radius = np.arange(256)
    for channel, axis in enumerate(axes):
        scale = max(float(np.std(profiles["fine"][channel])), abs(float(np.mean(profiles["fine"][channel]))), TINY)
        for source, style in (("fine", "-"), ("native_g2", "--"), ("reconstructed", "-.")):
            axis.plot(radius, profiles[source][channel] / scale, style, label=source)
        axis.set_title(DISPLAY[channel])
        axis.set_xlabel("radial index")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("time/azimuth mean / fine scale")
    axes[0].legend(fontsize=8)
    fig.savefig(output / "radial_mean_profiles.png", dpi=180)
    plt.close(fig)

    index = len(prediction) // 2
    fig, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)
    sources = (fine, baseline, prediction)
    titles = ("fine E20 reference", "native G2 interpolation", "reconstructed")
    for channel in range(3):
        low, high = np.percentile(fine[index, channel], [1, 99])
        for column, source in enumerate(sources):
            image = axes[channel, column].imshow(
                source[index, channel], origin="lower", aspect="auto", vmin=low, vmax=high
            )
            fig.colorbar(image, ax=axes[channel, column], shrink=0.72)
            if channel == 0:
                axes[channel, column].set_title(titles[column])
        axes[channel, 0].set_ylabel(DISPLAY[channel])
    fig.suptitle("Same clock time; fine and native G2 are independent PIC realizations")
    fig.savefig(output / "independent_realization_snapshot.png", dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.workdir / "checkpoint_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    with h5py.File(args.normalization_h5, "r") as handle:
        norm_low = np.asarray(handle["norm_low"], dtype=np.float32)
        norm_high = np.asarray(handle["norm_high"], dtype=np.float32)
    fine, fine_time, x_m, y_m = load_fine_reference(
        args.fine_h5, args.start_us, args.end_us
    )
    baseline_physical, coarse_time, coarse_indices = load_native_baseline(
        args.coarse_h5, args.start_us, args.end_us, x_m, y_m
    )
    if len(fine_time) != len(coarse_time) or not np.allclose(fine_time, coarse_time, atol=1.0e-14):
        raise ValueError("Fine and native coarse time grids differ")
    span = norm_high - norm_low
    unclipped = (baseline_physical - norm_low[None, :, None, None]) / span[None, :, None, None]
    clip_low = np.mean(unclipped < 0.0, axis=(0, 2, 3))
    clip_high = np.mean(unclipped > 1.0, axis=(0, 2, 3))
    baseline_normalized = np.clip(unclipped, 0.0, 1.0).astype(np.float32)
    del baseline_physical, unclipped
    prediction_normalized, covered, coverage = predict_native(
        baseline_normalized, checkpoint, torch.device(args.device), args.window_hop
    )
    baseline = (
        baseline_normalized[covered, :, :256].astype(np.float32)
        * span[None, :, None, None]
        + norm_low[None, :, None, None]
    )
    prediction = (
        prediction_normalized[:, :, :256].astype(np.float32)
        * span[None, :, None, None]
        + norm_low[None, :, None, None]
    )
    fine = fine[covered]
    evaluation_time = coarse_time[covered]
    del baseline_normalized, prediction_normalized

    field_rows = []
    profiles = {
        "fine": np.mean(fine, axis=(0, 3), dtype=np.float64),
        "native_g2": np.mean(baseline, axis=(0, 3), dtype=np.float64),
        "reconstructed": np.mean(prediction, axis=(0, 3), dtype=np.float64),
    }
    profile_rows = []
    for channel, name in enumerate(CHANNELS):
        base_metrics = distribution_metrics(fine[:, channel], baseline[:, channel])
        model_metrics = distribution_metrics(fine[:, channel], prediction[:, channel])
        field_rows.append(
            {
                "field": name,
                **{f"native_g2_{key}": value for key, value in base_metrics.items()},
                **{f"reconstructed_{key}": value for key, value in model_metrics.items()},
                "reconstruction_improves_quantile_distance": (
                    model_metrics["quantile_rmse_over_reference_std"]
                    < base_metrics["quantile_rmse_over_reference_std"]
                ),
            }
        )
        base_profile = profile_metrics(profiles["fine"][channel], profiles["native_g2"][channel])
        model_profile = profile_metrics(profiles["fine"][channel], profiles["reconstructed"][channel])
        profile_rows.append(
            {
                "field": name,
                **{f"native_g2_{key}": value for key, value in base_profile.items()},
                **{f"reconstructed_{key}": value for key, value in model_profile.items()},
                "reconstruction_improves_profile_l2": model_profile["relative_l2"] < base_profile["relative_l2"],
            }
        )

    powers = {
        "fine": modal_power(fine),
        "native_g2": modal_power(baseline),
        "reconstructed": modal_power(prediction),
    }
    mode_summary = {
        "native_g2": modal_summary(powers["fine"], powers["native_g2"]),
        "reconstructed": modal_summary(powers["fine"], powers["reconstructed"]),
    }
    mode_rows = []
    for channel, name in enumerate(CHANNELS):
        for mode in range(1, 65):
            reference = float(powers["fine"][channel, mode])
            mode_rows.append(
                {
                    "field": name,
                    "mode": mode,
                    "fine_power": reference,
                    "native_g2_power": float(powers["native_g2"][channel, mode]),
                    "reconstructed_power": float(powers["reconstructed"][channel, mode]),
                    "native_g2_to_fine": float(powers["native_g2"][channel, mode] / max(reference, TINY)),
                    "reconstructed_to_fine": float(powers["reconstructed"][channel, mode] / max(reference, TINY)),
                }
            )

    dy = float(np.median(np.diff(y_m[:256])))
    dx = float(np.median(np.diff(x_m[:256])))
    transports = {
        "fine": modal_transport(fine, dy),
        "native_g2": modal_transport(baseline, dy),
        "reconstructed": modal_transport(prediction, dy),
    }
    transport_summary = {}
    for band in transports["fine"]:
        base = distribution_metrics(transports["fine"][band], transports["native_g2"][band])
        model = distribution_metrics(transports["fine"][band], transports["reconstructed"][band])
        transport_summary[band] = {
            "native_g2": base,
            "reconstructed": model,
            "reconstruction_improves_quantile_distance": (
                model["quantile_rmse_over_reference_std"]
                < base["quantile_rmse_over_reference_std"]
            ),
        }
    physics = {
        "fine": poisson_summary(fine, dx, dy),
        "native_g2": poisson_summary(baseline, dx, dy),
        "reconstructed": poisson_summary(prediction, dx, dy),
    }

    write_csv(args.output_dir / "field_distribution_metrics.csv", field_rows)
    write_csv(args.output_dir / "radial_profile_metrics.csv", profile_rows)
    write_csv(args.output_dir / "azimuthal_mode_power.csv", mode_rows)
    reconstruction_path = args.output_dir / "native_g2_reconstructed_3ch_20to30us.h5"
    save_reconstruction(
        reconstruction_path,
        prediction,
        evaluation_time,
        x_m,
        y_m,
        args,
        norm_low,
        norm_high,
    )
    make_plots(args.output_dir, fine, baseline, prediction, powers, profiles)

    field_wins = sum(row["reconstruction_improves_quantile_distance"] for row in field_rows)
    profile_wins = sum(row["reconstruction_improves_profile_l2"] for row in profile_rows)
    transport_wins = sum(
        item["reconstruction_improves_quantile_distance"] for item in transport_summary.values()
    )
    modal_wins = sum(
        mode_summary["reconstructed"][name]["log10_power_rmse_n1_32"]
        < mode_summary["native_g2"][name]["log10_power_rmse_n1_32"]
        for name in CHANNELS
    )
    summary = {
        "status": "complete",
        "scope": (
            "E20 synthetic-G2-trained same-time residual model applied without retraining "
            "to an independently run native G2 E20 PIC"
        ),
        "paired_truth_warning": (
            "Fine and native G2 are independent particle realizations; frame-wise pixel MSE, "
            "phase coherence, and time-aligned correlation are intentionally not reported."
        ),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_best_epoch": int(checkpoint.get("epoch", -1)),
        "native_coarse_h5": str(args.coarse_h5.resolve()),
        "fine_reference_h5": str(args.fine_h5.resolve()),
        "normalization_h5": str(args.normalization_h5.resolve()),
        "time_us": [float(evaluation_time[0] * 1.0e6), float(evaluation_time[-1] * 1.0e6)],
        "frames": int(len(evaluation_time)),
        "coverage_per_frame": [int(np.min(coverage)), int(np.max(coverage))],
        "native_input_clipped_low_fraction": dict(zip(CHANNELS, clip_low.tolist())),
        "native_input_clipped_high_fraction": dict(zip(CHANNELS, clip_high.tolist())),
        "field_distribution_metrics": {row["field"]: row for row in field_rows},
        "radial_profile_metrics": {row["field"]: row for row in profile_rows},
        "azimuthal_mode_summary": mode_summary,
        "modal_transport_distribution": transport_summary,
        "fine_grid_discrete_poisson": physics,
        "scorecard": {
            "field_distribution_wins_out_of_3": int(field_wins),
            "radial_profile_wins_out_of_3": int(profile_wins),
            "modal_spectrum_wins_out_of_3": int(modal_wins),
            "transport_distribution_wins_out_of_4": int(transport_wins),
            "total_wins_out_of_13": int(field_wins + profile_wins + modal_wins + transport_wins),
        },
        "reconstruction_h5": str(reconstruction_path.resolve()),
    }
    (args.output_dir / "native_g2_reconstruction_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    readme = f"""# Native G2 E20 reconstruction

The model was trained on paired fine E20 / synthetic G2 data and then applied
without retraining to an independently run native G2 E20 PIC.  Because the two
PIC runs do not share a particle realization, this report compares field
distributions, radial profiles, modal spectra, modal-transport distributions,
and discrete Poisson diagnostics; it does not use pixel-wise MSE.

Scorecard: {summary['scorecard']['total_wins_out_of_13']} of 13 distributional
or structural comparisons moved closer to the independent fine E20 reference.
See `native_g2_reconstruction_summary.json` for the complete metrics.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(json_safe(summary["scorecard"]), indent=2), flush=True)
    print(f"[saved] {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
