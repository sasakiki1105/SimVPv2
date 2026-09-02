#!/usr/bin/env python3
"""Train synchronous grid-coarsened-to-fine residual super-resolution.

G means grid-coarsening factor.  For example, G2 is produced by
non-overlapping 2x2 averaging and G4 by 4x4 averaging on the 256x256
physical core.  The coarse field is interpolated back to the model grid and
SimVP predicts only the missing same-time fine-grid residual.  This
deliberately isolates spatial reconstruction from future forecasting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from openstl.models.simvp_model import SimVP_Model


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
    / "SimVPv2_inputs"
    / "radaz_3ch_targetnorm_trainfixed_margin20_native257x256_pad260x256.h5"
)
DEFAULT_OUTPUT = ROOT / "workdirs" / "radaz_e25_g2_simvp_residual_sr_sync10_20to30us"
CHANNEL_NAMES = ("electron_den", "ion_den", "phi")
PHYSICAL_CORE = (256, 256)
VALID_RADIAL = 257


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--grid-factor",
        type=int,
        default=2,
        choices=(2, 4, 8),
        help="G factor used for isotropic non-overlapping block averaging",
    )
    parser.add_argument(
        "--coarse-size",
        type=int,
        choices=(37, 43, 51),
        help="Conservative arbitrary coarse grid; overrides --grid-factor",
    )
    parser.add_argument("--start-us", type=float, default=20.0)
    parser.add_argument("--end-us", type=float, default=30.0)
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument(
        "--window-hop",
        type=int,
        default=5,
        help="separation between window starts; frame stride inside a sequence stays 1",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--hid-s", type=int, default=64)
    parser.add_argument("--hid-t", type=int, default=512)
    parser.add_argument("--n-s", type=int, default=4)
    parser.add_argument("--n-t", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def interpolation_stencil(
    size: int, factor: int, periodic: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coarse_size = size // factor
    positions = np.arange(size, dtype=np.float64)
    centers = np.arange(coarse_size, dtype=np.float64) * factor + (factor - 1) / 2
    if periodic:
        coordinate = (positions - centers[0]) / factor
        left_unwrapped = np.floor(coordinate).astype(np.int64)
        alpha = coordinate - left_unwrapped
        left = np.mod(left_unwrapped, coarse_size)
        right = np.mod(left_unwrapped + 1, coarse_size)
        return left, right, (1.0 - alpha).astype(np.float32), alpha.astype(np.float32)

    right = np.searchsorted(centers, positions, side="right")
    right = np.clip(right, 1, coarse_size - 1)
    left = right - 1
    alpha = (positions - centers[left]) / (centers[right] - centers[left])
    below = positions <= centers[0]
    above = positions >= centers[-1]
    alpha = np.clip(alpha, 0.0, 1.0)
    left[below] = 0
    right[below] = 0
    alpha[below] = 0.0
    left[above] = coarse_size - 1
    right[above] = coarse_size - 1
    alpha[above] = 0.0
    return left, right, (1.0 - alpha).astype(np.float32), alpha.astype(np.float32)


def make_grid_interpolated(
    frames: np.ndarray, grid_factor: int, chunk_size: int = 24
) -> np.ndarray:
    """Return G-factor block averages interpolated to the padded model grid."""
    if frames.ndim != 4 or tuple(frames.shape[1:]) != (3, 260, 256):
        raise ValueError(f"Expected (T,3,260,256), received {frames.shape}")
    if grid_factor <= 1 or 256 % grid_factor:
        raise ValueError(f"grid_factor must divide 256 and exceed one: {grid_factor}")
    coarse_size = 256 // grid_factor
    radial = interpolation_stencil(256, grid_factor, periodic=False)
    azimuth = interpolation_stencil(256, grid_factor, periodic=True)
    output = np.empty_like(frames, dtype=np.float32)
    for start in range(0, len(frames), chunk_size):
        stop = min(start + chunk_size, len(frames))
        core = np.asarray(frames[start:stop, :, :256, :], dtype=np.float32)
        count = len(core)
        coarse = core.reshape(
            count,
            3,
            coarse_size,
            grid_factor,
            coarse_size,
            grid_factor,
        ).mean(axis=(3, 5))
        r0, r1, rw0, rw1 = radial
        radial_fine = (
            coarse[:, :, r0, :] * rw0[None, None, :, None]
            + coarse[:, :, r1, :] * rw1[None, None, :, None]
        )
        a0, a1, aw0, aw1 = azimuth
        fine = (
            radial_fine[:, :, :, a0] * aw0[None, None, None, :]
            + radial_fine[:, :, :, a1] * aw1[None, None, None, :]
        )
        output[start:stop, :, :256, :] = fine
        output[start:stop, :, 256:, :] = fine[:, :, -1:, :]
    return output


def conservative_average_matrix(fine_size: int, coarse_size: int) -> np.ndarray:
    """Cell-overlap averages from a uniform fine grid to an arbitrary grid."""
    edges = np.linspace(0.0, float(fine_size), coarse_size + 1)
    matrix = np.zeros((coarse_size, fine_size), dtype=np.float32)
    for coarse in range(coarse_size):
        left, right = edges[coarse], edges[coarse + 1]
        first, last = int(np.floor(left)), int(np.ceil(right))
        for fine in range(first, min(last, fine_size)):
            overlap = max(0.0, min(right, fine + 1.0) - max(left, float(fine)))
            matrix[coarse, fine] = overlap / (right - left)
    return matrix


def arbitrary_interpolation_stencil(size: int, coarse_size: int, periodic: bool):
    positions = np.arange(size, dtype=np.float64)
    spacing = size / coarse_size
    first_center = 0.5 * spacing - 0.5
    coordinate = (positions - first_center) / spacing
    if periodic:
        left_raw = np.floor(coordinate).astype(np.int64)
        alpha = coordinate - left_raw
        return np.mod(left_raw, coarse_size), np.mod(left_raw + 1, coarse_size), (1-alpha).astype(np.float32), alpha.astype(np.float32)
    right = np.clip(np.searchsorted(first_center + spacing*np.arange(coarse_size), positions, side="right"), 1, coarse_size-1)
    left = right - 1
    centers = first_center + spacing*np.arange(coarse_size)
    alpha = np.clip((positions-centers[left])/(centers[right]-centers[left]), 0.0, 1.0)
    below, above = positions <= centers[0], positions >= centers[-1]
    left[below] = right[below] = 0; alpha[below] = 0.0
    left[above] = right[above] = coarse_size-1; alpha[above] = 0.0
    return left, right, (1-alpha).astype(np.float32), alpha.astype(np.float32)


def make_coarse_size_interpolated(frames: np.ndarray, coarse_size: int, chunk_size: int = 12) -> np.ndarray:
    """Conservative N x N averaging followed by center-based linear interpolation."""
    if frames.ndim != 4 or tuple(frames.shape[1:]) != (3, 260, 256):
        raise ValueError(f"Expected (T,3,260,256), received {frames.shape}")
    average = conservative_average_matrix(256, coarse_size)
    radial = arbitrary_interpolation_stencil(256, coarse_size, periodic=False)
    azimuth = arbitrary_interpolation_stencil(256, coarse_size, periodic=True)
    output = np.empty_like(frames, dtype=np.float32)
    for start in range(0, len(frames), chunk_size):
        stop = min(start + chunk_size, len(frames))
        core = np.asarray(frames[start:stop, :, :256, :], dtype=np.float32)
        radial_coarse = np.einsum("ri,tcij->tcrj", average, core, optimize=True)
        coarse = np.einsum("aj,tcrj->tcra", average, radial_coarse, optimize=True)
        r0, r1, rw0, rw1 = radial
        radial_fine = coarse[:, :, r0, :] * rw0[None,None,:,None] + coarse[:, :, r1, :] * rw1[None,None,:,None]
        a0, a1, aw0, aw1 = azimuth
        fine = radial_fine[:, :, :, a0] * aw0[None,None,None,:] + radial_fine[:, :, :, a1] * aw1[None,None,None,:]
        output[start:stop, :, :256, :] = fine
        output[start:stop, :, 256:, :] = fine[:, :, -1:, :]
    return output


def make_g2_interpolated(frames: np.ndarray, chunk_size: int = 24) -> np.ndarray:
    """Backward-compatible G2 wrapper used by existing analysis scripts."""
    return make_grid_interpolated(frames, grid_factor=2, chunk_size=chunk_size)


def compute_residual_rms(
    fine: np.ndarray, baseline: np.ndarray, frame_stop: int
) -> np.ndarray:
    square_sum = np.zeros(3, dtype=np.float64)
    count = 0
    for start in range(0, frame_stop, 32):
        stop = min(start + 32, frame_stop)
        residual = (
            fine[start:stop, :, :VALID_RADIAL, :]
            - baseline[start:stop, :, :VALID_RADIAL, :]
        ).astype(np.float64)
        square_sum += np.sum(residual * residual, axis=(0, 2, 3))
        count += residual.shape[0] * residual.shape[2] * residual.shape[3]
    rms = np.sqrt(square_sum / count).astype(np.float32)
    if np.any(~np.isfinite(rms)) or np.any(rms <= 0):
        raise ValueError(f"Invalid training residual RMS: {rms}")
    return rms


def segment_starts(
    begin: int, end: int, sequence_length: int, hop: int
) -> np.ndarray:
    final = end - sequence_length
    if final < begin:
        raise ValueError(
            f"Segment [{begin},{end}) is shorter than sequence length {sequence_length}"
        )
    return np.arange(begin, final + 1, hop, dtype=np.int64)


class SynchronousResidualDataset(Dataset):
    def __init__(
        self, fine: np.ndarray, baseline: np.ndarray, starts: np.ndarray, length: int
    ) -> None:
        self.fine = fine
        self.baseline = baseline
        self.starts = starts
        self.length = length

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.starts[index])
        stop = start + self.length
        return (
            torch.from_numpy(self.baseline[start:stop]),
            torch.from_numpy(self.fine[start:stop]),
        )


@dataclass
class MetricSums:
    residual_loss_sum: float = 0.0
    residual_elements: int = 0
    model_sse: float = 0.0
    baseline_sse: float = 0.0
    truth_sum: float = 0.0
    truth_square_sum: float = 0.0
    physical_elements: int = 0


def reconstruct(
    baseline: torch.Tensor, prediction_scaled: torch.Tensor, residual_rms: torch.Tensor
) -> torch.Tensor:
    return baseline + prediction_scaled * residual_rms.view(1, 1, -1, 1, 1)


def update_metrics(
    sums: MetricSums,
    residual_loss: torch.Tensor,
    target_scaled: torch.Tensor,
    prediction: torch.Tensor,
    baseline: torch.Tensor,
    fine: torch.Tensor,
    residual_rms: torch.Tensor,
) -> None:
    physical = slice(0, VALID_RADIAL)
    predicted_fine = reconstruct(baseline, prediction, residual_rms)
    model_error = predicted_fine[..., physical, :] - fine[..., physical, :]
    baseline_error = baseline[..., physical, :] - fine[..., physical, :]
    truth = fine[..., physical, :]
    sums.residual_loss_sum += float(residual_loss) * target_scaled[..., physical, :].numel()
    sums.residual_elements += target_scaled[..., physical, :].numel()
    sums.model_sse += float(torch.sum(model_error.float() ** 2))
    sums.baseline_sse += float(torch.sum(baseline_error.float() ** 2))
    sums.truth_sum += float(torch.sum(truth.float()))
    sums.truth_square_sum += float(torch.sum(truth.float() ** 2))
    sums.physical_elements += truth.numel()


def finalize_metrics(sums: MetricSums) -> dict[str, float]:
    model_mse = sums.model_sse / sums.physical_elements
    baseline_mse = sums.baseline_sse / sums.physical_elements
    truth_mean = sums.truth_sum / sums.physical_elements
    truth_variance = max(
        sums.truth_square_sum / sums.physical_elements - truth_mean**2, 0.0
    )
    return {
        "residual_scaled_mse": sums.residual_loss_sum / sums.residual_elements,
        "reconstruction_mse": model_mse,
        "baseline_interpolation_mse": baseline_mse,
        "reconstruction_rmse": math.sqrt(model_mse),
        "baseline_interpolation_rmse": math.sqrt(baseline_mse),
        "reconstruction_nrmse_std": math.sqrt(model_mse)
        / max(math.sqrt(truth_variance), np.finfo(float).tiny),
        "skill_over_interpolation": 1.0 - model_mse / baseline_mse,
    }


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    residual_rms: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
    amp_enabled: bool,
    max_batches: int,
    log_interval: int = 20,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums = MetricSums()
    started = time.perf_counter()
    for batch_index, (baseline, fine) in enumerate(loader, start=1):
        if max_batches and batch_index > max_batches:
            break
        baseline = baseline.to(device, non_blocking=True)
        fine = fine.to(device, non_blocking=True)
        target_scaled = (fine - baseline) / residual_rms.view(1, 1, -1, 1, 1)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                prediction = model(baseline)
                loss = torch.mean(
                    (prediction[..., :VALID_RADIAL, :] - target_scaled[..., :VALID_RADIAL, :])
                    ** 2
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at batch {batch_index}")
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
        with torch.no_grad():
            update_metrics(
                sums, loss, target_scaled, prediction, baseline, fine, residual_rms
            )
        if training and batch_index % log_interval == 0:
            print(
                f"  batch={batch_index}/{len(loader)} residual_mse={float(loss):.6g}",
                flush=True,
            )
    metrics = finalize_metrics(sums)
    metrics["elapsed_sec"] = time.perf_counter() - started
    metrics["batches"] = min(len(loader), max_batches or len(loader))
    return metrics


def save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def append_history(path: Path, row: dict[str, float | int]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    coarse_size = args.coarse_size or 256 // args.grid_factor
    effective_factor = 256.0 / coarse_size
    grid_label = f"C{coarse_size}" if args.coarse_size else f"G{args.grid_factor}"
    if args.output_dir is None:
        args.output_dir = (
            ROOT
            / "workdirs"
            / f"radaz_e25_{grid_label.lower()}_simvp_residual_sr_sync10_20to30us"
        )
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    amp_enabled = device.type == "cuda" and not args.no_amp

    print(f"[data] reading {args.h5}", flush=True)
    with h5py.File(args.h5, "r") as handle:
        times_us = np.asarray(handle["time_s"], dtype=np.float64) * 1.0e6
        selected = np.flatnonzero(
            (times_us >= args.start_us - 1.0e-9)
            & (times_us <= args.end_us + 1.0e-9)
        )
        if not len(selected) or not np.all(np.diff(selected) == 1):
            raise ValueError("Requested time range is empty or non-contiguous")
        fine = np.asarray(
            handle["data_tchw"][int(selected[0]) : int(selected[-1]) + 1],
            dtype=np.float32,
        )
        selected_times_us = times_us[selected]
        valid_shape = tuple(np.asarray(handle["valid_spatial_shape"], dtype=int))
        model_shape = tuple(np.asarray(handle["model_spatial_shape"], dtype=int))
    if valid_shape != (257, 256) or model_shape != (260, 256):
        raise ValueError(f"Unexpected valid/model shapes: {valid_shape}, {model_shape}")
    print(
        f"[data] selected {len(fine)} frames; constructing {grid_label} interpolation",
        flush=True,
    )
    baseline = make_coarse_size_interpolated(fine, coarse_size) if args.coarse_size else make_grid_interpolated(fine, args.grid_factor)

    frame_count = len(fine)
    train_end = int(math.floor(0.8 * frame_count))
    val_end = int(math.floor(0.9 * frame_count))
    split_bounds = {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, frame_count)}
    starts = {
        name: segment_starts(begin, end, args.sequence_length, args.window_hop)
        for name, (begin, end) in split_bounds.items()
    }
    residual_rms_np = compute_residual_rms(fine, baseline, train_end)
    residual_rms = torch.from_numpy(residual_rms_np).to(device)

    loaders = {}
    for name in ("train", "val", "test"):
        dataset = SynchronousResidualDataset(
            fine, baseline, starts[name], args.sequence_length
        )
        generator = torch.Generator().manual_seed(args.seed)
        loaders[name] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            generator=generator if name == "train" else None,
        )

    model_kwargs = {
        "in_shape": (args.sequence_length, 3, 260, 256),
        "hid_S": args.hid_s,
        "hid_T": args.hid_t,
        "N_S": args.n_s,
        "N_T": args.n_t,
        "model_type": "gSTA",
        "spatio_kernel_enc": 3,
        "spatio_kernel_dec": 3,
        "aft_seq_length": args.sequence_length,
        "simvp_direct_aft_seq": True,
    }
    model = SimVP_Model(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.05
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    metadata = {
        "experiment": f"E25 {grid_label} synchronous residual super-resolution",
        "grid_factor": args.grid_factor if args.coarse_size is None else effective_factor,
        "grid_label": grid_label,
        "coarse_size": coarse_size,
        "effective_grid_factor": effective_factor,
        "coarse_spatial_shape": [coarse_size, coarse_size],
        "G_definition": (
            (f"grid-coarsening factor; G{args.grid_factor} = non-overlapping {args.grid_factor}x{args.grid_factor} mean")
            if args.coarse_size is None else
            f"C{coarse_size} = conservative cell-overlap average; effective factor {effective_factor:.6g}"
        ),
        "temporal_task": "same-time sequence reconstruction, not future forecasting",
        "source_h5": str(args.h5.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "selected_source_indices": [int(selected[0]), int(selected[-1])],
        "selected_time_us": [float(selected_times_us[0]), float(selected_times_us[-1])],
        "frame_count": frame_count,
        "split_bounds_local_end_exclusive": split_bounds,
        "windows": {name: len(value) for name, value in starts.items()},
        "sequence_length": args.sequence_length,
        "frame_stride": 1,
        "window_hop": args.window_hop,
        "residual_rms_train_only": dict(zip(CHANNEL_NAMES, residual_rms_np.tolist())),
        "model_kwargs": {key: list(value) if isinstance(value, tuple) else value for key, value in model_kwargs.items()},
        "parameter_count": parameter_count,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[setup] windows={metadata['windows']} residual_rms={residual_rms_np.tolist()} "
        f"parameters={parameter_count:,} amp={amp_enabled}",
        flush=True,
    )

    start_epoch = 1
    best_val = math.inf
    latest_path = args.output_dir / "checkpoint_latest.pth"
    if args.resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint["best_val_reconstruction_mse"])
        print(f"[resume] continuing from epoch {start_epoch}", flush=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history_path = args.output_dir / "history.csv"
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            loaders["train"],
            device,
            residual_rms,
            optimizer,
            scaler,
            amp_enabled,
            args.max_train_batches,
        )
        with torch.inference_mode():
            val_metrics = run_epoch(
                model,
                loaders["val"],
                device,
                residual_rms,
                None,
                scaler,
                amp_enabled,
                args.max_val_batches,
            )
        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        append_history(history_path, row)
        improved = val_metrics["reconstruction_mse"] < best_val
        if improved:
            best_val = val_metrics["reconstruction_mse"]
        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_reconstruction_mse": best_val,
            "residual_rms": residual_rms_np,
            "model_kwargs": model_kwargs,
            "metadata": metadata,
        }
        save_checkpoint(latest_path, payload)
        if improved:
            save_checkpoint(args.output_dir / "checkpoint_best.pth", payload)
        print(
            f"[epoch {epoch:03d}/{args.epochs}] "
            f"train_mse={train_metrics['reconstruction_mse']:.6g} "
            f"val_mse={val_metrics['reconstruction_mse']:.6g} "
            f"baseline={val_metrics['baseline_interpolation_mse']:.6g} "
            f"skill={val_metrics['skill_over_interpolation']:.4f} "
            f"sec={train_metrics['elapsed_sec'] + val_metrics['elapsed_sec']:.1f}",
            flush=True,
        )

    best_checkpoint = torch.load(
        args.output_dir / "checkpoint_best.pth", map_location="cpu"
    )
    model.load_state_dict(best_checkpoint["model"])
    with torch.inference_mode():
        test_metrics = run_epoch(
            model,
            loaders["test"],
            device,
            residual_rms,
            None,
            scaler,
            amp_enabled,
            args.max_val_batches,
        )
    summary = {
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_val_reconstruction_mse": float(
            best_checkpoint["best_val_reconstruction_mse"]
        ),
        "test": test_metrics,
        "peak_cuda_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024.0**2)
            if device.type == "cuda"
            else 0.0
        ),
    }
    (args.output_dir / "final_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[complete] {json.dumps(summary, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
