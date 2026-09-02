#!/usr/bin/env python3
"""Compare data and azimuthal-mode errors for the spectral-loss ablation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
import torch

from openstl.methods.pepapic_spectral_loss import PEPAPICSpectralLoss

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA_H5 = (
    ROOT.parent
    / "PEPAPIC"
    / "test"
    / "results"
    / "2D_Landmark"
    / "2D_RadAz_Xe1p_Bx20mT_Ez10kVm_dt15ps_out15ns"
    / "2D_RadAz_Xe1p_Bx20mT_Ez10kVm_dt15ps_out15ns"
    / "SimVPv2_inputs"
    / "radaz_3ch_trainfixed_margin20_native257x256_pad260x256.h5"
)
PREFIX = "radaz_xe1p_bx20mt_ez10kvm_out15ns_spectral_ablation"
EXPERIMENTS = [
    ("data-only", f"{PREFIX}_baseline_5ep"),
    ("amplitude", f"{PREFIX}_amplitude_5ep"),
    ("amplitude + phase + cross", f"{PREFIX}_amplitude_phase_cross_5ep"),
]
OUTPUT_DIR = ROOT / "workdirs" / "compare_radaz_spectral_loss_ablation"
CHANNELS = ("electron_den", "ion_den", "phi")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def load_arrays(ex_name):
    saved = ROOT / "workdirs" / ex_name / "saved"
    pred_path = saved / "preds.npy"
    true_path = saved / "trues.npy"
    if not pred_path.exists() or not true_path.exists():
        raise FileNotFoundError(f"Missing saved predictions for {ex_name}")
    return np.load(pred_path, mmap_mode="r"), np.load(true_path, mmap_mode="r")


def spectral_means(pred, true, module, device, batch_size):
    totals = {"amplitude": 0.0, "phase": 0.0, "cross_phase": 0.0}
    count = 0
    with torch.no_grad():
        for start in range(0, pred.shape[0], batch_size):
            end = min(start + batch_size, pred.shape[0])
            pred_batch = torch.as_tensor(
                np.array(pred[start:end], copy=True), dtype=torch.float32, device=device
            )
            true_batch = torch.as_tensor(
                np.array(true[start:end], copy=True), dtype=torch.float32, device=device
            )
            losses = module(pred_batch, true_batch)
            batch_count = end - start
            for name, value in losses.items():
                totals[name] += float(value.cpu()) * batch_count
            count += batch_count
    return {name: value / count for name, value in totals.items()}


def standard_mse_metrics(pred, true, batch_size):
    total_sum = 0.0
    channel_sum = np.zeros(pred.shape[2], dtype=np.float64)
    horizon_sum = np.zeros(pred.shape[1], dtype=np.float64)
    for start in range(0, pred.shape[0], batch_size):
        end = min(start + batch_size, pred.shape[0])
        difference = (
            np.asarray(pred[start:end], dtype=np.float32)
            - np.asarray(true[start:end], dtype=np.float32)
        )
        squared = difference * difference
        total_sum += float(np.sum(squared, dtype=np.float64))
        channel_sum += np.sum(squared, axis=(0, 1, 3, 4), dtype=np.float64)
        horizon_sum += np.sum(squared, axis=(0, 2, 3, 4), dtype=np.float64)

    n, t, c, h, w = pred.shape
    return (
        total_sum / float(n * t * c * h * w),
        channel_sum / float(n * t * h * w),
        horizon_sum / float(n * c * h * w),
    )


def main():
    args = parse_args()
    device = torch.device(args.device)
    module = PEPAPICSpectralLoss(DATA_H5).to(device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    horizon_curves = {}
    for label, ex_name in EXPERIMENTS:
        pred, true = load_arrays(ex_name)
        if pred.shape != true.shape:
            raise ValueError(f"{label}: prediction/true shape mismatch")
        normalized_mse, channel_mse, horizon_mse = standard_mse_metrics(
            pred, true, max(1, args.batch_size)
        )
        spectral = spectral_means(
            pred, true, module, device, max(1, args.batch_size)
        )
        row = {
            "model": label,
            "normalized_mse": normalized_mse,
            **{
                f"normalized_mse_{channel}": float(value)
                for channel, value in zip(CHANNELS, channel_mse)
            },
            "spectral_amplitude_loss": spectral["amplitude"],
            "spectral_phase_increment_loss": spectral["phase"],
            "spectral_cross_phase_loss": spectral["cross_phase"],
        }
        rows.append(row)
        horizon_curves[label] = horizon_mse
        print(row)

    csv_path = OUTPUT_DIR / "radaz_spectral_ablation_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metrics = [
        ("normalized_mse", "Normalized data MSE"),
        ("spectral_amplitude_loss", "Azimuthal amplitude loss"),
        ("spectral_phase_increment_loss", "Phase-increment loss"),
        ("spectral_cross_phase_loss", "Density-Ey cross-phase loss"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = ("#4C78A8", "#F58518", "#54A24B")
    labels = [row["model"] for row in rows]
    for axis, (key, title) in zip(axes.ravel(), metrics):
        values = [row[key] for row in rows]
        axis.bar(labels, values, color=colors)
        axis.set_title(title)
        axis.set_ylabel("error")
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=12)
    fig.savefig(
        OUTPUT_DIR / "radaz_spectral_ablation_summary.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    horizon_ns = 15.0 * np.arange(1, 11)
    fig, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for color, (label, values) in zip(colors, horizon_curves.items()):
        axis.plot(horizon_ns, values, marker="o", color=color, label=label)
    axis.set_xlabel("Forecast horizon [ns]")
    axis.set_ylabel("Normalized MSE")
    axis.set_title("RadAz direct10 error by forecast horizon")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    fig.savefig(
        OUTPUT_DIR / "radaz_spectral_ablation_mse_by_horizon.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"saved={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
