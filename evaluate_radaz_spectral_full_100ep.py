#!/usr/bin/env python3
"""Compare 100-epoch data-only and full spectral-loss RadAz models."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch

from evaluate_radaz_spectral_loss_ablation import (
    spectral_means,
    standard_mse_metrics,
)
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
EXPERIMENTS = [
    (
        "data-only 100ep",
        "radaz_xe1p_bx20mt_ez10kvm_out15ns_native257x256_"
        "direct10_trainfixed_disjoint_811_bs1_100ep",
    ),
    (
        "spectral-full 100ep",
        "radaz_xe1p_bx20mt_ez10kvm_out15ns_native257x256_direct10_"
        "spectral_full_trainfixed_disjoint_811_bs1_100ep",
    ),
]
OUTPUT_DIR = ROOT / "workdirs" / "compare_radaz_spectral_full_100ep"
CHANNELS = ("electron_den", "ion_den", "phi")
B_T = 0.020
MTSI_MODES = np.arange(1, 7)
ECDI_MODES = np.arange(9, 22)


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
        raise FileNotFoundError(f"Missing saved arrays for {ex_name}")
    pred = np.load(pred_path, mmap_mode="r")
    true = np.load(true_path, mmap_mode="r")
    if pred.shape != true.shape:
        raise ValueError(f"{ex_name}: prediction/true shape mismatch")
    return pred, true


def corrcoef(left, right):
    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def transport_values(normalized, low, scale, x_mask, dy_m):
    ne = (
        np.asarray(normalized[:, :, 0, : x_mask.size, :], dtype=np.float64)
        * scale[0]
        + low[0]
    )
    phi = (
        np.asarray(normalized[:, :, 2, : x_mask.size, :], dtype=np.float64)
        * scale[2]
        + low[2]
    )
    ey = -(np.roll(phi, -1, axis=-1) - np.roll(phi, 1, axis=-1)) / (
        2.0 * dy_m
    )
    ne = ne[:, :, x_mask]
    ey = ey[:, :, x_mask]
    dne = ne - np.mean(ne, axis=-1, keepdims=True)
    dey = ey - np.mean(ey, axis=-1, keepdims=True)
    total = -np.mean(dne * dey, axis=(-2, -1)) / B_T

    ne_fft = np.fft.rfft(dne, axis=-1, norm="forward")
    ey_fft = np.fft.rfft(dey, axis=-1, norm="forward")
    weights = np.full(ne_fft.shape[-1], 2.0, dtype=np.float64)
    weights[0] = 1.0
    weights[-1] = 1.0
    modal = (
        -weights[None, None, None, :]
        * np.real(ne_fft * np.conj(ey_fft))
        / B_T
    )
    modal = np.mean(modal, axis=2)
    return (
        total,
        np.sum(modal[:, :, MTSI_MODES], axis=-1),
        np.sum(modal[:, :, ECDI_MODES], axis=-1),
    )


def physical_and_transport_metrics(pred, true, batch_size):
    with h5py.File(DATA_H5, "r") as handle:
        low = np.asarray(handle["norm_low"], dtype=np.float64)
        high = np.asarray(handle["norm_high"], dtype=np.float64)
        valid_h, valid_w = (
            int(value) for value in np.asarray(handle["valid_spatial_shape"])
        )
        x_m = np.asarray(handle["x_m"], dtype=np.float64)
        y_m = np.asarray(handle["y_m"], dtype=np.float64)
    scale = high - low
    x_mask = (x_m >= 0.09e-2 - 1.0e-15) & (x_m <= 1.19e-2 + 1.0e-15)
    dy_m = float(np.median(np.diff(y_m)))

    channel_sum = np.zeros(3, dtype=np.float64)
    ey_sum = 0.0
    element_count = pred.shape[0] * pred.shape[1] * valid_h * valid_w
    pred_transport_parts = [[], [], []]
    true_transport_parts = [[], [], []]
    for start in range(0, pred.shape[0], batch_size):
        end = min(start + batch_size, pred.shape[0])
        pred_chunk = np.asarray(pred[start:end], dtype=np.float32)
        true_chunk = np.asarray(true[start:end], dtype=np.float32)
        difference = (
            pred_chunk[:, :, :, :valid_h, :valid_w]
            - true_chunk[:, :, :, :valid_h, :valid_w]
        )
        physical_difference = difference * scale[None, None, :, None, None]
        channel_sum += np.sum(
            physical_difference * physical_difference,
            axis=(0, 1, 3, 4),
            dtype=np.float64,
        )

        phi_pred = (
            pred_chunk[:, :, 2, :valid_h, :valid_w] * scale[2] + low[2]
        )
        phi_true = (
            true_chunk[:, :, 2, :valid_h, :valid_w] * scale[2] + low[2]
        )
        ey_pred = -(np.roll(phi_pred, -1, axis=-1) - np.roll(phi_pred, 1, axis=-1)) / (
            2.0 * dy_m
        )
        ey_true = -(np.roll(phi_true, -1, axis=-1) - np.roll(phi_true, 1, axis=-1)) / (
            2.0 * dy_m
        )
        ey_sum += float(np.sum((ey_pred - ey_true) ** 2, dtype=np.float64))

        pred_transport = transport_values(
            pred_chunk, low, scale, x_mask, dy_m
        )
        true_transport = transport_values(
            true_chunk, low, scale, x_mask, dy_m
        )
        for index in range(3):
            pred_transport_parts[index].append(pred_transport[index])
            true_transport_parts[index].append(true_transport[index])

    pred_transport = [
        np.concatenate(parts, axis=0) for parts in pred_transport_parts
    ]
    true_transport = [
        np.concatenate(parts, axis=0) for parts in true_transport_parts
    ]
    transport_metrics = {}
    for name, pred_values, true_values in zip(
        ("total", "mtsi", "ecdi"), pred_transport, true_transport
    ):
        transport_metrics[f"transport_{name}_mae"] = float(
            np.mean(np.abs(pred_values - true_values))
        )
        transport_metrics[f"transport_{name}_correlation"] = corrcoef(
            pred_values, true_values
        )
    return (
        channel_sum / float(element_count),
        ey_sum / float(element_count),
        transport_metrics,
    )


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def relative_change_rows(rows):
    baseline, spectral = rows
    changes = []
    for metric in baseline:
        if metric == "model":
            continue
        baseline_value = float(baseline[metric])
        spectral_value = float(spectral[metric])
        row = {
            "metric": metric,
            "data_only_100ep": baseline_value,
            "spectral_full_100ep": spectral_value,
            "absolute_change": spectral_value - baseline_value,
        }
        if "correlation" in metric or baseline_value == 0.0:
            row["relative_change_percent"] = ""
        else:
            row["relative_change_percent"] = (
                spectral_value / baseline_value - 1.0
            ) * 100.0
        changes.append(row)
    return changes


def per_window_channel_mse(pred, true, batch_size):
    values = []
    for start in range(0, pred.shape[0], batch_size):
        end = min(start + batch_size, pred.shape[0])
        difference = (
            np.asarray(pred[start:end], dtype=np.float32)
            - np.asarray(true[start:end], dtype=np.float32)
        )
        values.append(np.mean(difference * difference, axis=(1, 3, 4)))
    return np.concatenate(values, axis=0)


def main():
    args = parse_args()
    batch_size = max(1, args.batch_size)
    device = torch.device(args.device)
    spectral_module = PEPAPICSpectralLoss(DATA_H5).to(device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    horizon_curves = {}
    window_channel_curves = {}
    for label, ex_name in EXPERIMENTS:
        pred, true = load_arrays(ex_name)
        normalized_mse, channel_mse_norm, horizon_mse = standard_mse_metrics(
            pred, true, batch_size
        )
        spectral = spectral_means(
            pred, true, spectral_module, device, batch_size
        )
        channel_mse_phys, ey_mse, transport = physical_and_transport_metrics(
            pred, true, batch_size
        )
        row = {
            "model": label,
            "normalized_mse": normalized_mse,
            **{
                f"normalized_mse_{name}": float(value)
                for name, value in zip(CHANNELS, channel_mse_norm)
            },
            **{
                f"physical_mse_{name}": float(value)
                for name, value in zip(CHANNELS, channel_mse_phys)
            },
            "physical_mse_ey": ey_mse,
            "spectral_amplitude_loss": spectral["amplitude"],
            "spectral_phase_increment_loss": spectral["phase"],
            "spectral_cross_phase_loss": spectral["cross_phase"],
            **transport,
        }
        rows.append(row)
        horizon_curves[label] = horizon_mse
        window_channel_curves[label] = per_window_channel_mse(
            pred, true, batch_size
        )
        print(row)

    write_csv(OUTPUT_DIR / "spectral_full_100ep_summary.csv", rows)
    write_csv(
        OUTPUT_DIR / "spectral_full_100ep_relative_change.csv",
        relative_change_rows(rows),
    )
    colors = ("#4C78A8", "#54A24B")
    labels = [row["model"] for row in rows]
    metrics = [
        ("normalized_mse", "Normalized data MSE"),
        ("spectral_amplitude_loss", "Azimuthal amplitude loss"),
        ("spectral_phase_increment_loss", "Phase-increment loss"),
        ("spectral_cross_phase_loss", "Density-Ey cross-phase loss"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, (key, title) in zip(axes.ravel(), metrics):
        axis.bar(labels, [row[key] for row in rows], color=colors)
        axis.set_title(title)
        axis.set_ylabel("error")
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(
        OUTPUT_DIR / "spectral_full_100ep_error_summary.png",
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
    axis.set_title("100-epoch RadAz direct10 error")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    fig.savefig(
        OUTPUT_DIR / "spectral_full_100ep_mse_by_horizon.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    with h5py.File(DATA_H5, "r") as handle:
        time_s = np.asarray(handle["time_s"], dtype=np.float64)
        pre_length = int(handle["pre_seq_length"][()])
        aft_length = int(handle["aft_seq_length"][()])
    total_length = pre_length + aft_length
    test_frame_start = int(np.floor(time_s.size * 0.9))
    test_window_starts = np.arange(
        test_frame_start,
        time_s.size - total_length + 1,
        dtype=np.int64,
    )
    target_start_us = time_s[test_window_starts + pre_length] * 1.0e6
    if target_start_us.size != next(iter(window_channel_curves.values())).shape[0]:
        raise RuntimeError("test target-time count does not match saved predictions")

    window_rows = []
    for label, values in window_channel_curves.items():
        for index, target_time in enumerate(target_start_us):
            window_rows.append(
                {
                    "model": label,
                    "window_index": index,
                    "target_start_us": target_time,
                    **{
                        f"normalized_mse_{channel}": float(values[index, channel_index])
                        for channel_index, channel in enumerate(CHANNELS)
                    },
                }
            )
    write_csv(
        OUTPUT_DIR / "spectral_full_100ep_channel_mse_by_target_time.csv",
        window_rows,
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    for channel_index, (axis, channel) in enumerate(zip(axes, CHANNELS)):
        for color, (label, values) in zip(colors, window_channel_curves.items()):
            axis.plot(
                target_start_us,
                values[:, channel_index],
                color=color,
                linewidth=1.5,
                label=label,
            )
        axis.set_ylabel("Normalized MSE")
        axis.set_title(channel)
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right")
    axes[-1].set_xlabel("First target-frame time [us]")
    fig.savefig(
        OUTPUT_DIR / "spectral_full_100ep_channel_mse_by_target_time.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    for axis, transport_name in zip(axes, ("total", "mtsi", "ecdi")):
        axis.bar(
            labels,
            [row[f"transport_{transport_name}_correlation"] for row in rows],
            color=colors,
        )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(f"{transport_name.upper()} transport")
        axis.set_ylabel("prediction-truth correlation")
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(
        OUTPUT_DIR / "spectral_full_100ep_transport_correlation.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"saved={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
