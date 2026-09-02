#!/usr/bin/env python3
"""Evaluate RadAz zero-shot direct predictions against a copy baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CHANNELS = ("electron_den", "ion_den", "phi")
LABELS = {
    "electron_den": "Electron density",
    "ion_den": "Ion density",
    "phi": "Potential phi",
}
MSE_UNITS = {
    "electron_den": r"MSE [$\mathrm{m}^{-6}$]",
    "ion_den": r"MSE [$\mathrm{m}^{-6}$]",
    "phi": r"MSE [$\mathrm{V}^{2}$]",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_csv", type=Path)
    parser.add_argument("target_h5", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-label", default="Bx=20 mT, Ez=10 kV/m")
    parser.add_argument("--target-label", default="Bx=20 mT, Ez=20 kV/m")
    parser.add_argument(
        "--mse-already-physical",
        action="store_true",
        help="Treat MSE columns in the raw CSV as physical-unit errors.",
    )
    return parser.parse_args()


def finite(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values.copy()
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def grouped_means(
    rows: list[dict], channel: str, group_key: str, scale: float
) -> list[dict]:
    groups: dict[float, list[dict]] = {}
    for row in rows:
        if row["channel"] == channel:
            groups.setdefault(float(row[group_key]), []).append(row)

    output = []
    for key, group in sorted(groups.items()):
        model = finite([float(row["model_mse"]) * scale * scale for row in group])
        copy = finite([float(row["copy_mse"]) * scale * scale for row in group])
        ratios = finite([float(row["model_over_copy"]) for row in group])
        output.append(
            {
                group_key: key,
                "channel": channel,
                "n": len(group),
                "model_mse_mean": float(np.mean(model)),
                "copy_mse_mean": float(np.mean(copy)),
                "ratio_of_mean_mse": float(np.mean(model) / np.mean(copy)),
                "model_over_copy_median": float(np.median(ratios)),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    raw_csv = args.raw_csv.resolve()
    target_h5 = args.target_h5.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with raw_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No prediction rows in {raw_csv}")

    with h5py.File(target_h5, "r") as handle:
        low = np.asarray(handle["norm_low"], dtype=np.float64)
        high = np.asarray(handle["norm_high"], dtype=np.float64)
        clipped_low = np.asarray(handle["clipped_low_count"], dtype=np.int64)
        clipped_high = np.asarray(handle["clipped_high_count"], dtype=np.int64)
        counts = np.asarray(handle["normalization_value_count"], dtype=np.int64)
        normalization_source = handle["normalization_source"][()].decode()
    scales = high - low
    metric_scales = np.ones_like(scales) if args.mse_already_physical else scales

    overall_rows = []
    for index, channel in enumerate(CHANNELS):
        group = [row for row in rows if row["channel"] == channel]
        model = finite(
            [float(row["model_mse"]) * metric_scales[index] ** 2 for row in group]
        )
        copy = finite(
            [float(row["copy_mse"]) * metric_scales[index] ** 2 for row in group]
        )
        ratios = finite([float(row["model_over_copy"]) for row in group])
        correlations = finite([float(row["corr"]) for row in group])
        copy_correlations = finite([float(row["copy_corr"]) for row in group])
        overall_rows.append(
            {
                "channel": channel,
                "samples": len(group),
                "model_mse_mean": float(np.mean(model)),
                "copy_mse_mean": float(np.mean(copy)),
                "ratio_of_mean_mse": float(np.mean(model) / np.mean(copy)),
                "model_over_copy_median": float(np.median(ratios)),
                "model_correlation_mean": float(np.mean(correlations)),
                "copy_correlation_mean": float(np.mean(copy_correlations)),
                "clipped_low_fraction": float(clipped_low[index] / counts[index]),
                "clipped_high_fraction": float(clipped_high[index] / counts[index]),
            }
        )
    write_csv(output_dir / "zero_shot_overall_summary.csv", overall_rows)

    horizon_rows = []
    for index, channel in enumerate(CHANNELS):
        horizon_rows.extend(
            grouped_means(rows, channel, "horizon_ns", metric_scales[index])
        )
    write_csv(output_dir / "zero_shot_metrics_by_horizon.csv", horizon_rows)

    phi_time_rows = grouped_means(
        rows, "phi", "target_time_us", metric_scales[CHANNELS.index("phi")]
    )
    write_csv(output_dir / "zero_shot_phi_by_target_time.csv", phi_time_rows)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    for axis, channel in zip(axes, CHANNELS):
        channel_rows = [row for row in horizon_rows if row["channel"] == channel]
        horizon = np.asarray([row["horizon_ns"] for row in channel_rows])
        model = np.asarray([row["model_mse_mean"] for row in channel_rows])
        copy = np.asarray([row["copy_mse_mean"] for row in channel_rows])
        axis.plot(horizon, model, marker="o", label="zero-shot model")
        axis.plot(
            horizon,
            copy,
            marker="x",
            linestyle="--",
            color="#737373",
            label="copy baseline",
        )
        axis.set_yscale("log")
        axis.set_title(LABELS[channel])
        axis.set_xlabel("Direct prediction horizon [ns]")
        axis.set_ylabel(MSE_UNITS[channel])
        axis.legend(loc="lower right")
    fig.suptitle(
        f"Zero-shot direct10: {args.source_label} to {args.target_label}"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "zero_shot_model_vs_copy_by_horizon.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9.5, 5.5))
    for channel in CHANNELS:
        channel_rows = [row for row in horizon_rows if row["channel"] == channel]
        horizon = np.asarray([row["horizon_ns"] for row in channel_rows])
        ratio = np.asarray([row["ratio_of_mean_mse"] for row in channel_rows])
        axis.plot(horizon, ratio, marker="o", label=LABELS[channel])
    axis.axhline(1.0, color="black", linestyle=":", label="equal to copy")
    axis.set_xlabel("Direct prediction horizon [ns]")
    axis.set_ylabel("Model MSE / copy MSE")
    axis.set_yscale("log")
    axis.legend(loc="lower right")
    axis.set_title("Values below 1 mean the zero-shot model beats copy")
    fig.tight_layout()
    fig.savefig(output_dir / "zero_shot_model_over_copy_by_horizon.png", dpi=180)
    plt.close(fig)

    target_time = np.asarray(
        [row["target_time_us"] for row in phi_time_rows], dtype=np.float64
    )
    phi_model = np.asarray(
        [row["model_mse_mean"] for row in phi_time_rows], dtype=np.float64
    )
    phi_copy = np.asarray(
        [row["copy_mse_mean"] for row in phi_time_rows], dtype=np.float64
    )
    fig, axis = plt.subplots(figsize=(12, 6))
    for label, values, color in (
        ("zero-shot model", phi_model, "#2563eb"),
        ("copy baseline", phi_copy, "#6b7280"),
    ):
        axis.plot(target_time, values, color=color, alpha=0.16, linewidth=0.7)
        axis.plot(
            target_time,
            rolling_mean(values, 11),
            color=color,
            linewidth=1.8,
            label=f"{label} (11-point mean)",
        )
    axis.set_yscale("log")
    axis.set_xlabel("Target simulation time [us]")
    axis.set_ylabel(r"phi MSE [$\mathrm{V}^{2}$]")
    axis.set_title("Zero-shot phi error over the target simulation")
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "zero_shot_phi_mse_by_target_time.png", dpi=180)
    plt.close(fig)

    summary = {
        "status": "PASS",
        "source_label": args.source_label,
        "target_label": args.target_label,
        "raw_predictions_csv": str(raw_csv),
        "target_h5": str(target_h5),
        "normalization_source": normalization_source,
        "teacher_forced_direct_prediction": True,
        "input_frames": 10,
        "output_frames": 10,
        "frame_interval_ns": 15.0,
        "mse_space": (
            "physical_unclipped"
            if args.mse_already_physical
            else "denormalized_from_model_input_h5"
        ),
        "windows": len(rows) // (len(CHANNELS) * 10),
        "overall": {row["channel"]: row for row in overall_rows},
    }
    (output_dir / "zero_shot_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    table = "\n".join(
        "| "
        f"{row['channel']} | {row['model_mse_mean']:.4e} | "
        f"{row['copy_mse_mean']:.4e} | {row['ratio_of_mean_mse']:.4f} | "
        f"{row['model_correlation_mean']:.4f} | "
        f"{100.0 * row['clipped_high_fraction']:.3f}% |"
        for row in overall_rows
    )
    readme = f"""# RadAz zero-shot transfer

Source: `{args.source_label}`

Target: `{args.target_label}`

The source direct10 model is applied without retraining. Every prediction uses
10 true target PIC frames and predicts the next 10 frames at 15 ns intervals.
This is teacher-forced direct prediction, not rollout.

| Channel | Model MSE | Copy MSE | Model/Copy | Model correlation | Above source norm range |
|---|---:|---:|---:|---:|---:|
{table}

`Model/Copy < 1` means the model beats the copy baseline.

## 日本語

学習元の正規化範囲と学習済み重みを固定した、再学習なしの
ゼロショット転移です。正規化範囲を超えた転移先データはモデル入力で
0または1にクリップされるため、その割合も表に記録しています。

## Files

- `zero_shot_overall_summary.csv`
- `zero_shot_metrics_by_horizon.csv`
- `zero_shot_phi_by_target_time.csv`
- `zero_shot_model_vs_copy_by_horizon.png`
- `zero_shot_model_over_copy_by_horizon.png`
- `zero_shot_phi_mse_by_target_time.png`
- `zero_shot_evaluation_summary.json`
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[PASS] output={output_dir}")


if __name__ == "__main__":
    main()
