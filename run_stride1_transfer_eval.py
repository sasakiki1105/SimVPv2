import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
WORKDIRS = ROOT / "workdirs"
PYTHON = Path(r"C:\Users\astro\anaconda3\envs\OpenSTL\python.exe")
OUTDIR = WORKDIRS / "compare_transfer_stride1_from_high3b_stride1"

MODEL = {
    "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_trainfixed_disjoint_811_bs2_100ep",
    "config": ROOT / "configs/custom/pepapic/SimVP_gSTA_pepapic.py",
}

DATASETS = [
    {
        "key": "low_magnet_3a",
        "label": "Low magnet 3a, stride1",
        "h5": Path(
            r"C:\Users\astro\research\PEPAPIC\test\results"
            r"\2D_ExB_low_magnet_dt2.5e-9_maxt50e-6_macro5"
            r"\SimVPv2_inputs\global_norm_from_high3b_stride1_trainfixed_minmax_margin20_t0_to_t4000_training_compatible.h5"
        ),
        "base_dt_ns": 12.5,
        "pred_dir": WORKDIRS / "transfer_low_magnet_stride1_direct10_from_high3b_stride1norm_training_compatible",
    },
    {
        "key": "exhigh_fine_5us",
        "label": "Ex-high magnet 5 us, stride1",
        "h5": Path(
            r"C:\Users\astro\research\PEPAPIC\test\results"
            r"\2D_ExB_exhigh_magnet_dt2.5e-10_maxt5e-6_macro5"
            r"\SimVPv2_inputs\global_norm_from_high3b_stride1_trainfixed_minmax_margin20_t0_to_t4000_training_compatible.h5"
        ),
        "base_dt_ns": 1.25,
        "pred_dir": WORKDIRS / "transfer_exhigh_fine_stride1_direct10_from_high3b_stride1norm_training_compatible",
    },
    {
        "key": "exhigh_50us",
        "label": "Ex-high magnet 50 us, stride1",
        "h5": Path(
            r"C:\Users\astro\research\PEPAPIC\test\results"
            r"\2D_ExB_exhigh_magnet_dt2.5e-9_maxt50e-6_macro5"
            r"\SimVPv2_inputs\global_norm_from_high3b_stride1_trainfixed_minmax_margin20_t0_to_t4000_training_compatible.h5"
        ),
        "base_dt_ns": 12.5,
        "pred_dir": WORKDIRS / "transfer_exhigh_50us_stride1_direct10_from_high3b_stride1norm_training_compatible",
    },
]

CHANNELS = ["electron_den", "ion_den", "phi"]


def raw_csv_path(pred_dir):
    return pred_dir / "low_magnet_direct10_raw_predictions.csv"


def run_prediction(dataset, device, batch_size, force):
    csv_path = raw_csv_path(dataset["pred_dir"])
    if csv_path.exists() and not force:
        print(f"[SKIP] {csv_path}", flush=True)
        return
    cmd = [
        str(PYTHON),
        "predict_low_magnet_transfer_stride2.py",
        "--h5",
        str(dataset["h5"]),
        "--workdir",
        str(MODEL["workdir"]),
        "--config",
        str(MODEL["config"]),
        "--outdir",
        str(dataset["pred_dir"]),
        "--device",
        device,
        "--batch-size",
        str(batch_size),
        "--base-dt-ns",
        str(dataset["base_dt_ns"]),
    ]
    print(f"[RUN] {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"prediction failed for {dataset['key']}: {result.returncode}")


def read_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] {path}", flush=True)


def finite_stats(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
    }


def summarize_dataset(dataset):
    rows = read_rows(raw_csv_path(dataset["pred_dir"]))
    summary = []
    for channel in CHANNELS:
        group = [row for row in rows if row["channel"] == channel]
        model_mse = finite_stats([float(row["model_mse"]) for row in group])
        copy_mse = finite_stats([float(row["copy_mse"]) for row in group])
        model_corr = finite_stats([float(row["corr"]) for row in group])
        copy_corr = finite_stats([float(row["copy_corr"]) for row in group])
        summary.append({
            "dataset": dataset["key"],
            "label": dataset["label"],
            "channel": channel,
            "n": model_mse["n"],
            "model_mse_mean": model_mse["mean"],
            "model_mse_median": model_mse["median"],
            "model_mse_q25": model_mse["q25"],
            "model_mse_q75": model_mse["q75"],
            "copy_mse_mean": copy_mse["mean"],
            "copy_mse_median": copy_mse["median"],
            "copy_mse_q25": copy_mse["q25"],
            "copy_mse_q75": copy_mse["q75"],
            "model_over_copy_mean": model_mse["mean"] / copy_mse["mean"] if copy_mse["mean"] > 0 else np.nan,
            "model_corr_median": model_corr["median"],
            "copy_corr_median": copy_corr["median"],
        })
    return rows, summary


def group_phi_by(rows, key, value_key):
    groups = {}
    for row in rows:
        if row["channel"] != "phi":
            continue
        groups.setdefault(float(row[key]), []).append(float(row[value_key]))
    return {x: finite_stats(v) for x, v in sorted(groups.items())}


def rolling_mean(y, window):
    if len(y) < window:
        return y.copy()
    left = window // 2
    right = window - 1 - left
    padded = np.pad(y, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def plot_channel_summary(summary_rows):
    fig, ax = plt.subplots(figsize=(11.8, 6.2))
    labels = [dataset["key"] for dataset in DATASETS]
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.22
    for ci, channel in enumerate(CHANNELS):
        model_vals = []
        copy_vals = []
        for dataset in DATASETS:
            row = next(r for r in summary_rows if r["dataset"] == dataset["key"] and r["channel"] == channel)
            model_vals.append(row["model_mse_mean"])
            copy_vals.append(row["copy_mse_mean"])
        offset = (ci - 1) * width
        ax.bar(x + offset - width * 0.28, copy_vals, width=width * 0.5, color="#9ca3af", alpha=0.85, label="copy" if ci == 0 else None)
        ax.bar(x + offset + width * 0.28, model_vals, width=width * 0.5, alpha=0.9, label=f"model {channel}")
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.set_ylabel("Mean MSE (3b stride1 normalized)")
    ax.set_title("3b stride1 model transferred to stride1 datasets")
    ax.set_yscale("log")
    ax.grid(True, axis="y", which="both", linestyle=":", alpha=0.55)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = OUTDIR / "stride1_transfer_channel_mse_mean.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f"[PLOT] {path}", flush=True)


def plot_phi_time(dataset, rows):
    model = group_phi_by(rows, "target_time_us", "model_mse")
    copy = group_phi_by(rows, "target_time_us", "copy_mse")
    fig, ax = plt.subplots(figsize=(11.2, 6.0))
    for label, grouped, color in [
        ("Copy baseline", copy, "#6b7280"),
        ("3b stride1 model", model, "#2563eb"),
    ]:
        x = np.asarray(list(grouped.keys()), dtype=np.float64)
        y = np.asarray([grouped[v]["mean"] for v in x], dtype=np.float64)
        ax.plot(x, y, color=color, alpha=0.18, linewidth=0.7)
        ax.plot(x, rolling_mean(y, 11), color=color, linewidth=1.9, label=f"{label} (11-point mean)")
    ax.set_xlabel("Target simulation time (us)")
    ax.set_ylabel("Mean phi MSE (3b stride1 normalized)")
    ax.set_title(f"{dataset['label']}: phi error over time")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.55)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = OUTDIR / f"stride1_transfer_phi_mse_target_time_{dataset['key']}.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f"[PLOT] {path}", flush=True)


def plot_phi_horizon(dataset, rows):
    model = group_phi_by(rows, "horizon_ns", "model_mse")
    copy = group_phi_by(rows, "horizon_ns", "copy_mse")
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    for label, grouped, color in [
        ("Copy baseline", copy, "#6b7280"),
        ("3b stride1 model", model, "#2563eb"),
    ]:
        x = np.asarray(list(grouped.keys()), dtype=np.float64)
        y = np.asarray([grouped[v]["mean"] for v in x], dtype=np.float64)
        ax.plot(x, y, marker="o", color=color, linewidth=1.8, label=label)
    ax.set_xlabel("Prediction horizon from last input frame (ns)")
    ax.set_ylabel("Mean phi MSE (3b stride1 normalized)")
    ax.set_title(f"{dataset['label']}: phi error by horizon")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.55)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = OUTDIR / f"stride1_transfer_phi_mse_by_horizon_{dataset['key']}.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f"[PLOT] {path}", flush=True)


def write_readme(summary_rows):
    text = """# Stride1 Transfer From High-Magnet 3b

This folder evaluates the high-magnet 3b stride1 direct10 data-only model on other testcase datasets that are also retained at stride1.

Important interpretation:

- The source model was trained on 3b raw frames with no downsampling.
- Low magnet 3a and ex-high 50us stride1 both correspond to 12.5 ns between retained frames.
- Ex-high 5us stride1 corresponds to 1.25 ns between retained frames, because that PIC run used a smaller output interval.
- Inputs are true PIC windows, shifted by one frame. This is teacher-forced direct prediction, not rollout.

Main files:

- `stride1_transfer_summary.csv`
- `stride1_transfer_channel_mse_mean.png`
- `stride1_transfer_phi_mse_target_time_*.png`
- `stride1_transfer_phi_mse_by_horizon_*.png`

Japanese note:

これは「3b stride1で学習したモデルを、他test caseでもstride1の入力列に適用したらどうなるか」を見る実験です。
モデルは時間間隔を入力として受け取っていないので、特にex-high 5usのようにstride1の物理時間が1.25 nsになるケースでは、同じstride1でも物理的な意味が3bとは違います。
"""
    path = OUTDIR / "README.md"
    path.write_text(text, encoding="utf-8")
    print(f"[README] {path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_predict:
        for dataset in DATASETS:
            run_prediction(dataset, args.device, args.batch_size, args.force)

    all_summary = []
    for dataset in DATASETS:
        rows, summary = summarize_dataset(dataset)
        all_summary.extend(summary)
        plot_phi_time(dataset, rows)
        plot_phi_horizon(dataset, rows)

    write_csv(all_summary, OUTDIR / "stride1_transfer_summary.csv")
    plot_channel_summary(all_summary)
    write_readme(all_summary)

    meta = {
        "source_model": str(MODEL["workdir"]),
        "source_config": str(MODEL["config"]),
        "datasets": [
            {key: str(value) if isinstance(value, Path) else value for key, value in dataset.items()}
            for dataset in DATASETS
        ],
    }
    (OUTDIR / "stride1_transfer_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
