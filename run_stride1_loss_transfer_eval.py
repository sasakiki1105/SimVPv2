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

CHANNELS = ["electron_den", "ion_den", "phi"]
COLORS = {
    "copy": "#6b7280",
    "data_only": "#2563eb",
    "poisson_zero": "#16a34a",
    "floor_hinge": "#dc2626",
    "efield": "#7c3aed",
    "poisson_zero_efield": "#ea580c",
}
LABELS = {
    "copy": "Copy baseline",
    "data_only": "Data-only",
    "poisson_zero": "Weak Poisson",
    "floor_hinge": "Poisson floor hinge",
    "efield": "Weak E-field",
    "poisson_zero_efield": "Weak Poisson + E-field",
}

MODELS = [
    {
        "key": "data_only",
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_trainfixed_disjoint_811_bs2_100ep",
        "config": ROOT / "configs/custom/pepapic/SimVP_gSTA_pepapic.py",
        "out_tag": "data_only",
    },
    {
        "key": "poisson_zero",
        "workdir": WORKDIRS / (
            "pepapic_simvp_gsta_highmag_macro5_direct10_"
            "poisson_zero_lam1em3_trainfixed_disjoint_811_bs2_100ep"
        ),
        "config": ROOT / "configs/custom/pepapic/SimVP_gSTA_pepapic_direct_poisson_zero_weak.py",
        "out_tag": "poisson_zero_lam1em3",
    },
    {
        "key": "floor_hinge",
        "workdir": WORKDIRS / (
            "pepapic_simvp_gsta_highmag_macro5_direct10_"
            "poisson_floor_hinge_lam1em3_floor086_alpha11_trainfixed_disjoint_811_bs2_100ep"
        ),
        "config": ROOT / "configs/custom/pepapic/SimVP_gSTA_pepapic_direct_poisson_floor_hinge.py",
        "out_tag": "floor_hinge_lam1em3_alpha11",
    },
    {
        "key": "efield",
        "workdir": WORKDIRS / (
            "pepapic_simvp_gsta_highmag_macro5_direct10_"
            "efield_lam1em3_trainfixed_disjoint_811_bs2_100ep"
        ),
        "config": ROOT / "configs/custom/pepapic/SimVP_gSTA_pepapic_direct_efield_weak.py",
        "out_tag": "efield_lam1em3",
    },
    {
        "key": "poisson_zero_efield",
        "workdir": WORKDIRS / (
            "pepapic_simvp_gsta_highmag_macro5_direct10_"
            "poisson_zero_lam1em3_efield_lam1em3_trainfixed_disjoint_811_bs2_100ep"
        ),
        "config": ROOT / "configs/custom/pepapic/SimVP_gSTA_pepapic_direct_poisson_zero_efield_weak.py",
        "out_tag": "poisson_zero_lam1em3_efield_lam1em3",
    },
]

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
        "prefix": "transfer_low_magnet_stride1_direct10_from_high3b_stride1loss",
        "compare_dir": WORKDIRS / "compare_transfer_stride1_loss_low_magnet_3a",
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
        "prefix": "transfer_exhigh_fine_stride1_direct10_from_high3b_stride1loss",
        "compare_dir": WORKDIRS / "compare_transfer_stride1_loss_exhigh_fine_5us",
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
        "prefix": "transfer_exhigh_50us_stride1_direct10_from_high3b_stride1loss",
        "compare_dir": WORKDIRS / "compare_transfer_stride1_loss_exhigh_50us",
    },
]


def raw_csv_path(outdir):
    return outdir / "low_magnet_direct10_raw_predictions.csv"


def read_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path):
    if not rows:
        return
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


def prediction_outdir(dataset, model):
    return WORKDIRS / f"{dataset['prefix']}_{model['out_tag']}_training_compatible"


def ensure_prediction(dataset, model, device, batch_size, force):
    outdir = prediction_outdir(dataset, model)
    csv_path = raw_csv_path(outdir)
    if csv_path.exists() and not force:
        print(f"[SKIP] {csv_path}", flush=True)
        return outdir
    ckpt = model["workdir"] / "checkpoints" / "best.ckpt"
    if not ckpt.exists():
        print(f"[MISSING] {model['key']} checkpoint: {ckpt}", flush=True)
        return None
    cmd = [
        str(PYTHON),
        "predict_low_magnet_transfer_stride2.py",
        "--h5",
        str(dataset["h5"]),
        "--workdir",
        str(model["workdir"]),
        "--config",
        str(model["config"]),
        "--outdir",
        str(outdir),
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
        raise RuntimeError(f"prediction failed for {dataset['key']} / {model['key']}: {result.returncode}")
    return outdir


def group_phi(rows, group_key, value_key):
    grouped = {}
    for row in rows:
        if row["channel"] != "phi":
            continue
        grouped.setdefault(float(row[group_key]), []).append(float(row[value_key]))
    return {key: finite_stats(values) for key, values in sorted(grouped.items())}


def rolling_mean(y, window):
    if len(y) < window:
        return y.copy()
    left = window // 2
    right = window - 1 - left
    padded = np.pad(y, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def summarize(dataset, model_rows, outdir):
    summary = []
    data_rows = model_rows["data_only"]
    for channel in CHANNELS:
        copy_vals = [float(row["copy_mse"]) for row in data_rows if row["channel"] == channel]
        copy_corrs = [float(row["copy_corr"]) for row in data_rows if row["channel"] == channel]
        copy_stats = finite_stats(copy_vals)
        copy_corr = finite_stats(copy_corrs)
        summary.append({
            "dataset": dataset["key"],
            "channel": channel,
            "method": "copy",
            "label": LABELS["copy"],
            "mse_mean": copy_stats["mean"],
            "mse_median": copy_stats["median"],
            "corr_median": copy_corr["median"],
            "mean_mse_over_copy": 1.0,
            "n": copy_stats["n"],
        })
        for model in MODELS:
            key = model["key"]
            if key not in model_rows:
                continue
            rows = model_rows[key]
            vals = [float(row["model_mse"]) for row in rows if row["channel"] == channel]
            corrs = [float(row["corr"]) for row in rows if row["channel"] == channel]
            stats = finite_stats(vals)
            corr_stats = finite_stats(corrs)
            summary.append({
                "dataset": dataset["key"],
                "channel": channel,
                "method": key,
                "label": LABELS[key],
                "mse_mean": stats["mean"],
                "mse_median": stats["median"],
                "corr_median": corr_stats["median"],
                "mean_mse_over_copy": stats["mean"] / copy_stats["mean"] if copy_stats["mean"] > 0 else np.nan,
                "n": stats["n"],
            })
    write_csv(summary, outdir / "stride1_loss_transfer_summary.csv")
    return summary


def plot_channel_summary(summary, dataset, outdir):
    methods = ["copy"] + [model["key"] for model in MODELS if any(row["method"] == model["key"] for row in summary)]
    x = np.arange(len(CHANNELS), dtype=np.float64)
    width = 0.82 / len(methods)
    fig, ax = plt.subplots(figsize=(12.4, 6.4))
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width
    for offset, method in zip(offsets, methods):
        ys = []
        for channel in CHANNELS:
            row = next(r for r in summary if r["channel"] == channel and r["method"] == method)
            ys.append(row["mse_mean"])
        ax.bar(x + offset, ys, width=width, color=COLORS[method], label=LABELS[method])
    ax.set_xticks(x, CHANNELS)
    ax.set_ylabel("Mean MSE (3b stride1-normalized)")
    ax.set_title(f"{dataset['label']}: stride1 transfer error")
    ax.set_yscale("log")
    ax.grid(True, axis="y", which="both", linestyle=":", alpha=0.55)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = outdir / "stride1_loss_transfer_channel_mse_mean.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f"[PLOT] {path}", flush=True)


def plot_phi(dataset, model_rows, outdir, group_key, filename, xlabel, title_suffix):
    series = {"copy": group_phi(model_rows["data_only"], group_key, "copy_mse")}
    for model in MODELS:
        key = model["key"]
        if key in model_rows:
            series[key] = group_phi(model_rows[key], group_key, "model_mse")

    rows = []
    fig, ax = plt.subplots(figsize=(12.0, 6.2))
    for key, grouped in series.items():
        x = np.asarray(list(grouped.keys()), dtype=np.float64)
        y = np.asarray([grouped[v]["mean"] for v in x], dtype=np.float64)
        if group_key == "target_time_us":
            ax.plot(x, y, color=COLORS[key], alpha=0.14, linewidth=0.7)
            ax.plot(x, rolling_mean(y, 11), color=COLORS[key], linewidth=1.8, label=LABELS[key])
        else:
            ax.plot(x, y, color=COLORS[key], marker="o", linewidth=1.7, label=LABELS[key])
        for value in x:
            rows.append({
                "dataset": dataset["key"],
                "method": key,
                "label": LABELS[key],
                group_key: value,
                **grouped[value],
            })
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean phi MSE (3b stride1-normalized)")
    ax.set_title(f"{dataset['label']}: {title_suffix}")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.55)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = outdir / filename
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f"[PLOT] {path}", flush=True)
    csv_name = filename.replace(".png", ".csv")
    write_csv(rows, outdir / csv_name)


def write_readme(dataset, outdir, keys):
    text = f"""# Stride1 Loss Transfer Comparison

Dataset: `{dataset['label']}`

This folder compares high-magnet 3b stride1 direct10 models transferred to this stride1 testcase.
All models were trained on 3b stride1 data. Inputs are true PIC windows, so this is teacher-forced direct prediction, not rollout.

Available methods: `{', '.join(keys)}`

Main files:

- `stride1_loss_transfer_summary.csv`
- `stride1_loss_transfer_channel_mse_mean.png`
- `stride1_loss_transfer_phi_mse_by_horizon.png`
- `stride1_loss_transfer_phi_mse_target_time_smoothed.png`

Japanese note:

これはstride1の時間間隔に揃えたうえで、data-only / weak Poisson / floor hinge / weak E-field / weak Poisson+E-fieldを比較するための結果です。
"""
    path = outdir / "README.md"
    path.write_text(text, encoding="utf-8")
    print(f"[README] {path}", flush=True)


def compare_dataset(dataset):
    model_rows = {}
    for model in MODELS:
        path = raw_csv_path(prediction_outdir(dataset, model))
        if path.exists():
            model_rows[model["key"]] = read_rows(path)
    if "data_only" not in model_rows:
        print(f"[SKIP] {dataset['key']}: data_only prediction is required for copy baseline", flush=True)
        return

    outdir = dataset["compare_dir"]
    outdir.mkdir(parents=True, exist_ok=True)
    summary = summarize(dataset, model_rows, outdir)
    plot_channel_summary(summary, dataset, outdir)
    plot_phi(
        dataset,
        model_rows,
        outdir,
        "horizon_ns",
        "stride1_loss_transfer_phi_mse_by_horizon.png",
        "Prediction horizon from last input frame (ns)",
        "phi error by horizon",
    )
    plot_phi(
        dataset,
        model_rows,
        outdir,
        "target_time_us",
        "stride1_loss_transfer_phi_mse_target_time_smoothed.png",
        "Target simulation time (us)",
        "phi error over simulation time",
    )
    write_readme(dataset, outdir, ["copy"] + list(model_rows.keys()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    if not args.skip_predict:
        for dataset in DATASETS:
            for model in MODELS:
                ensure_prediction(dataset, model, args.device, args.batch_size, args.force)

    for dataset in DATASETS:
        compare_dataset(dataset)


if __name__ == "__main__":
    main()
