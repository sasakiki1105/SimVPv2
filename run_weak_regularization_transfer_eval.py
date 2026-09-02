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
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_subsample2_trainfixed_disjoint_811_bs2_100ep",
        "config": ROOT / "configs/custom/pepapic/SimVP_gSTA_pepapic_direct_poisson_baseline.py",
        "out_tag": "data_only",
    },
    {
        "key": "poisson_zero",
        "workdir": WORKDIRS / (
            "pepapic_simvp_gsta_highmag_macro5_subsample2_direct10_"
            "poisson_zero_lam1em3_trainfixed_disjoint_811_bs2_100ep"
        ),
        "config": ROOT / "configs/custom/pepapic/SimVP_gSTA_pepapic_direct_poisson_zero_weak.py",
        "out_tag": "poisson_zero_lam1em3",
    },
    {
        "key": "floor_hinge",
        "workdir": WORKDIRS / (
            "pepapic_simvp_gsta_highmag_macro5_subsample2_direct10_"
            "poisson_floor_hinge_lam1em3_floor086_alpha11_trainfixed_disjoint_811_bs2_100ep"
        ),
        "config": ROOT / "configs/custom/pepapic/SimVP_gSTA_pepapic_direct_poisson_floor_hinge.py",
        "out_tag": "floor_hinge_lam1em3_alpha11",
    },
    {
        "key": "efield",
        "workdir": WORKDIRS / (
            "pepapic_simvp_gsta_highmag_macro5_subsample2_direct10_"
            "efield_lam1em3_trainfixed_disjoint_811_bs2_100ep"
        ),
        "config": ROOT / "configs/custom/pepapic/SimVP_gSTA_pepapic_direct_efield_weak.py",
        "out_tag": "efield_lam1em3",
    },
    {
        "key": "poisson_zero_efield",
        "workdir": WORKDIRS / (
            "pepapic_simvp_gsta_highmag_macro5_subsample2_direct10_"
            "poisson_zero_lam1em3_efield_lam1em3_trainfixed_disjoint_811_bs2_100ep"
        ),
        "config": ROOT / "configs/custom/pepapic/SimVP_gSTA_pepapic_direct_poisson_zero_efield_weak.py",
        "out_tag": "poisson_zero_lam1em3_efield_lam1em3",
    },
]

DATASETS = [
    {
        "key": "low_magnet_3a",
        "label": "Low magnet 3a",
        "h5": Path(
            r"C:\Users\astro\research\PEPAPIC\test\results"
            r"\2D_ExB_low_magnet_dt2.5e-9_maxt50e-6_macro5"
            r"\SimVPv2_inputs\global_norm_from_high3b_trainfixed_minmax_margin20_t0_to_t4000_step2_training_compatible.h5"
        ),
        "base_dt_ns": 12.5,
        "transfer_prefix": "transfer_low_magnet_stride2_direct10_from_high3b",
        "compare_dir": WORKDIRS / "compare_transfer_low_magnet_stride2_weak_regularization",
    },
    {
        "key": "exhigh_fine_step20",
        "label": "Ex-high magnet 5 us, retained 25 ns",
        "h5": Path(
            r"C:\Users\astro\research\PEPAPIC\test\results"
            r"\2D_ExB_exhigh_magnet_dt2.5e-10_maxt5e-6_macro5"
            r"\SimVPv2_inputs\global_norm_from_high3b_trainfixed_minmax_margin20_t0_to_t4000_step20_training_compatible.h5"
        ),
        "base_dt_ns": 1.25,
        "transfer_prefix": "transfer_exhigh_fine_step20_direct10_from_high3b",
        "compare_dir": WORKDIRS / "compare_transfer_exhigh_fine_step20_weak_regularization",
    },
    {
        "key": "exhigh_50us_step2",
        "label": "Ex-high magnet 50 us, retained 25 ns",
        "h5": Path(
            r"C:\Users\astro\research\PEPAPIC\test\results"
            r"\2D_ExB_exhigh_magnet_dt2.5e-9_maxt50e-6_macro5"
            r"\SimVPv2_inputs\global_norm_from_high3b_trainfixed_minmax_margin20_t0_to_t4000_step2_training_compatible.h5"
        ),
        "base_dt_ns": 12.5,
        "transfer_prefix": "transfer_exhigh_50us_step2_direct10_from_high3b",
        "compare_dir": WORKDIRS / "compare_transfer_exhigh_50us_step2_weak_regularization",
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
    if arr.size == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "q25": np.nan, "q75": np.nan}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
    }


def ensure_prediction(dataset, model, device, batch_size, force):
    outdir = WORKDIRS / f"{dataset['transfer_prefix']}_{model['out_tag']}_training_compatible"
    csv_path = raw_csv_path(outdir)
    if csv_path.exists() and not force:
        print(f"[SKIP] {csv_path}", flush=True)
        return outdir

    ckpt = model["workdir"] / "checkpoints" / "best.ckpt"
    if not ckpt.exists():
        print(f"[MISSING] {model['key']} checkpoint: {ckpt}", flush=True)
        return None
    if not dataset["h5"].exists():
        print(f"[MISSING] dataset H5: {dataset['h5']}", flush=True)
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
        key = float(row[group_key])
        grouped.setdefault(key, []).append(float(row[value_key]))
    return {
        key: finite_stats(values)
        for key, values in sorted(grouped.items())
    }


def summarize(dataset, model_rows, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    data_rows = model_rows["data_only"]
    summary = []
    for channel in CHANNELS:
        copy_vals = [
            float(row["copy_mse"])
            for row in data_rows
            if row["channel"] == channel
        ]
        copy_corr = [
            float(row["copy_corr"])
            for row in data_rows
            if row["channel"] == channel
        ]
        copy_stats = finite_stats(copy_vals)
        corr_stats = finite_stats(copy_corr)
        summary.append({
            "dataset": dataset["key"],
            "channel": channel,
            "method": "copy",
            "label": LABELS["copy"],
            "mse_mean": copy_stats["mean"],
            "mse_median": copy_stats["median"],
            "mse_q25": copy_stats["q25"],
            "mse_q75": copy_stats["q75"],
            "corr_mean": corr_stats["mean"],
            "corr_median": corr_stats["median"],
            "mean_mse_over_copy": 1.0,
            "n": copy_stats["n"],
        })

        for model in MODELS:
            key = model["key"]
            if key not in model_rows:
                continue
            rows = model_rows[key]
            vals = [
                float(row["model_mse"])
                for row in rows
                if row["channel"] == channel
            ]
            corrs = [
                float(row["corr"])
                for row in rows
                if row["channel"] == channel
            ]
            stats = finite_stats(vals)
            corr_stats = finite_stats(corrs)
            summary.append({
                "dataset": dataset["key"],
                "channel": channel,
                "method": key,
                "label": LABELS[key],
                "mse_mean": stats["mean"],
                "mse_median": stats["median"],
                "mse_q25": stats["q25"],
                "mse_q75": stats["q75"],
                "corr_mean": corr_stats["mean"],
                "corr_median": corr_stats["median"],
                "mean_mse_over_copy": stats["mean"] / copy_stats["mean"] if copy_stats["mean"] > 0 else np.nan,
                "n": stats["n"],
            })
    write_csv(summary, outdir / "transfer_weak_regularization_summary.csv")
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
    ax.set_ylabel("Mean MSE (high3b-normalized)")
    ax.set_title(f"{dataset['label']}: direct10 transfer error")
    ax.set_yscale("log")
    ax.grid(True, axis="y", which="both", linestyle=":", alpha=0.55)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = outdir / "transfer_weak_regularization_channel_mse_mean.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f"[PLOT] {path}", flush=True)


def plot_phi_by_horizon(dataset, model_rows, outdir):
    series = {"copy": group_phi(model_rows["data_only"], "horizon_ns", "copy_mse")}
    for model in MODELS:
        key = model["key"]
        if key in model_rows:
            series[key] = group_phi(model_rows[key], "horizon_ns", "model_mse")

    rows = []
    fig, ax = plt.subplots(figsize=(10.4, 6.0))
    for key, grouped in series.items():
        xs = np.asarray(list(grouped.keys()), dtype=np.float64)
        ys = np.asarray([grouped[x]["mean"] for x in xs], dtype=np.float64)
        ax.plot(xs, ys, marker="o", linewidth=1.6, color=COLORS[key], label=LABELS[key])
        for x in xs:
            rows.append({
                "dataset": dataset["key"],
                "method": key,
                "label": LABELS[key],
                "horizon_ns": x,
                **grouped[x],
            })
    ax.set_xlabel("Prediction horizon from last input frame (ns)")
    ax.set_ylabel("Mean phi MSE (high3b-normalized)")
    ax.set_title(f"{dataset['label']}: phi error by horizon")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.55)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = outdir / "transfer_weak_regularization_phi_mse_by_horizon.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f"[PLOT] {path}", flush=True)
    write_csv(rows, outdir / "transfer_weak_regularization_phi_mse_by_horizon.csv")


def rolling_mean(y, window):
    if len(y) < window:
        return y.copy()
    left = window // 2
    right = window - 1 - left
    padded = np.pad(y, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def plot_phi_by_time(dataset, model_rows, outdir):
    series = {"copy": group_phi(model_rows["data_only"], "target_time_us", "copy_mse")}
    for model in MODELS:
        key = model["key"]
        if key in model_rows:
            series[key] = group_phi(model_rows[key], "target_time_us", "model_mse")

    rows = []
    fig, ax = plt.subplots(figsize=(12.6, 6.4))
    for key, grouped in series.items():
        xs = np.asarray(list(grouped.keys()), dtype=np.float64)
        ys = np.asarray([grouped[x]["mean"] for x in xs], dtype=np.float64)
        ax.plot(xs, ys, color=COLORS[key], alpha=0.16, linewidth=0.7)
        ax.plot(xs, rolling_mean(ys, 11), color=COLORS[key], linewidth=1.8, label=LABELS[key])
        for x in xs:
            rows.append({
                "dataset": dataset["key"],
                "method": key,
                "label": LABELS[key],
                "target_time_us": x,
                **grouped[x],
            })
    ax.set_xlabel("Target simulation time (us)")
    ax.set_ylabel("Mean phi MSE (high3b-normalized)")
    ax.set_title(f"{dataset['label']}: phi error over simulation time")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.55)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = outdir / "transfer_weak_regularization_phi_mse_target_time_smoothed.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f"[PLOT] {path}", flush=True)
    write_csv(rows, outdir / "transfer_weak_regularization_phi_mse_by_target_time.csv")


def write_readme(dataset, outdir, available_keys):
    text = f"""# Weak Regularization Transfer Comparison

Dataset: `{dataset['label']}`

This folder compares high-magnet 3b stride2 direct10 models transferred to this testcase.
All predictions use true PIC input windows, so this is teacher-forced direct prediction, not rollout.

Compared methods:

- `copy`: copy the last input frame.
- `data_only`: ordinary SimVPv2/gSTA trained with data MSE only.
- `poisson_zero`: weak Poisson residual regularization, no true-floor threshold.
- `floor_hinge`: previous Poisson true-floor hinge model.
- `efield`: weak electric-field regularization from `E = -grad(phi)`.
- `poisson_zero_efield`: weak Poisson residual plus weak electric-field regularization.

Available in this run: `{', '.join(available_keys)}`

Main files:

- `transfer_weak_regularization_summary.csv`
- `transfer_weak_regularization_channel_mse_mean.png`
- `transfer_weak_regularization_phi_mse_by_horizon.png`
- `transfer_weak_regularization_phi_mse_target_time_smoothed.png`

Japanese note:

これは「3bで学習したモデルが他のtest caseへどれくらい外挿できるか」を見る比較です。
`true_floor` に依存しない弱いPoisson lossと、`phi` の勾配から作る電場lossが、copy baselineやdata-onlyより有利になるかを確認します。
"""
    path = outdir / "README.md"
    path.write_text(text, encoding="utf-8")
    print(f"[README] {path}", flush=True)


def compare_dataset(dataset, transfer_dirs):
    model_rows = {}
    for model in MODELS:
        outdir = transfer_dirs.get((dataset["key"], model["key"]))
        if outdir is None:
            continue
        path = raw_csv_path(outdir)
        if path.exists():
            model_rows[model["key"]] = read_rows(path)

    if "data_only" not in model_rows:
        print(f"[SKIP] {dataset['key']}: data_only prediction is required for copy baseline", flush=True)
        return

    outdir = dataset["compare_dir"]
    summary = summarize(dataset, model_rows, outdir)
    plot_channel_summary(summary, dataset, outdir)
    plot_phi_by_horizon(dataset, model_rows, outdir)
    plot_phi_by_time(dataset, model_rows, outdir)
    write_readme(dataset, outdir, ["copy"] + list(model_rows.keys()))
    meta = {
        "dataset": dataset["key"],
        "label": dataset["label"],
        "h5": str(dataset["h5"]),
        "base_dt_ns": dataset["base_dt_ns"],
        "models": list(model_rows.keys()),
    }
    (outdir / "transfer_weak_regularization_summary.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="Only rebuild comparison plots from existing prediction CSV files.",
    )
    args = parser.parse_args()

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    transfer_dirs = {}
    for dataset in DATASETS:
        for model in MODELS:
            if args.skip_predict:
                outdir = WORKDIRS / f"{dataset['transfer_prefix']}_{model['out_tag']}_training_compatible"
            else:
                outdir = ensure_prediction(dataset, model, args.device, args.batch_size, args.force)
            if outdir is not None:
                transfer_dirs[(dataset["key"], model["key"])] = outdir

    for dataset in DATASETS:
        compare_dataset(dataset, transfer_dirs)


if __name__ == "__main__":
    main()
