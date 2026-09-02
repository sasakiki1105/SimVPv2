import argparse
import csv
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
WORKDIRS = ROOT / "workdirs"
RESULTS = Path(r"C:\Users\astro\research\PEPAPIC\test\results")
PYTHON = Path(r"C:\Users\astro\anaconda3\envs\OpenSTL\python.exe")

CHANNELS = ["electron_den", "ion_den", "phi"]
METHODS = {
    "data_only": {
        "label": "Data-only",
        "color": "#2563eb",
        "tag": None,
        "config": ROOT / "configs/custom/pepapic/SimVP_gSTA_pepapic_direct.py",
    },
    "efield": {
        "label": "Weak E-field",
        "color": "#7c3aed",
        "tag": "efield_lam1em3",
        "config": ROOT / "configs/custom/pepapic/SimVP_gSTA_pepapic_direct_efield_weak.py",
    },
    "poisson_zero_efield": {
        "label": "Weak Poisson + E-field",
        "color": "#ea580c",
        "tag": "poisson_zero_lam1em3_efield_lam1em3",
        "config": ROOT / "configs/custom/pepapic/SimVP_gSTA_pepapic_direct_poisson_zero_efield_weak.py",
    },
}
COPY = {"label": "Copy baseline", "color": "#6b7280"}

DATASETS = [
    {
        "key": "low_magnet_3a",
        "label": "3a low magnet, retained 25 ns",
        "prefix": "transfer_low_magnet_stride2",
        "h5": RESULTS
        / "2D_ExB_low_magnet_dt2.5e-9_maxt50e-6_macro5"
        / "SimVPv2_inputs"
        / "global_norm_from_high3b_trainfixed_minmax_margin20_t0_to_t4000_step2_training_compatible.h5",
        "base_dt_ns": 12.5,
    },
    {
        "key": "exhigh_50us",
        "label": "Ex-high magnet 50 us, retained 25 ns",
        "prefix": "transfer_exhigh_50us_step2",
        "h5": RESULTS
        / "2D_ExB_exhigh_magnet_dt2.5e-9_maxt50e-6_macro5"
        / "SimVPv2_inputs"
        / "global_norm_from_high3b_trainfixed_minmax_margin20_t0_to_t4000_step2_training_compatible.h5",
        "base_dt_ns": 12.5,
    },
]


def model_workdir(method, aft):
    if method == "data_only":
        return WORKDIRS / f"pepapic_simvp_gsta_highmag_macro5_subsample2_direct{aft}_trainfixed_disjoint_811_bs2_100ep"
    return WORKDIRS / (
        f"pepapic_simvp_gsta_highmag_macro5_subsample2_direct{aft}_"
        f"{METHODS[method]['tag']}_trainfixed_disjoint_811_bs2_100ep"
    )


def transfer_outdir(dataset, method, aft):
    if method == "data_only":
        tag = "data_only"
    else:
        tag = METHODS[method]["tag"]
    return WORKDIRS / f"{dataset['prefix']}_direct{aft}_from_high3b_{tag}_training_compatible"


def raw_path(dataset, method, aft):
    return transfer_outdir(dataset, method, aft) / f"low_magnet_direct{aft}_raw_predictions.csv"


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
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


def ensure_prediction(dataset, method, aft, device, batch_size, force):
    outdir = transfer_outdir(dataset, method, aft)
    csv_path = raw_path(dataset, method, aft)
    if csv_path.exists() and not force:
        print(f"[SKIP] {csv_path}", flush=True)
        return True
    workdir = model_workdir(method, aft)
    ckpt = workdir / "checkpoints" / "best.ckpt"
    if not ckpt.exists():
        print(f"[MISSING] {method} direct{aft} checkpoint: {ckpt}", flush=True)
        return False
    cmd = [
        str(PYTHON),
        "predict_low_magnet_transfer_stride2.py",
        "--h5",
        str(dataset["h5"]),
        "--workdir",
        str(workdir),
        "--config",
        str(METHODS[method]["config"]),
        "--outdir",
        str(outdir),
        "--device",
        device,
        "--batch-size",
        str(batch_size),
        "--base-dt-ns",
        str(dataset["base_dt_ns"]),
        "--aft",
        str(aft),
    ]
    print(f"[RUN] {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Prediction failed: {dataset['key']} {method} direct{aft}")
    return True


def aggregate_dataset(dataset, aft):
    model_rows = {}
    for method in METHODS:
        path = raw_path(dataset, method, aft)
        if path.exists():
            model_rows[method] = read_csv(path)
    if "data_only" not in model_rows:
        print(f"[SKIP] {dataset['key']}: data_only prediction is required for copy baseline", flush=True)
        return []

    rows = []
    rows.extend(aggregate_copy(dataset, aft, model_rows["data_only"]))
    for method, raw_rows in model_rows.items():
        rows.extend(aggregate_method(dataset, method, aft, raw_rows))
    return rows


def aggregate_copy(dataset, aft, raw_rows):
    grouped = defaultdict(list)
    for row in raw_rows:
        grouped[(row["channel"], float(row["horizon_ns"]))].append(float(row["copy_mse"]))
    rows = []
    for (channel, horizon), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        stats = finite_stats(values)
        rows.append(
            {
                "dataset": dataset["key"],
                "dataset_label": dataset["label"],
                "aft": aft,
                "channel": channel,
                "method": "copy",
                "label": COPY["label"],
                "horizon_ns": horizon,
                "model_mse_mean": stats["mean"],
                "model_mse_median": stats["median"],
                "copy_mse_mean": stats["mean"],
                "copy_mse_median": stats["median"],
                "model_over_copy_mean": 1.0,
                "skill_score_mean": 0.0,
                "n": stats["n"],
            }
        )
    return rows


def aggregate_method(dataset, method, aft, raw_rows):
    grouped = defaultdict(list)
    for row in raw_rows:
        grouped[(row["channel"], float(row["horizon_ns"]))].append(row)
    rows = []
    for (channel, horizon), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        model_stats = finite_stats([float(row["model_mse"]) for row in group])
        copy_stats = finite_stats([float(row["copy_mse"]) for row in group])
        ratio = model_stats["mean"] / copy_stats["mean"] if copy_stats["mean"] > 0 else np.nan
        rows.append(
            {
                "dataset": dataset["key"],
                "dataset_label": dataset["label"],
                "aft": aft,
                "channel": channel,
                "method": method,
                "label": METHODS[method]["label"],
                "horizon_ns": horizon,
                "model_mse_mean": model_stats["mean"],
                "model_mse_median": model_stats["median"],
                "copy_mse_mean": copy_stats["mean"],
                "copy_mse_median": copy_stats["median"],
                "model_over_copy_mean": ratio,
                "skill_score_mean": 1.0 - ratio if np.isfinite(ratio) else np.nan,
                "n": model_stats["n"],
            }
        )
    return rows


def plot_dataset(rows, dataset, aft, outdir):
    subset = [row for row in rows if row["dataset"] == dataset["key"] and row["channel"] == "phi"]
    if not subset:
        return
    plot_mse(subset, dataset, aft, outdir)
    plot_ratio(subset, dataset, aft, outdir)
    plot_skill(subset, dataset, aft, outdir)


def method_style(method):
    if method == "copy":
        return COPY["label"], COPY["color"]
    return METHODS[method]["label"], METHODS[method]["color"]


def plot_mse(rows, dataset, aft, outdir):
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    for method in ["copy", *METHODS.keys()]:
        group = [row for row in rows if row["method"] == method]
        if not group:
            continue
        label, color = method_style(method)
        x = np.asarray([float(row["horizon_ns"]) for row in group])
        y = np.asarray([float(row["model_mse_mean"]) for row in group])
        order = np.argsort(x)
        ax.plot(x[order], y[order], marker="o", linewidth=1.7, color=color, label=label)
    ax.set_xlabel("Physical prediction horizon from last input frame (ns)")
    ax.set_ylabel("Mean phi MSE (high3b-normalized)")
    ax.set_title(f"{dataset['label']}: direct{aft} phi MSE")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.55)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    path = outdir / f"{dataset['key']}_direct{aft}_phi_mse_by_horizon.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f"[PLOT] {path}", flush=True)


def plot_ratio(rows, dataset, aft, outdir):
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        if not group:
            continue
        label, color = method_style(method)
        x = np.asarray([float(row["horizon_ns"]) for row in group])
        y = np.asarray([float(row["model_over_copy_mean"]) for row in group])
        order = np.argsort(x)
        ax.plot(x[order], y[order], marker="o", linewidth=1.7, color=color, label=label)
    ax.axhline(1.0, color="#333333", linestyle=":", linewidth=1.2, label="copy parity")
    ax.set_xlabel("Physical prediction horizon from last input frame (ns)")
    ax.set_ylabel("Model MSE / copy MSE")
    ax.set_title(f"{dataset['label']}: direct{aft} phi model/copy")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.55)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    path = outdir / f"{dataset['key']}_direct{aft}_phi_model_over_copy.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f"[PLOT] {path}", flush=True)


def plot_skill(rows, dataset, aft, outdir):
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        if not group:
            continue
        label, color = method_style(method)
        x = np.asarray([float(row["horizon_ns"]) for row in group])
        y = np.asarray([float(row["skill_score_mean"]) for row in group])
        order = np.argsort(x)
        ax.plot(x[order], y[order], marker="o", linewidth=1.7, color=color, label=label)
    ax.axhline(0.0, color="#333333", linestyle=":", linewidth=1.2, label="copy parity")
    ax.set_xlabel("Physical prediction horizon from last input frame (ns)")
    ax.set_ylabel("Skill score = 1 - model MSE / copy MSE")
    ax.set_title(f"{dataset['label']}: direct{aft} phi skill")
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    path = outdir / f"{dataset['key']}_direct{aft}_phi_skill_score.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f"[PLOT] {path}", flush=True)


def write_readme(outdir, aft):
    text = f"""# Long Horizon Transfer Direct{aft}

This folder compares high-magnet 3b stride2 direct{aft} models on selected transfer testcases.

Compared methods:

- Copy baseline
- Data-only
- Weak E-field
- Weak Poisson + E-field

The evaluation is teacher-forced direct prediction. Inputs are true PIC frames from the transfer testcase, not previous predictions.

The main metric is `skill_score_mean = 1 - model_mse_mean / copy_mse_mean`.
Positive skill means the model beats copy. Negative skill means copy is better.

# Japanese Note

これは、copy baseline が強すぎない物理時間horizonを見たいという目的で作った direct{aft} 転移評価です。
3a と ex-high 50 us を対象に、data-only / E-field loss / Poisson+E-field loss を比較します。
"""
    path = outdir / "README.md"
    path.write_text(text, encoding="utf-8")
    print(f"[README] {path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aft", type=int, default=20)
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    outdir = WORKDIRS / f"compare_transfer_long_horizon_direct{args.aft}"
    outdir.mkdir(parents=True, exist_ok=True)

    if not args.skip_predict:
        for dataset in DATASETS:
            for method in METHODS:
                ensure_prediction(dataset, method, args.aft, args.device, args.batch_size, args.force)

    rows = []
    for dataset in DATASETS:
        rows.extend(aggregate_dataset(dataset, args.aft))
    rows.sort(key=lambda row: (row["dataset"], row["channel"], float(row["horizon_ns"]), row["method"]))
    write_csv(rows, outdir / f"transfer_direct{args.aft}_summary_by_horizon.csv")
    for dataset in DATASETS:
        plot_dataset(rows, dataset, args.aft, outdir)
    meta = {
        "description": f"Long horizon direct{args.aft} transfer evaluation.",
        "aft": args.aft,
        "datasets": [{k: str(v) if isinstance(v, Path) else v for k, v in dataset.items()} for dataset in DATASETS],
        "methods": list(METHODS.keys()),
    }
    (outdir / f"transfer_direct{args.aft}_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[JSON] {outdir / f'transfer_direct{args.aft}_summary.json'}", flush=True)
    write_readme(outdir, args.aft)


if __name__ == "__main__":
    main()
