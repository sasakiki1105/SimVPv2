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
HIGH_CASE_INPUTS = Path(
    r"C:\Users\astro\research\PEPAPIC\test\results"
    r"\2D_ExB_high_magnet_dt2.5e-9_maxt50e-6_macro5"
    r"\SimVPv2_inputs"
)

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


STRIDE1_MODELS = [
    {
        "key": "data_only",
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_trainfixed_disjoint_811_bs2_100ep",
        "config": ROOT / "configs/custom/pepapic/SimVP_gSTA_pepapic.py",
        "out_tag": "data_only",
    },
    {
        "key": "poisson_zero",
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_direct10_poisson_zero_lam1em3_trainfixed_disjoint_811_bs2_100ep",
        "config": ROOT / "configs/custom/pepapic/SimVP_gSTA_pepapic_direct_poisson_zero_weak.py",
        "out_tag": "poisson_zero_lam1em3",
    },
    {
        "key": "floor_hinge",
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_direct10_poisson_floor_hinge_lam1em3_floor086_alpha11_trainfixed_disjoint_811_bs2_100ep",
        "config": ROOT / "configs/custom/pepapic/SimVP_gSTA_pepapic_direct_poisson_floor_hinge.py",
        "out_tag": "floor_hinge_lam1em3_alpha11",
    },
    {
        "key": "efield",
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_direct10_efield_lam1em3_trainfixed_disjoint_811_bs2_100ep",
        "config": ROOT / "configs/custom/pepapic/SimVP_gSTA_pepapic_direct_efield_weak.py",
        "out_tag": "efield_lam1em3",
    },
    {
        "key": "poisson_zero_efield",
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_direct10_poisson_zero_lam1em3_efield_lam1em3_trainfixed_disjoint_811_bs2_100ep",
        "config": ROOT / "configs/custom/pepapic/SimVP_gSTA_pepapic_direct_poisson_zero_efield_weak.py",
        "out_tag": "poisson_zero_lam1em3_efield_lam1em3",
    },
]

STRIDE2_MODELS = [
    {
        "key": "data_only",
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_subsample2_trainfixed_disjoint_811_bs2_100ep",
        "config": ROOT / "configs/custom/pepapic/SimVP_gSTA_pepapic_direct_poisson_baseline.py",
        "out_tag": "data_only",
    },
    {
        "key": "poisson_zero",
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_subsample2_direct10_poisson_zero_lam1em3_trainfixed_disjoint_811_bs2_100ep",
        "config": ROOT / "configs/custom/pepapic/SimVP_gSTA_pepapic_direct_poisson_zero_weak.py",
        "out_tag": "poisson_zero_lam1em3",
    },
    {
        "key": "floor_hinge",
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_subsample2_direct10_poisson_floor_hinge_lam1em3_floor086_alpha11_trainfixed_disjoint_811_bs2_100ep",
        "config": ROOT / "configs/custom/pepapic/SimVP_gSTA_pepapic_direct_poisson_floor_hinge.py",
        "out_tag": "floor_hinge_lam1em3_alpha11",
    },
    {
        "key": "efield",
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_subsample2_direct10_efield_lam1em3_trainfixed_disjoint_811_bs2_100ep",
        "config": ROOT / "configs/custom/pepapic/SimVP_gSTA_pepapic_direct_efield_weak.py",
        "out_tag": "efield_lam1em3",
    },
    {
        "key": "poisson_zero_efield",
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_subsample2_direct10_poisson_zero_lam1em3_efield_lam1em3_trainfixed_disjoint_811_bs2_100ep",
        "config": ROOT / "configs/custom/pepapic/SimVP_gSTA_pepapic_direct_poisson_zero_efield_weak.py",
        "out_tag": "poisson_zero_lam1em3_efield_lam1em3",
    },
]


def run(cmd, cwd=ROOT):
    print("[RUN] " + " ".join(str(x) for x in cmd), flush=True)
    result = subprocess.run([str(x) for x in cmd], cwd=str(cwd), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def make_specs(args):
    case_folder = Path(args.case_folder)
    input_dir = case_folder / "SimVPv2_inputs"
    tag = args.tag
    return {
        "stride1": {
            "h5": input_dir / f"global_norm_from_high3b_stride1_trainfixed_minmax_margin20_t0_to_t4000_training_compatible.h5",
            "timesteps": "0:1:4000",
            "high_stats_h5": HIGH_CASE_INPUTS / "global_norm_trainfixed_minmax_margin20_t0_to_t4000.h5",
            "base_dt_ns": args.base_dt_ns,
            "compare_dir": WORKDIRS / f"compare_transfer_b_sweep_{tag}_stride1_physics_loss",
            "prefix": f"transfer_{tag}_stride1_direct10_from_high3b_stride1loss",
            "models": STRIDE1_MODELS,
        },
        "stride2": {
            "h5": input_dir / f"global_norm_from_high3b_trainfixed_minmax_margin20_t0_to_t4000_step2_training_compatible.h5",
            "timesteps": "0:2:4000",
            "high_stats_h5": HIGH_CASE_INPUTS / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step2.h5",
            "base_dt_ns": args.base_dt_ns,
            "compare_dir": WORKDIRS / f"compare_transfer_b_sweep_{tag}_stride2_physics_loss",
            "prefix": f"transfer_{tag}_stride2_direct10_from_high3b",
            "models": STRIDE2_MODELS,
        },
    }


def build_h5(case_folder, stride_key, spec, force):
    if spec["h5"].exists() and not force:
        print(f"[SKIP] {spec['h5']}", flush=True)
        return
    cmd = [
        PYTHON,
        "build_low_magnet_h5_from_high_stats.py",
        "--case-folder",
        case_folder,
        "--high-stats-h5",
        spec["high_stats_h5"],
        "--timesteps",
        spec["timesteps"],
        "--output",
        spec["h5"],
        "--trim-mode",
        "training_compatible",
        "--transpose-ranks",
        "0,1,2,3",
    ]
    print(f"[BUILD H5] {stride_key}", flush=True)
    run(cmd)


def prediction_outdir(spec, model):
    return WORKDIRS / f"{spec['prefix']}_{model['out_tag']}_training_compatible"


def raw_csv_path(outdir):
    return outdir / "low_magnet_direct10_raw_predictions.csv"


def ensure_prediction(stride_key, spec, model, device, batch_size, force):
    outdir = prediction_outdir(spec, model)
    csv_path = raw_csv_path(outdir)
    if csv_path.exists() and not force:
        print(f"[SKIP] {csv_path}", flush=True)
        return outdir
    ckpt = model["workdir"] / "checkpoints" / "best.ckpt"
    if not ckpt.exists():
        print(f"[MISSING] {model['key']} checkpoint: {ckpt}", flush=True)
        return None
    cmd = [
        PYTHON,
        "predict_low_magnet_transfer_stride2.py",
        "--h5",
        spec["h5"],
        "--workdir",
        model["workdir"],
        "--config",
        model["config"],
        "--outdir",
        outdir,
        "--device",
        device,
        "--batch-size",
        str(batch_size),
        "--base-dt-ns",
        str(spec["base_dt_ns"]),
    ]
    print(f"[PREDICT] {stride_key} {model['key']}", flush=True)
    run(cmd)
    return outdir


def read_rows(path):
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


def summarize(case_key, label, stride_key, spec, model_rows):
    data_rows = model_rows["data_only"]
    rows = []
    for channel in CHANNELS:
        copy_vals = [float(row["copy_mse"]) for row in data_rows if row["channel"] == channel]
        copy_corrs = [float(row["copy_corr"]) for row in data_rows if row["channel"] == channel]
        copy_stats = finite_stats(copy_vals)
        copy_corr_stats = finite_stats(copy_corrs)
        rows.append({
            "case": case_key,
            "label": label,
            "stride": stride_key,
            "channel": channel,
            "method": "copy",
            "method_label": LABELS["copy"],
            "mse_mean": copy_stats["mean"],
            "mse_median": copy_stats["median"],
            "mse_q25": copy_stats["q25"],
            "mse_q75": copy_stats["q75"],
            "corr_mean": copy_corr_stats["mean"],
            "corr_median": copy_corr_stats["median"],
            "mean_mse_over_copy": 1.0,
            "median_mse_over_copy": 1.0,
            "n": copy_stats["n"],
        })
        for model in spec["models"]:
            key = model["key"]
            if key not in model_rows:
                continue
            vals = [float(row["model_mse"]) for row in model_rows[key] if row["channel"] == channel]
            corrs = [float(row["corr"]) for row in model_rows[key] if row["channel"] == channel]
            ratios = [float(row["model_over_copy"]) for row in model_rows[key] if row["channel"] == channel]
            stats = finite_stats(vals)
            corr_stats = finite_stats(corrs)
            ratio_stats = finite_stats(ratios)
            rows.append({
                "case": case_key,
                "label": label,
                "stride": stride_key,
                "channel": channel,
                "method": key,
                "method_label": LABELS[key],
                "mse_mean": stats["mean"],
                "mse_median": stats["median"],
                "mse_q25": stats["q25"],
                "mse_q75": stats["q75"],
                "corr_mean": corr_stats["mean"],
                "corr_median": corr_stats["median"],
                "mean_mse_over_copy": ratio_stats["mean"],
                "median_mse_over_copy": ratio_stats["median"],
                "n": stats["n"],
            })
    write_csv(rows, spec["compare_dir"] / f"transfer_{case_key}_summary.csv")
    return rows


def plot_channel_summary(case_key, label, summary, spec):
    methods = [row["method"] for row in summary if row["channel"] == "phi"]
    methods = list(dict.fromkeys(methods))
    x = np.arange(len(CHANNELS), dtype=np.float64)
    width = 0.13
    fig, ax = plt.subplots(figsize=(11.8, 6.0))
    for mi, method in enumerate(methods):
        vals = []
        for channel in CHANNELS:
            row = next(r for r in summary if r["channel"] == channel and r["method"] == method)
            vals.append(row["mse_mean"])
        offset = (mi - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, color=COLORS[method], alpha=0.88, label=LABELS[method])
    ax.set_xticks(x, CHANNELS)
    ax.set_yscale("log")
    ax.set_ylabel("Mean MSE (high3b-normalized)")
    ax.set_title(f"{label} transfer: channel mean MSE")
    ax.grid(True, axis="y", which="both", linestyle=":", alpha=0.55)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = spec["compare_dir"] / f"transfer_{case_key}_channel_mse_mean.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f"[PLOT] {path}", flush=True)


def plot_phi_by_horizon(case_key, label, stride_key, spec, model_rows):
    series = {"copy": group_phi(model_rows["data_only"], "horizon_ns", "copy_mse")}
    for model in spec["models"]:
        key = model["key"]
        if key in model_rows:
            series[key] = group_phi(model_rows[key], "horizon_ns", "model_mse")
    plot_rows = []
    fig, ax = plt.subplots(figsize=(10.4, 6.0))
    for key, grouped in series.items():
        xs = np.asarray(list(grouped.keys()), dtype=np.float64)
        ys = np.asarray([grouped[x]["mean"] for x in xs], dtype=np.float64)
        ax.plot(xs, ys, marker="o", linewidth=1.6, color=COLORS[key], label=LABELS[key])
        for x in xs:
            plot_rows.append({
                "case": case_key,
                "label": label,
                "stride": stride_key,
                "method": key,
                "method_label": LABELS[key],
                "horizon_ns": x,
                **grouped[x],
            })
    ax.set_xlabel("Prediction horizon from last input frame (ns)")
    ax.set_ylabel("Mean phi MSE (high3b-normalized)")
    ax.set_title(f"{label} transfer: phi error by horizon")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.55)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = spec["compare_dir"] / f"transfer_{case_key}_phi_mse_by_horizon.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f"[PLOT] {path}", flush=True)
    write_csv(plot_rows, spec["compare_dir"] / f"transfer_{case_key}_phi_mse_by_horizon.csv")


def plot_phi_by_time(case_key, label, stride_key, spec, model_rows):
    series = {"copy": group_phi(model_rows["data_only"], "target_time_us", "copy_mse")}
    for model in spec["models"]:
        key = model["key"]
        if key in model_rows:
            series[key] = group_phi(model_rows[key], "target_time_us", "model_mse")
    plot_rows = []
    fig, ax = plt.subplots(figsize=(12.6, 6.4))
    for key, grouped in series.items():
        xs = np.asarray(list(grouped.keys()), dtype=np.float64)
        ys = np.asarray([grouped[x]["mean"] for x in xs], dtype=np.float64)
        ax.plot(xs, ys, color=COLORS[key], alpha=0.16, linewidth=0.7)
        ax.plot(xs, rolling_mean(ys, 11), color=COLORS[key], linewidth=1.8, label=LABELS[key])
        for x in xs:
            plot_rows.append({
                "case": case_key,
                "label": label,
                "stride": stride_key,
                "method": key,
                "method_label": LABELS[key],
                "target_time_us": x,
                **grouped[x],
            })
    ax.set_xlabel("Target simulation time (us)")
    ax.set_ylabel("Mean phi MSE (high3b-normalized)")
    ax.set_title(f"{label} transfer: phi error over simulation time")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.55)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = spec["compare_dir"] / f"transfer_{case_key}_phi_mse_target_time_smoothed.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f"[PLOT] {path}", flush=True)
    write_csv(plot_rows, spec["compare_dir"] / f"transfer_{case_key}_phi_mse_by_target_time.csv")


def write_readme(case_key, label, stride_key, spec, available):
    text = f"""# {label} Transfer Evaluation

This folder evaluates high-magnet testcase 3b models on the {label} PIC testcase.

Source testcase: 3b high magnet, B = 0.2 mT.
Target testcase: {label}.
Stride: {stride_key}.
Evaluation: teacher-forced direct10 prediction. The input window is true PIC data, not model rollout.

Compared methods:

- Copy baseline: copy the last input frame.
- Data-only: ordinary SimVPv2/gSTA MSE training.
- Weak Poisson: data loss plus weak Poisson-residual regularization.
- Poisson floor hinge: data loss plus floor-hinge Poisson residual regularization.
- Weak E-field: data loss plus weak electric-field regularization.
- Weak Poisson + E-field: both weak physics losses.

Available methods in this run: {', '.join(available)}
"""
    path = spec["compare_dir"] / "README.md"
    path.write_text(text, encoding="utf-8")
    print(f"[README] {path}", flush=True)


def compare(case_key, label, stride_key, spec, transfer_dirs):
    spec["compare_dir"].mkdir(parents=True, exist_ok=True)
    model_rows = {}
    for model in spec["models"]:
        outdir = transfer_dirs.get((stride_key, model["key"]))
        if outdir is None:
            continue
        path = raw_csv_path(outdir)
        if path.exists():
            model_rows[model["key"]] = read_rows(path)
    if "data_only" not in model_rows:
        print(f"[SKIP] {stride_key}: data_only prediction is required for copy baseline", flush=True)
        return []
    summary = summarize(case_key, label, stride_key, spec, model_rows)
    plot_channel_summary(case_key, label, summary, spec)
    plot_phi_by_horizon(case_key, label, stride_key, spec, model_rows)
    plot_phi_by_time(case_key, label, stride_key, spec, model_rows)
    write_readme(case_key, label, stride_key, spec, ["copy"] + list(model_rows.keys()))
    meta = {
        "case": case_key,
        "label": label,
        "stride": stride_key,
        "h5": str(spec["h5"]),
        "base_dt_ns": spec["base_dt_ns"],
        "models": list(model_rows.keys()),
    }
    (spec["compare_dir"] / f"transfer_{case_key}_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return summary


def write_combined(case_key, label, all_rows):
    outdir = WORKDIRS / f"compare_transfer_b_sweep_{case_key}_combined"
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(all_rows, outdir / f"transfer_{case_key}_stride1_stride2_summary.csv")
    phi_rows = [row for row in all_rows if row["channel"] == "phi"]
    methods = list(dict.fromkeys(row["method"] for row in phi_rows))
    strides = ["stride1", "stride2"]
    fig, ax = plt.subplots(figsize=(11.4, 6.0))
    x = np.arange(len(strides), dtype=np.float64)
    width = 0.13
    for mi, method in enumerate(methods):
        vals = []
        for stride in strides:
            row = next((r for r in phi_rows if r["stride"] == stride and r["method"] == method), None)
            vals.append(np.nan if row is None else row["mse_mean"] / next(r for r in phi_rows if r["stride"] == stride and r["method"] == "copy")["mse_mean"])
        ax.bar(x + (mi - (len(methods) - 1) / 2) * width, vals, width=width, color=COLORS[method], alpha=0.88, label=LABELS[method])
    ax.axhline(1.0, color="#111827", linewidth=1.0, linestyle="--")
    ax.set_xticks(x, strides)
    ax.set_ylabel("Mean phi MSE / copy mean phi MSE")
    ax.set_title(f"{label} transfer: phi model-over-copy")
    ax.grid(True, axis="y", linestyle=":", alpha=0.55)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = outdir / f"transfer_{case_key}_phi_model_over_copy_stride_compare.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    (outdir / "README.md").write_text(
        f"# {label} Transfer Combined Summary\n\nValues below 1.0 mean the model beats copy for phi.\n",
        encoding="utf-8",
    )
    print(f"[PLOT] {path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-folder", required=True)
    parser.add_argument("--case-key", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--base-dt-ns", type=float, default=12.5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--force-predict", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--only", choices=["stride1", "stride2", "both"], default="both")
    args = parser.parse_args()

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    specs = make_specs(args)
    case_folder = Path(args.case_folder)
    stride_keys = ["stride1", "stride2"] if args.only == "both" else [args.only]

    if not args.skip_build:
        for stride_key in stride_keys:
            build_h5(case_folder, stride_key, specs[stride_key], args.force_build)

    transfer_dirs = {}
    for stride_key in stride_keys:
        spec = specs[stride_key]
        for model in spec["models"]:
            if args.skip_predict:
                outdir = prediction_outdir(spec, model)
            else:
                outdir = ensure_prediction(stride_key, spec, model, args.device, args.batch_size, args.force_predict)
            if outdir is not None:
                transfer_dirs[(stride_key, model["key"])] = outdir

    all_rows = []
    for stride_key in stride_keys:
        all_rows.extend(compare(args.case_key, args.label, stride_key, specs[stride_key], transfer_dirs))
    if len(stride_keys) == 2:
        write_combined(args.case_key, args.label, all_rows)


if __name__ == "__main__":
    main()
