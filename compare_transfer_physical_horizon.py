import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"C:\Users\astro\research\SimVPv2")
WORKDIRS = ROOT / "workdirs"
RESULTS = Path(r"C:\Users\astro\research\PEPAPIC\test\results")
OUTDIR = WORKDIRS / "compare_transfer_physical_horizon"

CHANNEL_FALLBACK = ["electron_den", "ion_den", "phi"]
METHOD_ORDER = ["copy", "data_only", "poisson_zero", "floor_hinge", "efield", "poisson_zero_efield"]
METHOD_LABELS = {
    "copy": "Copy baseline",
    "data_only": "Data-only",
    "poisson_zero": "Weak Poisson",
    "floor_hinge": "Poisson floor hinge",
    "efield": "Weak E-field",
    "poisson_zero_efield": "Weak Poisson + E-field",
}
METHOD_COLORS = {
    "copy": "#6b7280",
    "data_only": "#2563eb",
    "poisson_zero": "#16a34a",
    "floor_hinge": "#dc2626",
    "efield": "#7c3aed",
    "poisson_zero_efield": "#ea580c",
}

MODEL_TAGS = {
    "data_only": "data_only",
    "poisson_zero": "poisson_zero_lam1em3",
    "floor_hinge": "floor_hinge_lam1em3_alpha11",
    "efield": "efield_lam1em3",
    "poisson_zero_efield": "poisson_zero_lam1em3_efield_lam1em3",
}

H5_CASES = [
    {
        "key": "high3b_stride1",
        "physical_dataset": "high3b",
        "label": "3b high magnet, stride1",
        "base_dt_ns": 12.5,
        "h5": RESULTS
        / "2D_ExB_high_magnet_dt2.5e-9_maxt50e-6_macro5"
        / "SimVPv2_inputs"
        / "global_norm_trainfixed_minmax_margin20_t0_to_t4000.h5",
    },
    {
        "key": "high3b_step2",
        "physical_dataset": "high3b",
        "label": "3b high magnet, stride2",
        "base_dt_ns": 12.5,
        "h5": RESULTS
        / "2D_ExB_high_magnet_dt2.5e-9_maxt50e-6_macro5"
        / "SimVPv2_inputs"
        / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step2.h5",
    },
    {
        "key": "low_magnet_3a_stride1",
        "physical_dataset": "low_magnet_3a",
        "label": "3a low magnet, stride1",
        "base_dt_ns": 12.5,
        "h5": RESULTS
        / "2D_ExB_low_magnet_dt2.5e-9_maxt50e-6_macro5"
        / "SimVPv2_inputs"
        / "global_norm_from_high3b_stride1_trainfixed_minmax_margin20_t0_to_t4000_training_compatible.h5",
    },
    {
        "key": "low_magnet_3a_step2",
        "physical_dataset": "low_magnet_3a",
        "label": "3a low magnet, stride2",
        "base_dt_ns": 12.5,
        "h5": RESULTS
        / "2D_ExB_low_magnet_dt2.5e-9_maxt50e-6_macro5"
        / "SimVPv2_inputs"
        / "global_norm_from_high3b_trainfixed_minmax_margin20_t0_to_t4000_step2_training_compatible.h5",
    },
    {
        "key": "exhigh_fine_5us_stride1",
        "physical_dataset": "exhigh_fine_5us",
        "label": "Ex-high 5 us, 1.25 ns frames",
        "base_dt_ns": 1.25,
        "h5": RESULTS
        / "2D_ExB_exhigh_magnet_dt2.5e-10_maxt5e-6_macro5"
        / "SimVPv2_inputs"
        / "global_norm_from_high3b_stride1_trainfixed_minmax_margin20_t0_to_t4000_training_compatible.h5",
    },
    {
        "key": "exhigh_fine_5us_step20",
        "physical_dataset": "exhigh_fine_5us",
        "label": "Ex-high 5 us, retained 25 ns",
        "base_dt_ns": 1.25,
        "h5": RESULTS
        / "2D_ExB_exhigh_magnet_dt2.5e-10_maxt5e-6_macro5"
        / "SimVPv2_inputs"
        / "global_norm_from_high3b_trainfixed_minmax_margin20_t0_to_t4000_step20_training_compatible.h5",
    },
    {
        "key": "exhigh_50us_stride1",
        "physical_dataset": "exhigh_50us",
        "label": "Ex-high 50 us, stride1",
        "base_dt_ns": 12.5,
        "h5": RESULTS
        / "2D_ExB_exhigh_magnet_dt2.5e-9_maxt50e-6_macro5"
        / "SimVPv2_inputs"
        / "global_norm_from_high3b_stride1_trainfixed_minmax_margin20_t0_to_t4000_training_compatible.h5",
    },
    {
        "key": "exhigh_50us_step2",
        "physical_dataset": "exhigh_50us",
        "label": "Ex-high 50 us, stride2",
        "base_dt_ns": 12.5,
        "h5": RESULTS
        / "2D_ExB_exhigh_magnet_dt2.5e-9_maxt50e-6_macro5"
        / "SimVPv2_inputs"
        / "global_norm_from_high3b_trainfixed_minmax_margin20_t0_to_t4000_step2_training_compatible.h5",
    },
]

TRANSFER_CASES = [
    {
        "key": "low_magnet_3a_stride1",
        "physical_dataset": "low_magnet_3a",
        "label": "3a stride1",
        "sampling": "stride1",
        "prefix": "transfer_low_magnet_stride1_direct10_from_high3b_stride1loss",
    },
    {
        "key": "low_magnet_3a_stride2",
        "physical_dataset": "low_magnet_3a",
        "label": "3a stride2",
        "sampling": "stride2",
        "prefix": "transfer_low_magnet_stride2_direct10_from_high3b",
    },
    {
        "key": "exhigh_fine_5us_stride1",
        "physical_dataset": "exhigh_fine_5us",
        "label": "Ex-high fine stride1",
        "sampling": "stride1",
        "prefix": "transfer_exhigh_fine_stride1_direct10_from_high3b_stride1loss",
    },
    {
        "key": "exhigh_fine_5us_step20",
        "physical_dataset": "exhigh_fine_5us",
        "label": "Ex-high fine retained 25 ns",
        "sampling": "step20",
        "prefix": "transfer_exhigh_fine_step20_direct10_from_high3b",
    },
    {
        "key": "exhigh_50us_stride1",
        "physical_dataset": "exhigh_50us",
        "label": "Ex-high 50 us stride1",
        "sampling": "stride1",
        "prefix": "transfer_exhigh_50us_stride1_direct10_from_high3b_stride1loss",
    },
    {
        "key": "exhigh_50us_step2",
        "physical_dataset": "exhigh_50us",
        "label": "Ex-high 50 us stride2",
        "sampling": "stride2",
        "prefix": "transfer_exhigh_50us_step2_direct10_from_high3b",
    },
]

COPY_HORIZONS_NS = [
    1.25,
    2.5,
    5.0,
    10.0,
    12.5,
    25.0,
    50.0,
    75.0,
    100.0,
    125.0,
    250.0,
    500.0,
    1000.0,
    2000.0,
]


def as_str_list(values):
    out = []
    for value in values:
        if isinstance(value, (bytes, bytearray)):
            out.append(value.decode())
        else:
            out.append(str(value))
    return out


def read_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
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


def load_tchw(path):
    with h5py.File(path, "r") as f:
        data = np.asarray(f["data_tchw"][()], dtype=np.float32)
        timesteps = np.asarray(f["timesteps"][()], dtype=np.int64)
        props = as_str_list(f["props"][()]) if "props" in f else CHANNEL_FALLBACK
    if data.ndim != 4:
        raise ValueError(f"data_tchw must be 4D, got {data.shape}")
    if data.shape[0] == len(timesteps):
        out = data
    elif data.shape[-1] == len(timesteps):
        out = np.transpose(data, (3, 2, 0, 1))
    else:
        raise ValueError(f"Cannot infer H5 layout: data={data.shape}, timesteps={len(timesteps)}")
    return np.ascontiguousarray(out.astype(np.float32)), timesteps, props


def infer_frame_dt_ns(timesteps, base_dt_ns):
    if len(timesteps) < 2:
        return float(base_dt_ns), 1
    step = int(round(float(np.median(np.diff(timesteps)))))
    return float(step * base_dt_ns), step


def copy_difficulty_case(case):
    if not case["h5"].exists():
        print(f"[SKIP] missing H5: {case['h5']}", flush=True)
        return []
    data, timesteps, channels = load_tchw(case["h5"])
    frame_dt_ns, timestep_stride = infer_frame_dt_ns(timesteps, case["base_dt_ns"])
    rows = []
    print(f"[COPY] {case['key']}: data={data.shape}, frame_dt={frame_dt_ns:g} ns", flush=True)
    for horizon_ns in COPY_HORIZONS_NS:
        frame_shift = int(round(horizon_ns / frame_dt_ns))
        if frame_shift < 1:
            continue
        actual_horizon = frame_shift * frame_dt_ns
        if abs(actual_horizon - horizon_ns) > max(1e-6, frame_dt_ns * 1e-4):
            continue
        if frame_shift >= data.shape[0]:
            continue
        for ci, channel in enumerate(channels):
            diff = data[frame_shift:, ci].astype(np.float64) - data[:-frame_shift, ci].astype(np.float64)
            mse_per_frame = np.mean(diff * diff, axis=(1, 2))
            stats = finite_stats(mse_per_frame)
            rows.append(
                {
                    "case": case["key"],
                    "physical_dataset": case["physical_dataset"],
                    "label": case["label"],
                    "h5": str(case["h5"]),
                    "base_dt_ns": case["base_dt_ns"],
                    "timestep_stride": timestep_stride,
                    "frame_dt_ns": frame_dt_ns,
                    "horizon_ns": actual_horizon,
                    "frame_shift": frame_shift,
                    "channel": channel,
                    "copy_mse_mean": stats["mean"],
                    "copy_mse_median": stats["median"],
                    "copy_mse_q25": stats["q25"],
                    "copy_mse_q75": stats["q75"],
                    "n_frame_pairs": stats["n"],
                }
            )
    return rows


def run_copy_difficulty():
    rows = []
    for case in H5_CASES:
        rows.extend(copy_difficulty_case(case))
    write_csv(rows, OUTDIR / "copy_difficulty_by_horizon.csv")
    if rows:
        plot_copy_difficulty(rows)
        plot_copy_phi_heatmap(rows)
    return rows


def plot_copy_difficulty(rows):
    channels = sorted({row["channel"] for row in rows})
    cases = {case["key"]: case for case in H5_CASES}
    for channel in channels:
        fig, ax = plt.subplots(figsize=(11.2, 6.4))
        for case_key in [case["key"] for case in H5_CASES]:
            case_rows = [row for row in rows if row["case"] == case_key and row["channel"] == channel]
            if not case_rows:
                continue
            x = np.asarray([float(row["horizon_ns"]) for row in case_rows], dtype=np.float64)
            y = np.asarray([float(row["copy_mse_mean"]) for row in case_rows], dtype=np.float64)
            order = np.argsort(x)
            ax.plot(x[order], y[order], marker="o", linewidth=1.6, label=cases[case_key]["label"])
        ax.set_xlabel("Physical horizon from copied frame (ns)")
        ax.set_ylabel(f"Copy MSE ({channel}, normalized)")
        ax.set_title(f"Copy baseline difficulty by physical horizon: {channel}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle=":", alpha=0.55)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
        fig.tight_layout(rect=(0, 0, 0.78, 1))
        path = OUTDIR / f"copy_difficulty_{channel}_by_horizon.png"
        fig.savefig(path, dpi=190)
        plt.close(fig)
        print(f"[PLOT] {path}", flush=True)


def plot_copy_phi_heatmap(rows):
    phi_rows = [row for row in rows if row["channel"] == "phi"]
    cases = [case["key"] for case in H5_CASES if any(row["case"] == case["key"] for row in phi_rows)]
    horizons = sorted({float(row["horizon_ns"]) for row in phi_rows})
    grid = np.full((len(cases), len(horizons)), np.nan, dtype=np.float64)
    by_key = {(row["case"], float(row["horizon_ns"])): float(row["copy_mse_mean"]) for row in phi_rows}
    for i, case_key in enumerate(cases):
        for j, horizon in enumerate(horizons):
            grid[i, j] = by_key.get((case_key, horizon), np.nan)
    fig, ax = plt.subplots(figsize=(12.8, 5.8))
    masked = np.ma.masked_invalid(np.log10(grid))
    im = ax.imshow(masked, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(cases)), [next(case["label"] for case in H5_CASES if case["key"] == key) for key in cases])
    ax.set_xticks(np.arange(len(horizons)), [f"{h:g}" for h in horizons], rotation=45, ha="right")
    ax.set_xlabel("Physical horizon from copied frame (ns)")
    ax.set_title("Copy baseline difficulty heatmap: phi log10(MSE)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10 copy MSE")
    fig.tight_layout()
    path = OUTDIR / "copy_difficulty_phi_heatmap.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f"[PLOT] {path}", flush=True)


def raw_prediction_path(case, method):
    tag = MODEL_TAGS[method]
    return WORKDIRS / f"{case['prefix']}_{tag}_training_compatible" / "low_magnet_direct10_raw_predictions.csv"


def run_transfer_skill():
    rows = []
    skipped = []
    for case in TRANSFER_CASES:
        for method in [m for m in METHOD_ORDER if m != "copy"]:
            path = raw_prediction_path(case, method)
            if not path.exists():
                skipped.append({"case": case["key"], "method": method, "path": str(path)})
                continue
            raw_rows = read_csv(path)
            rows.extend(aggregate_prediction_rows(case, method, raw_rows))
        data_path = raw_prediction_path(case, "data_only")
        if data_path.exists():
            rows.extend(aggregate_copy_rows(case, read_csv(data_path)))
    rows.sort(key=lambda row: (row["physical_dataset"], row["case"], row["channel"], row["horizon_ns"], row["method"]))
    write_csv(rows, OUTDIR / "transfer_physical_horizon_skill.csv")
    common_rows = build_common_horizon_rows(rows)
    write_csv(common_rows, OUTDIR / "transfer_stride1_vs_stride2_common_horizons.csv")
    best_rows = build_best_rows(rows)
    write_csv(best_rows, OUTDIR / "transfer_best_method_by_horizon.csv")
    if rows:
        plot_transfer_skill(rows)
        plot_transfer_ratio(rows)
    meta = {
        "description": (
            "Teacher-forced direct10 transfer results regrouped by physical prediction horizon. "
            "Skill score is 1 - model_mse_mean / copy_mse_mean; positive values beat copy."
        ),
        "output_dir": str(OUTDIR),
        "skipped_predictions": skipped,
    }
    path = OUTDIR / "transfer_physical_horizon_summary.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[JSON] {path}", flush=True)
    return rows


def aggregate_prediction_rows(case, method, raw_rows):
    grouped = defaultdict(list)
    for row in raw_rows:
        key = (row["channel"], float(row["horizon_ns"]))
        grouped[key].append(row)
    rows = []
    for (channel, horizon_ns), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        model_mse = [float(row["model_mse"]) for row in group]
        copy_mse = [float(row["copy_mse"]) for row in group]
        corr = [float(row["corr"]) for row in group]
        copy_stats = finite_stats(copy_mse)
        model_stats = finite_stats(model_mse)
        corr_stats = finite_stats(corr)
        ratio = model_stats["mean"] / copy_stats["mean"] if copy_stats["mean"] > 0 else np.nan
        rows.append(
            {
                "physical_dataset": case["physical_dataset"],
                "case": case["key"],
                "case_label": case["label"],
                "sampling": case["sampling"],
                "channel": channel,
                "method": method,
                "label": METHOD_LABELS[method],
                "horizon_ns": horizon_ns,
                "model_mse_mean": model_stats["mean"],
                "model_mse_median": model_stats["median"],
                "copy_mse_mean": copy_stats["mean"],
                "copy_mse_median": copy_stats["median"],
                "model_over_copy_mean": ratio,
                "skill_score_mean": 1.0 - ratio if np.isfinite(ratio) else np.nan,
                "corr_median": corr_stats["median"],
                "n": model_stats["n"],
            }
        )
    return rows


def aggregate_copy_rows(case, raw_rows):
    grouped = defaultdict(list)
    for row in raw_rows:
        key = (row["channel"], float(row["horizon_ns"]))
        grouped[key].append(float(row["copy_mse"]))
    rows = []
    for (channel, horizon_ns), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        stats = finite_stats(values)
        rows.append(
            {
                "physical_dataset": case["physical_dataset"],
                "case": case["key"],
                "case_label": case["label"],
                "sampling": case["sampling"],
                "channel": channel,
                "method": "copy",
                "label": METHOD_LABELS["copy"],
                "horizon_ns": horizon_ns,
                "model_mse_mean": stats["mean"],
                "model_mse_median": stats["median"],
                "copy_mse_mean": stats["mean"],
                "copy_mse_median": stats["median"],
                "model_over_copy_mean": 1.0,
                "skill_score_mean": 0.0,
                "corr_median": np.nan,
                "n": stats["n"],
            }
        )
    return rows


def build_common_horizon_rows(rows):
    out = []
    by_key = {}
    for row in rows:
        if row["channel"] != "phi":
            continue
        key = (row["physical_dataset"], row["method"], float(row["horizon_ns"]))
        by_key.setdefault(key, {})[row["sampling"]] = row
    for (dataset, method, horizon), vals in sorted(by_key.items()):
        if "stride1" not in vals or "stride2" not in vals:
            continue
        a = vals["stride1"]
        b = vals["stride2"]
        out.append(
            {
                "physical_dataset": dataset,
                "method": method,
                "label": a["label"],
                "horizon_ns": horizon,
                "stride1_model_mse_mean": a["model_mse_mean"],
                "stride2_model_mse_mean": b["model_mse_mean"],
                "stride1_over_stride2_mse": float(a["model_mse_mean"]) / float(b["model_mse_mean"])
                if float(b["model_mse_mean"]) > 0
                else np.nan,
                "stride1_skill": a["skill_score_mean"],
                "stride2_skill": b["skill_score_mean"],
                "stride1_minus_stride2_skill": float(a["skill_score_mean"]) - float(b["skill_score_mean"]),
            }
        )
    return out


def build_best_rows(rows):
    out = []
    grouped = defaultdict(list)
    for row in rows:
        if row["method"] == "copy":
            continue
        key = (row["physical_dataset"], row["case"], row["sampling"], row["channel"], float(row["horizon_ns"]))
        grouped[key].append(row)
    for key, group in sorted(grouped.items()):
        best = min(group, key=lambda row: float(row["model_mse_mean"]))
        out.append(
            {
                "physical_dataset": key[0],
                "case": key[1],
                "sampling": key[2],
                "channel": key[3],
                "horizon_ns": key[4],
                "best_method": best["method"],
                "best_label": best["label"],
                "best_model_mse_mean": best["model_mse_mean"],
                "copy_mse_mean": best["copy_mse_mean"],
                "best_model_over_copy_mean": best["model_over_copy_mean"],
                "best_skill_score_mean": best["skill_score_mean"],
            }
        )
    return out


def plot_transfer_skill(rows):
    for dataset in sorted({row["physical_dataset"] for row in rows}):
        for channel in ["phi"]:
            fig, ax = plt.subplots(figsize=(11.5, 6.3))
            subset = [row for row in rows if row["physical_dataset"] == dataset and row["channel"] == channel]
            for sampling in sorted({row["sampling"] for row in subset}):
                for method in METHOD_ORDER:
                    method_rows = [
                        row for row in subset if row["sampling"] == sampling and row["method"] == method
                    ]
                    if not method_rows:
                        continue
                    x = np.asarray([float(row["horizon_ns"]) for row in method_rows], dtype=np.float64)
                    y = np.asarray([float(row["skill_score_mean"]) for row in method_rows], dtype=np.float64)
                    order = np.argsort(x)
                    linestyle = "-" if sampling == "stride1" else "--"
                    marker = "o" if sampling == "stride1" else "s"
                    ax.plot(
                        x[order],
                        y[order],
                        color=METHOD_COLORS[method],
                        linestyle=linestyle,
                        marker=marker,
                        linewidth=1.5,
                        label=f"{METHOD_LABELS[method]} ({sampling})",
                    )
            ax.axhline(0.0, color="#333333", linestyle=":", linewidth=1.2, label="copy parity")
            ax.set_xlabel("Physical prediction horizon from last input frame (ns)")
            ax.set_ylabel("Skill score = 1 - model MSE / copy MSE")
            ax.set_title(f"{dataset}: phi skill score by physical horizon")
            ax.grid(True, linestyle=":", alpha=0.55)
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
            fig.tight_layout(rect=(0, 0, 0.76, 1))
            path = OUTDIR / f"transfer_phi_skill_{dataset}.png"
            fig.savefig(path, dpi=190)
            plt.close(fig)
            print(f"[PLOT] {path}", flush=True)


def plot_transfer_ratio(rows):
    for dataset in sorted({row["physical_dataset"] for row in rows}):
        subset = [row for row in rows if row["physical_dataset"] == dataset and row["channel"] == "phi"]
        fig, ax = plt.subplots(figsize=(11.5, 6.3))
        for sampling in sorted({row["sampling"] for row in subset}):
            for method in METHOD_ORDER:
                method_rows = [row for row in subset if row["sampling"] == sampling and row["method"] == method]
                if not method_rows:
                    continue
                x = np.asarray([float(row["horizon_ns"]) for row in method_rows], dtype=np.float64)
                y = np.asarray([float(row["model_over_copy_mean"]) for row in method_rows], dtype=np.float64)
                order = np.argsort(x)
                linestyle = "-" if sampling == "stride1" else "--"
                marker = "o" if sampling == "stride1" else "s"
                ax.plot(
                    x[order],
                    y[order],
                    color=METHOD_COLORS[method],
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=1.5,
                    label=f"{METHOD_LABELS[method]} ({sampling})",
                )
        ax.axhline(1.0, color="#333333", linestyle=":", linewidth=1.2, label="copy parity")
        ax.set_xlabel("Physical prediction horizon from last input frame (ns)")
        ax.set_ylabel("Model MSE / copy MSE")
        ax.set_title(f"{dataset}: phi model/copy by physical horizon")
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle=":", alpha=0.55)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
        fig.tight_layout(rect=(0, 0, 0.76, 1))
        path = OUTDIR / f"transfer_phi_model_over_copy_{dataset}.png"
        fig.savefig(path, dpi=190)
        plt.close(fig)
        print(f"[PLOT] {path}", flush=True)


def write_readme():
    text = """# Transfer Physical Horizon Comparison

This folder separates two questions that were mixed together in the previous transfer plots.

1. Copy baseline difficulty:
   `copy_difficulty_by_horizon.csv` measures how much each PIC testcase changes if the last
   frame is simply copied to a later physical time. Small values mean the testcase is easy for
   copy baseline.

2. Physical-horizon transfer skill:
   `transfer_physical_horizon_skill.csv` regroupes the existing direct10 transfer predictions
   by physical horizon. The key metric is `skill_score_mean = 1 - model_mse_mean / copy_mse_mean`.
   Positive skill means the model beats copy. Negative skill means copy is better.

Japanese note:

この結果は「モデルが悪いのか、copy baseline が強すぎる条件なのか」を切り分けるためのものです。
`copy_difficulty_*` は、各テストケースが何 ns 先でどの程度変化するかを表します。
`transfer_phi_skill_*` は、同じ物理時間先で見たときにモデルが copy に勝てるかを表します。
"""
    path = OUTDIR / "README.md"
    path.write_text(text, encoding="utf-8")
    print(f"[README] {path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-copy", action="store_true")
    parser.add_argument("--skip-transfer", action="store_true")
    args = parser.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if not args.skip_copy:
        run_copy_difficulty()
    if not args.skip_transfer:
        run_transfer_skill()
    write_readme()


if __name__ == "__main__":
    main()
