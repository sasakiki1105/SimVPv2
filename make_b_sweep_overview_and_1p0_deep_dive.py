import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
WORKDIRS = ROOT / "workdirs"
OUTDIR = WORKDIRS / "compare_b_sweep_overview_and_1p0_deep_dive"

RAW_05_10 = WORKDIRS / "compare_raw_pic_b_sweep_0p5_1p0_metrics"
RAW_SINGLE_CASES = {
    "1.25 mT": WORKDIRS / "compare_raw_pic_b_sweep_1p25_metrics" / "raw_pic_metrics_summary.json",
    "1.5 mT": WORKDIRS / "compare_raw_pic_b_sweep_1p5_metrics" / "raw_pic_1p5_metrics_summary.json",
    "1.75 mT": WORKDIRS / "compare_raw_pic_b_sweep_1p75_metrics" / "raw_pic_metrics_summary.json",
}
TRANSFER_DATA_ONLY = WORKDIRS / "compare_transfer_b_sweep_stride1_stride2_data_only"
TRANSFER_STRIDE1_PHYSICS = WORKDIRS / "compare_transfer_b_sweep_stride1_physics_loss"
TRANSFER_STRIDE2_PHYSICS = WORKDIRS / "compare_transfer_b_sweep_stride2_physics_loss"
TRANSFER_COMBINED_CASES = {
    "1.25 mT": (
        WORKDIRS / "compare_transfer_b_sweep_1p25mT_combined"
        / "transfer_1p25mT_stride1_stride2_summary.csv"
    ),
    "1.5 mT": (
        WORKDIRS / "compare_transfer_b_sweep_1p5_combined"
        / "transfer_1p5_stride1_stride2_summary.csv"
    ),
    "1.75 mT": (
        WORKDIRS / "compare_transfer_b_sweep_1p75mT_combined"
        / "transfer_1p75mT_stride1_stride2_summary.csv"
    ),
}


CASE_ORDER = ["0.5 mT", "1.0 mT", "1.25 mT", "1.5 mT", "1.75 mT"]
B_MT = {"0.5 mT": 0.5, "1.0 mT": 1.0, "1.25 mT": 1.25, "1.5 mT": 1.5, "1.75 mT": 1.75}
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


def f(value):
    return float(value)


def load_raw_characteristics():
    raw = []
    with open(RAW_05_10 / "raw_pic_b_sweep_metrics_summary.json", "r", encoding="utf-8") as fp:
        summary = json.load(fp)["cases"]
    for key in ["b0p5mT", "b1p0mT"]:
        item = summary[key]
        collapse = item["copy_collapse_by_nrmse_std"]["1.0"]
        raw.append({
            "case": item["label"],
            "B_mT": B_MT[item["label"]],
            "dt_frame_ns": item["dt_ns"],
            "duration_us": item["duration_us"],
            "phi_std_mean": item["phi_std_mean"],
            "phi_rms_mean": item["phi_rms_mean"],
            "efield_rms_mean": item["efield_rms_mean_raw_efx_efy"],
            "main_fft_peak_MHz": item["fft_top_peaks_mhz"][0],
            "main_fft_period_ns": 1000.0 / item["fft_top_peaks_mhz"][0],
            "copy_collapse_ns_nrmse_std_ge_1": collapse["horizon_ns"],
            "copy_collapse_nrmse_std": collapse["copy_nrmse_by_mean_std"],
        })

    for label, path in RAW_SINGLE_CASES.items():
        if not path.exists():
            print(f"[WARN] missing raw summary: {path}", flush=True)
            continue
        with open(path, "r", encoding="utf-8") as fp:
            item = json.load(fp)
        collapse = item["copy_collapse_by_nrmse_std"]["1.0"]
        raw.append({
            "case": item["label"],
            "B_mT": item["B_mT"],
            "dt_frame_ns": item["dt_ns"],
            "duration_us": item["duration_us"],
            "phi_std_mean": item["phi_std_mean"],
            "phi_rms_mean": item["phi_rms_mean"],
            "efield_rms_mean": item["efield_rms_mean_raw_efx_efy"],
            "main_fft_peak_MHz": item["fft_top_peaks_mhz"][0],
            "main_fft_period_ns": 1000.0 / item["fft_top_peaks_mhz"][0],
            "copy_collapse_ns_nrmse_std_ge_1": collapse["horizon_ns"],
            "copy_collapse_nrmse_std": collapse["copy_nrmse_by_mean_std"],
        })
    return sorted(raw, key=lambda row: row["B_mT"])


def append_transfer_data_only(rows):
    for row in read_csv(TRANSFER_DATA_ONLY / "transfer_b_sweep_summary_by_channel.csv"):
        if row["channel"] != "phi":
            continue
        rows.append({
            "case": row["case"],
            "B_mT": B_MT[row["case"]],
            "stride": row["stride"],
            "method": "data_only",
            "label": "Data-only",
            "phi_model_mse_mean": row["model_mse_mean"],
            "phi_copy_mse_mean": row["copy_mse_mean"],
            "phi_model_mean_over_copy_mean": row["model_mean_over_copy_mean"],
            "phi_model_over_copy_median_of_ratios": row["model_over_copy_median_of_ratios"],
            "phi_model_corr_median": row["model_corr_median"],
            "phi_copy_corr_median": row["copy_corr_median"],
        })


def append_transfer_physics(rows, stride, path):
    for row in read_csv(path):
        if row["channel"] != "phi":
            continue
        method = row["method"] if "method" in row else row["method_key"]
        label = row["method_label"] if "method_label" in row else LABELS.get(method, method)
        rows.append({
            "case": row["dataset_label"] if "dataset_label" in row else row["case"],
            "B_mT": B_MT[row["dataset_label"] if "dataset_label" in row else row["case"]],
            "stride": str(stride),
            "method": method,
            "label": label,
            "phi_model_mse_mean": row["model_mse_mean"],
            "phi_copy_mse_mean": row["copy_mse_mean"],
            "phi_model_mean_over_copy_mean": row["model_mean_over_copy_mean"] if "model_mean_over_copy_mean" in row else row["model_over_copy_mean"],
            "phi_model_over_copy_median_of_ratios": row["model_over_copy_median_of_ratio"] if "model_over_copy_median_of_ratio" in row else row["model_over_copy_median_of_ratios"],
            "phi_model_corr_median": "",
            "phi_copy_corr_median": "",
        })


def append_transfer_combined(rows, label, path):
    if not path.exists():
        print(f"[WARN] missing transfer summary: {path}", flush=True)
        return
    table = read_csv(path)
    copy_by_stride = {}
    for row in table:
        if row["channel"] == "phi" and row["method"] == "copy":
            copy_by_stride[row["stride"].replace("stride", "")] = f(row["mse_mean"])
    for row in table:
        if row["channel"] != "phi" or row["method"] == "copy":
            continue
        stride = row["stride"].replace("stride", "")
        copy_mean = copy_by_stride[stride]
        model_mean = f(row["mse_mean"])
        rows.append({
            "case": label,
            "B_mT": B_MT[label],
            "stride": stride,
            "method": row["method"],
            "label": row.get("method_label", row.get("label", row["method"])),
            "phi_model_mse_mean": model_mean,
            "phi_copy_mse_mean": copy_mean,
            "phi_model_mean_over_copy_mean": model_mean / copy_mean,
            "phi_model_over_copy_median_of_ratios": row["median_mse_over_copy"],
            "phi_model_corr_median": row["corr_median"],
            "phi_copy_corr_median": "",
        })


def load_transfer_summary():
    rows = []
    append_transfer_data_only(rows)
    append_transfer_physics(
        rows,
        1,
        TRANSFER_STRIDE1_PHYSICS / "transfer_b_sweep_stride1_physics_summary_by_channel.csv",
    )
    append_transfer_physics(
        rows,
        2,
        TRANSFER_STRIDE2_PHYSICS / "transfer_b_sweep_stride2_physics_summary_by_channel.csv",
    )
    for label, path in TRANSFER_COMBINED_CASES.items():
        append_transfer_combined(rows, label, path)

    dedup = {}
    for row in rows:
        key = (row["case"], str(row["stride"]), row["method"])
        if key not in dedup:
            dedup[key] = row
    return sorted(dedup.values(), key=lambda row: (f(row["B_mT"]), int(row["stride"]), row["method"]))


def best_by_case_and_stride(transfer_rows):
    best = []
    for case in CASE_ORDER:
        for stride in ["1", "2"]:
            candidates = [
                row for row in transfer_rows
                if row["case"] == case and str(row["stride"]) == stride
            ]
            if not candidates:
                continue
            winner = min(candidates, key=lambda row: f(row["phi_model_mean_over_copy_mean"]))
            best.append({
                "case": case,
                "B_mT": B_MT[case],
                "stride": stride,
                "best_method": winner["method"],
                "best_label": winner["label"],
                "best_phi_model_mean_over_copy_mean": winner["phi_model_mean_over_copy_mean"],
                "best_phi_model_mse_mean": winner["phi_model_mse_mean"],
                "copy_phi_mse_mean": winner["phi_copy_mse_mean"],
            })
    return best


def plot_raw_characteristics(raw_rows):
    x = np.array([row["B_mT"] for row in raw_rows], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4))
    axes = axes.ravel()

    series = [
        ("phi_std_mean", "Mean spatial std of phi", "V"),
        ("efield_rms_mean", "Mean electric-field RMS", "raw gradient units"),
        ("main_fft_peak_MHz", "Top FFT peak frequency", "MHz"),
        ("copy_collapse_ns_nrmse_std_ge_1", "First copy-collapse horizon", "ns"),
    ]
    for ax, (key, title, ylabel) in zip(axes, series):
        y = np.array([row[key] for row in raw_rows], dtype=float)
        ax.plot(x, y, marker="o", color="#2563eb", linewidth=2.0)
        for row in raw_rows:
            ax.annotate(row["case"], (row["B_mT"], row[key]), textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.set_xlabel("Bz (mT)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.55)
    fig.tight_layout()
    path = OUTDIR / "b_sweep_raw_characteristics.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f"[PLOT] {path}", flush=True)


def plot_transfer_overview(transfer_rows):
    methods = ["data_only", "poisson_zero", "efield", "poisson_zero_efield", "floor_hinge"]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.5), sharey=True)
    for ax, stride in zip(axes, ["1", "2"]):
        for method in methods:
            rows = [
                row for row in transfer_rows
                if str(row["stride"]) == stride and row["method"] == method
            ]
            if not rows:
                continue
            rows = sorted(rows, key=lambda row: f(row["B_mT"]))
            xs = [f(row["B_mT"]) for row in rows]
            ys = [f(row["phi_model_mean_over_copy_mean"]) for row in rows]
            ax.plot(xs, ys, marker="o", linewidth=1.8, color=COLORS[method], label=LABELS[method])
        ax.axhline(1.0, color="#111827", linestyle="--", linewidth=1.0)
        ax.set_title(f"stride{stride}")
        ax.set_xlabel("Bz (mT)")
        ax.grid(True, which="both", linestyle=":", alpha=0.55)
        ax.set_yscale("log")
    axes[0].set_ylabel("Mean phi MSE / copy mean phi MSE")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.suptitle("Transfer to B-sweep cases: lower than 1 beats copy")
    fig.tight_layout(rect=[0, 0, 0.86, 0.94])
    path = OUTDIR / "b_sweep_transfer_phi_model_over_copy.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f"[PLOT] {path}", flush=True)


def plot_best(best_rows):
    fig, ax = plt.subplots(figsize=(10.6, 5.8))
    width = 0.32
    x = np.arange(len(CASE_ORDER), dtype=float)
    for si, stride in enumerate(["1", "2"]):
        vals = []
        labels = []
        for case in CASE_ORDER:
            row = next((r for r in best_rows if r["case"] == case and r["stride"] == stride), None)
            vals.append(np.nan if row is None else f(row["best_phi_model_mean_over_copy_mean"]))
            labels.append("" if row is None else row["best_label"])
        bars = ax.bar(x + (si - 0.5) * width, vals, width=width, label=f"stride{stride}")
        for bar, label in zip(bars, labels):
            if not label:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                label.replace("Weak ", "W. ").replace("Poisson ", "P. ").replace("E-field", "E"),
                ha="center",
                va="bottom",
                rotation=80,
                fontsize=7,
            )
    ax.axhline(1.0, color="#111827", linestyle="--", linewidth=1.0)
    ax.set_xticks(x, CASE_ORDER)
    ax.set_yscale("log")
    ax.set_ylabel("Best mean phi MSE / copy mean phi MSE")
    ax.set_title("Best transfer result at each B and stride")
    ax.grid(True, axis="y", which="both", linestyle=":", alpha=0.55)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = OUTDIR / "b_sweep_best_phi_model_over_copy.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f"[PLOT] {path}", flush=True)


def plot_1p0_raw_deep_dive():
    fft_rows = [
        row for row in read_csv(RAW_05_10 / "raw_pic_phi_fft_top_peaks.csv")
        if row["case"] == "b1p0mT"
    ]
    copy_rows = [
        row for row in read_csv(RAW_05_10 / "raw_pic_phi_copy_baseline_horizon.csv")
        if row["case"] == "b1p0mT" and f(row["horizon_ns"]) <= 500.0
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.6))

    ranks = [int(row["rank"]) for row in fft_rows]
    freqs = [f(row["frequency_mhz"]) for row in fft_rows]
    powers = [f(row["power_norm"]) for row in fft_rows]
    axes[0].bar(ranks, powers, color="#2563eb", alpha=0.85)
    for rank, freq, power in zip(ranks, freqs, powers):
        axes[0].text(rank, power, f"{freq:.2f}", ha="center", va="bottom", rotation=75, fontsize=8)
    axes[0].set_xlabel("FFT peak rank")
    axes[0].set_ylabel("Normalized power")
    axes[0].set_title("1.0 mT: top phi FFT peaks (labels: MHz)")
    axes[0].grid(True, axis="y", linestyle=":", alpha=0.55)

    xs = [f(row["horizon_ns"]) for row in copy_rows]
    ys = [f(row["phi_copy_nrmse_by_mean_std"]) for row in copy_rows]
    axes[1].plot(xs, ys, marker="o", color="#dc2626", linewidth=1.8)
    axes[1].axhline(1.0, color="#111827", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("Horizon (ns)")
    axes[1].set_ylabel("Copy RMSE / mean spatial std(phi)")
    axes[1].set_title("1.0 mT: copy baseline collapse")
    axes[1].grid(True, linestyle=":", alpha=0.55)

    fig.tight_layout()
    path = OUTDIR / "one_mT_raw_fft_and_copy_baseline.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f"[PLOT] {path}", flush=True)


def plot_1p0_data_only_horizon():
    rows = [
        row for row in read_csv(TRANSFER_DATA_ONLY / "transfer_b_sweep_phi_by_horizon.csv")
        if row["case"] == "1.0 mT"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6))
    for stride, color in [("1", "#7c3aed"), ("2", "#2563eb")]:
        sub = sorted([row for row in rows if row["stride"] == stride], key=lambda row: f(row["horizon_ns"]))
        xs = [f(row["horizon_ns"]) for row in sub]
        model = [f(row["model_mse_mean"]) for row in sub]
        copy = [f(row["copy_mse_mean"]) for row in sub]
        ratio = [f(row["model_mean_over_copy_mean"]) for row in sub]
        axes[0].plot(xs, model, marker="o", color=color, linewidth=1.8, label=f"model stride{stride}")
        axes[0].plot(xs, copy, marker="x", color=color, linewidth=1.3, linestyle="--", alpha=0.75, label=f"copy stride{stride}")
        axes[1].plot(xs, ratio, marker="o", color=color, linewidth=1.8, label=f"stride{stride}")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Prediction horizon (ns)")
    axes[0].set_ylabel("Mean phi MSE")
    axes[0].set_title("1.0 mT: data-only model vs copy")
    axes[0].grid(True, which="both", linestyle=":", alpha=0.55)
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].axhline(1.0, color="#111827", linestyle="--", linewidth=1.0)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Prediction horizon (ns)")
    axes[1].set_ylabel("Model MSE / copy MSE")
    axes[1].set_title("1.0 mT: why stride2 is the useful comparison")
    axes[1].grid(True, which="both", linestyle=":", alpha=0.55)
    axes[1].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    path = OUTDIR / "one_mT_data_only_horizon_deep_dive.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f"[PLOT] {path}", flush=True)


def plot_1p0_stride2_methods(transfer_rows):
    rows = [
        row for row in transfer_rows
        if row["case"] == "1.0 mT" and str(row["stride"]) == "2"
    ]
    rows = sorted(rows, key=lambda row: f(row["phi_model_mean_over_copy_mean"]))
    fig, ax = plt.subplots(figsize=(10.6, 5.8))
    labels = [row["label"] for row in rows]
    vals = [f(row["phi_model_mean_over_copy_mean"]) for row in rows]
    colors = [COLORS[row["method"]] for row in rows]
    bars = ax.bar(np.arange(len(rows)), vals, color=colors, alpha=0.88)
    ax.axhline(1.0, color="#111827", linestyle="--", linewidth=1.0)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(np.arange(len(rows)), labels, rotation=25, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Mean phi MSE / copy mean phi MSE")
    ax.set_title("1.0 mT stride2: method comparison")
    ax.grid(True, axis="y", which="both", linestyle=":", alpha=0.55)
    fig.tight_layout()
    path = OUTDIR / "one_mT_stride2_method_comparison.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f"[PLOT] {path}", flush=True)


def write_readme(raw_rows, best_rows):
    best_1p0 = next(row for row in best_rows if row["case"] == "1.0 mT" and row["stride"] == "2")
    text = f"""# B Sweep Overview and 1.0 mT Deep Dive

This folder summarizes the B-sweep transfer experiments from the high-magnet 3b source model.

Source training case:

- Testcase 3b high magnet.
- Bz = 0.2 mT.
- dt between raw output frames = 12.5 ns.

Target cases summarized here:

- 0.5 mT
- 1.0 mT
- 1.25 mT
- 1.5 mT
- 1.75 mT

Main files:

- `b_sweep_raw_characteristics.csv`
- `b_sweep_transfer_phi_summary.csv`
- `b_sweep_best_phi_transfer.csv`
- `b_sweep_raw_characteristics.png`
- `b_sweep_transfer_phi_model_over_copy.png`
- `b_sweep_best_phi_model_over_copy.png`
- `one_mT_raw_fft_and_copy_baseline.png`
- `one_mT_data_only_horizon_deep_dive.png`
- `one_mT_stride2_method_comparison.png`

Important reading:

- Values below 1.0 in model-over-copy plots mean the model beats the copy baseline.
- 1.0 mT stride2 remains the clearest transfer success among the current B sweep.
- Best 1.0 mT stride2 method: `{best_1p0['best_label']}`, model/copy = {float(best_1p0['best_phi_model_mean_over_copy_mean']):.3f}.
- 1.5 mT has a very strong copy baseline, especially around even-frame/25 ns structure, so beating copy is difficult even when the dynamics looks structured.

Interpretation:

The result is not monotonic in B. Transfer does not simply get worse as B moves away from 0.2 mT.
Instead, success depends on the relation between the target temporal pattern, the output interval, and the stride used for the model input/output.
The 1.0 mT case appears to sit in a useful regime where copy is no longer trivial, but the learned 3b dynamics still contains reusable image-time structure.
"""
    path = OUTDIR / "README.md"
    path.write_text(text, encoding="utf-8")
    print(f"[README] {path}", flush=True)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    raw_rows = load_raw_characteristics()
    transfer_rows = load_transfer_summary()
    best_rows = best_by_case_and_stride(transfer_rows)

    write_csv(raw_rows, OUTDIR / "b_sweep_raw_characteristics.csv")
    write_csv(transfer_rows, OUTDIR / "b_sweep_transfer_phi_summary.csv")
    write_csv(best_rows, OUTDIR / "b_sweep_best_phi_transfer.csv")

    plot_raw_characteristics(raw_rows)
    plot_transfer_overview(transfer_rows)
    plot_best(best_rows)
    plot_1p0_raw_deep_dive()
    plot_1p0_data_only_horizon()
    plot_1p0_stride2_methods(transfer_rows)
    write_readme(raw_rows, best_rows)


if __name__ == "__main__":
    main()
