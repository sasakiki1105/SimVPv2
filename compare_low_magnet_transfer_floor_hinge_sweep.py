import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"C:\Users\astro\research\SimVPv2")
WORKDIRS = ROOT / "workdirs"
OUTDIR = WORKDIRS / "compare_transfer_low_magnet_stride2_floor_hinge_sweep"

CASES = {
    "baseline": {
        "label": "baseline",
        "color": "#111111",
        "path": WORKDIRS
        / "transfer_low_magnet_stride2_direct10_from_high3b_training_compatible"
        / "low_magnet_direct10_raw_predictions.csv",
    },
    "lam1e-4_a1.0": {
        "label": "lambda=1e-4, alpha=1.0",
        "color": "#1f77b4",
        "path": WORKDIRS
        / "transfer_low_magnet_stride2_direct10_from_high3b_floor_hinge_lam1em4_alpha10_training_compatible"
        / "low_magnet_direct10_raw_predictions.csv",
    },
    "lam1e-3_a1.1": {
        "label": "lambda=1e-3, alpha=1.1",
        "color": "#d62728",
        "path": WORKDIRS
        / "transfer_low_magnet_stride2_direct10_from_high3b_floor_hinge_lam1em3_alpha11_training_compatible"
        / "low_magnet_direct10_raw_predictions.csv",
    },
    "lam1e-3_a1.2": {
        "label": "lambda=1e-3, alpha=1.2",
        "color": "#2ca02c",
        "path": WORKDIRS
        / "transfer_low_magnet_stride2_direct10_from_high3b_floor_hinge_lam1em3_alpha12_training_compatible"
        / "low_magnet_direct10_raw_predictions.csv",
    },
}

CHANNELS = ("electron_den", "ion_den", "phi")


def read_rows(case_name, path):
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out = {"case": case_name}
            for key, value in row.items():
                if key == "channel":
                    out[key] = value
                elif value == "" or value.lower() == "nan":
                    out[key] = np.nan
                else:
                    try:
                        out[key] = float(value)
                    except ValueError:
                        out[key] = value
            rows.append(out)
    return rows


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_summary(rows):
    out_rows = []
    for case in CASES:
        case_rows = [row for row in rows if row["case"] == case]
        for channel in CHANNELS:
            group = [row for row in case_rows if row["channel"] == channel]
            out = {"case": case, "label": CASES[case]["label"], "channel": channel}
            for metric in (
                "model_mse",
                "copy_mse",
                "model_over_copy",
                "corr",
                "peak_val_err",
                "peak_loc_err_px",
            ):
                stats = summarize([row[metric] for row in group])
                for stat_name, stat_value in stats.items():
                    out[f"{metric}_{stat_name}"] = stat_value
            out_rows.append(out)
    return out_rows


def aggregate_by(rows, group_key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["case"], row["channel"], row[group_key])].append(row)
    out_rows = []
    for (case, channel, key), group in sorted(grouped.items()):
        out = {
            "case": case,
            "label": CASES[case]["label"],
            "channel": channel,
            group_key: key,
            "n": len(group),
        }
        for metric in ("model_mse", "copy_mse", "model_over_copy", "corr"):
            stats = summarize([row[metric] for row in group])
            for stat_name, stat_value in stats.items():
                out[f"{metric}_{stat_name}"] = stat_value
        out_rows.append(out)
    return out_rows


def smooth_same(y, window=61):
    y = np.asarray(y, dtype=np.float64)
    if len(y) < 5:
        return y
    window = min(window, len(y))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return y
    pad = window // 2
    padded = np.pad(y, pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def setup_legend_space(ax, all_x, all_y, logy=False):
    if all_x:
        x_min = min(all_x)
        x_max = max(all_x)
        span = x_max - x_min
        ax.set_xlim(x_min, x_max + 0.22 * span)
    if all_y:
        finite = np.asarray(all_y, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if len(finite):
            y_min = float(np.min(finite))
            y_max = float(np.max(finite))
            if logy:
                y_min = max(y_min, 1e-12)
                ax.set_ylim(y_min * 0.8, y_max * 1.8)
            else:
                ax.set_ylim(min(0.0, y_min * 0.9), y_max * 1.18)


def plot_phi_target_time(time_rows, metric, stat, ylabel, path, logy=True):
    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    all_x = []
    all_y = []
    for case, meta in CASES.items():
        rows = [row for row in time_rows if row["case"] == case and row["channel"] == "phi"]
        rows.sort(key=lambda row: row["target_time_us"])
        x = np.asarray([row["target_time_us"] for row in rows], dtype=np.float64)
        y = np.asarray([row[f"{metric}_{stat}"] for row in rows], dtype=np.float64)
        y_smooth = smooth_same(y, window=61)
        ax.plot(x, y_smooth, linewidth=1.9, label=meta["label"], color=meta["color"])
        all_x.extend(x.tolist())
        all_y.extend(y_smooth.tolist())
    ax.set_xlabel("target simulation time [us]")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", linewidth=0.75, alpha=0.55)
    setup_legend_space(ax, all_x, all_y, logy=logy)
    ax.legend(loc="lower right", framealpha=0.94, facecolor="white", edgecolor="0.75")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_phi_by_horizon(horizon_rows, metric, stat, ylabel, path, logy=True):
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    all_x = []
    all_y = []
    for case, meta in CASES.items():
        rows = [row for row in horizon_rows if row["case"] == case and row["channel"] == "phi"]
        rows.sort(key=lambda row: row["horizon_ns"])
        x = np.asarray([row["horizon_ns"] for row in rows], dtype=np.float64)
        y = np.asarray([row[f"{metric}_{stat}"] for row in rows], dtype=np.float64)
        ax.plot(x, y, marker="o", markersize=4.0, linewidth=1.8, label=meta["label"], color=meta["color"])
        all_x.extend(x.tolist())
        all_y.extend(y.tolist())
    ax.set_xlabel("prediction horizon from last input frame [ns]")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", linewidth=0.75, alpha=0.55)
    setup_legend_space(ax, all_x, all_y, logy=logy)
    ax.legend(loc="lower right", framealpha=0.94, facecolor="white", edgecolor="0.75")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_channel_bars(summary_rows, metric, stat, ylabel, path, logy=False, legend_loc="lower right"):
    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    x = np.arange(len(CHANNELS))
    width = 0.19
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(CASES))
    for offset, (case, meta) in zip(offsets, CASES.items()):
        values = []
        for channel in CHANNELS:
            row = next(row for row in summary_rows if row["case"] == case and row["channel"] == channel)
            values.append(row[f"{metric}_{stat}"])
        ax.bar(x + offset, values, width=width, label=meta["label"], color=meta["color"], alpha=0.86)
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNELS)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.75, alpha=0.55)
    ax.legend(loc=legend_loc, framealpha=0.94, facecolor="white", edgecolor="0.75")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_delta_rows(summary_rows):
    baseline = {
        row["channel"]: row
        for row in summary_rows
        if row["case"] == "baseline"
    }
    rows = []
    for row in summary_rows:
        if row["case"] == "baseline":
            continue
        base = baseline[row["channel"]]
        out = {"case": row["case"], "label": row["label"], "channel": row["channel"]}
        for metric in ("model_mse", "model_over_copy", "corr", "peak_val_err", "peak_loc_err_px"):
            for stat in ("mean", "median"):
                b = base.get(f"{metric}_{stat}", np.nan)
                v = row.get(f"{metric}_{stat}", np.nan)
                out[f"{metric}_{stat}_baseline"] = b
                out[f"{metric}_{stat}_value"] = v
                out[f"{metric}_{stat}_delta"] = v - b
                out[f"{metric}_{stat}_ratio"] = v / b if np.isfinite(b) and b != 0 else np.nan
        rows.append(out)
    return rows


def write_readme(summary_rows):
    phi_rows = [row for row in summary_rows if row["channel"] == "phi"]
    best_phi = min(phi_rows, key=lambda row: row["model_mse_mean"])
    best_corr = max(phi_rows, key=lambda row: row["corr_median"])
    text = f"""# Low-magnet 3a transfer: floor-hinge sweep subset

This folder compares four high-magnet 3b stride2 direct10 models on low-magnet 3a:

- baseline data-MSE-only model
- lambda=1e-4, alpha=1.0
- lambda=1e-3, alpha=1.1
- lambda=1e-3, alpha=1.2

All evaluations use true low-magnet PIC input windows shifted by one retained frame. This is teacher-forced direct prediction, not rollout.

## Key result

- best phi mean MSE: `{best_phi['model_mse_mean']:.8g}` at `{best_phi['label']}`
- best phi median correlation: `{best_corr['corr_median']:.8g}` at `{best_corr['label']}`

## Key files

- `transfer_3a_floor_hinge_sweep_summary.csv`: per-channel aggregate metrics.
- `transfer_3a_floor_hinge_sweep_delta_vs_baseline.csv`: each floor-hinge case compared with baseline.
- `transfer_3a_phi_mse_target_time_smoothed.png`: main target-time smoothed phi MSE plot.
- `transfer_3a_phi_mse_target_time_smoothed_linear.png`: same plot with linear y-axis.
- `transfer_3a_phi_model_over_copy_target_time_smoothed.png`: target-time smoothed model/copy ratio.
- `transfer_3a_phi_mse_by_horizon.png`: phi MSE by output horizon.
- `transfer_3a_channel_mse_mean.png`: per-channel MSE.
- `transfer_3a_channel_corr_median.png`: per-channel median correlation.
"""
    (OUTDIR / "README.md").write_text(text, encoding="utf-8")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for case, meta in CASES.items():
        rows.extend(read_rows(case, meta["path"]))

    summary_rows = aggregate_summary(rows)
    delta_rows = make_delta_rows(summary_rows)
    horizon_rows = aggregate_by(rows, "horizon_ns")
    time_rows = aggregate_by(rows, "target_time_us")

    write_csv(OUTDIR / "transfer_3a_floor_hinge_sweep_summary.csv", summary_rows)
    write_csv(OUTDIR / "transfer_3a_floor_hinge_sweep_delta_vs_baseline.csv", delta_rows)
    write_csv(OUTDIR / "transfer_3a_floor_hinge_sweep_by_horizon.csv", horizon_rows)
    write_csv(OUTDIR / "transfer_3a_floor_hinge_sweep_by_target_time.csv", time_rows)

    plot_phi_target_time(
        time_rows,
        "model_mse",
        "mean",
        "phi MSE, high3b-normalized",
        OUTDIR / "transfer_3a_phi_mse_target_time_smoothed.png",
        logy=True,
    )
    plot_phi_target_time(
        time_rows,
        "model_mse",
        "mean",
        "phi MSE, high3b-normalized",
        OUTDIR / "transfer_3a_phi_mse_target_time_smoothed_linear.png",
        logy=False,
    )
    plot_phi_target_time(
        time_rows,
        "model_over_copy",
        "median",
        "phi median model/copy MSE ratio",
        OUTDIR / "transfer_3a_phi_model_over_copy_target_time_smoothed.png",
        logy=True,
    )
    plot_phi_by_horizon(
        horizon_rows,
        "model_mse",
        "mean",
        "phi mean MSE, high3b-normalized",
        OUTDIR / "transfer_3a_phi_mse_by_horizon.png",
        logy=True,
    )
    plot_phi_by_horizon(
        horizon_rows,
        "model_over_copy",
        "median",
        "phi median model/copy MSE ratio",
        OUTDIR / "transfer_3a_phi_model_over_copy_by_horizon.png",
        logy=True,
    )
    plot_channel_bars(
        summary_rows,
        "model_mse",
        "mean",
        "mean MSE, high3b-normalized",
        OUTDIR / "transfer_3a_channel_mse_mean.png",
        logy=True,
        legend_loc="upper right",
    )
    plot_channel_bars(
        summary_rows,
        "corr",
        "median",
        "median Pearson correlation",
        OUTDIR / "transfer_3a_channel_corr_median.png",
        logy=False,
    )
    write_readme(summary_rows)

    summary = {
        "description": "Low-magnet 3a transfer comparison for selected high3b Poisson floor-hinge models.",
        "cases": {case: str(meta["path"]) for case, meta in CASES.items()},
        "outdir": str(OUTDIR),
    }
    (OUTDIR / "transfer_3a_floor_hinge_sweep_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"saved: {OUTDIR}")


if __name__ == "__main__":
    main()
