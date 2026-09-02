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
OUTDIR = WORKDIRS / "compare_transfer_low_magnet_stride2_baseline_vs_floor_hinge"

CASES = {
    "baseline": {
        "label": "baseline",
        "color": "#1f77b4",
        "path": WORKDIRS
        / "transfer_low_magnet_stride2_direct10_from_high3b_training_compatible"
        / "low_magnet_direct10_raw_predictions.csv",
    },
    "floor_hinge": {
        "label": "floor hinge",
        "color": "#d62728",
        "path": WORKDIRS
        / "transfer_low_magnet_stride2_direct10_from_high3b_floor_hinge_lam1em3_alpha11_training_compatible"
        / "low_magnet_direct10_raw_predictions.csv",
    },
}

CHANNELS = ("electron_den", "ion_den", "phi")


def read_rows(case_name, path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out = {"case": case_name}
            for key, value in row.items():
                if key in ("channel",):
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
    summary_rows = []
    for case in CASES:
        case_rows = [r for r in rows if r["case"] == case]
        for channel in CHANNELS:
            group = [r for r in case_rows if r["channel"] == channel]
            out = {"case": case, "channel": channel}
            for metric in (
                "model_mse",
                "copy_mse",
                "model_over_copy",
                "corr",
                "peak_val_err",
                "peak_loc_err_px",
            ):
                stats = summarize([r[metric] for r in group])
                for stat_name, stat_value in stats.items():
                    out[f"{metric}_{stat_name}"] = stat_value
            summary_rows.append(out)

    by_channel = {}
    for row in summary_rows:
        by_channel[(row["case"], row["channel"])] = row
    delta_rows = []
    for channel in CHANNELS:
        base = by_channel[("baseline", channel)]
        hinge = by_channel[("floor_hinge", channel)]
        out = {"channel": channel}
        for metric in ("model_mse", "model_over_copy", "corr", "peak_val_err", "peak_loc_err_px"):
            for stat in ("mean", "median"):
                b = base.get(f"{metric}_{stat}", np.nan)
                h = hinge.get(f"{metric}_{stat}", np.nan)
                out[f"{metric}_{stat}_baseline"] = b
                out[f"{metric}_{stat}_floor_hinge"] = h
                out[f"{metric}_{stat}_delta_floor_minus_base"] = h - b
                out[f"{metric}_{stat}_ratio_floor_over_base"] = h / b if np.isfinite(b) and b != 0 else np.nan
        delta_rows.append(out)
    return summary_rows, delta_rows


def aggregate_by(rows, group_key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["case"], row["channel"], row[group_key])].append(row)
    out_rows = []
    for (case, channel, key), group in sorted(grouped.items()):
        row = {"case": case, "channel": channel, group_key: key}
        for metric in ("model_mse", "copy_mse", "model_over_copy", "corr"):
            stats = summarize([r[metric] for r in group])
            for stat_name, stat_value in stats.items():
                row[f"{metric}_{stat_name}"] = stat_value
        out_rows.append(row)
    return out_rows


def plot_channel_bars(summary_rows, metric, stat, ylabel, out_png, logy=False):
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    x = np.arange(len(CHANNELS))
    width = 0.36
    for i, case in enumerate(CASES):
        values = []
        for channel in CHANNELS:
            row = next(r for r in summary_rows if r["case"] == case and r["channel"] == channel)
            values.append(row[f"{metric}_{stat}"])
        offset = (i - 0.5) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            label=CASES[case]["label"],
            color=CASES[case]["color"],
            alpha=0.86,
            edgecolor="0.25",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNELS)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.92, facecolor="white", edgecolor="0.75")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_phi_by_horizon(horizon_rows, metric, stat, ylabel, out_png, logy=False):
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    all_x = []
    all_y = []
    for case in CASES:
        rows = [r for r in horizon_rows if r["case"] == case and r["channel"] == "phi"]
        rows.sort(key=lambda r: r["horizon_ns"])
        x = np.asarray([r["horizon_ns"] for r in rows], dtype=np.float64)
        y = np.asarray([r[f"{metric}_{stat}"] for r in rows], dtype=np.float64)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.0,
            markersize=4.2,
            label=CASES[case]["label"],
            color=CASES[case]["color"],
        )
        all_x.extend(x.tolist())
        all_y.extend(y.tolist())
    ax.set_xlabel("prediction horizon from last input frame [ns]")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    if all_x:
        span = max(all_x) - min(all_x)
        ax.set_xlim(min(all_x), max(all_x) + 0.22 * span)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.92, facecolor="white", edgecolor="0.75")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_phi_target_time(time_rows, out_png):
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    for case in CASES:
        rows = [r for r in time_rows if r["case"] == case and r["channel"] == "phi"]
        rows.sort(key=lambda r: r["target_time_us"])
        x = np.asarray([r["target_time_us"] for r in rows], dtype=np.float64)
        y = np.asarray([r["model_mse_mean"] for r in rows], dtype=np.float64)
        # Smooth only for visualization of the long time trend.
        if len(y) >= 21:
            kernel = np.ones(21, dtype=np.float64) / 21.0
            y_smooth = np.convolve(y, kernel, mode="same")
        else:
            y_smooth = y
        ax.plot(
            x,
            y_smooth,
            linewidth=1.8,
            label=f"{CASES[case]['label']} mean, smoothed",
            color=CASES[case]["color"],
        )
    ax.set_xlabel("target simulation time [us]")
    ax.set_ylabel("phi MSE, high3b-normalized")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.92, facecolor="white", edgecolor="0.75")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def write_readme(summary_rows, delta_rows):
    phi_delta = next(r for r in delta_rows if r["channel"] == "phi")
    electron_delta = next(r for r in delta_rows if r["channel"] == "electron_den")
    ion_delta = next(r for r in delta_rows if r["channel"] == "ion_den")
    text = f"""# Experiment

Compare low-magnet testcase 3a transfer results for two high-magnet testcase 3b models:

- `baseline`: stride2 direct10 model trained with data MSE only.
- `floor_hinge`: stride2 direct10 model trained with MSE plus the Poisson true-floor hinge loss.

Both evaluations use the same low-magnet H5 normalized with high3b training statistics and true input windows shifted by one retained frame. This is teacher-forced direct prediction, not rollout.

# Key Result

For `phi`, floor hinge did **not** improve the transfer MSE on low-magnet 3a:

- mean phi MSE ratio floor/baseline: `{phi_delta['model_mse_mean_ratio_floor_over_base']:.4g}`
- median phi MSE ratio floor/baseline: `{phi_delta['model_mse_median_ratio_floor_over_base']:.4g}`

For `electron_den`, floor hinge improved MSE slightly:

- mean electron density MSE ratio floor/baseline: `{electron_delta['model_mse_mean_ratio_floor_over_base']:.4g}`
- median electron density MSE ratio floor/baseline: `{electron_delta['model_mse_median_ratio_floor_over_base']:.4g}`

For `ion_den`, floor hinge slightly worsened MSE:

- mean ion density MSE ratio floor/baseline: `{ion_delta['model_mse_mean_ratio_floor_over_base']:.4g}`
- median ion density MSE ratio floor/baseline: `{ion_delta['model_mse_median_ratio_floor_over_base']:.4g}`

# Key Files

- `transfer_3a_baseline_vs_floor_hinge_summary.csv`: per-channel aggregate metrics.
- `transfer_3a_baseline_vs_floor_hinge_delta.csv`: floor-hinge minus baseline and floor/baseline ratios.
- `transfer_3a_baseline_vs_floor_hinge_by_horizon.csv`: horizon-wise aggregate metrics.
- `transfer_3a_phi_mse_by_horizon.png`: phi MSE by prediction horizon.
- `transfer_3a_phi_model_over_copy_by_horizon.png`: phi model/copy ratio by horizon.
- `transfer_3a_channel_mse_mean.png`: per-channel mean MSE comparison.
- `transfer_3a_channel_corr_median.png`: per-channel median correlation comparison.

# 日本語訳

このフォルダは、3b high magnetで学習した `baseline` モデルと、Poisson true-floor hinge lossを加えた `floor_hinge` モデルを、3a low magnetへ転移した結果の比較です。

どちらも同じ low magnet H5 を使い、入力には常に low magnet の真値PICフレームを使っています。予測結果を次の入力には使っていないため、rolloutではなく teacher-forced direct prediction です。

結論として、3b内では `floor_hinge` は有効でしたが、3a転移では `phi` のMSEはbaselineより少し悪化しました。一方で `electron_den` は少し改善しています。したがって、このPoisson lossだけで3aへの条件外汎化が改善したとはまだ言えません。
"""
    (OUTDIR / "README.md").write_text(text, encoding="utf-8")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for case, meta in CASES.items():
        rows.extend(read_rows(case, meta["path"]))

    summary_rows, delta_rows = aggregate_summary(rows)
    horizon_rows = aggregate_by(rows, "horizon_ns")
    time_rows = aggregate_by(rows, "target_time_us")

    write_csv(OUTDIR / "transfer_3a_baseline_vs_floor_hinge_summary.csv", summary_rows)
    write_csv(OUTDIR / "transfer_3a_baseline_vs_floor_hinge_delta.csv", delta_rows)
    write_csv(OUTDIR / "transfer_3a_baseline_vs_floor_hinge_by_horizon.csv", horizon_rows)
    write_csv(OUTDIR / "transfer_3a_baseline_vs_floor_hinge_by_target_time.csv", time_rows)

    plot_channel_bars(
        summary_rows,
        "model_mse",
        "mean",
        "mean MSE, high3b-normalized",
        OUTDIR / "transfer_3a_channel_mse_mean.png",
        logy=True,
    )
    plot_channel_bars(
        summary_rows,
        "corr",
        "median",
        "median Pearson correlation",
        OUTDIR / "transfer_3a_channel_corr_median.png",
        logy=False,
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
    plot_phi_target_time(time_rows, OUTDIR / "transfer_3a_phi_mse_target_time_smoothed.png")
    write_readme(summary_rows, delta_rows)

    summary = {
        "description": "Low-magnet 3a transfer comparison: high3b baseline vs Poisson floor-hinge model.",
        "cases": {case: str(meta["path"]) for case, meta in CASES.items()},
        "outputs": str(OUTDIR),
    }
    (OUTDIR / "transfer_3a_baseline_vs_floor_hinge_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"saved: {OUTDIR}")


if __name__ == "__main__":
    main()
