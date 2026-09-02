import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CHANNELS = ["electron_den", "ion_den", "phi"]
COLORS = {
    "copy": "#6b7280",
    "data_only": "#2563eb",
    "floor_hinge": "#dc2626",
}
LABELS = {
    "copy": "Copy baseline",
    "data_only": "SimVPv2 data-only",
    "floor_hinge": "SimVPv2 + Poisson floor hinge",
}


def read_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def values(rows, channel, key):
    return np.asarray(
        [float(row[key]) for row in rows if row["channel"] == channel],
        dtype=np.float64,
    )


def finite_stats(array):
    array = np.asarray(array, dtype=np.float64)
    array = array[np.isfinite(array)]
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def write_csv(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] {path}")


def verify_alignment(data_rows, floor_rows):
    fields = ["window_start", "output_index", "target_timestep", "channel"]
    data_keys = [tuple(row[field] for field in fields) for row in data_rows]
    floor_keys = [tuple(row[field] for field in fields) for row in floor_rows]
    if data_keys != floor_keys:
        raise ValueError("The data-only and floor-hinge prediction rows are not aligned.")


def build_summary(data_rows, floor_rows):
    summary = []
    for channel in CHANNELS:
        metric_sets = {
            "copy": (
                values(data_rows, channel, "copy_mse"),
                values(data_rows, channel, "copy_corr"),
            ),
            "data_only": (
                values(data_rows, channel, "model_mse"),
                values(data_rows, channel, "corr"),
            ),
            "floor_hinge": (
                values(floor_rows, channel, "model_mse"),
                values(floor_rows, channel, "corr"),
            ),
        }
        copy_mean = finite_stats(metric_sets["copy"][0])["mean"]
        for method, (mse_values, corr_values) in metric_sets.items():
            mse_stats = finite_stats(mse_values)
            corr_stats = finite_stats(corr_values)
            summary.append({
                "channel": channel,
                "method": method,
                "label": LABELS[method],
                "n": mse_stats["n"],
                "mse_mean": mse_stats["mean"],
                "mse_median": mse_stats["median"],
                "mse_q25": mse_stats["q25"],
                "mse_q75": mse_stats["q75"],
                "corr_mean": corr_stats["mean"],
                "corr_median": corr_stats["median"],
                "mean_mse_over_copy": mse_stats["mean"] / copy_mean,
            })
    return summary


def plot_channel_mse(summary, outdir):
    x = np.arange(len(CHANNELS), dtype=np.float64)
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    for offset, method in zip([-width, 0.0, width], ["copy", "data_only", "floor_hinge"]):
        rows = [
            next(row for row in summary if row["channel"] == channel and row["method"] == method)
            for channel in CHANNELS
        ]
        ax.bar(
            x + offset,
            [row["mse_mean"] for row in rows],
            width=width,
            color=COLORS[method],
            label=LABELS[method],
        )
    ax.set_xticks(x, CHANNELS)
    ax.set_ylabel("Mean MSE (high3b-normalized)")
    ax.set_title("Ex-high-magnet transfer: channel-wise direct10 error")
    ax.set_yscale("log")
    ax.grid(True, axis="y", which="both", linestyle=":", alpha=0.55)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = outdir / "transfer_exhigh_channel_mse_mean.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f"[PLOT] {path}")


def group_phi_by(rows, group_key, value_key):
    groups = {}
    for row in rows:
        if row["channel"] != "phi":
            continue
        key = float(row[group_key])
        groups.setdefault(key, []).append(float(row[value_key]))
    return {
        key: {
            "mean": float(np.mean(group)),
            "median": float(np.median(group)),
            "q25": float(np.quantile(group, 0.25)),
            "q75": float(np.quantile(group, 0.75)),
            "n": int(len(group)),
        }
        for key, group in sorted(groups.items())
    }


def plot_phi_horizon(data_rows, floor_rows, outdir):
    series = {
        "copy": group_phi_by(data_rows, "horizon_ns", "copy_mse"),
        "data_only": group_phi_by(data_rows, "horizon_ns", "model_mse"),
        "floor_hinge": group_phi_by(floor_rows, "horizon_ns", "model_mse"),
    }
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    for method, grouped in series.items():
        x = np.asarray(list(grouped.keys()), dtype=np.float64)
        y = np.asarray([grouped[key]["mean"] for key in x], dtype=np.float64)
        ax.plot(x, y, marker="o", linewidth=1.8, color=COLORS[method], label=LABELS[method])
    ax.set_xlabel("Prediction horizon from last input frame (ns)")
    ax.set_ylabel("Mean phi MSE (high3b-normalized)")
    ax.set_title("Ex-high-magnet transfer: phi error by direct prediction horizon")
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = outdir / "transfer_exhigh_phi_mse_by_horizon.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f"[PLOT] {path}")

    rows = []
    for method, grouped in series.items():
        for horizon, stats in grouped.items():
            rows.append({
                "method": method,
                "label": LABELS[method],
                "horizon_ns": horizon,
                **stats,
            })
    write_csv(rows, outdir / "transfer_exhigh_phi_mse_by_horizon.csv")


def rolling_mean(y, window):
    if len(y) < window:
        return y.copy()
    left = window // 2
    right = window - 1 - left
    padded = np.pad(y, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def plot_phi_time(data_rows, floor_rows, outdir):
    series = {
        "copy": group_phi_by(data_rows, "target_time_us", "copy_mse"),
        "data_only": group_phi_by(data_rows, "target_time_us", "model_mse"),
        "floor_hinge": group_phi_by(floor_rows, "target_time_us", "model_mse"),
    }
    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    for method, grouped in series.items():
        x = np.asarray(list(grouped.keys()), dtype=np.float64)
        y = np.asarray([grouped[key]["mean"] for key in x], dtype=np.float64)
        ax.plot(x, y, color=COLORS[method], alpha=0.18, linewidth=0.8)
        ax.plot(
            x,
            rolling_mean(y, 11),
            color=COLORS[method],
            linewidth=2.0,
            label=f"{LABELS[method]} (11-point mean)",
        )
    ax.set_xlabel("Target simulation time (us)")
    ax.set_ylabel("Mean phi MSE (high3b-normalized)")
    max_time_us = max(float(row["target_time_us"]) for row in data_rows)
    ax.set_title(f"Ex-high-magnet transfer: phi error over the {max_time_us:g} us sequence")
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = outdir / "transfer_exhigh_phi_mse_target_time_smoothed.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f"[PLOT] {path}")


def clipping_summary(h5_path):
    with h5py.File(h5_path, "r") as f:
        data = np.asarray(f["data_tchw"][()], dtype=np.float32)
        timesteps = np.asarray(f["timesteps"][()], dtype=np.int64)
        stride = int(f["stride"][()]) if "stride" in f else -1
    rows = []
    for index, channel in enumerate(CHANNELS):
        values_channel = data[:, index]
        rows.append({
            "channel": channel,
            "fraction_at_zero": float(np.mean(values_channel <= 0.0)),
            "fraction_at_one": float(np.mean(values_channel >= 1.0)),
            "fraction_at_either_boundary": float(
                np.mean((values_channel <= 0.0) | (values_channel >= 1.0))
            ),
            "normalized_min": float(np.min(values_channel)),
            "normalized_max": float(np.max(values_channel)),
        })
    metadata = {
        "data_shape": list(data.shape),
        "first_timestep": int(timesteps[0]),
        "last_timestep": int(timesteps[-1]),
        "stored_timestep_stride": stride,
    }
    return rows, metadata


def write_readme(outdir, metadata, clip_rows, case_label, raw_dt_ns, retained_step):
    clip_text = "\n".join(
        f"- {row['channel']}: boundary fraction = {row['fraction_at_either_boundary']:.6f}"
        for row in clip_rows
    )
    effective_dt_ns = raw_dt_ns * retained_step
    sequence_us = metadata["last_timestep"] * raw_dt_ns / 1000.0
    text = f"""# Experiment

The 3b stride2 direct10 models are transferred without retraining to {case_label}, whose raw PIC frame interval is {raw_dt_ns:g} ns.

Every {retained_step} raw frame(s) are retained, so the effective model interval is {effective_dt_ns:g} ns, matching the 3b stride2 training interval. Each trial uses 10 true input frames and predicts the following 10 frames. The input window then advances by one retained true frame. This is teacher-forced direct prediction, not rollout.

Compared methods:

- Copy baseline: repeat the last input frame.
- SimVPv2 data-only: original 3b model.
- SimVPv2 + Poisson floor hinge: 3b model trained with lambda=1e-3 and alpha=1.2.

Data shape: `{metadata['data_shape']}`. Raw timestep range: `{metadata['first_timestep']}..{metadata['last_timestep']}`. Stored raw-timestep stride: `{metadata['stored_timestep_stride']}`.

The target data is normalized and clipped using the 3b training range. Boundary fractions are:

{clip_text}

# 日本語

3bで学習したstride2・direct10モデルを、再学習せずに{case_label}へ適用した比較です。

元PICの出力間隔は{raw_dt_ns:g} nsで、{retained_step}枚ごとに保持してモデルへ入れる実効間隔を{effective_dt_ns:g} nsにしています。これは3bのstride2学習条件と同じです。各試行では真値10枚から次の10枚を予測し、次は真値の入力窓を保持フレーム1枚だけ進めます。予測値を次の入力に戻さないためロールアウトではありません。

`transfer_exhigh_channel_mse_mean.png` は3チャネルの平均MSE、`transfer_exhigh_phi_mse_by_horizon.png` は25--250 ns先のphi誤差、`transfer_exhigh_phi_mse_target_time_smoothed.png` は{sequence_us:g} us全体でのphi誤差変化を示します。

正規化後に0または1へ張り付く割合が大きい場合、3bの学習分布外に出た値がクリップされているため、転用性能の解釈には注意が必要です。
"""
    path = outdir / "README.md"
    path.write_text(text, encoding="utf-8")
    print(f"[README] {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-only-csv", required=True)
    parser.add_argument("--floor-hinge-csv", required=True)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--case-label", default="the ex-high-magnet case")
    parser.add_argument("--raw-dt-ns", type=float, default=1.25)
    parser.add_argument("--retained-step", type=int, default=20)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data_rows = read_rows(args.data_only_csv)
    floor_rows = read_rows(args.floor_hinge_csv)
    verify_alignment(data_rows, floor_rows)

    summary = build_summary(data_rows, floor_rows)
    write_csv(summary, outdir / "transfer_exhigh_summary.csv")
    plot_channel_mse(summary, outdir)
    plot_phi_horizon(data_rows, floor_rows, outdir)
    plot_phi_time(data_rows, floor_rows, outdir)

    clip_rows, metadata = clipping_summary(args.h5)
    write_csv(clip_rows, outdir / "transfer_exhigh_normalization_clipping.csv")
    with open(outdir / "transfer_exhigh_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    write_readme(
        outdir,
        metadata,
        clip_rows,
        args.case_label,
        args.raw_dt_ns,
        args.retained_step,
    )


if __name__ == "__main__":
    main()
