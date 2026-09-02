from pathlib import Path
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"C:\Users\astro\research\SimVPv2")
WORKDIRS = ROOT / "workdirs"
OUTDIR = WORKDIRS / "compare_b_sweep_overview_and_1p0_deep_dive"

CASES = [
    ("0.5 mT", "transfer_b0p5mT_stride{stride}_direct10_from_high3b"),
    ("1.0 mT", "transfer_b1p0mT_stride{stride}_direct10_from_high3b"),
    ("1.25 mT", "transfer_1p25_stride{stride}_direct10_from_high3b"),
    ("1.5 mT", "transfer_1p5mT_stride{stride}_direct10_from_high3b"),
    ("1.75 mT", "transfer_1p75_stride{stride}_direct10_from_high3b"),
]

COLORS = {
    "0.5 mT": "#64748b",
    "1.0 mT": "#16a34a",
    "1.25 mT": "#f59e0b",
    "1.5 mT": "#dc2626",
    "1.75 mT": "#7c3aed",
}


def find_case_dir(prefix):
    matches = sorted(
        d
        for d in WORKDIRS.iterdir()
        if d.is_dir()
        and d.name.startswith(prefix)
        and "data_only_training_compatible" in d.name
    )
    if not matches:
        raise FileNotFoundError(f"No workdir found for prefix: {prefix}")
    return matches[0]


def load_series():
    rows = []
    for label, pattern in CASES:
        for stride in (1, 2):
            prefix = pattern.format(stride=stride)
            d = find_case_dir(prefix)
            raw_path = d / "low_magnet_direct10_raw_predictions.csv"
            raw = pd.read_csv(raw_path, usecols=["target_time_us", "channel", "model_mse", "copy_mse"])
            raw = raw[raw["channel"].eq("phi")]
            df = (
                raw.groupby("target_time_us", as_index=False)
                .agg(model_mse_mean=("model_mse", "mean"), copy_mse_mean=("copy_mse", "mean"))
                .sort_values("target_time_us")
            )
            # Smooth only for visual readability; raw mean is still the source data.
            window = 41 if stride == 1 else 21
            df["model_mse_smooth"] = (
                df["model_mse_mean"].rolling(window=window, center=True, min_periods=1).median()
            )
            df["copy_mse_smooth"] = (
                df["copy_mse_mean"].rolling(window=window, center=True, min_periods=1).median()
            )
            df["model_over_copy"] = df["model_mse_mean"] / df["copy_mse_mean"]
            df["model_over_copy_smooth"] = (
                df["model_over_copy"].rolling(window=window, center=True, min_periods=1).median()
            )
            df["case"] = label
            df["stride"] = stride
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def plot(df, log=False):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.83, bottom=0.17, wspace=0.08)

    for ax, stride in zip(axes, (1, 2)):
        sub = df[df["stride"].eq(stride)]
        for case, g in sub.groupby("case", sort=False):
            ax.plot(
                g["target_time_us"],
                g["model_mse_smooth"],
                color=COLORS[case],
                linewidth=2.1,
                label=case,
            )
        ax.set_title(f"stride{stride}")
        ax.set_xlabel("target physical time [us]")
        ax.set_xlim(0.0, 50.0)
        ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
        if log:
            ax.set_yscale("log")

    axes[0].set_ylabel("phi MSE (normalized, rolling median)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.01))
    scale = "log" if log else "linear"
    fig.suptitle(f"3b data-only transfer: phi MSE by target time ({scale})", fontsize=15)

    suffix = "log" if log else "linear"
    png_path = OUTDIR / f"b_sweep_data_only_phi_mse_by_target_time_{suffix}.png"
    pdf_path = OUTDIR / f"b_sweep_data_only_phi_mse_by_target_time_{suffix}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {png_path}")
    print(f"[PLOT] {pdf_path}")


def plot_with_copy(df, log=False):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.83, bottom=0.19, wspace=0.08)

    for ax, stride in zip(axes, (1, 2)):
        sub = df[df["stride"].eq(stride)]
        for case, g in sub.groupby("case", sort=False):
            color = COLORS[case]
            ax.plot(
                g["target_time_us"],
                g["copy_mse_smooth"],
                color=color,
                linewidth=1.8,
                linestyle="--",
                alpha=0.55,
            )
            ax.plot(
                g["target_time_us"],
                g["model_mse_smooth"],
                color=color,
                linewidth=2.15,
                linestyle="-",
                label=case,
            )
        ax.set_title(f"stride{stride}")
        ax.set_xlabel("target physical time [us]")
        ax.set_xlim(0.0, 50.0)
        ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
        if log:
            ax.set_yscale("log")

    axes[0].set_ylabel("phi MSE (normalized, rolling median)")

    case_handles, case_labels = axes[0].get_legend_handles_labels()
    style_handles = [
        plt.Line2D([0], [0], color="#111827", linewidth=2.2, linestyle="-"),
        plt.Line2D([0], [0], color="#111827", linewidth=1.9, linestyle="--", alpha=0.6),
    ]
    style_labels = ["model", "copy baseline"]
    fig.legend(
        style_handles + case_handles,
        style_labels + case_labels,
        loc="lower center",
        ncol=7,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    scale = "log" if log else "linear"
    fig.suptitle(f"3b data-only transfer: model vs copy by target time ({scale})", fontsize=15)

    suffix = "log" if log else "linear"
    png_path = OUTDIR / f"b_sweep_data_only_phi_mse_by_target_time_with_copy_{suffix}.png"
    pdf_path = OUTDIR / f"b_sweep_data_only_phi_mse_by_target_time_with_copy_{suffix}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {png_path}")
    print(f"[PLOT] {pdf_path}")


def plot_model_over_copy(df, log=True):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.83, bottom=0.18, wspace=0.08)

    for ax, stride in zip(axes, (1, 2)):
        sub = df[df["stride"].eq(stride)]
        ax.axhline(1.0, color="#111827", linewidth=1.15, linestyle="--", alpha=0.8)
        for case, g in sub.groupby("case", sort=False):
            color = COLORS[case]
            ax.plot(
                g["target_time_us"],
                g["model_over_copy_smooth"],
                color=color,
                linewidth=2.1,
                label=case,
            )
        ax.set_title(f"stride{stride}")
        ax.set_xlabel("target physical time [us]")
        ax.set_xlim(0.0, 50.0)
        ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
        if log:
            ax.set_yscale("log")

    axes[0].set_ylabel("model / copy phi MSE (rolling median)")
    handles, labels = axes[0].get_legend_handles_labels()
    baseline = plt.Line2D([0], [0], color="#111827", linewidth=1.15, linestyle="--", alpha=0.8)
    fig.legend(
        [baseline] + handles,
        ["copy baseline (=1)"] + labels,
        loc="lower center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    scale = "log" if log else "linear"
    fig.suptitle(f"3b data-only transfer: phi MSE relative to copy ({scale})", fontsize=15)

    suffix = "log" if log else "linear"
    png_path = OUTDIR / f"b_sweep_data_only_phi_model_over_copy_by_target_time_{suffix}.png"
    pdf_path = OUTDIR / f"b_sweep_data_only_phi_model_over_copy_by_target_time_{suffix}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {png_path}")
    print(f"[PLOT] {pdf_path}")


def main():
    df = load_series()
    plot(df, log=False)
    plot(df, log=True)
    plot_with_copy(df, log=False)
    plot_with_copy(df, log=True)
    plot_model_over_copy(df, log=False)
    plot_model_over_copy(df, log=True)


if __name__ == "__main__":
    main()
