import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"C:\Users\astro\research\SimVPv2")
OUTDIR = ROOT / "workdirs" / "compare_rollout_mse_stride_overlay"
CSV_PATH = OUTDIR / "mse_phi_rollout_overlay.csv"

STYLE = {
    "stride1 tp0, 12.5 ns": {"color": "#333333", "linestyle": "-", "label": "stride1 tp0"},
    "stride1 tout10, 12.5 ns": {"color": "#333333", "linestyle": "--", "label": "stride1 tout10"},
    "stride2 tp0, 25 ns": {"color": "#2563eb", "linestyle": "-", "label": "stride2 tp0"},
    "stride2 tout10, 25 ns": {"color": "#2563eb", "linestyle": "--", "label": "stride2 tout10"},
    "stride4 tp0, 50 ns": {"color": "#16a34a", "linestyle": "-", "label": "stride4 tp0"},
    "stride4 tout10, 50 ns": {"color": "#16a34a", "linestyle": "--", "label": "stride4 tout10"},
}


def read_overlay_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    series = []
    for i in range(0, len(header), 2):
        time_name = header[i]
        mse_name = header[i + 1]
        label = time_name.removesuffix(" time_us")
        times = []
        mses = []
        for row in rows:
            if row[i] == "" or row[i + 1] == "":
                continue
            times.append(float(row[i]))
            mses.append(float(row[i + 1]))
        series.append((label, np.asarray(times), np.asarray(mses)))
    return series


def plot_slide(series):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.2, 4.8),
        gridspec_kw={"width_ratios": [1.2, 1.0]},
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.82, bottom=0.23, wspace=0.22)

    for ax in axes:
        for label, time_us, mse in series:
            style = STYLE[label]
            ax.plot(
                time_us,
                mse,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.2,
                label=style["label"],
                alpha=0.95,
            )
        ax.set_xlabel("rollout physical time [us]")
        ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.55)
        ax.set_xlim(0.0, 5.0)

    axes[0].set_yscale("log")
    axes[0].set_ylabel("phi MSE (normalized)")
    axes[0].set_title("All rollout curves")

    axes[1].set_ylim(0.0, 1.4e-3)
    axes[1].set_title("Zoom: low-MSE region (stride1 clipped)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.03),
        fontsize=10,
    )
    fig.suptitle("3b rollout phi MSE: stride and rollout update comparison", fontsize=15)

    png_path = OUTDIR / "mse_phi_rollout_overlay_slide.png"
    pdf_path = OUTDIR / "mse_phi_rollout_overlay_slide.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {png_path}")
    print(f"[PLOT] {pdf_path}")


def main():
    series = read_overlay_csv(CSV_PATH)
    plot_slide(series)


if __name__ == "__main__":
    main()
