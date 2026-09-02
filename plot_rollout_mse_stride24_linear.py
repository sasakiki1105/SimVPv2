import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"C:\Users\astro\research\SimVPv2")
OUTDIR = ROOT / "workdirs" / "compare_rollout_mse_stride_overlay"
CSV_PATH = OUTDIR / "mse_phi_rollout_overlay.csv"

KEEP = [
    "stride2 tp0, 25 ns",
    "stride2 tout10, 25 ns",
    "stride4 tp0, 50 ns",
    "stride4 tout10, 50 ns",
]

STYLE = {
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
        label = header[i].removesuffix(" time_us")
        if label not in KEEP:
            continue
        times = []
        mses = []
        for row in rows:
            if row[i] == "" or row[i + 1] == "":
                continue
            times.append(float(row[i]))
            mses.append(float(row[i + 1]))
        series.append((label, np.asarray(times), np.asarray(mses)))
    return series


def plot(series):
    fig, ax = plt.subplots(figsize=(8.8, 5.0))

    ymax = 0.0
    for label, time_us, mse in series:
        style = STYLE[label]
        ymax = max(ymax, float(np.nanmax(mse)))
        ax.plot(
            time_us,
            mse,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.4,
            label=style["label"],
            alpha=0.96,
        )

    ax.set_xlim(0.0, 5.0)
    ax.set_ylim(0.0, ymax * 1.12)
    ax.set_xlabel("rollout physical time [us]")
    ax.set_ylabel("phi MSE (normalized)")
    ax.set_title("3b rollout phi MSE: stride2/4 only")
    ax.grid(True, linestyle=":", linewidth=0.75, alpha=0.6)
    ax.legend(frameon=False, ncol=2, loc="upper left")

    fig.tight_layout()
    png_path = OUTDIR / "mse_phi_rollout_overlay_stride24_linear.png"
    pdf_path = OUTDIR / "mse_phi_rollout_overlay_stride24_linear.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {png_path}")
    print(f"[PLOT] {pdf_path}")


def main():
    series = read_overlay_csv(CSV_PATH)
    missing = [label for label in KEEP if label not in {s[0] for s in series}]
    if missing:
        raise RuntimeError(f"Missing series in CSV: {missing}")
    plot(series)


if __name__ == "__main__":
    main()
