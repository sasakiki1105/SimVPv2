from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"C:\Users\astro\research\SimVPv2")
OUTDIR = ROOT / "workdirs" / "compare_b_sweep_overview_and_1p0_deep_dive"
CSV_PATH = OUTDIR / "b_sweep_transfer_phi_summary.csv"


def main():
    df = pd.read_csv(CSV_PATH)
    df = df[df["method"].eq("data_only")].copy()
    df = df.sort_values(["B_mT", "stride"])

    cases = list(dict.fromkeys(df["case"]))
    x = np.arange(len(cases), dtype=float)
    width = 0.34

    stride1 = df[df["stride"].eq(1)].set_index("case")
    stride2 = df[df["stride"].eq(2)].set_index("case")
    y1 = [stride1.loc[c, "phi_model_mean_over_copy_mean"] for c in cases]
    y2 = [stride2.loc[c, "phi_model_mean_over_copy_mean"] for c in cases]

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.bar(
        x - width / 2,
        y1,
        width,
        label="stride1",
        color="#64748b",
        edgecolor="#334155",
        linewidth=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        y2,
        width,
        label="stride2",
        color="#2563eb",
        edgecolor="#1e3a8a",
        linewidth=0.8,
    )

    ax.axhline(1.0, color="#dc2626", linestyle="--", linewidth=1.8, label="copy baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.set_ylabel("phi MSE: model / copy baseline")
    ax.set_xlabel("transfer target B")
    ax.set_title("3b data-only model transfer to other B cases")
    ax.set_ylim(0.0, max(max(y1), max(y2)) * 1.18)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.75, alpha=0.6)

    # Highlight the only data-only case that beats copy.
    for bar, value in zip(bars2, y2):
        if value < 1.0:
            bar.set_color("#16a34a")
            bar.set_edgecolor("#166534")
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.12,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color="#166534",
                fontweight="bold",
            )

    for xpos, y in zip(x - width / 2, y1):
        ax.text(xpos, y + 0.12, f"{y:.2f}", ha="center", va="bottom", fontsize=9)
    for xpos, y in zip(x + width / 2, y2):
        if y >= 1.0:
            ax.text(xpos, y + 0.12, f"{y:.2f}", ha="center", va="bottom", fontsize=9)

    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()

    png_path = OUTDIR / "b_sweep_data_only_phi_model_over_copy.png"
    pdf_path = OUTDIR / "b_sweep_data_only_phi_model_over_copy.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[PLOT] {png_path}")
    print(f"[PLOT] {pdf_path}")


if __name__ == "__main__":
    main()
