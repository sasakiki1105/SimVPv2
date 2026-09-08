"""Static comparison figure for the completed source-development audit."""
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    output = Path(__file__).resolve().parent / "workdirs/2D_RadAz/radaz_nextstep_baselines_v1"
    results = json.loads((output / "results.json").read_text(encoding="utf-8"))
    cases = list(results["cases"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.3), sharey=True)
    colors = ["#006C9C", "#D36B25"]
    for ax, observable, title in zip(axes, ("gamma", "flux_full"),
                                      ("Modal flux (n = 1–64)", "Band flux (all modes)")):
        for offset, label, caption, color in zip((-.16, .16),
                ("A_last_input", "ar10_validation_selected"),
                ("Existing A, last / input BN", "AR(10), validation-selected"), colors):
            values = [results["cases"][k]["comparisons"]["source_test"][observable][label]["skill_vs_copy"] for k in cases]
            ys = np.arange(len(cases)) + offset
            ax.scatter(values, ys, color=color, s=55, label=caption, zorder=3)
            for value, y in zip(values, ys):
                ax.annotate(f"{value:+.3f}", (value, y), xytext=(8, 0),
                            textcoords="offset points", ha="left", va="center", fontsize=9, color=color)
        ax.axvline(0, color="#777777", lw=1, ls="--")
        ax.axvline(1, color="#b0b0b0", lw=.7, ls=":")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Skill vs persistence = 1 − SSE / SSE(copy)")
        ax.grid(axis="x", alpha=.2)
        ax.set_xlim((-.95, 1.25) if observable == "gamma" else (-3.95, 1.65))
        ax.set_yticks(np.arange(len(cases)), cases)
        ax.tick_params(axis="y", labelleft=True)
    axes[0].invert_yaxis()
    fig.suptitle("A failure is condition dependent: a scalar baseline exposes remaining gaps", fontsize=13, y=.98)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .08), ncol=2, frameon=False)
    fig.text(.5, .025, "Source-test development windows only · 10 matched windows per condition · AR fits late-train 400 frames\n"
             "One PIC realization per condition · No independent-test or causal claim · Axis ranges differ", ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .14, 1, .94))
    fig.savefig(output / "baseline_skill.png", dpi=180)
    fig.savefig(output / "baseline_skill.pdf")
    plt.close(fig)
    print(output / "baseline_skill.png")


if __name__ == "__main__":
    main()
