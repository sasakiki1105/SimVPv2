import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"C:\Users\astro\research\SimVPv2")
OUTDIR = ROOT / "workdirs" / "compare_direct_prediction_target_time_aggregate"
COMMON_CSV = OUTDIR / "direct_mse_by_target_time_common_times.csv"
ALL_CSV = OUTDIR / "direct_mse_by_target_time_all_times.csv"

STYLE = {
    "stride1": {"color": "#333333", "label": "stride1, 12.5 ns"},
    "stride2": {"color": "#2563eb", "label": "stride2, 25 ns"},
    "stride3": {"color": "#9333ea", "label": "stride3, 37.5 ns"},
    "stride4": {"color": "#16a34a", "label": "stride4, 50 ns"},
}


def read_wide_common(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    x = np.asarray([float(row["target_time_us"]) for row in rows], dtype=np.float64)
    data = {}
    for stride in STYLE:
        if f"{stride}_mean" not in rows[0]:
            continue
        data[stride] = {
            "x": x,
            "mean": np.asarray([float(row[f"{stride}_mean"]) for row in rows], dtype=np.float64),
            "median": np.asarray([float(row[f"{stride}_median"]) for row in rows], dtype=np.float64),
            "q25": np.asarray([float(row[f"{stride}_q25"]) for row in rows], dtype=np.float64),
            "q75": np.asarray([float(row[f"{stride}_q75"]) for row in rows], dtype=np.float64),
            "count": np.asarray([int(row[f"{stride}_count"]) for row in rows], dtype=np.int32),
        }
    return data


def read_long_all(path):
    data = {stride: {"x": [], "mean": [], "median": [], "q25": [], "q75": [], "count": []} for stride in STYLE}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stride = row["case"]
            if stride not in data:
                continue
            data[stride]["x"].append(float(row["target_time_us"]))
            for key in ("mean", "median", "q25", "q75"):
                data[stride][key].append(float(row[key]))
            data[stride]["count"].append(int(row["count"]))

    out = {}
    for stride, values in data.items():
        if not values["x"]:
            continue
        order = np.argsort(values["x"])
        out[stride] = {
            key: np.asarray(values[key], dtype=np.float64)[order]
            for key in ("x", "mean", "median", "q25", "q75")
        }
        out[stride]["count"] = np.asarray(values["count"], dtype=np.int32)[order]
    return out


def plot_common_slide(data, strides, suffix):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.82, bottom=0.23, wspace=0.22)

    for stride in strides:
        style = STYLE[stride]
        values = data[stride]
        x = values["x"]
        axes[0].plot(x, values["mean"], color=style["color"], linewidth=2.4, label=style["label"])
        axes[0].fill_between(x, values["q25"], values["q75"], color=style["color"], alpha=0.12, linewidth=0)
        axes[1].plot(x, values["median"], color=style["color"], linewidth=2.4, label=style["label"])
        axes[1].fill_between(x, values["q25"], values["q75"], color=style["color"], alpha=0.12, linewidth=0)

    axes[0].set_title("Mean MSE at common target times")
    axes[1].set_title("Median MSE at common target times")
    for ax in axes:
        ax.set_xlabel("target simulation time [us]")
        ax.set_ylabel("phi MSE (normalized)")
        ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.55)
        ax.set_xlim(min(data[strides[0]]["x"]), max(data[strides[0]]["x"]))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(strides), frameon=False, bbox_to_anchor=(0.5, 0.03), fontsize=10)
    fig.suptitle("3b teacher-forced direct prediction: stride comparison", fontsize=15)

    png_path = OUTDIR / f"direct_mse_target_time_overlay_common_{suffix}.png"
    pdf_path = OUTDIR / f"direct_mse_target_time_overlay_common_{suffix}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {png_path}")
    print(f"[PLOT] {pdf_path}")


def plot_all_times_slide(data, strides, suffix):
    fig, ax = plt.subplots(figsize=(10.8, 4.9), constrained_layout=False)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.82, bottom=0.24)

    for stride in strides:
        style = STYLE[stride]
        values = data[stride]
        ax.plot(values["x"], values["median"], color=style["color"], linewidth=2.1, label=style["label"])
        ax.fill_between(values["x"], values["q25"], values["q75"], color=style["color"], alpha=0.10, linewidth=0)

    ax.set_title("Median MSE over all available target times")
    ax.set_xlabel("target simulation time [us]")
    ax.set_ylabel("phi MSE (normalized)")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.55)
    ax.set_xlim(45.0, 50.0)
    ax.legend(loc="lower center", ncol=len(strides), frameon=False, bbox_to_anchor=(0.5, -0.28), fontsize=10)
    fig.suptitle("3b teacher-forced direct prediction over the test interval", fontsize=15)

    png_path = OUTDIR / f"direct_mse_target_time_overlay_all_times_{suffix}.png"
    pdf_path = OUTDIR / f"direct_mse_target_time_overlay_all_times_{suffix}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {png_path}")
    print(f"[PLOT] {pdf_path}")


def write_summary(data, strides, suffix):
    summary = {}
    for stride in strides:
        values = data[stride]
        summary[stride] = {
            "label": STYLE[stride]["label"],
            "n_target_times": int(len(values["x"])),
            "target_time_us_first": float(values["x"][0]),
            "target_time_us_last": float(values["x"][-1]),
            "mean_of_mean": float(np.mean(values["mean"])),
            "median_of_median": float(np.median(values["median"])),
            "mean_of_median": float(np.mean(values["median"])),
        }
    path = OUTDIR / f"direct_mse_target_time_overlay_summary_{suffix}.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[JSON] {path}")


def main():
    common = read_wide_common(COMMON_CSV)
    all_times = read_long_all(ALL_CSV)

    strides_124 = ["stride1", "stride2", "stride4"]
    strides_1234 = ["stride1", "stride2", "stride3", "stride4"]

    plot_common_slide(common, strides_124, "stride124")
    plot_common_slide(common, strides_1234, "stride1234")
    plot_all_times_slide(all_times, strides_124, "stride124")
    plot_all_times_slide(all_times, strides_1234, "stride1234")
    write_summary(common, strides_124, "common_stride124")
    write_summary(common, strides_1234, "common_stride1234")


if __name__ == "__main__":
    main()
