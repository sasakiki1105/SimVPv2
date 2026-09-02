import json
import csv

import numpy as np
import plot_forecast_horizon_baseline as base


WORKDIRS = base.WORKDIRS
H5_ROOT = base.H5_ROOT
OUTDIR = WORKDIRS / "compare_direct_horizon_baseline"


CASES = [
    {
        "name": "stride2_direct10",
        "label": "stride2 Direct10",
        "stride": 2,
        "tout": 10,
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_subsample2_trainfixed_disjoint_811_bs2_100ep",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step2.h5",
        "color": "#1f77b4",
        "marker": "s",
    },
    {
        "name": "stride2_direct20",
        "label": "stride2 Direct20",
        "stride": 2,
        "tout": 20,
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_subsample2_direct20_trainfixed_disjoint_811_bs2_100ep",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step2.h5",
        "color": "#6baed6",
        "marker": "s",
    },
    {
        "name": "stride2_direct40",
        "label": "stride2 Direct40",
        "stride": 2,
        "tout": 40,
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_subsample2_direct40_trainfixed_disjoint_811_bs2_100ep",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step2.h5",
        "color": "#3182bd",
        "marker": "s",
    },
    {
        "name": "stride2_direct80",
        "label": "stride2 Direct80",
        "stride": 2,
        "tout": 80,
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_subsample2_direct80_trainfixed_disjoint_811_bs2_100ep",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step2.h5",
        "color": "#08519c",
        "marker": "s",
    },
    {
        "name": "stride2_direct160",
        "label": "stride2 Direct160",
        "stride": 2,
        "tout": 160,
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_subsample2_direct160_trainfixed_disjoint_811_bs2_100ep",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step2.h5",
        "color": "#08306b",
        "marker": "s",
    },
    {
        "name": "stride2_direct180",
        "label": "stride2 Direct180",
        "stride": 2,
        "tout": 180,
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_subsample2_direct180_trainfixed_disjoint_811_bs2_100ep",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step2.h5",
        "color": "#4a1486",
        "marker": "s",
    },
]


def write_rows(rows):
    path = OUTDIR / "direct_horizon_model_vs_copy.csv"
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] {path}")


def plot_channel_scaled(rows, channel, cases, yscale, suffix):
    plt = base.plt
    plt.figure(figsize=(10, 6))
    for case in cases:
        r = [row for row in rows if row["case"] == case["name"] and row["channel"] == channel]
        if not r:
            continue
        x = np.asarray([row["horizon_ns"] for row in r])
        y = np.asarray([row["model_mse_mean"] for row in r])
        y_copy = np.asarray([row["copy_mse_mean"] for row in r])
        order = np.argsort(x)
        plt.plot(
            x[order],
            y[order],
            color=case["color"],
            marker=case["marker"],
            linewidth=1.8,
            label=f"{case['label']} model",
        )
        plt.plot(
            x[order],
            y_copy[order],
            color=case["color"],
            linestyle="--",
            linewidth=1.2,
            alpha=0.7,
            label=f"{case['label']} copy",
        )
    plt.xlabel("Prediction horizon from last input frame (ns)")
    plt.ylabel(f"MSE ({channel}, normalized)")
    scale_label = "log scale" if yscale == "log" else "linear scale"
    plt.title(f"True direct horizon MSE vs copy baseline: {channel} ({scale_label})")
    plt.yscale(yscale)
    grid_which = "both" if yscale == "log" else "major"
    plt.grid(True, which=grid_which, linestyle=":", linewidth=0.7, alpha=0.6)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out_png = OUTDIR / f"direct_horizon_model_vs_copy_{channel}{suffix}.png"
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[PLOT] {out_png}")


def plot_channel(rows, channel, cases):
    plot_channel_scaled(rows, channel, cases, "log", "")
    plot_channel_scaled(rows, channel, cases, "linear", "_linear")
    plot_model_only_scaled(rows, channel, cases, "linear", "_linear")


def plot_model_only_scaled(rows, channel, cases, yscale, suffix):
    plt = base.plt
    plt.figure(figsize=(10, 5.8))
    for case in cases:
        r = [row for row in rows if row["case"] == case["name"] and row["channel"] == channel]
        if not r:
            continue
        x = np.asarray([row["horizon_ns"] for row in r])
        y = np.asarray([row["model_mse_mean"] for row in r])
        order = np.argsort(x)
        plt.plot(
            x[order],
            y[order],
            color=case["color"],
            marker=case["marker"],
            linewidth=1.8,
            label=case["label"],
        )
    plt.xlabel("Prediction horizon from last input frame (ns)")
    plt.ylabel(f"Model MSE ({channel}, normalized)")
    scale_label = "log scale" if yscale == "log" else "linear scale"
    plt.title(f"True direct horizon model MSE: {channel} ({scale_label})")
    plt.yscale(yscale)
    grid_which = "both" if yscale == "log" else "major"
    plt.grid(True, which=grid_which, linestyle=":", linewidth=0.7, alpha=0.6)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out_png = OUTDIR / f"direct_horizon_model_mse_{channel}{suffix}.png"
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[PLOT] {out_png}")


def plot_improvement_scaled(rows, channel, cases, yscale, suffix):
    plt = base.plt
    plt.figure(figsize=(10, 5.8))
    for case in cases:
        r = [row for row in rows if row["case"] == case["name"] and row["channel"] == channel]
        if not r:
            continue
        x = np.asarray([row["horizon_ns"] for row in r])
        y = np.asarray([row["model_over_copy_mean"] for row in r])
        order = np.argsort(x)
        plt.plot(
            x[order],
            y[order],
            color=case["color"],
            marker=case["marker"],
            linewidth=1.8,
            label=case["label"],
        )
    plt.axhline(1.0, color="#555555", linestyle="--", linewidth=1.0, label="copy parity")
    plt.xlabel("Prediction horizon from last input frame (ns)")
    plt.ylabel("Model MSE / copy MSE")
    scale_label = "log scale" if yscale == "log" else "linear scale"
    plt.title(f"True direct horizon improvement over copy baseline: {channel} ({scale_label})")
    plt.yscale(yscale)
    grid_which = "both" if yscale == "log" else "major"
    plt.grid(True, which=grid_which, linestyle=":", linewidth=0.7, alpha=0.6)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out_png = OUTDIR / f"direct_horizon_model_over_copy_{channel}{suffix}.png"
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[PLOT] {out_png}")


def plot_improvement(rows, channel, cases):
    plot_improvement_scaled(rows, channel, cases, "log", "")
    plot_improvement_scaled(rows, channel, cases, "linear", "_linear")


def write_summary(loaded_cases, skipped, rows):
    summary = {
        "description": (
            "True direct long-horizon evaluation. Cases with Direct20/40/80 use "
            "simvp_direct_aft_seq=True, so aft_seq_length frames are produced in one "
            "model forward pass rather than by internal rollout."
        ),
        "base_dt_ns": base.BASE_DT_NS,
        "pre_seq_length": base.PRE,
        "output_dir": str(OUTDIR),
        "loaded_cases": [
            {
                "name": case["name"],
                "label": case["label"],
                "stride": case["stride"],
                "tout": case["tout"],
                "dt_ns": case["dt_ns"],
                "h5": str(case["h5"]),
                "saved": str(case["saved"]),
                "preds_shape": list(case["preds"].shape),
            }
            for case in loaded_cases
        ],
        "skipped_cases": skipped,
        "channels": sorted({row["channel"] for row in rows}),
    }
    path = OUTDIR / "direct_horizon_model_vs_copy_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[JSON] {path}")


def main():
    base.OUTDIR = OUTDIR
    base.CASES = CASES
    base.write_rows = write_rows
    base.plot_channel = plot_channel
    base.plot_improvement = plot_improvement
    base.write_summary = write_summary
    base.main()


if __name__ == "__main__":
    main()
