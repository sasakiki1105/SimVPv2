import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"C:\Users\astro\research\SimVPv2")
WORKDIRS = ROOT / "workdirs"

CASES = [
    {
        "label": "stride1 tp0, 12.5 ns",
        "path": WORKDIRS
        / "pepapic_simvp_gsta_highmag_macro5_trainfixed_disjoint_811_bs2_100ep"
        / "rollout_tp0_quads_assets",
        "dt_ns": 12.5,
        "color": "#111111",
        "linestyle": "-",
        "linewidth": 2.0,
    },
    {
        "label": "stride1 tout10, 12.5 ns",
        "path": WORKDIRS
        / "pepapic_simvp_gsta_highmag_macro5_trainfixed_disjoint_811_bs2_100ep"
        / "rollout_tout10_quads_assets",
        "dt_ns": 12.5,
        "color": "#111111",
        "linestyle": "--",
        "linewidth": 2.0,
    },
    {
        "label": "stride2 tp0, 25 ns",
        "path": WORKDIRS
        / "pepapic_simvp_gsta_highmag_macro5_subsample2_trainfixed_disjoint_811_bs2_100ep"
        / "rollout_tp0_quads_assets",
        "dt_ns": 25.0,
        "color": "#1f77b4",
        "linestyle": "-",
        "linewidth": 1.8,
    },
    {
        "label": "stride2 tout10, 25 ns",
        "path": WORKDIRS
        / "pepapic_simvp_gsta_highmag_macro5_subsample2_trainfixed_disjoint_811_bs2_100ep"
        / "rollout_tout10_quads_assets",
        "dt_ns": 25.0,
        "color": "#1f77b4",
        "linestyle": "--",
        "linewidth": 1.8,
    },
    {
        "label": "stride4 tp0, 50 ns",
        "path": WORKDIRS
        / "pepapic_simvp_gsta_highmag_macro5_subsample4_trainfixed_disjoint_811_bs2_100ep"
        / "rollout_tp0_quads_assets",
        "dt_ns": 50.0,
        "color": "#2ca02c",
        "linestyle": "-",
        "linewidth": 1.8,
    },
    {
        "label": "stride4 tout10, 50 ns",
        "path": WORKDIRS
        / "pepapic_simvp_gsta_highmag_macro5_subsample4_trainfixed_disjoint_811_bs2_100ep"
        / "rollout_tout10_quads_assets",
        "dt_ns": 50.0,
        "color": "#2ca02c",
        "linestyle": "--",
        "linewidth": 1.8,
    },
]

OUTDIR = WORKDIRS / "compare_rollout_mse_stride_overlay"
PHI_CHANNEL = 2


def load_phi_mse(case):
    preds = np.load(case["path"] / "preds_roll.npy", mmap_mode="r")
    trues = np.load(case["path"] / "trues_roll.npy", mmap_mode="r")
    if preds.shape != trues.shape:
        raise ValueError(f"shape mismatch for {case['label']}: {preds.shape} vs {trues.shape}")
    if preds.ndim != 5 or preds.shape[1] != 1:
        raise ValueError(f"expected (K,1,C,H,W) for {case['label']}, got {preds.shape}")

    diff = preds[:, 0, PHI_CHANNEL].astype(np.float64) - trues[:, 0, PHI_CHANNEL].astype(np.float64)
    mse = np.mean(diff * diff, axis=(1, 2))
    time_us = np.arange(len(mse), dtype=np.float64) * case["dt_ns"] / 1000.0
    return time_us, mse


def summarize(time_us, mse):
    return {
        "n_frames": int(len(mse)),
        "time_us_first": float(time_us[0]),
        "time_us_last": float(time_us[-1]),
        "mse_first": float(mse[0]),
        "mse_last": float(mse[-1]),
        "mse_mean": float(np.mean(mse)),
        "mse_median": float(np.median(mse)),
        "mse_max": float(np.max(mse)),
    }


def save_overlay_png(results, out_png, yscale):
    plt.figure(figsize=(10, 6))
    for case, time_us, mse in results:
        plt.plot(
            time_us,
            mse,
            label=case["label"],
            color=case["color"],
            linestyle=case["linestyle"],
            linewidth=case["linewidth"],
            alpha=0.95,
        )

    plt.xlabel("Physical time since rollout start (us)")
    plt.ylabel("MSE (phi, normalized)")
    plt.title("Rollout phi MSE comparison")
    plt.xlim(left=0.0)
    plt.yscale(yscale)
    plt.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    plt.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[PLOT] {out_png}")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    results = []
    summary = {}
    for case in CASES:
        time_us, mse = load_phi_mse(case)
        results.append((case, time_us, mse))
        summary[case["label"]] = summarize(time_us, mse)

    save_overlay_png(results, OUTDIR / "mse_phi_rollout_overlay_log.png", "log")
    save_overlay_png(results, OUTDIR / "mse_phi_rollout_overlay_linear.png", "linear")

    max_len = max(len(mse) for _, _, mse in results)
    csv_path = OUTDIR / "mse_phi_rollout_overlay.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = []
        for case, _, _ in results:
            header.extend([f"{case['label']} time_us", f"{case['label']} mse_phi"])
        writer.writerow(header)
        for i in range(max_len):
            row = []
            for _, time_us, mse in results:
                if i < len(mse):
                    row.extend([float(time_us[i]), float(mse[i])])
                else:
                    row.extend(["", ""])
            writer.writerow(row)
    print(f"[CSV] {csv_path}")

    summary_path = OUTDIR / "mse_phi_rollout_overlay_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[JSON] {summary_path}")

    print("[SUMMARY]")
    for label, values in summary.items():
        print(
            f"{label}: mean={values['mse_mean']:.6g}, "
            f"last={values['mse_last']:.6g}, max={values['mse_max']:.6g}"
        )


if __name__ == "__main__":
    main()
