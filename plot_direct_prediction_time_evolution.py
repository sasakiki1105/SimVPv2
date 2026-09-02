import csv
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from matplotlib.colors import LogNorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"C:\Users\astro\research\SimVPv2")
WORKDIRS = ROOT / "workdirs"
H5_ROOT = Path(
    r"C:\Users\astro\research\PEPAPIC\test\results"
    r"\2D_ExB_high_magnet_dt2.5e-9_maxt50e-6_macro5"
    r"\SimVPv2_inputs"
)

BASE_DT_NS = 12.5
PRE = 10
AFT = 10
PHI_CHANNEL = 2
OUTDIR = WORKDIRS / "compare_direct_prediction_time_evolution"


CASES = [
    {
        "name": "stride1",
        "label": "stride1, 12.5 ns",
        "saved": WORKDIRS
        / "pepapic_simvp_gsta_highmag_macro5_trainfixed_disjoint_811_bs2_100ep"
        / "saved",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000.h5",
    },
    {
        "name": "stride2",
        "label": "stride2, 25 ns",
        "saved": WORKDIRS
        / "pepapic_simvp_gsta_highmag_macro5_subsample2_trainfixed_disjoint_811_bs2_100ep"
        / "saved",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step2.h5",
    },
    {
        "name": "stride3",
        "label": "stride3, 37.5 ns",
        "saved": WORKDIRS
        / "pepapic_simvp_gsta_highmag_macro5_subsample3_trainfixed_disjoint_811_bs2_100ep"
        / "saved",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step3.h5",
    },
    {
        "name": "stride4",
        "label": "stride4, 50 ns",
        "saved": WORKDIRS
        / "pepapic_simvp_gsta_highmag_macro5_subsample4_trainfixed_disjoint_811_bs2_100ep"
        / "saved",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step4.h5",
    },
]


def load_timesteps(h5_path):
    with h5py.File(h5_path, "r") as f:
        if "timesteps" in f:
            return np.asarray(f["timesteps"][()], dtype=np.int64)
        key = "data_tchw" if "data_tchw" in f else "data"
        shape = f[key].shape
        return np.arange(max(shape), dtype=np.int64)


def build_test_starts(t_len):
    total = PRE + AFT
    val_end = int(np.floor(t_len * 0.9))
    last_start = t_len - total
    if last_start < val_end:
        return np.empty((0,), dtype=np.int64)
    return np.arange(val_end, last_start + 1, dtype=np.int64)


def load_case(case):
    preds = np.load(case["saved"] / "preds.npy", mmap_mode="r")
    trues = np.load(case["saved"] / "trues.npy", mmap_mode="r")
    if preds.shape != trues.shape:
        raise ValueError(f"{case['name']} shape mismatch: {preds.shape} vs {trues.shape}")
    if preds.ndim != 5 or preds.shape[1] != AFT:
        raise ValueError(f"{case['name']} expected (N,{AFT},C,H,W), got {preds.shape}")

    timesteps = load_timesteps(case["h5"])
    starts = build_test_starts(len(timesteps))
    if len(starts) != preds.shape[0]:
        raise ValueError(
            f"{case['name']} saved sample count {preds.shape[0]} does not match "
            f"computed test starts {len(starts)}"
        )

    mse = np.empty((preds.shape[0], AFT), dtype=np.float64)
    for tp in range(AFT):
        diff = (
            preds[:, tp, PHI_CHANNEL].astype(np.float64)
            - trues[:, tp, PHI_CHANNEL].astype(np.float64)
        )
        mse[:, tp] = np.mean(diff * diff, axis=(1, 2))

    target_indices = starts[:, None] + PRE + np.arange(AFT, dtype=np.int64)[None, :]
    target_timesteps = timesteps[target_indices]
    target_time_us = target_timesteps.astype(np.float64) * BASE_DT_NS / 1000.0

    last_input_indices = starts + PRE - 1
    last_input_time_us = timesteps[last_input_indices].astype(np.float64) * BASE_DT_NS / 1000.0

    if len(timesteps) > 1:
        stride_steps = int(round(float(np.median(np.diff(timesteps)))))
    else:
        stride_steps = 1

    return {
        **case,
        "preds_shape": preds.shape,
        "timesteps": timesteps,
        "starts": starts,
        "mse": mse,
        "target_time_us": target_time_us,
        "last_input_time_us": last_input_time_us,
        "dt_ns": stride_steps * BASE_DT_NS,
    }


def save_case_lines(case):
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, AFT))
    plt.figure(figsize=(10, 6))
    for tp in range(AFT):
        plt.plot(
            case["target_time_us"][:, tp],
            case["mse"][:, tp],
            color=colors[tp],
            linewidth=1.2,
            label=f"out {tp + 1}",
        )
    plt.xlabel("Absolute target time in simulation (us)")
    plt.ylabel("Direct prediction MSE (phi, normalized)")
    plt.title(f"{case['label']}: direct prediction error for each output frame")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    out_png = OUTDIR / f"direct_mse_time_lines_{case['name']}.png"
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[PLOT] {out_png}")


def save_case_heatmap(case):
    z = case["mse"].T
    positive = z[z > 0]
    vmin = float(np.min(positive))
    vmax = float(np.max(positive))

    plt.figure(figsize=(10, 4.8))
    extent = [
        float(case["last_input_time_us"][0]),
        float(case["last_input_time_us"][-1]),
        0.5,
        AFT + 0.5,
    ]
    plt.imshow(
        z,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="magma",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        interpolation="nearest",
    )
    plt.colorbar(label="Direct prediction MSE (phi, normalized)")
    plt.xlabel("Last input frame time (us)")
    plt.ylabel("Output index")
    plt.yticks(np.arange(1, AFT + 1))
    plt.title(f"{case['label']}: direct prediction MSE heatmap")
    plt.tight_layout()
    out_png = OUTDIR / f"direct_mse_heatmap_{case['name']}.png"
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[PLOT] {out_png}")


def save_all_stride_lines(cases):
    fig, axes = plt.subplots(len(cases), 1, figsize=(10, 2.8 * len(cases) + 1.6), sharex=False, sharey=True)
    axes = np.atleast_1d(axes)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, AFT))
    for ax, case in zip(axes, cases):
        for tp in range(AFT):
            ax.plot(
                case["target_time_us"][:, tp],
                case["mse"][:, tp],
                color=colors[tp],
                linewidth=1.0,
                label=f"out {tp + 1}" if ax is axes[0] else None,
            )
        ax.set_title(case["label"])
        ax.set_ylabel("MSE")
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    axes[-1].set_xlabel("Absolute target time in simulation (us)")
    fig.suptitle("Direct prediction phi MSE over test simulation time")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=8)
    fig.tight_layout(rect=[0, 0, 0.9, 0.97])
    out_png = OUTDIR / "direct_mse_time_lines_all_strides.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[PLOT] {out_png}")


def save_long_csv(cases):
    csv_path = OUTDIR / "direct_mse_time_evolution.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "case",
            "label",
            "sample_index",
            "output_index",
            "target_time_us",
            "last_input_time_us",
            "horizon_ns",
            "mse_phi",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            for i in range(case["mse"].shape[0]):
                last_input_time = float(case["last_input_time_us"][i])
                for tp in range(AFT):
                    target_time = float(case["target_time_us"][i, tp])
                    writer.writerow({
                        "case": case["name"],
                        "label": case["label"],
                        "sample_index": i,
                        "output_index": tp + 1,
                        "target_time_us": target_time,
                        "last_input_time_us": last_input_time,
                        "horizon_ns": (target_time - last_input_time) * 1000.0,
                        "mse_phi": float(case["mse"][i, tp]),
                    })
    print(f"[CSV] {csv_path}")


def save_summary(cases):
    summary = {
        "description": (
            "Teacher-forced direct prediction over every test window. "
            "Inputs are always ground truth; predictions are not fed back."
        ),
        "base_dt_ns": BASE_DT_NS,
        "pre_seq_length": PRE,
        "aft_seq_length": AFT,
        "phi_channel": PHI_CHANNEL,
        "cases": [],
    }
    for case in cases:
        per_output = []
        for tp in range(AFT):
            y = case["mse"][:, tp]
            per_output.append({
                "output_index": tp + 1,
                "horizon_ns_from_last_input": float((tp + 1) * case["dt_ns"]),
                "mse_mean": float(np.mean(y)),
                "mse_median": float(np.median(y)),
                "mse_max": float(np.max(y)),
            })
        summary["cases"].append({
            "name": case["name"],
            "label": case["label"],
            "h5": str(case["h5"]),
            "saved": str(case["saved"]),
            "dt_ns": case["dt_ns"],
            "preds_shape": list(case["preds_shape"]),
            "target_time_us_first": float(np.min(case["target_time_us"])),
            "target_time_us_last": float(np.max(case["target_time_us"])),
            "per_output": per_output,
        })

    summary_path = OUTDIR / "direct_mse_time_evolution_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[JSON] {summary_path}")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cases = [load_case(case) for case in CASES]

    for case in cases:
        save_case_lines(case)
        save_case_heatmap(case)
    save_all_stride_lines(cases)
    save_long_csv(cases)
    save_summary(cases)

    print("[SUMMARY]")
    for case in cases:
        print(
            f"{case['name']}: target time "
            f"{np.min(case['target_time_us']):.3f}-{np.max(case['target_time_us']):.3f} us, "
            f"mean MSE={np.mean(case['mse']):.6g}"
        )


if __name__ == "__main__":
    main()
