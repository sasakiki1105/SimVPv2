import csv
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

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
OUTDIR = WORKDIRS / "compare_direct_prediction_physical_horizon"


CASES = [
    {
        "name": "stride1",
        "label": "stride1, 12.5 ns",
        "saved": WORKDIRS
        / "pepapic_simvp_gsta_highmag_macro5_trainfixed_disjoint_811_bs2_100ep"
        / "saved",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000.h5",
        "color": "#111111",
        "marker": "o",
    },
    {
        "name": "stride2",
        "label": "stride2, 25 ns",
        "saved": WORKDIRS
        / "pepapic_simvp_gsta_highmag_macro5_subsample2_trainfixed_disjoint_811_bs2_100ep"
        / "saved",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step2.h5",
        "color": "#1f77b4",
        "marker": "s",
    },
    {
        "name": "stride3",
        "label": "stride3, 37.5 ns",
        "saved": WORKDIRS
        / "pepapic_simvp_gsta_highmag_macro5_subsample3_trainfixed_disjoint_811_bs2_100ep"
        / "saved",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step3.h5",
        "color": "#ff7f0e",
        "marker": "D",
    },
    {
        "name": "stride4",
        "label": "stride4, 50 ns",
        "saved": WORKDIRS
        / "pepapic_simvp_gsta_highmag_macro5_subsample4_trainfixed_disjoint_811_bs2_100ep"
        / "saved",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step4.h5",
        "color": "#2ca02c",
        "marker": "^",
    },
]


def load_h5_time_info(h5_path):
    with h5py.File(h5_path, "r") as f:
        if "timesteps" in f:
            timesteps = np.asarray(f["timesteps"][()], dtype=np.int64)
        else:
            key = "data_tchw" if "data_tchw" in f else "data"
            shape = f[key].shape
            t_len = max(shape)
            timesteps = np.arange(t_len, dtype=np.int64)

    if len(timesteps) < 2:
        stride_steps = 1
    else:
        diffs = np.diff(timesteps)
        stride_steps = int(round(float(np.median(diffs))))
    return timesteps, stride_steps


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
        raise ValueError(f"{case['name']} preds/trues shape mismatch: {preds.shape} vs {trues.shape}")
    if preds.ndim != 5 or preds.shape[1] != AFT:
        raise ValueError(f"{case['name']} expected (N,{AFT},C,H,W), got {preds.shape}")

    timesteps, stride_steps = load_h5_time_info(case["h5"])
    starts = build_test_starts(len(timesteps))
    if len(starts) != preds.shape[0]:
        raise ValueError(
            f"{case['name']} saved sample count {preds.shape[0]} does not match "
            f"computed test starts {len(starts)}"
        )

    return {
        **case,
        "preds": preds,
        "trues": trues,
        "timesteps": timesteps,
        "stride_steps": stride_steps,
        "dt_ns": stride_steps * BASE_DT_NS,
        "starts": starts,
    }


def mse_values_for_tp(loaded_case, tp):
    preds = loaded_case["preds"][:, tp, PHI_CHANNEL].astype(np.float64)
    trues = loaded_case["trues"][:, tp, PHI_CHANNEL].astype(np.float64)
    diff = preds - trues
    return np.mean(diff * diff, axis=(1, 2))


def target_timesteps_for_tp(loaded_case, tp):
    idx = loaded_case["starts"] + PRE + tp
    return loaded_case["timesteps"][idx]


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "first": float(values[0]),
        "last": float(values[-1]),
        "max": float(np.max(values)),
    }


def write_all_windows_summary(loaded_cases):
    rows = []
    for case in loaded_cases:
        for tp in range(AFT):
            mse = mse_values_for_tp(case, tp)
            horizon_ns = (tp + 1) * case["dt_ns"]
            s = summarize(mse)
            rows.append({
                "case": case["name"],
                "label": case["label"],
                "tp_index_zero_based": tp,
                "output_index_one_based": tp + 1,
                "horizon_ns": horizon_ns,
                "n_samples": s["n"],
                "mse_mean": s["mean"],
                "mse_median": s["median"],
                "mse_std": s["std"],
                "mse_first": s["first"],
                "mse_last": s["last"],
                "mse_max": s["max"],
            })

    csv_path = OUTDIR / "direct_mse_all_windows_by_horizon.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] {csv_path}")
    return rows


def build_target_mse_maps(loaded_cases):
    maps = {}
    for case in loaded_cases:
        maps[case["name"]] = {}
        for tp in range(AFT):
            horizon_ns = (tp + 1) * case["dt_ns"]
            targets = target_timesteps_for_tp(case, tp)
            values = mse_values_for_tp(case, tp)
            maps[case["name"]][horizon_ns] = {
                int(t): float(v) for t, v in zip(targets, values)
            }
    return maps


def write_aligned_summary(loaded_cases, target_maps):
    horizons = sorted({h for m in target_maps.values() for h in m.keys()})
    rows = []

    for horizon_ns in horizons:
        supporting = [case for case in loaded_cases if horizon_ns in target_maps[case["name"]]]
        if len(supporting) < 2:
            continue

        common_targets = None
        for case in supporting:
            keys = set(target_maps[case["name"]][horizon_ns].keys())
            common_targets = keys if common_targets is None else common_targets & keys
        common_targets = sorted(common_targets)
        if not common_targets:
            continue

        model_names = "+".join(case["name"] for case in supporting)
        for case in supporting:
            values = [target_maps[case["name"]][horizon_ns][t] for t in common_targets]
            s = summarize(values)
            rows.append({
                "horizon_ns": horizon_ns,
                "aligned_models": model_names,
                "case": case["name"],
                "label": case["label"],
                "n_common_targets": s["n"],
                "target_timestep_first": int(common_targets[0]),
                "target_timestep_last": int(common_targets[-1]),
                "mse_mean": s["mean"],
                "mse_median": s["median"],
                "mse_std": s["std"],
                "mse_first": s["first"],
                "mse_last": s["last"],
                "mse_max": s["max"],
            })

    csv_path = OUTDIR / "direct_mse_aligned_common_targets.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] {csv_path}")
    return rows


def plot_all_windows(rows, loaded_cases):
    plt.figure(figsize=(9, 5.5))
    for case in loaded_cases:
        r = [row for row in rows if row["case"] == case["name"]]
        x = [row["horizon_ns"] for row in r]
        y = [row["mse_mean"] for row in r]
        plt.plot(
            x,
            y,
            marker=case["marker"],
            color=case["color"],
            linewidth=1.8,
            label=case["label"],
        )
    plt.xlabel("Prediction horizon from last input frame (ns)")
    plt.ylabel("Direct prediction MSE (phi, normalized)")
    plt.title("Direct prediction error by physical horizon")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    out_png = OUTDIR / "direct_mse_by_horizon_all_windows.png"
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[PLOT] {out_png}")


def plot_aligned(rows, loaded_cases):
    plt.figure(figsize=(9, 5.5))
    for case in loaded_cases:
        r = [row for row in rows if row["case"] == case["name"]]
        if not r:
            continue
        x = [row["horizon_ns"] for row in r]
        y = [row["mse_mean"] for row in r]
        plt.plot(
            x,
            y,
            marker=case["marker"],
            color=case["color"],
            linewidth=1.8,
            label=case["label"],
        )
    plt.xlabel("Prediction horizon from last input frame (ns)")
    plt.ylabel("Direct prediction MSE (phi, normalized)")
    plt.title("Direct prediction error on common absolute target times for shared horizons")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    out_png = OUTDIR / "direct_mse_aligned_shared_horizon_common_targets.png"
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[PLOT] {out_png}")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    loaded_cases = [load_case(case) for case in CASES]
    all_rows = write_all_windows_summary(loaded_cases)
    target_maps = build_target_mse_maps(loaded_cases)
    aligned_rows = write_aligned_summary(loaded_cases, target_maps)

    plot_all_windows(all_rows, loaded_cases)
    plot_aligned(aligned_rows, loaded_cases)

    summary = {
        "base_dt_ns": BASE_DT_NS,
        "pre_seq_length": PRE,
        "aft_seq_length": AFT,
        "phi_channel": PHI_CHANNEL,
        "cases": [
            {
                "name": case["name"],
                "label": case["label"],
                "h5": str(case["h5"]),
                "saved": str(case["saved"]),
                "dt_ns": case["dt_ns"],
                "n_test_samples": int(case["preds"].shape[0]),
            }
            for case in loaded_cases
        ],
        "all_windows_rows": all_rows,
        "aligned_rows": aligned_rows,
    }
    summary_path = OUTDIR / "direct_prediction_physical_horizon_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[JSON] {summary_path}")

    print("[SUMMARY] all-window mean MSE at common 50 ns and 100 ns horizons")
    for horizon_ns in [50.0, 100.0]:
        parts = []
        for row in all_rows:
            if row["horizon_ns"] == horizon_ns:
                parts.append(f"{row['case']}={row['mse_mean']:.6g}")
        print(f"  {horizon_ns:g} ns: " + ", ".join(parts))


if __name__ == "__main__":
    main()
