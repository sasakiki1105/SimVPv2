import csv
import json
from collections import defaultdict
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
OUTDIR = WORKDIRS / "compare_direct_prediction_target_time_aggregate"


CASES = [
    {
        "name": "stride1",
        "label": "stride1, 12.5 ns",
        "saved": WORKDIRS
        / "pepapic_simvp_gsta_highmag_macro5_trainfixed_disjoint_811_bs2_100ep"
        / "saved",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000.h5",
        "color": "#111111",
    },
    {
        "name": "stride2",
        "label": "stride2, 25 ns",
        "saved": WORKDIRS
        / "pepapic_simvp_gsta_highmag_macro5_subsample2_trainfixed_disjoint_811_bs2_100ep"
        / "saved",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step2.h5",
        "color": "#1f77b4",
    },
    {
        "name": "stride3",
        "label": "stride3, 37.5 ns",
        "saved": WORKDIRS
        / "pepapic_simvp_gsta_highmag_macro5_subsample3_trainfixed_disjoint_811_bs2_100ep"
        / "saved",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step3.h5",
        "color": "#ff7f0e",
    },
    {
        "name": "stride4",
        "label": "stride4, 50 ns",
        "saved": WORKDIRS
        / "pepapic_simvp_gsta_highmag_macro5_subsample4_trainfixed_disjoint_811_bs2_100ep"
        / "saved",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step4.h5",
        "color": "#2ca02c",
    },
]


def load_timesteps(h5_path):
    with h5py.File(h5_path, "r") as f:
        if "timesteps" in f:
            return np.asarray(f["timesteps"][()], dtype=np.int64)
        key = "data_tchw" if "data_tchw" in f else "data"
        return np.arange(max(f[key].shape), dtype=np.int64)


def build_test_starts(t_len):
    total = PRE + AFT
    val_end = int(np.floor(t_len * 0.9))
    last_start = t_len - total
    if last_start < val_end:
        return np.empty((0,), dtype=np.int64)
    return np.arange(val_end, last_start + 1, dtype=np.int64)


def summarize_values(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def load_case_aggregates(case):
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

    values_by_timestep = defaultdict(list)
    raw_rows = []
    for tp in range(AFT):
        diff = (
            preds[:, tp, PHI_CHANNEL].astype(np.float64)
            - trues[:, tp, PHI_CHANNEL].astype(np.float64)
        )
        mse = np.mean(diff * diff, axis=(1, 2))
        target_indices = starts + PRE + tp
        target_timesteps = timesteps[target_indices]
        horizon_ns = (tp + 1) * case["dt_ns"]

        for sample_idx, target_timestep, value in zip(
            range(len(starts)), target_timesteps, mse
        ):
            target_timestep = int(target_timestep)
            value = float(value)
            values_by_timestep[target_timestep].append(value)
            raw_rows.append({
                "case": case["name"],
                "sample_index": int(sample_idx),
                "output_index": int(tp + 1),
                "horizon_ns": float(horizon_ns),
                "target_timestep": target_timestep,
                "target_time_us": target_timestep * BASE_DT_NS / 1000.0,
                "mse_phi": value,
            })

    aggregate_rows = []
    for target_timestep in sorted(values_by_timestep):
        summary = summarize_values(values_by_timestep[target_timestep])
        aggregate_rows.append({
            "case": case["name"],
            "label": case["label"],
            "target_timestep": int(target_timestep),
            "target_time_us": target_timestep * BASE_DT_NS / 1000.0,
            **summary,
        })

    return {
        **case,
        "preds_shape": list(preds.shape),
        "timesteps": timesteps,
        "starts": starts,
        "raw_rows": raw_rows,
        "aggregate_rows": aggregate_rows,
        "aggregate_by_timestep": {
            row["target_timestep"]: row for row in aggregate_rows
        },
    }


def add_dt_to_cases(cases):
    out = []
    for case in cases:
        timesteps = load_timesteps(case["h5"])
        if len(timesteps) > 1:
            stride_steps = int(round(float(np.median(np.diff(timesteps)))))
        else:
            stride_steps = 1
        out.append({**case, "dt_ns": stride_steps * BASE_DT_NS})
    return out


def write_aggregate_csv(cases):
    path = OUTDIR / "direct_mse_by_target_time_all_times.csv"
    fieldnames = [
        "case",
        "label",
        "target_timestep",
        "target_time_us",
        "count",
        "mean",
        "median",
        "q25",
        "q75",
        "min",
        "max",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            writer.writerows(case["aggregate_rows"])
    print(f"[CSV] {path}")


def write_raw_csv(cases):
    path = OUTDIR / "direct_mse_by_target_time_raw_predictions.csv"
    fieldnames = [
        "case",
        "sample_index",
        "output_index",
        "horizon_ns",
        "target_timestep",
        "target_time_us",
        "mse_phi",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            writer.writerows(case["raw_rows"])
    print(f"[CSV] {path}")


def common_timestep_rows(cases):
    common = None
    for case in cases:
        keys = set(case["aggregate_by_timestep"].keys())
        common = keys if common is None else common & keys
    common = sorted(common)

    rows = []
    for target_timestep in common:
        row = {
            "target_timestep": int(target_timestep),
            "target_time_us": target_timestep * BASE_DT_NS / 1000.0,
        }
        for case in cases:
            agg = case["aggregate_by_timestep"][target_timestep]
            prefix = case["name"]
            row[f"{prefix}_count"] = agg["count"]
            row[f"{prefix}_mean"] = agg["mean"]
            row[f"{prefix}_median"] = agg["median"]
            row[f"{prefix}_q25"] = agg["q25"]
            row[f"{prefix}_q75"] = agg["q75"]
        rows.append(row)
    return rows


def write_common_csv(rows):
    path = OUTDIR / "direct_mse_by_target_time_common_times.csv"
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] {path}")


def plot_metric(cases, metric, out_png, title, common_rows=None):
    plt.figure(figsize=(10, 5.8))

    if common_rows is None:
        for case in cases:
            x = np.asarray([row["target_time_us"] for row in case["aggregate_rows"]])
            y = np.asarray([row[metric] for row in case["aggregate_rows"]])
            q25 = np.asarray([row["q25"] for row in case["aggregate_rows"]])
            q75 = np.asarray([row["q75"] for row in case["aggregate_rows"]])
            plt.plot(
                x,
                y,
                color=case["color"],
                linewidth=1.7,
                label=case["label"],
            )
            if metric in ("median", "mean"):
                plt.fill_between(x, q25, q75, color=case["color"], alpha=0.14, linewidth=0)
    else:
        x = np.asarray([row["target_time_us"] for row in common_rows])
        for case in cases:
            prefix = case["name"]
            y = np.asarray([row[f"{prefix}_{metric}"] for row in common_rows])
            q25 = np.asarray([row[f"{prefix}_q25"] for row in common_rows])
            q75 = np.asarray([row[f"{prefix}_q75"] for row in common_rows])
            plt.plot(
                x,
                y,
                color=case["color"],
                linewidth=1.7,
                label=case["label"],
            )
            if metric in ("median", "mean"):
                plt.fill_between(x, q25, q75, color=case["color"], alpha=0.14, linewidth=0)

    plt.xlabel("Absolute target time in simulation (us)")
    plt.ylabel(f"Aggregated direct prediction {metric} MSE (phi, normalized)")
    plt.title(title)
    plt.yscale("log")
    plt.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[PLOT] {out_png}")


def plot_counts(cases, common_rows):
    plt.figure(figsize=(10, 4.8))
    for case in cases:
        x = np.asarray([row["target_time_us"] for row in case["aggregate_rows"]])
        y = np.asarray([row["count"] for row in case["aggregate_rows"]])
        plt.plot(x, y, color=case["color"], linewidth=1.7, label=case["label"])
    plt.xlabel("Absolute target time in simulation (us)")
    plt.ylabel("Number of direct predictions aggregated")
    plt.title("Number of predictions contributing to each target-time aggregate")
    plt.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    out_png = OUTDIR / "direct_mse_target_time_aggregate_counts_all_times.png"
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[PLOT] {out_png}")

    plt.figure(figsize=(10, 4.8))
    x = np.asarray([row["target_time_us"] for row in common_rows])
    for case in cases:
        y = np.asarray([row[f"{case['name']}_count"] for row in common_rows])
        plt.plot(x, y, color=case["color"], linewidth=1.7, label=case["label"])
    plt.xlabel("Absolute target time in simulation (us)")
    plt.ylabel("Number of direct predictions aggregated")
    plt.title("Aggregate counts on common target times")
    plt.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    out_png = OUTDIR / "direct_mse_target_time_aggregate_counts_common_times.png"
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[PLOT] {out_png}")


def save_summary(cases, common_rows):
    summary = {
        "description": (
            "Direct prediction errors grouped by absolute target simulation time. "
            "For each target time and stride, all output indices/windows that predict "
            "that same target frame are aggregated."
        ),
        "base_dt_ns": BASE_DT_NS,
        "pre_seq_length": PRE,
        "aft_seq_length": AFT,
        "phi_channel": PHI_CHANNEL,
        "n_common_target_times": len(common_rows),
        "common_target_time_us": [
            float(common_rows[0]["target_time_us"]),
            float(common_rows[-1]["target_time_us"]),
        ],
        "cases": [],
    }
    for case in cases:
        medians = np.asarray([row["median"] for row in case["aggregate_rows"]])
        means = np.asarray([row["mean"] for row in case["aggregate_rows"]])
        common_medians = np.asarray([
            row[f"{case['name']}_median"] for row in common_rows
        ])
        common_means = np.asarray([
            row[f"{case['name']}_mean"] for row in common_rows
        ])
        summary["cases"].append({
            "name": case["name"],
            "label": case["label"],
            "h5": str(case["h5"]),
            "saved": str(case["saved"]),
            "dt_ns": case["dt_ns"],
            "preds_shape": case["preds_shape"],
            "n_target_times_all": len(case["aggregate_rows"]),
            "target_time_us_all": [
                float(case["aggregate_rows"][0]["target_time_us"]),
                float(case["aggregate_rows"][-1]["target_time_us"]),
            ],
            "all_times_median_of_medians": float(np.median(medians)),
            "all_times_mean_of_medians": float(np.mean(medians)),
            "all_times_mean_of_means": float(np.mean(means)),
            "common_times_median_of_medians": float(np.median(common_medians)),
            "common_times_mean_of_medians": float(np.mean(common_medians)),
            "common_times_mean_of_means": float(np.mean(common_means)),
        })

    path = OUTDIR / "direct_mse_target_time_aggregate_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[JSON] {path}")

    print("[SUMMARY] common target times")
    for item in summary["cases"]:
        print(
            f"{item['name']}: median-of-medians={item['common_times_median_of_medians']:.6g}, "
            f"mean-of-medians={item['common_times_mean_of_medians']:.6g}"
        )


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    case_configs = add_dt_to_cases(CASES)
    cases = [load_case_aggregates(case) for case in case_configs]

    write_raw_csv(cases)
    write_aggregate_csv(cases)
    common_rows = common_timestep_rows(cases)
    write_common_csv(common_rows)

    plot_metric(
        cases,
        "median",
        OUTDIR / "direct_mse_target_time_median_all_times.png",
        "Target-time aggregated direct MSE, all available target times",
    )
    plot_metric(
        cases,
        "mean",
        OUTDIR / "direct_mse_target_time_mean_all_times.png",
        "Target-time aggregated direct MSE mean, all available target times",
    )
    plot_metric(
        cases,
        "median",
        OUTDIR / "direct_mse_target_time_median_common_times.png",
        "Target-time aggregated direct MSE on common target times",
        common_rows=common_rows,
    )
    plot_metric(
        cases,
        "mean",
        OUTDIR / "direct_mse_target_time_mean_common_times.png",
        "Target-time aggregated direct MSE mean on common target times",
        common_rows=common_rows,
    )
    plot_counts(cases, common_rows)
    save_summary(cases, common_rows)


if __name__ == "__main__":
    main()
