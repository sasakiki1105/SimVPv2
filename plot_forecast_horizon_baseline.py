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
OUTDIR = WORKDIRS / "compare_forecast_horizon_baseline"

CHANNEL_FALLBACK = ["electron_den", "ion_den", "phi"]


CASES = [
    {
        "name": "stride1_tout10",
        "label": "stride1 Tout10",
        "stride": 1,
        "tout": 10,
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_trainfixed_disjoint_811_bs2_100ep",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000.h5",
        "color": "#111111",
        "marker": "o",
    },
    {
        "name": "stride1_tout20",
        "label": "stride1 Tout20",
        "stride": 1,
        "tout": 20,
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_tout20_trainfixed_disjoint_811_bs2_100ep",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000.h5",
        "color": "#444444",
        "marker": "o",
    },
    {
        "name": "stride1_tout30",
        "label": "stride1 Tout30",
        "stride": 1,
        "tout": 30,
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_tout30_trainfixed_disjoint_811_bs2_100ep",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000.h5",
        "color": "#777777",
        "marker": "o",
    },
    {
        "name": "stride1_tout40",
        "label": "stride1 Tout40",
        "stride": 1,
        "tout": 40,
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_tout40_trainfixed_disjoint_811_bs2_100ep",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000.h5",
        "color": "#aaaaaa",
        "marker": "o",
    },
    {
        "name": "stride2_tout10",
        "label": "stride2 Tout10",
        "stride": 2,
        "tout": 10,
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_subsample2_trainfixed_disjoint_811_bs2_100ep",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step2.h5",
        "color": "#1f77b4",
        "marker": "s",
    },
    {
        "name": "stride2_tout20",
        "label": "stride2 Tout20",
        "stride": 2,
        "tout": 20,
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_subsample2_tout20_trainfixed_disjoint_811_bs2_100ep",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step2.h5",
        "color": "#6baed6",
        "marker": "s",
    },
    {
        "name": "stride2_tout40",
        "label": "stride2 Tout40",
        "stride": 2,
        "tout": 40,
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_subsample2_tout40_trainfixed_disjoint_811_bs2_100ep",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step2.h5",
        "color": "#3182bd",
        "marker": "s",
    },
    {
        "name": "stride2_tout80",
        "label": "stride2 Tout80",
        "stride": 2,
        "tout": 80,
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_subsample2_tout80_trainfixed_disjoint_811_bs2_100ep",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step2.h5",
        "color": "#08519c",
        "marker": "s",
    },
    {
        "name": "stride3_tout10",
        "label": "stride3 Tout10",
        "stride": 3,
        "tout": 10,
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_subsample3_trainfixed_disjoint_811_bs2_100ep",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step3.h5",
        "color": "#ff7f0e",
        "marker": "D",
    },
    {
        "name": "stride3_tout14",
        "label": "stride3 Tout14",
        "stride": 3,
        "tout": 14,
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_subsample3_tout14_trainfixed_disjoint_811_bs2_100ep",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step3.h5",
        "color": "#fdae6b",
        "marker": "D",
    },
    {
        "name": "stride4_tout10",
        "label": "stride4 Tout10",
        "stride": 4,
        "tout": 10,
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_subsample4_trainfixed_disjoint_811_bs2_100ep",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step4.h5",
        "color": "#2ca02c",
        "marker": "^",
    },
    {
        "name": "stride4_tout20",
        "label": "stride4 Tout20",
        "stride": 4,
        "tout": 20,
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_subsample4_tout20_trainfixed_disjoint_811_bs2_100ep",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step4.h5",
        "color": "#74c476",
        "marker": "^",
    },
    {
        "name": "stride4_tout40",
        "label": "stride4 Tout40",
        "stride": 4,
        "tout": 40,
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_subsample4_tout40_trainfixed_disjoint_811_bs2_100ep",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step4.h5",
        "color": "#006d2c",
        "marker": "^",
    },
]


def as_str_list(values):
    out = []
    for value in values:
        if isinstance(value, (bytes, bytearray)):
            out.append(value.decode())
        else:
            out.append(str(value))
    return out


def load_h5_info(h5_path):
    with h5py.File(h5_path, "r") as f:
        timesteps = np.asarray(f["timesteps"][()], dtype=np.int64)
        props = as_str_list(f["props"][()]) if "props" in f else CHANNEL_FALLBACK
    if len(timesteps) > 1:
        stride_steps = int(round(float(np.median(np.diff(timesteps)))))
    else:
        stride_steps = 1
    return timesteps, props, stride_steps


def build_test_starts(t_len, pre, aft):
    total = pre + aft
    val_end = int(np.floor(t_len * 0.9))
    last_start = t_len - total
    if last_start < val_end:
        return np.empty((0,), dtype=np.int64)
    return np.arange(val_end, last_start + 1, dtype=np.int64)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def load_case(case):
    saved = case["workdir"] / "saved"
    paths = {name: saved / f"{name}.npy" for name in ("inputs", "preds", "trues")}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        return None, f"missing saved arrays: {missing}"

    inputs = np.load(paths["inputs"], mmap_mode="r")
    preds = np.load(paths["preds"], mmap_mode="r")
    trues = np.load(paths["trues"], mmap_mode="r")
    if preds.shape != trues.shape:
        return None, f"preds/trues shape mismatch: {preds.shape} vs {trues.shape}"
    if inputs.ndim != 5 or preds.ndim != 5:
        return None, f"expected 5D arrays, got inputs={inputs.shape}, preds={preds.shape}"

    pre = int(inputs.shape[1])
    aft = int(preds.shape[1])
    if pre != PRE:
        return None, f"unexpected pre length {pre}, expected {PRE}"
    if aft != case["tout"]:
        return None, f"case tout={case['tout']} but preds have aft={aft}"

    timesteps, props, stride_steps = load_h5_info(case["h5"])
    starts = build_test_starts(len(timesteps), pre, aft)
    if len(starts) != preds.shape[0]:
        return None, (
            f"sample count mismatch for {case['name']}: "
            f"saved={preds.shape[0]}, computed={len(starts)}"
        )

    return {
        **case,
        "saved": saved,
        "inputs": inputs,
        "preds": preds,
        "trues": trues,
        "pre": pre,
        "aft": aft,
        "timesteps": timesteps,
        "props": props,
        "stride_steps": stride_steps,
        "dt_ns": stride_steps * BASE_DT_NS,
        "starts": starts,
    }, None


def case_rows(case):
    rows = []
    last_input_index = case["starts"] + case["pre"] - 1
    last_input_timestep = case["timesteps"][last_input_index]

    for channel_index, channel in enumerate(case["props"]):
        persistence = case["inputs"][:, -1, channel_index].astype(np.float64)
        for tp in range(case["aft"]):
            pred = case["preds"][:, tp, channel_index].astype(np.float64)
            true = case["trues"][:, tp, channel_index].astype(np.float64)
            model_mse = np.mean((pred - true) ** 2, axis=(1, 2))
            copy_mse = np.mean((persistence - true) ** 2, axis=(1, 2))
            target_index = case["starts"] + case["pre"] + tp
            target_timestep = case["timesteps"][target_index]
            horizon_ns = (target_timestep - last_input_timestep).astype(np.float64) * BASE_DT_NS
            model_summary = summarize(model_mse)
            copy_summary = summarize(copy_mse)
            ratio = model_summary["mean"] / copy_summary["mean"] if copy_summary["mean"] > 0 else np.nan
            rows.append({
                "case": case["name"],
                "label": case["label"],
                "stride": case["stride"],
                "tout": case["tout"],
                "dt_ns": case["dt_ns"],
                "channel": channel,
                "channel_index": channel_index,
                "output_index": tp + 1,
                "horizon_ns": float(np.median(horizon_ns)),
                "n_samples": int(len(model_mse)),
                "model_mse_mean": model_summary["mean"],
                "model_mse_median": model_summary["median"],
                "model_mse_q25": model_summary["q25"],
                "model_mse_q75": model_summary["q75"],
                "copy_mse_mean": copy_summary["mean"],
                "copy_mse_median": copy_summary["median"],
                "copy_mse_q25": copy_summary["q25"],
                "copy_mse_q75": copy_summary["q75"],
                "model_over_copy_mean": float(ratio),
                "improvement_fraction": float(1.0 - ratio) if np.isfinite(ratio) else np.nan,
            })
    return rows


def write_rows(rows):
    path = OUTDIR / "forecast_horizon_model_vs_copy.csv"
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] {path}")


def plot_channel(rows, channel, cases):
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
    plt.title(f"Forecast horizon MSE vs copy baseline: {channel}")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out_png = OUTDIR / f"forecast_horizon_model_vs_copy_{channel}.png"
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[PLOT] {out_png}")


def plot_improvement(rows, channel, cases):
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
    plt.title(f"Forecast horizon improvement over copy baseline: {channel}")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out_png = OUTDIR / f"forecast_horizon_model_over_copy_{channel}.png"
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[PLOT] {out_png}")


def write_summary(loaded_cases, skipped, rows):
    summary = {
        "description": (
            "Forecast-horizon internal-rollout evaluation. Solid model lines are "
            "compared against a persistence/copy baseline that repeats the last input frame."
        ),
        "base_dt_ns": BASE_DT_NS,
        "pre_seq_length": PRE,
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
    path = OUTDIR / "forecast_horizon_model_vs_copy_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[JSON] {path}")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    loaded_cases = []
    skipped = []
    rows = []

    for case in CASES:
        loaded, reason = load_case(case)
        if loaded is None:
            skipped.append({"name": case["name"], "label": case["label"], "reason": reason})
            print(f"[SKIP] {case['name']}: {reason}")
            continue
        loaded_cases.append(loaded)
        rows.extend(case_rows(loaded))
        print(f"[LOAD] {case['name']}: preds={loaded['preds'].shape}")

    if not rows:
        raise RuntimeError("No cases were loaded.")

    write_rows(rows)
    channels = sorted({row["channel"] for row in rows})
    for channel in channels:
        plot_channel(rows, channel, loaded_cases)
        plot_improvement(rows, channel, loaded_cases)
    write_summary(loaded_cases, skipped, rows)

    print("[SUMMARY] final-horizon phi model/copy")
    for case in loaded_cases:
        phi_rows = [row for row in rows if row["case"] == case["name"] and row["channel"] == "phi"]
        if not phi_rows:
            continue
        last = max(phi_rows, key=lambda row: row["horizon_ns"])
        print(
            f"  {case['name']}: horizon={last['horizon_ns']:.1f} ns, "
            f"model={last['model_mse_mean']:.6g}, copy={last['copy_mse_mean']:.6g}, "
            f"model/copy={last['model_over_copy_mean']:.3g}"
        )


if __name__ == "__main__":
    main()
