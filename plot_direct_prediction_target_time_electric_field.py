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
DOMAIN_INFO = Path(
    r"C:\Users\astro\research\PEPAPIC\test\results"
    r"\2D_ExB_high_magnet_dt2.5e-9_maxt50e-6_macro5"
    r"\Domain_info\Global_domain_info.json"
)

BASE_DT_NS = 12.5
PRE = 10
AFT = 10
OUTDIR = WORKDIRS / "compare_direct_prediction_target_time_electric_field"
COMPONENTS = ("Ex", "Ey", "Emag")


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
        props = as_str_list(f["props"][()]) if "props" in f else ["electron_den", "ion_den", "phi"]
        timesteps = np.asarray(f["timesteps"][()], dtype=np.int64)
        train_min = np.asarray(f["train_min"][()], dtype=np.float64)
        train_max = np.asarray(f["train_max"][()], dtype=np.float64)
        margin = float(f["margin"][()]) if "margin" in f else 0.0
        norm_mode = f["norm_mode"][()]
        if isinstance(norm_mode, (bytes, bytearray)):
            norm_mode = norm_mode.decode()
        else:
            norm_mode = str(norm_mode)
    phi_index = props.index("phi")
    if len(timesteps) > 1:
        stride_steps = int(round(float(np.median(np.diff(timesteps)))))
    else:
        stride_steps = 1
    return {
        "props": props,
        "phi_index": phi_index,
        "timesteps": timesteps,
        "stride_steps": stride_steps,
        "phi_min": float(train_min[phi_index]),
        "phi_max": float(train_max[phi_index]),
        "margin": margin,
        "norm_mode": norm_mode,
    }


def domain_spacing(shape):
    h, w = shape
    try:
        with open(DOMAIN_INFO, "r", encoding="utf-8") as f:
            domain = json.load(f)
        mins = np.asarray(domain["min_zones"][0], dtype=np.float64)
        maxs = np.asarray(domain["max_zones"][0], dtype=np.float64)
        lx = float(maxs[0] - mins[0])
        ly = float(maxs[1] - mins[1])
    except Exception:
        lx = float(w - 1)
        ly = float(h - 1)
    dx = lx / float(w - 1) if w > 1 else 1.0
    dy = ly / float(h - 1) if h > 1 else 1.0
    return dx, dy


def build_test_starts(t_len):
    total = PRE + AFT
    val_end = int(np.floor(t_len * 0.9))
    last_start = t_len - total
    if last_start < val_end:
        return np.empty((0,), dtype=np.int64)
    return np.arange(val_end, last_start + 1, dtype=np.int64)


def denorm_phi(phi_norm, info):
    if "minmax" not in info["norm_mode"]:
        raise ValueError(f"Unsupported norm mode: {info['norm_mode']}")
    value_range = info["phi_max"] - info["phi_min"]
    lo = info["phi_min"] - info["margin"] * value_range
    hi = info["phi_max"] + info["margin"] * value_range
    return phi_norm.astype(np.float32) * np.float32(hi - lo) + np.float32(lo)


def electric_field(phi, dx, dy):
    dphi_dy, dphi_dx = np.gradient(phi, dy, dx, axis=(-2, -1), edge_order=1)
    return -dphi_dx, -dphi_dy


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


def component_errors(pred_ex, pred_ey, true_ex, true_ey, copy_ex, copy_ey):
    return {
        "Ex": (
            np.mean((pred_ex - true_ex) ** 2, axis=(1, 2)),
            np.mean((copy_ex - true_ex) ** 2, axis=(1, 2)),
        ),
        "Ey": (
            np.mean((pred_ey - true_ey) ** 2, axis=(1, 2)),
            np.mean((copy_ey - true_ey) ** 2, axis=(1, 2)),
        ),
        "Emag": (
            np.mean((pred_ex - true_ex) ** 2 + (pred_ey - true_ey) ** 2, axis=(1, 2)),
            np.mean((copy_ex - true_ex) ** 2 + (copy_ey - true_ey) ** 2, axis=(1, 2)),
        ),
    }


def load_case_aggregates(case):
    inputs = np.load(case["saved"] / "inputs.npy", mmap_mode="r")
    preds = np.load(case["saved"] / "preds.npy", mmap_mode="r")
    trues = np.load(case["saved"] / "trues.npy", mmap_mode="r")
    if preds.shape != trues.shape:
        raise ValueError(f"{case['name']} shape mismatch: {preds.shape} vs {trues.shape}")
    if preds.ndim != 5 or preds.shape[1] != AFT:
        raise ValueError(f"{case['name']} expected (N,{AFT},C,H,W), got {preds.shape}")
    if inputs.ndim != 5 or inputs.shape[1] != PRE:
        raise ValueError(f"{case['name']} expected inputs (N,{PRE},C,H,W), got {inputs.shape}")

    info = load_h5_info(case["h5"])
    timesteps = info["timesteps"]
    starts = build_test_starts(len(timesteps))
    if len(starts) != preds.shape[0]:
        raise ValueError(
            f"{case['name']} saved sample count {preds.shape[0]} does not match "
            f"computed test starts {len(starts)}"
        )

    dx, dy = domain_spacing(preds.shape[-2:])
    phi_i = info["phi_index"]
    copy_phi = denorm_phi(inputs[:, -1, phi_i], info)
    copy_ex, copy_ey = electric_field(copy_phi, dx, dy)

    values_by_key = defaultdict(list)
    raw_rows = []

    for tp in range(AFT):
        pred_phi = denorm_phi(preds[:, tp, phi_i], info)
        true_phi = denorm_phi(trues[:, tp, phi_i], info)
        pred_ex, pred_ey = electric_field(pred_phi, dx, dy)
        true_ex, true_ey = electric_field(true_phi, dx, dy)
        component_values = component_errors(pred_ex, pred_ey, true_ex, true_ey, copy_ex, copy_ey)

        target_indices = starts + PRE + tp
        target_timesteps = timesteps[target_indices]
        horizon_ns = (tp + 1) * case["dt_ns"]

        for component, (model_mse, copy_mse) in component_values.items():
            for sample_idx, target_timestep, model_value, copy_value in zip(
                range(len(starts)), target_timesteps, model_mse, copy_mse
            ):
                target_timestep = int(target_timestep)
                model_value = float(model_value)
                copy_value = float(copy_value)
                values_by_key[(component, target_timestep)].append((model_value, copy_value))
                raw_rows.append({
                    "case": case["name"],
                    "component": component,
                    "sample_index": int(sample_idx),
                    "output_index": int(tp + 1),
                    "horizon_ns": float(horizon_ns),
                    "target_timestep": target_timestep,
                    "target_time_us": target_timestep * BASE_DT_NS / 1000.0,
                    "model_mse": model_value,
                    "copy_mse": copy_value,
                    "model_over_copy": model_value / copy_value if copy_value > 0 else np.nan,
                })

    aggregate_rows = []
    aggregate_by_component_timestep = {component: {} for component in COMPONENTS}
    for component, target_timestep in sorted(values_by_key):
        pairs = values_by_key[(component, target_timestep)]
        model_values = [pair[0] for pair in pairs]
        copy_values = [pair[1] for pair in pairs]
        model_summary = summarize_values(model_values)
        copy_summary = summarize_values(copy_values)
        ratio = model_summary["mean"] / copy_summary["mean"] if copy_summary["mean"] > 0 else np.nan
        row = {
            "case": case["name"],
            "label": case["label"],
            "component": component,
            "target_timestep": int(target_timestep),
            "target_time_us": target_timestep * BASE_DT_NS / 1000.0,
            "count": model_summary["count"],
            "model_mean": model_summary["mean"],
            "model_median": model_summary["median"],
            "model_q25": model_summary["q25"],
            "model_q75": model_summary["q75"],
            "model_min": model_summary["min"],
            "model_max": model_summary["max"],
            "copy_mean": copy_summary["mean"],
            "copy_median": copy_summary["median"],
            "copy_q25": copy_summary["q25"],
            "copy_q75": copy_summary["q75"],
            "copy_min": copy_summary["min"],
            "copy_max": copy_summary["max"],
            "model_over_copy_mean": float(ratio),
            "improvement_fraction": float(1.0 - ratio) if np.isfinite(ratio) else np.nan,
        }
        aggregate_rows.append(row)
        aggregate_by_component_timestep[component][target_timestep] = row

    return {
        **case,
        "dt_ns": info["stride_steps"] * BASE_DT_NS,
        "dx": dx,
        "dy": dy,
        "preds_shape": list(preds.shape),
        "timesteps": timesteps,
        "starts": starts,
        "raw_rows": raw_rows,
        "aggregate_rows": aggregate_rows,
        "aggregate_by_component_timestep": aggregate_by_component_timestep,
        "h5_info": info,
    }


def add_dt_to_cases(cases):
    out = []
    for case in cases:
        info = load_h5_info(case["h5"])
        out.append({**case, "dt_ns": info["stride_steps"] * BASE_DT_NS})
    return out


def write_raw_csv(cases):
    path = OUTDIR / "electric_field_by_target_time_raw_predictions.csv"
    fieldnames = [
        "case",
        "component",
        "sample_index",
        "output_index",
        "horizon_ns",
        "target_timestep",
        "target_time_us",
        "model_mse",
        "copy_mse",
        "model_over_copy",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            writer.writerows(case["raw_rows"])
    print(f"[CSV] {path}")


def write_aggregate_csv(cases):
    path = OUTDIR / "electric_field_by_target_time_all_times.csv"
    fieldnames = [
        "case",
        "label",
        "component",
        "target_timestep",
        "target_time_us",
        "count",
        "model_mean",
        "model_median",
        "model_q25",
        "model_q75",
        "model_min",
        "model_max",
        "copy_mean",
        "copy_median",
        "copy_q25",
        "copy_q75",
        "copy_min",
        "copy_max",
        "model_over_copy_mean",
        "improvement_fraction",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            writer.writerows(case["aggregate_rows"])
    print(f"[CSV] {path}")


def common_timestep_rows(cases, component):
    common = None
    for case in cases:
        keys = set(case["aggregate_by_component_timestep"][component].keys())
        common = keys if common is None else common & keys
    common = sorted(common)

    rows = []
    for target_timestep in common:
        row = {
            "component": component,
            "target_timestep": int(target_timestep),
            "target_time_us": target_timestep * BASE_DT_NS / 1000.0,
        }
        for case in cases:
            agg = case["aggregate_by_component_timestep"][component][target_timestep]
            prefix = case["name"]
            for key in (
                "count",
                "model_mean",
                "model_median",
                "model_q25",
                "model_q75",
                "copy_mean",
                "copy_median",
                "copy_q25",
                "copy_q75",
                "model_over_copy_mean",
                "improvement_fraction",
            ):
                row[f"{prefix}_{key}"] = agg[key]
        rows.append(row)
    return rows


def write_common_csv(common_rows_by_component):
    rows = []
    for component_rows in common_rows_by_component.values():
        rows.extend(component_rows)
    path = OUTDIR / "electric_field_by_target_time_common_times.csv"
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] {path}")


def aggregate_rows_for(case, component):
    return [row for row in case["aggregate_rows"] if row["component"] == component]


def plot_model_and_copy(cases, component, metric, out_png, title, common_rows=None):
    plt.figure(figsize=(10.5, 5.8))
    if common_rows is None:
        for case in cases:
            rows = aggregate_rows_for(case, component)
            x = np.asarray([row["target_time_us"] for row in rows])
            model_y = np.asarray([row[f"model_{metric}"] for row in rows])
            copy_y = np.asarray([row[f"copy_{metric}"] for row in rows])
            plt.plot(x, model_y, color=case["color"], linewidth=1.7, label=f"{case['label']} model")
            plt.plot(
                x,
                copy_y,
                color=case["color"],
                linestyle="--",
                linewidth=1.0,
                alpha=0.65,
                label=f"{case['label']} copy",
            )
    else:
        x = np.asarray([row["target_time_us"] for row in common_rows])
        for case in cases:
            prefix = case["name"]
            model_y = np.asarray([row[f"{prefix}_model_{metric}"] for row in common_rows])
            copy_y = np.asarray([row[f"{prefix}_copy_{metric}"] for row in common_rows])
            plt.plot(x, model_y, color=case["color"], linewidth=1.7, label=f"{case['label']} model")
            plt.plot(
                x,
                copy_y,
                color=case["color"],
                linestyle="--",
                linewidth=1.0,
                alpha=0.65,
                label=f"{case['label']} copy",
            )
    plt.xlabel("Absolute target time in simulation (us)")
    plt.ylabel(f"Aggregated {metric} E-MSE ({component})")
    plt.title(title)
    plt.yscale("log")
    plt.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[PLOT] {out_png}")


def plot_ratio(cases, component, out_png, title, common_rows=None):
    plt.figure(figsize=(10.5, 5.5))
    if common_rows is None:
        for case in cases:
            rows = aggregate_rows_for(case, component)
            x = np.asarray([row["target_time_us"] for row in rows])
            y = np.asarray([row["model_over_copy_mean"] for row in rows])
            plt.plot(x, y, color=case["color"], linewidth=1.7, label=case["label"])
    else:
        x = np.asarray([row["target_time_us"] for row in common_rows])
        for case in cases:
            prefix = case["name"]
            y = np.asarray([row[f"{prefix}_model_over_copy_mean"] for row in common_rows])
            plt.plot(x, y, color=case["color"], linewidth=1.7, label=case["label"])
    plt.axhline(1.0, color="#555555", linestyle="--", linewidth=1.0, label="copy parity")
    plt.xlabel("Absolute target time in simulation (us)")
    plt.ylabel("Model E-MSE / copy E-MSE")
    plt.title(title)
    plt.yscale("log")
    plt.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[PLOT] {out_png}")


def plot_counts(cases, common_rows_by_component):
    component = "Emag"
    plt.figure(figsize=(10, 4.8))
    for case in cases:
        rows = aggregate_rows_for(case, component)
        x = np.asarray([row["target_time_us"] for row in rows])
        y = np.asarray([row["count"] for row in rows])
        plt.plot(x, y, color=case["color"], linewidth=1.7, label=case["label"])
    plt.xlabel("Absolute target time in simulation (us)")
    plt.ylabel("Number of direct predictions aggregated")
    plt.title("Electric-field target-time aggregate counts, all available target times")
    plt.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    out_png = OUTDIR / "electric_field_target_time_aggregate_counts_all_times.png"
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[PLOT] {out_png}")

    common_rows = common_rows_by_component[component]
    plt.figure(figsize=(10, 4.8))
    x = np.asarray([row["target_time_us"] for row in common_rows])
    for case in cases:
        y = np.asarray([row[f"{case['name']}_count"] for row in common_rows])
        plt.plot(x, y, color=case["color"], linewidth=1.7, label=case["label"])
    plt.xlabel("Absolute target time in simulation (us)")
    plt.ylabel("Number of direct predictions aggregated")
    plt.title("Electric-field aggregate counts on common target times")
    plt.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    out_png = OUTDIR / "electric_field_target_time_aggregate_counts_common_times.png"
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[PLOT] {out_png}")


def save_summary(cases, common_rows_by_component):
    summary = {
        "description": (
            "Electric-field direct prediction errors grouped by absolute target simulation time. "
            "E is computed from denormalized phi using E=-grad(phi). For each target time "
            "and stride, all output indices/windows that predict that same target frame are aggregated."
        ),
        "domain_info": str(DOMAIN_INFO),
        "base_dt_ns": BASE_DT_NS,
        "pre_seq_length": PRE,
        "aft_seq_length": AFT,
        "components": list(COMPONENTS),
        "cases": [],
        "common_target_times": {},
    }
    for component, common_rows in common_rows_by_component.items():
        summary["common_target_times"][component] = {
            "n": len(common_rows),
            "range_us": [
                float(common_rows[0]["target_time_us"]),
                float(common_rows[-1]["target_time_us"]),
            ],
        }

    for case in cases:
        case_summary = {
            "name": case["name"],
            "label": case["label"],
            "h5": str(case["h5"]),
            "saved": str(case["saved"]),
            "dt_ns": case["dt_ns"],
            "dx": case["dx"],
            "dy": case["dy"],
            "preds_shape": case["preds_shape"],
            "components": {},
        }
        for component in COMPONENTS:
            rows = aggregate_rows_for(case, component)
            common_rows = common_rows_by_component[component]
            all_model_medians = np.asarray([row["model_median"] for row in rows])
            all_model_means = np.asarray([row["model_mean"] for row in rows])
            all_ratios = np.asarray([row["model_over_copy_mean"] for row in rows])
            common_model_medians = np.asarray([
                row[f"{case['name']}_model_median"] for row in common_rows
            ])
            common_model_means = np.asarray([
                row[f"{case['name']}_model_mean"] for row in common_rows
            ])
            common_ratios = np.asarray([
                row[f"{case['name']}_model_over_copy_mean"] for row in common_rows
            ])
            case_summary["components"][component] = {
                "n_target_times_all": len(rows),
                "target_time_us_all": [
                    float(rows[0]["target_time_us"]),
                    float(rows[-1]["target_time_us"]),
                ],
                "all_times_median_of_model_medians": float(np.median(all_model_medians)),
                "all_times_mean_of_model_means": float(np.mean(all_model_means)),
                "all_times_median_model_over_copy": float(np.median(all_ratios)),
                "common_times_median_of_model_medians": float(np.median(common_model_medians)),
                "common_times_mean_of_model_means": float(np.mean(common_model_means)),
                "common_times_median_model_over_copy": float(np.median(common_ratios)),
            }
        summary["cases"].append(case_summary)

    path = OUTDIR / "electric_field_target_time_aggregate_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[JSON] {path}")

    print("[SUMMARY] common target times, Emag")
    for case in cases:
        item = next(s for s in summary["cases"] if s["name"] == case["name"])
        comp = item["components"]["Emag"]
        print(
            f"{case['name']}: median model E-MSE={comp['common_times_median_of_model_medians']:.6g}, "
            f"median model/copy={comp['common_times_median_model_over_copy']:.3g}"
        )


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    case_configs = add_dt_to_cases(CASES)
    cases = [load_case_aggregates(case) for case in case_configs]

    write_raw_csv(cases)
    write_aggregate_csv(cases)
    common_rows_by_component = {
        component: common_timestep_rows(cases, component) for component in COMPONENTS
    }
    write_common_csv(common_rows_by_component)

    for component in COMPONENTS:
        common_rows = common_rows_by_component[component]
        plot_model_and_copy(
            cases,
            component,
            "median",
            OUTDIR / f"electric_field_target_time_median_all_times_{component}.png",
            f"Target-time aggregated electric-field median E-MSE, all times: {component}",
        )
        plot_model_and_copy(
            cases,
            component,
            "mean",
            OUTDIR / f"electric_field_target_time_mean_all_times_{component}.png",
            f"Target-time aggregated electric-field mean E-MSE, all times: {component}",
        )
        plot_model_and_copy(
            cases,
            component,
            "median",
            OUTDIR / f"electric_field_target_time_median_common_times_{component}.png",
            f"Target-time aggregated electric-field median E-MSE, common times: {component}",
            common_rows=common_rows,
        )
        plot_model_and_copy(
            cases,
            component,
            "mean",
            OUTDIR / f"electric_field_target_time_mean_common_times_{component}.png",
            f"Target-time aggregated electric-field mean E-MSE, common times: {component}",
            common_rows=common_rows,
        )
        plot_ratio(
            cases,
            component,
            OUTDIR / f"electric_field_target_time_model_over_copy_all_times_{component}.png",
            f"Target-time electric-field model/copy E-MSE, all times: {component}",
        )
        plot_ratio(
            cases,
            component,
            OUTDIR / f"electric_field_target_time_model_over_copy_common_times_{component}.png",
            f"Target-time electric-field model/copy E-MSE, common times: {component}",
            common_rows=common_rows,
        )

    plot_counts(cases, common_rows_by_component)
    save_summary(cases, common_rows_by_component)


if __name__ == "__main__":
    main()
