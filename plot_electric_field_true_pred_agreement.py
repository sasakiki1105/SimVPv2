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
OUTDIR = WORKDIRS / "compare_electric_field_true_pred_agreement"
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


def component_arrays(ex, ey):
    return {
        "Ex": ex,
        "Ey": ey,
        "Emag": np.sqrt(ex * ex + ey * ey),
    }


def agreement_metrics(pred, true):
    pred = pred.reshape(pred.shape[0], -1).astype(np.float64)
    true = true.reshape(true.shape[0], -1).astype(np.float64)
    diff = pred - true
    mse = np.mean(diff * diff, axis=1)
    true_power = np.mean(true * true, axis=1)
    nrmse = np.sqrt(mse / np.maximum(true_power, 1e-30))

    pred_centered = pred - np.mean(pred, axis=1, keepdims=True)
    true_centered = true - np.mean(true, axis=1, keepdims=True)
    denom = np.sqrt(
        np.sum(pred_centered * pred_centered, axis=1)
        * np.sum(true_centered * true_centered, axis=1)
    )
    corr = np.sum(pred_centered * true_centered, axis=1) / np.maximum(denom, 1e-30)
    return mse, nrmse, corr


def summarize(values):
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


def load_case(case):
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
    return {
        **case,
        "inputs": inputs,
        "preds": preds,
        "trues": trues,
        "info": info,
        "timesteps": timesteps,
        "starts": starts,
        "dt_ns": BASE_DT_NS * info["stride_steps"],
        "dx": dx,
        "dy": dy,
    }


def compute_case_rows(case):
    phi_i = case["info"]["phi_index"]
    values_by_key = defaultdict(lambda: {"mse": [], "nrmse": [], "corr": []})
    raw_rows = []
    for tp in range(AFT):
        pred_phi = denorm_phi(case["preds"][:, tp, phi_i], case["info"])
        true_phi = denorm_phi(case["trues"][:, tp, phi_i], case["info"])
        pred_ex, pred_ey = electric_field(pred_phi, case["dx"], case["dy"])
        true_ex, true_ey = electric_field(true_phi, case["dx"], case["dy"])
        pred_components = component_arrays(pred_ex, pred_ey)
        true_components = component_arrays(true_ex, true_ey)

        target_indices = case["starts"] + PRE + tp
        target_timesteps = case["timesteps"][target_indices]
        horizon_ns = (tp + 1) * case["dt_ns"]
        for component in COMPONENTS:
            mse, nrmse, corr = agreement_metrics(
                pred_components[component], true_components[component]
            )
            for sample_idx, target_timestep, mse_i, nrmse_i, corr_i in zip(
                range(len(case["starts"])), target_timesteps, mse, nrmse, corr
            ):
                target_timestep = int(target_timestep)
                values = values_by_key[(component, target_timestep)]
                values["mse"].append(float(mse_i))
                values["nrmse"].append(float(nrmse_i))
                values["corr"].append(float(corr_i))
                raw_rows.append({
                    "case": case["name"],
                    "component": component,
                    "sample_index": int(sample_idx),
                    "output_index": int(tp + 1),
                    "horizon_ns": float(horizon_ns),
                    "target_timestep": target_timestep,
                    "target_time_us": target_timestep * BASE_DT_NS / 1000.0,
                    "mse": float(mse_i),
                    "nrmse": float(nrmse_i),
                    "corr": float(corr_i),
                })

    aggregate_rows = []
    aggregate_by_component_timestep = {component: {} for component in COMPONENTS}
    for component, target_timestep in sorted(values_by_key):
        values = values_by_key[(component, target_timestep)]
        mse_summary = summarize(values["mse"])
        nrmse_summary = summarize(values["nrmse"])
        corr_summary = summarize(values["corr"])
        row = {
            "case": case["name"],
            "label": case["label"],
            "component": component,
            "target_timestep": int(target_timestep),
            "target_time_us": target_timestep * BASE_DT_NS / 1000.0,
            "count": mse_summary["count"],
            "mse_mean": mse_summary["mean"],
            "mse_median": mse_summary["median"],
            "mse_q25": mse_summary["q25"],
            "mse_q75": mse_summary["q75"],
            "nrmse_mean": nrmse_summary["mean"],
            "nrmse_median": nrmse_summary["median"],
            "nrmse_q25": nrmse_summary["q25"],
            "nrmse_q75": nrmse_summary["q75"],
            "corr_mean": corr_summary["mean"],
            "corr_median": corr_summary["median"],
            "corr_q25": corr_summary["q25"],
            "corr_q75": corr_summary["q75"],
        }
        aggregate_rows.append(row)
        aggregate_by_component_timestep[component][target_timestep] = row
    return raw_rows, aggregate_rows, aggregate_by_component_timestep


def write_csv(path, rows):
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] {path}")


def common_rows(cases, component):
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
            source = case["aggregate_by_component_timestep"][component][target_timestep]
            prefix = case["name"]
            for key in (
                "count",
                "mse_mean",
                "mse_median",
                "nrmse_mean",
                "nrmse_median",
                "corr_mean",
                "corr_median",
            ):
                row[f"{prefix}_{key}"] = source[key]
        rows.append(row)
    return rows


def plot_metric(cases, component, metric, ylabel, out_png, common=None, ylim=None):
    plt.figure(figsize=(10.5, 5.5))
    if common is None:
        for case in cases:
            rows = [
                row for row in case["aggregate_rows"]
                if row["component"] == component
            ]
            x = np.asarray([row["target_time_us"] for row in rows])
            y = np.asarray([row[metric] for row in rows])
            plt.plot(x, y, color=case["color"], linewidth=1.7, label=case["label"])
    else:
        x = np.asarray([row["target_time_us"] for row in common])
        for case in cases:
            y = np.asarray([row[f"{case['name']}_{metric}"] for row in common])
            plt.plot(x, y, color=case["color"], linewidth=1.7, label=case["label"])
    plt.xlabel("Absolute target time in simulation (us)")
    plt.ylabel(ylabel)
    plt.title(f"True vs predicted electric field agreement: {component}")
    if metric.startswith("mse") or metric.startswith("nrmse"):
        plt.yscale("log")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[PLOT] {out_png}")


def collect_scatter_points(case, target_timestep, component, max_points=30000):
    phi_i = case["info"]["phi_index"]
    points_true = []
    points_pred = []
    for tp in range(AFT):
        target_indices = case["starts"] + PRE + tp
        hit = np.where(case["timesteps"][target_indices] == target_timestep)[0]
        if len(hit) == 0:
            continue
        pred_phi = denorm_phi(case["preds"][hit, tp, phi_i], case["info"])
        true_phi = denorm_phi(case["trues"][hit, tp, phi_i], case["info"])
        pred_ex, pred_ey = electric_field(pred_phi, case["dx"], case["dy"])
        true_ex, true_ey = electric_field(true_phi, case["dx"], case["dy"])
        pred_arr = component_arrays(pred_ex, pred_ey)[component].reshape(-1)
        true_arr = component_arrays(true_ex, true_ey)[component].reshape(-1)
        points_true.append(true_arr)
        points_pred.append(pred_arr)
    if not points_true:
        return None, None
    true = np.concatenate(points_true)
    pred = np.concatenate(points_pred)
    if len(true) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(true), size=max_points, replace=False)
        true = true[idx]
        pred = pred[idx]
    return true, pred


def plot_scatter_grid(cases, component, target_timestep):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    axes = axes.ravel()
    all_true = []
    all_pred = []
    collected = []
    for case in cases:
        true, pred = collect_scatter_points(case, target_timestep, component)
        collected.append((case, true, pred))
        if true is not None:
            all_true.append(true)
            all_pred.append(pred)
    if not all_true:
        return
    joined = np.concatenate(all_true + all_pred)
    lo, hi = np.quantile(joined, [0.002, 0.998])
    if lo == hi:
        lo, hi = float(np.min(joined)), float(np.max(joined))
    for ax, (case, true, pred) in zip(axes, collected):
        if true is None:
            ax.set_axis_off()
            continue
        ax.scatter(true, pred, s=2, alpha=0.18, color=case["color"], edgecolors="none")
        ax.plot([lo, hi], [lo, hi], color="#555555", linestyle="--", linewidth=1.0)
        mse = float(np.mean((pred - true) ** 2))
        nrmse = float(np.sqrt(mse / max(float(np.mean(true * true)), 1e-30)))
        corr = float(np.corrcoef(true, pred)[0, 1])
        ax.set_title(f"{case['label']}\nR={corr:.3f}, NRMSE={nrmse:.3g}")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
    fig.supxlabel(f"True {component} from PIC phi")
    fig.supylabel(f"Predicted {component} from predicted phi")
    target_time_us = target_timestep * BASE_DT_NS / 1000.0
    fig.suptitle(f"True vs predicted {component} at target time {target_time_us:.3f} us")
    out_png = OUTDIR / f"electric_field_true_vs_pred_scatter_{component}_t{target_timestep}.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[PLOT] {out_png}")


def save_summary(cases, common_by_component):
    summary = {
        "description": (
            "Agreement between electric fields computed from true PIC phi and from predicted phi. "
            "E is computed as E=-grad(phi). NRMSE is normalized by RMS(true E)."
        ),
        "domain_info": str(DOMAIN_INFO),
        "base_dt_ns": BASE_DT_NS,
        "components": list(COMPONENTS),
        "cases": [],
    }
    for case in cases:
        item = {
            "name": case["name"],
            "label": case["label"],
            "dt_ns": case["dt_ns"],
            "preds_shape": list(case["preds"].shape),
            "components": {},
        }
        for component in COMPONENTS:
            rows = common_by_component[component]
            nrmse = np.asarray([row[f"{case['name']}_nrmse_median"] for row in rows])
            corr = np.asarray([row[f"{case['name']}_corr_median"] for row in rows])
            item["components"][component] = {
                "common_times_median_nrmse": float(np.median(nrmse)),
                "common_times_median_corr": float(np.median(corr)),
            }
        summary["cases"].append(item)
    path = OUTDIR / "electric_field_true_pred_agreement_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[JSON] {path}")
    print("[SUMMARY] common target times, Emag")
    for item in summary["cases"]:
        comp = item["components"]["Emag"]
        print(
            f"{item['name']}: median NRMSE={comp['common_times_median_nrmse']:.4g}, "
            f"median corr={comp['common_times_median_corr']:.4g}"
        )


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cases = []
    raw_rows = []
    aggregate_rows = []
    for case_config in CASES:
        case = load_case(case_config)
        case_raw, case_agg, by_component_timestep = compute_case_rows(case)
        case["raw_rows"] = case_raw
        case["aggregate_rows"] = case_agg
        case["aggregate_by_component_timestep"] = by_component_timestep
        cases.append(case)
        raw_rows.extend(case_raw)
        aggregate_rows.extend(case_agg)
        print(f"[LOAD] {case['name']}: preds={case['preds'].shape}")

    write_csv(OUTDIR / "electric_field_true_pred_raw.csv", raw_rows)
    write_csv(OUTDIR / "electric_field_true_pred_by_target_time.csv", aggregate_rows)

    common_by_component = {component: common_rows(cases, component) for component in COMPONENTS}
    common_all = []
    for rows in common_by_component.values():
        common_all.extend(rows)
    write_csv(OUTDIR / "electric_field_true_pred_common_times.csv", common_all)

    for component in COMPONENTS:
        rows = common_by_component[component]
        plot_metric(
            cases,
            component,
            "corr_median",
            "Median spatial correlation R",
            OUTDIR / f"electric_field_true_pred_corr_common_times_{component}.png",
            common=rows,
            ylim=(-0.05, 1.05),
        )
        plot_metric(
            cases,
            component,
            "nrmse_median",
            "Median relative RMSE, normalized by RMS(true E)",
            OUTDIR / f"electric_field_true_pred_nrmse_common_times_{component}.png",
            common=rows,
        )

    common_emag = common_by_component["Emag"]
    middle = common_emag[len(common_emag) // 2]["target_timestep"]
    late = common_emag[-3]["target_timestep"] if len(common_emag) >= 3 else common_emag[-1]["target_timestep"]
    for target_timestep in (middle, late):
        for component in ("Ex", "Ey", "Emag"):
            plot_scatter_grid(cases, component, int(target_timestep))

    save_summary(cases, common_by_component)


if __name__ == "__main__":
    main()
