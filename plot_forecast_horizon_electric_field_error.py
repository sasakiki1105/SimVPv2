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
DOMAIN_INFO = Path(
    r"C:\Users\astro\research\PEPAPIC\test\results"
    r"\2D_ExB_high_magnet_dt2.5e-9_maxt50e-6_macro5"
    r"\Domain_info\Global_domain_info.json"
)

BASE_DT_NS = 12.5
PRE = 10
OUTDIR = WORKDIRS / "compare_forecast_horizon_electric_field"


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


def domain_spacing(shape):
    h, w = shape
    # Global_domain_info.json currently stores min_zones=[0,0,0], max_zones=[1,1,0].
    # Use the domain span when available; otherwise fall back to cell units.
    try:
        import json as json_module

        with open(DOMAIN_INFO, "r", encoding="utf-8") as f:
            domain = json_module.load(f)
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
        "phi_min": float(train_min[phi_index]),
        "phi_max": float(train_max[phi_index]),
        "margin": margin,
        "norm_mode": norm_mode,
        "stride_steps": stride_steps,
    }


def denorm_phi(phi_norm, info):
    if "minmax" not in info["norm_mode"]:
        raise ValueError(f"Unsupported norm_mode for phi denorm: {info['norm_mode']}")
    value_range = info["phi_max"] - info["phi_min"]
    lo = info["phi_min"] - info["margin"] * value_range
    hi = info["phi_max"] + info["margin"] * value_range
    return phi_norm.astype(np.float32) * np.float32(hi - lo) + np.float32(lo)


def electric_field(phi, dx, dy):
    # phi shape: (..., H, W). np.gradient returns dphi/dy, dphi/dx.
    dphi_dy, dphi_dx = np.gradient(phi, dy, dx, axis=(-2, -1), edge_order=1)
    return -dphi_dx, -dphi_dy


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
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

    info = load_h5_info(case["h5"])
    dx, dy = domain_spacing(preds.shape[-2:])
    loaded = dict(case)
    loaded.update({
        "inputs": inputs,
        "preds": preds,
        "trues": trues,
        "h5_info": info,
        "dt_ns": BASE_DT_NS * info["stride_steps"],
        "dx": dx,
        "dy": dy,
        "saved": saved,
    })
    return loaded, None


def component_rows(case):
    inputs = case["inputs"]
    preds = case["preds"]
    trues = case["trues"]
    info = case["h5_info"]
    phi_i = info["phi_index"]
    dx, dy = case["dx"], case["dy"]

    last_input_phi = denorm_phi(inputs[:, -1, phi_i], info)
    copy_ex, copy_ey = electric_field(last_input_phi, dx, dy)

    rows = []
    for tp in range(preds.shape[1]):
        pred_phi = denorm_phi(preds[:, tp, phi_i], info)
        true_phi = denorm_phi(trues[:, tp, phi_i], info)
        pred_ex, pred_ey = electric_field(pred_phi, dx, dy)
        true_ex, true_ey = electric_field(true_phi, dx, dy)
        horizon_ns = (tp + 1) * case["dt_ns"]
        comp_values = {
            "Ex": (
                np.mean((pred_ex - true_ex) ** 2, axis=(1, 2)),
                np.mean((copy_ex - true_ex) ** 2, axis=(1, 2)),
            ),
            "Ey": (
                np.mean((pred_ey - true_ey) ** 2, axis=(1, 2)),
                np.mean((copy_ey - true_ey) ** 2, axis=(1, 2)),
            ),
            "Emag": (
                np.mean(
                    (pred_ex - true_ex) ** 2
                    + (pred_ey - true_ey) ** 2,
                    axis=(1, 2),
                ),
                np.mean(
                    (copy_ex - true_ex) ** 2
                    + (copy_ey - true_ey) ** 2,
                    axis=(1, 2),
                ),
            ),
        }
        for component, (model_values, copy_values) in comp_values.items():
            model_summary = summarize(model_values)
            copy_summary = summarize(copy_values)
            ratio = model_summary["mean"] / copy_summary["mean"] if copy_summary["mean"] > 0 else np.nan
            rows.append({
                "case": case["name"],
                "label": case["label"],
                "stride": case["stride"],
                "tout": case["tout"],
                "dt_ns": case["dt_ns"],
                "output_index": tp + 1,
                "horizon_ns": float(horizon_ns),
                "component": component,
                "n_samples": int(len(model_values)),
                "dx": float(dx),
                "dy": float(dy),
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
    path = OUTDIR / "electric_field_model_vs_copy.csv"
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] {path}")


def plot_component(rows, component, cases):
    plt.figure(figsize=(10, 6))
    for case in cases:
        r = [row for row in rows if row["case"] == case["name"] and row["component"] == component]
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
    plt.ylabel(f"MSE ({component}, denormalized phi gradient)")
    plt.title(f"Electric field error from phi: {component}")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out_png = OUTDIR / f"electric_field_model_vs_copy_{component}.png"
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[PLOT] {out_png}")


def plot_ratio(rows, component, cases):
    plt.figure(figsize=(10, 5.8))
    for case in cases:
        r = [row for row in rows if row["case"] == case["name"] and row["component"] == component]
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
    plt.ylabel("Model E-MSE / copy E-MSE")
    plt.title(f"Electric field improvement over copy baseline: {component}")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out_png = OUTDIR / f"electric_field_model_over_copy_{component}.png"
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[PLOT] {out_png}")


def write_summary(loaded_cases, skipped):
    summary = {
        "description": (
            "Electric-field forecast-horizon evaluation. E is computed from denormalized phi "
            "using E=-grad(phi). Copy baseline repeats the last input phi frame before taking grad."
        ),
        "domain_info": str(DOMAIN_INFO),
        "base_dt_ns": BASE_DT_NS,
        "pre_seq_length": PRE,
        "loaded_cases": [
            {
                "name": case["name"],
                "label": case["label"],
                "stride": case["stride"],
                "tout": case["tout"],
                "dt_ns": case["dt_ns"],
                "dx": case["dx"],
                "dy": case["dy"],
                "h5": str(case["h5"]),
                "saved": str(case["saved"]),
                "preds_shape": list(case["preds"].shape),
                "phi_min": case["h5_info"]["phi_min"],
                "phi_max": case["h5_info"]["phi_max"],
                "margin": case["h5_info"]["margin"],
                "norm_mode": case["h5_info"]["norm_mode"],
            }
            for case in loaded_cases
        ],
        "skipped_cases": skipped,
    }
    path = OUTDIR / "electric_field_model_vs_copy_summary.json"
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
        rows.extend(component_rows(loaded))
        print(f"[LOAD] {case['name']}: preds={loaded['preds'].shape}")

    if not rows:
        raise RuntimeError("No cases were loaded.")

    write_rows(rows)
    for component in ("Ex", "Ey", "Emag"):
        plot_component(rows, component, loaded_cases)
        plot_ratio(rows, component, loaded_cases)
    write_summary(loaded_cases, skipped)

    print("[SUMMARY] final-horizon Emag model/copy")
    for case in loaded_cases:
        r = [row for row in rows if row["case"] == case["name"] and row["component"] == "Emag"]
        if not r:
            continue
        last = max(r, key=lambda row: row["horizon_ns"])
        print(
            f"  {case['name']}: horizon={last['horizon_ns']:.1f} ns, "
            f"model={last['model_mse_mean']:.6g}, copy={last['copy_mse_mean']:.6g}, "
            f"model/copy={last['model_over_copy_mean']:.3g}"
        )


if __name__ == "__main__":
    main()
