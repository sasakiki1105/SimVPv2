import csv
import json
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import factorized

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
OUTDIR = WORKDIRS / "compare_poisson_phi_from_density"

EPS0 = 8.8541878128e-12
E_CHARGE = 1.602176634e-19

SOURCES = (
    "model_phi",
    "poisson_pred_density",
    "poisson_true_density",
    "poisson_pred_vs_model_phi",
)


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
    if "minmax" not in norm_mode:
        raise ValueError(f"Unsupported norm mode: {norm_mode}")
    if len(timesteps) > 1:
        stride_steps = int(round(float(np.median(np.diff(timesteps)))))
    else:
        stride_steps = 1
    return {
        "props": props,
        "timesteps": timesteps,
        "train_min": train_min,
        "train_max": train_max,
        "margin": margin,
        "norm_mode": norm_mode,
        "stride_steps": stride_steps,
        "electron_index": props.index("electron_den"),
        "ion_index": props.index("ion_den"),
        "phi_index": props.index("phi"),
    }


def denorm_channel(x, info, channel_index):
    mn = float(info["train_min"][channel_index])
    mx = float(info["train_max"][channel_index])
    value_range = mx - mn
    lo = mn - info["margin"] * value_range
    hi = mx + info["margin"] * value_range
    return x.astype(np.float64) * (hi - lo) + lo


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


def build_poisson_solver(shape, dx, dy):
    h, w = shape
    nx = w - 2
    ny = h - 2
    if nx <= 0 or ny <= 0:
        raise ValueError(f"Need at least 3x3 grid, got {shape}")
    tx = diags(
        [np.ones(nx - 1), -2.0 * np.ones(nx), np.ones(nx - 1)],
        offsets=[-1, 0, 1],
        shape=(nx, nx),
    ) / (dx * dx)
    ty = diags(
        [np.ones(ny - 1), -2.0 * np.ones(ny), np.ones(ny - 1)],
        offsets=[-1, 0, 1],
        shape=(ny, ny),
    ) / (dy * dy)
    operator = kron(eye(ny), tx) + kron(ty, eye(nx))
    solve = factorized(operator.tocsc())
    return solve, nx, ny


def solve_phi_from_density(ne, ni, solve, nx, ny):
    rho = E_CHARGE * (ni - ne)
    rhs = (-rho[:, 1:-1, 1:-1] / EPS0).reshape(rho.shape[0], -1)
    sol = solve(rhs.T).T.reshape(rho.shape[0], ny, nx)
    phi = np.zeros_like(ne, dtype=np.float64)
    phi[:, 1:-1, 1:-1] = sol
    return phi


def build_test_starts(t_len):
    total = PRE + AFT
    val_end = int(np.floor(t_len * 0.9))
    last_start = t_len - total
    if last_start < val_end:
        return np.empty((0,), dtype=np.int64)
    return np.arange(val_end, last_start + 1, dtype=np.int64)


def per_sample_metrics(pred, true):
    pred = pred.reshape(pred.shape[0], -1)
    true = true.reshape(true.shape[0], -1)
    diff = pred - true
    mse = np.mean(diff * diff, axis=1)
    rmse = np.sqrt(mse)
    true_power = np.mean(true * true, axis=1)
    nrmse = rmse / np.sqrt(np.maximum(true_power, 1e-30))

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
    preds = np.load(case["saved"] / "preds.npy", mmap_mode="r")
    trues = np.load(case["saved"] / "trues.npy", mmap_mode="r")
    if preds.shape != trues.shape:
        raise ValueError(f"{case['name']} shape mismatch: {preds.shape} vs {trues.shape}")
    if preds.ndim != 5 or preds.shape[1] != AFT:
        raise ValueError(f"{case['name']} expected (N,{AFT},C,H,W), got {preds.shape}")
    info = load_h5_info(case["h5"])
    starts = build_test_starts(len(info["timesteps"]))
    if len(starts) != preds.shape[0]:
        raise ValueError(
            f"{case['name']} saved sample count {preds.shape[0]} does not match "
            f"computed test starts {len(starts)}"
        )
    dx, dy = domain_spacing(preds.shape[-2:])
    solve, nx, ny = build_poisson_solver(preds.shape[-2:], dx, dy)
    return {
        **case,
        "preds": preds,
        "trues": trues,
        "info": info,
        "starts": starts,
        "dt_ns": BASE_DT_NS * info["stride_steps"],
        "dx": dx,
        "dy": dy,
        "solve": solve,
        "nx": nx,
        "ny": ny,
    }


def compute_case_rows(case):
    info = case["info"]
    e_i = info["electron_index"]
    ion_i = info["ion_index"]
    phi_i = info["phi_index"]
    values_by_key = defaultdict(lambda: {"mse": [], "nrmse": [], "corr": []})
    raw_rows = []

    for tp in range(AFT):
        pred_ne = denorm_channel(case["preds"][:, tp, e_i], info, e_i)
        pred_ni = denorm_channel(case["preds"][:, tp, ion_i], info, ion_i)
        true_ne = denorm_channel(case["trues"][:, tp, e_i], info, e_i)
        true_ni = denorm_channel(case["trues"][:, tp, ion_i], info, ion_i)
        model_phi = denorm_channel(case["preds"][:, tp, phi_i], info, phi_i)
        true_phi = denorm_channel(case["trues"][:, tp, phi_i], info, phi_i)

        poisson_pred_phi = solve_phi_from_density(
            pred_ne, pred_ni, case["solve"], case["nx"], case["ny"]
        )
        poisson_true_phi = solve_phi_from_density(
            true_ne, true_ni, case["solve"], case["nx"], case["ny"]
        )

        comparisons = {
            "model_phi": (model_phi, true_phi),
            "poisson_pred_density": (poisson_pred_phi, true_phi),
            "poisson_true_density": (poisson_true_phi, true_phi),
            "poisson_pred_vs_model_phi": (poisson_pred_phi, model_phi),
        }

        target_indices = case["starts"] + PRE + tp
        target_timesteps = info["timesteps"][target_indices]
        horizon_ns = (tp + 1) * case["dt_ns"]

        for source, (pred_arr, true_arr) in comparisons.items():
            mse, nrmse, corr = per_sample_metrics(pred_arr, true_arr)
            for sample_idx, target_timestep, mse_i, nrmse_i, corr_i in zip(
                range(len(case["starts"])), target_timesteps, mse, nrmse, corr
            ):
                target_timestep = int(target_timestep)
                bucket = values_by_key[(source, target_timestep)]
                bucket["mse"].append(float(mse_i))
                bucket["nrmse"].append(float(nrmse_i))
                bucket["corr"].append(float(corr_i))
                raw_rows.append({
                    "case": case["name"],
                    "source": source,
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
    aggregate_by_source_timestep = {source: {} for source in SOURCES}
    for source, target_timestep in sorted(values_by_key):
        values = values_by_key[(source, target_timestep)]
        mse_summary = summarize(values["mse"])
        nrmse_summary = summarize(values["nrmse"])
        corr_summary = summarize(values["corr"])
        row = {
            "case": case["name"],
            "label": case["label"],
            "source": source,
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
        aggregate_by_source_timestep[source][target_timestep] = row
    return raw_rows, aggregate_rows, aggregate_by_source_timestep


def write_csv(path, rows):
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] {path}")


def common_rows(cases, source):
    common = None
    for case in cases:
        keys = set(case["aggregate_by_source_timestep"][source].keys())
        common = keys if common is None else common & keys
    common = sorted(common)
    rows = []
    for target_timestep in common:
        row = {
            "source": source,
            "target_timestep": int(target_timestep),
            "target_time_us": target_timestep * BASE_DT_NS / 1000.0,
        }
        for case in cases:
            source_row = case["aggregate_by_source_timestep"][source][target_timestep]
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
                row[f"{prefix}_{key}"] = source_row[key]
        rows.append(row)
    return rows


def rows_for_source(case, source):
    return [row for row in case["aggregate_rows"] if row["source"] == source]


def plot_metric(cases, source, metric, ylabel, out_png, common=None, yscale=None, ylim=None):
    plt.figure(figsize=(10.5, 5.5))
    if common is None:
        for case in cases:
            rows = rows_for_source(case, source)
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
    plt.title(f"Phi comparison: {source}")
    if yscale is not None:
        plt.yscale(yscale)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[PLOT] {out_png}")


def plot_sources_for_stride(case, metric, ylabel, out_png, yscale=None, ylim=None):
    styles = {
        "model_phi": ("-", "model phi vs true phi"),
        "poisson_pred_density": ("--", "Poisson(pred density) vs true phi"),
        "poisson_true_density": (":", "Poisson(true density) vs true phi"),
        "poisson_pred_vs_model_phi": ("-.", "Poisson(pred density) vs model phi"),
    }
    plt.figure(figsize=(10.5, 5.5))
    for source, (linestyle, label) in styles.items():
        rows = rows_for_source(case, source)
        x = np.asarray([row["target_time_us"] for row in rows])
        y = np.asarray([row[metric] for row in rows])
        plt.plot(x, y, color=case["color"], linestyle=linestyle, linewidth=1.7, label=label)
    plt.xlabel("Absolute target time in simulation (us)")
    plt.ylabel(ylabel)
    plt.title(f"Poisson-density phi comparisons, {case['label']}")
    if yscale is not None:
        plt.yscale(yscale)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[PLOT] {out_png}")


def save_summary(cases, common_by_source):
    summary = {
        "description": (
            "Phi reconstructed from predicted/true densities by solving 2D Poisson with zero "
            "Dirichlet boundaries, compared against true PIC phi and model-output phi. "
            "The H5 min-max margin is included when denormalizing."
        ),
        "poisson_equation": "laplacian(phi) = -e*(ion_den - electron_den)/eps0",
        "boundary_condition": "zero Dirichlet on x/y boundaries",
        "domain_info": str(DOMAIN_INFO),
        "base_dt_ns": BASE_DT_NS,
        "sources": list(SOURCES),
        "cases": [],
    }
    for case in cases:
        item = {
            "name": case["name"],
            "label": case["label"],
            "dt_ns": case["dt_ns"],
            "dx": case["dx"],
            "dy": case["dy"],
            "preds_shape": list(case["preds"].shape),
            "sources": {},
        }
        for source in SOURCES:
            rows = common_by_source[source]
            nrmse = np.asarray([row[f"{case['name']}_nrmse_median"] for row in rows])
            corr = np.asarray([row[f"{case['name']}_corr_median"] for row in rows])
            item["sources"][source] = {
                "common_times_median_nrmse": float(np.median(nrmse)),
                "common_times_median_corr": float(np.median(corr)),
            }
        summary["cases"].append(item)

    path = OUTDIR / "poisson_phi_from_density_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[JSON] {path}")
    print("[SUMMARY] common target times")
    for case_item in summary["cases"]:
        print(f"  {case_item['name']}")
        for source in SOURCES:
            source_item = case_item["sources"][source]
            print(
                f"    {source}: median NRMSE={source_item['common_times_median_nrmse']:.4g}, "
                f"median corr={source_item['common_times_median_corr']:.4g}"
            )


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cases = []
    raw_rows = []
    aggregate_rows = []
    for case_config in CASES:
        case = load_case(case_config)
        case_raw, case_agg, by_source_timestep = compute_case_rows(case)
        case["raw_rows"] = case_raw
        case["aggregate_rows"] = case_agg
        case["aggregate_by_source_timestep"] = by_source_timestep
        cases.append(case)
        raw_rows.extend(case_raw)
        aggregate_rows.extend(case_agg)
        print(f"[LOAD] {case['name']}: preds={case['preds'].shape}")

    write_csv(OUTDIR / "poisson_phi_from_density_raw.csv", raw_rows)
    write_csv(OUTDIR / "poisson_phi_from_density_by_target_time.csv", aggregate_rows)

    common_by_source = {source: common_rows(cases, source) for source in SOURCES}
    common_all = []
    for rows in common_by_source.values():
        common_all.extend(rows)
    write_csv(OUTDIR / "poisson_phi_from_density_common_times.csv", common_all)

    for source in SOURCES:
        common = common_by_source[source]
        plot_metric(
            cases,
            source,
            "nrmse_median",
            "Median relative RMSE, normalized by RMS(reference phi)",
            OUTDIR / f"poisson_phi_nrmse_common_times_{source}.png",
            common=common,
            yscale="log",
        )
        plot_metric(
            cases,
            source,
            "corr_median",
            "Median spatial correlation R",
            OUTDIR / f"poisson_phi_corr_common_times_{source}.png",
            common=common,
            ylim=(-0.05, 1.05),
        )

    for case in cases:
        plot_sources_for_stride(
            case,
            "nrmse_median",
            "Median relative RMSE",
            OUTDIR / f"poisson_phi_source_nrmse_{case['name']}.png",
            yscale="log",
        )
        plot_sources_for_stride(
            case,
            "corr_median",
            "Median spatial correlation R",
            OUTDIR / f"poisson_phi_source_corr_{case['name']}.png",
            ylim=(-0.05, 1.05),
        )

    save_summary(cases, common_by_source)


if __name__ == "__main__":
    main()
