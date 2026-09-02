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
OUTDIR = WORKDIRS / "compare_weak_regularization_3b_direct10"
H5_PATH = Path(
    r"C:\Users\astro\research\PEPAPIC\test\results"
    r"\2D_ExB_high_magnet_dt2.5e-9_maxt50e-6_macro5"
    r"\SimVPv2_inputs\global_norm_trainfixed_minmax_margin20_t0_to_t4000_step2.h5"
)
DOMAIN_INFO = Path(
    r"C:\Users\astro\research\PEPAPIC\test\results"
    r"\2D_ExB_high_magnet_dt2.5e-9_maxt50e-6_macro5"
    r"\Domain_info\Global_domain_info.json"
)

EPS0 = 8.8541878128e-12
E_CHARGE = 1.602176634e-19
EPS = 1e-30
CHANNELS = ["electron_den", "ion_den", "phi"]
COLORS = {
    "copy": "#6b7280",
    "data_only": "#2563eb",
    "poisson_zero": "#16a34a",
    "floor_hinge": "#dc2626",
    "efield": "#7c3aed",
    "poisson_zero_efield": "#ea580c",
}
LABELS = {
    "copy": "Copy baseline",
    "data_only": "Data-only",
    "poisson_zero": "Weak Poisson",
    "floor_hinge": "Poisson floor hinge",
    "efield": "Weak E-field",
    "poisson_zero_efield": "Weak Poisson + E-field",
}
CASES = [
    {
        "key": "data_only",
        "ex_name": "pepapic_simvp_gsta_highmag_macro5_subsample2_trainfixed_disjoint_811_bs2_100ep",
    },
    {
        "key": "poisson_zero",
        "ex_name": (
            "pepapic_simvp_gsta_highmag_macro5_subsample2_direct10_"
            "poisson_zero_lam1em3_trainfixed_disjoint_811_bs2_100ep"
        ),
    },
    {
        "key": "floor_hinge",
        "ex_name": (
            "pepapic_simvp_gsta_highmag_macro5_subsample2_direct10_"
            "poisson_floor_hinge_lam1em3_floor086_alpha11_trainfixed_disjoint_811_bs2_100ep"
        ),
    },
    {
        "key": "efield",
        "ex_name": (
            "pepapic_simvp_gsta_highmag_macro5_subsample2_direct10_"
            "efield_lam1em3_trainfixed_disjoint_811_bs2_100ep"
        ),
    },
    {
        "key": "poisson_zero_efield",
        "ex_name": (
            "pepapic_simvp_gsta_highmag_macro5_subsample2_direct10_"
            "poisson_zero_lam1em3_efield_lam1em3_trainfixed_disjoint_811_bs2_100ep"
        ),
    },
]


def write_csv(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] {path}", flush=True)


def as_str_list(values):
    out = []
    for value in values:
        if isinstance(value, (bytes, bytearray)):
            out.append(value.decode())
        else:
            out.append(str(value))
    return out


def load_h5_info():
    with h5py.File(H5_PATH, "r") as f:
        props = as_str_list(f["props"][()])
        train_min = np.asarray(f["train_min"][()], dtype=np.float64)
        train_max = np.asarray(f["train_max"][()], dtype=np.float64)
        margin = float(f["margin"][()])
    return {
        "props": props,
        "train_min": train_min,
        "train_max": train_max,
        "margin": margin,
        "indices": {name: props.index(name) for name in props},
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
    return lx / float(w - 1), ly / float(h - 1)


def denorm(x, info, channel_index):
    mn = float(info["train_min"][channel_index])
    mx = float(info["train_max"][channel_index])
    value_range = mx - mn
    lo = mn - info["margin"] * value_range
    hi = mx + info["margin"] * value_range
    return np.asarray(x, dtype=np.float64) * (hi - lo) + lo


def finite_stats(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "q25": np.nan, "q75": np.nan}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
    }


def corr_per_frame(a, b):
    flat_a = a.reshape(a.shape[0], -1)
    flat_b = b.reshape(b.shape[0], -1)
    a0 = flat_a - np.mean(flat_a, axis=1, keepdims=True)
    b0 = flat_b - np.mean(flat_b, axis=1, keepdims=True)
    denom = np.sqrt(np.sum(a0 * a0, axis=1) * np.sum(b0 * b0, axis=1))
    return np.sum(a0 * b0, axis=1) / np.maximum(denom, EPS)


def laplacian_interior(phi, dx, dy):
    return (
        (phi[:, 1:-1, 2:] - 2.0 * phi[:, 1:-1, 1:-1] + phi[:, 1:-1, :-2]) / (dx * dx)
        + (phi[:, 2:, 1:-1] - 2.0 * phi[:, 1:-1, 1:-1] + phi[:, :-2, 1:-1]) / (dy * dy)
    )


def relative_poisson_residual(phi, ne, ni, dx, dy):
    lap = laplacian_interior(phi, dx, dy)
    source = E_CHARGE * (ni[:, 1:-1, 1:-1] - ne[:, 1:-1, 1:-1]) / EPS0
    residual = lap + source
    residual_rms = np.sqrt(np.mean(residual * residual, axis=(1, 2)))
    source_rms = np.sqrt(np.mean(source * source, axis=(1, 2)))
    return residual_rms / np.maximum(source_rms, EPS)


def electric_field(phi, dx, dy):
    dphi_dy, dphi_dx = np.gradient(phi, dy, dx, axis=(-2, -1), edge_order=1)
    return -dphi_dx, -dphi_dy


def nrmse(error_sq_sum, true_sq_sum):
    return float(np.sqrt(error_sq_sum / max(true_sq_sum, EPS)))


def load_saved(case):
    saved = WORKDIRS / case["ex_name"] / "saved"
    inputs = np.load(saved / "inputs.npy", mmap_mode="r")
    preds = np.load(saved / "preds.npy", mmap_mode="r")
    trues = np.load(saved / "trues.npy", mmap_mode="r")
    if preds.shape != trues.shape:
        raise ValueError(f"{case['key']} shape mismatch: {preds.shape} vs {trues.shape}")
    return inputs, preds, trues


def summarize_model(case, info):
    inputs, preds, trues = load_saved(case)
    idx = info["indices"]
    phi_i = idx["phi"]
    e_i = idx["electron_den"]
    ion_i = idx["ion_den"]
    dx, dy = domain_spacing(preds.shape[-2:])

    channel_sq = {channel: 0.0 for channel in CHANNELS}
    channel_count = {channel: 0 for channel in CHANNELS}
    by_horizon = []
    phi_corrs = []
    pred_rel_values = []
    true_rel_values = []
    e_sq = 0.0
    e_true_sq = 0.0

    copy_channel_sq = {channel: 0.0 for channel in CHANNELS}
    copy_channel_count = {channel: 0 for channel in CHANNELS}
    copy_e_sq = 0.0

    copy_phi = denorm(inputs[:, -1, phi_i], info, phi_i)
    copy_ex, copy_ey = electric_field(copy_phi, dx, dy)

    for t in range(preds.shape[1]):
        horizon = t + 1
        horizon_row = {"method": case["key"], "label": LABELS[case["key"]], "output_index": horizon}
        for channel in CHANNELS:
            c = idx[channel]
            diff = np.asarray(preds[:, t, c], dtype=np.float64) - np.asarray(trues[:, t, c], dtype=np.float64)
            channel_sq[channel] += float(np.sum(diff * diff))
            channel_count[channel] += int(diff.size)
            horizon_row[f"mse_{channel}"] = float(np.mean(diff * diff))

            copy_diff = np.asarray(inputs[:, -1, c], dtype=np.float64) - np.asarray(trues[:, t, c], dtype=np.float64)
            copy_channel_sq[channel] += float(np.sum(copy_diff * copy_diff))
            copy_channel_count[channel] += int(copy_diff.size)
        by_horizon.append(horizon_row)

        pred_phi = denorm(preds[:, t, phi_i], info, phi_i)
        true_phi = denorm(trues[:, t, phi_i], info, phi_i)
        pred_ne = denorm(preds[:, t, e_i], info, e_i)
        pred_ni = denorm(preds[:, t, ion_i], info, ion_i)
        true_ne = denorm(trues[:, t, e_i], info, e_i)
        true_ni = denorm(trues[:, t, ion_i], info, ion_i)

        phi_corrs.append(corr_per_frame(pred_phi, true_phi))
        pred_rel_values.append(relative_poisson_residual(pred_phi, pred_ne, pred_ni, dx, dy))
        true_rel_values.append(relative_poisson_residual(true_phi, true_ne, true_ni, dx, dy))

        pred_ex, pred_ey = electric_field(pred_phi, dx, dy)
        true_ex, true_ey = electric_field(true_phi, dx, dy)
        e_sq += float(np.sum((pred_ex - true_ex) ** 2 + (pred_ey - true_ey) ** 2))
        e_true_sq += float(np.sum(true_ex ** 2 + true_ey ** 2))
        copy_e_sq += float(np.sum((copy_ex - true_ex) ** 2 + (copy_ey - true_ey) ** 2))

    phi_corr = np.concatenate(phi_corrs)
    pred_rel = np.concatenate(pred_rel_values)
    true_rel = np.concatenate(true_rel_values)
    summary = {
        "method": case["key"],
        "label": LABELS[case["key"]],
        "n_samples": int(preds.shape[0]),
        "n_outputs": int(preds.shape[1]),
        "phi_corr_mean": float(np.mean(phi_corr)),
        "phi_corr_median": float(np.median(phi_corr)),
        "poisson_pred_rel_mean": float(np.mean(pred_rel)),
        "poisson_pred_rel_median": float(np.median(pred_rel)),
        "poisson_true_rel_mean": float(np.mean(true_rel)),
        "poisson_true_rel_median": float(np.median(true_rel)),
        "poisson_pred_over_true_median": float(np.median(pred_rel) / max(np.median(true_rel), EPS)),
        "efield_nrmse": nrmse(e_sq, e_true_sq),
        "copy_efield_nrmse": nrmse(copy_e_sq, e_true_sq),
    }
    for channel in CHANNELS:
        summary[f"mse_{channel}"] = channel_sq[channel] / channel_count[channel]
        summary[f"copy_mse_{channel}"] = copy_channel_sq[channel] / copy_channel_count[channel]
        summary[f"mse_over_copy_{channel}"] = summary[f"mse_{channel}"] / max(summary[f"copy_mse_{channel}"], EPS)
    return summary, by_horizon


def copy_horizon_rows(info):
    inputs, _, trues = load_saved(CASES[0])
    idx = info["indices"]
    rows = []
    for t in range(trues.shape[1]):
        row = {"method": "copy", "label": LABELS["copy"], "output_index": t + 1}
        for channel in CHANNELS:
            c = idx[channel]
            diff = np.asarray(inputs[:, -1, c], dtype=np.float64) - np.asarray(trues[:, t, c], dtype=np.float64)
            row[f"mse_{channel}"] = float(np.mean(diff * diff))
        rows.append(row)
    return rows


def plot_channel_mse(rows):
    methods = ["copy"] + [case["key"] for case in CASES]
    x = np.arange(len(CHANNELS), dtype=np.float64)
    width = 0.82 / len(methods)
    fig, ax = plt.subplots(figsize=(12.4, 6.4))
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width
    for offset, method in zip(offsets, methods):
        ys = []
        if method == "copy":
            source = rows[0]
            ys = [source[f"copy_mse_{channel}"] for channel in CHANNELS]
        else:
            source = next(row for row in rows if row["method"] == method)
            ys = [source[f"mse_{channel}"] for channel in CHANNELS]
        ax.bar(x + offset, ys, width=width, color=COLORS[method], label=LABELS[method])
    ax.set_xticks(x, CHANNELS)
    ax.set_ylabel("Mean MSE (normalized)")
    ax.set_title("3b direct10 test prediction: channel-wise error")
    ax.set_yscale("log")
    ax.grid(True, axis="y", which="both", linestyle=":", alpha=0.55)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = OUTDIR / "weak_regularization_3b_channel_mse_mean.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f"[PLOT] {path}", flush=True)


def plot_phi_horizon(rows):
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    for method in ["copy"] + [case["key"] for case in CASES]:
        group = [row for row in rows if row["method"] == method]
        xs = np.asarray([row["output_index"] for row in group], dtype=np.float64)
        ys = np.asarray([row["mse_phi"] for row in group], dtype=np.float64)
        ax.plot(xs, ys, marker="o", linewidth=1.7, color=COLORS[method], label=LABELS[method])
    ax.set_xlabel("Output frame index")
    ax.set_ylabel("Mean phi MSE (normalized)")
    ax.set_title("3b direct10 test prediction: phi error by horizon")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.55)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = OUTDIR / "weak_regularization_3b_phi_mse_by_horizon.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f"[PLOT] {path}", flush=True)


def plot_physics_metrics(rows):
    methods = [case["key"] for case in CASES]
    labels = [LABELS[key] for key in methods]
    x = np.arange(len(methods), dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6))
    e_vals = [next(row for row in rows if row["method"] == key)["efield_nrmse"] for key in methods]
    p_vals = [next(row for row in rows if row["method"] == key)["poisson_pred_rel_median"] for key in methods]
    axes[0].bar(x, e_vals, color=[COLORS[key] for key in methods])
    axes[0].axhline(rows[0]["copy_efield_nrmse"], color=COLORS["copy"], linestyle="--", linewidth=1.4, label="Copy baseline")
    axes[0].set_title("Electric-field NRMSE")
    axes[0].set_ylabel("NRMSE of E = -grad(phi)")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].bar(x, p_vals, color=[COLORS[key] for key in methods])
    axes[1].axhline(rows[0]["poisson_true_rel_median"], color="#111827", linestyle="--", linewidth=1.4, label="True PIC median")
    axes[1].set_title("Predicted Poisson relative residual")
    axes[1].set_ylabel("Median relative residual")
    axes[1].legend(loc="upper right", fontsize=8)

    for ax in axes:
        ax.set_xticks(x, labels, rotation=24, ha="right")
        ax.grid(True, axis="y", linestyle=":", alpha=0.55)
    fig.suptitle("3b direct10 test prediction: physics-related metrics", y=1.02)
    fig.tight_layout()
    path = OUTDIR / "weak_regularization_3b_physics_metrics.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {path}", flush=True)


def write_readme():
    text = """# 3b Weak Regularization Comparison

This folder compares stride2 direct10 models on the original high-magnet 3b test split.

Compared methods:

- `copy`: copy the last input frame.
- `data_only`: ordinary SimVPv2/gSTA trained with data MSE only.
- `poisson_zero`: weak Poisson residual regularization without a true-floor threshold.
- `floor_hinge`: previous Poisson true-floor hinge model.
- `efield`: weak electric-field regularization from `E = -grad(phi)`.
- `poisson_zero_efield`: weak Poisson residual plus weak electric-field regularization.

Main files:

- `weak_regularization_3b_summary.csv`: aggregate metrics.
- `weak_regularization_3b_horizon.csv`: output-frame-wise normalized MSE.
- `weak_regularization_3b_channel_mse_mean.png`: channel-wise MSE.
- `weak_regularization_3b_phi_mse_by_horizon.png`: phi MSE by output frame.
- `weak_regularization_3b_physics_metrics.png`: electric-field NRMSE and Poisson residual.

Japanese note:

これは転移評価ではなく、学習元と同じ high-magnet 3b の test split での比較です。
E-field loss が本当に `E=-grad(phi)` の誤差を下げるのか、Poisson系のlossがPoisson残差を下げるのかを確認するための基準結果です。
"""
    path = OUTDIR / "README.md"
    path.write_text(text, encoding="utf-8")
    print(f"[README] {path}", flush=True)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    info = load_h5_info()
    summary_rows = []
    horizon_rows = copy_horizon_rows(info)
    for case in CASES:
        summary, case_horizon = summarize_model(case, info)
        summary_rows.append(summary)
        horizon_rows.extend(case_horizon)
    write_csv(summary_rows, OUTDIR / "weak_regularization_3b_summary.csv")
    write_csv(horizon_rows, OUTDIR / "weak_regularization_3b_horizon.csv")
    plot_channel_mse(summary_rows)
    plot_phi_horizon(horizon_rows)
    plot_physics_metrics(summary_rows)
    write_readme()


if __name__ == "__main__":
    main()
