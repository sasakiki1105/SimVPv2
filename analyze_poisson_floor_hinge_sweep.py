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
OUTDIR = WORKDIRS / "compare_poisson_floor_hinge_sweep"
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
TRUE_FLOOR = 0.086
LAMBDAS = [1e-4, 1e-3, 1e-2]
ALPHAS = [1.0, 1.1, 1.2]
CHANNELS = ["electron_den", "ion_den", "phi"]


def lambda_tag(value):
    mapping = {1e-4: "lam1em4", 1e-3: "lam1em3", 1e-2: "lam1em2"}
    return mapping[value]


def alpha_tag(value):
    return f"alpha{int(round(value * 10)):02d}"


def floor_ex_name(lam, alpha):
    return (
        "pepapic_simvp_gsta_highmag_macro5_subsample2_direct10_"
        f"poisson_floor_hinge_{lambda_tag(lam)}_floor086_{alpha_tag(alpha)}_"
        "trainfixed_disjoint_811_bs2_100ep"
    )


def cases():
    out = [
        {
            "case": "baseline",
            "lambda": np.nan,
            "alpha": np.nan,
            "loss": "data_mse_only",
            "ex_name": "pepapic_simvp_gsta_highmag_macro5_subsample2_trainfixed_disjoint_811_bs2_100ep",
        }
    ]
    for lam in LAMBDAS:
        for alpha in ALPHAS:
            out.append(
                {
                    "case": f"floor_hinge_{lambda_tag(lam)}_{alpha_tag(alpha)}",
                    "lambda": lam,
                    "alpha": alpha,
                    "loss": "poisson_floor_hinge",
                    "ex_name": floor_ex_name(lam, alpha),
                }
            )
    return out


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
        norm_mode = f["norm_mode"][()]
        if isinstance(norm_mode, (bytes, bytearray)):
            norm_mode = norm_mode.decode()
        else:
            norm_mode = str(norm_mode)
    if "minmax" not in norm_mode:
        raise ValueError(f"Unsupported norm mode: {norm_mode}")
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
    dx = lx / float(w - 1) if w > 1 else 1.0
    dy = ly / float(h - 1) if h > 1 else 1.0
    return dx, dy


def denorm(x, info, channel_index):
    mn = float(info["train_min"][channel_index])
    mx = float(info["train_max"][channel_index])
    value_range = mx - mn
    lo = mn - info["margin"] * value_range
    hi = mx + info["margin"] * value_range
    return x.astype(np.float64) * (hi - lo) + lo


def corr_per_frame(a, b):
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    a0 = a - np.mean(a, axis=1, keepdims=True)
    b0 = b - np.mean(b, axis=1, keepdims=True)
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


def nrmse(numer_sq_sum, denom_sq_sum):
    return float(np.sqrt(numer_sq_sum / max(denom_sq_sum, EPS)))


def summarize_case(case, info):
    saved = WORKDIRS / case["ex_name"] / "saved"
    inputs_path = saved / "inputs.npy"
    preds_path = saved / "preds.npy"
    trues_path = saved / "trues.npy"
    if not preds_path.exists() or not trues_path.exists():
        raise FileNotFoundError(f"Missing saved arrays for {case['case']}: {saved}")

    preds = np.load(preds_path, mmap_mode="r")
    trues = np.load(trues_path, mmap_mode="r")
    inputs = np.load(inputs_path, mmap_mode="r") if inputs_path.exists() else None
    if preds.shape != trues.shape:
        raise ValueError(f"{case['case']} shape mismatch: {preds.shape} vs {trues.shape}")

    dx, dy = domain_spacing(preds.shape[-2:])
    idx = info["indices"]
    phi_i = idx["phi"]
    e_i = idx["electron_den"]
    ion_i = idx["ion_den"]

    total_sq = 0.0
    total_count = 0
    channel_sq = {channel: 0.0 for channel in CHANNELS}
    channel_count = {channel: 0 for channel in CHANNELS}
    phi_corr_values = []
    pred_rel_values = []
    true_rel_values = []
    hinge_values = []
    ex_sq = ey_sq = emag_sq = 0.0
    ex_true_sq = ey_true_sq = emag_true_sq = 0.0

    copy_phi_sq = 0.0
    copy_phi_count = 0
    copy_ex_sq = copy_ey_sq = copy_emag_sq = 0.0

    if inputs is not None:
        copy_phi = denorm(inputs[:, -1, phi_i], info, phi_i)
        copy_ex, copy_ey = electric_field(copy_phi, dx, dy)
    else:
        copy_phi = copy_ex = copy_ey = None

    for t in range(preds.shape[1]):
        diff = np.asarray(preds[:, t], dtype=np.float64) - np.asarray(trues[:, t], dtype=np.float64)
        total_sq += float(np.sum(diff * diff))
        total_count += int(diff.size)
        for channel in CHANNELS:
            c = idx[channel]
            cdiff = diff[:, c]
            channel_sq[channel] += float(np.sum(cdiff * cdiff))
            channel_count[channel] += int(cdiff.size)

        pred_phi = denorm(preds[:, t, phi_i], info, phi_i)
        true_phi = denorm(trues[:, t, phi_i], info, phi_i)
        pred_ne = denorm(preds[:, t, e_i], info, e_i)
        true_ne = denorm(trues[:, t, e_i], info, e_i)
        pred_ni = denorm(preds[:, t, ion_i], info, ion_i)
        true_ni = denorm(trues[:, t, ion_i], info, ion_i)

        phi_corr_values.append(corr_per_frame(pred_phi, true_phi))

        pred_rel = relative_poisson_residual(pred_phi, pred_ne, pred_ni, dx, dy)
        true_rel = relative_poisson_residual(true_phi, true_ne, true_ni, dx, dy)
        pred_rel_values.append(pred_rel)
        true_rel_values.append(true_rel)
        if not np.isnan(case["alpha"]):
            hinge_values.append(np.maximum(0.0, pred_rel - TRUE_FLOOR * float(case["alpha"])))

        pred_ex, pred_ey = electric_field(pred_phi, dx, dy)
        true_ex, true_ey = electric_field(true_phi, dx, dy)
        ex_diff = pred_ex - true_ex
        ey_diff = pred_ey - true_ey
        ex_sq += float(np.sum(ex_diff * ex_diff))
        ey_sq += float(np.sum(ey_diff * ey_diff))
        emag_sq += float(np.sum(ex_diff * ex_diff + ey_diff * ey_diff))
        ex_true_sq += float(np.sum(true_ex * true_ex))
        ey_true_sq += float(np.sum(true_ey * true_ey))
        emag_true_sq += float(np.sum(true_ex * true_ex + true_ey * true_ey))

        if copy_phi is not None:
            cphi_diff = copy_phi - true_phi
            copy_phi_sq += float(np.sum(cphi_diff * cphi_diff))
            copy_phi_count += int(cphi_diff.size)
            cex_diff = copy_ex - true_ex
            cey_diff = copy_ey - true_ey
            copy_ex_sq += float(np.sum(cex_diff * cex_diff))
            copy_ey_sq += float(np.sum(cey_diff * cey_diff))
            copy_emag_sq += float(np.sum(cex_diff * cex_diff + cey_diff * cey_diff))

    phi_corr = np.concatenate(phi_corr_values)
    pred_rel_all = np.concatenate(pred_rel_values)
    true_rel_all = np.concatenate(true_rel_values)
    if hinge_values:
        hinge_all = np.concatenate(hinge_values)
        hinge_mean = float(np.mean(hinge_all))
        hinge_sq_mean = float(np.mean(hinge_all * hinge_all))
    else:
        hinge_mean = np.nan
        hinge_sq_mean = np.nan

    row = {
        **case,
        "samples": int(preds.shape[0]),
        "out_frames": int(preds.shape[1]),
        "norm_mse_all": total_sq / total_count,
        "norm_mse_electron_den": channel_sq["electron_den"] / channel_count["electron_den"],
        "norm_mse_ion_den": channel_sq["ion_den"] / channel_count["ion_den"],
        "norm_mse_phi": channel_sq["phi"] / channel_count["phi"],
        "copy_norm_mse_phi": copy_phi_sq / copy_phi_count if copy_phi_count else np.nan,
        "model_over_copy_phi": (channel_sq["phi"] / channel_count["phi"]) / (copy_phi_sq / copy_phi_count)
        if copy_phi_count
        else np.nan,
        "phi_corr_mean": float(np.mean(phi_corr)),
        "phi_corr_median": float(np.median(phi_corr)),
        "phi_corr_q25": float(np.quantile(phi_corr, 0.25)),
        "phi_corr_q75": float(np.quantile(phi_corr, 0.75)),
        "poisson_pred_rel_mean": float(np.mean(pred_rel_all)),
        "poisson_pred_rel_median": float(np.median(pred_rel_all)),
        "poisson_true_rel_mean": float(np.mean(true_rel_all)),
        "poisson_true_rel_median": float(np.median(true_rel_all)),
        "poisson_pred_over_true_median": float(np.median(pred_rel_all) / np.median(true_rel_all)),
        "poisson_hinge_excess_mean": hinge_mean,
        "poisson_hinge_excess_sq_mean": hinge_sq_mean,
        "ex_nrmse": nrmse(ex_sq, ex_true_sq),
        "ey_nrmse": nrmse(ey_sq, ey_true_sq),
        "emag_nrmse": nrmse(emag_sq, emag_true_sq),
        "copy_ex_nrmse": nrmse(copy_ex_sq, ex_true_sq) if copy_phi_count else np.nan,
        "copy_ey_nrmse": nrmse(copy_ey_sq, ey_true_sq) if copy_phi_count else np.nan,
        "copy_emag_nrmse": nrmse(copy_emag_sq, emag_true_sq) if copy_phi_count else np.nan,
    }
    row["emag_over_copy"] = row["emag_nrmse"] / row["copy_emag_nrmse"]
    return row


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def heatmap(rows, metric, title, path, cmap="viridis", fmt=".3g", lower_is_better=True):
    matrix = np.full((len(LAMBDAS), len(ALPHAS)), np.nan, dtype=np.float64)
    for row in rows:
        if row["loss"] != "poisson_floor_hinge":
            continue
        i = LAMBDAS.index(float(row["lambda"]))
        j = ALPHAS.index(float(row["alpha"]))
        matrix[i, j] = float(row[metric])

    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(ALPHAS)), [str(a) for a in ALPHAS])
    ax.set_yticks(range(len(LAMBDAS)), [f"{lam:.0e}" for lam in LAMBDAS])
    ax.set_xlabel("alpha")
    ax.set_ylabel("lambda")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            ax.text(j, i, format(value, fmt), ha="center", va="center", color="white", fontsize=9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)
    if lower_is_better:
        best = np.nanmin(matrix)
        best_pos = np.where(matrix == best)
    else:
        best = np.nanmax(matrix)
        best_pos = np.where(matrix == best)
    if len(best_pos[0]):
        ax.scatter(best_pos[1][0], best_pos[0][0], marker="s", s=420, facecolors="none", edgecolors="red", linewidths=2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def bar_compare(rows):
    baseline = next(row for row in rows if row["case"] == "baseline")
    best_phi = min((row for row in rows if row["loss"] == "poisson_floor_hinge"), key=lambda r: r["norm_mse_phi"])
    best_poisson = min((row for row in rows if row["loss"] == "poisson_floor_hinge"), key=lambda r: r["poisson_pred_rel_median"])
    labels = ["baseline", f"best phi\n{best_phi['case']}", f"best poisson\n{best_poisson['case']}"]
    selected = [baseline, best_phi, best_poisson]
    metrics = ["norm_mse_phi", "phi_corr_median", "poisson_pred_rel_median", "emag_nrmse"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, metric in zip(axes.ravel(), metrics):
        values = [row[metric] for row in selected]
        ax.bar(labels, values, color=["#777777", "#1f77b4", "#2ca02c"])
        ax.set_title(metric)
        ax.tick_params(axis="x", labelrotation=15)
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.3g}", ha="center", va="bottom", fontsize=8)
    fig.savefig(OUTDIR / "poisson_floor_hinge_sweep_best_vs_baseline.png", dpi=180)
    plt.close(fig)


def write_readme(rows):
    baseline = next(row for row in rows if row["case"] == "baseline")
    best_phi = min((row for row in rows if row["loss"] == "poisson_floor_hinge"), key=lambda r: r["norm_mse_phi"])
    best_corr = max((row for row in rows if row["loss"] == "poisson_floor_hinge"), key=lambda r: r["phi_corr_median"])
    best_poisson = min((row for row in rows if row["loss"] == "poisson_floor_hinge"), key=lambda r: r["poisson_pred_rel_median"])
    best_e = min((row for row in rows if row["loss"] == "poisson_floor_hinge"), key=lambda r: r["emag_nrmse"])
    text = f"""# Poisson floor-hinge sweep

This folder compares stride2 direct10 high-magnet testcase 3b models trained with:

`L = L_data + lambda * mean(max(0, relative_residual_pred - alpha * 0.086)^2)`

The baseline row is the original data-MSE-only stride2 direct10 model.

## Key result

- baseline phi MSE: `{baseline['norm_mse_phi']:.8g}`
- best phi MSE: `{best_phi['norm_mse_phi']:.8g}` at `{best_phi['case']}`
- best phi median correlation: `{best_corr['phi_corr_median']:.8g}` at `{best_corr['case']}`
- best predicted Poisson relative residual median: `{best_poisson['poisson_pred_rel_median']:.8g}` at `{best_poisson['case']}`
- best electric-field magnitude NRMSE: `{best_e['emag_nrmse']:.8g}` at `{best_e['case']}`

## Files

- `poisson_floor_hinge_sweep_summary.csv`: all aggregate metrics.
- `poisson_floor_hinge_sweep_ranked_phi_mse.csv`: floor-hinge cases sorted by phi MSE.
- `poisson_floor_hinge_sweep_phi_mse_heatmap.png`: phi MSE over lambda/alpha.
- `poisson_floor_hinge_sweep_phi_corr_heatmap.png`: phi median correlation over lambda/alpha.
- `poisson_floor_hinge_sweep_poisson_pred_rel_heatmap.png`: predicted relative Poisson residual over lambda/alpha.
- `poisson_floor_hinge_sweep_emag_nrmse_heatmap.png`: electric-field magnitude NRMSE over lambda/alpha.
- `poisson_floor_hinge_sweep_best_vs_baseline.png`: compact baseline/best comparison.
"""
    (OUTDIR / "README.md").write_text(text, encoding="utf-8")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    info = load_h5_info()
    rows = []
    for case in cases():
        print(f"analyzing {case['case']}...", flush=True)
        rows.append(summarize_case(case, info))

    write_csv(OUTDIR / "poisson_floor_hinge_sweep_summary.csv", rows)
    ranked_phi = sorted(
        [row for row in rows if row["loss"] == "poisson_floor_hinge"],
        key=lambda row: row["norm_mse_phi"],
    )
    write_csv(OUTDIR / "poisson_floor_hinge_sweep_ranked_phi_mse.csv", ranked_phi)
    ranked_poisson = sorted(
        [row for row in rows if row["loss"] == "poisson_floor_hinge"],
        key=lambda row: row["poisson_pred_rel_median"],
    )
    write_csv(OUTDIR / "poisson_floor_hinge_sweep_ranked_poisson_residual.csv", ranked_poisson)

    heatmap(rows, "norm_mse_phi", "phi normalized MSE (lower is better)", OUTDIR / "poisson_floor_hinge_sweep_phi_mse_heatmap.png")
    heatmap(
        rows,
        "phi_corr_median",
        "phi median spatial correlation (higher is better)",
        OUTDIR / "poisson_floor_hinge_sweep_phi_corr_heatmap.png",
        cmap="magma",
        lower_is_better=False,
    )
    heatmap(
        rows,
        "poisson_pred_rel_median",
        "predicted Poisson relative residual median",
        OUTDIR / "poisson_floor_hinge_sweep_poisson_pred_rel_heatmap.png",
    )
    heatmap(rows, "emag_nrmse", "electric-field magnitude NRMSE", OUTDIR / "poisson_floor_hinge_sweep_emag_nrmse_heatmap.png")
    bar_compare(rows)
    write_readme(rows)
    print(f"saved: {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
