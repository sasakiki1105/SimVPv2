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
OUTDIR = WORKDIRS / "compare_poisson_residual_consistency"

EPS0 = 8.8541878128e-12
E_CHARGE = 1.602176634e-19
EPS = 1e-30


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

CASE_STYLE = {
    "stride1": {"marker": "o", "linestyle": "-", "zorder": 4},
    "stride2": {"marker": "s", "linestyle": "--", "zorder": 3},
    "stride3": {"marker": "^", "linestyle": "-.", "zorder": 2},
    "stride4": {"marker": "D", "linestyle": ":", "zorder": 1},
}


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


def build_test_starts(t_len):
    total = PRE + AFT
    val_end = int(np.floor(t_len * 0.9))
    last_start = t_len - total
    if last_start < val_end:
        return np.empty((0,), dtype=np.int64)
    return np.arange(val_end, last_start + 1, dtype=np.int64)


def laplacian_interior(phi, dx, dy):
    return (
        (phi[:, 1:-1, 2:] - 2.0 * phi[:, 1:-1, 1:-1] + phi[:, 1:-1, :-2]) / (dx * dx)
        + (phi[:, 2:, 1:-1] - 2.0 * phi[:, 1:-1, 1:-1] + phi[:, :-2, 1:-1]) / (dy * dy)
    )


def centered_corr(a, b):
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    a0 = a - np.mean(a, axis=1, keepdims=True)
    b0 = b - np.mean(b, axis=1, keepdims=True)
    denom = np.sqrt(np.sum(a0 * a0, axis=1) * np.sum(b0 * b0, axis=1))
    return np.sum(a0 * b0, axis=1) / np.maximum(denom, EPS)


def residual_metrics(phi, ne, ni, dx, dy):
    lap = laplacian_interior(phi, dx, dy)
    source = E_CHARGE * (ni[:, 1:-1, 1:-1] - ne[:, 1:-1, 1:-1]) / EPS0
    rhs = -source
    residual = lap + source

    residual_rms = np.sqrt(np.mean(residual * residual, axis=(1, 2)))
    lap_rms = np.sqrt(np.mean(lap * lap, axis=(1, 2)))
    source_rms = np.sqrt(np.mean(source * source, axis=(1, 2)))
    rhs_rms = np.sqrt(np.mean(rhs * rhs, axis=(1, 2)))
    balance_rmse = np.sqrt(np.mean((lap - rhs) * (lap - rhs), axis=(1, 2)))
    balance_nrmse = balance_rmse / np.maximum(rhs_rms, EPS)
    relative_residual = residual_rms / np.maximum(source_rms, EPS)
    balance_corr = centered_corr(lap, rhs)
    return {
        "residual_rms": residual_rms,
        "relative_residual": relative_residual,
        "balance_nrmse": balance_nrmse,
        "balance_corr": balance_corr,
        "laplacian_rms": lap_rms,
        "source_rms": source_rms,
    }


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
    return {
        **case,
        "preds": preds,
        "trues": trues,
        "info": info,
        "starts": starts,
        "dt_ns": BASE_DT_NS * info["stride_steps"],
        "dx": dx,
        "dy": dy,
    }


def compute_case_rows(case):
    info = case["info"]
    e_i = info["electron_index"]
    ion_i = info["ion_index"]
    phi_i = info["phi_index"]

    raw_rows = []
    aggregate_values = defaultdict(lambda: defaultdict(list))

    for tp in range(AFT):
        target_indices = case["starts"] + PRE + tp
        target_timesteps = info["timesteps"][target_indices]
        horizon_ns = (tp + 1) * case["dt_ns"]

        pred_phi = denorm_channel(case["preds"][:, tp, phi_i], info, phi_i)
        pred_ne = denorm_channel(case["preds"][:, tp, e_i], info, e_i)
        pred_ni = denorm_channel(case["preds"][:, tp, ion_i], info, ion_i)
        true_phi = denorm_channel(case["trues"][:, tp, phi_i], info, phi_i)
        true_ne = denorm_channel(case["trues"][:, tp, e_i], info, e_i)
        true_ni = denorm_channel(case["trues"][:, tp, ion_i], info, ion_i)

        metrics_by_source = {
            "pred": residual_metrics(pred_phi, pred_ne, pred_ni, case["dx"], case["dy"]),
            "true": residual_metrics(true_phi, true_ne, true_ni, case["dx"], case["dy"]),
        }

        pred_rel = metrics_by_source["pred"]["relative_residual"]
        true_rel = metrics_by_source["true"]["relative_residual"]
        pred_over_true_rel = pred_rel / np.maximum(true_rel, EPS)

        for sample_idx, target_timestep in enumerate(target_timesteps):
            target_timestep = int(target_timestep)
            target_time_us = target_timestep * BASE_DT_NS / 1000.0
            for source, metrics in metrics_by_source.items():
                row = {
                    "case": case["name"],
                    "source": source,
                    "sample_index": int(sample_idx),
                    "output_index": int(tp + 1),
                    "horizon_ns": float(horizon_ns),
                    "target_timestep": target_timestep,
                    "target_time_us": float(target_time_us),
                }
                for key, values in metrics.items():
                    value = float(values[sample_idx])
                    row[key] = value
                    aggregate_values[(source, target_timestep)][key].append(value)
                    aggregate_values[(source, f"horizon_{tp + 1}")][key].append(value)
                if source == "pred":
                    ratio_value = float(pred_over_true_rel[sample_idx])
                    row["pred_over_true_relative_residual"] = ratio_value
                    aggregate_values[("pred_over_true", target_timestep)][
                        "relative_residual_ratio"
                    ].append(ratio_value)
                    aggregate_values[("pred_over_true", f"horizon_{tp + 1}")][
                        "relative_residual_ratio"
                    ].append(ratio_value)
                raw_rows.append(row)

    target_rows = []
    horizon_rows = []
    for (source, key), metric_values in aggregate_values.items():
        out = {"case": case["name"], "source": source}
        for metric_name, values in metric_values.items():
            stats = summarize(values)
            for stat_name, stat_value in stats.items():
                out[f"{metric_name}_{stat_name}"] = stat_value
        if isinstance(key, str) and key.startswith("horizon_"):
            output_index = int(key.split("_", 1)[1])
            out["output_index"] = output_index
            out["horizon_ns"] = float(output_index * case["dt_ns"])
            horizon_rows.append(out)
        else:
            target_timestep = int(key)
            out["target_timestep"] = target_timestep
            out["target_time_us"] = target_timestep * BASE_DT_NS / 1000.0
            target_rows.append(out)
    return raw_rows, target_rows, horizon_rows


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def common_target_rows(target_rows, metric_name, statistic, source):
    by_case = defaultdict(dict)
    for row in target_rows:
        if row["source"] != source:
            continue
        value_key = f"{metric_name}_{statistic}"
        if value_key not in row:
            continue
        by_case[row["case"]][int(row["target_timestep"])] = row
    if not by_case:
        return []
    common = set.intersection(*(set(rows.keys()) for rows in by_case.values()))
    out = []
    for case in sorted(by_case):
        for timestep in sorted(common):
            out.append(by_case[case][timestep])
    return out


def finish_axes(ax, x_values, y_values, logy=False, legend=True):
    if x_values:
        xmin = min(x_values)
        xmax = max(x_values)
        span = xmax - xmin
        if span <= 0:
            span = max(abs(xmax), 1.0)
        ax.set_xlim(xmin, xmax + 0.22 * span)
    if y_values:
        y = np.asarray(y_values, dtype=np.float64)
        y = y[np.isfinite(y)]
        if len(y):
            ymin = float(np.min(y))
            ymax = float(np.max(y))
            if logy:
                positive = y[y > 0]
                if len(positive):
                    ymin = float(np.min(positive))
                    ymax = float(np.max(positive))
                    ax.set_ylim(ymin / 1.35, ymax * 1.25)
            else:
                span = ymax - ymin
                if span <= 0:
                    span = max(abs(ymax), 1.0)
                lower = ymin - 0.14 * span
                upper = ymax + 0.18 * span
                if ymin >= 0 and lower < 0:
                    lower = 0.0
                ax.set_ylim(lower, upper)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    if legend:
        ax.legend(loc="lower right", framealpha=0.92, facecolor="white", edgecolor="0.75")


def plot_metric_by_time(
    rows,
    metric_name,
    statistic,
    source,
    ylabel,
    out_png,
    logy=False,
    title=None,
    x_jitter=False,
):
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    all_x = []
    all_y = []
    for case_index, case in enumerate(CASES):
        case_rows = [r for r in rows if r["case"] == case["name"] and r["source"] == source]
        case_rows.sort(key=lambda r: r["target_time_us"])
        if not case_rows:
            continue
        x = np.asarray([r["target_time_us"] for r in case_rows], dtype=np.float64)
        y = np.asarray([r[f"{metric_name}_{statistic}"] for r in case_rows], dtype=np.float64)
        if x_jitter:
            x = x + (case_index - (len(CASES) - 1) / 2.0) * 0.035
        style = CASE_STYLE[case["name"]]
        ax.plot(
            x,
            y,
            label=case["label"],
            color=case["color"],
            linewidth=2.0,
            marker=style["marker"],
            markersize=3.5,
            markevery=max(len(x) // 12, 1),
            linestyle=style["linestyle"],
            zorder=style["zorder"],
        )
        all_x.extend(x.tolist())
        all_y.extend(y.tolist())
    ax.set_xlabel("target time [us]")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    finish_axes(ax, all_x, all_y, logy=logy)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_metric_by_horizon(
    rows,
    metric_name,
    statistic,
    source,
    ylabel,
    out_png,
    logy=False,
    title=None,
    include_true_floor=False,
    reference_y=None,
    reference_label=None,
):
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    all_x = []
    all_y = []
    for case in CASES:
        case_rows = [r for r in rows if r["case"] == case["name"] and r["source"] == source]
        case_rows.sort(key=lambda r: r["horizon_ns"])
        if not case_rows:
            continue
        x = np.asarray([r["horizon_ns"] for r in case_rows], dtype=np.float64)
        y = np.asarray([r[f"{metric_name}_{statistic}"] for r in case_rows], dtype=np.float64)
        style = CASE_STYLE[case["name"]]
        ax.plot(
            x,
            y,
            marker=style["marker"],
            linestyle=style["linestyle"],
            label=case["label"],
            color=case["color"],
            linewidth=2.0,
            markersize=4.5,
            zorder=style["zorder"],
        )
        all_x.extend(x.tolist())
        all_y.extend(y.tolist())
        if include_true_floor:
            true_rows = [
                r for r in rows if r["case"] == case["name"] and r["source"] == "true"
            ]
            true_rows.sort(key=lambda r: r["horizon_ns"])
            if true_rows:
                xt = np.asarray([r["horizon_ns"] for r in true_rows], dtype=np.float64)
                yt = np.asarray(
                    [r[f"{metric_name}_{statistic}"] for r in true_rows],
                    dtype=np.float64,
                )
                ax.plot(
                    xt,
                    yt,
                    color=case["color"],
                    linewidth=1.0,
                    alpha=0.45,
                    linestyle=":",
                    label=f"{case['label']} true floor",
                )
                all_x.extend(xt.tolist())
                all_y.extend(yt.tolist())
    ax.set_xlabel("forecast horizon [ns]")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if reference_y is not None:
        ax.axhline(
            reference_y,
            color="0.2",
            linestyle="--",
            linewidth=1.2,
            label=reference_label or f"reference {reference_y:g}",
        )
        all_y.append(float(reference_y))
    finish_axes(ax, all_x, all_y, logy=logy)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_reference_line_by_time(rows, out_png, linear=False):
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    all_x = []
    all_y = []
    for case_index, case in enumerate(CASES):
        case_rows = [r for r in rows if r["case"] == case["name"] and r["source"] == "true"]
        case_rows.sort(key=lambda r: r["target_time_us"])
        if not case_rows:
            continue
        x = np.asarray([r["target_time_us"] for r in case_rows], dtype=np.float64)
        y = np.asarray([r["relative_residual_median"] for r in case_rows], dtype=np.float64)
        # The true curves are almost the same PIC frames. Offset x slightly so
        # the viewer can see that all strides are present.
        x = x + (case_index - (len(CASES) - 1) / 2.0) * 0.04
        style = CASE_STYLE[case["name"]]
        ax.plot(
            x,
            y,
            label=case["label"],
            color=case["color"],
            linewidth=1.9,
            marker=style["marker"],
            markersize=3.5,
            markevery=max(len(x) // 12, 1),
            linestyle=style["linestyle"],
            zorder=style["zorder"],
        )
        all_x.extend(x.tolist())
        all_y.extend(y.tolist())
    ax.set_xlabel("target time [us]")
    ax.set_ylabel("median relative Poisson residual (true diagnostic floor)")
    ax.set_title("True-frame diagnostic floor (x-offsets reveal overlapping stride curves)")
    finish_axes(ax, all_x, all_y, logy=not linear)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_ratio_bar(horizon_rows, out_png):
    labels = []
    medians = []
    q25 = []
    q75 = []
    colors = []
    for case in CASES:
        rows = [
            r
            for r in horizon_rows
            if r["case"] == case["name"] and r["source"] == "pred_over_true"
        ]
        values = np.asarray([r["relative_residual_ratio_median"] for r in rows], dtype=np.float64)
        labels.append(case["name"])
        medians.append(float(np.median(values)))
        q25.append(float(np.quantile(values, 0.25)))
        q75.append(float(np.quantile(values, 0.75)))
        colors.append(case["color"])
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    x = np.arange(len(labels))
    yerr = np.vstack([np.asarray(medians) - np.asarray(q25), np.asarray(q75) - np.asarray(medians)])
    ax.bar(x, medians, yerr=yerr, capsize=5, color=colors, edgecolor="0.2", alpha=0.86)
    ax.axhline(1.0, color="0.2", linestyle="--", linewidth=1.4, label="true-frame floor")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("median pred residual / true residual")
    ax.set_title("Poisson consistency relative to true-frame diagnostic floor")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.92, facecolor="white", edgecolor="0.75")
    ymax = max(max(medians), 1.0)
    ax.set_ylim(0.0, ymax * 1.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_ratio_heatmap(horizon_rows, out_png):
    matrix = []
    ylabels = []
    for case in CASES:
        rows = [
            r
            for r in horizon_rows
            if r["case"] == case["name"] and r["source"] == "pred_over_true"
        ]
        rows.sort(key=lambda r: r["output_index"])
        matrix.append([r["relative_residual_ratio_median"] for r in rows])
        ylabels.append(case["name"])
    data = np.asarray(matrix, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    im = ax.imshow(data, aspect="auto", cmap="coolwarm", vmin=0.6, vmax=1.4)
    ax.set_xticks(np.arange(AFT))
    ax.set_xticklabels([str(i) for i in range(1, AFT + 1)])
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("output index")
    ax.set_ylabel("stride")
    ax.set_title("Pred/true Poisson residual ratio by output frame")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("pred residual / true residual")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_excess_by_time(rows, out_png):
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    all_x = []
    all_y = []
    for case in CASES:
        case_rows = [
            r for r in rows if r["case"] == case["name"] and r["source"] == "pred_over_true"
        ]
        case_rows.sort(key=lambda r: r["target_time_us"])
        if not case_rows:
            continue
        x = np.asarray([r["target_time_us"] for r in case_rows], dtype=np.float64)
        ratio = np.asarray(
            [r["relative_residual_ratio_median"] for r in case_rows], dtype=np.float64
        )
        y = ratio - 1.0
        style = CASE_STYLE[case["name"]]
        ax.plot(
            x,
            y,
            label=case["label"],
            color=case["color"],
            linewidth=2.0,
            marker=style["marker"],
            markersize=3.5,
            markevery=max(len(x) // 12, 1),
            linestyle=style["linestyle"],
            zorder=style["zorder"],
        )
        all_x.extend(x.tolist())
        all_y.extend(y.tolist())
    ax.axhline(0.0, color="0.2", linewidth=1.2, linestyle="--", label="same as true floor")
    ax.set_xlabel("target time [us]")
    ax.set_ylabel("excess residual ratio (pred/true - 1)")
    ax.set_title("Excess Poisson inconsistency above true-frame floor")
    finish_axes(ax, all_x, all_y, logy=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def write_summary(cases, target_rows, horizon_rows):
    summary_cases = []
    for case in cases:
        case_summary = {
            "name": case["name"],
            "label": case["label"],
            "dt_ns": case["dt_ns"],
            "dx": case["dx"],
            "dy": case["dy"],
            "preds_shape": list(case["preds"].shape),
            "sources": {},
        }
        for source in ("pred", "true", "pred_over_true"):
            rows = [r for r in horizon_rows if r["case"] == case["name"] and r["source"] == source]
            case_summary["sources"][source] = {}
            for metric in (
                "relative_residual",
                "balance_nrmse",
                "balance_corr",
                "residual_rms",
                "source_rms",
                "relative_residual_ratio",
            ):
                values = []
                for row in rows:
                    key = f"{metric}_median"
                    if key in row:
                        values.append(row[key])
                if values:
                    case_summary["sources"][source][f"horizon_median_of_{metric}_median"] = float(
                        np.median(values)
                    )
        summary_cases.append(case_summary)

    summary = {
        "description": (
            "Poisson residual diagnostic for existing SimVP/gSTA predictions. "
            "R = laplacian(phi) + e*(ion_den - electron_den)/eps0 is evaluated on "
            "interior cells using centered finite differences. Metrics are computed "
            "for predicted frames and true frames with the same diagnostic."
        ),
        "poisson_equation": "laplacian(phi) = -e*(ion_den - electron_den)/eps0",
        "residual": "R = laplacian(phi) + e*(ion_den - electron_den)/eps0",
        "normalization": "relative_residual = rms(R) / rms(e*(ion_den-electron_den)/eps0)",
        "domain_info": str(DOMAIN_INFO),
        "base_dt_ns": BASE_DT_NS,
        "cases": summary_cases,
        "caution": (
            "This is a finite-difference consistency diagnostic, not necessarily the "
            "exact PIC Poisson solver. True-frame residuals should be treated as the "
            "diagnostic floor caused by boundary conditions, staggering, averaging, "
            "and grid/operator mismatch."
        ),
    }
    path = OUTDIR / "poisson_residual_consistency_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_readme():
    text = """# Experiment

Evaluate Poisson residuals on existing SimVP/gSTA predictions.

# Meaning

For each predicted and true output frame, this computes

```text
R = laplacian(phi) + e * (ion_den - electron_den) / eps0
```

on interior grid cells. The main diagnostic is

```text
relative_residual = RMS(R) / RMS(e * (ion_den - electron_den) / eps0)
```

The same residual is also computed for the true PIC output frames. The true-frame
value is useful as a diagnostic floor because this script uses a simple centered
finite-difference operator and may not match the exact PIC Poisson solve,
boundary treatment, ghost cells, staggering, or averaging.

# Key Files

- `poisson_residual_raw.csv`: per-sample, per-output residual metrics.
- `poisson_residual_by_target_time.csv`: aggregated by target physical time.
- `poisson_residual_by_horizon.csv`: aggregated by output index / forecast horizon.
- `poisson_residual_consistency_summary.json`: compact summary.
- `poisson_residual_relative_pred_common_times.png`: predicted-frame residual over common physical times, log scale.
- `poisson_residual_relative_pred_common_times_linear.png`: same as above, linear scale.
- `poisson_residual_relative_true_common_times.png`: true-frame diagnostic floor over common physical times, log scale. X positions are slightly offset only to reveal overlapping stride curves.
- `poisson_residual_relative_true_common_times_linear.png`: same as above, linear scale.
- `poisson_residual_pred_over_true_common_times.png`: predicted relative residual divided by true-frame relative residual.
- `poisson_residual_excess_over_true_common_times.png`: `pred/true - 1`, so zero means equal to the true-frame diagnostic floor.
- `poisson_balance_corr_pred_common_times.png`: correlation between `laplacian(phi_pred)` and `-rho_pred/eps0`.
- `poisson_residual_relative_pred_by_horizon.png`: predicted residual by output horizon, with true floors shown as dotted lines.
- `poisson_residual_relative_pred_by_horizon_linear.png`: same as above, linear scale.
- `poisson_residual_pred_over_true_by_horizon.png`: predicted/true residual ratio by output horizon with a reference line at 1.
- `poisson_residual_pred_over_true_summary_bar.png`: compact per-stride summary. Below 1 is not necessarily better; it can also mean model smoothing.
- `poisson_residual_pred_over_true_horizon_heatmap.png`: output-index heatmap of predicted/true residual ratio.

# 日本語訳

このフォルダは、既存の SimVP/gSTA の予測結果に対して Poisson 方程式の残差を計算したものです。

目的は、モデルの出力した `phi`, `electron_den`, `ion_den` が

```text
laplacian(phi) = -e * (ion_den - electron_den) / eps0
```

とどの程度整合しているかを見ることです。

ただし、この計算は中央差分で interior cell のみを見た簡易診断です。PIC 本体の Poisson solver、境界条件、ゴーストセル、格子配置、時間平均とは完全に一致しない可能性があります。そのため、真値フレームにも同じ残差を計算し、真値側の残差をこの診断の基準線として扱います。

PINN 的な Poisson loss を入れる前の下調べとして、まず `pred` の残差が `true` の残差よりどれくらい大きいかを見るための結果です。

図の凡例は右下に統一しています。`true` 側の線は同じ PIC 真値フレームを見ているためほぼ重なります。そのため true-frame floor の図だけは、表示上の見やすさのために x 方向へごく小さくずらしています。y の値は変えていません。
"""
    (OUTDIR / "README.md").write_text(text, encoding="utf-8")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    loaded_cases = [load_case(case) for case in CASES]
    raw_rows = []
    target_rows = []
    horizon_rows = []
    for case in loaded_cases:
        print(f"computing {case['name']} {case['preds'].shape}")
        raw, target, horizon = compute_case_rows(case)
        raw_rows.extend(raw)
        target_rows.extend(target)
        horizon_rows.extend(horizon)

    common_pred = common_target_rows(target_rows, "relative_residual", "median", "pred")
    common_true = common_target_rows(target_rows, "relative_residual", "median", "true")
    common_ratio = common_target_rows(
        target_rows, "relative_residual_ratio", "median", "pred_over_true"
    )
    common_corr = common_target_rows(target_rows, "balance_corr", "median", "pred")

    write_csv(OUTDIR / "poisson_residual_raw.csv", raw_rows)
    write_csv(OUTDIR / "poisson_residual_by_target_time.csv", target_rows)
    write_csv(OUTDIR / "poisson_residual_by_horizon.csv", horizon_rows)
    write_csv(OUTDIR / "poisson_residual_common_times_pred.csv", common_pred)
    write_csv(OUTDIR / "poisson_residual_common_times_true.csv", common_true)
    write_csv(OUTDIR / "poisson_residual_common_times_pred_over_true.csv", common_ratio)

    plot_metric_by_time(
        common_pred,
        "relative_residual",
        "median",
        "pred",
        "median relative Poisson residual (pred)",
        OUTDIR / "poisson_residual_relative_pred_common_times.png",
        logy=True,
        title="Predicted-frame Poisson residual over common target times",
    )
    plot_metric_by_time(
        common_pred,
        "relative_residual",
        "median",
        "pred",
        "median relative Poisson residual (pred)",
        OUTDIR / "poisson_residual_relative_pred_common_times_linear.png",
        logy=False,
        title="Predicted-frame Poisson residual over common target times",
    )
    plot_reference_line_by_time(
        common_true,
        OUTDIR / "poisson_residual_relative_true_common_times.png",
        linear=False,
    )
    plot_reference_line_by_time(
        common_true,
        OUTDIR / "poisson_residual_relative_true_common_times_linear.png",
        linear=True,
    )
    plot_metric_by_time(
        common_ratio,
        "relative_residual_ratio",
        "median",
        "pred_over_true",
        "pred relative residual / true relative residual",
        OUTDIR / "poisson_residual_pred_over_true_common_times.png",
        logy=False,
        title="Poisson residual relative to true-frame diagnostic floor",
    )
    plot_metric_by_time(
        common_corr,
        "balance_corr",
        "median",
        "pred",
        "corr(laplacian(phi_pred), -rho_pred/eps0)",
        OUTDIR / "poisson_balance_corr_pred_common_times.png",
        logy=False,
        title="Poisson balance pattern correlation",
    )
    plot_metric_by_horizon(
        horizon_rows,
        "relative_residual",
        "median",
        "pred",
        "median relative Poisson residual (pred)",
        OUTDIR / "poisson_residual_relative_pred_by_horizon.png",
        logy=True,
        title="Predicted residual by output horizon, with true floors",
        include_true_floor=True,
    )
    plot_metric_by_horizon(
        horizon_rows,
        "relative_residual",
        "median",
        "pred",
        "median relative Poisson residual (pred)",
        OUTDIR / "poisson_residual_relative_pred_by_horizon_linear.png",
        logy=False,
        title="Predicted residual by output horizon, with true floors",
        include_true_floor=True,
    )
    plot_metric_by_horizon(
        horizon_rows,
        "relative_residual_ratio",
        "median",
        "pred_over_true",
        "pred relative residual / true relative residual",
        OUTDIR / "poisson_residual_pred_over_true_by_horizon.png",
        logy=False,
        title="Predicted residual divided by true diagnostic floor",
        reference_y=1.0,
        reference_label="same as true floor",
    )
    plot_metric_by_horizon(
        horizon_rows,
        "relative_residual_ratio",
        "median",
        "pred_over_true",
        "pred relative residual / true relative residual",
        OUTDIR / "poisson_residual_pred_over_true_by_horizon_linear.png",
        logy=False,
        title="Predicted residual divided by true diagnostic floor",
        reference_y=1.0,
        reference_label="same as true floor",
    )
    plot_ratio_bar(horizon_rows, OUTDIR / "poisson_residual_pred_over_true_summary_bar.png")
    plot_ratio_heatmap(horizon_rows, OUTDIR / "poisson_residual_pred_over_true_horizon_heatmap.png")
    plot_excess_by_time(
        common_ratio,
        OUTDIR / "poisson_residual_excess_over_true_common_times.png",
    )

    summary = write_summary(loaded_cases, target_rows, horizon_rows)
    write_readme()
    print(json.dumps(summary["cases"], indent=2))
    print(f"saved: {OUTDIR}")


if __name__ == "__main__":
    main()
