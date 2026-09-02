import argparse
import csv
import json
import runpy
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"C:\Users\astro\research\SimVPv2")
WORKDIRS = ROOT / "workdirs"
OUTDIR = WORKDIRS / "compare_transfer_low_magnet_stride2_efield_sweep"
H5_PATH = Path(
    r"C:\Users\astro\research\PEPAPIC\test\results"
    r"\2D_ExB_low_magnet_dt2.5e-9_maxt50e-6_macro5"
    r"\SimVPv2_inputs\global_norm_from_high3b_trainfixed_minmax_margin20_t0_to_t4000_step2_training_compatible.h5"
)
DOMAIN_INFO = Path(
    r"C:\Users\astro\research\PEPAPIC\test\results"
    r"\2D_ExB_low_magnet_dt2.5e-9_maxt50e-6_macro5"
    r"\Domain_info\Global_domain_info.json"
)

PRE = 10
AFT = 10
BASE_DT_NS = 12.5
PHI_INDEX = 2
EPS = 1e-30

MODEL_CASES = {
    "baseline": {
        "label": "data-only baseline",
        "color": "#111111",
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_subsample2_trainfixed_disjoint_811_bs2_100ep",
        "config": ROOT / "configs" / "custom" / "pepapic" / "SimVP_gSTA_pepapic_direct.py",
    },
    "lam1e-4_a1.0": {
        "label": "lambda=1e-4, alpha=1.0",
        "color": "#1f77b4",
        "workdir": WORKDIRS
        / "pepapic_simvp_gsta_highmag_macro5_subsample2_direct10_poisson_floor_hinge_lam1em4_floor086_alpha10_trainfixed_disjoint_811_bs2_100ep",
        "config": WORKDIRS
        / "poisson_floor_hinge_sweep_configs"
        / "SimVP_gSTA_pepapic_direct_poisson_floor_hinge_lam1em4_floor086_alpha10.py",
    },
    "lam1e-3_a1.1": {
        "label": "lambda=1e-3, alpha=1.1",
        "color": "#d62728",
        "workdir": WORKDIRS
        / "pepapic_simvp_gsta_highmag_macro5_subsample2_direct10_poisson_floor_hinge_lam1em3_floor086_alpha11_trainfixed_disjoint_811_bs2_100ep",
        "config": ROOT / "configs" / "custom" / "pepapic" / "SimVP_gSTA_pepapic_direct_poisson_floor_hinge.py",
    },
    "lam1e-3_a1.2": {
        "label": "lambda=1e-3, alpha=1.2",
        "color": "#2ca02c",
        "workdir": WORKDIRS
        / "pepapic_simvp_gsta_highmag_macro5_subsample2_direct10_poisson_floor_hinge_lam1em3_floor086_alpha12_trainfixed_disjoint_811_bs2_100ep",
        "config": WORKDIRS
        / "poisson_floor_hinge_sweep_configs"
        / "SimVP_gSTA_pepapic_direct_poisson_floor_hinge_lam1em3_floor086_alpha12.py",
    },
}

CASE_ORDER = ["copy", *MODEL_CASES.keys()]
CASE_STYLE = {
    "copy": {"label": "copy baseline", "color": "#7f7f7f"},
    **{name: {"label": meta["label"], "color": meta["color"]} for name, meta in MODEL_CASES.items()},
}


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    device = torch.device(name)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"[DEVICE] cuda: {torch.cuda.get_device_name(device)}", flush=True)
    else:
        print("[DEVICE] cpu", flush=True)
    return device


def as_str_list(values):
    out = []
    for value in values:
        if isinstance(value, (bytes, bytearray)):
            out.append(value.decode())
        else:
            out.append(str(value))
    return out


def load_h5(path):
    with h5py.File(path, "r") as f:
        data = np.asarray(f["data_tchw"][()], dtype=np.float32)
        timesteps = np.asarray(f["timesteps"][()], dtype=np.int64)
        props = as_str_list(f["props"][()])
        train_min = np.asarray(f["train_min"][()], dtype=np.float64)
        train_max = np.asarray(f["train_max"][()], dtype=np.float64)
        margin = float(f["margin"][()])
        norm_mode = f["norm_mode"][()]
        if isinstance(norm_mode, (bytes, bytearray)):
            norm_mode = norm_mode.decode()
        else:
            norm_mode = str(norm_mode)
    if data.ndim != 4:
        raise ValueError(f"data_tchw must be 4D, got {data.shape}")
    if data.shape[0] == len(timesteps):
        data_tchw = data
    elif data.shape[-1] == len(timesteps):
        data_tchw = np.transpose(data, (3, 2, 0, 1))
    else:
        raise ValueError(f"Cannot infer H5 layout: data={data.shape}, timesteps={len(timesteps)}")
    if "minmax" not in norm_mode:
        raise ValueError(f"Unsupported norm_mode: {norm_mode}")
    return {
        "data": np.ascontiguousarray(data_tchw.astype(np.float32)),
        "timesteps": timesteps,
        "props": props,
        "train_min": train_min,
        "train_max": train_max,
        "margin": margin,
        "norm_mode": norm_mode,
        "phi_index": props.index("phi"),
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


def denorm_phi(phi_norm, info):
    phi_i = info["phi_index"]
    mn = float(info["train_min"][phi_i])
    mx = float(info["train_max"][phi_i])
    value_range = mx - mn
    lo = mn - info["margin"] * value_range
    hi = mx + info["margin"] * value_range
    return phi_norm.astype(np.float64) * (hi - lo) + lo


def electric_field(phi, dx, dy):
    dphi_dy, dphi_dx = np.gradient(phi, dy, dx, axis=(-2, -1), edge_order=1)
    return -dphi_dx, -dphi_dy


def build_model(config_path, ckpt_path, device):
    from openstl.models.simvp_model import SimVP_Model

    cfg = runpy.run_path(str(config_path))
    model = SimVP_Model(
        in_shape=(PRE, 3, 100, 100),
        hid_S=cfg.get("hid_S", 64),
        hid_T=cfg.get("hid_T", 512),
        N_S=cfg.get("N_S", 4),
        N_T=cfg.get("N_T", 8),
        model_type=cfg.get("model_type", "gSTA"),
        drop_path=float(cfg.get("drop_path", 0.0)),
        spatio_kernel_enc=int(cfg.get("spatio_kernel_enc", 3)),
        spatio_kernel_dec=int(cfg.get("spatio_kernel_dec", 3)),
    )
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    state_dict = {
        (key.replace("model.", "", 1) if key.startswith("model.") else key): value
        for key, value in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[MODEL] {ckpt_path.parent.parent.name} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    return model.to(device).eval()


def batch_windows(data, starts):
    x = np.stack([data[start : start + PRE] for start in starts], axis=0)
    y = np.stack([data[start + PRE : start + PRE + AFT] for start in starts], axis=0)
    return x, y


def frame_metrics(case, target, pred_phi_norm, true_phi_norm, copy_phi_norm, info, dx, dy):
    pred_phi = denorm_phi(pred_phi_norm, info)
    true_phi = denorm_phi(true_phi_norm, info)
    copy_phi = denorm_phi(copy_phi_norm, info)

    pred_ex, pred_ey = electric_field(pred_phi, dx, dy)
    true_ex, true_ey = electric_field(true_phi, dx, dy)
    copy_ex, copy_ey = electric_field(copy_phi, dx, dy)

    if case == "copy":
        use_ex, use_ey = copy_ex, copy_ey
        use_phi_norm = copy_phi_norm
    else:
        use_ex, use_ey = pred_ex, pred_ey
        use_phi_norm = pred_phi_norm

    ex_diff = use_ex - true_ex
    ey_diff = use_ey - true_ey
    copy_ex_diff = copy_ex - true_ex
    copy_ey_diff = copy_ey - true_ey

    ex_mse = np.mean(ex_diff * ex_diff, axis=(1, 2))
    ey_mse = np.mean(ey_diff * ey_diff, axis=(1, 2))
    emag_mse = np.mean(ex_diff * ex_diff + ey_diff * ey_diff, axis=(1, 2))
    true_ex_ms = np.mean(true_ex * true_ex, axis=(1, 2))
    true_ey_ms = np.mean(true_ey * true_ey, axis=(1, 2))
    true_emag_ms = np.mean(true_ex * true_ex + true_ey * true_ey, axis=(1, 2))
    copy_emag_mse = np.mean(copy_ex_diff * copy_ex_diff + copy_ey_diff * copy_ey_diff, axis=(1, 2))

    phi_mse = np.mean((use_phi_norm - true_phi_norm) ** 2, axis=(1, 2))
    copy_phi_mse = np.mean((copy_phi_norm - true_phi_norm) ** 2, axis=(1, 2))
    rows = []
    for i in range(len(phi_mse)):
        row = {
            **target[i],
            "case": case,
            "label": CASE_STYLE[case]["label"],
            "phi_mse": float(phi_mse[i]),
            "copy_phi_mse": float(copy_phi_mse[i]),
            "phi_over_copy": float(phi_mse[i] / copy_phi_mse[i]) if copy_phi_mse[i] > 0 else np.nan,
            "ex_mse": float(ex_mse[i]),
            "ey_mse": float(ey_mse[i]),
            "emag_mse": float(emag_mse[i]),
            "ex_nrmse": float(np.sqrt(ex_mse[i] / max(true_ex_ms[i], EPS))),
            "ey_nrmse": float(np.sqrt(ey_mse[i] / max(true_ey_ms[i], EPS))),
            "emag_nrmse": float(np.sqrt(emag_mse[i] / max(true_emag_ms[i], EPS))),
            "emag_over_copy": float(emag_mse[i] / copy_emag_mse[i]) if copy_emag_mse[i] > 0 else np.nan,
        }
        rows.append(row)
    return rows


@torch.inference_mode()
def collect_case_rows(case, model, data, timesteps, info, dx, dy, device, batch_size):
    starts = np.arange(0, data.shape[0] - PRE - AFT + 1, dtype=np.int64)
    rows = []
    for b0 in range(0, len(starts), batch_size):
        b_starts = starts[b0 : b0 + batch_size]
        x_np, y_np = batch_windows(data, b_starts)
        if case == "copy":
            pred_np = None
        else:
            x = torch.from_numpy(x_np).to(device)
            pred_np = model(x).detach().cpu().numpy()
            if pred_np.shape != y_np.shape:
                raise RuntimeError(f"Unexpected prediction shape {pred_np.shape}; expected {y_np.shape}")

        copy_phi = x_np[:, -1, info["phi_index"]]
        last_input_steps = timesteps[b_starts + PRE - 1]
        for out_i in range(AFT):
            target_indices = b_starts + PRE + out_i
            target_steps = timesteps[target_indices]
            targets = [
                {
                    "window_start": int(start),
                    "output_index": int(out_i + 1),
                    "target_index": int(target_index),
                    "target_timestep": int(target_step),
                    "target_time_us": float(target_step * BASE_DT_NS / 1000.0),
                    "horizon_ns": float((target_step - last_step) * BASE_DT_NS),
                }
                for start, target_index, target_step, last_step in zip(
                    b_starts, target_indices, target_steps, last_input_steps
                )
            ]
            pred_phi = copy_phi if case == "copy" else pred_np[:, out_i, info["phi_index"]]
            rows.extend(
                frame_metrics(
                    case,
                    targets,
                    pred_phi,
                    y_np[:, out_i, info["phi_index"]],
                    copy_phi,
                    info,
                    dx,
                    dy,
                )
            )
        if b0 == 0 or (b0 + batch_size) % 200 == 0 or b0 + batch_size >= len(starts):
            print(f"[{case}] {min(b0 + batch_size, len(starts))}/{len(starts)} windows", flush=True)
    return rows


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"count": 0, "mean": np.nan, "median": np.nan, "q25": np.nan, "q75": np.nan}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
    }


def aggregate_summary(rows):
    out = []
    for case in CASE_ORDER:
        group = [row for row in rows if row["case"] == case]
        row = {"case": case, "label": CASE_STYLE[case]["label"]}
        for metric in ("phi_mse", "phi_over_copy", "ex_mse", "ey_mse", "emag_mse", "ex_nrmse", "ey_nrmse", "emag_nrmse", "emag_over_copy"):
            stats = summarize([g[metric] for g in group])
            for stat, value in stats.items():
                row[f"{metric}_{stat}"] = value
        out.append(row)
    return out


def aggregate_by(rows, group_key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["case"], row[group_key])].append(row)
    out = []
    for (case, key), group in sorted(grouped.items()):
        row = {"case": case, "label": CASE_STYLE[case]["label"], group_key: key, "n": len(group)}
        for metric in ("phi_mse", "phi_over_copy", "emag_mse", "emag_nrmse", "emag_over_copy"):
            stats = summarize([g[metric] for g in group])
            for stat, value in stats.items():
                row[f"{metric}_{stat}"] = value
        out.append(row)
    return out


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


def smooth_same(y, window=61):
    y = np.asarray(y, dtype=np.float64)
    if len(y) < 5:
        return y
    window = min(window, len(y))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return y
    pad = window // 2
    padded = np.pad(y, pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def setup_legend_space(ax, xs, ys, logy):
    if xs:
        x_min = min(xs)
        x_max = max(xs)
        span = x_max - x_min
        ax.set_xlim(x_min, x_max + 0.22 * span)
    if ys:
        values = np.asarray(ys, dtype=np.float64)
        values = values[np.isfinite(values)]
        if len(values):
            y_min = float(np.min(values))
            y_max = float(np.max(values))
            if logy:
                ax.set_ylim(max(y_min, 1e-20) * 0.8, y_max * 1.8)
            else:
                ax.set_ylim(min(0.0, y_min * 0.9), y_max * 1.18)


def plot_target_time(rows, metric, stat, ylabel, path, logy=True):
    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    all_x = []
    all_y = []
    for case in CASE_ORDER:
        group = [row for row in rows if row["case"] == case]
        group.sort(key=lambda row: row["target_time_us"])
        x = np.asarray([row["target_time_us"] for row in group], dtype=np.float64)
        y = np.asarray([row[f"{metric}_{stat}"] for row in group], dtype=np.float64)
        y_smooth = smooth_same(y, window=61)
        ax.plot(x, y_smooth, linewidth=1.9, label=CASE_STYLE[case]["label"], color=CASE_STYLE[case]["color"])
        all_x.extend(x.tolist())
        all_y.extend(y_smooth.tolist())
    ax.set_xlabel("target simulation time [us]")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", linewidth=0.75, alpha=0.55)
    setup_legend_space(ax, all_x, all_y, logy)
    ax.legend(loc="lower right", framealpha=0.94, facecolor="white", edgecolor="0.75")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_horizon(rows, metric, stat, ylabel, path, logy=True):
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    all_x = []
    all_y = []
    for case in CASE_ORDER:
        group = [row for row in rows if row["case"] == case]
        group.sort(key=lambda row: row["horizon_ns"])
        x = np.asarray([row["horizon_ns"] for row in group], dtype=np.float64)
        y = np.asarray([row[f"{metric}_{stat}"] for row in group], dtype=np.float64)
        ax.plot(x, y, marker="o", markersize=4.0, linewidth=1.8, label=CASE_STYLE[case]["label"], color=CASE_STYLE[case]["color"])
        all_x.extend(x.tolist())
        all_y.extend(y.tolist())
    ax.set_xlabel("prediction horizon from last input frame [ns]")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", linewidth=0.75, alpha=0.55)
    setup_legend_space(ax, all_x, all_y, logy)
    ax.legend(loc="lower right", framealpha=0.94, facecolor="white", edgecolor="0.75")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_summary_bars(summary_rows, metric, stat, ylabel, path, logy=False):
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    cases = CASE_ORDER
    x = np.arange(len(cases))
    values = [
        next(row for row in summary_rows if row["case"] == case)[f"{metric}_{stat}"]
        for case in cases
    ]
    colors = [CASE_STYLE[case]["color"] for case in cases]
    labels = [CASE_STYLE[case]["label"] for case in cases]
    ax.bar(x, values, color=colors, alpha=0.86, edgecolor="0.25")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.75, alpha=0.55)
    for i, value in enumerate(values):
        ax.text(i, value, f"{value:.3g}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_readme(summary_rows):
    best_emag = min(summary_rows, key=lambda row: row["emag_nrmse_mean"])
    best_phi = min(summary_rows, key=lambda row: row["phi_mse_mean"])
    copy = next(row for row in summary_rows if row["case"] == "copy")
    text = f"""# Low-magnet 3a transfer electric-field comparison

This folder compares electric-field errors for high-magnet 3b stride2 direct10 models transferred to low-magnet 3a.

`E` is computed from the predicted or copied potential:

```text
E = -grad(phi)
```

The copy baseline uses the last true input phi frame. All model cases use true low-magnet input windows and are evaluated as teacher-forced direct prediction, not rollout.

## Key Result

- copy baseline mean |E| NRMSE: `{copy['emag_nrmse_mean']:.8g}`
- best mean |E| NRMSE: `{best_emag['emag_nrmse_mean']:.8g}` at `{best_emag['label']}`
- best mean phi MSE: `{best_phi['phi_mse_mean']:.8g}` at `{best_phi['label']}`

## Files

- `transfer_3a_efield_summary.csv`: aggregate phi and electric-field metrics.
- `transfer_3a_efield_by_horizon.csv`: horizon-wise aggregates.
- `transfer_3a_efield_by_target_time.csv`: target-time aggregates.
- `transfer_3a_emag_nrmse_target_time_smoothed.png`: main smoothed |E| NRMSE plot.
- `transfer_3a_emag_mse_target_time_smoothed.png`: smoothed |E| MSE plot.
- `transfer_3a_emag_over_copy_target_time_smoothed.png`: model/copy |E| MSE ratio.
- `transfer_3a_emag_nrmse_by_horizon.png`: |E| NRMSE by output horizon.
- `transfer_3a_emag_nrmse_summary.png`: aggregate |E| NRMSE bars.

## Japanese Note

この比較では、phi の画像MSEではなく、phi から計算した電場 `E=-grad(phi)` の誤差を見ています。
静かな3aでは copy baseline が phi MSE で強くなりやすいため、物理的に重要な勾配量である電場で差が出るかを確認するための結果です。
"""
    (OUTDIR / "README.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    info = load_h5(H5_PATH)
    data = info["data"]
    timesteps = info["timesteps"]
    dx, dy = domain_spacing(data.shape[-2:])
    print(f"[DATA] {H5_PATH}", flush=True)
    print(f"[DATA] data={data.shape}, timesteps={timesteps[0]}..{timesteps[-1]}, dx={dx}, dy={dy}", flush=True)

    rows = []
    rows.extend(collect_case_rows("copy", None, data, timesteps, info, dx, dy, device, args.batch_size))
    for case, meta in MODEL_CASES.items():
        ckpt = meta["workdir"] / "checkpoints" / "best.ckpt"
        model = build_model(meta["config"], ckpt, device)
        rows.extend(collect_case_rows(case, model, data, timesteps, info, dx, dy, device, args.batch_size))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_rows = aggregate_summary(rows)
    horizon_rows = aggregate_by(rows, "horizon_ns")
    time_rows = aggregate_by(rows, "target_time_us")

    write_csv(OUTDIR / "transfer_3a_efield_raw.csv", rows)
    write_csv(OUTDIR / "transfer_3a_efield_summary.csv", summary_rows)
    write_csv(OUTDIR / "transfer_3a_efield_by_horizon.csv", horizon_rows)
    write_csv(OUTDIR / "transfer_3a_efield_by_target_time.csv", time_rows)

    plot_target_time(
        time_rows,
        "emag_nrmse",
        "mean",
        "|E| NRMSE from phi",
        OUTDIR / "transfer_3a_emag_nrmse_target_time_smoothed.png",
        logy=False,
    )
    plot_target_time(
        time_rows,
        "emag_mse",
        "mean",
        "|E| MSE from phi",
        OUTDIR / "transfer_3a_emag_mse_target_time_smoothed.png",
        logy=True,
    )
    plot_target_time(
        time_rows,
        "emag_over_copy",
        "median",
        "median model/copy |E| MSE ratio",
        OUTDIR / "transfer_3a_emag_over_copy_target_time_smoothed.png",
        logy=True,
    )
    plot_horizon(
        horizon_rows,
        "emag_nrmse",
        "mean",
        "|E| NRMSE from phi",
        OUTDIR / "transfer_3a_emag_nrmse_by_horizon.png",
        logy=False,
    )
    plot_horizon(
        horizon_rows,
        "emag_over_copy",
        "median",
        "median model/copy |E| MSE ratio",
        OUTDIR / "transfer_3a_emag_over_copy_by_horizon.png",
        logy=True,
    )
    plot_summary_bars(
        summary_rows,
        "emag_nrmse",
        "mean",
        "mean |E| NRMSE from phi",
        OUTDIR / "transfer_3a_emag_nrmse_summary.png",
        logy=False,
    )
    plot_summary_bars(
        summary_rows,
        "phi_mse",
        "mean",
        "mean phi MSE, high3b-normalized",
        OUTDIR / "transfer_3a_phi_mse_summary.png",
        logy=True,
    )
    write_readme(summary_rows)

    meta = {
        "description": "Electric-field error comparison for low-magnet 3a transfer results.",
        "h5": str(H5_PATH),
        "domain_info": str(DOMAIN_INFO),
        "outdir": str(OUTDIR),
        "cases": {case: CASE_STYLE[case]["label"] for case in CASE_ORDER},
    }
    (OUTDIR / "transfer_3a_efield_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved: {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
