import argparse
import csv
import json
import runpy
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
WORKDIRS = ROOT / "workdirs"
CONFIG = ROOT / "configs" / "custom" / "pepapic" / "SimVP_gSTA_pepapic_direct_poisson_baseline.py"
OUTDIR = WORKDIRS / "compare_mixed_b_sweep_leaveout_full_time_direct"

PRE = 10
AFT = 10
BASE_DT_NS = 12.5
PHI_INDEX = 2


CASES = [
    {
        "case_key": "1p0mT",
        "label": "1.0 mT",
        "B_mT": 1.0,
        "h5": Path(
            r"C:\Users\astro\research\PEPAPIC\test\results"
            r"\2D_ExB_1.0mT_magnet_dt2.5e-9_maxt50e-6_macro5"
            r"\SimVPv2_inputs"
            r"\global_norm_from_high3b_trainfixed_minmax_margin20_t0_to_t4000_step2_training_compatible.h5"
        ),
        "workdir": WORKDIRS
        / "pepapic_simvp_gsta_mixed_b_sweep_stride2_direct10_leaveout_1p0mT_data_only_trainfixed_disjoint_811_bs2_100ep",
    },
    {
        "case_key": "1p75mT",
        "label": "1.75 mT",
        "B_mT": 1.75,
        "h5": Path(
            r"C:\Users\astro\research\PEPAPIC\test\results"
            r"\2D_ExB_1.75mT_magnet_dt2.5e-9_maxt50e-6_macro5"
            r"\SimVPv2_inputs"
            r"\global_norm_from_high3b_trainfixed_minmax_margin20_t0_to_t4000_step2_training_compatible.h5"
        ),
        "workdir": WORKDIRS
        / "pepapic_simvp_gsta_mixed_b_sweep_stride2_direct10_leaveout_1p75mT_data_only_trainfixed_disjoint_811_bs2_100ep",
    },
]


COLORS = {
    "1p0mT": "#16a34a",
    "1p75mT": "#7c3aed",
}


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    device = torch.device(name)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"[DEVICE] cuda: {torch.cuda.get_device_name(device)}")
    else:
        print("[DEVICE] cpu")
    return device


def load_tchw(path):
    with h5py.File(path, "r") as f:
        data = np.asarray(f["data_tchw"][()], dtype=np.float32)
        timesteps = np.asarray(f["timesteps"][()], dtype=np.int64)
    if data.shape[0] == len(timesteps):
        out = data
    elif data.shape[-1] == len(timesteps):
        out = np.transpose(data, (3, 2, 0, 1))
    else:
        raise ValueError(f"Cannot infer H5 layout: data={data.shape}, timesteps={len(timesteps)}")
    return np.ascontiguousarray(out.astype(np.float32)), timesteps


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
        aft_seq_length=AFT,
        simvp_direct_aft_seq=bool(cfg.get("simvp_direct_aft_seq", False)),
    )
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    state_dict = {
        (key.replace("model.", "", 1) if key.startswith("model.") else key): value
        for key, value in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[MODEL] {ckpt_path}")
    print(f"[MODEL] missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device).eval()
    return model


def batch_windows(data, starts):
    x = np.stack([data[start:start + PRE] for start in starts], axis=0)
    y = np.stack([data[start + PRE:start + PRE + AFT] for start in starts], axis=0)
    return x, y


@torch.inference_mode()
def evaluate_case(case, config_path, device, batch_size):
    data, timesteps = load_tchw(case["h5"])
    starts = np.arange(0, data.shape[0] - PRE - AFT + 1, dtype=np.int64)
    ckpt = case["workdir"] / "checkpoints" / "best.ckpt"
    model = build_model(config_path, ckpt, device)

    model_sum = np.zeros(data.shape[0], dtype=np.float64)
    copy_sum = np.zeros(data.shape[0], dtype=np.float64)
    ratio_sum = np.zeros(data.shape[0], dtype=np.float64)
    counts = np.zeros(data.shape[0], dtype=np.int64)
    horizon_model_sum = np.zeros(AFT, dtype=np.float64)
    horizon_copy_sum = np.zeros(AFT, dtype=np.float64)
    horizon_counts = np.zeros(AFT, dtype=np.int64)

    for b0 in range(0, len(starts), batch_size):
        b_starts = starts[b0:b0 + batch_size]
        x_np, y_np = batch_windows(data, b_starts)
        pred = model(torch.from_numpy(x_np).to(device)).detach().cpu().numpy()
        if pred.shape != y_np.shape:
            raise RuntimeError(f"Unexpected prediction shape {pred.shape}; expected {y_np.shape}")

        phi_pred = pred[:, :, PHI_INDEX].astype(np.float64)
        phi_true = y_np[:, :, PHI_INDEX].astype(np.float64)
        phi_copy = x_np[:, -1:, PHI_INDEX].astype(np.float64)

        model_mse = np.mean((phi_pred - phi_true) ** 2, axis=(2, 3))
        copy_mse = np.mean((phi_copy - phi_true) ** 2, axis=(2, 3))
        ratio = np.divide(
            model_mse,
            copy_mse,
            out=np.full_like(model_mse, np.nan, dtype=np.float64),
            where=copy_mse > 0,
        )

        target_indices = b_starts[:, None] + PRE + np.arange(AFT, dtype=np.int64)[None, :]
        np.add.at(model_sum, target_indices.ravel(), model_mse.ravel())
        np.add.at(copy_sum, target_indices.ravel(), copy_mse.ravel())
        np.add.at(ratio_sum, target_indices.ravel(), np.nan_to_num(ratio, nan=0.0).ravel())
        np.add.at(counts, target_indices.ravel(), 1)

        horizon_model_sum += np.sum(model_mse, axis=0)
        horizon_copy_sum += np.sum(copy_mse, axis=0)
        horizon_counts += model_mse.shape[0]

        print(f"[PRED] {case['label']} {min(b0 + batch_size, len(starts))}/{len(starts)} windows", flush=True)

    valid = counts > 0
    time_rows = []
    for idx in np.where(valid)[0]:
        model_mean = float(model_sum[idx] / counts[idx])
        copy_mean = float(copy_sum[idx] / counts[idx])
        time_rows.append({
            "case_key": case["case_key"],
            "label": case["label"],
            "B_mT": float(case["B_mT"]),
            "target_index": int(idx),
            "target_timestep": int(timesteps[idx]),
            "target_time_us": float(timesteps[idx] * BASE_DT_NS / 1000.0),
            "n_predictions": int(counts[idx]),
            "model_mse_mean": model_mean,
            "copy_mse_mean": copy_mean,
            "model_over_copy": model_mean / copy_mean if copy_mean > 0 else np.nan,
            "model_over_copy_mean_of_predictions": float(ratio_sum[idx] / counts[idx]),
        })

    horizon_rows = []
    for h in range(AFT):
        model_mean = float(horizon_model_sum[h] / horizon_counts[h])
        copy_mean = float(horizon_copy_sum[h] / horizon_counts[h])
        horizon_rows.append({
            "case_key": case["case_key"],
            "label": case["label"],
            "B_mT": float(case["B_mT"]),
            "output_index": h + 1,
            "horizon_ns": float((h + 1) * 25.0),
            "n_predictions": int(horizon_counts[h]),
            "model_mse_mean": model_mean,
            "copy_mse_mean": copy_mean,
            "model_over_copy": model_mean / copy_mean if copy_mean > 0 else np.nan,
        })

    return {
        "case": case,
        "data_shape": list(data.shape),
        "timesteps": [int(timesteps[0]), int(timesteps[-1])],
        "n_windows": int(len(starts)),
        "time_rows": time_rows,
        "horizon_rows": horizon_rows,
    }


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] {path}")


def rolling_mean(values, window):
    arr = np.asarray(values, dtype=np.float64)
    if window <= 1 or arr.size < window:
        return arr
    left = window // 2
    right = window - 1 - left
    padded = np.pad(arr, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window, dtype=np.float64) / window, mode="valid")


def plot_mse_by_time(rows, outdir, smooth_window, log_y):
    fig, ax = plt.subplots(figsize=(10.6, 5.6))
    for case_key in ["1p0mT", "1p75mT"]:
        sub = [row for row in rows if row["case_key"] == case_key]
        sub = sorted(sub, key=lambda row: row["target_time_us"])
        x = np.asarray([row["target_time_us"] for row in sub], dtype=np.float64)
        model = np.asarray([row["model_mse_mean"] for row in sub], dtype=np.float64)
        copy = np.asarray([row["copy_mse_mean"] for row in sub], dtype=np.float64)
        label = sub[0]["label"]
        color = COLORS[case_key]
        ax.plot(x, model, color=color, alpha=0.16, linewidth=0.8)
        ax.plot(x, copy, color=color, alpha=0.10, linewidth=0.8, linestyle="--")
        ax.plot(x, rolling_mean(model, smooth_window), color=color, linewidth=1.8, label=f"{label} model")
        ax.plot(
            x,
            rolling_mean(copy, smooth_window),
            color=color,
            linewidth=1.5,
            linestyle="--",
            label=f"{label} copy",
        )
    ax.set_xlabel("Target simulation time (us)")
    ax.set_ylabel("MSE (phi, high3b-normalized)")
    ax.set_title("Leave-one-B-out transfer: full-time direct10 phi MSE")
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.55)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    suffix = "log" if log_y else "linear"
    path = outdir / f"leaveout_1p0_1p75_full_time_direct10_phi_mse_{suffix}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"[PLOT] {path}")


def plot_ratio_by_time(rows, outdir, smooth_window):
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    for case_key in ["1p0mT", "1p75mT"]:
        sub = [row for row in rows if row["case_key"] == case_key]
        sub = sorted(sub, key=lambda row: row["target_time_us"])
        x = np.asarray([row["target_time_us"] for row in sub], dtype=np.float64)
        ratio = np.asarray([row["model_over_copy"] for row in sub], dtype=np.float64)
        label = sub[0]["label"]
        color = COLORS[case_key]
        ax.plot(x, ratio, color=color, alpha=0.14, linewidth=0.8)
        ax.plot(x, rolling_mean(ratio, smooth_window), color=color, linewidth=1.8, label=label)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.set_xlabel("Target simulation time (us)")
    ax.set_ylabel("Model / copy MSE (phi)")
    ax.set_title("Leave-one-B-out transfer: full-time direct10 phi MSE ratio")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.55)
    ax.legend()
    fig.tight_layout()
    path = outdir / "leaveout_1p0_1p75_full_time_direct10_phi_model_over_copy_log.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"[PLOT] {path}")


def plot_horizon(horizon_rows, outdir):
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for case_key in ["1p0mT", "1p75mT"]:
        sub = [row for row in horizon_rows if row["case_key"] == case_key]
        sub = sorted(sub, key=lambda row: row["output_index"])
        x = np.asarray([row["horizon_ns"] for row in sub], dtype=np.float64)
        model = np.asarray([row["model_mse_mean"] for row in sub], dtype=np.float64)
        copy = np.asarray([row["copy_mse_mean"] for row in sub], dtype=np.float64)
        label = sub[0]["label"]
        color = COLORS[case_key]
        ax.plot(x, model, marker="o", color=color, linewidth=1.8, label=f"{label} model")
        ax.plot(x, copy, marker="o", color=color, linewidth=1.4, linestyle="--", label=f"{label} copy")
    ax.set_xlabel("Prediction horizon from last input frame (ns)")
    ax.set_ylabel("MSE (phi, high3b-normalized)")
    ax.set_title("Leave-one-B-out transfer: direct10 phi MSE by horizon")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.55)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    path = outdir / "leaveout_1p0_1p75_full_time_direct10_phi_mse_by_horizon.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"[PLOT] {path}")


def write_readme(outdir, summary):
    lines = [
        "# Leave-one-B-out Full-Time Direct Prediction",
        "",
        "Teacher-forced direct10 evaluation over each target testcase's full retained 0-50 us sequence.",
        "The first plotted target time is 0.25 us because 10 stride2 input frames are required before prediction.",
        "",
        "Model and copy MSE are averaged over all direct predictions that land on the same target time.",
        "",
        "## Files",
        "",
        "- `leaveout_full_time_direct10_phi_by_target_time.csv`: phi MSE by target simulation time.",
        "- `leaveout_full_time_direct10_phi_by_horizon.csv`: phi MSE by output horizon.",
        "- `leaveout_1p0_1p75_full_time_direct10_phi_mse_linear.png`: linear-y time plot.",
        "- `leaveout_1p0_1p75_full_time_direct10_phi_mse_log.png`: log-y time plot.",
        "- `leaveout_1p0_1p75_full_time_direct10_phi_model_over_copy_log.png`: model/copy ratio by time.",
        "- `leaveout_1p0_1p75_full_time_direct10_phi_mse_by_horizon.png`: horizon summary.",
        "",
        "## Summary",
        "",
    ]
    for item in summary["cases"]:
        lines.append(
            f"- {item['label']}: windows={item['n_windows']}, "
            f"mean model/copy={item['phi_model_over_copy_full_mean']:.3f}, "
            f"model MSE={item['phi_model_mse_full_mean']:.4e}, "
            f"copy MSE={item['phi_copy_mse_full_mean']:.4e}"
        )
    path = outdir / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[README] {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--outdir", default=str(OUTDIR))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--smooth-window", type=int, default=21)
    args = parser.parse_args()

    device = resolve_device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_time_rows = []
    all_horizon_rows = []
    summary = {
        "config": str(Path(args.config)),
        "base_dt_ns": BASE_DT_NS,
        "pre_seq_length": PRE,
        "aft_seq_length": AFT,
        "device": str(device),
        "cases": [],
    }

    for case in CASES:
        print(f"[CASE] {case['label']}")
        result = evaluate_case(case, Path(args.config), device, args.batch_size)
        all_time_rows.extend(result["time_rows"])
        all_horizon_rows.extend(result["horizon_rows"])

        model_mean = float(np.mean([row["model_mse_mean"] for row in result["time_rows"]]))
        copy_mean = float(np.mean([row["copy_mse_mean"] for row in result["time_rows"]]))
        summary["cases"].append({
            "case_key": case["case_key"],
            "label": case["label"],
            "B_mT": float(case["B_mT"]),
            "h5": str(case["h5"]),
            "workdir": str(case["workdir"]),
            "ckpt": str(case["workdir"] / "checkpoints" / "best.ckpt"),
            "data_shape": result["data_shape"],
            "timesteps": result["timesteps"],
            "n_windows": result["n_windows"],
            "phi_model_mse_full_mean": model_mean,
            "phi_copy_mse_full_mean": copy_mean,
            "phi_model_over_copy_full_mean": model_mean / copy_mean if copy_mean > 0 else np.nan,
        })

    write_csv(all_time_rows, outdir / "leaveout_full_time_direct10_phi_by_target_time.csv")
    write_csv(all_horizon_rows, outdir / "leaveout_full_time_direct10_phi_by_horizon.csv")
    plot_mse_by_time(all_time_rows, outdir, args.smooth_window, log_y=False)
    plot_mse_by_time(all_time_rows, outdir, args.smooth_window, log_y=True)
    plot_ratio_by_time(all_time_rows, outdir, args.smooth_window)
    plot_horizon(all_horizon_rows, outdir)

    with open(outdir / "leaveout_full_time_direct10_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[JSON] {outdir / 'leaveout_full_time_direct10_summary.json'}")
    write_readme(outdir, summary)


if __name__ == "__main__":
    main()
