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
OUTDIR = WORKDIRS / "compare_mixed_b_sweep_leaveout_1p75_b_conditioned_4ch_full_time_direct"
CONFIG = ROOT / "configs" / "custom" / "pepapic" / "SimVP_gSTA_pepapic_direct_b_conditioned.py"
H5_1P75 = Path(
    r"C:\Users\astro\research\PEPAPIC\test\results"
    r"\2D_ExB_1.75mT_magnet_dt2.5e-9_maxt50e-6_macro5"
    r"\SimVPv2_inputs"
    r"\global_norm_from_high3b_trainfixed_minmax_margin20_t0_to_t4000_step2_training_compatible.h5"
)
WORKDIR_4CH = (
    WORKDIRS
    / "pepapic_simvp_gsta_mixed_b_sweep_stride2_direct10_leaveout_1p75mT_b_conditioned_4ch_trainfixed_disjoint_811_bs2_100ep"
)
PREV_3CH_OUTDIR = WORKDIRS / "compare_mixed_b_sweep_leaveout_full_time_direct"

PRE = 10
AFT = 10
BASE_DT_NS = 12.5
B_MT = 1.75
PHI_INDEX = 2

COLORS = {
    "copy": "#111827",
    "3ch": "#7c3aed",
    "4ch": "#ea580c",
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
        in_shape=(PRE, 4, 100, 100),
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
        out_channels=int(cfg.get("out_channels", 3)),
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


def batch_windows_4ch(data, starts, b_mT):
    x3 = np.stack([data[start:start + PRE] for start in starts], axis=0)
    y = np.stack([data[start + PRE:start + PRE + AFT] for start in starts], axis=0)
    cond = np.full((len(starts), PRE, 1, data.shape[2], data.shape[3]), float(b_mT), dtype=np.float32)
    x4 = np.concatenate([x3, cond], axis=2)
    return x4, x3, y


@torch.inference_mode()
def evaluate_4ch(device, batch_size):
    data, timesteps = load_tchw(H5_1P75)
    starts = np.arange(0, data.shape[0] - PRE - AFT + 1, dtype=np.int64)
    ckpt = WORKDIR_4CH / "checkpoints" / "best.ckpt"
    model = build_model(CONFIG, ckpt, device)

    model_sum = np.zeros(data.shape[0], dtype=np.float64)
    copy_sum = np.zeros(data.shape[0], dtype=np.float64)
    counts = np.zeros(data.shape[0], dtype=np.int64)
    horizon_model_sum = np.zeros(AFT, dtype=np.float64)
    horizon_copy_sum = np.zeros(AFT, dtype=np.float64)
    horizon_counts = np.zeros(AFT, dtype=np.int64)

    for b0 in range(0, len(starts), batch_size):
        b_starts = starts[b0:b0 + batch_size]
        x4_np, x3_np, y_np = batch_windows_4ch(data, b_starts, B_MT)
        pred = model(torch.from_numpy(x4_np).to(device)).detach().cpu().numpy()
        if pred.shape != y_np.shape:
            raise RuntimeError(f"Unexpected prediction shape {pred.shape}; expected {y_np.shape}")

        phi_pred = pred[:, :, PHI_INDEX].astype(np.float64)
        phi_true = y_np[:, :, PHI_INDEX].astype(np.float64)
        phi_copy = x3_np[:, -1:, PHI_INDEX].astype(np.float64)
        model_mse = np.mean((phi_pred - phi_true) ** 2, axis=(2, 3))
        copy_mse = np.mean((phi_copy - phi_true) ** 2, axis=(2, 3))

        target_indices = b_starts[:, None] + PRE + np.arange(AFT, dtype=np.int64)[None, :]
        np.add.at(model_sum, target_indices.ravel(), model_mse.ravel())
        np.add.at(copy_sum, target_indices.ravel(), copy_mse.ravel())
        np.add.at(counts, target_indices.ravel(), 1)

        horizon_model_sum += np.sum(model_mse, axis=0)
        horizon_copy_sum += np.sum(copy_mse, axis=0)
        horizon_counts += model_mse.shape[0]

        done = min(b0 + batch_size, len(starts))
        if done == len(starts) or done % (batch_size * 50) == 0:
            print(f"[PRED] 4ch B-conditioned 1.75 mT {done}/{len(starts)} windows", flush=True)

    time_rows = []
    for idx in np.where(counts > 0)[0]:
        model_mean = float(model_sum[idx] / counts[idx])
        copy_mean = float(copy_sum[idx] / counts[idx])
        time_rows.append({
            "case_key": "1p75mT",
            "label": "1.75 mT",
            "model": "b_conditioned_4ch",
            "B_mT": B_MT,
            "target_index": int(idx),
            "target_timestep": int(timesteps[idx]),
            "target_time_us": float(timesteps[idx] * BASE_DT_NS / 1000.0),
            "n_predictions": int(counts[idx]),
            "model_mse_mean": model_mean,
            "copy_mse_mean": copy_mean,
            "model_over_copy": model_mean / copy_mean if copy_mean > 0 else np.nan,
        })

    horizon_rows = []
    for h in range(AFT):
        model_mean = float(horizon_model_sum[h] / horizon_counts[h])
        copy_mean = float(horizon_copy_sum[h] / horizon_counts[h])
        horizon_rows.append({
            "case_key": "1p75mT",
            "label": "1.75 mT",
            "model": "b_conditioned_4ch",
            "B_mT": B_MT,
            "output_index": h + 1,
            "horizon_ns": float((h + 1) * 25.0),
            "n_predictions": int(horizon_counts[h]),
            "model_mse_mean": model_mean,
            "copy_mse_mean": copy_mean,
            "model_over_copy": model_mean / copy_mean if copy_mean > 0 else np.nan,
        })

    return {
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


def read_csv_rows(path):
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_3ch_time_rows():
    path = PREV_3CH_OUTDIR / "leaveout_full_time_direct10_phi_by_target_time.csv"
    rows = []
    for row in read_csv_rows(path):
        if row.get("case_key") != "1p75mT":
            continue
        rows.append({
            "target_time_us": float(row["target_time_us"]),
            "model_mse_mean": float(row["model_mse_mean"]),
            "copy_mse_mean": float(row["copy_mse_mean"]),
            "model_over_copy": float(row["model_over_copy"]),
        })
    return rows


def load_3ch_horizon_rows():
    path = PREV_3CH_OUTDIR / "leaveout_full_time_direct10_phi_by_horizon.csv"
    rows = []
    for row in read_csv_rows(path):
        if row.get("case_key") != "1p75mT":
            continue
        rows.append({
            "horizon_ns": float(row["horizon_ns"]),
            "model_mse_mean": float(row["model_mse_mean"]),
            "copy_mse_mean": float(row["copy_mse_mean"]),
            "model_over_copy": float(row["model_over_copy"]),
        })
    return rows


def rolling_mean(values, window):
    arr = np.asarray(values, dtype=np.float64)
    if window <= 1 or arr.size < window:
        return arr
    left = window // 2
    right = window - 1 - left
    padded = np.pad(arr, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window, dtype=np.float64) / window, mode="valid")


def series_xy(rows, key):
    rows = sorted(rows, key=lambda row: row["target_time_us"])
    x = np.asarray([row["target_time_us"] for row in rows], dtype=np.float64)
    y = np.asarray([row[key] for row in rows], dtype=np.float64)
    return x, y


def plot_mse(time_rows_4ch, time_rows_3ch, outdir, smooth_window, log_y):
    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    x4, y4 = series_xy(time_rows_4ch, "model_mse_mean")
    _, copy4 = series_xy(time_rows_4ch, "copy_mse_mean")
    ax.plot(x4, y4, color=COLORS["4ch"], alpha=0.14, linewidth=0.8)
    ax.plot(x4, rolling_mean(y4, smooth_window), color=COLORS["4ch"], linewidth=2.0, label="4ch B-conditioned")
    if time_rows_3ch:
        x3, y3 = series_xy(time_rows_3ch, "model_mse_mean")
        ax.plot(x3, rolling_mean(y3, smooth_window), color=COLORS["3ch"], linewidth=1.9, label="3ch data-only")
    ax.plot(
        x4,
        rolling_mean(copy4, smooth_window),
        color=COLORS["copy"],
        linewidth=1.4,
        linestyle="--",
        alpha=0.75,
        label="copy baseline",
    )
    ax.set_xlabel("Target simulation time (us)")
    ax.set_ylabel("MSE (phi, high3b-normalized)")
    ax.set_title("1.75 mT leave-out: full-time direct10 phi MSE")
    ax.set_xlim(0.0, 50.0)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.55)
    ax.legend()
    fig.tight_layout()
    suffix = "log" if log_y else "linear"
    png_path = outdir / f"b_conditioned_4ch_1p75_full_time_direct10_phi_mse_{suffix}.png"
    pdf_path = outdir / f"b_conditioned_4ch_1p75_full_time_direct10_phi_mse_{suffix}.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[PLOT] {png_path}")
    print(f"[PLOT] {pdf_path}")


def plot_ratio(time_rows_4ch, time_rows_3ch, outdir, smooth_window):
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    x4, r4 = series_xy(time_rows_4ch, "model_over_copy")
    ax.plot(x4, r4, color=COLORS["4ch"], alpha=0.14, linewidth=0.8)
    ax.plot(x4, rolling_mean(r4, smooth_window), color=COLORS["4ch"], linewidth=2.0, label="4ch B-conditioned")
    if time_rows_3ch:
        x3, r3 = series_xy(time_rows_3ch, "model_over_copy")
        ax.plot(x3, rolling_mean(r3, smooth_window), color=COLORS["3ch"], linewidth=1.9, label="3ch data-only")
    ax.axhline(1.0, color=COLORS["copy"], linestyle="--", linewidth=1.0, alpha=0.75)
    ax.set_xlabel("Target simulation time (us)")
    ax.set_ylabel("Model / copy MSE (phi)")
    ax.set_title("1.75 mT leave-out: full-time direct10 phi MSE ratio")
    ax.set_xlim(0.0, 50.0)
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.55)
    ax.legend()
    fig.tight_layout()
    png_path = outdir / "b_conditioned_4ch_1p75_full_time_direct10_phi_model_over_copy_log.png"
    pdf_path = outdir / "b_conditioned_4ch_1p75_full_time_direct10_phi_model_over_copy_log.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[PLOT] {png_path}")
    print(f"[PLOT] {pdf_path}")


def plot_horizon(horizon_rows_4ch, horizon_rows_3ch, outdir):
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    h4 = sorted(horizon_rows_4ch, key=lambda row: row["horizon_ns"])
    x4 = np.asarray([row["horizon_ns"] for row in h4], dtype=np.float64)
    y4 = np.asarray([row["model_mse_mean"] for row in h4], dtype=np.float64)
    c4 = np.asarray([row["copy_mse_mean"] for row in h4], dtype=np.float64)
    ax.plot(x4, y4, marker="o", color=COLORS["4ch"], linewidth=1.9, label="4ch B-conditioned")
    if horizon_rows_3ch:
        h3 = sorted(horizon_rows_3ch, key=lambda row: row["horizon_ns"])
        x3 = np.asarray([row["horizon_ns"] for row in h3], dtype=np.float64)
        y3 = np.asarray([row["model_mse_mean"] for row in h3], dtype=np.float64)
        ax.plot(x3, y3, marker="o", color=COLORS["3ch"], linewidth=1.7, label="3ch data-only")
    ax.plot(x4, c4, marker="o", color=COLORS["copy"], linestyle="--", linewidth=1.3, alpha=0.75, label="copy")
    ax.set_xlabel("Prediction horizon from last input frame (ns)")
    ax.set_ylabel("MSE (phi, high3b-normalized)")
    ax.set_title("1.75 mT leave-out: direct10 phi MSE by horizon")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.55)
    ax.legend()
    fig.tight_layout()
    png_path = outdir / "b_conditioned_4ch_1p75_full_time_direct10_phi_mse_by_horizon.png"
    pdf_path = outdir / "b_conditioned_4ch_1p75_full_time_direct10_phi_mse_by_horizon.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[PLOT] {png_path}")
    print(f"[PLOT] {pdf_path}")


def summarize(time_rows, label):
    model_mean = float(np.mean([row["model_mse_mean"] for row in time_rows]))
    copy_mean = float(np.mean([row["copy_mse_mean"] for row in time_rows]))
    return {
        "label": label,
        "phi_model_mse_full_mean": model_mean,
        "phi_copy_mse_full_mean": copy_mean,
        "phi_model_over_copy_full_mean": model_mean / copy_mean if copy_mean > 0 else np.nan,
    }


def write_readme(outdir, summary):
    lines = [
        "# 1.75 mT 4ch B-conditioned Full-Time Direct Prediction",
        "",
        "Teacher-forced direct10 evaluation over the full retained 0-50 us sequence.",
        "The fourth input channel is the constant `B_mT` condition channel, set to 1.75.",
        "",
        "## Files",
        "",
        "- `b_conditioned_4ch_full_time_direct10_phi_by_target_time.csv`: 4ch phi MSE by target time.",
        "- `b_conditioned_4ch_full_time_direct10_phi_by_horizon.csv`: 4ch phi MSE by output horizon.",
        "- `b_conditioned_4ch_1p75_full_time_direct10_phi_mse_log.png`: time plot with copy and 3ch comparison.",
        "- `b_conditioned_4ch_1p75_full_time_direct10_phi_model_over_copy_log.png`: ratio plot; below 1 beats copy.",
        "- `b_conditioned_4ch_1p75_full_time_direct10_phi_mse_by_horizon.png`: horizon summary.",
        "",
        "## Summary",
        "",
    ]
    for item in summary["models"]:
        lines.append(
            f"- {item['label']}: model/copy={item['phi_model_over_copy_full_mean']:.3f}, "
            f"model MSE={item['phi_model_mse_full_mean']:.4e}, "
            f"copy MSE={item['phi_copy_mse_full_mean']:.4e}"
        )
    path = outdir / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[README] {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=str(OUTDIR))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--smooth-window", type=int, default=21)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    result = evaluate_4ch(device, args.batch_size)
    time_rows_4ch = result["time_rows"]
    horizon_rows_4ch = result["horizon_rows"]
    write_csv(time_rows_4ch, outdir / "b_conditioned_4ch_full_time_direct10_phi_by_target_time.csv")
    write_csv(horizon_rows_4ch, outdir / "b_conditioned_4ch_full_time_direct10_phi_by_horizon.csv")

    time_rows_3ch = load_3ch_time_rows()
    horizon_rows_3ch = load_3ch_horizon_rows()
    plot_mse(time_rows_4ch, time_rows_3ch, outdir, args.smooth_window, log_y=False)
    plot_mse(time_rows_4ch, time_rows_3ch, outdir, args.smooth_window, log_y=True)
    plot_ratio(time_rows_4ch, time_rows_3ch, outdir, args.smooth_window)
    plot_horizon(horizon_rows_4ch, horizon_rows_3ch, outdir)

    summary = {
        "description": "1.75 mT leave-one-B-out 4-channel B-conditioned full-time direct10 evaluation.",
        "h5": str(H5_1P75),
        "workdir_4ch": str(WORKDIR_4CH),
        "ckpt_4ch": str(WORKDIR_4CH / "checkpoints" / "best.ckpt"),
        "config": str(CONFIG),
        "base_dt_ns": BASE_DT_NS,
        "pre_seq_length": PRE,
        "aft_seq_length": AFT,
        "B_mT_condition_channel": B_MT,
        "data_shape": result["data_shape"],
        "timesteps": result["timesteps"],
        "n_windows": result["n_windows"],
        "device": str(device),
        "models": [summarize(time_rows_4ch, "4ch B-conditioned")],
    }
    if time_rows_3ch:
        summary["models"].append(summarize(time_rows_3ch, "3ch data-only"))

    with open(outdir / "b_conditioned_4ch_full_time_direct10_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[JSON] {outdir / 'b_conditioned_4ch_full_time_direct10_summary.json'}")
    write_readme(outdir, summary)


if __name__ == "__main__":
    main()
