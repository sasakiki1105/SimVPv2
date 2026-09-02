import argparse
import csv
import json
import os
import runpy
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"C:\Users\astro\research\SimVPv2")
WORKDIRS = ROOT / "workdirs"
DEFAULT_H5 = Path(
    r"C:\Users\astro\research\PEPAPIC\test\results"
    r"\2D_ExB_low_magnet_dt2.5e-9_maxt50e-6_macro5"
    r"\SimVPv2_inputs\global_norm_from_high3b_trainfixed_minmax_margin20_t0_to_t4000_step2_training_compatible.h5"
)
DEFAULT_WORKDIR = WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_subsample2_trainfixed_disjoint_811_bs2_100ep"
DEFAULT_CFG = ROOT / "configs" / "custom" / "pepapic" / "SimVP_gSTA_pepapic.py"
DEFAULT_OUTDIR = WORKDIRS / "transfer_low_magnet_stride2_direct10_from_high3b_training_compatible"

PRE = 10
AFT = 10
BASE_DT_NS = 12.5
CHANNELS = ["electron_den", "ion_den", "phi"]
PHI_INDEX = 2


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
    if data.ndim != 4:
        raise ValueError(f"data_tchw must be 4D, got {data.shape}")
    if data.shape[0] == len(timesteps):
        out = data
    elif data.shape[-1] == len(timesteps):
        out = np.transpose(data, (3, 2, 0, 1))
    else:
        raise ValueError(f"Cannot infer H5 layout: data={data.shape}, timesteps={len(timesteps)}")
    return np.ascontiguousarray(out.astype(np.float32)), timesteps


def load_physical_tchw(path):
    with h5py.File(path, "r") as handle:
        arrays = [
            np.asarray(handle[f"fields/{channel}"][:, :257, :256], dtype=np.float32)
            for channel in CHANNELS
        ]
    data = np.stack(arrays, axis=1)
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite physical truth values in {path}")
    return np.ascontiguousarray(data)


def build_model(config_path, ckpt_path, device, aft_seq_length, in_shape):
    from openstl.models.simvp_model import SimVP_Model

    cfg = runpy.run_path(str(config_path))
    model = SimVP_Model(
        in_shape=in_shape,
        hid_S=cfg.get("hid_S", 64),
        hid_T=cfg.get("hid_T", 512),
        N_S=cfg.get("N_S", 4),
        N_T=cfg.get("N_T", 8),
        model_type=cfg.get("model_type", "gSTA"),
        drop_path=float(cfg.get("drop_path", 0.0)),
        spatio_kernel_enc=int(cfg.get("spatio_kernel_enc", 3)),
        spatio_kernel_dec=int(cfg.get("spatio_kernel_dec", 3)),
        aft_seq_length=int(aft_seq_length),
        simvp_direct_aft_seq=bool(cfg.get("simvp_direct_aft_seq", False)),
    )
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    state_dict = {
        (k.replace("model.", "", 1) if k.startswith("model.") else k): v
        for k, v in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[MODEL] missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device).eval()
    return model


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom == 0:
        return np.nan
    return float(np.sum(a * b) / denom)


def argmax_2d(a2d):
    idx = int(np.argmax(a2d))
    return np.unravel_index(idx, a2d.shape)


def batch_windows(data, starts, aft_seq_length):
    x = np.stack([data[start:start + PRE] for start in starts], axis=0)
    y = np.stack([data[start + PRE:start + PRE + aft_seq_length] for start in starts], axis=0)
    return x, y


@torch.inference_mode()
def predict_all(
    model,
    data,
    timesteps,
    device,
    batch_size,
    base_dt_ns,
    aft_seq_length,
    physical_truth=None,
    norm_low=None,
    norm_high=None,
):
    starts = np.arange(0, data.shape[0] - PRE - aft_seq_length + 1, dtype=np.int64)
    rows = []
    for b0 in range(0, len(starts), batch_size):
        b_starts = starts[b0:b0 + batch_size]
        x_np, y_np = batch_windows(data, b_starts, aft_seq_length)
        x = torch.from_numpy(x_np).to(device)
        pred = model(x).detach().cpu().numpy()
        if pred.shape != y_np.shape:
            raise RuntimeError(f"Unexpected prediction shape {pred.shape}; expected {y_np.shape}")

        persistence = x_np[:, -1]
        for bi, start in enumerate(b_starts):
            last_input_step = timesteps[start + PRE - 1]
            for out_idx in range(aft_seq_length):
                target_index = int(start + PRE + out_idx)
                target_step = int(timesteps[target_index])
                horizon_ns = float((target_step - last_input_step) * base_dt_ns)
                target_time_us = float(target_step * base_dt_ns / 1000.0)
                for ci, channel in enumerate(CHANNELS):
                    if physical_truth is None:
                        p = pred[bi, out_idx, ci].astype(np.float64)
                        t = y_np[bi, out_idx, ci].astype(np.float64)
                        c = persistence[bi, ci].astype(np.float64)
                    else:
                        p = (
                            pred[bi, out_idx, ci, :257, :256].astype(np.float64)
                            * (norm_high[ci] - norm_low[ci])
                            + norm_low[ci]
                        )
                        t = physical_truth[target_index, ci].astype(np.float64)
                        c = physical_truth[start + PRE - 1, ci].astype(np.float64)
                    mse = float(np.mean((p - t) ** 2))
                    copy_mse = float(np.mean((c - t) ** 2))
                    row = {
                        "window_start": int(start),
                        "output_index": int(out_idx + 1),
                        "target_index": target_index,
                        "target_timestep": target_step,
                        "target_time_us": target_time_us,
                        "horizon_ns": horizon_ns,
                        "channel": channel,
                        "channel_index": int(ci),
                        "model_mse": mse,
                        "copy_mse": copy_mse,
                        "model_over_copy": mse / copy_mse if copy_mse > 0 else np.nan,
                        "corr": pearson(p, t),
                        "copy_corr": pearson(c, t),
                        "mse_space": (
                            "physical_unclipped"
                            if physical_truth is not None
                            else "normalized"
                        ),
                    }
                    if ci == PHI_INDEX:
                        py, px = argmax_2d(p)
                        ty, tx = argmax_2d(t)
                        row["peak_val_err"] = float(abs(np.max(p) - np.max(t)))
                        row["peak_loc_err_px"] = float(np.sqrt((py - ty) ** 2 + (px - tx) ** 2))
                    else:
                        row["peak_val_err"] = np.nan
                        row["peak_loc_err_px"] = np.nan
                    rows.append(row)
        print(f"[PRED] {min(b0 + batch_size, len(starts))}/{len(starts)} windows", flush=True)
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] {path}")


def aggregate_phi(rows):
    phi_rows = [row for row in rows if row["channel"] == "phi"]
    by_time = {}
    by_horizon = {}
    for row in phi_rows:
        by_time.setdefault(row["target_time_us"], []).append(row)
        by_horizon.setdefault(row["horizon_ns"], []).append(row)

    time_rows = []
    for t, group in sorted(by_time.items()):
        vals = np.asarray([r["model_mse"] for r in group], dtype=np.float64)
        ratios = np.asarray([r["model_over_copy"] for r in group], dtype=np.float64)
        time_rows.append({
            "target_time_us": float(t),
            "n": int(len(group)),
            "model_mse_mean": float(np.mean(vals)),
            "model_mse_median": float(np.median(vals)),
            "model_mse_q25": float(np.quantile(vals, 0.25)),
            "model_mse_q75": float(np.quantile(vals, 0.75)),
            "model_over_copy_mean": float(np.nanmean(ratios)),
            "model_over_copy_median": float(np.nanmedian(ratios)),
        })

    horizon_rows = []
    for h, group in sorted(by_horizon.items()):
        vals = np.asarray([r["model_mse"] for r in group], dtype=np.float64)
        ratios = np.asarray([r["model_over_copy"] for r in group], dtype=np.float64)
        horizon_rows.append({
            "horizon_ns": float(h),
            "n": int(len(group)),
            "model_mse_mean": float(np.mean(vals)),
            "model_mse_median": float(np.median(vals)),
            "model_mse_q25": float(np.quantile(vals, 0.25)),
            "model_mse_q75": float(np.quantile(vals, 0.75)),
            "model_over_copy_mean": float(np.nanmean(ratios)),
            "model_over_copy_median": float(np.nanmedian(ratios)),
        })
    return time_rows, horizon_rows


def write_rows(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] {path}")


def plot_time(time_rows, outdir, prefix, aft_seq_length):
    x = np.asarray([row["target_time_us"] for row in time_rows], dtype=np.float64)
    y = np.asarray([row["model_mse_mean"] for row in time_rows], dtype=np.float64)
    q25 = np.asarray([row["model_mse_q25"] for row in time_rows], dtype=np.float64)
    q75 = np.asarray([row["model_mse_q75"] for row in time_rows], dtype=np.float64)
    plt.figure(figsize=(10, 5.8))
    plt.plot(x, y, color="#1f77b4", linewidth=1.4, label="mean")
    plt.fill_between(x, q25, q75, color="#1f77b4", alpha=0.18, linewidth=0, label="IQR")
    plt.xlabel("Target simulation time (us)")
    plt.ylabel("MSE (phi, high3b-normalized)")
    plt.title(f"High3b model transfer: direct{aft_seq_length} phi MSE")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    path = outdir / f"{prefix}_phi_mse_target_time.png"
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"[PLOT] {path}")


def plot_horizon(horizon_rows, outdir, prefix):
    x = np.asarray([row["horizon_ns"] for row in horizon_rows], dtype=np.float64)
    y = np.asarray([row["model_mse_mean"] for row in horizon_rows], dtype=np.float64)
    q25 = np.asarray([row["model_mse_q25"] for row in horizon_rows], dtype=np.float64)
    q75 = np.asarray([row["model_mse_q75"] for row in horizon_rows], dtype=np.float64)
    plt.figure(figsize=(8.2, 5.5))
    plt.plot(x, y, color="#1f77b4", marker="o", linewidth=1.6, label="mean")
    plt.fill_between(x, q25, q75, color="#1f77b4", alpha=0.18, linewidth=0, label="IQR")
    plt.xlabel("Prediction horizon from last input frame (ns)")
    plt.ylabel("MSE (phi, high3b-normalized)")
    plt.title("High3b stride2 model on low-magnet testcase: phi MSE by horizon")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    path = outdir / f"{prefix}_phi_mse_horizon.png"
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"[PLOT] {path}")


def write_readme(outdir):
    text = """# Experiment

Transfer evaluation: high-magnet testcase 3b stride2 direct10 model applied to the low-magnet testcase 3a.

# Meaning

The low-magnet PIC frames are normalized with the high-magnet training min/max statistics, then the already trained high-magnet stride2 model is used without retraining.

Inputs are always true low-magnet PIC frames. The window is shifted by one retained frame each time:

```text
true frames [s : s+10] -> predict [s+10 : s+20]
next window start = s + 1
```

This is teacher-forced direct prediction over the whole 50 us sequence, not rollout.

The script uses `--device auto` by default. After training jobs finish, this will use CUDA when available. If another training job is still using the GPU, run with `--device cpu` to avoid competing for GPU memory.

# 日本語訳

これは high-magnet test case 3b で学習した stride2 direct10 モデルを、low-magnet test case 3a にそのまま適用する転移評価です。

low-magnet 側の PIC フレームは、high-magnet 学習時の `train_min/train_max` で正規化しています。モデルの再学習はしていません。

入力には常に low-magnet の真値 PIC フレームを使い、次の試行では window を1 retained frame だけずらします。予測値を次の入力には使わないため、rollout ではなく teacher-forced direct prediction です。

スクリプトのデフォルトは `--device auto` です。学習ジョブが終わった後に実行すれば、CUDA が使える環境では自動的に GPU 推論になります。学習中に並行して確認したい場合だけ、`--device cpu` を指定して GPU と競合しないようにします。
"""
    path = outdir / "README.md"
    path.write_text(text, encoding="utf-8")
    print(f"[README] {path}")


def write_transfer_readme(outdir, description):
    text = f"""# Zero-shot direct prediction

{description}

The target PIC frames are normalized with the source model's training
statistics, then the trained model is applied without retraining.

```text
true frames [s : s+10] -> predict [s+10 : s+20]
next window start = s + 1
```

This is teacher-forced direct prediction over the whole sequence, not rollout.

## 日本語

転移先のPICフレームを学習元モデルの訓練時正規化統計で変換し、
再学習なしで推論しています。入力は毎回、転移先PICの真値10枚です。
予測値を次の入力へ戻さないteacher-forced direct predictionであり、
rolloutではありません。
"""
    path = outdir / "README.md"
    path.write_text(text, encoding="utf-8")
    print(f"[README] {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default=str(DEFAULT_H5))
    parser.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    parser.add_argument("--config", default=str(DEFAULT_CFG))
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--physical-truth-h5",
        type=Path,
        default=None,
        help="Consolidated target H5 used for unclipped physical-unit metrics.",
    )
    parser.add_argument(
        "--base-dt-ns",
        type=float,
        default=BASE_DT_NS,
        help="Physical interval represented by one raw PIC timestep index.",
    )
    parser.add_argument(
        "--aft",
        type=int,
        default=AFT,
        help="Number of future frames produced by the direct model.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix (default: low_magnet_direct{aft}).",
    )
    parser.add_argument(
        "--description",
        default="Zero-shot transfer evaluation.",
        help="Description written to README and summary JSON.",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    h5_path = Path(args.h5)
    workdir = Path(args.workdir)
    if args.ckpt:
        ckpt = Path(args.ckpt)
    else:
        ckpt = workdir / "checkpoints" / "best.ckpt"
        if not ckpt.exists():
            ckpt = workdir / "best.ckpt"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data, timesteps = load_tchw(h5_path)
    print(f"[DATA] {h5_path}")
    print(f"[DATA] data={data.shape}, timesteps={timesteps[0]}..{timesteps[-1]}")
    physical_truth = None
    norm_low = None
    norm_high = None
    if args.physical_truth_h5 is not None:
        physical_truth = load_physical_tchw(args.physical_truth_h5.resolve())
        if physical_truth.shape[0] != data.shape[0]:
            raise ValueError(
                f"Physical truth frames {physical_truth.shape[0]} != "
                f"model input frames {data.shape[0]}"
            )
        with h5py.File(h5_path, "r") as handle:
            norm_low = np.asarray(handle["norm_low"], dtype=np.float64)
            norm_high = np.asarray(handle["norm_high"], dtype=np.float64)
        print(
            f"[PHYSICAL TRUTH] {args.physical_truth_h5.resolve()} "
            f"shape={physical_truth.shape}"
        )
    if data.shape[1] != len(CHANNELS):
        raise ValueError(f"Expected {len(CHANNELS)} channels, got {data.shape[1]}")
    in_shape = (PRE, data.shape[1], data.shape[2], data.shape[3])
    model = build_model(args.config, ckpt, device, args.aft, in_shape)
    rows = predict_all(
        model,
        data,
        timesteps,
        device,
        args.batch_size,
        args.base_dt_ns,
        args.aft,
        physical_truth=physical_truth,
        norm_low=norm_low,
        norm_high=norm_high,
    )

    prefix = args.prefix or f"low_magnet_direct{args.aft}"
    write_csv(rows, outdir / f"{prefix}_raw_predictions.csv")
    time_rows, horizon_rows = aggregate_phi(rows)
    write_rows(time_rows, outdir / f"{prefix}_phi_by_target_time.csv")
    write_rows(horizon_rows, outdir / f"{prefix}_phi_by_horizon.csv")
    plot_time(time_rows, outdir, prefix, args.aft)
    plot_horizon(horizon_rows, outdir, prefix)
    write_transfer_readme(outdir, args.description)

    summary = {
        "description": args.description,
        "in_shape": list(in_shape),
        "h5": str(h5_path),
        "workdir": str(workdir),
        "ckpt": str(ckpt),
        "config": str(args.config),
        "device": str(device),
        "data_shape": list(data.shape),
        "timesteps": [int(timesteps[0]), int(timesteps[-1])],
        "base_dt_ns": float(args.base_dt_ns),
        "aft_seq_length": int(args.aft),
        "n_windows": int(data.shape[0] - PRE - args.aft + 1),
        "n_raw_rows": int(len(rows)),
        "physical_truth_h5": (
            str(args.physical_truth_h5.resolve())
            if args.physical_truth_h5 is not None
            else None
        ),
        "mse_space": (
            "physical_unclipped"
            if args.physical_truth_h5 is not None
            else "normalized"
        ),
    }
    with open(outdir / f"{prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[JSON] {outdir / f'{prefix}_summary.json'}")


if __name__ == "__main__":
    main()
