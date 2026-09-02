import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_STRIDE1_ROLLOUT = (
    r"C:\Users\astro\research\SimVPv2\workdirs"
    r"\pepapic_simvp_gsta_highmag_macro5_trainfixed_disjoint_811_bs2_100ep"
    r"\rollout_tp0_quads_assets"
)
DEFAULT_STRIDE4_ROLLOUT = (
    r"C:\Users\astro\research\SimVPv2\workdirs"
    r"\pepapic_simvp_gsta_highmag_macro5_subsample4_trainfixed_disjoint_811_bs2_100ep"
    r"\rollout_tp0_quads_assets"
)
DEFAULT_STRIDE1_H5 = (
    r"C:\Users\astro\research\PEPAPIC\test\results"
    r"\2D_ExB_high_magnet_dt2.5e-9_maxt50e-6_macro5"
    r"\SimVPv2_inputs\global_norm_trainfixed_minmax_margin20_t0_to_t4000.h5"
)
DEFAULT_STRIDE4_H5 = (
    r"C:\Users\astro\research\PEPAPIC\test\results"
    r"\2D_ExB_high_magnet_dt2.5e-9_maxt50e-6_macro5"
    r"\SimVPv2_inputs\global_norm_trainfixed_minmax_margin20_t0_to_t4000_step4.h5"
)
DEFAULT_OUTDIR = (
    r"C:\Users\astro\research\SimVPv2\workdirs"
    r"\compare_rollout_stride1_sub4_vs_stride4_tp0"
)


def load_rollout(base):
    base = Path(base)
    arrays = {
        "preds": np.load(base / "preds_roll.npy", mmap_mode="r"),
        "trues": np.load(base / "trues_roll.npy", mmap_mode="r"),
    }
    for key, x in arrays.items():
        if x.ndim != 5:
            raise ValueError(f"{base / (key + '_roll.npy')} must be 5D, got {x.shape}")
        if x.shape[1] != 1:
            raise ValueError(f"expected second dim to be 1 for {key}, got {x.shape}")
    return arrays


def read_minmax(path):
    with h5py.File(path, "r") as f:
        if "train_min" not in f or "train_max" not in f:
            return None
        margin = float(f["margin"][()]) if "margin" in f else 0.0
        props = []
        if "props" in f:
            for v in f["props"][()]:
                props.append(v.decode() if isinstance(v, (bytes, bytearray)) else str(v))
        return {
            "train_min": np.asarray(f["train_min"][()], dtype=np.float64),
            "train_max": np.asarray(f["train_max"][()], dtype=np.float64),
            "margin": margin,
            "props": props,
        }


def denorm_minmax(x, meta):
    if meta is None:
        return None
    mn = meta["train_min"].reshape((1, 1, -1, 1, 1))
    mx = meta["train_max"].reshape((1, 1, -1, 1, 1))
    r = mx - mn
    mn2 = mn - meta["margin"] * r
    mx2 = mx + meta["margin"] * r
    return x.astype(np.float64) * (mx2 - mn2) + mn2


def mse_per_frame(pred, true, channel):
    d = pred[:, 0, channel].astype(np.float64) - true[:, 0, channel].astype(np.float64)
    return np.mean(d * d, axis=(1, 2))


def argmax_2d(a2d):
    idx = int(np.argmax(a2d))
    return np.unravel_index(idx, a2d.shape)


def peak_metrics(pred, true, channel):
    p = pred[:, 0, channel].astype(np.float64)
    t = true[:, 0, channel].astype(np.float64)
    peak_val_err = np.zeros(p.shape[0], dtype=np.float64)
    peak_loc_err = np.zeros(p.shape[0], dtype=np.float64)
    for k in range(p.shape[0]):
        peak_val_err[k] = abs(float(np.max(p[k])) - float(np.max(t[k])))
        py, px = argmax_2d(p[k])
        ty, tx = argmax_2d(t[k])
        peak_loc_err[k] = float(np.hypot(py - ty, px - tx))
    return peak_val_err, peak_loc_err


def save_line_plot(out_png, time_us, series, ylabel, title):
    plt.figure(figsize=(9, 5))
    for label, y in series:
        plt.plot(time_us, y, label=label, linewidth=1.3)
    plt.xlabel("Physical time since rollout start (us)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[PLOT] {out_png}")


def summarize(x):
    return {
        "first": float(x[0]),
        "last": float(x[-1]),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "max": float(np.max(x)),
    }


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Compare stride1 tp0 rollout sampled every 4 frames with stride4 tp0 rollout "
            "at the same physical times."
        )
    )
    ap.add_argument("--stride1-rollout", default=DEFAULT_STRIDE1_ROLLOUT)
    ap.add_argument("--stride4-rollout", default=DEFAULT_STRIDE4_ROLLOUT)
    ap.add_argument("--stride1-h5", default=DEFAULT_STRIDE1_H5)
    ap.add_argument("--stride4-h5", default=DEFAULT_STRIDE4_H5)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--subsample", type=int, default=4)
    ap.add_argument("--dt-stride1-ns", type=float, default=12.5)
    ap.add_argument("--phi-channel", type=int, default=2)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    s1 = load_rollout(args.stride1_rollout)
    s4 = load_rollout(args.stride4_rollout)
    s1_meta = read_minmax(args.stride1_h5)
    s4_meta = read_minmax(args.stride4_h5)

    k_common = min(
        s4["preds"].shape[0],
        (s1["preds"].shape[0] - 1) // args.subsample + 1,
    )
    idx1 = np.arange(k_common, dtype=np.int64) * args.subsample
    time_us = idx1.astype(np.float64) * args.dt_stride1_ns / 1000.0

    s1_pred = np.asarray(s1["preds"][idx1], dtype=np.float32)
    s1_true = np.asarray(s1["trues"][idx1], dtype=np.float32)
    s4_pred = np.asarray(s4["preds"][:k_common], dtype=np.float32)
    s4_true = np.asarray(s4["trues"][:k_common], dtype=np.float32)

    phi = args.phi_channel
    s1_mse_phi_norm = mse_per_frame(s1_pred, s1_true, phi)
    s4_mse_phi_norm = mse_per_frame(s4_pred, s4_true, phi)
    truth_mse_phi_norm = mse_per_frame(s1_true, s4_true, phi)
    pred_mse_phi_norm = mse_per_frame(s1_pred, s4_pred, phi)

    s1_peak_val_norm, s1_peak_loc = peak_metrics(s1_pred, s1_true, phi)
    s4_peak_val_norm, s4_peak_loc = peak_metrics(s4_pred, s4_true, phi)

    s1_pred_phys = denorm_minmax(s1_pred, s1_meta)
    s1_true_phys = denorm_minmax(s1_true, s1_meta)
    s4_pred_phys = denorm_minmax(s4_pred, s4_meta)
    s4_true_phys = denorm_minmax(s4_true, s4_meta)

    phys_available = all(x is not None for x in [s1_pred_phys, s1_true_phys, s4_pred_phys, s4_true_phys])
    if phys_available:
        s1_mse_phi_phys = mse_per_frame(s1_pred_phys, s1_true_phys, phi)
        s4_mse_phi_phys = mse_per_frame(s4_pred_phys, s4_true_phys, phi)
        truth_mse_phi_phys = mse_per_frame(s1_true_phys, s4_true_phys, phi)
        s1_peak_val_phys, _ = peak_metrics(s1_pred_phys, s1_true_phys, phi)
        s4_peak_val_phys, _ = peak_metrics(s4_pred_phys, s4_true_phys, phi)
    else:
        s1_mse_phi_phys = np.full(k_common, np.nan)
        s4_mse_phi_phys = np.full(k_common, np.nan)
        truth_mse_phi_phys = np.full(k_common, np.nan)
        s1_peak_val_phys = np.full(k_common, np.nan)
        s4_peak_val_phys = np.full(k_common, np.nan)

    per_channel_norm = {}
    for ch in range(s1_pred.shape[2]):
        per_channel_norm[f"mse_ch{ch}_stride1_sub{args.subsample}"] = mse_per_frame(s1_pred, s1_true, ch)
        per_channel_norm[f"mse_ch{ch}_stride4"] = mse_per_frame(s4_pred, s4_true, ch)

    csv_path = outdir / "same_physical_time_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "frame_50ns_index",
            "stride1_rollout_frame",
            "time_us",
            "mse_phi_norm_stride1_sub4",
            "mse_phi_norm_stride4",
            "truth_mse_phi_norm_stride1_vs_stride4",
            "pred_mse_phi_norm_stride1_vs_stride4",
            "mse_phi_phys_stride1_sub4",
            "mse_phi_phys_stride4",
            "truth_mse_phi_phys_stride1_vs_stride4",
            "peak_val_err_phi_norm_stride1_sub4",
            "peak_val_err_phi_norm_stride4",
            "peak_val_err_phi_phys_stride1_sub4",
            "peak_val_err_phi_phys_stride4",
            "peak_loc_err_phi_px_stride1_sub4",
            "peak_loc_err_phi_px_stride4",
        ]
        for key in per_channel_norm:
            fieldnames.append(key)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for k in range(k_common):
            row = {
                "frame_50ns_index": k,
                "stride1_rollout_frame": int(idx1[k]),
                "time_us": float(time_us[k]),
                "mse_phi_norm_stride1_sub4": float(s1_mse_phi_norm[k]),
                "mse_phi_norm_stride4": float(s4_mse_phi_norm[k]),
                "truth_mse_phi_norm_stride1_vs_stride4": float(truth_mse_phi_norm[k]),
                "pred_mse_phi_norm_stride1_vs_stride4": float(pred_mse_phi_norm[k]),
                "mse_phi_phys_stride1_sub4": float(s1_mse_phi_phys[k]),
                "mse_phi_phys_stride4": float(s4_mse_phi_phys[k]),
                "truth_mse_phi_phys_stride1_vs_stride4": float(truth_mse_phi_phys[k]),
                "peak_val_err_phi_norm_stride1_sub4": float(s1_peak_val_norm[k]),
                "peak_val_err_phi_norm_stride4": float(s4_peak_val_norm[k]),
                "peak_val_err_phi_phys_stride1_sub4": float(s1_peak_val_phys[k]),
                "peak_val_err_phi_phys_stride4": float(s4_peak_val_phys[k]),
                "peak_loc_err_phi_px_stride1_sub4": float(s1_peak_loc[k]),
                "peak_loc_err_phi_px_stride4": float(s4_peak_loc[k]),
            }
            for key, values in per_channel_norm.items():
                row[key] = float(values[k])
            writer.writerow(row)
    print(f"[CSV] {csv_path}")

    save_line_plot(
        outdir / "mse_phi_norm_same_physical_time.png",
        time_us,
        [
            (f"stride1 sampled every {args.subsample} frames", s1_mse_phi_norm),
            ("stride4", s4_mse_phi_norm),
        ],
        "MSE (phi, normalized)",
        "Phi MSE at the same 50 ns physical times",
    )
    save_line_plot(
        outdir / "peak_loc_err_phi_same_physical_time.png",
        time_us,
        [
            (f"stride1 sampled every {args.subsample} frames", s1_peak_loc),
            ("stride4", s4_peak_loc),
        ],
        "Peak location error (px)",
        "Phi peak location error at the same 50 ns physical times",
    )
    if phys_available:
        save_line_plot(
            outdir / "mse_phi_phys_same_physical_time.png",
            time_us,
            [
                (f"stride1 sampled every {args.subsample} frames", s1_mse_phi_phys),
                ("stride4", s4_mse_phi_phys),
            ],
            "MSE (phi, denormalized)",
            "Denormalized phi MSE at the same 50 ns physical times",
        )

    summary = {
        "stride1_rollout": str(args.stride1_rollout),
        "stride4_rollout": str(args.stride4_rollout),
        "stride1_h5": str(args.stride1_h5),
        "stride4_h5": str(args.stride4_h5),
        "subsample": args.subsample,
        "dt_stride1_ns": args.dt_stride1_ns,
        "dt_compared_ns": args.dt_stride1_ns * args.subsample,
        "n_compared_frames": int(k_common),
        "stride1_indices": [int(idx1[0]), int(idx1[-1])],
        "time_us": [float(time_us[0]), float(time_us[-1])],
        "phi_channel": phi,
        "normalized": {
            "mse_phi_stride1_sub4": summarize(s1_mse_phi_norm),
            "mse_phi_stride4": summarize(s4_mse_phi_norm),
            "truth_mse_phi_stride1_vs_stride4": summarize(truth_mse_phi_norm),
            "pred_mse_phi_stride1_vs_stride4": summarize(pred_mse_phi_norm),
            "peak_loc_err_phi_stride1_sub4": summarize(s1_peak_loc),
            "peak_loc_err_phi_stride4": summarize(s4_peak_loc),
            "peak_val_err_phi_stride1_sub4": summarize(s1_peak_val_norm),
            "peak_val_err_phi_stride4": summarize(s4_peak_val_norm),
        },
        "denormalized_available": bool(phys_available),
    }
    if phys_available:
        summary["denormalized"] = {
            "mse_phi_stride1_sub4": summarize(s1_mse_phi_phys),
            "mse_phi_stride4": summarize(s4_mse_phi_phys),
            "truth_mse_phi_stride1_vs_stride4": summarize(truth_mse_phi_phys),
            "peak_val_err_phi_stride1_sub4": summarize(s1_peak_val_phys),
            "peak_val_err_phi_stride4": summarize(s4_peak_val_phys),
        }

    summary_path = outdir / "same_physical_time_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[JSON] {summary_path}")

    print("[SUMMARY] normalized phi MSE")
    print("  stride1_sub4:", summary["normalized"]["mse_phi_stride1_sub4"])
    print("  stride4     :", summary["normalized"]["mse_phi_stride4"])
    print("  truth mismatch:", summary["normalized"]["truth_mse_phi_stride1_vs_stride4"])
    if phys_available:
        print("[SUMMARY] denormalized phi MSE")
        print("  stride1_sub4:", summary["denormalized"]["mse_phi_stride1_sub4"])
        print("  stride4     :", summary["denormalized"]["mse_phi_stride4"])
        print("  truth mismatch:", summary["denormalized"]["truth_mse_phi_stride1_vs_stride4"])


if __name__ == "__main__":
    main()
