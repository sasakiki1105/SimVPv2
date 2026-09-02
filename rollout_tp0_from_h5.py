import argparse
import csv
import json
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import runpy

H5_PATH = r"C:\Users\astro\research\PEPAPIC\test\results\2D_ExB_high_magnet_dt2.5e-9_maxt50e-6_macro5\SimVPv2_inputs\global_norm_trainfixed_minmax_margin20_t0_to_t4000_step4.h5"
H5_KEY  = r"data_tchw"

WORKDIR = r"C:\Users\astro\research\SimVPv2\workdirs\pepapic_simvp_gsta_highmag_macro5_subsample4_trainfixed_disjoint_811_bs2_100ep"
CFG_PATH = r"C:\Users\astro\research\SimVPv2\configs\custom\pepapic\SimVP_gSTA_pepapic.py"
CKPT_PATH = os.path.join(WORKDIR, "checkpoints", "best.ckpt")

OUTDIR  = os.path.join(WORKDIR, "rollout_tp0_quads_assets")

TIN = 10
C_TARGET = 2
DT_NS = 50.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_tchw_from_h5(path, key):
    with h5py.File(path, "r") as f:
        x = f[key][...]
    x = np.asarray(x)

    if x.ndim != 4:
        raise ValueError(f"expected 4D array, got {x.shape}")

    s = x.shape

    # already (T,C,H,W)
    if s[0] >= 20 and s[1] <= 20 and s[2] >= 16 and s[3] >= 16:
        out = x

    # (H,W,C,T) -> (T,C,H,W)
    elif s[0] >= 16 and s[1] >= 16 and s[2] <= 20 and s[3] >= 20:
        out = np.transpose(x, (3, 2, 0, 1))

    # (C,H,W,T) -> (T,C,H,W)
    elif s[0] <= 20 and s[1] >= 16 and s[2] >= 16 and s[3] >= 20:
        out = np.transpose(x, (3, 0, 1, 2))

    # (T,H,W,C) -> (T,C,H,W)
    elif s[0] >= 20 and s[1] >= 16 and s[2] >= 16 and s[3] <= 20:
        out = np.transpose(x, (0, 3, 1, 2))

    else:
        raise ValueError(f"cannot infer layout from shape {s}")

    out = np.ascontiguousarray(out.astype(np.float32))
    print(f"[INFO] raw H5 shape={s} -> TCHW shape={out.shape}")
    return out


def split_frames_tchw_disjoint_811(x_tchw, total=20):
    """
    Frame-disjoint split matching the current dataloader logic.
    For T=1001:
      train: [0, 800)
      val  : [800, 900)
      test : [900, 1001)
    """
    T = x_tchw.shape[0]
    train_end = int(np.floor(T * 0.8))
    val_end   = int(np.floor(T * 0.9))

    x_tr = x_tchw[:train_end]
    x_va = x_tchw[train_end:val_end]
    x_te = x_tchw[val_end:T]
    return x_tr, x_va, x_te


def build_model_from_config_and_ckpt(cfg_path, ckpt_path):
    from openstl.models.simvp_model import SimVP_Model

    cfg = runpy.run_path(cfg_path)

    in_shape = (TIN, 3, 100, 100)

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
    )

    sd = torch.load(ckpt_path, map_location="cpu")
    state_dict = sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd
    state_dict = {
        (k.replace("model.", "", 1) if k.startswith("model.") else k): v
        for k, v in state_dict.items()
    }

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[INFO] load_state_dict strict=False: missing={len(missing)} unexpected={len(unexpected)}")

    model.to(DEVICE).eval()
    return model


@torch.no_grad()
def forward_tp0(model, x_10chw):
    xin = x_10chw.unsqueeze(0)  # (1,10,C,H,W)
    y = model(xin)

    if isinstance(y, (list, tuple)):
        y = y[0]

    if hasattr(y, "ndim") and y.ndim == 5:
        return y[:, 0]   # (1,C,H,W)

    if hasattr(y, "ndim") and y.ndim == 4:
        return y[0:1]    # (1,C,H,W)

    raise RuntimeError(f"unexpected output shape: {getattr(y,'shape',None)}")


def argmax_2d(a2d):
    idx = int(np.argmax(a2d))
    return np.unravel_index(idx, a2d.shape)


def compute_phi_metrics(preds_roll, trues_roll):
    pred_phi = preds_roll[:, 0, C_TARGET].astype(np.float64)
    true_phi = trues_roll[:, 0, C_TARGET].astype(np.float64)
    mse = np.mean((pred_phi - true_phi) ** 2, axis=(1, 2))
    peak_val_err = np.zeros(len(mse), dtype=np.float64)
    peak_loc_err = np.zeros(len(mse), dtype=np.float64)

    for k in range(len(mse)):
        p = pred_phi[k]
        t = true_phi[k]
        peak_val_err[k] = abs(float(p.max()) - float(t.max()))
        py, px = argmax_2d(p)
        ty, tx = argmax_2d(t)
        peak_loc_err[k] = float(((py - ty) ** 2 + (px - tx) ** 2) ** 0.5)

    return mse, peak_val_err, peak_loc_err


def save_line_plot(x, y, xlabel, ylabel, title, out_png):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print("[PLOT]", out_png)


def save_metrics_csv(out_csv, mse, peak_val_err, peak_loc_err, dt_ns):
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_index",
            "time_us",
            "mse_phi",
            "peak_val_err_phi",
            "peak_loc_err_phi_px",
        ])
        for k in range(len(mse)):
            writer.writerow([
                k,
                k * dt_ns / 1000.0,
                float(mse[k]),
                float(peak_val_err[k]),
                float(peak_loc_err[k]),
            ])
    print("[CSV]", out_csv)


def main():
    ap = argparse.ArgumentParser(
        description="Closed-loop rollout that feeds back only the first predicted frame."
    )
    ap.add_argument("--h5-path", default=H5_PATH)
    ap.add_argument("--h5-key", default=H5_KEY)
    ap.add_argument("--workdir", default=WORKDIR)
    ap.add_argument("--config-path", default=CFG_PATH)
    ap.add_argument("--ckpt-path", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--dt-ns", type=float, default=DT_NS)
    args = ap.parse_args()

    workdir = args.workdir
    ckpt_path = args.ckpt_path or os.path.join(workdir, "checkpoints", "best.ckpt")
    outdir = args.outdir or os.path.join(workdir, "rollout_tp0_quads_assets")
    os.makedirs(outdir, exist_ok=True)

    data = load_tchw_from_h5(args.h5_path, args.h5_key)
    print("[INFO] data:", data.shape, data.dtype)

    tr, va, te = split_frames_tchw_disjoint_811(data, total=TIN+TIN)
    print("[INFO] split:", tr.shape, va.shape, te.shape)

    # current training split:
    # val = t 3200..3599 (400 frames)
    # test = t 3600..4000 (401 frames)
    #
    # closed-loop seed: val last 10 = t 3590..3599
    # ground truth: test first K frames = t 3600..3699
    seed = va[-TIN:]      # (10,C,H,W)

    # 最大 rollout 長は test の長さ
    K = te.shape[0]       # 今回は 401
    gt = te[:K]           # (K,C,H,W)
    model = build_model_from_config_and_ckpt(args.config_path, ckpt_path)

    x = torch.from_numpy(seed).to(DEVICE)
    gt_t = torch.from_numpy(gt).to(DEVICE)

    inputs_roll, preds_roll, trues_roll = [], [], []

    for k in range(K):
        inputs_roll.append(x.detach().cpu().numpy())     # (10,C,H,W)
        y0 = forward_tp0(model, x)                       # (1,C,H,W)
        preds_roll.append(y0.detach().cpu().numpy())     # (1,C,H,W)
        trues_roll.append(gt_t[k:k+1].detach().cpu().numpy())
        x = torch.cat([x[1:], y0], dim=0)

        if (k + 1) % 10 == 0:
            print(f"[INFO] rollout {k+1}/{K}")

    inputs_roll = np.stack(inputs_roll, axis=0)   # (K,10,C,H,W)
    preds_roll  = np.stack(preds_roll,  axis=0)   # (K,1,C,H,W)
    trues_roll  = np.stack(trues_roll,  axis=0)   # (K,1,C,H,W)

    np.save(os.path.join(outdir, "inputs_roll.npy"), inputs_roll)
    np.save(os.path.join(outdir, "preds_roll.npy"),  preds_roll)
    np.save(os.path.join(outdir, "trues_roll.npy"),  trues_roll)

    mse, peak_val_err, peak_loc_err = compute_phi_metrics(preds_roll, trues_roll)
    x_iter = np.arange(len(mse), dtype=np.float64)
    x_time = x_iter * args.dt_ns / 1000.0

    save_line_plot(
        x_iter,
        mse,
        "Rollout frame index",
        "MSE (phi)",
        "TP0 rollout MSE vs frame index",
        os.path.join(outdir, "mse_curve_phi.png"),
    )
    save_line_plot(
        x_time,
        mse,
        "Physical time since rollout start (us)",
        "MSE (phi)",
        "TP0 rollout MSE vs physical time",
        os.path.join(outdir, "mse_curve_phi_time_us.png"),
    )
    save_line_plot(
        x_time,
        peak_val_err,
        "Physical time since rollout start (us)",
        "Peak value error (abs)",
        "TP0 rollout peak value error (phi)",
        os.path.join(outdir, "peak_val_err_phi_time_us.png"),
    )
    save_line_plot(
        x_time,
        peak_loc_err,
        "Physical time since rollout start (us)",
        "Peak location error (pixels)",
        "TP0 rollout peak location error (phi)",
        os.path.join(outdir, "peak_loc_err_phi_time_us.png"),
    )
    save_metrics_csv(
        os.path.join(outdir, "rollout_metrics_phi.csv"),
        mse,
        peak_val_err,
        peak_loc_err,
        args.dt_ns,
    )

    metadata = {
        "rollout_type": "tp0",
        "description": "Predict 10 retained frames, feed back only the first predicted frame as the newest input frame.",
        "h5_path": args.h5_path,
        "h5_key": args.h5_key,
        "workdir": workdir,
        "ckpt_path": ckpt_path,
        "outdir": outdir,
        "tin": TIN,
        "tout_used_per_step": 1,
        "dt_ns_per_retained_frame": args.dt_ns,
        "device": DEVICE,
        "split_shapes": {
            "train": list(tr.shape),
            "val": list(va.shape),
            "test": list(te.shape),
        },
        "seed_frame_range_in_h5": [data.shape[0] - te.shape[0] - TIN, data.shape[0] - te.shape[0] - 1],
        "test_frame_count": int(K),
        "saved_shapes": {
            "inputs_roll": list(inputs_roll.shape),
            "preds_roll": list(preds_roll.shape),
            "trues_roll": list(trues_roll.shape),
        },
        "metrics_phi": {
            "mse_first": float(mse[0]),
            "mse_last": float(mse[-1]),
            "mse_mean": float(np.mean(mse)),
            "mse_max": float(np.max(mse)),
            "peak_val_err_mean": float(np.mean(peak_val_err)),
            "peak_loc_err_mean_px": float(np.mean(peak_loc_err)),
        },
    }
    with open(os.path.join(outdir, "rollout_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("[DONE] saved to:", outdir)
    print(" inputs_roll:", inputs_roll.shape)
    print(" preds_roll :", preds_roll.shape)
    print(" trues_roll :", trues_roll.shape)


if __name__ == "__main__":
    main()
