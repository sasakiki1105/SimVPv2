import os
import h5py
import numpy as np
import torch
import runpy

H5_PATH = r"C:\Users\astro\research\PEPAPIC\test\results\2D_ExB_high_magnet_dt2.5e-9_maxt50e-6_macro5\SimVPv2_inputs\global_norm_trainfixed_minmax_margin20_t0_to_t4000.h5"
H5_KEY  = r"data_tchw"

WORKDIR = r"C:\Users\astro\research\SimVPv2\workdirs\pepapic_simvp_gsta_highmag_macro5_trainfixed_disjoint_811_bs2_100ep"
CFG_PATH = r"C:\Users\astro\research\SimVPv2\configs\custom\pepapic\SimVP_gSTA_pepapic.py"
CKPT_PATH = os.path.join(WORKDIR, "checkpoints", "best.ckpt")

OUTDIR  = os.path.join(WORKDIR, "rollout_tp0_quads_assets")
os.makedirs(OUTDIR, exist_ok=True)

TIN = 10
K   = 100   # rollout length
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
    Frame-disjoint split matching the current dataloader logic:
      train: [0, 3200)
      val  : [3200, 3600)
      test : [3600, 4001)
    for T=4001
    """
    T = x_tchw.shape[0]
    train_end = int(np.floor(T * 0.8))   # 3200
    val_end   = int(np.floor(T * 0.9))   # 3600

    x_tr = x_tchw[:train_end]
    x_va = x_tchw[train_end:val_end]
    x_te = x_tchw[val_end:T]
    return x_tr, x_va, x_te


def build_model_from_config_and_ckpt():
    from openstl.models.simvp_model import SimVP_Model

    cfg = runpy.run_path(CFG_PATH)

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

    sd = torch.load(CKPT_PATH, map_location="cpu")
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


def main():
    data = load_tchw_from_h5(H5_PATH, H5_KEY)   # (4001,3,100,100)
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
    model = build_model_from_config_and_ckpt()

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

    np.save(os.path.join(OUTDIR, "inputs_roll.npy"), inputs_roll)
    np.save(os.path.join(OUTDIR, "preds_roll.npy"),  preds_roll)
    np.save(os.path.join(OUTDIR, "trues_roll.npy"),  trues_roll)

    print("[DONE] saved to:", OUTDIR)
    print(" inputs_roll:", inputs_roll.shape)
    print(" preds_roll :", preds_roll.shape)
    print(" trues_roll :", trues_roll.shape)


if __name__ == "__main__":
    main()