import os
import glob
import h5py
import numpy as np
import torch

H5_PATH = r"C:\Users\astro\research\PEPAPIC\test\results\2D_ExB_high_magnet_dt2.5e-9_maxt50e-6\SimVPv2_inputs\global_norm_trainfixed_minmax_margin20_t0_to_t1000_TCHW.h5"
H5_KEY  = r"data_tchw"

DEFAULT_WORKDIR = r"C:\Users\astro\research\SimVPv2\workdirs\pepapic_simvp_gsta_highmag_trainfixed"

WORKDIR = os.environ.get("WORKDIR", DEFAULT_WORKDIR)
OUTDIR  = os.environ.get("OUTDIR", os.path.join(WORKDIR, "rollout_tp0_quads_assets"))
os.makedirs(OUTDIR, exist_ok=True)

TIN = 10
K   = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def find_latest(patterns, root):
    cands = []
    for p in patterns:
        cands += glob.glob(os.path.join(root, "**", p), recursive=True)
    if not cands:
        return None, []
    cands = sorted(cands, key=lambda x: os.path.getmtime(x), reverse=True)
    return cands[0], cands

def load_tchw_from_h5(path, key):
    with h5py.File(path, "r") as f:
        x = f[key][...]
    x = np.asarray(x)
    if x.ndim != 4:
        raise ValueError(f"expected (T,C,H,W) but got {x.shape}")
    return x

def split_frames_tchw(x_tchw):
    T = x_tchw.shape[0]
    n_tr = int(np.floor(0.8 * T))
    n_va = int(np.floor(0.1 * T))
    n_te = int(np.floor(0.1 * T))
    x_tr = x_tchw[:n_tr]
    x_va = x_tchw[n_tr:n_tr+n_va]
    x_te = x_tchw[n_tr+n_va:n_tr+n_va+n_te]
    return x_tr, x_va, x_te

def build_model_from_config_and_ckpt():
    import runpy
    from openstl.models.simvp_model import SimVP_Model

    # WORKDIR から config と ckpt を推定/指定できるようにする
    default_cfg = r"C:\Users\astro\research\SimVPv2\configs\custom\pepapic\SimVP_gSTA_pepapic.py"
    default_ckpt = os.path.join(WORKDIR, "checkpoints", "best.ckpt")

    CFG_PATH  = os.environ.get("CFG_PATH", default_cfg)
    CKPT_PATH = os.environ.get("CKPT_PATH", default_ckpt)

    print("[INFO] config:", CFG_PATH)
    print("[INFO] ckpt  :", CKPT_PATH)

    cfg = runpy.run_path(CFG_PATH)

    # ★学習時と同じ並び (T,C,H,W)
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

    # Lightning等で 'model.' prefix が付くことがあるので除去
    state_dict = {(k.replace("model.", "", 1) if k.startswith("model.") else k): v
                  for k, v in state_dict.items()}

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

    # 期待: (B,Tout,C,H,W)
    if hasattr(y, "ndim") and y.ndim == 5:
        return y[:, 0]          # (1,C,H,W)

    # まれに (Tout,C,H,W) を返す実装もある
    if hasattr(y, "ndim") and y.ndim == 4:
        return y[0:1]           # (1,C,H,W)

    raise RuntimeError(f"unexpected output shape: {getattr(y,'shape',None)}")


def main():
    data = load_tchw_from_h5(H5_PATH, H5_KEY).astype(np.float32)  # (1001,3,100,100)
    print("[INFO] data:", data.shape, data.dtype)

    tr, va, te = split_frames_tchw(data)
    print("[INFO] split:", tr.shape, va.shape, te.shape)

    seed = va[-TIN:]       # (10,C,H,W) from val last 10
    gt   = te[:K]          # (100,C,H,W) test

    model = build_model_from_config_and_ckpt()

    x = torch.from_numpy(seed).to(DEVICE)
    gt_t = torch.from_numpy(gt).to(DEVICE)

    inputs_roll, preds_roll, trues_roll = [], [], []

    for k in range(K):
        inputs_roll.append(x.detach().cpu().numpy())      # (10,C,H,W)
        y0 = forward_tp0(model, x)                        # (1,C,H,W)
        preds_roll.append(y0.detach().cpu().numpy())
        trues_roll.append(gt_t[k:k+1].detach().cpu().numpy())
        x = torch.cat([x[1:], y0], dim=0)
        if (k+1) % 10 == 0:
            print(f"[INFO] rollout {k+1}/{K}")

    inputs_roll = np.stack(inputs_roll, axis=0)  # (100,10,C,H,W)
    preds_roll  = np.stack(preds_roll,  axis=0)  # (100,1,C,H,W)
    trues_roll  = np.stack(trues_roll,  axis=0)  # (100,1,C,H,W)

    np.save(os.path.join(OUTDIR, "inputs_roll.npy"), inputs_roll)
    np.save(os.path.join(OUTDIR, "preds_roll.npy"),  preds_roll)
    np.save(os.path.join(OUTDIR, "trues_roll.npy"),  trues_roll)

    print("[DONE] saved to:", OUTDIR)
    print(" inputs_roll:", inputs_roll.shape)
    print(" preds_roll :", preds_roll.shape)
    print(" trues_roll :", trues_roll.shape)

if __name__ == "__main__":
    main()
