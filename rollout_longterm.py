import os
import csv
import argparse
import importlib.util
import numpy as np
import torch
import h5py

# -----------------------
# utils
# -----------------------
def load_config_py(path: str) -> dict:
    """Load a python config file as a dict (variables in global scope)."""
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("cfg_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    d = {k: getattr(mod, k) for k in dir(mod) if not k.startswith("_")}
    return d

def ensure_tchw(arr: np.ndarray) -> np.ndarray:
    """
    Accept common H5 layouts and return (T,C,H,W) float32 contiguous.
    Supported:
      - (T,C,H,W) -> ok
      - (H,W,C,T) -> transpose to (T,C,H,W)
      - (C,H,W,T) -> transpose to (T,C,H,W)
      - (T,H,W,C) -> transpose to (T,C,H,W)
    """
    a = np.asarray(arr)
    if a.ndim != 4:
        raise ValueError(f"data must be 4D, got {a.shape}")

    # Heuristics by dimension sizes
    sh = a.shape

    # If first dim looks like time (>= 50) and second dim looks like channels (<= 20)
    if sh[0] >= 20 and sh[1] <= 20 and sh[2] >= 16 and sh[3] >= 16:
        out = a  # (T,C,H,W)
        return np.ascontiguousarray(out.astype(np.float32))

    # (H,W,C,T)
    if sh[0] >= 16 and sh[1] >= 16 and sh[2] <= 20 and sh[3] >= 20:
        out = np.transpose(a, (3, 2, 0, 1))
        return np.ascontiguousarray(out.astype(np.float32))

    # (C,H,W,T)
    if sh[0] <= 20 and sh[1] >= 16 and sh[2] >= 16 and sh[3] >= 20:
        out = np.transpose(a, (3, 0, 1, 2))
        return np.ascontiguousarray(out.astype(np.float32))

    # (T,H,W,C)
    if sh[0] >= 20 and sh[1] >= 16 and sh[2] >= 16 and sh[3] <= 20:
        out = np.transpose(a, (0, 3, 1, 2))
        return np.ascontiguousarray(out.astype(np.float32))

    raise ValueError(f"cannot infer layout for data shape {sh}")

def make_starts(T: int, pre: int, aft: int, stride: int):
    seq_len = pre + aft
    starts = list(range(0, T - seq_len + 1, stride))
    if len(starts) <= 0:
        raise ValueError(f"No samples: T={T}, pre={pre}, aft={aft}, stride={stride}")
    return starts

def split_indices(starts, train_ratio: float, val_ratio: float):
    n = len(starts)
    n_train = max(1, int(n * train_ratio))
    n_val = int(n * val_ratio)
    if n - n_train - n_val < 1:
        n_val = max(0, n - n_train - 1)

    train_idx = starts[:n_train]
    val_idx   = starts[n_train:n_train+n_val] if n_val > 0 else []
    test_idx  = starts[n_train+n_val:] if (n_train+n_val) < n else starts[-1:]
    return train_idx, val_idx, test_idx

def safe_makedirs(p: str):
    os.makedirs(p, exist_ok=True)

def save_csv(path, steps, mse, mae, mse_b, mae_b):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "mse_model", "mae_model", "mse_persist", "mae_persist"])
        for i in range(len(steps)):
            w.writerow([int(steps[i]), float(mse[i]), float(mae[i]), float(mse_b[i]), float(mae_b[i])])

def try_plot_png(path, steps, mse, mae, mse_b, mae_b):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    # MSE
    plt.figure()
    plt.plot(steps, mse, label="model")
    plt.plot(steps, mse_b, label="persistence")
    plt.xlabel("rollout step")
    plt.ylabel("pixel-mean MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path.replace(".png", "_mse.png"), dpi=150)
    plt.close()

    # MAE
    plt.figure()
    plt.plot(steps, mae, label="model")
    plt.plot(steps, mae_b, label="persistence")
    plt.xlabel("rollout step")
    plt.ylabel("pixel-mean MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path.replace(".png", "_mae.png"), dpi=150)
    plt.close()

# -----------------------
# model loader (robust-ish)
# -----------------------
def load_simvp_lightning(ckpt_path: str, cfg: dict, device: str):
    """
    Try to load OpenSTL SimVP LightningModule from checkpoint.
    If it fails, raise with a helpful message.
    """
    ckpt_path = os.path.abspath(ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    # Prefer Lightning load_from_checkpoint
    last_err = None
    try:
        from openstl.methods.simvp import SimVP  # typical location in OpenSTL
        try:
            m = SimVP.load_from_checkpoint(ckpt_path, map_location=device)
        except TypeError:
            # older lightning versions
            m = SimVP.load_from_checkpoint(ckpt_path)
        m.eval()
        m.to(device)
        return m
    except Exception as e:
        last_err = e

    # Fallback: instantiate and load state_dict
    try:
        from openstl.methods.simvp import SimVP
        m = SimVP(**cfg)
        sd = torch.load(ckpt_path, map_location="cpu")
        state = sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd
        # lightning prefixes sometimes
        try:
            m.load_state_dict(state, strict=False)
        except Exception:
            # strip "model." prefix if needed
            new_state = {}
            for k, v in state.items():
                nk = k
                if nk.startswith("model."):
                    nk = nk[len("model."):]
                new_state[nk] = v
            m.load_state_dict(new_state, strict=False)
        m.eval()
        m.to(device)
        return m
    except Exception as e2:
        raise RuntimeError(
            "Failed to load SimVP from checkpoint.\n"
            f"First error: {last_err}\n"
            f"Second error: {e2}\n"
            "If this happens, paste the top of your checkpoint loading stacktrace and "
            "I’ll tailor the loader to your OpenSTL version."
        )

# -----------------------
# main rollout
# -----------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="best.ckpt path")
    ap.add_argument("--config", required=True, help="SimVP_gSTA_pepapic.py path")
    ap.add_argument("--data_root", required=True, help="H5 path")
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--pre", type=int, default=10)
    ap.add_argument("--aft", type=int, default=10)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)

    ap.add_argument("--rollout", type=int, default=200, help="K steps (1-step each) to roll out")
    ap.add_argument("--tp_use", type=int, default=0, help="which predicted tp to feed back (0 means 1-step)")
    ap.add_argument("--channel", type=int, default=2, help="evaluate only this channel (phi=2)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--nmax", type=int, default=0, help="0=all test samples, else limit")
    args = ap.parse_args()

    safe_makedirs(args.outdir)

    # ---- load H5
    if not os.path.exists(args.data_root):
        raise FileNotFoundError(args.data_root)

    with h5py.File(args.data_root, "r") as f:
        if "data_tchw" in f:
            raw = f["data_tchw"][:]
        elif "data" in f:
            raw = f["data"][:]
        else:
            raise KeyError("H5 must contain 'data_tchw' or 'data'")

    data = ensure_tchw(raw)  # (T,C,H,W)
    T, C, H, W = data.shape
    if not (0 <= args.channel < C):
        raise ValueError(f"channel {args.channel} out of range (C={C})")

    # ---- starts/split (same logic as your dataloader)
    starts = make_starts(T, args.pre, args.aft, args.stride)
    tr_idx, va_idx, te_idx = split_indices(starts, args.train_ratio, args.val_ratio)

    N_test_all = len(te_idx)
    N_test = N_test_all if args.nmax == 0 else min(args.nmax, N_test_all)
    te_idx = te_idx[:N_test]

    # ---- load config & model
    cfg_all = load_config_py(args.config)

    # Prepare minimal cfg for method init if fallback path is used
    # (load_from_checkpoint path usually ignores this)
    cfg_min = {}
    for k in ["method","model_type","hid_S","hid_T","N_S","N_T","spatio_kernel_enc","spatio_kernel_dec",
              "drop_path","drop","metrics","pre_seq_length","aft_seq_length","in_shape"]:
        if k in cfg_all:
            cfg_min[k] = cfg_all[k]

    # For SimVP in OpenSTL, in_shape should be (pre, C, H, W)
    cfg_min["pre_seq_length"] = int(args.pre)
    cfg_min["aft_seq_length"] = int(args.aft)
    cfg_min["in_shape"] = (int(args.pre), int(C), int(H), int(W))
    if "method" not in cfg_min:
        cfg_min["method"] = "SimVP"
    if "model_type" not in cfg_min:
        cfg_min["model_type"] = "gSTA"

    model = load_simvp_lightning(args.ckpt, cfg_min, args.device)

    # ---- rollout metrics buffers (pixel-mean)
    K = int(args.rollout)
    steps = np.arange(1, K+1, dtype=np.int32)
    mse_curve = np.zeros(K, dtype=np.float64)
    mae_curve = np.zeros(K, dtype=np.float64)
    mse_base  = np.zeros(K, dtype=np.float64)
    mae_base  = np.zeros(K, dtype=np.float64)

    # ---- rollout for each test start
    # We use 1-step target: GT = data[start + pre + step]
    # constraint: start + pre + K must be within T
    valid_te = []
    for s in te_idx:
        if s + args.pre + K < T:
            valid_te.append(s)
    if len(valid_te) == 0:
        raise RuntimeError(f"No valid test starts for rollout={K}. "
                           f"Try smaller --rollout. Max possible is roughly {T - args.pre - te_idx[0] - 1}.")

    print(f"[INFO] data (T,C,H,W)=({T},{C},{H},{W})")
    print(f"[INFO] starts n={len(starts)} train={len(tr_idx)} val={len(va_idx)} test={N_test_all} (using {len(valid_te)})")
    print(f"[INFO] rollout K={K}, tp_use={args.tp_use}, eval channel={args.channel}, device={args.device}")

    # batch size: keep small to avoid VRAM issues; you can tune
    B = 8
    for i0 in range(0, len(valid_te), B):
        batch_starts = valid_te[i0:i0+B]
        bsz = len(batch_starts)

        # init x: (B, pre, C, H, W)
        x = np.stack([data[s:s+args.pre] for s in batch_starts], axis=0).astype(np.float32)
        # baseline input copy
        x_base = x.copy()

        # convert to torch
        xt = torch.from_numpy(x).to(args.device)           # (B,pre,C,H,W)
        xt_base = torch.from_numpy(x_base).to(args.device) # (B,pre,C,H,W)

        for step in range(K):
            # --- model prediction (expects B,T,C,H,W)
            out = model(xt)  # should be (B, aft, C, H, W)
            if isinstance(out, (list, tuple)):
                out = out[0]
            if out.ndim != 5:
                raise RuntimeError(f"model output must be 5D (B,T,C,H,W), got {tuple(out.shape)}")

            # pick tp to feed back
            tp = int(args.tp_use)
            if tp < 0 or tp >= out.shape[1]:
                raise ValueError(f"--tp_use {tp} out of range for Tout={out.shape[1]}")

            yhat_1 = out[:, tp]  # (B,C,H,W)
            # GT 1-step at this rollout step
            gt = torch.from_numpy(
                np.stack([data[s+args.pre+step] for s in batch_starts], axis=0).astype(np.float32)
            ).to(args.device)  # (B,C,H,W)

            # compute errors (phi only)
            ch = int(args.channel)
            diff = (yhat_1[:, ch] - gt[:, ch])
            mse_curve[step] += float(torch.mean(diff * diff).item()) * bsz
            mae_curve[step] += float(torch.mean(torch.abs(diff)).item()) * bsz

            # --- baseline persistence: copy last input frame forward
            ybase_1 = xt_base[:, -1]  # (B,C,H,W)
            diffb = (ybase_1[:, ch] - gt[:, ch])
            mse_base[step] += float(torch.mean(diffb * diffb).item()) * bsz
            mae_base[step] += float(torch.mean(torch.abs(diffb)).item()) * bsz

            # shift window by 1 (feed back)
            xt = torch.cat([xt[:, 1:], yhat_1[:, None, ...]], dim=1)
            xt_base = torch.cat([xt_base[:, 1:], ybase_1[:, None, ...]], dim=1)

    # average over samples
    denom = float(len(valid_te))
    mse_curve /= denom
    mae_curve /= denom
    mse_base  /= denom
    mae_base  /= denom

    # save
    np.save(os.path.join(args.outdir, "steps.npy"), steps)
    np.save(os.path.join(args.outdir, "mse_model.npy"), mse_curve)
    np.save(os.path.join(args.outdir, "mae_model.npy"), mae_curve)
    np.save(os.path.join(args.outdir, "mse_persist.npy"), mse_base)
    np.save(os.path.join(args.outdir, "mae_persist.npy"), mae_base)

    csv_path = os.path.join(args.outdir, "curves.csv")
    save_csv(csv_path, steps, mse_curve, mae_curve, mse_base, mae_base)
    try_plot_png(os.path.join(args.outdir, "curves.png"), steps, mse_curve, mae_curve, mse_base, mae_base)

    # print summary
    print("[DONE] saved:", os.path.abspath(args.outdir))
    print("[SUMMARY] step=1:  mse(model)=%.6g  mae(model)=%.6g | mse(persist)=%.6g  mae(persist)=%.6g" %
          (mse_curve[0], mae_curve[0], mse_base[0], mae_base[0]))
    print("[SUMMARY] step=%d: mse(model)=%.6g  mae(model)=%.6g | mse(persist)=%.6g  mae(persist)=%.6g" %
          (K, mse_curve[-1], mae_curve[-1], mse_base[-1], mae_base[-1]))

if __name__ == "__main__":
    main()
