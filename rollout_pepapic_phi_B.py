# rollout_pepapic_phi_B.py (FULL REPLACE)
import os
import argparse
import datetime
import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt

# ----------------------------
# utilities
# ----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def guess_and_to_TCHW(arr: np.ndarray) -> np.ndarray:
    """
    Convert common H5 layouts to (T,C,H,W) float32.
    Accepts:
      - (T,C,H,W) -> ok
      - (H,W,C,T) -> (T,C,H,W)
      - (C,H,W,T) -> (T,C,H,W)
      - (T,H,W,C) -> (T,C,H,W)
    """
    a = np.asarray(arr)
    if a.ndim != 4:
        raise ValueError(f"Expected 4D array in H5, got shape={a.shape}")
    s = a.shape

    # (T,C,H,W)
    if s[0] >= 20 and s[1] <= 20 and s[2] >= 16 and s[3] >= 16:
        return a.astype(np.float32, copy=False)

    # (H,W,C,T)
    if s[0] >= 16 and s[1] >= 16 and s[2] <= 20 and s[3] >= 20:
        return np.transpose(a, (3, 2, 0, 1)).astype(np.float32, copy=False)

    # (C,H,W,T)
    if s[0] <= 20 and s[1] >= 16 and s[2] >= 16 and s[3] >= 20:
        return np.transpose(a, (3, 0, 1, 2)).astype(np.float32, copy=False)

    # (T,H,W,C)
    if s[0] >= 20 and s[1] >= 16 and s[2] >= 16 and s[3] <= 20:
        return np.transpose(a, (0, 3, 1, 2)).astype(np.float32, copy=False)

    raise ValueError(f"Could not infer layout from shape={s}. Please convert to (T,C,H,W).")

def save_quad_png(path, in_last, true, pred, title, cmap="viridis", scale_mode="true_p1p99"):
    """
    in_last/true/pred: (H,W)
    scale_mode:
      - true_minmax
      - true_p1p99  (recommended)
      - panel       (each panel auto scale)
    """
    err = np.abs(pred - true)

    if scale_mode == "panel":
        vmin = vmax = None
    elif scale_mode == "true_minmax":
        vmin = float(true.min())
        vmax = float(true.max())
    elif scale_mode == "true_p1p99":
        vmin, vmax = np.percentile(true, [1, 99]).astype(float)
    else:
        raise ValueError(f"unknown scale_mode={scale_mode}")

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)

    if scale_mode == "panel":
        axes[0].imshow(in_last, **im_kw); axes[0].set_title("Input (last)")
        axes[1].imshow(true,    **im_kw); axes[1].set_title("True")
        axes[2].imshow(pred,    **im_kw); axes[2].set_title("Pred")
    else:
        im_kw2 = dict(**im_kw, vmin=vmin, vmax=vmax)
        axes[0].imshow(in_last, **im_kw2); axes[0].set_title("Input (last)")
        axes[1].imshow(true,    **im_kw2); axes[1].set_title("True")
        axes[2].imshow(pred,    **im_kw2); axes[2].set_title("Pred")

    evmin = float(err.min())
    evmax = float(err.max())
    axes[3].imshow(err, vmin=evmin, vmax=evmax, cmap=cmap)
    axes[3].set_title("|Error|")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)

# ----------------------------
# model loader (OpenSTL/SimVPv2)
# ----------------------------
def build_experiment_and_get_method(simvp_dir: str,
                                    config_file: str,
                                    dataname: str,
                                    data_root: str,
                                    pre_seq_length: int,
                                    aft_seq_length: int,
                                    ckpt_path: str,
                                    device_str: str = "cuda"):
    """
    Build OpenSTL Method exactly like test does and load ckpt.
    Keep args.in_shape=None (auto) and args.model_type='gSTA' as training.
    """
    from openstl.api.exp import BaseExperiment
    from openstl.utils.parser import create_parser

    parser = create_parser()
    args = parser.parse_args([
        "-d", dataname,
        "--method", "SimVP",
        "-c", config_file,
        "--data_root", data_root,
        "--pre_seq_length", str(pre_seq_length),
        "--aft_seq_length", str(aft_seq_length),
        "--num_workers", "0",
        "--batch_size", "1",
        "--val_batch_size", "1",
        "--res_dir", os.path.join(simvp_dir, "workdirs"),
        "--ex_name", "_tmp_rollout_build",
        "--ckpt_path", ckpt_path,
        "--device", device_str,
        "--test",
        "--no_display_method_info",
    ])

    # match training
    args.in_shape = None
    args.model_type = "gSTA"
    print(f"[INFO] in_shape=None (auto), model_type={args.model_type}")

    exp = BaseExperiment(args)
    method = exp.method
    method.eval()
    return method

@torch.no_grad()
def predict_next_1step(method, x_btchw: torch.Tensor) -> torch.Tensor:
    """
    x_btchw: (B,Tin,C,H,W)
    return:  (B,C,H,W) first future frame
    """
    out = method(x_btchw)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if out.ndim != 5:
        raise RuntimeError(f"Unexpected model output shape: {tuple(out.shape)}")
    return out[:, 0]  # (B,C,H,W)

# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=r"configs\custom\pepapic\SimVP_gSTA_pepapic.py")
    ap.add_argument("--dataname", default="pepapic_h5")
    ap.add_argument("--pre", type=int, default=10)
    ap.add_argument("--aft", type=int, default=10)
    ap.add_argument("--start", type=int, default=903)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--phi_c", type=int, default=2)
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--scale_mode", default="true_p1p99", choices=["true_p1p99", "true_minmax", "panel"])
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--clip01", action="store_true",
                    help="clip prediction to [0,1] before saving and before sliding window (recommended for minmax-normalized data)")
    args = ap.parse_args()

    simvp_dir = os.path.abspath(os.path.dirname(__file__))

    # output dir
    if args.out_root is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = os.path.join(simvp_dir, f"viz_rollout_phi_B_trainfixed_{stamp}")
    else:
        out_root = os.path.abspath(args.out_root)
    ensure_dir(out_root)

    # choose device
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # load H5 -> data (T,C,H,W)
    with h5py.File(args.h5, "r") as f:
        if "data_tchw" in f:
            raw = f["data_tchw"][:]
        elif "data" in f:
            raw = f["data"][:]
        else:
            raise KeyError("H5 must contain 'data_tchw' or 'data'")

        data = guess_and_to_TCHW(raw)  # (T,C,H,W)
        T, C, H, W = data.shape

        props = None
        if "props" in f:
            props_raw = f["props"][:]
            props = [p.decode() if isinstance(p, (bytes, bytearray, np.bytes_)) else str(p) for p in props_raw]

    pre = args.pre
    assert 0 <= args.phi_c < C, f"phi_c={args.phi_c} out of range. C={C}"
    assert args.start >= 0
    assert args.start + pre < T, f"Need start+pre < T. start={args.start}, pre={pre}, T={T}"

    max_horizon = T - (args.start + pre)
    horizon = max_horizon if args.horizon is None else min(args.horizon, max_horizon)

    # build model method + move to device
    method = build_experiment_and_get_method(
        simvp_dir=simvp_dir,
        config_file=args.config,
        dataname=args.dataname,
        data_root=args.h5,
        pre_seq_length=args.pre,
        aft_seq_length=args.aft,
        ckpt_path=args.ckpt,
        device_str=args.device,
    )
    method = method.to(device)
    method.eval()

    # seed window (pre,C,H,W)
    window = data[args.start:args.start + pre].copy()

    # meta
    meta_path = os.path.join(out_root, "rollout_meta.txt")
    with open(meta_path, "w", encoding="utf-8") as w:
        w.write(f"h5={args.h5}\n")
        w.write(f"ckpt={args.ckpt}\n")
        w.write(f"config={args.config}\n")
        w.write(f"dataname={args.dataname}\n")
        w.write(f"data_shape(T,C,H,W)={data.shape}\n")
        w.write(f"props={props}\n")
        w.write(f"start={args.start}\n")
        w.write(f"pre={pre}\n")
        w.write(f"horizon={horizon}\n")
        w.write(f"phi_c={args.phi_c}\n")
        w.write(f"scale_mode={args.scale_mode}\n")
        w.write(f"clip01={args.clip01}\n")
        w.write(f"Interpretation: seed uses t=[start..start+pre-1], true compares against t=[start+pre .. start+pre+horizon-1]\n")

    print(f"[INFO] data (T,C,H,W)={data.shape} props={props}")
    print(f"[INFO] seed start={args.start}, pre={pre}, horizon={horizon} (true t={args.start+pre}..{args.start+pre+horizon-1})")
    print(f"[INFO] saving -> {out_root}")
    if args.clip01:
        print("[INFO] clip01 enabled: pred clipped to [0,1] before viz and before sliding window")

    # rollout
    for k in range(horizon):
        x = torch.from_numpy(window[None]).to(device=device, dtype=torch.float32)  # (1,pre,C,H,W)
        y1 = predict_next_1step(method, x)  # (1,C,H,W)

        if k == 0:
            print("[DBG] x:", tuple(x.shape),
                  "min", float(x.min()), "max", float(x.max()),
                  "mean", float(x.mean()), "std", float(x.std()))
            print("[DBG] y1:", tuple(y1.shape),
                  "min", float(y1.min()), "max", float(y1.max()),
                  "mean", float(y1.mean()), "std", float(y1.std()))

        y1_np = y1[0].detach().cpu().numpy()  # (C,H,W)

        # ---- IMPORTANT: clip if your data is minmax-normalized ----
        if args.clip01:
            y1_np = np.clip(y1_np, 0.0, 1.0)

        t_true = args.start + pre + k
        y_true = data[t_true]  # (C,H,W)

        c = args.phi_c
        in_last = window[-1, c]
        pred_phi = y1_np[c]
        true_phi = y_true[c]

        if k in [0, 1, 2]:
            print(f"[DBG] k={k} t={t_true} true(std)={true_phi.std():.4g} pred(std)={pred_phi.std():.4g}")

        fn = f"t{t_true:04d}_step{k+1:03d}_phi_quad.png"
        save_quad_png(
            os.path.join(out_root, fn),
            in_last, true_phi, pred_phi,
            f"rollout step {k+1}/{horizon} | abs t={t_true} | phi (c={c})",
            cmap=args.cmap,
            scale_mode=args.scale_mode
        )

        # slide window
        window = np.concatenate([window[1:], y1_np[None]], axis=0)

    print("[DONE] rollout pngs saved:", out_root)
    print("[DONE] meta:", meta_path)

if __name__ == "__main__":
    main()
