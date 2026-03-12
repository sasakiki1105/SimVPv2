import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def to_5d(x, name="array"):
    """
    Convert common shapes to (N, T, C, H, W).
    Supports:
      - (N, T, C, H, W)  -> ok
      - (T, C, H, W)     -> N=1
      - (N, C, T, H, W)  -> swap to N,T,C,H,W
      - (C, T, H, W)     -> N=1, swap to T,C,H,W
      - (N, T, H, W, C)  -> move C
      - (T, H, W, C)     -> N=1
    """
    x = np.asarray(x)
    if x.ndim == 5:
        # N,T,C,H,W
        if x.shape[2] <= 20 and x.shape[3] >= 16 and x.shape[4] >= 16:
            return x
        # N,C,T,H,W
        if x.shape[1] <= 20 and x.shape[3] >= 16 and x.shape[4] >= 16:
            return np.transpose(x, (0, 2, 1, 3, 4))
        # N,T,H,W,C
        if x.shape[4] <= 20 and x.shape[2] >= 16 and x.shape[3] >= 16:
            return np.transpose(x, (0, 1, 4, 2, 3))
        raise ValueError(f"{name}: unsupported 5D shape {x.shape}")
    elif x.ndim == 4:
        # (T,C,H,W)
        if x.shape[1] <= 20 and x.shape[2] >= 16 and x.shape[3] >= 16:
            return x[None, ...]
        # (C,T,H,W)
        if x.shape[0] <= 20 and x.shape[2] >= 16 and x.shape[3] >= 16:
            return np.transpose(x, (1, 0, 2, 3))[None, ...]
        # (T,H,W,C)
        if x.shape[3] <= 20 and x.shape[1] >= 16 and x.shape[2] >= 16:
            return np.transpose(x, (0, 3, 1, 2))[None, ...]
        raise ValueError(f"{name}: unsupported 4D shape {x.shape}")
    else:
        raise ValueError(f"{name}: expected 4D or 5D, got shape {x.shape}")

def parse_int_list(s):
    return [int(x) for x in s.split(",") if x.strip() != ""]

def save_one(fig_path, in_last, true, pred, title, cmap="viridis"):
    err = np.abs(pred - true)

    # same scale for input/true/pred
    vmin = float(np.min([in_last.min(), true.min(), pred.min()]))
    vmax = float(np.max([in_last.max(), true.max(), pred.max()]))

    # error scale per-image
    evmin = float(err.min())
    evmax = float(err.max())

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    axes[0].imshow(in_last, vmin=vmin, vmax=vmax, cmap=cmap)
    axes[0].set_title("Input (last)")
    axes[1].imshow(true, vmin=vmin, vmax=vmax, cmap=cmap)
    axes[1].set_title("True")
    axes[2].imshow(pred, vmin=vmin, vmax=vmax, cmap=cmap)
    axes[2].set_title("Pred")
    axes[3].imshow(err, vmin=evmin, vmax=evmax, cmap=cmap)
    axes[3].set_title("|Error|")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--preds",  required=True)
    ap.add_argument("--trues",  required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--nmax", type=int, default=0,
                    help="how many samples to export. 0 means ALL samples.")
    ap.add_argument("--tmax", type=int, default=None,
                    help="how many future steps to export. default=ALL Tout.")
    ap.add_argument("--channels", default="2",
                    help="comma-separated channel indices. default=2 (phi)")
    ap.add_argument("--chan_names", default="electron_den,ion_den,phi")
    ap.add_argument("--cmap", default="viridis")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    X = to_5d(np.load(args.inputs), "inputs")   # (N, Tin, C, H, W)
    P = to_5d(np.load(args.preds),  "preds")    # (N, Tout, C, H, W)
    Y = to_5d(np.load(args.trues),  "trues")    # (N, Tout, C, H, W)

    N_all = min(X.shape[0], P.shape[0], Y.shape[0])
    Tin = X.shape[1]
    Tout_all = min(P.shape[1], Y.shape[1])

    N = N_all if args.nmax == 0 else min(args.nmax, N_all)
    Tout = Tout_all if args.tmax is None else min(args.tmax, Tout_all)

    ch_list = parse_int_list(args.channels)
    names = [s.strip() for s in args.chan_names.split(",") if s.strip() != ""]
    def cname(c): return names[c] if 0 <= c < len(names) else f"ch{c}"

    print("inputs:", X.shape, X.dtype)
    print("preds: ", P.shape, P.dtype)
    print("trues: ", Y.shape, Y.dtype)
    print(f"export N={N}/{N_all}, Tout={Tout}/{Tout_all}, channels={ch_list}")

    for n in range(N):
        sample_dir = os.path.join(args.outdir, f"sample_{n:03d}")
        os.makedirs(sample_dir, exist_ok=True)

        in_last_cache = {}
        for c in ch_list:
            in_last_cache[c] = X[n, Tin-1, c]

        for tp in range(Tout):
            for c in ch_list:
                in_last = in_last_cache[c]
                true    = Y[n, tp, c]
                pred    = P[n, tp, c]

                title = f"sample {n} | future t+{tp+1} | {cname(c)} (c={c})"
                fn = f"s{n:03d}_tp{tp+1:02d}_c{c:02d}_{cname(c)}_quad.png"
                out = os.path.join(sample_dir, fn)
                save_one(out, in_last, true, pred, title, cmap=args.cmap)

    print("saved to:", os.path.abspath(args.outdir))

if __name__ == "__main__":
    main()
