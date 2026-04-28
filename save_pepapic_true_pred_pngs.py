import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def to_5d(x, name="array"):
    x = np.asarray(x)
    if x.ndim == 5:
        if x.shape[2] <= 20 and x.shape[3] >= 16 and x.shape[4] >= 16:
            return x
        if x.shape[1] <= 20 and x.shape[3] >= 16 and x.shape[4] >= 16:
            return np.transpose(x, (0, 2, 1, 3, 4))
        if x.shape[4] <= 20 and x.shape[2] >= 16 and x.shape[3] >= 16:
            return np.transpose(x, (0, 1, 4, 2, 3))
        raise ValueError(f"{name}: unsupported 5D shape {x.shape}")
    elif x.ndim == 4:
        if x.shape[1] <= 20 and x.shape[2] >= 16 and x.shape[3] >= 16:
            return x[None, ...]
        if x.shape[0] <= 20 and x.shape[2] >= 16 and x.shape[3] >= 16:
            return np.transpose(x, (1, 0, 2, 3))[None, ...]
        if x.shape[3] <= 20 and x.shape[1] >= 16 and x.shape[2] >= 16:
            return np.transpose(x, (0, 3, 1, 2))[None, ...]
        raise ValueError(f"{name}: unsupported 4D shape {x.shape}")
    else:
        raise ValueError(f"{name}: expected 4D or 5D, got shape {x.shape}")

def parse_int_list(s):
    return [int(x) for x in s.split(",") if x.strip()]

def save_true_pred(fig_path, true_img, pred_img, title, cmap="viridis"):
    vmin = float(np.min([true_img.min(), pred_img.min()]))
    vmax = float(np.max([true_img.max(), pred_img.max()]))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    axes[0].imshow(true_img, vmin=vmin, vmax=vmax, cmap=cmap)
    axes[0].set_title("True")
    axes[1].imshow(pred_img, vmin=vmin, vmax=vmax, cmap=cmap)
    axes[1].set_title("Pred")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--trues", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--samples", default="0,125,250,375")
    ap.add_argument("--tmax", type=int, default=10)
    ap.add_argument("--channel", type=int, default=2)
    ap.add_argument("--chan_name", default="phi")
    ap.add_argument("--cmap", default="viridis")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    P = to_5d(np.load(args.preds), "preds")
    Y = to_5d(np.load(args.trues), "trues")

    sample_list = parse_int_list(args.samples)
    n_all = min(P.shape[0], Y.shape[0])
    t_all = min(P.shape[1], Y.shape[1])
    tmax = min(args.tmax, t_all)

    print("preds:", P.shape)
    print("trues:", Y.shape)

    for n in sample_list:
        if not (0 <= n < n_all):
            print(f"skip sample {n}: out of range")
            continue

        sample_dir = os.path.join(args.outdir, f"sample_{n:03d}")
        os.makedirs(sample_dir, exist_ok=True)

        for tp in range(tmax):
            true_img = Y[n, tp, args.channel]
            pred_img = P[n, tp, args.channel]
            title = f"sample {n} | future t+{tp+1} | {args.chan_name}"
            fn = f"s{n:03d}_tp{tp+1:02d}_c{args.channel:02d}_{args.chan_name}_true_pred.png"
            save_true_pred(os.path.join(sample_dir, fn), true_img, pred_img, title, cmap=args.cmap)

    print("saved to:", os.path.abspath(args.outdir))

if __name__ == "__main__":
    main()