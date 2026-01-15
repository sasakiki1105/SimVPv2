import os
import argparse
import numpy as np

def parse_int_list(s: str):
    if s is None or s == "":
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]

def to_5d(a: np.ndarray, name: str):
    """
    Accept:
      (N, T, C, H, W)  -> ok
      (T, C, H, W)     -> (1, T, C, H, W)
      (N, C, H, W)     -> (N, 1, C, H, W)
      (C, H, W)        -> (1, 1, C, H, W)
    """
    if a.ndim == 5:
        return a
    if a.ndim == 4:
        # assume (T,C,H,W)
        return a[None, ...]
    if a.ndim == 3:
        return a[None, None, ...]
    if a.ndim == 4 and name == "inputs":
        return a[:, None, ...]
    raise ValueError(f"{name}: unsupported shape {a.shape}")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def normalize_to_uint8(img: np.ndarray, vmin: float, vmax: float):
    eps = 1e-12
    x = (img - vmin) / (vmax - vmin + eps)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)

def save_png_uint8(path, u8):
    # Pillow不要、imageioも不要。matplotlib最小依存で保存
    import matplotlib.pyplot as plt
    plt.imsave(path, u8, cmap="gray", vmin=0, vmax=255)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="inputs.npy (N,Tin,C,H,W)")
    ap.add_argument("--preds",  required=True, help="preds.npy  (N,Tout,C,H,W)")
    ap.add_argument("--trues",  required=True, help="trues.npy  (N,Tout,C,H,W)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--nmax", type=int, default=1, help="how many samples to export")
    ap.add_argument("--tmax", type=int, default=None, help="how many future frames to export (default=all)")
    ap.add_argument("--channels", default=None, help="e.g. 0,1,2 (default=all)")
    ap.add_argument("--chan_names", default=None, help="e.g. electron_den,ion_den,phi")
    ap.add_argument("--save_inputs", action="store_true", help="also export all input frames")
    ap.add_argument("--save_triptych", action="store_true", help="also export triptych (pred vs true) per frame")
    args = ap.parse_args()

    X = to_5d(np.load(args.inputs), "inputs")  # (N, Tin, C, H, W)
    P = to_5d(np.load(args.preds),  "preds")   # (N, Tout, C, H, W)
    Y = to_5d(np.load(args.trues),  "trues")   # (N, Tout, C, H, W)

    N, Tin, C, H, W = X.shape
    N2, Tout, C2, H2, W2 = P.shape
    assert (N2, Tout, C2, H2, W2) == P.shape
    assert Y.shape == P.shape
    assert C2 == C and H2 == H and W2 == W, "channel/size mismatch between inputs and preds/trues"

    chan_list = parse_int_list(args.channels)
    if chan_list is None:
        chan_list = list(range(C))

    chan_names = None
    if args.chan_names:
        chan_names = [x.strip() for x in args.chan_names.split(",")]
        if len(chan_names) != C:
            # channels指定だけに合わせるのが面倒なので、Cと一致しないなら無視
            chan_names = None

    n_export = min(args.nmax, N)
    t_export = Tout if args.tmax is None else min(args.tmax, Tout)

    ensure_dir(args.outdir)

    for s in range(n_export):
        # サンプルごとに出力先
        sdir = os.path.join(args.outdir, f"s{s:03d}")
        ensure_dir(sdir)

        for c in chan_list:
            cname = chan_names[c] if chan_names else f"c{c:02d}"

            # ★ 可視化の最重要ポイント：同一サンプル・同一チャネルで vmin/vmax を固定
            #   → 「うっすらピーク」が見えなくなる問題がかなり減る
            all_vals = np.concatenate([
                X[s, :, c].reshape(-1),
                P[s, :, c].reshape(-1),
                Y[s, :, c].reshape(-1),
            ])
            vmin = float(all_vals.min())
            vmax = float(all_vals.max())

            # inputs（過去10枚）も全部出すなら
            if args.save_inputs:
                for ti in range(Tin):
                    u8 = normalize_to_uint8(X[s, ti, c], vmin, vmax)
                    out = os.path.join(sdir, f"s{s:03d}_in{ti:02d}_{cname}.png")
                    save_png_uint8(out, u8)

            # future（予測10枚）を全部出す
            for tp in range(t_export):
                u8p = normalize_to_uint8(P[s, tp, c], vmin, vmax)
                u8y = normalize_to_uint8(Y[s, tp, c], vmin, vmax)

                out_p = os.path.join(sdir, f"s{s:03d}_tp{tp:02d}_pred_{cname}.png")
                out_y = os.path.join(sdir, f"s{s:03d}_tp{tp:02d}_true_{cname}.png")
                save_png_uint8(out_p, u8p)
                save_png_uint8(out_y, u8y)

                # pred/true を横に並べた比較画像も欲しければ
                if args.save_triptych:
                    # (H, 2W)
                    combo = np.concatenate([u8p, u8y], axis=1)
                    out_c = os.path.join(sdir, f"s{s:03d}_tp{tp:02d}_pred_true_{cname}.png")
                    save_png_uint8(out_c, combo)

    print(f"done -> {args.outdir}")

if __name__ == "__main__":
    main()
