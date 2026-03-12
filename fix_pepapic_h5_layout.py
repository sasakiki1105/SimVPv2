# fix_pepapic_h5_layout.py
import argparse
import numpy as np
import h5py

def to_TCHW(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 4:
        raise ValueError(f"expected 4D, got {a.shape}")

    s = a.shape
    # already (T,C,H,W)
    if s[0] >= 20 and s[1] <= 20 and s[2] >= 16 and s[3] >= 16:
        return a

    # (H,W,C,T) -> (T,C,H,W)
    if s[0] >= 16 and s[1] >= 16 and s[2] <= 20 and s[3] >= 20:
        return np.transpose(a, (3, 2, 0, 1))

    # (C,H,W,T) -> (T,C,H,W)
    if s[0] <= 20 and s[1] >= 16 and s[2] >= 16 and s[3] >= 20:
        return np.transpose(a, (3, 0, 1, 2))

    # (T,H,W,C) -> (T,C,H,W)
    if s[0] >= 20 and s[1] >= 16 and s[2] >= 16 and s[3] <= 20:
        return np.transpose(a, (0, 3, 1, 2))

    raise ValueError(f"cannot infer layout from shape={s}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_h5", required=True)
    ap.add_argument("--out_h5", required=True)
    ap.add_argument("--key", default="data_tchw", help="dataset key (default data_tchw)")
    args = ap.parse_args()

    with h5py.File(args.in_h5, "r") as f:
        if args.key not in f:
            raise KeyError(f"'{args.key}' not found. keys={list(f.keys())}")
        raw = f[args.key][:]
        props = f["props"][:] if "props" in f else None
        timesteps = f["timesteps"][:] if "timesteps" in f else None

        # copy a few useful scalars if present
        meta = {}
        for k in ["pre_seq_length","aft_seq_length","stride","train_ratio","margin","norm_mode",
                  "train_t_first","train_t_last"]:
            if k in f:
                meta[k] = f[k][()]

    data = to_TCHW(raw).astype(np.float32, copy=False)
    print("[INFO] converted:", raw.shape, "->", data.shape, "(T,C,H,W)")

    with h5py.File(args.out_h5, "w") as g:
        g["data_tchw"] = data  # ★必ず(T,C,H,W)で保存
        if props is not None:
            g["props"] = props
        if timesteps is not None:
            g["timesteps"] = timesteps
        for k,v in meta.items():
            g[k] = v
        g["layout"] = "data_tchw (T,C,H,W)"

    print("[DONE] saved:", args.out_h5)

if __name__ == "__main__":
    main()
