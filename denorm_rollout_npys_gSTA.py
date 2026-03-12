import os
import h5py
import numpy as np

# --- stats (raw min/max per timestep) ---
H5_STATS = r"C:\Users\astro\research\PEPAPIC\test\results\2D_ExB_high_magnet_dt2.5e-9_maxt50e-6\SimVPv2_inputs\global_norm_t0_to_t1000_margin20_TCHW.h5"
TRAIN_T_FIRST = 0
TRAIN_T_LAST  = 803
MARGIN = 0.2

# --- normalized rollout npys (already generated) ---
BASE = r"C:\Users\astro\research\SimVPv2\workdirs\pepapic_simvp_gsta_highmag_trainfixed\rollout_tp0_quads_assets"
IN_NPY   = os.path.join(BASE, "inputs_roll.npy")
PR_NPY   = os.path.join(BASE, "preds_roll.npy")
TR_NPY   = os.path.join(BASE, "trues_roll.npy")

# --- output denorm npys ---
OUTDIR = os.path.join(BASE, "denorm_npys")
os.makedirs(OUTDIR, exist_ok=True)

def compute_trainfixed_minmax():
    with h5py.File(H5_STATS, "r") as f:
        min_raw = np.asarray(f["min_raw"])  # (C,T)
        max_raw = np.asarray(f["max_raw"])  # (C,T)
    xmin = min_raw[:, TRAIN_T_FIRST:TRAIN_T_LAST+1].min(axis=1)  # (C,)
    xmax = max_raw[:, TRAIN_T_FIRST:TRAIN_T_LAST+1].max(axis=1)  # (C,)
    r = xmax - xmin
    xmin2 = xmin - MARGIN * r
    xmax2 = xmax + MARGIN * r
    denom = xmax2 - xmin2
    return xmin2.astype(np.float32), denom.astype(np.float32)

def denorm(x, xmin2, denom):
    # x: (..., C, H, W) or (N,T,C,H,W)
    return x * denom.reshape(1,1,-1,1,1) + xmin2.reshape(1,1,-1,1,1)

def main():
    xmin2, denom = compute_trainfixed_minmax()

    X = np.load(IN_NPY).astype(np.float32)  # (K,10,C,H,W)
    P = np.load(PR_NPY).astype(np.float32)  # (K,1,C,H,W)
    Y = np.load(TR_NPY).astype(np.float32)  # (K,1,C,H,W)

    Xd = denorm(X, xmin2, denom)
    Pd = denorm(P, xmin2, denom)
    Yd = denorm(Y, xmin2, denom)

    np.save(os.path.join(OUTDIR, "inputs_roll_denorm.npy"), Xd)
    np.save(os.path.join(OUTDIR, "preds_roll_denorm.npy"),  Pd)
    np.save(os.path.join(OUTDIR, "trues_roll_denorm.npy"),  Yd)

    print("[DONE] wrote:", OUTDIR)
    print(" inputs denorm min/max:", float(Xd.min()), float(Xd.max()))
    print(" preds  denorm min/max:", float(Pd.min()), float(Pd.max()))
    print(" trues  denorm min/max:", float(Yd.min()), float(Yd.max()))
    print(" xmin2:", xmin2)
    print(" denom:", denom)

if __name__ == "__main__":
    main()