import numpy as np
import matplotlib.pyplot as plt

BASE = r"C:\Users\astro\research\SimVPv2\workdirs\pepapic_simvp_gsta_highmag_trainfixed\rollout_tp0_quads_assets"

# どちらでもOK（まずは正規化空間を推奨）
P_PATH = BASE + r"\preds_roll.npy"
Y_PATH = BASE + r"\trues_roll.npy"
# 逆正規化空間で見たい場合はこっち
# P_PATH = BASE + r"\denorm_npys\preds_roll_denorm.npy"
# Y_PATH = BASE + r"\denorm_npys\trues_roll_denorm.npy"

C = 2  # phi channel index

def argmax_2d(a2d: np.ndarray):
    # returns (y, x)
    idx = int(np.argmax(a2d))
    y, x = np.unravel_index(idx, a2d.shape)
    return y, x

def main():
    P = np.load(P_PATH)  # (K,1,C,H,W)
    Y = np.load(Y_PATH)

    assert P.shape == Y.shape, (P.shape, Y.shape)
    K = P.shape[0]

    peak_val_err = np.zeros(K, dtype=np.float64)
    peak_loc_err = np.zeros(K, dtype=np.float64)

    for k in range(K):
        p = P[k, 0, C]
        t = Y[k, 0, C]

        pmax = float(p.max())
        tmax = float(t.max())
        peak_val_err[k] = abs(pmax - tmax)

        py, px = argmax_2d(p)
        ty, tx = argmax_2d(t)
        peak_loc_err[k] = ((py - ty) ** 2 + (px - tx) ** 2) ** 0.5  # Euclidean distance (pixels)

    x = np.arange(K)

    # 1) peak location error
    plt.figure()
    plt.plot(x, peak_loc_err)
    plt.xlabel("Rollout iteration (k)")
    plt.ylabel("Peak location error (pixels)")
    plt.title("Closed-loop rollout peak location error vs iteration (phi)")
    plt.tight_layout()
    out1 = BASE + r"\peak_loc_err_phi.png"
    plt.savefig(out1, dpi=150)
    print("saved:", out1)

    # 2) peak value error
    plt.figure()
    plt.plot(x, peak_val_err)
    plt.xlabel("Rollout iteration (k)")
    plt.ylabel("Peak value error (abs)")
    plt.title("Closed-loop rollout peak value error vs iteration (phi)")
    plt.tight_layout()
    out2 = BASE + r"\peak_val_err_phi.png"
    plt.savefig(out2, dpi=150)
    print("saved:", out2)

if __name__ == "__main__":
    main()