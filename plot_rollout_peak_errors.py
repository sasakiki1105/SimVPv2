import os
import numpy as np
import matplotlib.pyplot as plt

BASE = r"C:\Users\astro\research\SimVPv2\workdirs\pepapic_simvp_gsta_highmag_macro5_subsample4_trainfixed_disjoint_811_bs2_100ep\rollout_tp0_quads_assets"

X_MODE = "time_us"
DT_NS = 50.0

P_PATH = os.path.join(BASE, "preds_roll.npy")
Y_PATH = os.path.join(BASE, "trues_roll.npy")

C = 2  # phi channel index

# ===== x-axis mode =====
# "iteration" : 横軸 = rollout iteration
# "time_ns"   : 横軸 = physical time [ns]
# "time_us"   : 横軸 = physical time [μs]
X_MODE = "time_us"

# 前回の run は 1 step = 50 ns
DT_NS = 50.0


def argmax_2d(a2d: np.ndarray):
    # returns (y, x)
    idx = int(np.argmax(a2d))
    y, x = np.unravel_index(idx, a2d.shape)
    return y, x


def get_x_axis(K: int, mode: str, dt_ns: float):
    if mode == "iteration":
        x = np.arange(K, dtype=np.float64)
        xlabel = "Rollout iteration (k)"
        suffix = "iter"
    elif mode == "time_ns":
        x = np.arange(K, dtype=np.float64) * dt_ns
        xlabel = "Physical time since rollout start (ns)"
        suffix = "time_ns"
    elif mode == "time_us":
        x = np.arange(K, dtype=np.float64) * dt_ns / 1000.0
        xlabel = "Physical time since rollout start (μs)"
        suffix = "time_us"
    else:
        raise ValueError(f"Unknown X_MODE: {mode}")
    return x, xlabel, suffix


def main():
    if not os.path.exists(P_PATH):
        raise FileNotFoundError(P_PATH)
    if not os.path.exists(Y_PATH):
        raise FileNotFoundError(Y_PATH)

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

    x, xlabel, suffix = get_x_axis(K, X_MODE, DT_NS)

    # 1) peak location error
    plt.figure()
    plt.plot(x, peak_loc_err)
    plt.xlabel(xlabel)
    plt.ylabel("Peak location error (pixels)")
    plt.title("Closed-loop rollout peak location error (phi)")
    plt.tight_layout()
    out1 = os.path.join(BASE, f"peak_loc_err_phi_{suffix}.png")
    plt.savefig(out1, dpi=150)
    print("saved:", out1)

    # 2) peak value error
    plt.figure()
    plt.plot(x, peak_val_err)
    plt.xlabel(xlabel)
    plt.ylabel("Peak value error (abs)")
    plt.title("Closed-loop rollout peak value error (phi)")
    plt.tight_layout()
    out2 = os.path.join(BASE, f"peak_val_err_phi_{suffix}.png")
    plt.savefig(out2, dpi=150)
    print("saved:", out2)


if __name__ == "__main__":
    main()