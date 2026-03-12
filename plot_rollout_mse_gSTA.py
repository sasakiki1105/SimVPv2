import numpy as np
import matplotlib.pyplot as plt

# どっちでもOK：
# - 正規化空間：trues_roll.npy / preds_roll.npy
# - 逆正規化空間：trues_roll_denorm.npy / preds_roll_denorm.npy

BASE = r"C:\Users\astro\research\SimVPv2\workdirs\pepapic_simvp_gsta_highmag_trainfixed\rollout_tp0_quads_assets"

# --- choose one set ---
P_PATH = BASE + r"\preds_roll.npy"
Y_PATH = BASE + r"\trues_roll.npy"
# P_PATH = BASE + r"\denorm_npys\preds_roll_denorm.npy"
# Y_PATH = BASE + r"\denorm_npys\trues_roll_denorm.npy"

# チャネル名（あなたの並び）
CHAN_NAMES = ["electron_den", "ion_den", "phi"]

# どのチャネルで見る？
# None -> 全チャネルまとめて（平均）
# 2 -> phi だけ
C_TARGET = 2

def mse_per_step(P, Y, c=None):
    """
    P, Y: (K, 1, C, H, W) を想定
    return: (K,) の MSE
    """
    assert P.shape == Y.shape, (P.shape, Y.shape)
    K = P.shape[0]
    if c is None:
        diff = P - Y  # (K,1,C,H,W)
    else:
        diff = P[:, :, c:c+1] - Y[:, :, c:c+1]
    # 平均を取る次元：1,T(=1),C(=1 or all),H,W
    return np.mean(diff**2, axis=(1,2,3,4))

def main():
    P = np.load(P_PATH).astype(np.float64)
    Y = np.load(Y_PATH).astype(np.float64)

    mses = mse_per_step(P, Y, c=C_TARGET)

    x = np.arange(len(mses))

    plt.figure()
    plt.plot(x, mses)
    label = "all_channels" if C_TARGET is None else CHAN_NAMES[C_TARGET]
    plt.xlabel("Rollout iteration (k)")
    plt.ylabel(f"MSE ({label})")
    plt.title("Closed-loop rollout MSE vs iteration")
    plt.tight_layout()

    out_png = BASE + (r"\mse_curve_all.png" if C_TARGET is None else rf"\mse_curve_{label}.png")
    plt.savefig(out_png, dpi=150)
    print("saved:", out_png)

if __name__ == "__main__":
    main()