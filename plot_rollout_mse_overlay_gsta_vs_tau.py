import numpy as np
import matplotlib.pyplot as plt

# workdir roots
BASE_G = r"C:\Users\astro\research\SimVPv2\workdirs\pepapic_simvp_gsta_highmag_trainfixed\rollout_tp0_quads_assets"
BASE_T = r"C:\Users\astro\research\SimVPv2\workdirs\pepapic_simvp_tau_highmag_trainfixed\rollout_tp0_quads_assets"

# --- choose one set (norm or denorm) ---
P_G = BASE_G + r"\preds_roll.npy"
Y_G = BASE_G + r"\trues_roll.npy"
P_T = BASE_T + r"\preds_roll.npy"
Y_T = BASE_T + r"\trues_roll.npy"

# denorm版にしたいなら上の4行をコメントアウトしてこっち
# P_G = BASE_G + r"\denorm_npys\preds_roll_denorm.npy"
# Y_G = BASE_G + r"\denorm_npys\trues_roll_denorm.npy"
# P_T = BASE_T + r"\denorm_npys\preds_roll_denorm.npy"
# Y_T = BASE_T + r"\denorm_npys\trues_roll_denorm.npy"

CHAN_NAMES = ["electron_den", "ion_den", "phi"]
C_TARGET = 2  # None -> all channels, 2 -> phi

def mse_per_step(P, Y, c=None):
    assert P.shape == Y.shape, (P.shape, Y.shape)
    if c is None:
        diff = P - Y
    else:
        diff = P[:, :, c:c+1] - Y[:, :, c:c+1]
    return np.mean(diff**2, axis=(1,2,3,4))

def main():
    Pg = np.load(P_G).astype(np.float64)
    Yg = np.load(Y_G).astype(np.float64)
    Pt = np.load(P_T).astype(np.float64)
    Yt = np.load(Y_T).astype(np.float64)

    mg = mse_per_step(Pg, Yg, c=C_TARGET)
    mt = mse_per_step(Pt, Yt, c=C_TARGET)

    K = min(len(mg), len(mt))
    x = np.arange(K)

    label = "all_channels" if C_TARGET is None else CHAN_NAMES[C_TARGET]

    plt.figure()
    plt.plot(x, mg[:K], label="gSTA")
    plt.plot(x, mt[:K], label="TAU")
    plt.xlabel("Rollout iteration (k)")
    plt.ylabel(f"MSE ({label})")
    plt.title("Closed-loop rollout MSE: gSTA vs TAU")
    plt.legend()
    plt.tight_layout()

    out_png = BASE_T + (r"\mse_overlay_gsta_vs_tau_all.png" if C_TARGET is None else rf"\mse_overlay_gsta_vs_tau_{label}.png")
    plt.savefig(out_png, dpi=150)
    print("saved:", out_png)

if __name__ == "__main__":
    main()