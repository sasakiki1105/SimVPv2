import numpy as np
import matplotlib.pyplot as plt

BASE = r"C:\Users\astro\research\SimVPv2\workdirs\pepapic_simvp_tau_highmag_trainfixed\rollout_tp0_quads_assets"

# --- choose one set ---
P_PATH = BASE + r"\preds_roll.npy"
Y_PATH = BASE + r"\trues_roll.npy"
# P_PATH = BASE + r"\denorm_npys\preds_roll_denorm.npy"
# Y_PATH = BASE + r"\denorm_npys\trues_roll_denorm.npy"

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
    P = np.load(P_PATH).astype(np.float64)
    Y = np.load(Y_PATH).astype(np.float64)

    mses = mse_per_step(P, Y, c=C_TARGET)
    x = np.arange(len(mses))

    label = "all_channels" if C_TARGET is None else CHAN_NAMES[C_TARGET]

    plt.figure()
    plt.plot(x, mses)
    plt.xlabel("Rollout iteration (k)")
    plt.ylabel(f"MSE ({label})")
    plt.title("TAU: Closed-loop rollout MSE vs iteration")
    plt.tight_layout()

    out_png = BASE + (r"\mse_curve_tau_all.png" if C_TARGET is None else rf"\mse_curve_tau_{label}.png")
    plt.savefig(out_png, dpi=150)
    print("saved:", out_png)

if __name__ == "__main__":
    main()