import numpy as np
from pathlib import Path

# ===== paths =====
direct_path = Path(r"C:\Users\astro\research\SimVPv2\workdirs\pepapic_simvp_gsta_highmag\saved\trues.npy")
roll_path   = Path(r"C:\Users\astro\research\SimVPv2\workdirs\pepapic_simvp_gsta_highmag_trainfixed\rollout_tp0_quads_assets\trues_roll.npy")

# ===== settings =====
sample_idx = 0
channel_idx = 0   # 0=potential の想定。必要なら変えてください。
top_k = 5         # 近い候補をいくつ出すか

# ===== load =====
direct = np.load(direct_path)
roll   = np.load(roll_path)

print("direct shape:", direct.shape)
print("roll   shape:", roll.shape)

# 想定 shape: [N, T, C, H, W]
assert direct.ndim == 5, f"Unexpected direct ndim: {direct.ndim}"
assert roll.ndim   == 5, f"Unexpected roll ndim: {roll.ndim}"

assert sample_idx < direct.shape[0]
assert sample_idx < roll.shape[0]
assert channel_idx < direct.shape[2]
assert channel_idx < roll.shape[2]

Td = direct.shape[1]
Tr = roll.shape[1]

print(f"\nComparing sample={sample_idx}, channel={channel_idx}")
print(f"direct steps: {Td}, rollout steps: {Tr}\n")

# MSE matrix: [Td, Tr]
mse_mat = np.zeros((Td, Tr), dtype=np.float64)

for td in range(Td):
    d = direct[sample_idx, td, channel_idx]
    for tr in range(Tr):
        r = roll[sample_idx, tr, channel_idx]
        mse_mat[td, tr] = np.mean((d - r) ** 2)

# 各 direct step に対して、最も近い rollout step を出す
print("Best rollout step for each direct step:")
for td in range(Td):
    best_tr = int(np.argmin(mse_mat[td]))
    best_mse = mse_mat[td, best_tr]
    print(f"  direct step {td:2d} -> rollout step {best_tr:2d}   MSE = {best_mse:.6e}")

print("\nTop candidates for each direct step:")
for td in range(Td):
    order = np.argsort(mse_mat[td])[:top_k]
    pairs = ", ".join([f"(roll {int(tr)}, mse={mse_mat[td, tr]:.3e})" for tr in order])
    print(f"  direct step {td:2d}: {pairs}")

# rollout 側から見た対応も出す
print("\nBest direct step for each rollout step:")
for tr in range(Tr):
    best_td = int(np.argmin(mse_mat[:, tr]))
    best_mse = mse_mat[best_td, tr]
    print(f"  rollout step {tr:2d} -> direct step {best_td:2d}   MSE = {best_mse:.6e}")

# 全体で最も近いペアを出す
best_flat = np.argmin(mse_mat)
best_td, best_tr = np.unravel_index(best_flat, mse_mat.shape)
print("\nGlobally closest pair:")
print(f"  direct step {best_td} <-> rollout step {best_tr}   MSE = {mse_mat[best_td, best_tr]:.6e}")