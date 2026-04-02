import numpy as np
from pathlib import Path

# ===== paths =====
direct_path = Path(r"C:\Users\astro\research\SimVPv2\workdirs\pepapic_simvp_gsta_highmag\saved\trues.npy")
roll_path   = Path(r"C:\Users\astro\research\SimVPv2\workdirs\pepapic_simvp_gsta_highmag_trainfixed\rollout_tp0_quads_assets\trues_roll.npy")

# ===== settings =====
roll_sample_idx = 0
roll_step_idx = 0       # rollout 側は shape が (N,1,C,H,W) なので普通 0
channel_idx = 0         # 0=potential を想定
top_k = 10              # 上位何件出すか

# ===== load =====
direct = np.load(direct_path)   # shape: [Nd, Td, C, H, W]
roll   = np.load(roll_path)     # shape: [Nr, Tr, C, H, W]

print("direct shape:", direct.shape)
print("roll   shape:", roll.shape)

assert direct.ndim == 5, f"Unexpected direct ndim: {direct.ndim}"
assert roll.ndim   == 5, f"Unexpected roll ndim: {roll.ndim}"

Nd, Td, Cd, Hd, Wd = direct.shape
Nr, Tr, Cr, Hr, Wr = roll.shape

assert channel_idx < Cd and channel_idx < Cr
assert roll_sample_idx < Nr
assert roll_step_idx < Tr

target = roll[roll_sample_idx, roll_step_idx, channel_idx]

results = []

for ds in range(Nd):
    for dt in range(Td):
        cand = direct[ds, dt, channel_idx]
        mse = np.mean((cand - target) ** 2)
        results.append((mse, ds, dt))

results.sort(key=lambda x: x[0])

print(f"\nTarget: rollout sample={roll_sample_idx}, step={roll_step_idx}, channel={channel_idx}")
print(f"Top {top_k} closest matches in direct trues:\n")

for rank, (mse, ds, dt) in enumerate(results[:top_k], start=1):
    print(f"{rank:2d}. direct sample={ds:3d}, step={dt:2d}, MSE={mse:.6e}")

best_mse, best_ds, best_dt = results[0]
print("\nBest match:")
print(f"  direct sample={best_ds}, step={best_dt}, MSE={best_mse:.6e}")

# 追加情報: 同じ rollout sample index に限定した場合の最良候補
if roll_sample_idx < Nd:
    same_sample_results = []
    for dt in range(Td):
        cand = direct[roll_sample_idx, dt, channel_idx]
        mse = np.mean((cand - target) ** 2)
        same_sample_results.append((mse, dt))
    same_sample_results.sort(key=lambda x: x[0])

    print(f"\nWithin the same sample index ({roll_sample_idx}) in direct:")
    for rank, (mse, dt) in enumerate(same_sample_results[:min(top_k, Td)], start=1):
        print(f"{rank:2d}. direct sample={roll_sample_idx}, step={dt:2d}, MSE={mse:.6e}")