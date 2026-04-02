import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

direct_path = Path(r"C:\Users\astro\research\SimVPv2\workdirs\pepapic_simvp_gsta_highmag\saved\trues.npy")
roll_path   = Path(r"C:\Users\astro\research\SimVPv2\workdirs\pepapic_simvp_gsta_highmag_trainfixed\rollout_tp0_quads_assets\trues_roll.npy")

sample_idx = 0
channel_idx = 0

direct_step = 0
roll_step = 4   # ← find_true_alignment.py の結果に合わせて変更

out_path = Path("aligned_true_comparison.png")

direct = np.load(direct_path)
roll   = np.load(roll_path)

img_d = direct[sample_idx, direct_step, channel_idx]
img_r = roll[sample_idx, roll_step, channel_idx]

# 同じカラースケールで表示
vmin = min(img_d.min(), img_r.min())
vmax = max(img_d.max(), img_r.max())

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

im0 = axes[0].imshow(img_d, origin="lower", vmin=vmin, vmax=vmax)
axes[0].set_title(f"Direct true (step {direct_step})")
axes[0].axis("off")

im1 = axes[1].imshow(img_r, origin="lower", vmin=vmin, vmax=vmax)
axes[1].set_title(f"Rollout true (step {roll_step})")
axes[1].axis("off")

fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.8)
plt.tight_layout()
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.show()

mse = np.mean((img_d - img_r) ** 2)
print(f"MSE between these two true images: {mse:.6e}")
print(f"Saved to: {out_path.resolve()}")