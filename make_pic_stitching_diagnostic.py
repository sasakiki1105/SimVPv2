import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diagnose_pic_rank_stitching import (
    RANKS,
    load_domain,
    read_field,
    trim_int_offset,
)


DEFAULT_LOW_CASE = Path(
    r"C:\Users\astro\research\PEPAPIC\test\results"
    r"\2D_ExB_low_magnet_dt2.5e-9_maxt50e-6_macro5"
)
DEFAULT_OUT = Path(r"C:\Users\astro\research\SimVPv2\workdirs\debug_low_magnet_stitching")


def block_index(value, grid_values):
    diffs = [abs(float(value) - float(g)) for g in grid_values]
    return int(np.argmin(diffs))


def tile_transform(tile, mode, rank):
    if mode == "raw":
        return tile
    if mode == "rot12":
        return np.rot90(tile, 2) if rank in (1, 2) else tile
    if mode == "transpose":
        return tile.T
    if mode == "transpose_rot12":
        tile = tile.T
        return np.rot90(tile, 2) if rank in (1, 2) else tile
    raise ValueError(mode)


def build_global(case_folder, timestep, prop, mode):
    domains = {rank: load_domain(case_folder, rank) for rank in RANKS}
    sample = trim_int_offset(read_field(case_folder, 0, timestep, prop), domains[0])
    nx_tile, ny_tile = sample.shape
    xmins = sorted({float(domains[rank]["Xp_min"][0]) for rank in RANKS})
    ymins = sorted({float(domains[rank]["Xp_min"][1]) for rank in RANKS})
    global_xy = np.zeros((nx_tile * len(xmins), ny_tile * len(ymins)), dtype=np.float32)

    for rank in RANKS:
        tile = trim_int_offset(read_field(case_folder, rank, timestep, prop), domains[rank])
        tile = tile_transform(tile, mode, rank)
        bx = block_index(domains[rank]["Xp_min"][0], xmins)
        by = block_index(domains[rank]["Xp_min"][1], ymins)
        ox = bx * nx_tile
        oy = by * ny_tile
        global_xy[ox:ox + nx_tile, oy:oy + ny_tile] = tile
    return global_xy


def add_rank_labels(ax):
    kw = dict(color="white", fontsize=12, fontweight="bold", ha="center", va="center")
    ax.text(25, 25, "r0", **kw)
    ax.text(75, 25, "r1", **kw)
    ax.text(25, 75, "r2", **kw)
    ax.text(75, 75, "r3", **kw)
    ax.axvline(49.5, color="white", lw=0.8, alpha=0.9)
    ax.axhline(49.5, color="white", lw=0.8, alpha=0.9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-folder", type=Path, default=DEFAULT_LOW_CASE)
    parser.add_argument("--timestep", type=int, default=400)
    parser.add_argument("--prop", default="phi")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    modes = [
        ("raw", "raw tiles"),
        ("rot12", "rank1/rank2 rot180"),
        ("transpose", "transpose every rank"),
        ("transpose_rot12", "transpose + rank1/rank2 rot180"),
    ]
    arrays = []
    for mode, title in modes:
        xy = build_global(args.case_folder, args.timestep, args.prop, mode)
        # Physical view: rows are y and columns are x.
        hw = xy.T
        arrays.append((mode, title, hw))

    vmin = min(float(np.nanmin(a)) for _, _, a in arrays)
    vmax = max(float(np.nanmax(a)) for _, _, a in arrays)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), dpi=170)
    for ax, (mode, title, arr) in zip(axes.ravel(), arrays):
        im = ax.imshow(arr, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
        add_rank_labels(ax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), label=args.prop, shrink=0.86)
    fig.suptitle(f"Stitching diagnostic | {args.case_folder.name} | {args.prop} | timestep={args.timestep}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"stitching_diagnostic_{args.case_folder.name}_{args.prop}_t{args.timestep}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
