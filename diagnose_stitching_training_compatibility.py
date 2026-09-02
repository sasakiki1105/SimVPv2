import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RANKS = [0, 1, 2, 3]
PROPS = ["electron_den", "ion_den", "phi"]


def load_domain(case_folder, rank):
    path = case_folder / "Domain_info" / f"Local_domain_info_rank{rank}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_field(case_folder, rank, timestep, prop):
    path = (
        case_folder
        / "Macro"
        / f"Macro_tn_{timestep}"
        / f"ISP1_domain_macro_tn_{timestep}_rank{rank}.h5"
    )
    with h5py.File(path, "r") as f:
        return np.squeeze(np.asarray(f[prop][()], dtype=np.float32))


def trim_int_offset(array, domain):
    offset = domain["Int_offst"][0]
    n_int = domain["N_int_sp"][0]
    i0, j0 = int(offset[0]), int(offset[1])
    return array[i0:i0 + int(n_int[0]), j0:j0 + int(n_int[1])]


def trim_internal_overlap(array_xy, domain):
    xminus, xplus, yminus, yplus = domain["ranks_bc"][:4]
    i0 = 1 if xminus != -1 else 0
    i1 = array_xy.shape[0] - 1 if xplus != -1 else array_xy.shape[0]
    j0 = 1 if yminus != -1 else 0
    j1 = array_xy.shape[1] - 1 if yplus != -1 else array_xy.shape[1]
    return array_xy[i0:i1, j0:j1]


def place_tiles(tiles, domains):
    global_xy = np.zeros((100, 100), dtype=np.float32)
    for rank, tile in tiles.items():
        bx = int(float(domains[rank]["Xp_min"][0]) > 0.0)
        by = int(float(domains[rank]["Xp_min"][1]) > 0.0)
        global_xy[bx * 50:(bx + 1) * 50, by * 50:(by + 1) * 50] = tile
    return global_xy


def build_legacy(case_folder, domains, timestep, prop):
    tiles = {}
    for rank in RANKS:
        raw = read_field(case_folder, rank, timestep, prop)
        tiles[rank] = trim_int_offset(raw, domains[rank]).T
    return place_tiles(tiles, domains)


def build_training_compatible(case_folder, domains, timestep, prop):
    tiles = {}
    for rank in RANKS:
        raw_xy = read_field(case_folder, rank, timestep, prop).T
        tiles[rank] = trim_internal_overlap(raw_xy, domains[rank])
    return place_tiles(tiles, domains)


def edge_jump_stats(global_xy):
    return {
        "internal_vertical": float(np.mean(np.abs(global_xy[49, :] - global_xy[50, :]))),
        "internal_horizontal": float(np.mean(np.abs(global_xy[:, 49] - global_xy[:, 50]))),
        "outer_left": float(np.mean(np.abs(global_xy[0, :] - global_xy[1, :]))),
        "outer_right": float(np.mean(np.abs(global_xy[-1, :] - global_xy[-2, :]))),
        "outer_bottom": float(np.mean(np.abs(global_xy[:, 0] - global_xy[:, 1]))),
        "outer_top": float(np.mean(np.abs(global_xy[:, -1] - global_xy[:, -2]))),
    }


def add_rank_labels(ax):
    labels = [
        (25, 25, "rank1"),
        (75, 25, "rank0"),
        (25, 75, "rank3"),
        (75, 75, "rank2"),
    ]
    for x, y, label in labels:
        ax.text(
            x,
            y,
            label,
            color="white",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.axvline(49.5, color="white", linewidth=0.7, alpha=0.8)
    ax.axhline(49.5, color="white", linewidth=0.7, alpha=0.8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-folder", type=Path, required=True)
    parser.add_argument("--timestep", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    domains = {rank: load_domain(args.case_folder, rank) for rank in RANKS}
    fig, axes = plt.subplots(len(PROPS), 3, figsize=(13.5, 12.0), dpi=180)

    for row, prop in enumerate(PROPS):
        legacy_xy = build_legacy(args.case_folder, domains, args.timestep, prop)
        correct_xy = build_training_compatible(args.case_folder, domains, args.timestep, prop)
        legacy_view = np.rot90(legacy_xy, k=-1)
        correct_view = np.rot90(correct_xy, k=-1)
        difference = np.abs(legacy_view - correct_view)
        vmin = float(min(np.min(legacy_view), np.min(correct_view)))
        vmax = float(max(np.max(legacy_view), np.max(correct_view)))

        im0 = axes[row, 0].imshow(
            legacy_view, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax
        )
        axes[row, 0].set_title(f"{prop}: legacy")
        add_rank_labels(axes[row, 0])

        axes[row, 1].imshow(
            correct_view, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax
        )
        axes[row, 1].set_title(f"{prop}: training-compatible")
        add_rank_labels(axes[row, 1])

        im2 = axes[row, 2].imshow(difference, origin="lower", cmap="magma")
        axes[row, 2].set_title(f"{prop}: absolute difference")
        add_rank_labels(axes[row, 2])

        fig.colorbar(im0, ax=axes[row, :2].tolist(), shrink=0.82, label=prop)
        fig.colorbar(im2, ax=axes[row, 2], shrink=0.82, label="absolute difference")

        print(f"[{prop}] mean_abs_difference={float(np.mean(difference)):.8g}")
        print(f"[{prop}] legacy_edges={edge_jump_stats(legacy_xy)}")
        print(f"[{prop}] compatible_edges={edge_jump_stats(correct_xy)}")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"Rank stitching order diagnostic | {args.case_folder.name} | timestep={args.timestep}",
        fontsize=14,
    )
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.94, wspace=0.28, hspace=0.22)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    print(f"[OK] {args.output}")


if __name__ == "__main__":
    main()
