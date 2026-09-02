import argparse
import json
from pathlib import Path

import h5py
import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_CASE = Path(
    r"C:\Users\astro\research\PEPAPIC\test\results"
    r"\2D_ExB_low_magnet_dt2.5e-9_maxt50e-6_macro5"
)
DEFAULT_OUTPUT = DEFAULT_CASE / "raw_rotated_interior_phi.gif"
RANKS = [0, 1, 2, 3]


def parse_timestep_spec(spec):
    spec = str(spec).strip()
    if "," in spec:
        return [int(x.strip()) for x in spec.split(",") if x.strip()]
    if ":" in spec:
        parts = [int(x.strip()) for x in spec.split(":")]
        if len(parts) == 2:
            start, end = parts
            step = 1
        elif len(parts) == 3:
            start, step, end = parts
        else:
            raise ValueError(f"Invalid timestep spec: {spec}")
        return list(range(start, end + 1, step))
    return [int(spec)]


def parse_rank_list(spec):
    spec = str(spec).strip()
    if not spec:
        return []
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def load_domain(case_folder, rank):
    path = case_folder / "Domain_info" / f"Local_domain_info_rank{rank}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_field(case_folder, rank, timestep, prop):
    candidates = [
        case_folder / "Macro" / f"Macro_tn_{timestep}" / f"ISP1_domain_macro_tn_{timestep}_rank{rank}.h5",
        case_folder / f"Macro_tn_{timestep}" / f"ISP1_domain_macro_tn_{timestep}_rank{rank}.h5",
    ]
    h5_path = next((path for path in candidates if path.exists()), None)
    if h5_path is None:
        tried = "\n".join(str(path) for path in candidates)
        raise FileNotFoundError(f"H5 file not found for t={timestep}, rank={rank}. Tried:\n{tried}")

    with h5py.File(h5_path, "r") as f:
        arr = np.asarray(f[prop][()], dtype=np.float32)

    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D field after squeeze, got {arr.shape} from {h5_path}:{prop}")
    return arr


def trim_int_offset(arr, domain):
    offset = domain["Int_offst"][0]
    n_int = domain["N_int_sp"][0]
    i0 = int(offset[0])
    j0 = int(offset[1])
    i1 = i0 + int(n_int[0])
    j1 = j0 + int(n_int[1])
    return arr[i0:i1, j0:j1]


def trim_internal_overlap(arr_xy, domain):
    xminus, xplus, yminus, yplus = domain["ranks_bc"][:4]
    i0 = 1 if xminus != -1 else 0
    i1 = arr_xy.shape[0] - 1 if xplus != -1 else arr_xy.shape[0]
    j0 = 1 if yminus != -1 else 0
    j1 = arr_xy.shape[1] - 1 if yplus != -1 else arr_xy.shape[1]
    return arr_xy[i0:i1, j0:j1]


def prepare_tile(arr, domain, rank, stitch_mode, transpose_ranks):
    if stitch_mode == "physical_int_offst":
        # h5py sees Julia's (X,Y) plane as (Y,X).  Restore (X,Y) first,
        # then apply the X/Y interior offsets recorded for this rank.
        return trim_int_offset(arr.T, domain)
    if stitch_mode == "training_compatible":
        # h5py exposes the two spatial axes in the reverse order from the
        # Julia pipeline that created the 3b training H5.
        return trim_internal_overlap(arr.T, domain)
    tile = trim_int_offset(arr, domain)
    return tile.T if rank in transpose_ranks else tile


def block_index(value, grid_values):
    diffs = [abs(float(value) - float(g)) for g in grid_values]
    return int(np.argmin(diffs))


def build_layout(case_folder, ranks, prop, stitch_mode, transpose_ranks):
    domains = {rank: load_domain(case_folder, rank) for rank in ranks}
    first = ranks[0]
    sample = prepare_tile(
        read_field(case_folder, first, 0, prop),
        domains[first],
        first,
        stitch_mode,
        set(transpose_ranks),
    )
    nx_tile, ny_tile = sample.shape
    xmins = sorted({float(domains[rank]["Xp_min"][0]) for rank in ranks})
    ymins = sorted({float(domains[rank]["Xp_min"][1]) for rank in ranks})
    return domains, nx_tile, ny_tile, xmins, ymins


def build_global_xy(
    case_folder,
    domains,
    ranks,
    timestep,
    prop,
    nx_tile,
    ny_tile,
    xmins,
    ymins,
    rotate_ranks,
    transpose_ranks,
    stitch_mode,
):
    global_xy = np.zeros((nx_tile * len(xmins), ny_tile * len(ymins)), dtype=np.float32)
    rotate_ranks = set(rotate_ranks)
    transpose_ranks = set(transpose_ranks)

    for rank in ranks:
        domain = domains[rank]
        tile = prepare_tile(
            read_field(case_folder, rank, timestep, prop),
            domain,
            rank,
            stitch_mode,
            transpose_ranks,
        )
        if tile.shape != (nx_tile, ny_tile):
            raise ValueError(
                f"Tile size mismatch rank={rank} t={timestep}: got {tile.shape}, expected {(nx_tile, ny_tile)}"
            )
        if rank in rotate_ranks:
            tile = np.rot90(tile, 2)

        bx = block_index(domain["Xp_min"][0], xmins)
        by = block_index(domain["Xp_min"][1], ymins)
        ox = bx * nx_tile
        oy = by * ny_tile
        global_xy[ox:ox + nx_tile, oy:oy + ny_tile] = tile

    return global_xy


def to_display_yx(data_xy, display_mode):
    if display_mode == "physical_yx":
        # imshow expects rows=Y and columns=X.  The stitched array is (X,Y).
        return data_xy.T
    if display_mode == "legacy_rotcw90":
        return np.rot90(data_xy, k=-1)
    raise ValueError(display_mode)


class FrameRenderer:
    def __init__(self, data, timestep, dt_ns, prop, figsize, dpi, display_mode):
        displayed = to_display_yx(data, display_mode)
        vmin = float(np.nanmin(displayed))
        vmax = float(np.nanmax(displayed))
        time_us = timestep * dt_ns / 1000.0
        self.prop = prop
        self.dt_ns = dt_ns
        self.dpi = dpi
        self.display_mode = display_mode
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        self.image = self.ax.imshow(
            displayed,
            origin="lower",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_title(
            f"Raw PIC correctly stitched | {prop} | timestep={timestep} | t={time_us:.4f} us"
        )
        self.colorbar = self.fig.colorbar(self.image, ax=self.ax, label=prop)
        self.fig.tight_layout()

    def render(self, data, timestep, png_path=None):
        displayed = to_display_yx(data, self.display_mode)
        self.image.set_data(displayed)
        self.image.set_clim(float(np.nanmin(displayed)), float(np.nanmax(displayed)))
        self.colorbar.update_normal(self.image)
        time_us = timestep * self.dt_ns / 1000.0
        self.ax.set_title(
            f"Raw PIC correctly stitched | {self.prop} | timestep={timestep} | t={time_us:.4f} us"
        )
        if png_path is not None:
            self.fig.savefig(png_path, dpi=self.dpi)
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy()

    def close(self):
        plt.close(self.fig)


def main():
    parser = argparse.ArgumentParser(
        description="Create a raw low-magnet PIC GIF from all true phi frames."
    )
    parser.add_argument("--case-folder", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--timesteps", default="0:4000")
    parser.add_argument("--prop", default="phi")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--png-dir",
        type=Path,
        default=None,
        help="Optional directory for one PNG per timestep.",
    )
    parser.add_argument("--dt-ns", type=float, default=12.5)
    parser.add_argument("--gif-duration-ms", type=int, default=200)
    parser.add_argument("--rotate-ranks-180", default="")
    parser.add_argument("--transpose-ranks", default="0,1,2,3")
    parser.add_argument(
        "--stitch-mode",
        choices=["physical_int_offst", "training_compatible", "legacy_int_offst"],
        default="physical_int_offst",
        help=(
            "physical_int_offst restores Julia's (X,Y) axes before applying "
            "the rank-specific interior offsets; training_compatible reproduces "
            "the historical 3b training H5."
        ),
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--fig-width", type=float, default=7.6)
    parser.add_argument("--fig-height", type=float, default=6.2)
    parser.add_argument(
        "--display-mode",
        choices=["physical_yx", "legacy_rotcw90"],
        default="physical_yx",
        help="physical_yx displays rows=Y, columns=X after coordinate-based stitching.",
    )
    args = parser.parse_args()

    case_folder = args.case_folder
    timesteps = parse_timestep_spec(args.timesteps)
    rotate_ranks = parse_rank_list(args.rotate_ranks_180)
    transpose_ranks = parse_rank_list(args.transpose_ranks)
    domains, nx_tile, ny_tile, xmins, ymins = build_layout(
        case_folder,
        RANKS,
        args.prop,
        args.stitch_mode,
        transpose_ranks,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.png_dir is not None:
        args.png_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] case={case_folder}", flush=True)
    print(f"[INFO] output={args.output}", flush=True)
    print(f"[INFO] timesteps={timesteps[0]}..{timesteps[-1]} count={len(timesteps)}", flush=True)
    print(f"[INFO] tile=({nx_tile},{ny_tile}) blocks=({len(xmins)},{len(ymins)})", flush=True)
    print(f"[INFO] rotate_ranks_180={rotate_ranks}", flush=True)
    print(f"[INFO] transpose_ranks={transpose_ranks}", flush=True)
    print(f"[INFO] stitch_mode={args.stitch_mode}", flush=True)
    print(f"[INFO] gif_duration_ms={args.gif_duration_ms}", flush=True)
    print(f"[INFO] png_dir={args.png_dir}", flush=True)

    # The Pillow-backed imageio GIF writer used in this environment records
    # frame duration in milliseconds.
    renderer = None
    try:
        with imageio.get_writer(args.output, mode="I", duration=args.gif_duration_ms, loop=0) as writer:
            for i, timestep in enumerate(timesteps):
                if i == 0 or (i + 1) % 100 == 0 or i + 1 == len(timesteps):
                    print(f"[GIF] frame {i + 1}/{len(timesteps)} timestep={timestep}", flush=True)
                data = build_global_xy(
                    case_folder,
                    domains,
                    RANKS,
                    timestep,
                    args.prop,
                    nx_tile,
                    ny_tile,
                    xmins,
                    ymins,
                    rotate_ranks,
                    transpose_ranks,
                    args.stitch_mode,
                )
                if renderer is None:
                    renderer = FrameRenderer(
                        data,
                        timestep,
                        args.dt_ns,
                        args.prop,
                        (args.fig_width, args.fig_height),
                        args.dpi,
                        args.display_mode,
                    )
                frame = renderer.render(
                    data,
                    timestep,
                    (
                        args.png_dir / f"raw_rotcw90_interior_{args.prop}_ts{timestep}.png"
                        if args.png_dir is not None
                        else None
                    ),
                )
                writer.append_data(frame)
    finally:
        if renderer is not None:
            renderer.close()

    print(f"[OK] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
