import argparse
import json
from pathlib import Path

import h5py
import numpy as np


DEFAULT_LOW_CASE = Path(
    r"C:\Users\astro\research\PEPAPIC\test\results"
    r"\2D_ExB_low_magnet_dt2.5e-9_maxt50e-6_macro5"
)
DEFAULT_HIGH_STATS_H5 = Path(
    r"C:\Users\astro\research\PEPAPIC\test\results"
    r"\2D_ExB_high_magnet_dt2.5e-9_maxt50e-6_macro5"
    r"\SimVPv2_inputs\global_norm_trainfixed_minmax_margin20_t0_to_t4000_step2.h5"
)
PROPS = ["electron_den", "ion_den", "phi"]
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
        raise FileNotFoundError(f"H5 file not found for t={timestep}, rank={rank}, prop={prop}. Tried:\n{tried}")

    with h5py.File(h5_path, "r") as f:
        arr = np.asarray(f[prop][()], dtype=np.float32)

    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D field after squeeze, got {arr.shape} from {h5_path}:{prop}")
    return arr


def trim_internal_overlap(arr, ranks_bc):
    xminus, xplus, yminus, yplus = ranks_bc[:4]
    i0 = 1 if xminus != -1 else 0
    i1 = arr.shape[0] - 1 if xplus != -1 else arr.shape[0]
    j0 = 1 if yminus != -1 else 0
    j1 = arr.shape[1] - 1 if yplus != -1 else arr.shape[1]
    return arr[i0:i1, j0:j1]


def trim_int_offset(arr, domain):
    offset = domain["Int_offst"][0]
    n_int = domain["N_int_sp"][0]
    i0 = int(offset[0])
    j0 = int(offset[1])
    i1 = i0 + int(n_int[0])
    j1 = j0 + int(n_int[1])
    return arr[i0:i1, j0:j1]


def trim_tile(arr, domain, trim_mode):
    if trim_mode == "training_compatible":
        return trim_internal_overlap(arr.T, domain["ranks_bc"])
    if trim_mode == "ranks_bc":
        return trim_internal_overlap(arr, domain["ranks_bc"])
    if trim_mode == "int_offst":
        return trim_int_offset(arr, domain)
    raise ValueError(f"Unknown trim mode: {trim_mode}")


def block_index(value, grid_values):
    diffs = [abs(float(value) - float(g)) for g in grid_values]
    return int(np.argmin(diffs))


def build_layout(case_folder, ranks, trim_mode="ranks_bc"):
    domains = {rank: load_domain(case_folder, rank) for rank in ranks}
    first_rank = ranks[0]
    sample = read_field(case_folder, first_rank, 0, PROPS[0])
    sample_trim = trim_tile(sample, domains[first_rank], trim_mode)
    nx_tile, ny_tile = sample_trim.shape
    xmins = sorted({float(domains[rank]["Xp_min"][0]) for rank in ranks})
    ymins = sorted({float(domains[rank]["Xp_min"][1]) for rank in ranks})
    return domains, nx_tile, ny_tile, xmins, ymins


def suffix_rank_list(prefix, ranks):
    if not ranks:
        return ""
    if list(ranks) == RANKS:
        return f"_{prefix}"
    return "_" + prefix + "".join(str(rank) for rank in ranks)


def build_global_frame(
    case_folder,
    domains,
    ranks,
    timestep,
    prop,
    nx_tile,
    ny_tile,
    xmins,
    ymins,
    rotate_ranks_180=None,
    transpose_ranks=None,
    trim_mode="ranks_bc",
):
    rotate_ranks_180 = set([] if rotate_ranks_180 is None else rotate_ranks_180)
    transpose_ranks = set([] if transpose_ranks is None else transpose_ranks)
    nx_global = nx_tile * len(xmins)
    ny_global = ny_tile * len(ymins)
    global_xy = np.zeros((nx_global, ny_global), dtype=np.float32)

    for rank in ranks:
        dom = domains[rank]
        arr = read_field(case_folder, rank, timestep, prop)
        tile = trim_tile(arr, dom, trim_mode)
        if trim_mode != "training_compatible" and rank in transpose_ranks:
            tile = tile.T
        if tile.shape != (nx_tile, ny_tile):
            raise ValueError(
                f"Tile size mismatch rank={rank} t={timestep} prop={prop}: "
                f"got {tile.shape}, expected {(nx_tile, ny_tile)}"
            )
        if rank in rotate_ranks_180:
            tile = np.rot90(tile, 2)
        bx = block_index(dom["Xp_min"][0], xmins)
        by = block_index(dom["Xp_min"][1], ymins)
        ox = bx * nx_tile
        oy = by * ny_tile
        global_xy[ox:ox + nx_tile, oy:oy + ny_tile] = tile

    if trim_mode == "training_compatible":
        return global_xy.copy()
    return global_xy.T.copy()


def decode_props(values):
    out = []
    for value in values:
        if isinstance(value, (bytes, bytearray)):
            out.append(value.decode())
        else:
            out.append(str(value))
    return out


def load_high_stats(path):
    with h5py.File(path, "r") as f:
        props = decode_props(f["props"][()]) if "props" in f else PROPS
        train_min = np.asarray(f["train_min"][()], dtype=np.float32)
        train_max = np.asarray(f["train_max"][()], dtype=np.float32)
        margin = float(f["margin"][()]) if "margin" in f else 0.2
    if props != PROPS:
        raise ValueError(f"Unexpected props in high stats H5: {props}, expected {PROPS}")
    return train_min, train_max, margin


def normalize_with_high_stats(frame_hwc, train_min, train_max, margin):
    out = np.empty_like(frame_hwc, dtype=np.float32)
    for ci in range(frame_hwc.shape[2]):
        mn = float(train_min[ci])
        mx = float(train_max[ci])
        span = mx - mn
        if span == 0:
            out[:, :, ci] = 0.0
            continue
        lo = mn - margin * span
        hi = mx + margin * span
        out[:, :, ci] = np.clip((frame_hwc[:, :, ci] - lo) / (hi - lo), 0.0, 1.0)
    return out


def build_h5(
    case_folder,
    high_stats_h5,
    timestep_spec,
    output=None,
    ranks=None,
    rotate_ranks_180=None,
    transpose_ranks=None,
    trim_mode="training_compatible",
):
    case_folder = Path(case_folder)
    high_stats_h5 = Path(high_stats_h5)
    timesteps = parse_timestep_spec(timestep_spec)
    ranks = list(RANKS if ranks is None else ranks)
    rotate_ranks_180 = parse_rank_list(rotate_ranks_180 or "")
    transpose_ranks = parse_rank_list(transpose_ranks or "")

    train_min, train_max, margin = load_high_stats(high_stats_h5)
    domains, nx_tile, ny_tile, xmins, ymins = build_layout(case_folder, ranks, trim_mode=trim_mode)
    h = ny_tile * len(ymins)
    w = nx_tile * len(xmins)
    c = len(PROPS)
    t_len = len(timesteps)

    if output is None:
        diffs = np.diff(timesteps)
        suffix = ""
        if len(diffs) and np.all(diffs == diffs[0]) and int(diffs[0]) != 1:
            suffix = f"_step{int(diffs[0])}"
        if trim_mode != "ranks_bc":
            suffix += f"_{trim_mode}"
        if rotate_ranks_180:
            suffix += suffix_rank_list("rot", rotate_ranks_180)
        if transpose_ranks:
            suffix += suffix_rank_list("transpose", transpose_ranks)
        output = (
            case_folder
            / "SimVPv2_inputs"
            / f"global_norm_from_high3b_trainfixed_minmax_margin{int(round(margin * 100))}"
              f"_t{timesteps[0]}_to_t{timesteps[-1]}{suffix}.h5"
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    timestep_diffs = np.diff(timesteps)
    timestep_stride = (
        int(timestep_diffs[0])
        if len(timestep_diffs) and np.all(timestep_diffs == timestep_diffs[0])
        else 1
    )

    data = np.empty((t_len, c, h, w), dtype=np.float32)
    for ti, timestep in enumerate(timesteps):
        if (ti + 1) % 100 == 0 or ti == 0 or ti + 1 == t_len:
            print(f"[BUILD] timestep {timestep} ({ti + 1}/{t_len})", flush=True)
        frame_hwc = np.empty((h, w, c), dtype=np.float32)
        for ci, prop in enumerate(PROPS):
            frame_hwc[:, :, ci] = build_global_frame(
                case_folder,
                domains,
                ranks,
                timestep,
                prop,
                nx_tile,
                ny_tile,
                xmins,
                ymins,
                rotate_ranks_180=rotate_ranks_180,
                transpose_ranks=transpose_ranks,
                trim_mode=trim_mode,
            )
        data[ti] = np.transpose(normalize_with_high_stats(frame_hwc, train_min, train_max, margin), (2, 0, 1))

    with h5py.File(output, "w") as f:
        f.create_dataset("data_tchw", data=data, compression="gzip", compression_opts=4)
        f.create_dataset("timesteps", data=np.asarray(timesteps, dtype=np.int32))
        f.create_dataset("props", data=np.asarray(PROPS, dtype="S"))
        f.create_dataset("layout", data=np.bytes_("data_tchw[T,C,H,W], normalized with high-magnet testcase 3b train stats"))
        f.create_dataset("detected_tile", data=np.asarray([nx_tile, ny_tile], dtype=np.int32))
        f.create_dataset("detected_blocks", data=np.asarray([len(xmins), len(ymins)], dtype=np.int32))
        f.create_dataset("pre_seq_length", data=np.int32(10))
        f.create_dataset("aft_seq_length", data=np.int32(10))
        f.create_dataset("stride", data=np.int32(timestep_stride))
        f.create_dataset("source_case", data=np.bytes_(str(case_folder)))
        f.create_dataset("stats_source_h5", data=np.bytes_(str(high_stats_h5)))
        f.create_dataset("rotate_ranks_180", data=np.asarray(rotate_ranks_180, dtype=np.int32))
        f.create_dataset("transpose_ranks", data=np.asarray(transpose_ranks, dtype=np.int32))
        f.create_dataset("trim_mode", data=np.bytes_(trim_mode))
        f.create_dataset("train_min", data=train_min)
        f.create_dataset("train_max", data=train_max)
        f.create_dataset("margin", data=np.float32(margin))
        f.create_dataset("norm_mode", data=np.bytes_("minmax_from_high3b_trainfixed_per_channel"))

    print(f"[OK] wrote {output}")
    print(f"     data_tchw={data.shape}, timesteps={timesteps[0]}..{timesteps[-1]}")
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Build low-magnet SimVPv2 H5 normalized with high-magnet testcase 3b train stats."
    )
    parser.add_argument("--case-folder", default=str(DEFAULT_LOW_CASE))
    parser.add_argument("--high-stats-h5", default=str(DEFAULT_HIGH_STATS_H5))
    parser.add_argument("--timesteps", default="0:2:4000")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--trim-mode",
        default="training_compatible",
        choices=["training_compatible", "ranks_bc", "int_offst"],
        help=(
            "How to extract each rank tile. The default reproduces the Julia "
            "pipeline used for the testcase 3b training H5."
        ),
    )
    parser.add_argument(
        "--rotate-ranks-180",
        default="",
        help="Comma-separated ranks to rotate by 180 degrees after overlap trimming, e.g. 1,2.",
    )
    parser.add_argument(
        "--transpose-ranks",
        default="0,1,2,3",
        help="Comma-separated ranks to transpose after overlap trimming.",
    )
    args = parser.parse_args()

    build_h5(
        args.case_folder,
        args.high_stats_h5,
        args.timesteps,
        output=args.output,
        rotate_ranks_180=args.rotate_ranks_180,
        transpose_ranks=args.transpose_ranks,
        trim_mode=args.trim_mode,
    )


if __name__ == "__main__":
    main()
