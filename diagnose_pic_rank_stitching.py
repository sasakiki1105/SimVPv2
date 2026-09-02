import argparse
import itertools
import json
from pathlib import Path

import h5py
import numpy as np


DEFAULT_CASE = Path(
    r"C:\Users\astro\research\PEPAPIC\test\results"
    r"\2D_ExB_low_magnet_dt2.5e-9_maxt50e-6_macro5"
)
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


def load_domain(case_folder, rank):
    with open(case_folder / "Domain_info" / f"Local_domain_info_rank{rank}.json", encoding="utf-8") as f:
        return json.load(f)


def read_field(case_folder, rank, timestep, prop):
    candidates = [
        case_folder / "Macro" / f"Macro_tn_{timestep}" / f"ISP1_domain_macro_tn_{timestep}_rank{rank}.h5",
        case_folder / f"Macro_tn_{timestep}" / f"ISP1_domain_macro_tn_{timestep}_rank{rank}.h5",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"Missing H5 for timestep={timestep}, rank={rank}")
    with h5py.File(path, "r") as f:
        arr = np.asarray(f[prop][()], dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D data, got {arr.shape}")
    return arr


def trim_int_offset(arr, domain):
    offset = domain["Int_offst"][0]
    n_int = domain["N_int_sp"][0]
    i0 = int(offset[0])
    j0 = int(offset[1])
    return arr[i0:i0 + int(n_int[0]), j0:j0 + int(n_int[1])]


def trim_ranks_bc(arr, domain):
    xminus, xplus, yminus, yplus = domain["ranks_bc"][:4]
    i0 = 1 if xminus != -1 else 0
    i1 = arr.shape[0] - 1 if xplus != -1 else arr.shape[0]
    j0 = 1 if yminus != -1 else 0
    j1 = arr.shape[1] - 1 if yplus != -1 else arr.shape[1]
    return arr[i0:i1, j0:j1]


def transform_tile(tile, name):
    if name == "id":
        return tile
    if name == "rot90":
        return np.rot90(tile, 1)
    if name == "rot180":
        return np.rot90(tile, 2)
    if name == "rot270":
        return np.rot90(tile, 3)
    if name == "flipx":
        return tile[::-1, :]
    if name == "flipy":
        return tile[:, ::-1]
    if name == "transpose":
        return tile.T
    if name == "transpose_rot180":
        return tile.T[::-1, ::-1]
    raise ValueError(name)


TRANSFORMS = ["id", "rot90", "rot180", "rot270", "flipx", "flipy", "transpose", "transpose_rot180"]


def seam_score(tiles):
    # tiles are indexed by physical rank position:
    # rank0 bottom-left, rank1 bottom-right, rank2 top-left, rank3 top-right.
    pairs = [
        (tiles[0][-1, :], tiles[1][0, :]),   # rank0 x+ to rank1 x-
        (tiles[2][-1, :], tiles[3][0, :]),   # rank2 x+ to rank3 x-
        (tiles[0][:, -1], tiles[2][:, 0]),   # rank0 y+ to rank2 y-
        (tiles[1][:, -1], tiles[3][:, 0]),   # rank1 y+ to rank3 y-
    ]
    vals = []
    for a, b in pairs:
        vals.append(float(np.mean(np.abs(a - b))))
    return float(np.mean(vals)), vals


def load_tiles(case_folder, timestep, prop, trim_mode):
    domains = {r: load_domain(case_folder, r) for r in RANKS}
    tiles = {}
    for rank in RANKS:
        arr = read_field(case_folder, rank, timestep, prop)
        if trim_mode == "int_offst":
            tiles[rank] = trim_int_offset(arr, domains[rank])
        elif trim_mode == "ranks_bc":
            tiles[rank] = trim_ranks_bc(arr, domains[rank])
        else:
            raise ValueError(trim_mode)
    return tiles


def score_combo(raw_tiles_by_time, combo):
    total = 0.0
    edges = np.zeros(4, dtype=np.float64)
    count = 0
    for raw_tiles in raw_tiles_by_time:
        tiles = {rank: transform_tile(raw_tiles[rank], combo[rank]) for rank in RANKS}
        score, edge_scores = seam_score(tiles)
        total += score
        edges += np.asarray(edge_scores)
        count += 1
    return total / count, edges / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-folder", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--timesteps", default="0,38,400,1000,2000,3000,4000")
    parser.add_argument("--prop", default="phi")
    parser.add_argument("--trim-mode", choices=["int_offst", "ranks_bc"], default="int_offst")
    parser.add_argument("--topn", type=int, default=20)
    args = parser.parse_args()

    timesteps = parse_timestep_spec(args.timesteps)
    raw_tiles_by_time = [load_tiles(args.case_folder, t, args.prop, args.trim_mode) for t in timesteps]

    fixed = {
        "all_id": {0: "id", 1: "id", 2: "id", 3: "id"},
        "rot12": {0: "id", 1: "rot180", 2: "rot180", 3: "id"},
        "transpose_all": {0: "transpose", 1: "transpose", 2: "transpose", 3: "transpose"},
    }
    print(f"case={args.case_folder}")
    print(f"timesteps={timesteps}")
    print(f"prop={args.prop} trim_mode={args.trim_mode}")
    print("fixed combos:")
    for name, combo in fixed.items():
        score, edges = score_combo(raw_tiles_by_time, combo)
        print(f"  {name:14s} score={score:.8g} edges={','.join(f'{x:.8g}' for x in edges)} combo={combo}")

    best = []
    # Fix rank0 to identity to remove global orientation degeneracy.
    for names in itertools.product(TRANSFORMS, repeat=3):
        combo = {0: "id", 1: names[0], 2: names[1], 3: names[2]}
        score, edges = score_combo(raw_tiles_by_time, combo)
        best.append((score, edges, combo))
    best.sort(key=lambda x: x[0])

    print("best combos with rank0 fixed to id:")
    for score, edges, combo in best[: args.topn]:
        print(f"  score={score:.8g} edges={','.join(f'{x:.8g}' for x in edges)} combo={combo}")


if __name__ == "__main__":
    main()
