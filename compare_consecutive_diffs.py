import os
import json
import numpy as np
import h5py

OLD_CASE = r"C:\Users\astro\research\PEPAPIC\test\results\2D_ExB_high_magnet_dt2.5e-9_maxt50e-6"
NEW_CASE = r"C:\Users\astro\research\PEPAPIC\test\results\2D_ExB_high_magnet_dt2.5e-9_maxt50e-6_macro5"

PROP = "phi"

OLD_TS = [900, 901, 902, 903]
NEW_TS = [3600, 3601, 3602, 3603]

def load_domain_info(case_folder, rank):
    p = os.path.join(case_folder, "Domain_info", f"Local_domain_info_rank{rank}.json")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def read_field(case_folder, rank, t, prop):
    h5 = os.path.join(case_folder, "Macro", f"Macro_tn_{t}", f"ISP1_domain_macro_tn_{t}_rank{rank}.h5")
    with h5py.File(h5, "r") as f:
        A = f[prop][...]
    A = np.asarray(A)
    if A.ndim == 3 and A.shape[2] == 1:
        A = A[:, :, 0]
    elif A.ndim == 3 and A.shape[0] == 1:
        A = A[0]
    elif A.ndim == 3 and A.shape[1] == 1:
        A = A[:, 0, :]
    return A.astype(np.float64)

def extract_interior(A, domain_info):
    offs = domain_info["Int_offst"][0]
    nint = domain_info["N_int_sp"][0]
    ox = int(offs[0]); oy = int(offs[1])
    nx = int(nint[0]); ny = int(nint[1])
    return A[ox:ox+nx, oy:oy+ny]

def stitch_case(case_folder, t, prop):
    doms = {r: load_domain_info(case_folder, r) for r in range(4)}
    tiles = []
    for r in range(4):
        A = extract_interior(read_field(case_folder, r, t, prop), doms[r])
        xp = doms[r]["Xp_min"]
        if isinstance(xp[0], (list, tuple)):
            x0 = xp[0][0]
            y0 = xp[0][1]
        else:
            x0 = xp[0]
            y0 = xp[1]
        tiles.append((x0, y0, A))

    xmins = sorted(set(x for x, _, _ in tiles))
    ymins = sorted(set(y for _, y, _ in tiles))
    nx_tile, ny_tile = tiles[0][2].shape
    G = np.zeros((len(xmins)*nx_tile, len(ymins)*ny_tile), dtype=np.float64)

    for x0, y0, A in tiles:
        ix = xmins.index(x0)
        iy = ymins.index(y0)
        G[ix*nx_tile:(ix+1)*nx_tile, iy*ny_tile:(iy+1)*ny_tile] = A

    return np.rot90(G, k=-1)

def diff_stats(seq, case_name):
    print(f"\n[{case_name}]")
    arrs = [stitch_case(case_name, t, PROP) for t in seq]
    for i in range(len(arrs)-1):
        d = arrs[i+1] - arrs[i]
        print(f"{seq[i]} -> {seq[i+1]} : mse={np.mean(d**2):.6g}, mae={np.mean(np.abs(d)):.6g}, std={d.std():.6g}")

def main():
    diff_stats(OLD_TS, OLD_CASE)
    diff_stats(NEW_TS, NEW_CASE)

if __name__ == "__main__":
    main()