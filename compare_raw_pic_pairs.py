import os
import json
import numpy as np
import h5py

OLD_CASE = r"C:\Users\astro\research\PEPAPIC\test\results\2D_ExB_high_magnet_dt2.5e-9_maxt50e-6"
NEW_CASE = r"C:\Users\astro\research\PEPAPIC\test\results\2D_ExB_high_magnet_dt2.5e-9_maxt50e-6_macro5"

PROPS = ["phi", "electron_den", "ion_den"]

# 物理時間対応で比較するペア
PAIRS = [
    (900, 3600),
    (901, 3604),
    (950, 3800),
    (999, 3996),
]

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
        A = read_field(case_folder, r, t, prop)
        A = extract_interior(A, doms[r])

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
        xr = slice(ix*nx_tile, (ix+1)*nx_tile)
        yr = slice(iy*ny_tile, (iy+1)*ny_tile)
        G[xr, yr] = A

    # 前回の可視化に合わせて 90° clockwise
    G = np.rot90(G, k=-1)
    return G

def stats(a):
    return dict(
        min=float(a.min()),
        max=float(a.max()),
        mean=float(a.mean()),
        std=float(a.std())
    )

def peak_info(a):
    idx = np.unravel_index(np.argmax(a), a.shape)
    return dict(
        peak_val=float(a[idx]),
        peak_pos=(int(idx[0]), int(idx[1]))
    )

def compare(a, b):
    af = a.ravel()
    bf = b.ravel()

    mse = float(np.mean((af - bf)**2))
    mae = float(np.mean(np.abs(af - bf)))

    if af.std() == 0 or bf.std() == 0:
        corr = np.nan
    else:
        corr = float(np.corrcoef(af, bf)[0, 1])

    pa = np.unravel_index(np.argmax(a), a.shape)
    pb = np.unravel_index(np.argmax(b), b.shape)
    peak_dist = float(np.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2))

    return dict(
        mse=mse,
        mae=mae,
        corr=corr,
        peak_old=(int(pa[0]), int(pa[1])),
        peak_new=(int(pb[0]), int(pb[1])),
        peak_dist=peak_dist
    )

def main():
    for prop in PROPS:
        print("="*80)
        print("PROP:", prop)
        print("="*80)
        for told, tnew in PAIRS:
            a = stitch_case(OLD_CASE, told, prop)
            b = stitch_case(NEW_CASE, tnew, prop)

            print(f"\nOLD tn={told}  <->  NEW tn={tnew}")
            print("old stats :", stats(a))
            print("new stats :", stats(b))
            print("old peak  :", peak_info(a))
            print("new peak  :", peak_info(b))
            print("compare   :", compare(a, b))

if __name__ == "__main__":
    main()