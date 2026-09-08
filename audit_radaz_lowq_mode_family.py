"""What is the low-q transport branch?  Truth only, no model.

Along the E = 10 kV/m sweep the transport carrier moves from q ~ 1 to q << 1 as
B rises.  Before attributing the surrogate's organisation failure to a
translator resolution limit, that physical transition has to be characterised
on its own, because the failure onset and the transition coincide in B.

For each of B = 15, 20, 25, 30 mT a three-dimensional spectrum is taken over
(radial x, azimuthal y, time t) of the electron density and the potential, and
the transport cospectrum is formed:

    E_theta(k) = -i k_theta phi(k)
    T(k_r, n, omega) = -Re[ n_e(k) conj(E_theta(k)) ] / B

so T is resolved in radial wavenumber, azimuthal mode and frequency.  The
azimuthal direction is periodic, so its transform is exact; the radial
direction has Dirichlet walls, so a Hann window is applied in x and k_r is
read as a characteristic radial scale rather than an exact eigenvalue.  Time is
Hann windowed too.

The windowed radial centroid is a descriptive spatial-scale statistic.
A centroid below the first nonzero FFT bin does not identify k_r=0;
even a radially constant signal is broadened by the Hann window. These
statistics alone cannot identify ECDI/MTSI eigenmodes or disambiguate temporal
alias branches. Electron drift velocity is not a general wave phase velocity.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import h5py
import numpy as np

from evaluate_radaz_conditioned_factorial import MANIFEST, condition_vector

LO, HI = 1488, 2000          # 512 frames, inside the steady window
MAXN = 40
SWEEP = ["E10_B15", "E10_B20", "E10_B25", "E10_B30"]
ME, QE, LY = 9.1093837015e-31, 1.602176634e-19, 1.28e-2


def load(case, names=("electron_den", "phi")):
    with h5py.File(case["path"], "r") as f:
        out = [np.asarray(f["fields/" + n][LO:HI, :256, :256], dtype=np.float64)
               for n in names]
        x = np.asarray(f["axes/x_m"], dtype=np.float64)[:256]
        t = np.asarray(f["axes/time_s"], dtype=np.float64)[LO:HI]
    return out, x, t


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cases = {c["case_key"]: c for c in manifest["cases"]}
    results = {}

    for key in SWEEP:
        case = cases[key]
        _, drift, n0 = condition_vector(case, manifest)
        B = n0 * 2.0 * np.pi * ME * drift / (QE * LY)
        (ne, phi), x, t = load(case)
        nt, nx, ny = ne.shape
        dt = float(np.median(np.diff(t)))
        dx = float(np.median(np.diff(x)))
        dy = LY / ny

        # fluctuations about the time-and-azimuth mean profile
        ne = ne - ne.mean(axis=(0, 2), keepdims=True)
        phi = phi - phi.mean(axis=(0, 2), keepdims=True)

        wt = np.hanning(nt)[:, None, None]
        wx = np.hanning(nx)[None, :, None]
        ne = ne * wt * wx
        phi = phi * wt * wx

        # azimuthal (periodic, exact) then radial (windowed) then time
        ne_k = np.fft.rfft(ne, axis=2, norm="forward")[:, :, :MAXN + 1]
        ph_k = np.fft.rfft(phi, axis=2, norm="forward")[:, :, :MAXN + 1]
        ne_k = np.fft.fft(ne_k, axis=1, norm="forward")
        ph_k = np.fft.fft(ph_k, axis=1, norm="forward")
        ne_k = np.fft.fft(ne_k, axis=0, norm="forward")
        ph_k = np.fft.fft(ph_k, axis=0, norm="forward")

        modes = np.arange(MAXN + 1)
        ky = 2.0 * np.pi * modes / LY
        e_k = -1j * ky[None, None, :] * ph_k
        T = -np.real(ne_k * np.conj(e_k)) / B          # [omega, k_r, n]

        kr = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
        fr = np.fft.fftfreq(nt, d=dt) * 1.0e-6         # MHz
        w = np.abs(T)

        rows = []
        for n in range(1, MAXN + 1):
            wn = w[:, :, n]
            if wn.sum() <= 0:
                continue
            tot = float(T[:, :, n].sum())
            mkr = float(np.sum(np.abs(kr)[None, :] * wn) / wn.sum())
            mf = float(np.sum(np.abs(fr)[:, None] * wn) / wn.sum())
            i, j = np.unravel_index(np.argmax(wn), wn.shape)
            rows.append({"n": n, "q": n / n0, "T_sum": tot,
                         "T_weight": float(wn.sum()),
                         "mean_abs_kr": mkr, "mean_abs_f_MHz": mf,
                         "peak_kr": float(kr[j]), "peak_f_MHz": float(fr[i])})
        tw = sum(r["T_weight"] for r in rows)
        for r in rows:
            r["share"] = r["T_weight"] / tw
        results[key] = {"n0": n0, "role": case["role"], "rows": rows}

        print("=" * 100)
        print("%s   n0 = %.2f   role = %s" % (key, n0, case["role"]))
        print("=" * 100)
        print("%5s %6s %9s %12s %12s %12s %12s" % (
            "n", "q", "share", "mean|k_r|", "peak k_r", "mean f MHz", "peak f MHz"))
        print("-" * 100)
        for r in sorted(rows, key=lambda a: -a["share"])[:8]:
            print("%5d %6.2f %9.3f %12.1f %12.1f %12.2f %12.2f" % (
                r["n"], r["q"], r["share"], r["mean_abs_kr"], r["peak_kr"],
                r["mean_abs_f_MHz"], r["peak_f_MHz"]))
        del ne, phi, ne_k, ph_k, T, w

    # ---- track the branches across the sweep ----
    print()
    print("=" * 100)
    print("BRANCH TRACKING: transport-weighted properties of the low-q and")
    print("resonant bands (low-q: q < 0.5, resonant: 0.85 <= q <= 1.15)")
    print("=" * 100)
    print("%-11s %7s | %8s %10s %10s | %8s %10s %10s" % (
        "case", "n0", "lowq sh", "|k_r|", "f MHz", "res sh", "|k_r|", "f MHz"))
    print("-" * 100)
    for key in SWEEP:
        r = results[key]
        def agg(sel):
            rs = [a for a in r["rows"] if sel(a["q"])]
            wsum = sum(a["T_weight"] for a in rs)
            if wsum <= 0:
                return 0.0, float("nan"), float("nan")
            return (sum(a["share"] for a in rs),
                    sum(a["mean_abs_kr"] * a["T_weight"] for a in rs) / wsum,
                    sum(a["mean_abs_f_MHz"] * a["T_weight"] for a in rs) / wsum)
        ls, lk, lf = agg(lambda q: q < 0.5)
        rs_, rk, rf = agg(lambda q: 0.85 <= q <= 1.15)
        print("%-11s %7.2f | %8.3f %10.1f %10.2f | %8.3f %10.1f %10.2f" % (
            key, r["n0"], ls, lk, lf, rs_, rk, rf))

    print()
    print("k_r is in rad/m; the radial box is %.4f m so the lowest resolved" % LY)
    print("radial wavenumber is about %.0f rad/m." % (2 * np.pi / LY))
    print("This is the nonzero FFT bin spacing, not a lower bound on a weighted centroid.")
    print("Hann broadening and radial envelopes prevent identification of pure/oblique eigenmodes from this centroid alone.")
    print("Frequencies are sampled frequencies; physical alias branches remain unidentified.")

    out = (MANIFEST.parent.parent / "radaz_physics_loss_v2_manifests"
           / "lowq_mode_family.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print()
    print("[written] " + str(out))


if __name__ == "__main__":
    main()
