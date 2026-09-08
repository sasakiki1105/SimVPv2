"""Does the organisation failure follow normalized mode q, or integer mode n?

In the E = 10 kV/m B sweep the dominant transport bin moved to q = 0.21, 0.18,
0.12 while sitting at integer n = 3, 4, 4.  Those two coordinates are almost
degenerate along that sweep, so "low-q failure" and "low-n failure" cannot be
told apart from it.  All ten conditions together break the degeneracy, because
n0 spans 3.58 to 32.25.

For every valid bin of every condition:

    O = r cos(delta) = Re(C)          the density-field organisation factor
    Gamma = -A * O / B

so O carries exactly the part of the transport that is not amplitude.  The
signed ratio O_pred / O_true is reported (a negative value is a sign flip, i.e.
transport predicted in the wrong direction) together with log|ratio|.

Bins are masked twice: |O_true| must exceed a floor, because the ratio is
meaningless where the organisation itself vanishes, and the bin must carry a
non-negligible share of the true transport.

Collapse is then judged quantitatively rather than by eye: bin the surviving
data on each axis, take the transport-weighted mean log|ratio| per condition
per axis-bin, and measure the spread ACROSS conditions.  The axis on which the
conditions agree better is the coordinate the failure actually follows.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import torch

from evaluate_radaz_conditioned_factorial import (
    MANIFEST, build_model, condition_vector, predict_direct10,
)
from evaluate_radaz_physics_loss_v2 import V2, REG_DIR, spectra
from audit_radaz_v2_transport_error_geometry import batch, MAXN
from openstl.methods.pepapic_spectral_loss import PEPAPICSpectralLoss

TEST_LO, TEST_HI = 1800, 2000
O_FLOOR = 0.05          # |Re C| below this is noise, ratio meaningless
W_FLOOR = 1.0e-3        # bin must carry at least 0.1% of the true transport
Q_EDGES = np.array([0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0])
N_EDGES = np.array([0.5, 2.5, 4.5, 6.5, 9.5, 13.5, 18.5, 25.5, 33.5, 45.5, 64.5])


def organisation(pred, true, B):
    pn_p, pe_p, cr_p = pred
    pn_t, pe_t, cr_t = true
    a_p, a_t = np.sqrt(pn_p * pe_p), np.sqrt(pn_t * pe_t)
    o_p = np.real(cr_p) / np.maximum(a_p, 1e-300)
    o_t = np.real(cr_t) / np.maximum(a_t, 1e-300)
    g_t = -np.real(cr_t) / B
    return o_p, o_t, g_t, a_p / np.maximum(a_t, 1e-300)


def profile(values, weights, coord, edges):
    """transport-weighted mean of `values` in each coordinate bin"""
    out = np.full(len(edges) - 1, np.nan)
    for i in range(len(edges) - 1):
        m = (coord >= edges[i]) & (coord < edges[i + 1])
        if m.sum() and weights[m].sum() > 0:
            out[i] = float(np.sum(values[m] * weights[m]) / np.sum(weights[m]))
    return out


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    low = np.asarray(manifest["normalization"]["low"], dtype=np.float64)
    high = np.asarray(manifest["normalization"]["high"], dtype=np.float64)
    dev = torch.device("cuda:0")
    lm = PEPAPICSpectralLoss(data_root=str(MANIFEST), max_mode=MAXN, radial_bands=4,
                             radial_min_m=0.09e-2, radial_max_m=1.19e-2,
                             coordinate_system="integer_power_cross").to(dev)
    model, epoch = build_model(V2["config"],
                               V2["workdir"] / "checkpoints" / "best.ckpt", dev)
    print("[model] v2 epoch=%d ; mask |O_true|>%.2f and w>%.0e"
          % (epoch, O_FLOOR, W_FLOOR))
    modes = np.arange(1, MAXN + 1)

    rows, qprof, nprof, keys = {}, {}, {}, []
    print()
    print("%-11s %-8s %7s %7s %10s %12s %10s" % (
        "case", "role", "n0", "bins", "sign flips", "med O_p/O_t", "med|log|"))
    print("-" * 84)
    for case in manifest["cases"]:
        key = case["case_key"]
        x, y, n0, B = batch(case, manifest, low, high, TEST_LO, TEST_HI)
        o_p, o_t, g_t, aratio = organisation(
            spectra(lm, predict_direct10(model, x, dev), dev),
            spectra(lm, y, dev), B)
        w = g_t ** 2 / (g_t ** 2).sum()
        mask = (np.abs(o_t) > O_FLOOR) & (w > W_FLOOR)
        q = np.broadcast_to(modes / n0, o_t.shape)
        nn = np.broadcast_to(modes.astype(float), o_t.shape)
        ratio = o_p[mask] / o_t[mask]
        ww = w[mask]
        flips = float(np.sum(ww[ratio < 0]) / max(ww.sum(), 1e-300))
        med = float(np.median(ratio))
        logabs = np.log(np.abs(ratio) + 1e-300)
        rows[key] = {"role": case["role"], "n0": n0, "n_bins": int(mask.sum()),
                     "sign_flip_weight": flips, "median_ratio": med,
                     "median_abs_log": float(np.median(np.abs(logabs)))}
        qprof[key] = profile(logabs, ww, q[mask], Q_EDGES)
        nprof[key] = profile(logabs, ww, nn[mask], N_EDGES)
        keys.append(key)
        print("%-11s %-8s %7.2f %7d %10.3f %12.3f %10.3f" % (
            key, case["role"][:7], n0, mask.sum(), flips, med,
            rows[key]["median_abs_log"]))
        del x, y

    def show(name, prof, edges, fmt):
        print()
        print("=" * 104)
        print("transport-weighted mean log|O_pred/O_true| binned on %s" % name)
        print("=" * 104)
        hdr = "".join("%9s" % (fmt % (edges[i], edges[i + 1]))
                      for i in range(len(edges) - 1))
        print("%-11s" % "case" + hdr)
        print("-" * 104)
        for k in keys:
            print("%-11s" % k + "".join(
                "      ---" if np.isnan(v) else "%9.2f" % v for v in prof[k]))
        arr = np.array([prof[k] for k in keys])
        count = np.sum(~np.isnan(arr), axis=0)
        spread = np.nanstd(arr, axis=0)
        centre = np.nanmean(arr, axis=0)
        print("-" * 104)
        print("%-11s" % "mean" + "".join(
            "      ---" if count[i] < 3 else "%9.2f" % centre[i]
            for i in range(len(centre))))
        print("%-11s" % "spread" + "".join(
            "      ---" if count[i] < 3 else "%9.2f" % spread[i]
            for i in range(len(spread))))
        print("%-11s" % "n cond" + "".join("%9d" % c for c in count))
        ok = count >= 3
        return (float(np.nanmedian(spread[ok])) if ok.any() else float("nan"),
                centre, spread, count)

    sq, cq, spq, nq = show("q = n/n0", qprof, Q_EDGES, "%.2f-%.2f")
    sn, cn, spn, nn_ = show("integer n", nprof, N_EDGES, "%.0f-%.0f")

    print()
    print("=" * 104)
    print("COLLAPSE TEST")
    print("=" * 104)
    print("  median across-condition spread, binned on q         : %.3f" % sq)
    print("  median across-condition spread, binned on integer n : %.3f" % sn)
    if sq < sn * 0.85:
        verdict = "collapses better on q -- normalized-resonance coordinate"
    elif sn < sq * 0.85:
        verdict = "collapses better on integer n -- low-order mode family"
    else:
        verdict = ("no clear collapse on either axis -- suspect mode-family / "
                   "regime dependence rather than a single coordinate")
    print()
    print("  VERDICT: " + verdict)

    p = REG_DIR / "v2_organisation_collapse.json"
    p.write_text(json.dumps({
        "mask": {"O_floor": O_FLOOR, "w_floor": W_FLOOR},
        "per_condition": rows,
        "q_edges": Q_EDGES.tolist(), "n_edges": N_EDGES.tolist(),
        "q_profiles": {k: [None if np.isnan(v) else v for v in qprof[k]] for k in keys},
        "n_profiles": {k: [None if np.isnan(v) else v for v in nprof[k]] for k in keys},
        "spread_q": sq, "spread_n": sn, "verdict": verdict}, indent=2),
        encoding="utf-8")
    print()
    print("[written] " + str(p))


if __name__ == "__main__":
    main()
