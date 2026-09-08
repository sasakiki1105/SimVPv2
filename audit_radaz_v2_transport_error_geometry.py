"""Where does the v2 transport error actually live, and does a local
amplitude calibration transfer?

Global calibration showed that a single scalar c, fitted on source validation
alone, removes the amplitude-product bias on all four holdout conditions --
yet it improved transport in only two of them and made E25_B20 markedly worse.
That points at cancellation between error components rather than at a scale
error, so the error has to be resolved per spectral bin.

Per radial band b and integer azimuthal mode n define

    e_bn = (Gamma_hat_bn - Gamma_bn)^2 / sum_bn Gamma_bn^2

so that sum_bn e_bn = NRMSE_Gamma^2 exactly.  e_bn is then the *share* of the
total transport error contributed by each bin, with no division by a vanishing
truth.  Alongside it: log amplitude ratio, phase error and coherence error per
bin, and the change in e_bn produced by the global calibration.

Two local calibrations are then fitted on the SOURCE conditions' validation
frames only and applied unchanged to the holdout:

  c_n[b,n]  indexed by absolute integer mode  -- the bias is a property of the
            grid / architecture
  c_q[b,q]  indexed by normalized mode q = n/n0 -- the bias is a property of
            the physics, following the resonance

Which one transfers (if either) says whether the residual defect is a fixed
spectral calibration or a condition-dependent power allocation.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import torch

from evaluate_radaz_conditioned_factorial import (
    AFT, MANIFEST, MODEL_H, MODEL_W, PRE,
    build_model, condition_vector, load_case_frames, predict_direct10,
)
from evaluate_radaz_physics_loss_v2 import V2, REG_DIR, spectra
from openstl.methods.pepapic_spectral_loss import PEPAPICSpectralLoss

ME, QE, LY = 9.1093837015e-31, 1.602176634e-19, 1.28e-2
TEST_LO, TEST_HI = 1800, 2000
VAL_LO, VAL_HI = 1600, 1800
QGRID = np.linspace(0.20, 4.00, 39)
MAXN = 64


def batch(case, manifest, low, high, lo, hi):
    starts = [s for s in range(lo, hi, PRE + AFT) if s + PRE + AFT <= hi]
    frames = load_case_frames(case, low, high, lo, lo + len(starts) * (PRE + AFT))[0]
    cond, drift, n0 = condition_vector(case, manifest)
    x = np.empty((len(starts), PRE, 5, MODEL_H, MODEL_W), dtype=np.float32)
    y = np.empty((len(starts), AFT, 3, MODEL_H, MODEL_W), dtype=np.float32)
    for i, s in enumerate(starts):
        j = s - lo
        x[i, :, :3] = frames[j:j + PRE]
        x[i, :, 3:] = cond[None, :, None, None]
        y[i] = frames[j + PRE:j + PRE + AFT]
    B = n0 * 2.0 * np.pi * ME * drift / (QE * LY)
    return x, y, n0, B


def parts(pred, true, B):
    pn_p, pe_p, cr_p = pred
    pn_t, pe_t, cr_t = true
    g_p = -np.real(cr_p) / B
    g_t = -np.real(cr_t) / B
    amp_p, amp_t = np.sqrt(pn_p * pe_p), np.sqrt(pn_t * pe_t)
    logr = np.log(amp_p + 1e-300) - np.log(amp_t + 1e-300)
    r_p = np.abs(cr_p) / np.maximum(amp_p, 1e-300)
    r_t = np.abs(cr_t) / np.maximum(amp_t, 1e-300)
    dd = np.angle(cr_p) - np.angle(cr_t)
    dd = np.arctan2(np.sin(dd), np.cos(dd))
    return g_p, g_t, logr, r_p - r_t, dd, amp_t


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
    print("[model] v2 epoch=%d" % epoch)
    src = [c for c in manifest["cases"] if c["role"] == "source"]
    hol = [c for c in manifest["cases"] if c["role"] != "source"]
    modes = np.arange(1, MAXN + 1)

    # ---- fit local calibrations on SOURCE validation only -------------
    logr_n, logr_q, masked_bias = [], [], []
    for case in src:
        x, y, n0, B = batch(case, manifest, low, high, VAL_LO, VAL_HI)
        _, _, lr, _, _, amp_t = parts(spectra(lm, predict_direct10(model, x, dev), dev),
                                      spectra(lm, y, dev), B)
        # the GLOBAL scalar must use the same power mask as the earlier
        # calibration; an unmasked median is dominated by the low-power bins
        # where the model hallucinates, and collapses c to ~0.004.
        msk = amp_t > 1e-3 * amp_t.max(axis=-1, keepdims=True)
        masked_bias.append(float(np.sum(lr * msk) / max(msk.sum(), 1)))
        logr_n.append(lr)
        q = modes / n0
        logr_q.append(np.stack([np.interp(QGRID, q, lr[b]) for b in range(lr.shape[0])]))
        del x, y
    cn = np.exp(-np.median(np.stack(logr_n), axis=0))          # [R,N]
    cq_curve = np.median(np.stack(logr_q), axis=0)             # [R,Q]
    gbias = float(np.median(masked_bias))
    cglob = float(np.exp(-gbias))
    print("global c = %.4f ; local c_n range %.3f-%.3f" % (cglob, cn.min(), cn.max()))

    print()
    print("=" * 104)
    print("TRANSPORT ERROR GEOMETRY, HOLDOUT (e_bn shares; sum e_bn = NRMSE^2)")
    print("=" * 104)
    out = {}
    print("%-11s %8s %8s %8s %8s %8s %8s %8s" % (
        "case", "NRMSE", "raw", "c_glob", "c_n", "c_q", "top-bin%", "top-5%"))
    print("-" * 104)
    for case in hol:
        key = case["case_key"]
        x, y, n0, B = batch(case, manifest, low, high, TEST_LO, TEST_HI)
        ps = spectra(lm, predict_direct10(model, x, dev), dev)
        ts = spectra(lm, y, dev)
        g_p, g_t, lr, dr, dd, amp_t = parts(ps, ts, B)
        denom = float(np.sum(g_t ** 2))
        e_raw = (g_p - g_t) ** 2 / denom
        nr = float(np.sqrt(e_raw.sum()))
        q = modes / n0
        cq = np.stack([np.exp(-np.interp(q, QGRID, cq_curve[b]))
                       for b in range(cn.shape[0])])
        res = {}
        for lbl, scale in (("c_glob", cglob), ("c_n", cn), ("c_q", cq)):
            e = (scale * g_p - g_t) ** 2 / denom
            res[lbl] = float(np.sqrt(e.sum()))
        flat = np.sort(e_raw.ravel())[::-1]
        print("%-11s %8.4f %8.4f %8.4f %8.4f %8.4f %8.1f %8.1f" % (
            key, nr, nr, res["c_glob"], res["c_n"], res["c_q"],
            100 * flat[0] / e_raw.sum(), 100 * flat[:5].sum() / e_raw.sum()))
        b, n = np.unravel_index(np.argmax(e_raw), e_raw.shape)
        # cancellation at the dominant bin: how much of the amplitude excess
        # survives into Gamma, and what the implied r*cos(delta) error is
        amp_ratio = float(np.exp(lr[b, n]))
        gam_ratio = float(g_p[b, n] / g_t[b, n]) if g_t[b, n] != 0 else float("nan")
        organ_ratio = gam_ratio / amp_ratio if amp_ratio != 0 else float("nan")
        out[key] = {"n0": n0, "nrmse_raw": nr, **res,
                    "top_bin": {"band": int(b), "n": int(n + 1),
                                "q": float((n + 1) / n0),
                                "share": float(flat[0] / e_raw.sum()),
                                "log_amp_ratio": float(lr[b, n]),
                                "dphase_rad": float(dd[b, n]),
                                "dcoherence": float(dr[b, n]),
                                "gamma_true": float(g_t[b, n]),
                                "gamma_pred": float(g_p[b, n]),
                                "amp_ratio": amp_ratio,
                                "gamma_ratio": gam_ratio,
                                "organisation_ratio": organ_ratio},
                   "top5_share": float(flat[:5].sum() / e_raw.sum())}
        del x, y

    print()
    print("dominant bin of each condition (band, integer n, q = n/n0)")
    print("%-11s %5s %4s %6s %8s %10s %9s %10s %10s" % (
        "case", "band", "n", "q", "share%", "log(A/A)", "dphase", "G_true", "G_pred"))
    print("-" * 104)
    for k, v in out.items():
        t = v["top_bin"]
        print("%-11s %5d %4d %6.2f %8.1f %10.3f %9.3f %10.3e %10.3e" % (
            k, t["band"], t["n"], t["q"], 100 * t["share"], t["log_amp_ratio"],
            t["dphase_rad"], t["gamma_true"], t["gamma_pred"]))

    print()
    print("=" * 104)
    print("CANCELLATION AT THE DOMINANT BIN")
    print("=" * 104)
    print("%-11s %12s %12s %14s  %s" % (
        "case", "A_hat/A", "G_hat/G", "(r cosd) ratio", "reading"))
    print("-" * 104)
    for k, v in out.items():
        t = v["top_bin"]
        over = t["amp_ratio"]; gam = t["gamma_ratio"]; org = t["organisation_ratio"]
        note = ("amplitude excess cancelled by weak organisation"
                if org < 0.9 else "no cancellation")
        print("%-11s %12.3f %12.3f %14.3f  %s" % (k, over, gam, org, note))
    print()
    print("A single scalar that divides out A_hat/A therefore drives Gamma to")
    print("(r cosd) ratio, which is further from 1 than G_hat/G wherever the two")
    print("errors were opposing.")

    print()
    print("=" * 104)
    print("LOCAL CALIBRATION TRANSFER (median over the four holdout conditions)")
    print("=" * 104)
    for lbl in ("nrmse_raw", "c_glob", "c_n", "c_q"):
        print("  %-8s median NRMSE_Gamma = %.4f   wins over raw: %d/4" % (
            lbl, float(np.median([out[k][lbl] for k in out])),
            sum(1 for k in out if out[k][lbl] < out[k]["nrmse_raw"])))

    p = REG_DIR / "v2_transport_error_geometry.json"
    p.write_text(json.dumps({"global_c": cglob, "conditions": out}, indent=2),
                 encoding="utf-8")
    print()
    print("[written] " + str(p))


if __name__ == "__main__":
    main()
