"""Is the transport error selectively placed, and does E10_B25 sit on a
mode-family transition?

Two questions, both answerable without retraining.

1. The boring explanation for the top-bin concentration.  If the true transport
   itself lives in five bins, then the error budget living there too is not a
   finding.  So compare

       p_truth_bn = Gamma_bn^2 / sum Gamma^2
       p_err_bn   = (Gamma_hat - Gamma)_bn^2 / sum (Gamma_hat - Gamma)^2

   and report C5 = (error share in the top-5 error bins) / (truth share in the
   SAME bins).  C5 ~ 1 means the error simply follows the transport.  C5 >> 1
   means those modes are being selectively missed.

2. The E = 10 kV/m B sweep, B = 15, 20, 25, 30 mT.  B20 and B30 are TRAINING
   conditions; B15 and B25 are holdout.  For each, locate the q = n/n0 that
   carries the true transport and split the true transport power into a low-q
   band, the resonant band and a high-q band.  If the carrier moves from q ~ 1
   to q << 1 somewhere in the sweep, B25 may be an unresolved transition rather
   than an outlier; if B30 (a training condition) also carries low-q transport
   and is predicted correctly, then the model can represent low-q organisation
   and the B25 failure is an interpolation failure instead.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import torch

from evaluate_radaz_conditioned_factorial import (
    MANIFEST, build_model, condition_vector,
)
from evaluate_radaz_physics_loss_v2 import V2, REG_DIR, spectra
from evaluate_radaz_conditioned_factorial import predict_direct10
from audit_radaz_v2_transport_error_geometry import batch, parts, MAXN
from openstl.methods.pepapic_spectral_loss import PEPAPICSpectralLoss

TEST_LO, TEST_HI = 1800, 2000
SWEEP = ["E10_B15", "E10_B20", "E10_B25", "E10_B30"]


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
    cases = {c["case_key"]: c for c in manifest["cases"]}
    modes = np.arange(1, MAXN + 1)

    # ---------- 1. selective failure audit, all ten conditions ----------
    print()
    print("=" * 108)
    print("1.  IS THE ERROR SELECTIVELY PLACED?   C5 = err share / truth share in the")
    print("    same top-5 error bins.  C5 ~ 1: error follows transport.  C5 >> 1: selective")
    print("=" * 108)
    print("%-11s %-8s %9s %9s %9s %9s" % (
        "case", "role", "err top5", "truth top5", "C5", "truth conc"))
    print("-" * 108)
    audit = {}
    for key, case in cases.items():
        x, y, n0, B = batch(case, manifest, low, high, TEST_LO, TEST_HI)
        g_p, g_t, lr, dr, dd, _ = parts(
            spectra(lm, predict_direct10(model, x, dev), dev), spectra(lm, y, dev), B)
        err = (g_p - g_t) ** 2
        p_err = err / err.sum()
        p_tru = g_t ** 2 / (g_t ** 2).sum()
        idx = np.argsort(p_err.ravel())[::-1][:5]
        e5 = float(p_err.ravel()[idx].sum())
        t5 = float(p_tru.ravel()[idx].sum())
        # how concentrated is the TRUE transport on its own top-5 bins
        tc = float(np.sort(p_tru.ravel())[::-1][:5].sum())
        c5 = e5 / max(t5, 1e-300)
        audit[key] = {"role": case["role"], "n0": n0, "err_top5": e5,
                      "truth_share_same_bins": t5, "C5": c5, "truth_top5_conc": tc}
        print("%-11s %-8s %9.3f %9.3f %9.2f %9.3f" % (
            key, case["role"][:7], e5, t5, c5, tc))
        del x, y
    print()
    print("truth conc = share of the TRUE transport power in ITS own top-5 bins")

    # ---------- 2. the E = 10 kV/m B sweep ----------
    print()
    print("=" * 108)
    print("2.  E = 10 kV/m B SWEEP:  where does the TRUE transport live?")
    print("=" * 108)
    print("%-11s %-8s %7s %8s %10s | %9s %9s %9s" % (
        "case", "role", "n0", "q_Gmax", "n_Gmax", "q<0.5", "0.85-1.15", "q>1.5"))
    print("-" * 108)
    sweep = {}
    for key in SWEEP:
        case = cases[key]
        x, y, n0, B = batch(case, manifest, low, high, TEST_LO, TEST_HI)
        ps = spectra(lm, predict_direct10(model, x, dev), dev)
        ts = spectra(lm, y, dev)
        g_p, g_t, lr, dr, dd, _ = parts(ps, ts, B)
        q = modes / n0
        power = g_t ** 2
        tot = power.sum()
        b, n = np.unravel_index(np.argmax(power), power.shape)
        lowq = float(power[:, q < 0.5].sum() / tot)
        res = float(power[:, (q >= 0.85) & (q <= 1.15)].sum() / tot)
        hiq = float(power[:, q > 1.5].sum() / tot)
        amp_ratio = float(np.exp(lr[b, n]))
        gam_ratio = float(g_p[b, n] / g_t[b, n]) if g_t[b, n] else float("nan")
        sweep[key] = {"role": case["role"], "n0": n0,
                      "q_Gmax": float((n + 1) / n0), "n_Gmax": int(n + 1),
                      "band": int(b), "low_q": lowq, "resonant": res, "high_q": hiq,
                      "amp_ratio": amp_ratio, "gamma_ratio": gam_ratio,
                      "organisation_ratio": gam_ratio / amp_ratio if amp_ratio else float("nan"),
                      "dphase": float(dd[b, n]), "dcoh": float(dr[b, n])}
        print("%-11s %-8s %7.2f %8.2f %10d | %9.3f %9.3f %9.3f" % (
            key, case["role"][:7], n0, sweep[key]["q_Gmax"], n + 1, lowq, res, hiq))
        del x, y

    print()
    print("prediction quality AT EACH CONDITION'S OWN dominant TRUE-transport bin")
    print("%-11s %-8s %6s %8s %10s %10s %14s %9s" % (
        "case", "role", "n", "q", "A_hat/A", "G_hat/G", "(r cosd) ratio", "dphase"))
    print("-" * 108)
    for key in SWEEP:
        v = sweep[key]
        print("%-11s %-8s %6d %8.2f %10.3f %10.3f %14.3f %9.3f" % (
            key, v["role"][:7], v["n_Gmax"], v["q_Gmax"], v["amp_ratio"],
            v["gamma_ratio"], v["organisation_ratio"], v["dphase"]))

    print()
    print("=" * 108)
    print("READING")
    print("=" * 108)
    lowq_train = [k for k in SWEEP if sweep[k]["role"] == "source" and sweep[k]["low_q"] > 0.4]
    lowq_hold = [k for k in SWEEP if sweep[k]["role"] != "source" and sweep[k]["low_q"] > 0.4]
    print("  training conditions carrying low-q transport : %s" % (lowq_train or "none"))
    print("  holdout  conditions carrying low-q transport : %s" % (lowq_hold or "none"))
    if lowq_train:
        worst = min(sweep[k]["organisation_ratio"] for k in lowq_train)
        print("  worst organisation ratio among those TRAINING conditions: %.3f" % worst)
        print("  -> if that is near 1, the model CAN represent low-q organisation,")
        print("     so a low-q holdout failure is an interpolation failure, not a")
        print("     missing capability.")
    else:
        print("  -> no training condition in this sweep carries low-q transport, so a")
        print("     low-q holdout failure cannot be separated from never having seen it.")

    p = REG_DIR / "v2_bsweep_transport_families.json"
    p.write_text(json.dumps({"selective_failure_audit": audit,
                             "b_sweep": sweep}, indent=2), encoding="utf-8")
    print()
    print("[written] " + str(p))


if __name__ == "__main__":
    main()
