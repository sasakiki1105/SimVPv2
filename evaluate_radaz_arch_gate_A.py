"""Default: corrected exploratory A evaluation; --replay-legacy: old gates.

The historical description below applies ONLY to explicit legacy replay.
It is not the current continuation policy; see RADAZ_V3_PROTOCOL.md.

Reads ARCH_CONTRAST_AMENDMENT_reduced.md and refuses to run unless its SHA256
matches the recorded digest, so the thresholds cannot drift.

SOURCE CONDITIONS ONLY.  The holdout is not opened for any cell until B and C
have finished, so that the continuation decision cannot leak holdout
information into the contrast.

Gate A1 -- the effect to be rescued is present:
    Spearman(n0, O_hat/O) over the six source conditions <= -0.70
    median O_hat/O over E10_B20 and E10_B30                <= 0.40

Gate A2 -- reduced A is still a competent baseline, against the original UPv2
source medians:
    field aggregate <= 0.0885   E_P n1-32 <= 5.075
    NRMSE_Gamma     <= 0.569    H         <= 0.01566

Both must pass for cells B and C to be run.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import torch
from scipy.stats import spearmanr

from evaluate_radaz_conditioned_factorial import (
    MANIFEST, build_model, condition_vector, denormalize, predict_direct10,
)
from evaluate_radaz_physics_loss_v2 import REG_DIR, spectra, metrics, load_registration
from audit_radaz_v2_transport_error_geometry import batch, MAXN
from openstl.methods.pepapic_spectral_loss import PEPAPICSpectralLoss

ROOT = Path(r"C:\Users\astro\research\SimVPv2")
AMEND = REG_DIR / "ARCH_CONTRAST_AMENDMENT_reduced.md"
CELL_A = {
    "workdir": ROOT / "workdirs/2D_RadAz/radaz_arch_redA_az64_hidT256_NT4_60ep",
    "config": ROOT / "configs/custom/pepapic/SimVP_gSTA_radaz_redA_60ep.py",
}
A1_RHO, A1_HIGH = -0.70, 0.40
A2 = {"field_aggregate": 0.0885, "E_P_n1-32": 5.075,
      "NRMSE_Gamma": 0.569, "H": 0.01566}
HIGH_N0 = ("E10_B20", "E10_B30")
O_FLOOR, W_FLOOR = 0.05, 1.0e-3
FIELDS = ("electron_den", "ion_den", "phi")


def check_amendment():
    digest = hashlib.sha256(AMEND.read_bytes()).hexdigest()
    recorded = (REG_DIR / "ARCH_CONTRAST_AMENDMENT_reduced.sha256"
                ).read_text(encoding="utf-8").split()[0]
    if digest != recorded:
        raise RuntimeError("Amendment digest mismatch: thresholds have been "
                           "edited.\n  recorded %s\n  actual   %s"
                           % (recorded, digest))
    print("[amendment verified] SHA256 %s" % digest)


def legacy_organisation_ratio(pred, true, B):
    """Historical unweighted implementation, retained only for exact replay."""
    pn_p, pe_p, cr_p = pred
    pn_t, pe_t, cr_t = true
    a_p, a_t = np.sqrt(pn_p * pe_p), np.sqrt(pn_t * pe_t)
    o_p = np.real(cr_p) / np.maximum(a_p, 1e-300)
    o_t = np.real(cr_t) / np.maximum(a_t, 1e-300)
    g = -np.real(cr_t) / B
    w = g ** 2 / (g ** 2).sum()
    m = (np.abs(o_t) > O_FLOOR) & (w > W_FLOOR)
    return float(np.median(o_p[m] / o_t[m])) if m.sum() else float("nan")


def organisation_ratio(pred, true, B):
    from radaz_metrics_v3 import organization
    return organization(pred, true, B)["O_ratio_weighted_median"]


def replay_legacy():
    check_amendment()
    registration = load_registration()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    low = np.asarray(manifest["normalization"]["low"], dtype=np.float64)
    high = np.asarray(manifest["normalization"]["high"], dtype=np.float64)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lm = PEPAPICSpectralLoss(data_root=str(MANIFEST), max_mode=MAXN, radial_bands=4,
                             radial_min_m=0.09e-2, radial_max_m=1.19e-2,
                             coordinate_system="integer_power_cross").to(dev)
    ckpt = CELL_A["workdir"] / "checkpoints" / "best.ckpt"
    if not ckpt.exists():
        raise SystemExit("cell A checkpoint not present yet: %s" % ckpt)
    model, epoch = build_model(CELL_A["config"], ckpt, dev)
    print("[model] reduced cell A, epoch=%d, params=%.1f M"
          % (epoch, sum(p.numel() for p in model.parameters()) / 1e6))

    rows = {}
    print()
    print("%-11s %8s %12s %12s %10s %10s %10s" % (
        "case", "n0", "O_hat/O", "field agg", "E_P", "NRMSE_G", "H"))
    print("-" * 80)
    for case in manifest["cases"]:
        if case["role"] != "source":
            continue
        key = case["case_key"]
        x, y, n0, B = batch(case, manifest, low, high, 1800, 2000)
        pred = predict_direct10(model, x, dev)
        ps, ts = spectra(lm, pred, dev), spectra(lm, y, dev)
        m = metrics(ps, ts, registration, key, B)
        pp, yy = denormalize(pred, low, high), denormalize(y, low, high)
        m["field_aggregate"] = float(np.mean([
            np.sqrt(np.mean((pp[:, :, i] - yy[:, :, i]) ** 2)
                    / max(np.mean(yy[:, :, i] ** 2), 1e-300))
            for i in range(3)]))
        m["O_ratio"] = legacy_organisation_ratio(ps, ts, B)
        m["n0"] = n0
        rows[key] = m
        print("%-11s %8.2f %12.4f %12.5f %10.4f %10.4f %10.5f" % (
            key, n0, m["O_ratio"], m["field_aggregate"], m["E_P_n1-32"],
            m["NRMSE_Gamma"], m["H"]))
        del x, y

    n0 = np.array([rows[k]["n0"] for k in rows])
    o = np.array([rows[k]["O_ratio"] for k in rows])
    good = np.isfinite(o)
    rho, p = spearmanr(n0[good], o[good])
    high_med = float(np.median([rows[k]["O_ratio"] for k in HIGH_N0]))

    print()
    print("=" * 80)
    print("GATE A1 -- is the effect to be rescued present?")
    print("=" * 80)
    a1a = rho <= A1_RHO
    a1b = high_med <= A1_HIGH
    print("  [%s] Spearman(n0, O_hat/O) = %+0.4f (p=%.4f)   required <= %.2f"
          % ("PASS" if a1a else "FAIL", rho, p, A1_RHO))
    print("  [%s] median O_hat/O over %s = %.4f   required <= %.2f"
          % ("PASS" if a1b else "FAIL", "/".join(HIGH_N0), high_med, A1_HIGH))
    print("       (original UPv2 reference: rho = -0.9856, high-n0 median = 0.2285)")

    print()
    print("=" * 80)
    print("GATE A2 -- is reduced A still a competent baseline?")
    print("=" * 80)
    a2 = True
    for key, limit in A2.items():
        med = float(np.median([rows[k][key] for k in rows]))
        ok = med <= limit
        a2 = a2 and ok
        print("  [%s] source median %-16s = %9.5f   limit %.5f"
              % ("PASS" if ok else "FAIL", key, med, limit))

    a1 = a1a and a1b
    print()
    print("=" * 80)
    if a1 and a2:
        verdict = "HISTORICAL criteria pass; this legacy replay does not authorize B/C"
    elif not a1:
        verdict = ("STOP under historical A1 criteria. This does not identify "
                   "capacity, normalization, or resolution as the cause.")
    else:
        verdict = "STOP under historical A2 competence criteria; mechanism unresolved."
    print("VERDICT: " + verdict)
    print("=" * 80)

    out = REG_DIR / "arch_gate_A_result_legacy_replay.json"
    out.write_text(json.dumps({
        "metric_status": "legacy_unweighted_mean_field_proxy",
        "architecture_continuation_authorized": False,
        "amendment_sha256": hashlib.sha256(AMEND.read_bytes()).hexdigest(),
        "epoch": epoch, "per_condition": rows,
        "A1": {"spearman": float(rho), "p": float(p), "pass_rho": bool(a1a),
               "high_n0_median": high_med, "pass_high": bool(a1b),
               "pass": bool(a1)},
        "A2": {k: float(np.median([rows[c][k] for c in rows])) for k in A2},
        "A2_pass": bool(a2), "verdict": verdict}, indent=2), encoding="utf-8")
    print()
    print("[written] " + str(out))


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--replay-legacy"]:
        replay_legacy()
    else:
        from evaluate_radaz_corrected_A import main
        main()
