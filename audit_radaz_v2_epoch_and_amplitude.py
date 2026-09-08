"""Two cheap secondary analyses after the v2 verdict.  Neither touches the
frozen gates: best.ckpt remains the checkpoint the registered evaluation used.

A. Epoch 13 vs epoch 59 on the holdout conditions.
   best.ckpt is selected on source-condition validation only.  If the late
   checkpoint is better on holdout physics while being worse on source
   validation, then the selection objective and the transferable-physics
   objective are not the same thing.  U-P and U-D are included as controls, so
   a difference can be attributed to v2 rather than to the schedule.

B. Post-hoc amplitude calibration.
   v2 improved the relative phase on holdout but not the amplitude product.
   Fit a single scalar c to the amplitude-product bias on the SOURCE
   conditions' validation frames (1600-1799) -- never on holdout, never on the
   1800-1999 evaluation window -- then apply it to the holdout predictions and
   recompute the transport error.  If that alone recovers transport, the
   residual problem is an amplitude scale.  If c does not transfer, the
   amplitude error is condition-dependent, which is a deeper problem.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import torch

from evaluate_radaz_conditioned_factorial import (
    AFT, CELLS, MANIFEST, MODEL_H, MODEL_W, PRE,
    build_model, condition_vector, denormalize, load_case_frames, predict_direct10,
)
from evaluate_radaz_physics_loss_v2 import V2, REG_DIR, spectra, metrics, load_registration
from openstl.methods.pepapic_spectral_loss import PEPAPICSpectralLoss

ELECTRON_MASS, ELECTRON_CHARGE, LY = 9.1093837015e-31, 1.602176634e-19, 1.28e-2
TEST_LO, TEST_HI = 1800, 2000       # registered evaluation window
VAL_LO, VAL_HI = 1600, 1800         # source validation split, used only for c
FIELDS = ("electron_den", "ion_den", "phi")


def windows(frames, lo, hi):
    return [s for s in range(lo, hi, PRE + AFT) if s + PRE + AFT <= hi]


def build_batch(case, manifest, low, high, lo, hi):
    starts = windows(None, lo, hi)
    frames = load_case_frames(case, low, high, lo, lo + len(starts) * (PRE + AFT))[0]
    condition, drift, n0 = condition_vector(case, manifest)
    x = np.empty((len(starts), PRE, 5, MODEL_H, MODEL_W), dtype=np.float32)
    y = np.empty((len(starts), AFT, 3, MODEL_H, MODEL_W), dtype=np.float32)
    for i, s in enumerate(starts):
        j = s - lo
        x[i, :, :3] = frames[j: j + PRE]
        x[i, :, 3:] = condition[None, :, None, None]
        y[i] = frames[j + PRE: j + PRE + AFT]
    B = n0 * 2.0 * np.pi * ELECTRON_MASS * drift / (ELECTRON_CHARGE * LY)
    return x, y, n0, B


def amplitude_bias(pred, true):
    """signed mean log( (A_N A_E)_pred / (A_N A_E)_true ) on the training mask."""
    pn_p, pe_p, _ = pred
    pn_t, pe_t, _ = true
    g_t = np.sqrt(pn_t * pe_t)
    mask = g_t > 1e-3 * g_t.max(axis=-1, keepdims=True)
    ratio = np.log(np.sqrt(pn_p * pe_p) + 1e-300) - np.log(g_t + 1e-300)
    return float(np.sum(ratio * mask) / max(mask.sum(), 1))


def main():
    registration = load_registration()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    low = np.asarray(manifest["normalization"]["low"], dtype=np.float64)
    high = np.asarray(manifest["normalization"]["high"], dtype=np.float64)
    device = torch.device("cuda:0")
    lm = PEPAPICSpectralLoss(
        data_root=str(MANIFEST), max_mode=64, radial_bands=4,
        radial_min_m=0.09e-2, radial_max_m=1.19e-2,
        coordinate_system="integer_power_cross").to(device)

    src = [c for c in manifest["cases"] if c["role"] == "source"]
    hol = [c for c in manifest["cases"] if c["role"] != "source"]
    specs = {"v2": V2, "U-P": CELLS["U-P"], "U-D": CELLS["U-D"]}

    # ---------------- A. epoch comparison on holdout ----------------
    print("=" * 104)
    print("A.  EPOCH 13/10 (validation-selected) vs EPOCH 59 (last), HOLDOUT conditions")
    print("=" * 104)
    keys = [("E_P_n1-32", "E_P"), ("H", "H"), ("E_A", "E_A"),
            ("E_delta", "E_delta"), ("NRMSE_Gamma", "NRMSE_G"),
            ("field_aggregate", "field")]
    table = {}
    for name, spec in specs.items():
        for tag in ("best", "last"):
            model, epoch = build_model(spec["config"],
                                       spec["workdir"] / "checkpoints" / (tag + ".ckpt"),
                                       device)
            rows = {}
            for case in hol:
                x, y, n0, B = build_batch(case, manifest, low, high, TEST_LO, TEST_HI)
                m = metrics(spectra(lm, predict_direct10(model, x, device), device),
                            spectra(lm, y, device), registration, case["case_key"], B)
                pp = denormalize(predict_direct10(model, x, device), low, high)
                yy = denormalize(y, low, high)
                agg = []
                for i, f in enumerate(FIELDS):
                    agg.append(float(np.sqrt(np.mean((pp[:, :, i] - yy[:, :, i]) ** 2)
                                             / max(np.mean(yy[:, :, i] ** 2), 1e-300))))
                m["field_aggregate"] = float(np.mean(agg))
                rows[case["case_key"]] = m
                del x, y
            table[(name, tag)] = (epoch, rows)
            del model
            torch.cuda.empty_cache()
            print("  [%s %s] epoch %d done" % (name, tag, epoch), flush=True)

    print()
    print("%-6s %-6s %6s " % ("model", "ckpt", "epoch")
          + " ".join("%10s" % l for _, l in keys))
    print("-" * 104)
    for name in specs:
        for tag in ("best", "last"):
            epoch, rows = table[(name, tag)]
            med = [float(np.median([rows[c][k] for c in rows])) for k, _ in keys]
            print("%-6s %-6s %6d " % (name, tag, epoch)
                  + " ".join("%10.4f" % v for v in med))
        e0, r0 = table[(name, "best")]
        e1, r1 = table[(name, "last")]
        deltas = []
        for k, _ in keys:
            a = float(np.median([r1[c][k] for c in r1]))
            b = float(np.median([r0[c][k] for c in r0]))
            deltas.append(a - b)
        print("%-6s %-6s %6s " % ("", "last-best", "")
              + " ".join("%+10.4f" % v for v in deltas))
        print()

    # ---------------- B. amplitude calibration ----------------
    print("=" * 104)
    print("B.  POST-HOC AMPLITUDE CALIBRATION  (c fitted on SOURCE validation frames %d-%d)"
          % (VAL_LO, VAL_HI - 1))
    print("=" * 104)
    model, epoch = build_model(V2["config"], V2["workdir"] / "checkpoints" / "best.ckpt",
                               device)
    biases = []
    for case in src:
        x, y, n0, B = build_batch(case, manifest, low, high, VAL_LO, VAL_HI)
        b = amplitude_bias(spectra(lm, predict_direct10(model, x, device), device),
                           spectra(lm, y, device))
        biases.append(b)
        print("  %-11s source-val amplitude log-bias %+0.4f" % (case["case_key"], b))
        del x, y
    bias = float(np.median(biases))
    c = float(np.exp(-bias))
    print()
    print("  median source-val log-bias %+0.4f  ->  c = exp(-bias) = %.4f" % (bias, c))
    print("  (holdout truth and the 1800-1999 window were not used to fit c)")

    print()
    print("%-11s %10s %10s %10s | %10s %10s" % (
        "case", "E_A raw", "E_A cal", "holdout bias", "NRMSE raw", "NRMSE cal"))
    print("-" * 78)
    raw, cal = [], []
    for case in hol:
        x, y, n0, B = build_batch(case, manifest, low, high, TEST_LO, TEST_HI)
        ps = spectra(lm, predict_direct10(model, x, device), device)
        ts = spectra(lm, y, device)
        m = metrics(ps, ts, registration, case["case_key"], B)
        hb = amplitude_bias(ps, ts)
        # scaling the predicted amplitude product by c scales Gamma by c
        pn_p, pe_p, cr_p = ps
        pn_t, pe_t, cr_t = ts
        g_p = -np.real(cr_p) / B
        g_t = -np.real(cr_t) / B
        nr = float(np.sqrt(np.mean((g_p - g_t) ** 2)) / np.sqrt(np.mean(g_t ** 2)))
        nc = float(np.sqrt(np.mean((c * g_p - g_t) ** 2)) / np.sqrt(np.mean(g_t ** 2)))
        ea_cal = abs(hb + np.log(c))
        raw.append(nr); cal.append(nc)
        print("%-11s %10.4f %10.4f %+10.4f | %10.4f %10.4f %s" % (
            case["case_key"], m["E_A"], ea_cal, hb, nr, nc,
            "better" if nc < nr else "worse"))
        del x, y
    print("-" * 78)
    print("%-11s %10s %10s %10s | %10.4f %10.4f" % (
        "MEDIAN", "", "", "", float(np.median(raw)), float(np.median(cal))))

    out = REG_DIR / "v2_secondary_epoch_and_amplitude.json"
    out.write_text(json.dumps({
        "note": "secondary analyses; the frozen gates used best.ckpt and are unchanged",
        "epoch_comparison": {"%s/%s" % (n, t): {"epoch": e,
                                                "per_condition": r}
                             for (n, t), (e, r) in table.items()},
        "amplitude_calibration": {
            "fit_window_frames": [VAL_LO, VAL_HI - 1],
            "fit_conditions": [c0["case_key"] for c0 in src],
            "source_val_log_biases": biases,
            "median_log_bias": bias, "c": c,
            "holdout_nrmse_raw": raw, "holdout_nrmse_calibrated": cal},
    }, indent=2), encoding="utf-8")
    print()
    print("[written] " + str(out))


if __name__ == "__main__":
    main()
