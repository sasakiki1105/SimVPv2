"""Audit 1b: where does the raw-n mode failure come from?

Audit 1 found that raw-n unexplained power is small over n=1-32 as a whole
(0.09-0.25) but explodes in the fixed band n=9-21 for the low-n0 conditions
(E40 99.8, E30 19.1, E25 11.9).  An unexplained ratio far above 1 cannot be a
phase error alone -- it means the error power exceeds the true power there, so
the model is putting power where the truth has none.

This separates the two possibilities per band:

    power_ratio = sum|pred|^2 / sum|true|^2      (>1 = spurious added power)
    phase_only  = unexplained score if the predicted amplitude were rescaled
                  to the true amplitude per mode (isolates the phase error)

and reports whether the band lies inside the q window [0.3 n0, 1.5 n0] that
the physics loss and the earlier q-space evaluation actually looked at.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import torch

from evaluate_radaz_conditioned_factorial import (
    AFT, CELLS, MANIFEST, MODEL_H, MODEL_W, PRE, TEST_START, TEST_STOP,
    build_model, condition_vector, load_case_frames, predict_direct10,
)
from audit_radaz_q_interpolation_and_raw_modes import band_coeff
from openstl.methods.pepapic_spectral_loss import PEPAPICSpectralLoss

CELL_ORDER = ["U-D", "C-D", "U-P", "C-P"]
BANDS = {"n1-6": (1, 6), "n9-21": (9, 21), "n22-64": (22, 64)}


def stats(pred, true, lo, hi):
    sl = slice(lo - 1, hi)
    p = pred[..., sl]
    t = true[..., sl]
    pw_p = float(np.sum(np.abs(p) ** 2))
    pw_t = float(np.sum(np.abs(t) ** 2))
    # amplitude-matched: rescale each predicted mode to the true amplitude,
    # keeping its phase.  What is left is purely a phase error.
    scale = np.abs(t) / np.maximum(np.abs(p), 1e-300)
    matched = p * scale
    phase_only = float(np.sum(np.abs(matched - t) ** 2) / max(pw_t, 1e-300))
    return {
        "power_ratio": pw_p / max(pw_t, 1e-300),
        "unexplained": float(np.sum(np.abs(p - t) ** 2) / max(pw_t, 1e-300)),
        "phase_only": phase_only,
        "true_power": pw_t,
    }


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    low = np.asarray(manifest["normalization"]["low"], dtype=np.float64)
    high = np.asarray(manifest["normalization"]["high"], dtype=np.float64)
    device = torch.device("cuda:0")
    loss_module = PEPAPICSpectralLoss(
        data_root=str(MANIFEST), max_mode=64, radial_bands=4,
        radial_min_m=0.09e-2, radial_max_m=1.19e-2,
        coordinate_system="q_normalized", q_min=0.30, q_max=1.50, q_bins=49,
    ).to(device)
    models = dict(
        (n, build_model(CELLS[n]["config"],
                        CELLS[n]["workdir"] / "checkpoints" / "best.ckpt", device)[0])
        for n in CELL_ORDER)
    starts = [s for s in range(TEST_START, TEST_STOP, PRE + AFT)
              if s + PRE + AFT <= TEST_STOP]

    collected = {}
    for case in manifest["cases"]:
        key = case["case_key"]
        condition, drift, mode_n0 = condition_vector(case, manifest)
        model_frames, _, _, _ = load_case_frames(
            case, low, high, TEST_START, TEST_START + len(starts) * (PRE + AFT))
        inputs = np.empty((len(starts), PRE, 5, MODEL_H, MODEL_W), dtype=np.float32)
        truth = np.empty((len(starts), AFT, 3, MODEL_H, MODEL_W), dtype=np.float32)
        for index, s in enumerate(starts):
            i0 = s - TEST_START
            inputs[index, :, :3] = model_frames[i0: i0 + PRE]
            inputs[index, :, 3:] = condition[None, :, None, None]
            truth[index] = model_frames[i0 + PRE: i0 + PRE + AFT]
        true_c = band_coeff(loss_module, truth, device)
        entry = {"n0": mode_n0, "role": case["role"],
                 "q_window_n": [0.30 * mode_n0, 1.50 * mode_n0], "cells": {}}
        for name in CELL_ORDER:
            pred_c = band_coeff(loss_module, predict_direct10(
                models[name], inputs, device), device)
            entry["cells"][name] = dict(
                (band, stats(pred_c[:, :, 1], true_c[:, :, 1], lo, hi))
                for band, (lo, hi) in BANDS.items())
        collected[key] = entry
        del model_frames, inputs, truth, true_c
        print("[done] " + key, flush=True)

    order = [k for k in collected if collected[k]["role"] == "source"] + \
            [k for k in collected if collected[k]["role"] != "source"]

    for band, (lo, hi) in BANDS.items():
        print()
        print("=" * 116)
        print("BAND %s : predicted / true power ratio  (1.0 = calibrated, >1 = spurious power)"
              % band)
        print("=" * 116)
        print("%-11s %-7s %8s %10s %12s  " % (
            "case", "role", "n0", "in q win?", "true power")
            + " ".join("%9s" % c for c in CELL_ORDER))
        print("-" * 116)
        for key in order:
            e = collected[key]
            win = e["q_window_n"]
            inside = "yes" if (win[0] <= hi and win[1] >= lo) else "NO"
            overlap = max(0.0, min(win[1], hi) - max(win[0], lo)) / (hi - lo)
            tp = e["cells"]["U-D"][band]["true_power"]
            print("%-11s %-7s %8.2f %6s(%3.0f%%) %12.3e  " % (
                key, e["role"][:6], e["n0"], inside, 100 * overlap, tp)
                + " ".join("%9.2f" % e["cells"][c][band]["power_ratio"]
                           for c in CELL_ORDER))

    print()
    print("=" * 116)
    print("PHASE-ONLY unexplained power (predicted amplitude rescaled to truth per mode)")
    print("=" * 116)
    for band in BANDS:
        print()
        print("--- %s" % band)
        print("%-11s %-7s " % ("case", "role")
              + " ".join("%9s" % c for c in CELL_ORDER))
        for key in order:
            e = collected[key]
            print("%-11s %-7s " % (key, e["role"][:6])
                  + " ".join("%9.3f" % e["cells"][c][band]["phase_only"]
                             for c in CELL_ORDER))
        for scope in ("source", "axis_holdout"):
            subset = [k for k in order if collected[k]["role"] == scope]
            print("%-11s %-7s " % ("MEDIAN", scope[:6])
                  + " ".join("%9.3f" % np.median(
                      [collected[k]["cells"][c][band]["phase_only"] for k in subset])
                      for c in CELL_ORDER))
    print()
    print("phase_only = 2(1 - cos(phase error)) averaged with true-power weights;")
    print("0 = phase perfect, 2 = phase uncorrelated, 4 = anti-phase.")

    out = (MANIFEST.parent.parent / "radaz_conditioned_factorial_evaluation"
           / "spectral_band_power_audit.json")
    out.write_text(json.dumps(collected, indent=2), encoding="utf-8")
    print("[written] " + str(out))


if __name__ == "__main__":
    main()
