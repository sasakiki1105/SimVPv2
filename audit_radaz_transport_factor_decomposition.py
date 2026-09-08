"""Audit 4: which factor of the modal transport is actually wrong?

With the ensemble cross spectrum

    <N E*> = A_N A_E C ,   A_N = sqrt<|N|^2>,  A_E = sqrt<|E|^2>,
    C = r exp(i delta)  (r = coherence, delta = ne-Ey cross-phase)

the modal transport is

    Gamma = -A_N A_E r cos(delta) / B .

Three factors: the amplitude product A_N A_E, the coherence r, and the phase
delta.  Substituting the predicted value of each factor into the true Gamma,
one at a time (2^3 = 8 counterfactuals), attributes the transport error to a
specific factor instead of inferring it from the 0.18 rad phase error.

Averages are taken over the 10 direct10 windows x 10 frames, per radial band
and q bin, on the same pre-registered test window as the factorial evaluation.
"""

from __future__ import annotations

import itertools
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
from openstl.methods.pepapic_spectral_loss import PEPAPICSpectralLoss

CELL_ORDER = ["U-D", "C-D", "U-P", "C-P"]


def factors(loss_module, normalized, mode_n0, device):
    """-> (A_N A_E, r, delta, valid) each [R, Q], ensemble-averaged."""
    tensor = torch.from_numpy(normalized).to(device)
    n0 = torch.full((tensor.shape[0],), float(mode_n0),
                    dtype=torch.float32, device=device)
    coeff, mask = loss_module._interpolate_to_q(
        loss_module._band_coefficients(tensor, physical_units=False), n0)
    values = coeff.detach().cpu().numpy()
    complexed = values[..., 0] + 1j * values[..., 1]     # [B,T,F,R,Q]
    # _interpolate_to_q broadcasts its mask as [B,1,1,1,Q,1]; it depends on q
    # only, so take one row and broadcast it back over the radial bands.
    q_valid = mask[0, 0, 0, 0, :, 0].detach().cpu().numpy().astype(bool)
    valid = np.broadcast_to(q_valid[None, :],
                            (complexed.shape[3], q_valid.shape[0]))
    ne = complexed[:, :, 1]
    ey = complexed[:, :, 2]
    pn = np.mean(np.abs(ne) ** 2, axis=(0, 1))
    pe = np.mean(np.abs(ey) ** 2, axis=(0, 1))
    cross = np.mean(ne * np.conj(ey), axis=(0, 1))
    amplitude = np.sqrt(pn * pe)
    coherence = np.abs(cross) / np.maximum(amplitude, 1e-300)
    delta = np.angle(cross)
    return amplitude, coherence, delta, valid


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

        ta, tr, td, valid = factors(loss_module, truth, mode_n0, device)
        gamma_true = -ta * tr * np.cos(td)
        denominator = np.sqrt(np.mean(gamma_true[valid] ** 2))

        entry = {"role": case["role"], "n0": mode_n0, "cells": {}}
        for name in CELL_ORDER:
            pa, pr, pd, _ = factors(
                loss_module, predict_direct10(models[name], inputs, device),
                mode_n0, device)
            rows = {}
            for use_a, use_r, use_d in itertools.product((0, 1), repeat=3):
                a = pa if use_a else ta
                r = pr if use_r else tr
                d = pd if use_d else td
                gamma = -a * r * np.cos(d)
                label = "".join(c if u else "." for c, u in
                                zip("ArD", (use_a, use_r, use_d)))
                rows[label] = float(
                    np.sqrt(np.mean((gamma - gamma_true)[valid] ** 2)) / denominator)
            rows["_amp_ratio"] = float(np.mean(pa[valid]) / np.mean(ta[valid]))
            rows["_coh_pred"] = float(np.mean(pr[valid]))
            rows["_coh_true"] = float(np.mean(tr[valid]))
            rows["_dphase"] = float(np.mean(np.abs(np.angle(
                np.exp(1j * (pd - td))))[valid]))
            entry["cells"][name] = rows
        collected[key] = entry
        del model_frames, inputs, truth
        print("[done] " + key, flush=True)

    order = [k for k in collected if collected[k]["role"] == "source"] + \
            [k for k in collected if collected[k]["role"] != "source"]
    labels = ["...", "A..", ".r.", "..D", "Ar.", "A.D", ".rD", "ArD"]
    names = {"...": "none (sanity=0)", "A..": "amplitude only",
             ".r.": "coherence only", "..D": "phase only",
             "Ar.": "amp+coh", "A.D": "amp+phase", ".rD": "coh+phase",
             "ArD": "all (= model)"}

    for cell in CELL_ORDER:
        print()
        print("=" * 104)
        print("TRANSPORT ERROR ATTRIBUTION -- %s   (nrmse of Gamma; 'all' = the model)"
              % cell)
        print("=" * 104)
        print("%-11s %-7s " % ("case", "role")
              + " ".join("%8s" % l for l in labels))
        print("-" * 104)
        for key in order:
            e = collected[key]["cells"][cell]
            print("%-11s %-7s " % (key, collected[key]["role"][:6])
                  + " ".join("%8.3f" % e[l] for l in labels))
        for scope in ("source", "axis_holdout"):
            subset = [k for k in order if collected[k]["role"] == scope]
            print("%-11s %-7s " % ("MEDIAN", scope[:6])
                  + " ".join("%8.3f" % np.median(
                      [collected[k]["cells"][cell][l] for k in subset])
                      for l in labels))

    print()
    print("column key: letters mark which factor is taken from the PREDICTION")
    for l in labels:
        print("   %-5s %s" % (l, names[l]))

    print()
    print("=" * 104)
    print("FACTOR DIAGNOSTICS (median over cases)")
    print("=" * 104)
    print("%-9s %-7s %14s %12s %12s %14s" % (
        "cell", "scope", "amp pred/true", "coh pred", "coh true", "|dphase| rad"))
    for cell in CELL_ORDER:
        for scope in ("source", "axis_holdout"):
            subset = [k for k in order if collected[k]["role"] == scope]
            g = lambda f: np.median([collected[k]["cells"][cell][f] for k in subset])
            print("%-9s %-7s %14.3f %12.3f %12.3f %14.3f" % (
                cell, scope[:6], g("_amp_ratio"), g("_coh_pred"),
                g("_coh_true"), g("_dphase")))

    out = (MANIFEST.parent.parent / "radaz_conditioned_factorial_evaluation"
           / "transport_factor_decomposition.json")
    out.write_text(json.dumps(collected, indent=2), encoding="utf-8")
    print()
    print("[written] " + str(out))


if __name__ == "__main__":
    main()
