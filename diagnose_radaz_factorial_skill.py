"""Self-normalized skill scores for the RadAz conditioned 2x2 factorial.

The Section 15.5 gates are ratios to the copy baseline.  For the *fields*
that is a fair contest, but for the q-mode phase and modal-transport
observables copy is nearly exact by construction: shifting both n_e and E_y
by the same 150 ns lag leaves their relative phase, and therefore
Gamma_q ~ -Re(dn_e dE_y*)/B, almost unchanged.  A ratio to copy therefore
cannot separate "no transport skill" from "slightly worse than an almost
perfect baseline".

This script reports instead, over the same pre-registered direct10 windows:

  mode_unexplained  = complex_mode_loss / <|true_q|^2>
                      1.0 = no better than predicting zero fluctuation,
                      0.0 = perfect complex coefficients.
  amp_power_ratio   = sum|pred_q|^2 / sum|true_q|^2   (1.0 = calibrated)
  cross_phase_rad   = power-weighted |phase(pred) - phase(true)| in radians
  transport_nrmse   = rms(Gamma_pred - Gamma_true) / rms(Gamma_true)
                      1.0 = no better than predicting zero transport.
  gamma_ecdi_ratio  = predicted / true mean Gamma over q in [0.85, 1.15]

Copy and a zero-fluctuation control are scored on the identical measures.
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
    AFT, CELLS, ELECTRON_CHARGE, ELECTRON_MASS, MANIFEST, MODEL_H, MODEL_W,
    PRE, TEST_START, TEST_STOP, VALID_W,
    build_model, condition_vector, load_case_frames, predict_direct10,
)
from openstl.methods.pepapic_spectral_loss import PEPAPICSpectralLoss

CELL_ORDER = ["U-D", "C-D", "U-P", "C-P"]


def skill(loss_module, pred_norm, truth_norm, mode_n0, drift, device):
    pred = torch.from_numpy(pred_norm).to(device)
    truth = torch.from_numpy(truth_norm).to(device)
    batch = pred.shape[0]
    n0 = torch.full((batch,), float(mode_n0), dtype=torch.float32, device=device)
    pred_q, mask = loss_module._interpolate_to_q(
        loss_module._band_coefficients(pred, physical_units=False), n0)
    true_q, _ = loss_module._interpolate_to_q(
        loss_module._band_coefficients(truth, physical_units=False), n0)
    valid = mask[:, :, 0, :, :, 0].to(torch.float64)
    total = torch.clamp(valid.sum(), min=1.0)

    def parts(values):
        return values[..., 0].to(torch.float64), values[..., 1].to(torch.float64)

    # field axis inside _band_coefficients: 0=phi, 1=electron_den, 2=Ey
    pr, pi = parts(pred_q[:, :, 1])
    tr, ti = parts(true_q[:, :, 1])
    epr, epi = parts(pred_q[:, :, 2])
    etr, eti = parts(true_q[:, :, 2])

    # complex_mode_loss averages over the 3 fields and the 2 real components
    fields_pred = [parts(pred_q[:, :, f]) for f in range(3)]
    fields_true = [parts(true_q[:, :, f]) for f in range(3)]
    err = sum(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
              for a, b in zip(fields_pred, fields_true))
    power = sum((b[0] ** 2 + b[1] ** 2) for b in fields_true)
    mode_unexplained = float((err * valid).sum() /
                             torch.clamp((power * valid).sum(), min=1e-300))

    pred_power = float(((pr ** 2 + pi ** 2) * valid).sum())
    true_power = float(((tr ** 2 + ti ** 2) * valid).sum())
    amp_power_ratio = pred_power / true_power if true_power > 0 else float("nan")

    cross_pred_r = pr * epr + pi * epi
    cross_pred_i = pi * epr - pr * epi
    cross_true_r = tr * etr + ti * eti
    cross_true_i = ti * etr - tr * eti
    delta = torch.atan2(
        torch.sin(torch.atan2(cross_pred_i, cross_pred_r)
                  - torch.atan2(cross_true_i, cross_true_r)),
        torch.cos(torch.atan2(cross_pred_i, cross_pred_r)
                  - torch.atan2(cross_true_i, cross_true_r)))
    weight = torch.sqrt(cross_true_r ** 2 + cross_true_i ** 2) * valid
    cross_phase_rad = float((torch.abs(delta) * weight).sum()
                            / torch.clamp(weight.sum(), min=1e-300))

    magnetic_t = mode_n0 * (2.0 * np.pi * ELECTRON_MASS) * drift \
        / (ELECTRON_CHARGE * 1.28e-2)
    relative_b = magnetic_t / 0.020
    gamma_pred = -cross_pred_r / relative_b
    gamma_true = -cross_true_r / relative_b
    num = float((((gamma_pred - gamma_true) ** 2) * valid).sum() / total)
    den = float(((gamma_true ** 2) * valid).sum() / total)
    transport_nrmse = float(np.sqrt(num / den)) if den > 0 else float("nan")

    q_grid = loss_module.q_grid.to(torch.float64)
    ecdi = ((q_grid >= 0.85) & (q_grid <= 1.15)).to(torch.float64)
    ecdi_w = valid * ecdi[None, None, None, :]
    ecdi_total = torch.clamp(ecdi_w.sum(), min=1.0)
    g_pred = float((gamma_pred * ecdi_w).sum() / ecdi_total)
    g_true = float((gamma_true * ecdi_w).sum() / ecdi_total)

    return {
        "mode_unexplained": mode_unexplained,
        "amp_power_ratio": amp_power_ratio,
        "cross_phase_rad": cross_phase_rad,
        "transport_nrmse": transport_nrmse,
        "gamma_ecdi_ratio": g_pred / g_true if g_true != 0 else float("nan"),
        "gamma_ecdi_true": g_true,
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
        (name, build_model(CELLS[name]["config"],
                           CELLS[name]["workdir"] / "checkpoints" / "best.ckpt",
                           device)[0])
        for name in CELL_ORDER)
    starts = [s for s in range(TEST_START, TEST_STOP, PRE + AFT)
              if s + PRE + AFT <= TEST_STOP]

    collected = {}
    for case in manifest["cases"]:
        key = case["case_key"]
        condition, drift, mode_n0 = condition_vector(case, manifest)
        model_frames, _, _, _ = load_case_frames(
            case, low, high, TEST_START, TEST_START + len(starts) * (PRE + AFT))
        inputs = np.empty((len(starts), PRE, 5, MODEL_H, MODEL_W), dtype=np.float32)
        truth_norm = np.empty((len(starts), AFT, 3, MODEL_H, MODEL_W), dtype=np.float32)
        for index, s in enumerate(starts):
            i0 = s - TEST_START
            inputs[index, :, :3] = model_frames[i0: i0 + PRE]
            inputs[index, :, 3:] = condition[None, :, None, None]
            truth_norm[index] = model_frames[i0 + PRE: i0 + PRE + AFT]
        copy_norm = np.repeat(inputs[:, -1:, :3], AFT, axis=1)
        zero_norm = np.repeat(
            truth_norm[..., :VALID_W].mean(axis=-1, keepdims=True),
            MODEL_W, axis=-1).astype(np.float32)

        entry = {"role": case["role"], "n0": mode_n0}
        entry["copy"] = skill(loss_module, copy_norm, truth_norm, mode_n0, drift, device)
        entry["zero"] = skill(loss_module, zero_norm, truth_norm, mode_n0, drift, device)
        for name in CELL_ORDER:
            pred = predict_direct10(models[name], inputs, device)
            entry[name] = skill(loss_module, pred, truth_norm, mode_n0, drift, device)
        collected[key] = entry
        del model_frames, inputs, truth_norm
        print("[done] " + key, flush=True)

    order = [k for k in collected if collected[k]["role"] == "source"] + \
            [k for k in collected if collected[k]["role"] != "source"]
    names = ["copy", "zero"] + CELL_ORDER
    titles = {
        "mode_unexplained": "q-COMPLEX-MODE unexplained power  (1.0 = no skill, 0 = perfect)",
        "amp_power_ratio": "q-BAND AMPLITUDE power ratio pred/true  (1.0 = calibrated)",
        "cross_phase_rad": "ne-Ey CROSS-PHASE error [rad]  (pi/2 = transport sign lost)",
        "transport_nrmse": "MODAL TRANSPORT nrmse  (1.0 = no better than Gamma=0)",
        "gamma_ecdi_ratio": "ECDI-band Gamma  pred/true  (1.0 = correct magnitude and sign)",
    }
    for metric, title in titles.items():
        print()
        print("=" * 104)
        print(title)
        print("=" * 104)
        print("%-11s %-7s %-8s " % ("case", "role", "n0")
              + " ".join("%9s" % n for n in names))
        print("-" * 104)
        for key in order:
            entry = collected[key]
            print("%-11s %-7s %-8.2f " % (key, entry["role"][:6], entry["n0"])
                  + " ".join("%9.3f" % entry[n][metric] for n in names))
        for scope in ("source", "axis_holdout"):
            subset = [k for k in order if collected[k]["role"] == scope]
            values = []
            for n in names:
                vals = [collected[k][n][metric] for k in subset]
                vals = [v for v in vals if np.isfinite(v)]
                values.append(float(np.median(vals)) if vals else float("nan"))
            print("%-11s %-7s %-8s " % ("MEDIAN", scope[:6], "")
                  + " ".join("%9.3f" % v for v in values))

    out = Path(CELLS["U-D"]["workdir"]).parent / \
        "radaz_conditioned_factorial_evaluation" / "factorial_skill_scores.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(collected, indent=2), encoding="utf-8")
    print()
    print("[written] " + str(out))


if __name__ == "__main__":
    main()
