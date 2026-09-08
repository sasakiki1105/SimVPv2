"""Is the transport failure a catastrophic-cancellation problem?

Gamma_q ~ -Re(dn_e dE_y*) / B = -|dn_e||dE_y| cos(theta) / B, where theta is
the ne-Ey cross-phase.  If the true theta sits near +-pi/2 (quadrature), then
|cos theta| is small: the transport is a small residual of a nearly cancelling
product, and a phase error eps produces a relative transport error of roughly
eps / |cot(theta)| -- i.e. O(1) as soon as eps approaches the departure from
quadrature.  If instead theta is broadly distributed away from quadrature, the
transport failure has some other cause and reparametrising the phase will not
help.

Reports, per case, power-weighted over the same q grid and windows as the
factorial evaluation:

  |cos theta|_true      transport efficiency; small = near quadrature
  delta = |pi/2-|theta||   departure from quadrature [rad]
  model phase error     from diagnose_radaz_factorial_skill.py
  eps / delta           predicted relative transport error

and the same cancellation ratio for charge:

  rho_rms / (e * ne_rms)   how much of the density is net charge
  required density accuracy for 10% accurate rho
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import torch

from evaluate_radaz_conditioned_factorial import (
    AFT, ELECTRON_CHARGE, MANIFEST, MODEL_H, MODEL_W, PRE,
    TEST_START, TEST_STOP, VALID_H, VALID_W,
    condition_vector, load_case_frames,
)
from openstl.methods.pepapic_spectral_loss import PEPAPICSpectralLoss


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

    skill_path = (MANIFEST.parent.parent
                  / "radaz_conditioned_factorial_evaluation"
                  / "factorial_skill_scores.json")
    skill = json.loads(skill_path.read_text(encoding="utf-8"))

    starts = [s for s in range(TEST_START, TEST_STOP, PRE + AFT)
              if s + PRE + AFT <= TEST_STOP]

    print("=" * 112)
    print("TRANSPORT: is Gamma a small residual of a near-cancelling product?")
    print("=" * 112)
    print("%-11s %-7s %10s %10s %10s %10s %10s" % (
        "case", "role", "|cos0|", "delta[rad]", "eps_UP", "eps_CP", "eps/delta"))
    print("-" * 112)

    rows = []
    charge_rows = []
    for case in manifest["cases"]:
        key = case["case_key"]
        condition, drift, mode_n0 = condition_vector(case, manifest)
        model_frames, physical, _, _ = load_case_frames(
            case, low, high, TEST_START, TEST_START + len(starts) * (PRE + AFT))
        truth = np.empty((len(starts), AFT, 3, MODEL_H, MODEL_W), dtype=np.float32)
        for index, s in enumerate(starts):
            i0 = s - TEST_START
            truth[index] = model_frames[i0 + PRE: i0 + PRE + AFT]

        tensor = torch.from_numpy(truth).to(device)
        n0 = torch.full((len(starts),), float(mode_n0), dtype=torch.float32,
                        device=device)
        coeff, mask = loss_module._interpolate_to_q(
            loss_module._band_coefficients(tensor, physical_units=False), n0)
        valid = mask[:, :, 0, :, :, 0].to(torch.float64)

        nr = coeff[:, :, 1, ..., 0].to(torch.float64)
        ni = coeff[:, :, 1, ..., 1].to(torch.float64)
        er = coeff[:, :, 2, ..., 0].to(torch.float64)
        ei = coeff[:, :, 2, ..., 1].to(torch.float64)
        cross_r = nr * er + ni * ei
        cross_i = ni * er - nr * ei
        magnitude = torch.sqrt(cross_r ** 2 + cross_i ** 2)
        theta = torch.atan2(cross_i, cross_r)
        cos_abs = torch.abs(cross_r) / torch.clamp(magnitude, min=1e-300)
        delta = torch.abs(np.pi / 2.0 - torch.abs(theta))

        weight = magnitude * valid
        total = torch.clamp(weight.sum(), min=1e-300)
        cos_w = float((cos_abs * weight).sum() / total)
        delta_w = float((delta * weight).sum() / total)

        eps_up = skill[key]["U-P"]["cross_phase_rad"]
        eps_cp = skill[key]["C-P"]["cross_phase_rad"]
        rows.append((key, case["role"], cos_w, delta_w, eps_up, eps_cp))
        print("%-11s %-7s %10.4f %10.4f %10.4f %10.4f %10.2f" % (
            key, case["role"][:6], cos_w, delta_w, eps_up, eps_cp,
            eps_up / delta_w if delta_w > 0 else float("nan")))

        # charge cancellation, in physical units on the valid core
        ne = physical[:, 0].astype(np.float64)
        nion = physical[:, 1].astype(np.float64)
        rho = ELECTRON_CHARGE * (nion - ne)
        charge_rows.append((
            key,
            float(np.sqrt(np.mean(rho ** 2))),
            float(ELECTRON_CHARGE * np.sqrt(np.mean(ne ** 2))),
            float(np.sqrt(np.mean((nion - ne) ** 2)) / np.sqrt(np.mean(ne ** 2))),
        ))
        del model_frames, physical, truth

    print()
    print("|cos0| = power-weighted |cos(cross-phase)|; 1 = all of the product")
    print("         becomes transport, 0 = perfect quadrature and zero transport.")
    print("delta  = power-weighted departure from quadrature.  eps = model phase error.")
    print("eps/delta >~ 1 means the model's phase error alone destroys Gamma.")

    print()
    print("=" * 112)
    print("CHARGE: how much of the density survives the n_i - n_e cancellation?")
    print("=" * 112)
    print("%-11s %14s %14s %12s %14s" % (
        "case", "rho_rms[C/m3]", "e*ne_rms", "ratio", "need for 10%"))
    print("-" * 112)
    for key, rho_rms, ene_rms, ratio in charge_rows:
        print("%-11s %14.4g %14.4g %12.5f %13.3f%%" % (
            key, rho_rms, ene_rms, ratio, 100.0 * 0.10 * ratio))
    print()
    print("ratio = rms(n_i - n_e) / rms(n_e).  'need for 10%' is the relative")
    print("accuracy each density channel needs for 10%-accurate rho, if their")
    print("errors are independent.")


if __name__ == "__main__":
    main()
