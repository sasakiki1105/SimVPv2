"""Set lambda_P and lambda_C for physics loss v2 by gradient audit.

Matches the v1 protocol: choose fixed weights so that each auxiliary loss
contributes a few percent of the data loss's gradient on the model output,
measured across all six source conditions before any training starts.  The raw
loss magnitudes differ by orders of magnitude between the log-power term and
the normalised cross-spectrum term, so a weight is only interpretable through
this ratio.

Also reports what fraction of azimuthal modes the cross-spectrum mask keeps
per condition, since a mask that keeps almost nothing would make the term
inert and a mask that keeps everything would put loss on noise phases.
"""

from __future__ import annotations

import json
import os
import runpy
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import torch

from evaluate_radaz_conditioned_factorial import (
    AFT, MANIFEST, MODEL_H, MODEL_W, PRE, TEST_START,
    condition_vector, load_case_frames,
)
from openstl.methods.pepapic_spectral_loss import PEPAPICSpectralLoss

ROOT = Path(r"C:\Users\astro\research\SimVPv2")
CONFIG = ROOT / "configs/custom/pepapic/SimVP_gSTA_radaz_factorial_UPv2_60ep.py"
OUT = ROOT / "workdirs/2D_RadAz/radaz_physics_loss_v2_manifests"
TARGET = 0.05          # aim: each auxiliary term ~5% of the data gradient


def replay_legacy():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = runpy.run_path(str(CONFIG))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    low = np.asarray(manifest["normalization"]["low"], dtype=np.float64)
    high = np.asarray(manifest["normalization"]["high"], dtype=np.float64)
    device = torch.device("cuda:0")

    loss_module = PEPAPICSpectralLoss(
        data_root=str(MANIFEST),
        max_mode=int(cfg["pepapic_spectral_max_mode"]),
        radial_bands=int(cfg["pepapic_spectral_radial_bands"]),
        radial_min_m=float(cfg["pepapic_spectral_radial_min_m"]),
        radial_max_m=float(cfg["pepapic_spectral_radial_max_m"]),
        coordinate_system=str(cfg["pepapic_spectral_coordinate_system"]),
        power_eps_relative=float(cfg["pepapic_spectral_power_eps_relative"]),
        cross_mask_kappa=float(cfg["pepapic_spectral_cross_mask_kappa"]),
    ).to(device)

    rows = []
    print("=" * 104)
    print("GRADIENT AUDIT for physics loss v2 (raw losses and output-gradient norms)")
    print("=" * 104)
    print("%-11s %8s %12s %12s %12s %12s %12s %10s" % (
        "case", "n0", "L_data", "L_power", "L_cross", "|g_P|/|g_D|",
        "|g_C|/|g_D|", "mask kept"))
    print("-" * 104)

    for case in manifest["cases"]:
        if case["role"] != "source":
            continue
        condition, drift, mode_n0 = condition_vector(case, manifest)
        frames = load_case_frames(
            case, low, high, TEST_START, TEST_START + PRE + AFT)[0]
        truth = torch.from_numpy(frames[PRE: PRE + AFT][None]).to(device)

        # A perturbed copy of the truth stands in for an untrained prediction,
        # so the audit does not depend on any particular checkpoint.
        generator = torch.Generator(device="cpu").manual_seed(0)
        noise = torch.randn(truth.shape, generator=generator).to(device)
        prediction = (truth + 0.05 * noise).detach().requires_grad_(True)

        data_loss = torch.mean((prediction - truth) ** 2)
        spectral = loss_module(prediction, truth)
        grads = {}
        for label, value in (("data", data_loss),
                             ("power", spectral["power"]),
                             ("cross", spectral["cross_spectrum"])):
            g = torch.autograd.grad(value, prediction, retain_graph=True)[0]
            grads[label] = float(torch.linalg.norm(g))

        # mask retention, from the true spectra
        coeff = loss_module._band_coefficients(truth, physical_units=False)
        pn, pe, _, _ = loss_module._ensemble_spectra(coeff)
        product = torch.sqrt(pn * pe)
        kept = float(((product > loss_module.cross_mask_kappa
                       * torch.amax(product, dim=-1, keepdim=True))
                      ).to(torch.float64).mean())

        rows.append({
            "case": case["case_key"], "n0": mode_n0,
            "L_data": float(data_loss), "L_power": float(spectral["power"]),
            "L_cross": float(spectral["cross_spectrum"]),
            "g_power_over_g_data": grads["power"] / max(grads["data"], 1e-300),
            "g_cross_over_g_data": grads["cross"] / max(grads["data"], 1e-300),
            "mask_fraction_kept": kept,
        })
        print("%-11s %8.2f %12.4e %12.4e %12.4e %12.4e %12.4e %10.3f" % (
            case["case_key"], mode_n0, rows[-1]["L_data"], rows[-1]["L_power"],
            rows[-1]["L_cross"], rows[-1]["g_power_over_g_data"],
            rows[-1]["g_cross_over_g_data"], kept))
        del frames, truth, prediction

    ratio_p = np.median([r["g_power_over_g_data"] for r in rows])
    ratio_c = np.median([r["g_cross_over_g_data"] for r in rows])
    lambda_p = TARGET / ratio_p
    lambda_c = TARGET / ratio_c

    def round_sig(x, digits=2):
        if x == 0:
            return 0.0
        return float(np.round(x, -int(np.floor(np.log10(abs(x)))) + digits - 1))

    lambda_p = round_sig(lambda_p)
    lambda_c = round_sig(lambda_c)

    print()
    print("median |g_power|/|g_data| = %.4e  -> lambda_P = %g for a %.0f%% target"
          % (ratio_p, lambda_p, 100 * TARGET))
    print("median |g_cross|/|g_data| = %.4e  -> lambda_C = %g for a %.0f%% target"
          % (ratio_c, lambda_c, 100 * TARGET))
    print()
    print("resulting per-case auxiliary gradient share:")
    for r in rows:
        print("   %-11s power %5.1f%%   cross %5.1f%%" % (
            r["case"], 100 * lambda_p * r["g_power_over_g_data"],
            100 * lambda_c * r["g_cross_over_g_data"]))

    payload = {
        "config": str(CONFIG),
        "target_gradient_share": TARGET,
        "lambda_power": lambda_p,
        "lambda_crossspec": lambda_c,
        "power_eps_relative": loss_module.power_eps_relative,
        "cross_mask_kappa": loss_module.cross_mask_kappa,
        "transport_lambda": 0.0,
        "note": "transport loss deliberately off; Gamma is derived from power "
                "and cross spectrum, and whether it improves constructively is "
                "the mechanism test",
        "cases": rows,
    }
    payload["status"] = "legacy_replay_uses_source_test_targets_not_train_only"
    payload["data_frames"] = [TEST_START, TEST_START + PRE + AFT - 1]
    (OUT / "gradient_audit_v2_legacy_replay.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print()
    print("[written] " + str(OUT / "gradient_audit_v2_legacy_replay.json"))


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--replay-legacy"]:
        replay_legacy()
    else:
        raise SystemExit("Old calibration uses source-test targets. Run prepare_radaz_v3.py for source-TRAIN calibration, or --replay-legacy to reproduce history.")
