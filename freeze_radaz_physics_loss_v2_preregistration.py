"""Freeze the physics-loss v2 evaluation protocol before any v2 test result.

Everything the evaluation needs that is not already fixed by the config is
derived here from TRAINING frames of the SOURCE conditions only (frames
1200-1599, inside the training split 0-1599), and written to a registration
JSON with a SHA256 over the frozen content.  The evaluation script reads that
file and is not permitted to recompute any of it.

Frozen here:
  * phi   -- the relative power floor defining the low-power set L used by the
             high-mode hallucination statistic H
  * L(condition)          -- integer modes below that floor
  * resonance band        -- integer n with 0.30 <= n/n0 <= 1.50, purely a
                             priori from (E, B), no data involved
  * every gate threshold and the verdict rules

Runs on CPU so it does not contend with the v2 training job.
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

from evaluate_radaz_conditioned_factorial import (
    MANIFEST, condition_vector, load_case_frames,
)
from openstl.methods.pepapic_spectral_loss import PEPAPICSpectralLoss

OUT = Path(r"C:\Users\astro\research\SimVPv2\workdirs\2D_RadAz\radaz_physics_loss_v2_manifests")
TRAIN_LO, TRAIN_HI = 1200, 1600      # inside the training split (0-1599)
PHI_CANDIDATES = [1e-3, 1e-4, 1e-5, 1e-6]
MAX_MODE = 64


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    low = np.asarray(manifest["normalization"]["low"], dtype=np.float64)
    high = np.asarray(manifest["normalization"]["high"], dtype=np.float64)
    device = torch.device("cpu")
    loss_module = PEPAPICSpectralLoss(
        data_root=str(MANIFEST), max_mode=MAX_MODE, radial_bands=4,
        radial_min_m=0.09e-2, radial_max_m=1.19e-2,
        coordinate_system="integer_power_cross",
    ).to(device)

    print("=" * 100)
    print("TRAINING-WINDOW SPECTRA  (source conditions, frames %d-%d)"
          % (TRAIN_LO, TRAIN_HI - 1))
    print("=" * 100)
    print("%-11s %8s %12s %12s %12s   %s" % (
        "case", "n0", "peak P_N", "min/peak", "n at min",
        "  ".join("|L| @phi=%g" % p for p in PHI_CANDIDATES)))
    print("-" * 100)

    spectra = {}
    for case in manifest["cases"]:
        if case["role"] != "source":
            continue
        _, _, mode_n0 = condition_vector(case, manifest)
        frames = load_case_frames(case, low, high, TRAIN_LO, TRAIN_HI)[0]
        tensor = torch.from_numpy(frames[None]).to(device)
        coeff = loss_module._band_coefficients(tensor, physical_units=False)
        pn, pe, _, _ = loss_module._ensemble_spectra(coeff)
        # average the radial bands to a single spectrum per condition
        power_n = pn.mean(dim=1).squeeze(0).numpy()
        power_e = pe.mean(dim=1).squeeze(0).numpy()
        geometric = np.sqrt(power_n * power_e)
        relative = geometric / geometric.max()
        counts = [int(np.sum(relative < p)) for p in PHI_CANDIDATES]
        spectra[case["case_key"]] = {
            "n0": mode_n0,
            "relative_geometric_power": relative.tolist(),
            "peak_power_N": float(power_n.max()),
        }
        print("%-11s %8.2f %12.4e %12.4e %12d   %s" % (
            case["case_key"], mode_n0, power_n.max(), relative.min(),
            int(np.argmin(relative)) + 1,
            "  ".join("%11d" % c for c in counts)))
        del frames, tensor, coeff

    # A power-threshold definition of L turns out to be degenerate: choosing
    # phi small enough never to mark a resonant mode leaves L empty for the
    # high-n0 conditions, where the resonance band covers most of the spectrum
    # (E10_B30 has n0 = 32.25, band n = 10-48).  L is therefore defined
    # geometrically instead, as the modes ABOVE the resonance band:
    #
    #     L(condition) = { n <= 64 : n > 1.5 * n0 }
    #
    # This is fully a priori from (E, B), needs no data at all, is never empty
    # (1.5 * 32.25 = 48.4 < 64), and is exactly the region where the v1 audit
    # measured 2-830x hallucinated power.  The training spectra above are kept
    # in the registration as context showing where the roll-off actually sits.
    print()
    print("L is defined geometrically as n > 1.5*n0, not by a power threshold:")
    print("a threshold small enough to spare the resonance band leaves L empty")
    print("for the high-n0 conditions.  Training-spectrum roll-off is recorded")
    print("as context only.")

    # frozen per-condition mode sets, for every condition including holdouts
    conditions = {}
    for case in manifest["cases"]:
        _, drift, mode_n0 = condition_vector(case, manifest)
        modes = np.arange(1, MAX_MODE + 1)
        resonance = modes[(modes >= 0.30 * mode_n0) & (modes <= 1.50 * mode_n0)]
        above = modes[modes > 1.50 * mode_n0]
        conditions[case["case_key"]] = {
            "role": case["role"], "B_mT": case["B_mT"], "Ez_kVm": case["Ez_kVm"],
            "n0": mode_n0,
            "resonance_band_modes": resonance.tolist(),
            "resonance_band_size": int(len(resonance)),
            "low_power_set_L_modes": above.tolist(),
            "low_power_set_L_size": int(len(above)),
        }

    print()
    print("=" * 100)
    print("FROZEN PER-CONDITION MODE SETS")
    print("=" * 100)
    print("%-11s %-13s %8s %10s %-12s %6s %s" % (
        "case", "role", "n0", "|resonance|", "resonance", "|L|", "L (n > 1.5 n0)"))
    for key, entry in conditions.items():
        band = entry["resonance_band_modes"]
        lset = entry["low_power_set_L_modes"]
        print("%-11s %-13s %8.2f %10d %-12s %6d %s" % (
            key, entry["role"], entry["n0"], entry["resonance_band_size"],
            "%d-%d" % (band[0], band[-1]) if band else "(empty)",
            entry["low_power_set_L_size"],
            "%d-%d" % (lset[0], lset[-1]) if lset else "(empty)"))

    registration = {
        "registered_utc": "2026-09-04",
        "status": "frozen before any physics-loss v2 test result was computed",
        "training_state_at_registration":
            "v2 run in progress; only training/validation loss observed",
        "comparators": {
            "primary": "U-P (v1 q-complex + transport loss)",
            "secondary": "U-D (no physics loss)",
            "note": "v2 differs from U-P in the physics-loss definition only; "
                    "data, seed, epochs, schedule, architecture and "
                    "normalization are identical",
        },
        "evaluation_order": [
            "in-distribution source conditions",
            "E-axis interpolation holdout: E22.5/B20, E25/B20",
            "B-axis interpolation holdout: E10/B15, E10/B25",
            "B15/E22.5 reserved as a genuine off-axis blind test, NOT evaluated here",
        ],
        "naming_note":
            "the training set is L-shaped, so both holdout groups are "
            "interpolations along an arm that was trained; they are named "
            "E-axis and B-axis interpolation rather than same-axis and "
            "orthogonal-axis, which would suggest extrapolation",
        "primary_protocol": {
            "mode": "direct10", "frame_ns": 15.0, "horizon_ns": 150.0,
            "test_frames": [1800, 1999],
            "windows": "ten non-overlapping, starts 1800..1980 step 20",
            "closed_loop_40": "secondary only, not part of any success criterion",
        },
        "spectral_gate": {
            "space": "exact integer azimuthal modes; NO q interpolation "
                     "anywhere in the evaluation",
            "bands": ["n1-6", "n9-21", "n1-32", "resonance band (integer modes "
                      "with 0.30 <= n/n0 <= 1.50)"],
            "metric": "log-power RMSE E_P with the frozen epsilon",
            "epsilon_relative_to_true_peak": 1e-8,
            "hallucination_statistic":
                "H = sum_{n in L} max(P_pred - P_true, 0) / sum_{n=1..32} P_true",
            "low_power_set_rule":
                "L(condition) = { n <= 64 : n > 1.50 * n0(E,B) }, the modes "
                "above the resonance band.  Fully a priori from the condition "
                "parameters, uses no data, and is never empty. A power-threshold "
                "definition was rejected because any threshold that spares the "
                "resonance band leaves L empty for the high-n0 conditions.",
            "low_power_set_context":
                "training-window spectra (frames %d-%d of the source "
                "conditions) are stored in this file to document where the "
                "roll-off sits; they are context, not part of the rule"
                % (TRAIN_LO, TRAIN_HI - 1),
            "pass": [
                "v2 better than v1 in at least 3 of the 4 holdout conditions",
                "median E_P over the 4 holdout conditions at least 15% lower than v1",
                "H does not get worse than v1",
            ],
        },
        "factor_gate": {
            "definitions": {
                "S_NE": "<N_n E_n*>", "C_n": "S_NE / sqrt(P_N P_E) = r exp(i delta)",
                "E_C": "<M_n |C_hat - C|^2> on the frozen training mask",
                "E_A": "|log( (A_N A_E)_pred / (A_N A_E)_true )|",
                "E_delta": "1 - cos(delta_hat - delta)",
                "E_r": "|r_hat - r|",
            },
            "cross_mask_kappa": 1e-3,
            "cross_mask_rule":
                "threshold on the geometric mean sqrt(P_N P_E) relative to its "
                "per-(band) maximum, identical to training; mask coverage "
                "reported per condition",
            "pass": [
                "holdout median E_A at least 15% better than v1",
                "holdout median E_delta at least 15% better than v1",
                "E_r no more than 10% worse than v1",
            ],
            "rationale":
                "the v1 audit attributed transport error to amplitude "
                "(holdout 0.37-0.61) and relative phase (0.46-0.66) with "
                "coherence essentially correct (0.03-0.07), so E_A and "
                "E_delta are the mechanistic targets",
        },
        "constructive_transport_gate": {
            "definition": "Gamma_n = -Re(N_n E_n*)/B on exact integer modes",
            "metric": "NRMSE_Gamma = rms(Gamma_hat - Gamma) / rms(Gamma)",
            "reference": "a zero-transport predictor scores 1.0; the copy "
                         "baseline is NOT used here because a 150 ns lag "
                         "nearly preserves the relative phase",
            "no_transport_supervision":
                "v2 training uses no direct transport loss, so any improvement "
                "is an independent downstream consequence",
            "mechanistic_pass": [
                "E-axis holdout median better than v1",
                "B-axis holdout median better than v1",
                "better than v1 in at least 3 of the 4 holdout conditions",
            ],
            "application_level_success": "median NRMSE_Gamma < 0.8, reported "
                                         "separately and never conflated with "
                                         "mechanistic success",
        },
        "field_safeguard": {
            "fields": ["electron_den", "ion_den", "phi"],
            "pass": [
                "three-field aggregate median error no more than 10% worse than v1",
                "no individual field more than 25% worse than v1",
            ],
            "note": "if this fails, spectral or transport improvement is not "
                    "described as resolving the trade-off",
        },
        "generalization_levels": {
            "A": "source reconstruction improves over v1 -- the loss repair "
                 "works, but this is not evidence of parameter generalization",
            "B": "E-axis interpolation (E22.5, E25 at B20) passes the gates",
            "C": "B-axis interpolation (B15, B25 at E10) passes the gates",
            "B_without_C":
                "the representation interpolates along a trained parameter path "
                "but the dynamical operator is not shared between the E sweep "
                "and the B sweep",
        },
        "off_axis_reservation": {
            "point": {"B_mT": 15.0, "Ez_kVm": 22.5},
            "why": "belongs to neither the B=20 E sweep nor the E=10 B sweep",
            "rules": [
                "v2 architecture, loss and hyperparameters are frozen before "
                "B15/E22.5 is looked at",
                "any run retuned using B15/E22.5 is excluded from the "
                "confirmatory test",
                "B15/E22.5 is a same-S point, so a single success there does "
                "not establish 2D E-B map generalization; at least one "
                "additional non-same-S off-axis point is required first",
            ],
        },
        "statistics": {
            "summary_unit": "condition, not frame",
            "reason": "15 ns frames are not independent samples",
            "secondary": "block bootstrap accounting for autocorrelation",
            "seeds": "single seed; no model-superiority claim from a p-value. "
                     "The purpose is loss-mechanism falsification. Replication "
                     "seeds run only after v2 passes the mechanistic gates",
        },
        "verdicts": {
            "FAIL": "integer-mode spectral error or high-mode hallucination "
                    "does not improve",
            "PARTIAL": "spectral and factor gates pass but transport does not "
                       "improve -- consider that (P_N, P_E, C_n) is insufficient "
                       "for transport closure, or that time history / state "
                       "information is missing",
            "MECHANISTIC_SUCCESS": "spectral, factor, constructive-transport "
                                   "and field-safeguard gates all pass",
            "MAP_GENERALIZATION_CANDIDATE":
                "mechanistic success plus both E-axis and B-axis levels; only "
                "then proceed to B15/E22.5 and further off-axis PIC",
        },
        "conditions": conditions,
        "training_window_spectra": spectra,
    }

    body = json.dumps(registration, indent=2, sort_keys=True)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    path = OUT / "v2_evaluation_preregistration.json"
    path.write_text(body, encoding="utf-8")
    (OUT / "v2_evaluation_preregistration.sha256").write_text(
        digest + "  v2_evaluation_preregistration.json\n", encoding="utf-8")
    print()
    print("[written] " + str(path))
    print("SHA256    " + digest)


if __name__ == "__main__":
    main()
