"""Evaluate physics-loss v2 against the frozen preregistration.

Reads workdirs/2D_RadAz/radaz_physics_loss_v2_manifests/v2_evaluation_preregistration.json
and refuses to run if its SHA256 does not match the recorded digest, so the
gates, bands, masks and thresholds cannot drift after the fact.  Nothing in
this script recomputes a frozen quantity.

All spectral quantities are on exact integer azimuthal modes.  The q
coordinate appears only as a label defining the resonance band; no complex
coefficient is ever interpolated.

Usage
    python evaluate_radaz_physics_loss_v2.py --stage source
    python evaluate_radaz_physics_loss_v2.py --stage e_axis
    python evaluate_radaz_physics_loss_v2.py --stage b_axis
    python evaluate_radaz_physics_loss_v2.py --stage all --verdict

The stages exist so the preregistered order can be followed literally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import torch

from evaluate_radaz_conditioned_factorial import (
    AFT, CELLS, MANIFEST, MODEL_H, MODEL_W, PRE, TEST_START, TEST_STOP,
    VALID_H, VALID_W, build_model, condition_vector, denormalize,
    load_case_frames, predict_direct10,
)
from openstl.methods.pepapic_spectral_loss import PEPAPICSpectralLoss

ROOT = Path(r"C:\Users\astro\research\SimVPv2")
REG_DIR = ROOT / "workdirs/2D_RadAz/radaz_physics_loss_v2_manifests"
REG = REG_DIR / "v2_evaluation_preregistration.json"

V2 = {
    "workdir": ROOT / "workdirs/2D_RadAz/radaz_axis_factorial_UPv2_powercross_direct10_bs1_60ep",
    "config": ROOT / "configs/custom/pepapic/SimVP_gSTA_radaz_factorial_UPv2_60ep.py",
}
STAGES = {
    "source": lambda role: role == "source",
    "e_axis": lambda key: key in ("E22.5_B20", "E25_B20"),
    "b_axis": lambda key: key in ("E10_B15", "E10_B25"),
}
FIELDS = ("electron_den", "ion_den", "phi")


def load_registration():
    body = REG.read_text(encoding="utf-8")
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    recorded = (REG_DIR / "v2_evaluation_preregistration.sha256"
                ).read_text(encoding="utf-8").split()[0]
    if digest != recorded:
        raise RuntimeError(
            "Preregistration digest mismatch: the frozen protocol has been "
            "edited.\n  recorded %s\n  actual   %s" % (recorded, digest))
    print("[preregistration verified] SHA256 %s" % digest)
    return json.loads(body)


def spectra(loss_module, normalized, device):
    """-> P_N, P_E, cross (complex), each [R, N] on exact integer modes."""
    tensor = torch.from_numpy(normalized).to(device)
    coeff = loss_module._band_coefficients(tensor, physical_units=False)
    pn, pe, cr, ci = loss_module._ensemble_spectra(coeff)
    # average the direct10 windows, keep radial bands and modes
    return (pn.mean(dim=0).cpu().numpy(), pe.mean(dim=0).cpu().numpy(),
            (cr.mean(dim=0) + 1j * ci.mean(dim=0)).cpu().numpy())


def metrics(pred, true, registration, key, magnetic_t):
    """All preregistered scalars for one condition, from integer-mode spectra."""
    pn_p, pe_p, cross_p = pred
    pn_t, pe_t, cross_t = true
    entry = registration["conditions"][key]
    modes = np.arange(1, pn_t.shape[-1] + 1)
    eps_rel = registration["spectral_gate"]["epsilon_relative_to_true_peak"]
    kappa = registration["factor_gate"]["cross_mask_kappa"]

    out = {}
    # --- spectral: log-power RMSE per band
    bands = {
        "n1-6": (modes >= 1) & (modes <= 6),
        "n9-21": (modes >= 9) & (modes <= 21),
        "n1-32": (modes >= 1) & (modes <= 32),
        "resonance": np.isin(modes, entry["resonance_band_modes"]),
    }
    for name, selection in bands.items():
        total = 0.0
        for predicted, truth in ((pn_p, pn_t), (pe_p, pe_t)):
            floor = eps_rel * truth.max(axis=-1, keepdims=True)
            difference = (np.log(predicted[:, selection] + floor)
                          - np.log(truth[:, selection] + floor))
            total += float(np.mean(difference ** 2))
        out["E_P_" + name] = float(np.sqrt(0.5 * total))

    # --- high-mode hallucination over the frozen low-power set L
    low = np.isin(modes, entry["low_power_set_L_modes"])
    denominator = float(np.sum(pn_t[:, modes <= 32])) + 1e-300
    out["H"] = float(np.sum(np.maximum(pn_p[:, low] - pn_t[:, low], 0.0))
                     / denominator)

    # --- factor decomposition on the frozen mask
    geometric_t = np.sqrt(pn_t * pe_t)
    mask = geometric_t > kappa * geometric_t.max(axis=-1, keepdims=True)
    coherent_p = cross_p / np.maximum(np.sqrt(pn_p * pe_p), 1e-300)
    coherent_t = cross_t / np.maximum(geometric_t, 1e-300)
    out["mask_coverage"] = float(mask.mean())
    out["E_C"] = float(np.sum(np.abs(coherent_p - coherent_t) ** 2 * mask)
                       / max(mask.sum(), 1))
    amp_p = np.sqrt(pn_p * pe_p)
    amp_t = np.sqrt(pn_t * pe_t)
    out["E_A"] = float(np.sum(np.abs(np.log((amp_p + 1e-300) / (amp_t + 1e-300)))
                              * mask) / max(mask.sum(), 1))
    delta = np.angle(coherent_p) - np.angle(coherent_t)
    out["E_delta"] = float(np.sum((1.0 - np.cos(delta)) * mask)
                           / max(mask.sum(), 1))
    out["E_r"] = float(np.sum(np.abs(np.abs(coherent_p) - np.abs(coherent_t))
                              * mask) / max(mask.sum(), 1))

    # --- constructive transport, exact integer modes
    gamma_p = -np.real(cross_p) / magnetic_t
    gamma_t = -np.real(cross_t) / magnetic_t
    out["NRMSE_Gamma"] = float(
        np.sqrt(np.mean((gamma_p - gamma_t) ** 2))
        / max(np.sqrt(np.mean(gamma_t ** 2)), 1e-300))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["source", "e_axis", "b_axis", "all"])
    parser.add_argument("--checkpoint", default="best", choices=["best", "last"])
    parser.add_argument("--verdict", action="store_true")
    args = parser.parse_args()

    registration = load_registration()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    low = np.asarray(manifest["normalization"]["low"], dtype=np.float64)
    high = np.asarray(manifest["normalization"]["high"], dtype=np.float64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_module = PEPAPICSpectralLoss(
        data_root=str(MANIFEST), max_mode=64, radial_bands=4,
        radial_min_m=0.09e-2, radial_max_m=1.19e-2,
        coordinate_system="integer_power_cross").to(device)

    models = {}
    for name, spec in (("v2", V2), ("U-P", CELLS["U-P"]), ("U-D", CELLS["U-D"])):
        ckpt = spec["workdir"] / "checkpoints" / (args.checkpoint + ".ckpt")
        if not ckpt.exists():
            print("[skip] %s: no %s" % (name, ckpt))
            continue
        models[name], epoch = build_model(spec["config"], ckpt, device)
        print("[model] %-4s epoch=%d" % (name, epoch))
    if "v2" not in models:
        raise SystemExit("v2 checkpoint not present yet; training still running")

    starts = [s for s in range(TEST_START, TEST_STOP, PRE + AFT)
              if s + PRE + AFT <= TEST_STOP]
    electron_mass, electron_charge, ly = 9.1093837015e-31, 1.602176634e-19, 1.28e-2

    selected = []
    for case in manifest["cases"]:
        key = case["case_key"]
        if args.stage == "all":
            keep = True
        elif args.stage == "source":
            keep = case["role"] == "source"
        else:
            keep = STAGES[args.stage](key)
        if keep:
            selected.append(case)

    results = {}
    for case in selected:
        key = case["case_key"]
        condition, drift, mode_n0 = condition_vector(case, manifest)
        magnetic_t = (mode_n0 * 2.0 * np.pi * electron_mass * drift
                      / (electron_charge * ly))
        frames = load_case_frames(
            case, low, high, TEST_START, TEST_START + len(starts) * (PRE + AFT))[0]
        inputs = np.empty((len(starts), PRE, 5, MODEL_H, MODEL_W), dtype=np.float32)
        truth = np.empty((len(starts), AFT, 3, MODEL_H, MODEL_W), dtype=np.float32)
        for index, s in enumerate(starts):
            i0 = s - TEST_START
            inputs[index, :, :3] = frames[i0: i0 + PRE]
            inputs[index, :, 3:] = condition[None, :, None, None]
            truth[index] = frames[i0 + PRE: i0 + PRE + AFT]
        true_spec = spectra(loss_module, truth, device)
        truth_phys = denormalize(truth, low, high)

        row = {"role": case["role"], "n0": mode_n0, "models": {}}
        for name, model in models.items():
            pred = predict_direct10(model, inputs, device)
            entry = metrics(spectra(loss_module, pred, device), true_spec,
                            registration, key, magnetic_t)
            pred_phys = denormalize(pred, low, high)
            for index, field in enumerate(FIELDS):
                error = np.mean((pred_phys[:, :, index] - truth_phys[:, :, index]) ** 2)
                scale = np.mean(truth_phys[:, :, index] ** 2)
                entry["field_" + field] = float(np.sqrt(error / max(scale, 1e-300)))
            entry["field_aggregate"] = float(np.mean(
                [entry["field_" + f] for f in FIELDS]))
            row["models"][name] = entry
        results[key] = row
        print("[done] " + key, flush=True)
        del frames, inputs, truth

    order = list(results)
    show = [("E_P_n1-32", "E_P n1-32"), ("E_P_resonance", "E_P reson"),
            ("H", "H halluc"), ("E_A", "E_A"), ("E_delta", "E_delta"),
            ("E_r", "E_r"), ("NRMSE_Gamma", "NRMSE_G"),
            ("field_aggregate", "field agg")]
    for key_metric, label in show:
        print()
        print("=" * 96)
        print("%s   (lower is better)" % label)
        print("=" * 96)
        print("%-11s %-8s " % ("case", "role")
              + " ".join("%12s" % n for n in models))
        for key in order:
            print("%-11s %-8s " % (key, results[key]["role"][:7])
                  + " ".join("%12.5f" % results[key]["models"][n][key_metric]
                             for n in models))

    if args.verdict and "U-P" in models:
        holdout = [k for k in order if results[k]["role"] != "source"]
        if len(holdout) == 4:
            print()
            print("=" * 96)
            print("PREREGISTERED GATES  (v2 vs U-P, holdout conditions)")
            print("=" * 96)

            def med(name, metric):
                return float(np.median([results[k]["models"][name][metric]
                                        for k in holdout]))

            def wins(metric):
                return sum(1 for k in holdout
                           if results[k]["models"]["v2"][metric]
                           < results[k]["models"]["U-P"][metric])

            checks = []
            ep2, ep1 = med("v2", "E_P_n1-32"), med("U-P", "E_P_n1-32")
            checks.append(("spectral: >=3/4 conditions improve",
                           wins("E_P_n1-32") >= 3))
            checks.append(("spectral: median E_P >=15%% lower (%.4f -> %.4f)"
                           % (ep1, ep2), ep2 <= 0.85 * ep1))
            checks.append(("spectral: H not worse (%.4f -> %.4f)"
                           % (med("U-P", "H"), med("v2", "H")),
                           med("v2", "H") <= med("U-P", "H")))
            ea2, ea1 = med("v2", "E_A"), med("U-P", "E_A")
            checks.append(("factor: median E_A >=15%% better (%.4f -> %.4f)"
                           % (ea1, ea2), ea2 <= 0.85 * ea1))
            ed2, ed1 = med("v2", "E_delta"), med("U-P", "E_delta")
            checks.append(("factor: median E_delta >=15%% better (%.4f -> %.4f)"
                           % (ed1, ed2), ed2 <= 0.85 * ed1))
            er2, er1 = med("v2", "E_r"), med("U-P", "E_r")
            checks.append(("factor: E_r <=10%% worse (%.4f -> %.4f)"
                           % (er1, er2), er2 <= 1.10 * er1))
            e_axis = [k for k in holdout if k in ("E22.5_B20", "E25_B20")]
            b_axis = [k for k in holdout if k in ("E10_B15", "E10_B25")]
            for label_axis, group in (("E-axis", e_axis), ("B-axis", b_axis)):
                a = np.median([results[k]["models"]["v2"]["NRMSE_Gamma"] for k in group])
                b = np.median([results[k]["models"]["U-P"]["NRMSE_Gamma"] for k in group])
                checks.append(("transport: %s median improves (%.4f -> %.4f)"
                               % (label_axis, b, a), a < b))
            checks.append(("transport: >=3/4 conditions improve",
                           wins("NRMSE_Gamma") >= 3))
            fa2, fa1 = med("v2", "field_aggregate"), med("U-P", "field_aggregate")
            checks.append(("safeguard: field aggregate <=10%% worse (%.4f -> %.4f)"
                           % (fa1, fa2), fa2 <= 1.10 * fa1))
            checks.append(("safeguard: each field <=25%% worse",
                           all(med("v2", "field_" + f) <= 1.25 * med("U-P", "field_" + f)
                               for f in FIELDS)))
            for text, ok in checks:
                print("  [%s] %s" % ("PASS" if ok else "FAIL", text))

            spectral_ok = all(ok for text, ok in checks if text.startswith("spectral"))
            factor_ok = all(ok for text, ok in checks if text.startswith("factor"))
            transport_ok = all(ok for text, ok in checks if text.startswith("transport"))
            field_ok = all(ok for text, ok in checks if text.startswith("safeguard"))
            if not spectral_ok:
                verdict = "FAIL - representation repair failed"
            elif spectral_ok and factor_ok and not transport_ok:
                verdict = "PARTIAL - representation repaired, transport not recovered"
            elif spectral_ok and factor_ok and transport_ok and field_ok:
                verdict = "MECHANISTIC SUCCESS"
            else:
                verdict = "PARTIAL - see individual gates"
            print()
            print("  VERDICT: " + verdict)
            gamma = med("v2", "NRMSE_Gamma")
            print("  application-level (median NRMSE_Gamma < 0.8): %.4f -> %s"
                  % (gamma, "yes" if gamma < 0.8 else "no"))
            print("  reported separately; never conflated with mechanistic success")

    out = REG_DIR / ("v2_evaluation_%s.json" % args.stage)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print()
    print("[written] " + str(out))


if __name__ == "__main__":
    main()
