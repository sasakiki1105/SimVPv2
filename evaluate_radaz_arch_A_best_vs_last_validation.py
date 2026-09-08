"""Validation-window diagnostic for reduced cell A best vs last.

Runs source conditions only and preserves the immutable primary STOP.  The
protocol hash is checked before any model evaluation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import torch
from scipy.stats import spearmanr

from audit_radaz_v2_transport_error_geometry import MAXN, batch
from evaluate_radaz_arch_gate_A import (
    A1_HIGH,
    A1_RHO,
    A2,
    CELL_A,
    HIGH_N0,
    check_amendment,
)
from evaluate_radaz_conditioned_factorial import (
    MANIFEST,
    build_model,
    denormalize,
    predict_direct10,
)
from evaluate_radaz_physics_loss_v2 import (
    REG_DIR,
    load_registration,
    metrics,
    spectra,
)
from openstl.methods.pepapic_spectral_loss import PEPAPICSpectralLoss


PROTOCOL = REG_DIR / "ARCH_A_BEST_VS_LAST_VALIDATION_PROTOCOL.md"
PROTOCOL_SHA = REG_DIR / "ARCH_A_BEST_VS_LAST_VALIDATION_PROTOCOL.sha256"
SOURCE_TEST_RESULT = REG_DIR / "arch_A_best_vs_last_secondary.json"
OUTPUT = REG_DIR / "legacy_replay_arch_A_best_vs_last_validation.json"
CHECKPOINTS = {
    "best": CELL_A["workdir"] / "checkpoints" / "best.ckpt",
    "last": CELL_A["workdir"] / "checkpoints" / "last.ckpt",
}
TRAIN_LOG = CELL_A["workdir"] / "train_20260906_223254.log"
VAL_LO, VAL_HI = 1600, 1800
O_FLOOR, W_FLOOR, O_EPS = 0.05, 1.0e-3, 1.0e-6
FACTOR_METRICS = ("E_A", "E_delta", "E_r", "E_C")
LOWER_METRICS = (
    "field_aggregate",
    "E_P_n1-32",
    "NRMSE_Gamma",
    "H",
    "E_A",
    "E_delta",
    "E_r",
    "E_C",
)


def verify_protocol():
    digest = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    recorded = PROTOCOL_SHA.read_text(encoding="utf-8").split()[0]
    if digest != recorded:
        raise RuntimeError(
            "Validation diagnostic protocol mismatch:\n"
            f"  recorded {recorded}\n  actual   {digest}"
        )
    print(f"[validation protocol verified] SHA256 {digest}")
    return digest


def logged_val_losses():
    pattern = re.compile(r"Epoch\s+(29|59):.*Vali Loss:\s+([0-9.eE+-]+)")
    found = {}
    for line in TRAIN_LOG.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            found[int(match.group(1))] = float(match.group(2))
    if set(found) != {29, 59}:
        raise RuntimeError(f"Could not recover epoch 29/59 val losses: {found}")
    return {"best_epoch29": found[29], "last_epoch59": found[59]}


def organisation_metrics(pred, true, magnetic_t):
    pn_p, pe_p, cr_p = pred
    pn_t, pe_t, cr_t = true
    amp_p = np.sqrt(pn_p * pe_p)
    amp_t = np.sqrt(pn_t * pe_t)
    o_p = np.real(cr_p) / np.maximum(amp_p, 1e-300)
    o_t = np.real(cr_t) / np.maximum(amp_t, 1e-300)
    gamma_t = -np.real(cr_t) / magnetic_t
    weight = gamma_t ** 2
    weight /= max(float(weight.sum()), 1e-300)
    mask = (np.abs(o_t) > O_FLOOR) & (weight > W_FLOOR)
    if not mask.any():
        return float("nan"), float("nan"), 0
    ratio = float(np.median(o_p[mask] / o_t[mask]))
    e_o = float(
        np.median(
            np.abs(
                np.log(
                    (np.abs(o_p[mask]) + O_EPS)
                    / (np.abs(o_t[mask]) + O_EPS)
                )
            )
        )
    )
    return ratio, e_o, int(mask.sum())


def evaluate(label, path, manifest, low, high, registration, loss_module, device):
    model, epoch = build_model(CELL_A["config"], path, device)
    print(f"\n[model] {label}: epoch={epoch}")
    rows = {}
    source_cases = [case for case in manifest["cases"] if case["role"] == "source"]
    if len(source_cases) != 6:
        raise RuntimeError(f"Expected 6 source cases, got {len(source_cases)}")
    for case in source_cases:
        key = case["case_key"]
        x, y, n0, magnetic_t = batch(case, manifest, low, high, VAL_LO, VAL_HI)
        pred = predict_direct10(model, x, device)
        pred_spectra = spectra(loss_module, pred, device)
        true_spectra = spectra(loss_module, y, device)
        row = metrics(pred_spectra, true_spectra, registration, key, magnetic_t)
        physical_pred = denormalize(pred, low, high)
        physical_true = denormalize(y, low, high)
        row["field_aggregate"] = float(
            np.mean(
                [
                    np.sqrt(
                        np.mean((physical_pred[:, :, i] - physical_true[:, :, i]) ** 2)
                        / max(np.mean(physical_true[:, :, i] ** 2), 1e-300)
                    )
                    for i in range(3)
                ]
            )
        )
        ratio, e_o, count = organisation_metrics(
            pred_spectra, true_spectra, magnetic_t
        )
        row.update({"O_ratio": ratio, "E_O": e_o, "O_mask_count": count, "n0": n0})
        rows[key] = row
        print(
            f"{key:11s} n0={n0:6.2f} O={ratio:7.4f} E_O={e_o:7.4f} "
            f"field={row['field_aggregate']:.5f} E_P={row['E_P_n1-32']:.4f} "
            f"NRMSE_G={row['NRMSE_Gamma']:.4f} H={row['H']:.5f}"
        )
        del x, y, pred, physical_pred, physical_true
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return int(epoch), rows


def summarize(rows):
    keys = list(rows)
    n0 = np.asarray([rows[key]["n0"] for key in keys])
    ratio = np.asarray([rows[key]["O_ratio"] for key in keys])
    good = np.isfinite(ratio)
    rho, p = spearmanr(n0[good], ratio[good])
    a2_values = {
        metric: float(np.median([rows[key][metric] for key in keys]))
        for metric in A2
    }
    a2_items = {metric: a2_values[metric] <= limit for metric, limit in A2.items()}
    high_median = float(np.median([rows[key]["O_ratio"] for key in HIGH_N0]))
    return {
        "A1_descriptive_on_validation_window": {
            "spearman": float(rho),
            "p": float(p),
            "high_n0_median": high_median,
            "pass_rho": bool(rho <= A1_RHO),
            "pass_high": bool(high_median <= A1_HIGH),
            "pass": bool(rho <= A1_RHO and high_median <= A1_HIGH),
        },
        "A2_descriptive_on_validation_window": {
            "values": a2_values,
            "item_pass": a2_items,
            "pass": bool(all(a2_items.values())),
        },
        "factor_medians": {
            metric: float(np.median([rows[key][metric] for key in keys]))
            for metric in FACTOR_METRICS
        },
        "E_O": {
            "median": float(np.median([rows[key]["E_O"] for key in keys])),
            "worst": float(np.max([rows[key]["E_O"] for key in keys])),
            "worst_condition": max(keys, key=lambda key: rows[key]["E_O"]),
        },
    }


def flatten(summary):
    values = dict(summary["A2_descriptive_on_validation_window"]["values"])
    values.update(summary["factor_medians"])
    values["E_O"] = summary["E_O"]["median"]
    return values


def main():
    import sys
    if "--replay-legacy" not in sys.argv:
        raise SystemExit("Historical sign-insensitive metrics only. Use evaluate_radaz_corrected_A.py, or --replay-legacy.")
    check_amendment()
    protocol_sha = verify_protocol()
    logged_loss = logged_val_losses()
    if not logged_loss["best_epoch29"] < logged_loss["last_epoch59"]:
        raise RuntimeError("Expected epoch 29 to have lower logged val_loss than epoch 59")

    source_test = json.loads(SOURCE_TEST_RESULT.read_text(encoding="utf-8"))
    if source_test["B_C_authorized"]:
        raise RuntimeError("Source-test secondary result unexpectedly authorized B/C")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    low = np.asarray(manifest["normalization"]["low"], dtype=np.float64)
    high = np.asarray(manifest["normalization"]["high"], dtype=np.float64)
    registration = load_registration()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_module = PEPAPICSpectralLoss(
        data_root=str(MANIFEST),
        max_mode=MAXN,
        radial_bands=4,
        radial_min_m=0.09e-2,
        radial_max_m=1.19e-2,
        coordinate_system="integer_power_cross",
    ).to(device)

    results = {}
    for label in ("best", "last"):
        epoch, rows = evaluate(
            label,
            CHECKPOINTS[label],
            manifest,
            low,
            high,
            registration,
            loss_module,
            device,
        )
        results[label] = {
            "checkpoint": str(CHECKPOINTS[label]),
            "epoch": epoch,
            "per_condition": rows,
            "summary": summarize(rows),
        }

    best_values = flatten(results["best"]["summary"])
    last_values = flatten(results["last"]["summary"])
    validation_deltas = {
        metric: last_values[metric] - best_values[metric]
        for metric in best_values
    }
    e_o_improvement_count = sum(
        results["last"]["per_condition"][key]["E_O"]
        < results["best"]["per_condition"][key]["E_O"]
        for key in results["best"]["per_condition"]
    )

    test_best = source_test["results"]["best"]["summary"]
    test_last = source_test["results"]["last"]["summary"]
    test_values = {
        label: {
            **summary["A2"]["values"],
            **summary["factor_medians_no_threshold"],
        }
        for label, summary in (("best", test_best), ("last", test_last))
    }
    direction = {}
    for metric in LOWER_METRICS:
        validation_delta = validation_deltas[metric]
        test_delta = test_values["last"][metric] - test_values["best"][metric]
        direction[metric] = {
            "validation_last_minus_best": validation_delta,
            "source_test_last_minus_best": test_delta,
            "validation_last_better_despite_higher_logged_val_loss": bool(
                validation_delta < 0
            ),
            "window_direction_reversal": bool(validation_delta * test_delta < 0),
        }

    payload = {
        "status": "post_primary_validation_window_diagnostic",
        "protocol_sha256": protocol_sha,
        "data_scope": "six source conditions; validation frames 1600-1799; no holdout",
        "logged_checkpoint_selection_val_loss": logged_loss,
        "source_test_result_used_for_direction_only": str(SOURCE_TEST_RESULT),
        "results": results,
        "validation_last_minus_best": validation_deltas,
        "E_O_last_improvement_count_out_of_6": int(e_o_improvement_count),
        "per_metric_window_directions": direction,
        "primary_STOP_immutable": True,
        "B_C_authorized": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 100)
    print("VALIDATION-WINDOW BEST-vs-LAST (SOURCE ONLY; PRIMARY STOP IMMUTABLE)")
    print("=" * 100)
    print(f"logged val_loss: best={logged_loss['best_epoch29']:.7g}, last={logged_loss['last_epoch59']:.7g}")
    for label in ("best", "last"):
        summary = results[label]["summary"]
        a1 = summary["A1_descriptive_on_validation_window"]
        a2 = summary["A2_descriptive_on_validation_window"]
        print(
            f"{label:>4}: rho={a1['spearman']:+.4f}, high={a1['high_n0_median']:.4f}, "
            f"E_O_med={summary['E_O']['median']:.4f}, E_O_worst={summary['E_O']['worst']:.4f}, "
            f"field={a2['values']['field_aggregate']:.5f}, E_P={a2['values']['E_P_n1-32']:.4f}, "
            f"NRMSE_G={a2['values']['NRMSE_Gamma']:.4f}, H={a2['values']['H']:.5f}"
        )
    print(f"E_O conditions improved by last: {e_o_improvement_count}/6")
    print("same-window last-better metrics:", [m for m, x in direction.items() if x["validation_last_better_despite_higher_logged_val_loss"]])
    print("window-direction reversals:", [m for m, x in direction.items() if x["window_direction_reversal"]])
    print(f"[written] {OUTPUT}")


if __name__ == "__main__":
    main()
