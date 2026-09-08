"""Source-only secondary comparison of reduced cell A best vs last.

This analysis is diagnostic and cannot alter the preregistered Gate A STOP.
It verifies the frozen secondary protocol, reproduces the primary best-checkpoint
numbers, and then evaluates the fixed final checkpoint.  Holdouts and
intermediate snapshots are deliberately not accessed.
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

from audit_radaz_v2_transport_error_geometry import MAXN, batch
from evaluate_radaz_arch_gate_A import (
    A1_HIGH,
    A1_RHO,
    A2,
    CELL_A,
    HIGH_N0,
    check_amendment,
    legacy_organisation_ratio as organisation_ratio,
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


PROTOCOL = REG_DIR / "ARCH_A_BEST_VS_LAST_SECONDARY_PROTOCOL.md"
PROTOCOL_SHA = REG_DIR / "ARCH_A_BEST_VS_LAST_SECONDARY_PROTOCOL.sha256"
PRIMARY_RESULT = REG_DIR / "arch_gate_A_result.json"
OUTPUT = REG_DIR / "legacy_replay_arch_A_best_vs_last_secondary.json"
CHECKPOINTS = {
    "best": CELL_A["workdir"] / "checkpoints" / "best.ckpt",
    "last": CELL_A["workdir"] / "checkpoints" / "last.ckpt",
}
FACTOR_METRICS = ("E_A", "E_delta", "E_r", "E_C")


def verify_protocol() -> str:
    digest = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    recorded = PROTOCOL_SHA.read_text(encoding="utf-8").split()[0]
    if digest != recorded:
        raise RuntimeError(
            "Secondary protocol digest mismatch:\n"
            f"  recorded {recorded}\n  actual   {digest}"
        )
    print(f"[secondary protocol verified] SHA256 {digest}")
    return digest


def evaluate_checkpoint(label, path, manifest, low, high, registration, lm, dev):
    if not path.exists():
        raise FileNotFoundError(path)
    model, epoch = build_model(CELL_A["config"], path, dev)
    print(
        f"\n[model] {label}: epoch={epoch}, "
        f"params={sum(p.numel() for p in model.parameters()) / 1e6:.1f} M"
    )
    rows = {}
    print(
        "%-11s %8s %10s %11s %9s %10s %9s %9s %9s %9s"
        % ("case", "n0", "O_hat/O", "field", "E_P", "NRMSE_G", "H", "E_A", "E_d", "E_r")
    )
    print("-" * 110)
    source_cases = [c for c in manifest["cases"] if c["role"] == "source"]
    if len(source_cases) != 6:
        raise RuntimeError(f"Expected exactly 6 source conditions, got {len(source_cases)}")
    for case in source_cases:
        key = case["case_key"]
        x, y, n0, B = batch(case, manifest, low, high, 1800, 2000)
        pred = predict_direct10(model, x, dev)
        ps, ts = spectra(lm, pred, dev), spectra(lm, y, dev)
        row = metrics(ps, ts, registration, key, B)
        pp, yy = denormalize(pred, low, high), denormalize(y, low, high)
        row["field_aggregate"] = float(
            np.mean(
                [
                    np.sqrt(
                        np.mean((pp[:, :, i] - yy[:, :, i]) ** 2)
                        / max(np.mean(yy[:, :, i] ** 2), 1e-300)
                    )
                    for i in range(3)
                ]
            )
        )
        row["O_ratio"] = organisation_ratio(ps, ts, B)
        row["n0"] = n0
        rows[key] = row
        print(
            "%-11s %8.2f %10.4f %11.5f %9.4f %10.4f %9.5f %9.4f %9.4f %9.4f"
            % (
                key,
                n0,
                row["O_ratio"],
                row["field_aggregate"],
                row["E_P_n1-32"],
                row["NRMSE_Gamma"],
                row["H"],
                row["E_A"],
                row["E_delta"],
                row["E_r"],
            )
        )
        del x, y, pred, pp, yy
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return epoch, rows


def summarize(rows):
    keys = list(rows)
    n0 = np.asarray([rows[k]["n0"] for k in keys])
    o = np.asarray([rows[k]["O_ratio"] for k in keys])
    good = np.isfinite(o)
    rho, p = spearmanr(n0[good], o[good])
    high_median = float(np.median([rows[k]["O_ratio"] for k in HIGH_N0]))
    a2_values = {
        metric: float(np.median([rows[k][metric] for k in keys]))
        for metric in A2
    }
    a2_items = {metric: value <= A2[metric] for metric, value in a2_values.items()}
    factor_medians = {
        metric: float(np.median([rows[k][metric] for k in keys]))
        for metric in FACTOR_METRICS
    }
    a1_items = {
        "spearman": bool(rho <= A1_RHO),
        "high_n0_median": bool(high_median <= A1_HIGH),
    }
    return {
        "A1": {
            "spearman": float(rho),
            "p": float(p),
            "high_n0_median": high_median,
            "item_pass": a1_items,
            "pass": bool(all(a1_items.values())),
        },
        "A2": {
            "values": a2_values,
            "item_pass": a2_items,
            "pass": bool(all(a2_items.values())),
        },
        "factor_medians_no_threshold": factor_medians,
    }


def verify_best_reproduction(rows, primary):
    fields = tuple(A2) + ("O_ratio", "n0", "E_A", "E_delta", "E_r", "E_C")
    worst = 0.0
    for case, old in primary["per_condition"].items():
        for field in fields:
            new_value, old_value = float(rows[case][field]), float(old[field])
            worst = max(worst, abs(new_value - old_value))
            if not np.isclose(new_value, old_value, rtol=1e-7, atol=1e-10):
                raise RuntimeError(
                    f"best.ckpt failed primary reproduction: {case} {field}: "
                    f"new={new_value}, primary={old_value}"
                )
    print(f"[primary best reproduction PASS] max absolute difference={worst:.3e}")
    return worst


def diagnostic_verdict(last_summary):
    a1 = last_summary["A1"]["pass"]
    a2 = last_summary["A2"]["pass"]
    transport = last_summary["A2"]["item_pass"]["NRMSE_Gamma"]
    if a1 and a2:
        return (
            "LAST_COMPETENT_WITH_EFFECT -- evidence for a validation-MSE "
            "checkpoint-selector mismatch; primary STOP remains immutable."
        )
    if not a1:
        return (
            "LAST_LOST_TARGET_EFFECT -- final checkpoint cannot be an "
            "architecture-contrast baseline; primary STOP remains immutable."
        )
    if transport:
        return (
            "LAST_TRANSPORT_PASS_BUT_OTHER_A2_FAIL -- final checkpoint is not "
            "a competent baseline; no selector-mismatch conclusion."
        )
    return (
        "LAST_ALSO_FAILS_TRANSPORT -- best-vs-last selection does not repair "
        "baseline competence; capacity/training/optimization causes remain "
        "unresolved. Primary STOP remains immutable."
    )


def main():
    import sys
    if "--replay-legacy" not in sys.argv:
        raise SystemExit("Historical metrics only. Use evaluate_radaz_corrected_A.py, or --replay-legacy.")
    check_amendment()
    protocol_sha = verify_protocol()
    primary = json.loads(PRIMARY_RESULT.read_text(encoding="utf-8"))
    if primary["A2_pass"] or not primary["verdict"].startswith("STOP"):
        raise RuntimeError("Primary Gate A result is not the expected immutable STOP")

    registration = load_registration()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    low = np.asarray(manifest["normalization"]["low"], dtype=np.float64)
    high = np.asarray(manifest["normalization"]["high"], dtype=np.float64)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lm = PEPAPICSpectralLoss(
        data_root=str(MANIFEST),
        max_mode=MAXN,
        radial_bands=4,
        radial_min_m=0.09e-2,
        radial_max_m=1.19e-2,
        coordinate_system="integer_power_cross",
    ).to(dev)

    results = {}
    for label in ("best", "last"):
        epoch, rows = evaluate_checkpoint(
            label, CHECKPOINTS[label], manifest, low, high, registration, lm, dev
        )
        results[label] = {
            "checkpoint": str(CHECKPOINTS[label]),
            "epoch": int(epoch),
            "per_condition": rows,
            "summary": summarize(rows),
        }

    max_best_difference = verify_best_reproduction(results["best"]["per_condition"], primary)
    best_summary, last_summary = results["best"]["summary"], results["last"]["summary"]
    delta = {
        metric: last_summary["A2"]["values"][metric]
        - best_summary["A2"]["values"][metric]
        for metric in A2
    }
    factor_delta = {
        metric: last_summary["factor_medians_no_threshold"][metric]
        - best_summary["factor_medians_no_threshold"][metric]
        for metric in FACTOR_METRICS
    }
    verdict = diagnostic_verdict(last_summary)

    print("\n" + "=" * 96)
    print("FIXED SECONDARY BEST-vs-LAST SUMMARY (SOURCE ONLY)")
    print("=" * 96)
    for label, summary in (("best", best_summary), ("last", last_summary)):
        a1, a2 = summary["A1"], summary["A2"]
        print(
            f"{label:>4}: A1={a1['pass']} rho={a1['spearman']:+.4f} "
            f"high={a1['high_n0_median']:.4f} | A2={a2['pass']} "
            f"field={a2['values']['field_aggregate']:.5f} "
            f"E_P={a2['values']['E_P_n1-32']:.4f} "
            f"NRMSE_G={a2['values']['NRMSE_Gamma']:.4f} "
            f"H={a2['values']['H']:.5f}"
        )
    print("DIAGNOSTIC VERDICT: " + verdict)
    print("PRIMARY VERDICT (UNCHANGED): " + primary["verdict"])

    payload = {
        "status": "secondary_diagnostic_only",
        "protocol_sha256": protocol_sha,
        "primary_gate_result": str(PRIMARY_RESULT),
        "primary_verdict_immutable": primary["verdict"],
        "best_reproduction_max_abs_difference": max_best_difference,
        "data_scope": "six source conditions only; frames 1800-1999; no holdout",
        "results": results,
        "last_minus_best_A2": delta,
        "last_minus_best_factor_medians": factor_delta,
        "diagnostic_verdict": verdict,
        "B_C_authorized": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[written] {OUTPUT}")


if __name__ == "__main__":
    main()
