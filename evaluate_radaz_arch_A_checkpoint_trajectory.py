"""Exploratory checkpoint trajectory for reduced architecture cell A.

The primary Gate A STOP is immutable.  This script evaluates only the six
source conditions on the already opened validation and source-test windows.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import OrderedDict

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from audit_radaz_v2_transport_error_geometry import MAXN, batch
from evaluate_radaz_arch_A_best_vs_last_validation import organisation_metrics
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
from evaluate_radaz_physics_loss_v2 import REG_DIR, load_registration, metrics, spectra
from openstl.methods.pepapic_spectral_loss import PEPAPICSpectralLoss


PROTOCOL = REG_DIR / "ARCH_A_CHECKPOINT_TRAJECTORY_PROTOCOL.md"
PROTOCOL_SHA = REG_DIR / "ARCH_A_CHECKPOINT_TRAJECTORY_PROTOCOL.sha256"
BEST_LAST_TEST = REG_DIR / "arch_A_best_vs_last_secondary.json"
BEST_LAST_VALIDATION = REG_DIR / "arch_A_best_vs_last_validation.json"
OUTPUT = REG_DIR / "legacy_replay_arch_A_checkpoint_trajectory.json"
OVERVIEW_PLOT = REG_DIR / "legacy_replay_arch_A_checkpoint_trajectory_overview.png"
CONDITION_PLOT = REG_DIR / "legacy_replay_arch_A_checkpoint_trajectory_conditions.png"
TRAIN_LOG = CELL_A["workdir"] / "train_20260906_223254.log"

CHECKPOINTS = OrderedDict(
    [
        (5, CELL_A["workdir"] / "checkpoints" / "snapshot-epoch=05.ckpt"),
        (10, CELL_A["workdir"] / "checkpoints" / "snapshot-epoch=10.ckpt"),
        (15, CELL_A["workdir"] / "checkpoints" / "snapshot-epoch=15.ckpt"),
        (20, CELL_A["workdir"] / "checkpoints" / "snapshot-epoch=20.ckpt"),
        (29, CELL_A["workdir"] / "checkpoints" / "best.ckpt"),
        (30, CELL_A["workdir"] / "checkpoints" / "snapshot-epoch=30.ckpt"),
        (40, CELL_A["workdir"] / "checkpoints" / "snapshot-epoch=40.ckpt"),
        (50, CELL_A["workdir"] / "checkpoints" / "snapshot-epoch=50.ckpt"),
        (59, CELL_A["workdir"] / "checkpoints" / "last.ckpt"),
    ]
)
WINDOWS = OrderedDict([("validation", (1600, 1800)), ("source_test", (1800, 2000))])
FACTOR_METRICS = ("E_A", "E_delta", "E_r", "E_C")
TRACKED_METRICS = (
    "spearman",
    "high_n0_median",
    "E_O_median",
    "E_O_worst",
    "field_aggregate",
    "E_P_n1-32",
    "NRMSE_Gamma",
    "H",
    *FACTOR_METRICS,
)


def verify_protocol():
    digest = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    recorded = PROTOCOL_SHA.read_text(encoding="utf-8").split()[0]
    if digest != recorded:
        raise RuntimeError(
            "Checkpoint trajectory protocol mismatch:\n"
            f"  recorded {recorded}\n  actual   {digest}"
        )
    print(f"[trajectory protocol verified] SHA256 {digest}")
    return digest


def logged_val_losses():
    pattern = re.compile(r"Epoch\s+(\d+):.*Vali Loss:\s+([0-9.eE+-]+)")
    found = {}
    for line in TRAIN_LOG.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            found[int(match.group(1))] = float(match.group(2))
    missing = sorted(set(CHECKPOINTS) - set(found))
    if missing:
        raise RuntimeError(f"Missing logged val_loss for epochs {missing}")
    return {str(epoch): found[epoch] for epoch in CHECKPOINTS}


def evaluate_rows(model, manifest, low, high, registration, loss_module, device, lo, hi):
    rows = {}
    source_cases = [case for case in manifest["cases"] if case["role"] == "source"]
    if len(source_cases) != 6:
        raise RuntimeError(f"Expected 6 source cases, got {len(source_cases)}")
    for case in source_cases:
        key = case["case_key"]
        x, y, n0, magnetic_t = batch(case, manifest, low, high, lo, hi)
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
        del x, y, pred, physical_pred, physical_true
    return rows


def summarize(rows):
    keys = list(rows)
    n0 = np.asarray([rows[key]["n0"] for key in keys])
    ratio = np.asarray([rows[key]["O_ratio"] for key in keys])
    good = np.isfinite(ratio)
    rho, p = spearmanr(n0[good], ratio[good])
    high_median = float(np.median([rows[key]["O_ratio"] for key in HIGH_N0]))
    a2_values = {
        metric: float(np.median([rows[key][metric] for key in keys]))
        for metric in A2
    }
    a2_items = {metric: a2_values[metric] <= limit for metric, limit in A2.items()}
    e_o = {key: rows[key]["E_O"] for key in keys}
    return {
        "A1_descriptive": {
            "spearman": float(rho),
            "p": float(p),
            "high_n0_median": high_median,
            "pass_rho": bool(rho <= A1_RHO),
            "pass_high": bool(high_median <= A1_HIGH),
            "pass": bool(rho <= A1_RHO and high_median <= A1_HIGH),
        },
        "A2_descriptive": {
            "values": a2_values,
            "item_pass": a2_items,
            "pass": bool(all(a2_items.values())),
        },
        "factor_medians": {
            metric: float(np.median([rows[key][metric] for key in keys]))
            for metric in FACTOR_METRICS
        },
        "E_O": {
            "median": float(np.median(list(e_o.values()))),
            "worst": float(max(e_o.values())),
            "worst_condition": max(e_o, key=e_o.get),
        },
    }


def flatten(summary):
    return {
        "spearman": summary["A1_descriptive"]["spearman"],
        "high_n0_median": summary["A1_descriptive"]["high_n0_median"],
        "E_O_median": summary["E_O"]["median"],
        "E_O_worst": summary["E_O"]["worst"],
        **summary["A2_descriptive"]["values"],
        **summary["factor_medians"],
    }


def largest_steps(results):
    epochs = list(CHECKPOINTS)
    output = {}
    for window in WINDOWS:
        output[window] = {}
        flat = [flatten(results[str(epoch)]["windows"][window]["summary"]) for epoch in epochs]
        for metric in TRACKED_METRICS:
            values = np.asarray([row[metric] for row in flat], dtype=np.float64)
            deltas = np.diff(values)
            i = int(np.argmax(np.abs(deltas)))
            output[window][metric] = {
                "from_epoch": epochs[i],
                "to_epoch": epochs[i + 1],
                "delta": float(deltas[i]),
                "absolute_delta": float(abs(deltas[i])),
                "total_range": float(np.max(values) - np.min(values)),
            }
    return output


def make_plots(payload):
    epochs = np.asarray(list(CHECKPOINTS), dtype=int)
    display_ticks = [5, 10, 15, 20, 29.5, 40, 50, 59]
    display_labels = ["5", "10", "15", "20", "29/30", "40", "50", "59"]
    colors = {"validation": "tab:blue", "source_test": "tab:orange"}

    def series(window, metric):
        return np.asarray(
            [
                flatten(payload["results"][str(epoch)]["windows"][window]["summary"])[metric]
                for epoch in epochs
            ],
            dtype=float,
        )

    fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    panels = [
        ("spearman", r"$\rho_S(n_0,\hat O/O)$", A1_RHO),
        ("high_n0_median", r"high-$n_0$ median $\hat O/O$", A1_HIGH),
        ("NRMSE_Gamma", r"NRMSE$_\Gamma$", A2["NRMSE_Gamma"]),
        ("field_aggregate", "field aggregate", A2["field_aggregate"]),
    ]
    for ax, (metric, ylabel, threshold) in zip(axes.flat[:4], panels):
        for window in WINDOWS:
            ax.plot(epochs, series(window, metric), "o-", color=colors[window], label=window)
        ax.axhline(threshold, color="0.4", linestyle=":", linewidth=1, label="old gate reference")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    ax = axes.flat[4]
    for window in WINDOWS:
        ax.plot(epochs, series(window, "E_O_median"), "o-", color=colors[window], label=f"{window} median")
        ax.plot(epochs, series(window, "E_O_worst"), "o--", color=colors[window], alpha=0.65, label=f"{window} worst")
    ax.set_ylabel(r"organisation error $E_O$")
    ax.grid(alpha=0.25)
    ax = axes.flat[5]
    val_loss = np.asarray([payload["logged_val_loss"][str(epoch)] for epoch in epochs])
    ax.semilogy(epochs, val_loss, "o-", color="tab:green")
    ax.set_ylabel("logged validation loss")
    ax.grid(alpha=0.25)
    for ax in axes.flat:
        ax.set_xlabel("epoch")
        ax.set_xticks(display_ticks, display_labels)
        ax.tick_params(axis="x", labelrotation=45)
    axes.flat[0].legend(fontsize=8)
    axes.flat[4].legend(fontsize=7, ncol=2)
    fig.suptitle("Reduced cell A: exploratory checkpoint trajectory (source conditions only)")
    fig.savefig(OVERVIEW_PLOT, dpi=180)
    plt.close(fig)

    condition_keys = list(
        payload["results"][str(epochs[0])]["windows"]["validation"]["per_condition"]
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, constrained_layout=True)
    for row_index, window in enumerate(WINDOWS):
        for key in condition_keys:
            ratios = [
                payload["results"][str(epoch)]["windows"][window]["per_condition"][key]["O_ratio"]
                for epoch in epochs
            ]
            errors = [
                payload["results"][str(epoch)]["windows"][window]["per_condition"][key]["E_O"]
                for epoch in epochs
            ]
            axes[row_index, 0].plot(epochs, ratios, "o-", label=key)
            axes[row_index, 1].plot(epochs, errors, "o-", label=key)
        axes[row_index, 0].axhline(1.0, color="0.4", linestyle=":", linewidth=1)
        axes[row_index, 0].set_ylabel(f"{window}\n" + r"$\hat O/O$")
        axes[row_index, 1].set_ylabel(f"{window}\n" + r"$E_O$")
        for ax in axes[row_index]:
            ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("epoch")
        ax.set_xticks(display_ticks, display_labels)
        ax.tick_params(axis="x", labelrotation=45)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Condition-level redistribution of organisation error")
    fig.savefig(CONDITION_PLOT, dpi=180)
    plt.close(fig)


def main():
    if "--replay-legacy" not in sys.argv:
        raise SystemExit("Historical sign-insensitive metrics only. Use evaluate_radaz_corrected_A.py, or --replay-legacy.")
    check_amendment()
    protocol_sha = verify_protocol()
    for path in CHECKPOINTS.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    for prior in (BEST_LAST_TEST, BEST_LAST_VALIDATION):
        content = json.loads(prior.read_text(encoding="utf-8"))
        if content.get("B_C_authorized"):
            raise RuntimeError(f"Prior result unexpectedly authorized B/C: {prior}")

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
    for expected_epoch, path in CHECKPOINTS.items():
        model, loaded_epoch = build_model(CELL_A["config"], path, device)
        if int(loaded_epoch) != expected_epoch:
            raise RuntimeError(
                f"Checkpoint epoch mismatch for {path}: expected {expected_epoch}, got {loaded_epoch}"
            )
        print(f"\n[checkpoint] epoch={expected_epoch}: {path.name}")
        windows = {}
        for window, (lo, hi) in WINDOWS.items():
            rows = evaluate_rows(
                model, manifest, low, high, registration, loss_module, device, lo, hi
            )
            summary = summarize(rows)
            windows[window] = {"per_condition": rows, "summary": summary}
            compact = flatten(summary)
            print(
                f"  {window:11s} rho={compact['spearman']:+.4f} "
                f"high={compact['high_n0_median']:.4f} E_O={compact['E_O_median']:.4f} "
                f"worst={compact['E_O_worst']:.4f} field={compact['field_aggregate']:.5f} "
                f"NRMSE_G={compact['NRMSE_Gamma']:.4f}"
            )
        results[str(expected_epoch)] = {"checkpoint": str(path), "windows": windows}
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "status": "post_primary_exploratory_checkpoint_trajectory",
        "protocol_sha256": protocol_sha,
        "checkpoint_epochs": list(CHECKPOINTS),
        "data_scope": {
            "conditions": "six source conditions only; no holdout",
            "validation_frames": [1600, 1799],
            "source_test_frames": [1800, 1999],
            "sampling": "ten non-overlapping direct10 samples per condition and window",
        },
        "logged_val_loss": logged_val_losses(),
        "results": results,
        "largest_adjacent_changes": largest_steps(results),
        "primary_STOP_immutable": True,
        "B_C_authorized": False,
        "grokking_claim_authorized": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    make_plots(payload)
    print(f"\n[written] {OUTPUT}")
    print(f"[written] {OVERVIEW_PLOT}")
    print(f"[written] {CONDITION_PLOT}")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        make_plots(json.loads(OUTPUT.read_text(encoding="utf-8")))
        print(f"[rewritten] {OVERVIEW_PLOT}")
        print(f"[rewritten] {CONDITION_PLOT}")
    else:
        main()
