"""Per-input-normalization re-derivation of the cell-A checkpoint trajectory.

Registered as a re-verification task in ARCH_A_CHECKPOINT_TRAJECTORY_BN_CAVEAT.md:
same nine checkpoints, same windows, same (legacy, sign-insensitive) metric set as
the 2026-09-08 stored-BN trajectory -- only the translator inference convention
changes (stored BatchNorm buffers -> per-input batch statistics, which at the
batch-size-1 training of this run is the training-time normalization).

This is a NEW exploratory analysis. It does not modify or re-grade the frozen
2026-09-07 Gate A verdict; the A1/A2 "descriptive" flags it prints show what the
frozen limits would have said under the corrected convention, for the forward
record only. CPU-only by design: it must not touch the GPU while the paired
pilot queue is paused for the user.
"""
from __future__ import annotations

import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # never touch the GPU (user-reserved)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

import evaluate_radaz_arch_A_checkpoint_trajectory as traj
from evaluate_radaz_arch_gate_A import CELL_A, check_amendment
from evaluate_radaz_conditioned_factorial import MANIFEST, build_model
from evaluate_radaz_physics_loss_v2 import REG_DIR, load_registration
from openstl.methods.pepapic_spectral_loss import PEPAPICSpectralLoss
from openstl.models.simvp_factory import configure_translator_norm
from audit_radaz_v2_transport_error_geometry import MAXN

OUTPUT = REG_DIR / "arch_A_checkpoint_trajectory_input_norm.json"


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"Refusing overwrite: {OUTPUT}")
    torch.set_num_threads(int(os.environ.get("REAUDIT_TORCH_THREADS", "4")))
    check_amendment()
    protocol_sha = traj.verify_protocol()
    legacy_path = REG_DIR / "arch_A_checkpoint_trajectory.json"
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    low = np.asarray(manifest["normalization"]["low"], dtype=np.float64)
    high = np.asarray(manifest["normalization"]["high"], dtype=np.float64)
    registration = load_registration()
    device = torch.device("cpu")
    loss_module = PEPAPICSpectralLoss(
        data_root=str(MANIFEST), max_mode=MAXN, radial_bands=4,
        radial_min_m=0.09e-2, radial_max_m=1.19e-2,
        coordinate_system="integer_power_cross",
    ).to(device)

    results = {}
    for expected_epoch, path in traj.CHECKPOINTS.items():
        model, loaded_epoch = build_model(CELL_A["config"], path, device)
        if int(loaded_epoch) != expected_epoch:
            raise RuntimeError(f"Checkpoint epoch mismatch: {path} -> {loaded_epoch}")
        configure_translator_norm(model, "batch_input")
        model.eval()
        print(f"\n[checkpoint per-input] epoch={expected_epoch}: {path.name}", flush=True)
        windows = {}
        for window, (lo, hi) in traj.WINDOWS.items():
            with torch.no_grad():
                rows = traj.evaluate_rows(model, manifest, low, high, registration,
                                          loss_module, device, lo, hi)
            summary = traj.summarize(rows)
            windows[window] = {"per_condition": rows, "summary": summary}
            compact = traj.flatten(summary)
            print(f"  {window:11s} rho={compact['spearman']:+.4f} "
                  f"high={compact['high_n0_median']:.4f} E_O={compact['E_O_median']:.4f} "
                  f"field={compact['field_aggregate']:.5f} NRMSE_G={compact['NRMSE_Gamma']:.4f} "
                  f"A1={windows[window]['summary']['A1_descriptive']['pass']} "
                  f"A2={windows[window]['summary']['A2_descriptive']['pass']}", flush=True)
        results[str(expected_epoch)] = {"checkpoint": str(path), "windows": windows}
        del model

    payload = {
        "status": "exploratory_per_input_norm_trajectory",
        "inference_convention": "translator batch_input (per-input statistics); "
                                "legacy run used stored BatchNorm buffers",
        "metric_set": "identical legacy sign-insensitive metrics; E_O sign caveat applies",
        "protocol_sha256": protocol_sha,
        "legacy_trajectory": str(legacy_path),
        "checkpoint_epochs": list(traj.CHECKPOINTS),
        "logged_val_loss": traj.logged_val_losses(),
        "results": results,
        "largest_adjacent_changes": traj.largest_steps(results),
        "frozen_2026_09_07_verdict_unchanged": True,
        "B_C_authorized": False,
        "note": "A1/A2 flags are descriptive re-reads of the frozen limits under the "
                "corrected convention; they do not retroactively re-grade the STOP.",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[written] {OUTPUT}", flush=True)

    for window in traj.WINDOWS:
        print(f"\n=== {window}: stored-BN vs per-input (epoch, NRMSE_G, rho, A1, A2) ===", flush=True)
        for epoch in traj.CHECKPOINTS:
            old = traj.flatten(legacy["results"][str(epoch)]["windows"][window]["summary"])
            new = traj.flatten(results[str(epoch)]["windows"][window]["summary"])
            olds = legacy["results"][str(epoch)]["windows"][window]["summary"]
            news = results[str(epoch)]["windows"][window]["summary"]
            print(f"  ep{epoch:>2}: NRMSE_G {old['NRMSE_Gamma']:.4f}->{new['NRMSE_Gamma']:.4f}  "
                  f"rho {old['spearman']:+.4f}->{new['spearman']:+.4f}  "
                  f"A1 {olds['A1_descriptive']['pass']}->{news['A1_descriptive']['pass']}  "
                  f"A2 {olds['A2_descriptive']['pass']}->{news['A2_descriptive']['pass']}", flush=True)


if __name__ == "__main__":
    main()
