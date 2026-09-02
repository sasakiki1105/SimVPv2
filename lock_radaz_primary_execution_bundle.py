"""Freeze the executable bundle used for the blind E25 -> E22.5 test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT.parent
OUTPUT = ROOT / "workdirs" / "radaz_primary_execution_bundle_lock.json"
PRIMARY_INPUT = ROOT / "workdirs" / "radaz_e25_to_e22p5_primary"
PRIMARY_RESULT = (
    ROOT / "workdirs" / "evaluate_radaz_primary_e25_to_e22p5" / "primary_evaluation.json"
)
LATENT_WORKDIR = (
    ROOT
    / "workdirs"
    / "radaz_xe1p_bx20mt_ez10kvm_out15ns_native257x256_"
    "direct10_trainfixed_disjoint_811_bs1_100ep"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Execution bundle is already locked: {OUTPUT}")
    forbidden = [
        PRIMARY_INPUT / "fourier_latent_features.h5",
        PRIMARY_INPUT / "physical_fourier_targets.h5",
        PRIMARY_RESULT,
    ]
    present = [str(path) for path in forbidden if path.exists()]
    if present:
        raise RuntimeError(
            "Refusing a pre-primary lock after primary inputs/results exist: "
            f"{present}"
        )
    files = {
        "evaluation_protocol_lock": ROOT / "workdirs" / "radaz_primary_e25_to_e22p5_evaluation_lock.json",
        "primary_evaluator": ROOT / "evaluate_radaz_primary_e25_to_e22p5.py",
        "primary_intake_runner": ROOT / "prepare_and_evaluate_radaz_primary_e25_to_e22p5.py",
        "single_case_preprocessor": ROOT / "build_radaz_single_case_h5.py",
        "latent_extractor": ROOT / "analyze_radaz_fourier_latent_dynamics.py",
        "physical_extractor": ROOT / "analyze_radaz_fourier_latent_to_physical_modes.py",
        "raw_health_auditor": RESEARCH / "PEPAPIC" / "check_radial_azimuthal_h5_health.py",
        "rank_consolidator": RESEARCH / "PEPAPIC" / "consolidate_radial_azimuthal_case.py",
        "consolidated_validator": RESEARCH / "PEPAPIC" / "validate_consolidated_radial_azimuthal.py",
        "latent_config": ROOT / "configs" / "custom" / "pepapic" / "SimVP_gSTA_radaz_direct.py",
        "latent_checkpoint": LATENT_WORKDIR / "checkpoints" / "best.ckpt",
        "allowed_e25_latent_features": ROOT / "workdirs" / "analyze_radaz_bx20mt_ez25kvm_fourier_latent_raw" / "fourier_latent_features.h5",
        "allowed_e25_physical_targets": ROOT / "workdirs" / "analyze_radaz_bx20mt_ez25kvm_fourier_latent_to_physical_modes" / "physical_fourier_targets.h5",
        "allowed_up_latent_features": ROOT / "workdirs" / "radaz_e20_to_e22p5_transition" / "fourier_latent_features_e25targetnorm.h5",
        "allowed_up_physical_targets": ROOT / "workdirs" / "radaz_e20_to_e22p5_transition" / "physical_fourier_targets.h5",
        "allowed_down_latent_features": ROOT / "workdirs" / "radaz_e22p5_to_e20_transition" / "fourier_latent_features.h5",
        "allowed_down_physical_targets": ROOT / "workdirs" / "radaz_e22p5_to_e20_transition" / "physical_fourier_targets.h5",
        "augmented_physical": ROOT / "analyze_radaz_augmented_physical_state_dynamics.py",
        "blockwise_latent": ROOT / "analyze_radaz_blockwise_fourier_latent_dynamics.py",
        "factorized_builder": ROOT / "build_radaz_state_phase_factorized_rom.py",
        "physics_decoder": ROOT / "evaluate_radaz_factorized_physics_decoder.py",
        "carrier_high_branch": ROOT / "train_radaz_carrier_envelope_high_branch_rom.py",
        "carrier_time_control": ROOT / "train_radaz_carrier_envelope_time_controlled_rom.py",
        "direct_physical_state": ROOT / "train_radaz_direct_physical_state_rom.py",
        "mode_separated_carrier": ROOT / "train_radaz_mode_separated_controlled_carrier_rom.py",
        "modulation_carrier": ROOT / "train_radaz_modulation_controlled_carrier_rom.py",
        "regime_transition": ROOT / "train_radaz_regime_aware_transition_rom.py",
        "state_history": ROOT / "train_radaz_state_history_conditioned_rom.py",
        "state_history_physics": ROOT / "train_radaz_state_history_physics_rom.py",
    }
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot lock missing execution artifacts: {missing}")
    manifest = {
        "status": "LOCKED_BEFORE_PRIMARY_INPUTS",
        "primary_inputs_present_at_lock": False,
        "primary_result_present_at_lock": False,
        "purpose": (
            "Detect any executable, preprocessing, latent-checkpoint, or "
            "evaluation-protocol drift before the one-time primary run."
        ),
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in files.items()
        },
        "policy": {
            "verify_all_hashes_before_primary_run": True,
            "no_source_or_checkpoint_change_after_primary_opened": True,
            "failed_primary_gate_is_not_repaired_by_reselection": True,
        },
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
