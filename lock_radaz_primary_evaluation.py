"""Freeze the confirmatory E25 -> E22.5 RadAz evaluation protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT.parent
DEFAULT_OUTPUT = ROOT / "workdirs" / "radaz_primary_e25_to_e22p5_evaluation_lock.json"
E25_CASE = "2D_RadAz_Xe1p_Bx20mT_Ez25kVm_dt15ps_out15ns"
NORMALIZATION = (
    RESEARCH
    / "PEPAPIC"
    / "test"
    / "results"
    / "2D_Landmark"
    / E25_CASE
    / E25_CASE
    / "SimVPv2_inputs"
    / "radaz_3ch_targetnorm_trainfixed_margin20_native257x256_pad260x256.h5"
)
ARTIFACTS = {
    "amplitude_checkpoint": ROOT
    / "workdirs"
    / "train_radaz_state_history_conditioned_rom_bidirectional_noaug"
    / "state_history_conditioned_data_only.pt",
    "amplitude_lock": ROOT
    / "workdirs"
    / "train_radaz_state_history_conditioned_rom_bidirectional_noaug"
    / "model_lock.json",
    "amplitude_representation": ROOT
    / "workdirs"
    / "train_radaz_state_history_conditioned_rom_bidirectional_noaug"
    / "representation.h5",
    "phase_checkpoint": ROOT
    / "workdirs"
    / "train_radaz_mode_separated_controlled_carrier_rom"
    / "mode_separated_controlled_carrier_data_only.pt",
    "phase_lock": ROOT
    / "workdirs"
    / "train_radaz_mode_separated_controlled_carrier_rom"
    / "model_lock.json",
    "phase_representation": ROOT
    / "workdirs"
    / "train_radaz_mode_separated_controlled_carrier_rom"
    / "representation.h5",
    "factorized_lock": ROOT
    / "workdirs"
    / "build_radaz_state_phase_factorized_rom_bidirectional"
    / "model_lock.json",
    "physics_decoder_lock": ROOT
    / "workdirs"
    / "evaluate_radaz_factorized_physics_decoder_bidirectional"
    / "model_lock.json",
    "normalization": NORMALIZATION,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    missing = [str(path) for path in ARTIFACTS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing locked artifacts: {missing}")
    primary_candidates = sorted(
        str(path)
        for path in (
            RESEARCH / "PEPAPIC" / "test" / "results" / "2D_Landmark"
        ).glob("*restartFromEz25*")
    )
    if primary_candidates:
        raise RuntimeError(
            "Primary data are already present; do not create a post-data protocol lock: "
            + repr(primary_candidates)
        )
    result = {
        "status": "LOCKED_WAITING_FOR_PRIMARY_DATA",
        "created_before_primary_data_available": True,
        "primary_case": {
            "direction": "E25_to_E22p5",
            "source_Ez_kVm": 25.0,
            "target_Ez_kVm": 22.5,
            "step_time_us": 30.0,
            "expected_sampling_ns": 15.0,
            "expected_output_interval_us": [30.015, 34.995],
        },
        "frozen_artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in ARTIFACTS.items()
        },
        "preprocessing": {
            "normalization_refit": False,
            "spatial_stride": 1,
            "model_shape": [260, 256],
            "latent_checkpoint_refit": False,
            "radial_bands": 8,
            "maximum_fourier_mode": 21,
            "normalization_clipping_must_be_reported": True,
        },
        "forecast_protocol": {
            "context_steps": 40,
            "context_duration_us": 0.6,
            "context_role": "causal state initialization only",
            "free_rollout_starts_after_context": True,
            "future_primary_PIC_state_used_as_input": False,
            "phase_branch": "locked mode-separated recurrent branch",
            "amplitude_branch": "locked bidirectional state/E-history GRU",
            "fusion": "n2/n7 amplitude ratios rescale all four phase-branch fields",
            "physics_decoder": "truth-floor proximal field-gradient decoder",
            "lambda_E": 1.0,
        },
        "reported_horizons_us": [0.15, 0.30, 0.60, 1.20, 3.00, "full_available"],
        "predeclared_primary_gates": {
            "finite_fraction_equals_one": True,
            "selected_phi_skill_vs_raw_persistence_positive": True,
            "selected_Ey_skill_vs_raw_persistence_positive": True,
            "selected_phi_NRMSE_below_one": True,
            "selected_Ey_NRMSE_below_one": True,
            "n2_amplitude_skill_vs_raw_persistence_positive": True,
            "n7_amplitude_skill_vs_raw_persistence_positive": True,
            "physics_excess_hinge_reduced_by_decoder": True,
        },
        "confirmatory_policy": {
            "no_architecture_change_after_opening_primary": True,
            "no_checkpoint_or_epoch_reselection": True,
            "no_loss_or_decoder_weight_reselection": True,
            "no_normalization_or_representation_refit_on_primary": True,
            "failed_gate_is_reported_as_failure_not_retuned": True,
        },
        "hysteresis_report": {
            "compare_against": "allowed E20_to_E22p5 path at equal target Ez and elapsed time",
            "observables": [
                "n2 amplitude",
                "n7 amplitude",
                "ECDI n9-21 power",
                "selected phi RMS",
                "selected Ey RMS",
                "field-gradient residual",
            ],
            "interpretation": "history dependence, not by itself proof of bifurcation",
        },
        "primary_data_read": False,
        "primary_candidates_present_at_lock": primary_candidates,
        "script": str(Path(__file__).resolve()),
    }
    result["script_sha256"] = sha256(Path(__file__).resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
