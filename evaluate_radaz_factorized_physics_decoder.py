"""Truth-floor spectral physics decoder for the factorized RadAz ROM.

The decoder is the proximal output layer associated with a quadratic
field-gradient physics loss.  It holds phi fixed and shrinks only the part of
the predicted residual E_y + d(phi)/dy that exceeds the residual floor found
in allowed PIC training data.  No development or primary PIC truth is used by
the correction itself.  Predeclared lambda_E candidates are compared on the
same up-transition development split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

import analyze_radaz_augmented_physical_state_dynamics as augmented
import train_radaz_carrier_envelope_high_branch_rom as carrier
import train_radaz_direct_physical_state_rom as direct
import train_radaz_regime_aware_transition_rom as stage2
import train_radaz_state_history_physics_rom as physics


ROOT = Path(__file__).resolve().parent
DEFAULT_FACTOR = (
    ROOT / "workdirs" / "build_radaz_state_phase_factorized_rom"
)
DEFAULT_OUTPUT = (
    ROOT / "workdirs" / "evaluate_radaz_factorized_physics_decoder"
)


def json_safe(value):
    return carrier.json_safe(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for piece in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([json_safe(row) for row in rows])


def field_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    persistence: np.ndarray,
    modes: np.ndarray,
    statistics: physics.PhysicsStatistics,
) -> dict:
    ey_metrics = augmented.scalar_metrics(
        truth[:, direct.FIELD_NAMES.index("efy")],
        prediction[:, direct.FIELD_NAMES.index("efy")],
        persistence[:, direct.FIELD_NAMES.index("efy")],
    )
    wave_numbers = 2.0 * np.pi * modes / direct.AZIMUTHAL_LENGTH_M
    residual = (
        prediction[:, direct.FIELD_NAMES.index("efy")]
        + 1j
        * wave_numbers[None]
        * prediction[:, direct.FIELD_NAMES.index("phi")]
    )
    power = np.abs(residual / statistics.residual_scale[None]) ** 2
    return {
        "selected_efy_skill_vs_raw_persistence": ey_metrics[
            "skill_vs_persistence"
        ],
        "selected_efy_nrmse": ey_metrics["nrmse"],
        "selected_efy_temporal_anomaly_correlation": ey_metrics[
            "temporal_anomaly_correlation"
        ],
        "field_gradient_residual_over_ey_rms": float(
            np.sqrt(np.mean(np.abs(residual) ** 2))
            / np.sqrt(
                np.mean(
                    np.abs(
                        prediction[:, direct.FIELD_NAMES.index("efy")]
                    )
                    ** 2
                )
            )
        ),
        "field_gradient_normalized_rms": float(np.sqrt(np.mean(power))),
        "field_gradient_excess_hinge": float(
            np.mean(
                np.maximum(
                    power - statistics.truth_floor_power[None], 0.0
                )
            )
        ),
        "field_gradient_truth_floor_rms": float(
            np.sqrt(np.mean(statistics.truth_floor_power))
        ),
    }


def apply_decoder(
    prediction: np.ndarray,
    modes: np.ndarray,
    statistics: physics.PhysicsStatistics,
    lambda_e: float,
) -> np.ndarray:
    result = prediction.copy()
    phi_index = direct.FIELD_NAMES.index("phi")
    ey_index = direct.FIELD_NAMES.index("efy")
    wave_numbers = 2.0 * np.pi * modes / direct.AZIMUTHAL_LENGTH_M
    residual = result[:, ey_index] + 1j * wave_numbers[None] * result[:, phi_index]
    floor_amplitude = (
        np.sqrt(statistics.truth_floor_power) * statistics.residual_scale
    )
    target_residual = residual * np.minimum(
        1.0,
        floor_amplitude[None]
        / np.maximum(np.abs(residual), np.finfo(float).tiny),
    )
    proximal_residual = (
        residual + float(lambda_e) * target_residual
    ) / (1.0 + float(lambda_e))
    result[:, ey_index] += proximal_residual - residual
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor", type=Path, default=DEFAULT_FACTOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lambda-e", default="0.01,0.1,1.0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    factor = args.factor.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    factor_lock_path = factor / "model_lock.json"
    factor_rollout_path = factor / "development_rollout_35to40us.h5"
    factor_lock = json.loads(
        factor_lock_path.read_text(encoding="utf-8")
    )
    if not bool(factor_lock.get("accepted_for_physics_ablation", False)):
        raise ValueError("Factorized data-only ROM did not pass its lock gate")

    trajectories = {
        "e25_stationary": stage2.load_trajectory(
            "e25_stationary",
            25.0,
            stage2.DEFAULT_E25_FEATURES,
            stage2.DEFAULT_E25_PHYSICAL,
        ),
        "e20_to_e22p5": stage2.load_trajectory(
            "e20_to_e22p5",
            22.5,
            stage2.DEFAULT_UP_FEATURES,
            stage2.DEFAULT_UP_PHYSICAL,
        ),
    }
    e25 = trajectories["e25_stationary"]
    up = trajectories["e20_to_e22p5"]
    fit_masks = {
        "e25_stationary": stage2.interval_mask(e25.time_us, 12.0, 24.0),
        "e20_to_e22p5": up.time_us < 35.0 - 1.0e-10,
    }
    representation = carrier.fit_representation(trajectories, fit_masks)
    statistics = physics.physics_statistics(representation, fit_masks)
    (output / "truth_field_gradient_audit.json").write_text(
        json.dumps(json_safe(statistics.audit), indent=2), encoding="utf-8"
    )

    with h5py.File(factor_rollout_path, "r") as handle:
        time_us = np.asarray(handle["time_us"], dtype=np.float64)
        frame = np.asarray(handle["frame"], dtype=np.int64)
        modes = np.asarray(handle["selected_mode_numbers"], dtype=np.int64)
        truth = np.asarray(handle["truth_physical_coefficients"])
        data_only_prediction = np.asarray(
            handle["prediction_physical_coefficients"]
        )
        persistence = np.asarray(
            handle["raw_persistence_physical_coefficients"]
        )

    base_metrics = factor_lock["development_metrics"]
    baseline_physics = field_metrics(
        truth, data_only_prediction, persistence, modes, statistics
    )
    baseline = {**base_metrics, **baseline_physics}
    rows = [{"model": "data_only", "lambda_E": 0.0, **baseline}]
    candidates = []
    baseline_hinge = baseline["field_gradient_excess_hinge"]
    for lambda_e in [float(value) for value in args.lambda_e.split(",")]:
        prediction = apply_decoder(
            data_only_prediction, modes, statistics, lambda_e
        )
        metrics = {
            **base_metrics,
            **field_metrics(truth, prediction, persistence, modes, statistics),
        }
        metrics["passes_field_climatology_gate"] = bool(
            metrics["selected_phi_nrmse"] < 1.0
            and metrics["selected_efy_nrmse"] < 1.0
        )
        reduction = 1.0 - (
            metrics["field_gradient_excess_hinge"]
            / max(baseline_hinge, 1.0e-30)
        )
        accepted = bool(
            metrics["finite_fraction"] == 1.0
            and metrics["passes_all_persistence_gates"]
            and metrics["passes_field_climatology_gate"]
            and reduction > 0.0
        )
        row = {
            "model": "truth_floor_physics_decoder",
            "lambda_E": lambda_e,
            "physics_excess_reduction": reduction,
            "accepted": accepted,
            **metrics,
        }
        rows.append(row)
        candidates.append((reduction, accepted, lambda_e, prediction, metrics))

    eligible = [candidate for candidate in candidates if candidate[1]]
    selected = max(eligible, key=lambda item: item[0]) if eligible else None
    status = (
        "PHYSICS_DECODER_ACCEPTED"
        if selected is not None
        else "NO_PHYSICS_DECODER_ACCEPTED"
    )
    write_csv(output / "physics_comparison.csv", rows)
    (output / "physics_comparison.json").write_text(
        json.dumps(json_safe(rows), indent=2), encoding="utf-8"
    )

    selected_summary = None
    if selected is not None:
        reduction, _, lambda_e, prediction, metrics = selected
        selected_summary = {
            "lambda_E": lambda_e,
            "physics_excess_reduction": reduction,
            "metrics": metrics,
        }
        with h5py.File(output / "selected_development_rollout.h5", "w") as handle:
            handle.attrs["primary_E25_to_E22p5_loaded"] = False
            handle.attrs["future_PIC_truth_used_in_decoder"] = False
            handle.attrs["lambda_E"] = lambda_e
            handle.create_dataset("time_us", data=time_us)
            handle.create_dataset("frame", data=frame)
            handle.create_dataset("selected_mode_numbers", data=modes)
            handle.create_dataset("truth_physical_coefficients", data=truth, compression="gzip")
            handle.create_dataset("data_only_prediction", data=data_only_prediction, compression="gzip")
            handle.create_dataset("physics_prediction", data=prediction, compression="gzip")
            handle.create_dataset("raw_persistence_physical_coefficients", data=persistence, compression="gzip")
            handle.create_dataset("residual_scale", data=statistics.residual_scale)
            handle.create_dataset("truth_floor_power", data=statistics.truth_floor_power)

    lock = {
        "status": status,
        "selected": selected_summary,
        "baseline_metrics": baseline,
        "candidate_rows": rows,
        "physics_layer": {
            "type": "truth_floor_proximal_spectral_decoder",
            "residual": "Ey + i*k*phi",
            "phi_held_fixed": True,
            "only_excess_above_training_truth_floor_shrunk": True,
            "equivalent_to_output_level_quadratic_physics_loss": True,
            "end_to_end_network_weights_retrained": False,
            "future_PIC_truth_used": False,
        },
        "truth_residual_audit": statistics.audit,
        "factor_model_lock": str(factor_lock_path),
        "factor_model_lock_sha256": sha256(factor_lock_path),
        "factor_rollout": str(factor_rollout_path),
        "factor_rollout_sha256": sha256(factor_rollout_path),
        "primary_test": {
            "direction": "E25_to_E22.5",
            "data_loaded": False,
            "used_for_selection": False,
        },
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "arguments": vars(args),
    }
    (output / "model_lock.json").write_text(
        json.dumps(json_safe(lock), indent=2), encoding="utf-8"
    )
    if selected_summary is None:
        summary = "No physics decoder passed the preservation gate."
    else:
        selected_metrics = selected_summary["metrics"]
        summary = (
            f"Selected lambda_E={selected_summary['lambda_E']:g}; "
            f"physics excess reduction={selected_summary['physics_excess_reduction']:.3%}; "
            f"Ey NRMSE={selected_metrics['selected_efy_nrmse']:.6f}."
        )
    (output / "README.md").write_text(
        f"# Factorized RadAz physics decoder\n\nStatus: `{status}`.\n\n{summary}\n",
        encoding="utf-8",
    )
    print(json.dumps(json_safe(lock), indent=2), flush=True)


if __name__ == "__main__":
    main()
