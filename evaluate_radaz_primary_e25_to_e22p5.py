"""Run the frozen confirmatory RadAz E25 -> E22.5 evaluation.

The script verifies the pre-data evaluation lock before loading a primary
trajectory.  Forty causal primary frames initialize the two recurrent
branches; every later state is a free rollout.  No fit or model selection is
performed on the primary path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch

import analyze_radaz_augmented_physical_state_dynamics as augmented
import analyze_radaz_blockwise_fourier_latent_dynamics as block
import build_radaz_state_phase_factorized_rom as factorized
import evaluate_radaz_factorized_physics_decoder as decoder
import train_radaz_carrier_envelope_high_branch_rom as carrier
import train_radaz_carrier_envelope_time_controlled_rom as controlled
import train_radaz_direct_physical_state_rom as direct
import train_radaz_mode_separated_controlled_carrier_rom as separated
import train_radaz_modulation_controlled_carrier_rom as modulation
import train_radaz_regime_aware_transition_rom as stage2
import train_radaz_state_history_conditioned_rom as history
import train_radaz_state_history_physics_rom as physics


ROOT = Path(__file__).resolve().parent
DEFAULT_LOCK = ROOT / "workdirs" / "radaz_primary_e25_to_e22p5_evaluation_lock.json"
DEFAULT_OUTPUT = ROOT / "workdirs" / "evaluate_radaz_primary_e25_to_e22p5"
DEFAULT_PRIMARY_FEATURES = (
    ROOT
    / "workdirs"
    / "radaz_e25_to_e22p5_primary"
    / "fourier_latent_features.h5"
)
DEFAULT_PRIMARY_PHYSICAL = (
    ROOT
    / "workdirs"
    / "radaz_e25_to_e22p5_primary"
    / "physical_fourier_targets.h5"
)
CONTEXT_STEPS = 40
HORIZONS_US = (0.15, 0.30, 0.60, 1.20, 3.00)


def json_safe(value):
    return carrier.json_safe(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block_now in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block_now)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([json_safe(row) for row in rows])


def verify_lock(path: Path) -> dict:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock["status"] != "LOCKED_WAITING_FOR_PRIMARY_DATA":
        raise ValueError("Unexpected primary evaluation lock status")
    if not lock["created_before_primary_data_available"]:
        raise ValueError("Evaluation protocol was not locked before data availability")
    mismatches = []
    for name, item in lock["frozen_artifacts"].items():
        artifact = Path(item["path"])
        if not artifact.is_file() or sha256(artifact) != item["sha256"]:
            mismatches.append(name)
    if mismatches:
        raise ValueError(f"Frozen artifact hash mismatch: {mismatches}")
    if int(lock["forecast_protocol"]["context_steps"]) != CONTEXT_STEPS:
        raise ValueError("Context length differs from the frozen protocol")
    if float(lock["forecast_protocol"]["lambda_E"]) != 1.0:
        raise ValueError("Decoder strength differs from the frozen protocol")
    return lock


def allowed_trajectories() -> dict[str, stage2.Trajectory]:
    down_root = ROOT / "workdirs" / "radaz_e22p5_to_e20_transition"
    return {
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
        "e22p5_to_e20": stage2.load_trajectory(
            "e22p5_to_e20",
            20.0,
            down_root / "fourier_latent_features.h5",
            down_root / "physical_fourier_targets.h5",
        ),
    }


def fit_locked_representation(
    trajectories: dict[str, stage2.Trajectory],
    bidirectional: bool,
) -> tuple[carrier.CarrierRepresentation, direct.DirectRepresentation, dict[str, np.ndarray]]:
    selected = {
        "e25_stationary": trajectories["e25_stationary"],
        "e20_to_e22p5": trajectories["e20_to_e22p5"],
    }
    if bidirectional:
        selected["e22p5_to_e20"] = trajectories["e22p5_to_e20"]
    masks = {
        "e25_stationary": stage2.interval_mask(
            selected["e25_stationary"].time_us, 12.0, 24.0
        ),
        "e20_to_e22p5": selected["e20_to_e22p5"].time_us < 35.0 - 1.0e-10,
    }
    if bidirectional:
        masks["e22p5_to_e20"] = (
            selected["e22p5_to_e20"].time_us < 34.86 - 1.0e-10
        )
    representation = carrier.fit_representation(selected, masks)
    direct_representation = direct.fit_direct_representation(selected, masks)
    return representation, direct_representation, masks


def verify_representation(
    representation: carrier.CarrierRepresentation,
    path: Path,
) -> None:
    with h5py.File(path, "r") as handle:
        if not np.array_equal(
            np.asarray(handle["selected_mode_numbers"], dtype=np.int64),
            carrier.SELECTED_MODE_NUMBERS,
        ):
            raise ValueError("Locked selected modes changed")
        for name in carrier.GROUPS:
            if not np.allclose(
                handle[f"scaler/{name}/mean"],
                representation.scaler.means[name],
                atol=1.0e-10,
                rtol=1.0e-10,
            ) or not np.allclose(
                handle[f"scaler/{name}/scale"],
                representation.scaler.scales[name],
                atol=1.0e-10,
                rtol=1.0e-10,
            ):
                raise ValueError(f"Locked scaler mismatch for {name}")
        for name, values in representation.carrier_step_rad.items():
            if not np.allclose(
                handle[f"carrier/{name}/phase_step_rad"],
                values,
                atol=1.0e-10,
                rtol=1.0e-10,
            ):
                raise ValueError(f"Locked carrier mismatch for {name}")


def transform_trajectory(
    trajectory: stage2.Trajectory,
    representation: carrier.CarrierRepresentation,
    direct_representation: direct.DirectRepresentation,
) -> np.ndarray:
    latent = np.concatenate(
        [
            direct_representation.block_models[name].transform(trajectory.features)
            for name in block.BLOCKS
        ],
        axis=1,
    )
    packed_physical, radial_weights = direct.load_coefficients(trajectory.physical_path)
    _, locked_radial_weights = direct.load_coefficients(stage2.DEFAULT_E25_PHYSICAL)
    if not np.allclose(radial_weights, locked_radial_weights):
        raise ValueError("Radial weights differ from the frozen E25 reference")
    physical = augmented.flatten_physical(trajectory.physical)
    radial_indices = np.asarray([1, 3, 5, 7], dtype=np.int64)
    cross_indices = np.arange(16).reshape(4, 2, 2)[:, 1, :].reshape(-1)
    raw_groups = {
        "latent_high": latent[:, 10:20],
        "carrier_physical": carrier.demodulate(
            packed_physical,
            representation.carrier_step_rad["e20_to_e22p5"],
        ),
        "radial_ecdi": physical["radial"][:, radial_indices],
        "cross_ecdi": physical["cross"][:, cross_indices],
    }
    return representation.scaler.transform(raw_groups)


@torch.no_grad()
def rollout_amplitude(
    states: np.ndarray,
    controls: np.ndarray,
    representation: carrier.CarrierRepresentation,
    checkpoint_path: Path,
    device: torch.device,
) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = history.StateHistoryConditionedROM(
        representation,
        carrier.STATE_DIMENSION,
        history.CONTROL_DIMENSION,
        int(checkpoint["hidden_dimension"]),
        float(checkpoint["delta_limit"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    state_history = torch.as_tensor(
        states[:CONTEXT_STEPS][None], dtype=torch.float32, device=device
    )
    control_history = torch.as_tensor(
        controls[:CONTEXT_STEPS][None], dtype=torch.float32, device=device
    )
    future_controls = torch.as_tensor(
        controls[CONTEXT_STEPS:][None], dtype=torch.float32, device=device
    )
    return (
        model.rollout(state_history, control_history, future_controls)[0]
        .cpu()
        .numpy()
        .astype(np.float64)
    )


@torch.no_grad()
def rollout_phase(
    states: np.ndarray,
    controls: np.ndarray,
    checkpoint_path: Path,
    device: torch.device,
) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = separated.ModeSeparatedControlledDelayROM(
        carrier.STATE_DIMENSION,
        separated.CONTROL_DIMENSION,
        int(checkpoint["hidden_dimension"]),
        float(checkpoint["delta_limit"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return controlled.rollout(
        model,
        states[:CONTEXT_STEPS],
        controls[:CONTEXT_STEPS],
        controls[CONTEXT_STEPS:],
        device,
    )


def truth_selected_coefficients(path: Path) -> np.ndarray:
    packed, _ = direct.load_coefficients(path)
    coefficients = direct.unpack_physical_fourier(packed)
    return coefficients[:, :, carrier.SELECTED_MODE_INDICES]


def forecast_trajectory(
    trajectory: stage2.Trajectory,
    source_ez_kvm: float,
    amplitude_representation: carrier.CarrierRepresentation,
    amplitude_direct: direct.DirectRepresentation,
    phase_representation: carrier.CarrierRepresentation,
    phase_direct: direct.DirectRepresentation,
    amplitude_checkpoint: Path,
    phase_checkpoint: Path,
    statistics: physics.PhysicsStatistics,
    device: torch.device,
) -> dict:
    if len(trajectory.time_us) <= CONTEXT_STEPS + 2:
        raise ValueError("Trajectory is too short for the locked context")
    amplitude_states = transform_trajectory(
        trajectory, amplitude_representation, amplitude_direct
    )
    phase_states = transform_trajectory(trajectory, phase_representation, phase_direct)
    amplitude_controls = history.history_controls_for(
        trajectory, 22.5, source_ez_kvm, True
    )
    phase_controls = controlled.controls_for(trajectory, 22.5, source_ez_kvm, True)
    amplitude_prediction = rollout_amplitude(
        amplitude_states,
        amplitude_controls,
        amplitude_representation,
        amplitude_checkpoint,
        device,
    )
    phase_prediction = rollout_phase(
        phase_states,
        phase_controls,
        phase_checkpoint,
        device,
    )
    indices = np.arange(CONTEXT_STEPS, len(trajectory.time_us), dtype=np.int64)
    amplitude_coefficients = modulation.decode_coefficients(
        amplitude_prediction, amplitude_representation, indices
    )
    phase_coefficients = modulation.decode_coefficients(
        phase_prediction, phase_representation, indices
    )
    modes = carrier.SELECTED_MODE_NUMBERS
    selected_indices = np.asarray(
        [int(np.flatnonzero(modes == mode)[0]) for mode in (2, 7)],
        dtype=np.int64,
    )
    ratio = np.abs(amplitude_coefficients[:, 0, selected_indices]) / np.maximum(
        np.abs(phase_coefficients[:, 0, selected_indices]), np.finfo(float).tiny
    )
    factorized_prediction = phase_coefficients.copy()
    factorized_prediction[:, :, selected_indices] *= ratio[:, None, :]
    decoded_prediction = decoder.apply_decoder(
        factorized_prediction, modes, statistics, lambda_e=1.0
    )
    truth_all = truth_selected_coefficients(trajectory.physical_path)
    truth = truth_all[indices]
    persistence = np.repeat(truth_all[CONTEXT_STEPS - 1 : CONTEXT_STEPS], len(indices), axis=0)
    return {
        "time_us": trajectory.time_us[indices],
        "frame": trajectory.frame[indices],
        "indices": indices,
        "modes": modes,
        "truth": truth,
        "persistence": persistence,
        "factorized_prediction": factorized_prediction,
        "decoded_prediction": decoded_prediction,
        "amplitude_truth_state": amplitude_states[indices],
        "amplitude_prediction_state": amplitude_prediction,
        "phase_truth_state": phase_states[indices],
        "phase_prediction_state": phase_prediction,
        "amplitude_ratio": ratio,
    }


def phase_frequency_mhz(values: np.ndarray, dt_us: float) -> float:
    products = values[1:] * np.conj(values[:-1])
    return float(np.angle(np.sum(products)) / (2.0 * np.pi * dt_us))


def evaluate_forecast(
    forecast: dict,
    statistics: physics.PhysicsStatistics,
) -> dict:
    truth = forecast["truth"]
    persistence = forecast["persistence"]
    factor_prediction = forecast["factorized_prediction"]
    prediction = forecast["decoded_prediction"]
    modes = forecast["modes"]
    row = {
        "finite_fraction": float(np.mean(np.isfinite(prediction))),
    }
    for field_index, field in ((0, "phi"), (3, "efy")):
        metrics = augmented.scalar_metrics(
            truth[:, field_index], prediction[:, field_index], persistence[:, field_index]
        )
        row[f"selected_{field}_skill_vs_raw_persistence"] = metrics["skill_vs_persistence"]
        row[f"selected_{field}_nrmse"] = metrics["nrmse"]
        row[f"selected_{field}_temporal_anomaly_correlation"] = metrics[
            "temporal_anomaly_correlation"
        ]
    dt_us = float(np.median(np.diff(forecast["time_us"])))
    for mode in (2, 7):
        index = int(np.flatnonzero(modes == mode)[0])
        metrics = augmented.scalar_metrics(
            np.abs(truth[:, 0, index]),
            np.abs(prediction[:, 0, index]),
            np.abs(persistence[:, 0, index]),
        )
        row[f"phi_n{mode}_amplitude_skill"] = metrics["skill_vs_persistence"]
        row[f"phi_n{mode}_amplitude_correlation"] = metrics["correlation"]
        truth_frequency = phase_frequency_mhz(truth[:, 0, index], dt_us)
        prediction_frequency = phase_frequency_mhz(prediction[:, 0, index], dt_us)
        row[f"phi_n{mode}_frequency_abs_error_MHz"] = abs(
            prediction_frequency - truth_frequency
        )
    amplitude_persistence = np.repeat(
        forecast["amplitude_truth_state"][:1],
        len(forecast["amplitude_truth_state"]),
        axis=0,
    )
    phase_persistence = np.repeat(
        forecast["phase_truth_state"][:1],
        len(forecast["phase_truth_state"]),
        axis=0,
    )
    row["amplitude_state_skill_vs_persistence"] = augmented.scalar_metrics(
        forecast["amplitude_truth_state"],
        forecast["amplitude_prediction_state"],
        amplitude_persistence,
    )["skill_vs_persistence"]
    row["phase_state_skill_vs_persistence"] = augmented.scalar_metrics(
        forecast["phase_truth_state"],
        forecast["phase_prediction_state"],
        phase_persistence,
    )["skill_vs_persistence"]
    baseline_physics = decoder.field_metrics(
        truth, factor_prediction, persistence, modes, statistics
    )
    decoded_physics = decoder.field_metrics(
        truth, prediction, persistence, modes, statistics
    )
    row.update(
        {
            "factorized_field_gradient_excess_hinge": baseline_physics[
                "field_gradient_excess_hinge"
            ],
            "decoded_field_gradient_excess_hinge": decoded_physics[
                "field_gradient_excess_hinge"
            ],
            "physics_excess_reduction": 1.0
            - decoded_physics["field_gradient_excess_hinge"]
            / max(baseline_physics["field_gradient_excess_hinge"], 1.0e-30),
            "decoded_field_gradient_residual_over_ey_rms": decoded_physics[
                "field_gradient_residual_over_ey_rms"
            ],
        }
    )
    gates = {
        "finite_fraction_equals_one": row["finite_fraction"] == 1.0,
        "selected_phi_skill_positive": row["selected_phi_skill_vs_raw_persistence"] > 0.0,
        "selected_efy_skill_positive": row["selected_efy_skill_vs_raw_persistence"] > 0.0,
        "selected_phi_nrmse_below_one": row["selected_phi_nrmse"] < 1.0,
        "selected_efy_nrmse_below_one": row["selected_efy_nrmse"] < 1.0,
        "n2_amplitude_skill_positive": row["phi_n2_amplitude_skill"] > 0.0,
        "n7_amplitude_skill_positive": row["phi_n7_amplitude_skill"] > 0.0,
        "physics_excess_reduced": row["physics_excess_reduction"] > 0.0,
    }
    row["gates"] = gates
    row["passes_all_predeclared_primary_gates"] = bool(all(gates.values()))
    return row


def horizon_rows(forecast: dict, statistics: physics.PhysicsStatistics) -> list[dict]:
    rows = []
    start = float(forecast["time_us"][0])
    available = float(forecast["time_us"][-1] - start)
    horizons = [value for value in HORIZONS_US if value <= available + 1.0e-10]
    horizons.append(available)
    for horizon in horizons:
        stop = int(
            np.searchsorted(forecast["time_us"], start + horizon + 1.0e-10, side="right")
        )
        if stop < 3:
            continue
        subset = {
            key: (value[:stop] if isinstance(value, np.ndarray) and len(value) == len(forecast["time_us"]) else value)
            for key, value in forecast.items()
        }
        metrics = evaluate_forecast(subset, statistics)
        rows.append(
            {
                "horizon_us": horizon,
                "samples": stop,
                **{key: value for key, value in metrics.items() if not isinstance(value, dict)},
            }
        )
    return rows


def match_indices(source_time: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(source_time, target_time)
    indices = np.clip(indices, 0, len(source_time) - 1)
    previous = np.maximum(indices - 1, 0)
    use_previous = np.abs(source_time[previous] - target_time) < np.abs(
        source_time[indices] - target_time
    )
    indices[use_previous] = previous[use_previous]
    if np.max(np.abs(source_time[indices] - target_time)) > 1.0e-8:
        raise ValueError("Hysteresis trajectories do not share elapsed-time samples")
    return indices


def scalar_observables(coefficients: np.ndarray, modes: np.ndarray) -> dict:
    phi = coefficients[:, direct.FIELD_NAMES.index("phi")]
    ey = coefficients[:, direct.FIELD_NAMES.index("efy")]
    wave_numbers = 2.0 * np.pi * modes / direct.AZIMUTHAL_LENGTH_M
    residual = ey + 1j * wave_numbers[None] * phi
    result = {
        "selected_phi_rms": float(np.sqrt(np.mean(np.abs(phi) ** 2))),
        "selected_ey_rms": float(np.sqrt(np.mean(np.abs(ey) ** 2))),
        "field_gradient_residual_rms": float(np.sqrt(np.mean(np.abs(residual) ** 2))),
    }
    for mode in (2, 7):
        index = int(np.flatnonzero(modes == mode)[0])
        result[f"n{mode}_mean_amplitude"] = float(np.mean(np.abs(phi[:, index])))
    ecdi = np.flatnonzero((modes >= 9) & (modes <= 21))
    result["ecdi_n9_21_phi_power"] = float(np.mean(np.abs(phi[:, ecdi]) ** 2))
    return result


def hysteresis_comparison(primary: dict, up: dict) -> dict:
    indices = match_indices(up["time_us"], primary["time_us"])
    result = {}
    for source, key in (
        (primary["truth"], "truth_E25_history"),
        (up["truth"][indices], "truth_E20_history"),
        (primary["decoded_prediction"], "rom_E25_history"),
        (up["decoded_prediction"][indices], "rom_E20_history"),
    ):
        result[key] = scalar_observables(source, primary["modes"])
    differences = {}
    for kind in ("truth", "rom"):
        high = result[f"{kind}_E25_history"]
        low = result[f"{kind}_E20_history"]
        differences[kind] = {
            name: {
                "absolute_E25_minus_E20": high[name] - low[name],
                "relative_to_E20": (high[name] - low[name])
                / max(abs(low[name]), np.finfo(float).tiny),
            }
            for name in high
        }
    result["differences"] = differences
    result["interpretation"] = (
        "A nonzero equal-Ez history difference is hysteresis/path dependence; "
        "it is not by itself proof of a bifurcation."
    )
    return result


def save_rollout(path: Path, forecast: dict) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["primary_future_PIC_state_used_as_input"] = False
        handle.attrs["context_steps"] = CONTEXT_STEPS
        handle.attrs["lambda_E"] = 1.0
        for key in ("time_us", "frame", "modes", "amplitude_ratio"):
            handle.create_dataset(key, data=forecast[key])
        for key in (
            "truth",
            "persistence",
            "factorized_prediction",
            "decoded_prediction",
            "amplitude_truth_state",
            "amplitude_prediction_state",
            "phase_truth_state",
            "phase_prediction_state",
        ):
            handle.create_dataset(key, data=forecast[key], compression="gzip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_PRIMARY_FEATURES)
    parser.add_argument("--physical", type=Path, default=DEFAULT_PRIMARY_PHYSICAL)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--self-test-up",
        action="store_true",
        help="Exercise the evaluator on the allowed E20 -> E22.5 path; not a primary result.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lock = verify_lock(args.lock.resolve())
    device = torch.device(args.device)
    allowed = allowed_trajectories()
    amplitude_representation, amplitude_direct, amplitude_masks = fit_locked_representation(
        allowed, bidirectional=True
    )
    phase_representation, phase_direct, phase_masks = fit_locked_representation(
        allowed, bidirectional=False
    )
    artifacts = {
        name: Path(item["path"])
        for name, item in lock["frozen_artifacts"].items()
    }
    verify_representation(
        amplitude_representation, artifacts["amplitude_representation"]
    )
    verify_representation(phase_representation, artifacts["phase_representation"])
    statistics = physics.physics_statistics(phase_representation, phase_masks)

    if args.self_test_up:
        target = allowed["e20_to_e22p5"]
        source_ez_kvm = 20.0
        output = args.output.resolve()
        label = "SELF_TEST_ALLOWED_UP_NOT_PRIMARY"
    else:
        if not args.features.is_file() or not args.physical.is_file():
            raise FileNotFoundError(
                "Primary Fourier inputs are not available. Expected "
                f"{args.features} and {args.physical}."
            )
        target = stage2.load_trajectory(
            "e25_to_e22p5_primary", 22.5, args.features, args.physical
        )
        source_ez_kvm = 25.0
        output = args.output.resolve()
        label = "CONFIRMATORY_PRIMARY_E25_TO_E22P5"
    output.mkdir(parents=True, exist_ok=True)
    forecast = forecast_trajectory(
        target,
        source_ez_kvm,
        amplitude_representation,
        amplitude_direct,
        phase_representation,
        phase_direct,
        artifacts["amplitude_checkpoint"],
        artifacts["phase_checkpoint"],
        statistics,
        device,
    )
    metrics = evaluate_forecast(forecast, statistics)
    horizons = horizon_rows(forecast, statistics)
    up_forecast = forecast_trajectory(
        allowed["e20_to_e22p5"],
        20.0,
        amplitude_representation,
        amplitude_direct,
        phase_representation,
        phase_direct,
        artifacts["amplitude_checkpoint"],
        artifacts["phase_checkpoint"],
        statistics,
        device,
    )
    hysteresis = hysteresis_comparison(forecast, up_forecast)
    save_rollout(output / "primary_rollout.h5", forecast)
    write_csv(output / "primary_horizon_metrics.csv", horizons)
    result = {
        "status": label,
        "primary_data_read": not args.self_test_up,
        "evaluation_lock": str(args.lock.resolve()),
        "evaluation_lock_sha256": sha256(args.lock.resolve()),
        "features": str(target.feature_path),
        "features_sha256": sha256(target.feature_path),
        "physical": str(target.physical_path),
        "physical_sha256": sha256(target.physical_path),
        "context": {
            "steps": CONTEXT_STEPS,
            "first_time_us": float(target.time_us[0]),
            "last_time_us": float(target.time_us[CONTEXT_STEPS - 1]),
            "free_rollout_first_time_us": float(forecast["time_us"][0]),
            "free_rollout_last_time_us": float(forecast["time_us"][-1]),
        },
        "metrics": metrics,
        "hysteresis_at_equal_Ez22p5": hysteresis,
        "confirmatory_pass": (
            bool(metrics["passes_all_predeclared_primary_gates"])
            if not args.self_test_up
            else None
        ),
        "model_or_weight_reselection_after_primary": False,
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    (output / "primary_evaluation.json").write_text(
        json.dumps(json_safe(result), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(json_safe(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
