"""End-to-end physics-loss fine-tuning for the factorized RadAz ROM.

The recurrent phase branch is fine-tuned with data, complex modal, and
truth-floor field-gradient losses.  The state/E-history amplitude branch is
frozen and supplies prospective n=2/n=7 amplitudes during development
evaluation.  Candidate selection requires the fused forecast to preserve all
data gates while reducing the physics residual.  No primary E25 -> E22.5 data
are read.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import analyze_radaz_augmented_physical_state_dynamics as augmented
import evaluate_radaz_factorized_physics_decoder as decoder
import train_radaz_carrier_envelope_high_branch_rom as carrier
import train_radaz_carrier_envelope_time_controlled_rom as controlled
import train_radaz_direct_physical_state_rom as direct
import train_radaz_mode_separated_controlled_carrier_rom as separated
import train_radaz_modulation_controlled_carrier_rom as modulation
import train_radaz_regime_aware_transition_rom as stage2
import train_radaz_state_history_physics_rom as physics


ROOT = Path(__file__).resolve().parent
DEFAULT_PHASE = (
    ROOT / "workdirs" / "train_radaz_mode_separated_controlled_carrier_rom"
)
DEFAULT_FACTOR = (
    ROOT / "workdirs" / "build_radaz_state_phase_factorized_rom"
)
DEFAULT_OUTPUT = (
    ROOT / "workdirs" / "train_radaz_factorized_end_to_end_physics_rom"
)


@dataclass
class Candidate:
    lambda_e: float
    model: separated.ModeSeparatedControlledDelayROM
    epoch: int
    metrics: dict
    state_prediction: np.ndarray
    coefficient_prediction: np.ndarray
    rows: list[dict]


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


def fuse_amplitude(
    phase_coefficients: np.ndarray,
    amplitude_coefficients: np.ndarray,
    modes: np.ndarray,
) -> np.ndarray:
    indices = np.asarray(
        [int(np.flatnonzero(modes == mode)[0]) for mode in (2, 7)],
        dtype=np.int64,
    )
    ratio = (
        np.abs(amplitude_coefficients[:, 0, indices])
        / np.maximum(
            np.abs(phase_coefficients[:, 0, indices]),
            np.finfo(float).tiny,
        )
    )
    result = phase_coefficients.copy()
    result[:, :, indices] *= ratio[:, None, :]
    return result


def phase_frequency_mhz(coefficients: np.ndarray, dt_us: float) -> float:
    products = coefficients[1:] * np.conj(coefficients[:-1])
    return float(np.angle(np.sum(products)) / (2.0 * np.pi * dt_us))


def evaluate_fused(
    state_prediction: np.ndarray,
    truth_state: np.ndarray,
    persistence_state: np.ndarray,
    representation: carrier.CarrierRepresentation,
    base: dict,
    full_indices: np.ndarray,
    truth_coefficients: np.ndarray,
    persistence_coefficients: np.ndarray,
    amplitude_coefficients: np.ndarray,
    modes: np.ndarray,
    statistics: physics.PhysicsStatistics,
    dt_us: float,
) -> tuple[dict, np.ndarray]:
    metrics = modulation.evaluate_full_model(
        truth_state,
        state_prediction,
        persistence_state,
        representation,
        base,
        full_indices,
    )
    phase_coefficients = modulation.decode_coefficients(
        state_prediction, representation, full_indices
    )
    coefficients = fuse_amplitude(
        phase_coefficients, amplitude_coefficients, modes
    )
    for field_index, field in ((0, "phi"), (3, "efy")):
        field_result = augmented.scalar_metrics(
            truth_coefficients[:, field_index],
            coefficients[:, field_index],
            persistence_coefficients[:, field_index],
        )
        metrics[f"selected_{field}_skill_vs_raw_persistence"] = field_result[
            "skill_vs_persistence"
        ]
        metrics[f"selected_{field}_nrmse"] = field_result["nrmse"]
        metrics[f"selected_{field}_temporal_anomaly_correlation"] = (
            field_result["temporal_anomaly_correlation"]
        )
    for mode in (2, 7):
        index = int(np.flatnonzero(modes == mode)[0])
        amplitude_result = augmented.scalar_metrics(
            np.abs(truth_coefficients[:, 0, index]),
            np.abs(coefficients[:, 0, index]),
            np.abs(persistence_coefficients[:, 0, index]),
        )
        metrics[f"phi_n{mode}_amplitude_skill"] = amplitude_result[
            "skill_vs_persistence"
        ]
        metrics[f"phi_n{mode}_amplitude_correlation"] = amplitude_result[
            "correlation"
        ]
        truth_frequency = phase_frequency_mhz(
            truth_coefficients[:, 0, index], dt_us
        )
        prediction_frequency = phase_frequency_mhz(
            coefficients[:, 0, index], dt_us
        )
        metrics[f"phi_n{mode}_truth_frequency_MHz"] = truth_frequency
        metrics[f"phi_n{mode}_prediction_frequency_MHz"] = prediction_frequency
        metrics[f"phi_n{mode}_frequency_abs_error_MHz"] = abs(
            prediction_frequency - truth_frequency
        )
    metrics.update(
        decoder.field_metrics(
            truth_coefficients,
            coefficients,
            persistence_coefficients,
            modes,
            statistics,
        )
    )
    gates = (
        metrics["carrier_state_skill_vs_envelope_persistence"],
        metrics["composite_state_skill_vs_persistence"],
        metrics["radial_skill_vs_persistence"],
        metrics["MTSI_n1_6_transport_skill"],
        metrics["ECDI_n9_21_transport_skill"],
        metrics["selected_phi_skill_vs_raw_persistence"],
        metrics["selected_efy_skill_vs_raw_persistence"],
        metrics["phi_n2_amplitude_skill"],
        metrics["phi_n7_amplitude_skill"],
    )
    metrics["minimum_persistence_gate_skill"] = float(min(gates))
    metrics["passes_all_persistence_gates"] = bool(min(gates) > 0.0)
    metrics["passes_field_climatology_gate"] = bool(
        metrics["selected_phi_nrmse"] < 1.0
        and metrics["selected_efy_nrmse"] < 1.0
    )
    metrics["selection_score"] = float(
        metrics["minimum_persistence_gate_skill"]
        + 0.05
        * min(
            1.0 - metrics["selected_phi_nrmse"],
            1.0 - metrics["selected_efy_nrmse"],
        )
    )
    return metrics, coefficients


def accepted(metrics: dict) -> bool:
    return bool(
        metrics["finite_fraction"] == 1.0
        and metrics["passes_all_persistence_gates"]
        and metrics["passes_field_climatology_gate"]
    )


def train_candidate(
    lambda_e: float,
    initial_state: dict,
    representation: carrier.CarrierRepresentation,
    complex_loss: separated.ModeBalancedComplexLoss,
    constraint: physics.CarrierFieldGradientConstraint,
    dataset: controlled.ControlledWindowDataset,
    truth_state: np.ndarray,
    persistence_state: np.ndarray,
    state_history: np.ndarray,
    control_history: np.ndarray,
    future_controls: np.ndarray,
    base: dict,
    full_indices: np.ndarray,
    truth_coefficients: np.ndarray,
    persistence_coefficients: np.ndarray,
    amplitude_coefficients: np.ndarray,
    modes: np.ndarray,
    statistics: physics.PhysicsStatistics,
    dt_us: float,
    args: argparse.Namespace,
    device: torch.device,
) -> Candidate:
    controlled.seed_everything(args.seed + 30)
    model = separated.ModeSeparatedControlledDelayROM(
        carrier.STATE_DIMENSION,
        controlled.CONTROL_DIMENSION,
        args.hidden_dim,
        args.delta_limit,
    ).to(device)
    model.load_state_dict(initial_state)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 31),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    def forecast():
        state_prediction = controlled.rollout(
            model,
            state_history,
            control_history,
            future_controls,
            device,
        )
        metrics, coefficients = evaluate_fused(
            state_prediction,
            truth_state,
            persistence_state,
            representation,
            base,
            full_indices,
            truth_coefficients,
            persistence_coefficients,
            amplitude_coefficients,
            modes,
            statistics,
            dt_us,
        )
        return state_prediction, coefficients, metrics

    best_state_prediction, best_coefficients, best_metrics = forecast()
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale = 0
    rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {name: 0.0 for name in ("total", "data", "complex", "physics")}
        count = 0
        for state, controls, target, target_controls in loader:
            state = state.to(device)
            controls = controls.to(device)
            target = target.to(device)
            target_controls = target_controls.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model.rollout(state, controls, target_controls)
            data_term = carrier.group_loss(
                prediction, target, representation.scaler
            )
            complex_term, _ = complex_loss(prediction, target)
            physics_term = constraint.loss(prediction)
            loss = data_term + complex_term + lambda_e * physics_term
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_count = len(state)
            for name, value in (
                ("total", loss),
                ("data", data_term),
                ("complex", complex_term),
                ("physics", physics_term),
            ):
                sums[name] += float(value.detach()) * batch_count
            count += batch_count
        state_prediction, coefficients, metrics = forecast()
        row = {
            "lambda_E": lambda_e,
            "epoch": epoch,
            **{f"train_{name}": value / count for name, value in sums.items()},
            **metrics,
        }
        rows.append(row)
        current_accepted = accepted(metrics)
        best_accepted = accepted(best_metrics)
        improved = False
        if current_accepted and not best_accepted:
            improved = True
        elif current_accepted and best_accepted:
            improved = (
                metrics["field_gradient_excess_hinge"]
                < best_metrics["field_gradient_excess_hinge"] - 1.0e-8
            )
        elif not best_accepted:
            improved = (
                metrics["selection_score"]
                > best_metrics["selection_score"] + 1.0e-5
            )
        if improved:
            best_state = copy.deepcopy(model.state_dict())
            best_state_prediction = state_prediction
            best_coefficients = coefficients
            best_metrics = metrics
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"lambdaE={lambda_e:g} epoch={epoch:03d} "
                f"hinge={metrics['field_gradient_excess_hinge']:.4e} "
                f"phi={metrics['selected_phi_skill_vs_raw_persistence']:+.3f} "
                f"Ey={metrics['selected_efy_skill_vs_raw_persistence']:+.3f}",
                flush=True,
            )
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    return Candidate(
        lambda_e,
        model,
        best_epoch,
        best_metrics,
        best_state_prediction,
        best_coefficients,
        rows,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=Path, default=DEFAULT_PHASE)
    parser.add_argument("--factor", type=Path, default=DEFAULT_FACTOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lambda-e", default="0.01,0.1,1.0")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phase_dir = args.phase.resolve()
    factor_dir = args.factor.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    phase_lock_path = phase_dir / "model_lock.json"
    phase_checkpoint_path = (
        phase_dir / "mode_separated_controlled_carrier_data_only.pt"
    )
    factor_lock_path = factor_dir / "model_lock.json"
    factor_rollout_path = factor_dir / "development_rollout_35to40us.h5"
    phase_lock = json.loads(phase_lock_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(
        phase_checkpoint_path, map_location=device, weights_only=False
    )
    args.hidden_dim = int(checkpoint["hidden_dimension"])
    args.delta_limit = float(checkpoint["delta_limit"])
    args.history_steps = int(checkpoint["history_steps"])
    args.rollout_steps = int(checkpoint["rollout_steps"])

    trajectories = {
        "e25_stationary": stage2.load_trajectory(
            "e25_stationary", 25.0, stage2.DEFAULT_E25_FEATURES, stage2.DEFAULT_E25_PHYSICAL
        ),
        "e20_to_e22p5": stage2.load_trajectory(
            "e20_to_e22p5", 22.5, stage2.DEFAULT_UP_FEATURES, stage2.DEFAULT_UP_PHYSICAL
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
    constraint = physics.CarrierFieldGradientConstraint(
        representation, statistics
    ).to(device)
    controls = {
        "e25_stationary": controlled.controls_for(e25, 25.0, 25.0, False),
        "e20_to_e22p5": controlled.controls_for(up, 22.5, 20.0, True),
    }
    train_data = controlled.ControlledWindowDataset(
        [
            (e25, representation.states["e25_stationary"], controls["e25_stationary"], 12.0, 24.0, 2),
            (up, representation.states["e20_to_e22p5"], controls["e20_to_e22p5"], float(up.time_us[0]), 35.0, 1),
        ],
        args.history_steps,
        args.rollout_steps,
    )
    phase_args = phase_lock["arguments"]
    complex_loss = separated.ModeBalancedComplexLoss(
        representation.scaler,
        representation.amplitude_scale,
        float(phase_args["lambda_amplitude"]),
        float(phase_args["lambda_phase"]),
        float(phase_args["lambda_frequency"]),
        float(phase_args["lambda_n2"]),
        float(phase_args["lambda_n7"]),
    ).to(device)
    validation_start = int(np.flatnonzero(up.time_us >= 35.0 - 1.0e-10)[0])
    state = representation.states["e20_to_e22p5"]
    state_history = state[validation_start - args.history_steps : validation_start]
    control_history = controls["e20_to_e22p5"][validation_start - args.history_steps : validation_start]
    truth_state = state[validation_start:]
    future_controls = controls["e20_to_e22p5"][validation_start:]
    persistence_state = np.repeat(state_history[-1:], len(truth_state), axis=0)
    base = modulation.load_base(Path(phase_args["base"]).resolve())
    full_indices = np.arange(validation_start, validation_start + len(truth_state))
    with h5py.File(factor_rollout_path, "r") as handle:
        modes = np.asarray(handle["selected_mode_numbers"], dtype=np.int64)
        truth_coefficients = np.asarray(handle["truth_physical_coefficients"])
        persistence_coefficients = np.asarray(handle["raw_persistence_physical_coefficients"])
        amplitude_coefficients = np.asarray(handle["amplitude_branch_prediction"])
        time_us = np.asarray(handle["time_us"], dtype=np.float64)
        frame = np.asarray(handle["frame"], dtype=np.int64)
    dt_us = float(np.median(np.diff(time_us)))

    baseline_model = separated.ModeSeparatedControlledDelayROM(
        carrier.STATE_DIMENSION,
        controlled.CONTROL_DIMENSION,
        args.hidden_dim,
        args.delta_limit,
    ).to(device)
    baseline_model.load_state_dict(checkpoint["model_state_dict"])
    baseline_state = controlled.rollout(
        baseline_model, state_history, control_history, future_controls, device
    )
    baseline_metrics, baseline_coefficients = evaluate_fused(
        baseline_state,
        truth_state,
        persistence_state,
        representation,
        base,
        full_indices,
        truth_coefficients,
        persistence_coefficients,
        amplitude_coefficients,
        modes,
        statistics,
        dt_us,
    )
    baseline_hinge = baseline_metrics["field_gradient_excess_hinge"]

    candidates = []
    for lambda_e in [float(value) for value in args.lambda_e.split(",")]:
        print(f"training lambda_E={lambda_e:g}", flush=True)
        candidates.append(
            train_candidate(
                lambda_e,
                checkpoint["model_state_dict"],
                representation,
                complex_loss,
                constraint,
                train_data,
                truth_state,
                persistence_state,
                state_history,
                control_history,
                future_controls,
                base,
                full_indices,
                truth_coefficients,
                persistence_coefficients,
                amplitude_coefficients,
                modes,
                statistics,
                dt_us,
                args,
                device,
            )
        )

    rows = [{"model": "factorized_data_only", "lambda_E": 0.0, **baseline_metrics}]
    eligible = []
    for candidate in candidates:
        reduction = 1.0 - (
            candidate.metrics["field_gradient_excess_hinge"]
            / max(baseline_hinge, 1.0e-30)
        )
        row = {
            "model": "factorized_end_to_end_physics",
            "lambda_E": candidate.lambda_e,
            "best_epoch": candidate.epoch,
            "physics_excess_reduction": reduction,
            **candidate.metrics,
        }
        rows.append(row)
        if accepted(candidate.metrics) and reduction > 0.0 and candidate.epoch > 0:
            eligible.append((reduction, candidate))
        torch.save(
            {
                "model_state_dict": candidate.model.state_dict(),
                "lambda_E": candidate.lambda_e,
                "best_epoch": candidate.epoch,
                "metrics": candidate.metrics,
            },
            output / f"physics_phase_candidate_lambdaE_{candidate.lambda_e:g}.pt",
        )
        write_csv(
            output / f"training_history_lambdaE_{candidate.lambda_e:g}.csv",
            candidate.rows,
        )
    write_csv(output / "physics_comparison.csv", rows)
    (output / "physics_comparison.json").write_text(
        json.dumps(json_safe(rows), indent=2), encoding="utf-8"
    )
    selected = max(eligible, key=lambda item: item[0]) if eligible else None
    status = (
        "END_TO_END_PHYSICS_ACCEPTED"
        if selected is not None
        else "NO_END_TO_END_PHYSICS_ACCEPTED"
    )
    selected_summary = None
    if selected is not None:
        reduction, candidate = selected
        selected_summary = {
            "lambda_E": candidate.lambda_e,
            "best_epoch": candidate.epoch,
            "physics_excess_reduction": reduction,
            "metrics": candidate.metrics,
        }
        torch.save(
            {
                "model_state_dict": candidate.model.state_dict(),
                **selected_summary,
            },
            output / "selected_end_to_end_physics_phase_rom.pt",
        )
        with h5py.File(output / "selected_development_rollout.h5", "w") as handle:
            handle.attrs["primary_E25_to_E22p5_loaded"] = False
            handle.attrs["lambda_E"] = candidate.lambda_e
            handle.create_dataset("time_us", data=time_us)
            handle.create_dataset("frame", data=frame)
            handle.create_dataset("truth_state", data=truth_state, compression="gzip")
            handle.create_dataset("prediction_state", data=candidate.state_prediction, compression="gzip")
            handle.create_dataset("truth_physical_coefficients", data=truth_coefficients, compression="gzip")
            handle.create_dataset("prediction_physical_coefficients", data=candidate.coefficient_prediction, compression="gzip")
            handle.create_dataset("raw_persistence_physical_coefficients", data=persistence_coefficients, compression="gzip")

    lock = {
        "status": status,
        "selected": selected_summary,
        "baseline_metrics": baseline_metrics,
        "candidate_rows": rows,
        "physics_training": {
            "phase_branch_weights_fine_tuned": True,
            "amplitude_branch_frozen": True,
            "loss": "data + complex_modal + lambda_E*truth_floor_field_gradient",
            "future_PIC_truth_used_as_input": False,
        },
        "truth_residual_audit": statistics.audit,
        "phase_checkpoint": str(phase_checkpoint_path),
        "phase_checkpoint_sha256": sha256(phase_checkpoint_path),
        "factor_model_lock": str(factor_lock_path),
        "factor_model_lock_sha256": sha256(factor_lock_path),
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
    (output / "README.md").write_text(
        f"# Factorized end-to-end physics ROM\n\nStatus: `{status}`.\n",
        encoding="utf-8",
    )
    print(json.dumps(json_safe(lock), indent=2), flush=True)


if __name__ == "__main__":
    main()
