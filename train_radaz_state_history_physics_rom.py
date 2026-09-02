"""Physics-loss ablation for the state/history-conditioned RadAz ROM.

The ablation is gated by the locked data-only development result.  If that
model does not pass all persistence and field-climatology gates, no physics
candidate is trained.  Otherwise identical copies are fine-tuned with a
truth-floor hinge for the spectral relation E_y + d(phi)/dy = 0.  The primary
E25 -> E22.5 trajectory is never read.
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

import train_radaz_carrier_envelope_high_branch_rom as carrier
import train_radaz_carrier_envelope_time_controlled_rom as controlled
import train_radaz_direct_physical_state_rom as direct
import train_radaz_mode_separated_controlled_carrier_rom as separated
import train_radaz_modulation_controlled_carrier_rom as modulation
import train_radaz_regime_aware_transition_rom as stage2
import train_radaz_state_history_conditioned_rom as history_rom


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ONLY = (
    ROOT / "workdirs" / "train_radaz_state_history_conditioned_rom"
)
DEFAULT_OUTPUT = (
    ROOT / "workdirs" / "train_radaz_state_history_physics_rom"
)


@dataclass
class PhysicsStatistics:
    residual_scale: np.ndarray
    truth_floor_power: np.ndarray
    audit: dict


@dataclass
class Candidate:
    lambda_e: float
    model: history_rom.StateHistoryConditionedROM
    epoch: int
    metrics: dict
    prediction: np.ndarray
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
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([json_safe(row) for row in rows])


def physics_statistics(
    representation: carrier.CarrierRepresentation,
    fit_masks: dict[str, np.ndarray],
) -> PhysicsStatistics:
    packed = np.concatenate(
        [
            representation.groups[name]["carrier_physical"][fit_masks[name]]
            for name in representation.groups
        ],
        axis=0,
    )
    shaped = packed.reshape(
        len(packed),
        len(direct.FIELD_NAMES),
        len(carrier.SELECTED_MODE_NUMBERS),
        2,
    )
    coefficients = shaped[..., 0] + 1j * shaped[..., 1]
    phi = coefficients[:, direct.FIELD_NAMES.index("phi")]
    ey = coefficients[:, direct.FIELD_NAMES.index("efy")]
    wave_numbers = (
        2.0
        * np.pi
        * carrier.SELECTED_MODE_NUMBERS
        / direct.AZIMUTHAL_LENGTH_M
    )
    residual = ey + 1j * wave_numbers[None] * phi
    scale = np.sqrt(np.mean(np.abs(ey) ** 2, axis=0))
    scale = np.maximum(scale, np.max(scale) * 1.0e-6)
    floor = np.mean(np.abs(residual / scale[None]) ** 2, axis=0)
    return PhysicsStatistics(
        residual_scale=scale,
        truth_floor_power=floor,
        audit={
            "selected_modes": carrier.SELECTED_MODE_NUMBERS.tolist(),
            "truth_residual_over_ey_rms": float(
                np.sqrt(np.mean(np.abs(residual) ** 2))
                / np.sqrt(np.mean(np.abs(ey) ** 2))
            ),
            "truth_floor_normalized_rms": float(np.sqrt(np.mean(floor))),
            "fit_samples": int(len(packed)),
            "primary_data_loaded": False,
        },
    )


class CarrierFieldGradientConstraint(nn.Module):
    def __init__(
        self,
        representation: carrier.CarrierRepresentation,
        statistics: PhysicsStatistics,
    ) -> None:
        super().__init__()
        selected = representation.scaler.slices["carrier_physical"]
        self.slice_start = int(selected.start)
        self.slice_stop = int(selected.stop)
        self.register_buffer(
            "mean",
            torch.as_tensor(
                representation.scaler.means["carrier_physical"],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "scale",
            torch.as_tensor(
                representation.scaler.scales["carrier_physical"],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "wave_numbers",
            torch.as_tensor(
                2.0
                * np.pi
                * carrier.SELECTED_MODE_NUMBERS
                / direct.AZIMUTHAL_LENGTH_M,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "residual_scale",
            torch.as_tensor(statistics.residual_scale, dtype=torch.float32),
        )
        self.register_buffer(
            "truth_floor_power",
            torch.as_tensor(statistics.truth_floor_power, dtype=torch.float32),
        )

    def power(self, states: torch.Tensor) -> torch.Tensor:
        raw = states[..., self.slice_start : self.slice_stop] * self.scale + self.mean
        shaped = raw.reshape(
            *raw.shape[:-1],
            len(direct.FIELD_NAMES),
            len(carrier.SELECTED_MODE_NUMBERS),
            2,
        )
        phi = shaped[..., direct.FIELD_NAMES.index("phi"), :, :]
        ey = shaped[..., direct.FIELD_NAMES.index("efy"), :, :]
        residual_real = ey[..., 0] - self.wave_numbers * phi[..., 1]
        residual_imag = ey[..., 1] + self.wave_numbers * phi[..., 0]
        return (
            residual_real.square() + residual_imag.square()
        ) / self.residual_scale.square()

    def loss(self, states: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.power(states) - self.truth_floor_power).mean()


def physics_metrics(
    states: np.ndarray,
    representation: carrier.CarrierRepresentation,
    statistics: PhysicsStatistics,
) -> dict:
    packed = representation.scaler.inverse(states)["carrier_physical"]
    shaped = packed.reshape(
        len(packed),
        len(direct.FIELD_NAMES),
        len(carrier.SELECTED_MODE_NUMBERS),
        2,
    )
    coefficients = shaped[..., 0] + 1j * shaped[..., 1]
    phi = coefficients[:, direct.FIELD_NAMES.index("phi")]
    ey = coefficients[:, direct.FIELD_NAMES.index("efy")]
    wave_numbers = (
        2.0
        * np.pi
        * carrier.SELECTED_MODE_NUMBERS
        / direct.AZIMUTHAL_LENGTH_M
    )
    residual = ey + 1j * wave_numbers[None] * phi
    power = np.abs(residual / statistics.residual_scale[None]) ** 2
    return {
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


def accepted_data_metrics(metrics: dict) -> bool:
    return bool(
        metrics["passes_all_persistence_gates"]
        and metrics["passes_field_climatology_gate"]
        and metrics["finite_fraction"] == 1.0
    )


def train_candidate(
    lambda_e: float,
    initial_state: dict,
    representation: carrier.CarrierRepresentation,
    complex_loss: separated.ModeBalancedComplexLoss,
    constraint: CarrierFieldGradientConstraint,
    dataset: controlled.ControlledWindowDataset,
    truth: np.ndarray,
    persistence: np.ndarray,
    state_history: np.ndarray,
    control_history: np.ndarray,
    future_controls: np.ndarray,
    base: dict,
    validation_start: int,
    statistics: PhysicsStatistics,
    args: argparse.Namespace,
    device: torch.device,
) -> Candidate:
    controlled.seed_everything(args.seed + 20)
    model = history_rom.StateHistoryConditionedROM(
        representation,
        carrier.STATE_DIMENSION,
        history_rom.CONTROL_DIMENSION,
        args.hidden_dim,
        args.delta_limit,
    ).to(device)
    model.load_state_dict(initial_state)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 21),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    full_indices = np.arange(validation_start, validation_start + len(truth))

    def evaluate():
        prediction = controlled.rollout(
            model,
            state_history,
            control_history,
            future_controls,
            device,
        )
        metrics = modulation.evaluate_full_model(
            truth,
            prediction,
            persistence,
            representation,
            base,
            full_indices,
        )
        metrics.update(physics_metrics(prediction, representation, statistics))
        return prediction, metrics

    best_prediction, best_metrics = evaluate()
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale = 0
    rows: list[dict] = []
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
        prediction, metrics = evaluate()
        row = {
            "lambda_E": lambda_e,
            "epoch": epoch,
            **{f"train_{name}": value / count for name, value in sums.items()},
            **metrics,
        }
        rows.append(row)
        current_accepted = accepted_data_metrics(metrics)
        best_accepted = accepted_data_metrics(best_metrics)
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
            best_prediction = prediction
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"lambdaE={lambda_e:g} epoch={epoch:03d} "
                f"physics={metrics['field_gradient_excess_hinge']:.4e} "
                f"phi={metrics['selected_phi_skill_vs_raw_persistence']:+.3f} "
                f"n2={metrics['phi_n2_amplitude_skill']:+.3f} "
                f"n7={metrics['phi_n7_amplitude_skill']:+.3f}",
                flush=True,
            )
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    return Candidate(
        lambda_e, model, best_epoch, best_metrics, best_prediction, rows
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-only", type=Path, default=DEFAULT_DATA_ONLY)
    parser.add_argument("--base", type=Path, default=modulation.DEFAULT_BASE)
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
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_only = args.data_only.resolve()
    lock_path = data_only / "model_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if not bool(lock.get("accepted_for_physics_ablation", False)):
        skipped = {
            "status": "SKIPPED_DATA_GATE",
            "reason": "Locked data-only model failed a predeclared gate",
            "data_only_lock": str(lock_path),
            "data_only_lock_sha256": sha256(lock_path),
            "development_metrics": lock.get("development_metrics", {}),
            "physics_candidates_trained": False,
            "primary_E25_to_E22p5_loaded": False,
        }
        (output / "model_lock.json").write_text(
            json.dumps(json_safe(skipped), indent=2), encoding="utf-8"
        )
        (output / "README.md").write_text(
            "# State/history physics ROM\n\n"
            "Status: `SKIPPED_DATA_GATE`. No physics candidate was trained.\n",
            encoding="utf-8",
        )
        print(json.dumps(json_safe(skipped), indent=2), flush=True)
        return

    device = torch.device(args.device)
    controlled.seed_everything(args.seed)
    checkpoint_path = data_only / "state_history_conditioned_data_only.pt"
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model_args = lock["arguments"]
    args.hidden_dim = int(checkpoint["hidden_dimension"])
    args.delta_limit = float(checkpoint["delta_limit"])
    args.history_steps = int(checkpoint["history_steps"])
    args.rollout_steps = int(checkpoint["rollout_steps"])
    args.azimuth_shifts = int(checkpoint["azimuth_shifts"])

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
    statistics = physics_statistics(representation, fit_masks)
    (output / "truth_field_gradient_audit.json").write_text(
        json.dumps(json_safe(statistics.audit), indent=2), encoding="utf-8"
    )
    constraint = CarrierFieldGradientConstraint(
        representation, statistics
    ).to(device)
    controls = {
        "e25_stationary": history_rom.history_controls_for(
            e25, 25.0, 25.0, False
        ),
        "e20_to_e22p5": history_rom.history_controls_for(
            up, 22.5, 20.0, True
        ),
    }
    train_data = history_rom.augmented_dataset(
        [
            (
                e25,
                representation.states["e25_stationary"],
                controls["e25_stationary"],
                12.0,
                24.0,
                2,
            ),
            (
                up,
                representation.states["e20_to_e22p5"],
                controls["e20_to_e22p5"],
                float(up.time_us[0]),
                35.0,
                1,
            ),
        ],
        args.history_steps,
        args.rollout_steps,
        representation,
        args.azimuth_shifts,
    )
    complex_loss = separated.ModeBalancedComplexLoss(
        representation.scaler,
        representation.amplitude_scale,
        float(model_args["lambda_amplitude"]),
        float(model_args["lambda_phase"]),
        float(model_args["lambda_frequency"]),
        float(model_args["lambda_n2"]),
        float(model_args["lambda_n7"]),
    ).to(device)
    validation_start = int(
        np.flatnonzero(up.time_us >= 35.0 - 1.0e-10)[0]
    )
    state = representation.states["e20_to_e22p5"]
    state_history = state[
        validation_start - args.history_steps : validation_start
    ]
    control_history = controls["e20_to_e22p5"][
        validation_start - args.history_steps : validation_start
    ]
    truth = state[validation_start:]
    future_controls = controls["e20_to_e22p5"][validation_start:]
    persistence = np.repeat(state_history[-1:], len(truth), axis=0)
    base = modulation.load_base(args.base.resolve())

    baseline_model = history_rom.StateHistoryConditionedROM(
        representation,
        carrier.STATE_DIMENSION,
        history_rom.CONTROL_DIMENSION,
        args.hidden_dim,
        args.delta_limit,
    ).to(device)
    baseline_model.load_state_dict(checkpoint["model_state_dict"])
    baseline_prediction = controlled.rollout(
        baseline_model,
        state_history,
        control_history,
        future_controls,
        device,
    )
    full_indices = np.arange(validation_start, validation_start + len(truth))
    baseline_metrics = modulation.evaluate_full_model(
        truth,
        baseline_prediction,
        persistence,
        representation,
        base,
        full_indices,
    )
    baseline_metrics.update(
        physics_metrics(baseline_prediction, representation, statistics)
    )
    lambdas = [float(value) for value in args.lambda_e.split(",")]
    candidates = []
    for lambda_e in lambdas:
        print(f"training lambda_E={lambda_e:g}", flush=True)
        candidates.append(
            train_candidate(
                lambda_e,
                checkpoint["model_state_dict"],
                representation,
                complex_loss,
                constraint,
                train_data,
                truth,
                persistence,
                state_history,
                control_history,
                future_controls,
                base,
                validation_start,
                statistics,
                args,
                device,
            )
        )

    baseline_excess = baseline_metrics["field_gradient_excess_hinge"]
    rows = [{"model": "data_only", "lambda_E": 0.0, **baseline_metrics}]
    eligible = []
    for candidate in candidates:
        reduction = 1.0 - (
            candidate.metrics["field_gradient_excess_hinge"]
            / max(baseline_excess, 1.0e-30)
        )
        row = {
            "model": "physics",
            "lambda_E": candidate.lambda_e,
            "best_epoch": candidate.epoch,
            "physics_excess_reduction": reduction,
            **candidate.metrics,
        }
        rows.append(row)
        if accepted_data_metrics(candidate.metrics) and reduction > 0.0:
            eligible.append((reduction, candidate))
        torch.save(
            {
                "model_state_dict": candidate.model.state_dict(),
                "lambda_E": candidate.lambda_e,
                "best_epoch": candidate.epoch,
                "metrics": candidate.metrics,
            },
            output / f"physics_candidate_lambdaE_{candidate.lambda_e:g}.pt",
        )
        write_csv(
            output / f"training_history_lambdaE_{candidate.lambda_e:g}.csv",
            candidate.rows,
        )
    write_csv(output / "physics_comparison.csv", rows)

    selected = max(eligible, key=lambda item: item[0]) if eligible else None
    status = "PHYSICS_CANDIDATE_ACCEPTED" if selected else "NO_PHYSICS_CANDIDATE_ACCEPTED"
    selected_summary = None
    if selected:
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
            output / "selected_physics_rom.pt",
        )
        with h5py.File(output / "selected_development_rollout.h5", "w") as handle:
            handle.attrs["primary_E25_to_E22p5_loaded"] = False
            handle.create_dataset("time_us", data=up.time_us[validation_start:])
            handle.create_dataset("truth_state", data=truth, compression="gzip")
            handle.create_dataset(
                "prediction_state", data=candidate.prediction, compression="gzip"
            )

    result_lock = {
        "status": status,
        "baseline_metrics": baseline_metrics,
        "selected": selected_summary,
        "truth_residual_audit": statistics.audit,
        "candidate_rows": rows,
        "data_only_checkpoint": str(checkpoint_path),
        "data_only_checkpoint_sha256": sha256(checkpoint_path),
        "primary_test": {
            "direction": "E25_to_E22.5",
            "data_loaded": False,
            "used_for_selection": False,
        },
        "script": str(Path(__file__).resolve()),
        "arguments": vars(args),
    }
    (output / "model_lock.json").write_text(
        json.dumps(json_safe(result_lock), indent=2), encoding="utf-8"
    )
    (output / "README.md").write_text(
        f"# State/history physics ROM\n\nStatus: `{status}`.\n",
        encoding="utf-8",
    )
    print(json.dumps(json_safe(result_lock), indent=2), flush=True)


if __name__ == "__main__":
    main()
