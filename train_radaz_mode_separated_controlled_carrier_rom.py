"""Train one controlled carrier ROM with mode-separated output heads.

One shared GRU carries the complete 150D history.  Independent residual heads
advance n=2, n=7, and all remaining coordinates, so the two diagnostically
important modes need not share a fixed slow clock or compete in the final
projection.  Only current/source Ez and elapsed time are supplied; the slow
phase remains a state inferred from history.  The primary E25 -> E22.5 data are
never read.
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
from torch import nn

import train_radaz_carrier_envelope_high_branch_rom as carrier
import train_radaz_carrier_envelope_time_controlled_rom as controlled
import train_radaz_direct_physical_state_rom as direct
import train_radaz_modulation_controlled_carrier_rom as modulation
import train_radaz_regime_aware_transition_rom as stage2


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT / "workdirs" / "train_radaz_mode_separated_controlled_carrier_rom"
)
CONTROL_DIMENSION = 3


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


def physical_mode_state_indices(mode_number: int) -> np.ndarray:
    local_mode = int(
        np.flatnonzero(carrier.SELECTED_MODE_NUMBERS == mode_number)[0]
    )
    indices = []
    offset = 10
    mode_count = len(carrier.SELECTED_MODE_NUMBERS)
    for field_index in range(len(direct.FIELD_NAMES)):
        for component in range(2):
            indices.append(
                offset + (field_index * mode_count + local_mode) * 2 + component
            )
    return np.asarray(indices, dtype=np.int64)


class ModeSeparatedControlledDelayROM(nn.Module):
    """Shared recurrent memory with independent n=2/n=7/rest increments."""

    def __init__(
        self,
        state_dimension: int,
        control_dimension: int,
        hidden_dimension: int,
        delta_limit: float,
    ) -> None:
        super().__init__()
        self.state_dimension = state_dimension
        self.control_dimension = control_dimension
        self.hidden_dimension = hidden_dimension
        self.delta_limit = float(delta_limit)
        self.gru = nn.GRU(
            state_dimension + control_dimension,
            hidden_dimension,
            batch_first=True,
        )
        n2 = physical_mode_state_indices(2)
        n7 = physical_mode_state_indices(7)
        selected = np.concatenate((n2, n7))
        rest = np.setdiff1d(np.arange(state_dimension), selected)
        self.register_buffer("n2_indices", torch.as_tensor(n2, dtype=torch.long))
        self.register_buffer("n7_indices", torch.as_tensor(n7, dtype=torch.long))
        self.register_buffer("rest_indices", torch.as_tensor(rest, dtype=torch.long))
        context_dimension = hidden_dimension + state_dimension + control_dimension
        self.n2_head = self._make_head(context_dimension, len(n2), hidden_dimension)
        self.n7_head = self._make_head(context_dimension, len(n7), hidden_dimension)
        self.rest_head = self._make_head(context_dimension, len(rest), hidden_dimension)

    @staticmethod
    def _make_head(input_dimension: int, output_dimension: int, hidden_dimension: int):
        head = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, output_dimension),
        )
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
        return head

    @staticmethod
    def _scatter(
        base: torch.Tensor, indices: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        expanded = indices[None].expand(len(base), -1)
        return base.scatter(1, expanded, values)

    def rollout(
        self,
        state_history: torch.Tensor,
        control_history: torch.Tensor,
        future_controls: torch.Tensor,
    ) -> torch.Tensor:
        inputs = torch.cat((state_history, control_history), dim=-1)
        _, hidden = self.gru(inputs)
        current = state_history[:, -1]
        predictions = []
        for step in range(future_controls.shape[1]):
            control = future_controls[:, step]
            context = torch.cat((hidden[-1], current, control), dim=-1)
            delta = torch.zeros_like(current)
            delta = self._scatter(delta, self.n2_indices, self.n2_head(context))
            delta = self._scatter(delta, self.n7_indices, self.n7_head(context))
            delta = self._scatter(delta, self.rest_indices, self.rest_head(context))
            current = current + self.delta_limit * torch.tanh(delta)
            recurrent_input = torch.cat((current, control), dim=-1)[:, None]
            _, hidden = self.gru(recurrent_input, hidden)
            predictions.append(current)
        return torch.stack(predictions, dim=1)


class ModeBalancedComplexLoss(nn.Module):
    def __init__(
        self,
        scaler: carrier.BranchScaler,
        amplitude_scale: np.ndarray,
        lambda_amplitude: float,
        lambda_phase: float,
        lambda_frequency: float,
        lambda_n2: float,
        lambda_n7: float,
    ) -> None:
        super().__init__()
        self.base = carrier.ComplexLoss(
            scaler,
            amplitude_scale,
            lambda_amplitude,
            lambda_phase,
            lambda_frequency,
        )
        self.lambda_n2 = float(lambda_n2)
        self.lambda_n7 = float(lambda_n7)
        self.register_buffer(
            "selected_scale",
            torch.as_tensor(amplitude_scale[:2], dtype=torch.float32),
        )

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        base_loss, terms = self.base(prediction, target)
        pred = self.base.phi_ri(prediction)[..., :2, :]
        truth = self.base.phi_ri(target)[..., :2, :]
        epsilon = 1.0e-8
        pred_amplitude = torch.sqrt(torch.sum(pred.square(), dim=-1) + epsilon)
        truth_amplitude = torch.sqrt(torch.sum(truth.square(), dim=-1) + epsilon)
        pred_scaled = torch.log1p(pred_amplitude / self.selected_scale)
        truth_scaled = torch.log1p(truth_amplitude / self.selected_scale)
        n2 = nn.functional.smooth_l1_loss(
            pred_scaled[..., 0], truth_scaled[..., 0], beta=0.05
        )
        n7 = nn.functional.smooth_l1_loss(
            pred_scaled[..., 1], truth_scaled[..., 1], beta=0.05
        )
        total = base_loss + self.lambda_n2 * n2 + self.lambda_n7 * n7
        return total, {**terms, "n2_amplitude": n2, "n7_amplitude": n7}


def best_period(
    time_us: np.ndarray,
    amplitude: np.ndarray,
    period_min_us: float = 0.6,
    period_max_us: float = 2.5,
    step_us: float = 0.001,
) -> dict:
    periods = np.arange(period_min_us, period_max_us + 0.5 * step_us, step_us)
    centered = time_us - np.mean(time_us)
    scores = []
    for period in periods:
        phase = 2.0 * np.pi * (time_us - controlled.STEP_TIME_US) / period
        design = np.column_stack(
            (np.ones(len(time_us)), centered, np.sin(phase), np.cos(phase))
        )
        prediction = design @ np.linalg.lstsq(design, amplitude, rcond=None)[0]
        scores.append(
            np.mean((amplitude - prediction) ** 2)
            / max(np.var(amplitude), np.finfo(float).tiny)
        )
    index = int(np.argmin(scores))
    return {
        "period_us": float(periods[index]),
        "variance_explained": float(1.0 - scores[index]),
    }


def audit_period_drift(
    trajectory: stage2.Trajectory,
    representation: carrier.CarrierRepresentation,
) -> dict:
    packed = representation.groups[trajectory.name]["carrier_physical"]
    shaped = packed.reshape(
        len(packed), len(direct.FIELD_NAMES), len(carrier.SELECTED_MODE_NUMBERS), 2
    )
    amplitude = np.sqrt(np.sum(np.square(shaped[:, 0]), axis=-1))
    intervals = {
        "training": trajectory.time_us < 35.0 - 1.0e-10,
        "development": trajectory.time_us >= 35.0 - 1.0e-10,
    }
    result = {}
    for interval, mask in intervals.items():
        result[interval] = {}
        for mode in (2, 7):
            index = int(np.flatnonzero(carrier.SELECTED_MODE_NUMBERS == mode)[0])
            result[interval][f"n{mode}"] = best_period(
                trajectory.time_us[mask], amplitude[mask, index]
            )
        result[interval]["time_us"] = [
            float(trajectory.time_us[mask][0]),
            float(trajectory.time_us[mask][-1]),
        ]
    result["development_used_as_control"] = False
    result["primary_data_loaded"] = False
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=modulation.DEFAULT_BASE)
    parser.add_argument("--low", type=Path, default=carrier.DEFAULT_LOW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--history-steps", type=int, default=40)
    parser.add_argument("--rollout-steps", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--delta-limit", type=float, default=0.20)
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--finetune-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--pretrain-learning-rate", type=float, default=8.0e-4)
    parser.add_argument("--finetune-learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--lambda-amplitude", type=float, default=0.20)
    parser.add_argument("--lambda-phase", type=float, default=0.05)
    parser.add_argument("--lambda-frequency", type=float, default=0.10)
    parser.add_argument("--lambda-n2", type=float, default=0.35)
    parser.add_argument("--lambda-n7", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    controlled.seed_everything(args.seed)
    print(f"device={device}", flush=True)
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
    print("[1/6] Fitting representation and auditing nonstationary periods", flush=True)
    representation = carrier.fit_representation(trajectories, fit_masks)
    carrier.save_representation(output / "representation.h5", representation)
    period_audit = audit_period_drift(up, representation)
    (output / "mode_period_drift_audit.json").write_text(
        json.dumps(json_safe(period_audit), indent=2), encoding="utf-8"
    )
    print(json.dumps(json_safe(period_audit), indent=2), flush=True)
    controls = {
        "e25_stationary": controlled.controls_for(e25, 25.0, 25.0, False),
        "e20_to_e22p5": controlled.controls_for(up, 22.5, 20.0, True),
    }
    pretrain_data = controlled.ControlledWindowDataset(
        [(e25, representation.states["e25_stationary"], controls["e25_stationary"], 12.0, 24.0, 2)],
        args.history_steps,
        args.rollout_steps,
    )
    train_data = controlled.ControlledWindowDataset(
        [
            (e25, representation.states["e25_stationary"], controls["e25_stationary"], 12.0, 24.0, 2),
            (up, representation.states["e20_to_e22p5"], controls["e20_to_e22p5"], float(up.time_us[0]), 35.0, 1),
        ],
        args.history_steps,
        args.rollout_steps,
    )
    complex_loss = ModeBalancedComplexLoss(
        representation.scaler,
        representation.amplitude_scale,
        args.lambda_amplitude,
        args.lambda_phase,
        args.lambda_frequency,
        args.lambda_n2,
        args.lambda_n7,
    ).to(device)
    model = ModeSeparatedControlledDelayROM(
        carrier.STATE_DIMENSION, CONTROL_DIMENSION, args.hidden_dim, args.delta_limit
    ).to(device)
    print(
        f"state_dim={carrier.STATE_DIMENSION} controls={CONTROL_DIMENSION} "
        f"windows: pretrain={len(pretrain_data)} train={len(train_data)}",
        flush=True,
    )
    print("[2/6] Pretraining shared memory and separated heads", flush=True)
    pretrain_rows = controlled.train_pretrain(
        model, pretrain_data, representation, complex_loss, args, device
    )
    validation_start = int(np.flatnonzero(up.time_us >= 35.0 - 1.0e-10)[0])
    state = representation.states["e20_to_e22p5"]
    state_history = state[validation_start - args.history_steps:validation_start]
    control_history = controls["e20_to_e22p5"][validation_start - args.history_steps:validation_start]
    truth = state[validation_start:]
    future_controls = controls["e20_to_e22p5"][validation_start:]
    persistence = np.repeat(state_history[-1:], len(truth), axis=0)
    base = modulation.load_base(args.base.resolve())
    if not np.array_equal(base["frame"], up.frame[validation_start:]):
        raise ValueError("Base factorized rollout alignment mismatch")
    print("[3/6] Fine-tuning with common raw-persistence selection", flush=True)
    model, best_epoch, metrics, prediction, finetune_rows = modulation.train_transition(
        model,
        train_data,
        representation,
        complex_loss,
        truth,
        persistence,
        state_history,
        control_history,
        future_controls,
        base,
        validation_start,
        args,
        device,
    )
    print(json.dumps(json_safe(metrics), indent=2), flush=True)
    accepted = bool(
        metrics["passes_all_persistence_gates"]
        and metrics["passes_field_climatology_gate"]
    )
    status = (
        "READY_FOR_PHYSICS_ABLATION"
        if accepted
        else "PROVISIONAL_DATA_ONLY_DIAGNOSTIC"
        if metrics["passes_all_persistence_gates"]
        else "REJECTED_DEVELOPMENT"
    )
    print("[4/6] Saving checkpoint and rollout", flush=True)
    checkpoint = output / "mode_separated_controlled_carrier_data_only.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_dimension": carrier.STATE_DIMENSION,
            "control_dimension": CONTROL_DIMENSION,
            "hidden_dimension": args.hidden_dim,
            "delta_limit": args.delta_limit,
            "history_steps": args.history_steps,
            "rollout_steps": args.rollout_steps,
            "best_epoch": best_epoch,
            "head_groups": {"n2": 8, "n7": 8, "rest": carrier.STATE_DIMENSION - 16},
        },
        checkpoint,
    )
    write_csv(output / "training_history.csv", pretrain_rows + finetune_rows)
    write_csv(output / "development_metrics.csv", [metrics])
    (output / "development_metrics.json").write_text(
        json.dumps(json_safe(metrics), indent=2), encoding="utf-8"
    )
    full_indices = np.arange(validation_start, validation_start + len(truth))
    prediction_coeff = modulation.decode_coefficients(
        prediction, representation, full_indices
    )
    with h5py.File(output / "development_rollout_35to40us.h5", "w") as handle:
        handle.attrs["primary_E25_to_E22p5_loaded"] = False
        handle.attrs["fixed_modulation_clock_used"] = False
        handle.create_dataset("time_us", data=up.time_us[validation_start:])
        handle.create_dataset("frame", data=up.frame[validation_start:])
        handle.create_dataset("selected_mode_numbers", data=base["mode_numbers"])
        handle.create_dataset("truth_state", data=truth, compression="gzip")
        handle.create_dataset("prediction_state", data=prediction, compression="gzip")
        handle.create_dataset("carrier_persistence_state", data=persistence, compression="gzip")
        handle.create_dataset("future_controls", data=future_controls)
        handle.create_dataset("truth_physical_coefficients", data=base["truth"]["selected_physical_coefficients"], compression="gzip")
        handle.create_dataset("prediction_physical_coefficients", data=prediction_coeff, compression="gzip")
        handle.create_dataset("raw_persistence_physical_coefficients", data=base["persistence"]["selected_physical_coefficients"], compression="gzip")
        handle.create_dataset("truth_transport", data=base["truth"]["transport"])
        handle.create_dataset("prediction_transport", data=base["prediction"]["transport"])
        handle.create_dataset("persistence_transport", data=base["persistence"]["transport"])
    modulation.plot_rollout(
        output / "development_rollout_35to40us.png", base, prediction_coeff
    )
    print("[5/6] Writing prospective lock", flush=True)
    base_lock = args.base.resolve() / "model_lock.json"
    low_checkpoint = args.low.resolve() / "regime_aware_transition_rom_data_only.pt"
    direct_checkpoint = ROOT / "workdirs" / "train_radaz_direct_physical_state_rom" / "direct_physical_state_rom_data_only.pt"
    lock = {
        "status": status,
        "accepted_for_physics_ablation": accepted,
        "development_metrics": metrics,
        "period_drift_audit": period_audit,
        "architecture": {
            "shared_GRU": True,
            "n2_head": True,
            "n7_head": True,
            "rest_head": True,
            "fixed_modulation_clock": False,
            "single_checkpoint": True,
        },
        "known_controls": {
            "current_Ez": True,
            "source_Ez_before_step": True,
            "elapsed_time_since_step": True,
            "future_PIC_state": False,
        },
        "component_roles": {
            "single_mode_separated_carrier": "all selected field modes n2,n7--n21",
            "locked_low_branch": "radial and MTSI transport",
            "locked_direct_branch": "ECDI transport",
        },
        "primary_test": {
            "direction": "E25_to_E22.5",
            "data_loaded": False,
            "used_for_selection": False,
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "low_checkpoint": str(low_checkpoint),
        "low_checkpoint_sha256": sha256(low_checkpoint),
        "direct_checkpoint": str(direct_checkpoint),
        "direct_checkpoint_sha256": sha256(direct_checkpoint),
        "base_model_lock": str(base_lock),
        "base_model_lock_sha256": sha256(base_lock),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "arguments": vars(args),
    }
    (output / "model_lock.json").write_text(
        json.dumps(json_safe(lock), indent=2), encoding="utf-8"
    )
    readme = f"""# Mode-separated controlled carrier ROM

- Status: `{status}`
- Best epoch: {best_epoch}
- phi/Ey skill vs raw persistence: {metrics['selected_phi_skill_vs_raw_persistence']:.6f}/{metrics['selected_efy_skill_vs_raw_persistence']:.6f}
- n=2/n=7 amplitude skill: {metrics['phi_n2_amplitude_skill']:.6f}/{metrics['phi_n7_amplitude_skill']:.6f}
- phi/Ey NRMSE: {metrics['selected_phi_nrmse']:.6f}/{metrics['selected_efy_nrmse']:.6f}
- Primary E25 -> E22.5 data loaded: **no**
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print("[6/6] Complete", flush=True)
    print(f"status={status}", flush=True)
    print(f"output={output}", flush=True)


if __name__ == "__main__":
    main()
