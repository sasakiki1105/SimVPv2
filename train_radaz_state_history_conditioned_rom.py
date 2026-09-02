"""State- and electric-history-conditioned RadAz transition ROM.

This data-only stage addresses the drifting n=2/n=7 modulation without a
prescribed clock.  A shared recurrent memory receives only causal quantities:

* the complete carrier-envelope state history;
* current/source Ez and a deterministic multi-timescale step-memory vector;
* n=2/n=7 amplitude, phase, phase-increment, and amplitude-increment features
  computed from the state already available to the model.

During free rollout every modal feature is recomputed from predicted states;
no future PIC state is supplied.  Training windows are augmented by exact
azimuthal translations in Fourier space.  This changes spatial phase only and
is not claimed to reproduce an independent temporal-phase transition.  The
primary E25 -> E22.5 trajectory is never read.
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
import train_radaz_mode_separated_controlled_carrier_rom as separated
import train_radaz_modulation_controlled_carrier_rom as modulation
import train_radaz_regime_aware_transition_rom as stage2


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT / "workdirs" / "train_radaz_state_history_conditioned_rom"
)
DEFAULT_DOWN_FEATURES = (
    ROOT
    / "workdirs"
    / "radaz_e22p5_to_e20_transition"
    / "fourier_latent_features.h5"
)
DEFAULT_DOWN_PHYSICAL = (
    ROOT
    / "workdirs"
    / "radaz_e22p5_to_e20_transition"
    / "physical_fourier_targets.h5"
)
CONTROL_DIMENSION = 9
MODAL_FEATURE_DIMENSION = 14
ELECTRIC_MEMORY_TIMES_US = (0.30, 1.50, 5.00)


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


def history_controls_for(
    trajectory: stage2.Trajectory,
    current_ez_kvm: float,
    source_ez_kvm: float,
    transition: bool,
) -> np.ndarray:
    """Known control and electric-field memory, never future PIC state."""
    count = len(trajectory.time_us)
    current = np.full(
        count,
        (current_ez_kvm - stage2.PARAMETER_CENTER_KVM)
        / stage2.PARAMETER_SCALE_KVM,
        dtype=np.float64,
    )
    source = np.full(
        count,
        (source_ez_kvm - stage2.PARAMETER_CENTER_KVM)
        / stage2.PARAMETER_SCALE_KVM,
        dtype=np.float64,
    )
    delta = current - source
    flag = np.full(count, float(transition), dtype=np.float64)
    if transition:
        age_us = np.maximum(trajectory.time_us - controlled.STEP_TIME_US, 0.0)
    else:
        age_us = np.zeros(count, dtype=np.float64)
    age_linear = age_us / controlled.TIME_SCALE_US
    age_log = np.log1p(age_us) / np.log1p(10.0)
    memories = [
        delta * np.exp(-age_us / timescale)
        for timescale in ELECTRIC_MEMORY_TIMES_US
    ]
    result = np.column_stack(
        (current, source, delta, flag, age_linear, age_log, *memories)
    )
    if result.shape[1] != CONTROL_DIMENSION:
        raise AssertionError("Electric-history control dimension mismatch")
    return result


def rotate_azimuth_states(
    states: np.ndarray,
    angle_rad: float,
    representation: carrier.CarrierRepresentation,
) -> np.ndarray:
    """Apply y -> y + angle in the retained Fourier coefficients."""
    values = representation.scaler.inverse(np.asarray(states, dtype=np.float64))
    shaped = values["carrier_physical"].reshape(
        len(states),
        len(direct.FIELD_NAMES),
        len(carrier.SELECTED_MODE_NUMBERS),
        2,
    )
    coefficients = shaped[..., 0] + 1j * shaped[..., 1]
    rotation = np.exp(
        1j * np.asarray(carrier.SELECTED_MODE_NUMBERS, dtype=np.float64) * angle_rad
    )
    rotated = coefficients * rotation[None, None, :]
    values["carrier_physical"] = np.stack(
        (rotated.real, rotated.imag), axis=-1
    ).reshape(len(states), -1)
    return representation.scaler.transform(values)


def augmentation_audit(
    states: np.ndarray,
    representation: carrier.CarrierRepresentation,
    shifts: int,
) -> dict:
    angles = 2.0 * np.pi * np.arange(shifts, dtype=np.float64) / shifts
    sample = states[:: max(1, len(states) // 64)]
    roundtrip = []
    amplitude_absolute = []
    amplitude_relative = []
    original = representation.scaler.inverse(sample)["carrier_physical"]
    original_ri = original.reshape(
        len(sample), len(direct.FIELD_NAMES), len(carrier.SELECTED_MODE_NUMBERS), 2
    )
    original_amplitude = np.sqrt(np.sum(original_ri**2, axis=-1))
    for angle in angles:
        shifted = rotate_azimuth_states(sample, float(angle), representation)
        restored = rotate_azimuth_states(shifted, float(-angle), representation)
        shifted_raw = representation.scaler.inverse(shifted)["carrier_physical"]
        shifted_ri = shifted_raw.reshape(original_ri.shape)
        shifted_amplitude = np.sqrt(np.sum(shifted_ri**2, axis=-1))
        roundtrip.append(float(np.max(np.abs(restored - sample))))
        difference = np.abs(shifted_amplitude - original_amplitude)
        amplitude_absolute.append(float(np.max(difference)))
        amplitude_relative.append(
            float(
                np.max(
                    difference
                    / np.maximum(original_amplitude, np.finfo(float).tiny)
                )
            )
        )
    return {
        "number_of_azimuth_shifts": int(shifts),
        "angles_rad": angles.tolist(),
        "maximum_normalized_roundtrip_error": float(max(roundtrip)),
        "maximum_physical_amplitude_absolute_error": float(
            max(amplitude_absolute)
        ),
        "maximum_physical_amplitude_relative_error": float(
            max(amplitude_relative)
        ),
        "spatial_phase_only": True,
        "independent_temporal_phase_claimed": False,
    }


def augmented_dataset(
    specifications: list[
        tuple[stage2.Trajectory, np.ndarray, np.ndarray, float, float, int]
    ],
    history_steps: int,
    rollout_steps: int,
    representation: carrier.CarrierRepresentation,
    shifts: int,
) -> controlled.ControlledWindowDataset:
    expanded = []
    for trajectory, states, controls, start_us, end_us, stride in specifications:
        for shift in range(shifts):
            angle = 2.0 * np.pi * shift / shifts
            expanded.append(
                (
                    trajectory,
                    rotate_azimuth_states(states, angle, representation),
                    controls,
                    start_us,
                    end_us,
                    stride,
                )
            )
    return controlled.ControlledWindowDataset(
        expanded, history_steps, rollout_steps
    )


class CausalModalFeatures(nn.Module):
    """Differentiable n=2/n=7 descriptors from current and past states."""

    def __init__(self, representation: carrier.CarrierRepresentation) -> None:
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
        local_indices = [
            int(np.flatnonzero(carrier.SELECTED_MODE_NUMBERS == mode)[0])
            for mode in (2, 7)
        ]
        self.register_buffer(
            "mode_indices", torch.as_tensor(local_indices, dtype=torch.long)
        )
        self.register_buffer(
            "amplitude_scale",
            torch.as_tensor(
                representation.amplitude_scale[local_indices], dtype=torch.float32
            ),
        )

    def phi(self, states: torch.Tensor) -> torch.Tensor:
        raw = states[..., self.slice_start : self.slice_stop] * self.scale + self.mean
        shaped = raw.reshape(
            *raw.shape[:-1],
            len(direct.FIELD_NAMES),
            len(carrier.SELECTED_MODE_NUMBERS),
            2,
        )
        phi = shaped[..., direct.FIELD_NAMES.index("phi"), :, :]
        return torch.index_select(phi, dim=-2, index=self.mode_indices)

    def forward(
        self, current: torch.Tensor, previous: torch.Tensor
    ) -> torch.Tensor:
        epsilon = 1.0e-8
        current_phi = self.phi(current)
        previous_phi = self.phi(previous)
        current_amplitude = torch.sqrt(
            torch.sum(current_phi.square(), dim=-1) + epsilon
        )
        previous_amplitude = torch.sqrt(
            torch.sum(previous_phi.square(), dim=-1) + epsilon
        )
        log_amplitude = torch.log1p(
            current_amplitude / self.amplitude_scale
        )
        previous_log_amplitude = torch.log1p(
            previous_amplitude / self.amplitude_scale
        )
        unit_phase = current_phi / current_amplitude[..., None]
        dot = torch.sum(current_phi * previous_phi, dim=-1)
        cross = (
            current_phi[..., 1] * previous_phi[..., 0]
            - current_phi[..., 0] * previous_phi[..., 1]
        )
        pair_norm = current_amplitude * previous_amplitude + epsilon
        phase_increment = torch.stack((dot / pair_norm, cross / pair_norm), dim=-1)
        amplitude_increment = 20.0 * (
            log_amplitude - previous_log_amplitude
        )
        balance = (log_amplitude[..., 0] - log_amplitude[..., 1])[..., None]
        total = torch.log1p(
            (current_amplitude[..., 0] + current_amplitude[..., 1])
            / torch.sum(self.amplitude_scale)
        )[..., None]
        result = torch.cat(
            (
                log_amplitude,
                unit_phase.flatten(start_dim=-2),
                phase_increment.flatten(start_dim=-2),
                amplitude_increment,
                balance,
                total,
            ),
            dim=-1,
        )
        if result.shape[-1] != MODAL_FEATURE_DIMENSION:
            raise AssertionError("Causal modal feature dimension mismatch")
        return result


class StateHistoryConditionedROM(nn.Module):
    """Mode-separated residual ROM with explicit causal modal history."""

    def __init__(
        self,
        representation: carrier.CarrierRepresentation,
        state_dimension: int,
        control_dimension: int,
        hidden_dimension: int,
        delta_limit: float,
    ) -> None:
        super().__init__()
        self.state_dimension = int(state_dimension)
        self.control_dimension = int(control_dimension)
        self.hidden_dimension = int(hidden_dimension)
        self.delta_limit = float(delta_limit)
        self.features = CausalModalFeatures(representation)
        recurrent_dimension = (
            state_dimension + control_dimension + MODAL_FEATURE_DIMENSION
        )
        self.gru = nn.GRU(
            recurrent_dimension, hidden_dimension, batch_first=True
        )
        n2 = separated.physical_mode_state_indices(2)
        n7 = separated.physical_mode_state_indices(7)
        rest = np.setdiff1d(
            np.arange(state_dimension), np.concatenate((n2, n7))
        )
        self.register_buffer("n2_indices", torch.as_tensor(n2, dtype=torch.long))
        self.register_buffer("n7_indices", torch.as_tensor(n7, dtype=torch.long))
        self.register_buffer(
            "rest_indices", torch.as_tensor(rest, dtype=torch.long)
        )
        context_dimension = recurrent_dimension + hidden_dimension
        self.n2_head = separated.ModeSeparatedControlledDelayROM._make_head(
            context_dimension, len(n2), hidden_dimension
        )
        self.n7_head = separated.ModeSeparatedControlledDelayROM._make_head(
            context_dimension, len(n7), hidden_dimension
        )
        self.rest_head = separated.ModeSeparatedControlledDelayROM._make_head(
            context_dimension, len(rest), hidden_dimension
        )

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
        previous_history = torch.cat(
            (state_history[:, :1], state_history[:, :-1]), dim=1
        )
        feature_history = self.features(state_history, previous_history)
        recurrent_history = torch.cat(
            (state_history, control_history, feature_history), dim=-1
        )
        _, hidden = self.gru(recurrent_history)
        current = state_history[:, -1]
        previous = (
            state_history[:, -2] if state_history.shape[1] > 1 else current
        )
        dynamic = self.features(current, previous)
        predictions = []
        for step in range(future_controls.shape[1]):
            control = future_controls[:, step]
            context = torch.cat((hidden[-1], current, control, dynamic), dim=-1)
            delta = torch.zeros_like(current)
            delta = self._scatter(delta, self.n2_indices, self.n2_head(context))
            delta = self._scatter(delta, self.n7_indices, self.n7_head(context))
            delta = self._scatter(
                delta, self.rest_indices, self.rest_head(context)
            )
            next_state = current + self.delta_limit * torch.tanh(delta)
            next_dynamic = self.features(next_state, current)
            recurrent_input = torch.cat(
                (next_state, control, next_dynamic), dim=-1
            )[:, None]
            _, hidden = self.gru(recurrent_input, hidden)
            predictions.append(next_state)
            previous, current, dynamic = current, next_state, next_dynamic
        return torch.stack(predictions, dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=modulation.DEFAULT_BASE)
    parser.add_argument("--low", type=Path, default=carrier.DEFAULT_LOW)
    parser.add_argument("--down-features", type=Path, default=DEFAULT_DOWN_FEATURES)
    parser.add_argument("--down-physical", type=Path, default=DEFAULT_DOWN_PHYSICAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--history-steps", type=int, default=40)
    parser.add_argument("--rollout-steps", type=int, default=80)
    parser.add_argument("--azimuth-shifts", type=int, default=4)
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
    args = parser.parse_args()
    if args.azimuth_shifts < 1:
        parser.error("--azimuth-shifts must be positive")
    return args


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    controlled.seed_everything(args.seed)
    print(f"device={device}", flush=True)

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
        "e22p5_to_e20": stage2.load_trajectory(
            "e22p5_to_e20",
            20.0,
            args.down_features,
            args.down_physical,
        ),
    }
    e25 = trajectories["e25_stationary"]
    up = trajectories["e20_to_e22p5"]
    down = trajectories["e22p5_to_e20"]
    fit_masks = {
        "e25_stationary": stage2.interval_mask(e25.time_us, 12.0, 24.0),
        "e20_to_e22p5": up.time_us < 35.0 - 1.0e-10,
        "e22p5_to_e20": down.time_us < 34.86 - 1.0e-10,
    }

    print("[1/7] Fitting the locked carrier representation", flush=True)
    representation = carrier.fit_representation(trajectories, fit_masks)
    carrier.save_representation(output / "representation.h5", representation)
    period_audit = separated.audit_period_drift(up, representation)
    (output / "mode_period_drift_audit.json").write_text(
        json.dumps(json_safe(period_audit), indent=2), encoding="utf-8"
    )
    audit = augmentation_audit(
        representation.states["e20_to_e22p5"],
        representation,
        args.azimuth_shifts,
    )
    (output / "azimuth_augmentation_audit.json").write_text(
        json.dumps(json_safe(audit), indent=2), encoding="utf-8"
    )
    print(json.dumps(json_safe(audit), indent=2), flush=True)

    controls = {
        "e25_stationary": history_controls_for(e25, 25.0, 25.0, False),
        "e20_to_e22p5": history_controls_for(up, 22.5, 20.0, True),
        "e22p5_to_e20": history_controls_for(down, 20.0, 22.5, True),
    }
    pretrain_data = augmented_dataset(
        [
            (
                e25,
                representation.states["e25_stationary"],
                controls["e25_stationary"],
                12.0,
                24.0,
                2,
            )
        ],
        args.history_steps,
        args.rollout_steps,
        representation,
        args.azimuth_shifts,
    )
    train_data = augmented_dataset(
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
            (
                down,
                representation.states["e22p5_to_e20"],
                controls["e22p5_to_e20"],
                float(down.time_us[0]),
                34.86,
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
        args.lambda_amplitude,
        args.lambda_phase,
        args.lambda_frequency,
        args.lambda_n2,
        args.lambda_n7,
    ).to(device)
    model = StateHistoryConditionedROM(
        representation,
        carrier.STATE_DIMENSION,
        CONTROL_DIMENSION,
        args.hidden_dim,
        args.delta_limit,
    ).to(device)
    print(
        f"state_dim={carrier.STATE_DIMENSION} controls={CONTROL_DIMENSION} "
        f"causal_modal_features={MODAL_FEATURE_DIMENSION} "
        f"windows: pretrain={len(pretrain_data)} train={len(train_data)}",
        flush=True,
    )

    print("[2/7] Pretraining with azimuth-phase augmentation", flush=True)
    pretrain_rows = controlled.train_pretrain(
        model, pretrain_data, representation, complex_loss, args, device
    )

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
    if not np.array_equal(base["frame"], up.frame[validation_start:]):
        raise ValueError("Base factorized rollout alignment mismatch")

    print("[3/7] Fine-tuning on the prospective time block", flush=True)
    model, best_epoch, metrics, prediction, finetune_rows = (
        modulation.train_transition(
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
        if metrics["passes_field_climatology_gate"]
        else "REJECTED_DEVELOPMENT"
    )

    print("[4/7] Saving the data-only checkpoint and rollout", flush=True)
    checkpoint = output / "state_history_conditioned_data_only.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_dimension": carrier.STATE_DIMENSION,
            "control_dimension": CONTROL_DIMENSION,
            "modal_feature_dimension": MODAL_FEATURE_DIMENSION,
            "hidden_dimension": args.hidden_dim,
            "delta_limit": args.delta_limit,
            "history_steps": args.history_steps,
            "rollout_steps": args.rollout_steps,
            "azimuth_shifts": args.azimuth_shifts,
            "best_epoch": best_epoch,
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
        handle.attrs["future_PIC_state_used_as_control"] = False
        handle.attrs["independent_temporal_phase_claimed"] = False
        handle.create_dataset("time_us", data=up.time_us[validation_start:])
        handle.create_dataset("frame", data=up.frame[validation_start:])
        handle.create_dataset(
            "selected_mode_numbers", data=base["mode_numbers"]
        )
        handle.create_dataset("truth_state", data=truth, compression="gzip")
        handle.create_dataset(
            "prediction_state", data=prediction, compression="gzip"
        )
        handle.create_dataset(
            "carrier_persistence_state", data=persistence, compression="gzip"
        )
        handle.create_dataset("future_controls", data=future_controls)
        handle.create_dataset(
            "truth_physical_coefficients",
            data=base["truth"]["selected_physical_coefficients"],
            compression="gzip",
        )
        handle.create_dataset(
            "prediction_physical_coefficients",
            data=prediction_coeff,
            compression="gzip",
        )
        handle.create_dataset(
            "raw_persistence_physical_coefficients",
            data=base["persistence"]["selected_physical_coefficients"],
            compression="gzip",
        )
    modulation.plot_rollout(
        output / "development_rollout_35to40us.png", base, prediction_coeff
    )

    print("[5/7] Writing the prospective model lock", flush=True)
    lock = {
        "status": status,
        "accepted_for_physics_ablation": accepted,
        "development_metrics": metrics,
        "period_drift_audit": period_audit,
        "azimuth_augmentation_audit": audit,
        "architecture": {
            "shared_GRU": True,
            "n2_head": True,
            "n7_head": True,
            "rest_head": True,
            "explicit_causal_modal_history": True,
            "fixed_modulation_clock": False,
        },
        "known_controls": {
            "current_Ez": True,
            "source_Ez_before_step": True,
            "delta_Ez": True,
            "elapsed_time_since_step": True,
            "multi_timescale_Ez_memory_us": list(ELECTRIC_MEMORY_TIMES_US),
            "future_PIC_state": False,
        },
        "split": {
            "transition_training": "30.165 <= t < 35.0 us",
            "reverse_transition_training": "30.165 <= t < 34.86 us",
            "development": "35.0 <= t <= 39.855 us",
            "development_used_for_training": False,
        },
        "augmentation_scope": {
            "azimuthal_spatial_translation": True,
            "temporal_phase_replication": False,
        },
        "primary_test": {
            "direction": "E25_to_E22.5",
            "data_loaded": False,
            "used_for_selection": False,
        },
        "reverse_transition_inputs": {
            "features": str(args.down_features.resolve()),
            "features_sha256": sha256(args.down_features.resolve()),
            "physical": str(args.down_physical.resolve()),
            "physical_sha256": sha256(args.down_physical.resolve()),
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "arguments": vars(args),
    }
    (output / "model_lock.json").write_text(
        json.dumps(json_safe(lock), indent=2), encoding="utf-8"
    )
    readme = f"""# State/history-conditioned RadAz ROM

- Status: `{status}`
- Best epoch: {best_epoch}
- phi/Ey skill vs raw persistence: {metrics['selected_phi_skill_vs_raw_persistence']:.6f}/{metrics['selected_efy_skill_vs_raw_persistence']:.6f}
- n=2/n=7 amplitude skill: {metrics['phi_n2_amplitude_skill']:.6f}/{metrics['phi_n7_amplitude_skill']:.6f}
- phi/Ey NRMSE: {metrics['selected_phi_nrmse']:.6f}/{metrics['selected_efy_nrmse']:.6f}
- Azimuth shifts: {args.azimuth_shifts} (spatial phase only)
- Future PIC state used as input: **no**
- Primary E25 -> E22.5 data loaded: **no**
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print("[6/7] Physics gate", flush=True)
    if accepted:
        print("data-only gate passed; physics ablation may now start", flush=True)
    else:
        print("data-only gate failed; physics weights remain frozen", flush=True)
    print("[7/7] Complete", flush=True)
    print(f"status={status}", flush=True)
    print(f"output={output}", flush=True)


if __name__ == "__main__":
    main()
