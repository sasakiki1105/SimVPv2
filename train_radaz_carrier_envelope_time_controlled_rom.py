"""Time/source-controlled carrier-envelope ROM for the RadAz sweep.

The state representation and complex losses come from
``train_radaz_carrier_envelope_high_branch_rom``.  The propagator additionally
receives three known controls at every step: current Ez, source Ez before the
step, and elapsed time since the step.  No future PIC state is supplied and the
E25 -> E22.5 primary trajectory remains unread.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import train_radaz_carrier_envelope_high_branch_rom as carrier
import train_radaz_direct_physical_state_rom as direct
import train_radaz_regime_aware_transition_rom as stage2


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT / "workdirs" / "train_radaz_carrier_envelope_time_controlled_rom"
)
CONTROL_DIMENSION = 3
STEP_TIME_US = 30.0
TIME_SCALE_US = 5.0


class ControlledDelayROM(nn.Module):
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
        self.head = nn.Sequential(
            nn.Linear(
                hidden_dimension + state_dimension + control_dimension,
                hidden_dimension,
            ),
            nn.SiLU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, state_dimension),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

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
            delta = self.delta_limit * torch.tanh(self.head(context))
            current = current + delta
            recurrent_input = torch.cat((current, control), dim=-1)[:, None]
            _, hidden = self.gru(recurrent_input, hidden)
            predictions.append(current)
        return torch.stack(predictions, dim=1)


class ControlledWindowDataset(Dataset):
    def __init__(
        self,
        specifications: list[
            tuple[
                stage2.Trajectory,
                np.ndarray,
                np.ndarray,
                float,
                float,
                int,
            ]
        ],
        history_steps: int,
        rollout_steps: int,
    ) -> None:
        histories = []
        history_controls = []
        targets = []
        future_controls = []
        for trajectory, state, controls, start_us, end_us, stride in specifications:
            starts = np.flatnonzero(
                (trajectory.time_us >= start_us - 1.0e-10)
                & (trajectory.time_us < end_us - 1.0e-10)
            )
            for target_start in starts[::stride]:
                history_start = int(target_start) - history_steps
                target_stop = int(target_start) + rollout_steps
                if history_start < 0 or target_stop > len(state):
                    continue
                if trajectory.time_us[target_stop - 1] >= end_us - 1.0e-10:
                    continue
                histories.append(state[history_start:target_start])
                history_controls.append(controls[history_start:target_start])
                targets.append(state[target_start:target_stop])
                future_controls.append(controls[target_start:target_stop])
        if not histories:
            raise ValueError("No controlled rollout windows were constructed")
        self.histories = torch.from_numpy(np.asarray(histories, dtype=np.float32))
        self.history_controls = torch.from_numpy(
            np.asarray(history_controls, dtype=np.float32)
        )
        self.targets = torch.from_numpy(np.asarray(targets, dtype=np.float32))
        self.future_controls = torch.from_numpy(
            np.asarray(future_controls, dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.histories)

    def __getitem__(self, index: int):
        return (
            self.histories[index],
            self.history_controls[index],
            self.targets[index],
            self.future_controls[index],
        )


def json_safe(value):
    return carrier.json_safe(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for piece in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def controls_for(
    trajectory: stage2.Trajectory,
    current_ez_kvm: float,
    source_ez_kvm: float,
    transition: bool,
) -> np.ndarray:
    current = np.full(
        len(trajectory.time_us),
        (current_ez_kvm - stage2.PARAMETER_CENTER_KVM)
        / stage2.PARAMETER_SCALE_KVM,
    )
    source = np.full(
        len(trajectory.time_us),
        (source_ez_kvm - stage2.PARAMETER_CENTER_KVM)
        / stage2.PARAMETER_SCALE_KVM,
    )
    elapsed = (
        (trajectory.time_us - STEP_TIME_US) / TIME_SCALE_US
        if transition
        else np.zeros(len(trajectory.time_us), dtype=np.float64)
    )
    return np.column_stack((current, source, elapsed))


def total_loss(
    model: ControlledDelayROM,
    state_history: torch.Tensor,
    control_history: torch.Tensor,
    target: torch.Tensor,
    future_controls: torch.Tensor,
    representation: carrier.CarrierRepresentation,
    complex_loss: carrier.ComplexLoss,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction = model.rollout(
        state_history, control_history, future_controls
    )
    data = carrier.group_loss(prediction, target, representation.scaler)
    complex_term, terms = complex_loss(prediction, target)
    return data + complex_term, {"data": data, "complex": complex_term, **terms}


@torch.no_grad()
def rollout(
    model: ControlledDelayROM,
    state_history: np.ndarray,
    control_history: np.ndarray,
    future_controls: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    state_tensor = torch.as_tensor(
        state_history[None], dtype=torch.float32, device=device
    )
    history_control_tensor = torch.as_tensor(
        control_history[None], dtype=torch.float32, device=device
    )
    future_control_tensor = torch.as_tensor(
        future_controls[None], dtype=torch.float32, device=device
    )
    return model.rollout(
        state_tensor, history_control_tensor, future_control_tensor
    )[0].cpu().numpy().astype(np.float64)


def train_pretrain(
    model: ControlledDelayROM,
    dataset: ControlledWindowDataset,
    representation: carrier.CarrierRepresentation,
    complex_loss: carrier.ComplexLoss,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict]:
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.pretrain_learning_rate,
        weight_decay=args.weight_decay,
    )
    rows = []
    for epoch in range(1, args.pretrain_epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for state_history, control_history, target, future_controls in loader:
            state_history = state_history.to(device)
            control_history = control_history.to(device)
            target = target.to(device)
            future_controls = future_controls.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = total_loss(
                model,
                state_history,
                control_history,
                target,
                future_controls,
                representation,
                complex_loss,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(state_history)
            count += len(state_history)
        rows.append({"stage": "pretrain", "epoch": epoch, "loss": total / count})
        if epoch == 1 or epoch % 10 == 0:
            print(f"pretrain epoch={epoch:03d} loss={total / count:.6e}", flush=True)
    return rows


def train_transition(
    model: ControlledDelayROM,
    dataset: ControlledWindowDataset,
    representation: carrier.CarrierRepresentation,
    complex_loss: carrier.ComplexLoss,
    truth: np.ndarray,
    persistence: np.ndarray,
    state_history: np.ndarray,
    control_history: np.ndarray,
    future_controls: np.ndarray,
    low: dict,
    validation_start: int,
    dt_us: float,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ControlledDelayROM, int, dict, np.ndarray, list[dict]]:
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 1),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.finetune_learning_rate,
        weight_decay=args.weight_decay,
    )
    full_indices = np.arange(validation_start, validation_start + len(truth))
    initial_prediction = rollout(
        model, state_history, control_history, future_controls, device
    )
    best_metrics = carrier.evaluate(
        truth,
        initial_prediction,
        persistence,
        representation,
        low,
        full_indices,
        dt_us,
    )
    best_score = best_metrics["minimum_gate_skill"]
    best_state = copy.deepcopy(model.state_dict())
    best_prediction = initial_prediction
    best_epoch = 0
    stale = 0
    rows = []
    for epoch in range(1, args.finetune_epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for current_state, current_control, target, target_control in loader:
            current_state = current_state.to(device)
            current_control = current_control.to(device)
            target = target.to(device)
            target_control = target_control.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = total_loss(
                model,
                current_state,
                current_control,
                target,
                target_control,
                representation,
                complex_loss,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(current_state)
            count += len(current_state)
        prediction = rollout(
            model, state_history, control_history, future_controls, device
        )
        metrics = carrier.evaluate(
            truth,
            prediction,
            persistence,
            representation,
            low,
            full_indices,
            dt_us,
        )
        score = metrics["minimum_gate_skill"]
        rows.append({
            "stage": "finetune",
            "epoch": epoch,
            "loss": total / count,
            "minimum_gate_skill": score,
            "ECDI_transport_skill": metrics["ECDI_n9_21_transport_skill"],
            "phi_skill": metrics[
                "selected_phi_skill_vs_carrier_persistence"
            ],
            "Ey_skill": metrics[
                "selected_efy_skill_vs_carrier_persistence"
            ],
            "n2_amplitude_skill": metrics["phi_n2_amplitude_skill"],
            "n7_amplitude_skill": metrics["phi_n7_amplitude_skill"],
        })
        if score > best_score + 1.0e-5:
            best_score = score
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())
            best_prediction = prediction
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"finetune epoch={epoch:03d} loss={total / count:.6e} "
                f"min_skill={score:.6e} ECDI={metrics['ECDI_n9_21_transport_skill']:.6e} "
                f"n2={metrics['phi_n2_amplitude_skill']:.6e} "
                f"n7={metrics['phi_n7_amplitude_skill']:.6e}",
                flush=True,
            )
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    return model, best_epoch, best_metrics, best_prediction, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low", type=Path, default=carrier.DEFAULT_LOW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--history-steps", type=int, default=40)
    parser.add_argument("--rollout-steps", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--delta-limit", type=float, default=0.20)
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--pretrain-learning-rate", type=float, default=8.0e-4)
    parser.add_argument("--finetune-learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--lambda-amplitude", type=float, default=0.20)
    parser.add_argument("--lambda-phase", type=float, default=0.05)
    parser.add_argument("--lambda-frequency", type=float, default=0.10)
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
    seed_everything(args.seed)
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
    }
    e25 = trajectories["e25_stationary"]
    up = trajectories["e20_to_e22p5"]
    fit_masks = {
        "e25_stationary": stage2.interval_mask(e25.time_us, 12.0, 24.0),
        "e20_to_e22p5": up.time_us < 35.0 - 1.0e-10,
    }
    print("[1/6] Rebuilding carrier representation and known controls", flush=True)
    representation = carrier.fit_representation(trajectories, fit_masks)
    carrier.save_representation(output / "representation.h5", representation)
    controls = {
        "e25_stationary": controls_for(e25, 25.0, 25.0, False),
        "e20_to_e22p5": controls_for(up, 22.5, 20.0, True),
    }

    pretrain_data = ControlledWindowDataset(
        [(
            e25,
            representation.states["e25_stationary"],
            controls["e25_stationary"],
            12.0,
            24.0,
            2,
        )],
        args.history_steps,
        args.rollout_steps,
    )
    train_data = ControlledWindowDataset(
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
    )
    complex_loss = carrier.ComplexLoss(
        representation.scaler,
        representation.amplitude_scale,
        args.lambda_amplitude,
        args.lambda_phase,
        args.lambda_frequency,
    ).to(device)
    model = ControlledDelayROM(
        carrier.STATE_DIMENSION,
        CONTROL_DIMENSION,
        args.hidden_dim,
        args.delta_limit,
    ).to(device)
    print(
        f"state_dim={carrier.STATE_DIMENSION} control_dim={CONTROL_DIMENSION} "
        f"windows: pretrain={len(pretrain_data)} train={len(train_data)}",
        flush=True,
    )
    print("[2/6] Pretraining stationary E25 controlled envelopes", flush=True)
    pretrain_rows = train_pretrain(
        model, pretrain_data, representation, complex_loss, args, device
    )

    validation_start = int(np.flatnonzero(up.time_us >= 35.0 - 1.0e-10)[0])
    state = representation.states["e20_to_e22p5"]
    state_history = state[validation_start - args.history_steps:validation_start]
    control_history = controls["e20_to_e22p5"][
        validation_start - args.history_steps:validation_start
    ]
    truth = state[validation_start:]
    future_controls = controls["e20_to_e22p5"][validation_start:]
    persistence = np.repeat(state_history[-1:], len(truth), axis=0)
    low = carrier.load_low_branch(args.low.resolve())
    dt_us = float(np.median(np.diff(up.time_us)))
    print("[3/6] Fine-tuning with source-Ez and elapsed-time controls", flush=True)
    model, best_epoch, metrics, prediction, finetune_rows = train_transition(
        model,
        train_data,
        representation,
        complex_loss,
        truth,
        persistence,
        state_history,
        control_history,
        future_controls,
        low,
        validation_start,
        dt_us,
        args,
        device,
    )
    print(json.dumps(json_safe(metrics), indent=2), flush=True)
    accepted = bool(
        metrics["finite_fraction"] == 1.0
        and metrics["minimum_gate_skill"] > 0.0
        and metrics["radial_skill_vs_persistence"] > 0.0
        and metrics["MTSI_n1_6_transport_skill"] > 0.0
    )
    status = "READY_FOR_PHYSICS_ABLATION" if accepted else "REJECTED_DEVELOPMENT"

    print("[4/6] Saving controlled carrier checkpoint and rollout", flush=True)
    checkpoint = output / "carrier_envelope_time_controlled_data_only.pt"
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
            "control_definition": (
                "[(current_Ez-25)/5, (source_Ez-25)/5, "
                "(time_us-30)/5]"
            ),
        },
        checkpoint,
    )
    write_csv(output / "training_history.csv", pretrain_rows + finetune_rows)
    write_csv(output / "development_metrics.csv", [metrics])
    (output / "development_metrics.json").write_text(
        json.dumps(json_safe(metrics), indent=2), encoding="utf-8"
    )
    with h5py.File(output / "development_rollout_35to40us.h5", "w") as handle:
        handle.attrs["primary_E25_to_E22p5_loaded"] = False
        handle.create_dataset("time_us", data=up.time_us[validation_start:])
        handle.create_dataset("frame", data=up.frame[validation_start:])
        handle.create_dataset("truth", data=truth, compression="gzip")
        handle.create_dataset("prediction", data=prediction, compression="gzip")
        handle.create_dataset("carrier_persistence", data=persistence, compression="gzip")
        handle.create_dataset("future_controls", data=future_controls)
    carrier.plot_rollout(
        output / "development_rollout_35to40us.png",
        up.time_us[validation_start:],
        truth,
        prediction,
        persistence,
        representation,
        validation_start,
    )

    print("[5/6] Writing prospective lock", flush=True)
    lock = {
        "status": status,
        "accepted_for_physics_ablation": accepted,
        "development_metrics": metrics,
        "best_epoch": best_epoch,
        "known_controls": {
            "current_Ez": True,
            "source_Ez_before_step": True,
            "elapsed_time_since_step": True,
            "future_PIC_state": False,
        },
        "primary_test": {
            "direction": "E25_to_E22.5",
            "data_loaded": False,
            "used_for_carrier_fit": False,
            "used_for_selection": False,
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
    readme = f"""# Time-controlled carrier-envelope ROM

- Status: `{status}`
- Best epoch: {best_epoch}
- Composite state skill: {metrics['composite_state_skill_vs_persistence']:.6f}
- ECDI transport skill: {metrics['ECDI_n9_21_transport_skill']:.6f}
- Selected phi skill: {metrics['selected_phi_skill_vs_carrier_persistence']:.6f}
- Selected Ey skill: {metrics['selected_efy_skill_vs_carrier_persistence']:.6f}
- n=2 amplitude skill: {metrics['phi_n2_amplitude_skill']:.6f}
- n=7 amplitude skill: {metrics['phi_n7_amplitude_skill']:.6f}
- Primary E25 -> E22.5 data loaded: **no**
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print("[6/6] Complete", flush=True)
    print(f"status={status}", flush=True)
    print(f"output={output}", flush=True)


if __name__ == "__main__":
    main()
