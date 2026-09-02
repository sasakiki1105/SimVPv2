"""Train a nonlinear parameter-conditioned delay ROM for RadAz Ez sweeps.

This is the nonlinear follow-up to ``fit_radaz_state_conditioned_parametric_rom``.
It uses the same common 30-dimensional L+R+T state and the same prospective
protocol: E20/E30 train a development model, E25 is the interpolation
validation target, and no E22.5 data are loaded.  A recurrent residual model
is trained with free multi-step rollout loss so that electric-field-dependent
rotation frequencies need not be averaged at the propagator-output level.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import fit_radaz_state_conditioned_parametric_rom as baseline


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT / "workdirs" / "train_radaz_parametric_neural_delay_rom"
)
FIELDS = (20, 25, 30)
DEVELOPMENT_FIELDS = (20, 30)
VALIDATION_FIELD = 25
TRAIN_START_US = 12.0
TRAIN_END_US = 18.0
VALIDATION_START_US = 18.0
VALIDATION_END_US = 20.0
FINAL_TRAIN_END_US = 24.0
FORECAST_START_US = 24.0
FORECAST_END_US = 30.0
PARAMETER_CENTER_KVM = 25.0
PARAMETER_SCALE_KVM = 5.0


@dataclass
class TrainingResult:
    model: "ParametricDelayROM"
    best_epoch: int
    best_validation_loss: float
    history: list[dict]


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return baseline.json_safe(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SequenceDataset(Dataset):
    def __init__(
        self,
        cases: dict[int, baseline.RawCase],
        states: dict[int, np.ndarray],
        fields: tuple[int, ...],
        start_us: float,
        end_us: float,
        history_steps: int,
        rollout_steps: int,
        stride: int,
    ) -> None:
        histories = []
        targets = []
        parameters = []
        provenance = []
        for field in fields:
            case = cases[field]
            current = np.asarray(states[field], dtype=np.float32)
            valid = np.flatnonzero(
                (case.time_us >= start_us - 1.0e-10)
                & (case.time_us < end_us - 1.0e-10)
            )
            if len(valid) == 0:
                continue
            first = int(valid[0]) + history_steps
            last = int(valid[-1]) - rollout_steps + 1
            for target_start in range(first, last + 1, stride):
                history = current[
                    target_start - history_steps : target_start
                ]
                target = current[
                    target_start : target_start + rollout_steps
                ]
                if len(history) != history_steps or len(target) != rollout_steps:
                    continue
                histories.append(history)
                targets.append(target)
                parameters.append(
                    (field - PARAMETER_CENTER_KVM) / PARAMETER_SCALE_KVM
                )
                provenance.append((field, int(case.frame[target_start])))
        if not histories:
            raise ValueError("No training sequences were constructed")
        self.histories = torch.from_numpy(np.stack(histories))
        self.targets = torch.from_numpy(np.stack(targets))
        self.parameters = torch.as_tensor(parameters, dtype=torch.float32)
        self.provenance = provenance

    def __len__(self) -> int:
        return len(self.histories)

    def __getitem__(self, index: int):
        return (
            self.histories[index],
            self.targets[index],
            self.parameters[index],
        )


class ParametricDelayROM(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        delta_limit: float,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.delta_limit = float(delta_limit)
        self.gru = nn.GRU(
            input_size=state_dim + 1,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + state_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        # Starting from persistence makes the first optimization steps and
        # long free rollouts substantially safer.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def parameter_column(
        self, parameter: torch.Tensor, steps: int
    ) -> torch.Tensor:
        return parameter[:, None, None].expand(-1, steps, 1)

    def encode(
        self, history: torch.Tensor, parameter: torch.Tensor
    ) -> torch.Tensor:
        inputs = torch.cat(
            (history, self.parameter_column(parameter, history.shape[1])),
            dim=-1,
        )
        _, hidden = self.gru(inputs)
        return hidden

    def next_state(
        self,
        current: torch.Tensor,
        hidden: torch.Tensor,
        parameter: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = torch.cat(
            (hidden[-1], current, parameter[:, None]), dim=-1
        )
        delta = self.delta_limit * torch.tanh(self.head(context))
        following = current + delta
        recurrent_input = torch.cat(
            (following, parameter[:, None]), dim=-1
        )[:, None, :]
        _, following_hidden = self.gru(recurrent_input, hidden)
        return following, following_hidden

    def rollout(
        self,
        history: torch.Tensor,
        parameter: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        hidden = self.encode(history, parameter)
        current = history[:, -1]
        predictions = []
        for _ in range(steps):
            current, hidden = self.next_state(current, hidden, parameter)
            predictions.append(current)
        return torch.stack(predictions, dim=1)


def batch_loss(
    model: ParametricDelayROM,
    history: torch.Tensor,
    target: torch.Tensor,
    parameter: torch.Tensor,
) -> torch.Tensor:
    prediction = model.rollout(history, parameter, target.shape[1])
    # Huber loss is robust to occasional PIC deposition spikes while retaining
    # a quadratic basin for the typical modal state increments.
    return nn.functional.smooth_l1_loss(
        prediction, target, beta=0.05, reduction="mean"
    )


@torch.no_grad()
def dataset_loss(
    model: ParametricDelayROM,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for history, target, parameter in loader:
        history = history.to(device)
        target = target.to(device)
        parameter = parameter.to(device)
        loss = batch_loss(model, history, target, parameter)
        total += float(loss) * len(history)
        count += len(history)
    return total / max(count, 1)


def train_development(
    train_data: SequenceDataset,
    validation_data: SequenceDataset,
    state_dim: int,
    hidden_dim: int,
    delta_limit: float,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> TrainingResult:
    seed_everything(seed)
    model = ParametricDelayROM(state_dim, hidden_dim, delta_limit).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1.0e-5
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_data, batch_size=batch_size, shuffle=False
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for history, target, parameter in train_loader:
            history = history.to(device)
            target = target.to(device)
            parameter = parameter.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model, history, target, parameter)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(history)
            count += len(history)
        train_loss = total / max(count, 1)
        validation_loss = dataset_loss(model, validation_loader, device)
        scheduler.step(validation_loss)
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if validation_loss < best_loss - 1.0e-7:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:03d} train={train_loss:.6e} "
                f"validation={validation_loss:.6e}",
                flush=True,
            )
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("Neural ROM training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return TrainingResult(model, best_epoch, best_loss, rows)


def train_final(
    dataset: SequenceDataset,
    state_dim: int,
    hidden_dim: int,
    delta_limit: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> tuple[ParametricDelayROM, list[dict]]:
    seed_everything(seed)
    model = ParametricDelayROM(state_dim, hidden_dim, delta_limit).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for history, target, parameter in loader:
            history = history.to(device)
            target = target.to(device)
            parameter = parameter.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model, history, target, parameter)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(history)
            count += len(history)
        rows.append(
            {
                "epoch": epoch,
                "train_loss": total / max(count, 1),
            }
        )
    return model, rows


@torch.no_grad()
def rollout_numpy(
    model: ParametricDelayROM,
    history: np.ndarray,
    target_ez_kvm: float,
    steps: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    history_tensor = torch.as_tensor(
        history[None], dtype=torch.float32, device=device
    )
    parameter = torch.as_tensor(
        [(target_ez_kvm - PARAMETER_CENTER_KVM) / PARAMETER_SCALE_KVM],
        dtype=torch.float32,
        device=device,
    )
    prediction = model.rollout(history_tensor, parameter, steps)[0]
    return prediction.detach().cpu().numpy().astype(np.float64)


def evaluate_e25(
    cases: dict[int, baseline.RawCase],
    representation: baseline.SharedRepresentation,
    model: ParametricDelayROM,
    history_steps: int,
    device: torch.device,
) -> tuple[dict, dict[str, np.ndarray]]:
    case = cases[VALIDATION_FIELD]
    history_indices = np.flatnonzero(
        (case.time_us >= TRAIN_START_US - 1.0e-10)
        & (case.time_us < FORECAST_START_US - 1.0e-10)
    )
    forecast_mask = baseline.interval_mask(
        case.time_us,
        FORECAST_START_US,
        FORECAST_END_US,
        include_end=True,
    )
    history = representation.states[VALIDATION_FIELD][history_indices][
        -history_steps:
    ]
    truth = representation.states[VALIDATION_FIELD][forecast_mask]
    prediction = rollout_numpy(
        model, history, VALIDATION_FIELD, len(truth), device
    )
    persistence = np.repeat(history[-1:, :], len(truth), axis=0)
    row = baseline.evaluate_rollout(
        "neural_delay_rom",
        representation,
        VALIDATION_FIELD,
        forecast_mask,
        prediction,
        persistence,
    )
    return row, {
        "time_us": case.time_us[forecast_mask],
        "frame": case.frame[forecast_mask],
        "truth": truth,
        "prediction": prediction,
        "persistence": persistence,
    }


def plot_losses(path: Path, rows: list[dict]) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    epoch = [row["epoch"] for row in rows]
    axis.semilogy(epoch, [row["train_loss"] for row in rows], label="source train")
    axis.semilogy(
        epoch,
        [row["validation_loss"] for row in rows],
        label="source validation",
    )
    axis.set_xlabel("epoch")
    axis.set_ylabel("free-rollout Huber loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_e25(
    path: Path,
    representation: baseline.SharedRepresentation,
    arrays: dict[str, np.ndarray],
) -> None:
    decoded = {
        name: representation.scaler.inverse(arrays[name])
        for name in ("truth", "prediction", "persistence")
    }
    figure, axes = plt.subplots(3, 1, figsize=(10.5, 8.5), sharex=True)
    for band, label in enumerate(baseline.MODE_BAND_LABELS):
        axes[band].plot(
            arrays["time_us"],
            decoded["truth"]["transport"][:, band],
            color="#111111",
            label="PIC truth",
        )
        axes[band].plot(
            arrays["time_us"],
            decoded["prediction"]["transport"][:, band],
            color="#0072B2",
            label="neural parametric ROM",
        )
        axes[band].plot(
            arrays["time_us"],
            decoded["persistence"]["transport"][:, band],
            color="#999999",
            linestyle=":",
            label="persistence",
        )
        axes[band].set_ylabel(f"{label}\ntransport")
        axes[band].grid(alpha=0.25)
        axes[band].legend(fontsize=8)
    error = np.sqrt(
        np.mean((arrays["prediction"] - arrays["truth"]) ** 2, axis=1)
    )
    persistence_error = np.sqrt(
        np.mean((arrays["persistence"] - arrays["truth"]) ** 2, axis=1)
    )
    axes[2].plot(arrays["time_us"], error, label="neural ROM", color="#0072B2")
    axes[2].plot(
        arrays["time_us"],
        persistence_error,
        label="persistence",
        color="#999999",
    )
    axes[2].set_ylabel("state RMSE")
    axes[2].set_xlabel("time [us]")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)
    figure.suptitle("E20/E30-trained nonlinear pROM on held-out E25")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_rollout(
    path: Path,
    representation: baseline.SharedRepresentation,
    arrays: dict[str, np.ndarray],
) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["target_electric_field_kvm"] = VALIDATION_FIELD
        handle.attrs["target_transition_pairs_used"] = False
        for name in ("time_us", "frame"):
            handle.create_dataset(name, data=arrays[name])
        for name in ("truth", "prediction", "persistence"):
            group = handle.require_group(name)
            group.create_dataset("state", data=arrays[name], compression="gzip")
            for group_name, values in representation.scaler.inverse(
                arrays[name]
            ).items():
                group.create_dataset(group_name, data=values, compression="gzip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train nonlinear parameter-conditioned RadAz delay ROM."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--history-steps", type=int, default=40)
    parser.add_argument("--rollout-steps", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--delta-limit", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"device={device}", flush=True)

    print("[1/6] Loading stationary E20/E25/E30 data", flush=True)
    cases = {field: baseline.load_case(field) for field in FIELDS}
    development_representation = baseline.fit_shared_representation(
        cases,
        DEVELOPMENT_FIELDS,
        baseline.FIT_START_US,
        baseline.FINAL_FIT_END_US,
    )
    train_data = SequenceDataset(
        cases,
        development_representation.states,
        DEVELOPMENT_FIELDS,
        TRAIN_START_US,
        TRAIN_END_US,
        args.history_steps,
        args.rollout_steps,
        stride=2,
    )
    validation_data = SequenceDataset(
        cases,
        development_representation.states,
        DEVELOPMENT_FIELDS,
        VALIDATION_START_US,
        VALIDATION_END_US,
        args.history_steps,
        args.rollout_steps,
        stride=2,
    )
    print(
        f"[2/6] Training nonlinear pROM on {len(train_data)} source windows",
        flush=True,
    )
    result = train_development(
        train_data=train_data,
        validation_data=validation_data,
        state_dim=30,
        hidden_dim=args.hidden_dim,
        delta_limit=args.delta_limit,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
    )
    baseline.write_csv(output / "training_history.csv", result.history)
    plot_losses(output / "training_history.png", result.history)

    print("[3/6] Evaluating untouched E25 24--30 us", flush=True)
    e25_metrics, arrays = evaluate_e25(
        cases,
        development_representation,
        result.model,
        args.history_steps,
        device,
    )
    baseline.write_csv(output / "validation_e25_metrics.csv", [e25_metrics])
    save_rollout(
        output / "validation_e25_rollout.h5",
        development_representation,
        arrays,
    )
    plot_e25(
        output / "validation_e25_rollout.png",
        development_representation,
        arrays,
    )
    accepted = bool(
        e25_metrics["finite_fraction"] == 1.0
        and e25_metrics["state_skill_vs_persistence"] > 0.0
        and e25_metrics["transport_skill_vs_persistence"] > 0.0
    )
    status = "ACCEPTED_DEVELOPMENT" if accepted else "REJECTED_DEVELOPMENT"
    print(
        f"[4/6] E25 status={status}; state_skill={e25_metrics['state_skill_vs_persistence']:.3f}; "
        f"transport_skill={e25_metrics['transport_skill_vs_persistence']:.3f}",
        flush=True,
    )

    # Fit the final candidate for reproducibility even when rejected.  A
    # rejected checkpoint is retained as a negative baseline and must not be
    # promoted to the blind E22.5 comparison without an explicit protocol
    # revision recorded in the lock file.
    print("[5/6] Fitting reproducible E20/E25/E30 final candidate", flush=True)
    final_representation = baseline.fit_shared_representation(
        cases,
        FIELDS,
        baseline.FIT_START_US,
        baseline.FINAL_FIT_END_US,
    )
    final_data = SequenceDataset(
        cases,
        final_representation.states,
        FIELDS,
        TRAIN_START_US,
        FINAL_TRAIN_END_US,
        args.history_steps,
        args.rollout_steps,
        stride=2,
    )
    final_model, final_history = train_final(
        dataset=final_data,
        state_dim=30,
        hidden_dim=args.hidden_dim,
        delta_limit=args.delta_limit,
        epochs=result.best_epoch,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
    )
    baseline.write_csv(output / "final_training_history.csv", final_history)
    checkpoint_path = output / "neural_parametric_rom_data_only.pt"
    torch.save(
        {
            "state_dict": final_model.cpu().state_dict(),
            "config": {
                "state_dim": 30,
                "hidden_dim": args.hidden_dim,
                "delta_limit": args.delta_limit,
                "history_steps": args.history_steps,
                "rollout_steps": args.rollout_steps,
                "parameter_center_kvm": PARAMETER_CENTER_KVM,
                "parameter_scale_kvm": PARAMETER_SCALE_KVM,
            },
            "development_status": status,
            "development_e25_metrics": e25_metrics,
        },
        checkpoint_path,
    )
    script_path = Path(__file__).resolve()
    representation_path = (
        baseline.DEFAULT_OUTPUT.resolve() / "parametric_rom_data_only.h5"
    )
    lock = {
        "status": status,
        "accepted_for_blind_e22p5": accepted,
        "acceptance_rule": (
            "finite_fraction == 1 and state_skill_vs_persistence > 0 and "
            "transport_skill_vs_persistence > 0 on held-out E25"
        ),
        "development_training_fields_kvm": DEVELOPMENT_FIELDS,
        "development_validation_field_kvm": VALIDATION_FIELD,
        "development_e25_metrics": e25_metrics,
        "best_epoch": result.best_epoch,
        "best_source_validation_loss": result.best_validation_loss,
        "final_training_fields_kvm": FIELDS,
        "target_sweep": {
            "electric_field_kvm": 22.5,
            "directions": ["20_to_22.5", "25_to_22.5"],
            "data_loaded_for_fit": False,
            "data_loaded_for_selection": False,
        },
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "representation_bundle": str(representation_path),
        "representation_bundle_exists": representation_path.is_file(),
        "script": str(script_path),
        "script_sha256": sha256(script_path),
        "arguments": vars(args),
    }
    (output / "model_lock.json").write_text(
        json.dumps(json_safe(lock), indent=2), encoding="utf-8"
    )
    readme = [
        "# Nonlinear parameter-conditioned RadAz delay ROM",
        "",
        f"Development status: **{status}**",
        "",
        "The model was trained on stationary E20/E30 trajectories in the common L+R+T state. E25 transition pairs were completely excluded and E25 24--30 us was used as the declared interpolation validation.",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| state skill vs persistence | {float(e25_metrics['state_skill_vs_persistence']):.4f} |",
        f"| state correlation | {float(e25_metrics['state_correlation']):.4f} |",
        f"| radial anomaly correlation | {float(e25_metrics['radial_temporal_anomaly_correlation']):.4f} |",
        f"| transport anomaly correlation | {float(e25_metrics['transport_temporal_anomaly_correlation']):.4f} |",
        f"| transport skill vs persistence | {float(e25_metrics['transport_skill_vs_persistence']):.4f} |",
        "",
        "No E22.5 stationary or sweep data were loaded. A rejected model is retained only as a negative baseline; its lock file explicitly prevents promotion to the blind sweep comparison.",
    ]
    (output / "README.md").write_text(
        "\n".join(readme) + "\n", encoding="utf-8"
    )
    summary = {
        "status": status,
        "accepted_for_blind_e22p5": accepted,
        "development_e25_metrics": e25_metrics,
        "best_epoch": result.best_epoch,
        "checkpoint": str(checkpoint_path),
        "target_22p5_data_used": False,
    }
    (output / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2), encoding="utf-8"
    )
    print(f"[6/6] {status}: wrote {output}", flush=True)


if __name__ == "__main__":
    main()
