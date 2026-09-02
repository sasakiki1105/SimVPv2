"""Stage-3 field-gradient physics-loss ablation for the RadAz sweep ROM.

This script keeps the E25 -> E22.5 primary trajectory blind.  It starts each
candidate from the same Stage-2 recurrent checkpoint, attaches an identical
decoder for physical Fourier coefficients, and compares lambda_E values for

    E_y + d(phi)/dy = 0.

The residual uses the azimuthal spectral derivative and a truth-calibrated
floor/hinge, so discretization and PIC noise already present in the target are
not forced to zero.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

import analyze_radaz_augmented_physical_state_dynamics as augmented
import train_radaz_parametric_neural_delay_rom as neural
import train_radaz_regime_aware_transition_rom as stage2


ROOT = Path(__file__).resolve().parent
DEFAULT_STAGE2 = (
    ROOT
    / "workdirs"
    / "train_radaz_regime_aware_transition_rom_h160"
    / "regime_aware_transition_rom_data_only.pt"
)
DEFAULT_OUTPUT = ROOT / "workdirs" / "train_radaz_field_gradient_physics_rom"
LAMBDA_E_CANDIDATES = (0.0, 0.01, 0.1, 1.0)
FIELDS = ("phi", "efy")
MODES = np.arange(1, 22, dtype=np.int64)
RADIAL_BANDS = 8
AZIMUTHAL_LENGTH_M = 0.0128


@dataclass
class ObservationNormalizer:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.mean


@dataclass
class CandidateResult:
    lambda_e: float
    model: "PhysicsROM"
    best_epoch: int
    best_validation_loss: float
    history: list[dict]
    state_metrics: dict
    field_metrics: dict
    arrays: dict[str, np.ndarray]


class PhysicsWindowDataset(Dataset):
    def __init__(
        self,
        specifications: list[
            tuple[stage2.Trajectory, np.ndarray, np.ndarray, float, float, float, int]
        ],
        history_steps: int,
        rollout_steps: int,
    ) -> None:
        histories = []
        state_targets = []
        observation_targets = []
        parameters = []
        for trajectory, state, observation, start_us, end_us, parameter_kvm, stride in specifications:
            possible = np.flatnonzero(
                (trajectory.time_us >= start_us - 1.0e-10)
                & (trajectory.time_us < end_us - 1.0e-10)
            )
            for target_start in possible[::stride]:
                target_stop = int(target_start) + rollout_steps
                history_start = int(target_start) - history_steps
                if history_start < 0 or target_stop > len(state):
                    continue
                if trajectory.time_us[target_stop - 1] >= end_us - 1.0e-10:
                    continue
                histories.append(state[history_start:target_start])
                state_targets.append(state[target_start:target_stop])
                observation_targets.append(observation[target_start:target_stop])
                parameters.append(
                    (parameter_kvm - stage2.PARAMETER_CENTER_KVM)
                    / stage2.PARAMETER_SCALE_KVM
                )
        if not histories:
            raise ValueError("No physics-training windows were constructed")
        self.histories = torch.from_numpy(np.asarray(histories, dtype=np.float32))
        self.state_targets = torch.from_numpy(
            np.asarray(state_targets, dtype=np.float32)
        )
        self.observation_targets = torch.from_numpy(
            np.asarray(observation_targets, dtype=np.float32)
        )
        self.parameters = torch.as_tensor(parameters, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.histories)

    def __getitem__(self, index: int):
        return (
            self.histories[index],
            self.state_targets[index],
            self.observation_targets[index],
            self.parameters[index],
        )


class PhysicsROM(nn.Module):
    def __init__(
        self,
        dynamics: neural.ParametricDelayROM,
        observation_dimension: int,
        decoder_hidden: int,
        observation_mean: np.ndarray,
        observation_scale: np.ndarray,
        wave_numbers: np.ndarray,
        residual_scale: np.ndarray,
        truth_floor_power: np.ndarray,
    ) -> None:
        super().__init__()
        self.dynamics = dynamics
        self.decoder = nn.Sequential(
            nn.Linear(dynamics.state_dim, decoder_hidden),
            nn.SiLU(),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.SiLU(),
            nn.Linear(decoder_hidden, observation_dimension),
        )
        self.register_buffer(
            "observation_mean",
            torch.as_tensor(observation_mean, dtype=torch.float32),
        )
        self.register_buffer(
            "observation_scale",
            torch.as_tensor(observation_scale, dtype=torch.float32),
        )
        self.register_buffer(
            "wave_numbers",
            torch.as_tensor(wave_numbers, dtype=torch.float32),
        )
        self.register_buffer(
            "residual_scale",
            torch.as_tensor(residual_scale, dtype=torch.float32),
        )
        self.register_buffer(
            "truth_floor_power",
            torch.as_tensor(truth_floor_power, dtype=torch.float32),
        )

    def decode(self, states: torch.Tensor) -> torch.Tensor:
        return self.decoder(states)

    def rollout(
        self, history: torch.Tensor, parameter: torch.Tensor, steps: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        states = self.dynamics.rollout(history, parameter, steps)
        observations = self.decode(states)
        return states, observations

    def physical_observations(self, normalized: torch.Tensor) -> torch.Tensor:
        raw = normalized * self.observation_scale + self.observation_mean
        return raw.reshape(
            *raw.shape[:-1], len(FIELDS), RADIAL_BANDS, len(MODES), 2
        )

    def field_gradient_power(self, normalized: torch.Tensor) -> torch.Tensor:
        physical = self.physical_observations(normalized)
        phi = physical[..., 0, :, :, :]
        ey = physical[..., 1, :, :, :]
        k = self.wave_numbers
        residual_real = ey[..., 0] - k * phi[..., 1]
        residual_imag = ey[..., 1] + k * phi[..., 0]
        return (
            residual_real.square() + residual_imag.square()
        ) / self.residual_scale.square()

    def physics_loss(self, normalized: torch.Tensor) -> torch.Tensor:
        excess = self.field_gradient_power(normalized) - self.truth_floor_power
        return torch.relu(excess).mean()


def json_safe(value):
    return stage2.json_safe(value)


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
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_observations(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        coefficients = np.asarray(handle["coefficients"], dtype=np.complex128)
        raw_fields = np.asarray(handle["fields"])
    names = [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in raw_fields
    ]
    selected = np.stack(
        [coefficients[:, names.index(name), :, 1:22] for name in FIELDS],
        axis=1,
    )
    packed = np.stack([selected.real, selected.imag], axis=-1)
    return packed.reshape(len(packed), -1)


def fit_observation_statistics(
    observations: dict[str, np.ndarray],
    fit_masks: dict[str, np.ndarray],
) -> tuple[ObservationNormalizer, np.ndarray, np.ndarray, dict]:
    fit = np.concatenate(
        [observations[name][fit_masks[name]] for name in observations], axis=0
    )
    mean = np.mean(fit, axis=0)
    scale = np.std(fit, axis=0, ddof=1)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    normalizer = ObservationNormalizer(mean, scale)
    shaped = fit.reshape(len(fit), len(FIELDS), RADIAL_BANDS, len(MODES), 2)
    phi = shaped[:, 0]
    ey = shaped[:, 1]
    k = 2.0 * np.pi * MODES / AZIMUTHAL_LENGTH_M
    residual_real = ey[..., 0] - k[None, None, :] * phi[..., 1]
    residual_imag = ey[..., 1] + k[None, None, :] * phi[..., 0]
    residual_scale = np.sqrt(np.mean(ey[..., 0] ** 2 + ey[..., 1] ** 2, axis=0))
    residual_scale = np.maximum(residual_scale, np.max(residual_scale) * 1.0e-6)
    truth_floor_power = np.mean(
        (residual_real ** 2 + residual_imag ** 2)
        / residual_scale[None, :, :] ** 2,
        axis=0,
    )
    audit = {
        "azimuthal_length_m": AZIMUTHAL_LENGTH_M,
        "modes": MODES,
        "global_truth_residual_over_ey_rms": float(np.sqrt(
            np.mean(residual_real ** 2 + residual_imag ** 2)
            / np.mean(ey[..., 0] ** 2 + ey[..., 1] ** 2)
        )),
        "MTSI_truth_residual_over_ey_rms": float(np.sqrt(
            np.mean(residual_real[..., :6] ** 2 + residual_imag[..., :6] ** 2)
            / np.mean(ey[..., :6, 0] ** 2 + ey[..., :6, 1] ** 2)
        )),
        "ECDI_truth_residual_over_ey_rms": float(np.sqrt(
            np.mean(residual_real[..., 8:] ** 2 + residual_imag[..., 8:] ** 2)
            / np.mean(ey[..., 8:, 0] ** 2 + ey[..., 8:, 1] ** 2)
        )),
    }
    return normalizer, residual_scale, truth_floor_power, audit


def state_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scaler: augmented.GroupScaler,
) -> torch.Tensor:
    losses = []
    for name in stage2.STATE_GROUPS:
        selected = scaler.slices[name]
        weight = scaler.weights[name]
        losses.append(nn.functional.smooth_l1_loss(
            prediction[..., selected] / weight,
            target[..., selected] / weight,
            beta=0.10,
            reduction="mean",
        ))
    return torch.stack(losses).mean()


def combined_loss(
    model: PhysicsROM,
    history: torch.Tensor,
    state_target: torch.Tensor,
    observation_target: torch.Tensor,
    parameter: torch.Tensor,
    scaler: augmented.GroupScaler,
    lambda_observation: float,
    lambda_e: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    state_prediction, observation_prediction = model.rollout(
        history, parameter, state_target.shape[1]
    )
    state_term = state_loss(state_prediction, state_target, scaler)
    observation_term = nn.functional.smooth_l1_loss(
        observation_prediction,
        observation_target,
        beta=0.10,
        reduction="mean",
    )
    physics_term = model.physics_loss(observation_prediction)
    total = state_term + lambda_observation * observation_term + lambda_e * physics_term
    return total, {
        "state": state_term,
        "observation": observation_term,
        "physics": physics_term,
    }


def warm_decoder(
    model: PhysicsROM,
    state: np.ndarray,
    observation: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> list[dict]:
    for parameter in model.dynamics.parameters():
        parameter.requires_grad_(False)
    dataset = TensorDataset(
        torch.as_tensor(state, dtype=torch.float32),
        torch.as_tensor(observation, dtype=torch.float32),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=learning_rate)
    rows = []
    for epoch in range(1, epochs + 1):
        total = 0.0
        count = 0
        for states, targets in loader:
            states = states.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model.decode(states)
            loss = nn.functional.smooth_l1_loss(
                prediction, targets, beta=0.10
            )
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(states)
            count += len(states)
        rows.append({"stage": "decoder_warmup", "epoch": epoch, "loss": total / count})
        if epoch == 1 or epoch % 20 == 0:
            print(f"decoder epoch={epoch:03d} loss={total / count:.6e}", flush=True)
    for parameter in model.dynamics.parameters():
        parameter.requires_grad_(True)
    return rows


@torch.no_grad()
def validation_loss(
    model: PhysicsROM,
    loader: DataLoader,
    scaler: augmented.GroupScaler,
    lambda_observation: float,
    lambda_e: float,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for history, state_target, observation_target, parameter in loader:
        history = history.to(device)
        state_target = state_target.to(device)
        observation_target = observation_target.to(device)
        parameter = parameter.to(device)
        loss, _ = combined_loss(
            model,
            history,
            state_target,
            observation_target,
            parameter,
            scaler,
            lambda_observation,
            lambda_e,
        )
        total += float(loss) * len(history)
        count += len(history)
    return total / max(count, 1)


def train_candidate(
    initial_state: dict,
    make_model,
    train_data: PhysicsWindowDataset,
    validation_data: PhysicsWindowDataset,
    scaler: augmented.GroupScaler,
    lambda_observation: float,
    lambda_e: float,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> tuple[PhysicsROM, int, float, list[dict]]:
    seed_everything(seed)
    model = make_model().to(device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1.0e-5
    )
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        validation_data, batch_size=batch_size, shuffle=False
    )
    best_loss = validation_loss(
        model,
        validation_loader,
        scaler,
        lambda_observation,
        lambda_e,
        device,
    )
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    rows = [{
        "lambda_E": lambda_e,
        "epoch": 0,
        "train_loss": float("nan"),
        "train_state_loss": float("nan"),
        "train_observation_loss": float("nan"),
        "train_physics_loss": float("nan"),
        "validation_loss": best_loss,
        "learning_rate": learning_rate,
    }]
    for epoch in range(1, epochs + 1):
        model.train()
        sums = {"total": 0.0, "state": 0.0, "observation": 0.0, "physics": 0.0}
        count = 0
        for history, state_target, observation_target, parameter in train_loader:
            history = history.to(device)
            state_target = state_target.to(device)
            observation_target = observation_target.to(device)
            parameter = parameter.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, terms = combined_loss(
                model,
                history,
                state_target,
                observation_target,
                parameter,
                scaler,
                lambda_observation,
                lambda_e,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sums["total"] += float(loss.detach()) * len(history)
            for name, value in terms.items():
                sums[name] += float(value.detach()) * len(history)
            count += len(history)
        valid = validation_loss(
            model,
            validation_loader,
            scaler,
            lambda_observation,
            lambda_e,
            device,
        )
        scheduler.step(valid)
        row = {
            "lambda_E": lambda_e,
            "epoch": epoch,
            "train_loss": sums["total"] / count,
            "train_state_loss": sums["state"] / count,
            "train_observation_loss": sums["observation"] / count,
            "train_physics_loss": sums["physics"] / count,
            "validation_loss": valid,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        rows.append(row)
        if valid < best_loss - 1.0e-7:
            best_loss = valid
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"lambda_E={lambda_e:g} epoch={epoch:03d} "
                f"train={row['train_loss']:.6e} validation={valid:.6e}",
                flush=True,
            )
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("No candidate checkpoint was produced")
    model.load_state_dict(best_state)
    return model, best_epoch, best_loss, rows


@torch.no_grad()
def free_rollout(
    model: PhysicsROM,
    history: np.ndarray,
    steps: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    history_tensor = torch.as_tensor(
        history[None], dtype=torch.float32, device=device
    )
    parameter = torch.as_tensor([-0.5], dtype=torch.float32, device=device)
    state, observation = model.rollout(history_tensor, parameter, steps)
    return (
        state[0].detach().cpu().numpy().astype(np.float64),
        observation[0].detach().cpu().numpy().astype(np.float64),
    )


def field_metrics(
    observation_truth_raw: np.ndarray,
    observation_prediction_norm: np.ndarray,
    normalizer: ObservationNormalizer,
    residual_scale: np.ndarray,
    truth_floor_power: np.ndarray,
) -> dict:
    prediction = normalizer.inverse(observation_prediction_norm)
    truth = observation_truth_raw
    shaped_truth = truth.reshape(len(truth), len(FIELDS), RADIAL_BANDS, len(MODES), 2)
    shaped_prediction = prediction.reshape(
        len(prediction), len(FIELDS), RADIAL_BANDS, len(MODES), 2
    )
    metrics = {}
    for index, name in enumerate(FIELDS):
        error = shaped_prediction[:, index] - shaped_truth[:, index]
        centered = shaped_truth[:, index] - np.mean(shaped_truth[:, index], axis=0)
        metrics[f"{name}_nrmse"] = float(
            np.sqrt(np.mean(error ** 2))
            / max(np.sqrt(np.mean(centered ** 2)), np.finfo(float).tiny)
        )
    phi = shaped_prediction[:, 0]
    ey = shaped_prediction[:, 1]
    k = 2.0 * np.pi * MODES / AZIMUTHAL_LENGTH_M
    residual_real = ey[..., 0] - k[None, None, :] * phi[..., 1]
    residual_imag = ey[..., 1] + k[None, None, :] * phi[..., 0]
    power = (residual_real ** 2 + residual_imag ** 2) / residual_scale[None] ** 2
    metrics["field_gradient_normalized_rms"] = float(np.sqrt(np.mean(power)))
    metrics["field_gradient_excess_hinge"] = float(
        np.mean(np.maximum(power - truth_floor_power[None], 0.0))
    )
    metrics["field_gradient_truth_floor_rms"] = float(
        np.sqrt(np.mean(truth_floor_power))
    )
    metrics["field_gradient_MTSI_normalized_rms"] = float(
        np.sqrt(np.mean(power[..., :6]))
    )
    metrics["field_gradient_ECDI_normalized_rms"] = float(
        np.sqrt(np.mean(power[..., 8:]))
    )
    return metrics


def plot_comparison(path: Path, rows: list[dict]) -> None:
    lambdas = [row["lambda_E"] for row in rows]
    labels = [f"{value:g}" for value in lambdas]
    x = np.arange(len(rows))
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    axes[0].bar(x, [row["state_skill_vs_persistence"] for row in rows])
    axes[0].axhline(0.0, color="#555555", linewidth=0.8)
    axes[0].set_title("state skill")
    axes[1].bar(x, [row["transport_skill_vs_persistence"] for row in rows])
    axes[1].axhline(0.0, color="#555555", linewidth=0.8)
    axes[1].set_title("transport skill")
    axes[2].bar(x, [row["field_gradient_excess_hinge"] for row in rows])
    axes[2].set_yscale("log")
    axes[2].set_title("E + grad(phi) excess")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.set_xlabel("lambda_E")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-checkpoint", type=Path, default=DEFAULT_STAGE2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--history-steps", type=int, default=40)
    parser.add_argument("--rollout-steps", type=int, default=80)
    parser.add_argument("--decoder-hidden", type=int, default=192)
    parser.add_argument("--decoder-warmup-epochs", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--decoder-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--lambda-observation", type=float, default=0.2)
    parser.add_argument("--lambda-e", type=float, nargs="+", default=LAMBDA_E_CANDIDATES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    seed_everything(args.seed)
    print(f"device={device}", flush=True)

    checkpoint = torch.load(args.stage2_checkpoint, map_location="cpu")
    if checkpoint["state_dimension"] != 30:
        raise ValueError("Stage-2 checkpoint is not the declared 30D state")
    print("[1/6] Rebuilding the Stage-2 representation without primary data", flush=True)
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
        "e25_stationary": stage2.interval_mask(
            e25.time_us, stage2.E25_FIT_START_US, stage2.E25_FIT_END_US
        ),
        "e20_to_e22p5": up.time_us < stage2.UP_TRAIN_END_US - 1.0e-10,
    }
    representation = stage2.fit_representation(trajectories, fit_masks)
    observations_raw = {
        name: load_observations(trajectory.physical_path)
        for name, trajectory in trajectories.items()
    }
    normalizer, residual_scale, truth_floor_power, audit = fit_observation_statistics(
        observations_raw, fit_masks
    )
    observations = {
        name: normalizer.transform(values) for name, values in observations_raw.items()
    }
    (output / "truth_field_gradient_audit.json").write_text(
        json.dumps(json_safe(audit), indent=2), encoding="utf-8"
    )
    print(json.dumps(json_safe(audit), indent=2), flush=True)

    train_data = PhysicsWindowDataset(
        [
            (
                e25,
                representation.states["e25_stationary"],
                observations["e25_stationary"],
                stage2.E25_FIT_START_US,
                stage2.E25_FIT_END_US,
                25.0,
                2,
            ),
            (
                up,
                representation.states["e20_to_e22p5"],
                observations["e20_to_e22p5"],
                float(up.time_us[0]),
                stage2.UP_TRAIN_END_US,
                22.5,
                1,
            ),
        ],
        args.history_steps,
        args.rollout_steps,
    )
    validation_data = PhysicsWindowDataset(
        [(
            up,
            representation.states["e20_to_e22p5"],
            observations["e20_to_e22p5"],
            stage2.UP_VALIDATION_START_US,
            float(up.time_us[-1] + 0.015),
            22.5,
            1,
        )],
        args.history_steps,
        args.rollout_steps,
    )
    print(
        f"windows: train={len(train_data)} validation={len(validation_data)}",
        flush=True,
    )

    def make_model() -> PhysicsROM:
        dynamics = neural.ParametricDelayROM(
            state_dim=30,
            hidden_dim=int(checkpoint["hidden_dimension"]),
            delta_limit=float(checkpoint["delta_limit"]),
        )
        dynamics.load_state_dict(checkpoint["model_state_dict"])
        return PhysicsROM(
            dynamics,
            observation_dimension=len(normalizer.mean),
            decoder_hidden=args.decoder_hidden,
            observation_mean=normalizer.mean,
            observation_scale=normalizer.scale,
            wave_numbers=2.0 * np.pi * MODES / AZIMUTHAL_LENGTH_M,
            residual_scale=residual_scale,
            truth_floor_power=truth_floor_power,
        )

    print("[2/6] Warming a shared physical observation decoder", flush=True)
    warm_model = make_model().to(device)
    fit_state = np.concatenate([
        representation.states[name][fit_masks[name]] for name in trajectories
    ], axis=0)
    fit_observation = np.concatenate([
        observations[name][fit_masks[name]] for name in trajectories
    ], axis=0)
    warm_rows = warm_decoder(
        warm_model,
        fit_state,
        fit_observation,
        args.decoder_warmup_epochs,
        args.batch_size,
        args.decoder_learning_rate,
        args.seed,
        device,
    )
    initial_state = copy.deepcopy(warm_model.state_dict())

    validation_start = int(np.flatnonzero(
        up.time_us >= stage2.UP_VALIDATION_START_US - 1.0e-10
    )[0])
    history = representation.states["e20_to_e22p5"][
        validation_start - args.history_steps:validation_start
    ]
    truth_state = representation.states["e20_to_e22p5"][validation_start:]
    persistence = np.repeat(history[-1:, :], len(truth_state), axis=0)
    truth_observation = observations_raw["e20_to_e22p5"][validation_start:]

    print("[3/6] Training paired data/physics candidates", flush=True)
    candidate_results: list[CandidateResult] = []
    all_training_rows = []
    for candidate_index, lambda_e in enumerate(args.lambda_e):
        model, best_epoch, best_loss, rows = train_candidate(
            initial_state,
            make_model,
            train_data,
            validation_data,
            representation.scaler,
            args.lambda_observation,
            float(lambda_e),
            args.epochs,
            args.patience,
            args.batch_size,
            args.learning_rate,
            args.weight_decay,
            # Identical minibatch order and stochastic state for every
            # lambda_E candidate; only the physics coefficient may differ.
            args.seed + 10,
            device,
        )
        prediction_state, prediction_observation = free_rollout(
            model, history, len(truth_state), device
        )
        state_metrics = stage2.metric_row(
            f"lambda_E_{lambda_e:g}",
            truth_state,
            prediction_state,
            persistence,
            representation.scaler,
        )
        physics_metrics = field_metrics(
            truth_observation,
            prediction_observation,
            normalizer,
            residual_scale,
            truth_floor_power,
        )
        candidate_results.append(CandidateResult(
            float(lambda_e),
            model,
            best_epoch,
            best_loss,
            rows,
            state_metrics,
            physics_metrics,
            {
                "state": prediction_state,
                "observation_normalized": prediction_observation,
            },
        ))
        all_training_rows.extend(rows)
        print(json.dumps(json_safe({**state_metrics, **physics_metrics}), indent=2), flush=True)

    print("[4/6] Selecting lambda_E on up-sweep development validation", flush=True)
    data_only = next(result for result in candidate_results if result.lambda_e == 0.0)
    eligible = []
    for result in candidate_results:
        if result.lambda_e <= 0.0:
            continue
        state = result.state_metrics
        residual_reduction = 1.0 - (
            result.field_metrics["field_gradient_excess_hinge"]
            / max(data_only.field_metrics["field_gradient_excess_hinge"], 1.0e-30)
        )
        if (
            state["finite_fraction"] == 1.0
            and state["state_skill_vs_persistence"] > 0.0
            and state["transport_skill_vs_persistence"] > 0.0
            and state["state_skill_vs_persistence"]
            >= data_only.state_metrics["state_skill_vs_persistence"] - 0.05
            and state["transport_skill_vs_persistence"]
            >= data_only.state_metrics["transport_skill_vs_persistence"] - 0.05
            and state["MTSI_n1_6_transport_skill"]
            >= data_only.state_metrics["MTSI_n1_6_transport_skill"] - 0.05
            and state["ECDI_n9_21_transport_skill"]
            >= data_only.state_metrics["ECDI_n9_21_transport_skill"] - 0.05
            and result.field_metrics["phi_nrmse"]
            <= data_only.field_metrics["phi_nrmse"] + 0.02
            and result.field_metrics["efy_nrmse"]
            <= data_only.field_metrics["efy_nrmse"] + 0.02
            and result.field_metrics["phi_nrmse"] < 1.0
            and result.field_metrics["efy_nrmse"] < 1.0
            and residual_reduction > 0.0
        ):
            eligible.append((residual_reduction, result))
    selected = max(eligible, key=lambda item: item[0])[1] if eligible else None
    status = "READY_FOR_BLIND_PRIMARY_TEST" if selected is not None else "REJECTED_DEVELOPMENT"

    print("[5/6] Saving the locked ablation", flush=True)
    metric_rows = []
    for result in candidate_results:
        row = {
            "lambda_E": result.lambda_e,
            "selected": bool(selected is result),
            "best_epoch": result.best_epoch,
            "best_window_validation_loss": result.best_validation_loss,
            **result.state_metrics,
            **result.field_metrics,
        }
        metric_rows.append(row)
        candidate_path = output / f"physics_rom_lambdaE_{result.lambda_e:g}.pt"
        torch.save({
            "model_state_dict": result.model.state_dict(),
            "lambda_E": result.lambda_e,
            "lambda_observation": args.lambda_observation,
            "history_steps": args.history_steps,
            "rollout_steps": args.rollout_steps,
            "stage2_checkpoint": str(args.stage2_checkpoint.resolve()),
        }, candidate_path)
    write_csv(output / "candidate_metrics.csv", metric_rows)
    write_csv(output / "candidate_training_history.csv", all_training_rows)
    (output / "candidate_metrics.json").write_text(
        json.dumps(json_safe(metric_rows), indent=2), encoding="utf-8"
    )
    plot_comparison(output / "physics_ablation.png", metric_rows)
    with h5py.File(output / "development_rollouts.h5", "w") as handle:
        handle.attrs["primary_E25_to_E22p5_loaded"] = False
        handle.create_dataset("time_us", data=up.time_us[validation_start:])
        handle.create_dataset("frame", data=up.frame[validation_start:])
        handle.create_dataset("truth/state", data=truth_state, compression="gzip")
        handle.create_dataset("truth/observation_raw", data=truth_observation, compression="gzip")
        handle.create_dataset("persistence/state", data=persistence, compression="gzip")
        for result in candidate_results:
            group = handle.require_group(f"lambda_E_{result.lambda_e:g}")
            group.create_dataset("state", data=result.arrays["state"], compression="gzip")
            group.create_dataset(
                "observation_normalized",
                data=result.arrays["observation_normalized"],
                compression="gzip",
            )
    with h5py.File(output / "observation_normalization.h5", "w") as handle:
        handle.create_dataset("mean", data=normalizer.mean)
        handle.create_dataset("scale", data=normalizer.scale)
        handle.create_dataset("residual_scale", data=residual_scale)
        handle.create_dataset("truth_floor_power", data=truth_floor_power)
        handle.create_dataset("modes", data=MODES)
        handle.attrs["fields"] = ",".join(FIELDS)
        handle.attrs["azimuthal_length_m"] = AZIMUTHAL_LENGTH_M

    selected_row = (
        next(row for row in metric_rows if row["selected"])
        if selected is not None
        else None
    )
    lock = {
        "status": status,
        "selected_lambda_E": selected.lambda_e if selected is not None else None,
        "selection_rule": (
            "positive state and transport skill; no more than 0.05 skill "
            "and bandwise transport degradation; no more than 0.02 phi/Ey "
            "NRMSE degradation from paired lambda_E=0; phi/Ey NRMSE < 1; "
            "positive field-gradient hinge reduction"
        ),
        "selected_metrics": selected_row,
        "truth_residual_audit": audit,
        "development_direction": "E20_to_E22.5",
        "development_training_end_us": stage2.UP_TRAIN_END_US,
        "development_validation_interval_us": [
            float(up.time_us[validation_start]), float(up.time_us[-1])
        ],
        "primary_test": {
            "direction": "E25_to_E22.5",
            "data_loaded": False,
            "used_for_normalization": False,
            "used_for_training": False,
            "used_for_selection": False,
        },
        "stage2_checkpoint": str(args.stage2_checkpoint.resolve()),
        "stage2_checkpoint_sha256": sha256(args.stage2_checkpoint.resolve()),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "arguments": vars(args),
    }
    (output / "model_lock.json").write_text(
        json.dumps(json_safe(lock), indent=2), encoding="utf-8"
    )
    readme = f"""# RadAz Stage-3 field-gradient physics ROM

- Status: `{status}`
- Selected lambda_E: `{lock['selected_lambda_E']}`
- Truth E_y + d(phi)/dy residual / E_y RMS: {audit['global_truth_residual_over_ey_rms']:.6g}
- Primary E25 -> E22.5 data loaded: **no**

All candidates use the same Stage-2 checkpoint, decoder initialization,
training windows, and validation windows.  `lambda_E=0` is the paired
data-only auxiliary-decoder control.  The physical penalty is a spectral
field-gradient residual with a PIC-truth-calibrated floor/hinge.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print("[6/6] Complete", flush=True)
    print(f"status={status}", flush=True)
    print(f"selected_lambda_E={lock['selected_lambda_E']}", flush=True)
    print(f"output={output}", flush=True)


if __name__ == "__main__":
    main()
