"""Train a regime-aware RadAz ROM with physical Fourier modes in the state.

The recurrent state is

* 20 frozen-SimVP block-PCA coordinates,
* 168 global physical Fourier coefficients: phi, ne, ni, Ey; n=1--21;
  real and imaginary parts,
* 8 radial phi band envelopes, and
* 16 radial density--Ey cross-spectrum components.

Modal transport is derived from the predicted cross spectrum and is not an
independent state variable.  Stationary E25 is used for local pretraining,
E20 -> E22.5 over 30--35 us is used for transition training, and 35--40 us is
development validation.  The E25 -> E22.5 primary trajectory is never read.
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
from torch.utils.data import DataLoader, Dataset

import analyze_radaz_augmented_physical_state_dynamics as augmented
import analyze_radaz_blockwise_fourier_latent_dynamics as block
import train_radaz_parametric_neural_delay_rom as neural
import train_radaz_regime_aware_transition_rom as stage2


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "workdirs" / "train_radaz_direct_physical_state_rom"
GROUPS = ("latent", "physical_fourier", "radial", "cross")
FIELD_NAMES = ("phi", "electron_den", "ion_den", "efy")
MODE_NUMBERS = np.arange(1, 22, dtype=np.int64)
MODE_BANDS = {
    "MTSI_n1_6": np.arange(1, 7, dtype=np.int64),
    "ECDI_n9_21": np.arange(9, 22, dtype=np.int64),
}
STATE_DIMENSION = 20 + 4 * 21 * 2 + 8 + 4 * 2 * 2
AZIMUTHAL_LENGTH_M = 0.0128


@dataclass
class DirectScaler:
    names: tuple[str, ...]
    slices: dict[str, slice]
    means: dict[str, np.ndarray]
    scales: dict[str, np.ndarray]

    @classmethod
    def fit(cls, groups: dict[str, np.ndarray]) -> "DirectScaler":
        slices = {}
        means = {}
        scales = {}
        offset = 0
        for name in GROUPS:
            values = groups[name]
            width = values.shape[1]
            slices[name] = slice(offset, offset + width)
            offset += width
            means[name] = np.mean(values, axis=0)
            standard = np.std(values, axis=0, ddof=1)
            # The physical-Fourier group mixes density, potential, and field
            # units.  A group-relative floor would incorrectly classify the
            # smaller-unit phi/Ey coefficients as constants, so every
            # coordinate must be standardized against its own variance.
            scales[name] = np.where(standard > 1.0e-20, standard, 1.0)
        return cls(GROUPS, slices, means, scales)

    def transform(self, groups: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([
            (groups[name] - self.means[name]) / self.scales[name]
            for name in self.names
        ], axis=1)

    def inverse(self, state: np.ndarray) -> dict[str, np.ndarray]:
        return {
            name: state[:, self.slices[name]] * self.scales[name] + self.means[name]
            for name in self.names
        }


@dataclass
class DirectRepresentation:
    scaler: DirectScaler
    groups: dict[str, dict[str, np.ndarray]]
    states: dict[str, np.ndarray]
    macro_weights: np.ndarray
    block_models: dict[str, block.BlockPCA]
    pca_rows: list[dict]


class DirectWindowDataset(Dataset):
    def __init__(
        self,
        specifications: list[
            tuple[stage2.Trajectory, np.ndarray, float, float, float, int]
        ],
        history_steps: int,
        rollout_steps: int,
    ) -> None:
        histories = []
        targets = []
        parameters = []
        for trajectory, state, start_us, end_us, electric_field_kvm, stride in specifications:
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
                targets.append(state[target_start:target_stop])
                parameters.append(
                    (electric_field_kvm - stage2.PARAMETER_CENTER_KVM)
                    / stage2.PARAMETER_SCALE_KVM
                )
        if not histories:
            raise ValueError("No direct-state rollout windows were constructed")
        self.histories = torch.from_numpy(np.asarray(histories, dtype=np.float32))
        self.targets = torch.from_numpy(np.asarray(targets, dtype=np.float32))
        self.parameters = torch.as_tensor(parameters, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.histories)

    def __getitem__(self, index: int):
        return self.histories[index], self.targets[index], self.parameters[index]


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
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_coefficients(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        coefficients = np.asarray(handle["coefficients"], dtype=np.complex128)
        radial_weights = np.asarray(handle["radial_weights"], dtype=np.float64)
        raw_fields = np.asarray(handle["fields"])
    fields = [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in raw_fields
    ]
    order = [fields.index(name) for name in FIELD_NAMES]
    selected = coefficients[:, order, :, 1:22]
    global_coefficients = np.einsum(
        "r,tfrn->tfn", radial_weights, selected
    )
    packed = np.stack(
        [global_coefficients.real, global_coefficients.imag], axis=-1
    ).reshape(len(coefficients), -1)
    return packed, radial_weights


def fit_direct_representation(
    trajectories: dict[str, stage2.Trajectory],
    fit_masks: dict[str, np.ndarray],
) -> DirectRepresentation:
    base = stage2.fit_representation(trajectories, fit_masks)
    groups = {}
    radial_weights = None
    for name, trajectory in trajectories.items():
        physical_fourier, current_weights = load_coefficients(
            trajectory.physical_path
        )
        if radial_weights is None:
            radial_weights = current_weights
        elif not np.allclose(radial_weights, current_weights):
            raise ValueError("Radial-weight mismatch between trajectories")
        physical = augmented.flatten_physical(trajectory.physical)
        groups[name] = {
            "latent": base.groups[name]["latent"],
            "physical_fourier": physical_fourier,
            "radial": physical["radial"],
            "cross": physical["cross"],
        }
    stacked = {
        group: np.concatenate([
            groups[name][group][fit_masks[name]] for name in trajectories
        ], axis=0)
        for group in GROUPS
    }
    scaler = DirectScaler.fit(stacked)
    states = {name: scaler.transform(values) for name, values in groups.items()}
    dimensions = {name: groups["e25_stationary"][name].shape[1] for name in GROUPS}
    expected = {"latent": 20, "physical_fourier": 168, "radial": 8, "cross": 16}
    if dimensions != expected or sum(dimensions.values()) != STATE_DIMENSION:
        raise ValueError(f"Unexpected direct-state dimensions: {dimensions}")
    return DirectRepresentation(
        scaler=scaler,
        groups=groups,
        states=states,
        macro_weights=trajectories["e25_stationary"].physical.macro_weights,
        block_models=base.block_models,
        pca_rows=base.pca_rows,
    )


def group_balanced_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scaler: DirectScaler,
) -> torch.Tensor:
    return torch.stack([
        nn.functional.smooth_l1_loss(
            prediction[..., scaler.slices[name]],
            target[..., scaler.slices[name]],
            beta=0.10,
            reduction="mean",
        )
        for name in GROUPS
    ]).mean()


def batch_loss(
    model: neural.ParametricDelayROM,
    history: torch.Tensor,
    target: torch.Tensor,
    parameter: torch.Tensor,
    scaler: DirectScaler,
) -> torch.Tensor:
    prediction = model.rollout(history, parameter, target.shape[1])
    return group_balanced_loss(prediction, target, scaler)


@torch.no_grad()
def loader_loss(
    model: neural.ParametricDelayROM,
    loader: DataLoader,
    scaler: DirectScaler,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for history, target, parameter in loader:
        history = history.to(device)
        target = target.to(device)
        parameter = parameter.to(device)
        loss = batch_loss(model, history, target, parameter, scaler)
        total += float(loss) * len(history)
        count += len(history)
    return total / max(count, 1)


@torch.no_grad()
def rollout(
    model: neural.ParametricDelayROM,
    history: np.ndarray,
    steps: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    history_tensor = torch.as_tensor(
        history[None], dtype=torch.float32, device=device
    )
    parameter = torch.as_tensor([-0.5], dtype=torch.float32, device=device)
    return model.rollout(history_tensor, parameter, steps)[0].cpu().numpy().astype(np.float64)


def unpack_physical_fourier(values: np.ndarray) -> np.ndarray:
    shaped = values.reshape(len(values), len(FIELD_NAMES), len(MODE_NUMBERS), 2)
    return shaped[..., 0] + 1j * shaped[..., 1]


def unpack_cross(values: np.ndarray) -> np.ndarray:
    shaped = values.reshape(len(values), 4, 2, 2)
    return shaped[..., 0] + 1j * shaped[..., 1]


def field_gradient_metrics(coefficients: np.ndarray) -> dict[str, float]:
    phi = coefficients[:, FIELD_NAMES.index("phi")]
    ey = coefficients[:, FIELD_NAMES.index("efy")]
    k = 2.0 * np.pi * MODE_NUMBERS / AZIMUTHAL_LENGTH_M
    residual = ey + 1j * k[None, :] * phi
    return {
        "field_gradient_residual_over_ey_rms": float(
            np.sqrt(np.mean(np.abs(residual) ** 2))
            / max(np.sqrt(np.mean(np.abs(ey) ** 2)), np.finfo(float).tiny)
        )
    }


def evaluate_prediction(
    method: str,
    truth: np.ndarray,
    prediction: np.ndarray,
    persistence: np.ndarray,
    representation: DirectRepresentation,
) -> dict:
    decoded_truth = representation.scaler.inverse(truth)
    decoded_prediction = representation.scaler.inverse(prediction)
    decoded_persistence = representation.scaler.inverse(persistence)
    state = augmented.scalar_metrics(truth, prediction, persistence)
    row = {
        "method": method,
        "finite_fraction": float(np.mean(np.isfinite(prediction))),
        "state_nrmse": state["nrmse"],
        "state_correlation": state["correlation"],
        "state_temporal_anomaly_correlation": state["temporal_anomaly_correlation"],
        "state_skill_vs_persistence": state["skill_vs_persistence"],
    }
    for name in GROUPS:
        metrics = augmented.scalar_metrics(
            decoded_truth[name],
            decoded_prediction[name],
            decoded_persistence[name],
        )
        row[f"{name}_nrmse"] = metrics["nrmse"]
        row[f"{name}_temporal_anomaly_correlation"] = metrics[
            "temporal_anomaly_correlation"
        ]
        row[f"{name}_skill_vs_persistence"] = metrics["skill_vs_persistence"]

    truth_coefficients = unpack_physical_fourier(decoded_truth["physical_fourier"])
    prediction_coefficients = unpack_physical_fourier(
        decoded_prediction["physical_fourier"]
    )
    persistence_coefficients = unpack_physical_fourier(
        decoded_persistence["physical_fourier"]
    )
    for field_index, field in enumerate(FIELD_NAMES):
        metrics = augmented.scalar_metrics(
            truth_coefficients[:, field_index],
            prediction_coefficients[:, field_index],
            persistence_coefficients[:, field_index],
        )
        row[f"{field}_nrmse"] = metrics["nrmse"]
        row[f"{field}_skill_vs_persistence"] = metrics["skill_vs_persistence"]
        row[f"{field}_temporal_anomaly_correlation"] = metrics[
            "temporal_anomaly_correlation"
        ]
    for mode in (2, 7):
        index = mode - 1
        truth_amplitude = np.abs(truth_coefficients[:, 0, index])
        prediction_amplitude = np.abs(prediction_coefficients[:, 0, index])
        persistence_amplitude = np.abs(persistence_coefficients[:, 0, index])
        metrics = augmented.scalar_metrics(
            truth_amplitude, prediction_amplitude, persistence_amplitude
        )
        row[f"phi_n{mode}_amplitude_skill"] = metrics["skill_vs_persistence"]
        row[f"phi_n{mode}_amplitude_correlation"] = metrics["correlation"]

    truth_cross = unpack_cross(decoded_truth["cross"])
    prediction_cross = unpack_cross(decoded_prediction["cross"])
    persistence_cross = unpack_cross(decoded_persistence["cross"])
    truth_transport = augmented.transport_from_cross(
        truth_cross, representation.macro_weights
    )
    prediction_transport = augmented.transport_from_cross(
        prediction_cross, representation.macro_weights
    )
    persistence_transport = augmented.transport_from_cross(
        persistence_cross, representation.macro_weights
    )
    transport = augmented.scalar_metrics(
        truth_transport, prediction_transport, persistence_transport
    )
    row["transport_nrmse"] = transport["nrmse"]
    row["transport_temporal_anomaly_correlation"] = transport[
        "temporal_anomaly_correlation"
    ]
    row["transport_skill_vs_persistence"] = transport["skill_vs_persistence"]
    for band_index, band in enumerate(MODE_BANDS):
        metrics = augmented.scalar_metrics(
            truth_transport[:, band_index],
            prediction_transport[:, band_index],
            persistence_transport[:, band_index],
        )
        row[f"{band}_transport_skill"] = metrics["skill_vs_persistence"]
        row[f"{band}_transport_correlation"] = metrics["correlation"]
    row.update(field_gradient_metrics(prediction_coefficients))
    gate_values = (
        row["state_skill_vs_persistence"],
        row["radial_skill_vs_persistence"],
        row["MTSI_n1_6_transport_skill"],
        row["ECDI_n9_21_transport_skill"],
        row["phi_skill_vs_persistence"],
        row["efy_skill_vs_persistence"],
    )
    row["selection_score"] = float(min(gate_values))
    return row


def train_fixed(
    model: neural.ParametricDelayROM,
    dataset: DirectWindowDataset,
    scaler: DirectScaler,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> list[dict]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
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
            loss = batch_loss(model, history, target, parameter, scaler)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(history)
            count += len(history)
        rows.append({"stage": "e25_pretrain", "epoch": epoch, "train_loss": total / count})
        if epoch == 1 or epoch % 10 == 0:
            print(f"pretrain epoch={epoch:03d} loss={total / count:.6e}", flush=True)
    return rows


def train_transition(
    model: neural.ParametricDelayROM,
    train_data: DirectWindowDataset,
    validation_data: DirectWindowDataset,
    scaler: DirectScaler,
    truth: np.ndarray,
    persistence: np.ndarray,
    history: np.ndarray,
    representation: DirectRepresentation,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> tuple[neural.ParametricDelayROM, int, list[dict], dict, np.ndarray]:
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        validation_data, batch_size=batch_size, shuffle=False
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8, min_lr=1.0e-5
    )
    initial_prediction = rollout(model, history, len(truth), device)
    initial_metrics = evaluate_prediction(
        "direct_physical_state_rom", truth, initial_prediction, persistence, representation
    )
    best_score = initial_metrics["selection_score"]
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = initial_metrics
    best_prediction = initial_prediction
    stale = 0
    rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for current_history, target, parameter in train_loader:
            current_history = current_history.to(device)
            target = target.to(device)
            parameter = parameter.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model, current_history, target, parameter, scaler)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(current_history)
            count += len(current_history)
        window_validation = loader_loss(
            model, validation_loader, scaler, device
        )
        prediction = rollout(model, history, len(truth), device)
        metrics = evaluate_prediction(
            "direct_physical_state_rom", truth, prediction, persistence, representation
        )
        score = metrics["selection_score"]
        scheduler.step(score)
        rows.append({
            "stage": "transition_finetune",
            "epoch": epoch,
            "train_loss": total / count,
            "window_validation_loss": window_validation,
            "long_validation_score": score,
            "long_state_skill": metrics["state_skill_vs_persistence"],
            "long_radial_skill": metrics["radial_skill_vs_persistence"],
            "long_MTSI_transport_skill": metrics["MTSI_n1_6_transport_skill"],
            "long_ECDI_transport_skill": metrics["ECDI_n9_21_transport_skill"],
            "long_phi_skill": metrics["phi_skill_vs_persistence"],
            "long_efy_skill": metrics["efy_skill_vs_persistence"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        if score > best_score + 1.0e-5:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = metrics
            best_prediction = prediction
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"finetune epoch={epoch:03d} train={total / count:.6e} "
                f"window={window_validation:.6e} long_min_skill={score:.6e}",
                flush=True,
            )
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return model, best_epoch, rows, best_metrics, best_prediction


def save_representation(path: Path, representation: DirectRepresentation) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["state_dimension"] = STATE_DIMENSION
        handle.attrs["groups"] = ",".join(GROUPS)
        handle.attrs["field_names"] = ",".join(FIELD_NAMES)
        handle.create_dataset("mode_numbers", data=MODE_NUMBERS)
        handle.create_dataset("macro_weights", data=representation.macro_weights)
        for name in GROUPS:
            group = handle.require_group(f"scaler/{name}")
            group.attrs["slice_start"] = representation.scaler.slices[name].start
            group.attrs["slice_stop"] = representation.scaler.slices[name].stop
            group.create_dataset("mean", data=representation.scaler.means[name])
            group.create_dataset("scale", data=representation.scaler.scales[name])
        for name, model in representation.block_models.items():
            group = handle.require_group(f"block_pca/{name}")
            group.attrs["mode_start"] = model.mode_start
            group.attrs["mode_end"] = model.mode_end
            group.create_dataset("full_mean", data=model.full_mean)
            group.create_dataset("active", data=model.active)
            group.create_dataset("pca_mean", data=model.pca.mean_)
            group.create_dataset("pca_components", data=model.pca.components_)


def plot_rollout(
    path: Path,
    time_us: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    persistence: np.ndarray,
    representation: DirectRepresentation,
) -> None:
    decoded = {
        "truth": representation.scaler.inverse(truth),
        "prediction": representation.scaler.inverse(prediction),
        "persistence": representation.scaler.inverse(persistence),
    }
    transport = {}
    for name, values in decoded.items():
        transport[name] = augmented.transport_from_cross(
            unpack_cross(values["cross"]), representation.macro_weights
        )
    coefficients = {
        name: unpack_physical_fourier(values["physical_fourier"])
        for name, values in decoded.items()
    }
    figure, axes = plt.subplots(4, 1, figsize=(10.5, 10.5), sharex=True)
    for band_index, band in enumerate(MODE_BANDS):
        for name, color, style in (
            ("truth", "#111111", "-"),
            ("prediction", "#0072B2", "-"),
            ("persistence", "#999999", ":"),
        ):
            axes[band_index].plot(
                time_us, transport[name][:, band_index], color=color,
                linestyle=style, label=name
            )
        axes[band_index].set_ylabel(f"{band}\ntransport")
        axes[band_index].grid(alpha=0.25)
        axes[band_index].legend(fontsize=8)
    for axis_offset, mode in enumerate((2, 7), start=2):
        for name, color, style in (
            ("truth", "#111111", "-"),
            ("prediction", "#0072B2", "-"),
            ("persistence", "#999999", ":"),
        ):
            amplitude = np.abs(coefficients[name][:, 0, mode - 1])
            axes[axis_offset].plot(
                time_us, amplitude, color=color, linestyle=style, label=name
            )
        axes[axis_offset].set_ylabel(f"global phi n={mode}\namplitude")
        axes[axis_offset].grid(alpha=0.25)
        axes[axis_offset].legend(fontsize=8)
    axes[-1].set_xlabel("time [us]")
    figure.suptitle("Direct physical-state ROM: E20 to E22.5 development rollout")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--history-steps", type=int, default=40)
    parser.add_argument("--rollout-steps", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--delta-limit", type=float, default=0.20)
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--pretrain-learning-rate", type=float, default=8.0e-4)
    parser.add_argument("--finetune-learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
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

    print("[1/6] Loading allowed stationary and up-transition data", flush=True)
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
    representation = fit_direct_representation(trajectories, fit_masks)
    save_representation(output / "representation.h5", representation)
    write_csv(output / "block_pca_summary.csv", representation.pca_rows)

    pretrain_data = DirectWindowDataset(
        [(
            e25, representation.states["e25_stationary"],
            stage2.E25_FIT_START_US, stage2.E25_FIT_END_US, 25.0, 2,
        )],
        args.history_steps,
        args.rollout_steps,
    )
    train_data = DirectWindowDataset(
        [
            (
                e25, representation.states["e25_stationary"],
                stage2.E25_FIT_START_US, stage2.E25_FIT_END_US, 25.0, 2,
            ),
            (
                up, representation.states["e20_to_e22p5"],
                float(up.time_us[0]), stage2.UP_TRAIN_END_US, 22.5, 1,
            ),
        ],
        args.history_steps,
        args.rollout_steps,
    )
    validation_data = DirectWindowDataset(
        [(
            up, representation.states["e20_to_e22p5"],
            stage2.UP_VALIDATION_START_US, float(up.time_us[-1] + 0.015),
            22.5, 1,
        )],
        args.history_steps,
        args.rollout_steps,
    )
    print(
        f"state_dim={STATE_DIMENSION} windows: pretrain={len(pretrain_data)} "
        f"train={len(train_data)} validation={len(validation_data)}",
        flush=True,
    )

    model = neural.ParametricDelayROM(
        STATE_DIMENSION, args.hidden_dim, args.delta_limit
    ).to(device)
    print("[2/6] Pretraining stationary E25 direct-state dynamics", flush=True)
    pretrain_rows = train_fixed(
        model, pretrain_data, representation.scaler,
        args.pretrain_epochs, args.batch_size, args.pretrain_learning_rate,
        args.weight_decay, args.seed, device,
    )

    validation_start = int(np.flatnonzero(
        up.time_us >= stage2.UP_VALIDATION_START_US - 1.0e-10
    )[0])
    up_state = representation.states["e20_to_e22p5"]
    history = up_state[validation_start - args.history_steps:validation_start]
    truth = up_state[validation_start:]
    persistence = np.repeat(history[-1:, :], len(truth), axis=0)
    print("[3/6] Fine-tuning and selecting on full 35--40 us rollout", flush=True)
    model, best_epoch, finetune_rows, metrics, prediction = train_transition(
        model, train_data, validation_data, representation.scaler,
        truth, persistence, history, representation,
        args.finetune_epochs, args.patience, args.batch_size,
        args.finetune_learning_rate, args.weight_decay, args.seed + 1, device,
    )
    print(json.dumps(json_safe(metrics), indent=2), flush=True)

    accepted = bool(
        metrics["finite_fraction"] == 1.0
        and metrics["state_skill_vs_persistence"] > 0.0
        and metrics["radial_skill_vs_persistence"] > 0.0
        and metrics["MTSI_n1_6_transport_skill"] > 0.0
        and metrics["ECDI_n9_21_transport_skill"] > 0.0
        and metrics["phi_skill_vs_persistence"] > 0.0
        and metrics["efy_skill_vs_persistence"] > 0.0
    )
    status = "READY_FOR_PHYSICS_ABLATION" if accepted else "REJECTED_DEVELOPMENT"

    print("[4/6] Saving locked direct-state candidate", flush=True)
    checkpoint_path = output / "direct_physical_state_rom_data_only.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "state_dimension": STATE_DIMENSION,
        "hidden_dimension": args.hidden_dim,
        "delta_limit": args.delta_limit,
        "history_steps": args.history_steps,
        "rollout_steps": args.rollout_steps,
        "best_epoch": best_epoch,
        "representation": str((output / "representation.h5").resolve()),
    }, checkpoint_path)
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
        handle.create_dataset("persistence", data=persistence, compression="gzip")
    plot_rollout(
        output / "development_rollout_35to40us.png",
        up.time_us[validation_start:], truth, prediction, persistence, representation,
    )

    print("[5/6] Writing audit lock", flush=True)
    lock = {
        "status": status,
        "accepted_for_physics_ablation": accepted,
        "acceptance_rule": (
            "finite and positive skill versus persistence for state, radial, "
            "phi, Ey, MTSI transport, and ECDI transport"
        ),
        "development_metrics": metrics,
        "state_definition": {
            "latent": 20,
            "global_physical_fourier_phi_ne_ni_Ey_n1_21_ri": 168,
            "radial_phi_envelopes": 8,
            "radial_cross_spectrum_ri": 16,
            "total": STATE_DIMENSION,
        },
        "best_epoch": best_epoch,
        "development_training_intervals_us": {
            "E25_stationary": [stage2.E25_FIT_START_US, stage2.E25_FIT_END_US],
            "E20_to_E22p5": [float(up.time_us[0]), stage2.UP_TRAIN_END_US],
        },
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
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "representation": str((output / "representation.h5").resolve()),
        "representation_sha256": sha256(output / "representation.h5"),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "arguments": vars(args),
    }
    (output / "model_lock.json").write_text(
        json.dumps(json_safe(lock), indent=2), encoding="utf-8"
    )
    readme = f"""# Direct physical-state RadAz ROM

- Status: `{status}`
- State dimension: {STATE_DIMENSION}
- Best transition epoch: {best_epoch}
- State skill: {metrics['state_skill_vs_persistence']:.6f}
- Phi skill: {metrics['phi_skill_vs_persistence']:.6f}
- Ey skill: {metrics['efy_skill_vs_persistence']:.6f}
- MTSI transport skill: {metrics['MTSI_n1_6_transport_skill']:.6f}
- ECDI transport skill: {metrics['ECDI_n9_21_transport_skill']:.6f}
- Primary E25 -> E22.5 data loaded: **no**

The transition checkpoint was selected by the minimum of the full-rollout
state, radial, phi, Ey, MTSI-transport, and ECDI-transport skills.  Short-window
validation loss was recorded but was not used for checkpoint selection.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print("[6/6] Complete", flush=True)
    print(f"status={status}", flush=True)
    print(f"output={output}", flush=True)


if __name__ == "__main__":
    main()
