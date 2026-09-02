"""Train the Stage-2 regime-aware ROM for the RadAz electric-field sweep.

The model is deliberately developed without reading the primary
E25 -> E22.5 kV/m hysteresis test.  It is first pretrained on the stationary
E25 trajectory and then updated with the E20 -> E22.5 transition over
30--35 us.  The continuation over 35--40 us is used only as development
validation.

The common 30-dimensional state contains 20 blockwise Fourier latent PCA
coordinates, eight radial potential-envelope observables, and two modal
transport observables.  A GRU residual propagator consumes a finite state
history and the applied electric field and is trained with a group-balanced
free-rollout loss.
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
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, Dataset

import analyze_radaz_augmented_physical_state_dynamics as augmented
import analyze_radaz_blockwise_fourier_latent_dynamics as block
import analyze_radaz_hankel_havok as hankel
import train_radaz_parametric_neural_delay_rom as neural


ROOT = Path(__file__).resolve().parent
TRANSITION_ROOT = ROOT / "workdirs" / "radaz_e20_to_e22p5_transition"
DEFAULT_OUTPUT = ROOT / "workdirs" / "train_radaz_regime_aware_transition_rom"
DEFAULT_E25_FEATURES = (
    ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_ez25kvm_fourier_latent_raw"
    / "fourier_latent_features.h5"
)
DEFAULT_E25_PHYSICAL = (
    ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_ez25kvm_fourier_latent_to_physical_modes"
    / "physical_fourier_targets.h5"
)
DEFAULT_UP_FEATURES = (
    TRANSITION_ROOT / "fourier_latent_features_e25targetnorm.h5"
)
DEFAULT_UP_PHYSICAL = TRANSITION_ROOT / "physical_fourier_targets.h5"

STATE_GROUPS = ("latent", "radial", "transport")
MODE_BAND_LABELS = ("MTSI_n1_6", "ECDI_n9_21")
PARAMETER_CENTER_KVM = 25.0
PARAMETER_SCALE_KVM = 5.0
E25_FIT_START_US = 12.0
E25_FIT_END_US = 24.0
UP_TRAIN_END_US = 35.0
UP_VALIDATION_START_US = 35.0
LATENT_BUDGET = "medium_20"


@dataclass
class Trajectory:
    name: str
    electric_field_kvm: float
    feature_path: Path
    physical_path: Path
    features: np.ndarray
    time_us: np.ndarray
    frame: np.ndarray
    physical: augmented.PhysicalStates


@dataclass
class Representation:
    block_models: dict[str, block.BlockPCA]
    scaler: augmented.GroupScaler
    groups: dict[str, dict[str, np.ndarray]]
    states: dict[str, np.ndarray]
    pca_rows: list[dict]


@dataclass
class FitResult:
    model: neural.ParametricDelayROM
    best_epoch: int
    best_validation_loss: float
    rows: list[dict]


class WindowDataset(Dataset):
    """Free-rollout windows with target-time interval controls."""

    def __init__(
        self,
        specifications: list[
            tuple[Trajectory, np.ndarray, float, float, float, int]
        ],
        history_steps: int,
        rollout_steps: int,
    ) -> None:
        histories: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        parameters: list[float] = []
        provenance: list[tuple[str, int]] = []
        for trajectory, state, start_us, end_us, parameter_kvm, stride in specifications:
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
                targets.append(state[target_start:target_stop])
                parameters.append(
                    (parameter_kvm - PARAMETER_CENTER_KVM)
                    / PARAMETER_SCALE_KVM
                )
                provenance.append(
                    (trajectory.name, int(trajectory.frame[target_start]))
                )
        if not histories:
            raise ValueError("No trajectory windows were constructed")
        self.histories = torch.from_numpy(
            np.asarray(histories, dtype=np.float32)
        )
        self.targets = torch.from_numpy(
            np.asarray(targets, dtype=np.float32)
        )
        self.parameters = torch.as_tensor(parameters, dtype=torch.float32)
        self.provenance = provenance

    def __len__(self) -> int:
        return len(self.histories)

    def __getitem__(self, index: int):
        return self.histories[index], self.targets[index], self.parameters[index]


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


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


def load_trajectory(
    name: str,
    electric_field_kvm: float,
    feature_path: Path,
    physical_path: Path,
) -> Trajectory:
    if not feature_path.is_file():
        raise FileNotFoundError(feature_path)
    if not physical_path.is_file():
        raise FileNotFoundError(physical_path)
    features, time_us, frame = block.load_features(feature_path)
    physical = augmented.load_physical_states(physical_path)
    if not np.array_equal(frame, physical.frame):
        raise ValueError(f"Frame mismatch for {name}")
    if not np.allclose(time_us, physical.time_us, atol=1.0e-9, rtol=0.0):
        raise ValueError(f"Time mismatch for {name}")
    if not np.all(np.diff(frame) == 1):
        raise ValueError(f"Non-contiguous frames for {name}")
    if not np.allclose(np.diff(time_us), 0.015, atol=1.0e-9, rtol=0.0):
        raise ValueError(f"Non-uniform 15 ns sampling for {name}")
    return Trajectory(
        name=name,
        electric_field_kvm=electric_field_kvm,
        feature_path=feature_path.resolve(),
        physical_path=physical_path.resolve(),
        features=features,
        time_us=time_us,
        frame=frame,
        physical=physical,
    )


def interval_mask(time_us: np.ndarray, start_us: float, end_us: float) -> np.ndarray:
    return (time_us >= start_us - 1.0e-10) & (time_us < end_us - 1.0e-10)


def fit_representation(
    trajectories: dict[str, Trajectory],
    fit_masks: dict[str, np.ndarray],
) -> Representation:
    budget = block.BUDGETS[LATENT_BUDGET]
    models: dict[str, block.BlockPCA] = {}
    pca_rows: list[dict] = []
    for name, (mode_start, mode_end) in block.BLOCKS.items():
        fit_arrays = []
        feature_shape = None
        for trajectory_name, trajectory in trajectories.items():
            values = block.block_slice(
                trajectory.features, mode_start, mode_end
            )
            feature_shape = values.shape[1:]
            fit_arrays.append(values[fit_masks[trajectory_name]].reshape(
                int(np.count_nonzero(fit_masks[trajectory_name])), -1
            ))
        fit = np.concatenate(fit_arrays, axis=0)
        full_mean = np.mean(fit, axis=0)
        variance = np.var(fit, axis=0)
        floor = max(float(np.max(variance)) * 1.0e-12, 1.0e-20)
        active = variance > floor
        components = min(
            int(budget[name]),
            int(np.count_nonzero(active)),
            len(fit) - 1,
        )
        pca = PCA(
            n_components=components,
            svd_solver="randomized",
            random_state=0,
            iterated_power=5,
        )
        pca.fit(fit[:, active])
        models[name] = block.BlockPCA(
            name=name,
            mode_start=mode_start,
            mode_end=mode_end,
            components=components,
            feature_shape=tuple(feature_shape),
            full_mean=full_mean,
            active=active,
            pca=pca,
        )
        pca_rows.append({
            "block": name,
            "mode_start": mode_start,
            "mode_end": mode_end,
            "retained_components": components,
            "active_features": int(np.count_nonzero(active)),
            "total_features": int(fit.shape[1]),
            "explained_variance": float(np.sum(pca.explained_variance_ratio_)),
        })

    groups: dict[str, dict[str, np.ndarray]] = {}
    for trajectory_name, trajectory in trajectories.items():
        latent = np.concatenate(
            [models[name].transform(trajectory.features) for name in block.BLOCKS],
            axis=1,
        )
        physical = augmented.flatten_physical(trajectory.physical)
        groups[trajectory_name] = {
            "latent": latent,
            "radial": physical["radial"],
            "transport": physical["transport"],
        }
    stacked = {
        group: np.concatenate([
            groups[name][group][fit_masks[name]] for name in trajectories
        ], axis=0)
        for group in STATE_GROUPS
    }
    scaler = augmented.GroupScaler.fit(
        stacked, np.ones(len(stacked["latent"]), dtype=bool)
    )
    states = {
        name: scaler.transform(trajectory_groups)
        for name, trajectory_groups in groups.items()
    }
    dimensions = {name: groups["e25_stationary"][name].shape[1] for name in STATE_GROUPS}
    if dimensions != {"latent": 20, "radial": 8, "transport": 2}:
        raise ValueError(f"Unexpected state dimensions: {dimensions}")
    return Representation(models, scaler, groups, states, pca_rows)


def group_balanced_loss(
    model: neural.ParametricDelayROM,
    history: torch.Tensor,
    target: torch.Tensor,
    parameter: torch.Tensor,
    scaler: augmented.GroupScaler,
) -> torch.Tensor:
    prediction = model.rollout(history, parameter, target.shape[1])
    losses = []
    for name in STATE_GROUPS:
        selected = scaler.slices[name]
        weight = scaler.weights[name]
        losses.append(nn.functional.smooth_l1_loss(
            prediction[..., selected] / weight,
            target[..., selected] / weight,
            beta=0.10,
            reduction="mean",
        ))
    return torch.stack(losses).mean()


@torch.no_grad()
def dataset_loss(
    model: neural.ParametricDelayROM,
    loader: DataLoader,
    scaler: augmented.GroupScaler,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for history, target, parameter in loader:
        history = history.to(device)
        target = target.to(device)
        parameter = parameter.to(device)
        loss = group_balanced_loss(model, history, target, parameter, scaler)
        total += float(loss) * len(history)
        count += len(history)
    return total / max(count, 1)


def train_epochs(
    model: neural.ParametricDelayROM,
    train_data: WindowDataset,
    scaler: augmented.GroupScaler,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    stage: str,
    validation_data: WindowDataset | None = None,
    patience: int = 40,
) -> FitResult:
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = None
    if validation_data is not None:
        validation_loader = DataLoader(
            validation_data, batch_size=batch_size, shuffle=False
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1.0e-5
    )
    best_loss = float("inf")
    best_epoch = epochs
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    rows: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for history, target, parameter in train_loader:
            history = history.to(device)
            target = target.to(device)
            parameter = parameter.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = group_balanced_loss(
                model, history, target, parameter, scaler
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(history)
            count += len(history)
        train_loss = total / max(count, 1)
        if validation_loader is None:
            validation_loss = train_loss
        else:
            validation_loss = dataset_loss(
                model, validation_loader, scaler, device
            )
        scheduler.step(validation_loss)
        rows.append({
            "stage": stage,
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        if validation_loss < best_loss - 1.0e-7:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{stage} epoch={epoch:03d} train={train_loss:.6e} "
                f"validation={validation_loss:.6e}",
                flush=True,
            )
        if validation_loader is not None and stale >= patience:
            break
    model.load_state_dict(best_state)
    return FitResult(model, best_epoch, best_loss, rows)


@torch.no_grad()
def rollout_neural(
    model: neural.ParametricDelayROM,
    history: np.ndarray,
    electric_field_kvm: float,
    steps: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    history_tensor = torch.as_tensor(
        history[None], dtype=torch.float32, device=device
    )
    parameter = torch.as_tensor(
        [(electric_field_kvm - PARAMETER_CENTER_KVM) / PARAMETER_SCALE_KVM],
        dtype=torch.float32,
        device=device,
    )
    prediction = model.rollout(history_tensor, parameter, steps)[0]
    return prediction.detach().cpu().numpy().astype(np.float64)


def metric_row(
    method: str,
    truth: np.ndarray,
    prediction: np.ndarray,
    persistence: np.ndarray,
    scaler: augmented.GroupScaler,
) -> dict:
    state = augmented.scalar_metrics(truth, prediction, persistence)
    decoded_truth = scaler.inverse(truth)
    decoded_prediction = scaler.inverse(prediction)
    decoded_persistence = scaler.inverse(persistence)
    radial = augmented.scalar_metrics(
        decoded_truth["radial"],
        decoded_prediction["radial"],
        decoded_persistence["radial"],
    )
    transport = augmented.scalar_metrics(
        decoded_truth["transport"],
        decoded_prediction["transport"],
        decoded_persistence["transport"],
    )
    row = {
        "method": method,
        "finite_fraction": float(np.mean(np.isfinite(prediction))),
        "state_nrmse": state["nrmse"],
        "state_correlation": state["correlation"],
        "state_temporal_anomaly_correlation": state["temporal_anomaly_correlation"],
        "state_skill_vs_persistence": state["skill_vs_persistence"],
        "radial_nrmse": radial["nrmse"],
        "radial_temporal_anomaly_correlation": radial["temporal_anomaly_correlation"],
        "radial_skill_vs_persistence": radial["skill_vs_persistence"],
        "transport_nrmse": transport["nrmse"],
        "transport_temporal_anomaly_correlation": transport["temporal_anomaly_correlation"],
        "transport_skill_vs_persistence": transport["skill_vs_persistence"],
    }
    for band, label in enumerate(MODE_BAND_LABELS):
        selected = augmented.scalar_metrics(
            decoded_truth["transport"][:, band],
            decoded_prediction["transport"][:, band],
            decoded_persistence["transport"][:, band],
        )
        row[f"{label}_transport_correlation"] = selected["correlation"]
        row[f"{label}_transport_skill"] = selected["skill_vs_persistence"]
    return row


def save_representation(path: Path, representation: Representation) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["state_groups"] = ",".join(STATE_GROUPS)
        handle.attrs["state_dimension"] = 30
        for name, model in representation.block_models.items():
            group = handle.require_group(f"block_pca/{name}")
            group.attrs["mode_start"] = model.mode_start
            group.attrs["mode_end"] = model.mode_end
            group.attrs["components"] = model.components
            group.create_dataset("feature_shape", data=model.feature_shape)
            group.create_dataset("full_mean", data=model.full_mean)
            group.create_dataset("active", data=model.active)
            group.create_dataset("pca_mean", data=model.pca.mean_)
            group.create_dataset("pca_components", data=model.pca.components_)
            group.create_dataset(
                "pca_explained_variance", data=model.pca.explained_variance_
            )
        for name in STATE_GROUPS:
            group = handle.require_group(f"scaler/{name}")
            group.attrs["weight"] = representation.scaler.weights[name]
            group.attrs["slice_start"] = representation.scaler.slices[name].start
            group.attrs["slice_stop"] = representation.scaler.slices[name].stop
            group.create_dataset("mean", data=representation.scaler.means[name])
            group.create_dataset("scale", data=representation.scaler.scales[name])


def save_rollout(
    path: Path,
    time_us: np.ndarray,
    frame: np.ndarray,
    arrays: dict[str, np.ndarray],
    scaler: augmented.GroupScaler,
) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["development_direction"] = "E20_to_E22.5"
        handle.attrs["primary_E25_to_E22.5_loaded"] = False
        handle.create_dataset("time_us", data=time_us)
        handle.create_dataset("frame", data=frame)
        for method, values in arrays.items():
            group = handle.require_group(method)
            group.create_dataset("state", data=values, compression="gzip")
            for name, decoded in scaler.inverse(values).items():
                group.create_dataset(name, data=decoded, compression="gzip")


def plot_results(
    path: Path,
    time_us: np.ndarray,
    arrays: dict[str, np.ndarray],
    scaler: augmented.GroupScaler,
) -> None:
    decoded = {
        method: scaler.inverse(values) for method, values in arrays.items()
    }
    colors = {
        "truth": "#111111",
        "regime_aware_rom": "#0072B2",
        "e25_pretrained_rom": "#E69F00",
        "e25_fixed_hankel": "#009E73",
        "persistence": "#999999",
    }
    labels = {
        "truth": "PIC truth",
        "regime_aware_rom": "regime-aware ROM",
        "e25_pretrained_rom": "E25 pretrained ROM",
        "e25_fixed_hankel": "E25 fixed Hankel",
        "persistence": "persistence",
    }
    figure, axes = plt.subplots(3, 1, figsize=(10.5, 8.5), sharex=True)
    for band, band_label in enumerate(MODE_BAND_LABELS):
        for method in arrays:
            axes[band].plot(
                time_us,
                decoded[method]["transport"][:, band],
                color=colors[method],
                linewidth=1.5 if method == "truth" else 1.0,
                label=labels[method],
            )
        axes[band].set_ylabel(f"{band_label}\ntransport")
        axes[band].grid(alpha=0.25)
        axes[band].legend(fontsize=7, ncol=3)
    truth = arrays["truth"]
    for method in ("regime_aware_rom", "e25_pretrained_rom", "e25_fixed_hankel", "persistence"):
        error = np.sqrt(np.mean((arrays[method] - truth) ** 2, axis=1))
        axes[2].plot(time_us, error, color=colors[method], label=labels[method])
    axes[2].set_ylabel("standardized state RMSE")
    axes[2].set_xlabel("time [us]")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)
    figure.suptitle("E20 to E22.5 development continuation: 35--40 us")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_losses(path: Path, rows: list[dict]) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    offset = 0
    for stage in ("e25_pretrain", "transition_finetune"):
        selected = [row for row in rows if row["stage"] == stage]
        x = np.arange(1, len(selected) + 1) + offset
        axis.semilogy(x, [row["train_loss"] for row in selected], label=f"{stage} train")
        if stage == "transition_finetune":
            axis.semilogy(
                x,
                [row["validation_loss"] for row in selected],
                label="up-sweep 35--40 validation",
            )
        offset += len(selected)
    axis.set_xlabel("epoch (stage-concatenated)")
    axis.set_ylabel("group-balanced rollout loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--e25-features", type=Path, default=DEFAULT_E25_FEATURES)
    parser.add_argument("--e25-physical", type=Path, default=DEFAULT_E25_PHYSICAL)
    parser.add_argument("--up-features", type=Path, default=DEFAULT_UP_FEATURES)
    parser.add_argument("--up-physical", type=Path, default=DEFAULT_UP_PHYSICAL)
    parser.add_argument("--history-steps", type=int, default=40)
    parser.add_argument("--rollout-steps", type=int, default=40)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--delta-limit", type=float, default=0.25)
    parser.add_argument("--pretrain-epochs", type=int, default=80)
    parser.add_argument("--finetune-epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pretrain-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--finetune-learning-rate", type=float, default=5.0e-4)
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

    print("[1/7] Loading stationary E25 and E20->E22.5 development data", flush=True)
    trajectories = {
        "e25_stationary": load_trajectory(
            "e25_stationary", 25.0, args.e25_features, args.e25_physical
        ),
        "e20_to_e22p5": load_trajectory(
            "e20_to_e22p5", 22.5, args.up_features, args.up_physical
        ),
    }
    e25 = trajectories["e25_stationary"]
    up = trajectories["e20_to_e22p5"]
    if up.time_us[0] >= UP_TRAIN_END_US or up.time_us[-1] < UP_VALIDATION_START_US:
        raise ValueError("Transition trajectory does not span the declared split")
    fit_masks = {
        "e25_stationary": interval_mask(
            e25.time_us, E25_FIT_START_US, E25_FIT_END_US
        ),
        "e20_to_e22p5": up.time_us < UP_TRAIN_END_US - 1.0e-10,
    }

    print("[2/7] Fitting leakage-free common L+R+T representation", flush=True)
    representation = fit_representation(trajectories, fit_masks)
    save_representation(output / "representation.h5", representation)
    write_csv(output / "block_pca_summary.csv", representation.pca_rows)

    pretrain_data = WindowDataset(
        [(
            e25,
            representation.states["e25_stationary"],
            E25_FIT_START_US,
            E25_FIT_END_US,
            25.0,
            2,
        )],
        args.history_steps,
        args.rollout_steps,
    )
    mixed_data = WindowDataset(
        [
            (
                e25,
                representation.states["e25_stationary"],
                E25_FIT_START_US,
                E25_FIT_END_US,
                25.0,
                2,
            ),
            (
                up,
                representation.states["e20_to_e22p5"],
                float(up.time_us[0]),
                UP_TRAIN_END_US,
                22.5,
                1,
            ),
        ],
        args.history_steps,
        args.rollout_steps,
    )
    validation_data = WindowDataset(
        [(
            up,
            representation.states["e20_to_e22p5"],
            UP_VALIDATION_START_US,
            float(up.time_us[-1] + 0.015),
            22.5,
            1,
        )],
        args.history_steps,
        args.rollout_steps,
    )
    print(
        f"windows: pretrain={len(pretrain_data)} mixed={len(mixed_data)} "
        f"development_validation={len(validation_data)}",
        flush=True,
    )

    model = neural.ParametricDelayROM(
        state_dim=30,
        hidden_dim=args.hidden_dim,
        delta_limit=args.delta_limit,
    ).to(device)
    print("[3/7] Pretraining the E25 local neural ROM", flush=True)
    pretrain = train_epochs(
        model,
        pretrain_data,
        representation.scaler,
        device,
        args.pretrain_epochs,
        args.batch_size,
        args.pretrain_learning_rate,
        args.weight_decay,
        args.seed,
        "e25_pretrain",
    )
    pretrained_state = copy.deepcopy(pretrain.model.state_dict())

    print("[4/7] Fine-tuning with the 30--35 us up-sweep", flush=True)
    finetune = train_epochs(
        pretrain.model,
        mixed_data,
        representation.scaler,
        device,
        args.finetune_epochs,
        args.batch_size,
        args.finetune_learning_rate,
        args.weight_decay,
        args.seed + 1,
        "transition_finetune",
        validation_data=validation_data,
        patience=args.patience,
    )

    print("[5/7] Running the autonomous 35--40 us development forecast", flush=True)
    up_state = representation.states["e20_to_e22p5"]
    validation_start = int(np.flatnonzero(
        up.time_us >= UP_VALIDATION_START_US - 1.0e-10
    )[0])
    history = up_state[validation_start - args.history_steps:validation_start]
    truth = up_state[validation_start:]
    persistence = np.repeat(history[-1:, :], len(truth), axis=0)
    regime_prediction = rollout_neural(
        finetune.model, history, 22.5, len(truth), device
    )

    pretrained_model = neural.ParametricDelayROM(
        state_dim=30,
        hidden_dim=args.hidden_dim,
        delta_limit=args.delta_limit,
    ).to(device)
    pretrained_model.load_state_dict(pretrained_state)
    pretrained_prediction = rollout_neural(
        pretrained_model, history, 25.0, len(truth), device
    )

    e25_fit = representation.states["e25_stationary"][fit_masks["e25_stationary"]]
    fixed_hankel = hankel.fit_hankel_dmd(e25_fit, delay=40, rank=30)
    hankel_prediction = hankel.rollout_hankel(
        fixed_hankel, up_state[:validation_start], len(truth)
    )
    arrays = {
        "truth": truth,
        "regime_aware_rom": regime_prediction,
        "e25_pretrained_rom": pretrained_prediction,
        "e25_fixed_hankel": hankel_prediction,
        "persistence": persistence,
    }
    rows = [
        metric_row(method, truth, values, persistence, representation.scaler)
        for method, values in arrays.items()
        if method != "truth"
    ]
    selected = next(row for row in rows if row["method"] == "regime_aware_rom")
    accepted = bool(
        selected["finite_fraction"] == 1.0
        and selected["state_skill_vs_persistence"] > 0.0
        and selected["radial_skill_vs_persistence"] > 0.0
        and selected["MTSI_n1_6_transport_skill"] > 0.0
        and selected["ECDI_n9_21_transport_skill"] > 0.0
    )
    print(json.dumps(json_safe(selected), indent=2), flush=True)

    print("[6/7] Saving checkpoint, metrics, and diagnostic plots", flush=True)
    checkpoint = output / "regime_aware_transition_rom_data_only.pt"
    torch.save({
        "model_state_dict": finetune.model.state_dict(),
        "model_class": "ParametricDelayROM",
        "state_dimension": 30,
        "hidden_dimension": args.hidden_dim,
        "delta_limit": args.delta_limit,
        "history_steps": args.history_steps,
        "rollout_steps": args.rollout_steps,
        "parameter_center_kvm": PARAMETER_CENTER_KVM,
        "parameter_scale_kvm": PARAMETER_SCALE_KVM,
        "best_finetune_epoch": finetune.best_epoch,
        "representation": str((output / "representation.h5").resolve()),
    }, checkpoint)
    training_rows = pretrain.rows + finetune.rows
    write_csv(output / "training_history.csv", training_rows)
    write_csv(output / "development_metrics.csv", rows)
    (output / "development_metrics.json").write_text(
        json.dumps(json_safe(rows), indent=2), encoding="utf-8"
    )
    save_rollout(
        output / "development_rollout_35to40us.h5",
        up.time_us[validation_start:],
        up.frame[validation_start:],
        arrays,
        representation.scaler,
    )
    plot_results(
        output / "development_rollout_35to40us.png",
        up.time_us[validation_start:],
        arrays,
        representation.scaler,
    )
    plot_losses(output / "training_history.png", training_rows)

    script_path = Path(__file__).resolve()
    lock = {
        "status": "READY_FOR_BLIND_PRIMARY_TEST" if accepted else "REJECTED_DEVELOPMENT",
        "accepted_for_blind_E25_to_E22p5": accepted,
        "acceptance_rule": (
            "finite_fraction == 1 and state, radial, MTSI transport, and "
            "ECDI transport skills_vs_persistence are all > 0 on "
            "E20->E22.5 35--40 us"
        ),
        "development_metrics": selected,
        "development_direction": "E20_to_E22.5",
        "development_training_intervals_us": {
            "E25_stationary_pretrain": [E25_FIT_START_US, E25_FIT_END_US],
            "E20_to_E22.5_transition": [float(up.time_us[0]), UP_TRAIN_END_US],
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
        "best_finetune_epoch": finetune.best_epoch,
        "best_window_validation_loss": finetune.best_validation_loss,
        "fixed_E25_hankel_spectral_radius": float(
            np.max(np.abs(fixed_hankel.eigenvalues))
        ),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "representation": str((output / "representation.h5").resolve()),
        "representation_sha256": sha256(output / "representation.h5"),
        "script": str(script_path),
        "script_sha256": sha256(script_path),
        "source_files": {
            name: {
                "features": str(trajectory.feature_path),
                "physical": str(trajectory.physical_path),
            }
            for name, trajectory in trajectories.items()
        },
        "arguments": vars(args),
    }
    (output / "model_lock.json").write_text(
        json.dumps(json_safe(lock), indent=2), encoding="utf-8"
    )
    readme = f"""# RadAz Stage-2 regime-aware transition ROM

- Status: `{lock['status']}`
- Pretraining: stationary E25, {E25_FIT_START_US:g}--{E25_FIT_END_US:g} us
- Transition training: E20 -> E22.5, {up.time_us[0]:.3f}--{UP_TRAIN_END_US:g} us
- Development validation: E20 -> E22.5, {up.time_us[validation_start]:.3f}--{up.time_us[-1]:.3f} us
- Primary E25 -> E22.5 data loaded: **no**
- State skill versus persistence: {selected['state_skill_vs_persistence']:.6f}
- Transport skill versus persistence: {selected['transport_skill_vs_persistence']:.6f}
- State anomaly correlation: {selected['state_temporal_anomaly_correlation']:.6f}
- Transport anomaly correlation: {selected['transport_temporal_anomaly_correlation']:.6f}

The checkpoint may be applied once to the untouched E25 -> E22.5 primary test
only when the status is `READY_FOR_BLIND_PRIMARY_TEST`.  No architecture,
normalization, loss, epoch, or threshold may be changed after inspecting that
primary trajectory.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print("[7/7] Complete", flush=True)
    print(f"status={lock['status']}", flush=True)
    print(f"output={output}", flush=True)


if __name__ == "__main__":
    main()
