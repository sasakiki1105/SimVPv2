"""Train a carrier/envelope high-mode branch for the RadAz transition ROM.

Mode-wise carrier phase increments are estimated only from the final 1 us of
each allowed training trajectory.  The recurrent model advances a 150D slow
state containing high latent coordinates, demodulated physical coefficients
for n=2 and n=7--21, ECDI radial envelopes, and ECDI cross spectra.  Complex
amplitude, phase, and phase-increment losses supplement the state rollout loss.
The low/MTSI branch is frozen and the E25 -> E22.5 primary data are not read.
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
from torch.utils.data import DataLoader

import analyze_radaz_augmented_physical_state_dynamics as augmented
import build_radaz_mode_factorized_rom as factorized
import train_radaz_direct_physical_state_rom as direct
import train_radaz_parametric_neural_delay_rom as neural
import train_radaz_regime_aware_transition_rom as stage2


ROOT = Path(__file__).resolve().parent
DEFAULT_LOW = ROOT / "workdirs" / "train_radaz_regime_aware_transition_rom_h160"
DEFAULT_OUTPUT = ROOT / "workdirs" / "train_radaz_carrier_envelope_high_branch_rom"
GROUPS = ("latent_high", "carrier_physical", "radial_ecdi", "cross_ecdi")
SELECTED_MODE_INDICES = factorized.PHYSICS_MODE_INDICES
SELECTED_MODE_NUMBERS = direct.MODE_NUMBERS[SELECTED_MODE_INDICES]
STATE_DIMENSION = 10 + 4 * len(SELECTED_MODE_INDICES) * 2 + 4 + 8


@dataclass
class BranchScaler:
    slices: dict[str, slice]
    means: dict[str, np.ndarray]
    scales: dict[str, np.ndarray]

    @classmethod
    def fit(cls, groups: dict[str, np.ndarray]) -> "BranchScaler":
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
            scales[name] = np.where(standard > 1.0e-20, standard, 1.0)
        return cls(slices, means, scales)

    def transform(self, groups: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([
            (groups[name] - self.means[name]) / self.scales[name]
            for name in GROUPS
        ], axis=1)

    def inverse(self, states: np.ndarray) -> dict[str, np.ndarray]:
        return {
            name: states[:, self.slices[name]] * self.scales[name] + self.means[name]
            for name in GROUPS
        }


@dataclass
class CarrierRepresentation:
    scaler: BranchScaler
    groups: dict[str, dict[str, np.ndarray]]
    states: dict[str, np.ndarray]
    carrier_step_rad: dict[str, np.ndarray]
    carrier_coherence: dict[str, np.ndarray]
    macro_weights: np.ndarray
    amplitude_scale: np.ndarray


class ComplexLoss(nn.Module):
    def __init__(
        self,
        scaler: BranchScaler,
        amplitude_scale: np.ndarray,
        lambda_amplitude: float,
        lambda_phase: float,
        lambda_frequency: float,
    ) -> None:
        super().__init__()
        selected = scaler.slices["carrier_physical"]
        self.slice_start = selected.start
        self.slice_stop = selected.stop
        self.register_buffer(
            "mean",
            torch.as_tensor(scaler.means["carrier_physical"], dtype=torch.float32),
        )
        self.register_buffer(
            "scale",
            torch.as_tensor(scaler.scales["carrier_physical"], dtype=torch.float32),
        )
        self.register_buffer(
            "amplitude_scale",
            torch.as_tensor(amplitude_scale, dtype=torch.float32),
        )
        self.lambda_amplitude = float(lambda_amplitude)
        self.lambda_phase = float(lambda_phase)
        self.lambda_frequency = float(lambda_frequency)

    def phi_ri(self, states: torch.Tensor) -> torch.Tensor:
        selected = states[..., self.slice_start:self.slice_stop]
        raw = selected * self.scale + self.mean
        shaped = raw.reshape(
            *raw.shape[:-1],
            len(direct.FIELD_NAMES),
            len(SELECTED_MODE_INDICES),
            2,
        )
        return shaped[..., direct.FIELD_NAMES.index("phi"), :, :]

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pred = self.phi_ri(prediction)
        truth = self.phi_ri(target)
        epsilon = 1.0e-8
        pred_amplitude = torch.sqrt(torch.sum(pred.square(), dim=-1) + epsilon)
        truth_amplitude = torch.sqrt(torch.sum(truth.square(), dim=-1) + epsilon)
        amplitude_loss = nn.functional.smooth_l1_loss(
            torch.log1p(pred_amplitude / self.amplitude_scale),
            torch.log1p(truth_amplitude / self.amplitude_scale),
            beta=0.05,
        )
        dot = torch.sum(pred * truth, dim=-1)
        phase_cosine = dot / (pred_amplitude * truth_amplitude + epsilon)
        phase_weight = torch.clamp(
            truth_amplitude / self.amplitude_scale, min=0.0, max=3.0
        )
        phase_loss = torch.sum((1.0 - phase_cosine) * phase_weight) / torch.clamp(
            torch.sum(phase_weight), min=epsilon
        )
        pred_left = pred[:, :-1]
        pred_right = pred[:, 1:]
        truth_left = truth[:, :-1]
        truth_right = truth[:, 1:]
        pred_dot = torch.sum(pred_right * pred_left, dim=-1)
        pred_cross = (
            pred_right[..., 1] * pred_left[..., 0]
            - pred_right[..., 0] * pred_left[..., 1]
        )
        truth_dot = torch.sum(truth_right * truth_left, dim=-1)
        truth_cross = (
            truth_right[..., 1] * truth_left[..., 0]
            - truth_right[..., 0] * truth_left[..., 1]
        )
        pred_pair_norm = torch.sqrt(pred_dot.square() + pred_cross.square() + epsilon)
        truth_pair_norm = torch.sqrt(
            truth_dot.square() + truth_cross.square() + epsilon
        )
        increment_cosine = (
            pred_dot * truth_dot + pred_cross * truth_cross
        ) / (pred_pair_norm * truth_pair_norm + epsilon)
        frequency_weight = torch.clamp(
            truth_amplitude[:, 1:] / self.amplitude_scale, min=0.0, max=3.0
        )
        frequency_loss = torch.sum(
            (1.0 - increment_cosine) * frequency_weight
        ) / torch.clamp(torch.sum(frequency_weight), min=epsilon)
        total = (
            self.lambda_amplitude * amplitude_loss
            + self.lambda_phase * phase_loss
            + self.lambda_frequency * frequency_loss
        )
        return total, {
            "amplitude": amplitude_loss,
            "phase": phase_loss,
            "frequency": frequency_loss,
        }


def json_safe(value):
    return direct.json_safe(value)


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


def estimate_carrier(
    packed_physical: np.ndarray,
    time_us: np.ndarray,
    start_us: float,
    end_us: float,
) -> tuple[np.ndarray, np.ndarray]:
    coefficients = direct.unpack_physical_fourier(packed_physical)
    phi = coefficients[:, direct.FIELD_NAMES.index("phi")]
    mask = (time_us >= start_us - 1.0e-10) & (time_us < end_us - 1.0e-10)
    selected = phi[mask]
    products = selected[1:] * np.conj(selected[:-1])
    summed = np.sum(products, axis=0)
    phase_step = np.angle(summed)
    coherence = np.abs(summed) / np.maximum(
        np.sum(np.abs(products), axis=0), np.finfo(float).tiny
    )
    return phase_step, coherence


def demodulate(
    packed_physical: np.ndarray,
    phase_step: np.ndarray,
) -> np.ndarray:
    coefficients = direct.unpack_physical_fourier(packed_physical)
    index = np.arange(len(coefficients), dtype=np.float64)
    rotation = np.exp(-1j * index[:, None] * phase_step[None])
    demodulated = coefficients * rotation[:, None, :]
    selected = demodulated[:, :, SELECTED_MODE_INDICES]
    return np.stack([selected.real, selected.imag], axis=-1).reshape(
        len(selected), -1
    )


def remodulate(
    packed_selected: np.ndarray,
    phase_step: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    shaped = packed_selected.reshape(
        len(packed_selected),
        len(direct.FIELD_NAMES),
        len(SELECTED_MODE_INDICES),
        2,
    )
    coefficients = shaped[..., 0] + 1j * shaped[..., 1]
    selected_step = phase_step[SELECTED_MODE_INDICES]
    rotation = np.exp(1j * indices[:, None] * selected_step[None])
    return coefficients * rotation[:, None, :]


def fit_representation(
    trajectories: dict[str, stage2.Trajectory],
    fit_masks: dict[str, np.ndarray],
) -> CarrierRepresentation:
    base = direct.fit_direct_representation(trajectories, fit_masks)
    phase_steps = {}
    coherences = {}
    raw_groups = {}
    tail_intervals = {
        "e25_stationary": (23.0, 24.0),
        "e20_to_e22p5": (34.0, 35.0),
        "e22p5_to_e20": (33.8, 34.86),
    }
    for name, trajectory in trajectories.items():
        if name not in tail_intervals:
            raise KeyError(f"No carrier-fit tail interval for {name}")
        packed = base.groups[name]["physical_fourier"]
        phase_steps[name], coherences[name] = estimate_carrier(
            packed, trajectory.time_us, *tail_intervals[name]
        )
        radial_indices = np.asarray([1, 3, 5, 7], dtype=np.int64)
        cross_indices = np.arange(16).reshape(4, 2, 2)[:, 1, :].reshape(-1)
        raw_groups[name] = {
            "latent_high": base.groups[name]["latent"][:, 10:20],
            "carrier_physical": demodulate(packed, phase_steps[name]),
            "radial_ecdi": base.groups[name]["radial"][:, radial_indices],
            "cross_ecdi": base.groups[name]["cross"][:, cross_indices],
        }
    stacked = {
        group: np.concatenate([
            raw_groups[name][group][fit_masks[name]] for name in trajectories
        ], axis=0)
        for group in GROUPS
    }
    scaler = BranchScaler.fit(stacked)
    states = {
        name: scaler.transform(values) for name, values in raw_groups.items()
    }
    phi_fit = stacked["carrier_physical"].reshape(
        len(stacked["carrier_physical"]),
        len(direct.FIELD_NAMES),
        len(SELECTED_MODE_INDICES),
        2,
    )[:, direct.FIELD_NAMES.index("phi")]
    amplitude_scale = np.mean(
        np.sqrt(np.sum(phi_fit ** 2, axis=-1)), axis=0
    )
    amplitude_scale = np.maximum(amplitude_scale, np.max(amplitude_scale) * 1.0e-5)
    if states["e25_stationary"].shape[1] != STATE_DIMENSION:
        raise ValueError("Carrier state dimension mismatch")
    return CarrierRepresentation(
        scaler, raw_groups, states, phase_steps, coherences,
        base.macro_weights, amplitude_scale,
    )


def group_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scaler: BranchScaler,
) -> torch.Tensor:
    return torch.stack([
        nn.functional.smooth_l1_loss(
            prediction[..., scaler.slices[name]],
            target[..., scaler.slices[name]],
            beta=0.10,
        )
        for name in GROUPS
    ]).mean()


def total_loss(
    model: neural.ParametricDelayROM,
    history: torch.Tensor,
    target: torch.Tensor,
    parameter: torch.Tensor,
    scaler: BranchScaler,
    complex_loss: ComplexLoss,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction = model.rollout(history, parameter, target.shape[1])
    data = group_loss(prediction, target, scaler)
    carrier, terms = complex_loss(prediction, target)
    return data + carrier, {"data": data, "carrier": carrier, **terms}


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


def unpack_cross_ecdi(values: np.ndarray) -> np.ndarray:
    shaped = values.reshape(len(values), 4, 2)
    return shaped[..., 0] + 1j * shaped[..., 1]


def load_low_branch(path: Path) -> dict:
    with h5py.File(path / "development_rollout_35to40us.h5", "r") as handle:
        return {
            "time_us": np.asarray(handle["time_us"], dtype=np.float64),
            "state_truth": np.asarray(handle["truth/state"], dtype=np.float64),
            "state_prediction": np.asarray(handle["regime_aware_rom/state"], dtype=np.float64),
            "state_persistence": np.asarray(handle["persistence/state"], dtype=np.float64),
            "radial_truth": np.asarray(handle["truth/radial"], dtype=np.float64),
            "radial_prediction": np.asarray(handle["regime_aware_rom/radial"], dtype=np.float64),
            "radial_persistence": np.asarray(handle["persistence/radial"], dtype=np.float64),
            "transport_truth": np.asarray(handle["truth/transport"], dtype=np.float64),
            "transport_prediction": np.asarray(handle["regime_aware_rom/transport"], dtype=np.float64),
            "transport_persistence": np.asarray(handle["persistence/transport"], dtype=np.float64),
        }


def phase_frequency_mhz(coefficients: np.ndarray, dt_us: float) -> np.ndarray:
    products = coefficients[1:] * np.conj(coefficients[:-1])
    return np.angle(np.sum(products, axis=0)) / (2.0 * np.pi * dt_us)


def evaluate(
    truth: np.ndarray,
    prediction: np.ndarray,
    persistence: np.ndarray,
    representation: CarrierRepresentation,
    low: dict,
    full_indices: np.ndarray,
    dt_us: float,
) -> dict:
    decoded = {
        name: representation.scaler.inverse(values)
        for name, values in (
            ("truth", truth), ("prediction", prediction), ("persistence", persistence)
        )
    }
    coefficients = {
        name: remodulate(
            values["carrier_physical"],
            representation.carrier_step_rad["e20_to_e22p5"],
            full_indices,
        )
        for name, values in decoded.items()
    }
    high_state = augmented.scalar_metrics(truth, prediction, persistence)
    composite = {
        "truth": np.concatenate((low["state_truth"], truth), axis=1),
        "prediction": np.concatenate((low["state_prediction"], prediction), axis=1),
        "persistence": np.concatenate((low["state_persistence"], persistence), axis=1),
    }
    composite_metrics = augmented.scalar_metrics(
        composite["truth"], composite["prediction"], composite["persistence"]
    )
    ecdi_transport = {}
    for name, values in decoded.items():
        cross = unpack_cross_ecdi(values["cross_ecdi"])
        ecdi_transport[name] = -2.0 * np.real(
            np.einsum("r,tr->t", representation.macro_weights, cross)
        ) / augmented.B_T
    hybrid_transport = {
        "truth": np.column_stack((low["transport_truth"][:, 0], ecdi_transport["truth"])),
        "prediction": np.column_stack((low["transport_prediction"][:, 0], ecdi_transport["prediction"])),
        "persistence": np.column_stack((low["transport_persistence"][:, 0], ecdi_transport["persistence"])),
    }
    row = {
        "finite_fraction": float(np.mean(np.isfinite(prediction))),
        "high_state_skill_vs_carrier_persistence": high_state["skill_vs_persistence"],
        "composite_state_skill_vs_persistence": composite_metrics["skill_vs_persistence"],
        "radial_skill_vs_persistence": augmented.scalar_metrics(
            low["radial_truth"], low["radial_prediction"], low["radial_persistence"]
        )["skill_vs_persistence"],
    }
    transport_all = augmented.scalar_metrics(
        hybrid_transport["truth"], hybrid_transport["prediction"], hybrid_transport["persistence"]
    )
    row["transport_skill_vs_persistence"] = transport_all["skill_vs_persistence"]
    for band, index in (("MTSI_n1_6", 0), ("ECDI_n9_21", 1)):
        metrics = augmented.scalar_metrics(
            hybrid_transport["truth"][:, index],
            hybrid_transport["prediction"][:, index],
            hybrid_transport["persistence"][:, index],
        )
        row[f"{band}_transport_skill"] = metrics["skill_vs_persistence"]
        row[f"{band}_transport_correlation"] = metrics["correlation"]
    for field_index, field in ((0, "phi"), (3, "efy")):
        metrics = augmented.scalar_metrics(
            coefficients["truth"][:, field_index],
            coefficients["prediction"][:, field_index],
            coefficients["persistence"][:, field_index],
        )
        row[f"selected_{field}_skill_vs_carrier_persistence"] = metrics["skill_vs_persistence"]
        row[f"selected_{field}_nrmse"] = metrics["nrmse"]
        row[f"selected_{field}_temporal_anomaly_correlation"] = metrics[
            "temporal_anomaly_correlation"
        ]
    phi = {name: values[:, 0] for name, values in coefficients.items()}
    for mode in (2, 7):
        selected_index = int(np.flatnonzero(SELECTED_MODE_NUMBERS == mode)[0])
        metrics = augmented.scalar_metrics(
            np.abs(phi["truth"][:, selected_index]),
            np.abs(phi["prediction"][:, selected_index]),
            np.abs(phi["persistence"][:, selected_index]),
        )
        row[f"phi_n{mode}_amplitude_skill"] = metrics["skill_vs_persistence"]
        row[f"phi_n{mode}_amplitude_correlation"] = metrics["correlation"]
        truth_frequency = phase_frequency_mhz(phi["truth"][:, selected_index], dt_us)
        prediction_frequency = phase_frequency_mhz(phi["prediction"][:, selected_index], dt_us)
        row[f"phi_n{mode}_truth_frequency_MHz"] = float(truth_frequency)
        row[f"phi_n{mode}_prediction_frequency_MHz"] = float(prediction_frequency)
        row[f"phi_n{mode}_frequency_abs_error_MHz"] = float(
            abs(prediction_frequency - truth_frequency)
        )
    phi_all = coefficients["prediction"][:, 0]
    ey_all = coefficients["prediction"][:, 3]
    k = 2.0 * np.pi * SELECTED_MODE_NUMBERS / direct.AZIMUTHAL_LENGTH_M
    residual = ey_all + 1j * k[None] * phi_all
    row["field_gradient_residual_over_ey_rms"] = float(
        np.sqrt(np.mean(np.abs(residual) ** 2))
        / np.sqrt(np.mean(np.abs(ey_all) ** 2))
    )
    gates = (
        row["high_state_skill_vs_carrier_persistence"],
        row["composite_state_skill_vs_persistence"],
        row["ECDI_n9_21_transport_skill"],
        row["selected_phi_skill_vs_carrier_persistence"],
        row["selected_efy_skill_vs_carrier_persistence"],
        row["phi_n2_amplitude_skill"],
        row["phi_n7_amplitude_skill"],
    )
    row["minimum_gate_skill"] = float(min(gates))
    return row


def train_fixed(
    model: neural.ParametricDelayROM,
    dataset: direct.DirectWindowDataset,
    representation: CarrierRepresentation,
    complex_loss: ComplexLoss,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict]:
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.pretrain_learning_rate, weight_decay=args.weight_decay
    )
    rows = []
    for epoch in range(1, args.pretrain_epochs + 1):
        model.train()
        totals = {name: 0.0 for name in ("loss", "data", "carrier", "amplitude", "phase", "frequency")}
        count = 0
        for history, target, parameter in loader:
            history, target, parameter = history.to(device), target.to(device), parameter.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, terms = total_loss(
                model, history, target, parameter, representation.scaler, complex_loss
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += float(loss.detach()) * len(history)
            for name, value in terms.items():
                totals[name] += float(value.detach()) * len(history)
            count += len(history)
        row = {"stage": "pretrain", "epoch": epoch, **{name: value / count for name, value in totals.items()}}
        rows.append(row)
        if epoch == 1 or epoch % 10 == 0:
            print(f"pretrain epoch={epoch:03d} loss={row['loss']:.6e}", flush=True)
    return rows


def train_transition(
    model: neural.ParametricDelayROM,
    dataset: direct.DirectWindowDataset,
    representation: CarrierRepresentation,
    complex_loss: ComplexLoss,
    truth: np.ndarray,
    persistence: np.ndarray,
    history: np.ndarray,
    low: dict,
    validation_start: int,
    dt_us: float,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[neural.ParametricDelayROM, int, dict, np.ndarray, list[dict]]:
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 1),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.finetune_learning_rate, weight_decay=args.weight_decay
    )
    full_indices = np.arange(validation_start, validation_start + len(truth))
    initial_prediction = rollout(model, history, len(truth), device)
    best_metrics = evaluate(
        truth, initial_prediction, persistence, representation, low, full_indices, dt_us
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
        for current_history, target, parameter in loader:
            current_history, target, parameter = (
                current_history.to(device), target.to(device), parameter.to(device)
            )
            optimizer.zero_grad(set_to_none=True)
            loss, _ = total_loss(
                model, current_history, target, parameter,
                representation.scaler, complex_loss,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(current_history)
            count += len(current_history)
        prediction = rollout(model, history, len(truth), device)
        metrics = evaluate(
            truth, prediction, persistence, representation, low, full_indices, dt_us
        )
        score = metrics["minimum_gate_skill"]
        rows.append({
            "stage": "finetune", "epoch": epoch, "loss": total / count,
            "minimum_gate_skill": score,
            "ECDI_transport_skill": metrics["ECDI_n9_21_transport_skill"],
            "phi_skill": metrics["selected_phi_skill_vs_carrier_persistence"],
            "Ey_skill": metrics["selected_efy_skill_vs_carrier_persistence"],
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
                f"n7={metrics['phi_n7_amplitude_skill']:.6e}", flush=True,
            )
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    return model, best_epoch, best_metrics, best_prediction, rows


def save_representation(path: Path, representation: CarrierRepresentation) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["state_dimension"] = STATE_DIMENSION
        handle.attrs["groups"] = ",".join(GROUPS)
        handle.create_dataset("selected_mode_numbers", data=SELECTED_MODE_NUMBERS)
        handle.create_dataset("macro_weights", data=representation.macro_weights)
        handle.create_dataset("amplitude_scale", data=representation.amplitude_scale)
        for name in GROUPS:
            group = handle.require_group(f"scaler/{name}")
            group.attrs["slice_start"] = representation.scaler.slices[name].start
            group.attrs["slice_stop"] = representation.scaler.slices[name].stop
            group.create_dataset("mean", data=representation.scaler.means[name])
            group.create_dataset("scale", data=representation.scaler.scales[name])
        for name in representation.carrier_step_rad:
            group = handle.require_group(f"carrier/{name}")
            group.create_dataset("phase_step_rad", data=representation.carrier_step_rad[name])
            group.create_dataset("coherence", data=representation.carrier_coherence[name])


def plot_rollout(
    path: Path,
    time_us: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    persistence: np.ndarray,
    representation: CarrierRepresentation,
    validation_start: int,
) -> None:
    indices = np.arange(validation_start, validation_start + len(truth))
    decoded = {
        name: representation.scaler.inverse(values)
        for name, values in (("truth", truth), ("prediction", prediction), ("persistence", persistence))
    }
    coefficients = {
        name: remodulate(
            values["carrier_physical"],
            representation.carrier_step_rad["e20_to_e22p5"], indices,
        )
        for name, values in decoded.items()
    }
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 6.5), sharex=True)
    for axis, mode in zip(axes, (2, 7)):
        selected = int(np.flatnonzero(SELECTED_MODE_NUMBERS == mode)[0])
        for name, color, style in (
            ("truth", "#111111", "-"),
            ("prediction", "#0072B2", "-"),
            ("persistence", "#999999", ":"),
        ):
            axis.plot(
                time_us, np.abs(coefficients[name][:, 0, selected]),
                color=color, linestyle=style, label=name,
            )
        axis.set_ylabel(f"global phi n={mode}\namplitude")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[-1].set_xlabel("time [us]")
    figure.suptitle("Carrier-envelope high branch: development rollout")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low", type=Path, default=DEFAULT_LOW)
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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
            "e25_stationary", 25.0, stage2.DEFAULT_E25_FEATURES, stage2.DEFAULT_E25_PHYSICAL
        ),
        "e20_to_e22p5": stage2.load_trajectory(
            "e20_to_e22p5", 22.5, stage2.DEFAULT_UP_FEATURES, stage2.DEFAULT_UP_PHYSICAL
        ),
    }
    e25, up = trajectories["e25_stationary"], trajectories["e20_to_e22p5"]
    fit_masks = {
        "e25_stationary": stage2.interval_mask(e25.time_us, 12.0, 24.0),
        "e20_to_e22p5": up.time_us < 35.0 - 1.0e-10,
    }
    print("[1/6] Fitting training-only carrier/envelope representation", flush=True)
    representation = fit_representation(trajectories, fit_masks)
    save_representation(output / "representation.h5", representation)
    audit = {}
    dt_us = float(np.median(np.diff(up.time_us)))
    for name in trajectories:
        audit[name] = {
            "frequency_MHz": representation.carrier_step_rad[name]
            / (2.0 * np.pi * dt_us),
            "coherence": representation.carrier_coherence[name],
        }
    (output / "carrier_audit.json").write_text(
        json.dumps(json_safe(audit), indent=2), encoding="utf-8"
    )

    pretrain_data = direct.DirectWindowDataset(
        [(e25, representation.states["e25_stationary"], 12.0, 24.0, 25.0, 2)],
        args.history_steps, args.rollout_steps,
    )
    train_data = direct.DirectWindowDataset(
        [
            (e25, representation.states["e25_stationary"], 12.0, 24.0, 25.0, 2),
            (up, representation.states["e20_to_e22p5"], float(up.time_us[0]), 35.0, 22.5, 1),
        ],
        args.history_steps, args.rollout_steps,
    )
    complex_loss = ComplexLoss(
        representation.scaler, representation.amplitude_scale,
        args.lambda_amplitude, args.lambda_phase, args.lambda_frequency,
    ).to(device)
    model = neural.ParametricDelayROM(
        STATE_DIMENSION, args.hidden_dim, args.delta_limit
    ).to(device)
    print(
        f"state_dim={STATE_DIMENSION} windows: pretrain={len(pretrain_data)} train={len(train_data)}",
        flush=True,
    )
    print("[2/6] Pretraining stationary E25 carrier envelopes", flush=True)
    pretrain_rows = train_fixed(
        model, pretrain_data, representation, complex_loss, args, device
    )

    validation_start = int(np.flatnonzero(up.time_us >= 35.0 - 1.0e-10)[0])
    state = representation.states["e20_to_e22p5"]
    history = state[validation_start - args.history_steps:validation_start]
    truth = state[validation_start:]
    # Persistence in the carrier frame is a frozen-envelope baseline.
    persistence = np.repeat(history[-1:], len(truth), axis=0)
    low = load_low_branch(args.low.resolve())
    if not np.allclose(low["time_us"], up.time_us[validation_start:]):
        raise ValueError("Low branch time alignment mismatch")
    print("[3/6] Fine-tuning and selecting on the full carrier-aware rollout", flush=True)
    model, best_epoch, metrics, prediction, finetune_rows = train_transition(
        model, train_data, representation, complex_loss,
        truth, persistence, history, low, validation_start, dt_us, args, device,
    )
    print(json.dumps(json_safe(metrics), indent=2), flush=True)
    accepted = bool(
        metrics["finite_fraction"] == 1.0
        and metrics["minimum_gate_skill"] > 0.0
        and metrics["radial_skill_vs_persistence"] > 0.0
        and metrics["MTSI_n1_6_transport_skill"] > 0.0
    )
    status = "READY_FOR_PHYSICS_ABLATION" if accepted else "REJECTED_DEVELOPMENT"

    print("[4/6] Saving carrier branch and development rollout", flush=True)
    checkpoint = output / "carrier_envelope_high_branch_data_only.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "state_dimension": STATE_DIMENSION,
        "hidden_dimension": args.hidden_dim,
        "delta_limit": args.delta_limit,
        "history_steps": args.history_steps,
        "rollout_steps": args.rollout_steps,
        "best_epoch": best_epoch,
        "representation": str((output / "representation.h5").resolve()),
    }, checkpoint)
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
    plot_rollout(
        output / "development_rollout_35to40us.png",
        up.time_us[validation_start:], truth, prediction, persistence,
        representation, validation_start,
    )

    print("[5/6] Writing prospective model lock", flush=True)
    lock = {
        "status": status,
        "accepted_for_physics_ablation": accepted,
        "development_metrics": metrics,
        "state_definition": {
            "latent_transition_ECDI": 10,
            "demodulated_phi_ne_ni_Ey_modes_n2_n7_21_ri": 128,
            "radial_ECDI": 4,
            "cross_ECDI_ri": 8,
            "total": STATE_DIMENSION,
        },
        "carrier_fit_intervals_us": {
            "E25_stationary": [23.0, 24.0],
            "E20_to_E22p5": [34.0, 35.0],
        },
        "best_epoch": best_epoch,
        "primary_test": {
            "direction": "E25_to_E22.5", "data_loaded": False,
            "used_for_carrier_fit": False, "used_for_selection": False,
        },
        "low_branch": str(args.low.resolve() / "regime_aware_transition_rom_data_only.pt"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "arguments": vars(args),
    }
    (output / "model_lock.json").write_text(
        json.dumps(json_safe(lock), indent=2), encoding="utf-8"
    )
    readme = f"""# Carrier-envelope high branch ROM

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
