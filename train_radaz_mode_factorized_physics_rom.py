"""Physics-loss ablation for the high branch of the mode-factorized RadAz ROM.

The low/MTSI branch is frozen.  The direct physical high branch is fine-tuned
with a truth-floor spectral constraint E_y + i*k*phi = 0 on n=2 and n=7--21.
Candidate checkpoints are selected on the complete 35--40 us development
rollout while the E25 -> E22.5 primary trajectory remains unread.
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
DEFAULT_HIGH = ROOT / "workdirs" / "train_radaz_direct_physical_state_rom"
DEFAULT_FACTOR = ROOT / "workdirs" / "build_radaz_mode_factorized_rom"
DEFAULT_OUTPUT = ROOT / "workdirs" / "train_radaz_mode_factorized_physics_rom"
LAMBDA_E_CANDIDATES = (0.01, 0.1, 1.0)


@dataclass
class PhysicsStatistics:
    residual_scale: np.ndarray
    truth_floor_power: np.ndarray
    audit: dict


@dataclass
class Candidate:
    lambda_e: float
    model: neural.ParametricDelayROM
    epoch: int
    metrics: dict
    prediction: np.ndarray
    history: list[dict]


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


def physics_statistics(
    representation: direct.DirectRepresentation,
    fit_masks: dict[str, np.ndarray],
) -> PhysicsStatistics:
    packed = np.concatenate([
        representation.groups[name]["physical_fourier"][fit_masks[name]]
        for name in representation.groups
    ], axis=0)
    coefficients = direct.unpack_physical_fourier(packed)
    modes = factorized.PHYSICS_MODE_INDICES
    phi = coefficients[:, direct.FIELD_NAMES.index("phi")][:, modes]
    ey = coefficients[:, direct.FIELD_NAMES.index("efy")][:, modes]
    k = 2.0 * np.pi * direct.MODE_NUMBERS[modes] / direct.AZIMUTHAL_LENGTH_M
    residual = ey + 1j * k[None, :] * phi
    scale = np.sqrt(np.mean(np.abs(ey) ** 2, axis=0))
    scale = np.maximum(scale, np.max(scale) * 1.0e-6)
    floor = np.mean(np.abs(residual / scale[None]) ** 2, axis=0)
    audit = {
        "selected_modes": (modes + 1).tolist(),
        "truth_residual_over_ey_rms": float(
            np.sqrt(np.mean(np.abs(residual) ** 2))
            / np.sqrt(np.mean(np.abs(ey) ** 2))
        ),
        "truth_floor_normalized_rms": float(np.sqrt(np.mean(floor))),
    }
    return PhysicsStatistics(scale, floor, audit)


class FieldGradientConstraint(nn.Module):
    def __init__(
        self,
        scaler: direct.DirectScaler,
        statistics: PhysicsStatistics,
    ) -> None:
        super().__init__()
        selected = scaler.slices["physical_fourier"]
        self.slice_start = selected.start
        self.slice_stop = selected.stop
        self.register_buffer(
            "mean",
            torch.as_tensor(
                scaler.means["physical_fourier"], dtype=torch.float32
            ),
        )
        self.register_buffer(
            "scale",
            torch.as_tensor(
                scaler.scales["physical_fourier"], dtype=torch.float32
            ),
        )
        self.register_buffer(
            "mode_indices",
            torch.as_tensor(
                factorized.PHYSICS_MODE_INDICES, dtype=torch.long
            ),
        )
        selected_modes = direct.MODE_NUMBERS[factorized.PHYSICS_MODE_INDICES]
        self.register_buffer(
            "wave_numbers",
            torch.as_tensor(
                2.0 * np.pi * selected_modes / direct.AZIMUTHAL_LENGTH_M,
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
        standardized = states[..., self.slice_start:self.slice_stop]
        raw = standardized * self.scale + self.mean
        shaped = raw.reshape(
            *raw.shape[:-1], len(direct.FIELD_NAMES), len(direct.MODE_NUMBERS), 2
        )
        phi = shaped[..., direct.FIELD_NAMES.index("phi"), :, :][
            ..., self.mode_indices, :
        ]
        ey = shaped[..., direct.FIELD_NAMES.index("efy"), :, :][
            ..., self.mode_indices, :
        ]
        residual_real = ey[..., 0] - self.wave_numbers * phi[..., 1]
        residual_imag = ey[..., 1] + self.wave_numbers * phi[..., 0]
        return (
            residual_real.square() + residual_imag.square()
        ) / self.residual_scale.square()

    def loss(self, states: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.power(states) - self.truth_floor_power).mean()


def physics_metrics(
    prediction: np.ndarray,
    representation: direct.DirectRepresentation,
    statistics: PhysicsStatistics,
) -> dict:
    physical = representation.scaler.inverse(prediction)["physical_fourier"]
    coefficients = direct.unpack_physical_fourier(physical)
    modes = factorized.PHYSICS_MODE_INDICES
    phi = coefficients[:, direct.FIELD_NAMES.index("phi")][:, modes]
    ey = coefficients[:, direct.FIELD_NAMES.index("efy")][:, modes]
    k = 2.0 * np.pi * direct.MODE_NUMBERS[modes] / direct.AZIMUTHAL_LENGTH_M
    residual = ey + 1j * k[None, :] * phi
    power = np.abs(residual / statistics.residual_scale[None]) ** 2
    return {
        "field_gradient_normalized_rms": float(np.sqrt(np.mean(power))),
        "field_gradient_excess_hinge": float(np.mean(
            np.maximum(power - statistics.truth_floor_power[None], 0.0)
        )),
        "field_gradient_truth_floor_rms": float(
            np.sqrt(np.mean(statistics.truth_floor_power))
        ),
    }


def load_low_branch(path: Path) -> dict:
    with h5py.File(path / "development_rollout_35to40us.h5", "r") as handle:
        return {
            "time_us": np.asarray(handle["time_us"], dtype=np.float64),
            "frame": np.asarray(handle["frame"], dtype=np.int64),
            "state_truth": np.asarray(handle["truth/state"], dtype=np.float64),
            "state_prediction": np.asarray(
                handle["regime_aware_rom/state"], dtype=np.float64
            ),
            "state_persistence": np.asarray(
                handle["persistence/state"], dtype=np.float64
            ),
            "radial_truth": np.asarray(handle["truth/radial"], dtype=np.float64),
            "radial_prediction": np.asarray(
                handle["regime_aware_rom/radial"], dtype=np.float64
            ),
            "radial_persistence": np.asarray(
                handle["persistence/radial"], dtype=np.float64
            ),
            "transport_truth": np.asarray(
                handle["truth/transport"], dtype=np.float64
            ),
            "transport_prediction": np.asarray(
                handle["regime_aware_rom/transport"], dtype=np.float64
            ),
            "transport_persistence": np.asarray(
                handle["persistence/transport"], dtype=np.float64
            ),
        }


def high_indices(scaler: direct.DirectScaler) -> np.ndarray:
    physical = np.arange(168).reshape(4, 21, 2)[:, 6:, :].reshape(-1)
    radial = np.asarray([1, 3, 5, 7], dtype=np.int64)
    cross = np.arange(16).reshape(4, 2, 2)[:, 1, :].reshape(-1)
    result = []
    for name, indices in (
        ("physical_fourier", physical),
        ("radial", radial),
        ("cross", cross),
    ):
        result.extend((scaler.slices[name].start + indices).tolist())
    return np.asarray(result, dtype=np.int64)


def evaluate_composite(
    prediction: np.ndarray,
    truth: np.ndarray,
    persistence: np.ndarray,
    representation: direct.DirectRepresentation,
    low: dict,
    statistics: PhysicsStatistics,
) -> dict:
    indices = high_indices(representation.scaler)
    composite_truth = np.concatenate((low["state_truth"], truth[:, indices]), axis=1)
    composite_prediction = np.concatenate(
        (low["state_prediction"], prediction[:, indices]), axis=1
    )
    composite_persistence = np.concatenate(
        (low["state_persistence"], persistence[:, indices]), axis=1
    )
    state = augmented.scalar_metrics(
        composite_truth, composite_prediction, composite_persistence
    )
    radial = augmented.scalar_metrics(
        low["radial_truth"], low["radial_prediction"], low["radial_persistence"]
    )
    decoded = {
        name: representation.scaler.inverse(values)
        for name, values in (
            ("truth", truth), ("prediction", prediction), ("persistence", persistence)
        )
    }
    high_transport = {
        name: augmented.transport_from_cross(
            direct.unpack_cross(values["cross"]), representation.macro_weights
        )
        for name, values in decoded.items()
    }
    transport = {
        "truth": np.column_stack(
            (low["transport_truth"][:, 0], high_transport["truth"][:, 1])
        ),
        "prediction": np.column_stack(
            (
                low["transport_prediction"][:, 0],
                high_transport["prediction"][:, 1],
            )
        ),
        "persistence": np.column_stack(
            (
                low["transport_persistence"][:, 0],
                high_transport["persistence"][:, 1],
            )
        ),
    }
    transport_all = augmented.scalar_metrics(
        transport["truth"], transport["prediction"], transport["persistence"]
    )
    coefficients = {
        name: direct.unpack_physical_fourier(values["physical_fourier"])
        for name, values in decoded.items()
    }
    row = {
        "finite_fraction": float(np.mean(np.isfinite(prediction))),
        "state_skill_vs_persistence": state["skill_vs_persistence"],
        "state_temporal_anomaly_correlation": state[
            "temporal_anomaly_correlation"
        ],
        "radial_skill_vs_persistence": radial["skill_vs_persistence"],
        "transport_skill_vs_persistence": transport_all["skill_vs_persistence"],
    }
    for band_index, band in enumerate(("MTSI_n1_6", "ECDI_n9_21")):
        metrics = augmented.scalar_metrics(
            transport["truth"][:, band_index],
            transport["prediction"][:, band_index],
            transport["persistence"][:, band_index],
        )
        row[f"{band}_transport_skill"] = metrics["skill_vs_persistence"]
        row[f"{band}_transport_correlation"] = metrics["correlation"]
    for field_index, field in ((0, "phi"), (3, "efy")):
        selected = {
            name: values[:, field_index, factorized.PHYSICS_MODE_INDICES]
            for name, values in coefficients.items()
        }
        metrics = augmented.scalar_metrics(
            selected["truth"], selected["prediction"], selected["persistence"]
        )
        row[f"selected_{field}_skill_vs_persistence"] = metrics[
            "skill_vs_persistence"
        ]
        row[f"selected_{field}_nrmse"] = metrics["nrmse"]
    for mode in (2, 7):
        index = mode - 1
        metrics = augmented.scalar_metrics(
            np.abs(coefficients["truth"][:, 0, index]),
            np.abs(coefficients["prediction"][:, 0, index]),
            np.abs(coefficients["persistence"][:, 0, index]),
        )
        row[f"phi_n{mode}_amplitude_skill"] = metrics["skill_vs_persistence"]
        row[f"phi_n{mode}_amplitude_correlation"] = metrics["correlation"]
    row.update(physics_metrics(prediction, representation, statistics))
    gate = (
        row["state_skill_vs_persistence"],
        row["radial_skill_vs_persistence"],
        row["MTSI_n1_6_transport_skill"],
        row["ECDI_n9_21_transport_skill"],
        row["selected_phi_skill_vs_persistence"],
        row["selected_efy_skill_vs_persistence"],
    )
    row["minimum_gate_skill"] = float(min(gate))
    return row


def train_candidate(
    lambda_e: float,
    checkpoint: dict,
    train_data: direct.DirectWindowDataset,
    validation_data: direct.DirectWindowDataset,
    representation: direct.DirectRepresentation,
    constraint: FieldGradientConstraint,
    truth: np.ndarray,
    persistence: np.ndarray,
    history: np.ndarray,
    low: dict,
    statistics: PhysicsStatistics,
    baseline: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> Candidate:
    seed_everything(args.seed + 20)
    model = neural.ParametricDelayROM(
        direct.STATE_DIMENSION,
        int(checkpoint["hidden_dimension"]),
        float(checkpoint["delta_limit"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 20),
    )
    validation_loader = DataLoader(
        validation_data, batch_size=args.batch_size, shuffle=False
    )
    initial_prediction = direct.rollout(model, history, len(truth), device)
    initial_metrics = evaluate_composite(
        initial_prediction, truth, persistence, representation, low, statistics
    )
    best_model = copy.deepcopy(model.state_dict())
    best_prediction = initial_prediction
    best_metrics = initial_metrics
    best_epoch = 0
    best_excess = initial_metrics["field_gradient_excess_hinge"]
    stale = 0
    rows = []
    thresholds = {
        key: baseline[key] - args.maximum_skill_degradation
        for key in (
            "state_skill_vs_persistence",
            "ECDI_n9_21_transport_skill",
            "selected_phi_skill_vs_persistence",
            "selected_efy_skill_vs_persistence",
        )
    }
    for epoch in range(1, args.epochs + 1):
        model.train()
        data_total = 0.0
        physics_total = 0.0
        count = 0
        for current_history, target, parameter in train_loader:
            current_history = current_history.to(device)
            target = target.to(device)
            parameter = parameter.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model.rollout(
                current_history, parameter, target.shape[1]
            )
            data_loss = direct.group_balanced_loss(
                prediction, target, representation.scaler
            )
            physics_loss = constraint.loss(prediction)
            loss = data_loss + lambda_e * physics_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            data_total += float(data_loss.detach()) * len(current_history)
            physics_total += float(physics_loss.detach()) * len(current_history)
            count += len(current_history)
        window_loss = direct.loader_loss(
            model, validation_loader, representation.scaler, device
        )
        long_prediction = direct.rollout(model, history, len(truth), device)
        metrics = evaluate_composite(
            long_prediction, truth, persistence, representation, low, statistics
        )
        eligible = bool(
            metrics["finite_fraction"] == 1.0
            and all(metrics[key] >= value for key, value in thresholds.items())
            and metrics["minimum_gate_skill"] > 0.0
        )
        excess = metrics["field_gradient_excess_hinge"]
        rows.append({
            "lambda_E": lambda_e,
            "epoch": epoch,
            "train_data_loss": data_total / count,
            "train_physics_loss": physics_total / count,
            "window_validation_loss": window_loss,
            "eligible": eligible,
            "minimum_gate_skill": metrics["minimum_gate_skill"],
            "ECDI_transport_skill": metrics["ECDI_n9_21_transport_skill"],
            "selected_phi_skill": metrics["selected_phi_skill_vs_persistence"],
            "selected_efy_skill": metrics["selected_efy_skill_vs_persistence"],
            "field_gradient_excess_hinge": excess,
        })
        if eligible and excess < best_excess - 1.0e-7:
            best_excess = excess
            best_epoch = epoch
            best_model = copy.deepcopy(model.state_dict())
            best_prediction = long_prediction
            best_metrics = metrics
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"lambda_E={lambda_e:g} epoch={epoch:03d} "
                f"data={data_total / count:.6e} physics={physics_total / count:.6e} "
                f"gate={metrics['minimum_gate_skill']:.6e} excess={excess:.6e}",
                flush=True,
            )
        if stale >= args.patience:
            break
    model.load_state_dict(best_model)
    return Candidate(lambda_e, model, best_epoch, best_metrics, best_prediction, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low", type=Path, default=DEFAULT_LOW)
    parser.add_argument("--high", type=Path, default=DEFAULT_HIGH)
    parser.add_argument("--factor", type=Path, default=DEFAULT_FACTOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lambda-e", type=float, nargs="+", default=LAMBDA_E_CANDIDATES)
    parser.add_argument("--history-steps", type=int, default=40)
    parser.add_argument("--rollout-steps", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--maximum-skill-degradation", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    low_path = args.low.resolve()
    high_path = args.high.resolve()
    factor_path = args.factor.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    seed_everything(args.seed)
    print(f"device={device}", flush=True)

    print("[1/5] Rebuilding the locked direct-state representation", flush=True)
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
    representation = direct.fit_direct_representation(trajectories, fit_masks)
    statistics = physics_statistics(representation, fit_masks)
    constraint = FieldGradientConstraint(representation.scaler, statistics).to(device)
    (output / "truth_field_gradient_audit.json").write_text(
        json.dumps(json_safe(statistics.audit), indent=2), encoding="utf-8"
    )
    print(json.dumps(json_safe(statistics.audit), indent=2), flush=True)

    train_data = direct.DirectWindowDataset(
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
        args.history_steps, args.rollout_steps,
    )
    validation_data = direct.DirectWindowDataset(
        [(
            up, representation.states["e20_to_e22p5"],
            stage2.UP_VALIDATION_START_US, float(up.time_us[-1] + 0.015),
            22.5, 1,
        )],
        args.history_steps, args.rollout_steps,
    )
    validation_start = int(np.flatnonzero(
        up.time_us >= stage2.UP_VALIDATION_START_US - 1.0e-10
    )[0])
    state = representation.states["e20_to_e22p5"]
    history = state[validation_start - args.history_steps:validation_start]
    truth = state[validation_start:]
    persistence = np.repeat(history[-1:], len(truth), axis=0)
    low = load_low_branch(low_path)
    if not np.allclose(low["time_us"], up.time_us[validation_start:]):
        raise ValueError("Low branch is not aligned to the rebuilt high branch")
    high_checkpoint_path = high_path / "direct_physical_state_rom_data_only.pt"
    checkpoint = torch.load(high_checkpoint_path, map_location="cpu")
    baseline = json.loads(
        (factor_path / "development_metrics.json").read_text(encoding="utf-8")
    )

    print("[2/5] Training field-gradient candidates with identical seeds", flush=True)
    candidates = []
    all_rows = []
    for lambda_e in args.lambda_e:
        candidate = train_candidate(
            float(lambda_e), checkpoint, train_data, validation_data,
            representation, constraint, truth, persistence, history, low,
            statistics, baseline, args, device,
        )
        candidates.append(candidate)
        all_rows.extend(candidate.history)
        print(json.dumps(json_safe({"lambda_E": lambda_e, **candidate.metrics}), indent=2), flush=True)

    print("[3/5] Applying the predeclared physics acceptance gate", flush=True)
    # The candidate epoch zero is the same locked high-branch checkpoint for
    # every lambda.  Recompute it explicitly for an unambiguous baseline.
    baseline_model = neural.ParametricDelayROM(
        direct.STATE_DIMENSION,
        int(checkpoint["hidden_dimension"]),
        float(checkpoint["delta_limit"]),
    ).to(device)
    baseline_model.load_state_dict(checkpoint["model_state_dict"])
    baseline_prediction = direct.rollout(
        baseline_model, history, len(truth), device
    )
    baseline_physics = physics_metrics(
        baseline_prediction, representation, statistics
    )
    baseline_excess = baseline_physics["field_gradient_excess_hinge"]
    eligible = []
    metric_rows = []
    for candidate in candidates:
        reduction = 1.0 - candidate.metrics[
            "field_gradient_excess_hinge"
        ] / max(baseline_excess, 1.0e-30)
        accepted = bool(
            candidate.epoch > 0
            and candidate.metrics["minimum_gate_skill"] > 0.0
            and candidate.metrics["selected_phi_nrmse"] < 1.0
            and candidate.metrics["selected_efy_nrmse"] < 1.0
            and reduction > 0.0
        )
        row = {
            "lambda_E": candidate.lambda_e,
            "selected_epoch": candidate.epoch,
            "accepted": accepted,
            "physics_excess_reduction": reduction,
            **candidate.metrics,
        }
        metric_rows.append(row)
        if accepted:
            eligible.append((reduction, candidate))
    selected = max(eligible, key=lambda item: item[0])[1] if eligible else None
    status = "READY_FOR_BLIND_PRIMARY_TEST" if selected is not None else "REJECTED_DEVELOPMENT"

    print("[4/5] Saving candidate checkpoints and locked comparison", flush=True)
    for candidate in candidates:
        torch.save({
            "model_state_dict": candidate.model.state_dict(),
            "lambda_E": candidate.lambda_e,
            "selected_epoch": candidate.epoch,
            "low_branch_checkpoint": str(
                low_path / "regime_aware_transition_rom_data_only.pt"
            ),
            "high_branch_parent": str(high_checkpoint_path),
        }, output / f"high_branch_physics_lambdaE_{candidate.lambda_e:g}.pt")
    write_csv(output / "candidate_metrics.csv", metric_rows)
    write_csv(output / "training_history.csv", all_rows)
    (output / "candidate_metrics.json").write_text(
        json.dumps(json_safe(metric_rows), indent=2), encoding="utf-8"
    )
    with h5py.File(output / "development_rollouts.h5", "w") as handle:
        handle.attrs["primary_E25_to_E22p5_loaded"] = False
        handle.create_dataset("time_us", data=up.time_us[validation_start:])
        handle.create_dataset("frame", data=up.frame[validation_start:])
        handle.create_dataset("truth_high_state", data=truth, compression="gzip")
        handle.create_dataset(
            "persistence_high_state", data=persistence, compression="gzip"
        )
        handle.create_dataset(
            "baseline_high_state", data=baseline_prediction, compression="gzip"
        )
        for candidate in candidates:
            handle.create_dataset(
                f"lambda_E_{candidate.lambda_e:g}/high_state",
                data=candidate.prediction,
                compression="gzip",
            )
    selected_row = (
        next(row for row in metric_rows if row["lambda_E"] == selected.lambda_e)
        if selected is not None else None
    )
    lock = {
        "status": status,
        "selected_lambda_E": selected.lambda_e if selected is not None else None,
        "selected_metrics": selected_row,
        "baseline_field_gradient_excess_hinge": baseline_excess,
        "truth_audit": statistics.audit,
        "acceptance_rule": (
            "positive composite gate, selected phi/Ey NRMSE < 1, positive "
            "truth-floor field-gradient excess reduction, and <=0.03 skill "
            "degradation from the locked data-only factorized ROM"
        ),
        "primary_test": {
            "direction": "E25_to_E22.5",
            "data_loaded": False,
            "used_for_selection": False,
        },
        "low_branch_checkpoint": str(
            low_path / "regime_aware_transition_rom_data_only.pt"
        ),
        "high_branch_parent": str(high_checkpoint_path),
        "factorized_manifest": str(factor_path / "model_lock.json"),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "arguments": vars(args),
    }
    (output / "model_lock.json").write_text(
        json.dumps(json_safe(lock), indent=2), encoding="utf-8"
    )
    readme = f"""# Mode-factorized field-gradient physics ROM

- Status: `{status}`
- Selected lambda_E: `{lock['selected_lambda_E']}`
- Truth residual / Ey RMS: {statistics.audit['truth_residual_over_ey_rms']:.6g}
- Primary E25 -> E22.5 data loaded: **no**

The low/MTSI branch is frozen.  Every high-branch candidate starts from the
same locked direct-state checkpoint and uses the same minibatch order.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print("[5/5] Complete", flush=True)
    print(f"status={status}", flush=True)
    print(f"selected_lambda_E={lock['selected_lambda_E']}", flush=True)
    print(f"output={output}", flush=True)


if __name__ == "__main__":
    main()
