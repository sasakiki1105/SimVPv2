"""Train one carrier ROM with an explicit training-only slow modulation clock.

The model uses the 150D carrier-envelope state and known electric-field
controls.  Four additional causal controls encode the fundamental and second
harmonic of a slow period fitted from n=2/n=7 amplitudes over the allowed
E20 -> E22.5 training interval (before 35 us).  No future PIC value and no
E25 -> E22.5 primary data are read.

The field branch is evaluated together with the already locked low/MTSI and
direct/ECDI branches.  Model selection uses a common raw-Fourier persistence
baseline, avoiding the easier frozen-envelope baseline.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
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
import build_radaz_mode_specific_expert_rom as expert
import train_radaz_carrier_envelope_high_branch_rom as carrier
import train_radaz_carrier_envelope_time_controlled_rom as controlled
import train_radaz_direct_physical_state_rom as direct
import train_radaz_regime_aware_transition_rom as stage2


ROOT = Path(__file__).resolve().parent
DEFAULT_BASE = ROOT / "workdirs" / "build_radaz_mode_factorized_rom"
DEFAULT_OUTPUT = (
    ROOT / "workdirs" / "train_radaz_modulation_controlled_carrier_rom"
)
CONTROL_DIMENSION = 7


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


def fit_modulation_period(
    trajectory: stage2.Trajectory,
    representation: carrier.CarrierRepresentation,
    fit_mask: np.ndarray,
    period_min_us: float,
    period_max_us: float,
    period_step_us: float,
) -> dict:
    """Fit one n=2/n=7 amplitude period using training samples only."""
    time_us = trajectory.time_us[fit_mask]
    packed = representation.groups[trajectory.name]["carrier_physical"][fit_mask]
    shaped = packed.reshape(
        len(packed), len(direct.FIELD_NAMES), len(carrier.SELECTED_MODE_NUMBERS), 2
    )
    amplitude = np.sqrt(np.sum(np.square(shaped[:, 0]), axis=-1))
    selected_indices = [
        int(np.flatnonzero(carrier.SELECTED_MODE_NUMBERS == mode)[0])
        for mode in (2, 7)
    ]
    periods = np.arange(
        period_min_us,
        period_max_us + 0.5 * period_step_us,
        period_step_us,
        dtype=np.float64,
    )
    scores = np.empty(len(periods), dtype=np.float64)
    explained = np.empty((len(periods), len(selected_indices)), dtype=np.float64)
    centered_time = time_us - np.mean(time_us)
    for period_index, period_us in enumerate(periods):
        phase = 2.0 * np.pi * (time_us - controlled.STEP_TIME_US) / period_us
        design = np.column_stack(
            (np.ones(len(time_us)), centered_time, np.sin(phase), np.cos(phase))
        )
        residual_ratios = []
        for column, mode_index in enumerate(selected_indices):
            truth = amplitude[:, mode_index]
            prediction = design @ np.linalg.lstsq(design, truth, rcond=None)[0]
            residual_ratio = np.mean((truth - prediction) ** 2) / max(
                np.var(truth), np.finfo(float).tiny
            )
            residual_ratios.append(residual_ratio)
            explained[period_index, column] = 1.0 - residual_ratio
        scores[period_index] = np.mean(residual_ratios)
    best_index = int(np.argmin(scores))
    return {
        "period_us": float(periods[best_index]),
        "joint_normalized_residual": float(scores[best_index]),
        "n2_variance_explained": float(explained[best_index, 0]),
        "n7_variance_explained": float(explained[best_index, 1]),
        "fit_interval_us": [float(time_us[0]), float(time_us[-1])],
        "fit_samples": int(len(time_us)),
        "search_interval_us": [period_min_us, period_max_us],
        "search_step_us": period_step_us,
        "primary_data_loaded": False,
    }


def modulation_controls_for(
    trajectory: stage2.Trajectory,
    current_ez_kvm: float,
    source_ez_kvm: float,
    transition: bool,
    period_us: float,
) -> np.ndarray:
    base = controlled.controls_for(
        trajectory, current_ez_kvm, source_ez_kvm, transition
    )
    if transition:
        phase = (
            2.0
            * np.pi
            * (trajectory.time_us - controlled.STEP_TIME_US)
            / period_us
        )
        harmonic = np.column_stack(
            (np.sin(phase), np.cos(phase), np.sin(2.0 * phase), np.cos(2.0 * phase))
        )
    else:
        harmonic = np.zeros((len(trajectory.time_us), 4), dtype=np.float64)
    return np.column_stack((base, harmonic))


def load_base(path: Path) -> dict:
    return expert.load_base(path)


def decode_coefficients(
    states: np.ndarray,
    representation: carrier.CarrierRepresentation,
    indices: np.ndarray,
) -> np.ndarray:
    decoded = representation.scaler.inverse(states)
    return carrier.remodulate(
        decoded["carrier_physical"],
        representation.carrier_step_rad["e20_to_e22p5"],
        indices,
    )


def evaluate_full_model(
    truth_state: np.ndarray,
    prediction_state: np.ndarray,
    carrier_persistence: np.ndarray,
    representation: carrier.CarrierRepresentation,
    base: dict,
    full_indices: np.ndarray,
) -> dict:
    truth_coeff = base["truth"]["selected_physical_coefficients"]
    raw_persistence = base["persistence"]["selected_physical_coefficients"]
    prediction_coeff = decode_coefficients(
        prediction_state, representation, full_indices
    )
    state_metrics = augmented.scalar_metrics(
        truth_state, prediction_state, carrier_persistence
    )
    base_state = augmented.scalar_metrics(
        base["truth"]["composite_state"],
        base["prediction"]["composite_state"],
        base["persistence"]["composite_state"],
    )
    radial = augmented.scalar_metrics(
        base["truth"]["radial"],
        base["prediction"]["radial"],
        base["persistence"]["radial"],
    )
    transport = augmented.scalar_metrics(
        base["truth"]["transport"],
        base["prediction"]["transport"],
        base["persistence"]["transport"],
    )
    row = {
        "finite_fraction": float(np.mean(np.isfinite(prediction_state))),
        "carrier_state_skill_vs_envelope_persistence": state_metrics[
            "skill_vs_persistence"
        ],
        "composite_state_skill_vs_persistence": base_state[
            "skill_vs_persistence"
        ],
        "radial_skill_vs_persistence": radial["skill_vs_persistence"],
        "transport_skill_vs_persistence": transport["skill_vs_persistence"],
    }
    for band_index, band in enumerate(("MTSI_n1_6", "ECDI_n9_21")):
        metric = augmented.scalar_metrics(
            base["truth"]["transport"][:, band_index],
            base["prediction"]["transport"][:, band_index],
            base["persistence"]["transport"][:, band_index],
        )
        row[f"{band}_transport_skill"] = metric["skill_vs_persistence"]
        row[f"{band}_transport_correlation"] = metric["correlation"]
    for field_index, field in ((0, "phi"), (3, "efy")):
        metric = augmented.scalar_metrics(
            truth_coeff[:, field_index],
            prediction_coeff[:, field_index],
            raw_persistence[:, field_index],
        )
        row[f"selected_{field}_skill_vs_raw_persistence"] = metric[
            "skill_vs_persistence"
        ]
        row[f"selected_{field}_nrmse"] = metric["nrmse"]
        row[f"selected_{field}_temporal_anomaly_correlation"] = metric[
            "temporal_anomaly_correlation"
        ]
    phi = {
        "truth": truth_coeff[:, 0],
        "prediction": prediction_coeff[:, 0],
        "persistence": raw_persistence[:, 0],
    }
    dt_us = float(np.median(np.diff(base["time_us"])))
    for mode in (2, 7):
        mode_index = int(np.flatnonzero(base["mode_numbers"] == mode)[0])
        metric = augmented.scalar_metrics(
            np.abs(phi["truth"][:, mode_index]),
            np.abs(phi["prediction"][:, mode_index]),
            np.abs(phi["persistence"][:, mode_index]),
        )
        row[f"phi_n{mode}_amplitude_skill"] = metric["skill_vs_persistence"]
        row[f"phi_n{mode}_amplitude_correlation"] = metric["correlation"]
        truth_frequency = expert.phase_frequency_mhz(
            phi["truth"][:, mode_index], dt_us
        )
        prediction_frequency = expert.phase_frequency_mhz(
            phi["prediction"][:, mode_index], dt_us
        )
        row[f"phi_n{mode}_truth_frequency_MHz"] = truth_frequency
        row[f"phi_n{mode}_prediction_frequency_MHz"] = prediction_frequency
        row[f"phi_n{mode}_frequency_abs_error_MHz"] = abs(
            prediction_frequency - truth_frequency
        )
    k = 2.0 * np.pi * base["mode_numbers"] / direct.AZIMUTHAL_LENGTH_M
    field_residual = prediction_coeff[:, 3] + 1j * k[None] * prediction_coeff[:, 0]
    row["field_gradient_residual_over_ey_rms"] = float(
        np.sqrt(np.mean(np.abs(field_residual) ** 2))
        / np.sqrt(np.mean(np.abs(prediction_coeff[:, 3]) ** 2))
    )
    gates = (
        row["carrier_state_skill_vs_envelope_persistence"],
        row["composite_state_skill_vs_persistence"],
        row["radial_skill_vs_persistence"],
        row["MTSI_n1_6_transport_skill"],
        row["ECDI_n9_21_transport_skill"],
        row["selected_phi_skill_vs_raw_persistence"],
        row["selected_efy_skill_vs_raw_persistence"],
        row["phi_n2_amplitude_skill"],
        row["phi_n7_amplitude_skill"],
    )
    row["minimum_persistence_gate_skill"] = float(min(gates))
    row["passes_all_persistence_gates"] = bool(
        row["finite_fraction"] == 1.0
        and row["minimum_persistence_gate_skill"] > 0.0
    )
    row["passes_field_climatology_gate"] = bool(
        row["selected_phi_nrmse"] < 1.0 and row["selected_efy_nrmse"] < 1.0
    )
    # First maximize the worst observable.  Once all skills pass, prefer the
    # candidate closest to the physical-field climatology gate.
    row["selection_score"] = float(
        row["minimum_persistence_gate_skill"]
        + 0.05
        * min(
            1.0 - row["selected_phi_nrmse"],
            1.0 - row["selected_efy_nrmse"],
        )
    )
    return row


def train_transition(
    model: controlled.ControlledDelayROM,
    dataset: controlled.ControlledWindowDataset,
    representation: carrier.CarrierRepresentation,
    complex_loss: carrier.ComplexLoss,
    truth: np.ndarray,
    persistence: np.ndarray,
    state_history: np.ndarray,
    control_history: np.ndarray,
    future_controls: np.ndarray,
    base: dict,
    validation_start: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[controlled.ControlledDelayROM, int, dict, np.ndarray, list[dict]]:
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

    def forecast_and_evaluate():
        prediction = controlled.rollout(
            model, state_history, control_history, future_controls, device
        )
        metrics = evaluate_full_model(
            truth,
            prediction,
            persistence,
            representation,
            base,
            full_indices,
        )
        return prediction, metrics

    best_prediction, best_metrics = forecast_and_evaluate()
    best_score = best_metrics["selection_score"]
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale = 0
    rows = []
    for epoch in range(1, args.finetune_epochs + 1):
        model.train()
        totals = {name: 0.0 for name in ("loss", "data", "complex")}
        count = 0
        for state, controls, target, target_controls in loader:
            state = state.to(device)
            controls = controls.to(device)
            target = target.to(device)
            target_controls = target_controls.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, terms = controlled.total_loss(
                model,
                state,
                controls,
                target,
                target_controls,
                representation,
                complex_loss,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += float(loss.detach()) * len(state)
            totals["data"] += float(terms["data"].detach()) * len(state)
            totals["complex"] += float(terms["complex"].detach()) * len(state)
            count += len(state)
        prediction, metrics = forecast_and_evaluate()
        row = {
            "stage": "finetune",
            "epoch": epoch,
            **{name: value / count for name, value in totals.items()},
            **metrics,
        }
        rows.append(row)
        score = metrics["selection_score"]
        if score > best_score + 1.0e-5:
            best_score = score
            best_metrics = metrics
            best_prediction = prediction
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"finetune epoch={epoch:03d} loss={totals['loss'] / count:.6e} "
                f"score={score:+.4f} phi={metrics['selected_phi_skill_vs_raw_persistence']:+.4f} "
                f"Ey={metrics['selected_efy_skill_vs_raw_persistence']:+.4f} "
                f"n2={metrics['phi_n2_amplitude_skill']:+.4f} "
                f"n7={metrics['phi_n7_amplitude_skill']:+.4f}",
                flush=True,
            )
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    return model, best_epoch, best_metrics, best_prediction, rows


def plot_rollout(
    path: Path,
    base: dict,
    prediction_coeff: np.ndarray,
) -> None:
    truth = base["truth"]["selected_physical_coefficients"]
    persistence = base["persistence"]["selected_physical_coefficients"]
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 6.5), sharex=True)
    for axis, mode in zip(axes, (2, 7)):
        index = int(np.flatnonzero(base["mode_numbers"] == mode)[0])
        for values, label, color, style in (
            (truth, "PIC truth", "black", "-"),
            (prediction_coeff, "single modulation ROM", "tab:blue", "-"),
            (persistence, "raw persistence", "0.65", "--"),
        ):
            axis.plot(
                base["time_us"], np.abs(values[:, 0, index]),
                label=label, color=color, linestyle=style,
            )
        axis.set_ylabel(f"|phi n={mode}|")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[-1].set_xlabel("time [us]")
    figure.suptitle("Single modulation-controlled carrier ROM")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
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
    parser.add_argument("--lambda-amplitude", type=float, default=0.35)
    parser.add_argument("--lambda-phase", type=float, default=0.05)
    parser.add_argument("--lambda-frequency", type=float, default=0.10)
    parser.add_argument("--period-min-us", type=float, default=0.8)
    parser.add_argument("--period-max-us", type=float, default=2.5)
    parser.add_argument("--period-step-us", type=float, default=0.001)
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
    print("[1/6] Fitting carrier representation and training-only slow period", flush=True)
    representation = carrier.fit_representation(trajectories, fit_masks)
    carrier.save_representation(output / "representation.h5", representation)
    period_audit = fit_modulation_period(
        up,
        representation,
        fit_masks["e20_to_e22p5"],
        args.period_min_us,
        args.period_max_us,
        args.period_step_us,
    )
    period_us = period_audit["period_us"]
    (output / "modulation_period_audit.json").write_text(
        json.dumps(json_safe(period_audit), indent=2), encoding="utf-8"
    )
    print(json.dumps(json_safe(period_audit), indent=2), flush=True)

    controls = {
        "e25_stationary": modulation_controls_for(e25, 25.0, 25.0, False, period_us),
        "e20_to_e22p5": modulation_controls_for(up, 22.5, 20.0, True, period_us),
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
    complex_loss = carrier.ComplexLoss(
        representation.scaler,
        representation.amplitude_scale,
        args.lambda_amplitude,
        args.lambda_phase,
        args.lambda_frequency,
    ).to(device)
    model = controlled.ControlledDelayROM(
        carrier.STATE_DIMENSION, CONTROL_DIMENSION, args.hidden_dim, args.delta_limit
    ).to(device)
    print(
        f"state_dim={carrier.STATE_DIMENSION} control_dim={CONTROL_DIMENSION} "
        f"period={period_us:.3f} us windows: pretrain={len(pretrain_data)} train={len(train_data)}",
        flush=True,
    )
    print("[2/6] Pretraining stationary E25", flush=True)
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
    base = load_base(args.base.resolve())
    if not np.array_equal(base["frame"], up.frame[validation_start:]):
        raise ValueError("Base factorized rollout alignment mismatch")
    print("[3/6] Fine-tuning and selecting the single field branch", flush=True)
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

    print("[4/6] Saving checkpoint and common-baseline rollout", flush=True)
    checkpoint = output / "modulation_controlled_carrier_data_only.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_dimension": carrier.STATE_DIMENSION,
            "control_dimension": CONTROL_DIMENSION,
            "hidden_dimension": args.hidden_dim,
            "delta_limit": args.delta_limit,
            "history_steps": args.history_steps,
            "rollout_steps": args.rollout_steps,
            "modulation_period_us": period_us,
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
    prediction_coeff = decode_coefficients(prediction, representation, full_indices)
    with h5py.File(output / "development_rollout_35to40us.h5", "w") as handle:
        handle.attrs["primary_E25_to_E22p5_loaded"] = False
        handle.attrs["modulation_period_us"] = period_us
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
    plot_rollout(output / "development_rollout_35to40us.png", base, prediction_coeff)

    print("[5/6] Writing prospective lock", flush=True)
    base_lock = args.base.resolve() / "model_lock.json"
    low_checkpoint = args.low.resolve() / "regime_aware_transition_rom_data_only.pt"
    direct_checkpoint = ROOT / "workdirs" / "train_radaz_direct_physical_state_rom" / "direct_physical_state_rom_data_only.pt"
    lock = {
        "status": status,
        "accepted_for_physics_ablation": accepted,
        "development_metrics": metrics,
        "modulation_period_audit": period_audit,
        "known_controls": {
            "current_Ez": True,
            "source_Ez_before_step": True,
            "elapsed_time_since_step": True,
            "slow_fundamental_sin_cos": True,
            "slow_second_harmonic_sin_cos": True,
            "future_PIC_state": False,
        },
        "component_roles": {
            "single_modulation_carrier": "all selected field modes n2,n7--n21",
            "locked_low_branch": "radial and MTSI transport",
            "locked_direct_branch": "ECDI transport",
        },
        "primary_test": {
            "direction": "E25_to_E22.5",
            "data_loaded": False,
            "used_for_period_fit": False,
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
    readme = f"""# Modulation-controlled single carrier ROM

- Status: `{status}`
- Training-only slow period: {period_us:.3f} us
- Best epoch: {best_epoch}
- phi skill vs raw persistence: {metrics['selected_phi_skill_vs_raw_persistence']:.6f}
- Ey skill vs raw persistence: {metrics['selected_efy_skill_vs_raw_persistence']:.6f}
- n=2 amplitude skill: {metrics['phi_n2_amplitude_skill']:.6f}
- n=7 amplitude skill: {metrics['phi_n7_amplitude_skill']:.6f}
- phi/Ey NRMSE: {metrics['selected_phi_nrmse']:.6f}/{metrics['selected_efy_nrmse']:.6f}
- Primary E25 -> E22.5 data loaded: **no**
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print("[6/6] Complete", flush=True)
    print(f"status={status}", flush=True)
    print(f"output={output}", flush=True)


if __name__ == "__main__":
    main()
