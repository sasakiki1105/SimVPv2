"""Identify a coupled slow-amplitude ODE for allowed RadAz Ez trajectories.

The primary E25 -> E22.5 trajectory is deliberately absent.  Global phi
Fourier amplitudes n=2 and n=7 are reduced to causal slow envelopes, and a
bounded polynomial controlled ODE is fit with trajectory-balanced ridge
regression.  Current-E-only, uncoupled, electric-history-controlled, and
leave-one-transition-out experiments use the same numerical protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "workdirs" / "train_radaz_coupled_amplitude_ode"
DEFAULT_E20 = ROOT / "workdirs" / "radaz_e20_stationary" / "physical_fourier_targets.h5"
DEFAULT_E25 = (
    ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_ez25kvm_fourier_latent_to_physical_modes"
    / "physical_fourier_targets.h5"
)
DEFAULT_UP = (
    ROOT / "workdirs" / "radaz_e20_to_e22p5_transition" / "physical_fourier_targets.h5"
)
DEFAULT_DOWN = (
    ROOT / "workdirs" / "radaz_e22p5_to_e20_transition" / "physical_fourier_targets.h5"
)
MODES = (2, 7)
CONTROL_NAMES = (
    "current_ez",
    "source_ez",
    "delta_ez",
    "transition_flag",
    "age_tanh",
    "memory_0p3us",
    "memory_1p5us",
    "memory_5us",
)
MEMORY_TIMES_US = (0.30, 1.50, 5.00)
RIDGES = (1.0e-6, 1.0e-4, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1000.0)
TIME_TOLERANCE_US = 1.0e-9


@dataclass
class Trajectory:
    name: str
    path: Path
    current_ez_kvm: float
    source_ez_kvm: float
    transition: bool
    time_us: np.ndarray
    frame: np.ndarray
    raw_amplitude: np.ndarray
    slow_amplitude: np.ndarray
    controls: np.ndarray


@dataclass
class Transform:
    mean: np.ndarray
    scale: np.ndarray

    def encode(self, amplitude: np.ndarray) -> np.ndarray:
        values = np.log(np.maximum(amplitude, np.finfo(float).tiny))
        return (values - self.mean) / self.scale

    def decode(self, state: np.ndarray) -> np.ndarray:
        exponent = state * self.scale + self.mean
        return np.exp(np.clip(exponent, -40.0, 40.0))


@dataclass
class ControlledODE:
    coupled: bool
    history_controlled: bool
    ridge: float
    feature_names: list[list[str]]
    weights: list[np.ndarray]
    transform: Transform
    state_clip: float = 8.0

    def derivative(self, state: np.ndarray, controls: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        controls = np.asarray(controls, dtype=np.float64)
        return np.asarray(
            [
                feature_vector(
                    state,
                    controls,
                    output_index,
                    self.coupled,
                    self.history_controlled,
                )[0]
                @ self.weights[output_index]
                for output_index in range(len(MODES))
            ],
            dtype=np.float64,
        )

    def rollout(
        self,
        initial_state: np.ndarray,
        controls: np.ndarray,
        dt_us: float,
    ) -> tuple[np.ndarray, int]:
        prediction = np.empty((len(controls), len(MODES)), dtype=np.float64)
        current = np.asarray(initial_state, dtype=np.float64).copy()
        clipped = 0
        for index, control in enumerate(controls):
            # Explicit midpoint integration is stable at dt=0.015 us for the
            # deliberately smoothed amplitude state.
            first = self.derivative(current, control)
            midpoint = current + 0.5 * dt_us * first
            following = current + dt_us * self.derivative(midpoint, control)
            bounded = np.clip(following, -self.state_clip, self.state_clip)
            clipped += int(np.count_nonzero(bounded != following))
            current = bounded
            prediction[index] = current
        return prediction, clipped


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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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


def causal_ewma(values: np.ndarray, dt_us: float, tau_us: float) -> np.ndarray:
    alpha = 1.0 - math.exp(-dt_us / tau_us)
    result = np.empty_like(values, dtype=np.float64)
    result[0] = values[0]
    for index in range(1, len(values)):
        result[index] = result[index - 1] + alpha * (
            values[index] - result[index - 1]
        )
    return result


def electric_controls(
    time_us: np.ndarray,
    current_ez_kvm: float,
    source_ez_kvm: float,
    transition: bool,
    step_time_us: float,
) -> np.ndarray:
    center = 22.5
    scale = 2.5
    current = np.full(len(time_us), (current_ez_kvm - center) / scale)
    source = np.full(len(time_us), (source_ez_kvm - center) / scale)
    delta = current - source
    flag = np.full(len(time_us), float(transition))
    age = (
        np.maximum(time_us - step_time_us, 0.0)
        if transition
        else np.zeros(len(time_us), dtype=np.float64)
    )
    memories = [delta * np.exp(-age / tau) for tau in MEMORY_TIMES_US]
    controls = np.column_stack(
        (current, source, delta, flag, np.tanh(age / 2.0), *memories)
    )
    if controls.shape[1] != len(CONTROL_NAMES):
        raise AssertionError("Control dimension mismatch")
    return controls


def load_trajectory(
    name: str,
    path: Path,
    current_ez_kvm: float,
    source_ez_kvm: float,
    transition: bool,
    smoothing_tau_us: float,
) -> Trajectory:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        coefficients = np.asarray(handle["coefficients"], dtype=np.complex128)
        radial_weights = np.asarray(handle["radial_weights"], dtype=np.float64)
        fields_raw = np.asarray(handle["fields"])
        modes = np.asarray(handle["modes"], dtype=np.int64)
        time_us = np.asarray(handle["time_us"], dtype=np.float64)
        frame = np.asarray(handle["frame"], dtype=np.int64)
    fields = [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in fields_raw
    ]
    mode_indices = [int(np.flatnonzero(modes == mode)[0]) for mode in MODES]
    phi = coefficients[:, fields.index("phi"), :, :][:, :, mode_indices]
    global_phi = np.einsum("r,trm->tm", radial_weights, phi)
    raw_amplitude = np.abs(global_phi)
    if not np.all(np.isfinite(raw_amplitude)):
        raise ValueError(f"Invalid amplitude in {path}")
    dt_us = float(np.median(np.diff(time_us)))
    if not np.allclose(np.diff(time_us), dt_us, atol=TIME_TOLERANCE_US, rtol=0.0):
        raise ValueError(f"Non-uniform time sampling in {path}")
    slow_amplitude = causal_ewma(raw_amplitude, dt_us, smoothing_tau_us)
    controls = electric_controls(
        time_us,
        current_ez_kvm,
        source_ez_kvm,
        transition,
        step_time_us=30.0,
    )
    return Trajectory(
        name,
        path,
        current_ez_kvm,
        source_ez_kvm,
        transition,
        time_us,
        frame,
        raw_amplitude,
        slow_amplitude,
        controls,
    )


def interval_indices(
    trajectory: Trajectory, start_us: float, end_us: float
) -> np.ndarray:
    return np.flatnonzero(
        (trajectory.time_us >= start_us - TIME_TOLERANCE_US)
        & (trajectory.time_us < end_us - TIME_TOLERANCE_US)
    )


def fit_transform(
    trajectories: dict[str, Trajectory],
    intervals: list[tuple[str, float, float]],
) -> Transform:
    logs = []
    for name, start_us, end_us in intervals:
        indices = interval_indices(trajectories[name], start_us, end_us)
        logs.append(np.log(trajectories[name].slow_amplitude[indices]))
    stacked = np.concatenate(logs, axis=0)
    scale = np.std(stacked, axis=0, ddof=1)
    scale = np.maximum(scale, 1.0e-6)
    return Transform(np.mean(stacked, axis=0), scale)


def selected_control_indices(history_controlled: bool) -> np.ndarray:
    if history_controlled:
        return np.arange(len(CONTROL_NAMES), dtype=np.int64)
    return np.asarray([CONTROL_NAMES.index("current_ez")], dtype=np.int64)


def feature_vector(
    state: np.ndarray,
    controls: np.ndarray,
    output_index: int,
    coupled: bool,
    history_controlled: bool,
) -> tuple[np.ndarray, list[str]]:
    z = np.tanh(np.asarray(state, dtype=np.float64))
    control_indices = selected_control_indices(history_controlled)
    u = np.asarray(controls, dtype=np.float64)[control_indices]
    u_names = [CONTROL_NAMES[index] for index in control_indices]
    state_indices = list(range(len(MODES))) if coupled else [output_index]
    values = [1.0]
    names = ["constant"]
    for index in state_indices:
        values.append(z[index])
        names.append(f"tanh_z_n{MODES[index]}")
    for left_position, left in enumerate(state_indices):
        for right in state_indices[left_position:]:
            values.append(z[left] * z[right])
            names.append(f"tanh_z_n{MODES[left]}*tanh_z_n{MODES[right]}")
    values.extend(u.tolist())
    names.extend(u_names)
    for index in state_indices:
        for control_value, control_name in zip(u, u_names):
            values.append(z[index] * control_value)
            names.append(f"tanh_z_n{MODES[index]}*{control_name}")
    return np.asarray(values, dtype=np.float64), names


def derivative_rows(
    trajectories: dict[str, Trajectory],
    intervals: list[tuple[str, float, float]],
    transform: Transform,
    output_index: int,
    coupled: bool,
    history_controlled: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    matrices = []
    targets = []
    weights = []
    feature_names = None
    for name, start_us, end_us in intervals:
        trajectory = trajectories[name]
        indices = interval_indices(trajectory, start_us, end_us)
        if len(indices) < 3 or not np.all(np.diff(indices) == 1):
            raise ValueError(f"Insufficient contiguous interval {name} {start_us}:{end_us}")
        current_indices = indices[:-1]
        following_indices = indices[1:]
        state = transform.encode(trajectory.slow_amplitude)
        dt_us = np.diff(trajectory.time_us[indices])
        target = (state[following_indices, output_index] - state[current_indices, output_index]) / dt_us
        rows = []
        for index in current_indices:
            values, names_now = feature_vector(
                state[index],
                trajectory.controls[index],
                output_index,
                coupled,
                history_controlled,
            )
            rows.append(values)
            if feature_names is None:
                feature_names = names_now
            elif feature_names != names_now:
                raise AssertionError("Feature library changed within a fit")
        matrices.append(np.asarray(rows))
        targets.append(target)
        # Equal total influence per trajectory interval.
        weights.append(np.full(len(rows), 1.0 / len(rows), dtype=np.float64))
    return (
        np.concatenate(matrices),
        np.concatenate(targets),
        np.concatenate(weights),
        list(feature_names),
    )


def fit_ode(
    trajectories: dict[str, Trajectory],
    intervals: list[tuple[str, float, float]],
    coupled: bool,
    history_controlled: bool,
    ridge: float,
) -> ControlledODE:
    transform = fit_transform(trajectories, intervals)
    feature_names = []
    weights = []
    for output_index in range(len(MODES)):
        matrix, target, sample_weight, names = derivative_rows(
            trajectories,
            intervals,
            transform,
            output_index,
            coupled,
            history_controlled,
        )
        root_weight = np.sqrt(sample_weight)
        weighted_matrix = matrix * root_weight[:, None]
        weighted_target = target * root_weight
        penalty = np.eye(matrix.shape[1], dtype=np.float64) * ridge
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            weighted_matrix.T @ weighted_matrix + penalty,
            weighted_matrix.T @ weighted_target,
        )
        feature_names.append(names)
        weights.append(coefficients)
    return ControlledODE(
        coupled,
        history_controlled,
        ridge,
        feature_names,
        weights,
        transform,
    )


def rollout_interval(
    model: ControlledODE,
    trajectory: Trajectory,
    start_us: float,
    end_us: float,
) -> dict:
    indices = interval_indices(trajectory, start_us, end_us)
    if len(indices) < 3 or not np.all(np.diff(indices) == 1):
        raise ValueError(f"Invalid rollout interval {trajectory.name} {start_us}:{end_us}")
    first = int(indices[0])
    if first > 0:
        initial_index = first - 1
        target_indices = indices
    else:
        initial_index = first
        target_indices = indices[1:]
    dt_us = float(np.median(np.diff(trajectory.time_us)))
    initial_state = model.transform.encode(
        trajectory.slow_amplitude[initial_index : initial_index + 1]
    )[0]
    prediction_state, clipped = model.rollout(
        initial_state,
        trajectory.controls[target_indices],
        dt_us,
    )
    truth_amplitude = trajectory.slow_amplitude[target_indices]
    prediction_amplitude = model.transform.decode(prediction_state)
    persistence_amplitude = np.repeat(
        trajectory.slow_amplitude[initial_index : initial_index + 1],
        len(target_indices),
        axis=0,
    )
    truth_state = model.transform.encode(truth_amplitude)
    return {
        "time_us": trajectory.time_us[target_indices],
        "truth_amplitude": truth_amplitude,
        "prediction_amplitude": prediction_amplitude,
        "persistence_amplitude": persistence_amplitude,
        "truth_state": truth_state,
        "prediction_state": prediction_state,
        "clipped_state_values": clipped,
    }


def scalar_metrics(truth: np.ndarray, prediction: np.ndarray, persistence: np.ndarray) -> dict:
    error = prediction - truth
    persistence_error = persistence - truth
    mse = float(np.mean(error**2))
    persistence_mse = float(np.mean(persistence_error**2))
    standard = float(np.std(truth, ddof=1))
    centered_truth = truth - np.mean(truth)
    centered_prediction = prediction - np.mean(prediction)
    denominator = float(np.linalg.norm(centered_truth) * np.linalg.norm(centered_prediction))
    return {
        "rmse": math.sqrt(mse),
        "nrmse": math.sqrt(mse) / max(standard, np.finfo(float).tiny),
        "skill_vs_persistence": 1.0 - mse / max(persistence_mse, np.finfo(float).tiny),
        "correlation": (
            float(np.dot(centered_truth, centered_prediction) / denominator)
            if denominator > np.finfo(float).tiny
            else float("nan")
        ),
        "endpoint_relative_error": float(
            abs(prediction[-1] - truth[-1])
            / max(abs(truth[-1]), np.finfo(float).tiny)
        ),
    }


def evaluate_rollout(rollout: dict) -> dict:
    result = {
        "frames": int(len(rollout["time_us"])),
        "first_time_us": float(rollout["time_us"][0]),
        "last_time_us": float(rollout["time_us"][-1]),
        "clipped_state_values": int(rollout["clipped_state_values"]),
        "normalized_log_state_mse": float(
            np.mean((rollout["prediction_state"] - rollout["truth_state"]) ** 2)
        ),
    }
    mode_skills = []
    for mode_index, mode in enumerate(MODES):
        metrics = scalar_metrics(
            rollout["truth_amplitude"][:, mode_index],
            rollout["prediction_amplitude"][:, mode_index],
            rollout["persistence_amplitude"][:, mode_index],
        )
        for key, value in metrics.items():
            result[f"n{mode}_{key}"] = value
        mode_skills.append(metrics["skill_vs_persistence"])
    result["minimum_mode_skill_vs_persistence"] = float(min(mode_skills))
    result["mean_mode_skill_vs_persistence"] = float(np.mean(mode_skills))
    return result


def validation_score(
    model: ControlledODE,
    trajectories: dict[str, Trajectory],
    intervals: list[tuple[str, float, float]],
) -> float:
    errors = []
    for name, start_us, end_us in intervals:
        rollout = rollout_interval(model, trajectories[name], start_us, end_us)
        errors.append(
            np.mean((rollout["prediction_state"] - rollout["truth_state"]) ** 2)
        )
    return float(np.mean(errors))


def fit_experiment(
    name: str,
    trajectories: dict[str, Trajectory],
    train_intervals: list[tuple[str, float, float]],
    validation_intervals: list[tuple[str, float, float]],
    test_intervals: list[tuple[str, str, float, float]],
    coupled: bool,
    history_controlled: bool,
) -> tuple[ControlledODE, dict[str, dict], dict[str, dict], list[dict]]:
    selection_rows = []
    best_model = None
    best_score = float("inf")
    for ridge in RIDGES:
        model = fit_ode(
            trajectories,
            train_intervals,
            coupled,
            history_controlled,
            ridge,
        )
        score = validation_score(model, trajectories, validation_intervals)
        row = {"experiment": name, "ridge": ridge, "validation_log_state_mse": score}
        selection_rows.append(row)
        if score < best_score:
            best_score = score
            best_model = model
    if best_model is None:
        raise AssertionError("No ODE candidate was fit")
    rollouts = {}
    metrics = {}
    for label, trajectory_name, start_us, end_us in test_intervals:
        rollout = rollout_interval(
            best_model,
            trajectories[trajectory_name],
            start_us,
            end_us,
        )
        rollouts[label] = rollout
        metrics[label] = evaluate_rollout(rollout)
    summary = {
        "coupled": coupled,
        "history_controlled": history_controlled,
        "selected_ridge": best_model.ridge,
        "validation_log_state_mse": best_score,
        "train_intervals": train_intervals,
        "validation_intervals": validation_intervals,
        "test_metrics": metrics,
    }
    print(
        f"[{name}] ridge={best_model.ridge:g} validation={best_score:.4e}",
        flush=True,
    )
    for label, row in metrics.items():
        print(
            f"  {label}: n2={row['n2_skill_vs_persistence']:+.3f} "
            f"n7={row['n7_skill_vs_persistence']:+.3f} "
            f"clip={row['clipped_state_values']}",
            flush=True,
        )
    return best_model, summary, rollouts, selection_rows


def save_dataset(path: Path, trajectories: dict[str, Trajectory], smoothing_tau_us: float) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["status"] = "PASS"
        handle.attrs["causal_smoothing"] = "exponential moving average"
        handle.attrs["smoothing_tau_us"] = smoothing_tau_us
        handle.attrs["step_time_us"] = 30.0
        handle.attrs["primary_e25_to_e22p5_read"] = False
        handle.create_dataset("modes", data=np.asarray(MODES, dtype=np.int64))
        handle.create_dataset(
            "control_names", data=np.asarray(CONTROL_NAMES, dtype=h5py.string_dtype())
        )
        for name, trajectory in trajectories.items():
            group = handle.require_group(name)
            group.attrs["source_h5"] = str(trajectory.path)
            group.attrs["current_ez_kvm"] = trajectory.current_ez_kvm
            group.attrs["source_ez_kvm"] = trajectory.source_ez_kvm
            group.attrs["transition"] = trajectory.transition
            group.create_dataset("time_us", data=trajectory.time_us)
            group.create_dataset("frame", data=trajectory.frame)
            group.create_dataset("raw_amplitude", data=trajectory.raw_amplitude)
            group.create_dataset("slow_amplitude", data=trajectory.slow_amplitude)
            group.create_dataset("controls", data=trajectory.controls)


def save_models(path: Path, models: dict[str, ControlledODE]) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["integration"] = "explicit midpoint"
        handle.attrs["state"] = "standardized log causal slow amplitudes"
        for name, model in models.items():
            group = handle.require_group(name)
            group.attrs["coupled"] = model.coupled
            group.attrs["history_controlled"] = model.history_controlled
            group.attrs["ridge"] = model.ridge
            group.attrs["state_clip"] = model.state_clip
            group.create_dataset("transform_mean", data=model.transform.mean)
            group.create_dataset("transform_scale", data=model.transform.scale)
            for index, mode in enumerate(MODES):
                mode_group = group.require_group(f"n{mode}")
                mode_group.create_dataset("weights", data=model.weights[index])
                mode_group.create_dataset(
                    "feature_names",
                    data=np.asarray(model.feature_names[index], dtype=h5py.string_dtype()),
                )


def save_rollouts(path: Path, all_rollouts: dict[str, dict[str, dict]]) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("modes", data=np.asarray(MODES, dtype=np.int64))
        for experiment, rollouts in all_rollouts.items():
            for label, rollout in rollouts.items():
                group = handle.require_group(f"{experiment}/{label}")
                for key in (
                    "time_us",
                    "truth_amplitude",
                    "prediction_amplitude",
                    "persistence_amplitude",
                    "truth_state",
                    "prediction_state",
                ):
                    group.create_dataset(key, data=rollout[key])
                group.attrs["clipped_state_values"] = rollout["clipped_state_values"]


def plot_comparison(path: Path, all_rollouts: dict[str, dict[str, dict]]) -> None:
    panels = (
        ("combined_current_coupled", "up_development", "up 22.5 kV/m: current Ez only"),
        ("combined_history_coupled", "up_development", "up 22.5 kV/m: Ez history"),
        ("loto_down_history_coupled", "down_holdout", "held-out down 22.5 -> 20"),
        ("loto_up_history_coupled", "up_holdout", "held-out up 20 -> 22.5"),
    )
    figure, axes = plt.subplots(len(panels), len(MODES), figsize=(12.0, 10.0))
    for row_index, (experiment, label, title) in enumerate(panels):
        rollout = all_rollouts[experiment][label]
        for mode_index, mode in enumerate(MODES):
            axis = axes[row_index, mode_index]
            axis.plot(
                rollout["time_us"],
                rollout["truth_amplitude"][:, mode_index],
                color="#111111",
                label="truth",
            )
            axis.plot(
                rollout["time_us"],
                rollout["prediction_amplitude"][:, mode_index],
                color="#0072B2",
                label="ODE",
            )
            axis.plot(
                rollout["time_us"],
                rollout["persistence_amplitude"][:, mode_index],
                color="#999999",
                linestyle=":",
                label="persistence",
            )
            axis.set_title(f"{title}; n={mode}", fontsize=9)
            axis.grid(alpha=0.25)
            if row_index == len(panels) - 1:
                axis.set_xlabel("time [us]")
            if mode_index == 0:
                axis.set_ylabel("slow amplitude")
            if row_index == 0 and mode_index == 0:
                axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e20", type=Path, default=DEFAULT_E20)
    parser.add_argument("--e25", type=Path, default=DEFAULT_E25)
    parser.add_argument("--up", type=Path, default=DEFAULT_UP)
    parser.add_argument("--down", type=Path, default=DEFAULT_DOWN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoothing-tau-us", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoothing_tau_us <= 0.0:
        raise ValueError("--smoothing-tau-us must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    trajectories = {
        "e20_stationary": load_trajectory(
            "e20_stationary", args.e20, 20.0, 20.0, False, args.smoothing_tau_us
        ),
        "e25_stationary": load_trajectory(
            "e25_stationary", args.e25, 25.0, 25.0, False, args.smoothing_tau_us
        ),
        "e20_to_e22p5": load_trajectory(
            "e20_to_e22p5", args.up, 22.5, 20.0, True, args.smoothing_tau_us
        ),
        "e22p5_to_e20": load_trajectory(
            "e22p5_to_e20", args.down, 20.0, 22.5, True, args.smoothing_tau_us
        ),
    }
    save_dataset(output / "coupled_amplitude_dataset.h5", trajectories, args.smoothing_tau_us)

    stationary = [
        ("e20_stationary", 12.0, 24.0),
        ("e25_stationary", 12.0, 24.0),
    ]
    combined_train = stationary + [
        ("e20_to_e22p5", 30.165, 35.0),
        ("e22p5_to_e20", 30.165, 33.5),
    ]
    combined_validation = [("e22p5_to_e20", 33.5, 34.86)]
    combined_tests = [
        ("up_development", "e20_to_e22p5", 35.0, 39.86),
        ("down_validation", "e22p5_to_e20", 33.5, 34.86),
    ]
    definitions = {
        "combined_current_coupled": (
            combined_train,
            combined_validation,
            combined_tests,
            True,
            False,
        ),
        "combined_history_uncoupled": (
            combined_train,
            combined_validation,
            combined_tests,
            False,
            True,
        ),
        "combined_history_coupled": (
            combined_train,
            combined_validation,
            combined_tests,
            True,
            True,
        ),
        "loto_down_history_coupled": (
            stationary + [("e20_to_e22p5", 30.165, 34.0)],
            [("e20_to_e22p5", 34.0, 34.86)],
            [("down_holdout", "e22p5_to_e20", 30.165, 34.86)],
            True,
            True,
        ),
        "loto_up_history_coupled": (
            stationary + [("e22p5_to_e20", 30.165, 33.8)],
            [("e22p5_to_e20", 33.8, 34.86)],
            [("up_holdout", "e20_to_e22p5", 30.165, 34.86)],
            True,
            True,
        ),
    }
    models = {}
    summaries = {}
    all_rollouts = {}
    selection_rows = []
    for name, definition in definitions.items():
        model, summary, rollouts, rows = fit_experiment(
            name,
            trajectories,
            *definition,
        )
        models[name] = model
        summaries[name] = summary
        all_rollouts[name] = rollouts
        selection_rows.extend(rows)

    save_models(output / "coupled_amplitude_ode_models.h5", models)
    save_rollouts(output / "coupled_amplitude_ode_rollouts.h5", all_rollouts)
    write_csv(output / "ridge_selection.csv", selection_rows)
    metric_rows = []
    for experiment, summary in summaries.items():
        for label, metrics in summary["test_metrics"].items():
            metric_rows.append(
                {
                    "experiment": experiment,
                    "rollout": label,
                    "selected_ridge": summary["selected_ridge"],
                    **metrics,
                }
            )
    write_csv(output / "rollout_metrics.csv", metric_rows)
    plot_comparison(output / "coupled_amplitude_ode_rollouts.png", all_rollouts)

    model_lock = {
        "primary_e25_to_e22p5_read": False,
        "primary_e25_to_e22p5_used_for_training": False,
        "primary_e25_to_e22p5_used_for_selection": False,
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "inputs": {
            name: {"path": str(trajectory.path), "sha256": sha256(trajectory.path)}
            for name, trajectory in trajectories.items()
        },
    }
    summary = {
        "status": "PASS",
        "model_class": "bounded polynomial coupled amplitude ODE",
        "integration": "explicit midpoint, dt=0.015 us",
        "amplitude_state": {
            "modes": MODES,
            "radial_reduction": "radial-weighted complex coefficient before magnitude",
            "smoothing": "causal exponential moving average",
            "smoothing_tau_us": args.smoothing_tau_us,
            "positivity": "log-amplitude state",
        },
        "experiments": summaries,
        "primary_e25_to_e22p5_read": False,
        "interpretation": [
            "LOTO tests measure direction transfer without fitting the held-out transition.",
            "The down-validation suffix selects ridge and is not an untouched primary test.",
            "The 35--40 us up continuation remains development validation.",
        ],
    }
    (output / "coupled_amplitude_ode_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output / "model_lock.json").write_text(
        json.dumps(json_safe(model_lock), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[DONE] {output}")


if __name__ == "__main__":
    main()
