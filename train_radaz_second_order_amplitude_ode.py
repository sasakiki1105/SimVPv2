"""Identify a history-controlled second-order n=2/n=7 amplitude ODE.

This extends the two-amplitude Markov audit with causal amplitude-rate states.
The kinematic relation d(log A)/dt = rate is imposed exactly, while coupled
rate accelerations are identified from allowed stationary, up-step, and
down-step development trajectories.  The blind E25 -> E22.5 primary case is
never read.
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

import train_radaz_coupled_amplitude_ode as first


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "workdirs" / "train_radaz_second_order_amplitude_ode"
RIDGES = (*first.RIDGES, 1.0e4)
VELOCITY_FEATURE_SCALE = 2.0


@dataclass
class SecondOrderODE:
    coupled: bool
    history_controlled: bool
    ridge: float
    feature_names: list[list[str]]
    weights: list[np.ndarray]
    transform: first.Transform
    rate_tau_us: float
    position_clip: float = 8.0
    velocity_clip: float = 20.0

    def acceleration(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        controls: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(
            [
                feature_vector(
                    position,
                    velocity,
                    controls,
                    output_index,
                    self.coupled,
                    self.history_controlled,
                )[0]
                @ self.weights[output_index]
                for output_index in range(len(first.MODES))
            ],
            dtype=np.float64,
        )

    def derivative(self, state: np.ndarray, controls: np.ndarray) -> np.ndarray:
        count = len(first.MODES)
        position = state[:count]
        velocity = state[count:]
        acceleration = self.acceleration(position, velocity, controls)
        return np.concatenate((velocity, acceleration))

    def rollout(
        self,
        initial_position: np.ndarray,
        initial_velocity: np.ndarray,
        controls: np.ndarray,
        dt_us: float,
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        count = len(first.MODES)
        state = np.concatenate((initial_position, initial_velocity)).astype(np.float64)
        positions = np.empty((len(controls), count), dtype=np.float64)
        velocities = np.empty_like(positions)
        position_clips = 0
        velocity_clips = 0
        for index, control in enumerate(controls):
            first_rate = self.derivative(state, control)
            midpoint = state + 0.5 * dt_us * first_rate
            following = state + dt_us * self.derivative(midpoint, control)
            bounded_position = np.clip(
                following[:count], -self.position_clip, self.position_clip
            )
            bounded_velocity = np.clip(
                following[count:], -self.velocity_clip, self.velocity_clip
            )
            position_clips += int(np.count_nonzero(bounded_position != following[:count]))
            velocity_clips += int(np.count_nonzero(bounded_velocity != following[count:]))
            state = np.concatenate((bounded_position, bounded_velocity))
            positions[index] = bounded_position
            velocities[index] = bounded_velocity
        return positions, velocities, position_clips, velocity_clips


def json_safe(value):
    return first.json_safe(value)


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


def causal_rate(
    trajectory: first.Trajectory,
    transform: first.Transform,
    rate_tau_us: float,
) -> tuple[np.ndarray, np.ndarray]:
    position = transform.encode(trajectory.slow_amplitude)
    dt_us = float(np.median(np.diff(trajectory.time_us)))
    raw = np.zeros_like(position)
    raw[1:] = np.diff(position, axis=0) / dt_us
    velocity = first.causal_ewma(raw, dt_us, rate_tau_us)
    return position, velocity


def feature_vector(
    position: np.ndarray,
    velocity: np.ndarray,
    controls: np.ndarray,
    output_index: int,
    coupled: bool,
    history_controlled: bool,
) -> tuple[np.ndarray, list[str]]:
    p = np.tanh(np.asarray(position, dtype=np.float64))
    v = np.tanh(np.asarray(velocity, dtype=np.float64) / VELOCITY_FEATURE_SCALE)
    control_indices = first.selected_control_indices(history_controlled)
    u = np.asarray(controls, dtype=np.float64)[control_indices]
    u_names = [first.CONTROL_NAMES[index] for index in control_indices]
    state_indices = list(range(len(first.MODES))) if coupled else [output_index]
    values = [1.0]
    names = ["constant"]
    for index in state_indices:
        values.extend((p[index], v[index]))
        names.extend(
            (
                f"tanh_z_n{first.MODES[index]}",
                f"tanh_rate_n{first.MODES[index]}",
            )
        )
    for left_position, left in enumerate(state_indices):
        for right in state_indices[left_position:]:
            values.extend(
                (
                    p[left] * p[right],
                    v[left] * v[right],
                )
            )
            names.extend(
                (
                    f"z_n{first.MODES[left]}*z_n{first.MODES[right]}",
                    f"rate_n{first.MODES[left]}*rate_n{first.MODES[right]}",
                )
            )
    for left in state_indices:
        for right in state_indices:
            values.append(p[left] * v[right])
            names.append(f"z_n{first.MODES[left]}*rate_n{first.MODES[right]}")
    values.extend(u.tolist())
    names.extend(u_names)
    for index in state_indices:
        for control_value, control_name in zip(u, u_names):
            values.extend((p[index] * control_value, v[index] * control_value))
            names.extend(
                (
                    f"z_n{first.MODES[index]}*{control_name}",
                    f"rate_n{first.MODES[index]}*{control_name}",
                )
            )
    return np.asarray(values, dtype=np.float64), names


def acceleration_rows(
    trajectories: dict[str, first.Trajectory],
    intervals: list[tuple[str, float, float]],
    transform: first.Transform,
    rate_tau_us: float,
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
        indices = first.interval_indices(trajectory, start_us, end_us)
        if len(indices) < 3 or not np.all(np.diff(indices) == 1):
            raise ValueError(f"Invalid acceleration interval {name} {start_us}:{end_us}")
        position, velocity = causal_rate(trajectory, transform, rate_tau_us)
        current_indices = indices[:-1]
        following_indices = indices[1:]
        dt_us = np.diff(trajectory.time_us[indices])
        target = (
            velocity[following_indices, output_index]
            - velocity[current_indices, output_index]
        ) / dt_us
        rows = []
        for index in current_indices:
            values, names_now = feature_vector(
                position[index],
                velocity[index],
                trajectory.controls[index],
                output_index,
                coupled,
                history_controlled,
            )
            rows.append(values)
            if feature_names is None:
                feature_names = names_now
            elif feature_names != names_now:
                raise AssertionError("Second-order feature library changed")
        matrices.append(np.asarray(rows))
        targets.append(target)
        weights.append(np.full(len(rows), 1.0 / len(rows), dtype=np.float64))
    return (
        np.concatenate(matrices),
        np.concatenate(targets),
        np.concatenate(weights),
        list(feature_names),
    )


def fit_ode(
    trajectories: dict[str, first.Trajectory],
    intervals: list[tuple[str, float, float]],
    coupled: bool,
    history_controlled: bool,
    ridge: float,
    rate_tau_us: float,
) -> SecondOrderODE:
    transform = first.fit_transform(trajectories, intervals)
    names_all = []
    weights_all = []
    for output_index in range(len(first.MODES)):
        matrix, target, sample_weight, names = acceleration_rows(
            trajectories,
            intervals,
            transform,
            rate_tau_us,
            output_index,
            coupled,
            history_controlled,
        )
        root_weight = np.sqrt(sample_weight)
        matrix_weighted = matrix * root_weight[:, None]
        target_weighted = target * root_weight
        penalty = np.eye(matrix.shape[1], dtype=np.float64) * ridge
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            matrix_weighted.T @ matrix_weighted + penalty,
            matrix_weighted.T @ target_weighted,
        )
        names_all.append(names)
        weights_all.append(coefficients)
    return SecondOrderODE(
        coupled,
        history_controlled,
        ridge,
        names_all,
        weights_all,
        transform,
        rate_tau_us,
    )


def rollout_interval(
    model: SecondOrderODE,
    trajectory: first.Trajectory,
    start_us: float,
    end_us: float,
) -> dict:
    indices = first.interval_indices(trajectory, start_us, end_us)
    if len(indices) < 3 or not np.all(np.diff(indices) == 1):
        raise ValueError(f"Invalid rollout interval {trajectory.name} {start_us}:{end_us}")
    first_index = int(indices[0])
    if first_index > 0:
        initial_index = first_index - 1
        target_indices = indices
    else:
        initial_index = first_index
        target_indices = indices[1:]
    position, velocity = causal_rate(trajectory, model.transform, model.rate_tau_us)
    dt_us = float(np.median(np.diff(trajectory.time_us)))
    prediction_state, prediction_velocity, position_clips, velocity_clips = model.rollout(
        position[initial_index],
        velocity[initial_index],
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
    return {
        "time_us": trajectory.time_us[target_indices],
        "truth_amplitude": truth_amplitude,
        "prediction_amplitude": prediction_amplitude,
        "persistence_amplitude": persistence_amplitude,
        "truth_state": position[target_indices],
        "prediction_state": prediction_state,
        "truth_velocity": velocity[target_indices],
        "prediction_velocity": prediction_velocity,
        "clipped_position_values": position_clips,
        "clipped_velocity_values": velocity_clips,
    }


def evaluate_rollout(rollout: dict) -> dict:
    result = {
        "frames": int(len(rollout["time_us"])),
        "first_time_us": float(rollout["time_us"][0]),
        "last_time_us": float(rollout["time_us"][-1]),
        "clipped_position_values": int(rollout["clipped_position_values"]),
        "clipped_velocity_values": int(rollout["clipped_velocity_values"]),
        "normalized_log_state_mse": float(
            np.mean((rollout["prediction_state"] - rollout["truth_state"]) ** 2)
        ),
    }
    skills = []
    for mode_index, mode in enumerate(first.MODES):
        metrics = first.scalar_metrics(
            rollout["truth_amplitude"][:, mode_index],
            rollout["prediction_amplitude"][:, mode_index],
            rollout["persistence_amplitude"][:, mode_index],
        )
        for key, value in metrics.items():
            result[f"n{mode}_{key}"] = value
        skills.append(metrics["skill_vs_persistence"])
    result["minimum_mode_skill_vs_persistence"] = float(min(skills))
    result["mean_mode_skill_vs_persistence"] = float(np.mean(skills))
    return result


def validation_score(
    model: SecondOrderODE,
    trajectories: dict[str, first.Trajectory],
    intervals: list[tuple[str, float, float]],
) -> float:
    scores = []
    for name, start_us, end_us in intervals:
        rollout = rollout_interval(model, trajectories[name], start_us, end_us)
        value = np.mean((rollout["prediction_state"] - rollout["truth_state"]) ** 2)
        if rollout["clipped_position_values"] or rollout["clipped_velocity_values"]:
            value += 1.0e3
        scores.append(value)
    return float(np.mean(scores))


def fit_experiment(
    name: str,
    trajectories: dict[str, first.Trajectory],
    selection_train: list[tuple[str, float, float]],
    validation: list[tuple[str, float, float]],
    final_train: list[tuple[str, float, float]],
    tests: list[tuple[str, str, float, float]],
    coupled: bool,
    history_controlled: bool,
    rate_tau_us: float,
) -> tuple[SecondOrderODE, dict, dict[str, dict], list[dict]]:
    rows = []
    best_ridge = None
    best_score = float("inf")
    for ridge in RIDGES:
        candidate = fit_ode(
            trajectories,
            selection_train,
            coupled,
            history_controlled,
            ridge,
            rate_tau_us,
        )
        score = validation_score(candidate, trajectories, validation)
        rows.append(
            {"experiment": name, "ridge": ridge, "validation_log_state_mse": score}
        )
        if score < best_score:
            best_score = score
            best_ridge = ridge
    if best_ridge is None:
        raise AssertionError("No second-order ODE was selected")
    model = fit_ode(
        trajectories,
        final_train,
        coupled,
        history_controlled,
        best_ridge,
        rate_tau_us,
    )
    rollouts = {}
    metrics = {}
    for label, trajectory_name, start_us, end_us in tests:
        rollout = rollout_interval(model, trajectories[trajectory_name], start_us, end_us)
        rollouts[label] = rollout
        metrics[label] = evaluate_rollout(rollout)
    summary = {
        "coupled": coupled,
        "history_controlled": history_controlled,
        "selected_ridge": best_ridge,
        "selection_validation_log_state_mse": best_score,
        "selection_train_intervals": selection_train,
        "validation_intervals": validation,
        "final_train_intervals": final_train,
        "test_metrics": metrics,
    }
    print(f"[{name}] ridge={best_ridge:g} validation={best_score:.4e}", flush=True)
    for label, result in metrics.items():
        print(
            f"  {label}: n2={result['n2_skill_vs_persistence']:+.3f} "
            f"n7={result['n7_skill_vs_persistence']:+.3f} "
            f"clips={result['clipped_position_values'] + result['clipped_velocity_values']}",
            flush=True,
        )
    return model, summary, rollouts, rows


def save_models(path: Path, models: dict[str, SecondOrderODE]) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["kinematic_constraint"] = "d standardized_log_amplitude / dt = rate"
        handle.attrs["integration"] = "explicit midpoint"
        for name, model in models.items():
            group = handle.require_group(name)
            group.attrs["coupled"] = model.coupled
            group.attrs["history_controlled"] = model.history_controlled
            group.attrs["ridge"] = model.ridge
            group.attrs["rate_tau_us"] = model.rate_tau_us
            group.create_dataset("transform_mean", data=model.transform.mean)
            group.create_dataset("transform_scale", data=model.transform.scale)
            for mode_index, mode in enumerate(first.MODES):
                mode_group = group.require_group(f"n{mode}")
                mode_group.create_dataset("acceleration_weights", data=model.weights[mode_index])
                mode_group.create_dataset(
                    "feature_names",
                    data=np.asarray(model.feature_names[mode_index], dtype=h5py.string_dtype()),
                )


def save_rollouts(path: Path, all_rollouts: dict[str, dict[str, dict]]) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("modes", data=np.asarray(first.MODES, dtype=np.int64))
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
                    "truth_velocity",
                    "prediction_velocity",
                ):
                    group.create_dataset(key, data=rollout[key])
                group.attrs["clipped_position_values"] = rollout["clipped_position_values"]
                group.attrs["clipped_velocity_values"] = rollout["clipped_velocity_values"]


def plot_comparison(path: Path, all_rollouts: dict[str, dict[str, dict]]) -> None:
    panels = (
        ("combined_current_coupled", "up_development", "up continuation: current Ez"),
        ("combined_history_uncoupled", "up_development", "up continuation: uncoupled"),
        ("combined_history_coupled", "up_development", "up continuation: coupled history"),
        ("loto_down_history_coupled", "down_holdout", "held-out down transition"),
        ("loto_up_history_coupled", "up_holdout", "held-out up transition"),
    )
    figure, axes = plt.subplots(len(panels), len(first.MODES), figsize=(12.0, 12.0))
    for row_index, (experiment, label, title) in enumerate(panels):
        rollout = all_rollouts[experiment][label]
        for mode_index, mode in enumerate(first.MODES):
            axis = axes[row_index, mode_index]
            for key, label_now, color, style in (
                ("truth_amplitude", "truth", "#111111", "-"),
                ("prediction_amplitude", "second-order ODE", "#0072B2", "-"),
                ("persistence_amplitude", "persistence", "#999999", ":"),
            ):
                axis.plot(
                    rollout["time_us"],
                    rollout[key][:, mode_index],
                    label=label_now,
                    color=color,
                    linestyle=style,
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
    parser.add_argument("--e20", type=Path, default=first.DEFAULT_E20)
    parser.add_argument("--e25", type=Path, default=first.DEFAULT_E25)
    parser.add_argument("--up", type=Path, default=first.DEFAULT_UP)
    parser.add_argument("--down", type=Path, default=first.DEFAULT_DOWN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoothing-tau-us", type=float, default=0.30)
    parser.add_argument("--rate-tau-us", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoothing_tau_us <= 0.0 or args.rate_tau_us <= 0.0:
        raise ValueError("Smoothing time constants must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    trajectories = {
        "e20_stationary": first.load_trajectory(
            "e20_stationary", args.e20, 20.0, 20.0, False, args.smoothing_tau_us
        ),
        "e25_stationary": first.load_trajectory(
            "e25_stationary", args.e25, 25.0, 25.0, False, args.smoothing_tau_us
        ),
        "e20_to_e22p5": first.load_trajectory(
            "e20_to_e22p5", args.up, 22.5, 20.0, True, args.smoothing_tau_us
        ),
        "e22p5_to_e20": first.load_trajectory(
            "e22p5_to_e20", args.down, 20.0, 22.5, True, args.smoothing_tau_us
        ),
    }
    stationary = [
        ("e20_stationary", 12.0, 24.0),
        ("e25_stationary", 12.0, 24.0),
    ]
    combined_selection_train = stationary + [
        ("e20_to_e22p5", 30.165, 34.0),
        ("e22p5_to_e20", 30.165, 33.5),
    ]
    combined_validation = [
        ("e20_to_e22p5", 34.0, 34.86),
        ("e22p5_to_e20", 33.5, 34.86),
    ]
    combined_final_train = stationary + [
        ("e20_to_e22p5", 30.165, 35.0),
        ("e22p5_to_e20", 30.165, 33.5),
    ]
    combined_tests = [
        ("up_development", "e20_to_e22p5", 35.0, 39.86),
        ("down_validation", "e22p5_to_e20", 33.5, 34.86),
    ]
    definitions = {
        "combined_current_coupled": (
            combined_selection_train,
            combined_validation,
            combined_final_train,
            combined_tests,
            True,
            False,
        ),
        "combined_history_uncoupled": (
            combined_selection_train,
            combined_validation,
            combined_final_train,
            combined_tests,
            False,
            True,
        ),
        "combined_history_coupled": (
            combined_selection_train,
            combined_validation,
            combined_final_train,
            combined_tests,
            True,
            True,
        ),
        "loto_down_history_coupled": (
            stationary + [("e20_to_e22p5", 30.165, 34.0)],
            [("e20_to_e22p5", 34.0, 34.86)],
            stationary + [("e20_to_e22p5", 30.165, 34.86)],
            [("down_holdout", "e22p5_to_e20", 30.165, 34.86)],
            True,
            True,
        ),
        "loto_up_history_coupled": (
            stationary + [("e22p5_to_e20", 30.165, 33.8)],
            [("e22p5_to_e20", 33.8, 34.86)],
            stationary + [("e22p5_to_e20", 30.165, 34.86)],
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
            args.rate_tau_us,
        )
        models[name] = model
        summaries[name] = summary
        all_rollouts[name] = rollouts
        selection_rows.extend(rows)

    save_models(output / "second_order_amplitude_ode_models.h5", models)
    save_rollouts(output / "second_order_amplitude_ode_rollouts.h5", all_rollouts)
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
    plot_comparison(output / "second_order_amplitude_ode_rollouts.png", all_rollouts)

    first_order_summary_path = (
        ROOT
        / "workdirs"
        / "train_radaz_coupled_amplitude_ode"
        / "coupled_amplitude_ode_summary.json"
    )
    first_order_reference = None
    if first_order_summary_path.is_file():
        first_order_reference = json.loads(
            first_order_summary_path.read_text(encoding="utf-8")
        )["experiments"]["combined_history_coupled"]["test_metrics"]["up_development"]
    current_metrics = summaries["combined_history_coupled"]["test_metrics"]["up_development"]
    summary = {
        "status": "PASS",
        "model_class": "second-order coupled controlled amplitude ODE",
        "kinematic_constraint": "d standardized log amplitude / dt = causal rate state",
        "smoothing_tau_us": args.smoothing_tau_us,
        "rate_tau_us": args.rate_tau_us,
        "experiments": summaries,
        "first_order_up_development_reference": first_order_reference,
        "second_order_up_development": current_metrics,
        "primary_e25_to_e22p5_read": False,
        "interpretation": [
            "Rate states provide a minimal causal phase coordinate for slow modulation.",
            "LOTO uses no amplitude samples from the held-out transition during fitting or selection.",
            "The 35--40 us up continuation remains development validation, not the blind primary test.",
        ],
    }
    (output / "second_order_amplitude_ode_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    script = Path(__file__).resolve()
    model_lock = {
        "primary_e25_to_e22p5_read": False,
        "primary_e25_to_e22p5_used_for_training": False,
        "primary_e25_to_e22p5_used_for_selection": False,
        "script": str(script),
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "inputs": {
            name: {"path": str(trajectory.path), "sha256": first.sha256(trajectory.path)}
            for name, trajectory in trajectories.items()
        },
    }
    (output / "model_lock.json").write_text(
        json.dumps(json_safe(model_lock), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[DONE] {output}")


if __name__ == "__main__":
    main()
