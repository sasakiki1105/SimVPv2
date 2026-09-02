"""Fit a causal delay-coordinate ROM for RadAz n=2/n=7 slow amplitudes.

The delay state supplies the missing modulation phase found by the first- and
second-order amplitude ODE audits.  Ridge and delay length are selected only
on allowed development trajectories.  Leave-one-transition-out forecasts use
an observed causal initialization history but never fit the held-out path.
The blind E25 -> E22.5 primary trajectory is never read.
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

import train_radaz_coupled_amplitude_ode as amplitude


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "workdirs" / "train_radaz_delay_amplitude_rom"
DELAYS = (20, 40, 80)
RIDGES = (1.0e-4, 1.0e-2, 1.0, 100.0, 1000.0)


@dataclass
class DelayROM:
    delay: int
    ridge: float
    coupled: bool
    history_controlled: bool
    transform: amplitude.Transform
    weights: list[np.ndarray]
    feature_names: list[list[str]]
    state_clip: float = 8.0

    def step(
        self,
        history: np.ndarray,
        controls: np.ndarray,
    ) -> np.ndarray:
        following = np.empty(len(amplitude.MODES), dtype=np.float64)
        current = history[-1]
        for output_index in range(len(amplitude.MODES)):
            features, _ = feature_vector(
                history,
                controls,
                output_index,
                self.coupled,
                self.history_controlled,
            )
            following[output_index] = current[output_index] + features @ self.weights[output_index]
        return following

    def rollout(
        self,
        initial_history: np.ndarray,
        controls: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        history = np.asarray(initial_history, dtype=np.float64).copy()
        if history.shape != (self.delay, len(amplitude.MODES)):
            raise ValueError(f"Expected history {(self.delay, len(amplitude.MODES))}, got {history.shape}")
        prediction = np.empty((len(controls), len(amplitude.MODES)), dtype=np.float64)
        clipped = 0
        for index, control in enumerate(controls):
            following = self.step(history, control)
            bounded = np.clip(following, -self.state_clip, self.state_clip)
            clipped += int(np.count_nonzero(bounded != following))
            history[:-1] = history[1:]
            history[-1] = bounded
            prediction[index] = bounded
        return prediction, clipped


def json_safe(value):
    return amplitude.json_safe(value)


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


def feature_vector(
    history: np.ndarray,
    controls: np.ndarray,
    output_index: int,
    coupled: bool,
    history_controlled: bool,
) -> tuple[np.ndarray, list[str]]:
    history = np.asarray(history, dtype=np.float64)
    control_indices = amplitude.selected_control_indices(history_controlled)
    selected_controls = np.asarray(controls, dtype=np.float64)[control_indices]
    control_names = [amplitude.CONTROL_NAMES[index] for index in control_indices]
    mode_indices = list(range(len(amplitude.MODES))) if coupled else [output_index]
    values = [1.0]
    names = ["constant"]
    # Most recent value is lag zero.  The complete causal delay line is the
    # lifted state, so oscillation phase need not be inferred from amplitude.
    for mode_index in mode_indices:
        for lag, value in enumerate(history[::-1, mode_index]):
            values.append(value)
            names.append(f"z_n{amplitude.MODES[mode_index]}_lag{lag}")
    values.extend(selected_controls.tolist())
    names.extend(control_names)
    current = np.tanh(history[-1])
    for mode_index in mode_indices:
        for control, control_name in zip(selected_controls, control_names):
            values.append(current[mode_index] * control)
            names.append(f"tanh_z_n{amplitude.MODES[mode_index]}*{control_name}")
    return np.asarray(values, dtype=np.float64), names


def regression_rows(
    trajectories: dict[str, amplitude.Trajectory],
    intervals: list[tuple[str, float, float]],
    transform: amplitude.Transform,
    delay: int,
    output_index: int,
    coupled: bool,
    history_controlled: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    matrices = []
    targets = []
    sample_weights = []
    feature_names = None
    for name, start_us, end_us in intervals:
        trajectory = trajectories[name]
        interval = amplitude.interval_indices(trajectory, start_us, end_us)
        state = transform.encode(trajectory.slow_amplitude)
        targets_now = [
            int(index)
            for index in interval
            if int(index) - delay >= 0
            and int(index) > int(interval[0])
            and int(index) - delay >= max(0, int(interval[0]) - 1)
        ]
        if len(targets_now) < 3:
            raise ValueError(f"Insufficient delay rows for {name} {start_us}:{end_us}")
        rows = []
        delta = []
        for target_index in targets_now:
            history = state[target_index - delay : target_index]
            values, names_now = feature_vector(
                history,
                trajectory.controls[target_index],
                output_index,
                coupled,
                history_controlled,
            )
            rows.append(values)
            delta.append(state[target_index, output_index] - state[target_index - 1, output_index])
            if feature_names is None:
                feature_names = names_now
            elif feature_names != names_now:
                raise AssertionError("Delay feature library changed")
        matrices.append(np.asarray(rows))
        targets.append(np.asarray(delta))
        sample_weights.append(np.full(len(rows), 1.0 / len(rows), dtype=np.float64))
    return (
        np.concatenate(matrices),
        np.concatenate(targets),
        np.concatenate(sample_weights),
        list(feature_names),
    )


def fit_model(
    trajectories: dict[str, amplitude.Trajectory],
    intervals: list[tuple[str, float, float]],
    delay: int,
    ridge: float,
    coupled: bool,
    history_controlled: bool,
) -> DelayROM:
    transform = amplitude.fit_transform(trajectories, intervals)
    weights = []
    names_all = []
    for output_index in range(len(amplitude.MODES)):
        matrix, target, sample_weight, names = regression_rows(
            trajectories,
            intervals,
            transform,
            delay,
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
        weights.append(coefficients)
        names_all.append(names)
    return DelayROM(
        delay,
        ridge,
        coupled,
        history_controlled,
        transform,
        weights,
        names_all,
    )


def rollout_interval(
    model: DelayROM,
    trajectory: amplitude.Trajectory,
    start_us: float,
    end_us: float,
) -> dict:
    indices = amplitude.interval_indices(trajectory, start_us, end_us)
    if len(indices) < 3 or not np.all(np.diff(indices) == 1):
        raise ValueError(f"Invalid delay rollout interval {trajectory.name} {start_us}:{end_us}")
    target_start = int(indices[0])
    history_stop = target_start
    history_start = history_stop - model.delay
    if history_start < 0:
        raise ValueError(
            f"{trajectory.name} needs {model.delay} observed frames before {start_us} us"
        )
    state = model.transform.encode(trajectory.slow_amplitude)
    initial_history = state[history_start:history_stop]
    prediction_state, clipped = model.rollout(
        initial_history,
        trajectory.controls[indices],
    )
    truth_amplitude = trajectory.slow_amplitude[indices]
    prediction_amplitude = model.transform.decode(prediction_state)
    persistence_amplitude = np.repeat(
        trajectory.slow_amplitude[history_stop - 1 : history_stop],
        len(indices),
        axis=0,
    )
    return {
        "time_us": trajectory.time_us[indices],
        "truth_amplitude": truth_amplitude,
        "prediction_amplitude": prediction_amplitude,
        "persistence_amplitude": persistence_amplitude,
        "truth_state": state[indices],
        "prediction_state": prediction_state,
        "clipped_state_values": clipped,
        "observed_history_start_us": float(trajectory.time_us[history_start]),
        "observed_history_end_us": float(trajectory.time_us[history_stop - 1]),
    }


def evaluate_rollout(rollout: dict) -> dict:
    result = {
        "frames": int(len(rollout["time_us"])),
        "first_time_us": float(rollout["time_us"][0]),
        "last_time_us": float(rollout["time_us"][-1]),
        "observed_history_start_us": rollout["observed_history_start_us"],
        "observed_history_end_us": rollout["observed_history_end_us"],
        "clipped_state_values": int(rollout["clipped_state_values"]),
        "normalized_log_state_mse": float(
            np.mean((rollout["prediction_state"] - rollout["truth_state"]) ** 2)
        ),
    }
    skills = []
    for mode_index, mode in enumerate(amplitude.MODES):
        metrics = amplitude.scalar_metrics(
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
    model: DelayROM,
    trajectories: dict[str, amplitude.Trajectory],
    intervals: list[tuple[str, float, float]],
) -> float:
    values = []
    for name, start_us, end_us in intervals:
        rollout = rollout_interval(model, trajectories[name], start_us, end_us)
        score = np.mean((rollout["prediction_state"] - rollout["truth_state"]) ** 2)
        if rollout["clipped_state_values"]:
            score += 1.0e3
        values.append(score)
    return float(np.mean(values))


def fit_experiment(
    name: str,
    trajectories: dict[str, amplitude.Trajectory],
    selection_train: list[tuple[str, float, float]],
    validation: list[tuple[str, float, float]],
    final_train: list[tuple[str, float, float]],
    tests: list[tuple[str, str, float, float]],
    coupled: bool,
    history_controlled: bool,
) -> tuple[DelayROM, dict, dict[str, dict], list[dict]]:
    best = None
    best_score = float("inf")
    rows = []
    for delay in DELAYS:
        for ridge in RIDGES:
            candidate = fit_model(
                trajectories,
                selection_train,
                delay,
                ridge,
                coupled,
                history_controlled,
            )
            score = validation_score(candidate, trajectories, validation)
            rows.append(
                {
                    "experiment": name,
                    "delay_steps": delay,
                    "delay_us": delay * 0.015,
                    "ridge": ridge,
                    "validation_log_state_mse": score,
                }
            )
            if score < best_score:
                best_score = score
                best = (delay, ridge)
    if best is None:
        raise AssertionError("No delay ROM was selected")
    model = fit_model(
        trajectories,
        final_train,
        best[0],
        best[1],
        coupled,
        history_controlled,
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
        "selected_delay_steps": best[0],
        "selected_delay_us": best[0] * 0.015,
        "selected_ridge": best[1],
        "selection_validation_log_state_mse": best_score,
        "selection_train_intervals": selection_train,
        "validation_intervals": validation,
        "final_train_intervals": final_train,
        "test_metrics": metrics,
    }
    print(
        f"[{name}] delay={best[0]} ({best[0] * .015:.3f} us) "
        f"ridge={best[1]:g} validation={best_score:.4e}",
        flush=True,
    )
    for label, result in metrics.items():
        print(
            f"  {label}: n2={result['n2_skill_vs_persistence']:+.3f} "
            f"n7={result['n7_skill_vs_persistence']:+.3f} "
            f"clip={result['clipped_state_values']}",
            flush=True,
        )
    return model, summary, rollouts, rows


def save_models(path: Path, models: dict[str, DelayROM]) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["state"] = "causal delay line of standardized log slow amplitudes"
        handle.attrs["prediction"] = "one-step amplitude increment, recursive rollout"
        for name, model in models.items():
            group = handle.require_group(name)
            group.attrs["delay_steps"] = model.delay
            group.attrs["ridge"] = model.ridge
            group.attrs["coupled"] = model.coupled
            group.attrs["history_controlled"] = model.history_controlled
            group.create_dataset("transform_mean", data=model.transform.mean)
            group.create_dataset("transform_scale", data=model.transform.scale)
            for mode_index, mode in enumerate(amplitude.MODES):
                mode_group = group.require_group(f"n{mode}")
                mode_group.create_dataset("weights", data=model.weights[mode_index])
                mode_group.create_dataset(
                    "feature_names",
                    data=np.asarray(model.feature_names[mode_index], dtype=h5py.string_dtype()),
                )


def save_rollouts(path: Path, rollouts_all: dict[str, dict[str, dict]]) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("modes", data=np.asarray(amplitude.MODES, dtype=np.int64))
        for experiment, rollouts in rollouts_all.items():
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
                group.attrs["observed_history_start_us"] = rollout["observed_history_start_us"]
                group.attrs["observed_history_end_us"] = rollout["observed_history_end_us"]


def plot_comparison(path: Path, rollouts_all: dict[str, dict[str, dict]]) -> None:
    panels = (
        ("combined_current_coupled", "up_development", "up continuation: current Ez"),
        ("combined_history_uncoupled", "up_development", "up continuation: uncoupled"),
        ("combined_history_coupled", "up_development", "up continuation: coupled history"),
        ("loto_down_history_coupled", "down_holdout", "held-out down after history"),
        ("loto_up_history_coupled", "up_holdout", "held-out up after history"),
    )
    figure, axes = plt.subplots(len(panels), len(amplitude.MODES), figsize=(12.0, 12.0))
    for row_index, (experiment, label, title) in enumerate(panels):
        rollout = rollouts_all[experiment][label]
        for mode_index, mode in enumerate(amplitude.MODES):
            axis = axes[row_index, mode_index]
            for key, label_now, color, style in (
                ("truth_amplitude", "truth", "#111111", "-"),
                ("prediction_amplitude", "delay ROM", "#0072B2", "-"),
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
    parser.add_argument("--e20", type=Path, default=amplitude.DEFAULT_E20)
    parser.add_argument("--e25", type=Path, default=amplitude.DEFAULT_E25)
    parser.add_argument("--up", type=Path, default=amplitude.DEFAULT_UP)
    parser.add_argument("--down", type=Path, default=amplitude.DEFAULT_DOWN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoothing-tau-us", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    trajectories = {
        "e20_stationary": amplitude.load_trajectory(
            "e20_stationary", args.e20, 20.0, 20.0, False, args.smoothing_tau_us
        ),
        "e25_stationary": amplitude.load_trajectory(
            "e25_stationary", args.e25, 25.0, 25.0, False, args.smoothing_tau_us
        ),
        "e20_to_e22p5": amplitude.load_trajectory(
            "e20_to_e22p5", args.up, 22.5, 20.0, True, args.smoothing_tau_us
        ),
        "e22p5_to_e20": amplitude.load_trajectory(
            "e22p5_to_e20", args.down, 20.0, 22.5, True, args.smoothing_tau_us
        ),
    }
    stationary = [
        ("e20_stationary", 12.0, 24.0),
        ("e25_stationary", 12.0, 24.0),
    ]
    combined_selection = stationary + [
        ("e20_to_e22p5", 30.165, 34.0),
        ("e22p5_to_e20", 30.165, 33.5),
    ]
    combined_validation = [
        ("e20_to_e22p5", 34.0, 34.86),
        ("e22p5_to_e20", 33.5, 34.86),
    ]
    combined_final = stationary + [
        ("e20_to_e22p5", 30.165, 35.0),
        ("e22p5_to_e20", 30.165, 33.5),
    ]
    combined_tests = [
        ("up_development", "e20_to_e22p5", 35.0, 39.86),
        ("down_validation", "e22p5_to_e20", 33.5, 34.86),
    ]
    definitions = {
        "combined_current_coupled": (
            combined_selection,
            combined_validation,
            combined_final,
            combined_tests,
            True,
            False,
        ),
        "combined_history_uncoupled": (
            combined_selection,
            combined_validation,
            combined_final,
            combined_tests,
            False,
            True,
        ),
        "combined_history_coupled": (
            combined_selection,
            combined_validation,
            combined_final,
            combined_tests,
            True,
            True,
        ),
        "loto_down_history_coupled": (
            stationary + [("e20_to_e22p5", 30.165, 34.0)],
            [("e20_to_e22p5", 34.0, 34.86)],
            stationary + [("e20_to_e22p5", 30.165, 34.86)],
            [("down_holdout", "e22p5_to_e20", 32.0, 34.86)],
            True,
            True,
        ),
        "loto_up_history_coupled": (
            stationary + [("e22p5_to_e20", 30.165, 33.8)],
            [("e22p5_to_e20", 33.8, 34.86)],
            stationary + [("e22p5_to_e20", 30.165, 34.86)],
            [("up_holdout", "e20_to_e22p5", 32.0, 34.86)],
            True,
            True,
        ),
    }
    models = {}
    summaries = {}
    rollouts_all = {}
    selection_rows = []
    for name, definition in definitions.items():
        model, summary, rollouts, rows = fit_experiment(
            name,
            trajectories,
            *definition,
        )
        models[name] = model
        summaries[name] = summary
        rollouts_all[name] = rollouts
        selection_rows.extend(rows)

    save_models(output / "delay_amplitude_rom_models.h5", models)
    save_rollouts(output / "delay_amplitude_rom_rollouts.h5", rollouts_all)
    write_csv(output / "delay_ridge_selection.csv", selection_rows)
    metric_rows = []
    for experiment, summary in summaries.items():
        for label, metrics in summary["test_metrics"].items():
            metric_rows.append(
                {
                    "experiment": experiment,
                    "rollout": label,
                    "selected_delay_steps": summary["selected_delay_steps"],
                    "selected_delay_us": summary["selected_delay_us"],
                    "selected_ridge": summary["selected_ridge"],
                    **metrics,
                }
            )
    write_csv(output / "rollout_metrics.csv", metric_rows)
    plot_comparison(output / "delay_amplitude_rom_rollouts.png", rollouts_all)
    summary = {
        "status": "PASS",
        "model_class": "causal coupled delay-coordinate amplitude ROM",
        "amplitude_smoothing_tau_us": args.smoothing_tau_us,
        "candidate_delays_steps": DELAYS,
        "candidate_delays_us": [value * 0.015 for value in DELAYS],
        "experiments": summaries,
        "primary_e25_to_e22p5_read": False,
        "interpretation": [
            "Delay state is an observed causal state, not a prescribed transition clock.",
            "LOTO rollouts begin at 32 us after an observed initialization history.",
            "No held-out transition sample is used to fit or select its LOTO model.",
        ],
    }
    (output / "delay_amplitude_rom_summary.json").write_text(
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
            name: {"path": str(trajectory.path), "sha256": amplitude.sha256(trajectory.path)}
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
