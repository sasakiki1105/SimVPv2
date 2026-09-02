#!/usr/bin/env python3
"""Compare fixed linear and fixed nonlinear ROMs for B25.

The experiment keeps the state coordinates and chronological holdouts used by
the preceding smooth-operator analysis. Hyperparameters are selected on the
last 2 us of the fitting interval, then each model is refit without test data
and rolled out autonomously for the held-out interval. B15 is a stationary
control. One-step scores are diagnostic and are not treated as closure.
"""

from __future__ import annotations

import csv
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze_radaz_b15_b25_smooth_operator_drift as smooth


ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT.parent
OUTPUT = (
    RESEARCH
    / "research_results"
    / "2D_RadAz"
    / "SimVPv2"
    / "workdirs"
    / "analyze_radaz_b15_b25_fixed_nonlinear_rom"
)

VALIDATION_US = 2.0
ALPHAS = (
    1.0e-6,
    1.0e-4,
    1.0e-3,
    1.0e-2,
    1.0e-1,
    1.0,
    10.0,
    100.0,
    1.0e3,
    1.0e4,
)
EDMD_ALPHAS = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1.0e3, 1.0e4)
MLP_CONFIGS = (
    {"hidden": 16, "alpha": 1.0e-4},
    {"hidden": 16, "alpha": 1.0e-2},
    {"hidden": 16, "alpha": 1.0},
    {"hidden": 16, "alpha": 100.0},
    {"hidden": 32, "alpha": 1.0e-4},
    {"hidden": 32, "alpha": 1.0e-2},
    {"hidden": 32, "alpha": 1.0},
    {"hidden": 32, "alpha": 100.0},
)
RNG_SEED = 20260825
MAX_ABS_STATE = 1.0e10


@dataclass(frozen=True)
class Protocol:
    case: str
    name: str
    fit_start_us: float
    fit_end_us: float
    test_end_us: float


PROTOCOLS = (
    Protocol("B25", "B25_12-24_to_30", 12.0, 24.0, 30.0),
    Protocol("B25", "B25_18-30_to_36", 18.0, 30.0, 36.0),
    Protocol("B15", "B15_12-24_to_29p75", 12.0, 24.0, 29.75),
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def interval_indices(
    time_us: np.ndarray, start_us: float, end_us: float
) -> np.ndarray:
    return np.flatnonzero(
        (time_us >= start_us - 1.0e-9) & (time_us < end_us - 1.0e-9)
    )


def transition_pairs(
    time_us: np.ndarray, states: np.ndarray, start_us: float, end_us: float
) -> tuple[np.ndarray, np.ndarray]:
    indices = interval_indices(time_us, start_us, end_us)
    if len(indices) < 4 or np.any(np.diff(indices) != 1):
        raise ValueError(f"invalid transition interval {start_us}-{end_us} us")
    values = states[indices]
    return values[:-1], values[1:]


def forecast_problem(
    time_us: np.ndarray,
    states: np.ndarray,
    history_start_us: float,
    forecast_start_us: float,
    forecast_end_us: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    history_indices = interval_indices(time_us, history_start_us, forecast_start_us)
    truth_indices = interval_indices(time_us, forecast_start_us, forecast_end_us)
    if len(history_indices) < 2 or len(truth_indices) < 1:
        raise ValueError("forecast interval is empty")
    initial = states[history_indices[-1]]
    return initial, states[truth_indices], time_us[truth_indices]


def metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    persistence: np.ndarray,
    dimensions: int,
) -> dict[str, float]:
    truth = np.asarray(truth[:, :dimensions], dtype=np.float64)
    prediction = np.asarray(prediction[:, :dimensions], dtype=np.float64)
    persistence = np.asarray(persistence[:, :dimensions], dtype=np.float64)
    finite = np.isfinite(prediction)
    finite_fraction = float(np.mean(finite))
    if finite_fraction < 1.0:
        return {
            "mse": float("inf"),
            "skill_vs_persistence": float("-inf"),
            "correlation": float("nan"),
            "finite_fraction": finite_fraction,
        }
    mse = float(np.mean((truth - prediction) ** 2))
    persistence_mse = float(np.mean((truth - persistence) ** 2))
    truth_flat = truth.ravel() - np.mean(truth)
    prediction_flat = prediction.ravel() - np.mean(prediction)
    denominator = np.linalg.norm(truth_flat) * np.linalg.norm(prediction_flat)
    correlation = (
        float(np.dot(truth_flat, prediction_flat) / denominator)
        if denominator > 1.0e-15
        else float("nan")
    )
    return {
        "mse": mse,
        "skill_vs_persistence": float(
            1.0 - mse / max(persistence_mse, 1.0e-15)
        ),
        "correlation": correlation,
        "finite_fraction": finite_fraction,
    }


class StateMap:
    family = "base"

    def predict_next(self, states: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def rollout(self, initial: np.ndarray, steps: int) -> np.ndarray:
        current = np.asarray(initial, dtype=np.float64).copy()
        prediction = np.full((steps, len(current)), np.nan, dtype=np.float64)
        for index in range(steps):
            current = np.asarray(self.predict_next(current[None])[0], dtype=np.float64)
            if not np.all(np.isfinite(current)) or np.max(np.abs(current)) > MAX_ABS_STATE:
                break
            prediction[index] = current
        return prediction


class PolynomialRidgeMap(StateMap):
    def __init__(self, degree: int, alpha: float, target: str):
        self.degree = degree
        self.alpha = alpha
        self.target = target
        self.family = "affine_ridge" if degree == 1 else "quadratic_ridge"
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        self.scaler = StandardScaler()
        self.regressor = Ridge(alpha=alpha, fit_intercept=True)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PolynomialRidgeMap":
        features = self.poly.fit_transform(x)
        features = self.scaler.fit_transform(features)
        target = y - x if self.target == "delta" else y
        self.regressor.fit(features, target)
        return self

    def predict_next(self, states: np.ndarray) -> np.ndarray:
        features = self.scaler.transform(self.poly.transform(states))
        values = self.regressor.predict(features)
        return states + values if self.target == "delta" else values


class QuadraticEDMDMap(StateMap):
    family = "quadratic_edmd"

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.scaler = StandardScaler()
        self.regressor = Ridge(alpha=alpha, fit_intercept=True)
        self.state_dimensions = 0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "QuadraticEDMDMap":
        self.state_dimensions = x.shape[1]
        lifted_x = self.poly.fit_transform(x)
        lifted_y = self.poly.transform(y)
        standardized_x = self.scaler.fit_transform(lifted_x)
        standardized_y = self.scaler.transform(lifted_y)
        self.regressor.fit(standardized_x, standardized_y)
        return self

    def predict_next(self, states: np.ndarray) -> np.ndarray:
        lifted = self.scaler.transform(self.poly.transform(states))
        following = self.regressor.predict(lifted)
        raw = self.scaler.inverse_transform(following)
        return raw[:, : self.state_dimensions]

    def rollout(self, initial: np.ndarray, steps: int) -> np.ndarray:
        lifted = self.scaler.transform(self.poly.transform(initial[None]))
        prediction = np.full((steps, self.state_dimensions), np.nan, dtype=np.float64)
        for index in range(steps):
            lifted = self.regressor.predict(lifted)
            raw = self.scaler.inverse_transform(lifted)
            current = raw[0, : self.state_dimensions]
            if not np.all(np.isfinite(current)) or np.max(np.abs(current)) > MAX_ABS_STATE:
                break
            prediction[index] = current
        return prediction


class ResidualMLPMap(StateMap):
    family = "residual_mlp"

    def __init__(self, hidden: int, alpha: float, seed: int = RNG_SEED):
        self.hidden = hidden
        self.alpha = alpha
        self.seed = seed
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        self.regressor = MLPRegressor(
            hidden_layer_sizes=(hidden,),
            activation="tanh",
            solver="lbfgs",
            alpha=alpha,
            max_iter=700,
            random_state=seed,
            tol=1.0e-8,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ResidualMLPMap":
        scaled_x = self.input_scaler.fit_transform(x)
        delta = y - x
        scaled_delta = self.output_scaler.fit_transform(delta)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.regressor.fit(scaled_x, scaled_delta)
        return self

    def predict_next(self, states: np.ndarray) -> np.ndarray:
        scaled = self.input_scaler.transform(states)
        delta = self.output_scaler.inverse_transform(self.regressor.predict(scaled))
        return states + delta


class OrganisationResidualMap(StateMap):
    family = "organisation_residual"

    def __init__(self, base_alpha: float, residual_alpha: float, latent_dimensions: int):
        self.base_alpha = base_alpha
        self.residual_alpha = residual_alpha
        self.latent_dimensions = latent_dimensions
        self.base_scaler = StandardScaler()
        self.base = Ridge(alpha=base_alpha, fit_intercept=True)
        self.residual_scaler = StandardScaler()
        self.residual = Ridge(alpha=residual_alpha, fit_intercept=True)

    def residual_features(self, states: np.ndarray) -> np.ndarray:
        latent = states[:, : self.latent_dimensions]
        organisation = states[:, self.latent_dimensions :]
        interaction = (latent[:, :, None] * organisation[:, None, :]).reshape(
            len(states), -1
        )
        return np.concatenate((organisation, interaction), axis=1)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "OrganisationResidualMap":
        if x.shape[1] <= self.latent_dimensions:
            raise ValueError("organisation residual requires L+A+P state")
        scaled_x = self.base_scaler.fit_transform(x)
        self.base.fit(scaled_x, y)
        base_prediction = self.base.predict(scaled_x)
        residual_target = (
            y[:, : self.latent_dimensions]
            - base_prediction[:, : self.latent_dimensions]
        )
        features = self.residual_scaler.fit_transform(self.residual_features(x))
        self.residual.fit(features, residual_target)
        return self

    def predict_next(self, states: np.ndarray) -> np.ndarray:
        prediction = self.base.predict(self.base_scaler.transform(states))
        features = self.residual_scaler.transform(self.residual_features(states))
        prediction[:, : self.latent_dimensions] += self.residual.predict(features)
        return prediction


def candidate_configs(family: str) -> list[dict]:
    if family in ("affine_ridge", "quadratic_ridge"):
        return [
            {"alpha": alpha, "target": target}
            for alpha in ALPHAS
            for target in ("direct", "delta")
        ]
    if family == "quadratic_edmd":
        return [{"alpha": alpha} for alpha in EDMD_ALPHAS]
    if family == "residual_mlp":
        return [dict(config) for config in MLP_CONFIGS]
    if family == "organisation_residual":
        return [
            {"base_alpha": base_alpha, "residual_alpha": residual_alpha}
            for base_alpha in (1.0e-2, 1.0, 100.0)
            for residual_alpha in (1.0e-2, 1.0, 100.0, 1.0e4)
        ]
    raise ValueError(family)


def make_model(family: str, config: dict, latent_dimensions: int) -> StateMap:
    if family == "affine_ridge":
        return PolynomialRidgeMap(1, float(config["alpha"]), str(config["target"]))
    if family == "quadratic_ridge":
        return PolynomialRidgeMap(2, float(config["alpha"]), str(config["target"]))
    if family == "quadratic_edmd":
        return QuadraticEDMDMap(float(config["alpha"]))
    if family == "residual_mlp":
        return ResidualMLPMap(int(config["hidden"]), float(config["alpha"]))
    if family == "organisation_residual":
        return OrganisationResidualMap(
            float(config["base_alpha"]),
            float(config["residual_alpha"]),
            latent_dimensions,
        )
    raise ValueError(family)


def fit_candidate(
    family: str,
    config: dict,
    x: np.ndarray,
    y: np.ndarray,
    latent_dimensions: int,
) -> StateMap:
    return make_model(family, config, latent_dimensions).fit(x, y)


def select_model(
    family: str,
    time_us: np.ndarray,
    states: np.ndarray,
    protocol: Protocol,
    latent_dimensions: int,
    representation: str,
) -> tuple[dict, list[dict]]:
    validation_start = protocol.fit_end_us - VALIDATION_US
    train_x, train_y = transition_pairs(
        time_us, states, protocol.fit_start_us, validation_start
    )
    initial, truth, _ = forecast_problem(
        time_us,
        states,
        protocol.fit_start_us,
        validation_start,
        protocol.fit_end_us,
    )
    persistence = np.repeat(initial[None], len(truth), axis=0)
    rows = []
    for config in candidate_configs(family):
        try:
            model = fit_candidate(
                family, config, train_x, train_y, latent_dimensions
            )
            prediction = model.rollout(initial, len(truth))
            score = metrics(truth, prediction, persistence, latent_dimensions)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            score = {
                "mse": float("inf"),
                "skill_vs_persistence": float("-inf"),
                "correlation": float("nan"),
                "finite_fraction": 0.0,
            }
        row = {
            "protocol": protocol.name,
            "case": protocol.case,
            "representation": representation,
            "family": family,
            **config,
            **{f"validation_{key}": value for key, value in score.items()},
        }
        rows.append(row)
    finite = [row for row in rows if math.isfinite(float(row["validation_mse"]))]
    if not finite:
        selected = min(rows, key=lambda row: -float(row["validation_finite_fraction"]))
    else:
        selected = min(
            finite,
            key=lambda row: (
                float(row["validation_mse"]),
                -float(row["validation_correlation"])
                if math.isfinite(float(row["validation_correlation"]))
                else float("inf"),
            ),
        )
    config_keys = set(candidate_configs(family)[0])
    selected_config = {key: selected[key] for key in config_keys}
    for row in rows:
        row["selected"] = int(all(row.get(key) == value for key, value in selected_config.items()))
    return selected_config, rows


def one_step_prediction(
    model: StateMap,
    time_us: np.ndarray,
    states: np.ndarray,
    start_us: float,
    end_us: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth_indices = interval_indices(time_us, start_us, end_us)
    previous = states[truth_indices - 1]
    truth = states[truth_indices]
    prediction = model.predict_next(previous)
    persistence = previous
    return truth, prediction, persistence


def evaluate_protocol(
    protocol: Protocol,
    representation: str,
    time_us: np.ndarray,
    states: np.ndarray,
    latent_dimensions: int,
) -> tuple[list[dict], list[dict], dict[str, np.ndarray]]:
    selection_rows = []
    forecast_rows = []
    trajectories: dict[str, np.ndarray] = {}
    initial, truth, forecast_time = forecast_problem(
        time_us,
        states,
        protocol.fit_start_us,
        protocol.fit_end_us,
        protocol.test_end_us,
    )
    persistence = np.repeat(initial[None], len(truth), axis=0)
    base = metrics(truth, persistence, persistence, latent_dimensions)
    forecast_rows.append(
        {
            "protocol": protocol.name,
            "case": protocol.case,
            "representation": representation,
            "method": "persistence",
            "evaluation": "autonomous_rollout",
            "fit_start_us": protocol.fit_start_us,
            "fit_end_us": protocol.fit_end_us,
            "test_end_us": float(forecast_time[-1]),
            **base,
        }
    )
    trajectories["time_us"] = forecast_time
    trajectories["truth"] = truth[:, :latent_dimensions]
    trajectories["persistence"] = persistence[:, :latent_dimensions]

    full_x, full_y = transition_pairs(
        time_us, states, protocol.fit_start_us, protocol.fit_end_us
    )
    families = [
        "affine_ridge",
        "quadratic_ridge",
        "quadratic_edmd",
        "residual_mlp",
    ]
    if representation == "L+A+P":
        families.append("organisation_residual")
    for family in families:
        selected, rows = select_model(
            family,
            time_us,
            states,
            protocol,
            latent_dimensions,
            representation,
        )
        selection_rows.extend(rows)
        model = fit_candidate(
            family, selected, full_x, full_y, latent_dimensions
        )
        prediction = model.rollout(initial, len(truth))
        score = metrics(truth, prediction, persistence, latent_dimensions)
        full_score = metrics(truth, prediction, persistence, states.shape[1])
        forecast_rows.append(
            {
                "protocol": protocol.name,
                "case": protocol.case,
                "representation": representation,
                "method": family,
                "evaluation": "autonomous_rollout",
                "fit_start_us": protocol.fit_start_us,
                "fit_end_us": protocol.fit_end_us,
                "test_end_us": float(forecast_time[-1]),
                **selected,
                **score,
                "full_state_mse": full_score["mse"],
                "full_state_skill_vs_persistence": full_score["skill_vs_persistence"],
                "full_state_correlation": full_score["correlation"],
            }
        )
        one_truth, one_prediction, one_persistence = one_step_prediction(
            model,
            time_us,
            states,
            protocol.fit_end_us,
            protocol.test_end_us,
        )
        one_score = metrics(
            one_truth, one_prediction, one_persistence, latent_dimensions
        )
        forecast_rows.append(
            {
                "protocol": protocol.name,
                "case": protocol.case,
                "representation": representation,
                "method": family,
                "evaluation": "teacher_forced_one_step",
                "fit_start_us": protocol.fit_start_us,
                "fit_end_us": protocol.fit_end_us,
                "test_end_us": float(forecast_time[-1]),
                **selected,
                **one_score,
            }
        )
        trajectories[family] = prediction[:, :latent_dimensions]
    return selection_rows, forecast_rows, trajectories


def plot_skills(rows: list[dict], output: Path) -> None:
    selected = [row for row in rows if row["evaluation"] == "autonomous_rollout"]
    protocols = [protocol.name for protocol in PROTOCOLS]
    base_methods = [
        "affine_ridge",
        "quadratic_ridge",
        "quadratic_edmd",
        "residual_mlp",
    ]
    labels = {
        "affine_ridge": "Affine ridge",
        "quadratic_ridge": "Quadratic map",
        "quadratic_edmd": "Quadratic EDMD",
        "residual_mlp": "Residual MLP",
        "organisation_residual": "Organisation residual",
    }
    colors = {
        "affine_ridge": "#4C78A8",
        "quadratic_ridge": "#F58518",
        "quadratic_edmd": "#54A24B",
        "residual_mlp": "#E45756",
        "organisation_residual": "#B279A2",
    }
    figure, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    x = np.arange(len(protocols))
    for axis, representation in zip(axes, ("L", "L+A+P")):
        methods = list(base_methods)
        if representation == "L+A+P":
            methods.append("organisation_residual")
        width = 0.72 / len(methods)
        for offset, method in enumerate(methods):
            values = []
            for protocol in protocols:
                match = next(
                    row
                    for row in selected
                    if row["protocol"] == protocol
                    and row["representation"] == representation
                    and row["method"] == method
                )
                value = float(match["skill_vs_persistence"])
                values.append(np.clip(value, -2.0, 1.0))
            axis.bar(
                x + (offset - (len(methods) - 1) / 2.0) * width,
                values,
                width,
                label=labels[method],
                color=colors[method],
            )
        axis.axhline(0.0, color="black", linewidth=1.0)
        axis.set_ylim(-2.05, 1.05)
        axis.set_ylabel("Skill vs persistence\n(clipped below at -2)")
        axis.set_title(f"State {representation}")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(loc="lower right", ncol=3)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(protocols)
    figure.suptitle("Fixed linear vs fixed nonlinear autonomous ROM", fontsize=16)
    figure.tight_layout()
    figure.savefig(output / "autonomous_skill_comparison.png", dpi=180)
    plt.close(figure)


def plot_trajectories(
    stored: dict[tuple[str, str], dict[str, np.ndarray]], output: Path
) -> None:
    protocols = ("B25_12-24_to_30", "B25_18-30_to_36")
    base_methods = (
        "truth",
        "affine_ridge",
        "quadratic_ridge",
        "quadratic_edmd",
        "residual_mlp",
    )
    labels = {
        "truth": "Truth",
        "affine_ridge": "Affine ridge",
        "quadratic_ridge": "Quadratic map",
        "quadratic_edmd": "Quadratic EDMD",
        "residual_mlp": "Residual MLP",
        "organisation_residual": "Organisation residual",
    }
    colors = {
        "truth": "black",
        "affine_ridge": "#4C78A8",
        "quadratic_ridge": "#F58518",
        "quadratic_edmd": "#54A24B",
        "residual_mlp": "#E45756",
        "organisation_residual": "#B279A2",
    }
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), sharex="row")
    for row_index, protocol in enumerate(protocols):
        for column_index, representation in enumerate(("L", "L+A+P")):
            axis = axes[row_index, column_index]
            data = stored[(protocol, representation)]
            time_us = data["time_us"]
            methods = list(base_methods)
            if representation == "L+A+P":
                methods.append("organisation_residual")
            truth_pc1 = data["truth"][:, 0]
            lower = float(np.min(truth_pc1))
            upper = float(np.max(truth_pc1))
            span = max(upper - lower, 0.5)
            display_lower = lower - 0.25 * span
            display_upper = upper + 0.25 * span
            for method in methods:
                values = data[method][:, 0]
                axis.plot(
                    time_us,
                    np.clip(values, display_lower, display_upper),
                    label=labels[method],
                    color=colors[method],
                    linewidth=2.2 if method == "truth" else 1.3,
                    alpha=0.95 if method == "truth" else 0.8,
                )
            axis.set_title(f"{protocol}, {representation}, latent PC1")
            axis.set_ylabel("Standardized PC1")
            axis.set_ylim(display_lower, display_upper)
            axis.grid(alpha=0.25)
            axis.legend(loc="lower right", ncol=2, fontsize=8)
    axes[-1, 0].set_xlabel("Target time (us)")
    axes[-1, 1].set_xlabel("Target time (us)")
    figure.suptitle(
        "B25 autonomous trajectories after chronological holdout "
        "(display clipped outside truth range + 25%)",
        fontsize=16,
    )
    figure.tight_layout()
    figure.savefig(output / "b25_autonomous_pc1_trajectories.png", dpi=180)
    plt.close(figure)


def plot_one_step_vs_rollout(rows: list[dict], output: Path) -> None:
    methods = (
        "affine_ridge",
        "quadratic_ridge",
        "quadratic_edmd",
        "residual_mlp",
        "organisation_residual",
    )
    colors = {
        "affine_ridge": "#4C78A8",
        "quadratic_ridge": "#F58518",
        "quadratic_edmd": "#54A24B",
        "residual_mlp": "#E45756",
        "organisation_residual": "#B279A2",
    }
    figure, axis = plt.subplots(figsize=(9, 8))
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        for protocol in PROTOCOLS:
            for representation in ("L", "L+A+P"):
                pair = [
                    row
                    for row in subset
                    if row["protocol"] == protocol.name
                    and row["representation"] == representation
                ]
                if not pair:
                    continue
                rollout = next(
                    float(row["skill_vs_persistence"])
                    for row in pair
                    if row["evaluation"] == "autonomous_rollout"
                )
                one_step = next(
                    float(row["skill_vs_persistence"])
                    for row in pair
                    if row["evaluation"] == "teacher_forced_one_step"
                )
                axis.scatter(
                    np.clip(one_step, -2.0, 1.0),
                    np.clip(rollout, -2.0, 1.0),
                    color=colors[method],
                    s=65,
                    alpha=0.8,
                )
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.axvline(0.0, color="black", linewidth=1.0)
    axis.plot([-2, 1], [-2, 1], color="gray", linestyle="--", linewidth=1.0)
    for method in methods:
        axis.scatter([], [], color=colors[method], label=method)
    axis.set_xlim(-2.05, 1.05)
    axis.set_ylim(-2.05, 1.05)
    axis.set_xlabel("Teacher-forced one-step skill (clipped at -2)")
    axis.set_ylabel("Autonomous rollout skill (clipped at -2)")
    axis.set_title("One-step fit is not autonomous closure")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output / "one_step_vs_autonomous_skill.png", dpi=180)
    plt.close(figure)


def summarize(rows: list[dict]) -> dict:
    autonomous = [row for row in rows if row["evaluation"] == "autonomous_rollout"]
    summary = {}
    for protocol in PROTOCOLS:
        summary[protocol.name] = {}
        for representation in ("L", "L+A+P"):
            local = [
                row
                for row in autonomous
                if row["protocol"] == protocol.name
                and row["representation"] == representation
                and row["method"] != "persistence"
            ]
            affine = next(row for row in local if row["method"] == "affine_ridge")
            nonlinear = [row for row in local if row["method"] != "affine_ridge"]
            best = max(nonlinear, key=lambda row: float(row["skill_vs_persistence"]))
            summary[protocol.name][representation] = {
                "affine_skill": float(affine["skill_vs_persistence"]),
                "affine_correlation": float(affine["correlation"]),
                "best_nonlinear_method": best["method"],
                "best_nonlinear_skill": float(best["skill_vs_persistence"]),
                "best_nonlinear_correlation": float(best["correlation"]),
                "nonlinear_skill_gain": float(best["skill_vs_persistence"])
                - float(affine["skill_vs_persistence"]),
            }
    b25_gains = [
        summary[protocol.name][representation]["nonlinear_skill_gain"]
        for protocol in PROTOCOLS
        if protocol.case == "B25"
        for representation in ("L", "L+A+P")
    ]
    summary["interpretation"] = {
        "b25_positive_gain_count": int(sum(gain > 0.05 for gain in b25_gains)),
        "b25_comparisons": len(b25_gains),
        "consistent_fixed_nonlinear_support": bool(all(gain > 0.05 for gain in b25_gains)),
        "criterion": "nonlinear skill exceeds affine skill by >0.05 in all four B25 protocol/state comparisons",
    }
    return summary


def make_readme(summary: dict, rows: list[dict], output: Path) -> None:
    lines = [
        "# B15/B25 fixed nonlinear ROM comparison",
        "",
        "This experiment tests whether B25 is better described by one stationary nonlinear map than by a time-varying affine map.",
        "",
        "## Protocol",
        "",
        "- The frozen SimVP/PCA coordinates and L / L+A+P state definitions match the preceding operator-drift study.",
        "- The final 2 us of each fitting interval selects hyperparameters; no test target is used for selection.",
        "- Each selected model is refit on the full fitting interval and autonomously rolled out for about 6 us.",
        "- B15 is a stationary control. Teacher-forced one-step scores are diagnostic only.",
        "- Quadratic ridge recomputes polynomial features from its predicted state; quadratic EDMD evolves the lifted state linearly.",
        "- On L+A+P, the organisation-residual model adds only latent-organisation interaction corrections to a global affine full-state map.",
        "",
        "## Autonomous latent forecast",
        "",
        "| protocol | state | affine skill | best nonlinear | nonlinear skill | gain | nonlinear corr |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for protocol in PROTOCOLS:
        for representation in ("L", "L+A+P"):
            item = summary[protocol.name][representation]
            lines.append(
                f"| {protocol.name} | {representation} | {item['affine_skill']:.3f} | "
                f"{item['best_nonlinear_method']} | {item['best_nonlinear_skill']:.3f} | "
                f"{item['nonlinear_skill_gain']:.3f} | {item['best_nonlinear_correlation']:.3f} |"
            )
    criterion = summary["interpretation"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"A nonlinear gain above 0.05 occurred in {criterion['b25_positive_gain_count']} of {criterion['b25_comparisons']} B25 protocol/state comparisons.",
        ]
    )
    if criterion["consistent_fixed_nonlinear_support"]:
        lines.append(
            "The preregistered descriptive criterion is met: fixed nonlinearity consistently improves B25 over the affine baseline. This supports, but does not prove, a stationary nonlinear closure because a flexible map can still approximate hidden nonstationarity over a finite interval."
        )
    else:
        lines.append(
            "The descriptive consistency criterion is not met. Fixed low-complexity nonlinearity does not provide a reproducible explanation of B25 closure failure across both chronological holdouts and both state definitions."
        )
    lines.extend(
        [
            "",
            "A high one-step score with a poor autonomous rollout indicates local curve fitting rather than a closed predictive state. Results remain exploratory because B15/B25 each have one PIC realization and the state coordinates were designed during earlier analyses.",
            "",
            "## Files",
            "",
            "- `model_selection.csv`: validation-only hyperparameter selection",
            "- `forecast_metrics.csv`: autonomous and teacher-forced metrics",
            "- `analysis_summary.json`: machine-readable conclusions",
            "- `selected_trajectories.npz`: latent truth and selected forecasts",
            "- `autonomous_skill_comparison.png`",
            "- `b25_autonomous_pc1_trajectories.png`",
            "- `one_step_vs_autonomous_skill.png`",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    representations = {
        case: smooth.load_representation(case) for case in ("B15", "B25")
    }
    selection_rows: list[dict] = []
    forecast_rows: list[dict] = []
    stored: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    archive = {}
    for protocol in PROTOCOLS:
        case = representations[protocol.case]
        for representation, states in case.states.items():
            selected, forecasts, trajectories = evaluate_protocol(
                protocol,
                representation,
                case.time_us,
                states,
                case.latent_dimensions,
            )
            selection_rows.extend(selected)
            forecast_rows.extend(forecasts)
            stored[(protocol.name, representation)] = trajectories
            for key, value in trajectories.items():
                safe_key = f"{protocol.name}__{representation.replace('+', '_')}__{key}"
                archive[safe_key] = value
            print(f"completed {protocol.name} {representation}", flush=True)

    summary = summarize(forecast_rows)
    write_csv(OUTPUT / "model_selection.csv", selection_rows)
    write_csv(OUTPUT / "forecast_metrics.csv", forecast_rows)
    np.savez_compressed(OUTPUT / "selected_trajectories.npz", **archive)
    (OUTPUT / "analysis_summary.json").write_text(
        json.dumps(
            {
                "protocols": [asdict(protocol) for protocol in PROTOCOLS],
                "summary": json_safe(summary),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_skills(forecast_rows, OUTPUT)
    plot_trajectories(stored, OUTPUT)
    plot_one_step_vs_rollout(forecast_rows, OUTPUT)
    make_readme(summary, forecast_rows, OUTPUT)
    print(json.dumps(json_safe(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
