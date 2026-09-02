"""Fit blockwise reduced dynamics to E25 unpooled Fourier latent states."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import analyze_radaz_hankel_havok as hankel
import analyze_radaz_reduced_dynamics as reduced


ROOT = Path(__file__).resolve().parent
DEFAULT_FEATURES = (
    ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_ez25kvm_fourier_latent_raw"
    / "fourier_latent_features.h5"
)
DEFAULT_OUTPUT = (
    ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_ez25kvm_blockwise_fourier_latent"
)

FIT_START_US = 20.0
VALIDATION_START_US = 23.0
FORECAST_START_US = 24.0
FORECAST_END_US = 30.0

BLOCKS = {
    "background_n0": (0, 0),
    "MTSI_n1_6": (1, 6),
    "transition_n7_8": (7, 8),
    "ECDI_n9_21": (9, 21),
}
BUDGETS = {
    "compact_12": {
        "background_n0": 2,
        "MTSI_n1_6": 4,
        "transition_n7_8": 1,
        "ECDI_n9_21": 5,
    },
    "medium_20": {
        "background_n0": 3,
        "MTSI_n1_6": 7,
        "transition_n7_8": 2,
        "ECDI_n9_21": 8,
    },
    "extended_32": {
        "background_n0": 4,
        "MTSI_n1_6": 10,
        "transition_n7_8": 3,
        "ECDI_n9_21": 15,
    },
}
METHODS = (
    "oracle_block_pca",
    "persistence",
    "standard_dmd",
    "hankel_dmd",
    "havok_zero_forcing",
)
COLORS = {
    "truth": "#111111",
    "oracle_block_pca": "#777777",
    "persistence": "#cc79a7",
    "standard_dmd": "#56b4e9",
    "hankel_dmd": "#0072b2",
    "havok_zero_forcing": "#009e73",
}


@dataclass
class BlockPCA:
    name: str
    mode_start: int
    mode_end: int
    components: int
    feature_shape: tuple[int, ...]
    full_mean: np.ndarray
    active: np.ndarray
    pca: PCA

    def transform(self, features: np.ndarray) -> np.ndarray:
        flat = block_slice(features, self.mode_start, self.mode_end).reshape(
            len(features), -1
        )
        return self.pca.transform(flat[:, self.active])

    def inverse(self, scores: np.ndarray) -> np.ndarray:
        active_values = self.pca.inverse_transform(scores)
        flat = np.broadcast_to(
            self.full_mean[None, :], (len(scores), len(self.full_mean))
        ).copy()
        flat[:, self.active] = active_values
        return flat.reshape((len(scores),) + self.feature_shape)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    finite = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(finite) < 3:
        return float("nan")
    left = left[finite] - np.mean(left[finite])
    right = right[finite] - np.mean(right[finite])
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= np.finfo(float).tiny:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def block_slice(
    features: np.ndarray, mode_start: int, mode_end: int
) -> np.ndarray:
    return features[:, :, :, mode_start : mode_end + 1, :]


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        features = np.asarray(
            handle["translator_fourier_ri"], dtype=np.float32
        )
        time_us = (
            np.asarray(handle["translator_time_s"], dtype=np.float64)
            * 1.0e6
        )
        frames = np.asarray(handle["translator_frame"], dtype=np.int64)
    if features.shape[3] < 22:
        raise ValueError("Latent feature file does not contain modes n=0-21")
    if not np.all(np.isfinite(features)):
        raise ValueError("Non-finite Fourier latent features")
    return features, time_us, frames


def fit_block_models(
    features: np.ndarray,
    fit_mask: np.ndarray,
    budget: dict[str, int],
) -> tuple[dict[str, BlockPCA], np.ndarray, list[dict]]:
    models: dict[str, BlockPCA] = {}
    scores = []
    rows = []
    for name, (mode_start, mode_end) in BLOCKS.items():
        values = block_slice(features, mode_start, mode_end)
        flat = values.reshape(len(values), -1)
        fit = flat[fit_mask]
        full_mean = np.mean(fit, axis=0)
        variance = np.var(fit, axis=0)
        floor = max(float(np.max(variance)) * 1.0e-12, 1.0e-20)
        active = variance > floor
        components = min(
            int(budget[name]),
            int(np.count_nonzero(active)),
            int(np.count_nonzero(fit_mask) - 1),
        )
        pca = PCA(
            n_components=components,
            svd_solver="randomized",
            random_state=0,
            iterated_power=5,
        )
        pca.fit(fit[:, active])
        model = BlockPCA(
            name,
            mode_start,
            mode_end,
            components,
            values.shape[1:],
            full_mean,
            active,
            pca,
        )
        models[name] = model
        scores.append(model.transform(features))
        rows.append(
            {
                "block": name,
                "mode_start": mode_start,
                "mode_end": mode_end,
                "requested_components": budget[name],
                "retained_components": components,
                "active_features": int(np.count_nonzero(active)),
                "total_features": int(flat.shape[1]),
                "explained_variance": float(
                    np.sum(pca.explained_variance_ratio_)
                ),
            }
        )
    return models, np.concatenate(scores, axis=1), rows


def decode_blocks(
    models: dict[str, BlockPCA], scores: np.ndarray, template: np.ndarray
) -> np.ndarray:
    decoded = np.zeros((len(scores),) + template.shape[1:], dtype=np.float32)
    offset = 0
    for name, (mode_start, mode_end) in BLOCKS.items():
        model = models[name]
        stop = offset + model.components
        decoded[:, :, :, mode_start : mode_end + 1, :] = model.inverse(
            scores[:, offset:stop]
        )
        offset = stop
    if offset != scores.shape[1]:
        raise ValueError("Block score dimension mismatch")
    return decoded


def make_rank_model(
    states: np.ndarray,
    delay: int,
    rank: int,
    delay_vectors: np.ndarray,
    delay_mean: np.ndarray,
    right: np.ndarray,
    singular_values: np.ndarray,
) -> hankel.HankelModel:
    basis = right[:rank].T
    coordinates = (delay_vectors - delay_mean) @ basis
    matrix = coordinates[1:].T @ np.linalg.pinv(
        coordinates[:-1].T, rcond=1.0e-10
    )
    return hankel.HankelModel(
        delay,
        rank,
        states.shape[1],
        delay_mean,
        basis,
        matrix,
        np.linalg.eigvals(matrix),
        singular_values,
    )


def search_hankel(
    standardized: np.ndarray,
    time_us: np.ndarray,
    delays: list[int],
    ranks: list[int],
) -> tuple[dict, list[dict]]:
    subtrain_mask = (time_us >= FIT_START_US) & (
        time_us < VALIDATION_START_US
    )
    validation_mask = (time_us >= VALIDATION_START_US) & (
        time_us < FORECAST_START_US
    )
    subtrain = standardized[subtrain_mask]
    validation = standardized[validation_mask]
    validation_time = time_us[validation_mask]
    persistence = np.repeat(subtrain[-1][None], len(validation), axis=0)
    dt_us = float(np.median(np.diff(time_us)))
    rows = []
    for delay in delays:
        delay_vectors = hankel.make_delay_vectors(subtrain, delay)
        delay_mean = np.mean(delay_vectors, axis=0)
        centered = delay_vectors - delay_mean
        _, singular_values, right = np.linalg.svd(
            centered, full_matrices=False
        )
        maximum_rank = min(len(delay_vectors) - 1, right.shape[0])
        for rank in ranks:
            if rank > maximum_rank:
                continue
            try:
                model = make_rank_model(
                    subtrain,
                    delay,
                    rank,
                    delay_vectors,
                    delay_mean,
                    right,
                    singular_values,
                )
                prediction = hankel.rollout_hankel(
                    model, subtrain, len(validation)
                )
                metrics, _ = reduced.evaluate_prediction(
                    validation, prediction, persistence, validation_time
                )
                mse = float(metrics["standardized_mse"])
                radius = float(np.max(np.abs(model.eigenvalues)))
            except (ValueError, np.linalg.LinAlgError):
                mse = float("inf")
                radius = float("nan")
                metrics = {}
            rows.append(
                {
                    "delay": delay,
                    "history_us": delay * dt_us,
                    "rank": rank,
                    "validation_mse": mse,
                    "validation_skill_vs_persistence": metrics.get(
                        "skill_vs_persistence", float("-inf")
                    ),
                    "validation_correlation": metrics.get(
                        "flattened_correlation", float("nan")
                    ),
                    "spectral_radius": radius,
                }
            )
    finite = [row for row in rows if np.isfinite(row["validation_mse"])]
    if not finite:
        raise RuntimeError("All blockwise Hankel candidates failed")
    best = min(
        finite,
        key=lambda row: (
            row["validation_mse"],
            row["delay"],
            row["rank"],
        ),
    )
    return best, rows


def block_balanced_nrmse(
    truth: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
) -> tuple[float, dict[str, float]]:
    values = {}
    for name, (mode_start, mode_end) in BLOCKS.items():
        target = block_slice(truth, mode_start, mode_end)
        estimate = block_slice(prediction, mode_start, mode_end)
        training = block_slice(reference, mode_start, mode_end)
        mse = float(np.mean((estimate - target) ** 2))
        variance = float(
            np.mean((training - np.mean(training, axis=0)) ** 2)
        )
        values[name] = float(np.sqrt(mse / max(variance, 1.0e-20)))
    return float(np.mean(list(values.values()))), values


def validate_budget(
    name: str,
    budget: dict[str, int],
    features: np.ndarray,
    time_us: np.ndarray,
    delays: list[int],
    ranks: list[int],
) -> tuple[dict, list[dict]]:
    subtrain_mask = (time_us >= FIT_START_US) & (
        time_us < VALIDATION_START_US
    )
    validation_mask = (time_us >= VALIDATION_START_US) & (
        time_us < FORECAST_START_US
    )
    models, scores, pca_rows = fit_block_models(
        features, subtrain_mask, budget
    )
    standardizer = reduced.fit_standardizer(scores[subtrain_mask])
    standardized = standardizer.transform(scores)
    selected, candidates = search_hankel(
        standardized, time_us, delays, ranks
    )
    model = hankel.fit_hankel_dmd(
        standardized[subtrain_mask], selected["delay"], selected["rank"]
    )
    prediction_std = hankel.rollout_hankel(
        model,
        standardized[subtrain_mask],
        int(np.count_nonzero(validation_mask)),
    )
    prediction_scores = standardizer.inverse(prediction_std)
    prediction = decode_blocks(models, prediction_scores, features)
    oracle = decode_blocks(models, scores[validation_mask], features)
    forecast_nrmse, block_nrmse = block_balanced_nrmse(
        features[validation_mask], prediction, features[subtrain_mask]
    )
    oracle_nrmse, oracle_blocks = block_balanced_nrmse(
        features[validation_mask], oracle, features[subtrain_mask]
    )
    summary = {
        "budget": name,
        "components": int(scores.shape[1]),
        "selected_delay": int(selected["delay"]),
        "selected_rank": int(selected["rank"]),
        "validation_state_mse": float(selected["validation_mse"]),
        "validation_block_balanced_feature_nrmse": forecast_nrmse,
        "validation_oracle_block_balanced_feature_nrmse": oracle_nrmse,
        "validation_nrmse_by_block": block_nrmse,
        "validation_oracle_nrmse_by_block": oracle_blocks,
        "pca": pca_rows,
    }
    for row in candidates:
        row["budget"] = name
        row["selected_within_budget"] = (
            row["delay"] == selected["delay"]
            and row["rank"] == selected["rank"]
        )
    print(
        f"[VALIDATE] {name}: dim={scores.shape[1]}, "
        f"q={selected['delay']}, rank={selected['rank']}, "
        f"feature NRMSE={forecast_nrmse:.4f}, oracle={oracle_nrmse:.4f}",
        flush=True,
    )
    return summary, candidates


def band_amplitude(
    features: np.ndarray, mode_start: int, mode_end: int
) -> np.ndarray:
    selected = block_slice(features, mode_start, mode_end)
    return np.sqrt(
        np.mean(np.sum(selected * selected, axis=-1), axis=(1, 2, 3))
    )


def coefficient_nrmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    rmse = float(np.sqrt(np.mean((prediction - truth) ** 2)))
    scale = float(
        np.sqrt(
            np.mean(
                (truth - np.mean(truth, axis=0, keepdims=True)) ** 2
            )
        )
    )
    return rmse / max(scale, 1.0e-20)


def final_analysis(
    budget_name: str,
    budget: dict[str, int],
    selection: dict,
    features: np.ndarray,
    time_us: np.ndarray,
) -> tuple[dict, list[dict], dict]:
    fit_mask = (time_us >= FIT_START_US) & (time_us < FORECAST_START_US)
    forecast_mask = (time_us >= FORECAST_START_US) & (
        time_us <= FORECAST_END_US
    )
    models, scores, pca_rows = fit_block_models(features, fit_mask, budget)
    standardizer = reduced.fit_standardizer(scores[fit_mask])
    standardized = standardizer.transform(scores)
    fit_states = standardized[fit_mask]
    truth_states = standardized[forecast_mask]
    forecast_time = time_us[forecast_mask]
    persistence_state = np.repeat(
        fit_states[-1][None], len(truth_states), axis=0
    )

    delay = int(selection["selected_delay"])
    rank = int(selection["selected_rank"])
    hankel_model = hankel.fit_hankel_dmd(fit_states, delay, rank)
    hankel_state = hankel.rollout_hankel(
        hankel_model, fit_states, len(truth_states)
    )
    standard_matrix, standard_eigenvalues = reduced.fit_dmd(fit_states)
    standard_state = reduced.rollout_linear(
        standard_matrix, fit_states[-1], len(truth_states)
    )
    havok_model, _ = hankel.fit_havok(hankel_model, fit_states)
    havok_state = hankel.rollout_havok_zero_forcing(
        hankel_model, havok_model, fit_states, len(truth_states)
    )
    state_predictions = {
        "persistence": persistence_state,
        "standard_dmd": standard_state,
        "hankel_dmd": hankel_state,
        "havok_zero_forcing": havok_state,
    }

    score_predictions = {
        "oracle_block_pca": scores[forecast_mask],
        **{
            method: standardizer.inverse(values)
            for method, values in state_predictions.items()
        },
    }
    truth = features[forecast_mask]
    metric_rows = []
    traces = {"truth": {}, "methods": {}}
    for block, (mode_start, mode_end) in BLOCKS.items():
        truth_amplitude = band_amplitude(truth, mode_start, mode_end)
        traces["truth"][block] = truth_amplitude

    for method, method_scores in score_predictions.items():
        prediction = decode_blocks(models, method_scores, features)
        traces["methods"][method] = {}
        for block, (mode_start, mode_end) in BLOCKS.items():
            truth_block = block_slice(truth, mode_start, mode_end)
            truth_amplitude = traces["truth"][block]
            prediction_block = block_slice(
                prediction, mode_start, mode_end
            )
            prediction_amplitude = band_amplitude(
                prediction, mode_start, mode_end
            )
            traces["methods"][method][block] = prediction_amplitude
            metric_rows.append(
                {
                    "method": method,
                    "block": block,
                    "mode_start": mode_start,
                    "mode_end": mode_end,
                    "coefficient_nrmse": coefficient_nrmse(
                        truth_block, prediction_block
                    ),
                    "amplitude_correlation": safe_correlation(
                        truth_amplitude, prediction_amplitude
                    ),
                    "mean_amplitude_ratio": float(
                        np.mean(prediction_amplitude)
                        / max(float(np.mean(truth_amplitude)), 1.0e-20)
                    ),
                }
            )
        del prediction

    state_metrics = {}
    for method, prediction in state_predictions.items():
        metrics, _ = hankel.evaluate_method_intervals(
            truth_states,
            prediction,
            persistence_state,
            forecast_time,
        )
        state_metrics[method] = metrics
    summary = {
        "selected_budget": budget_name,
        "components": int(scores.shape[1]),
        "selected_delay": delay,
        "selected_history_us": float(
            delay * np.median(np.diff(time_us))
        ),
        "selected_rank": rank,
        "hankel_spectral_radius": float(
            np.max(np.abs(hankel_model.eigenvalues))
        ),
        "standard_dmd_spectral_radius": float(
            np.max(np.abs(standard_eigenvalues))
        ),
        "state_metrics": state_metrics,
        "pca": pca_rows,
    }
    traces["time_us"] = forecast_time
    return summary, metric_rows, traces


def plot_validation(path: Path, rows: list[dict]) -> None:
    names = [row["budget"] for row in rows]
    x = np.arange(len(names), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(
        x,
        [row["validation_block_balanced_feature_nrmse"] for row in rows],
        color="#0072b2",
        label="Hankel forecast",
    )
    axes[0].scatter(
        x,
        [
            row["validation_oracle_block_balanced_feature_nrmse"]
            for row in rows
        ],
        color="#111111",
        marker="D",
        label="PCA oracle ceiling",
    )
    axes[0].set_ylabel("block-balanced coefficient NRMSE")
    axes[0].set_title("23-24 us validation")
    axes[0].legend(loc="upper right")
    axes[1].bar(
        x,
        [row["validation_state_mse"] for row in rows],
        color="#009e73",
    )
    axes[1].set_ylabel("standardized state MSE")
    axes[1].set_title("Hankel validation state error")
    for axis in axes:
        axis.set_xticks(x, names)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Blockwise Fourier latent state selection without holdout")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_rollouts(path: Path, traces: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    time_us = traces["time_us"]
    for axis, block in zip(axes.flat, BLOCKS):
        axis.plot(
            time_us,
            traces["truth"][block],
            color=COLORS["truth"],
            linewidth=2.0,
            label="truth",
        )
        for method in METHODS:
            axis.plot(
                time_us,
                traces["methods"][method][block],
                color=COLORS[method],
                linewidth=1.15,
                label=method,
            )
        axis.set_title(block)
        axis.set_ylabel("latent Fourier RMS")
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right", fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("time [us]")
    fig.suptitle("Selected blockwise state: autonomous 24-30 us rollout")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_readme(
    path: Path,
    validation_rows: list[dict],
    final_summary: dict,
    metrics: list[dict],
) -> None:
    selected = final_summary["selected_budget"]
    selected_row = next(
        row for row in validation_rows if row["budget"] == selected
    )
    lookup = {
        (row["method"], row["block"]): row for row in metrics
    }
    table = []
    for method in METHODS:
        for block in ("MTSI_n1_6", "ECDI_n9_21"):
            row = lookup[(method, block)]
            table.append(
                f"| {method} | {block} | {row['coefficient_nrmse']:.4f} | "
                f"{row['amplitude_correlation']:.4f} | "
                f"{row['mean_amplitude_ratio']:.4f} |"
            )
    text = f"""# E25 blockwise Fourier latent dynamics

The unpooled translator Fourier tensor is split into four physically motivated
mode blocks before PCA: background `n=0`, MTSI candidate `n=1-6`, transition
`n=7-8`, and ECDI candidate `n=9-21`.

- Candidate construction: `20-23 us`
- Budget/delay/rank selection: `23-24 us`
- Final identification: `20-24 us`
- Untouched autonomous forecast: `24-30 us`
- Selected budget: `{selected}` ({final_summary['components']} dimensions)
- Selected Hankel delay/rank: `{final_summary['selected_delay']}` / `{final_summary['selected_rank']}`
- Validation block-balanced NRMSE: `{selected_row['validation_block_balanced_feature_nrmse']:.4f}`

| Method | block | coefficient NRMSE | amplitude correlation | mean amplitude ratio |
|---|---|---:|---:|---:|
{chr(10).join(table)}

## 日本語メモ

全modeを一つのPCAへ入れると、振幅の大きい低波数成分が状態次元を使い切る。
そこでmode帯ごとに最低限の自由度を確保し、その状態が24-30 usを自律予測
できるかを調べた。候補の選択には24 us以降を使っていない。

`oracle_block_pca`は真の未来状態を各block PCAで圧縮・復元した上限であり、
予測ではない。Oracleが良くHankel/HAVOKが悪い場合は、表現力より時間発展
モデルが律速である。Oracle自体が悪いblockは、そのblockへ割り当てた状態
次元または潜在特徴そのものが不足している。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--delays", type=int, nargs="+", default=[20, 40, 60, 80, 100]
    )
    parser.add_argument(
        "--ranks", type=int, nargs="+", default=[8, 12, 20, 30, 40]
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    features, time_us, frames = load_features(args.features)
    validation_rows = []
    candidate_rows = []
    for budget_name, budget in BUDGETS.items():
        summary, candidates = validate_budget(
            budget_name,
            budget,
            features,
            time_us,
            args.delays,
            args.ranks,
        )
        validation_rows.append(summary)
        candidate_rows.extend(candidates)
    selected = min(
        validation_rows,
        key=lambda row: (
            row["validation_block_balanced_feature_nrmse"],
            row["components"],
        ),
    )
    selected_budget = selected["budget"]
    print(f"[SELECTED] {selected_budget}", flush=True)

    final_summary, metrics, traces = final_analysis(
        selected_budget,
        BUDGETS[selected_budget],
        selected,
        features,
        time_us,
    )
    write_csv(args.output / "budget_validation_metrics.csv", validation_rows)
    write_csv(args.output / "hankel_candidate_metrics.csv", candidate_rows)
    write_csv(args.output / "blockwise_mode_forecast_metrics.csv", metrics)
    plot_validation(
        args.output / "blockwise_budget_validation.png", validation_rows
    )
    plot_rollouts(
        args.output / "blockwise_mode_amplitude_rollouts.png", traces
    )

    rollout_rows = []
    for index, value in enumerate(traces["time_us"]):
        row = {"time_us": float(value), "frame": int(frames[np.flatnonzero((time_us >= FORECAST_START_US) & (time_us <= FORECAST_END_US))[index]])}
        for block in BLOCKS:
            row[f"truth_{block}"] = float(traces["truth"][block][index])
            for method in METHODS:
                row[f"{method}_{block}"] = float(
                    traces["methods"][method][block][index]
                )
        rollout_rows.append(row)
    write_csv(args.output / "blockwise_mode_amplitude_rollouts.csv", rollout_rows)

    result = {
        "status": "PASS",
        "feature_source": str(args.features),
        "blocks": BLOCKS,
        "budgets": BUDGETS,
        "validation": validation_rows,
        "final": final_summary,
        "mode_metrics": metrics,
        "notes": [
            "Budget, delay and rank were selected without using 24-30 us.",
            "The latent azimuthal width was not pooled; modes n=0-21 are explicit.",
        ],
    }
    (args.output / "blockwise_fourier_dynamics_summary.json").write_text(
        json.dumps(json_safe(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_readme(
        args.output / "README.md", validation_rows, final_summary, metrics
    )
    print(f"[DONE] {args.output}")


if __name__ == "__main__":
    main()
