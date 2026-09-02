"""Test whether a few physical Fourier observables close latent dynamics.

The experiment augments a fixed blockwise latent state with radial mode
envelopes, density--electric-field cross spectra, or modal transport.  Every
model is identified on 20--24 us and autonomously rolled out over 24--30 us.
No physical observable from the forecast interval is used as an input.
"""

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

import analyze_radaz_blockwise_fourier_latent_dynamics as block
import analyze_radaz_hankel_havok as hankel
import analyze_radaz_reduced_dynamics as reduced


ROOT = Path(__file__).resolve().parent
DEFAULT_PHYSICAL = (
    ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_ez25kvm_fourier_latent_to_physical_modes"
    / "physical_fourier_targets.h5"
)
DEFAULT_OUTPUT = (
    ROOT
    / "workdirs"
    / "compare_radaz_e25_augmented_physical_state_dynamics"
)

CHECKPOINTS = {
    "data_only": {
        "features": ROOT
        / "workdirs"
        / "analyze_radaz_bx20mt_ez25kvm_fourier_latent_raw"
        / "fourier_latent_features.h5",
        "budget": "medium_20",
    },
    "spectral_full": {
        "features": ROOT
        / "workdirs"
        / "analyze_radaz_bx20mt_ez25kvm_fourier_latent_spectral_full_raw"
        / "fourier_latent_features.h5",
        "budget": "extended_32",
    },
}

VARIANTS = {
    "latent_only": (),
    "latent_radial": ("radial",),
    "latent_transport": ("transport",),
    "latent_cross": ("cross",),
    "latent_radial_transport": ("radial", "transport"),
    "latent_radial_cross": ("radial", "cross"),
}
MODE_BANDS = {
    "MTSI_n1_6": np.arange(1, 7, dtype=np.int64),
    "ECDI_n9_21": np.arange(9, 22, dtype=np.int64),
}
RADIAL_GROUPS = (
    np.asarray([0, 1]),
    np.asarray([2, 3]),
    np.asarray([4, 5]),
    np.asarray([6, 7]),
)
B_T = 0.020
FIT_START_US = 20.0
VALIDATION_START_US = 23.0
FORECAST_START_US = 24.0
FORECAST_END_US = 30.0


@dataclass
class PhysicalStates:
    time_us: np.ndarray
    frame: np.ndarray
    radial: np.ndarray
    cross: np.ndarray
    transport: np.ndarray
    macro_weights: np.ndarray


@dataclass
class GroupScaler:
    names: tuple[str, ...]
    slices: dict[str, slice]
    means: dict[str, np.ndarray]
    scales: dict[str, np.ndarray]
    weights: dict[str, float]

    @classmethod
    def fit(
        cls, groups: dict[str, np.ndarray], fit_mask: np.ndarray
    ) -> "GroupScaler":
        names = tuple(groups)
        slices: dict[str, slice] = {}
        means: dict[str, np.ndarray] = {}
        scales: dict[str, np.ndarray] = {}
        weights: dict[str, float] = {}
        offset = 0
        for name, values in groups.items():
            width = values.shape[1]
            slices[name] = slice(offset, offset + width)
            offset += width
            means[name] = np.mean(values[fit_mask], axis=0)
            scale = np.std(values[fit_mask], axis=0, ddof=1)
            scales[name] = np.where(scale > 1.0e-12, scale, 1.0)
            weights[name] = 1.0 / math.sqrt(width)
        return cls(names, slices, means, scales, weights)

    def transform(self, groups: dict[str, np.ndarray]) -> np.ndarray:
        transformed = []
        for name in self.names:
            values = groups[name]
            transformed.append(
                (values - self.means[name])
                / self.scales[name]
                * self.weights[name]
            )
        return np.concatenate(transformed, axis=1)

    def inverse(self, values: np.ndarray) -> dict[str, np.ndarray]:
        result = {}
        for name in self.names:
            selected = values[:, self.slices[name]]
            result[name] = (
                selected / self.weights[name] * self.scales[name]
                + self.means[name]
            )
        return result


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


def load_physical_states(path: Path) -> PhysicalStates:
    with h5py.File(path, "r") as handle:
        coefficients = np.asarray(handle["coefficients"], dtype=np.complex128)
        time_us = np.asarray(handle["time_us"], dtype=np.float64)
        frame = np.asarray(handle["frame"], dtype=np.int64)
        radial_weights = np.asarray(handle["radial_weights"], dtype=np.float64)
        raw_fields = np.asarray(handle["fields"])
    fields = [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in raw_fields
    ]
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("Non-finite physical Fourier coefficients")
    phi = coefficients[:, fields.index("phi")]
    electron = coefficients[:, fields.index("electron_den")]
    efy = coefficients[:, fields.index("efy")]

    radial = np.empty((len(time_us), 4, 2), dtype=np.float64)
    cross = np.empty((len(time_us), 4, 2), dtype=np.complex128)
    macro_weights = np.empty(4, dtype=np.float64)
    for radial_index, indices in enumerate(RADIAL_GROUPS):
        macro_weights[radial_index] = np.sum(radial_weights[indices])
        local_weights = radial_weights[indices] / macro_weights[radial_index]
        for band_index, modes in enumerate(MODE_BANDS.values()):
            phi_selected = phi[:, indices][:, :, modes]
            modal_power = np.mean(np.abs(phi_selected) ** 2, axis=2)
            radial[:, radial_index, band_index] = np.sqrt(
                np.einsum("r,tr->t", local_weights, modal_power)
            )
            ne_selected = electron[:, indices][:, :, modes]
            ey_selected = efy[:, indices][:, :, modes]
            modal_cross = np.sum(
                ne_selected * np.conj(ey_selected), axis=2
            )
            cross[:, radial_index, band_index] = np.einsum(
                "r,tr->t", local_weights, modal_cross
            )

    global_cross = np.einsum("r,trb->tb", macro_weights, cross)
    transport = -2.0 * np.real(global_cross) / B_T
    macro_weights = macro_weights / np.sum(macro_weights)
    return PhysicalStates(
        time_us=time_us,
        frame=frame,
        radial=radial,
        cross=cross,
        transport=transport,
        macro_weights=macro_weights,
    )


def flatten_physical(physical: PhysicalStates) -> dict[str, np.ndarray]:
    cross_ri = np.stack(
        [physical.cross.real, physical.cross.imag], axis=-1
    )
    return {
        "radial": physical.radial.reshape(len(physical.time_us), -1),
        "cross": cross_ri.reshape(len(physical.time_us), -1),
        "transport": physical.transport.copy(),
    }


def unflatten_cross(values: np.ndarray) -> np.ndarray:
    shaped = values.reshape(len(values), 4, 2, 2)
    return shaped[..., 0] + 1j * shaped[..., 1]


def nrmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    finite = np.isfinite(prediction)
    if not np.all(finite):
        return float("inf")
    rmse = float(np.sqrt(np.mean(np.abs(prediction - truth) ** 2)))
    scale = float(
        np.sqrt(np.mean(np.abs(truth - np.mean(truth, axis=0)) ** 2))
    )
    return rmse / max(scale, np.finfo(float).tiny)


def correlation(truth: np.ndarray, prediction: np.ndarray) -> float:
    if np.iscomplexobj(truth) or np.iscomplexobj(prediction):
        truth = np.stack([np.real(truth), np.imag(truth)], axis=-1)
        prediction = np.stack(
            [np.real(prediction), np.imag(prediction)], axis=-1
        )
    return block.safe_correlation(truth, prediction)


def temporal_anomaly_correlation(
    truth: np.ndarray, prediction: np.ndarray
) -> float:
    if truth.ndim == 1:
        return correlation(truth, prediction)
    truth_anomaly = truth - np.mean(truth, axis=0, keepdims=True)
    prediction_anomaly = prediction - np.mean(
        prediction, axis=0, keepdims=True
    )
    return correlation(truth_anomaly, prediction_anomaly)


def scalar_metrics(
    truth: np.ndarray, prediction: np.ndarray, persistence: np.ndarray
) -> dict[str, float]:
    finite = np.all(np.isfinite(prediction), axis=1) if prediction.ndim > 1 else np.isfinite(prediction)
    if not np.all(finite):
        mse = float("inf")
    else:
        mse = float(np.mean(np.abs(prediction - truth) ** 2))
    persistence_mse = float(np.mean(np.abs(persistence - truth) ** 2))
    return {
        "nrmse": nrmse(truth, prediction),
        "correlation": correlation(truth, prediction),
        "temporal_anomaly_correlation": temporal_anomaly_correlation(
            truth, prediction
        ),
        "skill_vs_persistence": (
            1.0 - mse / persistence_mse
            if np.isfinite(mse) and persistence_mse > 0.0
            else float("-inf")
        ),
    }


def weighted_phase_mae(
    truth: np.ndarray, prediction: np.ndarray, radial_weights: np.ndarray
) -> float:
    error = np.angle(prediction) - np.angle(truth)
    error = np.abs((error + np.pi) % (2.0 * np.pi) - np.pi)
    weight = np.abs(truth) * radial_weights[None, :, None]
    denominator = float(np.sum(weight))
    return float(np.sum(error * weight) / max(denominator, 1.0e-30))


def transport_from_cross(
    cross: np.ndarray, radial_weights: np.ndarray
) -> np.ndarray:
    global_cross = np.einsum("r,trb->tb", radial_weights, cross)
    return -2.0 * np.real(global_cross) / B_T


def groups_for_variant(
    latent: np.ndarray,
    physical_flat: dict[str, np.ndarray],
    variant: str,
) -> dict[str, np.ndarray]:
    groups = {"latent": latent}
    for name in VARIANTS[variant]:
        groups[name] = physical_flat[name]
    return groups


def fit_and_forecast(
    standardized: np.ndarray,
    fit_mask: np.ndarray,
    forecast_count: int,
    delay: int,
    rank: int,
) -> dict[str, np.ndarray]:
    fit_states = standardized[fit_mask]
    model = hankel.fit_hankel_dmd(fit_states, delay, rank)
    hankel_prediction = hankel.rollout_hankel(
        model, fit_states, forecast_count
    )
    havok_model, _ = hankel.fit_havok(model, fit_states)
    havok_prediction = hankel.rollout_havok_zero_forcing(
        model, havok_model, fit_states, forecast_count
    )
    return {
        "hankel_dmd": hankel_prediction,
        "havok_zero_forcing": havok_prediction,
    }


def latent_metrics(
    checkpoint: str,
    variant: str,
    method: str,
    truth_scores: np.ndarray,
    prediction_scores: np.ndarray,
    fit_scores: np.ndarray,
) -> dict:
    standardizer = reduced.fit_standardizer(fit_scores)
    truth = standardizer.transform(truth_scores)
    prediction = standardizer.transform(prediction_scores)
    persistence = np.repeat(
        standardizer.transform(fit_scores[-1:]), len(truth), axis=0
    )
    metrics, _ = reduced.evaluate_prediction(
        truth,
        prediction,
        persistence,
        np.arange(len(truth), dtype=np.float64),
    )
    return {
        "checkpoint": checkpoint,
        "variant": variant,
        "method": method,
        **metrics,
    }


def mode_metric_rows(
    checkpoint: str,
    variant: str,
    method: str,
    models: dict[str, block.BlockPCA],
    truth_features: np.ndarray,
    prediction_scores: np.ndarray,
) -> list[dict]:
    prediction = block.decode_blocks(models, prediction_scores, truth_features)
    rows = []
    for band_name, modes in MODE_BANDS.items():
        start, stop = int(modes[0]), int(modes[-1])
        truth_amplitude = block.band_amplitude(truth_features, start, stop)
        pred_amplitude = block.band_amplitude(prediction, start, stop)
        rows.append(
            {
                "checkpoint": checkpoint,
                "variant": variant,
                "method": method,
                "band": band_name,
                "coefficient_nrmse": block.coefficient_nrmse(
                    block.block_slice(truth_features, start, stop),
                    block.block_slice(prediction, start, stop),
                ),
                "amplitude_correlation": correlation(
                    truth_amplitude, pred_amplitude
                ),
                "amplitude_ratio": float(
                    np.mean(pred_amplitude)
                    / max(np.mean(truth_amplitude), 1.0e-30)
                ),
            }
        )
    return rows


def auxiliary_metric_rows(
    checkpoint: str,
    variant: str,
    method: str,
    predicted: dict[str, np.ndarray],
    truth: PhysicalStates,
    fit: PhysicalStates,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    rows: list[dict] = []
    traces: dict[str, np.ndarray] = {}
    if "radial" in predicted:
        radial_prediction = predicted["radial"].reshape(len(truth.time_us), 4, 2)
        for band_index, band in enumerate(MODE_BANDS):
            target = truth.radial[:, :, band_index]
            estimate = radial_prediction[:, :, band_index]
            persistence = np.repeat(
                fit.radial[-1:, :, band_index], len(target), axis=0
            )
            metrics = scalar_metrics(target, estimate, persistence)
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "variant": variant,
                    "method": method,
                    "quantity": "radial_envelope",
                    "band": band,
                    **metrics,
                }
            )

    transport_prediction = None
    if "transport" in predicted:
        transport_prediction = predicted["transport"]
    elif "cross" in predicted:
        cross_prediction = unflatten_cross(predicted["cross"])
        transport_prediction = transport_from_cross(
            cross_prediction, truth.macro_weights
        )
        for band_index, band in enumerate(MODE_BANDS):
            target_cross = truth.cross[:, :, band_index]
            estimate_cross = cross_prediction[:, :, band_index]
            persistence_cross = np.repeat(
                fit.cross[-1:, :, band_index], len(target_cross), axis=0
            )
            metrics = scalar_metrics(
                target_cross, estimate_cross, persistence_cross
            )
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "variant": variant,
                    "method": method,
                    "quantity": "cross_spectrum",
                    "band": band,
                    **metrics,
                    "weighted_phase_mae_rad": weighted_phase_mae(
                        target_cross[:, :, None],
                        estimate_cross[:, :, None],
                        truth.macro_weights,
                    ),
                }
            )

    if transport_prediction is not None:
        for band_index, band in enumerate(MODE_BANDS):
            target = truth.transport[:, band_index]
            estimate = transport_prediction[:, band_index]
            persistence = np.repeat(fit.transport[-1, band_index], len(target))
            metrics = scalar_metrics(target, estimate, persistence)
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "variant": variant,
                    "method": method,
                    "quantity": "modal_transport",
                    "band": band,
                    **metrics,
                }
            )
            traces[f"transport_{band}"] = estimate
    return rows, traces


def subset_physical(states: PhysicalStates, mask: np.ndarray) -> PhysicalStates:
    return PhysicalStates(
        states.time_us[mask],
        states.frame[mask],
        states.radial[mask],
        states.cross[mask],
        states.transport[mask],
        states.macro_weights,
    )


def analyze_checkpoint(
    checkpoint: str,
    specification: dict,
    physical: PhysicalStates,
    physical_flat: dict[str, np.ndarray],
    delays: list[int],
    ranks: list[int],
) -> dict:
    features, time_us, frames = block.load_features(specification["features"])
    if len(time_us) != len(physical.time_us) or not np.allclose(
        time_us, physical.time_us, atol=1.0e-9
    ):
        raise ValueError(f"Time mismatch for {checkpoint}")
    if not np.array_equal(frames, physical.frame):
        raise ValueError(f"Frame mismatch for {checkpoint}")

    subtrain_mask = (time_us >= FIT_START_US) & (
        time_us < VALIDATION_START_US
    )
    fit_mask = (time_us >= FIT_START_US) & (
        time_us < FORECAST_START_US
    )
    forecast_mask = (time_us >= FORECAST_START_US) & (
        time_us <= FORECAST_END_US
    )
    budget_name = specification["budget"]
    budget = block.BUDGETS[budget_name]
    sub_models, sub_scores, _ = block.fit_block_models(
        features, subtrain_mask, budget
    )
    final_models, final_scores, pca_rows = block.fit_block_models(
        features, fit_mask, budget
    )
    del sub_models

    validation_rows = []
    latent_rows = []
    mode_rows = []
    auxiliary_rows = []
    rollout_rows = []
    selections = {}
    for variant in VARIANTS:
        sub_groups = groups_for_variant(sub_scores, physical_flat, variant)
        sub_scaler = GroupScaler.fit(sub_groups, subtrain_mask)
        sub_standardized = sub_scaler.transform(sub_groups)
        selected, candidates = block.search_hankel(
            sub_standardized, time_us, delays, ranks
        )
        selections[variant] = selected
        for row in candidates:
            validation_rows.append(
                {
                    "checkpoint": checkpoint,
                    "budget": budget_name,
                    "variant": variant,
                    "selected": (
                        row["delay"] == selected["delay"]
                        and row["rank"] == selected["rank"]
                    ),
                    **row,
                }
            )

        groups = groups_for_variant(final_scores, physical_flat, variant)
        scaler = GroupScaler.fit(groups, fit_mask)
        standardized = scaler.transform(groups)
        forecasts = fit_and_forecast(
            standardized,
            fit_mask,
            int(np.count_nonzero(forecast_mask)),
            int(selected["delay"]),
            int(selected["rank"]),
        )
        truth_physical = subset_physical(physical, forecast_mask)
        fit_physical = subset_physical(physical, fit_mask)
        for method, forecast in forecasts.items():
            predicted = scaler.inverse(forecast)
            latent_rows.append(
                latent_metrics(
                    checkpoint,
                    variant,
                    method,
                    final_scores[forecast_mask],
                    predicted["latent"],
                    final_scores[fit_mask],
                )
            )
            mode_rows.extend(
                mode_metric_rows(
                    checkpoint,
                    variant,
                    method,
                    final_models,
                    features[forecast_mask],
                    predicted["latent"],
                )
            )
            rows, traces = auxiliary_metric_rows(
                checkpoint,
                variant,
                method,
                predicted,
                truth_physical,
                fit_physical,
            )
            auxiliary_rows.extend(rows)
            for index, value in enumerate(truth_physical.time_us):
                row = {
                    "checkpoint": checkpoint,
                    "variant": variant,
                    "method": method,
                    "time_us": float(value),
                    "frame": int(truth_physical.frame[index]),
                }
                for band_index, band in enumerate(MODE_BANDS):
                    row[f"truth_transport_{band}"] = float(
                        truth_physical.transport[index, band_index]
                    )
                    key = f"transport_{band}"
                    row[f"pred_transport_{band}"] = (
                        float(traces[key][index]) if key in traces else ""
                    )
                rollout_rows.append(row)
        print(
            f"[{checkpoint}] {variant}: q={selected['delay']} "
            f"rank={selected['rank']}",
            flush=True,
        )

    del features
    return {
        "checkpoint": checkpoint,
        "budget": budget_name,
        "components": int(final_scores.shape[1]),
        "pca": pca_rows,
        "selections": selections,
        "validation_rows": validation_rows,
        "latent_rows": latent_rows,
        "mode_rows": mode_rows,
        "auxiliary_rows": auxiliary_rows,
        "rollout_rows": rollout_rows,
    }


def grouped_bar_plot(
    path: Path,
    rows: list[dict],
    quantity: str,
    metric: str,
    variants: list[str],
    title: str,
    ylabel: str,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    methods = ["hankel_dmd", "havok_zero_forcing"]
    colors = {"hankel_dmd": "#0072b2", "havok_zero_forcing": "#009e73"}
    for row_index, checkpoint in enumerate(CHECKPOINTS):
        for column_index, band in enumerate(MODE_BANDS):
            axis = axes[row_index, column_index]
            x = np.arange(len(variants), dtype=np.float64)
            for method_index, method in enumerate(methods):
                values = []
                for variant in variants:
                    matches = [
                        row
                        for row in rows
                        if row.get("checkpoint") == checkpoint
                        and row.get("variant") == variant
                        and row.get("method") == method
                        and row.get("band") == band
                        and (
                            quantity == "mode"
                            or row.get("quantity") == quantity
                        )
                    ]
                    values.append(
                        float(matches[0][metric]) if matches else np.nan
                    )
                axis.bar(
                    x + (method_index - 0.5) * 0.34,
                    values,
                    width=0.34,
                    color=colors[method],
                    label=method,
                )
            axis.axhline(0.0, color="#777777", linewidth=0.8)
            axis.set_title(f"{checkpoint} / {band}")
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", alpha=0.25)
            axis.set_xticks(x)
            axis.set_xticklabels(
                [value.replace("latent_", "") for value in variants],
                rotation=25,
                ha="right",
            )
            axis.set_xlim(-0.7, len(variants) + 1.1)
            axis.legend(loc="lower right")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_transport_rollouts(path: Path, rows: list[dict]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    choices = (
        ("latent_transport", "hankel_dmd", "transport / Hankel", "#0072b2"),
        ("latent_cross", "havok_zero_forcing", "cross / HAVOK", "#009e73"),
    )
    for row_index, checkpoint in enumerate(CHECKPOINTS):
        for column_index, band in enumerate(MODE_BANDS):
            axis = axes[row_index, column_index]
            checkpoint_rows = [
                row for row in rows if row["checkpoint"] == checkpoint
            ]
            reference = next(
                row
                for row in checkpoint_rows
                if row["variant"] == "latent_transport"
                and row["method"] == "hankel_dmd"
            )
            times = np.asarray(
                [
                    row["time_us"]
                    for row in checkpoint_rows
                    if row["variant"] == "latent_transport"
                    and row["method"] == "hankel_dmd"
                ]
            )
            truth = np.asarray(
                [
                    row[f"truth_transport_{band}"]
                    for row in checkpoint_rows
                    if row["variant"] == "latent_transport"
                    and row["method"] == "hankel_dmd"
                ]
            )
            del reference
            axis.plot(times, truth, color="#111111", linewidth=2.0, label="truth")
            for variant, method, label, color in choices:
                selected = [
                    row
                    for row in checkpoint_rows
                    if row["variant"] == variant and row["method"] == method
                ]
                prediction = np.asarray(
                    [row[f"pred_transport_{band}"] for row in selected],
                    dtype=np.float64,
                )
                axis.plot(times, prediction, color=color, linewidth=1.5, label=label)
            axis.set_title(f"{checkpoint} / {band}")
            axis.set_xlabel("time [us]")
            axis.set_ylabel("modal transport [m^-2 s^-1]")
            axis.set_xlim(FORECAST_START_US, FORECAST_END_US + 1.4)
            axis.grid(alpha=0.25)
            axis.legend(loc="lower right")
    figure.suptitle("Autonomous modal-transport rollouts (24--30 us)")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def best_rows(rows: list[dict], quantity: str, band: str) -> list[dict]:
    selected = [
        row
        for row in rows
        if row.get("quantity") == quantity and row.get("band") == band
    ]
    result = []
    for checkpoint in CHECKPOINTS:
        candidates = [row for row in selected if row["checkpoint"] == checkpoint]
        finite = [row for row in candidates if np.isfinite(row["correlation"])]
        if finite:
            result.append(max(finite, key=lambda row: row["correlation"]))
    return result


def write_readme(path: Path, summaries: list[dict], auxiliary_rows: list[dict]) -> None:
    lines = [
        "# E25 augmented physical-state dynamics comparison",
        "",
        "## 目的",
        "",
        "25 kV/mケースの未プール潜在Fourier状態に、少数の物理状態を追加したとき、24--30 usの自律予測が閉じるかを比較する。data-onlyとspectral-lossモデルを同じ手順で評価する。",
        "",
        "## 追加状態",
        "",
        "- `radial envelope`: radial 8帯域を4帯域にまとめ、MTSI/ECDI候補帯ごとのphi RMS振幅を保持する8変数。",
        "- `complex cross-spectrum`: 各radial帯域・モード帯の `n_e_hat * conj(E_y_hat)` の実部と虚部を保持する16変数。実部はmodal transport、偏角はcross-phaseに対応する。",
        "- `modal transport`: cross-spectrum実部をradial・mode方向に積分したMTSI/ECDIの2変数。",
        "- cross-spectrumとtransportは代数的に重複するため、同じvariantには同時投入していない。",
        "",
        "## 実験条件",
        "",
        "- 同定: 20--24 us、内部選択: 20--23 usで学習して23--24 usでdelay/rankを選択。",
        "- 自律予測: 24--30 us。予測区間の真値は入力しない。",
        "- data-only: 固定20次元潜在状態。spectral-loss: 固定32次元潜在状態。",
        "- 補助群は群ごとに標準化し、`1/sqrt(group dimension)`で重み付けした。",
        "- 各variantでHankel DMDとzero-forcing HAVOKを比較した。",
        "",
        "## 最良のmodal transport相関",
        "",
        "| checkpoint | band | variant | method | correlation | NRMSE | skill vs persistence |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for band in MODE_BANDS:
        for row in best_rows(auxiliary_rows, "modal_transport", band):
            lines.append(
                f"| {row['checkpoint']} | {band} | {row['variant']} | {row['method']} | "
                f"{row['correlation']:.4f} | {row['nrmse']:.4f} | {row['skill_vs_persistence']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## ファイル",
            "",
            "- `validation_candidates.csv`: delay/rankの内部検証。",
            "- `latent_state_metrics.csv`: 潜在状態そのものの自律予測指標。",
            "- `latent_mode_metrics.csv`: MTSI/ECDI候補帯の潜在振幅・係数指標。",
            "- `auxiliary_state_metrics.csv`: radial envelope、cross-spectrum、transportの指標。",
            "- `transport_rollouts.csv`: modal transport真値と自律予測時系列。",
            "- `latent_mode_amplitude_correlation.png`: 補助状態による潜在モード振幅再現の変化。",
            "- `transport_correlation.png`: modal transport自律予測相関。",
            "- `radial_envelope_correlation.png`: radial envelope自律予測相関。",
            "- `cross_phase_error.png`: cross-spectrumの重み付き位相誤差。",
            "- `transport_rollout_comparison.png`: 代表的なtransport/cross状態による24--30 us時系列。",
            "- `summary.json`: 設定と選択結果。",
            "",
            "## 注意",
            "",
            "補助状態を入力するとは、予測開始時までの履歴を状態同定に使うという意味であり、24 us以降の真値を逐次与えるteacher forcingではない。したがって、ここでの改善は少数状態を含む閉じた自律力学系に近づいたかを示す。",
            "radial/cross-spectrumの複数成分指標では、帯域間の静的な差に引っ張られる通常相関に加え、各成分の時間平均を引いた`temporal_anomaly_correlation`を主な時間変動指標としている。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical", type=Path, default=DEFAULT_PHYSICAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--delays", default="20,40,60,80,100")
    parser.add_argument("--ranks", default="8,12,20,30,40")
    args = parser.parse_args()
    delays = [int(value) for value in args.delays.split(",")]
    ranks = [int(value) for value in args.ranks.split(",")]
    args.output.mkdir(parents=True, exist_ok=True)

    physical = load_physical_states(args.physical)
    physical_flat = flatten_physical(physical)
    summaries = []
    for checkpoint, specification in CHECKPOINTS.items():
        summaries.append(
            analyze_checkpoint(
                checkpoint,
                specification,
                physical,
                physical_flat,
                delays,
                ranks,
            )
        )

    validation_rows = sum(
        [summary["validation_rows"] for summary in summaries], []
    )
    latent_rows = sum([summary["latent_rows"] for summary in summaries], [])
    mode_rows = sum([summary["mode_rows"] for summary in summaries], [])
    auxiliary_rows = sum(
        [summary["auxiliary_rows"] for summary in summaries], []
    )
    rollout_rows = sum(
        [summary["rollout_rows"] for summary in summaries], []
    )
    write_csv(args.output / "validation_candidates.csv", validation_rows)
    write_csv(args.output / "latent_state_metrics.csv", latent_rows)
    write_csv(args.output / "latent_mode_metrics.csv", mode_rows)
    write_csv(args.output / "auxiliary_state_metrics.csv", auxiliary_rows)
    write_csv(args.output / "transport_rollouts.csv", rollout_rows)

    all_variants = list(VARIANTS)
    grouped_bar_plot(
        args.output / "latent_mode_amplitude_correlation.png",
        mode_rows,
        "mode",
        "amplitude_correlation",
        all_variants,
        "Latent mode-amplitude closure after adding physical states",
        "amplitude correlation",
    )
    grouped_bar_plot(
        args.output / "transport_correlation.png",
        auxiliary_rows,
        "modal_transport",
        "correlation",
        [
            "latent_transport",
            "latent_cross",
            "latent_radial_transport",
            "latent_radial_cross",
        ],
        "Autonomous modal-transport prediction",
        "correlation",
    )
    grouped_bar_plot(
        args.output / "radial_envelope_correlation.png",
        auxiliary_rows,
        "radial_envelope",
        "temporal_anomaly_correlation",
        [
            "latent_radial",
            "latent_radial_transport",
            "latent_radial_cross",
        ],
        "Autonomous radial-envelope prediction",
        "correlation",
    )
    grouped_bar_plot(
        args.output / "cross_phase_error.png",
        auxiliary_rows,
        "cross_spectrum",
        "weighted_phase_mae_rad",
        ["latent_cross", "latent_radial_cross"],
        "Cross-phase reconstruction in autonomous rollout",
        "weighted phase MAE [rad]",
    )
    plot_transport_rollouts(
        args.output / "transport_rollout_comparison.png", rollout_rows
    )

    summary = {
        "physical_source": str(args.physical),
        "fit_interval_us": [FIT_START_US, FORECAST_START_US],
        "validation_interval_us": [VALIDATION_START_US, FORECAST_START_US],
        "forecast_interval_us": [FORECAST_START_US, FORECAST_END_US],
        "delays": delays,
        "ranks": ranks,
        "variants": VARIANTS,
        "mode_bands": {key: value.tolist() for key, value in MODE_BANDS.items()},
        "group_scaling": "per-component z-score, then 1/sqrt(group dimension)",
        "checkpoints": [
            {
                key: value
                for key, value in item.items()
                if key
                not in {
                    "validation_rows",
                    "latent_rows",
                    "mode_rows",
                    "auxiliary_rows",
                    "rollout_rows",
                }
            }
            for item in summaries
        ],
    }
    (args.output / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2), encoding="utf-8"
    )
    write_readme(args.output / "README.md", summaries, auxiliary_rows)
    print(f"Saved augmented-state comparison to {args.output}", flush=True)


if __name__ == "__main__":
    main()
