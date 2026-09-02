"""Strictly transfer the fixed E25 ROM to E30 without target fitting.

The source representation, scaler, delay/rank and Hankel operator are fitted
only on E25 over 12--20 us.  E30 contributes only the final 80 observed
states before 20 us to initialize the delay vector.  E30 forecast truth is
used exclusively after rollout for evaluation.
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

import analyze_radaz_augmented_physical_state_dynamics as augmented
import analyze_radaz_blockwise_fourier_latent_dynamics as block
import analyze_radaz_e25_carrier_state_closure as carrier
import analyze_radaz_e25_transport_residual_closure as closure
import analyze_radaz_hankel_havok as hankel
import analyze_radaz_reduced_dynamics as reduced


ROOT = Path(__file__).resolve().parent
SOURCE_FEATURES = (
    ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_ez25kvm_fourier_latent_raw"
    / "fourier_latent_features.h5"
)
SOURCE_PHYSICAL = carrier.DEFAULT_PHYSICAL
TARGET_FEATURES = (
    ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_electric_field_sweep_forcing"
    / "cases"
    / "E30kVm"
    / "fourier_latent_features.h5"
)
TARGET_PHYSICAL = (
    ROOT
    / "workdirs"
    / "compare_radaz_rom_transfer_e25_e30_e40"
    / "physical_cache"
    / "E30_physical_fourier_targets.h5"
)
DEFAULT_OUTPUT = (
    ROOT / "workdirs" / "compare_radaz_e25_fixed_rom_zero_shot_e30"
)

FIT_START_US = 12.0
FORECAST_START_US = 20.0
FORECAST_END_US = 30.0
DELAY = 80
RANK = 40
SYSTEM = "latent_phi_transport"
SYSTEM_LABEL = "L+Pcirc+T"
SEGMENTS = {
    "full20_30": (20.0, 30.0),
    "early20_24": (20.0, 24.0),
    "late24_30_no_reset": (24.0, 30.0),
}


@dataclass
class CaseRepresentation:
    name: str
    raw: carrier.RawPhysical
    features: np.ndarray
    latent: np.ndarray
    phi_circular: np.ndarray
    selected_phi: np.ndarray
    transport: np.ndarray
    transport_state: np.ndarray

    @property
    def groups(self) -> dict[str, np.ndarray]:
        return {
            "latent": self.latent,
            "phi_circular": self.phi_circular,
            "transport_direct": self.transport_state,
        }


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
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


def sha256_array(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def feature_metadata(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        return {
            "path": str(path),
            "checkpoint": str(handle.attrs["checkpoint"]),
            "source_workdir": str(handle.attrs["source_workdir"]),
            "config": str(handle.attrs["config"]),
            "full_latent_shape": np.asarray(
                handle["full_latent_shape"], dtype=int
            ).tolist(),
            "radial_boundaries": np.asarray(
                handle["radial_boundaries"], dtype=int
            ).tolist(),
            "azimuthal_modes": np.asarray(
                handle["azimuthal_modes"], dtype=int
            ).tolist(),
            "source_data": str(handle.attrs["source_data"]),
        }


def compatible_feature_maps(source: dict, target: dict) -> bool:
    source_checkpoint = Path(source["checkpoint"]).name
    target_checkpoint = Path(target["checkpoint"]).name
    source_workdir = Path(source["source_workdir"]).name
    target_workdir = Path(target["source_workdir"]).name
    return bool(
        source_checkpoint == target_checkpoint
        and source_workdir == target_workdir
        and source["full_latent_shape"] == target["full_latent_shape"]
        and source["radial_boundaries"] == target["radial_boundaries"]
        and source["azimuthal_modes"] == target["azimuthal_modes"]
    )


def source_block_scores(
    source: carrier.CarrierBlock,
    values: np.ndarray,
    radial_weights: np.ndarray,
) -> np.ndarray:
    if not np.allclose(
        radial_weights, source.radial_weights, atol=1.0e-12, rtol=1.0e-9
    ):
        raise ValueError("Source and target radial weights do not match")
    selected = values[:, :, source.modes]
    result = np.empty(
        (len(values), len(source.modes), source.bases.shape[1]),
        dtype=np.complex128,
    )
    sqrt_weights = np.sqrt(radial_weights)
    for mode_index in range(len(source.modes)):
        weighted = selected[:, :, mode_index] * sqrt_weights[None]
        result[:, mode_index] = weighted @ source.bases[mode_index].conj().T
    return result


def transform_with_source_carrier(
    source: carrier.CarrierBlock,
    values: np.ndarray,
    frames: np.ndarray,
    radial_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Express a target field in the unchanged E25 carrier coordinates."""
    scores = source_block_scores(source, values, radial_weights)
    normalized = scores / source.scales[None]
    envelope = normalized * np.exp(
        -1j * frames[:, None, None] * source.carrier_angles[None]
    )
    amplitude = np.maximum(np.abs(normalized), 1.0e-10)
    log_amplitude = np.log(amplitude)
    phase = np.unwrap(np.angle(envelope), axis=0)
    phase -= source.phase_references[None]
    growth = carrier.causal_difference(log_amplitude, carrier.FRAME_DT_US)
    phase_rate = carrier.causal_difference(phase, carrier.FRAME_DT_US)
    circular = np.empty(
        (len(values), len(source.modes), source.bases.shape[1], 5),
        dtype=np.float64,
    )
    circular[..., 0] = log_amplitude
    circular[..., 1] = np.cos(phase)
    circular[..., 2] = np.sin(phase)
    circular[..., 3] = growth
    for mode_index, mode in enumerate(source.modes):
        ky = 2.0 * np.pi * float(mode) / carrier.DOMAIN_LENGTH_Y_M
        total_rate = (
            source.carrier_angles[mode_index][None] / carrier.FRAME_DT_US
            + phase_rate[:, mode_index]
        )
        circular[:, mode_index, :, 4] = -(total_rate * 1.0e6) / ky
    return (
        circular.reshape(len(values), -1),
        values[:, :, source.modes],
        normalized,
    )


def make_transport_state(
    raw: carrier.RawPhysical, modes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    selected_cross = raw.cross[:, :, modes]
    transport = carrier.transport_from_selected_cross(
        selected_cross, raw.radial_weights
    )
    rate = carrier.causal_difference(
        transport[:, None], carrier.FRAME_DT_US
    )[:, 0]
    return transport, np.column_stack([transport, rate])


def fit_source_model(
    source_raw: carrier.RawPhysical,
    source_features: np.ndarray,
) -> tuple[
    dict[str, block.BlockPCA],
    carrier.CarrierBlock,
    augmented.GroupScaler,
    hankel.HankelModel,
    CaseRepresentation,
    list[dict],
]:
    fit_mask = (source_raw.time_us >= FIT_START_US) & (
        source_raw.time_us < FORECAST_START_US
    )
    modes = carrier.select_modes(
        source_raw.phi, source_raw.radial_weights, fit_mask
    )
    phi_block = carrier.build_carrier_block(
        "phi",
        source_raw.phi,
        modes,
        source_raw.radial_weights,
        source_raw.frame,
        fit_mask,
    )
    latent_models, latent, pca_rows = block.fit_block_models(
        source_features, fit_mask, block.BUDGETS["medium_20"]
    )
    transport, transport_state = make_transport_state(source_raw, modes)
    source = CaseRepresentation(
        "E25",
        source_raw,
        source_features,
        latent,
        phi_block.circular,
        source_raw.phi[:, :, modes],
        transport,
        transport_state,
    )
    scaler = augmented.GroupScaler.fit(source.groups, fit_mask)
    standardized = scaler.transform(source.groups)
    fit_states = standardized[fit_mask]
    model = hankel.fit_hankel_dmd(fit_states, DELAY, RANK)
    return latent_models, phi_block, scaler, model, source, pca_rows


def transform_target(
    raw: carrier.RawPhysical,
    features: np.ndarray,
    latent_models: dict[str, block.BlockPCA],
    phi_block: carrier.CarrierBlock,
) -> CaseRepresentation:
    latent = np.concatenate(
        [latent_models[name].transform(features) for name in block.BLOCKS],
        axis=1,
    )
    phi_circular, selected_phi, _ = transform_with_source_carrier(
        phi_block, raw.phi, raw.frame, raw.radial_weights
    )
    transport, transport_state = make_transport_state(raw, phi_block.modes)
    return CaseRepresentation(
        "E30",
        raw,
        features,
        latent,
        phi_circular,
        selected_phi,
        transport,
        transport_state,
    )


def target_carrier_baseline(
    source: carrier.CarrierBlock,
    target_raw: carrier.RawPhysical,
    history_last_index: int,
    forecast_frames: np.ndarray,
) -> np.ndarray:
    scores = source_block_scores(
        source, target_raw.phi, target_raw.radial_weights
    )
    normalized_last = scores[history_last_index] / source.scales
    delta_frame = forecast_frames - target_raw.frame[history_last_index]
    propagated = normalized_last[None] * np.exp(
        1j * delta_frame[:, None, None] * source.carrier_angles[None]
    )
    return source._scores_to_coefficients(propagated * source.scales[None])


def safe_skill(truth, prediction, baseline) -> float:
    if not np.all(np.isfinite(prediction)):
        return float("nan")
    mse = float(np.mean(np.abs(prediction - truth) ** 2))
    baseline_mse = float(np.mean(np.abs(baseline - truth) ** 2))
    if baseline_mse <= np.finfo(float).tiny:
        return float("nan")
    return 1.0 - mse / baseline_mse


def scalar_summary(
    truth: np.ndarray,
    prediction: np.ndarray,
    persistence: np.ndarray,
    history_mean: np.ndarray,
) -> dict[str, float]:
    truth_std = float(np.std(truth, ddof=1))
    return {
        "correlation": augmented.correlation(truth, prediction),
        "nrmse": augmented.nrmse(truth, prediction),
        "skill_vs_persistence": safe_skill(truth, prediction, persistence),
        "skill_vs_initial_history_mean": safe_skill(
            truth, prediction, history_mean
        ),
        "prediction_std_over_truth_std": float(
            np.std(prediction, ddof=1)
            / max(truth_std, np.finfo(float).tiny)
        ),
        "normalized_bias": float(
            np.mean(prediction - truth)
            / max(truth_std, np.finfo(float).tiny)
        ),
    }


def initial_history_ood_rows(
    source: CaseRepresentation,
    target: CaseRepresentation,
    scaler: augmented.GroupScaler,
) -> list[dict]:
    source_fit = (source.raw.time_us >= FIT_START_US) & (
        source.raw.time_us < FORECAST_START_US
    )
    target_history = np.flatnonzero(
        target.raw.time_us < FORECAST_START_US
    )[-DELAY:]
    rows = []
    for name in scaler.names:
        source_z = (
            source.groups[name][source_fit] - scaler.means[name]
        ) / scaler.scales[name]
        target_z = (
            target.groups[name][target_history] - scaler.means[name]
        ) / scaler.scales[name]
        rows.append(
            {
                "group": name,
                "dimensions": int(target_z.shape[1]),
                "source_fit_rms_z": float(np.sqrt(np.mean(source_z**2))),
                "target_initial_history_rms_z": float(
                    np.sqrt(np.mean(target_z**2))
                ),
                "target_initial_history_mean_abs_z": float(
                    np.mean(np.abs(target_z))
                ),
                "target_initial_history_fraction_abs_z_gt_3": float(
                    np.mean(np.abs(target_z) > 3.0)
                ),
                "target_initial_history_max_abs_z": float(
                    np.max(np.abs(target_z))
                ),
            }
        )
    return rows


def evaluate_case(
    case: CaseRepresentation,
    scaler: augmented.GroupScaler,
    model: hankel.HankelModel,
    phi_block: carrier.CarrierBlock,
) -> tuple[list[dict], list[dict], dict]:
    standardized = scaler.transform(case.groups)
    history_mask = case.raw.time_us < FORECAST_START_US
    forecast_mask = (case.raw.time_us >= FORECAST_START_US) & (
        case.raw.time_us <= FORECAST_END_US
    )
    history_indices = np.flatnonzero(history_mask)
    forecast_indices = np.flatnonzero(forecast_mask)
    history = standardized[history_indices[-DELAY:]]
    prediction_standardized = hankel.rollout_hankel(
        model, history, len(forecast_indices)
    )
    prediction_groups = scaler.inverse(prediction_standardized)

    perturbed = standardized.copy()
    perturbed[forecast_mask] = 12345.0
    repeated = hankel.rollout_hankel(
        model, perturbed[history_indices[-DELAY:]], len(forecast_indices)
    )
    rollout_difference = float(
        np.nanmax(np.abs(prediction_standardized - repeated))
    )

    truth_state = standardized[forecast_mask]
    persistence_state = np.repeat(history[-1:], len(truth_state), axis=0)
    state_metrics, _ = reduced.evaluate_prediction(
        truth_state,
        prediction_standardized,
        persistence_state,
        case.raw.time_us[forecast_mask],
    )

    transport_truth = case.transport[forecast_mask]
    transport_prediction = prediction_groups["transport_direct"][:, 0]
    transport_persistence = np.repeat(
        case.transport[history_indices[-1]], len(transport_truth)
    )
    initial_history_mean = float(
        np.mean(case.transport[history_indices[-DELAY:]])
    )
    transport_history_mean = np.repeat(
        initial_history_mean, len(transport_truth)
    )

    forecast_frames = case.raw.frame[forecast_mask]
    phi_prediction = phi_block.decode_circular(
        prediction_groups["phi_circular"], forecast_frames
    )
    phi_truth = case.selected_phi[forecast_mask]
    phi_persistence = np.repeat(
        case.selected_phi[history_indices[-1] : history_indices[-1] + 1],
        len(phi_truth),
        axis=0,
    )
    phi_carrier = target_carrier_baseline(
        phi_block, case.raw, history_indices[-1], forecast_frames
    )
    phi_envelope_truth = carrier.phi_envelope(
        phi_truth, case.raw.radial_weights
    )
    phi_envelope_prediction = carrier.phi_envelope(
        phi_prediction, case.raw.radial_weights
    )
    phi_envelope_persistence = np.repeat(
        phi_envelope_truth[:1] * 0.0
        + carrier.phi_envelope(
            case.selected_phi[history_indices[-1] : history_indices[-1] + 1],
            case.raw.radial_weights,
        ),
        len(phi_envelope_truth),
    )

    metric_rows: list[dict] = []
    time_rows: list[dict] = []
    forecast_time = case.raw.time_us[forecast_mask]
    for segment, (start, end) in SEGMENTS.items():
        local = (forecast_time >= start) & (forecast_time <= end)
        transport_metrics = scalar_summary(
            transport_truth[local],
            transport_prediction[local],
            transport_persistence[local],
            transport_history_mean[local],
        )
        metric_rows.append(
            {
                "case": case.name,
                "segment": segment,
                "quantity": "selected_modal_transport",
                **transport_metrics,
            }
        )
        envelope_metrics = scalar_summary(
            phi_envelope_truth[local],
            phi_envelope_prediction[local],
            phi_envelope_persistence[local],
            np.repeat(
                float(
                    np.mean(
                        carrier.phi_envelope(
                            case.selected_phi[history_indices[-DELAY:]],
                            case.raw.radial_weights,
                        )
                    )
                ),
                np.count_nonzero(local),
            ),
        )
        metric_rows.append(
            {
                "case": case.name,
                "segment": segment,
                "quantity": "phi_envelope",
                **envelope_metrics,
            }
        )
        coefficient = carrier.coefficient_metrics(
            phi_truth[local],
            phi_prediction[local],
            phi_persistence[local],
            phi_carrier[local],
            case.raw.radial_weights,
        )
        metric_rows.append(
            {
                "case": case.name,
                "segment": segment,
                "quantity": "phi_coefficients",
                **coefficient,
            }
        )

    for index, time_us in enumerate(forecast_time):
        time_rows.append(
            {
                "case": case.name,
                "time_us": float(time_us),
                "transport_truth": float(transport_truth[index]),
                "transport_prediction": float(transport_prediction[index]),
                "transport_persistence": float(transport_persistence[index]),
                "transport_initial_history_mean": initial_history_mean,
                "phi_envelope_truth": float(phi_envelope_truth[index]),
                "phi_envelope_prediction": float(
                    phi_envelope_prediction[index]
                ),
            }
        )
    audit = {
        "case": case.name,
        "history_frames": int(len(history)),
        "history_start_us": float(case.raw.time_us[history_indices[-DELAY]]),
        "history_end_us": float(case.raw.time_us[history_indices[-1]]),
        "forecast_start_us": float(forecast_time[0]),
        "forecast_end_us": float(forecast_time[-1]),
        "forecast_frames": int(len(forecast_time)),
        "initial_history_sha256": sha256_array(history),
        "future_truth_perturbation_rollout_max_difference": rollout_difference,
        "future_truth_unused": bool(rollout_difference == 0.0),
        "state_metrics": state_metrics,
    }
    return metric_rows, time_rows, audit


def plot_result(path: Path, rows: list[dict]) -> None:
    by_case = {
        name: [row for row in rows if row["case"] == name]
        for name in ("E25", "E30")
    }
    figure, axes = plt.subplots(
        2, 2, figsize=(15.5, 9.0), constrained_layout=True
    )
    colors = {"truth": "#111111", "prediction": "#0072b2"}
    for column, case in enumerate(("E25", "E30")):
        values = by_case[case]
        time = np.asarray([row["time_us"] for row in values])
        axes[0, column].plot(
            time,
            [row["transport_truth"] for row in values],
            color=colors["truth"],
            linewidth=1.5,
            label="PIC truth",
        )
        axes[0, column].plot(
            time,
            [row["transport_prediction"] for row in values],
            color=colors["prediction"],
            linewidth=1.3,
            label="fixed E25 ROM",
        )
        axes[0, column].plot(
            time,
            [row["transport_persistence"] for row in values],
            color="#cc79a7",
            linestyle="--",
            linewidth=1.0,
            label="target persistence",
        )
        axes[0, column].set_title(
            "E25 source replay" if case == "E25" else "E30 strict zero-shot"
        )
        axes[0, column].set_ylabel("selected-mode transport")
        axes[0, column].grid(alpha=0.25)
        axes[0, column].legend(loc="lower right", fontsize=8)

        axes[1, column].plot(
            time,
            [row["phi_envelope_truth"] for row in values],
            color=colors["truth"],
            linewidth=1.5,
            label="PIC truth",
        )
        axes[1, column].plot(
            time,
            [row["phi_envelope_prediction"] for row in values],
            color=colors["prediction"],
            linewidth=1.3,
            label="fixed E25 ROM",
        )
        axes[1, column].set_xlabel("physical time [us]")
        axes[1, column].set_ylabel("selected phi envelope")
        axes[1, column].grid(alpha=0.25)
        axes[1, column].legend(loc="lower right", fontsize=8)
    figure.suptitle(
        "Fixed E25 L+Pcirc+T ROM: source replay and strict E30 zero-shot"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_ood(path: Path, rows: list[dict]) -> None:
    labels = [
        {"latent": "L", "phi_circular": "Pcirc", "transport_direct": "T"}[
            row["group"]
        ]
        for row in rows
    ]
    values = [row["target_initial_history_rms_z"] for row in rows]
    figure, axis = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    bars = axis.bar(labels, values, color=["#56b4e9", "#d55e00", "#009e73"])
    axis.axhline(1.0, color="#333333", linestyle="--", linewidth=1.1)
    axis.set_yscale("log")
    axis.set_ylabel("E30 initial-history RMS in E25 standard deviations")
    axis.set_title("Strict zero-shot initialization: group-wise distribution shift")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value * 1.08,
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_readme(
    path: Path,
    metric_rows: list[dict],
    audits: list[dict],
    protocol: dict,
    ood_rows: list[dict],
) -> None:
    def find(case: str, quantity: str) -> dict:
        return next(
            row
            for row in metric_rows
            if row["case"] == case
            and row["segment"] == "full20_30"
            and row["quantity"] == quantity
        )

    e25_t = find("E25", "selected_modal_transport")
    e30_t = find("E30", "selected_modal_transport")
    e25_p = find("E25", "phi_coefficients")
    e30_p = find("E30", "phi_coefficients")
    lines = [
        "# Fixed E25 ROM strict zero-shot to E30",
        "",
        "E25で固定した`L+Pcirc+T / Hankel DMD`をE30へ無調整で適用した。",
        "PCA、carrier、scaler、delay/rank、operatorはE25の12--20 usだけで固定した。",
        "E30から利用したのは20 us直前の80フレーム（1.2 us）の初期Hankel履歴だけである。",
        "E30の平均・分散合わせ、係数補正、再fit、fine-tuningは行っていない。",
        "",
        "## Protocol",
        "",
        f"- State: `{SYSTEM_LABEL}` ({protocol['state_dimension']} dimensions)",
        f"- Selected source modes: `{protocol['selected_modes']}`",
        f"- Fixed delay/rank: `{DELAY}` / `{RANK}`",
        "- Source fit: E25, 12--20 us",
        "- Forecast: 20--30 us, one uninterrupted rollout",
        "- Target initialization: E30 history only; no target fitting",
        "- Both latent H5 files use the same frozen E10 SimVP checkpoint and the same case-local train-only min-max definition.",
        "",
        "## Full 20--30 us result",
        "",
        "| case | transport corr | transport skill vs persistence | transport skill vs initial-history mean | phi coefficient corr | phi phase MAE [rad] |",
        "|---|---:|---:|---:|---:|---:|",
        f"| E25 source replay | {e25_t['correlation']:.3f} | {e25_t['skill_vs_persistence']:.3f} | {e25_t['skill_vs_initial_history_mean']:.3f} | {e25_p['coefficient_correlation']:.3f} | {e25_p['weighted_phase_mae_rad']:.3f} |",
        f"| E30 strict zero-shot | {e30_t['correlation']:.3f} | {e30_t['skill_vs_persistence']:.3f} | {e30_t['skill_vs_initial_history_mean']:.3f} | {e30_p['coefficient_correlation']:.3f} | {e30_p['weighted_phase_mae_rad']:.3f} |",
        "",
        "## Initial-history distribution shift",
        "",
        "E30の初期履歴をE25 source scalerで測った。1前後ならsource内と同程度、3を大きく超えるほどsourceから外れている。",
        "",
        "| state group | E30 RMS z | fraction abs(z)>3 | max abs(z) |",
        "|---|---:|---:|---:|",
    ]
    for row in ood_rows:
        lines.append(
            f"| {row['group']} | {row['target_initial_history_rms_z']:.3f} | "
            f"{row['target_initial_history_fraction_abs_z_gt_3']:.3f} | "
            f"{row['target_initial_history_max_abs_z']:.3f} |"
        )
    lines.extend(
        [
        "",
        "The latent image features remain near the E25 range, while Pcirc and T are strongly out of distribution. The fixed operator therefore reproduces an E25-like attractor scale instead of E30 dynamics; this is a representation/dynamics transfer failure, not numerical divergence.",
        "",
        "## Leakage audit",
        "",
        ]
    )
    for audit in audits:
        lines.append(
            f"- {audit['case']}: target future perturbation changed the rollout by "
            f"`{audit['future_truth_perturbation_rollout_max_difference']:.3e}`."
        )
    lines.extend(
        [
            "",
            "This is strict at the ROM-operator level, but E30 is not a pristine never-inspected scientific holdout because it was used in earlier analyses. No E30 values were used to choose or fit this fixed ROM in this script.",
            "",
            "## Files",
            "",
            "- `zero_shot_metrics.csv`: segment-wise transport and phi metrics.",
            "- `zero_shot_time_series.csv`: E25 replay and E30 zero-shot traces.",
            "- `protocol_and_audit.json`: frozen protocol, provenance and leakage audit.",
            "- `e25_fixed_rom_zero_shot_e30.png`: direct visual comparison.",
            "- `initial_history_ood.csv` and `initial_history_ood.png`: group-wise target distribution shift measured in the fixed E25 coordinates.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-features", type=Path, default=SOURCE_FEATURES)
    parser.add_argument("--source-physical", type=Path, default=SOURCE_PHYSICAL)
    parser.add_argument("--target-features", type=Path, default=TARGET_FEATURES)
    parser.add_argument("--target-physical", type=Path, default=TARGET_PHYSICAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    source_meta = feature_metadata(args.source_features)
    target_meta = feature_metadata(args.target_features)
    if not compatible_feature_maps(source_meta, target_meta):
        raise ValueError("E25 and E30 latent feature maps are incompatible")

    source_raw = carrier.load_raw_physical(args.source_physical)
    target_raw = carrier.load_raw_physical(args.target_physical)
    source_features, source_time, source_frames = block.load_features(
        args.source_features
    )
    target_features, target_time, target_frames = block.load_features(
        args.target_features
    )
    if not np.allclose(source_raw.time_us, source_time, atol=1.0e-9):
        raise ValueError("E25 physical/latent times differ")
    if not np.allclose(target_raw.time_us, target_time, atol=1.0e-9):
        raise ValueError("E30 physical/latent times differ")
    if not np.array_equal(source_raw.frame, source_frames):
        raise ValueError("E25 physical/latent frames differ")
    if not np.array_equal(target_raw.frame, target_frames):
        raise ValueError("E30 physical/latent frames differ")

    (
        latent_models,
        phi_block,
        scaler,
        model,
        source,
        pca_rows,
    ) = fit_source_model(source_raw, source_features)
    target = transform_target(
        target_raw, target_features, latent_models, phi_block
    )
    ood_rows = initial_history_ood_rows(source, target, scaler)

    all_metrics: list[dict] = []
    all_times: list[dict] = []
    audits: list[dict] = []
    for case in (source, target):
        metrics, times, audit = evaluate_case(case, scaler, model, phi_block)
        all_metrics.extend(metrics)
        all_times.extend(times)
        audits.append(audit)
        transport = next(
            row
            for row in metrics
            if row["segment"] == "full20_30"
            and row["quantity"] == "selected_modal_transport"
        )
        print(
            f"[{case.name}] transport corr={transport['correlation']:.4f} "
            f"skill_persist={transport['skill_vs_persistence']:.4f} "
            f"skill_history={transport['skill_vs_initial_history_mean']:.4f}",
            flush=True,
        )

    protocol = {
        "source_case": "E25",
        "target_case": "E30",
        "source_fit_interval_us": [FIT_START_US, FORECAST_START_US],
        "forecast_interval_us": [FORECAST_START_US, FORECAST_END_US],
        "system": SYSTEM,
        "system_label": SYSTEM_LABEL,
        "state_dimension": int(sum(v.shape[1] for v in source.groups.values())),
        "selected_modes": phi_block.modes.tolist(),
        "delay": DELAY,
        "rank": RANK,
        "history_us": DELAY * carrier.FRAME_DT_US,
        "target_fitting": False,
        "target_scaling_or_affine_calibration": False,
        "target_history_only_for_initialization": True,
        "source_feature_metadata": source_meta,
        "target_feature_metadata": target_meta,
        "feature_maps_compatible": True,
        "source_pca": pca_rows,
        "model_spectral_radius": float(np.max(np.abs(model.eigenvalues))),
        "model_matrix_sha256": sha256_array(model.matrix),
        "audits": audits,
    }
    write_csv(args.output / "zero_shot_metrics.csv", all_metrics)
    write_csv(args.output / "zero_shot_time_series.csv", all_times)
    write_csv(args.output / "initial_history_ood.csv", ood_rows)
    (args.output / "protocol_and_audit.json").write_text(
        json.dumps(json_safe(protocol), indent=2), encoding="utf-8"
    )
    plot_result(args.output / "e25_fixed_rom_zero_shot_e30.png", all_times)
    plot_ood(args.output / "initial_history_ood.png", ood_rows)
    write_readme(
        args.output / "README.md", all_metrics, audits, protocol, ood_rows
    )
    print(f"[DONE] {args.output}", flush=True)


if __name__ == "__main__":
    main()
