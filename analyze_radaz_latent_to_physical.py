#!/usr/bin/env python3
"""Decode autonomous RadAz latent rollouts and evaluate physical observables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
CASE = (
    ROOT.parent
    / "PEPAPIC"
    / "test"
    / "results"
    / "2D_Landmark"
    / "2D_RadAz_Xe1p_Bx20mT_Ez25kVm_dt15ps_out15ns"
    / "2D_RadAz_Xe1p_Bx20mT_Ez25kVm_dt15ps_out15ns"
)
DATA_H5 = (
    CASE
    / "SimVPv2_inputs"
    / "radaz_3ch_targetnorm_trainfixed_margin20_native257x256_pad260x256.h5"
)
LATENT_DIR = (
    ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_ez25kvm_latent_e10frozen_targetnorm"
)
HANKEL_DIR = (
    ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_ez25kvm_hankel_havok_e10frozen_targetnorm_extended"
)
OUTPUT = (
    ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_ez25kvm_latent_to_physical"
)

CHANNELS = ("electron_den", "ion_den", "phi")
METHOD_COLUMNS = {
    "oracle": "truth",
    "latent_persistence": "persistence",
    "standard_dmd": "standard_dmd",
    "hankel_dmd": "hankel_dmd",
    "havok": "havok_zero_forcing",
}
PLOT_METHODS = ("oracle", "hankel_dmd", "havok", "physical_copy")
METHOD_LABELS = {
    "oracle": "Oracle reconstruction",
    "latent_persistence": "Latent persistence",
    "standard_dmd": "Standard DMD",
    "hankel_dmd": "Hankel DMD",
    "havok": "HAVOK",
    "physical_copy": "Physical copy",
}
COLORS = {
    "truth": "#111111",
    "oracle": "#0072b2",
    "latent_persistence": "#8c8c8c",
    "standard_dmd": "#cc79a7",
    "hankel_dmd": "#009e73",
    "havok": "#d55e00",
    "physical_copy": "#e69f00",
}
B_T = 0.020
MTSI = np.arange(1, 7)
ECDI = np.arange(9, 22)
RIDGES = (0.0, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0)
VARIANTS = ("linear", "quadratic")


@dataclass
class ObservationDecoder:
    variant: str
    ridge: float
    z_mean: np.ndarray
    z_scale: np.ndarray
    feature_mean: np.ndarray
    target_mean: np.ndarray
    weights: np.ndarray

    def predict(self, z: np.ndarray) -> np.ndarray:
        features = feature_matrix(z, self.z_mean, self.z_scale, self.variant)
        return (features - self.feature_mean) @ self.weights + self.target_mean


def finite_float(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    finite = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    a = a[finite]
    b = b[finite]
    if np.std(a) <= np.finfo(float).tiny or np.std(b) <= np.finfo(float).tiny:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rollout(path: Path, components: int) -> dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row["layer"] == "translator"]
    if not rows:
        raise ValueError("No translator rows in latent rollout")
    result = {"time_us": np.asarray([float(row["time_us"]) for row in rows])}
    for method, prefix in METHOD_COLUMNS.items():
        result[method] = np.asarray(
            [
                [float(row[f"{prefix}_pc{index}"]) for index in range(1, components + 1)]
                for row in rows
            ],
            dtype=np.float64,
        )
    return result


def latent_scores(
    latent_h5: Path,
    pca_npz: Path,
    components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pca = np.load(pca_npz)
    mean = np.asarray(pca["mean"], dtype=np.float64)
    basis = np.asarray(pca["components"][:components], dtype=np.float64)
    with h5py.File(latent_h5, "r") as handle:
        pooled = np.asarray(handle["translator_pooled"], dtype=np.float32)
        frames = np.asarray(handle["translator_frame"], dtype=np.int64)
        times_us = np.asarray(handle["translator_time_s"], dtype=np.float64) * 1.0e6
    flattened = pooled.reshape(len(pooled), -1).astype(np.float64)
    scores = (flattened - mean) @ basis.T
    return frames, times_us, scores


def feature_matrix(
    z: np.ndarray,
    z_mean: np.ndarray,
    z_scale: np.ndarray,
    variant: str,
) -> np.ndarray:
    standardized = (np.asarray(z, dtype=np.float64) - z_mean) / z_scale
    if variant == "linear":
        return standardized
    if variant != "quadratic":
        raise ValueError(f"Unknown observation variant: {variant}")
    columns = [standardized]
    for left in range(standardized.shape[1]):
        columns.append(standardized[:, left : left + 1] * standardized[:, left:])
    return np.concatenate(columns, axis=1)


def fit_decoder(
    z: np.ndarray,
    target: np.ndarray,
    variant: str,
    ridge: float,
) -> ObservationDecoder:
    z_mean = np.mean(z, axis=0)
    z_scale = np.std(z, axis=0)
    z_scale[z_scale <= np.finfo(float).eps] = 1.0
    features = feature_matrix(z, z_mean, z_scale, variant)
    feature_mean = np.mean(features, axis=0)
    target_mean = np.mean(target, axis=0, dtype=np.float64)
    x = features - feature_mean
    y = np.asarray(target, dtype=np.float64) - target_mean
    gram = x.T @ x
    if ridge > 0.0:
        gram.flat[:: gram.shape[0] + 1] += ridge
    weights = np.linalg.pinv(gram, rcond=1.0e-12) @ (x.T @ y)
    return ObservationDecoder(
        variant=variant,
        ridge=ridge,
        z_mean=z_mean,
        z_scale=z_scale,
        feature_mean=feature_mean,
        target_mean=target_mean,
        weights=weights,
    )


def load_normalized_fields(
    handle: h5py.File,
    frames: np.ndarray,
    valid_h: int,
    valid_w: int,
) -> np.ndarray:
    first = int(np.min(frames))
    last = int(np.max(frames)) + 1
    block = np.asarray(handle["data_tchw"][first:last, :, :valid_h, :valid_w], dtype=np.float32)
    return block[frames - first]


def select_observation_model(
    scores: np.ndarray,
    times_us: np.ndarray,
    frames: np.ndarray,
    data_h5: Path,
    valid_h: int,
    valid_w: int,
) -> tuple[str, float, list[dict]]:
    subtrain = (times_us >= 20.0) & (times_us < 23.0)
    validation = (times_us >= 23.0) & (times_us < 24.0)
    with h5py.File(data_h5, "r") as handle:
        train_fields = load_normalized_fields(handle, frames[subtrain], valid_h, valid_w)
        validation_fields = load_normalized_fields(handle, frames[validation], valid_h, valid_w)
    # A regular spatial subset is sufficient for choosing the observation-map
    # family and ridge strength. The selected map is refit on every valid cell.
    train_fields = train_fields[:, :, ::4, ::4]
    validation_fields = validation_fields[:, :, ::4, ::4]

    rows = []
    best = None
    for variant in VARIANTS:
        for ridge in RIDGES:
            channel_mse = []
            for channel in range(len(CHANNELS)):
                decoder = fit_decoder(
                    scores[subtrain],
                    train_fields[:, channel].reshape(np.count_nonzero(subtrain), -1),
                    variant,
                    ridge,
                )
                prediction = decoder.predict(scores[validation])
                truth = validation_fields[:, channel].reshape(np.count_nonzero(validation), -1)
                channel_mse.append(float(np.mean((prediction - truth) ** 2)))
            objective = float(np.mean(channel_mse))
            row = {
                "variant": variant,
                "ridge": ridge,
                "validation_normalized_mse": objective,
                **{
                    f"{channel}_mse": channel_mse[index]
                    for index, channel in enumerate(CHANNELS)
                },
            }
            rows.append(row)
            if best is None or objective < best[0]:
                best = (objective, variant, ridge)
            print(
                f"[OBSERVATION] {variant} ridge={ridge:g} validation MSE={objective:.6e}",
                flush=True,
            )
    assert best is not None
    return best[1], best[2], rows


def fit_final_decoders(
    scores: np.ndarray,
    times_us: np.ndarray,
    frames: np.ndarray,
    data_h5: Path,
    valid_h: int,
    valid_w: int,
    variant: str,
    ridge: float,
) -> list[ObservationDecoder]:
    train = (times_us >= 20.0) & (times_us < 24.0)
    with h5py.File(data_h5, "r") as handle:
        fields = load_normalized_fields(handle, frames[train], valid_h, valid_w)
    decoders = []
    for channel in range(len(CHANNELS)):
        decoders.append(
            fit_decoder(
                scores[train],
                fields[:, channel].reshape(np.count_nonzero(train), -1),
                variant,
                ridge,
            )
        )
    return decoders


def decode_fields(
    decoders: list[ObservationDecoder],
    z: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    result = np.empty((len(z), len(CHANNELS), *shape), dtype=np.float32)
    for channel, decoder in enumerate(decoders):
        result[:, channel] = decoder.predict(z).reshape(len(z), *shape).astype(np.float32)
    return result


def denormalize(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return (
        values * (high - low)[None, :, None, None]
        + low[None, :, None, None]
    ).astype(np.float32)


def field_metrics(truth: np.ndarray, prediction: np.ndarray, method: str) -> list[dict]:
    rows = []
    for channel, name in enumerate(CHANNELS):
        actual = np.asarray(truth[:, channel], dtype=np.float64)
        estimate = np.asarray(prediction[:, channel], dtype=np.float64)
        mse = float(np.mean((estimate - actual) ** 2))
        scale = float(np.std(actual))
        frame_corr = np.asarray(
            [corrcoef(actual[index], estimate[index]) for index in range(len(actual))]
        )
        rows.append(
            {
                "method": method,
                "channel": name,
                "mse": mse,
                "nrmse_truth_std": math.sqrt(mse) / scale if scale > 0.0 else float("nan"),
                "global_correlation": corrcoef(actual, estimate),
                "mean_spatial_correlation": float(np.nanmean(frame_corr)),
            }
        )
    return rows


def physics_diagnostics(
    values: np.ndarray,
    x_mask: np.ndarray,
    dy_m: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    ne = np.asarray(values[:, 0, x_mask], dtype=np.float64)
    phi = np.asarray(values[:, 2, x_mask], dtype=np.float64)
    ey = -(np.roll(phi, -1, axis=-1) - np.roll(phi, 1, axis=-1)) / (2.0 * dy_m)
    fluctuations = {
        "ne": ne - np.mean(ne, axis=-1, keepdims=True),
        "phi": phi - np.mean(phi, axis=-1, keepdims=True),
        "ey": ey - np.mean(ey, axis=-1, keepdims=True),
    }
    spectra = {
        name: np.fft.rfft(field, axis=-1, norm="forward")
        for name, field in fluctuations.items()
    }
    weights = np.full(spectra["phi"].shape[-1], 2.0, dtype=np.float64)
    weights[0] = 1.0
    weights[-1] = 1.0
    result: dict[str, np.ndarray] = {}
    for name, spectrum in spectra.items():
        power = np.mean(np.abs(spectrum) ** 2, axis=1)
        for band_name, modes in (("mtsi", MTSI), ("ecdi", ECDI)):
            result[f"{name}_{band_name}_amplitude"] = np.sqrt(
                np.sum(weights[modes][None, :] * power[:, modes], axis=-1)
            )

    cross = spectra["ne"] * np.conj(spectra["ey"])
    for band_name, modes in (("mtsi", MTSI), ("ecdi", ECDI)):
        aggregate = np.sum(
            weights[modes][None, None, :] * cross[:, :, modes], axis=(1, 2)
        )
        result[f"cross_phase_{band_name}"] = np.angle(aggregate)

    result["transport_total"] = -np.mean(
        fluctuations["ne"] * fluctuations["ey"], axis=(1, 2)
    ) / B_T
    modal_transport = (
        -weights[None, None, :] * np.real(cross) / B_T
    )
    modal_transport = np.mean(modal_transport, axis=1)
    result["transport_mtsi"] = np.sum(modal_transport[:, MTSI], axis=-1)
    result["transport_ecdi"] = np.sum(modal_transport[:, ECDI], axis=-1)
    return result, cross


def weighted_phase_mae(
    truth_cross: np.ndarray,
    prediction_cross: np.ndarray,
    modes: np.ndarray,
) -> float:
    truth = truth_cross[:, :, modes]
    prediction = prediction_cross[:, :, modes]
    error = np.abs(np.angle(prediction * np.conj(truth)))
    weights = np.abs(truth)
    finite = np.isfinite(error) & np.isfinite(weights)
    denominator = float(np.sum(weights[finite]))
    if denominator <= np.finfo(float).tiny:
        return float("nan")
    return float(np.sum(error[finite] * weights[finite]) / denominator)


def observable_metrics(
    truth_series: dict[str, np.ndarray],
    truth_cross: np.ndarray,
    prediction_series: dict[str, np.ndarray],
    prediction_cross: np.ndarray,
    method: str,
) -> tuple[list[dict], list[dict]]:
    rows = []
    for name, truth in truth_series.items():
        prediction = prediction_series[name]
        mae = float(np.mean(np.abs(prediction - truth)))
        scale = float(np.std(truth))
        rows.append(
            {
                "method": method,
                "observable": name,
                "mae": mae,
                "nmae_truth_std": mae / scale if scale > 0.0 else float("nan"),
                "correlation": corrcoef(truth, prediction),
            }
        )
    phase_rows = []
    for band, modes in (("mtsi", MTSI), ("ecdi", ECDI)):
        mae = weighted_phase_mae(truth_cross, prediction_cross, modes)
        phase_rows.append(
            {
                "method": method,
                "band": band,
                "weighted_cross_phase_mae_rad": mae,
                "weighted_cross_phase_skill_vs_random": 1.0 - mae / (math.pi / 2.0),
            }
        )
    return rows, phase_rows


def plot_metric_summary(
    output: Path,
    field_rows: list[dict],
    observable_rows: list[dict],
    phase_rows: list[dict],
) -> None:
    methods = list(PLOT_METHODS)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    x = np.arange(len(methods))
    width = 0.24
    for offset, channel in enumerate(CHANNELS):
        values = [
            next(
                row["nrmse_truth_std"]
                for row in field_rows
                if row["method"] == method and row["channel"] == channel
            )
            for method in methods
        ]
        axes[0, 0].bar(x + (offset - 1) * width, values, width, label=channel)
    axes[0, 0].set_title("Physical-field reconstruction error")
    axes[0, 0].set_ylabel("NRMSE / truth standard deviation")
    axes[0, 0].set_xticks(x, [METHOD_LABELS[m] for m in methods], rotation=18)
    axes[0, 0].legend(loc="lower right")

    for offset, band in enumerate(("mtsi", "ecdi")):
        values = [
            next(
                row["correlation"]
                for row in observable_rows
                if row["method"] == method
                and row["observable"] == f"phi_{band}_amplitude"
            )
            for method in methods
        ]
        axes[0, 1].bar(x + (offset - 0.5) * 0.34, values, 0.34, label=band.upper())
    axes[0, 1].axhline(0.0, color="#555555", linewidth=0.8)
    axes[0, 1].set_title("Instability-band amplitude correlation")
    axes[0, 1].set_ylabel("correlation with truth")
    axes[0, 1].set_xticks(x, [METHOD_LABELS[m] for m in methods], rotation=18)
    axes[0, 1].legend(loc="lower right")

    for offset, band in enumerate(("mtsi", "ecdi")):
        values = [
            next(
                row["weighted_cross_phase_mae_rad"]
                for row in phase_rows
                if row["method"] == method and row["band"] == band
            )
            for method in methods
        ]
        axes[1, 0].bar(x + (offset - 0.5) * 0.34, values, 0.34, label=band.upper())
    axes[1, 0].axhline(math.pi / 2.0, color="#777777", linestyle="--", label="random phase")
    axes[1, 0].set_title("Density-Ey cross-phase error")
    axes[1, 0].set_ylabel("power-weighted MAE [rad]")
    axes[1, 0].set_xticks(x, [METHOD_LABELS[m] for m in methods], rotation=18)
    axes[1, 0].legend(loc="lower right")

    for offset, band in enumerate(("mtsi", "ecdi")):
        values = [
            next(
                row["correlation"]
                for row in observable_rows
                if row["method"] == method
                and row["observable"] == f"transport_{band}"
            )
            for method in methods
        ]
        axes[1, 1].bar(x + (offset - 0.5) * 0.34, values, 0.34, label=band.upper())
    axes[1, 1].axhline(0.0, color="#555555", linewidth=0.8)
    axes[1, 1].set_title("Modal transport correlation")
    axes[1, 1].set_ylabel("correlation with truth")
    axes[1, 1].set_xticks(x, [METHOD_LABELS[m] for m in methods], rotation=18)
    axes[1, 1].legend(loc="lower right")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_observable_rollouts(
    output: Path,
    time_us: np.ndarray,
    series: dict[str, dict[str, np.ndarray]],
) -> None:
    panels = (
        ("phi_mtsi_amplitude", "phi MTSI-band amplitude"),
        ("phi_ecdi_amplitude", "phi ECDI-band amplitude"),
        ("cross_phase_mtsi", "density-Ey MTSI cross-phase [rad]"),
        ("cross_phase_ecdi", "density-Ey ECDI cross-phase [rad]"),
        ("transport_mtsi", "MTSI-band transport proxy"),
        ("transport_ecdi", "ECDI-band transport proxy"),
    )
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex=True, constrained_layout=True)
    for axis, (quantity, title) in zip(axes.flat, panels):
        for method in ("truth", "oracle", "havok", "physical_copy"):
            axis.plot(
                time_us,
                series[method][quantity],
                color=COLORS[method],
                linewidth=1.5 if method == "truth" else 1.1,
                label="Truth" if method == "truth" else METHOD_LABELS[method],
            )
        axis.set_title(title)
        axis.grid(alpha=0.22)
        axis.legend(loc="lower right")
    axes[-1, 0].set_xlabel("time [us]")
    axes[-1, 1].set_xlabel("time [us]")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_snapshots(
    output_dir: Path,
    time_us: np.ndarray,
    truth: np.ndarray,
    oracle: np.ndarray,
    havok: np.ndarray,
) -> None:
    targets = (24.015, 26.0, 28.0, float(time_us[-1]))
    indices = [int(np.argmin(np.abs(time_us - target))) for target in targets]
    for channel, name in enumerate(CHANNELS):
        selected_truth = truth[indices, channel]
        selected_oracle = oracle[indices, channel]
        selected_havok = havok[indices, channel]
        low, high = np.percentile(selected_truth, [1.0, 99.0])
        error_limit = np.percentile(np.abs(selected_havok - selected_truth), 99.0)
        fig, axes = plt.subplots(4, len(indices), figsize=(16, 12), constrained_layout=True)
        images = []
        for column, index in enumerate(indices):
            for row, values in enumerate((truth, oracle, havok)):
                image = axes[row, column].imshow(
                    values[index, channel], origin="lower", aspect="auto", vmin=low, vmax=high, cmap="viridis"
                )
                images.append(image)
            axes[3, column].imshow(
                havok[index, channel] - truth[index, channel],
                origin="lower",
                aspect="auto",
                vmin=-error_limit,
                vmax=error_limit,
                cmap="RdBu_r",
            )
            axes[0, column].set_title(f"t={time_us[index]:.3f} us")
        for row, label in enumerate(("Truth", "Oracle", "HAVOK", "HAVOK - truth")):
            axes[row, 0].set_ylabel(label)
        fig.suptitle(f"Latent-to-physical reconstruction: {name}")
        fig.colorbar(images[0], ax=axes[:3], location="right", shrink=0.72)
        fig.savefig(output_dir / f"latent_to_physical_snapshots_{name}.png", dpi=160)
        plt.close(fig)


def save_prediction_h5(
    path: Path,
    time_us: np.ndarray,
    frames: np.ndarray,
    prediction: np.ndarray,
    metadata: dict,
) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("time_us", data=time_us)
        handle.create_dataset("frame", data=frames)
        handle.create_dataset("props", data=np.asarray(CHANNELS, dtype="S"))
        handle.create_dataset(
            "fields_tchw",
            data=prediction,
            chunks=(1, 1, prediction.shape[-2], prediction.shape[-1]),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        for key, value in metadata.items():
            handle.attrs[key] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_H5)
    parser.add_argument("--latent-dir", type=Path, default=LATENT_DIR)
    parser.add_argument("--hankel-dir", type=Path, default=HANKEL_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--components", type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    latent_h5 = args.latent_dir / "radaz_latent_features.h5"
    pca_npz = args.latent_dir / "pca_translator_steady.npz"
    rollout_csv = args.hankel_dir / "hankel_havok_rollout.csv"
    frames, latent_times, scores = latent_scores(latent_h5, pca_npz, args.components)
    rollout = read_rollout(rollout_csv, args.components)
    fit_mask = (latent_times >= 20.0) & (latent_times < 24.0)
    score_mean = np.mean(scores[fit_mask], axis=0)
    score_scale = np.std(scores[fit_mask], axis=0, ddof=1)
    score_scale[score_scale <= 1.0e-12] = 1.0
    for method in METHOD_COLUMNS:
        rollout[method] = rollout[method] * score_scale + score_mean

    with h5py.File(args.data, "r") as handle:
        valid_h, valid_w = np.asarray(handle["valid_spatial_shape"], dtype=int)
        time_s = np.asarray(handle["time_s"], dtype=np.float64)
        low = np.asarray(handle["norm_low"], dtype=np.float32)
        high = np.asarray(handle["norm_high"], dtype=np.float32)
        x_m = np.asarray(handle["x_m"], dtype=np.float64)
        y_m = np.asarray(handle["y_m"], dtype=np.float64)

    variant, ridge, selection_rows = select_observation_model(
        scores, latent_times, frames, args.data, valid_h, valid_w
    )
    write_rows(args.output / "observation_decoder_selection.csv", selection_rows)
    print(f"[SELECTED] observation={variant}, ridge={ridge:g}")
    decoders = fit_final_decoders(
        scores,
        latent_times,
        frames,
        args.data,
        valid_h,
        valid_w,
        variant,
        ridge,
    )

    target_frames = np.asarray(
        [int(np.argmin(np.abs(time_s * 1.0e6 - value))) for value in rollout["time_us"]],
        dtype=np.int64,
    )
    if not np.allclose(time_s[target_frames] * 1.0e6, rollout["time_us"], atol=1.0e-8):
        raise ValueError("Latent rollout times do not align with physical frames")
    latent_lookup = {int(frame): index for index, frame in enumerate(frames)}
    expected_truth_scores = np.asarray([scores[latent_lookup[int(frame)]] for frame in target_frames])
    max_score_difference = float(np.max(np.abs(expected_truth_scores - rollout["oracle"])))
    if max_score_difference > 1.0e-4:
        raise ValueError(f"Rollout/PCA score mismatch: {max_score_difference}")

    with h5py.File(args.data, "r") as handle:
        truth_normalized = load_normalized_fields(handle, target_frames, valid_h, valid_w)
        copy_normalized = np.repeat(
            np.asarray(handle["data_tchw"][target_frames[0] - 1 : target_frames[0], :, :valid_h, :valid_w], dtype=np.float32),
            len(target_frames),
            axis=0,
        )
    truth = denormalize(truth_normalized, low, high)
    physical_copy = denormalize(copy_normalized, low, high)
    decoded = {
        method: denormalize(
            decode_fields(decoders, rollout[method], (valid_h, valid_w)), low, high
        )
        for method in METHOD_COLUMNS
    }
    decoded["physical_copy"] = physical_copy

    x_mask = (x_m >= 0.09e-2 - 1.0e-15) & (x_m <= 1.19e-2 + 1.0e-15)
    dy_m = float(np.mean(np.diff(y_m)))
    truth_series, truth_cross = physics_diagnostics(truth, x_mask, dy_m)
    all_series = {"truth": truth_series}
    field_rows: list[dict] = []
    observable_rows: list[dict] = []
    phase_rows: list[dict] = []
    for method, values in decoded.items():
        field_rows.extend(field_metrics(truth, values, method))
        method_series, method_cross = physics_diagnostics(values, x_mask, dy_m)
        all_series[method] = method_series
        rows, phases = observable_metrics(
            truth_series, truth_cross, method_series, method_cross, method
        )
        observable_rows.extend(rows)
        phase_rows.extend(phases)
        print(f"[METRICS] {method}", flush=True)

    write_rows(args.output / "physical_field_metrics.csv", field_rows)
    write_rows(args.output / "physical_observable_metrics.csv", observable_rows)
    write_rows(args.output / "cross_phase_metrics.csv", phase_rows)
    per_frame_rows = []
    for index, time_us in enumerate(rollout["time_us"]):
        row = {"frame": int(target_frames[index]), "time_us": float(time_us)}
        for method, quantities in all_series.items():
            for name, values in quantities.items():
                row[f"{method}_{name}"] = float(values[index])
        per_frame_rows.append(row)
    write_rows(args.output / "physical_observables_by_time.csv", per_frame_rows)

    plot_metric_summary(
        args.output / "latent_to_physical_metrics.png",
        field_rows,
        observable_rows,
        phase_rows,
    )
    plot_observable_rollouts(
        args.output / "physical_observable_rollouts.png",
        rollout["time_us"],
        all_series,
    )
    plot_snapshots(
        args.output,
        rollout["time_us"],
        truth,
        decoded["oracle"],
        decoded["havok"],
    )
    save_prediction_h5(
        args.output / "havok_reconstructed_physical_fields.h5",
        rollout["time_us"],
        target_frames,
        decoded["havok"],
        {
            "source_data": str(args.data),
            "source_latent": str(latent_h5),
            "source_rollout": str(rollout_csv),
            "observation_variant": variant,
            "observation_ridge": ridge,
            "units": "electron_den and ion_den [m^-3], phi [V]",
        },
    )

    def lookup(rows: list[dict], method: str, key: str, value: str) -> dict:
        return next(row for row in rows if row["method"] == method and row[key] == value)

    summary_methods = {}
    for method in decoded:
        summary_methods[method] = {
            "field": {
                channel: {
                    key: finite_float(value)
                    for key, value in lookup(field_rows, method, "channel", channel).items()
                    if key not in ("method", "channel")
                }
                for channel in CHANNELS
            },
            "phi_amplitude": {
                band: {
                    key: finite_float(value)
                    for key, value in lookup(
                        observable_rows, method, "observable", f"phi_{band}_amplitude"
                    ).items()
                    if key not in ("method", "observable")
                }
                for band in ("mtsi", "ecdi")
            },
            "transport": {
                band: {
                    key: finite_float(value)
                    for key, value in lookup(
                        observable_rows, method, "observable", f"transport_{band}"
                    ).items()
                    if key not in ("method", "observable")
                }
                for band in ("total", "mtsi", "ecdi")
            },
            "cross_phase": {
                band: {
                    key: finite_float(value)
                    for key, value in lookup(phase_rows, method, "band", band).items()
                    if key not in ("method", "band")
                }
                for band in ("mtsi", "ecdi")
            },
        }

    summary = {
        "case": CASE.name,
        "latent_source": "frozen E10 SimVP translator, target-normalized E25 input",
        "components": args.components,
        "observation_decoder": {
            "selected_variant": variant,
            "selected_ridge": ridge,
            "selection_interval_us": [20.0, 23.0, 24.0],
            "final_fit_interval_us": [20.0, 24.0],
        },
        "holdout": {
            "start_us": float(rollout["time_us"][0]),
            "end_us": float(rollout["time_us"][-1]),
            "frames": len(target_frames),
        },
        "rollout_score_consistency_max_abs_difference": max_score_difference,
        "methods": summary_methods,
        "caveats": [
            "The observation decoder is fitted on E25 20-24 us and is not a zero-shot physical decoder.",
            "Oracle reconstruction measures the observation-decoder ceiling and is not a forecast.",
            "The saved 8x8 pooled state cannot be passed through the original SimVP decoder because full-resolution latent and skip tensors were not retained.",
            "MTSI/ECDI are candidate azimuthal mode bands n=1-6 and n=9-21, not dispersion-relation identification.",
        ],
    }
    (args.output / "latent_to_physical_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    oracle_phi = summary_methods["oracle"]["field"]["phi"]
    havok_phi = summary_methods["havok"]["field"]["phi"]
    oracle_transport = summary_methods["oracle"]["transport"]
    havok_transport = summary_methods["havok"]["transport"]
    readme = f"""# E25 latent-to-physical reconstruction

10 kV/mで学習した凍結SimVPから抽出した25 kV/m translator潜在状態を12主成分へ圧縮し、Hankel/HAVOKの24.015--{rollout['time_us'][-1]:.3f} us自律予測を物理場へ戻した。

保存済み潜在状態は8 x 8平均プーリング後で、SimVP decoderに必要な高解像度skipを持たない。そのため、20--23 usで候補を選び、20--24 usで最終fitした `{variant}` ridge観測写像を使った。oracleは真値潜在PCを同じ写像へ通した復元上限であり、予測ではない。

## 主要値

| 指標 | Oracle | HAVOK | Physical copy |
|---|---:|---:|---:|
| phi NRMSE / truth std | {oracle_phi['nrmse_truth_std']:.4f} | {havok_phi['nrmse_truth_std']:.4f} | {summary_methods['physical_copy']['field']['phi']['nrmse_truth_std']:.4f} |
| phi spatial correlation | {oracle_phi['mean_spatial_correlation']:.4f} | {havok_phi['mean_spatial_correlation']:.4f} | {summary_methods['physical_copy']['field']['phi']['mean_spatial_correlation']:.4f} |
| MTSI transport correlation | {oracle_transport['mtsi']['correlation']:.4f} | {havok_transport['mtsi']['correlation']:.4f} | {summary_methods['physical_copy']['transport']['mtsi']['correlation']:.4f} |
| ECDI transport correlation | {oracle_transport['ecdi']['correlation']:.4f} | {havok_transport['ecdi']['correlation']:.4f} | {summary_methods['physical_copy']['transport']['ecdi']['correlation']:.4f} |

| 指標 | Oracle | HAVOK | Physical copy |
|---|---:|---:|---:|
| phi MTSI amplitude correlation | {summary_methods['oracle']['phi_amplitude']['mtsi']['correlation']:.4f} | {summary_methods['havok']['phi_amplitude']['mtsi']['correlation']:.4f} | {summary_methods['physical_copy']['phi_amplitude']['mtsi']['correlation']:.4f} |
| phi ECDI amplitude correlation | {summary_methods['oracle']['phi_amplitude']['ecdi']['correlation']:.4f} | {summary_methods['havok']['phi_amplitude']['ecdi']['correlation']:.4f} | {summary_methods['physical_copy']['phi_amplitude']['ecdi']['correlation']:.4f} |
| MTSI cross-phase MAE [rad] | {summary_methods['oracle']['cross_phase']['mtsi']['weighted_cross_phase_mae_rad']:.4f} | {summary_methods['havok']['cross_phase']['mtsi']['weighted_cross_phase_mae_rad']:.4f} | {summary_methods['physical_copy']['cross_phase']['mtsi']['weighted_cross_phase_mae_rad']:.4f} |
| ECDI cross-phase MAE [rad] | {summary_methods['oracle']['cross_phase']['ecdi']['weighted_cross_phase_mae_rad']:.4f} | {summary_methods['havok']['cross_phase']['ecdi']['weighted_cross_phase_mae_rad']:.4f} | {summary_methods['physical_copy']['cross_phase']['ecdi']['weighted_cross_phase_mae_rad']:.4f} |

## 解釈

大域的な場の復元はcopyより高精度である。一方、oracleでさえECDI振幅・輸送の時間相関は低い。したがってECDIの失敗はHankel/HAVOKだけでなく、8 x 8平均プーリングと12 PCへの圧縮による情報欠落が主因である。潜在軌道の高い相関は場の大域構造については有効だが、ECDIと輸送を閉じる十分状態ではない。

次は未プール潜在状態のFourier係数、またはmode-aware poolingを使い、まずoracleでECDI振幅と輸送が復元できる状態を作る。

## 出力

- `latent_to_physical_metrics.png`
- `physical_observable_rollouts.png`
- `latent_to_physical_snapshots_*.png`
- `physical_field_metrics.csv`
- `physical_observable_metrics.csv`
- `cross_phase_metrics.csv`
- `physical_observables_by_time.csv`
- `havok_reconstructed_physical_fields.h5`
- `latent_to_physical_summary.json`
"""
    (args.output / "README.md").write_text(readme, encoding="utf-8")
    print(f"[DONE] {args.output}")


if __name__ == "__main__":
    main()
