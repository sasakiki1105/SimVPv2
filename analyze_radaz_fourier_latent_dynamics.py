"""Extract mode-aware SimVP latent features and fit reduced dynamics.

The full 65x64 latent grid is never stored. Radial-band averages and selected
azimuthal Fourier coefficients are computed during the frozen-model forward
pass, then reused for PCA, DMD, Hankel DMD and HAVOK analyses.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

import analyze_radaz_hankel_havok as hankel
import analyze_radaz_latent_features as latent
import analyze_radaz_reduced_dynamics as reduced


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_ez10kvm_fourier_latent_dynamics"
)

PRE = 10
AFT = 10
FIT_START_US = 20.0
FORECAST_START_US = 24.0
FORECAST_END_US = 30.0


def radial_boundaries(height: int, bands: int) -> np.ndarray:
    groups = np.array_split(np.arange(height, dtype=np.int64), bands)
    boundaries = [int(group[0]) for group in groups]
    boundaries.append(int(groups[-1][-1]) + 1)
    return np.asarray(boundaries, dtype=np.int64)


def fourier_features(
    state: torch.Tensor,
    boundaries: np.ndarray,
    maximum_mode: int,
) -> torch.Tensor:
    band_values = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        band_values.append(state[:, :, int(start) : int(end)].mean(dim=2))
    banded = torch.stack(band_values, dim=2)
    coefficients = torch.fft.rfft(banded, dim=-1, norm="forward")
    coefficients = coefficients[..., : maximum_mode + 1]
    return torch.view_as_real(coefficients)


@torch.inference_mode()
def extract_fourier_latents(
    data_path: Path,
    workdir: Path,
    config_path: Path,
    output_path: Path,
    device: torch.device,
    batch_size: int,
    bands: int,
    maximum_mode: int,
) -> dict:
    with h5py.File(data_path, "r") as source:
        dataset = source["data_tchw"]
        frames, channels, height, width = dataset.shape
        timesteps = np.asarray(source["timesteps"], dtype=np.int64)
        time_s = np.asarray(source["time_s"], dtype=np.float64)
        train_end = int(source["train_frame_end_exclusive"][()])
    starts = np.arange(0, frames - PRE - AFT + 1, dtype=np.int64)
    model = latent.build_model(
        config_path,
        latent.checkpoint_path(workdir),
        device,
        (PRE, channels, height, width),
    )

    temporary = output_path.with_suffix(".partial.h5")
    if temporary.exists():
        temporary.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latent_shape = None
    boundaries = None
    encoder_dataset = None
    translator_dataset = None

    with h5py.File(data_path, "r") as source, h5py.File(
        temporary, "w"
    ) as target:
        dataset = source["data_tchw"]
        for batch_start in range(0, len(starts), batch_size):
            batch_starts = starts[batch_start : batch_start + batch_size]
            x_np = latent.read_windows(dataset, batch_starts)
            x = torch.from_numpy(x_np).to(device)
            batch, sequence, input_channels, input_h, input_w = x.shape
            encoded, _ = model.enc(
                x.reshape(
                    batch * sequence,
                    input_channels,
                    input_h,
                    input_w,
                )
            )
            _, latent_channels, latent_h, latent_w = encoded.shape
            encoded = encoded.reshape(
                batch,
                sequence,
                latent_channels,
                latent_h,
                latent_w,
            )
            translated = model.hid(encoded)
            if latent_shape is None:
                latent_shape = (
                    int(sequence),
                    int(latent_channels),
                    int(latent_h),
                    int(latent_w),
                )
                if maximum_mode > latent_w // 2:
                    raise ValueError(
                        f"maximum_mode={maximum_mode} exceeds "
                        f"latent Nyquist mode {latent_w // 2}"
                    )
                boundaries = radial_boundaries(latent_h, bands)
                feature_shape = (
                    int(latent_channels),
                    int(bands),
                    int(maximum_mode + 1),
                    2,
                )
                compression = {
                    "compression": "gzip",
                    "compression_opts": 4,
                    "shuffle": True,
                }
                encoder_dataset = target.create_dataset(
                    "encoder_fourier_ri",
                    shape=(len(starts),) + feature_shape,
                    dtype=np.float32,
                    chunks=(1,) + feature_shape,
                    **compression,
                )
                translator_dataset = target.create_dataset(
                    "translator_fourier_ri",
                    shape=(len(starts),) + feature_shape,
                    dtype=np.float32,
                    chunks=(1,) + feature_shape,
                    **compression,
                )
                print(
                    f"[LATENT] full={latent_shape}, Fourier={feature_shape}, "
                    f"radial boundaries={boundaries.tolist()}"
                )

            encoder_fourier = fourier_features(
                encoded[:, -1], boundaries, maximum_mode
            )
            translator_fourier = fourier_features(
                translated[:, 0], boundaries, maximum_mode
            )
            batch_end = batch_start + len(batch_starts)
            encoder_dataset[batch_start:batch_end] = (
                encoder_fourier.cpu().numpy().astype(np.float32)
            )
            translator_dataset[batch_start:batch_end] = (
                translator_fourier.cpu().numpy().astype(np.float32)
            )
            completed = min(batch_end, len(starts))
            if completed == len(starts) or completed % 50 == 0:
                print(
                    f"[EXTRACT] {completed}/{len(starts)} windows",
                    flush=True,
                )

        encoder_frame = starts + PRE - 1
        translator_frame = starts + PRE
        target.create_dataset("window_start", data=starts)
        target.create_dataset("encoder_frame", data=encoder_frame)
        target.create_dataset("translator_frame", data=translator_frame)
        target.create_dataset(
            "encoder_time_s", data=time_s[encoder_frame]
        )
        target.create_dataset(
            "translator_time_s", data=time_s[translator_frame]
        )
        target.create_dataset("timesteps", data=timesteps)
        target.create_dataset("radial_boundaries", data=boundaries)
        target.create_dataset(
            "azimuthal_modes",
            data=np.arange(maximum_mode + 1, dtype=np.int64),
        )
        target.create_dataset(
            "full_latent_shape", data=np.asarray(latent_shape)
        )
        target.create_dataset(
            "train_frame_end_exclusive", data=train_end
        )
        target.attrs["source_data"] = str(data_path)
        target.attrs["source_workdir"] = str(workdir)
        target.attrs["checkpoint"] = str(
            latent.checkpoint_path(workdir)
        )
        target.attrs["config"] = str(config_path)
        target.attrs["fft_normalization"] = "forward"
        target.attrs["encoder_alignment"] = "last true input frame"
        target.attrs["translator_alignment"] = "first predicted output frame"

    temporary.replace(output_path)
    result = {
        "windows": int(len(starts)),
        "input_shape": [PRE, channels, height, width],
        "full_latent_shape": list(latent_shape),
        "fourier_feature_shape": [
            latent_shape[1],
            bands,
            maximum_mode + 1,
            2,
        ],
        "radial_boundaries": boundaries.tolist(),
        "maximum_mode": maximum_mode,
        "output_h5": str(output_path),
        "file_bytes": output_path.stat().st_size,
    }
    print(f"[H5] {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    return result


def fit_fourier_pca(
    feature_path: Path,
    output_dir: Path,
    maximum_components: int,
    variance_target: float,
    feature_scaling: str,
) -> tuple[dict[str, reduced.LayerData], dict]:
    layers: dict[str, reduced.LayerData] = {}
    summaries: dict[str, dict] = {}
    score_path = output_dir / "fourier_latent_pca_scores.h5"
    with h5py.File(feature_path, "r") as source, h5py.File(
        score_path, "w"
    ) as score_file:
        starts = np.asarray(source["window_start"], dtype=np.int64)
        train_end = int(source["train_frame_end_exclusive"][()])
        full_train = starts + PRE + AFT - 1 < train_end

        for layer_name in ("encoder", "translator"):
            dataset = source[f"{layer_name}_fourier_ri"]
            time_us = (
                np.asarray(
                    source[f"{layer_name}_time_s"], dtype=np.float64
                )
                * 1.0e6
            )
            feature_shape = dataset.shape[1:]
            feature_tensor = np.asarray(dataset, dtype=np.float32)
            fit_mask = (
                full_train
                & (time_us >= FIT_START_US)
                & (time_us < FORECAST_START_US)
            )
            if feature_scaling == "mode_rms":
                fit_center = np.mean(
                    feature_tensor[fit_mask], axis=0, keepdims=True
                )
                mode_scale = np.sqrt(
                    np.mean(
                        (feature_tensor[fit_mask] - fit_center) ** 2,
                        axis=(0, 1, 2, 4),
                    )
                )
                positive = mode_scale[mode_scale > 0.0]
                floor = (
                    float(np.median(positive)) * 1.0e-6
                    if positive.size
                    else 1.0
                )
                mode_scale = np.where(mode_scale > floor, mode_scale, 1.0)
            elif feature_scaling == "none":
                mode_scale = np.ones(feature_shape[2], dtype=np.float32)
            else:
                raise ValueError(f"Unknown feature scaling: {feature_scaling}")
            scale_tensor = np.broadcast_to(
                mode_scale[None, None, :, None], feature_shape
            ).astype(np.float32)
            features = (
                feature_tensor / scale_tensor[None, ...]
            ).reshape(len(starts), -1)
            fit_features = features[fit_mask]
            feature_variance = np.var(fit_features, axis=0)
            variance_floor = max(
                float(np.max(feature_variance)) * 1.0e-12, 1.0e-14
            )
            active = feature_variance > variance_floor
            active_features = features[:, active]
            component_limit = min(
                maximum_components,
                int(np.count_nonzero(fit_mask) - 1),
                int(np.count_nonzero(active)),
            )
            pca = PCA(
                n_components=component_limit,
                svd_solver="randomized",
                random_state=0,
                iterated_power=4,
            )
            pca.fit(active_features[fit_mask])
            scores = pca.transform(active_features).astype(np.float32)
            cumulative = np.cumsum(pca.explained_variance_ratio_)
            reached = np.flatnonzero(cumulative >= variance_target)
            retained = (
                int(reached[0] + 1)
                if reached.size
                else int(component_limit)
            )
            target_reached = bool(reached.size)
            layers[layer_name] = reduced.LayerData(
                layer_name,
                retained,
                time_us,
                scores[:, :retained].astype(np.float64),
            )
            summaries[layer_name] = {
                "fit_samples": int(np.count_nonzero(fit_mask)),
                "total_features": int(features.shape[1]),
                "active_features": int(np.count_nonzero(active)),
                "computed_components": component_limit,
                "variance_target": variance_target,
                "components_for_target": retained,
                "variance_target_reached": target_reached,
                "variance_at_retained_components": float(
                    cumulative[retained - 1]
                ),
                "components_for_90_percent": int(
                    np.searchsorted(cumulative, 0.90) + 1
                )
                if cumulative[-1] >= 0.90
                else None,
                "components_for_95_percent": int(
                    np.searchsorted(cumulative, 0.95) + 1
                )
                if cumulative[-1] >= 0.95
                else None,
                "components_for_99_percent": int(
                    np.searchsorted(cumulative, 0.99) + 1
                )
                if cumulative[-1] >= 0.99
                else None,
                "variance_pc1": float(
                    pca.explained_variance_ratio_[0]
                ),
                "variance_pc1_to_pc10": float(
                    cumulative[min(9, len(cumulative) - 1)]
                ),
                "feature_scaling": feature_scaling,
                "mode_scale": mode_scale.astype(float).tolist(),
            }
            np.savez_compressed(
                output_dir / f"fourier_pca_{layer_name}.npz",
                mean=pca.mean_,
                components=pca.components_,
                explained_variance=pca.explained_variance_,
                explained_variance_ratio=pca.explained_variance_ratio_,
                singular_values=pca.singular_values_,
                active_feature_mask=active,
                feature_shape=np.asarray(feature_shape, dtype=np.int64),
                mode_scale=mode_scale,
                feature_scaling=np.asarray(feature_scaling),
            )
            score_file.create_dataset(
                f"{layer_name}_scores",
                data=scores,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            score_file.create_dataset(
                f"{layer_name}_time_us", data=time_us
            )
            print(
                f"[PCA] {layer_name}: PC1={cumulative[0]:.4f}, "
                f"PC1-PC10={cumulative[min(9, len(cumulative)-1)]:.4f}, "
                f"{variance_target:.0%} components={retained}"
            )
            del feature_tensor, features, fit_features, active_features, scores
    return layers, {
        "score_h5": str(score_path),
        "layers": summaries,
    }


def plot_pca_variance(
    path: Path,
    pca_summary: dict,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for axis, layer in zip(axes, ("encoder", "translator")):
        with np.load(output_dir / f"fourier_pca_{layer}.npz") as pca:
            ratio = np.asarray(pca["explained_variance_ratio"])
        cumulative = np.cumsum(ratio)
        axis.plot(
            np.arange(1, len(cumulative) + 1),
            cumulative,
            color="#0068b5",
        )
        axis.axhline(0.95, color="#555555", linestyle="--")
        retained = pca_summary["layers"][layer][
            "components_for_target"
        ]
        axis.axvline(retained, color="#c34a36", linestyle=":")
        axis.set_xlim(1, len(cumulative))
        axis.set_ylim(0.0, 1.01)
        axis.set_xlabel("PCA components")
        axis.set_ylabel("cumulative explained variance")
        axis.set_title(f"{layer}: 95% at {retained} PCs")
        axis.grid(alpha=0.2)
    fig.suptitle("Mode-aware Fourier latent PCA")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_mode_energy(path: Path, feature_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    with h5py.File(feature_path, "r") as source:
        modes = np.asarray(source["azimuthal_modes"], dtype=np.int64)
        for axis, layer in zip(axes, ("encoder", "translator")):
            data = np.asarray(
                source[f"{layer}_fourier_ri"], dtype=np.float32
            )
            amplitude = np.sqrt(
                data[..., 0] ** 2 + data[..., 1] ** 2
            )
            mode_rms = np.sqrt(
                np.mean(amplitude * amplitude, axis=(0, 1, 2))
            )
            axis.bar(modes, mode_rms, color="#1675b8")
            axis.axvspan(1, 6, color="#e69f00", alpha=0.15, label="MTSI band")
            axis.axvspan(9, 21, color="#009e73", alpha=0.12, label="ECDI band")
            axis.set_ylabel("latent coefficient RMS")
            axis.set_title(layer)
            axis.grid(axis="y", alpha=0.2)
            axis.legend(loc="upper right")
            del data, amplitude
    axes[-1].set_xlabel("azimuthal mode n")
    fig.suptitle("Azimuthal Fourier content of unpooled SimVP latent states")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def dynamics_analysis(
    layers: dict[str, reduced.LayerData],
    output_dir: Path,
    delays: list[int],
    ranks: list[int],
    physical_path: Path,
) -> tuple[dict, dict[str, dict]]:
    physical = hankel.read_physical_metrics(physical_path)
    results: dict[str, dict] = {}
    layer_summaries: dict[str, dict] = {}
    candidate_rows: list[dict] = []
    correlation_rows: list[dict] = []
    for layer_name, layer in layers.items():
        print(
            f"[DYNAMICS] {layer_name}: {layer.components} Fourier PCs"
        )
        result, summary, candidates, correlations = hankel.analyze_layer(
            layer, physical, delays, ranks
        )
        results[layer_name] = result
        layer_summaries[layer_name] = summary
        selected = (
            summary["selected_delay"],
            summary["selected_rank"],
        )
        for row in candidates:
            candidate_rows.append(
                {
                    "layer": layer_name,
                    **row,
                    "selected": (
                        row["delay"],
                        row["rank"],
                    )
                    == selected,
                }
            )
        for row in correlations:
            correlation_rows.append({"layer": layer_name, **row})
        metrics = summary["metrics"]["hankel_dmd"]["24-30"]
        print(
            f"[RESULT] {layer_name}: q={selected[0]}, "
            f"rank={selected[1]}, "
            f"skill persistence={metrics['skill_vs_persistence']:.4f}, "
            f"skill mean={metrics['skill_vs_training_mean']:.4f}, "
            f"correlation={metrics['flattened_correlation']:.4f}"
        )

    safe_summary = reduced.json_safe({"layers": layer_summaries})
    hankel.write_candidate_csv(
        output_dir / "fourier_hankel_candidate_metrics.csv",
        candidate_rows,
    )
    hankel.write_metric_csv(
        output_dir / "fourier_hankel_havok_metrics.csv",
        safe_summary,
    )
    hankel.write_rollout_csv(
        output_dir / "fourier_hankel_havok_rollout.csv", results
    )
    hankel.write_forcing_csv(
        output_dir / "fourier_havok_forcing_correlations.csv",
        correlation_rows,
    )
    hankel.plot_rollout(
        output_dir / "fourier_hankel_havok_rollout_overview.png",
        results,
    )
    hankel.plot_phase_portrait(
        output_dir / "fourier_hankel_havok_phase_portraits.png",
        results,
    )
    hankel.plot_eigenvalues(
        output_dir / "fourier_hankel_dmd_eigenvalues.png", results
    )
    hankel.plot_forcing(
        output_dir / "fourier_havok_forcing_diagnostics.png",
        results,
        correlation_rows,
    )
    return layer_summaries, results


MODE_BANDS = {
    "MTSI candidate (n=1-6)": (1, 6),
    "ECDI candidate (n=9-21)": (9, 21),
}


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(finite) < 3:
        return float("nan")
    left_valid = left[finite]
    right_valid = right[finite]
    if np.std(left_valid) <= 1.0e-14 or np.std(right_valid) <= 1.0e-14:
        return float("nan")
    return float(np.corrcoef(left_valid, right_valid)[0, 1])


def inverse_fourier_pca(
    scores: np.ndarray,
    pca_path: Path,
) -> np.ndarray:
    with np.load(pca_path) as pca:
        mean = np.asarray(pca["mean"], dtype=np.float64)
        components = np.asarray(pca["components"], dtype=np.float64)
        active = np.asarray(pca["active_feature_mask"], dtype=bool)
        feature_shape = tuple(
            np.asarray(pca["feature_shape"], dtype=np.int64).tolist()
        )
        mode_scale = np.asarray(pca["mode_scale"], dtype=np.float64)
    retained = scores.shape[1]
    scaled_active = scores @ components[:retained] + mean
    scaled = np.zeros((len(scores), active.size), dtype=np.float64)
    scaled[:, active] = scaled_active
    scale_tensor = np.broadcast_to(
        mode_scale[None, None, :, None], feature_shape
    )
    return (
        scaled.reshape((len(scores),) + feature_shape) * scale_tensor
    ).astype(np.float32)


def coefficient_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    mode_start: int,
    mode_end: int,
) -> dict[str, float]:
    truth_band = truth[:, :, :, mode_start : mode_end + 1]
    prediction_band = prediction[:, :, :, mode_start : mode_end + 1]
    finite = np.isfinite(prediction_band)
    finite_fraction = float(np.mean(finite))
    if not np.all(finite):
        return {
            "coefficient_nrmse": float("inf"),
            "amplitude_nrmse": float("inf"),
            "amplitude_correlation": float("nan"),
            "mean_amplitude_ratio": float("nan"),
            "finite_fraction": finite_fraction,
        }
    coefficient_rmse = float(
        np.sqrt(np.mean((prediction_band - truth_band) ** 2))
    )
    coefficient_scale = float(
        np.sqrt(
            np.mean(
                (truth_band - np.mean(truth_band, axis=0, keepdims=True))
                ** 2
            )
        )
    )
    truth_power = np.sum(truth_band * truth_band, axis=-1)
    prediction_power = np.sum(
        prediction_band * prediction_band, axis=-1
    )
    truth_amplitude = np.sqrt(np.mean(truth_power, axis=(1, 2, 3)))
    prediction_amplitude = np.sqrt(
        np.mean(prediction_power, axis=(1, 2, 3))
    )
    amplitude_rmse = float(
        np.sqrt(np.mean((prediction_amplitude - truth_amplitude) ** 2))
    )
    amplitude_scale = float(np.std(truth_amplitude, ddof=1))
    return {
        "coefficient_nrmse": coefficient_rmse
        / max(coefficient_scale, 1.0e-14),
        "amplitude_nrmse": amplitude_rmse / max(amplitude_scale, 1.0e-14),
        "amplitude_correlation": safe_correlation(
            truth_amplitude, prediction_amplitude
        ),
        "mean_amplitude_ratio": float(
            np.mean(prediction_amplitude)
            / max(float(np.mean(truth_amplitude)), 1.0e-14)
        ),
        "finite_fraction": finite_fraction,
    }


def band_amplitude(
    features: np.ndarray, mode_start: int, mode_end: int
) -> np.ndarray:
    selected = features[:, :, :, mode_start : mode_end + 1]
    return np.sqrt(
        np.mean(np.sum(selected * selected, axis=-1), axis=(1, 2, 3))
    )


def evaluate_mode_forecasts(
    feature_path: Path,
    output_dir: Path,
    layers: dict[str, reduced.LayerData],
    results: dict[str, dict],
) -> dict:
    rows: list[dict] = []
    traces: dict[str, dict] = {}
    with h5py.File(feature_path, "r") as source:
        maximum_mode = int(np.max(source["azimuthal_modes"]))
        for layer_name, layer in layers.items():
            fit_mask = (layer.time_us >= FIT_START_US) & (
                layer.time_us < FORECAST_START_US
            )
            forecast_mask = (layer.time_us >= FORECAST_START_US) & (
                layer.time_us <= FORECAST_END_US
            )
            forecast_indices = np.flatnonzero(forecast_mask)
            if not np.all(np.diff(forecast_indices) == 1):
                raise ValueError("Forecast indices must be contiguous")
            truth = np.asarray(
                source[f"{layer_name}_fourier_ri"][
                    forecast_indices[0] : forecast_indices[-1] + 1
                ],
                dtype=np.float32,
            )
            standardizer = reduced.fit_standardizer(layer.scores[fit_mask])
            result = results[layer_name]
            score_predictions = {
                "oracle_pca": layer.scores[forecast_mask],
                "persistence": standardizer.inverse(result["persistence"]),
                "standard_dmd": standardizer.inverse(result["standard_dmd"]),
                "hankel_dmd": standardizer.inverse(result["hankel_dmd"]),
                "havok_zero_forcing": standardizer.inverse(
                    result["havok_zero_forcing"]
                ),
            }
            traces[layer_name] = {
                "time_us": result["forecast_time_us"],
                "truth": {
                    band: band_amplitude(truth, low, high)
                    for band, (low, high) in MODE_BANDS.items()
                },
                "methods": {},
            }
            for method, scores in score_predictions.items():
                prediction = inverse_fourier_pca(
                    scores, output_dir / f"fourier_pca_{layer_name}.npz"
                )
                method_traces = {}
                for mode in range(maximum_mode + 1):
                    metrics = coefficient_metrics(
                        truth, prediction, mode, mode
                    )
                    rows.append(
                        {
                            "layer": layer_name,
                            "method": method,
                            "scope": "mode",
                            "label": f"n={mode}",
                            "mode_start": mode,
                            "mode_end": mode,
                            **metrics,
                        }
                    )
                for band, (low, high) in MODE_BANDS.items():
                    metrics = coefficient_metrics(
                        truth, prediction, low, high
                    )
                    rows.append(
                        {
                            "layer": layer_name,
                            "method": method,
                            "scope": "band",
                            "label": band,
                            "mode_start": low,
                            "mode_end": high,
                            **metrics,
                        }
                    )
                    method_traces[band] = band_amplitude(
                        prediction, low, high
                    )
                traces[layer_name]["methods"][method] = method_traces
                del prediction

    csv_path = output_dir / "fourier_mode_forecast_metrics.csv"
    fields = [
        "layer",
        "method",
        "scope",
        "label",
        "mode_start",
        "mode_end",
        "coefficient_nrmse",
        "amplitude_nrmse",
        "amplitude_correlation",
        "mean_amplitude_ratio",
        "finite_fraction",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    plot_mode_band_rollouts(
        output_dir / "fourier_mode_band_autonomous_rollouts.png", traces
    )
    plot_mode_retention(
        output_dir / "fourier_mode_retention_and_prediction.png", rows
    )
    band_summary = {
        layer: {
            method: {
                row["label"]: {
                    key: row[key]
                    for key in (
                        "coefficient_nrmse",
                        "amplitude_nrmse",
                        "amplitude_correlation",
                        "mean_amplitude_ratio",
                    )
                }
                for row in rows
                if row["layer"] == layer
                and row["method"] == method
                and row["scope"] == "band"
            }
            for method in (
                "oracle_pca",
                "persistence",
                "standard_dmd",
                "hankel_dmd",
                "havok_zero_forcing",
            )
        }
        for layer in layers
    }
    return {
        "metrics_csv": str(csv_path),
        "band_metrics": band_summary,
    }


def plot_mode_band_rollouts(path: Path, traces: dict[str, dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    colors = {
        "oracle_pca": "#777777",
        "persistence": "#cc79a7",
        "hankel_dmd": "#0072b2",
        "havok_zero_forcing": "#009e73",
    }
    for row, layer in enumerate(("encoder", "translator")):
        time_us = traces[layer]["time_us"]
        for column, band in enumerate(MODE_BANDS):
            axis = axes[row, column]
            axis.plot(
                time_us,
                traces[layer]["truth"][band],
                color="#111111",
                linewidth=2.0,
                label="truth",
            )
            for method in (
                "oracle_pca",
                "persistence",
                "hankel_dmd",
                "havok_zero_forcing",
            ):
                axis.plot(
                    time_us,
                    traces[layer]["methods"][method][band],
                    color=colors[method],
                    linewidth=1.2,
                    alpha=0.9,
                    label=method,
                )
            axis.set_title(f"{layer}: {band}")
            axis.set_ylabel("latent Fourier RMS amplitude")
            axis.grid(alpha=0.2)
            axis.legend(loc="upper right", fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("time [us]")
    fig.suptitle("Autonomous forecasts of unpooled latent Fourier bands")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_mode_retention(path: Path, rows: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    methods = ("oracle_pca", "hankel_dmd", "havok_zero_forcing")
    colors = {
        "oracle_pca": "#777777",
        "hankel_dmd": "#0072b2",
        "havok_zero_forcing": "#009e73",
    }
    for row_index, layer in enumerate(("encoder", "translator")):
        layer_rows = [
            row
            for row in rows
            if row["layer"] == layer and row["scope"] == "mode"
        ]
        for method in methods:
            selected = sorted(
                [row for row in layer_rows if row["method"] == method],
                key=lambda row: row["mode_start"],
            )
            modes = [row["mode_start"] for row in selected]
            axes[row_index, 0].plot(
                modes,
                [row["mean_amplitude_ratio"] for row in selected],
                marker="o",
                markersize=2.5,
                color=colors[method],
                label=method,
            )
            axes[row_index, 1].plot(
                modes,
                [row["amplitude_correlation"] for row in selected],
                marker="o",
                markersize=2.5,
                color=colors[method],
                label=method,
            )
        axes[row_index, 0].axhline(1.0, color="#111111", linestyle="--")
        axes[row_index, 0].set_ylabel(f"{layer}\nmean amplitude / truth")
        axes[row_index, 1].set_ylabel(f"{layer}\namplitude correlation")
        for axis in axes[row_index]:
            axis.axvspan(1, 6, color="#e69f00", alpha=0.10)
            axis.axvspan(9, 21, color="#009e73", alpha=0.08)
            axis.grid(alpha=0.2)
            axis.legend(loc="lower right", fontsize=8)
    axes[0, 0].set_title("Mode amplitude retention")
    axes[0, 1].set_title("Mode amplitude trajectory correlation")
    axes[-1, 0].set_xlabel("azimuthal mode n")
    axes[-1, 1].set_xlabel("azimuthal mode n")
    fig.suptitle("Mode-resolved Fourier latent reconstruction and forecast")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_readme_legacy(
    path: Path,
    extraction: dict,
    pca: dict,
    dynamics: dict,
) -> None:
    rows = []
    for layer, summary in dynamics.items():
        for method in ("standard_dmd", "hankel_dmd", "havok_zero_forcing"):
            metrics = summary["metrics"][method]["24-30"]
            rows.append(
                "| {layer} | {pcs} | {method} | {delay} | {rank} | "
                "{rmse:.4f} | {skill_p:.4f} | {skill_m:.4f} | "
                "{corr:.4f} |".format(
                    layer=layer,
                    pcs=pca["layers"][layer]["components_for_target"],
                    method=method,
                    delay=summary["selected_delay"],
                    rank=summary["selected_rank"],
                    rmse=metrics["standardized_rmse"],
                    skill_p=metrics["skill_vs_persistence"],
                    skill_m=metrics["skill_vs_training_mean"],
                    corr=metrics["flattened_correlation"],
                )
            )
    text = f"""# RadAz mode-aware Fourier latent dynamics

The frozen SimVP model was evaluated on every direct10 window. Its full
`64 x 65 x 64` representative latent state was transformed in memory and was
not stored.

- Radial representation: 8 contiguous bands
- Azimuthal coefficients: complex modes `n=0-21`
- Stored representation: real and imaginary parts
- Fourier feature H5: `{extraction['file_bytes'] / 1e6:.1f} MB`
- PCA fit: original SimVP training windows at `20-24 us`
- Dynamics selection: `23-24 us`
- Autonomous holdout: `24-30 us`

## PCA

| Layer | raw Fourier features | active features | PCs for 95% |
|---|---:|---:|---:|
| encoder | {pca['layers']['encoder']['total_features']} | {pca['layers']['encoder']['active_features']} | {pca['layers']['encoder']['components_for_target']} |
| translator | {pca['layers']['translator']['total_features']} | {pca['layers']['translator']['active_features']} | {pca['layers']['translator']['components_for_target']} |

## Autonomous dynamics

| Layer | PCs | Method | delay | rank | RMSE | skill vs persistence | skill vs training mean | correlation |
|---|---:|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Positive skill against the training mean and non-trivial trajectory
correlation are both required before claiming a closed low-dimensional
dynamical model. Positive skill against persistence alone can result from
mean reversion.

## 日本語メモ

未プール潜在特徴をディスクへ保存せず、forward中に方位角FFTを行いました。
radial方向は8領域に分け、各hidden channelについて`n=0-21`の複素係数を
実部・虚部として保存しています。

Fourier潜在PCAが少数成分で表せるか、その成分の時間発展がpersistenceだけで
なく訓練平均にも勝つか、さらに位相相関を保てるかを分けて評価しています。
"""
    path.write_text(text, encoding="utf-8")


def write_readme(
    path: Path,
    extraction: dict,
    pca: dict,
    dynamics: dict,
    mode_forecast: dict,
) -> None:
    dynamics_rows = []
    for layer, summary in dynamics.items():
        for method in ("standard_dmd", "hankel_dmd", "havok_zero_forcing"):
            metrics = summary["metrics"][method]["24-30"]
            dynamics_rows.append(
                "| {layer} | {pcs} | {method} | {delay} | {rank} | "
                "{rmse:.4f} | {skill_p:.4f} | {skill_m:.4f} | "
                "{corr:.4f} |".format(
                    layer=layer,
                    pcs=pca["layers"][layer]["components_for_target"],
                    method=method,
                    delay=summary["selected_delay"],
                    rank=summary["selected_rank"],
                    rmse=metrics["standardized_rmse"],
                    skill_p=metrics["skill_vs_persistence"],
                    skill_m=metrics["skill_vs_training_mean"],
                    corr=metrics["flattened_correlation"],
                )
            )
    band_rows = []
    translator = mode_forecast["band_metrics"]["translator"]
    for method in (
        "oracle_pca",
        "persistence",
        "hankel_dmd",
        "havok_zero_forcing",
    ):
        for band, metrics in translator[method].items():
            band_rows.append(
                "| {method} | {band} | {ratio:.4f} | {corr:.4f} | "
                "{nrmse:.4f} |".format(
                    method=method,
                    band=band,
                    ratio=metrics["mean_amplitude_ratio"],
                    corr=metrics["amplitude_correlation"],
                    nrmse=metrics["coefficient_nrmse"],
                )
            )
    radial_bands = len(extraction["radial_boundaries"]) - 1
    maximum_mode = extraction["maximum_mode"]
    scaling = pca["layers"]["translator"]["feature_scaling"]
    text = f"""# RadAz mode-aware Fourier latent dynamics

The frozen SimVP model was evaluated on every direct10 window. Its full
`64 x 65 x 64` representative latent state was transformed in memory and was
not stored.

- Radial representation: {radial_bands} contiguous bands
- Azimuthal coefficients: complex modes `n=0-{maximum_mode}`
- Stored representation: real and imaginary parts
- PCA feature scaling: `{scaling}`
- Fourier feature H5: `{extraction['file_bytes'] / 1e6:.1f} MB`
- PCA and dynamics fit: `20-24 us`
- Hyperparameter selection: `23-24 us`
- Strict autonomous holdout: `24-30 us`

## PCA

| Layer | raw features | active features | PCs for target variance |
|---|---:|---:|---:|
| encoder | {pca['layers']['encoder']['total_features']} | {pca['layers']['encoder']['active_features']} | {pca['layers']['encoder']['components_for_target']} |
| translator | {pca['layers']['translator']['total_features']} | {pca['layers']['translator']['active_features']} | {pca['layers']['translator']['components_for_target']} |

## Autonomous dynamics

| Layer | PCs | Method | delay | rank | RMSE | skill vs persistence | skill vs training mean | correlation |
|---|---:|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(dynamics_rows)}

## Mode-resolved translator forecast

`oracle_pca` passes true retained PCA scores through inverse PCA. It measures
information retained by the reduced representation before temporal forecast
error is introduced.

| Method | Fourier band | mean amplitude / truth | amplitude correlation | coefficient NRMSE |
|---|---|---:|---:|---:|
{chr(10).join(band_rows)}

## 日本語メモ

未プール潜在テンソルをディスクに保存せず、forward中に方位角FFTを行った。
radial方向は{radial_bands}帯域に分け、各hidden channelについて
`n=0-{maximum_mode}`の複素Fourier係数を実部・虚部として保持した。

PCA、モデル選択、Hankel DMD/HAVOKの同定には20-24 usだけを使い、
24-30 usは完全な未学習区間である。`oracle_pca`でも高波数振幅が失われる
場合は、時間発展モデルではなく低次元状態設計がボトルネックである。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=latent.DEFAULT_DATA)
    parser.add_argument(
        "--workdir", type=Path, default=latent.DEFAULT_WORKDIR
    )
    parser.add_argument(
        "--config", type=Path, default=latent.DEFAULT_CONFIG
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--feature-path",
        type=Path,
        default=None,
        help="Reuse an existing Fourier feature H5 without copying it.",
    )
    parser.add_argument(
        "--physical",
        type=Path,
        default=hankel.DEFAULT_PHYSICAL,
        help="Case-matched physical metrics for HAVOK diagnostics.",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--radial-bands", type=int, default=8)
    parser.add_argument("--maximum-mode", type=int, default=21)
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--variance-target", type=float, default=0.95)
    parser.add_argument(
        "--feature-scaling",
        choices=("none", "mode_rms"),
        default="none",
        help="Equalize per-mode fluctuation RMS before PCA when requested.",
    )
    parser.add_argument(
        "--delays", type=int, nargs="+", default=[10, 20, 40, 80]
    )
    parser.add_argument(
        "--ranks", type=int, nargs="+", default=[8, 15, 20, 30]
    )
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help=(
            "Stop after writing the mode-aware latent feature cache. "
            "This is useful for transition trajectories outside the "
            "legacy 20--30 us dynamics-analysis interval."
        ),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    feature_path = (
        args.feature_path
        if args.feature_path is not None
        else args.output / "fourier_latent_features.h5"
    )

    if args.skip_extraction or args.feature_path is not None:
        if not feature_path.is_file():
            raise FileNotFoundError(feature_path)
        with h5py.File(feature_path, "r") as source:
            extraction = {
                "windows": int(len(source["window_start"])),
                "full_latent_shape": np.asarray(
                    source["full_latent_shape"]
                ).tolist(),
                "fourier_feature_shape": list(
                    source["encoder_fourier_ri"].shape[1:]
                ),
                "radial_boundaries": np.asarray(
                    source["radial_boundaries"]
                ).tolist(),
                "maximum_mode": int(
                    np.max(source["azimuthal_modes"])
                ),
                "output_h5": str(feature_path),
                "file_bytes": feature_path.stat().st_size,
            }
    else:
        device = latent.resolve_device(args.device)
        extraction = extract_fourier_latents(
            args.data,
            args.workdir,
            args.config,
            feature_path,
            device,
            args.batch_size,
            args.radial_bands,
            args.maximum_mode,
        )

    if args.extract_only:
        summary = {
            "status": "PASS",
            "extraction": extraction,
            "notes": [
                "The full latent grid was transformed in memory and not stored.",
                "PCA and the legacy 20--30 us dynamics analysis were not run.",
            ],
        }
        summary_path = args.output / "fourier_latent_extraction_summary.json"
        summary_path.write_text(
            json.dumps(
                reduced.json_safe(summary),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"[DONE] {feature_path}")
        return

    layers, pca_summary = fit_fourier_pca(
        feature_path,
        args.output,
        args.pca_components,
        args.variance_target,
        args.feature_scaling,
    )
    plot_pca_variance(
        args.output / "fourier_latent_pca_explained_variance.png",
        pca_summary,
        args.output,
    )
    plot_mode_energy(
        args.output / "fourier_latent_mode_energy.png", feature_path
    )
    dynamics_summary, dynamics_results = dynamics_analysis(
        layers, args.output, args.delays, args.ranks, args.physical
    )
    mode_forecast = evaluate_mode_forecasts(
        feature_path, args.output, layers, dynamics_results
    )
    summary = {
        "status": "PASS",
        "extraction": extraction,
        "pca": pca_summary,
        "dynamics": dynamics_summary,
        "mode_forecast": mode_forecast,
        "notes": [
            "The full latent grid was transformed in memory and not stored.",
            "PCA and dynamics fitting use only original SimVP training windows.",
            "No 24-30 us state was used for PCA, model or hyperparameter fitting.",
            "Complex Fourier coefficients were represented by real and imaginary parts.",
            f"PCA feature scaling: {args.feature_scaling}.",
        ],
    }
    (args.output / "fourier_latent_dynamics_summary.json").write_text(
        json.dumps(
            reduced.json_safe(summary),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_readme(
        args.output / "README.md",
        extraction,
        pca_summary,
        dynamics_summary,
        mode_forecast,
    )
    print(f"[DONE] {args.output}")


if __name__ == "__main__":
    main()
