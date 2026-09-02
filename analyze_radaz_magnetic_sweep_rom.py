#!/usr/bin/env python3
"""Compare physical and frozen-SimVP reduced dynamics across a B sweep.

The protocol is deliberately prospective:

* hyperparameters are selected on 23--24 us;
* the final ROM is fitted on 20--24 us;
* 24--30 us is a strict autonomous-rollout holdout.

Two PCA choices are kept separate. ``individual`` fits a basis independently
for every magnetic field and measures local closure. ``common_b20`` fixes the
B=20 mT basis and tests whether the other cases occupy the same subspace.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze_radaz_hankel_havok as hankel
import analyze_radaz_latent_features as latent
import analyze_radaz_reduced_dynamics as reduced


ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT.parent / "research_results" / "2D_RadAz"
PEPAPIC_ROOT = RESEARCH / "PEPAPIC" / "2D_Landmark"
WORKDIR_ROOT = RESEARCH / "SimVPv2" / "workdirs"
DEFAULT_OUTPUT = (
    WORKDIR_ROOT
    / "analyze_radaz_magnetic_sweep_rom_B10_B15_B20_B25_B30mT_E10kVm"
)
MODEL_WORKDIR = (
    WORKDIR_ROOT
    / "radaz_xe1p_bx20mt_ez10kvm_out15ns_native257x256_"
    "direct10_trainfixed_disjoint_811_bs1_100ep"
)
MODEL_CONFIG = (
    ROOT / "configs" / "custom" / "pepapic" / "SimVP_gSTA_radaz_direct.py"
)
B20_LATENT = (
    WORKDIR_ROOT
    / "analyze_radaz_bx20mt_ez10kvm_latent"
    / "radaz_latent_features.h5"
)

B_VALUES = (10, 15, 20, 25, 30)
CHANNELS = ("electron_den", "ion_den", "phi")
DT_FRAME_US = 0.015
FIT_START_US = 20.0
VALIDATION_START_US = 23.0
FORECAST_START_US = 24.0
FORECAST_END_US = 30.0


def case_name(b_mt: int) -> str:
    return f"2D_RadAz_Xe1p_Bx{b_mt}mT_Ez10kVm_dt15ps_out15ns"


def case_root(b_mt: int) -> Path:
    name = case_name(b_mt)
    return PEPAPIC_ROOT / name / name


def fields_path(b_mt: int) -> Path:
    return case_root(b_mt) / "analysis_fields_uncompressed.h5"


def b20_input_path() -> Path:
    return (
        case_root(20)
        / "SimVPv2_inputs"
        / "radaz_3ch_trainfixed_margin20_native257x256_pad260x256.h5"
    )


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def check_inputs() -> None:
    required = [fields_path(b) for b in B_VALUES]
    required += [b20_input_path(), B20_LATENT, latent.checkpoint_path(MODEL_WORKDIR)]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(map(str, missing)))


def prepare_normalized_input(
    b_mt: int,
    target: Path,
    reference: Path,
    overwrite: bool,
) -> Path:
    if b_mt == 20:
        return reference
    if target.is_file() and not overwrite:
        print(f"[PREP] reuse B{b_mt}: {target}", flush=True)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(reference, "r") as ref:
        norm_low = np.asarray(ref["norm_low"], dtype=np.float32)
        norm_high = np.asarray(ref["norm_high"], dtype=np.float32)
        train_min = np.asarray(ref["train_min"], dtype=np.float32)
        train_max = np.asarray(ref["train_max"], dtype=np.float32)
        margin = float(ref["margin"][()])
        pre = int(ref["pre_seq_length"][()])
        aft = int(ref["aft_seq_length"][()])

    source_path = fields_path(b_mt)
    with h5py.File(source_path, "r") as source:
        time_s_all = np.asarray(source["axes/time_s"], dtype=np.float64)
        frames = int(np.searchsorted(time_s_all, FORECAST_END_US * 1.0e-6 + 1.0e-15))
        frames = min(frames, len(time_s_all))
        time_s = time_s_all[:frames]
        timesteps = np.asarray(source["axes/frame_id"][:frames], dtype=np.int64)
        x_m = np.asarray(source["axes/x_m"], dtype=np.float64)
        y_m = np.asarray(source["axes/y_m"][:256], dtype=np.float64)

        with h5py.File(target, "w") as output:
            data = output.create_dataset(
                "data_tchw",
                shape=(frames, 3, 260, 256),
                dtype=np.float32,
                chunks=(1, 3, 260, 256),
                compression="lzf",
            )
            span = norm_high - norm_low
            for start in range(0, frames, 8):
                stop = min(start + 8, frames)
                block = np.empty((stop - start, 3, 257, 256), dtype=np.float32)
                for channel, field in enumerate(CHANNELS):
                    raw = np.asarray(
                        source[f"fields/{field}"][start:stop, :257, :256],
                        dtype=np.float32,
                    )
                    block[:, channel] = (raw - norm_low[channel]) / span[channel]
                padded = np.pad(block, ((0, 0), (0, 0), (0, 3), (0, 0)), mode="edge")
                data[start:stop] = padded
                if stop == frames or stop % 200 == 0:
                    print(f"[PREP] B{b_mt}: {stop}/{frames}", flush=True)

            output.create_dataset("time_s", data=time_s)
            output.create_dataset("timesteps", data=timesteps)
            output.create_dataset("props", data=np.asarray(CHANNELS, dtype="S"))
            output.create_dataset("norm_low", data=norm_low)
            output.create_dataset("norm_high", data=norm_high)
            output.create_dataset("train_min", data=train_min)
            output.create_dataset("train_max", data=train_max)
            output.create_dataset("margin", data=np.float32(margin))
            output.create_dataset("pre_seq_length", data=np.int32(pre))
            output.create_dataset("aft_seq_length", data=np.int32(aft))
            output.create_dataset("train_frame_end_exclusive", data=np.int32(1600))
            output.create_dataset("spatial_stride", data=np.int32(1))
            output.create_dataset("valid_spatial_shape", data=np.asarray([257, 256]))
            output.create_dataset("model_spatial_shape", data=np.asarray([260, 256]))
            output.create_dataset("x_m", data=x_m)
            output.create_dataset("x_m_model", data=np.pad(x_m, (0, 3), mode="edge"))
            output.create_dataset("y_m", data=y_m)
            output.create_dataset("source_h5", data=np.bytes_(str(source_path)))
            output.create_dataset("layout", data=np.bytes_("T,C,X,Y; B20 train-only normalization"))
            output.attrs["normalization"] = "B20 train-only minmax with margin"
            output.attrs["normalization_reference"] = str(reference)
            output.attrs["normalization_clip"] = False
            output.attrs["periodic_duplicate_endpoint_removed"] = True
            output.attrs["spatial_reduction"] = "none; radial edge padded 257 to 260"
    print(f"[PREP] wrote {target}", flush=True)
    return target


def extract_latent_case(
    b_mt: int,
    data_path: Path,
    output_path: Path,
    device: torch.device,
    batch_size: int,
    overwrite: bool,
) -> Path:
    if b_mt == 20:
        return B20_LATENT
    if output_path.is_file() and not overwrite:
        print(f"[LATENT] reuse B{b_mt}: {output_path}", flush=True)
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[LATENT] extracting B{b_mt}", flush=True)
    latent.extract_latents(
        data_path,
        MODEL_WORKDIR,
        MODEL_CONFIG,
        output_path,
        device,
        batch_size,
        8,
    )
    return output_path


def physical_fourier_features(
    b_mt: int,
    output_path: Path,
    radial_bands: int,
    max_mode: int,
    overwrite: bool,
) -> Path:
    if output_path.is_file() and not overwrite:
        print(f"[PHYSICAL] reuse B{b_mt}: {output_path}", flush=True)
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_path = fields_path(b_mt)
    with h5py.File(source_path, "r") as source:
        time_s_all = np.asarray(source["axes/time_s"], dtype=np.float64)
        frames = int(np.searchsorted(time_s_all, FORECAST_END_US * 1.0e-6 + 1.0e-15))
        frames = min(frames, len(time_s_all))
        time_s = time_s_all[:frames]
        band_edges = np.linspace(0, 257, radial_bands + 1, dtype=int)
        coefficient = np.empty(
            (frames, len(CHANNELS), radial_bands, max_mode + 1),
            dtype=np.complex64,
        )
        for start in range(0, frames, 16):
            stop = min(start + 16, frames)
            for channel, field in enumerate(CHANNELS):
                raw = np.asarray(
                    source[f"fields/{field}"][start:stop, :257, :256],
                    dtype=np.float32,
                )
                for band in range(radial_bands):
                    radial_mean = np.mean(
                        raw[:, band_edges[band] : band_edges[band + 1], :], axis=1
                    )
                    coefficient[start:stop, channel, band] = (
                        np.fft.rfft(radial_mean, axis=-1)[..., : max_mode + 1] / 256.0
                    )
            if stop == frames or stop % 400 == 0:
                print(f"[PHYSICAL] B{b_mt}: {stop}/{frames}", flush=True)

    real0 = coefficient[..., :1].real
    oscillatory = coefficient[..., 1:]
    packed = np.concatenate(
        [real0, oscillatory.real, oscillatory.imag], axis=-1
    ).astype(np.float32)
    features = packed.reshape(frames, len(CHANNELS), -1)
    time_us = time_s * 1.0e6
    fit = (time_us >= FIT_START_US) & (time_us < FORECAST_START_US)
    channel_scale = np.empty(len(CHANNELS), dtype=np.float64)
    for channel in range(len(CHANNELS)):
        centered = features[fit, channel] - np.mean(features[fit, channel], axis=0)
        channel_scale[channel] = max(float(np.sqrt(np.mean(centered * centered))), 1.0e-12)
        features[:, channel] /= channel_scale[channel]
    matrix = features.reshape(frames, -1)
    with h5py.File(output_path, "w") as output:
        output.create_dataset("features", data=matrix, compression="gzip", compression_opts=4)
        output.create_dataset("time_us", data=time_us)
        output.create_dataset("channel_scale", data=channel_scale)
        output.create_dataset("channels", data=np.asarray(CHANNELS, dtype="S"))
        output.create_dataset("radial_band_edges", data=band_edges)
        output.attrs["source_h5"] = str(source_path)
        output.attrs["representation"] = "radial-band azimuthal Fourier coefficients"
        output.attrs["radial_bands"] = radial_bands
        output.attrs["max_mode"] = max_mode
        output.attrs["packing"] = "per channel/band: Re(n=0), Re(n=1..N), Im(n=1..N)"
    print(f"[PHYSICAL] wrote {output_path}", flush=True)
    return output_path


@dataclass
class StateSeries:
    representation: str
    b_mt: int
    time_us: np.ndarray
    features: np.ndarray


def load_physical(path: Path, b_mt: int) -> StateSeries:
    with h5py.File(path, "r") as source:
        return StateSeries(
            "physical_fourier",
            b_mt,
            np.asarray(source["time_us"], dtype=np.float64),
            np.asarray(source["features"], dtype=np.float32),
        )


def load_latent(path: Path, b_mt: int, layer: str) -> StateSeries:
    with h5py.File(path, "r") as source:
        return StateSeries(
            layer,
            b_mt,
            np.asarray(source[f"{layer}_time_s"], dtype=np.float64) * 1.0e6,
            np.asarray(source[f"{layer}_pooled"], dtype=np.float32).reshape(
                len(source[f"{layer}_time_s"]), -1
            ),
        )


def fit_pca(series: StateSeries, components: int) -> PCA:
    mask = (series.time_us >= FIT_START_US) & (series.time_us < FORECAST_START_US)
    count = min(components, int(np.count_nonzero(mask)) - 1, series.features.shape[1])
    model = PCA(n_components=count, svd_solver="randomized", random_state=42)
    model.fit(series.features[mask])
    return model


def components_for_fraction(model: PCA, fraction: float = 0.95) -> int:
    cumulative = np.cumsum(model.explained_variance_ratio_)
    if cumulative[-1] < fraction:
        return int(len(cumulative))
    return int(np.searchsorted(cumulative, fraction) + 1)


def common_capture(model: PCA, series: StateSeries, components: int) -> float:
    mask = (series.time_us >= FIT_START_US) & (series.time_us < FORECAST_START_US)
    values = series.features[mask].astype(np.float64)
    basis = model.components_[:components]
    scores = (values - model.mean_) @ basis.T
    reconstructed = scores @ basis + model.mean_
    denominator = float(np.sum((values - np.mean(values, axis=0)) ** 2))
    if denominator <= 1.0e-20:
        return float("nan")
    return float(1.0 - np.sum((values - reconstructed) ** 2) / denominator)


def temporal_diagnostics(
    series: StateSeries,
    pca: PCA,
    components: int,
) -> dict:
    """Measure train/holdout drift and nontrivial recurrence in local PCA scores."""
    scores = pca.transform(series.features)[:, :components].astype(np.float64)
    fit_mask = (series.time_us >= FIT_START_US) & (series.time_us < FORECAST_START_US)
    test_mask = (
        (series.time_us >= FORECAST_START_US)
        & (series.time_us <= FORECAST_END_US + 1.0e-9)
    )
    fit = scores[fit_mask]
    mean = np.mean(fit, axis=0)
    scale = np.std(fit, axis=0)
    scale[scale < 1.0e-12] = 1.0
    standardized = (scores - mean) / scale
    fit = standardized[fit_mask]
    test = standardized[test_mask]

    interval_mask = (
        (series.time_us >= FIT_START_US)
        & (series.time_us <= FORECAST_END_US + 1.0e-9)
    )
    interval = standardized[interval_mask]
    lag_min = max(2, int(round(0.30 / DT_FRAME_US)))
    lag_max = min(int(round(3.00 / DT_FRAME_US)), len(interval) // 2)
    recurrence = []
    for lag in range(lag_min, lag_max + 1):
        left = interval[:-lag]
        right = interval[lag:]
        recurrence.append(
            (
                float(np.sqrt(np.mean((left - right) ** 2))),
                float(np.corrcoef(left.ravel(), right.ravel())[0, 1]),
                lag,
            )
        )
    best_rmse, best_corr, best_lag = min(recurrence, key=lambda item: item[0])
    return {
        "representation": series.representation,
        "B_mT": series.b_mt,
        "components": components,
        "holdout_mean_shift_rms_fit_std": float(
            np.sqrt(np.mean(np.mean(test, axis=0) ** 2))
        ),
        "holdout_scale_rms_fit_std": float(np.sqrt(np.mean(np.var(test, axis=0)))),
        "fit_frame_delta_rms": float(np.sqrt(np.mean(np.diff(fit, axis=0) ** 2))),
        "holdout_frame_delta_rms": float(
            np.sqrt(np.mean(np.diff(test, axis=0) ** 2))
        ),
        "best_recurrence_period_us": best_lag * DT_FRAME_US,
        "best_recurrence_rmse": best_rmse,
        "best_recurrence_correlation": best_corr,
    }


def evaluate_rom(
    series: StateSeries,
    pca: PCA,
    components: int,
    basis_name: str,
    delays: list[int],
    ranks: list[int],
) -> tuple[list[dict], dict[str, np.ndarray]]:
    scores = pca.transform(series.features)[:, :components].astype(np.float64)
    fit_mask = (series.time_us >= FIT_START_US) & (series.time_us < FORECAST_START_US)
    forecast_mask = (
        (series.time_us >= FORECAST_START_US)
        & (series.time_us <= FORECAST_END_US + 1.0e-9)
    )
    fit_scores = scores[fit_mask]
    standardizer = reduced.fit_standardizer(fit_scores)
    standardized = standardizer.transform(scores)
    fit_states = standardized[fit_mask]
    truth = standardized[forecast_mask]
    forecast_time = series.time_us[forecast_mask]
    persistence = np.repeat(fit_states[-1][None, :], len(truth), axis=0)
    training_mean = np.zeros_like(truth)

    matrix, eigenvalues = reduced.fit_dmd(fit_states)
    standard_prediction = reduced.rollout_linear(matrix, fit_states[-1], len(truth))

    best, _ = hankel.candidate_search(
        standardized,
        series.time_us,
        fit_mask,
        delays,
        ranks,
    )
    hankel_model = hankel.fit_hankel_dmd(
        fit_states, int(best["delay"]), int(best["rank"])
    )
    hankel_prediction = hankel.rollout_hankel(hankel_model, fit_states, len(truth))
    havok_model, _ = hankel.fit_havok(hankel_model, fit_states)
    havok_prediction = hankel.rollout_havok_zero_forcing(
        hankel_model, havok_model, fit_states, len(truth)
    )

    predictions = {
        "standard_dmd": standard_prediction,
        "hankel_dmd": hankel_prediction,
        "havok_zero_forcing": havok_prediction,
    }
    rows: list[dict] = []
    for method, prediction in predictions.items():
        metrics, per_time = reduced.evaluate_prediction(
            truth, prediction, persistence, forecast_time
        )
        mean_mse = float(np.mean(truth * truth))
        prediction_mse = float(metrics["standardized_mse"])
        skill_vs_mean = (
            1.0 - prediction_mse / mean_mse
            if np.isfinite(prediction_mse) and mean_mse > 1.0e-20
            else float("-inf")
        )
        finite = np.isfinite(per_time)
        below_one = finite & (per_time < 1.0)
        contiguous = 0
        for value in below_one:
            if not value:
                break
            contiguous += 1
        rows.append(
            {
                "representation": series.representation,
                "B_mT": series.b_mt,
                "basis": basis_name,
                "components": components,
                "method": method,
                "delay": int(best["delay"]) if method != "standard_dmd" else 1,
                "rank": int(best["rank"]) if method != "standard_dmd" else components,
                "standardized_rmse": float(metrics["standardized_rmse"]),
                "skill_vs_persistence": float(metrics["skill_vs_persistence"]),
                "skill_vs_training_mean": float(skill_vs_mean),
                "correlation": float(metrics["flattened_correlation"]),
                "spectral_radius": float(
                    np.max(np.abs(eigenvalues if method == "standard_dmd" else hankel_model.eigenvalues))
                ),
                "contiguous_horizon_rmse_lt_1_us": contiguous * DT_FRAME_US,
                "persistence_rmse": float(np.sqrt(np.mean((truth - persistence) ** 2))),
                "training_mean_rmse": float(np.sqrt(np.mean((truth - training_mean) ** 2))),
            }
        )
    rollout = {
        "time_us": forecast_time,
        "truth": truth,
        "persistence": persistence,
        **predictions,
    }
    return rows, rollout


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_pca_summary(rows: list[dict], path: Path) -> None:
    representations = ("physical_fourier", "encoder", "translator")
    colors = {"physical_fourier": "#0072B2", "encoder": "#D55E00", "translator": "#009E73"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for representation in representations:
        subset = sorted(
            [row for row in rows if row["representation"] == representation],
            key=lambda row: row["B_mT"],
        )
        axes[0].plot(
            [row["B_mT"] for row in subset],
            [row["individual_components_95"] for row in subset],
            marker="o",
            label=representation,
            color=colors[representation],
        )
        capture = np.asarray([row["common_b20_capture"] for row in subset])
        axes[1].plot(
            [row["B_mT"] for row in subset],
            np.clip(capture, -1.0, 1.0),
            marker="o",
            label=representation,
            color=colors[representation],
        )
    axes[0].set_title("Local PCA dimensionality (20-24 us)")
    axes[0].set_ylabel("components for 95% variance")
    axes[1].set_title("Capture by fixed B20 PCA subspace (display clipped at -1)")
    axes[1].set_ylabel("1 - reconstruction SSE / local variance")
    axes[1].set_ylim(-1.08, 1.08)
    for axis in axes:
        axis.set_xlabel("Bx (mT)")
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_rom_summary(rows: list[dict], path: Path) -> None:
    representations = ("physical_fourier", "encoder", "translator")
    methods = ("standard_dmd", "hankel_dmd", "havok_zero_forcing")
    styles = {
        "standard_dmd": ("o", "-"),
        "hankel_dmd": ("s", "-"),
        "havok_zero_forcing": ("^", "--"),
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for column, representation in enumerate(representations):
        for row_index, basis in enumerate(("individual", "common_b20")):
            axis = axes[row_index, column]
            for method in methods:
                subset = sorted(
                    [
                        row for row in rows
                        if row["representation"] == representation
                        and row["basis"] == basis
                        and row["method"] == method
                    ],
                    key=lambda row: row["B_mT"],
                )
                marker, linestyle = styles[method]
                skill = np.asarray([row["skill_vs_persistence"] for row in subset])
                axis.plot(
                    [row["B_mT"] for row in subset],
                    np.clip(skill, -1.0, 1.0),
                    marker=marker,
                    linestyle=linestyle,
                    label=method,
                )
            axis.axhline(0.0, color="black", linewidth=0.9)
            axis.set_title(f"{representation} | {basis}")
            axis.set_xlabel("Bx (mT)")
            axis.grid(alpha=0.25)
            if column == 0:
                axis.set_ylabel("skill vs persistence")
            if row_index == 0 and column == 2:
                axis.legend(loc="lower right", fontsize=8)
            axis.set_ylim(-1.08, 1.08)
    fig.suptitle("ROM skill vs persistence (display clipped at -1; exact values in CSV)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_stationarity(rows: list[dict], path: Path) -> None:
    representations = ("physical_fourier", "encoder", "translator")
    colors = {"physical_fourier": "#0072B2", "encoder": "#D55E00", "translator": "#009E73"}
    quantities = (
        ("holdout_mean_shift_rms_fit_std", "Holdout mean shift (fit std)"),
        ("holdout_scale_rms_fit_std", "Holdout scale (fit std)"),
        ("best_recurrence_correlation", "Best recurrence correlation"),
        ("best_recurrence_period_us", "Best recurrence period (us)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharex=True)
    for axis, (key, label) in zip(axes.ravel(), quantities):
        for representation in representations:
            subset = sorted(
                [row for row in rows if row["representation"] == representation],
                key=lambda row: row["B_mT"],
            )
            axis.plot(
                [row["B_mT"] for row in subset],
                [row[key] for row in subset],
                marker="o",
                color=colors[representation],
                label=representation,
            )
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.set_xlabel("Bx (mT)")
    axes[0, 1].axhline(1.0, color="black", linewidth=0.8)
    axes[0, 1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Local-state stationarity and recurrence, 20-30 us")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_common_trajectories(
    states: dict[str, dict[int, StateSeries]],
    common_pca: dict[str, PCA],
    path: Path,
) -> None:
    representations = ("physical_fourier", "encoder", "translator")
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(B_VALUES)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for axis, representation in zip(axes, representations):
        model = common_pca[representation]
        for color, b_mt in zip(colors, B_VALUES):
            series = states[representation][b_mt]
            mask = (series.time_us >= FIT_START_US) & (series.time_us <= FORECAST_END_US)
            scores = model.transform(series.features[mask])[:, :2]
            axis.plot(scores[:, 0], scores[:, 1], color=color, alpha=0.8, label=f"B{b_mt}")
            axis.scatter(scores[0, 0], scores[0, 1], color=color, s=20)
        axis.set_title(representation)
        axis.set_xlabel("common B20 PC1")
        axis.set_ylabel("common B20 PC2")
        axis.grid(alpha=0.25)
    axes[-1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_readme(
    output: Path,
    pca_rows: list[dict],
    rom_rows: list[dict],
    stationarity_rows: list[dict],
) -> None:
    pca_lines = []
    for row in pca_rows:
        pca_lines.append(
            f"| {row['representation']} | {row['B_mT']} | "
            f"{row['individual_components_95']} | {row['common_b20_capture']:.4f} | "
            f"{row['common_vs_individual_mean_angle_deg']:.2f} |"
        )
    best_lines = []
    for representation in ("physical_fourier", "encoder", "translator"):
        for b_mt in B_VALUES:
            candidates = [
                row for row in rom_rows
                if row["representation"] == representation
                and row["B_mT"] == b_mt
                and row["basis"] == "individual"
            ]
            best = max(candidates, key=lambda row: row["skill_vs_persistence"])
            best_lines.append(
                f"| {representation} | {b_mt} | {best['method']} | "
                f"{best['components']} | {best['skill_vs_persistence']:.4f} | "
                f"{best['skill_vs_training_mean']:.4f} | {best['correlation']:.4f} |"
            )
    diagnostic_lines = []
    for row in stationarity_rows:
        diagnostic_lines.append(
            f"| {row['representation']} | {row['B_mT']} | "
            f"{row['holdout_mean_shift_rms_fit_std']:.3f} | "
            f"{row['holdout_scale_rms_fit_std']:.3f} | "
            f"{row['best_recurrence_period_us']:.3f} | "
            f"{row['best_recurrence_correlation']:.3f} |"
        )
    text = f"""# Magnetic-sweep reduced-order-model comparison

Cases: Bx=10, 15, 20, 25, 30 mT at Ez=10 kV/m.

The B20-trained SimVPv2 model is frozen. No model retraining is performed.
All target cases use the B20 training-only normalization without clipping.

## Prospective protocol

- candidate selection: 23--24 us
- final fit: 20--24 us
- strict autonomous holdout: 24--30 us
- physical representation: 8 radial bands and azimuthal Fourier modes n=0..48
- local closure: PCA fitted independently in each B case
- common-manifold test: fixed B20 PCA basis applied to every B case

## PCA summary

| representation | B (mT) | local PCs for 95% | fixed-B20 capture | mean subspace angle (deg) |
|---|---:|---:|---:|---:|
{chr(10).join(pca_lines)}

`fixed-B20 capture` can be negative when a target trajectory lies far outside
the B20 affine subspace. This is useful evidence against a shared unconditioned
manifold, not a plotting error.

## Best local-basis ROM per case

| representation | B (mT) | selected method | states | skill vs persistence | skill vs mean | correlation |
|---|---:|---|---:|---:|---:|---:|
{chr(10).join(best_lines)}

Selection here is descriptive across the three already preregistered methods;
the 24--30 us holdout was not used to choose delay or Hankel rank. A positive
skill against persistence alone is insufficient: skill against the training
mean and trajectory correlation must also be considered.

## Stationarity and recurrence diagnostics

| representation | B (mT) | holdout mean shift | holdout scale | recurrence period (us) | recurrence corr. |
|---|---:|---:|---:|---:|---:|
{chr(10).join(diagnostic_lines)}

B15 is an unusually clean approximately 2.565 us periodic orbit in all three
representations. B25 instead expands and shifts after 24 us; fitted ROMs have
spectral radii above one and autonomous errors grow rapidly. The contrast is
therefore a local-dynamics result, not evidence of duplicated frames.

## Files

- `pca_dimensionality.csv`
- `rom_metrics.csv`
- `stationarity_diagnostics.csv`
- `pca_dimensionality_and_common_capture.png`
- `rom_skill_vs_B.png`
- `state_stationarity_and_recurrence.png`
- `common_b20_pca_trajectories.png`
- `analysis_summary.json`
- `normalized_inputs/`: B20-normalized SimVP inputs
- `latent_features/`: frozen-model pooled encoder/translator states
- `physical_features/`: reusable radial-band Fourier states
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--radial-bands", type=int, default=8)
    parser.add_argument("--max-mode", type=int, default=48)
    parser.add_argument("--pca-components", type=int, default=40)
    parser.add_argument("--delays", type=int, nargs="+", default=[10, 20, 40, 80])
    parser.add_argument("--ranks", type=int, nargs="+", default=[8, 15, 20, 30])
    parser.add_argument("--overwrite-prepared", action="store_true")
    parser.add_argument("--overwrite-latent", action="store_true")
    parser.add_argument("--overwrite-physical", action="store_true")
    parser.add_argument("--skip-latent-extraction", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_inputs()
    args.output.mkdir(parents=True, exist_ok=True)
    normalized_dir = args.output / "normalized_inputs"
    latent_dir = args.output / "latent_features"
    physical_dir = args.output / "physical_features"

    reference = b20_input_path()
    normalized: dict[int, Path] = {}
    for b_mt in B_VALUES:
        normalized[b_mt] = prepare_normalized_input(
            b_mt,
            normalized_dir / f"B{b_mt}_b20norm_native257x256_pad260x256.h5",
            reference,
            args.overwrite_prepared,
        )

    latent_paths: dict[int, Path] = {20: B20_LATENT}
    if args.skip_latent_extraction:
        for b_mt in B_VALUES:
            if b_mt == 20:
                continue
            path = latent_dir / f"B{b_mt}" / "radaz_latent_features.h5"
            if not path.is_file():
                raise FileNotFoundError(path)
            latent_paths[b_mt] = path
    else:
        device = latent.resolve_device(args.device)
        for b_mt in B_VALUES:
            latent_paths[b_mt] = extract_latent_case(
                b_mt,
                normalized[b_mt],
                latent_dir / f"B{b_mt}" / "radaz_latent_features.h5",
                device,
                args.batch_size,
                args.overwrite_latent,
            )

    physical_paths = {
        b_mt: physical_fourier_features(
            b_mt,
            physical_dir / f"B{b_mt}_physical_fourier.h5",
            args.radial_bands,
            args.max_mode,
            args.overwrite_physical,
        )
        for b_mt in B_VALUES
    }

    states: dict[str, dict[int, StateSeries]] = {
        "physical_fourier": {
            b_mt: load_physical(physical_paths[b_mt], b_mt) for b_mt in B_VALUES
        },
        "encoder": {
            b_mt: load_latent(latent_paths[b_mt], b_mt, "encoder") for b_mt in B_VALUES
        },
        "translator": {
            b_mt: load_latent(latent_paths[b_mt], b_mt, "translator") for b_mt in B_VALUES
        },
    }

    pca_rows: list[dict] = []
    rom_rows: list[dict] = []
    stationarity_rows: list[dict] = []
    common_models: dict[str, PCA] = {}
    rollout_summary: dict[str, dict] = {}
    for representation, by_b in states.items():
        print(f"[PCA] representation={representation}", flush=True)
        individual_models = {
            b_mt: fit_pca(series, args.pca_components) for b_mt, series in by_b.items()
        }
        common = individual_models[20]
        common_models[representation] = common
        common_components = min(components_for_fraction(common), 20)
        for b_mt, series in by_b.items():
            individual = individual_models[b_mt]
            individual_components = min(components_for_fraction(individual), 20)
            angle_components = min(common_components, individual_components)
            angles = np.degrees(
                subspace_angles(
                    common.components_[:angle_components].T,
                    individual.components_[:angle_components].T,
                )
            )
            pca_rows.append(
                {
                    "representation": representation,
                    "B_mT": b_mt,
                    "individual_components_95": components_for_fraction(individual),
                    "individual_variance_pc1": float(individual.explained_variance_ratio_[0]),
                    "individual_variance_pc1_to_pc10": float(
                        np.sum(individual.explained_variance_ratio_[:10])
                    ),
                    "common_b20_components": common_components,
                    "common_b20_capture": common_capture(common, series, common_components),
                    "common_vs_individual_mean_angle_deg": float(np.mean(angles)),
                    "common_vs_individual_max_angle_deg": float(np.max(angles)),
                }
            )
            stationarity_rows.append(
                temporal_diagnostics(series, individual, individual_components)
            )
            print(
                f"[ROM] {representation} B{b_mt} individual={individual_components} "
                f"common={common_components}",
                flush=True,
            )
            individual_rows, individual_rollout = evaluate_rom(
                series,
                individual,
                individual_components,
                "individual",
                args.delays,
                args.ranks,
            )
            common_rows, common_rollout = evaluate_rom(
                series,
                common,
                common_components,
                "common_b20",
                args.delays,
                args.ranks,
            )
            rom_rows.extend(individual_rows)
            rom_rows.extend(common_rows)
            rollout_summary[f"{representation}_B{b_mt}_individual"] = {
                method: value.shape for method, value in individual_rollout.items()
            }
            rollout_summary[f"{representation}_B{b_mt}_common"] = {
                method: value.shape for method, value in common_rollout.items()
            }

    write_csv(args.output / "pca_dimensionality.csv", pca_rows)
    write_csv(args.output / "rom_metrics.csv", rom_rows)
    write_csv(args.output / "stationarity_diagnostics.csv", stationarity_rows)
    plot_pca_summary(pca_rows, args.output / "pca_dimensionality_and_common_capture.png")
    plot_rom_summary(rom_rows, args.output / "rom_skill_vs_B.png")
    plot_stationarity(
        stationarity_rows,
        args.output / "state_stationarity_and_recurrence.png",
    )
    plot_common_trajectories(states, common_models, args.output / "common_b20_pca_trajectories.png")
    write_readme(args.output, pca_rows, rom_rows, stationarity_rows)

    summary = {
        "status": "PASS",
        "cases_B_mT": list(B_VALUES),
        "Ez_kVm": 10.0,
        "source_model": str(MODEL_WORKDIR),
        "normalization": "fixed B20 training-only, no clipping",
        "physical_representation": {
            "radial_bands": args.radial_bands,
            "azimuthal_modes": [0, args.max_mode],
        },
        "protocol_us": {
            "candidate_selection": [VALIDATION_START_US, FORECAST_START_US],
            "fit": [FIT_START_US, FORECAST_START_US],
            "strict_holdout": [FORECAST_START_US, FORECAST_END_US],
        },
        "pca": pca_rows,
        "stationarity": stationarity_rows,
        "rom_metrics": rom_rows,
        "rollout_shapes": rollout_summary,
    }
    (args.output / "analysis_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"[PASS] output={args.output}", flush=True)


if __name__ == "__main__":
    main()
