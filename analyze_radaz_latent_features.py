#!/usr/bin/env python3
"""Extract and analyze latent states from the trained RadAz SimVP model."""

from __future__ import annotations

import argparse
import csv
import json
import runpy
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
CASE_NAME = "2D_RadAz_Xe1p_Bx20mT_Ez10kVm_dt15ps_out15ns"
CASE_ROOT = (
    ROOT.parent
    / "PEPAPIC"
    / "test"
    / "results"
    / "2D_Landmark"
    / CASE_NAME
    / CASE_NAME
)
DEFAULT_DATA = (
    CASE_ROOT
    / "SimVPv2_inputs"
    / "radaz_3ch_trainfixed_margin20_native257x256_pad260x256.h5"
)
DEFAULT_DIAGNOSTICS = (
    CASE_ROOT
    / "bifurcation_analysis_B20mT_E10kVm"
    / "bifurcation_diagnostics_uncompressed.h5"
)
DEFAULT_MODE_CSV = (
    CASE_ROOT
    / "bifurcation_analysis_B20mT_E10kVm"
    / "mode_band_time_series.csv"
)
DEFAULT_STATS_CSV = (
    CASE_ROOT
    / "bifurcation_analysis_B20mT_E10kVm"
    / "global_time_statistics.csv"
)
DEFAULT_FIELDS = CASE_ROOT / "analysis_fields_uncompressed.h5"
EXPERIMENT = (
    "radaz_xe1p_bx20mt_ez10kvm_out15ns_native257x256_"
    "direct10_trainfixed_disjoint_811_bs1_100ep"
)
DEFAULT_WORKDIR = ROOT / "workdirs" / EXPERIMENT
DEFAULT_CONFIG = (
    ROOT / "configs" / "custom" / "pepapic" / "SimVP_gSTA_radaz_direct.py"
)
DEFAULT_OUTPUT = ROOT / "workdirs" / "analyze_radaz_bx20mt_ez10kvm_latent"

PRE = 10
AFT = 10
B_T = 0.020
MTSI_MODES = np.arange(1, 7)
ECDI_MODES = np.arange(9, 22)
PHYSICAL_METRICS = (
    "phi_a_mtsi",
    "phi_a_ecdi",
    "phi_delta",
    "efy_a_mtsi",
    "efy_a_ecdi",
    "efy_delta",
    "electron_den_a_mtsi",
    "electron_den_a_ecdi",
    "electron_den_delta",
    "phi_std",
    "efy_rms",
    "electron_den_mean",
    "electron_den_std",
    "transport_total",
    "transport_mtsi",
    "transport_ecdi",
)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(name)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"[DEVICE] cuda: {torch.cuda.get_device_name(device)}")
    else:
        print("[DEVICE] cpu")
    return device


def checkpoint_path(workdir: Path) -> Path:
    for candidate in (
        workdir / "checkpoints" / "best.ckpt",
        workdir / "best.ckpt",
        workdir / "checkpoints" / "last.ckpt",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No checkpoint found under {workdir}")


def build_model(
    config_path: Path,
    checkpoint: Path,
    device: torch.device,
    in_shape: tuple[int, int, int, int],
) -> torch.nn.Module:
    from openstl.models.simvp_model import SimVP_Model

    cfg = runpy.run_path(str(config_path))
    model = SimVP_Model(
        in_shape=in_shape,
        hid_S=int(cfg.get("hid_S", 64)),
        hid_T=int(cfg.get("hid_T", 512)),
        N_S=int(cfg.get("N_S", 4)),
        N_T=int(cfg.get("N_T", 8)),
        model_type=cfg.get("model_type", "gSTA"),
        drop_path=float(cfg.get("drop_path", 0.0)),
        spatio_kernel_enc=int(cfg.get("spatio_kernel_enc", 3)),
        spatio_kernel_dec=int(cfg.get("spatio_kernel_dec", 3)),
        aft_seq_length=AFT,
        simvp_direct_aft_seq=bool(cfg.get("simvp_direct_aft_seq", True)),
    )
    loaded = torch.load(str(checkpoint), map_location="cpu")
    state = loaded["state_dict"] if "state_dict" in loaded else loaded
    state = {
        (key[6:] if key.startswith("model.") else key): value
        for key, value in state.items()
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    return model.to(device).eval()


def read_windows(dataset: h5py.Dataset, starts: np.ndarray) -> np.ndarray:
    first = int(starts[0])
    last = int(starts[-1]) + PRE
    block = np.asarray(dataset[first:last], dtype=np.float32)
    return np.stack(
        [block[int(start) - first : int(start) - first + PRE] for start in starts],
        axis=0,
    )


@torch.inference_mode()
def extract_latents(
    data_path: Path,
    workdir: Path,
    config_path: Path,
    output_path: Path,
    device: torch.device,
    batch_size: int,
    pool_size: int,
) -> dict:
    with h5py.File(data_path, "r") as source:
        dataset = source["data_tchw"]
        frames, channels, height, width = dataset.shape
        timesteps = np.asarray(source["timesteps"], dtype=np.int64)
        time_s = np.asarray(source["time_s"], dtype=np.float64)
        train_end = int(source["train_frame_end_exclusive"][()])
    starts = np.arange(0, frames - PRE - AFT + 1, dtype=np.int64)
    model = build_model(
        config_path,
        checkpoint_path(workdir),
        device,
        (PRE, channels, height, width),
    )

    encoder_batches = []
    translator_batches = []
    latent_shape = None
    with h5py.File(data_path, "r") as source:
        dataset = source["data_tchw"]
        for batch_start in range(0, len(starts), batch_size):
            batch_starts = starts[batch_start : batch_start + batch_size]
            x_np = read_windows(dataset, batch_starts)
            x = torch.from_numpy(x_np).to(device)
            batch, sequence, input_channels, input_h, input_w = x.shape
            encoded, _ = model.enc(
                x.reshape(batch * sequence, input_channels, input_h, input_w)
            )
            _, latent_channels, latent_h, latent_w = encoded.shape
            encoded = encoded.reshape(
                batch, sequence, latent_channels, latent_h, latent_w
            )
            translated = model.hid(encoded)
            if latent_shape is None:
                latent_shape = (
                    int(sequence),
                    int(latent_channels),
                    int(latent_h),
                    int(latent_w),
                )
                print(f"[LATENT] encoder/translator shape={latent_shape}")

            encoder_state = encoded[:, -1]
            translator_state = translated[:, 0]
            encoder_pooled = F.adaptive_avg_pool2d(
                encoder_state, (pool_size, pool_size)
            )
            translator_pooled = F.adaptive_avg_pool2d(
                translator_state, (pool_size, pool_size)
            )
            encoder_batches.append(encoder_pooled.cpu().numpy().astype(np.float32))
            translator_batches.append(
                translator_pooled.cpu().numpy().astype(np.float32)
            )
            completed = min(batch_start + batch_size, len(starts))
            if completed == len(starts) or completed % 50 == 0:
                print(
                    f"[EXTRACT] {completed}/{len(starts)} windows",
                    flush=True,
                )

    encoder = np.concatenate(encoder_batches, axis=0)
    translator = np.concatenate(translator_batches, axis=0)
    encoder_frame = starts + PRE - 1
    translator_frame = starts + PRE

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as target:
        compression = {"compression": "gzip", "compression_opts": 4, "shuffle": True}
        target.create_dataset("encoder_pooled", data=encoder, chunks=(1,) + encoder.shape[1:], **compression)
        target.create_dataset(
            "translator_pooled",
            data=translator,
            chunks=(1,) + translator.shape[1:],
            **compression,
        )
        target.create_dataset("window_start", data=starts)
        target.create_dataset("encoder_frame", data=encoder_frame)
        target.create_dataset("translator_frame", data=translator_frame)
        target.create_dataset("encoder_time_s", data=time_s[encoder_frame])
        target.create_dataset("translator_time_s", data=time_s[translator_frame])
        target.create_dataset("timesteps", data=timesteps)
        target.create_dataset("pool_shape", data=np.asarray([pool_size, pool_size]))
        target.create_dataset("full_latent_shape", data=np.asarray(latent_shape))
        target.create_dataset("train_frame_end_exclusive", data=train_end)
        target.attrs["source_data"] = str(data_path)
        target.attrs["source_workdir"] = str(workdir)
        target.attrs["checkpoint"] = str(checkpoint_path(workdir))
        target.attrs["config"] = str(config_path)
        target.attrs["encoder_alignment"] = "last true input frame"
        target.attrs["translator_alignment"] = "first predicted output frame"

    summary = {
        "windows": int(len(starts)),
        "input_shape": [PRE, channels, height, width],
        "full_latent_shape": list(latent_shape),
        "pooled_state_shape": list(encoder.shape[1:]),
        "encoder_alignment": "last true input frame",
        "translator_alignment": "first predicted output frame",
        "train_frame_end_exclusive": train_end,
        "output_h5": str(output_path),
    }
    print(f"[H5] {output_path}")
    return summary


def read_csv_columns(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows in {path}")
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in rows[0]
    }


def transport_time_series(fields_path: Path) -> dict[str, np.ndarray]:
    with h5py.File(fields_path, "r") as source:
        x_m = np.asarray(source["axes/x_m"], dtype=np.float64)
        frames = int(source["fields/electron_den"].shape[0])
        x_mask = (x_m >= 0.09e-2 - 1.0e-15) & (x_m <= 1.19e-2 + 1.0e-15)
        total = np.empty(frames, dtype=np.float64)
        mtsi = np.empty(frames, dtype=np.float64)
        ecdi = np.empty(frames, dtype=np.float64)
        chunk_size = 32
        for first in range(0, frames, chunk_size):
            last = min(first + chunk_size, frames)
            ne = np.asarray(
                source["fields/electron_den"][first:last, x_mask, :256],
                dtype=np.float64,
            )
            ey = np.asarray(
                source["fields/efy"][first:last, x_mask, :256],
                dtype=np.float64,
            )
            dne = ne - np.mean(ne, axis=-1, keepdims=True)
            dey = ey - np.mean(ey, axis=-1, keepdims=True)
            total[first:last] = -np.mean(dne * dey, axis=(-2, -1)) / B_T

            ne_fft = np.fft.rfft(dne, axis=-1, norm="forward")
            ey_fft = np.fft.rfft(dey, axis=-1, norm="forward")
            weights = np.full(ne_fft.shape[-1], 2.0, dtype=np.float64)
            weights[0] = 1.0
            weights[-1] = 1.0
            contribution = (
                -weights[None, None, :]
                * np.real(ne_fft * np.conj(ey_fft))
                / B_T
            )
            contribution = np.mean(contribution, axis=1)
            mtsi[first:last] = np.sum(contribution[:, MTSI_MODES], axis=-1)
            ecdi[first:last] = np.sum(contribution[:, ECDI_MODES], axis=-1)
            print(f"[TRANSPORT] {last}/{frames} frames", flush=True)
    return {
        "transport_total": total,
        "transport_mtsi": mtsi,
        "transport_ecdi": ecdi,
    }


def build_physical_metrics(
    mode_csv: Path,
    stats_csv: Path,
    fields_path: Path,
    output_csv: Path,
) -> dict[str, np.ndarray]:
    mode = read_csv_columns(mode_csv)
    stats = read_csv_columns(stats_csv)
    transport = transport_time_series(fields_path)
    frames = mode["frame"].astype(np.int64)
    if not np.array_equal(frames, stats["frame"].astype(np.int64)):
        raise ValueError("Mode and statistics frame indices do not match")

    metrics = {
        "frame": frames,
        "time_us": mode["time_us"],
    }
    for name in PHYSICAL_METRICS:
        if name in transport:
            metrics[name] = transport[name]
        elif name in mode:
            metrics[name] = mode[name]
        elif name in stats:
            metrics[name] = stats[name]
        else:
            raise KeyError(f"Missing physical metric {name}")

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        columns = ["frame", "time_us", *PHYSICAL_METRICS]
        writer.writerow(columns)
        for index in range(len(frames)):
            writer.writerow([metrics[column][index] for column in columns])
    print(f"[CSV] {output_csv}")
    return metrics


def component_count(cumulative: np.ndarray, threshold: float) -> str | int:
    indices = np.flatnonzero(cumulative >= threshold)
    return int(indices[0] + 1) if len(indices) else f">{len(cumulative)}"


def fit_pca_case(
    features: np.ndarray,
    fit_mask: np.ndarray,
    components: int,
    output_path: Path,
) -> tuple[PCA, np.ndarray, dict]:
    fit_values = np.asarray(features[fit_mask], dtype=np.float32)
    count = min(components, fit_values.shape[0] - 1, fit_values.shape[1])
    pca = PCA(n_components=count, svd_solver="randomized", random_state=0)
    pca.fit(fit_values)
    scores = pca.transform(features).astype(np.float32)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    np.savez_compressed(
        output_path,
        mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        explained_variance=pca.explained_variance_.astype(np.float64),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float64),
        singular_values=pca.singular_values_.astype(np.float64),
    )
    summary = {
        "fit_samples": int(np.sum(fit_mask)),
        "feature_count": int(features.shape[1]),
        "computed_components": int(count),
        "components_for_90_percent": component_count(cumulative, 0.90),
        "components_for_95_percent": component_count(cumulative, 0.95),
        "components_for_99_percent": component_count(cumulative, 0.99),
        "variance_pc1": float(pca.explained_variance_ratio_[0]),
        "variance_pc1_to_pc2": float(cumulative[min(1, len(cumulative) - 1)]),
        "variance_pc1_to_pc5": float(cumulative[min(4, len(cumulative) - 1)]),
        "variance_pc1_to_pc10": float(cumulative[min(9, len(cumulative) - 1)]),
        "variance_captured": float(cumulative[-1]),
    }
    return pca, scores, summary


def finite_corr(a: np.ndarray, b: np.ndarray, method: str) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if np.sum(mask) < 3 or np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return float("nan")
    if method == "pearson":
        return float(np.corrcoef(a[mask], b[mask])[0, 1])
    return float(spearmanr(a[mask], b[mask]).statistic)


def correlation_rows(
    scores: np.ndarray,
    label_frames: np.ndarray,
    metrics: dict[str, np.ndarray],
    analysis_mask: np.ndarray,
    layer: str,
    scope: str,
    pc_count: int,
) -> list[dict]:
    rows = []
    frames = label_frames[analysis_mask]
    for pc_index in range(min(pc_count, scores.shape[1])):
        pc = scores[analysis_mask, pc_index].astype(np.float64)
        for metric in PHYSICAL_METRICS:
            values = metrics[metric][frames]
            nonoverlap = np.arange(len(pc)) % PRE == 0
            rows.append(
                {
                    "layer": layer,
                    "scope": scope,
                    "pc": pc_index + 1,
                    "metric": metric,
                    "pearson": finite_corr(pc, values, "pearson"),
                    "spearman": finite_corr(pc, values, "spearman"),
                    "spearman_nonoverlap": finite_corr(
                        pc[nonoverlap], values[nonoverlap], "spearman"
                    ),
                    "delta_pearson": finite_corr(
                        np.diff(pc), np.diff(values), "pearson"
                    ),
                    "delta_spearman": finite_corr(
                        np.diff(pc), np.diff(values), "spearman"
                    ),
                    "samples": int(len(frames)),
                    "nonoverlap_samples": int(np.sum(nonoverlap)),
                }
            )
    return rows


def write_dict_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] {path}")


def plot_explained_variance(
    pca_results: dict[tuple[str, str], tuple[PCA, np.ndarray]],
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for axis, ((layer, scope), (pca, _)) in zip(axes.flat, pca_results.items()):
        count = min(50, len(pca.explained_variance_ratio_))
        x = np.arange(1, count + 1)
        cumulative = np.cumsum(pca.explained_variance_ratio_)[:count]
        axis.plot(x, cumulative, marker="o", markersize=3)
        axis.axhline(0.90, color="#737373", linestyle=":", label="90%")
        axis.axhline(0.95, color="#252525", linestyle="--", label="95%")
        axis.set_title(f"{layer}: {scope}")
        axis.set_ylim(0.0, 1.02)
        axis.set_xlabel("Number of principal components")
        axis.set_ylabel("Cumulative explained variance")
        axis.legend(loc="lower right")
    fig.suptitle("SimVP latent dimensionality")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories(
    pca_results: dict[tuple[str, str], tuple[PCA, np.ndarray]],
    layer_times: dict[str, np.ndarray],
    path: Path,
) -> None:
    fig = plt.figure(figsize=(14, 10), layout="constrained")
    grid = fig.add_gridspec(2, 3, width_ratios=(1.0, 1.0, 0.045))
    axes = np.asarray(
        [
            [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])],
            [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])],
        ]
    )
    color_axis = fig.add_subplot(grid[:, 2])
    scatter = None
    for axis, ((layer, scope), (_, scores)) in zip(axes.flat, pca_results.items()):
        time_us = layer_times[layer]
        mask = time_us >= 20.0 if scope == "steady" else np.ones(len(time_us), bool)
        scatter = axis.scatter(
            scores[mask, 0],
            scores[mask, 1],
            c=time_us[mask],
            cmap="viridis",
            s=8,
            alpha=0.8,
            rasterized=True,
        )
        axis.plot(
            scores[mask, 0],
            scores[mask, 1],
            color="#969696",
            linewidth=0.25,
            alpha=0.35,
        )
        axis.set_title(f"{layer}: {scope}")
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
    if scatter is not None:
        fig.colorbar(scatter, cax=color_axis, label="Physical time [us]")
    fig.suptitle("Latent trajectories projected onto PC1-PC2")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pc_time_series(
    scores: np.ndarray,
    time_us: np.ndarray,
    path: Path,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for index, axis in enumerate(axes):
        axis.plot(time_us, scores[:, index], linewidth=0.8)
        axis.axvline(20.0, color="#737373", linestyle=":", label="steady analysis start")
        axis.set_ylabel(f"PC{index + 1}")
        axis.legend(loc="lower right")
    axes[-1].set_xlabel("Physical time [us]")
    fig.suptitle("Encoder latent principal components (steady-state PCA basis)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(
    rows: list[dict],
    path: Path,
    value_column: str = "spearman",
    title_suffix: str = "Spearman correlation",
) -> None:
    fig = plt.figure(figsize=(16, 11), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.0, 0.035))
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0])]
    color_axis = fig.add_subplot(grid[:, 1])
    for axis, layer in zip(axes, ("encoder", "translator")):
        selected = [
            row
            for row in rows
            if row["layer"] == layer and row["scope"] == "steady" and row["pc"] <= 10
        ]
        matrix = np.full((10, len(PHYSICAL_METRICS)), np.nan, dtype=np.float64)
        for row in selected:
            matrix[
                int(row["pc"]) - 1, PHYSICAL_METRICS.index(row["metric"])
            ] = row[value_column]
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
        axis.set_yticks(np.arange(10), labels=[f"PC{i}" for i in range(1, 11)])
        axis.set_title(f"{layer}: {title_suffix}, 20-30 us")
    axes[-1].set_xticks(
        np.arange(len(PHYSICAL_METRICS)),
        labels=PHYSICAL_METRICS,
        rotation=55,
        ha="right",
    )
    fig.colorbar(image, cax=color_axis, label=title_suffix)
    fig.suptitle("Latent components versus physical instability diagnostics")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_score_csv(
    path: Path,
    starts: np.ndarray,
    layer_frames: dict[str, np.ndarray],
    layer_times: dict[str, np.ndarray],
    pca_results: dict[tuple[str, str], tuple[PCA, np.ndarray]],
    score_count: int = 10,
) -> None:
    columns = ["window_start"]
    for layer in ("encoder", "translator"):
        columns.extend([f"{layer}_frame", f"{layer}_time_us"])
        for scope in ("global", "steady"):
            columns.extend(
                [f"{layer}_{scope}_pc{i}" for i in range(1, score_count + 1)]
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for index, start in enumerate(starts):
            row = [int(start)]
            for layer in ("encoder", "translator"):
                row.extend(
                    [
                        int(layer_frames[layer][index]),
                        float(layer_times[layer][index]),
                    ]
                )
                for scope in ("global", "steady"):
                    scores = pca_results[(layer, scope)][1]
                    row.extend(
                        [float(value) for value in scores[index, :score_count]]
                    )
            writer.writerow(row)
    print(f"[CSV] {path}")


def analyze(
    latent_path: Path,
    mode_csv: Path,
    stats_csv: Path,
    fields_path: Path,
    output_dir: Path,
    pca_components: int,
) -> dict:
    with h5py.File(latent_path, "r") as source:
        source_data = Path(str(source.attrs.get("source_data", "")))
        source_workdir = Path(str(source.attrs.get("source_workdir", "")))
        analyzed_case = (
            source_data.parent.parent.name if source_data.name else CASE_NAME
        )
        analyzed_model = (
            source_workdir.name if source_workdir.name else EXPERIMENT
        )
        starts = np.asarray(source["window_start"], dtype=np.int64)
        features = {
            "encoder": np.asarray(source["encoder_pooled"], dtype=np.float32).reshape(
                len(starts), -1
            ),
            "translator": np.asarray(
                source["translator_pooled"], dtype=np.float32
            ).reshape(len(starts), -1),
        }
        layer_frames = {
            "encoder": np.asarray(source["encoder_frame"], dtype=np.int64),
            "translator": np.asarray(source["translator_frame"], dtype=np.int64),
        }
        layer_times = {
            "encoder": np.asarray(source["encoder_time_s"], dtype=np.float64) * 1.0e6,
            "translator": np.asarray(source["translator_time_s"], dtype=np.float64)
            * 1.0e6,
        }
        train_end = int(source["train_frame_end_exclusive"][()])

    metrics = build_physical_metrics(
        mode_csv,
        stats_csv,
        fields_path,
        output_dir / "physical_metrics_by_frame.csv",
    )
    pca_results: dict[tuple[str, str], tuple[PCA, np.ndarray]] = {}
    pca_summary = {}
    correlation = []

    for layer in ("encoder", "translator"):
        full_train_window = starts + PRE + AFT - 1 < train_end
        steady_train = full_train_window & (layer_times[layer] >= 20.0)
        masks = {"global": full_train_window, "steady": steady_train}
        for scope, fit_mask in masks.items():
            pca, scores, summary = fit_pca_case(
                features[layer],
                fit_mask,
                pca_components,
                output_dir / f"pca_{layer}_{scope}.npz",
            )
            pca_results[(layer, scope)] = (pca, scores)
            pca_summary[f"{layer}_{scope}"] = summary
            analysis_mask = (
                layer_times[layer] >= 20.0
                if scope == "steady"
                else np.ones(len(starts), dtype=bool)
            )
            correlation.extend(
                correlation_rows(
                    scores,
                    layer_frames[layer],
                    metrics,
                    analysis_mask,
                    layer,
                    scope,
                    pc_count=10,
                )
            )
            print(
                f"[PCA] {layer}/{scope}: "
                f"PC1={summary['variance_pc1']:.3f}, "
                f"PC1-10={summary['variance_pc1_to_pc10']:.3f}, "
                f"n95={summary['components_for_95_percent']}"
            )

    write_score_csv(
        output_dir / "latent_pca_scores.csv",
        starts,
        layer_frames,
        layer_times,
        pca_results,
    )
    write_dict_rows(output_dir / "latent_physics_correlations.csv", correlation)
    top_correlations = sorted(
        [
            row
            for row in correlation
            if row["scope"] == "steady" and np.isfinite(row["spearman"])
        ],
        key=lambda row: abs(row["spearman"]),
        reverse=True,
    )[:30]
    write_dict_rows(output_dir / "latent_top_physics_correlations.csv", top_correlations)
    top_dynamic_correlations = sorted(
        [
            row
            for row in correlation
            if row["scope"] == "steady" and np.isfinite(row["delta_spearman"])
        ],
        key=lambda row: abs(row["delta_spearman"]),
        reverse=True,
    )[:30]
    write_dict_rows(
        output_dir / "latent_top_dynamic_correlations.csv",
        top_dynamic_correlations,
    )

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    plot_explained_variance(
        pca_results, output_dir / "latent_pca_explained_variance.png"
    )
    plot_trajectories(
        pca_results, layer_times, output_dir / "latent_pca_trajectories.png"
    )
    plot_pc_time_series(
        pca_results[("encoder", "steady")][1],
        layer_times["encoder"],
        output_dir / "encoder_steady_pca_time_series.png",
    )
    plot_correlation_heatmap(
        correlation, output_dir / "latent_physics_correlation_heatmap.png"
    )
    plot_correlation_heatmap(
        correlation,
        output_dir / "latent_physics_delta_correlation_heatmap.png",
        value_column="delta_spearman",
        title_suffix="Delta Spearman correlation",
    )

    summary = {
        "status": "PASS",
        "case": analyzed_case,
        "model": analyzed_model,
        "source_data": str(source_data),
        "latent_h5": str(latent_path),
        "pca": pca_summary,
        "steady_interval_us": [20.0, 30.0],
        "top_steady_spearman_correlations": top_correlations[:15],
        "top_steady_delta_spearman_correlations": top_dynamic_correlations[:15],
        "notes": [
            "PCA is fitted only on windows fully inside the original training interval.",
            "Global PCA includes startup; steady PCA is fitted on training windows at t >= 20 us.",
            "Encoder state is aligned to the last true input frame.",
            "Translator state is aligned to the first predicted output frame.",
            "PCA dimensionality applies to 8x8 average-pooled latent grids, not the full latent tensor.",
        ],
    }
    (output_dir / "latent_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def write_readme(output_dir: Path, summary: dict) -> None:
    global_encoder = summary["pca"]["encoder_global"]
    steady_encoder = summary["pca"]["encoder_steady"]
    top = summary["top_steady_spearman_correlations"][:8]
    top_lines = "\n".join(
        f"| {row['layer']} PC{row['pc']} | {row['metric']} | {row['spearman']:.4f} |"
        for row in top
    )
    top_dynamic = summary["top_steady_delta_spearman_correlations"][:8]
    top_dynamic_lines = "\n".join(
        f"| {row['layer']} PC{row['pc']} | {row['metric']} | {row['delta_spearman']:.4f} |"
        for row in top_dynamic
    )
    text = f"""# RadAz SimVP latent feature analysis

Source model: `{summary['model']}`

Analyzed case: `{summary['case']}`

No SimVP retraining was performed. The trained model was frozen and its
encoder and gSTA-translator latent tensors were extracted from every direct10
window of the analyzed case.

The full latent tensor has shape `10 x 64 x 65 x 64`. For tractable PCA while
retaining spatial structure, the last encoder state and first translated state
were average-pooled to `64 x 8 x 8`.

## PCA summary

| Analysis | PC1 variance | PC1-PC10 variance | Components for 95% |
|---|---:|---:|---:|
| Encoder, startup included | {global_encoder['variance_pc1']:.4f} | {global_encoder['variance_pc1_to_pc10']:.4f} | {global_encoder['components_for_95_percent']} |
| Encoder, steady basis | {steady_encoder['variance_pc1']:.4f} | {steady_encoder['variance_pc1_to_pc10']:.4f} | {steady_encoder['components_for_95_percent']} |

The steady basis is fitted only to original-training windows at 20 us or
later, so startup relaxation does not create a false impression of low
dimensionality.

## Strongest steady-state associations

| Latent component | Physical diagnostic | Spearman correlation |
|---|---|---:|
{top_lines}

These are associations, not proof that a latent component is a unique physical
mode. Adjacent direct10 windows overlap strongly, so the number of rows is not
the number of statistically independent samples.

## Strongest frame-to-frame associations

| Latent component | Physical diagnostic | Delta Spearman correlation |
|---|---|---:|
{top_dynamic_lines}

The delta correlation compares frame-to-frame changes. It is stricter than a
raw correlation because a shared slow trend cannot create a high value by
itself. The complete table also records correlations from non-overlapping
10-frame windows.

## Files

- `radaz_latent_features.h5`: reusable pooled encoder/translator features
- `latent_pca_scores.csv`: PC scores aligned to physical frame and time
- `physical_metrics_by_frame.csv`: mode, field, and transport diagnostics
- `latent_physics_correlations.csv`: all PC-to-physics correlations
- `latent_top_physics_correlations.csv`: strongest steady-state correlations
- `latent_top_dynamic_correlations.csv`: strongest frame-to-frame correlations
- `latent_pca_explained_variance.png`
- `latent_pca_trajectories.png`
- `encoder_steady_pca_time_series.png`
- `latent_physics_correlation_heatmap.png`
- `latent_physics_delta_correlation_heatmap.png`
- `pca_*.npz`: reusable PCA bases
- `latent_analysis_summary.json`

## 日本語

学習済みSimVPを固定し、再学習せずにencoderとgSTA時間発展器の潜在特徴を
抽出した解析です。PCAは立ち上がりを含む全期間用と、20 us以降の準定常状態用を
分けています。ここで得られる相関は、潜在成分と物理モードの対応候補であり、
因果関係や一意な物理解釈を直接証明するものではありません。
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--mode-csv", type=Path, default=DEFAULT_MODE_CSV)
    parser.add_argument("--stats-csv", type=Path, default=DEFAULT_STATS_CSV)
    parser.add_argument("--fields", type=Path, default=DEFAULT_FIELDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--pool-size", type=int, default=8)
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--skip-extraction", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    latent_path = args.output / "radaz_latent_features.h5"
    extraction = None
    if not args.skip_extraction:
        extraction = extract_latents(
            args.data,
            args.workdir,
            args.config,
            latent_path,
            resolve_device(args.device),
            args.batch_size,
            args.pool_size,
        )
    elif not latent_path.is_file():
        raise FileNotFoundError(f"--skip-extraction requested but missing {latent_path}")

    summary = analyze(
        latent_path,
        args.mode_csv,
        args.stats_csv,
        args.fields,
        args.output,
        args.pca_components,
    )
    if extraction is not None:
        summary["extraction"] = extraction
        (args.output / "latent_analysis_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    write_readme(args.output, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[PASS] output={args.output}")


if __name__ == "__main__":
    main()
