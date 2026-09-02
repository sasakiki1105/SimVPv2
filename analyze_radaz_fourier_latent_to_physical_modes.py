"""Map E25 Fourier latent states to physical Fourier observables."""

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


ROOT = Path(__file__).resolve().parent
CASE_NAME = "2D_RadAz_Xe1p_Bx20mT_Ez25kVm_dt15ps_out15ns"
DEFAULT_PHYSICAL = (
    ROOT.parent
    / "PEPAPIC"
    / "test"
    / "results"
    / "2D_Landmark"
    / CASE_NAME
    / CASE_NAME
    / "analysis_fields_uncompressed.h5"
)
DEFAULT_LATENT = (
    ROOT / "workdirs" / "analyze_radaz_bx20mt_ez25kvm_fourier_latent_raw"
)
DEFAULT_OUTPUT = (
    ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_ez25kvm_fourier_latent_to_physical_modes"
)

FIELDS = ("phi", "electron_den", "ion_den", "efy")
FIELD_LABELS = {
    "phi": "phi",
    "electron_den": "electron density",
    "ion_den": "ion density",
    "efy": "Ey",
}
MODE_BANDS = {
    "MTSI": np.arange(1, 7, dtype=np.int64),
    "ECDI": np.arange(9, 22, dtype=np.int64),
}
METHODS = (
    "oracle_pca",
    "latent_persistence",
    "standard_dmd",
    "hankel_dmd",
    "havok_zero_forcing",
    "physical_copy",
)
PLOT_METHODS = (
    "oracle_pca",
    "hankel_dmd",
    "havok_zero_forcing",
    "physical_copy",
)
COLORS = {
    "truth": "#111111",
    "oracle_pca": "#777777",
    "latent_persistence": "#cc79a7",
    "standard_dmd": "#56b4e9",
    "hankel_dmd": "#0072b2",
    "havok_zero_forcing": "#009e73",
    "physical_copy": "#e69f00",
}
RIDGES = (0.0, 1.0e-6, 1.0e-4, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0)
VARIANTS = ("linear", "quadratic")
FIT_START_US = 20.0
VALIDATION_START_US = 23.0
FORECAST_START_US = 24.0
FORECAST_END_US = 30.0
B_T = 0.020


@dataclass
class ObservationMap:
    variant: str
    ridge: float
    z_mean: np.ndarray
    z_scale: np.ndarray
    feature_mean: np.ndarray
    target_mean: np.ndarray
    weights: np.ndarray

    def predict(self, scores: np.ndarray) -> np.ndarray:
        features = feature_matrix(
            scores, self.z_mean, self.z_scale, self.variant
        )
        return (features - self.feature_mean) @ self.weights + self.target_mean


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


def load_latent_scores(
    latent_dir: Path, components: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(
        latent_dir / "fourier_latent_pca_scores.h5", "r"
    ) as handle:
        scores = np.asarray(
            handle["translator_scores"][:, :components], dtype=np.float64
        )
        time_us = np.asarray(
            handle["translator_time_us"], dtype=np.float64
        )
    with h5py.File(
        latent_dir / "fourier_latent_features.h5", "r"
    ) as handle:
        frames = np.asarray(handle["translator_frame"], dtype=np.int64)
    if len(scores) != len(frames):
        raise ValueError("Latent score/frame length mismatch")
    return frames, time_us, scores


def physical_radial_groups(nx: int, bands: int) -> list[np.ndarray]:
    groups = [
        np.asarray(group, dtype=np.int64)
        for group in np.array_split(np.arange(nx, dtype=np.int64), bands)
    ]
    if any(len(group) < 2 for group in groups):
        raise ValueError("Physical radial groups are too small")
    return groups


def extract_physical_fourier(
    source_path: Path,
    output_path: Path,
    frames: np.ndarray,
    time_us: np.ndarray,
    bands: int,
    maximum_mode: int,
) -> np.ndarray:
    if output_path.is_file():
        with h5py.File(output_path, "r") as handle:
            cached_frames = np.asarray(handle["frame"], dtype=np.int64)
            if np.array_equal(cached_frames, frames):
                print(f"[PHYSICAL] reuse {output_path}")
                return np.asarray(handle["coefficients"], dtype=np.complex64)
    if not np.all(np.diff(frames) == 1):
        raise ValueError("Physical target frames must be contiguous")
    with h5py.File(source_path, "r") as source:
        sample = source[f"fields/{FIELDS[0]}"]
        _, nx, ny_with_duplicate = sample.shape
        ny = ny_with_duplicate - 1
        if maximum_mode > ny // 2:
            raise ValueError("maximum_mode exceeds physical Nyquist mode")
        groups = physical_radial_groups(nx, bands)
        coefficients = np.empty(
            (len(frames), len(FIELDS), bands, maximum_mode + 1),
            dtype=np.complex64,
        )
        first_frame = int(frames[0])
        chunk_size = 32
        for field_index, field in enumerate(FIELDS):
            dataset = source[f"fields/{field}"]
            for local_start in range(0, len(frames), chunk_size):
                local_stop = min(len(frames), local_start + chunk_size)
                source_start = first_frame + local_start
                source_stop = first_frame + local_stop
                block = np.asarray(
                    dataset[source_start:source_stop, :, :ny],
                    dtype=np.float64,
                )
                for band_index, group in enumerate(groups):
                    radial_mean = np.mean(block[:, group, :], axis=1)
                    transformed = np.fft.rfft(
                        radial_mean, axis=-1, norm="forward"
                    )
                    coefficients[
                        local_start:local_stop, field_index, band_index
                    ] = transformed[:, : maximum_mode + 1]
            print(f"[PHYSICAL] extracted {field}", flush=True)
        x_m = np.asarray(source["axes/x_m"], dtype=np.float64)
        radial_edges = np.asarray(
            [x_m[group[0]] for group in groups]
            + [x_m[groups[-1][-1]]],
            dtype=np.float64,
        )
        radial_weights = np.asarray(
            [len(group) for group in groups], dtype=np.float64
        )
        radial_weights /= np.sum(radial_weights)

    temporary = output_path.with_suffix(".partial.h5")
    if temporary.exists():
        temporary.unlink()
    with h5py.File(temporary, "w") as target:
        target.create_dataset(
            "coefficients",
            data=coefficients,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        target.create_dataset("frame", data=frames)
        target.create_dataset("time_us", data=time_us)
        target.create_dataset("radial_edges_m", data=radial_edges)
        target.create_dataset("radial_weights", data=radial_weights)
        target.create_dataset(
            "modes", data=np.arange(maximum_mode + 1, dtype=np.int64)
        )
        target.create_dataset(
            "fields", data=np.asarray(FIELDS, dtype=h5py.string_dtype())
        )
        target.attrs["source_h5"] = str(source_path)
        target.attrs["azimuthal_duplicate_endpoint_removed"] = True
    temporary.replace(output_path)
    print(
        f"[PHYSICAL] saved {output_path} "
        f"({output_path.stat().st_size / 1e6:.1f} MB)"
    )
    return coefficients


def feature_matrix(
    scores: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    variant: str,
) -> np.ndarray:
    z = (np.asarray(scores, dtype=np.float64) - mean) / scale
    if variant == "linear":
        return z
    if variant != "quadratic":
        raise ValueError(variant)
    columns = [z]
    for left in range(z.shape[1]):
        columns.append(z[:, left : left + 1] * z[:, left:])
    return np.concatenate(columns, axis=1)


def fit_observation_map(
    scores: np.ndarray,
    targets: np.ndarray,
    variant: str,
    ridge: float,
) -> ObservationMap:
    z_mean = np.mean(scores, axis=0)
    z_scale = np.std(scores, axis=0, ddof=1)
    z_scale[z_scale <= 1.0e-12] = 1.0
    features = feature_matrix(scores, z_mean, z_scale, variant)
    feature_mean = np.mean(features, axis=0)
    target_mean = np.mean(targets, axis=0)
    x = features - feature_mean
    y = targets - target_mean
    gram = x.T @ x
    gram.flat[:: gram.shape[0] + 1] += ridge
    weights = np.linalg.pinv(gram, rcond=1.0e-12) @ (x.T @ y)
    return ObservationMap(
        variant,
        ridge,
        z_mean,
        z_scale,
        feature_mean,
        target_mean,
        weights,
    )


def target_normalization(
    coefficients: np.ndarray, fit_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(coefficients[fit_mask], axis=0)
    centered = coefficients - mean[None, ...]
    # One scale per field and mode keeps radial structure while preventing
    # low-amplitude ECDI coefficients from disappearing in a global MSE.
    mode_scale = np.sqrt(
        np.mean(np.abs(centered[fit_mask]) ** 2, axis=(0, 2))
    )
    positive = mode_scale[mode_scale > 0.0]
    floor = (
        float(np.median(positive)) * 1.0e-8 if positive.size else 1.0
    )
    mode_scale = np.maximum(mode_scale, floor)
    scale = np.broadcast_to(
        mode_scale[:, None, :], coefficients.shape[1:]
    )
    normalized = centered / scale[None, ...]
    return normalized, mean, scale


def flatten_complex(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(len(values), -1)
    return np.concatenate((flat.real, flat.imag), axis=1)


def unflatten_complex(
    values: np.ndarray, shape: tuple[int, int, int]
) -> np.ndarray:
    half = values.shape[1] // 2
    complex_flat = values[:, :half] + 1j * values[:, half:]
    return complex_flat.reshape((len(values),) + shape)


def select_observation_map(
    scores: np.ndarray,
    normalized_targets: np.ndarray,
    time_us: np.ndarray,
) -> tuple[str, float, list[dict]]:
    subtrain = (time_us >= FIT_START_US) & (
        time_us < VALIDATION_START_US
    )
    validation = (time_us >= VALIDATION_START_US) & (
        time_us < FORECAST_START_US
    )
    target_flat = flatten_complex(normalized_targets)
    rows = []
    best = None
    for variant in VARIANTS:
        for ridge in RIDGES:
            model = fit_observation_map(
                scores[subtrain], target_flat[subtrain], variant, ridge
            )
            prediction = model.predict(scores[validation])
            error = prediction - target_flat[validation]
            mse = float(np.mean(error * error))
            row = {
                "variant": variant,
                "ridge": ridge,
                "validation_normalized_coefficient_mse": mse,
            }
            rows.append(row)
            if best is None or mse < best[0]:
                best = (mse, variant, ridge)
            print(
                f"[MAP] {variant} ridge={ridge:g} validation MSE={mse:.6e}"
            )
    assert best is not None
    return best[1], best[2], rows


def read_rollout_scores(
    path: Path,
    components: int,
    all_scores: np.ndarray,
    all_time_us: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["layer"] == "translator"
        ]
    forecast_time = np.asarray(
        [float(row["time_us"]) for row in rows], dtype=np.float64
    )
    fit_mask = (all_time_us >= FIT_START_US) & (
        all_time_us < FORECAST_START_US
    )
    mean = np.mean(all_scores[fit_mask], axis=0)
    scale = np.std(all_scores[fit_mask], axis=0, ddof=1)
    scale[scale <= 1.0e-12] = 1.0

    def inverse(prefix: str) -> np.ndarray:
        standardized = np.asarray(
            [
                [
                    float(row[f"{prefix}_pc{component + 1}"])
                    for component in range(components)
                ]
                for row in rows
            ],
            dtype=np.float64,
        )
        return standardized * scale + mean

    score_predictions = {
        "oracle_pca": inverse("truth"),
        "latent_persistence": inverse("persistence"),
        "standard_dmd": inverse("standard_dmd"),
        "hankel_dmd": inverse("hankel_dmd"),
        "havok_zero_forcing": inverse("havok_zero_forcing"),
    }
    return forecast_time, score_predictions


def match_times(source: np.ndarray, requested: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(source, requested)
    indices = np.clip(indices, 0, len(source) - 1)
    previous = np.maximum(indices - 1, 0)
    use_previous = np.abs(source[previous] - requested) < np.abs(
        source[indices] - requested
    )
    indices[use_previous] = previous[use_previous]
    if np.max(np.abs(source[indices] - requested)) > 1.0e-8:
        raise ValueError("Could not align forecast and physical times")
    return indices


def amplitude(values: np.ndarray, modes: np.ndarray, field: int) -> np.ndarray:
    selected = np.take(values[:, field], modes, axis=-1)
    return np.sqrt(np.mean(np.abs(selected) ** 2, axis=(1, 2)))


def coefficient_nrmse(
    truth: np.ndarray, prediction: np.ndarray
) -> float:
    rmse = float(np.sqrt(np.mean(np.abs(prediction - truth) ** 2)))
    scale = float(
        np.sqrt(
            np.mean(
                np.abs(truth - np.mean(truth, axis=0, keepdims=True)) ** 2
            )
        )
    )
    return rmse / max(scale, np.finfo(float).tiny)


def weighted_phase_mae(
    truth_cross: np.ndarray, prediction_cross: np.ndarray
) -> float:
    phase_delta = np.angle(prediction_cross) - np.angle(truth_cross)
    error = np.abs((phase_delta + np.pi) % (2.0 * np.pi) - np.pi)
    weight = np.abs(truth_cross)
    denominator = float(np.sum(weight))
    if denominator <= np.finfo(float).tiny:
        return float("nan")
    return float(np.sum(error * weight) / denominator)


def transport_series(
    coefficients: np.ndarray,
    modes: np.ndarray,
    radial_weights: np.ndarray,
) -> np.ndarray:
    electron = np.take(
        coefficients[:, FIELDS.index("electron_den")], modes, axis=-1
    )
    efy = np.take(coefficients[:, FIELDS.index("efy")], modes, axis=-1)
    modal = -2.0 * np.real(electron * np.conj(efy)) / B_T
    per_band = np.sum(modal, axis=2)
    return np.einsum("b,tb->t", radial_weights, per_band, optimize=True)


def evaluate_predictions(
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    radial_weights: np.ndarray,
) -> tuple[list[dict], list[dict], dict]:
    field_rows = []
    physics_rows = []
    series: dict = {"truth": {}}
    for band, modes in MODE_BANDS.items():
        series["truth"][f"transport_{band}"] = transport_series(
            truth, modes, radial_weights
        )
        for field_index, field in enumerate(FIELDS):
            series["truth"][f"{field}_{band}_amplitude"] = amplitude(
                truth, modes, field_index
            )

    copy_mse: dict[tuple[str, str], float] = {}
    copy_transport_mse: dict[str, float] = {}
    for method, prediction in predictions.items():
        series[method] = {}
        for band, modes in MODE_BANDS.items():
            for field_index, field in enumerate(FIELDS):
                truth_selected = np.take(
                    truth[:, field_index], modes, axis=-1
                )
                pred_selected = np.take(
                    prediction[:, field_index], modes, axis=-1
                )
                truth_amp = series["truth"][f"{field}_{band}_amplitude"]
                pred_amp = amplitude(prediction, modes, field_index)
                series[method][f"{field}_{band}_amplitude"] = pred_amp
                mse = float(np.mean(np.abs(pred_selected - truth_selected) ** 2))
                if method == "physical_copy":
                    copy_mse[(field, band)] = mse
                field_rows.append(
                    {
                        "method": method,
                        "field": field,
                        "band": band,
                        "coefficient_nrmse": coefficient_nrmse(
                            truth_selected, pred_selected
                        ),
                        "coefficient_mse_skill_vs_physical_copy": None,
                        "amplitude_correlation": safe_correlation(
                            truth_amp, pred_amp
                        ),
                        "mean_amplitude_ratio": float(
                            np.mean(pred_amp)
                            / max(float(np.mean(truth_amp)), np.finfo(float).tiny)
                        ),
                        "coefficient_mse": mse,
                    }
                )

            electron_index = FIELDS.index("electron_den")
            efy_index = FIELDS.index("efy")
            truth_cross = (
                np.take(truth[:, electron_index], modes, axis=-1)
                * np.conj(np.take(truth[:, efy_index], modes, axis=-1))
            )
            prediction_cross = (
                np.take(prediction[:, electron_index], modes, axis=-1)
                * np.conj(
                    np.take(prediction[:, efy_index], modes, axis=-1)
                )
            )
            truth_transport = series["truth"][f"transport_{band}"]
            prediction_transport = transport_series(
                prediction, modes, radial_weights
            )
            series[method][f"transport_{band}"] = prediction_transport
            transport_mse = float(
                np.mean((prediction_transport - truth_transport) ** 2)
            )
            if method == "physical_copy":
                copy_transport_mse[band] = transport_mse
            physics_rows.append(
                {
                    "method": method,
                    "band": band,
                    "cross_phase_mae_rad": weighted_phase_mae(
                        truth_cross, prediction_cross
                    ),
                    "transport_correlation": safe_correlation(
                        truth_transport, prediction_transport
                    ),
                    "transport_nrmse": float(
                        np.sqrt(transport_mse)
                        / max(
                            float(np.std(truth_transport, ddof=1)),
                            np.finfo(float).tiny,
                        )
                    ),
                    "transport_mse_skill_vs_physical_copy": None,
                    "transport_mse": transport_mse,
                }
            )

    for row in field_rows:
        baseline = copy_mse[(row["field"], row["band"])]
        row["coefficient_mse_skill_vs_physical_copy"] = (
            1.0 - row["coefficient_mse"] / baseline
            if baseline > 0.0
            else float("nan")
        )
    for row in physics_rows:
        baseline = copy_transport_mse[row["band"]]
        row["transport_mse_skill_vs_physical_copy"] = (
            1.0 - row["transport_mse"] / baseline
            if baseline > 0.0
            else float("nan")
        )
    return field_rows, physics_rows, series


def plot_amplitudes(
    path: Path, time_us: np.ndarray, series: dict
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    for row, field in enumerate(("phi", "electron_den", "efy")):
        for column, band in enumerate(MODE_BANDS):
            axis = axes[row, column]
            key = f"{field}_{band}_amplitude"
            axis.plot(
                time_us,
                series["truth"][key],
                color=COLORS["truth"],
                linewidth=2.0,
                label="truth",
            )
            for method in PLOT_METHODS:
                axis.plot(
                    time_us,
                    series[method][key],
                    color=COLORS[method],
                    linewidth=1.2,
                    label=method,
                )
            axis.set_title(f"{FIELD_LABELS[field]}: {band}")
            axis.set_ylabel("physical Fourier RMS")
            axis.grid(alpha=0.2)
            axis.legend(loc="upper right", fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("time [us]")
    fig.suptitle("Physical mode amplitudes decoded from Fourier latent states")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_transport(
    path: Path, time_us: np.ndarray, series: dict
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for axis, band in zip(axes, MODE_BANDS):
        key = f"transport_{band}"
        axis.plot(
            time_us,
            series["truth"][key],
            color=COLORS["truth"],
            linewidth=2.0,
            label="truth",
        )
        for method in PLOT_METHODS:
            axis.plot(
                time_us,
                series[method][key],
                color=COLORS[method],
                linewidth=1.2,
                label=method,
            )
        axis.set_title(f"{band} modal transport proxy")
        axis.set_ylabel("-2 Re(ne Ey*) / B")
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [us]")
    fig.suptitle("Transport derived from decoded physical Fourier coefficients")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_predictions(
    path: Path,
    time_us: np.ndarray,
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("time_us", data=time_us)
        handle.create_dataset("truth", data=truth, compression="gzip")
        for method, values in predictions.items():
            handle.create_dataset(method, data=values, compression="gzip")
        handle.create_dataset(
            "fields", data=np.asarray(FIELDS, dtype=h5py.string_dtype())
        )
        handle.create_dataset(
            "modes", data=np.arange(truth.shape[-1], dtype=np.int64)
        )


def write_readme(
    path: Path,
    variant: str,
    ridge: float,
    components: int,
    field_rows: list[dict],
    physics_rows: list[dict],
) -> None:
    field_lookup = {
        (row["method"], row["field"], row["band"]): row
        for row in field_rows
    }
    physics_lookup = {
        (row["method"], row["band"]): row for row in physics_rows
    }
    table_rows = []
    for method in (
        "oracle_pca",
        "hankel_dmd",
        "havok_zero_forcing",
        "physical_copy",
    ):
        for band in MODE_BANDS:
            phi = field_lookup[(method, "phi", band)]
            physics = physics_lookup[(method, band)]
            table_rows.append(
                "| {method} | {band} | {phi_corr:.4f} | {phase:.4f} | "
                "{transport_corr:.4f} | {transport_skill:.4f} |".format(
                    method=method,
                    band=band,
                    phi_corr=phi["amplitude_correlation"],
                    phase=physics["cross_phase_mae_rad"],
                    transport_corr=physics["transport_correlation"],
                    transport_skill=physics[
                        "transport_mse_skill_vs_physical_copy"
                    ],
                )
            )
    text = f"""# E25 Fourier latent to physical modes

The raw Fourier translator state was reduced to `{components}` PCs. A
train-only `{variant}` observation map with ridge `{ridge:g}` maps those PCs
to complex physical Fourier coefficients for `phi`, electron density, ion
density and `Ey`.

- Observation-map subtrain: `20-23 us`
- Model/ridge selection: `23-24 us`
- Final map fit: `20-24 us`
- Autonomous evaluation: `24-30 us`
- Spatial representation: 8 radial bands, azimuthal modes `n=0-21`

`oracle_pca` uses the true future PCA scores and is not a forecast. It is the
information ceiling of the reduced latent state plus observation map.

| Method | band | phi amplitude correlation | cross-phase MAE [rad] | transport correlation | transport skill vs copy |
|---|---|---:|---:|---:|---:|
{chr(10).join(table_rows)}

## 日本語メモ

未プール潜在Fourier状態が保持している情報を、物理場の複素Fourier係数へ
直接戻した。全画像を復元してから再度FFTしたのではないため、高波数modeが
画素MSEに埋もれにくい評価である。

Oracleが悪ければ12PC状態または観測写像に物理情報がない。Oracleが良く
Hankel/HAVOKだけが悪ければ、主な問題は時間発展モデルにある。cross-phaseと
輸送は観測写像の直接出力ではなく、予測した電子密度とEyの複素係数から
独立に計算した。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical", type=Path, default=DEFAULT_PHYSICAL)
    parser.add_argument("--latent-dir", type=Path, default=DEFAULT_LATENT)
    parser.add_argument(
        "--feature-path",
        type=Path,
        default=None,
        help="Fourier latent feature H5 used by --extract-only.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--components", type=int, default=12)
    parser.add_argument("--radial-bands", type=int, default=8)
    parser.add_argument("--maximum-mode", type=int, default=21)
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help=(
            "Extract case-matched physical Fourier targets without fitting "
            "the legacy E25 observation map."
        ),
    )
    parser.add_argument(
        "--all-physical-frames",
        action="store_true",
        help=(
            "With --extract-only, use every frame in --physical instead of "
            "taking aligned frames from a latent feature cache."
        ),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.extract_only:
        if args.all_physical_frames:
            feature_path = None
            with h5py.File(args.physical, "r") as handle:
                time_us = np.asarray(
                    handle["axes/time_s"], dtype=np.float64
                ) * 1.0e6
                frames = np.arange(len(time_us), dtype=np.int64)
        else:
            feature_path = (
                args.feature_path
                if args.feature_path is not None
                else args.latent_dir / "fourier_latent_features.h5"
            )
            with h5py.File(feature_path, "r") as handle:
                frames = np.asarray(handle["translator_frame"], dtype=np.int64)
                time_us = np.asarray(
                    handle["translator_time_s"], dtype=np.float64
                ) * 1.0e6
        physical_cache = args.output / "physical_fourier_targets.h5"
        coefficients = extract_physical_fourier(
            args.physical,
            physical_cache,
            frames,
            time_us,
            args.radial_bands,
            args.maximum_mode,
        )
        summary = {
            "status": "PASS",
            "source_h5": str(args.physical.resolve()),
            "feature_h5": (
                str(feature_path.resolve()) if feature_path is not None else None
            ),
            "output_h5": str(physical_cache.resolve()),
            "frames": int(len(frames)),
            "first_time_us": float(time_us[0]),
            "last_time_us": float(time_us[-1]),
            "coefficient_shape": list(coefficients.shape),
            "radial_bands": int(args.radial_bands),
            "maximum_mode": int(args.maximum_mode),
        }
        (args.output / "physical_fourier_extraction_summary.json").write_text(
            json.dumps(json_safe(summary), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[DONE] {physical_cache}")
        return

    frames, time_us, scores = load_latent_scores(
        args.latent_dir, args.components
    )
    physical_cache = args.output / "physical_fourier_targets.h5"
    coefficients = extract_physical_fourier(
        args.physical,
        physical_cache,
        frames,
        time_us,
        args.radial_bands,
        args.maximum_mode,
    )
    fit_mask = (time_us >= FIT_START_US) & (
        time_us < FORECAST_START_US
    )
    normalized, target_mean, target_scale = target_normalization(
        coefficients, fit_mask
    )
    variant, ridge, selection_rows = select_observation_map(
        scores, normalized, time_us
    )
    write_csv(args.output / "observation_map_selection.csv", selection_rows)
    target_flat = flatten_complex(normalized)
    observation = fit_observation_map(
        scores[fit_mask], target_flat[fit_mask], variant, ridge
    )

    forecast_time, score_predictions = read_rollout_scores(
        args.latent_dir / "fourier_hankel_havok_rollout.csv",
        args.components,
        scores,
        time_us,
    )
    forecast_indices = match_times(time_us, forecast_time)
    truth = coefficients[forecast_indices]
    predictions = {}
    target_shape = coefficients.shape[1:]
    for method, method_scores in score_predictions.items():
        normalized_prediction = unflatten_complex(
            observation.predict(method_scores), target_shape
        )
        predictions[method] = (
            normalized_prediction * target_scale[None, ...]
            + target_mean[None, ...]
        ).astype(np.complex64)
    last_fit = np.flatnonzero(fit_mask)[-1]
    predictions["physical_copy"] = np.repeat(
        coefficients[last_fit : last_fit + 1], len(truth), axis=0
    )

    with h5py.File(physical_cache, "r") as handle:
        radial_weights = np.asarray(
            handle["radial_weights"], dtype=np.float64
        )
    field_rows, physics_rows, series = evaluate_predictions(
        truth, predictions, radial_weights
    )
    write_csv(args.output / "physical_mode_field_metrics.csv", field_rows)
    write_csv(args.output / "physical_mode_physics_metrics.csv", physics_rows)
    plot_amplitudes(
        args.output / "physical_mode_amplitude_rollouts.png",
        forecast_time,
        series,
    )
    plot_transport(
        args.output / "physical_mode_transport_rollouts.png",
        forecast_time,
        series,
    )
    save_predictions(
        args.output / "decoded_physical_fourier_rollouts.h5",
        forecast_time,
        truth,
        predictions,
    )

    summary = {
        "status": "PASS",
        "case": CASE_NAME,
        "latent_components": args.components,
        "observation_map": {"variant": variant, "ridge": ridge},
        "fit_interval_us": [FIT_START_US, FORECAST_START_US],
        "forecast_interval_us": [
            float(forecast_time[0]),
            float(forecast_time[-1]),
        ],
        "physical_fields": FIELDS,
        "radial_bands": args.radial_bands,
        "azimuthal_modes": [0, args.maximum_mode],
        "field_metrics": field_rows,
        "physics_metrics": physics_rows,
    }
    (args.output / "physical_mode_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_readme(
        args.output / "README.md",
        variant,
        ridge,
        args.components,
        field_rows,
        physics_rows,
    )
    print(f"[DONE] {args.output}")


if __name__ == "__main__":
    main()
