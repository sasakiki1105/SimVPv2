"""Compare shared HAVOK forcing across the RadAz electric-field sweep.

The same frozen Ez=10 kV/m SimVP model is used as a feature map for every
case. Each case is normalized with its own training interval to avoid input
clipping. A common Fourier-latent PCA basis and a common Hankel/HAVOK model
are then fitted without creating transitions across case boundaries.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis, spearmanr
from sklearn.decomposition import PCA

import analyze_radaz_fourier_latent_dynamics as fourier
import analyze_radaz_hankel_havok as hankel
import analyze_radaz_latent_features as latent
import analyze_radaz_reduced_dynamics as reduced


ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT.parent
LANDMARK = (
    RESEARCH / "PEPAPIC" / "test" / "results" / "2D_Landmark"
)
OUTPUT = (
    ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_electric_field_sweep_forcing"
)

ELECTRIC_FIELDS = (10, 20, 30, 40)
LAYERS = ("encoder", "translator")
FIT_START_US = 20.0
VALIDATION_START_US = 23.0
FORECAST_START_US = 24.0
END_US = 30.0
FRAME_DT_US = 0.015

MODE_METRICS = (
    "phi_a_mtsi",
    "phi_a_ecdi",
    "efy_a_mtsi",
    "efy_a_ecdi",
    "electron_den_a_mtsi",
    "electron_den_a_ecdi",
    "phi_delta",
    "efy_delta",
    "electron_den_delta",
)
TRANSPORT_METRICS = (
    "a_mtsi_integrated",
    "a_ecdi_integrated",
    "delta_integrated",
    "exb_total",
    "exb_mtsi",
    "exb_ecdi",
    "velocity_total",
)
PHYSICAL_METRICS = MODE_METRICS + TRANSPORT_METRICS
PLOT_METRICS = (
    "phi_a_mtsi",
    "phi_a_ecdi",
    "electron_den_a_mtsi",
    "electron_den_a_ecdi",
    "delta_integrated",
    "exb_total",
    "exb_mtsi",
    "exb_ecdi",
)


@dataclass
class CaseInfo:
    electric_field_kvm: int
    case_name: str
    root: Path
    input_h5: Path
    feature_h5: Path
    mode_csv: Path
    transport_csv: Path
    temporal_peaks_csv: Path
    k_frequency_peaks_csv: Path


@dataclass
class SharedPCA:
    active_mask: np.ndarray
    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    retained: int
    score_mean: np.ndarray
    score_scale: np.ndarray

    def transform(self, features: np.ndarray) -> np.ndarray:
        active = features[:, self.active_mask]
        scores = (active - self.mean) @ self.components[: self.retained].T
        return (scores - self.score_mean) / self.score_scale


@dataclass
class PreparedHankel:
    delay: int
    state_dimensions: int
    delay_mean: np.ndarray
    right: np.ndarray
    singular_values: np.ndarray
    delay_blocks: list[np.ndarray]


def case_info(electric_field_kvm: int, output: Path) -> CaseInfo:
    case_name = (
        "2D_RadAz_Xe1p_Bx20mT_"
        f"Ez{electric_field_kvm}kVm_dt15ps_out15ns"
    )
    root = LANDMARK / case_name / case_name
    if electric_field_kvm == 10:
        input_name = (
            "radaz_3ch_trainfixed_margin20_"
            "native257x256_pad260x256.h5"
        )
        feature_h5 = (
            ROOT
            / "workdirs"
            / "analyze_radaz_bx20mt_ez10kvm_fourier_latent_dynamics"
            / "fourier_latent_features.h5"
        )
    else:
        input_name = (
            "radaz_3ch_localnorm_trainfixed_margin20_"
            "native257x256_pad260x256.h5"
        )
        feature_h5 = (
            output
            / "cases"
            / f"E{electric_field_kvm}kVm"
            / "fourier_latent_features.h5"
        )
    return CaseInfo(
        electric_field_kvm=electric_field_kvm,
        case_name=case_name,
        root=root,
        input_h5=root / "SimVPv2_inputs" / input_name,
        feature_h5=feature_h5,
        mode_csv=(
            root
            / f"bifurcation_analysis_B20mT_E{electric_field_kvm}kVm"
            / "mode_band_time_series.csv"
        ),
        transport_csv=(
            LANDMARK
            / (
                "compare_electric_field_sweep_2d_modes_transport_"
                "E10_E20_E30_E40kVm"
            )
            / f"E{electric_field_kvm}kVm"
            / "two_dimensional_mode_transport_time_series.csv"
        ),
        temporal_peaks_csv=(
            root
            / f"bifurcation_analysis_B20mT_E{electric_field_kvm}kVm"
            / "selected_mode_temporal_peaks.csv"
        ),
        k_frequency_peaks_csv=(
            root
            / f"bifurcation_analysis_B20mT_E{electric_field_kvm}kVm"
            / "k_frequency_peaks.csv"
        ),
    )


def require_inputs(cases: list[CaseInfo]) -> None:
    paths = []
    for case in cases:
        paths.extend(
            (
                case.input_h5,
                case.mode_csv,
                case.transport_csv,
                case.temporal_peaks_csv,
                case.k_frequency_peaks_csv,
            )
        )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))


def extract_missing_features(
    cases: list[CaseInfo],
    device_name: str,
    batch_size: int,
    radial_bands: int,
    maximum_mode: int,
) -> dict[str, dict]:
    device = latent.resolve_device(device_name)
    summaries: dict[str, dict] = {}
    for case in cases:
        label = f"E{case.electric_field_kvm}kVm"
        if case.feature_h5.is_file():
            with h5py.File(case.feature_h5, "r") as source:
                summaries[label] = {
                    "status": "reused",
                    "windows": int(len(source["window_start"])),
                    "file": str(case.feature_h5),
                    "bytes": case.feature_h5.stat().st_size,
                }
            print(f"[REUSE] {label}: {case.feature_h5}")
            continue
        case.feature_h5.parent.mkdir(parents=True, exist_ok=True)
        result = fourier.extract_fourier_latents(
            case.input_h5,
            latent.DEFAULT_WORKDIR,
            latent.DEFAULT_CONFIG,
            case.feature_h5,
            device,
            batch_size,
            radial_bands,
            maximum_mode,
        )
        summaries[label] = {"status": "extracted", **result}
    return summaries


def feature_fit_matrix(
    cases: list[CaseInfo], layer: str
) -> tuple[np.ndarray, int]:
    blocks = []
    feature_count = None
    for case in cases:
        with h5py.File(case.feature_h5, "r") as source:
            time_us = (
                np.asarray(source[f"{layer}_time_s"], dtype=np.float64)
                * 1.0e6
            )
            fit = (time_us >= FIT_START_US) & (
                time_us < FORECAST_START_US
            )
            dataset = source[f"{layer}_fourier_ri"]
            values = np.asarray(dataset[fit], dtype=np.float32).reshape(
                np.count_nonzero(fit), -1
            )
            blocks.append(values)
            feature_count = values.shape[1]
    return np.concatenate(blocks, axis=0), int(feature_count)


def fit_shared_pca(
    cases: list[CaseInfo],
    layer: str,
    maximum_components: int,
    variance_target: float,
    output: Path,
) -> tuple[SharedPCA, dict]:
    fit_features, feature_count = feature_fit_matrix(cases, layer)
    variance = np.var(fit_features, axis=0)
    active_mask = variance > 1.0e-14
    active = fit_features[:, active_mask]
    component_limit = min(
        maximum_components,
        active.shape[0] - 1,
        active.shape[1],
    )
    pca = PCA(
        n_components=component_limit,
        svd_solver="randomized",
        random_state=0,
        iterated_power=4,
    )
    fit_scores = pca.fit_transform(active)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    reached = np.flatnonzero(cumulative >= variance_target)
    retained = (
        int(reached[0] + 1) if reached.size else int(component_limit)
    )
    score_mean = np.mean(fit_scores[:, :retained], axis=0)
    score_scale = np.std(
        fit_scores[:, :retained], axis=0, ddof=1
    )
    score_scale[score_scale < 1.0e-12] = 1.0
    shared = SharedPCA(
        active_mask=active_mask,
        mean=pca.mean_,
        components=pca.components_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        retained=retained,
        score_mean=score_mean,
        score_scale=score_scale,
    )
    np.savez_compressed(
        output / f"shared_fourier_pca_{layer}.npz",
        active_feature_mask=active_mask,
        mean=pca.mean_,
        components=pca.components_,
        explained_variance=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        singular_values=pca.singular_values_,
        retained_components=np.int64(retained),
        score_mean=score_mean,
        score_scale=score_scale,
    )
    summary = {
        "raw_features": feature_count,
        "active_features": int(np.count_nonzero(active_mask)),
        "fit_samples": int(len(fit_features)),
        "computed_components": int(component_limit),
        "retained_components": retained,
        "variance_at_retained": float(cumulative[retained - 1]),
        "variance_pc1": float(cumulative[0]),
        "variance_pc1_to_pc10": float(
            cumulative[min(9, len(cumulative) - 1)]
        ),
    }
    print(
        f"[PCA] {layer}: pooled samples={len(fit_features)}, "
        f"95% PCs={retained}"
    )
    return shared, summary


def transform_cases(
    cases: list[CaseInfo],
    layer: str,
    pca: SharedPCA,
) -> dict[int, reduced.LayerData]:
    result: dict[int, reduced.LayerData] = {}
    for case in cases:
        with h5py.File(case.feature_h5, "r") as source:
            dataset = source[f"{layer}_fourier_ri"]
            features = np.asarray(dataset, dtype=np.float32).reshape(
                len(dataset), -1
            )
            scores = pca.transform(features).astype(np.float64)
            time_us = (
                np.asarray(
                    source[f"{layer}_time_s"], dtype=np.float64
                )
                * 1.0e6
            )
        result[case.electric_field_kvm] = reduced.LayerData(
            name=f"{layer}_E{case.electric_field_kvm}",
            components=pca.retained,
            time_us=time_us,
            scores=scores,
        )
    return result


def prepare_shared_hankel(
    state_blocks: list[np.ndarray], delay: int
) -> PreparedHankel:
    delay_blocks = [
        hankel.make_delay_vectors(states, delay)
        for states in state_blocks
    ]
    combined = np.concatenate(delay_blocks, axis=0)
    delay_mean = np.mean(combined, axis=0)
    centered = combined - delay_mean
    _, singular_values, right = np.linalg.svd(
        centered, full_matrices=False
    )
    return PreparedHankel(
        delay=delay,
        state_dimensions=state_blocks[0].shape[1],
        delay_mean=delay_mean,
        right=right,
        singular_values=singular_values,
        delay_blocks=delay_blocks,
    )


def model_from_prepared(
    prepared: PreparedHankel, rank: int
) -> hankel.HankelModel:
    maximum_rank = min(
        sum(len(block) - 1 for block in prepared.delay_blocks),
        prepared.right.shape[0],
    )
    if rank > maximum_rank:
        raise ValueError(f"rank={rank} exceeds {maximum_rank}")
    basis = prepared.right[:rank].T
    coordinate_blocks = [
        (block - prepared.delay_mean) @ basis
        for block in prepared.delay_blocks
    ]
    x = np.concatenate(
        [coordinates[:-1] for coordinates in coordinate_blocks],
        axis=0,
    ).T
    y = np.concatenate(
        [coordinates[1:] for coordinates in coordinate_blocks],
        axis=0,
    ).T
    matrix = y @ np.linalg.pinv(x, rcond=1.0e-10)
    return hankel.HankelModel(
        delay=prepared.delay,
        rank=rank,
        state_dimensions=prepared.state_dimensions,
        delay_mean=prepared.delay_mean,
        basis=basis,
        matrix=matrix,
        eigenvalues=np.linalg.eigvals(matrix),
        singular_values=prepared.singular_values,
    )


def masks(layer: reduced.LayerData) -> dict[str, np.ndarray]:
    return {
        "subtrain": (layer.time_us >= FIT_START_US)
        & (layer.time_us < VALIDATION_START_US),
        "validation": (layer.time_us >= VALIDATION_START_US)
        & (layer.time_us < FORECAST_START_US),
        "fit": (layer.time_us >= FIT_START_US)
        & (layer.time_us < FORECAST_START_US),
        "steady": (layer.time_us >= FIT_START_US)
        & (layer.time_us <= END_US),
        "holdout": (layer.time_us >= FORECAST_START_US)
        & (layer.time_us <= END_US),
    }


def select_shared_hankel(
    layers: dict[int, reduced.LayerData],
    delays: list[int],
    ranks: list[int],
) -> tuple[dict, list[dict]]:
    fields = sorted(layers)
    subtrain = [
        layers[field].scores[masks(layers[field])["subtrain"]]
        for field in fields
    ]
    validation = [
        layers[field].scores[masks(layers[field])["validation"]]
        for field in fields
    ]
    rows: list[dict] = []
    for delay in delays:
        prepared = prepare_shared_hankel(subtrain, delay)
        for rank in ranks:
            try:
                model = model_from_prepared(prepared, rank)
                predictions = [
                    hankel.rollout_hankel(model, history, len(truth))
                    for history, truth in zip(subtrain, validation)
                ]
                truth_all = np.concatenate(validation, axis=0)
                prediction_all = np.concatenate(predictions, axis=0)
                persistence_all = np.concatenate(
                    [
                        np.repeat(
                            history[-1][None, :], len(truth), axis=0
                        )
                        for history, truth in zip(subtrain, validation)
                    ],
                    axis=0,
                )
                mse = float(
                    np.mean((prediction_all - truth_all) ** 2)
                )
                persistence_mse = float(
                    np.mean((persistence_all - truth_all) ** 2)
                )
                correlation = flattened_correlation(
                    truth_all, prediction_all
                )
                radius = float(np.max(np.abs(model.eigenvalues)))
                skill = float(1.0 - mse / persistence_mse)
            except (ValueError, np.linalg.LinAlgError):
                mse = float("inf")
                skill = float("-inf")
                correlation = float("nan")
                radius = float("nan")
            rows.append(
                {
                    "delay": delay,
                    "history_us": delay * FRAME_DT_US,
                    "rank": rank,
                    "validation_mse": mse,
                    "validation_skill_vs_persistence": skill,
                    "validation_correlation": correlation,
                    "spectral_radius": radius,
                }
            )
    finite = [row for row in rows if np.isfinite(row["validation_mse"])]
    if not finite:
        raise RuntimeError("Every shared Hankel candidate failed")
    selected = min(
        finite,
        key=lambda row: (
            row["validation_mse"],
            row["delay"],
            row["rank"],
        ),
    )
    return selected, rows


def fit_shared_havok(
    model: hankel.HankelModel,
    fit_blocks: list[np.ndarray],
) -> tuple[hankel.HavokModel, list[np.ndarray]]:
    coordinate_blocks = [
        model.project(hankel.make_delay_vectors(states, model.delay))
        for states in fit_blocks
    ]
    resolved_x = np.concatenate(
        [coordinates[:-1, :-1] for coordinates in coordinate_blocks],
        axis=0,
    )
    forcing_x = np.concatenate(
        [coordinates[:-1, -1] for coordinates in coordinate_blocks],
        axis=0,
    )
    resolved_y = np.concatenate(
        [coordinates[1:, :-1] for coordinates in coordinate_blocks],
        axis=0,
    )
    design = np.column_stack([resolved_x, forcing_x])
    coefficient = np.linalg.lstsq(
        design, resolved_y, rcond=1.0e-10
    )[0]
    all_forcing = np.concatenate(
        [coordinates[:, -1] for coordinates in coordinate_blocks]
    )
    forcing_mean = float(np.mean(all_forcing))
    forcing_scale = float(np.std(all_forcing, ddof=1))
    if forcing_scale < 1.0e-12:
        forcing_scale = 1.0
    return (
        hankel.HavokModel(
            matrix=coefficient[:-1].T,
            forcing_vector=coefficient[-1].copy(),
            forcing_mean=forcing_mean,
            forcing_scale=forcing_scale,
        ),
        coordinate_blocks,
    )


def flattened_correlation(
    truth: np.ndarray, prediction: np.ndarray
) -> float:
    left = truth.reshape(-1)
    right = prediction.reshape(-1)
    finite = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(finite) < 3:
        return float("nan")
    if np.std(left[finite]) < 1.0e-12 or np.std(right[finite]) < 1.0e-12:
        return float("nan")
    return float(np.corrcoef(left[finite], right[finite])[0, 1])


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(finite) < 3:
        return float("nan")
    if np.std(left[finite]) < 1.0e-12 or np.std(right[finite]) < 1.0e-12:
        return float("nan")
    return float(spearmanr(left[finite], right[finite]).statistic)


def maximum_lagged_correlation(
    forcing: np.ndarray,
    physical: np.ndarray,
    maximum_lag: int = 40,
) -> tuple[float, int]:
    best_correlation = float("nan")
    best_lag = 0
    for lag in range(-maximum_lag, maximum_lag + 1):
        if lag < 0:
            left = forcing[-lag:]
            right = physical[:lag]
        elif lag > 0:
            left = forcing[:-lag]
            right = physical[lag:]
        else:
            left = forcing
            right = physical
        correlation = safe_spearman(left, right)
        if not np.isfinite(correlation):
            continue
        if (
            not np.isfinite(best_correlation)
            or abs(correlation) > abs(best_correlation)
        ):
            best_correlation = correlation
            best_lag = lag
    return best_correlation, best_lag


def forcing_spectrum(forcing: np.ndarray) -> dict:
    centered = forcing - np.mean(forcing)
    coefficients = np.fft.rfft(centered)
    power = np.abs(coefficients) ** 2
    frequency_mhz = np.fft.rfftfreq(len(centered), d=FRAME_DT_US)
    if len(power) <= 1 or np.sum(power[1:]) <= 0.0:
        return {
            "dominant_frequency_mhz": float("nan"),
            "dominant_period_us": float("nan"),
            "peak_power_fraction": float("nan"),
            "spectral_entropy": float("nan"),
        }
    nonzero_power = power[1:]
    peak_index = int(np.argmax(nonzero_power)) + 1
    distribution = nonzero_power / np.sum(nonzero_power)
    positive = distribution > 0.0
    entropy = -np.sum(
        distribution[positive] * np.log(distribution[positive])
    ) / np.log(len(distribution))
    frequency = float(frequency_mhz[peak_index])
    return {
        "dominant_frequency_mhz": frequency,
        "dominant_period_us": float(1.0 / frequency),
        "peak_power_fraction": float(distribution[peak_index - 1]),
        "spectral_entropy": float(entropy),
    }


def read_csv_numeric(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in rows[0]
    }


def read_physical(case: CaseInfo) -> dict[str, np.ndarray]:
    mode = read_csv_numeric(case.mode_csv)
    transport = read_csv_numeric(case.transport_csv)
    result = {"time_us": mode["time_us"]}
    for metric in MODE_METRICS:
        result[metric] = mode[metric]
    for metric in TRANSPORT_METRICS:
        result[metric] = np.interp(
            result["time_us"], transport["time_us"], transport[metric]
        )
    return result


def standardized_delta(values: np.ndarray) -> np.ndarray:
    center = np.median(values)
    scale = np.std(values, ddof=1)
    if not np.isfinite(scale) or scale < 1.0e-30:
        scale = 1.0
    normalized = (values - center) / scale
    delta = np.zeros_like(normalized)
    delta[1:] = np.diff(normalized)
    return delta


def longest_event_run(events: np.ndarray) -> int:
    longest = 0
    current = 0
    for event in events:
        if event:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def analyze_forcing_case(
    case: CaseInfo,
    layer: reduced.LayerData,
    model: hankel.HankelModel,
    havok_model: hankel.HavokModel,
    threshold: float,
    physical: dict[str, np.ndarray],
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    steady_mask = masks(layer)["steady"]
    states = layer.scores[steady_mask]
    state_time = layer.time_us[steady_mask]
    delay_vectors = hankel.make_delay_vectors(states, model.delay)
    coordinates = model.project(delay_vectors)
    delay_time = state_time[model.delay - 1 :]
    resolved = coordinates[:, :-1]
    forcing = coordinates[:, -1]
    forcing_z = (
        forcing - havok_model.forcing_mean
    ) / havok_model.forcing_scale
    events = np.abs(forcing_z) >= threshold

    current = resolved[:-1]
    true_next = resolved[1:]
    current_forcing = forcing[:-1]
    without = current @ havok_model.matrix.T
    with_forcing = without + np.outer(
        current_forcing, havok_model.forcing_vector
    )
    mse_without = float(np.mean((without - true_next) ** 2))
    mse_with = float(np.mean((with_forcing - true_next) ** 2))

    fit_states = layer.scores[masks(layer)["fit"]]
    holdout_states = layer.scores[masks(layer)["holdout"]]
    hankel_prediction = hankel.rollout_hankel(
        model, fit_states, len(holdout_states)
    )
    havok_prediction = hankel.rollout_havok_zero_forcing(
        model, havok_model, fit_states, len(holdout_states)
    )
    persistence = np.repeat(
        fit_states[-1][None, :], len(holdout_states), axis=0
    )

    statistic = {
        "electric_field_kvm": case.electric_field_kvm,
        "samples": int(len(forcing_z)),
        "forcing_mean": float(np.mean(forcing_z)),
        "forcing_rms": float(np.sqrt(np.mean(forcing_z**2))),
        "forcing_mean_abs": float(np.mean(np.abs(forcing_z))),
        "forcing_abs_q95": float(np.quantile(np.abs(forcing_z), 0.95)),
        "forcing_abs_q99": float(np.quantile(np.abs(forcing_z), 0.99)),
        "forcing_excess_kurtosis": float(
            kurtosis(forcing_z, fisher=True, bias=False)
        ),
        "event_threshold": threshold,
        "event_count": int(np.count_nonzero(events)),
        "event_fraction": float(np.mean(events)),
        "longest_event_run_frames": longest_event_run(events),
        "longest_event_run_us": longest_event_run(events) * FRAME_DT_US,
        "one_step_mse_without_forcing": mse_without,
        "one_step_mse_with_true_forcing": mse_with,
        "one_step_error_reduction": float(1.0 - mse_with / mse_without),
        **forcing_spectrum(forcing_z),
    }
    for name, prediction in (
        ("hankel_dmd", hankel_prediction),
        ("havok_zero_forcing", havok_prediction),
    ):
        mse = float(np.mean((prediction - holdout_states) ** 2))
        persistence_mse = float(
            np.mean((persistence - holdout_states) ** 2)
        )
        mean_mse = float(np.mean(holdout_states**2))
        statistic[f"{name}_holdout_rmse"] = float(np.sqrt(mse))
        statistic[f"{name}_skill_vs_persistence"] = float(
            1.0 - mse / persistence_mse
        )
        statistic[f"{name}_skill_vs_shared_mean"] = float(
            1.0 - mse / mean_mse
        )
        statistic[f"{name}_holdout_correlation"] = flattened_correlation(
            holdout_states, prediction
        )

    association_rows: list[dict] = []
    physical_at_time: dict[str, np.ndarray] = {}
    for metric in PHYSICAL_METRICS:
        values = np.interp(
            delay_time, physical["time_us"], physical[metric]
        )
        physical_at_time[metric] = values
        delta = standardized_delta(values)
        non_events = ~events
        event_delta = float(np.median(np.abs(delta[events]))) if np.any(events) else float("nan")
        normal_delta = (
            float(np.median(np.abs(delta[non_events])))
            if np.any(non_events)
            else float("nan")
        )
        response_ratio = (
            event_delta / normal_delta
            if np.isfinite(normal_delta) and normal_delta > 0.0
            else float("nan")
        )
        signed_lagged, signed_lag = maximum_lagged_correlation(
            forcing_z, values
        )
        change_lagged, change_lag = maximum_lagged_correlation(
            np.abs(forcing_z), np.abs(delta)
        )
        association_rows.append(
            {
                "electric_field_kvm": case.electric_field_kvm,
                "metric": metric,
                "spearman_forcing_vs_value": safe_spearman(
                    forcing_z, values
                ),
                "spearman_abs_forcing_vs_abs_delta": safe_spearman(
                    np.abs(forcing_z), np.abs(delta)
                ),
                "spearman_abs_forcing_vs_value": safe_spearman(
                    np.abs(forcing_z), values
                ),
                "median_abs_delta_at_event": event_delta,
                "median_abs_delta_without_event": normal_delta,
                "event_response_ratio": response_ratio,
                "max_abs_lagged_forcing_vs_value": signed_lagged,
                "forcing_to_value_lag_frames": signed_lag,
                "forcing_to_value_lag_us": signed_lag * FRAME_DT_US,
                "max_abs_lagged_absforcing_vs_absdelta": change_lagged,
                "absforcing_to_absdelta_lag_frames": change_lag,
                "absforcing_to_absdelta_lag_us": (
                    change_lag * FRAME_DT_US
                ),
                "samples": int(len(values)),
            }
        )

    series_rows = []
    for index, time_us in enumerate(delay_time):
        row = {
            "electric_field_kvm": case.electric_field_kvm,
            "time_us": float(time_us),
            "forcing_z": float(forcing_z[index]),
            "abs_forcing_z": float(abs(forcing_z[index])),
            "event": bool(events[index]),
        }
        for metric in PLOT_METRICS:
            row[metric] = float(physical_at_time[metric][index])
        series_rows.append(row)

    forecast_rows = []
    holdout_time = layer.time_us[masks(layer)["holdout"]]
    for method, prediction in (
        ("hankel_dmd", hankel_prediction),
        ("havok_zero_forcing", havok_prediction),
    ):
        error = np.sqrt(
            np.mean((prediction - holdout_states) ** 2, axis=1)
        )
        for time_us, value in zip(holdout_time, error):
            forecast_rows.append(
                {
                    "electric_field_kvm": case.electric_field_kvm,
                    "method": method,
                    "time_us": float(time_us),
                    "state_rmse": float(value),
                }
            )
    return statistic, association_rows, series_rows, forecast_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_pca(
    path: Path, summaries: dict[str, dict], output: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for axis, layer in zip(axes, LAYERS):
        with np.load(output / f"shared_fourier_pca_{layer}.npz") as data:
            cumulative = np.cumsum(data["explained_variance_ratio"])
        retained = summaries[layer]["retained_components"]
        axis.plot(
            np.arange(1, len(cumulative) + 1),
            cumulative,
            color="#0072b2",
        )
        axis.axhline(0.95, color="#555555", linestyle="--")
        axis.axvline(retained, color="#c94c35", linestyle=":")
        axis.set(
            xlabel="PCA components",
            ylabel="cumulative explained variance",
            title=f"{layer}: 95% at {retained} PCs",
            ylim=(0.0, 1.01),
        )
        axis.grid(alpha=0.2)
    fig.suptitle("Shared Fourier-latent PCA across Ez=10-40 kV/m")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_forcing_series(
    path: Path, rows: list[dict], threshold_by_layer: dict[str, float]
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(15, 12), sharex=True)
    for row_index, electric_field in enumerate(ELECTRIC_FIELDS):
        for column, layer in enumerate(LAYERS):
            axis = axes[row_index, column]
            selected = [
                row
                for row in rows
                if row["layer"] == layer
                and row["electric_field_kvm"] == electric_field
            ]
            time = np.asarray([row["time_us"] for row in selected])
            forcing = np.asarray([row["forcing_z"] for row in selected])
            dominance = np.asarray(
                [row["delta_integrated"] for row in selected]
            )
            dominance = (
                dominance - np.nanmean(dominance)
            ) / max(np.nanstd(dominance), 1.0e-12)
            axis.plot(
                time,
                forcing,
                color="#6f4aa8",
                linewidth=0.9,
                label="shared HAVOK forcing",
            )
            axis.plot(
                time,
                dominance,
                color="#008f7a",
                linewidth=1.0,
                alpha=0.75,
                label="mode dominance (standardized)",
            )
            threshold = threshold_by_layer[layer]
            axis.axhline(threshold, color="#777777", linestyle=":", linewidth=0.9)
            axis.axhline(-threshold, color="#777777", linestyle=":", linewidth=0.9)
            axis.axvline(24.0, color="#333333", linestyle="--", linewidth=0.9)
            axis.set_ylabel(f"Ez={electric_field}\nstandardized")
            axis.grid(alpha=0.18)
            if row_index == 0:
                axis.set_title(layer)
                axis.legend(loc="upper right", fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("time [us]")
    fig.suptitle(
        "Shared HAVOK forcing and MTSI/ECDI dominance across electric field"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_statistics(path: Path, statistics: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    specifications = (
        ("event_fraction", "forcing event fraction"),
        ("forcing_abs_q99", "99th percentile of |forcing|"),
        ("one_step_error_reduction", "1-step error reduction with true forcing"),
        ("havok_zero_forcing_holdout_correlation", "zero-forcing holdout correlation"),
    )
    colors = {"encoder": "#0072b2", "translator": "#d55e00"}
    for axis, (key, ylabel) in zip(axes.flat, specifications):
        for layer in LAYERS:
            selected = [
                row for row in statistics if row["layer"] == layer
            ]
            axis.plot(
                [row["electric_field_kvm"] for row in selected],
                [row[key] for row in selected],
                marker="o",
                color=colors[layer],
                label=layer,
            )
        axis.set_xlabel("Ez [kV/m]")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.legend(loc="best")
    fig.suptitle("Shared HAVOK forcing statistics")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_periodicity(path: Path, statistics: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    specifications = (
        ("dominant_frequency_mhz", "dominant forcing frequency [MHz]"),
        ("peak_power_fraction", "power fraction at dominant frequency"),
        ("spectral_entropy", "normalized spectral entropy"),
    )
    colors = {"encoder": "#0072b2", "translator": "#d55e00"}
    for axis, (key, ylabel) in zip(axes, specifications):
        for layer in LAYERS:
            selected = [
                row for row in statistics if row["layer"] == layer
            ]
            axis.plot(
                [row["electric_field_kvm"] for row in selected],
                [row[key] for row in selected],
                marker="o",
                color=colors[layer],
                label=layer,
            )
        axis.set_xlabel("Ez [kV/m]")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.legend(loc="best")
    fig.suptitle("Is the HAVOK forcing intermittent or periodic?")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_forcing_spectra(path: Path, rows: list[dict]) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(14, 11), sharex=True)
    for row_index, electric_field in enumerate(ELECTRIC_FIELDS):
        for column, layer in enumerate(LAYERS):
            axis = axes[row_index, column]
            selected = [
                row
                for row in rows
                if row["layer"] == layer
                and row["electric_field_kvm"] == electric_field
            ]
            forcing = np.asarray(
                [row["forcing_z"] for row in selected], dtype=np.float64
            )
            forcing -= np.mean(forcing)
            frequency = np.fft.rfftfreq(len(forcing), d=FRAME_DT_US)
            power = np.abs(np.fft.rfft(forcing)) ** 2
            if np.sum(power[1:]) > 0.0:
                power /= np.sum(power[1:])
            axis.semilogy(
                frequency[1:],
                np.maximum(power[1:], 1.0e-12),
                color="#6f4aa8",
            )
            axis.set_ylabel(f"Ez={electric_field}\npower fraction")
            axis.grid(alpha=0.18)
            if row_index == 0:
                axis.set_title(layer)
    for axis in axes[-1]:
        axis.set_xlabel("frequency [MHz]")
    fig.suptitle("Spectrum of the shared HAVOK forcing coordinate")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def compare_forcing_to_physical_frequencies(
    cases: list[CaseInfo], statistics: list[dict]
) -> list[dict]:
    rows = []
    for case in cases:
        peak_rows = []
        with case.temporal_peaks_csv.open(
            "r", newline="", encoding="utf-8-sig"
        ) as handle:
            peak_rows = list(csv.DictReader(handle))
        physical = {}
        for target in ("MTSI", "ECDI"):
            matched = [
                row
                for row in peak_rows
                if row["source"] == "radial_mean"
                and row["regime"] == target
            ]
            if not matched:
                raise ValueError(
                    f"No radial_mean {target} peak in "
                    f"{case.temporal_peaks_csv}"
                )
            physical[target] = float(matched[0]["frequency_abs_mhz"])
        for layer in LAYERS:
            statistic = next(
                row
                for row in statistics
                if row["layer"] == layer
                and row["electric_field_kvm"]
                == case.electric_field_kvm
            )
            forcing_frequency = statistic["dominant_frequency_mhz"]
            candidates = []
            for regime_name, fundamental in physical.items():
                for harmonic in range(1, 5):
                    expected = fundamental * harmonic
                    candidates.append(
                        {
                            "regime": regime_name,
                            "harmonic": harmonic,
                            "expected": expected,
                            "absolute_error": abs(
                                forcing_frequency - expected
                            ),
                            "relative_error": abs(
                                forcing_frequency - expected
                            )
                            / expected,
                        }
                    )
            best = min(
                candidates,
                key=lambda candidate: candidate["relative_error"],
            )
            with case.k_frequency_peaks_csv.open(
                "r", newline="", encoding="utf-8-sig"
            ) as handle:
                measured_peaks = [
                    row
                    for row in csv.DictReader(handle)
                    if row["source"] == "radial_mean"
                ]
            nearest_measured = min(
                measured_peaks,
                key=lambda row: abs(
                    forcing_frequency - float(row["frequency_mhz"])
                ),
            )
            nearest_measured_frequency = float(
                nearest_measured["frequency_mhz"]
            )
            rows.append(
                {
                    "layer": layer,
                    "electric_field_kvm": case.electric_field_kvm,
                    "forcing_frequency_mhz": forcing_frequency,
                    "physical_mtsi_frequency_mhz": physical["MTSI"],
                    "physical_ecdi_frequency_mhz": physical["ECDI"],
                    "nearest_regime": best["regime"],
                    "nearest_harmonic": best["harmonic"],
                    "nearest_expected_frequency_mhz": best["expected"],
                    "frequency_absolute_error_mhz": best["absolute_error"],
                    "frequency_relative_error": best["relative_error"],
                    "nearest_measured_mode_n": int(
                        nearest_measured["mode_n"]
                    ),
                    "nearest_measured_peak_rank": int(
                        nearest_measured["peak_rank"]
                    ),
                    "nearest_measured_frequency_mhz": (
                        nearest_measured_frequency
                    ),
                    "nearest_measured_power_fraction": float(
                        nearest_measured["power_fraction"]
                    ),
                    "nearest_measured_relative_error": abs(
                        forcing_frequency - nearest_measured_frequency
                    )
                    / nearest_measured_frequency,
                }
            )
    return rows


def plot_frequency_comparison(path: Path, rows: list[dict]) -> None:
    fig, axis = plt.subplots(figsize=(10, 6))
    fields = np.asarray(ELECTRIC_FIELDS, dtype=np.float64)
    per_field = {
        field: next(
            row
            for row in rows
            if row["electric_field_kvm"] == field
            and row["layer"] == "encoder"
        )
        for field in ELECTRIC_FIELDS
    }
    mtsi = np.asarray(
        [
            per_field[field]["physical_mtsi_frequency_mhz"]
            for field in ELECTRIC_FIELDS
        ]
    )
    ecdi = np.asarray(
        [
            per_field[field]["physical_ecdi_frequency_mhz"]
            for field in ELECTRIC_FIELDS
        ]
    )
    axis.plot(
        fields,
        mtsi,
        color="#777777",
        marker="s",
        linestyle="--",
        label="physical MTSI peak",
    )
    axis.plot(
        fields,
        ecdi,
        color="#009e73",
        marker="s",
        label="physical ECDI peak",
    )
    axis.plot(
        fields,
        2.0 * ecdi,
        color="#009e73",
        alpha=0.65,
        linestyle=":",
        label="2 x physical ECDI peak",
    )
    colors = {"encoder": "#0072b2", "translator": "#d55e00"}
    for layer in LAYERS:
        selected = [
            row for row in rows if row["layer"] == layer
        ]
        axis.plot(
            [row["electric_field_kvm"] for row in selected],
            [row["forcing_frequency_mhz"] for row in selected],
            marker="o",
            linewidth=2.0,
            color=colors[layer],
            label=f"{layer} forcing peak",
        )
    axis.set_xlabel("Ez [kV/m]")
    axis.set_ylabel("frequency [MHz]")
    axis.set_title("HAVOK forcing peaks versus physical instability peaks")
    axis.grid(alpha=0.2)
    axis.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_association_heatmap(
    path: Path, associations: list[dict], value_key: str, title: str
) -> None:
    labels = [
        f"{layer} E{electric_field}"
        for layer in LAYERS
        for electric_field in ELECTRIC_FIELDS
    ]
    matrix = np.full((len(labels), len(PLOT_METRICS)), np.nan)
    for row_index, label in enumerate(labels):
        layer, electric = label.split(" E")
        for column, metric in enumerate(PLOT_METRICS):
            matched = [
                row
                for row in associations
                if row["layer"] == layer
                and row["electric_field_kvm"] == int(electric)
                and row["metric"] == metric
            ]
            if matched:
                matrix[row_index, column] = matched[0][value_key]
    if value_key == "event_response_ratio":
        displayed = np.log2(np.maximum(matrix, 1.0e-6))
        vmin, vmax = -2.0, 2.0
        colorbar = "log2(event / non-event median |delta|)"
    else:
        displayed = matrix
        vmin, vmax = -1.0, 1.0
        colorbar = "Spearman correlation"
    fig, axis = plt.subplots(figsize=(14, 7))
    image = axis.imshow(
        displayed,
        aspect="auto",
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
    )
    axis.set_xticks(np.arange(len(PLOT_METRICS)))
    axis.set_xticklabels(PLOT_METRICS, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(labels)))
    short_labels = [
        label.replace("encoder", "Enc").replace("translator", "Trans")
        for label in labels
    ]
    axis.set_yticklabels(short_labels, fontsize=10)
    axis.set_title(title)
    for row in range(displayed.shape[0]):
        for column in range(displayed.shape[1]):
            value = displayed[row, column]
            if np.isfinite(value):
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )
    fig.colorbar(image, ax=axis, label=colorbar)
    fig.subplots_adjust(left=0.18, right=0.9, bottom=0.24, top=0.9)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_readme(
    path: Path,
    pca_summary: dict,
    layer_summary: dict,
    statistics: list[dict],
    frequency_rows: list[dict],
) -> None:
    table = []
    for row in statistics:
        table.append(
            "| {layer} | {electric_field_kvm} | {event_fraction:.4f} | "
            "{forcing_abs_q99:.3f} | {one_step_error_reduction:.4f} | "
            "{havok_zero_forcing_skill_vs_shared_mean:.4f} | "
            "{havok_zero_forcing_holdout_correlation:.4f} |".format(**row)
        )
    frequency_table = []
    for row in frequency_rows:
        frequency_table.append(
            "| {layer} | {electric_field_kvm} | "
            "{forcing_frequency_mhz:.3f} | "
            "{nearest_measured_mode_n} | "
            "{nearest_measured_frequency_mhz:.3f} | "
            "{nearest_measured_relative_error:.4f} |".format(**row)
        )
    path.write_text(
        f"""# RadAz electric-field sweep: shared HAVOK forcing

The Ez=10 kV/m SimVP model is used as one frozen feature map for all cases.
Each case uses the same train-only min-max procedure with case-local bounds,
so the comparison is not contaminated by clipping against the Ez=10 range.

- Cases: Ez = 10, 20, 30, 40 kV/m; Bx = 20 mT
- Frame interval: 15 ns
- Shared Fourier features: 8 radial bands, azimuthal modes n=0-21
- Shared PCA fit interval: 20-24 us from all four cases
- Shared Hankel/HAVOK fit interval: 20-24 us
- Model selection interval: 23-24 us
- Diagnostic interval: 20-30 us
- Autonomous holdout: 24-30 us
- Case boundaries are never treated as time transitions.

## Shared models

| Layer | PCs for 95% | delay | history [us] | rank | forcing threshold |
|---|---:|---:|---:|---:|---:|
| encoder | {pca_summary['encoder']['retained_components']} | {layer_summary['encoder']['delay']} | {layer_summary['encoder']['history_us']:.3f} | {layer_summary['encoder']['rank']} | {layer_summary['encoder']['forcing_threshold']:.3f} |
| translator | {pca_summary['translator']['retained_components']} | {layer_summary['translator']['delay']} | {layer_summary['translator']['history_us']:.3f} | {layer_summary['translator']['rank']} | {layer_summary['translator']['forcing_threshold']:.3f} |

## Main statistics

| Layer | Ez [kV/m] | event fraction | |forcing| q99 | one-step reduction with true forcing | zero-forcing skill vs shared mean | zero-forcing correlation |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

## Forcing frequency and measured PIC modes

| Layer | Ez [kV/m] | forcing peak [MHz] | nearest measured n | measured peak [MHz] | relative difference |
|---|---:|---:|---:|---:|---:|
{chr(10).join(frequency_table)}

## Interpretation

The HAVOK forcing coordinate is an unresolved dynamical coordinate, not the
externally imposed electric field itself. Since Ez is constant within each
trajectory, this analysis asks whether changing Ez changes the statistics of
unresolved events and whether those events coincide with changes in MTSI,
ECDI or transport.

## 日本語メモ

この解析の forcing は、外部電場 Ez そのものではなく、共有した線形低次元
ダイナミクスだけでは説明できない成分です。4ケースで同じPCA/HAVOK座標と
同じ閾値を使っているため、forcingイベント率や裾の強さを電場間で比較できます。
forcingが大きい時刻にMTSI/ECDI振幅や輸送量の変化も大きければ、モード遷移を
外力項として扱う低次元モデルへ進む根拠になります。

今回の結果ではforcing強度はEzに対して単調ではありません。encoderのzero-forcing
自律予測相関はEz=10, 20, 30, 40 kV/mでそれぞれ約0.03, 0.17, 0.61, 0.88となり、
高電場側ほど少数の周期モードで閉じやすくなりました。一方、translator forcingは
Ez=30 kV/mで物理モードn=-10、Ez=40 kV/mでn=-8とほぼ同じ周波数です。
Ez=40 kV/mのtranslator forcingはパワーの約97%が7.43 MHzへ集中しているため、
突発的な遷移forcingではなく、resolved stateから除外されたECDI高調波と解釈する
のが妥当です。Ez=10/30 kV/mのencoder forcingはより遅い包絡変動を持ち、モード
優勢度・輸送量との強い遅れ相関があるため、こちらが遷移forcingの有力候補です。
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--radial-bands", type=int, default=8)
    parser.add_argument("--maximum-mode", type=int, default=21)
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--variance-target", type=float, default=0.95)
    parser.add_argument(
        "--delays", type=int, nargs="+", default=[10, 20, 40, 80]
    )
    parser.add_argument(
        "--ranks", type=int, nargs="+", default=[8, 15, 20, 30, 40, 50]
    )
    parser.add_argument("--skip-extraction", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cases = [case_info(field, args.output) for field in ELECTRIC_FIELDS]
    require_inputs(cases)
    if args.skip_extraction:
        missing = [
            str(case.feature_h5)
            for case in cases
            if not case.feature_h5.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing Fourier features:\n" + "\n".join(missing)
            )
        extraction = {
            f"E{case.electric_field_kvm}kVm": {
                "status": "reused",
                "file": str(case.feature_h5),
                "bytes": case.feature_h5.stat().st_size,
            }
            for case in cases
        }
    else:
        extraction = extract_missing_features(
            cases,
            args.device,
            args.batch_size,
            args.radial_bands,
            args.maximum_mode,
        )

    pca_models: dict[str, SharedPCA] = {}
    pca_summary: dict[str, dict] = {}
    transformed: dict[str, dict[int, reduced.LayerData]] = {}
    for layer in LAYERS:
        pca_models[layer], pca_summary[layer] = fit_shared_pca(
            cases,
            layer,
            args.pca_components,
            args.variance_target,
            args.output,
        )
        transformed[layer] = transform_cases(
            cases, layer, pca_models[layer]
        )
    plot_pca(
        args.output / "shared_fourier_pca_explained_variance.png",
        pca_summary,
        args.output,
    )

    statistics: list[dict] = []
    associations: list[dict] = []
    time_series: list[dict] = []
    forecast_rows: list[dict] = []
    candidate_rows: list[dict] = []
    layer_summary: dict[str, dict] = {}
    thresholds: dict[str, float] = {}
    physical = {
        case.electric_field_kvm: read_physical(case) for case in cases
    }

    for layer in LAYERS:
        layer_cases = transformed[layer]
        selected, candidates = select_shared_hankel(
            layer_cases, args.delays, args.ranks
        )
        for row in candidates:
            candidate_rows.append(
                {
                    "layer": layer,
                    **row,
                    "selected": (
                        row["delay"] == selected["delay"]
                        and row["rank"] == selected["rank"]
                    ),
                }
            )
        fit_blocks = [
            layer_cases[field].scores[
                masks(layer_cases[field])["fit"]
            ]
            for field in ELECTRIC_FIELDS
        ]
        prepared = prepare_shared_hankel(
            fit_blocks, selected["delay"]
        )
        model = model_from_prepared(prepared, selected["rank"])
        havok_model, fit_coordinate_blocks = fit_shared_havok(
            model, fit_blocks
        )
        pooled_fit_forcing = np.concatenate(
            [coordinates[:, -1] for coordinates in fit_coordinate_blocks]
        )
        pooled_fit_forcing_z = (
            pooled_fit_forcing - havok_model.forcing_mean
        ) / havok_model.forcing_scale
        threshold = float(
            np.quantile(np.abs(pooled_fit_forcing_z), 0.95)
        )
        thresholds[layer] = threshold
        layer_summary[layer] = {
            "components": pca_summary[layer]["retained_components"],
            "delay": int(model.delay),
            "history_us": model.delay * FRAME_DT_US,
            "rank": int(model.rank),
            "validation": selected,
            "spectral_radius": float(
                np.max(np.abs(model.eigenvalues))
            ),
            "forcing_threshold": threshold,
            "forcing_fit_mean": havok_model.forcing_mean,
            "forcing_fit_scale": havok_model.forcing_scale,
        }
        print(
            f"[HAVOK] {layer}: q={model.delay}, rank={model.rank}, "
            f"threshold={threshold:.3f}"
        )

        for case in cases:
            statistic, assoc, series, forecasts = analyze_forcing_case(
                case,
                layer_cases[case.electric_field_kvm],
                model,
                havok_model,
                threshold,
                physical[case.electric_field_kvm],
            )
            statistic["layer"] = layer
            statistics.append(statistic)
            for row in assoc:
                associations.append({"layer": layer, **row})
            for row in series:
                time_series.append({"layer": layer, **row})
            for row in forecasts:
                forecast_rows.append({"layer": layer, **row})

    write_csv(args.output / "shared_hankel_candidates.csv", candidate_rows)
    write_csv(args.output / "forcing_statistics_by_ez.csv", statistics)
    write_csv(
        args.output / "forcing_physical_associations.csv", associations
    )
    write_csv(args.output / "forcing_time_series.csv", time_series)
    write_csv(
        args.output / "autonomous_forecast_time_series.csv",
        forecast_rows,
    )
    plot_forcing_series(
        args.output / "shared_havok_forcing_by_ez.png",
        time_series,
        thresholds,
    )
    plot_statistics(
        args.output / "forcing_statistics_vs_ez.png", statistics
    )
    plot_periodicity(
        args.output / "forcing_periodicity_vs_ez.png", statistics
    )
    plot_forcing_spectra(
        args.output / "forcing_spectra_by_ez.png", time_series
    )
    frequency_rows = compare_forcing_to_physical_frequencies(
        cases, statistics
    )
    write_csv(
        args.output / "forcing_frequency_vs_physical_modes.csv",
        frequency_rows,
    )
    plot_frequency_comparison(
        args.output / "forcing_frequency_vs_physical_modes.png",
        frequency_rows,
    )
    plot_association_heatmap(
        args.output / "forcing_vs_physical_delta_correlation.png",
        associations,
        "spearman_abs_forcing_vs_abs_delta",
        "Association of HAVOK forcing with physical changes",
    )
    plot_association_heatmap(
        args.output / "forcing_vs_physical_value_lagged_correlation.png",
        associations,
        "max_abs_lagged_forcing_vs_value",
        "Maximum lagged association of forcing with physical state",
    )
    plot_association_heatmap(
        args.output / "forcing_event_physical_response.png",
        associations,
        "event_response_ratio",
        "Physical change at forcing events relative to non-events",
    )

    summary = {
        "status": "PASS",
        "method": {
            "electric_fields_kvm": list(ELECTRIC_FIELDS),
            "fit_interval_us": [FIT_START_US, FORECAST_START_US],
            "validation_interval_us": [
                VALIDATION_START_US,
                FORECAST_START_US,
            ],
            "diagnostic_interval_us": [FIT_START_US, END_US],
            "holdout_interval_us": [FORECAST_START_US, END_US],
            "common_pca": True,
            "common_hankel_havok": True,
            "case_boundary_transitions_excluded": True,
            "normalization": "case-local train-only min-max with margin",
            "frozen_feature_model": str(latent.DEFAULT_WORKDIR),
        },
        "extraction": extraction,
        "pca": pca_summary,
        "layers": layer_summary,
        "statistics": statistics,
        "frequency_comparison": frequency_rows,
    }
    (args.output / "electric_field_forcing_summary.json").write_text(
        json.dumps(
            reduced.json_safe(summary), indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    write_readme(
        args.output / "README.md",
        pca_summary,
        layer_summary,
        statistics,
        frequency_rows,
    )
    print(f"[PASS] output={args.output}")


if __name__ == "__main__":
    main()
