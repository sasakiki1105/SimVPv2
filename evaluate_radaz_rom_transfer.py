"""Evaluate source-frozen RadAz reduced-order models across electric fields.

The development stage transfers an E25 ROM to E30.  The final stage compares
E25-only, E30-only, and pooled E25+E30 ROMs on E40.  Source trajectories are
never concatenated across case boundaries when fitting Hankel DMD or HAVOK.
Target data before 24 us is used only to initialize the delay state; target
data from 24--30 us is reserved for evaluation.
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
from sklearn.decomposition import PCA

import analyze_radaz_augmented_physical_state_dynamics as augmented
import analyze_radaz_blockwise_fourier_latent_dynamics as block
import analyze_radaz_fourier_latent_to_physical_modes as physical_modes
import analyze_radaz_hankel_havok as hankel


ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT.parent / "PEPAPIC" / "test" / "results" / "2D_Landmark"
DEFAULT_OUTPUT = ROOT / "workdirs" / "compare_radaz_rom_transfer_e25_e30_e40"

FIT_START_US = 20.0
VALIDATION_START_US = 23.0
FORECAST_START_US = 24.0
FORECAST_END_US = 30.0
PCA_BUDGET = block.BUDGETS["medium_20"]
MODE_BANDS = tuple(augmented.MODE_BANDS)


def case_directory(ez_kvm: int) -> Path:
    name = f"2D_RadAz_Xe1p_Bx20mT_Ez{ez_kvm}kVm_dt15ps_out15ns"
    return RESULTS_ROOT / name / name


CASE_SPECS = {
    "E25": {
        "ez_kvm": 25,
        "features": ROOT
        / "workdirs"
        / "analyze_radaz_bx20mt_ez25kvm_fourier_latent_raw"
        / "fourier_latent_features.h5",
        "physical_cache": ROOT
        / "workdirs"
        / "analyze_radaz_bx20mt_ez25kvm_fourier_latent_to_physical_modes"
        / "physical_fourier_targets.h5",
    },
    "E30": {
        "ez_kvm": 30,
        "features": ROOT
        / "workdirs"
        / "analyze_radaz_bx20mt_electric_field_sweep_forcing"
        / "cases"
        / "E30kVm"
        / "fourier_latent_features.h5",
    },
    "E40": {
        "ez_kvm": 40,
        "features": ROOT
        / "workdirs"
        / "analyze_radaz_bx20mt_electric_field_sweep_forcing"
        / "cases"
        / "E40kVm"
        / "fourier_latent_features.h5",
    },
}

SYSTEM_GROUPS = {
    "transport_only": ("transport",),
    "cross_only": ("cross",),
    "latent_cross": ("latent", "cross"),
}


@dataclass
class CaseData:
    name: str
    ez_kvm: int
    features: np.ndarray
    time_us: np.ndarray
    frames: np.ndarray
    physical: augmented.PhysicalStates
    physical_flat: dict[str, np.ndarray]
    feature_path: Path
    physical_path: Path


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


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def time_mask(case: CaseData, start: float, stop: float, inclusive=False):
    upper = case.time_us <= stop if inclusive else case.time_us < stop
    return (case.time_us >= start) & upper


def load_case(name: str, cache_root: Path) -> CaseData:
    specification = CASE_SPECS[name]
    feature_path = Path(specification["features"])
    features, time_us, frames = block.load_features(feature_path)
    if name == "E25":
        physical_path = Path(specification["physical_cache"])
    else:
        physical_path = cache_root / f"{name}_physical_fourier_targets.h5"
        physical_path.parent.mkdir(parents=True, exist_ok=True)
        source = (
            case_directory(int(specification["ez_kvm"]))
            / "analysis_fields_uncompressed.h5"
        )
        physical_modes.extract_physical_fourier(
            source,
            physical_path,
            frames,
            time_us,
            bands=8,
            maximum_mode=21,
        )
    physical = augmented.load_physical_states(physical_path)
    if not np.array_equal(frames, physical.frame):
        raise ValueError(f"{name}: latent and physical frames do not match")
    if not np.allclose(time_us, physical.time_us, atol=1.0e-10, rtol=0.0):
        raise ValueError(f"{name}: latent and physical times do not match")
    if len(time_us) < 1000 or time_us[-1] < 29.8:
        raise ValueError(f"{name}: incomplete time series")
    print(
        f"[LOAD] {name}: features={features.shape}, "
        f"time={time_us[0]:.3f}--{time_us[-1]:.3f} us",
        flush=True,
    )
    return CaseData(
        name=name,
        ez_kvm=int(specification["ez_kvm"]),
        features=features,
        time_us=time_us,
        frames=frames,
        physical=physical,
        physical_flat=augmented.flatten_physical(physical),
        feature_path=feature_path,
        physical_path=physical_path,
    )


def fit_shared_block_pca(
    sources: list[CaseData], start_us: float, stop_us: float
) -> tuple[dict[str, block.BlockPCA], list[dict]]:
    models: dict[str, block.BlockPCA] = {}
    rows: list[dict] = []
    for block_name, (mode_start, mode_end) in block.BLOCKS.items():
        selected = []
        feature_shape = None
        for case in sources:
            mask = time_mask(case, start_us, stop_us)
            values = block.block_slice(
                case.features, mode_start, mode_end
            )
            feature_shape = values.shape[1:]
            selected.append(values[mask].reshape(np.count_nonzero(mask), -1))
        fit = np.concatenate(selected, axis=0)
        full_mean = np.mean(fit, axis=0)
        variance = np.var(fit, axis=0)
        floor = max(float(np.max(variance)) * 1.0e-12, 1.0e-20)
        active = variance > floor
        components = min(
            int(PCA_BUDGET[block_name]),
            int(np.count_nonzero(active)),
            len(fit) - 1,
        )
        pca = PCA(
            n_components=components,
            svd_solver="randomized",
            random_state=0,
            iterated_power=5,
        )
        pca.fit(fit[:, active])
        assert feature_shape is not None
        models[block_name] = block.BlockPCA(
            block_name,
            mode_start,
            mode_end,
            components,
            feature_shape,
            full_mean,
            active,
            pca,
        )
        rows.append(
            {
                "block": block_name,
                "mode_start": mode_start,
                "mode_end": mode_end,
                "components": components,
                "active_features": int(np.count_nonzero(active)),
                "total_features": len(full_mean),
                "explained_variance": float(
                    np.sum(pca.explained_variance_ratio_)
                ),
            }
        )
    return models, rows


def transform_block_pca(
    models: dict[str, block.BlockPCA], features: np.ndarray
) -> np.ndarray:
    return np.concatenate(
        [models[name].transform(features) for name in block.BLOCKS], axis=1
    )


def groups_for_case(
    case: CaseData, scores: np.ndarray, system: str
) -> dict[str, np.ndarray]:
    available = {"latent": scores, **case.physical_flat}
    return {name: available[name] for name in SYSTEM_GROUPS[system]}


def fit_source_scaler(
    groups: dict[str, dict[str, np.ndarray]],
    sources: list[CaseData],
    start_us: float,
    stop_us: float,
) -> augmented.GroupScaler:
    combined: dict[str, np.ndarray] = {}
    for group_name in next(iter(groups.values())):
        combined[group_name] = np.concatenate(
            [
                groups[case.name][group_name][
                    time_mask(case, start_us, stop_us)
                ]
                for case in sources
            ],
            axis=0,
        )
    fit_mask = np.ones(len(next(iter(combined.values()))), dtype=bool)
    return augmented.GroupScaler.fit(combined, fit_mask)


def make_multitrajectory_hankel(
    trajectories: list[np.ndarray], delay: int, rank: int
) -> hankel.HankelModel:
    delay_sets = [hankel.make_delay_vectors(values, delay) for values in trajectories]
    combined = np.concatenate(delay_sets, axis=0)
    delay_mean = np.mean(combined, axis=0)
    centered = combined - delay_mean
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    maximum_rank = min(
        sum(len(values) - 1 for values in delay_sets), right.shape[0]
    )
    if rank > maximum_rank:
        raise ValueError(f"rank={rank} exceeds maximum={maximum_rank}")
    basis = right[:rank].T
    coordinate_sets = [
        (values - delay_mean) @ basis for values in delay_sets
    ]
    left = np.concatenate([values[:-1] for values in coordinate_sets], axis=0)
    right_side = np.concatenate(
        [values[1:] for values in coordinate_sets], axis=0
    )
    matrix = right_side.T @ np.linalg.pinv(left.T, rcond=1.0e-10)
    return hankel.HankelModel(
        delay=delay,
        rank=rank,
        state_dimensions=trajectories[0].shape[1],
        delay_mean=delay_mean,
        basis=basis,
        matrix=matrix,
        eigenvalues=np.linalg.eigvals(matrix),
        singular_values=singular_values,
    )


def fit_multitrajectory_havok(
    model: hankel.HankelModel, trajectories: list[np.ndarray]
) -> hankel.HavokModel:
    coordinate_sets = [
        model.project(hankel.make_delay_vectors(values, model.delay))
        for values in trajectories
    ]
    feature_sets = []
    target_sets = []
    forcing_values = []
    for coordinates in coordinate_sets:
        resolved = coordinates[:, :-1]
        forcing = coordinates[:, -1]
        feature_sets.append(
            np.column_stack([resolved[:-1], forcing[:-1]])
        )
        target_sets.append(resolved[1:])
        forcing_values.append(forcing)
    features = np.concatenate(feature_sets, axis=0)
    targets = np.concatenate(target_sets, axis=0)
    coefficient = np.linalg.lstsq(features, targets, rcond=1.0e-10)[0]
    forcing = np.concatenate(forcing_values)
    forcing_scale = float(np.std(forcing, ddof=1))
    if forcing_scale < 1.0e-12:
        forcing_scale = 1.0
    return hankel.HavokModel(
        matrix=coefficient[:-1].T,
        forcing_vector=coefficient[-1].copy(),
        forcing_mean=float(np.mean(forcing)),
        forcing_scale=forcing_scale,
    )


def candidate_search(
    source_states: dict[str, np.ndarray],
    sources: list[CaseData],
    delays: list[int],
    ranks: list[int],
    system: str,
) -> tuple[dict, list[dict]]:
    subtrains = [
        source_states[case.name][
            time_mask(case, FIT_START_US, VALIDATION_START_US)
        ]
        for case in sources
    ]
    validations = [
        source_states[case.name][
            time_mask(case, VALIDATION_START_US, FORECAST_START_US)
        ]
        for case in sources
    ]
    rows: list[dict] = []
    for delay in delays:
        for rank in ranks:
            try:
                model = make_multitrajectory_hankel(subtrains, delay, rank)
                predictions = [
                    hankel.rollout_hankel(model, history, len(validation))
                    for history, validation in zip(subtrains, validations)
                ]
                truth = np.concatenate(validations, axis=0)
                prediction = np.concatenate(predictions, axis=0)
                persistence = np.concatenate(
                    [
                        np.repeat(history[-1:], len(validation), axis=0)
                        for history, validation in zip(subtrains, validations)
                    ],
                    axis=0,
                )
                mse = float(np.mean((prediction - truth) ** 2))
                persistence_mse = float(
                    np.mean((persistence - truth) ** 2)
                )
                correlation = block.safe_correlation(truth, prediction)
                radius = float(np.max(np.abs(model.eigenvalues)))
            except (ValueError, np.linalg.LinAlgError):
                mse = float("inf")
                persistence_mse = float("nan")
                correlation = float("nan")
                radius = float("nan")
            rows.append(
                {
                    "system": system,
                    "delay": delay,
                    "history_us": delay
                    * float(np.median(np.diff(sources[0].time_us))),
                    "rank": rank,
                    "validation_state_mse": mse,
                    "validation_skill_vs_persistence": (
                        1.0 - mse / persistence_mse
                        if np.isfinite(mse) and persistence_mse > 0.0
                        else float("-inf")
                    ),
                    "validation_state_correlation": correlation,
                    "spectral_radius": radius,
                }
            )
    finite = [row for row in rows if np.isfinite(row["validation_state_mse"])]
    if not finite:
        raise RuntimeError(f"No valid candidates for {system}")
    selected = min(
        finite,
        key=lambda row: (
            row["validation_state_mse"],
            row["delay"],
            row["rank"],
        ),
    )
    return selected, rows


def fit_source_selections(
    sources: list[CaseData], delays: list[int], ranks: list[int]
) -> tuple[dict[str, dict], list[dict]]:
    pca_models, _ = fit_shared_block_pca(
        sources, FIT_START_US, VALIDATION_START_US
    )
    scores = {
        case.name: transform_block_pca(pca_models, case.features)
        for case in sources
    }
    selections: dict[str, dict] = {}
    candidate_rows: list[dict] = []
    source_label = "+".join(case.name for case in sources)
    for system in SYSTEM_GROUPS:
        groups = {
            case.name: groups_for_case(case, scores[case.name], system)
            for case in sources
        }
        scaler = fit_source_scaler(
            groups, sources, FIT_START_US, VALIDATION_START_US
        )
        states = {
            case.name: scaler.transform(groups[case.name]) for case in sources
        }
        selected, rows = candidate_search(
            states, sources, delays, ranks, system
        )
        selections[system] = selected
        for row in rows:
            row["source"] = source_label
        candidate_rows.extend(rows)
        print(
            f"[SELECT] {source_label} {system}: "
            f"q={selected['delay']} r={selected['rank']} "
            f"mse={selected['validation_state_mse']:.4e}",
            flush=True,
        )
    return selections, candidate_rows


def predict_system(
    system: str,
    delay: int,
    rank: int,
    sources: list[CaseData],
    target: CaseData,
    source_scores: dict[str, np.ndarray],
    target_scores: np.ndarray,
) -> tuple[
    dict[str, dict[str, np.ndarray]], augmented.GroupScaler, hankel.HankelModel
]:
    source_groups = {
        case.name: groups_for_case(case, source_scores[case.name], system)
        for case in sources
    }
    target_groups = groups_for_case(target, target_scores, system)
    scaler = fit_source_scaler(
        source_groups, sources, FIT_START_US, FORECAST_START_US
    )
    source_states = [
        scaler.transform(source_groups[case.name])[
            time_mask(case, FIT_START_US, FORECAST_START_US)
        ]
        for case in sources
    ]
    model = make_multitrajectory_hankel(source_states, delay, rank)
    havok_model = fit_multitrajectory_havok(model, source_states)
    target_states = scaler.transform(target_groups)
    history = target_states[target.time_us < FORECAST_START_US]
    forecast_mask = time_mask(
        target, FORECAST_START_US, FORECAST_END_US, inclusive=True
    )
    steps = int(np.count_nonzero(forecast_mask))
    standardized_predictions = {
        "hankel_dmd": hankel.rollout_hankel(model, history, steps),
        "havok_zero_forcing": hankel.rollout_havok_zero_forcing(
            model, havok_model, history, steps
        ),
    }
    predictions = {
        method: scaler.inverse(values)
        for method, values in standardized_predictions.items()
    }
    return predictions, scaler, model


def transport_and_cross(
    prediction: dict[str, np.ndarray], radial_weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray | None]:
    if "transport" in prediction:
        return prediction["transport"], None
    cross = augmented.unflatten_cross(prediction["cross"])
    return augmented.transport_from_cross(cross, radial_weights), cross


def evaluate_predictions(
    source_label: str,
    target: CaseData,
    system_label: str,
    method: str,
    delay: int,
    rank: int,
    prediction: dict[str, np.ndarray],
) -> tuple[list[dict], list[dict]]:
    forecast_mask = time_mask(
        target, FORECAST_START_US, FORECAST_END_US, inclusive=True
    )
    history_mask = target.time_us < FORECAST_START_US
    truth_transport = target.physical.transport[forecast_mask]
    truth_cross = target.physical.cross[forecast_mask]
    predicted_transport, predicted_cross = transport_and_cross(
        prediction, target.physical.macro_weights
    )
    persistence_transport = np.repeat(
        target.physical.transport[history_mask][-1:],
        len(truth_transport),
        axis=0,
    )
    metric_rows: list[dict] = []
    rollout_rows: list[dict] = []
    for band_index, band in enumerate(MODE_BANDS):
        metrics = augmented.scalar_metrics(
            truth_transport[:, band_index],
            predicted_transport[:, band_index],
            persistence_transport[:, band_index],
        )
        row = {
            "source": source_label,
            "target": target.name,
            "system": system_label,
            "method": method,
            "band": band,
            "delay": delay,
            "history_us": delay * float(np.median(np.diff(target.time_us))),
            "rank": rank,
            **{f"transport_{key}": value for key, value in metrics.items()},
        }
        if predicted_cross is not None:
            target_band = truth_cross[:, :, band_index]
            estimate_band = predicted_cross[:, :, band_index]
            persistence_cross = np.repeat(
                target.physical.cross[history_mask][-1:, :, band_index],
                len(target_band),
                axis=0,
            )
            cross_metrics = augmented.scalar_metrics(
                target_band, estimate_band, persistence_cross
            )
            row.update(
                {f"cross_{key}": value for key, value in cross_metrics.items()}
            )
            row["cross_weighted_phase_mae_rad"] = augmented.weighted_phase_mae(
                target_band[:, :, None],
                estimate_band[:, :, None],
                target.physical.macro_weights,
            )
        metric_rows.append(row)
        for local_index, time_value in enumerate(target.time_us[forecast_mask]):
            rollout_rows.append(
                {
                    "source": source_label,
                    "target": target.name,
                    "system": system_label,
                    "method": method,
                    "band": band,
                    "time_us": float(time_value),
                    "truth_transport": float(
                        truth_transport[local_index, band_index]
                    ),
                    "predicted_transport": float(
                        predicted_transport[local_index, band_index]
                    ),
                    "persistence_transport": float(
                        persistence_transport[local_index, band_index]
                    ),
                }
            )
    return metric_rows, rollout_rows


def pca_oracle_rows(
    source_label: str,
    target: CaseData,
    models: dict[str, block.BlockPCA],
    target_scores: np.ndarray,
) -> list[dict]:
    forecast_mask = time_mask(
        target, FORECAST_START_US, FORECAST_END_US, inclusive=True
    )
    truth = target.features[forecast_mask]
    prediction = block.decode_blocks(
        models, target_scores[forecast_mask], truth
    )
    rows = []
    for name, (mode_start, mode_end) in block.BLOCKS.items():
        truth_block = block.block_slice(truth, mode_start, mode_end)
        prediction_block = block.block_slice(
            prediction, mode_start, mode_end
        )
        rows.append(
            {
                "source": source_label,
                "target": target.name,
                "block": name,
                "coefficient_nrmse": block.coefficient_nrmse(
                    truth_block, prediction_block
                ),
                "amplitude_correlation": block.safe_correlation(
                    block.band_amplitude(truth, mode_start, mode_end),
                    block.band_amplitude(prediction, mode_start, mode_end),
                ),
            }
        )
    return rows


def state_shift_rows(
    source_label: str,
    target: CaseData,
    scaler: augmented.GroupScaler,
    target_scores: np.ndarray,
    system: str,
    system_label: str,
) -> list[dict]:
    groups = groups_for_case(target, target_scores, system)
    history_mask = time_mask(
        target, FIT_START_US, FORECAST_START_US
    )
    rows = []
    for name in scaler.names:
        z = (groups[name][history_mask] - scaler.means[name]) / scaler.scales[name]
        rows.append(
            {
                "source": source_label,
                "target": target.name,
                "system": system_label,
                "group": name,
                "history_mean_abs_z": float(np.mean(np.abs(z))),
                "history_rms_z": float(np.sqrt(np.mean(z * z))),
                "history_max_abs_z": float(np.max(np.abs(z))),
                "history_fraction_abs_z_gt3": float(np.mean(np.abs(z) > 3.0)),
            }
        )
    return rows


def run_source_to_target(
    sources: list[CaseData],
    target: CaseData,
    delays: list[int],
    ranks: list[int],
    locked_selections: dict[str, dict] | None = None,
    locked_candidates: list[dict] | None = None,
) -> dict[str, list[dict]]:
    source_label = "+".join(case.name for case in sources)
    if locked_selections is None:
        selections, candidates = fit_source_selections(
            sources, delays, ranks
        )
    else:
        selections = locked_selections
        candidates = list(locked_candidates or [])
    final_models, pca_rows = fit_shared_block_pca(
        sources, FIT_START_US, FORECAST_START_US
    )
    source_scores = {
        case.name: transform_block_pca(final_models, case.features)
        for case in sources
    }
    target_scores = transform_block_pca(final_models, target.features)

    metrics: list[dict] = []
    rollouts: list[dict] = []
    shifts: list[dict] = []
    variants = (
        (
            "transport_only_selected",
            "transport_only",
            selections["transport_only"],
        ),
        ("cross_only_selected", "cross_only", selections["cross_only"]),
        (
            "cross_only_matched",
            "cross_only",
            selections["latent_cross"],
        ),
        (
            "latent_cross_coupled",
            "latent_cross",
            selections["latent_cross"],
        ),
    )
    for system_label, system, selection in variants:
        predictions, scaler, model = predict_system(
            system,
            int(selection["delay"]),
            int(selection["rank"]),
            sources,
            target,
            source_scores,
            target_scores,
        )
        shifts.extend(
            state_shift_rows(
                source_label,
                target,
                scaler,
                target_scores,
                system,
                system_label,
            )
        )
        for method, prediction in predictions.items():
            metric_rows, rollout_rows = evaluate_predictions(
                source_label,
                target,
                system_label,
                method,
                int(selection["delay"]),
                int(selection["rank"]),
                prediction,
            )
            for row in metric_rows:
                row["source_hankel_spectral_radius"] = float(
                    np.max(np.abs(model.eigenvalues))
                )
            metrics.extend(metric_rows)
            rollouts.extend(rollout_rows)

    selection_rows = []
    for system, values in selections.items():
        selection_rows.append({"source": source_label, **values})
    for row in pca_rows:
        row["source"] = source_label
    oracle = pca_oracle_rows(
        source_label, target, final_models, target_scores
    )
    return {
        "metrics": metrics,
        "rollouts": rollouts,
        "candidates": candidates,
        "selections": selection_rows,
        "pca": pca_rows,
        "oracle": oracle,
        "shifts": shifts,
    }


def numeric(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def select_rows(rows: list[dict], **criteria) -> list[dict]:
    return [
        row
        for row in rows
        if all(str(row.get(key)) == str(value) for key, value in criteria.items())
    ]


def plot_rollouts(path: Path, rows: list[dict], stage: str) -> None:
    targets = sorted({row["target"] for row in rows})
    if len(targets) != 1:
        raise ValueError("Expected exactly one target in rollout plot")
    target = targets[0]
    sources = sorted({row["source"] for row in rows})
    figure, axes = plt.subplots(2, 1, figsize=(12.0, 8.0), sharex=True)
    colors = {"E25": "#0072b2", "E30": "#d55e00", "E25+E30": "#009e73"}
    for axis, band in zip(axes, MODE_BANDS):
        reference = select_rows(
            rows,
            source=sources[0],
            system="latent_cross_coupled",
            method="havok_zero_forcing",
            band=band,
        )
        axis.plot(
            numeric(reference, "time_us"),
            numeric(reference, "truth_transport"),
            color="#111111",
            linewidth=2.0,
            label=f"{target} truth",
        )
        axis.plot(
            numeric(reference, "time_us"),
            numeric(reference, "persistence_transport"),
            color="#777777",
            linestyle="--",
            linewidth=1.5,
            label="target persistence",
        )
        for source in sources:
            selected = select_rows(
                rows,
                source=source,
                system="latent_cross_coupled",
                method="havok_zero_forcing",
                band=band,
            )
            axis.plot(
                numeric(selected, "time_us"),
                numeric(selected, "predicted_transport"),
                color=colors[source],
                linewidth=1.7,
                label=f"{source} latent+cross HAVOK",
            )
        if stage == "development":
            matched = select_rows(
                rows,
                source="E25",
                system="cross_only_matched",
                method="havok_zero_forcing",
                band=band,
            )
            axis.plot(
                numeric(matched, "time_us"),
                numeric(matched, "predicted_transport"),
                color="#56b4e9",
                linestyle=":",
                linewidth=1.5,
                label="E25 cross-only matched",
            )
        axis.set_ylabel("modal transport")
        axis.set_title(band)
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right", fontsize=8, ncol=2)
    axes[-1].set_xlabel("target time [us]")
    figure.suptitle(
        f"{stage.capitalize()} ROM transfer: source-frozen dynamics to {target}"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_correlations(path: Path, rows: list[dict], stage: str) -> None:
    sources = sorted({row["source"] for row in rows})
    systems = (
        "transport_only_selected",
        "cross_only_selected",
        "cross_only_matched",
        "latent_cross_coupled",
    )
    colors = ("#e69f00", "#56b4e9", "#0072b2", "#009e73")
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharey=True)
    width = 0.18
    x = np.arange(len(sources), dtype=np.float64)
    for axis, band in zip(axes, MODE_BANDS):
        for index, (system, color) in enumerate(zip(systems, colors)):
            values = []
            for source in sources:
                selected = select_rows(
                    rows,
                    source=source,
                    system=system,
                    method="havok_zero_forcing",
                    band=band,
                )
                values.append(float(selected[0]["transport_correlation"]))
            axis.bar(
                x + (index - 1.5) * width,
                values,
                width,
                color=color,
                label=system,
            )
        axis.axhline(0.0, color="#333333", linewidth=1.0)
        axis.set_xticks(x, sources)
        axis.set_title(band)
        axis.set_ylabel("transport correlation")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(loc="lower right", fontsize=7)
    figure.suptitle(f"{stage.capitalize()} transfer model comparison")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def coupling_gain_rows(metrics: list[dict]) -> list[dict]:
    rows = []
    sources = sorted({row["source"] for row in metrics})
    targets = sorted({row["target"] for row in metrics})
    methods = ("hankel_dmd", "havok_zero_forcing")
    for source in sources:
        for target in targets:
            for method in methods:
                for band in MODE_BANDS:
                    coupled = select_rows(
                        metrics,
                        source=source,
                        target=target,
                        system="latent_cross_coupled",
                        method=method,
                        band=band,
                    )
                    matched = select_rows(
                        metrics,
                        source=source,
                        target=target,
                        system="cross_only_matched",
                        method=method,
                        band=band,
                    )
                    selected = select_rows(
                        metrics,
                        source=source,
                        target=target,
                        system="cross_only_selected",
                        method=method,
                        band=band,
                    )
                    if not coupled or not matched or not selected:
                        continue
                    rows.append(
                        {
                            "source": source,
                            "target": target,
                            "method": method,
                            "band": band,
                            "coupled_correlation": float(
                                coupled[0]["transport_correlation"]
                            ),
                            "gain_vs_cross_matched": float(
                                coupled[0]["transport_correlation"]
                            )
                            - float(matched[0]["transport_correlation"]),
                            "gain_vs_cross_selected": float(
                                coupled[0]["transport_correlation"]
                            )
                            - float(selected[0]["transport_correlation"]),
                        }
                    )
    return rows


def plot_coupling_gain(path: Path, rows: list[dict], stage: str) -> None:
    sources = sorted({row["source"] for row in rows})
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    width = 0.34
    x = np.arange(len(sources), dtype=np.float64)
    for axis, band in zip(axes, MODE_BANDS):
        matched = []
        selected = []
        for source in sources:
            row = select_rows(
                rows,
                source=source,
                method="havok_zero_forcing",
                band=band,
            )[0]
            matched.append(float(row["gain_vs_cross_matched"]))
            selected.append(float(row["gain_vs_cross_selected"]))
        axis.bar(
            x - width / 2,
            matched,
            width,
            color="#009e73",
            label="coupled - cross matched",
        )
        axis.bar(
            x + width / 2,
            selected,
            width,
            color="#0072b2",
            label="coupled - cross selected",
        )
        axis.axhline(0.0, color="#333333", linewidth=1.0)
        axis.set_xticks(x, sources)
        axis.set_title(band)
        axis.set_ylabel("transport correlation gain")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(loc="lower right", fontsize=8)
    figure.suptitle(f"{stage.capitalize()} incremental latent information")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def format_metric_table(rows: list[dict]) -> list[str]:
    lines = [
        "| source | target | system | band | corr | NRMSE | persistence skill |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    primary = [
        row
        for row in rows
        if row["method"] == "havok_zero_forcing"
    ]
    for row in primary:
        lines.append(
            f"| {row['source']} | {row['target']} | {row['system']} | "
            f"{row['band']} | {float(row['transport_correlation']):.3f} | "
            f"{float(row['transport_nrmse']):.3f} | "
            f"{float(row['transport_skill_vs_persistence']):.3f} |"
        )
    return lines


def write_readme(output: Path) -> None:
    development = read_csv(output / "development_e25_to_e30_metrics.csv")
    final = read_csv(output / "final_e40_metrics.csv")
    lines = [
        "# RadAz ROM transfer: E25/E30 to E40",
        "",
        "## Protocol",
        "",
        "- State: data-only SimVP blockwise latent (20D) plus complex radial cross-spectrum (16D).",
        "- Dynamics: source-fitted Hankel DMD and HAVOK zero-forcing.",
        "- Source fitting: 20--23 us subtrain, 23--24 us validation, then 20--24 us refit.",
        "- Target use: only the pre-24 us delay history initializes the forecast.",
        "- Evaluation: autonomous 24--30 us target forecast.",
        "- Per-case SimVP input normalization uses only frames before 24 us; it is target-history calibration, not forecast leakage.",
        "- Pooled E25+E30 fitting keeps trajectories separate and never creates an E25-to-E30 transition pair.",
        "",
        "## Development: E25 to E30",
        "",
    ]
    if development:
        lines.extend(format_metric_table(development))
    else:
        lines.append("Pending.")
    lines.extend(["", "## Locked final test: E40", ""])
    if final:
        lines.extend(format_metric_table(final))
    else:
        lines.append("Pending until all E40 candidate models are frozen.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The decisive comparisons are latent+cross versus cross-only with matched delay/rank, and pooled E25+E30 versus E30-only. A positive target correlation gain supports incremental latent information. Pooled improvement over E30-only supports shared multi-condition dynamics; otherwise the ROM is more likely local to an electric-field regime.",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run_stage(
    stage: str,
    output: Path,
    delays: list[int],
    ranks: list[int],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "physical_cache"
    locked_by_source = None
    candidates_by_source = None
    if stage == "development":
        needed = ("E25", "E30")
        source_sets = (("E25",),)
        target_name = "E30"
        prefix = "development_e25_to_e30"
        cases = {name: load_case(name, cache) for name in needed}
    else:
        source_sets = (("E25",), ("E30",), ("E25", "E30"))
        target_name = "E40"
        prefix = "final_e40"
        cases = {name: load_case(name, cache) for name in ("E25", "E30")}
        locked_by_source = {}
        candidates_by_source = {}
        lock_rows = []
        for names in source_sets:
            label = "+".join(names)
            selections, candidates = fit_source_selections(
                [cases[name] for name in names], delays, ranks
            )
            locked_by_source[label] = selections
            candidates_by_source[label] = candidates
            for system, values in selections.items():
                lock_rows.append({"source": label, **values})
        write_csv(output / "final_e40_locked_selections.csv", lock_rows)
        write_csv(
            output / "final_e40_locked_candidates.csv",
            [
                row
                for label in candidates_by_source
                for row in candidates_by_source[label]
            ],
        )
        script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        lock = {
            "locked_before_e40_load": True,
            "script_sha256": script_hash,
            "source_sets": source_sets,
            "target": target_name,
            "systems": SYSTEM_GROUPS,
            "pca_budget": PCA_BUDGET,
            "fit_us": [FIT_START_US, FORECAST_START_US],
            "validation_us": [VALIDATION_START_US, FORECAST_START_US],
            "forecast_us": [FORECAST_START_US, FORECAST_END_US],
            "selections": locked_by_source,
        }
        (output / "final_e40_protocol_lock_before_target_load.json").write_text(
            json.dumps(json_safe(lock), indent=2), encoding="utf-8"
        )
        print(
            "[LOCK] E25-only, E30-only, and pooled selections saved "
            "before E40 was loaded",
            flush=True,
        )
        cases["E40"] = load_case("E40", cache)
    target = cases[target_name]
    combined = {
        "metrics": [],
        "rollouts": [],
        "candidates": [],
        "selections": [],
        "pca": [],
        "oracle": [],
        "shifts": [],
    }
    for names in source_sets:
        source_label = "+".join(names)
        print(
            f"[TRANSFER] {source_label} -> {target_name}", flush=True
        )
        result = run_source_to_target(
            [cases[name] for name in names],
            target,
            delays,
            ranks,
            (
                locked_by_source[source_label]
                if locked_by_source is not None
                else None
            ),
            (
                candidates_by_source[source_label]
                if candidates_by_source is not None
                else None
            ),
        )
        for key in combined:
            combined[key].extend(result[key])
        for key in combined:
            if combined[key]:
                write_csv(output / f"{prefix}_{key}.csv", combined[key])
    gains = coupling_gain_rows(combined["metrics"])
    write_csv(output / f"{prefix}_coupling_gain.csv", gains)
    plot_rollouts(
        output / f"{prefix}_transport_rollouts.png",
        combined["rollouts"],
        stage,
    )
    plot_correlations(
        output / f"{prefix}_transport_correlation.png",
        combined["metrics"],
        stage,
    )
    plot_coupling_gain(
        output / f"{prefix}_latent_increment_gain.png", gains, stage
    )
    summary = {
        "stage": stage,
        "source_sets": source_sets,
        "target": target_name,
        "fit_us": [FIT_START_US, FORECAST_START_US],
        "validation_us": [VALIDATION_START_US, FORECAST_START_US],
        "forecast_us": [FORECAST_START_US, FORECAST_END_US],
        "target_forecast_truth_used_as_input": False,
        "target_input_normalization_end_us": 23.985,
        "trajectory_boundaries_preserved": True,
        "delays": delays,
        "ranks": ranks,
        "cases": {
            name: {
                "feature_path": str(case.feature_path),
                "physical_path": str(case.physical_path),
                "frames": len(case.frames),
            }
            for name, case in cases.items()
        },
    }
    (output / f"{prefix}_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2), encoding="utf-8"
    )
    write_readme(output)
    print(f"[DONE] {stage}: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("development", "final"), required=True
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--delays", default="20,40,60,80,100")
    parser.add_argument("--ranks", default="8,12,20,30,40")
    args = parser.parse_args()
    delays = [int(value) for value in args.delays.split(",")]
    ranks = [int(value) for value in args.ranks.split(",")]
    run_stage(args.stage, args.output.resolve(), delays, ranks)


if __name__ == "__main__":
    main()
