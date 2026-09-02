"""Diagnose why local ROM closure varies across the RadAz magnetic sweep.

This script reuses the compact radial-band Fourier caches produced by
``analyze_radaz_magnetic_sweep_rom.py``.  It does not read the large stitched
PIC field files and does not use the GPU.
"""

from __future__ import annotations

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
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA


RESEARCH = Path(r"C:\Users\astro\research")
ROM_ROOT = (
    RESEARCH
    / "research_results"
    / "2D_RadAz"
    / "SimVPv2"
    / "workdirs"
    / "analyze_radaz_magnetic_sweep_rom_B10_B15_B20_B25_B30mT_E10kVm"
)
PEPAPIC_ANALYSIS = (
    RESEARCH
    / "research_results"
    / "2D_RadAz"
    / "PEPAPIC"
    / "2D_Landmark"
    / "analysis_results"
)
BIFURCATION_CSV = (
    PEPAPIC_ANALYSIS
    / "bifurcation_comparison_B10_B15_B20_B25_B30mT_E10kVm"
    / "bifurcation_sweep_summary.csv"
)
OUTPUT = (
    RESEARCH
    / "research_results"
    / "2D_RadAz"
    / "SimVPv2"
    / "workdirs"
    / "analyze_radaz_magnetic_sweep_dynamical_organization_B10_B15_B20_B25_B30mT_E10kVm"
)

B_VALUES = (10, 15, 20, 25, 30)
ANALYSIS_START_US = 20.0
ANALYSIS_END_US = 30.0
PCA_FIT_END_US = 24.0
ROLLING_WIDTH_US = 2.0
ROLLING_STEP_US = 0.5
BICOHERENCE_MAX_MODE = 30
BICOHERENCE_SURROGATES = 200
RNG_SEED = 20260825


@dataclass
class FourierCase:
    b_mt: int
    time_us: np.ndarray
    features: np.ndarray
    coefficient: np.ndarray
    channels: list[str]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_case(b_mt: int) -> FourierCase:
    path = ROM_ROOT / "physical_features" / f"B{b_mt}_physical_fourier.h5"
    with h5py.File(path, "r") as source:
        time_us = np.asarray(source["time_us"], dtype=np.float64)
        features = np.asarray(source["features"], dtype=np.float64)
        channels = [item.decode("ascii") for item in source["channels"][...]]
        bands = int(source.attrs["radial_bands"])
        max_mode = int(source.attrs["max_mode"])
    packed = features.reshape(len(time_us), len(channels), bands, 1 + 2 * max_mode)
    coefficient = np.empty(
        (len(time_us), len(channels), bands, max_mode + 1), dtype=np.complex128
    )
    coefficient[..., 0] = packed[..., 0]
    coefficient[..., 1:] = (
        packed[..., 1 : max_mode + 1]
        + 1j * packed[..., max_mode + 1 : 2 * max_mode + 1]
    )
    return FourierCase(b_mt, time_us, features, coefficient, channels)


def interval_mask(time_us: np.ndarray, start: float, stop: float) -> np.ndarray:
    return (time_us >= start - 1.0e-9) & (time_us <= stop + 1.0e-9)


def normalized_entropy(weights: np.ndarray, axis: int = -1) -> np.ndarray:
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    total = np.sum(weights, axis=axis, keepdims=True)
    probability = np.divide(weights, total, out=np.zeros_like(weights), where=total > 0)
    raw = -np.sum(
        np.where(probability > 0, probability * np.log(probability + 1.0e-300), 0.0),
        axis=axis,
    )
    count = weights.shape[axis]
    return raw / math.log(max(count, 2))


def temporal_spectral_entropy(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values - np.mean(values)
    window = np.hanning(len(values))
    power = np.abs(np.fft.rfft(values * window)) ** 2
    power = power[1:]
    entropy = float(normalized_entropy(power))
    probability = power / max(float(np.sum(power)), 1.0e-300)
    return entropy, float(np.exp(-np.sum(probability * np.log(probability + 1.0e-300))))


def dominant_complex_frequency(values: np.ndarray, dt_us: float) -> float:
    values = np.asarray(values, dtype=np.complex128)
    values = values - np.mean(values)
    spectrum = np.fft.fft(values * np.hanning(len(values)))
    frequency = np.fft.fftfreq(len(values), d=dt_us)
    valid = np.abs(frequency) > 1.0e-9
    return float(frequency[valid][np.argmax(np.abs(spectrum[valid]) ** 2)])


def phase_locking_diagnostics(
    case: FourierCase,
    mtsi_mode: int,
    ecdi_mode: int,
) -> dict:
    phi_index = case.channels.index("phi")
    mask = interval_mask(case.time_us, ANALYSIS_START_US, ANALYSIS_END_US)
    coefficient = case.coefficient[mask, phi_index]
    mtsi_band = int(np.argmax(np.mean(np.abs(coefficient[..., mtsi_mode]) ** 2, axis=0)))
    ecdi_band = int(np.argmax(np.mean(np.abs(coefficient[..., ecdi_mode]) ** 2, axis=0)))
    mtsi = coefficient[:, mtsi_band, mtsi_mode]
    ecdi = coefficient[:, ecdi_band, ecdi_mode]
    divisor = math.gcd(mtsi_mode, ecdi_mode)
    mtsi_multiplier = mtsi_mode // divisor
    ecdi_multiplier = ecdi_mode // divisor
    invariant = (
        mtsi_multiplier * np.angle(ecdi)
        - ecdi_multiplier * np.angle(mtsi)
    )
    weight = np.abs(mtsi) * np.abs(ecdi)
    unweighted = float(np.abs(np.mean(np.exp(1j * invariant))))
    weighted = float(
        np.abs(np.sum(weight * np.exp(1j * invariant))) / max(np.sum(weight), 1.0e-300)
    )
    unwrapped = np.unwrap(invariant)
    centered_time = case.time_us[mask] - np.mean(case.time_us[mask])
    slope, intercept = np.polyfit(centered_time, unwrapped, 1)
    residual = unwrapped - (slope * centered_time + intercept)
    increment_resultant = float(np.abs(np.mean(np.exp(1j * np.diff(invariant)))))
    slip_count = int(np.count_nonzero(np.abs(np.diff(residual)) > np.pi / 2.0))
    dt_us = float(np.median(np.diff(case.time_us[mask])))
    f_mtsi = dominant_complex_frequency(mtsi, dt_us)
    f_ecdi = dominant_complex_frequency(ecdi, dt_us)
    mismatch = mtsi_multiplier * f_ecdi - ecdi_multiplier * f_mtsi
    return {
        "B_mT": case.b_mt,
        "mtsi_mode": mtsi_mode,
        "ecdi_mode": ecdi_mode,
        "phase_relation_mtsi_multiplier": ecdi_multiplier,
        "phase_relation_ecdi_multiplier": mtsi_multiplier,
        "mtsi_radial_band": mtsi_band,
        "ecdi_radial_band": ecdi_band,
        "generalized_phase_resultant": unweighted,
        "amplitude_weighted_phase_resultant": weighted,
        "phase_increment_resultant": increment_resultant,
        "phase_drift_cycles_per_us": float(slope / (2.0 * np.pi)),
        "phase_residual_std_rad": float(np.std(residual)),
        "phase_slip_count": slip_count,
        "mtsi_signed_frequency_mhz": f_mtsi,
        "ecdi_signed_frequency_mhz": f_ecdi,
        "generalized_frequency_mismatch_mhz": float(mismatch),
    }


def pod_basis(values: np.ndarray, rank: int) -> tuple[np.ndarray, float]:
    _, singular, vh = np.linalg.svd(values, full_matrices=False)
    rank = min(rank, len(singular))
    fraction = float(np.sum(singular[:rank] ** 2) / max(np.sum(singular**2), 1.0e-300))
    return vh.conj().T[:, :rank], fraction


def subspace_overlap(left: np.ndarray, right: np.ndarray) -> float:
    singular = np.linalg.svd(left.conj().T @ right, compute_uv=False)
    return float(np.mean(np.clip(singular, 0.0, 1.0) ** 2))


def radial_structure_diagnostics(
    case: FourierCase,
    mtsi_mode: int,
    ecdi_mode: int,
) -> list[dict]:
    phi_index = case.channels.index("phi")
    windows = ((20.0, 24.0), (24.0, 27.0), (27.0, 30.0))
    rows: list[dict] = []
    for label, mode in (("MTSI", mtsi_mode), ("ECDI", ecdi_mode)):
        bases1: list[np.ndarray] = []
        bases2: list[np.ndarray] = []
        fractions1: list[float] = []
        fractions2: list[float] = []
        centroid_std: list[float] = []
        for start, stop in windows:
            mask = interval_mask(case.time_us, start, stop)
            values = case.coefficient[mask, phi_index, :, mode]
            basis1, fraction1 = pod_basis(values, rank=1)
            basis2, fraction2 = pod_basis(values, rank=2)
            power = np.abs(values) ** 2
            radius = np.arange(power.shape[1], dtype=np.float64)
            centroid = np.sum(power * radius[None, :], axis=1) / np.maximum(
                np.sum(power, axis=1), 1.0e-300
            )
            bases1.append(basis1)
            bases2.append(basis2)
            fractions1.append(fraction1)
            fractions2.append(fraction2)
            centroid_std.append(float(np.std(centroid)))
        rows.append(
            {
                "B_mT": case.b_mt,
                "regime": label,
                "mode_n": mode,
                "reference_window_us": "20-24",
                "target_window_us": "24-27",
                "pod1_overlap": subspace_overlap(bases1[0], bases1[1]),
                "pod2_overlap": subspace_overlap(bases2[0], bases2[1]),
                "pod1_energy_fraction_reference": fractions1[0],
                "pod1_energy_fraction_target": fractions1[1],
                "pod2_energy_fraction_reference": fractions2[0],
                "pod2_energy_fraction_target": fractions2[1],
                "radial_centroid_std_target_bands": centroid_std[1],
            }
        )
        rows.append(
            {
                "B_mT": case.b_mt,
                "regime": label,
                "mode_n": mode,
                "reference_window_us": "20-24",
                "target_window_us": "27-30",
                "pod1_overlap": subspace_overlap(bases1[0], bases1[2]),
                "pod2_overlap": subspace_overlap(bases2[0], bases2[2]),
                "pod1_energy_fraction_reference": fractions1[0],
                "pod1_energy_fraction_target": fractions1[2],
                "pod2_energy_fraction_reference": fractions2[0],
                "pod2_energy_fraction_target": fractions2[2],
                "radial_centroid_std_target_bands": centroid_std[2],
            }
        )
    return rows


def standard_dmd_eigenvalues(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = values[:-1].T
    y = values[1:].T
    operator = y @ np.linalg.pinv(x, rcond=1.0e-10)
    eigenvalues, eigenvectors = np.linalg.eig(operator)
    amplitudes = np.linalg.pinv(eigenvectors, rcond=1.0e-10) @ values[0]
    return eigenvalues, np.abs(amplitudes)


def rolling_dmd_diagnostics(
    case: FourierCase,
    components: int,
) -> tuple[list[dict], dict]:
    fit = interval_mask(case.time_us, ANALYSIS_START_US, PCA_FIT_END_US - 1.0e-9)
    model = PCA(n_components=components, svd_solver="randomized", random_state=42)
    model.fit(case.features[fit])
    scores = model.transform(case.features)
    scale = np.std(scores[fit], axis=0)
    scale[scale < 1.0e-12] = 1.0
    scores = (scores - np.mean(scores[fit], axis=0)) / scale
    dt_us = float(np.median(np.diff(case.time_us)))
    starts = np.arange(
        ANALYSIS_START_US,
        ANALYSIS_END_US - ROLLING_WIDTH_US + 1.0e-9,
        ROLLING_STEP_US,
    )
    rows: list[dict] = []
    eigen_sets: list[np.ndarray] = []
    for start in starts:
        stop = start + ROLLING_WIDTH_US
        mask = (case.time_us >= start - 1.0e-9) & (case.time_us < stop - 1.0e-9)
        eigenvalues, amplitudes = standard_dmd_eigenvalues(scores[mask])
        eigen_sets.append(eigenvalues)
        angle = np.angle(eigenvalues)
        candidates = np.flatnonzero(np.abs(angle) > 1.0e-4)
        if len(candidates):
            dominant = int(candidates[np.argmax(amplitudes[candidates])])
        else:
            dominant = int(np.argmax(amplitudes))
        value = eigenvalues[dominant]
        rows.append(
            {
                "B_mT": case.b_mt,
                "window_start_us": float(start),
                "window_end_us": float(stop),
                "components": components,
                "spectral_radius": float(np.max(np.abs(eigenvalues))),
                "dominant_frequency_mhz": float(abs(np.angle(value)) / (2.0 * np.pi * dt_us)),
                "dominant_growth_per_us": float(np.log(max(abs(value), 1.0e-300)) / dt_us),
                "dominant_amplitude_fraction": float(
                    amplitudes[dominant] / max(np.sum(amplitudes), 1.0e-300)
                ),
            }
        )
    drifts = []
    for left, right in zip(eigen_sets[:-1], eigen_sets[1:]):
        cost = np.abs(left[:, None] - right[None, :])
        row, col = linear_sum_assignment(cost)
        drifts.append(float(np.mean(cost[row, col])))
    frequency = np.asarray([row["dominant_frequency_mhz"] for row in rows])
    growth = np.asarray([row["dominant_growth_per_us"] for row in rows])
    radius = np.asarray([row["spectral_radius"] for row in rows])
    summary = {
        "B_mT": case.b_mt,
        "rolling_dmd_components": components,
        "rolling_dmd_frequency_mean_mhz": float(np.mean(frequency)),
        "rolling_dmd_frequency_std_mhz": float(np.std(frequency)),
        "rolling_dmd_frequency_cv": float(np.std(frequency) / max(np.mean(frequency), 1.0e-12)),
        "rolling_dmd_growth_std_per_us": float(np.std(growth)),
        "rolling_dmd_spectral_radius_mean": float(np.mean(radius)),
        "rolling_dmd_spectral_radius_std": float(np.std(radius)),
        "rolling_dmd_eigenvalue_set_drift": float(np.mean(drifts)),
    }
    return rows, summary


def triad_indices(max_mode: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triads = [
        (a, b, a + b)
        for a in range(1, max_mode + 1)
        for b in range(a, max_mode + 1)
        if a + b <= max_mode
    ]
    return tuple(np.asarray([item[i] for item in triads], dtype=np.int64) for i in range(3))


def spatial_bicoherence(
    coefficient: np.ndarray,
    indices: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    a, b, c = indices
    product = coefficient[..., a] * coefficient[..., b]
    numerator = np.abs(np.mean(product * np.conj(coefficient[..., c]), axis=(0, 1))) ** 2
    denominator = np.mean(np.abs(product) ** 2, axis=(0, 1)) * np.mean(
        np.abs(coefficient[..., c]) ** 2, axis=(0, 1)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 1.0e-300,
    )


def bicoherence_network_diagnostics(
    case: FourierCase,
    rng: np.random.Generator,
) -> tuple[list[dict], list[dict], dict]:
    phi_index = case.channels.index("phi")
    mask = interval_mask(case.time_us, ANALYSIS_START_US, ANALYSIS_END_US)
    coefficient = case.coefficient[mask, phi_index, :, : BICOHERENCE_MAX_MODE + 1]
    coefficient = coefficient - np.mean(coefficient, axis=0, keepdims=True)
    indices = triad_indices(BICOHERENCE_MAX_MODE)
    observed = spatial_bicoherence(coefficient, indices)
    null = np.empty((BICOHERENCE_SURROGATES, len(observed)), dtype=np.float32)
    for sample in range(BICOHERENCE_SURROGATES):
        shifted = coefficient.copy()
        for mode in range(1, BICOHERENCE_MAX_MODE + 1):
            shifted[..., mode] = np.roll(
                shifted[..., mode], int(rng.integers(0, len(shifted))), axis=0
            )
        null[sample] = spatial_bicoherence(shifted, indices)
    threshold = np.quantile(null, 0.99, axis=0)
    p_value = (1.0 + np.sum(null >= observed[None, :], axis=0)) / (
        BICOHERENCE_SURROGATES + 1.0
    )
    active = observed > threshold
    a, b, c = indices
    triad_rows = [
        {
            "B_mT": case.b_mt,
            "mode_a": int(a[index]),
            "mode_b": int(b[index]),
            "mode_c": int(c[index]),
            "bicoherence_squared": float(observed[index]),
            "pointwise_null_q99": float(threshold[index]),
            "pointwise_empirical_p": float(p_value[index]),
            "exceeds_pointwise_q99": int(active[index]),
        }
        for index in np.argsort(observed)[::-1]
    ]

    rolling_rows: list[dict] = []
    rolling_vectors: list[np.ndarray] = []
    for start in np.arange(20.0, 28.0 + 1.0e-9, 0.5):
        stop = start + 2.0
        local = (case.time_us[mask] >= start - 1.0e-9) & (
            case.time_us[mask] < stop - 1.0e-9
        )
        vector = spatial_bicoherence(coefficient[local], indices)
        rolling_vectors.append(vector)
        entropy = float(normalized_entropy(vector))
        probability = vector / max(float(np.sum(vector)), 1.0e-300)
        top = np.sort(probability)[-10:]
        rolling_rows.append(
            {
                "B_mT": case.b_mt,
                "window_start_us": float(start),
                "window_end_us": float(stop),
                "network_entropy_normalized": entropy,
                "effective_triad_count": float(np.exp(entropy * math.log(len(vector)))),
                "top10_weight_fraction": float(np.sum(top)),
                "adjacent_cosine_distance": float("nan"),
                "adjacent_top10_jaccard_distance": float("nan"),
            }
        )
    cosine_distances = []
    jaccard_distances = []
    for index in range(1, len(rolling_vectors)):
        left = rolling_vectors[index - 1]
        right = rolling_vectors[index]
        cosine = float(
            1.0
            - np.dot(left, right)
            / max(np.linalg.norm(left) * np.linalg.norm(right), 1.0e-300)
        )
        left_top = set(np.argsort(left)[-10:].tolist())
        right_top = set(np.argsort(right)[-10:].tolist())
        jaccard = float(1.0 - len(left_top & right_top) / max(len(left_top | right_top), 1))
        rolling_rows[index]["adjacent_cosine_distance"] = cosine
        rolling_rows[index]["adjacent_top10_jaccard_distance"] = jaccard
        cosine_distances.append(cosine)
        jaccard_distances.append(jaccard)

    entropy = float(normalized_entropy(observed))
    probability = observed / max(float(np.sum(observed)), 1.0e-300)
    summary = {
        "B_mT": case.b_mt,
        "bicoherence_network_entropy_normalized": entropy,
        "bicoherence_effective_triad_count": float(
            np.exp(entropy * math.log(len(observed)))
        ),
        "bicoherence_top10_weight_fraction": float(np.sum(np.sort(probability)[-10:])),
        "bicoherence_pointwise_q99_exceedance_count": int(np.count_nonzero(active)),
        "bicoherence_network_cosine_drift_mean": float(np.mean(cosine_distances)),
        "bicoherence_top10_jaccard_drift_mean": float(np.mean(jaccard_distances)),
        "bicoherence_top_triad": f"{a[np.argmax(observed)]}+{b[np.argmax(observed)]}->{c[np.argmax(observed)]}",
        "bicoherence_top_value": float(np.max(observed)),
    }
    return triad_rows, rolling_rows, summary


def spectral_diagnostics(case: FourierCase) -> dict:
    phi_index = case.channels.index("phi")
    mask = interval_mask(case.time_us, ANALYSIS_START_US, ANALYSIS_END_US)
    coefficient = case.coefficient[mask, phi_index, :, 1:]
    mode_power = np.sum(np.abs(coefficient) ** 2, axis=1)
    frame_entropy = normalized_entropy(mode_power, axis=1)
    mean_power = np.mean(mode_power, axis=0)
    mean_probability = mean_power / max(float(np.sum(mean_power)), 1.0e-300)
    mean_entropy = float(normalized_entropy(mean_power))
    top = np.sort(mean_probability)

    fit = interval_mask(case.time_us, ANALYSIS_START_US, PCA_FIT_END_US - 1.0e-9)
    pca = PCA(n_components=1, svd_solver="randomized", random_state=42)
    pca.fit(case.features[fit])
    score = pca.transform(case.features[mask])[:, 0]
    temporal_entropy, effective_frequency_count = temporal_spectral_entropy(score)
    return {
        "B_mT": case.b_mt,
        "spatial_spectral_entropy_mean": float(np.mean(frame_entropy)),
        "spatial_spectral_entropy_std": float(np.std(frame_entropy)),
        "mean_spectrum_entropy": mean_entropy,
        "spatial_effective_mode_count": float(np.exp(mean_entropy * math.log(len(mean_power)))),
        "mean_spectrum_top1_fraction": float(top[-1]),
        "mean_spectrum_top3_fraction": float(np.sum(top[-3:])),
        "pc1_temporal_spectral_entropy": temporal_entropy,
        "pc1_effective_frequency_count": effective_frequency_count,
    }


def closure_rows() -> tuple[dict[int, dict], dict[int, int], dict[int, dict]]:
    pca_rows = read_csv(ROM_ROOT / "pca_dimensionality.csv")
    components = {
        int(row["B_mT"]): int(row["individual_components_95"])
        for row in pca_rows
        if row["representation"] == "physical_fourier"
    }
    stationarity = {
        int(row["B_mT"]): row
        for row in read_csv(ROM_ROOT / "stationarity_diagnostics.csv")
        if row["representation"] == "physical_fourier"
    }
    grouped: dict[int, list[dict[str, str]]] = {b: [] for b in B_VALUES}
    for row in read_csv(ROM_ROOT / "rom_metrics.csv"):
        if row["representation"] == "physical_fourier" and row["basis"] == "individual":
            grouped[int(row["B_mT"])].append(row)
    selected = {
        b: max(rows, key=lambda row: float(row["skill_vs_persistence"]))
        for b, rows in grouped.items()
    }
    return selected, components, stationarity


def mode_rows() -> dict[int, tuple[int, int]]:
    rows = read_csv(BIFURCATION_CSV)
    return {
        int(round(float(row["b_mt"]))):
        (int(row["phi_mtsi_mode"]), int(row["phi_ecdi_mode"]))
        for row in rows
    }


def plot_summary(rows: list[dict], path: Path) -> None:
    b = np.asarray([row["B_mT"] for row in rows])
    fig, axes = plt.subplots(3, 2, figsize=(13, 13), sharex=True)
    axes[0, 0].plot(b, [row["rom_correlation"] for row in rows], "o-", label="ROM correlation")
    axes[0, 0].plot(b, [row["rom_skill_vs_persistence"] for row in rows], "s--", label="skill vs copy")
    axes[0, 0].axhline(0.0, color="0.5", lw=1)
    axes[0, 0].set_ylabel("Closure skill")
    axes[0, 0].legend(loc="lower right")

    axes[0, 1].plot(b, [row["physical_pcs95"] for row in rows], "o-", label="PCs for 95%")
    axes[0, 1].plot(b, [row["spatial_effective_mode_count"] for row in rows], "s--", label="effective azimuthal modes")
    axes[0, 1].set_ylabel("Effective dimension / count")
    axes[0, 1].legend(loc="upper left")

    axes[1, 0].plot(b, [row["mean_spectrum_entropy"] for row in rows], "o-", label="spatial entropy")
    axes[1, 0].plot(b, [row["pc1_temporal_spectral_entropy"] for row in rows], "s--", label="PC1 temporal entropy")
    axes[1, 0].set_ylabel("Normalized entropy")
    axes[1, 0].legend(loc="lower right")

    axes[1, 1].plot(b, [row["amplitude_weighted_phase_resultant"] for row in rows], "o-", label="MTSI-ECDI phase resultant")
    axes[1, 1].plot(b, [row["radial_pod1_overlap_min"] for row in rows], "s--", label="minimum radial POD1 overlap")
    axes[1, 1].set_ylabel("Coherence / overlap")
    axes[1, 1].set_ylim(-0.03, 1.03)
    axes[1, 1].legend(loc="lower right")

    axes[2, 0].plot(b, [row["bicoherence_effective_triad_count"] for row in rows], "o-", label="effective triads")
    axes[2, 0].plot(b, [row["bicoherence_pointwise_q99_exceedance_count"] for row in rows], "s--", label="q99 exceedances")
    axes[2, 0].set_ylabel("Bicoherence network size")
    axes[2, 0].legend(loc="upper left")

    axes[2, 1].plot(b, [row["rolling_dmd_eigenvalue_set_drift"] for row in rows], "o-", label="DMD eigen-set drift")
    axes[2, 1].plot(b, [row["bicoherence_network_cosine_drift_mean"] for row in rows], "s--", label="network drift")
    axes[2, 1].set_ylabel("Window-to-window drift")
    axes[2, 1].legend(loc="upper left")
    for axis in axes[-1]:
        axis.set_xlabel("Bx [mT]")
        axis.set_xticks(b)
    fig.suptitle("Magnetic-sweep dynamical organization and local ROM closure")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_rolling_dmd(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for b in B_VALUES:
        selected = [row for row in rows if row["B_mT"] == b]
        center = [(row["window_start_us"] + row["window_end_us"]) / 2.0 for row in selected]
        axes[0].plot(center, [row["dominant_frequency_mhz"] for row in selected], marker="o", ms=3, label=f"B{b}")
        axes[1].plot(center, [row["spectral_radius"] for row in selected], marker="o", ms=3, label=f"B{b}")
    axes[0].set_ylabel("Dominant DMD frequency [MHz]")
    axes[0].legend(ncol=5, loc="upper right")
    axes[1].axhline(1.0, color="0.4", lw=1)
    axes[1].set_ylabel("Spectral radius")
    axes[1].set_xlabel("Rolling-window center time [us]")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_bicoherence_rolling(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for b in B_VALUES:
        selected = [row for row in rows if row["B_mT"] == b]
        center = [(row["window_start_us"] + row["window_end_us"]) / 2.0 for row in selected]
        axes[0].plot(center, [row["effective_triad_count"] for row in selected], marker="o", ms=3, label=f"B{b}")
        axes[1].plot(center, [row["adjacent_cosine_distance"] for row in selected], marker="o", ms=3, label=f"B{b}")
    axes[0].set_ylabel("Effective bicoherence triads")
    axes[0].legend(ncol=5, loc="upper right")
    axes[1].set_ylabel("Adjacent network cosine distance")
    axes[1].set_xlabel("Rolling-window center time [us]")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_readme(rows: list[dict]) -> None:
    by_b = {row["B_mT"]: row for row in rows}
    lines = [
        "# Magnetic-sweep dynamical-organization diagnosis",
        "",
        "Cases: Bx=10, 15, 20, 25, 30 mT at Ez=10 kV/m.",
        "The common analysis interval is 20--30 us. The calculation reuses the",
        "radial-band Fourier caches from the previous ROM comparison and does not",
        "retrain SimVPv2.",
        "",
        "## Question",
        "",
        "The previous comparison found nearly perfect local ROM closure at B15,",
        "while B20 shares the nominal ECDI/MTSI coexistence regime but closes much",
        "more weakly. This analysis tests whether B15 is distinguished by spectral",
        "concentration, phase locking, a sparse/stationary quadratic-coupling network,",
        "stable radial structures, or stable rolling DMD eigenvalues.",
        "",
        "## Common summary",
        "",
        "| B [mT] | PCs95 | ROM corr | spatial entropy | temporal entropy | weighted phase R | min radial overlap | effective triads | network drift | DMD eigendrift |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for b in B_VALUES:
        row = by_b[b]
        lines.append(
            "| {B_mT} | {physical_pcs95} | {rom_correlation:.4f} | "
            "{mean_spectrum_entropy:.3f} | {pc1_temporal_spectral_entropy:.3f} | "
            "{amplitude_weighted_phase_resultant:.3f} | {radial_pod1_overlap_min:.3f} | "
            "{bicoherence_effective_triad_count:.1f} | "
            "{bicoherence_network_cosine_drift_mean:.3f} | "
            "{rolling_dmd_eigenvalue_set_drift:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Metric definitions and cautions",
            "",
            "- Spatial entropy is the normalized Shannon entropy of the mean phi",
            "  azimuthal-mode power for n=1..48. Lower values mean concentration into",
            "  fewer modes; they do not by themselves prove low-dimensional dynamics.",
            "- The generalized phase is n_M*phase(phi_E)-n_E*phase(phi_M), which is",
            "  invariant under an azimuthal coordinate shift. A resultant near one is",
            "  evidence for a stationary integer phase relation. It is not a causal test.",
            "- Bicoherence is computed for spatial triads a+b=c with radial bands and",
            "  time samples as an ensemble. `q99 exceedance` is pointwise relative to",
            "  200 independent circular-shift surrogates and is exploratory; it is not",
            "  family-wise significance. Bicoherence does not determine energy-flow direction.",
            "- Radial overlap in the summary compares the leading complex POD mode from",
            "  20--24 us with 24--27 and 27--30 us. Rank-2 overlap is also saved, but",
            "  can be ill-conditioned when the second mode carries negligible energy.",
            "- Rolling DMD uses a fixed PCA basis fitted on 20--24 us and 2 us windows",
            "  stepped by 0.5 us. Eigen-set drift is the mean optimal-matching distance",
            "  between eigenvalue sets in adjacent windows.",
            "",
            "## Interpretation",
            "",
            "The numerical interpretation is intentionally generated after inspecting",
            "the CSVs. See `case_summary.csv` together with the research memo for the",
            "full conclusion and limitations.",
            "",
            "## Files",
            "",
            "- `case_summary.csv`",
            "- `phase_locking.csv`",
            "- `radial_structure_overlap.csv`",
            "- `rolling_dmd.csv`",
            "- `bicoherence_triads.csv`",
            "- `bicoherence_rolling.csv`",
            "- `dynamical_organization_summary.png`",
            "- `rolling_dmd_stability.png`",
            "- `bicoherence_network_stability.png`",
            "- `analysis_summary.json`",
        ]
    )
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected_rom, pca_components, stationarity = closure_rows()
    modes = mode_rows()
    rng = np.random.default_rng(RNG_SEED)

    phase_rows: list[dict] = []
    radial_rows: list[dict] = []
    dmd_rows: list[dict] = []
    triad_rows: list[dict] = []
    bicoherence_rolling_rows: list[dict] = []
    case_rows: list[dict] = []

    for b_mt in B_VALUES:
        print(f"[CASE] B{b_mt}", flush=True)
        case = load_case(b_mt)
        mtsi_mode, ecdi_mode = modes[b_mt]
        spectral = spectral_diagnostics(case)
        phase = phase_locking_diagnostics(case, mtsi_mode, ecdi_mode)
        radial = radial_structure_diagnostics(case, mtsi_mode, ecdi_mode)
        rolling, dmd_summary = rolling_dmd_diagnostics(case, pca_components[b_mt])
        triads, bico_rolling, bico_summary = bicoherence_network_diagnostics(case, rng)

        phase_rows.append(phase)
        radial_rows.extend(radial)
        dmd_rows.extend(rolling)
        triad_rows.extend(triads)
        bicoherence_rolling_rows.extend(bico_rolling)

        rom = selected_rom[b_mt]
        recurrence = stationarity[b_mt]
        row = {
            "B_mT": b_mt,
            "mtsi_mode": mtsi_mode,
            "ecdi_mode": ecdi_mode,
            "physical_pcs95": pca_components[b_mt],
            "rom_method": rom["method"],
            "rom_skill_vs_persistence": float(rom["skill_vs_persistence"]),
            "rom_skill_vs_training_mean": float(rom["skill_vs_training_mean"]),
            "rom_correlation": float(rom["correlation"]),
            "recurrence_period_us": float(recurrence["best_recurrence_period_us"]),
            "recurrence_correlation": float(recurrence["best_recurrence_correlation"]),
            **{key: value for key, value in spectral.items() if key != "B_mT"},
            **{key: value for key, value in phase.items() if key != "B_mT"},
            "radial_pod1_overlap_mean": float(np.mean([item["pod1_overlap"] for item in radial])),
            "radial_pod1_overlap_min": float(np.min([item["pod1_overlap"] for item in radial])),
            "radial_pod2_overlap_mean": float(np.mean([item["pod2_overlap"] for item in radial])),
            "radial_pod2_overlap_min": float(np.min([item["pod2_overlap"] for item in radial])),
            **{key: value for key, value in dmd_summary.items() if key != "B_mT"},
            **{key: value for key, value in bico_summary.items() if key != "B_mT"},
        }
        case_rows.append(row)

    write_csv(OUTPUT / "case_summary.csv", case_rows)
    write_csv(OUTPUT / "phase_locking.csv", phase_rows)
    write_csv(OUTPUT / "radial_structure_overlap.csv", radial_rows)
    write_csv(OUTPUT / "rolling_dmd.csv", dmd_rows)
    write_csv(OUTPUT / "bicoherence_triads.csv", triad_rows)
    write_csv(OUTPUT / "bicoherence_rolling.csv", bicoherence_rolling_rows)
    plot_summary(case_rows, OUTPUT / "dynamical_organization_summary.png")
    plot_rolling_dmd(dmd_rows, OUTPUT / "rolling_dmd_stability.png")
    plot_bicoherence_rolling(
        bicoherence_rolling_rows, OUTPUT / "bicoherence_network_stability.png"
    )
    write_readme(case_rows)
    summary = {
        "status": "PASS",
        "cases_B_mT": list(B_VALUES),
        "Ez_kVm": 10.0,
        "analysis_interval_us": [ANALYSIS_START_US, ANALYSIS_END_US],
        "case_summary": case_rows,
        "phase_locking": phase_rows,
        "radial_structure": radial_rows,
        "rolling_dmd": dmd_rows,
        "bicoherence_rolling": bicoherence_rolling_rows,
        "bicoherence_surrogates": BICOHERENCE_SURROGATES,
        "random_seed": RNG_SEED,
    }
    (OUTPUT / "analysis_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=True), encoding="utf-8"
    )
    print(f"[PASS] wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
