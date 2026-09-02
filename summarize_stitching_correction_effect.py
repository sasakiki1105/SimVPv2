import csv
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent
WORKDIRS = ROOT / "workdirs"
TEMP = WORKDIRS / "_stitching_effect_temp"
OUTDIR = WORKDIRS / "compare_stitching_correction_effect"
CHANNELS = ["electron_den", "ion_den", "phi"]


H5_CASES = {
    "lowmag_3a_stride2": (
        TEMP / "lowmag_step2_legacy.h5",
        Path(
            r"C:\Users\astro\research\PEPAPIC\test\results"
            r"\2D_ExB_low_magnet_dt2.5e-9_maxt50e-6_macro5"
            r"\SimVPv2_inputs"
            r"\global_norm_from_high3b_trainfixed_minmax_margin20_t0_to_t4000_step2_training_compatible.h5"
        ),
    ),
    "exhigh_5us_step20": (
        TEMP / "exhigh_fine_step20_legacy.h5",
        Path(
            r"C:\Users\astro\research\PEPAPIC\test\results"
            r"\2D_ExB_exhigh_magnet_dt2.5e-10_maxt5e-6_macro5"
            r"\SimVPv2_inputs"
            r"\global_norm_from_high3b_trainfixed_minmax_margin20_t0_to_t4000_step20_training_compatible.h5"
        ),
    ),
    "exhigh_50us_step2": (
        TEMP / "exhigh_50us_step2_legacy.h5",
        Path(
            r"C:\Users\astro\research\PEPAPIC\test\results"
            r"\2D_ExB_exhigh_magnet_dt2.5e-9_maxt50e-6_macro5"
            r"\SimVPv2_inputs"
            r"\global_norm_from_high3b_trainfixed_minmax_margin20_t0_to_t4000_step2_training_compatible.h5"
        ),
    ),
}


PREDICTION_CASES = {
    "lowmag_3a_data_only": (
        TEMP / "lowmag_legacy_baseline" / "low_magnet_direct10_raw_predictions.csv",
        WORKDIRS
        / "transfer_low_magnet_stride2_direct10_from_high3b_training_compatible"
        / "low_magnet_direct10_raw_predictions.csv",
    ),
    "lowmag_3a_floor_alpha11": (
        TEMP / "lowmag_legacy_floor_alpha11" / "low_magnet_direct10_raw_predictions.csv",
        WORKDIRS
        / "transfer_low_magnet_stride2_direct10_from_high3b_floor_hinge_lam1em3_alpha11_training_compatible"
        / "low_magnet_direct10_raw_predictions.csv",
    ),
    "exhigh_5us_data_only": (
        TEMP / "exhigh_fine_legacy_baseline" / "low_magnet_direct10_raw_predictions.csv",
        WORKDIRS
        / "transfer_exhigh_fine_step20_direct10_from_high3b_data_only_training_compatible"
        / "low_magnet_direct10_raw_predictions.csv",
    ),
    "exhigh_5us_floor_alpha12": (
        TEMP / "exhigh_fine_legacy_floor_alpha12" / "low_magnet_direct10_raw_predictions.csv",
        WORKDIRS
        / "transfer_exhigh_fine_step20_direct10_from_high3b_floor_hinge_training_compatible"
        / "low_magnet_direct10_raw_predictions.csv",
    ),
    "exhigh_50us_data_only": (
        TEMP / "exhigh_50us_legacy_baseline" / "low_magnet_direct10_raw_predictions.csv",
        WORKDIRS
        / "transfer_exhigh_50us_step2_direct10_from_high3b_data_only_training_compatible"
        / "low_magnet_direct10_raw_predictions.csv",
    ),
    "exhigh_50us_floor_alpha12": (
        TEMP / "exhigh_50us_legacy_floor_alpha12" / "low_magnet_direct10_raw_predictions.csv",
        WORKDIRS
        / "transfer_exhigh_50us_step2_direct10_from_high3b_floor_hinge_training_compatible"
        / "low_magnet_direct10_raw_predictions.csv",
    ),
}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def h5_difference_rows():
    rows = []
    for case, (legacy_path, corrected_path) in H5_CASES.items():
        with h5py.File(legacy_path, "r") as legacy, h5py.File(corrected_path, "r") as corrected:
            old = legacy["data_tchw"]
            new = corrected["data_tchw"]
            if old.shape != new.shape:
                raise ValueError(f"Shape mismatch for {case}: {old.shape} != {new.shape}")
            for channel_index, channel in enumerate(CHANNELS):
                sums = defaultdict(float)
                max_abs = 0.0
                changed = 0
                count = 0
                for start in range(0, old.shape[0], 32):
                    stop = min(start + 32, old.shape[0])
                    a = np.asarray(old[start:stop, channel_index], dtype=np.float64)
                    b = np.asarray(new[start:stop, channel_index], dtype=np.float64)
                    diff = b - a
                    count += diff.size
                    changed += int(np.count_nonzero(np.abs(diff) > 1.0e-7))
                    max_abs = max(max_abs, float(np.max(np.abs(diff))))
                    sums["abs"] += float(np.sum(np.abs(diff)))
                    sums["sq"] += float(np.sum(diff * diff))
                    sums["a"] += float(np.sum(a))
                    sums["b"] += float(np.sum(b))
                    sums["aa"] += float(np.sum(a * a))
                    sums["bb"] += float(np.sum(b * b))
                    sums["ab"] += float(np.sum(a * b))
                cov = sums["ab"] - sums["a"] * sums["b"] / count
                var_a = sums["aa"] - sums["a"] ** 2 / count
                var_b = sums["bb"] - sums["b"] ** 2 / count
                corr = cov / np.sqrt(var_a * var_b)
                rows.append(
                    {
                        "case": case,
                        "channel": channel,
                        "n_values": count,
                        "changed_fraction_gt_1e-7": changed / count,
                        "mae_correct_minus_legacy": sums["abs"] / count,
                        "rmse_correct_minus_legacy": np.sqrt(sums["sq"] / count),
                        "max_abs_difference": max_abs,
                        "legacy_corrected_correlation": corr,
                    }
                )
    return rows


def aggregate_prediction_csv(path):
    values = defaultdict(lambda: defaultdict(list))
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            channel = row["channel"]
            for metric in ["model_mse", "copy_mse", "model_over_copy", "corr", "copy_corr"]:
                value = float(row[metric])
                if np.isfinite(value):
                    values[channel][metric].append(value)
    return {
        channel: {
            metric: float(np.mean(metric_values))
            for metric, metric_values in channel_values.items()
        }
        for channel, channel_values in values.items()
    }


def prediction_difference_rows():
    rows = []
    for case, (legacy_path, corrected_path) in PREDICTION_CASES.items():
        legacy = aggregate_prediction_csv(legacy_path)
        corrected = aggregate_prediction_csv(corrected_path)
        for channel in CHANNELS:
            for metric in ["model_mse", "copy_mse", "model_over_copy", "corr", "copy_corr"]:
                old = legacy[channel][metric]
                new = corrected[channel][metric]
                rows.append(
                    {
                        "case": case,
                        "channel": channel,
                        "metric": metric,
                        "legacy_mean": old,
                        "corrected_mean": new,
                        "corrected_minus_legacy": new - old,
                        "percent_change": 100.0 * (new - old) / old if old != 0 else np.nan,
                    }
                )
    return rows


def fft_peak_rows():
    def read_peaks(path):
        peaks = {}
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row["channel_name"], int(row["rank"]))
                peaks[key] = row
        return peaks

    legacy = read_peaks(TEMP / "fft_lowmag_legacy_step2" / "temporal_fft_top_peaks.csv")
    corrected = read_peaks(TEMP / "fft_lowmag_correct_step2" / "temporal_fft_top_peaks.csv")
    rows = []
    for key in sorted(legacy):
        old = legacy[key]
        new = corrected[key]
        old_freq = float(old["frequency_mhz"])
        new_freq = float(new["frequency_mhz"])
        old_power = float(old["power_norm"])
        new_power = float(new["power_norm"])
        rows.append(
            {
                "channel": key[0],
                "peak_rank": key[1],
                "legacy_frequency_mhz": old_freq,
                "corrected_frequency_mhz": new_freq,
                "frequency_shift_mhz": new_freq - old_freq,
                "legacy_power_norm": old_power,
                "corrected_power_norm": new_power,
                "power_norm_percent_change": (
                    100.0 * (new_power - old_power) / old_power if old_power != 0 else np.nan
                ),
            }
        )
    return rows


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    h5_rows = h5_difference_rows()
    prediction_rows = prediction_difference_rows()
    fft_rows = fft_peak_rows()
    write_csv(OUTDIR / "h5_data_difference.csv", h5_rows)
    write_csv(OUTDIR / "prediction_metric_difference.csv", prediction_rows)
    write_csv(OUTDIR / "fft_peak_difference.csv", fft_rows)
    print(f"wrote {OUTDIR}")


if __name__ == "__main__":
    main()
