"""Summarize factorized data/physics RadAz forecasts at fixed horizons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

import analyze_radaz_augmented_physical_state_dynamics as augmented
import train_radaz_direct_physical_state_rom as direct


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "workdirs" / "build_radaz_state_phase_factorized_rom"
DEFAULT_E2E = ROOT / "workdirs" / "train_radaz_factorized_end_to_end_physics_rom"
DEFAULT_DECODER = ROOT / "workdirs" / "evaluate_radaz_factorized_physics_decoder"
DEFAULT_OUTPUT = ROOT / "workdirs" / "compare_radaz_factorized_physics_horizons"
HORIZONS_US = (0.15, 0.30, 0.60, 1.20, 3.00)


def load_coefficients(path: Path, prediction_key: str) -> dict:
    with h5py.File(path, "r") as handle:
        return {
            "time_us": np.asarray(handle["time_us"], dtype=np.float64),
            "modes": np.asarray(handle["selected_mode_numbers"], dtype=np.int64)
            if "selected_mode_numbers" in handle
            else None,
            "truth": np.asarray(handle["truth_physical_coefficients"]),
            "prediction": np.asarray(handle[prediction_key]),
            "persistence": np.asarray(
                handle["raw_persistence_physical_coefficients"]
            ),
        }


def require_same(reference: dict, candidate: dict, name: str) -> None:
    for key in ("time_us", "truth", "persistence"):
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"{name} mismatch: {key}")


def scalar_row(
    label: str,
    horizon_us: float,
    count: int,
    values: dict,
    modes: np.ndarray,
    residual_scale: np.ndarray,
    truth_floor_power: np.ndarray,
) -> dict:
    truth = values["truth"][:count]
    prediction = values["prediction"][:count]
    persistence = values["persistence"][:count]
    row = {
        "model": label,
        "horizon_us": float(horizon_us),
        "samples": int(count),
    }
    for field_index, field in ((0, "phi"), (3, "efy")):
        metrics = augmented.scalar_metrics(
            truth[:, field_index],
            prediction[:, field_index],
            persistence[:, field_index],
        )
        row[f"{field}_nrmse"] = metrics["nrmse"]
        row[f"{field}_skill_vs_persistence"] = metrics[
            "skill_vs_persistence"
        ]
    for mode in (2, 7):
        index = int(np.flatnonzero(modes == mode)[0])
        metrics = augmented.scalar_metrics(
            np.abs(truth[:, 0, index]),
            np.abs(prediction[:, 0, index]),
            np.abs(persistence[:, 0, index]),
        )
        row[f"n{mode}_amplitude_skill"] = metrics["skill_vs_persistence"]
    wave_numbers = 2.0 * np.pi * modes / direct.AZIMUTHAL_LENGTH_M
    residual = prediction[:, 3] + 1j * wave_numbers[None] * prediction[:, 0]
    power = np.abs(residual / residual_scale[None]) ** 2
    row["field_gradient_residual_over_ey_rms"] = float(
        np.sqrt(np.mean(np.abs(residual) ** 2))
        / np.sqrt(np.mean(np.abs(prediction[:, 3]) ** 2))
    )
    row["field_gradient_excess_hinge"] = float(
        np.mean(np.maximum(power - truth_floor_power[None], 0.0))
    )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--end-to-end", type=Path, default=DEFAULT_E2E)
    parser.add_argument("--decoder", type=Path, default=DEFAULT_DECODER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = load_coefficients(
        args.data.resolve() / "development_rollout_35to40us.h5",
        "prediction_physical_coefficients",
    )
    end_to_end = load_coefficients(
        args.end_to_end.resolve() / "selected_development_rollout.h5",
        "prediction_physical_coefficients",
    )
    decoded = load_coefficients(
        args.decoder.resolve() / "selected_development_rollout.h5",
        "physics_prediction",
    )
    require_same(data, end_to_end, "end-to-end")
    require_same(data, decoded, "decoder")
    modes = data["modes"]
    if modes is None:
        raise ValueError("Data-only rollout has no selected mode numbers")
    with h5py.File(
        args.decoder.resolve() / "selected_development_rollout.h5", "r"
    ) as handle:
        residual_scale = np.asarray(handle["residual_scale"], dtype=np.float64)
        truth_floor_power = np.asarray(
            handle["truth_floor_power"], dtype=np.float64
        )
    relative_time = data["time_us"] - data["time_us"][0]
    full_horizon = float(relative_time[-1])
    requested = [*HORIZONS_US, full_horizon]
    rows = []
    for horizon in requested:
        count = int(np.searchsorted(relative_time, horizon + 1.0e-10, side="right"))
        for label, values in (
            ("factorized_data_only", data),
            ("end_to_end_physics", end_to_end),
            ("truth_floor_physics_decoder", decoded),
        ):
            rows.append(
                scalar_row(
                    label,
                    horizon,
                    count,
                    values,
                    modes,
                    residual_scale,
                    truth_floor_power,
                )
            )
    fields = list(rows[0])
    with (output / "horizon_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "horizon_comparison.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    availability = {
        "development_start_us": float(data["time_us"][0]),
        "development_end_us": float(data["time_us"][-1]),
        "available_horizon_us": full_horizon,
        "evaluated_horizons_us": requested,
        "requested_6us_available": bool(full_horizon >= 6.0),
        "primary_E25_to_E22p5_loaded": False,
    }
    (output / "availability.json").write_text(
        json.dumps(availability, indent=2), encoding="utf-8"
    )
    lines = [
        "# Factorized RadAz horizon comparison",
        "",
        f"Available development horizon: {full_horizon:.3f} us. "
        "The requested 6.0-us horizon is unavailable and was not extrapolated.",
        "",
        "| model | horizon us | phi NRMSE | Ey NRMSE | n=2 amp skill | n=7 amp skill | physics hinge |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['horizon_us']:.3f} | "
            f"{row['phi_nrmse']:.3f} | {row['efy_nrmse']:.3f} | "
            f"{row['n2_amplitude_skill']:+.3f} | {row['n7_amplitude_skill']:+.3f} | "
            f"{row['field_gradient_excess_hinge']:.3e} |"
        )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(availability, indent=2), flush=True)


if __name__ == "__main__":
    main()
