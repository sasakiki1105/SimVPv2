"""Lock a mode-factorized RadAz ROM from two development-only branches.

The low-mode branch is the 30D regime-aware ROM (radial state and MTSI
transport).  The high-mode branch is the 212D direct-physical-state ROM
(n=7--21 fields, ECDI radial/cross state, and ECDI transport).  Mode n=2 is
also retained from the direct branch for the requested carrier diagnostic.
No E25 -> E22.5 primary data are read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_radaz_augmented_physical_state_dynamics as augmented
import train_radaz_direct_physical_state_rom as direct


ROOT = Path(__file__).resolve().parent
DEFAULT_LOW = ROOT / "workdirs" / "train_radaz_regime_aware_transition_rom_h160"
DEFAULT_HIGH = ROOT / "workdirs" / "train_radaz_direct_physical_state_rom"
DEFAULT_OUTPUT = ROOT / "workdirs" / "build_radaz_mode_factorized_rom"
PHYSICS_MODE_INDICES = np.r_[1, 6:21]  # n=2 and n=7--21, zero based.


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for piece in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def load_direct_scaler(path: Path):
    means = {}
    scales = {}
    slices = {}
    with h5py.File(path, "r") as handle:
        macro_weights = np.asarray(handle["macro_weights"], dtype=np.float64)
        for name in direct.GROUPS:
            group = handle[f"scaler/{name}"]
            means[name] = np.asarray(group["mean"], dtype=np.float64)
            scales[name] = np.asarray(group["scale"], dtype=np.float64)
            slices[name] = slice(
                int(group.attrs["slice_start"]), int(group.attrs["slice_stop"])
            )
    return means, scales, slices, macro_weights


def inverse_group(
    state: np.ndarray,
    name: str,
    means: dict[str, np.ndarray],
    scales: dict[str, np.ndarray],
    slices: dict[str, slice],
) -> np.ndarray:
    return state[:, slices[name]] * scales[name] + means[name]


def write_csv(path: Path, row: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def plot_rollout(
    path: Path,
    time_us: np.ndarray,
    transport: dict[str, np.ndarray],
    coefficients: dict[str, np.ndarray],
) -> None:
    figure, axes = plt.subplots(4, 1, figsize=(10.5, 10.5), sharex=True)
    styles = (
        ("truth", "#111111", "-", "PIC truth"),
        ("prediction", "#0072B2", "-", "mode-factorized ROM"),
        ("persistence", "#999999", ":", "persistence"),
    )
    for band, label in enumerate(("MTSI n=1--6", "ECDI n=9--21")):
        for name, color, line, display in styles:
            axes[band].plot(
                time_us,
                transport[name][:, band],
                color=color,
                linestyle=line,
                label=display,
            )
        axes[band].set_ylabel(f"{label}\ntransport")
        axes[band].grid(alpha=0.25)
        axes[band].legend(fontsize=8)
    for axis_index, mode in enumerate((2, 7), start=2):
        for name, color, line, display in styles:
            axes[axis_index].plot(
                time_us,
                np.abs(coefficients[name][:, 0, mode - 1]),
                color=color,
                linestyle=line,
                label=display,
            )
        axes[axis_index].set_ylabel(f"global phi n={mode}\namplitude")
        axes[axis_index].grid(alpha=0.25)
        axes[axis_index].legend(fontsize=8)
    axes[-1].set_xlabel("time [us]")
    figure.suptitle("Mode-factorized ROM: E20 to E22.5 development rollout")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low", type=Path, default=DEFAULT_LOW)
    parser.add_argument("--high", type=Path, default=DEFAULT_HIGH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    low = args.low.resolve()
    high = args.high.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    low_rollout = low / "development_rollout_35to40us.h5"
    high_rollout = high / "development_rollout_35to40us.h5"
    with h5py.File(low_rollout, "r") as handle:
        time_us = np.asarray(handle["time_us"], dtype=np.float64)
        frame = np.asarray(handle["frame"], dtype=np.int64)
        low_state = {
            "truth": np.asarray(handle["truth/state"], dtype=np.float64),
            "prediction": np.asarray(
                handle["regime_aware_rom/state"], dtype=np.float64
            ),
            "persistence": np.asarray(
                handle["persistence/state"], dtype=np.float64
            ),
        }
        low_radial = {
            "truth": np.asarray(handle["truth/radial"], dtype=np.float64),
            "prediction": np.asarray(
                handle["regime_aware_rom/radial"], dtype=np.float64
            ),
            "persistence": np.asarray(
                handle["persistence/radial"], dtype=np.float64
            ),
        }
        low_transport = {
            "truth": np.asarray(handle["truth/transport"], dtype=np.float64),
            "prediction": np.asarray(
                handle["regime_aware_rom/transport"], dtype=np.float64
            ),
            "persistence": np.asarray(
                handle["persistence/transport"], dtype=np.float64
            ),
        }
    with h5py.File(high_rollout, "r") as handle:
        high_time = np.asarray(handle["time_us"], dtype=np.float64)
        high_frame = np.asarray(handle["frame"], dtype=np.int64)
        high_state = {
            name: np.asarray(handle[name], dtype=np.float64)
            for name in ("truth", "prediction", "persistence")
        }
    if not np.array_equal(frame, high_frame) or not np.allclose(
        time_us, high_time, atol=1.0e-12, rtol=0.0
    ):
        raise ValueError("Low/high branch rollout alignment mismatch")

    means, scales, slices, macro_weights = load_direct_scaler(
        high / "representation.h5"
    )
    decoded_high = {
        kind: {
            group: inverse_group(values, group, means, scales, slices)
            for group in direct.GROUPS
        }
        for kind, values in high_state.items()
    }
    high_radial_indices = np.asarray([1, 3, 5, 7], dtype=np.int64)
    high_cross_indices = np.arange(16).reshape(4, 2, 2)[:, 1, :].reshape(-1)
    high_physical_indices = (
        np.arange(168).reshape(4, 21, 2)[:, 6:, :].reshape(-1)
    )
    standardized_high_indices = []
    for group, indices in (
        ("physical_fourier", high_physical_indices),
        ("radial", high_radial_indices),
        ("cross", high_cross_indices),
    ):
        standardized_high_indices.extend(
            (slices[group].start + indices).tolist()
        )
    standardized_high_indices = np.asarray(
        standardized_high_indices, dtype=np.int64
    )
    composite_state = {
        kind: np.concatenate(
            (low_state[kind], high_state[kind][:, standardized_high_indices]),
            axis=1,
        )
        for kind in low_state
    }

    high_coefficients = {
        kind: direct.unpack_physical_fourier(
            decoded_high[kind]["physical_fourier"]
        )
        for kind in decoded_high
    }
    high_cross = {
        kind: direct.unpack_cross(decoded_high[kind]["cross"])
        for kind in decoded_high
    }
    high_transport = {
        kind: augmented.transport_from_cross(values, macro_weights)
        for kind, values in high_cross.items()
    }
    hybrid_transport = {
        kind: np.column_stack(
            (low_transport[kind][:, 0], high_transport[kind][:, 1])
        )
        for kind in low_transport
    }

    state_metrics = augmented.scalar_metrics(
        composite_state["truth"],
        composite_state["prediction"],
        composite_state["persistence"],
    )
    radial_metrics = augmented.scalar_metrics(
        low_radial["truth"],
        low_radial["prediction"],
        low_radial["persistence"],
    )
    transport_metrics = augmented.scalar_metrics(
        hybrid_transport["truth"],
        hybrid_transport["prediction"],
        hybrid_transport["persistence"],
    )
    row = {
        "method": "mode_factorized_rom",
        "finite_fraction": float(np.mean(np.isfinite(composite_state["prediction"]))),
        "composite_state_dimension": int(composite_state["truth"].shape[1]),
        "state_skill_vs_persistence": state_metrics["skill_vs_persistence"],
        "state_temporal_anomaly_correlation": state_metrics[
            "temporal_anomaly_correlation"
        ],
        "radial_skill_vs_persistence": radial_metrics["skill_vs_persistence"],
        "radial_temporal_anomaly_correlation": radial_metrics[
            "temporal_anomaly_correlation"
        ],
        "transport_skill_vs_persistence": transport_metrics[
            "skill_vs_persistence"
        ],
        "transport_temporal_anomaly_correlation": transport_metrics[
            "temporal_anomaly_correlation"
        ],
    }
    for band_index, band in enumerate(("MTSI_n1_6", "ECDI_n9_21")):
        metrics = augmented.scalar_metrics(
            hybrid_transport["truth"][:, band_index],
            hybrid_transport["prediction"][:, band_index],
            hybrid_transport["persistence"][:, band_index],
        )
        row[f"{band}_transport_skill"] = metrics["skill_vs_persistence"]
        row[f"{band}_transport_correlation"] = metrics["correlation"]
    for field_index, field in ((0, "phi"), (3, "efy")):
        selected = {
            kind: coefficients[:, field_index, PHYSICS_MODE_INDICES]
            for kind, coefficients in high_coefficients.items()
        }
        metrics = augmented.scalar_metrics(
            selected["truth"], selected["prediction"], selected["persistence"]
        )
        row[f"selected_{field}_skill_vs_persistence"] = metrics[
            "skill_vs_persistence"
        ]
        row[f"selected_{field}_nrmse"] = metrics["nrmse"]
        row[f"selected_{field}_temporal_anomaly_correlation"] = metrics[
            "temporal_anomaly_correlation"
        ]
    for mode in (2, 7):
        mode_index = mode - 1
        metrics = augmented.scalar_metrics(
            np.abs(high_coefficients["truth"][:, 0, mode_index]),
            np.abs(high_coefficients["prediction"][:, 0, mode_index]),
            np.abs(high_coefficients["persistence"][:, 0, mode_index]),
        )
        row[f"phi_n{mode}_amplitude_skill"] = metrics[
            "skill_vs_persistence"
        ]
        row[f"phi_n{mode}_amplitude_correlation"] = metrics["correlation"]

    accepted = bool(
        row["finite_fraction"] == 1.0
        and row["state_skill_vs_persistence"] > 0.0
        and row["radial_skill_vs_persistence"] > 0.0
        and row["MTSI_n1_6_transport_skill"] > 0.0
        and row["ECDI_n9_21_transport_skill"] > 0.0
        and row["selected_phi_skill_vs_persistence"] > 0.0
        and row["selected_efy_skill_vs_persistence"] > 0.0
    )
    status = "READY_FOR_PHYSICS_ABLATION" if accepted else "REJECTED_DEVELOPMENT"

    write_csv(output / "development_metrics.csv", row)
    (output / "development_metrics.json").write_text(
        json.dumps(json_safe(row), indent=2), encoding="utf-8"
    )
    with h5py.File(output / "development_rollout_35to40us.h5", "w") as handle:
        handle.attrs["primary_E25_to_E22p5_loaded"] = False
        handle.create_dataset("time_us", data=time_us)
        handle.create_dataset("frame", data=frame)
        for kind in ("truth", "prediction", "persistence"):
            group = handle.require_group(kind)
            group.create_dataset(
                "composite_state", data=composite_state[kind], compression="gzip"
            )
            group.create_dataset(
                "radial", data=low_radial[kind], compression="gzip"
            )
            group.create_dataset(
                "transport", data=hybrid_transport[kind], compression="gzip"
            )
            group.create_dataset(
                "selected_physical_coefficients",
                data=high_coefficients[kind][:, :, PHYSICS_MODE_INDICES],
                compression="gzip",
            )
        handle.create_dataset(
            "selected_mode_numbers", data=PHYSICS_MODE_INDICES + 1
        )
    plot_rollout(
        output / "development_rollout_35to40us.png",
        time_us,
        hybrid_transport,
        high_coefficients,
    )

    low_checkpoint = low / "regime_aware_transition_rom_data_only.pt"
    high_checkpoint = high / "direct_physical_state_rom_data_only.pt"
    lock = {
        "status": status,
        "accepted_for_physics_ablation": accepted,
        "development_metrics": row,
        "factorization": {
            "low_branch": "30D regime-aware ROM: radial and MTSI transport",
            "high_branch": (
                "212D direct physical-state ROM: n=7--21 physical state, "
                "ECDI radial/cross state and ECDI transport; n=2 diagnostic"
            ),
            "physics_modes": (PHYSICS_MODE_INDICES + 1).tolist(),
        },
        "primary_test": {
            "direction": "E25_to_E22.5",
            "data_loaded": False,
            "used_for_selection": False,
        },
        "low_checkpoint": str(low_checkpoint),
        "low_checkpoint_sha256": sha256(low_checkpoint),
        "high_checkpoint": str(high_checkpoint),
        "high_checkpoint_sha256": sha256(high_checkpoint),
        "low_model_lock": str(low / "model_lock.json"),
        "high_model_lock": str(high / "model_lock.json"),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    (output / "model_lock.json").write_text(
        json.dumps(json_safe(lock), indent=2), encoding="utf-8"
    )
    readme = f"""# Mode-factorized RadAz ROM

- Status: `{status}`
- Composite state skill: {row['state_skill_vs_persistence']:.6f}
- Radial skill: {row['radial_skill_vs_persistence']:.6f}
- MTSI transport skill: {row['MTSI_n1_6_transport_skill']:.6f}
- ECDI transport skill: {row['ECDI_n9_21_transport_skill']:.6f}
- Selected phi skill: {row['selected_phi_skill_vs_persistence']:.6f}
- Selected Ey skill: {row['selected_efy_skill_vs_persistence']:.6f}
- Primary E25 -> E22.5 data loaded: **no**

This is a locked two-branch manifest, not a post-hoc change to either branch.
Both branches were trained and selected only on the up-sweep development split.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(json_safe(row), indent=2))
    print(f"status={status}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
