"""Lock a mode-specific RadAz ROM ensemble on the up-sweep development split.

This builder does not train or read the primary E25 -> E22.5 trajectory.  It
combines already-locked development models as follows:

* low-dimensional regime-aware branch: composite low state, radial state,
  and MTSI transport;
* direct physical-state branch: ECDI transport;
* autonomous carrier-envelope branch: n=2 and n=8--21 field coefficients;
* time-controlled carrier-envelope branch: n=7 field coefficients.

The n=2/n=7 expert choice is development-set model selection, and is recorded
explicitly in the output lock.  Field skill uses the same raw-Fourier
persistence stored by the factorized ROM, not carrier-frame persistence.
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
DEFAULT_BASE = ROOT / "workdirs" / "build_radaz_mode_factorized_rom"
DEFAULT_AUTONOMOUS = (
    ROOT / "workdirs" / "train_radaz_carrier_envelope_high_branch_rom"
)
DEFAULT_CONTROLLED = (
    ROOT / "workdirs" / "train_radaz_carrier_envelope_time_controlled_rom"
)
DEFAULT_OUTPUT = ROOT / "workdirs" / "build_radaz_mode_specific_expert_rom"


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


def write_csv(path: Path, row: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(json_safe(row))


def load_base(path: Path) -> dict:
    with h5py.File(path / "development_rollout_35to40us.h5", "r") as handle:
        result = {
            "time_us": np.asarray(handle["time_us"], dtype=np.float64),
            "frame": np.asarray(handle["frame"], dtype=np.int64),
            "mode_numbers": np.asarray(
                handle["selected_mode_numbers"], dtype=np.int64
            ),
        }
        for kind in ("truth", "prediction", "persistence"):
            result[kind] = {
                name: np.asarray(handle[f"{kind}/{name}"])
                for name in (
                    "composite_state",
                    "radial",
                    "transport",
                    "selected_physical_coefficients",
                )
            }
    return result


def load_carrier_coefficients(
    path: Path,
    carrier_origin_time_us: float,
) -> dict:
    rollout_path = path / "development_rollout_35to40us.h5"
    representation_path = path / "representation.h5"
    with h5py.File(rollout_path, "r") as handle:
        time_us = np.asarray(handle["time_us"], dtype=np.float64)
        frame = np.asarray(handle["frame"], dtype=np.int64)
        states = {
            kind: np.asarray(handle[kind], dtype=np.float64)
            for kind in ("truth", "prediction", "carrier_persistence")
        }
    with h5py.File(representation_path, "r") as handle:
        mode_numbers = np.asarray(
            handle["selected_mode_numbers"], dtype=np.int64
        )
        mean = np.asarray(
            handle["scaler/carrier_physical/mean"], dtype=np.float64
        )
        scale = np.asarray(
            handle["scaler/carrier_physical/scale"], dtype=np.float64
        )
        phase_step = np.asarray(
            handle["carrier/e20_to_e22p5/phase_step_rad"],
            dtype=np.float64,
        )[mode_numbers - 1]

    dt_us = float(np.median(np.diff(time_us)))
    first_index_float = (time_us[0] - carrier_origin_time_us) / dt_us
    first_index = int(round(first_index_float))
    if not np.isclose(first_index_float, first_index, atol=1.0e-7, rtol=0.0):
        raise ValueError("Carrier time origin does not map to an integer sample")
    indices = np.arange(first_index, first_index + len(time_us), dtype=np.float64)
    rotation = np.exp(
        1j * indices[:, None, None] * phase_step[None, None, :]
    )

    coefficients = {}
    carrier_slice = slice(10, 10 + len(mean))
    for kind, state in states.items():
        packed = state[:, carrier_slice] * scale + mean
        shaped = packed.reshape(
            len(state), len(direct.FIELD_NAMES), len(mode_numbers), 2
        )
        demodulated = shaped[..., 0] + 1j * shaped[..., 1]
        coefficients[kind] = demodulated * rotation
    return {
        "time_us": time_us,
        "frame": frame,
        "mode_numbers": mode_numbers,
        "first_carrier_index": first_index,
        "coefficients": coefficients,
    }


def phase_frequency_mhz(coefficients: np.ndarray, dt_us: float) -> float:
    products = coefficients[1:] * np.conj(coefficients[:-1])
    return float(np.angle(np.sum(products)) / (2.0 * np.pi * dt_us))


def evaluate(base: dict, mixed: dict) -> dict:
    truth = base["truth"]
    prediction = base["prediction"]
    persistence = base["persistence"]
    coefficients = {
        "truth": truth["selected_physical_coefficients"],
        "prediction": mixed["prediction"],
        "persistence": persistence["selected_physical_coefficients"],
    }
    state_metrics = augmented.scalar_metrics(
        truth["composite_state"],
        prediction["composite_state"],
        persistence["composite_state"],
    )
    radial_metrics = augmented.scalar_metrics(
        truth["radial"], prediction["radial"], persistence["radial"]
    )
    transport_metrics = augmented.scalar_metrics(
        truth["transport"], prediction["transport"], persistence["transport"]
    )
    row = {
        "method": "mode_specific_expert_rom",
        "finite_fraction": float(
            np.mean(np.isfinite(mixed["prediction"]))
        ),
        "state_skill_vs_persistence": state_metrics["skill_vs_persistence"],
        "state_temporal_anomaly_correlation": state_metrics[
            "temporal_anomaly_correlation"
        ],
        "radial_skill_vs_persistence": radial_metrics[
            "skill_vs_persistence"
        ],
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
            truth["transport"][:, band_index],
            prediction["transport"][:, band_index],
            persistence["transport"][:, band_index],
        )
        row[f"{band}_transport_skill"] = metrics["skill_vs_persistence"]
        row[f"{band}_transport_correlation"] = metrics["correlation"]

    for field_index, field in ((0, "phi"), (3, "efy")):
        metrics = augmented.scalar_metrics(
            coefficients["truth"][:, field_index],
            coefficients["prediction"][:, field_index],
            coefficients["persistence"][:, field_index],
        )
        row[f"selected_{field}_skill_vs_raw_persistence"] = metrics[
            "skill_vs_persistence"
        ]
        row[f"selected_{field}_nrmse"] = metrics["nrmse"]
        row[f"selected_{field}_temporal_anomaly_correlation"] = metrics[
            "temporal_anomaly_correlation"
        ]

    dt_us = float(np.median(np.diff(base["time_us"])))
    phi = {kind: values[:, 0] for kind, values in coefficients.items()}
    for mode in (2, 7):
        mode_index = int(np.flatnonzero(base["mode_numbers"] == mode)[0])
        metrics = augmented.scalar_metrics(
            np.abs(phi["truth"][:, mode_index]),
            np.abs(phi["prediction"][:, mode_index]),
            np.abs(phi["persistence"][:, mode_index]),
        )
        row[f"phi_n{mode}_amplitude_skill"] = metrics[
            "skill_vs_persistence"
        ]
        row[f"phi_n{mode}_amplitude_correlation"] = metrics["correlation"]
        truth_frequency = phase_frequency_mhz(
            phi["truth"][:, mode_index], dt_us
        )
        prediction_frequency = phase_frequency_mhz(
            phi["prediction"][:, mode_index], dt_us
        )
        row[f"phi_n{mode}_truth_frequency_MHz"] = truth_frequency
        row[f"phi_n{mode}_prediction_frequency_MHz"] = prediction_frequency
        row[f"phi_n{mode}_frequency_abs_error_MHz"] = abs(
            prediction_frequency - truth_frequency
        )

    k = 2.0 * np.pi * base["mode_numbers"] / direct.AZIMUTHAL_LENGTH_M
    residual = (
        coefficients["prediction"][:, 3]
        + 1j * k[None] * coefficients["prediction"][:, 0]
    )
    row["field_gradient_residual_over_ey_rms"] = float(
        np.sqrt(np.mean(np.abs(residual) ** 2))
        / np.sqrt(np.mean(np.abs(coefficients["prediction"][:, 3]) ** 2))
    )

    persistence_gates = (
        row["state_skill_vs_persistence"],
        row["radial_skill_vs_persistence"],
        row["MTSI_n1_6_transport_skill"],
        row["ECDI_n9_21_transport_skill"],
        row["selected_phi_skill_vs_raw_persistence"],
        row["selected_efy_skill_vs_raw_persistence"],
        row["phi_n2_amplitude_skill"],
        row["phi_n7_amplitude_skill"],
    )
    row["minimum_persistence_gate_skill"] = float(min(persistence_gates))
    row["passes_all_persistence_gates"] = bool(
        row["finite_fraction"] == 1.0
        and row["minimum_persistence_gate_skill"] > 0.0
    )
    row["passes_field_climatology_gate"] = bool(
        row["selected_phi_nrmse"] < 1.0
        and row["selected_efy_nrmse"] < 1.0
    )
    return row


def plot_rollout(path: Path, base: dict, mixed_prediction: np.ndarray) -> None:
    time_us = base["time_us"]
    truth_coeff = base["truth"]["selected_physical_coefficients"]
    persistence_coeff = base["persistence"]["selected_physical_coefficients"]
    figure, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    colors = {"truth": "black", "prediction": "tab:blue", "persistence": "0.65"}
    for axis, mode in zip(axes[:2], (2, 7)):
        index = int(np.flatnonzero(base["mode_numbers"] == mode)[0])
        axis.plot(
            time_us, np.abs(truth_coeff[:, 0, index]),
            color=colors["truth"], label="PIC truth",
        )
        axis.plot(
            time_us, np.abs(mixed_prediction[:, 0, index]),
            color=colors["prediction"], label="mode-specific ROM",
        )
        axis.plot(
            time_us, np.abs(persistence_coeff[:, 0, index]),
            color=colors["persistence"], linestyle="--", label="raw persistence",
        )
        axis.set_ylabel(f"|phi n={mode}|")
        axis.grid(alpha=0.25)
    for axis, band_index, label in zip(
        axes[2:], (0, 1), ("MTSI n=1--6", "ECDI n=9--21")
    ):
        for kind, style in (
            ("truth", "-"), ("prediction", "-"), ("persistence", "--")
        ):
            axis.plot(
                time_us,
                base[kind]["transport"][:, band_index],
                color=colors[kind],
                linestyle=style,
                label=("PIC truth" if kind == "truth" else
                       "mode-specific ROM" if kind == "prediction" else
                       "persistence"),
            )
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=3)
    axes[-1].set_xlabel("time [us]")
    figure.suptitle("Mode-specific expert ROM: E20 to E22.5 development rollout")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--autonomous", type=Path, default=DEFAULT_AUTONOMOUS)
    parser.add_argument("--controlled", type=Path, default=DEFAULT_CONTROLLED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--carrier-origin-time-us", type=float, default=30.165)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = args.base.resolve()
    autonomous_path = args.autonomous.resolve()
    controlled_path = args.controlled.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    base = load_base(base_path)
    autonomous = load_carrier_coefficients(
        autonomous_path, args.carrier_origin_time_us
    )
    controlled = load_carrier_coefficients(
        controlled_path, args.carrier_origin_time_us
    )
    for name, branch in (("autonomous", autonomous), ("controlled", controlled)):
        if not np.array_equal(base["frame"], branch["frame"]) or not np.allclose(
            base["time_us"], branch["time_us"], atol=1.0e-12, rtol=0.0
        ):
            raise ValueError(f"{name} carrier rollout alignment mismatch")
        if not np.array_equal(base["mode_numbers"], branch["mode_numbers"]):
            raise ValueError(f"{name} selected-mode mismatch")

    # Autonomous is the default field expert; controlled replaces only n=7.
    mixed_prediction = autonomous["coefficients"]["prediction"].copy()
    n7_index = int(np.flatnonzero(base["mode_numbers"] == 7)[0])
    mixed_prediction[:, :, n7_index] = controlled["coefficients"][
        "prediction"
    ][:, :, n7_index]
    mixed = {"prediction": mixed_prediction}
    metrics = evaluate(base, mixed)
    accepted = bool(
        metrics["passes_all_persistence_gates"]
        and metrics["passes_field_climatology_gate"]
    )
    status = (
        "READY_FOR_PHYSICS_ABLATION"
        if accepted
        else "PROVISIONAL_DATA_ONLY_DIAGNOSTIC"
        if metrics["passes_all_persistence_gates"]
        else "REJECTED_DEVELOPMENT"
    )

    write_csv(output / "development_metrics.csv", metrics)
    (output / "development_metrics.json").write_text(
        json.dumps(json_safe(metrics), indent=2), encoding="utf-8"
    )
    with h5py.File(output / "development_rollout_35to40us.h5", "w") as handle:
        handle.attrs["primary_E25_to_E22p5_loaded"] = False
        handle.attrs["selection_split"] = "E20_to_E22p5_35to40us"
        handle.create_dataset("time_us", data=base["time_us"])
        handle.create_dataset("frame", data=base["frame"])
        handle.create_dataset("selected_mode_numbers", data=base["mode_numbers"])
        for kind in ("truth", "prediction", "persistence"):
            group = handle.require_group(kind)
            group.create_dataset(
                "composite_state", data=base[kind]["composite_state"], compression="gzip"
            )
            group.create_dataset(
                "radial", data=base[kind]["radial"], compression="gzip"
            )
            group.create_dataset(
                "transport", data=base[kind]["transport"], compression="gzip"
            )
            coefficients = (
                mixed_prediction
                if kind == "prediction"
                else base[kind]["selected_physical_coefficients"]
            )
            group.create_dataset(
                "selected_physical_coefficients", data=coefficients, compression="gzip"
            )
    plot_rollout(
        output / "development_rollout_35to40us.png", base, mixed_prediction
    )

    factor_lock_path = base_path / "model_lock.json"
    factor_lock = json.loads(factor_lock_path.read_text(encoding="utf-8"))
    checkpoints = {
        "low_regime_aware": Path(factor_lock["low_checkpoint"]),
        "direct_ECDI_transport": Path(factor_lock["high_checkpoint"]),
        "autonomous_carrier": (
            autonomous_path / "carrier_envelope_high_branch_data_only.pt"
        ),
        "time_controlled_carrier": (
            controlled_path / "carrier_envelope_time_controlled_data_only.pt"
        ),
    }
    lock = {
        "status": status,
        "accepted_for_physics_ablation": accepted,
        "development_only": True,
        "development_metrics": metrics,
        "selection_policy": {
            "composite_low_state_radial_MTSI": "low_regime_aware",
            "ECDI_transport": "direct_physical_state",
            "field_modes_n2_n8_to_n21": "autonomous_carrier_envelope",
            "field_mode_n7": "time_controlled_carrier_envelope",
            "selection_data": "E20_to_E22p5, 35--40 us development split",
            "field_baseline": "raw Fourier persistence from factorized ROM",
            "posthoc_expert_selection_disclosed": True,
        },
        "gate_policy": {
            "all_persistence_skills_must_be_positive": True,
            "phi_and_Ey_climatology_nrmse_must_be_below_one": True,
            "physics_ablation_deferred_because_climatology_gate_failed": not accepted,
        },
        "carrier_indexing": {
            "origin_time_us": args.carrier_origin_time_us,
            "validation_first_index": autonomous["first_carrier_index"],
        },
        "primary_test": {
            "direction": "E25_to_E22.5",
            "data_loaded": False,
            "used_for_selection": False,
        },
        "checkpoints": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in checkpoints.items()
        },
        "component_model_locks": {
            "factorized": str(factor_lock_path),
            "autonomous_carrier": str(autonomous_path / "model_lock.json"),
            "time_controlled_carrier": str(controlled_path / "model_lock.json"),
        },
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "arguments": vars(args),
    }
    (output / "model_lock.json").write_text(
        json.dumps(json_safe(lock), indent=2), encoding="utf-8"
    )
    readme = f"""# Mode-specific expert RadAz ROM

- Status: `{status}`
- All raw-persistence skill gates pass: **{metrics['passes_all_persistence_gates']}**
- Field climatology gate (phi/Ey NRMSE < 1): **{metrics['passes_field_climatology_gate']}**
- MTSI transport skill: {metrics['MTSI_n1_6_transport_skill']:.6f}
- ECDI transport skill: {metrics['ECDI_n9_21_transport_skill']:.6f}
- phi field skill vs raw persistence: {metrics['selected_phi_skill_vs_raw_persistence']:.6f}
- Ey field skill vs raw persistence: {metrics['selected_efy_skill_vs_raw_persistence']:.6f}
- n=2 amplitude skill: {metrics['phi_n2_amplitude_skill']:.6f}
- n=7 amplitude skill: {metrics['phi_n7_amplitude_skill']:.6f}
- Primary E25 -> E22.5 data loaded: **no**

This is a development-selected, data-only diagnostic ensemble.  Its n=2 and
n=7 expert assignment is written explicitly into `model_lock.json`.  It is not
promoted to physics-loss training because phi/Ey NRMSE remains above one.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(json_safe(metrics), indent=2))
    print(f"status={status}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
