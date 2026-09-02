"""Lock a factorized state-history amplitude / recurrent phase RadAz ROM.

The history-conditioned branch supplies prospective n=2 and n=7 phi
amplitudes.  The earlier mode-separated recurrent branch supplies complex
phase, all other mode amplitudes, and the relative phase between phi and the
other physical fields.  For n=2/n=7 one causal amplitude ratio rescales every
field coefficient of the recurrent branch.  No PIC truth enters the fusion.

The factorization was selected on the E20 -> E22.5 development trajectory and
is disclosed as such.  The primary E25 -> E22.5 trajectory remains unread.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

import analyze_radaz_augmented_physical_state_dynamics as augmented
import train_radaz_carrier_envelope_high_branch_rom as carrier
import train_radaz_direct_physical_state_rom as direct


ROOT = Path(__file__).resolve().parent
DEFAULT_PHASE = (
    ROOT / "workdirs" / "train_radaz_mode_separated_controlled_carrier_rom"
)
DEFAULT_AMPLITUDE = (
    ROOT / "workdirs" / "train_radaz_state_history_conditioned_rom_noaug"
)
DEFAULT_OUTPUT = (
    ROOT / "workdirs" / "build_radaz_state_phase_factorized_rom"
)


def json_safe(value):
    return carrier.json_safe(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for piece in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([json_safe(row) for row in rows])


def require_equal(name: str, left: np.ndarray, right: np.ndarray) -> None:
    if left.shape != right.shape or not np.array_equal(left, right):
        raise ValueError(f"Factorized source mismatch: {name}")


def phase_frequency_mhz(coefficients: np.ndarray, dt_us: float) -> float:
    products = coefficients[1:] * np.conj(coefficients[:-1])
    return float(np.angle(np.sum(products)) / (2.0 * np.pi * dt_us))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=Path, default=DEFAULT_PHASE)
    parser.add_argument("--amplitude", type=Path, default=DEFAULT_AMPLITUDE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phase_dir = args.phase.resolve()
    amplitude_dir = args.amplitude.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    phase_h5 = phase_dir / "development_rollout_35to40us.h5"
    amplitude_h5 = amplitude_dir / "development_rollout_35to40us.h5"
    phase_lock_path = phase_dir / "model_lock.json"
    amplitude_lock_path = amplitude_dir / "model_lock.json"
    phase_lock = json.loads(phase_lock_path.read_text(encoding="utf-8"))
    amplitude_lock = json.loads(
        amplitude_lock_path.read_text(encoding="utf-8")
    )

    with h5py.File(phase_h5, "r") as phase_handle, h5py.File(
        amplitude_h5, "r"
    ) as amplitude_handle:
        time_us = np.asarray(phase_handle["time_us"], dtype=np.float64)
        frame = np.asarray(phase_handle["frame"], dtype=np.int64)
        modes = np.asarray(
            phase_handle["selected_mode_numbers"], dtype=np.int64
        )
        truth = np.asarray(phase_handle["truth_physical_coefficients"])
        persistence = np.asarray(
            phase_handle["raw_persistence_physical_coefficients"]
        )
        phase_prediction = np.asarray(
            phase_handle["prediction_physical_coefficients"]
        )
        amplitude_prediction = np.asarray(
            amplitude_handle["prediction_physical_coefficients"]
        )
        require_equal(
            "time_us", time_us, np.asarray(amplitude_handle["time_us"])
        )
        require_equal(
            "frame", frame, np.asarray(amplitude_handle["frame"])
        )
        require_equal(
            "selected_mode_numbers",
            modes,
            np.asarray(amplitude_handle["selected_mode_numbers"]),
        )
        require_equal(
            "truth_physical_coefficients",
            truth,
            np.asarray(amplitude_handle["truth_physical_coefficients"]),
        )
        require_equal(
            "raw_persistence_physical_coefficients",
            persistence,
            np.asarray(
                amplitude_handle["raw_persistence_physical_coefficients"]
            ),
        )

    selected_indices = np.asarray(
        [int(np.flatnonzero(modes == mode)[0]) for mode in (2, 7)],
        dtype=np.int64,
    )
    denominator = np.maximum(
        np.abs(phase_prediction[:, 0, selected_indices]),
        np.finfo(float).tiny,
    )
    amplitude_ratio = (
        np.abs(amplitude_prediction[:, 0, selected_indices]) / denominator
    )
    prediction = phase_prediction.copy()
    prediction[:, :, selected_indices] *= amplitude_ratio[:, None, :]

    phase_metrics = phase_lock["development_metrics"]
    row = {
        "finite_fraction": float(np.mean(np.isfinite(prediction))),
        "carrier_state_skill_vs_envelope_persistence": phase_metrics[
            "carrier_state_skill_vs_envelope_persistence"
        ],
        "composite_state_skill_vs_persistence": phase_metrics[
            "composite_state_skill_vs_persistence"
        ],
        "radial_skill_vs_persistence": phase_metrics[
            "radial_skill_vs_persistence"
        ],
        "transport_skill_vs_persistence": phase_metrics[
            "transport_skill_vs_persistence"
        ],
        "MTSI_n1_6_transport_skill": phase_metrics[
            "MTSI_n1_6_transport_skill"
        ],
        "ECDI_n9_21_transport_skill": phase_metrics[
            "ECDI_n9_21_transport_skill"
        ],
    }
    for field_index, field in ((0, "phi"), (3, "efy")):
        metrics = augmented.scalar_metrics(
            truth[:, field_index],
            prediction[:, field_index],
            persistence[:, field_index],
        )
        row[f"selected_{field}_skill_vs_raw_persistence"] = metrics[
            "skill_vs_persistence"
        ]
        row[f"selected_{field}_nrmse"] = metrics["nrmse"]
        row[f"selected_{field}_temporal_anomaly_correlation"] = metrics[
            "temporal_anomaly_correlation"
        ]
    dt_us = float(np.median(np.diff(time_us)))
    for mode, local_index in zip((2, 7), selected_indices):
        metrics = augmented.scalar_metrics(
            np.abs(truth[:, 0, local_index]),
            np.abs(prediction[:, 0, local_index]),
            np.abs(persistence[:, 0, local_index]),
        )
        row[f"phi_n{mode}_amplitude_skill"] = metrics[
            "skill_vs_persistence"
        ]
        row[f"phi_n{mode}_amplitude_correlation"] = metrics["correlation"]
        truth_frequency = phase_frequency_mhz(
            truth[:, 0, local_index], dt_us
        )
        prediction_frequency = phase_frequency_mhz(
            prediction[:, 0, local_index], dt_us
        )
        row[f"phi_n{mode}_truth_frequency_MHz"] = truth_frequency
        row[f"phi_n{mode}_prediction_frequency_MHz"] = prediction_frequency
        row[f"phi_n{mode}_frequency_abs_error_MHz"] = abs(
            prediction_frequency - truth_frequency
        )
        row[f"phi_n{mode}_amplitude_ratio_min"] = float(
            np.min(amplitude_ratio[:, np.flatnonzero(selected_indices == local_index)[0]])
        )
        row[f"phi_n{mode}_amplitude_ratio_median"] = float(
            np.median(amplitude_ratio[:, np.flatnonzero(selected_indices == local_index)[0]])
        )
        row[f"phi_n{mode}_amplitude_ratio_max"] = float(
            np.max(amplitude_ratio[:, np.flatnonzero(selected_indices == local_index)[0]])
        )

    wave_numbers = (
        2.0 * np.pi * modes / direct.AZIMUTHAL_LENGTH_M
    )
    residual = prediction[:, 3] + 1j * wave_numbers[None] * prediction[:, 0]
    row["field_gradient_residual_over_ey_rms"] = float(
        np.sqrt(np.mean(np.abs(residual) ** 2))
        / np.sqrt(np.mean(np.abs(prediction[:, 3]) ** 2))
    )
    gate_values = (
        row["carrier_state_skill_vs_envelope_persistence"],
        row["composite_state_skill_vs_persistence"],
        row["radial_skill_vs_persistence"],
        row["MTSI_n1_6_transport_skill"],
        row["ECDI_n9_21_transport_skill"],
        row["selected_phi_skill_vs_raw_persistence"],
        row["selected_efy_skill_vs_raw_persistence"],
        row["phi_n2_amplitude_skill"],
        row["phi_n7_amplitude_skill"],
    )
    row["minimum_persistence_gate_skill"] = float(min(gate_values))
    row["passes_all_persistence_gates"] = bool(min(gate_values) > 0.0)
    row["passes_field_climatology_gate"] = bool(
        row["selected_phi_nrmse"] < 1.0
        and row["selected_efy_nrmse"] < 1.0
    )
    accepted = bool(
        row["finite_fraction"] == 1.0
        and row["passes_all_persistence_gates"]
        and row["passes_field_climatology_gate"]
    )
    status = (
        "READY_FOR_PHYSICS_ABLATION_POST_HOC"
        if accepted
        else "REJECTED_DEVELOPMENT"
    )

    write_csv(output / "development_metrics.csv", [row])
    (output / "development_metrics.json").write_text(
        json.dumps(json_safe(row), indent=2), encoding="utf-8"
    )
    with h5py.File(output / "development_rollout_35to40us.h5", "w") as handle:
        handle.attrs["primary_E25_to_E22p5_loaded"] = False
        handle.attrs["future_PIC_truth_used_in_fusion"] = False
        handle.attrs["post_hoc_factorization_on_development"] = True
        handle.create_dataset("time_us", data=time_us)
        handle.create_dataset("frame", data=frame)
        handle.create_dataset("selected_mode_numbers", data=modes)
        handle.create_dataset("truth_physical_coefficients", data=truth, compression="gzip")
        handle.create_dataset("prediction_physical_coefficients", data=prediction, compression="gzip")
        handle.create_dataset("raw_persistence_physical_coefficients", data=persistence, compression="gzip")
        handle.create_dataset("phase_branch_prediction", data=phase_prediction, compression="gzip")
        handle.create_dataset("amplitude_branch_prediction", data=amplitude_prediction, compression="gzip")
        handle.create_dataset("n2_n7_amplitude_ratio", data=amplitude_ratio)

    lock = {
        "status": status,
        "accepted_for_physics_ablation": accepted,
        "development_metrics": row,
        "factorization": {
            "phase_and_other_modes": "mode-separated recurrent branch",
            "n2_n7_phi_amplitudes": "state/E-history-conditioned branch",
            "n2_n7_all_field_rescaling": True,
            "amplitude_ratio_clipped": False,
            "future_PIC_truth_used": False,
            "post_hoc_selected_on_up_transition_development": True,
        },
        "phase_model_lock": str(phase_lock_path),
        "phase_model_lock_sha256": sha256(phase_lock_path),
        "amplitude_model_lock": str(amplitude_lock_path),
        "amplitude_model_lock_sha256": sha256(amplitude_lock_path),
        "phase_rollout": str(phase_h5),
        "phase_rollout_sha256": sha256(phase_h5),
        "amplitude_rollout": str(amplitude_h5),
        "amplitude_rollout_sha256": sha256(amplitude_h5),
        "primary_test": {
            "direction": "E25_to_E22.5",
            "data_loaded": False,
            "used_for_selection": False,
        },
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "arguments": vars(args),
    }
    (output / "model_lock.json").write_text(
        json.dumps(json_safe(lock), indent=2), encoding="utf-8"
    )
    (output / "README.md").write_text(
        f"""# State/phase-factorized RadAz ROM

- Status: `{status}`
- phi/Ey NRMSE: {row['selected_phi_nrmse']:.6f}/{row['selected_efy_nrmse']:.6f}
- n=2/n=7 amplitude skill: {row['phi_n2_amplitude_skill']:.6f}/{row['phi_n7_amplitude_skill']:.6f}
- Primary E25 -> E22.5 data loaded: **no**
- Factorization selected on up-transition development: **yes**
""",
        encoding="utf-8",
    )
    print(json.dumps(json_safe(lock), indent=2), flush=True)


if __name__ == "__main__":
    main()
