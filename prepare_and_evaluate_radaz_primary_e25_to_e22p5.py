"""Audit, preprocess, and evaluate the locked E25 -> E22.5 primary case.

The primary path is never used to refit normalization, the latent encoder,
the reduced representation, a recurrent checkpoint, or the physics-decoder
weight.  Existing large intermediate files are validated and reused so that
an interrupted run can be resumed without silently overwriting them.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np

import evaluate_radaz_primary_e25_to_e22p5 as primary


ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT.parent
PEPAPIC = RESEARCH / "PEPAPIC"
DEFAULT_OUTPUT = ROOT / "workdirs" / "radaz_e25_to_e22p5_primary"
EXECUTION_LOCK = ROOT / "workdirs" / "radaz_primary_execution_bundle_lock.json"
EXPECTED_FIELDS = {
    "efx",
    "efy",
    "electron_C",
    "electron_Temp",
    "electron_den",
    "electron_ud",
    "electron_vd",
    "electron_wd",
    "ion_C",
    "ion_Temp",
    "ion_den",
    "ion_ud",
    "ion_vd",
    "ion_wd",
    "phi",
}


def resolve_case(path: Path) -> Path:
    path = path.resolve()
    if (path / "Macro").is_dir():
        return path
    nested = path / path.name
    if (nested / "Macro").is_dir():
        return nested
    raise FileNotFoundError(f"Macro directory not found below {path}")


def run(command: list[str], cwd: Path, environment: dict[str, str]) -> None:
    print("[RUN]", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def verify_execution_bundle(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "LOCKED_BEFORE_PRIMARY_INPUTS":
        raise ValueError("Unexpected primary execution-bundle lock status")
    mismatches = []
    for name, item in manifest.get("artifacts", {}).items():
        artifact = Path(item["path"])
        if not artifact.is_file() or primary.sha256(artifact) != item["sha256"]:
            mismatches.append(name)
    script = Path(manifest["script"])
    if not script.is_file() or primary.sha256(script) != manifest["script_sha256"]:
        mismatches.append("execution_bundle_lock_script")
    if mismatches:
        raise ValueError(f"Primary execution bundle drift detected: {mismatches}")
    return manifest


def check_raw_report(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "passed": True,
        "expected_frames": 333,
        "expected_frame_start": 2001,
        "expected_ranks_per_frame": 64,
        "frame_count": 333,
        "file_count": 333 * 64,
        "zero_byte_files": 0,
        "nonfinite_total": 0,
    }
    mismatches = {
        key: (report.get(key), expected)
        for key, expected in required.items()
        if report.get(key) != expected
    }
    if report.get("missing_frames") or report.get("rank_problems"):
        mismatches["frame_or_rank_problems"] = (
            report.get("missing_frames"),
            report.get("rank_problems"),
        )
    if set(report.get("dominant_schema", [])) != EXPECTED_FIELDS:
        mismatches["dominant_schema"] = (
            report.get("dominant_schema"),
            sorted(EXPECTED_FIELDS),
        )
    if mismatches:
        raise ValueError(f"Raw primary audit failed: {mismatches}")
    return report


def check_consolidated(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        if not bool(handle.attrs.get("completed", False)):
            raise ValueError("Consolidated primary H5 is not marked completed")
        if int(handle.attrs.get("completed_frames", -1)) != 333:
            raise ValueError("Consolidated primary H5 does not contain 333 frames")
        time_us = np.asarray(handle["axes/time_s"], dtype=np.float64) * 1.0e6
        fields = set(handle["fields"].keys())
        efz = np.asarray(handle["static_fields/efz"], dtype=np.float64)
        bfx = np.asarray(handle["static_fields/bfx"], dtype=np.float64)
        shapes = {name: tuple(handle[f"fields/{name}"].shape) for name in fields}
        nonfinite = sum(
            int(np.size(handle[f"fields/{name}"]) - np.count_nonzero(np.isfinite(handle[f"fields/{name}"])))
            for name in fields
        )
    if fields != EXPECTED_FIELDS:
        raise ValueError(f"Consolidated field mismatch: {sorted(fields)}")
    if any(shape != (333, 257, 257) for shape in shapes.values()):
        raise ValueError(f"Unexpected consolidated field shapes: {shapes}")
    if nonfinite:
        raise ValueError(f"Consolidated H5 has {nonfinite} non-finite values")
    if not np.allclose(time_us, 30.015 + 0.015 * np.arange(333), atol=1.0e-9):
        raise ValueError(
            f"Unexpected primary time axis: {time_us[0]}--{time_us[-1]} us"
        )
    if not np.allclose(efz, 22_500.0):
        raise ValueError("Primary static Ez is not 22.5 kV/m")
    if not np.allclose(bfx, 0.020):
        raise ValueError("Primary static Bx is not 20 mT")
    return {
        "frames": 333,
        "time_us": [float(time_us[0]), float(time_us[-1])],
        "fields": sorted(fields),
        "shape": [333, 257, 257],
        "nonfinite": nonfinite,
        "Ez_kVm": float(np.mean(efz) / 1000.0),
        "Bx_mT": float(np.mean(bfx) * 1000.0),
    }


def check_normalized(path: Path, normalization: Path) -> dict:
    metadata_path = path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if Path(metadata["normalization_source"]).resolve() != normalization.resolve():
        raise ValueError("Primary preprocessing did not use the frozen normalization")
    if metadata.get("frames") != 333 or metadata.get("shape_tchw") != [333, 3, 260, 256]:
        raise ValueError("Unexpected normalized primary tensor shape")
    clipped = np.asarray(metadata.get("clipped_low_fraction", []), dtype=float)
    clipped_high = np.asarray(metadata.get("clipped_high_fraction", []), dtype=float)
    if clipped.shape != (3,) or clipped_high.shape != (3,):
        raise ValueError("Missing normalization clipping diagnostics")
    return {
        "h5": str(path),
        "normalization_source": str(normalization),
        "clipped_low_fraction": clipped.tolist(),
        "clipped_high_fraction": clipped_high.tolist(),
    }


def check_feature_inputs(features: Path, physical: Path) -> dict:
    with h5py.File(features, "r") as handle:
        windows = int(len(handle["window_start"]))
        feature_shape = list(handle["encoder_fourier_ri"].shape)
        maximum_mode = int(np.max(handle["azimuthal_modes"]))
        bands = int(len(handle["radial_boundaries"]) - 1)
    with h5py.File(physical, "r") as handle:
        frames = int(len(handle["time_us"]))
        coefficient_shape = list(handle["coefficients"].shape)
        physical_time = np.asarray(handle["time_us"], dtype=np.float64)
    if windows != 314 or frames != 314:
        raise ValueError(f"Expected 314 aligned forecast windows, got {windows}/{frames}")
    if maximum_mode != 21 or bands != 8:
        raise ValueError("Fourier extraction differs from the locked 8-band n=0--21 layout")
    if feature_shape != [314, 64, 8, 22, 2]:
        raise ValueError(f"Unexpected latent feature shape: {feature_shape}")
    if coefficient_shape != [314, 4, 8, 22]:
        raise ValueError(f"Unexpected physical coefficient shape: {coefficient_shape}")
    if not np.isclose(physical_time[0], 30.165) or not np.isclose(physical_time[-1], 34.86):
        raise ValueError("Unexpected aligned primary target times")
    return {
        "windows": windows,
        "latent_feature_shape": feature_shape,
        "physical_coefficient_shape": coefficient_shape,
        "time_us": [float(physical_time[0]), float(physical_time[-1])],
        "radial_bands": bands,
        "maximum_mode": maximum_mode,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path, help="Primary case folder or its outer wrapper")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Stop after frozen preprocessing; do not open the confirmatory metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_execution_bundle(EXECUTION_LOCK.resolve())
    lock_path = primary.DEFAULT_LOCK.resolve()
    lock = primary.verify_lock(lock_path)
    case = resolve_case(args.case)
    case_lower = case.name.lower()
    if "restartfromez25" not in case_lower or "ez22p5" not in case_lower:
        raise ValueError(
            "The primary directory name must identify Ez25 -> Ez22.5; "
            f"got {case.name}"
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    python = Path(sys.executable).resolve()

    raw_report_path = output / "raw_h5_health_check.json"
    run(
        [
            str(python),
            "check_radial_azimuthal_h5_health.py",
            str(case),
            "--expected-frames", "333",
            "--expected-frame-start", "2001",
            "--expected-ranks", "64",
            "--output", str(raw_report_path),
            "--progress-every", "25",
        ],
        PEPAPIC,
        environment,
    )
    raw_report = check_raw_report(raw_report_path)

    consolidated = case / "analysis_fields_uncompressed.h5"
    if not consolidated.is_file():
        run(
            [
                str(python),
                "consolidate_radial_azimuthal_case.py",
                str(case),
                "--output", str(consolidated),
                "--timesteps", "2001-2333",
                "--progress-every", "25",
            ],
            PEPAPIC,
            environment,
        )
    consolidated_summary = check_consolidated(consolidated)
    validation_report = output / "consolidated_validation.json"
    run(
        [
            str(python),
            "validate_consolidated_radial_azimuthal.py",
            str(case),
            str(consolidated),
            "--raw-timesteps", "2001,2167,2333",
            "--report", str(validation_report),
        ],
        PEPAPIC,
        environment,
    )
    validation = json.loads(validation_report.read_text(encoding="utf-8"))
    if validation.get("status") != "PASS":
        raise ValueError(f"Consolidated/raw validation failed: {validation}")

    normalization = Path(lock["frozen_artifacts"]["normalization"]["path"])
    normalized = output / "radaz_3ch_e25targetnorm_native257x256_pad260x256.h5"
    if not normalized.is_file():
        run(
            [
                str(python),
                "build_radaz_single_case_h5.py",
                str(consolidated),
                str(normalized),
                "--spatial-stride", "1",
                "--expected-frames", "333",
                "--normalization-h5", str(normalization),
            ],
            ROOT,
            environment,
        )
    normalized_summary = check_normalized(normalized, normalization)

    features = output / "fourier_latent_features.h5"
    if not features.is_file():
        run(
            [
                str(python),
                "analyze_radaz_fourier_latent_dynamics.py",
                "--data", str(normalized),
                "--output", str(output),
                "--device", args.device,
                "--radial-bands", "8",
                "--maximum-mode", "21",
                "--extract-only",
            ],
            ROOT,
            environment,
        )
    physical = output / "physical_fourier_targets.h5"
    if not physical.is_file():
        run(
            [
                str(python),
                "analyze_radaz_fourier_latent_to_physical_modes.py",
                "--physical", str(consolidated),
                "--feature-path", str(features),
                "--output", str(physical),
                "--radial-bands", "8",
                "--maximum-mode", "21",
                "--extract-only",
            ],
            ROOT,
            environment,
        )
    extraction_summary = check_feature_inputs(features, physical)
    intake = {
        "status": "PASS_PRIMARY_INTAKE_AND_FROZEN_PREPROCESSING",
        "primary_data_read": True,
        "evaluation_lock": str(lock_path),
        "evaluation_lock_sha256": primary.sha256(lock_path),
        "case": str(case),
        "raw_audit": {
            "frames": raw_report["frame_count"],
            "files": raw_report["file_count"],
            "zero_byte_files": raw_report["zero_byte_files"],
            "nonfinite_total": raw_report["nonfinite_total"],
        },
        "consolidated": consolidated_summary,
        "normalization": normalized_summary,
        "extraction": extraction_summary,
        "normalization_or_representation_refit": False,
    }
    intake_path = output / "primary_intake_summary.json"
    intake_path.write_text(json.dumps(intake, indent=2), encoding="utf-8")
    if args.preprocess_only:
        print(json.dumps(intake, indent=2))
        return

    evaluation_output = ROOT / "workdirs" / "evaluate_radaz_primary_e25_to_e22p5"
    existing_evaluation = evaluation_output / "primary_evaluation.json"
    if existing_evaluation.is_file():
        raise FileExistsError(
            "The one-time primary evaluation already exists and will not be overwritten: "
            f"{existing_evaluation}"
        )
    run(
        [
            str(python),
            "evaluate_radaz_primary_e25_to_e22p5.py",
            "--features", str(features),
            "--physical", str(physical),
            "--lock", str(lock_path),
            "--output", str(evaluation_output),
            "--device", args.device,
        ],
        ROOT,
        environment,
    )
    print(f"[DONE] {existing_evaluation}")


if __name__ == "__main__":
    main()
