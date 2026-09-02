"""Audit the locked bidirectional RadAz electric-sweep ROM artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT.parent
REVERSE_CASE = (
    RESEARCH
    / "PEPAPIC"
    / "test"
    / "results"
    / "2D_Landmark"
    / "2D_RadAz_Xe1p_Bx20mT_Ez20kVm_restartFromEz22p5_30to35us_dt15ps_out15ns"
)
TRANSITION = ROOT / "workdirs" / "radaz_e22p5_to_e20_transition"
LOCKS = (
    ROOT
    / "workdirs"
    / "train_radaz_state_history_conditioned_rom_bidirectional_noaug"
    / "model_lock.json",
    ROOT
    / "workdirs"
    / "build_radaz_state_phase_factorized_rom_bidirectional"
    / "model_lock.json",
    ROOT
    / "workdirs"
    / "evaluate_radaz_factorized_physics_decoder_bidirectional"
    / "model_lock.json",
    ROOT
    / "workdirs"
    / "train_radaz_factorized_end_to_end_physics_rom_bidirectional"
    / "model_lock.json",
    ROOT / "workdirs" / "train_radaz_coupled_amplitude_ode" / "model_lock.json",
    ROOT / "workdirs" / "train_radaz_second_order_amplitude_ode" / "model_lock.json",
    ROOT / "workdirs" / "train_radaz_delay_amplitude_rom" / "model_lock.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def one(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise ValueError(f"Expected one {name} under {root}, found {len(matches)}")
    return matches[0]


def primary_boole(value, prefix: str = "") -> list[tuple[str, bool]]:
    result = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            lowered = str(key).lower()
            if isinstance(item, bool) and (
                "primary" in lowered or "e25_to_e22p5" in lowered
            ):
                result.append((path, item))
            result.extend(primary_boole(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.extend(primary_boole(item, f"{prefix}[{index}]"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "workdirs" / "radaz_bidirectional_release_audit.json",
    )
    args = parser.parse_args()
    checks = {}

    health_path = one(REVERSE_CASE, "raw_h5_health_check.json")
    stitch_path = one(REVERSE_CASE, "analysis_fields_uncompressed_stitch_report.json")
    health = json.loads(health_path.read_text(encoding="utf-8"))
    stitch = json.loads(stitch_path.read_text(encoding="utf-8"))
    checks["reverse_raw_health_pass"] = bool(
        health.get("passed", False) or health.get("status") == "PASS"
    )
    checks["reverse_stitch_pass"] = stitch.get("status") == "PASS"
    checks["reverse_expected_frames"] = int(health["expected_frames"]) == 333
    checks["reverse_expected_files"] = int(health["expected_file_count"]) == 21312
    checks["reverse_zero_read_errors"] = len(health["read_errors"]) == 0
    checks["reverse_zero_nonfinite"] = int(health["nonfinite_total"]) == 0

    normalization = json.loads(
        (TRANSITION / "radaz_3ch_e25targetnorm_native257x256_pad260x256.json").read_text(
            encoding="utf-8"
        )
    )
    checks["reverse_normalization_zero_clipping"] = bool(
        np.allclose(normalization["clipped_low_fraction"], 0.0)
        and np.allclose(normalization["clipped_high_fraction"], 0.0)
    )
    feature_path = TRANSITION / "fourier_latent_features.h5"
    physical_path = TRANSITION / "physical_fourier_targets.h5"
    with h5py.File(feature_path, "r") as feature, h5py.File(physical_path, "r") as physical:
        feature_frame = np.asarray(feature["translator_frame"], dtype=np.int64)
        feature_time = np.asarray(feature["translator_time_s"], dtype=np.float64) * 1.0e6
        physical_frame = np.asarray(physical["frame"], dtype=np.int64)
        physical_time = np.asarray(physical["time_us"], dtype=np.float64)
        checks["reverse_feature_windows_314"] = len(feature_frame) == 314
        checks["reverse_feature_physical_frame_match"] = bool(
            np.array_equal(feature_frame, physical_frame)
        )
        checks["reverse_feature_physical_time_match"] = bool(
            np.allclose(feature_time, physical_time, atol=1.0e-9, rtol=0.0)
        )
        checks["reverse_fourier_shape_match"] = bool(
            feature["translator_fourier_ri"].shape == (314, 64, 8, 22, 2)
            and physical["coefficients"].shape == (314, 4, 8, 22)
        )

    lock_audit = {}
    for lock_path in LOCKS:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        booleans = primary_boole(lock)
        script_ok = True
        if "script" in lock and "script_sha256" in lock:
            script_path = Path(lock["script"])
            script_ok = script_path.is_file() and sha256(script_path) == lock["script_sha256"]
        lock_audit[lock_path.parent.name] = {
            "primary_boolean_flags": booleans,
            "all_primary_boolean_flags_false": all(not value for _, value in booleans),
            "script_hash_matches": script_ok,
        }
    checks["all_primary_boolean_flags_false"] = all(
        item["all_primary_boolean_flags_false"] for item in lock_audit.values()
    )
    checks["all_lock_script_hashes_match"] = all(
        item["script_hash_matches"] for item in lock_audit.values()
    )

    factor_lock = json.loads(LOCKS[1].read_text(encoding="utf-8"))
    decoder_lock = json.loads(LOCKS[2].read_text(encoding="utf-8"))
    end_to_end_lock = json.loads(LOCKS[3].read_text(encoding="utf-8"))
    checks["factorized_data_only_accepted"] = bool(
        factor_lock["accepted_for_physics_ablation"]
        and factor_lock["development_metrics"]["passes_all_persistence_gates"]
        and factor_lock["development_metrics"]["passes_field_climatology_gate"]
    )
    checks["physics_decoder_accepted"] = decoder_lock["status"] == "PHYSICS_DECODER_ACCEPTED"
    checks["end_to_end_physics_accepted"] = (
        end_to_end_lock["status"] == "END_TO_END_PHYSICS_ACCEPTED"
    )
    availability = json.loads(
        (
            ROOT
            / "workdirs"
            / "compare_radaz_factorized_physics_horizons_bidirectional"
            / "availability.json"
        ).read_text(encoding="utf-8")
    )
    checks["horizon_primary_unread"] = not availability["primary_E25_to_E22p5_loaded"]
    checks["horizon_available_4p845us"] = bool(
        np.isclose(availability["available_horizon_us"], 4.845)
        and not availability["requested_6us_available"]
    )

    protocol = (ROOT / "RADAZ_ELECTRIC_SWEEP_ROM_PROTOCOL.md").read_text(encoding="utf-8")
    memo = (RESEARCH / "ICL_reserch_memo.md").read_text(encoding="utf-8")
    checks["protocol_updated"] = "Stage 2f reverse-transition development" in protocol
    checks["memo_updated"] = "## 21. 逆遷移を追加した電場掃引ROMの更新" in memo

    failed = [name for name, passed in checks.items() if not passed]
    result = {
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed_checks": failed,
        "lock_audit": lock_audit,
        "primary_e25_to_e22p5_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
