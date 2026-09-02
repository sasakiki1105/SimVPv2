#!/usr/bin/env python3
"""Prepare and train the low/high RadAz regime source-pool models."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = Path(r"C:\Users\astro\anaconda3\envs\OpenSTL\python.exe")
OUTPUT_ROOT = ROOT / "workdirs" / "2D_RadAz"
MANIFEST_DIR = OUTPUT_ROOT / "radaz_regime_generalization_manifests"
LOG_DIR = OUTPUT_ROOT / "radaz_regime_generalization_queue_logs"
QUEUE_LOG = LOG_DIR / "queue.log"
PID_PATH = LOG_DIR / "queue.pid"
ERROR_PATH = LOG_DIR / "runner_error.log"
CONFIG = "configs/custom/pepapic/SimVP_gSTA_radaz_direct.py"
CASES = (
    {
        "key": "low_E10_E20",
        "manifest": MANIFEST_DIR / "radaz_low_source_manifest.json",
        "ex_name": (
            "radaz_bx20mt_lowE_E10_E20_mixed_direct10_"
            "sourcepool_noclip_disjoint811_bs1_100ep"
        ),
    },
    {
        "key": "high_E30_E40",
        "manifest": MANIFEST_DIR / "radaz_high_source_manifest.json",
        "ex_name": (
            "radaz_bx20mt_highE_E30_E40_mixed_direct10_"
            "sourcepool_noclip_disjoint811_bs1_100ep"
        ),
    },
)


def log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    with QUEUE_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def windows_free_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024.0**3)


def run_stage(name: str, command: list[str]) -> None:
    stage_log = LOG_DIR / f"{name}.log"
    log(f"{name} command: {' '.join(command)}")
    started = time.time()
    with stage_log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] START\n")
        handle.flush()
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            env=os.environ.copy(),
        )
    elapsed = time.time() - started
    log(
        f"{name} exit_code={result.returncode} elapsed_sec={elapsed:.1f} "
        f"stage_log={stage_log}"
    )
    if result.returncode != 0:
        raise RuntimeError(f"Stage {name} failed with exit code {result.returncode}")


def ensure_manifests() -> None:
    expected = [case["manifest"] for case in CASES]
    if all(path.exists() for path in expected):
        log("manifest preparation skipped: both manifests already exist")
        return
    run_stage(
        "prepare_manifests",
        [
            str(PYTHON),
            "prepare_radaz_regime_generalization.py",
            "--output",
            str(MANIFEST_DIR),
        ],
    )
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing prepared manifests: " + ", ".join(missing))


def completion_path(case) -> Path:
    return OUTPUT_ROOT / case["ex_name"] / "training_complete.json"


def train_command(case) -> list[str]:
    workdir = OUTPUT_ROOT / case["ex_name"]
    command = [
        str(PYTHON),
        "tools\\train_only.py",
        "--dataname",
        "pepapic_h5",
        "--config_file",
        CONFIG,
        "--data_root",
        str(case["manifest"]),
        "--res_dir",
        str(OUTPUT_ROOT),
        "--ex_name",
        case["ex_name"],
        "--method",
        "simvp",
        "--pre_seq_length",
        "10",
        "--aft_seq_length",
        "10",
        "--total_length",
        "20",
        "--epoch",
        "100",
        "--batch_size",
        "1",
        "--val_batch_size",
        "1",
        "--num_workers",
        "0",
        "--gpus",
        "0",
        "--no_display_method_info",
    ]
    last_checkpoint = workdir / "checkpoints" / "last.ckpt"
    if last_checkpoint.exists():
        command.extend(["--ckpt_path", str(last_checkpoint)])
        log(f"{case['key']} will resume from {last_checkpoint}")
    return command


def mark_complete(case, elapsed: float) -> None:
    workdir = OUTPUT_ROOT / case["ex_name"]
    best = workdir / "checkpoints" / "best.ckpt"
    last = workdir / "checkpoints" / "last.ckpt"
    if not best.exists() or not last.exists():
        raise FileNotFoundError(
            f"Training returned successfully but checkpoints are missing in {workdir}"
        )
    payload = {
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "case": case["key"],
        "experiment": case["ex_name"],
        "manifest": str(case["manifest"]),
        "epochs": 100,
        "elapsed_seconds_this_invocation": elapsed,
        "best_checkpoint": str(best),
        "last_checkpoint": str(last),
        "test_arrays_materialized": False,
        "reason": "Deferred streaming evaluation avoids exhausting local disk space.",
    }
    completion_path(case).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PID_PATH.write_text(str(os.getpid()), encoding="ascii")
    ERROR_PATH.unlink(missing_ok=True)
    log("RadAz regime-generalization queue started")
    free_gib = windows_free_gib(ROOT)
    log(f"disk_free_gib={free_gib:.2f}")
    if free_gib < 5.0:
        raise RuntimeError(
            f"At least 5 GiB free space is required for checkpoints; got {free_gib:.2f}"
        )

    if not PYTHON.exists():
        raise FileNotFoundError(PYTHON)
    ensure_manifests()

    smoke_marker = LOG_DIR / "gpu_smoke_test_passed.txt"
    if not smoke_marker.exists():
        run_stage(
            "gpu_smoke_test",
            [
                str(PYTHON),
                "smoke_test_radaz_manifest.py",
                str(CASES[0]["manifest"]),
                "--device",
                "cuda",
            ],
        )
        smoke_marker.write_text(
            datetime.now().isoformat(timespec="seconds") + "\n", encoding="ascii"
        )
    else:
        log(f"gpu smoke test skipped: {smoke_marker} exists")

    for case in CASES:
        complete = completion_path(case)
        if complete.exists():
            log(f"{case['key']} skipped: {complete} exists")
            continue
        command = train_command(case)
        started = time.time()
        run_stage(f"train_{case['key']}", command)
        elapsed = time.time() - started
        mark_complete(case, elapsed)
        log(f"{case['key']} training complete")
    log("RadAz regime-generalization queue finished")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        details = traceback.format_exc()
        ERROR_PATH.write_text(details, encoding="utf-8")
        log(f"FATAL: {details.splitlines()[-1]}")
        raise
