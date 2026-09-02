#!/usr/bin/env python3
"""Run the matched U-D/C-D/U-P/C-P RadAz experiment sequentially."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = Path(r"C:\Users\astro\anaconda3\envs\OpenSTL\python.exe")
OUTPUT_ROOT = ROOT / "workdirs" / "2D_RadAz"
MANIFEST = (
    OUTPUT_ROOT
    / "radaz_conditioned_factorial_manifests"
    / "radaz_axis_factorial_manifest.json"
)
LOG_DIR = OUTPUT_ROOT / "radaz_conditioned_factorial_queue_logs"
QUEUE_LOG = LOG_DIR / "queue.log"
PID_PATH = LOG_DIR / "queue.pid"
ERROR_PATH = LOG_DIR / "runner_error.log"
EPOCHS = 60
CASES = (
    {
        "key": "U-D",
        "config": "configs/custom/pepapic/SimVP_gSTA_radaz_factorial_UD_60ep.py",
        "ex_name": "radaz_axis_factorial_UD_direct10_bs1_60ep",
    },
    {
        "key": "C-D",
        "config": "configs/custom/pepapic/SimVP_gSTA_radaz_factorial_CD_60ep.py",
        "ex_name": "radaz_axis_factorial_CD_film_direct10_bs1_60ep",
    },
    {
        "key": "U-P",
        "config": "configs/custom/pepapic/SimVP_gSTA_radaz_factorial_UP_60ep.py",
        "ex_name": "radaz_axis_factorial_UP_qtransport_direct10_bs1_60ep",
    },
    {
        "key": "C-P",
        "config": "configs/custom/pepapic/SimVP_gSTA_radaz_factorial_CP_60ep.py",
        "ex_name": "radaz_axis_factorial_CP_film_qtransport_direct10_bs1_60ep",
    },
)


def log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    with QUEUE_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


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


def workdir(case: dict) -> Path:
    return OUTPUT_ROOT / case["ex_name"]


def complete_path(case: dict) -> Path:
    return workdir(case) / "training_complete.json"


def train_command(case: dict) -> list[str]:
    command = [
        str(PYTHON),
        "tools\\train_only.py",
        "--dataname", "pepapic_h5",
        "--config_file", case["config"],
        "--data_root", str(MANIFEST),
        "--res_dir", str(OUTPUT_ROOT),
        "--ex_name", case["ex_name"],
        "--method", "simvp",
        "--pre_seq_length", "10",
        "--aft_seq_length", "10",
        "--total_length", "20",
        "--epoch", str(EPOCHS),
        "--batch_size", "1",
        "--val_batch_size", "1",
        "--num_workers", "0",
        "--gpus", "0",
        "--seed", "42",
        "--no_display_method_info",
    ]
    last_checkpoint = workdir(case) / "checkpoints" / "last.ckpt"
    if last_checkpoint.exists():
        command.extend(["--ckpt_path", str(last_checkpoint)])
        log(f"{case['key']} will resume from {last_checkpoint}")
    return command


def mark_complete(case: dict, elapsed: float) -> None:
    best = workdir(case) / "checkpoints" / "best.ckpt"
    last = workdir(case) / "checkpoints" / "last.ckpt"
    if not best.is_file() or not last.is_file():
        raise FileNotFoundError(
            f"Training returned without best/last checkpoint: {workdir(case)}"
        )
    payload = {
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "factorial_cell": case["key"],
        "experiment": case["ex_name"],
        "manifest": str(MANIFEST),
        "config": case["config"],
        "epochs": EPOCHS,
        "seed": 42,
        "elapsed_seconds_this_invocation": elapsed,
        "best_checkpoint": str(best),
        "last_checkpoint": str(last),
        "checkpoint_selection": "minimum source-condition validation total loss",
        "condition_holdouts_used_for_selection": False,
        "off_axis_data_used": False,
        "test_arrays_materialized": False,
    }
    complete_path(case).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PID_PATH.write_text(str(os.getpid()), encoding="ascii")
    ERROR_PATH.unlink(missing_ok=True)
    log("RadAz conditioned 2x2 factorial queue started")
    free_gib = shutil.disk_usage(ROOT).free / 1024.0**3
    log(f"disk_free_gib={free_gib:.2f}")
    if free_gib < 20.0:
        raise RuntimeError(f"At least 20 GiB free is required; got {free_gib:.2f}")
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)
    for case in CASES:
        if not (ROOT / case["config"]).is_file():
            raise FileNotFoundError(ROOT / case["config"])

    smoke_marker = LOG_DIR / "preflight_passed.json"
    if not smoke_marker.is_file():
        audit_path = (
            OUTPUT_ROOT
            / "radaz_conditioned_factorial_manifests"
            / "gradient_audit.json"
        )
        audit_valid = False
        if audit_path.is_file():
            try:
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                audit_valid = (
                    len(audit.get("audits", [])) == 6
                    and audit.get("smoke", {}).get("output_shape")
                    == [1, 10, 3, 260, 256]
                    and float(audit.get("smoke", {}).get("total_loss", -1.0)) > 0.0
                )
            except Exception:
                audit_valid = False
        if audit_valid:
            log(f"preflight accepted from completed gradient audit: {audit_path}")
        else:
            run_stage(
                "preflight",
                [
                    str(PYTHON),
                    "smoke_test_radaz_conditioned_factorial.py",
                    str(MANIFEST),
                    "--device", "cuda",
                    "--output", str(audit_path),
                ],
            )
        smoke_marker.write_text(
            json.dumps(
                {
                    "completed_at": datetime.now().isoformat(timespec="seconds"),
                    "manifest": str(MANIFEST),
                    "gradient_audit": str(audit_path),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        log(f"preflight skipped: {smoke_marker} exists")

    for case in CASES:
        if complete_path(case).is_file():
            log(f"{case['key']} skipped: training_complete.json exists")
            continue
        started = time.time()
        run_stage(f"train_{case['key'].replace('-', '')}", train_command(case))
        elapsed = time.time() - started
        mark_complete(case, elapsed)
        log(f"{case['key']} training complete")
    log("RadAz conditioned 2x2 factorial queue finished")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        details = traceback.format_exc()
        ERROR_PATH.write_text(details, encoding="utf-8")
        log(f"FATAL: {details.splitlines()[-1]}")
        raise
