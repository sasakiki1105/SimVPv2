#!/usr/bin/env python3
"""Train low/high mixed RadAz models with spectral losses for 50 epochs."""

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
MANIFEST_DIR = OUTPUT_ROOT / "radaz_regime_generalization_manifests"
LOG_DIR = OUTPUT_ROOT / "radaz_regime_spectral_50ep_queue_logs"
QUEUE_LOG = LOG_DIR / "queue.log"
PID_PATH = LOG_DIR / "queue.pid"
ERROR_PATH = LOG_DIR / "runner_error.log"
CONFIG = "configs/custom/pepapic/SimVP_gSTA_radaz_spectral_mixed_50ep.py"
EPOCHS = 50
CASES = (
    {
        "key": "low_E10_E20_spectral",
        "manifest": MANIFEST_DIR / "radaz_low_source_manifest.json",
        "ex_name": (
            "radaz_bx20mt_lowE_E10_E20_mixed_direct10_sourcepool_noclip_"
            "disjoint811_bs1_spectral_full_50ep"
        ),
    },
    {
        "key": "high_E30_E40_spectral",
        "manifest": MANIFEST_DIR / "radaz_high_source_manifest.json",
        "ex_name": (
            "radaz_bx20mt_highE_E30_E40_mixed_direct10_sourcepool_noclip_"
            "disjoint811_bs1_spectral_full_50ep"
        ),
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


def training_complete(case: dict) -> Path:
    return OUTPUT_ROOT / case["ex_name"] / "training_complete.json"


def train_command(case: dict) -> list[str]:
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
        str(EPOCHS),
        "--batch_size",
        "1",
        "--val_batch_size",
        "1",
        "--num_workers",
        "0",
        "--gpus",
        "0",
        "--seed",
        "42",
        "--no_display_method_info",
    ]
    last_checkpoint = workdir / "checkpoints" / "last.ckpt"
    if last_checkpoint.exists():
        command.extend(["--ckpt_path", str(last_checkpoint)])
        log(f"{case['key']} will resume from {last_checkpoint}")
    return command


def mark_complete(case: dict, elapsed: float) -> None:
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
        "config": CONFIG,
        "epochs": EPOCHS,
        "elapsed_seconds_this_invocation": elapsed,
        "best_checkpoint": str(best),
        "last_checkpoint": str(last),
        "checkpoint_selection": "minimum validation total loss",
        "test_arrays_materialized": False,
    }
    training_complete(case).write_text(
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
    log("RadAz mixed spectral 50-epoch queue started")
    free_gib = shutil.disk_usage(ROOT).free / 1024.0**3
    log(f"disk_free_gib={free_gib:.2f}")
    if free_gib < 8.0:
        raise RuntimeError(
            f"At least 8 GiB free space is required; got {free_gib:.2f}"
        )
    if not PYTHON.exists():
        raise FileNotFoundError(PYTHON)
    for case in CASES:
        if not case["manifest"].is_file():
            raise FileNotFoundError(case["manifest"])

    smoke_marker = LOG_DIR / "mixed_spectral_smoke_passed.json"
    if not smoke_marker.exists():
        for case in CASES:
            run_stage(
                f"smoke_{case['key']}",
                [
                    str(PYTHON),
                    "smoke_test_radaz_mixed_spectral.py",
                    str(case["manifest"]),
                    "--device",
                    "cuda",
                ],
            )
        smoke_marker.write_text(
            json.dumps(
                {
                    "completed_at": datetime.now().isoformat(timespec="seconds"),
                    "manifests": [str(case["manifest"]) for case in CASES],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        log(f"spectral smoke tests skipped: {smoke_marker} exists")

    for case in CASES:
        complete = training_complete(case)
        if complete.exists():
            log(f"{case['key']} skipped: {complete} exists")
            continue
        started = time.time()
        run_stage(f"train_{case['key']}", train_command(case))
        elapsed = time.time() - started
        mark_complete(case, elapsed)
        log(f"{case['key']} training complete")
    log("RadAz mixed spectral 50-epoch queue finished")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        details = traceback.format_exc()
        ERROR_PATH.write_text(details, encoding="utf-8")
        log(f"FATAL: {details.splitlines()[-1]}")
        raise
