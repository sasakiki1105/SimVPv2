#!/usr/bin/env python3
"""Train the full RadAz spectral-loss model for 100 epochs and evaluate it."""

from __future__ import annotations

import os
import runpy
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PYTHON = Path(r"C:\Users\astro\anaconda3\envs\OpenSTL\python.exe")
DATA_H5 = (
    ROOT.parent
    / "PEPAPIC"
    / "test"
    / "results"
    / "2D_Landmark"
    / "2D_RadAz_Xe1p_Bx20mT_Ez10kVm_dt15ps_out15ns"
    / "2D_RadAz_Xe1p_Bx20mT_Ez10kVm_dt15ps_out15ns"
    / "SimVPv2_inputs"
    / "radaz_3ch_trainfixed_margin20_native257x256_pad260x256.h5"
)
EX_NAME = (
    "radaz_xe1p_bx20mt_ez10kvm_out15ns_native257x256_direct10_"
    "spectral_full_trainfixed_disjoint_811_bs1_100ep"
)
WORKDIR = ROOT / "workdirs" / EX_NAME
LOG_DIR = ROOT / "workdirs" / "radaz_spectral_full_100ep_queue_logs"
LOG_PATH = LOG_DIR / "queue.log"
PID_PATH = LOG_DIR / "queue.pid"
STDOUT_PATH = LOG_DIR / "runner_stdout.log"
STDERR_PATH = LOG_DIR / "runner_stderr.log"
CONFIG = "configs/custom/pepapic/SimVP_gSTA_radaz_spectral_full_100ep.py"


def log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run_python_stage(name: str, script: str, arguments: list[str]) -> None:
    log(f"{name} command: {PYTHON} {script} {' '.join(arguments)}")
    started = time.time()
    old_argv = sys.argv
    try:
        sys.argv = [script, *arguments]
        runpy.run_path(str(ROOT / script), run_name="__main__")
    finally:
        sys.argv = old_argv
    log(f"{name} exit_code=0 elapsed_sec={time.time() - started:.1f}")


def saved_status() -> str | None:
    paths = [WORKDIR / "saved" / name for name in ("inputs.npy", "preds.npy", "trues.npy")]
    if not all(path.exists() for path in paths):
        return None
    arrays = [np.load(path, mmap_mode="r") for path in paths]
    return (
        f"inputs={tuple(arrays[0].shape)} preds={tuple(arrays[1].shape)} "
        f"trues={tuple(arrays[2].shape)}"
    )


def train_arguments() -> list[str]:
    arguments = [
        "--dataname",
        "pepapic_h5",
        "--config_file",
        CONFIG,
        "--data_root",
        str(DATA_H5),
        "--res_dir",
        ".\\workdirs",
        "--ex_name",
        EX_NAME,
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
        "--seed",
        "42",
        "--no_display_method_info",
    ]
    checkpoint = WORKDIR / "checkpoints" / "last.ckpt"
    if checkpoint.exists():
        arguments.extend(["--ckpt_path", str(checkpoint)])
        log(f"resume checkpoint detected: {checkpoint}")
    return arguments


def redirect_process_streams() -> tuple[object, object, object, object]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stdout_handle = STDOUT_PATH.open("a", encoding="utf-8", buffering=1)
    stderr_handle = STDERR_PATH.open("a", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = stdout_handle
    sys.stderr = stderr_handle
    return stdout_handle, stderr_handle, original_stdout, original_stderr


def main() -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.chdir(ROOT)
    stdout_handle, stderr_handle, original_stdout, original_stderr = (
        redirect_process_streams()
    )
    PID_PATH.write_text(str(os.getpid()), encoding="ascii")
    try:
        log(
            f"RadAz spectral-full 100 epoch queue started pid={os.getpid()} "
            f"cwd={Path.cwd()}"
        )
        if not DATA_H5.exists():
            raise FileNotFoundError(DATA_H5)

        if saved_status() is None:
            log(
                "preflight smoke tests already passed before detached launch: "
                "CPU selectivity/gradient and GPU forward/backward"
            )
            run_python_stage(
                "train_and_test_spectral_full_100ep",
                "tools\\train.py",
                train_arguments(),
            )
        else:
            log(f"training skipped: {saved_status()}")

        status = saved_status()
        log(f"training final status: {status or 'saved arrays are missing'}")
        if status is None:
            raise SystemExit(1)
        run_python_stage(
            "evaluate_against_data_only_100ep",
            "evaluate_radaz_spectral_full_100ep.py",
            [
                "--device",
                "cuda",
                "--batch-size",
                "4",
            ],
        )
        log("RadAz spectral-full 100 epoch queue finished")
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        for handle in (stdout_handle, stderr_handle):
            handle.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        for handle in (stdout_handle, stderr_handle):
            handle.close()


if __name__ == "__main__":
    main()
