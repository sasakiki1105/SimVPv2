#!/usr/bin/env python3
"""Build, smoke-test, train, and test the first RadAz SimVPv2 experiment."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PYTHON = Path(r"C:\Users\astro\anaconda3\envs\OpenSTL\python.exe")
CASE_ROOT = (
    ROOT.parent
    / "PEPAPIC"
    / "test"
    / "results"
    / "2D_Landmark"
    / "2D_RadAz_Xe1p_Bx20mT_Ez10kVm_dt15ps_out15ns"
    / "2D_RadAz_Xe1p_Bx20mT_Ez10kVm_dt15ps_out15ns"
)
SOURCE_H5 = CASE_ROOT / "analysis_fields_uncompressed.h5"
INPUT_DIR = CASE_ROOT / "SimVPv2_inputs"
DATA_H5 = INPUT_DIR / "radaz_3ch_trainfixed_margin20_native257x256_pad260x256.h5"
EX_NAME = (
    "radaz_xe1p_bx20mt_ez10kvm_out15ns_native257x256_"
    "direct10_trainfixed_disjoint_811_bs1_100ep"
)
WORKDIR = ROOT / "workdirs" / EX_NAME
LOG_DIR = ROOT / "workdirs" / "radaz_single_case_queue_logs"
LOG_PATH = LOG_DIR / "queue.log"


def log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run_stage(name: str, command: list[str]) -> None:
    log(f"{name} command: {' '.join(command)}")
    started = time.time()
    result = subprocess.run(command, cwd=ROOT, check=False)
    elapsed = time.time() - started
    log(f"{name} exit_code={result.returncode} elapsed_sec={elapsed:.1f}")
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def saved_status() -> str | None:
    paths = [WORKDIR / "saved" / name for name in ("inputs.npy", "preds.npy", "trues.npy")]
    if not all(path.exists() for path in paths):
        return None
    arrays = [np.load(path, mmap_mode="r") for path in paths]
    return (
        f"inputs={tuple(arrays[0].shape)} preds={tuple(arrays[1].shape)} "
        f"trues={tuple(arrays[2].shape)}"
    )


def main() -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log("RadAz single-case queue started")

    if not SOURCE_H5.exists():
        raise FileNotFoundError(SOURCE_H5)
    if not DATA_H5.exists():
        run_stage(
            "build_h5",
            [
                str(PYTHON),
                "build_radaz_single_case_h5.py",
                str(SOURCE_H5),
                str(DATA_H5),
                "--spatial-stride",
                "1",
            ],
        )
    else:
        log(f"build_h5 skipped: {DATA_H5} already exists")

    if saved_status() is not None:
        log(f"training skipped: saved arrays already exist ({saved_status()})")
        return

    run_stage(
        "gpu_smoke_test",
        [
            str(PYTHON),
            "smoke_test_radaz_simvp.py",
            str(DATA_H5),
            "--device",
            "cuda",
            "--batch-size",
            "1",
        ],
    )
    run_stage(
        "train_and_test",
        [
            str(PYTHON),
            "tools\\train.py",
            "--dataname",
            "pepapic_h5",
            "--config_file",
            "configs/custom/pepapic/SimVP_gSTA_radaz_direct.py",
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
            "--no_display_method_info",
        ],
    )
    status = saved_status()
    log(f"final status: {status or 'saved arrays are missing'}")
    if status is None:
        raise SystemExit(1)
    log("RadAz single-case queue finished")


if __name__ == "__main__":
    main()
