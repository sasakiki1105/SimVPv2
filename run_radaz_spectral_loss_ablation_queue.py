#!/usr/bin/env python3
"""Run the 5-epoch RadAz spectral-loss ablation and evaluate all variants."""

from __future__ import annotations

import os
import subprocess
import time
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
PREFIX = "radaz_xe1p_bx20mt_ez10kvm_out15ns_spectral_ablation"
EXPERIMENTS = [
    (
        "baseline",
        "configs/custom/pepapic/SimVP_gSTA_radaz_direct.py",
        f"{PREFIX}_baseline_5ep",
    ),
    (
        "amplitude",
        "configs/custom/pepapic/SimVP_gSTA_radaz_spectral_amplitude.py",
        f"{PREFIX}_amplitude_5ep",
    ),
    (
        "amplitude_phase_cross",
        "configs/custom/pepapic/SimVP_gSTA_radaz_spectral_full.py",
        f"{PREFIX}_amplitude_phase_cross_5ep",
    ),
]
LOG_DIR = ROOT / "workdirs" / "radaz_spectral_loss_ablation_logs"
LOG_PATH = LOG_DIR / "queue.log"


def log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
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


def saved_status(ex_name: str) -> str | None:
    saved = ROOT / "workdirs" / ex_name / "saved"
    paths = [saved / name for name in ("inputs.npy", "preds.npy", "trues.npy")]
    if not all(path.exists() for path in paths):
        return None
    arrays = [np.load(path, mmap_mode="r") for path in paths]
    return (
        f"inputs={tuple(arrays[0].shape)} preds={tuple(arrays[1].shape)} "
        f"trues={tuple(arrays[2].shape)}"
    )


def train_command(config_file: str, ex_name: str) -> list[str]:
    return [
        str(PYTHON),
        "tools\\train.py",
        "--dataname",
        "pepapic_h5",
        "--config_file",
        config_file,
        "--data_root",
        str(DATA_H5),
        "--res_dir",
        ".\\workdirs",
        "--ex_name",
        ex_name,
        "--method",
        "simvp",
        "--pre_seq_length",
        "10",
        "--aft_seq_length",
        "10",
        "--total_length",
        "20",
        "--epoch",
        "5",
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


def main() -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    if not DATA_H5.exists():
        raise FileNotFoundError(DATA_H5)
    log("RadAz spectral-loss ablation queue started")

    run_stage(
        "spectral_unit_smoke",
        [str(PYTHON), "smoke_test_radaz_spectral_loss.py"],
    )
    run_stage(
        "spectral_gpu_smoke",
        [
            str(PYTHON),
            "smoke_test_radaz_simvp.py",
            str(DATA_H5),
            "--device",
            "cuda",
            "--batch-size",
            "1",
            "--spectral",
        ],
    )

    for label, config_file, ex_name in EXPERIMENTS:
        status = saved_status(ex_name)
        if status is not None:
            log(f"{label} skipped: {status}")
            continue
        run_stage(f"train_{label}", train_command(config_file, ex_name))
        status = saved_status(ex_name)
        log(f"{label} final status: {status or 'saved arrays are missing'}")
        if status is None:
            raise SystemExit(1)

    run_stage(
        "evaluate_ablation",
        [
            str(PYTHON),
            "evaluate_radaz_spectral_loss_ablation.py",
            "--device",
            "cuda",
            "--batch-size",
            "4",
        ],
    )
    log("RadAz spectral-loss ablation queue finished")


if __name__ == "__main__":
    main()
