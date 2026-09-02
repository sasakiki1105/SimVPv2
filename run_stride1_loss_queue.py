import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
WORKDIRS = ROOT / "workdirs"
LOG_DIR = WORKDIRS / "stride1_loss_queue_logs"
LOG_PATH = LOG_DIR / "queue.log"
PYTHON = Path(r"C:\Users\astro\anaconda3\envs\OpenSTL\python.exe")
DATA_ROOT = Path(
    r"C:\Users\astro\research\PEPAPIC\test\results"
    r"\2D_ExB_high_magnet_dt2.5e-9_maxt50e-6_macro5"
    r"\SimVPv2_inputs\global_norm_trainfixed_minmax_margin20_t0_to_t4000.h5"
)


CASES = [
    {
        "name": "data_only_existing",
        "ex_name": "pepapic_simvp_gsta_highmag_macro5_trainfixed_disjoint_811_bs2_100ep",
        "config": "configs/custom/pepapic/SimVP_gSTA_pepapic.py",
    },
    {
        "name": "poisson_zero_weak",
        "ex_name": (
            "pepapic_simvp_gsta_highmag_macro5_direct10_"
            "poisson_zero_lam1em3_trainfixed_disjoint_811_bs2_100ep"
        ),
        "config": "configs/custom/pepapic/SimVP_gSTA_pepapic_direct_poisson_zero_weak.py",
    },
    {
        "name": "poisson_floor_hinge",
        "ex_name": (
            "pepapic_simvp_gsta_highmag_macro5_direct10_"
            "poisson_floor_hinge_lam1em3_floor086_alpha11_trainfixed_disjoint_811_bs2_100ep"
        ),
        "config": "configs/custom/pepapic/SimVP_gSTA_pepapic_direct_poisson_floor_hinge.py",
    },
    {
        "name": "efield_weak",
        "ex_name": (
            "pepapic_simvp_gsta_highmag_macro5_direct10_"
            "efield_lam1em3_trainfixed_disjoint_811_bs2_100ep"
        ),
        "config": "configs/custom/pepapic/SimVP_gSTA_pepapic_direct_efield_weak.py",
    },
    {
        "name": "poisson_zero_efield_weak",
        "ex_name": (
            "pepapic_simvp_gsta_highmag_macro5_direct10_"
            "poisson_zero_lam1em3_efield_lam1em3_trainfixed_disjoint_811_bs2_100ep"
        ),
        "config": "configs/custom/pepapic/SimVP_gSTA_pepapic_direct_poisson_zero_efield_weak.py",
    },
]


def log(message):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def saved_status(case):
    saved = WORKDIRS / case["ex_name"] / "saved"
    paths = [saved / "inputs.npy", saved / "preds.npy", saved / "trues.npy"]
    if not all(path.exists() for path in paths):
        return None
    try:
        inputs = np.load(paths[0], mmap_mode="r")
        preds = np.load(paths[1], mmap_mode="r")
        trues = np.load(paths[2], mmap_mode="r")
    except Exception as exc:
        return f"SAVED_ERROR {exc}"
    return f"done: inputs={tuple(inputs.shape)}, preds={tuple(preds.shape)}, trues={tuple(trues.shape)}"


def train_command(case):
    return [
        str(PYTHON),
        "tools\\train.py",
        "--dataname",
        "pepapic_h5",
        "--config_file",
        case["config"],
        "--data_root",
        str(DATA_ROOT),
        "--res_dir",
        ".\\workdirs",
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
        "2",
        "--val_batch_size",
        "2",
        "--num_workers",
        "0",
        "--gpus",
        "0",
    ]


def main():
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log("stride1 loss queue started")
    for case in CASES:
        status = saved_status(case)
        log(f"{case['name']} initial status: {status or 'saved arrays are missing'}")
        if status is not None:
            continue

        cmd = train_command(case)
        log(f"{case['name']} train command: {' '.join(cmd)}")
        started = time.time()
        result = subprocess.run(cmd, cwd=str(ROOT), check=False)
        elapsed = time.time() - started
        log(f"{case['name']} train/test exit code: {result.returncode} elapsed_sec={elapsed:.1f}")
        status = saved_status(case)
        log(f"{case['name']} final status: {status or 'saved arrays are still missing'}")
        if result.returncode != 0:
            log(f"stopping queue after failure in {case['name']}")
            sys.exit(result.returncode)
    log("stride1 loss queue finished")


if __name__ == "__main__":
    main()
