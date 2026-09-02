import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
WORKDIRS = ROOT / "workdirs"
LOG_DIR = WORKDIRS / "long_horizon_physics_queue_logs"
LOG_PATH = LOG_DIR / "queue.log"
PYTHON = Path(r"C:\Users\astro\anaconda3\envs\OpenSTL\python.exe")
DATA_ROOT = Path(
    r"C:\Users\astro\research\PEPAPIC\test\results"
    r"\2D_ExB_high_magnet_dt2.5e-9_maxt50e-6_macro5"
    r"\SimVPv2_inputs\global_norm_trainfixed_minmax_margin20_t0_to_t4000_step2.h5"
)

METHODS = {
    "efield": {
        "tag": "efield_lam1em3",
        "config": "configs/custom/pepapic/SimVP_gSTA_pepapic_direct_efield_weak.py",
    },
    "poisson_zero_efield": {
        "tag": "poisson_zero_lam1em3_efield_lam1em3",
        "config": "configs/custom/pepapic/SimVP_gSTA_pepapic_direct_poisson_zero_efield_weak.py",
    },
}


def log(message):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def ex_name(method, aft):
    return (
        f"pepapic_simvp_gsta_highmag_macro5_subsample2_direct{aft}_"
        f"{METHODS[method]['tag']}_trainfixed_disjoint_811_bs2_100ep"
    )


def saved_status(method, aft):
    saved = WORKDIRS / ex_name(method, aft) / "saved"
    paths = [saved / "inputs.npy", saved / "preds.npy", saved / "trues.npy"]
    if not all(path.exists() for path in paths):
        return None
    try:
        inputs = np.load(paths[0], mmap_mode="r")
        preds = np.load(paths[1], mmap_mode="r")
        trues = np.load(paths[2], mmap_mode="r")
    except Exception as exc:
        return f"SAVED_ERROR {exc}"
    if preds.ndim != 5 or preds.shape[1] != aft or trues.shape != preds.shape:
        return f"SAVED_SHAPE_MISMATCH inputs={tuple(inputs.shape)} preds={tuple(preds.shape)} trues={tuple(trues.shape)}"
    return f"done: inputs={tuple(inputs.shape)}, preds={tuple(preds.shape)}, trues={tuple(trues.shape)}"


def train_command(method, aft):
    pre = 10
    return [
        str(PYTHON),
        "tools\\train.py",
        "--dataname",
        "pepapic_h5",
        "--config_file",
        METHODS[method]["config"],
        "--data_root",
        str(DATA_ROOT),
        "--res_dir",
        ".\\workdirs",
        "--ex_name",
        ex_name(method, aft),
        "--method",
        "simvp",
        "--pre_seq_length",
        str(pre),
        "--aft_seq_length",
        str(aft),
        "--total_length",
        str(pre + aft),
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


def run_command(cmd, log_name, env):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / log_name
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] RUN {' '.join(cmd)}\n")
        f.flush()
        started = time.time()
        result = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT, check=False)
        elapsed = time.time() - started
        f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] EXIT {result.returncode} elapsed_sec={elapsed:.1f}\n")
        f.flush()
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aft", type=int, action="append", default=None, help="Future frames to train, e.g. --aft 20.")
    parser.add_argument("--method", choices=sorted(METHODS), action="append", default=None)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    afts = args.aft or [20]
    methods = args.method or ["efield", "poisson_zero_efield"]

    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    log(f"long horizon physics queue started afts={afts} methods={methods}")
    for aft in afts:
        for method in methods:
            status = saved_status(method, aft)
            log(f"direct{aft}_{method} initial status: {status or 'saved arrays are missing'}")
            if status is None:
                code = run_command(train_command(method, aft), f"direct{aft}_{method}.log", env)
                log(f"direct{aft}_{method} train/test exit code: {code}")
                if code != 0:
                    return code
            status = saved_status(method, aft)
            log(f"direct{aft}_{method} final status: {status or 'saved arrays are still missing'}")
            if status is None or status.startswith("SAVED_"):
                return 2

        if not args.skip_eval:
            cmd = [str(PYTHON), "run_long_horizon_transfer_eval.py", "--aft", str(aft), "--device", "cuda", "--batch-size", "4"]
            code = run_command(cmd, f"direct{aft}_transfer_eval.log", env)
            log(f"direct{aft} transfer evaluation exit code: {code}")
            if code != 0:
                return code
    log("long horizon physics queue finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
