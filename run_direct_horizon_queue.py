import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PYTHON = Path(r"C:\Users\astro\anaconda3\envs\OpenSTL\python.exe")
CONFIG = Path(r"configs\custom\pepapic\SimVP_gSTA_pepapic_direct.py")
H5_ROOT = Path(
    r"C:\Users\astro\research\PEPAPIC\test\results"
    r"\2D_ExB_high_magnet_dt2.5e-9_maxt50e-6_macro5"
    r"\SimVPv2_inputs"
)
LOG_DIR = ROOT / "workdirs" / "direct_horizon_queue_logs"


TRAIN_CASES = [
    {
        "name": "stride2_direct20",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step2.h5",
        "ex_name": "pepapic_simvp_gsta_highmag_macro5_subsample2_direct20_trainfixed_disjoint_811_bs2_100ep",
        "aft": 20,
    },
    {
        "name": "stride2_direct40",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step2.h5",
        "ex_name": "pepapic_simvp_gsta_highmag_macro5_subsample2_direct40_trainfixed_disjoint_811_bs2_100ep",
        "aft": 40,
    },
    {
        "name": "stride2_direct80",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step2.h5",
        "ex_name": "pepapic_simvp_gsta_highmag_macro5_subsample2_direct80_trainfixed_disjoint_811_bs2_100ep",
        "aft": 80,
    },
    {
        "name": "stride2_direct160",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step2.h5",
        "ex_name": "pepapic_simvp_gsta_highmag_macro5_subsample2_direct160_trainfixed_disjoint_811_bs2_100ep",
        "aft": 160,
    },
    {
        "name": "stride2_direct180",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step2.h5",
        "ex_name": "pepapic_simvp_gsta_highmag_macro5_subsample2_direct180_trainfixed_disjoint_811_bs2_100ep",
        "aft": 180,
    },
]


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def saved_dir(case):
    return ROOT / "workdirs" / case["ex_name"] / "saved"


def case_done(case):
    saved = saved_dir(case)
    paths = [saved / "inputs.npy", saved / "preds.npy", saved / "trues.npy"]
    if not all(path.exists() for path in paths):
        return False, "saved arrays are missing"

    try:
        inputs = np.load(paths[0], mmap_mode="r")
        preds = np.load(paths[1], mmap_mode="r")
        trues = np.load(paths[2], mmap_mode="r")
    except Exception as exc:
        return False, f"could not load saved arrays: {exc}"

    if inputs.ndim != 5 or preds.ndim != 5 or trues.ndim != 5:
        return False, f"unexpected ndim: inputs={inputs.shape}, preds={preds.shape}, trues={trues.shape}"
    if int(inputs.shape[1]) != 10:
        return False, f"unexpected input length: {inputs.shape}"
    if int(preds.shape[1]) != int(case["aft"]) or int(trues.shape[1]) != int(case["aft"]):
        return False, f"unexpected output length: preds={preds.shape}, trues={trues.shape}"
    if preds.shape != trues.shape:
        return False, f"preds/trues shape mismatch: preds={preds.shape}, trues={trues.shape}"
    return True, f"done: inputs={inputs.shape}, preds={preds.shape}, trues={trues.shape}"


def train_command(case):
    pre = 10
    aft = int(case["aft"])
    total = pre + aft
    return [
        str(PYTHON),
        "tools\\train.py",
        "--dataname",
        "pepapic_h5",
        "--config_file",
        str(CONFIG),
        "--data_root",
        str(case["h5"]),
        "--res_dir",
        ".\\workdirs",
        "--ex_name",
        case["ex_name"],
        "--method",
        "simvp",
        "--pre_seq_length",
        str(pre),
        "--aft_seq_length",
        str(aft),
        "--total_length",
        str(total),
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


def run_logged(cmd, log_path, env):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n[{now()}] RUN {' '.join(cmd)}\n")
        log_file.flush()
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        log_file.write(f"\n[{now()}] EXIT {result.returncode}\n")
        log_file.flush()
    return result.returncode


def wait_for_pid(pid):
    if pid is None:
        return
    cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        f"Wait-Process -Id {int(pid)} -ErrorAction SilentlyContinue",
    ]
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, default=None)
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    queue_log = LOG_DIR / "queue.log"

    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    with queue_log.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n[{now()}] direct horizon queue started\n")
        if args.wait_pid is not None:
            log_file.write(f"[{now()}] waiting for pid {args.wait_pid}\n")
        log_file.flush()

    wait_for_pid(args.wait_pid)

    with queue_log.open("a", encoding="utf-8") as log_file:
        if args.wait_pid is not None:
            log_file.write(f"[{now()}] wait finished for pid {args.wait_pid}\n")
        log_file.flush()

    for case in TRAIN_CASES:
        ok, message = case_done(case)
        with queue_log.open("a", encoding="utf-8") as log_file:
            log_file.write(f"[{now()}] {case['name']} initial status: {message}\n")
            log_file.flush()
        if ok:
            continue

        code = run_logged(train_command(case), LOG_DIR / f"{case['name']}.log", env)
        if code != 0:
            with queue_log.open("a", encoding="utf-8") as log_file:
                log_file.write(f"[{now()}] {case['name']} failed with code {code}\n")
            return code

        ok, message = case_done(case)
        with queue_log.open("a", encoding="utf-8") as log_file:
            log_file.write(f"[{now()}] {case['name']} final status: {message}\n")
            log_file.flush()
        if not ok:
            return 2

    eval_cmd = [str(PYTHON), "plot_direct_horizon_baseline.py"]
    code = run_logged(eval_cmd, LOG_DIR / "evaluation.log", env)
    with queue_log.open("a", encoding="utf-8") as log_file:
        log_file.write(f"[{now()}] evaluation exit code: {code}\n")
        log_file.flush()
    if code != 0:
        return code

    block_eval_cmd = [str(PYTHON), "plot_stride2_direct_horizon_blocks.py"]
    code = run_logged(block_eval_cmd, LOG_DIR / "block_evaluation.log", env)
    with queue_log.open("a", encoding="utf-8") as log_file:
        log_file.write(f"[{now()}] block evaluation exit code: {code}\n")
        log_file.flush()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
