import argparse
import os
import subprocess
import time
import warnings
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
WORKDIRS = ROOT / "workdirs"
LOG_DIR = WORKDIRS / "long_horizon_physics_queue_logs"

METHODS = {
    "efield": "efield_lam1em3",
    "poisson_zero_efield": "poisson_zero_lam1em3_efield_lam1em3",
}


def ex_name(method, aft):
    return (
        f"pepapic_simvp_gsta_highmag_macro5_subsample2_direct{aft}_"
        f"{METHODS[method]}_trainfixed_disjoint_811_bs2_100ep"
    )


def fmt_time(path):
    if not path.exists():
        return "-"
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def saved_status(name, aft):
    saved = WORKDIRS / name / "saved"
    paths = [saved / "inputs.npy", saved / "preds.npy", saved / "trues.npy"]
    if not all(path.exists() for path in paths):
        return None
    try:
        inputs = np.load(paths[0], mmap_mode="r")
        preds = np.load(paths[1], mmap_mode="r")
        trues = np.load(paths[2], mmap_mode="r")
    except Exception as exc:
        return f"SAVED_ERROR {exc}"
    if preds.ndim == 5 and preds.shape[1] == aft and trues.shape == preds.shape:
        return f"DONE saved inputs={tuple(inputs.shape)} preds={tuple(preds.shape)} trues={tuple(trues.shape)}"
    return f"SAVED_SHAPE_MISMATCH inputs={tuple(inputs.shape)} preds={tuple(preds.shape)} trues={tuple(trues.shape)}"


def running_python_command_lines():
    if os.name != "nt":
        return ""
    cmd = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -eq 'python.exe' -or $_.Name -eq 'pythonw.exe' } | "
        "Select-Object -ExpandProperty CommandLine"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return ""
    return result.stdout or ""


def checkpoint_status(name, running_text):
    ckpt = WORKDIRS / name / "checkpoints" / "last.ckpt"
    if not ckpt.exists():
        return "PENDING no last.ckpt"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = torch.load(str(ckpt), map_location="cpu")
        epoch = data.get("epoch", "?")
        step = data.get("global_step", "?")
        prefix = "TRAINING_OR_TEST" if name in running_text else "INCOMPLETE_STOPPED"
        return f"{prefix} epoch={epoch}/100 global_step={step} last_ckpt={fmt_time(ckpt)}"
    except Exception as exc:
        return f"CKPT_READ_ERROR last_ckpt={fmt_time(ckpt)} error={exc}"


def print_status(afts):
    print("=" * 94, flush=True)
    print(f"long horizon physics queue monitor  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    queue_log = LOG_DIR / "queue.log"
    print(f"queue_log: {queue_log}  updated={fmt_time(queue_log)}", flush=True)
    print("-" * 94, flush=True)
    running = running_python_command_lines()
    for aft in afts:
        data_name = f"pepapic_simvp_gsta_highmag_macro5_subsample2_direct{aft}_trainfixed_disjoint_811_bs2_100ep"
        data_done = saved_status(data_name, aft)
        data_status = data_done if data_done else checkpoint_status(data_name, running)
        print(f"direct{aft}_data_only           {data_status}", flush=True)
        for method in METHODS:
            name = ex_name(method, aft)
            done = saved_status(name, aft)
            status = done if done else checkpoint_status(name, running)
            print(f"direct{aft}_{method:<20} {status}", flush=True)
    print("=" * 94, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aft", type=int, action="append", default=None)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    afts = args.aft or [20]
    print_status(afts)
    if args.once:
        return
    while True:
        time.sleep(args.interval)
        print_status(afts)


if __name__ == "__main__":
    main()
