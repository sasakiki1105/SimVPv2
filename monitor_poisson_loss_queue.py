import argparse
import os
import subprocess
import warnings
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
WORKDIRS = ROOT / "workdirs"
LOG_DIR = WORKDIRS / "poisson_loss_queue_logs"


CASES = [
    {
        "name": "baseline",
        "ex_name": "pepapic_simvp_gsta_highmag_macro5_subsample2_trainfixed_disjoint_811_bs2_100ep",
    },
    {
        "name": "poisson_zero_weak",
        "ex_name": (
            "pepapic_simvp_gsta_highmag_macro5_subsample2_direct10_"
            "poisson_zero_lam1em3_trainfixed_disjoint_811_bs2_100ep"
        ),
    },
    {
        "name": "poisson_floor_hinge",
        "ex_name": (
            "pepapic_simvp_gsta_highmag_macro5_subsample2_direct10_"
            "poisson_floor_hinge_lam1em3_floor086_alpha11_trainfixed_disjoint_811_bs2_100ep"
        ),
    },
    {
        "name": "efield_weak",
        "ex_name": (
            "pepapic_simvp_gsta_highmag_macro5_subsample2_direct10_"
            "efield_lam1em3_trainfixed_disjoint_811_bs2_100ep"
        ),
    },
    {
        "name": "poisson_zero_efield_weak",
        "ex_name": (
            "pepapic_simvp_gsta_highmag_macro5_subsample2_direct10_"
            "poisson_zero_lam1em3_efield_lam1em3_trainfixed_disjoint_811_bs2_100ep"
        ),
    },
]


def fmt_time(path):
    if not path.exists():
        return "-"
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def saved_status(case):
    saved = WORKDIRS / case["ex_name"] / "saved"
    paths = [saved / "inputs.npy", saved / "preds.npy", saved / "trues.npy"]
    if not all(path.exists() for path in paths):
        return None
    try:
        inputs = np.load(paths[0], mmap_mode="r")
        preds = np.load(paths[1], mmap_mode="r")
        trues = np.load(paths[2], mmap_mode="r")
        return f"DONE saved inputs={tuple(inputs.shape)} preds={tuple(preds.shape)} trues={tuple(trues.shape)}"
    except Exception as exc:
        return f"SAVED_ERROR {exc}"


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


def checkpoint_status(case, running_text=""):
    ckpt = WORKDIRS / case["ex_name"] / "checkpoints" / "last.ckpt"
    if not ckpt.exists():
        return "PENDING no last.ckpt"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = torch.load(str(ckpt), map_location="cpu")
        epoch = data.get("epoch", "?")
        step = data.get("global_step", "?")
        prefix = "TRAINING_OR_TEST" if case["ex_name"] in running_text else "INCOMPLETE_STOPPED"
        return f"{prefix} epoch={epoch}/100 global_step={step} last_ckpt={fmt_time(ckpt)}"
    except Exception as exc:
        return f"CKPT_READ_ERROR last_ckpt={fmt_time(ckpt)} error={exc}"


def print_status():
    print("=" * 88, flush=True)
    print(f"poisson loss queue monitor  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    queue_log = LOG_DIR / "queue.log"
    print(f"queue_log: {queue_log}  updated={fmt_time(queue_log)}", flush=True)
    print("-" * 88, flush=True)
    running_text = running_python_command_lines()
    for case in CASES:
        done = saved_status(case)
        status = done if done is not None else checkpoint_status(case, running_text)
        print(f"{case['name']:<20} {status}", flush=True)
    print("=" * 88, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    print_status()
    if args.once:
        return
    import time

    while True:
        time.sleep(args.interval)
        print_status()


if __name__ == "__main__":
    main()
