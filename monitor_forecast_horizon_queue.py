import argparse
import os
import subprocess
import time
import warnings
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
WORKDIRS = ROOT / "workdirs"
LOG_DIR = WORKDIRS / "forecast_horizon_queue_logs"


CASES = [
    {
        "name": "stride1_tout20",
        "ex_name": "pepapic_simvp_gsta_highmag_macro5_tout20_trainfixed_disjoint_811_bs2_100ep",
        "aft": 20,
    },
    {
        "name": "stride1_tout30",
        "ex_name": "pepapic_simvp_gsta_highmag_macro5_tout30_trainfixed_disjoint_811_bs2_100ep",
        "aft": 30,
    },
    {
        "name": "stride1_tout40",
        "ex_name": "pepapic_simvp_gsta_highmag_macro5_tout40_trainfixed_disjoint_811_bs2_100ep",
        "aft": 40,
    },
    {
        "name": "stride3_tout14",
        "ex_name": "pepapic_simvp_gsta_highmag_macro5_subsample3_tout14_trainfixed_disjoint_811_bs2_100ep",
        "aft": 14,
    },
    {
        "name": "stride2_tout40",
        "ex_name": "pepapic_simvp_gsta_highmag_macro5_subsample2_tout40_trainfixed_disjoint_811_bs2_100ep",
        "aft": 40,
    },
    {
        "name": "stride4_tout20",
        "ex_name": "pepapic_simvp_gsta_highmag_macro5_subsample4_tout20_trainfixed_disjoint_811_bs2_100ep",
        "aft": 20,
    },
    {
        "name": "stride2_tout80",
        "ex_name": "pepapic_simvp_gsta_highmag_macro5_subsample2_tout80_trainfixed_disjoint_811_bs2_100ep",
        "aft": 80,
    },
    {
        "name": "stride4_tout40",
        "ex_name": "pepapic_simvp_gsta_highmag_macro5_subsample4_tout40_trainfixed_disjoint_811_bs2_100ep",
        "aft": 40,
    },
]


def fmt_time(path):
    if not path.exists():
        return "-"
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def saved_status(case):
    saved = WORKDIRS / case["ex_name"] / "saved"
    inputs_path = saved / "inputs.npy"
    preds_path = saved / "preds.npy"
    trues_path = saved / "trues.npy"
    if not inputs_path.exists() or not preds_path.exists() or not trues_path.exists():
        return None
    try:
        inputs = np.load(inputs_path, mmap_mode="r")
        preds = np.load(preds_path, mmap_mode="r")
        trues = np.load(trues_path, mmap_mode="r")
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
    print(f"forecast horizon queue monitor  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    queue_log = LOG_DIR / "queue.log"
    print(f"queue_log: {queue_log}  updated={fmt_time(queue_log)}", flush=True)
    print("-" * 88, flush=True)
    running_text = running_python_command_lines()
    for case in CASES:
        done = saved_status(case)
        if done is not None:
            status = done
        else:
            status = checkpoint_status(case, running_text)
        print(f"{case['name']:<16} {status}", flush=True)
    print("=" * 88, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    print_status()
    if args.once:
        return
    while True:
        time.sleep(args.interval)
        print_status()


if __name__ == "__main__":
    main()
