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
LOG_DIR = WORKDIRS / "mixed_b_sweep_queue_logs"


CASES = [
    {
        "name": "stride2_all_cases_data_only",
        "ex_name": (
            "pepapic_simvp_gsta_mixed_b_sweep_stride2_direct10_"
            "all_cases_data_only_trainfixed_disjoint_811_bs2_100ep"
        ),
    },
    {
        "name": "stride2_leaveout_1p0mT_data_only",
        "ex_name": (
            "pepapic_simvp_gsta_mixed_b_sweep_stride2_direct10_"
            "leaveout_1p0mT_data_only_trainfixed_disjoint_811_bs2_100ep"
        ),
    },
    {
        "name": "stride2_leaveout_1p75mT_data_only",
        "ex_name": (
            "pepapic_simvp_gsta_mixed_b_sweep_stride2_direct10_"
            "leaveout_1p75mT_data_only_trainfixed_disjoint_811_bs2_100ep"
        ),
    },
    {
        "name": "stride2_leaveout_1p75mT_b_conditioned_4ch",
        "ex_name": (
            "pepapic_simvp_gsta_mixed_b_sweep_stride2_direct10_"
            "leaveout_1p75mT_b_conditioned_4ch_trainfixed_disjoint_811_bs2_100ep"
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
        result = None
    if result is not None and result.returncode == 0 and result.stdout:
        return result.stdout

    # Some restricted shells cannot read Win32_Process.CommandLine. Fall back to
    # a weaker check so the monitor does not falsely report a live run as stopped.
    fallback = "Get-Process python -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Id"
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", fallback],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return ""
    if proc.returncode == 0 and proc.stdout.strip():
        return "__PYTHON_RUNNING_CMDLINE_UNAVAILABLE__"
    return ""


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
        if case["ex_name"] in running_text:
            prefix = "TRAINING_OR_TEST"
        elif "__PYTHON_RUNNING_CMDLINE_UNAVAILABLE__" in running_text:
            prefix = "PYTHON_RUNNING_CMDLINE_UNAVAILABLE"
        else:
            prefix = "INCOMPLETE_STOPPED"
        return f"{prefix} epoch={epoch}/100 global_step={step} last_ckpt={fmt_time(ckpt)}"
    except Exception as exc:
        return f"CKPT_READ_ERROR last_ckpt={fmt_time(ckpt)} error={exc}"


def print_status():
    print("=" * 88, flush=True)
    print(f"mixed B-sweep queue monitor  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    queue_log = LOG_DIR / "queue.log"
    print(f"queue_log: {queue_log}  updated={fmt_time(queue_log)}", flush=True)
    print("-" * 88, flush=True)
    running_text = running_python_command_lines()
    for case in CASES:
        done = saved_status(case)
        status = done if done is not None else checkpoint_status(case, running_text)
        print(f"{case['name']:<30} {status}", flush=True)
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
