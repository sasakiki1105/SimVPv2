#!/usr/bin/env python3
"""Report progress and ETA for the RadAz spectral-full 100-epoch run."""

from __future__ import annotations

import argparse
import ctypes
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
EX_NAME = (
    "radaz_xe1p_bx20mt_ez10kvm_out15ns_native257x256_direct10_"
    "spectral_full_trainfixed_disjoint_811_bs1_100ep"
)
WORKDIR = ROOT / "workdirs" / EX_NAME
LOG_DIR = ROOT / "workdirs" / "radaz_spectral_full_100ep_queue_logs"
QUEUE_LOG = LOG_DIR / "queue.log"
PID_PATH = LOG_DIR / "queue.pid"
STDERR_PATH = LOG_DIR / "runner_stderr.log"


def process_running() -> tuple[int | None, bool]:
    if not PID_PATH.exists():
        return None, False
    try:
        pid = int(PID_PATH.read_text(encoding="ascii").strip())
    except Exception:
        return None, False
    process_handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
    running = bool(process_handle)
    if process_handle:
        ctypes.windll.kernel32.CloseHandle(process_handle)
    return pid, running


def process_status() -> str:
    pid, running = process_running()
    if pid is None:
        return "queue_pid=unknown running=False"
    return f"queue_pid={pid} running={running}"


def last_error_line() -> str | None:
    if not STDERR_PATH.exists():
        return None
    lines = [
        line.strip()
        for line in STDERR_PATH.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        if line.strip()
    ]
    for line in reversed(lines):
        lowered = line.lower()
        if any(
            marker in lowered
            for marker in ("error", "exception", "traceback", "failed", "forrtl")
        ):
            return line
    return lines[-1] if lines else None


def saved_status() -> str | None:
    paths = [WORKDIR / "saved" / name for name in ("inputs.npy", "preds.npy", "trues.npy")]
    if not all(path.exists() for path in paths):
        return None
    arrays = [np.load(path, mmap_mode="r") for path in paths]
    return (
        f"DONE inputs={tuple(arrays[0].shape)} preds={tuple(arrays[1].shape)} "
        f"trues={tuple(arrays[2].shape)}"
    )


def recent_epoch_timing():
    logs = sorted(WORKDIR.glob("train_*.log"), key=lambda path: path.stat().st_mtime)
    if not logs:
        return None
    pattern = re.compile(
        r"^(?P<stamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - "
        r"Epoch (?P<epoch>\d+):.*Train Loss: (?P<train>[0-9.eE+-]+) "
        r"\| Vali Loss: (?P<val>[0-9.eE+-]+)"
    )
    records = []
    for line in logs[-1].read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if match:
            records.append(
                (
                    int(match.group("epoch")),
                    datetime.strptime(match.group("stamp"), "%Y-%m-%d %H:%M:%S"),
                    float(match.group("train")),
                    float(match.group("val")),
                )
            )
    if not records:
        return None
    last = records[-1]
    if len(records) < 2:
        return last, None
    intervals = [
        (records[index][1] - records[index - 1][1]).total_seconds()
        for index in range(max(1, len(records) - 10), len(records))
    ]
    return last, sum(intervals) / len(intervals)


def checkpoint_status() -> str:
    checkpoint = WORKDIR / "checkpoints" / "last.ckpt"
    if not checkpoint.exists():
        model_param = WORKDIR / "model_param.json"
        if model_param.exists():
            return (
                "STARTING model initialized; waiting for first epoch checkpoint "
                f"(model_param updated={datetime.fromtimestamp(model_param.stat().st_mtime):%H:%M:%S})"
            )
        return "PENDING no model_param.json or last.ckpt"
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        epoch = int(state.get("epoch", -1)) + 1
        step = int(state.get("global_step", -1))
        status = (
            f"TRAINING_OR_TEST epoch={epoch}/100 global_step={step} "
            f"checkpoint={datetime.fromtimestamp(checkpoint.stat().st_mtime):%Y-%m-%d %H:%M:%S}"
        )
        timing = recent_epoch_timing()
        if timing is not None:
            last, seconds_per_epoch = timing
            status += (
                f"\nlast_log_epoch={last[0]} train_loss={last[2]:.6g} "
                f"val_loss={last[3]:.6g}"
            )
            if seconds_per_epoch is not None and epoch < 100:
                eta = datetime.now() + timedelta(seconds=(100 - epoch) * seconds_per_epoch)
                status += (
                    f" sec_per_epoch={seconds_per_epoch:.0f} "
                    f"estimated_finish={eta:%Y-%m-%d %H:%M}"
                )
        return status
    except Exception as exc:
        return f"checkpoint read error: {exc}"


def show_once() -> None:
    print("=" * 104)
    print(f"RadAz spectral-full 100 epoch monitor  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(process_status())
    if QUEUE_LOG.exists():
        print(
            f"queue_log={QUEUE_LOG} "
            f"updated={datetime.fromtimestamp(QUEUE_LOG.stat().st_mtime):%Y-%m-%d %H:%M:%S}"
        )
        lines = QUEUE_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        start_indices = [
            index
            for index, line in enumerate(lines)
            if "RadAz spectral-full 100 epoch queue started" in line
        ]
        if start_indices:
            lines = lines[start_indices[-1] :]
        for line in lines[-5:]:
            print(line)
    else:
        print(f"queue_log={QUEUE_LOG} missing")
    print("-" * 104)
    saved = saved_status()
    _, running = process_running()
    if saved is not None:
        print(saved)
    elif running:
        print(checkpoint_status())
    else:
        print("STOPPED/FAILED queue process is not running and saved arrays are missing")
        error = last_error_line()
        if error:
            print(f"last_stderr={error}")
    print("=" * 104)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.parse_args()
    show_once()


if __name__ == "__main__":
    main()
