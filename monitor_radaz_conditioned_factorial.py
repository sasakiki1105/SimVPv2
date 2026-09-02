#!/usr/bin/env python3
"""Report command-line progress for the conditioned factorial queue."""

from __future__ import annotations

import argparse
import ctypes
import json
import re
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "workdirs" / "2D_RadAz"
LOG_DIR = OUTPUT_ROOT / "radaz_conditioned_factorial_queue_logs"
QUEUE_LOG = LOG_DIR / "queue.log"
PID_PATH = LOG_DIR / "queue.pid"
ERROR_PATH = LOG_DIR / "runner_error.log"
EPOCHS = 60
CASES = (
    ("U-D", "radaz_axis_factorial_UD_direct10_bs1_60ep"),
    ("C-D", "radaz_axis_factorial_CD_film_direct10_bs1_60ep"),
    ("U-P", "radaz_axis_factorial_UP_qtransport_direct10_bs1_60ep"),
    ("C-P", "radaz_axis_factorial_CP_film_qtransport_direct10_bs1_60ep"),
)
EPOCH_PATTERN = re.compile(
    r"^(?P<stamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - "
    r"Epoch (?P<epoch>\d+):.*Train Loss: (?P<train>[0-9.eE+-]+) "
    r"\| Vali Loss: (?P<val>[0-9.eE+-]+)"
)


def process_state():
    if not PID_PATH.is_file():
        return None, False
    try:
        pid = int(PID_PATH.read_text(encoding="ascii").strip())
    except Exception:
        return None, False
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
    running = bool(handle)
    if handle:
        ctypes.windll.kernel32.CloseHandle(handle)
    return pid, running


def epoch_records(workdir: Path):
    logs = sorted(workdir.glob("train_*.log"), key=lambda path: path.stat().st_mtime)
    if not logs:
        return []
    records = []
    for line in logs[-1].read_text(encoding="utf-8", errors="replace").splitlines():
        match = EPOCH_PATTERN.match(line)
        if match:
            records.append(
                (
                    int(match.group("epoch")),
                    datetime.strptime(match.group("stamp"), "%Y-%m-%d %H:%M:%S"),
                    float(match.group("train")),
                    float(match.group("val")),
                )
            )
    return records


def status(ex_name: str, queue_running: bool) -> str:
    workdir = OUTPUT_ROOT / ex_name
    complete = workdir / "training_complete.json"
    if complete.is_file():
        payload = json.loads(complete.read_text(encoding="utf-8"))
        return f"DONE epochs={payload.get('epochs')} completed={payload.get('completed_at')}"
    records = epoch_records(workdir)
    if records:
        epoch, stamp, train_loss, val_loss = records[-1]
        message = (
            f"TRAINING epoch={epoch + 1}/{EPOCHS} train={train_loss:.6g} "
            f"val={val_loss:.6g} log={stamp:%Y-%m-%d %H:%M:%S}"
        )
        if len(records) >= 2 and epoch < EPOCHS - 1:
            recent = records[-min(11, len(records)) :]
            intervals = [
                (recent[i][1] - recent[i - 1][1]).total_seconds()
                for i in range(1, len(recent))
            ]
            seconds_per_epoch = sum(intervals) / len(intervals)
            finish = datetime.now() + timedelta(
                seconds=(EPOCHS - 1 - epoch) * seconds_per_epoch
            )
            message += f" sec/epoch={seconds_per_epoch:.0f} ETA={finish:%Y-%m-%d %H:%M}"
        return message
    if (workdir / "checkpoints" / "last.ckpt").is_file():
        return "TRAINING/RESUMING checkpoint exists; waiting for epoch log"
    if (workdir / "model_param.json").is_file():
        updated = datetime.fromtimestamp((workdir / "model_param.json").stat().st_mtime)
        return f"STARTING model initialized (updated={updated:%H:%M:%S})"
    return "PENDING/PREPARING" if queue_running else "PENDING queue is not running"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.parse_args()
    pid, running = process_state()
    print("=" * 116)
    print(f"RadAz conditioned factorial monitor  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"queue_pid={pid if pid is not None else 'unknown'} running={running}")
    if QUEUE_LOG.is_file():
        print(
            f"queue_log={QUEUE_LOG} "
            f"updated={datetime.fromtimestamp(QUEUE_LOG.stat().st_mtime):%Y-%m-%d %H:%M:%S}"
        )
        for line in QUEUE_LOG.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()[-7:]:
            print(line)
    else:
        print(f"queue_log={QUEUE_LOG} missing")
    print("-" * 116)
    for key, ex_name in CASES:
        print(f"{key:<6} {status(ex_name, running)}")
    if ERROR_PATH.is_file():
        lines = ERROR_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
        if lines:
            print(f"last_error={lines[-1]}")
    print("=" * 116)


if __name__ == "__main__":
    main()
