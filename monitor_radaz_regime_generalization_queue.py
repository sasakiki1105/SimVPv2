#!/usr/bin/env python3
"""Report command-line progress for the RadAz regime-generalization queue."""

from __future__ import annotations

import argparse
import ctypes
import json
import re
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "workdirs" / "2D_RadAz"
LOG_DIR = OUTPUT_ROOT / "radaz_regime_generalization_queue_logs"
QUEUE_LOG = LOG_DIR / "queue.log"
PID_PATH = LOG_DIR / "queue.pid"
ERROR_PATH = LOG_DIR / "runner_error.log"
MANIFEST_DIR = OUTPUT_ROOT / "radaz_regime_generalization_manifests"
CASES = (
    (
        "low_E10_E20",
        "radaz_bx20mt_lowE_E10_E20_mixed_direct10_sourcepool_noclip_disjoint811_bs1_100ep",
    ),
    (
        "high_E30_E40",
        "radaz_bx20mt_highE_E30_E40_mixed_direct10_sourcepool_noclip_disjoint811_bs1_100ep",
    ),
)
EPOCH_PATTERN = re.compile(
    r"^(?P<stamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - "
    r"Epoch (?P<epoch>\d+):.*Train Loss: (?P<train>[0-9.eE+-]+) "
    r"\| Vali Loss: (?P<val>[0-9.eE+-]+)"
)


def process_state():
    if not PID_PATH.exists():
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


def experiment_status(key: str, ex_name: str, queue_running: bool) -> str:
    workdir = OUTPUT_ROOT / ex_name
    complete = workdir / "training_complete.json"
    if complete.exists():
        payload = json.loads(complete.read_text(encoding="utf-8"))
        return (
            f"DONE epochs={payload.get('epochs')} "
            f"completed={payload.get('completed_at')} best.ckpt=yes"
        )
    records = epoch_records(workdir)
    last_ckpt = workdir / "checkpoints" / "last.ckpt"
    if records:
        epoch, stamp, train_loss, val_loss = records[-1]
        status = (
            f"TRAINING epoch={epoch + 1}/100 train={train_loss:.6g} "
            f"val={val_loss:.6g} log={stamp:%Y-%m-%d %H:%M:%S}"
        )
        if len(records) >= 2 and epoch < 99:
            recent = records[-min(11, len(records)) :]
            intervals = [
                (recent[index][1] - recent[index - 1][1]).total_seconds()
                for index in range(1, len(recent))
            ]
            seconds_per_epoch = sum(intervals) / len(intervals)
            finish = datetime.now() + timedelta(seconds=(99 - epoch) * seconds_per_epoch)
            status += (
                f" sec/epoch={seconds_per_epoch:.0f} ETA={finish:%Y-%m-%d %H:%M}"
            )
        return status
    if last_ckpt.exists():
        return (
            "TRAINING/RESUMING checkpoint exists; waiting for the first complete "
            "epoch log"
        )
    model_param = workdir / "model_param.json"
    if model_param.exists():
        return (
            "STARTING model initialized; waiting for first epoch "
            f"(updated={datetime.fromtimestamp(model_param.stat().st_mtime):%H:%M:%S})"
        )
    if queue_running:
        return "PENDING/PREPARING no model checkpoint yet"
    return "PENDING queue is not running"


def show_once() -> None:
    pid, running = process_state()
    print("=" * 116)
    print(f"RadAz regime-generalization queue monitor  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"queue_pid={pid if pid is not None else 'unknown'} running={running}")
    print(
        "manifests="
        f"low:{(MANIFEST_DIR / 'radaz_low_source_manifest.json').exists()} "
        f"high:{(MANIFEST_DIR / 'radaz_high_source_manifest.json').exists()}"
    )
    if QUEUE_LOG.exists():
        print(
            f"queue_log={QUEUE_LOG} "
            f"updated={datetime.fromtimestamp(QUEUE_LOG.stat().st_mtime):%Y-%m-%d %H:%M:%S}"
        )
        lines = QUEUE_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in lines[-6:]:
            print(line)
    else:
        print(f"queue_log={QUEUE_LOG} missing")
    print("-" * 116)
    for key, ex_name in CASES:
        print(f"{key:<18} {experiment_status(key, ex_name, running)}")
    if ERROR_PATH.exists():
        lines = ERROR_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
        if lines:
            print(f"last_error={lines[-1]}")
    print("=" * 116)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.parse_args()
    show_once()


if __name__ == "__main__":
    main()
