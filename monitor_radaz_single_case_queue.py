#!/usr/bin/env python3
"""Report progress for the first RadAz SimVPv2 experiment."""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
EX_NAME = (
    "radaz_xe1p_bx20mt_ez10kvm_out15ns_native257x256_"
    "direct10_trainfixed_disjoint_811_bs1_100ep"
)
WORKDIR = ROOT / "workdirs" / EX_NAME
QUEUE_LOG = ROOT / "workdirs" / "radaz_single_case_queue_logs" / "queue.log"


def checkpoint_status() -> str:
    checkpoint = WORKDIR / "checkpoints" / "last.ckpt"
    if not checkpoint.exists():
        return "no last.ckpt"
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        epoch = int(state.get("epoch", -1))
        step = int(state.get("global_step", -1))
        return (
            f"epoch={epoch + 1}/100 global_step={step} "
            f"updated={datetime.fromtimestamp(checkpoint.stat().st_mtime):%Y-%m-%d %H:%M:%S}"
        )
    except Exception as exc:
        return f"checkpoint read error: {exc}"


def saved_status() -> str | None:
    paths = [WORKDIR / "saved" / name for name in ("inputs.npy", "preds.npy", "trues.npy")]
    if not all(path.exists() for path in paths):
        return None
    arrays = [np.load(path, mmap_mode="r") for path in paths]
    return (
        f"DONE inputs={tuple(arrays[0].shape)} preds={tuple(arrays[1].shape)} "
        f"trues={tuple(arrays[2].shape)}"
    )


def show_once() -> None:
    print("=" * 88)
    print(f"RadAz single-case monitor  {datetime.now():%Y-%m-%d %H:%M:%S}")
    if QUEUE_LOG.exists():
        print(
            f"queue_log={QUEUE_LOG} "
            f"updated={datetime.fromtimestamp(QUEUE_LOG.stat().st_mtime):%Y-%m-%d %H:%M:%S}"
        )
        lines = QUEUE_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in lines[-5:]:
            print(line)
    else:
        print(f"queue_log={QUEUE_LOG} missing")
    print("-" * 88)
    done = saved_status()
    print(done if done else f"TRAINING_OR_PENDING {checkpoint_status()}")
    print("=" * 88)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    show_once()


if __name__ == "__main__":
    main()
