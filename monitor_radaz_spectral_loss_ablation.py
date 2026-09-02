#!/usr/bin/env python3
"""Report progress for the RadAz spectral-loss ablation queue."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
PREFIX = "radaz_xe1p_bx20mt_ez10kvm_out15ns_spectral_ablation"
EXPERIMENTS = [
    ("baseline", f"{PREFIX}_baseline_5ep"),
    ("amplitude", f"{PREFIX}_amplitude_5ep"),
    ("amplitude_phase_cross", f"{PREFIX}_amplitude_phase_cross_5ep"),
]
QUEUE_LOG = ROOT / "workdirs" / "radaz_spectral_loss_ablation_logs" / "queue.log"


def saved_status(workdir: Path) -> str | None:
    paths = [workdir / "saved" / name for name in ("inputs.npy", "preds.npy", "trues.npy")]
    if not all(path.exists() for path in paths):
        return None
    arrays = [np.load(path, mmap_mode="r") for path in paths]
    return (
        f"DONE inputs={tuple(arrays[0].shape)} preds={tuple(arrays[1].shape)} "
        f"trues={tuple(arrays[2].shape)}"
    )


def checkpoint_status(workdir: Path) -> str:
    checkpoint = workdir / "checkpoints" / "last.ckpt"
    if not checkpoint.exists():
        return "PENDING no last.ckpt"
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        epoch = int(state.get("epoch", -1)) + 1
        step = int(state.get("global_step", -1))
        updated = datetime.fromtimestamp(checkpoint.stat().st_mtime)
        return (
            f"TRAINING_OR_TEST epoch={epoch}/5 global_step={step} "
            f"updated={updated:%Y-%m-%d %H:%M:%S}"
        )
    except Exception as exc:
        return f"checkpoint read error: {exc}"


def show_once() -> None:
    print("=" * 96)
    print(f"RadAz spectral-loss ablation monitor  {datetime.now():%Y-%m-%d %H:%M:%S}")
    if QUEUE_LOG.exists():
        print(
            f"queue_log={QUEUE_LOG} "
            f"updated={datetime.fromtimestamp(QUEUE_LOG.stat().st_mtime):%Y-%m-%d %H:%M:%S}"
        )
        lines = QUEUE_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in lines[-4:]:
            print(line)
    else:
        print(f"queue_log={QUEUE_LOG} missing")
    print("-" * 96)
    for label, ex_name in EXPERIMENTS:
        workdir = ROOT / "workdirs" / ex_name
        print(f"{label:<26} {saved_status(workdir) or checkpoint_status(workdir)}")
    print("=" * 96)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.parse_args()
    show_once()


if __name__ == "__main__":
    main()
