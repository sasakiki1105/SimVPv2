"""Seed-43 D replicate: queued after the D/P pilot completes (user-approved 2026-09-09).

Waits for workdirs/2D_RadAz/radaz_paired_pilot_v2/status.json to reach
status=="complete", then trains the D (field-MSE) cell with the identical frozen
command except --seed 43 and a new ex_name, and finally computes the
pre-registered noise floor (RADAZ_PAIRED_PILOT_PRIMARY_ENDPOINT_ADDENDUM.md):
per-condition skill(D_seed43, truth, D_seed42) for gamma and flux_full on both
development splits. No bundle-frozen file is modified; the pilot bundle is
re-verified before training. Restartable and idempotent (own lock, terminal-
checkpoint checks); if this process dies, rerun:
    python run_radaz_seed43_replicate.py --run
"""
from __future__ import annotations

import argparse
import json
import msvcrt
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from run_radaz_paired_pilot import (
    ROOT, OUT as PILOT_OUT, digest, verify_bundle, checkpoint_epoch, previous_child_is_alive,
    process_identity,
)

OUT = ROOT / "workdirs/2D_RadAz/radaz_seed43_replicate"
EX_NAME = "radaz_pilot_D_GN_A_seed43_60ep"
CHECKPOINT = ROOT / f"workdirs/2D_RadAz/{EX_NAME}/checkpoints/last.ckpt"
POLL_SECONDS = 600


def now():
    return datetime.now(timezone.utc).isoformat()


def write_status(state):
    state["updated_utc"] = now()
    temporary = OUT / "status.tmp.json"
    temporary.write_text(json.dumps(state, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(OUT / "status.json")


def seed43_command(bundle):
    command = list(bundle["commands"]["D"])
    for flag, old, new in (("--seed", "42", "43"), ("--ex_name", "radaz_pilot_D_GN_A_seed42_60ep", EX_NAME)):
        index = command.index(flag) + 1
        if command[index] != old:
            raise RuntimeError(f"Unexpected frozen value for {flag}: {command[index]}")
        command[index] = new
    return command


def wait_for_pilot(state):
    while True:
        status = json.loads((PILOT_OUT / "status.json").read_text(encoding="utf-8"))
        if status.get("status") == "complete":
            return status
        if status.get("status") == "failed":
            raise RuntimeError("Pilot queue failed; not starting the seed-43 replicate. "
                               "Inspect " + str(PILOT_OUT / "status.json"))
        state.update(stage="waiting_for_pilot", pilot_status=status.get("status"),
                     pilot_stage=status.get("stage"))
        write_status(state)
        time.sleep(POLL_SECONDS)


def run_child(arguments, stage, state, env):
    import subprocess
    log_path = OUT / (stage + ".log")
    state.update(stage=stage, stage_started_utc=now(), log=str(log_path))
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n" + now() + " " + json.dumps(arguments) + "\n")
        log.flush()
        child = subprocess.Popen([sys.executable, "-u", *arguments], cwd=ROOT,
                                 env=env, stdout=log, stderr=subprocess.STDOUT)
        state.update(child_pid=child.pid, child_create_time=process_identity(child.pid))
        write_status(state)
        print(stage, "PID", child.pid, "log", log_path, flush=True)
        while child.poll() is None:
            time.sleep(15)
            write_status(state)
        state.update(child_pid=None, child_create_time=None, last_returncode=child.returncode)
        write_status(state)
        if child.returncode:
            raise RuntimeError(stage + " failed; inspect " + str(log_path))


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "queue.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as e:
            raise RuntimeError("Seed-43 queue already running") from e
        status_path = OUT / "status.json"
        previous = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        if previous_child_is_alive(previous):
            raise RuntimeError("Previous seed-43 child still running; refusing duplication")
        if previous.get("status") == "complete":
            print("Seed-43 replicate already completed", flush=True)
            return
        state = dict(previous, status="running", queue_pid=os.getpid(), queue_started_utc=now())
        state.pop("error", None)
        write_status(state)
        env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", PYTHONDONTWRITEBYTECODE="1",
                   OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1",
                   PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
        try:
            pilot = wait_for_pilot(state)
            bundle = verify_bundle()  # frozen pilot assets must still be intact
            state["pilot_bundle_sha256"] = digest(PILOT_OUT / "bundle.json")
            state["pilot_completed"] = {c: pilot["completed"][c]["sha256"] for c in ("D", "P")}
            (OUT / "seed43_manifest.json").exists() or (OUT / "seed43_manifest.json").write_text(
                json.dumps({
                    "purpose": "noise floor for the frozen D/P primary endpoint "
                               "(RADAZ_PAIRED_PILOT_PRIMARY_ENDPOINT_ADDENDUM.md, sha256 "
                               "8d654ed198c2641d3094693b38e8be194b954ba045b1861236ce302b74e20eda)",
                    "command": seed43_command(bundle),
                    "identical_to_D_except": ["--seed 43", "--ex_name " + EX_NAME],
                    "config_sha256": bundle["sha256"]["configs\\custom\\pepapic\\SimVP_gSTA_radaz_pilot_D_60ep.py"],
                    "created_utc": now(),
                }, indent=2), encoding="utf-8")
            epoch = checkpoint_epoch(CHECKPOINT)
            if epoch != 59:
                if shutil.disk_usage(ROOT).free < 8 * 1024**3:
                    raise RuntimeError("Less than 8 GiB free before seed-43 training")
                command = seed43_command(bundle)
                if epoch is not None:
                    command += ["--ckpt_path", str(CHECKPOINT)]
                    state["resumed_from_epoch_index"] = epoch
                run_child(command, "train_D_seed43", state, env)
                epoch = checkpoint_epoch(CHECKPOINT)
                if epoch != 59:
                    raise RuntimeError("Seed-43 training did not reach terminal index 59")
            state["completed_checkpoint"] = {"path": str(CHECKPOINT), "epoch_index": epoch,
                                             "sha256": digest(CHECKPOINT), "recorded_utc": now()}
            write_status(state)
            result = OUT / "noise_floor.json"
            if not result.exists():
                run_child(["evaluate_radaz_seed43_noise_floor.py"], "noise_floor", state, env)
            state.update(status="complete", stage="complete", result=str(result), finished_utc=now())
            write_status(state)
        except BaseException:
            state.update(status="failed", error=traceback.format_exc())
            write_status(state)
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        run()
    else:
        status = OUT / "status.json"
        print(status.read_text(encoding="utf-8") if status.exists() else "Not started")


if __name__ == "__main__":
    main()
