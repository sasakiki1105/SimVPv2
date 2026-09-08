"""Sequential, resumable execution of the frozen A-fixed D/P pilot."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "workdirs/2D_RadAz/radaz_paired_pilot_v2"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_status(state):
    state["updated_utc"] = now()
    temporary = OUT / "status.tmp.json"
    temporary.write_text(json.dumps(state, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(OUT / "status.json")


def verify_bundle():
    bundle = json.loads((OUT / "bundle.json").read_text(encoding="utf-8"))
    for rel, expected in bundle["sha256"].items():
        if digest(ROOT / rel) != expected:
            raise RuntimeError("Frozen pilot asset changed: " + rel)
    for item in bundle["source_fields"].values():
        stat = Path(item["path"]).stat()
        if stat.st_size != item["size"] or stat.st_mtime_ns != item["mtime_ns"]:
            raise RuntimeError("Source data changed: " + item["path"])
    if bundle["epochs"] != 60 or bundle["seed"] != 42 or set(bundle["commands"]) != {"D", "P"}:
        raise RuntimeError("Unexpected pilot plan")
    return bundle


def checkpoint_path(cell):
    return ROOT / f"workdirs/2D_RadAz/radaz_pilot_{cell}_GN_A_seed42_60ep/checkpoints/last.ckpt"


def checkpoint_epoch(path):
    if not path.exists():
        return None
    import torch
    checkpoint = torch.load(path, map_location="cpu")
    epoch = int(checkpoint["epoch"])
    if not 0 <= epoch <= 59 or "state_dict" not in checkpoint or not checkpoint.get("optimizer_states"):
        raise RuntimeError("Invalid training checkpoint: " + str(path))
    return epoch


def previous_child_is_alive(state):
    pid, created = state.get("child_pid"), state.get("child_create_time")
    if not pid or created is None:
        return False
    return process_identity(pid) == created


def process_identity(pid):
    """Creation FILETIME for a live Windows process; distinguish reused PIDs."""
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        if ctypes.get_last_error() == 87:  # No such process.
            return None
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        if code.value != 259:
            return None
        times = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
            raise ctypes.WinError(ctypes.get_last_error())
        return (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
    finally:
        kernel.CloseHandle(handle)


def run_child(arguments, stage, state, env):
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
            raise RuntimeError("Pilot queue already running") from e
        status_path = OUT / "status.json"
        previous = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        if previous_child_is_alive(previous):
            raise RuntimeError("Previous training/evaluation child is still running; refusing duplication")
        bundle = verify_bundle()
        if previous.get("status") == "complete":
            print("Pilot already completed", flush=True)
            return
        state = dict(previous, status="running", queue_pid=os.getpid(),
                     queue_started_utc=now(), bundle_sha256=digest(OUT / "bundle.json"))
        state.pop("error", None)
        state.setdefault("completed", {})
        write_status(state)
        env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", PYTHONDONTWRITEBYTECODE="1",
                   OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1",
                   PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
        try:
            for cell in ("D", "P"):
                verify_bundle()
                checkpoint = checkpoint_path(cell)
                epoch = checkpoint_epoch(checkpoint)
                if epoch != 59:
                    if shutil.disk_usage(ROOT).free < 8 * 1024**3:
                        raise RuntimeError("Less than 8 GiB free before training stage")
                    command = list(bundle["commands"][cell])
                    if epoch is not None:
                        command += ["--ckpt_path", str(checkpoint)]
                    elif checkpoint.parent.parent.exists():
                        # A recorded failed attempt without a checkpoint can restart from the fixed seed.
                        if previous.get("bundle_sha256") != state["bundle_sha256"]:
                            raise RuntimeError("Unregistered output directory exists for " + cell)
                    run_child(command, "train_" + cell, state, env)
                    epoch = checkpoint_epoch(checkpoint)
                    if epoch != 59:
                        raise RuntimeError("Training did not reach terminal checkpoint index 59: " + cell)
                state["completed"][cell] = {"checkpoint": str(checkpoint), "epoch_index": epoch,
                                             "sha256": digest(checkpoint), "recorded_utc": now()}
                write_status(state)
            verify_bundle()
            result = OUT / "evaluation/results.json"
            if not result.exists():
                run_child(["evaluate_radaz_paired_pilot.py"], "evaluate", state, env)
            evaluation = json.loads(result.read_text(encoding="utf-8"))
            if evaluation["bundle_sha256"] != state["bundle_sha256"]:
                raise RuntimeError("Evaluation belongs to a different bundle")
            for cell in ("D", "P"):
                if evaluation["checkpoint_sha256"][cell] != state["completed"][cell]["sha256"]:
                    raise RuntimeError("Evaluation checkpoint does not match completed training")
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
    parser.add_argument("--run", action="store_true", help="Execute or resume the registered D/P queue")
    parser.add_argument("--check", action="store_true", help="Read-only asset/data provenance verification")
    args = parser.parse_args()
    if args.run:
        run()
    elif args.check:
        bundle = verify_bundle()
        print(json.dumps({"verified_assets": len(bundle["sha256"]), "source_conditions": len(bundle["source_fields"])}))
    else:
        status = OUT / "status.json"
        print(status.read_text(encoding="utf-8") if status.exists() else "Not started")


if __name__ == "__main__":
    main()
