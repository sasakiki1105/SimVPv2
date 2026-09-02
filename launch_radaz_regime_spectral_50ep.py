#!/usr/bin/env python3
"""Launch the mixed RadAz spectral queue through Windows Task Scheduler."""

from __future__ import annotations

import subprocess
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TASK_NAME = "RadAzRegimeSpectral50ep"
PYTHONW = Path(r"C:\Users\astro\anaconda3\envs\OpenSTL\pythonw.exe")
PYTHON = Path(r"C:\Users\astro\anaconda3\envs\OpenSTL\python.exe")
QUEUE_SCRIPT = ROOT / "run_radaz_regime_spectral_50ep_queue.py"
LOG_DIR = ROOT / "workdirs" / "2D_RadAz" / "radaz_regime_spectral_50ep_queue_logs"


def run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    output = "\n".join(
        part.strip() for part in (result.stdout, result.stderr) if part.strip()
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n{output}"
        )
    return output


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not PYTHONW.exists():
        raise FileNotFoundError(PYTHONW)
    if not PYTHON.exists():
        raise FileNotFoundError(PYTHON)
    start_time = (datetime.now() + timedelta(minutes=2)).strftime("%H:%M")
    task_command = f'"{PYTHONW}" "{QUEUE_SCRIPT}"'
    try:
        print(
            run(
                [
                    "schtasks.exe",
                    "/Create",
                    "/TN",
                    TASK_NAME,
                    "/TR",
                    task_command,
                    "/SC",
                    "ONCE",
                    "/ST",
                    start_time,
                    "/F",
                ]
            )
        )
        print(run(["schtasks.exe", "/Run", "/TN", TASK_NAME]))
        print(f"launch_method=task_scheduler task_name={TASK_NAME}")
    except RuntimeError as exc:
        # Some Windows sessions cannot access Task Scheduler even though the
        # executable paths are valid. A detached, no-window python process
        # keeps the queue independent while preserving child-process DLL setup.
        process = subprocess.Popen(
            [str(PYTHON), str(QUEUE_SCRIPT)],
            cwd=ROOT,
            close_fds=True,
            creationflags=(
                subprocess.DETACHED_PROCESS
                | subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.CREATE_NO_WINDOW
            ),
        )
        print(f"task_scheduler_unavailable={exc}")
        print(f"launch_method=detached_pythonw launcher_pid={process.pid}")
    print(f"queue_log={LOG_DIR / 'queue.log'}")


if __name__ == "__main__":
    main()
