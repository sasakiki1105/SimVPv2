#!/usr/bin/env python3
"""Launch the RadAz regime-generalization queue through Task Scheduler."""

from __future__ import annotations

import subprocess
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TASK_NAME = "RadAzRegimeGeneralization"
PYTHONW = Path(r"C:\Users\astro\anaconda3\envs\OpenSTL\pythonw.exe")
QUEUE_SCRIPT = ROOT / "run_radaz_regime_generalization_queue.py"
LOG_DIR = ROOT / "workdirs" / "2D_RadAz" / "radaz_regime_generalization_queue_logs"


def run(command):
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
    if not QUEUE_SCRIPT.exists():
        raise FileNotFoundError(QUEUE_SCRIPT)
    start_time = (datetime.now() + timedelta(minutes=2)).strftime("%H:%M")
    task_command = f'"{PYTHONW}" "{QUEUE_SCRIPT}"'
    create_output = run(
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
    run_output = run(["schtasks.exe", "/Run", "/TN", TASK_NAME])
    print(create_output)
    print(run_output)
    print(f"task_name={TASK_NAME}")
    print(f"queue_log={LOG_DIR / 'queue.log'}")


if __name__ == "__main__":
    main()
