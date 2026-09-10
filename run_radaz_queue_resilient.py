"""Run an existing RadAz queue with bounded retries for Windows status-file locks.

The 2026-09-10 pilot supervisor exited on WinError 5 during atomic status
replacement, while its P training child continued to epoch 59. This launcher
changes only status publication; frozen training/evaluation sources and the
bundle remain intact. Queue locks, child identity checks, commands, checkpoint
validation, and experiment decisions remain those of the original runners.

Usage: python run_radaz_queue_resilient.py --queue pilot --run
       python run_radaz_queue_resilient.py --queue seed43 --run
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import run_radaz_paired_pilot as pilot


def atomic_json_replace(path, state, *, timeout=30.0):
    """Publish complete JSON, retrying access/sharing failures for at most 30 s.

    Persistent errors still fail visibly. Readers continue seeing the previous
    complete status while replacement is blocked; no non-atomic fallback is used.
    """
    path = Path(path)
    payload = json.dumps(state, indent=2, allow_nan=False)
    deadline = time.monotonic() + timeout
    temporary = None
    attempts = 0
    try:
        while True:
            try:
                if temporary is None:
                    fd, name = tempfile.mkstemp(prefix=".status-", suffix=".tmp", dir=path.parent)
                    temporary = Path(name)
                    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                        stream.write(payload)
                else:
                    # A prior write may itself have failed; rewrite before replacing.
                    temporary.write_text(payload, encoding="utf-8")
                os.replace(temporary, path)
                return attempts
            except PermissionError as error:
                winerror = getattr(error, "winerror", None)
                if winerror not in (None, 5, 32, 33):
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise
                attempts += 1
                print(f"{pilot.now()} status publication retry {attempts}: {error}",
                      file=sys.stderr, flush=True)
                time.sleep(min(0.1 * 2 ** min(attempts - 1, 4), 1.0, remaining))
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as error:
                print(f"Could not remove owned status temporary {temporary}: {error}",
                      file=sys.stderr, flush=True)


def install_status_writer(module):
    launcher = Path(__file__).resolve()
    provenance = {
        "launcher": str(launcher),
        "launcher_sha256": pilot.digest(launcher),
        "original_runner_sha256": pilot.digest(Path(module.__file__)),
        "change": "Atomic status publication retries access/sharing failures for up to 30 seconds",
        "frozen_assets_modified": False,
    }

    def write_status(state):
        state["updated_utc"] = pilot.now()
        state["status_writer_recovery"] = provenance
        atomic_json_replace(module.OUT / "status.json", state)

    module.write_status = write_status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", choices=("pilot", "seed43"), required=True)
    parser.add_argument("--run", action="store_true", required=True)
    args = parser.parse_args()
    if args.queue == "pilot":
        module = pilot
    else:
        import run_radaz_seed43_replicate as module
    install_status_writer(module)
    module.run()


if __name__ == "__main__":
    main()
