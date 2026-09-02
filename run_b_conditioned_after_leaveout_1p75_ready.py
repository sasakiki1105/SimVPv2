import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
WORKDIRS = ROOT / "workdirs"
LOG_DIR = WORKDIRS / "mixed_b_sweep_queue_logs"
LOG_PATH = LOG_DIR / "queue.log"
PYTHON = Path(r"C:\Users\astro\anaconda3\envs\OpenSTL\python.exe")

PREREQ_EX_NAME = (
    "pepapic_simvp_gsta_mixed_b_sweep_stride2_direct10_"
    "leaveout_1p75mT_data_only_trainfixed_disjoint_811_bs2_100ep"
)
EX_NAME = (
    "pepapic_simvp_gsta_mixed_b_sweep_stride2_direct10_"
    "leaveout_1p75mT_b_conditioned_4ch_trainfixed_disjoint_811_bs2_100ep"
)
MANIFEST = WORKDIRS / "mixed_b_sweep_manifests" / "b_sweep_stride2_leaveout_1p75mT_manifest.json"
CONFIG = "configs/custom/pepapic/SimVP_gSTA_pepapic_direct_b_conditioned.py"


def log(message):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def saved_status(ex_name):
    saved = WORKDIRS / ex_name / "saved"
    paths = [saved / "inputs.npy", saved / "preds.npy", saved / "trues.npy"]
    if not all(path.exists() for path in paths):
        return None
    try:
        arrays = [np.load(path, mmap_mode="r") for path in paths]
    except Exception as exc:
        return f"SAVED_ERROR {exc}"
    return (
        f"done: inputs={tuple(arrays[0].shape)}, "
        f"preds={tuple(arrays[1].shape)}, trues={tuple(arrays[2].shape)}"
    )


def ensure_manifest():
    if MANIFEST.exists():
        return
    cmd = [
        str(PYTHON),
        "make_mixed_b_sweep_manifest.py",
        "--stride",
        "2",
        "--leave-out",
        "1p75mT",
        "--output",
        str(MANIFEST),
    ]
    log("b_conditioned wait manifest command: " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if result.returncode != 0:
        raise RuntimeError("manifest creation failed")


def train_command():
    return [
        str(PYTHON),
        "tools\\train.py",
        "--dataname",
        "pepapic_h5",
        "--config_file",
        CONFIG,
        "--data_root",
        str(MANIFEST),
        "--res_dir",
        ".\\workdirs",
        "--ex_name",
        EX_NAME,
        "--method",
        "simvp",
        "--pre_seq_length",
        "10",
        "--aft_seq_length",
        "10",
        "--total_length",
        "20",
        "--epoch",
        "100",
        "--batch_size",
        "2",
        "--val_batch_size",
        "2",
        "--num_workers",
        "0",
        "--gpus",
        "0",
    ]


def main():
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    ensure_manifest()

    log("b-conditioned leaveout_1p75mT wait job started")
    existing = saved_status(EX_NAME)
    if existing is not None:
        log(f"stride2_leaveout_1p75mT_b_conditioned_4ch already done: {existing}")
        return

    while True:
        prereq = saved_status(PREREQ_EX_NAME)
        if prereq is not None:
            log(f"prerequisite ready: {prereq}")
            break
        log("waiting for stride2_leaveout_1p75mT_data_only saved arrays")
        time.sleep(300)

    cmd = train_command()
    log("stride2_leaveout_1p75mT_b_conditioned_4ch train command: " + " ".join(cmd))
    started = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT), check=False)
    elapsed = time.time() - started
    log(f"stride2_leaveout_1p75mT_b_conditioned_4ch train/test exit code: {result.returncode} elapsed_sec={elapsed:.1f}")
    status = saved_status(EX_NAME)
    log(f"stride2_leaveout_1p75mT_b_conditioned_4ch final status: {status or 'saved arrays are still missing'}")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
