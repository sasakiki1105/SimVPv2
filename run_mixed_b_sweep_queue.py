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
CONFIG = "configs/custom/pepapic/SimVP_gSTA_pepapic_direct_poisson_baseline.py"
CONFIG_B_CONDITIONED = "configs/custom/pepapic/SimVP_gSTA_pepapic_direct_b_conditioned.py"


CASES = [
    {
        "name": "stride2_all_cases_data_only",
        "stride": 2,
        "leave_out": [],
        "manifest": WORKDIRS
        / "mixed_b_sweep_manifests"
        / "b_sweep_stride2_all_cases_manifest.json",
        "ex_name": (
            "pepapic_simvp_gsta_mixed_b_sweep_stride2_direct10_"
            "all_cases_data_only_trainfixed_disjoint_811_bs2_100ep"
        ),
    },
    {
        "name": "stride2_leaveout_1p0mT_data_only",
        "stride": 2,
        "leave_out": ["1p0mT"],
        "manifest": WORKDIRS
        / "mixed_b_sweep_manifests"
        / "b_sweep_stride2_leaveout_1p0mT_manifest.json",
        "ex_name": (
            "pepapic_simvp_gsta_mixed_b_sweep_stride2_direct10_"
            "leaveout_1p0mT_data_only_trainfixed_disjoint_811_bs2_100ep"
        ),
    },
    {
        "name": "stride2_leaveout_1p75mT_data_only",
        "stride": 2,
        "config": CONFIG,
        "leave_out": ["1p75mT"],
        "manifest": WORKDIRS
        / "mixed_b_sweep_manifests"
        / "b_sweep_stride2_leaveout_1p75mT_manifest.json",
        "ex_name": (
            "pepapic_simvp_gsta_mixed_b_sweep_stride2_direct10_"
            "leaveout_1p75mT_data_only_trainfixed_disjoint_811_bs2_100ep"
        ),
    },
    {
        "name": "stride2_leaveout_1p75mT_b_conditioned_4ch",
        "stride": 2,
        "config": CONFIG_B_CONDITIONED,
        "leave_out": ["1p75mT"],
        "manifest": WORKDIRS
        / "mixed_b_sweep_manifests"
        / "b_sweep_stride2_leaveout_1p75mT_manifest.json",
        "ex_name": (
            "pepapic_simvp_gsta_mixed_b_sweep_stride2_direct10_"
            "leaveout_1p75mT_b_conditioned_4ch_trainfixed_disjoint_811_bs2_100ep"
        ),
    },
]


def log(message):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def saved_status(case):
    saved = WORKDIRS / case["ex_name"] / "saved"
    paths = [saved / "inputs.npy", saved / "preds.npy", saved / "trues.npy"]
    if not all(path.exists() for path in paths):
        return None
    try:
        inputs = np.load(paths[0], mmap_mode="r")
        preds = np.load(paths[1], mmap_mode="r")
        trues = np.load(paths[2], mmap_mode="r")
    except Exception as exc:
        return f"SAVED_ERROR {exc}"
    return f"done: inputs={tuple(inputs.shape)}, preds={tuple(preds.shape)}, trues={tuple(trues.shape)}"


def ensure_manifest(case):
    if case["manifest"].exists():
        return
    cmd = [
        str(PYTHON),
        "make_mixed_b_sweep_manifest.py",
        "--stride",
        str(case["stride"]),
        "--output",
        str(case["manifest"]),
    ]
    if case.get("leave_out"):
        cmd.extend(["--leave-out", *case["leave_out"]])
    log(f"{case['name']} manifest command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"manifest creation failed for {case['name']}")


def train_command(case):
    return [
        str(PYTHON),
        "tools\\train.py",
        "--dataname",
        "pepapic_h5",
        "--config_file",
        case.get("config", CONFIG),
        "--data_root",
        str(case["manifest"]),
        "--res_dir",
        ".\\workdirs",
        "--ex_name",
        case["ex_name"],
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
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log("mixed B-sweep queue started")
    for case in CASES:
        ensure_manifest(case)
        status = saved_status(case)
        log(f"{case['name']} initial status: {status or 'saved arrays are missing'}")
        if status is not None:
            continue

        cmd = train_command(case)
        log(f"{case['name']} train command: {' '.join(cmd)}")
        started = time.time()
        result = subprocess.run(cmd, cwd=str(ROOT), check=False)
        elapsed = time.time() - started
        log(f"{case['name']} train/test exit code: {result.returncode} elapsed_sec={elapsed:.1f}")
        status = saved_status(case)
        log(f"{case['name']} final status: {status or 'saved arrays are still missing'}")
        if result.returncode != 0:
            log(f"stopping queue after failure in {case['name']}")
            sys.exit(result.returncode)
    log("mixed B-sweep queue finished")


if __name__ == "__main__":
    main()
