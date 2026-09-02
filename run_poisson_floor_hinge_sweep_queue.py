import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
WORKDIRS = ROOT / "workdirs"
LOG_DIR = WORKDIRS / "poisson_floor_hinge_sweep_logs"
LOG_PATH = LOG_DIR / "queue.log"
CONFIG_DIR = WORKDIRS / "poisson_floor_hinge_sweep_configs"
PYTHON = Path(r"C:\Users\astro\anaconda3\envs\OpenSTL\python.exe")
DATA_ROOT = Path(
    r"C:\Users\astro\research\PEPAPIC\test\results"
    r"\2D_ExB_high_magnet_dt2.5e-9_maxt50e-6_macro5"
    r"\SimVPv2_inputs\global_norm_trainfixed_minmax_margin20_t0_to_t4000_step2.h5"
)

LAMBDAS = [1e-4, 1e-3, 1e-2]
ALPHAS = [1.0, 1.1, 1.2]


def lambda_tag(value):
    mapping = {1e-4: "lam1em4", 1e-3: "lam1em3", 1e-2: "lam1em2"}
    if value in mapping:
        return mapping[value]
    text = f"{value:.0e}".replace("-", "m").replace("+", "")
    return "lam" + text.replace("e", "e")


def alpha_tag(value):
    return f"alpha{int(round(value * 10)):02d}"


def make_case(lam, alpha):
    lam_tag = lambda_tag(lam)
    a_tag = alpha_tag(alpha)
    name = f"floor_hinge_{lam_tag}_{a_tag}"
    ex_name = (
        "pepapic_simvp_gsta_highmag_macro5_subsample2_direct10_"
        f"poisson_floor_hinge_{lam_tag}_floor086_{a_tag}_"
        "trainfixed_disjoint_811_bs2_100ep"
    )
    cfg_name = f"SimVP_gSTA_pepapic_direct_poisson_floor_hinge_{lam_tag}_floor086_{a_tag}.py"
    return {
        "name": name,
        "lambda": lam,
        "alpha": alpha,
        "ex_name": ex_name,
        "config": CONFIG_DIR / cfg_name,
    }


CASES = [make_case(lam, alpha) for lam in LAMBDAS for alpha in ALPHAS]


def log(message):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def config_text(case):
    return f"""method = 'SimVP'

# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gSTA'
hid_S = 64
hid_T = 512
N_T = 8
N_S = 4
simvp_direct_aft_seq = True

# training
lr = 1e-3
batch_size = 2
drop_path = 0
sched = 'onecycle'
epoch = 100

# dataset
pre_seq_length = 10
aft_seq_length = 10
in_shape = None

# physics-informed loss
# L = L_data + lambda * mean(max(0, relR_pred - alpha * true_floor)^2)
pepapic_poisson_loss = 'floor_hinge'
pepapic_poisson_lambda = {case["lambda"]:.12g}
pepapic_poisson_floor = 0.086
pepapic_poisson_floor_alpha = {case["alpha"]:.12g}

# evaluation
metrics = ['mse', 'mae']
"""


def ensure_config(case):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    text = config_text(case)
    if case["config"].exists() and case["config"].read_text(encoding="utf-8") == text:
        return
    case["config"].write_text(text, encoding="utf-8")


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


def train_command(case):
    return [
        str(PYTHON),
        "tools\\train.py",
        "--dataname",
        "pepapic_h5",
        "--config_file",
        str(case["config"]),
        "--data_root",
        str(DATA_ROOT),
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

    log("poisson floor-hinge lambda/alpha sweep queue started")
    for case in CASES:
        ensure_config(case)
        status = saved_status(case)
        log(
            f"{case['name']} lambda={case['lambda']:.12g} alpha={case['alpha']:.12g} "
            f"initial status: {status or 'saved arrays are missing'}"
        )
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
    log("poisson floor-hinge lambda/alpha sweep queue finished")


if __name__ == "__main__":
    main()
