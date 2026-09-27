"""Frozen assets and paths for E8 (matched continuation, cross-phase objective).

Three arms continue from the frozen B terminal checkpoints (epoch index 59):
CTRL = field MSE only, XT = field MSE + lambda_xt * time-resolved cross,
XA = field MSE + lambda_xa * time-averaged cross.  No frozen B/C or D/P file is
modified; the B/C bundle is verified as the parent.
"""
import json
from pathlib import Path
from radaz_bc_experiment import (read_json, atomic_json, verify as verify_bc, PLAN as PLAN_BC,
                                 OUT as OUT_BC, AUDIT, SPLITS, PILOT)
from run_radaz_paired_pilot import ROOT, digest

PLAN = ROOT / 'protocols/radaz_e8_20260927'
OUT = ROOT / 'workdirs/2D_RadAz/radaz_e8_20260927'
CONFIG = ROOT / 'configs/custom/pepapic/SimVP_gSTA_radaz_e8_B_10ep.py'
ARMS = {'CTRL': {'time_resolved': None, 'lambda_key': None},
        'XT': {'time_resolved': True, 'lambda_key': 'lambda_xt'},
        'XA': {'time_resolved': False, 'lambda_key': 'lambda_xa'}}
SEEDS = (42, 43)
ORDER = ['CTRL42', 'XT42', 'XA42', 'CTRL43', 'XT43', 'XA43']
EPOCH_OFFSET = 60      # continuation epochs use permutation seeds 60..69, never B/C's 0..59
EPOCHS = 10
TERMINAL_EPOCH_INDEX = EPOCHS - 1
CROSS_FRACTION = 0.10  # lambda calibration target: lambda*cross = 0.10 * field MSE on TRAIN


def job_name(arm, seed):
    return f'radaz_e8_B_{arm}_seed{seed}_10ep_20260927'


def job_checkpoint(arm, seed):
    return ROOT / 'workdirs/2D_RadAz' / job_name(arm, seed) / 'checkpoints/last.ckpt'


def parse_job(key):
    arm, seed = key[:-2], int(key[-2:])
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError('Unknown E8 job ' + key)
    return arm, seed


def verify():
    bundle = read_json(PLAN / 'bundle.json')
    bc = verify_bc()
    if digest(PLAN_BC / 'bundle.json') != bundle['parent_bc_bundle_sha256']:
        raise RuntimeError('Parent B/C bundle changed')
    for path, expected in bundle['assets_sha256'].items():
        if digest(path) != expected:
            raise RuntimeError('Frozen E8 asset changed: ' + path)
    for key, job in bundle['jobs'].items():
        if digest(job['source_checkpoint']) != job['source_checkpoint_sha256']:
            raise RuntimeError('Source B checkpoint changed: ' + key)
    if bundle['training_order'] != ORDER:
        raise RuntimeError('Unexpected E8 execution plan')
    return bundle, bc
