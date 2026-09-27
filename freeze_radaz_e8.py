"""Freeze the E8 bundle: hashes of code, config, protocol, calibration, source
checkpoints and manifest, plus the six job definitions.  Requires calibration.json
(TRAIN-only lambda) to exist.  Run before the local commit and before any E8 training."""
import json
from datetime import datetime, timezone
from pathlib import Path
from radaz_e8_experiment import (ROOT, PLAN, CONFIG, ARMS, SEEDS, ORDER, EPOCHS, EPOCH_OFFSET, CROSS_FRACTION,
                                 job_name, job_checkpoint, read_json, atomic_json, PLAN_BC)
from radaz_bc_experiment import verify as verify_bc
from run_radaz_paired_pilot import digest

ASSETS = [CONFIG, ROOT / 'openstl/methods/pepapic_spectral_loss_e8.py', ROOT / 'radaz_e8_experiment.py',
          ROOT / 'train_radaz_e8.py', ROOT / 'calibrate_radaz_e8.py', ROOT / 'evaluate_radaz_e8.py',
          ROOT / 'run_radaz_e8.py', ROOT / 'tests/test_radaz_e8.py', PLAN / 'PROTOCOL.md', PLAN / 'calibration.json']


def main():
    bc = verify_bc()
    calibration = read_json(PLAN / 'calibration.json')
    if calibration['parent_bc_bundle_sha256'] != digest(PLAN_BC / 'bundle.json') or calibration['config_sha256'] != digest(CONFIG):
        raise RuntimeError('calibration.json does not match the current parent bundle/config')
    assets = {str(p): digest(p) for p in ASSETS}
    assets[str(bc['manifest'])] = digest(bc['manifest'])
    jobs = {}
    for seed in SEEDS:
        source = bc['jobs']['B' + str(seed)]
        for arm in ARMS:
            key = arm + str(seed)
            jobs[key] = {'arm': arm, 'seed': seed, 'time_resolved': ARMS[arm]['time_resolved'],
                         'lambda_key': ARMS[arm]['lambda_key'], 'ex_name': job_name(arm, seed),
                         'config': str(CONFIG), 'checkpoint': str(job_checkpoint(arm, seed)),
                         'source_checkpoint': source['checkpoint'], 'source_checkpoint_sha256': digest(source['checkpoint']),
                         'source_job': 'B' + str(seed)}
    bundle = {'status': 'frozen_before_training', 'version': 'radaz_e8_20260927',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'parent_bc_bundle_sha256': digest(PLAN_BC / 'bundle.json'),
              'manifest': bc['manifest'], 'samples_per_epoch': bc['samples_per_epoch'],
              'train_sample_map_sha256': bc['train_sample_map_sha256'],
              'epochs': EPOCHS, 'epoch_offset': EPOCH_OFFSET, 'lr': 1e-4, 'sched': 'step (constant for the run)',
              'cross_max_mode': 32, 'cross_mask_kappa': 1e-3, 'cross_fraction': CROSS_FRACTION,
              'lambda': calibration['lambda'], 'calibration_sha256': digest(PLAN / 'calibration.json'),
              'jobs': jobs, 'training_order': ORDER, 'assets_sha256': assets,
              'primary': 'source_test: mean over seeds of median over six conditions of CTRL_minus_XT on time-resolved signed O RMSE; positive favours XT',
              'limitations': ['Already inspected source cases, one PIC realization per condition; exploratory',
                              'Continuation from B terminal weights with a fresh optimizer; both arms share this discontinuity',
                              'Same data order and update count across arms; GPU non-determinism remains (two seeds)',
                              'lambda calibrated on TRAIN windows with the starting models; not tuned on any evaluation split',
                              'No confidence intervals, significance claims, or thresholds',
                              'Local git commit is versioning, not an independent external timestamp; no remote push']}
    atomic_json(PLAN / 'bundle.json', bundle)
    print(json.dumps({'assets': len(assets), 'jobs': list(jobs), 'lambda': bundle['lambda'], 'bundle_sha256': digest(PLAN / 'bundle.json')}))


if __name__ == '__main__':
    main()
