"""Technical preflight for E8 (after freeze, before the queue starts).

For CTRL42, XT42 and XA42 it builds the experiment exactly as train_radaz_e8.prepare
does, but under OUT/preflight with the suffix _preflight (no registered job folder
is created), moves the module to the GPU, and runs ONE forward/backward on the first
TRAIN window of permutation epoch 60.  It reports the loss terms, the share of the
cross term in the total, the gradient norm, peak VRAM and step time, and checks that
the loss module stays out of the state_dict.  No frozen asset is modified.
"""
import math
import os
import time
from datetime import datetime, timezone
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import torch
from radaz_e8_experiment import OUT, atomic_json
from train_radaz_e8 import prepare, EXPECTED_MODEL_KEYS

JOBS = ('CTRL42', 'XT42', 'XA42')


def main():
    device = torch.device('cuda:0')
    report = {'started_utc': datetime.now(timezone.utc).isoformat(), 'jobs': {}}
    for key in JOBS:
        experiment, metadata = prepare(key, preflight_dir=OUT / 'preflight')
        m = experiment.method.to(device)
        m.train()
        x, y = next(iter(experiment.data.train_loader))
        x, y = x.to(device), y.to(device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        started = time.monotonic()
        pred = m(x)
        total, data_loss, _, _, spectral = m._total_loss(pred, y, batch_x=x)
        total.backward()
        torch.cuda.synchronize()
        step_seconds = time.monotonic() - started
        grad_norm = math.sqrt(sum(float(p.grad.norm()) ** 2 for p in m.model.parameters() if p.grad is not None))
        cross = float(spectral['cross_spectrum']) if spectral else None
        row = {'arm': metadata['arm'], 'lambda': metadata['lambda'], 'time_resolved': metadata['time_resolved'],
               'total_loss': float(total), 'field_mse': float(data_loss), 'cross_term': cross,
               'weighted_cross_share_of_total': (metadata['lambda'] * cross / float(total)) if cross is not None else 0.0,
               'grad_norm': grad_norm, 'peak_vram_mb': torch.cuda.max_memory_allocated(device) / 2 ** 20,
               'step_seconds': step_seconds, 'state_dict_keys': len(m.state_dict()),
               'loss_module_tracked': 'pepapic_spectral_loss_module' in dict(m.named_modules()),
               'initial_state_sha256': metadata['initial_state_sha256']}
        print(key, row, flush=True)
        if not torch.isfinite(total) or not math.isfinite(grad_norm) or grad_norm == 0.0:
            raise RuntimeError('Non-finite or zero gradient in preflight: ' + key)
        if row['state_dict_keys'] != EXPECTED_MODEL_KEYS or row['loss_module_tracked']:
            raise RuntimeError('Checkpoint layout would differ from B/C: ' + key)
        if metadata['arm'] != 'CTRL' and not (cross > 0.0):
            raise RuntimeError('Cross term inactive: ' + key)
        report['jobs'][key] = row
        del experiment, m, x, y, pred, total, data_loss, spectral
        torch.cuda.empty_cache()
    same_start = len({r['initial_state_sha256'] for r in report['jobs'].values()}) == 1
    report['all_arms_start_from_identical_weights'] = same_start
    if not same_start:
        raise RuntimeError('Arms do not start from identical weights')
    report['completed_utc'] = datetime.now(timezone.utc).isoformat()
    atomic_json(OUT / 'preflight' / 'preflight.json', report)
    print('PREFLIGHT OK', flush=True)


if __name__ == '__main__':
    main()
