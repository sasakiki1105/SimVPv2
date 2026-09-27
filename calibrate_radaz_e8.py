"""lambda calibration for E8 on TRAIN windows only (no evaluation split is read).

For the first 200 windows of the continuation permutation (epoch 60, seed 42),
compute with the frozen B42 and B43 terminal models: the field MSE (the training
criterion) and the two E8 cross terms.  lambda_arm = CROSS_FRACTION * mean(MSE) /
mean(cross_arm), pooled over both models.  Written to protocols/.../calibration.json
BEFORE the bundle is frozen; the same lambda is used for both seeds.
"""
import argparse
import os
import runpy
from datetime import datetime, timezone
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np
import torch
from radaz_e8_experiment import PLAN, OUT, CONFIG, CROSS_FRACTION, EPOCH_OFFSET, atomic_json, read_json, PLAN_BC
from radaz_bc_experiment import verify as verify_bc
from run_radaz_paired_pilot import digest
from train_radaz_bc import epoch_permutation
from evaluate_radaz_conditioned_factorial import build_model
from openstl.methods.pepapic_spectral_loss_e8 import E8CrossSpectrumLoss

N_WINDOWS = 200


def train_dataset(bc):
    """Same dataset object training uses, via the frozen B training args."""
    from openstl.api import BaseExperiment
    from openstl.utils import create_parser, default_parser, load_config, update_config
    targs = list(bc['jobs']['B42']['training_args'])
    i = targs.index('--ex_name'); targs[i + 1] = 'radaz_e8_calibration_tmp'
    i = targs.index('--res_dir'); targs[i + 1] = str(OUT / 'calibration_tmp')
    args = create_parser().parse_args(targs)
    config = args.__dict__
    update_config(config, load_config(args.config_file), exclude_keys=['method', 'val_batch_size', 'drop_path', 'warmup_epoch'])
    for k, v in default_parser().items():
        if config[k] is None:
            config[k] = v
    exp = BaseExperiment(args)
    return exp.data.train_loader.dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--windows', type=int, default=N_WINDOWS)
    a = parser.parse_args()
    bc = verify_bc()
    device = torch.device('cuda:0')
    ds = train_dataset(bc)
    idx = epoch_permutation(len(ds), 42, EPOCH_OFFSET)[:a.windows]
    cfg = runpy.run_path(str(CONFIG))   # the frozen E8 config: same loss settings the trainer injects
    if int(cfg['pepapic_spectral_max_mode']) != 32 or cfg['pepapic_spectral_loss'] != 'none':
        raise RuntimeError('E8 config changed')
    losses = {}
    for tr in (True, False):
        losses[tr] = E8CrossSpectrumLoss(
            data_root=bc['manifest'], max_mode=int(cfg['pepapic_spectral_max_mode']),
            radial_bands=int(cfg['pepapic_spectral_radial_bands']), radial_min_m=float(cfg['pepapic_spectral_radial_min_m']),
            radial_max_m=float(cfg['pepapic_spectral_radial_max_m']),
            coordinate_system=cfg['pepapic_spectral_coordinate_system'], radial_reduction=cfg['pepapic_spectral_radial_reduction'],
            power_eps_relative=float(cfg['pepapic_spectral_power_eps_relative']),
            cross_mask_kappa=float(cfg['pepapic_spectral_cross_mask_kappa']), time_resolved=tr).to(device)
    per_model = {}
    for seed in (42, 43):
        job = bc['jobs']['B' + str(seed)]
        model, epoch = build_model(job['config'], job['checkpoint'], device)
        if epoch != 59:
            raise RuntimeError('B terminal checkpoint required')
        mse, xt, xa = [], [], []
        with torch.inference_mode():
            for j, i in enumerate(idx):
                sample = ds[int(i)]
                x, y = (sample[0], sample[1]) if isinstance(sample, (tuple, list)) else (sample['x'], sample['y'])
                x = torch.as_tensor(np.asarray(x))[None].to(device)
                y = torch.as_tensor(np.asarray(y))[None].to(device)
                p = model(x)
                mse.append(float(torch.mean((p - y) ** 2)))
                xt.append(float(losses[True](p, y)['cross_spectrum']))
                xa.append(float(losses[False](p, y)['cross_spectrum']))
                if j % 50 == 0:
                    print(seed, j, mse[-1], xt[-1], xa[-1], flush=True)
        per_model[str(seed)] = {'checkpoint_sha256': digest(job['checkpoint']),
                                'mean_field_mse': float(np.mean(mse)), 'mean_cross_time': float(np.mean(xt)),
                                'mean_cross_avg': float(np.mean(xa)),
                                'field_mse': mse, 'cross_time': xt, 'cross_avg': xa}
        del model; torch.cuda.empty_cache()
    pooled = {k: float(np.mean([per_model[s][k] for s in per_model])) for k in ('mean_field_mse', 'mean_cross_time', 'mean_cross_avg')}
    lam = {'lambda_xt': CROSS_FRACTION * pooled['mean_field_mse'] / pooled['mean_cross_time'],
           'lambda_xa': CROSS_FRACTION * pooled['mean_field_mse'] / pooled['mean_cross_avg']}
    out = {'created_utc': datetime.now(timezone.utc).isoformat(), 'cross_fraction': CROSS_FRACTION,
           'windows': int(a.windows), 'window_indices': [int(v) for v in idx], 'permutation': f'epoch {EPOCH_OFFSET}, seed 42',
           'split': 'TRAIN only', 'max_mode': int(cfg['pepapic_spectral_max_mode']), 'cross_mask_kappa': float(cfg['pepapic_spectral_cross_mask_kappa']),
           'parent_bc_bundle_sha256': digest(PLAN_BC / 'bundle.json'), 'config_sha256': digest(CONFIG),
           'per_model': per_model, 'pooled': pooled, 'lambda': lam,
           'note': 'lambda_xt/xa make the cross term 10% of the field MSE at the start of continuation, pooled over B42/B43; same lambda for both seeds; no evaluation split read.'}
    PLAN.mkdir(parents=True, exist_ok=True)
    atomic_json(PLAN / 'calibration.json', out)
    print('lambda', lam, 'pooled', pooled, flush=True)


if __name__ == '__main__':
    main()
