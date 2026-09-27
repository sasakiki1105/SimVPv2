"""E8 continuation training: weights-only start from a frozen B terminal
checkpoint, isolated epoch permutations with offset 60, and an optionally
injected cross-phase objective (XT: time-resolved, XA: time-averaged).

Frozen B/C files are imported, never modified.  The loss module is attached
without submodule registration so E8 checkpoints keep the B/C state_dict layout.
``prepare`` builds the experiment (also used by the technical preflight with a
separate output folder); ``main`` trains.
"""
import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np
import torch
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import Callback
from radaz_e8_experiment import (PLAN, OUT, CONFIG, ARMS, EPOCH_OFFSET, EPOCHS, PLAN_BC,
                                 verify, parse_job, atomic_json)
from run_radaz_paired_pilot import digest
from train_radaz_bc import EpochSampler, epoch_permutation, permutation_digest

EXPECTED_MODEL_KEYS = 126   # frozen B state_dict size (model.* keys)


def now():
    return datetime.now(timezone.utc).isoformat()


def model_state_sha256(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class E8Provenance(Callback):
    def __init__(self, metadata):
        self.metadata = metadata
        self.epoch_started = None

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_started = time.monotonic()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0 or (batch_idx + 1) % 100 == 0:
            loss = outputs.get('loss') if isinstance(outputs, dict) else outputs
            if loss is not None:
                loss = float(loss.detach())
            atomic_json(Path(pl_module.hparams.save_dir) / 'progress.json', {
                'job': self.metadata['job'], 'arm': self.metadata['arm'], 'epoch_index': trainer.current_epoch,
                'batch_completed': batch_idx + 1, 'batches_per_epoch': self.metadata['samples'],
                'global_step': trainer.global_step, 'last_batch_loss': loss,
                'epoch_elapsed_seconds': time.monotonic() - self.epoch_started,
                'updated_utc': now()})
            if batch_idx == 0:
                print('TRAINING ACTIVE', self.metadata['job'], 'epoch', trainer.current_epoch,
                      'global_step', trainer.global_step, 'loss', loss, flush=True)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint['e8_training_provenance'] = {
            **self.metadata, 'saved_epoch': trainer.current_epoch,
            'epoch_permutation_sha256': permutation_digest(epoch_permutation(
                self.metadata['samples'], self.metadata['seed'], trainer.current_epoch + EPOCH_OFFSET))}

    def on_train_epoch_end(self, trainer, pl_module):
        if (OUT / 'PAUSE_AFTER_EPOCH').exists():
            trainer.should_stop = True


def prepare(job_key, resume=None, preflight_dir=None):
    """Build the Lightning experiment for one E8 job exactly as training runs it.

    preflight_dir: when given, the run is written under that folder with the
    suffix ``_preflight`` instead of the registered job folder (technical
    checks only; never a registered checkpoint)."""
    from openstl.api import BaseExperiment
    from openstl.utils import create_parser, default_parser, load_config, update_config
    bundle, bc = verify()
    arm, seed = parse_job(job_key)
    job = bundle['jobs'][job_key]
    source = bc['jobs']['B' + str(seed)]
    ex_name = job['ex_name'] + ('_preflight' if preflight_dir else '')

    training_args = list(source['training_args'])
    def override(flag, value):
        i = training_args.index(flag); training_args[i + 1] = value
    override('--config_file', str(CONFIG))
    override('--ex_name', ex_name)
    override('--epoch', str(EPOCHS))
    if preflight_dir:
        override('--res_dir', str(preflight_dir))
    args = create_parser().parse_args(training_args)
    config = args.__dict__
    update_config(config, load_config(args.config_file), exclude_keys=['method', 'val_batch_size', 'drop_path', 'warmup_epoch'])
    for key, value in default_parser().items():
        if config[key] is None:
            config[key] = value
    if resume:
        if preflight_dir or Path(resume).resolve() != Path(job['checkpoint']).resolve():
            raise RuntimeError('Resume checkpoint not registered for this job')
        args.ckpt_path = resume
    expected = dict(batch_size=1, num_workers=0, epoch=EPOCHS, sched='step', warmup_epoch=0,
                    snapshot_epochs='5', snapshot_epoch_numbering='completed',
                    pepapic_spectral_max_mode=32, pepapic_spectral_loss='none',
                    pepapic_spectral_coordinate_system='integer_power_cross',
                    pepapic_spectral_radial_reduction='local_product')
    for k, v in expected.items():
        if config[k] != v:
            raise RuntimeError(f'Unexpected training setting {k}={config[k]!r}, expected {v!r}')
    if abs(float(config['lr']) - 1e-4) > 1e-12 or int(config['decay_epoch']) < EPOCHS + 1:
        raise RuntimeError('LR schedule is not the frozen constant 1e-4')

    experiment = BaseExperiment(args)
    old = experiment.data.train_loader
    if len(old.dataset) != bc['samples_per_epoch']:
        raise RuntimeError('Training sample count changed')
    map_hash = permutation_digest(np.asarray(old.dataset.samples).reshape(-1))
    if map_hash != bc['train_sample_map_sha256']:
        raise RuntimeError('Training sample identity/order changed')
    folder = Path(preflight_dir) / ex_name if preflight_dir else Path(job['checkpoint']).parent.parent
    sampler = EpochSampler(len(old.dataset), seed,
                           lambda: experiment.trainer.current_epoch + EPOCH_OFFSET, folder / 'epoch_orders.jsonl')
    experiment.data.train_loader = DataLoader(old.dataset, batch_size=1, sampler=sampler, num_workers=0,
                                              pin_memory=old.pin_memory, drop_last=old.drop_last,
                                              generator=torch.Generator().manual_seed(2_000_003 + seed))

    source_sha = digest(job['source_checkpoint'])
    if source_sha != job['source_checkpoint_sha256']:
        raise RuntimeError('Source checkpoint changed')
    if not resume:
        ck = torch.load(job['source_checkpoint'], map_location='cpu')
        prov = ck.get('bc_training_provenance', {})
        if int(ck['epoch']) != 59 or prov.get('job') != 'B' + str(seed) or prov.get('bundle_sha256') != digest(PLAN_BC / 'bundle.json'):
            raise RuntimeError('Source checkpoint is not the frozen B terminal checkpoint')
        state = {k[6:]: v for k, v in ck['state_dict'].items() if k.startswith('model.')}
        if len(state) != len(ck['state_dict']) or len(state) != EXPECTED_MODEL_KEYS:
            raise RuntimeError('Source checkpoint state_dict layout differs from the frozen B layout')
        experiment.method.model.load_state_dict(state, strict=True)
        current = experiment.method.model.state_dict()
        for k, v in state.items():
            if not torch.equal(current[k].cpu(), v):
                raise RuntimeError('Loaded weights differ from source: ' + k)
        del ck, state, current

    lam, time_resolved = None, ARMS[arm]['time_resolved']
    if arm != 'CTRL':
        from openstl.methods.pepapic_spectral_loss_e8 import E8CrossSpectrumLoss
        h = experiment.method.hparams
        module = E8CrossSpectrumLoss(
            data_root=h.data_root, max_mode=int(h.pepapic_spectral_max_mode),
            radial_bands=int(h.pepapic_spectral_radial_bands), radial_min_m=float(h.pepapic_spectral_radial_min_m),
            radial_max_m=float(h.pepapic_spectral_radial_max_m),
            coordinate_system=str(h.pepapic_spectral_coordinate_system),
            radial_reduction=str(h.pepapic_spectral_radial_reduction),
            power_eps_relative=float(h.pepapic_spectral_power_eps_relative),
            cross_mask_kappa=float(h.pepapic_spectral_cross_mask_kappa), time_resolved=time_resolved)
        lam = float(bundle['lambda'][ARMS[arm]['lambda_key']])
        m = experiment.method
        object.__setattr__(m, 'pepapic_spectral_loss_module', module)   # untracked: not in state_dict
        for name in ('amplitude', 'phase', 'cross', 'complex', 'power'):
            setattr(m, f'pepapic_spectral_{name}_weight', 0.0)
        m.pepapic_transport_weight = 0.0
        m.pepapic_spectral_crossspec_weight = lam
        m.pepapic_spectral_enabled = True
        m.pepapic_spectral_loss_mode = 'e8_' + arm.lower()
        if 'pepapic_spectral_loss_module' in dict(m.named_modules()) or len(m.state_dict()) != EXPECTED_MODEL_KEYS:
            raise RuntimeError('Loss module must stay untracked')

    metadata = {'bundle_sha256': digest(PLAN / 'bundle.json'), 'parent_bc_bundle_sha256': digest(PLAN_BC / 'bundle.json'),
                'job': job_key, 'arm': arm, 'seed': seed, 'lambda': lam, 'time_resolved': time_resolved,
                'cross_max_mode': int(config['pepapic_spectral_max_mode']),
                'samples': len(old.dataset), 'train_sample_map_sha256': map_hash,
                'sampler': f'randperm with seed=seed+1000003*(epoch+{EPOCH_OFFSET}), isolated generator',
                'epoch_offset': EPOCH_OFFSET, 'epochs': EPOCHS, 'lr': float(config['lr']), 'sched': config['sched'],
                'source_checkpoint': str(job['source_checkpoint']), 'source_checkpoint_sha256': source_sha,
                'resume_from': resume, 'config_sha256': digest(CONFIG), 'preflight': bool(preflight_dir),
                'initial_state_sha256': model_state_sha256(experiment.method.model)}
    atomic_json(folder / ('training_start_' + ('resume' if resume else 'fresh') + '.json'), metadata)
    experiment.trainer.callbacks.append(E8Provenance(metadata))
    print('E8 TRAIN PROVENANCE', json.dumps(metadata), flush=True)
    return experiment, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--resume')
    selected = parser.parse_args()
    experiment, _ = prepare(selected.job, selected.resume)
    experiment.train()


if __name__ == '__main__':
    main()
