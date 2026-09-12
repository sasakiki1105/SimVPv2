"""Separate training entry point with architecture-independent epoch sampling.

Does not modify any frozen D/P loader/trainer. The B/C pair shares a complete
permutation for each (seed, epoch), independent of network initialization or
the number of earlier epochs in this process. Old A order was not recorded.
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
from torch.utils.data import DataLoader, Sampler
from lightning.pytorch.callbacks import Callback
from radaz_bc_experiment import PLAN, OUT, verify, atomic_json
from run_radaz_paired_pilot import digest


def epoch_permutation(length, seed, epoch):
    generator = torch.Generator().manual_seed(1_000_003*epoch+seed)
    return torch.randperm(length, generator=generator).numpy()


def permutation_digest(indices):
    return hashlib.sha256(np.asarray(indices,dtype='<i8').tobytes()).hexdigest()


class EpochSampler(Sampler):
    def __init__(self, length, seed, epoch_getter, log_path=None):
        self.length, self.seed, self.epoch_getter, self.log_path = length,seed,epoch_getter,log_path

    def __len__(self):
        return self.length

    def __iter__(self):
        epoch = int(self.epoch_getter())
        values = epoch_permutation(self.length,self.seed,epoch)
        if self.log_path is not None:
            with Path(self.log_path).open('a',encoding='utf-8') as f:
                f.write(json.dumps({'epoch_index':epoch,'seed':self.seed,'samples':self.length,
                                    'permutation_sha256':permutation_digest(values)})+'\n')
        return iter(values.tolist())


class Provenance(Callback):
    def __init__(self, metadata):
        self.metadata = metadata
        self.epoch_started = None

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_started = time.monotonic()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0 or (batch_idx+1) % 100 == 0:
            loss = outputs.get('loss') if isinstance(outputs,dict) else outputs
            if loss is not None:
                loss = float(loss.detach())
            atomic_json(Path(pl_module.hparams.save_dir)/'progress.json',{
                'job':self.metadata['job'],'epoch_index':trainer.current_epoch,
                'batch_completed':batch_idx+1,'batches_per_epoch':self.metadata['samples'],
                'global_step':trainer.global_step,'last_batch_loss':loss,
                'epoch_elapsed_seconds':time.monotonic()-self.epoch_started,
                'updated_utc':datetime.now(timezone.utc).isoformat()})
            if batch_idx == 0:
                print('TRAINING ACTIVE',self.metadata['job'],'epoch',trainer.current_epoch,
                      'global_step',trainer.global_step,'loss',loss,flush=True)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint['bc_training_provenance'] = {**self.metadata, 'saved_epoch':trainer.current_epoch,
            'epoch_permutation_sha256':permutation_digest(epoch_permutation(
                self.metadata['samples'], self.metadata['seed'], trainer.current_epoch))}

    def on_train_epoch_end(self, trainer, pl_module):
        if (OUT/'PAUSE_AFTER_EPOCH').exists():
            trainer.should_stop = True


def main():
    from openstl.api import BaseExperiment
    from openstl.utils import create_parser, default_parser, load_config, update_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cell',choices=['B','C'],required=True)
    parser.add_argument('--seed',type=int,choices=[42,43],required=True)
    parser.add_argument('--resume')
    selected = parser.parse_args()
    bundle = verify()
    job_key = selected.cell+str(selected.seed)
    job = bundle['jobs'][job_key]
    args = create_parser().parse_args(job['training_args'])
    config = args.__dict__
    update_config(config,load_config(args.config_file),exclude_keys=['method','val_batch_size','drop_path','warmup_epoch'])
    for key,value in default_parser().items():
        if config[key] is None:
            config[key] = value
    if selected.resume:
        if Path(selected.resume).resolve() != Path(job['checkpoint']).resolve():
            raise RuntimeError('Resume checkpoint not registered for this job')
        args.ckpt_path = selected.resume
    if args.batch_size != 1 or args.num_workers != 0 or args.epoch != 60:
        raise RuntimeError('Unexpected training settings')
    experiment = BaseExperiment(args)
    old = experiment.data.train_loader
    if len(old.dataset) != bundle['samples_per_epoch']:
        raise RuntimeError('Training sample count changed')
    map_hash = permutation_digest(np.asarray(old.dataset.samples).reshape(-1))
    if map_hash != bundle['train_sample_map_sha256']:
        raise RuntimeError('Training sample identity/order changed')
    folder = Path(job['checkpoint']).parent.parent
    sampler = EpochSampler(len(old.dataset),selected.seed,
                           lambda: experiment.trainer.current_epoch,folder/'epoch_orders.jsonl')
    experiment.data.train_loader = DataLoader(old.dataset,batch_size=1,sampler=sampler,
        num_workers=0,pin_memory=old.pin_memory,drop_last=old.drop_last,
        generator=torch.Generator().manual_seed(2_000_003+selected.seed))
    metadata = {'bundle_sha256':digest(PLAN/'bundle.json'),'job':job_key,'seed':selected.seed,
                'samples':len(old.dataset),'train_sample_map_sha256':map_hash,
                'sampler':'randperm with seed=seed+1000003*epoch, isolated generator',
                'resume_from':selected.resume,'initial_state_sha256':hashlib.sha256(b''.join(
                    p.detach().cpu().numpy().tobytes() for p in experiment.method.model.state_dict().values())).hexdigest()}
    if selected.resume:
        checkpoint = torch.load(selected.resume,map_location='cpu')
        previous = checkpoint.get('bc_training_provenance',{})
        if any(previous.get(k) != metadata[k] for k in ('bundle_sha256','job','samples','train_sample_map_sha256')):
            raise RuntimeError('Checkpoint training provenance differs')
        del checkpoint
    atomic_json(folder/('training_start_'+('resume' if selected.resume else 'fresh')+'.json'), metadata)
    experiment.trainer.callbacks.append(Provenance(metadata))
    print('B/C TRAIN PROVENANCE',json.dumps(metadata),flush=True)
    experiment.train()


if __name__ == '__main__':
    main()
