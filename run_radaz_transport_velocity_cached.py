"""Resume the same five-field experiment using lossless precomputed H5 inputs.

Parent code/bundle remain immutable. Runtime amendment and transition hashes
are recorded separately, preserving the scientific checkpoint identity.
"""
from __future__ import annotations
import argparse
import gc
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

import radaz_transport_channels as old
import radaz_transport_velocity as v2
import radaz_transport_velocity_cache as cache
import run_radaz_transport_velocity as parent
from run_radaz_transport_channels import exclusive, OrderedSampler, save_checkpoint, now, epoch_order


def verify_amendment(amend=cache.AMEND,full_hash=True):
    contract,_=parent.verify(v2.OUT)
    manifest=old.read_json(Path(amend)/'bundle.json')
    if manifest['parent_bundle_sha256']!=old.digest(v2.OUT/'bundle.json'):
        raise RuntimeError('Amendment parent mismatch')
    for path,sha in manifest['assets_sha256'].items():
        if cache.stream_digest(path)!=sha:
            raise RuntimeError('Runtime amendment asset changed: '+path)
    cache.verify_cache(amend,full_hash=full_hash)
    return contract,manifest


def validate_checkpoint_runtime(checkpoint,path,amendment,runtime_hash):
    previous=checkpoint.get('runtime_amendment_sha256')
    if previous is not None:
        if previous!=runtime_hash:
            raise RuntimeError('Checkpoint belongs to a different runtime amendment')
    elif cache.stream_digest(path)!=amendment['transition_checkpoint_sha256']:
        raise RuntimeError('Legacy checkpoint is not the audited epoch-boundary checkpoint')


def train_job(folder,arm,seed,device,preloaded,amend=cache.AMEND):
    contract,amendment=verify_amendment(amend,full_hash=False)
    runtime_hash=old.digest(Path(amend)/'bundle.json')
    path=folder/'jobs'/(arm+str(seed)); path.mkdir(parents=True,exist_ok=True)
    identity=dict(job=arm+str(seed),arm=arm,seed=seed,bundle_sha256=old.digest(folder/'bundle.json'))
    model=v2.matched_model(contract,arm,seed).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=contract['training']['lr'],weight_decay=0.)
    train=cache.CachedWindows(contract,'train',preloaded[0])
    val=cache.CachedWindows(contract,'val',preloaded[1])
    sampler=OrderedSampler(len(train),seed)
    loader=DataLoader(train,batch_size=1,sampler=sampler,num_workers=0,pin_memory=device.type=='cuda',
                      generator=torch.Generator().manual_seed(2000003+seed))
    validation=DataLoader(val,batch_size=1,shuffle=False,num_workers=0)
    epochs=contract['training']['epochs']; weight=contract['training']['gamma_weight']
    velocity_weight=contract['training'].get('velocity_weight',1/3)
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=contract['training']['lr'],
        total_steps=len(loader)*epochs,pct_start=contract['training']['pct_start'])
    first=global_step=0
    if (path/'last.pt').exists():
        checkpoint=torch.load(path/'last.pt',map_location=device,weights_only=False)
        if checkpoint['identity']!=identity or checkpoint.get('kind')!='transport_five_field_training':
            raise RuntimeError('Checkpoint belongs to a different experiment')
        validate_checkpoint_runtime(checkpoint,path/'last.pt',amendment,runtime_hash)
        model.load_state_dict(checkpoint['model'],strict=True)
        optimizer.load_state_dict(checkpoint['optimizer']); scheduler.load_state_dict(checkpoint['scheduler'])
        first,global_step=checkpoint['completed_epochs'],checkpoint['global_step']
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        if device.type=='cuda':
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint['cuda_rng']])
        del checkpoint
    for epoch in range(first,epochs):
        model.train(); sampler.epoch=epoch
        start=time.monotonic(); sums={k:0. for k in ('total','field','gamma','velocity')}
        for i,(x,y) in enumerate(loader):
            parts=v2.checked_step(model,optimizer,x.to(device),y.to(device),weight,velocity_weight)
            scheduler.step(); global_step+=1
            for key,value in parts.items():
                sums[key]+=value
            if i==0 or (i+1)%50==0 or i+1==len(loader):
                elapsed=time.monotonic()-start
                old.atomic_json(path/'status.json',dict(**identity,status='training',epoch=epoch+1,
                    epochs=epochs,step=i+1,steps_per_epoch=len(loader),global_step=global_step,loss=parts,
                    mean_step_seconds=elapsed/(i+1),runtime_amendment_sha256=runtime_hash,
                    data_pipeline='precomputed_five_field_h5',updated_utc=now()))
                print(identity['job'],f'epoch {epoch+1}/{epochs} step {i+1}/{len(loader)}',parts,flush=True)
        model.eval(); vsums={k:0. for k in sums}
        with torch.no_grad():
            for x,y in validation:
                parts=v2.loss_parts(model(x.to(device)),y.to(device),weight,velocity_weight)
                for key,value in parts.items():
                    vsums[key]+=float(value)
        save_checkpoint(path/'last.pt',dict(kind='transport_five_field_training',identity=identity,
            model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            completed_epochs=epoch+1,global_step=global_step,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
            model_config=v2.model_config(contract,arm),runtime_amendment_sha256=runtime_hash))
        row=dict(**identity,runtime_amendment_sha256=runtime_hash,
            data_pipeline='precomputed_five_field_h5',completed_epochs=epoch+1,train_loss={k:v/len(loader) for k,v in sums.items()},
            val_loss={k:v/len(validation) for k,v in vsums.items()},updated_utc=now(),elapsed_seconds=time.monotonic()-start)
        with (path/'epochs.jsonl').open('a',encoding='utf-8') as handle:
            import json
            handle.write(json.dumps(row)+'\n')
        paused=(folder/'PAUSE_AFTER_EPOCH').exists()
        old.atomic_json(path/'status.json',dict(**row,status='paused' if paused else 'complete' if epoch+1==epochs else 'epoch_saved'))
        if paused:
            return False
    return True


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value,dict):
        return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def assert_tree_equal(a,b,path='state'):
    if torch.is_tensor(a):
        if not torch.is_tensor(b) or a.dtype!=b.dtype or a.shape!=b.shape:
            raise AssertionError(path+' tensor metadata differs')
        if not torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8)):
            raise AssertionError(path+' tensor bits differ')
    elif isinstance(a,dict):
        if a.keys()!=b.keys():
            raise AssertionError(path+' keys differ')
        for key in a:
            assert_tree_equal(a[key],b[key],path+'/'+str(key))
    elif isinstance(a,(tuple,list)):
        if type(a)!=type(b) or len(a)!=len(b):
            raise AssertionError(path+' sequence differs')
        for i,(x,y) in enumerate(zip(a,b)):
            assert_tree_equal(x,y,path+'/'+str(i))
    elif a!=b:
        raise AssertionError(path+' differs')


def code_hashes():
    return {str(v2.ROOT/name):cache.stream_digest(v2.ROOT/name) for name in
        ('radaz_transport_velocity_cache.py','run_radaz_transport_velocity_cached.py',
         'tests/test_radaz_transport_velocity_cache.py')}


def transition_check(amend,device):
    """Read-only optimizer comparison, guarded by the original execution lock."""
    if device.type=='cuda' and not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError('Use the check CLI: this equality diagnostic requires deterministic GPU kernels')
    amend=Path(amend)
    if (amend/'transition.json').exists():
        raise RuntimeError('Transition already audited; preserve it')
    contract,manifest=cache.verify_cache(amend)
    status=old.read_json(v2.OUT/'jobs/corr_uez42/status.json')
    if status['status']!='paused' or not (v2.OUT/'PAUSE_AFTER_EPOCH').exists():
        raise RuntimeError('The legacy run must finish and pause at an epoch boundary')
    source=v2.OUT/'jobs/corr_uez42/last.pt'
    checkpoint=torch.load(source,map_location='cpu',weights_only=False)
    ncase=contract['splits']['train'][1]-contract['splits']['train'][0]-19
    steps=ncase*len(contract['manifest']['cases'])
    completed=checkpoint['completed_epochs']
    if (completed!=status['completed_epochs'] or checkpoint['global_step']!=completed*steps
        or checkpoint['identity']['bundle_sha256']!=old.digest(v2.OUT/'bundle.json')):
        raise RuntimeError('Checkpoint is not the recorded scientific epoch boundary')
    source_sha=cache.stream_digest(source)
    archive=amend/'transition'/f'seed42_epoch{completed}_original.pt'
    archive.parent.mkdir(parents=True,exist_ok=True)
    if archive.exists() and cache.stream_digest(archive)!=source_sha:
        raise RuntimeError('Existing archive differs')
    if not archive.exists():
        shutil.copy2(source,archive)
    if cache.stream_digest(archive)!=source_sha:
        raise RuntimeError('Archive verification failed')
    indices=epoch_order(steps,42,completed)[:3]
    samples=[]; descriptions=[]
    for index in indices:
        case_i,s=divmod(int(index),ncase); case=contract['manifest']['cases'][case_i]
        raw=v2.load_frames(case,s,s+20,contract)
        row=next(e for e in manifest['entries'] if e['case']==case['case_key'] and e['split']=='train')
        with h5py.File(row['path'],'r') as handle:
            frames=handle['state'][s:s+20]
        legacy=(v2.make_input(raw[:10],case,contract),v2.make_target(raw[10:],case,contract))
        cached=(cache.append_conditions(frames[:10],case,contract),frames[10:])
        if not all(cache.bit_equal(a,b) for a,b in zip(legacy,cached)):
            raise RuntimeError('Transition input/target bits differ')
        samples.append((legacy,cached)); descriptions.append(dict(index=int(index),case=case['case_key'],start=s))
    results=[]; losses=[]
    for route in range(2):
        model=v2.matched_model(contract,'corr_uez',42).to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=contract['training']['lr'],weight_decay=0.)
        scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=contract['training']['lr'],
            total_steps=steps*contract['training']['epochs'],pct_start=contract['training']['pct_start'])
        model.load_state_dict(checkpoint['model'],strict=True)
        optimizer.load_state_dict(cpu_tree(checkpoint['optimizer']))
        scheduler.load_state_dict(cpu_tree(checkpoint['scheduler']))
        torch.set_rng_state(checkpoint['torch_rng'])
        if device.type=='cuda':
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        model.train(); trace=[]
        for pair in samples:
            x,y=pair[route]
            # default_collate copies both routes to the same contiguous batch.
            x=torch.stack([torch.from_numpy(x)]).to(device)
            y=torch.stack([torch.from_numpy(y)]).to(device)
            trace.append(v2.checked_step(model,optimizer,x,y,contract['training']['gamma_weight'],
                contract['training']['velocity_weight']))
            scheduler.step()
        results.append(cpu_tree(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])))
        losses.append(trace)
        del model,optimizer,scheduler,x,y
        gc.collect()
        if device.type=='cuda':
            torch.cuda.empty_cache()
    assert_tree_equal(results[0],results[1]); assert_tree_equal(losses[0],losses[1])
    if cache.stream_digest(source)!=source_sha:
        raise RuntimeError('Original checkpoint changed during audit')
    old.atomic_json(amend/'transition.json',dict(status='passed',created_utc=now(),
        parent_bundle_sha256=old.digest(v2.OUT/'bundle.json'),source_checkpoint=str(source),
        source_checkpoint_sha256=source_sha,archive=str(archive),completed_epochs=completed,
        global_step=checkpoint['global_step'],device=str(device),torch=torch.__version__,
        compared_updates=3,samples=descriptions,losses=losses,
        input_target_bits_equal=True,model_optimizer_scheduler_rng_bits_equal=True,
        original_checkpoint_unchanged=True,code_sha256=code_hashes(),
        deterministic_diagnostic_only=True,production_gpu_settings_unchanged=True,
        gpu_verification_settings=dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))))
    print('TRANSITION PASSED: identical input, target, 3 Adam updates, scheduler and RNG',flush=True)


def seal(amend):
    amend=Path(amend)
    if (amend/'bundle.json').exists():
        raise RuntimeError('Amendment already sealed')
    _,manifest=cache.verify_cache(amend)
    transition=old.read_json(amend/'transition.json')
    tests=old.read_json(amend/'tests.json')
    if (transition['status']!='passed' or tests['status']!='passed'
        or transition['code_sha256']!=code_hashes() or tests['code_sha256']!=code_hashes()):
        raise RuntimeError('Current code requires passing unit and GPU transition checks')
    for p,sha in manifest['code_sha256'].items():
        if cache.stream_digest(p)!=sha:
            raise RuntimeError('Cache builder changed after preparation')
    files=[cache.PLAN/'PROTOCOL.md',amend/'cache_manifest.json',amend/'transition.json',
           amend/'tests.json',Path(transition['archive'])]
    assets=code_hashes(); assets.update({str(p):cache.stream_digest(p) for p in files})
    old.atomic_json(amend/'bundle.json',dict(status='sealed_data_pipeline_amendment',created_utc=now(),
        parent_bundle_sha256=old.digest(v2.OUT/'bundle.json'),assets_sha256=assets,
        transition_checkpoint_sha256=transition['source_checkpoint_sha256'],
        scientific_contract_unchanged=True,training_order=['corr_uez42','corr_uez43']))
    cache.PLAN.mkdir(parents=True,exist_ok=True)
    (cache.PLAN/'PREPARED.sha256').write_text(old.digest(amend/'bundle.json')+'  '+str(amend/'bundle.json')+'\n',encoding='utf-8')
    print('SEALED',old.digest(amend/'bundle.json'),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['build','check','seal','train','status'])
    parser.add_argument('--amend',type=Path,default=cache.AMEND)
    args=parser.parse_args(); torch.set_num_threads(4)
    if args.action=='status':
        for path in [v2.OUT/'runner.json',*sorted((v2.OUT/'jobs').glob('*/status.json'))]:
            if path.exists():
                print(path.name,old.read_json(path))
        return
    if args.action=='build':
        cache.build_cache(args.amend); return
    with exclusive(v2.OUT/'execution.lock'):
        if args.action=='seal':
            seal(args.amend); return
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        device=torch.device('cuda')
        if args.action=='check':
            # Isolated diagnostic process only. The production train branch keeps
            # the legacy GPU settings, which do not guarantee bit reproducibility.
            os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
            torch.backends.cudnn.benchmark=False
            torch.backends.cudnn.deterministic=True
            torch.use_deterministic_algorithms(True)
            transition_check(args.amend,device); return
        contract,amendment=verify_amendment(args.amend)
        if (v2.OUT/'PAUSE_AFTER_EPOCH').exists():
            raise RuntimeError('Pause flag must be explicitly removed before resume')
        runtime_hash=old.digest(args.amend/'bundle.json')
        def record(status,**extra):
            old.atomic_json(v2.OUT/'runner.json',dict(pid=os.getpid(),status=status,updated_utc=now(),
                command='run_radaz_transport_velocity_cached.py train',runtime_amendment_sha256=runtime_hash,**extra))
        record('loading',seeds=[42,43])
        try:
            manifest=old.read_json(args.amend/'cache_manifest.json')
            shared=tuple(cache.load_cache(contract,manifest,s) for s in ('train','val'))
            for seed in (42,43):
                record('training',job='corr_uez'+str(seed))
                if not train_job(v2.OUT,'corr_uez',seed,device,shared,args.amend):
                    record('paused',job='corr_uez'+str(seed)); return
                gc.collect(); torch.cuda.empty_cache()
            record('training_complete')
        except Exception as exc:
            record('failed',error=repr(exc)); raise


if __name__=='__main__':
    main()
