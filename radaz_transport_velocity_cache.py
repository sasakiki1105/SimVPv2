"""Lossless frame cache for the frozen five-field experiment.

Only data preparation changes. Legacy normalization, gamma arithmetic, model,
optimizer, finite checks and epoch order are reused verbatim.
"""
from pathlib import Path
import gc
import hashlib
import time

import h5py
import numpy as np
from torch.utils.data import Dataset

import radaz_transport_channels as old
import radaz_transport_velocity as state
from run_radaz_transport_channels import now

AMEND = state.OUT/'cache_amendment_20260930'
PLAN = state.ROOT/'protocols/radaz_transport_velocity_cache_20260930'


def stream_digest(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def bit_equal(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a.view(np.uint32),b.view(np.uint32))


def append_conditions(frames,case,contract):
    vector=old._condition_values(case,contract['manifest']['condition_channels'],
        normalization=contract['manifest']['condition_normalization'])
    cond=np.broadcast_to(vector[None,:,None,None],(len(frames),2,*frames.shape[-2:]))
    return np.concatenate((frames,cond),axis=1).astype(np.float32,copy=False)


class CachedWindows(Dataset):
    def __init__(self,contract,split,preloaded):
        if split not in ('train','val'):
            raise ValueError('The preparation cache contains TRAIN/VAL only')
        self.contract,self.cases,self.data=contract,contract['manifest']['cases'],preloaded
        self.pre,self.aft=contract['pre'],contract['aft']
        start,stop=contract['splits'][split]
        if len(preloaded)!=len(self.cases) or any(a.shape!=(stop-start,5,260,256)
                                                or a.dtype!=np.float32 for a in preloaded):
            raise ValueError('Cache shape/dtype/case count mismatch')
        self.samples=[(i,s) for i in range(len(self.cases))
                      for s in range(stop-start-self.pre-self.aft+1)]
        self.conditions=[old._condition_values(case,contract['manifest']['condition_channels'],
            normalization=contract['manifest']['condition_normalization']) for case in self.cases]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,index):
        i,s=self.samples[index]
        frames=self.data[i][s:s+self.pre]
        cond=np.broadcast_to(self.conditions[i][None,:,None,None],(self.pre,2,260,256))
        x=np.concatenate((frames,cond),axis=1).astype(np.float32,copy=False)
        y=self.data[i][s+self.pre:s+self.pre+self.aft]
        return x,y


def build_cache(amend=AMEND):
    amend=Path(amend)
    if (amend/'cache_manifest.json').exists():
        raise RuntimeError('Completed cache exists; verify and reuse it')
    contract,bundle=old.verify_prepared(state.OUT)
    parent_hash=old.digest(state.OUT/'bundle.json')
    entries=[]
    amend.mkdir(parents=True,exist_ok=True)
    for split in ('train','val'):
        start,stop=contract['splits'][split]
        for case in contract['manifest']['cases']:
            path=amend/'inputs'/f'{case["case_key"]}_{split}.h5'
            proof=path.with_suffix('.verified.json')
            if path.exists() and proof.exists():
                row=old.read_json(proof)
                if row['parent_bundle_sha256']!=parent_hash or stream_digest(path)!=row['sha256']:
                    raise RuntimeError('Existing cache identity mismatch')
                entries.append(row); continue
            if path.exists():
                raise RuntimeError('Unverified cache exists; investigate before overwrite')
            print('BUILD',case['case_key'],split,flush=True)
            t0=time.monotonic()
            raw=state.load_frames(case,start,stop,contract)
            path.parent.mkdir(parents=True,exist_ok=True)
            tmp=path.with_suffix('.tmp.h5')
            with h5py.File(tmp,'w') as handle:
                handle.attrs.update(parent_bundle_sha256=parent_hash,case=case['case_key'],
                    split=split,start=start,stop=stop,channels='ne,ni,phi,gamma_corr,u_ez')
                dataset=handle.create_dataset('state',shape=(len(raw),5,260,256),dtype='f4')
                for lo in range(0,len(raw),16):
                    dataset[lo:lo+16]=state.make_target(raw[lo:lo+16],case,contract)
                handle.flush()
                # Every stored frame is independently recomputed as a singleton.
                # This also checks independence of chunk/time-window length.
                for index in range(len(raw)):
                    expected=state.make_target(raw[index:index+1],case,contract)
                    actual=dataset[index:index+1]
                    if not bit_equal(actual,expected):
                        raise RuntimeError(f'Frame bits differ: {case["case_key"]}/{split}/{start+index}')
                origins=sorted(set([0,1,7,15,16,max(0,len(raw)//2-10),len(raw)-20]))
                for s in origins:
                    x=append_conditions(dataset[s:s+10],case,contract)
                    y=dataset[s+10:s+20]
                    if not bit_equal(x,state.make_input(raw[s:s+10],case,contract)):
                        raise RuntimeError('Cached window input differs from legacy')
                    if not bit_equal(y,state.make_target(raw[s+10:s+20],case,contract)):
                        raise RuntimeError('Cached window target differs from legacy')
                # Warm repeated CPU data-preparation timing, no GPU activity.
                sample=dataset[:20]; times={}
                for mode in ('legacy','cached'):
                    durations=[]
                    for _ in range(12):
                        t=time.perf_counter()
                        if mode=='legacy':
                            x=state.make_input(raw[:10],case,contract)
                            y=state.make_target(raw[10:20],case,contract)
                        else:
                            x=append_conditions(sample[:10],case,contract); y=sample[10:20]
                        durations.append(time.perf_counter()-t)
                    times[mode]=float(np.median(durations[2:]))
            tmp.replace(path)
            row=dict(path=str(path),sha256=stream_digest(path),size=path.stat().st_size,
                parent_bundle_sha256=parent_hash,case=case['case_key'],split=split,frames=[start,stop],
                shape=[stop-start,5,260,256],singleton_bit_checks=stop-start,
                complete_window_origins=origins,input_target_bits_equal=True,
                prep_median_seconds=times,elapsed_seconds=time.monotonic()-t0)
            old.atomic_json(proof,row); entries.append(row)
            old.atomic_json(amend/'build_progress.json',dict(status='building',entries=entries,updated_utc=now()))
            print('VERIFIED',case['case_key'],split,stop-start,'frames',times,flush=True)
            del raw,sample,x,y,actual,expected; gc.collect()
    old.verify_prepared(state.OUT)
    old.atomic_json(amend/'cache_manifest.json',dict(status='verified',created_utc=now(),
        parent_bundle_sha256=parent_hash,entries=entries,frame_count=sum(e['singleton_bit_checks'] for e in entries),
        bytes=sum(e['size'] for e in entries),test_opened=False,
        code_sha256={str(p):old.digest(p) for p in (Path(__file__),)}))
    print('CACHE COMPLETE',sum(e['singleton_bit_checks'] for e in entries),'frames',flush=True)


def verify_cache(amend=AMEND,full_hash=True):
    contract,_=old.verify_prepared(state.OUT)
    manifest=old.read_json(Path(amend)/'cache_manifest.json')
    if manifest['parent_bundle_sha256']!=old.digest(state.OUT/'bundle.json'):
        raise RuntimeError('Cache belongs to a different parent')
    expected={(c['case_key'],s) for c in contract['manifest']['cases'] for s in ('train','val')}
    if {(e['case'],e['split']) for e in manifest['entries']}!=expected or len(manifest['entries'])!=len(expected):
        raise RuntimeError('Missing or duplicate cache entries')
    for entry in manifest['entries']:
        path=Path(entry['path'])
        if path.stat().st_size!=entry['size'] or (full_hash and stream_digest(path)!=entry['sha256']):
            raise RuntimeError('Cache bytes changed: '+str(path))
        if not entry['input_target_bits_equal'] or entry['frames']!=contract['splits'][entry['split']]:
            raise RuntimeError('Unverified cache definition')
    return contract,manifest


def load_cache(contract,manifest,split):
    data=[]
    for case in contract['manifest']['cases']:
        row=next(e for e in manifest['entries'] if e['case']==case['case_key'] and e['split']==split)
        print('LOAD CACHE',split,case['case_key'],flush=True)
        with h5py.File(row['path'],'r') as handle:
            data.append(handle['state'][()])
    return data
