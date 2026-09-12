"""Local exclusive B42 -> C42 -> B43 -> C43 queue and frozen evaluations.

Failure stops the queue. --run resumes terminal-epoch checkpoints with verified
provenance; --pause requests a stop at the next training epoch boundary.
"""
import argparse
from datetime import datetime,timezone
import json
import msvcrt
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from radaz_bc_experiment import ROOT, PLAN, OUT, read_json, atomic_json, verify
from run_radaz_paired_pilot import digest, checkpoint_epoch, process_identity, previous_child_is_alive


def now():
    return datetime.now(timezone.utc).isoformat()


def status(state):
    state['updated_utc'] = now()
    atomic_json(OUT/'status.json',state)


def run_child(command,stage,state,env):
    log_path = OUT/(stage+'.log')
    state.update(stage=stage,stage_started_utc=now(),log=str(log_path))
    with log_path.open('a',encoding='utf-8') as log:
        log.write('\n'+now()+' '+json.dumps(command)+'\n')
        log.flush()
        child = subprocess.Popen([sys.executable,'-u',*command],cwd=ROOT,env=env,
            stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        state.update(child_pid=child.pid,child_create_time=process_identity(child.pid))
        status(state)
        print(stage,'PID',child.pid,flush=True)
        while child.poll() is None:
            time.sleep(15)
            status(state)
        state.update(child_pid=None,child_create_time=None,last_returncode=child.returncode)
        status(state)
        if child.returncode:
            raise RuntimeError(stage+' failed; inspect '+str(log_path))


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'queue.lock').open('a+b') as lock:
        if lock.tell() == 0:
            lock.write(b'0'); lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            bundle = verify()
            # A local commit is reproducible versioning, not an independent external timestamp.
            frozen_git = subprocess.check_output(['git','show','HEAD:protocols/radaz_bc_donly_20260912/bundle.json'],cwd=ROOT)
            if __import__('hashlib').sha256(frozen_git).hexdigest() != digest(PLAN/'bundle.json'):
                raise RuntimeError('Bundle must be committed before training')
            previous = read_json(OUT/'status.json') if (OUT/'status.json').exists() else {}
            if previous_child_is_alive(previous):
                raise RuntimeError('Existing training/evaluation child is alive; no duplication')
            if previous.get('status') == 'complete':
                print('B/C queue already complete',flush=True)
                return
            if previous and previous.get('bundle_sha256') != digest(PLAN/'bundle.json'):
                raise RuntimeError('Existing queue belongs to another bundle')
            if (OUT/'PAUSE_AFTER_EPOCH').exists():
                (OUT/'PAUSE_AFTER_EPOCH').unlink()
            state = dict(previous,status='running',queue_pid=os.getpid(),queue_create_time=process_identity(os.getpid()),
                         queue_started_utc=now(),bundle_sha256=digest(PLAN/'bundle.json'),
                         frozen_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
            state.pop('error',None)
            state.setdefault('completed_training',{})
            state.setdefault('resume_history',[])
            status(state)
            env = dict(os.environ,KMP_DUPLICATE_LIB_OK='TRUE',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
                OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                PYTHONPATH=str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH',''))
            try:
                for key in bundle['training_order']:
                    if (OUT/'PAUSE_AFTER_EPOCH').exists():
                        state.update(status='paused',stage='paused'); status(state); return
                    verify()
                    job = bundle['jobs'][key]
                    path = Path(job['checkpoint'])
                    epoch = checkpoint_epoch(path)
                    if epoch != 59:
                        if shutil.disk_usage(ROOT).free < 8*1024**3:
                            raise RuntimeError('Less than 8 GiB available; preserving checkpoints and stopping')
                        command = ['train_radaz_bc.py','--cell',job['cell'],'--seed',str(job['seed'])]
                        if epoch is not None:
                            command += ['--resume',str(path)]
                            state['resume_history'].append({'job':key,'from_epoch_index':epoch,'utc':now()})
                        elif path.parent.parent.exists() and not previous:
                            raise RuntimeError('Unregistered training output already exists: '+str(path.parent.parent))
                        run_child(command,'train_'+key,state,env)
                        epoch = checkpoint_epoch(path)
                        if epoch != 59:
                            if (OUT/'PAUSE_AFTER_EPOCH').exists() and epoch is not None:
                                state.update(status='paused',stage='paused',paused_job=key,paused_epoch_index=epoch)
                                status(state); return
                            raise RuntimeError('Training stopped without terminal epoch 59: '+key)
                    import torch
                    checkpoint = torch.load(path,map_location='cpu')
                    provenance = checkpoint.get('bc_training_provenance',{})
                    if checkpoint.get('global_step') != 60*bundle['samples_per_epoch'] or provenance.get('job') != key or provenance.get('bundle_sha256') != state['bundle_sha256']:
                        raise RuntimeError('Completed checkpoint has inconsistent step count/provenance')
                    del checkpoint
                    state['completed_training'][key] = {'checkpoint':str(path),'sha256':digest(path),'epoch_index':59,'verified_utc':now()}
                    status(state)
                    if (OUT/'PAUSE_AFTER_EPOCH').exists():
                        state.update(status='paused',stage='paused'); status(state); return
                    run_child(['evaluate_radaz_bc.py','--job',key],'evaluate_'+key,state,env)
                for key in ('A42','A43'):
                    if (OUT/'PAUSE_AFTER_EPOCH').exists():
                        state.update(status='paused',stage='paused'); status(state); return
                    run_child(['evaluate_radaz_bc.py','--job',key],'evaluate_'+key,state,env)
                run_child(['evaluate_radaz_bc.py','--aggregate'],'aggregate',state,env)
                state.update(status='complete',stage='complete',finished_utc=now(),result=str(OUT/'results.json'))
                status(state)
            except BaseException:
                state.update(status='failed',error=traceback.format_exc()); status(state); raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--pause',action='store_true')
    args = parser.parse_args()
    if args.run:
        run()
    elif args.check:
        b = verify()
        print(json.dumps({'verified_assets':len(b['assets_sha256']),'training_order':b['training_order']}))
    elif args.pause:
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/'PAUSE_AFTER_EPOCH').write_text(now(),encoding='utf-8')
        print('Stop requested at the next epoch boundary',flush=True)
    else:
        print((OUT/'status.json').read_text(encoding='utf-8') if (OUT/'status.json').exists() else 'Not started')
