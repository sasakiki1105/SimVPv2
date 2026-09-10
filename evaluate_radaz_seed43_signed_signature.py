"""After seed43/noise completion, extract signed local O on CPU and compare D42.

Never uses CUDA. --check is read-only; --wait queues CPU extraction after the
existing seed43 queue completes. No training or B/C launch is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
from analyze_radaz_step5_cached import rank_signature, memory_and_priority
from radaz_metrics_v3 import organization, band_pool, local_observables, skill
from run_radaz_paired_pilot import ROOT, OUT as PILOT, verify_bundle, digest, checkpoint_path
from run_radaz_queue_resilient import atomic_json_replace

SEED = ROOT/'workdirs/2D_RadAz/radaz_seed43_replicate'
CHECKPOINT = ROOT/'workdirs/2D_RadAz/radaz_pilot_D_GN_A_seed43_60ep/checkpoints/last.ckpt'
OUT = ROOT.parent/'research_results/audits/step5_cpu_review_20260910/seed43_signature'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def now():
    return datetime.now(timezone.utc).isoformat()


def status_write(state):
    state['updated_utc'] = now()
    atomic_json_replace(OUT/'status.json', state)


def readiness():
    seed = read_json(SEED/'status.json')
    return {'ready': seed.get('status') == 'complete' and (SEED/'noise_floor.json').exists(),
            'seed_status': seed.get('status'), 'seed_stage': seed.get('stage'),
            'cpu_threads': 1, 'gpu_used': False, 'checkpoint_exists': CHECKPOINT.exists()}


def extract(state):
    # Heavy imports and all model/data work are strictly behind the completion guard.
    if not readiness()['ready']:
        raise RuntimeError('Seed43 training and noise-floor evaluation must finish first')
    import torch
    from evaluate_radaz_conditioned_factorial import (
        build_model, condition_vector, load_case_frames, predict_direct10, denormalize, VALID_H)
    began, cpu_began = time.monotonic(), time.process_time()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    bundle = verify_bundle()
    pilot = read_json(PILOT/'evaluation/results.json')
    noise = read_json(SEED/'noise_floor.json')
    seed_state = read_json(SEED/'status.json')
    checkpoint_hash = digest(CHECKPOINT)
    assert checkpoint_hash == noise['checkpoint_sha256']['D43'] == seed_state['completed_checkpoint']['sha256']
    assert digest(checkpoint_path('D')) == noise['checkpoint_sha256']['D42'] == pilot['checkpoint_sha256']['D']
    assert digest(PILOT/'bundle.json') == pilot['bundle_sha256']
    manifest = read_json(Path(bundle['manifest']))
    assert len(manifest['cases']) == 6 and all(c['role'] == 'source' for c in manifest['cases'])
    low, high = (np.asarray(manifest['normalization'][k], dtype=float) for k in ('low', 'high'))
    protocol = {'device': 'cpu', 'threads': 1, 'D43_sha256': checkpoint_hash,
        'D42_sha256': pilot['checkpoint_sha256']['D'], 'bundle_sha256': pilot['bundle_sha256'],
        'pilot_results_sha256': digest(PILOT/'evaluation/results.json'),
        'noise_floor_sha256': digest(SEED/'noise_floor.json'), 'code_sha256': digest(Path(__file__)),
        'metrics_sha256': digest(ROOT/'radaz_metrics_v3.py'),
        'helper_sha256': digest(ROOT/'analyze_radaz_step5_cached.py'),
        'signature': 'Spearman(n0, v3 signed O_ratio_weighted_median); O_signed_rmse secondary; no p values',
        'noise_skill_tolerance': 'abs(cpu_skill-recorded_skill) <= 1e-3 * max(1, abs(recorded_skill))',
        'splits': {'source_validation': [1600, 1800], 'source_test': [1800, 2000]}}
    plan_path = OUT/'protocol.json'
    if plan_path.exists() and read_json(plan_path) != protocol:
        raise RuntimeError('Signature protocol changed; preserve partial output and use a new version')
    atomic_json_replace(plan_path, protocol)
    model, epoch = build_model(ROOT/bundle['configs']['D'], CHECKPOINT, torch.device('cpu'))
    if epoch != 59:
        raise RuntimeError('Signature requires the terminal epoch-59 D43 checkpoint')
    result = {'status': 'exploratory_local_signed_D42_D43_comparison', 'protocol': protocol, 'splits': {}}
    for split, (start, stop) in protocol['splits'].items():
        rows = {}
        for case in manifest['cases']:
            key = case['case_key']
            state.update(stage='cpu_inference', current=split+'/'+key)
            status_write(state)
            frames, raw, xm, ym = load_case_frames(case, low, high, start, stop)
            del raw
            cond, _, n0 = condition_vector(case, manifest)
            x = np.empty((10, 10, 5, *frames.shape[-2:]), dtype=np.float32)
            target = np.empty((10, 10, 3, *frames.shape[-2:]), dtype=np.float32)
            for j in range(10):
                x[j, :, :3] = frames[20*j:20*j+10]
                x[j, :, 3:] = cond[None, :, None, None]
                target[j] = frames[20*j+10:20*(j+1)]
            del frames
            h = hashlib.sha256(x.tobytes())
            h.update(target.tobytes())
            assert h.hexdigest() == pilot['input_hashes'][split+'/'+key]
            pool = band_pool(xm[:VALID_H])
            dy, b = float(np.median(np.diff(ym))), case['B_mT']*.001
            truth = local_observables(denormalize(target, low, high), pool, dy, b)
            pred = predict_direct10(model, x, torch.device('cpu'))
            obs = local_observables(denormalize(pred, low, high), pool, dy, b)
            diagnostic = organization(tuple(obs[k].mean((0, 1)) for k in ('pn', 'pe', 'cross')),
                tuple(truth[k].mean((0, 1)) for k in ('pn', 'pe', 'cross')), b)
            row = {'n0': n0, **diagnostic, 'input_target_sha256': h.hexdigest(), 'noise_check': {}}
            with np.load(PILOT/'evaluation'/f'{split}_{key}_D.npz') as d42:
                for name in ('gamma', 'flux_full'):
                    np.testing.assert_allclose(truth[name], d42['truth_'+name], rtol=1e-8,
                        atol=max(float(np.max(abs(truth[name])))*1e-10, 1e-30))
                    actual = skill(obs[name], truth[name], d42['pred_'+name])
                    expected = noise['splits'][split]['per_condition'][key][name]['skill_D43_vs_D42']
                    agrees = abs(actual-expected) <= 1e-3*max(1, abs(expected))
                    row['noise_check'][name] = {'cpu_skill': actual, 'recorded_skill': expected,
                        'absolute_difference': abs(actual-expected), 'within_tolerance': agrees}
                    if not agrees:
                        raise RuntimeError('CPU extraction differs from recorded noise evaluation: '+split+'/'+key+'/'+name)
            spectra = {prefix+k: data[k] for prefix, data in (('pred_', obs), ('truth_', truth))
                       for k in ('pn', 'pe', 'cross', 'gamma', 'flux_full')}
            archive = OUT/f'{split}_{key}.npz'
            np.savez_compressed(archive, **spectra)
            row['spectra_sha256'] = digest(archive)
            atomic_json_replace(OUT/f'{split}_{key}.json', row)
            rows[key] = row
            print(now(), split, key, 'complete', flush=True)
            del x, target, truth, pred, obs, spectra
        d42_rows = pilot['evaluations'][split]['D']['per_condition']
        old, new = rank_signature(d42_rows), rank_signature(rows)
        result['splits'][split] = {'D42': old, 'D43': new, 'D43_per_condition': rows,
            'per_condition_O_ratio_delta_D43_minus_D42': {
                k: rows[k]['O_ratio_weighted_median']-d42_rows[k]['O_ratio_weighted_median'] for k in rows}}
    verify_bundle()
    assert digest(CHECKPOINT) == checkpoint_hash
    result['completed_utc'] = now()
    result['resources'] = {'wall_seconds': time.monotonic()-began, 'cpu_seconds': time.process_time()-cpu_began,
        **memory_and_priority(), 'gpu_used': False, 'threads': 1}
    atomic_json_replace(OUT/'results.json', result)
    lines = ['# D42/D43 local signed signature (exploratory)', '',
        'Six previously inspected source conditions; no p values, gate, or B/C launch.', '',
        '| Split | D42 rho(n0,O ratio) | D43 rho(n0,O ratio) |', '|---|---:|---:|']
    for split, values in result['splits'].items():
        vals = [values[cell]['O_ratio_weighted_median']['rho_n0'] for cell in ('D42', 'D43')]
        lines.append(f'| {split} | {vals[0]} | {vals[1]} |')
    lines += ['', 'Metric is the signed local v3 definition, not the old Gate A metric.',
        'CPU predictions checked against the completed noise-floor evaluation.',
        'D42 has interruption history; a two-run difference does not isolate initialization from all execution variation.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    memo_entry = '\n\n## '+now()+'：⑤ D42/D43のCPU signature比較完了（自動記録）\n\n'
    memo_entry += '\n'.join(lines[2:])+'\n\n出力：`research_results/audits/step5_cpu_review_20260910/seed43_signature/`。'
    memo_entry += '\n同ディレクトリのprotocol.jsonにcheckpoint・noise評価・入力コードのSHAと固定窓を保存。'
    memo_entry += '\nGPU未使用。B/C開始の判断は行っていない。\n'
    with (ROOT.parent/'ICL_reserch_memo.md').open('ab') as stream:
        stream.write(memo_entry.encode('utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--check', action='store_true')
    actions.add_argument('--run', action='store_true')
    actions.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    if args.check:
        print(json.dumps(readiness(), indent=2))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    memory_and_priority(lower=True)
    with (OUT/'queue.lock').open('a+b') as lock:
        if lock.tell() == 0:
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        state = {'status': 'running', 'pid': os.getpid(), 'started_utc': now(),
                 'gpu_used': False, 'threads': 1, 'launcher_sha256': digest(Path(__file__)),
                 'helper_sha256': digest(ROOT/'analyze_radaz_step5_cached.py')}
        try:
            if (OUT/'results.json').exists():
                print('Completed signature result exists; refusing rerun', flush=True)
                return
            while True:
                ready = readiness()
                if ready['ready']:
                    break
                state.update(stage='waiting_for_seed43_noise', dependency=ready)
                status_write(state)
                if not args.wait or ready['seed_status'] in ('failed', 'paused_by_user'):
                    raise RuntimeError('Dependency not complete; refusing inference: '+str(ready))
                time.sleep(30)
            if digest(Path(__file__)) != state['launcher_sha256'] or digest(ROOT/'analyze_radaz_step5_cached.py') != state['helper_sha256']:
                raise RuntimeError('Queued extraction source changed while waiting; restart explicitly after review')
            extract(state)
            state.update(status='complete', stage='complete', finished_utc=now(), result=str(OUT/'results.json'))
            status_write(state)
        except BaseException:
            state.update(status='failed', error=traceback.format_exc())
            status_write(state)
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__':
    main()
