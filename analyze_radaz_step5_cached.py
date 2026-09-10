"""CPU-only step-5 review using saved spectra; no torch, model, or raw PIC data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
from scipy.stats import rankdata
from radaz_metrics_v3 import organization

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'workdirs/2D_RadAz/radaz_factorial_reaudit_20260909'
PILOT = ROOT / 'workdirs/2D_RadAz/radaz_paired_pilot_v2'
OUT = ROOT.parent / 'research_results/audits/step5_cpu_review_20260910'
PLAN = ROOT / 'audit_protocols/step5_cached_review_20260910.md'
CELLS = ('U-D', 'C-D', 'U-P', 'C-P', 'U-Pv2')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def memory_and_priority(lower=False):
    if os.name != 'nt':
        return {'peak_working_set_mib': None, 'priority': 'platform default'}
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.GetPriorityClass.argtypes = [wintypes.HANDLE]
    process = kernel.GetCurrentProcess()
    if lower and not kernel.SetPriorityClass(process, 0x4000):
        raise ctypes.WinError(ctypes.get_last_error())
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
            (key, ctypes.c_size_t) for key in ('peak_ws', 'ws', 'peak_paged', 'paged',
                'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile')]
    info = Counters()
    info.cb = ctypes.sizeof(info)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(process, ctypes.byref(info), info.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {'peak_working_set_mib': info.peak_ws / 1024**2,
            'priority_class': kernel.GetPriorityClass(process)}


def magnetic(key):
    return float(key.rsplit('_B', 1)[1]) * .001


def score(p, t, baseline):
    error = float(np.sum((p-t)**2))
    scale = float(np.sum(t**2))
    base = float(np.sum((baseline-t)**2))
    return {'nrmse': float(np.sqrt(error/scale)) if scale > 0 else None,
            'skill': 1-error/base if base > 0 else None}


def factors(pn, pe, cross):
    amplitude = np.sqrt(pn * pe)
    coherence = np.divide(abs(cross), amplitude, out=np.zeros_like(amplitude), where=amplitude > 0)
    if np.max(coherence) > 1 + 1e-8:
        raise ValueError('Cross spectrum violates Cauchy-Schwarz')
    return amplitude, coherence, np.angle(cross)


def factor_review(pred, truth, b):
    pa, pr, pd = factors(*pred)
    ta, tr, td = factors(*truth)
    truth_gamma = -2*truth[2].real/b
    pred_gamma = -2*pred[2].real/b
    norm = float(np.sum(truth_gamma**2))
    rows = {}
    for switches in itertools.product((False, True), repeat=3):
        values = [p if use else t for use, p, t in zip(switches, (pa, pr, pd), (ta, tr, td))]
        g = -2*values[0]*values[1]*np.cos(values[2])/b
        label = ''.join(letter if use else '.' for use, letter in zip(switches, 'ArD'))
        rows[label] = float(np.sqrt(np.sum((g-truth_gamma)**2)/norm)) if norm > 0 else None
        if all(switches) or not any(switches):
            reference = pred_gamma if all(switches) else truth_gamma
            np.testing.assert_allclose(g, reference, rtol=1e-10,
                atol=max(float(np.max(abs(reference)))*1e-12, 1e-30))
    return {'counterfactual_nrmse': rows,
            'amplitude_ratio': float(pa.sum()/ta.sum()) if ta.sum() > 0 else None,
            'coherence_pred_mean': float(pr.mean()), 'coherence_truth_mean': float(tr.mean()),
            'undefined_pred_phase_bins': int(np.count_nonzero(pred[2] == 0)),
            'undefined_truth_phase_bins': int(np.count_nonzero(truth[2] == 0)),
            'total_bins': int(pa.size)}


def rank_signature(rows):
    keys = sorted(rows)
    out = {'conditions': keys, 'n': len(keys), 'p_values_computed': False}
    for metric in ('O_ratio_weighted_median', 'O_signed_rmse'):
        values = [rows[k][metric] for k in keys]
        rho = None
        if all(v is not None and np.isfinite(v) for v in values):
            x = rankdata([rows[k]['n0'] for k in keys])
            y = rankdata(values)
            x, y = x-x.mean(), y-y.mean()
            den = np.linalg.norm(x)*np.linalg.norm(y)
            if den > 0:
                rho = float(x@y/den)
        out[metric] = {'rho_n0': rho, 'values': dict(zip(keys, values))}
    return out


def gram_check(data, b, recorded):
    pa, _, _ = factors(data['pred_pn'], data['pred_pe'], data['pred_cross'])
    ta, _, _ = factors(data['truth_pn'], data['truth_pe'], data['truth_cross'])
    po = np.divide(data['pred_cross'].real, pa, out=np.zeros_like(pa), where=pa > 0)
    to = np.divide(data['truth_cross'].real, ta, out=np.zeros_like(ta), where=ta > 0)
    terms = [-2/b*(pa-ta)*to, -2/b*ta*(po-to), -2/b*(pa-ta)*(po-to)]
    error = data['pred_gamma']-data['truth_gamma']
    np.testing.assert_allclose(sum(terms), error, rtol=1e-6,
        atol=max(float(np.max(abs(error)))*1e-10, 1e-30))
    den = float(np.sum(error**2))
    gram = np.array([[np.sum(a*c)/den for c in terms] for a in terms]) if den else None
    if gram is not None:
        np.testing.assert_allclose(gram, recorded, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(gram.sum(), 1, rtol=1e-8, atol=1e-10)


def med(values):
    values = [x for x in values if x is not None and np.isfinite(x)]
    return float(np.median(values)) if values else None


def summarize(rows):
    result = {}
    for scope in ('instant', 'ensemble'):
        result[scope] = {
            'amplitude_ratio_median': med([r[scope]['amplitude_ratio'] for r in rows]),
            'counterfactual_medians': {label: med([r[scope]['counterfactual_nrmse'][label] for r in rows])
                for label in rows[0][scope]['counterfactual_nrmse']}}
    result['n'] = len(rows)
    return result


def analyze_record(data, record, b):
    answer = {}
    for field in ('gamma', 'flux_full'):
        actual = score(data['pred_'+field], data['truth_'+field], data['copy_'+field])
        for ours, saved in (('nrmse', field+'_time_nrmse'), ('skill', field+'_skill_vs_copy')):
            np.testing.assert_allclose(actual[ours], record['raw'][saved], rtol=1e-8, atol=1e-10)
        answer[field] = actual
    gram_check(data, b, record['raw']['residual_geometry']['amplitude_organization_interaction_gram_over_total_sse'])
    pred = tuple(data['pred_'+k] for k in ('pn', 'pe', 'cross'))
    truth = tuple(data['truth_'+k] for k in ('pn', 'pe', 'cross'))
    answer['instant'] = factor_review(pred, truth, b)
    answer['ensemble'] = factor_review(tuple(v.mean((0, 1)) for v in pred),
                                      tuple(v.mean((0, 1)) for v in truth), b)
    if record['calibration_factors'] is not None:
        f = record['calibration_factors']
        multiplier = f['ne']*f['ey']
        calibrated = tuple(v*s for v, s in zip(pred, (f['ne']**2, f['ey']**2, multiplier)))
        answer['calibrated'] = {'instant': factor_review(calibrated, truth, b),
            'ensemble': factor_review(tuple(v.mean((0, 1)) for v in calibrated),
                tuple(v.mean((0, 1)) for v in truth), b)}
        for field in ('gamma', 'flux_full'):
            s = score(data['pred_'+field]*multiplier, data['truth_'+field], data['copy_'+field])
            np.testing.assert_allclose(s['skill'], record['calibrated'][field+'_skill_vs_copy'], rtol=1e-8, atol=1e-10)
            answer['calibrated'][field] = s
    answer['n0'] = record['n0']
    for key in ('O_ratio_weighted_median', 'O_signed_rmse', 'log_amplitude_rmse'):
        answer[key] = record['raw'][key]
    return answer


def main():
    began, cpu_began = time.monotonic(), time.process_time()
    memory_and_priority(lower=True)
    if (OUT/'results.json').exists():
        raise RuntimeError('Completed review exists; refusing overwrite')
    OUT.mkdir(parents=True, exist_ok=True)
    r = read_json(SOURCE/'results.json')
    proto = ROOT/'audit_protocols/factorial_reaudit_20260909.json'
    if digest(proto) != r['protocol_sha256'] or read_json(SOURCE/'status.json')['status'] != 'complete':
        raise RuntimeError('Source completion/protocol mismatch')
    input_hashes = {str(p.relative_to(ROOT)): digest(p) for p in (
        Path(__file__), PLAN, proto, SOURCE/'results.json', ROOT/'radaz_metrics_v3.py',
        PILOT/'evaluation/results.json')}
    manifest = {'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'exploratory_cached_followup',
        'input_sha256': input_hashes, 'gpu_used': False, 'numerical_threads': 1,
        'source_protocol_sha256': r['protocol_sha256']}
    write_json(OUT/'input_manifest.json', manifest)
    result = {'status': 'exploratory_cached_followup', 'by_split': {}, 'policy_comparisons': {},
        'factorial_contrasts': r['contrasts'], 'existing_calibration': r['calibration'],
        'pilot_signatures': {}, 'checks': {'records': 0, 'calibrated_records': 0},
        'limitations': ['Previously inspected data and metrics; no confirmatory claims or p values',
            'Local signed integer-mode definition differs from legacy q-grid section 7',
            'Single-factor counterfactual errors are not additive error percentages',
            'Signed O signature differs from the old sign-insensitive Gate A',
            'D43 and seed stability remain pending; no B/C gate or launch']}
    window_hashes = {}
    for split, variants in r['evaluations'].items():
        result['by_split'][split] = {}
        for variant, entry in variants.items():
            rows = {}
            for key, record in entry['per_condition'].items():
                if record['epoch_index'] != 59 or record['protocol_sha256'] != r['protocol_sha256']:
                    raise RuntimeError('Record provenance mismatch')
                path = SOURCE/(variant+'_'+split+'_'+key+'.npz')
                actual_hash = digest(path)
                if actual_hash != record['spectra_sha256']:
                    raise RuntimeError('Spectra hash mismatch')
                input_hashes[str(path.relative_to(ROOT))] = actual_hash
                signature = split+'/'+key
                assert window_hashes.setdefault(signature, record['input_target_sha256']) == record['input_target_sha256']
                with np.load(path) as z:
                    data = {k: z[k] for k in z.files}
                assert all(np.isfinite(v).all() for v in data.values())
                rows[key] = analyze_record(data, record, magnetic(key))
                result['checks']['records'] += 1
                result['checks']['calibrated_records'] += int('calibrated' in rows[key])
                del data
            summary = summarize(list(rows.values()))
            if not split.startswith('source_validation'):
                summary['calibrated'] = summarize([v['calibrated'] for v in rows.values()])
                improvements = [v['calibrated']['gamma']['skill']-v['gamma']['skill'] for v in rows.values()]
                summary['calibration_delta_gamma_skill_median'] = med(improvements)
                summary['calibration_gamma_improved_conditions'] = sum(x > 0 for x in improvements)
            result['by_split'][split][variant] = {'per_condition': rows, 'summary': summary}
            if split.startswith('source_'):
                result['by_split'][split][variant]['signed_signature'] = rank_signature(rows)
            print(split, variant, 'complete', flush=True)
        comparison = {}
        for cell in CELLS:
            inp = result['by_split'][split][cell+'_input']['per_condition']
            run = result['by_split'][split][cell+'_running']['per_condition']
            changes = {k: {'delta_gamma_nrmse': inp[k]['gamma']['nrmse']-run[k]['gamma']['nrmse'],
                'delta_gamma_skill': inp[k]['gamma']['skill']-run[k]['gamma']['skill'],
                'delta_log_amplitude_rmse': inp[k]['log_amplitude_rmse']-run[k]['log_amplitude_rmse']}
                for k in inp}
            comparison[cell] = {'per_condition': changes, 'n': len(changes),
                'gamma_improved_conditions': sum(v['delta_gamma_nrmse'] < 0 for v in changes.values()),
                'log_amplitude_improved_conditions': sum(v['delta_log_amplitude_rmse'] < 0 for v in changes.values()),
                'median_delta_gamma_nrmse': med([v['delta_gamma_nrmse'] for v in changes.values()])}
        result['policy_comparisons'][split] = comparison
    pilot = read_json(PILOT/'evaluation/results.json')
    for split, cells in pilot['evaluations'].items():
        result['pilot_signatures'][split] = {}
        for cell in ('D', 'P'):
            rows = {}
            for key, record in cells[cell]['per_condition'].items():
                path = PILOT/'evaluation'/f'{split}_{key}_{cell}.npz'
                input_hashes[str(path.relative_to(ROOT))] = digest(path)
                with np.load(path) as z:
                    means = {prefix: tuple(z[prefix+k].mean((0, 1)) for k in ('pn', 'pe', 'cross'))
                             for prefix in ('pred_', 'truth_')}
                diagnostic = organization(means['pred_'], means['truth_'], magnetic(key))
                for metric in ('O_ratio_weighted_median', 'O_signed_rmse'):
                    np.testing.assert_allclose(diagnostic[metric], record[metric], rtol=1e-8, atol=1e-10)
                rows[key] = {'n0': record['n0'], **diagnostic}
            result['pilot_signatures'][split][cell+'42'] = rank_signature(rows)
    assert result['checks'] == {'records': 160, 'calibrated_records': 100}
    assert 'torch' not in sys.modules
    result['resources'] = {'wall_seconds': time.monotonic()-began, 'cpu_seconds': time.process_time()-cpu_began,
        **memory_and_priority(), 'gpu_used': False, 'numerical_threads': 1,
        'torch_imported': False, 'model_inference': False}
    result['completed_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(OUT/'input_manifest.json', manifest)
    write_json(OUT/'results.json', result)
    print(json.dumps(result['resources']), flush=True)


def self_check():
    shape = (2, 3, 4, 5)
    truth = (np.ones(shape), np.ones(shape)*4, np.full(shape, 1.2+.4j))
    identical = factor_review(truth, truth, .02)
    assert max(identical['counterfactual_nrmse'].values()) < 1e-12
    doubled = tuple(v*4 for v in truth)
    changed = factor_review(doubled, truth, .02)
    np.testing.assert_allclose(changed['counterfactual_nrmse']['A..'], 3)
    np.testing.assert_allclose(changed['counterfactual_nrmse']['ArD'], 3)
    phase_only = (truth[0], truth[1], -truth[2])
    changed = factor_review(phase_only, truth, .02)
    np.testing.assert_allclose(changed['counterfactual_nrmse']['..D'], 2)
    zero = factor_review((truth[0], truth[1], np.zeros(shape, complex)), truth, .02)
    np.testing.assert_allclose(zero['counterfactual_nrmse']['ArD'], 1)
    assert zero['undefined_pred_phase_bins'] == np.prod(shape)
    rows = {str(k): {'n0': k, 'O_ratio_weighted_median': k, 'O_signed_rmse': -k} for k in range(6)}
    np.testing.assert_allclose(rank_signature(rows)['O_ratio_weighted_median']['rho_n0'], 1)
    np.testing.assert_allclose(rank_signature(rows)['O_signed_rmse']['rho_n0'], -1)
    print('PASS: identity, amplitude-only scale, phase reversal, zero-cross degeneracy, signed rank direction')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.run:
        main()
    else:
        parser.error('Choose --self-check or --run')
