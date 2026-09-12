"""CPU-only cached A audit and source-trained O nulls for new B/C design.

Old A forecasts cover phase 0 only (10 windows); nulls also cover all 181
starts. This never relabels the old forecast audit as all-phase evaluation.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'
import json
import time
from pathlib import Path
import numpy as np
from analyze_radaz_nextstep_baselines import fit_ar, predict_ar, RIDGES
from radaz_bc_metrics import (VERSION, factors, truth_weights, weighted_error, organization_error,
    spectrum_metrics, window_series, condition_summary)
from run_radaz_paired_pilot import ROOT, OUT as PILOT, verify_bundle, digest

OUT = ROOT.parent/'research_results/audits/bc_temporal_preflight_20260912'
CACHE = ROOT/'workdirs/2D_RadAz/radaz_nextstep_baselines_v1'
D43 = ROOT.parent/'research_results/audits/step5_cpu_review_20260910/seed43_signature'
SPLITS = {'source_validation': (400, 600), 'source_test': (600, 800)}
KEYS = ('pn', 'pe', 'cross', 'gamma', 'flux_full')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def main():
    started = time.monotonic()
    bundle = verify_bundle()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'results.json').exists():
        raise RuntimeError('Completed audit exists; refusing overwrite')
    manifest = json.loads(Path(bundle['manifest']).read_text())
    cases = manifest['cases']
    p_result = json.loads((PILOT/'evaluation/results.json').read_text())
    d43_result = json.loads((D43/'results.json').read_text())
    protocol = {'version': VERSION, 'kind': 'exploratory_existing_source_CPU_audit',
        'train_frames': [1200, 1600], 'splits': {'source_validation': [1600, 1800], 'source_test': [1800, 2000]},
        'old_forecast_stride': 20, 'old_forecast_windows': 10, 'null_stride': 1, 'null_windows': 181,
        'ridge_grid': list(RIDGES), 'AR_selection': 'equal-condition mean weighted O MSE on all validation starts; clip O to [-1,1]',
        'mask': 'truth only; |O|>0.05 and Gamma^2>0.001*each-frame band/mode sum',
        'weights': 'raw truth Gamma^2 on retained bins, pooled over windows/leads',
        'models_inspected': ['D42', 'D43', 'P42_secondary'], 'p_values': False,
        'code_sha256': {str(p): digest(p) for p in (Path(__file__), ROOT/'radaz_bc_metrics.py', ROOT/'analyze_radaz_nextstep_baselines.py')},
        'pilot_result_sha256': digest(PILOT/'evaluation/results.json'), 'D43_result_sha256': digest(D43/'results.json'),
        'cache_sha256': {c['case_key']: digest(CACHE/(c['case_key']+'_spectra.npz')) for c in cases}}
    write_json(OUT/'protocol.json', protocol)
    cache, fitted, scores = {}, {}, {str(a): [] for a in RIDGES}
    for c in cases:
        key = c['case_key']
        meta = json.loads((CACHE/(key+'_data.json')).read_text())['fingerprint']
        expected = bundle['source_fields'][key]
        if any(meta[k] != expected[k] for k in ('path', 'size', 'mtime_ns')):
            raise RuntimeError('Cache source fingerprint differs: '+key)
        with np.load(CACHE/(key+'_spectra.npz')) as a:
            cache[key] = {k: a[k].copy() for k in KEYS}
        values = cache[key]
        o = factors(values)['O']
        vx, vy = window_series(o, 400, 600)
        truth = {k: window_series(values[k], 400, 600)[1] for k in KEYS}
        w, _ = truth_weights(truth)
        for a in RIDGES:
            model = fit_ar(o[:400], a)
            fitted[key, a] = model
            scores[str(a)].append(weighted_error(np.clip(predict_ar(model, vx), -1, 1), vy, w))
    scores = {a: float(np.mean(v)) for a, v in scores.items()}
    chosen = min(RIDGES, key=lambda a: scores[str(a)])
    result = {'protocol_sha256': digest(OUT/'protocol.json'), 'selected_O_ridge': chosen,
              'validation_ridge_scores': scores, 'splits': {}}
    for key in cache:
        np.savez_compressed(OUT/(key+'_O_ar.npz'), **fitted[key, chosen])
    for split, (start, stop) in SPLITS.items():
        phase_rows = {label: {} for label in ('copy', 'input_mean', 'AR', 'D42', 'D43', 'P42_secondary', 'oracle_reversed_time')}
        all_rows = {label: {} for label in ('copy', 'input_mean', 'AR')}
        for c in cases:
            key, b = c['case_key'], c['B_mT']*.001
            values = cache[key]
            o = factors(values)['O']
            hx, _ = window_series(o, start, stop)
            truth = {k: window_series(values[k], start, stop)[1] for k in KEYS}
            copy_o = np.repeat(hx[:, -1:], 10, 1)
            mean_o = np.repeat(hx.mean(1, keepdims=True), 10, 1)
            ar_o = np.clip(predict_ar(fitted[key, chosen], hx), -1, 1)
            gh = window_series(values['gamma'], start, stop)[0]
            copy_gamma = np.repeat(gh[:, -1:], 10, 1)
            np.savez_compressed(OUT/f'{split}_{key}_nulls.npz', copy_O=copy_o, input_mean_O=mean_o, AR_O=ar_o,
                                copy_gamma=copy_gamma, **{'truth_'+k:v for k,v in truth.items()})
            for label, pred_o in (('copy', copy_o), ('input_mean', mean_o), ('AR', ar_o)):
                all_rows[label][key] = organization_error(pred_o, truth, copy_o, ar_o)
                phase_rows[label][key] = organization_error(pred_o[::20], {k:v[::20] for k,v in truth.items()}, copy_o[::20], ar_o[::20])
            phase_truth = {k:v[::20] for k,v in truth.items()}
            for label, path in (('D42', PILOT/'evaluation'/f'{split}_{key}_D.npz'),
                                ('D43', D43/f'{split}_{key}.npz'),
                                ('P42_secondary', PILOT/'evaluation'/f'{split}_{key}_P.npz')):
                if label == 'D43' and digest(path) != d43_result['splits'][split]['D43_per_condition'][key]['spectra_sha256']:
                    raise RuntimeError('D43 archive mismatch')
                with np.load(path) as a:
                    t = {k:a['truth_'+k] for k in KEYS}
                    pred = {k:a['pred_'+k] for k in KEYS}
                for k in KEYS:
                    np.testing.assert_allclose(phase_truth[k], t[k], rtol=1e-4, atol=max(abs(t[k]).max()*1e-8, 1e-30))
                # Match all methods to identical archived truth; tiny conversion differences allowed above.
                phase_rows[label][key] = spectrum_metrics(pred, t, copy_o[::20], ar_o[::20], copy_gamma[::20], b)
                phase_rows[label][key]['n0'] = p_result['evaluations'][split]['D']['per_condition'][key]['n0']
            rev = {k:v[:, ::-1] for k,v in phase_truth.items()}
            oracle = spectrum_metrics(rev, phase_truth, copy_o[::20], ar_o[::20], copy_gamma[::20], b)
            np.testing.assert_allclose(oracle['mean_spectrum_O']['O_signed_rmse'], 0, atol=1e-12)
            phase_rows['oracle_reversed_time'][key] = oracle
            print(split, key, 'O errors', {k: round(phase_rows[k][key]['O_time_rmse'], 5) for k in phase_rows}, flush=True)
        result['splits'][split] = {'phase0': phase_rows, 'all_phase_nulls': all_rows,
            'phase0_summary': {k: {metric: condition_summary(v, metric) for metric in ('O_time_rmse', 'O_skill_vs_copy', 'O_skill_vs_AR')} for k,v in phase_rows.items()},
            'all_phase_null_summary': {k: condition_summary(v, 'O_time_rmse') for k,v in all_rows.items()}}
    result['wall_seconds'] = time.monotonic()-started
    result['output_sha256'] = {p.name: digest(p) for p in sorted(OUT.glob('*.npz'))}
    write_json(OUT/'results.json', result)
    lines = ['# Cached A temporal O audit', '', 'Exploratory; model forecasts phase 0 only, null selection all 181 validation starts.',
             'No statistical significance or gate; oracle reversal uses future truth and is not a forecasting baseline.', '',
             '| Split | Method | O RMSE (condition median) | O skill vs copy | O skill vs AR |', '|---|---|---:|---:|---:|']
    for split, data in result['splits'].items():
        for label, row in data['phase0_summary'].items():
            lines.append('| '+split+' | '+label+' | '+' | '.join(f'{row[k]["median"]:.6f}' for k in ('O_time_rmse', 'O_skill_vs_copy', 'O_skill_vs_AR'))+' |')
    lines += ['', 'All-phase neural evaluation remains a separately registered next step.',
              f'O AR ridge: {chosen}; CPU wall seconds: {result["wall_seconds"]:.3f}.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    verify_bundle()
    print('Completed', OUT/'results.json', flush=True)


if __name__ == '__main__':
    main()
