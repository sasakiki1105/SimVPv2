"""Time-resolved, signed organisation endpoints for the separate B/C experiment.

NumPy only. Truth-only masks are shared by every method. No pass thresholds,
seed significance tests, temporal pooling, or model-dependent bin selection.
"""
import numpy as np
from radaz_metrics_v3 import weighted_median, organization

VERSION = 'radaz-bc-time-O-v1'
O_FLOOR, WEIGHT_FLOOR = 0.05, 0.001


def factors(spectra):
    pn, pe, cross = (np.asarray(spectra[k]) for k in ('pn', 'pe', 'cross'))
    if pn.shape != pe.shape or pn.shape != cross.shape:
        raise ValueError('Spectrum shapes differ')
    if not all(np.isfinite(a).all() for a in (pn, pe, cross)) or np.any(pn < 0) or np.any(pe < 0):
        raise ValueError('Nonfinite/negative spectra')
    amp = np.sqrt(pn * pe)
    coherence = np.divide(cross, amp, out=np.zeros_like(cross), where=amp > 0)
    if np.any(np.abs(coherence) > 1 + 1e-6):
        raise ValueError('Cross spectrum violates Cauchy-Schwarz')
    return {'O': coherence.real, 'A': amp, 'r': abs(coherence), 'delta': np.angle(cross),
            'valid': amp > 0, 'phase_valid': abs(cross) > 0}


def truth_weights(truth):
    """Threshold relative to EACH frame's 4-band x 64-mode Gamma^2 sum.

    Retained raw weights are pooled across windows and leads. Weak frames keep
    their physical weight; the number of windows does not alter bin selection.
    """
    f = factors(truth)
    raw = np.asarray(truth['gamma']) ** 2
    scale = raw.sum(axis=(-2, -1), keepdims=True)
    mask = f['valid'] & (abs(f['O']) > O_FLOOR) & (raw > WEIGHT_FLOOR * scale)
    weights = np.where(mask, raw, 0.)
    total = float(raw.sum())
    return weights, {'retained_truth_weight': float(weights.sum() / total) if total else None,
                     'selected_bins': int(mask.sum()), 'total_bins': int(mask.size),
                     'zero_weight_frames': int((weights.sum(axis=(-2, -1)) == 0).sum())}


def weighted_error(pred, truth, weights):
    if pred.shape != truth.shape or weights.shape != truth.shape:
        raise ValueError('Mismatched endpoint shapes')
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError('Nonfinite endpoint')
    total = float(weights.sum())
    return float(np.sum(weights * (pred-truth)**2) / total) if total > 0 else None


def organization_error(pred_o, truth, copy_o=None, ar_o=None):
    t = factors(truth)['O']
    w, coverage = truth_weights(truth)
    mse = weighted_error(pred_o, t, w)
    out = {'O_time_mse': mse, 'O_time_rmse': float(np.sqrt(mse)) if mse is not None else None,
           **coverage}
    out['O_lead_rmse'] = []
    for j in range(t.shape[1]):
        m = weighted_error(pred_o[:, j], t[:, j], w[:, j])
        out['O_lead_rmse'].append(float(np.sqrt(m)) if m is not None else None)
    for label, reference in (('copy', copy_o), ('AR', ar_o)):
        if reference is not None:
            den = weighted_error(reference, t, w)
            out['O_skill_vs_' + label] = 1-mse/den if mse is not None and den is not None and den > 0 else None
    if mse is not None:
        mask = w > 0
        ratio = pred_o[mask]/t[mask]
        out.update(O_time_ratio_weighted_median=weighted_median(ratio, w[mask]),
                   O_time_ratio_absolute_error=weighted_median(abs(ratio-1), w[mask]),
                   O_time_sign_flip_weight=float(w[(pred_o*t) < 0].sum()/w.sum()))
    return out


def transport_error(pred, truth, reference):
    e = pred-truth
    sse, tse, ref = (float(np.sum(a*a)) for a in (e, truth, reference-truth))
    # Exact orthogonal bias/centred decomposition over windows and leads per bin.
    bias = float(pred.shape[0]*pred.shape[1]*np.sum(e.mean((0, 1))**2))
    return {'nrmse': np.sqrt(sse/tse).item() if tse > 0 else None,
            'skill_vs_copy': 1-sse/ref if ref > 0 else None,
            'mean_bias_fraction_of_sse': bias/sse if sse > 0 else None,
            'lead_skill_vs_copy': [1-float(np.sum(e[:, j]**2))/float(np.sum((reference[:, j]-truth[:, j])**2))
                                  if np.sum((reference[:, j]-truth[:, j])**2) > 0 else None
                                  for j in range(truth.shape[1])]}


def spectrum_metrics(pred, truth, copy_o, ar_o, copy_gamma, magnetic_t):
    p, t = factors(pred), factors(truth)
    w, _ = truth_weights(truth)
    out = organization_error(p['O'], truth, copy_o, ar_o)
    out['gamma'] = transport_error(pred['gamma'], truth['gamma'], copy_gamma)
    means = [tuple(s[k].mean((0, 1)) for k in ('pn', 'pe', 'cross')) for s in (pred, truth)]
    out['mean_spectrum_O'] = organization(*means, magnetic_t)
    if w.sum() > 0:
        mask = w > 0
        floor = np.maximum(t['A'].max(axis=-1, keepdims=True)*1e-8, np.finfo(float).tiny)
        out['log_amplitude_time_rmse'] = float(np.sqrt(np.sum(w*(np.log(np.maximum(p['A'], floor))-
                                                                  np.log(np.maximum(t['A'], floor)))**2)/w.sum()))
        out['r_time_rmse'] = float(np.sqrt(np.sum(w*(p['r']-t['r'])**2)/w.sum()))
        phase_w = np.where(p['phase_valid'] & t['phase_valid'], w, 0.)
        out['phase_cosine_error'] = float(np.sum(phase_w*(1-np.cos(p['delta']-t['delta'])))/phase_w.sum()) if phase_w.sum() else None
        out['phase_defined_truth_weight'] = float(phase_w.sum()/w.sum())
        out['zero_predicted_amplitude_weight'] = float(w[~p['valid']].sum()/w.sum())
        # Verify factor reconstruction and retain cancellation terms.
        error = pred['gamma']-truth['gamma']
        terms = [-2/magnetic_t*(p['A']-t['A'])*t['O'],
                 -2/magnetic_t*t['A']*(p['O']-t['O']),
                 -2/magnetic_t*(p['A']-t['A'])*(p['O']-t['O'])]
        np.testing.assert_allclose(sum(terms), error, rtol=1e-6, atol=max(abs(error).max()*1e-10, 1e-30))
        den = float(np.sum(error**2))
        out['amplitude_O_interaction_gram_over_gamma_sse'] = [[float(np.sum(a*b)/den) if den else None for b in terms] for a in terms]
    return out


def window_series(series, start, stop, stride=1):
    starts = np.arange(start, stop-19, stride)
    if start < 0 or stop > len(series) or not len(starts):
        raise ValueError('Invalid 10 -> 10 interval')
    return (np.stack([series[i:i+10] for i in starts]),
            np.stack([series[i+10:i+20] for i in starts]))


def condition_summary(rows, key):
    vals = [r[key] for r in rows.values()]
    valid = [v for v in vals if v is not None and np.isfinite(v)]
    return {'median': float(np.median(valid)) if len(valid) == len(vals) else None,
            'valid_conditions': len(valid), 'conditions': len(vals)}
