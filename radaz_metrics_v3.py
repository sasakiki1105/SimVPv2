"""Versioned, signed and time-resolved RadAz diagnostics (NumPy reference).

No gate thresholds or automatic continuation decisions live here. Radial
products are local, FFT normalization is forward, Ey uses the PIC-analysis
periodic central difference. Gamma is the ExB flux proxy, not particle current.
"""
from itertools import permutations

import numpy as np
from scipy.stats import rankdata

VERSION = "radaz-local-signed-time-v3.1"


def weighted_median(values, weights):
    values, weights = np.asarray(values).ravel(), np.asarray(weights).ravel()
    if values.shape != weights.shape or not np.all(np.isfinite(values)):
        raise ValueError("Invalid weighted median values")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Invalid weighted median weights")
    keep = weights > 0
    if not keep.any():
        return None
    order = np.argsort(values[keep], kind="stable")
    v, w = values[keep][order], weights[keep][order]
    # Inverse empirical CDF, including the lower value at an exact half mass.
    return float(v[np.searchsorted(np.cumsum(w), 0.5 * w.sum(), side="left")])


def exact_spearman(x, y):
    """All label permutations, ties retained. n<=8 only; invalid != pass."""
    if any(v is None for v in y):
        return {"rho": None, "p_two_sided_exact": None, "status": "undefined"}
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 3 or len(x) > 8:
        raise ValueError("Exact diagnostic requires 3..8 paired conditions")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return {"rho": None, "p_two_sided_exact": None, "status": "undefined"}
    a, b = rankdata(x), rankdata(y)
    a, b = a - a.mean(), b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den == 0:
        return {"rho": None, "p_two_sided_exact": None, "status": "constant"}
    rho = float(a @ b / den)
    values = np.asarray(list(permutations(b))) @ a / den
    return {"rho": rho, "p_two_sided_exact": float(np.mean(np.abs(values) >= abs(rho) - 1e-12)),
            "permutations": len(values), "status": "descriptive_condition_label_permutation",
            "caveat": "Fixed operating conditions, not independent PIC realizations; not a causal test."}


def organization(pred, truth, magnetic_t, o_floor=0.05, weight_floor=1e-3):
    """Mean-spectrum signed O diagnostics with TRUE Gamma^2 weights."""
    if magnetic_t <= 0:
        raise ValueError("Expected positive B magnitude")
    pn_p, pe_p, cross_p = pred
    pn_t, pe_t, cross_t = truth
    amp_p, amp_t = np.sqrt(pn_p * pe_p), np.sqrt(pn_t * pe_t)
    op = np.divide(cross_p.real, amp_p, out=np.zeros_like(amp_p), where=amp_p > 0)
    ot = np.divide(cross_t.real, amp_t, out=np.zeros_like(amp_t), where=amp_t > 0)
    weights = (cross_t.real / magnetic_t) ** 2
    total = weights.sum()
    mask = (np.abs(ot) > o_floor) & (weights > weight_floor * total)
    empty = {"O_ratio_weighted_median": None, "O_signed_rmse": None,
             "O_ratio_abs_error_weighted_median": None, "sign_flip_weight": None,
             "zero_prediction_weight": None, "log_amplitude_rmse": None,
             "mask_bins": int(mask.sum()), "retained_truth_transport_weight": 0.0,
             "valid": False}
    if not total > 0 or not mask.any():
        return empty
    w = weights[mask] / weights[mask].sum()
    ratio = op[mask] / ot[mask]
    logamp = np.log(np.maximum(amp_p[mask], np.finfo(float).tiny)) - np.log(amp_t[mask])
    return {"O_ratio_weighted_median": weighted_median(ratio, w),
            "O_signed_rmse": float(np.sqrt(np.sum(w * (op[mask] - ot[mask]) ** 2))),
            "O_ratio_abs_error_weighted_median": weighted_median(np.abs(ratio - 1), w),
            "sign_flip_weight": float(w[(op[mask] * ot[mask]) < 0].sum()),
            "zero_prediction_weight": float(w[amp_p[mask] == 0].sum()),
            "log_amplitude_rmse": float(np.sqrt(np.sum(w * logamp ** 2))),
            "mask_bins": int(mask.sum()),
            "retained_truth_transport_weight": float(weights[mask].sum() / total), "valid": True}


def band_pool(x_m, bands=4, radial_min=0.0009, radial_max=0.0119):
    x_m = np.asarray(x_m)
    edges = np.linspace(radial_min, radial_max, bands + 1)
    pool = np.zeros((bands, len(x_m)))
    for i in range(bands):
        keep = (x_m >= edges[i]) & (x_m <= edges[i + 1] if i == bands - 1 else x_m < edges[i + 1])
        if not keep.any():
            raise ValueError("Empty radial band")
        pool[i, keep] = 1.0 / keep.sum()
    return pool


def local_observables(physical, pool, dy, magnetic_t, max_mode=64):
    """[sample,time,3,x,y] -> local-product spectra [sample,time,band,n].

    Returned modal Gamma includes conjugate negative modes (factor 2), with
    Nyquist counted once. Full modal sum equals radial mean of -ne*Ey/B.
    The retained 1..max_mode sum and omitted residual are explicitly separate.
    """
    if not np.isfinite(physical).all() or dy <= 0 or magnetic_t <= 0:
        raise ValueError("Invalid physical fields, dy or B")
    width = physical.shape[-1]
    if max_mode < 1 or max_mode > width // 2:
        raise ValueError("Invalid retained mode range")
    ne = np.fft.rfft(physical[:, :, 0], axis=-1, norm="forward")
    phi = np.fft.rfft(physical[:, :, 2], axis=-1, norm="forward")
    symbol = np.sin(2 * np.pi * np.arange(width // 2 + 1) / width) / dy
    symbol[0] = 0
    if width % 2 == 0:
        symbol[-1] = 0
    ey = -1j * symbol * phi
    pn = np.einsum("rh,sthn->strn", pool, abs(ne) ** 2)
    pe = np.einsum("rh,sthn->strn", pool, abs(ey) ** 2)
    cross = np.einsum("rh,sthn->strn", pool, ne * ey.conj())
    factor = np.full(width // 2 + 1, 2.0)
    factor[0] = 1
    if width % 2 == 0:
        factor[-1] = 1
    gamma_all = -factor * cross.real / magnetic_t
    retained = slice(1, max_mode + 1)
    # Complex coefficients are kept at each radial node for forecast skill;
    # averaging fields first could hide radial phase errors.
    return {"pn": pn[..., retained], "pe": pe[..., retained], "cross": cross[..., retained],
            "gamma": gamma_all[..., retained], "flux_full": gamma_all.sum(axis=-1),
            "flux_retained": gamma_all[..., retained].sum(axis=-1),
            "flux_omitted": gamma_all[..., max_mode + 1:].sum(axis=-1),
            "ne_coeff": ne[..., retained], "ey_coeff": ey[..., retained],
            "phi_coeff": phi[..., retained]}


def nrmse(pred, truth):
    denominator = float(np.sum(np.abs(truth) ** 2))
    return float(np.sqrt(np.sum(np.abs(pred - truth) ** 2) / denominator)) if denominator > 0 else None


def skill(pred, truth, copy):
    denominator = float(np.sum(np.abs(copy - truth) ** 2))
    # No epsilon that turns a zero-error baseline into an apparent success.
    return float(1 - np.sum(np.abs(pred - truth) ** 2) / denominator) if denominator > 0 else None


def compare_observables(pred, truth, baseline, magnetic_t, pool):
    pm = tuple(pred[k].mean(axis=(0, 1)) for k in ("pn", "pe", "cross"))
    tm = tuple(truth[k].mean(axis=(0, 1)) for k in ("pn", "pe", "cross"))
    out = organization(pm, tm, magnetic_t)
    out["mean_modal_Gamma_nrmse"] = nrmse(pred["gamma"].mean(axis=(0, 1)), truth["gamma"].mean(axis=(0, 1)))
    for key in ("gamma", "flux_full", "flux_retained"):
        p, t, c = pred[key], truth[key], baseline[key]
        out[key + "_time_nrmse"] = nrmse(p, t)
        out[key + "_skill_vs_copy"] = skill(p, t, c)
        out[key + "_lead_nrmse"] = [nrmse(p[:, j], t[:, j]) for j in range(t.shape[1])]
        out[key + "_lead_skill"] = [skill(p[:, j], t[:, j], c[:, j]) for j in range(t.shape[1])]
        out[key + "_window_nrmse"] = [nrmse(p[j], t[j]) for j in range(t.shape[0])]
    # Apply the same equal-band radial measure without averaging coefficients.
    radial_weight = np.sqrt(pool.mean(axis=0))[None, None, :, None]
    for key in ("ne_coeff", "ey_coeff", "phi_coeff"):
        p, t, c = (d[key] * radial_weight for d in (pred, truth, baseline))
        out[key + "_skill_vs_copy"] = skill(p, t, c)
        out[key + "_lead_skill"] = [skill(p[:, j], t[:, j], c[:, j]) for j in range(t.shape[1])]
    out["truth_omitted_flux_relative_rms"] = nrmse(truth["flux_retained"], truth["flux_full"])
    out["pred_omitted_flux_relative_rms"] = nrmse(pred["flux_retained"], pred["flux_full"])
    log_errors = []
    for p, t in zip(pm[:2], tm[:2]):
        floor = 1e-8 * np.maximum(t.max(axis=-1, keepdims=True), np.finfo(float).tiny)
        log_errors.append((np.log(p + floor) - np.log(t + floor)) ** 2)
    out["local_log_power_mse_n1_32"] = float(np.mean([e[..., :32].mean() for e in log_errors]))
    return out


def field_diagnostics(pred, truth, baseline, low, high):
    # Affine normalized inputs; physical diagnostics exclude padded radial rows.
    out, relative = {}, []
    for i, name in enumerate(("electron_den", "ion_den", "phi")):
        p, t, c = (v[:, :, i] for v in (pred, truth, baseline))
        out[name + "_normalized_mse"] = float(np.mean((p - t) ** 2))
        out[name + "_skill_vs_copy"] = skill(p, t, c)
        out[name + "_lead_skill"] = [skill(p[:, j], t[:, j], c[:, j]) for j in range(t.shape[1])]
        span = high[i] - low[i]
        rel = nrmse(p * span + low[i], t * span + low[i])
        out[name + "_relative_rmse"] = rel
        relative.append(rel)
    out["field_aggregate"] = float(np.mean(relative)) if None not in relative else None
    out["normalized_mse"] = float(np.mean((pred - truth) ** 2))
    out["field_skill_vs_copy"] = skill(pred, truth, baseline)
    return out


def summarize(rows):
    """Equal weight per condition. Preserve worst cases and invalid counts."""
    result = {}
    for key in next(iter(rows.values())):
        values = [r[key] for r in rows.values()]
        if any(isinstance(v, (dict, list, str, bool)) for v in values):
            continue
        valid = [float(v) for v in values if v is not None and np.isfinite(v)]
        result[key] = {"mean": float(np.mean(valid)) if valid else None,
                       "median": float(np.median(valid)) if valid else None,
                       "min": min(valid) if valid else None, "max": max(valid) if valid else None,
                       "valid_conditions": len(valid), "total_conditions": len(values)}
        if "skill" in key:
            result[key]["conditions_beating_copy"] = sum(v > 0 for v in valid)
    return result
