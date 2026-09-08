"""Where inside the network is the organisation lost?  No retraining.

The architecture is
    latent, enc1 = enc(x)          enc1 at 260x256, latent at 65x64
    hid          = translator(latent)
    Y            = dec[-1]( upsample(hid) + enc1 )   -> readout

so the skip is an additive full-resolution residual and the translator sees
only the 65x64 latent.  Three probes, all on the existing UPv2 checkpoint.

A. Autoencoder round trip, with and without the skip.  A TRUE field is encoded
   and decoded immediately, with no temporal evolution at all.  If the latent
   alone already loses the density-field organisation, and loses more of it as
   n0 rises, the spatial bottleneck is demonstrated without involving dynamics.

B. Prediction with the skip zeroed, to see how much of the predicted amplitude
   and organisation arrives through the skip rather than the translator.

C. Azimuthal spectrum of the latent itself.  The latent has 64 azimuthal
   points, so physical mode n maps to latent mode n for n <= 32 and aliases
   for n > 32.  Comparing the latent spectrum with the input spectrum shows
   the compression directly.

Caveat for A and B: the readout was trained on (upsampled hid + enc1), so
zeroing enc1 puts it off its training distribution.  The no-skip numbers are a
probe of where information flows, not a calibrated measurement of latent
information content.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import torch

from evaluate_radaz_conditioned_factorial import (
    AFT, MANIFEST, MODEL_H, MODEL_W, PRE, build_model, condition_vector,
)
from evaluate_radaz_physics_loss_v2 import V2, REG_DIR, spectra
from audit_radaz_v2_transport_error_geometry import batch, MAXN
from openstl.methods.pepapic_spectral_loss import PEPAPICSpectralLoss

TEST_LO, TEST_HI = 1800, 2000
O_FLOOR, W_FLOOR = 0.05, 1.0e-3


def org_ratio(pred, true, B):
    """transport-weighted median organisation ratio and amplitude ratio"""
    pn_p, pe_p, cr_p = pred
    pn_t, pe_t, cr_t = true
    a_p, a_t = np.sqrt(pn_p * pe_p), np.sqrt(pn_t * pe_t)
    o_p = np.real(cr_p) / np.maximum(a_p, 1e-300)
    o_t = np.real(cr_t) / np.maximum(a_t, 1e-300)
    g_t = -np.real(cr_t) / B
    w = g_t ** 2 / (g_t ** 2).sum()
    m = (np.abs(o_t) > O_FLOOR) & (w > W_FLOOR)
    if m.sum() == 0:
        return float("nan"), float("nan")
    return (float(np.median(o_p[m] / o_t[m])),
            float(np.median(a_p[m] / a_t[m])))


@torch.inference_mode()
def roundtrip(model, x, skip_on):
    """encode then immediately decode; no translator, no time evolution"""
    B, T = x.shape[:2]
    flat = torch.from_numpy(x[:, :, :3]).to(next(model.parameters()).device)
    flat = flat.reshape(B * T, 3, MODEL_H, MODEL_W)
    latent, enc1 = model.enc(flat)
    y = model.dec(latent, enc1 if skip_on else torch.zeros_like(enc1))
    return y.reshape(B, T, 3, MODEL_H, MODEL_W).float().cpu().numpy(), latent


@torch.inference_mode()
def predict_noskip(model, x):
    dev = next(model.parameters()).device
    xr = torch.from_numpy(x).to(dev)
    B, T, C, H, W = xr.shape
    cond = xr[:, :, 3:].mean(dim=(1, 3, 4)) if model.condition_dim else None
    flat = xr[:, :, :3].reshape(B * T, 3, H, W)
    latent, enc1 = model.enc(flat)
    _, C_, H_, W_ = latent.shape
    hid = model.hid(latent.view(B, T, C_, H_, W_), condition=cond)
    hid = hid.reshape(B * model.out_seq_length, C_, H_, W_)
    skip = model._match_skip(enc1, B, T, model.out_seq_length)
    y = model.dec(hid, torch.zeros_like(skip))
    return y.reshape(B, model.out_seq_length, 3, H, W).float().cpu().numpy()


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    low = np.asarray(manifest["normalization"]["low"], dtype=np.float64)
    high = np.asarray(manifest["normalization"]["high"], dtype=np.float64)
    dev = torch.device("cuda:0")
    lm = PEPAPICSpectralLoss(data_root=str(MANIFEST), max_mode=MAXN, radial_bands=4,
                             radial_min_m=0.09e-2, radial_max_m=1.19e-2,
                             coordinate_system="integer_power_cross").to(dev)
    model, epoch = build_model(V2["config"],
                               V2["workdir"] / "checkpoints" / "best.ckpt", dev)
    print("[model] v2 epoch=%d" % epoch)

    out = {}
    print()
    print("=" * 112)
    print("A/B.  organisation ratio O_hat/O  (amplitude ratio in brackets)")
    print("=" * 112)
    print("%-11s %7s | %-18s %-18s | %-18s %-18s" % (
        "case", "n0", "roundtrip +skip", "roundtrip NO skip",
        "prediction +skip", "prediction NO skip"))
    print("-" * 112)
    for case in manifest["cases"]:
        key = case["case_key"]
        x, y, n0, B = batch(case, manifest, low, high, TEST_LO, TEST_HI)
        ts = spectra(lm, y, dev)

        # A. round trip of the TRUE target frames
        yy = np.concatenate([y, np.zeros_like(y[:, :, :2])], axis=2)
        rt_s, latent = roundtrip(model, yy, True)
        rt_n, _ = roundtrip(model, yy, False)
        o_rs, a_rs = org_ratio(spectra(lm, rt_s, dev), ts, B)
        o_rn, a_rn = org_ratio(spectra(lm, rt_n, dev), ts, B)

        # B. prediction with and without skip
        from evaluate_radaz_conditioned_factorial import predict_direct10
        o_ps, a_ps = org_ratio(spectra(lm, predict_direct10(model, x, dev), dev), ts, B)
        o_pn, a_pn = org_ratio(spectra(lm, predict_noskip(model, x), dev), ts, B)

        # C. latent azimuthal spectrum vs input azimuthal spectrum
        lat = latent.float().cpu().numpy()
        lp = np.abs(np.fft.rfft(lat, axis=-1, norm="forward")) ** 2
        lp = lp.mean(axis=(0, 1, 2))                     # [33]
        inp = y[:, :, 0, :256, :]
        ip = np.abs(np.fft.rfft(inp - inp.mean(axis=-1, keepdims=True),
                                axis=-1, norm="forward")) ** 2
        ip = ip.mean(axis=(0, 1, 2))                     # [129]

        out[key] = {"n0": n0, "role": case["role"],
                    "roundtrip_skip": [o_rs, a_rs], "roundtrip_noskip": [o_rn, a_rn],
                    "prediction_skip": [o_ps, a_ps], "prediction_noskip": [o_pn, a_pn],
                    "latent_spectrum": lp.tolist(),
                    "input_spectrum": ip[:33].tolist()}
        print("%-11s %7.2f | %7.3f (%6.2f)  %7.3f (%6.2f)  | %7.3f (%6.2f)  %7.3f (%6.2f)" % (
            key, n0, o_rs, a_rs, o_rn, a_rn, o_ps, a_ps, o_pn, a_pn))
        del x, y, rt_s, rt_n, latent
        torch.cuda.empty_cache()

    print()
    print("=" * 112)
    print("C.  latent azimuthal spectrum, normalised to its own m=1 value")
    print("    (latent has 64 azimuthal points: physical mode n maps to latent m=n for n<=32)")
    print("=" * 112)
    ms = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]
    print("%-11s %7s " % ("case", "n0") + " ".join("%7s" % ("m=%d" % m) for m in ms))
    print("-" * 112)
    for key, v in out.items():
        lp = np.asarray(v["latent_spectrum"])
        print("%-11s %7.2f " % (key, v["n0"])
              + " ".join("%7.3f" % (lp[m] / max(lp[1], 1e-300)) for m in ms))

    print()
    print("=" * 112)
    print("SUMMARY: does the LATENT alone (no skip) lose organisation with n0?")
    print("=" * 112)
    from scipy.stats import spearmanr
    n0 = np.array([out[k]["n0"] for k in out])
    for lbl, idx in (("roundtrip +skip", "roundtrip_skip"),
                     ("roundtrip NO skip", "roundtrip_noskip"),
                     ("prediction +skip", "prediction_skip"),
                     ("prediction NO skip", "prediction_noskip")):
        o = np.array([out[k][idx][0] for k in out])
        good = np.isfinite(o)
        r, p = spearmanr(n0[good], o[good])
        print("  %-20s median O_hat/O = %6.3f   Spearman vs n0 = %+0.3f (p=%.4f)"
              % (lbl, float(np.median(o[good])), r, p))

    p = REG_DIR / "v2_network_internal_spectra.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print()
    print("[written] " + str(p))


if __name__ == "__main__":
    main()
