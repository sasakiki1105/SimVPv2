#!/usr/bin/env python3
"""Test cross-trajectory predictability and conditionally fit a G8 amplitude closure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import analyze_radaz_g8_mode_selective_oracle as oracle
import evaluate_radaz_g8_history_envelope_fusion as fusion
from radaz_electric_history_hidden_band_rom import HiddenBandEnvelopeBundle, apply_hidden_amplitude_to_carrier


ROOT = Path(__file__).resolve().parent
TINY = np.finfo(np.float64).tiny


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha", type=float, default=10.0)
    return p.parse_args()


def features(payload: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g8 = np.maximum(np.abs(payload["g8_coeff"]), TINY)
    rom = np.maximum(payload["rom_amplitude"], TINY)
    truth = np.maximum(np.abs(payload["truth_coeff"]), TINY)
    lg = np.log(g8); lr = np.log(rom)
    dlg = np.concatenate((np.zeros_like(lg[:1]), np.diff(lg, axis=0)), axis=0)
    dlr = np.concatenate((np.zeros_like(lr[:1]), np.diff(lr, axis=0)), axis=0)
    visible = np.abs(payload["visible_coeff"])
    visible_power = np.log1p(np.sum(visible * visible, axis=(1, 3)))  # time, radial
    nt, nf, nr, nm = g8.shape
    field = np.eye(nf)[np.arange(nf)][None, :, None, None, :]
    radial = np.eye(nr)[np.arange(nr)][None, None, :, None, :]
    mode = np.eye(nm)[np.arange(nm)][None, None, None, :, :]
    controls = payload["controls"][:, None, None, None, :]
    vp = visible_power[:, None, :, None, None]
    broadcast = lambda x, d: np.broadcast_to(x, (nt, nf, nr, nm, d))
    x = np.concatenate((
        lg[..., None], lr[..., None], (lr-lg)[..., None], dlg[..., None], dlr[..., None],
        broadcast(vp, 1), broadcast(controls, controls.shape[-1]),
        broadcast(field, nf), broadcast(radial, nr), broadcast(mode, nm),
    ), axis=-1).reshape(-1, 5 + 1 + controls.shape[-1] + nf + nr + nm)
    y = np.clip((np.log(truth) - lg).reshape(-1), -3.0, 3.0)
    weight = truth.reshape(-1) ** 2
    weight /= max(float(np.mean(weight)), TINY)
    weight = np.clip(weight, 0.01, 100.0)
    return x.astype(np.float32), y.astype(np.float32), weight.astype(np.float32)


def corrected(payload: dict, predicted_delta: np.ndarray) -> np.ndarray:
    g8 = payload["g8_coeff"]
    delta = predicted_delta.reshape(g8.shape)
    amplitude = np.abs(g8) * np.exp(np.clip(delta, -2.0, 2.0))
    return apply_hidden_amplitude_to_carrier(g8, amplitude)


def main():
    args = parse_args(); out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    g8_dir = ROOT / "workdirs/radaz_e25_g8_simvp_residual_sr_sync10_20to30us"
    env_dir = ROOT / "workdirs/radaz_electric_history_hidden_band_envelope_rom"
    checkpoint = torch.load(g8_dir / "checkpoint_best.pth", map_location="cpu", weights_only=False)
    bundle = HiddenBandEnvelopeBundle.load(env_dir, device=device)
    data = {}
    for case in oracle.cases():
        _, _, interval, payload = oracle.evaluate_case(case, checkpoint, bundle, device)
        x, y, w = features(payload)
        data[case.name] = {"x": x, "y": y, "w": w, "payload": payload, "interval": interval}
        print(f"prepared {case.name}: {len(y)} samples", flush=True)
    rows = []
    for heldout, test in data.items():
        train_names = [name for name in data if name != heldout]
        x = np.concatenate([data[n]["x"] for n in train_names])
        y = np.concatenate([data[n]["y"] for n in train_names])
        w = np.concatenate([data[n]["w"] for n in train_names])
        model = make_pipeline(StandardScaler(), Ridge(alpha=args.alpha))
        model.fit(x, y, ridge__sample_weight=w)
        prediction = model.predict(test["x"])
        payload = test["payload"]
        candidate = corrected(payload, prediction)
        base = fusion.metrics_for(payload["truth_coeff"], payload["g8_coeff"], payload["radial_weights"], payload["time_us"])
        metric = fusion.metrics_for(payload["truth_coeff"], candidate, payload["radial_weights"], payload["time_us"])
        reduction = base["power_series_relative_l2"] - metric["power_series_relative_l2"]
        rows.append({"heldout": heldout, "train_cases": "+".join(train_names), "base_power_error": base["power_series_relative_l2"], "corrected_power_error": metric["power_series_relative_l2"], "relative_power_error_reduction": reduction / base["power_series_relative_l2"], "base_amplitude_error": base["amplitude_relative_l2"], "corrected_amplitude_error": metric["amplitude_relative_l2"], "base_coherence": base["complex_coherence"], "corrected_coherence": metric["complex_coherence"]})
        print(json.dumps(rows[-1], indent=2), flush=True)
    transition_rows = [r for r in rows if r["heldout"] != "E25_stationary"]
    stationary = next(r for r in rows if r["heldout"] == "E25_stationary")
    proceed = all(r["relative_power_error_reduction"] > 0 for r in transition_rows) and np.mean([r["relative_power_error_reduction"] for r in transition_rows]) >= 0.10 and stationary["relative_power_error_reduction"] >= -0.05
    summary = {"status": "predictability_test_complete", "primary_E25_to_E22p5_used": False, "diagnostic_model": "standardized weighted ridge on log amplitude ratio", "alpha": args.alpha, "folds": rows, "go_no_go": {"proceed_to_full_training": bool(proceed), "rule": "both held-out transitions improve, mean transition improvement >=10%, E25 degradation <=5%"}}
    with (out / "cross_trajectory_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    if proceed:
        x = np.concatenate([v["x"] for v in data.values()]); y = np.concatenate([v["y"] for v in data.values()]); w = np.concatenate([v["w"] for v in data.values()])
        final_model = make_pipeline(StandardScaler(), Ridge(alpha=args.alpha)); final_model.fit(x, y, ridge__sample_weight=w)
        joblib.dump(final_model, out / "g8_log_amplitude_closure.joblib")
        summary["full_model_trained"] = True
    else:
        summary["full_model_trained"] = False
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["go_no_go"], indent=2), flush=True)


if __name__ == "__main__": main()
