#!/usr/bin/env python3
"""Apply the frozen soft-gated ECDI decoder to external electric histories.

The G8 checkpoint, E25 normalization, ECDI band, robust OOD reference, and
3--5 sigma smoothstep are imported unchanged from the completed Section 39
evaluation.  PIC truth is used only after decoder application for metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from compare_radaz_proximal_physics_decoders import (
    DEFAULT_G8,
    DEFAULT_OUTPUT as DEFAULT_REFERENCE_OUTPUT,
    SOFT_GATE_DECODER,
    SOFT_GATE_Z_FULL,
    SOFT_GATE_Z_START,
    Case,
    evaluate_case,
    plot_soft_gate_summary,
    write_csv,
)
from analyze_radaz_g2_stability_reconstruction import json_safe


ROOT = Path(__file__).resolve().parent
E20_TO_E22P5_H5 = (
    ROOT
    / "workdirs"
    / "radaz_e20_to_e22p5_transition"
    / "radaz_3ch_e25targetnorm_native257x256_pad260x256.h5"
)
E22P5_TO_E20_H5 = (
    ROOT
    / "workdirs"
    / "radaz_e22p5_to_e20_transition"
    / "radaz_3ch_e25targetnorm_native257x256_pad260x256.h5"
)
DEFAULT_OUTPUT = ROOT / "workdirs" / "radaz_softgate_external_histories"
CANDIDATES = (
    "raw_simvp",
    "positivity_joint_1pct",
    "positive_protect_ecdi_n9_21",
    SOFT_GATE_DECODER,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g8-workdir", type=Path, default=DEFAULT_G8)
    parser.add_argument(
        "--reference-summary",
        type=Path,
        default=DEFAULT_REFERENCE_OUTPUT / "summary.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budget", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_record(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": int(stat.st_size),
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        "sha256": sha256_file(path),
    }


def plot_external_comparison(path: Path, rows: list[dict]) -> None:
    cases = list(dict.fromkeys(row["case"] for row in rows))
    labels = ("raw", "positive joint", "hard ECDI", "soft-gated ECDI")
    metrics = (
        ("poisson_residual_ratio_to_raw", "full Poisson residual / raw"),
        ("electric_field_relative_l2", "electric-field relative L2"),
        ("phi_relative_l2", "phi relative L2"),
        ("phi_ecdi_power_ratio", "phi ECDI power / truth"),
        ("transport_ecdi_relative_l2", "ECDI transport relative L2"),
        ("transport_mtsi_relative_l2", "MTSI transport relative L2"),
    )
    lookup = {(row["case"], row["decoder"]): row for row in rows}
    x = np.arange(len(cases), dtype=np.float64)
    width = 0.19
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for axis, (metric, title) in zip(axes.ravel(), metrics):
        for index, (decoder, label) in enumerate(zip(CANDIDATES, labels)):
            values = [lookup[(case, decoder)][metric] for case in cases]
            axis.bar(x + (index - 1.5) * width, values, width, label=label)
        axis.set_xticks(x, cases, rotation=10)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.g8_workdir / "checkpoint_best.pth"
    required = (
        checkpoint_path,
        args.reference_summary,
        E20_TO_E22P5_H5,
        E22P5_TO_E20_H5,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required artifacts: {missing}")

    reference_summary = json.loads(
        args.reference_summary.read_text(encoding="utf-8")
    )
    ood_reference = dict(reference_summary["soft_gate_ood_reference"])
    if (
        float(ood_reference["z_start"]) != SOFT_GATE_Z_START
        or float(ood_reference["z_full"]) != SOFT_GATE_Z_FULL
    ):
        raise ValueError("The stored OOD gate no longer matches the frozen rule")

    cases = (
        Case(
            "E20_to_E22p5_30to35us",
            E20_TO_E22P5_H5,
            30.0,
            35.0,
            True,
            257,
            evaluation_start_us=30.360,
            evaluation_end_us=34.950,
        ),
        Case(
            "E20_to_E22p5_35to40us",
            E20_TO_E22P5_H5,
            35.0,
            40.0,
            True,
            257,
            evaluation_start_us=35.355,
            evaluation_end_us=39.945,
        ),
        Case(
            "E22p5_to_E20_30to35us",
            E22P5_TO_E20_H5,
            30.0,
            35.0,
            True,
            257,
            evaluation_start_us=30.360,
            evaluation_end_us=34.950,
        ),
    )

    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "external-history application of the Section 39 frozen soft gate",
        "truth_used_for_rule_or_weight_selection": False,
        "history_status": (
            "not used by the E25 G8 checkpoint, Section 38 spectral comparison, "
            "or Section 39 gate calibration; previously analyzed in separate ROM studies, "
            "so this is external reuse rather than a globally blind test"
        ),
        "frozen_rule": {
            "phi_band": "n=9-21",
            "ood_feature": "log raw relative Poisson residual",
            "reference": ood_reference,
            "gate": "cubic smoothstep: zero at z<=3 and one at z>=5",
            "density_rule": "unchanged positivity-aware weighted joint correction",
            "channel_budget": args.budget,
        },
        "evaluation_cases": [
            {
                "label": case.label,
                "time_us": [case.evaluation_start_us, case.evaluation_end_us],
                "h5": str(case.h5.resolve()),
            }
            for case in cases
        ],
        "artifacts": {
            "checkpoint": artifact_record(checkpoint_path),
            "reference_summary": artifact_record(args.reference_summary),
            "e20_to_e22p5_h5": artifact_record(E20_TO_E22P5_H5),
            "e22p5_to_e20_h5": artifact_record(E22P5_TO_E20_H5),
        },
    }
    protocol_path = args.output_dir / "pre_evaluation_protocol.json"
    protocol_path.write_text(
        json.dumps(json_safe(protocol), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    requested = args.device
    if requested.startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"
    device = torch.device(requested)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("metadata", {}).get("grid_factor", -1)) != 8:
        raise ValueError("checkpoint is not the frozen G8 model")

    all_rows: list[dict] = []
    summaries: dict[str, dict] = {}
    for case in cases:
        rows, case_summary, returned_reference = evaluate_case(
            case,
            checkpoint,
            device,
            args.budget,
            ood_reference=ood_reference,
            candidate_labels=CANDIDATES,
        )
        if returned_reference != ood_reference:
            raise RuntimeError("External evaluation unexpectedly changed OOD reference")
        all_rows.extend(rows)
        summaries[case.label] = case_summary

    comparison = {
        case.label: {
            row["decoder"]: row
            for row in all_rows
            if row["case"] == case.label
        }
        for case in cases
    }
    summary = {
        "description": "Frozen soft-gated ECDI decoder on external electric histories",
        "protocol": protocol,
        "device": str(device),
        "cases": summaries,
        "comparison": comparison,
        "claim_boundary": protocol["history_status"],
    }
    write_csv(args.output_dir / "external_history_comparison.csv", all_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plot_external_comparison(
        args.output_dir / "external_history_comparison.png", all_rows
    )
    plot_soft_gate_summary(args.output_dir / "soft_gate_diagnostics.png", summaries)
    readme = """# Frozen soft gate: external electric histories

The Section 39 E25-trained G8 checkpoint, positivity-aware 1% joint decoder,
ECDI n=9--21 band, log-MAD OOD reference, and z=3--5 smoothstep are applied
without recalibration to E20->E22.5 and E22.5->E20 histories.  PIC truth is
used only for the metrics after all decoder outputs have been fixed.

These histories were previously used in separate ROM-development studies.
They are external to this decoder and gate but are not globally blind data.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False), flush=True)
    print(f"[saved] {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
