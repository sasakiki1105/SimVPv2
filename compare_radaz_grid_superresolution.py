#!/usr/bin/env python3
"""Compare held-out E25 residual super-resolution results across G2/G4/G8."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FACTORS = (2, 4, 8)
OUTDIR = ROOT / "workdirs" / "compare_radaz_e25_g2_g4_g8_residual_sr"
CHANNELS = ("electron_den", "ion_den", "phi")
DISPLAY = {"electron_den": r"$n_e$", "ion_den": r"$n_i$", "phi": r"$\phi$"}


def workdir(factor: int) -> Path:
    return (
        ROOT
        / "workdirs"
        / f"radaz_e25_g{factor}_simvp_residual_sr_sync10_20to30us"
    )


def load_factor(factor: int) -> dict:
    directory = workdir(factor)
    final = json.loads((directory / "final_summary.json").read_text(encoding="utf-8"))
    analysis_dir = directory / "stability_reconstruction_analysis"
    analysis = json.loads(
        (analysis_dir / "stability_reconstruction_summary.json").read_text(
            encoding="utf-8"
        )
    )
    with (analysis_dir / "azimuthal_mode_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        modes = list(csv.DictReader(handle))
    n7 = {
        row["field"]: row
        for row in modes
        if int(row["mode"]) == 7 and row["field"] in CHANNELS
    }
    return {"final": final, "analysis": analysis, "n7": n7}


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    loaded = {factor: load_factor(factor) for factor in FACTORS}
    rows = []
    decisions = {}
    for factor in FACTORS:
        result = loaded[factor]
        final = result["final"]
        analysis = result["analysis"]
        stability = analysis["saturated_stability"]
        transport = analysis["modal_transport_proxy_ne_ey"]
        physics = analysis["physics"]
        row = {
            "grid": f"G{factor}",
            "factor": factor,
            "coarse_cells_per_direction": 256 // factor,
            "coarse_cell_fraction": 1.0 / factor**2,
            "best_epoch": final["best_epoch"],
            "test_mse": final["test"]["reconstruction_mse"],
            "interpolation_mse": final["test"]["baseline_interpolation_mse"],
            "skill_over_interpolation": final["test"]["skill_over_interpolation"],
            "electric_field_relative_l2": physics["model"]["electric_field_relative_l2"],
            "poisson_relative_residual_median": physics["model"]["relative_poisson_residual"]["median"],
            "poisson_residual_ratio_to_truth": physics["model"]["poisson_residual_median_ratio_to_truth"],
            "quasineutral_ratio_to_truth": physics["model"]["quasineutral_median_ratio_to_truth"],
            "mtsi_transport_relative_l2": transport["mtsi_candidate_n1_6"]["model"]["relative_l2"],
            "mtsi_transport_time_correlation": transport["mtsi_candidate_n1_6"]["model"]["time_correlation"],
            "ecdi_transport_relative_l2": transport["ecdi_candidate_n9_21"]["model"]["relative_l2"],
            "ecdi_transport_time_correlation": transport["ecdi_candidate_n9_21"]["model"]["time_correlation"],
        }
        for channel in CHANNELS:
            field = analysis["field_metrics"][channel]
            state = stability[channel]
            row[f"{channel}_nrmse_std"] = field["model_nrmse_std"]
            row[f"{channel}_skill"] = field["model_skill_over_interpolation"]
            row[f"{channel}_dominant_mode"] = state["model_dominant_mode_mean_spectrum"]
            row[f"{channel}_dominant_agreement"] = state[
                "model_dominant_mode_time_agreement"
            ]
            row[f"{channel}_mtsi_power_ratio"] = state["bands"][
                "mtsi_candidate_n1_6"
            ]["model"]["mean_power_ratio"]
            row[f"{channel}_ecdi_power_ratio"] = state["bands"][
                "ecdi_candidate_n9_21"
            ]["model"]["mean_power_ratio"]
            row[f"{channel}_n7_amplitude_ratio"] = float(
                result["n7"][channel]["model_amplitude_ratio"]
            )
            row[f"{channel}_n7_coherence"] = float(
                result["n7"][channel]["model_coherence"]
            )
            row[f"{channel}_n7_relative_error"] = float(
                result["n7"][channel]["model_relative_error"]
            )
        aggregate_pass = (
            min(row[f"{channel}_dominant_agreement"] for channel in CHANNELS) >= 0.9
            and row["mtsi_transport_relative_l2"] <= 0.10
            and row["mtsi_transport_time_correlation"] >= 0.90
            and row["ecdi_transport_relative_l2"] <= 0.15
            and row["ecdi_transport_time_correlation"] >= 0.90
        )
        coherent_n7_pass = (
            row["phi_n7_relative_error"] <= 0.20
            and row["phi_n7_coherence"] >= 0.95
        )
        poisson_pass = row["poisson_residual_ratio_to_truth"] <= 10.0
        decisions[f"G{factor}"] = {
            "aggregate_saturated_instability_gate": aggregate_pass,
            "coherent_phi_n7_gate": coherent_n7_pass,
            "poisson_truth_floor_gate": poisson_pass,
        }
        rows.append(row)

    with (OUTDIR / "grid_superresolution_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "description": "Held-out E25 synchronous residual super-resolution comparison",
        "factors": list(FACTORS),
        "gate_definitions": {
            "aggregate_saturated_instability": (
                "dominant-mode agreement >=0.9; MTSI/ECDI transport relative L2 <=0.10/0.15 "
                "and time correlation >=0.90"
            ),
            "coherent_phi_n7": "phi n7 relative error <=0.20 and coherence >=0.95",
            "poisson_truth_floor": "median predicted/true relative residual <=10",
        },
        "decisions": decisions,
        "rows": rows,
        "interpretation": {
            "aggregate_boundary": "not reached by G8 on the same-case saturated test",
            "coherent_n7_boundary": "between G4 and G8",
            "poisson_consistency": "not reached even at G2 without physics loss",
            "caution": (
                "G8 input Nyquist is n=16; n=17--21 cannot be directly observed and are inferred "
                "from same-case correlations rather than uniquely recovered"
            ),
        },
    }
    (OUTDIR / "grid_superresolution_comparison.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    x = np.arange(len(FACTORS))
    labels = [f"G{factor}\n{256//factor}x{256//factor}" for factor in FACTORS]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

    for channel in CHANNELS:
        axes[0, 0].plot(
            x,
            [row[f"{channel}_nrmse_std"] for row in rows],
            marker="o",
            label=DISPLAY[channel],
        )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Field NRMSE / truth std")
    axes[0, 0].legend()

    for channel in CHANNELS:
        axes[0, 1].plot(
            x,
            [row[f"{channel}_ecdi_power_ratio"] for row in rows],
            marker="o",
            label=DISPLAY[channel],
        )
    axes[0, 1].axhline(1.0, color="black", linewidth=0.8)
    axes[0, 1].set_title("ECDI candidate n=9--21 power ratio")

    for channel in CHANNELS:
        axes[0, 2].plot(
            x,
            [row[f"{channel}_n7_relative_error"] for row in rows],
            marker="o",
            label=DISPLAY[channel],
        )
    axes[0, 2].axhline(0.2, color="black", linestyle=":", linewidth=0.8)
    axes[0, 2].set_title("Individual n=7 relative error")

    axes[1, 0].plot(
        x,
        [row["mtsi_transport_relative_l2"] for row in rows],
        marker="o",
        label="MTSI n=1--6",
    )
    axes[1, 0].plot(
        x,
        [row["ecdi_transport_relative_l2"] for row in rows],
        marker="o",
        label="ECDI n=9--21",
    )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title(r"Modal $\langle n_eE_y\rangle$ relative error")
    axes[1, 0].legend()

    axes[1, 1].plot(
        x,
        [row["electric_field_relative_l2"] for row in rows],
        marker="o",
        label="electric field",
    )
    axes[1, 1].plot(
        x,
        [row["poisson_relative_residual_median"] for row in rows],
        marker="o",
        label="Poisson residual",
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Derived-field and physics errors")
    axes[1, 1].legend()

    axes[1, 2].plot(
        x,
        [row["test_mse"] for row in rows],
        marker="o",
        label="model",
    )
    axes[1, 2].plot(
        x,
        [row["interpolation_mse"] for row in rows],
        marker="o",
        label="interpolation",
    )
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_title("Held-out reconstruction MSE")
    axes[1, 2].legend()

    for axis in axes.flat:
        axis.set_xticks(x, labels)
        axis.grid(alpha=0.25)
    fig.suptitle("E25 grid-coarsening recoverability: G2 vs G4 vs G8")
    fig.savefig(OUTDIR / "grid_superresolution_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(decisions, indent=2))
    print(f"[saved] {OUTDIR}")


if __name__ == "__main__":
    main()
