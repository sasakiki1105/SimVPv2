"""Compare RadAz ROM closure with Ez-adaptive physical mode coordinates.

The preceding closure map selected two energetic modes from the fixed n=1..6
band.  That band contains the ECDI fundamental at high Ez but excludes it at
low Ez because n0 is proportional to 1/Ez.  This driver reuses the exact same
causal closure protocol while selecting modes in normalized n/n0 bands.

Three predeclared ablations are supported:

* ``mtsi_only``: one energetic low-wavenumber mode, n/n0 <= 0.60.
* ``ecdi_only``: one energetic fundamental candidate, 0.75 <= n/n0 <= 1.25.
* ``joint``: one mode from each band, kept as separate carrier coordinates.

Mode energy is evaluated only on the fit mask supplied by the underlying
analysis.  No forecast samples enter mode selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_radaz_local_rom_closure_map as base


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT / "workdirs" / "compare_radaz_local_rom_closure_map_adaptive"
)

E_CHARGE_C = 1.602176634e-19
ELECTRON_MASS_KG = 9.1093837015e-31
MAGNETIC_FIELD_T = 20.0e-3
AZIMUTHAL_LENGTH_M = 1.28e-2
MTSI_MAX_N_OVER_N0 = 0.60
ECDI_MIN_N_OVER_N0 = 0.75
ECDI_MAX_N_OVER_N0 = 1.25
STRATEGIES = ("joint", "mtsi_only", "ecdi_only")
PRIMARY_SYSTEM = "latent_phi_transport"

_ACTIVE_EZ_KVM: float | None = None
_ACTIVE_STRATEGY = "joint"
_ORIGINAL_PREPARE_CASE = base.prepare_case


def format_ez(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def ecdi_mode_number(ez_kvm: float) -> float:
    electric_field_vpm = float(ez_kvm) * 1.0e3
    return (
        E_CHARGE_C
        * MAGNETIC_FIELD_T**2
        * AZIMUTHAL_LENGTH_M
        / (2.0 * np.pi * ELECTRON_MASS_KG * electric_field_vpm)
    )


def candidate_bands(ez_kvm: float, maximum_mode: int) -> dict[str, np.ndarray]:
    n0 = ecdi_mode_number(ez_kvm)
    mtsi_upper = min(maximum_mode, max(1, int(math.floor(MTSI_MAX_N_OVER_N0 * n0))))
    mtsi = np.arange(1, mtsi_upper + 1, dtype=np.int64)

    ecdi_lower = max(1, int(math.ceil(ECDI_MIN_N_OVER_N0 * n0)))
    ecdi_upper = min(maximum_mode, int(math.floor(ECDI_MAX_N_OVER_N0 * n0)))
    if ecdi_lower <= ecdi_upper:
        ecdi = np.arange(ecdi_lower, ecdi_upper + 1, dtype=np.int64)
    else:
        ecdi = np.asarray(
            [min(maximum_mode, max(1, int(round(n0))))], dtype=np.int64
        )
    return {"mtsi": mtsi, "ecdi": ecdi}


def most_energetic(energy: np.ndarray, candidates: np.ndarray) -> int:
    if not len(candidates):
        raise ValueError("Adaptive mode band contains no resolvable mode")
    return int(candidates[int(np.argmax(energy[candidates]))])


def adaptive_select_modes(
    phi: np.ndarray,
    radial_weights: np.ndarray,
    fit_mask: np.ndarray,
) -> np.ndarray:
    if _ACTIVE_EZ_KVM is None:
        raise RuntimeError("Ez context was not set before adaptive mode selection")
    if not np.any(fit_mask):
        raise ValueError("Mode-selection fit mask is empty")
    energy = np.einsum(
        "r,trm->m", radial_weights, np.abs(phi[fit_mask]) ** 2
    )
    bands = candidate_bands(_ACTIVE_EZ_KVM, phi.shape[-1] - 1)
    selected = {
        name: most_energetic(energy, candidates)
        for name, candidates in bands.items()
    }
    if _ACTIVE_STRATEGY == "mtsi_only":
        modes = [selected["mtsi"]]
    elif _ACTIVE_STRATEGY == "ecdi_only":
        modes = [selected["ecdi"]]
    elif _ACTIVE_STRATEGY == "joint":
        modes = [selected["mtsi"], selected["ecdi"]]
    else:
        raise ValueError(f"Unknown adaptive mode strategy: {_ACTIVE_STRATEGY}")
    if len(set(modes)) != len(modes):
        raise RuntimeError(
            f"MTSI and ECDI bands selected the same mode at Ez={_ACTIVE_EZ_KVM}"
        )
    return np.asarray(modes, dtype=np.int64)


def prepare_case_with_context(ez_kvm: int, output: Path):
    global _ACTIVE_EZ_KVM
    _ACTIVE_EZ_KVM = float(ez_kvm)
    return _ORIGINAL_PREPARE_CASE(ez_kvm, output)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def strategy_readme(
    path: Path,
    fixed_rows: list[dict],
    rolling_summary: list[dict],
    diagnostics: list[dict],
) -> None:
    fixed = {
        (int(row["ez_kvm"]), row["system"]): row for row in fixed_rows
    }
    rolling = {
        (int(row["ez_kvm"]), row["system"]): row for row in rolling_summary
    }
    lines = [
        f"# Ez-adaptive local ROM closure: {_ACTIVE_STRATEGY}",
        "",
        "Modes are selected causally in coordinates normalized by the theoretical ECDI fundamental",
        "",
        "```text",
        "n0 = e Bx^2 Ly / (2 pi me Ez)",
        f"MTSI candidates: n/n0 <= {MTSI_MAX_N_OVER_N0:.2f}",
        f"ECDI candidates: {ECDI_MIN_N_OVER_N0:.2f} <= n/n0 <= {ECDI_MAX_N_OVER_N0:.2f}",
        "```",
        "",
        "The energetic mode inside each candidate band is selected from the fit interval only. The forecast truth is excluded from mode selection, POD/PCA, scaling, and delay/rank selection.",
        "",
        "| Ez [kV/m] | n0 | selected modes | fixed level | fixed transport corr | rolling transport pass | rolling median corr |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ez in base.EZ_VALUES:
        frow = fixed[(ez, PRIMARY_SYSTEM)]
        rrow = rolling[(ez, PRIMARY_SYSTEM)]
        lines.append(
            f"| {ez} | {ecdi_mode_number(ez):.3f} | {frow['selected_modes']} | "
            f"{frow['closure_level']} | {frow['transport_correlation']:.3f} | "
            f"{rrow['transport_pass_fraction']:.2f} | "
            f"{rrow['median_transport_correlation']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The closure levels are empirical out-of-sample criteria, not a proof of exact mathematical closure. `joint` keeps one MTSI-band carrier and one ECDI-band carrier as separate coordinates, although the scalar `T` observable remains their summed modal transport. The single-band runs isolate each contribution.",
            "",
            "The frozen SimVP feature extractor is still the Ez=10 data-only model for every case. This experiment changes only the physical mode coordinates.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_strategy(
    strategy: str,
    output_root: Path,
    delays: str,
    ranks: str,
    cases: str,
) -> Path:
    global _ACTIVE_STRATEGY, _ACTIVE_EZ_KVM
    _ACTIVE_STRATEGY = strategy
    _ACTIVE_EZ_KVM = None
    output = output_root / strategy
    previous_argv = sys.argv[:]
    try:
        sys.argv = [
            base.__file__,
            "--output", str(output),
            "--delays", delays,
            "--ranks", ranks,
            "--cases", cases,
        ]
        base.main()
    finally:
        sys.argv = previous_argv
    definition = {
        "strategy": strategy,
        "selection_is_fit_only": True,
        "n0_formula": "e * Bx^2 * Ly / (2*pi*me*Ez)",
        "Bx_T": MAGNETIC_FIELD_T,
        "Ly_m": AZIMUTHAL_LENGTH_M,
        "mtsi_n_over_n0_max": MTSI_MAX_N_OVER_N0,
        "ecdi_n_over_n0_min": ECDI_MIN_N_OVER_N0,
        "ecdi_n_over_n0_max": ECDI_MAX_N_OVER_N0,
    }
    (output / "mode_definition.json").write_text(
        json.dumps(definition, indent=2), encoding="utf-8"
    )
    return output


def collect_comparison(output_root: Path, strategies: list[str]) -> list[dict]:
    rows: list[dict] = []
    available = [
        strategy
        for strategy in STRATEGIES
        if (output_root / strategy / "fixed_closure_map.csv").is_file()
        and (output_root / strategy / "rolling_closure_summary.csv").is_file()
    ]
    sources = [(strategy, output_root / strategy) for strategy in available]
    legacy = ROOT / "workdirs" / "compare_radaz_local_rom_closure_map"
    if (legacy / "fixed_closure_map.csv").is_file():
        sources.append(("legacy_fixed_n1_6", legacy))
    for strategy, source in sources:
        fixed_rows = read_csv(source / "fixed_closure_map.csv")
        rolling_rows = read_csv(source / "rolling_closure_summary.csv")
        rolling_lookup = {
            (row["ez_kvm"], row["system"]): row for row in rolling_rows
        }
        for row in fixed_rows:
            if row["system"] != PRIMARY_SYSTEM:
                continue
            rolling = rolling_lookup[(row["ez_kvm"], PRIMARY_SYSTEM)]
            ez = float(row["ez_kvm"])
            bands = candidate_bands(ez, 21)
            rows.append(
                {
                    "strategy": strategy,
                    "ez_kvm": format_ez(ez),
                    "ecdi_n0": ecdi_mode_number(ez),
                    "mtsi_candidates": ",".join(map(str, bands["mtsi"])),
                    "ecdi_candidates": ",".join(map(str, bands["ecdi"])),
                    "selected_modes": row["selected_modes"],
                    "fixed_closure_level": row["closure_level"],
                    "fixed_transport_correlation": row["transport_correlation"],
                    "fixed_transport_std_ratio": row["transport_std_ratio"],
                    "fixed_phi_correlation": row["phi_coefficient_correlation"],
                    "fixed_phi_phase_mae_rad": row["phi_phase_mae_rad"],
                    "fixed_state_correlation": row["state_correlation"],
                    "rolling_transport_pass_fraction": rolling[
                        "transport_pass_fraction"
                    ],
                    "rolling_median_transport_correlation": rolling[
                        "median_transport_correlation"
                    ],
                }
            )
    return rows


def plot_comparison(output_root: Path, rows: list[dict]) -> None:
    strategies = list(dict.fromkeys(row["strategy"] for row in rows))
    ez_values = np.asarray(base.EZ_VALUES, dtype=np.float64)
    colors = {
        "legacy_fixed_n1_6": "#777777",
        "mtsi_only": "#0072b2",
        "ecdi_only": "#d55e00",
        "joint": "#009e73",
    }
    markers = {
        "legacy_fixed_n1_6": "x",
        "mtsi_only": "s",
        "ecdi_only": "^",
        "joint": "o",
    }
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    for strategy in strategies:
        selected = sorted(
            (row for row in rows if row["strategy"] == strategy),
            key=lambda row: float(row["ez_kvm"]),
        )
        axes[0].plot(
            [float(row["ez_kvm"]) for row in selected],
            [float(row["fixed_transport_correlation"]) for row in selected],
            marker=markers[strategy],
            color=colors[strategy],
            label=strategy,
        )
        axes[1].plot(
            [float(row["ez_kvm"]) for row in selected],
            [float(row["rolling_transport_pass_fraction"]) for row in selected],
            marker=markers[strategy],
            color=colors[strategy],
            label=strategy,
        )
    axes[0].axhline(base.TRANSPORT_CORRELATION_MIN, color="#333333", linestyle=":")
    axes[0].set_ylabel("fixed transport correlation")
    axes[1].set_ylabel("rolling transport pass fraction")
    for axis in axes:
        axis.set_xlabel("Ez [kV/m]")
        axis.set_xticks(ez_values)
        axis.grid(True, linestyle=":", alpha=0.55)
    axes[1].legend(loc="lower right", fontsize=8)
    figure.suptitle("Fixed-band versus Ez-adaptive physical mode closure")
    figure.savefig(output_root / "adaptive_mode_closure_comparison.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    axis.plot(
        ez_values,
        [ecdi_mode_number(value) for value in ez_values],
        color="#222222",
        linewidth=1.6,
        label="theoretical ECDI n0",
    )
    for strategy in ("mtsi_only", "ecdi_only", "joint"):
        if strategy not in strategies:
            continue
        selected = [row for row in rows if row["strategy"] == strategy]
        for row in selected:
            modes = [int(value) for value in row["selected_modes"].split(",")]
            axis.scatter(
                [float(row["ez_kvm"])] * len(modes),
                modes,
                color=colors[strategy],
                marker=markers[strategy],
                s=42,
                label=strategy if row is selected[0] else None,
            )
    axis.set_xlabel("Ez [kV/m]")
    axis.set_ylabel("azimuthal mode n")
    axis.set_xticks(ez_values)
    axis.grid(True, linestyle=":", alpha=0.55)
    axis.legend(loc="upper right", fontsize=8)
    figure.savefig(output_root / "adaptive_selected_modes.png", dpi=180)
    plt.close(figure)


def write_root_readme(output_root: Path, rows: list[dict]) -> None:
    lookup = {
        (row["strategy"], row["ez_kvm"]): row for row in rows
    }
    strategies = [
        strategy
        for strategy in ("legacy_fixed_n1_6", "mtsi_only", "ecdi_only", "joint")
        if any(row["strategy"] == strategy for row in rows)
    ]
    def metric(strategy: str, ez: int, name: str) -> float:
        return float(lookup[(strategy, str(ez))][name])

    lines = [
        "# Ez-adaptive physical-mode closure ablation",
        "",
        "This analysis removes the fixed n=1..6 confound from the local ROM closure map. Mode selection is fit-only and normalized by the theoretical ECDI fundamental n0(Ez).",
        "",
        "| strategy | Ez [kV/m] | n0 | selected modes | fixed transport corr | fixed level | rolling pass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy in strategies:
        for ez in base.EZ_VALUES:
            row = lookup[(strategy, format_ez(float(ez)))]
            lines.append(
                f"| {strategy} | {ez} | {float(row['ecdi_n0']):.3f} | "
                f"{row['selected_modes']} | "
                f"{float(row['fixed_transport_correlation']):.3f} | "
                f"{row['fixed_closure_level']} | "
                f"{float(row['rolling_transport_pass_fraction']):.2f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `legacy_fixed_n1_6` is the original low-mode result retained for provenance.",
            "- `mtsi_only` and `ecdi_only` test whether either instability band is locally sufficient.",
            "- `joint` includes one carrier from each band. The two carrier coordinates stay separate in Pcirc and dX; the scalar modal transport T is their physical sum.",
            "- A low-E improvement in `ecdi_only` or `joint` indicates that the fixed n=1..6 state omitted an important ECDI coordinate. Continued failure after adaptive selection points instead to memory, mode coupling, or unobserved kinetic variables.",
            "",
            "## Main findings",
            "",
            f"- E10 improves when n=15 is admitted: joint fixed transport correlation is {metric('joint', 10, 'fixed_transport_correlation'):.3f}, versus {metric('legacy_fixed_n1_6', 10, 'fixed_transport_correlation'):.3f} for fixed n=1..6, and rolling transport passes {metric('joint', 10, 'rolling_transport_pass_fraction'):.2f} versus {metric('legacy_fixed_n1_6', 10, 'rolling_transport_pass_fraction'):.2f}. However, MTSI-only is almost as correlated ({metric('mtsi_only', 10, 'fixed_transport_correlation'):.3f}) while ECDI-only is anticorrelated ({metric('ecdi_only', 10, 'fixed_transport_correlation'):.3f}). The missing ECDI coordinate was a confound, but it was not the sole cause of E10 non-closure.",
            f"- E20 remains non-closed in every ablation. MTSI-only correlation is {metric('mtsi_only', 20, 'fixed_transport_correlation'):.3f}; ECDI-only is {metric('ecdi_only', 20, 'fixed_transport_correlation'):.3f}; joint is {metric('joint', 20, 'fixed_transport_correlation'):.3f}. The joint forecast variance is unstable, so E20 cannot be rescued by replacing the fixed mode band alone.",
            f"- E25 remains the strongest complete local closure. Joint L+Pcirc+T has correlation {metric('joint', 25, 'fixed_transport_correlation'):.3f} and passes all rolling transport windows. Its selected set is the same physical pair as before, n=2 and n=6, only reordered by role. The separate strategy output confirms that L+Pcirc+T+dX still reaches level 4.",
            f"- E30 is mixed and window-sensitive. MTSI-only reaches fixed correlation {metric('mtsi_only', 30, 'fixed_transport_correlation'):.3f} but passes only {metric('mtsi_only', 30, 'rolling_transport_pass_fraction'):.2f} of rolling windows; ECDI-only has correlation {metric('ecdi_only', 30, 'fixed_transport_correlation'):.3f} and passes {metric('ecdi_only', 30, 'rolling_transport_pass_fraction'):.2f}. Combining both bands does not produce complete phase closure.",
            f"- E40 closure is ECDI-led. ECDI-only fixed correlation is {metric('ecdi_only', 40, 'fixed_transport_correlation'):.3f}, while MTSI-only falls to {metric('mtsi_only', 40, 'fixed_transport_correlation'):.3f}. Joint and ECDI-only therefore agree on the dominant closed subsystem.",
            "",
            "The current five Ez values remain exploratory. Ez=22.5 kV/m can be added as a predeclared interpolation case after its stitched physical fields and frozen-model latent features are available.",
        ]
    )
    (output_root / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strategies", default=",".join(STRATEGIES))
    parser.add_argument("--delays", default="20,40,60,80")
    parser.add_argument("--ranks", default="8,12,20,30,40")
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Rebuild the cross-strategy tables and plots from existing runs.",
    )
    parser.add_argument(
        "--cases", default=",".join(str(value) for value in base.EZ_VALUES)
    )
    args = parser.parse_args()
    strategies = [value.strip() for value in args.strategies.split(",") if value.strip()]
    unknown = set(strategies) - set(STRATEGIES)
    if unknown:
        raise ValueError(f"Unknown strategies: {sorted(unknown)}")
    args.output.mkdir(parents=True, exist_ok=True)

    if not args.summarize_only:
        base.carrier.select_modes = adaptive_select_modes
        base.prepare_case = prepare_case_with_context
        base.write_readme = strategy_readme
        for strategy in strategies:
            print(f"[STRATEGY] {strategy}", flush=True)
            run_strategy(
                strategy, args.output, args.delays, args.ranks, args.cases
            )

    comparison = collect_comparison(args.output, strategies)
    write_csv(args.output / "adaptive_mode_ablation.csv", comparison)
    plot_comparison(args.output, comparison)
    write_root_readme(args.output, comparison)
    print(f"PASS: wrote adaptive closure ablation to {args.output}", flush=True)


if __name__ == "__main__":
    main()
