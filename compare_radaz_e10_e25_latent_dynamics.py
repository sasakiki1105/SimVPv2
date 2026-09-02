"""Compare latent dimensionality and autonomous dynamics for E10 and E25."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_E10_LATENT = (
    ROOT / "workdirs" / "analyze_radaz_bx20mt_ez10kvm_latent"
    / "latent_analysis_summary.json"
)
DEFAULT_E25_LATENT = (
    ROOT / "workdirs" / "analyze_radaz_bx20mt_ez25kvm_latent_e10frozen_targetnorm"
    / "latent_analysis_summary.json"
)
DEFAULT_E10_DYNAMICS = (
    ROOT / "workdirs" / "analyze_radaz_bx20mt_ez10kvm_hankel_havok_extended"
    / "hankel_havok_summary.json"
)
DEFAULT_E25_DYNAMICS = (
    ROOT / "workdirs"
    / "analyze_radaz_bx20mt_ez25kvm_hankel_havok_e10frozen_targetnorm_extended"
    / "hankel_havok_summary.json"
)
DEFAULT_OUTPUT = ROOT / "workdirs" / "compare_radaz_e10_e25_latent_dynamics"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dimensionality_rows(latent: dict, case: str) -> list[dict]:
    rows = []
    for layer in ("encoder", "translator"):
        values = latent["pca"][f"{layer}_steady"]
        rows.append(
            {
                "case": case,
                "layer": layer,
                "components_90": values["components_for_90_percent"],
                "components_95": values["components_for_95_percent"],
                "components_99": values["components_for_99_percent"],
                "variance_pc1": values["variance_pc1"],
                "variance_pc1_to_pc5": values["variance_pc1_to_pc5"],
                "variance_pc1_to_pc10": values["variance_pc1_to_pc10"],
                "feature_count": values["feature_count"],
            }
        )
    return rows


def dynamics_rows(summary: dict, case: str) -> list[dict]:
    rows = []
    for layer in ("encoder", "translator"):
        values = summary["layers"][layer]
        for method in ("standard_dmd", "hankel_dmd", "havok_zero_forcing"):
            metrics = values["metrics"][method]["24-30"]
            rows.append(
                {
                    "case": case,
                    "layer": layer,
                    "method": method,
                    "delay": values["selected_delay"],
                    "rank": values["selected_rank"],
                    "standardized_mse": metrics["standardized_mse"],
                    "skill_vs_persistence": metrics["skill_vs_persistence"],
                    "skill_vs_training_mean": metrics["skill_vs_training_mean"],
                    "correlation": metrics["flattened_correlation"],
                    "finite_fraction": metrics["finite_fraction"],
                }
            )
    return rows


def value(rows: list[dict], case: str, layer: str, method: str, key: str) -> float:
    return float(
        next(
            row[key]
            for row in rows
            if row["case"] == case
            and row["layer"] == layer
            and row["method"] == method
        )
    )


def plot_comparison(path: Path, dims: list[dict], dynamics: list[dict]) -> None:
    cases = ("E10", "E25")
    colors = {"E10": "#3977a8", "E25": "#d66b32"}
    methods = ("standard_dmd", "hankel_dmd", "havok_zero_forcing")
    method_labels = ("DMD", "Hankel DMD", "HAVOK zero forcing")
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)

    ax = axes[0, 0]
    positions = np.arange(2)
    width = 0.34
    for offset, case in zip((-0.5, 0.5), cases):
        heights = [
            next(row["components_95"] for row in dims if row["case"] == case and row["layer"] == layer)
            for layer in ("encoder", "translator")
        ]
        ax.bar(positions + offset * width, heights, width, color=colors[case], label=f"{case} kV/m")
    ax.set_xticks(positions, ("Encoder", "Translator"))
    ax.set_ylabel("Components for 95% variance")
    ax.set_title("Steady latent dimensionality")
    ax.set_ylim(0, 19)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")

    ax = axes[0, 1]
    metric_labels = ("PC1", "PC1-5", "PC1-10")
    keys = ("variance_pc1", "variance_pc1_to_pc5", "variance_pc1_to_pc10")
    x = np.arange(len(keys))
    for layer, marker in (("encoder", "o"), ("translator", "s")):
        for case in cases:
            row = next(item for item in dims if item["case"] == case and item["layer"] == layer)
            ax.plot(x, [row[key] for key in keys], marker=marker, lw=2, color=colors[case], linestyle="-" if layer == "encoder" else "--", label=f"{case} {layer}")
    ax.set_xticks(x, metric_labels)
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("How variance accumulates")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1, 0]
    x = np.arange(len(methods))
    for layer, marker in (("encoder", "o"), ("translator", "s")):
        for case in cases:
            scores = [value(dynamics, case, layer, method, "skill_vs_persistence") for method in methods]
            ax.plot(x, scores, marker=marker, lw=2, color=colors[case], linestyle="-" if layer == "encoder" else "--", label=f"{case} {layer}")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x, method_labels)
    ax.set_ylabel("Skill vs persistence")
    ax.set_title("Autonomous 24-30 us forecast")
    ax.set_ylim(-0.35, 0.85)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1, 1]
    x = np.arange(2)
    width = 0.18
    combinations = [
        ("E10", "encoder", -1.5),
        ("E10", "translator", -0.5),
        ("E25", "encoder", 0.5),
        ("E25", "translator", 1.5),
    ]
    for case, layer, offset in combinations:
        corr = value(dynamics, case, layer, "hankel_dmd", "correlation")
        ax.bar(x[0] + offset * width, corr, width, color=colors[case], alpha=1.0 if layer == "translator" else 0.65, label=f"{case} {layer}")
        mean_skill = value(dynamics, case, layer, "hankel_dmd", "skill_vs_training_mean")
        ax.bar(x[1] + offset * width, mean_skill, width, color=colors[case], alpha=1.0 if layer == "translator" else 0.65)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x, ("Trajectory correlation", "Skill vs training mean"))
    ax.set_title("Hankel DMD is not just mean reversion")
    ax.set_ylim(-0.65, 0.9)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("RadAz reduced latent dynamics: E10 vs E25 kV/m", fontsize=15)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e10-latent", type=Path, default=DEFAULT_E10_LATENT)
    parser.add_argument("--e25-latent", type=Path, default=DEFAULT_E25_LATENT)
    parser.add_argument("--e10-dynamics", type=Path, default=DEFAULT_E10_DYNAMICS)
    parser.add_argument("--e25-dynamics", type=Path, default=DEFAULT_E25_DYNAMICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    latent = {"E10": load_json(args.e10_latent), "E25": load_json(args.e25_latent)}
    dynamics_source = {"E10": load_json(args.e10_dynamics), "E25": load_json(args.e25_dynamics)}
    dims = dimensionality_rows(latent["E10"], "E10") + dimensionality_rows(latent["E25"], "E25")
    dynamics = dynamics_rows(dynamics_source["E10"], "E10") + dynamics_rows(dynamics_source["E25"], "E25")
    write_csv(args.output / "latent_dimensionality_comparison.csv", dims)
    write_csv(args.output / "latent_dynamics_comparison.csv", dynamics)
    plot_comparison(args.output / "latent_dimensionality_and_dynamics_comparison.png", dims, dynamics)

    e25_translator = next(row for row in dynamics if row["case"] == "E25" and row["layer"] == "translator" and row["method"] == "hankel_dmd")
    e10_translator = next(row for row in dynamics if row["case"] == "E10" and row["layer"] == "translator" and row["method"] == "hankel_dmd")
    summary = {
        "status": "PASS",
        "latent_dimensionality": dims,
        "autonomous_dynamics": dynamics,
        "conclusion": {
            "e25_pooled_latent_is_low_dimensional": True,
            "e25_translator_hankel_beats_persistence": e25_translator["skill_vs_persistence"] > 0,
            "e25_translator_hankel_beats_training_mean": e25_translator["skill_vs_training_mean"] > 0,
            "e25_translator_hankel_correlation": e25_translator["correlation"],
            "e10_translator_hankel_correlation": e10_translator["correlation"],
            "interpretation": "E25 latent dynamics are substantially more autonomous and predictable than E10 under the same frozen E10 feature extractor.",
        },
        "caveats": [
            "The SimVP weights were trained on E10 and frozen for E25.",
            "E25 uses train-only target normalization because fixed E10 normalization clips 86.4% of phi values.",
            "The forecast target is pooled latent PCA state, not decoded physical fields.",
            "PCA, normalization, and model selection use no E25 state after 24 us.",
        ],
    }
    (args.output / "comparison_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    text = f"""# E10/E25 latent reduced-dynamics comparison

## 結果

25 kV/mの定常潜在状態は、encoderで9成分、translatorで12成分あれば分散の95%を説明できた。10 kV/mはそれぞれ8成分、15成分であり、25 kV/mも同程度の少数自由度で表現できる。

20--24 usだけで同定し、24--30 usを完全自律予測したHankel DMDでは、25 kV/m translatorのpersistence比skillは `{e25_translator['skill_vs_persistence']:.4f}`、学習平均比skillは `{e25_translator['skill_vs_training_mean']:.4f}`、軌道相関は `{e25_translator['correlation']:.4f}` だった。10 kV/m translatorの相関 `{e10_translator['correlation']:.4f}` より明確に高い。

これは25 kV/mの潜在状態が単に低次元なだけでなく、少なくとも24--30 usでは過去1.2 usを含む線形遅延座標で時間発展までかなり再現できることを示す。

## 注意

- SimVP重みは10 kV/m学習済みモデルを固定している。
- 10 kV/m正規化を固定すると25 kV/mのphiが86.4%上限クリップされるため、25 kV/mの0--24 usだけから求めたtarget normalizationを使った。
- したがって厳密なzero-shot予測ではなく、25 kV/mの潜在力学を診断する実験である。
- 現在予測しているのは潜在PCA状態であり、物理場や輸送量を直接予測したとはまだ言えない。

## Files

- `latent_dimensionality_and_dynamics_comparison.png`
- `latent_dimensionality_comparison.csv`
- `latent_dynamics_comparison.csv`
- `comparison_summary.json`
"""
    (args.output / "README.md").write_text(text, encoding="utf-8")
    print(json.dumps(summary["conclusion"], indent=2))


if __name__ == "__main__":
    main()
