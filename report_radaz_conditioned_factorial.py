"""Tabulate the RadAz conditioned 2x2 factorial evaluation.

Reads factorial_evaluation_*.json written by
evaluate_radaz_conditioned_factorial.py and prints the Section 15.5 gates:

  gate 1  field / copy < 1        (image-level conditional generalization)
  gate 2  mode observables better with the physics loss than data-only
  gate 3  modal transport / copy < 1  (physically usable surrogate)

plus the 2x2 main effects (conditioning C-U, physics loss P-D) and their
interaction, in log ratio to the copy baseline.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

CELLS = ["U-D", "C-D", "U-P", "C-P"]
FIELD_KEYS = ["mse_electron_den", "mse_ion_den", "mse_phi", "mse_Ey"]
MODE_KEYS = ["complex_mode_loss", "amplitude_mse", "cross_phase_error_rad"]
TRANSPORT_KEYS = ["transport_loss", "gamma_mse"]


def ratio(model, copy):
    if copy == 0 or not math.isfinite(copy) or not math.isfinite(model):
        return float("nan")
    return model / copy


def table(rows, headers):
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    line = "  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.path.read_text(encoding="utf-8"))
    results = payload["results"]
    order = [k for k in results if results[k]["role"] == "source"] + \
            [k for k in results if results[k]["role"] != "source"]

    print("=" * 110)
    print("RadAz conditioned 2x2 factorial -- direct10 on frames %s (%s us), checkpoint=%s"
          % (payload["protocol"]["window_frames"], payload["protocol"]["window_us"],
             payload["protocol"]["checkpoint"]))
    print("=" * 110)

    for group, title in (("field", "FIELD / COPY  (gate 1: <1 wins)"),
                         ("mode", "q-MODE / COPY  (gate 2)"),
                         ("transport", "MODAL TRANSPORT / COPY  (gate 3: <1 wins)")):
        keys = {"field": FIELD_KEYS, "mode": MODE_KEYS,
                "transport": TRANSPORT_KEYS}[group]
        for key in keys:
            print()
            print("--- %s :: %s" % (title, key))
            rows = []
            for case_key in order:
                case = results[case_key]
                copy = case["baselines"]["copy"][key]
                row = [case_key, case["role"][:6],
                       "n0=%.2f" % case["mode_n0"], "%.4g" % copy]
                for cell in CELLS:
                    if cell not in case["cells"]:
                        row.append("-")
                        continue
                    row.append("%.4f" % ratio(case["cells"][cell][key], copy))
                rows.append(row)
            table(rows, ["case", "role", "n0", "copy", *CELLS])

    # ---- 2x2 main effects on the log ratio to copy -------------------------
    print()
    print("=" * 110)
    print("2x2 MAIN EFFECTS on mean log10(metric/copy) over cases "
          "(negative = better than copy; effect<0 = the factor helps)")
    print("=" * 110)
    for scope, label in (("source", "source (in-condition)"),
                         ("axis_holdout", "axis holdout (unseen condition)")):
        subset = [k for k in order if results[k]["role"] == scope]
        rows = []
        for key in FIELD_KEYS + MODE_KEYS + TRANSPORT_KEYS:
            values = {}
            for cell in CELLS:
                logs = []
                for case_key in subset:
                    case = results[case_key]
                    if cell not in case["cells"]:
                        continue
                    r = ratio(case["cells"][cell][key],
                              case["baselines"]["copy"][key])
                    if r > 0 and math.isfinite(r):
                        logs.append(math.log10(r))
                values[cell] = sum(logs) / len(logs) if logs else float("nan")
            if any(math.isnan(v) for v in values.values()):
                continue
            cond = 0.5 * ((values["C-D"] - values["U-D"]) +
                          (values["C-P"] - values["U-P"]))
            phys = 0.5 * ((values["U-P"] - values["U-D"]) +
                          (values["C-P"] - values["C-D"]))
            inter = (values["C-P"] - values["U-P"]) - (values["C-D"] - values["U-D"])
            rows.append([key] + ["%+.3f" % values[c] for c in CELLS] +
                        ["%+.3f" % cond, "%+.3f" % phys, "%+.3f" % inter])
        print()
        print("--- %s  (n=%d cases)" % (label, len(subset)))
        table(rows, ["metric", *CELLS, "cond(C-U)", "phys(P-D)", "interaction"])

    # ---- closed loop -------------------------------------------------------
    print()
    print("=" * 110)
    print("CLOSED-LOOP 40 (4 chained direct10, 600 ns) -- electron density MSE / copy")
    print("=" * 110)
    rows = []
    for case_key in order:
        case = results[case_key]
        row = [case_key, case["role"][:6]]
        for cell in CELLS:
            entry = case["cells"].get(cell)
            if entry is None or not entry["closed_loop_40"].get("model"):
                row.append("-")
                continue
            cl = entry["closed_loop_40"]
            row.append("%.3f%s" % (
                ratio(cl["model"]["mse_electron_den"], cl["copy"]["mse_electron_den"]),
                "" if cl["finite"] else "!"))
        rows.append(row)
    table(rows, ["case", "role", *CELLS])

    print()
    print("--- closed-loop per-block ne MSE / copy (block b = frames 10b..10b+9 ahead)")
    for case_key in order:
        case = results[case_key]
        parts = []
        for cell in CELLS:
            entry = case["cells"].get(cell)
            if entry is None or "blocks" not in entry["closed_loop_40"]:
                continue
            seq = "/".join(
                "%.2f" % ratio(b["model"]["mse_electron_den"],
                               b["copy"]["mse_electron_den"])
                for b in entry["closed_loop_40"]["blocks"])
            parts.append("%s %s" % (cell, seq))
        print("  %-12s %s" % (case_key, "   ".join(parts)))

    # ---- B30 n=1 -----------------------------------------------------------
    print()
    print("=" * 110)
    print("B30 n=1 long-wavelength observable: ne mode-1 power (pred/truth)")
    print("=" * 110)
    rows = []
    for case_key in order:
        case = results[case_key]
        truth = case["baselines"]["truth"]["mode1_ne_power"]
        row = [case_key, "%.4g" % truth,
               "%.3f" % ratio(case["baselines"]["copy"]["mode1_ne_power"], truth)]
        for cell in CELLS:
            entry = case["cells"].get(cell)
            row.append("-" if entry is None
                       else "%.3f" % ratio(entry["mode1_ne_power"], truth))
        rows.append(row)
    table(rows, ["case", "truth", "copy", *CELLS])

    # ---- transport sign / magnitude ---------------------------------------
    print()
    print("=" * 110)
    print("ECDI-band modal transport gamma (q in 0.85-1.15), predicted vs true")
    print("=" * 110)
    rows = []
    for case_key in order:
        case = results[case_key]
        truth = case["baselines"]["copy"]["gamma_ecdi_true"]
        row = [case_key, "%.4g" % truth,
               "%.3f" % ratio(case["baselines"]["copy"]["gamma_ecdi_pred"], truth)]
        for cell in CELLS:
            entry = case["cells"].get(cell)
            row.append("-" if entry is None
                       else "%.3f" % ratio(entry["gamma_ecdi_pred"], truth))
        rows.append(row)
    table(rows, ["case", "gamma_true", "copy", *CELLS])

    # ---- gate verdicts -----------------------------------------------------
    print()
    print("=" * 110)
    print("GATE VERDICTS per case (field: all 4 channels/copy<1; "
          "transport: transport_loss/copy<1)")
    print("=" * 110)
    rows = []
    for case_key in order:
        case = results[case_key]
        row = [case_key, case["role"][:6]]
        for cell in CELLS:
            entry = case["cells"].get(cell)
            if entry is None:
                row.append("-")
                continue
            field_ok = all(
                ratio(entry[k], case["baselines"]["copy"][k]) < 1.0
                for k in FIELD_KEYS)
            transport_ok = ratio(entry["transport_loss"],
                                 case["baselines"]["copy"]["transport_loss"]) < 1.0
            mode_ok = ratio(entry["complex_mode_loss"],
                            case["baselines"]["copy"]["complex_mode_loss"]) < 1.0
            row.append("%s%s%s" % ("F" if field_ok else ".",
                                   "M" if mode_ok else ".",
                                   "T" if transport_ok else "."))
        rows.append(row)
    table(rows, ["case", "role", *CELLS])
    print()
    print("F = all field channels beat copy, M = q-complex-mode beats copy, "
          "T = modal transport beats copy")


if __name__ == "__main__":
    main()
