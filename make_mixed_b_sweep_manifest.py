import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent / "PEPAPIC" / "test" / "results"
OUTDIR = ROOT / "workdirs" / "mixed_b_sweep_manifests"


CASES = [
    {
        "case_key": "high3b_0p2mT",
        "label": "3b high magnet 0.2 mT",
        "B_mT": 0.2,
        "folder": "2D_ExB_high_magnet_dt2.5e-9_maxt50e-6_macro5",
        "stride1_h5": "global_norm_trainfixed_minmax_margin20_t0_to_t4000.h5",
        "stride2_h5": "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step2.h5",
    },
    {
        "case_key": "0p5mT",
        "label": "0.5 mT",
        "B_mT": 0.5,
        "folder": "2D_ExB_0.5mT_magnet_dt2.5e-9_maxt50e-6_macro5",
    },
    {
        "case_key": "1p0mT",
        "label": "1.0 mT",
        "B_mT": 1.0,
        "folder": "2D_ExB_1.0mT_magnet_dt2.5e-9_maxt50e-6_macro5",
    },
    {
        "case_key": "1p25mT",
        "label": "1.25 mT",
        "B_mT": 1.25,
        "folder": "2D_ExB_1.25mT_magnet_dt2.5e-9_maxt50e-6_macro5",
    },
    {
        "case_key": "1p5mT",
        "label": "1.5 mT",
        "B_mT": 1.5,
        "folder": "2D_ExB_1.5mT_magnet_dt2.5e-9_maxt50e-6_macro5",
    },
    {
        "case_key": "1p75mT",
        "label": "1.75 mT",
        "B_mT": 1.75,
        "folder": "2D_ExB_1.75mT_magnet_dt2.5e-9_maxt50e-6_macro5",
    },
]


def h5_name(case, stride):
    if stride == 1:
        return case.get(
            "stride1_h5",
            "global_norm_from_high3b_stride1_trainfixed_minmax_margin20_t0_to_t4000_training_compatible.h5",
        )
    if stride == 2:
        return case.get(
            "stride2_h5",
            "global_norm_from_high3b_trainfixed_minmax_margin20_t0_to_t4000_step2_training_compatible.h5",
        )
    raise ValueError(f"Unsupported stride: {stride}")


def build_manifest(stride, leave_out):
    leave_out = set(leave_out or [])
    cases = []
    missing = []
    for case in CASES:
        h5_path = RESULTS / case["folder"] / "SimVPv2_inputs" / h5_name(case, stride)
        if not h5_path.exists():
            missing.append(str(h5_path))
        splits = ["test"] if case["case_key"] in leave_out else ["train", "val", "test"]
        if leave_out and case["case_key"] not in leave_out:
            splits = ["train", "val"]
        cases.append(
            {
                "case_key": case["case_key"],
                "label": case["label"],
                "B_mT": case["B_mT"],
                "path": str(h5_path),
                "splits": splits,
            }
        )

    if missing:
        raise FileNotFoundError("Missing H5 files:\n" + "\n".join(missing))

    mode = "leaveout_" + "_".join(sorted(leave_out)) if leave_out else "all_cases"
    return {
        "name": f"b_sweep_stride{stride}_{mode}",
        "description": (
            "Label-free mixed PEPAPIC B-sweep dataset. "
            "Each case is split by its own frame ranges before samples are mixed."
        ),
        "stride": stride,
        "dt_frame_ns_raw": 12.5,
        "dt_frame_ns_effective": 12.5 * stride,
        "pre_seq_length": 10,
        "aft_seq_length": 10,
        "split": {
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "policy": "per-case frame-disjoint",
        },
        "normalization": "3b train-fixed minmax margin20; target cases use high3b stats",
        "leave_out": sorted(leave_out),
        "cases": cases,
    }


def default_output(stride, leave_out):
    mode = "leaveout_" + "_".join(sorted(leave_out)) if leave_out else "all_cases"
    return OUTDIR / f"b_sweep_stride{stride}_{mode}_manifest.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, choices=[1, 2], default=2)
    parser.add_argument("--leave-out", nargs="*", default=[])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    manifest = build_manifest(args.stride, args.leave_out)
    output = Path(args.output) if args.output else default_output(args.stride, args.leave_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(output)


if __name__ == "__main__":
    main()
