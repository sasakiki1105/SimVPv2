from pathlib import Path


ROOT = Path(r"C:\Users\astro\research\SimVPv2")
PYTHON = Path(r"C:\Users\astro\anaconda3\envs\OpenSTL\python.exe")
CONFIG = Path(r"configs\custom\pepapic\SimVP_gSTA_pepapic.py")
H5_ROOT = Path(
    r"C:\Users\astro\research\PEPAPIC\test\results"
    r"\2D_ExB_high_magnet_dt2.5e-9_maxt50e-6_macro5"
    r"\SimVPv2_inputs"
)


TRAIN_CASES = [
    {
        "name": "stride1_tout20",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000.h5",
        "ex_name": "pepapic_simvp_gsta_highmag_macro5_tout20_trainfixed_disjoint_811_bs2_100ep",
        "aft": 20,
    },
    {
        "name": "stride1_tout30",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000.h5",
        "ex_name": "pepapic_simvp_gsta_highmag_macro5_tout30_trainfixed_disjoint_811_bs2_100ep",
        "aft": 30,
    },
    {
        "name": "stride1_tout40",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000.h5",
        "ex_name": "pepapic_simvp_gsta_highmag_macro5_tout40_trainfixed_disjoint_811_bs2_100ep",
        "aft": 40,
    },
    {
        "name": "stride2_tout20",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step2.h5",
        "ex_name": "pepapic_simvp_gsta_highmag_macro5_subsample2_tout20_trainfixed_disjoint_811_bs2_100ep",
        "aft": 20,
    },
    {
        "name": "stride3_tout14",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step3.h5",
        "ex_name": "pepapic_simvp_gsta_highmag_macro5_subsample3_tout14_trainfixed_disjoint_811_bs2_100ep",
        "aft": 14,
    },
    {
        "name": "stride2_tout40",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step2.h5",
        "ex_name": "pepapic_simvp_gsta_highmag_macro5_subsample2_tout40_trainfixed_disjoint_811_bs2_100ep",
        "aft": 40,
    },
    {
        "name": "stride4_tout20",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step4.h5",
        "ex_name": "pepapic_simvp_gsta_highmag_macro5_subsample4_tout20_trainfixed_disjoint_811_bs2_100ep",
        "aft": 20,
    },
    {
        "name": "stride2_tout80",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step2.h5",
        "ex_name": "pepapic_simvp_gsta_highmag_macro5_subsample2_tout80_trainfixed_disjoint_811_bs2_100ep",
        "aft": 80,
    },
    {
        "name": "stride4_tout40",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step4.h5",
        "ex_name": "pepapic_simvp_gsta_highmag_macro5_subsample4_tout40_trainfixed_disjoint_811_bs2_100ep",
        "aft": 40,
    },
]


def command_for(case):
    pre = 10
    total = pre + case["aft"]
    parts = [
        "$env:KMP_DUPLICATE_LIB_OK='TRUE'",
        str(PYTHON),
        "tools\\train.py",
        "--dataname", "pepapic_h5",
        "--config_file", str(CONFIG),
        "--data_root", str(case["h5"]),
        "--res_dir", ".\\workdirs",
        "--ex_name", case["ex_name"],
        "--method", "simvp",
        "--pre_seq_length", str(pre),
        "--aft_seq_length", str(case["aft"]),
        "--total_length", str(total),
        "--epoch", "100",
        "--batch_size", "2",
        "--val_batch_size", "2",
        "--num_workers", "0",
        "--gpus", "0",
    ]
    return "; ".join(parts[:1]) + "; " + " ".join(parts[1:])


def main():
    print(f"# Run from: {ROOT}")
    for case in TRAIN_CASES:
        saved = ROOT / "workdirs" / case["ex_name"] / "saved" / "preds.npy"
        status = "DONE" if saved.exists() else "MISSING"
        print()
        print(f"# {case['name']} [{status}]")
        print(command_for(case))


if __name__ == "__main__":
    main()
