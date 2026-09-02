import argparse
from pathlib import Path

import h5py
import numpy as np


TIME_KEYS = {"timesteps"}


def _copy_dataset(src, dst, key, step, source_t_len, train_ratio):
    ds = src[key]

    if key in TIME_KEYS:
        dst.create_dataset(key, data=ds[()][::step])
        return

    if key in {"data", "data_tchw"} and ds.ndim == 4:
        shape = ds.shape
        if shape[0] == source_t_len:
            data = ds[::step, ...]
        elif shape[-1] == source_t_len:
            data = ds[..., ::step]
        else:
            raise ValueError(
                f"Cannot infer time axis for {key}: shape={shape}, T={source_t_len}"
            )
        dst.create_dataset(key, data=data, dtype=ds.dtype)
        return

    if key == "train_t_first":
        timesteps = src["timesteps"][()][::step]
        dst.create_dataset(key, data=np.asarray(timesteps[0], dtype=ds.dtype))
        return

    if key == "train_t_last":
        timesteps = src["timesteps"][()][::step]
        train_end = int(np.floor(len(timesteps) * train_ratio))
        last = timesteps[max(train_end - 1, 0)]
        dst.create_dataset(key, data=np.asarray(last, dtype=ds.dtype))
        return

    dst.create_dataset(key, data=ds[()])


def make_subsample_h5(src_path, dst_path, step, overwrite=False):
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    if not src_path.exists():
        raise FileNotFoundError(src_path)
    if dst_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {dst_path}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        dst_path.unlink()

    with h5py.File(src_path, "r") as src:
        if "timesteps" not in src:
            raise KeyError("Source H5 must contain timesteps")
        source_t_len = len(src["timesteps"])
        train_ratio = float(src["train_ratio"][()]) if "train_ratio" in src else 0.8

        with h5py.File(dst_path, "w") as dst:
            for attr_key, attr_value in src.attrs.items():
                dst.attrs[attr_key] = attr_value
            for key in src.keys():
                _copy_dataset(src, dst, key, step, source_t_len, train_ratio)

    with h5py.File(dst_path, "r") as out:
        timesteps = out["timesteps"][()]
        data_key = "data_tchw" if "data_tchw" in out else "data"
        print(f"[OK] wrote {dst_path}")
        print(f"  {data_key}.shape = {out[data_key].shape}")
        print(f"  timesteps = {timesteps[0]}..{timesteps[-1]} ({len(timesteps)} frames)")
        if "train_t_first" in out and "train_t_last" in out:
            print(f"  train_t = {out['train_t_first'][()]}..{out['train_t_last'][()]}")


def main():
    parser = argparse.ArgumentParser(description="Create a time-subsampled PEPAPIC H5 file.")
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.step <= 0:
        raise ValueError("--step must be positive")
    make_subsample_h5(args.src, args.dst, args.step, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
