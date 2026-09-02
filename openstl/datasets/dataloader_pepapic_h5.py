# openstl/datasets/dataloader_pepapic_h5.py
import json
import os
from pathlib import Path
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader


def _as_str_list(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    try:
        return [v.decode() if isinstance(v, (bytes, bytearray)) else str(v) for v in x]
    except Exception:
        return [str(x)]


def _canon_to_TCHW(f: h5py.File):
    """
    Return:
      data_tchw: np.ndarray float32 with shape (T,C,H,W)
      props: list[str] or None
      timesteps: np.ndarray or None
      layout: str or None
    """
    props = _as_str_list(f["props"][()]) if "props" in f else None
    timesteps = f["timesteps"][()] if "timesteps" in f else None
    layout = f["layout"][()] if "layout" in f else None
    if isinstance(layout, (bytes, bytearray)):
        layout = layout.decode()

    if "data_tchw" in f:
        X = f["data_tchw"][()]
        X = np.asarray(X, dtype=np.float32)
        assert X.ndim == 4, f"data_tchw must be 4D, got {X.shape}"

        if timesteps is not None:
            Tlen = len(timesteps)

            # already (T,C,H,W)
            if X.shape[0] == Tlen:
                return X, props, timesteps, layout

            # actually (H,W,C,T)
            if X.shape[3] == Tlen:
                X = np.transpose(X, (3, 2, 0, 1))  # -> (T,C,H,W)
                return X, props, timesteps, layout

            # actually (T,H,W,C)
            if X.shape[0] == Tlen and X.shape[3] <= 16:
                X = np.transpose(X, (0, 3, 1, 2))  # -> (T,C,H,W)
                return X, props, timesteps, layout

        # fallback
        if X.shape[2] <= 16 and X.shape[3] >= 10:
            X = np.transpose(X, (3, 2, 0, 1))  # assume (H,W,C,T)
            return X, props, timesteps, layout

        raise ValueError(
            f"'data_tchw' exists but shape looks wrong: {X.shape}. "
            f"Expected T on axis 0 or axis 3 matched with timesteps."
        )

    if "data" not in f:
        raise KeyError("H5 must contain either 'data_tchw' or 'data'")

    X = f["data"][()]
    X = np.asarray(X, dtype=np.float32)
    assert X.ndim == 4, f"data must be 4D, got {X.shape}"

    if timesteps is not None:
        Tlen = len(timesteps)

        # (H,W,C,T)
        if X.shape[3] == Tlen:
            X = np.transpose(X, (3, 2, 0, 1))  # -> (T,C,H,W)
            return X, props, timesteps, layout

        # (T,H,W,C)
        if X.shape[0] == Tlen:
            X = np.transpose(X, (0, 3, 1, 2))  # -> (T,C,H,W)
            return X, props, timesteps, layout

    if X.shape[2] <= 16 and X.shape[3] >= 10:
        X = np.transpose(X, (3, 2, 0, 1))
        return X, props, timesteps, layout

    raise ValueError(
        f"Cannot infer layout for data shape={X.shape}. "
        f"Please ensure timesteps exists or store as data_tchw."
    )


def _radaz_consolidated_metadata(h5_path, case):
    channels = tuple(
        str(name) for name in case.get(
            "channels", ("electron_den", "ion_den", "phi")
        )
    )
    spatial_stride = int(case.get("spatial_stride", 1))
    if spatial_stride <= 0 or 256 % spatial_stride:
        raise ValueError(
            f"RadAz spatial_stride must divide 256, got {spatial_stride}"
        )

    with h5py.File(h5_path, "r") as handle:
        missing = [name for name in channels if f"fields/{name}" not in handle]
        if missing:
            raise KeyError(f"Missing RadAz fields in {h5_path}: {missing}")
        if "axes/time_s" not in handle:
            raise KeyError(f"Missing axes/time_s in {h5_path}")
        T = int(len(handle["axes/time_s"]))
        first_shape = tuple(handle[f"fields/{channels[0]}"].shape)
        if first_shape != (T, 257, 257):
            raise ValueError(
                f"Expected consolidated RadAz shape {(T, 257, 257)}, "
                f"got {first_shape} in {h5_path}"
            )
        for name in channels[1:]:
            shape = tuple(handle[f"fields/{name}"].shape)
            if shape != first_shape:
                raise ValueError(
                    f"RadAz field shape mismatch for {name}: {shape} != {first_shape}"
                )

    native_grid = spatial_stride == 1
    valid_height = 257 if native_grid else 256 // spatial_stride
    valid_width = 256 // spatial_stride
    model_height = int(case.get("model_height", 260 if native_grid else valid_height))
    model_width = int(case.get("model_width", valid_width))
    if model_height < valid_height or model_width < valid_width:
        raise ValueError(
            f"Model shape {(model_height, model_width)} is smaller than "
            f"valid RadAz shape {(valid_height, valid_width)}"
        )

    norm_low = np.asarray(case.get("normalization_low"), dtype=np.float64)
    norm_high = np.asarray(case.get("normalization_high"), dtype=np.float64)
    if norm_low.shape != (len(channels),) or norm_high.shape != (len(channels),):
        raise ValueError(
            "Consolidated RadAz cases require normalization_low/high with "
            f"one value per channel; got {norm_low.shape} and {norm_high.shape}"
        )
    if np.any(~np.isfinite(norm_low)) or np.any(~np.isfinite(norm_high)):
        raise ValueError("RadAz normalization bounds must be finite")
    if np.any(norm_high <= norm_low):
        raise ValueError("RadAz normalization bounds must have positive spans")

    return {
        "T": T,
        "channels": channels,
        "spatial_stride": spatial_stride,
        "valid_height": valid_height,
        "valid_width": valid_width,
        "model_height": model_height,
        "model_width": model_width,
        "normalization_low": norm_low,
        "normalization_high": norm_high,
        "normalization_clip": bool(case.get("normalization_clip", False)),
        "chunk_frames": max(1, int(case.get("chunk_frames", 16))),
    }


def _load_radaz_consolidated_segment(h5_path, metadata, start, stop):
    """Load one frame-disjoint segment without materializing the 15 GiB source."""
    if not 0 <= start < stop <= metadata["T"]:
        raise ValueError(
            f"Invalid RadAz segment [{start}, {stop}) for T={metadata['T']}"
        )

    channels = metadata["channels"]
    spatial_stride = metadata["spatial_stride"]
    valid_height = metadata["valid_height"]
    valid_width = metadata["valid_width"]
    model_height = metadata["model_height"]
    model_width = metadata["model_width"]
    low = metadata["normalization_low"]
    high = metadata["normalization_high"]
    clip = metadata["normalization_clip"]
    chunk_frames = metadata["chunk_frames"]
    output = np.empty(
        (stop - start, len(channels), model_height, model_width),
        dtype=np.float32,
    )

    with h5py.File(h5_path, "r") as handle:
        for chunk_start in range(start, stop, chunk_frames):
            chunk_stop = min(chunk_start + chunk_frames, stop)
            out_start = chunk_start - start
            out_stop = chunk_stop - start
            count = chunk_stop - chunk_start
            for channel_index, name in enumerate(channels):
                if spatial_stride == 1:
                    raw = np.asarray(
                        handle[f"fields/{name}"][chunk_start:chunk_stop, :257, :256],
                        dtype=np.float64,
                    )
                else:
                    source_values = np.asarray(
                        handle[f"fields/{name}"][chunk_start:chunk_stop, :256, :256],
                        dtype=np.float64,
                    )
                    raw = source_values.reshape(
                        count,
                        valid_height,
                        spatial_stride,
                        valid_width,
                        spatial_stride,
                    ).mean(axis=(2, 4))
                normalized = (raw - low[channel_index]) / (
                    high[channel_index] - low[channel_index]
                )
                if clip:
                    normalized = np.clip(normalized, 0.0, 1.0)
                target = output[out_start:out_stop, channel_index]
                target[:, :valid_height, :valid_width] = normalized.astype(
                    np.float32, copy=False
                )
                if model_height > valid_height:
                    target[:, valid_height:, :valid_width] = target[
                        :, valid_height - 1 : valid_height, :valid_width
                    ]
                if model_width > valid_width:
                    target[:, :, valid_width:] = target[
                        :, :, valid_width - 1 : valid_width
                    ]

    if not np.all(np.isfinite(output)):
        raise ValueError(f"Non-finite values after loading {h5_path} [{start}, {stop})")
    return output


def _build_disjoint_starts(T, total, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split by FRAME ranges, not by start ranges.
    No sample in train/val/test shares any frame with another split.
    """
    if total <= 0:
        raise ValueError(f"total must be positive, got {total}")
    if T < total:
        raise ValueError(f"T={T} is smaller than total window length={total}")

    s = train_ratio + val_ratio + test_ratio
    if not np.isclose(s, 1.0):
        raise ValueError(f"train/val/test ratios must sum to 1.0, got {s}")

    train_end = int(np.floor(T * train_ratio))                 # exclusive
    val_end = int(np.floor(T * (train_ratio + val_ratio)))     # exclusive

    def starts_in_range(frame_start, frame_end):
        last_start = frame_end - total
        if last_start < frame_start:
            return np.empty((0,), dtype=np.int32)
        return np.arange(frame_start, last_start + 1, dtype=np.int32)

    train_starts = starts_in_range(0, train_end)
    val_starts = starts_in_range(train_end, val_end)
    test_starts = starts_in_range(val_end, T)

    info = {
        "T": T,
        "total": total,
        "train_frame_range": (0, train_end - 1),
        "val_frame_range": (train_end, val_end - 1),
        "test_frame_range": (val_end, T - 1),
        "n_train": len(train_starts),
        "n_val": len(val_starts),
        "n_test": len(test_starts),
    }
    return train_starts, val_starts, test_starts, info


class _PEPAPICWindows(Dataset):
    """
    Provides (X_pre, Y_aft) windows from data_tchw (T,C,H,W).
    Split is FRAME-disjoint: train/val/test do not share any frame.
    """
    def __init__(
        self,
        h5_path: str,
        pre_seq_length: int,
        aft_seq_length: int,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        force_test_all: bool = False,
    ):
        self.h5_path = h5_path
        self.pre = int(pre_seq_length)
        self.aft = int(aft_seq_length)
        self.total = self.pre + self.aft

        with h5py.File(h5_path, "r") as f:
            X, props, timesteps, layout = _canon_to_TCHW(f)

        self.data = X  # (T,C,H,W)
        self.props = props
        self.timesteps = timesteps
        self.layout = layout

        T, C, H, W = self.data.shape
        self.C, self.H, self.W = C, H, W
        self.in_shape = (self.pre, C, H, W)

        self.mean = 0.0
        self.std = 1.0
        self.data_name = "pepapic_h5"

        train_starts, val_starts, test_starts, info = _build_disjoint_starts(
            T, self.total,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
        self.split_info = info

        if force_test_all and split == "test":
            self.starts = test_starts
        elif split == "train":
            self.starts = train_starts
        elif split in ("val", "valid", "vali"):
            self.starts = val_starts
        elif split == "test":
            self.starts = test_starts
        else:
            raise ValueError(f"Unknown split: {split}")

        if len(self.starts) == 0:
            raise ValueError(
                f"No samples for split={split}. split_info={self.split_info}"
            )

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        st = int(self.starts[idx])
        X = self.data[st:st + self.pre]                         # (pre,C,H,W)
        Y = self.data[st + self.pre:st + self.pre + self.aft]  # (aft,C,H,W)
        return X, Y


def _decode_h5_scalar(value):
    if isinstance(value, (bytes, bytearray)):
        return value.decode()
    return value


def _load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if "cases" not in manifest:
        raise ValueError(f"PEPAPIC manifest must contain a 'cases' list: {path}")
    base = Path(path).resolve().parent
    cases = []
    for i, case in enumerate(manifest["cases"]):
        if "path" not in case:
            raise ValueError(f"manifest cases[{i}] is missing 'path'")
        case = dict(case)
        h5_path = Path(case["path"])
        if not h5_path.is_absolute():
            h5_path = base / h5_path
        case["path"] = str(h5_path)
        case.setdefault("case_key", h5_path.stem)
        case.setdefault("label", case["case_key"])
        case.setdefault("splits", ["train", "val", "test"])
        cases.append(case)
    manifest = dict(manifest)
    manifest["cases"] = cases
    return manifest


def _split_range_from_info(info, split):
    if split == "train":
        return info["train_frame_range"]
    if split in ("val", "valid", "vali"):
        return info["val_frame_range"]
    if split == "test":
        return info["test_frame_range"]
    raise ValueError(f"Unknown split: {split}")


def _split_starts_from_info(starts_tuple, split):
    train_starts, val_starts, test_starts = starts_tuple
    if split == "train":
        return train_starts
    if split in ("val", "valid", "vali"):
        return val_starts
    if split == "test":
        return test_starts
    raise ValueError(f"Unknown split: {split}")


def _condition_channel_names(condition_channels):
    if condition_channels is None:
        return []
    if isinstance(condition_channels, str):
        if condition_channels.strip().lower() in ("", "none", "off", "false", "0"):
            return []
        return [name.strip() for name in condition_channels.split(",") if name.strip()]
    return [str(name) for name in condition_channels]


def _condition_values(
    case,
    condition_channels,
    b_scale_mT=1.0,
    dt_scale_ns=1.0,
    normalization=None,
):
    values = []
    for name in condition_channels:
        key = name.lower()
        if key in ("b", "b_m_t", "b_mt", "bz", "b_z", "bz_mt"):
            values.append(float(case["B_mT"]) / float(b_scale_mT))
        elif key in ("dt", "dt_ns", "dt_frame_ns", "dt_eff_ns"):
            dt_ns = case.get("dt_eff_ns", case.get("dt_frame_ns_effective", case.get("dt_frame_ns")))
            if dt_ns is None:
                raise ValueError(f"Condition {name} requires dt_frame_ns/dt_eff_ns in case metadata: {case}")
            values.append(float(dt_ns) / float(dt_scale_ns))
        elif key in ("log_ve", "log_v_e", "log_e_over_b", "log_e_b"):
            electric_v_m = float(case["Ez_kVm"]) * 1.0e3
            magnetic_t = float(case["B_mT"]) * 1.0e-3
            values.append(float(np.log(electric_v_m / magnetic_t)))
        elif key in ("log_n0", "log_mode_n0", "log_ecdi_n0"):
            electric_v_m = float(case["Ez_kVm"]) * 1.0e3
            magnetic_t = float(case["B_mT"]) * 1.0e-3
            ly_m = float(case.get("Ly_m", 1.28e-2))
            electron_charge = 1.602176634e-19
            electron_mass = 9.1093837015e-31
            mode_n0 = (
                electron_charge * magnetic_t**2 * ly_m
                / (2.0 * np.pi * electron_mass * electric_v_m)
            )
            values.append(float(np.log(mode_n0)))
        else:
            raise ValueError(f"Unknown PEPAPIC condition channel: {name}")
    values = np.asarray(values, dtype=np.float64)
    if normalization:
        names = [str(value) for value in normalization.get("names", [])]
        if names != list(condition_channels):
            raise ValueError(
                "condition_normalization names must exactly match condition_channels: "
                f"{names} != {list(condition_channels)}"
            )
        mean = np.asarray(normalization["mean"], dtype=np.float64)
        std = np.asarray(normalization["std"], dtype=np.float64)
        if mean.shape != values.shape or std.shape != values.shape:
            raise ValueError(
                f"Condition normalization shape mismatch: values={values.shape}, "
                f"mean={mean.shape}, std={std.shape}"
            )
        if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(std)) or np.any(std <= 0.0):
            raise ValueError("Condition normalization mean/std must be finite with std > 0")
        values = (values - mean) / std
    return values.astype(np.float32)


class _PEPAPICMultiCaseWindows(Dataset):
    """
    Provides (X_pre, Y_aft) windows from a manifest of multiple H5 files.

    Each case is split by its own frame ranges before samples are mixed.
    This prevents nearly identical neighbouring windows from leaking across
    train/val/test through a global random split.
    """
    def __init__(
        self,
        manifest_path: str,
        pre_seq_length: int,
        aft_seq_length: int,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        force_test_all: bool = False,
        condition_channels=None,
        condition_b_scale_mT: float = 1.0,
        condition_dt_scale_ns: float = 1.0,
    ):
        self.manifest_path = manifest_path
        self.manifest = _load_manifest(manifest_path)
        self.pre = int(pre_seq_length)
        self.aft = int(aft_seq_length)
        self.total = self.pre + self.aft
        self.condition_channels = _condition_channel_names(
            condition_channels
            if condition_channels is not None
            else self.manifest.get("condition_channels")
        )
        self.condition_b_scale_mT = float(
            self.manifest.get("condition_b_scale_mT", condition_b_scale_mT)
        )
        self.condition_dt_scale_ns = float(
            self.manifest.get("condition_dt_scale_ns", condition_dt_scale_ns)
        )
        self.condition_normalization = self.manifest.get("condition_normalization")
        self.data = []
        self.conditions = []
        self.samples = []
        self.case_info = []
        self.case_keys = []
        self.case_labels = []
        self.split = split

        ref_shape = None
        ref_props = None
        ref_layout = None
        for case_index, case in enumerate(self.manifest["cases"]):
            allowed_splits = case.get("splits", ["train", "val", "test"])
            if isinstance(allowed_splits, str):
                allowed_splits = [allowed_splits]
            allowed_splits = set(str(s).lower() for s in allowed_splits)
            split_key = "val" if split in ("val", "valid", "vali") else split
            if "all" not in allowed_splits and split_key not in allowed_splits:
                continue

            h5_path = case["path"]
            case_format = str(case.get("format", "canonical_h5")).lower()
            is_radaz_consolidated = case_format in (
                "radaz_consolidated",
                "radaz_consolidated_fields",
            )
            radaz_metadata = None
            X = None
            if is_radaz_consolidated:
                radaz_metadata = _radaz_consolidated_metadata(h5_path, case)
                T = radaz_metadata["T"]
                C = len(radaz_metadata["channels"])
                H = radaz_metadata["model_height"]
                W = radaz_metadata["model_width"]
                props = list(radaz_metadata["channels"])
                timesteps = None
                layout = "data_tchw[T,C,x_radial,y_azimuthal]"
            else:
                with h5py.File(h5_path, "r") as f:
                    X, props, timesteps, layout = _canon_to_TCHW(f)
                T, C, H, W = X.shape
            if ref_shape is None:
                ref_shape = (C, H, W)
                ref_props = props
                ref_layout = layout
            elif (C, H, W) != ref_shape:
                raise ValueError(
                    f"All manifest cases must have the same C,H,W. "
                    f"Expected {ref_shape}, got {(C, H, W)} for {h5_path}"
                )

            starts_tuple = _build_disjoint_starts(
                T, self.total,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
            )
            train_starts, val_starts, test_starts, info = starts_tuple
            starts = _split_starts_from_info((train_starts, val_starts, test_starts), split)

            if force_test_all and split == "test":
                starts = np.arange(0, T - self.total + 1, dtype=np.int32)

            if len(starts) == 0:
                continue

            # Keep only the split's required frame segment to avoid tripling
            # memory for train/val/test in multi-case runs.
            if force_test_all and split == "test":
                seg_start, seg_end_exclusive = 0, T
                local_starts = starts
            else:
                frame_start, frame_end = _split_range_from_info(info, split)
                seg_start = int(frame_start)
                seg_end_exclusive = int(frame_end) + 1
                local_starts = starts - seg_start

            local_case_index = len(self.data)
            if is_radaz_consolidated:
                segment = _load_radaz_consolidated_segment(
                    h5_path,
                    radaz_metadata,
                    seg_start,
                    seg_end_exclusive,
                )
            else:
                segment = np.asarray(X[seg_start:seg_end_exclusive], dtype=np.float32)
            self.data.append(segment)
            self.conditions.append(
                _condition_values(
                    case,
                    self.condition_channels,
                    b_scale_mT=self.condition_b_scale_mT,
                    dt_scale_ns=self.condition_dt_scale_ns,
                    normalization=self.condition_normalization,
                )
            )
            self.case_keys.append(str(case.get("case_key", f"case{case_index}")))
            self.case_labels.append(str(case.get("label", self.case_keys[-1])))
            self.case_info.append({
                "case_index": case_index,
                "local_case_index": local_case_index,
                "case_key": self.case_keys[-1],
                "label": self.case_labels[-1],
                "path": h5_path,
                "format": case_format,
                "T": int(T),
                "segment_frame_range": (seg_start, seg_end_exclusive - 1),
                "n_samples": int(len(local_starts)),
                "split_info": info,
            })
            for st in local_starts:
                self.samples.append((local_case_index, int(st)))

            if X is not None:
                del X

        if ref_shape is None:
            raise ValueError(f"No cases are usable for split={split}: {manifest_path}")

        C, H, W = ref_shape
        self.base_C = C
        self.C, self.H, self.W = C + len(self.condition_channels), H, W
        self.target_C = C
        self.in_shape = (self.pre, self.C, H, W)
        self.props = ref_props
        if self.condition_channels:
            self.input_props = list(ref_props or []) + [f"condition_{name}" for name in self.condition_channels]
        else:
            self.input_props = ref_props
        self.layout = ref_layout
        self.mean = 0.0
        self.std = 1.0
        self.data_name = "pepapic_h5"
        self.split_info = {
            "manifest": manifest_path,
            "split": split,
            "total": self.total,
            "n_cases": len(self.case_info),
            "n_samples": len(self.samples),
            "condition_channels": self.condition_channels,
            "condition_normalization": self.condition_normalization,
            "cases": self.case_info,
        }

        if len(self.samples) == 0:
            raise ValueError(f"No samples for split={split}. split_info={self.split_info}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        case_index, st = self.samples[idx]
        case_data = self.data[case_index]
        X = case_data[st:st + self.pre]
        Y = case_data[st + self.pre:st + self.pre + self.aft]
        if self.condition_channels:
            values = self.conditions[case_index]
            cond = np.broadcast_to(
                values.reshape(1, len(values), 1, 1),
                (self.pre, len(values), self.H, self.W),
            ).astype(np.float32, copy=False)
            X = np.concatenate([X, cond], axis=1)
        return X, Y


def load_data(
    batch_size,
    val_batch_size,
    data_root,
    num_workers,
    pre_seq_length=10,
    aft_seq_length=10,
    in_shape=None,
    distributed=False,
    use_augment=False,
    use_prefetcher=False,
    drop_last=False,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    force_test_all=False,
    pepapic_condition_channels=None,
    pepapic_condition_b_scale_mT=1.0,
    pepapic_condition_dt_scale_ns=1.0,
    **kwargs
):
    data_path = data_root
    assert isinstance(data_path, (str, os.PathLike)), f"data_root must be a path, got {type(data_path)}"
    data_path = os.fspath(data_path)
    assert os.path.exists(data_path), f"not found: {data_path}"

    dataset_cls = _PEPAPICMultiCaseWindows if str(data_path).lower().endswith(".json") else _PEPAPICWindows

    common = dict(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    if dataset_cls is _PEPAPICMultiCaseWindows:
        common.update(
            condition_channels=pepapic_condition_channels,
            condition_b_scale_mT=pepapic_condition_b_scale_mT,
            condition_dt_scale_ns=pepapic_condition_dt_scale_ns,
        )

    train_set = dataset_cls(
        data_path, pre_seq_length, aft_seq_length, split="train",
        force_test_all=False,
        **common,
    )
    val_set = dataset_cls(
        data_path, pre_seq_length, aft_seq_length, split="val",
        force_test_all=False,
        **common,
    )
    test_set = dataset_cls(
        data_path, pre_seq_length, aft_seq_length, split="test",
        force_test_all=force_test_all,
        **common,
    )

    print("[PEPAPIC split info]")
    print(train_set.split_info)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=drop_last, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=val_batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False, pin_memory=True
    )

    return train_loader, val_loader, test_loader
