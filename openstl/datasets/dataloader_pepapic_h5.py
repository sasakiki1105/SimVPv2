# openstl/datasets/dataloader_pepapic_h5.py
import os
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
    **kwargs
):
    h5_path = data_root
    assert isinstance(h5_path, (str, os.PathLike)), f"data_root must be a path, got {type(h5_path)}"
    h5_path = os.fspath(h5_path)
    assert os.path.exists(h5_path), f"not found: {h5_path}"

    train_set = _PEPAPICWindows(
        h5_path, pre_seq_length, aft_seq_length, split="train",
        train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
        force_test_all=False
    )
    val_set = _PEPAPICWindows(
        h5_path, pre_seq_length, aft_seq_length, split="val",
        train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
        force_test_all=False
    )
    test_set = _PEPAPICWindows(
        h5_path, pre_seq_length, aft_seq_length, split="test",
        train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
        force_test_all=force_test_all
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