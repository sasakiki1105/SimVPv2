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
    # h5py で可変長文字列などの可能性
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

    # Case A: already T,C,H,W
    if "data_tchw" in f:
        X = f["data_tchw"][()]
        X = np.asarray(X, dtype=np.float32)
        assert X.ndim == 4, f"data_tchw must be 4D, got {X.shape}"
        return X, props, timesteps, layout

    # Case B: generic "data"
    if "data" not in f:
        raise KeyError("H5 must contain either 'data_tchw' or 'data'")

    X = f["data"][()]
    X = np.asarray(X, dtype=np.float32)
    assert X.ndim == 4, f"data must be 4D, got {X.shape}"

    # Heuristic using timesteps length, because your magnet case is (H,W,C,T)
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

    # Fallback by common layouts
    # If last dim looks like T (>= pre+aft) and third dim looks like C (<=16)
    if X.shape[2] <= 16 and X.shape[3] >= 10:
        # assume (H,W,C,T)
        X = np.transpose(X, (3, 2, 0, 1))
        return X, props, timesteps, layout

    raise ValueError(
        f"Cannot infer layout for data shape={X.shape}. "
        f"Please ensure timesteps exists or store as data_tchw."
    )


class _PEPAPICWindows(Dataset):
    """
    Provides (X_pre, Y_aft) windows from data_tchw (T,C,H,W).
    Also exposes:
      - in_shape = (pre_seq_length, C, H, W)  ★重要
      - mean/std (for BaseDataModule compatibility)
      - starts (window start indices)
    """
    def __init__(self, h5_path: str, pre_seq_length: int, aft_seq_length: int, split: str = "train",
                 train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1,
                 force_test_all: bool = False):
        self.h5_path = h5_path
        self.pre = int(pre_seq_length)
        self.aft = int(aft_seq_length)
        self.total = self.pre + self.aft

        with h5py.File(h5_path, "r") as f:
            X, props, timesteps, layout = _canon_to_TCHW(f)

        self.data = X  # (T,C,H,W) float32
        self.props = props
        self.timesteps = timesteps
        self.layout = layout

        T, C, H, W = self.data.shape
        self.C, self.H, self.W = C, H, W
        self.in_shape = (self.pre, C, H, W)   # ★ここが目的

        # BaseDataModule が参照するので持たせる（既に正規化済みなら mean=0 std=1 でOK）
        self.mean = 0.0
        self.std = 1.0
        self.data_name = "pepapic_h5"

        # window starts
        starts = np.arange(0, T - self.total + 1, dtype=np.int32)

        if force_test_all and split == "test":
            self.starts = starts
            return

        n = len(starts)
        n_train = int(round(train_ratio * n))
        n_val = int(round(val_ratio * n))
        n_test = max(0, n - n_train - n_val)

        train_starts = starts[:n_train]
        val_starts = starts[n_train:n_train + n_val]
        test_starts = starts[n_train + n_val:] if n_test > 0 else starts[-max(1, min(9, n)):]  # fallback

        if split == "train":
            self.starts = train_starts
        elif split in ("val", "valid", "vali"):
            self.starts = val_starts
        elif split == "test":
            self.starts = test_starts
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        st = int(self.starts[idx])
        X = self.data[st:st + self.pre]                    # (pre,C,H,W)
        Y = self.data[st + self.pre:st + self.pre + self.aft]  # (aft,C,H,W)
        return X, Y


def load_data(batch_size, val_batch_size, data_root, num_workers,
              pre_seq_length=10, aft_seq_length=10, in_shape=None, distributed=False,
              use_augment=False, use_prefetcher=False, drop_last=False,
              train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
              force_test_all=False, **kwargs):

    h5_path = data_root
    assert isinstance(h5_path, (str, os.PathLike)), f"data_root must be a path, got {type(h5_path)}"
    h5_path = os.fspath(h5_path)
    assert os.path.exists(h5_path), f"not found: {h5_path}"

    train_set = _PEPAPICWindows(h5_path, pre_seq_length, aft_seq_length, split="train",
                               train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
                               force_test_all=False)
    val_set = _PEPAPICWindows(h5_path, pre_seq_length, aft_seq_length, split="val",
                             train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
                             force_test_all=False)
    test_set = _PEPAPICWindows(h5_path, pre_seq_length, aft_seq_length, split="test",
                              train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
                              force_test_all=force_test_all)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=drop_last, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=False, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=val_batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=False, pin_memory=True)

    return train_loader, val_loader, test_loader
