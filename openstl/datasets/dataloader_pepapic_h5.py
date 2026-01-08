import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class _PEPAPICWindows(Dataset):
    """
    H5 dataset: data_tchw (T,C,H,W)
    returns:
      x: (Tin, C, H, W)
      y: (Tout, C, H, W)
    """
    def __init__(self, data_tchw: np.ndarray, pre_seq_length: int, aft_seq_length: int, start_indices):
        self.data = data_tchw  # (T,C,H,W) float32
        self.pre = int(pre_seq_length)
        self.aft = int(aft_seq_length)
        self.seq_len = self.pre + self.aft
        self.starts = list(start_indices)
        # ---- required by BaseDataModule (openstl/datasets/base_data.py) ----
        self.data_name = "pepapic_h5"

        # すでに [0,1] 正規化済みのデータを食わせるので、とりあえずダミーでOK
        # （逆正規化等を使わないならこれで十分）
        self.mean = 0.0
        self.std  = 1.0


    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        e = s + self.seq_len
        arr = self.data[s:e]          # (seq,C,H,W)
        x = arr[:self.pre]            # (pre,C,H,W)
        y = arr[self.pre:]            # (aft,C,H,W)
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


def load_data(batch_size, val_batch_size, data_root, num_workers, **kwargs):
    """
    OpenSTLの他dataloaderと同じシグネチャに合わせた版。
    data_root: H5 file path

    kwargs:
      - pre_seq_length (default 10)
      - aft_seq_length (default 10)
      - stride (default 1)
      - train_ratio (default 0.8)
      - val_ratio (default 0.1)
      - drop_last (default False)
    """
    h5_path = data_root
    assert os.path.exists(h5_path), f"not found: {h5_path}"

    pre = int(kwargs.get("pre_seq_length", 10))
    aft = int(kwargs.get("aft_seq_length", 10))
    stride = int(kwargs.get("stride", 1))
    train_ratio = float(kwargs.get("train_ratio", 0.8))
    val_ratio   = float(kwargs.get("val_ratio", 0.1))
    drop_last   = bool(kwargs.get("drop_last", False))

    with h5py.File(h5_path, "r") as f:
        # まず data_tchw を優先。無ければ data を読んで変換する。
        if "data_tchw" in f:
            data = f["data_tchw"][:]  # (T,C,H,W)
        elif "data" in f:
            d = f["data"][:]          # あなたの現状: (H,W,C,T)
            # (H,W,C,T) -> (T,C,H,W)
            data = np.transpose(d, (3, 2, 0, 1))
        else:
            raise KeyError("H5 must contain 'data_tchw' or 'data'")

    data = np.ascontiguousarray(data).astype(np.float32)
    T = data.shape[0]
    seq_len = pre + aft

    starts = list(range(0, T - seq_len + 1, stride))
    n = len(starts)
    if n <= 0:
        raise ValueError(f"No samples. T={T}, pre={pre}, aft={aft}")

    n_train = max(1, int(n * train_ratio))
    n_val = int(n * val_ratio)
    # 最低1はtestに残す
    if n - n_train - n_val < 1:
        n_val = max(0, n - n_train - 1)

    train_idx = starts[:n_train]
    val_idx   = starts[n_train:n_train+n_val] if n_val > 0 else []
    test_idx  = starts[n_train+n_val:] if (n_train+n_val) < n else starts[-1:]

    train_set = _PEPAPICWindows(data, pre, aft, train_idx)
    val_set   = _PEPAPICWindows(data, pre, aft, val_idx) if len(val_idx) else None
    test_set  = _PEPAPICWindows(data, pre, aft, test_idx)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=drop_last
    )
    val_loader = DataLoader(
        val_set, batch_size=val_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    ) if val_set else None
    test_loader = DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

    return train_loader, val_loader, test_loader
