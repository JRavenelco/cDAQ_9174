from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset


# Column keys we care about
COL_T = "t"   # seconds
COL_Y = "y"   # meters
COL_I = "i"   # amperes (coil current)


@dataclass
class DataStats:
    t_min: float
    t_max: float
    t_mu: float
    t_std: float
    y_mu: float
    y_std: float
    i_mu: float
    i_std: float

    @classmethod
    def from_arrays(cls, t: np.ndarray, y: np.ndarray, i: np.ndarray) -> "DataStats":
        eps = 1e-8
        return cls(
            t_min=float(np.min(t)),
            t_max=float(np.max(t)),
            t_mu=float(np.mean(t)),
            t_std=float(max(np.std(t), eps)),
            y_mu=float(np.mean(y)),
            y_std=float(max(np.std(y), eps)),
            i_mu=float(np.mean(i)),
            i_std=float(max(np.std(i), eps)),
        )


def load_monit_txt(fp: str) -> pd.DataFrame:
    """Load MONIT-like txt (tab-separated, no header) into a DataFrame with canonical columns.
    Expected columns: t, yd, y, ied, ie, u. We will keep only t, y, ie.
    """
    df = pd.read_csv(
        fp, sep=r"\t|→", engine="python", header=None, dtype=float
    )
    # Drop the index column if present (first column before the arrow)
    if 0 in df.columns and df.shape[1] >= 6:
        # Heuristic: when an extra leading column exists, it is a non-numeric index
        # Keep the last 6 numeric columns
        df = df.iloc[:, -6:]
    df.columns = ["t", "yd", "y", "ied", "ie", "u"]
    return df[["t", "y", "ie", "u"]].rename(columns={"ie": COL_I, "y": COL_Y, "t": COL_T})


class MonitDataset(Dataset):
    """PyTorch Dataset from MONIT-like txt logs.

    - Stores t normalized (z-score) as model input.
    - Targets are y and i in physical units (for Loss_data anchoring).
    - Provides DataStats for (optional) de/normalization in the model.
    """
    def __init__(self, fp: str):
        super().__init__()
        df = load_monit_txt(fp)
        t = df[COL_T].to_numpy(dtype=np.float32)
        y = df[COL_Y].to_numpy(dtype=np.float32)
        i = df[COL_I].to_numpy(dtype=np.float32)
        u = df["u"].to_numpy(dtype=np.float32)

        self.stats = DataStats.from_arrays(t, y, i)
        self.t_norm = ((t - self.stats.t_mu) / self.stats.t_std).astype(np.float32)
        self.y = y.astype(np.float32)
        self.i = i.astype(np.float32)
        self.u = u.astype(np.float32)

    def __len__(self) -> int:
        return self.t_norm.shape[0]

    def __getitem__(self, idx: int):
        t = torch.tensor(self.t_norm[idx:idx+1])  # shape [1]
        y = torch.tensor(self.y[idx:idx+1])
        i = torch.tensor(self.i[idx:idx+1])
        u = torch.tensor(self.u[idx:idx+1])
        return t, y, i, u


def concat_datasets(paths: List[str]) -> MonitDataset:
    """Concatenate multiple MONIT-like files into a single dataset using a common normalization.
    Returns a MonitDataset-like object with combined arrays.
    """
    # Load and concat
    frames = [load_monit_txt(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    t = df[COL_T].to_numpy(dtype=np.float32)
    y = df[COL_Y].to_numpy(dtype=np.float32)
    i = df[COL_I].to_numpy(dtype=np.float32)
    u = df["u"].to_numpy(dtype=np.float32)

    stats = DataStats.from_arrays(t, y, i)
    t_norm = ((t - stats.t_mu) / stats.t_std).astype(np.float32)

    ds = MonitDataset.__new__(MonitDataset)
    Dataset.__init__(ds)
    ds.stats = stats
    ds.t_norm = t_norm
    ds.y = y
    ds.i = i
    ds.u = u
    return ds


def build_datasets(ds: Dataset, val_split: float, seed: int) -> Tuple[Subset, Subset]:
    n = len(ds)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(n * val_split)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return Subset(ds, train_idx.tolist()), Subset(ds, val_idx.tolist())
