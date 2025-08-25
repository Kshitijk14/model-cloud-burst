import json
import math
import numpy as np
from typing import Dict, Any, Optional, Tuple


def save_array(path: str, arr: np.ndarray):
    np.save(path, arr)

def save_json(path: str, data: Dict[str, Any]):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def train_val_test_split_index(T: int, train_ratio: float, val_ratio: float):
    t_train = int(T * train_ratio)
    t_val = int(T * (train_ratio + val_ratio))
    idx = {
        "train": (0, t_train),
        "val": (t_train, t_val),
        "test": (t_val, T),
    }
    return idx

class Standardizer:
    def __init__(self):
        self.mean_: Optional[float] = None
        self.std_: Optional[float] = None
    def fit(self, x: np.ndarray):
        self.mean_ = float(np.mean(x))
        self.std_ = float(np.std(x) + 1e-8)
        return self
    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_
    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std_ + self.mean_

class MinMaxScaler:
    def __init__(self, feature_range=(0.0, 1.0)):
        self.min_ = None
        self.max_ = None
        self.fr = feature_range
    def fit(self, x: np.ndarray):
        self.min_ = float(np.min(x))
        self.max_ = float(np.max(x))
        return self
    def transform(self, x: np.ndarray) -> np.ndarray:
        low, high = self.fr
        denom = (self.max_ - self.min_) + 1e-8
        return (x - self.min_) / denom * (high - low) + low
    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        low, high = self.fr
        return (x - low) / ((high - low) + 1e-8) * (self.max_ - self.min_) + self.min_

def build_cyclic_features(t: np.ndarray, periods: Tuple[int, ...]) -> np.ndarray:
    """t: integer time index (0..T-1)"""
    feats = []
    for P in periods:
        theta = 2.0 * math.pi * t / P
        feats.append(np.sin(theta))
        feats.append(np.cos(theta))
    if len(feats) == 0:
        return np.empty((len(t), 0))
    return np.vstack(feats).T  # (T, 2*len(periods))

def build_lag_features(y: np.ndarray, max_lag: int) -> np.ndarray:
    T = len(y)
    if max_lag <= 0:
        return np.empty((T, 0))
    mats = []
    for k in range(1, max_lag + 1):
        lag = np.concatenate([np.full(k, np.nan), y[:-k]])
        mats.append(lag)
    return np.vstack(mats).T  # (T, max_lag)