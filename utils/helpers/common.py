from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class DataConfig:
    data_source: Optional[str] = None  # path to CSV with two columns: ["ds","y"] (optional for now)
    time_col: str = "ds"
    target_col: str = "y"
    freq: str = "D"  # used only for synthetic/demo generation
    length: int = 1000  # used only for synthetic/demo
    train_ratio: float = 0.7
    val_ratio: float = 0.15  # test = 1 - train - val

@dataclass
class PreprocessConfig:
    periods: Tuple[int, ...] = (7, 30, 365)  # for cyclic features; adapt to your freq
    max_lag: int = 7  # number of past lags to include as features
    use_minmax: bool = False  # set True if you want MinMax after standardization

@dataclass
class RunConfig:
    lookback: int = 96
    horizon: int = 24
    artifacts_root: str = "artifacts/ts_eemd_pipeline"
    run_name: str = "demo_run"