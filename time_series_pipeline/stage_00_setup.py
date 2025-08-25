import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any

from utils.helpers.common import DataConfig, RunConfig
from utils.helpers.pipeline import Step
from utils.helpers.general_utils import save_json, train_val_test_split_index


class Stage00_Setup(Step):
    name = "stage_00_setup"
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        cfg_data: DataConfig = self.cfg["data"]
        cfg_run: RunConfig = self.cfg["run"]
        step_dir = context["_current_step_dir"]

        # Load or synthesize data
        if cfg_data.data_source and os.path.exists(cfg_data.data_source):
            df = pd.read_csv(cfg_data.data_source)
            assert {cfg_data.time_col, cfg_data.target_col}.issubset(df.columns)
            df[cfg_data.time_col] = pd.to_datetime(df[cfg_data.time_col])
            df = df.sort_values(cfg_data.time_col).reset_index(drop=True)
        else:
            # Synthetic demo: trend + seasonality + noise
            T = cfg_data.length
            rng = np.random.default_rng(42)
            t = np.arange(T)
            trend = 0.002 * t
            season_w = np.sin(2*np.pi*t/7) + 0.5*np.sin(2*np.pi*t/30)
            noise = rng.normal(0, 0.2, T)
            y = 2.0 + trend + season_w + noise
            dt_index = pd.date_range("2020-01-01", periods=T, freq=cfg_data.freq)
            df = pd.DataFrame({cfg_data.time_col: dt_index, cfg_data.target_col: y})

        # Splits
        T = len(df)
        idx = train_val_test_split_index(
            T, cfg_data.train_ratio, cfg_data.val_ratio
        )

        # Save artifacts
        os.makedirs(step_dir, exist_ok=True)
        df_path = os.path.join(step_dir, "series.csv")
        df.to_csv(df_path, index=False)
        save_json(os.path.join(step_dir, "splits.json"), idx)

        # Shape prints
        print("[Stage_00] DataFrame shape:", df.shape)
        print("[Stage_00] Split indices:", idx)

        # Plot: raw series
        plt.figure(figsize=(10, 3))
        plt.plot(df[cfg_data.time_col], df[cfg_data.target_col])
        plt.title("Stage_00: Raw Time Series")
        plt.xlabel("Time")
        plt.ylabel("y")
        plt.tight_layout()
        raw_plot_path = os.path.join(step_dir, "raw_series.png")
        plt.savefig(raw_plot_path, dpi=150)
        plt.close()

        # Update context
        context.update({
            "df": df,
            "splits": idx,
            "cfg_data": cfg_data,
            "cfg_run": cfg_run,
            "artifacts_stage_00": {"series_csv": df_path, "raw_plot": raw_plot_path},
        })
        return context