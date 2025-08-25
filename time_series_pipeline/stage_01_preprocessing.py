import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

from utils.helpers.common import DataConfig, PreprocessConfig
from utils.helpers.pipeline import Step
from utils.helpers.general_utils import (
    save_array,
    save_json,  
    Standardizer,
    MinMaxScaler,
    build_cyclic_features, 
    build_lag_features,
)


class Stage01_Preprocess(Step):
    name = "stage_01_preprocess"
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        cfg_prep: PreprocessConfig = self.cfg["prep"]
        cfg_data: DataConfig = context["cfg_data"]
        step_dir = context["_current_step_dir"]
        df: pd.DataFrame = context["df"]
        splits: Dict[str, Tuple[int, int]] = context["splits"]

        y = df[cfg_data.target_col].values.astype(np.float64)
        t = np.arange(len(y))

        # Build feature blocks
        X_cyc = build_cyclic_features(t, cfg_prep.periods)  # (T, 2*len(periods))
        X_lag = build_lag_features(y, cfg_prep.max_lag)     # (T, max_lag)

        # Combine blocks
        X = np.concatenate([a for a in [X_cyc, X_lag] if a.shape[1] > 0], axis=1) if (X_cyc.size or X_lag.size) else np.empty((len(y),0))

        # Standardize target, fit on train only
        i0, i1 = splits["train"]
        stdzr = Standardizer().fit(y[i0:i1])
        y_std = stdzr.transform(y)

        # Optional MinMax for stability
        if cfg_prep.use_minmax:
            mm = MinMaxScaler().fit(y_std[i0:i1])
            y_scl = mm.transform(y_std)
            scaler_kind = "standardize+minmax"
        else:
            mm = None
            y_scl = y_std
            scaler_kind = "standardize-only"

        # Masks for NaN rows due to lags
        valid_rows = ~np.any(np.isnan(X), axis=1) if X.size else np.ones(len(y), dtype=bool)
        X_valid = X[valid_rows]
        y_scl_valid = y_scl[valid_rows]

        # Save artifacts
        npy = lambda fn, arr: save_array(os.path.join(step_dir, fn), arr)
        os.makedirs(step_dir, exist_ok=True)
        npy("X.npy", X_valid)
        npy("y_scl.npy", y_scl_valid)
        npy("y_raw.npy", y)
        save_json(os.path.join(step_dir, "preprocess_meta.json"), {
            "periods": cfg_prep.periods,
            "max_lag": cfg_prep.max_lag,
            "scaler": scaler_kind,
            "valid_rows_start_index": int(np.argmax(valid_rows)),  # first True index
            "X_shape": list(X_valid.shape),
            "y_shape": [int(y_scl_valid.shape[0])],
        })

        # Prints: shapes
        print("[Stage_01] Raw y shape:", y.shape)
        print("[Stage_01] Cyclic features shape:", X_cyc.shape)
        print("[Stage_01] Lag features shape:", X_lag.shape)
        print("[Stage_01] Combined X shape (with NaNs):", X.shape)
        print("[Stage_01] Valid rows after lag-drop:", X_valid.shape[0])
        print("[Stage_01] Final X shape:", X_valid.shape, "| Final y shape:", y_scl_valid.shape)

        # Plot: standardized/scaled vs raw
        plt.figure(figsize=(10, 3))
        plt.plot(df[cfg_data.time_col], y, label="raw")
        plt.plot(df[cfg_data.time_col], y_scl, label="standardized/minmax")
        plt.title("Stage_01: Raw vs. Standardized/Scaled Target")
        plt.xlabel("Time")
        plt.ylabel("value")
        plt.legend()
        plt.tight_layout()
        scaled_plot_path = os.path.join(step_dir, "scaled_vs_raw.png")
        plt.savefig(scaled_plot_path, dpi=150)
        plt.close()

        # Plot: example cyclic features (first period pair only, if exists)
        if X_cyc.shape[1] >= 2:
            plt.figure(figsize=(10, 3))
            plt.plot(df[cfg_data.time_col], X_cyc[:, 0], label="sin(P1)")
            plt.plot(df[cfg_data.time_col], X_cyc[:, 1], label="cos(P1)")
            plt.title("Stage_01: Example Cyclic Features (first period)")
            plt.xlabel("Time")
            plt.ylabel("value")
            plt.legend()
            plt.tight_layout()
            cyc_plot_path = os.path.join(step_dir, "cyclic_features_p1.png")
            plt.savefig(cyc_plot_path, dpi=150)
            plt.close()
        else:
            cyc_plot_path = None

        # Build a summary table for user
        summary = pd.DataFrame({
            "artifact": ["series_csv", "raw_plot", "scaled_vs_raw_plot", "cyclic_plot"],
            "path": [
                context["artifacts_stage_00"]["series_csv"],
                context["artifacts_stage_00"]["raw_plot"],
                scaled_plot_path,
                cyc_plot_path if cyc_plot_path else "N/A"
            ],
            "shape_info": [
                str(context["df"].shape),
                "png",
                "png",
                "png" if cyc_plot_path else "N/A"
            ],
        })
        # Display summary table Stage_00_01_Artifacts
        print("Stage_00_01_Artifacts:")
        print(df.head())

        # Update context
        context.update({
            "X": X_valid,
            "y_raw": y,
            "y_scl": y_scl_valid,
            "valid_rows_mask": valid_rows,
            "standardizer": {"mean": stdzr.mean_, "std": stdzr.std_},
            "minmax": ({"min": mm.min_, "max": mm.max_} if mm else None),
            "artifacts_stage_01": {
                "X_npy": os.path.join(step_dir, "X.npy"),
                "y_scl_npy": os.path.join(step_dir, "y_scl.npy"),
                "scaled_vs_raw_plot": scaled_plot_path,
                "cyclic_plot": cyc_plot_path,
                "meta": os.path.join(step_dir, "preprocess_meta.json"),
            }
        })
        return context

