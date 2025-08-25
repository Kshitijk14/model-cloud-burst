import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

from utils.helpers.pipeline import Step
from utils.helpers.general_utils import save_json


def make_windows(series: np.ndarray, lookback: int, horizon: int):
    X, Y = [], []
    T = len(series)
    for t in range(lookback, T - horizon + 1):
        X.append(series[t - lookback:t])
        Y.append(series[t:t + horizon])
    X = np.array(X).reshape(-1, lookback, 1)
    Y = np.array(Y).reshape(-1, horizon)
    return X, Y


class Stage04_PostProcessing(Step):
    name = "stage_04_postproc"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        step_dir = context["_current_step_dir"]
        os.makedirs(step_dir, exist_ok=True)

        run_cfg = self.cfg["run"]
        lookback = run_cfg.lookback
        horizon = run_cfg.horizon

        datasets_imf = {}
        datasets_fcr = {}

        # ---- IMF channels ----
        if "channels_imf" in context:
            imf_dir = os.path.join(step_dir, "imf")
            os.makedirs(imf_dir, exist_ok=True)

            for name, arr in context["channels_imf"].items():
                mean, std = arr.mean(), arr.std() + 1e-8
                arr_std = (arr - mean) / std

                X, Y = make_windows(arr_std, lookback, horizon)
                
                # compute split sizes
                n = len(X)
                cfg_data = context["cfg_data"]
                n_train = int(n * cfg_data.train_ratio)
                n_val   = int(n * cfg_data.val_ratio)
                n_test  = n - n_train - n_val

                datasets_imf[name] = {
                    "X_train": X[:n_train],
                    "Y_train": Y[:n_train],
                    "X_val":   X[n_train:n_train+n_val],
                    "Y_val":   Y[n_train:n_train+n_val],
                    "X_test":  X[n_train+n_val:],
                    "Y_test":  Y[n_train+n_val:],
                    "mean": mean,
                    "std": std
                }
                
                # save raw + stats
                np.save(os.path.join(imf_dir, f"X_{name}.npy"), X)
                np.save(os.path.join(imf_dir, f"Y_{name}.npy"), Y)
                
                save_json(os.path.join(imf_dir, f"stats_{name}.json"), {
                    "mean": float(mean),
                    "std": float(std),
                    "full_X": X.shape,
                    "full_Y": Y.shape,
                    "train_X": datasets_imf[name]["X_train"].shape,
                    "train_Y": datasets_imf[name]["Y_train"].shape,
                    "val_X": datasets_imf[name]["X_val"].shape,
                    "val_Y": datasets_imf[name]["Y_val"].shape,
                    "test_X": datasets_imf[name]["X_test"].shape,
                    "test_Y": datasets_imf[name]["Y_test"].shape,
                    "lookback": lookback,
                    "horizon": horizon
                })
                
                print(f"[Stage_04] IMF {name}: train{X[:n_train].shape}, "
                      f"val{X[n_train:n_train+n_val].shape}, "
                      f"test{X[n_train+n_val:].shape}")

                # sample plot
                plt.figure(figsize=(8, 3))
                plt.plot(arr[:lookback+horizon], label="raw")
                plt.axvline(lookback, color="red", linestyle="--", label="split")
                plt.legend()
                plt.title(f"Sample window for {name}")
                plt.tight_layout()
                plt.savefig(os.path.join(imf_dir, f"sample_{name}.png"))
                plt.close()

        # ---- FCR channels ----
        if "channels_fcr" in context:
            fcr_dir = os.path.join(step_dir, "fcr")
            os.makedirs(fcr_dir, exist_ok=True)

            for name, arr in context["channels_fcr"].items():
                mean, std = arr.mean(), arr.std() + 1e-8
                arr_std = (arr - mean) / std

                X, Y = make_windows(arr_std, lookback, horizon)
                
                # compute split sizes
                n = len(X)
                cfg_data = context["cfg_data"]
                n_train = int(n * cfg_data.train_ratio)
                n_val   = int(n * cfg_data.val_ratio)
                n_test  = n - n_train - n_val

                datasets_fcr[name] = {
                    "X_train": X[:n_train],
                    "Y_train": Y[:n_train],
                    "X_val":   X[n_train:n_train+n_val],
                    "Y_val":   Y[n_train:n_train+n_val],
                    "X_test":  X[n_train+n_val:],
                    "Y_test":  Y[n_train+n_val:],
                    "mean": mean,
                    "std": std
                }
                
                # save raw + stats
                np.save(os.path.join(fcr_dir, f"X_{name}.npy"), X)
                np.save(os.path.join(fcr_dir, f"Y_{name}.npy"), Y)
                
                save_json(os.path.join(fcr_dir, f"stats_{name}.json"), {
                    "mean": float(mean),
                    "std": float(std),
                    "full_X": X.shape,
                    "full_Y": Y.shape,
                    "train_X": datasets_fcr[name]["X_train"].shape,
                    "train_Y": datasets_fcr[name]["Y_train"].shape,
                    "val_X": datasets_fcr[name]["X_val"].shape,
                    "val_Y": datasets_fcr[name]["Y_val"].shape,
                    "test_X": datasets_fcr[name]["X_test"].shape,
                    "test_Y": datasets_fcr[name]["Y_test"].shape,
                    "lookback": lookback,
                    "horizon": horizon
                })

                print(f"[Stage_04] FCR {name}: train{X[:n_train].shape}, "
                      f"val{X[n_train:n_train+n_val].shape}, "
                      f"test{X[n_train+n_val:].shape}")

                # sample plot
                plt.figure(figsize=(8, 3))
                plt.plot(arr[:lookback+horizon], label="raw")
                plt.axvline(lookback, color="red", linestyle="--", label="split")
                plt.legend()
                plt.title(f"Sample window for {name}")
                plt.tight_layout()
                plt.savefig(os.path.join(fcr_dir, f"sample_{name}.png"))
                plt.close()

        # update context
        context.update({
            "datasets_imf": datasets_imf,
            "datasets_fcr": datasets_fcr,
            "artifacts_stage_04": {
                "imf_dir": os.path.join(step_dir, "imf"),
                "fcr_dir": os.path.join(step_dir, "fcr"),
            }
        })
        return context
