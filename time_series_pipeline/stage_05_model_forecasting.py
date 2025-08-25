import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, LSTM, GRU, Dense
from tensorflow.keras.optimizers import Adam

from utils.helpers.pipeline import Step


def build_model(model_type: str, lookback: int, horizon: int, hidden_dim: int = 64, layers: int = 1, lr: float = 1e-3):
    model = Sequential()
    model.add(InputLayer(shape=(lookback, 1)))
    
    for i in range(layers):
        return_seq = (i < layers - 1)
        if model_type == "LSTM":
            model.add(LSTM(hidden_dim, return_sequences=return_seq))
        elif model_type == "GRU":
            model.add(GRU(hidden_dim, return_sequences=return_seq))
    
    model.add(Dense(horizon))
    model.compile(optimizer=Adam(lr), loss="mse", metrics=["mae"])
    return model


class Stage05_ModelForecasting(Step):
    name = "stage_05_forecasting"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        step_dir = context["_current_step_dir"]
        os.makedirs(step_dir, exist_ok=True)

        run_cfg = self.cfg["run"]
        train_cfg = self.cfg.get("train", {
            "hidden_dim": 64,
            "layers": 1,
            "epochs": 10,
            "batch_size": 32,
            "lr": 1e-3,
        })

        lookback = run_cfg.lookback
        horizon = run_cfg.horizon

        preds_imf = {}
        preds_fcr = {}

        def train_channel(channel_name, data, out_dir):
            X_train, Y_train = data["X_train"], data["Y_train"]
            X_val, Y_val = data["X_val"], data["Y_val"]
            X_test, Y_test = data["X_test"], data["Y_test"]

            results = {}
            for model_type in ["LSTM", "GRU"]:
                print(f"[Stage_05] Training {model_type} on channel: {channel_name}")

                model = build_model(model_type, lookback, horizon,
                                    hidden_dim=train_cfg["hidden_dim"],
                                    layers=train_cfg["layers"],
                                    lr=train_cfg["lr"])

                hist = model.fit(
                    X_train, Y_train,
                    validation_data=(X_val, Y_val),
                    epochs=train_cfg["epochs"],
                    batch_size=train_cfg["batch_size"],
                    verbose=1
                )

                # save weights
                model_path = os.path.join(out_dir, f"{channel_name}_{model_type}.weights.h5")
                model.save_weights(model_path)

                # training curve
                plt.figure()
                plt.plot(hist.history["loss"], label="train")
                plt.plot(hist.history["val_loss"], label="val")
                plt.title(f"{channel_name} {model_type} Loss")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"{channel_name}_{model_type}_loss.png"))
                plt.close()

                # predictions for all splits
                preds = {
                    "train": model.predict(X_train),
                    "val":   model.predict(X_val),
                    "test":  model.predict(X_test)
                }

                # save predictions
                for split, arr in preds.items():
                    np.save(os.path.join(out_dir, f"{channel_name}_{model_type}_pred_{split}.npy"), arr)

                results[model_type] = {
                    "weights": model_path,
                    "pred": preds,
                    "history": hist.history
                }

            return results

        # ---- IMF channels ----
        if "datasets_imf" in context:
            imf_dir = os.path.join(step_dir, "imf")
            os.makedirs(imf_dir, exist_ok=True)
            for name, data in context["datasets_imf"].items():
                preds_imf[name] = train_channel(name, data, imf_dir)

        # ---- FCR channels ----
        if "datasets_fcr" in context:
            fcr_dir = os.path.join(step_dir, "fcr")
            os.makedirs(fcr_dir, exist_ok=True)
            for name, data in context["datasets_fcr"].items():
                preds_fcr[name] = train_channel(name, data, fcr_dir)

        # update context
        context.update({
            "preds_imf": preds_imf,
            "preds_fcr": preds_fcr,
            "artifacts_stage_05": step_dir
        })
        return context