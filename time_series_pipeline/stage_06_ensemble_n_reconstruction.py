import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

from utils.helpers.pipeline import Step
from utils.helpers.reconst import get_indices_for_split


class Stage06_EnsembleReconstruction(Step):
    name = "stage_06_ensemble_reconstruction"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        step_dir = context["_current_step_dir"]
        os.makedirs(step_dir, exist_ok=True)

        preds_imf = context["preds_imf"]
        preds_fcr = context["preds_fcr"]
        datasets_imf = context["datasets_imf"]
        datasets_fcr = context["datasets_fcr"]
        
        df = context["df"]
        time_col = context["cfg_data"].time_col
        splits = context["splits"]

        recons = {"LSTM": {}, "GRU": {}}

        for model_type in ["LSTM", "GRU"]:
            # ---- IMF reconstruction ----
            imf_forecasts = {"train": [], "val": [], "test": []}
            for name, preds_by_model in preds_imf.items():
                stats = datasets_imf[name]
                mean, std = stats["mean"], stats["std"]

                for split in ["train", "val", "test"]:
                    raw = preds_by_model[model_type]["pred"][split]
                    pred = raw * std + mean
                    imf_forecasts[split].append(pred)

            recon_imf = {split: np.sum(arrs, axis=0) for split, arrs in imf_forecasts.items()}

            # ---- FCR reconstruction ----
            fcr_forecasts = {"train": [], "val": [], "test": []}
            for name, preds_by_model in preds_fcr.items():
                stats = datasets_fcr[name]
                mean, std = stats["mean"], stats["std"]

                for split in ["train", "val", "test"]:
                    raw = preds_by_model[model_type]["pred"][split]
                    pred = raw * std + mean
                    fcr_forecasts[split].append(pred)

            recon_fcr = {split: np.sum(arrs, axis=0) for split, arrs in fcr_forecasts.items()}

            # ---- Meta ensemble (average of IMF + FCR) ----
            recon_meta = {split: 0.5 * (recon_imf[split] + recon_fcr[split]) for split in ["train", "val", "test"]}

            # save reconstructions
            for split in ["train", "val", "test"]:
                np.save(os.path.join(step_dir, f"recon_{model_type}_imf_{split}.npy"), recon_imf[split])
                np.save(os.path.join(step_dir, f"recon_{model_type}_fcr_{split}.npy"), recon_fcr[split])
                np.save(os.path.join(step_dir, f"recon_{model_type}_meta_{split}.npy"), recon_meta[split])

            recons[model_type] = {
                "imf": recon_imf,
                "fcr": recon_fcr,
                "meta": recon_meta,
            }

        # ---- Helper for plotting actual vs predicted ----
        def plot_vs_actual(kind: str, split: str, zoom: bool = False):
            # actual_idx = splits[split]
            # idx = np.array(actual_idx)[-len(recons["LSTM"][kind][split]):]
            # dates = df[time_col].iloc[idx].values
            # y_true = df[context["cfg_data"].target_col].iloc[idx].values
            
            pred_lstm = recons["LSTM"][kind][split][:, -1]
            pred_gru  = recons["GRU"][kind][split][:, -1]
            
            actual_idx = get_indices_for_split(splits, split, length=len(pred_lstm))
            dates = df[time_col].iloc[actual_idx].values
            y_true = df[context["cfg_data"].target_col].iloc[actual_idx].values

            plt.figure(figsize=(12, 4))
            plt.plot(dates, y_true, label="Actual", color="black")
            # plt.plot(dates, recons["LSTM"][kind][split].ravel(), label=f"LSTM {kind}", color="blue")
            # plt.plot(dates, recons["GRU"][kind][split].ravel(), label=f"GRU {kind}", color="orange")
            plt.plot(dates, pred_lstm, label=f"LSTM {kind}", color="blue")
            plt.plot(dates, pred_gru, label=f"GRU {kind}", color="orange")

            plt.legend()
            title = f"{kind.upper()} {split.title()} (Actual vs Predicted)"
            if zoom:
                title += " [Zoomed]"
                if len(dates) > 200:
                    dates = dates[-200:]
                    y_true = y_true[-200:]
                    lstm_pred = recons["LSTM"][kind][split].ravel()[-200:]
                    gru_pred = recons["GRU"][kind][split].ravel()[-200:]

                    plt.cla()
                    plt.figure(figsize=(12, 4))
                    plt.plot(dates, y_true, label="Actual", color="black")
                    plt.plot(dates, lstm_pred, label=f"LSTM {kind}", color="blue")
                    plt.plot(dates, gru_pred, label=f"GRU {kind}", color="orange")
                    plt.legend()

            plt.title(title)
            plt.tight_layout()
            fname = f"{kind}_{split}{'_zoom' if zoom else ''}.png"
            plt.savefig(os.path.join(step_dir, fname))
            plt.close()
        
        # ---- Generate plots for each kind & split ----
        for kind in ["imf", "fcr", "meta"]:
            for split in ["val", "test"]:  # focus on val & test
                plot_vs_actual(kind, split, zoom=False)
                plot_vs_actual(kind, split, zoom=True)
        
        
        # ---- Plots: LSTM vs GRU for IMF, FCR, Meta ----
        for split_to_plot in ["val", "test"]:
            # actual_idx = splits[split_to_plot]
            # dates = df[context["cfg_data"].time_col].iloc[actual_idx].values
            
            actual_idx = get_indices_for_split(splits, split_to_plot, length=len(recons["LSTM"]["imf"][split_to_plot]))
            dates = df[context["cfg_data"].time_col].iloc[actual_idx].values
            
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

            # IMF
            axes[0].plot(dates, recons["LSTM"]["imf"][split_to_plot][:, -1], label="LSTM IMF", color="blue")
            axes[0].plot(dates, recons["GRU"]["imf"][split_to_plot][:, -1], label="GRU IMF", color="cyan")
            axes[0].legend()
            axes[0].set_title("IMF Reconstruction (LSTM vs GRU)")

            # FCR
            axes[1].plot(dates, recons["LSTM"]["fcr"][split_to_plot][:, -1], label="LSTM FCR", color="green")
            axes[1].plot(dates, recons["GRU"]["fcr"][split_to_plot][:, -1], label="GRU FCR", color="lime")
            axes[1].legend()
            axes[1].set_title("FCR Reconstruction (LSTM vs GRU)")

            # Meta
            axes[2].plot(dates, recons["LSTM"]["meta"][split_to_plot][:, -1], label="LSTM Meta", color="red")
            axes[2].plot(dates, recons["GRU"]["meta"][split_to_plot][:, -1], label="GRU Meta", color="orange")
            axes[2].legend()
            axes[2].set_title("Meta Ensemble (LSTM vs GRU)")

            plt.xlabel("Date-Time")
            plt.tight_layout()
            fname = os.path.join(step_dir, f"reconstruction_comparison_{split_to_plot}.png")
            plt.savefig(fname)
            plt.close()

        print(f"[Stage_06] Reconstructions saved under {step_dir}")

        # update context
        context["reconstructions"] = recons
        context["artifacts_stage_06"] = {"recon_dir": step_dir}
        return context
