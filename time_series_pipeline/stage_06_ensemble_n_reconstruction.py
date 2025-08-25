import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

from utils.helpers.pipeline import Step


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
            imf_forecasts = []
            for name, preds_by_model in preds_imf.items():
                stats = datasets_imf[name]
                mean, std = stats["mean"], stats["std"]
                
                raw = preds_by_model[model_type]["pred"]["test"]
                pred = raw * std + mean
                
                imf_forecasts.append(pred)
            recon_imf = np.sum(imf_forecasts, axis=0)

            # ---- FCR reconstruction ----
            fcr_forecasts = []
            for name, preds_by_model in preds_fcr.items():
                stats = datasets_fcr[name]
                mean, std = stats["mean"], stats["std"]
                
                raw = preds_by_model[model_type]["pred"]["test"]
                pred = raw * std + mean
                
                fcr_forecasts.append(pred)
            recon_fcr = np.sum(fcr_forecasts, axis=0)

            # ---- Meta ensemble (average of IMF + FCR) ----
            recon_meta = 0.5 * (recon_imf + recon_fcr)

            # save reconstructions
            np.save(os.path.join(step_dir, f"recon_{model_type}_imf.npy"), recon_imf)
            np.save(os.path.join(step_dir, f"recon_{model_type}_fcr.npy"), recon_fcr)
            np.save(os.path.join(step_dir, f"recon_{model_type}_meta.npy"), recon_meta)

            recons[model_type] = {
                "imf": recon_imf,
                "fcr": recon_fcr,
                "meta": recon_meta,
            }

        # ---- Plots: LSTM vs GRU for IMF, FCR, Meta ----
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        axes[0].plot(recons["LSTM"]["imf"].ravel(), label="LSTM IMF", color="blue")
        axes[0].plot(recons["GRU"]["imf"].ravel(), label="GRU IMF", color="cyan")
        axes[0].legend(); axes[0].set_title("IMF Reconstruction (LSTM vs GRU)")

        axes[1].plot(recons["LSTM"]["fcr"].ravel(), label="LSTM FCR", color="green")
        axes[1].plot(recons["GRU"]["fcr"].ravel(), label="GRU FCR", color="lime")
        axes[1].legend(); axes[1].set_title("FCR Reconstruction (LSTM vs GRU)")

        axes[2].plot(recons["LSTM"]["meta"].ravel(), label="LSTM Meta", color="red")
        axes[2].plot(recons["GRU"]["meta"].ravel(), label="GRU Meta", color="orange")
        axes[2].legend(); axes[2].set_title("Meta Ensemble (LSTM vs GRU)")

        plt.tight_layout()
        plt.savefig(os.path.join(step_dir, "reconstruction_comparison.png"))
        plt.close()

        print(f"[Stage_06] Reconstructions saved under {step_dir}")

        # update context
        context["reconstructions"] = recons
        context["artifacts_stage_06"] = {"recon_dir": step_dir}
        return context
