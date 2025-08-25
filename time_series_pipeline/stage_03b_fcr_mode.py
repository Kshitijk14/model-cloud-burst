import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
from scipy.stats import ttest_1samp

from utils.helpers.pipeline import Step
from utils.helpers.general_utils import save_json


class Stage03b_FCR_Mode(Step):
    name = "stage_03b_fcr_mode"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        step_dir = context["_current_step_dir"]
        os.makedirs(step_dir, exist_ok=True)

        imfs: np.ndarray = context["imfs"]  # shape (n, T)
        residual: np.ndarray = context["residual"]
        n, T = imfs.shape
        print(f"[Stage_03b] Received {n} IMFs of length {T}")

        # --- Step 1: t-tests on IMF means ---
        pvals = []
        for i in range(n):
            _, p = ttest_1samp(imfs[i], 0.0)
            pvals.append(float(p))

        # Find cutoff IMF index
        cutoff_idx = None
        for i, p in enumerate(pvals):
            if p < 0.05:  # reject null -> mean significantly ≠ 0
                cutoff_idx = i
                break
        if cutoff_idx is None:
            cutoff_idx = n  # fallback: all IMFs considered HF

        print(f"[Stage_03b] Cutoff index i* = {cutoff_idx+1} (1-based)")
        print(f"[Stage_03b] p-values:", pvals)

        # --- Step 2: group into HF, LF, Trend ---
        HF = imfs[:cutoff_idx].sum(axis=0) if cutoff_idx > 0 else np.zeros(T)
        LF = imfs[cutoff_idx:].sum(axis=0) if cutoff_idx < n else np.zeros(T)
        Trend = residual

        channels = {"HF": HF, "LF": LF, "Trend": Trend}

        # Save each channel
        for name, arr in channels.items():
            np.save(os.path.join(step_dir, f"{name}.npy"), arr)

        # Save metadata
        meta = {"cutoff_index": int(cutoff_idx),
                "pvals": pvals,
                "channels": list(channels.keys())}
        save_json(os.path.join(step_dir, "fcr_meta.json"), meta)

        # --- Plot grouped channels ---
        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(np.sum(imfs, axis=0) + residual,
                     color="black", label="Reconstructed signal")
        axes[0].legend(loc="upper right")

        axes[1].plot(HF, label="HF")
        axes[1].legend(loc="upper right")

        axes[2].plot(LF, label="LF")
        axes[2].legend(loc="upper right")

        axes[3].plot(Trend, color="red", label="Trend")
        axes[3].legend(loc="upper right")

        plt.tight_layout()
        plot_path = os.path.join(step_dir, "fcr_channels.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"[Stage_03b] Saved FCR channels under {step_dir}")

        # --- Update context ---
        context.update({
            "channels_fcr": channels,
            "artifacts_stage_03b": {
                "HF_npy": os.path.join(step_dir, "HF.npy"),
                "LF_npy": os.path.join(step_dir, "LF.npy"),
                "Trend_npy": os.path.join(step_dir, "Trend.npy"),
                "meta": os.path.join(step_dir, "fcr_meta.json"),
                "plot": plot_path,
            }
        })
        return context
