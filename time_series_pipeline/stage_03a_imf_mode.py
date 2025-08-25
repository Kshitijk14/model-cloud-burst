import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

from utils.helpers.pipeline import Step
from utils.helpers.general_utils import save_json


class Stage03a_IMF_Mode(Step):
    name = "stage_03a_imf_mode"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        step_dir = context["_current_step_dir"]
        os.makedirs(step_dir, exist_ok=True)

        imfs: np.ndarray = context["imfs"]  # shape (n, T)
        residual: np.ndarray = context["residual"]

        n, T = imfs.shape
        print(f"[Stage_03a] Received {n} IMFs of length {T}")
        print(f"[Stage_03a] Residual length: {residual.shape}")

        # Build channels dict
        channels = {f"c{i+1}": imfs[i] for i in range(n)}
        channels["residual"] = residual

        # Save each channel
        for name, arr in channels.items():
            np.save(os.path.join(step_dir, f"{name}.npy"), arr)

        # Save metadata
        meta = {"n_imfs": n, "length": T, "channels": list(channels.keys())}
        save_json(os.path.join(step_dir, "channels_meta.json"), meta)

        # Plot grid of IMFs + residual
        fig, axes = plt.subplots(n + 2, 1, figsize=(10, 2*(n+2)), sharex=True)
        axes[0].plot(np.sum(imfs, axis=0) + residual, color="black", label="Reconstructed signal")
        axes[0].legend(loc="upper right")

        for i in range(n):
            axes[i+1].plot(imfs[i], label=f"IMF {i+1}")
            axes[i+1].legend(loc="upper right")

        axes[-1].plot(residual, color="red", label="Residual")
        axes[-1].legend(loc="upper right")

        plt.tight_layout()
        plot_path = os.path.join(step_dir, "imf_channels.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"[Stage_03a] Saved IMF channels + residual under {step_dir}")

        # Update context
        context.update({
            "channels_imf": channels,
            "artifacts_stage_03a": {
                "channels_meta": os.path.join(step_dir, "channels_meta.json"),
                "plot": plot_path,
            }
        })
        return context
