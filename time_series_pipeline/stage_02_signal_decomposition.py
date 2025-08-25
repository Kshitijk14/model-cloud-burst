import os
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, Any
from PyEMD import EEMD

from utils.helpers.pipeline import Step
from utils.helpers.general_utils import save_json


class Stage02_EEMD(Step):
    name = "stage_02_signal_decomposition"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        step_dir = context["_current_step_dir"]
        os.makedirs(step_dir, exist_ok=True)

        # --- get config params ---
        eemd_cfg = self.cfg.get("eemd", {})
        noise_strength = eemd_cfg.get("noise_strength", 0.2)
        trials = eemd_cfg.get("trials", 50)
        max_imfs = eemd_cfg.get("max_imfs", None)

        # --- get input series ---
        # Use standardized/scaled signal from Stage_01
        y = context["y_scl"]
        print(f"[Stage_02] Input series shape: {y.shape}")

        # --- run EEMD ---
        eemd = EEMD(trials=trials, noise_strength=noise_strength)
        if max_imfs is None:
            imfs = eemd.eemd(y)
        else:
            imfs = eemd.eemd(y, max_imf=max_imfs)  # correct arg name is max_imf
        residual = y - imfs.sum(axis=0)

        print(f"[Stage_02] IMFs shape: {imfs.shape}")
        print(f"[Stage_02] Residual shape: {residual.shape}")

        # --- energy ratio diagnostics ---
        energies = [np.sum(imf**2) for imf in imfs]
        res_energy = np.sum(residual**2)
        total_energy = sum(energies) + res_energy
        ratios = [float(e / total_energy) for e in energies] + [float(res_energy / total_energy)]
        labels = [f"IMF{i+1}" for i in range(imfs.shape[0])] + ["Residual"]

        # save diagnostics
        diag = {"labels": labels, "ratios": ratios}
        save_json(os.path.join(step_dir, "energy_ratios.json"), diag)
        np.savetxt(os.path.join(step_dir, "energy_ratios.csv"),
                   np.vstack([labels, ratios]).T, fmt="%s", delimiter=",")

        # print top-3 contributors
        top3 = sorted(zip(labels, ratios), key=lambda x: -x[1])[:3]
        print("[Stage_02] Top energy contributors:", top3)

        # plot energy distribution
        plt.figure(figsize=(8, 4))
        plt.bar(labels, ratios)
        plt.ylabel("Energy ratio")
        plt.title("Stage_02: Energy contribution of IMFs + Residual")
        plt.xticks(rotation=45)
        plt.tight_layout()
        energy_plot_path = os.path.join(step_dir, "energy_ratios.png")
        plt.savefig(energy_plot_path, dpi=150)
        plt.close()
        
        # --- save intermediates ---
        np.save(os.path.join(step_dir, "imfs.npy"), imfs)
        np.save(os.path.join(step_dir, "residual.npy"), residual)
        save_json(os.path.join(step_dir, "eemd_meta.json"), {
            "n_imfs": imfs.shape[0],
            "length": imfs.shape[1],
            "noise_strength": noise_strength,
            "trials": trials,
        })

        # --- plot decomposition ---
        fig, axes = plt.subplots(imfs.shape[0] + 2, 1,
                                 figsize=(10, 2*(imfs.shape[0]+2)),
                                 sharex=True)
        axes[0].plot(y, label="Original", color="black")
        axes[0].legend(loc="upper right")

        for i, imf in enumerate(imfs):
            axes[i+1].plot(imf, label=f"IMF {i+1}")
            axes[i+1].legend(loc="upper right")

        axes[-1].plot(residual, label="Residual", color="red")
        axes[-1].legend(loc="upper right")

        plt.tight_layout()
        plot_path = os.path.join(step_dir, "eemd_decomposition.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"[Stage_02] Saved IMFs/residual under {step_dir}")

        # --- update context ---
        context.update({
            "imfs": imfs,
            "residual": residual,
            "energy_ratios": dict(zip(labels, ratios)),
            "artifacts_stage_02": {
                "imfs_npy": os.path.join(step_dir, "imfs.npy"),
                "residual_npy": os.path.join(step_dir, "residual.npy"),
                "plot": plot_path,
                "energy_plot": energy_plot_path,
                "energy_json": os.path.join(step_dir, "energy_ratios.json"),
                "energy_csv": os.path.join(step_dir, "energy_ratios.csv"),
                "meta": os.path.join(step_dir, "eemd_meta.json"),
            }
        })
        return context