import os

from utils.helpers.common import DataConfig, PreprocessConfig, RunConfig
from utils.helpers.pipeline import Pipeline

from time_series_pipeline.stage_00_setup import Stage00_Setup
from time_series_pipeline.stage_01_preprocessing import Stage01_Preprocess
from time_series_pipeline.stage_02_signal_decomposition import Stage02_EEMD
from time_series_pipeline.stage_03a_imf_mode import Stage03a_IMF_Mode
from time_series_pipeline.stage_03b_fcr_mode import Stage03b_FCR_Mode
from time_series_pipeline.stage_04_postprocessing_per_channel import Stage04_PostProcessing
from time_series_pipeline.stage_05_model_forecasting import Stage05_ModelForecasting
from time_series_pipeline.stage_06_ensemble_n_reconstruction import Stage06_EnsembleReconstruction


DATA_CFG = DataConfig(
    data_source=None,   # set to your CSV later; expects ["ds","y"]
    time_col="ds",
    target_col="y",
    freq="D",
    length=1000,
    train_ratio=0.7,
    val_ratio=0.15,
)

PREP_CFG = PreprocessConfig(
    periods=(7, 30, 365),  # tune to your domain/freq
    max_lag=7,
    use_minmax=False,
)

EEMD_CFG = {
    "noise_strength": 0.2,   # std of added white noise
    "trials": 50,            # number of ensemble runs
    "max_imfs": None         # None = auto; or int for limit
}

RUN_CFG = RunConfig(
    lookback=96,
    horizon=24,
    artifacts_root="artifacts/ts_eemd_pipeline",
    run_name="demo_run",
)

TRAIN_CFG = {
    "hidden_dim": 64,
    "layers": 1,
    "epochs": 10,
    "batch_size": 32,
    "lr": 1e-3,
}


# ---------------- PIPELINE ----------------
run_dir = os.path.join(RUN_CFG.artifacts_root, RUN_CFG.run_name)

steps = [
    Stage00_Setup(cfg={"data": DATA_CFG, "run": RUN_CFG}),
    Stage01_Preprocess(cfg={"prep": PREP_CFG, "run": RUN_CFG}),
    Stage02_EEMD(cfg={"eemd": EEMD_CFG, "run": RUN_CFG}),
    Stage03a_IMF_Mode(cfg={"run": RUN_CFG}),
    Stage03b_FCR_Mode(cfg={"run": RUN_CFG}),
    Stage04_PostProcessing(cfg={"run": RUN_CFG}),
    Stage05_ModelForecasting(cfg={"run": RUN_CFG, "train": TRAIN_CFG}),
    Stage06_EnsembleReconstruction(cfg={"run": RUN_CFG}),
]

pipe = Pipeline(steps=steps, run_dir=run_dir)
ctx = pipe.run()

print("\n=== SHAPE SUMMARY ===")
print("X:", ctx["X"].shape, "| y_scl:", ctx["y_scl"].shape)
print("Standardizer:", ctx["standardizer"])
print("MinMax:", ctx["minmax"])


print("\nIMF mode channels:", list(ctx["channels_imf"].keys()))
print("\nFCR mode channels:", list(ctx["channels_fcr"].keys()))


print("\nStage_04 IMF Datasets:")
for name, ds in ctx["datasets_imf"].items():
    total_X = (len(ds['X_train']) + len(ds['X_val']) + len(ds['X_test']))
    total_Y = (len(ds['Y_train']) + len(ds['Y_val']) + len(ds['Y_test']))
    print(f" - {name}: "
          f"train {ds['X_train'].shape}/{ds['Y_train'].shape}, "
          f"val {ds['X_val'].shape}/{ds['Y_val'].shape}, "
          f"test {ds['X_test'].shape}/{ds['Y_test'].shape} "
          f"| total_X={total_X}, total_Y={total_Y}")

print("\nStage_04 FCR Datasets:")
for name, ds in ctx["datasets_fcr"].items():
    total_X = (len(ds['X_train']) + len(ds['X_val']) + len(ds['X_test']))
    total_Y = (len(ds['Y_train']) + len(ds['Y_val']) + len(ds['Y_test']))
    print(f" - {name}: "
          f"train {ds['X_train'].shape}/{ds['Y_train'].shape}, "
          f"val {ds['X_val'].shape}/{ds['Y_val'].shape}, "
          f"test {ds['X_test'].shape}/{ds['Y_test'].shape} "
          f"| total_X={total_X}, total_Y={total_Y}")


print("\nStage_05 IMF Predictions:")
for name, preds in ctx["preds_imf"].items():
    for model_type in preds.keys():
        for split, arr in preds[model_type]["pred"].items():
            print(f" - {name} [{model_type}] {split}: {arr.shape}")

print("\nStage_05 FCR Predictions:")
for name, preds in ctx["preds_fcr"].items():
    for model_type in preds.keys():
        for split, arr in preds[model_type]["pred"].items():
            print(f" - {name} [{model_type}] {split}: {arr.shape}")


print("\nStage_06 Reconstructions:")
for model_type, recon in ctx["reconstructions"].items():
    for kind, arr in recon.items():
        print(f" - {model_type} {kind}: {arr.shape}")


print("\nArtifacts saved under:", run_dir)
for k in [f"artifacts_stage_0{i}" for i in range(1, 6)]:
    if k in ctx:
        print(f" - {k}: {ctx[k]}")