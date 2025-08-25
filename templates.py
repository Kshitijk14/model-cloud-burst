import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


project_name = "time_series_forecasting"

list_of_files = [
    "time_series_pipeline/__init__.py",
    "time_series_pipeline/stage_01_setup.py",
    "time_series_pipeline/stage_02_preprocessing.py",
    "time_series_pipeline/stage_03_signal_decomposition.py",
    "time_series_pipeline/stage_03a_imf_mode.py",
    "time_series_pipeline/stage_03b_fcr_mode.py",
    "time_series_pipeline/stage_04_postprocessing_per_channel.py",
    "time_series_pipeline/stage_05_model_forecasting.py",
    "time_series_pipeline/stage_06_ensemble_n_reconstruction.py",
    "time_series_pipeline/stage_07_evaluation.py",

    "utils/__init__.py",
    "utils/config.py",
    "utils/logger.py",
    "utils/helpers/__init__.py",
    
    "main.py",
    
    "params.yaml",
    "DVC.yaml",
    ".env.local",
]


for filepath in list_of_files:
    filepath = Path(filepath) #to solve the windows path issue
    filedir, filename = os.path.split(filepath) # to handle the project_name folder


    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists")