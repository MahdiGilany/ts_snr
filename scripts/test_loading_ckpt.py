import sys
import os
from glob import glob
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.datasets import AirPassengersDataset

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

from src.data.registry.data_registry import create_dataset
train_ts, val_ts, scaler = create_dataset('air_passengers') # TimeSeries class object


from src.modeling.models.deep_time import DeepTIMeModel
from darts.models.forecasting.nbeats import NBEATSModel

import wandb

# exp_name = f"deeptime_airpassenger"
# dir = f"logs/experiments/runs/{exp_name}"
# work_dir = os.path.join(os.getcwd(), dir)
# work_dir = glob(os.path.join(work_dir, "**/darts_logs"), recursive=True)[1]

exp_name = f"nbeats_airpassenger"
dir = f"logs/experiments/runs/{exp_name}"
work_dir = os.path.join(os.getcwd(), dir)
work_dir = glob(os.path.join(work_dir, "**/darts_logs"), recursive=True)[0]

model = NBEATSModel(
    input_chunk_length=24,
    output_chunk_length=12,
    pl_trainer_kwargs={"num_sanity_val_steps": 0},
    model_name="nbeats",
    force_reset=True,
    ).load_from_checkpoint(model_name="nbeats", work_dir=work_dir, best=True)
print(model.load_ckpt_path)
# model.load_weights_from_checkpoint
pred = model.predict(series=train_ts, n=36)
print(pred)