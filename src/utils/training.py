import sys
import os

# sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional, Union, Dict, Literal, Tuple
import plotly.express as px
from einops import repeat


import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback

from src.utils import driver as utils
from src.utils.metrics import calculate_metrics_darts, calculate_metrics
from src.data.registry.data_registry import DataSeries

from darts.timeseries import TimeSeries, concatenate
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel, DEFAULT_DARTS_FOLDER
from darts.models.forecasting.forecasting_model import ForecastingModel, LocalForecastingModel
from darts.utils.data.sequential_dataset import PastCovariatesSequentialDataset

from tqdm import tqdm



log = utils.get_logger(__name__)


class SequeceDataset(Dataset):
    def __init__(
        self, 
        seq_data: torch.tensor, 
        seq_label: Optional[torch.tensor] = None,
        seq_len: int = 24,
        ) -> None:
        super().__init__()
        self.seq_data = seq_data
        self.seq_label = seq_label if seq_label is not None else seq_data
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.seq_data) - self.seq_len + 1
    
    def __getitem__(self, index) :
        return self.seq_data[index:index+self.seq_len], self.seq_label[index:index+self.seq_len]
        

def get_lookback_horizon_codes(
   deeptime_model: TorchForecastingModel, 
   time_series: TimeSeries,
   input_chunk_length: int,
   output_chunk_length: int,
   batch_size: int,
): 
    
    series_ds = PastCovariatesSequentialDataset(
        time_series,
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        covariates=None,
        use_static_covariates=False
        )
    series_dl = DataLoader(
        series_ds,
        batch_size=deeptime_model.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=deeptime_model._batch_collate_fn,
        )
    
    
    assert deeptime_model.model is not None, "model.model not found, please load the model using model.load_from_checkpoint()\
        or model.load_weights_from_checkpoint() which initializes model.model"
    
    
    pl_deeptime_model = deeptime_model.model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pl_deeptime_model = pl_deeptime_model.to(device) 
    pl_deeptime_model.eval()
    
    # time representation is took out of for loop for faster computation (time reprs as memory block)
    coords = pl_deeptime_model.get_coords(input_chunk_length, output_chunk_length).to(device)
    single_time_reprs = pl_deeptime_model.inr(coords)
    
    # list of signal and label codes
    W_Ls = []
    W_Hs = []
    for batch in tqdm(series_dl, desc="Acquiring signal and label codes (W_L and W_H)"):
        input_series, _, _, target_series = batch
        input_series = input_series.to(device=pl_deeptime_model.device, dtype=pl_deeptime_model.dtype)
        target_series = target_series.to(device=pl_deeptime_model.device, dtype=pl_deeptime_model.dtype)
        
        # # target_series = target_series.to(device=pl_model.device, dtype=pl_model.dtype)
        # pl_deeptime_model.y = target_series.to(device=pl_deeptime_model.device, dtype=pl_deeptime_model.dtype) # a hack for setting target for twoomp model
        
        # repeating time reprs for batch size
        batch_size = input_series.shape[0]
        time_reprs = repeat(single_time_reprs, '1 t d -> b t d', b=batch_size)
        
        # look back and horizon dictionaries        
        lookback_reprs = time_reprs[:, :-output_chunk_length] # shape = (batch_size, forecast_horizon_length, layer_size)
        horizon_reprs = time_reprs[:, -output_chunk_length:]
        
        w, b = pl_deeptime_model.adaptive_weights(lookback_reprs, input_series) # [bz, 256, 1], [bz, 1, 1]
        W_L = torch.cat([w, b], dim=1) # shape = (batch_size, layer_size + 1, 1)
        W_Ls.append(W_L.detach().cpu().numpy())
        
        # hack used for importance weights visualization (only first dim)
        pl_deeptime_model.learned_w = torch.cat([w, b], dim=1)[..., 0] # shape = (batch_size, layer_size + 1)
        
        w, b = pl_deeptime_model.adaptive_weights(horizon_reprs, target_series) # [bz, 256, 1], [bz, 1, 1]
        W_H = torch.cat([w, b], dim=1) # shape = (batch_size, layer_size + 1, 1)
        W_Hs.append(W_H.detach().cpu().numpy())

    # concatenate all batches
    W_Ls = np.concatenate(W_Ls, axis=0) # shape = (num_samples, layer_size + 1, 1)
    W_Hs = np.concatenate(W_Hs, axis=0)
    
    # flip back since dataset get item is designed in reverse order
    W_Ls = W_Ls[::-1, ...]
    W_Hs = W_Hs[::-1, ...]
    
    return W_Ls.copy(), W_Hs.copy()
  
def manual_train_seq_model(
    seq_config: DictConfig,
    seq_model: nn.Module,
    train_data: Tuple[np.array, np.array],
    val_data: Tuple[np.array, np.array],
    wandb_log: bool = False,
):
    # create a dataset and data loader
    train_x = torch.tensor(train_data[0])
    train_y = torch.tensor(train_data[1])
    val_x = torch.tensor(val_data[0])
    val_y = torch.tensor(val_data[1])
    
    train_ds = SequeceDataset(train_x, train_y, seq_len=seq_config.model.seq_len)
    val_ds = SequeceDataset(val_x, val_y, seq_len=seq_config.model.seq_len)
    
    train_dl = DataLoader(
            train_ds,
            batch_size=seq_config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            )
    val_dl = DataLoader(
            val_ds,
            batch_size=seq_config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            )
    
    # train the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seq_model.to(device=device, dtype=train_x.dtype)
    seq_model.train()
    
    optimizer = torch.optim.AdamW(seq_model.parameters(), lr=seq_config.lr)
    criterion = nn.MSELoss()
    
    best_val_loss = np.inf
    early_stop_counter = 0
    for epoch in tqdm(range(seq_config.epochs), desc="Training sequence model"):
        # training
        losses = []
        for batch in train_dl:
            seq_model.zero_grad()
            seq_data, seq_label = batch
            seq_data = seq_data.to(device)
            seq_label = seq_label.to(device)
            seq_pred = seq_model(seq_data)
            loss = criterion(seq_pred[:,-1,...], seq_label[:,-1,...])
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        if wandb_log:
            wandb.log({"Seq/train_loss": np.mean(losses), "Seq/epoch": epoch})
        
        # validation
        seq_model.eval()
        losses = []
        early_stop_counter += 1
        with torch.no_grad():
            for batch in val_dl:
                seq_data, seq_label = batch
                seq_data = seq_data.to(device)
                seq_label = seq_label.to(device)
                seq_pred = seq_model(seq_data)
                loss = criterion(seq_pred[:,-1,...], seq_label[:,-1,...])
                losses.append(loss.item())
            avg_loss = np.mean(losses)
            if wandb_log:
                wandb.log({"Seq/val_loss": avg_loss})
            if avg_loss < best_val_loss:
                early_stop_counter = 0
                best_val_loss = avg_loss
                torch.save(seq_model, f"./{seq_model.model_name}.pt")
            if early_stop_counter > seq_config.patience:
                log.info("Early stopping")
                return torch.load(f"./{seq_model.model_name}.pt")
        seq_model.train()
        
    return torch.load(f"./{seq_model.model_name}.pt")
        
  