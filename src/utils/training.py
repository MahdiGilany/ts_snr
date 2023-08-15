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


def sliding_window(
    series: TimeSeries,
    window_size: int,
):
    series_values = series.values()
    series_length = len(series)
    return np.array([
        series_values[i:i+window_size]
        for i in range(series_length-window_size+1)
        ])


def historical_forecasts_manual(
    model: TorchForecastingModel,
    test_series: TimeSeries,
    input_chunk_length: int,
    output_chunk_length: int,
    plot_weights: bool = False,
    ):
    # manually build test dataset and dataloader
    # test_ds = model._build_train_dataset(test_series, past_covariates=None, future_covariates=None, max_samples_per_ts=None)
    test_ds = PastCovariatesSequentialDataset(
        test_series,
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        covariates=None,
        use_static_covariates=False
        )
    test_dl = DataLoader(
        test_ds,
        batch_size=model.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=model._batch_collate_fn,
        )

    assert model.model is not None, "model.model not found, please load the model using model.load_from_checkpoint()\
        or model.load_weights_from_checkpoint() which initializes model.model"
    
    pl_model = model.model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pl_model = pl_model.to(device) 
    pl_model.eval()
    
    preds = []
    targets = []
    
    # a hack for visualizing bases weights importance for deeptime related models
    visualized_w = 0
    
    # one epoch of evaluation on test set. Note that for last forecast_horizon points in test set, we only have one prediction
    for batch in tqdm(test_dl, desc="Evaluating on test set"):
        input_series, _, _, target_series = batch
        input_series = input_series.to(device=pl_model.device, dtype=pl_model.dtype)
        
        # target_series = target_series.to(device=pl_model.device, dtype=pl_model.dtype)
        pl_model.y = target_series.to(device=pl_model.device, dtype=pl_model.dtype) # a hack for setting target for twoomp model

        # forward pass
        pred = pl_model((input_series, _))
        
        # # temporary pred for checking if memory view works
        # from einops import repeat
        # batch_size = input_series.shape[0]
        # coords = pl_model.get_coords(input_chunk_length, output_chunk_length).to(input_series.device)
        # time_reprs = repeat(pl_model.inr(coords), '1 t d -> b t d', b=batch_size)
        # horizon_reprs = time_reprs[:, -output_chunk_length:] # [bz, horizon, 256]
        # w, b = pl_model.adaptive_weights(horizon_reprs, pl_model.y) # [bz, 256, 1], [bz, 1, 1]
        # pl_model.learned_w = torch.cat([w, b], dim=1)[..., 0] # shape = (batch_size, layer_size + 1)
        # pred = torch.bmm(horizon_reprs, w) + b # [bz, horizon, 1]
        # pred = pred.view(pred.shape[0], output_chunk_length, pred.shape[2], pl_model.nr_params)
        
        
        if plot_weights:
            visualized_w += pl_model.learned_w.abs().sum(0).detach().cpu().numpy()
        preds.append(pred.detach().cpu())
        targets.append(target_series.detach().cpu())
    
    # preparing torch predictions and targets (not darts.timeseries predictions) 
    preds = torch.cat(preds, dim=0)
    preds = preds.flip(dims=[0]) # flip back since dataset get item is designed in reverse order
    targets = torch.cat(targets, dim=0)
    targets = targets.flip(dims=[0]) # flip back since dataset get item is designed in reverse order
    
    # visualize bases weights importance
    if plot_weights:
        plt.figure()
        plt.plot(range(len(visualized_w)), visualized_w)
        wandb.log({"Plots/weights_importance": plt})
    
    # turn into TimeSeries
    list_backtest_series = []
    for i in tqdm(range(preds.shape[0]), desc="Turn predictions into timeseries"):
        backtest_series = TimeSeries.from_times_and_values(
            test_series.time_index[input_chunk_length+i:input_chunk_length+i+output_chunk_length],
            preds[i,...].detach().cpu().numpy(),
            freq=test_series.freq,
            columns=test_series.components
            )
        list_backtest_series.append(backtest_series)
    return list_backtest_series, preds.squeeze(-1).numpy(), targets.numpy()


class SequeceDataset(Dataset):
    def __init__(
        self, 
        seq_data: torch.tensor, 
        seq_label: Optional(torch.tensor) = None,
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
        pin_memory=True,
        drop_last=False,
        collate_fn=deeptime_model._batch_collate_fn,
        )
    
    
    assert deeptime_model.model is not None, "model.model not found, please load the model using model.load_from_checkpoint()\
        or model.load_weights_from_checkpoint() which initializes model.model"
    
    
    pl_deeptime_model = deeptime_model.model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pl_deeptime_model = pl_deeptime_model.to(device) 
    pl_deeptime_model.eval()
    
    # fixed time representations as memory blocks
    from einops import repeat
    batch_size = input_series.shape[0]
    coords = pl_deeptime_model.get_coords(input_chunk_length, output_chunk_length).to(device)
    time_reprs = repeat(pl_deeptime_model.inr(coords), '1 t d -> b t d', b=batch_size)
    
    # list of signal and label codes
    W_Ls = []
    W_Hs = []
    for batch in tqdm(series_dl, desc="Acquiring signal and label codes (W_L and W_H)"):
        input_series, _, _, target_series = batch
        input_series = input_series.to(device=pl_deeptime_model.device, dtype=pl_deeptime_model.dtype)
        target_series = target_series.to(device=pl_deeptime_model.device, dtype=pl_deeptime_model.dtype)
        
        # # target_series = target_series.to(device=pl_model.device, dtype=pl_model.dtype)
        # pl_deeptime_model.y = target_series.to(device=pl_deeptime_model.device, dtype=pl_deeptime_model.dtype) # a hack for setting target for twoomp model
        
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
    W_Ls = np.concatenate(W_Ls, axis=0)
    W_Hs = np.concatenate(W_Hs, axis=0)
    
    # flip back since dataset get item is designed in reverse order
    W_Ls = W_Ls[::-1, ...]
    W_Hs = W_Hs[::-1, ...]
    
  
def manual_train_seq_model(
    seq_config: DictConfig,
    seq_model: nn.Module,
    train_data: Tuple(np.array, np.array),
    val_data: Tuple(np.array, np.array),
    wandb_log: bool = False,
):
    # create a dataset and data loader
    train_ds = SequeceDataset(torch.tensor(train_data[0]), torch.tensor(train_data[1]), seq_len=seq_config.model.seq_len)
    val_ds = SequeceDataset(torch.tensor(val_data[0]), torch.tensor(val_data[1]), seq_len=seq_config.model.seq_len)
    
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
    seq_model.to(device)
    seq_model.train()
    
    optimizer = torch.optim.Adam(seq_model.parameters(), lr=seq_config.lr)
    criterion = nn.MSELoss()
    
    best_val_loss = np.inf
    early_stop_counter = 0
    for epoch in range(seq_config.epochs):
        # training
        for batch in tqdm(train_dl, desc="Training sequence model"):
            seq_model.zero_grad()
            seq_data, seq_label = batch
            seq_data = seq_data.to(device)
            seq_label = seq_label.to(device)
            seq_pred = seq_model(seq_data)
            loss = criterion(seq_pred, seq_label)
            loss.backward()
            optimizer.step()
            if wandb_log:
                wandb.log({"Seq/train_loss": loss.item(), "Seq/epoch": epoch})
        
        # validation
        seq_model.eval()
        losses = []
        with torch.no_grad():
            for batch in tqdm(val_dl, desc="Validation sequence model"):
                seq_data, seq_label = batch
                seq_data = seq_data.to(device)
                seq_label = seq_label.to(device)
                seq_pred = seq_model(seq_data)
                loss = criterion(seq_pred, seq_label)
                losses.append(loss.item())
                if wandb_log:
                    wandb.log({"Seq/val_loss": loss.item()})
            if np.mean(losses) < best_val_loss:
                early_stop_counter += 1
                best_val_loss = loss.item()
                torch.save(seq_model, f"./seq_model/{seq_config.model.model_name}.pt")
            if early_stop_counter > seq_config.patience:
                log.info("Early stopping")
                return torch.load(f"./seq_model/{seq_config.model.model_name}.pt")
        seq_model.train()
        
    return torch.load(f"./seq_model/{seq_config.model.model_name}.pt")
        
  