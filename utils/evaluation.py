import sys
import os

# sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

import random
import logging
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from typing import List, Optional, Union, Dict, Literal, Tuple
from einops import repeat
from tqdm import tqdm

from darts.timeseries import TimeSeries, concatenate
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel, DEFAULT_DARTS_FOLDER
from darts.models.forecasting.forecasting_model import ForecastingModel, LocalForecastingModel
from darts.utils.data.sequential_dataset import PastCovariatesSequentialDataset


from data.data_registry import DataSeries
from utils.metrics import calculate_metrics


def wandb_log_bases(
    model,
    lookback_len,
    horizon_len,
    exp_dir,
    ):
    # getting time representations (bases)
    coords = model.get_coords(lookback_len, horizon_len).cuda()
    time_reprs = model.inr(coords).squeeze(0).detach().cpu().numpy() # shape = (lookback+horizon, layer_size)
    
    # # csv creation of bases
    # bases_df = pd.DataFrame(time_reprs)
    # bases_df.to_csv(os.path.join(exp_dir, "bases.csv"), index=False)
    
    # wandb table creation of bases
    columns = [f"dim_{i}" for i in range(time_reprs.shape[1])]
    # columns.insert(0, "plots")
    columns.insert(0, "ID")
    # bases_table = wandb.Table(columns=columns)
    id_column = np.arange(time_reprs.shape[0]).reshape(-1, 1)
    bases_data = np.concatenate([id_column, time_reprs], axis=1).tolist()
    # bases_data = time_reprs.tolist() # list of layer_size number of lists each with size lookback+horizon
    bases_table = wandb.Table(data=bases_data, columns=columns)
    wandb.log({"Table/bases" : bases_table})

    
    # plots_table = wandb.Table(columns=["ID", "plots"])
    x_values = np.arange(time_reprs.shape[0])
    _max = np.max(time_reprs)
    _min = np.min(time_reprs) 
    for i in range(time_reprs.shape[1]):
        plt.plot(x_values, time_reprs[:, i], label=f"basis_{i}")
        if i%5==4:
            # plt.axvline(x = lookback_len, color = 'red', linestyle = '--')
            plt.plot((lookback_len,lookback_len),(_min*1.5,_max*1.5), color='red', linestyle='--')
            plt.legend()
            wandb.log({f"Plots/bases_{i-4}_{i}": plt})
            plt.figure()
    plt.figure()   

def wandb_log_results_and_plots(
    metrics: Dict,
    metrics_unscaled: Dict,
    epoch: int,
    output_chunk_length: int, 
    components: List,
    test_series: TimeSeries,
    test_unscaled_series: TimeSeries,
    list_backtest_series: List[TimeSeries],
    list_backtest_unscaled_series: List[TimeSeries],
    train_val_series_trimmed: TimeSeries,
    train_val_unscaled_series_trimmed: TimeSeries,
    
):  
    # logging metrics
    metrics_dict = {
        f"test_best_historical_{result_name}": metrics[result_name].mean() 
        for result_name in metrics.keys() if not np.isnan(metrics[result_name]).any()
        }
    metrics_dict.update({"epoch": epoch})
    wandb.log(metrics_dict)
    
    metrics_unscaled_dict = {
        f"unscaled/test_best_historical_{result_name}": metrics_unscaled[result_name].mean()
        for result_name in metrics_unscaled.keys() if not np.isnan(metrics_unscaled[result_name]).any()
        }
    metrics_unscaled_dict.update({"epoch": epoch})
    wandb.log(metrics_unscaled_dict)
    
    
    # component-based metrics and timeseries plots
    for i, component in reversed(list(enumerate(components))):
        if i<(len(components)-14): # only plot 14 components
            break
        
        # component-based metrics
        component_metrics_dict = {
            f"test_best_historical_{result_name}_{component}": metrics[result_name][..., i].mean()
            for result_name in metrics.keys() if not np.isnan(metrics[result_name]).any()
            }
        component_metrics_dict.update({"epoch": epoch})
        wandb.log(component_metrics_dict)
        
        component_metrics_unscaled_dict = {
            f"unscaled/test_best_historical_{result_name}_{component}": metrics_unscaled[result_name][..., i].mean()
            for result_name in metrics_unscaled.keys() if not np.isnan(metrics_unscaled[result_name]).any()
            }
        component_metrics_unscaled_dict.update({"epoch": epoch})
        wandb.log(component_metrics_unscaled_dict)
    
        
        # timeseries backtest plots (we use last points only)
        backtest_unscaled_series = concatenate([backtest_series[-1:] for backtest_series in list_backtest_unscaled_series]) # semi-colon is important!!!
        plt.figure(figsize=(5, 3))
        train_val_unscaled_series_trimmed[component].plot(label="train_val_"+ component)
        test_unscaled_series[component].plot(label="test_" + component)
        backtest_unscaled_series[component].plot(label="backtest_" + component)
        wandb.log({"Media": plt})
        
        # plot scaled and unscaled series
        if (i==len(components)-1) or ('Target' in component):
            num_rainbow_plots = 100
            plot_interval = output_chunk_length if len(list_backtest_series)//output_chunk_length<num_rainbow_plots else len(list_backtest_series)//num_rainbow_plots
            
            plt.figure(figsize=(5, 3))
            train_val_series_trimmed[component].plot(label="scaled_train_val_"+ component, lw=0.5)
            [series[component].plot(label="pred_" + component)
             for j, series in enumerate(list_backtest_series)
             if j%plot_interval==0]
            wandb.log({"Media": plt})
            
            plt.figure(figsize=(5, 3))
            test_series[component].plot(label="scaled_test_" + component, lw=0.5)
            [series[component].plot(label="pred_" + component)
             for j, series in enumerate(list_backtest_series)
             if j%plot_interval==0]
            wandb.log({"Media": plt})


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
    
    from torch.utils.data import DataLoader
    
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
        plt.plot(range(len(visualized_w)), visualized_w)
        wandb.log({"Plots/weights_importance": plt})
    plt.close()
    
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
  

def seq_sliding_window(
    seq_data: torch.Tensor,
    window_size: int,
):
    seq_data_length = len(seq_data)
    return torch.stack(
        [
        seq_data[i:i+window_size]
        for i in range(seq_data_length-window_size+1)
        ],
        dim=0
        )
    
