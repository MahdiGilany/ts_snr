import sys
import os

# sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
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


def wandb_log_bases(
    pl_model,
    lookback_len,
    horizon_len
    ):
    # getting time representations (bases)
    pl_model.eval()
    coords = pl_model.get_coords(lookback_len, horizon_len).to(next(pl_model.parameters()).device)
    time_reprs = pl_model.inr(coords).squeeze(0).detach().cpu().numpy() # shape = (lookback+horizon, layer_size)
    
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
        

def wandb_log_results_and_plots(
    results: Dict,
    results_noisy: Dict,
    results_unscaled: Dict,
    components: List,
    test_series: TimeSeries,
    test_series_backtest: TimeSeries, # test series for backtest should be noisy if available
    test_unscaled_series: TimeSeries,
    list_backtest_series: List[TimeSeries],
    list_backtest_unscaled_series: List[TimeSeries],
    train_val_series_trimmed: TimeSeries,
    train_val_unscaled_series_trimmed: TimeSeries,
    output_chunk_length: int, 
):
    
    # for visualizing backtest, we use last points only
    backtest_unscaled_series = concatenate([backtest_series[-1:] for backtest_series in list_backtest_unscaled_series]) # semi-colon is important!!!
    
    log.info("logging results to wandb")
    # log best historical, best historical unscaled, best pred
    wandb.log({
        f"test_best_historical_{result_name}": results[result_name].mean() 
        for result_name in results.keys() if not np.isnan(results[result_name]).any()
        })
    
    wandb.log({
        f"test_best_historical_noisy_{result_name}": results_noisy[result_name].mean()
        for result_name in results_noisy.keys() if not np.isnan(results_noisy[result_name]).any()
        })
    
    wandb.log({
        f"test_best_historical_unscaled_{result_name}": results_unscaled[result_name].mean()
        for result_name in results_unscaled.keys() if not np.isnan(results_unscaled[result_name]).any()
        })
    
    
    # plot
    for i, component in reversed(list(enumerate(components))):
        if i<(len(components)-14): # only plot 14 components
            break
        
        wandb.log({
            f"test_best_historical_{result_name}_{component}": results[result_name][..., i].mean()
            for result_name in results.keys() if not np.isnan(results[result_name]).any()
            })
    
        wandb.log({
            f"test_best_historical_unscaled_{result_name}_{component}": results_unscaled[result_name][..., i].mean()
            for result_name in results_unscaled.keys() if not np.isnan(results_unscaled[result_name]).any()
            })
    
        
        plt.figure(figsize=(5, 3))
        train_val_unscaled_series_trimmed[component].plot(label="train_val_"+ component)
        test_unscaled_series[component].plot(label="test_" + component)
        backtest_unscaled_series[component].plot(label="backtest_" + component)
        wandb.log({"Media": plt})
        
        # plot scaled version and a couple of predictions
        if (i==len(components)-1) or ('Target' in component):
            plt.figure(figsize=(5, 3))
            train_val_series_trimmed[component].plot(label="scaled_train_val_"+ component, lw=0.5)
            test_series_backtest[component].plot(label="scaled_noisy_test_" + component, lw=0.5)
            
            no_rainbow_plots = 100
            plot_interval = output_chunk_length if len(list_backtest_series)//output_chunk_length<no_rainbow_plots else len(list_backtest_series)//no_rainbow_plots
            
            [
                series[component].plot(label="pred_" + component)
                for j, series in enumerate(list_backtest_series)
                if j%plot_interval==0
                ]
            wandb.log({"Media": plt})
            
            plt.figure(figsize=(5, 3))
            test_series[component].plot(label="scaled_noisy_test_" + component, lw=0.5)
            [
                series[component].plot(label="pred_" + component)
                for j, series in enumerate(list_backtest_series)
                if j%plot_interval==0
                ]
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
  
  
def eval_model(
    model: TorchForecastingModel,
    configs: DictConfig,
    data_series: DataSeries,
    logging: bool = True,
    forecasting_type: Literal['global', 'local'] = 'global',
    ) -> Union[Dict, DataSeries]:
    """Metrics are found based on the validation series on historical forcasts of 
    the model (using validation to predict future values, read darts documentation).
    The results are reported based on the scaled series and unscaled series.

    Args:
        model (TorchForecastingModel): _description_
        configs (DictConfig): _description_
        train_series (TimeSeries): _description_
        val_series (TimeSeries): _description_
        scaler (Optional[List], optional): _description_. Defaults to None.
        log (bool, optional): _description_. Defaults to True.

    Returns:
        Union[Dict, TimeSeries]: _description_
    """
    
    if forecasting_type=='global':
        #load model    
        model = model.load_from_checkpoint(model_name=configs.model.model_name, best=True) # TODO: needs work if not using wandb or not after training
    
    # input and output chunk length
    input_chunk_length = configs.model.input_chunk_length
    output_chunk_length = configs.model.output_chunk_length
    
    
    # get series from data_series
    train_series = data_series.train_series
    val_series = data_series.val_series
    test_series = data_series.test_series
    test_series_noisy = data_series.test_series_noisy
    scaler = data_series.scaler
    
    # get historical forecasts
    components = test_series.components
    test_series_backtest = test_series if test_series_noisy is None else test_series_noisy # test series for backtest should be noisy if available
    
    if forecasting_type=='global':
        num_trimmed_train_val = max(len(test_series_backtest),input_chunk_length)
        train_val_series_trimmed = concatenate([train_series, val_series])[-num_trimmed_train_val:] # TODO: this is not a good way to do it
        train_val_test_series_trimmed = concatenate([train_val_series_trimmed[-input_chunk_length:], test_series_backtest]) # use a lookback of val for testing
        log.info("Backtesting the model without retraining (testing on test series)")
        list_backtest_series, test_preds, test_targets = historical_forecasts_manual(
            model=model,
            test_series=train_val_test_series_trimmed,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            plot_weights=True if (("deeptime" in configs.model.model_name.lower()) and logging) else False,
            )
    
    elif forecasting_type=='local':
        train_val_series = concatenate([train_series, val_series])[len(val_series):] # removed the size of val series from the beginning for fair comparison with global forecasting
        train_val_test_series = concatenate([train_val_series, test_series_backtest])
        num_trimmed_train_val = max(len(test_series_backtest),input_chunk_length)
        train_val_series_trimmed = train_val_series[-num_trimmed_train_val:]
        log.info("Backtesting the model with retraining (testing on test series)")
        list_backtest_series = model.historical_forecasts(
            train_val_test_series,
            start=test_series_backtest.start_time(),
            forecast_horizon=output_chunk_length,
            retrain=True, # retrain the model at each step since local forecasting model is not parametrized
            verbose=False,
            last_points_only=False,
            )

    
    # unnormalize series
    train_val_unscaled_series_trimmed = scaler.inverse_transform(train_val_series_trimmed) if scaler else train_val_series_trimmed
    test_unscaled_series = scaler.inverse_transform(test_series) if scaler else test_series
    list_backtest_unscaled_series = [scaler.inverse_transform(backtest_series) for backtest_series in list_backtest_series] if scaler else list_backtest_series
    
    
    # calculating results for Target components only, if available (for crypto dataset)
    target_indices = np.array(['Target' in component for component in test_series_backtest.components])
    if target_indices.any():
        components = list(test_series.components[target_indices])
        test_series = test_series[components] 
        test_series_backtest = test_series_backtest[components]
        test_unscaled_series = test_unscaled_series[components]
        list_backtest_series = [backtest_series[components] for backtest_series in list_backtest_series]
        list_backtest_unscaled_series = [backtest_series[components] for backtest_series in list_backtest_unscaled_series]
    
    
    # calculate metrics    
    predictions = np.stack([series._xa.values for series in list_backtest_series], axis=0).squeeze(-1) # (len(test_series)-output_chunk_length+1, output_chunk_length, outdim)
    predictions_unscaled = np.stack([series._xa.values for series in list_backtest_unscaled_series], axis=0).squeeze(-1) # (len(test_series)-output_chunk_length+1, output_chunk_length, outdim)
    log.info("Calculating metrics for backtesting")
    results = calculate_metrics(
        true=sliding_window(test_series, output_chunk_length),
        pred=predictions
        )

    if test_series_noisy is not None:
        log.info("Calculating metrics for backtesting using noisy test")
        results_noisy = calculate_metrics(
            true=sliding_window(test_series_backtest, output_chunk_length),
            pred=predictions
            )

    log.info("Calculating metrics for unnormalized backtesting")
    results_unscaled = calculate_metrics(
        true=sliding_window(test_unscaled_series, output_chunk_length),
        pred=predictions_unscaled
        )
    

    # log best    
    if logging:
        wandb_log_results_and_plots(
            results=results,
            results_noisy=results_noisy,
            results_unscaled=results_unscaled,
            components=components,
            test_series=test_series,
            test_series_backtest=test_series_backtest,
            test_unscaled_series=test_unscaled_series,
            list_backtest_series=list_backtest_series,
            list_backtest_unscaled_series=list_backtest_unscaled_series,
            train_val_series_trimmed=train_val_series_trimmed,
            train_val_unscaled_series_trimmed=train_val_unscaled_series_trimmed,
            output_chunk_length=output_chunk_length,
            )
        if "deeptime" in configs.model.model_name.lower():
            wandb_log_bases(model.model, input_chunk_length, output_chunk_length)
                
    return results, list_backtest_series  


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
    

# evaluation for enhanced deeptime model with sequence model
def historical_forecasts_with_seq_manual(
    model: TorchForecastingModel,
    seq_model: nn.Module,
    test_series: TimeSeries,
    train_val_lookback_codes: np.ndarray,
    seq_len: int,
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
    seq_model = seq_model.to(device)
    
    pl_model.eval()
    seq_model.eval()
    
    preds = []
    targets = []
    
    # a hack for visualizing bases weights importance for deeptime related models
    visualized_w = 0
    
    # time representation is took out of for loop for faster computation
    coords = pl_model.get_coords(input_chunk_length, output_chunk_length).to(device)
    single_time_reprs = pl_model.inr(coords)
    
    # seq_WL = torch.tensor(train_val_lookback_codes, dtype=pl_model.dtype, device=device)[1:, ...] # [seq_len-1, layer_size + 1 (#codes), 1 or output_dim]
    # seq_WL = seq_WL.flip(dims=[0]) # fliping # order backward in time
    
    seq_WH = torch.tensor(train_val_lookback_codes, dtype=pl_model.dtype, device=device) # [seq_len + horizon, layer_size + 1 (#codes), 1 or output_dim]
    seq_WH = seq_WH.flip(dims=[0]) # fliping # order backward in time

    # one epoch of evaluation on test set. Note that for last forecast_horizon points in test set, we only have one prediction
    for batch in tqdm(test_dl, desc="Evaluating on test set"):
        input_series, _, _, target_series = batch
        input_series = input_series.to(device=pl_model.device, dtype=pl_model.dtype)
        target_series = target_series.to(device=pl_model.device, dtype=pl_model.dtype)
        
        # target_series = target_series.to(device=pl_model.device, dtype=pl_model.dtype)
        pl_model.y = target_series.to(device=pl_model.device, dtype=pl_model.dtype) # a hack for setting target for twoomp model

        # forward pass
        # pred = pl_model((input_series, _))
        
        # repeating time reprs for batch size
        batch_size = input_series.shape[0]
        time_reprs = repeat(single_time_reprs, '1 t d -> b t d', b=batch_size)
        
        # look back and horizon dictionaries        
        lookback_reprs = time_reprs[:, :-output_chunk_length] # shape = (batch_size, horizon, layer_size)
        horizon_reprs = time_reprs[:, -output_chunk_length:]
        
        w, b = pl_model.adaptive_weights(lookback_reprs, input_series) # [bz, 256, 1], [bz, 1, 1]
        W_L = torch.cat([w, b], dim=1) # shape = (batch_size, layer_size + 1, 1)
        
        # predicting WH using seq model
        # seq_WL = torch.cat([W_L, seq_WL], dim=0).flip(dims=[0])  # shape = (seq_len + batch_size - 1, layer_size + 1, 1) # order forward in time
        # sliding_window_seq_WL =  seq_sliding_window(seq_data=seq_WL, window_size=seq_len) # shape = (batch_size, seq_len, layer_size + 1, 1) # order forward in time
        
        # reset seq_WL
        # seq_WL = seq_WL[-seq_len:, ...][1:,...].flip(dims=[0]) # shape = (seq_len -1, layer_size + 1, 1) # order backward in time
        
        # prediction
        # predicted_WH = seq_model(sliding_window_seq_WL).flip(dims=[0])[:,-1,...]  # shape = (batch_size, layer_size + 1, 1) # order backward in time
        # predicted_WH = predicted_WH + W_L 
        
        # predicting WH using seq model
        w, b = pl_model.adaptive_weights(horizon_reprs, target_series) # [bz, 256, 1], [bz, 1, 1]
        W_H = torch.cat([w, b], dim=1) # shape = (batch_size, layer_size + 1, 1)
        seq_WH = torch.cat([W_H, seq_WH], dim=0).flip(dims=[0])  # shape = (seq_len + horizon + batch_size , layer_size + 1, 1) # order forward in time
        sliding_window_seq_WH =  seq_sliding_window(seq_data=seq_WH[:-output_chunk_length,...], window_size=seq_len) # shape = (batch_size + 1, seq_len, layer_size + 1, 1) # order forward in time
        predicted_WH = seq_model(sliding_window_seq_WH).flip(dims=[0])[:,-1,...]  # shape = (batch_size + 1, layer_size + 1, 1) # order backward in time
        predicted_WH = predicted_WH[1:, ...] # shape = (batch_size, layer_size + 1, 1) # order backward in time
        predicted_WH = predicted_WH + W_L 
        # reset seq_WH
        seq_WH = seq_WH[-seq_len-output_chunk_length:, ...].flip(dims=[0]) # shape = (seq_len, layer_size + 1, 1) # order backward in time

        
        pred = torch.bmm(horizon_reprs, predicted_WH[:, :-1]) + predicted_WH[:, -1:] # [bz, horizon, 1] 
        pred = pred.view(pred.shape[0], output_chunk_length, pred.shape[2], pl_model.nr_params)
        
        
        if plot_weights:
            visualized_w += predicted_WH.abs().sum(0).detach().cpu().numpy()
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
  

def eval_twostage_model(
    model: TorchForecastingModel,
    seq_model: nn.Module,
    configs: DictConfig,
    data_series: DataSeries,
    train_val_lookback_codes: Tuple[np.ndarray, np.ndarray],
    logging: bool = True,
    forecasting_type: Literal['global', 'local'] = 'global',
    ) -> Union[Dict, DataSeries]:
    """Metrics are found based on the validation series on historical forcasts of 
    the model (using validation to predict future values, read darts documentation).
    The results are reported based on the scaled series and unscaled series.

    Args:
        model (TorchForecastingModel): _description_
        configs (DictConfig): _description_
        train_series (TimeSeries): _description_
        val_series (TimeSeries): _description_
        scaler (Optional[List], optional): _description_. Defaults to None.
        log (bool, optional): _description_. Defaults to True.

    Returns:
        Union[Dict, TimeSeries]: _description_
    """
        
    # input and output chunk length
    input_chunk_length = configs.model.input_chunk_length
    output_chunk_length = configs.model.output_chunk_length
    
    
    # get series from data_series
    train_series = data_series.train_series
    val_series = data_series.val_series
    test_series = data_series.test_series
    test_series_noisy = data_series.test_series_noisy
    scaler = data_series.scaler
    
    # get historical forecasts
    components = test_series.components
    test_series_backtest = test_series if test_series_noisy is None else test_series_noisy # test series for backtest should be noisy if available
    
    num_trimmed_train_val = max(len(test_series_backtest),input_chunk_length)
    train_val_series_trimmed = concatenate([train_series, val_series])[-num_trimmed_train_val:] # TODO: this is not a good way to do it
    train_val_test_series_trimmed = concatenate([train_val_series_trimmed[-input_chunk_length:], test_series_backtest]) # use a lookback of val for testing
    
    # lookback codes from validation series (to be able to have prediction from the first step of test series)
    seq_len = configs.model.sequence_config.model.seq_len
    # train_val_lookback_codes = train_val_lookback_codes[-seq_len:, ...]
    
    log.info("Backtesting the model without retraining (testing on test series)")
    list_backtest_series, test_preds, test_targets = historical_forecasts_with_seq_manual(
            model=model,
            seq_model=seq_model,
            test_series=train_val_test_series_trimmed,
            train_val_lookback_codes=train_val_lookback_codes,
            seq_len=seq_len,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            plot_weights=True if (("deeptime" in configs.model.model_name.lower()) and logging) else False,
            )
    
    
    # unnormalize series
    train_val_unscaled_series_trimmed = scaler.inverse_transform(train_val_series_trimmed) if scaler else train_val_series_trimmed
    test_unscaled_series = scaler.inverse_transform(test_series) if scaler else test_series
    list_backtest_unscaled_series = [scaler.inverse_transform(backtest_series) for backtest_series in list_backtest_series] if scaler else list_backtest_series
    
    
    # calculating results for Target components only, if available (for crypto dataset)
    target_indices = np.array(['Target' in component for component in test_series_backtest.components])
    if target_indices.any():
        components = list(test_series.components[target_indices])
        test_series = test_series[components] 
        test_series_backtest = test_series_backtest[components]
        test_unscaled_series = test_unscaled_series[components]
        list_backtest_series = [backtest_series[components] for backtest_series in list_backtest_series]
        list_backtest_unscaled_series = [backtest_series[components] for backtest_series in list_backtest_unscaled_series]
    
    
    # calculate metrics    
    predictions = np.stack([series._xa.values for series in list_backtest_series], axis=0).squeeze(-1) # (len(test_series)-output_chunk_length+1, output_chunk_length, outdim)
    predictions_unscaled = np.stack([series._xa.values for series in list_backtest_unscaled_series], axis=0).squeeze(-1) # (len(test_series)-output_chunk_length+1, output_chunk_length, outdim)
    log.info("Calculating metrics for backtesting")
    results = calculate_metrics(
        true=sliding_window(test_series, output_chunk_length),
        pred=predictions
        )

    if test_series_noisy is not None:
        log.info("Calculating metrics for backtesting using noisy test")
        results_noisy = calculate_metrics(
            true=sliding_window(test_series_backtest, output_chunk_length),
            pred=predictions
            )

    log.info("Calculating metrics for unnormalized backtesting")
    results_unscaled = calculate_metrics(
        true=sliding_window(test_unscaled_series, output_chunk_length),
        pred=predictions_unscaled
        )
    

    # log best    
    if logging:
        wandb_log_results_and_plots(
            results=results,
            results_noisy=results_noisy,
            results_unscaled=results_unscaled,
            components=components,
            test_series=test_series,
            test_series_backtest=test_series_backtest,
            test_unscaled_series=test_unscaled_series,
            list_backtest_series=list_backtest_series,
            list_backtest_unscaled_series=list_backtest_unscaled_series,
            train_val_series_trimmed=train_val_series_trimmed,
            train_val_unscaled_series_trimmed=train_val_unscaled_series_trimmed,
            output_chunk_length=output_chunk_length,
            )
        if "deeptime" in configs.model.model_name.lower():
            wandb_log_bases(model.model, input_chunk_length, output_chunk_length)
                
    return results, list_backtest_series  


# evaluation for enhanced deeptime model with sequence model
def historical_forecasts_with_metadeeptime_manual(
    model: nn.Module,
    meta_config: DictConfig,
    test_series: TimeSeries,
    input_chunk_length: int,
    output_chunk_length: int,
    num_shots: int=5,
    plot_weights: bool = False,
    ):
    from src.utils.training import maml_fast_adapt
    from torch.utils.data import DataLoader
    # manually build test dataset and dataloader
    from src.utils.training import MetaDataset
    test_ds = MetaDataset(
        test_series,
        num_shots=num_shots,
        lookback=input_chunk_length,
        horizon=output_chunk_length,
        )
    test_dl = DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device=device, dtype=test_ds[0][0].dtype) 
    model.eval()
    
    preds = []
    targets = []
    
    # a hack for visualizing bases weights importance for deeptime related models
    visualized_w = 0

    preds = []
    targets = []
    # one epoch of evaluation on test set. Note that for last forecast_horizon points in test set, we only have one prediction
    for batch in tqdm(test_dl, desc="Evaluating on test set"):
        support_seqs, support_labels, query_seqs, query_labels = batch
        support_seqs = support_seqs.to(device)
        support_labels = support_labels.to(device)
        query_seqs = query_seqs.to(device)
        query_labels = query_labels.to(device)
        
        batch_size, num_shots, lookback, outdim = support_seqs.shape
        
        ## turn on for MAML
        # train_errors = []
        # for i in range(batch_size):
        #     cloned_maml_model = model.clone() # head of the model
        #     # fast adapt
        #     train_error = maml_fast_adapt(cloned_maml_model, support_seqs[i], support_labels[i], val=True, adaptation_steps=meta_config.adapt_steps)
        #     train_errors.append(train_error)
        #     #meta test
        #     pred = cloned_maml_model(query_seqs[i:i+1])
        #     preds.append(pred.detach().cpu())
        #     targets.append(query_labels[i:i+1].detach().cpu())
        
        ## turn on for closedform meta deeptime
        # fast adapt
        support_latent = model(support_seqs.reshape(-1, lookback, outdim)).reshape(batch_size, num_shots, -1) # (batch_size, shots, horizon)
        support_latentTsupport_latent = torch.matmul(support_latent.transpose(1, 2), support_latent) # (batch_size, horizon, horizon)
        support_latentTsupport_labels = torch.matmul(support_latent.transpose(1, 2), support_labels.squeeze(-1)) # (batch_size, horizon, horizon)
        support_latentTsupport_latent.diagonal(dim1=-2, dim2=-1).add_(0.6)
        W_HH = torch.matmul(torch.inverse(support_latentTsupport_latent), support_latentTsupport_labels) # (batch_size, horizon, horizon)
        # W_HH.diagonal(dim1=-2, dim2=-1).add_(100.) # start from I
        # W_HH = W_HH / 100.0
        
        #meta test
        query_latent = model(query_seqs).squeeze(-1).reshape(batch_size, 1, -1) # (batch_size, 1, horizon)
        pred = torch.matmul(query_latent, W_HH).transpose(1,2) # (batch_size, horizon, 1)
        preds.append(pred.detach().cpu())
        targets.append(query_labels.detach().cpu())
        
        # for i in range(batch_size):
        #     # fast adapt
        #     support_latent = model(support_seqs[i]).squeeze(-1).unsqueeze(0) # (1, shots, horizon)
        #     support_latentTsupport_latent = torch.matmul(support_latent.transpose(1, 2), support_latent) # (1, horizon, horizon)
        #     support_latentTsupport_labels = torch.matmul(support_latent.transpose(1, 2), support_labels[i].squeeze(-1).unsqueeze(0)) # (1, horizon, horizon)
        #     W_HH = torch.matmul(torch.inverse(support_latentTsupport_latent), support_latentTsupport_labels) # (1, horizon, horizon)
            
        #     #meta test
        #     query_latent = model(query_seqs[i:i+1]).squeeze(-1).unsqueeze(0) # (1, 1, horizon)
        #     pred = torch.matmul(query_latent, W_HH).squeeze(0).unsqueeze(-1) # (1, horizon, 1)
        #     preds.append(pred.detach().cpu())
        #     targets.append(query_labels[i:i+1].detach().cpu())


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
    offset = input_chunk_length + output_chunk_length + num_shots
    for i in tqdm(range(preds.shape[0]), desc="Turn predictions into timeseries"):
        backtest_series = TimeSeries.from_times_and_values(
            test_series.time_index[offset+i:offset+i+output_chunk_length],
            preds[i,...].detach().cpu().numpy(),
            freq=test_series.freq,
            columns=test_series.components
            )
        list_backtest_series.append(backtest_series)
    return list_backtest_series, preds.squeeze(-1).numpy(), targets.numpy()
  
  
def eval_twostage_metadeeptime_model(
    model: TorchForecastingModel,
    configs: DictConfig,
    data_series: DataSeries,
    logging: bool = True,
    forecasting_type: Literal['global', 'local'] = 'global',
    ) -> Union[Dict, DataSeries]:
    """Metrics are found based on the validation series on historical forcasts of 
    the model (using validation to predict future values, read darts documentation).
    The results are reported based on the scaled series and unscaled series.

    Args:
        model (TorchForecastingModel): _description_
        configs (DictConfig): _description_
        train_series (TimeSeries): _description_
        val_series (TimeSeries): _description_
        scaler (Optional[List], optional): _description_. Defaults to None.
        log (bool, optional): _description_. Defaults to True.

    Returns:
        Union[Dict, TimeSeries]: _description_
    """
        
    # input and output chunk length
    input_chunk_length = configs.model.input_chunk_length
    output_chunk_length = configs.model.output_chunk_length
    
    
    # get series from data_series
    train_series = data_series.train_series
    val_series = data_series.val_series
    test_series = data_series.test_series
    test_series_noisy = data_series.test_series_noisy
    scaler = data_series.scaler
    
    # get historical forecasts
    components = test_series.components
    test_series_backtest = test_series if test_series_noisy is None else test_series_noisy # test series for backtest should be noisy if available
    
    
    meta_config = configs.model.meta_config
    num_trimmed_train_val = max(len(test_series_backtest),input_chunk_length)
    train_val_series_trimmed = concatenate([train_series, val_series])[-num_trimmed_train_val:] # TODO: this is not a good way to do it
    train_val_test_series_trimmed = concatenate([train_val_series_trimmed[-input_chunk_length-output_chunk_length-meta_config.num_shots:], test_series_backtest]) # use a lookback of val for testing
    
    
    log.info("Backtesting the model without retraining (testing on test series)")
    list_backtest_series, test_preds, test_targets = historical_forecasts_with_metadeeptime_manual(
            model=model,
            meta_config=configs.model.meta_config,
            test_series=train_val_test_series_trimmed,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            plot_weights= False #True if (("deeptime" in configs.model.model_name.lower()) and logging) else False,
            )
    
    
    # unnormalize series
    train_val_unscaled_series_trimmed = scaler.inverse_transform(train_val_series_trimmed) if scaler else train_val_series_trimmed
    test_unscaled_series = scaler.inverse_transform(test_series) if scaler else test_series
    list_backtest_unscaled_series = [scaler.inverse_transform(backtest_series) for backtest_series in list_backtest_series] if scaler else list_backtest_series
    
    
    # calculating results for Target components only, if available (for crypto dataset)
    target_indices = np.array(['Target' in component for component in test_series_backtest.components])
    if target_indices.any():
        components = list(test_series.components[target_indices])
        test_series = test_series[components] 
        test_series_backtest = test_series_backtest[components]
        test_unscaled_series = test_unscaled_series[components]
        list_backtest_series = [backtest_series[components] for backtest_series in list_backtest_series]
        list_backtest_unscaled_series = [backtest_series[components] for backtest_series in list_backtest_unscaled_series]
    
    
    # calculate metrics    
    predictions = np.stack([series._xa.values for series in list_backtest_series], axis=0).squeeze(-1) # (len(test_series)-output_chunk_length+1, output_chunk_length, outdim)
    predictions_unscaled = np.stack([series._xa.values for series in list_backtest_unscaled_series], axis=0).squeeze(-1) # (len(test_series)-output_chunk_length+1, output_chunk_length, outdim)
    log.info("Calculating metrics for backtesting")
    results = calculate_metrics(
        true=sliding_window(test_series, output_chunk_length),
        pred=predictions
        )

    if test_series_noisy is not None:
        log.info("Calculating metrics for backtesting using noisy test")
        results_noisy = calculate_metrics(
            true=sliding_window(test_series_backtest, output_chunk_length),
            pred=predictions
            )

    log.info("Calculating metrics for unnormalized backtesting")
    results_unscaled = calculate_metrics(
        true=sliding_window(test_unscaled_series, output_chunk_length),
        pred=predictions_unscaled
        )
    

    # log best    
    if logging:
        wandb_log_results_and_plots(
            results=results,
            results_noisy=results_noisy,
            results_unscaled=results_unscaled,
            components=components,
            test_series=test_series,
            test_series_backtest=test_series_backtest,
            test_unscaled_series=test_unscaled_series,
            list_backtest_series=list_backtest_series,
            list_backtest_unscaled_series=list_backtest_unscaled_series,
            train_val_series_trimmed=train_val_series_trimmed,
            train_val_unscaled_series_trimmed=train_val_unscaled_series_trimmed,
            output_chunk_length=output_chunk_length,
            )
        if "deeptime" in configs.model.model_name.lower():
            wandb_log_bases(model, input_chunk_length, output_chunk_length)
                
    return results, list_backtest_series  



#############################################################

# def wandb_log_bases(pl_model, lookback_len, horizon_len):
#     # getting time representations (bases)
#     pl_model.eval()
#     coords = pl_model.get_coords(lookback_len, horizon_len).to(next(pl_model.parameters()).device)
#     time_reprs = pl_model.inr(coords).squeeze(0).detach().cpu().numpy() # shape = (lookback+horizon, layer_size)
    
#     # wandb table creation of bases
#     columns = [f"dim_{i}" for i in range(time_reprs.shape[1])]
#     # columns.insert(0, "plots")
#     columns.insert(0, "ID")
#     # bases_table = wandb.Table(columns=columns)
#     id_column = np.arange(time_reprs.shape[0]).reshape(-1, 1)
#     bases_data = np.concatenate([id_column, time_reprs], axis=1).tolist()
#     # bases_data = time_reprs.tolist() # list of layer_size number of lists each with size lookback+horizon
#     bases_table = wandb.Table(data=bases_data, columns=columns)
#     wandb.log({"Table/bases" : bases_table})

    
#     # plots_table = wandb.Table(columns=["ID", "plots"])
#     x_values = np.arange(time_reprs.shape[0])
#     for i in range(time_reprs.shape[1]):
#         # y_values = time_reprs[:, i]
#         # data = [[x, y] for (x, y) in zip(x_values, y_values)]
#         # table = wandb.Table(data=data, columns = ["x", "y"])
#         # line_plots_i = wandb.plot.line(table, "x", "y")#, title=f"basis_{i}")
#         # plots_table.add_data(i, wandb.Image(line_plots_i))
#         plt.plot(x_values, time_reprs[:, i])
#         if i%5==4:
#             wandb.log({f"Plots/bases_{i-4}_{i}": plt})
#             plt.figure()
#         # plots_table.add_data(i, wandb.Image(plt))
        
#     # wandb.log({"Table/plots" : plots_table})
        

#     # plt.figure()
#     # for i in range(time_reprs.shape[1]):
#     #     plt.plot(time_reprs[:, i], label=f"basis_{i}")
#     # wandb.log({"Bases_visualization": plt})
    
#     # x_values = np.arange(time_reprs.shape[0])
#     # for i in range(time_reprs.shape[1]):
#     #     y_values = time_reprs[:, i]
#     #     # data = [[x, y] for (x, y) in zip(x_values, y_values)]
#     #     # table = wandb.Table(data=data, columns = ["x", "y"])
#     #     # line_plots_i = wandb.plot.line(table, "x", "y")#, title=f"basis_{i}")
        
#     #     df = pd.DataFrame(dict(
#     #         x = x_values,
#     #         y = y_values
#     #         ))
        
#     #     # Create path for Plotly figure
#     #     path_to_plotly_html = f"./plotly_figure_{i}.html"
        
#     #     # plotly figure
#     #     line_plot_i = px.line(df, x="x", y="y") #, title=f"basis_{i}")
#     #     line_plot_i.write_html(path_to_plotly_html, auto_play = False) 
        
        
#     #     bases_data[i].insert(0, wandb.Html(path_to_plotly_html))
#     #     bases_data[i].insert(0, i)
#     #     # bases_table.add_data(*data_list)

#     # bases_table = wandb.Table(data=bases_data, columns=columns)
#     # wandb.log({"Table/bases" : bases_table})
#     # breakpoint()


# def eval_globalforecasting_model(
#     model: TorchForecastingModel,
#     configs: DictConfig,
#     data_series: DataSeries,
#     logging: bool = True,
#     ) -> Union[Dict, DataSeries]:
#     """Metrics are found based on the validation series on historical forcasts of 
#     the model (using validation to predict future values, read darts documentation).
#     The results are reported based on the scaled series and unscaled series.

#     Args:
#         model (TorchForecastingModel): _description_
#         configs (DictConfig): _description_
#         train_series (TimeSeries): _description_
#         val_series (TimeSeries): _description_
#         scaler (Optional[List], optional): _description_. Defaults to None.
#         log (bool, optional): _description_. Defaults to True.

#     Returns:
#         Union[Dict, TimeSeries]: _description_
#     """
    
#     #load model    
#     model = model.load_from_checkpoint(model_name=configs.model.model_name, best=True) # TODO: needs work if not using wandb or not after training
    
#     # input and output chunk length
#     input_chunk_length = configs.model.input_chunk_length
#     output_chunk_length = configs.model.output_chunk_length
    
    
#     # get series from data_series
#     train_series = data_series.train_series
#     val_series = data_series.val_series
#     test_series = data_series.test_series
#     test_series_noisy = data_series.test_series_noisy
#     scaler = data_series.scaler
    
#     # get historical forecasts
#     components = test_series.components
#     test_series_backtest = test_series if test_series_noisy is None else test_series_noisy # test series for backtest should be noisy if available
#     num_trimmed_train_val = max(len(test_series_backtest),input_chunk_length)
#     train_val_series_trimmed = concatenate([train_series, val_series])[-num_trimmed_train_val:] # TODO: this is not a good way to do it
#     train_val_test_series_trimmed = concatenate([train_val_series_trimmed[-input_chunk_length:], test_series_backtest]) # use a lookback of val for testing

#     log.info("Backtesting the model without retraining (testing on test series)")
#     list_backtest_series, test_preds, test_targets = historical_forecasts_manual(
#         model=model,
#         test_series=train_val_test_series_trimmed,
#         input_chunk_length=input_chunk_length,
#         output_chunk_length=output_chunk_length,
#         )
    
    
#     # unnormalize series
#     train_val_unscaled_series_trimmed = scaler.inverse_transform(train_val_series_trimmed) if scaler else train_val_series_trimmed
#     test_unscaled_series = scaler.inverse_transform(test_series) if scaler else test_series
#     list_backtest_unscaled_series = [scaler.inverse_transform(backtest_series) for backtest_series in list_backtest_series] if scaler else list_backtest_series
    
    
#     # calculating results for Target components only, if available (for crypto dataset)
#     target_indices = np.array(['Target' in component for component in test_series_backtest.components])
#     if target_indices.any():
#         components = list(test_series.components[target_indices])
#         test_series = test_series[components] 
#         test_series_backtest = test_series_backtest[components]
#         test_unscaled_series = test_unscaled_series[components]
#         list_backtest_series = [backtest_series[components] for backtest_series in list_backtest_series]
    
    
#     # calculate metrics    
#     predictions = np.stack([series._xa.values for series in list_backtest_series], axis=0).squeeze(-1) # (len(test_series)-output_chunk_length+1, output_chunk_length, outdim)
#     predictions_unscaled = np.stack([series._xa.values for series in list_backtest_unscaled_series], axis=0).squeeze(-1) # (len(test_series)-output_chunk_length+1, output_chunk_length, outdim)
#     log.info("Calculating metrics for backtesting")
#     results = calculate_metrics(
#         true=sliding_window(test_series, output_chunk_length),
#         pred=predictions
#         )

#     if test_series_noisy is not None:
#         log.info("Calculating metrics for backtesting using noisy test")
#         results_noisy = calculate_metrics(
#             true=sliding_window(test_series_backtest, output_chunk_length),
#             pred=predictions
#             )

#     log.info("Calculating metrics for unnormalized backtesting")
#     results_unscaled = calculate_metrics(
#         true=sliding_window(test_unscaled_series, output_chunk_length),
#         pred=predictions_unscaled
#         )
    

#     # log best    
#     if logging:
#         wandb_log_results_and_plots(
#             results=results,
#             results_noisy=results_noisy,
#             results_unscaled=results_unscaled,
#             components=components,
#             test_series=test_series,
#             test_series_backtest=test_series_backtest,
#             test_unscaled_series=test_unscaled_series,
#             list_backtest_series=list_backtest_series,
#             list_backtest_unscaled_series=list_backtest_unscaled_series,
#             train_val_series_trimmed=train_val_series_trimmed,
#             train_val_unscaled_series_trimmed=train_val_unscaled_series_trimmed,
#             output_chunk_length=output_chunk_length,
#             )
                
#     return results, list_backtest_series  

    
# def eval_localforecasting_model( 
#     model: LocalForecastingModel,
#     configs: DictConfig,
#     data_series: DataSeries,
#     logging: bool = True,
#     ) -> Union[Dict, DataSeries]:
#     #load model        
    
#     # input and output chunk length
#     input_chunk_length = configs.model.input_chunk_length
#     output_chunk_length = configs.model.output_chunk_length
    
#     # get series from data_series
#     train_series = data_series.train_series
#     val_series = data_series.val_series
#     test_series = data_series.test_series
#     test_series_noisy = data_series.test_series_noisy
#     scaler = data_series.scaler
    
#     # get historical forecasts
#     from darts.timeseries import concatenate 
#     components = test_series.components
#     test_series_backtest = test_series if test_series_noisy is None else test_series_noisy # test series for backtest should be noisy if available
#     train_val_series = concatenate([train_series, val_series])[len(val_series):] # remove the size of val series from the beginning for fair comparison
#     train_val_test_series = concatenate([train_val_series, test_series_backtest])
    

#     log.info("Backtesting the model with retraining (testing on test series)")
#     list_backtest_series = model.historical_forecasts(
#         train_val_test_series,
#         start=test_series_backtest.start_time(),
#         forecast_horizon=output_chunk_length,
#         retrain=True, # retrain the model at each step since local forecasting model is not parametrized
#         verbose=False,
#         last_points_only=False,
#         )
    
#     num_trimmed_train_val = max(len(test_series_backtest),input_chunk_length)
#     train_val_series_trimmed = train_val_series[-num_trimmed_train_val:]
    
#     # unnormalize series
#     train_val_unscaled_series_trimmed = scaler.inverse_transform(train_val_series_trimmed) if scaler else train_val_series_trimmed
#     test_unscaled_series = scaler.inverse_transform(test_series) if scaler else test_series
#     list_backtest_unscaled_series = [scaler.inverse_transform(backtest_series) for backtest_series in list_backtest_series] if scaler else list_backtest_series
    
#     # calculate metrics    
#     log.info("Calculating metrics for backtesting")
#     results = calculate_metrics(
#         [test_series]*len(list_backtest_series),
#         list_backtest_series,
#         reduction=np.array,
#         verbose=True,
#         n_jobs=-1
#         )    
    
#     if test_series_noisy is not None:
#         log.info("Calculating metrics for backtesting using noisy test")
#         results_noisy = calculate_metrics(
#             [test_series_noisy]*len(list_backtest_series),
#             list_backtest_series,
#             reduction=np.array,
#             verbose=True,
#             n_jobs=-1
#             )
    
#     log.info("Calculating metrics for unnormalized backtesting")
#     results_unscaled = calculate_metrics(
#         [test_unscaled_series]*len(list_backtest_unscaled_series),
#         list_backtest_unscaled_series, 
#         reduction=np.array,
#         verbose=True,
#         n_jobs=-1
#         )
    
    
    
#     results = {
#         result_name: np.vstack(results[result_name])
#         for result_name in results.keys()
#         if not np.isnan(np.array(results[result_name])).any()
#         }
#     results_noisy = {
#         result_name: np.vstack(results_noisy[result_name])
#         for result_name in results_noisy.keys()
#         if not np.isnan(np.array(results_noisy[result_name])).any()
#         }
#     results_unscaled = {
#         result_name: np.vstack(results_unscaled[result_name])
#         for result_name in results_unscaled.keys()
#         if not np.isnan(np.array(results_unscaled[result_name])).any()
#         }
    
#     # for visualizing backtest, we use last points only
#     backtest_unscaled_series = concatenate([_series[-1:] for _series in list_backtest_unscaled_series]) # semi-colon is important!!!
    
#     # log best    
#     if logging:
#         log.info("logging results to wandb")
#         # log best historical, best historical unscaled, best pred
#         wandb.log({
#             f"test_best_historical_{result_name}": results[result_name].mean() 
#             for result_name in results.keys() if not np.isnan(results[result_name]).any()
#             })
        
#         wandb.log({
#             f"test_best_historical_noisy_{result_name}": results_noisy[result_name].mean()
#             for result_name in results_noisy.keys() if not np.isnan(results_noisy[result_name]).any()
#             })
        
#         wandb.log({
#             f"test_best_historical_unscaled_{result_name}": results_unscaled[result_name].mean()
#             for result_name in results_unscaled.keys() if not np.isnan(results_unscaled[result_name]).any()
#             })
        
        
#         # plot
#         for i, component in reversed(list(enumerate(components))):
#             if i<(len(components)-10): # only plot 10 components
#                 break
            
#             wandb.log({
#                 f"test_best_historical_{result_name}_{component}": results[result_name][..., i].mean()
#                 for result_name in results.keys() if not np.isnan(results[result_name]).any()
#                 })
        
#             wandb.log({
#                 f"test_best_historical_unscaled_{result_name}_{component}": results_unscaled[result_name][..., i].mean()
#                 for result_name in results_unscaled.keys() if not np.isnan(results_unscaled[result_name]).any()
#                 })
            
            
#             plt.figure(figsize=(5, 3))
#             train_val_unscaled_series_trimmed[component].plot(label="train_val_"+ component)
#             test_unscaled_series[component].plot(label="test_" + component)
#             backtest_unscaled_series[component].plot(label="backtest_" + component)
#             # plt.title(configs.model.model_name + configs.data.dataset_name + component)
#             wandb.log({"Media": plt})
            
#             # plot scaled version and a couple of predictions
#             if i==len(components)-1:
#                 plt.figure(figsize=(5, 3))
#                 train_val_series_trimmed[component].plot(label="scaled_train_val_"+ component, lw=0.5)
#                 test_series_backtest[component].plot(label="scaled_noisy_test_" + component, lw=0.5)
#                 [
#                     series[component].plot(label="pred_" + component)
#                     for j, series in enumerate(list_backtest_series)
#                     if j%output_chunk_length==0
#                  ]
#                 wandb.log({"Media": plt})
                
#     return results, list_backtest_series
            
    
# def eval_model(
#     model: TorchForecastingModel,
#     configs: DictConfig,
#     data_series: DataSeries,
#     logging: bool = True,
#     ) -> Union[Dict, DataSeries]:
#     """Metrics are found based on the validation series on historical forcasts of 
#     the model (using validation to predict future values, read darts documentation).
#     The results are reported based on the scaled series and unscaled series.

#     Args:
#         model (TorchForecastingModel): _description_
#         configs (DictConfig): _description_
#         train_series (TimeSeries): _description_
#         val_series (TimeSeries): _description_
#         scaler (Optional[List], optional): _description_. Defaults to None.
#         log (bool, optional): _description_. Defaults to True.

#     Returns:
#         Union[Dict, TimeSeries]: _description_
#     """
    
#     #load model    
#     # manual loading
#     # from darts.models.forecasting.torch_forecasting_model import _get_checkpoint_fname, _get_checkpoint_folder
#     # model_name = configs.model.model_name
#     # work_dir = os.path.join(os.getcwd(), DEFAULT_DARTS_FOLDER)
#     # file_name = _get_checkpoint_fname(work_dir, model_name, best=True)
#     # checkpoint_dir = _get_checkpoint_folder(work_dir, model_name)
#     # file_path = os.path.join(checkpoint_dir, file_name)
#     # pl_model = model.model.load_from_checkpoint(file_path)
#     # model.model = model.model.load_from_checkpoint('/home/mahdigilany/offline_codes/codes/ts_snr/logs/experiments/runs/deeptime_test_model_load/2023-07-14_15-24-04/darts_logs/deeptime/checkpoints/best-epoch=20-val_loss=0.08.ckpt')
#     model = model.load_from_checkpoint(model_name=configs.model.model_name, best=True) # TODO: needs work if not using wandb or not after training
    
#     # input and output chunk length
#     input_chunk_length = configs.model.input_chunk_length
#     output_chunk_length = configs.model.output_chunk_length
    
    
#     # get series from data_series
#     train_series = data_series.train_series
#     val_series = data_series.val_series
#     test_series = data_series.test_series
#     test_series_noisy = data_series.test_series_noisy
#     scaler = data_series.scaler
    
#     # get historical forecasts
#     from darts.timeseries import concatenate 
#     components = test_series.components
#     test_series_backtest = test_series if test_series_noisy is None else test_series_noisy # test series for backtest should be noisy if available
#     num_trimmed_train_val = max(len(test_series_backtest),input_chunk_length)
#     train_val_series_trimmed = concatenate([train_series, val_series])[-num_trimmed_train_val:] # TODO: this is not a good way to do it
#     train_val_test_series_trimmed = concatenate([train_val_series_trimmed[-input_chunk_length:], test_series_backtest]) # use a lookback of val for testing
#     # train_val_test_series_trimmed = test_series_backtest

#     log.info("Backtesting the model without retraining (testing on test series)")
#     list_backtest_series, test_preds, test_targets = historical_forecasts_manual(
#         model=model,
#         test_series=train_val_test_series_trimmed,
#         input_chunk_length=input_chunk_length,
#         output_chunk_length=output_chunk_length,
#         ) # size = (len(test_series_backtest), output_chunk_length, outdim)
    
#     # # double check that the backtest target is the same as the test series and mse is the same
#     # list_backtest_values = [series.values() for series in list_backtest_series]
#     # backtest_array = np.stack(list_backtest_values, axis=0)
#     # targettest_array = test_targets.numpy()
#     # test_array = test_series.values()
#     # test_mse = ((backtest_array-targettest_array)**2).mean(1)
#     # print(f"Test MSE: {test_mse.mean()}")
    
#     ##################### OLD CODE #####################
#     # list_backtest_series = model.historical_forecasts(
#     #     train_val_test_series_trimmed,
#     #     start=test_series_backtest.start_time(),
#     #     forecast_horizon=output_chunk_length,
#     #     retrain=False,
#     #     verbose=False,
#     #     last_points_only=False,
#     #     )
#     #####################################################
    
#     # # rollinig predictions
#     # lengths_preds = min(3*output_chunk_length, len(test_series))
#     # rolling_pred = model.predict(
#     #     series=train_val_series_trimmed,
#     #     n=lengths_preds
#     #     ) # assumed val_series is bigger than input_chunk_length
#     # rolling_pred_middle = model.predict(
#     #     series=train_val_test_series_trimmed[:3*len(train_val_test_series_trimmed)//4],
#     #     n=lengths_preds
#     #     ) 
#     # rolling_pred_end = model.predict(
#     #     series=train_val_test_series_trimmed,
#     #     n=lengths_preds
#     #     ) 
    
#     # unnormalize series
#     # rolling_unscaled_pred = scaler.inverse_transform(rolling_pred) if scaler else rolling_pred
#     # rolling_unscaled_pred_middle = scaler.inverse_transform(rolling_pred_middle) if scaler else rolling_pred_middle
#     # rolling_unscaled_pred_end = scaler.inverse_transform(rolling_pred_end) if scaler else rolling_pred_end
#     train_val_unscaled_series_trimmed = scaler.inverse_transform(train_val_series_trimmed) if scaler else train_val_series_trimmed
#     test_unscaled_series = scaler.inverse_transform(test_series) if scaler else test_series
#     list_backtest_unscaled_series = [scaler.inverse_transform(backtest_series) for backtest_series in list_backtest_series] if scaler else list_backtest_series
#     # scaler.inverse_transform(list_backtest_series) if scaler else list_backtest_series 
    
#     # calculating results for Target components only, if available (for crypto dataset)
#     target_indices = np.array(['Target' in component for component in test_series_backtest.components])
#     if target_indices.any():
#         components = list(test_series.components[target_indices])
#         test_series = test_series[components] 
#         test_series_backtest = test_series_backtest[components]
#         test_unscaled_series = test_unscaled_series[components]
#         list_backtest_series = [backtest_series[components] for backtest_series in list_backtest_series]
        
#         # train_val_unscaled_series_trimmed = train_val_unscaled_series_trimmed[list(train_val_unscaled_series_trimmed.components[target_indices])]
#         # list_backtest_unscaled_series = [backtest_series[list(backtest_series.components[target_indices])] for backtest_series in list_backtest_unscaled_series]
#         # test_preds = test_preds[..., target_indices]
#         # test_targets = test_targets[..., target_indices]
    
#     # calculate metrics    
#     predictions = np.stack([series._xa.values for series in list_backtest_series], axis=0).squeeze(-1) # (len(test_series)-output_chunk_length+1, output_chunk_length, outdim)
#     log.info("Calculating metrics for backtesting")
#     results = calculate_metrics(
#         true=sliding_window(test_series, output_chunk_length),
#         pred=predictions
#         )
#     # results = calculate_metrics_darts(
#     #     [test_series]*len(list_backtest_series),
#     #     list_backtest_series,
#     #     reduction=np.array,
#     #     verbose=True,
#     #     n_jobs=-1
#     #     )    
    
#     if test_series_noisy is not None:
#         log.info("Calculating metrics for backtesting using noisy test")
#         results_noisy = calculate_metrics(
#             true=sliding_window(test_series_backtest, output_chunk_length),
#             pred=predictions
#             )
#         # results_noisy = calculate_metrics_darts(
#         #     [test_series_noisy]*len(list_backtest_series),
#         #     list_backtest_series,
#         #     reduction=np.array,
#         #     verbose=True,
#         #     n_jobs=-1
#         #     )
    
#     log.info("Calculating metrics for unnormalized backtesting")
#     results_unscaled = calculate_metrics(
#         true=sliding_window(test_unscaled_series, output_chunk_length),
#         pred=predictions
#         )
#     # results_unscaled = calculate_metrics_darts(
#     #     [test_unscaled_series]*len(list_backtest_unscaled_series),
#     #     list_backtest_unscaled_series, 
#     #     reduction=np.array,
#     #     verbose=True,
#     #     n_jobs=-1
#     #     )
    
#     # log.info("Calculating metrics for rolling predictions")
#     # results_pred = calculate_metrics_darts(
#     #     test_series,
#     #     rolling_pred[:output_chunk_length],
#     #     reduction=np.array,
#     #     verbose=False,
#     #     n_jobs=1,
#     #     )
#     # results_pred_middle = calculate_metrics_darts(
#     #     test_series,
#     #     rolling_pred_middle[:output_chunk_length],
#     #     reduction=np.array,
#     #     verbose=False,
#     #     n_jobs=1,
#     # )
    
    
#     # results = {
#     #     result_name: np.vstack(results[result_name])
#     #     for result_name in results.keys()
#     #     if not np.isnan(np.array(results[result_name])).any()
#     #     }
#     # results_noisy = {
#     #     result_name: np.vstack(results_noisy[result_name])
#     #     for result_name in results_noisy.keys()
#     #     if not np.isnan(np.array(results_noisy[result_name])).any()
#     #     }
#     # results_unscaled = {
#     #     result_name: np.vstack(results_unscaled[result_name])
#     #     for result_name in results_unscaled.keys()
#     #     if not np.isnan(np.array(results_unscaled[result_name])).any()
#     #     }
    
#     # for visualizing backtest, we use last points only
#     backtest_unscaled_series = concatenate([backtest_series[-1:] for backtest_series in list_backtest_unscaled_series]) # semi-colon is important!!!
    
#     # log best    
#     if logging:
#         log.info("logging results to wandb")
#         # log best historical, best historical unscaled, best pred
#         wandb.log({
#             f"test_best_historical_{result_name}": results[result_name].mean() 
#             for result_name in results.keys() if not np.isnan(results[result_name]).any()
#             })
        
#         wandb.log({
#             f"test_best_historical_noisy_{result_name}": results_noisy[result_name].mean()
#             for result_name in results_noisy.keys() if not np.isnan(results_noisy[result_name]).any()
#             })
        
#         wandb.log({
#             f"test_best_historical_unscaled_{result_name}": results_unscaled[result_name].mean()
#             for result_name in results_unscaled.keys() if not np.isnan(results_unscaled[result_name]).any()
#             })
        
#         # wandb.log({
#         #     f"test_best_pred_{result_name}": results_value.mean()
#         #     for result_name, results_value in results_pred.items() if not np.isnan(results_value).any()
#         #     })
        
#         # wandb.log({
#         #     f"test_best_pred_middle_{result_name}": results_value.mean()
#         #     for result_name, results_value in results_pred_middle.items() if not np.isnan(results_value).any()
#         #     })
        
#         # plot
#         for i, component in reversed(list(enumerate(components))):
#             if i<(len(components)-10): # only plot 10 components
#                 break
            
#             wandb.log({
#                 f"test_best_historical_{result_name}_{component}": results[result_name][..., i].mean()
#                 for result_name in results.keys() if not np.isnan(results[result_name]).any()
#                 })
        
#             wandb.log({
#                 f"test_best_historical_unscaled_{result_name}_{component}": results_unscaled[result_name][..., i].mean()
#                 for result_name in results_unscaled.keys() if not np.isnan(results_unscaled[result_name]).any()
#                 })
        
#             # wandb.log({
#             #     f"test_best_pred_{result_name}_{component}": results_value[..., i].mean()
#             #     for result_name, results_value in results_pred.items() if not np.isnan(results_value).any()
#             #     })
            
            
#             plt.figure(figsize=(5, 3))
#             train_val_unscaled_series_trimmed[component].plot(label="train_val_"+ component)
#             test_unscaled_series[component].plot(label="test_" + component)
#             backtest_unscaled_series[component].plot(label="backtest_" + component)
#             # rolling_unscaled_pred[component].plot(label="rolling_pred_" + component)
#             # rolling_unscaled_pred_middle[component].plot(label="rolling_pred_middle_" + component)
#             # rolling_unscaled_pred_end[component].plot(label="rolling_pred_end_" + component)
#             # plt.title(configs.model.model_name + configs.data.dataset_name + component)
#             wandb.log({"Media": plt})
            
#             # plot scaled version and a couple of predictions
#             if (i==len(components)-1) or ('Target' in component):
#                 plt.figure(figsize=(5, 3))
#                 train_val_series_trimmed[component].plot(label="scaled_train_val_"+ component, lw=0.5)
#                 test_series_backtest[component].plot(label="scaled_noisy_test_" + component, lw=0.5)
                
#                 no_rainbow_plots = 100
#                 plot_interval = output_chunk_length if len(list_backtest_series)//output_chunk_length<no_rainbow_plots else len(list_backtest_series)//no_rainbow_plots
                
#                 [
#                     series[component].plot(label="pred_" + component)
#                     for j, series in enumerate(list_backtest_series)
#                     if j%plot_interval==0
#                  ]
#                 wandb.log({"Media": plt})
                
#                 plt.figure(figsize=(5, 3))
#                 test_series[component].plot(label="scaled_noisy_test_" + component, lw=0.5)
#                 [
#                     series[component].plot(label="pred_" + component)
#                     for j, series in enumerate(list_backtest_series)
#                     if j%plot_interval==0
#                  ]
#                 wandb.log({"Media": plt})
                
#     return results, list_backtest_series
                
    
# def eval_localforecasting_model( 
#     model: LocalForecastingModel,
#     configs: DictConfig,
#     data_series: DataSeries,
#     logging: bool = True,
#     ) -> Union[Dict, DataSeries]:
#     #load model        
    
#     # input and output chunk length
#     input_chunk_length = configs.model.input_chunk_length
#     output_chunk_length = configs.model.output_chunk_length
    
#     # get series from data_series
#     train_series = data_series.train_series
#     val_series = data_series.val_series
#     test_series = data_series.test_series
#     test_series_noisy = data_series.test_series_noisy
#     scaler = data_series.scaler
    
#     # get historical forecasts
#     from darts.timeseries import concatenate 
#     components = test_series.components
#     test_series_backtest = test_series if test_series_noisy is None else test_series_noisy # test series for backtest should be noisy if available
#     train_val_series = concatenate([train_series, val_series])[len(val_series):] # remove the size of val series from the beginning for fair comparison
#     train_val_test_series = concatenate([train_val_series, test_series_backtest])
    

#     log.info("Backtesting the model with retraining (testing on test series)")
#     list_backtest_series = model.historical_forecasts(
#         train_val_test_series,
#         start=test_series_backtest.start_time(),
#         forecast_horizon=output_chunk_length,
#         retrain=True, # retrain the model at each step since local forecasting model is not parametrized
#         verbose=False,
#         last_points_only=False,
#         )
    
#     num_trimmed_train_val = max(len(test_series_backtest),input_chunk_length)
#     train_val_series_trimmed = train_val_series[-num_trimmed_train_val:]
    
#     # rollinig predictions
#     rolling_pred = list_backtest_series[0]
#     rolling_pred_middle = list_backtest_series[3*len(list_backtest_series)//4]
#     rolling_pred_end = list_backtest_series[-1]
    
#     # unnormalize series
#     rolling_unscaled_pred = scaler.inverse_transform(rolling_pred) if scaler else rolling_pred
#     rolling_unscaled_pred_middle = scaler.inverse_transform(rolling_pred_middle) if scaler else rolling_pred_middle
#     rolling_unscaled_pred_end = scaler.inverse_transform(rolling_pred_end) if scaler else rolling_pred_end
#     train_val_unscaled_series_trimmed = scaler.inverse_transform(train_val_series_trimmed) if scaler else train_val_series_trimmed
#     test_unscaled_series = scaler.inverse_transform(test_series) if scaler else test_series
#     list_backtest_unscaled_series = [scaler.inverse_transform(backtest_series) for backtest_series in list_backtest_series] if scaler else list_backtest_series
    
#     # calculate metrics    
#     log.info("Calculating metrics for backtesting")
#     results = calculate_metrics(
#         [test_series]*len(list_backtest_series),
#         list_backtest_series,
#         reduction=np.array,
#         verbose=True,
#         n_jobs=-1
#         )    
    
#     if test_series_noisy is not None:
#         log.info("Calculating metrics for backtesting using noisy test")
#         results_noisy = calculate_metrics(
#             [test_series_noisy]*len(list_backtest_series),
#             list_backtest_series,
#             reduction=np.array,
#             verbose=True,
#             n_jobs=-1
#             )
    
#     log.info("Calculating metrics for unnormalized backtesting")
#     results_unscaled = calculate_metrics(
#         [test_unscaled_series]*len(list_backtest_unscaled_series),
#         list_backtest_unscaled_series, 
#         reduction=np.array,
#         verbose=True,
#         n_jobs=-1
#         )
    
#     log.info("Calculating metrics for rolling predictions")
#     results_pred = calculate_metrics(
#         test_series,
#         rolling_pred[:output_chunk_length],
#         reduction=np.array,
#         verbose=False,
#         n_jobs=1,
#         )
#     results_pred_middle = calculate_metrics(
#         test_series,
#         rolling_pred_middle[:output_chunk_length],
#         reduction=np.array,
#         verbose=False,
#         n_jobs=1,
#     )
    
    
#     results = {
#         result_name: np.vstack(results[result_name])
#         for result_name in results.keys()
#         if not np.isnan(np.array(results[result_name])).any()
#         }
#     results_noisy = {
#         result_name: np.vstack(results_noisy[result_name])
#         for result_name in results_noisy.keys()
#         if not np.isnan(np.array(results_noisy[result_name])).any()
#         }
#     results_unscaled = {
#         result_name: np.vstack(results_unscaled[result_name])
#         for result_name in results_unscaled.keys()
#         if not np.isnan(np.array(results_unscaled[result_name])).any()
#         }
    
#     # for visualizing backtest, we use last points only
#     backtest_unscaled_series = concatenate([_series[-1:] for _series in list_backtest_unscaled_series]) # semi-colon is important!!!
    
#     # log best    
#     if logging:
#         log.info("logging results to wandb")
#         # log best historical, best historical unscaled, best pred
#         wandb.log({
#             f"test_best_historical_{result_name}": results[result_name].mean() 
#             for result_name in results.keys() if not np.isnan(results[result_name]).any()
#             })
        
#         wandb.log({
#             f"test_best_historical_noisy_{result_name}": results_noisy[result_name].mean()
#             for result_name in results_noisy.keys() if not np.isnan(results_noisy[result_name]).any()
#             })
        
#         wandb.log({
#             f"test_best_historical_unscaled_{result_name}": results_unscaled[result_name].mean()
#             for result_name in results_unscaled.keys() if not np.isnan(results_unscaled[result_name]).any()
#             })
        
#         wandb.log({
#             f"test_best_pred_{result_name}": results_value.mean()
#             for result_name, results_value in results_pred.items() if not np.isnan(results_value).any()
#             })
        
#         wandb.log({
#             f"test_best_pred_middle_{result_name}": results_value.mean()
#             for result_name, results_value in results_pred_middle.items() if not np.isnan(results_value).any()
#             })
        
#         # plot
#         for i, component in reversed(list(enumerate(components))):
#             if i<(len(components)-10): # only plot 10 components
#                 break
            
#             wandb.log({
#                 f"test_best_historical_{result_name}_{component}": results[result_name][..., i].mean()
#                 for result_name in results.keys() if not np.isnan(results[result_name]).any()
#                 })
        
#             wandb.log({
#                 f"test_best_historical_unscaled_{result_name}_{component}": results_unscaled[result_name][..., i].mean()
#                 for result_name in results_unscaled.keys() if not np.isnan(results_unscaled[result_name]).any()
#                 })
        
#             wandb.log({
#                 f"test_best_pred_{result_name}_{component}": results_value[..., i].mean()
#                 for result_name, results_value in results_pred.items() if not np.isnan(results_value).any()
#                 })
            
            
#             plt.figure(figsize=(5, 3))
#             train_val_unscaled_series_trimmed[component].plot(label="train_val_"+ component)
#             test_unscaled_series[component].plot(label="test_" + component)
#             backtest_unscaled_series[component].plot(label="backtest_" + component)
#             rolling_unscaled_pred[component].plot(label="rolling_pred_" + component)
#             rolling_unscaled_pred_middle[component].plot(label="rolling_pred_middle_" + component)
#             rolling_unscaled_pred_end[component].plot(label="rolling_pred_end_" + component)
#             # plt.title(configs.model.model_name + configs.data.dataset_name + component)
#             wandb.log({"Media": plt})
            
#             # plot scaled version and a couple of predictions
#             if i==len(components)-1:
#                 plt.figure(figsize=(5, 3))
#                 train_val_series_trimmed[component].plot(label="scaled_train_val_"+ component, lw=0.5)
#                 test_series_backtest[component].plot(label="scaled_noisy_test_" + component, lw=0.5)
#                 [
#                     series[component].plot(label="pred_" + component)
#                     for j, series in enumerate(list_backtest_series)
#                     if j%output_chunk_length==0
#                  ]
#                 wandb.log({"Media": plt})
                
#     return results, list_backtest_series
            