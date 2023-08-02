import sys
import os

# sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from typing import List, Optional, Union, Dict

import wandb
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback

from src.utils import driver as utils
from src.utils.metrics import calculate_metrics_darts, calculate_metrics
from src.data.registry.data_registry import DataSeries

from darts.timeseries import TimeSeries
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


def historical_forecast(
    model: TorchForecastingModel,
    test_series: TimeSeries,
    input_chunk_length: int,
    output_chunk_length: int,
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
    # one epoch of evaluation on test set. Note that for last forecast_horizon points in test set, we only have one prediction
    for batch in tqdm(test_dl, desc="Evaluating on test set"):
        input_series, _, _, target_series = batch
        input_series = input_series.to(device=pl_model.device, dtype=pl_model.dtype)
        # target_series = target_series.to(device=pl_model.device, dtype=pl_model.dtype)
        pl_model.y = target_series.to(device=pl_model.device, dtype=pl_model.dtype) # a hack for setting target for twoomp model
        
        pred = pl_model((input_series, _))
        preds.append(pred.detach().cpu())
        targets.append(target_series.detach().cpu())
    preds = torch.cat(preds, dim=0)
    preds = preds.flip(dims=[0]) # flip back since dataset get item is designed in reverse order
    targets = torch.cat(targets, dim=0)
    targets = targets.flip(dims=[0]) # flip back since dataset get item is designed in reverse order
    
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
    
    #load model    
    # manual loading
    # from darts.models.forecasting.torch_forecasting_model import _get_checkpoint_fname, _get_checkpoint_folder
    # model_name = configs.model.model_name
    # work_dir = os.path.join(os.getcwd(), DEFAULT_DARTS_FOLDER)
    # file_name = _get_checkpoint_fname(work_dir, model_name, best=True)
    # checkpoint_dir = _get_checkpoint_folder(work_dir, model_name)
    # file_path = os.path.join(checkpoint_dir, file_name)
    # pl_model = model.model.load_from_checkpoint(file_path)
    # model.model = model.model.load_from_checkpoint('/home/mahdigilany/offline_codes/codes/ts_snr/logs/experiments/runs/deeptime_test_model_load/2023-07-14_15-24-04/darts_logs/deeptime/checkpoints/best-epoch=20-val_loss=0.08.ckpt')
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
    from darts.timeseries import concatenate 
    components = test_series.components
    test_series_backtest = test_series if test_series_noisy is None else test_series_noisy # test series for backtest should be noisy if available
    num_trimmed_train_val = max(len(test_series_backtest),input_chunk_length)
    train_val_series_trimmed = concatenate([train_series, val_series])[-num_trimmed_train_val:] # TODO: this is not a good way to do it
    train_val_test_series_trimmed = concatenate([train_val_series_trimmed[-input_chunk_length:], test_series_backtest]) # use a lookback of val for testing
    # train_val_test_series_trimmed = test_series_backtest

    log.info("Backtesting the model without retraining (testing on test series)")
    list_backtest_series, test_preds, test_targets = historical_forecast(
        model=model,
        test_series=train_val_test_series_trimmed,
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        ) # size = (len(test_series_backtest), output_chunk_length, outdim)
    
    # # double check that the backtest target is the same as the test series and mse is the same
    # list_backtest_values = [series.values() for series in list_backtest_series]
    # backtest_array = np.stack(list_backtest_values, axis=0)
    # targettest_array = test_targets.numpy()
    # test_array = test_series.values()
    # test_mse = ((backtest_array-targettest_array)**2).mean(1)
    # print(f"Test MSE: {test_mse.mean()}")
    
    ##################### OLD CODE #####################
    # list_backtest_series = model.historical_forecasts(
    #     train_val_test_series_trimmed,
    #     start=test_series_backtest.start_time(),
    #     forecast_horizon=output_chunk_length,
    #     retrain=False,
    #     verbose=False,
    #     last_points_only=False,
    #     )
    #####################################################
    
    # # rollinig predictions
    # lengths_preds = min(3*output_chunk_length, len(test_series))
    # rolling_pred = model.predict(
    #     series=train_val_series_trimmed,
    #     n=lengths_preds
    #     ) # assumed val_series is bigger than input_chunk_length
    # rolling_pred_middle = model.predict(
    #     series=train_val_test_series_trimmed[:3*len(train_val_test_series_trimmed)//4],
    #     n=lengths_preds
    #     ) 
    # rolling_pred_end = model.predict(
    #     series=train_val_test_series_trimmed,
    #     n=lengths_preds
    #     ) 
    
    # unnormalize series
    # rolling_unscaled_pred = scaler.inverse_transform(rolling_pred) if scaler else rolling_pred
    # rolling_unscaled_pred_middle = scaler.inverse_transform(rolling_pred_middle) if scaler else rolling_pred_middle
    # rolling_unscaled_pred_end = scaler.inverse_transform(rolling_pred_end) if scaler else rolling_pred_end
    train_val_unscaled_series_trimmed = scaler.inverse_transform(train_val_series_trimmed) if scaler else train_val_series_trimmed
    test_unscaled_series = scaler.inverse_transform(test_series) if scaler else test_series
    list_backtest_unscaled_series = [scaler.inverse_transform(backtest_series) for backtest_series in list_backtest_series] if scaler else list_backtest_series
    # scaler.inverse_transform(list_backtest_series) if scaler else list_backtest_series 
    
    # calculating results for Target components only, if available (for crypto dataset)
    target_indices = np.array(['Target' in component for component in test_series_backtest.components])
    if target_indices.any():
        components = list(test_series.components[target_indices])
        test_series = test_series[components] 
        test_series_backtest = test_series_backtest[components]
        test_unscaled_series = test_unscaled_series[components]
        list_backtest_series = [backtest_series[components] for backtest_series in list_backtest_series]
        
        # train_val_unscaled_series_trimmed = train_val_unscaled_series_trimmed[list(train_val_unscaled_series_trimmed.components[target_indices])]
        # list_backtest_unscaled_series = [backtest_series[list(backtest_series.components[target_indices])] for backtest_series in list_backtest_unscaled_series]
        # test_preds = test_preds[..., target_indices]
        # test_targets = test_targets[..., target_indices]
    
    # calculate metrics    
    predictions = np.stack([series._xa.values for series in list_backtest_series], axis=0).squeeze(-1) # (len(test_series)-output_chunk_length+1, output_chunk_length, outdim)
    log.info("Calculating metrics for backtesting")
    results = calculate_metrics(
        true=sliding_window(test_series, output_chunk_length),
        pred=predictions
        )
    # results = calculate_metrics_darts(
    #     [test_series]*len(list_backtest_series),
    #     list_backtest_series,
    #     reduction=np.array,
    #     verbose=True,
    #     n_jobs=-1
    #     )    
    
    if test_series_noisy is not None:
        log.info("Calculating metrics for backtesting using noisy test")
        results_noisy = calculate_metrics(
            true=sliding_window(test_series_backtest, output_chunk_length),
            pred=predictions
            )
        # results_noisy = calculate_metrics_darts(
        #     [test_series_noisy]*len(list_backtest_series),
        #     list_backtest_series,
        #     reduction=np.array,
        #     verbose=True,
        #     n_jobs=-1
        #     )
    
    log.info("Calculating metrics for unnormalized backtesting")
    results_unscaled = calculate_metrics(
        true=sliding_window(test_unscaled_series, output_chunk_length),
        pred=predictions
        )
    # results_unscaled = calculate_metrics_darts(
    #     [test_unscaled_series]*len(list_backtest_unscaled_series),
    #     list_backtest_unscaled_series, 
    #     reduction=np.array,
    #     verbose=True,
    #     n_jobs=-1
    #     )
    
    # log.info("Calculating metrics for rolling predictions")
    # results_pred = calculate_metrics_darts(
    #     test_series,
    #     rolling_pred[:output_chunk_length],
    #     reduction=np.array,
    #     verbose=False,
    #     n_jobs=1,
    #     )
    # results_pred_middle = calculate_metrics_darts(
    #     test_series,
    #     rolling_pred_middle[:output_chunk_length],
    #     reduction=np.array,
    #     verbose=False,
    #     n_jobs=1,
    # )
    
    
    # results = {
    #     result_name: np.vstack(results[result_name])
    #     for result_name in results.keys()
    #     if not np.isnan(np.array(results[result_name])).any()
    #     }
    # results_noisy = {
    #     result_name: np.vstack(results_noisy[result_name])
    #     for result_name in results_noisy.keys()
    #     if not np.isnan(np.array(results_noisy[result_name])).any()
    #     }
    # results_unscaled = {
    #     result_name: np.vstack(results_unscaled[result_name])
    #     for result_name in results_unscaled.keys()
    #     if not np.isnan(np.array(results_unscaled[result_name])).any()
    #     }
    
    # for visualizing backtest, we use last points only
    backtest_unscaled_series = concatenate([backtest_series[-1:] for backtest_series in list_backtest_unscaled_series]) # semi-colon is important!!!
    
    # log best    
    if logging:
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
        
        # wandb.log({
        #     f"test_best_pred_{result_name}": results_value.mean()
        #     for result_name, results_value in results_pred.items() if not np.isnan(results_value).any()
        #     })
        
        # wandb.log({
        #     f"test_best_pred_middle_{result_name}": results_value.mean()
        #     for result_name, results_value in results_pred_middle.items() if not np.isnan(results_value).any()
        #     })
        
        # plot
        for i, component in reversed(list(enumerate(components))):
            if i<(len(components)-10): # only plot 10 components
                break
            
            wandb.log({
                f"test_best_historical_{result_name}_{component}": results[result_name][..., i].mean()
                for result_name in results.keys() if not np.isnan(results[result_name]).any()
                })
        
            wandb.log({
                f"test_best_historical_unscaled_{result_name}_{component}": results_unscaled[result_name][..., i].mean()
                for result_name in results_unscaled.keys() if not np.isnan(results_unscaled[result_name]).any()
                })
        
            # wandb.log({
            #     f"test_best_pred_{result_name}_{component}": results_value[..., i].mean()
            #     for result_name, results_value in results_pred.items() if not np.isnan(results_value).any()
            #     })
            
            
            plt.figure(figsize=(5, 3))
            train_val_unscaled_series_trimmed[component].plot(label="train_val_"+ component)
            test_unscaled_series[component].plot(label="test_" + component)
            backtest_unscaled_series[component].plot(label="backtest_" + component)
            # rolling_unscaled_pred[component].plot(label="rolling_pred_" + component)
            # rolling_unscaled_pred_middle[component].plot(label="rolling_pred_middle_" + component)
            # rolling_unscaled_pred_end[component].plot(label="rolling_pred_end_" + component)
            # plt.title(configs.model.model_name + configs.data.dataset_name + component)
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
                
    return results, list_backtest_series
                
    
def eval_localforecasting_model( 
    model: LocalForecastingModel,
    configs: DictConfig,
    data_series: DataSeries,
    logging: bool = True,
    ) -> Union[Dict, DataSeries]:
    #load model        
    
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
    from darts.timeseries import concatenate 
    components = test_series.components
    test_series_backtest = test_series if test_series_noisy is None else test_series_noisy # test series for backtest should be noisy if available
    train_val_series = concatenate([train_series, val_series])[len(val_series):] # remove the size of val series from the beginning for fair comparison
    train_val_test_series = concatenate([train_val_series, test_series_backtest])
    

    log.info("Backtesting the model with retraining (testing on test series)")
    list_backtest_series = model.historical_forecasts(
        train_val_test_series,
        start=test_series_backtest.start_time(),
        forecast_horizon=output_chunk_length,
        retrain=True, # retrain the model at each step since local forecasting model is not parametrized
        verbose=False,
        last_points_only=False,
        )
    
    num_trimmed_train_val = max(len(test_series_backtest),input_chunk_length)
    train_val_series_trimmed = train_val_series[-num_trimmed_train_val:]
    
    # rollinig predictions
    rolling_pred = list_backtest_series[0]
    rolling_pred_middle = list_backtest_series[3*len(list_backtest_series)//4]
    rolling_pred_end = list_backtest_series[-1]
    
    # unnormalize series
    rolling_unscaled_pred = scaler.inverse_transform(rolling_pred) if scaler else rolling_pred
    rolling_unscaled_pred_middle = scaler.inverse_transform(rolling_pred_middle) if scaler else rolling_pred_middle
    rolling_unscaled_pred_end = scaler.inverse_transform(rolling_pred_end) if scaler else rolling_pred_end
    train_val_unscaled_series_trimmed = scaler.inverse_transform(train_val_series_trimmed) if scaler else train_val_series_trimmed
    test_unscaled_series = scaler.inverse_transform(test_series) if scaler else test_series
    list_backtest_unscaled_series = [scaler.inverse_transform(backtest_series) for backtest_series in list_backtest_series] if scaler else list_backtest_series
    
    # calculate metrics    
    log.info("Calculating metrics for backtesting")
    results = calculate_metrics(
        [test_series]*len(list_backtest_series),
        list_backtest_series,
        reduction=np.array,
        verbose=True,
        n_jobs=-1
        )    
    
    if test_series_noisy is not None:
        log.info("Calculating metrics for backtesting using noisy test")
        results_noisy = calculate_metrics(
            [test_series_noisy]*len(list_backtest_series),
            list_backtest_series,
            reduction=np.array,
            verbose=True,
            n_jobs=-1
            )
    
    log.info("Calculating metrics for unnormalized backtesting")
    results_unscaled = calculate_metrics(
        [test_unscaled_series]*len(list_backtest_unscaled_series),
        list_backtest_unscaled_series, 
        reduction=np.array,
        verbose=True,
        n_jobs=-1
        )
    
    log.info("Calculating metrics for rolling predictions")
    results_pred = calculate_metrics(
        test_series,
        rolling_pred[:output_chunk_length],
        reduction=np.array,
        verbose=False,
        n_jobs=1,
        )
    results_pred_middle = calculate_metrics(
        test_series,
        rolling_pred_middle[:output_chunk_length],
        reduction=np.array,
        verbose=False,
        n_jobs=1,
    )
    
    
    results = {
        result_name: np.vstack(results[result_name])
        for result_name in results.keys()
        if not np.isnan(np.array(results[result_name])).any()
        }
    results_noisy = {
        result_name: np.vstack(results_noisy[result_name])
        for result_name in results_noisy.keys()
        if not np.isnan(np.array(results_noisy[result_name])).any()
        }
    results_unscaled = {
        result_name: np.vstack(results_unscaled[result_name])
        for result_name in results_unscaled.keys()
        if not np.isnan(np.array(results_unscaled[result_name])).any()
        }
    
    # for visualizing backtest, we use last points only
    backtest_unscaled_series = concatenate([_series[-1:] for _series in list_backtest_unscaled_series]) # semi-colon is important!!!
    
    # log best    
    if logging:
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
        
        wandb.log({
            f"test_best_pred_{result_name}": results_value.mean()
            for result_name, results_value in results_pred.items() if not np.isnan(results_value).any()
            })
        
        wandb.log({
            f"test_best_pred_middle_{result_name}": results_value.mean()
            for result_name, results_value in results_pred_middle.items() if not np.isnan(results_value).any()
            })
        
        # plot
        for i, component in reversed(list(enumerate(components))):
            if i<(len(components)-10): # only plot 10 components
                break
            
            wandb.log({
                f"test_best_historical_{result_name}_{component}": results[result_name][..., i].mean()
                for result_name in results.keys() if not np.isnan(results[result_name]).any()
                })
        
            wandb.log({
                f"test_best_historical_unscaled_{result_name}_{component}": results_unscaled[result_name][..., i].mean()
                for result_name in results_unscaled.keys() if not np.isnan(results_unscaled[result_name]).any()
                })
        
            wandb.log({
                f"test_best_pred_{result_name}_{component}": results_value[..., i].mean()
                for result_name, results_value in results_pred.items() if not np.isnan(results_value).any()
                })
            
            
            plt.figure(figsize=(5, 3))
            train_val_unscaled_series_trimmed[component].plot(label="train_val_"+ component)
            test_unscaled_series[component].plot(label="test_" + component)
            backtest_unscaled_series[component].plot(label="backtest_" + component)
            rolling_unscaled_pred[component].plot(label="rolling_pred_" + component)
            rolling_unscaled_pred_middle[component].plot(label="rolling_pred_middle_" + component)
            rolling_unscaled_pred_end[component].plot(label="rolling_pred_end_" + component)
            # plt.title(configs.model.model_name + configs.data.dataset_name + component)
            wandb.log({"Media": plt})
            
            # plot scaled version and a couple of predictions
            if i==len(components)-1:
                plt.figure(figsize=(5, 3))
                train_val_series_trimmed[component].plot(label="scaled_train_val_"+ component, lw=0.5)
                test_series_backtest[component].plot(label="scaled_noisy_test_" + component, lw=0.5)
                [
                    series[component].plot(label="pred_" + component)
                    for j, series in enumerate(list_backtest_series)
                    if j%output_chunk_length==0
                 ]
                wandb.log({"Media": plt})
                
    return results, list_backtest_series
            
    
def NOTUSED_simple_run(configs: DictConfig):
    """function supposed to be used for simple runs with torch and without lightning. 
    Not complete yet.

    Args:
        configs (DictConfig): _description_
    """
    # fix seed
    if (s := configs.get("seed")) is not None:
        log.info(f"Global seed set to {s}")
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
    
    
    # initialize wandb
    wandb_config = OmegaConf.to_container(
        configs, resolve=True, throw_on_missing=True
    )
    wandb.init(
        project=configs.logger.wandb.project,
        entity=configs.logger.wandb.entity,
        config=wandb_config, # saves all configs
        name=configs.name,
        group=configs.logger.wandb.group, # group name
        job_type=None, # type of run for grouping like train or eval
        dir=None,
        save_code=True,
        resume=None, # overwriting when id is same
        )
        
    # instantiate datamodule
    
    
    # instantiate model 


    # train_model()
    
    
    # eval_model()
    
    
    # finish wandb
    wandb.finish()
    return


def darts_globalforecasting_driver_run(configs: DictConfig):
    """Driver function for darts models with pytorch lightning.
    fixes seeds, instantiates datamodule, model, logger, trainer, callbacks, etc. and trains the model using model.fit().

    Args:
        configs (DictConfig): all the configs coming from hydra.
    """
    
    # fix seed
    if (s := configs.get("seed")) is not None:
        log.info(f"Global seed set to {s}")
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
    
    
    # pl_trainer_kwargs
    pl_trainer_kwargs = {}
    pl_trainer_kwargs.update(configs.trainer)
    
    
    # init lightning callbacks
    pl_trainer_kwargs["callbacks"]: List[Callback] = []
    if "callbacks" in configs:
        if configs.callbacks is not None:
            # pl_trainer_kwargs["enable_checkpointing"] = True
            for _, cb_conf in configs.callbacks.items():
                if "_target_" in cb_conf:
                    log.info(f"Instantiating callback <{cb_conf._target_}>")
                    pl_trainer_kwargs["callbacks"].append(instantiate(cb_conf))

    
    # init lightning logger    
    if "logger" in configs:
        if configs.logger is not None:
            from pytorch_lightning.loggers import WandbLogger
            lg_conf = dict(configs.logger.wandb)
            log.info(f"Instantiating logger <{configs.logger.wandb._target_}>")

            _config = OmegaConf.to_container(
                configs, resolve=True, throw_on_missing=True
                )
            lg_conf.pop("_target_")
            logger = WandbLogger(**lg_conf, config=_config) # saves all configs
            # logger = instantiate(configs.logger, config=_config) # saves all configs
            pl_trainer_kwargs["logger"] = logger
    
    
    # loading data
    log.info(f"Instantiating data <{configs.data._target_}>")
    data_series: DataSeries = instantiate(configs.data)
    
    
    # defining metrics
    log.info(f"Preparing metrics")
    from torchmetrics import MetricCollection, MeanAbsolutePercentageError, MeanSquaredError, MeanAbsoluteError
    mape = MeanAbsolutePercentageError()
    mse = MeanSquaredError()
    mae = MeanAbsoluteError()
    metrics = MetricCollection([mape, mse, mae])
    
    
    # instantiate darts model 
    log.info(f"Instantiating model <{configs.model._target_}>")
    model: TorchForecastingModel = instantiate(
        configs.model,
        pl_trainer_kwargs=pl_trainer_kwargs,
        torch_metrics=metrics,
        save_checkpoints=configs.save_checkpoints,
        work_dir=os.path.join(os.getcwd(), DEFAULT_DARTS_FOLDER), # this avoids saving darts_logs anywhere else
        )


    # train model
    log.info("Training model")
    model.fit(
        data_series.train_series,
        epochs=configs.epochs,
        verbose=configs.verbose,
        num_loader_workers=0,
        val_series=data_series.val_series,
        ) # trainer
        
    # eval model
    log.info("Evaluating model")
    results, data_series = eval_model(
        model=model, 
        configs=configs, 
        data_series=data_series,
        logging=configs.logger!=None,
        )
    
    if configs.logger is not None:
        # finish wandb
        wandb.finish()
        
    return model, data_series, results


# TODO change loading data
def inference_darts_lightning_driver_run(configs: DictConfig):
    """Driver function for darts models with pytorch lightning.
    fixes seeds, instantiates datamodule, model, logger, trainer, callbacks, etc. and trains the model using model.fit().

    Args:
        configs (DictConfig): all the configs coming from hydra.
    """
    
    # fix seed
    if (s := configs.get("seed")) is not None:
        log.info(f"Global seed set to {s}")
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
    
    
    # pl_trainer_kwargs
    pl_trainer_kwargs = {}
    pl_trainer_kwargs.update(configs.trainer)
    
    
    # init lightning callbacks
    pl_trainer_kwargs["callbacks"]: List[Callback] = []
    if "callbacks" in configs:
        if configs.callbacks is not None:
            # pl_trainer_kwargs["enable_checkpointing"] = True
            for _, cb_conf in configs.callbacks.items():
                if "_target_" in cb_conf:
                    log.info(f"Instantiating callback <{cb_conf._target_}>")
                    pl_trainer_kwargs["callbacks"].append(instantiate(cb_conf))

    
    
    # init lightning logger    
    if "logger" in configs:
        if configs.logger is not None:
            from pytorch_lightning.loggers import WandbLogger
            lg_conf = dict(configs.logger.wandb)
            log.info(f"Instantiating logger <{configs.logger.wandb._target_}>")

            _config = OmegaConf.to_container(
                configs, resolve=True, throw_on_missing=True
                )
            lg_conf.pop("_target_")
            logger = WandbLogger(**lg_conf, config=_config) # saves all configs
            # logger = instantiate(configs.logger, config=_config) # saves all configs
            pl_trainer_kwargs["logger"] = logger
    
    
    # loading data
    log.info(f"Instantiating data <{configs.data._target_}>")
    data_series: DataSeries = instantiate(configs.data)
    
    
    # defining metrics
    log.info(f"Preparing metrics")
    from torchmetrics import MetricCollection, MeanAbsolutePercentageError, MeanSquaredError, MeanAbsoluteError
    mape = MeanAbsolutePercentageError()
    mse = MeanSquaredError()
    mae = MeanAbsoluteError()
    metrics = MetricCollection([mape, mse, mae])
    
    
    # instantiate darts model 
    log.info(f"Instantiating model <{configs.model._target_}>")
    model: TorchForecastingModel = instantiate(
        configs.model,
        pl_trainer_kwargs=pl_trainer_kwargs,
        torch_metrics=metrics,
        save_checkpoints=configs.save_checkpoints,
        work_dir=os.path.join(os.getcwd(), DEFAULT_DARTS_FOLDER), # this avoids saving darts_logs anywhere else
        force_reset=True,
        )

    return model, data_series, metrics


def darts_localforecasting_driver_run(configs: DictConfig):
    """Driver function for darts models with pytorch lightning.
    fixes seeds, instantiates datamodule, model, logger, trainer, callbacks, etc. and trains the model using model.fit().

    Args:
        configs (DictConfig): all the configs coming from hydra.
    """
    
    # fix seed
    if (s := configs.get("seed")) is not None:
        log.info(f"Global seed set to {s}")
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
    
    
    # loading data
    log.info(f"Instantiating data <{configs.data._target_}>")
    data_series: DataSeries = instantiate(configs.data)
    
    
    # instantiate darts model 
    log.info(f"Instantiating model <{configs.model._target_}>")
    model: LocalForecastingModel = instantiate(
        configs.model,
        )


    # train model
    log.info("Training model")
    model.fit(
        data_series.train_series,
        )
        
    # eval model
    log.info("Evaluating model")
    results, data_series = eval_localforecasting_model(
        model=model, 
        configs=configs, 
        data_series=data_series,
        logging=configs.logger!=None,
        )
    
    if configs.logger is not None:
        # finish wandb
        wandb.finish()
        
    return model, data_series, results

