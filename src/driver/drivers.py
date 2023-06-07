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
from src.utils.metrics import calculate_metrics
from src.data.registry.data_registry import DataSeries

from darts.timeseries import TimeSeries
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel, DEFAULT_DARTS_FOLDER


log = utils.get_logger(__name__)


def train_model():
    ...


def eval_model(
    model: TorchForecastingModel,
    configs: DictConfig,
    data_series: DataSeries,
    log: bool = True,
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
    model.load_from_checkpoint(model_name=configs.model.model_name) # TODO: needs work if not using wandb or not after training
    
    # get series from data_series
    train_series = data_series.train_series
    val_series = data_series.val_series
    test_series = data_series.test_series
    test_series_noisy = data_series.test_series_noisy
    scaler = data_series.scaler
    
    # get historical forecasts
    from darts.timeseries import concatenate
    # test series for backtest should be noisy if available 
    test_series_backtest = test_series if test_series_noisy is None else test_series_noisy 
    train_val_series_trimmed = concatenate([train_series, val_series])[-max(len(test_series_backtest),configs.model.input_chunk_length):] # TODO: this is not a good way to do it
    train_val_test_series_trimmed = concatenate([train_val_series_trimmed, test_series_backtest])
    stride=configs.model.output_chunk_length # it can be 1 for more accurate results but it takes longer
    backtest_series = model.historical_forecasts(
        train_val_test_series_trimmed,
        start=test_series_backtest.start_time(),
        forecast_horizon=configs.model.output_chunk_length,
        retrain=False,
        verbose=configs.verbose,
        stride=stride,
        last_points_only=False if stride > 1 else True,
        )
    
    if isinstance(backtest_series, List):
        assert stride == configs.model.output_chunk_length, "stride should be equal to output_chunk_length, otherwise not implemented yet"
        backtest_series = concatenate(backtest_series)
    
    rolling_pred = model.predict(series=train_val_series_trimmed, n=min(10*configs.model.output_chunk_length, len(test_series))) # assumed val_series is bigger than input_chunk_length
    rolling_unscaled_pred = scaler.inverse_transform(rolling_pred) if scaler else rolling_pred
    
    # scale back series
    # train_unscaled_series = scaler.inverse_transform(train_series) if scaler else train_series # TODO: check if scaler is None for use_scaler=False
    # val_unscaled_series = scaler.inverse_transform(val_series) if scaler else val_series
    train_val_series_trimmed = scaler.inverse_transform(train_val_series_trimmed) if scaler else train_val_series_trimmed
    test_unscaled_series = scaler.inverse_transform(test_series) if scaler else test_series
    backtest_unscaled_series = scaler.inverse_transform(backtest_series) if scaler else backtest_series 
    
    # calculate metrics
    results = calculate_metrics(test_series, backtest_series, reduction=None)
    results_unscaled = calculate_metrics(test_unscaled_series, backtest_unscaled_series, reduction=None)
    results_pred = calculate_metrics(test_series, rolling_pred[:configs.model.output_chunk_length], reduction=None)
    
    print("Results of backtesting:", results)
    print("Results of backtesting unscaled:", results_unscaled)

    # log best    
    if log:
        # log best historical, best historical unscaled, best pred
        wandb.log({
            f"test_best_historical_{result_name}": np.mean(results_value) 
            for result_name, results_value in results.items()
            })
        
        wandb.log({
            f"test_best_historical_unscaled_{result_name}": np.mean(results_value)
            for result_name, results_value in results_unscaled.items()
            })
        
        wandb.log({
            f"test_best_pred_{result_name}": np.mean(results_value)
            for result_name, results_value in results_pred.items()
            })
    
        # plot
        for i, component in enumerate(test_unscaled_series.components):
            wandb.log({
                f"test_best_historical_{result_name}_{component}": np.array(results_value)[i]
                for result_name, results_value in results.items()
                })
        
            wandb.log({
                f"test_best_historical_unscaled_{result_name}_{component}": np.array(results_value)[i]
                for result_name, results_value in results_unscaled.items()
                })
        
            wandb.log({
                f"test_best_pred_{result_name}_{component}": np.array(results_value)[i]
                for result_name, results_value in results_pred.items()
                })
            
            plt.figure(figsize=(5, 3))
            # train_unscaled_series.plot(label="train")
            # val_unscaled_series.plot(label="val")
            train_val_series_trimmed[component].plot(label="train_val_"+ component)
            test_unscaled_series[component].plot(label="test_" + component)
            backtest_unscaled_series[component].plot(label="backtest_" + component)
            rolling_unscaled_pred[component].plot(label="rolling_pred_" + component)
            # plt.title(configs.model.model_name + configs.data.dataset_name + component)
            wandb.log({"Media": plt})
    return results, backtest_series
                
    
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


def darts_lightning_driver_run(configs: DictConfig):
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
        log=configs.logger!=None,
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
    train_series, val_series, *scaler = instantiate(configs.data)
    data = (train_series, val_series, *scaler)
    
    # defining metrics
    log.info(f"Preparing metrics")
    from torchmetrics import MetricCollection, MeanAbsolutePercentageError
    mape = MeanAbsolutePercentageError()
    metrics = MetricCollection([mape])
    
    
    # instantiate darts model 
    log.info(f"Instantiating model <{configs.model._target_}>")
    from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
    model: TorchForecastingModel = instantiate(
        configs.model,
        pl_trainer_kwargs=pl_trainer_kwargs,
        torch_metrics=metrics,
        save_checkpoints=configs.save_checkpoints,
        work_dir=os.path.join(os.getcwd(), DEFAULT_DARTS_FOLDER), # this avoids saving darts_logs anywhere else
        force_reset=True,
        )

    return model, data, metrics

