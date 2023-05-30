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

from darts.timeseries import TimeSeries
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel, DEFAULT_DARTS_FOLDER


log = utils.get_logger(__name__)


def train_model():
    ...


def eval_model(
    model: TorchForecastingModel,
    configs: DictConfig,
    train_series: TimeSeries,
    val_series: TimeSeries,
    scaler: Optional[List] = None,
    log: bool = True,
    ) -> Union[Dict, TimeSeries]:
    """Metrics are found based on the validation series on historical forcasts of 
    the model (using validation to predict future values, read darts documentation).
    The results are reported based on the scaled series.

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
    model.load_from_checkpoint(model_name=configs.model.model_name) # TODO: needs work
    
    # get historical forecasts
    from darts.timeseries import concatenate
    all_series = concatenate([train_series, val_series])
    backtest_series = model.historical_forecasts(
        all_series,
        start=val_series.start_time(),
        forecast_horizon=configs.model.output_chunk_length,
        retrain=False,
        verbose=True,
        )
    
    rolling_pred = model.predict(series=train_series, n=len(val_series))
    rolling_unscaled_pred = scaler[0].inverse_transform(rolling_pred) if scaler else rolling_pred
    
    # scale back series
    train_unscaled_series = scaler[0].inverse_transform(train_series) if scaler else train_series # TODO: check if scaler is None for use_scaler=False
    val_unscaled_series = scaler[0].inverse_transform(val_series) if scaler else val_series
    backtest_unscaled_series = scaler[0].inverse_transform(backtest_series) if scaler else backtest_series 
    
    # calculate metrics
    results = calculate_metrics(val_series, backtest_series)
    print("Results of backtesting:", results)

    # log best    
    if log:
        # log best 
        wandb.log({
            f"val_best_{result_name}": np.array(results_value) 
            for result_name, results_value in results.items()
            })
    
        # plot
        plt.figure(figsize=(5, 3))
        train_unscaled_series.plot(label="train")
        val_unscaled_series.plot(label="val")
        backtest_unscaled_series.plot(label="backtest")
        rolling_unscaled_pred.plot(label="rolling_pred")
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
            pl_trainer_kwargs["enable_checkpointing"] = True
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
        train_series,
        epochs=configs.epochs,
        verbose=True,
        num_loader_workers=0,
        val_series=val_series,
        ) # trainer
        
    # eval model
    log.info("Evaluating model")
    results, backtest_series = eval_model(
        model=model, 
        configs=configs, 
        train_series=train_series, 
        val_series=val_series, 
        scaler=scaler,
        log=configs.logger!=None,
        )
    
    if configs.logger is not None:
        # finish wandb
        wandb.finish()
        
    data = (train_series, val_series, backtest_series, *scaler)
    return model, data, results


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
            pl_trainer_kwargs["enable_checkpointing"] = True
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

