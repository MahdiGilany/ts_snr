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
from src.utils.evaluation import eval_model #eval_globalforecasting_model, eval_localforecasting_model
from src.data.registry.data_registry import DataSeries

from darts.timeseries import TimeSeries
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel, DEFAULT_DARTS_FOLDER
from darts.models.forecasting.forecasting_model import ForecastingModel, LocalForecastingModel
from darts.utils.data.sequential_dataset import PastCovariatesSequentialDataset

from tqdm import tqdm
from copy import copy, deepcopy



log = utils.get_logger(__name__)


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
    results, data_series = eval_model(
        model=model, 
        configs=configs, 
        data_series=data_series,
        logging=configs.logger!=None,
        forecasting_type="local",
        )
    
    if configs.logger is not None:
        # finish wandb
        wandb.finish()
        
    return model, data_series, results


# for two stage deeptime
def darts_twostage_globalforecasting_driver_run(configs: DictConfig):
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
    
    
    if not configs.resume_run:
        # instantiate darts model 
        log.info(f"Instantiating model <{configs.model._target_}>")
        model: TorchForecastingModel = instantiate(
            configs.model,
            pl_trainer_kwargs=pl_trainer_kwargs,
            torch_metrics=metrics,
            save_checkpoints=configs.save_checkpoints,
            work_dir=os.path.join(os.getcwd(), DEFAULT_DARTS_FOLDER), # this avoids saving darts_logs anywhere else
            # force_reset=True,
            )
    
    
        # train model
        log.info("Training stage 1 model")
        model.fit(
            data_series.train_series,
            epochs=configs.epochs,
            verbose=configs.verbose,
            num_loader_workers=0,
            val_series=data_series.val_series,
            ) # trainer
    
    # loading best model
    model = TorchForecastingModel.load_from_checkpoint(model_name=configs.model.model_name, best=True) # TODO: needs work if not using wandb or not after training

    # getting lookback and horizon codes (w_L and w_H)
    log.info("Get lookback and horizon codes")
    from src.utils.training import get_lookback_horizon_codes
    train_WLs, train_WHs = get_lookback_horizon_codes(
        model,
        data_series.train_series,
        configs.model.input_chunk_length,
        configs.model.output_chunk_length,
        batch_size=configs.batch_size,
        ) # (len(train), lookback, out_dim)
    
    val_WLs, val_WHs = get_lookback_horizon_codes(
        model,
        data_series.val_series,
        configs.model.input_chunk_length,
        configs.model.output_chunk_length,
        batch_size=configs.batch_size, 
        ) # (len(val), lookback, out_dim)
    
    # instantiate sequence model
    log.info("Instantiating sequence model")
    sequence_model = instantiate(configs.model.sequence_config.model) 
    
    # manually train sequence model
    if configs.train_stage2:
        log.info("Training stage 2 sequence model")
        from src.utils.training import manual_train_seq_model
        sequence_model = manual_train_seq_model(
            seq_config=configs.model.sequence_config,
            output_chunk_length=configs.model.output_chunk_length,
            seq_model=sequence_model,
            train_data=(train_WLs, train_WHs),
            val_data=(val_WLs, val_WHs),
            wandb_log=configs.logger!=None,
        )    
    else:
        log.info("Loading stage 2 sequence model")
        sequence_model = torch.load(f"./{configs.model.sequence_config.model.model_name}.pt")
    
    seq_len = configs.model.sequence_config.model.seq_len
    train_val_lookback_codes = np.concatenate([train_WHs, val_WHs], axis=0)[-seq_len-configs.model.output_chunk_length:]
    
    # free memory    
    del train_WLs, train_WHs, val_WLs, val_WHs
    
    # eval model
    log.info("Evaluating model")
    from src.utils.evaluation import eval_twostage_model
    results, data_series = eval_twostage_model(
        model=model, 
        seq_model=sequence_model,
        train_val_lookback_codes=train_val_lookback_codes,
        configs=configs, 
        data_series=data_series,
        logging=configs.logger!=None,
        )
    
    if configs.logger is not None:
        # finish wandb
        wandb.finish()
        
    return model, data_series, results


# for meta deeptime
def darts_twostage_metadeeptime_globalforecasting_driver_run(configs: DictConfig):
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
    
    model = None
    if not configs.resume_run:
        log.info(f"Initializing INR by DeepTime training")
        
        # instantiate darts model 
        log.info(f"Instantiating model <{configs.model._target_}>")
        model_config = configs.model.copy()
        model_config['model_name'] = 'deeptime'
        del model_config.meta_config
        model: TorchForecastingModel = instantiate(
            model_config,
            pl_trainer_kwargs=pl_trainer_kwargs,
            torch_metrics=metrics,
            save_checkpoints=configs.save_checkpoints,
            work_dir=os.path.join(os.getcwd(), DEFAULT_DARTS_FOLDER), # this avoids saving darts_logs anywhere else
            # force_reset=True,
            )
    
    
        # train model
        log.info("Training stage 1 model")
        model.fit(
            data_series.train_series,
            epochs=configs.epochs,
            verbose=configs.verbose,
            num_loader_workers=0,
            val_series=data_series.val_series,
            ) # trainer
    
        # loading best model
        model = TorchForecastingModel.load_from_checkpoint(model_name=model_config.model_name, best=True) # TODO: needs work if not using wandb or not after training


    from src.utils.training import MetaDeepTime
    meta_model = MetaDeepTime(
        inr=model.model.inr if model is not None else None,
        _lambda=model.model.adaptive_weights._lambda if model is not None else None,
        horizon=configs.model.output_chunk_length,
        
    )
    
    # manually train MAML using DeeoTime model
    if configs.train_stage2:
        log.info("Training stage 2 sequence model")
        from src.utils.training import manual_train_meta_deeptime, manual_train_meta_deeptime_closedform
        meta_model = manual_train_meta_deeptime_closedform( #manual_train_meta_deeptime( # turn on for MAML instead of closed form
            meta_config=configs.model.meta_config,
            input_chunk_length=configs.model.input_chunk_length,
            output_chunk_length=configs.model.output_chunk_length,
            meta_model=meta_model,
            data_series=data_series,
            wandb_log=configs.logger!=None,
        )    
    else:
        log.info("Loading stage 2 sequence model")
        meta_model = torch.load(f"./{configs.model.meta_config.name}.pt")
    
    # train_val_lookback_codes = np.concatenate([train_WHs, val_WHs], axis=0)[-seq_len-configs.model.output_chunk_length:]
        
    # eval model
    log.info("Evaluating model")
    from src.utils.evaluation import eval_twostage_metadeeptime_model
    results, data_series = eval_twostage_metadeeptime_model(
        model=meta_model, 
        configs=configs, 
        data_series=data_series,
        logging=configs.logger!=None,
        )
    
    if configs.logger is not None:
        # finish wandb
        wandb.finish()
        
    return model, data_series, results
