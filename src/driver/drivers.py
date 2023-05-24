import sys
import os

# sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from typing import List, Optional

import wandb
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback

from src.utils import driver as utils


log = utils.get_logger(__name__)


def train_model():
    ...


def eval_model():
    ...
    

def simple_run(configs: DictConfig):
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
    from torchmetrics import MetricCollection, MeanAbsolutePercentageError
    mape = MeanAbsolutePercentageError()
    metrics = MetricCollection([mape])
    
    
    # instantiate darts model 
    log.info(f"Instantiating model <{configs.model._target_}>")
    model = instantiate(configs.model, pl_trainer_kwargs=pl_trainer_kwargs, torch_metrics=metrics, save_checkpoints=False)
    
    
    # check model type
    from darts.models.forecasting.pl_forecasting_module import PLForecastingModule
    from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
    if isinstance(model, TorchForecastingModel) == False:
        raise ValueError("Model must not be a subclass of TorchForecastingModel")


    # train model
    log.info("Training model")
    model.fit(
        train_series,
        epochs=configs.epochs,
        verbose=True,
        num_loader_workers=0,
        val_series=val_series #TODO: does it get normalized?
        ) # trainer
    
    
    # eval model #TODO needs work
    log.info("Evaluating model")
    model.model.load_from_checkpoint(f"./checkpoints/{configs.name}.ckpt") # TODO: needs work
    
    pred_series = model.predict(series=train_series, n=configs.model.output_chunk_length)
    pred_series = scaler[0].inverse_transform(pred_series) if scaler else pred_series
    train_series = scaler[0].inverse_transform(train_series) if scaler else train_series
    
    
    # TODO: needs work
    from darts.metrics import mape
    mape_result = mape(val_series, pred_series)
    print(
        "Mean absolute percentage error: {:.2f}%.".format(
            mape_result
            )
        )
    
    
    wandb.log({"val_best_mape": np.array(mape_result)})
    
    # plot
    plt.figure(figsize=(10, 6))
    train_series.plot(label="train")
    val_series.plot(label="val")
    pred_series.plot(label="pred")
    # plt.savefig('pred.png')
    
    # finish wandb
    wandb.finish()
    return