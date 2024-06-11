import os
from dotenv import load_dotenv
# Loading environment variables
load_dotenv()

import random
import logging
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as tp
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import timm
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from einops import rearrange, repeat
from tqdm.auto import tqdm
from copy import copy, deepcopy
from simple_parsing import subgroups
from timm.optim.optim_factory import create_optimizer
from darts.models.forecasting.torch_forecasting_model import PastCovariatesTorchModel
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel, DEFAULT_DARTS_FOLDER

from utils.setup import BasicExperiment, BasicExperimentConfig


@dataclass
class SchedulerConfig:
    T_max: int = 50
    eta_min: float = 0.0
    eta_max: float = 0.001
    warmup_epochs: int = 5

@dataclass  
class OptimizerConfig:
    lr: float = 0.001
    weight_decay: float = 0
    amsgrad: bool = False

@dataclass
class TrainerConfig:
    accelerator: str = "auto"
    devices: int =  1 
    num_nodes: int = 1 
    max_epochs: int = None 
    limit_train_batches: int = None
    check_val_every_n_epoch: int = 5
    num_sanity_val_steps: tp.Optional[] = None
    # enable_checkpointing: null # makes training super slow, superisingly even if it is null
    enable_model_summary: tp.Optional[] = None
    log_every_n_steps: int = 5
    accumulate_grad_batches: int = 1 

@dataclass
class DataConfig:
    """Configuration for the data."""
    dataset_name: str = "exchange_rate"
    split_ratio: tp.Tuple[float] = (0.7, 0.1, 0.2)
    use_scaler: bool = True
    target_series_index: int = -1


@dataclass
class DeepTimeConfig:
    """Configuration for the model."""
    model_name: str = "deeptime"
    input_chunk_length: int = 288
    output_chunk_length: int = 96
    datetime_feats: int = 0
    layer_size: int = 256
    inr_layers: int = 5
    n_fourier_feats: int = 4096
    scales: tp.List[float] = [0.01, 0.1, 1, 5, 10, 20, 50, 100] 
    dict_reg_coef: float = 0.05

@dataclass
class DeepTimeExpConfig(BasicExperimentConfig):
    """Configuration for the experiment."""
    name: str = "deeptime"
    group: str = None
    project: str = "timeseries" 
    resume: bool = True
    debug: bool = False
    use_wandb: bool = True
    
    seed: int = 0
    epochs: int = 50
    batch_size: int = 32
    
    data_config: DataConfig = DataConfig(name="etth2", lookback=488, horizon=96)
    
    optimizer_config: OptimizerConfig = OptimizerConfig()
    scheduler_config: SchedulerConfig = SchedulerConfig(T_max=epochs, eta_max=optimizer_config.lr)
    trainer_config: TrainerConfig = TrainerConfig(max_epochs=epochs)
    model_config: DeepTimeConfig = DeepTimeConfig()

class DeepTimeExp(BasicExperiment):
    
    config: DeepTimeExpConfig
    config_class = DeepTimeExpConfig
    
    def __call__(self):
        self.setup()
        
        logging.info('Training model')
        self.model.fit(
            self.train_series,
            epochs=self.config.epochs,
            verbose=True,
            num_loader_workers=4,
            val_series=self.val_series,
            )
        
        logging.info('Evaluating model')
        self.eval()
        

                
    def setup(self):
        super().setup()
        
        self.fix_seed()
        self.setup_data()
        
        logging.info('Setting up model, optimizer, scheduler')
        self.callbacks = self.setup_callbacks()
        self.metrics = self.setup_metrics()
        self.model: PastCovariatesTorchModel = self.setup_model()
        
        # Setup epoch and best score
        self.epoch = 0 
        
        logging.info(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")
        logging.info(f"""Trainable parameters: 
                     {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}""")
    
    def fix_seed(self):
        # fix seed
        if (s := self.config.seed) is not None:
            logging.info(f"Global seed set to {s}")
            random.seed(s)
            np.random.seed(s)
            torch.manual_seed(s)
            torch.cuda.manual_seed_all(s)
    
    def setup_data(self):
        from data.data_registry import create_dataset, DataSeries 
        from darts.timeseries import TimeSeries, concatenate        
        
        data_series: DataSeries = create_dataset(self.config.data_config)
        
        self.train_series: TimeSeries = data_series.train_series
        self.val_series: TimeSeries = data_series.val_series
        self.test_series: TimeSeries = data_series.test_series
        self.scaler = data_series.scaler
        self.components = self.test_series.components
        
        # Backtest series
        num_trimmed_train_val = max(len(self.test_series),self.config.model_config.input_chunk_length)
        train_val_series_trimmed = concatenate([self.train_series, self.val_series])[-num_trimmed_train_val:] # TODO: this is not a good way to do it
        self.train_val_test_series_trimmed = concatenate([train_val_series_trimmed[-self.config.model_config.input_chunk_length:], self.test_series]) # use a lookback of val for testing
    
    def setup_callbacks(self):
        from pytorch_lightning.callbacks import EarlyStopping
        return [EarlyStopping(monitor="val_loss", mode="min", patience=10)]
    
    def setup_metrics(self):
        from torchmetrics import MetricCollection, MeanAbsolutePercentageError, MeanSquaredError, MeanAbsoluteError
        mape = MeanAbsolutePercentageError()
        mse = MeanSquaredError()
        mae = MeanAbsoluteError()
        metrics = MetricCollection([mape, mse, mae])
        return metrics
    
    def setup_model(self):
        from models.deep_time import DeepTIMeModel
        from models.utils.loss_registry import TorchLosses
        from models.utils.optimizer_scheduler import LambdaLRWrapper
        
        pl_trainer_kwargs = {}
        pl_trainer_kwargs.update(self.config.trainer_config)
        pl_trainer_kwargs["callbacks"] = self.callbacks
        
        loss_fn = TorchLosses('mse')
        
        model = DeepTIMeModel(
            self.config.model_config,
            pl_trainer_kwargs=pl_trainer_kwargs,
            loss_fn=loss_fn,
            optimizer_cls=optim.Adam,
            optimizer_kwargs=self.config.optimizer_config,
            lr_scheduler_cls=LambdaLRWrapper,
            lr_scheduler_kwargs=self.config.scheduler_config,
            torch_metrics=self.metrics,
            save_checkpoints=not self.config.debug,
            work_dir=self.ckpt_dir,
            random_state=self.config.seed,
            batch_size=self.config.batch_size
            )
        return model
    
    def eval(self):
        logging.info('Loading model')
        model = TorchForecastingModel.load_from_checkpoint(
            model_name=self.config.model_config.model_name,
            work_dir=self.ckpt_dir,
            best=True
            ) # TODO: needs work if not using wandb or not after training
        
        logging.info('Backtesting model')
        list_backtest_series, test_preds, test_targets = historical_forecasts_manual(
        model=model,
        test_series=self.train_val_test_series_trimmed,
        input_chunk_length=self.config.model_config.input_chunk_length,
        output_chunk_length=self.config.model_config.output_chunk_length,
        plot_weights=True if (("deeptime" in self.config.model_config.model_name.lower()) and logging) else False,
        )
        
        
        return
    
    def save_states(self, best_model=False):
        return
    
    def checkpoint(self):
        self.save_states()
        return super().checkpoint()
    
    
    
if __name__ == '__main__': 
    DeepTimeExp.submit()