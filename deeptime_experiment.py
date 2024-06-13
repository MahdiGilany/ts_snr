import os
from dotenv import load_dotenv
# Loading environment variables
load_dotenv()

import random
import math
import logging
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as tp
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler, LambdaLR
import matplotlib.pyplot as plt
import timm
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from einops import rearrange, repeat
from tqdm.auto import tqdm
from copy import copy, deepcopy
from simple_parsing import subgroups
from timm.optim.optim_factory import create_optimizer
from darts import TimeSeries
from darts.models.forecasting.torch_forecasting_model import PastCovariatesTorchModel
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel, DEFAULT_DARTS_FOLDER
from darts.utils.data.sequential_dataset import PastCovariatesSequentialDataset
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel as TFModel

from models.deeptime import DeepTIMeModel, DeepTimeConfig              
from utils.setup import BasicExperiment, BasicExperimentConfig
from utils.loss_registry import TorchLosses
from utils.optimizer_scheduler import LambdaLRWrapper
from utils.metrics import calculate_metrics
from utils.evaluation import ( 
    sliding_window,
    wandb_log_results_and_plots,
    wandb_log_bases)



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
class DataConfig:
    """Configuration for the data."""
    dataset_name: str = "exchange_rate"
    lookback: int = 288
    horizon: int = 96
    split_ratio: float = None
    use_scaler: bool = True
    target_series_index: int = -1
    
    def __post_init__(self):
        if self.dataset_name == "exchange_rate" and self.target_series_index == -1:
            self.target_series_index = -2
            

@dataclass
class DeepTimeExpConfig(BasicExperimentConfig):
    """Configuration for the experiment."""
    name: str = "deeptime"
    group: str = None
    project: str = "timeseries" 
    resume: bool = False
    debug: bool = False
    use_wandb: bool = True
    
    seed: int = 0
    epochs: int = 50
    batch_size: int = 256
    
    data_config: DataConfig = DataConfig(dataset_name="exchange_rate")
    model_config: DeepTimeConfig = DeepTimeConfig(model_horizon=data_config.horizon)
    
    optimizer_config: OptimizerConfig = OptimizerConfig()
    scheduler_config: SchedulerConfig = SchedulerConfig(T_max=epochs, eta_max=optimizer_config.lr)


class DeepTimeExp(BasicExperiment):
    
    config: DeepTimeExpConfig
    config_class = DeepTimeExpConfig
    
    @staticmethod
    def fix_seed(seed):
        # fix seed
        if (s := seed) is not None:
            logging.info(f"Global seed set to {s}")
            random.seed(s)
            np.random.seed(s)
            torch.manual_seed(s)
            torch.cuda.manual_seed_all(s)
    
    def __call__(self):
        self.setup()
        
        logging.info('Training model')
        for self.epoch in range(self.epoch, self.config.epochs):
            print(f"Epoch {self.epoch}")
            train_loss = self.run_epoch(self.train_loader, train=True, desc="train")
            val_loss = self.run_epoch(self.val_loader, train=False, desc="val")
            
            if self.best_val_loss >= val_loss:
                self.best_val_loss = val_loss
                self.save_states(best_model=True)
            
        logging.info('Test model')
        # Run test and save states if best score updated
        self.model.load_state_dict(torch.load(os.path.join(self.ckpt_dir,"best_model.pth")))
        self.run_epoch(self.test_loader, train=False, desc="test")
    
    def setup(self):
        super().setup()
        self.fix_seed(self.config.seed)
        self.setup_data()
        
        logging.info('Setting up model, optimizer, scheduler')
        self.model: DeepTIMeModel = self.setup_model()
        self.setup_optimizer()
        
        # Load checkpoint if exists
        state = None
        if "experiment.ckpt" in os.listdir(self.ckpt_dir) and self.config.resume:
            state = torch.load(self.ckpt_dir / "experiment.ckpt")
            logging.info(f"Resuming from epoch {state['epoch']}")
            self.model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.epoch = state["epoch"]
            self.best_val_loss = state["best_val_loss"]
        
        # Setup epoch and best score
        self.epoch = 0 if state is None else self.epoch
        self.best_val_loss = np.inf if state is None else self.best_val_loss
        
        logging.info(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")
        logging.info(f"""Trainable parameters: 
                     {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}""")

    def setup_data(self):
        from data.data_registry import create_dataset, DataSeries 
        from darts.timeseries import TimeSeries, concatenate        
        
        data_series: DataSeries = create_dataset(**self.config.data_config.__dict__)
        
        self.train_series: TimeSeries = data_series.train_series
        self.val_series: TimeSeries = data_series.val_series
        self.test_series: TimeSeries = data_series.test_series
        self.scaler = data_series.scaler
        self.components = self.test_series.components
        
        # backtest series
        num_trimmed_train_val = max(len(self.test_series),self.config.data_config.lookback)
        self.train_val_series_trimmed = concatenate([self.train_series, self.val_series])[-num_trimmed_train_val:] # TODO: this is not a good way to do it
        self.test_hat_series = concatenate([self.train_val_series_trimmed[-self.config.data_config.lookback:], self.test_series]) # use a lookback of val for testing

        # datasets and loaders
        train_ds = PastCovariatesSequentialDataset(
            self.train_series,
            input_chunk_length=self.config.data_config.lookback,
            output_chunk_length=self.config.data_config.horizon,
            covariates=None,
            use_static_covariates=False
            )
        val_ds = PastCovariatesSequentialDataset(
            self.val_series,
            input_chunk_length=self.config.data_config.lookback,
            output_chunk_length=self.config.data_config.horizon,
            covariates=None,
            use_static_covariates=False
            )
        test_ds = PastCovariatesSequentialDataset(
            self.test_hat_series,
            input_chunk_length=self.config.data_config.lookback,
            output_chunk_length=self.config.data_config.horizon,
            covariates=None,
            use_static_covariates=False
            )
        
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            collate_fn=TFModel._batch_collate_fn,
            )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            collate_fn=TFModel._batch_collate_fn,
            )
        self.test_loader = DataLoader(
            test_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            collate_fn=TFModel._batch_collate_fn,
            )
        
    def setup_optimizer(self):
        """configures optimizers and learning rate schedulers for model optimization."""
        no_decay_list = ('bias', 'norm',)
        group1_params = []  # lambda
        group2_params = []  # no decay
        group3_params = []  # decay
        for param_name, param in self.model.named_parameters():
            if '_lambda' in param_name:
                group1_params.append(param)
            elif any([mod in param_name for mod in no_decay_list]):
                group2_params.append(param)
            else:
                group3_params.append(param)

        list_params = [
            {'params': group1_params, 'weight_decay': 0, 'lr': 1.0, 'scheduler': 'cosine_annealing'},
            {'params': group2_params, 'weight_decay': 0, 'scheduler': 'cosine_annealing_with_linear_warmup'},
            {'params': group3_params, 'scheduler': 'cosine_annealing_with_linear_warmup'}
            ]
        self.optimizer: optim.optimizer.Optimizer = optim.Adam(
            list_params,
            lr=self.config.optimizer_config.lr,
            weight_decay=self.config.optimizer_config.weight_decay, 
            amsgrad=self.config.optimizer_config.amsgrad
            )
    
    
        eta_min = self.config.scheduler_config.eta_min
        warmup_epochs = self.config.scheduler_config.warmup_epochs * len(self.train_loader)
        T_max = self.config.scheduler_config.T_max * len(self.train_loader)
        
        scheduler_fns = []
        for param_group in self.optimizer.param_groups:
            scheduler = param_group['scheduler']
            
            if scheduler == 'none':
                fn = lambda T_cur: 1
                
            elif scheduler == 'cosine_annealing':
                lr = eta_max = param_group['lr']
                fn = lambda T_cur: (eta_min + 0.5 * (eta_max - eta_min) * (
                            1.0 + math.cos((T_cur - warmup_epochs) / (T_max - warmup_epochs) * math.pi))) / lr
                
            elif scheduler == 'cosine_annealing_with_linear_warmup':
                lr = eta_max = param_group['lr']
                fn = lambda T_cur: T_cur / warmup_epochs if T_cur < warmup_epochs else (eta_min + 0.5 * (
                            eta_max - eta_min) * (1.0 + math.cos(
                    (T_cur - warmup_epochs) / (T_max - warmup_epochs) * math.pi))) / lr
                
            else:
                raise ValueError(f'No such scheduler, {scheduler}')
            
            scheduler_fns.append(fn)
        self.scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=scheduler_fns)
    
    def setup_model(self):
        model = DeepTIMeModel(self.config.model_config).cuda()
        return model
    
    def run_epoch(self, loader, train=True, desc="train"):
        with torch.no_grad() if not train else torch.enable_grad():
            self.model.train() if train else self.model.eval()
        
        criterion = TorchLosses('mse')
        preds = []
        for i, batch in enumerate(tqdm(loader, desc=desc)):
            input_series, _, _, target_series = batch
            input_series = input_series.cuda()
            target_series = target_series.cuda()
            
            pred_series = self.model(input_series)
            norm_coef = self.config.model_config.dict_basis_norm_coeff
            cov_coef = self.config.model_config.dict_basis_cov_coeff
            loss = criterion(pred_series, target_series) + norm_coef*self.model.dict_basis_norm_loss() + cov_coef*self.model.dict_basis_cov_loss()
            wandb.log({f"{desc}_loss": loss.item(), "epoch": self.epoch})
            
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                wandb.log({"lr": self.scheduler.get_last_lr()[1], "epoch": self.epoch})
            else:
                # collecting predictions for val and test
                preds.append(pred_series.detach().cpu())
        
        if desc == "test":
            preds = torch.cat(preds, dim=0)
            # flip back as dataset get_item is designed in reverse order
            preds = preds.flip(dims=[0]) 
            self.eval_test(preds)

        return loss.item()
    
    def eval_test(self, preds):
        # TODO: this eval only works for self.test_series for now
        logging.info('Evaluation of test series')
        # turn preds and targets into TimeSeries
        list_backtest_series = []
        lookback = self.config.data_config.lookback
        horizon = self.config.data_config.horizon
        for i in tqdm(range(preds.shape[0]), desc="Turn predictions into timeseries"):
            backtest_series = TimeSeries.from_times_and_values(
                self.test_hat_series.time_index[lookback+i:lookback+i+horizon], # test_hat_series starts from inside val_series
                preds[i,...].detach().cpu().numpy(),
                freq=self.test_series.freq,
                columns=self.test_series.components
                )
            list_backtest_series.append(backtest_series)
        
        # calculate metrics        
        (metrics,
        metrics_unscaled,
        test_unscaled_series,
        list_backtest_unscaled_series,
        train_val_unscaled_series_trimmed,
        ) = self.calculate_metrics(list_backtest_series)
        
        logging.info("logging metrics")
        self.log_metrics(
            metrics,
            metrics_unscaled,
            self.components,
            self.test_series,
            test_unscaled_series,
            list_backtest_series,
            list_backtest_unscaled_series,
            self.train_val_series_trimmed,
            train_val_unscaled_series_trimmed
            )
    
    def calculate_metrics(self, list_backtest_series):
        # unnormalize series
        train_val_unscaled_series_trimmed = self.scaler.inverse_transform(self.train_val_series_trimmed) if self.scaler else self.train_val_series_trimmed
        test_unscaled_series = self.scaler.inverse_transform(self.test_series) if self.scaler else self.test_series
        list_backtest_unscaled_series = [self.scaler.inverse_transform(backtest_series) for backtest_series in list_backtest_series] if self.scaler else list_backtest_series
        
        # TODO: add this for crypto dataset
        # calculating results for Target components only, if available (for crypto dataset)
        # target_indices = np.array(['Target' in component for component in components])
        # if target_indices.any():
        #     components = list(self.test_series.components[target_indices])
        #     test_series = self.test_series[components] 
        #     test_unscaled_series = test_unscaled_series[components]
        #     list_backtest_series = [backtest_series[components] for backtest_series in list_backtest_series]
        #     list_backtest_unscaled_series = [backtest_series[components] for backtest_series in list_backtest_unscaled_series]
        
        # calculate metrics    
        predictions = np.stack([series._xa.values for series in list_backtest_series], axis=0).squeeze(-1) # (len(test_series)-horizon+1, horizon, outdim)
        predictions_unscaled = np.stack([series._xa.values for series in list_backtest_unscaled_series], axis=0).squeeze(-1) # (len(test_series)-horizon+1, horizon, outdim)
        logging.info("Calculating metrics for backtesting")
        metrics = calculate_metrics(
            true=sliding_window(self.test_series, self.config.data_config.horizon),
            pred=predictions
            )

        logging.info("Calculating metrics for unnormalized backtesting")
        metrics_unscaled = calculate_metrics(
            true=sliding_window(test_unscaled_series, self.config.data_config.horizon),
            pred=predictions_unscaled
            )
        
        return metrics, metrics_unscaled, test_unscaled_series, list_backtest_unscaled_series, train_val_unscaled_series_trimmed
    
    def log_metrics(
        self,
        metrics,
        metrics_unscaled,
        components,
        test_series,
        test_unscaled_series,
        list_backtest_series,
        list_backtest_unscaled_series,
        train_val_series_trimmed,
        train_val_unscaled_series_trimmed,
        ):
        wandb_log_results_and_plots(
            metrics=metrics,
            metrics_unscaled=metrics_unscaled,
            epoch=self.epoch,
            output_chunk_length=self.config.data_config.horizon,
            components=components,
            test_series=test_series,
            test_unscaled_series=test_unscaled_series,
            list_backtest_series=list_backtest_series,
            list_backtest_unscaled_series=list_backtest_unscaled_series,
            train_val_series_trimmed=train_val_series_trimmed,
            train_val_unscaled_series_trimmed=train_val_unscaled_series_trimmed,
            )
        
        # log DeepTime bases
        wandb_log_bases(
            self.model,
            self.config.data_config.lookback,
            self.config.data_config.horizon,
            self.config.exp_dir,
            )
    
    def save_states(self, best_model=False):
        if best_model:
            torch.save(
                self.model.state_dict(),
                os.path.join(self.ckpt_dir, "best_model.pth")
                )
        else:
            torch.save(
            {   
                "model": self.model.state_dict(), # if save_model else None,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": self.epoch,
                "best_val_loss": self.best_val_loss,
            },
            os.path.join(self.ckpt_dir,"experiment.ckpt"),
        )
    
    def checkpoint(self):
        self.save_states()
        return super().checkpoint()
    
    
    
if __name__ == '__main__': 
    DeepTimeExp.submit()