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

from deeptime_experiment import DeepTimeExp, DataConfig, OptimizerConfig, SchedulerConfig
from models.kernel_deeptime import KernelDeepTIMeModel, KernelDeepTimeConfig              
from utils.setup import BasicExperiment, BasicExperimentConfig
from utils.loss_registry import TorchLosses
from utils.optimizer_scheduler import LambdaLRWrapper
from utils.metrics import calculate_metrics
from utils.evaluation import ( 
    sliding_window,
    wandb_log_results_and_plots,
    wandb_log_bases)


@dataclass
class KernelDeepTimeExpConfig(BasicExperimentConfig):
    """Configuration for the experiment."""
    name: str = "kernel-deeptime_test"
    group: str = None
    project: str = "timeseries" 
    resume: bool = False
    debug: bool = False
    use_wandb: bool = True
    
    seed: int = 0
    epochs: int = 50
    batch_size: int = 256
    
    horizon: int = 96
    
    data_config: DataConfig = DataConfig()
    model_config: KernelDeepTimeConfig = KernelDeepTimeConfig()
    
    optimizer_config: OptimizerConfig = OptimizerConfig()
    scheduler_config: SchedulerConfig = SchedulerConfig()

    def __post_init__(self):
        super().__post_init__()
        self.data_config.horizon = self.horizon
        self.model_config.horizon = self.horizon
        self.scheduler_config.T_max = self.epochs
        self.scheduler_config.eta_max = self.optimizer_config.lr


class KernelDeepTimeExp(DeepTimeExp):
    
    config: KernelDeepTimeExpConfig
    config_class = KernelDeepTimeExpConfig

    def setup_model(self):
        model = KernelDeepTIMeModel(self.config.model_config).cuda()
        return model
    
    def run_epoch(self, loader, train=True, desc="train"):
        with torch.no_grad() if not train else torch.enable_grad():
            self.model.train() if train else self.model.eval()
        
        # criterion = TorchLosses('mse')
        criterion = torch.nn.MSELoss(reduction='none')
        preds = []
        targets = []
        losses = []
        for i, batch in enumerate(tqdm(loader, desc=desc)):
            input_series, _, _, target_series = batch
            input_series = input_series.cuda()
            target_series = target_series.cuda()
            
            pred_series = self.model(input_series)
            loss_mse = criterion(pred_series, target_series)
            loss = loss_mse.mean() # + self.model.regularization_losses(self.epoch) 
            # norm_coef*self.model.dict_basis_norm_loss() + cov_coef*self.model.dict_basis_cov_loss()
            losses.append(loss_mse.detach().cpu().numpy())
            
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optimizer.step()
                self.scheduler.step()
                wandb.log({"lr": self.scheduler.get_last_lr()[1], "epoch": self.epoch})
            else:
                # collecting predictions for val and test
                preds.append(pred_series.detach().cpu())
                targets.append(target_series.detach().cpu())
                if input_series.shape[0] == 1:
                    # skip final batch if batch_size == 1
                    # due to bug in torch.linalg.solve which raises error when batch_size == 1
                    continue
        
        if desc == "test":
            preds = torch.cat(preds, dim=0)
            targets = torch.cat(targets, dim=0)
            # flip back as dataset get_item is designed in reverse order
            preds = preds.flip(dims=[0]) 
            targets = targets.flip(dims=[0])
            self.eval_test(preds, targets)

        total_avg_loss = np.concatenate(losses).mean()
        wandb.log({f"{desc}_loss": total_avg_loss, "epoch": self.epoch})
        return total_avg_loss

if __name__ == '__main__': 
    KernelDeepTimeExp.submit()