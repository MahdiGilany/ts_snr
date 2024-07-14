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
class KernelOptimizerConfig(OptimizerConfig):
    lr_σ: float = 0.01 
    σ_warmup_epochs: int = 0

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
    
    data_config: DataConfig = DataConfig(dataset_name="ettm2")
    model_config: KernelDeepTimeConfig = KernelDeepTimeConfig()
    
    optimizer_config: KernelOptimizerConfig = KernelOptimizerConfig()
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
        
    def setup_optimizer(self):
        """configures optimizers and learning rate schedulers for model optimization."""
        no_decay_list = ('bias', 'norm',)
        group1_params = []  # lambda
        groupσ_params = []  # σ
        group2_params = []  # no decay
        group3_params = []  # decay
        for param_name, param in self.model.named_parameters():
            if '_lambda' in param_name:
                group1_params.append(param)
            elif 'kernel_σ' in param_name:
                groupσ_params.append(param)
            elif any([mod in param_name for mod in no_decay_list]):
                group2_params.append(param)
            else:
                group3_params.append(param)

        list_params = [
            {'params': group1_params, 'weight_decay': 0, 'lr': 1.0, 'scheduler': 'cosine_annealing'},
            {'params': group2_params, 'weight_decay': 0, 'scheduler': 'cosine_annealing_with_linear_warmup'},
            {'params': group3_params, 'scheduler': 'cosine_annealing_with_linear_warmup'},
            {'params': groupσ_params, 'weight_decay': 0, 'lr': self.config.optimizer_config.lr_σ, 'scheduler': 'cosine_annealing_with_zero_warmup'},
            ]
        self.optimizer: optim.optimizer.Optimizer = optim.Adam(
            list_params,
            lr=self.config.optimizer_config.lr,
            weight_decay=self.config.optimizer_config.weight_decay, 
            amsgrad=self.config.optimizer_config.amsgrad
            )
    
    
        eta_min = self.config.scheduler_config.eta_min
        warmup_epochs = self.config.scheduler_config.warmup_epochs * len(self.train_loader)
        σ_warmup_epochs = self.config.optimizer_config.σ_warmup_epochs * len(self.train_loader)
        T_max = self.config.scheduler_config.T_max * len(self.train_loader)
        
        scheduler_fns = []
        for param_group in self.optimizer.param_groups:
            scheduler = param_group['scheduler']
            
            if scheduler == 'none':
                fn = lambda T_cur: 1
                
            elif scheduler == 'cosine_annealing':
                lr = eta_max = param_group['lr']
                fn = lambda T_cur: (eta_min + 0.5 * (eta_max - eta_min)  * (
                            1.0 + math.cos((T_cur - warmup_epochs) / (T_max - warmup_epochs) * math.pi))) / lr
                
            elif scheduler == 'cosine_annealing_with_linear_warmup':
                lr = eta_max = param_group['lr']
                fn = lambda T_cur: T_cur / warmup_epochs if T_cur < warmup_epochs else (eta_min + 0.5 * (
                            eta_max - eta_min) * (1.0 + math.cos(
                    (T_cur - warmup_epochs) / (T_max - warmup_epochs) * math.pi))) / lr
            
            elif scheduler == 'cosine_annealing_with_zero_warmup':
                lr = eta_max = param_group['lr']
                fn = lambda T_cur: 0 if T_cur < σ_warmup_epochs else (eta_min + 0.5 * (
                            eta_max - eta_min) * (1.0 + math.cos(
                    (T_cur - warmup_epochs) / (T_max - warmup_epochs) * math.pi))) / lr
                
            else:
                raise ValueError(f'No such scheduler, {scheduler}')
            
            scheduler_fns.append(fn)
        self.scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=scheduler_fns)
    
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
            
            pred_series, kernel_σ = self.model(input_series)
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
                wandb.log({"kernel_σ": kernel_σ, "epoch": self.epoch})
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