'''Copied and modified from https://github.com/salesforce/DeepTime/tree/main #TODO: add to this
'''

import warnings
from typing import List, NewType, Tuple, Union

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import math
import wandb
from dataclasses import dataclass, field

from einops import rearrange, repeat, reduce
from .modules.inr import INR
from .modules.regressors import KernelRidgeRegressor, RidgeRegressor, RidgeRegressorTrimmed

from darts.logging import get_logger, raise_if_not, raise_log
from darts.models.forecasting.pl_forecasting_module import PLForecastingModule, PLPastCovariatesModule
from darts.models.forecasting.torch_forecasting_model import PastCovariatesTorchModel
from darts.utils.torch import MonteCarloDropout


@dataclass
class KernelDeepTimeConfig:
    """Configuration for the model."""
    horizon: int = 96
    datetime_feats: int = 0
    layer_size: int = 256
    inr_layers: int = 5
    n_fourier_feats: int = 4096
    scales: list = field(default_factory=lambda: [0.01, 0.1, 1, 5, 10, 20, 50, 100])
    # dict_basis_norm_coeff: float = 0.0
    # dict_basis_cov_coeff: float = 0.0
    # w_var_coeff: float = 0.0
    # w_cov_coeff: float = 0.0


class KernelDeepTIMeModel(nn.Module):
    '''DeepTime model from https://github.com/salesforce/DeepTime/tree/main
    '''
    def __init__(
        self,
        config: KernelDeepTimeConfig = KernelDeepTimeConfig(),
        ):
        super().__init__()
        self.config = config
        self.inr = INR(
            in_feats=config.datetime_feats + 1,
            layers=config.inr_layers, 
            layer_size=config.layer_size,
            n_fourier_feats=config.n_fourier_feats,
            scales=config.scales
            )
        self.adaptive_α = KernelRidgeRegressor()
        
       
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tgt_horizon_len = self.config.horizon
        batch_size, lookback_len, _ = x.shape
        coords = self.get_coords(lookback_len, tgt_horizon_len).to(x.device)
        
        # time_reprs = repeat(self.inr(coords), '1 t d -> b t d', b=batch_size)
        time_reprs = self.inr(coords) # (1, lookback + horizon, layer_size)
        self.time_reprs = time_reprs

        # time_reprs = time_reprs/torch.norm(time_reprs, dim=1, keepdim=True)
        lookback_reprs = time_reprs[:, :-tgt_horizon_len] # shape = (batch_size, horizen, layer_size)
        horizon_reprs = time_reprs[:, -tgt_horizon_len:]
    
        α = self.adaptive_α(lookback_reprs, x) 
        preds = self.adaptive_α.forecast(lookback_reprs, horizon_reprs, α) # (batch_size, horizen, n_dim)
  
        self.learned_α = α
        
        return preds
    
    '''
    def dict_basis_norm_loss(self) -> torch.Tensor:
        """Computes the regularization loss."""
        B, T, D = self.time_reprs.shape
        return (self.time_reprs.norm(dim=1)/T).mean() # + 0.05*self.learned_w.abs().mean()
    
    def dict_basis_cov_loss(self) -> torch.Tensor:
        """Computes the regularization loss."""
        B, T, D = self.time_reprs.shape
        time_reprs = self.time_reprs[0, :, :]
        
        mean_over_time = time_reprs.mean(dim=0, keepdim=True)
        normalized_reprs = time_reprs - mean_over_time 
        cov_reprs = (normalized_reprs.permute(1, 0) @ normalized_reprs) / T
        
        diag = torch.eye(D,device=time_reprs.device)
        # return torch.square(cov_reprs[~diag.bool()]).mean()
        return torch.square(cov_reprs-torch.eye(D,device=time_reprs.device)).mean()
    
    def w_var_cov_loss(self):
        B, D, O = self.learned_w.shape
        w = self.learned_w
        
        eps = 1e-4        
        w_bar = w - w.mean(dim=0)
        diag = torch.eye(D, device=w.device)
        cov_loss = torch.tensor(0.0, device=w.device)
        std_loss = torch.tensor(0.0, device=w.device)
        for o in range(O):
            cov_w = (w_bar[..., o].T @ w_bar[..., o]) / (B - 1)
            cov_loss += (cov_w[~diag.bool()].pow_(2).sum() / D)
            std_loss += torch.mean(F.relu(1 - torch.sqrt(cov_w.diag())))
        
        return cov_loss/O, std_loss/O
    
    def regularization_losses(self, epoch: int = None):
        dict_norm = self.dict_basis_norm_loss()
        dict_cov = self.dict_basis_cov_loss()
        # w_cov, w_std = self.w_var_cov_loss()
        
        reg_loss = self.config.dict_basis_norm_coeff * dict_norm + \
            self.config.dict_basis_cov_coeff * dict_cov # + \
            # self.config.w_cov_coeff * w_cov + \
            # self.config.w_var_coeff * w_std
        
        # if epoch is not None:
        #     wandb.log({'dict_cov': dict_cov, 'w_cov': w_cov, 'w_std': w_std, 'epoch': epoch})
        # else:
        #     wandb.log({'dict_cov': dict_cov, 'w_cov': w_cov, 'w_std': w_std})
        return reg_loss
    '''
    
    def get_coords(self, lookback_len: int, horizon_len: int) -> torch.Tensor:
        coords = torch.linspace(0, 1, lookback_len + horizon_len)
        return rearrange(coords, 't -> 1 t 1')
