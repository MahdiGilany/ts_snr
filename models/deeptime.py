'''Copied and modified from https://github.com/salesforce/DeepTime/tree/main #TODO: add to this
'''

import warnings
from typing import List, NewType, Tuple, Union

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import math
import wandb
from dataclasses import dataclass, field

from einops import rearrange, repeat, reduce
from .modules.inr import INR
from .modules.regressors import RidgeRegressor, RidgeRegressorTrimmed

from darts.logging import get_logger, raise_if_not, raise_log
from darts.models.forecasting.pl_forecasting_module import PLForecastingModule, PLPastCovariatesModule
from darts.models.forecasting.torch_forecasting_model import PastCovariatesTorchModel
from darts.utils.torch import MonteCarloDropout


@dataclass
class DeepTimeConfig:
    """Configuration for the model."""
    model_horizon: int = 96
    datetime_feats: int = 0
    layer_size: int = 256
    inr_layers: int = 5
    n_fourier_feats: int = 4096
    scales: list = field(default_factory=lambda: [0.01, 0.1, 1, 5, 10, 20, 50, 100])
    dict_reg_coef: float = 0.0


class DeepTIMeModel(nn.Module):
    '''DeepTime model from https://github.com/salesforce/DeepTime/tree/main
    '''
    def __init__(
        self,
        config: DeepTimeConfig = DeepTimeConfig(),
        ):
        super().__init__()
        self.config = config
        self.config.horizon = self.config.model_horizon
        self.inr = INR(
            in_feats=config.datetime_feats + 1,
            layers=config.inr_layers, 
            layer_size=config.layer_size,
            n_fourier_feats=config.n_fourier_feats,
            scales=config.scales
            )
        self.adaptive_weights = RidgeRegressor()
        # self.adaptive_weights = RidgeRegressorTrimmed(no_remained_after_trim=3)
        
       
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tgt_horizon_len = self.config.horizon
        batch_size, lookback_len, _ = x.shape
        coords = self.get_coords(lookback_len, tgt_horizon_len).to(x.device)

        if False: # self.data_time
            raise NotImplementedError("DeepTIMeModel does not support datetime_feats yet")
            x_time, y_time = x[1], x[2]
            time = torch.cat([x_time, y_time], dim=1)
            coords = repeat(coords, '1 t 1 -> b t 1', b=time.shape[0])
            coords = torch.cat([coords, time], dim=-1)
            time_reprs = self.inr(coords)
        else:
            time_reprs = repeat(self.inr(coords), '1 t d -> b t d', b=batch_size)

        self.time_reprs = time_reprs
        # time_reprs = time_reprs/torch.norm(time_reprs, dim=1, keepdim=True)
        lookback_reprs = time_reprs[:, :-tgt_horizon_len] # shape = (batch_size, forecast_horizon_length, layer_size)
        horizon_reprs = time_reprs[:, -tgt_horizon_len:]
        
        # # RevIN
        # eps = 1e-5
        # expectation = x.mean(dim=1, keepdim=True)
        # standard_deviation = x.std(dim=1, keepdim=True) + eps
        # x = (x - expectation) / standard_deviation
        
        w, b = self.adaptive_weights(lookback_reprs, x) # w.shape = (batch_size, layer_size, output_dim)
        preds = self.forecast(horizon_reprs, w, b)       
        
        # hack used for importance weights visualization (only first dim)
        self.learned_w = torch.cat([w, b], dim=1)[..., 0] # shape = (batch_size, layer_size + 1)
        
        # # reverse RevIN
        # preds = preds * standard_deviation + expectation
        
        
        # # reversible intrance normalization per 96 steps
        # eps = 1e-5
        # x = x.view(batch_size, -1, 96, x.shape[-1]) # shape = (batch_size, lookback_len/96, 96, input_dim)
        # expectation = x.mean(dim=2, keepdim=True)
        # standard_deviation = x.std(dim=2, keepdim=True) + eps
        # x = (x - expectation) / standard_deviation
        # # x = x.reshape(batch_size, -1, x.shape[-1]) # shape = (batch_size, lookback_len, input_dim)
        
        # # predictions
        # preds_all = []
        # lookback_reprs = lookback_reprs.view(batch_size, -1, 96, lookback_reprs.shape[-1]) 
        # w_all = []
        # b_all = []
        # for i in range(lookback_reprs.shape[1]):
        #     w, b = self.adaptive_weights(lookback_reprs[:,i,...], x[:,i,...]) # w.shape = (batch_size, layer_size, output_dim)
        #     w_all.append(w)
        #     b_all.append(b)
        #     preds_all.append(self.forecast(horizon_reprs, w, b)*standard_deviation[:,i,...]+expectation[:,i,...])
        # preds = torch.stack(preds_all, dim=0)
        # preds = preds.mean(dim=0)
        
        # w = torch.stack(w_all, dim=0).mean(dim=0)
        # b = torch.stack(b_all, dim=0).mean(dim=0)
        # self.learned_w = torch.cat([w, b], dim=1)[..., 0] # shape = (batch_size, layer_size + 1)
        
        
        # try: 
        #     # wandb.log({'lookback_reprs': lookback_reprs[0, 0, 0], 'horizon_reprs': horizon_reprs[0, 0, 0]})
        #     goodness_of_base_fit = (x - torch.einsum('... d o, ... t d -> ... t o', [w, lookback_reprs]) + b).squeeze(-1).norm(dim=1).mean()
        #     wandb.log({'goodness_of_base_fit': goodness_of_base_fit})
        #     # wandb.log({'rel_norm_res': goodness_of_base_fit/x.squeeze(-1).norm(dim=1).mean()})
        # except:
        #     pass
        return preds

    def reg_loss(self) -> torch.Tensor:
        """Computes the regularization loss."""
        return self.time_reprs.norm(dim=1).mean() # + 0.05*self.learned_w.abs().mean()
    
    def forecast(self, inp: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.einsum('... d o, ... t d -> ... t o', [w, inp]) + b

    def get_coords(self, lookback_len: int, horizon_len: int) -> torch.Tensor:
        coords = torch.linspace(0, 1, lookback_len + horizon_len)
        return rearrange(coords, 't -> 1 t 1')
