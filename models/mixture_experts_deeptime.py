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

from einops import rearrange, repeat, reduce
from ..modules.inr import INR
from ..modules.regressors import RidgeRegressor, RidgeRegressorTrimmed

from darts.logging import get_logger, raise_if_not, raise_log
from darts.models.forecasting.pl_forecasting_module import PLForecastingModule, PLPastCovariatesModule
from darts.models.forecasting.torch_forecasting_model import PastCovariatesTorchModel
from darts.utils.torch import MonteCarloDropout

logger = get_logger(__name__)


class _MixtureExpertsDeepTIMeModule(PLPastCovariatesModule):
    '''DeepTime model from https://github.com/salesforce/DeepTime/tree/main
    '''
    def __init__(
        self,
        lookback_window_length: int = 24,
        forecast_horizon_length: int = 12,
        datetime_feats: int = 0,
        layer_size: int = 256,
        inr_layers: int = 5,
        n_fourier_feats: int = 4096,
        scales: float = [0.01, 0.1, 1, 5, 10, 20, 50, 100], # TODO: don't understand
        nr_params: int = 1, # The number of parameters of the likelihood (or 1 if no likelihood is used).
        use_datetime: bool = False,
        K_value: int = 5,
        **kwargs,
        ):
        super().__init__(**kwargs)
        self.inr = INR(in_feats=datetime_feats + 1, layers=inr_layers, layer_size=layer_size,
                       n_fourier_feats=n_fourier_feats, scales=scales)
        self.adaptive_weights = RidgeRegressor()
        self.fc_gate = nn.Linear(lookback_window_length, layer_size)
        # self.adaptive_weights = RidgeRegressorTrimmed(no_remained_after_trim=3)

        self.output_chunk_length = forecast_horizon_length
        self.datetime_feats = datetime_feats
        self.inr_layers = inr_layers
        self.layer_size = layer_size
        self.n_fourier_feats = n_fourier_feats
        self.scales = scales
        self.K_value = K_value
        
        self.nr_params = nr_params
        self.use_datetime = use_datetime
        
    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x = x_in[0]
        tgt_horizon_len = self.output_chunk_length
        batch_size, lookback_len, _ = x.shape
        coords = self.get_coords(lookback_len, tgt_horizon_len).to(x.device)

        if self.use_datetime:
            raise NotImplementedError("DeepTIMeModel does not support datetime_feats yet")
            x_time, y_time = x[1], x[2]
            time = torch.cat([x_time, y_time], dim=1)
            coords = repeat(coords, '1 t 1 -> b t 1', b=time.shape[0])
            coords = torch.cat([coords, time], dim=-1)
            time_reprs = self.inr(coords)
        else:
            time_reprs = repeat(self.inr(coords), '1 t d -> b t d', b=batch_size)

        # gating mechanism
        batch_size, _, output_dim = x.shape
        batch_size, sum_chunk_length, _ = time_reprs.shape
        gate_logits = [self.fc_gate(x[..., i]) for i in range(output_dim)] # list of output_dim tensors of shape (batch_size, layer_size) 
        gate_logits = torch.stack(gate_logits, dim=-1).mean(dim=-1) #(batch_size, layer_size)
        top_K_gate_logits, top_k_indices = torch.topk(gate_logits, self.K_value, dim=-1, sorted=False) #(batch_size, K) and differentiable
        gate_values = torch.softmax(top_K_gate_logits, dim=-1) #(batch_size, K) 
        
        # selecting the experts
        selected_time_reprs = time_reprs[
            torch.arange(batch_size)[:, None, None],
            torch.arange(sum_chunk_length)[None, :, None],
            top_k_indices[:, None, :]
            ] * gate_values[:, None, :] # (batch_size, sum_chunk_length, K) 
        
        lookback_reprs = selected_time_reprs[:, :-tgt_horizon_len] # shape = (batch_size, input_chunk_length, K)
        horizon_reprs = selected_time_reprs[:, -tgt_horizon_len:] # shape = (batch_size, forecast_horizon_length, K)
        
        # # reversible intrance normalization
        # eps = 1e-5
        # expectation = x.mean(dim=1, keepdim=True)
        # standard_deviation = x.std(dim=1, keepdim=True) + eps
        # x = (x - expectation) / standard_deviation
        
        w, b = self.adaptive_weights(lookback_reprs, x) # w.shape = (batch_size, K, output_dim)
        preds = self.forecast(horizon_reprs, w, b)       
        
        # hack used for importance weights visualization (only first dim)
        self.learned_w = torch.cat([w, b], dim=1)[..., 0] # shape = (batch_size, layer_size + 1)
        
        # # reverse normalization
        # preds = preds * standard_deviation + expectation
        
        # wandb.log({'lookback_reprs': lookback_reprs[0, 0, 0], 'horizon_reprs': horizon_reprs[0, 0, 0]})
        goodness_of_base_fit = (x - torch.einsum('... d o, ... t d -> ... t o', [w, lookback_reprs]) + b).squeeze(-1).norm(dim=1).mean()
        wandb.log({'goodness_of_base_fit': goodness_of_base_fit})
        # wandb.log({'rel_norm_res': goodness_of_base_fit/x.squeeze(-1).norm(dim=1).mean()})
        
        preds = preds.view(
            preds.shape[0], self.output_chunk_length, preds.shape[2], self.nr_params
        )
        return preds

    def forecast(self, inp: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.einsum('... d o, ... t d -> ... t o', [w, inp]) + b

    def get_coords(self, lookback_len: int, horizon_len: int) -> torch.Tensor:
        coords = torch.linspace(0, 1, lookback_len + horizon_len)
        return rearrange(coords, 't -> 1 t 1')
    
    def configure_optimizers(self):
        # self.super().configure_optimizers()
        """configures optimizers and learning rate schedulers for model optimization."""

        # Create the optimizer and (optionally) the learning rate scheduler
        # we have to create copies because we cannot save model.parameters into object state (not serializable)
        optimizer_kws = {k: v for k, v in self.optimizer_kwargs.items()}
        
        no_decay_list = ('bias', 'norm',)
        group1_params = []  # lambda
        group2_params = []  # no decay
        group3_params = []  # decay
        for param_name, param in self.named_parameters():
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
        optimizer: torch.optim.optimizer.Optimizer = self.optimizer_cls(list_params, **optimizer_kws)
    

        if self.lr_scheduler_cls is not None:
            lr_sched_kws = {k: v for k, v in self.lr_scheduler_kwargs.items()}
            
            # ReduceLROnPlateau requires a metric to "monitor" which must be set separately, most others do not
            lr_monitor = lr_sched_kws.pop("monitor", None)
            
            eta_min = lr_sched_kws['eta_min']
            warmup_epochs = lr_sched_kws['warmup_epochs']
            T_max = lr_sched_kws['T_max']
            
            scheduler_fns = []
            for param_group in optimizer.param_groups:
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
            lr_scheduler = self.lr_scheduler_cls(optimizer=optimizer, lr_lambda=scheduler_fns, wandb_log=lr_sched_kws['wandb_log'])
            
            return [optimizer], {
                "scheduler": lr_scheduler,
                "monitor": lr_monitor if lr_monitor is not None else "val_loss",
            }
        else:
            return optimizer
    
class MixtureExpertsDeepTIMeModel(PastCovariatesTorchModel):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        datetime_feats: int = 0,
        layer_size: int = 256,
        inr_layers: int = 5,
        n_fourier_feats: int = 4096,
        scales: float = [0.01, 0.1, 1, 5, 10, 20, 50, 100], # TODO: don't understand
        K_value: int = 5,
        **kwargs,
        ):
        super().__init__(**self._extract_torch_model_params(**self.model_params))

        # extract pytorch lightning module kwargs
        self.pl_module_params = self._extract_pl_module_params(**self.model_params)
        
        # self.input_chunk_length = input_chunk_length
        self.datetime_feats = datetime_feats
        self.inr_layers = inr_layers
        self.layer_size = layer_size
        self.n_fourier_feats = n_fourier_feats
        self.scales = scales
        self.K_value = K_value
        
        # TODO: add this option
        if datetime_feats != 0:
            raise NotImplementedError("DeepTIMeModel does not support datetime_feats yet")
        
        
    def _create_model(self, train_sample: Tuple[torch.Tensor]) -> torch.nn.Module:     
        # samples are made of (past_target, past_covariates, ,future_target)
        input_dim = train_sample[0].shape[1]
        # (train_sample[1].shape[1] if train_sample[1] is not None else 0)
        output_dim = train_sample[-1].shape[1]
        output_chunk_length = train_sample[-1].shape[0]
        
        if self.likelihood:
            raise NotImplementedError("DeepTIMeModel does not support likelihoods yet")
        
        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters
        use_datetime = True if self.datetime_feats != 0 else False
        
        return _MixtureExpertsDeepTIMeModule(
            lookback_window_length=self.input_chunk_length,
            forecast_horizon_length=output_chunk_length,
            datetime_feats=self.datetime_feats,
            layer_size=self.layer_size,
            inr_layers=self.inr_layers,
            n_fourier_feats=self.n_fourier_feats,
            scales=self.scales,
            nr_params=nr_params,
            use_datetime=use_datetime,
            K_value=self.K_value,
            **self.pl_module_params,
            )