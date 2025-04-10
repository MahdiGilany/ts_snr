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
from .modules.inr import INR
from .modules.regressors import RidgeRegressor, RidgeRegressorTrimmed

from darts.logging import get_logger, raise_if_not, raise_log
from darts.models.forecasting.pl_forecasting_module import PLForecastingModule, PLPastCovariatesModule
from darts.models.forecasting.torch_forecasting_model import PastCovariatesTorchModel
from darts.utils.torch import MonteCarloDropout

logger = get_logger(__name__)


class _DeepTIMeModule(PLPastCovariatesModule):
    '''DeepTime model from https://github.com/salesforce/DeepTime/tree/main
    '''
    def __init__(
        self,
        forecast_horizon_length: int = 12,
        datetime_feats: int = 0,
        layer_size: int = 256,
        inr_layers: int = 5,
        n_fourier_feats: int = 4096,
        scales: float = [0.01, 0.1, 1, 5, 10, 20, 50, 100], # TODO: don't understand
        nr_params: int = 1, # The number of parameters of the likelihood (or 1 if no likelihood is used).
        use_datetime: bool = False,
        dict_reg_coef: float = 0.05,
        **kwargs,
        ):
        super().__init__(**kwargs)
        self.inr = INR(in_feats=datetime_feats + 1, layers=inr_layers, layer_size=layer_size,
                       n_fourier_feats=n_fourier_feats, scales=scales)
        self.adaptive_weights = RidgeRegressor()
        # self.adaptive_weights = RidgeRegressorTrimmed(no_remained_after_trim=3)

        self.output_chunk_length = forecast_horizon_length
        self.datetime_feats = datetime_feats
        self.inr_layers = inr_layers
        self.layer_size = layer_size
        self.n_fourier_feats = n_fourier_feats
        self.scales = scales
        self.dict_reg_coef = dict_reg_coef
        
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

        self.time_reprs = time_reprs
        # time_reprs = time_reprs/torch.norm(time_reprs, dim=1, keepdim=True)
        lookback_reprs = time_reprs[:, :-tgt_horizon_len] # shape = (batch_size, forecast_horizon_length, layer_size)
        horizon_reprs = time_reprs[:, -tgt_horizon_len:]
        
        # # reversible intrance normalization
        # eps = 1e-5
        # expectation = x.mean(dim=1, keepdim=True)
        # standard_deviation = x.std(dim=1, keepdim=True) + eps
        # x = (x - expectation) / standard_deviation
        
        w, b = self.adaptive_weights(lookback_reprs, x) # w.shape = (batch_size, layer_size, output_dim)
        preds = self.forecast(horizon_reprs, w, b)       
        
        # hack used for importance weights visualization (only first dim)
        self.learned_w = torch.cat([w, b], dim=1)[..., 0] # shape = (batch_size, layer_size + 1)
        
        # # reverse normalization
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
        
        
        try: 
            # wandb.log({'lookback_reprs': lookback_reprs[0, 0, 0], 'horizon_reprs': horizon_reprs[0, 0, 0]})
            goodness_of_base_fit = (x - torch.einsum('... d o, ... t d -> ... t o', [w, lookback_reprs]) + b).squeeze(-1).norm(dim=1).mean()
            wandb.log({'goodness_of_base_fit': goodness_of_base_fit})
            # wandb.log({'rel_norm_res': goodness_of_base_fit/x.squeeze(-1).norm(dim=1).mean()})
        except:
            pass
        
        preds = preds.view(
            preds.shape[0], self.output_chunk_length, preds.shape[2], self.nr_params
        )
        return preds

    def _compute_regularization_loss(self) -> torch.Tensor:
        """Computes the regularization loss."""
        return self.dict_reg_coef*self.time_reprs.norm(dim=1).mean() # + 0.05*self.learned_w.abs().mean()

    def training_step(self, train_batch, batch_idx) -> torch.Tensor:
        """performs the training step"""
        # tricks
        self.val = False
        self.y = train_batch[-1]
        
        output = self._produce_train_output(train_batch[:-1])
        target = train_batch[
            -1
        ]  # By convention target is always the last element returned by datasets
        loss = self._compute_loss(output, target)
        loss = loss + self._compute_regularization_loss()
        self.log(
            "train_loss",
            loss,
            batch_size=train_batch[0].shape[0],
            prog_bar=True,
            sync_dist=True,
        )
        self._calculate_metrics(output, target, self.train_metrics)
        return loss
    
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
    
class DeepTIMeModel(PastCovariatesTorchModel):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        datetime_feats: int = 0,
        layer_size: int = 256,
        inr_layers: int = 5,
        n_fourier_feats: int = 4096,
        scales: float = [0.01, 0.1, 1, 5, 10, 20, 50, 100], 
        dict_reg_coef: float = 0.05,
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
        self.dict_reg_coef = dict_reg_coef
        
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
        
        return _DeepTIMeModule(
            forecast_horizon_length=output_chunk_length,
            datetime_feats=self.datetime_feats,
            layer_size=self.layer_size,
            inr_layers=self.inr_layers,
            n_fourier_feats=self.n_fourier_feats,
            scales=self.scales,
            nr_params=nr_params,
            use_datetime=use_datetime,
            dict_reg_coef=self.dict_reg_coef,
            **self.pl_module_params,
            )