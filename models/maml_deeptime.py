'''Copied and modified from https://github.com/salesforce/DeepTime/tree/main #TODO: add to this
'''

import warnings
from typing import Any, List, NewType, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import math
import wandb
from copy import deepcopy

from einops import rearrange, repeat, reduce
from ..modules.inr import INR
from ..modules.regressors import RidgeRegressor, RidgeRegressorTrimmed
from ..modules.weight_decay import L2, L1

from darts.logging import get_logger, raise_if_not, raise_log
from darts.models.forecasting.pl_forecasting_module import PLForecastingModule, PLPastCovariatesModule
from darts.models.forecasting.torch_forecasting_model import PastCovariatesTorchModel
from darts.utils.torch import MonteCarloDropout

import learn2learn as l2l

logger = get_logger(__name__)

class linears(nn.Module):
    def __init__(self, in_feats, out_feats, no_parallel_fc, bias=True):
        super().__init__()
        # self.list_fc = [nn.Linear(in_feats, out_feats, bias=bias) for _ in range(no_parallel_fc)]
        # self.list_fc = nn.ModuleList(self.list_fc)
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.no_parallel_fc = no_parallel_fc
        self.weights = nn.Parameter(torch.randn(no_parallel_fc, in_feats, out_feats), requires_grad=True)
        self.biases = nn.Parameter(torch.randn(no_parallel_fc, 1, out_feats), requires_grad=True)
        # self.weights = nn.Parameter(torch.randn(in_feats, out_feats), requires_grad=True)
        # self.biases = nn.Parameter(torch.randn(1, out_feats), requires_grad=True)
        self.reset_parameters()
        
    def forward(self, x):
        batch_size = x.shape[0]
        assert (batch_size <= self.weights.shape[0]), "batch_size must be less or equal than the number of parallel fc layers"
        out = x @ self.weights[:batch_size,...] + self.biases[:batch_size,...]
        return out
        # weights = repeat(self.weights[None,...], '1 i o -> b i o', b=batch_size)
        # biases = repeat(self.biases[None,...], '1 i o -> b i o', b=batch_size)
        # return x @ weights + biases
        # assert (batch_size <= len(self.list_fc)), "batch_size must be less or equal than the number of parallel fc layers"
        # return torch.cat([self.list_fc[i](x[i:i+1]) for i in range(batch_size)], dim=0)
        
    def reset_parameters(self):
        # for fc in self.list_fc:
        #     fc.reset_parameters()
        # torch.nn.init.xavier_uniform_(self.weights)
        for i in range (self.no_parallel_fc):
            torch.nn.init.kaiming_uniform_(self.weights[i,...], a=math.sqrt(5))
            if self.biases is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weights[i,...])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                torch.nn.init.uniform_(self.biases[i,...], -bound, bound)
    
class _DeepTIMeModelMAML(PLPastCovariatesModule):
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
        adaptation_steps: int = 15,
        adaptation_lr: float = 0.01,
        batch_size: int = 256,
        fast_version: bool = False,
        reset_linears: bool = False,
        L1: bool = False,
        **kwargs,
        ):
        super().__init__(**kwargs)
        
        # activating manual optimization
        self.automatic_optimization = False
        
        self._lambda = nn.Parameter(torch.as_tensor(0.0, dtype=torch.float), requires_grad=True)
        
        self.inr = INR(in_feats=datetime_feats + 1, layers=inr_layers, layer_size=layer_size,
                       n_fourier_feats=n_fourier_feats, scales=scales)
        
        # for now accepts 1-dimensional targets only
        if fast_version:
            fc = linears(layer_size, 1, batch_size)
        else:
            fc = nn.Linear(layer_size, 1)

        self.maml = l2l.algorithms.MAML(fc,
                                        lr=adaptation_lr,
                                        first_order=False
                                        )
        
        self.adaptation_steps = adaptation_steps
        self.output_chunk_length = forecast_horizon_length
        self.datetime_feats = datetime_feats
        self.inr_layers = inr_layers
        self.layer_size = layer_size
        self.n_fourier_feats = n_fourier_feats
        self.scales = scales
        self.fast_version = fast_version
        self.reset_linears = reset_linears # reseting linear heads for 
        self.L1 = L1 
        self.val=True
        
        self.nr_params = nr_params
        self.use_datetime = use_datetime

    # MAML fast adapt
    def fast_adapt(self, base_learner, x, y, adaptation_steps=1):
        loss = nn.MSELoss()
        
        # this ensures that in meta validation, the weights are still updated 
        if self.val:
            # this ensures that inr is not affected by the adaptation
            x = x.detach().clone().requires_grad_(True)
            [p.requires_grad_(True) for p in base_learner.parameters()]
        
        
        # turns torch.no_grad() off in meta validation
        with torch.enable_grad():
            for step in range(adaptation_steps):
                _reg = torch.tensor(0., device=x.device, dtype=x.dtype)
                for param in base_learner.parameters():
                    if not self.L1:
                        _reg += torch.norm(param)**2 #L2
                    else:
                        _reg += torch.abs(param).sum() #L1
                train_error = loss(base_learner(x), y) + self.reg_coeff()*_reg
                base_learner.adapt(train_error) #, allow_unused=True)
        
        return train_error.item()
    
    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x = x_in[0]
        tgt_horizon_len = self.output_chunk_length
        batch_size, lookback_len, outdim = x.shape
        self._cur_batch_size = batch_size
        assert outdim == 1, "DeepTIMeModelMAML only supports 1-dimensional targets for now"
        coords = self.get_coords(lookback_len, tgt_horizon_len).to(x.device) # shape = (1, lookback_len + forecast_horizon_length, 1)
        
        time_reprs = repeat(self.inr(coords), '1 t d -> b t d', b=batch_size) # shape = (batch_size, lookback_len + forecast_horizon_length, layer_size)

        lookback_reprs = time_reprs[:, :-tgt_horizon_len] # shape = (batch_size, forecast_horizon_length, layer_size)
        horizon_reprs = time_reprs[:, -tgt_horizon_len:]
        
        # zero-grad the parameters
        for p in self.parameters():
            p.grad = torch.zeros_like(p.data)

        linear_errors = []
        if self.fast_version:
            linears_model = self.maml.clone() # head of the model
            linear_error = self.fast_adapt(linears_model, lookback_reprs, x, adaptation_steps=self.adaptation_steps)
            linear_errors.append(linear_error)
            preds = linears_model(horizon_reprs)
            if not self.val:
                loss = self._compute_loss(preds[...,None], self.y)
                loss.backward(retain_graph=True)
            preds = preds.detach() # no backward from this point so detach
            
            # for visualization
            w = linears_model.weights.detach()
            
            # reset parameters
            if self.reset_linears:
                self.maml.module.reset_parameters()
        else:
            preds = []
            ws = []
            for i in range(batch_size):
                linear_model = self.maml.clone() # head of the model
                linear_error = self.fast_adapt(linear_model, lookback_reprs[i:i+1,...], x[i:i+1,...], adaptation_steps=self.adaptation_steps)
                linear_errors.append(linear_error)
                
                pred = linear_model(horizon_reprs[i:i+1,...])
                if not self.val:
                    loss = self._compute_loss(pred[...,None], self.y[i:i+1,...])
                    loss.backward(retain_graph=True)
                preds.append(pred.detach()) # no backward from this point so detach
                
                # for visualization
                w = linear_model.weight[..., None].detach()
                b = linear_model.bias.detach()
                ws.append(w)
                

            preds = torch.cat(preds, dim=0)
            w = torch.cat(ws, dim=0)
        
        # for visualization
        self.learned_w = w[...,0] # torch.cat([w, b], dim=1)[..., 0] # shape = (batch_size, layer_size + 1)
        try: 
            wandb.log({'goodness_of_base_fit': np.mean(linear_errors)})
        except:
            pass
        
        preds = preds.view(
            preds.shape[0], self.output_chunk_length, preds.shape[2], self.nr_params
        )
        return preds

    def training_step(self, train_batch, batch_idx) -> torch.Tensor:
        """performs the training step"""
        # manual optimization
        self._opt = self.optimizers()
        self._opt.zero_grad()
        
        # tricks
        self.val = False
        self.y = train_batch[-1]
        
        output = self._produce_train_output(train_batch[:-1])
        target = train_batch[
            -1
        ]  # By convention target is always the last element returned by datasets
        loss = self._compute_loss(output, target)
        self.log(
            "train_loss",
            loss,
            batch_size=train_batch[0].shape[0],
            prog_bar=True,
            sync_dist=True,
        )
        self._calculate_metrics(output, target, self.train_metrics)
        
        # normalize gradients
        if self.fast_version:
            pass
            # for p in self.parameters():
            #     p.grad.data.mul_(1.0 / self._cur_batch_size)
        else:
            for p in self.parameters():
                p.grad.data.mul_(1.0 / self._cur_batch_size)
                
        self._opt.step()
        if self.trainer.is_last_batch:
            self.lr_schedulers().step()
        return loss

    def validation_step(self, val_batch, batch_idx) -> torch.Tensor:
        """performs the validation step"""
        self.val = True
        output = self._produce_train_output(val_batch[:-1])
        target = val_batch[-1]
        loss = self._compute_loss(output, target)
        self.log(
            "val_loss",
            loss,
            batch_size=val_batch[0].shape[0],
            prog_bar=True,
            sync_dist=True,
        )
        self._calculate_metrics(output, target, self.val_metrics)
        return loss

    def backward(self, loss: Tensor, *args: Any, **kwargs: Any) -> None:
        super().backward(loss, *args, **kwargs)
    
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
                                1.0 + math.cos((T_cur+1 - warmup_epochs) / (T_max - warmup_epochs) * math.pi))) / lr
                    
                elif scheduler == 'cosine_annealing_with_linear_warmup':
                    lr = eta_max = param_group['lr']
                    fn = lambda T_cur: (T_cur + 1) / warmup_epochs if T_cur < warmup_epochs else (eta_min + 0.5 * (
                                eta_max - eta_min) * (1.0 + math.cos(
                        (T_cur + 1 - warmup_epochs) / (T_max - warmup_epochs) * math.pi))) / lr
                    
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
 
    def reg_coeff(self) -> Tensor:
        return F.softplus(self._lambda)

class DeepTIMeModelMAML(PastCovariatesTorchModel):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        datetime_feats: int = 0,
        layer_size: int = 256,
        inr_layers: int = 5,
        n_fourier_feats: int = 4096,
        scales: float = [0.01, 0.1, 1, 5, 10, 20, 50, 100], # TODO: don't understand
        batch_size: int = 256,
        adaptation_steps: int = 15,
        adaptation_lr: float = 0.01,
        L1: bool = False,
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
        self.batch_size = batch_size
        self.adaptation_steps = adaptation_steps
        self.adaptation_lr = adaptation_lr
        self.L1 = L1
        
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
        
        return _DeepTIMeModelMAML(
            forecast_horizon_length=output_chunk_length,
            datetime_feats=self.datetime_feats,
            layer_size=self.layer_size,
            inr_layers=self.inr_layers,
            n_fourier_feats=self.n_fourier_feats,
            scales=self.scales,
            nr_params=nr_params,
            use_datetime=use_datetime,
            batch_size=self.batch_size,
            adaptation_steps=self.adaptation_steps,
            adaptation_lr=self.adaptation_lr,
            L1=self.L1,
            **self.pl_module_params,
            )