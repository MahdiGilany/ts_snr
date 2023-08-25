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

from darts.logging import get_logger, raise_if_not, raise_log
from darts.models.forecasting.pl_forecasting_module import PLForecastingModule, PLPastCovariatesModule
from darts.models.forecasting.torch_forecasting_model import PastCovariatesTorchModel
from darts.utils.torch import MonteCarloDropout

import learn2learn as l2l

logger = get_logger(__name__)

class linears(nn.Module):
    def __init__(self, in_feats, out_feats, no_parallel_fc, bias=True):
        super().__init__()
        self.list_fc = [nn.Linear(in_feats, out_feats, bias=bias) for _ in range(no_parallel_fc)]
        self.list_fc = nn.ModuleList(self.list_fc)
    
    def forward(self, x):
        batch_size = x.shape[0]
        assert (batch_size <= len(self.list_fc)), "batch_size must be less or equal than the number of parallel fc layers"
        return torch.cat([self.list_fc[i](x[i:i+1]) for i in range(batch_size)], dim=0)
        
    def reset_parameters(self):
        for fc in self.list_fc:
            fc.reset_parameters()
    
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
        batch_size: int = 256,
        **kwargs,
        ):
        super().__init__(**kwargs)
        
        # activating manual optimization
        self.automatic_optimization = False
        
        self.inr = INR(in_feats=datetime_feats + 1, layers=inr_layers, layer_size=layer_size,
                       n_fourier_feats=n_fourier_feats, scales=scales)
        self.fc = nn.Linear(layer_size, 1) # for now accepts 1-dimensional targets only
        # fcs = linears(layer_size, 1, batch_size)
        # self.maml = l2l.algorithms.MAML(fcs,
        #                                 lr=0.5,
        #                                 first_order=False
        #                                 )
        
        
        
        self._lambda = nn.Parameter(torch.as_tensor(0.0, dtype=torch.float), requires_grad=False)

        # self.adaptive_weights = RidgeRegressor()
        self.adaptation_steps = adaptation_steps
        self.output_chunk_length = forecast_horizon_length
        self.datetime_feats = datetime_feats
        self.inr_layers = inr_layers
        self.layer_size = layer_size
        self.n_fourier_feats = n_fourier_feats
        self.scales = scales
        self.val=True
        
        self.nr_params = nr_params
        self.use_datetime = use_datetime

    # # MAML fast adapt
    # def fast_adapt(self, base_learner, x, y, adaptation_steps=1):
    #     # self.maml.train()
    #     loss = nn.MSELoss()
        
    #     # define weights and biases every time
    #     # w, b = torch.tensor()
    #     # init.kaiming_uniform_(w, a=math.sqrt(5))
    #     # if self.bias is not None:
    #     #     fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
    #     #     bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    #     #     init.uniform_(self.bias, -bound, bound)
        
    #     # Adapt the model
    #     for step in range(adaptation_steps):
    #         # allow_nograd=False
    #         if self.val:
    #         #     # breakpoint()
    #         #     x.requires_grad_(True)
    #         #     # y.requires_grad_(True)
    #         #     allow_nograd = True
    #             [p.requires_grad_(True) for p in base_learner.parameters()]
    #             x = x.detach().clone().requires_grad_(True)
    #         with torch.enable_grad():
    #             # breakpoint()
    #             l2_reg = torch.tensor(0., device=x.device, dtype=x.dtype)
    #             for param in base_learner.parameters():
    #                 l2_reg += torch.norm(param)**2
    #             # l2_reg = torch.sum([p.norm(p=2)**2 for p in base_learner.parameters()])
    #             # train_error = loss(base_learner(x), y) + self.reg_coeff()*torch.norm(fc_w, p=2)**2
    #             train_error = loss(base_learner(x), y) #+ self.reg_coeff()*l2_reg
    #             # breakpoint()
    #             base_learner.adapt(train_error, allow_unused=True)# allow_nograd=allow_nograd)
    #             if not self.val:
    #                 try:
    #                     wandb.log({'MAML/train_error': train_error, 'MAML/l2_reg': l2_reg})
    #                 except:
    #                     pass
    
    # reptile fast adapt
    def fast_adapt(self, base_learner, x, y, adaptation_steps=1):
        optimizer = torch.optim.SGD(base_learner.parameters(), 0.01)
        loss = nn.MSELoss()
        
        # this ensures that inr is not affected by the adaptation
        x = x.detach().clone().requires_grad_(True)
        
        # this ensures that in meta validation, the weights are still updated 
        if self.val:
            [p.requires_grad_(True) for p in base_learner.parameters()]
        
        # turns torch.no_grad() off in meta validation
        with torch.enable_grad():
            for step in range(adaptation_steps):
                optimizer.zero_grad()
                train_error = loss(base_learner(x), y)
                train_error.backward()
                optimizer.step()
        
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
        
        # # MAML
        # self.maml.module.reset_parameters()
        # base_learner = self.maml.clone()
        # self.fast_adapt(base_learner, lookback_reprs, x, adaptation_steps=self.adaptation_steps)
        
        # preds = base_learner(horizon_reprs)
        
        
        # Reptile
        self.lookback_reprs = lookback_reprs
        
        # zero-grad the parameters
        for p in self.inr.parameters():
            p.grad = torch.zeros_like(p.data)
        for p in self.fc.parameters():
            p.grad = torch.zeros_like(p.data)
        
        
        linear_errors = []
        preds = []
        # losses = []
        for i in range(batch_size):
            inr_model = deepcopy(self.inr)
            linear_model = deepcopy(self.fc)
            linear_error = self.fast_adapt(linear_model, lookback_reprs[i:i+1,...], x[i:i+1,...], adaptation_steps=self.adaptation_steps)
            linear_errors.append(linear_error)
            
            if not self.val:
                # reptile update for self.fc
                for p, l in zip(self.fc.parameters(), linear_model.parameters()):
                    p.grad.data.add_(-1.0, l.data)
            
                # update inr model
                # sd = self._opt.state_dict()
                # lr = sd['param_groups'][2]['lr']
                lr = self.lr_schedulers().get_lr()[-1]
                optimizer = self.reptile_optimizers(inr_model.named_parameters(), lr=lr)
                # optimizer.load_state_dict(self._opt.state_dict())
                optimizer.zero_grad()
                loss = self._compute_loss(linear_model(inr_model(coords[:, :-tgt_horizon_len]))[...,None], x[i:i+1,...])            
                loss.backward()
                optimizer.step()            
                
                # reptile update for self.inr
                for p, l in zip(self.inr.parameters(), inr_model.parameters()):
                    p.grad.data.add_(-1.0, l.data)
            
            # loss = self._compute_loss(lookback_reprs[i:i+1,...] @ linear_model.weight[..., None].detach() + linear_model.bias.detach(), x[i:i+1,...])            
            # losses.append(loss)
            
            pred = horizon_reprs[i:i+1,...] @ linear_model.weight[..., None].detach() + linear_model.bias.detach()
            preds.append(pred)
            
        preds = torch.cat(preds, dim=0)
        # losses = torch.stack(losses, dim=0).mean()
        
        

        w = 0*lookback_reprs[:,0,:] #next(self.maml.parameters())
        self.learned_w = w # torch.cat([w, b], dim=1)[..., 0] # shape = (batch_size, layer_size + 1)
        
        
        
        try: 
            wandb.log({'goodness_of_base_fit': np.mean(linear_errors)})
        except:
            pass
        
        preds = preds.view(
            preds.shape[0], self.output_chunk_length, preds.shape[2], self.nr_params
        )
        return preds  #, losses
    
    # def _compute_regularization_loss(self, x_in: torch.Tensor):
    #     x = x_in[0]
    #     self._compute_loss()
    
    
    
    def training_step(self, train_batch, batch_idx) -> torch.Tensor:
        """performs the training step"""
        # manual optimization
        self._opt = self.optimizers()
        self._opt.zero_grad()
        
        self.val = False
        self.y = train_batch[-1]
        
        output = self._produce_train_output(train_batch[:-1])
        target = train_batch[
            -1
        ]  # By convention target is always the last element returned by datasets
        _loss = self._compute_loss(output, target)
        loss = _loss #+ lkbk_loss #+ self._compute_regularization_loss(train_batch[:-1])
        # self.log(
        #     "train_reg_loss",
        #     loss,
        #     batch_size=train_batch[0].shape[0],
        #     prog_bar=True,
        #     sync_dist=True,
        # )
        self.log(
            "train_loss",
            _loss,
            batch_size=train_batch[0].shape[0],
            prog_bar=True,
            sync_dist=True,
        )
        self._calculate_metrics(output, target, self.train_metrics)
        
        # normalize gradients
        for p in self.fc.parameters():
            p.grad.data.mul_(1.0 / self._cur_batch_size).add_(p.data)
        for p in self.inr.parameters():
            p.grad.data.mul_(1.0 / self._cur_batch_size).add_(p.data)
        
        # backward pass for inr
        self.manual_backward(loss) #or loss.backward()
        
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
        
        # [p.grad.data.mul_(1.0 / (self._cur_batch_size)) for p in self.maml.parameters()]
        # [p.grad.data.mul_(1.0 / (self._cur_batch_size)) for p in self.inr.parameters()]
    
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
    
    def reptile_optimizers(self, reptile_params, lr=0.01):
        # self.super().configure_optimizers()
        """configures optimizers and learning rate schedulers for model optimization."""

        # Create the optimizer and (optionally) the learning rate scheduler
        # we have to create copies because we cannot save model.parameters into object state (not serializable)
        optimizer_kws = {k: v for k, v in self.optimizer_kwargs.items()}
        
        no_decay_list = ('bias', 'norm',)
        group1_params = []  # lambda
        group2_params = []  # no decay
        group3_params = []  # decay
        for param_name, param in reptile_params:
            if '_lambda' in param_name:
                group1_params.append(param)
            elif any([mod in param_name for mod in no_decay_list]):
                group2_params.append(param)
            else:
                group3_params.append(param)
        
        list_params = [
            {'params': group1_params, 'weight_decay': 0, 'lr': 1.0, 'scheduler': 'cosine_annealing'},
            {'params': group2_params, 'weight_decay': 0, 'lr': lr},
            {'params': group3_params, 'lr': lr}
            ]
        optimizer: torch.optim.optimizer.Optimizer = self.optimizer_cls(list_params, **optimizer_kws)
    
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
            **self.pl_module_params,
            )