import sys
import os

# sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional, Union, Dict, Literal, Tuple
import plotly.express as px
from einops import repeat


import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback

from src.utils import driver as utils
from src.utils.metrics import calculate_metrics_darts, calculate_metrics
from src.data.registry.data_registry import DataSeries

from darts.timeseries import TimeSeries, concatenate
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel, DEFAULT_DARTS_FOLDER
from darts.models.forecasting.forecasting_model import ForecastingModel, LocalForecastingModel
from darts.utils.data.sequential_dataset import PastCovariatesSequentialDataset

from tqdm import tqdm
from einops import rearrange, repeat, reduce
from ..modeling.modules.regressors import RidgeRegressor
from ..modeling.modules.inr import INR



log = utils.get_logger(__name__)


class SequeceDataset(Dataset):
    def __init__(
        self, 
        seq_data: torch.tensor, 
        seq_label: Optional[torch.tensor] = None,
        seq_len: int = 24,
        ) -> None:
        super().__init__()
        self.seq_data = seq_data
        self.seq_label = seq_label if seq_label is not None else seq_data
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.seq_data) - self.seq_len + 1
    
    def __getitem__(self, index) :
        return self.seq_data[index:index+self.seq_len], self.seq_label[index:index+self.seq_len]
        

def get_lookback_horizon_codes(
   deeptime_model: TorchForecastingModel, 
   time_series: TimeSeries,
   input_chunk_length: int,
   output_chunk_length: int,
   batch_size: int,
): 
    
    series_ds = PastCovariatesSequentialDataset(
        time_series,
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        covariates=None,
        use_static_covariates=False
        )
    series_dl = DataLoader(
        series_ds,
        batch_size=deeptime_model.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=deeptime_model._batch_collate_fn,
        )
    
    
    assert deeptime_model.model is not None, "model.model not found, please load the model using model.load_from_checkpoint()\
        or model.load_weights_from_checkpoint() which initializes model.model"
    
    
    pl_deeptime_model = deeptime_model.model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pl_deeptime_model = pl_deeptime_model.to(device) 
    pl_deeptime_model.eval()
    
    # time representation is took out of for loop for faster computation (time reprs as memory block)
    coords = pl_deeptime_model.get_coords(input_chunk_length, output_chunk_length).to(device)
    single_time_reprs = pl_deeptime_model.inr(coords)
    
    # list of signal and label codes
    W_Ls = []
    W_Hs = []
    for batch in tqdm(series_dl, desc="Acquiring signal and label codes (W_L and W_H)"):
        input_series, _, _, target_series = batch
        input_series = input_series.to(device=pl_deeptime_model.device, dtype=pl_deeptime_model.dtype)
        target_series = target_series.to(device=pl_deeptime_model.device, dtype=pl_deeptime_model.dtype)
        
        # # target_series = target_series.to(device=pl_model.device, dtype=pl_model.dtype)
        # pl_deeptime_model.y = target_series.to(device=pl_deeptime_model.device, dtype=pl_deeptime_model.dtype) # a hack for setting target for twoomp model
        
        # repeating time reprs for batch size
        batch_size = input_series.shape[0]
        time_reprs = repeat(single_time_reprs, '1 t d -> b t d', b=batch_size)
        
        # look back and horizon dictionaries        
        lookback_reprs = time_reprs[:, :-output_chunk_length] # shape = (batch_size, forecast_horizon_length, layer_size)
        horizon_reprs = time_reprs[:, -output_chunk_length:]
        
        w, b = pl_deeptime_model.adaptive_weights(lookback_reprs, input_series) # [bz, 256, 1], [bz, 1, 1]
        W_L = torch.cat([w, b], dim=1) # shape = (batch_size, layer_size + 1, 1)
        W_Ls.append(W_L.detach().cpu().numpy())
        
        # hack used for importance weights visualization (only first dim)
        pl_deeptime_model.learned_w = torch.cat([w, b], dim=1)[..., 0] # shape = (batch_size, layer_size + 1)
        
        w, b = pl_deeptime_model.adaptive_weights(horizon_reprs, target_series) # [bz, 256, 1], [bz, 1, 1]
        W_H = torch.cat([w, b], dim=1) # shape = (batch_size, layer_size + 1, 1)
        W_Hs.append(W_H.detach().cpu().numpy())

    # concatenate all batches
    W_Ls = np.concatenate(W_Ls, axis=0) # shape = (num_samples, layer_size + 1, 1)
    W_Hs = np.concatenate(W_Hs, axis=0)
    
    # flip back since dataset get item is designed in reverse order
    W_Ls = W_Ls[::-1, ...]
    W_Hs = W_Hs[::-1, ...]
    
    return W_Ls.copy(), W_Hs.copy()
  
def manual_train_seq_model(
    seq_config: DictConfig,
    output_chunk_length: int,
    seq_model: nn.Module,
    train_data: Tuple[np.array, np.array],
    val_data: Tuple[np.array, np.array],
    wandb_log: bool = False,
):
    # create a dataset and data loader
    # train_x = torch.tensor(train_data[0])
    train_x = torch.tensor(train_data[1][:-output_chunk_length, ...])
    train_y = torch.tensor(train_data[1][output_chunk_length:, ...])
    train_y = train_y - torch.tensor(train_data[0][output_chunk_length:, ...])
    
    # val_x = torch.tensor(val_data[0])
    val_x = torch.tensor(val_data[1][:-output_chunk_length, ...])
    val_y = torch.tensor(val_data[1][output_chunk_length:, ...])
    val_y = val_y - torch.tensor(val_data[0][output_chunk_length:, ...])
    
    
    train_ds = SequeceDataset(train_x, train_y, seq_len=seq_config.model.seq_len)
    val_ds = SequeceDataset(val_x, val_y, seq_len=seq_config.model.seq_len)
    
    train_dl = DataLoader(
            train_ds,
            batch_size=seq_config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            )
    val_dl = DataLoader(
            val_ds,
            batch_size=seq_config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            )
    
    # train the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seq_model.to(device=device, dtype=train_x.dtype)
    seq_model.train()
    
    optimizer = torch.optim.AdamW(seq_model.parameters(), lr=seq_config.lr)
    criterion = nn.MSELoss()
    
    best_val_loss = np.inf
    early_stop_counter = 0
    for epoch in tqdm(range(seq_config.epochs), desc="Training sequence model"):
        # training
        losses = []
        for batch in train_dl:
            seq_model.zero_grad()
            seq_data, seq_label = batch
            seq_data = seq_data.to(device)
            seq_label = seq_label.to(device)
            seq_pred = seq_model(seq_data)
            loss = criterion(seq_pred[:,-1,...], seq_label[:,-1,...])
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        if wandb_log:
            wandb.log({"Seq/train_loss": np.mean(losses), "Seq/epoch": epoch})
        
        # validation
        seq_model.eval()
        losses = []
        early_stop_counter += 1
        with torch.no_grad():
            for batch in val_dl:
                seq_data, seq_label = batch
                seq_data = seq_data.to(device)
                seq_label = seq_label.to(device)
                seq_pred = seq_model(seq_data)
                loss = criterion(seq_pred[:,-1,...], seq_label[:,-1,...])
                losses.append(loss.item())
            avg_loss = np.mean(losses)
            if wandb_log:
                wandb.log({"Seq/val_loss": avg_loss})
            if avg_loss < best_val_loss:
                early_stop_counter = 0
                best_val_loss = avg_loss
                torch.save(seq_model, f"./{seq_model.model_name}.pt")
            if early_stop_counter > seq_config.patience:
                log.info("Early stopping")
                return torch.load(f"./{seq_model.model_name}.pt")
        seq_model.train()
        
    return torch.load(f"./{seq_model.model_name}.pt")
        

class MetaDataset(Dataset):
    def __init__(
        self, 
        timeseries_data: TimeSeries, 
        num_shots: int = 5,
        lookback: int = 24,
        horizon: int = 12,
        ) -> None:
        super().__init__()
        self.timeseries_data = timeseries_data._xa.values
        self.num_shots = num_shots
        self.lookback = lookback
        self.horizon = horizon
        
    def __len__(self):
        return len(self.timeseries_data) - self.lookback - 2*self.horizon - self.num_shots + 1
    
    def __getitem__(self, index):
        seq_data = []
        seq_label = []
        for i in range(self.num_shots):
            seq_data.append(self.timeseries_data[index+i:index+i+self.lookback])
            seq_label.append(self.timeseries_data[index+i+self.lookback:index+i+self.lookback+self.horizon])
        seq_data = np.stack(seq_data, axis=0)
        seq_label = np.stack(seq_label, axis=0)
        query_seq = self.timeseries_data[index+self.horizon+self.num_shots:index+self.lookback+self.horizon+self.num_shots]
        query_label = self.timeseries_data[index+self.lookback+self.horizon+self.num_shots:index+self.lookback+2*self.horizon+self.num_shots]
        return torch.tensor(seq_data).squeeze(-1), torch.tensor(seq_label).squeeze(-1), torch.tensor(query_seq).squeeze(-1), torch.tensor(query_label).squeeze(-1)


class MetaDeepTime(nn.Module):
    def __init__(self, horizon, inr=None, _lambda=None, val=False):
        super().__init__()
        self.inr = inr if inr is not None else INR(in_feats=1, layers=5, layer_size=256, n_fourier_feats=4096, scales=[0.01, 0.1, 1, 5, 10, 20, 50, 100])
        self._lambda = _lambda
        self.output_chunk_length = horizon
        self.val = val
        self.adaptive_weights = RidgeRegressor(lambda_init = _lambda if _lambda is not None else 0.)
        
    def forward(self, x):
        tgt_horizon_len = self.output_chunk_length
        batch_size, lookback_len, _ = x.shape
        coords = self.get_coords(lookback_len, tgt_horizon_len).to(x.device, dtype=x.dtype)

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
        
        try: 
            # wandb.log({'lookback_reprs': lookback_reprs[0, 0, 0], 'horizon_reprs': horizon_reprs[0, 0, 0]})
            goodness_of_base_fit = (x - torch.einsum('... d o, ... t d -> ... t o', [w, lookback_reprs]) + b).squeeze(-1).norm(dim=1).mean()
            wandb.log({'goodness_of_base_fit': goodness_of_base_fit})
            # wandb.log({'rel_norm_res': goodness_of_base_fit/x.squeeze(-1).norm(dim=1).mean()})
        except:
            pass
        
        # preds = preds.view(
        #     preds.shape[0], preds.shape[1], preds.shape[2], 1
        # )
        return preds
        
    def forecast(self, inp: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.einsum('... d o, ... t d -> ... t o', [w, inp]) + b

    def get_coords(self, lookback_len: int, horizon_len: int) -> torch.Tensor:
        coords = torch.linspace(0, 1, lookback_len + horizon_len)
        return rearrange(coords, 't -> 1 t 1')

# MAML fast adapt
def maml_fast_adapt(base_learner, x, y, val=False, adaptation_steps=1):
    loss = nn.MSELoss()
    
    # this ensures that in meta validation, the weights are still updated 
    if val:
        # this ensures that inr is not affected by the adaptation
        x = x.detach().clone().requires_grad_(True)
        [p.requires_grad_(True) for p in base_learner.parameters()]
    
    
    # turns torch.no_grad() off in meta validation
    with torch.enable_grad():
        for step in range(adaptation_steps):
            # _reg = torch.tensor(0., device=x.device, dtype=x.dtype)
            # for param in base_learner.parameters():
            #     if not self.L1:
            #         _reg += torch.norm(param)**2 #L2
            #     else:
                    # _reg += torch.abs(param).sum() #L1
            train_error = loss(base_learner(x), y) #+ self.reg_coeff()*_reg
            base_learner.adapt(train_error) #, allow_unused=True)
    
    return train_error.item()

def manual_train_meta_deeptime(
    meta_config: DictConfig,
    input_chunk_length: int,
    output_chunk_length: int,
    meta_model: nn.Module,
    data_series: DataSeries,
    wandb_log: bool = False,
    ):
    # create a dataset and data loader
    train_data = data_series.train_series
    val_data = data_series.val_series
    
    train_ds = MetaDataset(train_data, num_shots=meta_config.num_shots, lookback=input_chunk_length, horizon=output_chunk_length)
    val_ds = MetaDataset(val_data, num_shots=meta_config.num_shots, lookback=input_chunk_length, horizon=output_chunk_length)
    
    train_dl = DataLoader(
            train_ds,
            batch_size=meta_config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            )
    
    val_dl = DataLoader(
            val_ds,
            batch_size=meta_config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            )
    
    
    # train the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    meta_model = meta_model.to(device=device, dtype=train_ds[0][0].dtype)
    
    # maml model
    import learn2learn as l2l
    maml_model = l2l.algorithms.MAML(meta_model,
                                    lr=meta_config.adapt_lr,
                                    first_order=False
                                    )
    
    for name, param in maml_model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    
    optimizer = torch.optim.AdamW(meta_model.parameters(), lr=meta_config.lr)
    criterion = nn.MSELoss()
    
    best_val_loss = np.inf
    early_stop_counter = 0
    for epoch in tqdm(range(meta_config.epochs), desc="Training meta model, epochs"):
        # training
        maml_model.train()
        for batch in tqdm(train_dl, desc="Training meta model, batches"):
            optimizer.zero_grad()
            
            support_seqs, support_labels, query_seqs, query_labels = batch
            support_seqs = support_seqs.to(device)
            support_labels = support_labels.to(device)
            query_seqs = query_seqs.to(device)
            query_labels = query_labels.to(device)
            
            batch_size = support_seqs.shape[0]
            
            train_errors = []
            losses = []
            for i in range(batch_size):
                cloned_maml_model = maml_model.clone() # head of the model
                # fast adapt
                train_error = maml_fast_adapt(cloned_maml_model, support_seqs[i], support_labels[i], adaptation_steps=meta_config.adapt_steps)
                train_errors.append(train_error)
                #meta test
                pred = cloned_maml_model(query_seqs[i:i+1])
                loss = criterion(pred, query_labels[i:i+1])
                loss.backward(retain_graph=True)
                losses.append(loss.item())    
            
            for p in maml_model.parameters():
                p.grad.data.mul_(1.0 / batch_size)
            optimizer.step()

            if wandb_log:
                wandb.log({"Meta/train_adapt_errors": np.mean(train_errors), "Meta/train_loss": np.mean(losses)})
        
        
        # validation
        maml_model.eval()
        losses = []
        early_stop_counter += 1
        with torch.no_grad():
            for batch in val_dl:
                support_seqs, support_labels, query_seqs, query_labels = batch
                support_seqs = support_seqs.to(device)
                support_labels = support_labels.to(device)
                query_seqs = query_seqs.to(device)
                query_labels = query_labels.to(device)
                
                batch_size = support_seqs.shape[0]
                
                train_errors = []
                for i in range(batch_size):
                    cloned_maml_model = maml_model.clone() # head of the model
                    # fast adapt
                    train_error = maml_fast_adapt(cloned_maml_model, support_seqs[i], support_labels[i], val=True, adaptation_steps=meta_config.adapt_steps)
                    train_errors.append(train_error)
                    #meta test
                    pred = cloned_maml_model(query_seqs[i:i+1])
                    loss = criterion(pred, query_labels[i:i+1])
                    losses.append(loss.item())  
                    if wandb_log:
                        wandb.log({"Meta/val_adapt_errors": np.mean(train_errors)})
                    
            avg_loss = np.mean(losses)
            if wandb_log:
                wandb.log({"Meta/val_loss": avg_loss})
            if avg_loss < best_val_loss:
                early_stop_counter = 0
                best_val_loss = avg_loss
                torch.save(maml_model, f"./{meta_config.name}.pt")
            if early_stop_counter > meta_config.patience:
                log.info("Early stopping")
                return torch.load(f"./{meta_config.name}.pt")
        
    return torch.load(f"./{meta_config.name}.pt")

def manual_train_meta_deeptime_closedform(
    meta_config: DictConfig,
    input_chunk_length: int,
    output_chunk_length: int,
    meta_model: nn.Module,
    data_series: DataSeries,
    wandb_log: bool = False,
    ):
    # create a dataset and data loader
    train_data = data_series.train_series
    val_data = data_series.val_series
    
    train_ds = MetaDataset(train_data, num_shots=meta_config.num_shots, lookback=input_chunk_length, horizon=output_chunk_length)
    val_ds = MetaDataset(val_data, num_shots=meta_config.num_shots, lookback=input_chunk_length, horizon=output_chunk_length)
    
    train_dl = DataLoader(
            train_ds,
            batch_size=meta_config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            )
    
    val_dl = DataLoader(
            val_ds,
            batch_size=meta_config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            )
    
    
    # train the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    meta_model = meta_model.to(device=device, dtype=train_ds[0][0].dtype)
    
    # # maml model
    # import learn2learn as l2l
    # maml_model = l2l.algorithms.MAML(meta_model,
    #                                 lr=meta_config.adapt_lr,
    #                                 first_order=False
    #                                 )
    
    # for name, param in maml_model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.shape)
    
    optimizer = torch.optim.AdamW(meta_model.parameters(), lr=meta_config.lr)
    criterion = nn.MSELoss()
    
    best_val_loss = np.inf
    early_stop_counter = 0
    for epoch in tqdm(range(meta_config.epochs), desc="Training meta model, epochs"):
        # training
        meta_model.train()
        for batch in train_dl:
            optimizer.zero_grad()
            
            support_seqs, support_labels, query_seqs, query_labels = batch
            support_seqs = support_seqs.to(device)
            support_labels = support_labels.to(device)
            query_seqs = query_seqs.to(device)
            query_labels = query_labels.to(device)
            
            batch_size, num_shots, lookback, outdim = support_seqs.shape
            
            # fast adapt
            support_latent = meta_model(support_seqs.reshape(-1, lookback, outdim)).reshape(batch_size, num_shots, -1) # (batch_size, shots, horizon)
            support_latentTsupport_latent = torch.matmul(support_latent.transpose(1, 2), support_latent) # (batch_size, horizon, horizon)
            support_latentTsupport_labels = torch.matmul(support_latent.transpose(1, 2), support_labels.squeeze(-1)) # (batch_size, horizon, horizon)
            support_latentTsupport_latent.diagonal(dim1=-2, dim2=-1).add_(0.6)
            W_HH = torch.matmul(torch.inverse(support_latentTsupport_latent), support_latentTsupport_labels) # (batch_size, horizon, horizon)
            
            #meta test
            query_latent = meta_model(query_seqs).squeeze(-1).reshape(batch_size, 1, -1) # (batch_size, 1, horizon)
            preds = torch.matmul(query_latent, W_HH).transpose(1,2) # (batch_size, horizon, 1)
            
            # preds = []
            # for i in range(batch_size):
            #     # fast adapt
            #     support_latent = meta_model(support_seqs[i]).squeeze(-1).unsqueeze(0) # (1, shots, horizon)
            #     support_latentTsupport_latent = torch.matmul(support_latent.transpose(1, 2), support_latent) # (1, horizon, horizon)
            #     support_latentTsupport_labels = torch.matmul(support_latent.transpose(1, 2), support_labels[i].squeeze(-1).unsqueeze(0)) # (1, horizon, horizon)
            #     W_HH = torch.matmul(torch.inverse(support_latentTsupport_latent), support_latentTsupport_labels) # (1, horizon, horizon)
                
            #     #meta test
            #     query_latent = meta_model(query_seqs[i:i+1]).squeeze(-1).unsqueeze(0) # (1, 1, horizon)
            #     pred = torch.matmul(query_latent, W_HH).squeeze(0).unsqueeze(-1) # (1, horizon, 1)
            #     preds.append(pred)
            # preds = torch.cat(preds, dim=0) # (batch_size, horizon, 1)
            
            loss = criterion(preds, query_labels)
            loss.backward()
            optimizer.step()
            
            if wandb_log:
                wandb.log({"Meta/train_loss": loss.item()})
        
        
        # validation
        meta_model.eval()
        losses = []
        early_stop_counter += 1
        with torch.no_grad():
            for batch in val_dl:
                support_seqs, support_labels, query_seqs, query_labels = batch
                support_seqs = support_seqs.to(device)
                support_labels = support_labels.to(device)
                query_seqs = query_seqs.to(device)
                query_labels = query_labels.to(device)
                
                batch_size, num_shots, lookback, outdim = support_seqs.shape
            
                # fast adapt
                support_latent = meta_model(support_seqs.reshape(-1, lookback, outdim)).reshape(batch_size, num_shots, -1) # (batch_size, shots, horizon)
                support_latentTsupport_latent = torch.matmul(support_latent.transpose(1, 2), support_latent) # (batch_size, horizon, horizon)
                support_latentTsupport_labels = torch.matmul(support_latent.transpose(1, 2), support_labels.squeeze(-1)) # (batch_size, horizon, horizon)
                support_latentTsupport_latent.diagonal(dim1=-2, dim2=-1).add_(0.6)
                W_HH = torch.matmul(torch.inverse(support_latentTsupport_latent), support_latentTsupport_labels) # (batch_size, horizon, horizon)
                
                #meta test
                query_latent = meta_model(query_seqs).squeeze(-1).reshape(batch_size, 1, -1) # (batch_size, 1, horizon)
                preds = torch.matmul(query_latent, W_HH).transpose(1,2) # (batch_size, horizon, 1)
                
                # preds = []
                # for i in range(batch_size):
                #     # fast adapt
                #     support_latent = meta_model(support_seqs[i]).squeeze(-1).unsqueeze(0) # (1, shots, horizon)
                #     support_latentTsupport_latent = torch.matmul(support_latent.transpose(1, 2), support_latent) # (1, horizon, horizon)
                #     support_latentTsupport_labels = torch.matmul(support_latent.transpose(1, 2), support_labels[i].squeeze(-1).unsqueeze(0)) # (1, horizon, horizon)
                #     W_HH = torch.matmul(torch.inverse(support_latentTsupport_latent), support_latentTsupport_labels) # (1, horizon, horizon)
                    
                #     #meta test
                #     query_latent = meta_model(query_seqs[i:i+1]).squeeze(-1).unsqueeze(0) # (1, 1, horizon)
                #     pred = torch.matmul(query_latent, W_HH).squeeze(0).unsqueeze(-1) # (1, horizon, 1)
                #     preds.append(pred)
                # preds = torch.cat(preds, dim=0) # (batch_size, horizon, 1)
                
                loss = criterion(preds, query_labels)
                losses.append(loss.item())
                    
            avg_loss = np.mean(losses)
            if wandb_log:
                wandb.log({"Meta/val_loss": avg_loss})
            if avg_loss < best_val_loss:
                early_stop_counter = 0
                best_val_loss = avg_loss
                torch.save(meta_model, f"./{meta_config.name}.pt")
            if early_stop_counter > meta_config.patience:
                log.info("Early stopping")
                return torch.load(f"./{meta_config.name}.pt")
        
    return torch.load(f"./{meta_config.name}.pt")

