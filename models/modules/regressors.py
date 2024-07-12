'''Copied from https://github.com/salesforce/DeepTime/tree/main #TODO: add to this
'''

# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

from typing import Optional

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


class RidgeRegressor(nn.Module):
    def __init__(self, lambda_init: Optional[float] =0.):
        super().__init__()
        self._lambda = nn.Parameter(torch.as_tensor(lambda_init, dtype=torch.float), requires_grad=True)

    def forward(self, reprs: Tensor, x: Tensor, reg_coeff: Optional[float] = None) -> Tensor:
        if reg_coeff is None:
            reg_coeff = self.reg_coeff()
        w, b = self.get_weights(reprs, x, reg_coeff)
        return w, b

    def get_weights(self, X: Tensor, Y: Tensor, reg_coeff: float) -> Tensor:
        batch_size, n_samples, n_dim = X.shape
        ones = torch.ones(batch_size, n_samples, 1, device=X.device)
        X = torch.concat([X, ones], dim=-1)

        if n_samples >= n_dim:
            # standard
            A = torch.bmm(X.mT, X)
            A.diagonal(dim1=-2, dim2=-1).add_(reg_coeff)
            B = torch.bmm(X.mT, Y)
            weights = torch.linalg.solve(A, B)
        else:
            # Woodbury
            A = torch.bmm(X, X.mT)
            A.diagonal(dim1=-2, dim2=-1).add_(reg_coeff)
            weights = torch.bmm(X.mT, torch.linalg.solve(A, Y))

        return weights[:, :-1], weights[:, -1:]

    def reg_coeff(self) -> Tensor:
        return F.softplus(self._lambda)
    
    
class RidgeRegressorTrimmed(nn.Module):
    def __init__(self, lambda_init: Optional[float] =0., no_remained_after_trim: int = 10):
        super().__init__()
        self._lambda = nn.Parameter(torch.as_tensor(lambda_init, dtype=torch.float))
        self.no_remained_after_trim = no_remained_after_trim

    def forward(self, reprs: Tensor, x: Tensor, reg_coeff: Optional[float] = None) -> Tensor:
        if reg_coeff is None:
            reg_coeff = self.reg_coeff()
        w, b = self.get_weights(reprs, x, reg_coeff)
        return w, b

    def get_weights(self, X: Tensor, Y: Tensor, reg_coeff: float) -> Tensor:
        batch_size, n_samples, n_dim = X.shape
        ones = torch.ones(batch_size, n_samples, 1, device=X.device)
        X = torch.concat([X, ones], dim=-1)

        if n_samples >= n_dim:
            # standard
            A = torch.bmm(X.mT, X)
            A.diagonal(dim1=-2, dim2=-1).add_(reg_coeff)
            B = torch.bmm(X.mT, Y)
            weights = torch.linalg.solve(A, B)
        else:
            # Woodbury
            A = torch.bmm(X, X.mT)
            A.diagonal(dim1=-2, dim2=-1).add_(reg_coeff)
            weights = torch.bmm(X.mT, torch.linalg.solve(A, Y))
        
        # trimming the weights
        trim = weights.shape[1] - self.no_remained_after_trim
        weights = torch.cat([torch.zeros_like(weights[:, :trim], device=weights.device, dtype=weights.dtype), weights[:, trim:]], dim=1)
        
        return weights[:, :-1], weights[:, -1:]

    def reg_coeff(self) -> Tensor:
        return F.softplus(self._lambda)


def laplacian_kernel(X: Tensor, Y: Tensor, σ: float) -> Tensor:
    K = torch.exp(-torch.linalg.vector_norm(X.unsqueeze(1) - Y.unsqueeze(2), ord=1, dim=-1)/2*σ)
    return K

def rbf_kernel(X: Tensor, Y: Tensor, σ: float) -> Tensor:
    K = torch.exp(-(torch.norm(X.unsqueeze(1) - Y.unsqueeze(2), dim=-1)**2)/(2*σ**2))
    return K

class KernelRidgeRegressor(nn.Module):
    def __init__(self, lambda_init: Optional[float]=0.):
        super().__init__()
        self._lambda = nn.Parameter(torch.as_tensor(lambda_init, dtype=torch.float), requires_grad=True)
        self.kernel_σ = nn.Parameter(torch.as_tensor(15.0, dtype=torch.float), requires_grad=True)
        self.kernel_func = rbf_kernel # laplacian_kernel
        
    def forward(self, reprs: Tensor, x: Tensor, σ2_y: Optional[float] = None) -> Tensor:
        if σ2_y is None:
            σ2_y = self.reg_coeff()
        
        # find kernel on reprs
        K = self.kernel_func(reprs, reprs, self.reg_coeff_kernel_σ()) # exp(|x_i - x_j|^2 / 2\sigma^2)
        # K = torch.exp(-(torch.norm(reprs.unsqueeze(1) - reprs.unsqueeze(2), dim=-1)**2)/(2*1*self._lambda_rbf_σ**2)) # exp(|x_i - x_j|^2 / 2\sigma^2) 
        
        # kernel inverse
        K_inv = torch.linalg.inv(K + σ2_y*torch.eye(K.shape[1], device=K.device))
        
        α = K_inv @ x # (batch_size, lookback_len, n_dim)
        return α, self.kernel_σ.detach().cpu()

    def forecast(self, lookback_reprs: Tensor, horizon_reprs: Tensor, α: Tensor) -> Tensor:
        # find kernel on horizon reprs
        K_star = self.kernel_func(lookback_reprs, horizon_reprs, self.reg_coeff_kernel_σ())
        # K_star = torch.exp(-(torch.norm(lookback_reprs.unsqueeze(1) - horizon_reprs.unsqueeze(2), dim=-1)**2)/(2*1*self._lambda_rbf_σ**2)) # (1, horizon, lookback)

        return K_star @ α 
        
    def reg_coeff(self) -> Tensor:
        return F.softplus(self._lambda)
    
    def reg_coeff_kernel_σ(self) -> Tensor:
        return F.softplus(self.kernel_σ)