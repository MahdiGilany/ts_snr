'''Modified from https://github.com/salesforce/DeepTime/tree/main #TODO: add to this
'''

# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV

from tqdm import tqdm

def norm2(y):
    return torch.sqrt(torch.sum(y**2, dim=(1,2)))


class _OrthogonalMatchingPursuit(nn.Module):
    def __init__(
        self,
        stop: int = 500,
        r_thresh: float = 0.01,
        lambda_init: Optional[float] = -15.,
        n_nonzero_coefs: Optional[int] = None,
        ):
        super().__init__()
        
        self._lambda = torch.as_tensor(lambda_init, dtype=torch.float)

        self.n_nonzero_coefs = n_nonzero_coefs
        self.r_thresh = r_thresh
        self.stop = stop
        
    # def fit(self, dict: Tensor, y: Tensor):
    #     self.coef = []
    #     # for i in tqdm(range(dict.shape[0]), desc='OMP fitting'):
    #     for i in range(dict.shape[0]):
    #         self.omp = OrthogonalMatchingPursuit(n_nonzero_coefs=self.n_nonzero_coefs, tol=0.01)
    #         self.omp.fit(dict[i,...].detach().cpu().numpy(), y[i,...].detach().cpu().numpy())
    #         coef = torch.tensor(self.omp.coef_).to(device=dict.device, dtype=dict.dtype)
    #         self.coef.append(coef.unsqueeze(-1))
    #     self.coef = torch.stack(self.coef, dim=0)
    #     return self.coef
    
    def fit(self, dict: Tensor, y: Tensor):
        self.coef, _ = self.OMP(dict, y)
        return self.coef
    
    def forward(self, X: Tensor, coef: Tensor = None) -> Tensor:
        if coef is None:
            coef = self.coef
        
        # adding bias term
        ones = torch.ones(X.shape[0], X.shape[1], 1, device=X.device)
        dict = torch.concat([X, ones], dim=-1)
        
        return torch.bmm(dict, coef)
        
    def OMP(self, X: Tensor, y: Tensor):
        '''Orthogonal Matching pursuit algorithm
        :args
        A: measurement matrix
        y: 
        '''
        # bias added
        ones = torch.ones(X.shape[0], X.shape[1], 1, device=X.device)
        dict = torch.concat([X, ones], dim=-1)
        
        batch_sz, n_samples, input_dim = dict.shape
        batch_sz, n_samples, output_dim = y.shape
        
        assert output_dim == 1, 'OMP only supports single output so far'
        
        r = y
        weights = torch.zeros(batch_sz, input_dim, output_dim).to(device=dict.device, dtype=dict.dtype)
        Lambdas = []
        i = 0
        tolerance = True
        # Control stop interation with norm thresh or sparsity
        while tolerance and i<self.stop: # TODO: norm is the mean over all the batch which shouldn't be         
            # Compute the score of each atoms
            scores = torch.bmm(dict.mT, r) # (batch_sz, input_dim, 1)
            scores = scores.detach().cpu().numpy()
            
            # Select the atom with the max score
            Lambda = np.argmax(abs(scores), axis=1)
            # print(Lambda)
            Lambdas.append(Lambda)
            Lambdas_array = np.array(Lambdas) # (n_selected_atoms, batch_sz, 1)
            
            # All selected atoms form a basis
            basis = [] # (batch_sz, n_lambdas, input_dim)
            for j in range(batch_sz):
                basis.append(dict[j, :, Lambdas_array[:,j,0]])
            basis = torch.stack(basis, dim=0)

            # Least square solution for y=Ax
            # inv_basiss = torch.linalg.inv(torch.bmm(basis.mT, basis))
            # B = torch.bmm(basis.mT, y)
            # # inv_basiss.diagonal(dim1=-2, dim2=-1).add_(1e-2)
            # w = torch.bmm(inv_basiss, B)
            # for j in range(batch_sz):
            #     weights[j, Lambdas_array[:,j,0], 0] = w[j, :, :].squeeze()
            
            
            if n_samples >= input_dim:
                # standard
                A = torch.bmm(basis.mT, basis)
                A.diagonal(dim1=-2, dim2=-1).add_(self.reg_coeff)
                B = torch.bmm(basis.mT, y)
                w = torch.linalg.solve(A, B)
                for j in range(batch_sz):
                    weights[j, Lambdas_array[:,j,0], 0] = w[j, :, :].squeeze()
            else:
                # Woodbury
                A = torch.bmm(basis, basis.mT)
                A.diagonal(dim1=-2, dim2=-1).add_(self.reg_coeff)
                w = torch.bmm(basis.mT, torch.linalg.solve(A, y))
                for j in range(batch_sz):
                    weights[j, Lambdas_array[:,j,0], 0] = w[j, :, :].squeeze()
            
            # Compute the residual
            r = y - torch.bmm(dict, weights)
            
            i += 1
            norm_r = norm2(r)
            tolerance = (norm_r > self.r_thresh).any()
        return weights, Lambdas
    
    @property
    def reg_coeff(self) -> Tensor:
        return F.softplus(self._lambda)

  
