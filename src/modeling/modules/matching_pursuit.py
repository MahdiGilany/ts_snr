'''Copied from https://github.com/salesforce/DeepTime/tree/main #TODO: add to this
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

def norm2(x):
    return torch.sqrt(torch.sum([i**2 for i in x]))


class _OrthogonalMatchingPursuit(nn.Module):
    def __init__(self, n_nonzero_coefs: Optional[int] = None):
        super().__init__()
        self.omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
        self.r_thresh = 0.01
        self.stop = np.inf
        self.n_nonzero_coefs = n_nonzero_coefs
    # def fit(self, dictionary: Tensor, y: Tensor):
    #     self.coef = []
    #     # for i in tqdm(range(dictionary.shape[0]), desc='OMP fitting'):
    #     for i in range(dictionary.shape[0]):
    #         self.omp.fit(dictionary[i,...].detach().cpu().numpy(), y[i,...].detach().cpu().numpy())
    #         coef = torch.tensor(self.omp.coef_).to(device=dictionary.device, dtype=dictionary.dtype)
    #         self.coef.append(coef.unsqueeze(-1))
    #     self.coef = torch.stack(self.coef, dim=0)
    #     return self.coef
    
    def fit(self, dictionary: Tensor, y: Tensor):
        return self.OMP(dictionary, y)
    
    def forward(self, dictionary: Tensor, coef: Tensor = None) -> Tensor:
        if coef is None:
            coef = self.coef
        return torch.bmm(dictionary, coef)
        
    def OMP(self, dictionary: Tensor, y: Tensor):
        '''Orthogonal Matching pursuit algorithm
        :args
        A: measurement matrix
        y: 
        '''
        r = y
        x_pre = torch.zeros(self.n_nonzero_coefs)
        Lambdas = []
        i = 0
        # Control stop interation with norm thresh or sparsity
        while norm2(r)>self.r_thresh and i<self.stop:
        
            # Compute the score of each atoms
            scores = dictionary.T.dot(r)
            
            # Select the atom with the max score
            Lambda = torch.argmax(abs(scores))
            # print(Lambda)
            Lambdas.append(Lambda)
            
            # All selected atoms form a basis
            basis = dictionary[:,Lambdas]

            # Least square solution for y=Ax
            x_pre[Lambdas] = torch.linalg.inv(torch.dot(basis.T,basis)).dot(basis.T).dot(y)
            
            # A = torch.bmm(basis.mT, basis)
            # A.diagonal(dim1=-2, dim2=-1).add_(reg_coeff)
            # B = torch.bmm(X.mT, Y)
            # weights = torch.linalg.solve(A, B)
        
            # if n_samples >= n_dim:
            #     # standard
            #     A = torch.bmm(X.mT, X)
            #     A.diagonal(dim1=-2, dim2=-1).add_(reg_coeff)
            #     B = torch.bmm(X.mT, Y)
            #     weights = torch.linalg.solve(A, B)
            # else:
            #     # Woodbury
            #     A = torch.bmm(X, X.mT)
            #     A.diagonal(dim1=-2, dim2=-1).add_(reg_coeff)
            #     weights = torch.bmm(X.mT, torch.linalg.solve(A, Y))
            
            # Compute the residual
            r = y - dictionary.dot(x_pre)
            
            i += 1
        return x_pre.T, Lambdas
    
  
