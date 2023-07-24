'''Modified from https://github.com/salesforce/DeepTime/tree/main #TODO: add to this
'''

# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

from typing import Optional
import warnings

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
        stop: int = 256,
        r_thresh: float = 0.01,
        lambda_init: Optional[float] = -15.,
        n_nonzero_coefs: Optional[int] = None,
        ):
        super().__init__()
        
        self._lambda = torch.as_tensor(lambda_init, dtype=torch.float) # lambda is fixed and doesn't get updated during training

        self.n_nonzero_coefs = n_nonzero_coefs
        self.r_thresh = r_thresh
        # self.stop = stop + 1
        
    # def fit(self, dict: Tensor, y: Tensor):
    #     self.coef = []
        # self.omp = OrthogonalMatchingPursuit(n_nonzero_coefs=self.n_nonzero_coefs)
    #     # for i in tqdm(range(dict.shape[0]), desc='OMP fitting'):
    #     for i in range(dict.shape[0]):
    #         self.omp.fit(dict[i,...].detach().cpu().numpy(), y[i,...].detach().cpu().numpy())
    #         coef = torch.tensor(self.omp.coef_).to(device=dict.device, dtype=dict.dtype)
    #         self.coef.append(coef.unsqueeze(-1))
    #     self.coef = torch.stack(self.coef, dim=0)
    #     return self.coef
    
    def fit(self, dict: Tensor, y: Tensor):
        self.coef, _ = self.omp(dict, y)
        return self.coef
    
    def forward(self, X: Tensor, coef: Tensor = None) -> Tensor:
        if coef is None:
            coef = self.coef
        
        # adding bias term
        ones = torch.ones(X.shape[0], X.shape[1], 1, device=X.device)
        dict = torch.concat([X, ones], dim=-1)
        
        return torch.bmm(dict, coef)
        
    def omp(self, X: Tensor, y: Tensor):
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
        while tolerance and i<self.n_nonzero_coefs: 
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
        return weights, Lambdas_array
    
    @property
    def reg_coeff(self) -> Tensor:
        return F.softplus(self._lambda)



def batch_mm(matrix, matrix_batch, return_contiguous=True):
    """
    :param matrix: Sparse or dense matrix, size (m, n).
    :param matrix_batch: Batched dense matrices, size (b, n, k).
    :return: The batched matrix-matrix product, size (m, n) x (b, n, k) = (b, m, k).
    """
    # One dgemm is faster than many dgemv.
    # From https://github.com/pytorch/pytorch/issues/14489#issuecomment-607730242
    batch_size = matrix_batch.shape[0]
    # Stack the vector batch into columns. (b, n, k) -> (n, b, k) -> (n, b*k)
    vectors = matrix_batch.transpose([1, 0, 2]).reshape(matrix.shape[1], -1)

    # A matrix-matrix product is a batched matrix-vector product of the columns.
    # And then reverse the reshaping. (m, n) x (n, b*k) = (m, b*k) -> (m, b, k) -> (b, m, k)
    if return_contiguous:
        result = np.empty_like(matrix_batch, shape=(batch_size, matrix.shape[0], matrix_batch.shape[2]))
        np.matmul(matrix, vectors, out=result.transpose([1, 0, 2]).reshape(matrix.shape[0], -1))
    else:
        result = (matrix @ vectors).reshape(matrix.shape[0], batch_size, -1).transpose([1, 0, 2])

    return result

def innerp(x, y=None, out=None):
    if y is None:
        y = x
    if out is not None:
        out = out[:, None, None]  # Add space for two singleton dimensions.
    return torch.matmul(x[..., None, :], y[..., :, None], out=out)[..., 0, 0]

def cholesky_solve(ATA, ATy):
    # reg_coeff = torch.as_tensor(-5, dtype=ATA.dtype, device=ATA.device)
    # reg_coeff = F.softplus(reg_coeff)
    # ATA.diagonal(dim1=-2, dim2=-1).add_(reg_coeff)

    if ATA.dtype == torch.half or ATy.dtype == torch.half:
        return ATy.to(torch.float).cholesky_solve(torch.linalg.cholesky(ATA.to(torch.float))).to(ATy.dtype)
    return ATy.cholesky_solve(torch.linalg.cholesky(ATA)).to(ATy.dtype)

def linear_solve(ATA: torch.Tensor, ATy: torch.Tensor, reg_coeff=-15):
    if ATA.shape[0] < ATA.shape[1]:
        warnings.warn(f"not accurate if batch_size={ATA.shape[0]} > n_nonzero_coefs={ATA.shape[1]} is not hold.")
    # assert ATA.shape[0] > ATA.shape[1], f"not accurate if batch_size={ATA.shape[0]} > n_nonzero_coefs={ATA.shape[1]} is not hold."
    # standard
    reg_coeff = reg_coeff.to(dtype=ATA.dtype, device=ATA.device)
    reg_coeff = F.softplus(reg_coeff)
    ATA.diagonal(dim1=-2, dim2=-1).add_(reg_coeff)
    return torch.linalg.solve(ATA, ATy)

class OrthogonalMatchingPursuitParallel(nn.Module):
    def __init__(
        self, 
        n_nonzero_coefs,
        precompute=True,
        tol=0.0,
        normalize=False,
        fit_intercept=False,
        alg='naive', 
        ) -> None:
        super().__init__()
        self.n_nonzero_coefs = n_nonzero_coefs
        self.precompute = precompute
        self.tol = tol
        self.normalize = normalize
        self.fit_intercept = fit_intercept
        self.alg = alg
        
        self.reg_coeff = torch.tensor(-15.0)
    
    def fit(self, dict: Tensor, y: Tensor):
        dict = dict[0, ...] # (batch_sz, input_dim, n_atoms) -> (input_dim, n_atoms) since data over batch is repeated
        ones = torch.ones(dict.shape[0], 1, device=dict.device)
        X = torch.concat([dict, ones], dim=-1)
        coef = self.run_omp(
            X,
            y[:, :, 0],
            self.n_nonzero_coefs,
            self.precompute,
            self.tol,
            self.normalize,
            self.fit_intercept,
            self.alg
            ) # y is squeezed to match the shape needed in omp
        self.coef = coef.detach()
        return self.coef
    
    def forward(self, dict: Tensor, coef: Tensor = None) -> Tensor:
        if coef is None:
            coef = self.coef
        
        # adding bias term
        # dict --> (batch_sz, input_dim, n_atoms) data over batch is repeated
        ones = torch.ones(dict.shape[0], dict.shape[1], 1, device=dict.device)
        X = torch.concat([dict, ones], dim=-1)

        return torch.bmm(X, coef.unsqueeze(-1))
    
    def run_omp(self, X, y, n_nonzero_coefs, precompute=True, tol=0.0, normalize=False, fit_intercept=False, alg='naive'):
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X)
            y = torch.as_tensor(y)

        # We can either return sets, (sets, solutions), or xests
        # These are all equivalent, but are simply more and more dense representations.
        # Given sets and X and y one can (re-)construct xests. The second is just a sparse vector repr.

        # https://github.com/scikit-learn/scikit-learn/blob/15a949460dbf19e5e196b8ef48f9712b72a3b3c3/sklearn/linear_model/_omp.py#L690
        if fit_intercept or normalize:
            X = X.clone()
            assert not isinstance(precompute, torch.Tensor), "If user pre-computes XTX they can also pre-normalize X" \
                                                            " as well, so normalize and fit_intercept must be set false."

        if fit_intercept:
            X = X - X.mean(0)
            y = y - y.mean(1)[:, None]

        # To keep a good condition number on X, especially with Cholesky compared to LU factorization,
        # we should probably always normalize it (OMP is invariant anyways)
        if normalize is True:  # User can also just optionally supply pre-computed norms.
            normalize = (X * X).sum(0).sqrt()
            X /= normalize[None, :]

        if precompute is True or alg == 'v0':
            if precompute==False:
                precompute = None
            else:
                precompute = X.T @ X


        # If n_nonzero_coefs is equal to M, one should just return lstsq
        if alg == 'naive':
            sets, solutions, lengths = self.omp_naive(X, y, n_nonzero_coefs=n_nonzero_coefs, XTX=precompute, tol=tol)
        elif alg == 'v0':
            raise NotImplementedError
            # sets, solutions, lengths = omp_v0(X, y, n_nonzero_coefs=n_nonzero_coefs, XTX=precompute, tol=tol)


        solutions = solutions.squeeze(-1)
        if normalize is not False:
            solutions /= normalize[sets]

        xests = y.new_zeros(y.shape[0], X.shape[1])
        if lengths is None:
            xests[torch.arange(y.shape[0], dtype=sets.dtype, device=sets.device)[:, None], sets] = solutions
        else:
            for i in range(y.shape[0]):
                # xests[i].scatter_(-1, sets[i, :lengths[i]], solutions[i, :lengths[i]])
                xests[i, sets[i, :lengths[i]]] = solutions[i, :lengths[i]]

        return xests

    def omp_naive(self, X, y, n_nonzero_coefs, tol=None, XTX=None):
        """copied from https://github.com/ariellubonja/omp-parallel-gpu-python/tree/main

        Args:
            X (_type_): dictionary of shape (out/input_chunk_length, n_atoms)
            y (_type_): output of shape (batch, out/input_chunk_length)
            n_nonzero_coefs (_type_): number of non-zero coefficients
            tol (_type_, optional): _description_. Defaults to None.
            XTX (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        on_cpu = not (y.is_cuda or y.dtype == torch.half)
        # torch.cuda.synchronize()
        # Given X as an MxN array and y as an BxN array, do omp to approximately solve Xb=y

        # Base variables
        XT = X.contiguous().t()  # Store XT in fortran-order.
        y = y.contiguous()
        r = y.clone()

        sets = y.new_zeros((n_nonzero_coefs, y.shape[0]), dtype=torch.long).t()
        if tol:
            result_sets = sets.new_zeros(y.shape[0], n_nonzero_coefs)
            result_lengths = sets.new_zeros(y.shape[0])
            result_solutions = y.new_zeros((y.shape[0], n_nonzero_coefs, 1))
            original_indices = torch.arange(y.shape[0], dtype=sets.dtype, device=sets.device)

        # Trade b*k^2+bk+bkM = O(bkM) memory for much less compute time. (This has to be done anyways since we are batching,
        # otherwise one could just permute columns of X in-place as in https://github.com/scikit-learn/scikit-learn/blob/15a949460dbf19e5e196b8ef48f9712b72a3b3c3/sklearn/linear_model/_omp.py#L28 )
        ATs = y.new_zeros(r.shape[0], n_nonzero_coefs, X.shape[0])
        ATys = y.new_zeros(r.shape[0], n_nonzero_coefs, 1)
        ATAs = torch.eye(n_nonzero_coefs, dtype=y.dtype, device=y.device)[None].repeat(r.shape[0], 1, 1)
        if on_cpu:
            # For CPU it is faster to use a packed representation of the lower triangle in ATA.
            tri_idx = torch.tril_indices(n_nonzero_coefs, n_nonzero_coefs, device=sets.device, dtype=sets.dtype)
            ATAs = ATAs[:, tri_idx[0], tri_idx[1]]

        solutions = y.new_zeros((r.shape[0], 0))

        for k in range(n_nonzero_coefs+bool(tol)):
            # STOPPING CRITERIA
            if tol:
                problems_done = innerp(r) <= tol
                if k == n_nonzero_coefs:
                    problems_done[:] = True

                if problems_done.any():
                    remaining = ~problems_done

                    orig_idxs = original_indices[problems_done]
                    result_sets[orig_idxs, :k] = sets[problems_done, :k]
                    result_solutions[orig_idxs, :k] = solutions[problems_done]
                    result_lengths[orig_idxs] = k
                    original_indices = original_indices[remaining]

                    # original_indices = original_indices[remaining]
                    ATs = ATs[remaining]
                    ATys = ATys[remaining]
                    ATAs = ATAs[remaining]
                    sets = sets[remaining]
                    y = y[remaining]
                    r = r[remaining]
                    if problems_done.all():
                        return result_sets, result_solutions, result_lengths
            # GET PROJECTIONS AND INDICES TO ADD
            if on_cpu:
                projections = batch_mm(XT.numpy(), r[:, :, None].numpy())
                argmax_blast(projections.squeeze(-1), sets[:, k].numpy())
            else:
                projections = XT @ r[:, :, None]
                sets[:, k] = projections.abs().sum(-1).argmax(-1)  # Sum is just a squeeze, but would be relevant in SOMP.

            # UPDATE AT
            AT = ATs[:, :k + 1, :]
            updateA = XT[sets[:, k], :]
            AT[:, k, :] = updateA

            # UPDATE ATy based on AT
            ATy = ATys[:, :k + 1]
            ATy[:, k, 0] = innerp(updateA, y)

            # UPDATE ATA based on AT or precomputed XTX.
            if on_cpu:
                packed_idx = k * (k - 1) // 2
                if XTX is not None:  # Update based on precomputed XTX.
                    ATAs.t()[k + packed_idx:packed_idx + 2 * k + 1, :].t().numpy()[:] = XTX[sets[:, k, None], sets[:, :k + 1]]
                else:
                    np.matmul(AT[:, :k + 1, :].numpy(), updateA[:, :, None].numpy(),
                            out=ATAs.t()[k + packed_idx:packed_idx + 2 * k + 1, :].t()[:, :, None].numpy())
            else:
                ATA = ATAs[:, :k + 1, :k + 1]
                if XTX is not None:
                    ATA[:, k, :k + 1] = XTX[sets[:, k, None], sets[:, :k + 1]]
                else:
                    # Update ATAs by adding the new column of inner products.
                    ATA[:, k, :k + 1, None] = torch.bmm(AT[:, :k + 1, :], updateA[:, :, None])

            # SOLVE ATAx = ATy.
            if on_cpu:
                solutions = ATy.permute(0, 2, 1).clone().permute(0, 2, 1)  # Get a copy.
                ppsv(ATAs.t()[:packed_idx + 2 * k + 1, :].t().contiguous().numpy(), solutions.numpy())
            else:
                ATA[:, :k, k] = ATA[:, k, :k]  # Copy lower triangle to upper triangle.
                # solutions = cholesky_solve(ATA, ATy)
                solutions = linear_solve(ATA, ATy, self.reg_coeff)

            # FINALLY, GET NEW RESIDUAL r=y-Ax
            if on_cpu:
                np.subtract(y.numpy(), (AT.permute(0, 2, 1).numpy() @ solutions.numpy()).squeeze(-1), out=r.numpy())
            else:
                r[:, :, None] = y[:, :, None] - AT.permute(0, 2, 1) @ solutions

        return sets, solutions, None
    
    
    
class OrthogonalMatchingPursuitSecondVersion(nn.Module):
    def __init__(
        self,
        n_nonzero_coefs: int,
        tol: float = 0.001,
        lambda_init: Optional[float] = -5.,
        bias: bool = True,
        ):
        super().__init__()
        
        self._lambda = nn.Parameter(torch.as_tensor(lambda_init, dtype=torch.float), requires_grad=False) # lambda is fixed and doesn't get updated during training

        self.n_nonzero_coefs = n_nonzero_coefs
        self.tol = tol
        self.bias = bias
    
    def fit(self, dict: Tensor, y: Tensor):
        if self.bias:
            # bias added
            ones = torch.ones(dict.shape[0], dict.shape[1], 1, device=dict.device)
            dict = torch.concat([dict, ones], dim=-1)
            
        assert y.shape[-1] == 1, 'OMP only supports single output so far'
        
        self.coef, _, _ = self.omp(dict, y[:, :, 0]) # only one dimension is supported so far
        return self.coef
    
    def forward(self, dict: Tensor, coef: Tensor = None) -> Tensor:
        if coef is None:
            coef = self.coef
        
        if self.bias:
            # adding bias term
            ones = torch.ones(dict.shape[0], dict.shape[1], 1, device=dict.device)
            dict = torch.concat([dict, ones], dim=-1)

        # dict = dict/(F.relu(dict.norm(dim=1, keepdim=True)-1) + 1.0)
        return torch.bmm(dict, coef.unsqueeze(-1))
        
    def omp(self, X: Tensor, y: Tensor):
        '''Orthogonal Matching pursuit algorithm
        '''
        # X = X/(F.relu(X.norm(dim=1, keepdim=True)-1) + 1.0)
        dict = X[0, ...].detach().clone() # consider cloning the tensor
        # dict = dict/(F.relu(dict.norm(dim=0, keepdim=True)-1) + 1.0) # normalize the dictionary to have maximum unit norm
        
        chunk_length, n_atoms = dict.shape
        batch_sz, chunk_length = y.shape
        n_nonzero_coefs = self.n_nonzero_coefs
        
        # DTD = dict.T @ dict
        
        residuals = y.clone() # (batch_sz, chunk_length)
        max_score_indices = y.new_zeros((batch_sz, n_nonzero_coefs), dtype=torch.long) # (batch_sz, n_nonzero_coefs)
        
        # sparse weight matrix        
        selected_D = torch.zeros(batch_sz, chunk_length, n_nonzero_coefs).to(device=dict.device, dtype=dict.dtype) # (batch_sz, chunk_length, n_nonzero_coefs)
        selected_DTy = torch.zeros(batch_sz, n_nonzero_coefs, 1).to(device=dict.device, dtype=dict.dtype) # (batch_sz, n_nonzero_coefs, 1)
        selected_DTD = torch.eye(n_nonzero_coefs, dtype=dict.dtype, device=dict.device)[None].repeat(batch_sz, 1, 1)
        # selected_DTD = torch.zeros(batch_sz, n_nonzero_coefs, n_nonzero_coefs).to(device=dict.device, dtype=dict.dtype) # (batch_sz, n_nonzero_coefs, n_nonzero_coefs)
        
        # tolerance related 
        norm_y = y.detach().clone().norm(dim=(1)).mean()
        tolerance = True
        i = 0
        # Control stop interation with norm thresh or sparsity
        while tolerance and i<self.n_nonzero_coefs: 
        # for i in range(n_nonzero_coefs): 
            # Compute the score of each atoms
            projections = dict.T @ residuals[:, :, None] # (batch_sz, input_dim, 1)
            max_score_indices[:, i] = projections.abs().sum(-1).argmax(-1).detach() # Sum is just a squeeze, but would be relevant in SOMP.
            
            # update selected_D
            _selected_D = selected_D[:, :, :i + 1]
            curr_atoms = dict[:, max_score_indices[:, i]] # select the atom with the max score (chunk_length, batch_sz)
            _selected_D[:, :, i] = curr_atoms.T # an atom is added to sparse W per each datum in the batch at each iteration

            # update selected_DTy based on the current atom
            _selected_DTy = selected_DTy[:, :i + 1]
            _selected_DTy[:, i, 0] = torch.bmm(_selected_D[:, :, i:i+1].permute(0, 2, 1), y[:, :, None]).squeeze() # (batch_sz, 1, chunk_length) * (batch_sz, chunk_length, 1) -> (batch_sz, 1, 1) 

            
            # update selected_DTD based on selected_DT or precomputed XTX.
            _selected_DTD = selected_DTD[:, :i + 1, :i + 1]
            
            # Update selected_DTD by adding the new column of inner products.
            _selected_DTD[:, :, :] = torch.bmm(_selected_D[:, :, :i+1].permute(0, 2, 1), _selected_D[:, :, :i+1])
            # _selected_DTD[:, i, :i + 1] = DTD[max_score_indices[:, i, None], max_score_indices[:, :i + 1]]
            # _selected_DTD[:, i, :i + 1, None] = torch.bmm(_selected_D[:, :, :i+1].permute(0, 2, 1), curr_atoms.T[:, :, None])
            # _selected_DTD[:, :i, i] = _selected_DTD[:, i, :i] # Copy lower triangle to upper triangle.
            
            # solve selected_D @ solutions = y
            # solutions = cholesky_solve(_selected_DTD, _selected_DTy)
            # linear solve
            _selected_DTD.diagonal(dim1=-2, dim2=-1).add_(self.reg_coeff.detach()) # TODO: add multipath OMP
            solutions = torch.linalg.solve(_selected_DTD, _selected_DTy) # (batch_sz, n_nonzero_coefs, 1)

            # finally get residuals r=y-Wx
            residuals[:, :, None] = y[:, :, None] - _selected_D @ solutions
            
            # find tolerance and stopping criteria
            i += 1
            norm_res = residuals.norm(dim=(1)).mean()
            tolerance = (norm_res > self.tol*norm_y)
            if tolerance==False:
                comment='tolerance met'
        
        selected_atoms = X[
            torch.arange(batch_sz, dtype=max_score_indices.dtype, device=max_score_indices.device)[:, None, None],
            torch.arange(chunk_length, dtype=max_score_indices.dtype, device=max_score_indices.device)[None, :, None],
            max_score_indices[:, None, :]
            ]
        # selected_atoms_list = []
        # for j in range(batch_sz):
        #     selected_atoms_list.append(X[j, :, max_score_indices[j, :]])
        # selected_atoms = torch.stack(selected_atoms_list, dim=0)
        atomsTatoms = selected_atoms.permute(0, 2, 1) @ selected_atoms
        atomsTy = selected_atoms.permute(0, 2, 1) @ y[:, :, None]
        atomsTatoms.diagonal(dim1=-2, dim2=-1).add_(self.reg_coeff) # TODO: add multipath OMP
        coefs = torch.linalg.solve(atomsTatoms, atomsTy)
        W = torch.zeros(batch_sz, n_atoms, dtype=selected_atoms.dtype, device=selected_atoms.device)
        W[torch.arange(batch_sz, dtype=max_score_indices.dtype, device=max_score_indices.device)[:, None], max_score_indices] = coefs.squeeze()
        return W, max_score_indices, solutions
    
    @property
    def reg_coeff(self) -> Tensor:
        return F.softplus(self._lambda)


class _DifferentiableOrthogonalMatchingPursuit(OrthogonalMatchingPursuitSecondVersion):
    def __init__(self, *args, tau: float = 1e-3, hard=True, hard_mode=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau
        self.hard = hard
        self.hard_mode = hard_mode
        
    def omp(self, X: Tensor, y: Tensor):
        '''Orthogonal Matching pursuit algorithm
        '''
        dict = X[0, ...] # consider cloning the tensor
        
        chunk_length, n_atoms = dict.shape
        batch_sz, chunk_length = y.shape
        n_nonzero_coefs = self.n_nonzero_coefs
        tau = self.tau
        hard = self.hard
        hard_mode = self.hard_mode
        
        # DTD = dict.T @ dict

        residuals = y.clone().detach() # (batch_sz, chunk_length)
        residuals.requires_grad = True
        # max_score_indices = y.new_zeros((batch_sz, n_nonzero_coefs, n_atoms), dtype=X.dtype, device=X.device)
        max_score_indices = []
        sum_collector = torch.zeros((batch_sz, n_atoms), dtype=X.dtype, device=X.device)
        detached_indices = np.zeros((batch_sz, n_nonzero_coefs), dtype=np.int64) # (batch_sz, n_nonzero_coefs)


        tolerance = True
        # Control stop interation with norm thresh or sparsity
        for i in range(n_nonzero_coefs):
            # Compute the score of each atoms
            projections = dict.T @ residuals[:, :, None] # (batch_sz, n_atoms, 1)

            detached_indices[:, i] = projections.abs().squeeze(-1).argmax(-1).detach().cpu().numpy()
            soft_score_indices = (projections/tau).abs().squeeze(-1).softmax(-1) # (batch_sz, n_atoms)
            
            if hard:
                # copied and modified from https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax
                # Straight through.
                index = soft_score_indices.max(-1, keepdim=True)[1]
                hard_score_indices = torch.zeros_like(soft_score_indices).scatter_(-1, index, 1.0)
                ret = hard_score_indices - soft_score_indices.detach() + soft_score_indices   
                # max_score_indices[:, i] = ret
                if hard_mode == 0:
                    max_score_indices.append(ret[:, None, :]) # (batch_sz, i+1, n_atoms), it is list on dimension of i+1
                    
                    # update selected_D torch.cat(max_score_indices, dim=1)
                    ## first indexing then multiplication with max_score_indices
                    ind_0 = torch.arange(batch_sz)[:, None, None]
                    ind_1 = torch.arange(chunk_length)[None, :, None]
                    _selected_D = X[ind_0, ind_1, detached_indices[:, None, :i+1]] # (batch_sz, chunk_length, i+1)
                    ind_0 = torch.arange(batch_sz)[:, None]
                    ind_1 = torch.arange(i+1)[None, :]
                    _selected_max_score_indices = torch.cat(max_score_indices, dim=1)[ind_0, ind_1, detached_indices[:, :i+1]]
                    selected_D = _selected_D * _selected_max_score_indices[:, None, :] # (batch_sz, chunk_length, i+1)
                    ## first multiplication then indexing with max_score_indices 
                    # _selected_D = X[:, None, ...] * torch.cat(max_score_indices, dim=1)[:, :i+1, None, :] # max_score_indices[:, :i+1, None, :] # (batch_sz, i+1, chunk_length, n_atoms)
                    # _selected_D = _selected_D.permute(0, 2, 1, 3) # (batch_sz, chunk_length, i+1, n_atoms)
                    # ind_0 = torch.arange(batch_sz)[:, None, None]
                    # ind_1 = torch.arange(chunk_length)[None, :, None]
                    # ind_2 = torch.arange(i+1)[None, None, :]
                    # selected_D = _selected_D[ind_0, ind_1, ind_2, detached_indices[:, None, :i+1]] # (batch_sz, chunk_length, i+1)
                    
                    # calculate selected_DTy
                    selected_DTy = selected_D.permute(0, 2, 1) @ y[:, :, None] # (batch_sz, i+1, 1)

                    # calculate selected_DTD
                    selected_DTD = selected_D.permute(0, 2, 1) @ selected_D # (batch_sz, i+1, i+1)

                    # find W = (selected_DTD)^-1 @ selected_DTy
                    selected_DTD.diagonal(dim1=-2, dim2=-1).add_(self.reg_coeff) # TODO: add multipath OMP
                    nonzero_W = torch.linalg.solve(selected_DTD, selected_DTy) # (batch_sz, i+1, 1)

                    # finally get residuals r=y-Wx
                    residuals = y.detach() - (selected_D @ nonzero_W).squeeze(-1) # (batch_sz, chunk_length)
                else:
                    sum_collector = sum_collector + ret
                    selected_D = X * sum_collector[:, None, :] # (batch_sz, chunk_length, n_atoms)
                    
                    # calculate selected_DTy
                    selected_DTy = selected_D.permute(0, 2, 1) @ y[:, :, None] # (batch_sz, n_atoms, 1)

                    # calculate selected_DTD
                    selected_DTD = selected_D.permute(0, 2, 1) @ selected_D # (batch_sz, n_atoms, n_atoms)

                    # find W = (selected_DTD)^-1 @ selected_DTy
                    selected_DTD.diagonal(dim1=-2, dim2=-1).add_(self.reg_coeff) # TODO: add multipath OMP
                    nonzero_W = torch.linalg.solve(selected_DTD, selected_DTy) # (batch_sz, n_atoms, 1)

                    # finally get residuals r=y-Wx
                    residuals = y.detach() - (selected_D @ nonzero_W).squeeze(-1) # (batch_sz, chunk_length)
                    
                    
            else:
                # raise NotImplementedError
                sum_collector = sum_collector + soft_score_indices
                selected_D = X * sum_collector[:, None, :] # (batch_sz, chunk_length, n_atoms)
                
                # calculate selected_DTy
                selected_DTy = selected_D.permute(0, 2, 1) @ y[:, :, None] # (batch_sz, n_atoms, 1)

                # calculate selected_DTD
                selected_DTD = selected_D.permute(0, 2, 1) @ selected_D # (batch_sz, n_atoms, n_atoms)

                # find W = (selected_DTD)^-1 @ selected_DTy
                selected_DTD.diagonal(dim1=-2, dim2=-1).add_(self.reg_coeff) # TODO: add multipath OMP
                nonzero_W = torch.linalg.solve(selected_DTD, selected_DTy) # (batch_sz, n_atoms, 1)

                # finally get residuals r=y-Wx
                residuals = y.detach() - (selected_D @ nonzero_W).squeeze(-1) # (batch_sz, chunk_length)
        
        if hard:
            if hard_mode == 0:
                W = torch.zeros(batch_sz, n_atoms, dtype=selected_D.dtype, device=selected_D.device)
                W[torch.arange(batch_sz)[:, None], detached_indices] = nonzero_W.squeeze(-1)
                return W, max_score_indices, nonzero_W
            else:
                W = nonzero_W.squeeze(-1)
                return W, sum_collector, nonzero_W
        else:
            W = nonzero_W.squeeze(-1)
            return W, sum_collector, nonzero_W
        
    
class DifferentiableOrthogonalMatchingPursuit(nn.Module):
    def __init__(
        self,
        n_nonzero_coefs: int,
        tau: float = 1e-2,
        bias: bool = True,
        lambda_init = -5,
        tol: float = 0.001,
        hard: bool =True,
        **kwargs
        ):
        super().__init__()

        self._lambda = nn.Parameter(torch.as_tensor(lambda_init, dtype=torch.float), requires_grad=False) # lambda is fixed and doesn't get updated during training
        
        self.n_nonzero_coefs = n_nonzero_coefs
        self.tau = tau
        self.bias = bias
        self.tol = tol
        self.hard = hard
    
    @property
    def reg_coeff(self) -> Tensor:
        return F.softplus(self._lambda)

    def fit(self, dict: Tensor, y: Tensor):
        if self.bias:
            # bias added
            ones = torch.ones(dict.shape[0], dict.shape[1], 1, device=dict.device)
            dict = torch.concat([dict, ones], dim=-1)
            
        assert y.shape[-1] == 1, 'OMP only supports single output so far'
        
        self.coef, _, _ = self.omp(dict, y[:, :, 0]) # only one dimension is supported so far
        return self.coef
    
    def forward(self, dict: Tensor, coef: Tensor = None) -> Tensor:
        if coef is None:
            coef = self.coef
        
        if self.bias:
            # adding bias term
            ones = torch.ones(dict.shape[0], dict.shape[1], 1, device=dict.device)
            dict = torch.concat([dict, ones], dim=-1)
        
        # dict = dict/(F.relu(dict.norm(dim=1, keepdim=True)-1) + 1.0) # normalize the dictionary to have maximum unit norm
        return torch.bmm(dict, coef.unsqueeze(-1))
        
    def omp(self, X: Tensor, y: Tensor):
        '''Orthogonal Matching pursuit algorithm
        '''
        dict = X[0, ...] # consider cloning the tensor
        # dict = dict/(F.relu(dict.norm(dim=0, keepdim=True)-1) + 1.0) # normalize the dictionary to have maximum unit norm
        
        chunk_length, n_atoms = dict.shape
        batch_sz, chunk_length = y.shape
        n_nonzero_coefs = self.n_nonzero_coefs
        tau = self.tau
        hard = self.hard
        
        # DTD = dict.T @ dict

        residuals = y.clone().detach() # (batch_sz, chunk_length)
        residuals.requires_grad = True
        
        max_score_indices = []
        detached_indices = np.zeros((batch_sz, n_nonzero_coefs), dtype=np.int64) # (batch_sz, n_nonzero_coefs)
        # max_score_indices = y.new_zeros((batch_sz, n_nonzero_coefs, n_atoms), dtype=X.dtype, device=X.device)
        # sum_collector = torch.zeros((batch_sz, n_atoms), dtype=X.dtype, device=X.device)


        tolerance = True
        # Control stop interation with norm thresh or sparsity
        for i in range(n_nonzero_coefs):
            # Compute the score of each atoms
            projections = (dict.T)[None, ...] @ residuals[:, :, None] # (batch_sz, n_atoms, 1)
            soft_score_indices = (projections/tau).abs().squeeze(-1).softmax(-1) # (batch_sz, n_atoms)
            detached_indices[:, i] = projections.abs().squeeze(-1).argmax(-1).detach().cpu().numpy() # (batch_sz, ) for final W

            if hard:
                # copied and modified from https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax
                # Straight through.
                index = soft_score_indices.max(-1, keepdim=True)[1]
                hard_score_indices = torch.zeros_like(soft_score_indices).scatter_(-1, index, 1.0)
                ret = hard_score_indices - soft_score_indices.detach() + soft_score_indices # (batch_sz, n_atoms)  

                max_score_indices.append(ret[:, :, None]) # list[(batch_sz, n_atoms)] * i+1
                
                # update selected_D
                selected_D = dict[None, ...] @ torch.cat(max_score_indices, dim=2) # (batch_sz, chunk_length, i+1)
                
                # calculate selected_DTy
                selected_DTy = selected_D.permute(0, 2, 1) @ y[:, :, None] # (batch_sz, i+1, 1)

                # calculate selected_DTD
                selected_DTD = selected_D.permute(0, 2, 1) @ selected_D # (batch_sz, i+1, i+1)

                # find W = (selected_DTD)^-1 @ selected_DTy
                selected_DTD.diagonal(dim1=-2, dim2=-1).add_(self.reg_coeff) # TODO: add multipath OMP
                nonzero_W = torch.linalg.solve(selected_DTD, selected_DTy) # (batch_sz, i+1, 1)

                # finally get residuals r=y-Wx
                residuals = y.detach() - (selected_D @ nonzero_W).squeeze(-1) # (batch_sz, chunk_length)                    
                    
            else:
                max_score_indices.append(soft_score_indices[:, :, None]) # list[(batch_sz, n_atoms)] * i+1
                
                # update selected_D
                selected_D = dict[None, ...] @ torch.cat(max_score_indices, dim=2) # (batch_sz, chunk_length, i+1)
                
                # calculate selected_DTy
                selected_DTy = selected_D.permute(0, 2, 1) @ y[:, :, None] # (batch_sz, i+1, 1)

                # calculate selected_DTD
                selected_DTD = selected_D.permute(0, 2, 1) @ selected_D # (batch_sz, i+1, i+1)

                # find W = (selected_DTD)^-1 @ selected_DTy
                selected_DTD.diagonal(dim1=-2, dim2=-1).add_(self.reg_coeff) # TODO: add multipath OMP
                nonzero_W = torch.linalg.solve(selected_DTD, selected_DTy) # (batch_sz, i+1, 1)

                # finally get residuals r=y-Wx
                residuals = y.detach() - (selected_D @ nonzero_W).squeeze(-1) # (batch_sz, chunk_length)        
        

        W = torch.zeros(batch_sz, n_atoms, dtype=selected_D.dtype, device=selected_D.device)
        W[torch.arange(batch_sz)[:, None], detached_indices] = nonzero_W.squeeze(-1)
        return W, max_score_indices, nonzero_W
      
        
class DifferentiableMatchingPursuit(nn.Module):
    def __init__(
        self,
        n_nonzero_coefs: int,
        tau: float = 1e-2,
        bias: bool = True,
        lambda_init = -5,
        tol: float = 0.001,
        hard: bool =True,
        **kwargs
        ):
        super().__init__()

        self._lambda = nn.Parameter(torch.as_tensor(lambda_init, dtype=torch.float), requires_grad=False) # lambda is fixed and doesn't get updated during training
        
        self.n_nonzero_coefs = n_nonzero_coefs
        self.tau = tau
        self.bias = bias
        self.tol = tol
        self.hard = hard
    
    @property
    def reg_coeff(self) -> Tensor:
        return F.softplus(self._lambda)

    def fit(self, dict: Tensor, y: Tensor):
        if self.bias:
            # bias added
            ones = torch.ones(dict.shape[0], dict.shape[1], 1, device=dict.device)
            dict = torch.concat([dict, ones], dim=-1)
            
        assert y.shape[-1] == 1, 'OMP only supports single output so far'
        
        self.coef, _, _ = self.mp(dict, y[:, :, 0]) # only one dimension is supported so far
        return self.coef
    
    def forward(self, dict: Tensor, coef: Tensor = None) -> Tensor:
        if coef is None:
            coef = self.coef
        
        if self.bias:
            # adding bias term
            ones = torch.ones(dict.shape[0], dict.shape[1], 1, device=dict.device)
            dict = torch.concat([dict, ones], dim=-1)
        
        # dict = dict/(F.relu(dict.norm(dim=1, keepdim=True)-1) + 1.0) # normalize the dictionary to have maximum unit norm
        return torch.bmm(dict, coef.unsqueeze(-1))
        
    def mp(self, X: Tensor, y: Tensor):
        '''Orthogonal Matching pursuit algorithm
        '''
        dict = X[0, ...] # consider cloning the tensor
        # dict = dict/(F.relu(dict.norm(dim=0, keepdim=True)-1) + 1.0) # normalize the dictionary to have maximum unit norm
        
        chunk_length, n_atoms = dict.shape
        batch_sz, chunk_length = y.shape
        n_nonzero_coefs = self.n_nonzero_coefs
        tau = self.tau
        hard = self.hard
        
        # DTD = dict.T @ dict

        residuals = y.clone().detach() # (batch_sz, chunk_length)
        residuals.requires_grad = True
        
        nonzero_W_list = []
        detached_indices = np.zeros((batch_sz, n_nonzero_coefs), dtype=np.int64) # (batch_sz, n_nonzero_coefs)
        # max_score_indices = y.new_zeros((batch_sz, n_nonzero_coefs, n_atoms), dtype=X.dtype, device=X.device)
        # sum_collector = torch.zeros((batch_sz, n_atoms), dtype=X.dtype, device=X.device)


        tolerance = True
        # Control stop interation with norm thresh or sparsity
        for i in range(n_nonzero_coefs):
            # Compute the score of each atoms
            projections = (dict.T)[None, ...] @ residuals[:, :, None] # (batch_sz, n_atoms, 1)
            soft_score_indices = (projections/tau).abs().squeeze(-1).softmax(-1) # (batch_sz, n_atoms)
            detached_indices[:, i] = projections.abs().squeeze(-1).argmax(-1).detach().cpu().numpy() # (batch_sz, ) for final W

            if hard:
                # copied and modified from https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax
                # Straight through.
                index = soft_score_indices.max(-1, keepdim=True)[1]
                hard_score_indices = torch.zeros_like(soft_score_indices).scatter_(-1, index, 1.0)
                ret = hard_score_indices - soft_score_indices.detach() + soft_score_indices # (batch_sz, n_atoms)  
                
                # update selected_D
                selected_D = dict[None, ...] @ ret[:, :, None] # (batch_sz, chunk_length, 1)
                
                # calculate selected_DTy
                selected_DTy = selected_D.permute(0, 2, 1) @ y[:, :, None] # (batch_sz, 1, 1)

                # calculate selected_DTD
                selected_DTD = selected_D.permute(0, 2, 1) @ selected_D # (batch_sz, 1, 1)

                # find W = (selected_DTD)^-1 @ selected_DTy
                selected_DTD.diagonal(dim1=-2, dim2=-1).add_(self.reg_coeff) # TODO: add multipath OMP
                nonzero_W = torch.linalg.solve(selected_DTD, selected_DTy) # (batch_sz, 1, 1)
                nonzero_W_list.append(nonzero_W)

                # finally get residuals r=y-Wx
                residuals = y.detach() - (selected_D @ nonzero_W).squeeze(-1) # (batch_sz, chunk_length)                    
                    
            else:                
                # update selected_D
                selected_D = dict[None, ...] @ soft_score_indices[:, :, None] # (batch_sz, chunk_length, 1)
                
                # calculate selected_DTy
                selected_DTy = selected_D.permute(0, 2, 1) @ y[:, :, None] # (batch_sz, 1, 1)

                # calculate selected_DTD
                selected_DTD = selected_D.permute(0, 2, 1) @ selected_D # (batch_sz, 1, 1)

                # find W = (selected_DTD)^-1 @ selected_DTy
                selected_DTD.diagonal(dim1=-2, dim2=-1).add_(self.reg_coeff) # TODO: add multipath OMP
                nonzero_W = torch.linalg.solve(selected_DTD, selected_DTy) # (batch_sz, 1, 1)
                nonzero_W_list.append(nonzero_W)
                
                # finally get residuals r=y-Wx
                residuals = y.detach() - (selected_D @ nonzero_W).squeeze(-1) # (batch_sz, chunk_length)        
        

        W = torch.zeros(batch_sz, n_atoms, dtype=selected_D.dtype, device=selected_D.device)
        W[torch.arange(batch_sz)[:, None], detached_indices] = torch.cat(nonzero_W_list, dim=1).squeeze(-1)
        W.requires_grad = True
        return W, nonzero_W_list, nonzero_W
        
