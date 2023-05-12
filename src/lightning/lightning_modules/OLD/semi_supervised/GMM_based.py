from typing import Any, Optional, Literal, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import wandb
import numpy as np
from tqdm import tqdm

#from pl_bolts.models.self_supervised.ssl_finetuner import SSLFineTuner
from contextlib import nullcontext
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torchmetrics.classification.accuracy import Accuracy

from pytorch_lightning import LightningModule
from torch.nn import Module

from warnings import warn
import torch_optimizer

from ..self_supervised.vicreg import VICRegConfig

from ..configure_optimizer_mixin import OptimizerConfig

from ..supervised.supervised_patch_model import SupervisedModel

from ..evaluation_base import EvaluationBase
from ....typing import FeatureExtractionProtocol
from dataclasses import dataclass

from omegaconf import DictConfig
from hydra.utils import instantiate
import logging

from einops import rearrange


from src.lightning.lightning_modules.evaluation_base import EvaluationBase 
from typing import Optional, List
from src.modeling.optimizer_factory import OptimizerConfig
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
from contextlib import nullcontext



class SelfSemiOneGMM(EvaluationBase):
    """
    This class implements One network self-supervised DivideMix, a semi-supervised learning method for prostate cancer detection.
    """

    def __init__(
        self,
        backbone1: DictConfig,
        checkpoint: Optional[str] = None,
        semi_sup: bool = False,
        batch_size: int = 32,
        epochs: int = 100,
        num_classes: int = 2,
        temperature: float = 0.1,
        beta_alpha: float = 0.75,
        GMM_prob_threshold: float = 0.5,
        GMM_cycle: int = 3,
        opt_cfg: OptimizerConfig = OptimizerConfig(),
    ):

        super().__init__(batch_size, epochs, opt_cfg=opt_cfg)

        self.semi_sup = semi_sup
        self.temperature = temperature
        self.beta_alpha = beta_alpha
        self.num_classes = num_classes
        self.GMM_prob_threshold = GMM_prob_threshold
        self.GMM_cycle = GMM_cycle
        
        logging.getLogger(__name__).info(
            f"Instantiating model {backbone1._target_} as warmed up backbone and head"
        )
        finetune1 = instantiate(backbone1)

        assert isinstance(
            finetune1.backbone, FeatureExtractionProtocol
        ), "Finetuned backbone model must support feature extraction"
        self.backbone1 = finetune1.backbone

        self.linear_layer1 = finetune1.linear_layer

        self._checkpoint_is_loaded = False

        if checkpoint is not None:
            self.load_from_pretraining_ckpt(checkpoint)

        self.train_acc = Accuracy()

        self.inferred_no_centers = 1

        self.train_lossHistory = []
        self.val_macroLoss_all_centers = []
        self.test_macroLoss_all_centers = []
        
        self.log_kwargs = {
            "on_step": False,
            "on_epoch": True,
            "sync_dist": True,
            "add_dataloader_idx": False,
        }

        self.save_hyperparameters()

    def load_from_pretraining_ckpt(self, ckpt: str):
        if isinstance(self.backbone1, LightningModule):
            self.backbone1.load_from_checkpoint(ckpt)

        self._checkpoint_is_loaded = True

    @property
    def learnable_parameters(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """
        parameters = [
            {"name": "linear_layer1",  "params": self.linear_layer1.parameters()},
        ]
        if self.semi_sup:
            parameters.append(
                {"name": "backbone1", "params": self.backbone1.parameters()}
            )

        return parameters

    def co_divide_GMM(self, net1: nn.Module, dataloader: data.DataLoader):
        """implements divide mix GMM algorithm using two networks and a dataset"""
        
        from sklearn.mixture import GaussianMixture
        net1.eval()

        losses1 = []
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(dataloader), desc="GMM on training sets", total=len(dataloader)):
                # if batch_idx == 5:
                #     break
                
                (X1, X2), y, metadata = batch
                X1 = X1.to(self.device)
                # X2 = X2.to(self.device)
                y = y.to(self.device)

                # todo augmented data is passed to the network
                logits1, _ = self._forward(X1)
                
                loss1 = F.cross_entropy(logits1, y, reduction='none')
                losses1.append(loss1.detach())
        
        losses1 = torch.cat(losses1).reshape(-1,1).cpu().numpy()
        losses1 = (losses1 - losses1.min())/(losses1.max() - losses1.min())
        
        self.train_lossHistory.append(losses1)
                
        if len(self.train_lossHistory) > 5:
            self.train_lossHistory = self.train_lossHistory[-5:]
            input_loss1 = np.mean(self.train_lossHistory[-5:], axis=0).reshape(-1,1)
        else:
            input_loss1 = np.mean(self.train_lossHistory, axis=0).reshape(-1,1)
        
    
        GMM1 = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    
        GMM1.fit(input_loss1)
        
        prob1 = GMM1.predict_proba(input_loss1) 
        prob1 = prob1[:,GMM1.means_.argmin()] 
        
        labeled_train_idx1 = np.where(prob1 > self.GMM_prob_threshold)[0]
        unlabeled_train_idx1 = np.where(prob1 <= self.GMM_prob_threshold)[0] 
        
        # permute labeled_train_idx1
        labeled_train_idx1 = np.random.permutation(labeled_train_idx1)
        unlabeled_train_idx1 = np.random.permutation(unlabeled_train_idx1)
        
        prob1 = torch.tensor(prob1, device=self.device).view(-1,1)

        # plot histogram of GMM losses
        wandb.log({"GMM_losses_avg": wandb.Histogram(input_loss1.squeeze()), "step": self.trainer.global_step, })
        wandb.log({"GMM_losses": wandb.Histogram(losses1.squeeze()), "step": self.trainer.global_step})
        
        assert len(labeled_train_idx1) + len(unlabeled_train_idx1) == len(prob1)
        assert len(labeled_train_idx1) > 0
        assert len(unlabeled_train_idx1) > 0
        
        return labeled_train_idx1, unlabeled_train_idx1, prob1
    
    def on_train_epoch_start(self) -> None:
        """
        Changing model to eval() mode has to happen at the start of every epoch,
        and should only happen if we are not in semi-supervised mode
        """
        if not self.semi_sup:
            self.backbone1.eval()
        
        # every n epochs, co-divide the dataset to labeled and unlabeled sets
        if self.trainer.current_epoch%self.GMM_cycle == 0:
            self.trainer.datamodule.labeled_train_idx1, self.trainer.datamodule.unlabeled_train_idx1,\
            self.GMM_labeled_probs1 = self.co_divide_GMM(
                    nn.ModuleList([self.backbone1, self.linear_layer1]),
                    self.trainer.datamodule.train_GMM_loader
                )

    def shared_step(self, batch):
        x, y, metadata = batch
        logits1, feats1 = self._forward(x)        
        loss1 = F.cross_entropy(logits1, y)

        return loss1, logits1, y, metadata

    def get_feats(self, x):
        return self.backbone1.get_features(x)

    def _forward(self, x):
        with nullcontext() if self.semi_sup else torch.no_grad():
            feats1 = self.get_feats(x)
            feats1 = feats1.view(feats1.size(0), -1)
            
        logits1 = self.linear_layer1(feats1)
        return logits1, feats1
        
    def training_step(self, batch, batch_idx):
        labeled_batch = batch["labeled"]
        unlabeled_batch = batch["unlabeled"]
        
        (X1, X2), labeled_labels, metadata = labeled_batch
        (X1_unlabeled, X2_unlabeled), unlabeled_labels, metadata = unlabeled_batch
        batch_size = X1.size(0)
        
        # todo remove after adding the code to the pipeline
        X1, X2, X1_unlabeled, X2_unlabeled = X1.to(self.device), X2.to(self.device), X1_unlabeled.to(self.device), X2_unlabeled.to(self.device)
        labeled_labels = labeled_labels.to(self.device)
        labeled_labels = F.one_hot(labeled_labels, num_classes=self.num_classes).float()
        
        start_indx = batch_idx*batch_size
        end_indx = (batch_idx+1)*batch_size
        w_x = self.GMM_labeled_probs1[start_indx:end_indx].view(-1,1)

        with torch.no_grad():
            # get unlablled data pseudo labels
            logits1_unlabeled_model1, _ = self._forward(X1_unlabeled)
            logits2_unlabeled_model1, _ = self._forward(X2_unlabeled)
        
            unlabeled_avg_probs = (torch.softmax(logits1_unlabeled_model1, dim=1) + 
                                    torch.softmax(logits2_unlabeled_model1, dim=1)) / 2
            unlabeled_pseudo_targets = unlabeled_avg_probs**(1/self.temperature)
            unlabeled_pseudo_targets = unlabeled_pseudo_targets / unlabeled_pseudo_targets.sum(dim=1, keepdim=True) # normalize
            unlabeled_pseudo_targets = unlabeled_pseudo_targets.detach()    
        
            # get lablled data pseudo labels
            logits1_labeled_model1, _ = self._forward(X1)
            logits2_labeled_model1, _ = self._forward(X2)
        
        labeled_avg_probs = (torch.softmax(logits1_labeled_model1, dim=1) + 
                                 torch.softmax(logits2_labeled_model1, dim=1)) / 2
        # finding labels for labeled data using GMM1
        labeled_avg_probs1 = w_x*labeled_labels + (1-w_x)*labeled_avg_probs              
        labeled_pseudo_targets1 = labeled_avg_probs1**(1/self.temperature) # temparature sharpening 
        labeled_pseudo_targets1 = labeled_pseudo_targets1 / labeled_pseudo_targets1.sum(dim=1, keepdim=True) # normalize           
        labeled_pseudo_targets1 = labeled_pseudo_targets1.detach()       
        
        
        # mixmatch
        _lambda = np.random.beta(self.beta_alpha, self.beta_alpha)        
        _lambda = max(_lambda, 1-_lambda)
                
        all_inputs = torch.cat([X1, X2, X1_unlabeled, X2_unlabeled], dim=0)
        all_targets1 = torch.cat([labeled_pseudo_targets1, labeled_pseudo_targets1, unlabeled_pseudo_targets, unlabeled_pseudo_targets], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a1, target_b1 = all_targets1, all_targets1[idx]
        
        mixed_input = _lambda * input_a + (1 - _lambda) * input_b        
        mixed_target1 = _lambda * target_a1 + (1 - _lambda) * target_b1
                
        
        logits_mixed_model1, _ = self._forward(mixed_input)
        logits_labeled1 = logits_mixed_model1[:batch_size*2]
        logits_unlabeled1 = logits_mixed_model1[batch_size*2:]        
        
        
        loss_labeled1, loss_unlabeled1, lambda_unlabeled1 = self.semi_loss(logits_labeled1, mixed_target1[:batch_size*2], 
                                                                           logits_unlabeled1, mixed_target1[batch_size*2:], 
                                                                           self.trainer.current_epoch+batch_idx/self.num_training_steps
                                                                           )
        
        # regularization
        prior = torch.ones(self.num_classes)/self.num_classes
        prior = prior.to(self.device)      
          
        pred_mean1 = torch.softmax(logits_mixed_model1, dim=1).mean(0)
        penalty1 = torch.sum(prior*torch.log(prior/pred_mean1))

        # loss
        loss = loss_labeled1 + lambda_unlabeled1 * loss_unlabeled1  + penalty1
        
        self.log(
            "train/divide_loss",
            loss,
            **self.log_kwargs)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx: int):
        loss, logits, y, *metadata = self.shared_step(batch)

        self.logging_combined_centers_loss(dataloader_idx, loss)

        return loss, logits, y, *metadata

    def validation_epoch_end(self, outs):        
        kwargs = {
            "on_step": False,
            "on_epoch": True,
            "sync_dist": True,
            "add_dataloader_idx": False,
        }
        self.log(
            "val/finetune_loss",
            torch.mean(torch.tensor(self.val_macroLoss_all_centers)),
            prog_bar=True,
            **kwargs,
        )
        self.log(
            "test/finetune_loss",
            torch.mean(torch.tensor(self.test_macroLoss_all_centers)),
            prog_bar=True,
            **kwargs,
        )

    def test_step(self, batch, batch_idx):
        loss, logits, y, *metadata = self.shared_step(batch)
        return loss, logits, y, *metadata

    def on_epoch_end(self):
        self.train_acc.reset()

        self.val_macroLoss_all_centers = []
        self.test_macroLoss_all_centers = []
        
    def logging_combined_centers_loss(self, dataloader_idx, loss):
        """macro loss for centers"""
        self.inferred_no_centers = (
            dataloader_idx + 1
            if dataloader_idx + 1 > self.inferred_no_centers
            else self.inferred_no_centers
        )

        if dataloader_idx < self.inferred_no_centers / 2.0:
            self.val_macroLoss_all_centers.append(loss)
        else:
            self.test_macroLoss_all_centers.append(loss)

    def semi_loss(self, outputs_labeled, targets_labeled, outputs_unlabeled, targets_unlabeled, epoch, warm_up=0):
        probs_u = torch.softmax(outputs_unlabeled, dim=1)

        loss_labeled = -torch.mean(torch.sum(F.log_softmax(outputs_labeled, dim=1) * targets_labeled, dim=1))
        loss_unlabeled = torch.mean((probs_u - targets_unlabeled)**2)

        def linear_rampup(current, warm_up=0, rampup_length=16):
            current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
            return 25.0*float(current)

        return loss_labeled, loss_unlabeled, linear_rampup(epoch, warm_up)
    

class SelfSemiOneGMMperClass(SelfSemiOneGMM):
    def co_divide_GMM(self, net1: nn.Module, dataloader: data.DataLoader):
        """implements divide mix GMM algorithm using two networks and a dataset"""
        
        from sklearn.mixture import GaussianMixture
        net1.eval()

        losses = []
        targets = []
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(dataloader), desc="GMM on training sets", total=len(dataloader)):
                # if batch_idx == 5:
                #     break
                
                (X1, X2), y, metadata = batch
                X1 = X1.to(self.device)
                # X2 = X2.to(self.device)
                y = y.to(self.device)
                targets.append(y)

                # todo augmented data is passed to the network
                logits, _ = self._forward(X1)
                
                loss = F.cross_entropy(logits, y, reduction='none')
                losses.append(loss.detach())
        
        targets = torch.cat(targets).reshape(-1).cpu().numpy()        
        losses = torch.cat(losses).reshape(-1,1).cpu().numpy()
        losses = (losses - losses.min())/(losses.max() - losses.min())
        
                
        self.train_lossHistory.append(losses)
                
        if len(self.train_lossHistory) > 5:
            self.train_lossHistory = self.train_lossHistory[-5:]
            input_loss = np.mean(self.train_lossHistory[-5:], axis=0).reshape(-1,1)
        else:
            input_loss = np.mean(self.train_lossHistory, axis=0).reshape(-1,1)

        
        # fit GMM to benign
        input_loss_benign = input_loss[targets == 0]
        GMM_benign = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    
        GMM_benign.fit(input_loss_benign)
        
        prob_benign = GMM_benign.predict_proba(input_loss_benign) 
        prob_benign = prob_benign[:,GMM_benign.means_.argmin()] 
        
                
        # fit GMM to cancer
        input_loss_cancer = input_loss[targets == 1]
        GMM_cancer = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
        
        GMM_cancer.fit(input_loss_cancer)
        
        prob_cancer = GMM_cancer.predict_proba(input_loss_cancer)
        prob_cancer = prob_cancer[:,GMM_cancer.means_.argmin()]
        
        probs = []
        j, k = 0, 0
        for i, t in enumerate(targets):
            if t == 0:
                probs.append(prob_benign[j])
                j += 1
            else:
                probs.append(prob_cancer[k])
                k += 1
        # breakpoint()
        probs = torch.tensor(probs).reshape(-1)        
        
        # find labeled and unlabeled indices
        labeled_train_idx = np.where(probs > self.GMM_prob_threshold)[0]
        unlabeled_train_idx = np.where(probs <= self.GMM_prob_threshold)[0]
        
        # permute labeled_train_idx
        labeled_train_idx = np.random.permutation(labeled_train_idx)
        unlabeled_train_idx = np.random.permutation(unlabeled_train_idx)

        probs = torch.tensor(probs, device=self.device).view(-1,1)

        # plot histogram of GMM losses
        wandb.log({"GMM_losses_avg": wandb.Histogram(input_loss.squeeze()), "step": self.trainer.global_step, })
        wandb.log({"GMM_losses": wandb.Histogram(losses.squeeze()), "step": self.trainer.global_step})
        
        
        # plot histogram of GMM losses for benign
        wandb.log({"GMM_losses_benign_avg": wandb.Histogram(input_loss_benign.squeeze()), "step": self.trainer.global_step, })
        wandb.log({"GMM_losses_benign": wandb.Histogram(losses[targets == 0].squeeze()), "step": self.trainer.global_step})
        
        
        # plot histogram of GMM losses for cancer
        wandb.log({"GMM_losses_cancer_avg": wandb.Histogram(input_loss_cancer.squeeze()), "step": self.trainer.global_step, })
        wandb.log({"GMM_losses_cancer": wandb.Histogram(losses[targets == 1].squeeze()), "step": self.trainer.global_step})
        
        self.log("GMM_means_benign", GMM_benign.means_.argmin(), self.log_kwargs)
        self.log("GMM_means_cancer", GMM_cancer.means_.argmin(), self.log_kwargs)
        return labeled_train_idx, unlabeled_train_idx, probs
    
