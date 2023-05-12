import contextlib

from ....modeling.loss.loss_functions import create_loss_fx
from ....modeling.optimizer_factory import configure_optimizers as config_optimizers
from ....modeling.registry.registry import create_model

from ..configure_optimizer_mixin import OptimizerConfig
from ..evaluation_base import EvaluationBase, SharedStepOutput
import torch
from torch.nn import functional as F

from hydra.utils import instantiate
from omegaconf import DictConfig

def get_remember_rate(
    current_epoch,
    max_epochs,
    final_remember_rate,
    final_remember_rate_epoch_frac: float,
):

    x_end = int(max_epochs * final_remember_rate_epoch_frac)

    if current_epoch > x_end:
        return final_remember_rate

    y_0 = 1
    y_end = final_remember_rate

    x = current_epoch

    b = y_0
    a = (y_end - b) / x_end

    y = a * x + b
    return y


class SupervisedCoteachingModel_old(EvaluationBase):
    def __init__(
        self,
        model1_name: str,
        model2_name: str,
        final_remember_rate,
        final_remember_rate_epoch_frac,
        batch_size: int,
        epochs: int = 100,
        learning_rate: float = 0.1,
    ):

        super().__init__(
            batch_size=batch_size,
            epochs=epochs,
            opt_cfg=OptimizerConfig(learning_rate=learning_rate),
        )

        self.final_remember_rate = final_remember_rate
        self.final_remember_rate_epoch_frac = final_remember_rate_epoch_frac

        self.model1 = create_model(model1_name)
        self.model2 = create_model(model2_name)

    def get_learnable_parameters(self):
        from itertools import chain

        return chain(self.model1.parameters(), self.model2.parameters())

    def shared_step(self, batch) -> SharedStepOutput:

        X, y, metadata = batch

        with contextlib.nullcontext() if self.training else torch.no_grad():
            logits1 = self.model1(X)
            logits2 = self.model2(X)

            loss1 = F.cross_entropy(logits1, y, reduce=False)
            loss2 = F.cross_entropy(logits2, y, reduce=False)

            r_t = self.get_remember_rate()
            total_samples = len(y)
            samples_to_remember = int(r_t * total_samples)

            _, ind_for_loss1 = torch.topk(loss2, samples_to_remember, largest=False)
            _, ind_for_loss2 = torch.topk(loss1, samples_to_remember, largest=False)

            loss_filter_1 = torch.zeros((loss1.size(0))).to(self.device)
            loss_filter_1[ind_for_loss1] = 1.0
            loss1 = (loss_filter_1 * loss1).sum()

            loss_filter_2 = torch.zeros((loss2.size(0))).to(self.device)
            loss_filter_2[ind_for_loss2] = 1.0
            loss2 = (loss_filter_2 * loss2).sum()

            loss = loss1 + loss2

        return SharedStepOutput(logits=logits1, y=y, loss=loss, metadata=[metadata])

    def get_remember_rate(self):

        return get_remember_rate(
            self.current_epoch,
            self.epochs,
            self.final_remember_rate,
            self.final_remember_rate_epoch_frac,
        )

    def on_epoch_end(self):
        self.log("remember_rate", self.get_remember_rate())

    def forward(self, X):
        return self.model1(X)


class SupervisedCoteachingModel(EvaluationBase):
    def __init__(
        self,
        model1: DictConfig,
        model2: DictConfig,
        loss_name: str,
        final_remember_rate,
        final_remember_rate_epoch_frac,
        batch_size: int = 32,
        epochs: int = 100,
        semi_sup: bool = False,
        num_classes: int = 2,
    ):

        super().__init__(
            batch_size=batch_size,
            epochs=epochs,
        )
        self.semi_sup = semi_sup
        
        self.final_remember_rate = final_remember_rate
        self.final_remember_rate_epoch_frac = final_remember_rate_epoch_frac

        self.model1 = instantiate(model1)
        self.model2 = instantiate(model2)
        self.loss1 = create_loss_fx(loss_name, num_classes=num_classes)
        self.loss2 = create_loss_fx(loss_name, num_classes=num_classes)
        
        self.automatic_optimization = False
    
    def get_learnable_parameters(self):
        parameters1 = [
            {"name": "net1",  "params": self.model1.parameters()},
        ]
        parameters2 = [
            {"name": "net2",  "params": self.model1.parameters()},
        ]        
        return parameters1, parameters2

    @property
    def learnable_parameters(self):
        return self.get_learnable_parameters()
    
    def get_remember_rate(self):

        return get_remember_rate(
            self.current_epoch,
            self.num_epochs,
            self.final_remember_rate,
            self.final_remember_rate_epoch_frac,
        )

    def shared_step(self, batch):
        X, y, *metadata = batch

        with contextlib.nullcontext() if self.training else torch.no_grad():
            logits1 = self.model1(X)
            logits2 = self.model2(X)

            # loss1 = F.cross_entropy(logits1, y, reduce=False)
            # loss2 = F.cross_entropy(logits2, y, reduce=False)
            loss1 = self.loss1(logits1, y, self.current_epoch)
            loss2 = self.loss2(logits2, y, self.current_epoch)
            
            r_t = self.get_remember_rate()
            total_samples = len(y)
            samples_to_remember = int(r_t * total_samples)

            _, ind_for_loss1 = torch.topk(loss2, samples_to_remember, largest=False, dim=0)
            _, ind_for_loss2 = torch.topk(loss1, samples_to_remember, largest=False, dim=0)

            loss_filter_1 = torch.zeros((loss1.size(0))).to(self.device)
            loss_filter_1[ind_for_loss1] = 1.0
            loss1 = (loss_filter_1 * loss1).sum()

            loss_filter_2 = torch.zeros((loss2.size(0))).to(self.device)
            loss_filter_2[ind_for_loss2] = 1.0
            loss2 = (loss_filter_2 * loss2).sum()
            
        return (loss1, loss2), logits1, y, *metadata
    
    def training_step(self, batch, batch_idx):
        if not self.semi_sup:
            self.model1._modules['0'].eval()
            self.model2._modules['0'].eval()
        
        losses, logits1, y, metadata = self.shared_step(batch)
        loss1, loss2 = losses
        
        
        #### optimizer step
        model1_optimizer, model2_optimizer = self.optimizers()
        model1_optimizer.zero_grad()
        model2_optimizer.zero_grad()
        self.manual_backward(loss1)
        self.manual_backward(loss2)
        model1_optimizer.step()
        model2_optimizer.step()
        
        #### scheduler step
        scheduler1, scheduler2 = self.lr_schedulers()
        scheduler1.step()
        scheduler2.step()
        
        #### log step
        self.train_acc(logits1.softmax(-1), y)

        self.log(
            "train/finetune_loss",
            loss1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/finetune_acc",
            self.train_acc.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        
        # return loss1, loss2

    def validation_step(self, batch, batch_idx, dataloader_idx: int):
        losses, logits1, y, *metadata = self.shared_step(batch)
        loss1, loss2 = losses       
        
        self.logging_combined_centers_loss(dataloader_idx, loss1)
        return loss1, logits1, y, *metadata
    
    def test_step(self, batch, batch_idx):
        losses, logits1, y, *metadata = self.shared_step(batch)
        loss1, loss2 = losses
        return loss1, logits1, y, *metadata
    
    def on_epoch_end(self):
        self.log("remember_rate", self.get_remember_rate())

    def forward(self, X):
        return self.model1(X)
    
    def configure_optimizers(self):
        model1_parameters, model2_parameters = self.get_learnable_parameters()
        optimizer1, scheduler1 = config_optimizers(
            model1_parameters,
            self._opt_cfg,
            self.num_training_steps,
            self.num_epochs,
        )
        
        optimizer2, scheduler2 = config_optimizers(
            model2_parameters,
            self._opt_cfg,
            self.num_training_steps,
            self.num_epochs,
        )
        return [optimizer1[0], optimizer2[0]], [scheduler1[0], scheduler2[0]]
    

class X(SupervisedCoteachingModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.setups()
        
    def setups(self):
        ...
        