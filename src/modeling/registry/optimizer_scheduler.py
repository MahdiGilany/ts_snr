import math
from typing import Optional
from torch.optim.lr_scheduler import LRScheduler, LambdaLR
import wandb

def cosine_annealing_with_linear_warmup(T_max, eta_min, eta_max, warmup_epochs):
    def cosine_annealing_with_linear_warmup_fn(T_cur):
        if T_cur < warmup_epochs:
            return eta_min + (eta_max - eta_min) * T_cur / warmup_epochs
        else:
            return eta_min + 0.5 * (eta_max - eta_min) * (
                        1.0 + math.cos((T_cur - warmup_epochs) / (T_max - warmup_epochs) * math.pi))
    
    return cosine_annealing_with_linear_warmup_fn


class CosineAnnealingWithLinearWarmup(LambdaLR):
    def __init__(self, optimizer, T_max=100, eta_min=0, eta_max=1, warmup_epochs=0, last_epoch=-1, wandb_log=True):
        self.T_max = T_max
        self.eta_min = eta_min
        self.eta_max = eta_max # this should be one as it is inside LambdaLR since initial learning rate is already multiplied
        self.warmup_epochs = warmup_epochs
        self.wandb_log = wandb_log
        lr_lambda = cosine_annealing_with_linear_warmup(T_max, eta_min, eta_max, warmup_epochs)
        
        super().__init__(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)
    
    def step(self, *args, **kwargs) :
        super().step(*args, **kwargs)
        if self.wandb_log is not None:
            wandb.log({'lr': self.get_lr()[0]})    