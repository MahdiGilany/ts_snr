import torch
import torch.nn.functional as F

_LOSSES = {}


def register_loss(factory):
    _LOSSES[factory.__name__] = factory

    return factory


def create_loss_fx(loss_name, **kwargs):
    if loss_name not in _LOSSES:
        raise ValueError(f"Model <{loss_name}> not registered.")

    return _LOSSES[loss_name](**kwargs)


@register_loss
def crossentropy_loss(num_classes=2):
    
    def wrap_loss(output, target, epoch_num=0, reduction='none'):        
        loss = F.cross_entropy(
            output,
            target,
            reduction=reduction,
            )
        return loss
    
    
    return wrap_loss    


class TorchLosses(torch.nn.Module):
    def __init__(self, loss_name: 'mse',*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        if loss_name == 'mse':
            self.loss = torch.nn.MSELoss()
        elif loss_name == 'mae':
            self.loss = torch.nn.L1Loss()
        elif loss_name == 'smoothl1':
            self.loss = torch.nn.SmoothL1Loss()
            
    def forward(self, pred, target):
        return self.loss(pred, target)
    

# def get_loss_fn(loss_name: str,
#                 delta: Optional[float] = 1.0,
#                 beta: Optional[float] = 1.0) -> Callable:
#     return {'mse': F.mse_loss,
#             'mae': F.l1_loss,
#             'huber': partial(F.huber_loss, delta=delta),
#             'smooth_l1': partial(F.smooth_l1_loss, beta=beta)}[loss_name]