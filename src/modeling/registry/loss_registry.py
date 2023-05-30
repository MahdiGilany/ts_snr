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
def edl_loss(num_classes=2):
    from ..loss.evidential_loss import edl_mse_loss
    
    def wrap_loss(output, target, epoch_num, reduction='none'):
        if len(target.shape) == 1:
            target = F.one_hot(target, num_classes=num_classes)
        
        loss = edl_mse_loss(
            output,
            target,
            epoch_num,
            num_classes,
            50,
            reduction=reduction,
            device=output.device
            )
        return loss
    
    
    return wrap_loss    


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


# def get_loss_fn(loss_name: str,
#                 delta: Optional[float] = 1.0,
#                 beta: Optional[float] = 1.0) -> Callable:
#     return {'mse': F.mse_loss,
#             'mae': F.l1_loss,
#             'huber': partial(F.huber_loss, delta=delta),
#             'smooth_l1': partial(F.smooth_l1_loss, beta=beta)}[loss_name]