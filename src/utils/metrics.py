import torch
from torchmetrics import MetricCollection, Metric
from darts.metrics import mape, mse, mae, rmse, smape, ope, mase
from typing import Any
import numpy as np

# class MAPEMetric(Metric):
#     # def __init__(self, compute_on_step=True, dist_sync_on_step=False):
#     #     super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step)
#     #     self.add_state("mape", default=torch.tensor(0.0), dist_reduce_fx="sum")
#     #     self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
#     def __init__(
#         self,
#         **kwargs: Any,
#     ) -> None:
#         super().__init__(**kwargs)

#         self.add_state("sum_abs_per_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
#         self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

#     def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
#         """Update state with predictions and targets."""
#         sum_abs_per_error, num_obs = _mean_absolute_percentage_error_update(preds, target)

#         self.sum_abs_per_error += sum_abs_per_error
#         self.total += num_obs

#     def compute(self) -> Tensor:
#         """Computes mean absolute percentage error over state."""
#         return _mean_absolute_percentage_error_compute(self.sum_abs_per_error, self.total)


def calculate_metrics(true, pred):
    try:
        _mae =  mae(true, pred)
    except:
        _mae = torch.tensor(np.nan)
    
    try:
        _mse = mse(true, pred)
    except:
        _mse = torch.tensor(np.nan)
    
    try:
        _rmse = rmse(true, pred)
    except:
        _rmse = torch.tensor(np.nan)
    
    try:
        _mape = mape(true, pred)
    except:
        _mape = torch.tensor(np.nan)
    
    try:
        _smape = smape(true, pred)
    except:
        _smape = torch.tensor(np.nan)
    
    try:
        _ope = ope(true, pred)
    except:
        _ope = torch.tensor(np.nan)
        
    try:
        _mase = mase(true, pred)
    except:
        _mase = torch.tensor(np.nan)
    
        
    
    return {'mae': _mae,
            'mse': _mse,
            'rmse': _rmse,
            'mape': _mape,
            'smape': _smape,
            'ope': _ope,
            'mase': _mase}
    
    # return {'mae': mae(true, pred),
    #         'mse': mse(true, pred),
    #         'rmse': rmse(true, pred),
    #         'mape': mape(true, pred),
    #         'smape': smape(true, pred),
    #         'ope': ope(true, pred)}