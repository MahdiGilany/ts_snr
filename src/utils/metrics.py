from typing import Callable, Optional, Sequence, Tuple, Union

import torch
from torchmetrics import MetricCollection, Metric
from darts import TimeSeries
from darts.metrics import mape, mse, mae, rmse, smape, ope, mase, r2_score
from darts.metrics.metrics import multivariate_support, multi_ts_support, _get_values_or_raise, _get_values
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


@multi_ts_support
@multivariate_support
def corr(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    reduction: Callable[[np.ndarray], float] = np.mean,
    inter_reduction: Callable[[np.ndarray], Union[float, np.ndarray]] = lambda x: x,
    n_jobs: int = 1,
    verbose: bool = False
) -> Union[float, np.ndarray]:
    
    y1, y2 = _get_values_or_raise(
        actual_series, pred_series, intersect, remove_nan_union=True
    )
    # we don't need to specify axis any where because we are using multivariate_support (this is slightly different than what is comonly implemented by others)
    u = ((y1 - y1.mean(0)) * (y2 - y2.mean(0))).sum(0)
    d = np.sqrt(((y1 - y1.mean(0)) ** 2 * (y2 - y2.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


@multi_ts_support
@multivariate_support
def rse(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    reduction: Callable[[np.ndarray], float] = np.mean,
    inter_reduction: Callable[[np.ndarray], Union[float, np.ndarray]] = lambda x: x,
    n_jobs: int = 1,
    verbose: bool = False
) -> Union[float, np.ndarray]:
    
    y1, y2 = _get_values_or_raise(
        actual_series, pred_series, intersect, remove_nan_union=True
    )
    
    return np.sqrt(np.sum((y1 - y2) ** 2)) / np.sqrt(np.sum((y1 - y1.mean()) ** 2))


def calculate_metrics(true, pred, only_mse=False, **kwargs):
    try:
        _mse = mse(true, pred, **kwargs)
    except:
        _mse = torch.tensor(np.nan)
        
    if only_mse:
        return {'mse': _mse}
        
    try:
        _mae =  mae(true, pred, **kwargs)
    except:
        _mae = torch.tensor(np.nan)
    
    try:
        _rmse = rmse(true, pred, **kwargs)
    except:
        _rmse = torch.tensor(np.nan)
    
    try:
        _mape = mape(true, pred, **kwargs)
    except:
        _mape = torch.tensor(np.nan)
    
    try:
        _smape = smape(true, pred, **kwargs)
    except:
        _smape = torch.tensor(np.nan)
        
    try: 
        _corr = corr(true, pred, **kwargs)
    except:
        _corr = torch.tensor(np.nan)
    
    try:
        _rse = rse(true, pred, **kwargs)
    except:
        _rse = torch.tensor(np.nan)
    
    # try:
    #     _ope = ope(true, pred, **kwargs)
    # except:
    #     _ope = torch.tensor(np.nan)
        
    # try:
    #     _mase = mase(true, pred, **kwargs)
    # except:
    #     _mase = torch.tensor(np.nan)
    
    try:
        _r2 = r2_score(true, pred, **kwargs)
    except:
        _r2 = torch.tensor(np.nan)
    
    return {'mae': _mae,
            'mse': _mse,
            'rmse': _rmse,
            'mape': _mape,
            'smape': _smape,
            'corr': _corr,
            'rse': _rse,
            # 'ope': _ope,
            # 'mase': _mase,
            'r2': _r2,
            }
    
    # return {'mae': mae(true, pred),
    #         'mse': mse(true, pred),
    #         'rmse': rmse(true, pred),
    #         'mape': mape(true, pred),
    #         'smape': smape(true, pred),
    #         'ope': ope(true, pred)}