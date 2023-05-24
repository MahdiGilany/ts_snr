import torch
from torchmetrics import MetricCollection, Metric
from darts.metrics import mape
from typing import Any

class MAPEMetric(Metric):
    # def __init__(self, compute_on_step=True, dist_sync_on_step=False):
    #     super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step)
    #     self.add_state("mape", default=torch.tensor(0.0), dist_reduce_fx="sum")
    #     self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.add_state("sum_abs_per_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets."""
        sum_abs_per_error, num_obs = _mean_absolute_percentage_error_update(preds, target)

        self.sum_abs_per_error += sum_abs_per_error
        self.total += num_obs

    def compute(self) -> Tensor:
        """Computes mean absolute percentage error over state."""
        return _mean_absolute_percentage_error_compute(self.sum_abs_per_error, self.total)
