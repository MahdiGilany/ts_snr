
import numpy as np
from typing import Union, Tuple
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from dataclasses import dataclass


_DATASETS = {}


def register_dataset(factory):
    _DATASETS[factory.__name__] = factory
    return factory


def create_dataset(dataset_name, **kwargs):
    if dataset_name not in _DATASETS:
        raise ValueError(f"Dataset <{dataset_name}> not registered.")
    
    return _DATASETS[dataset_name](**kwargs)


@dataclass
class DataSeries:
    train_series: TimeSeries = None
    val_series: TimeSeries = None
    test_series: TimeSeries = None
    test_series_noisy: TimeSeries = None
    backtest_series: TimeSeries = None
    scaler: Scaler = None


def create_noisy_dataset(
    dataset_name,
    noise_mean=0,
    noise_std=1,
    use_scaler: bool = True,
    **kwargs
    ):
    """Adds noise to a registered dataset.
    """
    # get series
    data_series: DataSeries = create_dataset(dataset_name, use_scaler=False, **kwargs)
    train_series = data_series.train_series
    val_series = data_series.val_series
    test_series = data_series.test_series
    
    # generate noise
    train_series_noise = TimeSeries.from_times_and_values(
        train_series.time_index, np.random.randn(len(train_series))
        )
    val_series_noise = TimeSeries.from_times_and_values(
        val_series.time_index, np.random.randn(len(val_series))
        )
    test_series_noise = TimeSeries.from_times_and_values(
        test_series.time_index, np.random.randn(len(test_series))
        )
    
    # add noise
    train_series_noisy = train_series + (noise_mean + noise_std*train_series_noise)
    val_series_noisy = val_series + (noise_mean + noise_std*val_series_noise)
    test_series_noisy = test_series + (noise_mean + noise_std*test_series_noise)
    
    # scale if needed
    if use_scaler:
        scaler = Scaler()
        train_series_noisy_scaled = scaler.fit_transform(train_series_noisy)
        val_series_noisy_scaled = scaler.transform(val_series_noisy)
        test_series_noisy_scaled = scaler.transform(test_series_noisy)
        test_series_scaled = scaler.transform(test_series)
        data_series_noisy_scaled = DataSeries(
            train_series=train_series_noisy_scaled,
            val_series=val_series_noisy_scaled,
            test_series=test_series_scaled,
            test_series_noisy=test_series_noisy_scaled,
            scaler=scaler
            )
        return data_series_noisy_scaled
    
    data_series_noisy = DataSeries(
        train_series=train_series_noisy,
        val_series=val_series_noisy,
        test_series=test_series,
        test_series_noisy=test_series_noisy,
        )  
    return data_series_noisy
    

def split_series(
    series: TimeSeries,
    split_ratio: Tuple[float] = (0.7, 0.1, 0.2)
    ):
    assert sum(split_ratio) == 1, "Split ratio must sum to 1"
    train_series, test_series = series.split_before(sum(split_ratio[0:2]))
    train_series, val_series = train_series.split_before(split_ratio[0]/sum(split_ratio[0:2]))
    data_series = DataSeries(
        train_series=train_series,
        val_series=val_series,
        test_series=test_series,
        )
    return data_series    


@register_dataset
def air_passengers(
    split_ratio: Tuple[float] = (0.7, 0.1, 0.2),
    use_scaler: bool = True,
    **kwargs
    ) -> Union[Tuple[TimeSeries, TimeSeries], Tuple[TimeSeries, TimeSeries, Scaler]]:
    
    from darts.datasets import AirPassengersDataset    
    series_air = AirPassengersDataset().load().astype(np.float32)   
    data_series = split_series(series_air, split_ratio)
    
    if use_scaler:
        scaler = Scaler()
        train_air_scaled = scaler.fit_transform(data_series.train_series)
        val_air_scaled = scaler.transform(data_series.val_series)
        test_air_scaled = scaler.transform(data_series.test_series)
        data_scaled_series = DataSeries(
            train_series=train_air_scaled,
            val_series=val_air_scaled,
            test_series=test_air_scaled,
            scaler=scaler
            )
        return data_scaled_series
    return data_series

