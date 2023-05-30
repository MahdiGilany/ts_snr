
import numpy as np
from typing import Union, Tuple
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler



_DATASETS = {}


def register_dataset(factory):
    _DATASETS[factory.__name__] = factory
    return factory


def create_dataset(dataset_name, **kwargs):
    if dataset_name not in _DATASETS:
        raise ValueError(f"Dataset <{dataset_name}> not registered.")
    
    return _DATASETS[dataset_name](**kwargs)


def create_noisy_dataset(
    dataset_name,
    noise_mean=0,
    noise_std=1,
    use_scaler: bool = True,
    **kwargs
    ):
    # get series
    train_series, val_series, test_series = create_dataset(dataset_name, use_scaler=False, **kwargs)
    
    # generate noise
    train_series_noise = TimeSeries.from_times_and_values(
        train_series.time_index, np.random.randn(len(train_series))
        )
    val_series_noise = TimeSeries.from_times_and_values(
    val_series.time_index, np.random.randn(len(val_series))
    )
    
    # add noise
    train_noisy_series = train_series + (noise_mean + noise_std*train_series_noise)
    val_noisy_series = val_series + (noise_mean + noise_std*val_series_noise)
    
    # scale if needed
    if use_scaler:
        scaler = Scaler()
        train_noisy_series_scaled = scaler.fit_transform(train_noisy_series)
        val_noisy_series_scaled = scaler.transform(val_noisy_series)
        test_series_scaled = scaler.transform(test_series)
        return train_noisy_series_scaled, val_noisy_series_scaled, test_series_scaled, scaler
    
    return train_noisy_series, val_noisy_series, test_series
    

def split_series(
    series: TimeSeries,
    split_ratio: Tuple[float] = (0.7, 0.1, 0.2)
    ):
    assert sum(split_ratio) == 1, "Split ratio must sum to 1"
    train_series, test_series = series.split_before(sum(split_ratio[0:2]))
    train_series, val_series = train_series.split_before(split_ratio[0]/sum(split_ratio[0:2]))
    return train_series, val_series, test_series    


@register_dataset
def air_passengers(
    split_ratio: Tuple[float] = (0.7, 0.1, 0.2),
    use_scaler: bool = True,
    **kwargs
    ) -> Union[Tuple[TimeSeries, TimeSeries], Tuple[TimeSeries, TimeSeries, Scaler]]:
    
    from darts.datasets import AirPassengersDataset    
    series_air = AirPassengersDataset().load().astype(np.float32)   
    train_air, val_air, test_air = split_series(series_air, split_ratio)
    
    if use_scaler:
        scaler = Scaler()
        train_air_scaled = scaler.fit_transform(train_air)
        val_air_scaled = scaler.transform(val_air)
        test_air_scaled = scaler.transform(test_air)
        return train_air_scaled, val_air_scaled, test_air_scaled, scaler
    
    return train_air, val_air, test_air

