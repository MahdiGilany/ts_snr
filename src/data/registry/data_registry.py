
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
    train_series, val_series = create_dataset(dataset_name, use_scaler=False, **kwargs)
    series_noise = TimeSeries.from_times_and_values(
        train_series.time_index, np.random.randn(len(train_series))
        )
    train_series = train_series + (noise_mean + noise_std*series_noise)
    
    if use_scaler:
        scaler = Scaler()
        train_series_scaled = scaler.fit_transform(train_series)
        val_series_scaled = scaler.transform(val_series)
        return train_series_scaled, val_series_scaled, scaler
    
    return train_series, val_series
    
    

@register_dataset
def air_passengers(
    split_ratio: float = 0.75,
    use_scaler: bool = True,
    **kwargs
    ) -> Union[Tuple[TimeSeries, TimeSeries], Tuple[TimeSeries, TimeSeries, Scaler]]:
    
    from darts.datasets import AirPassengersDataset
    series_air = AirPassengersDataset().load().astype(np.float32)   
    train_air, val_air = series_air.split_before(split_ratio)
    
    if use_scaler:
        scaler = Scaler()
        train_air_scaled = scaler.fit_transform(train_air)
        val_air_scaled = scaler.transform(val_air)
        return train_air_scaled, val_air_scaled, scaler
    
    return train_air, val_air

