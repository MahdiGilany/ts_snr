import os
import numpy as np
from typing import Union, Tuple, Literal
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from dataclasses import dataclass
from sklearn import preprocessing

from darts.datasets.dataset_loaders import DatasetLoaderCSV
from darts import concatenate
from darts.datasets import (
    AirPassengersDataset,
    ETTh1Dataset,
    ETTh2Dataset,
    ETTm1Dataset,
    ETTm2Dataset,
    ElectricityDataset,
    ExchangeRateDataset,
    TrafficDataset,
    WeatherDataset,
    EnergyDataset,
    UberTLCDataset,
)

DATA_ROOT = os.environ.get("DATA_ROOT")
if DATA_ROOT is None:
    raise Warning("Environment variable DATA_ROOT is not set")


def register_dataset(factory):
    _DATASETS[factory.__name__] = factory
    return factory


def create_dataset(dataset_name, **kwargs):
    if dataset_name not in _DATASETS:
        raise ValueError(f"Dataset <{dataset_name}> not registered.")
    
    return _DATASETS[dataset_name](**kwargs)


def create_noisy_dataset(
    dataset_name,
    noise_type: Literal["gaussian", "laplace"] = "gaussian",
    noise_mean: float = 0,
    noise_std: float = 1,
    use_scaler: bool = True,
    **kwargs
    ):
    """Adds noise to a registered dataset.
    """
    # get series
    data_scaled_series: DataSeries = create_dataset(dataset_name, use_scaler=use_scaler, **kwargs)
    train_series = data_scaled_series.train_series
    val_series = data_scaled_series.val_series
    test_series = data_scaled_series.test_series
    scaler = data_scaled_series.scaler
    
    random_function = np.random.normal if noise_type=="gaussian" else np.random.laplace
    noise_scale = noise_std if noise_type=="gaussian" else noise_std/np.sqrt(2)
    
    # generate noise
    train_series_noise = TimeSeries.from_times_and_values(
        train_series.time_index, random_function(loc=noise_mean, scale=noise_scale, size=train_series._xa.shape) # TODO: not a good way to access values
        )
    val_series_noise = TimeSeries.from_times_and_values(
        val_series.time_index, random_function(loc=noise_mean, scale=noise_scale, size=val_series._xa.shape)
        )
    test_series_noise = TimeSeries.from_times_and_values(
        test_series.time_index, random_function(loc=noise_mean, scale=noise_scale, size=test_series._xa.shape)
        )
    
    # noise_std = noise_std * np.std(train_series._xa, axis=0).values # std is relative to std of train series for each dim separately

    # add noise
    train_series_noisy = train_series + train_series_noise._xa.values
    val_series_noisy = val_series + val_series_noise._xa.values
    test_series_noisy = test_series + test_series_noise._xa.values
    
    # scale if needed
    if use_scaler:
        data_series_noisy_scaled = DataSeries(
            train_series=train_series_noisy,
            val_series=val_series_noisy,
            test_series=test_series,
            test_series_noisy=test_series_noisy,
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
    split_ratio: Tuple[float] = (0.7, 0.1, 0.2),
    ):
    # assert sum(split_ratio) == 1, "Split ratio must sum to 1"
    total_series, _ = series.split_before(sum(split_ratio)) # in case split ratio is not sum to 1 (for ETT in literature)
    train_series, test_series = total_series.split_before(sum(split_ratio[0:2])/sum(split_ratio))
    train_series, val_series = train_series.split_before(split_ratio[0]/sum(split_ratio[0:2]))
    data_series = DatasetSeries(
        train_series=train_series,
        val_series=val_series,
        test_series=test_series,
        )
    return data_series    


def window_transform(
    series: TimeSeries,
    window_size: int,
    ):
    centered = False
    transformation = {
            "function": "mean",
            "mode": "rolling",
            "window": window_size,
            "center": centered,
            "min_periods": 1,
        }
    return series.window_transform(transforms=transformation, forecasting_safe=True)
    
@dataclass
class DatasetSeries:
    train_series: TimeSeries = None
    val_series: TimeSeries = None
    test_series: TimeSeries = None
    test_series_noisy: TimeSeries = None
    backtest_series: TimeSeries = None
    scaler: Scaler = None

@dataclass
class DataSeries:
    series: TimeSeries = None
    covariates: TimeSeries = None

@dataclass
class DatasetCovSeries:
    train_series: DataSeries = DataSeries()
    val_series: DataSeries = DataSeries()
    test_series: DataSeries = DataSeries()
    scaler: Scaler = None

def darts_predefined_datasets(DatasetClass: DatasetLoaderCSV):
    
    total_ratio_ett = 0.826663
    default_split = (0.6*total_ratio_ett, 0.2*total_ratio_ett, 0.2*total_ratio_ett) if "ETT" in DatasetClass.__name__ else (0.7, 0.1, 0.2)
    def datasets_from_DatasetLoaderCSV(
        split_ratio: Tuple[float] = default_split,
        use_scaler: bool = True,
        target_series_index: int = None,
        multiscales: list = None,
        **kwargs
        ) -> Union[Tuple[TimeSeries, TimeSeries], Tuple[TimeSeries, TimeSeries, Scaler]]:
        """Creates a Darts dataset from a Darts DatasetLoaderCSV class.
        """
        # set split ratio 
        if split_ratio is None:
            split_ratio = default_split
        if isinstance(split_ratio, str):
            split_ratio = tuple(map(float, split_ratio.split()))
        
        # load darts dataset
        data_class = DatasetClass()
        data_class._root_path = DATA_ROOT
        series = data_class.load().astype(np.float32) 
        
        if target_series_index is not None:
            assert target_series_index<=len(series.components), f"Target series index out of range, choose it from 0 to {len(series.components)}."
            component = series.components[target_series_index]
            series = series[component]
            print(f"Using component {component} as target series.")
        
        # creates a moving average series 
        if multiscales is not None:
            covariate_series = []
            for scale in multiscales:
                covariate_series.append(window_transform(series, window_size=scale))
            
        # create dataseries for each series   
        data_series = split_series(series, split_ratio)
        
        
        
        if use_scaler:
            scaler = Scaler(scaler=preprocessing.StandardScaler())
            train_series_scaled = scaler.fit_transform(data_series.train_series)
            val_series_scaled = scaler.transform(data_series.val_series)
            test_series_scaled = scaler.transform(data_series.test_series)
            data_scaled_series = DatasetSeries(
                train_series=train_series_scaled,
                val_series=val_series_scaled,
                test_series=test_series_scaled,
                scaler=scaler
                )
            return data_scaled_series
        return data_series
    
    return datasets_from_DatasetLoaderCSV


_DATASETS = {
    "air_passengers": darts_predefined_datasets(AirPassengersDataset),
    "etth1": darts_predefined_datasets(ETTh1Dataset),
    "etth2": darts_predefined_datasets(ETTh2Dataset),
    "ettm1": darts_predefined_datasets(ETTm1Dataset),
    "ettm2": darts_predefined_datasets(ETTm2Dataset),
    "electricity": darts_predefined_datasets(ElectricityDataset),
    "exchange_rate": darts_predefined_datasets(ExchangeRateDataset),
    "traffic": darts_predefined_datasets(TrafficDataset),
    "weather": darts_predefined_datasets(WeatherDataset),
    "energy": darts_predefined_datasets(EnergyDataset),
    "uber": darts_predefined_datasets(UberTLCDataset),
    
}


@register_dataset
def crypto(
    split_ratio: Tuple[float] = (0.8, 0.1, 0.1),
    crypto_name: Union[str, Tuple[str]] = "All", 
    # ('Bitcoin Cash',
    # 'Binance Coin' ,
    # 'Bitcoin',
    # 'EOS.IO',
    # 'Ethereum Classic',
    # 'Ethereum',
    # 'Litecoin',
    # 'Monero',
    # 'TRON',
    # 'Stellar',
    # 'Cardano',
    # 'IOTA',
    # 'Maker',
    # 'Dogecoin')
    use_scaler: bool = True,
    prct_rows_to_load: float = 1.0,
    target_series_index: int = None,
    chunk_number: int = None,
    **kwargs
    ) -> Union[Tuple[TimeSeries, TimeSeries], Tuple[TimeSeries, TimeSeries, Scaler]]:
    """Loads crypto dataset
    Download the csv files manually from https://www.kaggle.com/competitions/g-research-crypto-forecasting/data
    Code copied and modified from https://github.com/google-research/google-research/blob/master/KNF/data/Cryptos/cryptos_data_gen.py
    """
    import pandas as pd
    
    if isinstance(split_ratio, str):
        split_ratio = tuple(map(float, split_ratio.split()))
    
    LEN_CRYPTO = 24236806
    nrows = int(prct_rows_to_load * LEN_CRYPTO)
    skip_rows = []
    
    if chunk_number is not None:
        skip_rows = range(1, chunk_number * nrows + 1)
         
    asset_details_df = pd.read_csv('~/.darts/datasets/crypto_asset.csv')
    crypto_df = pd.read_csv('~/.darts/datasets/crypto.csv', nrows=nrows, skiprows=skip_rows)
    
    # dropped Dogecoin since it is short
    asset_details_df = asset_details_df.drop(index=13)
    
    if crypto_name == "All":
        assets_df = asset_details_df
    else:
        if isinstance(crypto_name, str):
            crypto_name = (crypto_name,)
        assets_df = asset_details_df.query(f'Asset_Name in {crypto_name}')
    
    assets_series = {}
    for index, row in assets_df.iterrows():
        asset_id = row["Asset_ID"]
        asset_name = row["Asset_Name"]
        
        # set timestamps as index
        selected_crypto_df = crypto_df[crypto_df["Asset_ID"] == asset_id].set_index("timestamp")
        
        # The target is 15-min residulized future returns
        # We need to shift this feature up by 15 rows
        # so that each data entry doesn't contain future information.
        selected_crypto_df["Target"] = selected_crypto_df["Target"].shift(15)
        
        # fill nan with 0
        selected_crypto_df = selected_crypto_df.fillna(0)
        
        # unknown why
        selected_crypto_df = selected_crypto_df[20:-1]
        
        # zero out inf values
        selected_crypto_df[selected_crypto_df == float('inf')] = 0
        selected_crypto_df[selected_crypto_df == float('-inf')] = 0
        
        # reindex to fill missing timestamps
        selected_crypto_df = selected_crypto_df.reindex(range(selected_crypto_df.index[0],selected_crypto_df.index[-1]+60,60),method='pad')
        
        # drop unused columns
        selected_crypto_df = selected_crypto_df.drop(columns=["Asset_ID"])
        selected_crypto_df = selected_crypto_df.rename(columns=lambda name: name + f"_{asset_name}")
        
        # change index timestamps to datetime for better visualization
        selected_crypto_df = selected_crypto_df.set_index(selected_crypto_df.index.values.astype('datetime64[s]'))
        
        # create darts timeseries
        crypto_series = TimeSeries.from_dataframe(selected_crypto_df)
        
        assets_series[asset_name] = crypto_series


    # intersecting the series to find the common time range
    minumum_len_index = np.array([len(asset_series) for asset_series in assets_series.values()]).argmin()
    smallest_series = assets_series[list(assets_series.keys())[minumum_len_index]]
    start_time, end_time = smallest_series.time_index[0], smallest_series.time_index[-1]
    assets_series = {name: series.slice(start_ts=start_time, end_ts=end_time) for name, series in assets_series.items()}

    # concatenate all series
    series = concatenate(series=list(assets_series.values()), axis='component')
    
    if target_series_index is not None:
        assert target_series_index<=len(series.components), f"Target series index out of range, choose it from 0 to {len(series.components)}."
        component = series.components[target_series_index]
        series = series[component]
        print(f"Using component {component} as target series.")
    
    # create dataseries for each crypto series
    data_series = split_series(series, split_ratio)
    
    
    if use_scaler:
        scaler = Scaler(scaler=preprocessing.StandardScaler())
        train_series_scaled = scaler.fit_transform(data_series.train_series)
        val_series_scaled = scaler.transform(data_series.val_series)
        test_series_scaled = scaler.transform(data_series.test_series)
        data_scaled_series = DatasetSeries(
            train_series=train_series_scaled,
            val_series=val_series_scaled,
            test_series=test_series_scaled,
            scaler=scaler
            )
        return data_scaled_series
    return data_series
