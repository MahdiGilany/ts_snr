
import numpy as np
from typing import Union, Tuple
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from dataclasses import dataclass

from darts.datasets.dataset_loaders import DatasetLoaderCSV
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


def darts_datasets(DatasetClass: DatasetLoaderCSV):
    
    def datasets_from_DatasetLoaderCSV(
        split_ratio: Tuple[float] = (0.7, 0.1, 0.2),
        use_scaler: bool = True,
        **kwargs
        ) -> Union[Tuple[TimeSeries, TimeSeries], Tuple[TimeSeries, TimeSeries, Scaler]]:
        """Creates a Darts dataset from a Darts DatasetLoaderCSV class.
        """
        series = DatasetClass().load().astype(np.float32)
        data_series = split_series(series, split_ratio)
        
        if use_scaler:
            scaler = Scaler()
            train_series_scaled = scaler.fit_transform(data_series.train_series)
            val_series_scaled = scaler.transform(data_series.val_series)
            test_series_scaled = scaler.transform(data_series.test_series)
            data_scaled_series = DataSeries(
                train_series=train_series_scaled,
                val_series=val_series_scaled,
                test_series=test_series_scaled,
                scaler=scaler
                )
            return data_scaled_series
        return data_series
    
    return datasets_from_DatasetLoaderCSV


_DATASETS = {
    "air_passengers": darts_datasets(AirPassengersDataset),
    "etth1": darts_datasets(ETTh1Dataset),
    "etth2": darts_datasets(ETTh2Dataset),
    "ettm1": darts_datasets(ETTm1Dataset),
    "ettm2": darts_datasets(ETTm2Dataset),
    "electricity": darts_datasets(ElectricityDataset),
    "exchange_rate": darts_datasets(ExchangeRateDataset),
    "traffic": darts_datasets(TrafficDataset),
    "weather": darts_datasets(WeatherDataset),
    "energy": darts_datasets(EnergyDataset),
    "uber": darts_datasets(UberTLCDataset),
    
}


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



# # Execute this cell once to download all three datasets
# # Some datasets are are in pickle for simplicity and speed.
# !curl -L https://github.com/unit8co/amld2022-forecasting-and-metalearning/blob/main/data/m3_dataset.xls\?raw\=true -o m3_dataset.xls
# !curl -L https://github.com/unit8co/amld2022-forecasting-and-metalearning/blob/main/data/passengers.pkl\?raw\=true -o passengers.pkl
# !curl -L https://github.com/unit8co/amld2022-forecasting-and-metalearning/blob/main/data/m4_monthly_scaled.pkl\?raw\=true -o m4_monthly_scaled.pkl

# def load_m3() -> Tuple[List[TimeSeries], List[TimeSeries]]:
#     print("building M3 TimeSeries...")

#     # Read DataFrame
#     df_m3 = pd.read_excel("m3_dataset.xls", "M3Month")

#     # Build TimeSeries
#     m3_series = []
#     for row in tqdm(df_m3.iterrows()):
#         s = row[1]
#         start_year = int(s["Starting Year"])
#         start_month = int(s["Starting Month"])
#         values_series = s[6:].dropna()
#         if start_month == 0:
#             continue

#         start_date = datetime(year=start_year, month=start_month, day=1)
#         time_axis = pd.date_range(start_date, periods=len(values_series), freq="M")
#         series = TimeSeries.from_times_and_values(
#             time_axis, values_series.values
#         ).astype(np.float32)
#         m3_series.append(series)

#     print("\nThere are {} monthly series in the M3 dataset".format(len(m3_series)))

#     # Split train/test
#     print("splitting train/test...")
#     m3_train = [s[:-HORIZON] for s in m3_series]
#     m3_test = [s[-HORIZON:] for s in m3_series]

#     # Scale so that the largest value is 1
#     print("scaling...")
#     scaler_m3 = Scaler(scaler=MaxAbsScaler())
#     m3_train_scaled: List[TimeSeries] = scaler_m3.fit_transform(m3_train)
#     m3_test_scaled: List[TimeSeries] = scaler_m3.transform(m3_test)

#     print(
#         "done. There are {} series, with average training length {}".format(
#             len(m3_train_scaled), np.mean([len(s) for s in m3_train_scaled])
#         )
#     )
#     return m3_train_scaled, m3_test_scaled


# def load_air() -> Tuple[List[TimeSeries], List[TimeSeries]]:
#     # load TimeSeries
#     print("loading air TimeSeries...")
#     with open("passengers.pkl", "rb") as f:
#         all_air_series = pickle.load(f)

#     # Split train/test
#     print("splitting train/test...")
#     air_train = [s[:-HORIZON] for s in all_air_series]
#     air_test = [s[-HORIZON:] for s in all_air_series]

#     # Scale so that the largest value is 1
#     print("scaling series...")
#     scaler_air = Scaler(scaler=MaxAbsScaler())
#     air_train_scaled: List[TimeSeries] = scaler_air.fit_transform(air_train)
#     air_test_scaled: List[TimeSeries] = scaler_air.transform(air_test)

#     print(
#         "done. There are {} series, with average training length {}".format(
#             len(air_train_scaled), np.mean([len(s) for s in air_train_scaled])
#         )
#     )
#     return air_train_scaled, air_test_scaled


# def load_m4() -> Tuple[List[TimeSeries], List[TimeSeries]]:
#     # load TimeSeries - the splitting and scaling has already been done
#     print("loading M4 TimeSeries...")
#     with open("m4_monthly_scaled.pkl", "rb") as f:
#         m4_series = pickle.load(f)

#     # filter and keep only series that contain at least 48 training points
#     m4_series = list(filter(lambda t: len(t[0]) >= 48, m4_series))

#     m4_train_scaled, m4_test_scaled = zip(*m4_series)

#     print(
#         "done. There are {} series, with average training length {}".format(
#             len(m4_train_scaled), np.mean([len(s) for s in m4_train_scaled])
#         )
#     )
#     return m4_train_scaled, m4_test_scaled
