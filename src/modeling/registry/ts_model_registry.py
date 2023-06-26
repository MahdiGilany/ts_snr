import numpy as np
from typing import Union, Tuple

_MODELS = {}


def register_model(factory):
    _MODELS[factory.__name__] = factory

    return factory


def create_model(model_name, **kwargs):
    if model_name not in _MODELS:
        raise ValueError(f"Model <{model_name}> not registered.")

    return _MODELS[model_name](model_name=model_name, **kwargs)


def list_models():
    return list(_MODELS.keys())


@register_model
def nbeats(
    input_chunk_length: int = 24,
    output_chunk_length: int = 12,
    random_state: int = 0, 
    batch_size: int = 32,
    **kwargs,
    ):
    from darts.models import NBEATSModel
    model = NBEATSModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        random_state=random_state,
        batch_size=batch_size,
        **kwargs
        )
    return model


@register_model
def deeptime(
    input_chunk_length: int = 24,
    output_chunk_length: int = 12,
    random_state: int = 0,
    batch_size: int = 32,
    **kwargs,
    ):
    from ..models.deep_time import DeepTIMeModel
    model = DeepTIMeModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        random_state=random_state,
        batch_size=batch_size,
        **kwargs
        )
    return model


@register_model
def omp_deeptime(
    input_chunk_length: int = 24,
    output_chunk_length: int = 12,
    random_state: int = 0,
    batch_size: int = 32,
    **kwargs,
    ):
    from ..models.omp_deeptime import OMPDeepTIMeModel
    model = OMPDeepTIMeModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        random_state=random_state,
        batch_size=batch_size,
        **kwargs
        )
    return model    
    

@register_model
def nhits(
    input_chunk_length: int = 24,
    output_chunk_length: int = 12,
    random_state: int = 0, 
    batch_size: int = 32,
    **kwargs,
    ):
    from darts.models import NHiTSModel
    model = NHiTSModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        random_state=random_state,
        batch_size=batch_size,
        **kwargs
        )
    return model


@register_model
def nlinear(
    input_chunk_length: int = 24,
    output_chunk_length: int = 12,
    random_state: int = 0, 
    batch_size: int = 32,
    **kwargs,
    ):
    from darts.models import NLinearModel
    model = NLinearModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        random_state=random_state,
        batch_size=batch_size,
        **kwargs
        )
    return model


@register_model
def dlinear(
    input_chunk_length: int = 24,
    output_chunk_length: int = 12,
    random_state: int = 0, 
    batch_size: int = 32,
    **kwargs,
    ):
    from darts.models import DLinearModel
    model = DLinearModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        random_state=random_state,
        batch_size=batch_size,
        **kwargs
        )
    return model


@register_model
def naive_seasonal(
    input_chunk_length: int = 24,
    output_chunk_length: int = 12,
    random_state: int = 0,
    batch_size: int = 32,
    add_drift: bool = False,
    regression_ensemble: bool = False,
    **kwargs,
    ):
    from darts.models import NaiveSeasonal, NaiveDrift, NaiveEnsembleModel, RegressionEnsembleModel
    model_seasonality = NaiveSeasonal(
        K=input_chunk_length,
    )
    
    if regression_ensemble and not add_drift:
        raise ValueError("Regression ensemble only works with drift.")
        
    if add_drift:
        model_drift = NaiveDrift()
        model = NaiveEnsembleModel([model_seasonality, model_drift]) if not regression_ensemble else RegressionEnsembleModel(
            [model_seasonality, model_drift],
            regression_train_n_points=input_chunk_length,
            )
        return model       
    
    return model_seasonality


@register_model
def naive_mean(
    input_chunk_length: int = 24,
    output_chunk_length: int = 12,
    random_state: int = 0,
    batch_size: int = 32,
    **kwargs,
    ):
    from darts.models import NaiveMean
    model = NaiveMean()
    return model


@register_model
def naive_movingaverage(
    input_chunk_length: int = 24,
    output_chunk_length: int = 12,
    random_state: int = 0,
    batch_size: int = 32,
    **kwargs,
    ):
    from darts.models import NaiveMovingAverage
    model = NaiveMovingAverage(input_chunk_length=input_chunk_length)
    return model


@register_model
def naive_martingle(
    input_chunk_length: int = 24,
    output_chunk_length: int = 12,
    random_state: int = 0,
    batch_size: int = 32,
    **kwargs,
    ):
    from darts.models import NaiveSeasonal, NaiveDrift, NaiveEnsembleModel, RegressionEnsembleModel
    model_seasonality = NaiveSeasonal(
        K=1,
    )
    
    return model_seasonality


@register_model
def arima(
    input_chunk_length: int = 24,
    output_chunk_length: int = 12,
    d: int = 0,
    q: int = 24,
    random_state: int = 0,
    batch_size: int = 32,
    **kwargs,
    ):
    from darts.models import ARIMA
    model = ARIMA(p=input_chunk_length, d=d, q=q, random_state=random_state)
    return model


@register_model
def varima(
    input_chunk_length: int = 24,
    output_chunk_length: int = 12,
    d: int = 0,
    q: int = 24,
    random_state: int = 0,
    batch_size: int = 32,
    **kwargs,
    ):
    from darts.models import VARIMA
    model = VARIMA(p=input_chunk_length, d=d, q=q, random_state=random_state)
    return model