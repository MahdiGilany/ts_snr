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
    from darts.models.forecasting.nbeats import NBEATSModel
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
def OMPdeeptime(
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
