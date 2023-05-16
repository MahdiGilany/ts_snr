# author: Alex Pashevish
import torch
import wandb
import numpy as np

from termcolor import colored
from sklearn.preprocessing import MinMaxScaler
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from darts.models import NBEATSModel, NHiTSModel, RNNModel, TCNModel, TransformerModel, TFTModel
from darts.models import VARIMA, Prophet, KalmanForecaster
from darts.models import NaiveDrift, NaiveMean, NaiveSeasonal
# TODO: maybe try to pass likelihood=GaussianLikelihood() to models (for probabilistic predictions)
from darts.utils.likelihood_models import GaussianLikelihood
from darts.utils.losses import MAELoss, MapeLoss, SmapeLoss
from darts.dataprocessing.transformers import Scaler
from darts.models.forecasting.torch_forecasting_model import (
    PastCovariatesTorchModel, FutureCovariatesTorchModel,
    MixedCovariatesTorchModel, DualCovariatesTorchModel)

from tsf import constants, utils
from tsf.transformers import NewTransformerModel, AutoformerModel, FEDformerModel
from tactis.darts.model import TACTiSModel

from abc import ABC, abstractmethod



@dataclass
class OptimizerConfig:
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    early_stopping: bool = True,
    loss_fn: str = 'mae',

class BaseTimeSeries(ABC):
    def __init__(
        self,
        callbacks,
        force_reset,
        covariate_config,
        epochs: int = 100,
        batch_size: int = 32,
        opt_config: dict = None,
        ) -> None:
        pass

    @abstractmethod
    def fit(self, train_series, **kwargs):
        ...
        
    
    

















def fixed_hyperparams(model_name, input_length, output_length):
    params = {
        'varima': {
            'output_chunk_length': output_length,
        },
        'kalman': {
            'output_chunk_length': output_length,
        },
        'prophet': {
            'output_chunk_length': output_length,
        },
        'nbeats': {
            'input_chunk_length': input_length,
            'output_chunk_length': output_length,
            'generic_architecture': True,
            'instance_norm': True,
            'date_covariates': True,
        },
        'nhits': {
            'input_chunk_length': input_length,
            'output_chunk_length': output_length,
            'instance_norm': True,
            'date_covariates': True,
        },
        'tcn': {
            'input_chunk_length': input_length,
            'output_chunk_length': output_length,
            'instance_norm': True,
            'date_covariates': True,
        },
        'deepar': {
            'input_chunk_length': input_length,
            'training_length': input_length + output_length,
            'instance_norm': True,
            'date_covariates': True,
        },

        'transformer': {
            'input_chunk_length': input_length,
            'output_chunk_length': output_length,
            'instance_norm': True,
            'date_covariates': True,
        },
        'ntransformer': {
            'input_chunk_length': input_length,
            'output_chunk_length': output_length,
            # 'instance_norm': True,
            # 'date_covariates': True,
        },
        'autoformer': {
            'input_chunk_length': input_length,
            'output_chunk_length': output_length,
            # 'instance_norm': True,
            # 'date_covariates': True,
        },
        'fedformer': {
            'input_chunk_length': input_length,
            'output_chunk_length': output_length,
            # 'instance_norm': True,
            # 'date_covariates': True,
        },

        'tactis': {
            'input_chunk_length': input_length,
            'output_chunk_length': output_length,
            'instance_norm': True,
            'instance_affine': False,
            'date_covariates': False, # TODO: start using them?
            # those are fred_md specific params
            "num_samples": 100, # only used for evaluation
            "series_embedding_dim": 5,
            "input_encoder_layers": 2,
            "input_encoding_normalization": True,
            "loss_normalization": "series",
            # "bagging_size": 20,
            "use_positional_encoding": True,
            "dropout": 0.0, # passed to both positional encoding and temporal_encoder
            "temporal_encoder": {
                "attention_layers": 2,
                "attention_heads": 1,
                "attention_dim": 16,
                "attention_feedforward_dim": 16,
            },
            "copula_decoder": {
                "min_u": 0.05,
                "max_u": 0.95,
                "attentional_copula": {
                    "attention_heads": 3,
                    "attention_layers": 1,
                    "attention_dim": 8,
                    "resolution": 20,
                },
                "dsf_marginal": {
                    "flow_layers": 2,
                    "flow_hid_dim": 8,
                },
            },
            "decoder_mlp_layers": 2, # passed to both copula and DSF
            "decoder_mlp_dim": 48, # passed to both copula and DSF
            "lr": 1e-3,
            "batch_size": 256,
            "optimizer_kwargs": {
                "weight_decay": 1e-4,
            },
            "pl_trainer_kwargs": {
                "gradient_clip_val": 1e3,
            },
            # "lr_scheduler_cls": torch.optim.lr_scheduler.OneCycleLR,
            # "lr_scheduler_kwargs": {
            #     "max_lr": 1e-3,
            #     "steps_per_epoch": 512,
            # },
        },
        'tft': {
            'input_chunk_length': input_length,
            'output_chunk_length': output_length,
            'instance_norm': True, # this is not implemented at the moment
            'date_covariates': True,
        },
    }
    assert model_name in params, 'unknown model {}'.format(model_name)
    return params[model_name]


def build_fit_nonparametric(model_name, train_series, fit=True, **model_args):
    if model_name == 'varima':
        ModelClass = VARIMA
    elif model_name == 'prophet':
        ModelClass = Prophet
    elif model_name == 'kalman':
        ModelClass = KalmanForecaster
    elif model_name == 'naive_drift':
        ModelClass = NaiveDrift
    elif model_name == 'naive_mean':
        ModelClass = NaiveMean
    elif model_name == 'naive_seasonal':
        ModelClass = NaiveSeasonal
    else:
        raise NotImplementedError('unknown model type: {}'.format(model_name))
    model = ModelClass(**model_args)
    if fit:
        model = model.fit(train_series)
    return model

def build_fit_parametric(
        model_name,
        model_work_dir,
        train_series,
        val_series,
        train_covariates=None,
        val_covariates=None,
        lr=1e-3,
        n_epochs=100,
        batch_size=64,
        callbacks=None,
        reload_best_model=False,
        verbose=False,
        save_checkpoints=False,
        force_reset=True,
        likelihood=None,
        loss_fn='mse',
        date_covariates=False,
        early_stopping_args=None,
        **model_args,
):
    callbacks = callbacks if callbacks else []
    if early_stopping_args:
        callbacks.append(EarlyStopping(**early_stopping_args))
    model_name_full = utils.model_args_to_full_name(model_name, model_args, verbose)
    print('Model name is {}'.format(model_name_full))

    loss_fns = {'mse': torch.nn.MSELoss, 'mae': MAELoss, 'mape': MapeLoss, 'smape': SmapeLoss}
    batches_per_epoch_train = int(np.ceil(len(train_series) / batch_size))
    model_args.update(dict(
        n_epochs=n_epochs,
        batch_size=batch_size,
        model_name=model_name_full,
        nr_epochs_val_period=1,
        force_reset=force_reset,
        save_checkpoints=save_checkpoints,
        work_dir=model_work_dir,
        likelihood=GaussianLikelihood() if likelihood == 'gaussian' else None,
        loss_fn=loss_fns[loss_fn](),
    ))
    model_args_second_level = dict(
        optimizer_kwargs=dict(
            lr=lr,
        ),
        pl_trainer_kwargs=dict(
            accelerator='gpu',
            devices=-1,
            auto_select_gpus=True,
            log_every_n_steps=batches_per_epoch_train,
            callbacks=callbacks,
        ),
    )
    for kwarg, kwarg_dict in model_args_second_level.items():
        if kwarg not in model_args:
            model_args[kwarg] = kwarg_dict
        else:
            model_args[kwarg].update(kwarg_dict)

    # # TODO: remove this ugly hack later on
    # if (model_args.get('lr_scheduler_cls') is torch.optim.lr_scheduler.OneCycleLR and
    #     'lr_scheduler_kwargs' in model_args and
    #     'steps_per_epoch' in model_args['lr_scheduler_kwargs']):
    #     model_args['lr_scheduler_kwargs']['epochs'] = int(np.ceil(
    #         n_epochs / model_args['lr_scheduler_kwargs']['steps_per_epoch'] * batches_per_epoch_train))

    if not isinstance(wandb.run, wandb.sdk.lib.disabled.RunDisabled):
        model_args['pl_trainer_kwargs']['logger'] = WandbLogger(
            experiment=wandb.run,
            log_model=False, # it did not work the last time I tried
        )
    else:
        # save logs locally with tensorboard
        model_args['log_tensorboard'] = True

    model_name_to_class = dict(
        nbeats=NBEATSModel,
        nhits=NHiTSModel,
        deepar=RNNModel,
        tcn=TCNModel,
        transformer=TransformerModel,
        ntransformer=NewTransformerModel,
        autoformer=AutoformerModel,
        fedformer=FEDformerModel,
        tactis=TACTiSModel,
        tft=TFTModel,
    )
    ModelClass = model_name_to_class[model_name]

    if date_covariates:
        print('Date covariates will be used')
        model_args['add_encoders'] = {
            'datetime_attribute': dict(),
            'transformer': Scaler(MinMaxScaler(feature_range=(0, 1)))}
        if issubclass(ModelClass, (PastCovariatesTorchModel, MixedCovariatesTorchModel)):
            model_args['add_encoders']['datetime_attribute']['past'] = ['dayofweek', 'month']
        if issubclass(ModelClass, (FutureCovariatesTorchModel, DualCovariatesTorchModel)):
            model_args['add_encoders']['datetime_attribute']['future'] = ['dayofweek', 'month']

    if model_name == 'tft':
        model_args['likelihood'] = None
    model = ModelClass(**model_args)

    if n_epochs > 0:
        fit_args = dict(
            series=train_series, val_series=val_series, verbose=verbose, num_loader_workers=8)
        if model.uses_past_covariates:
            fit_args.update(dict(
                past_covariates=train_covariates,
                val_past_covariates=val_covariates))
        if model.uses_future_covariates:
            fit_args.update(dict(
                future_covariates=train_covariates,
                val_future_covariates=val_covariates))
        model.fit(**fit_args)
        if reload_best_model:
            assert save_checkpoints, 'can not reload a model if checkpoints are not saved'
            model = model.load_from_checkpoint(model_name_full, work_dir=model_work_dir, best=True)
            print('Reloaded the model from {}'.format(model.load_ckpt_path))
    else:
        print('Attempting to reload the model...')
        model = model.load_from_checkpoint(model_name_full, work_dir=model_work_dir, best=True)
        print('Reloaded the model from {}'.format(model.load_ckpt_path))
    return model
