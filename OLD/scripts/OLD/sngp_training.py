import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from rich import print as pprint

## Loading all configs
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from hydra.core.global_hydra import GlobalHydra
from src.configuration import register_configs
from tqdm import tqdm

from src.data.registry import create_dataset
from src.data.sampler import get_weighted_sampler
from src.modeling.registry import create_model
from src.modeling.GP_approx_models import Laplace
from src.modeling.loss.loss_functions import create_loss_fx
from src.modeling.optimizer_factory import configure_optimizers
from torchmetrics import Accuracy, AUROC

import wandb
from src.utils import driver as utils
# from pytorch_lightning import seed_everything



def main():
    log = utils.get_logger(__name__)


    # Configs
    register_configs()

    GlobalHydra.instance().clear()
    initialize(config_path="../configs", version_base="1.1")

    configs = compose(
        config_name="experiment/01_SNGP_baseline_MG_2023-03-14.yaml",
        )
    pprint(OmegaConf.to_yaml(configs))


    # Seed
    if (s := configs.get("seed")) is not None:
        log.info(f"Global seed set to {s}")
        print("Global seed", s)
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
        # torch.backends.cudnn.benchmark = False 
        # torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms(mode=True, warn_only=True)
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # WandB
    wandb_config = OmegaConf.to_container(
        configs, resolve=True, throw_on_missing=True
    )
    wandb.init(
        project=configs.logger.wandb.project,
        entity=configs.logger.wandb.entity,
        config=wandb_config,
        name=configs.name,
        )


    # Data
    global_seed = configs.seed
    def seed_worker(worker_id):
        worker_seed = global_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)

    train_ds = create_dataset(configs.data.dataset_name, split="train")
    weighted_sampler = get_weighted_sampler(train_ds)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=configs.batch_size, sampler=weighted_sampler, 
        num_workers=configs.data.num_workers, worker_init_fn=seed_worker, generator=g
    )
    val_ds = create_dataset(configs.data.dataset_name, split="val")
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=configs.batch_size, shuffle=False, 
        num_workers=configs.data.num_workers, worker_init_fn=seed_worker, generator=g
    )
    test_ds = create_dataset(configs.data.dataset_name, split="test")
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=configs.batch_size, shuffle=False, 
        num_workers=configs.data.num_workers, worker_init_fn=seed_worker, generator=g
    )

    # Model
    model = create_model(**configs.model, num_data= len(train_ds))


    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.optimizer.lr, weight_decay=configs.optimizer.weight_decay)
    criterion = create_loss_fx(configs.loss_name, num_classes=configs.num_classes)
    # scheduler = LinearWarmupCosineAnnealingLR(
    #                     optimizer,  # type:ignore
    #                     warmup_epochs=0,
    #                     max_epochs=max_epochs * num_scheduling_steps_per_epoch,
    #                     warmup_start_lr=configs.optimizer.lr,
    #                 )




    # Training and validation
    from src.lightning.callbacks.components import metrics
    patch_metric_manager = metrics.PatchMetricManager(list(set(val_ds.metadata['center'])), device="cpu")
    core_metric_manager = metrics.CoreMetricManager(val_ds, test_ds, list(set(val_ds.metadata['center'])), device="cpu") # TODO: core_lenghts can be given instead
    acc = Accuracy(average="macro", num_classes=2).to("cuda")
    auc = AUROC(average="macro", num_classes=2).to("cuda")
    best_val_auc = 0
    model.to("cuda")
    for epoch in range(configs.num_epochs):
        print(f"Epoch {epoch}")
        # Training
        batches_Loss = []
        model.reset_precision_matrix() # reset the precision matrix based on SNGP paper
        model.train()
        
        # freeze the feature extractor
        if configs.freeze_backbone:
            model.feature_extractor.feature_extractor_.eval()
            model.feature_extractor.feature_extractor_.requires_grad_(requires_grad=False)
        
        for batch in tqdm(train_dl, desc="Training"):
            x, y, metadata = batch
            x = x.to("cuda")
            y = y.to("cuda")
            y_hat = model(x)
            loss = criterion(y_hat, y, reduction='mean')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batches_Loss.append(loss.item())
            acc(y_hat, y)    
            auc(y_hat, y)
        wandb.log({"train/train_loss": np.mean(batches_Loss), "train/train_acc": acc.compute()})
        wandb.log({"train/train_auc": auc.compute()})
        auc.reset()
        acc.reset()
        
        
        # Validation
        # model.seen_data = torch.tensor(len(train_ds))
        batches_Loss = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_dl, desc="Validation"):
                x, y, metadata = batch
                x = x.to("cuda")
                y = y.to("cuda")
                y_hat = model(x)
                # y_hat = y_hat[0]
                loss = criterion(y_hat, y, reduction='mean')
                batches_Loss.append(loss.item())
                patch_metric_manager.update_wMetadata(ds_type="val", logits=y_hat, labels=y, metadata=metadata)
        core_metric_manager.update_wPatchmanager(patch_metric_manager, "val")
        # metrics
        val_metrics = patch_metric_manager.compute('val', 'ALL')
        val_core_metrics = core_metric_manager.compute("val", "ALL")
        wandb.log(val_metrics)
        wandb.log(val_core_metrics)
        wandb.log({"val/val_loss": np.mean(batches_Loss)})
        
        # Best model
        auc_value = [v for k, v in val_metrics.items() if "auc" in k][0]
        if best_val_auc < auc_value:
            best_val_auc = auc_value
            torch.save(model.state_dict(), os.path.join("logs/ckpt_store",configs.name)+".pt")
            log.info(f"Model saved, val AUC improved to {best_val_auc}")
            print(f"Model saved, val AUC improved to {best_val_auc}")

        
        # Test
        batches_Loss = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_dl, desc="Test"):
                x, y, metadata = batch
                x = x.to("cuda")
                y = y.to("cuda")
                y_hat = model(x)
                # y_hat = y_hat[0]
                loss = criterion(y_hat, y, reduction='mean')
                batches_Loss.append(loss.item())
                patch_metric_manager.update_wMetadata(ds_type="test", logits=y_hat, labels=y, metadata=metadata)
        core_metric_manager.update_wPatchmanager(patch_metric_manager, "test")
        # metrics
        test_metrics = patch_metric_manager.compute('test', 'ALL')
        test_core_metrics = core_metric_manager.compute("test", "ALL")
        wandb.log(test_metrics)
        wandb.log(test_core_metrics)
        wandb.log({"test/test_loss": np.mean(batches_Loss)})
        
        # reset metrics
        patch_metric_manager.reset()
        core_metric_manager.reset()
    
    
    
if __name__ == "__main__":
    main()
    
