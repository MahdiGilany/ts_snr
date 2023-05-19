import sys
import os


sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from rich import print as pprint
from typing import Literal

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
from src.modeling.vicreg import VICReg
from src.modeling.GP_approx_models import Laplace
from src.modeling.loss.loss_functions import create_loss_fx
from src.modeling.optimizer_factory import configure_optimizers
from torchmetrics import Accuracy, AUROC
from src.lightning.callbacks.components import metrics

import wandb
from src.utils import driver as utils
# from pytorch_lightning import seed_everything


log = utils.get_logger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model: nn.Module, train_dl: data.DataLoader, optimizer):
    losses_accumulator = {}
    model.train()
    for batch in tqdm(train_dl, desc="Training", leave=False):
        (x1, x2), y, metadata = batch
        x1 = x1.to(device)
        x2 = x2.to(device)
        losses = model(x1, x2)
        optimizer.zero_grad()
        losses["loss"].backward()
        optimizer.step()
            
        for k, v in losses.items():
            if k not in losses_accumulator:
                losses_accumulator[k] = [v.item()]
            else:
                losses_accumulator[k].append(v.item())
    return losses_accumulator

def val_epoch(model: nn.Module, val_dl: data.DataLoader, set_type: Literal["val", "test"]):
    losses_accumulator = {}
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_dl, desc=set_type, leave=False):
            (x1, x2), y, metadata = batch
            x1 = x1.to(device)
            x2 = x2.to(device)
            losses = model(x1, x2)
            
            for k, v in losses.items():
                if k not in losses_accumulator:
                    losses_accumulator[k] = [v.item()]
                else:
                    losses_accumulator[k].append(v.item())
    return losses_accumulator

@hydra.main(
    config_path="../configs/experiment",
    config_name="01_pretraian_vicreg_MG_2023-03-27.yaml",
    version_base="1.1",
)
def main(configs):
    # Configs
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

    train_ds = create_dataset(
        configs.data.dataset_name,
        split="train", 
        )
    weighted_sampler = get_weighted_sampler(train_ds)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=configs.batch_size, sampler=weighted_sampler, 
        num_workers=configs.data.num_workers, worker_init_fn=seed_worker, generator=g
    )
    
    val_ds = create_dataset(
        configs.data.dataset_name,
        split="val", 
        )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=configs.batch_size, shuffle=False, 
        num_workers=configs.data.num_workers, worker_init_fn=seed_worker, generator=g
    )
    
    test_ds = create_dataset(
        configs.data.dataset_name,
        split="test", 
        )
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=configs.batch_size, shuffle=False, 
        num_workers=configs.data.num_workers, worker_init_fn=seed_worker, generator=g
    )

    # Model
    backbone = create_model(configs.model.backbone_name, **configs.model.backbone_configs)
    model = VICReg(backbone, [512, 512], 512)


    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.optimizer.lr, weight_decay=configs.optimizer.weight_decay)
    # scheduler = LinearWarmupCosineAnnealingLR(
    #                     optimizer,  # type:ignore
    #                     warmup_epochs=0,
    #                     max_epochs=max_epochs * num_scheduling_steps_per_epoch,
    #                     warmup_start_lr=configs.optimizer.lr,
    #                 )




    # Training and validation
    best_val_loss = torch.inf
    model.to(device)
    for epoch in range(configs.num_epochs):
        print(f"Epoch {epoch}")
        # Training
        train_epoch_loss = train_epoch(model, train_dl, optimizer)
        wandb.log({"train/total_loss": np.mean(train_epoch_loss["loss"]), "epoch": epoch})
        wandb.log({
            "train/sim_loss": np.mean(train_epoch_loss["sim_loss"]),
            "train/var_loss": np.mean(train_epoch_loss["var_loss"]),
            "train/cov_loss": np.mean(train_epoch_loss["cov_loss"]),
            })

        
        # Validation
        val_epoch_loss = val_epoch(model, val_dl, "val")
        wandb.log({"val/total_loss": np.mean(val_epoch_loss["loss"]),})
        wandb.log({
            "val/sim_loss": np.mean(val_epoch_loss["sim_loss"]),
            "val/var_loss": np.mean(val_epoch_loss["var_loss"]),
            "val/cov_loss": np.mean(val_epoch_loss["cov_loss"]),
            })
        
        # Best model
        if np.mean(val_epoch_loss["loss"]) < best_val_loss:
            best_val_loss = np.mean(val_epoch_loss["loss"])
            torch.save(model.state_dict(), os.path.join(configs.name)+".pt")
            log.info(f"Model saved, val loss decreased to {best_val_loss}")
            print(f"Model saved, val loss decreased to {best_val_loss}")

        
        # # Test
        # test_epoch_loss = val_epoch(model, test_dl, "test")
        # wandb.log({"test/test_loss": test_epoch_loss})

    log.info(f"Last model saved")
    torch.save(model.state_dict(), os.path.join(configs.name)+"_last.pt")
        
if __name__ == "__main__":
    register_configs()
    main()
    
