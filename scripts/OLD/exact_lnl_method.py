from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score
import sys
from functools import partial
import time
import os
import logging
from src.layers.losses.coteaching import loss_coteaching
from itertools import islice
import hydra
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from hydra.core.config_store import ConfigStore
from torchmetrics.functional import auroc


logging.basicConfig(level=logging.INFO)
import wandb


# append path one level up
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from src.utils.metrics import OutputCollector

from src.data.registry import list_datasets


LOG_INTERVAL = 50
INITIAL_FORGET_RATE = 0
FINAL_FORGET_RATE = 0.5
FINAL_FORGET_RATE_EPOCH = 5
WARMUP_EPOCHS = 1

EVAL_INTERVAL = 500


@dataclass
class Config:
    batch_size: int = 64
    backbone_lr: float = 1e-6
    lr: float = 1e-3
    weight_decay: float = 1e-8
    num_epochs: int = 100
    patience: int = 10
    start_training_backbone_at_epoch: int = 1
    debug: bool = False
    exp_dir: str = "experiments/exact_lnl_brainstorm"
    seed: int = 42
    device: str = "cuda"
    warmup_epochs: int = 2


ConfigStore.instance().store(name="exact_lnl_method_default", node=Config)


@hydra.main(config_path=".", config_name="exact_lnl_method.yaml", version_base="1.2")
def main(args: Config):
    import rich

    rich.print(args)

    # make the experiment directory
    # add a timestamp to the directory name to avoid overwriting
    exp_dir = Path(args.exp_dir)
    exp_dir = exp_dir / time.strftime("%Y-%m-%d")
    exp_dir.mkdir(parents=True, exist_ok=True)
    num_existing_dirs = len(os.listdir(exp_dir))
    exp_dir = exp_dir / str(num_existing_dirs)
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp_dir / "train").mkdir(parents=True, exist_ok=True)
    (exp_dir / "val").mkdir(parents=True, exist_ok=True)
    (exp_dir / "test").mkdir(parents=True, exist_ok=True)

    # save args to a file
    with open(exp_dir / "args.yaml", "w") as f:
        from omegaconf import OmegaConf

        yaml = OmegaConf.to_yaml(args)
        f.write(yaml)

    torch.manual_seed(args.seed)

    if not args.debug:
        wandb.init(project="SSL_plus_coteaching")
    else:
        wandb.log = lambda x: print(x)

    # load the dataset
    print("Loading the dataset...")
    from src.data.registry import (
        exact_patches_sl_tuffc_prostate,
        exact_patches_sl_tuffc_ndl,
    )

    train_dataset = exact_patches_sl_tuffc_prostate("train")
    if args.debug:
        import random

        subset = random.sample(range(len(train_dataset)), 1000)
        labels = train_dataset.labels
        train_dataset = torch.utils.data.Subset(train_dataset, subset)
        train_dataset.labels = [labels[i] for i in subset]

    val_dataset = exact_patches_sl_tuffc_ndl("val")
    test_dataset = exact_patches_sl_tuffc_ndl("test")

    # load the model
    print("Loading the model...")
    from src.modeling.registry import (
        create_model,
        vicreg_resnet10_pretrn_allcntrs_noPrst_ndl_crop,
    )

    backbone1 = vicreg_resnet10_pretrn_allcntrs_noPrst_ndl_crop(2).backbone
    fc1 = nn.Linear(512, 2)

    backbone2 = vicreg_resnet10_pretrn_allcntrs_noPrst_ndl_crop(2).backbone
    fc2 = nn.Linear(512, 2)

    # create the dataloaders
    print("Creating the dataloaders...")
    from torch.utils.data import DataLoader
    from torch.utils.data.sampler import WeightedRandomSampler
    import numpy as np

    labels = np.array(train_dataset.labels).astype("int")
    weight_for_classes = [1 / sum(labels == label) for label in np.unique(labels)]
    weights = [weight_for_classes[label] for label in labels]
    # logging.info(f"Weights for classes {weights}")

    train_sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(train_dataset),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    model = nn.ModuleDict(
        {
            "backbone1": backbone1,
            "fc1": fc1,
            "backbone2": backbone2,
            "fc2": fc2,
        }
    ).to(args.device)

    # create the optimizer
    print("Creating the optimizers...")
    from torch.optim import Adam

    backbone1_opt = Adam(
        backbone1.parameters(), lr=args.backbone_lr, weight_decay=args.weight_decay
    )
    fc1_opt = Adam(fc1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    backbone2_opt = Adam(
        backbone2.parameters(), lr=args.backbone_lr, weight_decay=args.weight_decay
    )
    fc2_opt = Adam(fc2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizers = [backbone1_opt, fc1_opt, backbone2_opt, fc2_opt]

    backbone2_scheduler = LinearWarmupCosineAnnealingLR(
        backbone2_opt,
        max_epochs=args.num_epochs * len(train_loader),
        warmup_epochs=WARMUP_EPOCHS * len(train_loader),
    )
    backbone1_scheduler = LinearWarmupCosineAnnealingLR(
        backbone1_opt,
        max_epochs=args.num_epochs * len(train_loader),
        warmup_epochs=WARMUP_EPOCHS * len(train_loader),
    )
    schedulers = [backbone1_scheduler, backbone2_scheduler]

    # train the model
    print("Training the model...")
    from src.utils.metrics import OutputCollector
    from src.utils.driver.monitor import ScoreImprovementMonitor
    from src.utils.driver.early_stopping import EarlyStoppingMonitor

    early_stopping_monitor = EarlyStoppingMonitor(args.patience)
    score_improvement_monitor = ScoreImprovementMonitor()
    score_improvement_monitor.on_improvement(
        lambda score: early_stopping_monitor.update(True)
    )
    score_improvement_monitor.on_no_improvement(
        lambda score: early_stopping_monitor.update(False)
    )
    from src.utils.checkpoints import save_checkpoint

    score_improvement_monitor.on_improvement(
        lambda score: save_checkpoint(
            model.state_dict(),
            exp_dir / "checkpoints" / f"best_{score:.4f}.ckpt",
            save_last=True,
            num_to_keep=2,
        )
    )

    for epoch in range(1, args.num_epochs + 1):

        if epoch <= args.warmup_epochs:
            model1 = nn.Sequential(backbone1, fc1)
            optimizers_1 = [backbone1_opt, fc1_opt]
            schedulers_1 = [backbone1_scheduler]
            warmup(
                model1,
                train_loader,
                val_loader,
                test_loader,
                args.device,
                optimizers_1,
                schedulers_1,
                epoch,
            )
            continue

        # training epoch
        collector = OutputCollector()
        epoch_outputs = {}
        model.train()
        from torchmetrics import AUROC

        auroc = AUROC(num_classes=2, pos_label=1, average="macro")
        for i, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        ):

            patch, label, metadata = batch
            patch = patch.to(args.device)
            label = label.to(args.device)

            [opt.zero_grad() for opt in optimizers]

            feats1 = model["backbone1"](patch)
            logits1 = model["fc1"](feats1)
            loss1 = F.cross_entropy(logits1, label, reduction="none")

            feats2 = model["backbone2"](patch)
            logits2 = model["fc2"](feats2)
            loss2 = F.cross_entropy(logits2, label, reduction="none")

            forget_rate = get_forget_rate(epoch, i, len(train_loader))
            _, ind1 = torch.topk(
                loss1, int((1 - forget_rate) * len(loss1)), dim=0, largest=False
            )
            _, ind2 = torch.topk(
                loss2, int((1 - forget_rate) * len(loss1)), dim=0, largest=False
            )

            clean_or_not1 = torch.zeros_like(label).float()
            clean_or_not1[ind1] = 1
            clean_or_not2 = torch.zeros_like(label).float()
            clean_or_not2[ind2] = 1

            loss1_update = loss1[ind2]
            loss2_update = loss2[ind1]

            loss = loss1_update.mean() + loss2_update.mean()
            loss.backward()

            [opt.step() for opt in optimizers]
            [scheduler.step() for scheduler in schedulers]

            agreement = len(set(ind1.tolist()) & set(ind2.tolist())) / len(ind1)

            collector.collect_batch(
                {
                    "label": label,
                    "cancer_prob": logits1.softmax(-1)[:, 1],
                    "loss": loss1,
                    "model1_says_clean": clean_or_not1,
                    "model2_says_clean": clean_or_not2,
                    **metadata,
                }
            )

            if i % LOG_INTERVAL == 0:
                wandb.log(
                    {
                        "update_loss": loss.item(),
                        "model1_positive_pred": (logits1.softmax(-1)[:, 1] > 0.5)
                        .float()
                        .mean()
                        .item(),
                        "model2_positive_pred": (logits2.softmax(-1)[:, 1] > 0.5)
                        .float()
                        .mean()
                        .item(),
                        "model1_clean_selection_expected_class": label[ind1]
                        .float()
                        .mean()
                        .item(),
                        "model2_clean_selection_expected_class": label[ind2]
                        .float()
                        .mean()
                        .item(),
                        "all_label_expected_class": label.float().mean().item(),
                        "model1_auroc_step": auroc(logits1, label).item(),
                        "model2_auroc_step": auroc(logits2, label).item(),
                        "epoch": epoch,
                        "forget_rate": forget_rate,
                        "network_agreement_on_clean": agreement,
                        "lr": backbone1_scheduler.get_last_lr()[-1],
                    }
                )

            if i % EVAL_INTERVAL == 0:
                # evaluation
                for split, loader in zip(["val", "test"], [val_loader, test_loader]):
                    output, metrics = evaluate(
                        torch.nn.Sequential(backbone1, fc1), loader, args.device, split
                    )
                    # output.to_csv(args.exp_dir / split / f"{epoch}.csv")
                    wandb.log(
                        {
                            **{f"{split}/{k}": metric for k, metric in metrics.items()},
                            "epoch": epoch,
                        }
                    )

        output = collector.compute(out_fmt="dataframe")
        output.to_csv(exp_dir / "train" / f"{epoch}.csv")
        epoch_outputs["train_auroc"] = roc_auc_score(
            output["label"], output["cancer_prob"]
        )
        loss = output["loss"].values
        loss_hist = wandb.Histogram(loss)
        wandb.log(
            {
                "train_auroc_epoch": epoch_outputs["train_auroc"],
                "epoch": epoch,
                "loss_hist": loss_hist,
            }
        )

        # evaluation
        for split, loader in zip(["val", "test"], [val_loader, test_loader]):
            output, metrics = evaluate(
                torch.nn.Sequential(backbone1, fc1), loader, args.device, split
            )
            output.to_csv(exp_dir / split / f"{epoch}.csv")
            wandb.log(
                {
                    **{f"{split}/{k}": metric for k, metric in metrics.items()},
                    "epoch": epoch,
                }
            )
            epoch_outputs[f"{split}_auroc"] = metrics["auroc"]

        wandb.log(epoch_outputs)
        if epoch == 1:
            table = pd.DataFrame([epoch_outputs])
        else:
            table = pd.concat([table, pd.DataFrame([epoch_outputs])])
        table.to_csv(exp_dir / "results.csv", index=False)
        score_improvement_monitor.update(epoch_outputs["val_auroc"])
        if early_stopping_monitor.should_early_stop():
            print("Early stopping triggered on epoch {epoch}".format(epoch=epoch))
            return


PENALTY_FACTOR = 100


def warmup(
    model, train_loader, val_loader, test_loader, device, optimizers, schedulers, epoch
):

    for i, batch in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch} - Warm up", leave=False)
    ):
        model.train()
        patch, label, metadata = batch
        patch = patch.to(device)
        label = label.to(device)
        logits = model(patch)
        loss = F.cross_entropy(logits, label)
        prob = logits.softmax(-1)
        entropy = -(prob * prob.log()).sum(-1)
        loss -= entropy.mean() * PENALTY_FACTOR
        [opt.zero_grad() for opt in optimizers]
        loss.backward()
        [opt.step() for opt in optimizers]
        [scheduler.step() for scheduler in schedulers]
        if i % 100 == 0:
            wandb.log(
                {
                    "warmup_loss": loss.item(),
                    "warmup_entropy": entropy.mean().item(),
                    "warmup_positive_pred": (logits.softmax(-1)[:, 1] > 0.5)
                    .float()
                    .mean()
                    .item(),
                    "warmup_expected_class": label.float().mean().item(),
                    "warmup_auroc_step": auroc(logits, label, num_classes=2).item(),
                    "epoch": epoch,
                    "lr": schedulers[0].get_last_lr()[-1],
                }
            )

        if i % EVAL_INTERVAL == 0:
            # evaluation
            for split, train_loader in zip(["val", "test"], [val_loader, test_loader]):
                output, metrics = evaluate(model, train_loader, device, split)
                # output.to_csv(args.exp_dir / split / f"{epoch}.csv")
                wandb.log(
                    {
                        **{f"{split}/{k}": metric for k, metric in metrics.items()},
                        "epoch": epoch,
                    }
                )


def evaluate(model, loader, device, split):
    model.eval()
    collector = OutputCollector()
    with torch.no_grad():
        for batch in (pbar := tqdm(loader, desc=f"{split}", leave=False)):
            patch, label, metadata = batch
            patch = patch.to(device)
            label = label.to(device)
            logits = model(patch)
            probs = logits.softmax(-1)
            collector.collect_batch(
                {
                    "label": label,
                    "cancer_prob": probs[:, 1],
                    "core_specifier": metadata["core_specifier"],
                }
            )
        output = collector.compute()
        output = pd.DataFrame(output)
        metrics = {}
        metrics["auroc"] = roc_auc_score(output["label"], output["cancer_prob"])
        metrics["positive_pred_ratio"] = (output["cancer_prob"] > 0.5).mean()
        metrics["positive_label_ratio"] = output["label"].mean()
        return output, metrics


def get_forget_rate(epoch, step, steps_per_epoch):
    epoch -= 1
    total_steps = epoch * steps_per_epoch + step

    if total_steps <= FINAL_FORGET_RATE_EPOCH * steps_per_epoch:
        return INITIAL_FORGET_RATE + (
            FINAL_FORGET_RATE - INITIAL_FORGET_RATE
        ) * total_steps / (FINAL_FORGET_RATE_EPOCH * steps_per_epoch)
    else:
        return FINAL_FORGET_RATE


if __name__ == "__main__":
    main()
