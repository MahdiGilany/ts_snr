# append path one level up
import sys
import os


from tqdm import tqdm

# from functools import partialmethod
#
# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from rich.logging import RichHandler
from src.modeling.vicreg import VICReg
from src.modeling.registry.registry import create_model
from src.data.registry import create_dataset
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from src.utils.driver.stateful import StatefulCollection
from tqdm import tqdm
from contextlib import nullcontext
import wandb
import pandas as pd
from functools import partial
import logging
import json
from src.data.registry import create_dataset, list_datasets

LOGGING_LEVEL = logging.INFO
VALIDATE_EVERY_N_EPOCHS = 1
BATCH_SIZE_FACTOR_FOR_VALIDATION = 4  # amount to increase the batch size for validation


from pathlib import Path
from src.utils import FileBackedDictionary


def parse_args():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Train a VICReg model")
    parser.add_argument("--workdir", type=Path, default="workdir")
    parser.add_argument("--wandb_project", type=str, default="vicreg")
    parser.add_argument("--wandb_name", type=str, default="test")
    parser.add_argument("--disable_wandb", action="store_true", default=False)
    parser.add_argument(
        "--unlabeled_dataset",
        type=str,
        default="exact_patches_ssl_tuffc_ndl",
        choices=list_datasets(),
    )
    parser.add_argument(
        "--labeled_dataset",
        type=str,
        default="exact_patches_sl_tuffc_ndl",
    )
    parser.add_argument(
        "--backbone_name", type=str, default="resnet18_feature_extractor"
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)

    return parser.parse_args()


def main(rank, args):
    # setup distributed training
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(
        "nccl", rank=rank, world_size=args.num_gpus, init_method="env://"
    )
    dist.barrier()

    # set up workdir
    if rank == 0:
        os.makedirs(args.workdir, exist_ok=True)
    dist.barrier()
    os.chdir(args.workdir)

    if rank != 0:
        # disable tqdm and logging for non-rank 0 processes
        logging.basicConfig(
            level=LOGGING_LEVEL,
            format=f"RANK {rank} %(message)s",
            handlers=[logging.NullHandler()],
        )
        from tqdm import tqdm
        from functools import partialmethod

        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    else:
        logging.basicConfig(
            level=LOGGING_LEVEL,
            format="%(message)s",
            handlers=[
                RichHandler(rich_tracebacks=True),
                logging.FileHandler("log.txt"),
            ],
        )

    # set up logging
    logging.info("Setting up logging")
    if not args.disable_wandb and rank == 0:
        if "run_id" in FileBackedDictionary("config_history.json"):
            logging.info("Resuming wandb run")
            run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                config=args,
                id=FileBackedDictionary("config_history.json")["run_id"],
            )
        else:
            logging.info("Starting new wandb run")
            run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                config=args,
            )
            FileBackedDictionary("config_history.json")["run_id"] = run.id
    else:
        wandb.log = lambda x: None

    # set up datasets
    logging.info("Setting up dataset")
    train_dataset = create_ssl_dataset(args)

    (
        supervised_dataset_train,
        supervised_dataset_val,
        supervised_dataset_test,
    ) = create_supervised_datasets(args)
    # set up dataloaders
    local_batch_size = args.batch_size // args.num_gpus
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.num_gpus, rank=rank
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler,
    )

    # set up model
    logging.info("Setting up model")
    backbone = create_model(args.backbone_name)

    from src.modeling.vicreg import VICReg

    model = VICReg(backbone, [512, 512], 512)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # set up optimizer

    logging.info("Setting up optimizer")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=5 * len(train_dataloader),
        max_epochs=args.num_epochs * len(train_dataloader),
    )

    # load checkpoint
    if os.path.exists("checkpoint.pth"):
        logging.info("Loading checkpoint from checkpoint.pth")
        epoch = load_checkpoint(model, optimizer, scheduler, "checkpoint.pth")
    else:
        logging.info("No checkpoint found, starting from scratch")
        epoch = 1
    dist.barrier()

    # train
    while epoch <= args.num_epochs:

        train_epoch(model, optimizer, scheduler, train_dataloader, rank, epoch)
        if epoch % VALIDATE_EVERY_N_EPOCHS == 0 and rank == 0:
            validate_epoch(
                args,
                epoch,
                backbone,
                supervised_dataset_train,
                supervised_dataset_val,
                supervised_dataset_test,
            )
        dist.barrier()
        epoch += 1
        if rank == 0:
            logging.info(
                f"Saving snapshot at end of epoch {epoch - 1} to {args.workdir}"
            )
            checkpoint(model, optimizer, scheduler, epoch, "checkpoint.pth")
        dist.barrier()

    if rank == 0:
        wandb.finish()

    dist.destroy_process_group()


def create_supervised_datasets(args):
    train = create_dataset(args.labeled_dataset, split="train")
    val = create_dataset(args.labeled_dataset, split="val")
    test = create_dataset(args.labeled_dataset, split="test")
    return train, val, test


def create_ssl_dataset(args):
    return create_dataset(args.unlabeled_dataset, split="train")


def setup_dist(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def train_epoch(model, optimizer, scheduler, dataloader, rank, epoch):

    model.train()
    logging.debug(f"Training epoch {epoch} - model.train()")

    losses_accumulator = {}

    for i, batch in enumerate(
        tqdm(dataloader, desc=f"Training epoch {epoch}", leave=False)
    ):
        logging.debug(f"Training epoch {epoch} - batch {i}")
        optimizer.zero_grad()
        X1, X2 = batch[0]
        X1 = X1.to(rank)
        logging.debug(f"Training epoch {epoch} - batch {i} - X1.to(rank)")
        X2 = X2.to(rank)
        logging.debug(f"Training epoch {epoch} - batch {i} - X2.to(rank)")
        losses = model(X1, X2)
        losses["loss"].backward()
        logging.debug(f"Training epoch {epoch} - batch {i} - backward")

        for k, v in losses.items():
            if k not in losses_accumulator:
                losses_accumulator[k] = v.item()
            else:
                losses_accumulator[k] += v.item()

        optimizer.step()
        scheduler.step()

        if rank == 0 and i % 10 == 0:
            metrics = {f"train/{k}": v / (i + 1) for k, v in losses_accumulator.items()}
            metrics["epoch"] = epoch
            metrics["lr"] = optimizer.param_groups[0]["lr"]
            wandb.log(metrics)


def validate_epoch(args, epoch, model, train_set, val_set, test_set):
    logging.info(f"Validating epoch {epoch}")

    model.eval()

    from src.utils.metrics import OutputCollector

    oc = OutputCollector()
    create_loader = partial(
        DataLoader,
        batch_size=args.batch_size * BATCH_SIZE_FACTOR_FOR_VALIDATION,
        num_workers=0,
        shuffle=False,
    )
    train_loader = create_loader(train_set)
    val_loader = create_loader(val_set)
    test_loader = create_loader(test_set)

    with torch.no_grad():
        for i, (X, y, info) in enumerate(
            tqdm(train_loader, desc="Collecting X_train for linear eval", leave=False)
        ):
            logging.debug(f"Collecting X_train for linear eval - batch {i}")
            oc.collect_batch(
                {
                    "label": y,
                    "core_specifier": info["core_specifier"],
                    "position": info["position"],
                    "feats": model(X.to(0)),
                }
            )
        out_train = oc.compute()
        oc.reset()

        for X, y, info in tqdm(
            val_loader, desc="Collecting X_val for linear eval", leave=False
        ):
            oc.collect_batch(
                {
                    "label": y,
                    "core_specifier": info["core_specifier"],
                    "position": info["position"],
                    "feats": model(X.to(0)),
                }
            )
        out_val = oc.compute()
        oc.reset()

        for X, y, info in tqdm(
            test_loader, desc="Collecting X_test for linear eval", leave=False
        ):
            oc.collect_batch(
                {
                    "label": y,
                    "core_specifier": info["core_specifier"],
                    "position": info["position"],
                    "feats": model(X.to(0)),
                }
            )
        out_test = oc.compute()
        oc.reset()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    logging.info("Training linear classifier")
    lr = LogisticRegression(max_iter=2000)
    lr.fit(out_train["feats"], out_train["label"])
    out_train["pred"] = lr.predict_proba(out_train["feats"])[:, 1]
    out_val["pred"] = lr.predict_proba(out_val["feats"])[:, 1]
    out_test["pred"] = lr.predict_proba(out_test["feats"])[:, 1]

    # create csv with predictions, labels, core_specifier, and patch position
    def make_csv(out):
        out_new = {}
        out_new["pred"] = out["pred"]
        out_new["label"] = out["label"]
        out_new["core_specifier"] = out["core_specifier"]
        out_new["pos_axial"] = out["position"][:, 0]
        out_new["pos_lateral"] = out["position"][:, 2]
        return pd.DataFrame(out_new)

    out_train = make_csv(out_train)
    out_val = make_csv(out_val)
    out_test = make_csv(out_test)

    # save csvs
    logging.info("Saving predictions to csv")
    out_train.to_csv(f"train_preds_{epoch}.csv")
    out_val.to_csv(f"val_preds_{epoch}.csv")
    out_test.to_csv(f"test_preds_{epoch}.csv")

    # log metrics
    metrics = {
        "train/roc_auc": roc_auc_score(out_train["label"], out_train["pred"]),
        "val/roc_auc": roc_auc_score(out_val["label"], out_val["pred"]),
        "test/roc_auc": roc_auc_score(out_test["label"], out_test["pred"]),
        "epoch": epoch,
    }
    wandb.log(metrics)


def checkpoint(model, optimizer, scheduler, epoch, checkpoint_path):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
        },
        checkpoint_path + ".tmp",
    )
    os.rename(
        checkpoint_path + ".tmp",
        checkpoint_path,
    )


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    epoch = checkpoint["epoch"]
    return epoch


if __name__ == "__main__":

    args = parse_args()
    import rich

    rich.print(args)

    torch.multiprocessing.spawn(main, nprocs=args.num_gpus, args=(args,))
