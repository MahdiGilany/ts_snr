# simple multi-gpu training example
from pathlib import Path
import os
from src.data.registry import create_dataset
from src.modeling.registry import create_model
from torch import distributed as dist
from torchmetrics.functional import accuracy
import torch
from tqdm import tqdm
import wandb
import pandas as pd
import numpy as np


def parse_args():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Train on multiple GPUs")
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--exp_dir", type=Path, default="./test/default")
    parser.add_argument("--dataset_name", type=str, default="cifar10_fast")
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--logger", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")

    return parser.parse_args()


def train(args):

    if not args.exp_dir.exists():
        args.exp_dir.mkdir(parents=True)

    # setup dataloaders
    train_ds = create_dataset(args.dataset_name, split="train")
    test_ds = create_dataset(args.dataset_name, split="test")
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # setup model
    # model = create_model(args.model_name).to(rank)
    from torchvision.models import resnet18

    model = resnet18(num_classes=10).to(args.device)

    # setup optimizer
    from torch.optim import Adam

    optimizer = Adam(model.parameters(), lr=args.lr)

    all_results = []

    for epoch in range(1, args.num_epochs):
        epoch_results = dict(epoch=epoch)

        model.train()
        all_outputs = []
        all_targets = []
        loss_accumulator = 0
        iterator = tqdm(
            train_dl, desc=f"Epoch {epoch}/{args.num_epochs} - Training", leave=False
        )
        for i, batch in enumerate(iterator):
            x, y = batch
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            loss_accumulator += loss.item()
            iterator.set_postfix(dict(loss=f"{loss_accumulator / (i + 1):.4f}"))
            all_outputs.append(logits.detach())
            all_targets.append(y.detach())

        # compute accuracy
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        train_acc = accuracy(all_outputs, all_targets)
        epoch_results["train_acc"] = train_acc.item()

        # evaluate
        model.eval()
        all_outputs = []
        all_targets = []
        iterator = tqdm(
            test_dl, desc=f"Epoch {epoch}/{args.num_epochs} - Testing", leave=False
        )
        for i, batch in enumerate(iterator):
            with torch.no_grad():
                x, y = batch
                x, y = x.to(args.device), y.to(args.device)
                logits = model(x)
                all_outputs.append(logits.detach())
                all_targets.append(y.detach())

        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        # compute accuracy
        test_acc = accuracy(
            all_outputs,
            all_targets,
        )
        epoch_results["test_acc"] = test_acc.item()

        all_results.append(epoch_results)

    results = pd.DataFrame(all_results)
    results.to_csv(args.exp_dir / "results.csv", index=False)

    return np.max(results["test_acc"].values)


def run(trial):
    args = parse_args()
    args.lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    args.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    args.num_epochs = trial.suggest_categorical("num_epochs", [10, 20, 30, 40, 50])
    args.exp_dir = (
        args.exp_dir / f"lr_{args.lr}_bs_{args.batch_size}_ne_{args.num_epochs}"
    )

    return train(args)


if __name__ == "__main__":
    import optuna

    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///optuna.db",
        study_name="optuna_example",
        load_if_exists=True,
    )
    study.optimize(run, n_trials=10)
