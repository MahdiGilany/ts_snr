import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
import argparse
import torch
from src.data.registry import create_dataset
from src.modeling.registry import create_model
from torchmetrics import Accuracy
from tqdm import tqdm
from pathlib import Path
import time
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on a dataset")
    parser.add_argument("--workdir", type=str, default="workdir")
    parser.add_argument(
        "--dataset_name", type=str, default="exact_patches_sl_all_centers_balanced_ndl"
    )
    parser.add_argument("--backbone_name", type=str, default="resnet10")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--project", type=str, default="default")
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--num_workers", type=int, default=4)

    return parser.parse_args()


def train(args):
    train_ds = create_dataset(args.dataset_name, "train")
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_ds = create_dataset(args.dataset_name, "val")
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_ds = create_dataset(args.dataset_name, "test")
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = create_model(args.backbone_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    model.to("cuda")

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch}")
        model.train()
        acc = Accuracy(average="macro", num_classes=2).to("cuda")
        times = {"fetch": [], "forward": [], "backward": [], "update": []}
        start_time = time.time()

        for batch in tqdm(train_dl, desc="Training"):

            fetch_finish = time.time()
            x, y, *_ = batch
            x = x.to("cuda")
            y = y.to("cuda")
            y_hat = model(x)
            loss = criterion(y_hat, y)
            forward_finish = time.time()
            optimizer.zero_grad()
            loss.backward()
            backward_finish = time.time()
            optimizer.step()
            update_finish = time.time()
            acc(y_hat, y)

            times["fetch"].append(fetch_finish - start_time)
            times["forward"].append(forward_finish - fetch_finish)
            times["backward"].append(backward_finish - forward_finish)
            times["update"].append(update_finish - backward_finish)

            start_time = time.time()

        print(f"Train accuracy: {acc.compute()}")
        print(f"Fetch time: {np.mean(times['fetch'])}")
        print(f"Forward time: {np.mean(times['forward'])}")
        print(f"Backward time: {np.mean(times['backward'])}")
        print(f"Update time: {np.mean(times['update'])}")
        acc.reset()

        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_dl, desc="Validation"):
                x, y, *_ = batch
                x = x.to("cuda")
                y = y.to("cuda")
                y_hat = model(x)
                loss = criterion(y_hat, y)
                acc(y_hat, y)
            print(f"Validation accuracy: {acc.compute()}")
            acc.reset()

            for batch in tqdm(test_dl, desc="Test"):
                x, y, *_ = batch
                x = x.to("cuda")
                y = y.to("cuda")
                y_hat = model(x)
                loss = criterion(y_hat, y)
                acc(y_hat, y)
            print(f"Test accuracy: {acc.compute()}")
            acc.reset()


if __name__ == "__main__":
    args = parse_args()
    train(args)
