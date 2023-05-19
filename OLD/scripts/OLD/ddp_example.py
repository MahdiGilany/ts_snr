# append path one level up
import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

import argparse
import os
import sys
import tempfile
from urllib.parse import urlparse

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm as tqdm_


def tqdm(iter, *args, **kwargs):
    if dist.get_rank() == 0:
        return tqdm_(iter, *args, **kwargs)
    else:
        return iter


from torch.nn.parallel import DistributedDataParallel as DDP


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch DDP Example")
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--checkpoint_file", type=str, default="")
    return parser.parse_args()


def run(rank, args, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    from src.data.registry import create_dataset

    dataset = create_dataset("exact_patches_sl_all_centers_balanced_ndl")
    from torch.utils.data.distributed import DistributedSampler

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    test_dataset = create_dataset(
        "exact_patches_sl_all_centers_balanced_ndl", split="test"
    )
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, sampler=test_sampler
    )

    from src.modeling.registry import create_model

    model = create_model("resnet18").to(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        if rank == 0:
            print(f"Starting epoch {epoch}")

        model.train()
        loss = 0
        correct = 0
        total = 0
        for data, _, target, *_ in tqdm(loader, desc=f"Epoch {epoch}"):
            data = data.to(rank)
            target = target.to(rank).long()
            optimizer.zero_grad()
            output = model(data)
            loss = nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)
            loss += loss.item()

        if rank == 0:
            print(
                f"Train epoch {epoch} with accuracy {100 * correct / total:.2f} and loss {loss / total:.5f}"
            )

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, _, target, *_ in tqdm(test_loader):
                data = data.to(rank)
                target = target.to(rank).long()
                output = model(data)
                test_loss += nn.functional.cross_entropy(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100.0 * correct / len(test_loader.dataset)

        if rank == 0:
            print(
                f"Test epoch {epoch} with accuracy {accuracy:.2f} and loss {test_loss:.5f}"
            )
    dist.destroy_process_group()


def main():
    import torch.multiprocessing as mp

    args = parse_args()
    mp.spawn(
        run,
        nprocs=2,
        args=(
            args,
            2,
        ),
    )


if __name__ == "__main__":
    main()
