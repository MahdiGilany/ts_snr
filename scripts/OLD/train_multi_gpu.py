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


def parse_args():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Train on multiple GPUs")
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--exp_dir", type=Path, default="./test/default")
    parser.add_argument("--dataset_name", type=str, default="cifar10")
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--logger", type=str, default=None)

    return parser.parse_args()


def gather_all_tensors(tensor, world_size):
    gathered_tensors = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)


class Logger:
    def __init__(self, args):
        if not dist.get_rank() == 0:
            return
        self.args = args
        if args.logger == "wandb":
            import wandb

            project, name = args.exp_dir.parent.name, args.exp_dir.name
            wandb.init(
                project=project, name=name, config=args.__dict__, dir=args.exp_dir
            )

        import json

        # with open(args.exp_dir / "args.json", "w") as f:
        #     json.dump(args.__dict__, f)

    def log(self, name, value):
        if not dist.get_rank() == 0:
            return

        if self.args.logger == "wandb":
            import wandb

            wandb.log({name: value})
        # else:
        #    print(f"{name}: {value}")

    def log_dict(self, d):
        [self.log(name, value) for name, value in d.items()]


def train(rank, args, scores={}):
    dist.init_process_group("nccl", rank=rank, world_size=args.num_gpus)

    # setup experiment directory
    exp_dir: Path = args.exp_dir
    if rank == 0:
        print(f"Saving experiment to {args.exp_dir}")
        exp_dir: Path = args.exp_dir
        if not exp_dir.is_dir():
            os.makedirs(exp_dir)
        main_ckpt = exp_dir / "main_ckpt.pth"
    dist.barrier()

    # setup dataloaders
    train_ds = create_dataset(args.dataset_name, split="train")
    test_ds = create_dataset(args.dataset_name, split="test")
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    train_sampler = DistributedSampler(train_ds, num_replicas=args.num_gpus, rank=rank)
    test_sampler = DistributedSampler(
        test_ds, num_replicas=args.num_gpus, rank=rank, shuffle=False
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, sampler=test_sampler)

    # setup model
    # model = create_model(args.model_name).to(rank)
    from torchvision.models import resnet18

    model = resnet18(num_classes=10).to(rank)
    from torch.nn.parallel import DistributedDataParallel as DDP

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])

    # setup optimizer
    from torch.optim import Adam

    optimizer = Adam(model.parameters(), lr=args.lr)

    if (exp_dir / "main_ckpt.pth").is_file():
        if rank == 0:
            print("Loading checkpoint...")
        ckpt = torch.load(exp_dir / "main_ckpt.pth", map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        rng = ckpt["rng"]
        torch.set_rng_state(rng)
    else:
        start_epoch = 1
        torch.manual_seed(0)

    # setup logging
    logger = Logger(args)

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.num_epochs):

        # start epoch
        if rank == 0:
            print(f"Epoch {epoch} / {args.num_epochs}")
        model.train()
        all_outputs = []
        all_targets = []
        loss = 0
        iterator = tqdm(train_dl, desc="Training") if rank == 0 else train_dl
        for i, batch in enumerate(iterator):
            x, y = batch
            x, y = x.to(rank), y.to(rank)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if i % 100 == 0:
                logger.log_dict({"loss": loss.item()})
            all_outputs.append(logits.detach())
            all_targets.append(y.detach())

        # gather all outputs and targets
        all_outputs = gather_all_tensors(torch.cat(all_outputs), args.num_gpus)
        all_targets = gather_all_tensors(torch.cat(all_targets), args.num_gpus)

        # compute accuracy
        train_acc = accuracy(all_outputs, all_targets)
        logger.log_dict({"train_acc": train_acc})

        # evaluate
        model.eval()
        all_outputs = []
        all_targets = []
        iterator = tqdm(test_dl, desc="Evaluating") if rank == 0 else test_dl
        for i, batch in enumerate(iterator):
            with torch.no_grad():
                x, y = batch
                x, y = x.to(rank), y.to(rank)
                logits = model(x)
                all_outputs.append(logits.detach())
                all_targets.append(y.detach())

        # gather all outputs and targets
        all_outputs = gather_all_tensors(torch.cat(all_outputs), args.num_gpus)
        all_targets = gather_all_tensors(torch.cat(all_targets), args.num_gpus)

        # compute accuracy
        test_acc = accuracy(
            all_outputs,
            all_targets,
        )
        logger.log_dict({"test_acc": test_acc})

        # save checkpoint
        if rank == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "rng": torch.get_rng_state(),
                },
                main_ckpt,
            )
        if rank == 0:
            print(
                f"Epoch {epoch} / {args.num_epochs} done with train_acc: {train_acc.item()}, test_acc: {test_acc.item()}"
            )

        dist.barrier()

    if rank == 0:
        if wandb.run is not None:
            wandb.run.finish()
        scores = {"test_acc": test_acc.item()}

    dist.destroy_process_group()


def run(trial):
    args = parse_args()
    args.lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    args.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    args.num_epochs = trial.suggest_categorical("num_epochs", [10, 20, 30, 40, 50])

    import torch.multiprocessing as mp

    scores = {}
    mp.spawn(train, args=(args, scores), nprocs=args.num_gpus, join=True)
    outcome = scores["test_acc"]
    return outcome


if __name__ == "__main__":
    import optuna

    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///optuna.db",
        study_name="optuna_example",
        load_if_exists=True,
    )
    study.optimize(run, n_trials=10)
