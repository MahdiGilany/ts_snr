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

logging.basicConfig(level=logging.INFO)


# append path one level up
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

from src.layers.losses.isomax import IsoMaxPlusLossFirstPart, IsoMaxPlusLossSecondPart
from src.data.registry import list_datasets


class IsoMaxPlusLossFirstPart(nn.Module):
    """This part replaces the model classifier output layer nn.Linear()"""

    def __init__(self, num_features, num_classes, temperature=1.0):
        super(IsoMaxPlusLossFirstPart, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.temperature = temperature
        self.prototypes = nn.Parameter(torch.Tensor(num_classes, num_features))
        self.distance_scale = nn.Parameter(torch.Tensor(1))
        nn.init.normal_(self.prototypes, mean=0.0, std=1.0)
        nn.init.constant_(self.distance_scale, 1.0)

    def forward(self, features):
        distances = torch.abs(self.distance_scale) * torch.cdist(
            F.normalize(features),
            F.normalize(self.prototypes),
            p=2.0,
            compute_mode="donot_use_mm_for_euclid_dist",
        )
        logits = -distances
        # The temperature may be calibrated after training to improve uncertainty estimation.
        return logits / self.temperature


def parse_args():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_name", type=str, choices=list_datasets(), required=True
    )
    parser.add_argument(
        "--eval_dataset_name",
        type=str,
        choices=[None] + list_datasets(),
        required=False,
        default=None,
    )
    parser.add_argument("--model_name", type=str, default="resnet10_feature_extractor")
    parser.add_argument("--model_weights", type=str, required=True)
    parser.add_argument("--exp_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--backbone_lr", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--start_training_backbone_at_epoch", type=int, default=1)

    parser.add_argument("--loss", type=str, choices=["ce", "isomaxplus"], default="ce")

    return parser.parse_args()


def main():
    args = parse_args()

    # make the experiment directory
    # add a timestamp to the directory name to avoid overwriting
    args.exp_dir = args.exp_dir / time.strftime("%Y-%m-%d")
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    num_existing_dirs = len(os.listdir(args.exp_dir))
    args.exp_dir = args.exp_dir / str(num_existing_dirs)
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    (args.exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (args.exp_dir / "train").mkdir(parents=True, exist_ok=True)
    (args.exp_dir / "val").mkdir(parents=True, exist_ok=True)
    (args.exp_dir / "test").mkdir(parents=True, exist_ok=True)

    # save args to a file
    with open(args.exp_dir / "args.yaml", "w") as f:
        import yaml

        yaml.dump(vars(args), f)

    torch.manual_seed(args.seed)

    # load the dataset
    print("Loading the dataset...")
    from src.data.registry import create_dataset

    train_dataset = create_dataset(args.dataset_name, "train")
    if args.eval_dataset_name is not None:
        val_dataset = create_dataset(args.eval_dataset_name, "val")
        test_dataset = create_dataset(args.eval_dataset_name, "test")
    else:
        val_dataset = create_dataset(args.dataset_name, "val")
        test_dataset = create_dataset(args.dataset_name, "test")

    # load the model
    print("Loading the model...")
    from src.modeling.registry import create_model

    backbone = create_model(args.model_name)
    print(backbone.load_state_dict(torch.load(args.model_weights)))
    if args.loss == "ce":
        classifier = nn.Linear(backbone.features_dim, train_dataset.num_classes)
        criterion = nn.CrossEntropyLoss()
    elif args.loss == "isomaxplus":
        classifier = IsoMaxPlusLossFirstPart(
            backbone.features_dim, train_dataset.num_classes
        )
        criterion = IsoMaxPlusLossSecondPart()
    else:
        raise NotImplementedError
    model = torch.nn.Sequential(backbone, classifier).to(args.device)
    criterion.to(args.device)

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

    # create the optimizer
    print("Creating the optimizers...")
    from torch.optim import Adam

    backbone_opt = Adam(
        backbone.parameters(), lr=args.backbone_lr, weight_decay=args.weight_decay
    )
    clf_opt = Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
            args.exp_dir / "checkpoints" / f"best_{score:.4f}.ckpt",
            save_last=True,
            num_to_keep=2,
        )
    )

    collector = OutputCollector()
    for epoch in range(1, args.num_epochs + 1):
        collector.reset()
        epoch_outputs = {}
        model.train()
        for batch in (pbar := tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)):
            patch, label, metadata = batch
            patch = patch.to(args.device)
            label = label.to(args.device)
            backbone_opt.zero_grad()
            clf_opt.zero_grad()
            logits = model(patch)
            loss = criterion(logits, label)
            loss.backward()
            if epoch >= args.start_training_backbone_at_epoch:
                backbone_opt.step()
            clf_opt.step()
            collector.collect_batch(
                {
                    "label": label,
                    "cancer_prob": logits[:, 1],
                    "core_specifier": metadata["core_specifier"],
                }
            )
        output = collector.compute()
        output = pd.DataFrame(output)
        output.to_csv(
            args.exp_dir / "train" / f"{epoch}.csv",
        )
        epoch_outputs["train_auroc"] = roc_auc_score(
            output["label"], output["cancer_prob"]
        )

        for split, loader in zip(["val", "test"], [val_loader, test_loader]):
            collector.reset()
            model.eval()
            with torch.no_grad():
                for batch in (pbar := tqdm(loader, desc=f"{split}", leave=False)):
                    patch, label, metadata = batch
                    patch = patch.to(args.device)
                    label = label.to(args.device)
                    logits = model(patch)
                    probs = logits.softmax(-1)
                    loss = criterion(logits, label)
                    collector.collect_batch(
                        {
                            "label": label,
                            "cancer_prob": probs[:, 1],
                            "core_specifier": metadata["core_specifier"],
                        }
                    )
                output = collector.compute()
                output = pd.DataFrame(output)
                output.to_csv(args.exp_dir / split / f"{epoch}.csv")
                epoch_outputs[f"{split}_auroc"] = roc_auc_score(
                    output["label"], output["cancer_prob"]
                )
                epoch_outputs[f"{split}_positive_pred_ratio"] = (
                    output["cancer_prob"] > 0.5
                ).mean()
                epoch_outputs[f"{split}_positive_label_ratio"] = output["label"].mean()

        print(epoch_outputs)
        epoch_outputs["epoch"] = epoch
        if epoch == 1:
            table = pd.DataFrame([epoch_outputs])
        else:
            table = pd.concat([table, pd.DataFrame([epoch_outputs])])
        table.to_csv(args.exp_dir / "results.csv", index=False)
        score_improvement_monitor.update(epoch_outputs["val_auroc"])
        if early_stopping_monitor.should_early_stop():
            print("Early stopping triggered on epoch {epoch}".format(epoch=epoch))
            return


if __name__ == "__main__":
    main()
