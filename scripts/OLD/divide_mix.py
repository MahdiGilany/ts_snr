from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import wandb
from typing import Literal
from functools import partial

# append path one level up
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

NUM_CLASSES = 2


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR Training")
    parser.add_argument("--batch_size", default=64, type=int, help="train batchsize")
    parser.add_argument(
        "--lr",
        "--learning_rate",
        default=0.02,
        type=float,
        help="initial learning rate",
    )
    parser.add_argument(
        "--lr_factor_for_backbone",
        default=0.1,
        type=float,
        help="learning rate factor for backbone",
    )
    parser.add_argument(
        "--weight_decay",
        default=5e-4,
        type=float,
        help="weight decay (default: 5e-4)",
    )
    parser.add_argument(
        "--optimizer",
        default="sgd",
        type=str,
        help="optimizer",
        choices=["sgd", "adam"],
    )
    parser.add_argument(
        "--scheduler",
        default="none",
        choices=["none", "cosine"],
        help="optionally specifiy the learning rate scheduler",
        type=str,
    )
    parser.add_argument(
        "--alpha",
        default=4,
        type=float,
        help="parameter for Beta distribution for MixMatch",
    )
    parser.add_argument(
        "--lambda_u", default=25, type=float, help="weight for unsupervised loss"
    )
    parser.add_argument(
        "--p_threshold", default=0.5, type=float, help="clean probability threshold"
    )
    parser.add_argument("--T", default=0.5, type=float, help="sharpening temperature")
    parser.add_argument("--num_epochs", default=300, type=int)
    parser.add_argument("--r", default=0.5, type=float, help="noise ratio")
    parser.add_argument("--id", default="")
    parser.add_argument("--seed", default=123)
    parser.add_argument("--device", default=0, type=int)

    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    return args


# set up logging
import logging

logging.basicConfig(level=logging.INFO)
wandb.init(project="exact_dividemix")

from src.data.semisupervised_dataloader_factory import LoaderFactory

# Training
def train(
    args, epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader
):
    criterion = SemiLoss()
    net.train()
    net2.eval()  # fix one network and train the other

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(
        tqdm(labeled_trainloader, f"EPOCH {epoch} TRAINING")
    ):
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, NUM_CLASSES).scatter_(
            1, labels_x.view(-1, 1), 1
        )
        w_x = w_x.view(-1, 1).type(torch.FloatTensor)

        inputs_x, inputs_x2, labels_x, w_x = (
            inputs_x.cuda(),
            inputs_x2.cuda(),
            labels_x.cuda(),
            w_x.cuda(),
        )
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)

            pu = (
                torch.softmax(outputs_u11, dim=1)
                + torch.softmax(outputs_u12, dim=1)
                + torch.softmax(outputs_u21, dim=1)
                + torch.softmax(outputs_u22, dim=1)
            ) / 4
            ptu = pu ** (1 / args.T)  # temparature sharpening

            targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
            targets_u = targets_u.detach()

            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)

            # breakpoint()

            px = (
                torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)
            ) / 2
            px = w_x * labels_x + (1 - w_x) * px
            ptx = px ** (1 / args.T)  # temparature sharpening

            targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
            targets_x = targets_x.detach()

        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1 - l)

        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        logits = net(mixed_input)
        logits_x = logits[: batch_size * 2]
        logits_u = logits[batch_size * 2 :]

        Lx, Lu, lamb = criterion(
            logits_x,
            mixed_target[: batch_size * 2],
            logits_u,
            mixed_target[batch_size * 2 :],
            epoch + batch_idx / num_iter,
            10,
        )

        # regularization
        prior = torch.ones(NUM_CLASSES) / NUM_CLASSES
        prior = prior.cuda()
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        loss = Lx + lamb * Lu + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            wandb.log(
                {
                    "loss": loss.item(),
                }
            )


# WE DONT NEED WARMUP
# def warmup(epoch, net, optimizer, dataloader):
#     net.train()
#     num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
#     CEloss = nn.CrossEntropyLoss()
#     for batch_idx, (inputs, labels, path) in enumerate(dataloader):
#         inputs, labels = inputs.cuda(), labels.cuda()
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = CEloss(outputs, labels)
#         if (
#             args.noise_mode == "asym"
#         ):  # penalize confident prediction for asymmetric noise
#             penalty = conf_penalty(outputs)
#             L = loss + penalty
#         elif args.noise_mode == "sym":
#             L = loss
#         L.backward()
#         optimizer.step()
#         sys.stdout.write("\r")
#         sys.stdout.write(
#             "%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f"
#             % (
#                 args.dataset,
#                 args.r,
#                 args.noise_mode,
#                 epoch,
#                 args.num_epochs,
#                 batch_idx + 1,
#                 num_iter,
#                 loss.item(),
#             )
#         )
#         sys.stdout.flush()


def test(epoch, test_loader, net1, net2):
    logging.info(f"Testing -- epoch {epoch}")
    net1.eval()
    net2.eval()
    from src.utils.metrics import OutputCollector

    oc = OutputCollector()
    with torch.no_grad():
        for batch_idx, (inputs, targets, info) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            outputs = outputs1 + outputs2
            _, predicted = torch.max(outputs, 1)

            oc.collect_batch(
                {
                    "logits": outputs,
                    "targets": targets,
                    "core_specifier": info["core_specifier"],
                    "position": info["position"],
                    "prostate_intersection": info["prostate_intersection"],
                    "needle_intersection": info["needle_intersection"],
                }
            )

    out = oc.compute()
    from torchmetrics.functional import auroc

    auroc_ = auroc(out["logits"], out["targets"], num_classes=2, average="macro")
    return {
        "auroc": auroc_.item(),
        "positive_pred_ratio": (out["logits"][:, 1] > 0.5).float().mean().item(),
    }


def eval_train(model, all_loss, eval_loader):
    model.eval()
    # losses = torch.zeros(len(eval_loader.dataset))
    losses = []
    indices = []
    CE = nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(
            tqdm(eval_loader, desc="Compute loss")
        ):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = CE(outputs, targets)
            losses.append(loss)
            indices.append(index)
            # for b in range(inputs.size(0)):
            #    losses[index[b]] = loss[b]
    losses = torch.cat(losses).cpu().numpy()
    indices = torch.cat(indices).cpu().numpy()
    # sort the losses in ascending order of indices
    losses = losses[np.argsort(indices)]

    # normalize the loss
    losses = (losses - losses.min()) / (losses.max() - losses.min())
    all_loss.append(losses)

    if (
        args.r == 0.9
    ):  # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1, 1)
    else:
        input_loss = losses.reshape(-1, 1)

    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    logging.info("Fitting GMM")
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:, gmm.means_.argmin()]
    return prob, losses


def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, linear_rampup(epoch, warm_up)


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


def separate_resnet10_to_head_and_backbone(resnet10):
    fc = resnet10.fc
    backbone = resnet10
    backbone.fc = nn.Identity()
    model = nn.Sequential(backbone, fc)

    return backbone, fc, model


def construct_optim(args, head, backbone):
    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            [
                {"params": head.parameters(), "lr": args.lr},
                {
                    "params": backbone.parameters(),
                    "lr": args.lr * args.lr_factor_for_backbone,
                },
            ],
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adam":
        optimizer = optim.Adam(
            [
                {"params": head.parameters(), "lr": args.lr},
                {
                    "params": backbone.parameters(),
                    "lr": args.lr * args.lr_factor_for_backbone,
                },
            ],
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError("Unknown optimizer")
    return optimizer


def main(args):

    torch.cuda.set_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logging.info("Starting training")
    from src.modeling.registry import resnet10_pretrained

    logging.info("Loading data")
    loader = LoaderFactory(args.batch_size, 4, debug=args.debug)

    logging.info("Loading model")
    net1 = resnet10_pretrained("resnet10_exact_patches_sl_tuffc_ndl_v0").cuda()
    backbone, fc, net1 = separate_resnet10_to_head_and_backbone(net1)
    optimizer1 = construct_optim(args, fc, backbone)

    net2 = resnet10_pretrained("resnet10_exact_patches_sl_tuffc_ndl_v0").cuda()
    backbone, fc, net2 = separate_resnet10_to_head_and_backbone(net2)
    optimizer2 = construct_optim(args, fc, backbone)

    cudnn.benchmark = True

    if args.scheduler == "none":
        scheduler1 = None
        scheduler2 = None
    elif args.scheduler == "cosine":
        scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, args.num_epochs)
        scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, args.num_epochs)
    else:
        raise NotImplementedError(f"Scheduler `{args.scheduler}` not implemented")

    loss_history = [[], []]  # save the history of losses from two networks

    for epoch in range(args.num_epochs + 1):

        test_loaders = loader.run("test")
        for name, loader_ in test_loaders.items():
            metrics = test(epoch, loader_, net1, net2)
            wandb.log({f"{name}_{k}": v for k, v in metrics.items()})

        eval_loader = loader.run("eval_train")

        prob1, losses1 = eval_train(net1, loss_history[0], eval_loader)
        prob2, losses2 = eval_train(net2, loss_history[1], eval_loader)

        # plot wandb histogram of losses
        wandb.log({"losses1": wandb.Histogram(losses1)})
        wandb.log({"losses2": wandb.Histogram(losses2)})
        # plot ratio of clean and noisy samples
        wandb.log({"clean_samples": (prob1 > args.p_threshold).sum() / len(prob1)})
        wandb.log({"noisy_samples": (prob1 < args.p_threshold).sum() / len(prob1)})

        pred1 = prob1 > args.p_threshold
        pred2 = prob2 > args.p_threshold

        print("Train Net1")
        labeled_trainloader, unlabeled_trainloader = loader.run(
            "train", pred2, prob2
        )  # co-divide
        train(
            args,
            epoch,
            net1,
            net2,
            optimizer1,
            labeled_trainloader,
            unlabeled_trainloader,
        )  # train net1

        print("\nTrain Net2")
        labeled_trainloader, unlabeled_trainloader = loader.run(
            "train", pred1, prob1
        )  # co-divide
        train(
            args,
            epoch,
            net2,
            net1,
            optimizer2,
            labeled_trainloader,
            unlabeled_trainloader,
        )  # train net2

        if scheduler1 is not None:
            scheduler1.step()
        if scheduler2 is not None:
            scheduler2.step()

        # log learning rate
        if scheduler1 is not None:
            wandb.log({"lr": scheduler1.get_last_lr()[-1]})
        else:
            wandb.log({"lr": args.lr})

        # test(epoch, test_loader, net1, net2)


if __name__ == "__main__":
    args = parse_args()

    import rich

    rich.print(args._get_kwargs())

    main(args)
