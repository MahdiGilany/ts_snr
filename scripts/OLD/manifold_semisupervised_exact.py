from pathlib import Path
import argparse
import json
import os
import random
import signal
import sys
import time
import urllib
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# from dataloader_cifar_ws import cifar_dataset
import torch
import pdb

# from preactresnet_simCLR import SupCEResNet  # Can use resnet_simCLR
import wandb
import numpy as np
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
from src.layers.losses import *
from src.layers.losses.sup_con_loss import *

# from autoaugment import CIFAR10Policy

# append path one level up
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

import rich
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="%H:%M:%S",
    handlers=[rich.logging.RichHandler()],
)


def get_arguments():
    parser = argparse.ArgumentParser(description="Evaluate a pretrained model on CIFAR")
    # Data
    parser.add_argument(
        "--data-dir",
        default="/raid/home/fahimehf/Codes/Self-Semi/datasets/cifar10/cifar-10-python/",
        type=Path,
        help="path to dataset",
    )
    parser.add_argument(
        "--noise-file",
        default="/raid/home/fahimehf/Codes/Self-Semi/datasets/cifar10_coisy_02",
        type=Path,
        help="path to noise file",
    )
    parser.add_argument("--noise-ratio", type=float, default=0.2, help="noise ratio")
    parser.add_argument("--p-threshold", type=float, default=0.3, help="noise ratio")
    # Checkpoint
    parser.add_argument("--pretrained", type=Path, help="path to pretrained model")
    parser.add_argument(
        "--exp-dir",
        default="./checkpoint/lincls/",
        type=Path,
        metavar="DIR",
        help="path to checkpoint directory",
    )
    parser.add_argument(
        "--print-freq", default=100, type=int, metavar="N", help="print frequency"
    )
    parser.add_argument("--num_class", default=10, type=int, help="number of classes")
    # Model
    parser.add_argument("--arch", type=str, default="resnet18")

    # Optim
    parser.add_argument(
        "--epochs",
        default=300,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--warmup-epoch",
        default=10,
        type=int,
        metavar="N",
        help="number of warmup epochs",
    )
    parser.add_argument(
        "--interval",
        default=10,
        type=int,
        metavar="N",
        help="number of interval to apply GMM",
    )
    parser.add_argument(
        "--batch-size", default=64, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument(
        "--lr-backbone",
        default=0.002,
        type=float,
        metavar="LR",
        help="backbone base learning rate",
    )
    parser.add_argument(
        "--lr-head",
        default=0.2,
        type=float,
        metavar="LR",
        help="classifier base learning rate",
    )
    parser.add_argument(
        "--drops",
        default=0.005,
        type=float,
        metavar="peercentage of zero out mixed up features",
    )
    parser.add_argument(
        "--weight-decay", default=5e-4, type=float, metavar="W", help="weight decay"
    )
    parser.add_argument(
        "--lambda_u", default=100, type=float, help="weight for unsupervised loss"
    )  # 30 50 100
    parser.add_argument(
        "--lambda_c", default=1, type=float, help="weight for contrastive loss"
    )  # 0.025 0.5 0.5
    parser.add_argument(
        "--lambda_ce",
        default=1,
        type=float,
        help="weight for embedding contrastive loss",
    )
    parser.add_argument(
        "--lambda_ue",
        default=100,
        type=float,
        help="weight for embedding of unsupervised loss",
    )  # 10 30 50
    parser.add_argument(
        "--lambda_x", default=1, type=float, help="weight for supervised loss"
    )
    parser.add_argument(
        "--lambda_xe",
        default=1,
        type=float,
        help="weight for supervised loss for embedding",
    )
    parser.add_argument(
        "--temperature", default=0.5, type=float, help="sharpening temperature"
    )
    parser.add_argument("--alpha", default=4, type=float, help="parameter for Beta")
    parser.add_argument(
        "--weights",
        default="finetune",
        type=str,
        choices=("finetune", "freeze"),
        help="finetune or freeze resnet weights",
    )
    parser.add_argument(
        "--emb-loss",
        default="no",
        type=str,
        choices=("mse", "kl", "no"),
        help="typr of loss function",
    )
    parser.add_argument(
        "--LN-loss",
        default="RL",
        type=str,
        choices=("CE", "RL"),
        help="type of loss function for supervised part of semisupervised step",
    )
    # Running
    parser.add_argument(
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loader workers",
    )
    parser.add_argument(
        "--noise-type",
        default="sym",
        type=str,
        choices=("no", "sym", "asym"),
        help="noise type",
    )
    parser.add_argument(
        "--clustering",
        default=False,
        type=bool,
        metavar="cluster",
        help="Do clustering or not",
    )
    return parser.parse_args()


def make_weights_for_balanced_classes(images, nclasses, mode=None):
    # if mode=='label':
    #     count = [0] * nclasses
    #     for item in images:
    #         count[item[4]] += 1
    #     weight_per_class = [0.] * nclasses
    #     N = float(sum(count))
    #     for i in range(nclasses):
    #         weight_per_class[i] = N/float(count[i])    #5
    #     weight = [0] * len(images)
    #     for idx, val in enumerate(images):
    #         weight[idx] = weight_per_class[val[4]]
    #     return weight
    # else:
    count = [0] * nclasses
    for item in images:
        count[item[2]] += 1
    weight_per_class = [0.0] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[2]]
        return weight


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, gpu, loss_type="CE"):
        probs_u = torch.softmax(outputs_u, dim=1)
        # criterion_robust = NCEandRCE(num_classes=10).cuda(gpu)
        criterion_robust = SCELoss(out_type="mean", num_classes=10).cuda(gpu)

        if loss_type == "RL":
            Lx = criterion_robust(outputs_x, targets_x)
        else:
            Lx = -torch.mean(
                torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1)
            )
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu


## Unsupervised Loss coefficient adjustment
def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return float(current)


def seperate_lul_sets(args, val_loader, model, gpu, epoch=5):
    criterion_loss = nn.CrossEntropyLoss(reduction="none").cuda(gpu)
    model.eval()
    sample_loss = []
    with torch.no_grad():
        for images, target, _ in val_loader:
            _, _, output = model(images.cuda(gpu, non_blocking=True))
            loss = criterion_loss(output, target.cuda(gpu, non_blocking=True))
            sample_loss.append(loss.cpu().numpy())

    all_loss = []
    sample_loss = np.concatenate(sample_loss)
    sample_loss = (sample_loss - sample_loss.min()) / (
        sample_loss.max() - sample_loss.min()
    )

    input_loss = sample_loss.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, max_iter=20, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:, gmm.means_.argmin()]
    pred = prob > args.p_threshold
    f, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.hist(
        sample_loss[np.where(prob > args.p_threshold)],
        bins=1000,
        color="green",
        alpha=0.5,
        ec="green",
        label="Clean Data",
    )
    ax.hist(
        sample_loss[np.where(prob <= args.p_threshold)],
        bins=1000,
        color="blue",
        alpha=0.5,
        ec="blue",
        label="Noisy Data",
    )
    ax.legend(loc="upper right")
    name_losscluster = "gmmloss_" + str(epoch) + ".png"
    plt.savefig(args.exp_dir / name_losscluster)
    return pred, prob


def zero_out(emb, drops):
    nn = emb.numel()
    mm = int(round(nn * drops))
    indices = np.random.choice(
        nn, mm, replace=False
    )  # alternative: indices = torch.randperm(n)[:m]
    emb = emb.contiguous()
    emb.flatten()[indices] = 0
    return emb


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
    wandb.init(project="SSL-Final-Experiments")
    gpu = 0
    resume = False
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
    print(" ".join(sys.argv))
    # print(" ".join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    # chekpoint = torch.load('/raid/home/fahimehf/Codes/Self-Semi/SupContrast-master/save/SupCon/cifar10_models/SimCLR_cifar10_resnet18_lr_0.5_decay_0.0001_bsz_1024_temp_0.5_trial_0_cosine_warm/last.pth')
    # chekpoint = torch.load(
    #     "/raid/home/fahimehf/Codes/Self-Semi/SupContrast-master/save/SupCon_preact/cifar10_models/SimCLR_cifar10_resnet18_lr_0.5_decay_0.0001_bsz_1024_temp_0.5_trial_1_cosine_warm_1500/last.pth"
    # )

    # sd = {}
    # for ke in chekpoint["model"]:
    #     nk = ke.replace("module.", "")
    #     sd[nk] = chekpoint["model"][ke]
    # model = SupCEResNet(args.arch, num_classes=10)
    # model.load_state_dict(sd, strict=False)
    # model.cuda(gpu)

    # if args.weights == "freeze":
    #     backbone.requires_grad_(False)
    #     head.requires_grad_(True)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)

    criterion_warmup = SCELoss(out_type="mean", num_classes=10).cuda(gpu)
    criterion = SemiLoss()
    contrastive_criterion = SupConLoss().cuda(gpu)

    # criterion_loss = NCEandRCE_mean(num_classes=10).cuda(gpu)
    # if args.emb_loss =='mse':
    #     criterion_emb = nn.MSELoss().cuda(gpu)
    # else:
    #     criterion_emb = nn.KLDivLoss().cuda(gpu)

    # encoder_p = []
    # classifier_p = []
    # for name, param in model.named_parameters():
    #     if "encoder" in name:
    #         encoder_p.append(param)
    #     else:
    #         classifier_p.append(param)

    from src.modelling.registry import resnet10_pretrained

    logging.info("Loading model")
    net = resnet10_pretrained("resnet10_exact_patches_sl_tuffc_ndl_v0").cuda()
    backbone, fc, net = separate_resnet10_to_head_and_backbone(net)
    optimizer1 = construct_optim(args, fc, backbone)

    param_groups = [dict(params=classifier_p, lr=args.lr_head)]
    if args.weights == "finetune":
        param_groups.append(
            dict(params=model.encoder.parameters(), lr=args.lr_backbone)
        )
    optimizer = optim.SGD(param_groups, 0, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # automatically resume from checkpoint if it exists
    if (args.exp_dir / "checkpoint.pth").is_file():
        ckpt = torch.load(args.exp_dir / "checkpoint.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        best_acc = ckpt["best_acc"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        # scheduler.load_state_dict(ckpt["scheduler"])
        best_acc_test = ckpt["best_acc"]
        resume = True

    else:
        start_epoch = 0
        best_acc_test = 0
        best_acc = argparse.Namespace(top1=0, top5=0)

    transform_weak_C10 = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_strong_C10 = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    val_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_dataset = cifar_dataset(
        dataset="cifar10",
        noise_mode=args.noise_type,
        r=args.noise_ratio,
        root_dir=args.data_dir,
        transform=transform_weak_C10,
        mode="all_sup",
        noise_file=args.noise_file,
    )
    val_dataset = cifar_dataset(
        dataset="cifar10",
        noise_mode=args.noise_type,
        r=args.noise_ratio,
        root_dir=args.data_dir,
        transform=val_transforms,
        mode="all_sup",
        noise_file=args.noise_file,
    )
    test_dataset = cifar_dataset(
        dataset="cifar10",
        noise_mode="no",
        r=args.noise_ratio,
        root_dir=args.data_dir,
        transform=val_transforms,
        mode="test",
    )

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        # pin_memory=True,
        shuffle=True,
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
    )

    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        # train
        if epoch < args.warmup_epoch:
            # train_sampler.set_epoch(epoch)
            for step, (images, target, _) in enumerate(
                train_loader, start=epoch * len(train_loader)
            ):
                _, _, output = model(images.cuda(gpu, non_blocking=True))
                loss = criterion_warmup(
                    output, target.cuda(gpu, non_blocking=True), onehot=True
                )  # , onehot=True
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % args.print_freq == 0:
                    pg = optimizer.param_groups
                    lr_head = pg[0]["lr"]
                    lr_backbone = pg[1]["lr"] if len(pg) == 2 else 0
                    stats = dict(
                        epoch=epoch,
                        step=step,
                        lr_backbone=lr_backbone,
                        lr_head=lr_head,
                        loss=loss.item(),
                        time=int(time.time() - start_time),
                    )
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
                    wandb.log({"TrainLoss": loss.item(), "custom_step": epoch})
                    wandb.log({"LR_head": lr_head, "custom_step": epoch})
                    wandb.log({"LR_backbone": lr_backbone, "custom_step": epoch})
        else:
            if (
                epoch == args.warmup_epoch
                or (epoch % args.interval == 0 and args.clustering)
                or resume
            ):

                pred, prob = seperate_lul_sets(args, val_loader, model, gpu, epoch)
                labeled_dataset = cifar_dataset(
                    dataset="cifar10",
                    noise_mode=args.noise_type,
                    r=args.noise_ratio,
                    root_dir=args.data_dir,
                    transform=transform_weak_C10,
                    mode="labeled",
                    noise_file=args.noise_file,
                    pred=pred,
                    probability=prob,
                    transform_st=transform_strong_C10,
                )
                unlabeled_dataset = cifar_dataset(
                    dataset="cifar10",
                    noise_mode=args.noise_type,
                    r=args.noise_ratio,
                    root_dir=args.data_dir,
                    transform=transform_weak_C10,
                    mode="unlabeled",
                    noise_file=args.noise_file,
                    pred=pred,
                    transform_st=transform_strong_C10,
                )

                # For unbalanced dataset we create a weighted sampler
                # weights = make_weights_for_balanced_classes(labeled_dataset, 10)
                # weights = torch.Tensor(weights)
                # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

                kwargs = dict(
                    batch_size=args.batch_size,
                    shuffle=True,
                    # sampler=sampler,
                    num_workers=args.workers,
                    # pin_memory=True,
                )
                labeled_trainloader = torch.utils.data.DataLoader(
                    labeled_dataset, **kwargs
                )
                unlabeled_trainloader = torch.utils.data.DataLoader(
                    unlabeled_dataset, **kwargs
                )

            unlabeled_train_iter = iter(unlabeled_trainloader)
            num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
            for step, (
                inputs_x,
                inputs_x2,
                inputs_x3,
                inputs_x4,
                labels_x,
                w_x,
            ) in enumerate(labeled_trainloader, start=epoch * len(labeled_trainloader)):
                try:
                    inputs_u, inputs_u2, inputs_u3, inputs_u4 = next(
                        unlabeled_train_iter
                    )
                except:
                    unlabeled_train_iter = iter(unlabeled_trainloader)
                    inputs_u, inputs_u2, inputs_u3, inputs_u4 = next(
                        unlabeled_train_iter
                    )
                batch_size = inputs_x.size(0)

                # Transform label to one-hot
                labels_x = torch.zeros(batch_size, args.num_class).scatter_(
                    1, labels_x.view(-1, 1), 1
                )
                w_x = w_x.view(-1, 1).type(torch.FloatTensor)

                w_x = w_x.cuda(gpu, non_blocking=True)
                labels_x = labels_x.cuda(gpu, non_blocking=True)
                # inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

                with torch.no_grad():

                    # Label co-guessing of unlabeled samples
                    _, _, outputs_u11 = model(inputs_u.cuda(gpu, non_blocking=True))
                    _, _, outputs_u12 = model(inputs_u2.cuda(gpu, non_blocking=True))
                    # _,_, outputs_u21 = net2(inputs_u.cuda(gpu, non_blocking=True))
                    # _,_, outputs_u22 = net2(inputs_u2.cuda(gpu, non_blocking=True))

                    pu = (
                        torch.softmax(outputs_u11, dim=1)
                        + torch.softmax(outputs_u12, dim=1)
                    ) / 2  # + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4
                    ptu = pu ** (1 / args.temperature)  # temparature sharpening

                    targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
                    targets_u = targets_u.detach()

                    ## Label refinement
                    _, _, outputs_x = model(inputs_x.cuda(gpu, non_blocking=True))
                    _, _, outputs_x2 = model(inputs_x2.cuda(gpu, non_blocking=True))

                    px = (
                        torch.softmax(outputs_x, dim=1)
                        + torch.softmax(outputs_x2, dim=1)
                    ) / 2
                    px = w_x * labels_x + (1 - w_x) * px
                    ptx = px ** (1 / args.temperature)  # temparature sharpening

                    targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
                    targets_x = targets_x.detach()

                emb_x3, _, _ = model(inputs_x3.cuda(gpu, non_blocking=True))
                emb_x4, _, _ = model(inputs_x4.cuda(gpu, non_blocking=True))
                emb_u3, f1, _ = model(inputs_u3.cuda(gpu, non_blocking=True))
                emb_u4, f2, _ = model(inputs_u4.cuda(gpu, non_blocking=True))

                f1 = F.normalize(f1, dim=1)
                f2 = F.normalize(f2, dim=1)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                loss_simCLR = contrastive_criterion(features)

                ## Mixup
                l = np.random.beta(args.alpha, args.alpha)
                l = max(l, 1 - l)
                all_inputs = torch.cat(
                    [inputs_x3, inputs_x4, inputs_u3, inputs_u4], dim=0
                )
                all_targets = torch.cat(
                    [targets_x, targets_x, targets_u, targets_u], dim=0
                )
                all_emb = torch.cat([emb_x3, emb_x4, emb_u3, emb_u4], dim=0)
                idx = torch.randperm(all_inputs.size(0))

                input_a, input_b = all_inputs, all_inputs[idx]
                target_a, target_b = all_targets, all_targets[idx]
                emb_a, emb_b = all_emb, all_emb[idx]

                mixed_input = l * input_a + (1 - l) * input_b
                mixed_target = l * target_a + (1 - l) * target_b
                mixed_emb = l * emb_a + (1 - l) * emb_b

                _, f3, logits = model(mixed_input.cuda(gpu, non_blocking=True))
                _, f4, logits_emb = model(mixed_emb.cuda(gpu, non_blocking=True), "emb")
                augemb = zero_out(mixed_emb.clone(), drops=args.drops)
                _, _, logits_augemb = model(augemb.cuda(gpu, non_blocking=True), "emb")

                ## Unsupervised Contrastive Loss for embedding space
                # f3 = F.normalize(f3, dim=1)
                # f4 = F.normalize(f4, dim=1)
                # features_emb = torch.cat([f3.unsqueeze(1), f4.unsqueeze(1)], dim=1)
                # loss_simCLR_emb = contrastive_criterion(features_emb)

                logits_x = logits[: batch_size * 2]
                logits_u = logits[batch_size * 2 :]
                logits_emx = logits_emb[: batch_size * 2]
                logits_emu = logits_emb[batch_size * 2 :]
                logits_augemx = logits_augemb[: batch_size * 2]
                logits_augemu = logits_augemb[batch_size * 2 :]
                ## Combined Loss
                Lx, Lu = criterion(
                    logits_x,
                    mixed_target[: batch_size * 2],
                    logits_u,
                    mixed_target[batch_size * 2 :],
                    gpu,
                    args.LN_loss,
                )
                Lxe, Lue = criterion(
                    logits_emx,
                    mixed_target[: batch_size * 2],
                    logits_emu,
                    mixed_target[batch_size * 2 :],
                    gpu,
                    args.LN_loss,
                )
                Lxae, Luae = criterion(
                    logits_augemx,
                    mixed_target[: batch_size * 2],
                    logits_augemu,
                    mixed_target[batch_size * 2 :],
                    gpu,
                    args.LN_loss,
                )

                lamb = args.lambda_u * linear_rampup(
                    epoch + step / num_iter, args.warmup_epoch
                )
                lambe = args.lambda_ue * linear_rampup(
                    epoch + step / num_iter, args.warmup_epoch
                )

                ## Regularization
                prior = torch.ones(args.num_class) / args.num_class
                prior = prior.cuda()
                pred_mean = torch.softmax(logits, dim=1).mean(0)
                penalty = torch.sum(prior * torch.log(prior / pred_mean))

                ## Total Loss
                loss = (
                    args.lambda_x * Lx
                    + args.lambda_xe * Lxe
                    + lamb * Lu
                    + lambe * Lue
                    + args.lambda_c * loss_simCLR
                    + penalty
                    + args.lambda_xe * Lxae
                    + lambe * Luae
                )  # args.lambda_ce*loss_simCLR_emb +

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % args.print_freq == 0:

                    pg = optimizer.param_groups
                    lr_head = pg[0]["lr"]
                    lr_backbone = pg[1]["lr"] if len(pg) == 2 else 0
                    stats = dict(
                        epoch=epoch,
                        step=step,
                        lr_backbone=lr_backbone,
                        lr_head=lr_head,
                        loss=loss.item(),
                        time=int(time.time() - start_time),
                    )
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
                    wandb.log({"LX": Lx, "custom_step": epoch})
                    wandb.log({"Lu": Lu, "custom_step": epoch})
                    wandb.log({"LXe": Lxe, "custom_step": epoch})
                    wandb.log({"Lue": Lue, "custom_step": epoch})
                    wandb.log({"Co": loss_simCLR, "custom_step": epoch})
                    wandb.log({"Aug-Emb": Lxae, "custom_step": epoch})
                    wandb.log({"TrainLoss": loss.item(), "custom_step": epoch})
                    wandb.log({"LR_head": lr_head, "custom_step": epoch})
                    wandb.log({"LR_backbone": lr_backbone, "custom_step": epoch})

        # evaluate
        model.eval()
        save_best = False
        top1 = AverageMeter("Acc@1")
        top5 = AverageMeter("Acc@5")
        with torch.no_grad():
            for images, target, _ in val_loader:
                _, _, output = model(images.cuda(gpu, non_blocking=True))
                acc1, acc5 = accuracy(
                    output, target.cuda(gpu, non_blocking=True), topk=(1, 5)
                )
                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))
        if best_acc.top1 < top1.avg:
            save_best = True
        best_acc.top1 = max(best_acc.top1, top1.avg)
        best_acc.top5 = max(best_acc.top5, top5.avg)
        wandb.log({"Top1": top1.avg, "custom_step": epoch})
        wandb.log({"Top5": top5.avg, "custom_step": epoch})
        wandb.log({"BestTop1": best_acc.top1, "custom_step": epoch})
        wandb.log({"BestTop5": best_acc.top5, "custom_step": epoch})

        stats = dict(
            epoch=epoch,
            acc1=top1.avg,
            acc5=top5.avg,
            best_acc1=best_acc.top1,
            best_acc5=best_acc.top5,
        )
        print(json.dumps(stats))
        print(json.dumps(stats), file=stats_file)

        scheduler.step()

        test_top1 = AverageMeter("Acc@1")
        test_top5 = AverageMeter("Acc@5")
        with torch.no_grad():
            for images, target in test_loader:
                _, _, output = model(images.cuda(gpu, non_blocking=True))
                test_acc1, test_acc5 = accuracy(
                    output, target.cuda(gpu, non_blocking=True), topk=(1, 5)
                )
                test_top1.update(test_acc1[0].item(), images.size(0))
                test_top5.update(test_acc5[0].item(), images.size(0))
        best_acc_test = max(best_acc_test, test_top1.avg)

        state = dict(
            epoch=epoch + 1,
            best_acc=best_acc,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            best_acc_test=best_acc_test,
        )
        if save_best:
            if epoch < args.warmup_epoch:
                torch.save(state, args.exp_dir / "best_checkpoint_w.pth")
                save_best = False
            elif epoch == args.warmup_epoch:
                torch.save(state, args.exp_dir / "best_checkpoint_warmup.pth")
                save_best = False
            else:
                torch.save(state, args.exp_dir / "best_checkpoint.pth")
                save_best = False
        elif epoch == args.warmup_epoch - 1:
            torch.save(state, args.exp_dir / "checkpoint_before_semi.pth")
        else:
            torch.save(state, args.exp_dir / "checkpoint.pth")

        if best_acc_test < test_top1.avg:
            torch.save(state, args.exp_dir / "besttest_checkpoint.pth")
        wandb.log({"Test Top1": test_top1.avg, "custom_step": epoch})
        wandb.log({"Test Top5": test_top5.avg, "custom_step": epoch})

    torch.save(model.state_dict(), args.exp_dir / "last_resnet18.pth")

    # wandb.finish()


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()
