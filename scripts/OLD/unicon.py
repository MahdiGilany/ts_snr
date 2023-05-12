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


class LabeledDatasetForUnicon(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        filter_rate,
        clean_probabilities: np.ndarray,
    ):
        self.base_dataset = dataset

        self.clean_or_not = clean_or_not
        self.clean_probabilities = clean_probabilities
        assert len(clean_or_not) == len(clean_probabilities)
        assert len(clean_or_not) == len(dataset)
        self.indices = clean_or_not.nonzero()[0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        index = self.indices[index]
        (x1, x2), y, *_ = self.base_dataset[index]
        return x1, x2, y, self.clean_probabilities[index]


class UnlabeledDatasetForUnicon(torch.utils.data.Dataset):
    def __init__(
        self, dataset, clean_or_not: np.ndarray, clean_probabilities: np.ndarray
    ):
        self.base_dataset = dataset
        self.clean_or_not = clean_or_not
        self.clean_probabilities = clean_probabilities
        assert len(clean_or_not) == len(clean_probabilities)
        assert len(clean_or_not) == len(dataset)
        self.indices = (1 - clean_or_not).nonzero()[0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        index = self.indices[index]
        (x1, x2), y, *_ = self.base_dataset[index]
        return x1, x2


class DatasetForLossComputation(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.base_dataset = dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        x, y, *_ = self.base_dataset[index]
        return x, y, index


class LoaderFactory:
    def __init__(self, batch_size, num_workers, debug=False):
        self.batch_size = batch_size
        self.num_workers = num_workers
        from src.data.registry import (
            exact_patches_sl_tuffc_prostate,
            exact_patches_sl_tuffc_ndl,
        )

        self.base_dataset_train = exact_patches_sl_tuffc_prostate(split="train")
        self.base_dataset_train_with_augs = exact_patches_sl_tuffc_prostate(
            split="train"
        )
        self.base_dataset_test = exact_patches_sl_tuffc_ndl(split="test")
        self.base_dataset_test_noisy = exact_patches_sl_tuffc_prostate(split="test")
        self.base_dataset_val = exact_patches_sl_tuffc_ndl(split="val")
        self.base_dataset_val_noisy = exact_patches_sl_tuffc_prostate(split="val")

        from src.data.exact.transforms import (
            TransformV3,
            TensorImageAugmentation,
            MultiTransform,
            UltrasoundArrayAugmentation,
        )

        basic_transform = TransformV3()
        self.base_dataset_train.patch_transform = basic_transform
        self.base_dataset_test.patch_transform = basic_transform
        self.base_dataset_test_noisy.patch_transform = basic_transform

        augmentations_weak = TransformV3(
            tensor_transform=TensorImageAugmentation(
                random_erasing=False,
                random_invert=True,
                random_horizontal_flip=True,
                random_vertical_flip=True,
                random_resized_crop=True,
                random_resized_crop_scale=(0.9, 1.0),
                random_affine_translation=[0, 0],
            )
        )
        augmentations_strong = TransformV3(
            tensor_transform=TensorImageAugmentation(
                random_erasing=True,
                random_invert=True,
                random_horizontal_flip=True,
                random_vertical_flip=True,
                random_affine_translation=[0.2, 0.2],
                random_affine_rotation=10,
            ),
            us_augmentation=UltrasoundArrayAugmentation(),
        )

        augmentations = MultiTransform(
            augmentations_weak,
            augmentations_weak,
            augmentations_strong,
            augmentations_strong,
        )
        self.base_dataset_train_with_augs.patch_transform = augmentations

        if debug:
            self.base_dataset_train = torch.utils.data.Subset(
                self.base_dataset_train, range(100)
            )
            self.base_dataset_train_with_augs = torch.utils.data.Subset(
                self.base_dataset_train_with_augs, range(100)
            )
            self.base_dataset_test = torch.utils.data.Subset(
                self.base_dataset_test, range(100)
            )

    def run(self, mode, pred=[], prob=[]):
        if mode == "train":
            unlabeled_dataset = UnlabeledDatasetForMixMatch(
                self.base_dataset_train_with_augs, pred, prob
            )
            labeled_dataset = LabeledDatasetForMixMatch(
                self.base_dataset_train_with_augs, pred, prob
            )
            unlabeled_trainloader = torch.utils.data.DataLoader(
                unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            labeled_trainloader = torch.utils.data.DataLoader(
                labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            return labeled_trainloader, unlabeled_trainloader
        elif mode == "test":
            out = {}
            dataset = self.base_dataset_test
            testloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size * 20,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            out["test"] = testloader
            dataset = self.base_dataset_test_noisy
            testloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size * 20,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            out["test_noisy"] = testloader
            dataset = self.base_dataset_val
            testloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size * 20,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            out["val"] = testloader
            dataset = self.base_dataset_val_noisy
            testloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size * 20,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            out["val_noisy"] = testloader
            return out

        elif mode == "test_noisy":
            dataset = self.base_dataset_test_noisy
            testloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size * 20,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            return testloader
        elif mode == "eval_train":
            dataset = DatasetForLossComputation(self.base_dataset_train)
            trainloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size * 20,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            return trainloader
        else:
            raise NotImplementedError


# Training
# SSL-Training
def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader):
    net2.eval()  # Freeze one network and train the other
    net.train()

    criterion = SemiLoss()

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1

    ## Loss statistics
    loss_x = 0
    loss_u = 0
    loss_scl = 0
    loss_ucl = 0

    for batch_idx, (
        inputs_x,
        inputs_x2,
        inputs_x3,
        inputs_x4,
        labels_x,
        w_x,
    ) in enumerate(labeled_trainloader):
        try:
            inputs_u, inputs_u2, inputs_u3, inputs_u4 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, inputs_u3, inputs_u4 = unlabeled_train_iter.next()

        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(
            1, labels_x.view(-1, 1), 1
        )
        w_x = w_x.view(-1, 1).type(torch.FloatTensor)

        inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x = (
            inputs_x.cuda(),
            inputs_x2.cuda(),
            inputs_x3.cuda(),
            inputs_x4.cuda(),
            labels_x.cuda(),
            w_x.cuda(),
        )
        inputs_u, inputs_u2, inputs_u3, inputs_u4 = (
            inputs_u.cuda(),
            inputs_u2.cuda(),
            inputs_u3.cuda(),
            inputs_u4.cuda(),
        )

        with torch.no_grad():
            # Label co-guessing of unlabeled samples
            _, outputs_u11 = net(inputs_u)
            _, outputs_u12 = net(inputs_u2)
            _, outputs_u21 = net2(inputs_u)
            _, outputs_u22 = net2(inputs_u2)

            ## Pseudo-label
            pu = (
                torch.softmax(outputs_u11, dim=1)
                + torch.softmax(outputs_u12, dim=1)
                + torch.softmax(outputs_u21, dim=1)
                + torch.softmax(outputs_u22, dim=1)
            ) / 4

            ptu = pu ** (1 / args.T)  ## Temparature Sharpening

            targets_u = ptu / ptu.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

            ## Label refinement
            _, outputs_x = net(inputs_x)
            _, outputs_x2 = net(inputs_x2)

            px = (
                torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)
            ) / 2

            px = w_x * labels_x + (1 - w_x) * px
            ptx = px ** (1 / args.T)  ## Temparature sharpening

            targets_x = ptx / ptx.sum(dim=1, keepdim=True)
            targets_x = targets_x.detach()

        ## Unsupervised Contrastive Loss
        f1, _ = net(inputs_u3)
        f2, _ = net(inputs_u4)
        f1 = F.normalize(f1, dim=1)
        f2 = F.normalize(f2, dim=1)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        # loss_simCLR = contrastive_criterion(features)

        # MixMatch
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1 - l)
        all_inputs = torch.cat([inputs_x3, inputs_x4, inputs_u3, inputs_u4], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        ## Mixup
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        _, logits = net(mixed_input)
        logits_x = logits[: batch_size * 2]
        logits_u = logits[batch_size * 2 :]

        ## Combined Loss
        Lx, Lu, lamb = criterion(
            logits_x,
            mixed_target[: batch_size * 2],
            logits_u,
            mixed_target[batch_size * 2 :],
            epoch + batch_idx / num_iter,
            warm_up,
        )

        ## Regularization
        prior = torch.ones(args.num_class) / args.num_class
        prior = prior.cuda()
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        ## Total Loss
        loss = Lx
        loss = loss + lamb * Lu
        # loss = loss + args.lambda_c * loss_simCLR
        loss = loss + penalty

        ## Accumulate Loss
        loss_x += Lx.item()
        loss_u += Lu.item()
        # loss_ucl += loss_simCLR.item()

        # Compute gradient and Do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


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


# KL divergence
def kl_divergence(p, q):
    return (p * ((p + 1e-10) / (q + 1e-10)).log()).sum(dim=1)


## Jensen-Shannon Divergence
class Jensen_Shannon(nn.Module):
    def __init__(self):
        super(Jensen_Shannon, self).__init__()
        pass

    def forward(self, p, q):
        m = (p + q) / 2
        return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


## Calculate JSD
def Calculate_JSD(model1, model2, num_samples, eval_loader):
    JS_dist = Jensen_Shannon()
    JSD = torch.zeros(num_samples)

    for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        batch_size = inputs.size()[0]

        ## Get outputs of both network
        with torch.no_grad():
            out1 = torch.nn.Softmax(dim=1).cuda()(model1(inputs)[1])
            out2 = torch.nn.Softmax(dim=1).cuda()(model2(inputs)[1])

        ## Get the Prediction
        out = (out1 + out2) / 2

        ## Divergence clculator to record the diff. between ground truth and output prob. dist.
        dist = JS_dist(out, F.one_hot(targets, num_classes=args.num_class))
        JSD[int(batch_idx * batch_size) : int((batch_idx + 1) * batch_size)] = dist

    return JSD


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
    net1 = resnet10_pretrained("resnet10_exact_patches_sl_tuffc_ndl_v0")
    net2 = resnet10_pretrained("resnet10_exact_patches_sl_tuffc_ndl_v1")
    net1.cuda()
    net2.cuda()
    cudnn.benchmark = True

    logging.info("Loading optimizer")
    if args.optimizer == "sgd":
        optim_factory = partial(
            optim.SGD, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
        )
    elif args.optimizer == "adam":
        optim_factory = partial(optim.Adam, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    optimizer1 = optim_factory(net1.parameters())
    optimizer2 = optim_factory(net2.parameters())

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
