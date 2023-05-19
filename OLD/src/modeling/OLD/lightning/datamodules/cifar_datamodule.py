import torch
import pytorch_lightning as pl

from src.utils.datamodule.dataloding import data_dir
from src.utils.datamodule.transforms import TransformNaturalImages

from dataclasses import dataclass
from typing import Callable, Optional, Literal
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

LABEL_NAMES = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

@dataclass
class CIFAR10DMConfig:
    _target_: str = "src.lightning.datamodules.cifar_datamodule.CIFARDataModule"
    
    data_dir: str = data_dir("cifar10")
    batch_size: int = 64
    split_seed: int = 0
    test_as_val: bool = False
    transform: Optional[Callable] = TransformNaturalImages()
    download: bool = True


# create a dataloader for torchvision cifar10 dataset
class CIFARDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = data_dir("cifar10"),
        batch_size: int = 64,
        split_seed: int = 0,
        test_as_val: bool = False,
        transform: Optional[Callable] = TransformNaturalImages(),
        download: bool = True,
        ):
        super().__init__()
        self.data_dir = data_dir
        self.batch = batch_size
        self.test_as_val = test_as_val
        
        self.train_dataset = CIFAR10(root = self.data_dir, train=True, download=download, transform=transform)
        self.test_dataset = CIFAR10(root = self.data_dir, train=False, download=download, transform=transform)
        
        # split CIFAR10 into train and validation set
        self.train_indx, self.val_indx = train_test_split(range(len(self.train_dataset)), test_size=0.2, random_state=split_seed)
        self.test_sampler = None
        
        
    def _make_loader(self, set_name: Literal["train", "val", "test"]):
        if set_name == "train":
            dataset = self.train_dataset
            sampler = self.train_indx
        elif set_name == "val":
            dataset = self.train_dataset
            sampler = self.val_indx
        elif set_name == "test":
            dataset = self.test_dataset
            sampler = self.test_sampler
        
        return DataLoader(
            dataset,
            batch_size=self.batch,
            sampler=sampler
            )
            
    def train_dataloader(self):
        return self._make_loader("train")
    
    def val_dataloader(self):
        if self.test_as_val:
            return [self._make_loader("val"), self._make_loader("test")]
        return self._make_loader("val")
    
    def test_dataloader(self):
        return self._make_loader("test")

    