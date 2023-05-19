from abc import ABC, abstractclassmethod, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Mapping, Union, overload, Sequence
import pytorch_lightning as pl
import os
from typing import Literal, List, Tuple, Optional, cast, Dict
import torch
from src.data.exact.core import Core
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from ...data.exact.splits import (
    HasProstateMaskFilter,
    InvolvementThresholdFilter,
    Splits,
    SplitsConfig,
)
from src.data import data_dir

from ...data.exact.dataset import (
    PatchesDataset,
    PatchesGroupedByCoreDataset,
    PatchesDatasetNew,
)
from ...data.exact.transforms import (
    TensorAugsConfig,
    Transform,
    TransformConfig,
    target_transform,
)

from warnings import warn

from omegaconf import ListConfig

# from src.data.exact.transforms import TransformV2, TransformConfig, PrebuiltConfigs
from src.data.exact.core import PatchViewConfig
from src.data.exact.splits import SplitsConfig

import logging

log = logging.getLogger(__name__)


def DEFAULT_LABEL_TRANSFORM(label):
    return torch.tensor(label).long()


ANATOMICAL_LOCATIONS = [
    "LML",
    "RBL",
    "LMM",
    "RMM",
    "LBL",
    "LAM",
    "RAM",
    "RML",
    "LBM",
    "RAL",
    "RBM",
    "LAL",
]
ANATOMICAL_LOCATIONS2IDX = {name: idx for idx, name in enumerate(ANATOMICAL_LOCATIONS)}
CENTERS = ["UVA", "CRCEO", "JH", "PCC", "PMCC"]
CENTER2IDX = {center: i for i, center in enumerate(CENTERS)}


def DEFAULT_METADATA_TRANSFORM(metadata):
    metadata["center_idx"] = torch.tensor(CENTER2IDX[metadata["center"]]).long()
    metadata["anatomical_loc_idx"] = torch.tensor(
        ANATOMICAL_LOCATIONS2IDX[metadata["loc"]]
    ).long()
    return metadata


def wrap_to_list(obj):
    if not isinstance(obj, Sequence):
        return [obj]
    else:
        return obj


from multiprocessing import cpu_count


@dataclass
class LoaderConfig:
    batch_size: int = 64
    num_workers: int = 8  # field(default_factory=cpu_count)
    balance_classes_train: bool = True
    train_strategy_ddp: bool = False
    balance_grades: bool = False


@dataclass
class ExactPatchDMConfig:

    _target_: str = "src.lightning.datamodules.exact_datamodule.ExactPatchDataModule"

    root: str = field(default_factory=data_dir)
    loader_config: LoaderConfig = LoaderConfig(
        batch_size=64,
    )
    minimum_involvement: float = 0.4
    splits: SplitsConfig = SplitsConfig()
    patch_view_config: PatchViewConfig = PatchViewConfig()
    patch_transform_train: TransformConfig = TransformConfig()
    patch_transform_eval: TransformConfig = TransformConfig(
        tensor_transform=None, us_augmentation=None
    )


@dataclass
class ExactPatchDMConfigSL(ExactPatchDMConfig):

    _target_: str = __name__ + ".PatchDataModuleForSelfSupervisedLearning"

    patch_transform_train: TransformConfig = TransformConfig(
        tensor_transform=None, us_augmentation=None
    )

    patch_view_config: PatchViewConfig = PatchViewConfig(
        prostate_region_only=True, needle_region_only=True
    )


from src.data.exact.dataset import CoreMixingOptions


@dataclass
class ExactCoreDMConfigWithMixing(ExactPatchDMConfig):
    _target_: str = __name__ + ".PatchesConcatenatedFromCoresDataModuleWithMixing"

    patch_transform_train: TransformConfig = TransformConfig(
        tensor_transform=None, us_augmentation=None
    )

    patch_view_config: PatchViewConfig = PatchViewConfig(
        prostate_region_only=False, needle_region_only=True
    )

    mixing_options: CoreMixingOptions = CoreMixingOptions()


def _build_dataloader(dataset, config: LoaderConfig, train: bool):
    from ...data.sampler import WeightedDistributedSampler, get_weighted_sampler
    from torch.utils.data import DistributedSampler

    sampler = None

    # Non distributed
    if not config.train_strategy_ddp:

        if train:
            if config.balance_classes_train:
                sampler = get_weighted_sampler(dataset, balance_grades=config.balance_grades)

    # Distributed
    else:
        if train:
            if config.balance_classes_train:
                sampler = WeightedDistributedSampler(
                    dataset,
                    dataset.labels,
                )
            else:
                sampler = DistributedSampler(dataset, shuffle=True)
        else:
            sampler = DistributedSampler(dataset, shuffle=False)

    shuffle = sampler is None and train

    return DataLoader(
        dataset,
        config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.num_workers,
    )


class ExactPatchDataModule(pl.LightningDataModule, ABC):
    def __init__(
        self,
        root: str = data_dir(),
        loader_config: LoaderConfig = LoaderConfig(),
        minimum_involvement: float = 0.4,
        splits: Splits = Splits(),
        patch_view_config: PatchViewConfig = PatchViewConfig(),
        patch_transform_train: Optional[Callable] = None,
        patch_transform_eval: Optional[Callable] = None,
    ):
        super().__init__()

        self.root = root
        self.minimum_involvement = minimum_involvement
        self.loader_config = loader_config
        self.splits = splits
        self.patch_view_config = patch_view_config
        self.train_transform = patch_transform_train
        self.eval_transform = patch_transform_eval
        self.val_ds = None
        self.test_ds = None

    @abstractmethod
    def _make_dataset_impl(self, cores: List[str], train: bool):
        ...

    def _make_dataset(self, cores, train):
        if isinstance(cores, dict):
            return {
                center: self._make_dataset_impl(cores, train)
                for center, cores in cores.items()
            }

        else:
            return self._make_dataset_impl(cores, train)

    def _make_loader(self, dataset, train: bool):
        if isinstance(dataset, dict):
            return [
                self._make_loader(_dataset, train) for name, _dataset in dataset.items()
            ]
        else:
            return _build_dataloader(dataset, self.loader_config, train)

    def _get_splits(self):
        splits = self.splits
        if tau := self.minimum_involvement:
            splits.apply_filters(InvolvementThresholdFilter(tau))
        if self.patch_view_config.prostate_region_only:
            splits.apply_filters(HasProstateMaskFilter())

        return splits

    def eval_loaders_as_dict(self):
        from itertools import chain

        assert (
            self.val_ds is not None and self.test_ds is not None
        ), "Call setup() first. "

        if isinstance(self.val_ds, dict):
            return {
                name: loader
                for name, loader in zip(
                    chain(
                        map(lambda name: f"val_{name}", self.val_ds.keys()),
                        map(lambda name: f"test_{name}", self.test_ds.keys()),
                    ),
                    chain(self.val_dataloader(), self.test_dataloader()),
                )
            }

        else:
            return {"val": self.val_dataloader(), "test": self.test_dataloader()}


class PatchDataModuleForSupervisedLearning(ExactPatchDataModule):
    def _make_dataset_impl(self, cores: List[str], train: bool):
        from ...data.exact.dataset import PatchesDatasetNew

        return PatchesDatasetNew(
            Core.default_data_dir(),
            cores,
            self.patch_view_config,
            self.train_transform if train else self.eval_transform,
            DEFAULT_LABEL_TRANSFORM,
            DEFAULT_METADATA_TRANSFORM,
        )

    def setup(self, stage=None):
        log.info("Setting up datamodule")

        log.info("Setting up cohort splits")
        log.info(f"Using centers {self.splits.cohort_specifier}")

        self.splits = self._get_splits()

        log.info("Setting up pre-processing transforms")
        # self.train_transform = TransformV2(self.config.transform_config)
        #
        ## eval transform does not use augmentations
        # self.eval_transform = TransformV2(
        #    TransformConfig(
        #        out_size=self.config.transform_config.out_size,
        #        norm_config=self.config.transform_config.norm_config,
        #        tensor_augs_config=None,
        #        us_augs_config=None,
        #    )
        # )

        log.info("Setting up datasets")
        self.train_ds = self._make_dataset(
            self.splits.get_train(merge_centers=True), train=True
        )
        self.val_ds = self._make_dataset(self.splits.get_val(), train=False)
        self.test_ds = self._make_dataset(self.splits.get_test(), train=False)

    def train_dataloader(self):
        return self._make_loader(self.train_ds, True)

    def val_dataloader(self):
        if self.splits.test_as_val:
            return wrap_to_list(self._make_loader(self.val_ds, False)) + wrap_to_list(
                self._make_loader(self.test_ds, False)
            )
        return self._make_loader(self.val_ds, False)

    def test_dataloader(self):
        return self._make_loader(self.test_ds, False)


class PatchDataModuleForSelfSupervisedLearning(ExactPatchDataModule):
    def _make_dataset_impl(self, cores: List[str], train: bool):

        return PatchesDatasetNew(
            Core.default_data_dir(),
            cores,
            self.patch_view_config,
            self.augmentations,
            DEFAULT_LABEL_TRANSFORM,
            DEFAULT_METADATA_TRANSFORM,
        )

    def setup(self, stage=None):

        log.info("Setting up datamodule")
        log.info("Setting up cohort splits")
        log.info(f"Using centers {self.splits.cohort_specifier}")

        self.splits = self._get_splits()

        log.info("Setting up augmentation transforms")
        from ...data.exact.transforms import MultiTransform

        self.augmentations = MultiTransform(
            self.train_transform,
            self.train_transform,
        )

        if not stage == "eval":
            self.train_ds = self._make_dataset(
                self.splits.get_train(merge_centers=True), True
            )

        self.val_ds = self._make_dataset(self.splits.get_val(merge_centers=True), False)

        self.test_ds = self._make_dataset(
            self.splits.get_test(merge_centers=True), False
        )

    def train_dataloader(self):
        return self._make_loader(self.train_ds, True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.splits.test_as_val:
            return wrap_to_list(self._make_loader(self.val_ds, False)) + wrap_to_list(
                self._make_loader(self.test_ds, False)
            )
        return self._make_loader(self.val_ds, False)

    def test_dataloader(self):
        return self._make_loader(self.test_ds, False)


class PatchesConcatenatedFromCoresDataModule(ExactPatchDataModule):
    def _make_dataset_impl(self, cores, train):
        from ...data.exact.dataset import PatchesGroupedByCoreDataset

        return PatchesGroupedByCoreDataset(
            Core.default_data_dir(),
            cores,
            self.patch_view_config,
            self.train_transform if train else self.eval_transform,
            DEFAULT_LABEL_TRANSFORM,
            DEFAULT_METADATA_TRANSFORM,
        )

    def setup(self, stage=None):
        # if self.loader_config.batch_size != 1:
        # raise ValueError(
        # f"Concatenating cores only works with batch size 1 (1 core = 1 batch)"
        # )

        log.info("Setting up datamodule")

        log.info("Setting up cohort splits")
        log.info(f"Using centers {self.splits.cohort_specifier}")

        self.splits = self._get_splits()

        log.info("Setting up pre-processing transforms")

        log.info("Setting up datasets")
        if not stage == "eval":
            self.train_ds = self._make_dataset(
                self.splits.get_train(merge_centers=True), train=True
            )
        self.val_ds = self._make_dataset(self.splits.get_val(), train=False)
        self.test_ds = self._make_dataset(self.splits.get_test(), train=False)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._make_loader(self.train_ds, True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.splits.test_as_val:
            return wrap_to_list(self._make_loader(self.val_ds, False)) + wrap_to_list(
                self._make_loader(self.test_ds, False)
            )
        return self._make_loader(self.val_ds, False)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._make_loader(self.test_ds, False)


from ...data.exact.dataset import CoreMixingOptions, CoresDatasetWithCoreMixing


class PatchesConcatenatedFromCoresDataModuleWithMixing(
    PatchesConcatenatedFromCoresDataModule
):
    def __init__(
        self,
        root: str = data_dir(),
        loader_config: LoaderConfig = LoaderConfig(),
        minimum_involvement: float = 0.4,
        splits: Splits = Splits(),
        patch_view_config: PatchViewConfig = PatchViewConfig(),
        patch_transform_train=None,
        patch_transform_eval=None,
        mixing_options: CoreMixingOptions = CoreMixingOptions(),
    ):
        super().__init__(
            root,
            loader_config,
            minimum_involvement,
            splits,
            patch_view_config,
            patch_transform_train,
            patch_transform_eval,
        )

        self.mixing_options = mixing_options

    def _make_dataset_impl(self, cores, train):
        if train:
            return CoresDatasetWithCoreMixing(
                Core.default_data_dir(),
                cores,
                self.mixing_options,
                self.patch_view_config,
                self.train_transform,
                DEFAULT_LABEL_TRANSFORM,
            )

        else:
            return super()._make_dataset_impl(cores, train)


class PatchDataModuleForSemiSupervisedLearning(PatchDataModuleForSupervisedLearning):
    def _make_dataset_impl(self, cores: List[str], train: bool):
        import copy
    
        patch_view_config = self.patch_view_config
        if not train:
            patch_view_config = copy.deepcopy(self.patch_view_config)
            patch_view_config.needle_region_only = True
            patch_view_config.patch_strides = (1, 1)
    
        return PatchesDatasetNew(
            Core.default_data_dir(),
            cores,
            patch_view_config,
            self.train_transform if train else self.eval_transform,
            DEFAULT_LABEL_TRANSFORM,
            DEFAULT_METADATA_TRANSFORM,
        )


class PatchDataModuleForSemiSupervisedLearningDivideMix(PatchDataModuleForSelfSupervisedLearning):
    def _make_dataset_impl(self, cores: List[str], train: bool):
        import copy

        patch_view_config = self.patch_view_config
        if not train:
            patch_view_config = copy.deepcopy(self.patch_view_config)
            patch_view_config.needle_region_only = True

        return PatchesDatasetNew(
            Core.default_data_dir(),
            cores,
            patch_view_config,
            self.augmentations if train else self.eval_transform,
            DEFAULT_LABEL_TRANSFORM,
            DEFAULT_METADATA_TRANSFORM,
        )
    
    def setup(self, stage=None):
        from sklearn.model_selection import train_test_split
        
        super().setup(stage)
        
        
        self.labeled_train_idx1, self.unlabeled_train_idx1 = train_test_split(range(len(self.train_ds)),
                                                                            test_size=0.01,
                                                                            random_state=0,
                                                                            )
        self.labeled_train_idx2, self.unlabeled_train_idx2 = train_test_split(range(len(self.train_ds)),
                                                                    test_size=0.01,
                                                                    random_state=0,
                                                                    )
        
        self.train_GMM_loader = DataLoader(self.train_ds,
                                           self.loader_config.batch_size,
                                           shuffle=False,
                                           num_workers=self.loader_config.num_workers,
                                           )
        
    def train_dataloader(self, model=1):
        from ...data.sampler import WeightedDistributedSampler, get_weighted_sampler
        
        # if self.loader_config.balance_classes_train:
        #     sampler = get_weighted_sampler(self.train_ds, balance_grades=self.loader_config.balance_grades)
        
        if model == 1: 
            labeled_train_loader = DataLoader(self.train_ds,
                                            self.loader_config.batch_size,
                                            sampler=self.labeled_train_idx1,
                                            num_workers=self.loader_config.num_workers,
                                            )
            
            
            unlabeled_train_loader = DataLoader(self.train_ds,
                                                self.loader_config.batch_size,
                                                sampler=self.unlabeled_train_idx1,
                                                num_workers=self.loader_config.num_workers,
                                                )
        
        elif model == 2:
            labeled_train_loader = DataLoader(self.train_ds,
                                            self.loader_config.batch_size,
                                            sampler=self.labeled_train_idx2,
                                            num_workers=self.loader_config.num_workers,
                                            )
            
            
            unlabeled_train_loader = DataLoader(self.train_ds,
                                                self.loader_config.batch_size,
                                                sampler=self.unlabeled_train_idx2,
                                                num_workers=self.loader_config.num_workers,
                                                )
        
        loaders = {"labeled": labeled_train_loader,
                   "unlabeled": unlabeled_train_loader,
                   }
        
        return loaders
    
 
class PatchDataModuleForSemiSupervisedLearningFahimeh(PatchDataModuleForSemiSupervisedLearningDivideMix):
    def setup(self, stage=None):
        from sklearn.model_selection import train_test_split
        
        super().setup(stage)
        
        
        self.labeled_train_idx1, self.unlabeled_train_idx1 = train_test_split(range(len(self.train_ds)),
                                                                            test_size=0.01,
                                                                            random_state=0,
                                                                            )

        
        self.train_GMM_loader = DataLoader(self.train_ds,
                                           self.loader_config.batch_size,
                                           shuffle=False,
                                           num_workers=self.loader_config.num_workers,
                                           )
        
    def train_dataloader(self,):
        from ...data.sampler import WeightedDistributedSampler, get_weighted_subset_sampler
        from torch.utils.data import SubsetRandomSampler
        
        if self.loader_config.balance_classes_train:
            labeled_subset, unlabeled_subset = get_weighted_subset_sampler(self.train_ds,
                                                                           labeled_indices=self.labeled_train_idx1,
                                                                           unlabeled_indices=self.unlabeled_train_idx1,)
        else:
            labeled_subset = SubsetRandomSampler(self.labeled_train_idx1)
            unlabeled_subset = SubsetRandomSampler(self.unlabeled_train_idx1)
        
        labeled_train_loader = DataLoader(self.train_ds,
                                          self.loader_config.batch_size,
                                          sampler=labeled_subset,
                                          num_workers=self.loader_config.num_workers,
                                          )
        
        
        unlabeled_train_loader = DataLoader(self.train_ds,
                                            self.loader_config.batch_size,
                                            sampler=unlabeled_subset,
                                            num_workers=self.loader_config.num_workers,
                                            )

        loaders = {"labeled": labeled_train_loader,
                   "unlabeled": unlabeled_train_loader,
                   }
        
        return loaders


class PatchDataModuleForSemiSupervisedLearningWithCertainPreds(PatchDataModuleForSemiSupervisedLearning):
    def setup(self, stage=None):   
        super().setup(stage)
        self.train_
        self.not_shuffled_train_loader = DataLoader(self.train_ds,
                                                    self.loader_config.batch_size,
                                                    shuffle=False,
                                                    num_workers=self.loader_config.num_workers,
                                                    )
        
    def train_dataloader(self):
        return self.not_shuffled_train_loader