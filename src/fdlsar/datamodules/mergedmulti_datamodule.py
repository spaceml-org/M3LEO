from __future__ import annotations

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from .components.mergeddataset import MergedMultiDataset


class MergedMultiDataModule(LightningDataModule):
    def __init__(
        self,
        datasets_spec,
        split_files,
        transforms=None,
        batch_size: int = 16,
        num_workers: int = 1,
        pin_memory: bool = False,
        prefetch_factor: int = 4,
        limit_pct_train=1.0,
        limit_pct_val=1.0,
        limit_pct_test=1.0,
        get_ids=False,
    ):
        super().__init__()

        if num_workers == 0:
            prefetch_factor = None
            self.persistent_workers = False
        else:
            prefetch_factor = prefetch_factor
            self.persistent_workers = True

        self.save_hyperparameters(logger=False)
        self.hparams.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory

        if isinstance(split_files, str):
            split_files = [split_files]

        self.split_files = split_files

        if limit_pct_train < 0 or limit_pct_train > 1:
            raise ValueError("limit_pct_train must be in [0,1]")

        if limit_pct_val < 0 or limit_pct_val > 1:
            raise ValueError("limit_pct_val must be in [0,1]")

        if limit_pct_test < 0 or limit_pct_test > 1:
            raise ValueError("limit_pct_test must be in [0,1]")

        self.train_dataset = MergedMultiDataset(
            datasets_spec=datasets_spec,
            split_files=split_files,
            split="train",
            transforms=transforms,
            limit_pct=limit_pct_train,
            get_ids=get_ids,
        )

        self.test_dataset = MergedMultiDataset(
            datasets_spec=datasets_spec,
            split_files=split_files,
            split="test",
            transforms=transforms,
            limit_pct=limit_pct_test,
            get_ids=get_ids,
        )

        self.val_dataset = MergedMultiDataset(
            datasets_spec=datasets_spec,
            split_files=split_files,
            split="val",
            transforms=transforms,
            limit_pct=limit_pct_val,
            get_ids=get_ids,
        )

    def prepare_data(self):
        self.train_dataset.prepare_data()
        self.test_dataset.prepare_data()
        self.val_dataset.prepare_data()

    def setup(self, stage):
        self.train_dataset.setup(stage)
        self.test_dataset.setup(stage)
        self.val_dataset.setup(stage)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.hparams.prefetch_factor,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.hparams.prefetch_factor,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.hparams.prefetch_factor,
        )
