from __future__ import annotations

import numpy as np
from loguru import logger
from torch.utils.data import Dataset

from . import multidataset


class MergedMultiDataset(Dataset):
    """
    a dataset made by combining several datasets that must contain common tiles

    """

    def __init__(
        self,
        datasets_spec,
        split_files,
        split,
        transforms=None,
        limit_pct=1.0,
        get_ids=False,
    ):
        """
        datasets_spec, split, transforms, limit_pct: like in multidaset

        split_files: a list of csvs file containing the split definitions (like for different AOIs)

        """
        self.split_files = split_files

        # holds a multidataset per split file
        self.multidatasets = [
            multidataset.MultiDataset(
                datasets_spec=datasets_spec,
                split_file=split_file,
                split=split,
                transforms=transforms,
                limit_pct=limit_pct,
                get_ids=get_ids,
            )
            for split_file in split_files
        ]

        self.idxs = []
        for i, d in enumerate(self.multidatasets):
            self.idxs += [f"{i}::{idx}" for idx in range(len(d))]

        self.idxs = np.random.permutation(self.idxs)

        logger.info("---------------------------")
        logger.info("---------------------------")
        logger.info(
            f"merged {len(self.multidatasets)} datasets, with a total of {len(self)} items"
        )
        logger.info(f"datasets sizes are {[len(d) for d in self.multidatasets]}")
        logger.info("---------------------------")
        logger.info("---------------------------")

    def prepare_data(self):
        """apply preprocessing step on all datasets"""
        for d in self.multidatasets:
            d.prepare_data()

    def setup(self, stage):
        """apply setup step on all datasets"""
        for d in self.multidatasets:
            d.setup(stage)

    def __len__(self):
        return sum(len(d) for d in self.multidatasets)

    def __repr__(self):
        return self.multidatasets[0].__repr__() + f"::{len(self.multidatasets)}_aois"

    def __getitem__(self, idx):
        """
        simply selects any item and retrieves it from the corresponding multidataset
        """

        dataset_id, item_idx = self.idxs[idx].split("::")
        dataset_id = int(dataset_id)
        item_idx = int(item_idx)

        return self.multidatasets[dataset_id][item_idx]
