from __future__ import annotations

import numpy as np
from loguru import logger
from omegaconf.dictconfig import DictConfig
from torch.utils.data import Dataset

from fdlsar import utils


class MultiDataset(Dataset):
    """
    a dataset made by combining several datasets that must contain common tiles

    """

    def __init__(
        self,
        datasets_spec,
        split_file,
        split,
        transforms=None,
        limit_pct=1.0,
        get_ids=False,
    ):
        """
        datasets_spec: a dict of dicts with three keys each: 'class', 'skip_constant_channels', 'kwargs'. For instance

            datasets_spec = {
                's1grd':
                    {
                    'class': s1grd.S1grdDataset,
                    'skip_constant_channels': True
                    'kwargs': dict(year= 2020, pols = 'vh', direction = 'asc', summarize_seasons=True, normalize=True)
                    },

                'veg':
                    {
                    'class': modisveg.ModisVegDataset,
                    'skip_constant_channels': False
                    'kwargs': dict(year= 2020)
                    }
            }

        split_file: the csv file containing the split definitions
        split: one of 'train', 'test', 'val'

        transforms: a dict of functions with keys included in datasets_spec.keys()
                    or a function that

        limit_pct: float in [0,1], pct of data to use
        """

        # if not present skip_constant_channels is set to false
        for v in datasets_spec.values():
            if not "skip_constant_channels" in v.keys():
                v["skip_constant_channels"] = False

        check = (
            (isinstance(datasets_spec, dict) or isinstance(datasets_spec, DictConfig))
            and (
                np.alltrue([isinstance(i, dict) for i in datasets_spec.values()])
                or np.alltrue(
                    [isinstance(i, DictConfig) for i in datasets_spec.values()]
                )
            )
            and np.alltrue(
                [
                    set(i.keys()) == {"class", "skip_constant_channels", "kwargs"}
                    for i in datasets_spec.values()
                ]
            )
        )

        if not check:
            raise ValueError(
                "'datasets_spec' must be a list of dicts with keys ['class', 'kwargs', 'skip_constant_channels']"
            )

        self.datasets_spec = datasets_spec
        self.transforms = transforms
        self.limit_pct = limit_pct
        self.get_ids = get_ids

        # if classes are strings with class names, then eval them
        for d in self.datasets_spec.values():
            if isinstance(d["class"], str):
                d["class"] = utils.get_class(d["class"])

        # instantiate datasets
        self.datasets = {
            k: d["class"](split_file=split_file, split=split, **d["kwargs"])
            for k, d in self.datasets_spec.items()
        }

        self.compute_common_tiles()

        """
        self.common_tileids = set(
            self.datasets[list(self.datasets.keys())[0]].get_tileids()
        )

        # retains the files (tiles) that are commong to all datasets
        for k in self.datasets.keys():
            self.common_tileids = self.common_tileids.intersection(
                self.datasets[k].get_tileids()
            )

        self.common_tileids = list(self.common_tileids)
        self.common_tileids = np.array(self.common_tileids, dtype=np.string_)

        if len(self.common_tileids) == 0:
            raise ValueError(f"no common tiles in {self}")

        if limit_pct is not None and limit_pct < 1.0:
            n = int(len(self.common_tileids) * limit_pct)
            logger.info(f"limiting data to {limit_pct} pct, this is {n} tiles")
            self.common_tileids = np.random.permutation(self.common_tileids)[:n]

        logger.info(f"{self}: using {len(self.common_tileids)} common tiles")

        # tells each dataset what tiles to use
        for d in self.datasets.values():
            d.set_tileids(self.common_tileids)
        """

    def compute_common_tiles(self, limit_dataset=True):
        self.common_tileids = set(
            self.datasets[list(self.datasets.keys())[0]].get_tileids()
        )
        # retains the files (tiles) that are commong to all datasets
        for k in self.datasets.keys():
            self.common_tileids = self.common_tileids.intersection(
                self.datasets[k].get_tileids()
            )

        self.common_tileids = sorted(list(self.common_tileids))  # to preserve order
        self.common_tileids = np.array(self.common_tileids, dtype=np.string_)

        if len(self.common_tileids) == 0:
            raise ValueError(f"no common tiles in {self}")

        if limit_dataset:
            if self.limit_pct is not None and self.limit_pct < 1.0:
                n = int(len(self.common_tileids) * self.limit_pct)
                logger.info(f"limiting data to {self.limit_pct} pct, this is {n} tiles")
                self.common_tileids = np.random.permutation(self.common_tileids)[:n]

        logger.info(f"{self}: using {len(self.common_tileids)} common tiles")

        # tells each dataset what tiles to use
        for d in self.datasets.values():
            d.set_tileids(self.common_tileids)

    def prepare_data(self):
        """apply preprocessing step on all datasets"""
        for d in self.datasets.values():
            d.prepare_data()

        self.compute_common_tiles(limit_dataset=False)

    def setup(self, stage):
        """apply setup step on all datasets"""
        for d in self.datasets.values():
            if "setup" in dir(d):
                d.setup(stage)

        self.compute_common_tiles(limit_dataset=False)

    def __len__(self):
        return len(self.common_tileids)

    def __repr__(self):
        return "/".join([str(d) for d in self.datasets.values()])

    def __getitem__(self, idx):
        """
        simply calls getitem in each dataset and returns the combined results.
        if errors in any dataset it attempts other tiles at most 20 times
        """
        max_attepts = 20
        attempts = 1

        exception = ""
        k = ""
        while True:
            try:
                sync_data = {k: d[idx] for k, d in self.datasets.items()}

                if sum(i is None for i in sync_data.values()) > 0:
                    raise ValueError(f"'None' type found in a dataset on item {idx}")

                for k, x in sync_data.items():
                    if (
                        "skip_constant_channels" in self.datasets_spec[k].keys()
                        and self.datasets_spec[k]["skip_constant_channels"]
                    ):
                        if (
                            x.reshape(-1, np.product(x.shape[1:])).std(axis=-1) < 1e-7
                        ).sum() > 0:
                            tileid = self.datasets[k].tileids[idx]
                            logger.warning(
                                f"constant channel value found in item {k}:{idx}, tileid {tileid}. skipping"
                            )
                            raise ValueError(
                                f"constant channel value found in item {k}:{idx}. tileid {tileid}. skipping"
                            )
                if self.get_ids:
                    sync_data["tile_id"] = self.common_tileids[idx]

                if isinstance(self.transforms, dict) or isinstance(
                    self.transforms, DictConfig
                ):
                    for k in self.transforms.keys():
                        if not k in sync_data.keys():
                            raise ValueError(
                                f"key '{k}' in transforms not found in data"
                            )
                        sync_data[k] = self.transforms[k](sync_data[k])

                elif self.transforms is not None:
                    sync_data = self.transforms(sync_data)

                return sync_data

            except Exception as e:
                exception = e
                if attempts == max_attepts:
                    raise e

            if attempts == 1:
                logger.warning(
                    f"error loading tileid {self.common_tileids[idx]}, attempting with other tiles. {exception}"
                )

            attempts += 1
            idx = np.random.randint(len(self))
