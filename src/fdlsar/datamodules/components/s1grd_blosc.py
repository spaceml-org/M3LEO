from __future__ import annotations

import numpy as np

from fdlsar import utils
from fdlsar.io import io_blosc

from . import base


class S1grdDataset(base.BaseDataset):
    def __init__(
        self,
        split_file,
        split,
        year,
        pols,
        directions,
        seasons=None,  # this takes all seasons
        normalize=True,
        summarize_seasons=False,
        transform=None,
        percentiles=None,
    ):
        if seasons is None:
            seasons = ["winter", "spring", "summer", "fall"]

        if isinstance(pols, str):
            pols = [pols]
        if isinstance(directions, str):
            directions = [directions]
        if isinstance(seasons, str):
            seasons = [seasons]

        for pol in pols:
            if not pol in ["vv", "vh", "vv-vh"]:
                raise ValueError(
                    f"invalid pol '{pol}', must be one of 'vv' or 'vh' or 'vv-vh'"
                )

        for direction in directions:
            if not direction in ["asc", "des"]:
                raise ValueError(
                    f"invalid direction '{direction}', must be one of 'asc' or 'des'"
                )

        # get all bands combinations
        band_comb = [
            f"{season}_{pol}{direction}"
            for pol in pols
            for direction in directions
            for season in seasons
        ]

        self.dataset = f"s1grd-{year}"
        self.year = year
        self.pols = pols
        self.directions = directions
        self.seasons = seasons
        self.summarize_seasons = summarize_seasons
        self.normalize = normalize
        self.transform = transform

        super().__init__(split_file, split, self.dataset, tile_format="tif")

        self.normalizer = utils.MeanStdNormalizer(
            self,
            {
                "year": year,
                "pols": pols,
                "directions": directions,
                "seasons": seasons,
                "summarize_seasons": summarize_seasons,
            },
            percentiles=percentiles,
        )

    def prepare_data(self):
        # compute stats for normalization
        if self.normalize:
            self.normalizer.prepare_data()

    def setup(self, stage):
        # compute stats for normalization
        if self.normalize:
            self.normalizer.prepare_data()

    def load_item(self, idx):
        """
        just load an item, without transforms
        """

        file_path = f"{self.dataset_folder}/{self.get_files()[idx]}"

        xpols = io_blosc.load_s1grd(
            file_path=file_path,
            pols=self.pols,
            directions=self.directions,
            seasons=self.seasons,
            summarize_seasons=self.summarize_seasons,
        )

        xpols = xpols.astype(np.float32)

        return xpols

    def __getitem__(self, idx):
        """
        loads tile and normalizes it
        """
        x = self.load_item(idx)

        if self.normalize:
            x = self.normalizer.normalize(x)

        if self.transform is not None:
            x = self.transform(x)

        return x

    def __repr__(self):
        return f"{super().__repr__()}_{self.pols}_{self.directions}_{self.seasons}"
