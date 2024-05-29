from __future__ import annotations

import numpy as np

from fdlsar import utils
from fdlsar.io import io

from . import base


class GSSICDataset(base.BaseDataset):
    def __init__(
        self,
        split_file,
        split,
        variable: str | list[str],
        season: str | list[str] | None = None,
        polarimetry: str | list[str] | None = None,
        deltadays: int | list[int] | None = None,
        param: str | list[str] | None = None,
        feature: str | list[str] | None = None,
        normalize=True,
        transform=None,
        percentiles=None,
    ):
        super().__init__(split_file, split, dataset=f"gssic", tile_format="nc")
        self.variable = variable
        if isinstance(self.variable, str):
            self.variable = [self.variable]

        self.season = season
        self.polarimetry = polarimetry
        self.deltadays = deltadays
        self.param = param
        self.feature = feature
        self.transform = transform
        self.normalize = normalize

        self.normalizer = utils.MeanStdNormalizer(
            self,
            {
                "variable": variable,
                "season": season,
                "polarimetry": polarimetry,
                "deltadays": deltadays,
                "param": param,
                "feature": feature,
            },
            percentiles=percentiles,
        )
        print(self.normalizer)

    def prepare_data(self):
        # compute stats for normalization
        if self.normalize:
            self.normalizer.prepare_data()
            print(self.normalizer.means)

    def setup(self, stage):
        # load stats for normalization
        if self.normalize:
            self.normalizer.prepare_data()

    def load_item(self, idx):
        files_to_load = f"{self.dataset_folder}/{self.get_files()[idx]}"
        stack = []

        for var in self.variable:
            arr = None
            if var == "coherence":
                arr = io.load_gssic(
                    file_path=files_to_load,
                    variable=var,
                    season=self.season,
                    polarimetry=None,
                    deltadays=self.deltadays,
                    param=None,
                    feature=None,
                )
            elif var == "amplitude":
                arr = io.load_gssic(
                    file_path=files_to_load,
                    variable=var,
                    season=self.season,
                    polarimetry=self.polarimetry,
                    deltadays=None,
                    param=None,
                    feature=None,
                )
            elif var == "decaymodel":
                arr = io.load_gssic(
                    file_path=files_to_load,
                    variable=var,
                    season=self.season,
                    polarimetry=None,
                    deltadays=None,
                    param=self.param,
                    feature=None,
                )
            elif var == "geometry":
                arr = io.load_gssic(
                    file_path=files_to_load,
                    variable=var,
                    season=None,
                    polarimetry=None,
                    deltadays=None,
                    param=None,
                    feature=self.feature,
                )

            if arr is None:
                raise ValueError(f"{arr} is empty.")
            stack.append(arr)

        stack = [i.reshape(-1, *i.shape[-2:]) for i in stack]
        arr = np.vstack(stack)
        return arr

    def __getitem__(self, idx):
        """
        loads tile
        """
        x = self.load_item(idx)

        if self.normalize:
            x = self.normalizer.normalize(x)

        if self.transform is not None:
            x = self.transform(x)
        return x
