from __future__ import annotations

from fdlsar import utils
from fdlsar.io import io_blosc

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

    def prepare_data(self):
        # compute stats for normalization
        if self.normalize:
            self.normalizer.prepare_data()

    def setup(self, stage):
        # load stats for normalization
        if self.normalize:
            self.normalizer.prepare_data()

    def load_item(self, idx):
        files_to_load = f"{self.dataset_folder}/{self.get_files()[idx]}"

        arr = io_blosc.load_gssic(
            file_path=files_to_load,
            variable=self.variable,
            season=self.season,
            polarimetry=self.polarimetry,
            deltadays=self.deltadays,
            param=self.param,
            feature=self.feature,
        )

        arr = arr.reshape(-1, *arr.shape[-2:])
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
