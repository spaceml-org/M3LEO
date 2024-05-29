from __future__ import annotations

from fdlsar import utils
from fdlsar.io import io

from . import base


class S2rgbmDataset(base.BaseDataset):
    def __init__(
        self,
        split_file,
        split,
        year,
        months=None,  # this takes all months
        normalize=True,
        transform=None,
    ):
        all_months = [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
        ]
        if months is None:
            months = all_months

        if isinstance(months, str):
            months = [months]

        for month in months:
            if not month in all_months:
                raise ValueError(f"invalid month {month}")

        self.dataset = f"s2rgbm-{year}"
        self.year = year
        self.months = months
        self.transform = transform
        self.normalize = normalize

        super().__init__(split_file, split, self.dataset, tile_format="tif")

        self.normalizer = utils.MeanStdNormalizer(
            self,
            {
                "year": year,
                "months": months,
            },
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

        arr = io.load_s2rgbm(
            file_path=file_path,
            months=self.months,
        )

        return arr

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
        return f"{super().__repr__()}_{self.months}"
