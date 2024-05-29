from __future__ import annotations

from fdlsar.io import io_blosc

from . import base


class BiomassDataset(base.BaseDataset):
    def __init__(self, split_file, split, year, transform=None):
        self.dataset = f"biomass-{year}"
        self.transform = transform

        super().__init__(
            split_file=split_file, split=split, dataset=self.dataset, tile_format="nc"
        )
        allowed_years = [2018, 2019, 2020]
        if not year in allowed_years:
            raise ValueError(f"invalid year {year}, only allowed {allowed_years}")

        self.year = str(year)

    def __getitem__(self, idx):
        """
        loads tiles and associated standard error
        """
        files_to_load = f"{self.dataset_folder}/{self.get_files()[idx]}"
        arr, _ = io_blosc.load_biomass(files_to_load)

        if self.transform is not None:
            arr = self.transform(arr)

        return arr
