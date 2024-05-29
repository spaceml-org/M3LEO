from __future__ import annotations

from fdlsar.io import io

from . import base


class GHSBuiltSDataset(base.BaseDataset):
    def __init__(self, split_file, split, year):
        self.dataset = f"ghsbuilts-{year}"
        super().__init__(
            split_file=split_file, split=split, dataset=self.dataset, tile_format="tif"
        )
        allowed_years = [2020]
        if not year in allowed_years:
            raise ValueError(f"invalid year {year}, only allowed {allowed_years}")

    def __getitem__(self, idx):
        """
        loads tile from the dataset
        """
        files_to_load = f"{self.dataset_folder}/{self.get_files()[idx]}"
        ds = io.load_ghsbuilts(files_to_load)
        arr = ds.squeeze()

        return arr
