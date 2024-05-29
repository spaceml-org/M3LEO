from __future__ import annotations

import numpy as np

from fdlsar.io import io

from . import base

# codes are the same for 2020 and 2021
ESA_WC_CODES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]


# map map_codes to 0-10
def map_codes_to_0_10(arr, codes):
    r = np.zeros_like(arr)
    for i in range(len(codes)):
        r[arr == codes[i]] = i
    return r.astype(np.uint8)


class ESAWCDataset(base.BaseDataset):
    def __init__(self, split_file, split, year):
        self.dataset = f"esaworldcover-{year}"
        super().__init__(
            split_file=split_file, split=split, dataset=self.dataset, tile_format="tif"
        )
        allowed_years = [2020, 2021]
        if not year in allowed_years:
            raise ValueError(f"invalid year {year}, only allowed {allowed_years}")

        self.year = str(year)

    def __getitem__(self, idx):
        """
        loads tile and maps codes to 0-10
        """
        files_to_load = f"{self.dataset_folder}/{self.get_files()[idx]}"
        arr = io.load_esawc(files_to_load)
        arr = arr.squeeze()
        arr = map_codes_to_0_10(arr, ESA_WC_CODES)
        return arr
