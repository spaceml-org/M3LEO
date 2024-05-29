from __future__ import annotations

from fdlsar.io import io

from . import base


class SRTMDEMDataset(base.BaseDataset):
    def __init__(self, split_file, split, transform=None, **kwargs):
        """
        kwargs is ignored, just for compatibility
        """
        self.dataset = f"srtmdem"
        self.transform = transform

        super().__init__(
            split_file=split_file, split=split, dataset=self.dataset, tile_format="tif"
        )

    def __getitem__(self, idx):
        """
        loads tiles and associated standard error
        """
        file_to_load = f"{self.dataset_folder}/{self.get_files()[idx]}"
        arr, (lon, lat) = io.load_srtmdem(file_to_load)

        if self.transform is not None:
            arr = self.transform(arr)

        return arr
