from __future__ import annotations

from fdlsar.io import io

from . import base


class ModisVegDataset(base.BaseDataset):
    def __init__(
        self,
        split_file,
        split,
        year: int | list[int],
        convert_water_to_zero: bool = True,
        transform=None,
    ):
        """
        Parameters
        ----------
        split_file : str
            Path to the csv file containing the split definitions.
        split : str
            One of 'train', 'test', 'val'.
        year : int | list[int]
            Year to load.
        convert_water_to_zero : bool, optional
            Whether to convert water pixels to zero, by default True.

        Raises
        ------
        ValueError
            If year is not one of 2016, 2017, 2018, 2019, 2020.

        """
        self.transform = transform
        allowed_years = [2016, 2017, 2018, 2019, 2020]
        if isinstance(year, int):
            year = [year]
        if not set(allowed_years).intersection(year):
            raise ValueError(f"invalid year {year}, only allowed {allowed_years}")

        self.year = year
        self.convert_water_to_zero = convert_water_to_zero
        super().__init__(
            split_file, split, dataset=f"modis44b006veg", tile_format="tif"
        )

    def __getitem__(self, idx):
        """
        loads tile
        """
        files_to_load = f"{self.dataset_folder}/{self.get_files()[idx]}"
        arr = io.load_modisveg(
            files_to_load,
            self.year,
            self.convert_water_to_zero,
        )

        if self.transform is not None:
            arr = self.transform(arr)

        return arr
