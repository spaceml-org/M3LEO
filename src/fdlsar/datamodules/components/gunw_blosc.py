from __future__ import annotations

from fdlsar import utils
from fdlsar.io import io_blosc

from . import base


class GUNWDataset(base.BaseDataset):
    def __init__(
        self,
        split_file,
        split,
        date,
        variable,
        max_days: int = 9999,
        min_days: int = 0,
        last_start_date: str = None,
        first_start_date: str = None,
        last_end_date: str = None,
        first_end_date: str = None,
        selection_rule: str = "none",
        normalize: bool = True,
        transform=None,
        mode=None,  # for legacy only
    ):
        self.dataset = f"gunw-{date}"
        super().__init__(split_file, split, dataset=self.dataset, tile_format="nc")

        """
        allowed_modes = ['full', 'mean', 'randomone']
        if not mode in allowed_modes:
            raise ValueError(f"invalid mode {mode}, only {allowed_modes} allowed")
        self.mode = mode
        """

        self.variable = variable
        self.min_days = min_days
        self.max_days = max_days
        self.selection_rule = selection_rule
        self.last_start_date = last_start_date
        self.first_start_date = first_start_date
        self.last_end_date = last_end_date
        self.first_end_date = first_end_date
        self.transform = transform
        self.date = date
        self.normalize = normalize

        if self.normalize:
            self.normalizer = utils.GunwMeanStdNormalizer(self)
        else:
            self.normalizer = None

    def prepare_data(self):
        # compute stats for normalization
        if self.normalize:
            self.normalizer.prepare_data()

    def setup(self, stage):
        # compute stats for normalization
        if self.normalize:
            self.normalizer.prepare_data()

    def __getitem__(self, idx):
        """
        loads tile
        """
        files_to_load = f"{self.dataset_folder}/{self.get_files()[idx]}"
        arr = io_blosc.load_gunw_extended(
            files_to_load,
            self.variable,
            max_days=self.max_days,
            min_days=self.min_days,
            last_start_date=self.last_start_date,
            first_start_date=self.first_start_date,
            last_end_date=self.last_end_date,
            first_end_date=self.first_end_date,
            selection_rule=self.selection_rule,
            normalizer=self.normalizer,
        )  # loads xarray with all datepairs

        if self.transform is not None:
            arr = self.transform(arr)

        """
        if self.mode=='randomone':
            idx = np.random.randint(arr.shape[0])
            arr = arr[idx:idx+1]
        elif self.mode=='mean':
            arr = arr.mean(axis=0).reshape(1, *arr.shape[1:])
        """
        return arr


class GUNWEventDataset(base.BaseDataset):
    def __init__(
        self,
        split_file,
        split,
        date,
        variable: str | list[str],
        event_start_date: str,
        event_end_date: str,
        event_pairs_mode: str | list[str],
        normalize: bool = True,
        transform=None,
    ):
        self.dataset = f"gunw-{date}"
        super().__init__(split_file, split, dataset=self.dataset, tile_format="nc")

        self.variable = variable
        self.event_start_date = event_start_date
        self.event_end_date = event_end_date
        self.event_pairs_mode = event_pairs_mode
        self.transform = transform
        self.date = date
        self.normalize = normalize

        if self.normalize:
            self.normalizer = utils.GunwMeanStdNormalizer(self)
        else:
            self.normalizer = None

    def prepare_data(self):
        # compute stats for normalization
        if self.normalize:
            self.normalizer.prepare_data()

    def setup(self, stage):
        # compute stats for normalization
        if self.normalize:
            self.normalizer.prepare_data()

    def __getitem__(self, idx):
        """
        loads tile
        """
        file_path = f"{self.dataset_folder}/{self.get_files()[idx]}"
        arr = io_blosc.load_gunw_event(
            file_path,
            variable=self.variable,
            event_start_date=self.event_start_date,
            event_end_date=self.event_end_date,
            event_pairs_mode=self.event_pairs_mode,
            normalizer=self.normalizer,
        )  # loads xarray with all datepairs

        if self.transform is not None:
            arr = self.transform(arr)

        return arr
