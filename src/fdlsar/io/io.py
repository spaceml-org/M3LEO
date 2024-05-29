from __future__ import annotations

import datetime as dt
import os
from contextlib import contextmanager

from rasterio import logging

log = logging.getLogger()
log.setLevel(logging.ERROR)


import blosc
import fsspec
import numpy as np
import xarray as xr
from loguru import logger

from fdlsar import utils
from fdlsar.utils import optional_memory_cache

fsspec.asyn.iothread[0] = None
fsspec.asyn.loop[0] = None

S1GRD_BANDS = [
    "winter_vvasc",
    "winter_vvdes",
    "winter_vhasc",
    "winter_vhdes",
    "spring_vvasc",
    "spring_vvdes",
    "spring_vhasc",
    "spring_vhdes",
    "summer_vvasc",
    "summer_vvdes",
    "summer_vhasc",
    "summer_vhdes",
    "fall_vvasc",
    "fall_vvdes",
    "fall_vhasc",
    "fall_vhdes",
]

S1GRDM_BANDS = [
    "01_vv",
    "01_vh",
    "02_vv",
    "02_vh",
    "03_vv",
    "03_vh",
    "04_vv",
    "04_vh",
    "05_vv",
    "05_vh",
    "06_vv",
    "06_vh",
    "07_vv",
    "07_vh",
    "08_vv",
    "08_vh",
    "09_vv",
    "09_vh",
    "10_vv",
    "10_vh",
    "11_vv",
    "11_vh",
    "12_vv",
    "12_vh",
]


@contextmanager
def open_file_as_dataset(path: str, ds_type: str = "dataarray", group=None):
    """
    Open a file from the local filesystem or from the cloud (GCS) as a dataset.

    Parameters
    ----------
    path : str
        Path to the file to open.

    Returns
    -------
    temp_file : str
        The temporary filename where the file is downloaded (for cloud files) or the original filename (for local files).
    """
    temp_file = None
    is_cloud_file = False

    if path.startswith("gs://"):
        # Use ffspec to open the file
        fs = fsspec.filesystem("gs")

        temp_file = fs.open(path)

        is_cloud_file = True
    else:
        temp_file = path

    try:
        # open the file as a dataset and yield it
        if ds_type == "dataarray":
            with xr.open_dataarray(temp_file, engine="rasterio") as ds:
                yield ds
        elif ds_type == "dataset":
            with xr.open_dataset(temp_file, group=group) as ds:
                yield ds

    finally:
        if is_cloud_file:
            try:
                temp_file.close()
            except OSError:
                pass


# @optional_memory_cache
def load_s1grd(
    file_path: str,
    pols: str | list[str],
    directions: str | list[str],
    seasons: str | list[str],
    summarize_seasons: bool = False,
    normalizer_mean: np.array = None,
    normalizer_std: np.array = None,
) -> np.ndarray:
    if seasons is None:
        seasons = ["winter", "spring", "summer", "fall"]

    # create all combinations based on bands
    if isinstance(pols, str):
        pols = [pols]
    if isinstance(directions, str):
        directions = [directions]
    if isinstance(seasons, str):
        seasons = [seasons]

    band_comb = [
        f"{season}_{pol}{direction}"
        for pol in pols
        for direction in directions
        for season in seasons
    ]
    # load band descriptions
    with open_file_as_dataset(file_path) as ds:
        # load data
        vv_vh = []
        # get the indices of the bands to load from dsc
        band_idx = [S1GRD_BANDS.index(b) for b in band_comb if "vv-vh" not in b]
        ds = ds.isel(band=band_idx)
        ds = ds.astype(np.int16)

        bands_used = [b for b in band_comb if "vv-vh" not in b]
        if "vv-vh" in pols:
            for season in seasons:
                for direction in directions:
                    idx_vv = band_comb.index(f"{season}_vv{direction}")
                    idx_vh = band_comb.index(f"{season}_vh{direction}")
                    # append new band

                    vv_vh.append(
                        (ds.isel(band=idx_vv) - ds.isel(band=idx_vh)).compute()
                    )
                    bands_used.append(f"{season}_vvvh{direction}")
            arr = ds.values
            # concatenate vv_vh with arr
            arr = np.concatenate([vv_vh, arr], axis=0)
        else:
            arr = ds.values

        if normalizer_mean is not None and normalizer_std is not None:
            arr = arr - normalizer_mean.reshape(
                arr.shape[0], 1, 1
            )  # / normalizer_std.reshape(x.shape[0], 1, 1)
    # summarize per season if asked for
    if summarize_seasons:
        noseason_bands = np.unique([i.split("_")[1] for i in bands_used])
        xx = []
        for b in noseason_bands:
            idxs = [
                i for i in range(len(bands_used)) if bands_used[i].endswith("_" + b)
            ]
            xx.append(arr[idxs].mean(axis=0))

        arr = np.r_[xx]
    return arr


def load_s1grdm(file_path: str, month: int | list[int], pols: str | list[str]):
    """
    Load S1GRD monthly data.

    Parameters
    ----------
    file_path : str
        Path to the file to open.
    month : str | list
        List of months to load
    pols : str | list
        List of polarizations to load

    Returns
    -------
    np.narray
        Numpy array containing the loaded data.
    """
    if isinstance(month, int):
        month = [month]
    if isinstance(pols, str):
        pols = [pols]

    with open_file_as_dataset(file_path) as ds:
        # month is always two digits
        month = [f"{i:02d}" for i in month]
        long_names = [f"{i}_{j}" for i in month for j in pols]
        # get index of element int tuple that is equal to "2020_ConfidenceLevel"
        idx = [i for i, x in enumerate(ds.long_name) if x in long_names]
        if len(idx) == 0:
            raise ValueError(f"month {month} not found in {file_path}")

        ds = ds.isel(band=idx)
        arr = ds.compute()

    arr = arr.values.astype(np.int16)
    return arr


@optional_memory_cache
def load_esawc(file_path: str) -> np.ndarray:
    """
    Load ESAWC data.

    Parameters
    ----------
    file_path : str
        Path to the file to open.

    Returns
    -------
    xr.DataArray
        DataArray containing the loaded data.
    """
    with open_file_as_dataset(file_path) as ds:
        arr = ds.compute()
    return arr.values


@optional_memory_cache
def load_srtmdem(file_path: str) -> np.ndarray:
    """
    Load SRTM digital elevation data.

    Parameters
    ----------
    file_path : str
        Path to the file to open.

    Returns
    -------
    xr.DataArray
        DataArray containing the loaded data.
    """
    with xr.open_dataarray(file_path) as f:
        x = f.data.copy()
        lon = f.coords.get("x").data.mean()
        lat = f.coords.get("y").data.mean()
    return x, (lon, lat)


@optional_memory_cache
def load_ghsbuilts(file_path: str) -> np.ndarray:
    """
    Load GHS-BUILT-S data.

    Parameters
    ----------
    file_path : str
        Path to the file to open.

    Returns
    -------
    xr.DataArray
        DataArray containing the loaded data.
    """
    with open_file_as_dataset(file_path) as ds:
        arr = ds.compute()
    return arr.values


@optional_memory_cache
def load_gssic(
    file_path: str,
    variable: str | list[str],
    season: str | list[str] | None = None,
    polarimetry: str | list[str] | None = None,
    deltadays: int | list[int] | None = None,
    param: str | list[str] | None = None,
    feature: str | list[str] | None = None,
) -> np.ndarray:
    """
    Load GSSIC data. There are different combinations of variables and dimensions that can be loaded

    amplitude (season, polarimetry, y, x)
    coherence (season, deltadays, y, x)
    decaymodel (season, param, y, x)
    geometry (feature, y, x)

    Parameters
    ----------
    file_path : str
        Path to the file to open.
    variable : str | list
        List of variables to load
    season : list, optional
        List of seasons to load, possible values "summer", "winter","fall","spring"
    polarimetry : list, optional
        List of polarimetry to load, possible values "vv", "vh"
    deltadays : list, optional
        List of deltadays to load, possible values [12 24 36 48]
    param : list, optional
        List of param to load, possible values 'rho' 'tau' 'rmse'
    feature : list, optional
        List of features to load, possible values 'inc' 'lsmap'

    Returns
    -------
    xr.DataArray
        DataArray containing the loaded data.
    """
    dimensions = {}

    with open_file_as_dataset(file_path, ds_type="dataset") as ds:
        # if amplitude then season and polarimetry are required
        if variable == "amplitude":
            if season is None or polarimetry is None:
                raise ValueError(
                    "If variable is amplitude then season and polarimetry are required"
                )
            dimensions = {"season": season, "polarimetry": polarimetry}

        # if coherence then season and deltadays are required
        if variable == "coherence":
            if season is None or deltadays is None:
                raise ValueError(
                    "If variable is coherence then season and deltadays are required"
                )
            dimensions = {"season": season, "deltadays": deltadays}

        # if decaymodel then season and param are required

        if variable == "decaymodel":
            if season is None or param is None:
                raise ValueError(
                    "If variable is decaymodel then season and param are required"
                )
            dimensions = {"season": season, "param": param}

        # if geometry then feature is required
        if variable == "geometry":
            if feature is None:
                raise ValueError("If variable is geometry then feature is required")
            dimensions = {"feature": feature}

        arr = ds.sel(**dimensions)[variable].compute()
    return arr.values


@optional_memory_cache
def load_gunw(
    file_path: str,
    variable: str | list[str],
    max_days: int,
    min_days: int,
    selection_rule: str,
) -> np.ndarray:
    """
    Load GUNW data.

    Parameters
    ----------
    file_path : str
        Path to the file to open.
    variable : str | list
        List of variables to load
    max_days : int
        Maximum number of days between the two SAR acquisitions
    min_days : int
        Minimum number of days between the two SAR acquisitions
    selection_rule : str
        Rule to select the SAR acquisitions, possible values "close", "far", "mean"

    Returns
    -------
    xr.DataArray
        DataArray containing the loaded data.

    """
    if selection_rule not in ["close", "far", "mean"]:
        raise ValueError("Selection rule must be one of 'close', 'far', or 'mean'.")

    with open_file_as_dataset(
        file_path, ds_type="dataset", group="/science/grids/data"
    ) as ds:
        ds = ds[variable]
        if max_days is not None and min_days is not None:
            # each datepair has '20180121_20170102' format
            # convert to datetime and calculate difference

            datepair_ranges = []
            for datepair in ds["datepair"].values:
                end_date = dt.datetime.strptime(str(datepair).split("_")[0], "%Y%m%d")
                start_date = dt.datetime.strptime(str(datepair).split("_")[1], "%Y%m%d")
                day_diff = end_date - start_date
                datepair_ranges.append(day_diff)

            selected_indices = np.where(
                (np.array(datepair_ranges) <= dt.timedelta(days=max_days))
                & (np.array(datepair_ranges) >= dt.timedelta(days=min_days))
            )[0]

            if len(selected_indices) == 0:
                raise ValueError(
                    f"no datepairs found with max_days={max_days} and min_days={min_days} in {file_path}. Current dates are {ds['datepair'].values}"
                )

            if selection_rule == "close":
                selected_indices = selected_indices[-1:]
                ds = ds.isel(datepair=selected_indices)
            elif selection_rule == "far":
                selected_indices = selected_indices[:1]
                ds = ds.isel(datepair=selected_indices)
            elif selection_rule == "mean":
                ds = ds.isel(datepair=selected_indices).mean(axis=0)

        arr = ds.squeeze().compute()

    return arr.values


def load_gunw_specific_ranges(
    file_path: str,
    variable: str | list[str],
    ranges: list[tuple[int, int]],
    average: bool = False,
) -> np.ndarray:
    """
    Load GUNW data.

    Parameters
    ----------
    file_path : str
        Path to the file to open.
    variable : str | list
        List of variables to load
    ranges : list[tuple[int, int]]
        List of ranges to load, example: [((20201028,20201030), (20201102,20201104)),
                                            ((20201128,20201130), (20201202,20201204))]

    Returns
    -------
    np.ndarray
        Array containing the loaded data.
    """
    with open_file_as_dataset(
        file_path, ds_type="dataset", group="/science/grids/data"
    ) as ds:
        ds = ds[variable]
        final_arr = None
        for r in ranges:
            r_start_date_min = dt.datetime.strptime(str(r[0][0]), "%Y%m%d")
            r_start_date_max = dt.datetime.strptime(str(r[0][1]), "%Y%m%d")
            r_end_date_min = dt.datetime.strptime(str(r[1][0]), "%Y%m%d")
            r_end_date_max = dt.datetime.strptime(str(r[1][1]), "%Y%m%d")
            range_indices = []
            for i, datepair in enumerate(ds["datepair"].values):
                end_date = dt.datetime.strptime(str(datepair).split("_")[0], "%Y%m%d")
                start_date = dt.datetime.strptime(str(datepair).split("_")[1], "%Y%m%d")
                if (
                    start_date >= r_start_date_min
                    and start_date <= r_start_date_max
                    and end_date >= r_end_date_min
                    and end_date <= r_end_date_max
                ):
                    range_indices.append(i)

            if len(range_indices) == 0:
                raise ValueError(
                    f"no datepairs found with range {r} in {file_path}. Current dates are {ds['datepair'].values}"
                )

            arr = ds.isel(datepair=range_indices)
            if average:
                arr = arr.mean(axis=0)
            arr = arr.compute()
            if final_arr is None:
                final_arr = arr
            else:
                final_arr = xr.concat([final_arr, arr], dim="datepair")
    return final_arr.values


def load_gunw_extended(
    file_path: str,
    variable: str | list[str],
    max_days: int = 9999,
    min_days: int = 0,
    last_start_date: str = None,
    first_start_date: str = None,
    last_end_date: str = None,
    first_end_date: str = None,
    selection_rule: str = "none",
    normalizer: GunwMeanStdNormalizer = None,
) -> np.ndarray:
    """
    Load GUNW data.

    Parameters
    ----------
    file_path : str
        Path to the file to open.
    variable : str | list
        List of variables to load
    max_days : int
        Maximum number of days between the two SAR acquisitions
    min_days : int
        Minimum number of days between the two SAR acquisitions
    selection_rule : str
        Rule to select the SAR acquisitions, possible values "close", "far", "mean", "none"
    last_start_date : str
        pairs with start_date later than last_start_date will be discarded. date format is "20201028" for instance
    first_start_date : str
        pairs with start_date sooner than first_start_date will be discarded
    last_end_date : str
        pairs with end_date later than last_end_date will be discarded
    first_end_date : str
        pairs with end_date sooner than first_end_date will be discarded

    Returns
    -------
    xr.DataArray
        DataArray containing the loaded data.

    """

    if selection_rule not in ["close", "far", "mean", "none"]:
        raise ValueError(
            "Selection rule must be one of 'close', 'far', 'none' or 'mean'."
        )

    if not isinstance(variable, list):
        variable = [variable]

    if last_start_date is None:
        last_start_date = "99999999"

    if first_start_date is None:
        first_start_date = "00000000"

    if last_end_date is None:
        last_end_date = "99999999"

    if first_end_date is None:
        first_end_date = "00000000"

    # attempt first to load cached blosc
    params = {
        "variable": variable,
        "max_days": max_days,
        "min_days": min_days,
        "last_start_date": last_start_date,
        "first_start_date": first_start_date,
        "last_end_date": last_end_date,
        "first_end_date": first_end_date,
        "selection_rule": selection_rule,
        "normalizer": normalizer,
    }
    phash = utils.get_hash(str(params))
    blosc_file = f"{file_path}--{phash}.blosc"

    # if blosc cache exists
    if not file_path.startswith("gs://") and os.path.isfile(blosc_file):
        try:
            with open(blosc_file, "rb") as f:
                arr = blosc.unpack_array(f.read())
            return arr
        except Exception as e:
            logger.info(f"could not load {blosc_file}. {e}")

    with open_file_as_dataset(
        file_path, ds_type="dataset", group="/science/grids/data"
    ) as ds:
        ds = ds[variable]

        if normalizer is not None:
            for v in variable:
                ds = normalizer.normalize(ds, v)

        if max_days is not None and min_days is not None:
            # each datepair has '20180121_20170102' format
            # convert to datetime and calculate difference

            datepair_ranges = []
            end_dates = []
            start_dates = []
            for datepair in ds["datepair"].values:
                end_date = dt.datetime.strptime(str(datepair).split("_")[0], "%Y%m%d")
                start_date = dt.datetime.strptime(str(datepair).split("_")[1], "%Y%m%d")
                day_diff = end_date - start_date
                datepair_ranges.append(day_diff)
                end_dates.append(end_date.strftime("%Y%m%d"))
                start_dates.append(start_date.strftime("%Y%m%d"))
            selected_indices = np.where(
                (np.array(datepair_ranges) <= dt.timedelta(days=max_days))
                & (np.array(datepair_ranges) >= dt.timedelta(days=min_days))
                & (np.array(start_dates) <= last_start_date)
                & (np.array(start_dates) >= first_start_date)
                & (np.array(end_dates) <= last_end_date)
                & (np.array(end_dates) >= first_end_date)
            )[0]

            if len(selected_indices) == 0:
                raise ValueError(
                    f"no datepairs found with max_days={max_days} and min_days={min_days} in {file_path} and start/end date limits. Current dates are {ds['datepair'].values}"
                )

            if selection_rule == "close":
                selected_indices = selected_indices[-1:]
                ds = ds.isel(datepair=selected_indices)
            elif selection_rule == "far":
                selected_indices = selected_indices[:1]
                ds = ds.isel(datepair=selected_indices)
            elif selection_rule == "mean":
                ds = ds.isel(datepair=selected_indices).mean(dim="datepair")
            else:
                ds = ds.isel(datepair=selected_indices)

        arr = ds.compute()
        # dont squeeze since will remove dimensions with only one component and will change the shape length
        # arr = ds.squeeze().compute()

    if selection_rule == "none":
        # double check date conditions
        xstart_dates = np.r_[[i.split("_")[1] for i in arr.datepair.values]]
        xend_dates = np.r_[[i.split("_")[0] for i in arr.datepair.values]]
        if last_start_date is not None:
            assert np.alltrue(xstart_dates <= last_start_date)

        if first_start_date is not None:
            assert np.alltrue(xstart_dates >= first_start_date)

        if last_end_date is not None:
            assert np.alltrue(xend_dates <= last_end_date)

        if first_end_date is not None:
            assert np.alltrue(xend_dates >= first_end_date)

    arr = arr.to_array().values
    arr = arr.reshape(-1, *arr.shape[-2:])

    # write blosc cache
    if not file_path.startswith("gs://"):
        with open(blosc_file, "wb") as f:
            f.write(blosc.pack_array(arr, cname="zstd"))

    return arr


@optional_memory_cache
def load_modisveg(
    file_path: str, year: int | list[int], convert_water_to_zero: bool = True
) -> np.ndarray:
    """
    Load MODIS44B006 vegetation data.

    Parameters
    ----------
    file_path : str
        Path to the file to open.
    year : str | list
        List of years to load
    convert_water_to_zero : bool, optional
        Whether to convert water pixels to zero, by default True.

    Returns
    -------
    xr.DataArray
        DataArray containing the loaded data.
    """
    with open_file_as_dataset(file_path) as ds:
        if isinstance(year, int):
            year = [year]
        long_names = [f"Percent_Tree_Cover_{i}" for i in year]
        # get index of element int tuple that is equal to "Percent_Tree_Cover_2020"
        idx = [i for i, x in enumerate(ds.long_name) if x in long_names]
        if len(idx) == 0:
            raise ValueError(f"year {year} not found in {file_path}")

        ds = ds.isel(band=idx)
        if convert_water_to_zero:
            # if value is 200, set to 0
            ds = ds.where(ds != 200, 0)
        arr = ds.compute()

    values = arr.values
    if convert_water_to_zero:
        values[values == 200] = 0

    values[np.isnan(values)] = 0

    return arr.values


@optional_memory_cache
def load_biomass(file_path: str) -> np.ndarray:
    """
    Load ESA Biomass data.
    We are not computing the standard error. .

    Parameters
    ----------
    file_path : str
        Path to the file to open.

    Returns
    -------
    xr.DataArray
        DataArray containing the loaded data.

    xr.DataArray
        DataArray containing the standard error of the loaded data.
    """
    with open_file_as_dataset(file_path, ds_type="dataset") as ds:
        agb_ds = ds["agb"].compute()
        agb_se_ds = ds["agb_se"].compute()

    return agb_ds.values, agb_se_ds.values


@optional_memory_cache
def load_s2rgbm(
    file_path: str,
    months: str | list[str] | list[list[str]],
) -> np.ndarray:
    """
    Load Sentinel-2 RGB data from a file.

    Args:
        file_path (str): The path to the file containing the data.
        months (str | list[str] | list[list[str]]): The months for which to load the data.
            It can be a single month as a string, a list of months as strings, or a list of lists
            where each inner list contains the months for a specific year.

    Returns:
        np.ndarray: The loaded RGBM data as a NumPy array.

    Raises:
        ValueError: If the specified months are not found in the file.

    """
    if isinstance(months, str):
        months = [months]

    # get all bands combinations
    long_names = [
        f"{month}_{rgb}" for month in months for rgb in ["red", "green", "blue"]
    ]

    with open_file_as_dataset(file_path) as ds:
        # get index of element int tuple that is equal to "2020_ConfidenceLevel"
        idx = [i for i, x in enumerate(ds.long_name) if x in long_names]
        if len(idx) == 0:
            raise ValueError(f"months {months} not found in {file_path}")

        ds = ds.isel(band=idx)
        arr = ds.compute()
        values = arr.values
        values = values / 255

    if arr is not None:
        return values
