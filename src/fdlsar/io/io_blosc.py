from __future__ import annotations

import datetime as dt
import os

from rasterio import logging

log = logging.getLogger()
log.setLevel(logging.ERROR)

from datetime import datetime

import blosc
import dotenv
import fsspec
import numpy as np
from loguru import logger

from fdlsar import utils

from . import io

fsspec.asyn.iothread[0] = None
fsspec.asyn.loop[0] = None


dotenv.load_dotenv()
blosc_cache_dir = os.getenv("BLOSC_CACHE_DIR")

if blosc_cache_dir is None:
    logger.info("-----------------------")
    logger.info(" NOT using blosc cache")
    logger.info("-----------------------")
else:
    logger.info("-----------------------")
    logger.info(f" using blosc cache at {blosc_cache_dir}")
    logger.info("-----------------------")


open_file_as_dataset = io.open_file_as_dataset

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

S2RGBM_BANDS = [
    "01_red",
    "01_green",
    "01_blue",
    "02_red",
    "02_green",
    "02_blue",
    "03_red",
    "03_green",
    "03_blue",
    "04_red",
    "04_green",
    "04_blue",
    "05_red",
    "05_green",
    "05_blue",
    "06_red",
    "06_green",
    "06_blue",
    "07_red",
    "07_green",
    "07_blue",
    "08_red",
    "08_green",
    "08_blue",
    "09_red",
    "09_green",
    "09_blue",
    "10_red",
    "10_green",
    "10_blue",
    "11_red",
    "11_green",
    "11_blue",
    "12_red",
    "12_green",
    "12_blue",
]


# -------------------------------------------
# blosc cache utils
# -------------------------------------------
def get_blosc_filename(file_path, params):
    if blosc_cache_dir is not None:
        # get last four folders from file path and append them to cache dir
        file_path = blosc_cache_dir + "/" + "/".join(file_path.split("/")[-4:])
        os.makedirs(os.path.split(file_path)[0], exist_ok=True)

    phash = utils.get_hash(str(params))
    blosc_file = f"{file_path}--{phash}.blosc"
    return blosc_file


def blosc_cache_load(file_path, params):
    blosc_file = get_blosc_filename(file_path, params)

    # if blosc cache exists
    if not file_path.startswith("gs://") and os.path.isfile(blosc_file):
        try:
            with open(blosc_file, "rb") as f:
                x = blosc.unpack_array(f.read())
            return x
        except Exception as e:
            logger.info(f"could not load {blosc_file}. {e}")


def blosc_cache_save(file_path, params, x):
    blosc_file = get_blosc_filename(file_path, params)

    if not file_path.startswith("gs://"):
        with open(blosc_file, "wb") as f:
            f.write(blosc.pack_array(x, cname="zstd"))


# -------------------------------------------


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

    if isinstance(variable, str):
        variable = [variable]

    # attempt first to load cached blosc
    params = {
        "variable": variable,
        "season": season,
        "polarimetry": polarimetry,
        "deltadays": deltadays,
        "param": param,
        "feature": feature,
    }

    arr = blosc_cache_load(file_path, params)
    if arr is not None:
        return arr

    stack = []

    for var in variable:
        arr = None
        if var == "coherence":
            arr = load_one_gssic(
                file_path=file_path,
                variable=var,
                season=season,
                polarimetry=None,
                deltadays=deltadays,
                param=None,
                feature=None,
            )
        elif var == "amplitude":
            arr = load_one_gssic(
                file_path=file_path,
                variable=var,
                season=season,
                polarimetry=polarimetry,
                deltadays=None,
                param=None,
                feature=None,
            )
        elif var == "decaymodel":
            arr = load_one_gssic(
                file_path=file_path,
                variable=var,
                season=season,
                polarimetry=None,
                deltadays=None,
                param=param,
                feature=None,
            )
        elif var == "geometry":
            arr = load_one_gssic(
                file_path=file_path,
                variable=var,
                season=None,
                polarimetry=None,
                deltadays=None,
                param=None,
                feature=feature,
            )

        if arr is None:
            raise ValueError(f"{arr} is empty.")
        stack.append(arr)

    x = np.vstack(stack)

    # write blosc cache
    blosc_cache_save(file_path, params, x)

    return x


def load_one_gssic(
    file_path: str,
    variable: str | list[str],
    season: str | list[str] | None = None,
    polarimetry: str | list[str] | None = None,
    deltadays: int | list[int] | None = None,
    param: str | list[str] | None = None,
    feature: str | list[str] | None = None,
) -> np.ndarray:
    """
    Load one GSSIC data. There are different combinations of variables and dimensions that can be loaded

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
    if season is None:
        season = ["summer", "winter", "fall", "spring"]

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
        "normalizer": normalizer.__class__.__name__
        if normalizer is not None
        else "None",
    }

    arr = blosc_cache_load(file_path, params)
    if arr is not None:
        return arr

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
    blosc_cache_save(file_path, params, arr)

    return arr


def load_gunw_event(
    file_path: str,
    variable: str | list[str],
    event_start_date: str,
    event_end_date: str,
    event_pairs_mode: str | list[str],
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
    event_start_date: str
        Start date of event in format 'yyyy-mm-dd', for instance 2018-12-05
    event_end_date: str
        End date of event in format 'yyyy-mm-dd', for instance 2018-12-05
    event_pairs_mode: str | list[str]
        A list containing 'pre', 'co', 'pos'

    Returns
    -------
    xr.DataArray
        DataArray containing the loaded data.

    """

    # attempt first to load cached blosc
    params = {
        "variable": variable,
        "event_start_date": event_start_date,
        "event_end_date": event_end_date,
        "event_pairs_mode": event_pairs_mode,
        "normalizer": normalizer.__class__.__name__
        if normalizer is not None
        else "None",
    }

    arr = blosc_cache_load(file_path, params)
    if arr is not None:
        return arr

    estart = datetime.strptime(event_start_date, "%Y-%m-%d")
    eend = datetime.strptime(event_end_date, "%Y-%m-%d")

    with open_file_as_dataset(
        file_path, ds_type="dataset", group="/science/grids/data"
    ) as ds:
        ds = ds[variable]
        if normalizer is not None:
            for v in variable:
                ds = normalizer.normalize(ds, v)

        # first split datapairs present into pre, co, pos event
        ppre = []
        ppos = []
        pco = []
        for pair in ds.datepair.values:
            dend, dstart = pair.split("_")

            dend = datetime.strptime(dend, "%Y%m%d")
            dstart = datetime.strptime(dstart, "%Y%m%d")

            if dend < estart:
                ppre.append([dstart, dend])

            elif dstart > eend:
                ppos.append([dstart, dend])

            elif dstart < estart and dend > eend:
                pco.append([dstart, dend])

        # from co select pair with shortest period
        pair_co = (
            pco[np.argmin([(i2 - i1).days for i1, i2 in pco])] if len(pco) > 0 else None
        )

        # from pre select pair whose end date is closest to start of event
        pair_pre = (
            ppre[np.argmin([(i2 - estart).days for i1, i2 in ppre])]
            if len(ppre) > 0
            else None
        )

        # from pre select pair whose start date closest to end of event
        pair_pos = (
            ppos[np.argmin([(i1 - eend).days for i1, i2 in ppos])]
            if len(ppos) > 0
            else None
        )

        # select the ones requested by the user
        datepairs_selected = []
        for p in event_pairs_mode:
            if p == "pre":
                datepairs_selected.append(pair_pre)
            if p == "co":
                datepairs_selected.append(pair_co)
            if p == "pos":
                datepairs_selected.append(pair_pos)

        if sum(i is None for i in datepairs_selected) > 0:
            raise ValueError("could not find all event pair modes")

        # reformat to select in xarray
        datepairs_selected = [
            f"{i2.strftime('%Y%m%d')}_{i1.strftime('%Y%m%d')}"
            for i1, i2 in datepairs_selected
        ]

        # sanity check
        if sum(i in ds.datepair.values for i in datepairs_selected) != len(
            datepairs_selected
        ):
            raise ValueError(
                "internal error computing datepairs, some selected datepair is not in the tile"
            )

        # finally select from xarray
        idxs = np.r_[
            [np.argwhere(ds.datepair.values == i)[0, 0] for i in datepairs_selected]
        ]
        arr = ds.isel(datepair=idxs).compute().values

    # write blosc cache
    blosc_cache_save(file_path, params, arr)

    return arr


def load_s1grd(
    file_path: str,
    pols: str | list[str],
    directions: str | list[str],
    seasons: str | list[str],
    summarize_seasons: bool = False,
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

    params = {
        "pols": pols,
        "directions": directions,
        "seasons": seasons,
        "summarize_seasons": summarize_seasons,
    }

    # attempt first to load cached blosc=====
    arr = blosc_cache_load(file_path, params)
    if arr is not None:
        return arr

    # Normal load================
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
    # write blosc cache
    blosc_cache_save(file_path, params, arr)

    return arr


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
    # attempt first to load cached blosc
    params = {"year": year, "convert_water_to_zero": convert_water_to_zero}
    arr = blosc_cache_load(file_path, params)
    if arr is not None:
        return arr

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

    # write blosc cache
    blosc_cache_save(file_path, params, values)
    return values


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
    # Try blosc load
    arr_ds = blosc_cache_load(file_path, params={"arr": "ds"})
    arr_se_ds = blosc_cache_load(file_path, params={"arr": "se_ds"})
    if arr_ds is not None and arr_se_ds is not None:
        return arr_ds, arr_se_ds

    # if not, normal load
    with open_file_as_dataset(file_path, ds_type="dataset") as ds:
        agb_ds = ds["agb"].compute().values
        agb_se_ds = ds["agb_se"].compute().values

    # Save blosc cache
    blosc_cache_save(file_path, params={"arr": "ds"}, x=agb_ds)
    blosc_cache_save(file_path, params={"arr": "se_ds"}, x=agb_se_ds)

    return agb_ds, agb_se_ds


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
    arr = blosc_cache_load(file_path, params={})
    if arr is not None:
        return arr

    with open_file_as_dataset(file_path) as ds:
        arr = ds.compute()

    # write blosc cache
    blosc_cache_save(file_path, params={}, x=arr.values)

    return arr.values


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

    arr = blosc_cache_load(file_path, params={})
    if arr is not None:
        return arr

    with open_file_as_dataset(file_path) as ds:
        arr = ds.compute()

    # write blosc cache
    blosc_cache_save(file_path, params={}, x=arr.values)

    return arr.values


def load_s2rgbm(
    file_path: str,
    months: str | list[str] | list[list[str]],
    ignore_blosc_cache=False,
) -> np.ndarray:
    # attempt first to load cached blosc
    params = {"months": months}
    arr = blosc_cache_load(file_path, params)
    if arr is not None:
        return arr

    if isinstance(months, str):
        months = [months]

    has_groups = isinstance(months[0], list)

    if has_groups and not sum(isinstance(i, list) for i in months) == len(months):
        raise ValueError("if using month groups 'months' must be a list of lists")

    if has_groups:
        months_flattened = [i for j in months for i in j]
    else:
        months_flattened = months

    for month in months_flattened:
        if not month in all_months:
            raise ValueError(f"invalid month {month}")

    def getband(ds, band):
        band_idx = S2RGBM_BANDS.index(band)
        ds = ds.isel(band=band_idx)
        return ds.values / 255

    arr = []
    with open_file_as_dataset(file_path) as ds:
        for month in months_flattened:
            r = getband(ds, f"{month}_red")
            g = getband(ds, f"{month}_green")
            b = getband(ds, f"{month}_blue")
            arr += [r, g, b]
        arr = np.r_[arr]

    # average per month group if needed
    if has_groups:
        n_channels = 3
        idxs = [0] + list(np.cumsum([len(i) for i in months]))
        _arr = []
        for ii in range(len(idxs) - 1):
            for ci in range(n_channels):
                pidxs = np.arange(idxs[ii], idxs[ii + 1]) * n_channels + ci
                _arr.append(arr[pidxs].mean(axis=0))
        arr = np.r_[_arr]

    # if normalizer is not None:
    #    arr = normalizer.normalize(arr)

    # write blosc cache
    blosc_cache_save(file_path, params, arr)

    return arr
