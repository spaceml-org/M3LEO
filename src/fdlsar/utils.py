from __future__ import annotations

import functools
import hashlib
import os
import pickle
import warnings
from functools import wraps
from glob import glob

import blosc
import dotenv
import gcsfs
import hydra
import lightning.pytorch as pl
import netCDF4 as nc
import numpy as np
import torch
import wandb
from joblib import Memory
from loguru import logger
from omegaconf import OmegaConf
from progressbar import progressbar as pbar
from torch.nn import functional as F
from tqdm import tqdm

fs = gcsfs.GCSFileSystem()

dotenv.load_dotenv()


def find_hydra_run_path(outputs_dir, wandb_runid):
    """
    find the hydra run parh that contains the wandb runid
    """

    files = glob(f"**/*{wandb_runid}*", root_dir=outputs_dir, recursive=True)
    if len(files) == 0:
        raise ValueError("no file found")

    files = [f for f in files if "/wandb/" in f]
    if len(files) == 0:
        raise ValueError("no wandb log found")

    r = outputs_dir + "/" + files[0].split("/wandb/")[0]
    return r


def recompute_loss(wandb_runid, outputs_dir, split, update_wandb=True, num_workers=48):
    """
    recomputes the loss of a given model from a hydra run and
    stores it in wandb

    outputs_dir: the folder where hydra keeps runs
    """

    if not split in ["train", "val", "test"]:
        raise ValueError("split must be train, val or test")

    logger.info("looking for hydra run")
    hydra_runpath = find_hydra_run_path(
        outputs_dir=outputs_dir, wandb_runid=wandb_runid
    )

    logger.info("loading model")
    m = load_ckpt_from_hydra_run(hydra_runpath)
    m = m.cuda()

    logger.info("loading dataloader and prepareing data")
    d = load_dataloader_from_hydra_run(hydra_runpath)
    d.prepare_data()

    d.hparams.num_workers = num_workers

    if split == "test":
        dataloader = d.test_dataloader()
    elif split == "train":
        dataloader = d.train_dataloader()
    elif split == "val":
        dataloader = d.val_dataloader()

    logger.info("recomputing loss")

    losses = []
    for batch in pbar(dataloader, max_value=len(dataloader)):
        # required to avoid filling up GPU memory
        with torch.no_grad():
            # take batch to gpu
            batch = {k: v.cuda() for k, v in batch.items()}

            # compute and accumulate loss
            loss = m.compute_loss(batch)
            losses.append(loss)

    loss_value = np.mean([i.detach().cpu().numpy() for i in losses])
    logger.info(f"wandb run {wandb_runid} {split} loss: {loss_value}")

    if update_wandb:
        logger.info("updating wandb")
        api = wandb.Api()
        run = api.run(f"fdlsar/baselines/{wandb_runid}")
        run.summary[f"{split}/loss_epoch"] = loss_value
        run.summary.update()


def optional_memory_cache(func=None):
    """
    decorator to optionally cache a function using joblib memory.
    if CACHE_DIR is not set, the function is not cached.
    To set the cache directory, set the CACHE_DIR environment variable in a .env file

    func: the function to cache

    returns: the cached function
    """
    location = os.getenv("CACHE_DIR")
    if func is None:
        return lambda f: optional_memory_cache(f)

    if location is not None:
        memory = Memory(location=location, verbose=0, compress=9)
        func = memory.cache(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)

    return wrapper


def get_class(class_name):
    """
    returns a class object from a fully specified class name by importing required modules
    """

    z = class_name.split(".")
    import_path = ".".join(z[:-1])
    import_class = z[-1]
    exec(f"from {import_path} import {import_class}")
    eval_fn = eval  # skips pre-commit check
    return eval_fn(import_class)


def get_activation_fn(activation_str):
    if activation_str == "relu":
        return F.relu
    elif activation_str == "elu":
        return F.elu
    elif activation_str == "linear":
        return lambda x: x

    raise ValueError(f"unknown activation '{activation_str}'")


def check_file_exists(file: str) -> bool:
    """
    checks if a file exists, works for local files and files in a bucket
    """
    if file.startswith("gs://"):
        return check_file_exists_in_bucket(file)
    else:
        return os.path.isfile(file)


def check_dir_exists(dir: str) -> bool:
    """
    checks if a directory exists, works for local directories and directories in a bucket
    """
    if dir.startswith("gs://"):
        return check_file_exists_in_bucket(dir)
    else:
        return os.path.isdir(dir)


def check_file_exists_in_bucket(file: str) -> bool:
    """
    checks if a file exists in a bucket
    """
    return fs.exists(file)


def list_all_files_in_bucket(bucket_name: str) -> list:
    """
    returns a list of all files in a bucket

    bucket_name: name of the bucket

    returns: list of files in the bucket
    """
    file_list = fs.ls(bucket_name)
    return file_list


def list_all_files(dir: str) -> list:
    """
    returns a list of all files in a directory, works for local directories and directories in a bucket
    """
    if dir.startswith("gs://"):
        return list_all_files_in_bucket(dir)
    else:
        return os.listdir(dir)


def get_full_file_path(
    chips_path: str, dataset_name: str, identifier: str, extension: str = "tif"
) -> str:
    """
    returns the full path to a file in a bucket

    chips_path: path to the chips folder
    dataset_name: name of the dataset
    identifier: identifier of the file
    extension: extension of the file

    returns: full path to the file
    """
    return f"{chips_path}/{dataset_name}/{identifier}.{extension}"


def load_ckpt_from_hydra_run(
    hydra_run_path: str,
    loading_from_state_dict=True,
    enable_loading_weights: bool = True,
    model: pl.LightningModule = None,
    config=None,
) -> pl.LightningModule:
    """
    loads a checkpoint model from a run's output hydra log

    hydra_run_path: the file path to the hydra run
    loading_from_state_dict: if True, load model using state_dict, otherwise use PyTorch Lightning checkpoint
    config: pass config if already loaded
    model: pass model if already instantiated

    returns: a pytorch lighting module
    """

    # load config
    if config is None:
        config_file = f"{hydra_run_path}/.hydra/config.yaml"
        if not os.path.isfile(config_file):
            raise ValueError(f"config file {config_file} not found")

        config = OmegaConf.load(config_file)

    # look for checkpoint
    ckpts_paths = [
        f"{hydra_run_path}/*{'/*'*trailings}/*ckpt" for trailings in range(6)
    ]
    ckpts = functools.reduce(
        lambda lista, elemento: glob(elemento) + lista, ckpts_paths, []
    )
    ckpts = sorted(ckpts)
    print(ckpts)
    if len(ckpts) == 0:
        raise ValueError(f"no checkpoints found in {hydra_run_path}")

    if len(ckpts) > 1:
        print(f"there are {len(ckpts)} checkpoints, attempting to use the best one")
        try:
            best_ckpt = glob(f"{hydra_run_path}/*/*best*")[0]
        except IndexError:  # if no "best" model exists
            print(f"could not load best model, attempting last model instead")
            best_ckpt = ckpts[-1]
    else:
        print(f"there is {len(ckpts)} checkpoint")
        best_ckpt = ckpts[-1]

    logger.info(f"loaded checkpoint: {best_ckpt}")

    if model is None:
        logger.info("Creating Model")
        # instantiate model class
        model = hydra.utils.instantiate(config.model)

    # load model
    if enable_loading_weights:
        logger.info("Loading weights from model checkpoint")
        if loading_from_state_dict:
            logger.info("Loading using state_dict")
            checkpoint = torch.load(best_ckpt)
            model.load_state_dict(checkpoint["state_dict"])
        else:
            logger.info("Loading using checkpoint from PyTorch Lightning")
            model = model.load_from_checkpoint(best_ckpt)

        logger.info("---------------------------------")
        logger.info(f"model checksum  {print_checksum_of_model(model):.4f}")
        logger.info("---------------------------------")
    else:
        logger.warning(
            "The weights of the pretrained model are not loaded. The model will be initialized with random weights."
        )
        logger.warning("---------------------------------")
        logger.warning(f"model checksum  {print_checksum_of_model(model):.4f}")
        logger.info("---------------------------------")

    return model


def print_checksum_of_model(model):
    return sum(torch.abs(p).sum() for p in model.parameters()).detach().cpu().numpy()


def load_dataloader_from_hydra_run(
    hydra_run_path: str, path_replace=None
):  # -> pl.LightningDataModule:
    """from lightning.pytorch import LightningDataModule

    loads a dataloader from a run's output hydra log

    hydra_run_path: the file path to the hydra run

    path_replace: a dict with strings to replace in split file

    returns: a pytorch lighting dataloader
    """
    # load config
    config_file = f"{hydra_run_path}/.hydra/config.yaml"
    if not os.path.isfile(config_file):
        raise ValueError(f"config file {config_file} not found")

    config = OmegaConf.load(config_file)

    s = config.dataloader.split_file
    if path_replace is not None:
        for k, v in path_replace.items():
            s = s.replace(k, v)
    config.dataloader["split_file"] = s

    # instantiate model class
    dataloader = hydra.utils.instantiate(config.dataloader)
    return dataloader


def get_hash(s: str):
    """
    s: a string
    returns a hash string for s
    """
    if not isinstance(s, str):
        s = str(s)
    k = int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % 10**15
    k = str(hex(k))[2:].zfill(13)
    return k


class MeanStdNormalizer:
    """
    an utility class to normalize per band mean/std

    it requires Dataset to implement load_item(idx), and the Dataset __get_item__ must call it after load_item and before the transforms
    """

    def __init__(
        self,
        dataset,
        params,
        n_samples=1000,
        force_compute=False,
        out_dtype=np.float32,
        percentiles=None,
    ):
        """
        dataset: the dataset to normalize
        params: the params combination for which to create normalization means/stds
                they are only used to create a hash value so that there is a different set
                of normalization constants for each params combination
        """
        self.dataset = dataset
        self.params = params
        self.hash = get_hash(str(params))
        self.n_samples = n_samples
        self.force_compute = force_compute
        self.percentiles = percentiles
        self.stats_file = f"{self.dataset.dataset_folder}_stats_{self.hash}.pkl"

        if (
            "gs://" in self.stats_file
            or "/mnt/" in self.stats_file
            or "/sardata/" in self.stats_file
        ):
            self.stats_file = self.stats_file.replace("gs://", "/tmp/")
            self.stats_file = self.stats_file.replace("/mnt/", "/tmp/")
            self.stats_file = self.stats_file.replace("/sardata/", "/sardata/stats/")
            os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)
            logger.info(f"Creating directory for stats_file as {self.stats_file}")

        self.out_dtype = out_dtype

    def prepare_data(self):
        if not os.path.isfile(self.stats_file) or self.force_compute:
            if self.dataset.split != "train":
                raise ValueError(
                    "train normalization data not found. run prepare_data on 'train' dataset first"
                )

            logger.info(
                f"{self.dataset.__class__.__name__}.{self.params}.{self.dataset.split}.{self.hash}: computing dataset stats"
            )

            x_sum = None
            xsq_sum = None

            n = 0
            failed_files = 0
            for idx in pbar(np.random.permutation(len(self.dataset))[: self.n_samples]):
                try:
                    x = self.dataset.load_item(idx)
                    if self.percentiles is not None:
                        perc_a = np.percentile(
                            x, self.percentiles[0], axis=(-2, -1)
                        ).reshape(-1, 1, 1)
                        perc_b = np.percentile(
                            x, self.percentiles[1], axis=(-2, -1)
                        ).reshape(-1, 1, 1)
                        x = (x - perc_a) / (perc_b - perc_a)
                except Exception as e:
                    logger.warning(f"failed to load item {idx} in {self.dataset}: {e}")
                    failed_files += 1
                    logger.warning(
                        f"{failed_files} out of {self.n_samples} failed to load. {e}"
                    )
                    continue

                if x_sum is None:
                    x_sum = np.zeros(x.shape[0])
                    xsq_sum = np.zeros(x.shape[0])

                xn = np.product(x.shape[1:])
                x_sum += np.nansum(x.reshape(-1, xn), axis=-1)
                xsq_sum += np.nansum((x**2).reshape(-1, xn), axis=-1)
                n += xn

            self.data_shape = x.shape

            # compute mean and std per channel
            self.means = x_sum / n
            self.stds = np.sqrt(xsq_sum / n - self.means**2)

            # save file
            with open(self.stats_file, "wb") as f:
                pickle.dump(
                    {
                        "means": self.means,
                        "stds": self.stds,
                        "data_shape": self.data_shape,
                    },
                    f,
                )

            logger.info(f"dataset stats saved to {self.stats_file}")

        logger.info(f"reading stats from {self.stats_file}")
        logger.info(
            f"    for config {self.dataset.__class__.__name__}.{self.params}.{self.dataset.split}.{self.hash}"
        )
        with open(self.stats_file, "rb") as f:
            k = pickle.load(f)
            self.means = k["means"]
            self.stds = k["stds"]
            self.data_shape = k["data_shape"]

        self.means = self.means.reshape(
            self.data_shape[0], *([1] * len(self.data_shape[1:]))
        ).astype(self.out_dtype)

        self.stds = self.stds.reshape(
            self.data_shape[0], *([1] * len(self.data_shape[1:]))
        ).astype(self.out_dtype)

    def normalize(self, x):
        if self.percentiles is not None:
            perc_a = np.percentile(x, self.percentiles[0], axis=(-2, -1)).reshape(
                -1, 1, 1
            )
            perc_b = np.percentile(x, self.percentiles[1], axis=(-2, -1)).reshape(
                -1, 1, 1
            )
            x = (x - perc_a) / (perc_b - perc_a)
        return (x - self.means) / (self.stds + 1e-5)


class GunwMeanStdNormalizer:
    """
    an utility class to normalize per band mean/std

    it requires Dataset to implement load_item(idx), and the Dataset __get_item__ must call it after load_item and before the transforms
    """

    def __init__(self, dataset, n_samples=1000, force_compute=False):
        """
        dataset: the dataset to normalize
        """
        self.dataset = dataset
        self.n_samples = n_samples
        self.force_compute = force_compute

        self.stats_file = f"{self.dataset.dataset_folder}_stats.pkl"

        if "gs://" in self.stats_file or "/mnt/" in self.stats_file:
            self.stats_file = self.stats_file.replace("gs://", "/tmp/")
            self.stats_file = self.stats_file.replace("/mnt/", "/tmp/")
            os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)
            logger.info(f"Creating directory for stats_file as {self.stats_file}")

        self.variables_to_normalize = ["amplitude", "unwrappedPhase", "coherence"]

    def prepare_data(self):
        if not os.path.isfile(self.stats_file) or self.force_compute:
            if self.dataset.split != "train":
                raise ValueError(
                    "train normalization data not found. run prepare_data on 'train' dataset first"
                )

            logger.info(
                f"{self.dataset.__class__.__name__}.{self.dataset.split}: computing dataset stats"
            )

            x_sum = {k: 0 for k in self.variables_to_normalize}
            xsq_sum = {k: 0 for k in self.variables_to_normalize}
            n = {k: 0 for k in self.variables_to_normalize}

            failed_files = 0
            for idx in pbar(np.random.permutation(len(self.dataset))[: self.n_samples]):
                file_path = (
                    f"{self.dataset.dataset_folder}/{self.dataset.get_files()[idx]}"
                )
                try:
                    d = nc.Dataset(file_path)["science"]["grids"]["data"]

                    for v in self.variables_to_normalize:
                        x = d[v][:].compressed()
                        x_sum[v] += x.sum()
                        xsq_sum[v] += (x**2).sum()
                        n[v] += len(x)

                except:
                    logger.warning(f"failed to load item {idx} in {self.dataset}")
                    failed_files += 1
                    logger.warning(
                        f"{failed_files} out of {self.n_samples} failed to load."
                    )
                    continue

            # compute mean and std per channel
            self.means = {k: x_sum[k] / n[k] for k in n.keys()}
            self.stds = {
                k: np.sqrt(xsq_sum[k] / n[k] - self.means[k] ** 2) for k in n.keys()
            }

            # save file
            with open(self.stats_file, "wb") as f:
                pickle.dump(
                    {
                        "means": self.means,
                        "stds": self.stds,
                    },
                    f,
                )

            logger.info(f"dataset stats saved to {self.stats_file}")

        logger.info(f"reading stats from {self.stats_file}")
        logger.info(
            f"    for config {self.dataset.__class__.__name__}.{self.dataset.split}"
        )
        with open(self.stats_file, "rb") as f:
            k = pickle.load(f)
            self.means = k["means"]
            self.stds = k["stds"]

        logger.info(
            f"normalization constants found for variables {list(self.means.keys())}"
        )

    def normalize(self, x, variable):
        if variable in list(self.means.keys()):
            x[variable] = (x[variable] - self.means[variable]) / (
                self.stds[variable] + 1e-5
            )
        return x


def invalid_identifiers_s1grdm(data_dir, bands, seconday_datadir=None):
    """
    returns a list of invalid identifiers for s1grdm

    data_dir: the data directory
    bands: a list of bands to check, range from 0 to 23 (vv, vh per month)
    """
    blosc_files = glob(f"{data_dir}/**/*.blosc")
    total_files = len(blosc_files)
    if seconday_datadir is not None:
        secondary_files = glob(f"{seconday_datadir}/**/*.blosc")
        secondary_ids = [f.split("-")[-1].split(".")[0] for f in secondary_files]
        print(secondary_ids[:5])
        primary_ids = [f.split("-")[-1].split(".")[0] for f in blosc_files]
        print(primary_ids[:5])
        blosc_files = [
            f for f in blosc_files if f.split("-")[-1].split(".")[0] in secondary_ids
        ]
        logger.info(f"Using {len(blosc_files)} out of {total_files} files")

    invalid_ids = []
    for f in tqdm(blosc_files):
        f_id = f.split("-")[-1].split(".")[0]
        if not os.path.isfile(f):
            invalid_ids.append(f_id)
            continue
        with open(f, "rb") as f:
            arr = blosc.unpack_array(f.read())
            # take bands
            arr = arr[bands]
            std_values = np.std(arr, axis=(-2, -1))
            if (std_values == 0).any() or np.isnan(std_values).any():
                invalid_ids.append(f_id)
            # if there is a band that has more than 10% 0 values, then it is invalid
            if (
                np.sum(arr == 0, axis=(-2, -1)) > 0.1 * arr.shape[1] * arr.shape[2]
            ).any():
                invalid_ids.append(f_id)
    return invalid_ids
