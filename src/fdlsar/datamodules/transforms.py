from __future__ import annotations

import ast
import random

import numpy as np
import torch
import xarray as xr
from scipy.ndimage import zoom
from torchvision.transforms import Compose, Resize
from torchvision.transforms.functional import rotate


# ------- general transforms -------
class AvgKey:
    """Average input image over time.

    Args:
        keys (list): List of keys to apply avg.
        axis (int): Dimension to apply avg.

    """

    def __init__(self, keys, axis):
        self.keys = keys
        self.axis = axis

    def __call__(self, sample):
        for key in self.keys:
            inputs = sample[key]
            inputs = np.mean(inputs, axis=self.axis)
            sample[key] = inputs
        return sample


class AvgBands:
    """
    Take the average of the bands in a tensor
    """

    def __call__(self, array):
        mean_t = torch.mean(array, dim=0, keepdim=True)
        return mean_t


class Mean:
    """Simply returns the mean of everything"""

    def __call__(self, sample):
        return sample.mean()


class FillNanByKey:
    """Fill NaN values with a given value.

    Args:
        value (float): Value to fill NaNs with.

    """

    def __init__(self, value, key="inputs"):
        self.value = value
        self.key = key

    def __call__(self, sample):
        item = sample[self.key]
        item[torch.isnan(item)] = self.value
        sample[self.key] = item
        return sample


class ResizeAntialias:
    """Resizes to a given size.

    Args:
        size (int or tuple): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)

    """

    def __init__(self, size):
        self.resize = Resize(size, antialias=True)

    def __call__(self, sample):
        # unsqueeze to add dimension
        if len(sample.shape) == 2:
            sample = sample.unsqueeze(0)
        sample = self.resize(sample)
        return sample


class ResizeByKey:
    """Resize any image to a given size.

    Args:
        size (int or tuple): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)

    """

    def __init__(self, size, key="inputs"):
        self.resize = Resize(size, antialias=True)
        self.key = key

    def __call__(self, sample):
        item = sample[self.key]

        # unsqueeze to add dimension for one channel images with no extra dim
        if item.dim() == 2:
            item = item.unsqueeze(0)

        item = self.resize(item)
        sample[self.key] = item
        return sample


class ItemToTensor:
    """Converts numpy arrays to tensors all with the same type.

    Args:
        keys (list): list of keys to convert to tensors
        dtype (torch.dtype): data type of the tensor

    """

    def __init__(self, keys, dtype):
        self._keys = keys
        self._dtype = dtype

    def __call__(self, item: dict):
        for key in self._keys:
            print(key)
            sample = item[key]
            if isinstance(sample, xr.DataArray):
                sample = sample.values
            sample = torch.as_tensor(sample, dtype=self._dtype)
            item[key] = sample

        return item


class ToTensor:
    """
    Converts a numpy array to a tensor

    Args:
        dtype (torch.dtype): data type of the tensor
    """

    def __init__(self, dtype=torch.float32):
        self._dtype = dtype

    def __call__(self, array):
        r = torch.as_tensor(array, dtype=self._dtype)
        return r


class ScaleMean:
    """
    Preserves the mean of a resized tensor
    """

    def __call__(self, array, original_mean):
        mean_t = torch.nanmean(array, dim=(-1, -2))
        if mean_t != 0.0 and original_mean != 0.0:
            array = array * (original_mean / mean_t)
        return array


class ScaleSum:
    """
    Preserves the sum of a resized tensor
    """

    def __call__(self, array, original_sum):
        sum_t = torch.nansum(array, dim=(-1, -2))
        if sum_t != 0.0 and original_sum != 0.0:
            array = array * (original_sum / sum_t)
        return array


# ------- data augmentations -------
class RandomPermuteAxisGroups:
    """
    Randomly permute the first axis of a PyTorch tensor by swapping elements within groups.
    """

    def __init__(self, swap_probability: float, group_size: int = None):
        """
        Initialize the permutation transform.

        Args:
            swap_probability (float): The probability of swapping elements within each group.
            group_size (int): The size of the groups for swapping elements.
        """
        self.swap_probability = swap_probability
        if group_size is None:
            raise ValueError("group_size must be specified")
        self.group_size = group_size

    def __call__(self, sample: torch.Tensor):
        """
        Apply the permutation transform to a sample.

        Args:
            sample (torch.Tensor): A sample containing the tensor to permute.

        Returns:
            torch.Tensor: The sample with the permuted tensor.
        """
        permuted_indices = []

        for i in range(0, sample.shape[0], self.group_size):
            group_indices = list(range(i, min(i + self.group_size, sample.shape[0])))
            if random.random() < self.swap_probability:
                random.shuffle(group_indices)
            permuted_indices.extend(group_indices)

        return sample[permuted_indices]


class FineRotation:
    """
    Perform a 10° rotation on the last 2 axis of a PyTorch tensor.
    """

    def __init__(self, angle=10, rotation_probability=0.5):
        """
        Initialize the rotation transform.

        Args:
            angle (int): The angle of rotation.
            rotation_probability (float): The probability of rotation for each pair of indices.
        """
        self.angle = angle
        self.rotation_probability = rotation_probability

    def __call__(self, sample):
        """
        Apply the rotation transform to a sample.

        Args:
            sample (dict): A sample containing the tensor to rotate.

        Returns:
            dict: The sample with the rotated tensor.
        """
        angle = self.angle
        if random.random() < self.rotation_probability:
            if random.random() < 0.5:
                angle = -self.angle
            sample = rotate(sample, angle)
        return sample


class HorizontalFlip:
    """
    Randomly flip the last 2 axis of a PyTorch tensor.
    """

    def __init__(self, flip_probability):
        """
        Initialize the flip transform.

        Args:
            flip_probability (float): The probability of flipping each pair of indices.
        """
        self.flip_probability = flip_probability

    def __call__(self, sample):
        """
        Apply the flip transform to a sample.

        Args:
            sample (dict): A sample containing the tensor to flip.

        Returns:
            dict: The sample with the flipped tensor.
        """
        if random.random() < self.flip_probability:
            sample = torch.flip(sample, dims=(-1,))
        return sample


class SelectRandomChannel:
    def __init__(self, channel_range: tuple[int, int] | None = None):
        """
        Args:
            channel_range (tuple[int, int] | None): range of channels to select from
        """
        self.channel_range = channel_range

    def __call__(self, x):
        if x.shape[0] == 1:
            return x

        # x is (C, H, W)
        if self.channel_range is None:
            channel_range = (0, x.shape[0] - 1)
        else:
            channel_range = self.channel_range

        # select channel range
        x = x[channel_range[0] : channel_range[1] + 1]

        # Calculate standard deviation across H and W for each channel
        channel_std = x.std(dim=(1, 2))

        # Find non-constant channels
        non_constant_channels = torch.where(channel_std > 1e-6)[0]

        # if non_constant_channels is empty, raise error
        if non_constant_channels.numel() == 0:
            raise ValueError("All channels are constant!")

        # Select a random channel from non-constant channels
        random_channel_idx = torch.randint(0, non_constant_channels.numel(), (1,))[0]
        random_channel = non_constant_channels[random_channel_idx]

        # Get the selected channel
        x = x[random_channel : random_channel + 1]

        return x


# ------- normalizations -------
class MinMaxNormalization:
    def __init__(self, key):
        self.key = key

    def __call__(self, sample):
        inputs = sample[self.key]
        # normalize to [0, 1] each band:
        for i in range(inputs.shape[0]):
            inputs[i] = (inputs[i] - inputs[i].min()) / (
                inputs[i].max() - inputs[i].min()
            )
        sample[self.key] = inputs
        return sample


# ------- dataset specific transforms -------
class RemovePermanentWater:
    # remove permanent water from flood maps
    def __call__(self, x):
        # channel 1 is flood, channel 2 is perm water
        x[0] = np.where(x[1] == 1, 0, x[0])
        return (x[0] > 0).astype(np.uint8)


class LandCoverResize:
    """
    Resizes any Land Cover data while preserving classes.
    """

    def __init__(self, target_size, mode="nearest"):
        self.target_size = target_size
        self.mode = mode

    def __call__(self, image_array):
        try:
            # Calculate the reduction factor along each axis
            reduction_factor = (
                image_array.shape[0] / self.target_size[0],
                image_array.shape[1] / self.target_size[1],
            )

            # Downsample the image array using scipy's zoom function
            downsampled_array = zoom(
                image_array,
                (1 / reduction_factor[0], 1 / reduction_factor[1]),
                order=0,
                mode=self.mode,
            )

            assert downsampled_array.shape == self.target_size
            return downsampled_array

        except Exception as e:
            print(f"An error occurred: {e}")


# ------- Experiment specific transforms -------
class LandCoverTransform:
    """Transform pipeline for any Land Cover data."""

    def __init__(self, target_size):
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        elif isinstance(target_size, str):  # assumes str of tuple
            self.target_size = ast.literal_eval(target_size)
        elif isinstance(target_size, tuple):
            self.target_size = target_size
        else:
            raise ValueError("wrong type for target_size. choices are int, str, tuple.")
        self.transform = Compose(
            [
                LandCoverResize(self.target_size),
                ToTensor(),
            ]
        )

    def __call__(self, sample):
        return self.transform(sample)


class GhsBuiltSTransform:
    """
    Transform pipeline for GhsBuiltS.
    Note, this only preserves the mean/sum if preserve=='mean'/'sum'.
    """

    def __init__(self, size=(50, 66), key="ghsbuilts", preserve=None):
        self.preserve = preserve
        if type(size) == str:
            size = ast.literal_eval(size)
        self.transform = Compose(
            [
                ToTensor(dtype=torch.float32),
                FillNanByKey(0.0, key=key),
                ResizeByKey(size, key),
            ]
        )

    def __call__(self, sample):
        s = self.transform(sample)
        if self.preserve is not None and self.preserve.lower() == "mean":
            s = ScaleMean()(array=s, original_mean=np.nanmean(sample, axis=(-1, -2)))
        elif self.preserve is not None and self.preserve.lower() == "sum":
            s = ScaleSum()(array=s, original_sum=np.nansum(sample, axis=(-1, -2)))
        return s


class GHSBuiltSMean:
    """
    Just computes the mean of all channels
    """

    def __init__(self, key="ghsbuilts"):
        self.transform = Compose(
            [
                ToTensor(dtype=torch.float32),
                FillNanByKey(0.0, key=key),
            ]
        )

    def __call__(self, sample):
        s = self.transform(sample)
        return torch.mean(s)


class GHSBuiltSLogMean:
    """
    Compute the log mean and clip it to zero
    """

    def __call__(self, sample):
        r = np.log(sample.mean() + 1e-5)
        if np.isnan(r):
            r = np.log(1e-5)
        if r < 0:
            r = 0
        return r


class BiomassTransform:
    """Transform pipeline for biomass."""

    def __init__(self, size=(45, 65), key="agb", preserve=None):
        self.preserve = preserve
        if type(size) == str:
            size = ast.literal_eval(size)
        self.transform = Compose(
            [
                ToTensor(dtype=torch.float32),
                FillNanByKey(0.0, key=key),
                ResizeByKey(size, key),
            ]
        )

    def __call__(self, sample):
        s = self.transform(sample)
        if self.preserve is not None and self.preserve.lower() == "mean":
            s = ScaleMean()(array=s, original_mean=np.nanmean(sample, axis=(-1, -2)))
        elif self.preserve is not None and self.preserve.lower() == "sum":
            s = ScaleSum()(array=s, original_sum=np.nansum(sample, axis=(-1, -2)))
        return s


class S2rgbmTransform:
    """
    Transform pipeline for S2RGB monthly data.
    """

    def __init__(self, key="s2rgbm"):
        self.transform = Compose(
            [
                ToTensor(dtype=torch.float32),
                FillNanByKey(0.0, key=key),
            ]
        )

    def __call__(self, sample):
        s = self.transform(sample)
        return s


class GSSICTransform:
    """Transform pipeline for GSSIC coherence only."""

    def __init__(self, size=(448, 448), key="gssic"):
        if type(size) == str:
            size = ast.literal_eval(size)
        self.transform = Compose(
            [
                ItemToTensor(keys=[key], dtype=torch.float32),
                FillNanByKey(0.0, key=key),
                ResizeByKey(size, key),
            ]
        )

    def __call__(self, sample):
        return self.transform(sample)


class GUNWTransform:
    """Transform pipeline for GUNW data."""

    def __init__(self, size=(448, 448), key="gunw"):
        if type(size) == str:
            size = ast.literal_eval(size)
        self.transform = Compose(
            [
                ToTensor(dtype=torch.float32),
                FillNanByKey(0.0, key=key),
                ResizeByKey(size, key),
            ]
        )

    def __call__(self, sample):
        s = self.transform(sample)
        return s


class S1GRDAugment:
    def __init__(self, select_one_random_valid_channel=False):
        self.select_one_random_channel = select_one_random_valid_channel
        self.transform = Compose(
            [
                ToTensor(dtype=torch.float32),
                RandomPermuteAxisGroups(swap_probability=0.5, group_size=3),
                FineRotation(angle=10, rotation_probability=0.5),
                HorizontalFlip(flip_probability=0.5),
            ]
        )

        if select_one_random_valid_channel:
            self.transform = Compose([self.transform, SelectRandomChannel()])

    def __call__(self, sample):
        s = self.transform(sample)
        return s


class S2RGBAugment:
    def __init__(self, select_one_random_valid_channel=False):
        self.select_one_random_channel = select_one_random_valid_channel
        self.transform = Compose(
            [
                ToTensor(dtype=torch.float32),
                RandomPermuteAxisGroups(swap_probability=0.5, group_size=3),
                FineRotation(angle=10, rotation_probability=0.5),
                HorizontalFlip(flip_probability=0.5),
            ]
        )

        if select_one_random_valid_channel:
            self.transform = Compose([self.transform, SelectRandomChannel()])

    def __call__(self, sample):
        s = self.transform(sample)
        return s
