from __future__ import annotations

import numpy as np
from loguru import logger
from torch.utils.data import Dataset

from fdlsar import utils


class InputsAndLabelsDataset(Dataset):
    """
    a dataset made by combining two base datasets, one named 'inputs' and another one named 'labels'
    both datasets must contain common tiles

    """

    def __init__(
        self,
        inputs_class,
        inputs_kwargs,
        labels_class,
        labels_kwargs,
        split_file,
        split,
        transforms=None,
    ):
        """
        inputs_class, inputs_kwargs: class and kwargs to pass to constructor
                                     inputs_class can be a class object or an str with the class name
        labels_class, labels_kwargs: class and kwargs to pass to constructor
                                     labels_class can be a class object or an str with the class name
        split_file: the csv file containing the split definitions
        split: one of 'train', 'test', 'val'
        transforms: a callable that applies transformations to the inputs and labels
        """
        self.transforms = transforms
        # if classes are strings with class names, then eval them
        if isinstance(inputs_class, str):
            inputs_class = utils.get_class(inputs_class)

        if isinstance(labels_class, str):
            labels_class = utils.get_class(labels_class)

        # instantiate datasets
        self.inputs_dataset = inputs_class(
            split_file=split_file, split=split, **inputs_kwargs
        )
        self.labels_dataset = labels_class(
            split_file=split_file, split=split, **labels_kwargs
        )

        # inspects the files (tiles) of each dataset and retains
        # those that are commong to both
        self.common_tileids = list(
            set(self.inputs_dataset.get_tileids()).intersection(
                self.labels_dataset.get_tileids()
            )
        )

        if len(self.common_tileids) == 0:
            raise ValueError(f"no common tiles in {self}")

        logger.info(f"{self}: using {len(self.common_tileids)} common tiles")

        # tells each dataset what tiles to use
        self.inputs_dataset.set_tileids(self.common_tileids)
        self.labels_dataset.set_tileids(self.common_tileids)

    def prepare_data(self):
        """Input preprocessing step and label preprocessing step are applied."""
        self.inputs_dataset.prepare_data()
        self.labels_dataset.prepare_data()

    def __len__(self):
        return len(self.common_tileids)

    def __repr__(self):
        return f"{self.inputs_dataset}/{self.labels_dataset}"

    def __getitem__(self, idx):
        """
        simply calls getitem in each dataset and returns the combined results.
        if errors in inputs or labels it attempts other tiles at most 20 times
        """
        max_attepts = 20
        attempts = 1

        while True:
            try:
                inputs, labels = self.inputs_dataset[idx], self.labels_dataset[idx]
                if inputs is None or labels is None:
                    raise ValueError(f"'None' type found in inputs or labels")

                if self.transforms is not None:
                    transform_output = self.transforms(
                        {"inputs": inputs, "labels": labels}
                    )
                    inputs, labels = (
                        transform_output["inputs"],
                        transform_output["labels"],
                    )

                return inputs, labels

            except Exception as e:
                logger.info(e)
                logger.info(self.common_tileids[idx])
                if attempts == max_attepts:
                    raise e

            if attempts == 1:
                logger.warning(
                    f"error loading tileid {self.common_tileids[idx]}, attempting with other tiles"
                )

            attempts += 1
            idx = np.random.randint(len(self))
