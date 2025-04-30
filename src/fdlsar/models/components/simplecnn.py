from __future__ import annotations

import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn import functional as F


def get_activation_fn(activation_str):
    if activation_str == "relu":
        return F.relu
    elif activation_str == "elu":
        return F.elu
    elif activation_str == "linear":
        return lambda x: x

    raise ValueError(f"unknown activation '{activation_str}'")


class SimpleCNN(pl.LightningModule):
    def __init__(
        self, input_shape, activation="relu", output_activation="elu", output_dim=128
    ):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.activtion_str = activation
        self.output_activation_str = output_activation
        self.output_dim = output_dim

        self.activation_fn = get_activation_fn(activation)
        self.output_activation_fn = get_activation_fn(output_activation)

        # model architecture
        self.conv1 = nn.Conv2d(input_shape[0], 32, 4, stride=1)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv4 = nn.Conv2d(64, 64, 4, stride=2)

        self.pool1 = torch.nn.MaxPool2d(2)
        self.pool2 = torch.nn.MaxPool2d(2)

        n_sizes = self._get_output_shape(input_shape)

        # linear layers for classifier head
        self.fc1 = nn.Linear(n_sizes, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def _get_output_shape(self, shape):
        """returns the size of the output tensor from the conv layers"""

        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._feature_extractor(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    # computations
    def _feature_extractor(self, x):
        """extract features from the conv blocks"""
        x = self.activation_fn(self.conv1(x))
        x = self.pool1(self.activation_fn(self.conv2(x)))
        x = self.activation_fn(self.conv3(x))
        x = self.pool2(self.activation_fn(self.conv4(x)))
        return x

    def forward(self, x):
        """produce final model output"""
        x = self._feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_fn(self.fc2(x))
        x = self.activation_fn(self.fc3(x))
        return x
