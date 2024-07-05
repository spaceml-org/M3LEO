from __future__ import annotations

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from fdlsar import utils


def double_conv(in_channels, out_channels, padding_mode="reflect"):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def down(in_channels, out_channels):
    return nn.Sequential(nn.MaxPool2d(2), double_conv(in_channels, out_channels))


class up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranpose2d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )

        self.conv = double_conv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # [?, C, H, W]
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)  ## why 1?
        return self.conv(x)


class UnetDownConvs(pl.LightningModule):
    """
    the downsampling part of a Unet
    """

    def __init__(
        self,
        in_channels=3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.in_channels = in_channels

        self.input = double_conv
        self.inc = double_conv(self.in_channels, 32)  # self.in_channels, 64
        self.down1 = down(32, 64)  # 64, 128
        self.down2 = down(64, 128)  # 128, 128
        self.down3 = down(128, 256)  # 128, 128
        self.down4 = down(256, 512)  # 128, 64

    def forward(self, x):
        x1 = self.inc(x)  # x1 dim 32
        x2 = self.down1(x1)  # x2 dim 64
        x3 = self.down2(x2)  # x3 dim 128
        x4 = self.down3(x3)  # x4 dim 256
        x5 = self.down4(x4)  # x5 dim 512

        return x1, x2, x3, x4, x5


class UnetDownConvs_simple(pl.LightningModule):
    """
    simple implementation of the downsampling part of a Unet.
    only one maxpool layer used
    """

    def __init__(
        self,
        in_channels=3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.in_channels = in_channels

        # Encoder
        self.inc = double_conv(self.in_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        return x


class UnetUpConvs(pl.LightningModule):
    def __init__(self, n_classes=11, kernel_size=1, stride=1):
        super().__init__()
        self.save_hyperparameters()
        self.n_classes = n_classes
        self.kernel_size = kernel_size
        self.stride = stride
        self.up1 = up(768, 256)  # 192, 128, dim_x5 + dim_x4 = 512 + 256 = 768
        self.up2 = up(384, 128)  # 256, 128, dim_up1 + dim_x3 = 256 + 128 = 384
        self.up3 = up(192, 64)  # 256, 64, dim_up2 + dim_x2 = 128 + 64 = 192
        self.up4 = up(96, 32)  # 128, 64, dim_up3 + dim_x1 = 64 + 32 = 96
        self.out = nn.Conv2d(32, self.n_classes, self.kernel_size, self.stride)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.out(x)
        return x


class UnetUpConvs_simple(pl.LightningModule):
    def __init__(self, n_classes=11):
        super().__init__()
        self.save_hyperparameters()
        self.n_classes = n_classes
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.n_classes, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


class Unet(pl.LightningModule):
    def __init__(self, downconvs, upconvs):
        super().__init__()
        self.save_hyperparameters()

        self.downconvs = downconvs
        self.unconvs = upconvs

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.downconvs(x)
        x = self.unconvs(x1, x2, x3, x4, x5)
        return x


class Unet_simple(pl.LightningModule):
    def __init__(self, downconvs=None, upconvs=None):
        super().__init__()
        self.save_hyperparameters()

        self.downconvs = downconvs
        self.unconvs = upconvs

    def update_convs(self, downconvs, upconvs):
        self.downconvs = downconvs
        self.unconvs = upconvs

    def forward(self, x):
        x = self.downconvs(x)
        x = self.unconvs(x)
        return x


class UnetEncoder(pl.LightningModule):
    def __init__(
        self, input_shape, output_activation="elu", activation="relu", output_dim=128
    ):
        """
        input_shape must be [channels, H, W]
        """
        super().__init__()
        self.save_hyperparameters()
        self.activation = activation
        self.activation_fn = utils.get_activation_fn(activation)

        self.output_activation = output_activation
        self.output_activation_fn = utils.get_activation_fn(output_activation)

        self.output_dim = output_dim
        self.input_shape = input_shape

        self.downconvs = UnetDownConvs(input_shape[0])
        n_sizes = self._get_output_shape(input_shape)
        # linear layers for classifier head
        self.fc1 = nn.Linear(n_sizes, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def _get_output_shape(self, shape):
        """returns the size of the output tensor from the conv layers"""
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self.downconvs(input)[-1]
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def forward(self, x):
        """produce final model output"""
        x = self.downconvs(x)[-1]
        x = x.view(x.size(0), -1)
        x = self.activation_fn(self.fc1(x))
        x = self.output_activation_fn(self.fc2(x))
        return x
