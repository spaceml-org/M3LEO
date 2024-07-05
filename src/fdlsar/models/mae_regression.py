from __future__ import annotations

import lightning.pytorch as pl
import torch
from loguru import logger
from torch import nn
import numpy as np

from .mae_utils import masked_autoencoder, vision_transformer

vit_arch_mapping = {
    "vit_t_sar_16": vision_transformer.vit_t_sar_16,
    "vit_t_sar_32": vision_transformer.vit_t_sar_32,
    "vit_s_sar_16": vision_transformer.vit_s_sar_16,
    "vit_s_sar_32": vision_transformer.vit_s_sar_32,
    "vit_b_sar_16": vision_transformer.vit_b_sar_16,
    "vit_b_sar_32": vision_transformer.vit_b_sar_32,
    "vit_l_sar_16": vision_transformer.vit_l_sar_16,
    "vit_l_sar_32": vision_transformer.vit_l_sar_32,
    "vit_h_sar_14": vision_transformer.vit_h_sar_14,
}


class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class MaskedAutoEncoder_Regression(pl.LightningModule):
    def __init__(
        self,
        backbone=None,
        vit=None,
        num_channels=1,
        decoder_type="fc_linear",
        regression_type="mean",
        learning_rate=1.5e-4
    ):
        super().__init__()

        if backbone is None and vit is None:
            raise ValueError("Must provide either load_from or vit_arch")

        self.num_channels = num_channels  # Changed from 3 in the paper implementation
        self.regression_type = regression_type
        self.learning_rate = learning_rate

        if backbone is not None:
            logger.info(f"Loading pretrained MAE encoder...")
            self.backbone = backbone.model
            self.patch_size = self.backbone.patch_size
        else:
            logger.info(f"Training from scratch...")
            vit_model = vit_arch_mapping[vit](num_channels=self.num_channels)
            self.backbone = masked_autoencoder.MaskedAutoEncoderBackbone.from_vit(
                vit_model
            )
            self.patch_size = vit_model.patch_size

        # Note - setting sequence length ~shouldn't~ matter. Used at pretraining to: define size of pos encodings, no. masked tokens.
        # Pos encodings will be interpolated if downstream sequence length is different.

        self.hidden_dim = self.backbone.hidden_dim
        self.image_size = self.backbone.image_size
        self.sequence_length = ((self.image_size // self.patch_size) ** 2) + 1

        self.decoder_type = decoder_type
        self.init_decoder()
        self.criterion = self.rmse_loss

    def init_decoder(self):
        # Encoder o/p shape #Shape = (batch_size, sequence_length, hidden_dim) [8, 196, 768][vit_b_32]
        valid_types = ["fc_linear", "conv"]

        if not self.decoder_type in valid_types:
            raise ValueError(f"Decoder must be one of {valid_types}")
        logger.info(f"Using {self.decoder_type} decoder...")

        if self.decoder_type == "fc_linear":
            self.decoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.sequence_length * self.hidden_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.ELU(),
            )
        elif self.decoder_type == "conv":
            conv_out_channels_1 = 196
            conv_out_channels_2 = self.hidden_dim // 4

            self.decoder = nn.Sequential(
                nn.Conv1d(
                    in_channels=self.sequence_length,  # Reduce num channels from seq length to 196
                    out_channels=conv_out_channels_1,
                    kernel_size=1,
                ),
                Permute(0, 2, 1),
                nn.Conv1d(
                    in_channels=self.hidden_dim,  # Reduce hidden dim -> conv_out_channels (hidden dim // 4)
                    out_channels=conv_out_channels_2,
                    kernel_size=1,
                ),
                Permute(0, 2, 1),
                nn.Flatten(),
                nn.Linear(conv_out_channels_1 * conv_out_channels_2, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.ELU(),
            )

    def rmse_loss(self, x_pred, targets):
        loss = torch.sqrt(torch.mean((x_pred - targets) ** 2))
        return loss

    def on_fit_start(self):
        """
        Called by lightning at the beginning of training. This is the point at which the logger is available, so
        we can now log parameters to wandb.
        """
        self.logger.experiment.log({"decoder stype": self.decoder_type})
        super().on_fit_start()

    def forward_encoder(self, images):
        # image.shape = [batch_size, n_channels, img_size, img_size]
        return self.backbone.encode(images)

    def forward_decoder(self, x_encoded):
        x = self.decoder(x_encoded)
        return x

    def compute_loss(self, batch, batch_idx=None):
        images = list(batch.values())[0]
        labels = list(batch.values())[1]

        if self.regression_type == "mean":
            targets = (
                labels.reshape(len(labels), -1)
                .type(torch.float32)
                .nanmean(axis=1)
                .reshape(len(labels), 1)
            )
        elif self.regression_type == "sum":
            targets = (
                labels.reshape(len(labels), -1)
                .type(torch.float32)
                .nansum(axis=1)
                .reshape(len(labels), 1)
            )

        x_encoded = self.forward_encoder(images)
        # Shape = (batch_size, sequence_length, hidden_dim) [8, 12, 768]

        x_pred = self.forward_decoder(x_encoded)

        loss = self.criterion(x_pred, targets)

        return loss

    def forward(self, x):
        x_encoded = self.forward_encoder(x)
        # Shape = (batch_size, sequence_length, hidden_dim) [8, 12, 768]
        x_pred = self.forward_decoder(x_encoded)

        return x_pred

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, batch_idx)

        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, batch_idx)

        self.log(
            "val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, batch_idx)

        self.log(
            "test/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optim
