from __future__ import annotations

import lightning.pytorch as pl
import torch
from loguru import logger
from torch import nn

from .unet_utils import unet


class Unet_regression(pl.LightningModule):
    def __init__(
        self,
        downconvs,
        upconvs,
        num_classes=1,
        learning_rate=1e-5,
        output_shape=(45, 55),
    ):
        super().__init__()
        self.save_hyperparameters()

        self.downconvs = downconvs
        self.upconvs = upconvs
        self.learning_rate = learning_rate
        self.output_shape = output_shape

        self.unet = unet.Unet(downconvs, upconvs)

        downconvs_abschecksum = (
            sum(torch.abs(p).sum() for p in downconvs.parameters())
            .detach()
            .cpu()
            .numpy()
        )

        upconvs_abschecksum = (
            sum(torch.abs(p).sum() for p in upconvs.parameters()).detach().cpu().numpy()
        )

        logger.info("---------------------------------")
        logger.info(f"downconvs abs checksum  {downconvs_abschecksum:.4f}")
        logger.info(f"upconvs abschecksum     {upconvs_abschecksum:.4f}")
        logger.info("---------------------------------")

        self.num_classes = num_classes

    def rmse_loss(self, y_hat, y):
        loss = torch.sqrt(torch.mean((y_hat - y) ** 2))
        return loss

    def forward(self, x):
        x = self.unet(x)
        # Output of unet should be 1 channel * 448 * 448
        # Downsample to self.output_shape
        pool = nn.AdaptiveAvgPool2d(self.output_shape)
        x = pool(x)

        return x

    def compute_loss(self, batch):
        # batch is a dict, get first key as x
        if isinstance(batch, dict):
            keys = list(batch.keys())
            x = torch.cat(
                [batch[k] for k in keys[:-1]], dim=1
            )  # All except the last key are inputs
            y = batch[keys[-1]]  # Last key is the labels
        else:
            x, y = batch
        y_hat = self(x)  # Forward pass

        y = y.long()  # Convert to long
        loss = self.rmse_loss(y_hat, y)  # Compute loss

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)

        # metric
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True
        )
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)

        # metric
        self.log(
            "test/loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True
        )
        self.log(
            "test_loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)

        # metric
        self.log(
            "val/loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True
        )
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def on_validation_epoch_end(self):
        # Print model summary at the end of each epoch
        # logger.info(self)

        num_frozen_params = sum(
            p.numel() for p in self.parameters() if not p.requires_grad
        )
        num_trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        logger.info(f"Number of frozen parameters: {num_frozen_params}")
        logger.info(f"Number of trainable parameters: {num_trainable_params}")
