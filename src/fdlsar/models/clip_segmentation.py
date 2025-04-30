from __future__ import annotations

#import pytorch_lightning as pl
import lightning.pytorch as pl

import torch
from torch import nn
from torch.nn import functional as F
from .components import simplecnn
from loguru import logger
import math

def safe_reshape(x):
    """
    reshapes [batch_size, flattened_squared_image_size, n_channels]
    into     [batch_size, h, w, n_channels]
    """
    n, hw, c = x.shape
    h = w = int(math.sqrt(hw))
    x = x.transpose(1,2).reshape(n, c, h, w)
    return x

class ClipSegmentationModel(pl.LightningModule):
    def __init__(self, backbone, decoder, learning_rate=5e-3):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.backbone = backbone
        self.decoder = decoder

        backbone_abschecksum = sum([torch.abs(p).sum() for p in self.backbone.parameters()]).detach().cpu().numpy()
        decoder_abschecksum  = sum([torch.abs(p).sum() for p in self.decoder.parameters()]).detach().cpu().numpy()
        
        logger.info("---------------------------------")
        logger.info(f"backbone abs checksum  {backbone_abschecksum:.4f}")
        logger.info(f"decoder abs checksum   {decoder_abschecksum:.4f}")
        logger.info("---------------------------------")
                
    def forward(self, x):
        x = self.backbone(x)[:,1:,:] # ignore class token
        x = safe_reshape(x)
        x = self.decoder(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def compute_loss(self, batch, batch_idx):
        images = list(batch.values())[0]
        targets = list(batch.values())[1]
        targets = targets.long()

        y_pred = self(images)
        loss = F.cross_entropy(y_pred, targets)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)

        # metric
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)

        # metric
        self.log(
            "test/loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)

        # metric
        self.log(
            "val/loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True
        )
        return loss
