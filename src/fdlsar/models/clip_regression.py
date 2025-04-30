from __future__ import annotations

import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn import functional as F
from .components import simplecnn
from loguru import logger


class ClipRegressionModel(pl.LightningModule):
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
        
        
        self.flat = nn.Flatten()
        
    def forward(self, x):
        """produce final model output"""
        x = self.backbone(x)
        x = self.flat(x)
        # a single output for regression
        x = self.decoder(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def compute_loss(self, batch):
        keys = list(batch.keys())
        x = batch[keys[0]]
        y = batch[keys[1]]

        prediction = self(x)
        
        # if we have 2d images, we want to predict their the mean value
        if len(y.shape) >= 3:
            y_target = (
                y.reshape(len(y), -1).type(torch.float32).mean(axis=1).reshape(len(y), 1)
            )
            
        # if we have and 2d batch, must be just a column of values to predict    
        elif len(y.shape)==2:
            if y.shape[1]!=1:
                raise ValueError(f"expected a batch shape [batch_size, 1] but got {y.shape}")
            y_target = y

        # len(y.shape)==1 , just reshape the value
        else: 
            y_target = y.reshape(-1,1)
            
        loss = torch.sqrt(torch.mean((prediction - y_target) ** 2))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)

        # metric
        self.log(
            "train/loss", loss, on_step=False, on_epoch=True, logger=True, prog_bar=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)

        # metric
        self.log(
            "test/loss", loss, on_step=False, on_epoch=True, logger=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)

        # metric
        self.log(
            "val/loss", loss, on_step=False, on_epoch=True, logger=True, prog_bar=True
        )
        return loss


class ClipRegressionSingleChannelModel(ClipRegressionModel):


    def forward(self, x):
        
        x_orig = x
        # reshape to make channels as items
        # [batch_size, n_channels, h, w] --> [batch_size*n_channels, 1, h, w]
        x = x.reshape(-1,1,*x.shape[2:])
        
        # forward pass
        x = super().forward(x)

        # reshape to one encoding per channel
        # [batch_size*n_channels, 1 ] --> [batch_size, n_channels]
        # this is one prediction per channel per item in the batch
        x = x.reshape(*x_orig.shape[:2]) 
        
        
        # return the mean prediction of all channels in each item
        x = x.mean(axis=-1)
        
        return x
