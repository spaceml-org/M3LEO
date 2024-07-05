from __future__ import annotations

import math

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn
from torchmetrics import JaccardIndex

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


class MaskedAutoEncoder_Segmentation(pl.LightningModule):
    def __init__(
        self,
        backbone=None,
        vit=None,
        num_channels=1,
        decoder_type="deconv",
        output_size=448,
        num_classes=11,
    ):
        super().__init__()

        if backbone is None and vit is None:
            raise ValueError("Must provide either load_from or vit_arch")

        self.num_channels = num_channels  # Changed from 3 in the paper implementation

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
        self.output_size = output_size
        self.sequence_length = ((self.image_size // self.patch_size) ** 2) + 1
        self.num_classes = num_classes
        self.decoder_type = decoder_type
        self.init_decoder()
        self.criterion = F.cross_entropy
        self.jaccard = JaccardIndex(task="multiclass", num_classes=self.num_classes)

    def init_decoder(self):
        # Reshape + Encoder o/p shape #Shape = (batch_size, patch size, patch size, hidden_dim) [8, 196, 768][vit_b_32]
        valid_types = ["deconv"]

        if not self.decoder_type in valid_types:
            raise ValueError(f"Decoder must be one of {valid_types}")
        logger.info(f"Using {self.decoder_type} decoder...")

        if self.decoder_type == "deconv":
            # Conv -> syncbn -> relu -> interpolate x2 until input resolution

            num_patches_dim = self.image_size // self.patch_size
            if self.output_size / num_patches_dim <= 2:
                raise ValueError("Patches are too small to construct deconv decoder")
            # in size (b, num_patches, num_patches, h)
            # out size (b, self.output_size, self.output_size,)

            layers = []

            layers.extend(
                [
                    nn.Conv2d(self.hidden_dim, 256, kernel_size=3, stride=1, padding=1),
                    nn.SyncBatchNorm(256),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                ]
            )

            current_size = num_patches_dim * 2

            while current_size * 2 < self.output_size:  # While we can still scale *2
                layers.extend(
                    [
                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                        nn.SyncBatchNorm(256),
                        nn.ReLU(inplace=True),
                        nn.Upsample(
                            scale_factor=2, mode="bilinear", align_corners=False
                        ),
                    ]
                )

                current_size = current_size * 2

            # Final layer
            layers.extend(
                [
                    nn.Conv2d(
                        256, self.num_classes, kernel_size=3, stride=1, padding=1
                    ),
                    nn.SyncBatchNorm(self.num_classes),
                    nn.ReLU(inplace=True),
                    nn.Upsample(
                        size=(self.output_size, self.output_size),
                        mode="bilinear",
                        align_corners=False,
                    ),
                ]
            )

            self.decoder = nn.Sequential(*layers)

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

    def _decoder_reshape(self, x_encoded):
        assert x_encoded.dim() == 3
        n, hw, c = x_encoded.shape
        h = w = int(math.sqrt(hw))
        x_encoded = x_encoded.transpose(1, 2).reshape(n, c, h, w)
        return x_encoded

    def forward_decoder(self, x_encoded):
        x_encoded = x_encoded[:, 1:]  # Remove class token
        x_encoded = self._decoder_reshape(x_encoded)
        x = self.decoder(x_encoded)
        return x

    def compute_loss(self, batch, batch_idx):
        images = list(batch.values())[0]
        targets = list(batch.values())[1]
        targets = targets.long()

        x_encoded = self.forward_encoder(images)

        y_pred = self.forward_decoder(x_encoded)
        loss = self.criterion(y_pred, targets)

        return loss

    def compute_miou(self, batch, batch_idx):
        images = list(batch.values())[0]
        targets = list(batch.values())[1]
        targets = targets.long()

        x_encoded = self.forward_encoder(images)

        y_pred = self.forward_decoder(x_encoded).argmax(dim=1)

        mIOU = self.jaccard(y_pred, targets)

        return mIOU

    def forward(self, x):
        x_encoded = self.forward_encoder(x)
        y_pred = self.forward_decoder(x_encoded).argmax(dim=1)

        return y_pred

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, batch_idx)

        # calculating mIOU without gradients
        with torch.no_grad():
            mIOU = self.compute_miou(batch, batch_idx)

        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        # Log mIOU as well
        self.log(
            "train/mIOU", mIOU, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):  # TODO
        loss = self.compute_loss(batch, batch_idx)

        # calculating mIOU without gradients
        with torch.no_grad():
            mIOU = self.compute_miou(batch, batch_idx)

        self.log(
            "val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        # Log mIOU as well
        self.log(
            "val/mIOU", mIOU, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def test_step(self, batch, batch_idx):  # TODO
        loss = self.compute_loss(batch, batch_idx)

        # calculating mIOU without gradients
        with torch.no_grad():
            mIOU = self.compute_miou(batch, batch_idx)

        self.log(
            "test/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        # Log mIOU as well
        self.log(
            "test/mIOU", mIOU, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=1.5e-4)
        return optim
