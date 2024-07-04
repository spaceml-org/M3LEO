from __future__ import annotations

import lightning.pytorch as pl
import torch
import wandb
from loguru import logger
from torch import nn
from torchvision.transforms import RandomResizedCrop

from .mae_utils import masked_autoencoder, utils, vision_transformer

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


class MaskedAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        mask_ratio,
        vit_arch,
        log_image_samples=8,
        num_channels=12,
        augment=True,
        image_size=448,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_channels = num_channels  # Changed from 3 in the paper implementation
        self.log_image_samples = log_image_samples
        self.image_size = image_size

        self.vit_arch_name = vit_arch

        decoder_dim = 512
        vit = vit_arch_mapping[vit_arch](
            num_channels=self.num_channels, image_size=self.image_size
        )

        logger.info(f"Using vit arch {self.vit_arch_name}...")

        self.mask_ratio = mask_ratio
        self.patch_size = vit.patch_size
        self.sequence_length = vit.seq_length  # 50
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.backbone = masked_autoencoder.MaskedAutoEncoderBackbone.from_vit(vit)
        self.decoder = masked_autoencoder.MaskedAutoEncoderDecoder(
            seq_length=vit.seq_length,
            num_layers=1,
            num_heads=16,
            embed_input_dim=vit.hidden_dim,
            hidden_dim=decoder_dim,
            mlp_dim=decoder_dim * 4,
            out_dim=vit.patch_size**2
            * self.num_channels,  # Added from the paper implementation
            dropout=0,
            attention_dropout=0,
        )
        self.criterion = nn.MSELoss()

        self.representation_dim = vit.seq_length * vit.hidden_dim

        self.augment = augment
        if self.augment:
            self.RandomResizedCrop = RandomResizedCrop(
                size=(self.image_size, self.image_size),
                antialias=True,
            )

    def on_fit_start(self):
        """
        Called by lightning at the beginning of training. This is the point at which the logger is available, so
        we can now log parameters to wandb.
        """
        self.logger.experiment.log({"model/vit_arch_name": self.vit_arch_name})
        self.logger.experiment.log({"model/mask_ratio": self.mask_ratio})
        super().on_fit_start()

    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(
            images, idx_keep
        )  # Note - this ONLY calls ENCODE, not FORWARD

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(
            self.mask_token, (batch_size, self.sequence_length)
        )
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def get_loss(self, batch, batch_idx, train=False):
        images = list(batch.values())[0]

        if self.augment and train:
            images = self.RandomResizedCrop(images)

        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        # print('Idx Keep: ', idx_keep.shape) # Shape = (batch_size, (1-mask_ratio)*sequence_length) [8, 12]
        # print('Idx Mask: ', idx_mask.shape) # Shape = (batch_size, (mask_ratio)*sequence_length) [8, 38]
        x_encoded = self.forward_encoder(
            images, idx_keep
        )  # Shape = (batch_size, (1-mask_ratio)*sequence_length, hidden_dim) [8, 12, 768]

        # Returns the prediction of the masked patches only
        x_pred = self.forward_decoder(
            x_encoded, idx_keep, idx_mask
        )  # Shape = (batch_size, mask_ratio*sequence_length, out_dim) [8, 38, 1024]

        # get image patches for masked tokens
        patches = utils.patchify(
            images, self.patch_size
        )  # Shape = (batch_size, 448/32*448/32, 32*32) [8, 196, 1024]
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(
            patches, idx_mask - 1
        )  # Shape = (batch_size, (mask_ratio)*sequence_length, 32*32) [8, 38, 1024]

        loss = self.criterion(x_pred, target)

        epoch_countfrom_one = self.current_epoch + 1
        epoch_countfrom_one % 25

        return loss

    def get_representation(self, images):
        x_encoded = self.forward_encoder(images).flatten(
            start_dim=1
        )  # (batch size, seq_length, hidden_dim)
        return x_encoded

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx, train=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True, logger=True)
        # self.log("epoch_logger", epoch_logger, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.val_batch = batch  # For image logging
        loss = self.get_loss(batch, batch_idx)
        self.log("val/loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        images = list(self.val_batch.values())[0]
        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        # Total number of patches (448/32)*(448/32) = 14*14 = 196 or 7*7=49 for 224x224
        x_encoded = self.forward_encoder(
            images, idx_keep
        )  # Shape (batch_size, 12, 768) #Encoded representation of unmasked tokens

        x_pred_patches = self.forward_decoder(
            x_encoded, idx_keep, idx_mask
        )  # Shape (batch_size, 38, 1024) #Predicted pixel values for masked tokens
        x_pred_mask_patches = torch.zeros(x_pred_patches.shape).to(images.device)

        patches = utils.patchify(
            images, self.patch_size
        )  # Shape = (batch_size, 224/32*224/32, 32*32) [8, 49, 1024]

        patches_with_pred = utils.set_at_index(
            patches, idx_mask - 1, x_pred_patches
        )  # idx_mask - 1 for cls token # Shape = (batch_size, 49, 1024)
        pred_image = utils.unpatchify(
            patches_with_pred, self.patch_size
        )  # Shape = (batch_size, 224, 224, 3)

        patches_with_zeros = utils.set_at_index(
            patches, idx_mask - 1, x_pred_mask_patches
        )  # idx_mask - 1 for cls token # Shape = (batch_size, 49, 1024)
        masked_image = utils.unpatchify(patches_with_zeros, self.patch_size)

        experiment = self.logger.experiment

        experiment.log(
            {
                "y_masked": [
                    wandb.Image(x.float(), caption="y_masked")
                    for x in masked_image[: self.log_image_samples, 0, :, :]
                ],
                "y_preds": [
                    wandb.Image(x.float(), caption="y_pred")
                    for x in pred_image[: self.log_image_samples, 0, :, :]
                ],
                "y_true": [
                    wandb.Image(x.float(), caption="y_true")
                    for x in images[: self.log_image_samples, 0, :, :]
                ],
            },
        )

    def test_step(self, batch, batch_idx):
        pass
        # loss = self.get_loss(batch, batch_idx)
        # self.log("test/loss", loss, on_step=True, on_epoch=True, logger=True)

        # return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=1.5e-4)
        return optim
