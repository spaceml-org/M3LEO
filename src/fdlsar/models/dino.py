from __future__ import annotations

import copy

import lightning.pytorch as pl
import numpy as np
import torch
import wandb
from loguru import logger

from fdlsar.models.dino_components.cosine_schedule import cosine_schedule
from fdlsar.models.dino_components.deactivate_requires_grad import (
    deactivate_requires_grad,
)
from fdlsar.models.dino_components.dino_loss import DINOLoss
from fdlsar.models.dino_components.dino_projection_head import DINOProjectionHead
from fdlsar.models.dino_components.update_momentum import update_momentum
from fdlsar.models.dino_components.vision_transformer import (
    vit_base,
    vit_small,
    vit_tiny,
)
from fdlsar.models.dino_components.vision_transformer_v2 import (
    vit_base_v2,
    vit_small_v2,
    vit_tiny_v2,
)
from fdlsar.models.dino_components.visualize_attention import (
    plot_attention_map_histograms,
    plot_segmentation_images,
    project_images_into_attention_head,
)

vit_arch_mapping = {
    "vit_tiny": vit_tiny,
    "vit_small": vit_small,
    "vit_base": vit_base,
}
vit_arch_mapping_v2 = {
    "vit_tiny": vit_tiny_v2,
    "vit_small": vit_small_v2,
    "vit_base": vit_base_v2,
}


class DINOModel(pl.LightningModule):
    """
    DINO's Model implementation.

    This model might be used with a vision transformer ViT or a ResNet architecture.

    Original paper: https://arxiv.org/abs/2104.14294
    """

    def __init__(
        self,
        vit_arch=None,
        patch_size=16,
        input_dim=448,
        img_size=[12, 448, 448],
        hidden_dim=256,
        output_dim=1024,
        bottleneck_dim=64,
        cosine_schedule__max_steps: float = 10,
        cosine_schedule__start_value: float = 0.996,
        cosine_schedule__end_value: float = 1,
        learning_rate=0.001,
        num_img_log=6,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.04,
        warmup_teacher_temp_epochs: int = 5,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        use_mae_encoder: bool = False,
    ):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()

        # torchvision.models.resnet18()
        self.num_img_log = num_img_log  # number of images to log in wandb
        self.patch_size = patch_size
        self.img_size = list(img_size)
        self.in_channels = self.img_size[0]
        self.vit_arch = vit_arch

        if self.vit_arch is None:
            raise ValueError(
                "DINO vit architecture needed; options are vit_tiny, vit_small, vit_base."
            )

        logger.info(f"Using vit arch {self.vit_arch}")

        if use_mae_encoder:
            logger.info("Instantiating model with MAE encoder blocks.")
            backbone = vit_arch_mapping_v2[self.vit_arch](
                img_size=self.img_size,
                patch_size=self.patch_size,
                in_chans=self.in_channels,
            )
        else:
            logger.info("Instantiating model with simple encoder blocks.")
            backbone = vit_arch_mapping[self.vit_arch](
                img_size=self.img_size,
                patch_size=self.patch_size,
                in_chans=self.in_channels,
            )

        input_dim = backbone.embed_dim

        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            output_dim=output_dim,
            freeze_last_layer=1,
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(
            input_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            output_dim=output_dim,
        )
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.cosine_schedule__max_steps = cosine_schedule__max_steps
        self.cosine_schedule__start_value = cosine_schedule__start_value
        self.cosine_schedule__end_value = cosine_schedule__end_value

        self.criterion = DINOLoss(
            output_dim=output_dim,
            warmup_teacher_temp=warmup_teacher_temp,
            teacher_temp=teacher_temp,
            warmup_teacher_temp_epochs=warmup_teacher_temp_epochs,
            student_temp=student_temp,
            center_momentum=center_momentum,
        )
        self.lr = learning_rate

    def forward(self, x):
        """ "
        The student's backbone could be a resnet or a visual transformer.
        Smaller crop images are passed to the student network compared to
        the ones used to train the teacher.
        """
        # x = x.reshape([-1] + self.img_size)
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        """ "
        The teacher's backbone could be a resnet or a visual transformer.
        """
        # x = x.reshape([-1] + self.img_size)
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def compute_loss(self, batch):
        # get features of both modalities
        views = self.get_dataset(batch)  # check if only one dataset is passed

        momentum = cosine_schedule(
            self.current_epoch,
            self.cosine_schedule__max_steps,
            self.cosine_schedule__start_value,
            self.cosine_schedule__end_value,
        )
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)

        views = [view.to(self.device) for view in views]
        global_views = views[:2]

        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        # compute dino loss
        return self.criterion(teacher_out, student_out, epoch=self.current_epoch)

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True
        )

        epoch_countfrom_one = self.current_epoch + 1
        epoch_logger = epoch_countfrom_one % 25
        self.log("epoch_logger", epoch_logger, on_step=True, on_epoch=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)

        # metric
        self.log("test/loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)

        # metric
        self.log("val/loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optim

    def validation_step(self, batch, batch_idx):
        if not hasattr(self, "val_batch"):
            self.val_batch = self.get_dataset(batch)

    def get_dataset(self, batch):
        """
        assumes batches are dicts which each dataset
        check that only one dataset is passed
        """

        if len(batch.keys()) > 1:
            raise ValueError(
                f"data contains {list(batch.keys())} but DINO expects only one dataset"
            )

        return batch[self.get_key_from_batch(batch)]

    def get_key_from_batch(self, batch):
        """
        assumes batches are dicts which each dataset
        get key of the only dataset available for dino
        """

        if not hasattr(self, "dataset_key"):
            self.dataset_key = next(iter(batch))

        return self.dataset_key

    def on_validation_epoch_end(self):
        views = self.val_batch
        views = [view.to(self.device) for view in views]
        global_views = views[:2]

        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        self.log(
            "val/loss", loss, on_step=False, on_epoch=True, logger=True, prog_bar=True
        )

        mean_teacher_out = np.mean(
            [teacher_out[x].cpu().mean() for x in range(len(teacher_out))]
        )
        std_teacher_out = np.mean(
            [teacher_out[x].cpu().std() for x in range(len(teacher_out))]
        )
        mean_student_out = np.mean(
            [student_out[x].cpu().mean() for x in range(len(student_out))]
        )
        std_student_out = np.mean(
            [student_out[x].cpu().std() for x in range(len(student_out))]
        )

        self.log(
            "mean-teacher-embedding",
            mean_teacher_out,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.log(
            "std-teacher-embedding",
            std_teacher_out,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.log(
            "mean-student-embedding",
            mean_student_out,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.log(
            "std-student-embedding",
            std_student_out,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        visualization_batch = global_views[0]
        number_of_heads, attention_maps = project_images_into_attention_head(
            self, visualization_batch, self.patch_size
        )

        fig = plot_segmentation_images(
            number_of_heads, attention_maps, visualization_batch, self.num_img_log
        )

        fig_hist = plot_attention_map_histograms(
            number_of_heads, attention_maps, visualization_batch, self.num_img_log
        )

        experiment = self.logger.experiment

        experiment.log(
            {
                "Input and attention maps": fig,
                "Attention map histograms": wandb.Image(fig_hist),
            },
        )
