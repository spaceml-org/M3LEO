from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch import nn
from loguru import logger
from fdlsar.models.mae_utils import vision_transformer as vt
from fdlsar.models.mae_utils import masked_autoencoder as mae
import numpy as np

class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)        
        
class SegmentationDecoder(pl.LightningModule):
    
    def __init__(self, image_size, num_channels, output_size, num_classes):
            super().__init__()
            self.save_hyperparameters()
            
            self.image_size = image_size
            self.num_channels = num_channels
            self.output_size = output_size
            self.num_classes = num_classes
            
            # Conv -> syncbn -> relu -> interpolate x2 until input resolution
            # in size (b, image_size, image_size, num_channels)
            # out size (b, output_size, output_size,)

            layers = []

            layers.extend(
                [
                    nn.Conv2d(self.num_channels, 256, kernel_size=3, stride=1, padding=1),
                    nn.SyncBatchNorm(256),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                ]
            )

            current_size = image_size * 2

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
                    nn.Conv2d(256, self.num_classes, kernel_size=3, stride=1, padding=1),
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
            
    def forward(self, x):
        return self.decoder(x)
        
class RegressionDecoder(pl.LightningModule):
    
    def __init__(self, decoder_type, 
                sequence_length, hidden_dim, 
                fc_layers = [512, 256, 128],
                output_dim=1, output_range = None):
        """
        squence_length, hidden_dim: like width and heigth
        
        expected input shape is [batch_size, sequence_length*hidden_dim]
        """
        super().__init__()
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        self.decoder_type = decoder_type
        self.sequence_length = sequence_length
        self.output_dim = 1
        self.fc_layers = fc_layers

        if output_range is not None:
            try:    
                output_range = list(output_range)
                if not isinstance(output_range, list):
                    r1, r2 = output_range
                    k = r1+r2 < 10. # chech numeric
            except Exception as e:
                raise ValueError(f"output_range must be a list of two numbers but got {output_range} of type {type(output_range)}, error is {e}")
            
        self.output_range = output_range

        # Encoder o/p shape #Shape = (batch_size, sequence_length, hidden_dim) [8, 196, 768][vit_b_32]
        valid_types = ["fc_linear", "conv"]
        if not self.decoder_type in valid_types:
            raise ValueError(f"Decoder must be one of {valid_types}")
        logger.info(f"Using {self.decoder_type} decoder...")

        if self.decoder_type == "fc_linear":
            if len(fc_layers)>0:
                layers = [
                           nn.Linear(self.sequence_length * self.hidden_dim, fc_layers[0]),
                           nn.ReLU()
                         ]
                for i in range(1, len(fc_layers)):
                    layers.append(nn.Linear(fc_layers[i-1], fc_layers[i]))
                    layers.append(nn.ReLU())   

                self.decoder = nn.Sequential(
                    nn.Flatten(),
                    *layers,
                    nn.Linear(fc_layers[-1], output_dim),
                    nn.ELU() if output_range is None else nn.Sigmoid(),
                )

            else:
                # if no layers, connect directly input to output
                self.decoder = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(self.sequence_length * self.hidden_dim, output_dim),
                    nn.ELU() if output_range is None else nn.Sigmoid(),
                )
        elif self.decoder_type == "conv":
            if not int(np.sqrt(self.sequence_length-1))**2==self.sequence_length-1:
                raise ValueError("sequence length must be a number such that sequence_length - 1, is squared")
                
            self.token_as_image_size = int(np.sqrt(self.sequence_length-1))

            conv_out_channels = self.hidden_dim // 2
            self.decoder = nn.Sequential(
                    Permute(0, 3, 1, 2),
                    nn.Conv2d(
                        in_channels=self.hidden_dim,
                        out_channels=self.hidden_dim//2,
                        kernel_size=4, stride=2, 
                        padding=1, padding_mode='replicate',
                    ),
                    nn.Conv2d(
                        in_channels=self.hidden_dim//2,
                        out_channels=self.hidden_dim//2,
                        kernel_size=4, stride=2, 
                        padding=1, padding_mode='replicate',
                    ),

                    nn.Flatten(),
                    nn.Linear( (self.hidden_dim//2)*(self.token_as_image_size//4)**2, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, self.output_dim),
                    nn.ELU() if self.output_range is None else nn.Sigmoid(),
                )
            
    def forward(self, x):
        if self.decoder_type=='conv':
            x = x[:,1:].reshape(len(x), self.token_as_image_size, self.token_as_image_size, -1)

        x = self.decoder(x)
    
        if self.output_range is not None:
            vmin, vmax = self.output_range
            x = (x-vmin)/(vmax-vmin)

        return x

    
class Encoder(pl.LightningModule):
    
    def __init__(self, 
                 image_size, 
                 num_channels, 
                 out_channels=16, 
                 vit_encoder='vit_t16',
                 output_mode='all_tokens'):
        super().__init__()
        self.save_hyperparameters()
        
        allowed_output_modes = ['class_token', 'all_tokens']
        if not output_mode in allowed_output_modes:
            raise ValueError(f"invalid output_mode '{output_mode}', only '{allowed_output_modes}' allowed")

        self.output_mode = output_mode
        self.image_size = image_size
        self.num_channels = num_channels
        self.out_channels = out_channels
        self.vit_encoder = vit_encoder

        self.out_dim = out_channels*785
        
        if self.vit_encoder == 'vit_t16':
            self.vit = vt.vit_t_sar_16(image_size=image_size, num_channels=num_channels)
            # in_channels = 192
        elif self.vit_encoder == 'vit_s16':
            self.vit = vt.vit_s_sar_16(image_size=image_size, num_channels=num_channels)        
            # in_channels = 384
        elif self.vit_encoder == 'vit_b16':
            self.vit = vt.vit_b_sar_16(image_size=image_size, num_channels=num_channels)
            # in_channels = 768
        elif self.vit_encoder == 'vit_l16':
            self.vit = vt.vit_l_sar_16(image_size=image_size, num_channels=num_channels)
            # in_channels = 1024
        elif self.vit_encoder == 'vit_h14':
            self.vit = vt.vit_h_sar_14(image_size=image_size, num_channels=num_channels)
            # in_channels = 1280
        else:
            raise ValueError(f"invalid vit encoder '{vit_encoder}', must be 'vit_t16', 'vit_s16' or 'vit_b16'")

        in_channels = self.vit.hidden_dim

        self.encoder = mae.MaskedAutoEncoderBackbone.from_vit(self.vit)
        # reduce the number of output channels
        self.reducer = nn.Sequential(
                Permute(0, 2, 1),
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                ),
                Permute(0, 2, 1),
            )
        
    def forward_get_classtoken(self, x):
        x = self.encoder(x)
        return x

    def forward_get_alltokens(self, x):
        x = self.encoder.encode(x)
        x = self.reducer(x)
        return x

    def forward(self, x):
        if self.output_mode == 'class_token':
            x = self.forward_get_classtoken(x)
        else:
            x = self.forward_get_alltokens(x)
        return x
        