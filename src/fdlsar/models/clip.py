from __future__ import annotations

import lightning.pytorch as pl
import torch
from torch import nn
import numpy as np
from loguru import logger
from progressbar import progressbar as pbar
from rlxutils import subplots
import matplotlib.pyplot as plt

from .components import cliploss

def unbatch_input(x):
    """
    strips out all channels in a batch i
    [batch_size, num_channels, h, w] -> [batch_size * num_channels, 1, h, w]
    use it before feeding data to SingleChannel models
    """
    x = x.reshape(-1,1,*x.shape[2:])
    return x

def rebatch_output(x, batch_size):
    """
    recover batches and flatten 
        [batch_size * num_channels, h, w]  -> [batch_size, num_channels, h*w]

    using after feeding data to SingleChannel models
    """
    x = x.reshape(batch_size, x.shape[0]//batch_size, *x.shape[1:])
    return x


class ClipModel(pl.LightningModule):
    """
    CLIP model with two input architectures ModifiedResNet

    encoders: a dict with the encoders to use. must have the same keys
              as the input data
    """

    def __init__(self, encoders, loss=None, learning_rate=5e-5, **kwargs):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.encoders = encoders
        if loss is None:
            self.loss_object = cliploss.ClipLoss()
        else:
            self.loss_object = loss

        # need this to allow torch find model parameters (since 'encoders' is a dict)
        self.encoders_params = nn.ParameterDict(encoders)
        
        # flatten
        self.flat = nn.Flatten()
        
    def forward(self, x):
        """
        assumes batches are dicts which each dataset
        """

        # use only items in batch for which there are encoders
        x = {k:v for k,v in x.items() if k in self.encoders.keys()}

        if set(x.keys()) != set(self.encoders.keys()):
            raise ValueError(
                f"data contains {list(x.keys())} but encoders expect {list(self.encoders.keys())}"
            )

        r = {
             k: self.flat(
                          self.encoders[k](x[k])
                         ) \
             for k in self.encoders.keys()
            }

        return r

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.optimizer = optimizer
        return optimizer

    """
    # binary compute loss
    def compute_loss(self, batch):
        # get features of both modalities
        x = self(batch)
        f0, f1 = [x[k] for k in self.encoders.keys()]
        
        # compute clip loss
        return self.loss_object(f0, f1)
    """

    def compute_loss(self, batch):
        x = self(batch)

        # compute loss between all pairs of inputs
        k = list(x.keys())
        kpairs = [ [k[i], k[j]] for j in range(len(k)) for i in range(j) ]
        val_losses = [self.loss_object(x[kpair[0]], x[kpair[1]]) for kpair in kpairs]
        total_loss = torch.stack(val_losses).mean()

        losses = {"-".join(kpair): val_loss for kpair, val_loss in zip(kpairs, val_losses)}

        return total_loss, losses        

    def training_step(self, batch, batch_idx):
        loss, losses = self.compute_loss(batch)
        # metric
        log_value = 0 if loss is None else loss
        self.log("train/loss", log_value, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        for k,v in losses.items():
            self.log(f"train/loss_{k}", v, on_step=False, on_epoch=True, logger=True, prog_bar=False)

        return loss

    def test_step(self, batch, batch_idx):
        loss, losses = self.compute_loss(batch)

        # metric
        log_value = 0 if loss is None else loss
        self.log("test/loss", log_value, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        for k,v in losses.items():
            self.log(f"test/loss_{k}", v, on_step=False, on_epoch=True, logger=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, losses = self.compute_loss(batch)

        # metric
        log_value = 0 if loss is None else loss
        self.log("val/loss", log_value, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        for k,v in losses.items():
            self.log(f"val/loss_{k}", v, on_step=False, on_epoch=True, logger=True, prog_bar=False)

        return loss
    
    
class ClipModelSingleChannel(ClipModel):
    
    """
    a model that assumes feeds each channel through en encoder
    
    """
    def __init__(self, encoders, learning_rate=1e-5, loss=None):
                
        super().__init__(encoders = encoders,
                         loss = loss,
                         learning_rate = learning_rate)

        self.save_hyperparameters()
        self.learning_rate = learning_rate
        if loss is None:
            self.loss_object = cliploss.ClipLossSingleChannel()
        else:
            self.loss_object = loss
        
        for k,encoder in self.encoders.items():
            if encoder.num_channels != 1:
                raise ValueError(f"encoders must have only 1 channel, 'but '{k}' has {encoder.num_channels} channels")

    def forward(self, batch):
        return self.forward_simple(batch)

    def forward_simple(self, batch):
        # reshape to make channels as items
        # [batch_size, n_channels, h, w] --> [batch_size*n_channels, 1, h, w]
        br = {k:v.reshape(-1,1,*v.shape[2:]) for k,v in batch.items()}

        # forward pass
        x = super().forward(br)

        # reshape to one encoding per channel
        # [batch_size*n_channels, encoding_size ] --> [batch_size, n_channels, encoding_size]
        x = {k:v.reshape(*batch[k].shape[:2],-1) for k,v in x.items()}     
        
        return x

    def forward_with_filter(self, batch):
        """
        this forward function removes constant channels from each image before feeding 
        them through the network.

        DONT USE IT: stalls after one epoch 
        """
        # decide what channels to keep on each img in each batch in each dataset
        keep = {k: [torch.argwhere(v[i].std(dim=[-2,-1])>1e-5).flatten() for i in range(len(v))] for k,v in batch.items()}

        # obtain a list of images per batch, each image keeping only non-constant channels
        # it must be a list because each image might endup with different number of channels
        bk = {k:[v[i,keep[k][i]] for i in range(len(v))] for k,v in batch.items()}

        # assemble all channels individually in a batch in each datset
        bv = {k:torch.vstack(v) for k,v in bk.items()}

        # add dimension as a single channel
        # [batch_size, n_channels] --> [batch_size*n_channels, 1, h, w]
        bv = {k:v[:,None,:,:] for k,v in bv.items()}

        # all images in a batch have all channels constant
        if sum([len(v)==0 for v in bv.values()])>0:
            logger.info('all images in batch have all constant channels')
            return None

        # forward pass
        bf = super(self.__class__, self).forward(bv)

        # split all embeddings in each image and batch
        keepi = {k: [0]+list(np.cumsum([len(i) for i in v])) for k,v in keep.items()}
        bg = {k: [bf[k][keepi[k][i]:keepi[k][i+1]] for i in range(len(keepi[k])-1)] for k,v in keepi.items()}

        # compute the mean of embeddings per image
        bm = {k: torch.vstack([i.mean(axis=0) for i in v]) for k,v in bg.items()}

        # remove images with all channels constant
        #bn = {k:v[torch.isnan(v).sum(axis=1)==0] for k,v in bm.items()}                                               
                
        return bn
    

    def compute_loss(self, batch):
        x = self(batch)
        # if we lost any image (because all of its channels were constant)
        # we discard the batch
        if x is None or np.std([v.shape[0] for k,v in x.items()])>1e-5:
            return None

        # compute loss between all pairs of inputs
        k = list(x.keys())
        kpairs = [ [k[i], k[j]] for j in range(len(k)) for i in range(j) ]
        val_losses = [self.loss_object(x[kpair[0]], x[kpair[1]]) for kpair in kpairs]
        total_loss = torch.stack(val_losses).mean()

        losses = {"-".join(kpair): val_loss for kpair, val_loss in zip(kpairs, val_losses)}

        return total_loss, losses  




class ClipEmbeddings:
    """
    class to compute embeddings, cross-simmilarities, plots, etc.
    """
    def __init__(self, dataloader, clipmodel, additional_data_funcs={}):
        self.dataloader = dataloader
        self.clipmodel = clipmodel

        self.model_keys = list(clipmodel.encoders.keys())
        self.kpairs = [ (self.model_keys[i], self.model_keys[j]) for j in range(len(self.model_keys)) for i in range(j) ]
        
        self.additional_data_funcs = additional_data_funcs
        
    def extract_embeddings(self):

        N = lambda x: x.detach().cpu().numpy()

        simm_diff_modalities_same_tile = {kpair: [] for kpair in self.kpairs}
        simm_channels_same_tile = {k: [] for k in self.model_keys}

        simm_diff_modalities_diff_tile = {kpair: [] for kpair in self.kpairs}
        simm_channels_diff_tile = {k: [] for k in self.model_keys}

        embeddings = {k: [] for k in self.model_keys}

        tile_ids = []

        self.additional_data = {k:[] for k in self.additional_data_funcs.keys()}


        for batch in pbar(self.dataloader, max_value=len(self.dataloader)):

            batch_cuda = {k:v.cuda() for k,v in batch.items() if k not in ['tile_id']}
            batch_size = batch[list(batch.keys())[0]].shape[0]    

            tile_ids += batch['tile_id']

            for k,f in self.additional_data_funcs.items():
                self.additional_data[k].append(N(f(batch)))

            with torch.no_grad():
                output = self.clipmodel(batch_cuda)
                for k in self.model_keys:
                    embeddings[k].append(N(output[k][:,0,:]))        
                    simm_channels_same_tile[k] += [(embeddings[k][-1][i]@embeddings[k][-1][i].T).mean() for i in range(batch_size)]
                    simm_channels_diff_tile[k] += [((embeddings[k][-1][i]@embeddings[k][-1][j].T)).mean() for i in range(batch_size) for j in range(i)]

            for kpair in self.kpairs:
                k0, k1 = kpair
                simm_diff_modalities_same_tile[kpair] += [(embeddings[k0][-1][i] @ embeddings[k1][-1][i].T).mean() for i in range(batch_size)]
                simm_diff_modalities_diff_tile[kpair] += [(embeddings[k0][-1][i] @ embeddings[k1][-1][j].T).mean()  for i in range(batch_size) for j in range(i)]

        self.embeddings = embeddings
        self.simm_diff_modalities_same_tile = simm_diff_modalities_same_tile
        self.simm_channels_same_tile = simm_channels_same_tile
        self.simm_diff_modalities_diff_tile = simm_diff_modalities_diff_tile
        self.simm_channels_diff_tile = simm_channels_diff_tile
        self.tile_ids = tile_ids
        
        for k,v in self.embeddings.items():
            t = np.r_[v[:-1]]
            self.embeddings[k] = t.reshape(-1, t.shape[-1])

        for k,v in self.additional_data.items():
            t = np.r_[v[:-1]]
            self.additional_data[k] = t.reshape(-1, t.shape[-1]).flatten()

    def plot_simmilarities(self, kpair):

        k0, k1 = kpair

        def pr(x, pa=10, pb = 90):
            pa,pb = np.percentile(x, [pa, pb])
            return np.r_[x][(x>pa)&(x<pb)]

        def false_positives(high_similarity, low_similarity, pct_high_similarity=10.):
            pa = np.percentile(np.r_[high_similarity], [pct_high_similarity])
            return np.mean(np.r_[low_similarity]>pa), pa    

        ppr = lambda x: pr(x, 1,99)
        for ax,i in subplots(3):
            if i==0:
                plt.hist(ppr(self.simm_diff_modalities_same_tile[kpair]), density=True, bins=100, alpha=.5, label="same tile", color="steelblue")
                plt.hist(ppr(self.simm_diff_modalities_diff_tile[kpair]), density=True, bins=100, alpha=.5, label="diff tile", color="orange");
                fp, th = false_positives(self.simm_diff_modalities_same_tile[kpair], self.simm_diff_modalities_diff_tile[kpair])
                plt.title(f"{k0} vs {k1}\nfalse positives at 10% {fp:.3f}")
                plt.axvline(th, color="steelblue", label="10% of same tile")
                plt.legend()
            if i==1:
                plt.hist(ppr(self.simm_channels_same_tile[k0]), bins=100, density=True, alpha=.5, label="same tile", color="steelblue")
                plt.hist(ppr(self.simm_channels_diff_tile[k0]), bins=100, density=True, alpha=.5, label="diff tile", color="orange");
                fp, th = false_positives(self.simm_channels_same_tile[k0], self.simm_channels_diff_tile[k0])
                plt.title(f"intra channel {k0}\nfalse positives at 10% {fp:.3f}")
                plt.axvline(th, color="steelblue", label="10% of same tile")            
            if i==2:
                plt.hist(ppr(self.simm_channels_same_tile[k1]), bins=100, density=True, alpha=.5, label="same tile", color="steelblue")
                plt.hist(ppr(self.simm_channels_diff_tile[k1]), bins=100, density=True, alpha=.5, label="diff tile", color="orange");
                fp, th = false_positives(self.simm_channels_same_tile[k1], self.simm_channels_diff_tile[k1])
                plt.title(f"intra channel {k1}\nfalse positives at 10% {fp:.3f}")
                plt.axvline(th, color="steelblue", label="10% of same tile")
                plt.legend()
            plt.grid();