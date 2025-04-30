"""
taken from https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/loss.py
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


softmax = lambda x: torch.exp(x) / torch.exp(x).sum(axis=1).reshape(-1,1)        

class BarlowTwinsLossSingleChannel(nn.Module):

    """
    loss according to https://arxiv.org/abs/2103.03230
    but adapted to embeddings per channels
    """
    
    def __init__(self, lmbda=1.):
        super().__init__()
        self.lmbda = lmbda
    
    def forward(self, m0_features, m1_features):
        # just short names
        m0 = m0_features
        m1 = m1_features
        
        #softmax = lambda x: torch.exp(x) / torch.exp(x).sum(axis=1).reshape(-1,1)        

        if m0.shape[0] != m1.shape[0]:
            raise ValueError(f"different number of batches for the two modalities. got shapes m0: {m0_features.shape}, m1: {m1_features.shape}")

        batch_size = m0.shape[0]
        num_group_elems = m0.shape[1] * m1.shape[1]

        nc0 = m0.shape[1] # num channels
        nc1 = m1.shape[1] # num_channels
        m0r = m0.reshape(-1, m0.shape[-1])
        m1r = m1.reshape(-1, m1.shape[-1])

        #m0r = softmax(m0r)
        #m1r = softmax(m1r)

        # normalize each vector item to mean 0 and std 1 across the batch
        m0rn = (m0r-m0r.mean(axis=0).reshape(1,-1))/(m0r.std(axis=0).reshape(1,-1))
        m1rn = (m1r-m1r.mean(axis=0).reshape(1,-1))/(m1r.std(axis=0).reshape(1,-1))

        # build correlation matrix
        dd = m0rn.mm(m1rn.T) 
        dd = dd / torch.prod(torch.tensor(dd.shape))

        # build ground truth 
        gt = torch.zeros_like(dd)
        for c0 in range(m0.shape[0]):
            for c1 in range(m1.shape[0]):
                if c1==c0:
                    gt[c0*nc0:(c0+1)*nc0, c0*nc1:(c1+1)*nc1]=1   


        # koff is the number of elements on off-diagonal groups between modaliteis
        koff = num_group_elems * (batch_size**2 - batch_size)

        # kdiag is the number of elements on diagonal groups between modaliteis
        kdiag = torch.prod(torch.tensor(dd.shape))-koff

        # invariance term, (1-gt*dd) will have ones in the off diagonal groups
        # so we substract them (-koff), and normalize by the number of elements used
        loss_invariance =( ((1-gt*dd)**2).sum() - koff )/kdiag

        # redundancy reduction term (1-gt) sets to zero all diagonal groups, so we are 
        # left with the off-diagonal groups, and normalize by the number of elments used
        loss_redundacy_reduction_term = (((1-gt)*dd)**2).sum() / koff

        # combine losses
        loss = loss_invariance - 1 * loss_redundacy_reduction_term
        
        return loss

    
class BarlowTwinsLoss(nn.Module):

    """
    loss according to https://arxiv.org/abs/2103.03230
    but adapted to embeddings per channels
    """
    
    def __init__(self, lmbda=1., mode='batch'):
        super().__init__()
        
        if not mode in ['batch', 'embedding']:
            raise ValueError(f"mode must be 'batch' or 'embedding' but found '{mode}'")
        self.lmbda = lmbda
        self.mode = mode
    
    def forward(self, m0_features, m1_features):
        # just short names
        m0 = m0_features
        m1 = m1_features
        
        #softmax = lambda x: torch.exp(x) / torch.exp(x).sum(axis=1).reshape(-1,1)        

        if m0.shape[0] != m1.shape[0]:
            raise ValueError(f"different number of batches for the two modalities. got shapes m0: {m0_features.shape}, m1: {m1_features.shape}")

        batch_size = m0.shape[0]

        N = batch_size
        emb_size = m0.shape[-1]

        # take the mean across embeddings of all channels
        if len(m0.shape)==3:
            m0r = m0.mean(axis=1)
            m1r = m1.mean(axis=1)
        else:
            m0r, m1r = m0, m1

        m0rn = (m0r-m0r.mean(axis=0).reshape(1,-1))/(m0r.std(axis=0).reshape(1,-1))
        m1rn = (m1r-m1r.mean(axis=0).reshape(1,-1))/(m1r.std(axis=0).reshape(1,-1))

        if self.mode=='batch':
            dd = m0rn.mm(m1rn.T) / emb_size

            gt = torch.eye(N).to(dd.device)

            koff = N**2 - N
            kdiag = N
        elif self.mode=='embedding':
            dd = m0rn.T.mm(m1rn) / N

            gt = torch.eye(emb_size).to(dd.device)

            koff = emb_size**2 - emb_size
            kdiag = emb_size
            

        loss_invariance = ((1-dd.diag())**2).sum() / kdiag
        loss_redundacy_reduction_term = (((1-gt)*dd)**2).sum() / koff

        # combine losses
        loss = loss_invariance + self.lmbda * loss_redundacy_reduction_term
        
        return loss
    
    
class ClipLossSingleChannel(nn.Module):
    
    def __init__(self,logit_scale = 1.):
        super().__init__()
        self.logit_scale = logit_scale
    
    def cross_entropy(self, m0_features, m1_features):
        # m0_features: shape [batch_size, num_channels_0, embedding_size]
        # m1_features: shape [batch_size, num_channels_1, embedding_size]
        
        # shorthands
        m0 = m0_features
        m1 = m1_features
        
        nc0 = m0.shape[1] # num channels
        nc1 = m1.shape[1] # num_channels
        
        # reshape to [batch_size*num_channels, embedding_size]
        m0r = m0.reshape(-1, m0.shape[-1])
        m1r = m1.reshape(-1, m1.shape[-1])
                
        # normalize each vector item to mean 0 and std 1 across the batch
        m0r = (m0r-m0r.mean(axis=0).reshape(1,-1))/(m0r.std(axis=0).reshape(1,-1))
        m1r = (m1r-m1r.mean(axis=0).reshape(1,-1))/(m1r.std(axis=0).reshape(1,-1))
            
        # dot products all vs all
        dd = m0r.mm(m1r.T) 

        # build ground truth for cross entropy
        gt = torch.zeros_like(dd)
        for c0 in range(m0.shape[0]):
            for c1 in range(m1.shape[0]):
                if c1==c0:
                    gt[c0*nc0:(c0+1)*nc0, c0*nc1:(c1+1)*nc1]=1        
        

        # scale and softmax
        dd = self.logit_scale * dd/dd.max()

        dd, gt = softmax(dd), softmax(gt)    
        
        return F.cross_entropy(dd, gt)
        

    def forward(self, m0_features, m1_features):
        
        # m0_features: shape [batch_size, num_channels_0, embedding_size]
        # m1_features: shape [batch_size, num_channels_1, embedding_size]
    
        loss1 = self.cross_entropy(m0_features, m1_features)
        loss2 = self.cross_entropy(m1_features, m0_features)
        
        loss = (loss1 + loss2) / 2

        return loss


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            logit_scale = 1.
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.logit_scale = logit_scale

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = self.logit_scale * image_features @ all_text_features.T
                logits_per_text = self.logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = self.logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = self.logit_scale * image_features @ text_features.T
            logits_per_text = self.logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, output_dict=False):
        
        # if single channel embeddings just compute the mean embedding of all channels
        if len(image_features.shape)==3:
            image_features = image_features.mean(axis=1)
        if len(text_features.shape)==3:
            text_features = text_features.mean(axis=1)
        
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss

ClipLossFullChannel = ClipLoss
BarlowTwinsLossFullChannel = BarlowTwinsLoss


class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        
        clip_loss = 0
        
        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss