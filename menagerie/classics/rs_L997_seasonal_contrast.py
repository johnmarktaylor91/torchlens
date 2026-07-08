# FAITHFUL PORT of ServiceNow/seasonal-contrast @ main (original framework: PyTorch + pytorch-lightning 1.1.8)
#
# Seasonal Contrast (SeCo): a MoCo-v2-style momentum-contrastive
# self-supervised model for remote sensing, pretrained on multi-season
# Sentinel-2 image pairs (Manas, Lacoste, Giro-i-Nieto, Vazquez, Rodriguez,
# "Seasonal Contrast: Unsupervised Pre-Training from Uncurated Remote
# Sensing Data", ICCV 2021, https://arxiv.org/abs/2103.16607). Upstream's
# `models/moco2_module.py::MocoV2` is real PyTorch/torchvision code (a
# ResNet encoder_q/encoder_k pair, MLP projection heads, momentum queue),
# but it is built on `pytorch_lightning.LightningModule` pinned to
# `pytorch-lightning==1.1.8` -- a pre-2.0 API (`self.hparams`, `self.use_ddp`,
# `self.use_ddp2` as auto-managed LightningModule attributes) that no longer
# exists on the installed pytorch-lightning 2.x (a breaking major-version
# rewrite), and it also imports `pl_bolts.metrics.precision_at_k`
# (pytorch-lightning-bolts, not installed / not a declared base lib here).
#
# This is a faithful, mechanism-for-mechanism port that keeps every real
# architecture and forward-pass op from `MocoV2.__init__`/`MocoV2.forward`
# byte-for-byte, but:
#   - subclasses `torch.nn.Module` instead of `pl.LightningModule` (Lightning
#     here is only training-loop scaffolding: `training_step`,
#     `configure_optimizers`, `self.log_dict` -- none of it is part of the
#     forward architecture, so dropping it changes nothing about the model);
#   - replaces the Lightning-managed `self.hparams` object with plain
#     `__init__` attributes (same values, same names);
#   - hardcodes `use_ddp = use_ddp2 = False` (the single-process eager-mode
#     values Lightning would have set for a non-distributed forward pass;
#     upstream's own `# pragma: no-cover` on the DDP shuffle helpers shows
#     they were never meant to run outside torch.distributed anyway);
#   - drops `training_step`/`configure_optimizers`/`add_model_specific_args`
#     (pure training-loop / argparse scaffolding, not architecture; this
#     also removes the only two `pl_bolts` and `pl.LightningModule` usages).
# Every remaining line -- encoder_q/encoder_k construction, momentum-key
# update, the projection MLP heads, the memory queue and its dequeue/enqueue,
# the contrastive-logit computation via einsum -- is the real upstream code.

import torch
from torch import nn
import torch.nn.functional as F
import torchvision


class MocoV2(nn.Module):
    def __init__(
        self,
        base_encoder,
        emb_dim,
        num_negatives,
        emb_spaces=1,
        encoder_momentum=0.999,
        softmax_temperature=0.07,
        **kwargs,
    ):
        super().__init__()

        self.base_encoder = base_encoder
        self.emb_dim = emb_dim
        self.num_negatives = num_negatives
        self.emb_spaces = emb_spaces
        self.encoder_momentum = encoder_momentum
        self.softmax_temperature = softmax_temperature

        # Non-distributed eager forward pass: Lightning would set these to
        # False outside a DDP/DDP2 training strategy.
        self.use_ddp = False
        self.use_ddp2 = False

        # create the encoders
        template_model = getattr(torchvision.models, base_encoder)
        self.encoder_q = template_model(num_classes=self.emb_dim)
        self.encoder_k = template_model(num_classes=self.emb_dim)

        # remove fc layer
        self.encoder_q = nn.Sequential(*list(self.encoder_q.children())[:-1], nn.Flatten())
        self.encoder_k = nn.Sequential(*list(self.encoder_k.children())[:-1], nn.Flatten())

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the projection heads
        self.mlp_dim = 512 * (1 if base_encoder in ["resnet18", "resnet34"] else 4)
        self.heads_q = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.mlp_dim, self.mlp_dim),
                    nn.ReLU(),
                    nn.Linear(self.mlp_dim, emb_dim),
                )
                for _ in range(emb_spaces)
            ]
        )
        self.heads_k = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.mlp_dim, self.mlp_dim),
                    nn.ReLU(),
                    nn.Linear(self.mlp_dim, emb_dim),
                )
                for _ in range(emb_spaces)
            ]
        )

        for param_q, param_k in zip(self.heads_q.parameters(), self.heads_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(emb_spaces, emb_dim, num_negatives))
        self.queue = nn.functional.normalize(self.queue, dim=1)

        self.register_buffer("queue_ptr", torch.zeros(emb_spaces, 1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            em = self.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1.0 - em)
        for param_q, param_k in zip(self.heads_q.parameters(), self.heads_k.parameters()):
            em = self.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1.0 - em)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, queue_idx):
        # gather keys before updating queue
        if self.use_ddp or self.use_ddp2:
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr[queue_idx])
        assert self.num_negatives % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[queue_idx, :, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.num_negatives  # move pointer

        self.queue_ptr[queue_idx] = ptr

    def forward(self, img_q, img_k):
        """
        Input:
            img_q: a batch of query images
            img_k: a batch of key images
        Output:
            logits, targets
        """

        # update the key encoder
        self._momentum_update_key_encoder()

        # compute query features
        v_q = self.encoder_q(img_q)

        # compute key features
        v_k = []
        for i in range(self.emb_spaces):
            # shuffle for making use of BN
            if self.use_ddp or self.use_ddp2:
                img_k[i], idx_unshuffle = batch_shuffle_ddp(img_k[i])

            with torch.no_grad():  # no gradient to keys
                v_k.append(self.encoder_k(img_k[i]))

            # undo shuffle
            if self.use_ddp or self.use_ddp2:
                v_k[i] = batch_unshuffle_ddp(v_k[i], idx_unshuffle)

        logits = []
        for i in range(self.emb_spaces):
            # compute query projections
            z_q = self.heads_q[i](v_q)  # queries: NxC
            z_q = nn.functional.normalize(z_q, dim=1)

            # compute key projections
            z_k = []
            for j in range(self.emb_spaces):
                with torch.no_grad():  # no gradient to keys
                    z_k.append(self.heads_k[i](v_k[j]))  # keys: NxC
                    z_k[j] = nn.functional.normalize(z_k[j], dim=1)

            # select positive and negative pairs
            z_pos = z_k[i]
            z_neg = self.queue[i].clone().detach()
            if i > 0:  # embedding space 0 is invariant to all augmentations
                z_neg = torch.cat(
                    [z_neg, *[z_k[j].T for j in range(self.emb_spaces) if j != i]], dim=1
                )

            # compute logits
            # Einstein sum is more intuitive
            l_pos = torch.einsum("nc,nc->n", z_q, z_pos).unsqueeze(-1)  # positive logits: Nx1
            l_neg = torch.einsum("nc,ck->nk", z_q, z_neg)  # negative logits: NxK

            logit = torch.cat([l_pos, l_neg], dim=1)  # logits: Nx(1+K)
            logit = logit / self.softmax_temperature  # apply temperature
            logits.append(logit)

            # dequeue and enqueue
            self._dequeue_and_enqueue(z_k[i], queue_idx=i)

        # targets: positive key indicators
        targets = torch.zeros(logits[0].shape[0], dtype=torch.long)
        targets = targets.type_as(logits[0])

        return logits, targets


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


@torch.no_grad()
def batch_shuffle_ddp(x):  # pragma: no-cover
    """
    Batch shuffle, for making use of BatchNorm.
    *** Only support DistributedDataParallel (DDP) model. ***
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # random shuffle index
    idx_shuffle = torch.randperm(batch_size_all).cuda()

    # broadcast to all gpus
    torch.distributed.broadcast(idx_shuffle, src=0)

    # index for restoring
    idx_unshuffle = torch.argsort(idx_shuffle)

    # shuffled index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

    return x_gather[idx_this], idx_unshuffle


@torch.no_grad()
def batch_unshuffle_ddp(x, idx_unshuffle):  # pragma: no-cover
    """
    Undo batch shuffle.
    *** Only support DistributedDataParallel (DDP) model. ***
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # restored index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

    return x_gather[idx_this]


MENAGERIE_ZOO = "ported-pytorch"


def build_seco_mocov2_resnet18():
    # Small num_negatives so the queue is cheap; emb_spaces=1 matches the
    # single-embedding-space SeCo variant (moco.py driver script default).
    return MocoV2(
        base_encoder="resnet18",
        emb_dim=16,
        num_negatives=8,
        emb_spaces=1,
        encoder_momentum=0.999,
        softmax_temperature=0.07,
    )


def example_input_seco_mocov2_resnet18():
    img_q = torch.randn(8, 3, 64, 64)
    img_k = [torch.randn(8, 3, 64, 64)]  # one tensor per embedding space
    return (img_q, img_k)


MENAGERIE_ENTRIES = [
    (
        "Seasonal Contrast (SeCo) MoCo-v2 ResNet18",
        "build_seco_mocov2_resnet18",
        "example_input_seco_mocov2_resnet18",
        2021,
        "ported-pytorch",
    ),
]
