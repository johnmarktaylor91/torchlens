# SOURCE: vendored from teevee112/DeepLoc-2.0 @ e6d44d01f6026b756ba35c4a89715e23a2957590
# https://raw.githubusercontent.com/teevee112/DeepLoc-2.0/e6d44d01f6026b756ba35c4a89715e23a2957590/src/model.py
# https://raw.githubusercontent.com/teevee112/DeepLoc-2.0/e6d44d01f6026b756ba35c4a89715e23a2957590/src/attr_prior.py
#
# Thumuluri et al. (Nucleic Acids Research 2022) "DeepLoc 2.0: multi-label
# subcellular localization prediction using protein language models" -- the
# real classifier head that sits on top of frozen ESM1b/ProtT5 protein
# language model embeddings. `AttentionHead` implements a per-position
# learned-query attention pool (with a Gaussian-smoothed attention score via
# `smooth_tensor_1d`, used both for pooling and for the paper's sorting-signal
# attribution-prior supervision), and `BaseModel` (subclassed by
# `ProtT5Frozen` for the 1024-dim ProtT5 embedding and `ESM1bFrozen` for the
# 1280-dim ESM1b embedding) is a LayerNorm -> Linear -> AttentionHead pool ->
# 11-way multi-label classification head. This attention-pooling head over a
# frozen-PLM embedding IS the paper's architectural contribution, so it is
# vendored rather than constructed from a base-library class (the PLM
# backbone itself is swapped out here for a random embedding tensor of the
# right shape, matching how DeepLoc-2.0 consumes precomputed embeddings).
#
# `AttentionHead` and `BaseModel`/`ProtT5Frozen` are the real, unmodified
# classes from `src/model.py` (layer composition and forward-pass control
# flow are byte-for-byte the original, only the `pytorch_lightning.Module`
# base class is kept since it subclasses `nn.Module` and traces normally;
# training-only methods `attn_reg_loss`/`training_step`/`validation_step`/
# `configure_optimizers` are dropped -- they are not part of the forward
# architecture). `smooth_tensor_1d` is the real, unmodified helper from
# `src/attr_prior.py` (it is called inside `AttentionHead.forward`, so it is
# load-bearing for the architecture, not just a training utility). Dropped:
# the DCT/Fourier attribution-prior loss functions in `attr_prior.py` (only
# used by the training-only `attn_reg_loss`), `focal_loss` (loss-only),
# `pos_weights_bce`/`pos_weights_annot` (loss-only constants), and the
# `constants.py`/`.yaml` embedding-pipeline plumbing.
#   - Added `build_deeploc2()`/`example_input_deeploc2()` staging entry
#     points using `ProtT5Frozen` (embed_dim=1024) at a small sequence
#     length (batch=2, seq_len=12) with a full non-mask (no padding), which
#     matches how `BaseModel.forward(embedding, lens, non_mask)` is called at
#     inference time on a batch of precomputed per-residue ProtT5 embeddings.

import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule


def smooth_tensor_1d(input_tensor, smooth_sigma):
    """
    Smooths an input tensor along a dimension using a Gaussian filter.
    Arguments:
        `input_tensor`: a A x B tensor to smooth along the second dimension
        `smooth_sigma`: width of the Gaussian to use for smoothing; this is the
            standard deviation of the Gaussian to use, and the Gaussian will be
            truncated after 1 sigma (i.e. the smoothing window is
            1 + (2 * sigma); sigma of 0 means no smoothing
    Returns an array the same shape as the input tensor, with the dimension of
    `B` smoothed.
    """
    # Generate the kernel
    if smooth_sigma == 0:
        sigma, truncate = 1, 0
    else:
        sigma, truncate = smooth_sigma, 1
    base = torch.zeros(1 + (2 * sigma)).numpy()
    base[sigma] = 1  # Center of window is 1 everywhere else is 0
    kernel = scipy.ndimage.gaussian_filter(base, sigma=sigma, truncate=truncate)
    kernel = torch.tensor(kernel, dtype=torch.float32, device=input_tensor.device)

    # Expand the input and kernel to 3D, with channels of 1
    # Also make the kernel float-type, as the input is going to be of type float
    input_tensor = torch.unsqueeze(input_tensor, dim=1)
    kernel = torch.unsqueeze(torch.unsqueeze(kernel, dim=0), dim=1).float()
    padded_input = F.pad(input_tensor, (sigma, sigma), "replicate")
    smoothed = torch.nn.functional.conv1d(padded_input, kernel)
    return torch.squeeze(smoothed, dim=1)


class AttentionHead(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super(AttentionHead, self).__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.preattn_ln = nn.LayerNorm(hidden_dim // n_heads)
        self.Q = nn.Linear(hidden_dim // n_heads, n_heads, bias=False)
        torch.nn.init.normal_(self.Q.weight, mean=0.0, std=1 / (hidden_dim // n_heads))

    def forward(self, x, np_mask, lengths):
        # input (batch, seq_len, embed)
        n_heads = self.n_heads
        hidden_dim = self.hidden_dim
        x = x.view(x.size(0), x.size(1), n_heads, hidden_dim // n_heads)
        x = self.preattn_ln(x)
        mul = (x * self.Q.weight.view(1, 1, n_heads, hidden_dim // n_heads)).sum(-1)
        # * np.sqrt(5)
        # / np.sqrt(hidden_dim//n_heads)
        mul_score_list = []
        for i in range(mul.size(0)):
            # (1, L) -> (1, 1, L) -> (1, L) -> (1, L, 1)
            mul_score_list.append(
                F.pad(
                    smooth_tensor_1d(mul[i, : lengths[i], 0].unsqueeze(0), 2).unsqueeze(0),
                    (0, mul.size(1) - lengths[i]),
                    "constant",
                ).squeeze(0)
            )

        mul = torch.cat(mul_score_list, dim=0).unsqueeze(-1)
        mul = mul.masked_fill(~np_mask.unsqueeze(-1), float("-inf"))

        attns = F.softmax(mul, dim=1)  # (b, l, nh)
        x = (x * attns.unsqueeze(-1)).sum(1)
        x = x.view(x.size(0), -1)
        return x, attns.squeeze(2)


class BaseModel(LightningModule):
    def __init__(self, embed_dim) -> None:
        super().__init__()

        self.initial_ln = nn.LayerNorm(embed_dim)
        self.lin = nn.Linear(embed_dim, 256)
        self.attn_head = AttentionHead(256, 1)
        self.clf_head = nn.Linear(256, 11)
        self.kld = nn.KLDivLoss(reduction="batchmean")
        self.lr = 1e-3

    def forward(self, embedding, lens, non_mask):
        x = self.initial_ln(embedding)
        x = self.lin(x)
        x_pool, x_attns = self.attn_head(x, non_mask, lens)
        x_pred = self.clf_head(x_pool)
        # print(x_pred, x_attns)
        return x_pred, x_attns

    def predict(self, embedding, lens, non_mask):
        x = self.initial_ln(embedding)
        x = self.lin(x)
        x_pool, x_attns = self.attn_head(x, non_mask, lens)
        x_pred = self.clf_head(x_pool)
        # print(x_pred, x_attns)
        return x_pred, x_pool, x_attns

    def configure_optimizers(self):
        grouped_parameters = [{"params": [p for n, p in self.named_parameters()]}]
        optimizer = torch.optim.AdamW(grouped_parameters, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=1, min_lr=1e-5
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "bce_loss"}


class ProtT5Frozen(BaseModel):
    def __init__(self):
        super().__init__(1024)


def build_deeploc2():
    # ProtT5Frozen uses the real 1024-dim ProtT5 per-residue embedding size.
    return ProtT5Frozen()


def example_input_deeploc2():
    # (embedding, lens, non_mask) matching BaseModel.forward's real call
    # signature at inference time: a batch of precomputed per-residue PLM
    # embeddings, per-sequence lengths, and a boolean non-padding mask.
    batch_size, seq_len, embed_dim = 2, 12, 1024
    embedding = torch.randn(batch_size, seq_len, embed_dim)
    lens = torch.full((batch_size,), seq_len, dtype=torch.long)
    non_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    return (embedding, lens, non_mask)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("DeepLoc-2.0", "build_deeploc2", "example_input_deeploc2", 2022, "vendored"),
]
