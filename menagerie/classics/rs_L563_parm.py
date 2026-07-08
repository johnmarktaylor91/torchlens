# SOURCE: vendored from vansteensellab/PARM @ main
# File vendored verbatim (architecture untouched):
#   PARM/PARM_utils_load_model.py
# No import fix needed: this file only imports torch, torch.nn,
# torch.nn.functional, and einops.layers.torch.Rearrange -- all base libs.
"""PARM: Promoter Activity Regulatory Model.

A ResNet + attention-pooling 1D-convolutional network that predicts
regulatory (promoter/enhancer) activity from a one-hot-encoded DNA sequence.
Reference: https://github.com/vansteensellab/PARM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

MENAGERIE_ZOO = "vendored-pytorch"


# --- PARM/PARM_utils_load_model.py (verbatim architecture) ---


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class GELU(nn.Module):
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def ConvBlock(dim, dim_out=None, kernel_size=1):
    return nn.Sequential(
        nn.BatchNorm1d(dim),
        GELU(),
        nn.Conv1d(dim, default(dim_out, dim), kernel_size, padding=kernel_size // 2),
    )


class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size=2):
        super().__init__()
        self.pool_size = pool_size
        # (n p) are length of sequence
        self.pool_fn = Rearrange("b d (n p) -> b d n p", p=pool_size)
        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0
        if needs_padding:
            x = F.pad(x, (0, remainder), value=0)
            mask = torch.zeros((b, 1, n), dtype=torch.bool, device=x.device)
            mask = F.pad(mask, (0, remainder), value=True)
        x = self.pool_fn(x)
        logits = self.to_attn_logits(x)
        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)
        attn = logits.softmax(dim=-1)
        return (x * attn).sum(dim=-1)


class ResNet_Attentionpool(nn.Module):
    def __init__(
        self,
        L_max,
        n_block,
        filter_size=125,
        weight_file=None,
        cell_line=False,
        type_loss="poisson",
        validation=False,
        index_interested_output=False,
        maxglobalpool=True,
        vocab=4,
        use_AttentionPool=True,
    ):
        super(ResNet_Attentionpool, self).__init__()

        self.type_loss = type_loss
        if type_loss == "heteroscedastic":
            self.heteroscedastic = True
        else:
            self.heteroscedastic = False
        self.index_interested_output = index_interested_output
        self.validation = validation
        self.maxglobalpool = maxglobalpool
        self.L_max = L_max  # Max length of sequence
        self.vocab = vocab  # N nucleotides

        kernel_size = 7
        stem_kernel_size = 7

        if cell_line:
            output_nodes = len(cell_line.split("__"))
        else:
            output_nodes = 1

        self.n_blocks = n_block

        # create stem
        self.stem = nn.Sequential(
            nn.Conv1d(vocab, filter_size, stem_kernel_size, padding="same"),
            Residual(ConvBlock(filter_size)),
            AttentionPool(filter_size, pool_size=2) if use_AttentionPool else nn.MaxPool1d(2),
        )

        # create conv tower
        conv_layers = []

        initial_filter_size = filter_size
        prev_filter_size = filter_size
        for block in range(self.n_blocks):
            if block > 4:
                filter_size = int(initial_filter_size * 0.2)

            conv_layers.append(
                nn.Sequential(
                    ConvBlock(prev_filter_size, filter_size, kernel_size=kernel_size),
                    Residual(ConvBlock(filter_size, filter_size, kernel_size=1)),
                    AttentionPool(filter_size, pool_size=2)
                    if use_AttentionPool
                    else nn.MaxPool1d(2),
                )
            )

            prev_filter_size = filter_size

        self.conv_tower = nn.Sequential(*conv_layers)

        self.linear1 = nn.Linear(filter_size, 1)
        if self.heteroscedastic:
            self.log_var = nn.Linear(filter_size, output_nodes)  # Log-variance output

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.stem(x)

        out = self.conv_tower(out)

        if self.maxglobalpool:
            # max in length
            out = torch.max(out, dim=-1).values

        out = out.view(out.size(0), -1)

        if self.heteroscedastic:
            mu = self.linear1(out)
            log_var = self.log_var(out)  # Log variance
            if self.validation:
                return mu
            return mu, log_var

        else:
            out = self.linear1(out)

        if self.type_loss == "poisson":
            out = self.relu(out)
        if self.index_interested_output:
            out = out[:, self.index_interested_output].unsqueeze(1)

        return out


# --- staging harness ---

_L_MAX = 128
_N_BLOCK = 3
_FILTER_SIZE = 16


def build_parm_resnet_attentionpool():
    model = ResNet_Attentionpool(
        L_max=_L_MAX,
        n_block=_N_BLOCK,
        filter_size=_FILTER_SIZE,
        weight_file=None,
        type_loss="poisson",
        cell_line=False,
        validation=True,
        index_interested_output=False,
        maxglobalpool=True,
        use_AttentionPool=True,
    )
    model.eval()
    return model


def example_input_parm_resnet_attentionpool():
    # (batch, 4 nucleotides, L_max) one-hot-encoded DNA sequence, matching
    # the real PARM_predict.get_prediction() input layout.
    return torch.rand(2, 4, _L_MAX)


MENAGERIE_ENTRIES = [
    (
        "PARM-ResNet-AttentionPool",
        "build_parm_resnet_attentionpool",
        "example_input_parm_resnet_attentionpool",
        2024,
        "vendored-pytorch",
    ),
]
