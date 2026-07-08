# SOURCE: vendored from OpenMotionLab/MotionGPT @ main
# Files: mGPT/archs/mgpt_vq.py, mGPT/archs/tools/resnet.py,
#        mGPT/archs/tools/quantize_cnn.py (QuantizeEMAReset only -- the class
#        MotionGPT's default config selects)
# Architecture unmodified from the real repo (a ResNet1D convolutional
# VQ-VAE motion tokenizer, following T2M-GPT); only import paths were
# flattened to fit into a single staging module. MotionGPT's language-model
# head (mGPT/archs/mgpt_lm.py) reuses a stock HuggingFace T5/GPT2 checkpoint
# with resized token embeddings -- not an architectural contribution beyond
# base transformers -- so this staging module captures the genuinely novel
# piece: the motion VQ-VAE tokenizer.
from collections import OrderedDict  # noqa: F401 (kept for parity with source imports)
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions.distribution import Distribution  # noqa: F401

MENAGERIE_ZOO = "vendored-pytorch"


# --- mGPT/archs/tools/resnet.py ----------------------------------------------
class nonlinearity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # swish
        return x * torch.sigmoid(x)


class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation="silu", norm=None, dropout=None):
        super().__init__()
        padding = dilation
        self.norm = norm
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()
        elif activation == "silu":
            self.activation1 = nonlinearity()
            self.activation2 = nonlinearity()
        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()

        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0)

    def forward(self, x):
        x_orig = x
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)

        x = self.conv1(x)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)

        x = self.conv2(x)
        x = x + x_orig
        return x


class Resnet1D(nn.Module):
    def __init__(
        self,
        n_in,
        n_depth,
        dilation_growth_rate=1,
        reverse_dilation=True,
        activation="relu",
        norm=None,
    ):
        super().__init__()

        blocks = [
            ResConv1DBlock(
                n_in, n_in, dilation=dilation_growth_rate**depth, activation=activation, norm=norm
            )
            for depth in range(n_depth)
        ]
        if reverse_dilation:
            blocks = blocks[::-1]

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


# --- mGPT/archs/tools/quantize_cnn.py (QuantizeEMAReset only) ----------------
class QuantizeEMAReset(nn.Module):
    def __init__(self, nb_code, code_dim, mu):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu
        self.reset_codebook()

    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.register_buffer("codebook", torch.zeros(self.nb_code, self.code_dim).to(device))

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = out[: self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)
        code_count = code_onehot.sum(dim=-1)

        out = self._tile(x)
        code_rand = out[: self.nb_code]

        self.code_sum = self.mu * self.code_sum + (1.0 - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1.0 - self.mu) * code_count

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(
            self.nb_code, 1
        )

        self.codebook = usage * code_update + (1 - usage) * code_rand
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return perplexity

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])
        return x

    def quantize(self, x):
        k_w = self.codebook.t()
        distance = (
            torch.sum(x**2, dim=-1, keepdim=True)
            - 2 * torch.matmul(x, k_w)
            + torch.sum(k_w**2, dim=0, keepdim=True)
        )
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    def forward(self, x):
        N, width, T = x.shape

        x = self.preprocess(x)

        if self.training and not self.init:
            self.init_codebook(x)

        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)

        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else:
            perplexity = self.compute_perplexity(code_idx)

        commit_loss = F.mse_loss(x, x_d.detach())

        x_d = x + (x_d - x).detach()

        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()
        code_idx = code_idx.view(N, T).contiguous()

        return x_d, commit_loss, perplexity


# --- mGPT/archs/mgpt_vq.py ----------------------------------------------------
class Encoder(nn.Module):
    def __init__(
        self,
        input_emb_width=3,
        output_emb_width=512,
        down_t=3,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
    ):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_emb_width=3,
        output_emb_width=512,
        down_t=3,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
    ):
        super().__init__()
        blocks = []

        filter_t, pad_t = stride_t * 2, stride_t // 2  # noqa: F841 (unused in original repo code)
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(
                    width,
                    depth,
                    dilation_growth_rate,
                    reverse_dilation=True,
                    activation=activation,
                    norm=norm,
                ),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv1d(width, out_dim, 3, 1, 1),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class VQVae(nn.Module):
    def __init__(
        self,
        nfeats: int,
        quantizer: str = "ema_reset",
        code_num=512,
        code_dim=512,
        output_emb_width=512,
        down_t=3,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        norm=None,
        activation: str = "relu",
        **kwargs,
    ) -> None:
        super().__init__()

        self.code_dim = code_dim

        self.encoder = Encoder(
            nfeats,
            output_emb_width,
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=norm,
        )

        self.decoder = Decoder(
            nfeats,
            output_emb_width,
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=norm,
        )

        # This staging module only vendors QuantizeEMAReset (MotionGPT's
        # default "ema_reset" quantizer); Quantizer/QuantizeEMA/QuantizeReset
        # are alternate ablations in the real repo, not the shipped config.
        assert quantizer == "ema_reset"
        self.quantizer = QuantizeEMAReset(code_num, code_dim, mu=0.99)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1)
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, features: Tensor):
        x_in = self.preprocess(features)
        x_encoder = self.encoder(x_in)
        x_quantized, loss, perplexity = self.quantizer(x_encoder)
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)
        return x_out, loss, perplexity

    def encode(self, features: Tensor) -> Union[Tensor, Distribution]:
        N, T, _ = features.shape
        x_in = self.preprocess(features)
        x_encoder = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)
        return code_idx, None

    def decode(self, z: Tensor):
        x_d = self.quantizer.dequantize(z)
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out


# --- staging glue -------------------------------------------------------------
def build_motiongpt_vqvae():
    torch.manual_seed(0)
    model = VQVae(
        nfeats=64,
        quantizer="ema_reset",
        code_num=64,
        code_dim=32,
        output_emb_width=32,
        down_t=2,
        stride_t=2,
        width=32,
        depth=2,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
    )
    # The EMA codebook is lazily initialized on the first *training*-mode
    # forward pass (see QuantizeEMAReset.init_codebook, called from
    # QuantizeEMAReset.forward). Run one warmup pass so the codebook is
    # populated before this module is traced in eval mode.
    model.train()
    with torch.no_grad():
        model(example_input_motiongpt_vqvae()[0])
    return model.eval()


def example_input_motiongpt_vqvae():
    torch.manual_seed(0)
    return (torch.randn(1, 16, 64),)


MENAGERIE_ENTRIES = [
    ("MotionGPT-VQVAE", build_motiongpt_vqvae, example_input_motiongpt_vqvae, 2023, MENAGERIE_ZOO),
]
