# SOURCE: vendored from EricGuo5513/momask-codes @ main
# Files: models/vq/model.py, models/vq/encdec.py, models/vq/resnet.py,
#        models/vq/residual_vq.py, models/vq/quantizer.py
# Architecture unmodified from the real repo; only import paths were flattened
# to fit into a single staging module and a minimal `args` shim (SimpleNamespace
# with `.mu`) replaces the argparse Namespace the real code expects.
import random
from math import ceil
from random import randrange

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

MENAGERIE_ZOO = "vendored-pytorch"


# --- models/vq/resnet.py ---------------------------------------------------
class nonlinearity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation="silu", norm=None, dropout=0.2):
        super(ResConv1DBlock, self).__init__()

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
        self.dropout = nn.Dropout(dropout)

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
        x = self.dropout(x)
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


# --- models/vq/encdec.py ----------------------------------------------------
class Encoder(nn.Module):
    def __init__(
        self,
        input_emb_width=3,
        output_emb_width=512,
        down_t=2,
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
        down_t=2,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
    ):
        super().__init__()
        blocks = []

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
        x = self.model(x)
        return x.permute(0, 2, 1)


# --- models/vq/quantizer.py --------------------------------------------------
def gumbel_log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -gumbel_log(-gumbel_log(noise))


def gumbel_sample(logits, temperature=1.0, stochastic=False, dim=-1, training=True):
    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    else:
        sampling_logits = logits

    ind = sampling_logits.argmax(dim=dim)
    return ind


class QuantizeEMAReset(nn.Module):
    def __init__(self, nb_code, code_dim, args):
        super(QuantizeEMAReset, self).__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = args.mu
        self.reset_codebook()

    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer(
            "codebook", torch.zeros(self.nb_code, self.code_dim, requires_grad=False)
        )

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

    def quantize(self, x, sample_codebook_temp=0.0):
        k_w = self.codebook.t()
        distance = (
            torch.sum(x**2, dim=-1, keepdim=True)
            - 2 * torch.matmul(x, k_w)
            + torch.sum(k_w**2, dim=0, keepdim=True)
        )
        code_idx = gumbel_sample(
            -distance,
            dim=-1,
            temperature=sample_codebook_temp,
            stochastic=True,
            training=self.training,
        )
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    def get_codebook_entry(self, indices):
        return self.dequantize(indices).permute(0, 2, 1)

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
        # NCT -> [NT, C]
        x = x.permute(0, 2, 1).contiguous().view(-1, x.shape[1])
        return x

    def forward(self, x, return_idx=False, temperature=0.0):
        N, width, T = x.shape

        x = self.preprocess(x)
        if self.training and not self.init:
            self.init_codebook(x)

        code_idx = self.quantize(x, temperature)
        x_d = self.dequantize(code_idx)

        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else:
            perplexity = self.compute_perplexity(code_idx)

        commit_loss = F.mse_loss(x, x_d.detach())

        x_d = x + (x_d - x).detach()

        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()
        code_idx = code_idx.view(N, T).contiguous()
        if return_idx:
            return x_d, code_idx, commit_loss, perplexity
        return x_d, commit_loss, perplexity


class QuantizeEMA(QuantizeEMAReset):
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)
        code_count = code_onehot.sum(dim=-1)

        self.code_sum = self.mu * self.code_sum + (1.0 - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1.0 - self.mu) * code_count

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(
            self.nb_code, 1
        )
        self.codebook = usage * code_update + (1 - usage) * self.codebook

        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity


# --- models/vq/residual_vq.py ------------------------------------------------
def rvq_exists(val):
    return val is not None


def rvq_default(val, d):
    return val if rvq_exists(val) else d


def round_up_multiple(num, mult):
    return ceil(num / mult) * mult


class ResidualVQ(nn.Module):
    """Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf"""

    def __init__(
        self,
        num_quantizers,
        shared_codebook=False,
        quantize_dropout_prob=0.5,
        quantize_dropout_cutoff_index=0,
        **kwargs,
    ):
        super().__init__()

        self.num_quantizers = num_quantizers

        if shared_codebook:
            layer = QuantizeEMAReset(**kwargs)
            self.layers = nn.ModuleList([layer for _ in range(num_quantizers)])
        else:
            self.layers = nn.ModuleList([QuantizeEMAReset(**kwargs) for _ in range(num_quantizers)])

        assert quantize_dropout_cutoff_index >= 0 and quantize_dropout_prob >= 0

        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_prob = quantize_dropout_prob

    @property
    def codebooks(self):
        codebooks = [layer.codebook for layer in self.layers]
        codebooks = torch.stack(codebooks, dim=0)
        return codebooks

    def get_codes_from_indices(self, indices):
        batch, quantize_dim = indices.shape[0], indices.shape[-1]

        if quantize_dim < self.num_quantizers:
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value=-1)

        codebooks = repeat(self.codebooks, "q c d -> q b c d", b=batch)
        gather_indices = repeat(indices, "b n q -> q b n d", d=codebooks.shape[-1])

        mask = gather_indices == -1.0
        gather_indices = gather_indices.masked_fill(mask, 0)

        all_codes = codebooks.gather(2, gather_indices)
        all_codes = all_codes.masked_fill(mask, 0.0)

        return all_codes

    def get_codebook_entry(self, indices):
        all_codes = self.get_codes_from_indices(indices)
        latent = torch.sum(all_codes, dim=0)
        latent = latent.permute(0, 2, 1)
        return latent

    def forward(self, x, return_all_codes=False, sample_codebook_temp=None, force_dropout_index=-1):
        num_quant, quant_dropout_prob, device = (  # noqa: F841 (quant_dropout_prob unused in original repo code)
            self.num_quantizers,
            self.quantize_dropout_prob,
            x.device,
        )

        quantized_out = 0.0
        residual = x

        all_losses = []
        all_indices = []
        all_perplexity = []

        should_quantize_dropout = self.training and random.random() < self.quantize_dropout_prob

        start_drop_quantize_index = num_quant
        if should_quantize_dropout:
            start_drop_quantize_index = randrange(self.quantize_dropout_cutoff_index, num_quant)
            null_indices_shape = [x.shape[0], x.shape[-1]]
            null_indices = torch.full(null_indices_shape, -1.0, device=device, dtype=torch.long)

        if force_dropout_index >= 0:
            should_quantize_dropout = True
            start_drop_quantize_index = force_dropout_index
            null_indices_shape = [x.shape[0], x.shape[-1]]
            null_indices = torch.full(null_indices_shape, -1.0, device=device, dtype=torch.long)

        for quantizer_index, layer in enumerate(self.layers):
            if should_quantize_dropout and quantizer_index > start_drop_quantize_index:
                all_indices.append(null_indices)
                continue

            quantized, *rest = layer(residual, return_idx=True, temperature=sample_codebook_temp)

            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            embed_indices, loss, perplexity = rest
            all_indices.append(embed_indices)
            all_losses.append(loss)
            all_perplexity.append(perplexity)

        all_indices = torch.stack(all_indices, dim=-1)
        all_losses = sum(all_losses) / len(all_losses)
        all_perplexity = sum(all_perplexity) / len(all_perplexity)

        ret = (quantized_out, all_indices, all_losses, all_perplexity)

        if return_all_codes:
            all_codes = self.get_codes_from_indices(all_indices)
            ret = (*ret, all_codes)

        return ret

    def quantize(self, x, return_latent=False):
        all_indices = []
        quantized_out = 0.0
        residual = x
        all_codes = []
        for quantizer_index, layer in enumerate(self.layers):
            quantized, *rest = layer(residual, return_idx=True)

            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            embed_indices, loss, perplexity = rest
            all_indices.append(embed_indices)
            all_codes.append(quantized)

        code_idx = torch.stack(all_indices, dim=-1)
        all_codes = torch.stack(all_codes, dim=0)
        if return_latent:
            return code_idx, all_codes
        return code_idx


# --- models/vq/model.py -------------------------------------------------------
class RVQVAE(nn.Module):
    def __init__(
        self,
        args,
        input_width=263,
        nb_code=1024,
        code_dim=512,
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
        assert output_emb_width == code_dim
        self.code_dim = code_dim
        self.num_code = nb_code
        self.encoder = Encoder(
            input_width,
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
            input_width,
            output_emb_width,
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=norm,
        )
        rvqvae_config = {
            "num_quantizers": args.num_quantizers,
            "shared_codebook": args.shared_codebook,
            "quantize_dropout_prob": args.quantize_dropout_prob,
            "quantize_dropout_cutoff_index": 0,
            "nb_code": nb_code,
            "code_dim": code_dim,
            "args": args,
        }
        self.quantizer = ResidualVQ(**rvqvae_config)

    def preprocess(self, x):
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        code_idx, all_codes = self.quantizer.quantize(x_encoder, return_latent=True)
        return code_idx, all_codes

    def forward(self, x):
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(
            x_encoder, sample_codebook_temp=0.5
        )
        x_out = self.decoder(x_quantized)
        return x_out, commit_loss, perplexity

    def forward_decoder(self, x):
        x_d = self.quantizer.get_codes_from_indices(x)
        x = x_d.sum(dim=0).permute(0, 2, 1)
        x_out = self.decoder(x)
        return x_out


# --- staging glue -------------------------------------------------------------
def _momask_args():
    from types import SimpleNamespace

    return SimpleNamespace(
        mu=0.99,
        num_quantizers=3,
        shared_codebook=False,
        quantize_dropout_prob=0.0,
    )


def build_momask_rvqvae():
    torch.manual_seed(0)
    args = _momask_args()
    model = RVQVAE(
        args,
        input_width=64,
        nb_code=64,
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
    # The EMA codebooks are lazily initialized on the first *training*-mode
    # forward pass (see QuantizeEMAReset.init_codebook, called from
    # QuantizeEMAReset.forward). Run one warmup pass so the codebooks are
    # populated before this module is traced in eval mode.
    model.train()
    with torch.no_grad():
        model(example_input_momask_rvqvae()[0])
    return model.eval()


def example_input_momask_rvqvae():
    return (torch.randn(1, 16, 64),)


MENAGERIE_ENTRIES = [
    ("MoMask-RVQVAE", build_momask_rvqvae, example_input_momask_rvqvae, 2023, MENAGERIE_ZOO),
]
