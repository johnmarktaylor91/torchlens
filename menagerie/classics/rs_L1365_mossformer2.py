# SOURCE: vendored from https://github.com/alibabasglab/MossFormer2 @ 5727f739947605f884da8d7a1448c64f32d37b9d
#   Vendored files (MossFormer2_standalone/):
#     - model/mossformer2.py                 -> `Mossformer2Wrapper` (top-level encoder/masknet
#       /decoder wrapper for monaural speech separation, "MossFormer2: Combining Transformer
#       and RNN-Free Recurrent Network for Enhanced Time-Domain Monaural Speech Separation",
#       Zhao & Yip 2024, https://arxiv.org/pdf/2312.11825v1.pdf).
#     - model/utils/one_path_flash_fsmn.py    -> `Encoder`, `Decoder`, `Dual_Path_Model`,
#       `Dual_Computation_Block`, `SBFLASHBlock_DualA` (the masking network; despite the
#       "dual-path" naming this standalone build is single-path -- `Dual_Computation_Block`
#       only runs the intra branch, reproduced verbatim including that detail).
#     - model/utils/Transformer.py            -> `TransformerEncoder_FLASH_DualA_FSMN`,
#       `FLASHTransformer_DualA_FSMN`, `FLASH_ShareA_FFConvM` (FLASH-style gated attention +
#       rotary positional encoding), `Gated_FSMN_Block_Dilated` / `Gated_FSMN_dilated` (the
#       RNN-free recurrent gated-FSMN module that distinguishes MossFormer2 from MossFormer).
#     - model/utils/conv_module.py, fsmn.py, normalization.py -> supporting nn.Module building
#       blocks (ConvModule, UniDeepFsmn/UniDeepFsmn_dilated/DilatedDenseNet,
#       LayerNorm/CLayerNorm/ScaleNorm).
#
# This is the alibabasglab MossFormer2_standalone reference implementation (distinct from
# MossFormer -- confirmed same lab, but MossFormer2 replaces the plain gated single-head
# attention with FLASH-style dual-attention plus a dilated gated-FSMN recurrent module).
# Every module is reproduced verbatim; only relative imports were flattened into this single
# file, and the HuggingFace checkpoint download / `PyTorchModelHubMixin.from_pretrained`
# machinery is stripped since we construct the wrapper with random init directly
# (`Mossformer2Wrapper(config)`).

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch import einsum

MENAGERIE_ZOO = "vendored-pytorch"


# ---- model/utils/normalization.py (verbatim) ----


class LayerNorm(nn.Module):
    """Applies layer normalization to the input tensor (ported from sb.nnet.normalization)."""

    def __init__(self, input_size=None, input_shape=None, eps=1e-05, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if input_shape is not None:
            input_size = input_shape[2:]

        self.norm = torch.nn.LayerNorm(
            input_size, eps=self.eps, elementwise_affine=self.elementwise_affine
        )

    def forward(self, x):
        return self.norm(x)


class CLayerNorm(nn.LayerNorm):
    """Channel-wise layer normalization for [N, C, T] tensors."""

    def forward(self, sample):
        if sample.dim() != 3:
            raise RuntimeError("{} only accept 3-D tensor as input".format(self.__class__.__name__))
        sample = torch.transpose(sample, 1, 2)
        sample = super().forward(sample)
        sample = torch.transpose(sample, 1, 2)
        return sample


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


# ---- model/utils/conv_module.py (verbatim) ----


class Transpose(nn.Module):
    def __init__(self, shape: tuple):
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(*self.shape)


class DepthwiseConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        assert out_channels % in_channels == 0
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(inputs)


class ConvModule(nn.Module):
    def __init__(
        self, in_channels, kernel_size: int = 17, expansion_factor: int = 2, dropout_p: float = 0.1
    ):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        assert expansion_factor == 2
        self.sequential = nn.Sequential(
            Transpose(shape=(1, 2)),
            DepthwiseConv1d(
                in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2
            ),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.sequential(inputs).transpose(1, 2)


# ---- model/utils/fsmn.py (verbatim) ----


class UniDeepFsmn(nn.Module):
    def __init__(self, input_dim, output_dim, lorder=None, hidden_size=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        if lorder is None:
            return
        self.lorder = lorder
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_dim, hidden_size)
        self.project = nn.Linear(hidden_size, output_dim, bias=False)
        self.conv1 = nn.Conv2d(
            output_dim, output_dim, [lorder + lorder - 1, 1], [1, 1], groups=output_dim, bias=False
        )

    def forward(self, input):
        f1 = F.relu(self.linear(input))
        p1 = self.project(f1)
        x = torch.unsqueeze(p1, 1)
        x_per = x.permute(0, 3, 2, 1)
        y = F.pad(x_per, [0, 0, self.lorder - 1, self.lorder - 1])
        out = x_per + self.conv1(y)
        out1 = out.permute(0, 3, 2, 1)
        return input + out1.squeeze()


class DilatedDenseNet(nn.Module):
    def __init__(self, depth=4, lorder=20, in_channels=64):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.0)
        self.twidth = lorder * 2 - 1
        self.kernel_size = (self.twidth, 1)
        for i in range(self.depth):
            dil = 2**i
            pad_length = lorder + (dil - 1) * (lorder - 1) - 1
            setattr(
                self,
                "pad{}".format(i + 1),
                nn.ConstantPad2d((0, 0, pad_length, pad_length), value=0.0),
            )
            setattr(
                self,
                "conv{}".format(i + 1),
                nn.Conv2d(
                    self.in_channels * (i + 1),
                    self.in_channels,
                    kernel_size=self.kernel_size,
                    dilation=(dil, 1),
                    groups=self.in_channels,
                    bias=False,
                ),
            )
            setattr(self, "norm{}".format(i + 1), nn.InstanceNorm2d(in_channels, affine=True))
            setattr(self, "prelu{}".format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, "pad{}".format(i + 1))(skip)
            out = getattr(self, "conv{}".format(i + 1))(out)
            out = getattr(self, "norm{}".format(i + 1))(out)
            out = getattr(self, "prelu{}".format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out


class UniDeepFsmn_dilated(nn.Module):
    def __init__(self, input_dim, output_dim, lorder=None, hidden_size=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        if lorder is None:
            return
        self.lorder = lorder
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_dim, hidden_size)
        self.project = nn.Linear(hidden_size, output_dim, bias=False)
        self.conv = DilatedDenseNet(depth=2, lorder=lorder, in_channels=output_dim)

    def forward(self, input):
        f1 = F.relu(self.linear(input))
        p1 = self.project(f1)
        x = torch.unsqueeze(p1, 1)
        x_per = x.permute(0, 3, 2, 1)
        out = self.conv(x_per)
        out1 = out.permute(0, 3, 2, 1)
        return input + out1.squeeze()


# ---- model/utils/Transformer.py (verbatim: FLASH-style gated attention block) ----


def exists(val):
    return val is not None


def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder


def default(val, d):
    return val if exists(val) else d


class FFConvM(nn.Module):
    def __init__(self, dim_in, dim_out, norm_klass=nn.LayerNorm, dropout=0.1):
        super().__init__()
        self.mdl = nn.Sequential(
            norm_klass(dim_in),
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            ConvModule(dim_out),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.mdl(x)


class Gated_FSMN_dilated(nn.Module):
    def __init__(self, in_channels, out_channels, lorder, hidden_size):
        super().__init__()
        self.to_u = FFConvM(
            dim_in=in_channels, dim_out=hidden_size, norm_klass=nn.LayerNorm, dropout=0.1
        )
        self.to_v = FFConvM(
            dim_in=in_channels, dim_out=hidden_size, norm_klass=nn.LayerNorm, dropout=0.1
        )
        self.fsmn = UniDeepFsmn_dilated(in_channels, out_channels, lorder, hidden_size)

    def forward(self, x):
        input = x
        x_u = self.to_u(x)
        x_v = self.to_v(x)
        x_u = self.fsmn(x_u)
        x = x_v * x_u + input
        return x


class Gated_FSMN_Block_Dilated(nn.Module):
    """1-D convolutional block: conv1x1 -> gated dilated FSMN -> conv1x1, residual."""

    def __init__(self, dim, inner_channels=256, group_size=256, norm_type="scalenorm"):
        super().__init__()
        if norm_type == "scalenorm":
            norm_klass = ScaleNorm  # noqa: F841 (unused upstream artifact, kept for parity)
        elif norm_type == "layernorm":
            norm_klass = nn.LayerNorm  # noqa: F841

        self.group_size = group_size

        self.conv1 = nn.Sequential(nn.Conv1d(dim, inner_channels, kernel_size=1), nn.PReLU())
        self.norm1 = CLayerNorm(inner_channels)
        self.gated_fsmn = Gated_FSMN_dilated(
            inner_channels, inner_channels, lorder=20, hidden_size=inner_channels
        )
        self.norm2 = CLayerNorm(inner_channels)
        self.conv2 = nn.Conv1d(inner_channels, dim, kernel_size=1)

    def forward(self, input):
        conv1 = self.conv1(input.transpose(2, 1))
        norm1 = self.norm1(conv1)
        seq_out = self.gated_fsmn(norm1.transpose(2, 1))
        norm2 = self.norm2(seq_out.transpose(2, 1))
        conv2 = self.conv2(norm2)
        return conv2.transpose(2, 1) + input


class OffsetScale(nn.Module):
    def __init__(self, dim, heads=1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std=0.02)

    def forward(self, x):
        out = einsum("... d, h d -> ... h d", x, self.gamma) + self.beta
        return out.unbind(dim=-2)


class FLASH_ShareA_FFConvM(nn.Module):
    def __init__(
        self,
        *,
        dim,
        group_size=256,
        query_key_dim=128,
        expansion_factor=1.0,
        causal=False,
        dropout=0.1,
        rotary_pos_emb=None,
        norm_klass=nn.LayerNorm,
        shift_tokens=True,
    ):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.group_size = group_size
        self.causal = causal
        self.shift_tokens = shift_tokens

        self.rotary_pos_emb = rotary_pos_emb
        self.dropout = nn.Dropout(dropout)

        self.to_hidden = FFConvM(
            dim_in=dim, dim_out=hidden_dim, norm_klass=norm_klass, dropout=dropout
        )
        self.to_qk = FFConvM(
            dim_in=dim, dim_out=query_key_dim, norm_klass=norm_klass, dropout=dropout
        )

        self.qk_offset_scale = OffsetScale(query_key_dim, heads=4)

        self.to_out = FFConvM(dim_in=dim * 2, dim_out=dim, norm_klass=norm_klass, dropout=dropout)

        self.gateActivate = nn.Sigmoid()

    def forward(self, x, *, mask=None):
        normed_x = x
        residual = x  # noqa: F841 (upstream artifact, kept for parity)

        if self.shift_tokens:
            x_shift, x_pass = normed_x.chunk(2, dim=-1)
            x_shift = F.pad(x_shift, (0, 0, 1, -1), value=0.0)
            normed_x = torch.cat((x_shift, x_pass), dim=-1)

        v, u = self.to_hidden(normed_x).chunk(2, dim=-1)
        qk = self.to_qk(normed_x)

        quad_q, lin_q, quad_k, lin_k = self.qk_offset_scale(qk)
        att_v, att_u = self.cal_attention(x, quad_q, lin_q, quad_k, lin_k, v, u)

        out = (att_u * v) * self.gateActivate(att_v * u)

        x = x + self.to_out(out)
        return x

    def cal_attention(self, x, quad_q, lin_q, quad_k, lin_k, v, u, mask=None):
        b, n, device, g = x.shape[0], x.shape[-2], x.device, self.group_size

        if exists(mask):
            lin_mask = rearrange(mask, "... -> ... 1")
            lin_k = lin_k.masked_fill(~lin_mask, 0.0)

        if exists(self.rotary_pos_emb):
            quad_q, lin_q, quad_k, lin_k = map(
                self.rotary_pos_emb.rotate_queries_or_keys, (quad_q, lin_q, quad_k, lin_k)
            )

        padding = padding_to_multiple_of(n, g)

        if padding > 0:
            quad_q, quad_k, lin_q, lin_k, v, u = map(
                lambda t: F.pad(t, (0, 0, 0, padding), value=0.0),
                (quad_q, quad_k, lin_q, lin_k, v, u),
            )

            mask = default(mask, torch.ones((b, n), device=device, dtype=torch.bool))
            mask = F.pad(mask, (0, padding), value=False)

        quad_q, quad_k, lin_q, lin_k, v, u = map(
            lambda t: rearrange(t, "b (g n) d -> b g n d", n=self.group_size),
            (quad_q, quad_k, lin_q, lin_k, v, u),
        )

        if exists(mask):
            mask = rearrange(mask, "b (g j) -> b g 1 j", j=g)

        sim = einsum("... i d, ... j d -> ... i j", quad_q, quad_k) / g

        attn = F.relu(sim) ** 2
        attn = self.dropout(attn)

        if exists(mask):
            attn = attn.masked_fill(~mask, 0.0)

        if self.causal:
            causal_mask = torch.ones((g, g), dtype=torch.bool, device=device).triu(1)
            attn = attn.masked_fill(causal_mask, 0.0)

        quad_out_v = einsum("... i j, ... j d -> ... i d", attn, v)
        quad_out_u = einsum("... i j, ... j d -> ... i d", attn, u)

        if self.causal:
            lin_kv = einsum("b g n d, b g n e -> b g d e", lin_k, v) / g
            lin_kv = lin_kv.cumsum(dim=1)
            lin_kv = F.pad(lin_kv, (0, 0, 0, 0, 1, -1), value=0.0)
            lin_out_v = einsum("b g d e, b g n d -> b g n e", lin_kv, lin_q)

            lin_ku = einsum("b g n d, b g n e -> b g d e", lin_k, u) / g
            lin_ku = lin_ku.cumsum(dim=1)
            lin_ku = F.pad(lin_ku, (0, 0, 0, 0, 1, -1), value=0.0)
            lin_out_u = einsum("b g d e, b g n d -> b g n e", lin_ku, lin_q)
        else:
            lin_kv = einsum("b g n d, b g n e -> b d e", lin_k, v) / n
            lin_out_v = einsum("b g n d, b d e -> b g n e", lin_q, lin_kv)

            lin_ku = einsum("b g n d, b g n e -> b d e", lin_k, u) / n
            lin_out_u = einsum("b g n d, b d e -> b g n e", lin_q, lin_ku)

        return map(
            lambda t: rearrange(t, "b g n d -> b (g n) d")[:, :n],
            (quad_out_v + lin_out_v, quad_out_u + lin_out_u),
        )


class FLASHTransformer_DualA_FSMN(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        group_size=256,
        query_key_dim=128,
        expansion_factor=4.0,
        causal=False,
        attn_dropout=0.1,
        norm_type="scalenorm",
        shift_tokens=True,
    ):
        super().__init__()
        assert norm_type in ("scalenorm", "layernorm"), (
            "norm_type must be one of scalenorm or layernorm"
        )

        if norm_type == "scalenorm":
            norm_klass = ScaleNorm
        elif norm_type == "layernorm":
            norm_klass = nn.LayerNorm

        self.group_size = group_size

        rotary_pos_emb = RotaryEmbedding(dim=min(32, query_key_dim))
        self.fsmn = nn.ModuleList([Gated_FSMN_Block_Dilated(dim) for _ in range(depth)])
        self.layers = nn.ModuleList(
            [
                FLASH_ShareA_FFConvM(
                    dim=dim,
                    group_size=group_size,
                    query_key_dim=query_key_dim,
                    expansion_factor=expansion_factor,
                    causal=causal,
                    dropout=attn_dropout,
                    rotary_pos_emb=rotary_pos_emb,
                    norm_klass=norm_klass,
                    shift_tokens=shift_tokens,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x, *, mask=None):
        ii = 0
        for flash in self.layers:
            x = flash(x, mask=mask)
            x = self.fsmn[ii](x)
            ii = ii + 1
        return x


class TransformerEncoder_FLASH_DualA_FSMN(nn.Module):
    """Transformer encoder wrapping `FLASHTransformer_DualA_FSMN` + a final LayerNorm.
    `nhead`/`d_ffn`/`dropout`/`activation`/etc. are accepted for signature parity with the
    real class but (exactly as in the real code) are unused -- only `num_layers` and
    `d_model` feed into the actual computation."""

    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn,
        input_shape=None,
        d_model=None,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=False,
        causal=False,
        attention_type="regularMHA",
    ):
        super().__init__()
        self.flashT = FLASHTransformer_DualA_FSMN(dim=d_model, depth=num_layers)
        self.norm = LayerNorm(input_size=d_model, eps=1e-6)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos_embs=None):
        output = self.flashT(src)
        output = self.norm(output)
        return output, None


# ---- model/utils/one_path_flash_fsmn.py (verbatim call path) ----


class ScaledSinuEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(
            torch.ones(
                1,
            )
        )
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        n, device = x.shape[1], x.device
        t = torch.arange(n, device=device).type_as(self.inv_freq)
        sinu = einsum("i , j -> i j", t, self.inv_freq)
        emb = torch.cat((sinu.sin(), sinu.cos()), dim=-1)
        return emb * self.scale


class Linear(nn.Module):
    """Computes a linear transformation y = wx + b."""

    def __init__(self, n_neurons, input_shape=None, input_size=None, bias=True, combine_dims=False):
        super().__init__()
        self.combine_dims = combine_dims

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size")

        if input_size is None:
            input_size = input_shape[-1]
            if len(input_shape) == 4 and self.combine_dims:
                input_size = input_shape[2] * input_shape[3]

        self.w = nn.Linear(input_size, n_neurons, bias=bias)

    def forward(self, x):
        if x.ndim == 4 and self.combine_dims:
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        wx = self.w(x)
        return wx


class GlobalLayerNorm(nn.Module):
    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x - mean) ** 2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)
        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x - mean) ** 2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    def __init__(self, dim, elementwise_affine=True):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=1e-8)

    def forward(self, x):
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
            x = super().forward(x)
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            x = torch.transpose(x, 1, 2)
            x = super().forward(x)
            x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim, shape):
    if norm == "gln":
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == "cln":
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == "ln":
        return nn.GroupNorm(1, dim, eps=1e-8)
    else:
        return nn.BatchNorm1d(dim)


class Encoder(nn.Module):
    """Convolutional Encoder Layer: Conv1d over raw waveform -> ReLU."""

    def __init__(self, kernel_size=2, out_channels=64, in_channels=1):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            groups=1,
            bias=False,
        )
        self.in_channels = in_channels

    def forward(self, x):
        if self.in_channels == 1:
            x = torch.unsqueeze(x, dim=1)
        x = self.conv1d(x)
        x = F.relu(x)
        return x


class Decoder(nn.ConvTranspose1d):
    """A decoder layer consisting of ConvTranspose1d."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 3/4D tensor as input".format(self.__class__.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)
        return x


class SBFLASHBlock_DualA(nn.Module):
    """A wrapper for the (Speechbrain-style) transformer encoder."""

    def __init__(
        self,
        num_layers,
        d_model,
        nhead,
        d_ffn=2048,
        input_shape=None,
        kdim=None,
        vdim=None,
        dropout=0.1,
        activation="relu",
        use_positional_encoding=False,
        norm_before=False,
        attention_type="regularMHA",
    ):
        super().__init__()
        self.use_positional_encoding = use_positional_encoding

        if activation == "relu":
            activation = nn.ReLU
        elif activation == "gelu":
            activation = nn.GELU
        else:
            raise ValueError("unknown activation")

        self.mdl = TransformerEncoder_FLASH_DualA_FSMN(
            num_layers=num_layers,
            nhead=nhead,
            d_ffn=d_ffn,
            input_shape=input_shape,
            d_model=d_model,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout,
            activation=activation,
            normalize_before=norm_before,
            attention_type=attention_type,
        )

    def forward(self, x):
        # NOTE (faithfulness deviation, minimal glue-code fix): the real
        # `SBFLASHBlock_DualA.forward` returns `self.mdl(x)` un-unpacked, but
        # `TransformerEncoder_FLASH_DualA_FSMN.forward` returns a `(output, None)` tuple
        # (see its own docstring: `output, _ = net(x)`). As literally written upstream this
        # makes `Dual_Computation_Block.forward`'s `intra.permute(...)` crash on a tuple for
        # any `intra_numlayers >= 1` (every shipped config uses `intra_numlayers >= 1`) --
        # a real, unreachable-as-written bug in the MossFormer2_standalone repo, confirmed
        # against the maintained ClearerVoice-Studio training fork (which refactored this
        # wrapper away entirely and never hits the bug). We unpack the tuple here, matching
        # every other real caller of this Transformer encoder pattern; no nn.Module
        # architecture is altered, only this wiring omission.
        output, _ = self.mdl(x)
        return output


class Dual_Computation_Block(nn.Module):
    """Computation block for (single-path, "intra"-only) processing -- reproduced verbatim
    from the real MossFormer2_standalone code, which drops the inter branch entirely."""

    def __init__(
        self,
        intra_mdl,
        out_channels,
        norm="ln",
        skip_around_intra=True,
        linear_layer_after_inter_intra=True,
    ):
        super().__init__()

        self.intra_mdl = intra_mdl
        self.skip_around_intra = skip_around_intra
        self.linear_layer_after_inter_intra = linear_layer_after_inter_intra

        self.norm = norm
        if norm is not None:
            self.intra_norm = select_norm(norm, out_channels, 3)

        if linear_layer_after_inter_intra:
            self.intra_linear = Linear(out_channels, input_size=out_channels)

    def forward(self, x):
        # x: [B, N, S]
        intra = x.permute(0, 2, 1).contiguous()

        intra = self.intra_mdl(intra)

        if self.linear_layer_after_inter_intra:
            intra = self.intra_linear(intra)

        intra = intra.permute(0, 2, 1).contiguous()
        if self.norm is not None:
            intra = self.intra_norm(intra)

        if self.skip_around_intra:
            intra = intra + x

        out = intra
        return out


class Dual_Path_Model(nn.Module):
    """Masking network (single-path despite the "dual-path" naming legacy): normalize ->
    conv1d encoder -> global positional encoding -> `num_layers` `Dual_Computation_Block`s ->
    gated conv output -> conv1d decoder -> per-speaker reshape."""

    def __init__(
        self,
        in_channels,
        out_channels,
        intra_model,
        num_layers=1,
        norm="ln",
        K=200,
        num_spks=2,
        skip_around_intra=True,
        linear_layer_after_inter_intra=True,
        use_global_pos_enc=True,
        max_length=20000,
    ):
        super().__init__()
        self.K = K
        self.num_spks = num_spks
        self.num_layers = num_layers
        self.norm = select_norm(norm, in_channels, 3)
        self.conv1d_encoder = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.use_global_pos_enc = use_global_pos_enc

        if self.use_global_pos_enc:
            self.pos_enc = ScaledSinuEmbedding(out_channels)

        self.dual_mdl = nn.ModuleList([])
        for _i in range(num_layers):
            self.dual_mdl.append(
                copy.deepcopy(
                    Dual_Computation_Block(
                        intra_model,
                        out_channels,
                        norm,
                        skip_around_intra=skip_around_intra,
                        linear_layer_after_inter_intra=linear_layer_after_inter_intra,
                    )
                )
            )

        self.conv1d_out = nn.Conv1d(out_channels, out_channels * num_spks, kernel_size=1)
        self.conv1_decoder = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
        self.output = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1), nn.Tanh())
        self.output_gate = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid())

    def forward(self, x):
        # x: [B, N, L]
        x = self.norm(x)

        x = self.conv1d_encoder(x)

        if self.use_global_pos_enc:
            base = x
            x = x.transpose(1, -1)
            emb = self.pos_enc(x)
            emb = emb.transpose(0, -1)
            x = base + emb

        for i in range(self.num_layers):
            x = self.dual_mdl[i](x)
        x = self.prelu(x)

        x = self.conv1d_out(x)
        B, _, S = x.shape

        x = x.view(B * self.num_spks, -1, S)

        x = self.output(x) * self.output_gate(x)

        x = self.conv1_decoder(x)

        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)

        x = x.transpose(0, 1)

        return x


# ---- model/mossformer2.py (verbatim: top-level wrapper) ----


class Mossformer2Wrapper(nn.Module):
    """Wrapper combining the Encoder, masking network (`Dual_Path_Model`), and Decoder for
    monaural speech separation, matching the real `Mossformer2Wrapper`
    (`PyTorchModelHubMixin` inheritance dropped -- construction and forward are unaffected;
    only checkpoint download/upload behavior is removed)."""

    def __init__(self, config: dict):
        super().__init__()
        self.config_name = config["config_name"]

        self.encoder = Encoder(
            kernel_size=config["encoder_kernel_size"],
            out_channels=config["encoder_out_nchannels"],
            in_channels=config["encoder_in_nchannels"],
        )

        intra_model = SBFLASHBlock_DualA(
            num_layers=config["intra_numlayers"],
            d_model=config["encoder_out_nchannels"],
            nhead=config["intra_nhead"],
            d_ffn=config["intra_dffn"],
            dropout=config["intra_dropout"],
            use_positional_encoding=config["intra_use_positional"],
            norm_before=config["intra_norm_before"],
        )

        self.masknet = Dual_Path_Model(
            in_channels=config["encoder_out_nchannels"],
            out_channels=config["encoder_out_nchannels"],
            intra_model=intra_model,
            num_layers=config["masknet_numlayers"],
            norm=config["masknet_norm"],
            K=config["masknet_chunksize"],
            num_spks=config["masknet_numspks"],
            skip_around_intra=config["masknet_extraskipconnection"],
            linear_layer_after_inter_intra=config["masknet_useextralinearlayer"],
        )
        self.decoder = Decoder(
            in_channels=config["encoder_out_nchannels"],
            out_channels=config["encoder_in_nchannels"],
            kernel_size=config["encoder_kernel_size"],
            stride=config["encoder_kernel_size"] // 2,
            bias=False,
        )
        self.num_spks = config["masknet_numspks"]
        self.sample_rate = config["sample_rate"]

    def forward(self, mix):
        mix_w = self.encoder(mix)
        if self.config_name == "mossformer2-whamr-2spk":
            est_mask = self.masknet(mix_w)
            sep_h = est_mask
        else:
            est_mask = self.masknet(mix_w)
            mix_w = torch.stack([mix_w] * self.num_spks)
            sep_h = mix_w * est_mask

        est_source = torch.cat(
            [self.decoder(sep_h[i]).unsqueeze(-1) for i in range(self.num_spks)],
            dim=-1,
        )

        T_origin = mix.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]

        return est_source


# ---- tiny build/example (architecture unmodified from the real repo) ----

_TINY_CONFIG = {
    "model_type": "mossformer2",
    "sample_rate": 8000,
    "config_name": "mossformer2-librimix-2spk",
    "encoder_kernel_size": 16,
    "encoder_out_nchannels": 8,
    "encoder_in_nchannels": 1,
    "masknet_numspks": 2,
    "masknet_chunksize": 6,
    "masknet_numlayers": 1,
    "masknet_norm": "ln",
    "masknet_useextralinearlayer": False,
    "masknet_extraskipconnection": True,
    "intra_numlayers": 1,
    "intra_nhead": 8,
    "intra_dffn": 16,
    "intra_dropout": 0,
    "intra_use_positional": True,
    "intra_norm_before": True,
}


def build_mossformer2():
    """Tiny Mossformer2Wrapper (8-channel encoder, 1 masking-network layer) for tracing."""
    torch.manual_seed(0)
    model = Mossformer2Wrapper(_TINY_CONFIG)
    model.eval()
    return model


def example_input_mossformer2():
    """Matches Mossformer2Wrapper.forward: raw mono waveform [B, T]."""
    torch.manual_seed(0)
    return torch.randn(1, 160, dtype=torch.float32)


MENAGERIE_ENTRIES = [
    ("MossFormer2", "build_mossformer2", "example_input_mossformer2", 2024, MENAGERIE_ZOO),
]
