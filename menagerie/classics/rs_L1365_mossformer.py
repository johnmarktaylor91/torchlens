# SOURCE: vendored from https://github.com/modelscope/ClearerVoice-Studio @ 6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61
#   Vendored files (clearvoice/clearvoice/models/mossformer_gan_se/):
#     - mossformer.py             -> `MossFormer` (the gated single-head transformer with joint
#       quadratic + linear self-attention, the core block from "MossFormer: Pushing the
#       Performance Limit of Monaural Speech Separation using Gated Single-head Transformer
#       with Convolution-augmented Joint Self-Attentions", Zhao et al. 2023).
#     - generator.py               -> `MossFormerGAN_SE_16K` / `SyncANet` / `SyncANetBlock` /
#       dense encoder-decoder + mask/complex decoders (the GAN-based speech-enhancement
#       generator that wraps `MossFormer` blocks inside a TF-GridNet-style synchronous
#       attention network).
#     - conv_module.py, fsmn.py, se_layer.py, get_layer_from_string.py -> the exact supporting
#       nn.Module building blocks (FFConvM, ConvModule, UniDeepFsmn, SELayer, get_layer).
#
# This is the ClearerVoice-Studio (alibabasglab) "MossFormerGAN_SE_16K" checkpoint family --
# the only standalone runnable PyTorch code for the original MossFormer architecture (the
# upstream github.com/alibabasglab/MossFormer repo ships only audio samples + a ModelScope/
# SpeechBrain inference pointer, no model-definition code). Every module is reproduced
# verbatim; only relative imports were flattened to this single file and the discriminator
# (training-only, GAN adversary) is dropped since we trace the generator alone.

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch import einsum, nn
from torch.nn import init
from torch.nn.parameter import Parameter

MENAGERIE_ZOO = "vendored-pytorch"


# ---- conv_module.py (verbatim, trimmed to the classes generator.py actually uses) ----


class Swish(nn.Module):
    """Swish activation: x * sigmoid(x)."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * inputs.sigmoid()


class Transpose(nn.Module):
    """Wrapper class of torch.transpose() for Sequential module."""

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
    """Modified from Conformer convolution module: transpose -> depthwise conv1d -> residual."""

    def __init__(
        self, in_channels, kernel_size: int = 31, expansion_factor: int = 2, dropout_p: float = 0.1
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


# ---- fsmn.py (verbatim) ----


class UniDeepFsmn(nn.Module):
    def __init__(self, input_dim, output_dim, lorder=None, hidden_size=None, dropout_p=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        if lorder is None:
            return
        self.lorder = lorder
        self.rorder = lorder
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_dim, hidden_size)
        self.project = nn.Linear(hidden_size, output_dim, bias=False)
        self.conv1 = nn.Conv2d(
            input_dim,
            output_dim,
            [self.lorder + self.rorder - 1, 1],
            [1, 1],
            groups=input_dim,
            bias=False,
        )
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        self.swish = Swish()

    def forward(self, input):
        # input: batch (b) x sequence(T) x feature (h)
        f1 = self.swish(self.linear(self.norm(input)))
        p1 = self.project(f1)
        x = torch.unsqueeze(p1, 1)
        x_per = x.permute(0, 3, 2, 1)
        y = F.pad(x_per, [0, 0, self.lorder - 1, self.rorder - 1])
        out = x_per + self.conv1(y)
        out1 = out.permute(0, 3, 2, 1)
        return input + out1.squeeze()


# ---- se_layer.py (verbatim) ----


class SELayer(nn.Module):
    """Squeeze-and-Excitation layer (avg+max pool variant used by SyncANetBlock)."""

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_layer = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.max_pool_layer = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        x_avg = self.avg_pool(x).view(b, c)
        x_avg = self.avg_pool_layer(x_avg).view(b, c, 1, 1)
        x_max = self.max_pool(x).view(b, c)
        x_max = self.max_pool_layer(x_max).view(b, c, 1, 1)
        y = (x_avg + x_max) * x
        return y


# ---- get_layer_from_string.py (verbatim) ----


def get_layer(l_name, library=torch.nn):
    """Return layer object handler from library e.g. from torch.nn."""
    import difflib

    all_torch_layers = [x for x in dir(torch.nn)]
    match = [x for x in all_torch_layers if l_name.lower() == x.lower()]
    if len(match) == 0:
        close_matches = difflib.get_close_matches(l_name, [x.lower() for x in all_torch_layers])
        raise NotImplementedError(
            f"Layer with name {l_name} not found in {library}.\n Closest matches: {close_matches}"
        )
    elif len(match) > 1:
        close_matches = difflib.get_close_matches(l_name, [x.lower() for x in all_torch_layers])
        raise NotImplementedError(
            f"Multiple matchs for layer with name {l_name} not found in {library}.\n All matches: {close_matches}"
        )
    else:
        return getattr(library, match[0])


# ---- mossformer.py (verbatim: the MossFormer gated-attention block) ----


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder


class OffsetScale(nn.Module):
    def __init__(self, dim, heads=1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std=0.02)

    def forward(self, x):
        out = einsum("... d, h d -> ... h d", x, self.gamma) + self.beta
        return out.unbind(dim=-2)


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


class MossFormer(nn.Module):
    """The MossFormer gated single-head transformer: joint quadratic (chunked) +
    linear (full-sequence) self-attention with a multiplicative gate."""

    def __init__(
        self,
        dim,
        group_size=256,
        query_key_dim=128,
        expansion_factor=4.0,
        causal=False,
        dropout=0.1,
        norm_klass=nn.LayerNorm,
        shift_tokens=True,
    ):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.group_size = group_size
        self.causal = causal
        self.shift_tokens = shift_tokens

        self.rotary_pos_emb = RotaryEmbedding(dim=min(32, query_key_dim))
        self.dropout = nn.Dropout(dropout)

        self.to_hidden = FFConvM(
            dim_in=dim, dim_out=hidden_dim, norm_klass=norm_klass, dropout=dropout
        )
        self.to_qk = FFConvM(
            dim_in=dim, dim_out=query_key_dim, norm_klass=norm_klass, dropout=dropout
        )
        self.qk_offset_scale = OffsetScale(query_key_dim, heads=4)
        self.to_out = FFConvM(
            dim_in=dim * int(expansion_factor // 2),
            dim_out=dim,
            norm_klass=norm_klass,
            dropout=dropout,
        )
        self.gateActivate = nn.Sigmoid()

    def forward(self, x, *, mask=None):
        B, T, Q, C = x.size()
        x = x.view(B * T, Q, C)

        normed_x = x
        residual = x  # noqa: F841 (kept for parity with upstream; not used further, upstream artifact)

        if self.shift_tokens:
            x_shift, x_pass = normed_x.chunk(2, dim=-1)
            x_shift = F.pad(x_shift, (0, 0, 1, -1), value=0.0)
            normed_x = torch.cat((x_shift, x_pass), dim=-1)

        v, u = self.to_hidden(normed_x).chunk(2, dim=-1)
        qk = self.to_qk(normed_x)

        quad_q, lin_q, quad_k, lin_k = self.qk_offset_scale(qk)
        att_v, att_u = self.cal_attention(x, quad_q, lin_q, quad_k, lin_k, v, u, B)

        out = (att_u * v) * self.gateActivate(att_v * u)
        x = x + self.to_out(out)
        return x

    def cal_attention(self, x, quad_q, lin_q, quad_k, lin_k, v, u, B, mask=None):
        b, n, device, g = x.shape[0], x.shape[-2], x.device, self.group_size

        if exists(mask):
            lin_mask = rearrange(mask, "... -> ... 1")
            lin_k = lin_k.masked_fill(~lin_mask, 0.0)

        if exists(self.rotary_pos_emb):
            quad_q, lin_q, quad_k, lin_k = map(
                self.rotary_pos_emb.rotate_queries_or_keys, (quad_q, lin_q, quad_k, lin_k)
            )

        padding = padding_to_multiple_of(n, n)

        if padding > 0:
            quad_q, quad_k, lin_q, lin_k, v, u = map(
                lambda t: F.pad(t, (0, 0, 0, padding), value=0.0),
                (quad_q, quad_k, lin_q, lin_k, v, u),
            )
            mask = default(mask, torch.ones((b, n), device=device, dtype=torch.bool))
            mask = F.pad(mask, (0, padding), value=False)

        quad_q, quad_k, lin_q, lin_k, v, u = map(
            lambda t: rearrange(t, "b (g n) d -> b g n d", n=n),
            (quad_q, quad_k, lin_q, lin_k, v, u),
        )

        BT, K, Q, C = quad_q.size()
        quad_q_c = quad_q.view(B, -1, Q, C).transpose(2, 1)
        quad_k_c = quad_k.view(B, -1, Q, C).transpose(2, 1)
        v_c = v.view(B, -1, Q, C).transpose(2, 1)
        u_c = u.view(B, -1, Q, C).transpose(2, 1)

        if exists(mask):
            mask = rearrange(mask, "b (g j) -> b g 1 j", j=n)

        sim = einsum("... i d, ... j d -> ... i j", quad_q, quad_k) / n
        sim_c = einsum("... i d, ... j d -> ... i j", quad_q_c, quad_k_c) / quad_q_c.shape[-2]

        attn = F.relu(sim) ** 2
        attn = self.dropout(attn)

        attn_c = F.relu(sim_c) ** 2
        attn_c = self.dropout(attn_c)
        mask_c = torch.eye(quad_q_c.shape[-2], dtype=torch.bool, device=device)
        attn_c = attn_c.masked_fill(mask_c, 0.0)

        if exists(mask):
            attn = attn.masked_fill(~mask, 0.0)

        if self.causal:
            causal_mask = torch.ones((g, g), dtype=torch.bool, device=device).triu(1)
            attn = attn.masked_fill(causal_mask, 0.0)

        quad_out_v = einsum("... i j, ... j d -> ... i d", attn, v)
        quad_out_u = einsum("... i j, ... j d -> ... i d", attn, u)

        quad_out_v_c = einsum("... i j, ... j d -> ... i d", attn_c, v_c)
        quad_out_u_c = einsum("... i j, ... j d -> ... i d", attn_c, u_c)
        quad_out_v_c = quad_out_v_c.transpose(2, 1).contiguous().view(BT, K, Q, C)
        quad_out_u_c = quad_out_u_c.transpose(2, 1).contiguous().view(BT, K, Q, C)

        quad_out_v = quad_out_v + quad_out_v_c
        quad_out_u = quad_out_u + quad_out_u_c

        if self.causal:
            lin_kv = einsum("b g n d, b g n e -> b g d e", lin_k, v) / n
            lin_kv = lin_kv.cumsum(dim=1)
            lin_kv = F.pad(lin_kv, (0, 0, 0, 0, 1, -1), value=0.0)
            lin_out_v = einsum("b g d e, b g n d -> b g n e", lin_kv, lin_q)

            lin_ku = einsum("b g n d, b g n e -> b g d e", lin_k, u) / n
            lin_ku = lin_ku.cumsum(dim=1)
            lin_ku = F.pad(lin_ku, (0, 0, 0, 0, 1, -1), value=0.0)
            lin_out_u = einsum("b g d e, b g n d -> b g n e", lin_ku, lin_q)
        else:
            lin_kv = einsum("b g n d, b g n e -> b d e", lin_k, v) / n
            lin_out_v = einsum("b g n d, b d e -> b g n e", lin_q, lin_kv)

            lin_ku = einsum("b g n d, b g n e -> b d e", lin_k, u) / n
            lin_out_u = einsum("b g n d, b d e -> b g n e", lin_q, lin_ku)

        quad_attn_out_v, lin_attn_out_v = map(
            lambda t: rearrange(t, "b g n d -> b (g n) d")[:, :n], (quad_out_v, lin_out_v)
        )
        quad_attn_out_u, lin_attn_out_u = map(
            lambda t: rearrange(t, "b g n d -> b (g n) d")[:, :n], (quad_out_u, lin_out_u)
        )

        return quad_attn_out_v + lin_attn_out_v, quad_attn_out_u + lin_attn_out_u


# ---- generator.py (verbatim: DenseEncoder/Decoders + SyncANetBlock + SyncANet + top wrapper) ----


class FSMN_Wrap(nn.Module):
    def __init__(self, nIn, nHidden=128, lorder=20, nOut=128):
        super().__init__()
        self.fsmn = UniDeepFsmn(nIn, nHidden, lorder, nHidden)

    def forward(self, x):
        # x: [b, c, h, T]
        b, c, T, h = x.size()
        x = x.permute(0, 2, 3, 1)
        x = torch.reshape(x, (b * T, h, c))
        output = self.fsmn(x)
        output = torch.reshape(output, (b, T, h, c))
        return output.permute(0, 3, 1, 2)


class DilatedDenseNet(nn.Module):
    def __init__(self, depth=4, in_channels=64):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.0)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)

        for i in range(self.depth):
            dil = 2**i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, "pad{}".format(i + 1), nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0))
            setattr(
                self,
                "conv{}".format(i + 1),
                nn.Conv2d(
                    self.in_channels * (i + 1),
                    self.in_channels,
                    kernel_size=self.kernel_size,
                    dilation=(dil, 1),
                ),
            )
            setattr(self, "norm{}".format(i + 1), nn.InstanceNorm2d(in_channels, affine=True))
            setattr(self, "prelu{}".format(i + 1), nn.PReLU(self.in_channels))
            setattr(
                self,
                "fsmn{}".format(i + 1),
                FSMN_Wrap(
                    nIn=self.in_channels, nHidden=self.in_channels, lorder=5, nOut=self.in_channels
                ),
            )

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, "pad{}".format(i + 1))(skip)
            out = getattr(self, "conv{}".format(i + 1))(out)
            out = getattr(self, "norm{}".format(i + 1))(out)
            out = getattr(self, "prelu{}".format(i + 1))(out)
            out = getattr(self, "fsmn{}".format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out


class DenseEncoder(nn.Module):
    def __init__(self, in_channel, channels=64):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, (1, 1), (1, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.dilated_dense = DilatedDenseNet(depth=4, in_channels=channels)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.dilated_dense(x)
        x = self.conv_2(x)
        return x


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super().__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class MaskDecoder(nn.Module):
    def __init__(self, num_features, num_channel=64, out_channel=1):
        super().__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.conv_1 = nn.Conv2d(num_channel, out_channel, (1, 2))
        self.norm = nn.InstanceNorm2d(out_channel, affine=True)
        self.prelu = nn.PReLU(out_channel)
        self.final_conv = nn.Conv2d(out_channel, out_channel, (1, 1))
        self.prelu_out = nn.PReLU(num_features, init=-0.25)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.conv_1(x)
        x = self.prelu(self.norm(x))
        x = self.final_conv(x).permute(0, 3, 2, 1).squeeze(-1)
        return self.prelu_out(x).permute(0, 2, 1).unsqueeze(1)


class ComplexDecoder(nn.Module):
    def __init__(self, num_channel=64):
        super().__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.prelu = nn.PReLU(num_channel)
        self.norm = nn.InstanceNorm2d(num_channel, affine=True)
        self.conv = nn.Conv2d(num_channel, 2, (1, 2))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.prelu(self.norm(x))
        x = self.conv(x)
        return x


class LayerNormalization4D(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        param_size = [1, input_dimension, 1, 1]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            _, C, _, _ = x.shape
            stat_dim = (1,)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)
        std_ = torch.sqrt(x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps)
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            stat_dim = (1, 3)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)
        std_ = torch.sqrt(x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps)
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


class SyncANetBlock(nn.Module):
    """Modified TF-GridNet-style block combining gated triple-attention (via `MossFormer`)
    and FSMN modules for intra-/inter-chunk audio processing."""

    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(
        self,
        emb_dim,
        emb_ks,
        emb_hs,
        n_freqs,
        hidden_channels,
        n_head=4,
        approx_qk_dim=512,
        activation="prelu",
        eps=1e-5,
    ):
        super().__init__()
        in_channels = emb_dim * emb_ks

        self.Fconv = nn.Conv2d(
            emb_dim, in_channels, kernel_size=(1, emb_ks), stride=(1, 1), groups=emb_dim
        )
        self.intra_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.intra_to_u = FFConvM(
            dim_in=in_channels, dim_out=hidden_channels, norm_klass=nn.LayerNorm, dropout=0.1
        )
        self.intra_to_v = FFConvM(
            dim_in=in_channels, dim_out=hidden_channels, norm_klass=nn.LayerNorm, dropout=0.1
        )
        self.intra_rnn = self._build_repeats(
            in_channels, hidden_channels, 20, hidden_channels, repeats=1
        )
        self.intra_mossformer = MossFormer(dim=emb_dim, group_size=n_freqs)
        self.intra_linear = nn.ConvTranspose1d(hidden_channels, emb_dim, emb_ks, stride=emb_hs)
        self.intra_se = SELayer(channel=emb_dim, reduction=1)

        self.inter_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.inter_to_u = FFConvM(
            dim_in=in_channels, dim_out=hidden_channels, norm_klass=nn.LayerNorm, dropout=0.1
        )
        self.inter_to_v = FFConvM(
            dim_in=in_channels, dim_out=hidden_channels, norm_klass=nn.LayerNorm, dropout=0.1
        )
        self.inter_rnn = self._build_repeats(
            in_channels, hidden_channels, 20, hidden_channels, repeats=1
        )
        self.inter_mossformer = MossFormer(dim=emb_dim, group_size=256)
        self.inter_linear = nn.ConvTranspose1d(hidden_channels, emb_dim, emb_ks, stride=emb_hs)
        self.inter_se = SELayer(channel=emb_dim, reduction=1)

        E = math.ceil(approx_qk_dim * 1.0 / n_freqs)
        assert emb_dim % n_head == 0

        for ii in range(n_head):
            self.add_module(
                f"attn_conv_Q_{ii}",
                nn.Sequential(
                    nn.Conv2d(emb_dim, E, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                ),
            )
            self.add_module(
                f"attn_conv_K_{ii}",
                nn.Sequential(
                    nn.Conv2d(emb_dim, E, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                ),
            )
            self.add_module(
                f"attn_conv_V_{ii}",
                nn.Sequential(
                    nn.Conv2d(emb_dim, emb_dim // n_head, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((emb_dim // n_head, n_freqs), eps=eps),
                ),
            )

        self.add_module(
            "attn_concat_proj",
            nn.Sequential(
                nn.Conv2d(emb_dim, emb_dim, 1),
                get_layer(activation)(),
                LayerNormalization4DCF((emb_dim, n_freqs), eps=eps),
            ),
        )

        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

    def _build_repeats(self, in_channels, out_channels, lorder, hidden_size, repeats=1):
        repeats = [
            UniDeepFsmn(in_channels, out_channels, lorder, hidden_size) for _ in range(repeats)
        ]
        return nn.Sequential(*repeats)

    def forward(self, x):
        B, C, old_T, old_Q = x.shape

        T = math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        Q = math.ceil((old_Q - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks

        x = F.pad(x, (0, Q - old_Q, 0, T - old_T))

        # Intra-process
        input_ = x
        intra_rnn = self.intra_norm(input_)
        intra_rnn = self.Fconv(intra_rnn)
        intra_rnn = intra_rnn.transpose(1, 2).contiguous().view(B * T, C * self.emb_ks, -1)

        intra_rnn = intra_rnn.transpose(1, 2)
        intra_rnn_u = self.intra_to_u(intra_rnn)
        intra_rnn_v = self.intra_to_v(intra_rnn)
        intra_rnn_u = self.intra_rnn(intra_rnn_u)
        intra_rnn = intra_rnn_v * intra_rnn_u
        intra_rnn = intra_rnn.transpose(1, 2)
        intra_rnn = self.intra_linear(intra_rnn)
        intra_rnn = intra_rnn.transpose(1, 2)
        intra_rnn = intra_rnn.view([B, T, Q, C])
        intra_rnn = self.intra_mossformer(intra_rnn)
        intra_rnn = intra_rnn.transpose(1, 2)
        intra_rnn = intra_rnn.view([B, T, C, Q])
        intra_rnn = intra_rnn.transpose(1, 2).contiguous()
        intra_rnn = self.intra_se(intra_rnn)
        intra_rnn = intra_rnn + input_

        # Inter-process
        input_ = intra_rnn
        inter_rnn = self.inter_norm(input_)
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous().view(B * Q, C, T)
        inter_rnn = F.unfold(inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1))
        inter_rnn = inter_rnn.transpose(1, 2)
        inter_rnn_u = self.inter_to_u(inter_rnn)
        inter_rnn_v = self.inter_to_v(inter_rnn)
        inter_rnn_u = self.inter_rnn(inter_rnn_u)
        inter_rnn = inter_rnn_v * inter_rnn_u
        inter_rnn = inter_rnn.transpose(1, 2)
        inter_rnn = self.inter_linear(inter_rnn)
        inter_rnn = inter_rnn.transpose(1, 2)
        inter_rnn = inter_rnn.view([B, Q, T, C])
        inter_rnn = self.inter_mossformer(inter_rnn)
        inter_rnn = inter_rnn.transpose(1, 2)
        inter_rnn = inter_rnn.view([B, Q, C, T])
        inter_rnn = inter_rnn.permute(0, 2, 3, 1).contiguous()
        inter_rnn = self.inter_se(inter_rnn)
        inter_rnn = inter_rnn + input_

        inter_rnn = inter_rnn[..., :old_T, :old_Q]

        batch = inter_rnn
        all_Q, all_K, all_V = [], [], []

        for ii in range(self.n_head):
            all_Q.append(self["attn_conv_Q_%d" % ii](batch))
            all_K.append(self["attn_conv_K_%d" % ii](batch))
            all_V.append(self["attn_conv_V_%d" % ii](batch))

        Qc = torch.cat(all_Q, dim=0)
        Kc = torch.cat(all_K, dim=0)
        Vc = torch.cat(all_V, dim=0)

        Qc = Qc.transpose(1, 2)
        Qc = Qc.flatten(start_dim=2)
        Kc = Kc.transpose(1, 2)
        Kc = Kc.flatten(start_dim=2)
        Vc = Vc.transpose(1, 2)
        old_shape = Vc.shape
        Vc = Vc.flatten(start_dim=2)
        emb_dim = Qc.shape[-1]

        attn_mat = torch.matmul(Qc, Kc.transpose(1, 2)) / (emb_dim**0.5)
        attn_mat = F.softmax(attn_mat, dim=2)
        Vc = torch.matmul(attn_mat, Vc)

        Vc = Vc.reshape(old_shape)
        Vc = Vc.transpose(1, 2)
        emb_dim = Vc.shape[1]

        batch = Vc.view([self.n_head, B, emb_dim, old_T, -1])
        batch = batch.transpose(0, 1)
        batch = batch.contiguous().view([B, self.n_head * emb_dim, old_T, -1])
        batch = self["attn_concat_proj"](batch)

        out = batch + inter_rnn
        return out


class SyncANet(nn.Module):
    """Synchronous audio processing network: dense encoder + 6 SyncANetBlocks + mask/complex
    decoders, matching the real MossFormerGAN_SE_16K generator."""

    def __init__(self, num_channel=64, num_features=201):
        super().__init__()
        self.dense_encoder = DenseEncoder(in_channel=3, channels=num_channel)
        self.n_layers = 6
        self.blocks = nn.ModuleList([])

        for _ in range(self.n_layers):
            self.blocks.append(
                SyncANetBlock(
                    emb_dim=num_channel,
                    emb_ks=2,
                    emb_hs=1,
                    n_freqs=int(num_features // 2) + 1,
                    hidden_channels=num_channel * 2,
                    n_head=4,
                    approx_qk_dim=512,
                    activation="prelu",
                    eps=1.0e-5,
                )
            )

        self.mask_decoder = MaskDecoder(num_features, num_channel=num_channel, out_channel=1)
        self.complex_decoder = ComplexDecoder(num_channel=num_channel)

    def forward(self, x):
        out_list = []
        mag = torch.sqrt(x[:, 0, :, :] ** 2 + x[:, 1, :, :] ** 2).unsqueeze(1)
        noisy_phase = torch.angle(torch.complex(x[:, 0, :, :], x[:, 1, :, :])).unsqueeze(1)
        x_in = torch.cat([mag, x], dim=1)

        x = self.dense_encoder(x_in)
        for ii in range(self.n_layers):
            x = self.blocks[ii](x)

        mask = self.mask_decoder(x)
        out_mag = mask * mag

        complex_out = self.complex_decoder(x)
        mag_real = out_mag * torch.cos(noisy_phase)
        mag_imag = out_mag * torch.sin(noisy_phase)
        final_real = mag_real + complex_out[:, 0, :, :].unsqueeze(1)
        final_imag = mag_imag + complex_out[:, 1, :, :].unsqueeze(1)
        out_list.append(final_real)
        out_list.append(final_imag)

        return out_list


class MossFormerGAN_SE_16K(nn.Module):
    """MossFormerGAN_SE_16K: GAN-based speech-enhancement model for 16kHz audio wrapping
    SyncANet (the discriminator is training-only and omitted at inference)."""

    def __init__(self, fft_len=400):
        super().__init__()
        self.model = SyncANet(num_channel=64, num_features=fft_len // 2 + 1)

    def forward(self, x):
        output_real, output_imag = self.model(x)
        return output_real, output_imag


# ---- tiny build/example (architecture unmodified from the real repo) ----


def build_mossformer():
    """Tiny MossFormerGAN_SE_16K (fft_len=64 -> num_features=33) for tracing."""
    torch.manual_seed(0)
    model = MossFormerGAN_SE_16K(fft_len=64)
    model.eval()
    return model


def example_input_mossformer():
    """Matches SyncANet.forward: complex STFT input [B, 2, T, F] (real/imag channels)."""
    torch.manual_seed(0)
    return torch.randn(1, 2, 8, 33, dtype=torch.float32)


MENAGERIE_ENTRIES = [
    ("MossFormer (GAN-SE)", "build_mossformer", "example_input_mossformer", 2023, MENAGERIE_ZOO),
]
