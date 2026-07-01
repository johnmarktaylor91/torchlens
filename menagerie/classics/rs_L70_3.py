# SOURCE: vendored from piddnad/DDColor @ master
# (files basicsr/archs/ddcolor_arch.py, basicsr/archs/ddcolor_arch_utils/unet.py,
#  basicsr/archs/ddcolor_arch_utils/convnext.py,
#  basicsr/archs/ddcolor_arch_utils/transformer_utils.py,
#  basicsr/archs/ddcolor_arch_utils/position_encoding.py,
#  basicsr/archs/ddcolor_arch_utils/transformer.py)
#
# DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders, ICCV 2023.
# https://github.com/piddnad/DDColor
#
# This vendors the real DDColor architecture: a ConvNeXt encoder (with forward
# hooks tapping intermediate stage outputs), a "quasi-UNet" pixel-shuffle
# decoder (UnetBlockWide/CustomPixelShuffle_ICNR), and the multi-scale
# transformer color-query decoder (MultiScaleColorDecoder: DETR-style
# self-/cross-attention over learned color queries against multi-scale
# encoder/decoder features), finishing with a spectral-norm refine conv. No
# architectural changes -- only the `@ARCH_REGISTRY.register()` decorator (a
# basicsr bookkeeping utility, not part of the architecture) and the
# `basicsr.utils.registry` / `basicsr.utils.get_root_logger` package imports
# were dropped, since they are unrelated to the forward-pass computation
# graph (registry bookkeeping and checkpoint-loading logging respectively).

from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# basicsr/archs/ddcolor_arch_utils/convnext.py
# ---------------------------------------------------------------------------

try:
    from timm.layers import DropPath, trunc_normal_  # type: ignore
except Exception:
    import math

    def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
        if hasattr(torch.nn.init, "trunc_normal_"):
            return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

        def norm_cdf(x):
            return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

        with torch.no_grad():
            lo = norm_cdf((a - mean) / std)
            u = norm_cdf((b - mean) / std)

            tensor.uniform_(2 * lo - 1, 2 * u - 1)
            tensor.erfinv_()

            tensor.mul_(std * math.sqrt(2.0))
            tensor.add_(mean)
            tensor.clamp_(min=a, max=b)
            return tensor

    class DropPath(nn.Module):
        def __init__(self, drop_prob: float = 0.0):
            super().__init__()
            self.drop_prob = float(drop_prob)

        def forward(self, x):
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep_prob = 1.0 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            return x.div(keep_prob) * random_tensor


class ConvNeXtBlock(nn.Module):
    r"""ConvNeXt Block (DwConv -> Permute -> LayerNorm(channels_last) -> Linear -> GELU -> Linear -> Permute back)."""

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = ConvNeXtLayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r"""ConvNeXt encoder, `A ConvNet for the 2020s` (https://arxiv.org/pdf/2201.03545.pdf)."""

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
    ):
        super().__init__()
        depths = list(depths)
        dims = list(dims)

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            ConvNeXtLayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                ConvNeXtLayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    ConvNeXtBlock(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        out_indices = (0, 1, 2, 3)
        for i in out_indices:
            layer = ConvNeXtLayerNorm(dims[i], eps=1e-6, data_format="channels_first")
            layer_name = f"norm{i}"
            self.add_module(layer_name, layer)

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

            norm_layer = getattr(self, f"norm{i}")
            norm_layer(x)

        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        x = self.forward_features(x)
        return x


class ConvNeXtLayerNorm(nn.Module):
    r"""LayerNorm supporting channels_last (default) or channels_first."""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# ---------------------------------------------------------------------------
# basicsr/archs/ddcolor_arch_utils/unet.py
# ---------------------------------------------------------------------------

NormType = Enum("NormType", "Batch BatchZero Weight Spectral")


class Hook:
    def __init__(self, module):
        self.feature = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        if isinstance(output, torch.Tensor):
            self.feature = output
        elif isinstance(output, dict):
            self.feature = output["out"]

    def remove(self):
        self.hook.remove()


class SelfAttentionUnet(nn.Module):
    "Self attention layer for nd (unet.py `SelfAttention`, renamed to avoid a name clash with the transformer's self-attention layer)."

    def __init__(self, n_channels: int):
        super().__init__()
        self.query = conv1d(n_channels, n_channels // 8)
        self.key = conv1d(n_channels, n_channels // 8)
        self.value = conv1d(n_channels, n_channels)
        self.gamma = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        size = x.size()
        x = x.view(*size[:2], -1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f.permute(0, 2, 1).contiguous(), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()


def batchnorm_2d(nf: int, norm_type: NormType = NormType.Batch):
    bn = nn.BatchNorm2d(nf)
    with torch.no_grad():
        bn.bias.fill_(1e-3)
        bn.weight.fill_(0.0 if norm_type == NormType.BatchZero else 1.0)
    return bn


def init_default(m: nn.Module, func=nn.init.kaiming_normal_) -> None:
    if func:
        if hasattr(m, "weight"):
            func(m.weight)
        if hasattr(m, "bias") and hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    return m


def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    ni, nf, h, w = x.shape
    ni2 = int(ni / (scale**2))
    k = init(torch.zeros([ni2, nf, h, w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    k = k.contiguous().view([nf, ni, h, w]).transpose(0, 1)
    x.data.copy_(k)


def conv1d(ni: int, no: int, ks: int = 1, stride: int = 1, padding: int = 0, bias: bool = False):
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias:
        conv.bias.data.zero_()
    return nn.utils.spectral_norm(conv)


def custom_conv_layer(
    ni: int,
    nf: int,
    ks: int = 3,
    stride: int = 1,
    padding=None,
    bias=None,
    is_1d: bool = False,
    norm_type=NormType.Batch,
    use_activ: bool = True,
    transpose: bool = False,
    init=nn.init.kaiming_normal_,
    self_attention: bool = False,
    extra_bn: bool = False,
):
    if padding is None:
        padding = (ks - 1) // 2 if not transpose else 0
    bn = norm_type in (NormType.Batch, NormType.BatchZero) or extra_bn is True
    if bias is None:
        bias = not bn
    conv_func = nn.ConvTranspose2d if transpose else nn.Conv1d if is_1d else nn.Conv2d
    conv = init_default(
        conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding), init
    )

    if norm_type == NormType.Weight:
        conv = nn.utils.weight_norm(conv)
    elif norm_type == NormType.Spectral:
        conv = nn.utils.spectral_norm(conv)
    layers = [conv]
    if use_activ:
        layers.append(nn.ReLU(True))
    if bn:
        layers.append((nn.BatchNorm1d if is_1d else nn.BatchNorm2d)(nf))
    if self_attention:
        layers.append(SelfAttentionUnet(nf))
    return nn.Sequential(*layers)


class CustomPixelShuffle_ICNR(nn.Module):
    "Upsample by `scale` using `nn.PixelShuffle`, `icnr` init, and `weight_norm`."

    def __init__(
        self,
        ni: int,
        nf: int = None,
        scale: int = 2,
        blur: bool = True,
        norm_type=NormType.Spectral,
        extra_bn=False,
    ):
        super().__init__()
        self.conv = custom_conv_layer(
            ni, nf * (scale**2), ks=1, use_activ=False, norm_type=norm_type, extra_bn=extra_bn
        )
        icnr(self.conv[0].weight)
        self.shuf = nn.PixelShuffle(scale)
        self.do_blur = blur
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x)) if self.do_blur else x


class UnetBlockWide(nn.Module):
    "A quasi-UNet block, using PixelShuffle_ICNR upsampling."

    def __init__(
        self,
        up_in_c: int,
        x_in_c: int,
        n_out: int,
        hook,
        blur: bool = False,
        self_attention: bool = False,
        norm_type=NormType.Spectral,
    ):
        super().__init__()

        self.hook = hook
        up_out = n_out
        self.shuf = CustomPixelShuffle_ICNR(
            up_in_c, up_out, blur=blur, norm_type=norm_type, extra_bn=True
        )
        self.bn = batchnorm_2d(x_in_c)
        ni = up_out + x_in_c
        self.conv = custom_conv_layer(
            ni, n_out, norm_type=norm_type, self_attention=self_attention, extra_bn=True
        )
        self.relu = nn.ReLU()

    def forward(self, up_in):
        s = self.hook.feature
        up_out = self.shuf(up_in)
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv(cat_x)


# ---------------------------------------------------------------------------
# basicsr/archs/ddcolor_arch_utils/position_encoding.py
# ---------------------------------------------------------------------------


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * torch.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# ---------------------------------------------------------------------------
# basicsr/archs/ddcolor_arch_utils/transformer_utils.py
# ---------------------------------------------------------------------------


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None
    ):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None
    ):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos
            )
        return self.forward_post(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):
    def __init__(
        self, d_model, dim_feedforward=2048, dropout=0.0, activation="relu", normalize_before=False
    ):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# ---------------------------------------------------------------------------
# basicsr/archs/ddcolor_arch.py -- Encoder / Decoder / MultiScaleColorDecoder /
# DDColor (only the default `decoder_name='MultiScaleColorDecoder'` path is
# vendored; `SingleColorDecoder` is the paper's ablation alternative and is
# not part of the default forward path, so it is omitted here).
# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    def __init__(self, encoder_name, hook_names, from_pretrain=False, **kwargs):
        super().__init__()

        if encoder_name == "convnext-t" or encoder_name == "convnext":
            self.arch = ConvNeXt()
        elif encoder_name == "convnext-s":
            self.arch = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768])
        elif encoder_name == "convnext-b":
            self.arch = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
        elif encoder_name == "convnext-l":
            self.arch = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])
        else:
            raise NotImplementedError

        self.encoder_name = encoder_name
        self.hook_names = hook_names
        self.hooks = self.setup_hooks()

    def setup_hooks(self):
        hooks = [Hook(self.arch._modules[name]) for name in self.hook_names]
        return hooks

    def forward(self, x):
        return self.arch(x)


class MultiScaleColorDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dim=256,
        num_queries=100,
        nheads=8,
        dim_feedforward=2048,
        dec_layers=9,
        pre_norm=False,
        color_embed_dim=256,
        enforce_input_project=True,
        num_scales=3,
    ):
        super().__init__()

        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm
                )
            )
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm
                )
            )
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.num_feature_levels = num_scales
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

        self.input_proj = nn.ModuleList()
        for i in range(self.num_feature_levels):
            if in_channels[i] != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels[i], hidden_dim, kernel_size=1))
                nn.init.kaiming_uniform_(self.input_proj[-1].weight, a=1)
                if self.input_proj[-1].bias is not None:
                    nn.init.constant_(self.input_proj[-1].bias, 0)
            else:
                self.input_proj.append(nn.Sequential())

        self.color_embed = MLP(hidden_dim, hidden_dim, color_embed_dim, 3)

    def forward(self, x, img_features):
        assert len(x) == self.num_feature_levels
        src = []
        pos = []

        for i in range(self.num_feature_levels):
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(
                self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None]
            )

            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            output = self.transformer_cross_attention_layers[i](
                output,
                src[level_index],
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=pos[level_index],
                query_pos=query_embed,
            )
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_embed
            )
            output = self.transformer_ffn_layers[i](output)

        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        color_embed = self.color_embed(decoder_output)
        out = torch.einsum("bqc,bchw->bqhw", color_embed, img_features)

        return out


class Decoder(nn.Module):
    def __init__(
        self,
        hooks,
        nf=512,
        blur=True,
        last_norm="Weight",
        num_queries=256,
        num_scales=3,
        dec_layers=9,
        decoder_name="MultiScaleColorDecoder",
    ):
        super().__init__()
        self.hooks = hooks
        self.nf = nf
        self.blur = blur
        self.last_norm = getattr(NormType, last_norm)
        self.decoder_name = decoder_name

        self.layers = self.make_layers()
        embed_dim = nf // 2

        self.last_shuf = CustomPixelShuffle_ICNR(
            embed_dim, embed_dim, blur=self.blur, norm_type=self.last_norm, scale=4
        )

        self.color_decoder = MultiScaleColorDecoder(
            in_channels=[512, 512, 256],
            num_queries=num_queries,
            num_scales=num_scales,
            dec_layers=dec_layers,
        )

    def forward(self):
        encode_feat = self.hooks[-1].feature
        out0 = self.layers[0](encode_feat)
        out1 = self.layers[1](out0)
        out2 = self.layers[2](out1)
        out3 = self.last_shuf(out2)

        out = self.color_decoder([out0, out1, out2], out3)

        return out

    def make_layers(self):
        decoder_layers = []

        e_in_c = self.hooks[-1].feature.shape[1]
        in_c = e_in_c

        out_c = self.nf
        setup_hooks = self.hooks[-2::-1]
        for layer_index, hook in enumerate(setup_hooks):
            feature_c = hook.feature.shape[1]
            if layer_index == len(setup_hooks) - 1:
                out_c = out_c // 2
            decoder_layers.append(
                UnetBlockWide(
                    in_c,
                    feature_c,
                    out_c,
                    hook,
                    blur=self.blur,
                    self_attention=False,
                    norm_type=NormType.Spectral,
                )
            )
            in_c = out_c
        return nn.Sequential(*decoder_layers)


class DDColor(nn.Module):
    def __init__(
        self,
        encoder_name="convnext-l",
        decoder_name="MultiScaleColorDecoder",
        num_input_channels=3,
        input_size=(256, 256),
        nf=512,
        num_output_channels=3,
        last_norm="Weight",
        do_normalize=False,
        num_queries=256,
        num_scales=3,
        dec_layers=9,
        encoder_from_pretrain=False,
    ):
        super().__init__()

        self.encoder = Encoder(
            encoder_name, ["norm0", "norm1", "norm2", "norm3"], from_pretrain=encoder_from_pretrain
        )
        self.encoder.eval()
        test_input = torch.randn(1, num_input_channels, *input_size)

        with torch.no_grad():
            self.encoder(test_input)

        self.decoder = Decoder(
            self.encoder.hooks,
            nf=nf,
            last_norm=last_norm,
            num_queries=num_queries,
            num_scales=num_scales,
            dec_layers=dec_layers,
            decoder_name=decoder_name,
        )
        self.refine_net = nn.Sequential(
            custom_conv_layer(
                num_queries + 3,
                num_output_channels,
                ks=1,
                use_activ=False,
                norm_type=NormType.Spectral,
            )
        )

        self.do_normalize = do_normalize
        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, img):
        return (img - self.mean) / self.std

    def denormalize(self, img):
        return img * self.std + self.mean

    def forward(self, x):
        if x.shape[1] == 3:
            x = self.normalize(x)

        self.encoder(x)
        out_feat = self.decoder()
        coarse_input = torch.cat([out_feat, x], dim=1)
        out = self.refine_net(coarse_input)

        if self.do_normalize:
            out = self.denormalize(out)
        return out


# ---------------------------------------------------------------------------
# menagerie build/example helpers
# ---------------------------------------------------------------------------


def build_ddcolor():
    # Paper default is 'convnext-l' (depths [3,3,27,3], dims [192,384,768,1536])
    # at 256x256/nf=512/num_queries=256/dec_layers=9 -- far too large for the
    # menagerie. `Decoder.make_layers()` and `MultiScaleColorDecoder(in_channels=
    # [512, 512, 256])` hardcode those channel widths as a function of `nf=512`
    # (upstream `Decoder` never threads `nf` into that constant), so `nf` must
    # stay 512 for the decoder wiring to line up -- we shrink cost instead via
    # 'convnext-t'-scale encoder depths/dims, a smaller num_queries/dec_layers,
    # and a smaller input_size, keeping every module type in the real
    # architecture (ConvNeXt encoder w/ hooks, UnetBlockWide pixel-shuffle
    # decoder, MultiScaleColorDecoder transformer) exercised.
    model = DDColor(
        encoder_name="convnext-t",
        decoder_name="MultiScaleColorDecoder",
        num_input_channels=3,
        input_size=(64, 64),
        nf=512,
        num_output_channels=3,
        last_norm="Weight",
        do_normalize=False,
        num_queries=8,
        num_scales=3,
        dec_layers=2,
        encoder_from_pretrain=False,
    )
    return model


def example_input_ddcolor():
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("DDColor", "build_ddcolor", "example_input_ddcolor", 2023, MENAGERIE_ZOO),
]
