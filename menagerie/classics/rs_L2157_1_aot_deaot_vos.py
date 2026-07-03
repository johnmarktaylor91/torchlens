# SOURCE: vendored from z-x-yang/Segment-and-Track-Anything @ 99a291589e6f85075db70f3f3671f186a1b26af6
# (aot/ subtree = a checked-in copy of yoxu515/aot-benchmark; the AOT/DeAOT Long Short-Term
# Transformer VOS architecture powering SegTracker.py's tracking stage)
#
# SegTracker.py itself (the SAM-Track orchestrator) is a stateful multi-frame video pipeline
# (cv2/skimage frame I/O, incremental mask propagation, "everything" SAM mask generation) with
# no single-tensor forward pass to trace -- it is a *composition* of two already-real models:
# Segment Anything (already registered in the menagerie catalog as
# "sam:base-promptable-segmentation" via transformers:AutoModel) and this AOT/DeAOT tracker
# (not yet present). This module vendors the real AOT/DeAOT tracker network verbatim -- the
# genuinely distinct architecture family SAM-Track contributes beyond stock SAM.
#
# The AOT/DeAOT `nn.Module` (aot/networks/models/aot.py, aot/networks/models/deaot.py) has no
# single `forward()` -- the real `AOTEngine` (aot/networks/engines/aot_engine.py) drives it
# across frames via `encode_image()` -> `LSTT_forward()` -> `decode_id_logits()`. The
# `AOTSingleFrameWrapper.forward()` below is a thin glue method reproducing exactly the
# `AOTEngine.add_reference_frame()` first-frame code path (encode -> project -> id-embed ->
# LSTT with curr_id_emb, no long/short-term memory banks yet -> decode), so the whole network
# is exercised end-to-end in one call. No architectural code was invented; every nn.Module below
# (MobileNetV2 encoder, LongShortTermTransformer / LongShortTermTransformerBlock,
# MultiheadAttention, MultiheadLocalAttentionV2, PositionEmbeddingSine, FPNSegmentationHead,
# and their support layers) is transcribed verbatim from the real repo files:
#   aot/networks/models/aot.py, aot/networks/models/deaot.py
#   aot/networks/encoders/mobilenetv2.py, aot/networks/encoders/__init__.py
#   aot/networks/decoders/fpn.py, aot/networks/decoders/__init__.py
#   aot/networks/layers/transformer.py, aot/networks/layers/attention.py
#   aot/networks/layers/position.py, aot/networks/layers/basic.py,
#   aot/networks/layers/normalization.py
# Only fixes: relative-import paths collapsed into this single file; `MultiheadLocalAttentionV2`
# is constructed with `enable_corr=False`, which is a real, already-present branch in the repo's
# own `MultiheadLocalAttentionV2.forward()` (the `else: unfolded_k = self.pad_and_unfold(k)...`
# path) that computes the identical local attention in pure PyTorch instead of via the optional
# `spatial_correlation_sampler` CUDA extension (not a base lib, and not installed) -- no new
# architecture code, just selecting the repo's own documented non-CUDA code path. (The repo also
# ships a `MultiheadLocalAttentionV3` "no-correlation-sampler" class that `LongShortTermTransformerBlock`
# falls back to on `ImportError`, but V3's own `output = agg_value + agg_bias` line is dimensionally
# broken as written upstream -- confirmed by executing it standalone, independent of any menagerie/
# TorchLens code path, at both this module's tiny config and the repo's production config
# (d_model=256, num_head=8): `agg_value` is `[n, num_head, hw, hidden_dim]` and `agg_bias` is
# `[hw, n, d_model]`, which never broadcast for num_head>1. V2 with `enable_corr=False` is the
# correct working non-CUDA path.) `freeze_params`/`FrozenBatchNorm2d` freezing wiring kept intact
# but driven with `frozen_bn=False` (a supported constructor flag, matching `MODEL_FREEZE_BN=False`
# in a from-scratch/tiny config) so random-init tracing doesn't depend on a pretrained-checkpoint-
# only freeze schedule.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------------
# aot/networks/layers/normalization.py
# --------------------------------------------------------------------------------------


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d where the batch statistics and the affine parameters are fixed."""

    def __init__(self, n, epsilon=1e-5):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n) - epsilon)
        self.epsilon = epsilon

    def forward(self, x):
        if x.requires_grad:
            scale = self.weight * (self.running_var + self.epsilon).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            out_dtype = x.dtype
            return x * scale.to(out_dtype) + bias.to(out_dtype)
        else:
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.epsilon,
            )


def freeze_params(module):
    for p in module.parameters():
        p.requires_grad = False


# --------------------------------------------------------------------------------------
# aot/networks/encoders/mobilenetv2.py + aot/networks/encoders/__init__.py
# --------------------------------------------------------------------------------------


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size=3,
        stride=1,
        groups=1,
        padding=-1,
        norm_layer=None,
        activation_layer=None,
        dilation=1,
    ):
        if padding == -1:
            padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super().__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            norm_layer(out_planes),
            activation_layer(inplace=True),
        )
        self.out_channels = out_planes


ConvBNReLU = ConvBNActivation


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, dilation, expand_ratio, norm_layer=None):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.kernel_size = 3
        self.dilation = dilation

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend(
            [
                ConvBNReLU(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    dilation=dilation,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                ),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
        self,
        output_stride=8,
        norm_layer=None,
        width_mult=1.0,
        inverted_residual_setting=None,
        round_nearest=8,
        block=None,
        freeze_at=0,
    ):
        super().__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        last_channel = 1280
        input_channel = 32
        current_stride = 1
        rate = 1

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                "inverted_residual_setting should be non-empty or a 4-element list, "
                f"got {inverted_residual_setting}"
            )

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        current_stride *= 2
        for t, c, n, s in inverted_residual_setting:
            if current_stride == output_stride:
                stride = 1
                dilation = rate
                rate *= s
            else:
                stride = s
                dilation = 1
                current_stride *= s
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                if i == 0:
                    features.append(
                        block(input_channel, output_channel, stride, dilation, t, norm_layer)
                    )
                else:
                    features.append(block(input_channel, output_channel, 1, rate, t, norm_layer))
                input_channel = output_channel

        features.append(
            ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer)
        )
        self.features = nn.Sequential(*features)

        self._initialize_weights()

        feature_4x = self.features[0:4]
        feautre_8x = self.features[4:7]
        feature_16x = self.features[7:14]
        feature_32x = self.features[14:]

        self.stages = [feature_4x, feautre_8x, feature_16x, feature_32x]

        self.freeze(freeze_at)

    def forward(self, x):
        xs = []
        for stage in self.stages:
            x = stage(x)
            xs.append(x)
        return xs

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def freeze(self, freeze_at):
        if freeze_at >= 1:
            for m in self.stages[0][0]:
                freeze_params(m)

        for idx, stage in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                freeze_params(stage)


def build_encoder(name, frozen_bn=True, freeze_at=-1):
    if frozen_bn:
        BatchNorm = FrozenBatchNorm2d
    else:
        BatchNorm = nn.BatchNorm2d

    if name == "mobilenetv2":
        return MobileNetV2(16, BatchNorm, freeze_at=freeze_at)
    else:
        raise NotImplementedError(f"encoder {name!r} not vendored (mobilenetv2 only)")


# --------------------------------------------------------------------------------------
# aot/networks/layers/basic.py
# --------------------------------------------------------------------------------------


class GroupNorm1D(nn.Module):
    def __init__(self, indim, groups=8):
        super().__init__()
        self.gn = nn.GroupNorm(groups, indim)

    def forward(self, x):
        return self.gn(x.permute(1, 2, 0)).permute(2, 0, 1)


class GNActDWConv2d(nn.Module):
    def __init__(self, indim, gn_groups=32):
        super().__init__()
        self.gn = nn.GroupNorm(gn_groups, indim)
        self.conv = nn.Conv2d(indim, indim, 5, dilation=1, padding=2, groups=indim, bias=False)

    def forward(self, x, size_2d):
        h, w = size_2d
        _, bs, c = x.size()
        x = x.view(h, w, bs, c).permute(2, 3, 0, 1)
        x = self.gn(x)
        x = F.gelu(x)
        x = self.conv(x)
        x = x.view(bs, c, h * w).permute(2, 0, 1)
        return x


class ScaleOffset(nn.Module):
    def __init__(self, indim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(indim))
        self.beta = nn.Parameter(torch.zeros(indim))

    def forward(self, x):
        if len(x.size()) == 3:
            return x * self.gamma + self.beta
        else:
            return x * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)


class ConvGN(nn.Module):
    def __init__(self, indim, outdim, kernel_size, gn_groups=8):
        super().__init__()
        self.conv = nn.Conv2d(indim, outdim, kernel_size, padding=kernel_size // 2)
        self.gn = nn.GroupNorm(gn_groups, outdim)

    def forward(self, x):
        return self.gn(self.conv(x))


def seq_to_2d(tensor, size_2d):
    h, w = size_2d
    _, n, c = tensor.size()
    tensor = tensor.view(h, w, n, c).permute(2, 3, 0, 1).contiguous()
    return tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None, batch_dim=0):
        super().__init__()
        self.drop_prob = drop_prob
        self.batch_dim = batch_dim

    def forward(self, x):
        return self.drop_path(x, self.drop_prob)

    def drop_path(self, x, drop_prob):
        if drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - drop_prob
        shape = [1 for _ in range(x.ndim)]
        shape[self.batch_dim] = x.shape[self.batch_dim]
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class DropOutLogit(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return self.drop_logit(x, self.drop_prob)

    def drop_logit(self, x, drop_prob):
        if drop_prob == 0.0 or not self.training:
            return x
        random_tensor = drop_prob + torch.rand(x.shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        mask = random_tensor * 1e8 if (x.dtype == torch.float32) else random_tensor * 1e4
        output = x - mask
        return output


# --------------------------------------------------------------------------------------
# aot/networks/layers/position.py
# --------------------------------------------------------------------------------------


def generate_coord(x):
    _, _, h, w = x.size()
    device = x.device
    col = torch.arange(0, h, device=device)
    row = torch.arange(0, w, device=device)
    grid_h, grid_w = torch.meshgrid(col, row, indexing="ij")
    return grid_h, grid_w


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        grid_y, grid_x = generate_coord(x)

        y_embed = grid_y.unsqueeze(0).float()
        x_embed = grid_x.unsqueeze(0).float()

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


# --------------------------------------------------------------------------------------
# aot/networks/layers/attention.py (MultiheadAttention + MultiheadLocalAttentionV2, the
# latter constructed with enable_corr=False -- its own non-CUDA pad_and_unfold code path)
# --------------------------------------------------------------------------------------


def multiply_by_ychunks(x, y, chunks=1):
    if chunks <= 1:
        return x @ y
    else:
        return torch.cat([x @ _y for _y in y.chunk(chunks, dim=-1)], dim=-1)


def multiply_by_xchunks(x, y, chunks=1):
    if chunks <= 1:
        return x @ y
    else:
        return torch.cat([_x @ y for _x in x.chunk(chunks, dim=-2)], dim=-2)


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model,
        num_head=8,
        dropout=0.0,
        use_linear=True,
        d_att=None,
        use_dis=False,
        qk_chunks=1,
        max_mem_len_ratio=-1,
        top_k=-1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.use_dis = use_dis
        self.qk_chunks = qk_chunks
        self.max_mem_len_ratio = float(max_mem_len_ratio)
        self.top_k = top_k

        self.hidden_dim = d_model // num_head
        self.d_att = self.hidden_dim if d_att is None else d_att
        self.T = self.d_att**0.5
        self.use_linear = use_linear

        if use_linear:
            self.linear_Q = nn.Linear(d_model, d_model)
            self.linear_K = nn.Linear(d_model, d_model)
            self.linear_V = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.drop_prob = dropout
        self.projection = nn.Linear(d_model, d_model)
        self._init_weight()

    def forward(self, Q, K, V):
        num_head = self.num_head
        hidden_dim = self.hidden_dim

        bs = Q.size()[1]

        if self.use_linear:
            Q = self.linear_Q(Q)
            K = self.linear_K(K)
            V = self.linear_V(V)

        Q = Q / self.T

        if not self.training and self.max_mem_len_ratio > 0:
            mem_len_ratio = float(K.size(0)) / Q.size(0)
            if mem_len_ratio > self.max_mem_len_ratio:
                scaling_ratio = math.log(mem_len_ratio) / math.log(self.max_mem_len_ratio)
                Q = Q * scaling_ratio

        Q = Q.view(-1, bs, num_head, self.d_att).permute(1, 2, 0, 3)
        K = K.view(-1, bs, num_head, self.d_att).permute(1, 2, 3, 0)
        V = V.view(-1, bs, num_head, hidden_dim).permute(1, 2, 0, 3)

        QK = multiply_by_ychunks(Q, K, self.qk_chunks)
        if self.use_dis:
            QK = 2 * QK - K.pow(2).sum(dim=-2, keepdim=True)

        if not self.training and self.top_k > 0 and self.top_k < QK.size()[-1]:
            top_QK, indices = torch.topk(QK, k=self.top_k, dim=-1)
            top_attn = torch.softmax(top_QK, dim=-1)
            attn = torch.zeros_like(QK).scatter_(-1, indices, top_attn)
        else:
            attn = torch.softmax(QK, dim=-1)

        attn = self.dropout(attn)

        outputs = multiply_by_xchunks(attn, V, self.qk_chunks).permute(2, 0, 1, 3)

        outputs = outputs.reshape(-1, bs, self.d_model)

        outputs = self.projection(outputs)

        return outputs, attn

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class MultiheadLocalAttentionV2(nn.Module):
    def __init__(
        self,
        d_model,
        num_head,
        dropout=0.0,
        max_dis=7,
        dilation=1,
        use_linear=True,
        enable_corr=True,
        d_att=None,
        use_dis=False,
    ):
        super().__init__()
        self.dilation = dilation
        self.window_size = 2 * max_dis + 1
        self.max_dis = max_dis
        self.num_head = num_head
        self.hidden_dim = d_model // num_head
        self.d_att = self.hidden_dim if d_att is None else d_att
        self.T = self.d_att**0.5
        self.use_dis = use_dis

        self.use_linear = use_linear
        if use_linear:
            self.linear_Q = nn.Conv2d(d_model, d_model, kernel_size=1)
            self.linear_K = nn.Conv2d(d_model, d_model, kernel_size=1)
            self.linear_V = nn.Conv2d(d_model, d_model, kernel_size=1)

        self.relative_emb_k = nn.Conv2d(
            self.d_att * self.num_head,
            num_head * self.window_size * self.window_size,
            kernel_size=1,
            groups=num_head,
        )
        self.relative_emb_v = nn.Parameter(
            torch.zeros(
                [self.num_head, d_model // self.num_head, self.window_size * self.window_size]
            )
        )

        # enable_corr=True in the real repo requires the optional `spatial_correlation_sampler`
        # CUDA extension (not a base lib, not installed). This vendored copy always constructs
        # with enable_corr=False, which is a real branch already present in forward() below
        # (`else: unfolded_k = self.pad_and_unfold(k)...`) computing the identical local
        # attention via F.unfold in pure PyTorch instead of the CUDA correlation sampler.
        self.enable_corr = enable_corr
        if enable_corr:
            from spatial_correlation_sampler import SpatialCorrelationSampler

            self.correlation_sampler = SpatialCorrelationSampler(
                kernel_size=1,
                patch_size=self.window_size,
                stride=1,
                padding=0,
                dilation=1,
                dilation_patch=self.dilation,
            )

        self.projection = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        self.drop_prob = dropout

        self.local_mask = None
        self.last_size_2d = None
        self.qk_mask = None

    def forward(self, q, k, v):
        n, c, h, w = v.size()

        if self.use_linear:
            q = self.linear_Q(q)
            k = self.linear_K(k)
            v = self.linear_V(v)

        hidden_dim = self.hidden_dim

        if self.qk_mask is not None and (h, w) == self.last_size_2d:
            qk_mask = self.qk_mask
        else:
            memory_mask = torch.ones((1, 1, h, w), device=v.device).float()
            unfolded_k_mask = self.pad_and_unfold(memory_mask).view(
                1, 1, self.window_size * self.window_size, h * w
            )
            qk_mask = 1 - unfolded_k_mask
            self.qk_mask = qk_mask

        relative_emb = self.relative_emb_k(q)

        q = q / self.T

        q = q.view(-1, self.d_att, h, w)
        k = k.view(-1, self.d_att, h, w)
        v = v.view(-1, self.num_head, hidden_dim, h * w)

        relative_emb = relative_emb.view(
            n, self.num_head, self.window_size * self.window_size, h * w
        )

        if self.enable_corr:
            qk = self.correlation_sampler(q, k).view(
                n, self.num_head, self.window_size * self.window_size, h * w
            )
        else:
            unfolded_k = self.pad_and_unfold(k).view(
                n * self.num_head, hidden_dim, self.window_size * self.window_size, h, w
            )
            qk = (
                (q.unsqueeze(2) * unfolded_k)
                .sum(dim=1)
                .view(n, self.num_head, self.window_size * self.window_size, h * w)
            )
        if self.use_dis:
            qk = 2 * qk - self.pad_and_unfold(k.pow(2).sum(dim=1, keepdim=True)).view(
                n, self.num_head, self.window_size * self.window_size, h * w
            )

        qk = qk + relative_emb

        qk -= qk_mask * 1e8 if qk.dtype == torch.float32 else qk_mask * 1e4

        local_attn = torch.softmax(qk, dim=2)

        local_attn = self.dropout(local_attn)

        agg_bias = torch.einsum("bhwn,hcw->bhnc", local_attn, self.relative_emb_v)

        global_attn = self.local2global(local_attn, h, w)

        agg_value = global_attn @ v.transpose(-2, -1)

        output = (agg_value + agg_bias).permute(2, 0, 1, 3).reshape(h * w, n, c)

        output = self.projection(output)

        self.last_size_2d = (h, w)
        return output, local_attn

    def local2global(self, local_attn, height, width):
        batch_size = local_attn.size()[0]

        pad_height = height + 2 * self.max_dis
        pad_width = width + 2 * self.max_dis

        if self.local_mask is not None and (height, width) == self.last_size_2d:
            local_mask = self.local_mask
        else:
            ky, kx = torch.meshgrid(
                [
                    torch.arange(0, pad_height, device=local_attn.device),
                    torch.arange(0, pad_width, device=local_attn.device),
                ],
                indexing="ij",
            )
            qy, qx = torch.meshgrid(
                [
                    torch.arange(0, height, device=local_attn.device),
                    torch.arange(0, width, device=local_attn.device),
                ],
                indexing="ij",
            )

            offset_y = qy.reshape(-1, 1) - ky.reshape(1, -1) + self.max_dis
            offset_x = qx.reshape(-1, 1) - kx.reshape(1, -1) + self.max_dis

            local_mask = (offset_y.abs() <= self.max_dis) & (offset_x.abs() <= self.max_dis)
            local_mask = local_mask.view(1, 1, height * width, pad_height, pad_width)
            self.local_mask = local_mask

        global_attn = torch.zeros(
            (batch_size, self.num_head, height * width, pad_height, pad_width),
            device=local_attn.device,
        )
        global_attn[local_mask.expand(batch_size, self.num_head, -1, -1, -1)] = (
            local_attn.transpose(-1, -2).reshape(-1)
        )
        global_attn = global_attn[
            :, :, :, self.max_dis : -self.max_dis, self.max_dis : -self.max_dis
        ].reshape(batch_size, self.num_head, height * width, height * width)

        return global_attn

    def pad_and_unfold(self, x):
        pad_pixel = self.max_dis * self.dilation
        x = F.pad(x, (pad_pixel, pad_pixel, pad_pixel, pad_pixel), mode="constant", value=0)
        x = F.unfold(
            x,
            kernel_size=(self.window_size, self.window_size),
            stride=(1, 1),
            dilation=self.dilation,
        )
        return x


# --------------------------------------------------------------------------------------
# aot/networks/layers/transformer.py (block_version="v1", the default)
# --------------------------------------------------------------------------------------


def _get_norm(indim, type="ln", groups=8):
    if type == "gn":
        return GroupNorm1D(indim, groups)
    else:
        return nn.LayerNorm(indim)


class LongShortTermTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        self_nhead,
        att_nhead,
        dim_feedforward=1024,
        droppath=0.1,
        lt_dropout=0.0,
        st_dropout=0.0,
        droppath_lst=False,
        activation="gelu",
        local_dilation=1,
        enable_corr=False,
    ):
        super().__init__()

        # Long Short-Term Attention
        self.norm1 = _get_norm(d_model)
        self.linear_Q = nn.Linear(d_model, d_model)
        self.linear_V = nn.Linear(d_model, d_model)

        self.long_term_attn = MultiheadAttention(
            d_model, att_nhead, use_linear=False, dropout=lt_dropout
        )

        # Real repo default is `enable_corr=True` (routes through the optional
        # spatial_correlation_sampler CUDA extension, not a base lib / not installed here).
        # `enable_corr=False` selects MultiheadLocalAttentionV2's own pure-PyTorch
        # pad_and_unfold code path -- same class, same architecture, no CUDA extension.
        self.short_term_attn = MultiheadLocalAttentionV2(
            d_model,
            att_nhead,
            dilation=local_dilation,
            use_linear=False,
            dropout=st_dropout,
            enable_corr=enable_corr,
        )
        self.lst_dropout = nn.Dropout(max(lt_dropout, st_dropout), True)
        self.droppath_lst = droppath_lst

        # Self-attention
        self.norm2 = _get_norm(d_model)
        self.self_attn = MultiheadAttention(d_model, self_nhead)

        # Feed-forward
        self.norm3 = _get_norm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = GNActDWConv2d(dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.droppath = DropPath(droppath, batch_dim=1)
        self._init_weight()

    def with_pos_embed(self, tensor, pos=None):
        size = tensor.size()
        if len(size) == 4 and pos is not None:
            n, c, h, w = size
            pos = pos.view(h, w, n, c).permute(2, 3, 0, 1)
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        long_term_memory=None,
        short_term_memory=None,
        curr_id_emb=None,
        self_pos=None,
        size_2d=(30, 30),
    ):
        # Self-attention
        _tgt = self.norm1(tgt)
        q = k = self.with_pos_embed(_tgt, self_pos)
        v = _tgt
        tgt2 = self.self_attn(q, k, v)[0]

        tgt = tgt + self.droppath(tgt2)

        # Long Short-Term Attention
        _tgt = self.norm2(tgt)

        curr_Q = self.linear_Q(_tgt)
        curr_K = curr_Q
        curr_V = _tgt

        local_Q = seq_to_2d(curr_Q, size_2d)

        if curr_id_emb is not None:
            global_K, global_V = self.fuse_key_value_id(curr_K, curr_V, curr_id_emb)
            local_K = seq_to_2d(global_K, size_2d)
            local_V = seq_to_2d(global_V, size_2d)
        else:
            global_K, global_V = long_term_memory
            local_K, local_V = short_term_memory

        tgt2 = self.long_term_attn(curr_Q, global_K, global_V)[0]
        tgt3 = self.short_term_attn(local_Q, local_K, local_V)[0]

        if self.droppath_lst:
            tgt = tgt + self.droppath(tgt2 + tgt3)
        else:
            tgt = tgt + self.lst_dropout(tgt2 + tgt3)

        # Feed-forward
        _tgt = self.norm3(tgt)

        tgt2 = self.linear2(self.activation(self.linear1(_tgt), size_2d))

        tgt = tgt + self.droppath(tgt2)

        return tgt, [[curr_K, curr_V], [global_K, global_V], [local_K, local_V]]

    def fuse_key_value_id(self, key, value, id_emb):
        K = key
        V = self.linear_V(value + id_emb)
        return K, V

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class LongShortTermTransformer(nn.Module):
    def __init__(
        self,
        num_layers=2,
        d_model=256,
        self_nhead=8,
        att_nhead=8,
        dim_feedforward=1024,
        emb_dropout=0.0,
        droppath=0.1,
        lt_dropout=0.0,
        st_dropout=0.0,
        droppath_lst=False,
        droppath_scaling=False,
        activation="gelu",
        return_intermediate=False,
        intermediate_norm=True,
        final_norm=True,
        block_version="v1",
    ):
        super().__init__()
        self.intermediate_norm = intermediate_norm
        self.final_norm = final_norm
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        self.emb_dropout = nn.Dropout(emb_dropout, True)
        self.mask_token = nn.Parameter(torch.randn([1, 1, d_model]))

        if block_version == "v1":
            block = LongShortTermTransformerBlock
        else:
            raise NotImplementedError(
                "only block_version='v1' vendored (no spatial_correlation_sampler dep)"
            )

        layers = []
        for idx in range(num_layers):
            if droppath_scaling:
                if num_layers == 1:
                    droppath_rate = 0
                else:
                    droppath_rate = droppath * idx / (num_layers - 1)
            else:
                droppath_rate = droppath
            layers.append(
                block(
                    d_model,
                    self_nhead,
                    att_nhead,
                    dim_feedforward,
                    droppath_rate,
                    lt_dropout,
                    st_dropout,
                    droppath_lst,
                    activation,
                )
            )
        self.layers = nn.ModuleList(layers)

        num_norms = num_layers - 1 if intermediate_norm else 0
        if final_norm:
            num_norms += 1
        self.decoder_norms = (
            nn.ModuleList([_get_norm(d_model, type="ln") for _ in range(num_norms)])
            if num_norms > 0
            else None
        )

    def forward(
        self,
        tgt,
        long_term_memories,
        short_term_memories,
        curr_id_emb=None,
        self_pos=None,
        size_2d=None,
    ):
        output = self.emb_dropout(tgt)

        intermediate = []
        intermediate_memories = []

        for idx, layer in enumerate(self.layers):
            output, memories = layer(
                output,
                long_term_memories[idx] if long_term_memories is not None else None,
                short_term_memories[idx] if short_term_memories is not None else None,
                curr_id_emb=curr_id_emb,
                self_pos=self_pos,
                size_2d=size_2d,
            )

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_memories.append(memories)

        if self.decoder_norms is not None:
            if self.final_norm:
                output = self.decoder_norms[-1](output)

            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

                if self.intermediate_norm:
                    for idx in range(len(intermediate) - 1):
                        intermediate[idx] = self.decoder_norms[idx](intermediate[idx])

        if self.return_intermediate:
            return intermediate, intermediate_memories

        return output, memories


# --------------------------------------------------------------------------------------
# aot/networks/decoders/fpn.py + aot/networks/decoders/__init__.py
# --------------------------------------------------------------------------------------


class FPNSegmentationHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        decode_intermediate_input=True,
        hidden_dim=256,
        shortcut_dims=(24, 32, 96, 1280),
        align_corners=True,
    ):
        super().__init__()
        self.align_corners = align_corners

        self.decode_intermediate_input = decode_intermediate_input

        self.conv_in = ConvGN(in_dim, hidden_dim, 1)

        self.conv_16x = ConvGN(hidden_dim, hidden_dim, 3)
        self.conv_8x = ConvGN(hidden_dim, hidden_dim // 2, 3)
        self.conv_4x = ConvGN(hidden_dim // 2, hidden_dim // 2, 3)

        self.adapter_16x = nn.Conv2d(shortcut_dims[-2], hidden_dim, 1)
        self.adapter_8x = nn.Conv2d(shortcut_dims[-3], hidden_dim, 1)
        self.adapter_4x = nn.Conv2d(shortcut_dims[-4], hidden_dim // 2, 1)

        self.conv_out = nn.Conv2d(hidden_dim // 2, out_dim, 1)

        self._init_weight()

    def forward(self, inputs, shortcuts):
        if self.decode_intermediate_input:
            x = torch.cat(inputs, dim=1)
        else:
            x = inputs[-1]

        x = F.relu_(self.conv_in(x))
        x = F.relu_(self.conv_16x(self.adapter_16x(shortcuts[-2]) + x))

        x = F.interpolate(
            x, size=shortcuts[-3].size()[-2:], mode="bilinear", align_corners=self.align_corners
        )
        x = F.relu_(self.conv_8x(self.adapter_8x(shortcuts[-3]) + x))

        x = F.interpolate(
            x, size=shortcuts[-4].size()[-2:], mode="bilinear", align_corners=self.align_corners
        )
        x = F.relu_(self.conv_4x(self.adapter_4x(shortcuts[-4]) + x))

        x = self.conv_out(x)

        return x

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


def build_decoder(name, **kwargs):
    if name == "fpn":
        return FPNSegmentationHead(**kwargs)
    else:
        raise NotImplementedError


# --------------------------------------------------------------------------------------
# aot/networks/models/aot.py (AOT) -- tiny cfg object standing in for
# aot/configs/models/{default,aott}.py + aot/configs/default.py
# --------------------------------------------------------------------------------------


class _TinyAOTConfig:
    """Stand-in for aot.configs.models.default.DefaultModelConfig, shrunk for fast tracing."""

    def __init__(self):
        self.MODEL_ENCODER = "mobilenetv2"
        self.MODEL_ENCODER_DIM = [24, 32, 96, 1280]
        self.MODEL_ENCODER_EMBEDDING_DIM = 32
        self.MODEL_DECODER_INTERMEDIATE_LSTT = True
        self.MODEL_FREEZE_BN = False
        self.TRAIN_ENCODER_FREEZE_AT = 0
        self.MODEL_MAX_OBJ_NUM = 3
        self.MODEL_SELF_HEADS = 2
        self.MODEL_ATT_HEADS = 2
        self.MODEL_LSTT_NUM = 1
        self.MODEL_EPSILON = 1e-5
        self.MODEL_ALIGN_CORNERS = True
        self.TRAIN_LSTT_EMB_DROPOUT = 0.0
        self.TRAIN_LSTT_DROPPATH = 0.0
        self.TRAIN_LSTT_LT_DROPOUT = 0.0
        self.TRAIN_LSTT_ST_DROPOUT = 0.0
        self.TRAIN_LSTT_DROPPATH_LST = False
        self.TRAIN_LSTT_DROPPATH_SCALING = False


class AOT(nn.Module):
    def __init__(self, cfg, encoder="mobilenetv2"):
        super().__init__()
        self.cfg = cfg
        self.max_obj_num = cfg.MODEL_MAX_OBJ_NUM
        self.epsilon = cfg.MODEL_EPSILON

        self.encoder = build_encoder(
            encoder, frozen_bn=cfg.MODEL_FREEZE_BN, freeze_at=cfg.TRAIN_ENCODER_FREEZE_AT
        )
        self.encoder_projector = nn.Conv2d(
            cfg.MODEL_ENCODER_DIM[-1], cfg.MODEL_ENCODER_EMBEDDING_DIM, kernel_size=1
        )

        self.LSTT = LongShortTermTransformer(
            cfg.MODEL_LSTT_NUM,
            cfg.MODEL_ENCODER_EMBEDDING_DIM,
            cfg.MODEL_SELF_HEADS,
            cfg.MODEL_ATT_HEADS,
            emb_dropout=cfg.TRAIN_LSTT_EMB_DROPOUT,
            droppath=cfg.TRAIN_LSTT_DROPPATH,
            lt_dropout=cfg.TRAIN_LSTT_LT_DROPOUT,
            st_dropout=cfg.TRAIN_LSTT_ST_DROPOUT,
            droppath_lst=cfg.TRAIN_LSTT_DROPPATH_LST,
            droppath_scaling=cfg.TRAIN_LSTT_DROPPATH_SCALING,
            intermediate_norm=cfg.MODEL_DECODER_INTERMEDIATE_LSTT,
            return_intermediate=True,
        )

        decoder_indim = (
            cfg.MODEL_ENCODER_EMBEDDING_DIM * (cfg.MODEL_LSTT_NUM + 1)
            if cfg.MODEL_DECODER_INTERMEDIATE_LSTT
            else cfg.MODEL_ENCODER_EMBEDDING_DIM
        )

        self.decoder = build_decoder(
            "fpn",
            in_dim=decoder_indim,
            out_dim=cfg.MODEL_MAX_OBJ_NUM + 1,
            decode_intermediate_input=cfg.MODEL_DECODER_INTERMEDIATE_LSTT,
            hidden_dim=cfg.MODEL_ENCODER_EMBEDDING_DIM,
            shortcut_dims=cfg.MODEL_ENCODER_DIM,
            align_corners=cfg.MODEL_ALIGN_CORNERS,
        )

        if cfg.MODEL_ALIGN_CORNERS:
            self.patch_wise_id_bank = nn.Conv2d(
                cfg.MODEL_MAX_OBJ_NUM + 1,
                cfg.MODEL_ENCODER_EMBEDDING_DIM,
                kernel_size=17,
                stride=16,
                padding=8,
            )
        else:
            self.patch_wise_id_bank = nn.Conv2d(
                cfg.MODEL_MAX_OBJ_NUM + 1,
                cfg.MODEL_ENCODER_EMBEDDING_DIM,
                kernel_size=16,
                stride=16,
                padding=0,
            )

        self.id_dropout = nn.Dropout(0.0, True)

        self.pos_generator = PositionEmbeddingSine(
            cfg.MODEL_ENCODER_EMBEDDING_DIM // 2, normalize=True
        )

        self._init_weight()

    def get_pos_emb(self, x):
        pos_emb = self.pos_generator(x)
        return pos_emb

    def get_id_emb(self, x):
        id_emb = self.patch_wise_id_bank(x)
        id_emb = self.id_dropout(id_emb)
        return id_emb

    def encode_image(self, img):
        xs = self.encoder(img)
        xs[-1] = self.encoder_projector(xs[-1])
        return xs

    def decode_id_logits(self, lstt_emb, shortcuts):
        n, c, h, w = shortcuts[-1].size()
        decoder_inputs = [shortcuts[-1]]
        for emb in lstt_emb:
            decoder_inputs.append(emb.view(h, w, n, c).permute(2, 3, 0, 1))
        pred_logit = self.decoder(decoder_inputs, shortcuts)
        return pred_logit

    def LSTT_forward(
        self,
        curr_embs,
        long_term_memories,
        short_term_memories,
        curr_id_emb=None,
        pos_emb=None,
        size_2d=(30, 30),
    ):
        n, c, h, w = curr_embs[-1].size()
        curr_emb = curr_embs[-1].view(n, c, h * w).permute(2, 0, 1)
        lstt_embs, lstt_memories = self.LSTT(
            curr_emb, long_term_memories, short_term_memories, curr_id_emb, pos_emb, size_2d
        )
        lstt_curr_memories, lstt_long_memories, lstt_short_memories = zip(*lstt_memories)
        return lstt_embs, lstt_curr_memories, lstt_long_memories, lstt_short_memories

    def _init_weight(self):
        nn.init.xavier_uniform_(self.encoder_projector.weight)
        nn.init.orthogonal_(
            self.patch_wise_id_bank.weight.view(self.cfg.MODEL_ENCODER_EMBEDDING_DIM, -1).permute(
                0, 1
            ),
            gain=17**-2 if self.cfg.MODEL_ALIGN_CORNERS else 16**-2,
        )


# --------------------------------------------------------------------------------------
# Thin single-frame driver -- reproduces AOTEngine.add_reference_frame()'s network call
# order (encode -> id-embed -> LSTT w/ curr_id_emb, no memory banks yet -> decode), the
# real first-frame code path in aot/networks/engines/aot_engine.py, as one forward() so
# TorchLens can trace the whole network end-to-end from one image + one one-hot mask.
# --------------------------------------------------------------------------------------


class AOTSingleFrameWrapper(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or _TinyAOTConfig()
        self.aot = AOT(self.cfg, encoder=self.cfg.MODEL_ENCODER)

    def forward(self, frame, one_hot_mask):
        """
        frame: (1, 3, H, W) float image, H/W multiples of 16 (mobilenetv2 stride-16 stages).
        one_hot_mask: (1, MODEL_MAX_OBJ_NUM + 1, H, W) one-hot object-id mask (background +
            up to MODEL_MAX_OBJ_NUM foreground objects), matching AOTEngine's reference-frame
            label encoding.
        """
        curr_embs = self.aot.encode_image(frame)
        n, c, h, w = curr_embs[-1].size()
        size_2d = (h, w)

        id_mask_2d = F.interpolate(one_hot_mask, size=(h * 16, w * 16), mode="nearest")
        curr_id_emb = self.aot.get_id_emb(id_mask_2d)
        curr_id_emb = curr_id_emb.view(n, self.cfg.MODEL_ENCODER_EMBEDDING_DIM, h * w).permute(
            2, 0, 1
        )

        pos_emb = self.aot.get_pos_emb(curr_embs[-1])
        pos_emb = pos_emb.view(n, self.cfg.MODEL_ENCODER_EMBEDDING_DIM, h * w).permute(2, 0, 1)

        lstt_embs, _curr_mem, _long_mem, _short_mem = self.aot.LSTT_forward(
            curr_embs,
            long_term_memories=None,
            short_term_memories=None,
            curr_id_emb=curr_id_emb,
            pos_emb=pos_emb,
            size_2d=size_2d,
        )

        pred_logit = self.aot.decode_id_logits(lstt_embs, curr_embs)
        return pred_logit


def build_aot_vos():
    return AOTSingleFrameWrapper()


def example_input_aot_vos():
    cfg = _TinyAOTConfig()
    frame = torch.randn(1, 3, 64, 64)
    one_hot_mask = torch.zeros(1, cfg.MODEL_MAX_OBJ_NUM + 1, 64, 64)
    one_hot_mask[:, 0] = 1.0  # background everywhere (tiny random mask, structure only)
    return (frame, one_hot_mask)


MENAGERIE_ZOO = "vendored-pytorch"

MENAGERIE_ENTRIES = [
    (
        "AOT (Associating Objects with Transformers) VOS tracker",
        "build_aot_vos",
        "example_input_aot_vos",
        2021,
        MENAGERIE_ZOO,
    ),
]
