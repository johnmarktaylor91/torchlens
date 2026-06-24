"""Wave-B OpenMMLab classification, segmentation, and OCR reimplementations.

Paper/source families: MMPreTrain image classifiers/backbones, MMSegmentation
EncoderDecoder heads, and MMOCR text detectors.  The implementations are compact
random-initialized PyTorch modules for base environments where OpenMMLab packages
cannot be installed, but preserve each target family's load-bearing operators:
residual/group/split/SE/CSP/dense/mobile CNN stages, patch/token transformer
classifiers, local-window and cross-covariance attention, UPerNet/FPN-style
segmentation decoders, context/non-local/OCR/PSA heads, Mask2Former/SAN-style
query heads, and MMOCR PAN/PSE/FCE/Mask-RCNN text-detection heads.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _make_divisible(value: int, divisor: int) -> int:
    """Round a channel count up to a group-compatible value.

    Parameters
    ----------
    value:
        Requested channel count.
    divisor:
        Divisibility requirement.

    Returns
    -------
    int
        Channel count divisible by ``divisor``.
    """

    return max(divisor, int(math.ceil(value / divisor) * divisor))


def _channel_shuffle(x: Tensor, groups: int) -> Tensor:
    """Shuffle channels across groups as used by ShuffleNet.

    Parameters
    ----------
    x:
        Feature map.
    groups:
        Number of channel groups.

    Returns
    -------
    Tensor
        Channel-shuffled feature map.
    """

    batch, channels, height, width = x.shape
    x = x.view(batch, groups, channels // groups, height, width)
    return x.transpose(1, 2).reshape(batch, channels, height, width)


def _pairwise_attention(tokens: Tensor) -> Tensor:
    """Apply simple scaled dot-product self-attention to token tensors.

    Parameters
    ----------
    tokens:
        Token tensor of shape ``(B, N, C)``.

    Returns
    -------
    Tensor
        Attention-refined tokens.
    """

    scale = tokens.shape[-1] ** -0.5
    weights = torch.softmax(tokens @ tokens.transpose(1, 2) * scale, dim=-1)
    return weights @ tokens


class ConvBNAct(nn.Module):
    """Convolution, batch normalization, and activation."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        groups: int = 1,
        act: str = "silu",
    ) -> None:
        """Initialize a convolutional block.

        Parameters
        ----------
        in_ch:
            Input channels.
        out_ch:
            Output channels.
        kernel:
            Kernel size.
        stride:
            Convolution stride.
        groups:
            Group count.
        act:
            Activation family.
        """

        super().__init__()
        padding = kernel // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        """Apply convolutional feature extraction.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        Tensor
            Activated feature map.
        """

        y = self.bn(self.conv(x))
        if self.act == "relu":
            return F.relu(y)
        if self.act == "gelu":
            return F.gelu(y)
        if self.act == "hswish":
            return y * F.relu6(y + 3.0) / 6.0
        return F.silu(y)


class SqueezeExcite(nn.Module):
    """Squeeze-excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        """Initialize SE projections.

        Parameters
        ----------
        channels:
            Feature channels.
        reduction:
            Bottleneck reduction ratio.
        """

        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Gate channels using global context.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        Tensor
            Channel-gated feature map.
        """

        gate = F.adaptive_avg_pool2d(x, 1)
        gate = torch.sigmoid(self.fc2(F.silu(self.fc1(gate))))
        return x * gate


class ResidualUnit(nn.Module):
    """Residual CNN unit covering ResNet, ResNeXt, SE-ResNet, Res2Net, and ResNeSt."""

    def __init__(self, channels: int, kind: str = "resnet", groups: int = 1) -> None:
        """Initialize residual branches.

        Parameters
        ----------
        channels:
            Feature channels.
        kind:
            Residual variant.
        groups:
            Group count for grouped variants.
        """

        super().__init__()
        grouped = groups if kind in {"resnext", "regnet"} else 1
        self.kind = kind
        self.conv1 = ConvBNAct(channels, channels, kernel=1)
        self.conv2 = ConvBNAct(channels, channels, groups=grouped)
        self.conv3 = ConvBNAct(channels, channels, kernel=1, act="relu")
        self.scale_convs = nn.ModuleList(
            [ConvBNAct(channels // 4, channels // 4) for _ in range(3)]
        )
        self.se = SqueezeExcite(channels)
        self.radix = nn.Conv2d(channels, channels * 2, 1)

    def _res2net(self, x: Tensor) -> Tensor:
        """Run split-scale Res2Net aggregation.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        Tensor
            Aggregated feature map.
        """

        splits = list(torch.chunk(x, 4, dim=1))
        outputs = [splits[0]]
        state = splits[0]
        for split, conv in zip(splits[1:], self.scale_convs, strict=True):
            state = conv(split + state)
            outputs.append(state)
        return torch.cat(outputs, dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual unit.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        Tensor
            Residual output.
        """

        y = self.conv1(x)
        if self.kind == "res2net":
            y = self._res2net(y)
        else:
            y = self.conv2(y)
        if self.kind in {"seresnet", "efficientnet", "mobilenet_v3"}:
            y = self.se(y)
        if self.kind == "resnest":
            radix = self.radix(F.adaptive_avg_pool2d(y, 1)).view(x.shape[0], 2, y.shape[1], 1, 1)
            weights = torch.softmax(radix, dim=1)
            y = y * weights[:, 0] + torch.roll(y, shifts=1, dims=1) * weights[:, 1]
        return F.relu(x + self.conv3(y))


class DenseUnit(nn.Module):
    """DenseNet-style growth block."""

    def __init__(self, in_ch: int, growth: int) -> None:
        """Initialize dense block.

        Parameters
        ----------
        in_ch:
            Input channels.
        growth:
            Added growth channels.
        """

        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(in_ch, growth * 2, kernel=1),
            ConvBNAct(growth * 2, growth),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Concatenate newly produced features with the input.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        Tensor
            Dense feature map.
        """

        return torch.cat([x, self.net(x)], dim=1)


class MBConv(nn.Module):
    """Mobile inverted bottleneck block used by MobileNet and EfficientNet."""

    def __init__(
        self,
        channels: int,
        expand: int = 4,
        fused: bool = False,
        use_se: bool = True,
        act: str = "silu",
    ) -> None:
        """Initialize mobile bottleneck layers.

        Parameters
        ----------
        channels:
            Feature channels.
        expand:
            Expansion ratio.
        fused:
            Whether to use EfficientNetV2 fused expansion.
        use_se:
            Whether to include squeeze-excitation.
        act:
            Activation family.
        """

        super().__init__()
        hidden = channels * expand
        self.fused = fused
        if fused:
            self.expand = ConvBNAct(channels, hidden, act=act)
            self.depthwise = nn.Identity()
        else:
            self.expand = ConvBNAct(channels, hidden, kernel=1, act=act)
            self.depthwise = ConvBNAct(hidden, hidden, groups=hidden, act=act)
        self.se = SqueezeExcite(hidden) if use_se else nn.Identity()
        self.project = ConvBNAct(hidden, channels, kernel=1, act="relu")

    def forward(self, x: Tensor) -> Tensor:
        """Apply mobile inverted bottleneck residual.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        Tensor
            Refined feature map.
        """

        y = self.expand(x)
        y = self.depthwise(y)
        y = self.project(self.se(y))
        return x + y


class ConvMixerBlock(nn.Module):
    """ConvMixer depthwise token mixing and pointwise channel mixing."""

    def __init__(self, channels: int) -> None:
        """Initialize ConvMixer block.

        Parameters
        ----------
        channels:
            Feature channels.
        """

        super().__init__()
        self.depthwise = ConvBNAct(channels, channels, kernel=5, groups=channels, act="gelu")
        self.pointwise = ConvBNAct(channels, channels, kernel=1, act="gelu")

    def forward(self, x: Tensor) -> Tensor:
        """Apply ConvMixer residual mixing.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        Tensor
            Mixed feature map.
        """

        return self.pointwise(x + self.depthwise(x))


class ConvNeXtBlock(nn.Module):
    """ConvNeXt/ConvNeXtV2 block with large depthwise kernel and optional GRN."""

    def __init__(self, channels: int, grn: bool = False) -> None:
        """Initialize ConvNeXt block.

        Parameters
        ----------
        channels:
            Feature channels.
        grn:
            Whether to include global response normalization.
        """

        super().__init__()
        self.grn = grn
        self.dw = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.pw1 = nn.Conv2d(channels, channels * 4, 1)
        self.pw2 = nn.Conv2d(channels * 4, channels, 1)
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * 1e-3)

    def forward(self, x: Tensor) -> Tensor:
        """Apply ConvNeXt feature mixing.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        Tensor
            Refined feature map.
        """

        y = self.dw(x)
        y = F.gelu(self.pw1(y))
        if self.grn:
            norm = torch.linalg.vector_norm(y, dim=(2, 3), keepdim=True)
            y = y * norm / (norm.mean(dim=1, keepdim=True) + 1e-6)
        y = self.pw2(y)
        return x + self.gamma * y


class TransformerBlock(nn.Module):
    """Batch-first transformer block for compact vision token models."""

    def __init__(self, dim: int, heads: int = 4, mlp_ratio: int = 2) -> None:
        """Initialize self-attention and MLP layers.

        Parameters
        ----------
        dim:
            Token width.
        heads:
            Attention heads.
        mlp_ratio:
            Feed-forward expansion.
        """

        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply transformer attention and feed-forward mixing.

        Parameters
        ----------
        x:
            Token tensor.

        Returns
        -------
        Tensor
            Refined token tensor.
        """

        y = self.norm1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class WindowAttentionBlock(nn.Module):
    """Swin/TinyViT-style local window attention with optional shifted windows."""

    def __init__(self, channels: int, shift: bool = False) -> None:
        """Initialize local attention block.

        Parameters
        ----------
        channels:
            Feature channels.
        shift:
            Whether to cyclically shift features before window attention.
        """

        super().__init__()
        self.shift = shift
        self.block = TransformerBlock(channels, heads=4)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply window attention over a compact feature grid.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        Tensor
            Attention-refined feature map.
        """

        if self.shift:
            x = torch.roll(x, shifts=(1, 1), dims=(2, 3))
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        y = self.block(tokens).transpose(1, 2).reshape(batch, channels, height, width)
        if self.shift:
            y = torch.roll(y, shifts=(-1, -1), dims=(2, 3))
        return x + self.proj(y)


class XCiTBlock(nn.Module):
    """Cross-covariance attention block from XCiT."""

    def __init__(self, dim: int) -> None:
        """Initialize token projections.

        Parameters
        ----------
        dim:
            Token width.
        """

        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.local = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply cross-covariance channel attention.

        Parameters
        ----------
        x:
            Token tensor.

        Returns
        -------
        Tensor
            Attention-refined tokens.
        """

        y = self.norm(x)
        q = F.normalize(self.q(y), dim=1)
        k = F.normalize(self.k(y), dim=1)
        v = self.v(y)
        attn = torch.softmax(q.transpose(1, 2) @ k * (x.shape[1] ** -0.5), dim=-1)
        mixed = v @ attn
        local = self.local(y.transpose(1, 2)).transpose(1, 2)
        return x + self.proj(mixed + local)


class MLPBlock(nn.Module):
    """MLP-Mixer token and channel MLP block."""

    def __init__(self, tokens: int, dim: int) -> None:
        """Initialize mixer layers.

        Parameters
        ----------
        tokens:
            Number of patch tokens.
        dim:
            Token width.
        """

        super().__init__()
        self.token_mix = nn.Sequential(
            nn.Linear(tokens, tokens), nn.GELU(), nn.Linear(tokens, tokens)
        )
        self.channel_mix = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        """Mix tokens and channels.

        Parameters
        ----------
        x:
            Token tensor.

        Returns
        -------
        Tensor
            Mixed token tensor.
        """

        y = self.norm1(x).transpose(1, 2)
        x = x + self.token_mix(y).transpose(1, 2)
        return x + self.channel_mix(self.norm2(x))


class SAMImageEncoderCompact(nn.Module):
    """SAM ViT-Det image encoder with window/global blocks and convolutional neck."""

    def __init__(self, classes: int) -> None:
        """Initialize patch embedding, local/global attention blocks, and neck.

        Parameters
        ----------
        classes:
            Number of output classes for the compact catalog head.
        """

        super().__init__()
        self.patch = nn.Conv2d(3, 64, 8, stride=8)
        self.window_a = WindowAttentionBlock(64)
        self.global_a = TransformerBlock(64)
        self.window_b = WindowAttentionBlock(64, shift=True)
        self.global_b = TransformerBlock(64)
        self.neck = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.LayerNorm([64, 8, 8]),
            nn.Conv2d(64, 64, 3, padding=1),
        )
        self.head = nn.Linear(64, classes)

    def forward(self, x: Tensor) -> Tensor:
        """Encode an image using local windows with periodic global attention.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        Tensor
            Compact image-encoder logits.
        """

        grid = self.patch(x)
        grid = self.window_a(grid)
        tokens = self.global_a(grid.flatten(2).transpose(1, 2))
        grid = tokens.transpose(1, 2).reshape_as(grid)
        grid = self.window_b(grid)
        tokens = self.global_b(grid.flatten(2).transpose(1, 2))
        grid = self.neck(tokens.transpose(1, 2).reshape_as(grid))
        return self.head(grid.mean(dim=(2, 3)))


class HiViTCompact(nn.Module):
    """HiViT hierarchical patch-merging vision transformer without a class token."""

    def __init__(self, classes: int) -> None:
        """Initialize hierarchical stages and patch-merging projections.

        Parameters
        ----------
        classes:
            Number of output classes for the compact catalog head.
        """

        super().__init__()
        self.patch = nn.Conv2d(3, 32, 4, stride=4)
        self.stage1 = TransformerBlock(32, heads=4)
        self.merge1 = nn.Conv2d(32, 48, 2, stride=2)
        self.stage2 = TransformerBlock(48, heads=4)
        self.merge2 = nn.Conv2d(48, 64, 2, stride=2)
        self.stage3 = TransformerBlock(64, heads=4)
        self.head = nn.Linear(64, classes)

    def _stage(self, grid: Tensor, block: TransformerBlock) -> Tensor:
        """Apply a token block to a spatial grid and restore grid layout.

        Parameters
        ----------
        grid:
            Feature grid.
        block:
            Token mixing block.

        Returns
        -------
        Tensor
            Refined feature grid.
        """

        batch, channels, height, width = grid.shape
        tokens = block(grid.flatten(2).transpose(1, 2))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)

    def forward(self, x: Tensor) -> Tensor:
        """Run hierarchical ViT stages and global-average the final tokens.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        Tensor
            Compact classifier logits.
        """

        grid = self._stage(self.patch(x), self.stage1)
        grid = self._stage(self.merge1(grid), self.stage2)
        grid = self._stage(self.merge2(grid), self.stage3)
        return self.head(grid.mean(dim=(2, 3)))


class EVA02Block(nn.Module):
    """EVA-02 transformer block with RoPE attention, SwiGLU FFN, and sub-LN."""

    def __init__(self, dim: int = 64, heads: int = 4) -> None:
        """Initialize RoPE attention and SwiGLU feed-forward paths.

        Parameters
        ----------
        dim:
            Token width.
        heads:
            Attention head count.
        """

        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_subln = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.gate = nn.Linear(dim, dim * 4)
        self.value = nn.Linear(dim, dim * 4)
        self.ffn_subln = nn.LayerNorm(dim * 4)
        self.out = nn.Linear(dim * 4, dim)

    def _rope(self, tensor: Tensor) -> Tensor:
        """Apply rotary position embedding to query or key heads.

        Parameters
        ----------
        tensor:
            Head tensor shaped ``(batch, heads, tokens, head_dim)``.

        Returns
        -------
        Tensor
            Rotary-positioned tensor.
        """

        pos = torch.arange(tensor.shape[-2], device=tensor.device, dtype=tensor.dtype)
        freq = torch.arange(0, tensor.shape[-1], 2, device=tensor.device, dtype=tensor.dtype)
        angles = pos[:, None] / (10000.0 ** (freq[None, :] / tensor.shape[-1]))
        sin = angles.sin()[None, None]
        cos = angles.cos()[None, None]
        even = tensor[..., 0::2]
        odd = tensor[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def forward(self, x: Tensor) -> Tensor:
        """Apply one EVA-02 block.

        Parameters
        ----------
        x:
            Token tensor.

        Returns
        -------
        Tensor
            Refined token tensor.
        """

        batch, tokens, dim = x.shape
        qkv = self.qkv(self.norm1(x)).view(batch, tokens, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        q = self._rope(q)
        k = self._rope(k)
        attn = torch.softmax(q @ k.transpose(-2, -1) * (self.head_dim**-0.5), dim=-1)
        mixed = (attn @ v).transpose(1, 2).reshape(batch, tokens, dim)
        x = x + self.proj(self.attn_subln(mixed))
        y = self.norm2(x)
        y = F.silu(self.gate(y)) * self.value(y)
        return x + self.out(self.ffn_subln(y))


class EVA02Compact(nn.Module):
    """EVA-02 compact ViT with RoPE, SwiGLU feed-forward layers, and sub-LN."""

    def __init__(self, classes: int) -> None:
        """Initialize patch tokens, EVA-02 blocks, and classifier head.

        Parameters
        ----------
        classes:
            Number of output classes for the compact catalog head.
        """

        super().__init__()
        self.patch = PatchTokenizer(dim=64, patch=8, extra_tokens=1)
        self.blocks = nn.Sequential(EVA02Block(), EVA02Block())
        self.head = nn.Linear(64, classes)

    def forward(self, x: Tensor) -> Tensor:
        """Classify an image with EVA-02 token mixing.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        Tensor
            Compact classifier logits.
        """

        tokens = self.blocks(self.patch(x))
        return self.head(tokens[:, 0])


class PatchTokenizer(nn.Module):
    """Image-to-token projection with learnable class and position tokens."""

    def __init__(self, dim: int = 64, patch: int = 8, extra_tokens: int = 1) -> None:
        """Initialize token projection.

        Parameters
        ----------
        dim:
            Token width.
        patch:
            Patch size.
        extra_tokens:
            Number of prepended learned tokens.
        """

        super().__init__()
        self.patch = nn.Conv2d(3, dim, patch, stride=patch)
        self.extra = nn.Parameter(torch.zeros(1, extra_tokens, dim))
        self.pos = nn.Parameter(torch.randn(1, 66, dim) * 0.02)

    def forward(self, x: Tensor) -> Tensor:
        """Tokenize an image.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        Tensor
            Token sequence.
        """

        tokens = self.patch(x).flatten(2).transpose(1, 2)
        extra = self.extra.expand(x.shape[0], -1, -1)
        tokens = torch.cat([extra, tokens], dim=1)
        return tokens + self.pos[:, : tokens.shape[1]]


class MMPPretrainClassifier(nn.Module):
    """Compact MMPreTrain family classifier/backbone preserving target primitives."""

    def __init__(self, kind: str, classes: int = 11, width: int = 32) -> None:
        """Initialize a classification family.

        Parameters
        ----------
        kind:
            MMPreTrain family key.
        classes:
            Number of output classes.
        width:
            Base width.
        """

        super().__init__()
        self.kind = kind
        self.classes = classes
        self.width = _make_divisible(width, 8)
        self.cnn_stem = ConvBNAct(3, self.width, stride=2)
        self.patch = PatchTokenizer(dim=64, patch=8, extra_tokens=2 if kind == "deit" else 1)
        self.head = nn.Linear(self.width, classes)
        self.token_head = nn.Linear(64, classes)
        self.arc_margin = nn.Parameter(torch.ones(classes, 64))
        self.text_tokens = nn.Parameter(torch.randn(1, 6, 64) * 0.02)
        self.text_block = TransformerBlock(64)
        self.sam_encoder = SAMImageEncoderCompact(classes)
        self.hivit_encoder = HiViTCompact(classes)
        self.eva02_encoder = EVA02Compact(classes)
        self._build_family_modules(kind)

    def _build_family_modules(self, kind: str) -> None:
        """Create modules for a requested model family.

        Parameters
        ----------
        kind:
            MMPreTrain family key.
        """

        w = self.width
        residual_kind = {
            "arcface": "resnet",
            "csra": "resnet",
            "resnet": "resnet",
            "wrn": "resnet",
            "resnext": "resnext",
            "regnet": "regnet",
            "res2net": "res2net",
            "resnest": "resnest",
            "seresnet": "seresnet",
        }.get(kind)
        if residual_kind is not None:
            groups = 8 if residual_kind in {"resnext", "regnet"} else 1
            self.cnn_body = nn.Sequential(
                ResidualUnit(w, residual_kind, groups=groups),
                ConvBNAct(w, w * 2, stride=2),
                ResidualUnit(w * 2, residual_kind, groups=groups),
                ResidualUnit(w * 2, residual_kind, groups=groups),
            )
            self.head = nn.Linear(w * 2, self.classes)
            self.csra_score = nn.Conv2d(w * 2, self.classes, 1)
        elif kind == "densenet":
            self.dense1 = DenseUnit(w, 16)
            self.dense2 = DenseUnit(w + 16, 16)
            self.transition = ConvBNAct(w + 32, w * 2, kernel=1, stride=2)
            self.cnn_body = nn.Sequential(self.dense1, self.dense2, self.transition)
            self.head = nn.Linear(w * 2, self.classes)
        elif kind in {"efficientnet", "efficientnet_v2", "mobilenet_v2", "mobilenet_v3"}:
            fused = kind == "efficientnet_v2"
            act = "hswish" if kind == "mobilenet_v3" else "silu"
            self.cnn_body = nn.Sequential(
                MBConv(w, expand=2, fused=fused, use_se=kind != "mobilenet_v2", act=act),
                ConvBNAct(w, w * 2, stride=2, act=act),
                MBConv(w * 2, expand=3, fused=fused, use_se=True, act=act),
            )
            self.head = nn.Linear(w * 2, self.classes)
        elif kind in {"shufflenet_v1", "shufflenet_v2"}:
            self.group1 = ConvBNAct(w, w, kernel=1, groups=4)
            self.depth = ConvBNAct(w, w, groups=w)
            self.group2 = ConvBNAct(w, w * 2, kernel=1, groups=4 if kind == "shufflenet_v1" else 1)
            self.head = nn.Linear(w * 2, self.classes)
        elif kind in {"convnext", "convnext_v2", "hornet", "van", "edgenext", "repvgg"}:
            self.cnn_body = nn.Sequential(
                ConvNeXtBlock(w, grn=kind == "convnext_v2"),
                ConvBNAct(w, w * 2, stride=2),
                ConvNeXtBlock(w * 2, grn=kind == "convnext_v2"),
            )
            self.gate = nn.Conv2d(w * 2, w * 2, 7, padding=3, groups=w * 2)
            self.rep_1x1 = nn.Conv2d(w * 2, w * 2, 1)
            self.head = nn.Linear(w * 2, self.classes)
        elif kind in {"convmixer", "poolformer"}:
            self.cnn_stem = ConvBNAct(3, w, kernel=8, stride=8, act="gelu")
            self.cnn_body = nn.Sequential(ConvMixerBlock(w), ConvMixerBlock(w), ConvMixerBlock(w))
            self.head = nn.Linear(w, self.classes)
        elif kind in {"conformer", "efficientformer", "hrnet", "repmlp"}:
            self.cnn_body = nn.Sequential(ResidualUnit(w), ConvBNAct(w, w * 2, stride=2))
            self.high_branch = nn.Sequential(ResidualUnit(w), ResidualUnit(w))
            self.low_branch = nn.Sequential(ConvBNAct(w, w * 2, stride=2), ResidualUnit(w * 2))
            self.low_to_high = nn.Conv2d(w * 2, w, 1)
            self.high_to_low = nn.Conv2d(w, w * 2, 3, stride=2, padding=1)
            self.token_blocks = nn.Sequential(TransformerBlock(64), TransformerBlock(64))
            self.token_proj = nn.Linear(64, w * 2)
            self.repmlp_global = nn.Linear(w, w)
            self.repmlp_local = ConvBNAct(w, w, kernel=3)
            self.head = nn.Linear(w * 2 if kind != "hrnet" else w * 3, self.classes)
        elif kind == "cspnet":
            self.part1 = nn.Sequential(ResidualUnit(w // 2), ResidualUnit(w // 2))
            self.part2 = nn.Conv2d(w // 2, w // 2, 1)
            self.csp_fuse = ConvBNAct(w, w * 2, kernel=1)
            self.head = nn.Linear(w * 2, self.classes)
        elif kind == "vgg":
            self.cnn_body = nn.Sequential(
                ConvBNAct(w, w, act="relu"),
                nn.MaxPool2d(2),
                ConvBNAct(w, w * 2, act="relu"),
                nn.MaxPool2d(2),
                ConvBNAct(w * 2, w * 2, act="relu"),
            )
            self.head = nn.Linear(w * 2, self.classes)
        elif kind == "inception_v3":
            self.branch1 = ConvBNAct(w, w, kernel=1)
            self.branch3 = ConvBNAct(w, w, kernel=3)
            self.branch5 = ConvBNAct(w, w, kernel=5)
            self.incept_fuse = ConvBNAct(w * 3, w * 2, kernel=1)
            self.head = nn.Linear(w * 2, self.classes)
        elif kind == "mlp_mixer":
            self.mixer = nn.Sequential(MLPBlock(64, 64), MLPBlock(64, 64))
        elif kind in {"xcit"}:
            self.token_blocks = nn.Sequential(XCiTBlock(64), XCiTBlock(64))
        else:
            self._build_transformer_family(kind)

    def _build_transformer_family(self, kind: str) -> None:
        """Create transformer-family modules.

        Parameters
        ----------
        kind:
            Transformer-like MMPreTrain family key.
        """

        self.token_blocks = nn.Sequential(TransformerBlock(64), TransformerBlock(64))
        self.window1 = WindowAttentionBlock(64)
        self.window2 = WindowAttentionBlock(64, shift=True)
        self.mobile_local = ConvBNAct(64, 64)
        self.channel_gate = nn.Linear(64, 64)
        self.pixel_head = nn.Linear(64, 3 * 8 * 8)
        self.graph_proj = nn.Linear(64, 64)
        self.rev_f = nn.Linear(32, 32)
        self.rev_g = nn.Linear(32, 32)
        self.dinov2_registers = nn.Parameter(torch.randn(1, 4, 64) * 0.02)
        self.tnt_inner = TransformerBlock(16)
        self.tnt_outer = TransformerBlock(64)
        self.t2t_unfold = nn.Unfold(kernel_size=3, padding=1, stride=2)
        self.t2t_proj = nn.Linear(3 * 3 * 3, 64)

    def _cnn_forward(self, x: Tensor) -> Tensor:
        """Run CNN-family forward pass.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        Tensor
            Class logits.
        """

        feat = self.cnn_stem(x)
        if self.kind == "cspnet":
            left, right = feat.chunk(2, dim=1)
            feat = self.csp_fuse(torch.cat([self.part1(left), self.part2(right)], dim=1))
        elif self.kind in {"shufflenet_v1", "shufflenet_v2"}:
            feat = _channel_shuffle(self.group1(feat), 4)
            feat = self.group2(self.depth(feat))
        elif self.kind == "inception_v3":
            feat = self.incept_fuse(
                torch.cat([self.branch1(feat), self.branch3(feat), self.branch5(feat)], dim=1)
            )
        elif self.kind == "hrnet":
            high = self.high_branch(feat)
            low = self.low_branch(feat)
            high = high + F.interpolate(self.low_to_high(low), size=high.shape[-2:], mode="nearest")
            low = low + self.high_to_low(high)
            pooled = torch.cat([high.mean(dim=(2, 3)), low.mean(dim=(2, 3))], dim=1)
            return self.head(pooled)
        elif self.kind == "conformer":
            feat = self.cnn_body(feat)
            token_summary = self.token_proj(self.token_blocks(self.patch(x))[:, 0]).view(
                x.shape[0], -1, 1, 1
            )
            feat = feat + token_summary
        elif self.kind == "efficientformer":
            feat = self.cnn_body(feat)
            tokens = feat.flatten(2).transpose(1, 2)
            feat = self.token_blocks(tokens).transpose(1, 2).reshape_as(feat)
        elif self.kind == "repmlp":
            local = self.repmlp_local(feat)
            gate = torch.sigmoid(self.repmlp_global(feat.mean(dim=(2, 3)))).view(
                x.shape[0], -1, 1, 1
            )
            feat = self.cnn_body(local * gate + feat)
        else:
            feat = self.cnn_body(feat)
        if self.kind == "hornet":
            feat = feat + torch.sin(self.gate(feat))
        elif self.kind == "van":
            feat = feat * torch.sigmoid(self.gate(feat))
        elif self.kind == "repvgg":
            feat = feat + self.rep_1x1(feat)
        pooled = feat.mean(dim=(2, 3))
        if self.kind == "csra":
            attn = torch.softmax(self.csra_score(feat).flatten(2), dim=-1)
            class_feat = torch.einsum("bcn,bkn->bkc", feat.flatten(2), attn)
            return self.head(class_feat.mean(dim=1))
        if self.kind == "arcface":
            emb = F.normalize(pooled, dim=-1)
            margin_weight = F.normalize(self.arc_margin[:, : emb.shape[-1]], dim=-1)
            return emb @ margin_weight.t()
        return self.head(pooled)

    def _transformer_forward(self, x: Tensor) -> Tensor:
        """Run transformer-family forward pass.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        Tensor
            Class logits.
        """

        if self.kind == "t2t_vit":
            patches = self.t2t_unfold(x).transpose(1, 2)
            tokens = self.t2t_proj(patches)
            tokens = torch.cat([self.patch.extra[:, :1].expand(x.shape[0], -1, -1), tokens], dim=1)
        else:
            tokens = self.patch(x)
        if self.kind in {"swin_transformer", "swin_transformer_v2", "tinyvit"}:
            grid = tokens[:, 1:].transpose(1, 2).reshape(x.shape[0], 64, 8, 8)
            grid = self.window2(self.window1(grid))
            tokens = torch.cat([tokens[:, :1], grid.flatten(2).transpose(1, 2)], dim=1)
        elif self.kind in {"davit", "twins", "mvit"}:
            tokens = self.token_blocks(tokens)
            channel = torch.sigmoid(self.channel_gate(tokens.mean(dim=1))).unsqueeze(1)
            tokens = tokens * channel + _pairwise_attention(tokens)
        elif self.kind == "mobilevit":
            grid = tokens[:, 1:].transpose(1, 2).reshape(x.shape[0], 64, 8, 8)
            local = self.mobile_local(grid).flatten(2).transpose(1, 2)
            tokens = self.token_blocks(torch.cat([tokens[:, :1], local], dim=1))
        elif self.kind == "tnt":
            inner = tokens[:, 1:].reshape(x.shape[0], 16, 4, 64).mean(dim=2)[..., :16]
            inner = self.tnt_inner(inner)
            tokens = self.tnt_outer(tokens + F.pad(inner.mean(dim=1, keepdim=True), (0, 48)))
        elif self.kind == "revvit":
            a, b = tokens.chunk(2, dim=-1)
            tokens = torch.cat([a + self.rev_f(b), b + self.rev_g(a + self.rev_f(b))], dim=-1)
            tokens = self.token_blocks(tokens)
        elif self.kind == "vig":
            affinity = torch.softmax(tokens @ tokens.transpose(1, 2) * 0.125, dim=-1)
            tokens = tokens + self.graph_proj(affinity @ tokens)
        elif self.kind == "itpn":
            tokens = self.token_blocks(tokens)
            _ = self.pixel_head(tokens[:, 1:]).mean()
        elif self.kind == "dinov2":
            registers = self.dinov2_registers.expand(x.shape[0], -1, -1)
            tokens = torch.cat([tokens[:, :1], registers, tokens[:, 1:]], dim=1)
            tokens = self.token_blocks(tokens)
        elif self.kind == "clip":
            image_tokens = self.token_blocks(tokens)
            text_tokens = self.text_block(self.text_tokens.expand(x.shape[0], -1, -1))
            image = F.normalize(image_tokens[:, 0], dim=-1)
            text = F.normalize(text_tokens.mean(dim=1), dim=-1)
            return (image * text).sum(dim=-1, keepdim=True).repeat(1, self.classes)
        else:
            tokens = self.token_blocks(tokens)
        if self.kind == "dinov2":
            return self.token_head(tokens[:, :5].mean(dim=1))
        pooled = tokens[:, :2].mean(dim=1) if self.kind == "deit" else tokens[:, 0]
        return self.token_head(pooled)

    def forward(self, x: Tensor) -> Tensor:
        """Classify an image or expose a compact backbone representation.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        Tensor
            Class logits.
        """

        cnn_kinds = {
            "arcface",
            "csra",
            "resnet",
            "wrn",
            "resnext",
            "regnet",
            "res2net",
            "resnest",
            "seresnet",
            "densenet",
            "efficientnet",
            "efficientnet_v2",
            "mobilenet_v2",
            "mobilenet_v3",
            "shufflenet_v1",
            "shufflenet_v2",
            "convnext",
            "convnext_v2",
            "hornet",
            "van",
            "edgenext",
            "repvgg",
            "convmixer",
            "poolformer",
            "conformer",
            "efficientformer",
            "hrnet",
            "repmlp",
            "cspnet",
            "vgg",
            "inception_v3",
        }
        if self.kind in cnn_kinds:
            return self._cnn_forward(x)
        if self.kind == "mlp_mixer":
            tokens = self.patch(x)[:, 1:]
            return self.token_head(self.mixer(tokens).mean(dim=1))
        if self.kind == "xcit":
            tokens = self.patch(x)
            return self.token_head(self.token_blocks(tokens)[:, 0])
        if self.kind == "sam":
            return self.sam_encoder(x)
        if self.kind == "hivit":
            return self.hivit_encoder(x)
        if self.kind == "eva02":
            return self.eva02_encoder(x)
        return self._transformer_forward(x)


class FeaturePyramid(nn.Module):
    """Shared compact backbone producing four-scale feature maps."""

    def __init__(self, width: int = 24) -> None:
        """Initialize FPN backbone.

        Parameters
        ----------
        width:
            Base channel width.
        """

        super().__init__()
        self.stem = ConvBNAct(3, width, stride=2)
        self.c2 = ConvBNAct(width, width, stride=1)
        self.c3 = ConvBNAct(width, width * 2, stride=2)
        self.c4 = ConvBNAct(width * 2, width * 4, stride=2)
        self.c5 = ConvBNAct(width * 4, width * 4, stride=2)
        self.lateral = nn.ModuleList(
            [
                nn.Conv2d(width, width, 1),
                nn.Conv2d(width * 2, width, 1),
                nn.Conv2d(width * 4, width, 1),
                nn.Conv2d(width * 4, width, 1),
            ]
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        """Build multi-resolution features.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        list[Tensor]
            Four feature maps from high to low resolution.
        """

        c2 = self.c2(self.stem(x))
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        c5 = self.c5(c4)
        raw = [c2, c3, c4, c5]
        feats = [lat(feat) for lat, feat in zip(self.lateral, raw, strict=True)]
        for idx in range(len(feats) - 2, -1, -1):
            feats[idx] = feats[idx] + F.interpolate(
                feats[idx + 1], size=feats[idx].shape[-2:], mode="nearest"
            )
        return feats


class ContextHead(nn.Module):
    """MMSeg-style decode head variants."""

    def __init__(self, kind: str, width: int = 24, classes: int = 7) -> None:
        """Initialize a segmentation head.

        Parameters
        ----------
        kind:
            Segmentation family key.
        width:
            Feature width.
        classes:
            Output classes.
        """

        super().__init__()
        self.kind = kind
        self.classes = classes
        self.fuse = ConvBNAct(width * 4, width)
        self.context = nn.Conv2d(width, width, 1)
        self.context2 = nn.Conv2d(width, width, 3, padding=2, dilation=2)
        self.context3 = nn.Conv2d(width, width, 3, padding=4, dilation=4)
        self.query = nn.Conv2d(width, width, 1)
        self.key = nn.Conv2d(width, width, 1)
        self.value = nn.Conv2d(width, width, 1)
        self.object_proj = nn.Linear(width, width)
        self.query_embed = nn.Parameter(torch.randn(1, 8, width) * 0.02)
        self.query_block = TransformerBlock(width, heads=4)
        self.cls = nn.Conv2d(width, classes, 1)
        self.mask_embed = nn.Linear(width, width)

    def _resize_concat(self, feats: list[Tensor]) -> Tensor:
        """Resize features to the finest level and concatenate.

        Parameters
        ----------
        feats:
            Feature pyramid.

        Returns
        -------
        Tensor
            Concatenated feature map.
        """

        size = feats[0].shape[-2:]
        return torch.cat([F.interpolate(feat, size=size, mode="nearest") for feat in feats], dim=1)

    def _attention_context(self, feat: Tensor) -> Tensor:
        """Apply non-local or spatial attention context.

        Parameters
        ----------
        feat:
            Feature map.

        Returns
        -------
        Tensor
            Context-enhanced feature map.
        """

        batch, channels, height, width = feat.shape
        q = self.query(feat).flatten(2).transpose(1, 2)
        k = self.key(feat).flatten(2)
        v = self.value(feat).flatten(2).transpose(1, 2)
        attn = torch.softmax(q @ k * (channels**-0.5), dim=-1)
        return (attn @ v).transpose(1, 2).reshape(batch, channels, height, width)

    def forward(self, feats: list[Tensor]) -> Tensor:
        """Decode segmentation logits.

        Parameters
        ----------
        feats:
            Feature pyramid.

        Returns
        -------
        Tensor
            Segmentation logits.
        """

        feat = self.fuse(self._resize_concat(feats))
        if self.kind in {"apcnet", "semantic-fpn", "upernet", "beit", "mae", "lraspp"}:
            pooled = [
                F.interpolate(
                    F.adaptive_avg_pool2d(feat, scale), size=feat.shape[-2:], mode="nearest"
                )
                for scale in (1, 2, 4)
            ]
            feat = feat + self.context(sum(pooled))
        elif self.kind in {"ann-seg", "nonlocal-seg", "ccnet", "psanet"}:
            feat = feat + self._attention_context(feat)
            if self.kind == "ccnet":
                feat = (
                    feat + torch.roll(feat, shifts=1, dims=2) + torch.roll(feat, shifts=1, dims=3)
                )
        elif self.kind == "danet":
            spatial = self._attention_context(feat)
            channel = torch.softmax(feat.flatten(2) @ feat.flatten(2).transpose(1, 2), dim=-1)
            channel_context = (channel @ feat.flatten(2)).reshape_as(feat)
            feat = feat + spatial + channel_context
        elif self.kind == "ocrnet":
            coarse = torch.softmax(self.cls(feat), dim=1)
            objects = torch.einsum("bkhw,bchw->bkc", coarse, feat)
            obj = self.object_proj(objects).mean(dim=1).view(feat.shape[0], -1, 1, 1)
            feat = feat + obj
        elif self.kind == "mask2former":
            tokens = self.query_embed.expand(feat.shape[0], -1, -1)
            tokens = self.query_block(tokens + feat.mean(dim=(2, 3)).unsqueeze(1))
            masks = torch.einsum("bqc,bchw->bqhw", self.mask_embed(tokens), feat)
            return masks[:, : self.classes]
        elif self.kind == "san":
            adapter = torch.sigmoid(self.context(feat))
            feat = feat * adapter + self._attention_context(feat) * (1.0 - adapter)
        elif self.kind in {"bisenetv1", "bisenetv2", "ddrnet", "icnet", "cgnet"}:
            high = feat
            low = F.interpolate(feats[-1], size=feat.shape[-2:], mode="nearest")
            feat = high + self.context(low) + self.context2(high)
        elif self.kind == "dpt-seg":
            feat = feat + self.context3(self.context2(feat))
        return self.cls(feat)


class MMSegSegmenter(nn.Module):
    """Compact MMSeg EncoderDecoder model with family-specific decode heads."""

    def __init__(self, kind: str, classes: int = 7, width: int = 24) -> None:
        """Initialize segmentation model.

        Parameters
        ----------
        kind:
            MMSeg family key.
        classes:
            Output classes.
        width:
            Feature width.
        """

        super().__init__()
        self.kind = kind
        self.backbone = FeaturePyramid(width)
        self.head = ContextHead(kind, width, classes)
        self.aux = nn.Conv2d(width, classes, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Segment an RGB image.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        Tensor
            Upsampled segmentation logits.
        """

        feats = self.backbone(x)
        logits = self.head(feats)
        aux = F.interpolate(self.aux(feats[0]), size=logits.shape[-2:], mode="nearest")
        logits = logits + aux[:, : logits.shape[1]]
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


class MMOCRTextDetector(nn.Module):
    """Compact MMOCR text detector preserving PANet, PSENet, FCENet, and Mask R-CNN heads."""

    def __init__(self, kind: str, width: int = 24) -> None:
        """Initialize text detector.

        Parameters
        ----------
        kind:
            MMOCR detector family key.
        width:
            Feature width.
        """

        super().__init__()
        self.kind = kind
        self.backbone = FeaturePyramid(width)
        self.fpem = nn.ModuleList([ConvBNAct(width, width) for _ in range(4)])
        self.fuse = ConvBNAct(width * 4, width)
        self.text = nn.Conv2d(width, 1, 1)
        self.kernel = nn.Conv2d(width, 6, 1)
        self.fourier = nn.Conv2d(width, 18, 1)
        self.rpn = nn.Conv2d(width, 3, 1)
        self.roi_fc = nn.Linear(width, width)
        self.mask = nn.Conv2d(width, 2, 1)
        self.offset = nn.Conv2d(width, 2, 3, padding=1)

    def _merged_features(self, x: Tensor) -> Tensor:
        """Build MMOCR neck features.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        Tensor
            Fused feature map.
        """

        feats = self.backbone(x)
        size = feats[0].shape[-2:]
        refined = [block(feat) for block, feat in zip(self.fpem, feats, strict=True)]
        return self.fuse(
            torch.cat([F.interpolate(feat, size=size, mode="nearest") for feat in refined], dim=1)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        """Detect text regions.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        tuple[Tensor, ...]
            Family-specific text-detection outputs.
        """

        feat = self._merged_features(x)
        score = self.text(feat)
        if self.kind == "panet":
            embedding = F.normalize(feat[:, :4], dim=1)
            return score, self.kernel(feat), embedding
        if self.kind == "psenet":
            kernels = torch.sigmoid(self.kernel(feat))
            progressive = torch.cummax(kernels, dim=1).values
            return score, progressive
        if self.kind == "fcenet":
            coeff = self.fourier(feat)
            radius = F.softplus(coeff[:, 0::2])
            phase = torch.tanh(coeff[:, 1::2])
            return score, radius, phase
        if self.kind == "fcenet_dcn":
            offset = torch.tanh(self.offset(feat)).permute(0, 2, 3, 1)
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, feat.shape[2], device=x.device),
                torch.linspace(-1, 1, feat.shape[3], device=x.device),
                indexing="ij",
            )
            grid = torch.stack([xx, yy], dim=-1).unsqueeze(0) + 0.1 * offset
            warped = F.grid_sample(feat, grid, align_corners=False)
            coeff = self.fourier(warped)
            return score, F.softplus(coeff[:, 0::2]), torch.tanh(coeff[:, 1::2])
        proposals = torch.softmax(self.rpn(feat).flatten(2), dim=-1)
        pooled = torch.einsum("bcn,bkn->bkc", feat.flatten(2), proposals)
        roi = F.relu(self.roi_fc(pooled)).mean(dim=1).view(x.shape[0], -1, 1, 1)
        return self.rpn(feat), self.mask(feat + roi), score


def build_mmpretrain(kind: str) -> nn.Module:
    """Build a compact MMPreTrain family model.

    Parameters
    ----------
    kind:
        Family key from the catalog row.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return MMPPretrainClassifier(kind)


def build_mmseg(kind: str) -> nn.Module:
    """Build a compact MMSegmentation family model.

    Parameters
    ----------
    kind:
        Segmentation family key from the catalog row.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    normalized = {
        "encoder_decoder": "upernet",
        "beit": "upernet",
        "mae": "upernet",
        "mobilenet_v3": "lraspp",
    }.get(kind, kind)
    return MMSegSegmenter(normalized)


def build_mmocr(kind: str) -> nn.Module:
    """Build a compact MMOCR detector.

    Parameters
    ----------
    kind:
        OCR family key from the catalog row.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return MMOCRTextDetector(kind)


def example_image() -> Tensor:
    """Return an RGB image example input.

    Returns
    -------
    Tensor
        Example image tensor.
    """

    return torch.randn(1, 3, 64, 64)


def notes_for_family(kind: str) -> str:
    """Return provenance note text for a family key.

    Parameters
    ----------
    kind:
        Catalog family key.

    Returns
    -------
    str
        Concise provenance note.
    """

    return (
        f"wave-B base-env faithful compact OpenMMLab reimplementation for {kind}; "
        "random-init PyTorch module preserving the source family load-bearing blocks; "
        "OpenMMLab package path unavailable under torch 2.8/mmcv ABI constraints"
    )


def constructor_for(group: str, kind: str) -> str:
    """Return a catalog constructor statement.

    Parameters
    ----------
    group:
        Builder group, one of ``mmpretrain``, ``mmseg``, or ``mmocr``.
    kind:
        Family key.

    Returns
    -------
    str
        Executable constructor statement assigning ``model``.
    """

    builders = {
        "mmpretrain": "build_mmpretrain",
        "mmseg": "build_mmseg",
        "mmocr": "build_mmocr",
    }
    builder = builders[group]
    return f"from menagerie.classics.wave_b_openmmlab import {builder}; model={builder}({kind!r})"
