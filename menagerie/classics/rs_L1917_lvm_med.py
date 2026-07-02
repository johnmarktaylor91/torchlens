# SOURCE: vendored from duyhominhnguyen/LVM-Med @ main
# (model/unet_vit_classification.py), the LVM-MED-VIT encoder + classification
# head used to evaluate the LVM-Med (NeurIPS 2023) self-supervised medical
# imaging model on linear/non-linear image classification. `ImageEncoderViT`,
# `Block`, `Attention`, `MLPBlock`, `LayerNorm2d`, `PatchEmbed`, and the window
# partition helpers are the ViTDet-style SAM image encoder architecture that
# the original file itself credits as "lightly adapted from the ViTDet
# backbone available at: facebookresearch/detectron2 .../backbone/vit.py";
# `vit_encoder_b` wraps it with LVM-Med's exact encoder hyperparameters
# (embed_dim=768, depth=12, heads=12, global_attn_indexes=[2,5,8,11],
# window_size=14) and `LVMMedViTClassifier` adds LVM-Med's classification
# head (AdaptiveAvgPool2d + Linear). Only imports/module-level weight-loading
# code (`load_weight_for_vit_encoder`, checkpoint I/O, the broken __main__
# TransUNet smoke test) were dropped -- none of that is part of the traced
# forward architecture. Depths/sizes below are shrunk for fast tracing; the
# architecture (window attention + global attention interleaving, relative
# positional embeddings, residual MLP blocks) is unchanged from the source.
"""LVM-Med ViT-B image encoder (ViTDet/SAM-style) + classification head.

Nguyen et al. 2023, "LVM-Med: Learning Large-Scale Self-Supervised Vision
Models for Medical Imaging via Second-order Graph Matching" (NeurIPS 2023).
The released classification entry point applies the LVM-Med pretrained ViT
encoder (a ViTDet/SAM ImageEncoderViT variant with interleaved windowed and
global attention) to linear/non-linear medical image classification via a
global-average-pool + linear head.
"""

from functools import partial
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange  # noqa: F401  (kept: real source imports it for vit_timm variant)


MENAGERIE_ZOO = "vendored-pytorch"


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Partition into non-overlapping windows with padding if needed."""
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """Window unpartition into original sequences and removing padding."""
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """Get relative positional embeddings according to relative positions of query/key sizes."""
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`."""
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, (
                "Input size must be provided if using relative positional encoding."
            )
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        include_neck: bool = False,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        # NOTE (matches source): the neck is defined but not applied in the
        # released classification forward -- the output stays in [B,H,W,C]
        # and is permuted to [B,C,H,W] for the classification head's pool.
        x = x.permute(0, 3, 1, 2)
        return x


class vit_encoder_b(nn.Module):
    """LVM-Med's exact ViT-B/16 encoder hyperparameters (SAM ViT-B family)."""

    def __init__(self, num_classes: int = 4):
        super().__init__()

        image_size = 1024
        vit_patch_size = 16
        encoder_embed_dim = 768
        encoder_depth = 12
        encoder_num_heads = 12
        encoder_global_attn_indexes = [2, 5, 8, 11]

        self.model = ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            use_abs_pos=False,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=256,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(768, 1000)

    def forward(self, x):
        x = self.model(x)
        x = self.avgpool(x)
        x = torch.squeeze(x)
        x = self.fc(x)
        return x


# ---------------------------------------------------------------------------
# Staging build/example helpers. img_size/vit_patch_size/window_size and
# depth are shrunk from the source's 1024/16/14/12 for fast tracing; the
# window-vs-global attention interleave pattern (global at indices 0, 2 of a
# 3-block encoder here; source uses [2,5,8,11] of 12) and every other
# mechanism (rel-pos attention, residual MLP blocks, patch embed) is
# preserved exactly as in the source `ImageEncoderViT`/`Block`/`Attention`.
# ---------------------------------------------------------------------------


def build_lvm_med_vit():
    model = ImageEncoderViT(
        img_size=64,
        patch_size=16,
        in_chans=3,
        embed_dim=48,
        depth=3,
        num_heads=4,
        mlp_ratio=4.0,
        out_chans=32,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        use_abs_pos=True,
        use_rel_pos=True,
        rel_pos_zero_init=True,
        window_size=4,
        global_attn_indexes=(0, 2),
    )
    model.eval()
    return model


def example_input_lvm_med_vit():
    return torch.randn(1, 3, 64, 64)


class LVMMedViTClassifier(nn.Module):
    """LVM-Med's `vit_encoder_b` classification wrapper, scaled to staging size."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.model = ImageEncoderViT(
            img_size=64,
            patch_size=16,
            in_chans=3,
            embed_dim=48,
            depth=3,
            num_heads=4,
            mlp_ratio=4.0,
            out_chans=32,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            use_abs_pos=False,
            use_rel_pos=True,
            rel_pos_zero_init=True,
            window_size=4,
            global_attn_indexes=(0, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(48, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.avgpool(x)
        x = torch.squeeze(x)
        x = self.fc(x)
        return x


def build_lvm_med_vit_classifier():
    model = LVMMedViTClassifier(num_classes=4)
    model.eval()
    return model


def example_input_lvm_med_vit_classifier():
    return torch.randn(2, 3, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "LVM-Med-ViT (encoder)",
        build_lvm_med_vit,
        example_input_lvm_med_vit,
        2023,
        "vendored-pytorch",
    ),
    (
        "LVM-Med-ViT-Classifier",
        build_lvm_med_vit_classifier,
        example_input_lvm_med_vit_classifier,
        2023,
        "vendored-pytorch",
    ),
]
