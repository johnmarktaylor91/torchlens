# SOURCE: vendored from owkin/HistoSSLscaling @ main
# https://raw.githubusercontent.com/owkin/HistoSSLscaling/main/rl_benchmarks/models/feature_extractors/encoders/vision_transformer.py
# https://raw.githubusercontent.com/owkin/HistoSSLscaling/main/rl_benchmarks/models/feature_extractors/encoders/weight_init.py
# https://raw.githubusercontent.com/owkin/HistoSSLscaling/main/rl_benchmarks/models/feature_extractors/ibot_vit.py
#
# Phikon (Filiot et al. 2023, MICCAI/medRxiv "Scaling Self-Supervised Learning
# for Histopathology with Masked Image Modeling") -- an iBOT-pretrained ViT-B/16
# released by Owkin as `owkin/phikon` on HuggingFace, and via this official
# HistoSSLscaling repo as `iBOTViT`. The architecture is the repo's own
# `VisionTransformer` class (a DINO/timm-derived standard ViT-B/16 with
# LayerScale support and optional mean-pooling head), copied verbatim from
# `encoders/vision_transformer.py` + `encoders/weight_init.py`. The pretrained
# iBOT objective changes only the training procedure, not the architecture, so
# this is the model's real class -- no architecture was rewritten. Only the
# pretrained-checkpoint loading, torchvision transform, and logging plumbing in
# `iBOTViT.__init__` were dropped (irrelevant to the traced architecture);
# `PhikonExtractor.forward` below reproduces `iBOTViT.__call__`'s
# `get_intermediate_layers` + CLS-token-concat feature-extraction path exactly.
#
# Upstream license: owkin/HistoSSLscaling is licensed for non-commercial
# research use (see repo LICENSE.txt); random init only, no pretrained weights
# used here.

from functools import partial
from typing import Callable, List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# --- encoders/weight_init.py -------------------------------------------------


def _no_grad_trunc_normal_(
    tensor: torch.Tensor, mean: float, std: float, a: float, b: float
) -> torch.Tensor:
    """Truncated normal initialization."""

    def _norm_cdf(x: float) -> float:
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        lb = _norm_cdf((a - mean) / std)
        ub = _norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lb - 1, 2 * ub - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> torch.Tensor:
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# --- encoders/vision_transformer.py ------------------------------------------


def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        init_values: int = 0,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        if self.gamma_1 is None:
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class VisionTransformer(nn.Module):
    """Vision Transformer in PyTorch, as implemented in HistoSSLscaling
    (DINO/timm-derived; this is the exact class used by Phikon's iBOTViT)."""

    def __init__(
        self,
        img_size: List[int] = [224],
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 0,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Callable = partial(nn.LayerNorm, eps=1e-6),
        return_all_tokens: bool = False,
        init_values: int = 0,
        use_mean_pooling: bool = False,
        masked_im_modeling: bool = False,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.return_all_tokens = return_all_tokens

        self.patch_embed = PatchEmbed(
            img_size=img_size[0],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        self.masked_im_modeling = masked_im_modeling
        if masked_im_modeling:
            self.masked_embed = nn.Parameter(torch.zeros(1, embed_dim))

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(
                0, 3, 1, 2
            ),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if mask is not None:
            x = self.mask_model(x, mask)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        return self.pos_drop(x)

    def forward(
        self,
        x: torch.Tensor,
        return_all_tokens: Optional[bool] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.masked_im_modeling:
            assert mask is not None
            x = self.prepare_tokens(x, mask=mask)
        else:
            x = self.prepare_tokens(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        if self.fc_norm is not None:
            x[:, 0] = self.fc_norm(x[:, 1:, :].mean(1))

        return_all_tokens = (
            self.return_all_tokens if return_all_tokens is None else return_all_tokens
        )
        if return_all_tokens:
            return x
        return x[:, 0]

    def get_intermediate_layers(self, x: torch.Tensor, n: int = 1) -> List[torch.Tensor]:
        x = self.prepare_tokens(x)
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def vit_base(patch_size: int = 16, **kwargs) -> VisionTransformer:
    """Create a Base Vision Transformer model (ViT-B), as used by Phikon."""
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs,
    )
    return model


# --- ibot_vit.py (PhikonExtractor, trimmed of weight-loading/transform I/O) --


class PhikonExtractor(nn.Module):
    """Phikon's `iBOTViT` feature extractor, minus checkpoint loading and the
    torchvision preprocessing transform (irrelevant to the traced architecture).
    Reproduces `iBOTViT.__call__` exactly: last-block CLS-token features from
    `vit_base(patch_size=16, num_classes=0, use_mean_pooling=False)`."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = vit_base(patch_size=16, num_classes=0, use_mean_pooling=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor.get_intermediate_layers(images, 1)
        features = torch.cat([x[:, 0] for x in features], dim=-1)
        return features


def build_phikon():
    model = PhikonExtractor()
    model.eval()
    return model


def example_input_phikon():
    return (torch.randn(1, 3, 224, 224),)


MENAGERIE_ENTRIES = [
    ("Phikon", "build_phikon", "example_input_phikon", 2023, "vendored-pytorch"),
]
