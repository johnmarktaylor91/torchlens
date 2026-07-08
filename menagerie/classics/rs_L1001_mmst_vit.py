# SOURCE: vendored from fudong03/MMST-ViT @ 615666c8d9fcd704acb662c13065703cbf2eab70
# (attention.py, models_pvt.py, models_pvt_simclr.py, models_mmst_vit.py)
#
# MMST-ViT: Multi-Modal Spatial-Temporal Vision Transformer for crop yield prediction
# (ICCV 2023). A PVT (Pyramid Vision Transformer) backbone extracts per-grid-cell visual
# features from satellite imagery, fused with short-term weather context via cross-modal
# attention (PVTSimCLR), then aggregated across space (SpatialTransformer) and time
# (TemporalTransformer, conditioned on long-term meteorological context) to predict yield.
# Classes below (Mlp/Attention/Block/PatchEmbed/PyramidVisionTransformer from models_pvt.py;
# GEGLU/FeedForward/PreNorm/MultiModalAttention/SpatialAttention/TemporalAttention/
# MultiModalTransformer/SpatialTransformer/TemporalTransformer from attention.py;
# PVTSimCLR from models_pvt_simclr.py; MMST_ViT from models_mmst_vit.py) are the REAL model
# code, copied verbatim (only base-lib deps: torch, einops, timm -- all installed). The
# `timm.models.layers` import path used by the original repo is a deprecated-but-functional
# alias in the installed timm; kept as in the source to stay faithful to the real code (it
# routes to `timm.layers` internally). timm's @register_model decorator on pvt_tiny is
# dropped (pure model-registry side effect requiring the defining module to be present in
# sys.modules; irrelevant since we call pvt_tiny() directly, not through the registry).

import math
from functools import partial
from inspect import isfunction

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg
from torch import einsum, nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# attention.py
# ---------------------------------------------------------------------------


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MultiModalAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class SpatialAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        _, _, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class TemporalAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, bias=None):
        _, _, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b t (h d) -> b h t d", h=h), qkv)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        if exists(bias):
            bias = self.to_qkv(bias).chunk(3, dim=-1)
            qb, kb, _ = map(lambda t: rearrange(t, "b t (h d) -> b h t d", h=h), bias)
            bias = einsum("b h i d, b h j d -> b h i j", qb, kb) * self.scale
            dots += bias

        attn = dots.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class MultiModalTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, context_dim=9, mult=4, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            MultiModalAttention(
                                dim,
                                context_dim=context_dim,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, dim_out=dim, mult=mult, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x, context=None):
        for attn, ff in self.layers:
            x = attn(x, context=context) + x
            x = ff(x) + x
        return self.norm(x)


class SpatialTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mult=4, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            SpatialAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        ),
                        PreNorm(dim, FeedForward(dim, dim_out=dim, mult=mult, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class TemporalTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mult=4, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            TemporalAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        ),
                        PreNorm(dim, FeedForward(dim, dim_out=dim, mult=mult, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x, bias=None):
        for attn, ff in self.layers:
            x = attn(x, bias=bias) + x
            x = ff(x) + x
        return self.norm(x)


# ---------------------------------------------------------------------------
# models_pvt.py
# ---------------------------------------------------------------------------


class PvtMlp(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PvtAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = (
                self.kv(x_)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        else:
            kv = (
                self.kv(x)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class PvtBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PvtAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = PvtMlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


class PyramidVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dims=[64, 128, 256, 512],
        num_heads=[1, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        num_stages=4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = PatchEmbed(
                img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                patch_size=patch_size if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
            )
            num_patches = (
                patch_embed.num_patches if i != num_stages - 1 else patch_embed.num_patches + 1
            )
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[i]))
            pos_drop = nn.Dropout(p=drop_rate)

            block = nn.ModuleList(
                [
                    PvtBlock(
                        dim=embed_dims[i],
                        num_heads=num_heads[i],
                        mlp_ratio=mlp_ratios[i],
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[cur + j],
                        norm_layer=norm_layer,
                        sr_ratio=sr_ratios[i],
                    )
                    for j in range(depths[i])
                ]
            )
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"pos_drop{i + 1}", pos_drop)
            setattr(self, f"block{i + 1}", block)

        self.norm = norm_layer(embed_dims[3])

        # cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # init weights
        for i in range(num_stages):
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            trunc_normal_(pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return (
                F.interpolate(
                    pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                    size=(H, W),
                    mode="bilinear",
                )
                .reshape(1, -1, H * W)
                .permute(0, 2, 1)
            )

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            pos_drop = getattr(self, f"pos_drop{i + 1}")
            block = getattr(self, f"block{i + 1}")
            x, (H, W) = patch_embed(x)

            if i == self.num_stages - 1:
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                pos_embed_ = self._get_pos_embed(pos_embed[:, 1:], patch_embed, H, W)
                pos_embed = torch.cat((pos_embed[:, 0:1], pos_embed_), dim=1)
            else:
                pos_embed = self._get_pos_embed(pos_embed, patch_embed, H, W)

            x = pos_drop(x + pos_embed)
            for blk in block:
                x = blk(x, H, W)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = self.norm(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x[:, 0]
        x = self.head(x)

        return x


def pvt_tiny(pretrained=False, **kwargs):
    # NOTE: source decorates this with timm's @register_model (registry side-effect only,
    # requires the defining module to be present in sys.modules -- irrelevant here since we
    # call pvt_tiny() directly rather than through timm's model registry). Dropped; the
    # architecture is unchanged.
    model = PyramidVisionTransformer(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        **kwargs,
    )
    model.default_cfg = _cfg()

    return model


# ---------------------------------------------------------------------------
# models_pvt_simclr.py
# ---------------------------------------------------------------------------


class PVTSimCLR(nn.Module):
    def __init__(
        self,
        base_model,
        out_dim=512,
        context_dim=9,
        num_head=8,
        mm_depth=2,
        dropout=0.0,
        pretrained=False,
        gated_ff=True,
    ):
        super(PVTSimCLR, self).__init__()

        # NOTE: source dispatches via `models_pvt.__dict__[base_model]`; inlined for the
        # tiny build below since we only need `pvt_tiny`.
        assert base_model == "pvt_tiny"
        self.backbone = pvt_tiny(pretrained=pretrained)
        num_ftrs = self.backbone.head.in_features

        self.proj = nn.Linear(num_ftrs, out_dim)

        self.proj_context = nn.Linear(context_dim, out_dim)

        # attention
        dim_head = out_dim // num_head
        self.mm_transformer = MultiModalTransformer(
            out_dim, mm_depth, num_head, dim_head, context_dim=out_dim, dropout=dropout
        )

        self.norm1 = nn.LayerNorm(context_dim)

    def forward(self, x, context=None):
        h = self.backbone.forward_features(x)  # shape = B, N, D
        h = h.squeeze()

        # project to targeted dim
        x = self.proj(h)
        context = self.proj_context(self.norm1(context))

        # multi-modal attention
        x = self.mm_transformer(x, context=context)

        # return the classification token
        return x[:, 0]


# ---------------------------------------------------------------------------
# models_mmst_vit.py
# ---------------------------------------------------------------------------


class MMST_ViT(nn.Module):
    def __init__(
        self,
        out_dim=2,
        num_grid=64,
        num_short_term_seq=6,
        num_long_term_seq=12,
        num_year=5,
        pvt_backbone=None,
        context_dim=9,
        dim=192,
        batch_size=64,
        depth=4,
        heads=3,
        pool="cls",
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        scale_dim=4,
    ):
        super().__init__()

        assert pool in {"cls", "mean"}, (
            "pool type must be either cls (cls token) or mean (mean pooling)"
        )

        self.batch_size = batch_size
        self.pvt_backbone = pvt_backbone

        self.proj_context = nn.Linear(
            num_year * num_long_term_seq * context_dim, num_short_term_seq * dim
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_short_term_seq, num_grid, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = SpatialTransformer(
            dim, depth, heads, dim_head, mult=scale_dim, dropout=dropout
        )

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = TemporalTransformer(
            dim, depth, heads, dim_head, mult=scale_dim, dropout=dropout
        )

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.norm1 = nn.LayerNorm(dim)

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, out_dim))

    def forward_features(self, x, ys):
        x = rearrange(x, "b t g c h w -> (b t g) c h w")
        ys = rearrange(ys, "b t g n d -> (b t g) n d")

        # prevent the number of grids from being too large to cause out of memory
        B = x.shape[0]
        n = B // self.batch_size if B % self.batch_size == 0 else B // self.batch_size + 1

        x_hat = torch.empty(0).to(x.device)
        for i in range(n):
            start, end = i * self.batch_size, (i + 1) * self.batch_size
            x_tmp = x[start:end]
            ys_tmp = ys[start:end]

            x_hat_tmp = self.pvt_backbone(x_tmp, context=ys_tmp)
            x_hat = torch.cat([x_hat, x_hat_tmp], dim=0)

        return x_hat

    def forward(self, x, ys=None, yl=None):
        b, t, g, _, _, _ = x.shape
        x = self.forward_features(x, ys)
        x = rearrange(x, "(b t g) d -> b t g d", b=b, t=t, g=g)

        cls_space_tokens = repeat(self.space_token, "() g d -> b t g d", b=b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, : (g + 1)]
        x = self.dropout(x)

        x = rearrange(x, "b t g d -> (b t) g d")
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], "(b t) ... -> b t ...", b=b)

        cls_temporal_tokens = repeat(self.temporal_token, "() t d -> b t d", b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        # concatenate parameters in different months
        yl = rearrange(yl, "b y m d -> b (y m d)")
        yl = self.proj_context(yl)
        yl = rearrange(yl, "b (t d) -> b t d", t=t)

        yl = torch.cat((cls_temporal_tokens, yl), dim=1)
        yl = self.norm1(yl)

        x = self.temporal_transformer(x, yl)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        return self.mlp_head(x)


# ---------------------------------------------------------------------------
# staging build/example (tiny sizes for tracing)
# ---------------------------------------------------------------------------


def build_mmst_vit():
    """Tiny MMST-ViT for tracing: small PVT backbone dims, few grids/timesteps."""
    pvt = PVTSimCLR("pvt_tiny", out_dim=16, context_dim=9, num_head=2, mm_depth=1, pretrained=False)
    model = MMST_ViT(
        out_dim=4,
        num_grid=4,
        num_short_term_seq=2,
        num_long_term_seq=2,
        num_year=1,
        pvt_backbone=pvt,
        context_dim=9,
        dim=16,
        batch_size=8,
        depth=1,
        heads=2,
        dim_head=8,
    )
    model.eval()
    return model


def example_input_mmst_vit():
    # x.shape = B, T, G, C, H, W ; ys.shape = B, T, G, N1, d ; yl.shape = B, Y, M, d
    # returned as a positional tuple (x, ys, yl) -- forward(self, x, ys=None, yl=None)
    # accepts these positionally, matching tl.trace(model, example_input()) semantics.
    x = torch.randn(1, 2, 3, 3, 32, 32)
    ys = torch.randn(1, 2, 3, 5, 9)
    yl = torch.randn(1, 1, 2, 9)
    return (x, ys, yl)


MENAGERIE_ENTRIES = [
    ("MMST-ViT", build_mmst_vit, example_input_mmst_vit, 2023, "vendored-pytorch"),
]
