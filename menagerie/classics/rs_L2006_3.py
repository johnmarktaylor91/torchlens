# SOURCE: vendored from https://github.com/saic-fi/edgevit @ master (src/edgevit.py)
#
# EdgeViT (Pan et al., ECCV 2022) - a local-global-local (LGL) attention
# bottleneck hybrid CNN/ViT for edge devices. Official Samsung AI Centre
# repo. Copied verbatim; only fixes: `_cfg` comes from timm's deprecated
# `timm.models.vision_transformer` shim path (still present in the installed
# timm, emits a FutureWarning but works); the repo's `@register_model`
# decorators are dropped (timm's model-registry decorator requires the
# defining function's module to already be registered in `sys.modules`,
# which does not hold for a module loaded via `importlib.util` -- we don't
# use timm's registry lookup here anyway, only the constructor functions
# directly, so dropping the decorator changes nothing architecturally).

from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg

MENAGERIE_ZOO = "vendored-pytorch"


class Mlp(nn.Module):
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


class CMlp(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalSparseAttn(nn.Module):
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
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr = sr_ratio
        if self.sr > 1:
            self.sampler = nn.AvgPool2d(1, sr_ratio)
            kernel_size = sr_ratio
            self.LocalProp = nn.ConvTranspose2d(dim, dim, kernel_size, stride=sr_ratio, groups=dim)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sampler = nn.Identity()
            self.upsample = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x, H: int, W: int):
        B, N, C = x.shape
        if self.sr > 1.0:
            x = x.transpose(1, 2).reshape(B, C, H, W)
            x = self.sampler(x)
            x = x.flatten(2).transpose(1, 2)

        qkv = (
            self.qkv(x)
            .reshape(B, -1, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)

        if self.sr > 1:
            x = x.permute(0, 2, 1).reshape(B, C, int(H / self.sr), int(W / self.sr))
            x = self.LocalProp(x)
            x = x.reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LocalAgg(nn.Module):
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
    ):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SelfAttn(nn.Module):
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
        sr_ratio=1.0,
    ):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = GlobalSparseAttn(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, N, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x


class LGLBlock(nn.Module):
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
        sr_ratio=1.0,
    ):
        super().__init__()

        if sr_ratio > 1:
            self.LocalAgg = LocalAgg(
                dim,
                num_heads,
                mlp_ratio,
                qkv_bias,
                qk_scale,
                drop,
                attn_drop,
                drop_path,
                act_layer,
                norm_layer,
            )
        else:
            self.LocalAgg = nn.Identity()

        self.SelfAttn = SelfAttn(
            dim,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop,
            attn_drop,
            drop_path,
            act_layer,
            norm_layer,
            sr_ratio,
        )

    def forward(self, x):
        x = self.LocalAgg(x)
        x = self.SelfAttn(x)
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        )
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


class EdgeVit(nn.Module):
    """A PyTorch impl of the EdgeViT LGL (local-global-local) hybrid backbone."""

    def __init__(
        self,
        depth=[1, 2, 5, 3],
        img_size=224,
        in_chans=3,
        num_classes=1000,
        embed_dim=[48, 96, 240, 384],
        head_dim=64,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        representation_size=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        sr_ratios=[4, 2, 2, 1],
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed1 = PatchEmbed(
            img_size=img_size, patch_size=4, in_chans=in_chans, embed_dim=embed_dim[0]
        )
        self.patch_embed2 = PatchEmbed(
            img_size=img_size // 4, patch_size=2, in_chans=embed_dim[0], embed_dim=embed_dim[1]
        )
        self.patch_embed3 = PatchEmbed(
            img_size=img_size // 8, patch_size=2, in_chans=embed_dim[1], embed_dim=embed_dim[2]
        )
        self.patch_embed4 = PatchEmbed(
            img_size=img_size // 16, patch_size=2, in_chans=embed_dim[2], embed_dim=embed_dim[3]
        )

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        num_heads = [dim // head_dim for dim in embed_dim]
        self.blocks1 = nn.ModuleList(
            [
                LGLBlock(
                    dim=embed_dim[0],
                    num_heads=num_heads[0],
                    mlp_ratio=mlp_ratio[0],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0],
                )
                for i in range(depth[0])
            ]
        )
        self.blocks2 = nn.ModuleList(
            [
                LGLBlock(
                    dim=embed_dim[1],
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratio[1],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i + depth[0]],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[1],
                )
                for i in range(depth[1])
            ]
        )
        self.blocks3 = nn.ModuleList(
            [
                LGLBlock(
                    dim=embed_dim[2],
                    num_heads=num_heads[2],
                    mlp_ratio=mlp_ratio[2],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i + depth[0] + depth[1]],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[2],
                )
                for i in range(depth[2])
            ]
        )
        self.blocks4 = nn.ModuleList(
            [
                LGLBlock(
                    dim=embed_dim[3],
                    num_heads=num_heads[3],
                    mlp_ratio=mlp_ratio[3],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i + depth[0] + depth[1] + depth[2]],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[3],
                )
                for i in range(depth[3])
            ]
        )
        self.norm = nn.BatchNorm2d(embed_dim[-1])

        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict(
                    [
                        ("fc", nn.Linear(embed_dim, representation_size)),
                        ("act", nn.Tanh()),
                    ]
                )
            )
        else:
            self.pre_logits = nn.Identity()

        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()

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
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x)
        x = self.patch_embed4(x)
        for blk in self.blocks4:
            x = blk(x)
        x = self.norm(x)
        x = self.pre_logits(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.flatten(2).mean(-1)
        x = self.head(x)
        return x


def edgevit_xxs(pretrained=False, **kwargs):
    model = EdgeVit(
        depth=[1, 1, 3, 2],
        embed_dim=[36, 72, 144, 288],
        head_dim=36,
        mlp_ratio=[4] * 4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        sr_ratios=[4, 2, 2, 1],
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


def edgevit_xs(pretrained=False, **kwargs):
    model = EdgeVit(
        depth=[1, 1, 3, 1],
        embed_dim=[48, 96, 240, 384],
        head_dim=48,
        mlp_ratio=[4] * 4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        sr_ratios=[4, 2, 2, 1],
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


def edgevit_s(pretrained=False, **kwargs):
    model = EdgeVit(
        depth=[1, 2, 5, 3],
        embed_dim=[48, 96, 240, 384],
        head_dim=48,
        mlp_ratio=[4] * 4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        sr_ratios=[4, 2, 2, 1],
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


def build_edgevit_xxs():
    return edgevit_xxs(pretrained=False, img_size=224, num_classes=10)


def example_edgevit_input():
    return torch.randn(1, 3, 224, 224)


MENAGERIE_ENTRIES = [
    ("EdgeViT_XXS", "build_edgevit_xxs", "example_edgevit_input", "2022", "CV"),
]
