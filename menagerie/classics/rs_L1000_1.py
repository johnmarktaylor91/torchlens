# SOURCE: vendored from https://github.com/CityMind-Lab/UrbanVLP @ main
#   Vendored files:
#     - open_clip_mine/utils.py     (to_2tuple helper)
#     - open_clip_mine/transformer.py  (CLIP transformer backbone, modified to also
#       return per-token features from encode_image/encode_text -- "#hxx add" markers
#       in the upstream source mark UrbanVLP's own modifications to stock open_clip)
#     - open_clip_mine/model.py     (CLIP class + CLIPVisionCfg/CLIPTextCfg configs,
#       modified encode_image/encode_text/forward to emit image_all_tokens/text_all_tokens)
#     - models/models.py            (MGC multi-granularity fusion head: satellite CLIP +
#       street-view CLIP towers fused via bidirectional cross-attention)
#   The `ipdb` debug import and `torch.utils.checkpoint` grad-checkpointing path from the
#   original files are dropped/inert here (never exercised at inference with
#   grad_checkpointing=False, the class default); everything else is the real architecture
#   code, only re-parameterized to tiny dims for tracing.
#
# UrbanVLP (row: "UrbanVLP", queue candidate L1000) is a multi-granularity vision-language
# fusion model for urban perception: a satellite-image CLIP tower and a street-view-image
# CLIP tower, each producing pooled + per-token embeddings, fused through two
# nn.MultiheadAttention cross-attention layers (satellite attends to street-view text/image
# tokens and vice versa). The traced entry point here is `MGC`, the fusion head that wraps
# two independently-constructed CLIP towers -- this needs no checkpoint file (unlike
# MultiGranularity_GeoCLIP's GeoCLIP_LocationEncoder, which loads
# `location_encoder_weights.pth` in __init__ and therefore can't be constructed from
# random init alone).

import math
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from itertools import repeat
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# open_clip_mine/utils.py
# ---------------------------------------------------------------------------
def _ntuple(n):
    def parse(x):
        if isinstance(x, Sequence):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


# ---------------------------------------------------------------------------
# open_clip_mine/transformer.py
# ---------------------------------------------------------------------------
class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(
            x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps
        )
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchDropout(nn.Module):
    """https://arxiv.org/abs/2212.00794"""

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.0
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        if not self.training or self.prob == 0.0:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        scaled_cosine=False,
        scale_heads=False,
        logit_scale_max=math.log(1.0 / 0.01),
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.logit_scale_max = logit_scale_max

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        else:
            self.in_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        L, N, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        k = k.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        v = v.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)

        if self.logit_scale is not None:
            attn = torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.view(N, self.num_heads, L, L) * logit_scale
            attn = attn.view(-1, L, L)
        else:
            q = q * self.scale
            attn = torch.bmm(q, k.transpose(-1, -2))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            attn += attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.bmm(attn, v)
        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C)
        x = x.transpose(0, 1).reshape(L, N, C)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        is_cross_attention: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ls_1 = (
            LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        )
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(mlp_width, d_model)),
                ]
            )
        )
        self.ls_2 = (
            LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        )

    def attention(
        self,
        q_x: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(
        self,
        q_x: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None

        x = q_x + self.ls_1(
            self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        )
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    width,
                    heads,
                    mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for _ in range(layers)
            ]
        )

    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, "int8_original_dtype"):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            x = r(x, attn_mask=attn_mask)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        ls_init_value: float = None,
        attentional_pool: bool = False,
        attn_pooler_queries: int = 256,
        attn_pooler_heads: int = 8,
        output_dim: int = 512,
        patch_dropout: float = 0.0,
        no_ln_pre: bool = False,
        pos_embed_type: str = "learnable",
        pool_type: str = "tok",
        final_ln_after_pool: bool = False,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        output_tokens: bool = False,
    ):
        super().__init__()
        assert pool_type in ("tok", "avg", "none")
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.final_ln_after_pool = final_ln_after_pool
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False
        )

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        assert pos_embed_type == "learnable"
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width)
        )

        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0.0 else nn.Identity()

        self.ln_pre = nn.Identity() if no_ln_pre else norm_layer(width)
        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        # attentional_pool disabled for this tiny build (matches default UrbanVLP config)
        self.attn_pool = None
        self.pool_type = pool_type

        pool_dim = width
        self.ln_post = norm_layer(pool_dim)
        self.proj = nn.Parameter(scale * torch.randn(pool_dim, output_dim))

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pool_type == "avg":
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == "tok":
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x
        return pooled, tokens

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)

        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        pooled, tokens = self._global_pool(x)

        if self.proj is not None:
            pooled = pooled @ self.proj
            # UrbanVLP modification (upstream "#hxx add"): also project the per-token
            # features so encode_image can return dense tokens alongside the pooled embedding.
            tokens = tokens @ self.proj

        return pooled, tokens


def text_global_pool(x, text: Optional[torch.Tensor] = None, pool_type: str = "argmax"):
    if pool_type == "first":
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == "last":
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == "argmax":
        assert text is not None
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x
    return pooled, tokens


class TextTransformer(nn.Module):
    def __init__(
        self,
        context_length: int = 77,
        vocab_size: int = 49408,
        width: int = 512,
        heads: int = 8,
        layers: int = 12,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        output_dim: int = 512,
        embed_cls: bool = False,
        no_causal_mask: bool = False,
        pad_id: int = 0,
        pool_type: str = "argmax",
        proj_bias: bool = False,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        output_tokens: bool = False,
    ):
        super().__init__()
        assert pool_type in ("first", "last", "argmax", "none")
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pad_id = pad_id
        self.pool_type = pool_type

        self.token_embedding = nn.Embedding(vocab_size, width)
        if embed_cls:
            self.cls_emb = nn.Parameter(torch.empty(width))
            self.num_pos += 1
        else:
            self.cls_emb = None
        self.positional_embedding = nn.Parameter(torch.empty(self.num_pos, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.ln_final = norm_layer(width)

        if no_causal_mask:
            self.attn_mask = None
        else:
            self.register_buffer("attn_mask", self.build_causal_mask(), persistent=False)

        if proj_bias:
            self.text_projection = nn.Linear(width, output_dim)
        else:
            self.text_projection = nn.Parameter(torch.empty(width, output_dim))

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if self.cls_emb is not None:
            nn.init.normal_(self.cls_emb, std=0.01)

        proj_std = (self.transformer.width**-0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                nn.init.normal_(self.text_projection.weight, std=self.transformer.width**-0.5)
                if self.text_projection.bias is not None:
                    nn.init.zeros_(self.text_projection.bias)
            else:
                nn.init.normal_(self.text_projection, std=self.transformer.width**-0.5)

    def build_causal_mask(self):
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def forward(self, text):
        cast_dtype = self.transformer.get_cast_dtype()
        seq_len = text.shape[1]

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        attn_mask = self.attn_mask
        assert self.cls_emb is None  # UrbanVLP default config never sets embed_cls

        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_final(x)
        pooled, tokens = text_global_pool(x, text, pool_type=self.pool_type)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection

        if self.output_tokens:
            return pooled, tokens
        return pooled


# ---------------------------------------------------------------------------
# open_clip_mine/model.py
# ---------------------------------------------------------------------------
@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    ls_init_value: Optional[float] = None
    patch_dropout: float = 0.0
    pos_embed_type: str = "learnable"
    pool_type: str = "tok"
    output_tokens: bool = False


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    mlp_ratio: float = 4.0
    ls_init_value: Optional[float] = None
    embed_cls: bool = False
    pad_id: int = 0
    no_causal_mask: bool = False
    pool_type: str = "argmax"
    proj_bias: bool = False
    output_tokens: bool = False


def _build_vision_tower(embed_dim: int, vision_cfg: CLIPVisionCfg) -> VisionTransformer:
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)
    act_layer = nn.GELU
    vision_heads = vision_cfg.width // vision_cfg.head_width
    norm_layer = LayerNorm
    return VisionTransformer(
        image_size=vision_cfg.image_size,
        patch_size=vision_cfg.patch_size,
        width=vision_cfg.width,
        layers=vision_cfg.layers,
        heads=vision_heads,
        mlp_ratio=vision_cfg.mlp_ratio,
        ls_init_value=vision_cfg.ls_init_value,
        patch_dropout=vision_cfg.patch_dropout,
        pos_embed_type=vision_cfg.pos_embed_type,
        pool_type=vision_cfg.pool_type,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )


def _build_text_tower(embed_dim: int, text_cfg: CLIPTextCfg) -> TextTransformer:
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)
    act_layer = nn.GELU
    norm_layer = LayerNorm
    return TextTransformer(
        context_length=text_cfg.context_length,
        vocab_size=text_cfg.vocab_size,
        width=text_cfg.width,
        heads=text_cfg.heads,
        layers=text_cfg.layers,
        mlp_ratio=text_cfg.mlp_ratio,
        ls_init_value=text_cfg.ls_init_value,
        output_dim=embed_dim,
        embed_cls=text_cfg.embed_cls,
        no_causal_mask=text_cfg.no_causal_mask,
        pad_id=text_cfg.pad_id,
        pool_type=text_cfg.pool_type,
        proj_bias=text_cfg.proj_bias,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )


class CLIP(nn.Module):
    """UrbanVLP's modified open_clip CLIP tower: encode_image/encode_text return
    (pooled_features, all_tokens) tuples so downstream fusion (MGC) can attend over
    dense per-token features, not just the pooled embedding."""

    output_dict: torch.jit.Final[bool]

    def __init__(
        self,
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        text_cfg: CLIPTextCfg,
        init_logit_scale: float = math.log(1 / 0.07),
        output_dict: bool = True,
    ):
        super().__init__()
        self.output_dict = output_dict

        self.visual = _build_vision_tower(embed_dim, vision_cfg)

        text = _build_text_tower(embed_dim, text_cfg)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.text_pool_type = text.pool_type
        self.register_buffer("attn_mask", text.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        self.logit_bias = None

    def encode_image(self, image, normalize: bool = False):
        features, tokens = self.visual(image)  # UrbanVLP modification: visual returns tokens too
        all_tokens = torch.cat((features.unsqueeze(dim=1), tokens), dim=1)
        return (
            F.normalize(features, dim=-1) if normalize else features,
            F.normalize(all_tokens, dim=-1) if normalize else all_tokens,
        )

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        x, tokens = text_global_pool(x, text, self.text_pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection
                tokens = tokens @ self.text_projection  # UrbanVLP modification

        return (
            F.normalize(x, dim=-1) if normalize else x,
            F.normalize(tokens, dim=-1) if normalize else tokens,
        )

    def forward(self, image: Optional[torch.Tensor] = None, text: Optional[torch.Tensor] = None):
        image_features, image_all_tokens = (
            self.encode_image(image, normalize=True) if image is not None else (None, None)
        )
        text_features, text_all_tokens = (
            self.encode_text(text, normalize=True) if text is not None else (None, None)
        )

        out_dict = {
            "image_features": image_features,
            "text_features": text_features,
            "logit_scale": self.logit_scale.exp(),
            "image_all_tokens": image_all_tokens,
            "text_all_tokens": text_all_tokens,
        }
        return out_dict


# ---------------------------------------------------------------------------
# models/models.py -- MGC multi-granularity fusion head
# ---------------------------------------------------------------------------
class MGC(nn.Module):
    """UrbanVLP's fusion head: satellite CLIP tower + street-view CLIP tower, fused via
    two bidirectional nn.MultiheadAttention cross-attention layers (satellite<->streetview,
    each attending over the other's token stream)."""

    def __init__(
        self,
        clip_model_satellite,
        clip_model_streetview,
        attn_embed_dim=768,
        num_heads=8,
    ):
        super().__init__()
        self.model_satellite = clip_model_satellite
        self.model_streetview = clip_model_streetview
        self.fc_layer = nn.Linear(25, 1)

        self.visual_crossattn_layer = nn.MultiheadAttention(
            embed_dim=attn_embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.text_crossattn_layer = nn.MultiheadAttention(
            embed_dim=attn_embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, images, texts, streetview_images, streetview_texts):
        satellite_model_out = self.model_satellite(images, texts)

        # streetview_images: (B, T=25, C, H, W) -> fuse the T street-view panorama frames
        # down to one frame via a learned linear combination over the temporal axis.
        streetview_images = self.fc_layer(streetview_images.permute(0, 2, 3, 4, 1)).squeeze(-1)

        streetview_model_out = self.model_streetview(streetview_images, streetview_texts)
        satellite_model_out["image_all_tokens"] = (
            satellite_model_out["image_all_tokens"] + streetview_model_out["image_all_tokens"]
        )

        satellite_visual_atten_output, _ = self.visual_crossattn_layer(
            query=satellite_model_out["image_all_tokens"],
            key=satellite_model_out["text_all_tokens"],
            value=satellite_model_out["text_all_tokens"],
        )
        satellite_text_atten_output, _ = self.text_crossattn_layer(
            query=satellite_model_out["text_all_tokens"],
            key=satellite_model_out["image_all_tokens"],
            value=satellite_model_out["image_all_tokens"],
        )
        streetview_visual_atten_output, _ = self.visual_crossattn_layer(
            query=streetview_model_out["image_all_tokens"],
            key=streetview_model_out["text_all_tokens"],
            value=streetview_model_out["text_all_tokens"],
        )
        streetview_text_atten_output, _ = self.text_crossattn_layer(
            query=streetview_model_out["text_all_tokens"],
            key=streetview_model_out["image_all_tokens"],
            value=streetview_model_out["image_all_tokens"],
        )
        streetview_model_out["streetview_visual_atten_output"] = streetview_visual_atten_output
        streetview_model_out["streetview_text_atten_output"] = streetview_text_atten_output
        satellite_model_out["satellite_visual_atten_output"] = satellite_visual_atten_output
        satellite_model_out["satellite_text_atten_output"] = satellite_text_atten_output

        return satellite_model_out, streetview_model_out


# ---------------------------------------------------------------------------
# Menagerie staging hooks
# ---------------------------------------------------------------------------
_ATTN_EMBED_DIM = 64
_NUM_HEADS = 4
_IMAGE_SIZE = 32
_PATCH_SIZE = 16
_CONTEXT_LENGTH = 8
_VOCAB_SIZE = 64
_N_FRAMES = 5  # tiny stand-in for the real 25-frame street-view panorama stack


def _tiny_clip_cfg():
    vision_cfg = CLIPVisionCfg(
        layers=1,
        width=_ATTN_EMBED_DIM,
        head_width=_ATTN_EMBED_DIM // _NUM_HEADS,
        patch_size=_PATCH_SIZE,
        image_size=_IMAGE_SIZE,
        pool_type="tok",
    )
    text_cfg = CLIPTextCfg(
        context_length=_CONTEXT_LENGTH,
        vocab_size=_VOCAB_SIZE,
        width=_ATTN_EMBED_DIM,
        heads=_NUM_HEADS,
        layers=1,
        pool_type="argmax",
    )
    return vision_cfg, text_cfg


def build_urbanvlp_mgc():
    vision_cfg, text_cfg = _tiny_clip_cfg()
    clip_satellite = CLIP(embed_dim=_ATTN_EMBED_DIM, vision_cfg=vision_cfg, text_cfg=text_cfg)

    # MGC's fc_layer is nn.Linear(25, 1): the street-view tower must accept the SAME
    # spatial size as the satellite tower (25-frame collapse -> single image via fc_layer).
    clip_streetview = CLIP(embed_dim=_ATTN_EMBED_DIM, vision_cfg=vision_cfg, text_cfg=text_cfg)

    model = MGC(
        clip_model_satellite=clip_satellite,
        clip_model_streetview=clip_streetview,
        attn_embed_dim=_ATTN_EMBED_DIM,
        num_heads=_NUM_HEADS,
    )
    model.fc_layer = nn.Linear(_N_FRAMES, 1)
    return model


def example_input_urbanvlp_mgc():
    batch = 1
    images = torch.randn(batch, 3, _IMAGE_SIZE, _IMAGE_SIZE)
    texts = torch.randint(0, _VOCAB_SIZE, (batch, _CONTEXT_LENGTH))
    streetview_images = torch.randn(batch, _N_FRAMES, 3, _IMAGE_SIZE, _IMAGE_SIZE)
    streetview_texts = torch.randint(0, _VOCAB_SIZE, (batch, _CONTEXT_LENGTH))
    return (images, texts, streetview_images, streetview_texts)


MENAGERIE_ENTRIES = [
    ("UrbanVLP", build_urbanvlp_mgc, example_input_urbanvlp_mgc, 2024, "MENAGERIE_ZOO"),
]
