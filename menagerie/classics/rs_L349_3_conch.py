# SOURCE: vendored from mahmoodlab/CONCH @ main (conch/open_clip_custom/{transformer,vision_tower,coca_model}.py)
"""CONCH: vision-language foundation model for histopathology, pretrained on
1.17M image-caption pairs with contrastive + captioning objectives (Nature
Medicine 2024). The real architecture is an open_clip-style CoCa: a timm
``VisionTransformer`` trunk wrapped by a ``VisualModel`` head, a custom
``TextTransformer`` for the contrastive text branch, and a
``MultimodalTransformer`` text decoder with cross-attention to image tokens
for the captioning branch.

Code below is copied verbatim from the official repo's three
``open_clip_custom`` modules (only the deprecated ``timm.models.layers``
import path is swapped for the current ``timm.layers`` path -- same objects,
no architecture change -- and the HF-``transformers``-gated ``generate()``
helper's lazy imports are left as in the original). Architecture logic is
untouched.
"""

import logging
import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from timm.layers import Mlp, to_2tuple
from timm.models.vision_transformer import VisionTransformer
from torchvision.ops.misc import FrozenBatchNorm2d

MENAGERIE_ZOO = "vendored-pytorch"


# --- conch/open_clip_custom/utils.py (freeze_batch_norm_2d only) -----------


def freeze_batch_norm_2d(module, module_match={}, name=""):
    res = module
    is_match = True
    if module_match:
        is_match = name in module_match
    if is_match and isinstance(
        module, (nn.modules.batchnorm.BatchNorm2d, nn.modules.batchnorm.SyncBatchNorm)
    ):
        res = FrozenBatchNorm2d(module.num_features)
        res.num_features = module.num_features
        res.affine = module.affine
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for child_name, child in module.named_children():
            full_child_name = ".".join([name, child_name]) if name else child_name
            new_child = freeze_batch_norm_2d(child, module_match, full_child_name)
            if new_child is not child:
                res.add_module(child_name, new_child)
    return res


# --- conch/open_clip_custom/transformer.py ----------------------------------


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
    """
    https://arxiv.org/abs/2212.00794
    """

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


class AttentionalPooler(nn.Module):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        n_head: int = 8,
        n_queries: int = 256,
        norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim)
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = self.ln_k(x).permute(1, 0, 2)  # NLD -> LND
        N = x.shape[1]
        q = self.ln_q(self.query)
        # For a binary mask, a ``True`` value indicates that the
        # corresponding position is not allowed to attend.
        if attn_mask is not None:
            attn_mask = ~attn_mask.bool()
        out = self.attn(self._repeat(q, N), x, x, need_weights=False, key_padding_mask=attn_mask)[0]
        return out.permute(1, 0, 2)  # LND -> NLD

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)


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
        self.heads = heads

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
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x


class TextTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
        self,
        context_length: int = 77,
        vocab_size: int = 49408,
        width: int = 512,
        heads: int = 8,
        layers: int = 12,
        ls_init_value: float = None,
        output_dim: int = 512,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        embed_cls: bool = False,
        pad_id: int = 0,
        output_tokens: bool = False,
    ):
        super().__init__()
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pad_id = pad_id

        self.text_projection = nn.Parameter(torch.empty(width, output_dim))

        if embed_cls:
            self.cls_emb = nn.Parameter(torch.empty(width))
        else:
            self.cls_emb = None

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(self.num_pos, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.ln_final = norm_layer(width)

        self.register_buffer("attn_mask", self.build_attention_mask(), persistent=False)

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
            nn.init.normal_(self.text_projection, std=self.transformer.width**-0.5)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def build_attention_mask(self):
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def build_cls_mask(self, text, cast_dtype: torch.dtype):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=1.0)
        additive_mask = torch.empty(cls_mask.shape, dtype=cast_dtype, device=cls_mask.device)
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = torch.repeat_interleave(additive_mask, self.heads, 0)
        return additive_mask

    def _repeat(self, t, N: int):
        return t.reshape(1, 1, -1).repeat(N, 1, 1)

    def forward(self, text):
        cast_dtype = self.transformer.get_cast_dtype()
        seq_len = text.shape[1]
        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        attn_mask = self.attn_mask
        if self.cls_emb is not None:
            seq_len += 1
            x = torch.cat([x, self._repeat(self.cls_emb, x.shape[0])], dim=1)
            cls_mask = self.build_cls_mask(text, cast_dtype)
            attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if self.cls_emb is not None:
            pooled, tokens = x[:, -1], x[:, :-1]
            pooled = self.ln_final(pooled)
        else:
            x = self.ln_final(x)
            pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x

        if self.text_projection is not None:
            pooled = pooled @ self.text_projection

        if self.output_tokens:
            return pooled, tokens

        return pooled


class MultimodalTransformer(Transformer):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        context_length: int = 77,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        output_dim: int = 512,
        mask_prob: float = 0.0,
    ):
        super().__init__(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.context_length = context_length
        self.cross_attn = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    width,
                    heads,
                    mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    is_cross_attention=True,
                )
                for _ in range(layers)
            ]
        )

        self.register_buffer("attn_mask", self.build_attention_mask(), persistent=False)

        self.ln_final = norm_layer(width)
        self.text_projection = nn.Parameter(torch.empty(width, output_dim))

        self.mask_prob = mask_prob

    def init_parameters(self):
        proj_std = (self.transformer.width**-0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        for block in self.transformer.cross_attn:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width**-0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def set_mask_prob(self, mask_prob=0.0):
        self.mask_prob = mask_prob

    def lock_self_attention(self):
        for param in self.resblocks.parameters():
            param.requires_grad = False

    def forward(self, image_embs, text_embs):
        seq_len = text_embs.shape[1]

        attn_mask = self.attn_mask[:seq_len, :seq_len]
        if self.mask_prob > 0.0 and self.training:
            batch_size = text_embs.shape[0]
            attn_mask = attn_mask[None, :seq_len, :seq_len].repeat(batch_size, self.heads, 1, 1)
            p = random.random() * self.mask_prob
            rand = torch.randn(text_embs.shape[:2], device=text_embs.device)  # [N, L] token pos.
            rand[:, 0] = -torch.finfo(rand.dtype).max  # first token should not be masked out
            num_mask = min(int(seq_len * p), seq_len - 1)
            indices = rand.topk(num_mask, dim=-1).indices[:, None, :].repeat(1, seq_len, 1)
            mask = torch.zeros_like(attn_mask[:, 0, :, :]).scatter(2, indices, 1).bool()
            attn_mask = attn_mask.masked_fill(mask[:, None, :, :], float("-inf"))
            attn_mask = attn_mask.view(batch_size * self.heads, seq_len, seq_len)

        text_embs = text_embs.permute(1, 0, 2)  # NLD -> LNDsq
        image_embs = image_embs.permute(1, 0, 2)  # NLD -> LND
        seq_len = text_embs.shape[0]

        for resblock, cross_attn in zip(self.resblocks, self.cross_attn):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                text_embs = checkpoint(resblock, text_embs, None, None, attn_mask)
                text_embs = checkpoint(cross_attn, text_embs, image_embs, image_embs, None)
            else:
                text_embs = resblock(text_embs, attn_mask=attn_mask)
                text_embs = cross_attn(text_embs, k_x=image_embs, v_x=image_embs)

        x = text_embs.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        if self.text_projection is not None:
            x = x @ self.text_projection

        return x

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable


# --- conch/open_clip_custom/vision_tower.py --------------------------------


class VisualModel(nn.Module):
    def __init__(
        self,
        embed_dim_contrast,
        embed_dim_caption,
        trunk,
        image_size=224,
        proj="",
        proj_bias=False,
        drop=0.0,
        global_average_pool=False,
        use_attentional_pool_contrast=False,
        use_attentional_pool_caption=False,
        n_queries_contrast=1,
        n_queries_caption=256,
        attn_pooler_heads=8,
        norm_layer=nn.LayerNorm,
        output_tokens=False,
        trunk_kwargs={},
    ):
        super().__init__()

        self.trunk = trunk
        self.trunk_kwargs = trunk_kwargs
        self.image_size = to_2tuple(image_size)
        prev_chs = self.trunk.num_features
        head_layers = OrderedDict()

        # whether to use attentional pooling
        self.use_attentional_pool_contrast = use_attentional_pool_contrast
        self.use_attentional_pool_caption = use_attentional_pool_caption
        self.global_average_pool = global_average_pool
        self.output_tokens = output_tokens
        if use_attentional_pool_contrast:
            scale = prev_chs**-0.5
            self.attn_pool_contrast = AttentionalPooler(
                d_model=embed_dim_contrast,
                context_dim=prev_chs,
                n_head=attn_pooler_heads,
                n_queries=n_queries_contrast,
            )
            self.ln_contrast = norm_layer(embed_dim_contrast)
            self.proj_contrast = nn.Parameter(
                scale * torch.randn(embed_dim_contrast, embed_dim_contrast)
            )
        else:
            assert proj, "projection layer needed if not using attentional pooling."
            if proj == "linear":
                head_layers["drop"] = nn.Dropout(drop)
                head_layers["proj"] = nn.Linear(prev_chs, embed_dim_contrast, bias=proj_bias)
            elif proj == "mlp":
                head_layers["mlp"] = Mlp(
                    prev_chs,
                    2 * embed_dim_contrast,
                    embed_dim_contrast,
                    drop=(drop, 0),
                    bias=(True, proj_bias),
                )

        self.head = nn.Sequential(head_layers)

        if use_attentional_pool_caption:
            self.attn_pool_caption = AttentionalPooler(
                d_model=embed_dim_caption,
                context_dim=prev_chs,
                n_head=attn_pooler_heads,
                n_queries=n_queries_caption,
            )
            self.ln_caption = norm_layer(embed_dim_caption)

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        """lock modules
        Args:
            unlocked_groups (int): leave last n layer groups unlocked (default: 0)
        """
        if not unlocked_groups:
            for param in self.trunk.parameters():
                param.requires_grad = False
            if freeze_bn_stats:
                freeze_batch_norm_2d(self.trunk)
        else:
            from timm.models.helpers import group_parameters, group_modules

            matcher = self.trunk.group_matcher()
            gparams = group_parameters(self.trunk, matcher)
            max_layer_id = max(gparams.keys())
            max_layer_id = max_layer_id - unlocked_groups
            for group_idx in range(max_layer_id + 1):
                group = gparams[group_idx]
                for param in group:
                    self.trunk.get_parameter(param).requires_grad = False
            if freeze_bn_stats:
                gmodules = group_modules(self.trunk, matcher, reverse=True)
                gmodules = {k for k, v in gmodules.items() if v <= max_layer_id}
                freeze_batch_norm_2d(self.trunk, gmodules)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        try:
            self.trunk.set_grad_checkpointing(enable)
        except Exception:
            logging.warning(
                "grad checkpointing not supported for this timm image tower, continuing without..."
            )

    def _global_pool(self, x):
        if self.global_average_pool:
            return x.mean(dim=1), x
        else:
            return x[:, 0], x[:, 1:]

    def forward_project(self, x):
        if self.use_attentional_pool_contrast:
            x = x @ self.proj_contrast
            return x
        else:
            x = self.head(x)
            return x

    def forward_attn_pool_caption(self, tokens, attn_mask=None):
        if self.use_attentional_pool_caption:
            tokens = self.attn_pool_caption(tokens, attn_mask=attn_mask)
            tokens = self.ln_caption(tokens)
            return tokens
        else:
            raise NotImplementedError

    def forward_no_head(self, x, normalize=False):
        x = self.trunk(x, **self.trunk_kwargs)
        if self.use_attentional_pool_contrast:
            pooled = self.attn_pool_contrast(x)[:, 0]
            pooled = self.ln_contrast(pooled)
        else:
            pooled, _ = self._global_pool(x)
        if normalize:
            pooled = nn.functional.normalize(pooled, dim=-1)
        return pooled

    def forward(self, x):
        x = self.trunk(x, **self.trunk_kwargs)
        tokens = None
        if self.use_attentional_pool_contrast:
            pooled = self.attn_pool_contrast(x)[:, 0]  # single query
            pooled = self.ln_contrast(pooled)
            pooled = pooled @ self.proj_contrast
        else:
            pooled, tokens = self._global_pool(x)
            pooled = self.head(x)

        if self.use_attentional_pool_caption:
            tokens = self.attn_pool_caption(x)
            tokens = self.ln_caption(tokens)
        else:
            tokens = None

        if self.output_tokens:
            return pooled, tokens
        else:
            return pooled


# --- conch/open_clip_custom/coca_model.py -----------------------------------

try:
    from transformers import (
        LogitsProcessorList,
        TopPLogitsWarper,
        TopKLogitsWarper,
        RepetitionPenaltyLogitsProcessor,
        MinLengthLogitsProcessor,
        MaxLengthCriteria,
        StoppingCriteriaList,
    )

    GENERATION_TYPES = {"top_k": TopKLogitsWarper, "top_p": TopPLogitsWarper}
    _has_transformers = True
except ImportError:
    GENERATION_TYPES = {"top_k": None, "top_p": None}
    _has_transformers = False


@dataclass
class CoCaVisionCfg:
    layers: int = 12
    width: int = 768
    num_heads: int = 12
    mlp_ratio: int = 4
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    attentional_pool_contrast: bool = False  # perceiver resampler for contrastive loss
    attentional_pool_caption: bool = False  # perceiver resampler for captioning
    n_queries_contrast: int = 1  # n_queries for contrastive loss
    n_queries_caption: int = 256  # n_queries for captioning
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    output_tokens: bool = False


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value
    proj: str = "mlp"
    pooler_type: str = "mean_pooler"
    embed_cls: bool = False
    pad_id: int = 0
    output_tokens: bool = False


def _build_vision_tower(
    embed_dim: int, vision_cfg: CoCaVisionCfg, embed_dim_caption: Optional[int] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CoCaVisionCfg(**vision_cfg)

    trunk = VisionTransformer(
        embed_dim=vision_cfg.width,
        depth=vision_cfg.layers,
        num_heads=vision_cfg.num_heads,
        mlp_ratio=vision_cfg.mlp_ratio,
        img_size=vision_cfg.image_size,
        patch_size=vision_cfg.patch_size,
        num_classes=0,
        dynamic_img_size=True,
    )

    trunk_kwargs = {}
    trunk.forward = trunk.forward_features

    visual = VisualModel(
        trunk=trunk,
        trunk_kwargs=trunk_kwargs,
        use_attentional_pool_contrast=vision_cfg.attentional_pool_contrast,
        use_attentional_pool_caption=vision_cfg.attentional_pool_caption,
        n_queries_contrast=vision_cfg.n_queries_contrast,
        n_queries_caption=vision_cfg.n_queries_caption,
        output_tokens=vision_cfg.output_tokens,
        embed_dim_contrast=embed_dim,
        embed_dim_caption=embed_dim_caption,
        image_size=vision_cfg.image_size,
    )
    return visual


def _build_text_tower(embed_dim: int, text_cfg: CLIPTextCfg):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)
    act_layer = nn.GELU
    norm_layer = nn.LayerNorm

    text = TextTransformer(
        context_length=text_cfg.context_length,
        vocab_size=text_cfg.vocab_size,
        width=text_cfg.width,
        heads=text_cfg.heads,
        layers=text_cfg.layers,
        ls_init_value=text_cfg.ls_init_value,
        output_dim=embed_dim,
        embed_cls=text_cfg.embed_cls,
        output_tokens=text_cfg.output_tokens,
        pad_id=text_cfg.pad_id,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )
    return text


def _build_text_decoder_tower(embed_dim, multimodal_cfg):
    multimodal_cfg = (
        CLIPTextCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
    )
    act_layer = nn.GELU
    norm_layer = nn.LayerNorm

    decoder = MultimodalTransformer(
        context_length=multimodal_cfg.context_length,
        width=multimodal_cfg.width,
        heads=multimodal_cfg.heads,
        layers=multimodal_cfg.layers,
        ls_init_value=multimodal_cfg.ls_init_value,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )

    return decoder


class CoCa(nn.Module):
    def __init__(
        self,
        embed_dim,
        embed_dim_caption,
        multimodal_cfg: CLIPTextCfg,
        text_cfg: CLIPTextCfg,
        vision_cfg: CoCaVisionCfg,
        pad_id: int = 0,
    ):
        super().__init__()
        multimodal_cfg = (
            CLIPTextCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
        )
        text_cfg = CLIPTextCfg(**text_cfg) if isinstance(text_cfg, dict) else text_cfg
        vision_cfg = CoCaVisionCfg(**vision_cfg) if isinstance(vision_cfg, dict) else vision_cfg

        self.text = _build_text_tower(embed_dim=embed_dim, text_cfg=text_cfg)

        vocab_size = text_cfg.vocab_size

        self.visual = _build_vision_tower(
            embed_dim=embed_dim, embed_dim_caption=embed_dim_caption, vision_cfg=vision_cfg
        )

        if multimodal_cfg.layers > 0:
            self.text_decoder = _build_text_decoder_tower(vocab_size, multimodal_cfg=multimodal_cfg)
        else:
            self.text_decoder = None

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.pad_id = pad_id
        self.context_length = text_cfg.context_length

        self.embed_dim = embed_dim
        self.embed_dim_caption = embed_dim_caption

    def lock_temperature(self):
        self.logit_scale.requires_grad = False

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)
        self.text_decoder.set_grad_checkpointing(enable)

    def _encode_image(self, images=None, normalize=True):
        image_latent, tokens_embs = self.visual(images)
        image_latent = F.normalize(image_latent, dim=-1) if normalize else image_latent
        return image_latent, tokens_embs

    def _encode_text(self, text, normalize=True, embed_cls=True):
        text = text[:, :-1] if embed_cls else text  # make space for CLS token
        text_latent, token_emb = self.text(text)
        text_latent = F.normalize(text_latent, dim=-1) if normalize else text_latent
        return text_latent, token_emb

    def encode_image(self, images, normalize=True, proj_contrast=True):
        if proj_contrast:
            image_latent, _ = self._encode_image(images, normalize=normalize)
        else:
            image_latent = self.visual.forward_no_head(images, normalize=normalize)
        return image_latent

    def encode_text(self, text, normalize=True, embed_cls=True):
        text_latent, _ = self._encode_text(text, normalize=normalize, embed_cls=embed_cls)
        return text_latent

    def forward(self, image, text, embed_cls=True, image_latent=None, image_embs=None):
        text_latent, token_embs = self._encode_text(text, embed_cls=embed_cls)
        if image_latent is None or image_embs is None:
            image_latent, image_embs = self._encode_image(image)

        labels = text[:, -token_embs.shape[1] :]
        if self.text_decoder is not None:
            logits = self.text_decoder(image_embs, token_embs)
        else:
            logits = torch.empty(text.shape[0], 1, device=text.device)
        return {
            "image_features": image_latent,
            "text_features": text_latent,
            "logits": logits,
            "labels": labels,
            "logit_scale": self.logit_scale.exp(),
        }


# --- staging harness -------------------------------------------------------
# Tiny end-to-end CoCa config: small ViT trunk, small text transformer,
# small multimodal decoder -- matches the real construction path
# (`_build_vision_tower` / `_build_text_tower` / `_build_text_decoder_tower`
# used inside `CoCa.__init__`), just at toy scale.


def build_conch():
    embed_dim = 32
    embed_dim_caption = 32
    # NOTE: `_build_vision_tower` never forwards a `proj` kwarg to
    # `VisualModel`, so its default `proj=''` triggers `VisualModel`'s own
    # assert unless attentional pooling is enabled. This mirrors the real
    # repo's actual shipped CONCH configs, which use attentional pooling for
    # BOTH contrast and caption branches -- `VisualModel.forward` only
    # returns non-None `tokens` (needed by the multimodal decoder's
    # cross-attention) when `use_attentional_pool_caption=True`.
    vision_cfg = dict(
        layers=2,
        width=32,
        num_heads=4,
        mlp_ratio=2,
        patch_size=16,
        image_size=64,
        attentional_pool_contrast=True,
        attentional_pool_caption=True,
        n_queries_contrast=1,
        n_queries_caption=8,
        attn_pooler_heads=4,
        # `CoCa._encode_image` unpacks `self.visual(images)` as
        # `(image_latent, tokens_embs)`, which requires the visual tower's
        # own `output_tokens=True` (otherwise `forward()` returns a bare
        # `pooled` tensor that Python unpacks along dim 0 by accident).
        output_tokens=True,
    )
    # NOTE: `CoCa.forward` unpacks `self.text(text)` as `(text_latent,
    # token_emb)` and feeds `token_emb` to the multimodal decoder, so
    # `text_cfg.output_tokens` must be True whenever `multimodal_cfg.layers`
    # > 0 -- matching the real shipped CONCH text-tower configs.
    text_cfg = dict(
        context_length=16,
        vocab_size=100,
        width=32,
        heads=4,
        layers=2,
        proj="linear",
        embed_cls=True,
        pad_id=0,
        output_tokens=True,
    )
    multimodal_cfg = dict(
        context_length=16,
        vocab_size=100,
        width=32,
        heads=4,
        layers=2,
    )
    return CoCa(
        embed_dim=embed_dim,
        embed_dim_caption=embed_dim_caption,
        multimodal_cfg=multimodal_cfg,
        text_cfg=text_cfg,
        vision_cfg=vision_cfg,
        pad_id=0,
    )


def example_input_conch():
    batch = 2
    image = torch.randn(batch, 3, 64, 64)
    text = torch.randint(1, 100, (batch, 16), dtype=torch.long)
    text[:, 0] = 1  # sot token
    return (image, text)


MENAGERIE_ENTRIES = [
    ("CONCH", "build_conch", "example_input_conch", 2024, "vendored"),
]
