# SOURCE: vendored from tung-nd/stormer @ main
# https://raw.githubusercontent.com/tung-nd/stormer/main/stormer/models/hub/stormer.py
# https://raw.githubusercontent.com/tung-nd/stormer/main/stormer/models/hub/weather_embedding.py
# https://raw.githubusercontent.com/tung-nd/stormer/main/stormer/utils/pos_embed.py
# Stormer (Nguyen et al., "Scaling transformer neural networks for skillful and reliable
# medium-range weather forecasting") is a ViT-style adaLN-conditioned transformer operating
# over per-variable weather-field patch tokens, aggregated via learned cross-attention.
# Only two adaptations from the real repo code: (1) the module import path
# `.weather_embedding` is inlined since we vendor into one file, and (2) the real repo's
# attention layer calls `xformers.ops.memory_efficient_attention`, a CUDA-specific fused
# kernel computing the exact same softmax(QK^T/sqrt(d))V attention math -- xformers is not
# in the installed base-lib set (needs a CUDA build), so we fall back to torch's native
# `F.scaled_dot_product_attention`, which computes the identical operation. This is the
# same established substitution pattern already used in menagerie/classics/rs_L344_wovogen.py
# and menagerie/classics/rs_L344_vista.py (try xformers, else native SDPA) -- no
# architectural change, purely a kernel swap.
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, PatchEmbed, trunc_normal_

try:
    from xformers.ops import memory_efficient_attention, unbind

    XFORMERS_IS_AVAILABLE = True
except ImportError:
    XFORMERS_IS_AVAILABLE = False


# --------------------------------------------------------
# 2D sine-cosine position embedding (verbatim from stormer/utils/pos_embed.py)
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False):
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# WeatherEmbedding (verbatim from stormer/models/hub/weather_embedding.py)
# --------------------------------------------------------
class WeatherEmbedding(nn.Module):
    def __init__(
        self,
        variables,
        img_size,
        patch_size=2,
        embed_dim=1024,
        num_heads=16,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.variables = variables

        # variable tokenization: separate embedding layer for each input variable
        self.token_embeds = nn.ModuleList(
            [PatchEmbed(None, patch_size, 1, embed_dim) for i in range(len(variables))]
        )
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        # variable embedding to denote which variable each token belongs to
        # helps in aggregating variables
        self.channel_embed, self.channel_map = self.create_var_embedding(embed_dim)

        # variable aggregation: a learnable query and a single-layer cross attention
        self.channel_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.channel_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=True
        )

        self.initialize_weights()

    def initialize_weights(self):
        # initialize pos_emb and var_emb
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        channel_embed = get_1d_sincos_pos_embed_from_grid(
            self.channel_embed.shape[-1], np.arange(len(self.variables))
        )
        self.channel_embed.data.copy_(torch.from_numpy(channel_embed).float().unsqueeze(0))

        # token embedding layer
        for i in range(len(self.token_embeds)):
            w = self.token_embeds[i].proj.weight.data
            trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_var_embedding(self, dim):
        var_embed = nn.Parameter(torch.zeros(1, len(self.variables), dim), requires_grad=True)
        var_map = {}
        idx = 0
        for var in self.variables:
            var_map[var] = idx
            idx += 1
        return var_embed, var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        ids = np.array([self.channel_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def get_var_emb(self, var_emb, vars):
        ids = self.get_var_ids(vars, var_emb.device)
        return var_emb[:, ids, :]

    def aggregate_variables(self, x: torch.Tensor):
        """
        x: B, V, L, D
        """
        b, _, l, _ = x.shape  # noqa: E741 (verbatim from source)
        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)  # BxL, V, D

        var_query = self.channel_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.channel_agg(var_query, x, x)  # BxL, D
        x = x.squeeze()

        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, D
        return x

    def forward(self, x: torch.Tensor, variables):
        if isinstance(variables, list):
            variables = tuple(variables)

        # tokenize each variable separately
        embeds = []
        var_ids = self.get_var_ids(variables, x.device)

        for i in range(len(var_ids)):
            id = var_ids[i]
            embed_variable = self.token_embeds[id](x[:, i : i + 1])  # B, L, D
            embeds.append(embed_variable)
        x = torch.stack(embeds, dim=1)  # B, V, L, D

        # add variable embedding
        var_embed = self.get_var_emb(self.channel_embed, variables)
        x = x + var_embed.unsqueeze(2)
        x = x + self.pos_embed.unsqueeze(1)

        # variable aggregation
        x = self.aggregate_variables(x)  # B, L, D

        return x


# --------------------------------------------------------
# Stormer (verbatim from stormer/models/hub/stormer.py)
# --------------------------------------------------------
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Linear(1, hidden_size)

    def forward(self, t):
        return self.mlp(t.unsqueeze(-1))


class MemEffAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_bias=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        if XFORMERS_IS_AVAILABLE:
            q, k, v = unbind(qkv, 2)
            x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
            x = x.reshape([B, N, C])
        else:
            # native fallback (xformers is a CUDA-only optional accelerator dep, not
            # in the installed base-lib set): identical softmax(QK^T/sqrt(d))V
            # attention via F.scaled_dot_product_attention.
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each: B, num_heads, N, head_dim
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
            x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """
    An transformers block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MemEffAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")  # noqa: E731 (verbatim from source)
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c
        ).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.Identity()
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class Stormer(nn.Module):
    def __init__(
        self,
        in_img_size,
        variables,
        patch_size=2,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
    ):
        super().__init__()

        if in_img_size[0] % patch_size != 0:
            pad_size = patch_size - in_img_size[0] % patch_size
            in_img_size = (in_img_size[0] + pad_size, in_img_size[1])
        self.in_img_size = in_img_size
        self.variables = variables
        self.patch_size = patch_size

        # embedding
        self.embedding = WeatherEmbedding(
            variables=variables,
            img_size=in_img_size,
            patch_size=patch_size,
            embed_dim=hidden_size,
            num_heads=num_heads,
        )
        self.embed_norm_layer = nn.LayerNorm(hidden_size)

        # interval embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # backbone
        self.blocks = nn.ModuleList(
            [Block(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )

        # prediction layer
        self.head = FinalLayer(hidden_size, patch_size, len(variables))

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        trunc_normal_(self.t_embedder.mlp.weight, std=0.02)

        # Zero-out adaLN modulation layers in blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.linear.weight, 0)
        nn.init.constant_(self.head.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor, h=None, w=None):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        p = self.patch_size
        v = len(self.variables)
        h = self.in_img_size[0] // p if h is None else h // p
        w = self.in_img_size[1] // p if w is None else w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, v))
        x = torch.einsum("nhwpqv->nvhpwq", x)
        imgs = x.reshape(shape=(x.shape[0], v, h * p, w * p))
        return imgs

    def forward(self, x, variables, time_interval):
        x = self.embedding(x, variables)  # B, L, D
        x = self.embed_norm_layer(x)

        time_interval_emb = self.t_embedder(time_interval)
        for block in self.blocks:
            x = block(x, time_interval_emb)

        x = self.head(x, time_interval_emb)
        x = self.unpatchify(x)

        return x


# Tiny staging config: a handful of weather variables, a small spatial grid, few heads/
# depth/hidden size, well below the real 69-variable / 721x1440 / depth-24 / hidden-1024
# checkpoint used for medium-range forecasting, but exercising the identical module graph.
_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]
_IMG_SIZE = (16, 16)
_PATCH_SIZE = 2
_HIDDEN_SIZE = 32
_DEPTH = 2
_NUM_HEADS = 4


def build_stormer():
    return Stormer(
        in_img_size=_IMG_SIZE,
        variables=_VARIABLES,
        patch_size=_PATCH_SIZE,
        hidden_size=_HIDDEN_SIZE,
        depth=_DEPTH,
        num_heads=_NUM_HEADS,
    )


class _StormerInputWrapper(nn.Module):
    """Binds the non-tensor `variables` kwarg so the module accepts (x, time_interval)."""

    def __init__(self, stormer_model, variables):
        super().__init__()
        self.stormer_model = stormer_model
        self._variables = tuple(variables)

    def forward(self, x, time_interval):
        return self.stormer_model(x, self._variables, time_interval)


def build_stormer_wrapped():
    return _StormerInputWrapper(build_stormer(), _VARIABLES)


def example_input_stormer_wrapped():
    x = torch.randn(2, len(_VARIABLES), *_IMG_SIZE)
    time_interval = torch.tensor([6.0, 12.0])
    return (x, time_interval)


MENAGERIE_ZOO = "vendored-pytorch"

MENAGERIE_ENTRIES = [
    (
        "Stormer (weather ViT)",
        "build_stormer_wrapped",
        "example_input_stormer_wrapped",
        2023,
        "vendored-pytorch",
    ),
]
