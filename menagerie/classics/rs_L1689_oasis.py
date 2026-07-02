# SOURCE: vendored from etched-ai/open-oasis @ f59deef2c019c212bd0c5a3a5b986a51f3701847
# https://github.com/etched-ai/open-oasis -- Etched AI's open reproduction of "Oasis", a
# real-time, autoregressive, diffusion-based Minecraft neural world model. Two distinct
# architectures make up the system: a ViT-based spatial autoencoder (`vae.py`, patch-based
# attention encoder/decoder around a diagonal-Gaussian latent bottleneck) and a
# spatio-temporal Diffusion Transformer (`dit.py` + `attention.py`) that denoises sequences
# of VAE latents autoregressively, conditioned on diffusion timestep and external
# (action/control) embeddings, using axial (spatial then temporal) rotary-embedded
# attention blocks with adaLN-Zero modulation (à la DiT/Latte/Diffusion-Forcing). Classes
# below (`PatchEmbed`, `TimestepEmbedder`, `FinalLayer`, `SpatioTemporalDiTBlock`, `DiT`,
# `TemporalAxialAttention`, `SpatialAxialAttention`, `DiagonalGaussianDistribution`,
# `Attention`, `AttentionBlock`, `AutoencoderKL`) are transcribed verbatim from
# dit.py / attention.py / vae.py; only the `DiT_S_2` / `ViT_L_20_Shallow_Encoder` factory
# wrappers are re-parameterized below at tiny size for fast tracing (same architecture,
# far fewer layers/channels), and the two `rotate_queries_or_keys(q, self.rotary_emb.freqs)`
# calls in `TemporalAxialAttention.forward` are updated to `rotate_queries_or_keys(q)` --
# a minimal library-API compat fix for the installed rotary-embedding-torch>=0.6 (which
# computes freqs internally instead of taking them as an explicit 2nd positional arg), not
# an architectural change.
import functools
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from timm.layers.helpers import to_2tuple
from timm.models.vision_transformer import Mlp

MENAGERIE_ZOO = "vendored-pytorch"


# ---- verbatim from dit.py ----
def modulate(x, shift, scale):
    fixed_dims = [1] * len(shift.shape[1:])
    shift = shift.repeat(x.shape[0] // shift.shape[0], *fixed_dims)
    scale = scale.repeat(x.shape[0] // scale.shape[0], *fixed_dims)
    while shift.dim() < x.dim():
        shift = shift.unsqueeze(-2)
        scale = scale.unsqueeze(-2)
    return x * (1 + scale) + shift


def gate(x, g):
    fixed_dims = [1] * len(g.shape[1:])
    g = g.repeat(x.shape[0] // g.shape[0], *fixed_dims)
    while g.dim() < x.dim():
        g = g.unsqueeze(-2)
    return g * x


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_height=256,
        img_width=256,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        img_size = (img_height, img_width)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x, random_sample=False):
        B, C, H, W = x.shape
        assert random_sample or (H == self.img_size[0] and W == self.img_size[1]), (
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        )
        x = self.proj(x)
        if self.flatten:
            x = rearrange(x, "B C H W -> B (H W) C")
        else:
            x = rearrange(x, "B C H W -> B H W C")
        x = self.norm(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(
                frequency_embedding_size, hidden_size, bias=True
            ),  # hidden_size is diffusion model hidden size
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# ---- verbatim from attention.py ----
class TemporalAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: RotaryEmbedding,
        is_causal: bool = True,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim)

        self.rotary_emb = rotary_emb
        self.is_causal = is_causal

    def forward(self, x: torch.Tensor):
        B, T, H, W, D = x.shape

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q = rearrange(q, "B T H W (h d) -> (B H W) h T d", h=self.heads)
        k = rearrange(k, "B T H W (h d) -> (B H W) h T d", h=self.heads)
        v = rearrange(v, "B T H W (h d) -> (B H W) h T d", h=self.heads)

        # NOTE: original repo code called rotate_queries_or_keys(q, self.rotary_emb.freqs)
        # against an older rotary-embedding-torch where the 2nd positional arg was `freqs`;
        # installed rotary-embedding-torch>=0.6 computes freqs internally from `seq_dim`
        # (2nd positional arg is now `seq_dim`), so `freqs` is no longer passed explicitly.
        # This is a library-API compat fix only -- no architectural change.
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        x = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=self.is_causal)

        x = rearrange(x, "(B H W) h T d -> B T H W (h d)", B=B, H=H, W=W)
        x = x.to(q.dtype)

        # linear proj
        x = self.to_out(x)
        return x


class SpatialAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: RotaryEmbedding,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim)

        self.rotary_emb = rotary_emb

    def forward(self, x: torch.Tensor):
        B, T, H, W, D = x.shape

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q = rearrange(q, "B T H W (h d) -> (B T) h H W d", h=self.heads)
        k = rearrange(k, "B T H W (h d) -> (B T) h H W d", h=self.heads)
        v = rearrange(v, "B T H W (h d) -> (B T) h H W d", h=self.heads)

        freqs = self.rotary_emb.get_axial_freqs(H, W)
        q = apply_rotary_emb(freqs, q)
        k = apply_rotary_emb(freqs, k)

        # prepare for attn
        q = rearrange(q, "(B T) h H W d -> (B T) h (H W) d", B=B, T=T, h=self.heads)
        k = rearrange(k, "(B T) h H W d -> (B T) h (H W) d", B=B, T=T, h=self.heads)
        v = rearrange(v, "(B T) h H W d -> (B T) h (H W) d", B=B, T=T, h=self.heads)

        x = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=False)

        x = rearrange(x, "(B T) h (H W) d -> B T H W (h d)", B=B, H=H, W=W)
        x = x.to(q.dtype)

        # linear proj
        x = self.to_out(x)
        return x


class SpatioTemporalDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        is_causal=True,
        spatial_rotary_emb: Optional[RotaryEmbedding] = None,
        temporal_rotary_emb: Optional[RotaryEmbedding] = None,
    ):
        super().__init__()
        self.is_causal = is_causal
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")  # noqa: E731 -- verbatim from dit.py

        self.s_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.s_attn = SpatialAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            rotary_emb=spatial_rotary_emb,
        )
        self.s_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.s_mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.s_adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        self.t_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.t_attn = TemporalAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            is_causal=is_causal,
            rotary_emb=temporal_rotary_emb,
        )
        self.t_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.t_mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.t_adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        B, T, H, W, D = x.shape

        # spatial block
        s_shift_msa, s_scale_msa, s_gate_msa, s_shift_mlp, s_scale_mlp, s_gate_mlp = (
            self.s_adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate(self.s_attn(modulate(self.s_norm1(x), s_shift_msa, s_scale_msa)), s_gate_msa)
        x = x + gate(self.s_mlp(modulate(self.s_norm2(x), s_shift_mlp, s_scale_mlp)), s_gate_mlp)

        # temporal block
        t_shift_msa, t_scale_msa, t_gate_msa, t_shift_mlp, t_scale_mlp, t_gate_mlp = (
            self.t_adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate(self.t_attn(modulate(self.t_norm1(x), t_shift_msa, t_scale_msa)), t_gate_msa)
        x = x + gate(self.t_mlp(modulate(self.t_norm2(x), t_shift_mlp, t_scale_mlp)), t_gate_mlp)

        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_h=18,
        input_w=32,
        patch_size=2,
        in_channels=16,
        hidden_size=1024,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        external_cond_dim=25,
        max_frames=32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.max_frames = max_frames

        self.x_embedder = PatchEmbed(
            input_h, input_w, patch_size, in_channels, hidden_size, flatten=False
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        frame_h, frame_w = self.x_embedder.grid_size

        self.spatial_rotary_emb = RotaryEmbedding(
            dim=hidden_size // num_heads // 2, freqs_for="pixel", max_freq=256
        )
        self.temporal_rotary_emb = RotaryEmbedding(dim=hidden_size // num_heads)
        self.external_cond = (
            nn.Linear(external_cond_dim, hidden_size) if external_cond_dim > 0 else nn.Identity()
        )

        self.blocks = nn.ModuleList(
            [
                SpatioTemporalDiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    is_causal=True,
                    spatial_rotary_emb=self.spatial_rotary_emb,
                    temporal_rotary_emb=self.temporal_rotary_emb,
                )
                for _ in range(depth)
            ]
        )

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.s_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.s_adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.t_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.t_adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, H, W, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = x.shape[1]
        w = x.shape[2]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, t, external_cond=None):
        """
        Forward pass of DiT.
        x: (B, T, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (B, T,) tensor of diffusion timesteps
        """

        B, T, C, H, W = x.shape

        # add spatial embeddings
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.x_embedder(x)  # (B*T, C, H, W) -> (B*T, H/2, W/2, D) , C = 16, D = d_model
        # restore shape
        x = rearrange(x, "(b t) h w d -> b t h w d", t=T)
        # embed noise steps
        t = rearrange(t, "b t -> (b t)")
        c = self.t_embedder(t)  # (N, D)
        c = rearrange(c, "(b t) d -> b t d", t=T)
        if torch.is_tensor(external_cond):
            c += self.external_cond(external_cond)
        for block in self.blocks:
            x = block(x, c)  # (N, T, H, W, D)
        x = self.final_layer(x, c)  # (N, T, H, W, patch_size ** 2 * out_channels)
        # unpatchify
        x = rearrange(x, "b t h w d -> (b t) h w d")
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        x = rearrange(x, "(b t) c h w -> b t c h w", t=T)

        return x


# ---- verbatim from vae.py ----
class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False, dim=1):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=dim)
        if dim == 1:
            self.dims = [1, 2, 3]
        elif dim == 2:
            self.dims = [1, 2]
        else:
            raise NotImplementedError
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def mode(self):
        return self.mean


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        frame_height,
        frame_width,
        qkv_bias=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.frame_height = frame_height
        self.frame_width = frame_width

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        rotary_freqs = RotaryEmbedding(
            dim=head_dim // 4,
            freqs_for="pixel",
            max_freq=frame_height * frame_width,
        ).get_axial_freqs(frame_height, frame_width)
        self.register_buffer("rotary_freqs", rotary_freqs, persistent=False)

    def forward(self, x):
        B, N, C = x.shape
        assert N == self.frame_height * self.frame_width

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = rearrange(
            q,
            "b (H W) (h d) -> b h H W d",
            H=self.frame_height,
            W=self.frame_width,
            h=self.num_heads,
        )
        k = rearrange(
            k,
            "b (H W) (h d) -> b h H W d",
            H=self.frame_height,
            W=self.frame_width,
            h=self.num_heads,
        )
        v = rearrange(
            v,
            "b (H W) (h d) -> b h H W d",
            H=self.frame_height,
            W=self.frame_width,
            h=self.num_heads,
        )

        q = apply_rotary_emb(self.rotary_freqs, q)
        k = apply_rotary_emb(self.rotary_freqs, k)

        q = rearrange(q, "b h H W d -> b h (H W) d")
        k = rearrange(k, "b h H W d -> b h (H W) d")
        v = rearrange(v, "b h H W d -> b h (H W) d")

        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b h N d -> b N (h d)")

        x = self.proj(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        frame_height,
        frame_width,
        mlp_ratio=4.0,
        qkv_bias=False,
        attn_causal=False,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads,
            frame_height,
            frame_width,
            qkv_bias=qkv_bias,
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class AutoencoderKL(nn.Module):
    def __init__(
        self,
        latent_dim,
        input_height=256,
        input_width=256,
        patch_size=16,
        enc_dim=768,
        enc_depth=6,
        enc_heads=12,
        dec_dim=768,
        dec_depth=6,
        dec_heads=12,
        mlp_ratio=4.0,
        norm_layer=functools.partial(nn.LayerNorm, eps=1e-6),
        use_variational=True,
        **kwargs,
    ):
        super().__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.patch_size = patch_size
        self.seq_h = input_height // patch_size
        self.seq_w = input_width // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.patch_dim = 3 * patch_size**2

        self.latent_dim = latent_dim
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim

        # patch
        self.patch_embed = PatchEmbed(input_height, input_width, patch_size, 3, enc_dim)

        # encoder
        self.encoder = nn.ModuleList(
            [
                AttentionBlock(
                    enc_dim,
                    enc_heads,
                    self.seq_h,
                    self.seq_w,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(enc_depth)
            ]
        )
        self.enc_norm = norm_layer(enc_dim)

        # bottleneck
        self.use_variational = use_variational
        mult = 2 if self.use_variational else 1
        self.quant_conv = nn.Linear(enc_dim, mult * latent_dim)
        self.post_quant_conv = nn.Linear(latent_dim, dec_dim)

        # decoder
        self.decoder = nn.ModuleList(
            [
                AttentionBlock(
                    dec_dim,
                    dec_heads,
                    self.seq_h,
                    self.seq_w,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm = norm_layer(dec_dim)
        self.predictor = nn.Linear(dec_dim, self.patch_dim)  # decoder to patch

        # initialize this weight first
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        # patchify
        bsz, _, h, w = x.shape
        x = x.reshape(
            bsz,
            3,
            self.seq_h,
            self.patch_size,
            self.seq_w,
            self.patch_size,
        ).permute([0, 1, 3, 5, 2, 4])  # [b, c, h, p, w, p] --> [b, c, p, p, h, w]
        x = x.reshape(bsz, self.patch_dim, self.seq_h, self.seq_w)  # --> [b, cxpxp, h, w]
        x = x.permute([0, 2, 3, 1]).reshape(
            bsz, self.seq_len, self.patch_dim
        )  # --> [b, hxw, cxpxp]
        return x

    def unpatchify(self, x):
        bsz = x.shape[0]
        # unpatchify
        x = x.reshape(bsz, self.seq_h, self.seq_w, self.patch_dim).permute(
            [0, 3, 1, 2]
        )  # [b, h, w, cxpxp] --> [b, cxpxp, h, w]
        x = x.reshape(
            bsz,
            3,
            self.patch_size,
            self.patch_size,
            self.seq_h,
            self.seq_w,
        ).permute([0, 1, 4, 2, 5, 3])  # [b, c, p, p, h, w] --> [b, c, h, p, w, p]
        x = x.reshape(
            bsz,
            3,
            self.input_height,
            self.input_width,
        )  # [b, c, hxp, wxp]
        return x

    def encode(self, x):
        # patchify
        x = self.patch_embed(x)

        # encoder
        for blk in self.encoder:
            x = blk(x)
        x = self.enc_norm(x)

        # bottleneck
        moments = self.quant_conv(x)
        if not self.use_variational:
            moments = torch.cat((moments, torch.zeros_like(moments)), 2)
        posterior = DiagonalGaussianDistribution(
            moments, deterministic=(not self.use_variational), dim=2
        )
        return posterior

    def decode(self, z):
        # bottleneck
        z = self.post_quant_conv(z)

        # decoder
        for blk in self.decoder:
            z = blk(z)
        z = self.dec_norm(z)

        # predictor
        z = self.predictor(z)

        # unpatchify
        dec = self.unpatchify(z)
        return dec

    def autoencode(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if self.use_variational and sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior, z

    def forward(self, inputs, labels=None, split="train"):
        rec, post, latent = self.autoencode(inputs)
        return rec, post, latent


# ---- staging build/example helpers (tiny sizes for fast tracing) ----
def build_oasis_dit():
    torch.manual_seed(0)
    model = DiT(
        input_h=6,
        input_w=8,
        patch_size=2,
        in_channels=4,
        hidden_size=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        external_cond_dim=8,
        max_frames=4,
    )
    model.eval()
    return model


def example_input_oasis_dit():
    torch.manual_seed(0)
    batch_size = 1
    num_frames = 3
    in_channels = 4
    input_h, input_w = 6, 8
    x = torch.randn(batch_size, num_frames, in_channels, input_h, input_w)
    t = torch.randint(0, 1000, (batch_size, num_frames)).float()
    external_cond = torch.randn(batch_size, num_frames, 8)
    return (x, t, external_cond)


def build_oasis_vae():
    torch.manual_seed(0)
    model = AutoencoderKL(
        latent_dim=4,
        input_height=32,
        input_width=32,
        patch_size=8,
        enc_dim=64,
        enc_depth=2,
        enc_heads=4,
        dec_dim=64,
        dec_depth=2,
        dec_heads=4,
        mlp_ratio=2.0,
    )
    model.eval()
    return model


def example_input_oasis_vae():
    torch.manual_seed(0)
    x = torch.randn(1, 3, 32, 32)
    return (x,)


MENAGERIE_ENTRIES = [
    (
        "Oasis-DiT (world model transformer)",
        build_oasis_dit,
        example_input_oasis_dit,
        2024,
        "vendored-pytorch",
    ),
    (
        "Oasis-VAE (ViT spatial autoencoder)",
        build_oasis_vae,
        example_input_oasis_vae,
        2024,
        "vendored-pytorch",
    ),
]
