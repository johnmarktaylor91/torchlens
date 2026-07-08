# SOURCE: vendored from tr3e/InterGen @ main (models/layers.py, models/blocks.py,
# models/utils.py, models/nets.py -- InterDenoiser class only)
# https://github.com/tr3e/InterGen -- "InterGen: Diffusion-based Multi-human Motion
# Generation under Complex Interactions" (Liang et al., IJCV 2024). The repo's full
# `InterGen` wrapper (models/intergen.py) additionally loads OpenAI CLIP for text
# conditioning and a `MotionDiffusion` DDPM sampling/training loop
# (models/gaussian_diffusion.py, borrowed from openai/guided-diffusion) -- neither of
# those is the traced architecture itself (CLIP is an off-the-shelf frozen text
# encoder; the diffusion wrapper is a training/sampling procedure around the network,
# not a static forward graph). `InterDenoiser` is the actual novel contribution: a
# dual-stream cross-attending transformer denoiser that alternately lets each of the
# two interacting humans' motion streams attend to itself (VanillaSelfAttention) and
# to the other stream (VanillaCrossAttention), conditioned on the diffusion timestep +
# text embedding via AdaLN. Every class body below is transcribed verbatim from the
# real repo files; only imports were adjusted (dropped the CLIP text-conditioning path
# from InterDenoiser's caller and the package-relative `from .layers import *`/
# `from .utils import *` were flattened into this single module).
import numpy as np
import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---- verbatim from models/utils.py ----
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[: x.shape[1], :].unsqueeze(0)
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps])


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


# ---- verbatim from models/layers.py ----
class AdaLN(nn.Module):
    def __init__(self, latent_dim, embed_dim=None):
        super().__init__()
        if embed_dim is None:
            embed_dim = latent_dim
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            zero_module(nn.Linear(embed_dim, 2 * latent_dim, bias=True)),
        )
        self.norm = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, h, emb):
        """
        h: B, T, D
        emb: B, D
        """
        emb_out = self.emb_layers(emb)
        scale, shift = torch.chunk(emb_out, 2, dim=-1)
        h = self.norm(h) * (1 + scale[:, None]) + shift[:, None]
        return h


class VanillaSelfAttention(nn.Module):
    def __init__(self, latent_dim, num_head, dropout, embed_dim=None):
        super().__init__()
        self.num_head = num_head
        self.norm = AdaLN(latent_dim, embed_dim)
        self.attention = nn.MultiheadAttention(
            latent_dim, num_head, dropout=dropout, batch_first=True, add_zero_attn=True
        )

    def forward(self, x, emb, key_padding_mask=None):
        """
        x: B, T, D
        """
        x_norm = self.norm(x, emb)
        y = self.attention(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=None,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return y


class VanillaCrossAttention(nn.Module):
    def __init__(self, latent_dim, xf_latent_dim, num_head, dropout, embed_dim=None):
        super().__init__()
        self.num_head = num_head
        self.norm = AdaLN(latent_dim, embed_dim)
        self.xf_norm = AdaLN(xf_latent_dim, embed_dim)
        self.attention = nn.MultiheadAttention(
            latent_dim,
            num_head,
            kdim=xf_latent_dim,
            vdim=xf_latent_dim,
            dropout=dropout,
            batch_first=True,
            add_zero_attn=True,
        )

    def forward(self, x, xf, emb, key_padding_mask=None):
        """
        x: B, T, D
        xf: B, N, L
        """
        x_norm = self.norm(x, emb)
        xf_norm = self.xf_norm(xf, emb)
        y = self.attention(
            x_norm,
            xf_norm,
            xf_norm,
            attn_mask=None,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return y


class FFN(nn.Module):
    def __init__(self, latent_dim, ffn_dim, dropout, embed_dim=None):
        super().__init__()
        self.norm = AdaLN(latent_dim, embed_dim)
        self.linear1 = nn.Linear(latent_dim, ffn_dim, bias=True)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim, bias=True))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, emb=None):
        if emb is not None:
            x_norm = self.norm(x, emb)
        else:
            x_norm = x
        y = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
        return y


class FinalLayer(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super().__init__()
        self.linear = zero_module(nn.Linear(latent_dim, out_dim, bias=True))

    def forward(self, x):
        x = self.linear(x)
        return x


# ---- verbatim from models/blocks.py ----
class TransformerBlock(nn.Module):
    def __init__(
        self, latent_dim=512, num_heads=8, ff_size=1024, dropout=0.0, cond_abl=False, **kargs
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.cond_abl = cond_abl

        self.sa_block = VanillaSelfAttention(latent_dim, num_heads, dropout)
        self.ca_block = VanillaCrossAttention(
            latent_dim, latent_dim, num_heads, dropout, latent_dim
        )
        self.ffn = FFN(latent_dim, ff_size, dropout, latent_dim)

    def forward(self, x, y, emb=None, key_padding_mask=None):
        h1 = self.sa_block(x, emb, key_padding_mask)
        h1 = h1 + x
        h2 = self.ca_block(h1, y, emb, key_padding_mask)
        h2 = h2 + h1
        out = self.ffn(h2, emb)
        out = out + h2
        return out


# ---- verbatim from models/nets.py (InterDenoiser only -- MotionEncoder and
# InterDiffusion are the training/CFG-sampling wrapper, not additional architecture) ----
class InterDenoiser(nn.Module):
    def __init__(
        self,
        input_feats,
        latent_dim=512,
        num_frames=240,
        ff_size=1024,
        num_layers=8,
        num_heads=8,
        dropout=0.1,
        activation="gelu",
        cfg_weight=0.0,
        **kargs,
    ):
        super().__init__()

        self.cfg_weight = cfg_weight
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim

        self.text_emb_dim = 768

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # Input Embedding
        self.motion_embed = nn.Linear(self.input_feats, self.latent_dim)
        self.text_embed = nn.Linear(self.text_emb_dim, self.latent_dim)

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(
                TransformerBlock(
                    num_heads=num_heads, latent_dim=latent_dim, dropout=dropout, ff_size=ff_size
                )
            )
        # Output Module
        self.out = zero_module(FinalLayer(self.latent_dim, self.input_feats))

    def forward(self, x, timesteps, mask=None, cond=None):
        """
        x: B, T, D
        """
        B, T = x.shape[0], x.shape[1]
        x_a, x_b = x[..., : self.input_feats], x[..., self.input_feats :]

        if mask is not None:
            mask = mask[..., 0]

        emb = self.embed_timestep(timesteps) + self.text_embed(cond)

        a_emb = self.motion_embed(x_a)
        b_emb = self.motion_embed(x_b)
        h_a_prev = self.sequence_pos_encoder(a_emb)
        h_b_prev = self.sequence_pos_encoder(b_emb)

        if mask is None:
            mask = torch.ones(B, T).to(x_a.device)
        key_padding_mask = ~(mask > 0.5)

        for i, block in enumerate(self.blocks):
            h_a = block(h_a_prev, h_b_prev, emb, key_padding_mask)
            h_b = block(h_b_prev, h_a_prev, emb, key_padding_mask)
            h_a_prev = h_a
            h_b_prev = h_b

        output_a = self.out(h_a)
        output_b = self.out(h_b)

        output = torch.cat([output_a, output_b], dim=-1)

        return output


# ---- staging build/example helpers (tiny sizes for fast tracing) ----
def build_intergen_denoiser():
    torch.manual_seed(0)
    return InterDenoiser(
        input_feats=32,
        latent_dim=64,
        num_frames=16,
        ff_size=128,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
    )


def example_input_intergen_denoiser():
    torch.manual_seed(0)
    batch_size, seq_len = 2, 16
    # x packs both interacting humans' motion features along the last dim (2*input_feats)
    x = torch.randn(batch_size, seq_len, 64)
    timesteps = torch.randint(0, 1000, (batch_size,))
    cond = torch.randn(batch_size, 768)
    return (x, timesteps, None, cond)


MENAGERIE_ENTRIES = [
    (
        "InterGen",
        build_intergen_denoiser,
        example_input_intergen_denoiser,
        2024,
        "vendored-pytorch",
    ),
]
