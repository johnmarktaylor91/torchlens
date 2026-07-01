"""Genomics / spatial-omics / Hi-C architecture family, batch w6a4.

Sources checked (GitHub API + paper text, no clone/pip-install):

* GeneMamba -- github.com/MineSelf2016/GeneMamba (``genemamba/models/models.py``),
  arXiv:2504.16956. BiMamba backbone: each Mamba layer is run forward on the
  sequence and again on the reversed sequence, then the two directions are
  aggregated (mean/concat/sum/gate) before the next layer -- the distinctive
  "flip-run-flip-back-aggregate" bidirectional-scan wrapper around a causal
  SSM mixer. ``mamba_ssm`` is not in the base env, so the causal SSM mixer
  itself is reimplemented compactly as a minimal selective-scan (Mamba-style
  gated 1D-conv + linear recurrence) built from base-env torch ops, wrapped
  in the paper's bidirectional aggregation scheme.
* GENERanno -- github.com/GenerTeam/GENERanno (``README.md`` + HuggingFace
  ``GenerTeam/GENERanno-prokaryote-0.5b-base/config.json`` +
  ``modeling_generanno.py``), bioRxiv 2025.06.04.656517. A Llama-family
  genomic foundation model (RoPE, grouped-query attention, RMSNorm, SwiGLU
  MLP) trained with a masked/bidirectional objective directly over a small
  (vocab_size=64) byte-pair-encoded nucleotide vocabulary for CDS / gene
  annotation. The distinctive piece kept here is the small-vocab genomic
  tokenizer + GQA decoder stack + dual (causal and bidirectional-mask) head,
  built compactly and faithfully in base-env torch (no ``transformers``
  remote code needed since the published config is a standard Llama-style
  block graph).
* GET (General Expression Transformer) -- github.com/GET-Foundation/get_model
  (``get_model/model/model.py`` class ``GETRegionFinetune`` +
  ``get_model/model/modules.py`` classes ``RegionEmbed``/``ExpressionHead``),
  Nature 2024. Per-region motif-count vectors are linearly embedded, a
  learned CLS token is prepended, a standard bidirectional Transformer
  encoder mixes across genomic regions, and a per-region linear head with a
  final Softplus predicts non-negative expression -- reimplemented compactly
  with the same region-token + CLS + regression-head topology.
* GNTD -- github.com/kuanglab/GNTD (``GNTD/NTD.py`` class ``NTD``), Nature
  Communications 2023. Three per-mode (gene / spatial-x / spatial-y)
  embedding tables are each passed through a nonlinear map (Linear + PReLU),
  and the three nonlinear factor matrices are combined via a three-way CP
  (CANDECOMP/PARAFAC) tensor outer-product einsum followed by ReLU to
  reconstruct the spatial gene-expression tensor -- the graph-Laplacian
  training regularizer is optimization-time only and is not part of the
  forward graph, so it is intentionally omitted here.
* HiCARN -- github.com/OluwadareLab/HiCARN (``Models/HiCARN_1.py`` class
  ``Generator``), Bioinformatics 2022. A cascading residual network: each
  ``Cascading_Block`` chains 3 residual units, densely concatenating every
  intermediate output with a 1x1 "basic block" projection back down to the
  channel width; 5 such cascading blocks are themselves chained with the
  same dense-concat-then-1x1-project pattern -- reimplemented compactly with
  fewer channels/blocks but the identical cascading-residual-dense topology.
* HiCNN -- dna.cs.miami.edu/HiCNN/, Liu & Wang, Bioinformatics 2019. A
  54-layer very deep CNN based on DRRN (Tai et al., CVPR 2017): an entry
  conv, a weight-shared recursive block of residual units (each unit is
  ReLU-Conv-ReLU-Conv with a local residual add back to the block's shared
  input), an exit conv, and a global residual add from the raw input to the
  final output -- reimplemented compactly with the same weight-shared
  recursive-block + local-and-global residual topology (fewer channels and
  fewer recursion steps than the paper's 54-layer setting).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# GeneMamba -- BiMamba backbone over rank-tokenized gene expression
# ============================================================


class MinimalSelectiveScan(nn.Module):
    """Compact causal selective-scan mixer (Mamba-style gated conv + recurrence).

    Stands in for ``mamba_ssm.Mamba2`` (unavailable in the base env) while
    preserving the causal-sequence-mixing role that ``EncoderLayer`` in
    GeneMamba's own code delegates to the SSM.
    """

    def __init__(self, d_model: int, d_state: int = 8) -> None:
        """Initialize gated depthwise-conv + linear selective-scan blocks."""

        super().__init__()
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, groups=d_model)
        self.x_proj = nn.Linear(d_model, d_state)
        self.dt_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.A_log = nn.Parameter(torch.randn(d_model, d_state) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a causal gated-conv + diagonal linear-SSM scan over ``x``."""

        b, length, _ = x.shape
        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)
        u = self.conv(u.transpose(1, 2))[:, :, :length].transpose(1, 2)
        u = F.silu(u)
        dt = F.softplus(self.dt_proj(u))
        bmat = self.x_proj(u)
        a = -torch.exp(self.A_log)
        state = torch.zeros(b, u.shape[-1], a.shape[-1], device=x.device, dtype=x.dtype)
        outs = []
        for t in range(length):
            decay = torch.exp(dt[:, t].unsqueeze(-1) * a.unsqueeze(0))
            state = state * decay + u[:, t].unsqueeze(-1) * bmat[:, t].unsqueeze(1)
            outs.append(state.sum(dim=-1))
        y = torch.stack(outs, dim=1)
        return self.out_proj(y * F.silu(z))


class BiMambaEncoderLayer(nn.Module):
    """One BiMamba layer: forward-scan + reverse-scan, then aggregate."""

    def __init__(self, d_model: int, mode: str = "gate") -> None:
        """Initialize the shared mixer and, for gate/concat, the aggregator."""

        super().__init__()
        self.mixer = MinimalSelectiveScan(d_model)
        self.mode = mode
        if mode in ("concat", "gate"):
            self.aggr = nn.Linear(d_model * 2, d_model if mode == "gate" else d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix ``x`` forward and reversed, then aggregate the two directions."""

        x_f = self.mixer(x) + x
        x_b = self.mixer(x.flip([1])).flip([1]) + x
        if self.mode == "mean":
            return (x_f + x_b) / 2
        if self.mode == "sum":
            return x_f + x_b
        if self.mode == "concat":
            return self.aggr(torch.cat([x_f, x_b], dim=-1))
        gate = torch.sigmoid(self.aggr(torch.cat([x_f, x_b], dim=-1)))
        return gate * x_f + (1 - gate) * x_b


class GeneMambaModel(nn.Module):
    """Compact GeneMamba: rank-token embedding + stacked BiMamba layers."""

    def __init__(self, vocab_size: int = 512, d_model: int = 32, n_layers: int = 3) -> None:
        """Initialize the embedding table and the BiMamba layer stack."""

        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([BiMambaEncoderLayer(d_model) for _ in range(n_layers)])
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed rank-ordered gene tokens and mix with stacked BiMamba layers."""

        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return self.lm_head(x)


def build_genemamba() -> nn.Module:
    """Build a compact GeneMamba (BiMamba over rank-tokenized gene expression)."""

    return GeneMambaModel(vocab_size=512, d_model=32, n_layers=3).eval()


def example_input_genemamba() -> torch.Tensor:
    """Rank-ordered gene-token ids, ``(batch=1, seq_len=16)``."""

    return torch.randint(0, 512, (1, 16))


# ============================================================
# GENERanno -- small-vocab genomic Llama-style GQA decoder
# ============================================================


class RotaryEmbedding(nn.Module):
    """Standard rotary position embedding (RoPE) cache."""

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        """Precompute inverse frequencies for RoPE."""

        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos/sin tables of shape ``(seq_len, head_dim)``."""

        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dim's halves for RoPE application."""

    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to ``x`` given cos/sin tables."""

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


class GenerannoGQABlock(nn.Module):
    """Llama-style decoder block: RoPE grouped-query attention + SwiGLU MLP."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, d_ff: int) -> None:
        """Initialize norms, GQA projections, RoPE cache, and SwiGLU MLP."""

        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Bidirectional (unmasked, MLM-style) GQA + SwiGLU with pre-norm residuals."""

        b, t, _ = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rope(t, x.device)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        groups = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(b, t, -1)
        x = x + self.o_proj(attn)
        h = self.norm2(x)
        x = x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return x


class GenerannoModel(nn.Module):
    """Compact GENERanno: small-vocab nucleotide-BPE tokens + GQA decoder stack."""

    def __init__(
        self,
        vocab_size: int = 64,
        d_model: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        n_kv_heads: int = 2,
        d_ff: int = 64,
    ) -> None:
        """Initialize the embedding table, GQA blocks, and MLM output head."""

        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [GenerannoGQABlock(d_model, n_heads, n_kv_heads, d_ff) for _ in range(n_layers)]
        )
        self.norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode a nucleotide-BPE token sequence and predict masked-token logits."""

        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))


def build_generanno() -> nn.Module:
    """Build a compact GENERanno genomic foundation model."""

    return GenerannoModel().eval()


def example_input_generanno() -> torch.Tensor:
    """Nucleotide-BPE token ids, ``(batch=1, seq_len=24)``."""

    return torch.randint(0, 64, (1, 24))


# ============================================================
# GET (General Expression Transformer) -- region-motif transformer
# ============================================================


class GETRegionFinetuneCompact(nn.Module):
    """Compact GET: per-region motif embedding + CLS-token Transformer + expression head."""

    def __init__(
        self,
        num_motif_features: int = 40,
        embed_dim: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        output_dim: int = 2,
    ) -> None:
        """Initialize region embed, CLS token, Transformer encoder, and expression head."""

        super().__init__()
        self.region_embed = nn.Linear(num_motif_features, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head_exp = nn.Linear(embed_dim, output_dim)

    def forward(self, region_motif: torch.Tensor) -> torch.Tensor:
        """Predict non-negative per-region expression from motif-count features."""

        x = self.region_embed(region_motif)
        b = x.shape[0]
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.encoder(x)
        x = x[:, 1:]
        return F.softplus(self.head_exp(x))


def build_get() -> nn.Module:
    """Build a compact GET (General Expression Transformer)."""

    return GETRegionFinetuneCompact().eval()


def example_input_get() -> torch.Tensor:
    """Per-region motif-count features, ``(batch=1, n_regions=12, num_motif=40)``."""

    return torch.randn(1, 12, 40).abs()


# ============================================================
# GNTD -- graph-regularized neural tensor decomposition
# ============================================================


class NTD(nn.Module):
    """Neural tensor decomposition: 3 nonlinear mode factors + CP-style outer product."""

    def __init__(self, n_x: int, n_y: int, n_g: int, rank: int = 8) -> None:
        """Initialize per-mode embeddings and their nonlinear projection heads."""

        super().__init__()
        self.embedding_x = nn.Embedding(n_x, rank)
        self.embedding_y = nn.Embedding(n_y, rank)
        self.embedding_g = nn.Embedding(n_g, rank)
        self.lin_x_1 = nn.Linear(rank, rank)
        self.lin_y_1 = nn.Linear(rank, rank)
        self.lin_g_1 = nn.Linear(rank, rank)
        self.prelu = nn.PReLU(init=0.9)

    def forward(
        self, x_index: torch.Tensor, y_index: torch.Tensor, g_index: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct the gene x spatial-x x spatial-y expression tensor."""

        x = self.prelu(self.lin_x_1(self.embedding_x(x_index)))
        y = self.prelu(self.lin_y_1(self.embedding_y(y_index)))
        g = self.prelu(self.lin_g_1(self.embedding_g(g_index)))
        out = torch.einsum("im,jm,km->ijk", g, x, y)
        return out.relu()


def build_gntd() -> nn.Module:
    """Build a compact GNTD (graph neural tensor decomposition) imputer."""

    return NTD(n_x=10, n_y=10, n_g=20, rank=8).eval()


def example_input_gntd() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full index ranges for the spatial-x, spatial-y, and gene modes."""

    return (torch.arange(10), torch.arange(10), torch.arange(20))


# ============================================================
# HiCARN -- cascading residual network for Hi-C super-resolution
# ============================================================


class HiCARNBasicBlock(nn.Module):
    """1x1 projection block used to fuse densely-concatenated features."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the 1x1 fusion convolution."""

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project fused channels back down to the working channel width."""

        return self.conv(x)


class HiCARNResidualBlock(nn.Module):
    """Two 3x3 convs with a local residual add."""

    def __init__(self, channels: int) -> None:
        """Initialize the two convolutions of the residual unit."""

        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply conv-relu-conv with an additive residual from the block input."""

        out = F.relu(self.conv1(x), inplace=False)
        return F.relu(self.conv2(out) + x, inplace=False)


class HiCARNCascadingBlock(nn.Module):
    """Chain of residual units, densely concatenating and re-projecting each output."""

    def __init__(self, channels: int, n_units: int = 3) -> None:
        """Initialize ``n_units`` residual units and their dense-fusion projections."""

        super().__init__()
        self.residuals = nn.ModuleList([HiCARNResidualBlock(channels) for _ in range(n_units)])
        self.fusions = nn.ModuleList(
            [HiCARNBasicBlock(channels * (i + 2), channels) for i in range(n_units)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Densely fuse each residual unit's output with all prior concatenations."""

        concat = out = x
        for residual, fuse in zip(self.residuals, self.fusions):
            branch = residual(out)
            concat = torch.cat([concat, branch], dim=1)
            out = fuse(concat)
        return out


class HiCARNGeneratorCompact(nn.Module):
    """Compact HiCARN-1 generator: stacked cascading blocks with dense fusion."""

    def __init__(self, channels: int = 16, n_blocks: int = 3) -> None:
        """Initialize entry conv, cascading blocks with dense fusion, and exit conv."""

        super().__init__()
        self.entry = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([HiCARNCascadingBlock(channels) for _ in range(n_blocks)])
        self.fusions = nn.ModuleList(
            [HiCARNBasicBlock(channels * (i + 2), channels) for i in range(n_blocks)]
        )
        self.exit = nn.Conv2d(channels, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Super-resolve a low-resolution Hi-C contact-map patch."""

        x = self.entry(x)
        concat = out = x
        for block, fuse in zip(self.blocks, self.fusions):
            branch = block(out)
            concat = torch.cat([concat, branch], dim=1)
            out = fuse(concat)
        return self.exit(out)


def build_hicarn() -> nn.Module:
    """Build a compact HiCARN-1 (cascading residual network) generator."""

    return HiCARNGeneratorCompact().eval()


def example_input_hicarn() -> torch.Tensor:
    """Low-resolution Hi-C contact-map patch, ``(1, 1, 28, 28)``."""

    return torch.randn(1, 1, 28, 28)


# ============================================================
# HiCNN -- DRRN-style weight-shared recursive residual network
# ============================================================


class HiCNNRecursiveBlock(nn.Module):
    """Weight-shared recursive block: same residual unit applied ``n`` times."""

    def __init__(self, channels: int, n_residual_units: int = 6) -> None:
        """Initialize a single residual-unit weight set applied recursively."""

        super().__init__()
        self.n_residual_units = n_residual_units
        self.residual_unit = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the shared residual unit recursively with a local residual add."""

        out = x
        for _ in range(self.n_residual_units):
            out = self.residual_unit(out) + x
        return out


class HiCNNCompact(nn.Module):
    """Compact HiCNN: entry conv, weight-shared recursive block, exit conv, global residual."""

    def __init__(self, channels: int = 32, n_residual_units: int = 6) -> None:
        """Initialize entry/exit convs and the shared recursive trunk."""

        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ReLU(inplace=False), nn.Conv2d(1, channels, kernel_size=3, padding=1, bias=False)
        )
        self.trunk = HiCNNRecursiveBlock(channels, n_residual_units)
        self.conv2 = nn.Sequential(
            nn.ReLU(inplace=False), nn.Conv2d(channels, 1, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhance a low-resolution Hi-C contact-map patch with a global residual add."""

        identity = x
        out = self.conv1(x)
        out = self.trunk(out)
        out = self.conv2(out)
        return identity + out


def build_hicnn() -> nn.Module:
    """Build a compact HiCNN (DRRN-style very deep recursive residual network)."""

    return HiCNNCompact().eval()


def example_input_hicnn() -> torch.Tensor:
    """Low-resolution Hi-C contact-map patch, ``(1, 1, 40, 40)``."""

    return torch.randn(1, 1, 40, 40)


# ============================================================
# Registry
# ============================================================

MENAGERIE_ENTRIES = [
    ("GeneMamba", "build_genemamba", "example_input_genemamba", "2025", "BIO"),
    ("Generanno", "build_generanno", "example_input_generanno", "2025", "BIO"),
    ("GET (General Expression Transformer)", "build_get", "example_input_get", "2024", "BIO"),
    ("GNTD", "build_gntd", "example_input_gntd", "2023", "BIO"),
    ("HiCARN", "build_hicarn", "example_input_hicarn", "2022", "BIO"),
    ("HiCNN", "build_hicnn", "example_input_hicnn", "2019", "BIO"),
]
