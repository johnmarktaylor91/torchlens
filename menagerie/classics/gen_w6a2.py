"""Menagerie batch w6a2: single-cell/genomics foundation models and DNA sequence-to-function models.

Sources checked (reference only; no cloning, no pip installs):
  - DigNet (cand_00712): Wang Chuanyuan et al., Genome Research 2024/2025
    "Diffusion-based generation of gene regulatory networks from scRNA-seq
    data with DigNet", official repo https://github.com/zpliulab/DigNet
    (confirmed via GitHub API + DeepWiki summary of the repo). The
    distinctive mechanism is a **discrete graph-diffusion denoiser**: genes
    are nodes of a graph carrying node features ``X`` (expression), edge
    features ``E`` (candidate regulatory-edge state, the quantity being
    diffused/denoised) and a global feature ``y``; a ``DigNetGraphTransformer``
    made of stacked ``NodeEdgeBlock`` layers jointly updates node and edge
    representations via FiLM-modulated multi-head attention (edge features
    gate the attention logits, and are themselves updated from the attention
    scores), conditioned on a diffusion timestep embedding injected into the
    global feature. This module reproduces the joint node/edge graph-
    transformer denoiser block (attention gated by edge features, edge
    features updated from attention output, timestep conditioning of node
    and edge streams) operating on a compact synthetic gene graph, followed
    by a head that reads out edge (regulatory-link) logits -- the
    architecture-defining mechanism, independent of the outer multi-step
    diffusion sampling loop (which is not itself a traceable nn.Module).
  - DNABERT (cand_00713): Ji, Zhou, Liu & Davuluri, Bioinformatics 2021
    "DNABERT: pre-trained Bidirectional Encoder Representations from
    Transformers model for DNA-language in genome", official repo
    https://github.com/jerryji1993/DNABERT (README fetched via GitHub API).
    DNABERT tokenizes DNA into overlapping **k-mers (k=3..6)** and forms a
    fixed k-mer vocabulary (4**k + special tokens), then trains a standard
    BERT encoder (the repo forks huggingface/transformers directly) with
    masked k-mer-language-model pretraining. This is architecturally
    *distinct* from DNABERT-2 (byte-pair-encoding tokenizer + ALiBi,
    already registered in this catalog as ``dnabert2``/``DNABERT-S``): the
    original DNABERT's defining mechanism is the overlapping k-mer
    vocabulary/tokenization feeding a vanilla BERT stack, which is what this
    module reproduces (k=6 overlapping-kmer embedding table sized
    ``4**6 + 5`` feeding a compact ``TransformerEncoder`` stack with a
    masked-kmer-LM head), not the BPE tokenizer of the v2 lineage.
  - Enigma (cand_00715): DeepGenomics, bioRxiv 2025.12.18.694875 "Enigma: An
    Efficient Model for Deciphering Regulatory Genomics", repo
    https://github.com/deepgenomics/enigma (existence confirmed via GitHub
    API; architecture confirmed via bioRxiv abstract/search summary). Enigma
    is a 202M-parameter **UNet-transformer hybrid** that ingests long
    (524,288 bp) one-hot DNA windows, downsamples with a strided
    convolutional tower (Enformer/Borzoi-lineage conv-tower design), applies
    a self-attention transformer bottleneck over the downsampled sequence,
    then **upsamples back with a symmetric transposed-conv tower using
    U-Net-style skip connections from each downsampling stage** to recover
    single-base resolution (the key differentiator vs. Borzoi/Enformer,
    which output at 32 bp bins without a decoder-side upsampling path), and
    reads out multiple per-base genomic tracks (RNA-seq/DNase/ATAC/splice
    junction channels). This module reproduces the strided-conv-tower ->
    transformer-bottleneck -> transposed-conv-tower-with-skip-connections
    U-Net topology at a compact scale (short sequence window, few channels/
    layers) with a multi-track single-base output head.
  - EPCOT (cand_00716): Zhang, Liu et al., Nucleic Acids Research 2023 "A
    generalizable framework to comprehensively predict epigenome, chromatin
    organization, and transcriptome", official repo
    https://github.com/liu-bioinfo-lab/EPCOT (README fetched via GitHub
    API). EPCOT jointly encodes a **1.6 kb one-hot DNA window and paired
    cell-type-specific chromatin-accessibility (DNase/ATAC-seq) signal**
    through a shared convolutional stem (concatenated as extra input
    channels) followed by a transformer encoder layer that produces a
    generic sequence representation; **multiple task-specific decoder heads**
    then branch off this shared representation to predict epigenomic
    tracks, gene expression, and (in the full model) chromatin contact maps
    -- the defining "one encoder, many genomic-modality decoders" topology.
    This module reproduces the dual-channel (DNA + accessibility) conv stem
    + transformer encoder + multi-task decoder-head topology at a compact
    scale.
  - EpiAgent (cand_00717): Chen, Li, Cui et al., Nature Methods 2025
    "EpiAgent: foundation model for single-cell epigenomics", official repo
    https://github.com/xy-chen16/EpiAgent (README fetched via GitHub API).
    EpiAgent encodes each cell's chromatin-accessibility profile (a scATAC-
    seq cell-by-cCRE matrix) as a **"cell sentence"**: the accessible cCREs
    for that cell, ranked by TF-IDF importance and truncated to a fixed
    length, tokenized as a sequence of peak-identity tokens (plus a
    prepended ``[CLS]`` token) fed through a **bidirectional
    (BERT-style, non-causal) Transformer encoder**, whose ``[CLS]`` output
    embedding is the cell representation used for all downstream tasks
    (annotation, imputation, perturbation response). This module reproduces
    the cell-sentence tokenization (peak-id embedding + rank-position
    embedding + prepended CLS) feeding a bidirectional transformer encoder
    with a CLS-pooled cell-embedding head, at compact scale.
  - EpiFoundation (cand_00718): Wu, Wan, Ji, Zhou & Hou, bioRxiv
    2025.02.05.636688 "EpiFoundation: A Foundation Model for Single-Cell
    ATAC-seq via Peak-to-Gene Alignment", official repo
    https://github.com/UCSC-VLAA/EpiFoundation (README + constructor
    signature fetched via GitHub API: ``EpiFoundation(num_class_cell,
    num_rnas, num_atacs, num_values, num_chrs, embed_dim, depth, heads,
    head_dim, encoder, ...)``). The defining mechanism is **dual-stream
    cross-modality pretraining**: a scRNA-seq transformer encoder (dense,
    binned gene-expression tokens) and a scATAC-seq transformer encoder
    (sparse, *non-zero-peak-only* token set -- the paper's key sparsity
    trick to avoid processing the huge zero-dominated peak space) each
    produce a pooled cell embedding; the ATAC-stream cell embedding is
    projected and trained (masked-value-prediction / MVC-style) to align
    with dense gene-expression targets from the paired RNA stream, i.e. an
    explicit **peak-to-gene cross-modal alignment head** on top of two
    parallel transformer towers. This module reproduces the two-tower
    (RNA-dense-token-encoder + ATAC-sparse-nonzero-token-encoder)
    architecture with a cross-modal peak-to-gene alignment projection head,
    at compact scale.
"""

from __future__ import annotations

import math

import torch
from torch import nn


# ---------------------------------------------------------------------------
# DigNet: joint node/edge graph-transformer diffusion denoiser.
# ---------------------------------------------------------------------------
class NodeEdgeBlock(nn.Module):
    """Joint node/edge graph-transformer block, DigNet's ``NodeEdgeBlock``.

    Multi-head attention logits between gene nodes are gated (FiLM-style) by
    the current edge-feature state, and the (post-softmax) attention weights
    are in turn used to refresh the edge features -- so node and edge
    streams update each other every block, which is the mechanism DigNet
    uses to keep denoised regulatory-edge predictions consistent with
    denoised node (expression) predictions.
    """

    def __init__(self, dim: int, edge_dim: int, n_heads: int = 4) -> None:
        """Initialize the joint node/edge attention block.

        Parameters
        ----------
        dim : int
            Node feature dimension.
        edge_dim : int
            Edge feature dimension.
        n_heads : int
            Number of attention heads.
        """
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.edge_gate = nn.Linear(edge_dim, n_heads)
        self.edge_update = nn.Linear(n_heads, edge_dim)
        self.out_proj = nn.Linear(dim, dim)
        self.node_norm = nn.LayerNorm(dim)
        self.edge_norm = nn.LayerNorm(edge_dim)
        self.time_to_node = nn.Linear(dim, dim)
        self.time_to_edge = nn.Linear(dim, edge_dim)

    def forward(
        self, x: torch.Tensor, e: torch.Tensor, t_embed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Jointly refresh node and edge features for one denoising block.

        Parameters
        ----------
        x : torch.Tensor
            Node features, shape ``(n_nodes, dim)``.
        e : torch.Tensor
            Edge features, shape ``(n_nodes, n_nodes, edge_dim)``.
        t_embed : torch.Tensor
            Diffusion timestep embedding, shape ``(dim,)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated ``(x, e)`` with the same shapes as the inputs.
        """
        n = x.shape[0]
        x_t = x + self.time_to_node(t_embed).unsqueeze(0)

        q = self.q_proj(x_t).view(n, self.n_heads, self.head_dim)
        k = self.k_proj(x_t).view(n, self.n_heads, self.head_dim)
        v = self.v_proj(x_t).view(n, self.n_heads, self.head_dim)

        logits = torch.einsum("ihd,jhd->ijh", q, k) / math.sqrt(self.head_dim)
        # FiLM-style gate: current edge state modulates attention logits.
        logits = logits + self.edge_gate(e)
        attn = torch.softmax(logits, dim=1)

        # Refresh edge features from the attention weights themselves.
        e_new = self.edge_norm(
            e + self.edge_update(attn) + self.time_to_edge(t_embed).view(1, 1, -1)
        )

        out = torch.einsum("ijh,jhd->ihd", attn, v).reshape(n, -1)
        x_new = self.node_norm(x + self.out_proj(out))
        return x_new, e_new


class DigNetGraphTransformer(nn.Module):
    """DigNet's denoising network: stacked joint node/edge transformer blocks.

    Reads a noised gene-regulatory graph (node = gene expression, edge =
    candidate regulatory-link state) plus a diffusion timestep, and outputs
    denoised edge (regulatory-link) logits, mirroring the role of
    ``DigNetGraphTransformer`` inside the repo's ``GaussianDiffusion1D``
    wrapper (the outer multi-step sampling loop is not itself a traceable
    ``nn.Module`` and is intentionally not reproduced here).
    """

    def __init__(
        self,
        dim: int = 32,
        edge_dim: int = 8,
        n_blocks: int = 3,
        n_heads: int = 4,
    ) -> None:
        """Initialize the DigNet graph-transformer denoiser.

        Parameters
        ----------
        dim : int
            Node feature/embedding dimension.
        edge_dim : int
            Edge feature/embedding dimension.
        n_blocks : int
            Number of stacked ``NodeEdgeBlock`` layers.
        n_heads : int
            Number of attention heads per block.
        """
        super().__init__()
        self.node_in = nn.Linear(1, dim)
        self.edge_in = nn.Linear(1, edge_dim)
        self.time_embed = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(
            [NodeEdgeBlock(dim, edge_dim, n_heads) for _ in range(n_blocks)]
        )
        self.edge_out = nn.Linear(edge_dim, 1)

    def forward(
        self, node_expr: torch.Tensor, edge_state: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """Denoise a candidate gene-regulatory graph.

        Parameters
        ----------
        node_expr : torch.Tensor
            Per-gene expression, shape ``(n_genes, 1)``.
        edge_state : torch.Tensor
            Noised candidate regulatory-edge state, shape
            ``(n_genes, n_genes, 1)``.
        timestep : torch.Tensor
            Scalar diffusion timestep, shape ``(1,)``.

        Returns
        -------
        torch.Tensor
            Denoised edge logits, shape ``(n_genes, n_genes)``.
        """
        x = self.node_in(node_expr)
        e = self.edge_in(edge_state)
        t_embed = self.time_embed(timestep.view(1, 1).float()).squeeze(0)
        for block in self.blocks:
            x, e = block(x, e, t_embed)
        return self.edge_out(e).squeeze(-1)


def build_dignet() -> nn.Module:
    """Build a compact DigNet graph-transformer diffusion denoiser.

    Returns
    -------
    nn.Module
        DigNet reconstruction in evaluation mode.
    """
    return DigNetGraphTransformer(dim=32, edge_dim=8, n_blocks=3, n_heads=4).eval()


def example_input_dignet() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create example input for :func:`build_dignet`.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(node_expr, edge_state, timestep)`` for a 12-gene toy graph.
    """
    n_genes = 12
    node_expr = torch.randn(n_genes, 1)
    edge_state = torch.randn(n_genes, n_genes, 1)
    timestep = torch.tensor([0.4])
    return node_expr, edge_state, timestep


# ---------------------------------------------------------------------------
# DNABERT: overlapping k-mer tokenization + vanilla BERT encoder.
# ---------------------------------------------------------------------------
class DNABERT(nn.Module):
    """Original DNABERT: overlapping-kmer embedding + BERT encoder + MLM head.

    Distinct from DNABERT-2 (already in this catalog), which replaced the
    fixed overlapping-kmer vocabulary with byte-pair encoding. Here the DNA
    sequence is assumed pre-tokenized into overlapping 6-mer ids (vocabulary
    size ``4**6`` plus 5 special tokens), embedded, added to learned
    positional embeddings, and passed through a standard
    ``TransformerEncoder`` stack (the repo forks huggingface/transformers'
    ``BertModel`` directly) with a final masked-kmer-language-model head.
    """

    def __init__(
        self,
        kmer: int = 6,
        hidden_dim: int = 96,
        n_layers: int = 4,
        n_heads: int = 4,
        max_len: int = 64,
    ) -> None:
        """Initialize DNABERT.

        Parameters
        ----------
        kmer : int
            k-mer size (3-6 in the original repo); sets the vocabulary size.
        hidden_dim : int
            Transformer hidden dimension.
        n_layers : int
            Number of transformer encoder layers.
        n_heads : int
            Number of attention heads.
        max_len : int
            Maximum overlapping-kmer sequence length.
        """
        super().__init__()
        vocab_size = 4**kmer + 5  # 4**k kmers + [PAD]/[MASK]/[CLS]/[SEP]/[UNK]
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_len, hidden_dim)
        self.embed_norm = nn.LayerNorm(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.mlm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, kmer_ids: torch.Tensor) -> torch.Tensor:
        """Predict masked-kmer logits for an overlapping-kmer DNA sequence.

        Parameters
        ----------
        kmer_ids : torch.Tensor
            Shape ``(batch, seq_len)`` overlapping-kmer token ids.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, seq_len, vocab_size)`` masked-kmer-LM logits.
        """
        seq_len = kmer_ids.shape[1]
        positions = torch.arange(seq_len, device=kmer_ids.device).unsqueeze(0)
        h = self.token_embed(kmer_ids) + self.pos_embed(positions)
        h = self.embed_norm(h)
        h = self.encoder(h)
        return self.mlm_head(h)


def build_dnabert() -> nn.Module:
    """Build a compact original-DNABERT (overlapping 6-mer BERT) model.

    Returns
    -------
    nn.Module
        DNABERT reconstruction in evaluation mode.
    """
    return DNABERT(kmer=6, hidden_dim=96, n_layers=4, n_heads=4, max_len=64).eval()


def example_input_dnabert() -> torch.Tensor:
    """Create example input for :func:`build_dnabert`.

    Returns
    -------
    torch.Tensor
        Shape ``(2, 48)`` overlapping-6mer token ids (vocab size ``4**6+5``).
    """
    vocab_size = 4**6 + 5
    return torch.randint(0, vocab_size, (2, 48))


# ---------------------------------------------------------------------------
# Enigma: strided-conv tower -> transformer bottleneck -> U-Net upsample tower.
# ---------------------------------------------------------------------------
class ConvDownBlock(nn.Module):
    """One strided-conv downsampling stage of Enigma's encoder tower."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        """Initialize a downsampling conv block.

        Parameters
        ----------
        in_ch : int
            Input channel count.
        out_ch : int
            Output channel count.
        """
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=5, stride=2, padding=2)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample by a factor of 2 along the sequence axis.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, in_ch, length)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, out_ch, length // 2)``.
        """
        return self.act(self.norm(self.conv(x)))


class ConvUpBlock(nn.Module):
    """One transposed-conv upsampling stage with a U-Net skip connection."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        """Initialize an upsampling conv block.

        Parameters
        ----------
        in_ch : int
            Input channel count (from the coarser stage).
        skip_ch : int
            Channel count of the skip connection from the encoder.
        out_ch : int
            Output channel count.
        """
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.merge = nn.Conv1d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample by a factor of 2 and fuse with the matching encoder skip.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, in_ch, length)`` coarser-resolution features.
        skip : torch.Tensor
            Shape ``(batch, skip_ch, 2 * length)`` encoder skip features.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, out_ch, 2 * length)``.
        """
        up = self.up(x)
        up = up[:, :, : skip.shape[-1]]
        merged = torch.cat([up, skip], dim=1)
        return self.act(self.norm(self.merge(merged)))


class Enigma(nn.Module):
    """Enigma: UNet-transformer hybrid for single-base-resolution genomic tracks.

    A strided-conv encoder tower downsamples a one-hot DNA window; a
    self-attention transformer bottleneck operates on the downsampled
    sequence; a symmetric transposed-conv decoder tower upsamples back to
    the original (single-base) resolution using U-Net skip connections from
    every encoder stage, and a final head predicts multiple per-base
    genomic tracks (RNA-seq / DNase / ATAC / splice-junction channels).
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_ch: int = 16,
        depth: int = 3,
        bottleneck_dim: int = 64,
        n_tracks: int = 4,
    ) -> None:
        """Initialize Enigma.

        Parameters
        ----------
        in_channels : int
            Input channels (4 for one-hot ACGT).
        base_ch : int
            Base channel width of the conv tower.
        depth : int
            Number of downsampling/upsampling stages.
        bottleneck_dim : int
            Transformer bottleneck hidden dimension.
        n_tracks : int
            Number of output genomic tracks predicted per base.
        """
        super().__init__()
        self.stem = nn.Conv1d(in_channels, base_ch, kernel_size=7, padding=3)

        channels = [base_ch * (2**i) for i in range(depth + 1)]
        self.down_blocks = nn.ModuleList(
            [ConvDownBlock(channels[i], channels[i + 1]) for i in range(depth)]
        )
        self.to_bottleneck = nn.Conv1d(channels[-1], bottleneck_dim, kernel_size=1)
        bottleneck_layer = nn.TransformerEncoderLayer(
            d_model=bottleneck_dim, nhead=4, dim_feedforward=bottleneck_dim * 2, batch_first=True
        )
        self.bottleneck = nn.TransformerEncoder(bottleneck_layer, num_layers=2)
        self.from_bottleneck = nn.Conv1d(bottleneck_dim, channels[-1], kernel_size=1)

        self.up_blocks = nn.ModuleList(
            [
                ConvUpBlock(channels[depth - i], channels[depth - i - 1], channels[depth - i - 1])
                for i in range(depth)
            ]
        )
        self.track_head = nn.Conv1d(base_ch, n_tracks, kernel_size=1)

    def forward(self, one_hot_dna: torch.Tensor) -> torch.Tensor:
        """Predict per-base genomic tracks from a one-hot DNA window.

        Parameters
        ----------
        one_hot_dna : torch.Tensor
            Shape ``(batch, 4, length)`` one-hot ACGT encoding, ``length``
            divisible by ``2 ** depth``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_tracks, length)`` single-base-resolution track
            predictions.
        """
        h = self.stem(one_hot_dna)
        skips = [h]
        for block in self.down_blocks:
            h = block(h)
            skips.append(h)
        skips = skips[:-1]  # drop the deepest (matches the bottleneck input)

        bottleneck_in = self.to_bottleneck(h).transpose(1, 2)
        bottleneck_out = self.bottleneck(bottleneck_in).transpose(1, 2)
        h = self.from_bottleneck(bottleneck_out)

        for block, skip in zip(self.up_blocks, reversed(skips)):
            h = block(h, skip)

        return self.track_head(h)


def build_enigma() -> nn.Module:
    """Build a compact Enigma UNet-transformer genomic-track predictor.

    Returns
    -------
    nn.Module
        Enigma reconstruction in evaluation mode.
    """
    return Enigma(in_channels=4, base_ch=16, depth=3, bottleneck_dim=64, n_tracks=4).eval()


def example_input_enigma() -> torch.Tensor:
    """Create example input for :func:`build_enigma`.

    Returns
    -------
    torch.Tensor
        Shape ``(1, 4, 256)`` one-hot DNA window (length divisible by 8).
    """
    seq = torch.randint(0, 4, (256,))
    return torch.eye(4)[seq].transpose(0, 1).unsqueeze(0)


# ---------------------------------------------------------------------------
# EPCOT: dual-channel (DNA + accessibility) conv stem + transformer + heads.
# ---------------------------------------------------------------------------
class EPCOT(nn.Module):
    """EPCOT: one shared encoder, multiple genomic-modality decoder heads.

    A convolutional stem jointly encodes a one-hot DNA window and its
    paired cell-type-specific chromatin-accessibility (DNase/ATAC) track
    (concatenated as extra input channels), followed by a transformer
    encoder layer producing a generic sequence representation. Separate
    lightweight decoder heads then branch off that shared representation to
    predict (1) per-position epigenomic-feature tracks, (2) pooled gene
    expression, and (3) a pairwise chromatin-contact map -- mirroring
    EPCOT's "one encoder, many task heads" design.
    """

    def __init__(
        self,
        seq_len: int = 128,
        conv_ch: int = 32,
        hidden_dim: int = 64,
        n_epi_tracks: int = 6,
    ) -> None:
        """Initialize EPCOT.

        Parameters
        ----------
        seq_len : int
            Length of the input DNA / accessibility window.
        conv_ch : int
            Convolutional stem channel width.
        hidden_dim : int
            Transformer encoder hidden dimension.
        n_epi_tracks : int
            Number of predicted epigenomic-feature output tracks.
        """
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(5, conv_ch, kernel_size=9, padding=4),  # 4 (DNA) + 1 (accessibility)
            nn.GELU(),
            nn.Conv1d(conv_ch, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.epigenome_head = nn.Linear(hidden_dim, n_epi_tracks)
        self.expression_head = nn.Sequential(nn.Linear(hidden_dim, 1))
        self.contact_proj = nn.Linear(hidden_dim, 16)

    def forward(
        self, dna_onehot: torch.Tensor, accessibility: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict epigenome, gene expression, and a chromatin contact map.

        Parameters
        ----------
        dna_onehot : torch.Tensor
            Shape ``(batch, 4, seq_len)`` one-hot DNA window.
        accessibility : torch.Tensor
            Shape ``(batch, 1, seq_len)`` paired chromatin-accessibility
            signal (DNase/ATAC-seq).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``(epigenome_tracks, gene_expression, contact_map)`` with shapes
            ``(batch, seq_len, n_epi_tracks)``, ``(batch,)``, and
            ``(batch, seq_len, seq_len)`` respectively.
        """
        stacked = torch.cat([dna_onehot, accessibility], dim=1)
        h = self.stem(stacked).transpose(1, 2)  # (batch, seq_len, hidden_dim)
        h = self.encoder(h)

        epigenome_tracks = self.epigenome_head(h)
        gene_expression = self.expression_head(h.mean(dim=1)).squeeze(-1)
        contact_repr = self.contact_proj(h)
        contact_map = torch.einsum("bid,bjd->bij", contact_repr, contact_repr)
        return epigenome_tracks, gene_expression, contact_map


def build_epcot() -> nn.Module:
    """Build a compact EPCOT multi-task genomic-modality model.

    Returns
    -------
    nn.Module
        EPCOT reconstruction in evaluation mode.
    """
    return EPCOT(seq_len=128, conv_ch=32, hidden_dim=64, n_epi_tracks=6).eval()


def example_input_epcot() -> tuple[torch.Tensor, torch.Tensor]:
    """Create example input for :func:`build_epcot`.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(dna_onehot, accessibility)`` shaped ``(2, 4, 128)`` and
        ``(2, 1, 128)``.
    """
    seq = torch.randint(0, 4, (2, 128))
    dna_onehot = torch.eye(4)[seq].transpose(1, 2)
    accessibility = torch.rand(2, 1, 128)
    return dna_onehot, accessibility


# ---------------------------------------------------------------------------
# EpiAgent: cell-sentence tokenization + bidirectional transformer encoder.
# ---------------------------------------------------------------------------
class EpiAgent(nn.Module):
    """EpiAgent: scATAC-seq "cell sentences" through a bidirectional encoder.

    Each cell's accessible cCREs (ranked by TF-IDF importance) are treated
    as tokens of a "cell sentence": a peak-identity embedding plus a
    rank-position embedding, with a prepended ``[CLS]`` token, fed through a
    bidirectional (BERT-style) transformer encoder. The pooled ``[CLS]``
    output is EpiAgent's cell embedding, used for all downstream tasks.
    """

    def __init__(
        self,
        n_ccres: int = 512,
        hidden_dim: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        max_sentence_len: int = 32,
    ) -> None:
        """Initialize EpiAgent.

        Parameters
        ----------
        n_ccres : int
            Vocabulary size (number of candidate cis-regulatory elements).
        hidden_dim : int
            Transformer hidden dimension.
        n_layers : int
            Number of transformer encoder layers.
        n_heads : int
            Number of attention heads.
        max_sentence_len : int
            Maximum cell-sentence length (including ``[CLS]``).
        """
        super().__init__()
        self.cls_token_id = n_ccres  # reserved id for [CLS]
        self.peak_embed = nn.Embedding(n_ccres + 1, hidden_dim)
        self.rank_embed = nn.Embedding(max_sentence_len, hidden_dim)
        self.embed_norm = nn.LayerNorm(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cell_head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, peak_ids: torch.Tensor) -> torch.Tensor:
        """Embed a batch of scATAC-seq cell sentences.

        Parameters
        ----------
        peak_ids : torch.Tensor
            Shape ``(batch, sentence_len)`` cCRE token ids, TF-IDF-ranked
            with a prepended ``[CLS]`` id at position 0.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, hidden_dim)`` pooled cell embedding (from the
            ``[CLS]`` position).
        """
        sentence_len = peak_ids.shape[1]
        ranks = torch.arange(sentence_len, device=peak_ids.device).unsqueeze(0)
        h = self.peak_embed(peak_ids) + self.rank_embed(ranks)
        h = self.embed_norm(h)
        h = self.encoder(h)
        cls_repr = h[:, 0, :]
        return self.cell_head(cls_repr)


def build_epiagent() -> nn.Module:
    """Build a compact EpiAgent cell-sentence transformer.

    Returns
    -------
    nn.Module
        EpiAgent reconstruction in evaluation mode.
    """
    return EpiAgent(n_ccres=512, hidden_dim=64, n_layers=3, n_heads=4, max_sentence_len=32).eval()


def example_input_epiagent() -> torch.Tensor:
    """Create example input for :func:`build_epiagent`.

    Returns
    -------
    torch.Tensor
        Shape ``(4, 32)`` cell-sentence token ids: column 0 is the
        ``[CLS]`` id (512), remaining columns are TF-IDF-ranked cCRE ids.
    """
    n_ccres = 512
    body = torch.randint(0, n_ccres, (4, 31))
    cls_col = torch.full((4, 1), n_ccres, dtype=torch.long)
    return torch.cat([cls_col, body], dim=1)


# ---------------------------------------------------------------------------
# EpiFoundation: dual-stream RNA/ATAC transformers + peak-to-gene alignment.
# ---------------------------------------------------------------------------
class TokenTransformerTower(nn.Module):
    """A single-stream binned-token transformer encoder with CLS pooling."""

    def __init__(self, vocab_size: int, hidden_dim: int, n_layers: int, n_heads: int) -> None:
        """Initialize one modality tower.

        Parameters
        ----------
        vocab_size : int
            Number of distinct (gene- or peak-) tokens for this modality.
        hidden_dim : int
            Transformer hidden dimension.
        n_layers : int
            Number of transformer encoder layers.
        n_heads : int
            Number of attention heads.
        """
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size + 1, hidden_dim)  # +1 for [CLS]
        self.value_embed = nn.Embedding(32, hidden_dim)  # binned expression/accessibility value
        self.cls_token_id = vocab_size
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, token_ids: torch.Tensor, value_bins: torch.Tensor) -> torch.Tensor:
        """Encode one modality's tokens and pool the ``[CLS]`` output.

        Parameters
        ----------
        token_ids : torch.Tensor
            Shape ``(batch, seq_len)`` gene/peak identity ids, with
            ``[CLS]`` prepended at position 0 (value bin ignored there).
        value_bins : torch.Tensor
            Shape ``(batch, seq_len)`` discretized expression/accessibility
            value bins.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, hidden_dim)`` pooled ``[CLS]`` cell embedding.
        """
        h = self.token_embed(token_ids) + self.value_embed(value_bins)
        h = self.encoder(h)
        return h[:, 0, :]


class EpiFoundation(nn.Module):
    """EpiFoundation: parallel RNA/ATAC towers with cross-modal alignment.

    A dense scRNA-seq token tower and a sparse (non-zero-peak-only)
    scATAC-seq token tower each produce a pooled cell embedding via
    ``[CLS]``-token pooling; the ATAC-stream embedding is linearly projected
    into the RNA embedding space and trained to align with it
    (peak-to-gene cross-modal alignment), mirroring the constructor
    signature of the official ``EpiFoundation(num_rnas, num_atacs, ...)``
    model.
    """

    def __init__(
        self,
        num_rnas: int = 256,
        num_atacs: int = 512,
        hidden_dim: int = 48,
        depth: int = 2,
        heads: int = 4,
    ) -> None:
        """Initialize EpiFoundation.

        Parameters
        ----------
        num_rnas : int
            Gene vocabulary size for the RNA tower.
        num_atacs : int
            Peak vocabulary size for the ATAC tower.
        hidden_dim : int
            Shared transformer hidden dimension.
        depth : int
            Number of transformer encoder layers per tower.
        heads : int
            Number of attention heads per tower.
        """
        super().__init__()
        self.rna_tower = TokenTransformerTower(num_rnas, hidden_dim, depth, heads)
        self.atac_tower = TokenTransformerTower(num_atacs, hidden_dim, depth, heads)
        self.atac_to_rna_align = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        rna_ids: torch.Tensor,
        rna_values: torch.Tensor,
        atac_ids: torch.Tensor,
        atac_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode paired RNA/ATAC cells and align the ATAC embedding to RNA space.

        Parameters
        ----------
        rna_ids : torch.Tensor
            Shape ``(batch, rna_len)`` dense binned gene tokens (``[CLS]``
            prepended).
        rna_values : torch.Tensor
            Shape ``(batch, rna_len)`` binned expression values.
        atac_ids : torch.Tensor
            Shape ``(batch, atac_len)`` sparse non-zero peak tokens
            (``[CLS]`` prepended).
        atac_values : torch.Tensor
            Shape ``(batch, atac_len)`` binned accessibility values.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``(rna_embedding, atac_embedding, atac_aligned_to_rna)``, each
            shape ``(batch, hidden_dim)``.
        """
        rna_embedding = self.rna_tower(rna_ids, rna_values)
        atac_embedding = self.atac_tower(atac_ids, atac_values)
        atac_aligned = self.atac_to_rna_align(atac_embedding)
        return rna_embedding, atac_embedding, atac_aligned


def build_epifoundation() -> nn.Module:
    """Build a compact EpiFoundation dual-stream RNA/ATAC alignment model.

    Returns
    -------
    nn.Module
        EpiFoundation reconstruction in evaluation mode.
    """
    return EpiFoundation(num_rnas=256, num_atacs=512, hidden_dim=48, depth=2, heads=4).eval()


def example_input_epifoundation() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create example input for :func:`build_epifoundation`.

    Returns
    -------
    tuple of torch.Tensor
        ``(rna_ids, rna_values, atac_ids, atac_values)``: dense RNA tokens
        shaped ``(2, 24)`` (``[CLS]`` id 256 at position 0) and sparse ATAC
        tokens shaped ``(2, 16)`` (``[CLS]`` id 512 at position 0), each
        paired with a binned-value tensor of matching shape.
    """
    num_rnas, num_atacs = 256, 512
    rna_body = torch.randint(0, num_rnas, (2, 23))
    rna_cls = torch.full((2, 1), num_rnas, dtype=torch.long)
    rna_ids = torch.cat([rna_cls, rna_body], dim=1)
    rna_values = torch.randint(0, 32, (2, 24))

    atac_body = torch.randint(0, num_atacs, (2, 15))
    atac_cls = torch.full((2, 1), num_atacs, dtype=torch.long)
    atac_ids = torch.cat([atac_cls, atac_body], dim=1)
    atac_values = torch.randint(0, 32, (2, 16))

    return rna_ids, rna_values, atac_ids, atac_values


MENAGERIE_ENTRIES = [
    ("DigNet", "build_dignet", "example_input_dignet", "2024", "BIO"),
    ("DNABERT", "build_dnabert", "example_input_dnabert", "2021", "BIO"),
    ("Enigma", "build_enigma", "example_input_enigma", "2025", "BIO"),
    ("EPCOT", "build_epcot", "example_input_epcot", "2023", "BIO"),
    ("EpiAgent", "build_epiagent", "example_input_epiagent", "2025", "BIO"),
    ("EpiFoundation", "build_epifoundation", "example_input_epifoundation", "2025", "BIO"),
]
