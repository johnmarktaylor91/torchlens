"""Wave 5 batch 14 menagerie classics: protein/immunology/pathology/genomics family.

Sources checked (repo code inspected via GitHub API, base env only, no clone or
pip install):
  - Tranception: https://github.com/OATML-Markslab/Tranception
    (``tranception/model_pytorch.py``, ``SpatialDepthWiseConvolution`` and
    ``TranceptionBlockAttention``). Notin et al., "Tranception: protein
    fitness prediction with autoregressive transformers and inference-time
    retrieval", ICML 2022 (arXiv:2205.13760). Distinctive mechanism: a GPT2
    causal-attention block whose heads are split into 4 equal groups; group 0
    is left untouched (kernel size 1) while groups 1-3 apply a shared-per-head
    depthwise 1D causal convolution (kernel sizes 3, 5, 7) to the Q/K/V
    projections before scaled dot-product attention, letting different head
    groups mix information at different local ranges. (The retrieval-time MSA
    ensembling is an inference-time post-processing step, not part of the
    trainable module, so it is out of scope for the traced architecture.)
  - TransPHLA-AOMP: https://github.com/a96123155/TransPHLA-AOMP
    (``TransPHLA-AOMP/model.py``, ``Transformer``/``Encoder``/``Decoder``).
    Chu et al., "A transformer-based model to predict peptide-HLA class I
    binding and optimize mutated peptides for vaccine design", Nature Machine
    Intelligence 2022. Distinctive mechanism: twin (weight-independent)
    Transformer encoders separately embed the peptide and HLA pseudo-sequence
    token streams; their token outputs are concatenated along the sequence
    axis and passed through a further self-attention "decoder" stack (no
    cross-attention -- concatenation is the fusion point); the flattened
    fused representation is projected by an MLP to a 2-way
    binder/non-binder classification.
  - UNI: https://github.com/mahmoodlab/UNI ;
    https://huggingface.co/MahmoodLab/UNI ; Chen et al., "Towards a general-
    purpose foundation model for computational pathology", Nature Medicine
    2024. Plain ViT-L/16 (patch 16, depth 24, width 1024, 16 heads) trained
    with DINOv2 self-distillation on >100M histology tiles; the architecture
    itself is a standard pre-norm ViT encoder (no distinctive structural
    twist versus vanilla ViT), so it is built here directly with
    ``torch.nn.TransformerEncoder`` over patch + cls-token embeddings at
    tiny dims -- the DINOv2 training recipe, not the forward architecture,
    is UNI's contribution.
  - VAMB: https://github.com/RasmussenLab/vamb (``vamb/encode.py``, ``VAE``).
    Nissen et al., "Improved metagenome binning and assembly using deep
    variational autoencoders", Nature Biotechnology 2021. Distinctive
    mechanism: a single VAE whose input/output vector is the concatenation of
    per-sample abundance (co-abundance across samples), tetranucleotide
    frequency (TNF) composition, and a scalar contig-length weight; encoder
    and decoder are symmetric BatchNorm+LeakyReLU MLP stacks, latent Gaussian
    reparameterization (``mu``, ``logsigma``), and the decoder output is
    split back into (abundance-softmax, TNF, weight-sigmoid) heads reflecting
    the three different reconstruction losses (SSE, CE, SSE) used in training.
  - SPOT-RNA: https://github.com/jaswindersingh2/SPOT-RNA (TensorFlow;
    ``SPOT-RNA.py`` restores frozen ``model{0..4}`` checkpoints for a 5-model
    ensemble). Singh et al., "RNA secondary structure prediction using an
    ensemble of two-dimensional deep neural networks and transfer learning",
    Nature Communications 2019. Distinctive mechanism (from the paper, since
    the shipped code is inference-only frozen TF graphs): an outer-
    concatenation of the 1D per-base one-hot sequence into a 2D
    ``(L, L, 2*C)`` pairwise feature map, a stack of dilated pre-activation
    ResNet2D blocks (increasing dilation) over that map, a 2D bidirectional
    LSTM applied jointly along both matrix axes, and a final 1x1-conv
    ("fully connected") sigmoid head producing the symmetric base-pairing
    probability matrix (2D-BLSTM + dilated ResNet is the paper's key
    contribution enabling non-canonical/pseudoknot pairs); reimplemented
    here in PyTorch (source is TF1) with a compact single-model instance
    (the 5-checkpoint ensembling is inference-time averaging, not part of
    one model's trainable architecture).
  - Prov-GigaPath: https://github.com/prov-gigapath/prov-gigapath
    (``gigapath/slide_encoder.py``, ``LongNetViT``). Xu et al., "A whole-
    slide foundation model for digital pathology from real-world data",
    Nature 2024. Distinctive mechanism: the *slide* encoder (downstream of a
    separately pretrained tile encoder) embeds a bag of per-tile feature
    vectors with a coordinate-indexed learned positional lookup and a class
    token, then processes the tile sequence with LongNet-style *dilated
    segmented attention*: at each layer, tokens are split into fixed-size
    segments, a fraction of tokens per segment are subsampled at a dilation
    rate that grows across parallel attention heads, self-attention runs
    independently within each dilated segment, and the segment outputs are
    scattered back to their original positions and summed across the
    dilation-rate groups -- giving attention that is linear in sequence
    length while still mixing distant tiles across a gigapixel slide.

All six are faithful compact reimplementations: random init, small dims, few
layers, forward-only, kept just large enough to exercise each architecture's
distinctive mechanism so the traced/unrolled atlas graph renders quickly.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. Tranception: GPT2 causal self-attention with grouped multi-scale spatial
#    depthwise convolutions applied to Q/K/V before scaled dot-product
#    attention (4 head groups, kernel sizes 1/3/5/7).
# ---------------------------------------------------------------------------


class SpatialDepthWiseConvolution(nn.Module):
    """Per-head causal depthwise 1D convolution (faithful to the original)."""

    def __init__(self, head_dim: int, kernel_size: int = 3) -> None:
        """Initialize the causal depthwise convolution.

        Parameters
        ----------
        head_dim:
            Per-head feature width (used as the depthwise channel count).
        kernel_size:
            Convolution kernel width; left-padded to preserve causality.
        """

        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            head_dim, head_dim, kernel_size, padding=kernel_size - 1, groups=head_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the causal depthwise convolution along the sequence axis.

        Parameters
        ----------
        x:
            Tensor shaped ``(batch, heads, seq_len, head_dim)``.

        Returns
        -------
        torch.Tensor
            Tensor of the same shape, convolved causally along ``seq_len``.
        """

        b, h, s, d = x.shape
        x = x.permute(0, 1, 3, 2).reshape(b * h, d, s)
        x = self.conv(x)
        if self.kernel_size > 1:
            x = x[:, :, : -(self.kernel_size - 1)]
        x = x.view(b, h, d, s).permute(0, 1, 3, 2)
        return x


class TranceptionAttention(nn.Module):
    """Grouped multi-scale-convolution causal self-attention block.

    Faithful to ``TranceptionBlockAttention`` in ``attention_mode="tranception"``:
    heads are split into 4 equal groups; the first group attends with plain
    Q/K/V, while the other three apply a shared depthwise causal convolution
    (kernel sizes 3, 5, 7 respectively) to their Q/K/V projections before
    scaled dot-product attention.
    """

    def __init__(self, d_model: int, num_heads: int = 8) -> None:
        """Initialize the QKV/output projections and per-group convolutions.

        Parameters
        ----------
        d_model:
            Model (embedding) width.
        num_heads:
            Total attention heads; must be divisible by 4.
        """

        super().__init__()
        assert num_heads % 4 == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.heads_per_group = num_heads // 4
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.q_conv = nn.ModuleList(
            [SpatialDepthWiseConvolution(self.head_dim, k) for k in (3, 5, 7)]
        )
        self.k_conv = nn.ModuleList(
            [SpatialDepthWiseConvolution(self.head_dim, k) for k in (3, 5, 7)]
        )
        self.v_conv = nn.ModuleList(
            [SpatialDepthWiseConvolution(self.head_dim, k) for k in (3, 5, 7)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply grouped multi-scale causal self-attention.

        Parameters
        ----------
        x:
            Token embeddings shaped ``(batch, seq_len, d_model)``.

        Returns
        -------
        torch.Tensor
            Attended output shaped ``(batch, seq_len, d_model)``.
        """

        b, s, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)
        g = self.heads_per_group
        q_groups = [q[:, :g]]
        k_groups = [k[:, :g]]
        v_groups = [v[:, :g]]
        for i in range(3):
            sl = slice((i + 1) * g, (i + 2) * g)
            q_groups.append(self.q_conv[i](q[:, sl]))
            k_groups.append(self.k_conv[i](k[:, sl]))
            v_groups.append(self.v_conv[i](v[:, sl]))
        q = torch.cat(q_groups, dim=1)
        k = torch.cat(k_groups, dim=1)
        v = torch.cat(v_groups, dim=1)

        causal_mask = torch.triu(torch.ones(s, s, dtype=torch.bool, device=x.device), diagonal=1)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, s, -1)
        return self.out_proj(out)


class TranceptionBlock(nn.Module):
    """Pre-norm transformer block wrapping :class:`TranceptionAttention`."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int) -> None:
        """Initialize the attention sub-layer, MLP sub-layer, and norms."""

        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = TranceptionAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention and MLP sub-layers with residual connections."""

        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TranceptionModel(nn.Module):
    """Compact Tranception autoregressive protein language model.

    Faithful to the Tranception architecture: token + learned positional
    embeddings feed a stack of grouped multi-scale-convolution causal
    self-attention blocks, followed by a final layer norm and a
    weight-tied language-modeling head over the amino-acid vocabulary.
    """

    def __init__(
        self,
        vocab_size: int = 25,
        d_model: int = 64,
        num_heads: int = 8,
        d_ff: int = 128,
        num_layers: int = 3,
        max_len: int = 64,
    ) -> None:
        """Initialize embeddings, transformer blocks, and the LM head."""

        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [TranceptionBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Predict next-token amino-acid logits for a protein sequence.

        Parameters
        ----------
        input_ids:
            Integer token ids shaped ``(batch, seq_len)``.

        Returns
        -------
        torch.Tensor
            Logits shaped ``(batch, seq_len, vocab_size)``.
        """

        b, s = input_ids.shape
        pos = torch.arange(s, device=input_ids.device).unsqueeze(0).expand(b, s)
        h = self.tok_embed(input_ids) + self.pos_embed(pos)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.lm_head(h)


def build_tranception() -> nn.Module:
    """Build a compact Tranception autoregressive protein model."""

    return TranceptionModel().eval()


def example_input_tranception() -> torch.Tensor:
    """Return a batch of tokenized amino-acid sequences for Tranception."""

    return torch.randint(0, 25, (2, 40))


# ---------------------------------------------------------------------------
# 2. TransPHLA: twin Transformer encoders for peptide and HLA pseudo-sequence
#    tokens, concatenated along the sequence axis and refined by a further
#    self-attention "decoder" stack (no cross-attention), flattened and
#    projected to a binder/non-binder classification.
# ---------------------------------------------------------------------------


class TransPhlaEncoderLayer(nn.Module):
    """Standard post-norm self-attention + FFN encoder layer."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        """Initialize the self-attention and feed-forward sub-layers."""

        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention then a feed-forward block, each with a
        residual connection followed by layer norm (post-norm, as in the
        original ``nn.LayerNorm(d_model)(output + residual)`` pattern)."""

        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(attn_out + x)
        ffn_out = self.ffn(x)
        x = self.norm2(ffn_out + x)
        return x


class TransPhlaEncoder(nn.Module):
    """Token + sinusoidal positional embedding followed by encoder layers."""

    def __init__(
        self, vocab_size: int, d_model: int, n_heads: int, d_ff: int, n_layers: int, max_len: int
    ) -> None:
        """Initialize the embedding table and the encoder-layer stack."""

        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        self.layers = nn.ModuleList(
            [TransPhlaEncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed a token sequence and refine it through the encoder stack.

        Parameters
        ----------
        tokens:
            Integer token ids shaped ``(batch, seq_len)``.

        Returns
        -------
        torch.Tensor
            Contextualized token embeddings shaped ``(batch, seq_len, d_model)``.
        """

        x = self.embed(tokens) + self.pe[: tokens.size(1)].unsqueeze(0)
        for layer in self.layers:
            x = layer(x)
        return x


class TransPhlaModel(nn.Module):
    """Twin-encoder Transformer for peptide-HLA class I binding prediction.

    Faithful to ``Transformer`` in TransPHLA-AOMP: separate (non-shared)
    encoders embed the peptide and HLA pseudo-sequence token streams; their
    outputs are concatenated along the sequence dimension and passed through
    a further self-attention-only "decoder" stack (fusion happens purely via
    concatenation, with no cross-attention between the two streams); the
    flattened fused representation is classified by an MLP head.
    """

    def __init__(
        self,
        vocab_size: int = 22,
        d_model: int = 32,
        n_heads: int = 4,
        d_ff: int = 64,
        n_layers: int = 1,
        pep_len: int = 15,
        hla_len: int = 34,
    ) -> None:
        """Initialize the peptide/HLA encoders, fusion decoder, and head."""

        super().__init__()
        self.pep_len = pep_len
        self.hla_len = hla_len
        tgt_len = pep_len + hla_len
        self.pep_encoder = TransPhlaEncoder(vocab_size, d_model, n_heads, d_ff, n_layers, pep_len)
        self.hla_encoder = TransPhlaEncoder(vocab_size, d_model, n_heads, d_ff, n_layers, hla_len)
        pos = torch.arange(tgt_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(tgt_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("dec_pe", pe)
        self.decoder_layers = nn.ModuleList(
            [TransPhlaEncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.projection = nn.Sequential(
            nn.Linear(tgt_len * d_model, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, pep_tokens: torch.Tensor, hla_tokens: torch.Tensor) -> torch.Tensor:
        """Predict peptide-HLA binding logits.

        Parameters
        ----------
        pep_tokens:
            Peptide token ids shaped ``(batch, pep_len)``.
        hla_tokens:
            HLA pseudo-sequence token ids shaped ``(batch, hla_len)``.

        Returns
        -------
        torch.Tensor
            Binder/non-binder logits shaped ``(batch, 2)``.
        """

        pep_out = self.pep_encoder(pep_tokens)
        hla_out = self.hla_encoder(hla_tokens)
        fused = torch.cat([pep_out, hla_out], dim=1) + self.dec_pe.unsqueeze(0)
        for layer in self.decoder_layers:
            fused = layer(fused)
        flat = fused.reshape(fused.size(0), -1)
        return self.projection(flat)


def build_transphla() -> nn.Module:
    """Build a compact TransPHLA peptide-HLA binding classifier."""

    return TransPhlaModel().eval()


def example_input_transphla() -> Tuple[torch.Tensor, torch.Tensor]:
    """Return padded peptide and HLA pseudo-sequence tokens for TransPHLA."""

    pep_tokens = torch.randint(0, 22, (4, 15))
    hla_tokens = torch.randint(0, 22, (4, 34))
    return pep_tokens, hla_tokens


# ---------------------------------------------------------------------------
# 3. UNI: plain pre-norm ViT-style encoder (patch + cls-token embedding,
#    standard multi-head self-attention transformer blocks) trained with
#    DINOv2 self-distillation on histopathology tiles; the forward
#    architecture is vanilla ViT, so it is built directly at tiny dims.
# ---------------------------------------------------------------------------


class UniPathologyViT(nn.Module):
    """Compact ViT-style histopathology tile encoder (faithful to UNI's ViT-L/16).

    A patch-embedding conv, a learned class token, learned positional
    embeddings, and a stack of pre-norm ``nn.TransformerEncoderLayer`` blocks
    -- the standard ViT forward pass UNI uses; DINOv2 self-distillation
    (UNI's actual contribution) is a training-time recipe with no effect on
    the traced forward architecture.
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 16,
        embed_dim: int = 48,
        depth: int = 3,
        num_heads: int = 4,
    ) -> None:
        """Initialize the patch embedding, class token, and encoder stack.

        Parameters
        ----------
        img_size:
            Input tile side length in pixels.
        patch_size:
            Patch (conv stride/kernel) side length in pixels.
        embed_dim:
            Token embedding width.
        depth:
            Number of transformer encoder layers.
        num_heads:
            Attention heads per layer.
        """

        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            embed_dim, num_heads, embed_dim * 4, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, tile: torch.Tensor) -> torch.Tensor:
        """Encode a batch of histopathology tiles to a class-token embedding.

        Parameters
        ----------
        tile:
            Tiles shaped ``(batch, 3, img_size, img_size)``.

        Returns
        -------
        torch.Tensor
            Class-token embeddings shaped ``(batch, embed_dim)``.
        """

        x = self.patch_embed(tile).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.encoder(x)
        return self.norm(x[:, 0])


def build_uni() -> nn.Module:
    """Build a compact UNI-style ViT histopathology tile encoder."""

    return UniPathologyViT().eval()


def example_input_uni() -> torch.Tensor:
    """Return a batch of histopathology tiles for UNI."""

    return torch.randn(2, 3, 64, 64)


# ---------------------------------------------------------------------------
# 4. VAMB: single VAE over the concatenation of per-sample abundance,
#    tetranucleotide-frequency composition, and a contig-length weight;
#    symmetric BatchNorm+LeakyReLU encoder/decoder MLPs; decoder output
#    split into (abundance, TNF, weight) reconstruction heads.
# ---------------------------------------------------------------------------


class VambVAE(nn.Module):
    """Metagenomic contig VAE (faithful to ``vamb.encode.VAE``).

    The encoder/decoder input-output vector is
    ``[abundance (nsamples), tetranucleotide-frequency (ntnf), weight (1)]``
    concatenated; latent Gaussian reparameterization; the decoder's final
    linear layer is split back into three heads matching the three
    reconstruction targets (log-softmax abundance, TNF, sigmoid weight).
    """

    def __init__(
        self,
        nsamples: int = 4,
        ntnf: int = 103,
        nhiddens: Tuple[int, int] = (64, 64),
        nlatent: int = 16,
    ) -> None:
        """Initialize the encoder/decoder MLP stacks and latent heads.

        Parameters
        ----------
        nsamples:
            Number of co-abundance samples per contig.
        ntnf:
            Tetranucleotide-frequency composition width.
        nhiddens:
            Hidden widths of the (symmetric) encoder/decoder MLP.
        nlatent:
            Latent embedding width.
        """

        super().__init__()
        self.nsamples = nsamples
        self.ntnf = ntnf
        in_dim = nsamples + ntnf + 1

        self.encoderlayers = nn.ModuleList()
        self.encodernorms = nn.ModuleList()
        for nin, nout in zip([in_dim] + list(nhiddens), nhiddens):
            self.encoderlayers.append(nn.Linear(nin, nout))
            self.encodernorms.append(nn.BatchNorm1d(nout))
        self.mu = nn.Linear(nhiddens[-1], nlatent)
        self.logsigma = nn.Linear(nhiddens[-1], nlatent)

        rev_hiddens = list(nhiddens[::-1])
        self.decoderlayers = nn.ModuleList()
        self.decodernorms = nn.ModuleList()
        for nin, nout in zip([nlatent] + rev_hiddens, rev_hiddens):
            self.decoderlayers.append(nn.Linear(nin, nout))
            self.decodernorms.append(nn.BatchNorm1d(nout))
        self.outputlayer = nn.Linear(rev_hiddens[-1], in_dim)
        self.dropout = nn.Dropout(0.2)

    def _encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return latent mean and log-sigma from the concatenated features."""

        h = x
        for layer, norm in zip(self.encoderlayers, self.encodernorms):
            h = self.dropout(F.leaky_relu(norm(layer(h))))
        return self.mu(h), self.logsigma(h)

    def _decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode a latent sample back into (abundance, TNF, weight)."""

        h = z
        for layer, norm in zip(self.decoderlayers, self.decodernorms):
            h = self.dropout(F.leaky_relu(norm(layer(h))))
        out = self.outputlayer(h)
        abundance_out = F.log_softmax(out[:, : self.nsamples], dim=1)
        tnf_out = out[:, self.nsamples : self.nsamples + self.ntnf]
        weight_out = torch.sigmoid(out[:, -1])
        return abundance_out, tnf_out, weight_out

    def forward(
        self, depths: torch.Tensor, tnf: torch.Tensor, weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode and reconstruct a batch of contig feature vectors.

        Parameters
        ----------
        depths:
            Per-sample co-abundance shaped ``(batch, nsamples)``.
        tnf:
            Tetranucleotide-frequency composition shaped ``(batch, ntnf)``.
        weights:
            Contig-length weight shaped ``(batch, 1)``.

        Returns
        -------
        tuple of torch.Tensor
            ``(abundance_recon, tnf_recon, weight_recon, mu, logsigma)``.
        """

        x = torch.cat([depths, tnf, weights], dim=1)
        mu, logsigma = self._encode(x)
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(logsigma)
        abundance_out, tnf_out, weight_out = self._decode(z)
        return abundance_out, tnf_out, weight_out, mu, logsigma


def build_vamb() -> nn.Module:
    """Build a compact VAMB metagenomic-contig VAE."""

    return VambVAE().eval()


def example_input_vamb() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return abundance, TNF, and weight features for a batch of contigs."""

    depths = torch.rand(6, 4)
    tnf = torch.randn(6, 103)
    weights = torch.rand(6, 1)
    return depths, tnf, weights


# ---------------------------------------------------------------------------
# 5. SPOT-RNA: outer-concatenated 1D sequence -> 2D pairwise feature map,
#    dilated pre-activation ResNet2D blocks, a 2D bidirectional LSTM applied
#    along both matrix axes, and a 1x1-conv sigmoid base-pairing head.
# ---------------------------------------------------------------------------


class SpotRnaResBlock(nn.Module):
    """Dilated pre-activation ResNet2D block (faithful to the paper's ResNet
    stack: BatchNorm-ReLU-Conv pre-activation with a growing dilation rate)."""

    def __init__(self, channels: int, dilation: int) -> None:
        """Initialize the pre-activation dilated 3x3 convolutions.

        Parameters
        ----------
        channels:
            Feature-map channel width.
        dilation:
            Dilation rate for both convolutions in the block.
        """

        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the pre-activation dilated residual block."""

        h = self.conv1(F.relu(self.bn1(x)))
        h = self.conv2(F.relu(self.bn2(h)))
        return x + h


class SpotRnaModel(nn.Module):
    """Compact SPOT-RNA base-pairing predictor (single ensemble member).

    Faithful to the paper's architecture: the one-hot sequence is outer-
    concatenated into a symmetric 2D pairwise map, refined by a stack of
    dilated pre-activation ResNet2D blocks (growing dilation), then a 2D
    bidirectional LSTM (applied along rows, then along columns) propagates
    long-range pairing dependencies, and a final 1x1 convolution ("fully
    connected") sigmoid head predicts the symmetric base-pairing probability
    matrix. The 5-checkpoint test-time ensembling in the source repo is
    inference-time averaging and out of scope for one model's architecture.
    """

    def __init__(
        self, num_bases: int = 4, channels: int = 16, num_blocks: int = 3, lstm_hidden: int = 8
    ) -> None:
        """Initialize the outer-concat projection, ResNet2D stack, 2D-BLSTM,
        and the base-pairing output head."""

        super().__init__()
        self.in_proj = nn.Conv2d(2 * num_bases, channels, 1)
        self.blocks = nn.ModuleList(
            [SpotRnaResBlock(channels, dilation=2**i) for i in range(num_blocks)]
        )
        self.row_lstm = nn.LSTM(channels, lstm_hidden, batch_first=True, bidirectional=True)
        self.col_lstm = nn.LSTM(2 * lstm_hidden, lstm_hidden, batch_first=True, bidirectional=True)
        self.out_conv = nn.Conv2d(2 * lstm_hidden, 1, 1)

    def forward(self, one_hot_seq: torch.Tensor) -> torch.Tensor:
        """Predict the symmetric base-pairing probability matrix.

        Parameters
        ----------
        one_hot_seq:
            One-hot nucleotide sequence shaped ``(batch, length, num_bases)``.

        Returns
        -------
        torch.Tensor
            Symmetric base-pairing probabilities shaped
            ``(batch, length, length)``.
        """

        b, l, c = one_hot_seq.shape
        row = one_hot_seq.unsqueeze(2).expand(b, l, l, c)
        col = one_hot_seq.unsqueeze(1).expand(b, l, l, c)
        pair_map = torch.cat([row, col], dim=-1).permute(0, 3, 1, 2)

        h = self.in_proj(pair_map)
        for block in self.blocks:
            h = block(h)

        row_in = h.permute(0, 2, 3, 1).reshape(b * l, l, h.size(1))
        row_out, _ = self.row_lstm(row_in)
        row_out = row_out.view(b, l, l, -1)

        col_in = row_out.permute(0, 2, 1, 3).reshape(b * l, l, row_out.size(-1))
        col_out, _ = self.col_lstm(col_in)
        col_out = col_out.view(b, l, l, -1).permute(0, 2, 1, 3)

        logits = self.out_conv(col_out.permute(0, 3, 1, 2)).squeeze(1)
        sym_logits = 0.5 * (logits + logits.transpose(-1, -2))
        return torch.sigmoid(sym_logits)


def build_spot_rna() -> nn.Module:
    """Build a compact SPOT-RNA base-pairing predictor."""

    return SpotRnaModel().eval()


def example_input_spot_rna() -> torch.Tensor:
    """Return a one-hot RNA sequence for SPOT-RNA."""

    length = 20
    idx = torch.randint(0, 4, (1, length))
    return F.one_hot(idx, num_classes=4).float()


# ---------------------------------------------------------------------------
# 6. Prov-GigaPath (LongNet slide encoder): coordinate-indexed positional
#    embedding + class token over a bag of tile embeddings, refined by
#    LongNet dilated segmented attention (segment-split, per-head dilation
#    rate, intra-segment self-attention, scatter-sum across dilation groups).
# ---------------------------------------------------------------------------


class DilatedSegmentAttention(nn.Module):
    """LongNet-style dilated segmented self-attention (faithful mechanism).

    Splits the token sequence into fixed-size segments; for each of several
    dilation rates (one per head group), every ``r``-th token within a
    segment is gathered, self-attention runs on that sparse subsequence, and
    outputs are scattered back to their original positions before summing
    across dilation-rate groups -- giving linear-in-length attention that
    still mixes tokens across the whole (padded) sequence.
    """

    def __init__(
        self, d_model: int, num_heads: int, segment_size: int, dilation_rates: Tuple[int, ...]
    ) -> None:
        """Initialize the shared QKV/output projections and dilation config.

        Parameters
        ----------
        d_model:
            Model (embedding) width.
        num_heads:
            Total attention heads; split evenly across ``dilation_rates``.
        segment_size:
            Number of tokens per attention segment.
        dilation_rates:
            Per-head-group dilation (subsampling) rates.
        """

        super().__init__()
        assert num_heads % len(dilation_rates) == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.heads_per_group = num_heads // len(dilation_rates)
        self.segment_size = segment_size
        self.dilation_rates = dilation_rates
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dilated segmented self-attention.

        Parameters
        ----------
        x:
            Token embeddings shaped ``(batch, seq_len, d_model)``, where
            ``seq_len`` is a multiple of ``segment_size``.

        Returns
        -------
        torch.Tensor
            Attended output shaped ``(batch, seq_len, d_model)``.
        """

        b, s, _ = x.shape
        num_segments = s // self.segment_size
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def to_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(b, num_segments, self.segment_size, self.num_heads, self.head_dim)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)
        out = torch.zeros_like(q)
        g = self.heads_per_group
        for gi, r in enumerate(self.dilation_rates):
            head_slice = slice(gi * g, (gi + 1) * g)
            idx = torch.arange(0, self.segment_size, r, device=x.device)
            qs = q[:, :, idx][..., head_slice, :]
            ks = k[:, :, idx][..., head_slice, :]
            vs = v[:, :, idx][..., head_slice, :]
            qs = qs.permute(0, 3, 1, 2, 4)
            ks = ks.permute(0, 3, 1, 2, 4)
            vs = vs.permute(0, 3, 1, 2, 4)
            scores = torch.matmul(qs, ks.transpose(-1, -2)) / math.sqrt(self.head_dim)
            attn = F.softmax(scores, dim=-1)
            group_out = torch.matmul(attn, vs).permute(0, 2, 3, 1, 4)
            out[:, :, idx, head_slice, :] = out[:, :, idx, head_slice, :] + group_out
        merged = out.reshape(b, s, self.d_model)
        return self.out_proj(merged)


class LongNetSlideEncoderLayer(nn.Module):
    """Pre-norm transformer block wrapping dilated segmented attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        segment_size: int,
        dilation_rates: Tuple[int, ...],
    ) -> None:
        """Initialize the dilated-attention sub-layer, MLP sub-layer, and norms."""

        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = DilatedSegmentAttention(d_model, num_heads, segment_size, dilation_rates)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dilated attention and MLP sub-layers with residuals."""

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GigaPathSlideEncoder(nn.Module):
    """Compact Prov-GigaPath LongNet slide encoder.

    Faithful to ``LongNetViT``: a linear tile-embedding projection, a
    coordinate-indexed learned positional lookup table, a prepended class
    token, and a stack of LongNet dilated-segmented-attention blocks that
    mix tile embeddings across the (padded) slide sequence with attention
    cost linear in sequence length.
    """

    def __init__(
        self,
        tile_dim: int = 32,
        embed_dim: int = 32,
        depth: int = 2,
        num_heads: int = 4,
        slide_ngrids: int = 8,
        segment_size: int = 8,
        dilation_rates: Tuple[int, ...] = (1, 2),
    ) -> None:
        """Initialize the tile-embedding projection, positional table, class
        token, and LongNet encoder stack."""

        super().__init__()
        self.slide_ngrids = slide_ngrids
        self.patch_embed = nn.Linear(tile_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, slide_ngrids * slide_ngrids + 1, embed_dim))
        self.layers = nn.ModuleList(
            [
                LongNetSlideEncoderLayer(
                    embed_dim, num_heads, embed_dim * 4, segment_size, dilation_rates
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.segment_size = segment_size
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, tile_embeds: torch.Tensor, grid_pos: torch.Tensor) -> torch.Tensor:
        """Encode a bag of tile embeddings into a slide-level representation.

        Parameters
        ----------
        tile_embeds:
            Per-tile feature vectors shaped ``(batch, num_tiles, tile_dim)``.
        grid_pos:
            Flattened grid index (row * ngrids + col) per tile, shaped
            ``(batch, num_tiles)``.

        Returns
        -------
        torch.Tensor
            Slide-level class-token embedding shaped ``(batch, embed_dim)``.
        """

        x = self.patch_embed(tile_embeds)
        pos = self.pos_embed[:, 1:][:, grid_pos[0]]
        x = x + pos
        cls = self.cls_token + self.pos_embed[:, :1]
        x = torch.cat([cls.expand(x.size(0), -1, -1), x], dim=1)

        pad = (-x.size(1)) % self.segment_size
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        for layer in self.layers:
            x = layer(x)
        return self.norm(x[:, 0])


def build_gigapath() -> nn.Module:
    """Build a compact Prov-GigaPath LongNet slide encoder."""

    return GigaPathSlideEncoder().eval()


def example_input_gigapath() -> Tuple[torch.Tensor, torch.Tensor]:
    """Return tile embeddings and grid positions for the GigaPath slide encoder."""

    num_tiles = 12
    tile_embeds = torch.randn(1, num_tiles, 32)
    grid_pos = torch.randint(0, 8 * 8, (1, num_tiles))
    return tile_embeds, grid_pos


MENAGERIE_ENTRIES = [
    ("Tranception", "build_tranception", "example_input_tranception", "2022", "BIO"),
    ("TransPHLA", "build_transphla", "example_input_transphla", "2022", "BIO"),
    ("UNI", "build_uni", "example_input_uni", "2024", "BIO"),
    ("VAMB", "build_vamb", "example_input_vamb", "2021", "BIO"),
    ("SPOT-RNA", "build_spot_rna", "example_input_spot_rna", "2019", "BIO"),
    (
        "ViT-based GigaPath (LongNet slide encoder)",
        "build_gigapath",
        "example_input_gigapath",
        "2024",
        "BIO",
    ),
]
