"""Menagerie batch w5a16: computational-biology deep-learning classics for
CRISPR efficiency prediction, cross-modal single-cell translation, genomic
sequence-to-track regression, chromatin-feature prediction, 3D-genome
contact-map prediction, and semi-supervised single-cell embedding.

Sources checked (reference only; no cloning, no pip installs):
  - AttnToCrispr (cand_00648): Zhang, Zhang, Wu, et al. (as used by
    qiaoliuhub), official repo https://github.com/qiaoliuhub/AttnToCrispr
    (``attention_model.py``, ``Layers.py``, ``Sublayers.py``), paper
    https://doi.org/10.1016/j.csbj.2021.03.001. The reference model is an
    encoder-decoder Transformer (``Transformer`` in ``attention_model.py``):
    a source (guide RNA one-hot) sequence and target (target-site one-hot)
    sequence are each embedded (token + learned positional embedding) and
    passed through a standard multi-head-attention Encoder / Decoder stack
    (``EncoderLayer``/``DecoderLayer`` in ``Layers.py``: self-attention,
    cross-attention for the decoder, then a feed-forward block, each with a
    residual + optional LayerNorm), after which the decoder output is
    reshaped to an image-like tensor and passed through a small 2D CNN
    (``customized_CNN``: two Conv2d+MaxPool2d+ReLU stages) before a final
    feed-forward regression head (``OutputFeedForward``) predicts one
    CRISPR/Cas9 editing-efficiency score. Reimplemented compactly and
    faithfully: token+positional embeddings for guide and target one-hot
    sequences, a small Transformer encoder-decoder, the 2D-CNN branch over
    the decoder output, and an MLP regression head -- matching the
    "2D-CNN-over-attention-decoded guide-target alignment" mechanism the
    paper is named for.
  - BABEL (cand_00649): Wu, Kunder, et al. (wukevin), official repo
    https://github.com/wukevin/babel (``babel/models/autoencoders.py``,
    class ``SplicedAutoEncoder``), paper
    https://www.pnas.org/content/118/15/e2023070118. BABEL's namesake
    mechanism is the *spliced* autoencoder: two independent modality-specific
    encoders (scRNA-seq gene-expression encoder, scATAC-seq chromatin
    accessibility encoder) are trained so that both map into a **shared**
    latent space, and two independent decoders (RNA decoder, ATAC decoder)
    can each decode from *either* encoder's latent code -- giving four
    reconstruction paths (RNA->RNA, RNA->ATAC, ATAC->RNA, ATAC->ATAC), which
    is exactly the cross-modal "translation" the BABEL name refers to.
    Reimplemented with the same two-encoder / shared-latent / two-decoder
    topology as ``SplicedAutoEncoder.forward``, using compact
    Linear-BatchNorm-PReLU MLP encoders/decoders in place of the reference's
    ``Encoder``/``Decoder`` (which use the identical topology), returning all
    four cross-modal reconstructions.
  - Basenji1 (cand_00650): Kelley, Reshef, Bileschi, Belanger, McLean &
    Snoek, Genome Research 2018, https://genome.cshlp.org/content/28/5/739,
    official repo https://github.com/calico/basenji (original params at
    ``manuscripts/genome_research2018/params.txt``, model code
    ``basenji/blocks.py``/``basenji/seqnn.py``). The 2018 architecture
    (distinct from the already-catalogued Basenji2/Enformer-style attention
    tower) is a pure dilated-CNN "conv tower + dilated dense-residual tower"
    over one-hot DNA: an initial wide-kernel conv, several
    conv-batchnorm-ReLU-maxpool stages progressively downsampling and
    widening the sequence (mirroring the ``cnn_pool`` stages in
    ``params.txt``), followed by a stack of *dilated* Conv1d blocks with
    exponentially increasing dilation (matching ``cnn_dilation`` doubling
    1,2,4,...,64) whose outputs are concatenated onto a running "dense"
    feature stream (``cnn_dense=1`` in the reference params, i.e. a
    DenseNet-style skip-concatenation rather than additive ResNet skip) --
    the defining "dilated residual/dense tower for a very long receptive
    field over the input DNA sequence" mechanism -- ending in a 1x1 conv
    Poisson-regression head over many genomic tracks per bin. Reimplemented
    compactly with the same downsample-then-dilate-then-densely-concatenate
    topology and a small multi-track linear-exponential (``exp``) Poisson
    read-out head, at greatly reduced channel counts/sequence length for
    catalog size.
  - Beluga (cand_00653): Zhou (DeepSEA-lineage model shipped as part of
    ExPecto), Zhou et al. Nature Genetics 2018,
    https://www.nature.com/articles/s41588-018-0160-6, official repo
    https://github.com/FunctionLab/ExPecto (``chromatin.py``, class
    ``Beluga``). Ported near-verbatim from the reference (also available as
    the Kipoi ``DeepSEA/beluga`` model): a 2D-CNN over one-hot DNA encoded as
    a ``(4, 1, L)`` "image" (4 nucleotide channels, sequence along one
    spatial axis, height fixed at 1), three widening conv stages (320 -> 480
    -> 640 channels) each with two ``Conv2d(kernel=(1,8))`` + ReLU layers,
    dropout, and max pooling ``(1,4)`` after the first two stages, followed
    by flatten -> Dropout -> a large FC "2003-unit" hidden layer -> ReLU ->
    a final FC layer to the (compacted) chromatin-feature count, and a
    sigmoid output over per-feature binary chromatin-state probabilities.
    Reimplemented with the same conv-stage/channel-doubling/two-stage-FC
    topology at a reduced sequence length and feature count for catalog
    size.
  - C.Origami (cand_00657): Tan, Xiong, Kuang, et al., Nature Biotechnology
    2023, https://www.nature.com/articles/s41587-022-01601-z, official repo
    https://github.com/tanjimin/C.Origami (``src/corigami/model/blocks.py``,
    ``corigami_models.py``, class ``ConvTransModel``). Ported faithfully:
    a dual-branch dilated 1D-ResNet encoder (``EncoderSplit``, one branch for
    the one-hot DNA sequence, one for stacked epigenomic tracks e.g.
    CTCF ChIP-seq + ATAC-seq, each independently downsampled through
    residual conv blocks, concatenated along channels), an 8-layer pre-LN
    Transformer encoder over the resulting 1D feature track (position
    embeddings + multi-head self-attention, matching ``AttnModule``), a
    "diagonalize" step that outer-product-broadcasts the 1D feature track
    into a symmetric pairwise ``(2*hidden, N, N)`` map (the CNN+Transformer
    -to-2D-contact-map mechanism that defines C.Origami), and a dilated
    2D-ResNet decoder (``Decoder``, dilation doubling per block) predicting
    a single-channel Hi-C-style contact-frequency map. Reimplemented at
    reduced channel widths / block counts / sequence length for catalog
    size, preserving the split-encoder -> transformer -> diagonalize ->
    dilated-2D-decoder pipeline exactly.
  - Cell BLAST / DIRECTi (cand_00658): Cao, Gao, et al., Nature
    Communications 2020, https://www.nature.com/articles/s41467-020-17281-7,
    official repo https://github.com/gao-lab/Cell_BLAST (``directi.py``,
    ``latent.py`` class ``Gau``, ``rmbatch.py`` class ``Adversarial``,
    ``prob.py`` class ``NB``). DIRECTi's namesake mechanism is a
    semi-supervised, *adversarially batch-corrected* variational
    autoencoder: a Gaussian-latent MLP encoder (``Gau``) maps gene-expression
    input to a low-dimensional cell embedding; a negative-binomial MLP
    decoder (``NB``) reconstructs per-gene counts from the latent code
    (softmax-normalized mean scaled by library size, plus a learned
    dispersion); and an **adversarial batch discriminator**
    (``rmbatch.Adversarial``: an MLP classifying batch/donor identity
    directly from the latent code) is trained in a GAN-style minimax against
    the encoder, whose gradient is reversed/negated on the generator step so
    the latent space becomes uninformative about batch identity while
    remaining informative about cell state -- the defining
    "adversarially-regularized VAE for batch-effect removal" trick.
    Reimplemented with the same Gaussian-VAE-encoder / NB-decoder /
    adversarial-batch-discriminator topology (gradient reversal implemented
    via a ``GradientReversalFunction`` autograd op, matching the reference's
    "flip the discriminator gradient" semantics for the encoder's objective)
    at compact dimensions.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.autograd import Function


# ============================================================
# AttnToCrispr -- encoder-decoder Transformer + 2D-CNN head over
# guide/target CRISPR sequences (qiaoliuhub/AttnToCrispr)
# ============================================================


class _TransformerBlock(nn.Module):
    """Pre-embedded encoder or decoder block: self-attn (+ optional cross-attn) + FFN."""

    def __init__(self, d_model: int, heads: int, dropout: float, is_decoder: bool) -> None:
        super().__init__()
        self.is_decoder = is_decoder
        self.self_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        if is_decoder:
            self.cross_attn = nn.MultiheadAttention(
                d_model, heads, dropout=dropout, batch_first=True
            )
            self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, enc_out: Tensor | None = None) -> Tensor:
        """Apply self-attention, optional cross-attention, and a feed-forward block."""
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        if self.is_decoder:
            cross_out, _ = self.cross_attn(x, enc_out, enc_out)
            x = self.norm2(x + cross_out)
        x = self.norm3(x + self.ffn(x))
        return x


class AttnToCrispr(nn.Module):
    """Encoder-decoder Transformer + 2D-CNN head for CRISPR efficiency scoring.

    Reproduces the ``Transformer`` + ``customized_CNN`` pipeline from
    ``attention_model.py``: guide-RNA and target-site one-hot sequences are
    each embedded (token + positional embedding), pushed through a small
    multi-head-attention encoder/decoder stack, and the decoder output is
    treated as a single-channel image and refined by a small 2D CNN before a
    feed-forward head regresses one editing-efficiency score.
    """

    def __init__(
        self,
        vocab_size: int = 5,
        seq_len: int = 23,
        d_model: int = 32,
        heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.src_embed = nn.Embedding(vocab_size, d_model)
        self.trg_embed = nn.Embedding(vocab_size, d_model)
        self.src_pos = nn.Embedding(seq_len, d_model)
        self.trg_pos = nn.Embedding(seq_len, d_model)
        self.encoder_layers = nn.ModuleList(
            [_TransformerBlock(d_model, heads, dropout, is_decoder=False) for _ in range(n_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [_TransformerBlock(d_model, heads, dropout, is_decoder=True) for _ in range(n_layers)]
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 1), padding=(1, 0)),
            nn.MaxPool2d(kernel_size=(2, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 1), padding=(1, 0)),
            nn.MaxPool2d(kernel_size=(2, 1), padding=(1, 0)),
            nn.ReLU(),
        )
        cnn_out_len = ((seq_len + 2) // 2 + 2) // 2
        self.head = nn.Sequential(
            nn.Linear(32 * cnn_out_len * d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, guide_seq: Tensor, target_seq: Tensor) -> Tensor:
        """Predict one editing-efficiency score per guide/target pair.

        Parameters
        ----------
        guide_seq : Tensor
            Shape ``(batch, seq_len)`` long token ids for the guide RNA.
        target_seq : Tensor
            Shape ``(batch, seq_len)`` long token ids for the target site.
        """
        pos = torch.arange(self.seq_len, device=guide_seq.device).unsqueeze(0)
        src = self.src_embed(guide_seq) + self.src_pos(pos)
        trg = self.trg_embed(target_seq) + self.trg_pos(pos)

        enc_out = src
        for layer in self.encoder_layers:
            enc_out = layer(enc_out)

        dec_out = trg
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out)

        image = dec_out.unsqueeze(1)  # (batch, 1, seq_len, d_model)
        feat = self.cnn(image).flatten(1)
        return self.head(feat).squeeze(-1)


def build_attntocrispr() -> nn.Module:
    """Build a small AttnToCrispr encoder-decoder Transformer + CNN scorer."""
    return AttnToCrispr(vocab_size=5, seq_len=23, d_model=32, heads=4, n_layers=2).eval()


def example_input_attntocrispr() -> tuple[Tensor, Tensor]:
    """Return guide-RNA and target-site token sequences for AttnToCrispr."""
    guide_seq = torch.randint(0, 5, (2, 23))
    target_seq = torch.randint(0, 5, (2, 23))
    return guide_seq, target_seq


# ============================================================
# BABEL -- spliced dual-encoder / shared-latent / dual-decoder
# cross-modal (RNA <-> ATAC) autoencoder (wukevin/babel)
# ============================================================


class _BabelEncoder(nn.Module):
    """Modality-specific MLP encoder (mirrors ``babel.models.autoencoders.Encoder``)."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.Linear(64, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encode a modality-specific feature vector to the shared latent space."""
        return self.net(x)


class _BabelDecoder(nn.Module):
    """Modality-specific MLP decoder (mirrors ``babel.models.autoencoders.Decoder``)."""

    def __init__(self, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.PReLU(),
        )
        self.out = nn.Linear(64, output_dim)

    def forward(self, z: Tensor) -> Tensor:
        """Decode a shared latent code back to a modality-specific feature vector."""
        return self.out(self.net(z))


class BABEL(nn.Module):
    """Spliced dual-encoder / shared-latent / dual-decoder RNA<->ATAC translator.

    Ports ``SplicedAutoEncoder.forward``: independent RNA and ATAC encoders
    map into one shared latent space; independent RNA and ATAC decoders can
    each decode from either encoder's latent code, giving all four
    cross-modal reconstructions (RNA->RNA, RNA->ATAC, ATAC->RNA, ATAC->ATAC)
    -- the "translate between modalities" mechanism the BABEL name refers to.
    """

    def __init__(self, rna_dim: int = 64, atac_dim: int = 128, latent_dim: int = 16) -> None:
        super().__init__()
        self.encoder_rna = _BabelEncoder(rna_dim, latent_dim)
        self.encoder_atac = _BabelEncoder(atac_dim, latent_dim)
        self.decoder_rna = _BabelDecoder(rna_dim, latent_dim)
        self.decoder_atac = _BabelDecoder(atac_dim, latent_dim)

    def forward(self, rna: Tensor, atac: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return the four cross-modal reconstructions RNA->RNA/ATAC and ATAC->RNA/ATAC."""
        z_rna = self.encoder_rna(rna)
        z_atac = self.encoder_atac(atac)

        rna_to_rna = self.decoder_rna(z_rna)
        rna_to_atac = self.decoder_atac(z_rna)
        atac_to_rna = self.decoder_rna(z_atac)
        atac_to_atac = self.decoder_atac(z_atac)
        return rna_to_rna, rna_to_atac, atac_to_rna, atac_to_atac


def build_babel() -> nn.Module:
    """Build a small BABEL spliced RNA<->ATAC autoencoder."""
    return BABEL(rna_dim=64, atac_dim=128, latent_dim=16).eval()


def example_input_babel() -> tuple[Tensor, Tensor]:
    """Return paired RNA and ATAC feature vectors for BABEL."""
    rna = torch.randn(4, 64)
    atac = torch.randn(4, 128)
    return rna, atac


# ============================================================
# Basenji1 -- conv-pool tower + dilated dense-residual tower
# (Kelley et al. 2018 Genome Research; calico/basenji)
# ============================================================


class _Basenji1DilatedDenseBlock(nn.Module):
    """One dilated Conv1d block whose output is concatenated onto the running stream.

    Mirrors the ``cnn_dense=1`` dilated stages in the 2018 params
    (``cnn_dilation`` doubling 1,2,4,...,64): DenseNet-style
    skip-concatenation rather than additive ResNet skip, so channel width
    grows by ``growth`` at every dilated stage.
    """

    def __init__(self, in_channels: int, growth: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, growth, kernel_size, dilation=dilation, padding=pad),
            nn.BatchNorm1d(growth),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the dilated conv and concatenate its output onto the input."""
        return torch.cat([x, self.block(x)], dim=1)


class Basenji1(nn.Module):
    """Dilated-CNN "conv tower + dense dilated tower" genomic-track predictor.

    Reproduces the 2018 Basenji architecture (``manuscripts/genome_research
    2018/params.txt``): an initial wide-kernel conv, several
    conv-batchnorm-ReLU-maxpool stages that progressively downsample and
    widen the one-hot DNA sequence, then a stack of dilated Conv1d blocks
    with exponentially increasing dilation whose outputs are densely
    concatenated onto a running feature stream (``cnn_dense=1``), ending in
    a 1x1 conv exponential-link Poisson-regression head predicting many
    genomic tracks per output bin.
    """

    def __init__(
        self,
        in_channels: int = 4,
        seq_len: int = 512,
        pool_channels: tuple[int, ...] = (32, 48, 64),
        dilated_growth: int = 16,
        n_dilated: int = 4,
        n_tracks: int = 8,
    ) -> None:
        super().__init__()
        pool_layers = []
        c_in = in_channels
        pool_layers.append(
            nn.Sequential(
                nn.Conv1d(c_in, pool_channels[0], kernel_size=15, padding=7),
                nn.BatchNorm1d(pool_channels[0]),
                nn.ReLU(),
            )
        )
        c_in = pool_channels[0]
        for c_out in pool_channels[1:]:
            pool_layers.append(
                nn.Sequential(
                    nn.Conv1d(c_in, c_out, kernel_size=5, padding=2),
                    nn.BatchNorm1d(c_out),
                    nn.ReLU(),
                    nn.MaxPool1d(4),
                )
            )
            c_in = c_out
        self.conv_tower = nn.Sequential(*pool_layers)

        dilated_blocks = []
        for i in range(n_dilated):
            dilation = 2**i
            dilated_blocks.append(_Basenji1DilatedDenseBlock(c_in, dilated_growth, 3, dilation))
            c_in += dilated_growth
        self.dilated_tower = nn.Sequential(*dilated_blocks)

        self.readout = nn.Conv1d(c_in, n_tracks, kernel_size=1)

    def forward(self, dna_onehot: Tensor) -> Tensor:
        """Predict multi-track genomic signal from one-hot DNA sequence.

        Parameters
        ----------
        dna_onehot : Tensor
            Shape ``(batch, 4, seq_len)`` one-hot encoded DNA sequence.
        """
        x = self.conv_tower(dna_onehot)
        x = self.dilated_tower(x)
        return torch.exp(self.readout(x))


def build_basenji1() -> nn.Module:
    """Build a small Basenji1 dilated-dense-tower genomic-track predictor."""
    return Basenji1(
        in_channels=4,
        seq_len=512,
        pool_channels=(32, 48, 64),
        dilated_growth=16,
        n_dilated=4,
        n_tracks=8,
    ).eval()


def example_input_basenji1() -> Tensor:
    """Return a one-hot DNA sequence tensor for Basenji1."""
    return torch.randn(2, 4, 512)


# ============================================================
# Beluga -- 2D-CNN chromatin-feature predictor (DeepSEA lineage,
# shipped in FunctionLab/ExPecto chromatin.py)
# ============================================================


class Beluga(nn.Module):
    """2D-CNN over one-hot DNA predicting per-feature chromatin-state probabilities.

    Ports ``chromatin.py``'s ``Beluga`` class near-verbatim: three widening
    conv stages (each two ``Conv2d(kernel=(1,8))`` + ReLU layers, with
    dropout and ``(1,4)`` max pooling after the first two stages) over DNA
    one-hot encoded as a ``(4, 1, L)`` image, followed by flatten -> dropout
    -> a large FC hidden layer -> ReLU -> a final FC layer -> sigmoid,
    giving one probability per chromatin/TF-binding feature.
    """

    def __init__(self, seq_len: int = 200, n_features: int = 64) -> None:
        super().__init__()
        self.conv_tower = nn.Sequential(
            nn.Conv2d(4, 32, (1, 8)),
            nn.ReLU(),
            nn.Conv2d(32, 32, (1, 8)),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d((1, 4), (1, 4)),
            nn.Conv2d(32, 48, (1, 8)),
            nn.ReLU(),
            nn.Conv2d(48, 48, (1, 8)),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d((1, 4), (1, 4)),
            nn.Conv2d(48, 64, (1, 8)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (1, 8)),
            nn.ReLU(),
        )
        # Compute the flattened conv-tower output width for the given seq_len
        # by running a dummy forward pass (avoids hand-derived off-by-ones from
        # the stacked kernel-8 convs + (1,4) pools).
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 1, seq_len)
            flat_dim = self.conv_tower(dummy).flatten(1).shape[1]
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_features),
            nn.Sigmoid(),
        )

    def forward(self, dna_onehot: Tensor) -> Tensor:
        """Predict per-feature chromatin-state probabilities from one-hot DNA.

        Parameters
        ----------
        dna_onehot : Tensor
            Shape ``(batch, 4, 1, seq_len)`` one-hot encoded DNA sequence.
        """
        x = self.conv_tower(dna_onehot)
        return self.head(x)


def build_beluga() -> nn.Module:
    """Build a small Beluga 2D-CNN chromatin-feature predictor."""
    return Beluga(seq_len=500, n_features=64).eval()


def example_input_beluga() -> Tensor:
    """Return a one-hot DNA sequence tensor (as a (4,1,L) image) for Beluga."""
    return torch.randn(2, 4, 1, 500)


# ============================================================
# C.Origami -- split dilated-ResNet encoder + Transformer +
# diagonalize + dilated-2D-ResNet decoder Hi-C predictor
# (Tan, Xiong, Kuang et al. 2023; tanjimin/C.Origami)
# ============================================================


class _COrigamiResBlock1D(nn.Module):
    """Strided Conv1d + residual refinement block (mirrors ``blocks.ConvBlock``)."""

    def __init__(self, channels_in: int, channels_out: int, kernel_size: int = 5) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.scale = nn.Sequential(
            nn.Conv1d(channels_in, channels_out, kernel_size, stride=2, padding=pad),
            nn.BatchNorm1d(channels_out),
            nn.ReLU(),
        )
        self.res = nn.Sequential(
            nn.Conv1d(channels_out, channels_out, kernel_size, padding=pad),
            nn.BatchNorm1d(channels_out),
            nn.ReLU(),
            nn.Conv1d(channels_out, channels_out, kernel_size, padding=pad),
            nn.BatchNorm1d(channels_out),
        )
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """Downsample by 2 then apply a residual refinement."""
        scaled = self.scale(x)
        return self.relu(self.res(scaled) + scaled)


class _COrigamiEncoderSplit(nn.Module):
    """Dual-branch encoder: DNA one-hot branch + epigenomic-track branch, concatenated.

    Mirrors ``blocks.EncoderSplit``: sequence and epigenomic-track inputs
    are each independently downsampled through a small stack of residual
    conv blocks, then concatenated channel-wise before a final 1x1 conv.
    """

    def __init__(self, seq_channels: int, epi_channels: int, n_blocks: int, out_dim: int) -> None:
        super().__init__()
        half = 16
        self.start_seq = nn.Sequential(
            nn.Conv1d(seq_channels, half, 3, stride=2, padding=1), nn.BatchNorm1d(half), nn.ReLU()
        )
        self.start_epi = nn.Sequential(
            nn.Conv1d(epi_channels, half, 3, stride=2, padding=1), nn.BatchNorm1d(half), nn.ReLU()
        )
        seq_blocks, epi_blocks = [], []
        c = half
        for _ in range(n_blocks):
            seq_blocks.append(_COrigamiResBlock1D(c, c))
            epi_blocks.append(_COrigamiResBlock1D(c, c))
        self.res_seq = nn.Sequential(*seq_blocks)
        self.res_epi = nn.Sequential(*epi_blocks)
        self.conv_end = nn.Conv1d(c * 2, out_dim, 1)

    def forward(self, seq: Tensor, epi: Tensor) -> Tensor:
        """Encode DNA one-hot and epigenomic tracks to one concatenated feature track."""
        seq = self.res_seq(self.start_seq(seq))
        epi = self.res_epi(self.start_epi(epi))
        return self.conv_end(torch.cat([seq, epi], dim=1))


class _COrigamiResBlockDilated2D(nn.Module):
    """Dilated Conv2d residual block used in the 2D decoder tower."""

    def __init__(self, channels: int, dilation: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """Apply the dilated residual block."""
        return self.relu(self.res(x) + x)


class CORIGAMI(nn.Module):
    """Split-encoder + Transformer + diagonalize + dilated-2D-decoder Hi-C predictor.

    Ports ``ConvTransModel`` from ``corigami_models.py``: DNA sequence and
    epigenomic tracks are encoded by :class:`_COrigamiEncoderSplit`, refined
    by a multi-head-attention Transformer encoder over genomic position, then
    "diagonalized" -- broadcast via an outer-product-style tiling into a
    symmetric 2D pairwise feature map -- and decoded by a dilated 2D-ResNet
    tower into a single-channel Hi-C-style contact-frequency map, matching
    the reference's CNN-Transformer-to-contact-map pipeline.
    """

    def __init__(
        self,
        seq_channels: int = 5,
        epi_channels: int = 2,
        hidden: int = 32,
        n_enc_blocks: int = 3,
        n_transformer_layers: int = 2,
        n_transformer_heads: int = 4,
        n_dec_blocks: int = 3,
        map_len: int = 16,
    ) -> None:
        super().__init__()
        self.map_len = map_len
        self.encoder = _COrigamiEncoderSplit(seq_channels, epi_channels, n_enc_blocks, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden, n_transformer_heads, dim_feedforward=hidden * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_transformer_layers)

        dec_blocks = []
        for i in range(n_dec_blocks):
            dec_blocks.append(_COrigamiResBlockDilated2D(hidden, dilation=2 ** (i + 1)))
        self.dec_start = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU()
        )
        self.dec_blocks = nn.Sequential(*dec_blocks)
        self.dec_end = nn.Conv2d(hidden, 1, 1)

    def diagonalize(self, x: Tensor) -> Tensor:
        """Outer-product-broadcast a (batch, C, L) track into a (batch, 2C, L, L) map."""
        length = x.size(-1)
        x_i = x.unsqueeze(2).repeat(1, 1, length, 1)
        x_j = x.unsqueeze(3).repeat(1, 1, 1, length)
        return torch.cat([x_i, x_j], dim=1)

    def forward(self, seq: Tensor, epi: Tensor) -> Tensor:
        """Predict a Hi-C-style contact map from DNA sequence and epigenomic tracks.

        Parameters
        ----------
        seq : Tensor
            Shape ``(batch, seq_channels, length)`` one-hot DNA sequence.
        epi : Tensor
            Shape ``(batch, epi_channels, length)`` epigenomic tracks.
        """
        feat = self.encoder(seq, epi)  # (batch, hidden, L')
        feat = self.transformer(feat.transpose(1, 2)).transpose(1, 2)
        feat = feat[..., : self.map_len]
        contact_map = self.diagonalize(feat)
        x = self.dec_start(contact_map)
        x = self.dec_blocks(x)
        return self.dec_end(x).squeeze(1)


def build_corigami() -> nn.Module:
    """Build a small C.Origami CNN+Transformer Hi-C contact-map predictor."""
    return CORIGAMI(
        seq_channels=5,
        epi_channels=2,
        hidden=32,
        n_enc_blocks=3,
        n_transformer_layers=2,
        n_transformer_heads=4,
        n_dec_blocks=3,
        map_len=16,
    ).eval()


def example_input_corigami() -> tuple[Tensor, Tensor]:
    """Return DNA one-hot and epigenomic-track tensors for C.Origami."""
    seq = torch.randn(2, 5, 128)
    epi = torch.randn(2, 2, 128)
    return seq, epi


# ============================================================
# Cell BLAST (DIRECTi) -- Gaussian VAE encoder + NB decoder +
# adversarial batch-effect discriminator (gao-lab/Cell_BLAST)
# ============================================================


class _GradientReversalFunction(Function):
    """Reverses (negates) the gradient in the backward pass, scaled by ``lambd``.

    Implements the "flip the discriminator's gradient before it reaches the
    encoder" trick used by DIRECTi's adversarial batch-correction training
    (``rmbatch.Adversarial.g_loss`` negates the discriminator loss on the
    generator/encoder step, which is mechanistically a gradient reversal).
    """

    @staticmethod
    def forward(ctx, x: Tensor, lambd: float) -> Tensor:  # noqa: D102
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # noqa: D102
        return -ctx.lambd * grad_output, None


def _grad_reverse(x: Tensor, lambd: float = 1.0) -> Tensor:
    """Apply the gradient-reversal identity function."""
    return _GradientReversalFunction.apply(x, lambd)


class CellBLASTEncoder(nn.Module):
    """Gaussian-latent MLP encoder (mirrors ``latent.Gau``)."""

    def __init__(self, input_dim: int, latent_dim: int, h_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, h_dim), nn.BatchNorm1d(h_dim), nn.ReLU())
        self.gau = nn.Linear(h_dim, latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Encode gene-expression counts to a latent cell embedding."""
        return self.gau(self.mlp(x))


class CellBLASTDecoder(nn.Module):
    """Negative-binomial MLP decoder (mirrors ``prob.NB``)."""

    def __init__(self, output_dim: int, latent_dim: int, h_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(latent_dim, h_dim), nn.BatchNorm1d(h_dim), nn.ReLU())
        self.mu = nn.Linear(h_dim, output_dim)
        self.log_theta = nn.Linear(h_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, z: Tensor, library_size: Tensor) -> tuple[Tensor, Tensor]:
        """Decode a latent code to negative-binomial mean/dispersion parameters."""
        h = self.mlp(z)
        mu = self.softmax(self.mu(h)) * library_size.unsqueeze(1)
        log_theta = self.log_theta(h)
        return mu, log_theta


class CellBLASTDiscriminator(nn.Module):
    """Adversarial batch-identity discriminator over the latent space (mirrors
    ``rmbatch.Adversarial``)."""

    def __init__(self, latent_dim: int, n_batches: int, h_dim: int = 32) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(latent_dim, h_dim), nn.ReLU())
        self.pred = nn.Linear(h_dim, n_batches)

    def forward(self, z: Tensor) -> Tensor:
        """Predict batch-identity logits from a latent embedding."""
        return self.pred(self.mlp(z))


class CellBLAST(nn.Module):
    """Adversarially batch-corrected semi-supervised VAE for single-cell embedding.

    Ports the core DIRECTi pipeline used by Cell BLAST: a Gaussian-latent MLP
    encoder maps gene-expression counts to a cell embedding
    (:class:`CellBLASTEncoder`); a negative-binomial MLP decoder reconstructs
    per-gene counts from the embedding (:class:`CellBLASTDecoder`); and an
    adversarial batch discriminator (:class:`CellBLASTDiscriminator`) is
    applied to a gradient-reversed copy of the latent code, so encoder
    training implicitly minimizes batch-identity information in the latent
    space while decoder training maximizes reconstruction fidelity -- the
    "adversarially-regularized VAE" mechanism that defines DIRECTi.
    """

    def __init__(
        self, n_genes: int = 128, latent_dim: int = 16, n_batches: int = 3, h_dim: int = 64
    ) -> None:
        super().__init__()
        self.encoder = CellBLASTEncoder(n_genes, latent_dim, h_dim)
        self.decoder = CellBLASTDecoder(n_genes, latent_dim, h_dim)
        self.discriminator = CellBLASTDiscriminator(latent_dim, n_batches)

    def forward(self, exprs: Tensor, library_size: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode, decode, and adversarially discriminate a batch of expression counts.

        Parameters
        ----------
        exprs : Tensor
            Shape ``(batch, n_genes)`` raw gene-expression counts.
        library_size : Tensor
            Shape ``(batch,)`` per-cell total counts (library size).
        """
        z = self.encoder(exprs)
        mu, log_theta = self.decoder(z, library_size)
        batch_logits = self.discriminator(_grad_reverse(z, lambd=1.0))
        return mu, log_theta, batch_logits


def build_cell_blast() -> nn.Module:
    """Build a small Cell BLAST (DIRECTi) adversarially batch-corrected VAE."""
    return CellBLAST(n_genes=128, latent_dim=16, n_batches=3, h_dim=64).eval()


def example_input_cell_blast() -> tuple[Tensor, Tensor]:
    """Return gene-expression counts and library sizes for Cell BLAST."""
    exprs = torch.rand(8, 128) * 10
    library_size = exprs.sum(dim=1)
    return exprs, library_size


# ============================================================
# Registry
# ============================================================

MENAGERIE_ENTRIES = [
    ("AttnToCrispr", "build_attntocrispr", "example_input_attntocrispr", "2021", "BIO"),
    ("BABEL", "build_babel", "example_input_babel", "2021", "BIO"),
    ("Basenji1", "build_basenji1", "example_input_basenji1", "2018", "BIO"),
    ("Beluga", "build_beluga", "example_input_beluga", "2018", "BIO"),
    ("C.Origami", "build_corigami", "example_input_corigami", "2023", "BIO"),
    ("Cell BLAST", "build_cell_blast", "example_input_cell_blast", "2020", "BIO"),
]
