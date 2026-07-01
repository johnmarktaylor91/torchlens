"""Menagerie batch w6a3: computational-genomics deep-learning classics for
epigenomic-track pretraining, 3D-genome contact-map prediction from
epigenomic tracks, tissue-specific expression-effect prediction from
chromatin features, cell-type-specific TF-binding prediction, long-context
DNA language modeling, and TF-gene regulatory-link prediction from
scRNA-seq.

Sources checked (reference only; no cloning, no pip installs):
  - EpiGePT (cand_00719): Gao, Yu, et al., Genome Biology 2024,
    https://github.com/ZjGaothu/EpiGePT (``model/EpiGePT.py``, class
    ``EpiGePT``). The defining mechanism: a per-bin 1D-CNN
    (``Convmodule``, five widening conv+maxpool stages) embeds one-hot DNA
    sequence into a per-genomic-bin token; each token is concatenated with a
    matching TF-binding-motif-score embedding for that bin
    (``batch_inputs_tf``); the resulting bin-token sequence (sequence
    embedding + TF-context, *not* a fixed vocabulary) is fed as
    ``inputs_embeds`` directly into a BERT transformer encoder
    (``transformermodule``), and a linear multi-task head
    (``Multitaskmodule``) predicts many epigenomic-signal tracks per bin --
    i.e. "sequence CNN token stream + TF-context, self-attended by a BERT
    encoder, multi-task regression head" is EpiGePT's namesake contribution
    over plain DeepSEA-style CNNs. Reimplemented with the same
    CNN-tokenizer -> concat-TF-context -> BERT-encoder -> multi-task-linear
    topology at reduced channel widths / sequence length / layer count.
  - Epiphany (cand_00720): Yang, Das, et al. (arnavmdas), Genome Biology
    2023, https://github.com/arnavmdas/epiphany (``epiphany/model_5kb.py``,
    class ``Net``). The defining mechanism: five epigenomic tracks (e.g.
    CTCF, H3K27ac, ATAC-seq, ...) are read with a sliding window
    (``torch.as_strided`` over the genome-wide track) into a sequence of
    overlapping windows; each window is embedded by a 4-stage 1D-CNN
    (``ConvBlock`` x4 with maxpooling + dropout, matching the reference's
    conv1-conv4 stack) into one token per genomic bin; the resulting
    bin-token sequence is passed through two **stacked, residually-summed
    bidirectional LSTMs** (``rnn1``/``rnn2``, ``res2 = res2 + res1``) and a
    2-layer MLP head predicts a vector of Hi-C contact-map values (one
    output "stripe" per bin) -- the sliding-window-CNN-tokenizer +
    residual-BiLSTM-stack is exactly the "predict 3D genome contacts from
    1D epigenomic tracks" mechanism Epiphany is named for. Reimplemented
    with the same strided-window CNN tokenizer -> residual stacked BiLSTM
    -> MLP contact-map head topology at reduced window/track/hidden sizes
    (the reference's optional adversarial ``Disc`` GAN head is a training
    add-on, not part of the core predictive architecture, so it is omitted
    here).
  - ExPecto (cand_00721): Zhou, Theesfeld, Yao, Chen, Wong & Troyanskaya,
    Nature Genetics 2018, https://github.com/FunctionLab/ExPecto
    (``chromatin.py`` class ``Beluga`` for stage 1 -- already catalogued
    separately as the ``Beluga`` classic in ``gen_w5a16.py`` -- and
    ``predict.py``'s ``compute_effects`` for stage 2). ExPecto's namesake
    contribution over Beluga alone is stage 2: **spatial expression-effect
    aggregation**. ``compute_effects`` builds, for a genomic variant at a
    given distance to the gene's TSS, a bank of exponential-decay spatial
    weights at multiple decay rates (0.01/0.02/0.05/0.1/0.2 per 200bp bin,
    separately for upstream/downstream) and multiplies each weight against
    the (tiled) per-feature chromatin-effect vector predicted by the stage-1
    CNN at that offset, then a regression model maps the resulting
    "chromatin features x spatial-decay-weighted distance-to-TSS" tensor to
    a predicted expression log-fold-change (the reference uses gradient
    boosted trees for this regressor; reimplemented here as a differentiable
    linear read-out so the whole pipeline is one traceable ``nn.Module``,
    preserving the reference's neural, learned part -- the stage-1 CNN --
    and its distinctive multi-decay-rate spatial-weighting feature
    construction exactly). Reimplemented as ``ExPecto``: a compact
    Beluga-style 2D-CNN chromatin-effect predictor evaluated at several
    genomic offsets around a TSS, whose outputs are combined with the
    reference's multi-decay-rate exponential spatial-weight bank before a
    final linear expression-effect head.
  - FactorNet (cand_00722): Quang & Xie, Methods 2019,
    https://github.com/uci-cbcl/FactorNet (``utils.py``'s ``make_model``).
    The defining mechanism: a **weight-shared "siamese" forward/reverse-
    complement** architecture -- the *same* hidden-layer stack
    (``Convolution1D`` motif-scan -> ``TimeDistributed(Dense)`` motif
    rescore -> ``MaxPooling1D`` -> ``Bidirectional(LSTM)`` -> ``Flatten`` ->
    ``Dense`` -> sigmoid, matching ``hidden_layers`` in ``make_model``) is
    applied independently to the forward-strand one-hot-DNA-plus-bigWig-
    track input and to the reverse-complement-strand input, and the two
    strand predictions are averaged (``merge(..., mode='ave')``) -- encoding
    the biological prior that TF binding is strand-symmetric. Reimplemented
    with the same tied-weight dual-branch (forward strand / reverse-
    complement strand, one shared ``nn.Module`` applied to both, outputs
    averaged) conv -> per-position dense motif rescore -> maxpool -> BiLSTM
    -> dense -> sigmoid topology at reduced channel widths / sequence
    length.
  - GENA-LM (cand_00724): Fishman, Kuratov, Petrov, et al., AIRI Institute,
    https://github.com/AIRI-Institute/GENA_LM
    (``src/gena_lm/modeling_rmt.py``, class
    ``RMTEncoderForSequenceClassification``). GENA-LM's headline mechanism
    for handling DNA sequences far longer than a normal transformer's
    context window (up to 36kb) is the **Recurrent Memory Transformer**
    wrapper: a long sequence is split into fixed-size segments; a small
    bank of learned "memory token" embeddings is prepended to each segment
    before it is run through a standard BERT encoder; the *updated* memory
    tokens read back out of the encoder's final hidden state
    (``out.hidden_states[-1][:, self.memory_position]``) become the memory
    fed into the *next* segment's forward pass -- i.e. information is
    carried recurrently, segment to segment, through a small set of learned
    memory slots rather than by attending over the whole 36kb at once.
    Reimplemented as ``GenaLMRMT``: a compact BERT-style transformer encoder
    wrapped with the same prepend-memory-tokens -> encode-segment ->
    read-back-updated-memory -> feed-into-next-segment recurrence, applied
    across a *fixed* number of DNA segments (a traceable, fixed-iteration
    analogue of the reference's variable-length ``pad_and_segment`` loop),
    at reduced hidden size / segment length / segment count / layer count.
  - GENELink (cand_00725): Yuan, Bar-Joseph, et al. (Chen & Liu group),
    Bioinformatics 2022, https://github.com/zpliulab/GENELink
    (``Code/scGNN.py``, classes ``GENELink``/``AttentionLayer``). The
    defining mechanism: a two-layer, multi-head **graph attention network**
    over a cell-by-gene expression matrix and a TF-gene candidate-edge
    adjacency (``AttentionLayer``'s ``_prepare_attentional_mechanism_input``
    splits a learned attention vector ``a`` in half, applies each half to
    the source/target linear projections, sums, and masks with the
    adjacency before a softmax -- a GAT-style additive-attention score
    computed via a projection split rather than a concatenation, matching
    the reference exactly) produces a shared gene embedding; separate
    "TF-role" and "target-role" MLP heads (``tf_linear1/2``,
    ``target_linear1/2``) project the shared embedding into role-specific
    embeddings, and a decoder (dot-product / cosine / concat-MLP, matching
    ``decode``) scores candidate TF-gene pairs for regulatory-link
    prediction -- the "one shared multi-head-GAT encoder, two role-specific
    projection heads, pairwise link decoder" pattern is GENELink's namesake
    contribution over plain GCN-based gene-network models. Reimplemented
    with the same multi-head attentional-graph-conv stack (``reduction=
    "concat"`` across heads on layer 1, mean-pooled heads on layer 2,
    matching the reference's two supported reduction modes) -> role-split
    MLP heads -> dot-product decoder topology at reduced gene-count/hidden
    dims.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


# ============================================================
# EpiGePT -- CNN-tokenized DNA + TF-context, BERT-encoded,
# multi-task epigenomic-track regression (ZjGaothu/EpiGePT)
# ============================================================


class _EpiGePTConvModule(nn.Module):
    """Five-stage widening 1D-CNN tokenizer over one-hot DNA (per bin)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(16, 24, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(24, 32, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool1d(2)
        self.conv4 = nn.Conv1d(32, 48, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool1d(2)
        self.conv5 = nn.Conv1d(48, out_channels, kernel_size=3, padding=1)
        self.pool5 = nn.AdaptiveMaxPool1d(1)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """Embed one bin's one-hot DNA window into a single token vector."""
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = self.pool4(self.relu(self.conv4(x)))
        x = self.pool5(self.relu(self.conv5(x)))
        return x


class EpiGePT(nn.Module):
    """CNN-tokenized DNA + TF-context self-attended by a BERT encoder.

    Ports ``model/EpiGePT.py``'s ``EpiGePT`` class: a per-bin CNN
    (``Convmodule``) embeds one-hot DNA into a sequence-dimension token,
    which is concatenated with a per-bin TF-binding-context embedding and
    fed as ``inputs_embeds`` into a ``BertModel`` encoder; a linear
    multi-task head predicts many epigenomic tracks per bin.
    """

    def __init__(
        self,
        n_bins: int = 12,
        bin_len: int = 64,
        seq_dim: int = 24,
        tf_dim: int = 8,
        n_heads: int = 4,
        n_layers: int = 2,
        n_signals: int = 10,
    ) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.bin_len = bin_len
        self.convmodule = _EpiGePTConvModule(4, seq_dim)
        hidden = seq_dim + tf_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 4,
            batch_first=True,
        )
        self.transformermodule = nn.TransformerEncoder(encoder_layer, n_layers)
        self.multitaskmodule = nn.Linear(hidden, n_signals)

    def forward(self, dna_onehot: Tensor, tf_context: Tensor) -> Tensor:
        """Predict per-bin, multi-task epigenomic-signal tracks.

        Parameters
        ----------
        dna_onehot : Tensor
            Shape ``(batch * n_bins, 4, bin_len)`` one-hot DNA windows,
            one window per genomic bin (bins flattened into the batch dim
            for the per-bin CNN tokenizer).
        tf_context : Tensor
            Shape ``(batch, n_bins, tf_dim)`` per-bin TF-binding-motif
            context embedding.
        """
        batch = tf_context.shape[0]
        tokens = self.convmodule(dna_onehot)  # (batch*n_bins, seq_dim, 1)
        tokens = tokens.squeeze(-1).view(batch, self.n_bins, -1)
        x = torch.cat([tokens, tf_context], dim=2)
        x = self.transformermodule(x)
        return torch.relu(self.multitaskmodule(x))


def build_epigept() -> nn.Module:
    """Build a small EpiGePT CNN-tokenizer + BERT-encoder model."""
    return EpiGePT(n_bins=12, bin_len=64, seq_dim=24, tf_dim=8, n_signals=10).eval()


def example_input_epigept() -> tuple[Tensor, Tensor]:
    """Return (one-hot DNA bins, TF-context) inputs for EpiGePT."""
    n_bins, bin_len = 12, 64
    dna = torch.randn(2 * n_bins, 4, bin_len)
    tf_ctx = torch.randn(2, n_bins, 8)
    return dna, tf_ctx


# ============================================================
# Epiphany -- strided-window CNN tokenizer + residual stacked
# BiLSTM predicting Hi-C contact maps (arnavmdas/epiphany)
# ============================================================


class _EpiphanyConvBlock(nn.Module):
    """One Conv1d + ReLU (+ optional MaxPool1d) stage."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_width: int, pool_size: int = 0
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_width)
        self.act = nn.ReLU()
        self.pool_size = pool_size
        if pool_size > 0:
            self.pool = nn.MaxPool1d(pool_size, pool_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.conv(x))
        if self.pool_size > 0:
            x = self.pool(x)
        return x


class Epiphany(nn.Module):
    """Sliding-window CNN tokenizer + residual stacked BiLSTM for Hi-C contacts.

    Ports ``epiphany/model_5kb.py``'s ``Net`` class: a strided window over
    multi-track epigenomic signal is embedded per-bin by a 4-stage 1D-CNN,
    then two stacked bidirectional LSTMs (second LSTM's output added
    residually to the first's) and a 2-layer MLP head predict a Hi-C
    contact-map row per bin.
    """

    def __init__(
        self,
        n_tracks: int = 5,
        window: int = 256,
        seq_length: int = 16,
        hidden: int = 64,
        contact_dim: int = 32,
    ) -> None:
        super().__init__()
        self.n_tracks = n_tracks
        self.window = window
        self.seq_length = seq_length
        self.conv1 = _EpiphanyConvBlock(n_tracks, 24, 9, pool_size=4)
        self.conv2 = _EpiphanyConvBlock(24, 32, 5, pool_size=4)
        self.conv3 = _EpiphanyConvBlock(32, 24, 5, pool_size=2)
        self.conv4 = _EpiphanyConvBlock(24, 8, 3, pool_size=0)
        self.pool = nn.AdaptiveMaxPool1d(4)
        self.rnn1 = nn.LSTM(32, hidden, batch_first=True, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * hidden, hidden, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(2 * hidden, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, contact_dim)

    def forward(self, tracks: Tensor) -> Tensor:
        """Predict per-bin Hi-C contact-map rows from sliding-window tracks.

        Parameters
        ----------
        tracks : Tensor
            Shape ``(batch, seq_length, n_tracks, window)``: for each of
            ``seq_length`` genomic bins, a ``(n_tracks, window)``
            epigenomic-track window (pre-extracted sliding windows, in
            place of the reference's ``torch.as_strided`` view over a
            genome-wide track).
        """
        batch = tracks.shape[0]
        x = tracks.reshape(batch * self.seq_length, self.n_tracks, self.window)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = x.reshape(batch, self.seq_length, -1)
        res1, _ = self.rnn1(x)
        res2, _ = self.rnn2(res1)
        res2 = res2 + res1
        x = self.act(self.fc1(res2))
        return self.fc2(x)


def build_epiphany() -> nn.Module:
    """Build a small Epiphany strided-CNN + residual-BiLSTM contact predictor."""
    return Epiphany(n_tracks=5, window=256, seq_length=16, hidden=64, contact_dim=32).eval()


def example_input_epiphany() -> Tensor:
    """Return a batch of sliding-window multi-track epigenomic signal."""
    return torch.randn(2, 16, 5, 256)


# ============================================================
# ExPecto -- Beluga-style CNN chromatin-effect predictor at
# multiple TSS offsets, combined via multi-decay-rate spatial
# weighting into an expression-effect readout
# (FunctionLab/ExPecto)
# ============================================================


class _ExPectoChromatinCNN(nn.Module):
    """Compact Beluga-style 2D-CNN chromatin-effect predictor."""

    def __init__(self, seq_len: int, n_features: int) -> None:
        super().__init__()
        self.conv_tower = nn.Sequential(
            nn.Conv2d(4, 24, (1, 8)),
            nn.ReLU(),
            nn.Conv2d(24, 24, (1, 8)),
            nn.ReLU(),
            nn.MaxPool2d((1, 4), (1, 4)),
            nn.Conv2d(24, 32, (1, 8)),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 1, seq_len)
            flat_dim = self.conv_tower(dummy).flatten(1).shape[1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_features),
            nn.Sigmoid(),
        )

    def forward(self, dna_onehot: Tensor) -> Tensor:
        return self.head(self.conv_tower(dna_onehot))


class ExPecto(nn.Module):
    """Beluga-style CNN chromatin predictions, spatially decay-weighted by
    distance to TSS, linearly combined into a predicted expression effect.

    Ports the two-stage ``FunctionLab/ExPecto`` pipeline: stage 1 is the
    ``chromatin.py`` ``Beluga`` CNN (reimplemented compactly here) run at
    each of several genomic offsets flanking a gene's TSS; stage 2 ports
    ``predict.py``'s ``compute_effects`` multi-decay-rate exponential
    spatial-weight bank (five decay rates x upstream/downstream, matching
    the reference's ``0.01/0.02/0.05/0.1/0.2`` constants) that reweights
    each offset's chromatin-effect vector by its distance to the TSS,
    before a final linear read-out predicts one expression log-fold-change
    value (a differentiable stand-in for the reference's gradient-boosted-
    tree regressor, keeping the whole pipeline one traceable module).
    """

    _DECAY_RATES = (0.01, 0.02, 0.05, 0.1, 0.2)

    def __init__(
        self,
        seq_len: int = 200,
        n_chromatin_features: int = 16,
        n_offsets: int = 3,
        bin_size: float = 200.0,
    ) -> None:
        super().__init__()
        self.chromatin_cnn = _ExPectoChromatinCNN(seq_len, n_chromatin_features)
        self.n_offsets = n_offsets
        self.bin_size = bin_size
        n_spatial_features = n_chromatin_features * len(self._DECAY_RATES) * 2
        self.expression_head = nn.Linear(n_spatial_features, 1)

    def forward(self, dna_onehot_by_offset: Tensor, tss_distance: Tensor) -> Tensor:
        """Predict expression log-fold-change from flanking DNA windows.

        Parameters
        ----------
        dna_onehot_by_offset : Tensor
            Shape ``(batch, n_offsets, 4, 1, seq_len)``: one-hot DNA windows
            at each of ``n_offsets`` genomic positions flanking the TSS.
        tss_distance : Tensor
            Shape ``(batch, n_offsets)``: signed distance (in bp) from each
            offset window to the gene TSS.
        """
        batch, n_off = dna_onehot_by_offset.shape[:2]
        flat = dna_onehot_by_offset.reshape(batch * n_off, 4, 1, -1)
        chromatin_effects = self.chromatin_cnn(flat).view(batch, n_off, -1)

        bins = tss_distance / self.bin_size
        upstream_mask = (tss_distance <= 0).float()
        downstream_mask = (tss_distance >= 0).float()
        spatial_terms = []
        for rate in self._DECAY_RATES:
            decay = torch.exp(-rate * torch.floor(torch.abs(bins)))
            spatial_terms.append(decay * upstream_mask)
            spatial_terms.append(decay * downstream_mask)
        # (batch, n_offsets, 2*n_decay_rates)
        spatial_weights = torch.stack(spatial_terms, dim=-1)
        # (batch, n_offsets, n_features, 2*n_decay_rates) -> sum over offsets
        weighted = chromatin_effects.unsqueeze(-1) * spatial_weights.unsqueeze(-2)
        pooled = weighted.sum(dim=1).flatten(1)
        return self.expression_head(pooled)


def build_expecto() -> nn.Module:
    """Build a small ExPecto chromatin-CNN + spatial-weighting model."""
    return ExPecto(seq_len=200, n_chromatin_features=16, n_offsets=3).eval()


def example_input_expecto() -> tuple[Tensor, Tensor]:
    """Return (flanking one-hot DNA windows, TSS distances) for ExPecto."""
    dna = torch.randn(2, 3, 4, 1, 200)
    tss_distance = torch.tensor([[-400.0, 0.0, 400.0], [-200.0, 0.0, 600.0]])
    return dna, tss_distance


# ============================================================
# FactorNet -- weight-shared forward/reverse-complement siamese
# CNN-BiLSTM for cell-type-specific TF-binding prediction
# (uci-cbcl/FactorNet)
# ============================================================


class _FactorNetBranch(nn.Module):
    """Conv motif-scan -> per-position dense rescore -> pool -> BiLSTM -> dense."""

    def __init__(
        self,
        in_channels: int,
        length: int,
        n_motifs: int,
        n_recurrent: int,
        n_dense: int,
        n_tfs: int,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, n_motifs, kernel_size=26)
        self.motif_dense = nn.Linear(n_motifs, n_motifs)
        self.pool = nn.MaxPool1d(13, 13)
        self.blstm = nn.LSTM(n_motifs, n_recurrent, batch_first=True, bidirectional=True)
        # Compute the flattened post-BiLSTM width for the given length by
        # running a dummy forward pass (avoids hand-derived off-by-ones from
        # the conv/pool stride arithmetic).
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, length)
            pooled_len = self.pool(self.conv(dummy)).shape[-1]
        flat_dim = pooled_len * 2 * n_recurrent
        self.dense = nn.Linear(flat_dim, n_dense)
        self.out = nn.Linear(n_dense, n_tfs)

    def forward(self, x: Tensor) -> Tensor:
        """Score TF binding for one strand of one-hot-DNA + bigWig tracks.

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, in_channels, length)``.
        """
        x = torch.relu(self.conv(x))
        x = x.transpose(1, 2)
        x = torch.relu(self.motif_dense(x))
        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.transpose(1, 2)
        x, _ = self.blstm(x)
        x = x.flatten(1)
        x = torch.relu(self.dense(x))
        return torch.sigmoid(self.out(x))


class FactorNet(nn.Module):
    """Weight-shared forward/reverse-complement siamese CNN-BiLSTM.

    Ports ``utils.py``'s ``make_model``: the *same* hidden-layer branch
    (conv motif scan, per-position dense motif rescore, maxpool,
    bidirectional LSTM, dense, sigmoid) is applied independently to the
    forward-strand and reverse-complement-strand input, and the two
    strand predictions are averaged -- encoding strand symmetry of TF
    binding.
    """

    def __init__(
        self,
        in_channels: int = 5,
        length: int = 200,
        n_motifs: int = 16,
        n_recurrent: int = 12,
        n_dense: int = 16,
        n_tfs: int = 1,
    ) -> None:
        super().__init__()
        self.branch = _FactorNetBranch(in_channels, length, n_motifs, n_recurrent, n_dense, n_tfs)

    def forward(self, forward_strand: Tensor, reverse_strand: Tensor) -> Tensor:
        """Average TF-binding probability over forward and reverse-complement strands."""
        forward_out = self.branch(forward_strand)
        reverse_out = self.branch(reverse_strand)
        return (forward_out + reverse_out) / 2.0


def build_factornet() -> nn.Module:
    """Build a small FactorNet siamese CNN-BiLSTM TF-binding predictor."""
    return FactorNet(
        in_channels=5, length=200, n_motifs=16, n_recurrent=12, n_dense=16, n_tfs=1
    ).eval()


def example_input_factornet() -> tuple[Tensor, Tensor]:
    """Return (forward-strand, reverse-complement-strand) input for FactorNet."""
    forward_strand = torch.randn(2, 5, 200)
    reverse_strand = torch.randn(2, 5, 200)
    return forward_strand, reverse_strand


# ============================================================
# GENA-LM (RMT) -- recurrent memory transformer for long-context
# DNA language modeling (AIRI-Institute/GENA_LM)
# ============================================================


class GenaLMRMT(nn.Module):
    """BERT-style encoder wrapped with a recurrent-memory-token relay.

    Ports ``src/gena_lm/modeling_rmt.py``'s
    ``RMTEncoderForSequenceClassification``: a long DNA sequence, split
    into a fixed number of segments, is processed segment by segment; a
    small bank of learned memory-token embeddings is prepended to each
    segment before it is encoded by a shared BERT encoder, and the
    *updated* memory-token hidden states read back out of the encoder
    become the memory fed into the next segment -- carrying information
    across segments through a handful of learned slots rather than full
    long-range attention.
    """

    def __init__(
        self,
        vocab_size: int = 32,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        segment_len: int = 16,
        n_segments: int = 3,
        n_mem_tokens: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.segment_len = segment_len
        self.n_segments = n_segments
        self.n_mem_tokens = n_mem_tokens
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_mem_tokens + segment_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.init_memory = nn.Parameter(torch.randn(1, n_mem_tokens, d_model) * 0.02)

    def forward(self, segment_ids: Tensor) -> Tensor:
        """Encode a fixed number of DNA segments with a memory-token relay.

        Parameters
        ----------
        segment_ids : Tensor
            Shape ``(batch, n_segments, segment_len)`` integer token ids,
            one row of tokens per fixed-length segment.
        """
        batch = segment_ids.shape[0]
        memory = self.init_memory.expand(batch, -1, -1)
        outputs = []
        for seg in range(self.n_segments):
            tokens = self.token_embed(segment_ids[:, seg, :])
            x = torch.cat([memory, tokens], dim=1) + self.pos_embed
            encoded = self.encoder(x)
            memory = encoded[:, : self.n_mem_tokens, :]
            outputs.append(encoded[:, self.n_mem_tokens :, :])
        return torch.cat(outputs, dim=1)


def build_gena_lm() -> nn.Module:
    """Build a small GENA-LM recurrent-memory-transformer encoder."""
    return GenaLMRMT(
        vocab_size=32,
        d_model=64,
        n_heads=4,
        n_layers=2,
        segment_len=16,
        n_segments=3,
        n_mem_tokens=4,
    ).eval()


def example_input_gena_lm() -> Tensor:
    """Return a batch of fixed-length DNA segment token ids for GENA-LM."""
    return torch.randint(0, 32, (2, 3, 16))


# ============================================================
# GENELink -- multi-head graph-attention gene encoder with
# role-split TF/target heads and pairwise link decoder
# (zpliulab/GENELink)
# ============================================================


class _GENELinkAttentionLayer(nn.Module):
    """One GAT-style attention head (additive score via split projection)."""

    def __init__(self, input_dim: int, output_dim: int, alpha: float = 0.2) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.alpha = alpha
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim))
        self.a = nn.Parameter(torch.empty(2 * output_dim, 1))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        nn.init.xavier_uniform_(self.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a, gain=1.414)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """Apply one masked-softmax additive-attention graph-conv step.

        Parameters
        ----------
        x : Tensor
            Shape ``(n_genes, input_dim)`` gene feature matrix.
        adj : Tensor
            Shape ``(n_genes, n_genes)`` dense candidate-edge adjacency mask.
        """
        h = torch.matmul(x, self.weight)
        wh1 = torch.matmul(h, self.a[: self.output_dim, :])
        wh2 = torch.matmul(h, self.a[self.output_dim :, :])
        e = torch.nn.functional.leaky_relu(wh1 + wh2.transpose(0, 1), negative_slope=self.alpha)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = torch.softmax(attention, dim=1)
        h_pass = torch.matmul(attention, h)
        h_pass = torch.nn.functional.leaky_relu(h_pass, negative_slope=self.alpha)
        h_pass = torch.nn.functional.normalize(h_pass, p=2, dim=1)
        return h_pass + self.bias


class GENELink(nn.Module):
    """Multi-head GAT gene encoder, role-split TF/target heads, link decoder.

    Ports ``Code/scGNN.py``'s ``GENELink``/``AttentionLayer``: a two-layer
    multi-head graph-attention network (layer 1 heads concatenated, layer 2
    heads mean-pooled, matching the reference's ``reduction`` modes)
    encodes a cell-by-gene expression matrix plus a TF-gene candidate-edge
    adjacency into a shared gene embedding; separate TF-role and
    target-role MLP heads project into role-specific embeddings; a
    dot-product decoder scores candidate TF-gene pairs.
    """

    def __init__(
        self,
        input_dim: int = 32,
        hidden1_dim: int = 16,
        hidden2_dim: int = 16,
        hidden3_dim: int = 12,
        output_dim: int = 8,
        num_head1: int = 3,
        num_head2: int = 2,
    ) -> None:
        super().__init__()
        self.layer1 = nn.ModuleList(
            [_GENELinkAttentionLayer(input_dim, hidden1_dim) for _ in range(num_head1)]
        )
        layer2_in = num_head1 * hidden1_dim
        self.layer2 = nn.ModuleList(
            [_GENELinkAttentionLayer(layer2_in, hidden2_dim) for _ in range(num_head2)]
        )
        self.tf_linear1 = nn.Linear(hidden2_dim, hidden3_dim)
        self.tf_linear2 = nn.Linear(hidden3_dim, output_dim)
        self.target_linear1 = nn.Linear(hidden2_dim, hidden3_dim)
        self.target_linear2 = nn.Linear(hidden3_dim, output_dim)

    def encode(self, x: Tensor, adj: Tensor) -> Tensor:
        """Two-layer multi-head graph-attention gene encoder."""
        x = torch.cat([head(x, adj) for head in self.layer1], dim=1)
        x = torch.nn.functional.elu(x)
        out = torch.mean(torch.stack([head(x, adj) for head in self.layer2]), dim=0)
        return out

    def forward(self, x: Tensor, adj: Tensor, tf_idx: Tensor, target_idx: Tensor) -> Tensor:
        """Score candidate TF-gene regulatory links.

        Parameters
        ----------
        x : Tensor
            Shape ``(n_genes, input_dim)`` gene expression feature matrix.
        adj : Tensor
            Shape ``(n_genes, n_genes)`` dense candidate-edge adjacency.
        tf_idx : Tensor
            Shape ``(n_pairs,)`` integer indices of TF genes to score.
        target_idx : Tensor
            Shape ``(n_pairs,)`` integer indices of target genes to score.
        """
        embed = self.encode(x, adj)
        tf_embed = torch.nn.functional.leaky_relu(self.tf_linear1(embed))
        tf_embed = torch.nn.functional.leaky_relu(self.tf_linear2(tf_embed))
        target_embed = torch.nn.functional.leaky_relu(self.target_linear1(embed))
        target_embed = torch.nn.functional.leaky_relu(self.target_linear2(target_embed))
        tf_sel = tf_embed[tf_idx]
        target_sel = target_embed[target_idx]
        prob = torch.sum(tf_sel * target_sel, dim=1, keepdim=True)
        return prob


def build_genelink() -> nn.Module:
    """Build a small GENELink multi-head-GAT TF-gene link predictor."""
    return GENELink(
        input_dim=32,
        hidden1_dim=16,
        hidden2_dim=16,
        hidden3_dim=12,
        output_dim=8,
        num_head1=3,
        num_head2=2,
    ).eval()


def example_input_genelink() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return (gene features, adjacency, TF indices, target indices) for GENELink."""
    n_genes = 20
    x = torch.randn(n_genes, 32)
    adj = (torch.rand(n_genes, n_genes) > 0.7).float()
    tf_idx = torch.randint(0, n_genes, (6,))
    target_idx = torch.randint(0, n_genes, (6,))
    return x, adj, tf_idx, target_idx


MENAGERIE_ENTRIES = [
    ("EpiGePT", "build_epigept", "example_input_epigept", "2024", "BIO"),
    ("Epiphany", "build_epiphany", "example_input_epiphany", "2023", "BIO"),
    ("ExPecto", "build_expecto", "example_input_expecto", "2018", "BIO"),
    ("FactorNet", "build_factornet", "example_input_factornet", "2019", "BIO"),
    ("GENA-LM", "build_gena_lm", "example_input_gena_lm", "2023", "BIO"),
    ("GENELink", "build_genelink", "example_input_genelink", "2022", "BIO"),
]
