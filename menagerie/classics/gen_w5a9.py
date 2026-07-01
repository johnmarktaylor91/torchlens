"""Menagerie batch w5a9: phylodynamics, phylogenetics, immunology, and
proteomics/genomics deep-learning classics.

Sources checked (reference only; no cloning, no pip installs):
  - PhyloCNN (cand_00600): Voznica, Zhukova, Boskova, Saulnier, Lemoine,
    Moslonka-Lefebvre & Gascuel, Systematic Biology 2025, official repo
    https://github.com/manolofperez/phyloCNN (training notebooks under
    ``Preprocessing_Training/PhyloCNN_Train_BD.ipynb`` etc.). The published
    ``build_model()`` is a grouped-Conv2D tower over a ``(n_leaves, n_feat,
    2)`` tree encoding (channel 0 = leaf features, channel 1 = internal-node
    features per leaf-neighborhood row, following the CBLV/CDV-style
    "encode a phylogenetic tree as a flat leaf x feature matrix" trick from
    Voznica et al. 2022): a first ``Conv2D(groups=2, kernel=(1, n_feat))``
    collapses the feature axis per channel independently (mirroring the
    ``groups=2`` PhyloCNN uses to keep the leaf-channel and node-channel
    convolutions decoupled at the first layer), two ``Conv2D(kernel=(1,1))``
    refinement layers, global average pooling over the leaf axis, and a
    decreasing MLP head (64-32-16-8-2) regressing the 2 birth-death
    parameters. Reimplemented verbatim with ``nn.Conv2d(groups=2)`` +
    ``AdaptiveAvgPool2d`` + the same decreasing ELU MLP.
  - Phyloformer (cand_00601): Nesterenko, Boussau & Jacob et al., official
    repo https://github.com/lucanest/Phyloformer (``phyloformer/model.py``,
    ``phyloformer/attention.py``), paper https://arxiv.org/abs/2003.05786.
    Ported near-verbatim: a per-site Conv2d embedding of the one-hot+gap MSA
    (22 channels), an axial "row/column attention" stack (``PhyloformerLayer``)
    using a linear-kernel ("favor"-style ELU-feature-map) attention variant
    (``ScaledLinearAttention``) applied alternately across the pair axis and
    the site axis with Conv2d-based pointwise FFN sub-blocks, and a
    ``seq2pair`` binary indicator matrix that lifts per-sequence embeddings
    to per-pair (i,j) embeddings before the attention stack (the paper's
    "attention over sequence *pairs*, not raw sequences" trick), followed by
    a Conv2d readout + softplus + mean-over-sites to output one non-negative
    evolutionary distance per pair.
  - pMTnet (cand_00602): Lu, Zhang, Zhu et al., Nature Machine Intelligence
    2021, https://www.nature.com/articles/s42256-021-00383-2, official repo
    https://github.com/tianshilu/pMTnet (``pMTnet.py``, Keras/TF). The
    shipped code loads three separately-trained Keras sub-networks: a TCR
    CDR3 (Atchley-factor-encoded, ``(80, 5, 1)``) autoencoder whose
    bottleneck (taken 12 layers before the output, per
    ``TCR_encoder.layers[-12]``) is used as a 30-d TCR embedding; an
    HLA+antigen (BLOSUM50-encoded, ``(34, 21)`` and ``(15, 21)``)
    autoencoder whose bottleneck (one layer before the output, per
    ``HLA_antigen_encoder.layers[-2]``) gives a 30-d pMHC embedding; and a
    "ternary" fully-connected classifier (``ternary_prediction``:
    ``concatenate([pos_in, hla_antigen_in])`` -> Dense(300) -> Dropout(0.2)
    -> Dense(200) -> Dense(100) -> Dense(1)``) that scores a TCR/pMHC pair
    from the concatenated 60-d embedding. Reimplemented faithfully in
    PyTorch as three sub-modules (Conv-based TCR encoder, Conv-based
    HLA+antigen encoder, ternary MLP scorer) matching the published
    dimensions and Dense-stack topology.
  - PointNovo (cand_00604): Qiao, Liu, Ma, Su, Yin, Li & Bandeira, Nature
    Machine Intelligence 2021, https://www.nature.com/articles/s42256-021-00304-3,
    codebase is volpato30/DeepNovoV2 (``model.py``). PointNovo's namesake
    mechanism is the order-invariant **T-Net over MS/MS peaks** (a PointNet
    analogue, ``TNet``/``DeepNovoPointNetWithLSTM`` in the reference): each
    candidate next-amino-acid position is scored by comparing its expected
    fragment-ion *locations* against every observed peak location via an
    ``exp(-|location_index - peak_location| * scale)`` distance kernel
    (rather than rasterizing peaks onto a fixed spectrum-image grid, as the
    already-catalogued image-based DeepNovo/DeepNovo-DIA does), stacking
    that per-peak distance feature with the peak intensity, and running a
    shared-weight 1D-conv T-Net (3 Conv1d + BatchNorm + global max-pool +
    2 FC layers) over the (unordered) peak set -- the point-cloud, not
    2D-image, ion representation is exactly the "Point" in PointNovo. The
    ion-T-Net feature is concatenated with an amino-acid LSTM decoder's
    hidden state and projected to next-residue logits, matching
    ``DeepNovoPointNetWithLSTM.forward`` in the reference.
  - PrimateAI (cand_00606): Sundaram, Gao, Padigepati et al., Science 2018,
    https://www.science.org/doi/10.1126/science.aaw5309, official repo
    https://github.com/Illumina/PrimateAI (``source/model.py``,
    ``primateAI_model`` + ``get_ss_model``/``get_sa_model``, Keras). The
    published v1 architecture is a dilated residual Conv1D network over
    five parallel 51-residue one-hot-amino-acid tracks (reference sequence,
    SNP-substituted sequence, and three multiple-species conservation
    tracks for primates/mammals/other-vertebrates); the conservation tracks
    are additionally summed and fed through two auxiliary pretrained
    residual-Conv1D towers (secondary-structure and solvent-accessibility
    predictors, ``get_ss_model``/``get_sa_model``) whose outputs are fused
    additively into the main sequence branches; the reference and SNP
    branches each pass through an initial non-residual block, are
    concatenated/merged, and pushed through a deep residual-Conv1D tower
    (three stages of 2 residual units each, matching ``N=[2,2,2]``,
    ``W=[5,5,5]``) with running skip-accumulation, ending in a
    per-position sigmoid Conv1D and ``GlobalMaxPooling1D`` to output one
    pathogenicity score per variant. Reimplemented faithfully with the same
    five-track input, auxiliary SS/SA towers, additive conservation fusion,
    and residual-tower topology (compact channel/depth for catalog size).

Not built (see ``popV`` note in the discovery notes / build queue for
skip rationale): popV (cand_00605) is an orchestration/ensemble ("popular
vote") framework over externally-trained, mostly non-neural classifiers
(random forest, SVM, XGBoost, Harmony, BBKNN, Scanorama, and the
already-catalogued scANVI), plus OnClass (an external ontology-based
classifier package popV imports and calls, not a model popV itself
defines). There is no single popV-specific trainable ``nn.Module``
architecture distinct from its already-catalogued/non-neural ensemble
members to reimplement.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


# ============================================================
# PhyloCNN -- grouped-Conv2D tree encoding tower (Voznica et al. 2025)
# ============================================================


class PhyloCNN(nn.Module):
    """Grouped-Conv2D CNN over a flattened phylogenetic-tree encoding.

    Reproduces ``build_model()`` from the PhyloCNN birth-death training
    notebook: a ``(n_leaves, n_feat, 2)`` leaf/internal-node feature tensor
    is convolved with a first grouped ``Conv2d`` (one group per channel,
    kernel spanning the full feature axis) so leaf-channel and node-channel
    information stay decoupled at the first layer, followed by two
    pointwise refinement convolutions, global average pooling over the
    leaf axis, and a decreasing ELU MLP head regressing the birth-death
    parameters.
    """

    def __init__(self, n_leaves: int = 32, n_feat: int = 19, n_params: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=(1, n_feat), groups=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(32, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 16),
            nn.ELU(),
            nn.Linear(16, 8),
            nn.ELU(),
            nn.Linear(8, n_params),
            nn.ELU(),
        )
        self.act = nn.ELU()

    def forward(self, tree_encoding: Tensor) -> Tensor:
        """Regress birth-death parameters from a leaf/node feature tensor.

        Parameters
        ----------
        tree_encoding : Tensor
            Shape ``(batch, 2, n_leaves, n_feat)`` -- channel 0 holds
            per-leaf features, channel 1 holds the paired internal-node
            (ancestor) features, following the PhyloCNN CBLV-style
            flattened tree encoding.
        """
        x = self.act(self.bn1(self.conv1(tree_encoding)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.pool(x).flatten(1)
        return self.head(x)


def build_phylocnn() -> nn.Module:
    """Build a small PhyloCNN birth-death parameter estimator."""
    return PhyloCNN(n_leaves=32, n_feat=19, n_params=2).eval()


def example_input_phylocnn() -> Tensor:
    """Return a batch of leaf/node tree-encoding tensors for PhyloCNN."""
    return torch.randn(2, 2, 32, 19)


# ============================================================
# Phyloformer -- axial linear-attention transformer over MSA pairs
# (Nesterenko, Boussau & Jacob; lucanest/Phyloformer)
# ============================================================


def _seq2pair(n_seqs: int) -> Tensor:
    """Binary indicator matrix mapping sequences to unordered pairs."""
    n_pairs = n_seqs * (n_seqs - 1) // 2
    mat = torch.zeros(n_pairs, n_seqs)
    k = 0
    for i in range(n_seqs):
        for j in range(i + 1, n_seqs):
            mat[k, i] = 1.0
            mat[k, j] = 1.0
            k += 1
    return mat


class ScaledLinearAttention(nn.Module):
    """Linear-kernel (ELU-feature-map) attention used inside Phyloformer.

    Ports ``ScaledLinearAttention`` from ``phyloformer/attention.py``: Q/K
    are mapped through ``elu(x) + 1`` and separately normalized (Q by its
    mean, K by its sum) before the linear-attention ``Q @ (K^T V)``
    recombination, avoiding the quadratic-in-sequence softmax attention
    matrix.
    """

    def __init__(self, embed_dim: int, nb_heads: int) -> None:
        super().__init__()
        self.nb_heads = nb_heads
        self.head_dim = embed_dim // nb_heads
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.elu = nn.ELU()

    def forward(self, x: Tensor) -> Tensor:
        """Apply linear attention over the second-to-last axis of ``x``."""
        b, nb_row, nb_col, embed_dim = x.shape
        k = self.k_proj(x).view(b, nb_row, nb_col, self.nb_heads, self.head_dim).transpose(2, 3)
        q = self.q_proj(x).view(b, nb_row, nb_col, self.nb_heads, self.head_dim).transpose(2, 3)
        v = self.v_proj(x).view(b, nb_row, nb_col, self.nb_heads, self.head_dim).transpose(2, 3)

        q = self.elu(q) + 1
        k = self.elu(k) + 1
        q = q / q.mean(dim=-2, keepdim=True)
        k = k / k.sum(dim=-2, keepdim=True)

        ktv = k.transpose(-1, -2) @ v
        out = q @ ktv
        out = out.transpose(2, 3).contiguous().view(b, nb_row, nb_col, embed_dim)
        return self.out_proj(out)


class PhyloformerLayer(nn.Module):
    """One axial row/column linear-attention block with a Conv2d FFN."""

    def __init__(self, embed_dim: int, nb_heads: int) -> None:
        super().__init__()
        self.row_attention = ScaledLinearAttention(embed_dim, nb_heads)
        self.col_attention = ScaledLinearAttention(embed_dim, nb_heads)
        self.row_norm = nn.LayerNorm(embed_dim)
        self.col_norm = nn.LayerNorm(embed_dim)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply row attention, then column attention, then a pointwise FFN."""
        res_row = x
        out = self.row_norm(x.transpose(-1, -3)).transpose(-1, -3)
        out = self.row_attention(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = out + res_row

        res_col = out
        out = self.col_norm(out.transpose(-1, -3)).transpose(-1, -3)
        out = self.col_attention(out.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        out = out + res_col

        res_ffn = out
        out = self.ffn_norm(out.transpose(-1, -3)).transpose(-1, -3)
        out = self.ffn(out)
        return out + res_ffn


class Phyloformer(nn.Module):
    """Axial-attention transformer predicting pairwise evolutionary distances.

    Ports ``Phyloformer`` from ``phyloformer/model.py``: a per-site Conv2d
    embedding of a one-hot(+gap) multiple sequence alignment, a
    ``seq2pair`` lift from per-sequence to per-pair embeddings, a stack of
    axial (row=pair, column=site) linear-attention blocks, and a Conv2d +
    softplus readout averaged over sites to give one non-negative distance
    estimate per sequence pair.
    """

    def __init__(
        self,
        n_blocks: int = 2,
        n_heads: int = 4,
        embed_dim: int = 16,
        n_seqs: int = 6,
    ) -> None:
        super().__init__()
        self.n_seqs = n_seqs
        self.register_buffer("seq2pair", _seq2pair(n_seqs))
        self.embedding_block = nn.Sequential(
            nn.Conv2d(22, embed_dim, kernel_size=1),
            nn.ReLU(),
        )
        self.attention_blocks = nn.ModuleList(
            [PhyloformerLayer(embed_dim, n_heads) for _ in range(n_blocks)]
        )
        self.readout = nn.Sequential(
            nn.Conv2d(embed_dim, 1, kernel_size=1),
            nn.Softplus(),
        )

    def forward(self, alignment: Tensor) -> Tensor:
        """Predict one evolutionary distance per sequence pair.

        Parameters
        ----------
        alignment : Tensor
            Shape ``(batch, 22, seq_len, n_seqs)`` one-hot(+gap) encoded
            multiple sequence alignment.
        """
        out = self.embedding_block(alignment)
        # Lift per-sequence embeddings to per-pair embeddings.
        out = torch.matmul(self.seq2pair, out.transpose(-1, -2))
        for block in self.attention_blocks:
            out = block(out)
        out = self.readout(out)
        return out.mean(dim=-1).squeeze(1)


def build_phyloformer() -> nn.Module:
    """Build a small Phyloformer pairwise-distance transformer."""
    return Phyloformer(n_blocks=2, n_heads=4, embed_dim=16, n_seqs=6).eval()


def example_input_phyloformer() -> Tensor:
    """Return a one-hot(+gap) multiple sequence alignment for Phyloformer."""
    return torch.randn(1, 22, 24, 6)


# ============================================================
# pMTnet -- TCR/pMHC binding predictor via twin autoencoders + ternary MLP
# (Lu, Zhang, Zhu et al. 2021; tianshilu/pMTnet)
# ============================================================


class PMTNetTCREncoder(nn.Module):
    """Conv-based TCR CDR3 autoencoder bottleneck (30-d embedding).

    Stands in for the pretrained Keras ``TCR_encoder`` in the reference
    ``pMTnet.py``, whose 30-d bottleneck (``layers[-12]`` of the full
    autoencoder) is used directly as the TCR embedding fed to the ternary
    classifier. Input matches the Atchley-factor CDR3 encoding shape
    ``(80, 5, 1)`` used by ``TCRMap`` in the reference.
    """

    def __init__(self, embed_dim: int = 30) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 5), padding=(1, 0)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((10, 1)),
        )
        self.bottleneck = nn.Linear(16 * 10, embed_dim)

    def forward(self, cdr3: Tensor) -> Tensor:
        """Encode a batch of Atchley-factor CDR3 tensors to 30-d embeddings."""
        x = self.conv(cdr3).flatten(1)
        return self.bottleneck(x)


class PMTNetHLAAntigenEncoder(nn.Module):
    """Conv-based HLA+antigen autoencoder bottleneck (30-d embedding).

    Stands in for the pretrained Keras ``HLA_antigen_encoder``, whose 30-d
    bottleneck (``layers[-2]``) embeds the concatenated BLOSUM50-encoded
    antigen (``(15, 21)``) and HLA pseudo-sequence (``(34, 21)``) tensors
    from ``antigenMap``/``HLAMap`` in the reference.
    """

    def __init__(self, embed_dim: int = 30) -> None:
        super().__init__()
        self.antigen_conv = nn.Sequential(
            nn.Conv1d(21, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.hla_conv = nn.Sequential(
            nn.Conv1d(21, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.bottleneck = nn.Linear(32, embed_dim)

    def forward(self, antigen: Tensor, hla: Tensor) -> Tensor:
        """Encode antigen + HLA pseudo-sequence tensors to a 30-d embedding."""
        a = self.antigen_conv(antigen.transpose(1, 2)).flatten(1)
        h = self.hla_conv(hla.transpose(1, 2)).flatten(1)
        return self.bottleneck(torch.cat([a, h], dim=1))


class PMTNet(nn.Module):
    """Twin-autoencoder + ternary MLP TCR/pMHC binding-rank predictor.

    Ports the full ``pMTnet.py`` inference pipeline: the TCR encoder and
    HLA+antigen encoder each produce a 30-d embedding, which are
    concatenated to a 60-d vector and scored by the "ternary" classifier
    (``Dense(300)-Dropout(0.2)-Dense(200)-Dense(100)-Dense(1)``), matching
    ``ternary_prediction`` in the reference exactly in topology.
    """

    def __init__(self) -> None:
        super().__init__()
        self.tcr_encoder = PMTNetTCREncoder(embed_dim=30)
        self.pmhc_encoder = PMTNetHLAAntigenEncoder(embed_dim=30)
        self.ternary = nn.Sequential(
            nn.Linear(60, 300),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, cdr3: Tensor, antigen: Tensor, hla: Tensor) -> Tensor:
        """Score a batch of TCR/antigen/HLA triples for binding strength."""
        tcr_embed = self.tcr_encoder(cdr3)
        pmhc_embed = self.pmhc_encoder(antigen, hla)
        fused = torch.cat([tcr_embed, pmhc_embed], dim=1)
        return self.ternary(fused)


def build_pmtnet() -> nn.Module:
    """Build a small pMTnet TCR/pMHC binding-rank predictor."""
    return PMTNet().eval()


def example_input_pmtnet() -> tuple[Tensor, Tensor, Tensor]:
    """Return CDR3, antigen, and HLA pseudo-sequence tensors for pMTnet."""
    cdr3 = torch.randn(2, 1, 80, 5)
    antigen = torch.randn(2, 15, 21)
    hla = torch.randn(2, 34, 21)
    return cdr3, antigen, hla


# ============================================================
# PointNovo -- PointNet-style T-Net over MS/MS peaks + LSTM decoder
# (Qiao, Liu, Ma et al. 2021; volpato30/DeepNovoV2)
# ============================================================


class PointNovoTNet(nn.Module):
    """Order-invariant T-Net over a candidate residue's expected peak set.

    Ports ``TNet`` from the DeepNovoV2/PointNovo ``model.py``: three shared
    ``Conv1d`` layers (BatchNorm + ReLU) process a per-peak feature vector
    (distance-kernel match to each of ``vocab_size * num_ion`` expected
    fragment-ion locations, plus the observed peak intensity), a global
    max-pool over the (unordered) peak axis makes the representation
    permutation-invariant to peak order -- the defining "point cloud over
    MS/MS peaks" mechanism -- and two FC layers project to the ion feature
    used by the LSTM decoder.
    """

    def __init__(self, in_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.input_bn = nn.BatchNorm1d(in_dim)
        self.conv1 = nn.Conv1d(in_dim, hidden, 1)
        self.conv2 = nn.Conv1d(hidden, hidden * 2, 1)
        self.conv3 = nn.Conv1d(hidden * 2, hidden * 4, 1)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden * 2)
        self.bn3 = nn.BatchNorm1d(hidden * 4)
        self.fc1 = nn.Linear(hidden * 4, hidden * 2)
        self.fc2 = nn.Linear(hidden * 2, hidden)
        self.bn4 = nn.BatchNorm1d(hidden * 2)
        self.bn5 = nn.BatchNorm1d(hidden)
        self.relu = nn.ReLU()

    def forward(self, peak_features: Tensor) -> Tensor:
        """Pool an unordered set of per-peak features to one ion feature.

        Parameters
        ----------
        peak_features : Tensor
            Shape ``(batch * T, in_dim, n_peaks)``.
        """
        x = self.input_bn(peak_features)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x, _ = torch.max(x, dim=2)  # global max pooling over peaks
        x = self.relu(self.bn4(self.fc1(x)))
        return self.relu(self.bn5(self.fc2(x)))


class PointNovo(nn.Module):
    """PointNet-style ion T-Net + LSTM de novo peptide sequencer.

    Ports ``DeepNovoPointNetWithLSTM``: at each decoding step, the expected
    fragment-ion locations for every candidate next amino acid are compared
    against every observed spectrum peak via an
    ``exp(-|location - peak| * scale)`` distance kernel (rather than
    rasterizing the spectrum onto a fixed image grid, distinguishing
    PointNovo from the image-based DeepNovo/DeepNovo-DIA already in this
    catalog), pooled through :class:`PointNovoTNet`, and concatenated with
    an LSTM's hidden state over the partial amino-acid sequence to predict
    the next-residue logits.
    """

    def __init__(
        self,
        vocab_size: int = 26,
        num_ion: int = 8,
        embed_dim: int = 16,
        lstm_hidden: int = 32,
        tnet_hidden: int = 16,
        distance_scale_factor: float = 0.01,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_ion = num_ion
        self.distance_scale_factor = distance_scale_factor
        self.t_net = PointNovoTNet(vocab_size * num_ion + 1, hidden=tnet_hidden)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden, num_layers=1, batch_first=True)
        self.output_layer = nn.Linear(tnet_hidden + lstm_hidden, vocab_size)

    def forward(
        self,
        location_index: Tensor,
        peaks_location: Tensor,
        peaks_intensity: Tensor,
        aa_input: Tensor,
    ) -> Tensor:
        """Predict next-amino-acid logits from candidate ion locations and peaks.

        Parameters
        ----------
        location_index : Tensor
            Shape ``(batch, T, vocab_size, num_ion)`` expected fragment-ion
            m/z locations for each candidate next amino acid.
        peaks_location : Tensor
            Shape ``(batch, n_peaks)`` observed peak m/z locations.
        peaks_intensity : Tensor
            Shape ``(batch, n_peaks)`` observed peak intensities.
        aa_input : Tensor
            Shape ``(batch, T)`` partial amino-acid sequence (long ids).
        """
        batch, t, vocab_size, num_ion = location_index.shape
        n_peaks = peaks_location.size(1)

        peaks_loc = peaks_location.view(batch, 1, n_peaks, 1).expand(-1, t, -1, -1)
        peaks_int = peaks_intensity.view(batch, 1, n_peaks, 1).expand(-1, t, -1, -1)
        loc_idx = location_index.view(batch, t, 1, vocab_size * num_ion)

        dist = torch.exp(-torch.abs((peaks_loc - loc_idx) * self.distance_scale_factor))
        feat = torch.cat([dist, peaks_int], dim=3)
        feat = feat.view(batch * t, n_peaks, vocab_size * num_ion + 1).transpose(1, 2)

        ion_feature = self.t_net(feat).view(batch, t, -1)

        aa_embedded = self.embedding(aa_input)
        lstm_out, _ = self.lstm(aa_embedded)
        fused = torch.cat([ion_feature, torch.relu(lstm_out)], dim=2)
        return self.output_layer(fused)


def build_pointnovo() -> nn.Module:
    """Build a small PointNovo ion-T-Net + LSTM de novo peptide sequencer."""
    return PointNovo(vocab_size=8, num_ion=4, embed_dim=16, lstm_hidden=16, tnet_hidden=16).eval()


def example_input_pointnovo() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return candidate ion locations, peaks, and partial sequence for PointNovo."""
    batch, t, vocab_size, num_ion, n_peaks = 2, 4, 8, 4, 12
    location_index = torch.rand(batch, t, vocab_size, num_ion) * 100
    peaks_location = torch.rand(batch, n_peaks) * 100
    peaks_intensity = torch.rand(batch, n_peaks)
    aa_input = torch.randint(0, vocab_size, (batch, t))
    return location_index, peaks_location, peaks_intensity, aa_input


# ============================================================
# PrimateAI -- dilated residual Conv1D variant-pathogenicity predictor
# (Sundaram, Gao, Padigepati et al. 2018; Illumina/PrimateAI)
# ============================================================


class PrimateAIResidualUnit(nn.Module):
    """Pre-activation residual Conv1D unit (BN-ReLU-Conv x2 + skip).

    Ports the ``residual_unit`` closure used throughout
    ``source/model.py``: two ``BatchNorm -> ReLU -> Conv1d`` stages with an
    optional additive skip connection, and a configurable dilation rate.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int, residual: bool) -> None:
        super().__init__()
        self.residual = residual
        pad = dilation * (kernel_size - 1) // 2
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad)
        self.bn2 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """Apply the pre-activation residual convolution block."""
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        return out + x if self.residual else out


class PrimateAIAuxTower(nn.Module):
    """Auxiliary secondary-structure / solvent-accessibility Conv1D tower.

    Ports ``get_ss_model``/``get_sa_model``: a shared-topology residual
    Conv1D tower (with a channel-preserving skip accumulator) over the
    summed multi-species conservation tracks, used to inject structural
    context into the main PrimateAI branches.
    """

    def __init__(self, channels: int = 16, n_stages: int = 2, units_per_stage: int = 2) -> None:
        super().__init__()
        self.in_conv = nn.Conv1d(20, channels, 1)
        self.skip_conv = nn.Conv1d(20, channels, 1)
        self.stages = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PrimateAIResidualUnit(channels, 5, 1, residual=True)
                        for _ in range(units_per_stage)
                    ]
                )
                for _ in range(n_stages)
            ]
        )
        self.dense_for_skip = nn.ModuleList(
            [nn.Conv1d(channels, channels, 1) for _ in range(n_stages)]
        )
        self.final = PrimateAIResidualUnit(channels, 1, 1, residual=True)

    def forward(self, x: Tensor) -> Tensor:
        """Encode a conservation track into a channel-matched context tensor."""
        conv = self.in_conv(x)
        skip = self.skip_conv(x)
        for units, dense in zip(self.stages, self.dense_for_skip):
            for unit in units:
                conv = unit(conv)
            skip = skip + dense(conv)
        return self.final(skip)


class PrimateAI(nn.Module):
    """Dilated residual Conv1D missense-variant pathogenicity predictor.

    Ports ``primateAI_model`` from ``source/model.py``: five parallel
    51-residue one-hot tracks (reference sequence, SNP-substituted
    sequence, and primate/mammal/other-vertebrate conservation profiles)
    are embedded with 1x1 convolutions; the summed conservation tracks
    additionally drive two auxiliary residual towers
    (:class:`PrimateAIAuxTower`, standing in for the pretrained
    secondary-structure and solvent-accessibility sub-networks) whose
    outputs are fused additively into the reference and SNP branches; the
    fused branches are merged and pushed through a deep residual Conv1D
    tower with running skip-accumulation, ending in a per-position sigmoid
    Conv1D and global max pool giving one pathogenicity score per variant.
    """

    def __init__(self, channels: int = 16, n_stages: int = 2, units_per_stage: int = 2) -> None:
        super().__init__()
        self.conv_orig = nn.Conv1d(20, channels, 1)
        self.conv_snp = nn.Conv1d(20, channels, 1)
        self.conv_primate = nn.Conv1d(20, channels, 1)
        self.conv_mammal = nn.Conv1d(20, channels, 1)
        self.conv_other = nn.Conv1d(20, channels, 1)

        self.ss_tower = PrimateAIAuxTower(channels, n_stages, units_per_stage)
        self.sa_tower = PrimateAIAuxTower(channels, n_stages, units_per_stage)

        self.orig_residual = PrimateAIResidualUnit(channels, 5, 1, residual=False)
        self.snp_residual = PrimateAIResidualUnit(channels, 5, 1, residual=False)

        self.conv_reduce = nn.Conv1d(channels * 2, channels, 5, padding=2)
        self.skip_reduce = nn.Conv1d(channels * 2, channels, 5, padding=2)

        self.stages = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PrimateAIResidualUnit(channels, 5, 1, residual=True)
                        for _ in range(units_per_stage)
                    ]
                )
                for _ in range(n_stages)
            ]
        )
        self.dense_for_skip = nn.ModuleList(
            [nn.Conv1d(channels, channels, 1) for _ in range(n_stages)]
        )
        self.final_residual = PrimateAIResidualUnit(channels, 1, 1, residual=True)
        self.final_conv = nn.Conv1d(channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        orig_seq: Tensor,
        snp_seq: Tensor,
        cons_primates: Tensor,
        cons_mammals: Tensor,
        cons_other: Tensor,
    ) -> Tensor:
        """Score a batch of missense variants for pathogenicity.

        All inputs are one-hot amino-acid tracks of shape
        ``(batch, 20, seq_len)``.
        """
        orig = self.conv_orig(orig_seq)
        snp = self.conv_snp(snp_seq)
        primate = self.conv_primate(cons_primates)
        mammal = self.conv_mammal(cons_mammals)
        other = self.conv_other(cons_other)

        cons_sum = cons_primates + cons_mammals + cons_other
        struct = self.ss_tower(cons_sum)
        solv = self.sa_tower(cons_sum)

        orig = orig + primate + mammal + other + struct + solv
        snp = snp + primate + mammal + other + struct + solv

        orig = self.orig_residual(orig)
        snp = self.snp_residual(snp)

        conv = torch.cat([orig, snp], dim=1)
        skip = torch.cat([orig, snp], dim=1)
        conv = self.conv_reduce(conv)
        skip = self.skip_reduce(skip)

        for units, dense in zip(self.stages, self.dense_for_skip):
            for unit in units:
                conv = unit(conv)
            skip = skip + dense(conv)

        conv = self.final_residual(skip)
        conv = self.sigmoid(self.final_conv(conv))
        return conv.amax(dim=2).squeeze(1)


def build_primateai() -> nn.Module:
    """Build a small PrimateAI dilated-residual pathogenicity predictor."""
    return PrimateAI(channels=16, n_stages=2, units_per_stage=2).eval()


def example_input_primateai() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return the five one-hot amino-acid tracks used by PrimateAI."""
    batch, seq_len = 2, 51
    tracks = tuple(torch.rand(batch, 20, seq_len) for _ in range(5))
    return tracks  # type: ignore[return-value]


# ============================================================
# Registry
# ============================================================

MENAGERIE_ENTRIES = [
    ("PhyloCNN", "build_phylocnn", "example_input_phylocnn", "2025", "BIO"),
    ("Phyloformer", "build_phyloformer", "example_input_phyloformer", "2022", "BIO"),
    ("pMTnet", "build_pmtnet", "example_input_pmtnet", "2021", "BIO"),
    ("PointNovo", "build_pointnovo", "example_input_pointnovo", "2021", "BIO"),
    ("PrimateAI", "build_primateai", "example_input_primateai", "2018", "BIO"),
]
