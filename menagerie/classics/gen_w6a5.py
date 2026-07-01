"""Menagerie batch w6a5: computational-genomics deep-learning classics for
Hi-C contact-map super-resolution, RNA-binding-protein sequence+structure
binding prediction, DNA-enhancer identification/strength classification,
heterogeneous-graph spatial-transcriptomics imputation, hybrid Mamba-
attention-MoE DNA language modeling, and knowledge-graph-guided
cell-type-specific gene-regulatory-network inference.

Sources checked (reference only; no cloning, no pip installs):
  - HiCPlus (cand_00733): Zhang, Chen, et al., Nature Communications 2018,
    https://github.com/zhangyan32/HiCPlus (``src/trainConvNet.py``). The
    defining mechanism: a compact three-layer 2D-CNN super-resolution
    network (SRCNN-style, no pooling, VALID/no padding so the output map
    is a smaller centered crop of the input map) maps a low-coverage
    (down-sampled) Hi-C contact-map sub-matrix directly to its
    high-resolution counterpart -- conv(9x9,8 filters)-relu ->
    conv(1x1,8 filters)-relu -> conv(5x5,1 filter)-relu, matching the
    reference's ``conv2d1_filter_size=9`` / ``conv2d2_filter_size=1`` /
    ``conv2d3_filter_size=5`` Lasagne layer stack exactly, applied to a
    single-channel Hi-C sub-matrix. Reimplemented as ``HiCPlus`` with the
    identical 9x9/1x1/5x5 no-padding conv-relu tower at a reduced
    contact-map patch size.
  - iDeepS (cand_00734): Pan & Shen, BMC Genomics 2018,
    https://github.com/xypan1232/iDeepS (``ideeps.py``'s
    ``set_cnn_model``/``get_cnn_network``). The defining mechanism: two
    parallel 1D-CNN branches -- one over one-hot RNA sequence (4 input
    channels) and one over the predicted secondary-structure profile (6
    input channels, matching the reference's ``set_cnn_model(4, 111)`` /
    ``set_cnn_model(6, 111)`` twin calls) -- each conv(filter_length=10,
    16 filters)-relu-maxpool(3)-dropout, are concatenated along the
    channel axis (``Merge([seq_model, struct_model], mode='concat')``),
    then fed through a bidirectional LSTM (``Bidirectional(LSTM(2*
    nbfilter))``) and a dense ReLU head -- i.e. "sequence-CNN +
    structure-CNN, fused, then BiLSTM" is iDeepS's namesake contribution
    over sequence-only CNN binding predictors. Reimplemented with the
    same twin-branch conv-relu-maxpool-dropout towers, channel-axis
    concatenation, BiLSTM, and dense head at reduced channel widths /
    sequence length.
  - iEnhancer-ECNN (cand_00735): Nguyen, Nguyen-Vo, Le, Do, Rahardja &
    Nguyen, BMC Genomics 2019, https://github.com/ngphubinh/enhancers
    (README's "CNN Architecture" section and ``cnn.svg``; the linked code
    archive is not browsable via the GitHub API, so the architecture is
    reimplemented from the paper's own detailed textual + diagram
    description). The defining mechanism: a 200x8-channel one-hot-encoded
    DNA input is passed through six 1D-CNN blocks with BatchNorm, with a
    1D max-pool inserted after every group of three conv blocks
    (matching "the network consists of six 1-D CNN blocks with Batch
    Normalization... for every three blocks of 1-D CNN, there is one 1-D
    Max Pooling layer"), producing 768 pooled features that feed two
    fully-connected layers (768 -> 256, ReLU then sigmoid) predicting an
    enhancer-vs-non-enhancer (or strong-vs-weak) probability; the
    reference ensembles five such CNNs trained on different data folds.
    Reimplemented as ``IEnhancerECNN``: one member of that six-block
    Conv1d+BatchNorm tower (grouped 3-and-3 with intervening max-pools)
    feeding the same two-layer sigmoid head, at reduced channel widths /
    sequence length (the training-time five-fold ensembling and
    cell-line-specific re-training are an evaluation protocol, not part
    of the per-model architecture).
  - Impeller (cand_00736): the aicb-ZhangLabs group, Bioinformatics 2024,
    https://github.com/aicb-ZhangLabs/Impeller (``model.py``, classes
    ``PathGNN``/``PathGNNLayer``). The defining mechanism: rather than
    ordinary one-hop message passing, each ``PathGNNLayer`` gathers node
    features along pre-sampled, fixed-length multi-hop **paths** grouped
    by edge type (``paths: (num_paths, num_nodes, path_length)``,
    ``path_types``), multiplies each path's per-position features by a
    **learnable path operator** (a per-position, per-channel weight
    tensor -- the reference's ``operator_type="independent"`` variant,
    one such tensor per layer per edge type) and sums along the path
    dimension, mean-pools over paths of the same type, concatenates the
    per-edge-type results, and projects back to hidden size with a
    linear layer + ReLU; stacked ``PathGNN`` layers residually blend
    each layer's output with the initial projected features via a
    learned ``alpha`` (``feats = alpha*in_feats + (1-alpha)*feats``) --
    this "learnable path operator over sampled heterogeneous paths"
    (rather than fixed-hop adjacency convolution) is Impeller's namesake
    contribution for imputing missing spatial-transcriptomic gene
    expression. Reimplemented as ``Impeller`` with the same
    linear-in -> stacked path-operator layers (independent per-layer,
    per-edge-type learnable path weights, path-mean-pool, edge-type
    concat, alpha-blended residual) -> linear-out topology, at reduced
    hidden size / path count / path length / layer count, with two edge
    types (matching the reference's default spatial + expression-kNN
    graph pair).
  - JanusDNA (cand_00737): Duan, et al., arXiv:2505.17257 / ICLR 2026
    OpenReview, https://github.com/Qihao-Duan/JanusDNA
    (``janusdna/modeling_janusdna.py``, classes
    ``JanusDNAMambaMixer``/``BiJanusDNAMambaWrapper``/
    ``JanusDNASparseMoeBlock``/``JanusDNAAttentionDecoderLayer``). The
    defining mechanism: JanusDNA interleaves **bidirectional selective-
    state-space (Mamba) blocks** with standard self-attention blocks,
    where each Mamba block runs the *same* selective-scan mixer once on
    the forward-order sequence and once on the sequence flipped along
    the length dimension (``hidden_states.flip(dims=(1,))``, matching
    ``BiJanusDNAMambaWrapper.forward``'s ``mamba_fwd`` / ``mamba_rev``
    pair, combined by addition), and every block's feed-forward stage is
    a **sparse top-k Mixture-of-Experts** router (``JanusDNASparseMoeBlock``,
    top-k gating over a bank of SwiGLU experts, matching the reference's
    ``JanusDNAMLP`` expert + linear router) -- i.e. "bidirectional Mamba
    SSM mixers interleaved with attention layers, both feeding sparse-MoE
    FFNs" is JanusDNA's namesake two-faced ("Janus") long-context DNA
    foundation-model contribution. Reimplemented as ``JanusDNA`` with the
    same selective-scan SSM recurrence (input-dependent ``dt``/``B``/``C``
    via ``x_proj``+RMSNorm, discretized ``A``/``B``, sequential scan,
    skip term ``D``, gated by ``in_proj``'s gate branch, matching
    ``JanusDNAMambaMixer.slow_forward`` exactly) run forward and on the
    time-flipped sequence and summed (bidirectional block), alternated
    with a standard ``nn.MultiheadAttention`` block (attention block),
    each followed by a top-k-routed sparse MoE FFN, at reduced hidden
    size / state size / expert count / layer count / sequence length.
  - KEGNI (cand_00738): Li, Xiao, et al., Genome Biology 2025,
    https://github.com/Lipxiao/KEGNI (``model/models.py`` class
    ``KEGNI``; ``model/KGE/KGEmodel.py`` class ``KGEmodel``;
    ``model/MAE/MAEmodel.py`` class ``MAEmodel``). The defining
    mechanism: KEGNI couples two learned embedding spaces for
    cell-type-specific gene-regulatory-network inference -- (1) a
    **GraphMAE-style masked graph autoencoder** (``MAEmodel``) that
    randomly masks a fraction of single-cell gene nodes (zeroing their
    features, replacing a subset with a learned ``enc_mask_token``,
    matching ``encoding_mask_noise``), encodes the corrupted cell-gene
    graph with a multi-layer multi-head graph-attention encoder, and
    reconstructs the masked nodes' original features through a second
    GAT decoder (trained with the reference's scaled-cosine-error loss);
    and (2) a **TransE/RotatE-style knowledge-graph embedding table**
    (``KGEmodel``) holding one learned vector per KEGG/TRRUST
    pathway-database gene and relation-type entity, looked up by integer
    id (matching ``KGEmodel.forward``'s ``kgg_embedding[[...]]``,
    ``relation_embedding[[...]]`` gather) -- combining the single-cell
    expression-graph autoencoder's gene embedding with the prior-
    knowledge KG embedding is KEGNI's namesake mechanism for
    cell-type-specific GRN inference. Reimplemented as ``KEGNI`` with a
    compact multi-head-GAT masked-autoencoder branch (mask -> GAT encode
    -> linear encoder-to-decoder bridge -> re-mask -> GAT decode,
    matching ``mask_attr_prediction``'s topology) plus a parallel
    embedding-table gather branch for KEGG-derived genes and relation
    types, at reduced node count / hidden size / layer count.
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ============================================================
# HiCPlus -- three-layer no-padding 2D-CNN super-resolution for
# Hi-C contact maps (zhangyan32/HiCPlus)
# ============================================================


class HiCPlus(nn.Module):
    """Compact three-layer CNN super-resolving low-coverage Hi-C maps.

    Ports ``src/trainConvNet.py``'s Lasagne ``NeuralNet``: a
    9x9-conv-relu -> 1x1-conv-relu -> 5x5-conv-relu tower, all with
    *no* padding (matching Lasagne's default ``pad=0``/"valid" mode), so
    the output contact-map patch is a smaller centered crop of the input
    patch -- exactly the reference's ``conv2d1``/``conv2d2``/``conv2d3``
    filter-size sequence (9, 1, 5) applied to a single-channel Hi-C
    sub-matrix.
    """

    def __init__(self, n_filters1: int = 8, n_filters2: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, n_filters1, kernel_size=9)
        self.conv2 = nn.Conv2d(n_filters1, n_filters2, kernel_size=1)
        self.conv3 = nn.Conv2d(n_filters2, 1, kernel_size=5)

    def forward(self, low_res_patch: Tensor) -> Tensor:
        """Super-resolve a low-coverage Hi-C contact-map patch.

        Parameters
        ----------
        low_res_patch : Tensor
            Shape ``(batch, 1, size, size)`` down-sampled Hi-C sub-matrix.
        """
        x = torch.relu(self.conv1(low_res_patch))
        x = torch.relu(self.conv2(x))
        return torch.relu(self.conv3(x))


def build_hicplus() -> nn.Module:
    """Build a small HiCPlus three-layer no-padding CNN."""
    return HiCPlus(n_filters1=8, n_filters2=8).eval()


def example_input_hicplus() -> Tensor:
    """Return a batch of low-coverage Hi-C contact-map patches."""
    return torch.rand(2, 1, 40, 40)


# ============================================================
# iDeepS -- twin sequence/structure 1D-CNN branches concatenated
# and fed through a BiLSTM for RBP binding prediction
# (xypan1232/iDeepS)
# ============================================================


class _IDeepSBranch(nn.Module):
    """Conv1d(filter_length=10) -> ReLU -> MaxPool(3) -> Dropout."""

    def __init__(self, in_channels: int, n_filters: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, n_filters, kernel_size=10)
        self.pool = nn.MaxPool1d(3)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        return self.dropout(x)


class IDeepS(nn.Module):
    """Twin sequence/structure CNN branches, concatenated, then BiLSTM.

    Ports ``ideeps.py``'s ``get_cnn_network``: two parallel
    ``set_cnn_model``-style branches -- one over one-hot RNA sequence (4
    channels), one over predicted secondary-structure profile (6
    channels) -- are concatenated along the channel axis after their
    conv-relu-maxpool-dropout towers, then a bidirectional LSTM and a
    dense ReLU head produce the pooled RBP-binding representation.
    """

    def __init__(
        self,
        seq_len: int = 111,
        n_filters: int = 16,
        lstm_hidden: int | None = None,
        n_classes: int = 1,
    ) -> None:
        super().__init__()
        self.seq_branch = _IDeepSBranch(4, n_filters)
        self.struct_branch = _IDeepSBranch(6, n_filters)
        lstm_hidden = lstm_hidden if lstm_hidden is not None else 2 * n_filters
        self.blstm = nn.LSTM(2 * n_filters, lstm_hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(2 * lstm_hidden, n_filters * 2)
        self.out = nn.Linear(n_filters * 2, n_classes)

    def forward(self, seq_onehot: Tensor, struct_onehot: Tensor) -> Tensor:
        """Predict RBP-binding probability from sequence + structure input.

        Parameters
        ----------
        seq_onehot : Tensor
            Shape ``(batch, 4, seq_len)`` one-hot RNA sequence.
        struct_onehot : Tensor
            Shape ``(batch, 6, seq_len)`` predicted secondary-structure
            profile (one channel per RNAshapes structural context).
        """
        seq_feat = self.seq_branch(seq_onehot)
        struct_feat = self.struct_branch(struct_onehot)
        fused = torch.cat([seq_feat, struct_feat], dim=1)
        fused = fused.transpose(1, 2)
        rnn_out, _ = self.blstm(fused)
        pooled = rnn_out.mean(dim=1)
        pooled = self.dropout(pooled)
        hidden = torch.relu(self.dense(pooled))
        return torch.sigmoid(self.out(hidden))


def build_ideeps() -> nn.Module:
    """Build a small iDeepS twin-branch CNN + BiLSTM RBP-binding model."""
    return IDeepS(seq_len=111, n_filters=16, n_classes=1).eval()


def example_input_ideeps() -> tuple[Tensor, Tensor]:
    """Return (one-hot sequence, structure-profile) input for iDeepS."""
    seq_onehot = torch.zeros(2, 4, 111)
    seq_onehot[
        torch.arange(2).repeat_interleave(111),
        torch.randint(0, 4, (222,)),
        torch.arange(111).repeat(2),
    ] = 1.0
    struct_onehot = torch.rand(2, 6, 111)
    return seq_onehot, struct_onehot


# ============================================================
# iEnhancer-ECNN -- six-block Conv1d+BatchNorm tower (grouped
# 3-and-3 with intervening max-pools) + two-layer sigmoid head
# for enhancer identification/strength classification
# (ngphubinh/enhancers)
# ============================================================


class _IEnhancerConvBlock(nn.Module):
    """Conv1d -> BatchNorm1d -> ReLU, same-length padding."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return torch.relu(self.bn(self.conv(x)))


class IEnhancerECNN(nn.Module):
    """One member of the iEnhancer-ECNN five-fold CNN ensemble.

    Ports the README's "CNN Architecture" description: a
    ``(seq_len, 8)``-channel one-hot-encoded DNA input passes through six
    1D-CNN blocks with BatchNorm, with a 1D max-pool inserted after every
    group of three conv blocks, giving 768 pooled features that feed two
    fully-connected layers (ReLU then sigmoid) predicting an
    enhancer-vs-non-enhancer (or strength) probability. The reference
    trains and ensembles five such CNNs across data folds; this class is
    one ensemble member's architecture.
    """

    def __init__(self, seq_len: int = 200, base_channels: int = 32, fc_hidden: int = 64) -> None:
        super().__init__()
        c = base_channels
        self.block1 = _IEnhancerConvBlock(8, c)
        self.block2 = _IEnhancerConvBlock(c, c)
        self.block3 = _IEnhancerConvBlock(c, c)
        self.pool1 = nn.MaxPool1d(2)
        self.block4 = _IEnhancerConvBlock(c, 2 * c)
        self.block5 = _IEnhancerConvBlock(2 * c, 2 * c)
        self.block6 = _IEnhancerConvBlock(2 * c, 2 * c)
        self.pool2 = nn.AdaptiveMaxPool1d(4)
        flat_dim = 2 * c * 4
        self.fc1 = nn.Linear(flat_dim, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, 1)

    def forward(self, dna_onehot: Tensor) -> Tensor:
        """Predict enhancer probability from one-hot DNA (+ context channels).

        Parameters
        ----------
        dna_onehot : Tensor
            Shape ``(batch, 8, seq_len)`` one-hot-encoded DNA sequence
            (4 base channels plus the reference's auxiliary structural /
            dinucleotide context channels, matching the README's
            "200 x 8 matrix" input).
        """
        x = self.block1(dna_onehot)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool1(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.pool2(x)
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


def build_ienhancer_ecnn() -> nn.Module:
    """Build one small iEnhancer-ECNN six-block CNN ensemble member."""
    return IEnhancerECNN(seq_len=200, base_channels=32, fc_hidden=64).eval()


def example_input_ienhancer_ecnn() -> Tensor:
    """Return a batch of one-hot-encoded DNA sequences for iEnhancer-ECNN."""
    return torch.rand(2, 8, 200)


# ============================================================
# Impeller -- learnable-path-operator heterogeneous GNN for
# spatial-transcriptomics imputation (aicb-ZhangLabs/Impeller)
# ============================================================


class _ImpellerPathLayer(nn.Module):
    """One PathGNN layer: per-edge-type learnable path operator + mean-pool."""

    def __init__(self, hidden_dim: int, path_length: int, num_edge_types: int) -> None:
        super().__init__()
        self.num_edge_types = num_edge_types
        self.path_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(1, path_length, hidden_dim)) for _ in range(num_edge_types)]
        )
        for w in self.path_weights:
            nn.init.xavier_normal_(w, gain=1.414)
        self.fc = nn.Linear(num_edge_types * hidden_dim, hidden_dim, bias=False)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

    def forward(self, feats: Tensor, paths_by_type: list[Tensor]) -> Tensor:
        """Apply the learnable path operator to each edge type's path bank.

        Parameters
        ----------
        feats : Tensor
            Shape ``(num_nodes, hidden_dim)`` current node features.
        paths_by_type : list[Tensor]
            One ``(num_paths, num_nodes, path_length)`` integer node-index
            tensor per edge type (pre-sampled multi-hop paths).
        """
        results = []
        for edge_type, paths in enumerate(paths_by_type):
            path_feats = feats[paths]  # (num_paths, num_nodes, path_length, d)
            weighted = (path_feats * self.path_weights[edge_type]).sum(dim=2)
            results.append(weighted.mean(dim=0))  # (num_nodes, d)
        fout = torch.cat(results, dim=1) if self.num_edge_types > 1 else results[0]
        return torch.relu(self.fc(fout))


class Impeller(nn.Module):
    """Learnable-path-operator heterogeneous GNN for spatial imputation.

    Ports ``model.py``'s ``PathGNN``/``PathGNNLayer`` (``operator_type=
    "independent"``): node features are linearly projected, then each
    stacked layer gathers features along pre-sampled fixed-length paths
    per edge type, reweights each path position by a learnable
    per-layer, per-edge-type operator, mean-pools over paths, concatenates
    across edge types, and the result is residually blended with the
    initial projection via a learned ``alpha``; a final linear head
    projects back to the gene-expression output dimension.
    """

    def __init__(
        self,
        in_dim: int = 32,
        hidden_dim: int = 24,
        out_dim: int = 32,
        num_layers: int = 2,
        num_edge_types: int = 2,
        path_length: int = 3,
        alpha: float = 0.3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.dropout = dropout
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        nn.init.xavier_normal_(self.fc_in.weight, gain=1.414)
        self.layers = nn.ModuleList(
            [_ImpellerPathLayer(hidden_dim, path_length, num_edge_types) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        nn.init.xavier_normal_(self.fc_out.weight, gain=1.414)

    def forward(self, x: Tensor, paths_by_type: list[Tensor]) -> Tensor:
        """Impute gene expression via the stacked learnable-path-operator GNN.

        Parameters
        ----------
        x : Tensor
            Shape ``(num_nodes, in_dim)`` spot/cell gene-expression matrix.
        paths_by_type : list[Tensor]
            One ``(num_paths, num_nodes, path_length)`` integer node-index
            tensor per edge type.
        """
        in_feats = F.dropout(x, p=self.dropout, training=self.training)
        in_feats = torch.relu(self.fc_in(in_feats))
        feats = in_feats
        for layer in self.layers:
            feats = layer(feats, paths_by_type)
            feats = self.alpha * in_feats + (1 - self.alpha) * feats
        feats = F.dropout(feats, p=self.dropout, training=self.training)
        return torch.relu(self.fc_out(feats))


def build_impeller() -> nn.Module:
    """Build a small Impeller learnable-path-operator GNN imputer."""
    return Impeller(
        in_dim=32, hidden_dim=24, out_dim=32, num_layers=2, num_edge_types=2, path_length=3
    ).eval()


def example_input_impeller() -> tuple[Tensor, list[Tensor]]:
    """Return (node features, per-edge-type path banks) for Impeller."""
    num_nodes, num_paths, path_length = 20, 6, 3
    x = torch.randn(num_nodes, 32)
    paths_by_type = [
        torch.randint(0, num_nodes, (num_paths, num_nodes, path_length)) for _ in range(2)
    ]
    return x, paths_by_type


# ============================================================
# JanusDNA -- bidirectional selective-state-space (Mamba)
# mixers interleaved with attention, both feeding sparse-MoE
# FFNs, for long-context DNA language modeling
# (Qihao-Duan/JanusDNA)
# ============================================================


class _JanusMambaMixer(nn.Module):
    """Selective-scan SSM mixer (ports ``JanusDNAMambaMixer.slow_forward``)."""

    def __init__(self, hidden_size: int, state_size: int, dt_rank: int, conv_kernel: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.dt_rank = dt_rank
        self.intermediate_size = hidden_size
        self.conv1d = nn.Conv1d(
            self.intermediate_size,
            self.intermediate_size,
            kernel_size=conv_kernel,
            groups=self.intermediate_size,
            padding=conv_kernel - 1,
        )
        self.in_proj = nn.Linear(hidden_size, self.intermediate_size * 2)
        self.x_proj = nn.Linear(self.intermediate_size, dt_rank + state_size * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.intermediate_size)
        A = torch.arange(1, state_size + 1, dtype=torch.float32).expand(self.intermediate_size, -1)
        self.A_log = nn.Parameter(torch.log(A.contiguous()))
        self.D = nn.Parameter(torch.ones(self.intermediate_size))
        self.out_proj = nn.Linear(self.intermediate_size, hidden_size)
        self.dt_norm = nn.RMSNorm(dt_rank)
        self.b_norm = nn.RMSNorm(state_size)
        self.c_norm = nn.RMSNorm(state_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Run the selective-scan SSM recurrence over one sequence direction."""
        batch, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype
        projected = self.in_proj(hidden_states).transpose(1, 2)
        x, gate = projected.chunk(2, dim=1)
        x = F.silu(self.conv1d(x)[..., :seq_len])

        ssm_params = self.x_proj(x.transpose(1, 2))
        dt, b, c = torch.split(ssm_params, [self.dt_rank, self.state_size, self.state_size], dim=-1)
        dt = self.dt_norm(dt)
        b = self.b_norm(b)
        c = self.c_norm(c)
        discrete_dt = F.softplus(self.dt_proj(dt)).transpose(1, 2)

        a = -torch.exp(self.A_log.float())
        discrete_a = torch.exp(a[None, :, None, :] * discrete_dt[:, :, :, None])
        discrete_b = discrete_dt[:, :, :, None] * b[:, None, :, :].float()
        delta_bu = discrete_b * x[:, :, :, None].float()

        ssm_state = torch.zeros(batch, self.intermediate_size, self.state_size, dtype=dtype)
        outputs = []
        for t in range(seq_len):
            ssm_state = discrete_a[:, :, t, :] * ssm_state + delta_bu[:, :, t, :]
            outputs.append(torch.matmul(ssm_state.to(dtype), c[:, t, :].unsqueeze(-1))[:, :, 0])
        scan_out = torch.stack(outputs, dim=-1)
        scan_out = scan_out + x * self.D[None, :, None]
        scan_out = scan_out * F.silu(gate)
        return self.out_proj(scan_out.transpose(1, 2))


class _JanusMoE(nn.Module):
    """Top-k sparse Mixture-of-Experts SwiGLU FFN (ports ``JanusDNASparseMoeBlock``)."""

    def __init__(self, hidden_size: int, ffn_size: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.gate_proj = nn.ModuleList(
            [nn.Linear(hidden_size, ffn_size, bias=False) for _ in range(num_experts)]
        )
        self.up_proj = nn.ModuleList(
            [nn.Linear(hidden_size, ffn_size, bias=False) for _ in range(num_experts)]
        )
        self.down_proj = nn.ModuleList(
            [nn.Linear(ffn_size, hidden_size, bias=False) for _ in range(num_experts)]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Route each token to its top-k experts and combine (dense, trace-safe form)."""
        batch, seq_len, hidden = x.shape
        flat = x.reshape(-1, hidden)
        logits = self.router(flat)
        weights = F.softmax(logits, dim=-1, dtype=torch.float32).to(flat.dtype)
        top_weights, top_idx = torch.topk(weights, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(flat)
        for expert_idx in range(self.num_experts):
            expert_out = self.down_proj[expert_idx](
                F.silu(self.gate_proj[expert_idx](flat)) * self.up_proj[expert_idx](flat)
            )
            gate = (top_idx == expert_idx).to(flat.dtype) * top_weights
            gate = gate.sum(dim=-1, keepdim=True)
            out = out + expert_out * gate
        return out.reshape(batch, seq_len, hidden)


class _JanusMambaBlock(nn.Module):
    """Bidirectional Mamba mixer + sparse-MoE FFN, both pre-normed and residual."""

    def __init__(
        self,
        hidden_size: int,
        state_size: int,
        dt_rank: int,
        conv_kernel: int,
        ffn_size: int,
        num_experts: int,
        top_k: int,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size)
        self.mamba_fwd = _JanusMambaMixer(hidden_size, state_size, dt_rank, conv_kernel)
        self.mamba_rev = _JanusMambaMixer(hidden_size, state_size, dt_rank, conv_kernel)
        self.norm2 = nn.RMSNorm(hidden_size)
        self.ffn = _JanusMoE(hidden_size, ffn_size, num_experts, top_k)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        h = self.norm1(x)
        fwd = self.mamba_fwd(h)
        rev = self.mamba_rev(h.flip(dims=(1,))).flip(dims=(1,))
        x = residual + (fwd + rev)
        residual = x
        x = residual + self.ffn(self.norm2(x))
        return x


class _JanusAttnBlock(nn.Module):
    """Standard bidirectional self-attention block + sparse-MoE FFN."""

    def __init__(
        self, hidden_size: int, n_heads: int, ffn_size: int, num_experts: int, top_k: int
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)
        self.norm2 = nn.RMSNorm(hidden_size)
        self.ffn = _JanusMoE(hidden_size, ffn_size, num_experts, top_k)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = residual + attn_out
        residual = x
        x = residual + self.ffn(self.norm2(x))
        return x


class JanusDNA(nn.Module):
    """Bidirectional Mamba mixers interleaved with attention, both MoE-fed.

    Ports ``janusdna/modeling_janusdna.py``: ``BiJanusDNAMambaWrapper``
    (a forward + sequence-flipped selective-scan SSM pair, summed) and
    ``JanusDNAAttentionDecoderLayer`` (standard self-attention) blocks
    alternate, each followed by ``JanusDNASparseMoeBlock``'s top-k-routed
    sparse Mixture-of-Experts SwiGLU FFN.
    """

    def __init__(
        self,
        vocab_size: int = 16,
        hidden_size: int = 48,
        state_size: int = 8,
        dt_rank: int = 4,
        conv_kernel: int = 4,
        n_heads: int = 4,
        ffn_size: int = 96,
        num_experts: int = 4,
        top_k: int = 2,
        n_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            if i % 2 == 0:
                self.blocks.append(
                    _JanusMambaBlock(
                        hidden_size, state_size, dt_rank, conv_kernel, ffn_size, num_experts, top_k
                    )
                )
            else:
                self.blocks.append(
                    _JanusAttnBlock(hidden_size, n_heads, ffn_size, num_experts, top_k)
                )
        self.final_norm = nn.RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Predict per-position next-nucleotide-token logits.

        Parameters
        ----------
        token_ids : Tensor
            Shape ``(batch, seq_len)`` integer DNA token ids.
        """
        x = self.embed(token_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)


def build_janusdna() -> nn.Module:
    """Build a small JanusDNA bidirectional-Mamba/attention MoE model."""
    return JanusDNA(
        vocab_size=16,
        hidden_size=48,
        state_size=8,
        dt_rank=4,
        conv_kernel=4,
        n_heads=4,
        ffn_size=96,
        num_experts=4,
        top_k=2,
        n_blocks=2,
    ).eval()


def example_input_janusdna() -> Tensor:
    """Return a batch of integer DNA token ids for JanusDNA."""
    return torch.randint(0, 16, (2, 24))


# ============================================================
# KEGNI -- GraphMAE-style masked-graph-autoencoder gene
# embedding + TransE-style KEGG/TRRUST knowledge-graph
# embedding lookup for GRN inference (Lipxiao/KEGNI)
# ============================================================


class _KEGNIGATLayer(nn.Module):
    """One dense multi-head graph-attention layer (additive-attention GAT)."""

    def __init__(self, in_dim: int, out_dim: int, n_heads: int, concat: bool = True) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.concat = concat
        self.weight = nn.Parameter(torch.empty(n_heads, in_dim, out_dim))
        self.attn_src = nn.Parameter(torch.empty(n_heads, out_dim))
        self.attn_dst = nn.Parameter(torch.empty(n_heads, out_dim))
        nn.init.xavier_normal_(self.weight, gain=1.414)
        nn.init.xavier_normal_(self.attn_src.unsqueeze(0), gain=1.414)
        nn.init.xavier_normal_(self.attn_dst.unsqueeze(0), gain=1.414)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """Apply a masked-softmax multi-head graph-attention step.

        Parameters
        ----------
        x : Tensor
            Shape ``(num_nodes, in_dim)`` node features.
        adj : Tensor
            Shape ``(num_nodes, num_nodes)`` dense adjacency mask.
        """
        h = torch.einsum("nd,hde->hne", x, self.weight)  # (heads, nodes, out_dim)
        src_score = (h * self.attn_src.unsqueeze(1)).sum(-1)  # (heads, nodes)
        dst_score = (h * self.attn_dst.unsqueeze(1)).sum(-1)  # (heads, nodes)
        e = F.leaky_relu(src_score.unsqueeze(2) + dst_score.unsqueeze(1), 0.2)
        mask = adj.unsqueeze(0) > 0
        e = e.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(e, dim=2)
        out = torch.einsum("hij,hjd->hid", attn, h)  # (heads, nodes, out_dim)
        if self.concat:
            return F.elu(out.transpose(0, 1).reshape(x.shape[0], self.n_heads * self.out_dim))
        return F.elu(out.mean(dim=0))


class KEGNI(nn.Module):
    """GraphMAE-style masked GAT autoencoder + KG-embedding lookup branch.

    Ports ``model/models.py``'s ``KEGNI`` wrapper of ``MAEmodel`` (a
    mask-node-features -> multi-head-GAT-encode -> linear bridge ->
    re-mask -> multi-head-GAT-decode attribute-reconstruction
    autoencoder, matching ``mask_attr_prediction``) and ``KGEmodel`` (a
    TransE/RotatE-style embedding table over KEGG/TRRUST pathway genes
    and relation types, looked up by integer id).
    """

    def __init__(
        self,
        n_genes: int = 24,
        in_dim: int = 16,
        hidden_dim: int = 16,
        n_heads: int = 2,
        n_kg_entities: int = 30,
        n_relations: int = 5,
        kg_dim: int = 16,
        mask_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.mask_rate = mask_rate
        self.mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.encoder = _KEGNIGATLayer(in_dim, hidden_dim // n_heads, n_heads, concat=True)
        self.encoder_to_decoder = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.decoder = _KEGNIGATLayer(hidden_dim, in_dim, 1, concat=False)

        self.kgg_embedding = nn.Parameter(torch.empty(n_kg_entities, kg_dim))
        self.relation_embedding = nn.Parameter(torch.empty(n_relations, kg_dim))
        bound = 1.0 / math.sqrt(kg_dim)
        nn.init.uniform_(self.kgg_embedding, -bound, bound)
        nn.init.uniform_(self.relation_embedding, -bound, bound)

    def forward(
        self, x: Tensor, adj: Tensor, mask_nodes: Tensor, kgg_ids: Tensor, relation_ids: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Reconstruct masked gene features and gather KG embeddings.

        Parameters
        ----------
        x : Tensor
            Shape ``(num_genes, in_dim)`` single-cell gene feature matrix.
        adj : Tensor
            Shape ``(num_genes, num_genes)`` dense cell-gene graph adjacency.
        mask_nodes : Tensor
            Shape ``(num_masked,)`` integer indices of nodes to mask.
        kgg_ids : Tensor
            Shape ``(num_kg_genes,)`` integer indices into the KEGG/TRRUST
            gene embedding table.
        relation_ids : Tensor
            Shape ``(num_kg_rels,)`` integer indices into the relation
            embedding table.
        """
        masked_x = x.index_copy(0, mask_nodes, self.mask_token.expand(mask_nodes.shape[0], -1))
        enc_rep = self.encoder(masked_x, adj)
        rep = self.encoder_to_decoder(enc_rep)
        rep = rep.index_copy(0, mask_nodes, torch.zeros(mask_nodes.shape[0], rep.shape[1]))
        recon = self.decoder(rep, adj)

        kgg_embedding = self.kgg_embedding[kgg_ids]
        relation_embedding = self.relation_embedding[relation_ids]
        return cast(Tensor, recon), kgg_embedding, relation_embedding


def build_kegni() -> nn.Module:
    """Build a small KEGNI masked-GAT autoencoder + KG-embedding model."""
    return KEGNI(
        n_genes=24,
        in_dim=16,
        hidden_dim=16,
        n_heads=2,
        n_kg_entities=30,
        n_relations=5,
        kg_dim=16,
        mask_rate=0.3,
    ).eval()


def example_input_kegni() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return (features, adjacency, mask_nodes, kgg_ids, relation_ids) for KEGNI."""
    n_genes = 24
    x = torch.randn(n_genes, 16)
    adj = (torch.rand(n_genes, n_genes) > 0.5).float()
    adj = adj + torch.eye(n_genes)
    mask_nodes = torch.randperm(n_genes)[: int(0.3 * n_genes)]
    kgg_ids = torch.randint(0, 30, (8,))
    relation_ids = torch.randint(0, 5, (4,))
    return x, adj, mask_nodes, kgg_ids, relation_ids


MENAGERIE_ENTRIES = [
    ("HiCPlus", "build_hicplus", "example_input_hicplus", "2018", "BIO"),
    ("iDeepS", "build_ideeps", "example_input_ideeps", "2018", "BIO"),
    ("iEnhancer-ECNN", "build_ienhancer_ecnn", "example_input_ienhancer_ecnn", "2019", "BIO"),
    ("Impeller", "build_impeller", "example_input_impeller", "2024", "BIO"),
    ("JanusDNA", "build_janusdna", "example_input_janusdna", "2025", "BIO"),
    ("KEGNI", "build_kegni", "example_input_kegni", "2025", "BIO"),
]
