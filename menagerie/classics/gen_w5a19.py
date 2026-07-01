"""Bioinformatics / genomics classics (batch w5a19).

Sources checked (repo/paper architecture; no clone, no pip install --
reimplemented from scratch in base-env torch):

- CRISPRon: Xiang, Corsi, Anthon, Qu, Pan, Liang, Han, Dong, Liu, Ma, Wang,
  Zheng, Lin, Yang, Song, Vejnar, Zhang, Bhatt, Lauschke, Kong, Bagi,
  Vinther & Bak, Nature Communications 2021,
  https://www.nature.com/articles/s41467-021-23576-0. Official repo
  https://github.com/RTH-tools/crispron (``bin/DeepCRISPRon_train.py``,
  Keras functional model built at lines ~185-260). CRISPRon predicts SpCas9
  on-target editing efficiency from a 30-nt one-hot target sequence *plus*
  a scalar RNA-DNA hybridization free-energy feature (``deltaGb``): three
  parallel ``Conv1D`` branches with different kernel widths (3, 5, 7;
  channel widths 100/70/40) each followed by dropout, average-pooling and
  flatten (a multi-scale motif-detector bank scanning the same one-hot
  input at three different receptive-field widths -- the paper's
  namesake departure from earlier single-kernel CRISPR CNNs), concatenated
  and passed through a dense layer; the ``deltaGb`` scalar is then
  concatenated in *raw* (unprocessed) alongside the pooled conv features
  before two more dense layers and a scalar regression head (exactly the
  official ``for_dense1.append(input_g)`` -- the free-energy feature
  bypasses the conv trunk entirely and is fused post-hoc). Reimplemented
  here as ``CRISPRon`` with the three parallel multi-width ``Conv1d``
  branches, average pooling, flatten+concat, dense fusion, raw-``deltaGb``
  concat, and the two-stage dense regression head.

- DeepARG: Arango-Argoty, Garner, Pruden, Heath, Vikesland & Zhang,
  Microbiome 2018, https://doi.org/10.1186/s40168-018-0401-z. Official repo
  https://github.com/gaarangoa/deeparg (modern PyTorch/Transformers runtime,
  ``deeparg/modern/modeling_deeparg.py`` -- class
  ``DeepARGForSequenceClassification``, and
  ``deeparg/modern/configuration_deeparg.py`` for the default
  hyperparameters). DeepARG classifies antibiotic-resistance genes (ARGs)
  directly from a fixed-length vector of per-reference-gene alignment
  bit-scores (a short-read or full gene is aligned against every entry in
  the curated ARG database; the resulting bit-score row, min-max normalized
  per batch, *is* the feature vector -- so the "distinctive mechanism" is
  the alignment-bitscore-as-feature representation feeding a plain deep
  MLP, not a sequence encoder) through a deep fully-connected stack with
  default hidden sizes ``[2000, 1000, 500, 100]``, ReLU activations and
  dropout after every hidden layer except the last, ending in a linear
  classifier head over the ARG category labels. Reimplemented here as
  ``DeepARG`` with the same 4-hidden-layer MLP trunk (dropout after all but
  the final hidden layer, matching ``dropout_after_layers = len(hidden)-1``
  default) and linear classifier head.

- DeepC: Schwessinger, Gosden, Downes, Brown, Oudelaar, Telenius, Teh,
  Lunter & Hughes, Nature Methods 2020,
  https://www.nature.com/articles/s41592-020-0960-3. Official repo
  https://github.com/Hughes-Genome-Group/deepC
  (``tensorflow1_version/deepCregr.py``, functions ``convolutional_layer``,
  ``dilated_layer``, ``inference``; TF1 codebase, used here purely as the
  architecture reference, not executed). DeepC predicts Hi-C chromatin
  interaction (contact-map) tracks from raw one-hot DNA sequence using a
  WaveNet-style *gated dilated* convolutional stack: an upstream plain
  Conv1d+ReLU+max-pool feature-extraction stack narrows a long one-hot
  input, then a cascade of dilated-conv layers with **exponentially
  increasing dilation** apply two independent same-width convolutions per
  layer (a "filter" branch and a "gate" branch) combined as
  ``tanh(filter) * sigmoid(gate)`` (the official ``dilated_gated =
  tf.tanh(dilated) * tf.sigmoid(gated)``), then added back to the layer
  input as a residual connection through a 1x1 "dense" projection
  (``dense_residual``) -- exactly WaveNet's gated-activation-plus-residual
  block, here driving a genomic long-range regression head instead of
  audio. Reimplemented here as ``DeepC`` with a small conv+pool front-end
  followed by a stack of ``GatedDilatedConv1d`` blocks (dilation doubling
  each layer, tanh*sigmoid gate, 1x1 residual projection) and a final dense
  regression head.

- DeepCas9 (Xue, Tang, Cheng & Luo naming used by the ``lje00006/DeepCas9``
  repo; not to be confused with Kim et al.'s "DeepSpCas9"), Xue, Liu, Cai,
  Yang, Zheng, Dong & Luo, J. Chem. Inf. Model. 2019 (per the repo's own
  citation). Official repo https://github.com/lje00006/DeepCas9
  (R + mxnet driver: ``main.R``, ``encodeOntargetSeq.R``,
  ``DeepCas9_scores.R`` -- no architecture-defining code ships in the repo,
  only pretrained mxnet binary weights, so the network topology itself is
  reimplemented from the repo's documented I/O contract). The documented
  mechanism: a 30-nt spCas9 target (4 bp upstream + 20 bp protospacer + 3 bp
  PAM + 3 bp downstream) is one-hot encoded into a ``4 x 30`` matrix, fed
  through a convolutional network, and *three independently trained
  cell-line-specific sub-models* (HEK293T, HL60, mEL4) each score the same
  input; the reported ``DeepCas9_score`` is a fixed-weight ensemble of the
  three (``0.5*HL60 + 0.3*T293 + 0.2*mEL4``, per ``DeepCas9_scores.R``) --
  the distinctive mechanism is this per-cell-line sub-network ensemble with
  fixed mixing weights, not a generic single CNN. Reimplemented here as
  ``DeepCas9`` with a shared 2D-conv-over-the-one-hot-image trunk
  (``Conv2d`` over the ``(4, 30)`` grid, matching the repo's
  ``dim(test_onehot) <- c(4, 30, 1, N)`` mxnet input layout) branching into
  three independent small conv+dense heads (one per cell line) whose scalar
  outputs are combined with the exact published fixed weights.

- DeepCCI: Wang, Sun, Sun, Zhang, Ma, Jiang & Jiang, Bioinformatics 2023,
  https://doi.org/10.1093/bioinformatics/btad596. Official repo
  https://github.com/JiangBioLab/DeepCCI (``Cluster_model/Cluster.py``
  class ``ClusterModel``, ``Cluster_model/GNN.py`` class ``GNNLayer``).
  DeepCCI's unsupervised clustering branch follows the SDCN
  (Structural-Deep-Clustering-Network) pattern: a stacked autoencoder (AE)
  learns per-cell latent embeddings while a parallel stack of graph
  convolutional layers (``GNNLayer``: ``adj @ (X @ W)`` with a symmetric
  cell-cell KNN graph) propagates neighborhood structure -- at *every*
  encoder layer the GCN hidden state is injected additively into the
  AE hidden state before the next AE layer (``enc_h2 = relu(enc_2(enc_h1 +
  h1))``), the namesake cross-fusion between the two branches. The learned
  bottleneck ``z`` then drives a Student's-t soft cluster assignment
  (``q_ij = (1 + ||z_i - mu_j||^2)^{-1}``, normalized) exactly as in DEC/
  SDCN, alongside a softmax GCN classification head and an AE
  reconstruction. Reimplemented here as ``DeepCCIClusterModel`` with the
  shared AE+GCN dual-fusion trunk (AE hidden states summed with the
  parallel GCN hidden states at every layer), the Student's-t cluster
  layer, and the softmax GCN branch, using a dense adjacency matrix (no
  ``torch.spmm``) for traceability.

- DeepCDR: Liu, Peng, Sun, Yang & Zhao, Bioinformatics 2020,
  https://academic.oup.com/bioinformatics/article/36/Supplement_2/i911/6055929.
  Official repo https://github.com/kimmo1019/DeepCDR (``prog/model.py``
  class ``KerasMultiSourceGCNModel.createMaster``, ``prog/layers/graph.py``
  class ``GraphConv``). DeepCDR predicts cancer drug response from four
  parallel encoders fused into one regression head: (1) a drug molecular
  graph encoder using a *uniform-aggregation* graph conv
  (``GraphConv``: ``h_i = sigma(sum_j A_ij (h_j W + b) / sum_j A_ij)``,
  i.e. mean-neighbor aggregation over the per-atom feature matrix and
  binary adjacency, stacked for several steps) followed by global max
  pooling over atoms; (2) a genomic point-mutation encoder using two
  ``Conv2d`` + max-pool stages over a ``(1, mutation_dim)`` one-hot-like
  strip (treating the mutation vector as a 1-D "image" with a dummy height
  axis, the official ``mutation_input`` shape ``(1, mutation_dim, 1)``);
  (3) a gene-expression MLP encoder; (4) a DNA-methylation MLP encoder --
  the four pooled/encoded vectors are concatenated, projected through a
  dense layer, then reshaped back into a synthetic 1x1 "image" and passed
  through a second small ``Conv2d`` stack (kernel width shrinking
  150->5->5) before the final scalar IC50 regression head -- the "graph
  conv on drug molecular graph + multi-omics encoder" fusion is the
  namesake distinctive mechanism. Reimplemented here as ``DeepCDR`` with
  the mean-aggregation ``UniformGraphConv`` drug-graph branch (global max
  pool), the mutation ``Conv2d`` branch, the gene-expression and
  methylation MLP branches, concatenation + dense fusion, and the
  post-fusion ``Conv2d`` stack feeding the scalar regression head.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# CRISPRon
# ---------------------------------------------------------------------------


class CRISPRon(nn.Module):
    """CRISPRon: multi-width Conv1d motif bank + raw deltaG fusion for
    spCas9 on-target efficiency regression.

    Three parallel ``Conv1d`` branches (kernel widths 3/5/7) scan the same
    30-nt one-hot target sequence, are pooled and concatenated, fused with a
    dense layer, then concatenated with the raw (unprocessed) RNA-DNA
    hybridization free-energy scalar before a two-stage dense regression
    head -- matching the official Keras functional model.
    """

    def __init__(self, seq_len: int = 30, in_channels: int = 4) -> None:
        super().__init__()
        self.conv3 = nn.Conv1d(in_channels, 100, kernel_size=3)
        self.conv5 = nn.Conv1d(in_channels, 70, kernel_size=5)
        self.conv7 = nn.Conv1d(in_channels, 40, kernel_size=7)
        self.drop = nn.Dropout(0.3)
        self.pool3 = nn.AvgPool1d(2, ceil_mode=True)
        self.pool5 = nn.AvgPool1d(2, ceil_mode=True)
        self.pool7 = nn.AvgPool1d(2, ceil_mode=True)

        len3 = (seq_len - 3 + 1 + 1) // 2
        len5 = (seq_len - 5 + 1 + 1) // 2
        len7 = (seq_len - 7 + 1 + 1) // 2
        concat_dim = 100 * len3 + 70 * len5 + 40 * len7

        self.dense0 = nn.Linear(concat_dim, 80)
        self.dense1 = nn.Linear(80 + 1, 80)
        self.dense2 = nn.Linear(80, 60)
        self.out = nn.Linear(60, 1)

    def forward(self, seq_onehot: Tensor, delta_g: Tensor) -> Tensor:
        """Predict a scalar on-target editing-efficiency score.

        Parameters
        ----------
        seq_onehot : Tensor
            One-hot target sequence, shape ``(batch, 4, seq_len)``.
        delta_g : Tensor
            Raw RNA-DNA hybridization free-energy scalar, shape
            ``(batch, 1)``.

        Returns
        -------
        Tensor
            Predicted efficiency score, shape ``(batch, 1)``.
        """
        b = seq_onehot.shape[0]
        h3 = self.pool3(self.drop(F.relu(self.conv3(seq_onehot)))).reshape(b, -1)
        h5 = self.pool5(self.drop(F.relu(self.conv5(seq_onehot)))).reshape(b, -1)
        h7 = self.pool7(self.drop(F.relu(self.conv7(seq_onehot)))).reshape(b, -1)
        concat = torch.cat([h3, h5, h7], dim=1)
        d0 = self.drop(F.relu(self.dense0(concat)))
        fused = torch.cat([d0, delta_g], dim=1)
        d1 = self.drop(F.relu(self.dense1(fused)))
        d2 = self.drop(F.relu(self.dense2(d1)))
        return self.out(d2)


def build_crispron() -> nn.Module:
    """Build a compact CRISPRon on-target efficiency predictor.

    Returns
    -------
    nn.Module
        Random-initialized ``CRISPRon`` in eval mode.
    """
    return CRISPRon().eval()


def example_input_crispron() -> tuple[Tensor, Tensor]:
    """Create an example one-hot 30-nt sequence plus raw deltaG scalar.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(seq_onehot, delta_g)`` of shapes ``(2, 4, 30)`` and ``(2, 1)``.
    """
    seq = torch.zeros(2, 4, 30)
    idx = torch.randint(0, 4, (2, 30))
    seq.scatter_(1, idx.unsqueeze(1), 1.0)
    delta_g = torch.randn(2, 1)
    return seq, delta_g


# ---------------------------------------------------------------------------
# DeepARG
# ---------------------------------------------------------------------------


class DeepARG(nn.Module):
    """DeepARG: deep MLP classifier over per-reference alignment bit-score
    features for antibiotic-resistance-gene (ARG) categorization.

    Matches ``DeepARGForSequenceClassification``: a stack of ``Linear +
    ReLU (+ Dropout)`` layers with default hidden sizes
    ``[2000, 1000, 500, 100]`` (dropout after all but the last hidden
    layer) feeding a linear classifier head.
    """

    def __init__(
        self,
        input_size: int = 64,
        hidden_sizes: tuple[int, ...] = (128, 64, 32, 16),
        num_labels: int = 6,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_size
        dropout_after_layers = max(0, len(hidden_sizes) - 1)
        for i, hidden in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            if i < dropout_after_layers:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        self.trunk = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, num_labels)

    def forward(self, alignment_features: Tensor) -> Tensor:
        """Classify a batch of alignment-bitscore feature vectors.

        Parameters
        ----------
        alignment_features : Tensor
            Per-reference-gene bit-score vector, shape
            ``(batch, input_size)``.

        Returns
        -------
        Tensor
            Class logits, shape ``(batch, num_labels)``.
        """
        hidden = self.trunk(alignment_features)
        return self.classifier(hidden)


def build_deeparg() -> nn.Module:
    """Build a compact DeepARG bit-score MLP classifier.

    Returns
    -------
    nn.Module
        Random-initialized ``DeepARG`` in eval mode.
    """
    return DeepARG().eval()


def example_input_deeparg() -> Tensor:
    """Create an example batch of alignment-bitscore feature vectors.

    Returns
    -------
    Tensor
        Random feature matrix of shape ``(4, 64)``.
    """
    return torch.randn(4, 64)


# ---------------------------------------------------------------------------
# DeepC
# ---------------------------------------------------------------------------


class GatedDilatedConv1d(nn.Module):
    """One WaveNet-style gated dilated Conv1d block with 1x1 residual.

    Computes ``out = input + conv1x1(tanh(dilated_filter(x)) *
    sigmoid(dilated_gate(x)))``, matching the official
    ``dilated_layer`` with ``residual=True, dense_residual=True``.
    """

    def __init__(self, channels: int, kernel_width: int, dilation: int) -> None:
        super().__init__()
        # Symmetric same-length padding for an even kernel width: pad more
        # on the left, then trim the one extra output position on the
        # right so every block preserves the input length exactly.
        self._extra_pad = (kernel_width - 1) * dilation % 2
        pad = ((kernel_width - 1) * dilation + self._extra_pad) // 2
        self.filter_conv = nn.Conv1d(
            channels, channels, kernel_width, dilation=dilation, padding=pad
        )
        self.gate_conv = nn.Conv1d(channels, channels, kernel_width, dilation=dilation, padding=pad)
        self.dense = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply one gated-dilated-residual block.

        Parameters
        ----------
        x : Tensor
            Input feature map, shape ``(batch, channels, length)``.

        Returns
        -------
        Tensor
            Residual-updated feature map, same shape as ``x``.
        """
        filt = self.filter_conv(x)
        gate = self.gate_conv(x)
        if self._extra_pad:
            filt = filt[..., :-1]
            gate = gate[..., :-1]
        gated = torch.tanh(filt) * torch.sigmoid(gate)
        transformed = self.dense(gated)
        return x + transformed


class DeepC(nn.Module):
    """DeepC: conv+pool front-end followed by an exponentially-dilated
    gated-residual conv stack, regressing chromatin-interaction tracks.

    Matches ``deepCregr.inference``: a small stack of plain
    ``Conv1d + ReLU + AvgPool1d`` layers narrows the raw one-hot sequence,
    then a cascade of ``GatedDilatedConv1d`` blocks with dilation doubling
    every layer (1, 2, 4, 8, ...) refines the representation before a
    final dense regression head over the flattened sequence axis.
    """

    def __init__(
        self,
        in_channels: int = 4,
        conv_channels: tuple[int, ...] = (32, 32),
        dilation_channels: int = 16,
        n_dilated_layers: int = 4,
        num_classes: int = 5,
    ) -> None:
        super().__init__()
        conv_layers = []
        prev = in_channels
        for ch in conv_channels:
            conv_layers.append(nn.Conv1d(prev, ch, kernel_size=5, padding=2))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.AvgPool1d(2, ceil_mode=True))
            prev = ch
        self.conv_stack = nn.Sequential(*conv_layers)

        self.project = nn.Conv1d(prev, dilation_channels, kernel_size=1)
        self.dilated_layers = nn.ModuleList(
            [
                GatedDilatedConv1d(dilation_channels, kernel_width=2, dilation=2**i)
                for i in range(n_dilated_layers)
            ]
        )
        self.dilation_channels = dilation_channels
        self.out_head = nn.Linear(dilation_channels, num_classes)

    def forward(self, seq_onehot: Tensor) -> Tensor:
        """Predict per-position chromatin-interaction regression scores.

        Parameters
        ----------
        seq_onehot : Tensor
            One-hot DNA sequence, shape ``(batch, 4, length)``.

        Returns
        -------
        Tensor
            Pooled regression output, shape ``(batch, num_classes)``.
        """
        h = self.conv_stack(seq_onehot)
        h = self.project(h)
        for layer in self.dilated_layers:
            h = layer(h)
        pooled = h.mean(dim=-1)
        return self.out_head(pooled)


def build_deepc() -> nn.Module:
    """Build a compact DeepC dilated-gated-conv regressor.

    Returns
    -------
    nn.Module
        Random-initialized ``DeepC`` in eval mode.
    """
    return DeepC().eval()


def example_input_deepc() -> Tensor:
    """Create an example one-hot DNA sequence window.

    Returns
    -------
    Tensor
        Random one-hot tensor of shape ``(2, 4, 256)``.
    """
    seq = torch.zeros(2, 4, 256)
    idx = torch.randint(0, 4, (2, 256))
    seq.scatter_(1, idx.unsqueeze(1), 1.0)
    return seq


# ---------------------------------------------------------------------------
# DeepCas9
# ---------------------------------------------------------------------------


class _DeepCas9Head(nn.Module):
    """One cell-line-specific scoring sub-network sharing the conv trunk."""

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Score a shared conv-trunk feature vector for one cell line.

        Parameters
        ----------
        x : Tensor
            Flattened shared conv-trunk features, shape ``(batch, in_features)``.

        Returns
        -------
        Tensor
            Scalar cell-line-specific activity score, shape ``(batch, 1)``.
        """
        return self.fc2(F.relu(self.fc1(x)))


class DeepCas9(nn.Module):
    """DeepCas9: shared 2D-conv trunk over the one-hot (4, 30) target image
    with three independent cell-line heads combined by fixed weights.

    Matches the repo's documented mechanism: the 30-nt one-hot target is
    treated as a ``(4, 30)`` 2-D image (``dim(test_onehot) <-
    c(4, 30, 1, N)``); a shared conv trunk feeds three independently
    trained sub-models (HEK293T, HL60, mEL4), whose scalar outputs are
    combined with the published fixed ensemble weights
    ``0.5*HL60 + 0.3*T293 + 0.2*mEL4``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(4, 5))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 5))
        self.pool = nn.MaxPool2d((1, 2), ceil_mode=True)

        flat_dim = 32 * 11
        self.head_hl60 = _DeepCas9Head(flat_dim)
        self.head_mel4 = _DeepCas9Head(flat_dim)
        self.head_293t = _DeepCas9Head(flat_dim)

    def forward(self, seq_onehot: Tensor) -> Tensor:
        """Score a batch of one-hot 30-nt targets with the fixed ensemble.

        Parameters
        ----------
        seq_onehot : Tensor
            One-hot target image, shape ``(batch, 1, 4, 30)``.

        Returns
        -------
        Tensor
            Ensembled ``DeepCas9_score``, shape ``(batch, 1)``.
        """
        h = F.relu(self.conv1(seq_onehot))
        h = self.pool(F.relu(self.conv2(h)))
        b = h.shape[0]
        h = h.reshape(b, -1)
        hl60 = self.head_hl60(h)
        mel4 = self.head_mel4(h)
        t293 = self.head_293t(h)
        return 0.5 * hl60 + 0.3 * t293 + 0.2 * mel4


def build_deepcas9() -> nn.Module:
    """Build a compact DeepCas9 cell-line-ensemble activity predictor.

    Returns
    -------
    nn.Module
        Random-initialized ``DeepCas9`` in eval mode.
    """
    return DeepCas9().eval()


def example_input_deepcas9() -> Tensor:
    """Create an example one-hot ``(4, 30)`` target image batch.

    Returns
    -------
    Tensor
        Random one-hot tensor of shape ``(2, 1, 4, 30)``.
    """
    seq = torch.zeros(2, 4, 30)
    idx = torch.randint(0, 4, (2, 30))
    seq.scatter_(1, idx.unsqueeze(1), 1.0)
    return seq.unsqueeze(1)


# ---------------------------------------------------------------------------
# DeepCCI
# ---------------------------------------------------------------------------


class _GraphConvLayer(nn.Module):
    """One GCN layer: ``relu(adj @ (x @ W))`` with a dense adjacency."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: Tensor, adj: Tensor, active: bool = True) -> Tensor:
        """Propagate features over a dense adjacency matrix.

        Parameters
        ----------
        x : Tensor
            Node feature matrix, shape ``(n_nodes, in_features)``.
        adj : Tensor
            Dense normalized adjacency, shape ``(n_nodes, n_nodes)``.
        active : bool
            Whether to apply the ReLU nonlinearity.

        Returns
        -------
        Tensor
            Propagated node features, shape ``(n_nodes, out_features)``.
        """
        support = x @ self.weight
        out = adj @ support
        return F.relu(out) if active else out


class DeepCCIClusterModel(nn.Module):
    """DeepCCI unsupervised clustering model: SDCN-style dual AE/GCN fusion
    with a Student's-t soft cluster assignment head.

    Matches ``ClusterModel.forward``: a stacked autoencoder and a parallel
    stack of ``GNNLayer`` graph-conv layers process the same cell feature
    matrix over the same KNN cell graph; at every AE layer the GCN hidden
    state from the corresponding depth is added into the AE's pre-activation
    input (the encoder-GCN cross-fusion), and the shared bottleneck ``z``
    drives both a Student's-t soft cluster assignment ``q`` and a softmax
    GCN classification head ``predict``, alongside the AE reconstruction
    ``x_bar``.
    """

    def __init__(
        self,
        n_input: int = 50,
        n_enc_1: int = 24,
        n_enc_2: int = 16,
        n_enc_3: int = 16,
        n_z: int = 8,
        n_clusters: int = 4,
    ) -> None:
        super().__init__()
        self.enc_1 = nn.Linear(n_input, n_enc_1)
        self.enc_2 = nn.Linear(n_enc_1, n_enc_2)
        self.enc_3 = nn.Linear(n_enc_2, n_enc_3)
        self.z_layer = nn.Linear(n_enc_3, n_z)

        self.dec_1 = nn.Linear(n_z, n_enc_3)
        self.dec_2 = nn.Linear(n_enc_3, n_enc_2)
        self.dec_3 = nn.Linear(n_enc_2, n_enc_1)
        self.x_bar_layer = nn.Linear(n_enc_1, n_input)

        self.gnn_1 = _GraphConvLayer(n_input, n_enc_1)
        self.gnn_2 = _GraphConvLayer(n_enc_1, n_enc_2)
        self.gnn_3 = _GraphConvLayer(n_enc_2, n_enc_3)
        self.gnn_4 = _GraphConvLayer(n_enc_3, n_z)
        self.gnn_5 = _GraphConvLayer(n_z, n_clusters)

        self.cluster_layer = nn.Parameter(torch.empty(n_clusters, n_z))
        nn.init.xavier_normal_(self.cluster_layer.data)
        self.v = 1.0

    def forward(self, x: Tensor, adj: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run the dual AE/GCN fusion and Student's-t clustering head.

        Parameters
        ----------
        x : Tensor
            Per-cell feature matrix, shape ``(n_cells, n_input)``.
        adj : Tensor
            Dense normalized cell-cell KNN adjacency, shape
            ``(n_cells, n_cells)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            ``(x_bar, q, predict, z)`` reconstruction, soft cluster
            assignment, softmax GCN prediction, and bottleneck embedding.
        """
        h1 = self.gnn_1(x, adj)
        h2 = self.gnn_2(h1, adj)
        h3 = self.gnn_3(h2, adj)
        h4 = self.gnn_4(h3, adj)
        h5 = self.gnn_5(h4, adj, active=False)
        predict = F.softmax(h5, dim=1)

        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1 + h1))
        enc_h3 = F.relu(self.enc_3(enc_h2 + h2))
        z = self.z_layer(enc_h3 + h3)

        dec_h1 = F.relu(self.dec_1(z + h4))
        dec_h2 = F.relu(self.dec_2(dec_h1 + h3))
        dec_h3 = F.relu(self.dec_3(dec_h2 + h2))
        x_bar = self.x_bar_layer(dec_h3 + h1)

        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.cluster_layer) ** 2, dim=2) / self.v)
        q = q ** ((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()

        return x_bar, q, predict, z


def build_deepcci() -> nn.Module:
    """Build a compact DeepCCI dual AE/GCN clustering model.

    Returns
    -------
    nn.Module
        Random-initialized ``DeepCCIClusterModel`` in eval mode.
    """
    return DeepCCIClusterModel().eval()


def example_input_deepcci() -> tuple[Tensor, Tensor]:
    """Create an example cell-feature matrix and dense KNN adjacency.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(x, adj)`` of shapes ``(12, 50)`` and ``(12, 12)``; ``adj`` is a
        symmetric row-normalized dense adjacency (self-loops included).
    """
    n = 12
    x = torch.randn(n, 50)
    raw = (torch.rand(n, n) > 0.7).float()
    raw = raw + raw.t() + torch.eye(n)
    raw = (raw > 0).float()
    adj = raw / raw.sum(dim=1, keepdim=True)
    return x, adj


# ---------------------------------------------------------------------------
# DeepCDR
# ---------------------------------------------------------------------------


class _UniformGraphConv(nn.Module):
    """Mean-aggregation graph conv: ``sigma(sum_j A_ij (h_j W + b) / deg_i)``."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, h: Tensor, adj: Tensor) -> Tensor:
        """Propagate atom features over the drug molecular graph.

        Parameters
        ----------
        h : Tensor
            Per-atom feature matrix, shape ``(batch, n_atoms, in_features)``.
        adj : Tensor
            Dense binary atom adjacency, shape ``(batch, n_atoms, n_atoms)``.

        Returns
        -------
        Tensor
            Propagated atom features, shape ``(batch, n_atoms, out_features)``.
        """
        transformed = self.linear(h)
        aggregated = torch.bmm(adj, transformed)
        degree = adj.sum(dim=2, keepdim=True).clamp(min=1.0)
        return aggregated / degree


class DeepCDR(nn.Module):
    """DeepCDR: drug-graph GCN branch + genomic-mutation / gene-expression /
    methylation multi-omics branches, fused and regressed to IC50.

    Matches ``KerasMultiSourceGCNModel.createMaster``: a stack of
    mean-aggregation graph-conv layers over the drug's atom graph followed
    by global max pooling; a two-stage ``Conv2d`` encoder over the
    mutation vector (treated as a ``(1, mutation_dim)`` image); plain MLP
    encoders for gene expression and methylation; concatenation, a dense
    fusion layer, then a *second* small ``Conv2d`` stack (the fused vector
    reshaped back into a synthetic 1x1 image) before the scalar IC50
    regression head.
    """

    def __init__(
        self,
        drug_feat_dim: int = 16,
        gcn_units: tuple[int, ...] = (32, 32, 32),
        mutation_dim: int = 200,
        gexpr_dim: int = 64,
        methy_dim: int = 32,
    ) -> None:
        super().__init__()
        gcn_layers = []
        prev = drug_feat_dim
        for units in (*gcn_units, 20):
            gcn_layers.append(_UniformGraphConv(prev, units))
            prev = units
        self.gcn_layers = nn.ModuleList(gcn_layers)
        self.drug_out_dim = prev

        self.mut_conv1 = nn.Conv2d(1, 8, kernel_size=(1, 7), stride=(1, 2))
        self.mut_pool1 = nn.MaxPool2d((1, 3), ceil_mode=True)
        self.mut_conv2 = nn.Conv2d(8, 6, kernel_size=(1, 5), stride=(1, 2))
        self.mut_pool2 = nn.MaxPool2d((1, 3), ceil_mode=True)
        mut_flat_len = ((((mutation_dim - 7) // 2 + 1 + 2) // 3) - 5) // 2 + 1
        mut_flat_len = max(mut_flat_len, 1)
        mut_flat_len = (mut_flat_len + 2) // 3
        self.mut_fc = nn.Linear(6 * max(mut_flat_len, 1), 20)

        self.gexpr_fc1 = nn.Linear(gexpr_dim, 32)
        self.gexpr_bn = nn.BatchNorm1d(32)
        self.gexpr_fc2 = nn.Linear(32, 20)

        self.methy_fc1 = nn.Linear(methy_dim, 32)
        self.methy_bn = nn.BatchNorm1d(32)
        self.methy_fc2 = nn.Linear(32, 20)

        fusion_in = self.drug_out_dim + 20 + 20 + 20
        self.fusion = nn.Linear(fusion_in, 40)

        self.post_conv1 = nn.Conv2d(1, 6, kernel_size=(1, 5))
        self.post_conv2 = nn.Conv2d(6, 4, kernel_size=(1, 3))
        self.post_pool = nn.MaxPool2d((1, 2), ceil_mode=True)
        post_len = 40 - 5 + 1
        post_len = post_len - 3 + 1
        post_len = (post_len + 1) // 2
        self.out_head = nn.Linear(4 * post_len, 1)

    def forward(
        self,
        drug_feat: Tensor,
        drug_adj: Tensor,
        mutation: Tensor,
        gexpr: Tensor,
        methy: Tensor,
    ) -> Tensor:
        """Predict a scalar drug-response (IC50) score.

        Parameters
        ----------
        drug_feat : Tensor
            Per-atom drug feature matrix, shape ``(batch, n_atoms, drug_feat_dim)``.
        drug_adj : Tensor
            Dense binary atom adjacency, shape ``(batch, n_atoms, n_atoms)``.
        mutation : Tensor
            Genomic mutation strip, shape ``(batch, 1, 1, mutation_dim)``.
        gexpr : Tensor
            Gene-expression vector, shape ``(batch, gexpr_dim)``.
        methy : Tensor
            DNA-methylation vector, shape ``(batch, methy_dim)``.

        Returns
        -------
        Tensor
            Predicted IC50-like score, shape ``(batch, 1)``.
        """
        h = drug_feat
        for layer in self.gcn_layers:
            h = F.relu(layer(h, drug_adj))
        x_drug = h.max(dim=1).values

        x_mut = F.relu(self.mut_conv1(mutation))
        x_mut = self.mut_pool1(x_mut)
        x_mut = F.relu(self.mut_conv2(x_mut))
        x_mut = self.mut_pool2(x_mut)
        x_mut = x_mut.reshape(x_mut.shape[0], -1)
        x_mut = F.relu(self.mut_fc(x_mut))

        x_gexpr = torch.tanh(self.gexpr_fc1(gexpr))
        x_gexpr = self.gexpr_bn(x_gexpr)
        x_gexpr = F.relu(self.gexpr_fc2(x_gexpr))

        x_methy = torch.tanh(self.methy_fc1(methy))
        x_methy = self.methy_bn(x_methy)
        x_methy = F.relu(self.methy_fc2(x_methy))

        fused = torch.cat([x_drug, x_mut, x_gexpr, x_methy], dim=1)
        fused = torch.tanh(self.fusion(fused))

        img = fused.unsqueeze(1).unsqueeze(1)
        h2 = F.relu(self.post_conv1(img))
        h2 = F.relu(self.post_conv2(h2))
        h2 = self.post_pool(h2)
        h2 = h2.reshape(h2.shape[0], -1)
        return self.out_head(h2)


def build_deepcdr() -> nn.Module:
    """Build a compact DeepCDR multi-source drug-response regressor.

    Returns
    -------
    nn.Module
        Random-initialized ``DeepCDR`` in eval mode.
    """
    return DeepCDR().eval()


def example_input_deepcdr() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Create an example drug graph plus three omics feature vectors.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
        ``(drug_feat, drug_adj, mutation, gexpr, methy)`` batch of size 2:
        drug graph with 10 atoms / 16-dim features, ``(2, 1, 1, 200)``
        mutation strip, and 64-/32-dim gene-expression / methylation
        vectors.
    """
    b, n_atoms = 2, 10
    drug_feat = torch.randn(b, n_atoms, 16)
    raw = (torch.rand(b, n_atoms, n_atoms) > 0.6).float()
    raw = raw + raw.transpose(1, 2)
    eye = torch.eye(n_atoms).unsqueeze(0).expand(b, -1, -1)
    drug_adj = ((raw + eye) > 0).float()
    mutation = torch.zeros(b, 1, 1, 200)
    idx = torch.randint(0, 200, (b, 40))
    mutation.scatter_(3, idx.unsqueeze(1).unsqueeze(1), 1.0)
    gexpr = torch.randn(b, 64)
    methy = torch.rand(b, 32)
    return drug_feat, drug_adj, mutation, gexpr, methy


MENAGERIE_ENTRIES = [
    ("CRISPRon", "build_crispron", "example_input_crispron", "2021", "BIO"),
    ("DeepARG", "build_deeparg", "example_input_deeparg", "2018", "BIO"),
    ("DeepC", "build_deepc", "example_input_deepc", "2020", "BIO"),
    ("DeepCas9", "build_deepcas9", "example_input_deepcas9", "2019", "BIO"),
    ("DeepCCI", "build_deepcci", "example_input_deepcci", "2023", "BIO"),
    ("DeepCDR", "build_deepcdr", "example_input_deepcdr", "2020", "BIO"),
]
