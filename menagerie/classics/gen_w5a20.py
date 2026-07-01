"""Bioinformatics / genomics classics (batch w5a20).

Sources checked (repo/paper architecture; no clone, no pip install --
reimplemented from scratch in base-env torch):

- DeepChrome: Singh, Yang, Poczos & Ma, Bioinformatics 2016,
  https://academic.oup.com/bioinformatics/article/32/17/i639/2450757.
  Official repo https://github.com/QData/DeepChrome
  (``deepChrome-TorchCode/2_model.lua``, the ``convnet`` branch -- exact
  layer sizes read from source). DeepChrome predicts binary (high/low)
  gene expression from a fixed ``5 x 100`` matrix of five core histone
  modification ChIP-seq signal bins (100 bp bins across +/-5000 bp flanking
  the TSS): a single 1-D convolution over the bin axis with the 5 marks as
  input channels (``TemporalConvolution(5, 50, filtsize=10)``) --
  automatically learning *combinatorial interactions among histone marks*
  is the paper's namesake departure from single-mark threshold rules --
  followed by ReLU, max-pool (size 5), flatten, dropout, and a 3-layer
  dense classifier head (625 -> 125 -> 2 hidden units, matching
  ``nstates = {50, 625, 125}`` in the source) ending in a 2-way softmax
  score. Reimplemented here as ``DeepChrome`` with the identical
  Conv1d(5->50, k=10) + ReLU + MaxPool1d(5) trunk and 3-layer dense head.

- DeepCLIP: Groenning, Doktor, Larsen, Petersen, Holm, Bramsen, Sorensen,
  Bramsen & Vinther, Nucleic Acids Research 2020,
  https://academic.oup.com/nar/article/48/13/7099/5859960. Official repo
  https://github.com/deepclip/deepclip (``network.py``, function
  ``build_model`` -- Theano/Lasagne reference, used only as architecture
  source, not executed). DeepCLIP predicts RNA-binding-protein binding
  preference directly from one-hot RNA sequence using a **multi-width
  convolutional filter bank run in parallel with a BiLSTM**, then folds
  the BiLSTM's *per-position* hidden state back onto the raw one-hot input
  as a per-nucleotide importance weight before the final classifier --
  the mechanism the paper exploits to score point-mutation effects
  (re-run the same forward pass on a 1-nt-edited sequence and read off the
  change in the per-position profile). Concretely: several same-padded
  ``Conv1d`` branches at different kernel widths scan the one-hot
  sequence in parallel (a multi-scale motif bank, mirroring the official
  ``FILTER_SIZES`` list), the concatenated conv features are appended
  along the channel axis to the raw one-hot at every position, a
  bidirectional LSTM sweeps the augmented per-position sequence, the two
  directions are summed per position into a scalar profile weight, that
  weight is broadcast-multiplied back onto the original one-hot input
  (``l_sumz2x`` elementwise-multiplied with ``l_in`` in the source), and
  the weighted one-hot is summed over the sequence axis and passed through
  a final linear+sigmoid binding-score head. Reimplemented here as
  ``DeepCLIP`` with the multi-width conv bank, one-hot concatenation,
  BiLSTM, per-position profile projection back onto the one-hot input, and
  sigmoid scalar head.

- DeepCpf1: Kim, Song, Koh, Kim, Kim, Kim & Kim, Nature Biotechnology 2018,
  https://www.nature.com/articles/nbt.4061. Official repo
  https://github.com/lje00006/DeepCpf1 (also mirrored on Kipoi; a Keras
  ``Convolution1D`` on a 34/34-nt one-hot Cpf1/Cas12a target-plus-context
  window is the documented architecture: 80 filters of width 5 over the
  4-channel one-hot sequence, ReLU, average-pool, flatten, dropout, and a
  2-layer dense regression head predicting a scalar on-target indel
  efficiency score -- the paper's namesake departure from earlier Cas9
  guide-scoring CNNs is scoring the AT-rich Cpf1 PAM-proximal seed
  directly from an extended one-hot window rather than a fixed 20-nt
  protospacer). Reimplemented here as ``DeepCpf1`` with a single wide
  ``Conv1d`` motif-scanning layer, ReLU, average-pool, flatten, dropout,
  and a 2-layer dense regression head.

- DeepCpG: Angermueller, Lee, Reik & Stegle, Genome Biology 2017,
  https://link.springer.com/article/10.1186/s13059-017-1189-z. Official
  repo https://github.com/cangermueller/deepcpg (``deepcpg/models/dna.py``
  for the DNA CNN module and ``deepcpg/models/cpg.py`` for the CpG
  bidirectional-GRU module; ``deepcpg/models/joint.py`` for the fusion
  head -- architecture reference, not executed / not pip-installed).
  DeepCpG imputes single-cell CpG methylation state from **two
  independent branches fused at a joint module**: a DNA module (two
  Conv1d+pool stages over the local one-hot DNA sequence context around
  each CpG site, plus a dense layer, extracting local sequence motifs)
  and a CpG module (a *bidirectional GRU scanning neighbouring CpG sites
  across cells*, compressing sparse per-cell methylation-state + distance
  observations into a fixed-size vector per cell) -- the paper's namesake
  departure from single-branch predictors is exploiting both local
  sequence content and the correlation structure of nearby CpG sites
  across cells simultaneously. The two branch outputs are concatenated
  and passed through a joint dense module to a per-cell sigmoid
  methylation-probability head. Reimplemented here as ``DeepCpG`` with
  the two-Conv1d DNA branch, the bidirectional-GRU CpG branch scanning a
  neighbouring-CpG-site window per cell, concatenation, and the joint
  dense + per-cell sigmoid head.

- DeepCRISPR: Chuai, Ma, Yan, Chen, Hu, Zhang, Zhan, Lin, Cai, Peng, Wu,
  Ouyang, Zhang, Nie, Bhatia, Wang, Zhai, Cai & Zhang, Genome Biology 2018,
  https://www.biorxiv.org/content/10.1101/288340 /
  https://link.springer.com/article/10.1186/s13059-018-1459-4. Official
  repo https://github.com/bm2-lab/DeepCRISPR (``deepcrispr.py``, classes
  ``DCModelOntar`` / ``DCModelOfftar`` built on a shared denoising
  convolutional autoencoder ``Config`` stack -- architecture reference,
  not executed). DeepCRISPR unifies on-target and off-target sgRNA
  efficiency prediction on one **deep convolutional denoising autoencoder
  (DCDNN) backbone**: an sgRNA (plus optional epigenetic tracks) is
  one-hot-encoded across 4 sequence channels (each additional epigenetic
  feature is its own channel), noise is injected during pretraining, an
  encoder of stride-2 ``Conv1d`` layers compresses the sequence into a
  latent representation, a mirrored decoder of transposed convolutions
  reconstructs the clean input (unsupervised representation pretraining
  -- the paper's namesake departure from purely supervised CRISPR CNNs),
  and the pretrained encoder is then topped with a small supervised CNN
  head (for on-target: two more conv layers + global pooling + linear
  regression head; the same encoder is reused with a paired-input Siamese
  head for off-target classification). Reimplemented here as
  ``DeepCRISPR`` exposing both the full encoder-decoder autoencoder
  forward pass (``forward(x, mode="autoencode")``) and the supervised
  on-target regression head built on top of the encoder
  (``forward(x, mode="ontarget")``), matching the two-stage
  pretrain-then-finetune design; the smoke-traced example uses the
  on-target regression path.

- DeepDDS: Wang, Liu, Luo, Cheng, Wang, Xie, Pei, Zhang & Yu, Briefings in
  Bioinformatics 2022, https://academic.oup.com/bib/article/23/1/bbab390/
  6375262 (preprint https://arxiv.org/abs/2107.02467). Official repo
  https://github.com/Sinwang404/DeepDDS (``models/gat.py``, class
  ``GATNet`` -- exact layer sizes and forward pass read from source).
  DeepDDS predicts synergistic anticancer drug-combination effect from a
  **drug pair plus a cell line**: each drug's 2-D molecular graph (RDKit
  atom features as node features, bonds as edges) is independently
  encoded by the *same shared* two-layer Graph Attention Network
  (``GATConv`` with 10 attention heads on the first layer, 1 head on the
  second, ELU + dropout between), global-max-pooled into a fixed drug
  embedding, and passed through a linear+ReLU projection; the cell line's
  gene-expression vector is separately reduced through a 3-layer dense
  bottleneck (954 -> 2048 -> 512 -> 2*output_dim); the two (weight-shared)
  drug embeddings and the cell-line embedding are L2-normalized and
  concatenated, then fused through a 3-layer dense head (2048 -> 512 ->
  128) ending in a 2-way synergy/antagonism classifier -- the paper's
  namesake departure from single-drug QSAR CNNs is scoring a *combination*
  by encoding both drug graphs with a shared graph-attention encoder and
  fusing with cell-line context. Reimplemented here as ``DeepDDS`` with
  the shared 2-layer ``GATConv`` drug encoder (applied independently to
  both drug graphs, exactly mirroring the official ``drug1_gcn1`` /
  ``drug1_gcn2`` weight reuse), the cell-line dense reduction branch, and
  the concatenation + dense fusion classifier head.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp


class DeepChrome(nn.Module):
    """CNN over a 5-histone-mark x 100-bin signal matrix for gene expression.

    A single wide temporal convolution over the bin axis (5 histone-mark
    channels in, 50 motif-detector channels out) learns combinatorial
    interactions among histone marks flanking the TSS, followed by
    max-pooling and a 3-layer dense classifier (matching the official
    ``nstates = {50, 625, 125}`` Torch reference).
    """

    def __init__(
        self,
        n_marks: int = 5,
        n_bins: int = 100,
        n_filters: int = 50,
        filter_size: int = 10,
        pool_size: int = 5,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(n_marks, n_filters, kernel_size=filter_size)
        self.pool = nn.MaxPool1d(pool_size)
        conv_out = (n_bins - filter_size + 1) // pool_size
        flat = conv_out * n_filters
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(flat, 625)
        self.fc2 = nn.Linear(625, 125)
        self.fc3 = nn.Linear(125, 2)

    def forward(self, x: Tensor) -> Tensor:
        """Predict binary gene-expression logits from histone-mark bins.

        Parameters
        ----------
        x : Tensor
            Histone-mark signal, shape ``(batch, n_marks, n_bins)``.

        Returns
        -------
        Tensor
            Class logits, shape ``(batch, 2)``.
        """
        h = F.relu(self.conv(x))
        h = self.pool(h)
        h = h.reshape(h.shape[0], -1)
        h = self.dropout(h)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


def build_deepchrome() -> nn.Module:
    """Build a compact DeepChrome histone-mark gene-expression classifier.

    Returns
    -------
    nn.Module
        Random-initialized ``DeepChrome`` in eval mode.
    """
    return DeepChrome().eval()


def example_input_deepchrome() -> Tensor:
    """Create an example 5-mark x 100-bin histone signal matrix.

    Returns
    -------
    Tensor
        Batch of shape ``(2, 5, 100)``.
    """
    return torch.rand(2, 5, 100)


class DeepCLIP(nn.Module):
    """Multi-width conv bank + BiLSTM RBP binding-preference scorer.

    Several parallel same-padded ``Conv1d`` branches scan the one-hot RNA
    sequence at different motif widths; their concatenated features are
    appended to the raw one-hot at every position and swept by a
    bidirectional LSTM. The two LSTM directions are summed into a scalar
    per-position profile weight that is broadcast back onto the one-hot
    input (the mechanism used to read out point-mutation effects), then
    summed over the sequence and passed through a sigmoid binding head.
    """

    def __init__(
        self,
        vocab: int = 4,
        seq_len: int = 50,
        filter_sizes: tuple[int, ...] = (4, 8, 12),
        filters_per_conv: int = 16,
        lstm_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(vocab, filters_per_conv, kernel_size=k, padding=k // 2)
                for k in filter_sizes
            ]
        )
        conv_channels = filters_per_conv * len(filter_sizes)
        self.lstm = nn.LSTM(
            input_size=vocab + conv_channels,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.profile_proj = nn.Linear(2 * lstm_hidden, 1)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(vocab, 1)
        self.seq_len = seq_len

    def forward(self, x: Tensor) -> Tensor:
        """Predict a scalar binding score from a one-hot RNA sequence.

        Parameters
        ----------
        x : Tensor
            One-hot sequence, shape ``(batch, seq_len, vocab)``.

        Returns
        -------
        Tensor
            Binding-preference score in ``(0, 1)``, shape ``(batch, 1)``.
        """
        x_cl = x.transpose(1, 2)
        conv_feats = [F.relu(c(x_cl)) for c in self.convs]
        conv_feats = [f[:, :, : x.shape[1]] for f in conv_feats]
        conv_cat = torch.cat(conv_feats, dim=1).transpose(1, 2)
        lstm_in = torch.cat([x, conv_cat], dim=2)
        h, _ = self.lstm(lstm_in)
        profile = self.profile_proj(h)
        weighted = x * profile
        weighted = self.dropout(weighted)
        pooled = weighted.sum(dim=1)
        return torch.sigmoid(self.out(pooled))


def build_deepclip() -> nn.Module:
    """Build a compact DeepCLIP CNN+BiLSTM RBP binding scorer.

    Returns
    -------
    nn.Module
        Random-initialized ``DeepCLIP`` in eval mode.
    """
    return DeepCLIP().eval()


def example_input_deepclip() -> Tensor:
    """Create an example one-hot RNA sequence batch.

    Returns
    -------
    Tensor
        Batch of shape ``(2, 50, 4)`` (one-hot ACGU).
    """
    idx = torch.randint(0, 4, (2, 50))
    return F.one_hot(idx, num_classes=4).float()


class DeepCpf1(nn.Module):
    """CNN over an extended one-hot Cas12a target window for guide activity.

    A single wide motif-scanning convolution over the 34-nt one-hot
    target-plus-context window, average-pooled and flattened into a
    2-layer dense regression head, predicting on-target indel efficiency.
    """

    def __init__(
        self,
        seq_len: int = 34,
        vocab: int = 4,
        n_filters: int = 80,
        filter_size: int = 5,
        pool_size: int = 2,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(vocab, n_filters, kernel_size=filter_size)
        self.pool = nn.AvgPool1d(pool_size)
        conv_out = (seq_len - filter_size + 1) // pool_size
        flat = conv_out * n_filters
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(flat, 80)
        self.fc2 = nn.Linear(80, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Predict a scalar on-target indel efficiency score.

        Parameters
        ----------
        x : Tensor
            One-hot target window, shape ``(batch, seq_len, vocab)``.

        Returns
        -------
        Tensor
            Predicted efficiency score, shape ``(batch, 1)``.
        """
        h = x.transpose(1, 2)
        h = F.relu(self.conv(h))
        h = self.pool(h)
        h = h.reshape(h.shape[0], -1)
        h = self.dropout(h)
        h = F.relu(self.fc1(h))
        return self.fc2(h)


def build_deepcpf1() -> nn.Module:
    """Build a compact DeepCpf1 Cas12a guide-activity regressor.

    Returns
    -------
    nn.Module
        Random-initialized ``DeepCpf1`` in eval mode.
    """
    return DeepCpf1().eval()


def example_input_deepcpf1() -> Tensor:
    """Create an example one-hot Cas12a target window.

    Returns
    -------
    Tensor
        Batch of shape ``(2, 34, 4)`` (one-hot ACGT).
    """
    idx = torch.randint(0, 4, (2, 34))
    return F.one_hot(idx, num_classes=4).float()


class DeepCpG(nn.Module):
    """Two-branch DNA-CNN + CpG-BiGRU model for single-cell methylation.

    A DNA module (two Conv1d+pool stages plus a dense layer) extracts
    local sequence motifs around each CpG site; a CpG module scans
    neighbouring CpG sites' methylation state + distance across cells with
    a bidirectional GRU. The two branch outputs are concatenated and fused
    through a joint dense module into a per-cell methylation-probability
    head.
    """

    def __init__(
        self,
        dna_len: int = 101,
        n_cpg_neighbors: int = 10,
        n_cells: int = 4,
        dna_channels: int = 4,
    ) -> None:
        super().__init__()
        self.dna_conv1 = nn.Conv1d(dna_channels, 32, kernel_size=11, padding=5)
        self.dna_pool1 = nn.MaxPool1d(4)
        self.dna_conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.dna_pool2 = nn.MaxPool1d(2)
        dna_out_len = dna_len // 4 // 2
        self.dna_fc = nn.Linear(64 * dna_out_len, 128)

        self.cpg_gru = nn.GRU(
            input_size=2,
            hidden_size=32,
            batch_first=True,
            bidirectional=True,
        )
        self.cpg_fc = nn.Linear(64, 128)

        self.joint_fc = nn.Linear(128 + 128, 128)
        self.out = nn.Linear(128, n_cells)
        self.n_cells = n_cells
        self.n_cpg_neighbors = n_cpg_neighbors

    def forward(self, dna: Tensor, cpg: Tensor) -> Tensor:
        """Predict per-cell methylation probabilities at a CpG site.

        Parameters
        ----------
        dna : Tensor
            One-hot local DNA sequence context, shape
            ``(batch, dna_len, 4)``.
        cpg : Tensor
            Neighbouring CpG-site (state, distance) pairs per cell, shape
            ``(batch, n_cells, n_cpg_neighbors, 2)``.

        Returns
        -------
        Tensor
            Per-cell methylation logits, shape ``(batch, n_cells)``.
        """
        h_dna = dna.transpose(1, 2)
        h_dna = F.relu(self.dna_conv1(h_dna))
        h_dna = self.dna_pool1(h_dna)
        h_dna = F.relu(self.dna_conv2(h_dna))
        h_dna = self.dna_pool2(h_dna)
        h_dna = h_dna.reshape(h_dna.shape[0], -1)
        h_dna = F.relu(self.dna_fc(h_dna))

        b, n_cells, n_neigh, _ = cpg.shape
        cpg_flat = cpg.reshape(b * n_cells, n_neigh, 2)
        _, h_n = self.cpg_gru(cpg_flat)
        h_cpg = h_n.transpose(0, 1).reshape(b * n_cells, -1)
        h_cpg = F.relu(self.cpg_fc(h_cpg))
        h_cpg = h_cpg.reshape(b, n_cells, -1).mean(dim=1)

        joint = torch.cat([h_dna, h_cpg], dim=1)
        joint = F.relu(self.joint_fc(joint))
        return self.out(joint)


def build_deepcpg() -> nn.Module:
    """Build a compact DeepCpG DNA-CNN + CpG-BiGRU methylation predictor.

    Returns
    -------
    nn.Module
        Random-initialized ``DeepCpG`` in eval mode.
    """
    return DeepCpG().eval()


def example_input_deepcpg() -> tuple[Tensor, Tensor]:
    """Create an example DNA context window plus neighbouring-CpG tensor.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(dna, cpg)`` batch of size 2: one-hot 101-bp DNA window and
        ``(2, 4, 10, 2)`` per-cell neighbouring CpG (state, distance)
        pairs.
    """
    idx = torch.randint(0, 4, (2, 101))
    dna = F.one_hot(idx, num_classes=4).float()
    state = torch.randint(0, 2, (2, 4, 10, 1)).float()
    dist = torch.rand(2, 4, 10, 1)
    cpg = torch.cat([state, dist], dim=-1)
    return dna, cpg


class DeepCRISPR(nn.Module):
    """Denoising convolutional autoencoder for CRISPR guide representation.

    A stride-2 ``Conv1d`` encoder compresses a one-hot (+ optional
    epigenetic-track channels) sgRNA region into a latent representation;
    a mirrored transposed-convolution decoder reconstructs the clean
    input for unsupervised pretraining. A small supervised head built on
    the pretrained encoder predicts scalar on-target editing efficiency.
    """

    def __init__(self, seq_len: int = 32, in_channels: int = 4) -> None:
        super().__init__()
        self.enc1 = nn.Conv1d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.enc2 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enc3 = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1)

        self.dec1 = nn.ConvTranspose1d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec3 = nn.ConvTranspose1d(
            32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1
        )

        latent_len = seq_len // 8
        self.head_conv = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.head_fc1 = nn.Linear(32 * latent_len, 64)
        self.head_fc2 = nn.Linear(64, 1)

    def encode(self, x: Tensor) -> Tensor:
        """Run the denoising-autoencoder encoder.

        Parameters
        ----------
        x : Tensor
            One-hot (+ epigenetic-channel) sgRNA region, shape
            ``(batch, in_channels, seq_len)``.

        Returns
        -------
        Tensor
            Latent representation, shape ``(batch, 64, seq_len // 8)``.
        """
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        return F.relu(self.enc3(h))

    def forward(self, x: Tensor, mode: str = "ontarget") -> Tensor:
        """Run the autoencoder reconstruction or the on-target head.

        Parameters
        ----------
        x : Tensor
            One-hot (+ epigenetic-channel) sgRNA region, shape
            ``(batch, in_channels, seq_len)``.
        mode : str
            ``"autoencode"`` returns the reconstructed input (pretraining
            forward pass); ``"ontarget"`` (default) returns the scalar
            on-target efficiency prediction from the finetuned head.

        Returns
        -------
        Tensor
            Reconstruction ``(batch, in_channels, seq_len)`` when
            ``mode="autoencode"``, else efficiency score ``(batch, 1)``.
        """
        z = self.encode(x)
        if mode == "autoencode":
            h = F.relu(self.dec1(z))
            h = F.relu(self.dec2(h))
            return self.dec3(h)
        h = F.relu(self.head_conv(z))
        h = h.reshape(h.shape[0], -1)
        h = F.relu(self.head_fc1(h))
        return self.head_fc2(h)


def build_deepcrispr() -> nn.Module:
    """Build a compact DeepCRISPR denoising-autoencoder guide scorer.

    Returns
    -------
    nn.Module
        Random-initialized ``DeepCRISPR`` in eval mode.
    """
    return DeepCRISPR().eval()


def example_input_deepcrispr() -> Tensor:
    """Create an example one-hot sgRNA region.

    Returns
    -------
    Tensor
        Batch of shape ``(2, 4, 32)`` (one-hot ACGT channels-first).
    """
    idx = torch.randint(0, 4, (2, 32))
    onehot = F.one_hot(idx, num_classes=4).float()
    return onehot.transpose(1, 2)


class DeepDDS(nn.Module):
    """Shared-GAT drug-pair encoder fused with a cell-line context branch.

    Both drugs in a candidate combination are encoded independently by
    the *same* two-layer Graph Attention Network (10 heads then 1 head),
    global-max-pooled and projected; the cell line's gene-expression
    vector is reduced through a 3-layer dense bottleneck. The three
    embeddings are concatenated and fused through a dense head predicting
    synergy vs. antagonism.
    """

    def __init__(
        self,
        atom_features: int = 12,
        n_genes: int = 32,
        output_dim: int = 16,
        n_classes: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.gat1 = GATConv(atom_features, output_dim, heads=heads, dropout=dropout)
        self.gat2 = GATConv(output_dim * heads, output_dim, dropout=dropout)
        self.drug_fc = nn.Linear(output_dim, output_dim)

        self.cell_reduction = nn.Sequential(
            nn.Linear(n_genes, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim * 2),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(output_dim * 4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.out = nn.Linear(16, n_classes)
        self.dropout = nn.Dropout(dropout)

    def _encode_drug(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.gat2(h, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        h = gmp(h, batch)
        return F.relu(self.drug_fc(h))

    def forward(
        self,
        drug1_x: Tensor,
        drug1_edge_index: Tensor,
        drug1_batch: Tensor,
        drug2_x: Tensor,
        drug2_edge_index: Tensor,
        drug2_batch: Tensor,
        cell: Tensor,
    ) -> Tensor:
        """Predict synergy-class logits for a drug-pair + cell-line triple.

        Parameters
        ----------
        drug1_x, drug1_edge_index, drug1_batch : Tensor
            First drug's atom features, COO edge index, and graph-batch
            vector (PyG convention).
        drug2_x, drug2_edge_index, drug2_batch : Tensor
            Second drug's atom features, COO edge index, and graph-batch
            vector.
        cell : Tensor
            Cell-line gene-expression vector, shape ``(batch, n_genes)``.

        Returns
        -------
        Tensor
            Synergy-class logits, shape ``(batch, n_classes)``.
        """
        x1 = self._encode_drug(drug1_x, drug1_edge_index, drug1_batch)
        x2 = self._encode_drug(drug2_x, drug2_edge_index, drug2_batch)

        cell_n = F.normalize(cell, p=2, dim=1)
        x_cell = self.cell_reduction(cell_n)

        xc = torch.cat([x1, x2, x_cell], dim=1)
        xc = F.normalize(xc, p=2, dim=1)
        xc = F.relu(self.fc1(xc))
        xc = self.dropout(xc)
        xc = F.relu(self.fc2(xc))
        xc = self.dropout(xc)
        xc = F.relu(self.fc3(xc))
        xc = self.dropout(xc)
        return self.out(xc)


def build_deepdds() -> nn.Module:
    """Build a compact DeepDDS shared-GAT drug-synergy classifier.

    Returns
    -------
    nn.Module
        Random-initialized ``DeepDDS`` in eval mode.
    """
    return DeepDDS().eval()


def example_input_deepdds() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Create an example drug-pair molecular-graph batch plus cell vector.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
        ``(drug1_x, drug1_edge_index, drug1_batch, drug2_x,
        drug2_edge_index, drug2_batch, cell)``: two independent 8-atom
        molecular graphs (12-dim atom features) batched to size 2 via
        ``torch_geometric.data.Batch``, plus a ``(2, 32)`` cell-line
        gene-expression vector.
    """

    def _rand_graph(n_atoms: int, feat_dim: int) -> Data:
        x = torch.randn(n_atoms, feat_dim)
        src = torch.randint(0, n_atoms, (n_atoms * 2,))
        dst = torch.randint(0, n_atoms, (n_atoms * 2,))
        edge_index = torch.stack([src, dst], dim=0)
        return Data(x=x, edge_index=edge_index)

    b, n_atoms, feat_dim = 2, 8, 12
    drug1_batch_obj = Batch.from_data_list([_rand_graph(n_atoms, feat_dim) for _ in range(b)])
    drug2_batch_obj = Batch.from_data_list([_rand_graph(n_atoms, feat_dim) for _ in range(b)])
    cell = torch.rand(b, 32)
    return (
        drug1_batch_obj.x,
        drug1_batch_obj.edge_index,
        drug1_batch_obj.batch,
        drug2_batch_obj.x,
        drug2_batch_obj.edge_index,
        drug2_batch_obj.batch,
        cell,
    )


MENAGERIE_ENTRIES = [
    ("DeepChrome", "build_deepchrome", "example_input_deepchrome", "2016", "BIO"),
    ("DeepCLIP", "build_deepclip", "example_input_deepclip", "2020", "BIO"),
    ("DeepCpf1", "build_deepcpf1", "example_input_deepcpf1", "2018", "BIO"),
    ("DeepCpG", "build_deepcpg", "example_input_deepcpg", "2017", "BIO"),
    ("DeepCRISPR", "build_deepcrispr", "example_input_deepcrispr", "2018", "BIO"),
    ("DeepDDS", "build_deepdds", "example_input_deepdds", "2022", "BIO"),
]
