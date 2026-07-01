"""Compact faithful classics for six structural-biology / cryo-EM architectures.

Sources checked (repo README/code inspected via web search + GitHub listing,
base env only, no clone or pip install):
  - DeepFoldRNA: https://github.com/robpearc/DeepFoldRNA. Pearce, Omenn &
    Zhang, "De Novo RNA Tertiary Structure Prediction at Atomic Resolution
    Using Geometric Potentials from Deep Learning", Nature Communications
    2023 (by way of the bioRxiv preprint). Fully-connected sequence
    embedding, transformer-style multi-head self-attention tower over the
    per-nucleotide token sequence (co-evolution-style pairwise features are
    normally concatenated in, here approximated by adding an outer-product
    pairwise bias into attention), producing a per-residue-pair 2D feature
    map that is decoded into discretized distance/orientation distributions
    (P-P, C4'-C4', N-N distances and torsion/orientation bins) used
    downstream as geometric restraints for structure minimization.
  - DeepFRI: https://github.com/flatironinstitute/DeepFRI. Gligorijevic
    et al., "Structure-based protein function prediction using graph
    convolutional networks", Nature Communications 2021. Two-stage model: a
    (frozen, here trainable-small) single-layer LSTM language model produces
    per-residue embeddings from the amino-acid sequence, which are then fed
    as node features into a 3-layer Graph Convolutional Network operating on
    the residue contact-map graph (adjacency from a distance threshold);
    the concatenation of all GCN layer outputs is pooled and passed through
    a sigmoid multi-label classifier head over GO-term-style function labels.
  - DeepH3: https://github.com/Graylab/deepH3-distances-orientations.
    Ruffolo, Guerra, Mahajan, Sulam & Gray, "Geometric potentials from deep
    learning improve prediction of CDR H3 loop structures", Bioinformatics
    2020 (ISMB). One-hot heavy+light chain sequence goes through a small
    stack of 1D residual conv blocks; the resulting per-residue features are
    combined pairwise (outer concatenation) into a 2D residue x residue
    feature map that is refined by a stack of dilated 2D residual conv
    blocks (RaptorX-style), then four parallel 1x1-conv heads predict
    binned inter-residue distance and three orientation-angle distributions.
  - DeepMainmast: https://github.com/kiharalab/DeepMainMast. Terashi, Wang,
    Christoffer, Zhu & Kihara (Kihara lab), "DeepMainmast: integrated
    protocol of protein structure modeling for cryo-EM with deep learning
    and structure prediction", Nature Methods 2024. The deep-learning stage
    (Emap2sf-style) is a small 3D CNN U-Net-like segmenter that scans a
    cryo-EM density map voxel grid and predicts, per voxel, a distribution
    over {backbone C-alpha present, amino-acid identity, no-atom}; those
    per-voxel class/type probabilities are what downstream main-chain
    tracing (a graph/VRP solver, not itself a network) consumes -- we
    reimplement the 3D-CNN voxel classifier, which is the trainable
    component of the pipeline.
  - DeepMetaPSICOV: https://github.com/psipred/DeepMetaPSICOV. Kandathil,
    Greener & Jones, "Prediction of inter-residue contacts with
    DeepMetaPSICOV in CASP13", Proteins 2019. Deep fully-convolutional 2D
    residual network: a wide multi-channel pairwise feature input (coupling
    maps, profile-derived features broadcast to 2D) is projected down by a
    1x1 "maxout-style" conv, then passed through a tower of residual blocks
    of two 5x5 dilated convolutions each, with the dilation rate cycling
    through {1,2,4,8,16,32,64} across the stack, ending in a 1x1 conv to a
    per-residue-pair contact-probability map.
  - DeepPicker: https://github.com/nejyeah/DeepPicker-python. Wang, Gong,
    Liu, Li, Yan, Xia, Li & Zeng, "DeepPicker: A deep learning approach for
    fully automated particle picking in cryo-EM", Journal of Structural
    Biology 2016 (arXiv:1605.01838). Small sliding-window CNN binary
    classifier: a handful of conv+max-pool blocks over a fixed-size cropped
    micrograph patch, flattened into a couple of fully-connected layers
    ending in a 2-way (particle / not-particle) softmax; applied
    convolutionally across a micrograph at inference to score candidate
    particle centers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# DeepFoldRNA: self-attention transformer predicting geometric restraints
# (distance/orientation distributions) for RNA tertiary structure folding.
# ---------------------------------------------------------------------------


class DeepFoldRNASelfAttention(nn.Module):
    """Single self-attention block with an additive pairwise attention bias."""

    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.pair_bias_proj = nn.Linear(dim, n_heads)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, seq_feat: torch.Tensor, pair_feat: torch.Tensor) -> torch.Tensor:
        """Attend over residues, biasing scores with the pairwise feature map.

        Parameters
        ----------
        seq_feat:
            Shape ``(batch, length, dim)`` per-residue features.
        pair_feat:
            Shape ``(batch, length, length, dim)`` pairwise features used as
            an additive attention bias (co-evolution-style restraint prior).

        Returns
        -------
        torch.Tensor
            Updated per-residue features, shape ``(batch, length, dim)``.
        """
        b, length, dim = seq_feat.shape
        qkv = self.qkv(seq_feat).reshape(b, length, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim**0.5)
        bias = self.pair_bias_proj(pair_feat).permute(0, 3, 1, 2)
        scores = scores + bias
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(b, length, dim)
        return self.norm(seq_feat + self.out_proj(out))


class DeepFoldRNA(nn.Module):
    """Self-attention transformer predicting RNA geometric restraints.

    Parameters
    ----------
    vocab_size:
        Number of nucleotide token types (A, C, G, U, + padding/mask).
    dim:
        Per-residue feature width.
    n_heads:
        Number of attention heads.
    n_layers:
        Number of self-attention blocks.
    n_bins:
        Number of discretized distance/orientation bins per output head.
    """

    def __init__(
        self,
        vocab_size: int = 6,
        dim: int = 32,
        n_heads: int = 4,
        n_layers: int = 3,
        n_bins: int = 16,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pair_init = nn.Linear(dim * 2, dim)
        self.layers = nn.ModuleList(
            [DeepFoldRNASelfAttention(dim, n_heads) for _ in range(n_layers)]
        )
        self.pair_update = nn.Linear(dim, dim)
        self.dist_head = nn.Conv2d(dim, n_bins, kernel_size=1)
        self.orient_head = nn.Conv2d(dim, n_bins, kernel_size=1)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict distance and orientation restraint distributions.

        Parameters
        ----------
        tokens:
            Integer nucleotide indices, shape ``(batch, length)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(distance_logits, orientation_logits)``, each shape
            ``(batch, n_bins, length, length)``.
        """
        seq_feat = self.embed(tokens)
        b, length, dim = seq_feat.shape
        left = seq_feat.unsqueeze(2).expand(b, length, length, dim)
        right = seq_feat.unsqueeze(1).expand(b, length, length, dim)
        pair_feat = self.pair_init(torch.cat([left, right], dim=-1))
        for layer in self.layers:
            seq_feat = layer(seq_feat, pair_feat)
            left = seq_feat.unsqueeze(2).expand(b, length, length, dim)
            right = seq_feat.unsqueeze(1).expand(b, length, length, dim)
            pair_feat = pair_feat + self.pair_update(left * right)
        pair_map = pair_feat.permute(0, 3, 1, 2)
        return self.dist_head(pair_map), self.orient_head(pair_map)


def build_deepfoldrna() -> nn.Module:
    """Construct a small DeepFoldRNA restraint-prediction transformer.

    Returns
    -------
    nn.Module
        DeepFoldRNA in eval mode.
    """
    return DeepFoldRNA(vocab_size=6, dim=32, n_heads=4, n_layers=3, n_bins=16).eval()


def example_input_deepfoldrna() -> torch.Tensor:
    """Example input for DeepFoldRNA: a batch of nucleotide token sequences.

    Returns
    -------
    torch.Tensor
        Shape ``(2, 24)`` integer nucleotide indices.
    """
    return torch.randint(0, 6, (2, 24))


# ---------------------------------------------------------------------------
# DeepFRI: LSTM sequence language model + 3-layer GCN over the residue
# contact-map graph for protein function (GO term) prediction.
# ---------------------------------------------------------------------------


class DeepFRIGraphConv(nn.Module):
    """One graph-convolution layer: normalized-adjacency message passing."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Propagate features across the (row-normalized) adjacency matrix.

        Parameters
        ----------
        x:
            Node features, shape ``(batch, n_nodes, in_dim)``.
        adj:
            Row-normalized adjacency, shape ``(batch, n_nodes, n_nodes)``.

        Returns
        -------
        torch.Tensor
            Updated node features, shape ``(batch, n_nodes, out_dim)``.
        """
        return F.relu(self.linear(torch.bmm(adj, x)))


class DeepFRI(nn.Module):
    """LSTM language-model embedding + 3-layer GCN protein function predictor.

    Parameters
    ----------
    vocab_size:
        Number of amino-acid token types.
    lstm_dim:
        LSTM hidden width (per-residue sequence embedding).
    gcn_dim:
        Width of each graph-convolution layer.
    n_labels:
        Number of GO-term-style output function labels.
    """

    def __init__(
        self,
        vocab_size: int = 21,
        lstm_dim: int = 32,
        gcn_dim: int = 32,
        n_labels: int = 10,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, lstm_dim)
        self.lm = nn.LSTM(lstm_dim, lstm_dim, batch_first=True, bidirectional=True)
        self.lm_proj = nn.Linear(lstm_dim * 2, gcn_dim)
        self.gcn1 = DeepFRIGraphConv(gcn_dim, gcn_dim)
        self.gcn2 = DeepFRIGraphConv(gcn_dim, gcn_dim)
        self.gcn3 = DeepFRIGraphConv(gcn_dim, gcn_dim)
        self.classifier = nn.Linear(gcn_dim * 3, n_labels)

    def forward(self, tokens: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Predict multi-label GO-term function probabilities.

        Parameters
        ----------
        tokens:
            Amino-acid token indices, shape ``(batch, n_residues)``.
        adj:
            Row-normalized residue contact-map adjacency, shape
            ``(batch, n_residues, n_residues)``.

        Returns
        -------
        torch.Tensor
            Sigmoid function-label probabilities, shape ``(batch, n_labels)``.
        """
        seq_feat, _ = self.lm(self.embed(tokens))
        node_feat = self.lm_proj(seq_feat)
        h1 = self.gcn1(node_feat, adj)
        h2 = self.gcn2(h1, adj)
        h3 = self.gcn3(h2, adj)
        pooled = torch.cat([h1, h2, h3], dim=-1).mean(dim=1)
        return torch.sigmoid(self.classifier(pooled))


def build_deepfri() -> nn.Module:
    """Construct a small DeepFRI GCN protein-function predictor.

    Returns
    -------
    nn.Module
        DeepFRI in eval mode.
    """
    return DeepFRI(vocab_size=21, lstm_dim=32, gcn_dim=32, n_labels=10).eval()


def example_input_deepfri() -> tuple[torch.Tensor, torch.Tensor]:
    """Example input for DeepFRI: token sequence plus contact-map adjacency.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(tokens, adjacency)`` with shapes ``(2, 20)`` and ``(2, 20, 20)``;
        the adjacency is row-normalized so each row sums to one.
    """
    tokens = torch.randint(0, 21, (2, 20))
    raw_adj = (torch.rand(2, 20, 20) > 0.7).float()
    raw_adj = raw_adj + torch.eye(20).unsqueeze(0)
    adj = raw_adj / raw_adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
    return tokens, adj


# ---------------------------------------------------------------------------
# DeepH3: 1D residual conv tower -> pairwise outer-concat -> 2D dilated
# residual conv tower -> four parallel distance/orientation output heads.
# ---------------------------------------------------------------------------


class DeepH3ResBlock1D(nn.Module):
    """1D residual block: two conv+BN+ReLU layers with a skip connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block.

        Parameters
        ----------
        x:
            Shape ``(batch, channels, length)``.

        Returns
        -------
        torch.Tensor
            Same shape as ``x``.
        """
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(x + h)


class DeepH3ResBlock2D(nn.Module):
    """2D dilated residual block: two dilated conv+BN+ReLU layers."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the dilated residual block.

        Parameters
        ----------
        x:
            Shape ``(batch, channels, length, length)``.

        Returns
        -------
        torch.Tensor
            Same shape as ``x``.
        """
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(x + h)


class DeepH3(nn.Module):
    """1D->2D residual conv network predicting CDR-H3 distance/orientations.

    Parameters
    ----------
    vocab_size:
        Number of amino-acid one-hot channels.
    channels_1d:
        Width of the 1D residual tower.
    channels_2d:
        Width of the 2D residual tower.
    n_blocks_1d:
        Number of 1D residual blocks.
    n_blocks_2d:
        Number of 2D dilated residual blocks.
    n_bins:
        Number of discretized bins per output head.
    """

    def __init__(
        self,
        vocab_size: int = 21,
        channels_1d: int = 16,
        channels_2d: int = 24,
        n_blocks_1d: int = 3,
        n_blocks_2d: int = 4,
        n_bins: int = 16,
    ) -> None:
        super().__init__()
        self.in_conv = nn.Conv1d(vocab_size, channels_1d, kernel_size=1)
        self.blocks_1d = nn.ModuleList([DeepH3ResBlock1D(channels_1d) for _ in range(n_blocks_1d)])
        self.pair_proj = nn.Conv2d(channels_1d * 2, channels_2d, kernel_size=1)
        dilations = [1, 2, 4, 8]
        self.blocks_2d = nn.ModuleList(
            [
                DeepH3ResBlock2D(channels_2d, dilations[i % len(dilations)])
                for i in range(n_blocks_2d)
            ]
        )
        self.dist_head = nn.Conv2d(channels_2d, n_bins, kernel_size=1)
        self.omega_head = nn.Conv2d(channels_2d, n_bins, kernel_size=1)
        self.theta_head = nn.Conv2d(channels_2d, n_bins, kernel_size=1)
        self.phi_head = nn.Conv2d(channels_2d, n_bins, kernel_size=1)

    def forward(self, seq_onehot: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Predict distance and orientation-angle logit maps.

        Parameters
        ----------
        seq_onehot:
            One-hot heavy+light chain sequence, shape
            ``(batch, vocab_size, length)``.

        Returns
        -------
        tuple[torch.Tensor, ...]
            ``(distance_logits, omega_logits, theta_logits, phi_logits)``,
            each shape ``(batch, n_bins, length, length)``.
        """
        h = self.in_conv(seq_onehot)
        for block in self.blocks_1d:
            h = block(h)
        b, c, length = h.shape
        left = h.unsqueeze(3).expand(b, c, length, length)
        right = h.unsqueeze(2).expand(b, c, length, length)
        pair = self.pair_proj(torch.cat([left, right], dim=1))
        for block in self.blocks_2d:
            pair = block(pair)
        return (
            self.dist_head(pair),
            self.omega_head(pair),
            self.theta_head(pair),
            self.phi_head(pair),
        )


def build_deeph3() -> nn.Module:
    """Construct a small DeepH3 CDR-H3 distance/orientation predictor.

    Returns
    -------
    nn.Module
        DeepH3 in eval mode.
    """
    return DeepH3(
        vocab_size=21,
        channels_1d=16,
        channels_2d=24,
        n_blocks_1d=3,
        n_blocks_2d=4,
        n_bins=16,
    ).eval()


def example_input_deeph3() -> torch.Tensor:
    """Example input for DeepH3: one-hot heavy+light chain sequence.

    Returns
    -------
    torch.Tensor
        Shape ``(2, 21, 30)`` one-hot amino-acid sequence.
    """
    idx = torch.randint(0, 21, (2, 30))
    return F.one_hot(idx, num_classes=21).permute(0, 2, 1).float()


# ---------------------------------------------------------------------------
# DeepMainmast (Emap2sf-style voxel classifier): 3D CNN scanning a cryo-EM
# density-map grid, predicting per-voxel C-alpha/amino-acid-type class
# probabilities that feed the downstream (non-network) main-chain tracer.
# ---------------------------------------------------------------------------


class DeepMainmastVoxelNet(nn.Module):
    """Small 3D CNN encoder-decoder classifying cryo-EM density voxels.

    Parameters
    ----------
    in_channels:
        Number of input density-map channels (typically 1: raw density).
    base_channels:
        Width of the first conv stage.
    n_classes:
        Number of per-voxel output classes (background/backbone-C-alpha/
        amino-acid-type bins).
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 8, n_classes: int = 22) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.up = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv3d(base_channels, n_classes, kernel_size=1)

    def forward(self, density: torch.Tensor) -> torch.Tensor:
        """Classify each voxel of a cryo-EM density-map crop.

        Parameters
        ----------
        density:
            Shape ``(batch, in_channels, depth, height, width)``.

        Returns
        -------
        torch.Tensor
            Per-voxel class logits, shape
            ``(batch, n_classes, depth, height, width)``.
        """
        e1 = self.enc1(density)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        u = self.up(b)
        d1 = self.dec1(torch.cat([u, e1], dim=1))
        return self.out_conv(d1)


def build_deepmainmast() -> nn.Module:
    """Construct a small DeepMainmast (Emap2sf-style) voxel classifier.

    Returns
    -------
    nn.Module
        DeepMainmastVoxelNet in eval mode.
    """
    return DeepMainmastVoxelNet(in_channels=1, base_channels=8, n_classes=22).eval()


def example_input_deepmainmast() -> torch.Tensor:
    """Example input for DeepMainmast: a cropped cryo-EM density-map grid.

    Returns
    -------
    torch.Tensor
        Shape ``(1, 1, 16, 16, 16)`` single-channel density crop.
    """
    return torch.randn(1, 1, 16, 16, 16)


# ---------------------------------------------------------------------------
# DeepMetaPSICOV: deep fully-convolutional 2D residual network for
# inter-residue contact prediction, with a dilation cycle of
# {1, 2, 4, 8, 16, 32, 64} across the residual-block tower.
# ---------------------------------------------------------------------------


class DeepMetaPSICOVResBlock(nn.Module):
    """Residual block of two 5x5 dilated convolutions."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        pad = 2 * dilation
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=5, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=5, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the dilated residual block.

        Parameters
        ----------
        x:
            Shape ``(batch, channels, length, length)``.

        Returns
        -------
        torch.Tensor
            Same shape as ``x``.
        """
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(x + h)


class DeepMetaPSICOV(nn.Module):
    """Deep dilated-residual 2D ConvNet for residue-residue contact prediction.

    Parameters
    ----------
    in_channels:
        Number of stacked pairwise feature channels (co-evolution coupling
        maps + broadcast profile features).
    width:
        Number of channels in the residual tower (post maxout-style
        1x1 reduction).
    n_blocks:
        Number of residual blocks; the dilation rate cycles through
        ``(1, 2, 4, 8, 16, 32, 64)`` across the tower.
    """

    def __init__(self, in_channels: int = 40, width: int = 24, n_blocks: int = 7) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, width, kernel_size=1)
        dilations = [1, 2, 4, 8, 16, 32, 64]
        self.blocks = nn.ModuleList(
            [DeepMetaPSICOVResBlock(width, dilations[i % len(dilations)]) for i in range(n_blocks)]
        )
        self.out_conv = nn.Conv2d(width, 1, kernel_size=1)

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        """Predict a residue-pair contact-probability map.

        Parameters
        ----------
        pair_features:
            Stacked pairwise feature maps, shape
            ``(batch, in_channels, length, length)``.

        Returns
        -------
        torch.Tensor
            Contact probabilities, shape ``(batch, 1, length, length)``.
        """
        h = F.relu(self.reduce(pair_features))
        for block in self.blocks:
            h = block(h)
        return torch.sigmoid(self.out_conv(h))


def build_deepmetapsicov() -> nn.Module:
    """Construct a small DeepMetaPSICOV contact-map predictor.

    Returns
    -------
    nn.Module
        DeepMetaPSICOV in eval mode.
    """
    return DeepMetaPSICOV(in_channels=40, width=24, n_blocks=7).eval()


def example_input_deepmetapsicov() -> torch.Tensor:
    """Example input for DeepMetaPSICOV: stacked pairwise feature maps.

    Returns
    -------
    torch.Tensor
        Shape ``(1, 40, 30, 30)`` pairwise coupling/profile feature maps.
    """
    return torch.randn(1, 40, 30, 30)


# ---------------------------------------------------------------------------
# DeepPicker: small sliding-window CNN binary classifier scoring cryo-EM
# micrograph patches as particle / not-particle.
# ---------------------------------------------------------------------------


class DeepPicker(nn.Module):
    """Sliding-window CNN classifier for cryo-EM particle picking.

    Parameters
    ----------
    in_channels:
        Number of input micrograph channels (1: raw grayscale density).
    patch_size:
        Side length of the square input patch in pixels.
    """

    def __init__(self, in_channels: int = 1, patch_size: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=9, padding=4)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        reduced = patch_size // 8
        self.fc1 = nn.Linear(32 * reduced * reduced, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        """Classify a cropped micrograph patch as particle / not-particle.

        Parameters
        ----------
        patch:
            Shape ``(batch, in_channels, patch_size, patch_size)``.

        Returns
        -------
        torch.Tensor
            Class log-probabilities, shape ``(batch, 2)``.
        """
        h = self.pool1(F.relu(self.conv1(patch)))
        h = self.pool2(F.relu(self.conv2(h)))
        h = self.pool3(F.relu(self.conv3(h)))
        h = h.flatten(1)
        h = F.relu(self.fc1(h))
        return F.log_softmax(self.fc2(h), dim=-1)


def build_deeppicker() -> nn.Module:
    """Construct a small DeepPicker particle/not-particle patch classifier.

    Returns
    -------
    nn.Module
        DeepPicker in eval mode.
    """
    return DeepPicker(in_channels=1, patch_size=64).eval()


def example_input_deeppicker() -> torch.Tensor:
    """Example input for DeepPicker: a batch of cropped micrograph patches.

    Returns
    -------
    torch.Tensor
        Shape ``(4, 1, 64, 64)`` single-channel micrograph patches.
    """
    return torch.randn(4, 1, 64, 64)


MENAGERIE_ENTRIES = [
    ("DeepFoldRNA", "build_deepfoldrna", "example_input_deepfoldrna", "2023", "BIO"),
    ("DeepFRI", "build_deepfri", "example_input_deepfri", "2021", "BIO"),
    ("DeepH3", "build_deeph3", "example_input_deeph3", "2020", "BIO"),
    ("DeepMainmast", "build_deepmainmast", "example_input_deepmainmast", "2024", "BIO"),
    ("DeepMetaPSICOV", "build_deepmetapsicov", "example_input_deepmetapsicov", "2019", "BIO"),
    ("DeepPicker", "build_deeppicker", "example_input_deeppicker", "2016", "BIO"),
]
