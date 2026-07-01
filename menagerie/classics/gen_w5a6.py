"""Six faithful, compact reimplementations of named bio-imaging/bioinformatics models.

Sources checked (repo code and/or paper, no clone/pip-install; base-env torch
reimplementation of the distinctive mechanism):
  - Mesmer:     https://github.com/vanvalenlab/deepcell-tf
                (deepcell/model_zoo/panopticnet.py, deepcell/model_zoo/fpn.py).
                Greenwald et al., Nature Biotechnology 2022. TensorFlow/Keras "PanopticNet"
                architecture: a shared CNN backbone feeds a Feature Pyramid Network (FPN,
                lateral 1x1 projections + top-down 2x nearest-neighbor upsample-and-add
                across levels C3..C5 producing P3..P7), and TWO semantic segmentation heads
                branch off the shared pyramid -- one for nuclear predictions, one for
                whole-cell predictions (Mesmer's defining "two-headed" multiplexed-tissue
                design vs. single-head predecessors like DeepCell/Mask R-CNN). Each head
                upsamples/refines pyramid features back to a 2x-downsampled resolution and
                emits a 4-channel map (inner-distance transform + outer-distance transform +
                foreground/background + pixelwise class), reimplemented here as same-shaped
                conv-refine-then-upsample blocks per head.
  - MetaNN:     https://github.com/ChiehLo/MetaNN (DataSet/IBD/Code/classifier/NN_IBD_cuda.py,
                class ``CNN1d``). Lo & Marculescu, BMC Bioinformatics 2019. The repo's
                headline classifier (compared against a plain-MLP baseline and paired with
                a VAE-based data-augmentation pipeline for low-sample-size host-phenotype
                classification) is a 1D CNN read directly over the raw microbial relative-
                abundance vector (treated as a 1-channel signal): two strided
                Conv1d+ReLU+Dropout stages each followed by max-pooling, flattened into a
                linear classification head -- the distinctive "treat OTU/species abundance
                as an ordered 1D signal for convolution" idea vs. a fully-connected MLP.
  - MetaPheno:  https://github.com/nlapier2/metapheno (classify.py:
                ``build_and_fit_autoencoder`` / ``autoencoder_pretrain`` /
                ``build_and_fit_model``). LaPierre et al., Frontiers in Genetics 2019. The
                paper's flagship pipeline greedily halves a symmetric autoencoder's width
                per layer to pretrain an unsupervised feature transform over k-mer/taxonomic
                abundance vectors, keeps only the trained ENCODER weights (applied as a
                linear projection chain, no bias/activation reapplied at inference in the
                original weight-extraction code), then classifies the transformed features
                with a feedforward network whose hidden width linearly shrinks toward the
                output, alternating relu/tanh activations with dropout, ending in a sigmoid.
  - MGraphDTA:  https://github.com/guaguabujianle/MGraphDTA (regression/model.py: classes
                ``TargetRepresentation``/``StackCNN``, ``GraphDenseNet``/``DenseBlock``,
                ``MGraphDTA``). Yang et al., Chemical Science 2022. Drug-target binding
                affinity via TWO multi-scale encoders fused by concatenation: (1) the
                protein sequence goes through parallel ``StackCNN`` towers of INCREASING
                conv depth (1, 2, ..., block_num stacked Conv1d+ReLU layers, each ending in
                adaptive max-pool) so each tower captures a different receptive-field
                "scale", concatenated and projected; (2) the ligand's molecular graph goes
                through a DenseNet-style GNN (``GraphConv`` + node-level BatchNorm,
                densely-concatenated feature growth across blocks, 1x1-conv-style
                transition layers halving channel count between blocks) pooled by global
                mean pooling. The two pooled representations are concatenated and scored
                by an MLP -- the "multiscale" (protein) + "densely-connected graph"
                (ligand) combination is MGraphDTA's namesake mechanism.
  - MHCflurry:  https://github.com/openvax/mhcflurry (mhcflurry/pytorch_layers.py class
                ``LocallyConnected1D``; mhcflurry/class1_neural_network.py class
                ``Class1NeuralNetworkModel``). O'Donnell et al., Cell Systems 2018 (v2.2+
                ships a PyTorch backend). Pan-allele MHC-I binding-affinity predictor: a
                fixed-length one-hot/BLOSUM-encoded peptide is passed through one or more
                ``LocallyConnected1D`` layers -- an UNSHARED-WEIGHT 1D convolution (a
                distinct weight tensor per output position, via ``einsum`` over unfolded
                windows) that lets the network learn position-specific peptide motifs
                without full weight tying -- then flattened into peptide-side dense layers;
                separately a (frozen) allele pseudosequence embedding is projected through
                its own dense layers; the two branches are merged by ELEMENTWISE MULTIPLY
                (not concatenation -- MHCflurry's chosen fusion mechanism) before a final
                dense stack and a sigmoid affinity-transform output.
  - MiSiC:      https://github.com/pswapnesh/misic (misic/misic.py:
                ``MiSiC.shapeindex_preprocess`` + Keras U-Net loaded from
                ``MiSiCv2.h5``, no architecture source shipped -- weights-only). Panigrahi
                et al., eLife 2021 (https://elifesciences.org/articles/65151). MiSiC's
                distinctive contribution is the PREPROCESSING + input representation: raw
                microscopy images are converted to a 3-channel "shape index" map (local
                surface-curvature descriptor per Koenderink & van Doorn, computed here at
                three Gaussian smoothing scales) that is scale- and contrast-invariant
                across microscopy modalities (phase-contrast, fluorescence, different
                magnifications) -- reimplemented here as a differentiable multi-scale
                shape-index approximation (Sobel-based local Hessian eigen-curvature proxy
                at 3 scales) feeding a standard compact U-Net encoder-decoder that emits a
                3-class (background / cell body / cell boundary) segmentation map, matching
                MiSiC's documented "shape-index maps -> U-Net -> categorical segmentation"
                pipeline.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Mesmer -- two-headed (nuclear + whole-cell) FPN semantic-segmentation network
# ---------------------------------------------------------------------------


class _FPNBackbone(nn.Module):
    """Compact strided-conv backbone producing 3 feature levels (stand-in for ResNet C3-C5)."""

    def __init__(self, in_channels: int, widths: tuple[int, int, int]) -> None:
        """Build a 3-stage strided-conv feature extractor.

        Parameters
        ----------
        in_channels:
            Number of input image channels.
        widths:
            Channel widths of the three output feature levels (C3, C4, C5).
        """
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], 3, stride=2, padding=1), nn.ReLU(inplace=True)
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(widths[0], widths[0], 3, stride=2, padding=1), nn.ReLU(inplace=True)
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(widths[0], widths[1], 3, stride=2, padding=1), nn.ReLU(inplace=True)
        )
        self.stage5 = nn.Sequential(
            nn.Conv2d(widths[1], widths[2], 3, stride=2, padding=1), nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Extract three progressively downsampled, progressively wider feature maps.

        Parameters
        ----------
        x:
            Input image, shape ``(B, C, H, W)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(C3, C4, C5)`` feature maps at strides 4, 8, 16.
        """
        x = self.stem(x)
        c3 = self.stage3(x)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return c3, c4, c5


class _SemanticHead(nn.Module):
    """FPN semantic segmentation head: refine + upsample a fused pyramid feature to full res."""

    def __init__(self, feature_size: int, n_classes: int, n_upsamples: int) -> None:
        """Build a conv-refine-then-upsample head.

        Parameters
        ----------
        feature_size:
            Channel width of the fused pyramid feature it consumes.
        n_classes:
            Number of output channels (e.g. distance-transform + fg/bg + class map).
        n_upsamples:
            Number of 2x nearest-neighbor upsampling stages back toward input resolution.
        """
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(feature_size, feature_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_size, feature_size // 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.n_upsamples = n_upsamples
        self.out = nn.Conv2d(feature_size // 2, n_classes, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Refine and upsample a fused pyramid feature into a per-pixel prediction map.

        Parameters
        ----------
        x:
            Fused pyramid feature, shape ``(B, feature_size, H, W)``.

        Returns
        -------
        Tensor
            Prediction map, shape ``(B, n_classes, H * 2**n_upsamples, W * 2**n_upsamples)``.
        """
        x = self.refine(x)
        for _ in range(self.n_upsamples):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.out(x)


class Mesmer(nn.Module):
    """PanopticNet: shared FPN backbone with two semantic segmentation heads.

    Reimplements deepcell-tf's ``PanopticNet``: a shared backbone produces multi-scale
    features (C3-C5); a Feature Pyramid Network fuses them top-down (1x1 lateral
    projection + nearest-neighbor 2x upsample-and-add) into a single pyramid level; two
    INDEPENDENT semantic heads (nuclear, whole-cell) each refine and upsample that fused
    feature back toward input resolution, producing Mesmer's two co-registered
    segmentation predictions from one shared trunk.
    """

    def __init__(
        self,
        in_channels: int = 2,
        backbone_widths: tuple[int, int, int] = (16, 24, 32),
        fpn_channels: int = 24,
        n_classes: int = 4,
    ) -> None:
        """Build the shared FPN backbone and two semantic heads.

        Parameters
        ----------
        in_channels:
            Number of input imaging channels (Mesmer uses 2: nuclear + membrane marker).
        backbone_widths:
            Channel widths of the backbone's three feature levels.
        fpn_channels:
            Channel width of the fused FPN feature.
        n_classes:
            Number of output channels per semantic head.
        """
        super().__init__()
        self.backbone = _FPNBackbone(in_channels, backbone_widths)
        self.lateral3 = nn.Conv2d(backbone_widths[0], fpn_channels, 1)
        self.lateral4 = nn.Conv2d(backbone_widths[1], fpn_channels, 1)
        self.lateral5 = nn.Conv2d(backbone_widths[2], fpn_channels, 1)
        self.smooth = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)

        self.nuclear_head = _SemanticHead(fpn_channels, n_classes, n_upsamples=2)
        self.whole_cell_head = _SemanticHead(fpn_channels, n_classes, n_upsamples=2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Predict co-registered nuclear and whole-cell semantic segmentation maps.

        Parameters
        ----------
        x:
            Multiplex tissue image, shape ``(B, in_channels, H, W)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(nuclear_pred, whole_cell_pred)``, each shape ``(B, n_classes, H, W)``.
        """
        c3, c4, c5 = self.backbone(x)

        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        fused = self.smooth(p3)

        nuclear = self.nuclear_head(fused)
        whole_cell = self.whole_cell_head(fused)
        return nuclear, whole_cell


def build_mesmer() -> nn.Module:
    """Construct a small Mesmer two-headed PanopticNet.

    Returns
    -------
    nn.Module
        Mesmer in eval mode.
    """
    return Mesmer(in_channels=2, backbone_widths=(16, 24, 32), fpn_channels=24, n_classes=4).eval()


def example_input_mesmer() -> Tensor:
    """Example input for Mesmer: a 2-channel (nuclear + membrane) multiplex tissue image.

    Returns
    -------
    Tensor
        Shape ``(1, 2, 64, 64)``.
    """
    return torch.randn(1, 2, 64, 64)


# ---------------------------------------------------------------------------
# MetaNN -- 1D CNN over microbial relative-abundance vectors
# ---------------------------------------------------------------------------


class MetaNN(nn.Module):
    """1D convolutional host-phenotype classifier over microbiome abundance vectors.

    Reimplements the repo's ``CNN1d`` (``NN_IBD_cuda.py``): the raw OTU/species
    relative-abundance vector is treated as a single-channel 1D signal and passed through
    two strided Conv1d+ReLU (+dropout on the first) stages, each followed by max-pooling,
    then flattened into a linear classification head -- convolving directly over
    abundance-vector order rather than the fully-connected MLP baseline it beats.
    """

    def __init__(self, n_features: int = 256, n_classes: int = 2, dropout: float = 0.1) -> None:
        """Build the two-stage strided-conv + max-pool feature extractor and linear head.

        Parameters
        ----------
        n_features:
            Length of the input microbial relative-abundance vector.
        n_classes:
            Number of host-phenotype classes.
        dropout:
            Dropout probability applied after the first convolution.
        """
        super().__init__()
        self.c1 = nn.Conv1d(1, 8, kernel_size=3, stride=2, padding=1)
        self.c2 = nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1)
        self.p1 = nn.MaxPool1d(2)
        self.p2 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(p=dropout)
        flat_len = n_features
        for _ in range(4):  # two (stride-2 conv, then pool-2) stages
            flat_len = flat_len // 2
        self.out = nn.Linear(flat_len * 8, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Classify host phenotype from a microbial abundance vector.

        Parameters
        ----------
        x:
            Relative-abundance vector, shape ``(B, 1, n_features)``.

        Returns
        -------
        Tensor
            Class logits, shape ``(B, n_classes)``.
        """
        x = F.relu(self.dropout1(self.c1(x)))
        x = self.p1(x)
        x = F.relu(self.c2(x))
        x = self.p2(x)
        x = x.reshape(x.size(0), -1)
        return self.out(x)


def build_metann() -> nn.Module:
    """Construct a small MetaNN 1D-CNN classifier.

    Returns
    -------
    nn.Module
        MetaNN in eval mode.
    """
    return MetaNN(n_features=256, n_classes=2, dropout=0.0).eval()


def example_input_metann() -> Tensor:
    """Example input for MetaNN: a batch of microbial relative-abundance vectors.

    Returns
    -------
    Tensor
        Shape ``(8, 1, 256)``.
    """
    return torch.randn(8, 1, 256)


# ---------------------------------------------------------------------------
# MetaPheno -- greedy-shrinking autoencoder pretraining + shrinking-width MLP
# ---------------------------------------------------------------------------


class GreedyShrinkingAutoencoder(nn.Module):
    """Symmetric autoencoder whose width halves per encoder layer (MetaPheno pretraining).

    Mirrors ``build_and_fit_autoencoder``: encoder layer sizes are
    ``input_dim / 2, input_dim / 4, ...``; the decoder mirrors them back up. Only the
    encoder half is used downstream (``autoencoder_pretrain`` keeps just the trained
    encoder weights as a feature-transform projection).
    """

    def __init__(self, input_dim: int, n_layers: int = 2) -> None:
        """Build the greedy-shrinking encoder/decoder stack.

        Parameters
        ----------
        input_dim:
            Dimensionality of the input k-mer/taxonomic abundance vector.
        n_layers:
            Number of encoder (and mirrored decoder) layers.
        """
        super().__init__()
        sizes = [max(input_dim // (2**i), 1) for i in range(n_layers + 1)]
        enc: list[nn.Module] = []
        for i in range(n_layers):
            enc.append(nn.Linear(sizes[i], sizes[i + 1]))
            enc.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc)

        dec: list[nn.Module] = []
        rev = list(reversed(sizes))
        for i in range(n_layers):
            dec.append(nn.Linear(rev[i], rev[i + 1]))
            dec.append(nn.ReLU())
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode then reconstruct an abundance vector.

        Parameters
        ----------
        x:
            Input abundance vector, shape ``(B, input_dim)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(encoded, reconstruction)``.
        """
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon


class MetaPheno(nn.Module):
    """Autoencoder-pretrained, linearly-shrinking-width phenotype classifier.

    Reimplements the ``run_autonn`` pipeline: a :class:`GreedyShrinkingAutoencoder`
    pretransforms the raw feature vector into its encoded (bottleneck) representation;
    ``build_and_fit_model`` then classifies that representation with a feedforward
    network whose hidden width shrinks linearly toward the output, alternating relu
    (first layer) then tanh (hidden layers) activations with dropout, ending in a
    sigmoid binary-phenotype score.
    """

    def __init__(
        self,
        input_dim: int = 128,
        auto_layers: int = 2,
        fc_layers: int = 3,
        dropout: float = 0.25,
    ) -> None:
        """Build the autoencoder feature transform and the shrinking-width MLP classifier.

        Parameters
        ----------
        input_dim:
            Dimensionality of the input k-mer/taxonomic abundance vector.
        auto_layers:
            Number of greedy-shrinking autoencoder encoder layers.
        fc_layers:
            Number of shrinking-width hidden layers in the classifier.
        dropout:
            Dropout probability (keep-probability complement) in the classifier.
        """
        super().__init__()
        self.autoencoder = GreedyShrinkingAutoencoder(input_dim, n_layers=auto_layers)
        encoded_dim = max(input_dim // (2**auto_layers), 1)

        layer_scale = 1.0 / float(fc_layers + 1)
        layers: list[nn.Module] = [nn.Linear(encoded_dim, encoded_dim), nn.ReLU()]
        cur = encoded_dim
        for i in range(fc_layers):
            nxt = max(cur - int(encoded_dim * (layer_scale * (i + 1))), 1)
            layers.append(nn.Linear(cur, nxt))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=dropout))
            cur = nxt
        self.classifier = nn.Sequential(*layers)
        self.output = nn.Linear(cur, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Predict a binary host-phenotype probability from a raw abundance vector.

        Parameters
        ----------
        x:
            Raw k-mer/taxonomic abundance vector, shape ``(B, input_dim)``.

        Returns
        -------
        Tensor
            Phenotype probability, shape ``(B, 1)``.
        """
        z, _ = self.autoencoder(x)
        hidden = self.classifier(z)
        return torch.sigmoid(self.output(hidden))


def build_metapheno() -> nn.Module:
    """Construct a small MetaPheno autoencoder+MLP classifier.

    Returns
    -------
    nn.Module
        MetaPheno in eval mode.
    """
    return MetaPheno(input_dim=128, auto_layers=2, fc_layers=3, dropout=0.0).eval()


def example_input_metapheno() -> Tensor:
    """Example input for MetaPheno: a batch of k-mer/taxonomic abundance vectors.

    Returns
    -------
    Tensor
        Shape ``(6, 128)``.
    """
    return torch.randn(6, 128)


# ---------------------------------------------------------------------------
# MGraphDTA -- multiscale protein CNN towers + DenseNet-style ligand GNN
# ---------------------------------------------------------------------------


class _StackCNN(nn.Module):
    """A tower of ``layer_num`` stacked Conv1d+ReLU layers ending in adaptive max-pool.

    One tower per "scale" in :class:`_TargetRepresentation`: tower ``i`` has ``i + 1``
    stacked conv layers, giving each tower a different receptive field.
    """

    def __init__(
        self, layer_num: int, in_channels: int, out_channels: int, kernel_size: int
    ) -> None:
        """Build a stack of same-width Conv1d+ReLU layers followed by adaptive max-pool.

        Parameters
        ----------
        layer_num:
            Number of stacked conv layers (this tower's "scale").
        in_channels:
            Number of input channels (embedding dimension).
        out_channels:
            Number of channels throughout the stack.
        kernel_size:
            Convolution kernel size (padding chosen to preserve length).
        """
        super().__init__()
        pad = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad),
            nn.ReLU(),
        ]
        for _ in range(layer_num - 1):
            layers.append(nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad))
            layers.append(nn.ReLU())
        layers.append(nn.AdaptiveMaxPool1d(1))
        self.inc = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Extract a fixed-size pooled feature at this tower's receptive-field scale.

        Parameters
        ----------
        x:
            Embedded protein sequence, shape ``(B, in_channels, L)``.

        Returns
        -------
        Tensor
            Pooled feature, shape ``(B, out_channels)``.
        """
        return self.inc(x).squeeze(-1)


class _TargetRepresentation(nn.Module):
    """Multi-scale protein sequence encoder: parallel :class:`_StackCNN` towers, concatenated."""

    def __init__(
        self, block_num: int, vocab_size: int, embedding_num: int, filter_num: int = 32
    ) -> None:
        """Build the embedding and parallel multi-scale conv towers.

        Parameters
        ----------
        block_num:
            Number of parallel towers (also the max conv depth of the deepest tower).
        vocab_size:
            Amino-acid vocabulary size.
        embedding_num:
            Amino-acid embedding dimensionality.
        filter_num:
            Channel width shared by every tower.
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_num, padding_idx=0)
        self.block_list = nn.ModuleList(
            [_StackCNN(i + 1, embedding_num, filter_num, kernel_size=3) for i in range(block_num)]
        )
        self.linear = nn.Linear(block_num * filter_num, filter_num)

    def forward(self, x: Tensor) -> Tensor:
        """Encode a protein sequence via multi-scale conv towers.

        Parameters
        ----------
        x:
            Amino-acid index sequence, shape ``(B, L)``.

        Returns
        -------
        Tensor
            Protein representation, shape ``(B, filter_num)``.
        """
        x = self.embed(x).permute(0, 2, 1)
        feats = [block(x) for block in self.block_list]
        return self.linear(torch.cat(feats, dim=-1))


class _GraphConvBnRelu(nn.Module):
    """Graph convolution (mean-neighbor-aggregation) + batchnorm + ReLU over node features."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Build one GraphConv+BatchNorm+ReLU block.

        Parameters
        ----------
        in_channels:
            Input node-feature dimensionality.
        out_channels:
            Output node-feature dimensionality.
        """
        super().__init__()
        self.self_lin = nn.Linear(in_channels, out_channels)
        self.neigh_lin = nn.Linear(in_channels, out_channels)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """Aggregate mean-neighbor features and combine with self features.

        Parameters
        ----------
        x:
            Node features, shape ``(B, N, in_channels)``.
        adj:
            Dense adjacency matrix, shape ``(B, N, N)``.

        Returns
        -------
        Tensor
            Updated node features, shape ``(B, N, out_channels)``.
        """
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        neigh = torch.bmm(adj, x) / deg
        out = self.self_lin(x) + self.neigh_lin(neigh)
        b, n, c = out.shape
        out = self.norm(out.reshape(b * n, c)).reshape(b, n, c)
        return F.relu(out)


class _DenseBlock(nn.Module):
    """Densely-connected stack of :class:`_GraphConvBnRelu` layers (feature-growth concat)."""

    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int) -> None:
        """Build a dense block of graph-conv layers with concatenated feature growth.

        Parameters
        ----------
        num_layers:
            Number of densely-connected graph-conv layers.
        num_input_features:
            Node-feature width entering the block.
        growth_rate:
            Number of new channels each layer contributes.
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _GraphConvBnRelu(num_input_features + i * growth_rate, growth_rate)
                for i in range(num_layers)
            ]
        )

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """Densely stack graph-conv features.

        Parameters
        ----------
        x:
            Node features, shape ``(B, N, num_input_features)``.
        adj:
            Dense adjacency matrix, shape ``(B, N, N)``.

        Returns
        -------
        Tensor
            Densely-concatenated node features.
        """
        features = [x]
        for layer in self.layers:
            new_feat = layer(torch.cat(features, dim=-1), adj)
            features.append(new_feat)
        return torch.cat(features, dim=-1)


class _GraphDenseNet(nn.Module):
    """DenseNet-style ligand molecular-graph encoder: dense blocks + halving transitions."""

    def __init__(
        self,
        num_input_features: int,
        out_dim: int,
        growth_rate: int = 16,
        block_config: tuple[int, ...] = (3, 3, 3),
    ) -> None:
        """Build the initial conv, dense-block/transition cascade, and classifier.

        Parameters
        ----------
        num_input_features:
            Input node-feature dimensionality.
        out_dim:
            Output ligand-representation dimensionality.
        growth_rate:
            Per-layer channel growth inside each dense block.
        block_config:
            Number of layers in each successive dense block.
        """
        super().__init__()
        self.conv0 = _GraphConvBnRelu(num_input_features, 32)
        n_feat = 32
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        for num_layers in block_config:
            self.blocks.append(_DenseBlock(num_layers, n_feat, growth_rate))
            n_feat += num_layers * growth_rate
            self.transitions.append(_GraphConvBnRelu(n_feat, n_feat // 2))
            n_feat = n_feat // 2
        self.classifier = nn.Linear(n_feat, out_dim)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """Encode a molecular graph into a pooled representation.

        Parameters
        ----------
        x:
            Atom (node) features, shape ``(B, N, num_input_features)``.
        adj:
            Dense adjacency matrix, shape ``(B, N, N)``.

        Returns
        -------
        Tensor
            Pooled ligand representation, shape ``(B, out_dim)``.
        """
        h = self.conv0(x, adj)
        for block, trans in zip(self.blocks, self.transitions):
            h = block(h, adj)
            h = trans(h, adj)
        pooled = h.mean(dim=1)
        return self.classifier(pooled)


class MGraphDTA(nn.Module):
    """Multiscale protein CNN + DenseNet-style ligand GNN drug-target affinity predictor.

    Reimplements MGraphDTA's fusion architecture: the protein sequence is encoded by
    :class:`_TargetRepresentation` (parallel multi-scale conv towers), the ligand
    molecular graph by :class:`_GraphDenseNet` (densely-connected graph convolutions with
    channel-halving transitions), and the two pooled representations are concatenated
    and scored by an MLP -- the "multiscale" and "densely-connected" combination that
    gives MGraphDTA its name and its Grad-AAM interpretability hooks (both encoders
    expose per-position/per-node intermediate activations, not reproduced here since
    inference only needs the forward affinity score).
    """

    def __init__(
        self,
        block_num: int = 3,
        vocab_protein_size: int = 26,
        embedding_size: int = 32,
        filter_num: int = 32,
        atom_feature_dim: int = 22,
        out_dim: int = 1,
    ) -> None:
        """Build the protein and ligand encoders and the fusion classifier.

        Parameters
        ----------
        block_num:
            Number of parallel multi-scale protein conv towers.
        vocab_protein_size:
            Amino-acid vocabulary size.
        embedding_size:
            Amino-acid embedding dimensionality.
        filter_num:
            Channel width shared by protein towers and the ligand output projection.
        atom_feature_dim:
            Dimensionality of per-atom input features for the ligand graph.
        out_dim:
            Output dimensionality (1 for scalar binding affinity).
        """
        super().__init__()
        self.protein_encoder = _TargetRepresentation(
            block_num, vocab_protein_size, embedding_size, filter_num
        )
        self.ligand_encoder = _GraphDenseNet(
            num_input_features=atom_feature_dim, out_dim=filter_num * 3, block_config=(2, 2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(filter_num * 3 + filter_num, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, out_dim),
        )

    def forward(self, protein: Tensor, atom_feats: Tensor, adj: Tensor) -> Tensor:
        """Predict drug-target binding affinity.

        Parameters
        ----------
        protein:
            Amino-acid index sequence, shape ``(B, L_protein)``.
        atom_feats:
            Ligand atom features, shape ``(B, N_atoms, atom_feature_dim)``.
        adj:
            Ligand dense adjacency matrix, shape ``(B, N_atoms, N_atoms)``.

        Returns
        -------
        Tensor
            Predicted binding affinity, shape ``(B, out_dim)``.
        """
        protein_x = self.protein_encoder(protein)
        ligand_x = self.ligand_encoder(atom_feats, adj)
        fused = torch.cat([protein_x, ligand_x], dim=-1)
        return self.classifier(fused)


def build_mgraphdta() -> nn.Module:
    """Construct a small MGraphDTA model.

    Returns
    -------
    nn.Module
        MGraphDTA in eval mode.
    """
    return MGraphDTA(
        block_num=3,
        vocab_protein_size=26,
        embedding_size=16,
        filter_num=12,
        atom_feature_dim=22,
        out_dim=1,
    ).eval()


def example_input_mgraphdta() -> tuple[Tensor, Tensor, Tensor]:
    """Example input for MGraphDTA: protein sequence, ligand atom features, and adjacency.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(protein, atom_feats, adj)`` shaped ``(2, 40)``, ``(2, 16, 22)``, ``(2, 16, 16)``.
    """
    protein = torch.randint(1, 26, (2, 40))
    atom_feats = torch.randn(2, 16, 22)
    adj = (torch.rand(2, 16, 16) > 0.6).float()
    adj = adj + adj.transpose(1, 2)
    adj = (adj > 0).float()
    return protein, atom_feats, adj


# ---------------------------------------------------------------------------
# MHCflurry -- locally-connected (unshared-weight) peptide conv + multiplicative
# allele fusion for pan-allele MHC-I binding-affinity prediction
# ---------------------------------------------------------------------------


class _LocallyConnected1D(nn.Module):
    """1D locally-connected layer: a distinct filter weight per output position.

    Reimplements ``mhcflurry.pytorch_layers.LocallyConnected1D``: unlike ``Conv1d``,
    weights are NOT shared across positions -- each output position has its own
    ``(out_channels, in_channels * kernel_size)`` weight matrix, applied via an unfold
    + einsum. This lets the network learn position-specific peptide-anchor motifs.
    """

    def __init__(
        self, in_channels: int, out_channels: int, input_length: int, kernel_size: int
    ) -> None:
        """Build the per-position unshared weight tensor.

        Parameters
        ----------
        in_channels:
            Number of input channels (amino-acid encoding width).
        out_channels:
            Number of output filters.
        input_length:
            Length of the input peptide encoding.
        kernel_size:
            Size of the local receptive-field window.
        """
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.output_length = input_length - kernel_size + 1
        self.weight = nn.Parameter(
            torch.randn(self.output_length, out_channels, in_channels * kernel_size) * 0.05
        )
        self.bias = nn.Parameter(torch.zeros(self.output_length, out_channels))

    def forward(self, x: Tensor) -> Tensor:
        """Apply position-specific unshared-weight local convolution.

        Parameters
        ----------
        x:
            Encoded peptide, shape ``(B, input_length, in_channels)``.

        Returns
        -------
        Tensor
            Shape ``(B, output_length, out_channels)``, tanh-activated.
        """
        x_unf = x.unfold(1, self.kernel_size, 1)  # (B, output_length, in_channels, kernel_size)
        x_unf = x_unf.permute(0, 1, 3, 2).reshape(x.size(0), self.output_length, -1)
        out = torch.einsum("boi,ofi->bof", x_unf, self.weight) + self.bias
        return torch.tanh(out)


class MHCflurry(nn.Module):
    """Pan-allele MHC-I binding-affinity predictor with locally-connected peptide conv.

    Reimplements ``Class1NeuralNetworkModel``'s default topology: a fixed-length
    BLOSUM-encoded peptide passes through a :class:`_LocallyConnected1D` layer then a
    peptide-side dense layer; a separate allele pseudosequence embedding passes through
    its own dense layer; the two branches are fused by ELEMENTWISE MULTIPLY (MHCflurry's
    chosen merge method, vs. concatenation) and passed through a final dense stack ending
    in a sigmoid-activated affinity-transform output.
    """

    def __init__(
        self,
        peptide_length: int = 15,
        encoding_dim: int = 21,
        lc_filters: int = 8,
        lc_kernel: int = 3,
        peptide_dense: int = 32,
        n_alleles: int = 64,
        allele_embed_dim: int = 32,
        hidden_dim: int = 32,
    ) -> None:
        """Build the locally-connected peptide branch, allele branch, and fusion head.

        Parameters
        ----------
        peptide_length:
            Fixed peptide encoding length.
        encoding_dim:
            Per-residue amino-acid encoding width (e.g. BLOSUM62 rows).
        lc_filters:
            Number of locally-connected output filters.
        lc_kernel:
            Locally-connected receptive-field size.
        peptide_dense:
            Width of the peptide-side dense layer (must equal ``allele_embed_dim`` after
            its dense projection, since fusion is elementwise multiply).
        n_alleles:
            Number of distinct MHC alleles in the pseudosequence embedding table.
        allele_embed_dim:
            Raw allele pseudosequence embedding dimensionality.
        hidden_dim:
            Width of the shared post-fusion dense stack.
        """
        super().__init__()
        self.lc_layer = _LocallyConnected1D(encoding_dim, lc_filters, peptide_length, lc_kernel)
        flat_size = self.lc_layer.output_length * lc_filters
        self.peptide_dense = nn.Linear(flat_size, peptide_dense)

        self.allele_embedding = nn.Embedding(n_alleles, allele_embed_dim)
        self.allele_dense = nn.Linear(allele_embed_dim, peptide_dense)

        self.dense1 = nn.Linear(peptide_dense, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, peptide: Tensor, allele_idx: Tensor) -> Tensor:
        """Predict binding-affinity transform score for a peptide-allele pair.

        Parameters
        ----------
        peptide:
            BLOSUM/one-hot encoded peptide, shape ``(B, peptide_length, encoding_dim)``.
        allele_idx:
            Allele index, shape ``(B,)``.

        Returns
        -------
        Tensor
            Sigmoid-activated affinity-transform score, shape ``(B, 1)``.
        """
        x = self.lc_layer(peptide)
        x = x.reshape(x.size(0), -1)
        x = torch.tanh(self.peptide_dense(x))

        allele_embed = self.allele_embedding(allele_idx)
        allele_embed = torch.tanh(self.allele_dense(allele_embed))

        fused = x * allele_embed
        h = torch.tanh(self.dense1(fused))
        h = torch.tanh(self.dense2(h))
        return torch.sigmoid(self.output(h))


def build_mhcflurry() -> nn.Module:
    """Construct a small MHCflurry pan-allele binding predictor.

    Returns
    -------
    nn.Module
        MHCflurry in eval mode.
    """
    return MHCflurry(
        peptide_length=15,
        encoding_dim=21,
        lc_filters=8,
        lc_kernel=3,
        peptide_dense=16,
        n_alleles=32,
        allele_embed_dim=16,
        hidden_dim=16,
    ).eval()


def example_input_mhcflurry() -> tuple[Tensor, Tensor]:
    """Example input for MHCflurry: an encoded peptide batch and allele indices.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(peptide, allele_idx)`` shaped ``(4, 15, 21)`` and ``(4,)``.
    """
    peptide = torch.randn(4, 15, 21)
    allele_idx = torch.randint(0, 32, (4,))
    return peptide, allele_idx


# ---------------------------------------------------------------------------
# MiSiC -- multi-scale shape-index preprocessing + U-Net bacterial segmentation
# ---------------------------------------------------------------------------


class _MultiScaleShapeIndex(nn.Module):
    """Differentiable 3-scale shape-index-map approximation (Koenderink & van Doorn).

    Reimplements ``MiSiC.shapeindex_preprocess``: the local shape index is a curvature
    descriptor derived from the Hessian's eigenvalues (``2/pi * atan((k1+k2)/(k1-k2))``,
    here approximated from a Sobel-based local-Hessian proxy since exact
    ``skimage.feature.shape_index`` uses a non-traceable eigen-decomposition per pixel);
    it is computed at three Gaussian smoothing scales (matching the reference's
    ``sigma in {1, 1.5, 2}``) and stacked as a 3-channel input, giving MiSiC its
    modality- and contrast-invariant input representation.
    """

    def __init__(self, sigmas: tuple[float, ...] = (1.0, 1.5, 2.0)) -> None:
        """Build fixed Sobel and Gaussian-blur kernels for each scale.

        Parameters
        ----------
        sigmas:
            Gaussian smoothing scales at which the shape-index proxy is computed.
        """
        super().__init__()
        self.sigmas = sigmas
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        sobel_y = sobel_x.t()
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    @staticmethod
    def _gaussian_kernel(sigma: float, device: torch.device, dtype: torch.dtype) -> Tensor:
        radius = max(int(3 * sigma), 1)
        coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        kernel = torch.exp(-(coords**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel

    def _blur(self, x: Tensor, sigma: float) -> Tensor:
        kernel = self._gaussian_kernel(sigma, x.device, x.dtype)
        k1 = kernel.view(1, 1, 1, -1)
        k2 = kernel.view(1, 1, -1, 1)
        pad = kernel.numel() // 2
        x = F.conv2d(x, k1, padding=(0, pad))
        x = F.conv2d(x, k2, padding=(pad, 0))
        return x

    def forward(self, x: Tensor) -> Tensor:
        """Compute the 3-scale shape-index-proxy map.

        Parameters
        ----------
        x:
            Grayscale image, shape ``(B, 1, H, W)``.

        Returns
        -------
        Tensor
            Shape-index map, shape ``(B, 3, H, W)``.
        """
        channels = []
        for sigma in self.sigmas:
            smoothed = self._blur(x, sigma)
            gx = F.conv2d(smoothed, self.sobel_x, padding=1)
            gy = F.conv2d(smoothed, self.sobel_y, padding=1)
            gxx = F.conv2d(gx, self.sobel_x, padding=1)
            gyy = F.conv2d(gy, self.sobel_y, padding=1)
            gxy = F.conv2d(gx, self.sobel_y, padding=1)
            # Local-curvature proxy: principal-curvature-sum vs. -difference ratio,
            # squashed by atan2 like the real shape index's 2/pi * atan(...) form.
            trace = gxx + gyy
            disc = torch.sqrt((gxx - gyy) ** 2 + 4 * gxy**2 + 1e-6)
            shape_idx = (2.0 / torch.pi) * torch.atan2(trace, disc)
            channels.append(shape_idx)
        return torch.cat(channels, dim=1)


class _UNetDown(nn.Module):
    """Two conv+ReLU layers then 2x max-pool (U-Net encoder stage)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Build one encoder stage.

        Parameters
        ----------
        in_channels:
            Input channel count.
        out_channels:
            Output channel count.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Apply conv block then downsample.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(skip_features, pooled)``.
        """
        feat = self.conv(x)
        return feat, self.pool(feat)


class _UNetUp(nn.Module):
    """Upsample, concatenate skip connection, then two conv+ReLU layers (U-Net decoder stage)."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        """Build one decoder stage.

        Parameters
        ----------
        in_channels:
            Channel count of the incoming (lower-resolution) feature map.
        skip_channels:
            Channel count of the encoder skip connection.
        out_channels:
            Output channel count.
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        """Upsample and fuse with the matching encoder skip connection.

        Parameters
        ----------
        x:
            Lower-resolution feature map.
        skip:
            Corresponding encoder feature map.

        Returns
        -------
        Tensor
            Fused, refined feature map.
        """
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MiSiC(nn.Module):
    """Shape-index-map preprocessing + compact U-Net for bacterial cell segmentation.

    Reimplements MiSiC's documented pipeline: raw microscopy input is converted to a
    3-channel multi-scale shape-index map (:class:`_MultiScaleShapeIndex`), which is
    fed through a standard encoder-decoder U-Net with skip connections, emitting a
    3-class (background / cell body / cell boundary) categorical segmentation --
    MiSiC's defining "modality-invariant shape-index representation" over raw pixel
    intensities.
    """

    def __init__(self, base_width: int = 16, n_classes: int = 3) -> None:
        """Build the shape-index preprocessing module and the U-Net encoder/decoder.

        Parameters
        ----------
        base_width:
            Channel width of the first U-Net stage (doubled at each deeper stage).
        n_classes:
            Number of output segmentation classes.
        """
        super().__init__()
        self.shape_index = _MultiScaleShapeIndex()
        self.down1 = _UNetDown(3, base_width)
        self.down2 = _UNetDown(base_width, base_width * 2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up2 = _UNetUp(base_width * 4, base_width * 2, base_width * 2)
        self.up1 = _UNetUp(base_width * 2, base_width, base_width)
        self.out = nn.Conv2d(base_width, n_classes, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Segment a grayscale microscopy image into background/body/boundary classes.

        Parameters
        ----------
        x:
            Grayscale microscopy image, shape ``(B, 1, H, W)``.

        Returns
        -------
        Tensor
            Per-pixel class logits, shape ``(B, n_classes, H, W)``.
        """
        sh = self.shape_index(x)
        skip1, x = self.down1(sh)
        skip2, x = self.down2(x)
        x = self.bottleneck(x)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        return self.out(x)


def build_misic() -> nn.Module:
    """Construct a small MiSiC shape-index U-Net.

    Returns
    -------
    nn.Module
        MiSiC in eval mode.
    """
    return MiSiC(base_width=8, n_classes=3).eval()


def example_input_misic() -> Tensor:
    """Example input for MiSiC: a grayscale microscopy image.

    Returns
    -------
    Tensor
        Shape ``(1, 1, 64, 64)``.
    """
    return torch.randn(1, 1, 64, 64)


MENAGERIE_ENTRIES = [
    ("Mesmer", "build_mesmer", "example_input_mesmer", "2022", "BIO"),
    ("MetaNN", "build_metann", "example_input_metann", "2019", "BIO"),
    ("MetaPheno", "build_metapheno", "example_input_metapheno", "2019", "BIO"),
    ("MGraphDTA", "build_mgraphdta", "example_input_mgraphdta", "2022", "BIO"),
    ("MHCflurry", "build_mhcflurry", "example_input_mhcflurry", "2018", "BIO"),
    ("MiSiC", "build_misic", "example_input_misic", "2021", "VIS"),
]
