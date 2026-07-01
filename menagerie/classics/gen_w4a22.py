"""Menagerie batch w4a22: CNN/RNN architectures for biomedical signal, imaging,
and genomic prediction tasks.

Sources checked (reference only; no cloning, no pip installs):
  - DeepCOVID-XR (cand_00512): Wehbe, Sheng, Dutta et al., Radiology 2021,
    "DeepCOVID-XR: An Artificial Intelligence Algorithm to Detect COVID-19 on
    Chest Radiographs Trained and Tested on a Large U.S. Clinical Data Set".
    https://pubs.rsna.org/doi/full/10.1148/radiol.2020203511, official code
    https://github.com/IVPLatNU/DeepCovidXR (branch ``emerged_models``).
    Inspected ``covid_models/Densenet_model.py`` (DenseNet-121 base +
    GlobalAveragePooling2D + sigmoid Dense(1) head), the sibling
    ``Efficientnet_model.py`` / ``Inceptionnet_model.py`` /
    ``Inceptionresnet_model.py`` / ``Resnet_model.py`` / ``Xception_model.py``
    (same GAP+sigmoid-head pattern over different torchvision-equivalent
    backbones), and ``ensemble.py`` (loads all per-architecture binary
    classifiers and combines their sigmoid outputs with a
    ``DirichletEnsemble`` -- i.e. a learned convex-combination weight vector
    over the per-model probabilities, one weight per member model). This
    module reproduces the defining mechanism: N independent CNN feature
    extractors (compact torchvision-family stand-ins: a DenseNet-style
    block, an Inception-style multi-branch block, a residual block, a
    depthwise-separable ("Xception-style") block) that each independently
    reduce an input chest X-ray to a single sigmoid COVID-probability logit,
    followed by a learned per-model convex-combination (softmax-normalized
    weight vector) that ensembles the six scalar probabilities into one
    output probability -- exactly the paper's "weighted ensemble of CNNs"
    idea, not a single generic backbone.
  - DeepD2V (cand_00514): Zhang et al., Int. J. Mol. Sci. 2021, 22, 5521,
    "Deep D2V: A Novel Deep Learning-Based Framework for In Silico
    Prediction of DNA-Binding Protein". https://www.mdpi.com/1422-0067/22/11/5521,
    official code https://github.com/Sparkleiii/DeepD2V. Inspected
    ``CNN_model.py`` (a VGG-style 1D-CNN over one-hot/embedded DNA sequence,
    ``in_channels=100`` i.e. a combined "sequence + shape/physicochemical"
    encoding channel stack, VGG conv-block cfg with 1D max-pools, flatten,
    Linear(3072->1024)->Dropout->Linear(1024->1)->Sigmoid) and ``models.py``
    / ``models2.py`` (parallel branch that appends a BiLSTM head reading the
    same channel-stacked sequence encoding, whose output is concatenated
    with the CNN branch before the final classifier -- matching the paper's
    "hybrid CNN + BiLSTM" name and the repo's actual "combined DNA sequence"
    framing: DeepD2V fuses a VGG-style 1D-CNN branch and a BiLSTM branch
    that both read the same one-hot DNA-sequence-and-shape-feature tensor).
    This module reproduces the CNN+BiLSTM dual-branch fusion structurally
    with small dims.
  - DeepDIA (cand_00516): Yang, Liu, Shen et al., Nat. Commun. 11, 146
    (2020), "In silico spectral libraries by deep learning facilitate
    data-independent acquisition proteomics".
    https://www.nature.com/articles/s41467-019-13866-z, official code
    https://github.com/lmsac/DeepDIA (TF/Keras). Inspected
    ``src/pepms2/modeling.py`` (MS/MS fragment-ion-intensity prediction:
    Conv1D(64, k=2) over one-hot amino-acid sequence -> Masking ->
    Bidirectional LSTM(128, return_sequences) -> Dropout ->
    TimeDistributed Dense(intensity_size, relu), i.e. a per-residue
    sequence-to-sequence fragment-intensity regressor) and
    ``src/peprt/modeling.py`` (retention-time regression: Conv1D(64, k=5) ->
    MaxPool1D(2) -> Bidirectional LSTM(128, return_sequences) -> Dropout ->
    Flatten -> Dense(512)->Dense(256)->Dense(1), a single scalar iRT
    regressor). This module reproduces DeepDIA as a shared Conv1D+BiLSTM
    peptide-sequence encoder with the two paper-defining heads: a
    per-position TimeDistributed fragment-ion-intensity head (pepms2) and a
    pooled scalar retention-time head (peprt), matching the "CNN + RNN for
    in silico spectral library generation" description exactly.
  - DeepECG (cand_00517): Goodfellow, Goodwin, Greer, Laussen, Mazwi, Eytan,
    "Towards understanding ECG rhythm classification using convolutional
    neural networks and attention mappings", ML4H 2018 (JMLR W&C Track vol.
    85). Official code https://github.com/Seb-Good/deepecg. Inspected
    ``deepecg/training/networks/deep_ecg_v1.py`` (the paper's 13-layer 1D
    CNN: 13 stacked Conv1d+BatchNorm+ReLU(+Dropout) blocks over a raw
    single-lead ECG waveform, with dilation rates that increase in stages
    (1, 2, 4x4, 6x4, 8x2) and interleaved max-pools after layers 1, 6, 11,
    ending in a Global Average Pooling layer along the temporal axis and a
    bias-free Dense logits layer -- the exact structure that lets the paper
    compute Class Activation Maps (CAM) as ``net @ logits_weights`` over
    the pre-GAP feature map without a spatial-pooling head, per Zhou et al.
    2016's CAM formulation adapted to 1D time series). This module
    reproduces the staged-dilation 13-layer 1D CNN with GAP+linear-logits
    CAM-compatible head structurally, with small channel counts.
  - DeepGS (cand_00519): Ma, Qiu, Song, Li, Cheng, Zhai, Ma, Planta 2018,
    248(5):1307-1318, "A deep convolutional neural network approach for
    predicting phenotypes from genotypes". Official code
    https://github.com/cma2015/DeepGS (R package, MXNet backend). Inspected
    the README's documented ``cnnFrame`` architecture used by
    ``train_deepGSModel``: a 1D convolution over a length-P vector of SNP
    marker genotypes (input reshaped as a "1 x numMarkers" image row, i.e.
    conv over the marker axis) with a single conv+ReLU+max-pool stage
    (``conv_kernel="1*18"``, 8 filters, ``pool_kernel="1*4"``), followed by
    fully-connected layers (``fullayer_num_hidden = c(32, 1)``) with a
    sigmoid activation on the hidden layer and a linear (default) output
    activation producing one scalar breeding-value/phenotype prediction
    per individual. This module reproduces DeepGS's genomic-selection CNN
    exactly: Conv1d over the marker sequence -> ReLU -> MaxPool1d -> FC(32,
    sigmoid) -> FC(1) regression head over a compact synthetic
    marker-genotype vector.
  - DeepHeart / awni/ecg (cand_00521): Rajpurkar, Hannun, Haghpanahi, Bourn,
    Ng (2017) / Hannun, Rajpurkar, Haghpanahi et al., Nat. Med. 25, 65-69
    (2019), "Cardiologist-level arrhythmia detection and classification in
    ambulatory electrocardiograms using a deep neural network". Official
    code https://github.com/awni/ecg. Inspected ``ecg/network.py``: a
    34-layer 1D ResNet over raw single-lead ECG (initial Conv1D+BN+ReLU
    stem, then a stack of residual blocks each containing 2 Conv1D
    sub-layers, with a doubling-channel-count "zero pad" shortcut identity
    trick every ``conv_increase_channels_at`` blocks -- the shortcut is
    max-pooled by the block's stride and, when the channel count doubles,
    zero-padded along the channel axis rather than projected by a 1x1 conv
    -- followed by a per-timestep TimeDistributed Dense+softmax producing a
    dense per-sample rhythm-class sequence output, exactly the paper's
    "sequence-to-sequence" 34-layer ResNet formulation). This module
    reproduces the zero-pad residual-shortcut 1D ResNet stack with a
    per-timestep classification head structurally, using a compact 8-block
    stack with small channel counts instead of the paper's full 16-block,
    256-to-512-channel network.

All models below are compact, faithfully-reimplemented-from-scratch nn.Modules
with random init and small dims for TorchLens architecture-catalog tracing
(not a trained-weights zoo).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# DeepCOVID-XR -- weighted ensemble of independent CNN backbones
# ============================================================


class _DenseBackbone(nn.Module):
    """Compact DenseNet-style backbone (stand-in for DenseNet-121)."""

    def __init__(self, in_ch: int = 3, growth: int = 8, n_layers: int = 3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True)
        )
        self.layers = nn.ModuleList()
        ch = 16
        for _ in range(n_layers):
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(ch), nn.ReLU(inplace=True), nn.Conv2d(ch, growth, 3, padding=1)
                )
            )
            ch += growth
        self.out_ch = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        for layer in self.layers:
            new_feat = layer(feat)
            feat = torch.cat([feat, new_feat], dim=1)  # dense connectivity
        return feat


class _InceptionBackbone(nn.Module):
    """Compact multi-branch backbone (stand-in for Inception-V3 / InceptionResNet)."""

    def __init__(self, in_ch: int = 3, branch_ch: int = 8) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True)
        )
        self.branch1x1 = nn.Conv2d(16, branch_ch, 1)
        self.branch3x3 = nn.Conv2d(16, branch_ch, 3, padding=1)
        self.branch5x5 = nn.Conv2d(16, branch_ch, 5, padding=2)
        self.out_ch = branch_ch * 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem = self.stem(x)
        b1 = self.branch1x1(stem)
        b2 = self.branch3x3(stem)
        b3 = self.branch5x5(stem)
        return torch.cat([b1, b2, b3], dim=1)  # multi-branch concatenation


class _ResNetBackbone(nn.Module):
    """Compact residual backbone (stand-in for ResNet)."""

    def __init__(self, in_ch: int = 3, ch: int = 16) -> None:
        super().__init__()
        self.stem = nn.Conv2d(in_ch, ch, 3, padding=1)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)
        self.out_ch = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)  # residual shortcut


class _XceptionBackbone(nn.Module):
    """Compact depthwise-separable backbone (stand-in for Xception)."""

    def __init__(self, in_ch: int = 3, ch: int = 16) -> None:
        super().__init__()
        self.stem = nn.Conv2d(in_ch, ch, 3, padding=1)
        self.depthwise = nn.Conv2d(ch, ch, 3, padding=1, groups=ch)
        self.pointwise = nn.Conv2d(ch, ch, 1)
        self.out_ch = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.depthwise(x)
        return self.pointwise(x)


class _CovidHead(nn.Module):
    """GlobalAveragePooling2D + sigmoid Dense(1) head, matching every
    DeepCOVID-XR per-architecture model file (``buildBaseModel``)."""

    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_ch, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        pooled = feat.mean(dim=(2, 3))  # GlobalAveragePooling2D
        return torch.sigmoid(self.fc(pooled))


class DeepCovidXR(nn.Module):
    """DeepCOVID-XR: weighted ensemble of six independent CNN classifiers.

    Each backbone (DenseNet-style, Inception-style, two ResNet-style, and
    Xception-style, matching the paper's DenseNet-121/EfficientNet-B2/
    Inception-V3/InceptionResNetV2/ResNet/Xception roster) independently
    reduces the chest X-ray to a scalar COVID-probability logit via its own
    GAP+sigmoid head. A learned, softmax-normalized per-model weight vector
    (stand-in for the paper's fitted ``DirichletEnsemble`` weights) then
    combines the six probabilities into one ensembled probability.
    """

    def __init__(self, in_ch: int = 3) -> None:
        super().__init__()
        self.densenet = _DenseBackbone(in_ch)
        self.inception = _InceptionBackbone(in_ch)
        self.resnet = _ResNetBackbone(in_ch)
        self.inception_resnet = _ResNetBackbone(in_ch)
        self.xception = _XceptionBackbone(in_ch)
        self.efficientnet = _ResNetBackbone(in_ch)

        self.densenet_head = _CovidHead(self.densenet.out_ch)
        self.inception_head = _CovidHead(self.inception.out_ch)
        self.resnet_head = _CovidHead(self.resnet.out_ch)
        self.inception_resnet_head = _CovidHead(self.inception_resnet.out_ch)
        self.xception_head = _CovidHead(self.xception.out_ch)
        self.efficientnet_head = _CovidHead(self.efficientnet.out_ch)

        self.ensemble_weights = nn.Parameter(torch.zeros(6))  # Dirichlet-ensemble stand-in

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        probs = torch.stack(
            [
                self.densenet_head(self.densenet(image)).squeeze(-1),
                self.inception_head(self.inception(image)).squeeze(-1),
                self.resnet_head(self.resnet(image)).squeeze(-1),
                self.inception_resnet_head(self.inception_resnet(image)).squeeze(-1),
                self.xception_head(self.xception(image)).squeeze(-1),
                self.efficientnet_head(self.efficientnet(image)).squeeze(-1),
            ],
            dim=1,
        )  # (B, 6)
        weights = torch.softmax(self.ensemble_weights, dim=0)  # convex combination
        return (probs * weights.unsqueeze(0)).sum(dim=1)


def build_deepcovidxr() -> nn.Module:
    """Build a compact DeepCOVID-XR six-backbone weighted CNN ensemble."""
    return DeepCovidXR(in_ch=3).eval()


def example_input_deepcovidxr() -> torch.Tensor:
    """Chest X-ray image batch ``(1, 3, 64, 64)`` (downscaled for tracing)."""
    return torch.randn(1, 3, 64, 64)


# ============================================================
# DeepD2V -- VGG-style 1D-CNN + BiLSTM dual-branch DNA-binding predictor
# ============================================================


class DeepD2V(nn.Module):
    """DeepD2V: hybrid VGG-style 1D-CNN and BiLSTM branches over a
    channel-stacked one-hot DNA sequence-and-shape encoding, fused before a
    binary DNA-binding-protein classification head (matching the official
    ``CNN_model.py`` VGG-cfg conv stack and the ``models.py`` BiLSTM
    branch fused prior to the final sigmoid classifier).
    """

    def __init__(
        self,
        in_channels: int = 8,
        seq_len: int = 32,
        cnn_ch: tuple[int, ...] = (16, 32),
        lstm_hidden: int = 16,
    ) -> None:
        super().__init__()
        # VGG-style 1D-CNN branch (Conv1d + BatchNorm1d + ReLU, MaxPool1d 'M' stages)
        cnn_layers: list[nn.Module] = []
        ch = in_channels
        for out_ch in cnn_ch:
            cnn_layers += [
                nn.Conv1d(ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
            ]
            ch = out_ch
        self.cnn_branch = nn.Sequential(*cnn_layers)
        cnn_out_len = seq_len // (2 ** len(cnn_ch))
        self.cnn_flat_dim = ch * cnn_out_len

        # BiLSTM branch reading the same channel-stacked sequence encoding
        self.lstm_branch = nn.LSTM(in_channels, lstm_hidden, batch_first=True, bidirectional=True)
        self.lstm_flat_dim = lstm_hidden * 2

        fused_dim = self.cnn_flat_dim + self.lstm_flat_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 32),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: (B, in_channels, seq_len) one-hot DNA sequence + shape features
        cnn_out = self.cnn_branch(seq)
        cnn_out = cnn_out.reshape(cnn_out.size(0), -1)

        lstm_in = seq.transpose(1, 2)  # (B, seq_len, in_channels)
        lstm_out, _ = self.lstm_branch(lstm_in)
        lstm_out = lstm_out[:, -1, :]  # final-timestep BiLSTM summary

        fused = torch.cat([cnn_out, lstm_out], dim=1)
        return self.classifier(fused)


def build_deepd2v() -> nn.Module:
    """Build a compact DeepD2V CNN+BiLSTM DNA-binding-site predictor."""
    return DeepD2V(in_channels=8, seq_len=32, cnn_ch=(16, 32), lstm_hidden=16).eval()


def example_input_deepd2v() -> torch.Tensor:
    """One-hot-encoded DNA sequence + shape-feature stack ``(1, 8, 32)``."""
    return torch.randn(1, 8, 32)


# ============================================================
# DeepDIA -- shared Conv1D+BiLSTM peptide encoder, dual MS2/RT heads
# ============================================================


class DeepDIAEncoder(nn.Module):
    """Conv1d + Bidirectional LSTM peptide-sequence encoder shared by
    DeepDIA's ``pepms2`` and ``peprt`` models (both build a Conv1D layer
    over one-hot amino-acid sequence followed by a BiLSTM)."""

    def __init__(
        self,
        n_amino_acids: int = 22,
        conv_filters: int = 16,
        lstm_hidden: int = 16,
        kernel_size: int = 2,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            n_amino_acids, conv_filters, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.lstm = nn.LSTM(conv_filters, lstm_hidden, batch_first=True, bidirectional=True)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: (B, n_amino_acids, L) one-hot peptide sequence
        conv_out = F.relu(self.conv(seq)).transpose(1, 2)  # (B, L', conv_filters)
        lstm_out, _ = self.lstm(conv_out)  # (B, L', 2*lstm_hidden)
        return lstm_out


class DeepDIAms2(nn.Module):
    """DeepDIA MS/MS fragment-ion-intensity head: shared Conv1D+BiLSTM
    encoder followed by a per-position (TimeDistributed) Dense-ReLU
    intensity regressor, matching ``pepms2/modeling.py``."""

    def __init__(self, n_amino_acids: int = 22, n_ion_types: int = 4) -> None:
        super().__init__()
        self.encoder = DeepDIAEncoder(n_amino_acids, conv_filters=16, lstm_hidden=16, kernel_size=2)
        self.dropout = nn.Dropout(0.5)
        self.intensity_head = nn.Linear(32, n_ion_types)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        feats = self.dropout(self.encoder(seq))
        return F.relu(self.intensity_head(feats))  # (B, L', n_ion_types) per-position intensities


class DeepDIArt(nn.Module):
    """DeepDIA retention-time regression head: shared Conv1D(+maxpool)+BiLSTM
    encoder pooled/flattened into a single scalar iRT prediction, matching
    ``peprt/modeling.py``."""

    def __init__(self, n_amino_acids: int = 22, seq_len: int = 20) -> None:
        super().__init__()
        self.conv = nn.Conv1d(n_amino_acids, 16, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm = nn.LSTM(16, 16, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        pooled_len = seq_len // 2
        self.fc = nn.Sequential(
            nn.Linear(32 * pooled_len, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        conv_out = F.relu(self.conv(seq))
        pooled = self.pool(conv_out).transpose(1, 2)  # (B, L//2, 16)
        lstm_out, _ = self.lstm(pooled)
        lstm_out = self.dropout(lstm_out)
        flat = lstm_out.reshape(lstm_out.size(0), -1)
        return self.fc(flat)


def build_deepdia_ms2() -> nn.Module:
    """Build the DeepDIA MS/MS fragment-intensity Conv1D+BiLSTM predictor."""
    return DeepDIAms2(n_amino_acids=22, n_ion_types=4).eval()


def example_input_deepdia_ms2() -> torch.Tensor:
    """One-hot peptide sequence ``(1, 22, 20)`` (20 residues, 22 tokens)."""
    return torch.randn(1, 22, 20)


def build_deepdia_rt() -> nn.Module:
    """Build the DeepDIA retention-time Conv1D+BiLSTM regressor."""
    return DeepDIArt(n_amino_acids=22, seq_len=20).eval()


def example_input_deepdia_rt() -> torch.Tensor:
    """One-hot peptide sequence ``(1, 22, 20)`` for iRT regression."""
    return torch.randn(1, 22, 20)


# ============================================================
# DeepECG -- 13-layer staged-dilation 1D CNN with CAM-compatible GAP head
# ============================================================


class DeepECG(nn.Module):
    """DeepECG: the paper's 13-layer 1D CNN over a raw single-lead ECG
    waveform, with dilation rates staged in four groups (1, 2, 4, 6, 8) and
    max-pools interleaved after layers 1, 6, and 11, ending in a Global
    Average Pooling layer along time and a bias-free linear logits layer --
    reproducing ``deep_ecg_v1.py``'s exact structure that supports Class
    Activation Map computation as ``feature_map @ logits_weights``.
    """

    _dilations = (1, 2, 4, 4, 4, 4, 6, 6, 6, 6, 8, 8, 8)
    _channels = (24, 20, 20, 20, 20, 10, 10, 10, 10, 10, 10, 6, 6)
    _kernels = (12, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4)
    _pool_after = {1, 6, 11}  # 1-indexed layer numbers followed by max-pool

    def __init__(self, in_channels: int = 1, n_classes: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        ch = in_channels
        for i, (dilation, out_ch, kernel) in enumerate(
            zip(self._dilations, self._channels, self._kernels), start=1
        ):
            layers.append(
                nn.Conv1d(
                    ch,
                    out_ch,
                    kernel_size=kernel,
                    padding=(kernel * dilation) // 2,
                    dilation=dilation,
                )
            )
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            if i in self._pool_after:
                layers.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=0))
            ch = out_ch
        self.conv_stack = nn.Sequential(*layers)
        self.logits = nn.Linear(ch, n_classes, bias=False)  # bias-free, CAM-compatible

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (B, in_channels, T)
        feat = self.conv_stack(waveform)  # (B, C, T')
        gap = feat.mean(dim=2)  # Global Average Pooling along time -> CAM base
        return self.logits(gap)


def build_deepecg() -> nn.Module:
    """Build a compact DeepECG 13-layer staged-dilation 1D CNN."""
    return DeepECG(in_channels=1, n_classes=3).eval()


def example_input_deepecg() -> torch.Tensor:
    """Single-lead ECG waveform ``(1, 1, 512)`` (downsampled for tracing)."""
    return torch.randn(1, 1, 512)


# ============================================================
# DeepGS -- 1D-CNN genomic-selection breeding-value predictor
# ============================================================


class DeepGS(nn.Module):
    """DeepGS: a single Conv1d+ReLU+MaxPool1d stage over a genome-wide SNP
    marker-genotype vector, followed by a sigmoid-activated hidden FC layer
    and a linear scalar phenotype/breeding-value output -- matching the
    documented ``cnnFrame`` (``conv_kernel="1*18"``, 8 filters,
    ``pool_kernel="1*4"``, ``fullayer_num_hidden = c(32, 1)``).
    """

    def __init__(self, n_markers: int = 256, conv_filters: int = 8, kernel_size: int = 18) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, conv_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
        pooled_len = (n_markers) // 4
        self.fc1 = nn.Linear(conv_filters * pooled_len, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, markers: torch.Tensor) -> torch.Tensor:
        # markers: (B, 1, n_markers) SNP genotype vector encoded as a "1 x P image row"
        feat = F.relu(self.conv(markers))
        feat = self.pool(feat)
        flat = feat.reshape(feat.size(0), -1)
        hidden = torch.sigmoid(self.fc1(flat))
        return self.fc2(hidden)  # linear breeding-value / phenotype output


def build_deepgs() -> nn.Module:
    """Build a compact DeepGS 1D-CNN genomic-selection predictor."""
    return DeepGS(n_markers=256, conv_filters=8, kernel_size=18).eval()


def example_input_deepgs() -> torch.Tensor:
    """Genome-wide SNP marker-genotype vector ``(1, 1, 256)``."""
    return torch.randn(1, 1, 256)


# ============================================================
# DeepHeart / awni-ecg -- zero-pad-shortcut 1D ResNet for ECG rhythm
# ============================================================


class _ZeroPadResBlock(nn.Module):
    """One residual sub-block of the awni/ecg 34-layer ResNet: two
    Conv1d+BN+ReLU sub-layers, a stride-matched max-pool shortcut, and a
    zero-padding of the shortcut's channel axis when the block doubles
    channel count (the paper's parameter-free "identity" channel upsample,
    as opposed to a projection shortcut).
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int, kernel_size: int = 8) -> None:
        super().__init__()
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.shortcut_pool = (
            nn.MaxPool1d(kernel_size=stride, stride=stride) if stride > 1 else nn.Identity()
        )
        self.bn1 = nn.BatchNorm1d(in_ch)
        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(
            out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut_pool(x)
        if self.out_ch != self.in_ch:
            pad_ch = self.out_ch - self.in_ch
            zeros = torch.zeros(
                shortcut.size(0), pad_ch, shortcut.size(2), device=x.device, dtype=x.dtype
            )
            shortcut = torch.cat(
                [shortcut, zeros], dim=1
            )  # zero-pad channel-axis identity shortcut

        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)

        out = out[..., : shortcut.size(-1)]  # length alignment after strided conv rounding
        return out + shortcut


class DeepHeartECGResNet(nn.Module):
    """DeepHeart (awni/ecg): a 34-layer-style 1D ResNet stack over raw
    single-lead ECG with a zero-pad channel-doubling shortcut every
    ``increase_channels_every`` blocks and a per-timestep (TimeDistributed)
    Dense+softmax classification head, matching ``ecg/network.py``'s
    ``add_resnet_layers`` / ``resnet_block`` / ``add_output_layer``. Uses a
    compact 8-block stack (vs. the paper's 16) with small channel counts.
    """

    def __init__(
        self,
        n_blocks: int = 8,
        base_ch: int = 16,
        n_classes: int = 4,
        increase_channels_every: int = 4,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_ch, kernel_size=8, padding=4),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True),
        )
        blocks = []
        ch = base_ch
        for i in range(n_blocks):
            stride = 2 if i % 2 == 1 else 1
            grow = (i % increase_channels_every == 0) and i > 0
            out_ch = ch * 2 if grow else ch
            blocks.append(_ZeroPadResBlock(ch, out_ch, stride=stride))
            ch = out_ch
        self.res_blocks = nn.Sequential(*blocks)
        self.bn_final = nn.BatchNorm1d(ch)
        self.classifier = nn.Linear(ch, n_classes)  # TimeDistributed Dense + softmax

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (B, 1, T)
        feat = self.stem(waveform)
        feat = self.res_blocks(feat)
        feat = F.relu(self.bn_final(feat))
        feat = feat.transpose(1, 2)  # (B, T', C)
        logits = self.classifier(feat)  # per-timestep class logits
        return F.softmax(logits, dim=-1)


def build_deepheart() -> nn.Module:
    """Build a compact DeepHeart (awni/ecg) zero-pad-shortcut 1D ResNet."""
    return DeepHeartECGResNet(n_blocks=8, base_ch=16, n_classes=4, increase_channels_every=4).eval()


def example_input_deepheart() -> torch.Tensor:
    """Single-lead ECG waveform ``(1, 1, 512)`` (downsampled for tracing)."""
    return torch.randn(1, 1, 512)


# ============================================================
# Registry
# ============================================================

MENAGERIE_ENTRIES = [
    ("DeepCOVID-XR", "build_deepcovidxr", "example_input_deepcovidxr", "2021", "BIO"),
    ("DeepD2V", "build_deepd2v", "example_input_deepd2v", "2021", "BIO"),
    ("DeepDIA", "build_deepdia_ms2", "example_input_deepdia_ms2", "2020", "BIO"),
    ("DeepECG", "build_deepecg", "example_input_deepecg", "2018", "BIO"),
    ("DeepGS", "build_deepgs", "example_input_deepgs", "2018", "BIO"),
    ("DeepHeart", "build_deepheart", "example_input_deepheart", "2019", "BIO"),
]
