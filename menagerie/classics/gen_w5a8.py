"""Compact faithful reimplementations of five bioinformatics / genomics / proteomics /
histopathology architecture families.

Sources checked (paper + official/community source; reimplemented compactly from
scratch in base-env torch, no clone/pip-install):
  - NetSurfP-2.0: Klausen, Jespersen, Nielsen, Jensen, Jurtz, Soenderby, Sommer,
    Winther, Nielsen, Petersen & Marcatili, "NetSurfP-2.0: Improved prediction of
    protein structural features by integrated deep learning" (Proteins 2019); the
    NetSurfP-3.0 successor repo Eryk96/NetSurfP-3.0 (PyTorch) publishes the exact
    same "CNNbLSTM" baseline architecture in ``nsp3/nsp3/models/CNNbLSMT/model.py``,
    class ``CNNbLSTM``. The distinctive mechanism is DENSELY-CONCATENATED
    MULTI-KERNEL CNN FEATURES FED INTO A DEEP BIDIRECTIONAL LSTM, THEN
    MULTI-TASK-SPLIT: several parallel 1-D convolutions with different kernel sizes
    scan the per-residue input profile (PSSM / HMM / one-hot sequence profile), each
    producing new channels that are CONCATENATED (not summed) onto the running
    feature tensor -- so the LSTM input width grows with the number of conv branches
    -- followed by batch norm and a 2-layer bidirectional LSTM, and finally SIX
    task-specific linear heads reading the same shared LSTM hidden state: 8-class
    and 3-class secondary structure, 2-class disorder, RSA (sigmoid), and phi/psi
    backbone angles (tanh, sine/cosine-style bounded regression). Reimplemented with
    two parallel dense multi-kernel conv branches, a batch norm, a 2-layer
    bidirectional LSTM, and the same six per-residue task heads.
  - Omnipose: Cutler, Stringer, Lo, Rappez, Stroustrup, Brook Peterson, Wiggins &
    Mougous, "Omnipose: a high-precision morphology-independent solution for
    bacterial cell segmentation" (Nature Methods 2022, biorxiv:2021.11.03.467199);
    official repo Makelalab/Omnipose (segmentation post-processing) with the
    trainable backbone shared from kevinjohncutler/cellpose-omni
    (``cellpose_omni/resnet_torch.py``, class ``CPnet``, and ``models.py``
    ``nclasses = dim + 2`` for omni mode). The distinctive mechanism is a residual
    U-Net (``CPnet``) whose global-average-pooled bottleneck feature vector is
    turned into a "STYLE" VECTOR that is broadcast-ADDED into every decoder residual
    block's input before each of its convolutions (style-conditioned decoding,
    inherited from Cellpose), combined with OMNIPOSE's key extension: instead of
    Cellpose's 3-channel output (2-D flow field + cell probability), Omnipose adds a
    DISTANCE-TRANSFORM-DERIVED BOUNDARY-FIELD CHANNEL (``nclasses = dim + 2`` = 2
    flow components + a distance/Euclidean-medial-axis field + a boundary logit),
    which lets the mask-reconstruction step separate the interior "skeleton" of
    elongated/filamentous cells from their boundary even when a naive centroid-based
    flow field (Cellpose's original design) would merge or shear long, curved cells.
    Reimplemented as a compact residual-downsample / style-conditioned
    residual-upsample U-Net (``CPnet``-equivalent) emitting the omni 4-channel head
    (2 flow components + distance field + boundary field) for 2-D input.
  - pDeep (pDeep2): Zeng, Zhou, Yu, Zhu, Lam, Yu, Chen & He, "MS/MS spectrum
    prediction for modified peptides using pDeep2 trained by deep learning" (Analytical
    Chemistry 2019); official repo pFindStudio/pDeep, ``pDeep2/model/lstm_tf.py``,
    class ``IonLSTM`` / function ``BuildModel``. The distinctive mechanism is a
    STACKED BIDIRECTIONAL LSTM OVER PER-RESIDUE PEPTIDE FEATURES WITH BROADCAST
    PRECURSOR CONTEXT: each residue position's one-hot amino-acid + PTM feature
    vector has the (scalar) precursor CHARGE broadcast-concatenated onto every
    timestep before a 2-layer bidirectional LSTM "encoder", and at every stacked
    BiLSTM layer the charge/instrument/NCE context is RE-CONCATENATED onto that
    layer's output before feeding the next layer (so fine-tuning can rewire only
    the later layers while global context stays available throughout); a final
    bidirectional LSTM "output" layer produces per-position ion-intensity logits
    where the forward and backward directions are ELEMENT-WISE SUMMED (not
    concatenated) to give one fragment-ion-type intensity per residue position.
    Reimplemented as a 2-layer stacked BiLSTM with charge broadcast-concatenated at
    every layer's input, followed by an output BiLSTM whose forward/backward halves
    are summed into per-position fragment-ion-intensity predictions.
  - PEPPER-Margin-DeepVariant (PEPPER's variant-candidate neural network):
    Shafin, Pesout, Chang, Nattestad, Kolesnikov, Goel, Baid, Kolmogorov & Paten,
    "Haplotype-aware variant calling with PEPPER-Margin-DeepVariant enables high
    accuracy in nanopore long-reads" (Nature Methods 2021); official repo
    kishwarshafin/pepper, ``pepper_variant/modules/python/models/simple_model.py``,
    class ``TransducerGRU``. The distinctive mechanism is a PILEUP-IMAGE-TO-VARIANT
    STACKED-LSTM TRANSDUCER: a per-candidate-window pileup "image" (read-base /
    quality / strand summary features stacked over a fixed window of reference
    positions) is passed through TWO STACKED BIDIRECTIONAL LSTMs ("encoder" then
    "decoder", each consuming the full previous layer's per-position hidden state,
    not just a summary vector), the decoder output is FLATTENED ACROSS THE ENTIRE
    CANDIDATE WINDOW into one long vector (rather than pooled), and that flattened
    window representation is pushed through a deep 5-layer SELU-activated MLP to a
    single per-window variant-type classification head -- i.e. the whole candidate
    window's sequential LSTM output becomes one flat feature vector for a window-level
    call, rather than a per-position sequence-tagging output. Reimplemented as a
    2-layer stacked BiLSTM encoder/decoder over a small pileup-image window, flattened
    across the window, and fed through a 5-layer SELU MLP to a variant-type head.
  - Phikon: Filiot, Ghermi, Olivier, Jacob, Fidon, Mac Kain, Saillard & Schiratti,
    "Scaling Self-Supervised Learning for Histopathology with Masked Image Modeling"
    (MedRxiv 2023); official repo owkin/HistoSSLscaling, and the released checkpoint
    ``owkin/phikon`` on HuggingFace. The distinctive mechanism is IBOT (IMAGE BERT
    PRE-TRAINING WITH ONLINE TOKENIZER) SELF-SUPERVISED PRETRAINING OF A PLAIN VIT-B/16
    ON 40 MILLION HISTOPATHOLOGY TILES: a standard ViT-B/16 patch-embedding
    transformer backbone (no architectural novelty over ViT itself) trained with a
    DINO-style multi-crop teacher/student objective PLUS a masked-image-modeling
    (MIM) token-prediction loss against an online (EMA teacher) discrete tokenizer,
    unlike iBOT's typical from-scratch-vocabulary tokenizer, applied to histology
    tiles at 20x magnification. Since the pretraining objective (iBOT) leaves no
    architectural trace in the released backbone -- the shipped, reusable trainable
    component is the plain ViT-B/16 encoder itself -- this is built directly from
    the installed ``timm`` VisionTransformer factory (``vit_base_patch16_224``) at
    tiny catalog dimensions, which is the exact module class Phikon publishes as its
    feature extractor (``timm``-style ``VisionTransformer``, class token pooled).
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# NetSurfP-2.0 (CNNbLSTM dense multi-kernel conv + bidirectional LSTM)
# ---------------------------------------------------------------------------


class DenseConvBranch(nn.Module):
    """One 1-D conv branch whose output is densely concatenated onto its input."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        """Initialize the branch.

        Parameters
        ----------
        in_channels:
            Number of channels in the running feature tensor.
        out_channels:
            Number of new channels this branch appends.
        kernel_size:
            Convolution kernel size (odd, same-padded).
        """

        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """Densely concatenate this branch's activation onto the input.

        Parameters
        ----------
        x:
            Running feature tensor, shape ``(batch, in_channels, length)``.

        Returns
        -------
        Tensor
            ``(batch, in_channels + out_channels, length)``.
        """

        return torch.cat([x, self.act(self.conv(x))], dim=1)


class CNNbLSTM(nn.Module):
    """Compact NetSurfP-2.0-style dense multi-kernel CNN + bidirectional LSTM."""

    def __init__(
        self,
        in_channels: int = 20,
        conv_channels: int = 8,
        kernel_sizes: tuple[int, ...] = (3, 5),
        hidden_size: int = 16,
        lstm_layers: int = 2,
    ) -> None:
        """Initialize the CNNbLSTM secondary-structure / RSA / angle predictor.

        Parameters
        ----------
        in_channels:
            Number of input per-residue profile channels (amino-acid one-hot etc.).
        conv_channels:
            Channels added by each dense conv branch.
        kernel_sizes:
            Kernel size of each parallel dense conv branch.
        hidden_size:
            Bidirectional LSTM hidden size (per direction).
        lstm_layers:
            Number of stacked bidirectional LSTM layers.
        """

        super().__init__()
        branches = []
        running_channels = in_channels
        for k in kernel_sizes:
            branches.append(DenseConvBranch(running_channels, conv_channels, k))
            running_channels += conv_channels
        self.branches = nn.ModuleList(branches)
        dense_channels = running_channels
        self.batch_norm = nn.BatchNorm1d(dense_channels)
        self.lstm = nn.LSTM(
            dense_channels,
            hidden_size,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
        )
        out_size = hidden_size * 2
        self.ss8 = nn.Linear(out_size, 8)
        self.ss3 = nn.Linear(out_size, 3)
        self.disorder = nn.Linear(out_size, 2)
        self.rsa = nn.Linear(out_size, 1)
        self.phi = nn.Linear(out_size, 2)
        self.psi = nn.Linear(out_size, 2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Predict per-residue structural features from a sequence profile.

        Parameters
        ----------
        x:
            Per-residue profile, shape ``(batch, length, in_channels)``.

        Returns
        -------
        tuple[Tensor, ...]
            ``(ss8, ss3, disorder, rsa, phi, psi)`` logits/regressions, each with a
            leading ``(batch, length, ...)`` shape.
        """

        h = x.transpose(1, 2)
        for branch in self.branches:
            h = branch(h)
        h = self.batch_norm(h)
        h = h.transpose(1, 2)
        h, _ = self.lstm(h)
        ss8 = self.ss8(h)
        ss3 = self.ss3(h)
        disorder = self.disorder(h)
        rsa = torch.sigmoid(self.rsa(h))
        phi = torch.tanh(self.phi(h))
        psi = torch.tanh(self.psi(h))
        return ss8, ss3, disorder, rsa, phi, psi


def build_netsurfp2() -> nn.Module:
    """Build a compact NetSurfP-2.0-style CNNbLSTM model.

    Returns
    -------
    nn.Module
        The model in ``eval()`` mode.
    """

    return CNNbLSTM(in_channels=20, conv_channels=8, kernel_sizes=(3, 5), hidden_size=16).eval()


def example_input_netsurfp2() -> Tensor:
    """Example per-residue sequence profile.

    Returns
    -------
    Tensor
        Shape ``(1, 40, 20)``: batch of one 40-residue protein, 20 profile channels.
    """

    return torch.rand(1, 40, 20)


# ---------------------------------------------------------------------------
# Omnipose (style-conditioned residual U-Net, 2-D flow + distance + boundary head)
# ---------------------------------------------------------------------------


class ResDown(nn.Module):
    """Residual downsampling block: projected skip plus two conv pairs."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the block.

        Parameters
        ----------
        in_channels:
            Input channel count.
        out_channels:
            Output channel count.
        """

        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 1)
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.conv4 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual downsampling block.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        Tensor
            Output feature map, same spatial size, ``out_channels`` channels.
        """

        x = self.proj(x) + self.conv2(self.conv1(x))
        x = x + self.conv4(self.conv3(x))
        return x


class StyleConv(nn.Module):
    """Conv block whose input is offset by a broadcast style vector."""

    def __init__(self, in_channels: int, out_channels: int, style_channels: int) -> None:
        """Initialize the style-conditioned conv block.

        Parameters
        ----------
        in_channels:
            Input channel count.
        out_channels:
            Output channel count.
        style_channels:
            Dimensionality of the global style vector.
        """

        super().__init__()
        self.to_bias = nn.Linear(style_channels, in_channels)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )

    def forward(self, x: Tensor, style: Tensor) -> Tensor:
        """Broadcast-add the style vector then convolve.

        Parameters
        ----------
        x:
            Input feature map, shape ``(batch, in_channels, h, w)``.
        style:
            Global style vector, shape ``(batch, style_channels)``.

        Returns
        -------
        Tensor
            Convolved feature map, ``out_channels`` channels.
        """

        bias = self.to_bias(style)[:, :, None, None]
        return self.conv(x + bias)


class ResUp(nn.Module):
    """Residual upsampling block, style-conditioned at every internal conv."""

    def __init__(self, in_channels: int, out_channels: int, style_channels: int) -> None:
        """Initialize the block.

        Parameters
        ----------
        in_channels:
            Input channel count (post-upsample, pre-skip-merge).
        out_channels:
            Output channel count.
        style_channels:
            Dimensionality of the global style vector.
        """

        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 1)
        self.conv0 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.sconv1 = StyleConv(out_channels, out_channels, style_channels)
        self.sconv2 = StyleConv(out_channels, out_channels, style_channels)
        self.sconv3 = StyleConv(out_channels, out_channels, style_channels)

    def forward(self, x: Tensor, skip: Tensor, style: Tensor) -> Tensor:
        """Apply the style-conditioned residual upsampling block.

        Parameters
        ----------
        x:
            Upsampled decoder feature map.
        skip:
            Matching encoder skip-connection feature map.
        style:
            Global style vector.

        Returns
        -------
        Tensor
            Decoder feature map at this resolution, ``out_channels`` channels.
        """

        h = self.proj(x) + self.sconv1(self.conv0(x) + skip, style)
        h = h + self.sconv3(self.sconv2(h, style), style)
        return h


class OmniposeCPnet(nn.Module):
    """Compact Omnipose/Cellpose-style style-conditioned residual U-Net."""

    def __init__(self, base_channels: tuple[int, ...] = (2, 8, 16, 32)) -> None:
        """Initialize the network.

        Parameters
        ----------
        base_channels:
            Channel widths from input through each downsampling stage.
        """

        super().__init__()
        self.downs = nn.ModuleList(
            [ResDown(base_channels[i], base_channels[i + 1]) for i in range(len(base_channels) - 1)]
        )
        self.pool = nn.MaxPool2d(2)
        style_channels = base_channels[-1]
        up_channels = list(base_channels[1:]) + [base_channels[-1]]
        self.ups = nn.ModuleList(
            [
                ResUp(up_channels[i + 1], up_channels[i], style_channels)
                for i in range(len(up_channels) - 2, -1, -1)
            ]
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        # Omnipose head: dim (2 flow components) + distance field + boundary field.
        self.output = nn.Conv2d(up_channels[0], 4, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Segment cells via style-conditioned flow/distance/boundary fields.

        Parameters
        ----------
        x:
            Input image, shape ``(batch, base_channels[0], h, w)``.

        Returns
        -------
        Tensor
            4-channel Omnipose head: ``(dy, dx, distance_field, boundary_logit)``,
            shape ``(batch, 4, h, w)``.
        """

        feats = []
        h = x
        for i, down in enumerate(self.downs):
            if i > 0:
                h = self.pool(h)
            h = down(h)
            feats.append(h)

        style = F.adaptive_avg_pool2d(feats[-1], 1).flatten(1)
        style = style / (style.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-6)

        h = feats[-1]
        h = self.ups[0](h, feats[-1], style)
        for i in range(1, len(self.ups)):
            h = self.upsample(h)
            h = self.ups[i](h, feats[-1 - i], style)
        return self.output(h)


def build_omnipose() -> nn.Module:
    """Build a compact Omnipose-style style-conditioned residual U-Net.

    Returns
    -------
    nn.Module
        The model in ``eval()`` mode.
    """

    return OmniposeCPnet(base_channels=(2, 8, 16, 32)).eval()


def example_input_omnipose() -> Tensor:
    """Example 2-channel (phase/fluorescence) cell image.

    Returns
    -------
    Tensor
        Shape ``(1, 2, 64, 64)``.
    """

    return torch.rand(1, 2, 64, 64)


# ---------------------------------------------------------------------------
# pDeep (pDeep2): stacked BiLSTM MS/MS fragment-intensity predictor
# ---------------------------------------------------------------------------


class ChargeBroadcastBiLSTM(nn.Module):
    """Bidirectional LSTM layer with the charge scalar re-concatenated at input."""

    def __init__(self, in_features: int, hidden_size: int) -> None:
        """Initialize the layer.

        Parameters
        ----------
        in_features:
            Per-timestep feature width, excluding the broadcast charge scalar.
        hidden_size:
            LSTM hidden size (per direction).
        """

        super().__init__()
        self.lstm = nn.LSTM(in_features + 1, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, x: Tensor, charge: Tensor) -> Tensor:
        """Broadcast-concatenate charge onto every timestep, then run the BiLSTM.

        Parameters
        ----------
        x:
            Per-residue features, shape ``(batch, length, in_features)``.
        charge:
            Precursor charge, shape ``(batch, 1)``.

        Returns
        -------
        Tensor
            Concatenated forward/backward hidden states, ``(batch, length, 2 * hidden_size)``.
        """

        length = x.size(1)
        ch = charge.unsqueeze(1).expand(-1, length, -1)
        h, _ = self.lstm(torch.cat([x, ch], dim=-1))
        return h


class PDeepIonLSTM(nn.Module):
    """Compact pDeep2-style stacked BiLSTM MS/MS fragment-intensity predictor."""

    def __init__(self, in_features: int = 27, hidden_size: int = 16, n_ion_types: int = 8) -> None:
        """Initialize the pDeep2-style ``IonLSTM``.

        Parameters
        ----------
        in_features:
            Per-residue one-hot amino-acid + PTM feature width.
        hidden_size:
            BiLSTM hidden size (per direction) at every stacked layer.
        n_ion_types:
            Number of fragment-ion-type intensities predicted per residue bond.
        """

        super().__init__()
        self.layer1 = ChargeBroadcastBiLSTM(in_features, hidden_size)
        self.layer2 = ChargeBroadcastBiLSTM(hidden_size * 2, hidden_size)
        self.out_fwd = nn.LSTM(hidden_size * 2 + 1, n_ion_types, batch_first=True)
        self.out_bwd = nn.LSTM(hidden_size * 2 + 1, n_ion_types, batch_first=True)

    def forward(self, x: Tensor, charge: Tensor) -> Tensor:
        """Predict per-bond fragment-ion intensities for a peptide.

        Parameters
        ----------
        x:
            Per-residue one-hot amino-acid + PTM features, shape
            ``(batch, length - 1, in_features)`` (one fewer than peptide length,
            one entry per backbone bond).
        charge:
            Precursor charge, shape ``(batch, 1)``.

        Returns
        -------
        Tensor
            Predicted fragment-ion intensities in ``[0, 1]``, shape
            ``(batch, length - 1, n_ion_types)``.
        """

        h = self.layer1(x, charge)
        h = self.layer2(h, charge)
        length = h.size(1)
        ch = charge.unsqueeze(1).expand(-1, length, -1)
        h_ch = torch.cat([h, ch], dim=-1)
        fwd, _ = self.out_fwd(h_ch)
        bwd, _ = self.out_bwd(h_ch.flip(1))
        bwd = bwd.flip(1)
        out = fwd + bwd
        return torch.clamp(out, 0.0, 1.0)


def build_pdeep() -> nn.Module:
    """Build a compact pDeep2-style stacked BiLSTM fragment-intensity predictor.

    Returns
    -------
    nn.Module
        The model in ``eval()`` mode.
    """

    return PDeepIonLSTM(in_features=27, hidden_size=16, n_ion_types=8).eval()


def example_input_pdeep() -> tuple[Tensor, Tensor]:
    """Example one-hot peptide-bond features and precursor charge.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(bond_features, charge)`` = ``((1, 12, 27), (1, 1))`` for a 13-residue
        peptide (12 backbone bonds) at charge 2.
    """

    bond_features = torch.rand(1, 12, 27)
    charge = torch.tensor([[2.0]])
    return bond_features, charge


# ---------------------------------------------------------------------------
# PEPPER-Margin-DeepVariant: pileup-image stacked-LSTM variant-type transducer
# ---------------------------------------------------------------------------


class TransducerGRU(nn.Module):
    """Compact PEPPER-style stacked-BiLSTM pileup-window variant-type transducer."""

    def __init__(
        self,
        image_features: int = 10,
        window_size: int = 16,
        lstm_hidden: int = 24,
        mlp_hidden: int = 64,
        num_classes_type: int = 6,
    ) -> None:
        """Initialize the ``TransducerGRU`` variant-candidate classifier.

        Parameters
        ----------
        image_features:
            Per-position pileup-image feature width (base/quality/strand summary).
        window_size:
            Number of reference positions in the candidate window.
        lstm_hidden:
            Hidden size (per direction) of the encoder and decoder BiLSTMs.
        mlp_hidden:
            Hidden width of the deep SELU MLP applied to the flattened window.
        num_classes_type:
            Number of candidate variant-type classes.
        """

        super().__init__()
        self.window_size = window_size
        self.encoder = nn.LSTM(image_features, lstm_hidden, bidirectional=True, batch_first=True)
        self.decoder = nn.LSTM(lstm_hidden * 2, lstm_hidden, bidirectional=True, batch_first=True)
        self.activation = nn.SELU()
        flat_size = lstm_hidden * 2 * window_size
        self.linear1 = nn.Linear(flat_size, mlp_hidden)
        self.linear2 = nn.Linear(mlp_hidden, mlp_hidden)
        self.linear3 = nn.Linear(mlp_hidden, mlp_hidden)
        self.linear4 = nn.Linear(mlp_hidden, mlp_hidden)
        self.linear5 = nn.Linear(mlp_hidden, mlp_hidden)
        self.output_layer_type = nn.Linear(mlp_hidden, num_classes_type)

    def forward(self, x: Tensor) -> Tensor:
        """Classify the variant type of a pileup-image candidate window.

        Parameters
        ----------
        x:
            Pileup image, shape ``(batch, window_size, image_features)``.

        Returns
        -------
        Tensor
            Per-window variant-type class probabilities, shape ``(batch, num_classes_type)``.
        """

        h, _ = self.encoder(x)
        h, _ = self.decoder(h)
        h = torch.flatten(h, start_dim=1, end_dim=2)
        h = self.activation(self.linear1(h))
        h = self.activation(self.linear2(h))
        h = self.activation(self.linear3(h))
        h = self.activation(self.linear4(h))
        h = self.activation(self.linear5(h))
        return F.softmax(self.output_layer_type(h), dim=1)


def build_pepper() -> nn.Module:
    """Build a compact PEPPER-Margin-DeepVariant-style pileup transducer.

    Returns
    -------
    nn.Module
        The model in ``eval()`` mode.
    """

    return TransducerGRU(image_features=10, window_size=16, lstm_hidden=24, mlp_hidden=64).eval()


def example_input_pepper() -> Tensor:
    """Example nanopore pileup-image candidate window.

    Returns
    -------
    Tensor
        Shape ``(1, 16, 10)``: a 16-position candidate window, 10 pileup features
        per position.
    """

    return torch.rand(1, 16, 10)


# ---------------------------------------------------------------------------
# Phikon: iBOT-pretrained plain ViT-B/16 histopathology tile encoder
# ---------------------------------------------------------------------------


def build_phikon() -> nn.Module:
    """Build a compact Phikon-style plain ViT-B/16 histopathology tile encoder.

    Returns
    -------
    nn.Module
        A ``timm`` ``VisionTransformer`` (class-token pooled, no classification
        head) at tiny catalog dimensions, in ``eval()`` mode.
    """

    return timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        img_size=64,
        patch_size=16,
        embed_dim=48,
        depth=2,
        num_heads=4,
        num_classes=0,
    ).eval()


def example_input_phikon() -> Tensor:
    """Example histopathology tile.

    Returns
    -------
    Tensor
        Shape ``(1, 3, 64, 64)``.
    """

    return torch.rand(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("NetSurfP-2.0", "build_netsurfp2", "example_input_netsurfp2", "2019", "BIO"),
    ("Omnipose", "build_omnipose", "example_input_omnipose", "2022", "BIO"),
    ("pDeep", "build_pdeep", "example_input_pdeep", "2019", "BIO"),
    (
        "PEPPER-Margin-DeepVariant",
        "build_pepper",
        "example_input_pepper",
        "2021",
        "BIO",
    ),
    ("Phikon", "build_phikon", "example_input_phikon", "2023", "BIO"),
]
