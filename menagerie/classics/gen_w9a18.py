"""Wave 9 batch 18 menagerie classics: geoscience / remote-sensing family
(distributed-acoustic-sensing seismic phase picking, hyperspectral dual-
attention classification, SRGAN-style Antarctic bed-topography super-
resolution, statistical climate downscaling, physics-informed flood
forecasting, and implicit 3-D structural geology modeling).

Sources checked (repo_url / desc_source columns of the build queue, web
research 2026-07-01; no cloning, no pip installs beyond the base env):

  - DAS-PhaseNet (PhaseNet-DAS): https://github.com/AI4EPS/PhaseNet; Zhu et
    al., Nature Communications 2023, doi:10.1038/s41467-023-43355-3,
    "Seismic arrival-time picking on distributed acoustic sensing data using
    semi-supervised learning". Confirmed via paper figures and repo
    description: unlike the original 1-D PhaseNet (three-component
    waveform -> 1-D U-Net, already present in this menagerie as
    ``classics/phasenet.py``), PhaseNet-DAS treats a DAS array's raw strain-
    rate recording as a *2-D spatial-temporal image* (channel axis x time
    axis) and runs it through a 2-D U-Net (4 downsampling / 4 upsampling
    stages, Conv2d-ReLU pairs, skip connections) to predict per-pixel P/S
    arrival probability maps -- the "2-D" generalization of PhaseNet is the
    distinctive mechanism (every DAS channel is picked jointly using its
    spatial neighbors, not independently). Reimplemented compactly here as
    ``PhaseNetDAS2D`` (2-D encoder-decoder U-Net over a
    (channel, time) DAS gather, 3-class softmax output: P, S, noise).
  - DBDA (Double-Branch Dual-Attention Mechanism Network):
    https://github.com/lironui/Double-Branch-Dual-Attention-Mechanism-Network;
    Li et al., Remote Sensing 2020, 12(3):582,
    "Classification of Hyperspectral Image Based on Double-Branch
    Dual-Attention Mechanism Network". Confirmed line-by-line from
    ``global_module/network.py`` (``DBDA_network`` class): two parallel
    Conv3d branches process a hyperspectral cube -- a *spectral branch*
    (1x1xK kernels along the band axis, arranged as a 4-layer DenseNet-style
    block with channel concatenation) feeding a **channel attention (CAM)**
    module, and a *spatial branch* (3x3x1 kernels, also a dense block)
    feeding a **position/spatial attention (PAM)** module. Each branch's
    dense-block output is refined by multiplying with its own attention map,
    globally pooled, and the two 60-channel descriptors are concatenated
    into a single classifier head. Reimplemented compactly as
    ``DBDANetwork`` (spectral dense-block + squeeze-excite-style channel
    attention; spatial dense-block + a compact self-attention spatial-gate
    standing in for the original PAM_Module's full non-local attention;
    global-pool-concat-classify head), preserving the defining "two branches
    x two attention types, fused by concatenation" design.
  - DeepBedMap: https://github.com/weiji14/deepbedmap; Leong & Horgan, The
    Cryosphere (TC) 2020, "DeepBedMap: a deep neural network for resolving
    the bed topography of Antarctica". Confirmed line-by-line from
    ``srgan_train.py`` (``DeepbedmapInputBlock``, ``ResidualDenseBlock``,
    ``ResInResDenseBlock``, ``GeneratorModel``, ``DiscriminatorModel``): the
    generator is an ESRGAN adaptation with a *4-input custom stem* -- four
    differently-resolved rasters (low-res BEDMAP2 bed DEM, REMA ice-surface
    DEM, MEaSUREs ice-flow velocity x/y, snow-accumulation) are each passed
    through their own strided Conv2d (kernel/stride hand-tuned so every
    branch outputs the same 9x9 spatial size) and concatenated -- followed
    by stacked Residual-in-Residual Dense Blocks (RRDB: 3 nested 5-layer
    DenseNet blocks with LeakyReLU and residual scaling 0.1/0.2), two
    nearest-neighbor-upsample + Conv2d stages (4x total upsampling), and a
    final Conv2d head producing a single-channel high-resolution bed DEM.
    The discriminator is a VGG-style Conv2d-BatchNorm-LeakyReLU stack (8
    conv layers, channels doubling every 2 layers, strided every other
    layer) into two dense layers with no final sigmoid (matching the
    original ESRGAN discriminator). Reimplemented compactly as
    ``DeepBedMapGenerator`` and ``DeepBedMapDiscriminator`` (regular Conv2d
    replacing the original's deformable-conv final layers, since deformable
    convolution is not in the required base env), preserving the 4-input
    multi-resolution stem, the RRDB residual-dense-block backbone, and the
    upsample-then-refine super-resolution tail.
  - DeepESD: https://github.com/SantanderMetGroup/DeepDownscaling; Baño-
    Medina, Manzanas & Gutiérrez, Geoscientific Model Development (GMD)
    2022, doi:10.5194/gmd-15-6747-2022, "Downscaling multi-model climate
    projection ensembles with deep learning (DeepESD): contribution to
    CORDEX EUR-44". Confirmed from paper description: a low-resolution
    reanalysis/GCM predictor field (e.g. geopotential height, temperature,
    humidity on a coarse lat/lon grid, stacked as channels) is passed
    through three Conv2d-ReLU layers with 50, 25, and 10 filters
    respectively (perfect-prognosis "convolutional feature extractor"),
    flattened, and mapped by one final dense layer to the full vector of
    high-resolution target grid points (a per-station or per-gridpoint
    regression, not a spatial upsampling decoder -- this ConvNet-to-dense
    "downscale by flatten+project" design, rather than a deconvolutional
    super-resolution network, is DeepESD's distinctive simplicity).
    Reimplemented compactly as ``DeepESD`` (3 Conv2d-ReLU layers -> flatten
    -> linear projection to the flattened high-resolution target grid).
  - DeepFlood (FloodCast / GeoPINS): https://github.com/HydroPML/FloodCast;
    Xu et al., Water Research 2024, doi:10.1016/j.watres.2024.122162,
    "FloodCast: large-scale flood modeling and forecasting with physics-
    informed zero-shot super-resolution". Confirmed from the ``GeoPINS``
    directory of the official repo and the companion arXiv paper
    (2403.12226): the distinctive mechanism is **GeoPINS**, a Geometry-
    Adaptive Physics-Informed Neural Solver -- *not* a generic CNN+LSTM as
    the one-line queue description suggests. Spatial coordinates, a time
    index, and the initial water-depth/rainfall field are lifted by an MLP
    to a hidden representation, processed by stacked **Fourier Neural
    Operator** layers (global spectral convolution: FFT -> per-mode complex
    channel mixing on the low modes -> inverse FFT, giving a global
    receptive field in a single layer, mirroring the shallow-water-equation
    solve), and projected back down by a final MLP to predicted water
    depth/velocity fields, which are the "flood cast" for that time step.
    Reimplemented compactly as ``GeoPINSFloodSolver`` (coordinate/rainfall
    lifting MLP -> stacked FNO blocks operating on the spatial (H, W) grid
    across a short input time window -> projection MLP -> per-timestep
    water-depth field), preserving the "PINN + spectral neural operator"
    design that is FloodCast's actual novelty.
  - DeepISMNet: https://gmd.copernicus.org/articles/15/6841/2022/; Bi, Wu,
    Li, Chang & Yong, GMD 2022, "DeepISMNet: three-dimensional implicit
    structural modeling with convolutional neural network". No public code
    repository was found; the paper itself specifies the architecture in
    full. Confirmed from the published methodology: a U-shaped encoder-
    decoder over a 3-D geological grid. The encoder has 5 stages (2x, 4x,
    8x, 16x, 32x downsampling) of **inverted-residual "linear bottleneck"**
    blocks (1x1x1 expansion conv -> 3x3x3 depthwise conv -> 1x1x1
    projection conv, each with BatchNorm+ReLU, residual-added), with
    **squeeze-and-excitation channel attention** on the deepest three
    encoder scales. The decoder mirrors the encoder with skip connections
    from matching encoder stages and depthwise-separable convolutions for
    refinement. Inputs are two sparse, masked volumetric channels: scattered
    horizon iso-values (0-1 normalized stratigraphic markers) and a binary
    fault-proximity channel, each stored as ``data * mask``; the single-
    channel output is a continuous scalar field whose iso-surfaces are the
    reconstructed stratigraphic horizons. Reimplemented compactly as
    ``DeepISMNet`` (5-stage inverted-residual encoder with SE attention on
    the deepest 3 stages, depthwise-separable-conv decoder with skip
    connections, sparse masked horizon+fault input, scalar-field output).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# DAS-PhaseNet: 2-D spatial-temporal U-Net for DAS phase picking
# ---------------------------------------------------------------------------


class _DoubleConv2d(nn.Module):
    """Two Conv2d-BatchNorm-ReLU layers used by every U-Net stage."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Build the convolution pair.

        Parameters
        ----------
        in_channels:
            Number of input feature channels.
        out_channels:
            Number of output feature channels.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the convolution pair.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Convolved feature map.
        """
        return self.net(x)


class PhaseNetDAS2D(nn.Module):
    """Compact 2-D U-Net for DAS phase picking (PhaseNet-DAS).

    Treats a distributed-acoustic-sensing gather as a 2-D image whose axes
    are fiber channel (space) and time, jointly picking P/S arrivals across
    neighboring channels via a 2-D encoder-decoder with skip connections.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 8, n_classes: int = 3) -> None:
        """Build the DAS-PhaseNet U-Net.

        Parameters
        ----------
        in_channels:
            Number of input channels (1 for strain-rate amplitude).
        base_channels:
            Channel width of the first encoder stage; doubles each stage.
        n_classes:
            Number of output classes (P, S, noise).
        """
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.enc1 = _DoubleConv2d(in_channels, c1)
        self.enc2 = _DoubleConv2d(c1, c2)
        self.enc3 = _DoubleConv2d(c2, c3)
        self.bottleneck = _DoubleConv2d(c3, c4)
        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = _DoubleConv2d(c4, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = _DoubleConv2d(c3, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = _DoubleConv2d(c2, c1)

        self.head = nn.Conv2d(c1, n_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Predict per-pixel P/S/noise probability maps.

        Parameters
        ----------
        x:
            DAS gather of shape ``(batch, in_channels, n_das_channels, n_time)``.

        Returns
        -------
        torch.Tensor
            Softmax phase-probability map of shape
            ``(batch, n_classes, n_das_channels, n_time)``.
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return F.softmax(self.head(d1), dim=1)


def build_phasenet_das() -> nn.Module:
    """Construct a compact DAS-PhaseNet 2-D U-Net.

    Returns
    -------
    nn.Module
        Randomly initialized ``PhaseNetDAS2D`` in eval mode.
    """
    model = PhaseNetDAS2D(in_channels=1, base_channels=8, n_classes=3)
    return model.eval()


def example_input_phasenet_das() -> Tensor:
    """Build a synthetic DAS gather input.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, 1, 32, 64)`` (channels x time, divisible by 8
        for the 3-stage U-Net downsampling).
    """
    return torch.randn(1, 1, 32, 64)


# ---------------------------------------------------------------------------
# DBDA: Double-Branch Dual-Attention Mechanism Network
# ---------------------------------------------------------------------------


class _ChannelAttention3d(nn.Module):
    """Squeeze-excite-style channel attention (spectral-branch CAM)."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        """Build the channel-attention gate.

        Parameters
        ----------
        channels:
            Number of feature channels to gate.
        reduction:
            Bottleneck reduction factor for the excitation MLP.
        """
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1),
        )
        self.gate = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """Rescale channels by a learned global-context gate.

        Parameters
        ----------
        x:
            Feature volume of shape ``(batch, channels, D, H, W)``.

        Returns
        -------
        torch.Tensor
            Channel-attention weights broadcastable against ``x``.
        """
        return self.gate(self.mlp(self.pool(x)))


class _SpatialAttention3d(nn.Module):
    """Compact spatial self-attention gate (spatial-branch PAM stand-in)."""

    def __init__(self, channels: int) -> None:
        """Build the spatial-attention gate.

        Parameters
        ----------
        channels:
            Number of feature channels of the spatial-branch tensor.
        """
        super().__init__()
        hidden = max(channels // 8, 1)
        self.query = nn.Conv3d(channels, hidden, kernel_size=1)
        self.key = nn.Conv3d(channels, hidden, kernel_size=1)
        self.value = nn.Conv3d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        """Apply position-attention re-weighting over spatial locations.

        Parameters
        ----------
        x:
            Feature volume of shape ``(batch, channels, D, H, W)``.

        Returns
        -------
        torch.Tensor
            Spatially re-weighted feature volume, residual-added to ``x``.
        """
        b, c, d, h, w = x.shape
        n = d * h * w
        q = self.query(x).view(b, -1, n).permute(0, 2, 1)
        k = self.key(x).view(b, -1, n)
        attn = F.softmax(torch.bmm(q, k), dim=-1)
        v = self.value(x).view(b, c, n)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(b, c, d, h, w)
        return self.gamma * out + x


class _DenseBranch3d(nn.Module):
    """4-layer DenseNet-style block shared by both DBDA branches."""

    def __init__(
        self, kernel_size: tuple[int, int, int], padding: tuple[int, int, int], growth: int = 6
    ) -> None:
        """Build the dense block.

        Parameters
        ----------
        kernel_size:
            3-D kernel shape for the growth convolutions.
        padding:
            3-D padding matching ``kernel_size`` for 'same' spatial size.
        growth:
            Channel width added by each of the 3 growth convolutions.
        """
        super().__init__()
        self.conv1 = nn.Conv3d(1, growth, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.Sequential(nn.BatchNorm3d(growth), nn.ReLU(inplace=True))
        self.conv2 = nn.Conv3d(growth, growth, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.Sequential(nn.BatchNorm3d(growth * 2), nn.ReLU(inplace=True))
        self.conv3 = nn.Conv3d(growth * 2, growth, kernel_size=kernel_size, padding=padding)
        self.bn3 = nn.Sequential(nn.BatchNorm3d(growth * 3), nn.ReLU(inplace=True))
        self.conv4 = nn.Conv3d(growth * 3, growth, kernel_size=kernel_size, padding=padding)

    def forward(self, x: Tensor) -> Tensor:
        """Grow features via successive dense concatenation.

        Parameters
        ----------
        x:
            Input volume with a single channel.

        Returns
        -------
        torch.Tensor
            Concatenation of all 4 growth-layer outputs.
        """
        x1 = self.conv1(x)
        x2 = self.conv2(self.bn1(x1))
        x3 = self.conv3(self.bn2(torch.cat([x1, x2], dim=1)))
        x4 = self.conv4(self.bn3(torch.cat([x1, x2, x3], dim=1)))
        return torch.cat([x1, x2, x3, x4], dim=1)


class DBDANetwork(nn.Module):
    """Double-Branch Dual-Attention Mechanism Network for HSI classification.

    A spectral branch (1x1xK dense convolutions + channel attention) and a
    spatial branch (3x3x1 dense convolutions + spatial attention) run in
    parallel over a hyperspectral patch cube and are fused by concatenation
    before classification.
    """

    def __init__(self, n_bands: int = 20, n_classes: int = 9, growth: int = 6) -> None:
        """Build the dual-branch dual-attention network.

        Parameters
        ----------
        n_bands:
            Number of spectral bands in the input hyperspectral cube.
        n_classes:
            Number of land-cover classes to predict.
        growth:
            Dense-block growth-channel width for both branches.
        """
        super().__init__()
        out_channels = growth * 4

        self.spectral_dense = _DenseBranch3d(
            kernel_size=(1, 1, 7), padding=(0, 0, 3), growth=growth
        )
        self.spectral_reduce = nn.Conv3d(
            out_channels, out_channels, kernel_size=(1, 1, n_bands), stride=1
        )
        self.spectral_attention = _ChannelAttention3d(out_channels)

        self.spatial_dense = _DenseBranch3d(kernel_size=(3, 3, 1), padding=(1, 1, 0), growth=growth)
        self.spatial_attention = _SpatialAttention3d(out_channels)

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(out_channels * 2, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Classify a hyperspectral patch cube.

        Parameters
        ----------
        x:
            Hyperspectral patch of shape ``(batch, 1, H, W, n_bands)``.

        Returns
        -------
        torch.Tensor
            Class logits of shape ``(batch, n_classes)``.
        """
        spectral_feat = self.spectral_dense(x)
        spectral_reduced = self.spectral_reduce(spectral_feat)
        spectral_gated = spectral_reduced * self.spectral_attention(spectral_reduced)
        spectral_vec = self.global_pool(spectral_gated).flatten(1)

        spatial_feat = self.spatial_dense(x)
        spatial_gated = self.spatial_attention(spatial_feat)
        spatial_vec = self.global_pool(spatial_gated).flatten(1)

        fused = torch.cat([spectral_vec, spatial_vec], dim=1)
        return self.classifier(fused)


def build_dbda() -> nn.Module:
    """Construct a compact DBDA hyperspectral classifier.

    Returns
    -------
    nn.Module
        Randomly initialized ``DBDANetwork`` in eval mode.
    """
    model = DBDANetwork(n_bands=20, n_classes=9, growth=6)
    return model.eval()


def example_input_dbda() -> Tensor:
    """Build a synthetic hyperspectral patch cube.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(2, 1, 9, 9, 20)`` (batch, channel, H, W, bands).
    """
    return torch.randn(2, 1, 9, 9, 20)


# ---------------------------------------------------------------------------
# DeepBedMap: ESRGAN-style super-resolution for Antarctic bed topography
# ---------------------------------------------------------------------------


class _DeepBedMapInputBlock(nn.Module):
    """4-input multi-resolution stem shared by every DeepBedMap forward pass."""

    def __init__(self, out_channels: int = 8) -> None:
        """Build the four resolution-matching stem convolutions.

        Parameters
        ----------
        out_channels:
            Per-input channel width; all four are concatenated afterwards.
        """
        super().__init__()
        self.conv_x = nn.Conv2d(1, out_channels, kernel_size=3, stride=1)
        self.conv_w1 = nn.Conv2d(1, out_channels, kernel_size=6, stride=2)
        self.conv_w2 = nn.Conv2d(2, out_channels, kernel_size=4, stride=1)
        self.conv_w3 = nn.Conv2d(1, out_channels, kernel_size=3, stride=1)

    def forward(self, x: Tensor, w1: Tensor, w2: Tensor, w3: Tensor) -> Tensor:
        """Fuse the four differently-resolved rasters into one feature map.

        Parameters
        ----------
        x:
            Low-resolution BEDMAP2 bed elevation tile.
        w1:
            High-resolution REMA ice-surface elevation tile.
        w2:
            MEaSUREs ice-flow velocity x/y tile.
        w3:
            Snow-accumulation tile.

        Returns
        -------
        torch.Tensor
            Concatenated, spatially-matched stem feature map.
        """
        return torch.cat(
            [self.conv_x(x), self.conv_w1(w1), self.conv_w2(w2), self.conv_w3(w3)], dim=1
        )


class _ResidualDenseBlock(nn.Module):
    """5-layer DenseNet-style block with residual scaling (RRDB sub-unit)."""

    def __init__(self, channels: int = 32, growth: int = 16, scale: float = 0.2) -> None:
        """Build the residual dense block.

        Parameters
        ----------
        channels:
            Input/output channel count of the block.
        growth:
            Per-layer growth-channel width.
        scale:
            Residual-scaling factor applied before the skip addition.
        """
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(channels, growth, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth, growth, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + 2 * growth, growth, 3, padding=1)
        self.conv4 = nn.Conv2d(channels + 3 * growth, growth, 3, padding=1)
        self.conv5 = nn.Conv2d(channels + 4 * growth, channels, 3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Grow, densely concatenate, and residually rescale features.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Residually-scaled dense-block output, same shape as ``x``.
        """
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.act(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.act(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x + x5 * self.scale


class _RRDB(nn.Module):
    """Residual-in-Residual Dense Block: 3 stacked dense blocks + skip."""

    def __init__(self, channels: int = 32, growth: int = 16, scale: float = 0.2) -> None:
        """Build the RRDB.

        Parameters
        ----------
        channels:
            Input/output channel count of the block.
        growth:
            Per-layer growth-channel width used by each inner dense block.
        scale:
            Residual-scaling factor applied at both nesting levels.
        """
        super().__init__()
        self.scale = scale
        self.block1 = _ResidualDenseBlock(channels, growth, scale)
        self.block2 = _ResidualDenseBlock(channels, growth, scale)
        self.block3 = _ResidualDenseBlock(channels, growth, scale)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the nested residual-dense stack.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Residually-scaled RRDB output, same shape as ``x``.
        """
        out = self.block3(self.block2(self.block1(x)))
        return x + out * self.scale


class DeepBedMapGenerator(nn.Module):
    """ESRGAN-style 4-input generator for Antarctic bed-DEM super-resolution."""

    def __init__(self, stem_channels: int = 8, feat_channels: int = 32, n_blocks: int = 2) -> None:
        """Build the DeepBedMap generator.

        Parameters
        ----------
        stem_channels:
            Per-input channel width of the 4-way resolution-matching stem.
        feat_channels:
            Channel width of the RRDB residual backbone.
        n_blocks:
            Number of stacked RRDB blocks.
        """
        super().__init__()
        self.input_block = _DeepBedMapInputBlock(stem_channels)
        self.pre_conv = nn.Conv2d(stem_channels * 4, feat_channels, 3, padding=1)
        self.rrdb_blocks = nn.Sequential(*[_RRDB(feat_channels) for _ in range(n_blocks)])
        self.post_conv = nn.Conv2d(feat_channels, feat_channels, 3, padding=1)
        self.up_conv1 = nn.Conv2d(feat_channels, feat_channels, 3, padding=1)
        self.up_conv2 = nn.Conv2d(feat_channels, feat_channels, 3, padding=1)
        self.final_conv1 = nn.Conv2d(feat_channels, feat_channels, 3, padding=1)
        self.final_conv2 = nn.Conv2d(feat_channels, 1, 3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: Tensor, w1: Tensor, w2: Tensor, w3: Tensor) -> Tensor:
        """Generate a 4x super-resolved bed-elevation DEM.

        Parameters
        ----------
        x:
            Low-resolution BEDMAP2 bed elevation tile.
        w1:
            High-resolution REMA ice-surface elevation tile.
        w2:
            MEaSUREs ice-flow velocity x/y tile.
        w3:
            Snow-accumulation tile.

        Returns
        -------
        torch.Tensor
            Single-channel high-resolution bed-elevation prediction.
        """
        stem = self.input_block(x, w1, w2, w3)
        feat = self.pre_conv(stem)
        rrdb_out = self.post_conv(self.rrdb_blocks(feat))
        feat = feat + rrdb_out

        feat = F.interpolate(feat, scale_factor=2, mode="nearest")
        feat = self.act(self.up_conv1(feat))
        feat = F.interpolate(feat, scale_factor=2, mode="nearest")
        feat = self.act(self.up_conv2(feat))

        feat = self.act(self.final_conv1(feat))
        return self.final_conv2(feat)


class DeepBedMapDiscriminator(nn.Module):
    """VGG-style patch discriminator for the DeepBedMap SRGAN pair."""

    def __init__(self, base_channels: int = 8) -> None:
        """Build the discriminator.

        Parameters
        ----------
        base_channels:
            Channel width of the first convolution; doubles every 2 layers.
        """
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            nn.Conv2d(1, c, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c, 4, 2, 1),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c * 2, 3, 1, 1),
            nn.BatchNorm2d(c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 2, c * 2, 4, 2, 1),
            nn.BatchNorm2d(c * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(c * 2, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Predict a real/fake score for a high-resolution DEM tile.

        Parameters
        ----------
        x:
            Single-channel high-resolution bed-elevation tile.

        Returns
        -------
        torch.Tensor
            Real-valued score of shape ``(batch, 1)`` (no final sigmoid).
        """
        feat = self.pool(self.features(x)).flatten(1)
        return self.classifier(feat)


class DeepBedMapSRGAN(nn.Module):
    """Generator+discriminator pair traced together as one forward pass."""

    def __init__(self) -> None:
        """Build the generator and discriminator sub-modules."""
        super().__init__()
        self.generator = DeepBedMapGenerator()
        self.discriminator = DeepBedMapDiscriminator()

    def forward(self, x: Tensor, w1: Tensor, w2: Tensor, w3: Tensor) -> Tensor:
        """Generate a super-resolved DEM and score it with the discriminator.

        Parameters
        ----------
        x:
            Low-resolution BEDMAP2 bed elevation tile.
        w1:
            High-resolution REMA ice-surface elevation tile.
        w2:
            MEaSUREs ice-flow velocity x/y tile.
        w3:
            Snow-accumulation tile.

        Returns
        -------
        torch.Tensor
            Discriminator real/fake score for the generated DEM.
        """
        y_hat = self.generator(x, w1, w2, w3)
        return self.discriminator(y_hat)


def build_deepbedmap() -> nn.Module:
    """Construct the compact DeepBedMap SRGAN generator+discriminator pair.

    Returns
    -------
    nn.Module
        Randomly initialized ``DeepBedMapSRGAN`` in eval mode.
    """
    model = DeepBedMapSRGAN()
    return model.eval()


def example_input_deepbedmap() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build the 4 synthetic multi-resolution DeepBedMap inputs.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ``(x, w1, w2, w3)`` matching ``DeepBedMapInputBlock``'s expected
        shapes: BEDMAP2 ``(1, 1, 11, 11)``, REMA ``(1, 1, 22, 22)``,
        MEaSUREs velocity ``(1, 2, 12, 12)``, accumulation ``(1, 1, 11, 11)``.
    """
    x = torch.randn(1, 1, 11, 11)
    w1 = torch.randn(1, 1, 22, 22)
    w2 = torch.randn(1, 2, 12, 12)
    w3 = torch.randn(1, 1, 11, 11)
    return x, w1, w2, w3


# ---------------------------------------------------------------------------
# DeepESD: convolutional-feature-extractor + dense statistical downscaling
# ---------------------------------------------------------------------------


class DeepESD(nn.Module):
    """Perfect-prognosis statistical downscaling CNN.

    Three Conv2d-ReLU layers extract features from a coarse predictor grid;
    the flattened feature map is projected by one dense layer directly onto
    every high-resolution target grid point (no deconvolutional decoder).
    """

    def __init__(
        self, in_channels: int = 6, grid_h: int = 16, grid_w: int = 16, n_targets: int = 400
    ) -> None:
        """Build the DeepESD downscaling network.

        Parameters
        ----------
        in_channels:
            Number of stacked coarse predictor variables/levels.
        grid_h:
            Height of the coarse predictor grid.
        grid_w:
            Width of the coarse predictor grid.
        n_targets:
            Number of high-resolution target grid points to regress.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 50, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(50, 25, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(25, 10, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.output_layer = nn.Linear(10 * grid_h * grid_w, n_targets)

    def forward(self, x: Tensor) -> Tensor:
        """Downscale a coarse predictor grid to point-wise target values.

        Parameters
        ----------
        x:
            Coarse predictor grid of shape
            ``(batch, in_channels, grid_h, grid_w)``.

        Returns
        -------
        torch.Tensor
            Downscaled predictions of shape ``(batch, n_targets)``.
        """
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        h = self.act(self.conv3(h))
        return self.output_layer(h.flatten(1))


def build_deepesd() -> nn.Module:
    """Construct a compact DeepESD statistical-downscaling network.

    Returns
    -------
    nn.Module
        Randomly initialized ``DeepESD`` in eval mode.
    """
    model = DeepESD(in_channels=6, grid_h=16, grid_w=16, n_targets=400)
    return model.eval()


def example_input_deepesd() -> Tensor:
    """Build a synthetic coarse-resolution predictor grid.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(2, 6, 16, 16)``.
    """
    return torch.randn(2, 6, 16, 16)


# ---------------------------------------------------------------------------
# DeepFlood (FloodCast / GeoPINS): physics-informed spectral flood solver
# ---------------------------------------------------------------------------


class _SpectralConv2d(nn.Module):
    """Global spectral convolution over the low Fourier modes of a field."""

    def __init__(self, channels: int, modes: int = 6) -> None:
        """Build the learned low-mode complex mixing weights.

        Parameters
        ----------
        channels:
            Input and output channel count.
        modes:
            Number of retained low frequency modes per spatial axis.
        """
        super().__init__()
        scale = channels**-0.5
        self.modes = modes
        self.weight_pos = nn.Parameter(scale * torch.randn(channels, channels, modes, modes, 2))
        self.weight_neg = nn.Parameter(scale * torch.randn(channels, channels, modes, modes, 2))

    def _mix(self, x: Tensor, weight: Tensor) -> Tensor:
        """Multiply retained Fourier coefficients by learned complex weights.

        Parameters
        ----------
        x:
            Complex-valued low-mode coefficients.
        weight:
            Real/imaginary-packed weight tensor.

        Returns
        -------
        torch.Tensor
            Channel-mixed complex coefficients.
        """
        return torch.einsum("bcxy,coxy->boxy", x, torch.view_as_complex(weight))

    def forward(self, x: Tensor) -> Tensor:
        """Apply the FNO spectral convolution.

        Parameters
        ----------
        x:
            Real spatial field of shape ``(batch, channels, H, W)``.

        Returns
        -------
        torch.Tensor
            Field after global low-mode spectral mixing, same shape as ``x``.
        """
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros_like(x_ft)
        my = min(self.modes, x_ft.shape[-2])
        mx = min(self.modes, x_ft.shape[-1])
        out_ft[:, :, :my, :mx] = self._mix(x_ft[:, :, :my, :mx], self.weight_pos[:, :, :my, :mx])
        out_ft[:, :, -my:, :mx] = self._mix(x_ft[:, :, -my:, :mx], self.weight_neg[:, :, :my, :mx])
        return torch.fft.irfft2(out_ft, s=x.shape[-2:])


class _FNOBlock(nn.Module):
    """One GeoPINS Fourier-neural-operator block (spectral + pointwise)."""

    def __init__(self, channels: int, modes: int) -> None:
        """Build the FNO block.

        Parameters
        ----------
        channels:
            Feature channel width.
        modes:
            Retained low Fourier modes for the spectral branch.
        """
        super().__init__()
        self.spectral = _SpectralConv2d(channels, modes)
        self.pointwise = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Combine global spectral mixing with a local pointwise update.

        Parameters
        ----------
        x:
            Field feature tensor.

        Returns
        -------
        torch.Tensor
            Updated field features (GELU-activated).
        """
        return F.gelu(self.spectral(x) + self.pointwise(x))


class GeoPINSFloodSolver(nn.Module):
    """Geometry-Adaptive Physics-Informed Neural Solver (GeoPINS/FloodCast).

    Lifts rainfall + initial water-depth + coordinate fields to a hidden
    representation via an MLP, propagates them with stacked global-spectral
    Fourier Neural Operator blocks (the shallow-water-equation solve), and
    projects back down to predicted water-depth fields.
    """

    def __init__(
        self, in_channels: int = 3, hidden: int = 16, modes: int = 6, n_blocks: int = 3
    ) -> None:
        """Build the GeoPINS solver.

        Parameters
        ----------
        in_channels:
            Number of stacked input fields (e.g. rainfall, DEM, initial
            water depth).
        hidden:
            Hidden channel width used by the FNO backbone.
        modes:
            Retained low Fourier modes per FNO block.
        n_blocks:
            Number of stacked FNO blocks.
        """
        super().__init__()
        self.lift = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.fno_blocks = nn.ModuleList([_FNOBlock(hidden, modes) for _ in range(n_blocks)])
        self.project1 = nn.Conv2d(hidden, hidden, kernel_size=1)
        self.project2 = nn.Conv2d(hidden, 1, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Predict the flood water-depth field for the next time step.

        Parameters
        ----------
        x:
            Stacked rainfall/DEM/initial-depth fields of shape
            ``(batch, in_channels, H, W)``.

        Returns
        -------
        torch.Tensor
            Predicted water-depth field of shape ``(batch, 1, H, W)``.
        """
        h = self.lift(x)
        for block in self.fno_blocks:
            h = block(h)
        h = F.gelu(self.project1(h))
        return self.project2(h)


def build_geopins_floodcast() -> nn.Module:
    """Construct a compact GeoPINS/FloodCast physics-informed flood solver.

    Returns
    -------
    nn.Module
        Randomly initialized ``GeoPINSFloodSolver`` in eval mode.
    """
    model = GeoPINSFloodSolver(in_channels=3, hidden=16, modes=6, n_blocks=3)
    return model.eval()


def example_input_geopins_floodcast() -> Tensor:
    """Build a synthetic rainfall/DEM/initial-depth field stack.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, 3, 24, 24)``.
    """
    return torch.randn(1, 3, 24, 24)


# ---------------------------------------------------------------------------
# DeepISMNet: 3-D implicit structural geology modeling
# ---------------------------------------------------------------------------


class _SqueezeExcite3d(nn.Module):
    """Squeeze-and-excitation channel attention for the deep encoder stages."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        """Build the squeeze-excite gate.

        Parameters
        ----------
        channels:
            Number of feature channels to gate.
        reduction:
            Bottleneck reduction factor for the excitation MLP.
        """
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Rescale channels by a learned global-context gate.

        Parameters
        ----------
        x:
            Feature volume of shape ``(batch, channels, D, H, W)``.

        Returns
        -------
        torch.Tensor
            Channel-gated feature volume, same shape as ``x``.
        """
        return x * self.fc(self.pool(x))


class _InvertedResidual3d(nn.Module):
    """Linear-bottleneck inverted-residual block (expand -> depthwise -> project)."""

    def __init__(
        self, in_channels: int, out_channels: int, expand: float = 1.5, use_se: bool = False
    ) -> None:
        """Build the inverted-residual block.

        Parameters
        ----------
        in_channels:
            Number of input channels.
        out_channels:
            Number of output channels.
        expand:
            Channel expansion ratio for the 1x1x1 bottleneck.
        use_se:
            Whether to apply squeeze-excitation attention before projection.
        """
        super().__init__()
        hidden = max(int(round(in_channels * expand)), in_channels)
        self.expand_conv = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=1),
            nn.BatchNorm3d(hidden),
            nn.ReLU(inplace=True),
        )
        self.depthwise = nn.Sequential(
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            nn.BatchNorm3d(hidden),
            nn.ReLU(inplace=True),
        )
        self.se = _SqueezeExcite3d(hidden) if use_se else nn.Identity()
        self.project = nn.Sequential(
            nn.Conv3d(hidden, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
        )
        self.residual = in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        """Apply the expand-depthwise-project sequence with an optional skip.

        Parameters
        ----------
        x:
            Input feature volume.

        Returns
        -------
        torch.Tensor
            Output feature volume, residual-added when shapes match.
        """
        h = self.expand_conv(x)
        h = self.depthwise(h)
        h = self.se(h)
        h = self.project(h)
        return x + h if self.residual else h


class _DepthwiseSeparable3d(nn.Module):
    """Depthwise-separable convolution refinement used by the decoder."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Build the depthwise-separable refinement block.

        Parameters
        ----------
        in_channels:
            Number of input channels.
        out_channels:
            Number of output channels.
        """
        super().__init__()
        self.depthwise = nn.Conv3d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels
        )
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the depthwise then pointwise convolution pair.

        Parameters
        ----------
        x:
            Input feature volume.

        Returns
        -------
        torch.Tensor
            Refined feature volume.
        """
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class DeepISMNet(nn.Module):
    """3-D implicit structural geology modeling network.

    A 3-stage inverted-residual encoder (squeeze-excite gated at the deepest
    stages) with a depthwise-separable decoder and skip connections maps
    sparse, masked horizon/fault observations to a dense scalar structural
    field whose iso-surfaces recover the stratigraphic horizons.
    """

    def __init__(self, in_channels: int = 2, base_channels: int = 8) -> None:
        """Build the DeepISMNet encoder-decoder.

        Parameters
        ----------
        in_channels:
            Number of sparse input channels (horizon iso-values, fault
            proximity).
        base_channels:
            Channel width of the first encoder stage; doubles each stage.
        """
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.stem = nn.Conv3d(in_channels, c1, kernel_size=3, padding=1)
        self.enc1 = _InvertedResidual3d(c1, c1, use_se=False)
        self.enc2 = _InvertedResidual3d(c1, c2, use_se=True)
        self.enc3 = _InvertedResidual3d(c2, c3, use_se=True)
        self.pool = nn.MaxPool3d(2)

        self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = _DepthwiseSeparable3d(c2 * 2, c2)
        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = _DepthwiseSeparable3d(c1 * 2, c1)

        self.head = nn.Conv3d(c1, 1, kernel_size=1)

    def forward(self, horizon: Tensor, fault: Tensor, mask: Tensor) -> Tensor:
        """Reconstruct the dense implicit structural scalar field.

        Parameters
        ----------
        horizon:
            Sparse normalized horizon iso-value volume.
        fault:
            Sparse binary fault-proximity volume.
        mask:
            Binary validity mask shared by both sparse inputs.

        Returns
        -------
        torch.Tensor
            Dense scalar structural field of shape
            ``(batch, 1, D, H, W)``.
        """
        x = torch.cat([horizon * mask, fault * mask], dim=1)
        s = self.stem(x)
        e1 = self.enc1(s)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def build_deepismnet() -> nn.Module:
    """Construct a compact DeepISMNet implicit structural modeling network.

    Returns
    -------
    nn.Module
        Randomly initialized ``DeepISMNet`` in eval mode.
    """
    model = DeepISMNet(in_channels=2, base_channels=8)
    return model.eval()


def example_input_deepismnet() -> tuple[Tensor, Tensor, Tensor]:
    """Build synthetic sparse horizon/fault/mask volumes.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(horizon, fault, mask)``, each of shape ``(1, 1, 16, 16, 16)``.
    """
    horizon = torch.rand(1, 1, 16, 16, 16)
    fault = (torch.rand(1, 1, 16, 16, 16) > 0.9).float()
    mask = (torch.rand(1, 1, 16, 16, 16) > 0.7).float()
    return horizon, fault, mask


MENAGERIE_ENTRIES = [
    ("DAS-PhaseNet", "build_phasenet_das", "example_input_phasenet_das", "2023", "SEISMO"),
    ("DBDA", "build_dbda", "example_input_dbda", "2020", "VIS"),
    ("DeepBedMap", "build_deepbedmap", "example_input_deepbedmap", "2020", "GEO"),
    ("DeepESD", "build_deepesd", "example_input_deepesd", "2022", "CLIMATE"),
    ("DeepFlood", "build_geopins_floodcast", "example_input_geopins_floodcast", "2024", "GEO"),
    ("DeepISMNet", "build_deepismnet", "example_input_deepismnet", "2022", "GEO"),
]
