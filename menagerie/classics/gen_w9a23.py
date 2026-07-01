"""Compact faithful reimplementations for build_queue rows 139-144 (W9A23).

Sources checked (repo/paper browsed via ``gh api`` / web, no clone/pip-install):
  - HydroGraphNet (cand_01399): NVIDIA PhysicsNeMo example
    ``examples/weather/flood_modeling/hydrographnet``, which trains a
    ``physicsnemo.models.meshgraphnet.meshgraphkan.MeshGraphKAN`` (paper:
    "Interpretable physics-informed graph neural networks for flood
    forecasting", J. of Hydroinformatics / MICE 2024, building on
    "Learning Mesh-Based Simulation with Graph Networks", Pfaff et al.,
    ICLR 2021, arXiv:2010.03409). Browsed
    ``physicsnemo/models/meshgraphnet/{meshgraphnet.py,meshgraphkan.py}``
    directly. Distinctive mechanism: an encode-process-decode mesh graph
    net whose *node encoder* is a Kolmogorov-Arnold Network (KAN) built
    from a bank of learned Fourier harmonics (``num_harmonics``) rather
    than a plain MLP, while the edge encoder/decoder and the L-block
    message-passing processor stay standard MeshGraphNet MLPs: each
    processor block first updates edge features from
    ``[src node, dst node, edge]`` via an edge-MLP with a residual
    connection, then updates node features from ``[node, aggregated
    incoming edge messages]`` via a node-MLP with a residual connection.
    Reimplemented here as a compact PyG-based mesh graph net: a
    ``FourierKANLinear`` node encoder (per-input-feature bank of learned
    cosine harmonics, summed and linearly combined -- the defining KAN
    substitution) feeding a small stack of MLP edge/node message-passing
    blocks with residual updates and a final MLP node decoder, faithfully
    reproducing the KAN-vs-MLP encoder asymmetry that is the model's
    distinctive contribution over a vanilla MeshGraphNet.
  - IceNet (cand_01400): Andersson et al., "Seasonal Arctic sea ice
    forecasting with probabilistic deep learning", Nature Communications
    12, 2021, doi:10.1038/s41467-021-25257-4. Official repo
    github.com/tom-andersson/icenet-paper, ``icenet/models.py``
    (function ``unet_batchnorm``, TensorFlow/Keras). Distinctive
    mechanism: a 5-level U-Net (double-conv + BatchNorm at every
    resolution, nearest-neighbor upsampling + 2x2 conv instead of
    transposed convolution, skip concatenation of the *pre-pool*
    BatchNorm features) whose final 1x1 conv is applied *independently
    per forecast lead time* (a list-comprehension over
    ``n_forecast_months`` producing one logit map each, stacked on a new
    lead-time axis) and then passed through a softmax over sea-ice-
    concentration classes taken along the class axis (not the lead-time
    axis) -- i.e. a shared-trunk U-Net with per-lead-time classification
    heads, trained as a "probabilistic" ensemble (many seeds) rather than
    a stochastic architecture. Reimplemented in PyTorch as
    ``IceNetUNet``: a 4-level BatchNorm U-Net trunk (nearest-upsample +
    conv decoder, matching skip-concat pattern) with one 1x1 conv head
    per forecast month stacked along a lead-time axis, softmax over the
    SIC class axis, at reduced channel widths/depth/months.
  - IGM-CNN (cand_01401): Instructed Glacier Model, Jouvet et al.,
    "Deep learning speeds up ice flow modelling by several orders of
    magnitude", J. of Glaciology 2022 / arXiv:2206.09795. Official org
    repo github.com/instructed-glacier-model/igm,
    ``igm/processes/iceflow/emulate/utils/architectures/cnns.py`` (class
    ``CNN``, TensorFlow/Keras). Distinctive mechanism: IGM replaces the
    classical Shallow-Ice-Approximation + Shallow-Shelf-Approximation
    (SIA+SSA) numerical ice-flow *solver* with a purely convolutional
    *emulator*: a fully-gridded 2D CNN that consumes per-pixel input
    fields (surface elevation, ice thickness, temperature/rheology
    fields, one field per vertical layer) and regresses the horizontal
    ice-velocity components at every vertical layer -- output channels
    equal ``2 * Nz`` (one (u, v) pair per vertical layer) -- via a stack
    of same-padded conv+activation layers with a learned 1x1 "skip
    projection" of the input added back to every layer's output
    (residual-CNN, no pooling: the field stays gridded at input
    resolution throughout, unlike a U-Net). Reimplemented as
    ``IGMCNNEmulator``: an input-projecting residual conv stack (1x1 skip
    projection + N same-padded conv/activation layers, residual add at
    the end) producing ``2 * Nz`` velocity channels from stacked 2D input
    fields, at reduced depth/width/Nz.
  - KARINA (cand_01402): Cheon, Kim, Yang, Yu, Cha, Han, Hong, "KARINA: An
    Efficient Deep Learning Model for Global Weather Forecast",
    arXiv:2403.10555 (2024). Official repo github.com/jmj2316/KARINA,
    ``networks/karina.py`` (classes ``KARINA``/``ConvNeXt``/``Block``/
    ``GeoCyclicPadding``/``SELayer``). Browsed the file directly.
    Distinctive mechanism: a ConvNeXt backbone (LayerNorm-normalized
    inverted-bottleneck depthwise-conv blocks) in which every block's
    depthwise 7x7 conv is preceded by ``GeoCyclicPadding`` -- a
    geometry-aware padding that wraps left/right circularly (longitude
    periodicity) and, for top/bottom (poles), pulls padding rows from the
    *opposite half of the same row* shifted by half the width (a polar
    wrap that respects the sphere's antipodal identification at the
    poles) instead of zero-padding -- and a squeeze-and-excitation (SE)
    channel-attention gate applied right after the depthwise conv, before
    the inverted-bottleneck MLP. Reimplemented here reproducing
    ``GeoCyclicPadding`` exactly (circular longitude wrap + antipodal
    half-shifted polar wrap) feeding SE-gated ConvNeXt blocks at reduced
    channel widths/depths/global-grid resolution.
  - MAResU-Net (cand_01405): Li, Zheng, Duan, Wang, Zhang, Bruzzone,
    "Multiattention Network for Semantic Segmentation of Fine-Resolution
    Remote Sensing Images", IEEE Geoscience and Remote Sensing Letters
    2021. Official repo github.com/lironui/MAResU-Net,
    ``MAResUNet.py``. Browsed the file directly. Distinctive mechanism: a
    ResNet encoder / U-Net-style decoder whose skip connections at every
    decoder stage pass through a ``PAM_CAM_Layer`` that runs a *linear*
    (kernel-trick, softmax-free) position-attention module in parallel
    with a channel-attention module and sums their projected outputs --
    the position attention uses L2-normalized query/key features and the
    associative-matmul-order Taylor-softmax-free "linear attention"
    identity (``(Q (K^T V)) / (1 + Q * sum(K))``) to get full spatial
    self-attention in linear time/memory instead of the quadratic
    ``softmax(QK^T)V`` used by the standard (non-linear) dual-attention
    DANet this architecture descends from. Reimplemented as
    ``MAResUNet``: a small strided-conv encoder stack (standing in for
    the reference's pretrained ResNet-34 stages, since a from-scratch
    catalog entry uses random init rather than ImageNet weights) with
    the reference's exact ``PAM_Module``/``CAM_Module``/``PAM_CAM_Layer``
    linear-attention math and ``DecoderBlock`` upsampling path,
    reduced to small channel widths/spatial size.
  - MC-LSTM (cand_01406): Hoedt, Kratzert, Klotz, Halmich, Holzleitner,
    Nearing, Hochreiter, Klambauer, "MC-LSTM: Mass-Conserving LSTM",
    ICML 2021, arXiv:2101.05186. Official repo github.com/ml-jku/mc-lstm
    (also shipped in the NeuralHydrology library), ``mclstm.py``
    (classes ``MassConservingLSTM``/``MCGate``). Browsed the file
    directly. Distinctive mechanism: an LSTM variant with *no forget
    gate and no output-nonlinearity on the cell*; instead every gate is a
    normalized *matrix*, not a vector, guaranteeing mass conservation by
    construction. At each step: a redistribution matrix ``r`` (out_dim x
    out_dim, row-normalized via softmax) redistributes existing mass
    among cells (``c @ r``), an input gate matrix ``i`` (in_dim x
    out_dim, column-normalized via softmax over the input axis) routes
    new "mass" inputs ``xm`` into cells (``xm @ i``) so total injected
    mass exactly equals ``sum(xm)``, and a sigmoid output gate ``o``
    (out_dim vector, no matrix) releases a fraction of each cell's mass
    as output ``h = o * c`` while the *same* released amount is
    subtracted from the cell state (``c = c - h``) -- so nothing is
    created or destroyed, only redistributed/released. All three gates
    are themselves small linear layers over ``[auxiliary inputs, mass
    inputs, cell state]`` (each mass/state contribution pre-normalized to
    sum to 1) followed by their respective normalizer. Reimplemented as
    ``MassConservingLSTM`` with the exact same redistribution/in-gate/
    out-gate matrix mechanics and mass-conserving cell recurrence, run
    over a short synthetic (mass, auxiliary) input sequence, at reduced
    cell count.

All six models are built with random initialization at small dimensions
purely to capture each architecture's distinctive forward-pass mechanism
for the menagerie catalog; none are trained or intended to produce
meaningful numerical outputs.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# HydroGraphNet (MeshGraphKAN): KAN-node-encoder mesh graph net
# ---------------------------------------------------------------------------


class FourierKANLinear(nn.Module):
    """Kolmogorov-Arnold layer built from a bank of learned Fourier harmonics.

    Each scalar input feature is expanded into ``num_harmonics`` learned
    cosine/sine harmonics, and the output is a learned linear combination
    of all ``in_features * num_harmonics`` harmonic activations -- the KAN
    substitution for a plain ``nn.Linear`` node encoder used by
    MeshGraphKAN (arXiv:2404.19756 for the general KAN idea).

    Parameters
    ----------
    in_features : int
        Number of scalar input features.
    out_features : int
        Number of output features.
    num_harmonics : int
        Number of Fourier harmonics per input feature.
    """

    def __init__(self, in_features: int, out_features: int, num_harmonics: int = 5) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_harmonics = num_harmonics
        freqs = torch.arange(1, num_harmonics + 1, dtype=torch.float32)
        self.register_buffer("freqs", freqs)
        self.coeff_cos = nn.Parameter(torch.randn(out_features, in_features, num_harmonics) * 0.1)
        self.coeff_sin = nn.Parameter(torch.randn(out_features, in_features, num_harmonics) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: Tensor) -> Tensor:
        """Apply the Fourier-KAN transform.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(N, in_features)``.

        Returns
        -------
        Tensor
            Output of shape ``(N, out_features)``.
        """
        # (N, in_features, num_harmonics)
        angles = x.unsqueeze(-1) * self.freqs.view(1, 1, -1)
        cos_terms = torch.cos(angles)
        sin_terms = torch.sin(angles)
        out = torch.einsum("nik,oik->no", cos_terms, self.coeff_cos)
        out = out + torch.einsum("nik,oik->no", sin_terms, self.coeff_sin)
        return out + self.bias


class _MeshMLP(nn.Module):
    """Two-layer MLP with LayerNorm, used for edge/node encoders/processor/decoder."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, normalize: bool = True) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim) if normalize else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Apply the MLP followed by optional LayerNorm."""
        return self.norm(self.net(x))


class MeshGraphKANBlock(nn.Module):
    """One edge-update + node-update message-passing block with residual adds."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.edge_mlp = _MeshMLP(3 * hidden_dim, hidden_dim, hidden_dim)
        self.node_mlp = _MeshMLP(2 * hidden_dim, hidden_dim, hidden_dim)

    def forward(
        self, node_feat: Tensor, edge_feat: Tensor, edge_index: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Run one message-passing block.

        Parameters
        ----------
        node_feat : Tensor
            Node features, shape ``(N, hidden_dim)``.
        edge_feat : Tensor
            Edge features, shape ``(E, hidden_dim)``.
        edge_index : Tensor
            Long tensor of shape ``(2, E)`` giving ``[src, dst]`` node indices.

        Returns
        -------
        tuple of Tensor
            Updated ``(node_feat, edge_feat)``.
        """
        src, dst = edge_index[0], edge_index[1]
        edge_in = torch.cat([node_feat[src], node_feat[dst], edge_feat], dim=-1)
        new_edge_feat = edge_feat + self.edge_mlp(edge_in)

        agg = torch.zeros_like(node_feat)
        agg.index_add_(0, dst, new_edge_feat)
        node_in = torch.cat([node_feat, agg], dim=-1)
        new_node_feat = node_feat + self.node_mlp(node_in)
        return new_node_feat, new_edge_feat


class MeshGraphKAN(nn.Module):
    """Encode-process-decode mesh graph net with a Fourier-KAN node encoder.

    Reproduces PhysicsNeMo's ``MeshGraphKAN`` (HydroGraphNet's model):
    the node encoder is a :class:`FourierKANLinear` layer (Kolmogorov-
    Arnold, harmonic basis) rather than an MLP, while the edge encoder,
    message-passing processor, and node decoder remain standard
    MeshGraphNet MLP blocks.

    Parameters
    ----------
    input_dim_nodes : int
        Number of node input features.
    input_dim_edges : int
        Number of edge input features.
    output_dim : int
        Number of output node features.
    hidden_dim : int
        Hidden width shared by all encoder/processor/decoder blocks.
    processor_size : int
        Number of message-passing blocks.
    num_harmonics : int
        Number of Fourier harmonics in the KAN node encoder.
    """

    def __init__(
        self,
        input_dim_nodes: int = 4,
        input_dim_edges: int = 3,
        output_dim: int = 2,
        hidden_dim: int = 16,
        processor_size: int = 3,
        num_harmonics: int = 5,
    ) -> None:
        super().__init__()
        self.node_encoder = FourierKANLinear(
            input_dim_nodes, hidden_dim, num_harmonics=num_harmonics
        )
        self.edge_encoder = _MeshMLP(input_dim_edges, hidden_dim, hidden_dim)
        self.processor = nn.ModuleList(
            [MeshGraphKANBlock(hidden_dim) for _ in range(processor_size)]
        )
        self.node_decoder = _MeshMLP(hidden_dim, hidden_dim, output_dim, normalize=False)

    def forward(self, node_features: Tensor, edge_features: Tensor, edge_index: Tensor) -> Tensor:
        """Run the encode-process-decode forward pass.

        Parameters
        ----------
        node_features : Tensor
            Shape ``(N, input_dim_nodes)``.
        edge_features : Tensor
            Shape ``(E, input_dim_edges)``.
        edge_index : Tensor
            Long tensor of shape ``(2, E)``.

        Returns
        -------
        Tensor
            Output node features, shape ``(N, output_dim)``.
        """
        node_feat = self.node_encoder(node_features)
        edge_feat = self.edge_encoder(edge_features)
        for block in self.processor:
            node_feat, edge_feat = block(node_feat, edge_feat, edge_index)
        return self.node_decoder(node_feat)


def build_hydrographnet() -> nn.Module:
    """Build a compact HydroGraphNet (MeshGraphKAN) mesh graph net.

    Returns
    -------
    nn.Module
        The model in eval mode.
    """
    torch.manual_seed(0)
    model = MeshGraphKAN(
        input_dim_nodes=4, input_dim_edges=3, output_dim=2, hidden_dim=16, processor_size=3
    )
    return model.eval()


def example_input_hydrographnet() -> tuple[Tensor, Tensor, Tensor]:
    """Build example mesh-graph input for HydroGraphNet.

    Returns
    -------
    tuple of Tensor
        ``(node_features, edge_features, edge_index)`` with 12 nodes and
        20 directed edges.
    """
    torch.manual_seed(0)
    num_nodes, num_edges = 12, 20
    node_features = torch.randn(num_nodes, 4)
    edge_features = torch.randn(num_edges, 3)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    return node_features, edge_features, edge_index


# ---------------------------------------------------------------------------
# IceNet: BatchNorm U-Net with per-lead-time forecast heads
# ---------------------------------------------------------------------------


class _DoubleConvBN(nn.Module):
    """Two same-padded 3x3 convs with ReLU, followed by BatchNorm (icenet's ``unet_batchnorm`` block)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: Tensor) -> Tensor:
        """Apply conv-relu-conv-relu-batchnorm."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.bn(x)


class _UpBlock(nn.Module):
    """Nearest-neighbor upsample + 2x2 conv, concat with skip, then a double conv (icenet decoder stage)."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up_conv = nn.Conv2d(in_ch, out_ch, 2, padding=0)
        self.double_conv = _DoubleConvBN(out_ch + skip_ch, out_ch)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        """Upsample ``x``, concat with ``skip``, and apply the double conv."""
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.pad(x, (0, 1, 0, 1))
        x = self.up_conv(x)
        x = torch.cat([skip, x], dim=1)
        return self.double_conv(x)


class IceNetUNet(nn.Module):
    """BatchNorm U-Net with independent 1x1-conv heads per forecast lead time.

    Reproduces the structure of icenet's ``unet_batchnorm``: a 4-level
    encoder/decoder U-Net trunk (nearest-upsample decoder, BatchNorm after
    every double-conv block, skip concatenation of pre-pool features)
    followed by ``n_forecast_months`` independent 1x1-conv logit heads
    stacked along a new lead-time axis, softmax-normalized over the SIC
    class axis.

    Parameters
    ----------
    in_channels : int
        Number of input climate/observation channels.
    base_filters : int
        Filter count at the first U-Net level.
    n_forecast_months : int
        Number of independent per-lead-time forecast heads.
    n_output_classes : int
        Number of sea-ice-concentration classes per lead time.
    """

    def __init__(
        self,
        in_channels: int = 8,
        base_filters: int = 8,
        n_forecast_months: int = 3,
        n_output_classes: int = 3,
    ) -> None:
        super().__init__()
        f = base_filters
        self.n_forecast_months = n_forecast_months
        self.n_output_classes = n_output_classes

        self.enc1 = _DoubleConvBN(in_channels, f)
        self.enc2 = _DoubleConvBN(f, f * 2)
        self.enc3 = _DoubleConvBN(f * 2, f * 4)
        self.bottleneck = _DoubleConvBN(f * 4, f * 8)

        self.dec3 = _UpBlock(f * 8, f * 4, f * 4)
        self.dec2 = _UpBlock(f * 4, f * 2, f * 2)
        self.dec1 = _UpBlock(f * 2, f, f)

        self.heads = nn.ModuleList(
            [nn.Conv2d(f, n_output_classes, 1) for _ in range(n_forecast_months)]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run the U-Net trunk and per-lead-time heads.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(B, in_channels, H, W)``.

        Returns
        -------
        Tensor
            Softmax-normalized SIC class probabilities, shape
            ``(B, n_output_classes, H, W, n_forecast_months)``.
        """
        e1 = self.enc1(x)
        p1 = F.max_pool2d(e1, 2)
        e2 = self.enc2(p1)
        p2 = F.max_pool2d(e2, 2)
        e3 = self.enc3(p2)
        p3 = F.max_pool2d(e3, 2)
        b = self.bottleneck(p3)

        d3 = self.dec3(b, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        logits = torch.stack([head(d1) for head in self.heads], dim=-1)
        return F.softmax(logits, dim=1)


def build_icenet() -> nn.Module:
    """Build a compact IceNet BatchNorm U-Net with per-lead-time heads.

    Returns
    -------
    nn.Module
        The model in eval mode.
    """
    torch.manual_seed(0)
    model = IceNetUNet(in_channels=8, base_filters=8, n_forecast_months=3, n_output_classes=3)
    return model.eval()


def example_input_icenet() -> Tensor:
    """Build an example gridded climate-input tensor for IceNet.

    Returns
    -------
    Tensor
        Shape ``(1, 8, 32, 32)``.
    """
    torch.manual_seed(0)
    return torch.randn(1, 8, 32, 32)


# ---------------------------------------------------------------------------
# IGM-CNN: residual gridded CNN ice-flow emulator
# ---------------------------------------------------------------------------


class IGMCNNEmulator(nn.Module):
    """Fully-gridded residual CNN ice-flow emulator (IGM's ``CNN`` architecture).

    Reproduces the structure of IGM's ``igm.processes.iceflow.emulate.
    utils.architectures.cnns.CNN``: a stack of same-padded conv+activation
    layers operating on gridded input fields at *fixed* spatial resolution
    (no pooling/downsampling, unlike a U-Net emulator), with a learned 1x1
    "skip projection" of the input added residually to the trunk output.
    The final layer regresses ``2 * Nz`` channels -- one horizontal
    velocity ``(u, v)`` pair per vertical ice layer -- replacing IGM's
    classical SIA+SSA numerical solver.

    Parameters
    ----------
    in_channels : int
        Number of stacked 2D input fields (surface elevation, thickness,
        temperature/rheology fields, etc).
    hidden_channels : int
        Trunk convolution width.
    n_layers : int
        Number of same-padded conv layers in the trunk.
    n_z : int
        Number of vertical ice layers; output channels are ``2 * n_z``.
    """

    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: int = 16,
        n_layers: int = 4,
        n_z: int = 3,
    ) -> None:
        super().__init__()
        self.n_z = n_z
        out_channels = 2 * n_z

        self.skip_proj = nn.Conv2d(in_channels, out_channels, 1)

        layers: list[nn.Module] = []
        prev = in_channels
        for i in range(n_layers):
            is_last = i == n_layers - 1
            out_ch = out_channels if is_last else hidden_channels
            layers.append(nn.Conv2d(prev, out_ch, 3, padding=1))
            if not is_last:
                layers.append(nn.LeakyReLU(0.01))
            prev = out_ch
        self.trunk = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Regress per-layer horizontal ice-velocity fields.

        Parameters
        ----------
        x : Tensor
            Gridded input fields, shape ``(B, in_channels, H, W)``.

        Returns
        -------
        Tensor
            Velocity fields, shape ``(B, 2 * n_z, H, W)``.
        """
        return self.trunk(x) + self.skip_proj(x)


def build_igm_cnn() -> nn.Module:
    """Build a compact IGM-CNN residual ice-flow emulator.

    Returns
    -------
    nn.Module
        The model in eval mode.
    """
    torch.manual_seed(0)
    model = IGMCNNEmulator(in_channels=6, hidden_channels=16, n_layers=4, n_z=3)
    return model.eval()


def example_input_igm_cnn() -> Tensor:
    """Build an example gridded glacier-field input tensor for IGM-CNN.

    Returns
    -------
    Tensor
        Shape ``(1, 6, 24, 24)``.
    """
    torch.manual_seed(0)
    return torch.randn(1, 6, 24, 24)


# ---------------------------------------------------------------------------
# KARINA: ConvNeXt + GeoCyclic padding + SE attention weather model
# ---------------------------------------------------------------------------


class GeoCyclicPadding(nn.Module):
    """Sphere-aware padding: circular in longitude, antipodal half-shift wrap at the poles.

    Reproduces KARINA's ``GeoCyclicPadding`` exactly: left/right padding
    wraps circularly (the grid is periodic in longitude); top/bottom
    padding is built by taking rows near the pole and shifting them by
    half the (padded) width, approximating the antipodal identification
    of points across the pole on a lat-lon grid, instead of zero-padding.

    Parameters
    ----------
    pad_width : int
        Padding amount on every side.
    """

    def __init__(self, pad_width: int) -> None:
        super().__init__()
        self.pad_width = pad_width

    def forward(self, x: Tensor) -> Tensor:
        """Apply geo-cyclic padding.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(B, C, H, W)``.

        Returns
        -------
        Tensor
            Padded output of shape ``(B, C, H + 2*pad, W + 2*pad)``.
        """
        b, c, h, w = x.shape
        pw = self.pad_width
        circular = torch.cat([x[:, :, :, -pw:], x, x[:, :, :, :pw]], dim=3)
        padded_w = circular.shape[3]

        top_bottom = torch.zeros(b, c, h + 2 * pw, padded_w, dtype=x.dtype, device=x.device)
        top_bottom[:, :, pw : h + pw, :] = circular

        mid = padded_w // 2
        for i in range(pw):
            top_row = (pw - i - 1) % h
            top_pad = torch.cat(
                [circular[:, :, top_row, mid:], circular[:, :, top_row, :mid]], dim=-1
            )
            top_bottom[:, :, i, :] = top_pad

            bottom_row = (h - i - 1) % h
            bottom_pad = torch.cat(
                [circular[:, :, bottom_row, mid:], circular[:, :, bottom_row, :mid]], dim=-1
            )
            top_bottom[:, :, h + pw + i, :] = bottom_pad

        return top_bottom


class _SELayer(nn.Module):
    """Squeeze-and-excitation channel gate."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 1), channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the SE gate."""
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class _LayerNormCF(nn.Module):
    """Channels-first LayerNorm (KARINA's ``LayerNorm`` with ``data_format="channels_first"``)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Normalize over the channel dimension."""
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class KARINABlock(nn.Module):
    """GeoCyclic-padded depthwise conv -> SE gate -> ConvNeXt inverted bottleneck."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.pad = GeoCyclicPadding(pad_width=3)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=0, groups=dim)
        self.se = _SELayer(dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Run one geo-cyclic SE-ConvNeXt block with a residual add."""
        residual = x
        x = self.pad(x)
        x = self.dwconv(x)
        x = self.se(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return residual + x


class KARINA(nn.Module):
    """ConvNeXt weather-forecast backbone with geo-cyclic padding and SE attention.

    Reproduces KARINA's ``ConvNeXt``/``KARINA`` classes: a single-stride
    stem, three same-resolution stages of :class:`KARINABlock` (no
    spatial downsampling -- KARINA keeps the full lat-lon grid resolution
    throughout, unlike a standard ConvNeXt classifier), and a small conv
    head regressing the same number of channels as the input (next-step
    forecast of every input weather channel).

    Parameters
    ----------
    in_chans : int
        Number of input weather channels.
    depths : list of int
        Number of :class:`KARINABlock` per stage.
    dim : int
        Channel width shared by all stages.
    """

    def __init__(self, in_chans: int = 8, depths: list[int] | None = None, dim: int = 16) -> None:
        super().__init__()
        if depths is None:
            depths = [2, 2, 2]
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dim, kernel_size=3, stride=1, padding=1),
            _LayerNormCF(dim, eps=1e-6),
        )
        self.stages = nn.ModuleList(
            [nn.Sequential(*[KARINABlock(dim) for _ in range(depth)]) for depth in depths]
        )
        self.head = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, in_chans, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run the geo-cyclic ConvNeXt backbone and regression head.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(B, in_chans, H, W)``.

        Returns
        -------
        Tensor
            Forecast of shape ``(B, in_chans, H, W)``.
        """
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.head(x)


def build_karina() -> nn.Module:
    """Build a compact KARINA global-weather ConvNeXt model.

    Returns
    -------
    nn.Module
        The model in eval mode.
    """
    torch.manual_seed(0)
    model = KARINA(in_chans=8, depths=[2, 2, 2], dim=16)
    return model.eval()


def example_input_karina() -> Tensor:
    """Build an example gridded global-weather input tensor for KARINA.

    Returns
    -------
    Tensor
        Shape ``(1, 8, 18, 36)`` (coarse lat/lon grid).
    """
    torch.manual_seed(0)
    return torch.randn(1, 8, 18, 36)


# ---------------------------------------------------------------------------
# MAResU-Net: ResNet-style encoder/decoder with linear PAM+CAM dual attention
# ---------------------------------------------------------------------------


def _l2_norm(x: Tensor) -> Tensor:
    """L2-normalize each channel-vector along the spatial-token axis."""
    return torch.einsum("bcn, bn->bcn", x, 1.0 / torch.norm(x, p=2, dim=-2))


class PAMModule(nn.Module):
    """Linear (kernel-trick) position-attention module (MAResU-Net's ``PAM_Module``).

    Uses L2-normalized query/key features and the associative-matmul-order
    linear-attention identity to obtain full spatial self-attention in
    linear time/memory instead of quadratic ``softmax(QK^T)V``.

    Parameters
    ----------
    channels : int
        Number of input/output channels.
    scale : int
        Channel reduction factor for the query/key projections.
    """

    def __init__(self, channels: int, scale: int = 8) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.eps = 1e-6
        reduced = max(channels // scale, 1)
        self.query_conv = nn.Conv2d(channels, reduced, 1)
        self.key_conv = nn.Conv2d(channels, reduced, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply linear position attention with a residual gate."""
        b, c, h, w = x.shape
        q = self.query_conv(x).view(b, -1, h * w)
        k = self.key_conv(x).view(b, -1, h * w)
        v = self.value_conv(x).view(b, -1, h * w)

        q = _l2_norm(q).permute(0, 2, 1)
        k = _l2_norm(k)

        tailor_sum = 1.0 / (h * w + torch.einsum("bnc, bc->bn", q, torch.sum(k, dim=-1) + self.eps))
        value_sum = torch.einsum("bcn->bc", v).unsqueeze(-1).expand(-1, c, h * w)

        matrix = torch.einsum("bmn, bcn->bmc", k, v)
        matrix_sum = value_sum + torch.einsum("bnm, bmc->bcn", q, matrix)

        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(b, c, h, w)
        return x + self.gamma * weight_value


class CAMModule(nn.Module):
    """Channel-attention module (MAResU-Net's ``CAM_Module``: quadratic channel-wise softmax attention)."""

    def __init__(self) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        """Apply channel-wise self-attention with a residual gate."""
        b, c, h, w = x.shape
        q = x.view(b, c, -1)
        k = x.view(b, c, -1).permute(0, 2, 1)
        energy = torch.bmm(q, k)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attn = F.softmax(energy_new, dim=-1)
        v = x.view(b, c, -1)
        out = torch.bmm(attn, v).view(b, c, h, w)
        return self.gamma * out + x


class PAMCAMLayer(nn.Module):
    """Fuse linear position attention and channel attention (MAResU-Net's ``PAM_CAM_Layer``)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.pam = PAMModule(channels)
        self.cam = CAMModule()
        self.conv2p = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.ReLU(inplace=True))
        self.conv2c = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.ReLU(inplace=True))

    def forward(self, x: Tensor) -> Tensor:
        """Fuse PAM and CAM branches and project."""
        x = self.conv1(x)
        x = self.conv2p(self.pam(x)) + self.conv2c(self.cam(x))
        return self.conv3(x)


class _ResEncBlock(nn.Module):
    """Small strided residual block standing in for one ResNet-34 stage."""

    def __init__(self, in_ch: int, out_ch: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride), nn.BatchNorm2d(out_ch))
            if (stride != 1 or in_ch != out_ch)
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual block."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class _DecoderBlock(nn.Module):
    """1x1 reduce -> stride-2 transposed conv -> 1x1 expand decoder stage (MAResU-Net's ``DecoderBlock``)."""

    def __init__(self, in_channels: int, n_filters: int) -> None:
        super().__init__()
        mid = max(in_channels // 4, 1)
        self.conv1 = nn.Conv2d(in_channels, mid, 1)
        self.norm1 = nn.BatchNorm2d(mid)
        self.deconv2 = nn.ConvTranspose2d(mid, mid, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)

    def forward(self, x: Tensor) -> Tensor:
        """Apply reduce -> upsample -> expand with ReLU/BatchNorm at each step."""
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.deconv2(x)))
        return F.relu(self.norm3(self.conv3(x)))


class MAResUNet(nn.Module):
    """ResNet-style encoder/decoder segmentation net with linear dual (PAM+CAM) attention skips.

    Reproduces MAResU-Net's structure: a strided-conv encoder stack (here
    a compact from-scratch stand-in for the reference's pretrained
    ResNet-34, since this catalog entry uses random init), a
    :class:`PAMCAMLayer` applied to every encoder stage's features before
    they are summed into the corresponding decoder stage, and a
    :class:`_DecoderBlock` upsampling path with a final conv head.

    Parameters
    ----------
    num_channels : int
        Number of input image channels.
    num_classes : int
        Number of output segmentation classes.
    filters : list of int
        Channel widths at the four encoder/decoder stages.
    """

    def __init__(
        self,
        num_channels: int = 3,
        num_classes: int = 5,
        filters: list[int] | None = None,
    ) -> None:
        super().__init__()
        if filters is None:
            filters = [8, 16, 32, 64]

        self.stem = nn.Sequential(
            nn.Conv2d(num_channels, filters[0], 3, stride=2, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
        )
        self.encoder1 = _ResEncBlock(filters[0], filters[0], stride=1)
        self.encoder2 = _ResEncBlock(filters[0], filters[1], stride=2)
        self.encoder3 = _ResEncBlock(filters[1], filters[2], stride=2)
        self.encoder4 = _ResEncBlock(filters[2], filters[3], stride=2)

        self.attention4 = PAMCAMLayer(filters[3])
        self.attention3 = PAMCAMLayer(filters[2])
        self.attention2 = PAMCAMLayer(filters[1])
        self.attention1 = PAMCAMLayer(filters[0])

        self.decoder4 = _DecoderBlock(filters[3], filters[2])
        self.decoder3 = _DecoderBlock(filters[2], filters[1])
        self.decoder2 = _DecoderBlock(filters[1], filters[0])
        self.decoder1 = _DecoderBlock(filters[0], filters[0])

        self.final_deconv = nn.ConvTranspose2d(filters[0], 8, 4, 2, 1)
        self.final_conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.final_conv3 = nn.Conv2d(8, num_classes, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """Run the encoder/attention/decoder segmentation forward pass.

        Parameters
        ----------
        x : Tensor
            Input image, shape ``(B, num_channels, H, W)``.

        Returns
        -------
        Tensor
            Segmentation logits.
        """
        x1 = self.stem(x)
        e1 = self.encoder1(x1)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e4 = self.attention4(e4)

        d4 = self.decoder4(e4) + self.attention3(e3)
        d3 = self.decoder3(d4) + self.attention2(e2)
        d2 = self.decoder2(d3) + self.attention1(e1)
        d1 = self.decoder1(d2)

        out = F.relu(self.final_deconv(d1))
        out = F.relu(self.final_conv2(out))
        return self.final_conv3(out)


def build_maresunet() -> nn.Module:
    """Build a compact MAResU-Net dual-attention segmentation model.

    Returns
    -------
    nn.Module
        The model in eval mode.
    """
    torch.manual_seed(0)
    model = MAResUNet(num_channels=3, num_classes=5, filters=[8, 16, 32, 64])
    return model.eval()


def example_input_maresunet() -> Tensor:
    """Build an example remote-sensing image tensor for MAResU-Net.

    Returns
    -------
    Tensor
        Shape ``(1, 3, 64, 64)``.
    """
    torch.manual_seed(0)
    return torch.randn(1, 3, 64, 64)


# ---------------------------------------------------------------------------
# MC-LSTM: Mass-Conserving LSTM
# ---------------------------------------------------------------------------


class MCGate(nn.Module):
    """Normalized gate matrix shared by MC-LSTM's redistribution/in-gate/out-gate.

    Computes a linear map over ``[auxiliary inputs, (normalized) mass
    inputs, (normalized) cell state]`` reshaped to ``out_shape`` and
    passed through ``normaliser`` (softmax for matrix gates, sigmoid for
    the vector output gate) -- reproducing ``mclstm.py``'s ``MCGate``.

    Parameters
    ----------
    out_shape : tuple of int
        Shape of the gate output (excluding the batch dimension).
    aux_dim : int
        Number of auxiliary input features.
    out_dim : int or None
        Number of cells, if the cell state feeds this gate.
    in_dim : int or None
        Number of mass inputs, if mass inputs feed this gate.
    normaliser : nn.Module
        Normalizing activation (``nn.Softmax(dim=-1)`` or ``nn.Sigmoid()``).
    """

    def __init__(
        self,
        out_shape: tuple[int, ...],
        aux_dim: int,
        out_dim: int | None = None,
        in_dim: int | None = None,
        normaliser: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.out_shape = out_shape
        self.use_mass = in_dim is not None
        self.use_state = out_dim is not None
        self.normaliser = normaliser if normaliser is not None else nn.Softmax(dim=-1)

        gate_dim = aux_dim
        if self.use_mass:
            gate_dim += in_dim
        if self.use_state:
            gate_dim += out_dim

        n_out = 1
        for s in out_shape:
            n_out *= s
        self.connections = nn.Linear(gate_dim, n_out)

    def forward(self, xm: Tensor, xa: Tensor, c: Tensor) -> Tensor:
        """Compute the normalized gate.

        Parameters
        ----------
        xm : Tensor
            Mass inputs, shape ``(B, in_dim)``.
        xa : Tensor
            Auxiliary inputs, shape ``(B, aux_dim)``.
        c : Tensor
            Cell state, shape ``(B, out_dim)``.

        Returns
        -------
        Tensor
            Normalized gate of shape ``(B, *out_shape)``.
        """
        inputs = [xa]
        if self.use_mass:
            xm_sum = torch.sum(xm, dim=-1, keepdim=True)
            scale = torch.where(xm_sum == 0, torch.ones_like(xm_sum), xm_sum)
            inputs.append(xm / scale)
        if self.use_state:
            c_sum = torch.sum(c, dim=-1, keepdim=True)
            scale = torch.where(c_sum == 0, torch.ones_like(c_sum), c_sum)
            inputs.append(c / scale)

        x = torch.cat(inputs, dim=-1)
        s = self.connections(x)
        s = s.view(x.shape[0], *self.out_shape)
        return self.normaliser(s)


class MassConservingLSTM(nn.Module):
    """Mass-Conserving LSTM: matrix gates guarantee exact mass conservation.

    Reproduces ``mclstm.py``'s ``MassConservingLSTM``: no forget gate and
    no cell nonlinearity; instead a row-normalized redistribution matrix
    moves existing cell mass between cells, a column-normalized input-gate
    matrix routes new mass inputs into cells (so total injected mass
    exactly equals ``sum(xm)``), and a sigmoid output-gate vector releases
    a fraction of each cell's mass as output while subtracting that same
    amount from the cell state.

    Parameters
    ----------
    in_dim : int
        Number of mass inputs per timestep.
    aux_dim : int
        Number of auxiliary (non-mass) inputs per timestep.
    out_dim : int
        Number of cells / outputs.
    """

    def __init__(self, in_dim: int = 1, aux_dim: int = 4, out_dim: int = 8) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.redistribution = MCGate(
            (out_dim, out_dim), aux_dim=aux_dim, out_dim=out_dim, normaliser=nn.Softmax(dim=-1)
        )
        self.in_gate = MCGate(
            (in_dim, out_dim),
            aux_dim=aux_dim,
            out_dim=out_dim,
            in_dim=in_dim,
            normaliser=nn.Softmax(dim=-1),
        )
        self.out_gate = MCGate(
            (out_dim,), aux_dim=aux_dim, out_dim=out_dim, normaliser=nn.Sigmoid()
        )

    def _step(self, xm_t: Tensor, xa_t: Tensor, c_t: Tensor) -> tuple[Tensor, Tensor]:
        """Run one mass-conserving recurrence step."""
        r = self.redistribution(xm_t, xa_t, c_t)
        i = self.in_gate(xm_t, xa_t, c_t)
        o = self.out_gate(xm_t, xa_t, c_t)

        c = torch.matmul(c_t.unsqueeze(-2), r).squeeze(-2)
        c = c + torch.matmul(xm_t.unsqueeze(-2), i).squeeze(-2)
        h = o * c
        c = c - h
        return h, c

    def forward(self, xm: Tensor, xa: Tensor) -> tuple[Tensor, Tensor]:
        """Run the MC-LSTM over a sequence.

        Parameters
        ----------
        xm : Tensor
            Mass inputs, shape ``(T, B, in_dim)``.
        xa : Tensor
            Auxiliary inputs, shape ``(T, B, aux_dim)``.

        Returns
        -------
        tuple of Tensor
            ``(outputs, cell_states)``, each of shape ``(T, B, out_dim)``.
        """
        xm_steps = xm.unbind(dim=0)
        xa_steps = xa.unbind(dim=0)

        c_t = torch.zeros(xm.shape[1], self.out_dim, dtype=xm.dtype, device=xm.device)
        hs, cs = [], []
        for xm_t, xa_t in zip(xm_steps, xa_steps):
            h, c_t = self._step(xm_t, xa_t, c_t)
            hs.append(h)
            cs.append(c_t)

        return torch.stack(hs, dim=0), torch.stack(cs, dim=0)


def build_mc_lstm() -> nn.Module:
    """Build a compact Mass-Conserving LSTM.

    Returns
    -------
    nn.Module
        The model in eval mode.
    """
    torch.manual_seed(0)
    model = MassConservingLSTM(in_dim=1, aux_dim=4, out_dim=8)
    return model.eval()


def example_input_mc_lstm() -> tuple[Tensor, Tensor]:
    """Build an example (mass, auxiliary) input sequence for MC-LSTM.

    Returns
    -------
    tuple of Tensor
        ``(xm, xa)`` of shapes ``(6, 2, 1)`` and ``(6, 2, 4)`` -- a 6-step
        sequence, batch size 2, one mass input (e.g. precipitation) and
        four auxiliary inputs (e.g. temperature, radiation, ...).
    """
    torch.manual_seed(0)
    xm = torch.rand(6, 2, 1)
    xa = torch.randn(6, 2, 4)
    return xm, xa


MENAGERIE_ENTRIES = [
    ("HydroGraphNet", "build_hydrographnet", "example_input_hydrographnet", "2024", "PHYS"),
    ("IceNet", "build_icenet", "example_input_icenet", "2021", "PHYS"),
    (
        "IGM-CNN (Instructed Glacier Model)",
        "build_igm_cnn",
        "example_input_igm_cnn",
        "2022",
        "PHYS",
    ),
    ("KARINA", "build_karina", "example_input_karina", "2024", "PHYS"),
    ("MAResU-Net", "build_maresunet", "example_input_maresunet", "2021", "VIS"),
    ("MC-LSTM", "build_mc_lstm", "example_input_mc_lstm", "2021", "SEQ"),
]
