"""Compact dependency-free classics for install-hostile menagerie rows.

This module contains faithful small PyTorch reconstructions for target models
whose original packages are heavy or dependency-gated.  Each model keeps the
architecture-defining primitive in the traced graph while using reduced widths,
depths, and input sizes so TorchLens can render it in a base CPU environment.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Small feed-forward network used by operator and tabular models."""

    def __init__(self, dims: list[int], activation: type[nn.Module] = nn.GELU) -> None:
        """Initialize a stack of linear layers.

        Parameters
        ----------
        dims:
            Layer dimensions including input and output sizes.
        activation:
            Activation module used between hidden layers.
        """

        super().__init__()
        layers: list[nn.Module] = []
        for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            layers.append(nn.Linear(din, dout))
            if i < len(dims) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the feed-forward network.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Network output.
        """

        return self.net(x)


class DeepONet(nn.Module):
    """DeepONet branch-trunk operator network."""

    def __init__(self, sensors: int = 16, coord_dim: int = 2, width: int = 32) -> None:
        """Initialize branch and trunk subnetworks.

        Parameters
        ----------
        sensors:
            Number of sampled input-function sensors.
        coord_dim:
            Dimension of query coordinates.
        width:
            Shared latent basis width.
        """

        super().__init__()
        self.branch = MLP([sensors, width, width])
        self.trunk = MLP([coord_dim, width, width])
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, data: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Evaluate the learned operator at query coordinates.

        Parameters
        ----------
        data:
            Tuple ``(u, y)`` with sampled function values and query coordinates.

        Returns
        -------
        torch.Tensor
            Operator values for each query point.
        """

        u, y = data
        branch = self.branch(u)
        trunk = self.trunk(y)
        return torch.einsum("bp,bqp->bq", branch, trunk) + self.bias


class CartesianDeepONet(nn.Module):
    """DeepXDE Cartesian-product DeepONet with separable query grid output."""

    def __init__(self, sensors: int = 16, coord_dim: int = 2, width: int = 32) -> None:
        """Initialize branch and trunk networks.

        Parameters
        ----------
        sensors:
            Number of function sensors.
        coord_dim:
            Coordinate dimension.
        width:
            Latent width.
        """

        super().__init__()
        self.branch = MLP([sensors, width, width])
        self.trunk = MLP([coord_dim, width, width])

    def forward(self, data: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Return all batch-by-grid Cartesian products.

        Parameters
        ----------
        data:
            Tuple ``(u, y_grid)``.

        Returns
        -------
        torch.Tensor
            Cartesian-product operator predictions.
        """

        u, y_grid = data
        return torch.einsum("bp,np->bn", self.branch(u), self.trunk(y_grid))


class PODDeepONet(nn.Module):
    """POD-DeepONet with fixed basis reconstruction from branch coefficients."""

    def __init__(self, sensors: int = 16, modes: int = 8, points: int = 12) -> None:
        """Initialize POD coefficient predictor and basis.

        Parameters
        ----------
        sensors:
            Number of input sensors.
        modes:
            Number of POD basis modes.
        points:
            Number of output grid points.
        """

        super().__init__()
        self.coeff = MLP([sensors, 32, modes])
        self.basis = nn.Parameter(torch.randn(modes, points) * 0.02)
        self.bias = nn.Parameter(torch.zeros(points))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Predict a field by combining POD basis modes.

        Parameters
        ----------
        u:
            Sensor values.

        Returns
        -------
        torch.Tensor
            Reconstructed field.
        """

        return self.coeff(u) @ self.basis + self.bias


class MIONet(nn.Module):
    """MIONet multi-input operator network with multiplicative branch fusion."""

    def __init__(self, sensors: int = 12, coord_dim: int = 2, width: int = 32) -> None:
        """Initialize two branch networks and one trunk network.

        Parameters
        ----------
        sensors:
            Sensor count for each input function.
        coord_dim:
            Query coordinate dimension.
        width:
            Shared latent width.
        """

        super().__init__()
        self.branch_a = MLP([sensors, width, width])
        self.branch_b = MLP([sensors, width, width])
        self.trunk = MLP([coord_dim, width, width])

    def forward(self, data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Evaluate a multi-input operator.

        Parameters
        ----------
        data:
            Tuple ``(u_a, u_b, y)``.

        Returns
        -------
        torch.Tensor
            Query predictions.
        """

        ua, ub, y = data
        fused = self.branch_a(ua) * self.branch_b(ub)
        return torch.einsum("bp,bqp->bq", fused, self.trunk(y))


class NBeatsBlock(nn.Module):
    """N-BEATS block with backcast residual and forecast contribution."""

    def __init__(self, backcast: int = 16, forecast: int = 8, theta: int = 16) -> None:
        """Initialize an N-BEATS block.

        Parameters
        ----------
        backcast:
            History length.
        forecast:
            Prediction length.
        theta:
            Latent expansion size.
        """

        super().__init__()
        self.fc = MLP([backcast, 64, 64, theta])
        self.backcast = nn.Linear(theta, backcast)
        self.forecast = nn.Linear(theta, forecast)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce residual backcast and additive forecast.

        Parameters
        ----------
        x:
            Backcast series.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Residual and forecast tensors.
        """

        theta = self.fc(x)
        return x - self.backcast(theta), self.forecast(theta)


class NBeats(nn.Module):
    """Stacked N-BEATS forecaster."""

    def __init__(self) -> None:
        """Initialize two residual N-BEATS blocks."""

        super().__init__()
        self.blocks = nn.ModuleList([NBeatsBlock(), NBeatsBlock()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecast a univariate sequence.

        Parameters
        ----------
        x:
            Input history ``(B, T)``.

        Returns
        -------
        torch.Tensor
            Forecast horizon.
        """

        forecast = x.new_zeros(x.shape[0], 8)
        residual = x
        for block in self.blocks:
            residual, step = block(residual)
            forecast = forecast + step
        return forecast


class TemporalConvNet(nn.Module):
    """Darts-style TCN with dilated causal residual convolutions."""

    def __init__(self, channels: int = 4, hidden: int = 16) -> None:
        """Initialize causal dilated convolution stack.

        Parameters
        ----------
        channels:
            Number of input channels.
        hidden:
            Hidden channel width.
        """

        super().__init__()
        self.in_proj = nn.Conv1d(channels, hidden, 1)
        self.convs = nn.ModuleList(
            [nn.Conv1d(hidden, hidden, 3, dilation=d, padding=2 * d) for d in (1, 2, 4)]
        )
        self.out = nn.Linear(hidden, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a causal TCN and read out the last time step.

        Parameters
        ----------
        x:
            Time series ``(B, T, C)``.

        Returns
        -------
        torch.Tensor
            Prediction logits.
        """

        z = self.in_proj(x.transpose(1, 2))
        for conv in self.convs:
            y = conv(F.gelu(z))[..., : z.shape[-1]]
            z = z + y
        return self.out(z[..., -1])


class TemporalFusionTransformer(nn.Module):
    """Compact Temporal Fusion Transformer with variable gates and attention."""

    def __init__(self, channels: int = 5, hidden: int = 32) -> None:
        """Initialize TFT-style components.

        Parameters
        ----------
        channels:
            Number of covariates.
        hidden:
            Hidden width.
        """

        super().__init__()
        self.var_score = nn.Linear(channels, channels)
        self.input_proj = nn.Linear(channels, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.static_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.attn = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.ff = MLP([hidden, hidden * 2, hidden])
        self.out = nn.Linear(hidden, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecast from multivariate history.

        Parameters
        ----------
        x:
            Sequence ``(B, T, C)``.

        Returns
        -------
        torch.Tensor
            Forecast vector.
        """

        weights = torch.softmax(self.var_score(x.mean(dim=1)), dim=-1).unsqueeze(1)
        z = self.input_proj(x * weights)
        z, _ = self.lstm(z)
        z = z * self.static_gate(z[:, :1, :])
        attn, _ = self.attn(z, z, z)
        z = z + attn
        z = z + self.ff(z)
        return self.out(z[:, -1])


class TiDE(nn.Module):
    """Darts TiDE encoder-decoder for long-horizon dense forecasting."""

    def __init__(self, hist: int = 12, cov: int = 3, horizon: int = 6) -> None:
        """Initialize residual dense encoder and decoder.

        Parameters
        ----------
        hist:
            History length.
        cov:
            Covariate count.
        horizon:
            Forecast horizon.
        """

        super().__init__()
        self.hist = hist
        self.cov = cov
        self.horizon = horizon
        self.encoder = MLP([hist * cov, 64, 64])
        self.decoder = MLP([64 + horizon * cov, 64, horizon])
        self.residual = nn.Linear(hist, horizon)

    def forward(self, data: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Forecast with known future covariates.

        Parameters
        ----------
        data:
            Tuple ``(history, future_covariates)``.

        Returns
        -------
        torch.Tensor
            Forecast horizon.
        """

        hist, fut_cov = data
        enc = self.encoder(hist.flatten(1))
        dec_in = torch.cat([enc, fut_cov.flatten(1)], dim=-1)
        return self.decoder(dec_in) + self.residual(hist[..., 0])


class RepLKBlock(nn.Module):
    """RepLKNet block with depthwise large-kernel convolution."""

    def __init__(self, channels: int, kernel_size: int = 13) -> None:
        """Initialize large-kernel depthwise block.

        Parameters
        ----------
        channels:
            Channel count.
        kernel_size:
            Spatial kernel size.
        """

        super().__init__()
        pad = kernel_size // 2
        self.pre = nn.Conv2d(channels, channels, 1)
        self.dw_large = nn.Conv2d(channels, channels, kernel_size, padding=pad, groups=channels)
        self.dw_small = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.post = nn.Conv2d(channels, channels, 1)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply large-kernel residual mixing.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        torch.Tensor
            Mixed feature map.
        """

        y = F.gelu(self.pre(x))
        y = self.dw_large(y) + self.dw_small(y)
        return x + self.post(F.gelu(self.norm(y)))


class RepLKNet(nn.Module):
    """Compact RepLKNet image classifier."""

    def __init__(self) -> None:
        """Initialize stem, large-kernel stages, and classifier."""

        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 24, 3, stride=2, padding=1), nn.BatchNorm2d(24))
        self.stage1 = nn.Sequential(RepLKBlock(24, 13), RepLKBlock(24, 13))
        self.down = nn.Conv2d(24, 48, 3, stride=2, padding=1)
        self.stage2 = nn.Sequential(RepLKBlock(48, 15), RepLKBlock(48, 15))
        self.head = nn.Linear(48, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify an image.

        Parameters
        ----------
        x:
            Input image.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        x = F.gelu(self.stem(x))
        x = self.stage1(x)
        x = self.stage2(self.down(x))
        return self.head(x.mean(dim=(2, 3)))


class CSPRepBlock(nn.Module):
    """CSPRepRes block used in PP-YOLOE backbones and necks."""

    def __init__(self, channels: int) -> None:
        """Initialize split-transform-merge block.

        Parameters
        ----------
        channels:
            Channel count.
        """

        super().__init__()
        half = channels // 2
        self.left = nn.Sequential(
            nn.Conv2d(half, half, 3, padding=1),
            nn.BatchNorm2d(half),
            nn.SiLU(),
            nn.Conv2d(half, half, 3, padding=1),
        )
        self.right = nn.Conv2d(half, half, 1)
        self.merge = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CSP split and merge.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        torch.Tensor
            Output feature map.
        """

        left, right = x.chunk(2, dim=1)
        return F.silu(self.merge(torch.cat([self.left(left), self.right(right)], dim=1)))


class PPYOLOE(nn.Module):
    """PP-YOLOE detector with CSPRepRes backbone, PAN neck, and ET head."""

    def __init__(self, width: int = 24, classes: int = 5) -> None:
        """Initialize a compact PP-YOLOE-like detector.

        Parameters
        ----------
        width:
            Base channel width.
        classes:
            Number of object classes.
        """

        super().__init__()
        self.stem = nn.Conv2d(3, width, 3, stride=2, padding=1)
        self.stage1 = nn.Sequential(CSPRepBlock(width), nn.Conv2d(width, width * 2, 3, 2, 1))
        self.stage2 = nn.Sequential(
            CSPRepBlock(width * 2), nn.Conv2d(width * 2, width * 4, 3, 2, 1)
        )
        self.lat2 = nn.Conv2d(width * 4, width * 2, 1)
        self.pan = CSPRepBlock(width * 2)
        self.cls_head = nn.Sequential(
            nn.Conv2d(width * 2, width * 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width * 2, classes, 1),
        )
        self.box_head = nn.Sequential(
            nn.Conv2d(width * 2, width * 2, 3, padding=1), nn.SiLU(), nn.Conv2d(width * 2, 4, 1)
        )
        self.obj_head = nn.Conv2d(width * 2, 1, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict anchor-free class, box, and objectness maps.

        Parameters
        ----------
        x:
            Input image.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Detection head outputs.
        """

        p3 = self.stage1(F.silu(self.stem(x)))
        p4 = self.stage2(p3)
        p4_up = F.interpolate(self.lat2(p4), size=p3.shape[-2:], mode="nearest")
        feat = self.pan(p3 + p4_up)
        return self.cls_head(feat), F.relu(self.box_head(feat)), self.obj_head(feat)


class PPYOLOERotated(nn.Module):
    """PP-YOLOE-R detector with a decoupled rotated-box angle path."""

    def __init__(self, width: int = 16, classes: int = 5, angle_bins: int = 8) -> None:
        """Initialize CSPRepRes backbone, ET heads, and oriented-box head.

        Parameters
        ----------
        width:
            Base channel width.
        classes:
            Number of object classes.
        angle_bins:
            Discrete angle bins for DFL-style angle prediction.
        """
        super().__init__()
        self.base = PPYOLOE(width=width, classes=classes)
        self.angle_head = nn.Sequential(
            nn.Conv2d(width * 2, width * 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width * 2, angle_bins, 1),
        )
        self.register_buffer("angle_bins", torch.linspace(-1.5708, 1.5708, angle_bins))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict class, oriented box, objectness, and angle logits.

        Parameters
        ----------
        x:
            Input image.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Class logits, ``ltrb+theta`` boxes, objectness, and angle-bin logits.
        """
        p3 = self.base.stage1(F.silu(self.base.stem(x)))
        p4 = self.base.stage2(p3)
        p4_up = F.interpolate(self.base.lat2(p4), size=p3.shape[-2:], mode="nearest")
        feat = self.base.pan(p3 + p4_up)
        cls = self.base.cls_head(feat)
        box = F.relu(self.base.box_head(feat))
        obj = self.base.obj_head(feat)
        angle_logits = self.angle_head(feat)
        angle = (torch.softmax(angle_logits, dim=1) * self.angle_bins.view(1, -1, 1, 1)).sum(
            dim=1, keepdim=True
        )
        return cls, torch.cat((box, angle), dim=1), obj, angle_logits


class DCGANDiscriminator(nn.Module):
    """DCGAN discriminator with strided conv, BatchNorm, and LeakyReLU."""

    def __init__(self, channels: int = 16) -> None:
        """Initialize discriminator.

        Parameters
        ----------
        channels:
            Base feature channels.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, channels, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels * 2, channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels * 4, 1, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score an image as real or fake.

        Parameters
        ----------
        x:
            Input image.

        Returns
        -------
        torch.Tensor
            Discriminator logits.
        """

        return self.net(x).flatten(1)


class DeepChemGraphConv(nn.Module):
    """DeepChem GraphConvModel with GraphConv, GraphPool, and GraphGather."""

    def __init__(self, node_dim: int = 12, hidden: int = 32) -> None:
        """Initialize graph convolution model.

        Parameters
        ----------
        node_dim:
            Node feature dimension.
        hidden:
            Hidden width.
        """

        super().__init__()
        self.degree_weights = nn.ModuleList([nn.Linear(node_dim, hidden) for _ in range(4)])
        self.pool_proj = nn.Linear(hidden, hidden)
        self.gather_gate = nn.Linear(hidden, 1)
        self.gather_proj = nn.Linear(hidden, hidden)
        self.readout = MLP([hidden * 2, hidden, 1])

    def forward(self, graph: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Run graph convolution and graph-level readout.

        Parameters
        ----------
        graph:
            Tuple ``(node_features, adjacency)``.

        Returns
        -------
        torch.Tensor
            Graph prediction.
        """

        x, adj = graph
        degree = adj.sum(-1).long().clamp(max=len(self.degree_weights) - 1)
        neigh = adj @ x
        h = torch.zeros(
            x.shape[0], x.shape[1], self.degree_weights[0].out_features, device=x.device
        )
        for deg_idx, layer in enumerate(self.degree_weights):
            mask = (degree == deg_idx).unsqueeze(-1).float()
            h = h + F.relu(layer(x + neigh)) * mask
        neighbor_h = h.unsqueeze(1).expand(-1, h.shape[1], -1, -1)
        pooled = (neighbor_h * adj.unsqueeze(-1)).amax(dim=2)
        h = F.relu(h + self.pool_proj(pooled))
        gate = torch.softmax(self.gather_gate(h).squeeze(-1), dim=-1).unsqueeze(-1)
        gathered_sum = (gate * self.gather_proj(h)).sum(dim=1)
        gathered_max = h.max(dim=1).values
        return self.readout(torch.cat((gathered_sum, gathered_max), dim=-1))


class DeepChemMPNN(nn.Module):
    """DeepChem MPNNModel with edge-conditioned messages and GRU updates."""

    def __init__(self, node_dim: int = 12, edge_dim: int = 4, hidden: int = 32) -> None:
        """Initialize message-passing network.

        Parameters
        ----------
        node_dim:
            Node feature dimension.
        edge_dim:
            Edge feature dimension.
        hidden:
            Hidden width.
        """

        super().__init__()
        self.node = nn.Linear(node_dim, hidden)
        self.edge_network = nn.Linear(edge_dim, hidden * hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.set2set_lstm = nn.LSTMCell(hidden * 2, hidden)
        self.out = MLP([hidden * 2, hidden, 1])

    def forward(self, graph: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Run edge-conditioned message passing.

        Parameters
        ----------
        graph:
            Tuple ``(node_features, adjacency, edge_features)``.

        Returns
        -------
        torch.Tensor
            Graph prediction.
        """

        x, adj, edge = graph
        h = F.relu(self.node(x))
        for _ in range(3):
            src = h.unsqueeze(1).expand(-1, h.shape[1], -1, -1)
            edge_kernel = self.edge_network(edge).view(*edge.shape[:-1], h.shape[-1], h.shape[-1])
            m = torch.matmul(src.unsqueeze(-2), edge_kernel).squeeze(-2) * adj.unsqueeze(-1)
            agg = m.sum(dim=2)
            h = self.gru(agg.reshape(-1, h.shape[-1]), h.reshape(-1, h.shape[-1])).view_as(h)
        query = torch.zeros(h.shape[0], h.shape[-1], device=h.device)
        memory = torch.zeros_like(query)
        readout = torch.cat((query, memory), dim=-1)
        for _ in range(2):
            query, memory = self.set2set_lstm(readout, (query, memory))
            weights = torch.softmax(torch.bmm(h, query.unsqueeze(-1)).squeeze(-1), dim=-1)
            context = torch.bmm(weights.unsqueeze(1), h).squeeze(1)
            readout = torch.cat((query, context), dim=-1)
        return self.out(readout)


class DeepChemWeave(nn.Module):
    """DeepChem WeaveModel alternating atom and pair feature updates."""

    def __init__(self, atom_dim: int = 12, pair_dim: int = 5, hidden: int = 24) -> None:
        """Initialize weave atom/pair update layers.

        Parameters
        ----------
        atom_dim:
            Atom feature dimension.
        pair_dim:
            Pair feature dimension.
        hidden:
            Hidden width.
        """

        super().__init__()
        self.atom = nn.Linear(atom_dim + pair_dim, hidden)
        self.pair = nn.Linear(pair_dim + 2 * hidden, hidden)
        self.out = MLP([hidden, hidden, 1])

    def forward(self, graph: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Run weave atom and pair updates.

        Parameters
        ----------
        graph:
            Tuple ``(atom_features, pair_features)``.

        Returns
        -------
        torch.Tensor
            Molecular prediction.
        """

        atoms, pairs = graph
        pair_summary = pairs.mean(dim=2)
        h = F.relu(self.atom(torch.cat([atoms, pair_summary], dim=-1)))
        hi = h.unsqueeze(2).expand(-1, -1, h.shape[1], -1)
        hj = h.unsqueeze(1).expand(-1, h.shape[1], -1, -1)
        p = F.relu(self.pair(torch.cat([pairs, hi, hj], dim=-1)))
        h = h + p.mean(dim=2)[..., : h.shape[-1]]
        return self.out(h.mean(dim=1))


class HiFiGANGenerator(nn.Module):
    """HiFi-GAN/HiFT-style neural vocoder generator."""

    def __init__(self, mel: int = 80, channels: int = 64) -> None:
        """Initialize transposed-conv upsampler with residual MRF blocks.

        Parameters
        ----------
        mel:
            Mel-spectrogram channel count.
        channels:
            Base channel count.
        """

        super().__init__()
        self.pre = nn.Conv1d(mel, channels, 7, padding=3)
        self.up1 = nn.ConvTranspose1d(channels, channels // 2, 8, stride=4, padding=2)
        self.res1 = nn.ModuleList(
            [nn.Conv1d(channels // 2, channels // 2, 3, padding=d, dilation=d) for d in (1, 3, 5)]
        )
        self.up2 = nn.ConvTranspose1d(channels // 2, channels // 4, 8, stride=4, padding=2)
        self.res2 = nn.ModuleList(
            [nn.Conv1d(channels // 4, channels // 4, 3, padding=d, dilation=d) for d in (1, 3, 5)]
        )
        self.post = nn.Conv1d(channels // 4, 1, 7, padding=3)

    def _res_stack(self, x: torch.Tensor, stack: nn.ModuleList) -> torch.Tensor:
        """Average parallel dilated residual branches.

        Parameters
        ----------
        x:
            Input feature map.
        stack:
            Dilated convolution branches.

        Returns
        -------
        torch.Tensor
            Residual branch average.
        """

        acc = x.new_zeros(x.shape)
        for conv in stack:
            acc = acc + F.leaky_relu(conv(F.leaky_relu(x, 0.1)), 0.1)
        return acc / len(stack)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Synthesize waveform from mel features.

        Parameters
        ----------
        mel:
            Mel-spectrogram ``(B, 80, T)``.

        Returns
        -------
        torch.Tensor
            Waveform samples.
        """

        x = F.leaky_relu(self.pre(mel), 0.1)
        x = self._res_stack(F.leaky_relu(self.up1(x), 0.1), self.res1)
        x = self._res_stack(F.leaky_relu(self.up2(x), 0.1), self.res2)
        return torch.tanh(self.post(F.leaky_relu(x, 0.1)))


class HiFTGenerator(nn.Module):
    """CosyVoice HiFTGenerator with NSF harmonic-noise source and iSTFT synthesis."""

    def __init__(
        self, mel: int = 80, channels: int = 48, harmonics: int = 4, n_fft: int = 16
    ) -> None:
        """Initialize F0 predictor, source filter, and iSTFT spectral head.

        Parameters
        ----------
        mel:
            Mel-spectrogram channel count.
        channels:
            Hidden channel count.
        harmonics:
            Number of harmonic sine sources.
        n_fft:
            FFT size used for inverse STFT synthesis.
        """
        super().__init__()
        self.harmonics = harmonics
        self.n_fft = n_fft
        self.f0_predictor = nn.Sequential(
            nn.Conv1d(mel, channels, 3, padding=1), nn.SiLU(), nn.Conv1d(channels, 1, 1)
        )
        self.filter = nn.Sequential(
            nn.Conv1d(mel + harmonics + 1, channels, 3, padding=1), nn.SiLU()
        )
        self.spec_head = nn.Conv1d(channels, n_fft + 2, 1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Synthesize waveform through harmonic-plus-noise source filtering and iSTFT.

        Parameters
        ----------
        mel:
            Mel-spectrogram ``(B, 80, T)``.

        Returns
        -------
        torch.Tensor
            Waveform samples.
        """
        f0 = F.softplus(self.f0_predictor(mel)) + 1.0
        phase = torch.cumsum(f0 / 100.0, dim=-1)
        harmonic_ids = torch.arange(1, self.harmonics + 1, device=mel.device, dtype=mel.dtype).view(
            1, -1, 1
        )
        harmonic_source = torch.sin(phase * harmonic_ids)
        noise_source = torch.randn_like(f0) * 0.003
        hidden = self.filter(torch.cat((mel, harmonic_source, noise_source), dim=1))
        spec = self.spec_head(hidden)
        real, imag = spec.chunk(2, dim=1)
        complex_spec = torch.complex(real, imag)
        return torch.istft(complex_spec, n_fft=self.n_fft, hop_length=4, length=mel.shape[-1] * 4)


class ChebyshevKANLayer(nn.Module):
    """KAN layer using Chebyshev polynomial edge bases."""

    def __init__(self, in_dim: int = 8, out_dim: int = 6, degree: int = 4) -> None:
        """Initialize Chebyshev basis coefficients.

        Parameters
        ----------
        in_dim:
            Input dimension.
        out_dim:
            Output dimension.
        degree:
            Highest polynomial degree.
        """

        super().__init__()
        self.degree = degree
        self.coeff = nn.Parameter(torch.randn(out_dim, in_dim, degree + 1) * 0.05)
        self.base = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply per-edge Chebyshev basis expansion.

        Parameters
        ----------
        x:
            Input features.

        Returns
        -------
        torch.Tensor
            KAN output.
        """

        z = torch.tanh(x)
        bases = [torch.ones_like(z), z]
        for _ in range(2, self.degree + 1):
            bases.append(2.0 * z * bases[-1] - bases[-2])
        basis = torch.stack(bases, dim=-1)
        return self.base(x) + torch.einsum("bid,oid->bo", basis, self.coeff)


class EfficientKANLayer(nn.Module):
    """EfficientKAN layer with direct B-spline basis computation."""

    def __init__(self, in_dim: int = 8, out_dim: int = 6, grids: int = 6, degree: int = 3) -> None:
        """Initialize RBF grid coefficients.

        Parameters
        ----------
        in_dim:
            Input dimension.
        out_dim:
            Output dimension.
        grids:
            Number of grid centers.
        degree:
            B-spline degree.
        """

        super().__init__()
        self.degree = degree
        self.register_buffer("grid", torch.linspace(-1.0, 1.0, grids + degree + 1))
        self.coeff = nn.Parameter(torch.randn(out_dim, in_dim, grids) * 0.03)
        self.base = nn.Linear(in_dim, out_dim)

    def _bspline_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Compute B-spline bases with Cox-de Boor recursion.

        Parameters
        ----------
        x:
            Input features.

        Returns
        -------
        torch.Tensor
            Basis tensor ``(batch, in_dim, grids)``.
        """
        z = torch.tanh(x).unsqueeze(-1)
        basis = ((z >= self.grid[:-1]) & (z < self.grid[1:])).to(x.dtype)
        for degree in range(1, self.degree + 1):
            left_num = z - self.grid[: -(degree + 1)]
            left_den = (self.grid[degree:-1] - self.grid[: -(degree + 1)]).clamp_min(1.0e-6)
            right_num = self.grid[degree + 1 :] - z
            right_den = (self.grid[degree + 1 :] - self.grid[1:-degree]).clamp_min(1.0e-6)
            basis = left_num / left_den * basis[..., :-1] + right_num / right_den * basis[..., 1:]
        return basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply efficient spline-basis KAN layer.

        Parameters
        ----------
        x:
            Input features.

        Returns
        -------
        torch.Tensor
            Layer output.
        """
        basis = self._bspline_basis(x)
        return self.base(x) + torch.einsum("big,oig->bo", basis, self.coeff)


class FastKANLayer(EfficientKANLayer):
    """Backward-compatible EfficientKAN builder using spline bases."""


def build_deeponet() -> nn.Module:
    """Build a compact DeepONet."""

    return DeepONet()


def example_input_deeponet() -> tuple[torch.Tensor, torch.Tensor]:
    """Return DeepONet sample function and query coordinates."""

    return torch.randn(2, 16), torch.randn(2, 12, 2)


def build_cartesian_deeponet() -> nn.Module:
    """Build Cartesian-product DeepONet."""

    return CartesianDeepONet()


def example_input_cartesian_deeponet() -> tuple[torch.Tensor, torch.Tensor]:
    """Return Cartesian DeepONet sample function and grid."""

    return torch.randn(2, 16), torch.randn(12, 2)


def build_pod_deeponet() -> nn.Module:
    """Build POD-DeepONet."""

    return PODDeepONet()


def example_input_pod_deeponet() -> torch.Tensor:
    """Return POD-DeepONet sensor vector."""

    return torch.randn(2, 16)


def build_mionet() -> nn.Module:
    """Build MIONet."""

    return MIONet()


def example_input_mionet() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return MIONet multi-input sample."""

    return torch.randn(2, 12), torch.randn(2, 12), torch.randn(2, 10, 2)


def build_nbeats() -> nn.Module:
    """Build Darts N-BEATS."""

    return NBeats()


def example_input_nbeats() -> torch.Tensor:
    """Return N-BEATS history."""

    return torch.randn(2, 16)


def build_tcn() -> nn.Module:
    """Build Darts TCN."""

    return TemporalConvNet()


def example_input_tcn() -> torch.Tensor:
    """Return TCN history."""

    return torch.randn(2, 16, 4)


def build_tft() -> nn.Module:
    """Build Temporal Fusion Transformer."""

    return TemporalFusionTransformer()


def example_input_tft() -> torch.Tensor:
    """Return TFT multivariate history."""

    return torch.randn(2, 12, 5)


def build_tide() -> nn.Module:
    """Build TiDE."""

    return TiDE()


def example_input_tide() -> tuple[torch.Tensor, torch.Tensor]:
    """Return TiDE history and future covariates."""

    return torch.randn(2, 12, 3), torch.randn(2, 6, 3)


def build_replknet() -> nn.Module:
    """Build RepLKNet."""

    return RepLKNet().eval()


def example_input_image() -> torch.Tensor:
    """Return small RGB image."""

    return torch.randn(1, 3, 32, 32)


def build_ppyoloe_s() -> nn.Module:
    """Build small PP-YOLOE."""

    return PPYOLOE(width=16).eval()


def build_ppyoloe_m() -> nn.Module:
    """Build medium PP-YOLOE."""

    return PPYOLOE(width=24).eval()


def build_ppyoloe_l() -> nn.Module:
    """Build large PP-YOLOE."""

    return PPYOLOE(width=32).eval()


def build_ppyoloe_r() -> nn.Module:
    """Build PP-YOLOE-R with oriented-box angle prediction."""

    return PPYOLOERotated(width=16).eval()


def build_dcgan_discriminator() -> nn.Module:
    """Build DCGAN discriminator."""

    return DCGANDiscriminator().eval()


def example_input_dcgan_discriminator() -> torch.Tensor:
    """Return DCGAN discriminator image."""

    return torch.randn(1, 3, 32, 32)


def _graph_adj(batch: int = 2, nodes: int = 6) -> torch.Tensor:
    """Return a small ring adjacency matrix.

    Parameters
    ----------
    batch:
        Batch size.
    nodes:
        Number of graph nodes.

    Returns
    -------
    torch.Tensor
        Batched adjacency matrix.
    """

    adj = torch.eye(nodes).roll(1, 0) + torch.eye(nodes).roll(-1, 0)
    return adj.unsqueeze(0).repeat(batch, 1, 1)


def build_graphconv() -> nn.Module:
    """Build DeepChem GraphConvModel."""

    return DeepChemGraphConv()


def example_input_graphconv() -> tuple[torch.Tensor, torch.Tensor]:
    """Return graph-convolution sample."""

    return torch.randn(2, 6, 12), _graph_adj()


def build_mpnn() -> nn.Module:
    """Build DeepChem MPNNModel."""

    return DeepChemMPNN()


def example_input_mpnn() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return MPNN sample graph."""

    return torch.randn(2, 6, 12), _graph_adj(), torch.randn(2, 6, 6, 4)


def build_weave() -> nn.Module:
    """Build DeepChem WeaveModel."""

    return DeepChemWeave()


def example_input_weave() -> tuple[torch.Tensor, torch.Tensor]:
    """Return WeaveModel atom and pair features."""

    return torch.randn(2, 6, 12), torch.randn(2, 6, 6, 5)


def build_hifigan() -> nn.Module:
    """Build HiFi-GAN/HiFT generator."""

    return HiFiGANGenerator().eval()


def build_hift_generator() -> nn.Module:
    """Build CosyVoice HiFTGenerator."""

    return HiFTGenerator().eval()


def example_input_mel() -> torch.Tensor:
    """Return compact mel spectrogram."""

    return torch.randn(1, 80, 8)


def build_chebyshev_kan() -> nn.Module:
    """Build Chebyshev KAN layer."""

    return ChebyshevKANLayer()


def build_fastkan() -> nn.Module:
    """Build FastKAN/EfficientKAN spline layer."""

    return FastKANLayer()


def build_efficient_kan() -> nn.Module:
    """Build EfficientKAN spline layer."""

    return EfficientKANLayer()


def example_input_kan() -> torch.Tensor:
    """Return KAN feature batch."""

    return torch.randn(3, 8)


MENAGERIE_ENTRIES = [
    (
        "DeepONet branch-trunk operator network",
        "build_deeponet",
        "example_input_deeponet",
        "2021",
        "DC",
    ),
    (
        "DeepONet Cartesian-product branch-trunk operator network",
        "build_cartesian_deeponet",
        "example_input_cartesian_deeponet",
        "2021",
        "DC",
    ),
    (
        "POD-DeepONet basis-coefficient operator network",
        "build_pod_deeponet",
        "example_input_pod_deeponet",
        "2021",
        "DC",
    ),
    ("MIONet multi-input operator network", "build_mionet", "example_input_mionet", "2022", "DC"),
    (
        "Darts N-BEATS residual backcast forecaster",
        "build_nbeats",
        "example_input_nbeats",
        "2019",
        "DC",
    ),
    (
        "Darts TCN dilated causal convolution forecaster",
        "build_tcn",
        "example_input_tcn",
        "2018",
        "DC",
    ),
    ("Darts Temporal Fusion Transformer", "build_tft", "example_input_tft", "2021", "DC"),
    (
        "Darts TiDE dense encoder-decoder forecaster",
        "build_tide",
        "example_input_tide",
        "2023",
        "DC",
    ),
    ("RepLKNet large-kernel ConvNet", "build_replknet", "example_input_image", "2022", "DC"),
    (
        "PP-YOLOE small CSPRepRes PAN ET-head detector",
        "build_ppyoloe_s",
        "example_input_image",
        "2022",
        "DC",
    ),
    (
        "PP-YOLOE medium CSPRepRes PAN ET-head detector",
        "build_ppyoloe_m",
        "example_input_image",
        "2022",
        "DC",
    ),
    (
        "PP-YOLOE large CSPRepRes PAN ET-head detector",
        "build_ppyoloe_l",
        "example_input_image",
        "2022",
        "DC",
    ),
    ("ppyoloe_r_crn_s", "build_ppyoloe_r", "example_input_image", "2023", "DC"),
    (
        "DCGAN discriminator strided-conv classifier",
        "build_dcgan_discriminator",
        "example_input_dcgan_discriminator",
        "2015",
        "DC",
    ),
    ("DeepChem GraphConvModel", "build_graphconv", "example_input_graphconv", "2015", "DC"),
    ("DeepChem MPNNModel", "build_mpnn", "example_input_mpnn", "2017", "DC"),
    ("DeepChem WeaveModel", "build_weave", "example_input_weave", "2016", "DC"),
    ("HiFi-GAN / HiFT generator", "build_hifigan", "example_input_mel", "2020", "DC"),
    ("CosyVoice_HiFTGenerator", "build_hift_generator", "example_input_mel", "2024", "DC"),
    ("Chebyshev KAN layer", "build_chebyshev_kan", "example_input_kan", "2024", "DC"),
    ("FastKAN / EfficientKAN RBF layer", "build_fastkan", "example_input_kan", "2024", "DC"),
    ("efficient-kan:EfficientKAN", "build_efficient_kan", "example_input_kan", "2024", "DC"),
    ("kan_efficient", "build_efficient_kan", "example_input_kan", "2024", "DC"),
]
