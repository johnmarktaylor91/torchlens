"""Faithful compact classics for final dependency-heavy menagerie stragglers.

These are Torch-only, random-initialized reimplementations of source-only or
dependency-heavy architectures. They preserve the defining computation paths
needed for TorchLens validation without requiring repo-local packages, CUDA
extensions, OpenMMLab, FairChem, Spyx, or GitHub checkouts.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _coord_noise(
    batch: int, channels: int, height: int, width: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Create deterministic coordinate noise for generative models.

    Parameters
    ----------
    batch:
        Batch size.
    channels:
        Number of channels.
    height:
        Spatial height.
    width:
        Spatial width.
    device:
        Output device.
    dtype:
        Output dtype.

    Returns
    -------
    Tensor
        Deterministic pseudo-noise tensor.
    """

    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    base = []
    for index in range(channels):
        freq = float(index + 1)
        base.append(torch.sin(freq * math.pi * grid_x) + torch.cos(freq * math.pi * grid_y))
    return torch.stack(base, dim=0).unsqueeze(0).expand(batch, -1, -1, -1)


class ConSinGANBlock(nn.Module):
    """ConSinGAN single-scale generator block."""

    def __init__(self, channels: int = 32, layers: int = 5) -> None:
        """Initialize a padded convolutional refinement block.

        Parameters
        ----------
        channels:
            Hidden channel count.
        layers:
            Number of convolutional layers.
        """

        super().__init__()
        modules: list[nn.Module] = [
            nn.Conv2d(3, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=False),
        ]
        for _ in range(max(layers - 2, 1)):
            modules.extend(
                [
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.LeakyReLU(0.2, inplace=False),
                ]
            )
        modules.append(nn.Conv2d(channels, 3, 3, padding=1))
        self.net = nn.Sequential(*modules)

    def forward(self, noise: Tensor, previous: Tensor | None) -> Tensor:
        """Refine one scale of the image pyramid.

        Parameters
        ----------
        noise:
            Current-scale noise image.
        previous:
            Upsampled previous-scale image, if any.

        Returns
        -------
        Tensor
            Current-scale generated image.
        """

        base = noise if previous is None else noise + previous
        return torch.tanh(self.net(base) + base)


class ConSinGANGrowingGenerator(nn.Module):
    """ConSinGAN progressive single-image generator with a scale pyramid."""

    def __init__(self, stages: int = 3, channels: int = 32) -> None:
        """Initialize progressive scale generators.

        Parameters
        ----------
        stages:
            Number of trained scales to unroll.
        channels:
            Hidden channel count per scale.
        """

        super().__init__()
        self.blocks = nn.ModuleList([ConSinGANBlock(channels) for _ in range(stages)])
        self.real_shapes = ((25, 25), (32, 32), (40, 40))
        self.noise_amp = (1.0, 0.1, 0.1)

    def forward(self, x: Tensor) -> Tensor:
        """Generate an image from a deterministic multiscale noise pyramid.

        Parameters
        ----------
        x:
            Sentinel tensor used for batch, device, and dtype.

        Returns
        -------
        Tensor
            Generated RGB image at the final scale.
        """

        batch = int(x.shape[0]) if x.ndim > 0 else 1
        image: Tensor | None = None
        for block, shape, amp in zip(self.blocks, self.real_shapes, self.noise_amp, strict=True):
            height, width = shape
            noise = amp * _coord_noise(batch, 3, height, width, x.device, x.dtype)
            if image is not None:
                image = F.interpolate(image, size=shape, mode="bilinear", align_corners=False)
            image = block(noise, image)
        if image is None:
            raise RuntimeError("ConSinGAN has no generator stages")
        return image


class ConSinGANDiscriminator(nn.Module):
    """ConSinGAN PatchGAN discriminator for one image scale."""

    def __init__(self, channels: int = 32, layers: int = 5) -> None:
        """Initialize the discriminator stack.

        Parameters
        ----------
        channels:
            Hidden channel count.
        layers:
            Number of convolutional layers.
        """

        super().__init__()
        modules: list[nn.Module] = [
            nn.Conv2d(3, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
        ]
        for _ in range(max(layers - 2, 1)):
            modules.extend(
                [
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.LeakyReLU(0.2, inplace=False),
                ]
            )
        modules.append(nn.Conv2d(channels, 1, 3, padding=1))
        self.net = nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        """Score an RGB image with a dense patch discriminator.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        Tensor
            Patch logits.
        """

        return self.net(x)


class MappingNetwork(nn.Module):
    """StyleGAN/ALAE latent mapping network."""

    def __init__(self, latent: int = 256, layers: int = 8) -> None:
        """Initialize equalized-style mapping layers.

        Parameters
        ----------
        latent:
            Latent feature width.
        layers:
            Number of mapping MLP layers.
        """

        super().__init__()
        modules: list[nn.Module] = []
        for _ in range(layers):
            modules.extend([nn.Linear(latent, latent), nn.LeakyReLU(0.2, inplace=False)])
        self.net = nn.Sequential(*modules)

    def forward(self, z: Tensor) -> Tensor:
        """Map normalized latent vectors to disentangled latents.

        Parameters
        ----------
        z:
            Input latent tensor.

        Returns
        -------
        Tensor
            Mapped latent tensor.
        """

        return self.net(F.normalize(z, dim=-1))


class StyleBlock(nn.Module):
    """Style-modulated synthesis block used by ALAE's decoder."""

    def __init__(self, in_channels: int, out_channels: int, latent: int) -> None:
        """Initialize upsampling, convolution, and style affine layers.

        Parameters
        ----------
        in_channels:
            Input feature channels.
        out_channels:
            Output feature channels.
        latent:
            Style latent width.
        """

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.noise = nn.Conv2d(1, out_channels, 1)
        self.style = nn.Linear(latent, out_channels * 2)

    def forward(self, x: Tensor, style: Tensor) -> Tensor:
        """Apply style modulation after spatial upsampling.

        Parameters
        ----------
        x:
            Feature map.
        style:
            Disentangled latent tensor.

        Returns
        -------
        Tensor
            Styled feature map.
        """

        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.conv(x)
        coord = _coord_noise(x.shape[0], 1, x.shape[-2], x.shape[-1], x.device, x.dtype)
        scale, bias = self.style(style).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        x = F.instance_norm(x + self.noise(coord))
        return F.leaky_relu(x * (scale + 1.0) + bias, 0.2)


class ALAEDecoder(nn.Module):
    """ALAE StyleGAN-like generator/decoder."""

    def __init__(self, latent: int = 256, channels: int = 64) -> None:
        """Initialize learned constant and style synthesis blocks.

        Parameters
        ----------
        latent:
            Style latent width.
        channels:
            Base channel count.
        """

        super().__init__()
        self.constant = nn.Parameter(torch.randn(1, channels * 4, 4, 4) * 0.02)
        self.blocks = nn.ModuleList(
            [
                StyleBlock(channels * 4, channels * 2, latent),
                StyleBlock(channels * 2, channels, latent),
                StyleBlock(channels, channels // 2, latent),
                StyleBlock(channels // 2, channels // 4, latent),
            ]
        )
        self.to_rgb = nn.Conv2d(channels // 4, 3, 1)

    def forward(self, style: Tensor) -> Tensor:
        """Synthesize an RGB image from a style latent.

        Parameters
        ----------
        style:
            Disentangled latent tensor.

        Returns
        -------
        Tensor
            Generated RGB image.
        """

        x = self.constant.expand(style.shape[0], -1, -1, -1)
        for block in self.blocks:
            x = block(x, style)
        return torch.tanh(self.to_rgb(x))


class ALAEEncoder(nn.Module):
    """ALAE encoder that maps images into the disentangled latent space."""

    def __init__(self, latent: int = 256, channels: int = 64) -> None:
        """Initialize progressive downsampling encoder.

        Parameters
        ----------
        latent:
            Latent output width.
        channels:
            Base channel count.
        """

        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(3, channels // 4, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(channels // 4, channels // 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(channels // 2, channels, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(channels * 2, channels * 4, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.to_latent = nn.Linear(channels * 4, latent)
        self.discriminator = nn.Linear(latent, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode an image into a latent and adversarial logit.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        tuple[Tensor, Tensor]
            Encoded latent and discriminator logit.
        """

        h = self.pool(self.blocks(x)).flatten(1)
        latent = self.to_latent(h)
        return latent, self.discriminator(latent)


class ALAEFullModel(nn.Module):
    """Adversarial Latent Autoencoder with encoder, mapper, and decoder."""

    def __init__(self) -> None:
        """Initialize ALAE submodules and moving latent-average buffer."""

        super().__init__()
        self.mapping_f = MappingNetwork()
        self.mapping_d = MappingNetwork()
        self.encoder = ALAEEncoder()
        self.decoder = ALAEDecoder()
        self.register_buffer("dlatent_avg", torch.zeros(256))

    def forward(self, x: Tensor) -> Tensor:
        """Encode and reconstruct an image through ALAE's latent space.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        Tensor
            Reconstructed image tensor.
        """

        latent, disc = self.encoder(x)
        style = self.mapping_d(latent) + self.dlatent_avg.unsqueeze(0) * 0.0
        recon = self.decoder(style)
        if recon.shape[-2:] != x.shape[-2:]:
            recon = F.interpolate(recon, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return recon + disc.mean() * 0.0


class ALAEGenerativeModel(nn.Module):
    """Generation-only ALAE wrapper with mapping F and decoder."""

    def __init__(self) -> None:
        """Initialize latent mapper and synthesis decoder."""

        super().__init__()
        self.mapping_f = MappingNetwork()
        self.decoder = ALAEDecoder()

    def forward(self, z: Tensor) -> Tensor:
        """Generate an image from a latent vector.

        Parameters
        ----------
        z:
            Input latent tensor.

        Returns
        -------
        Tensor
            Generated RGB image.
        """

        return self.decoder(self.mapping_f(z))


class DeepFusionBlock(nn.Module):
    """DF-GAN deep fusion block for text-conditioned visual features."""

    def __init__(self, channels: int, cond_dim: int = 256) -> None:
        """Initialize residual visual and conditioning projections.

        Parameters
        ----------
        channels:
            Visual feature channels.
        cond_dim:
            Text conditioning width.
        """

        super().__init__()
        self.gamma = nn.Linear(cond_dim, channels)
        self.beta = nn.Linear(cond_dim, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Fuse text conditioning into one visual stage.

        Parameters
        ----------
        x:
            Visual feature map.
        cond:
            Text embedding.

        Returns
        -------
        Tensor
            Conditioned feature map.
        """

        gamma = self.gamma(cond).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(cond).unsqueeze(-1).unsqueeze(-1)
        h = F.leaky_relu(self.conv1(x) * (1.0 + gamma) + beta, 0.2)
        return F.leaky_relu(x + self.conv2(h), 0.2)


class DFGANGenerator(nn.Module):
    """DF-GAN single-stage text-to-image generator with deep fusion blocks."""

    def __init__(self, noise_dim: int = 100, cond_dim: int = 256, channels: int = 64) -> None:
        """Initialize DF-GAN generator.

        Parameters
        ----------
        noise_dim:
            Noise latent width.
        cond_dim:
            Text conditioning width.
        channels:
            Base visual channels.
        """

        super().__init__()
        self.text_from_noise = nn.Linear(noise_dim, cond_dim)
        self.fc = nn.Linear(noise_dim + cond_dim, channels * 8 * 4 * 4)
        widths = [channels * 8, channels * 4, channels * 2, channels, channels // 2]
        self.to_width = nn.ModuleList(
            [nn.Conv2d(widths[i], widths[i + 1], 3, padding=1) for i in range(4)]
        )
        self.fusion = nn.ModuleList([DeepFusionBlock(width, cond_dim) for width in widths])
        self.to_rgb = nn.Conv2d(widths[-1], 3, 3, padding=1)

    def forward(self, noise: Tensor) -> Tensor:
        """Generate an RGB image from noise and derived text conditioning.

        Parameters
        ----------
        noise:
            Noise latent tensor.

        Returns
        -------
        Tensor
            Generated RGB image.
        """

        cond = torch.tanh(self.text_from_noise(noise))
        x = self.fc(torch.cat([noise, cond], dim=-1)).reshape(noise.shape[0], -1, 4, 4)
        for index, fusion in enumerate(self.fusion):
            x = fusion(x, cond)
            if index < len(self.to_width):
                x = F.interpolate(x, scale_factor=2.0, mode="nearest")
                x = F.leaky_relu(self.to_width[index](x), 0.2)
        return torch.tanh(self.to_rgb(x))


class PointNetLocalPool(nn.Module):
    """PointNet encoder that pools local features into planes or a 3-D grid."""

    def __init__(self, mode: str, resolution: int = 16, channels: int = 32) -> None:
        """Initialize point MLP and local pooling mode.

        Parameters
        ----------
        mode:
            Either ``"3plane"`` or ``"grid"``.
        resolution:
            Feature-grid resolution.
        channels:
            Local feature channel count.
        """

        super().__init__()
        self.mode = mode
        self.resolution = resolution
        self.channels = channels
        self.point_mlp = nn.Sequential(
            nn.Linear(3, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
        )

    def _indices(self, coords: Tensor, dims: tuple[int, ...]) -> Tensor:
        """Convert normalized coordinates to flattened grid indices.

        Parameters
        ----------
        coords:
            Coordinates in ``[-1, 1]``.
        dims:
            Coordinate dimensions to use.

        Returns
        -------
        Tensor
            Flattened integer indices.
        """

        scaled = ((coords[..., list(dims)] + 1.0) * 0.5 * (self.resolution - 1)).long()
        scaled = scaled.clamp(0, self.resolution - 1)
        if len(dims) == 2:
            return scaled[..., 0] * self.resolution + scaled[..., 1]
        return (scaled[..., 0] * self.resolution + scaled[..., 1]) * self.resolution + scaled[
            ..., 2
        ]

    def _scatter(self, feat: Tensor, indices: Tensor, cells: int) -> Tensor:
        """Average point features into grid cells.

        Parameters
        ----------
        feat:
            Point features ``(batch, points, channels)``.
        indices:
            Flattened cell indices ``(batch, points)``.
        cells:
            Number of grid cells.

        Returns
        -------
        Tensor
            Cell features ``(batch, channels, cells)``.
        """

        grids = []
        for batch_index in range(feat.shape[0]):
            grid = feat.new_zeros(cells, feat.shape[-1])
            count = feat.new_zeros(cells, 1)
            grid = grid.index_add(0, indices[batch_index], feat[batch_index])
            ones = torch.ones(indices.shape[1], 1, device=feat.device, dtype=feat.dtype)
            count = count.index_add(0, indices[batch_index], ones)
            grids.append((grid / count.clamp_min(1.0)).transpose(0, 1))
        return torch.stack(grids, dim=0)

    def forward(self, points: Tensor) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        """Encode points into local plane or volume features.

        Parameters
        ----------
        points:
            Point cloud tensor ``(batch, points, 3)``.

        Returns
        -------
        Tensor | tuple[Tensor, Tensor, Tensor]
            Grid features for the selected ConvONet mode.
        """

        coords = points.clamp(-1.0, 1.0)
        feat = self.point_mlp(coords)
        if self.mode == "grid":
            idx = self._indices(coords, (0, 1, 2))
            grid = self._scatter(feat, idx, self.resolution**3)
            return grid.reshape(
                points.shape[0], self.channels, self.resolution, self.resolution, self.resolution
            )
        planes = []
        for dims in ((0, 1), (0, 2), (1, 2)):
            idx = self._indices(coords, dims)
            plane = self._scatter(feat, idx, self.resolution**2)
            planes.append(
                plane.reshape(points.shape[0], self.channels, self.resolution, self.resolution)
            )
        return tuple(planes)


class OccupancyDecoder(nn.Module):
    """Local ConvONet occupancy decoder over plane or grid features."""

    def __init__(self, mode: str, channels: int = 32, hidden: int = 32) -> None:
        """Initialize decoder MLP.

        Parameters
        ----------
        mode:
            Feature sampling mode.
        channels:
            Local feature width.
        hidden:
            Decoder hidden width.
        """

        super().__init__()
        self.mode = mode
        self.mlp = nn.Sequential(
            nn.Linear(channels + 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def _sample_plane(self, plane: Tensor, query: Tensor, dims: tuple[int, int]) -> Tensor:
        """Sample one bilinear feature plane.

        Parameters
        ----------
        plane:
            Plane feature map.
        query:
            Query coordinates.
        dims:
            Coordinate dimensions for this plane.

        Returns
        -------
        Tensor
            Sampled point features.
        """

        grid = query[..., list(dims)].unsqueeze(2)
        sampled = F.grid_sample(plane, grid, align_corners=True, padding_mode="border")
        return sampled.squeeze(-1).transpose(1, 2)

    def forward(self, query: Tensor, features: Tensor | tuple[Tensor, Tensor, Tensor]) -> Tensor:
        """Decode occupancy logits at query points.

        Parameters
        ----------
        query:
            Query point coordinates.
        features:
            Local plane or grid features.

        Returns
        -------
        Tensor
            Occupancy logits.
        """

        if self.mode == "grid":
            if not isinstance(features, Tensor):
                raise TypeError("grid ConvONet expects volumetric features")
            grid = query.view(query.shape[0], query.shape[1], 1, 1, 3)
            sampled = F.grid_sample(features, grid, align_corners=True, padding_mode="border")
            local = sampled.squeeze(-1).squeeze(-1).transpose(1, 2)
        else:
            if isinstance(features, Tensor):
                raise TypeError("3plane ConvONet expects plane features")
            local = (
                self._sample_plane(features[0], query, (0, 1))
                + self._sample_plane(features[1], query, (0, 2))
                + self._sample_plane(features[2], query, (1, 2))
            ) / 3.0
        return self.mlp(torch.cat([query, local], dim=-1)).squeeze(-1)


class ConvolutionalOccupancyNetwork(nn.Module):
    """3D Convolutional Occupancy Network with local feature pooling."""

    def __init__(self, mode: str) -> None:
        """Initialize encoder and decoder.

        Parameters
        ----------
        mode:
            Either ``"3plane"`` or ``"grid"``.
        """

        super().__init__()
        self.encoder = PointNetLocalPool(mode)
        self.decoder = OccupancyDecoder(mode)

    def forward(self, points: Tensor) -> Tensor:
        """Predict occupancies for compact query points from an input cloud.

        Parameters
        ----------
        points:
            Input point cloud.

        Returns
        -------
        Tensor
            Occupancy logits for the first query points.
        """

        features = self.encoder(points)
        query = points[:, :128].clamp(-1.0, 1.0)
        return self.decoder(query, features)


class DarkResidual(nn.Module):
    """YOLOv3 Darknet residual block."""

    def __init__(self, channels: int) -> None:
        """Initialize bottleneck residual convolutions.

        Parameters
        ----------
        channels:
            Feature channel count.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.BatchNorm2d(channels // 2),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(channels // 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply a Darknet residual update.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        Tensor
            Updated feature map.
        """

        return x + self.net(x)


class JDETracker(nn.Module):
    """Joint Detection and Embedding tracker with YOLOv3-style heads."""

    def __init__(self, ids: int = 256, width: int = 24) -> None:
        """Initialize shared detector backbone and ReID heads.

        Parameters
        ----------
        ids:
            Number of identity logits in the compact ReID branch.
        width:
            Base backbone channel count.
        """

        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1),
            nn.BatchNorm2d(width),
            nn.LeakyReLU(0.1, inplace=False),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.LeakyReLU(0.1, inplace=False),
            DarkResidual(width * 2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(width * 4),
            nn.LeakyReLU(0.1, inplace=False),
            DarkResidual(width * 4),
            DarkResidual(width * 4),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(width * 4, width * 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(width * 8),
            nn.LeakyReLU(0.1, inplace=False),
            DarkResidual(width * 8),
        )
        self.det_small = nn.Conv2d(width * 4, 3 * (4 + 1 + 5), 1)
        self.det_large = nn.Conv2d(width * 8, 3 * (4 + 1 + 5), 1)
        self.emb_small = nn.Conv2d(width * 4, 64, 1)
        self.emb_large = nn.Conv2d(width * 8, 64, 1)
        self.id_head = nn.Linear(128, ids)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run joint detection and ReID embedding inference.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Detection heads, normalized embeddings, and identity logits.
        """

        x = self.stem(x)
        c3 = self.stage1(x)
        c4 = self.stage2(c3)
        c5 = self.stage3(c4)
        det = torch.cat(
            [
                self.det_small(c4).flatten(2).transpose(1, 2),
                self.det_large(c5).flatten(2).transpose(1, 2),
            ],
            dim=1,
        )
        emb_map = torch.cat(
            [
                F.adaptive_avg_pool2d(self.emb_small(c4), 1).flatten(1),
                F.adaptive_avg_pool2d(self.emb_large(c5), 1).flatten(1),
            ],
            dim=1,
        )
        emb = F.normalize(emb_map, dim=-1)
        return det, emb, self.id_head(emb)


class EquiformerV2Compact(nn.Module):
    """EquiformerV2-style equivariant molecular graph transformer."""

    def __init__(self, atoms: int = 12, hidden: int = 48) -> None:
        """Initialize atom embeddings, radial basis, and message layers.

        Parameters
        ----------
        atoms:
            Number of atoms in the compact molecular graph.
        hidden:
            Hidden representation width.
        """

        super().__init__()
        self.atoms = atoms
        self.embedding = nn.Embedding(16, hidden)
        self.radial = nn.Sequential(nn.Linear(8, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.scalar_msg = nn.Linear(hidden * 2 + 9, hidden)
        self.vector_gate = nn.Linear(hidden, 3)
        self.attn = nn.MultiheadAttention(hidden, 4, batch_first=True, dropout=0.0)
        self.norm = nn.LayerNorm(hidden)
        self.energy = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.register_buffer("atomic_numbers", torch.arange(atoms, dtype=torch.long) % 8)
        self.register_buffer("positions", self._make_positions(atoms))

    @staticmethod
    def _make_positions(atoms: int) -> Tensor:
        """Create deterministic compact molecule coordinates.

        Parameters
        ----------
        atoms:
            Number of atoms.

        Returns
        -------
        Tensor
            Coordinate tensor.
        """

        t = torch.linspace(0.0, 1.0, atoms)
        return torch.stack(
            [torch.sin(2 * math.pi * t), torch.cos(2 * math.pi * t), t * 2.0 - 1.0], dim=-1
        )

    def forward(self, x: Tensor) -> Tensor:
        """Compute an invariant molecular energy from geometric messages.

        Parameters
        ----------
        x:
            Sentinel tensor used for device and dtype.

        Returns
        -------
        Tensor
            Predicted molecular energy.
        """

        pos = self.positions.to(device=x.device, dtype=x.dtype)
        z = self.atomic_numbers.to(device=x.device)
        h = self.embedding(z)
        vec = pos.unsqueeze(1) - pos.unsqueeze(0)
        dist = vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        direction = vec / dist
        radial_basis = torch.sin(torch.arange(1, 9, device=x.device, dtype=x.dtype) * dist)
        angular = _angular_features(direction.reshape(-1, 3)).reshape(self.atoms, self.atoms, 9)
        pair = torch.cat(
            [
                h.unsqueeze(1).expand(-1, self.atoms, -1),
                h.unsqueeze(0).expand(self.atoms, -1, -1),
                angular,
            ],
            dim=-1,
        )
        msg = self.scalar_msg(pair) * self.radial(radial_basis)
        gate = torch.tanh(self.vector_gate(msg)).unsqueeze(-1)
        equivariant_mix = (gate * direction.unsqueeze(-2)).sum(dim=-1).mean(dim=1)
        h = h + msg.mean(dim=1) + F.pad(equivariant_mix, (0, h.shape[-1] - 3))
        attended = self.attn(h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0), need_weights=False)[
            0
        ].squeeze(0)
        return self.energy(self.norm(h + attended)).sum(dim=0)


def _angular_features(direction: Tensor) -> Tensor:
    """Compute compact spherical-harmonic-like angular features.

    Parameters
    ----------
    direction:
        Unit direction vectors.

    Returns
    -------
    Tensor
        Angular feature vectors.
    """

    x, y, z = direction.unbind(dim=-1)
    return torch.stack(
        [
            torch.ones_like(x),
            x,
            y,
            z,
            x * y,
            y * z,
            z * x,
            x.square() - y.square(),
            3.0 * z.square() - 1.0,
        ],
        dim=-1,
    )


class SpyxLIFNetwork(nn.Module):
    """Spyx-style linear LIF spiking network using Torch operations."""

    def __init__(self) -> None:
        """Initialize feed-forward weights and LIF parameters."""

        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        self.decay = nn.Parameter(torch.tensor(0.9))
        self.threshold = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: Tensor) -> Tensor:
        """Unroll leaky integrate-and-fire dynamics over several steps.

        Parameters
        ----------
        x:
            Flattened image features.

        Returns
        -------
        Tensor
            Class logits accumulated from spikes.
        """

        current = self.fc1(x)
        membrane = torch.zeros_like(current)
        logits = torch.zeros(x.shape[0], 10, device=x.device, dtype=x.dtype)
        beta = self.decay.sigmoid()
        for step in range(6):
            membrane = beta * membrane + current / float(step + 1)
            spike = torch.sigmoid(10.0 * (membrane - self.threshold))
            membrane = membrane * (1.0 - spike.detach())
            logits = logits + self.fc2(spike)
        return logits / 6.0


def build_consingan_discriminator() -> nn.Module:
    """Build a compact ConSinGAN discriminator.

    Returns
    -------
    nn.Module
        Evaluation-mode discriminator.
    """

    return ConSinGANDiscriminator().eval()


def build_consingan_generator() -> nn.Module:
    """Build a compact ConSinGAN growing generator.

    Returns
    -------
    nn.Module
        Evaluation-mode generator.
    """

    return ConSinGANGrowingGenerator().eval()


def build_alae_full_model() -> nn.Module:
    """Build a compact full ALAE model.

    Returns
    -------
    nn.Module
        Evaluation-mode ALAE model.
    """

    return ALAEFullModel().eval()


def build_alae_gen_model() -> nn.Module:
    """Build a compact ALAE generative wrapper.

    Returns
    -------
    nn.Module
        Evaluation-mode generative model.
    """

    return ALAEGenerativeModel().eval()


def build_dfgan_netg() -> nn.Module:
    """Build a compact DF-GAN generator.

    Returns
    -------
    nn.Module
        Evaluation-mode generator.
    """

    return DFGANGenerator().eval()


def build_conv_onet_3plane() -> nn.Module:
    """Build a compact tri-plane Convolutional Occupancy Network.

    Returns
    -------
    nn.Module
        Evaluation-mode ConvONet.
    """

    return ConvolutionalOccupancyNetwork("3plane").eval()


def build_conv_onet_grid() -> nn.Module:
    """Build a compact grid Convolutional Occupancy Network.

    Returns
    -------
    nn.Module
        Evaluation-mode ConvONet.
    """

    return ConvolutionalOccupancyNetwork("grid").eval()


def build_jde() -> nn.Module:
    """Build a compact JDE tracker.

    Returns
    -------
    nn.Module
        Evaluation-mode tracker.
    """

    return JDETracker().eval()


def build_equiformer_v2() -> nn.Module:
    """Build a compact EquiformerV2 molecular graph model.

    Returns
    -------
    nn.Module
        Evaluation-mode molecular model.
    """

    return EquiformerV2Compact().eval()


def build_spyx_lif_snn() -> nn.Module:
    """Build a compact Spyx-style LIF SNN.

    Returns
    -------
    nn.Module
        Evaluation-mode spiking network.
    """

    return SpyxLIFNetwork().eval()
