"""Erik Linder-Noren's PyTorch-GAN reference generators (and a couple discriminators).

Source: https://github.com/eriklindernoren/PyTorch-GAN
A widely-used educational collection of compact, faithful PyTorch ports of classic
GAN papers. Each generator below reproduces the *distinctive* architecture of its
paper at small scale (random init). Covered here:

  * CycleGAN      (Zhu et al. 2017, arXiv:1703.10593) -- ResNet image-translation generator
                  (c7s1 down, 2x downsample, N residual blocks, 2x upsample, c7s1 out).
  * DualGAN       (Yi et al. 2017, arXiv:1704.02510)  -- same ResNet translation generator.
  * DiscoGAN      (Kim et al. 2017, arXiv:1703.05192) -- U-Net (encoder/decoder w/ skips) generator.
  * BicycleGAN    (Zhu et al. 2017, arXiv:1711.11586) -- U-Net generator with latent z injected
                  per-layer (multimodal cVAE-GAN). Here: z tiled & concatenated to the input.
  * CoGAN         (Liu & Tuzel 2016, arXiv:1606.07536) -- Coupled GANs: two generators sharing
                  the early (low-level) layers, branching to two domains.
  * ccGAN         (Denton et al. 2016, arXiv:1611.06430) -- context-conditional inpainting U-Net
                  generator (masked-image in, completed image out).
  * ClusterGAN    (Mukherjee et al. 2019, arXiv:1809.03627) -- generator from continuous z PLUS a
                  one-hot categorical cluster code (concatenated latent -> deconv stack).
  * StarGAN       (Choi et al. 2018, arXiv:1711.09020) -- residual generator conditioned on a
                  target domain label spatially-replicated and concatenated to the image.
  * SRGAN         (Ledig et al. 2017, arXiv:1609.04802) -- SRResNet generator: B residual blocks
                  (no BN -> here with BN per Erik's port) + 2x PixelShuffle upsampling.
  * ESRGAN        (Wang et al. 2018, arXiv:1809.00219) -- RRDB (Residual-in-Residual Dense Block)
                  generator with residual scaling (beta=0.2), no BN.

All builders are zero-arg, return random-init `.eval()`-able modules, and use SMALL
image/latent sizes so the unrolled trace+draw finishes quickly. Generators that take
multiple real inputs (label, latent) are wrapped in a single-tensor adapter that
synthesizes the auxiliary input internally.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CycleGAN / DualGAN ResNet translation generator
# ============================================================


class _ResidualBlock(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    """CycleGAN / DualGAN ResNet generator (c7s1 + 2 down + N res + 2 up + c7s1)."""

    def __init__(
        self, input_shape: tuple[int, int, int] = (3, 64, 64), num_residual_blocks: int = 6
    ) -> None:
        super().__init__()
        channels = input_shape[0]
        out_features = 64
        model: list[nn.Module] = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features
        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [_ResidualBlock(out_features)]
        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ============================================================
# DiscoGAN U-Net generator
# ============================================================


class _UNetDown(nn.Module):
    def __init__(self, in_size: int, out_size: int, normalize: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_size, out_size, 4, 2, 1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class _UNetUp(nn.Module):
    def __init__(self, in_size: int, out_size: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip_input: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return torch.cat((x, skip_input), 1)


class GeneratorUNet(nn.Module):
    """DiscoGAN U-Net generator: 4 down (encoder) + 4 up (decoder) with skip connections."""

    def __init__(self, input_shape: tuple[int, int, int] = (3, 64, 64)) -> None:
        super().__init__()
        channels = input_shape[0]
        self.down1 = _UNetDown(channels, 64, normalize=False)
        self.down2 = _UNetDown(64, 128)
        self.down3 = _UNetDown(128, 256)
        self.down4 = _UNetDown(256, 512)
        self.up1 = _UNetUp(512, 256)
        self.up2 = _UNetUp(512, 128)
        self.up3 = _UNetUp(256, 64)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        return self.final(u3)


# ============================================================
# CoGAN coupled generators (shared early layers, two domain heads)
# ============================================================


class CoupledGenerators(nn.Module):
    """CoGAN: two generators sharing the low-level (shared_conv) layers, branching to 2 domains."""

    def __init__(self, latent_dim: int = 100, img_size: int = 32, channels: int = 3) -> None:
        super().__init__()
        self.init_size = img_size // 4
        self.fc = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size**2))
        self.shared_conv = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
        )
        self.G1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        self.G2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.fc(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img_emb = self.shared_conv(out)
        return self.G1(img_emb), self.G2(img_emb)


# ============================================================
# BicycleGAN U-Net generator with per-input latent injection
# ============================================================


class BicycleGenerator(nn.Module):
    """BicycleGAN U-Net generator: latent z tiled to a spatial map and concatenated to the image."""

    def __init__(self, latent_dim: int = 8, img_shape: tuple[int, int, int] = (3, 64, 64)) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        channels, _, _ = img_shape
        self.fc = nn.Linear(latent_dim, img_shape[1] * img_shape[2])
        self.down1 = _UNetDown(channels + 1, 64, normalize=False)
        self.down2 = _UNetDown(64, 128)
        self.down3 = _UNetDown(128, 256)
        self.down4 = _UNetDown(256, 512)
        self.up1 = _UNetUp(512, 256)
        self.up2 = _UNetUp(512, 128)
        self.up3 = _UNetUp(256, 64)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        z = self.fc(z).view(x.shape[0], 1, x.shape[2], x.shape[3])
        d1 = self.down1(torch.cat((x, z), 1))
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        return self.final(u3)


# ============================================================
# ccGAN context-conditional inpainting U-Net generator
# ============================================================


class CCGANGenerator(nn.Module):
    """ccGAN context-conditional generator: masked image in -> completed image out (U-Net)."""

    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        self.down1 = _UNetDown(channels, 64, normalize=False)
        self.down2 = _UNetDown(64, 128)
        self.down3 = _UNetDown(128, 256)
        self.down4 = _UNetDown(256, 512)
        self.up1 = _UNetUp(512, 256)
        self.up2 = _UNetUp(512, 128)
        self.up3 = _UNetUp(256, 64)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, channels, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        return self.final(u3)


# ============================================================
# ClusterGAN CNN generator (continuous z + one-hot cluster code)
# ============================================================


class ClusterGenerator(nn.Module):
    """ClusterGAN generator: concat continuous z and one-hot categorical code -> deconv stack."""

    def __init__(
        self, latent_dim: int = 30, n_c: int = 10, x_shape: tuple[int, int, int] = (1, 28, 28)
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_c = n_c
        self.ishape = (128, 7, 7)
        self.iels = int(torch.prod(torch.tensor(self.ishape)).item())
        self.model = nn.Sequential(
            nn.Linear(latent_dim + n_c, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, self.iels),
            nn.BatchNorm1d(self.iels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, x_shape[0], 4, stride=2, padding=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, zn: torch.Tensor, zc: torch.Tensor) -> torch.Tensor:
        z = torch.cat((zn, zc), 1)
        x = self.model(z)
        x = x.view(x.shape[0], *self.ishape)
        return self.deconv(x)


# ============================================================
# StarGAN residual generator conditioned on a domain label
# ============================================================


class _StarResidualBlock(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(in_features, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(in_features, affine=True, track_running_stats=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class StarGenerator(nn.Module):
    """StarGAN generator: spatially-replicated domain label concatenated to the image."""

    def __init__(
        self, img_shape: tuple[int, int, int] = (3, 64, 64), res_blocks: int = 6, c_dim: int = 5
    ) -> None:
        super().__init__()
        channels = img_shape[0]
        self.c_dim = c_dim
        model: list[nn.Module] = [
            nn.Conv2d(channels + c_dim, 64, 7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        ]
        curr_dim = 64
        for _ in range(2):
            model += [
                nn.Conv2d(curr_dim, curr_dim * 2, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ]
            curr_dim *= 2
        for _ in range(res_blocks):
            model += [_StarResidualBlock(curr_dim)]
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ]
            curr_dim //= 2
        model += [nn.Conv2d(curr_dim, channels, 7, stride=1, padding=3), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat((x, c), 1)
        return self.model(x)


# ============================================================
# SRGAN SRResNet generator (residual blocks + PixelShuffle upsampling)
# ============================================================


class _SRResidualBlock(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            nn.BatchNorm2d(in_features, 0.8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class SRResNetGenerator(nn.Module):
    """SRGAN generator (SRResNet): conv7 + N residual blocks + skip + 2x PixelShuffle upsampling."""

    def __init__(
        self, in_channels: int = 3, out_channels: int = 3, n_residual_blocks: int = 8
    ) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, 9, 1, 4), nn.PReLU())
        self.res_blocks = nn.Sequential(*[_SRResidualBlock(64) for _ in range(n_residual_blocks)])
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64, 0.8))
        upsampling: list[nn.Module] = []
        for _ in range(2):
            upsampling += [
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)
        self.conv3 = nn.Sequential(nn.Conv2d(64, out_channels, 9, 1, 4), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        return self.conv3(out)


# ============================================================
# ESRGAN RRDB generator (Residual-in-Residual Dense Blocks, no BN)
# ============================================================


class _DenseResidualBlock(nn.Module):
    def __init__(self, filters: int, res_scale: float = 0.2) -> None:
        super().__init__()
        self.res_scale = res_scale

        def block(in_features: int, non_linearity: bool = True) -> nn.Sequential:
            layers: list[nn.Module] = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x


class _ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters: int, res_scale: float = 0.2) -> None:
        super().__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            _DenseResidualBlock(filters),
            _DenseResidualBlock(filters),
            _DenseResidualBlock(filters),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense_blocks(x).mul(self.res_scale) + x


class GeneratorRRDB(nn.Module):
    """ESRGAN RRDB generator: N Residual-in-Residual Dense Blocks (no BN) + PixelShuffle upsampling."""

    def __init__(
        self, channels: int = 3, filters: int = 64, num_res_blocks: int = 4, num_upsample: int = 2
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, filters, 3, 1, 1)
        self.res_blocks = nn.Sequential(
            *[_ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)]
        )
        self.conv2 = nn.Conv2d(filters, filters, 3, 1, 1)
        upsample_layers: list[nn.Module] = []
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Conv2d(filters, filters * 4, 3, 1, 1),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor=2),
            ]
        self.upsampling = nn.Sequential(*upsample_layers)
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        return self.conv3(out)


# ============================================================
# Single-tensor adapter wrappers (synthesize aux inputs internally)
# ============================================================


class _CoGANWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        a, _b = self.model(z)
        return a


class _BicycleWrapper(nn.Module):
    def __init__(self, model: BicycleGenerator) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.zeros(x.shape[0], self.model.latent_dim)
        return self.model(x, z)


class _ClusterWrapper(nn.Module):
    def __init__(self, model: ClusterGenerator) -> None:
        super().__init__()
        self.model = model

    def forward(self, zn: torch.Tensor) -> torch.Tensor:
        zc = F.one_hot(torch.zeros(zn.shape[0], dtype=torch.long), self.model.n_c).float()
        return self.model(zn, zc)


class _StarWrapper(nn.Module):
    def __init__(self, model: StarGenerator) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = torch.zeros(x.shape[0], self.model.c_dim)
        c[:, 0] = 1.0
        return self.model(x, c)


# ============================================================
# Menagerie wiring
# ============================================================


def build_cyclegan() -> nn.Module:
    """CycleGAN ResNet image-translation generator (6 residual blocks)."""
    return GeneratorResNet(input_shape=(3, 64, 64), num_residual_blocks=6)


def build_dualgan() -> nn.Module:
    """DualGAN ResNet image-translation generator (same core as CycleGAN)."""
    return GeneratorResNet(input_shape=(3, 64, 64), num_residual_blocks=6)


def build_discogan() -> nn.Module:
    """DiscoGAN U-Net image-translation generator."""
    return GeneratorUNet(input_shape=(3, 64, 64))


def build_cogan() -> nn.Module:
    """CoGAN coupled generators (shared early layers, two domain heads)."""
    return _CoGANWrapper(CoupledGenerators(latent_dim=100, img_size=32, channels=3))


def build_bicyclegan() -> nn.Module:
    """BicycleGAN U-Net generator with injected latent z."""
    return _BicycleWrapper(BicycleGenerator(latent_dim=8, img_shape=(3, 64, 64)))


def build_ccgan() -> nn.Module:
    """ccGAN context-conditional inpainting U-Net generator."""
    return CCGANGenerator(channels=3)


def build_clustergan() -> nn.Module:
    """ClusterGAN generator (continuous z + one-hot cluster code)."""
    return _ClusterWrapper(ClusterGenerator(latent_dim=30, n_c=10, x_shape=(1, 28, 28)))


def build_stargan() -> nn.Module:
    """StarGAN residual generator conditioned on a domain label."""
    return _StarWrapper(StarGenerator(img_shape=(3, 64, 64), res_blocks=6, c_dim=5))


def build_srgan() -> nn.Module:
    """SRGAN SRResNet generator (residual blocks + PixelShuffle 4x upsample)."""
    return SRResNetGenerator(in_channels=3, out_channels=3, n_residual_blocks=8)


def build_esrgan() -> nn.Module:
    """ESRGAN RRDB generator (Residual-in-Residual Dense Blocks, no BN)."""
    return GeneratorRRDB(channels=3, filters=64, num_res_blocks=4, num_upsample=2)


def example_image_64() -> torch.Tensor:
    """RGB image tensor (1, 3, 64, 64)."""
    return torch.randn(1, 3, 64, 64)


def example_image_64_inpaint() -> torch.Tensor:
    """RGB masked-image tensor (1, 3, 64, 64) for ccGAN inpainting."""
    return torch.randn(1, 3, 64, 64)


def example_latent_100() -> torch.Tensor:
    """Noise latent (1, 100) for CoGAN."""
    return torch.randn(1, 100)


def example_latent_30() -> torch.Tensor:
    """Continuous latent (1, 30) for ClusterGAN."""
    return torch.randn(1, 30)


def example_lr_64() -> torch.Tensor:
    """Low-resolution image tensor (1, 3, 64, 64) for SR generators."""
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "CycleGAN (ResNet image-translation generator)",
        "build_cyclegan",
        "example_image_64",
        "2017",
        "DC",
    ),
    (
        "DualGAN (ResNet image-translation generator)",
        "build_dualgan",
        "example_image_64",
        "2017",
        "DC",
    ),
    (
        "DiscoGAN (U-Net image-translation generator)",
        "build_discogan",
        "example_image_64",
        "2017",
        "DC",
    ),
    ("CoGAN (coupled weight-shared generators)", "build_cogan", "example_latent_100", "2016", "DC"),
    (
        "BicycleGAN (U-Net generator with latent injection)",
        "build_bicyclegan",
        "example_image_64",
        "2017",
        "DC",
    ),
    (
        "ccGAN (context-conditional inpainting U-Net generator)",
        "build_ccgan",
        "example_image_64_inpaint",
        "2016",
        "DC",
    ),
    (
        "ClusterGAN (z + one-hot cluster-code generator)",
        "build_clustergan",
        "example_latent_30",
        "2019",
        "DC",
    ),
    (
        "StarGAN (domain-label-conditioned residual generator)",
        "build_stargan",
        "example_image_64",
        "2018",
        "DC",
    ),
    (
        "SRGAN (SRResNet generator, PixelShuffle upsampling)",
        "build_srgan",
        "example_lr_64",
        "2017",
        "DC",
    ),
    (
        "ESRGAN (RRDB generator, residual-in-residual dense blocks)",
        "build_esrgan",
        "example_lr_64",
        "2018",
        "DC",
    ),
]
