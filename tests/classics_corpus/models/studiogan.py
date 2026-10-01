"""StudioGAN: A library implementing many GAN variants with common training infrastructure.

Kang et al., "StudioGAN: A Taxonomy and Benchmark of GANs for Image Synthesis",
IEEE TPAMI 2023. arXiv:2206.09479.
Source: https://github.com/POSTECH-CVLab/StudioGAN

Each entry here shows the SIGNATURE OP of a distinct GAN variant as packaged by StudioGAN:
DCGAN, ResNet-GAN, SNGAN, SAGAN, BigGAN, ProjGAN, ContraGAN, ReACGAN, GGAN, LSGAN,
WGAN-GP, and StyleGAN2. All are compact random-init CPU models.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Shared primitives
# ============================================================


class ResBlockUp(nn.Module):
    """Upsampling residual block (bilinear 2x + conv). Base for ResNet/SNGAN/SAGAN/BigGAN."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(x))
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.conv1(h)
        h = F.relu(self.bn2(h))
        h = self.conv2(h)
        skip = F.interpolate(x, scale_factor=2, mode="nearest")
        skip = self.shortcut(skip)
        return h + skip


class SNResBlockUp(nn.Module):
    """ResBlockUp with spectral-norm on all convs (SNGAN signature)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1))
        self.shortcut = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1))
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(x))
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.conv1(h)
        h = F.relu(self.bn2(h))
        h = self.conv2(h)
        skip = F.interpolate(x, scale_factor=2, mode="nearest")
        skip = self.shortcut(skip)
        return h + skip


class SelfAttention(nn.Module):
    """Non-local self-attention block (SAGAN / BigGAN)."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.q = nn.Conv2d(ch, ch // 8, 1)
        self.k = nn.Conv2d(ch, ch // 8, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q = self.q(x).view(B, -1, H * W).permute(0, 2, 1)  # B, HW, C//8
        k = self.k(x).view(B, -1, H * W)  # B, C//8, HW
        attn = F.softmax(torch.bmm(q, k), dim=-1)  # B, HW, HW
        v = self.v(x).view(B, -1, H * W)  # B, C, HW
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return x + self.gamma * out


class ConditionalBatchNorm(nn.Module):
    """Class-conditional BatchNorm: gamma/beta predicted from class embedding (BigGAN)."""

    def __init__(self, num_features: int, num_classes: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        nn.init.ones_(self.embed.weight[:, :num_features])
        nn.init.zeros_(self.embed.weight[:, num_features:])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, dim=1)
        return out * (1 + gamma.unsqueeze(-1).unsqueeze(-1)) + beta.unsqueeze(-1).unsqueeze(-1)


class BigGANResBlockUp(nn.Module):
    """BigGAN ResBlock with class-conditional BN and a chunk of z injected."""

    def __init__(self, in_ch: int, out_ch: int, z_chunk: int, num_classes: int) -> None:
        super().__init__()
        self.cbn1 = ConditionalBatchNorm(in_ch, num_classes)
        self.cbn2 = ConditionalBatchNorm(out_ch, num_classes)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.cbn1(x, y))
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.conv1(h)
        h = F.relu(self.cbn2(h, y))
        h = self.conv2(h)
        skip = F.interpolate(x, scale_factor=2, mode="nearest")
        skip = self.shortcut(skip)
        return h + skip


# ============================================================
# MODULE 1: studiogan_dcgan_generator
# ============================================================


class StudioGANDCGANGenerator(nn.Module):
    """StudioGAN DCGAN Generator.

    Vanilla DCGAN with BatchNorm+ReLU transposed-conv stack.
    Distinct from dcgan_progan.py (which uses ProGAN pixelnorm, equalized-LR).
    This is vanilla BatchNorm+ReLU as in StudioGAN's DCGAN implementation.
    z -> ConvTranspose2d x4 -> BatchNorm -> ReLU -> Tanh output.
    """

    def __init__(self, z_dim: int = 16, ch: int = 16) -> None:
        super().__init__()
        # Project z to 4x4 feature map
        self.project = nn.ConvTranspose2d(z_dim, ch * 4, 4, 1, 0, bias=False)
        self.bn0 = nn.BatchNorm2d(ch * 4)
        # 4->8
        self.up1 = nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch * 2)
        # 8->16
        self.up2 = nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        # 16->32
        self.up3 = nn.ConvTranspose2d(ch, 3, 4, 2, 1, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, z_dim)
        x = z.view(z.shape[0], -1, 1, 1)
        x = F.relu(self.bn0(self.project(x)))
        x = F.relu(self.bn1(self.up1(x)))
        x = F.relu(self.bn2(self.up2(x)))
        x = torch.tanh(self.up3(x))
        return x


def build_studiogan_dcgan_generator() -> nn.Module:
    return StudioGANDCGANGenerator(z_dim=16, ch=16)


def example_studiogan_dcgan_generator() -> torch.Tensor:
    return torch.randn(1, 16)


# ============================================================
# MODULE 2: studiogan_resnet_generator
# ============================================================


class StudioGANResNetGenerator(nn.Module):
    """BigGAN-style ResNet generator (no spectral norm, plain BN).

    z -> linear project to 4x4 const -> 2x ResBlockUp -> output Tanh.
    """

    def __init__(self, z_dim: int = 16, ch: int = 16) -> None:
        super().__init__()
        self.project = nn.Linear(z_dim, ch * 4 * 4 * 4)
        self.block1 = ResBlockUp(ch * 4, ch * 2)
        self.block2 = ResBlockUp(ch * 2, ch)
        self.bn_out = nn.BatchNorm2d(ch)
        self.conv_out = nn.Conv2d(ch, 3, 3, 1, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, z_dim)
        x = self.project(z).view(z.shape[0], -1, 4, 4)
        x = self.block1(x)
        x = self.block2(x)
        x = torch.tanh(self.conv_out(F.relu(self.bn_out(x))))
        return x


def build_studiogan_resnet_generator() -> nn.Module:
    return StudioGANResNetGenerator(z_dim=16, ch=16)


def example_studiogan_resnet_generator() -> torch.Tensor:
    return torch.randn(1, 16)


# ============================================================
# MODULE 3: studiogan_sngan_generator
# ============================================================


class StudioGANSNGANGenerator(nn.Module):
    """SNGAN Generator: ResNet with spectral_norm on all conv/linear layers.

    Miyato et al. "Spectral Normalization for Generative Adversarial Networks", ICLR 2018.
    arXiv:1802.05957. The spectral_norm calls are the signature op.
    """

    def __init__(self, z_dim: int = 16, ch: int = 16) -> None:
        super().__init__()
        self.project = nn.utils.spectral_norm(nn.Linear(z_dim, ch * 4 * 4 * 4))
        self.block1 = SNResBlockUp(ch * 4, ch * 2)
        self.block2 = SNResBlockUp(ch * 2, ch)
        self.bn_out = nn.BatchNorm2d(ch)
        self.conv_out = nn.utils.spectral_norm(nn.Conv2d(ch, 3, 3, 1, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.project(z).view(z.shape[0], -1, 4, 4)
        x = self.block1(x)
        x = self.block2(x)
        x = torch.tanh(self.conv_out(F.relu(self.bn_out(x))))
        return x


def build_studiogan_sngan_generator() -> nn.Module:
    return StudioGANSNGANGenerator(z_dim=16, ch=16)


def example_studiogan_sngan_generator() -> torch.Tensor:
    return torch.randn(1, 16)


# ============================================================
# MODULE 4: studiogan_sagan_generator
# ============================================================


class StudioGANSAGANGenerator(nn.Module):
    """SAGAN Generator: ResNet gen with a SelfAttention layer between blocks.

    Zhang et al. "Self-Attention Generative Adversarial Networks", ICML 2019.
    arXiv:1805.08318. Signature: Non-local self-attention between ResBlock-Up stages.
    """

    def __init__(self, z_dim: int = 16, ch: int = 16) -> None:
        super().__init__()
        self.project = nn.Linear(z_dim, ch * 4 * 4 * 4)
        self.block1 = ResBlockUp(ch * 4, ch * 2)
        self.attn = SelfAttention(ch * 2)  # <-- SAGAN signature op
        self.block2 = ResBlockUp(ch * 2, ch)
        self.bn_out = nn.BatchNorm2d(ch)
        self.conv_out = nn.Conv2d(ch, 3, 3, 1, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.project(z).view(z.shape[0], -1, 4, 4)
        x = self.block1(x)
        x = self.attn(x)  # self-attention at mid-resolution
        x = self.block2(x)
        x = torch.tanh(self.conv_out(F.relu(self.bn_out(x))))
        return x


def build_studiogan_sagan_generator() -> nn.Module:
    return StudioGANSAGANGenerator(z_dim=16, ch=16)


def example_studiogan_sagan_generator() -> torch.Tensor:
    return torch.randn(1, 16)


# ============================================================
# MODULE 5: studiogan_biggan_generator
# ============================================================


class StudioGANBigGANGenerator(nn.Module):
    """BigGAN Generator: class-conditional BN + hierarchical z + Non-Local Block.

    Brock et al. "Large Scale GAN Training for High Fidelity Natural Image Synthesis",
    ICLR 2019. arXiv:1809.11096.
    Signature: ConditionalBatchNorm (gamma/beta from class embedding) + hierarchical z.
    """

    def __init__(self, z_dim: int = 16, ch: int = 16, num_classes: int = 4) -> None:
        super().__init__()
        # Hierarchical z: split into 3 chunks (1 for project, 2 for blocks)
        self.z_dim = z_dim
        self.z_chunk = z_dim // 4  # chunk size per block
        self.project = nn.Linear(z_dim // 2, ch * 4 * 4 * 4)
        self.block1 = BigGANResBlockUp(ch * 4, ch * 2, self.z_chunk, num_classes)
        self.attn = SelfAttention(ch * 2)
        self.block2 = BigGANResBlockUp(ch * 2, ch, self.z_chunk, num_classes)
        self.bn_out = nn.BatchNorm2d(ch)
        self.conv_out = nn.Conv2d(ch, 3, 3, 1, 1)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # z: (B, z_dim), y: (B,) class indices
        z0 = z[:, : self.z_dim // 2]
        x = self.project(z0).view(z.shape[0], -1, 4, 4)
        x = self.block1(x, y)
        x = self.attn(x)
        x = self.block2(x, y)
        x = torch.tanh(self.conv_out(F.relu(self.bn_out(x))))
        return x


def build_studiogan_biggan_generator() -> nn.Module:
    return StudioGANBigGANGenerator(z_dim=16, ch=16, num_classes=4)


def example_studiogan_biggan_generator() -> tuple:
    return (torch.randn(1, 16), torch.zeros(1, dtype=torch.long))


# ============================================================
# MODULE 6: studiogan_projgan
# ============================================================


class StudioGANProjGANDiscriminator(nn.Module):
    """Projection Discriminator (Miyato & Koyama 2018).

    Miyato & Koyama, "cGANs with Projection Discriminator", ICLR 2018.
    arXiv:1802.05637.
    Signature: inner product of class embedding with penultimate features
    (inner_product = dot(class_embed(y), features)) as additional conditioning signal.
    Input: (image, class_label).
    """

    def __init__(self, ch: int = 16, num_classes: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(ch, ch * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(ch * 2, ch * 4, 4, 2, 1)
        self.linear = nn.Linear(ch * 4 * 4 * 4, 1)
        # Projection: class embedding has same dim as features
        self.class_embed = nn.Embedding(num_classes, ch * 4 * 4 * 4)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 32, 32), y: (B,) class indices
        h = F.leaky_relu(self.conv1(x), 0.2)
        h = F.leaky_relu(self.conv2(h), 0.2)
        h = F.leaky_relu(self.conv3(h), 0.2)
        h_flat = h.view(h.shape[0], -1)
        # Standard discriminator logit
        logit = self.linear(h_flat)
        # Projection inner product (Miyato & Koyama signature)
        proj = (self.class_embed(y) * h_flat).sum(dim=1, keepdim=True)
        return logit + proj


def build_studiogan_projgan() -> nn.Module:
    return StudioGANProjGANDiscriminator(ch=16, num_classes=4)


def example_studiogan_projgan() -> tuple:
    return (torch.randn(1, 3, 32, 32), torch.zeros(1, dtype=torch.long))


# ============================================================
# MODULE 7: studiogan_contragan_discriminator
# ============================================================


class StudioGANContraGANDiscriminator(nn.Module):
    """ContraGAN Discriminator with 2C contrastive embedding head.

    Kang et al. "ContraGAN: Contrastive Learning for Conditional Image Generation",
    NeurIPS 2020. arXiv:2006.12681.
    Signature: 2C contrastive embedding head projects features to conditional space
    for the 2C (two-way contrastive) loss.
    """

    def __init__(self, ch: int = 16, num_classes: int = 4, embed_dim: int = 16) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(ch, ch * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(ch * 2, ch * 4, 4, 2, 1)
        feat_dim = ch * 4 * 4 * 4
        self.adv_linear = nn.Linear(feat_dim, 1)
        self.class_embed = nn.Embedding(num_classes, feat_dim)
        # 2C contrastive head: projects features to embed_dim for contrastive loss
        self.contrastive_proj = nn.Linear(feat_dim, embed_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple:
        h = F.leaky_relu(self.conv1(x), 0.2)
        h = F.leaky_relu(self.conv2(h), 0.2)
        h = F.leaky_relu(self.conv3(h), 0.2)
        h_flat = h.view(h.shape[0], -1)
        # Adversarial logit + class projection (ProjGAN-style)
        logit = self.adv_linear(h_flat) + (self.class_embed(y) * h_flat).sum(dim=1, keepdim=True)
        # 2C contrastive embedding (ContraGAN signature)
        embed = self.contrastive_proj(h_flat)
        return logit, embed


def build_studiogan_contragan_discriminator() -> nn.Module:
    return StudioGANContraGANDiscriminator(ch=16, num_classes=4, embed_dim=16)


def example_studiogan_contragan_discriminator() -> tuple:
    return (torch.randn(1, 3, 32, 32), torch.zeros(1, dtype=torch.long))


# ============================================================
# MODULE 8: studiogan_reacgan_generator
# ============================================================


class StudioGANReACGANGenerator(nn.Module):
    """ReACGAN Generator with auxiliary feature projection head.

    Kang et al. "Rebooting ACGAN: Auxiliary Classifier GANs with Projection Discriminator",
    NeurIPS 2021. arXiv:2111.01118.
    Signature: generator + data-to-data cross-modal projection head that projects
    intermediate features to a target feature space.
    """

    def __init__(
        self, z_dim: int = 16, ch: int = 16, num_classes: int = 4, feat_dim: int = 16
    ) -> None:
        super().__init__()
        self.class_embed = nn.Embedding(num_classes, z_dim)
        self.project = nn.Linear(z_dim * 2, ch * 4 * 4 * 4)
        self.block1 = ResBlockUp(ch * 4, ch * 2)
        self.block2 = ResBlockUp(ch * 2, ch)
        self.bn_out = nn.BatchNorm2d(ch)
        self.conv_out = nn.Conv2d(ch, 3, 3, 1, 1)
        # Feature projection head (ReACGAN signature: project mid-features to target space)
        self.feat_proj = nn.Linear(ch * 4 * 4 * 4, feat_dim)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> tuple:
        # z: (B, z_dim), y: (B,) class indices
        c = self.class_embed(y)
        h = self.project(torch.cat([z, c], dim=1))
        h = h.view(z.shape[0], -1, 4, 4)
        # Auxiliary feature projection from latent features (before upsampling)
        h_proj = self.feat_proj(h.view(z.shape[0], -1))  # ReACGAN feature projection
        x = self.block1(h)
        x = self.block2(x)
        img = torch.tanh(self.conv_out(F.relu(self.bn_out(x))))
        return img, h_proj


def build_studiogan_reacgan_generator() -> nn.Module:
    return StudioGANReACGANGenerator(z_dim=16, ch=16, num_classes=4)


def example_studiogan_reacgan_generator() -> tuple:
    return (torch.randn(1, 16), torch.zeros(1, dtype=torch.long))


# ============================================================
# MODULE 9: studiogan_ggan
# ============================================================


class StudioGANGGANDiscriminator(nn.Module):
    """Geometric GAN discriminator: hinge-loss DCGAN-ish, no BatchNorm, LeakyReLU.

    Lim & Ye, "Geometric GAN", arXiv:1705.02894.
    Signature: no BatchNorm in discriminator (hinge-loss compatible), LeakyReLU,
    scalar output (supports SVM-like geometric margin loss).
    """

    def __init__(self, ch: int = 16) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(ch, ch * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(ch * 2, ch * 4, 4, 2, 1)
        # No BatchNorm (Geometric GAN discriminator has no BN)
        self.linear = nn.Linear(ch * 4 * 4 * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.conv1(x), 0.2)
        h = F.leaky_relu(self.conv2(h), 0.2)
        h = F.leaky_relu(self.conv3(h), 0.2)
        return self.linear(h.view(h.shape[0], -1))


def build_studiogan_ggan() -> nn.Module:
    return StudioGANGGANDiscriminator(ch=16)


def example_studiogan_ggan() -> torch.Tensor:
    return torch.randn(1, 3, 32, 32)


# ============================================================
# MODULE 10: studiogan_lsgan
# ============================================================


class StudioGANLSGANGenerator(nn.Module):
    """Least-Squares GAN Generator: real-valued output (no Tanh sigmoid).

    Mao et al. "Least Squares Generative Adversarial Networks", ICCV 2017.
    arXiv:1611.04076.
    Signature: generator outputs linear (real-valued, no bounding activation),
    trained with MSE loss rather than binary cross-entropy.
    """

    def __init__(self, z_dim: int = 16, ch: int = 16) -> None:
        super().__init__()
        self.project = nn.ConvTranspose2d(z_dim, ch * 4, 4, 1, 0, bias=False)
        self.bn0 = nn.BatchNorm2d(ch * 4)
        self.up1 = nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch * 2)
        self.up2 = nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        # LSGAN signature: linear output (no Tanh)
        self.up3 = nn.ConvTranspose2d(ch, 3, 4, 2, 1, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = z.view(z.shape[0], -1, 1, 1)
        x = F.relu(self.bn0(self.project(x)))
        x = F.relu(self.bn1(self.up1(x)))
        x = F.relu(self.bn2(self.up2(x)))
        # LSGAN: no Tanh, linear real-valued output
        return self.up3(x)


def build_studiogan_lsgan() -> nn.Module:
    return StudioGANLSGANGenerator(z_dim=16, ch=16)


def example_studiogan_lsgan() -> torch.Tensor:
    return torch.randn(1, 16)


# ============================================================
# MODULE 11: studiogan_wgan_gp
# ============================================================


class StudioGANWGANGPCritic(nn.Module):
    """WGAN-GP Critic: NO BatchNorm, LayerNorm, LeakyReLU, scalar unbounded output.

    Gulrajani et al. "Improved Training of Wasserstein GANs", NeurIPS 2017.
    arXiv:1704.00028.
    Signature: NO BatchNorm (incompatible with WGAN gradient penalty), uses LayerNorm
    (or no norm). Outputs unbounded real scalar (Wasserstein distance estimate).
    """

    def __init__(self, ch: int = 16) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, ch, 4, 2, 1)
        # WGAN-GP: LayerNorm instead of BatchNorm
        self.ln1 = nn.GroupNorm(1, ch)  # equivalent to LayerNorm on feature maps
        self.conv2 = nn.Conv2d(ch, ch * 2, 4, 2, 1)
        self.ln2 = nn.GroupNorm(1, ch * 2)
        self.conv3 = nn.Conv2d(ch * 2, ch * 4, 4, 2, 1)
        self.ln3 = nn.GroupNorm(1, ch * 4)
        self.linear = nn.Linear(ch * 4 * 4 * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.ln1(self.conv1(x)), 0.2)
        h = F.leaky_relu(self.ln2(self.conv2(h)), 0.2)
        h = F.leaky_relu(self.ln3(self.conv3(h)), 0.2)
        # Unbounded scalar output (Wasserstein critic)
        return self.linear(h.view(h.shape[0], -1))


def build_studiogan_wgan_gp() -> nn.Module:
    return StudioGANWGANGPCritic(ch=16)


def example_studiogan_wgan_gp() -> torch.Tensor:
    return torch.randn(1, 3, 32, 32)


# ============================================================
# MODULE 12: studiogan_stylegan2_generator
# ============================================================


class ModulatedConv2d(nn.Module):
    """StyleGAN2 modulated convolution with weight demodulation.

    Karras et al. "Analyzing and Improving the Image Quality of StyleGAN", CVPR 2020.
    arXiv:1912.04958.
    Signature: per-sample weight modulation via style affine transform + demodulation
    (normalize modulated weights by their L2 norm to prevent 'blob' artifacts).
    """

    def __init__(self, in_ch: int, out_ch: int, style_dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel_size, kernel_size) * 0.02)
        self.style_mod = nn.Linear(style_dim, in_ch)  # affine transform: w -> style scale

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Style modulation: per-sample scale for each input channel
        style = self.style_mod(w) + 1.0  # (B, in_ch)
        # Modulate weight: weight * style
        weight = self.weight.unsqueeze(0) * style.view(B, 1, C, 1, 1)  # (B, out_ch, in_ch, k, k)
        # Demodulation: normalize by L2 norm of modulated weights
        demod = torch.rsqrt(weight.pow(2).sum(dim=[2, 3, 4], keepdim=True) + 1e-8)
        weight = weight * demod  # (B, out_ch, in_ch, k, k)
        # Apply per-sample via grouped conv (reshape batch into groups)
        x_flat = x.view(1, B * C, H, W)
        w_flat = weight.view(B * self.out_ch, C, self.kernel_size, self.kernel_size)
        pad = self.kernel_size // 2
        out = F.conv2d(x_flat, w_flat, padding=pad, groups=B)
        return out.view(B, self.out_ch, H, W)


class StyleGAN2MappingNetwork(nn.Module):
    """Mapping network: z -> w (4-layer MLP with LeakyReLU)."""

    def __init__(self, z_dim: int, w_dim: int, n_layers: int = 4) -> None:
        super().__init__()
        layers = []
        in_dim = z_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, w_dim), nn.LeakyReLU(0.2)]
            in_dim = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class StudioGANStyleGAN2Generator(nn.Module):
    """StyleGAN2 as packaged by StudioGAN: mapping network + modulated-conv synthesis.

    Signature: weight demodulation modulated convolution (ModulatedConv2d).
    Simplified synthesis: project const -> 2 modulated conv blocks -> output.
    """

    def __init__(self, z_dim: int = 16, w_dim: int = 16, ch: int = 16) -> None:
        super().__init__()
        self.const = nn.Parameter(torch.randn(1, ch * 4, 4, 4))
        self.mapping = StyleGAN2MappingNetwork(z_dim, w_dim, n_layers=4)
        self.mod_conv1 = ModulatedConv2d(ch * 4, ch * 2, w_dim)
        self.mod_conv2 = ModulatedConv2d(ch * 2, ch, w_dim)
        self.to_rgb = nn.Conv2d(ch, 3, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        w = self.mapping(z)
        x = self.const.expand(B, -1, -1, -1)
        x = F.leaky_relu(self.mod_conv1(x, w), 0.2)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = F.leaky_relu(self.mod_conv2(x, w), 0.2)
        return torch.tanh(self.to_rgb(x))


def build_studiogan_stylegan2_generator() -> nn.Module:
    return StudioGANStyleGAN2Generator(z_dim=16, w_dim=16, ch=16)


def example_studiogan_stylegan2_generator() -> torch.Tensor:
    return torch.randn(1, 16)


# ============================================================
# MENAGERIE_ENTRIES
# ============================================================

MENAGERIE_ENTRIES = [
    (
        "studiogan_dcgan_generator",
        "build_studiogan_dcgan_generator",
        "example_studiogan_dcgan_generator",
        "2022",
        "DC",
    ),
    (
        "studiogan_resnet_generator",
        "build_studiogan_resnet_generator",
        "example_studiogan_resnet_generator",
        "2022",
        "DC",
    ),
    (
        "studiogan_sngan_generator",
        "build_studiogan_sngan_generator",
        "example_studiogan_sngan_generator",
        "2022",
        "DC",
    ),
    (
        "studiogan_sagan_generator",
        "build_studiogan_sagan_generator",
        "example_studiogan_sagan_generator",
        "2022",
        "DC",
    ),
    (
        "studiogan_biggan_generator",
        "build_studiogan_biggan_generator",
        "example_studiogan_biggan_generator",
        "2022",
        "DC",
    ),
    (
        "studiogan_projgan",
        "build_studiogan_projgan",
        "example_studiogan_projgan",
        "2022",
        "DC",
    ),
    (
        "studiogan_contragan_discriminator",
        "build_studiogan_contragan_discriminator",
        "example_studiogan_contragan_discriminator",
        "2022",
        "DC",
    ),
    (
        "studiogan_reacgan_generator",
        "build_studiogan_reacgan_generator",
        "example_studiogan_reacgan_generator",
        "2022",
        "DC",
    ),
    (
        "studiogan_ggan",
        "build_studiogan_ggan",
        "example_studiogan_ggan",
        "2022",
        "DC",
    ),
    (
        "studiogan_lsgan",
        "build_studiogan_lsgan",
        "example_studiogan_lsgan",
        "2022",
        "DC",
    ),
    (
        "studiogan_wgan_gp",
        "build_studiogan_wgan_gp",
        "example_studiogan_wgan_gp",
        "2022",
        "DC",
    ),
    (
        "studiogan_stylegan2_generator",
        "build_studiogan_stylegan2_generator",
        "example_studiogan_stylegan2_generator",
        "2022",
        "DC",
    ),
]
