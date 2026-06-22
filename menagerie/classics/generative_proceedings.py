"""Generative-model proceedings gaps: compact faithful classics.

This module covers notable generative architectures found missing in the broad
catalog sweep:

* Gated PixelCNN: van den Oord et al., NeurIPS 2016, arXiv:1606.05328.
* Wasserstein Auto-Encoder: Tolstikhin et al., ICLR 2018, arXiv:1711.01558.
* VAE with a VampPrior: Tomczak & Welling, AISTATS 2018, arXiv:1705.07120.
* FactorVAE: Kim & Mnih, ICML 2018, PMLR 80.
* VQ-Diffusion: Gu et al., CVPR 2022, arXiv:2111.14822.
* Score-SDE: Song et al., ICLR 2021, arXiv:2011.13456.
* Poisson Flow Generative Model: Xu et al., NeurIPS 2022, arXiv:2209.11178.

All models are random initialized, CPU-friendly, and keep the load-bearing
primitive visible in the trace rather than implementing training losses in full.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedConv2d(nn.Conv2d):
    """Two-dimensional convolution with a fixed autoregressive mask.

    Parameters
    ----------
    mask_type:
        ``"A"`` masks the center pixel, while ``"B"`` keeps it.
    in_channels:
        Number of input channels.
    out_channels:
        Number of output channels.
    kernel_size:
        Spatial kernel size.
    vertical:
        If true, use a vertical-stack mask that only sees rows above the current
        row. Otherwise use the standard raster autoregressive mask.
    """

    def __init__(
        self,
        mask_type: str,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        vertical: bool = False,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        mask = torch.ones_like(self.weight)
        center = kernel_size // 2
        if vertical:
            mask[:, :, center:, :] = 0.0
        else:
            mask[:, :, center + 1 :, :] = 0.0
            mask[:, :, center, center + (mask_type == "B") :] = 0.0
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the masked convolution to an image tensor.

        Parameters
        ----------
        x:
            Image tensor of shape ``(batch, channels, height, width)``.

        Returns
        -------
        torch.Tensor
            Masked convolution output.
        """

        return F.conv2d(x, self.weight * self.mask, self.bias, self.stride, self.padding)


class GatedPixelBlock(nn.Module):
    """Gated PixelCNN vertical/horizontal block with tanh-sigmoid gates."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.v_conv = MaskedConv2d("B", channels, 2 * channels, 3, vertical=True)
        self.v_to_h = nn.Conv2d(2 * channels, 2 * channels, 1)
        self.h_conv = MaskedConv2d("B", channels, 2 * channels, 3)
        self.h_out = nn.Conv2d(channels, channels, 1)

    def _gate(self, x: torch.Tensor) -> torch.Tensor:
        """Split features into tanh and sigmoid gates.

        Parameters
        ----------
        x:
            Tensor with doubled channel dimension.

        Returns
        -------
        torch.Tensor
            Gated activation tensor.
        """

        a, b = x.chunk(2, dim=1)
        return torch.tanh(a) * torch.sigmoid(b)

    def forward(
        self, inputs: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one gated vertical/horizontal PixelCNN block.

        Parameters
        ----------
        inputs:
            Pair ``(vertical, horizontal)`` feature maps.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated vertical and horizontal feature maps.
        """

        vertical, horizontal = inputs
        v_features = self.v_conv(vertical)
        v_out = self._gate(v_features)
        h_features = self.h_conv(horizontal) + self.v_to_h(v_features)
        h_out = horizontal + self.h_out(self._gate(h_features))
        return v_out, h_out


class GatedPixelCNN(nn.Module):
    """Conditional Gated PixelCNN image density model."""

    def __init__(self, channels: int = 24, n_classes: int = 8) -> None:
        super().__init__()
        self.in_conv = MaskedConv2d("A", 1, channels, 7)
        self.label_embed = nn.Embedding(n_classes, channels)
        self.blocks = nn.ModuleList([GatedPixelBlock(channels) for _ in range(3)])
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, 16, 1),
        )

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Predict discrete pixel logits from an image and class label.

        Parameters
        ----------
        inputs:
            Pair ``(image, label)`` where image has shape ``(B, 1, H, W)``.

        Returns
        -------
        torch.Tensor
            Pixel logits with shape ``(B, 16, H, W)``.
        """

        image, label = inputs
        cond = self.label_embed(label).view(label.shape[0], -1, 1, 1)
        features = self.in_conv(image) + cond
        vertical = features
        horizontal = features
        for block in self.blocks:
            vertical, horizontal = block((vertical, horizontal))
        return self.out(horizontal)


class ConvEncoder(nn.Module):
    """Small convolutional Gaussian encoder shared by VAE-family classics."""

    def __init__(self, latent_dim: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(),
        )
        self.mu = nn.Linear(32 * 4 * 4, latent_dim)
        self.logvar = nn.Linear(32 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode an image into diagonal-Gaussian parameters.

        Parameters
        ----------
        x:
            Image tensor of shape ``(B, 1, 16, 16)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Mean and log-variance tensors.
        """

        features = self.net(x)
        return self.mu(features), self.logvar(features)


class ConvDecoder(nn.Module):
    """Small deconvolutional decoder shared by VAE-family classics."""

    def __init__(self, latent_dim: int = 8) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, 32 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors into images.

        Parameters
        ----------
        z:
            Latent tensor of shape ``(B, latent_dim)``.

        Returns
        -------
        torch.Tensor
            Reconstructed image tensor.
        """

        return self.net(self.fc(z).view(z.shape[0], 32, 4, 4))


def deterministic_sample(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Create a deterministic trace-friendly latent sample.

    Parameters
    ----------
    mu:
        Gaussian mean.
    logvar:
        Gaussian log-variance.

    Returns
    -------
    torch.Tensor
        Reparameterized latent using a deterministic pseudo-noise function.
    """

    eps = torch.tanh(mu)
    return mu + torch.exp(0.5 * logvar) * eps


class WassersteinAutoEncoder(nn.Module):
    """WAE with encoder-decoder path and MMD prior-matching penalty."""

    def __init__(self, latent_dim: int = 8) -> None:
        super().__init__()
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim)
        prior = torch.linspace(-1.0, 1.0, steps=4 * latent_dim).view(4, latent_dim)
        self.register_buffer("prior_points", prior)

    def _rbf_mmd(self, z: torch.Tensor) -> torch.Tensor:
        """Compute a compact RBF-MMD statistic against fixed prior points.

        Parameters
        ----------
        z:
            Encoded latent batch.

        Returns
        -------
        torch.Tensor
            Scalar MMD penalty.
        """

        prior = self.prior_points[: z.shape[0]].to(dtype=z.dtype, device=z.device)
        zz = torch.cdist(z, z).pow(2)
        pp = torch.cdist(prior, prior).pow(2)
        zp = torch.cdist(z, prior).pow(2)
        kzz = torch.exp(-0.5 * zz).mean()
        kpp = torch.exp(-0.5 * pp).mean()
        kzp = torch.exp(-0.5 * zp).mean()
        return kzz + kpp - 2.0 * kzp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return reconstruction plus a broadcast WAE-MMD channel.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        torch.Tensor
            Reconstruction with the scalar MMD penalty added for trace visibility.
        """

        mu, _ = self.encoder(x)
        reconstruction = self.decoder(mu)
        mmd = self._rbf_mmd(mu)
        return reconstruction + mmd.view(1, 1, 1, 1)


class VampPriorVAE(nn.Module):
    """VAE whose prior is a mixture of posteriors on learned pseudo-inputs."""

    def __init__(self, latent_dim: int = 8, n_pseudo_inputs: int = 6) -> None:
        super().__init__()
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim)
        self.pseudo_inputs = nn.Parameter(torch.randn(n_pseudo_inputs, 1, 16, 16) * 0.05)

    def _vamp_log_prior(self, z: torch.Tensor) -> torch.Tensor:
        """Approximate the VampPrior log mixture density.

        Parameters
        ----------
        z:
            Latent tensor.

        Returns
        -------
        torch.Tensor
            Per-example log-prior values.
        """

        pseudo_mu, pseudo_logvar = self.encoder(torch.sigmoid(self.pseudo_inputs))
        z_expanded = z[:, None, :]
        var = torch.exp(pseudo_logvar)[None, :, :]
        log_prob = -0.5 * (((z_expanded - pseudo_mu[None, :, :]).pow(2) / var) + pseudo_logvar)
        log_prob = log_prob.sum(dim=-1)
        return torch.logsumexp(log_prob, dim=1) - math.log(float(pseudo_mu.shape[0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode an input through a deterministic VampPrior VAE path.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        torch.Tensor
            Reconstruction modulated by the pseudo-input prior mixture.
        """

        mu, logvar = self.encoder(x)
        z = deterministic_sample(mu, logvar)
        reconstruction = self.decoder(z)
        prior_score = self._vamp_log_prior(z).mean()
        return reconstruction + prior_score.view(1, 1, 1, 1) * 0.01


class FactorVAE(nn.Module):
    """FactorVAE with total-correlation discriminator over latent codes."""

    def __init__(self, latent_dim: int = 8) -> None:
        super().__init__()
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim)
        self.tc_discriminator = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 2),
        )

    def _permute_dims(self, z: torch.Tensor) -> torch.Tensor:
        """Create a trace-stable dimension-wise batch permutation.

        Parameters
        ----------
        z:
            Latent tensor.

        Returns
        -------
        torch.Tensor
            Latent tensor with each dimension rolled by a different offset.
        """

        columns = [torch.roll(z[:, i], shifts=i + 1, dims=0) for i in range(z.shape[1])]
        return torch.stack(columns, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return reconstruction with visible TC discriminator computation.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        torch.Tensor
            Reconstruction modulated by the total-correlation logits.
        """

        mu, logvar = self.encoder(x)
        z = deterministic_sample(mu, logvar)
        reconstruction = self.decoder(z)
        real_logits = self.tc_discriminator(z)
        perm_logits = self.tc_discriminator(self._permute_dims(z))
        tc_signal = (real_logits[:, 0] - perm_logits[:, 0]).mean()
        return reconstruction + tc_signal.view(1, 1, 1, 1) * 0.01


class VQDiffusion(nn.Module):
    """Discrete latent mask-and-replace diffusion transformer."""

    def __init__(self, vocab_size: int = 32, text_vocab: int = 64, d_model: int = 48) -> None:
        super().__init__()
        self.mask_id = vocab_size
        self.code_embed = nn.Embedding(vocab_size + 1, d_model)
        self.text_embed = nn.Embedding(text_vocab, d_model)
        self.time_embed = nn.Linear(1, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=96,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Denoise masked discrete VQ tokens with text and timestep conditioning.

        Parameters
        ----------
        inputs:
            Tuple ``(tokens, text_tokens, timestep)``.

        Returns
        -------
        torch.Tensor
            Replacement logits for VQ-code tokens.
        """

        tokens, text_tokens, timestep = inputs
        mask_pattern = (torch.arange(tokens.shape[1], device=tokens.device) % 3) == 0
        noisy_tokens = torch.where(mask_pattern[None, :], tokens.new_full((), self.mask_id), tokens)
        code = self.code_embed(noisy_tokens)
        text = self.text_embed(text_tokens)
        time = self.time_embed(timestep.reshape(-1, 1).to(code.dtype)).unsqueeze(1)
        hidden = torch.cat([time, text, code], dim=1)
        denoised = self.transformer(hidden)
        return self.out(denoised[:, 1 + text.shape[1] :, :])


class TinyScoreUNet(nn.Module):
    """Small score network used by the Score-SDE sampler."""

    def __init__(self, channels: int = 24) -> None:
        super().__init__()
        self.time = nn.Linear(1, channels)
        self.down = nn.Conv2d(1, channels, 3, padding=1)
        self.mid = nn.Conv2d(channels, channels, 3, padding=1)
        self.up = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the score field for perturbed images.

        Parameters
        ----------
        x:
            Image tensor.
        t:
            Time tensor of shape ``(B,)``.

        Returns
        -------
        torch.Tensor
            Score estimate with the same shape as ``x``.
        """

        emb = self.time(t.reshape(-1, 1)).view(t.shape[0], -1, 1, 1)
        hidden = F.silu(self.down(x) + emb)
        hidden = F.silu(self.mid(hidden))
        return self.up(hidden)


class ScoreSDEModel(nn.Module):
    """Reverse-time SDE sampler with predictor-corrector updates."""

    def __init__(self, steps: int = 4) -> None:
        super().__init__()
        self.score = TinyScoreUNet()
        self.steps = steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Integrate a compact reverse SDE from noise toward data.

        Parameters
        ----------
        x:
            Initial noisy image tensor.

        Returns
        -------
        torch.Tensor
            Denoised sample estimate.
        """

        batch = x.shape[0]
        dt = 1.0 / float(self.steps)
        for idx in range(self.steps):
            t = x.new_full((batch,), 1.0 - idx * dt)
            score = self.score(x, t)
            x = x + (0.5 * score * dt)
            corrector = self.score(x, t)
            x = x + 0.1 * corrector
        return x


class PFGMField(nn.Module):
    """Poisson-flow field network in an augmented data-plus-radius space."""

    def __init__(self, data_dim: int = 8, hidden_dim: int = 48) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim + 1),
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        """Predict a normalized electric field direction.

        Parameters
        ----------
        x:
            Data coordinates.
        z:
            Augmented scalar coordinate.
        radius:
            Current hypersphere radius.

        Returns
        -------
        torch.Tensor
            Unit-normalized field over ``(x, z)``.
        """

        field = self.net(torch.cat([x, z, radius], dim=-1))
        return field / field.norm(dim=-1, keepdim=True).clamp_min(1e-6)


class PoissonFlowGenerativeModel(nn.Module):
    """PFGM backward ODE anchored by the augmented coordinate z."""

    def __init__(self, data_dim: int = 8, steps: int = 5) -> None:
        super().__init__()
        self.field = PFGMField(data_dim)
        self.steps = steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Flow from a hemisphere sample toward the data hyperplane.

        Parameters
        ----------
        x:
            Initial data coordinates on the augmented hemisphere.

        Returns
        -------
        torch.Tensor
            Generated data coordinates after the compact backward ODE.
        """

        z = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
        radius = torch.sqrt(x.pow(2).sum(dim=-1, keepdim=True) + z.pow(2))
        dt = 1.0 / float(self.steps)
        for _ in range(self.steps):
            direction = self.field(x, z, radius)
            x_direction = direction[:, :-1]
            z_direction = direction[:, -1:]
            x = x - dt * x_direction
            z = (z - dt * z_direction.abs()).clamp_min(0.0)
            radius = torch.sqrt(x.pow(2).sum(dim=-1, keepdim=True) + z.pow(2))
        return x


def build_gated_pixelcnn() -> nn.Module:
    """Build a compact conditional Gated PixelCNN."""

    return GatedPixelCNN()


def example_gated_pixelcnn() -> tuple[torch.Tensor, torch.Tensor]:
    """Create a small image and class-label input for Gated PixelCNN."""

    return torch.rand(1, 1, 12, 12), torch.tensor([2], dtype=torch.long)


def build_wae() -> nn.Module:
    """Build a compact Wasserstein Auto-Encoder."""

    return WassersteinAutoEncoder()


def build_vampprior_vae() -> nn.Module:
    """Build a compact VAE with a VampPrior."""

    return VampPriorVAE()


def build_factorvae() -> nn.Module:
    """Build a compact FactorVAE."""

    return FactorVAE()


def example_autoencoder() -> torch.Tensor:
    """Create a small grayscale-image input for VAE-family models."""

    return torch.rand(4, 1, 16, 16)


def build_vq_diffusion() -> nn.Module:
    """Build a compact VQ-Diffusion denoising transformer."""

    return VQDiffusion()


def example_vq_diffusion() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create discrete latent, text-token, and timestep inputs."""

    return (
        torch.randint(0, 32, (1, 16), dtype=torch.long),
        torch.randint(0, 64, (1, 5), dtype=torch.long),
        torch.tensor([0.4]),
    )


def build_score_sde() -> nn.Module:
    """Build a compact Score-SDE predictor-corrector sampler."""

    return ScoreSDEModel()


def example_score_sde() -> torch.Tensor:
    """Create a small noisy image input for Score-SDE."""

    return torch.randn(1, 1, 16, 16)


def build_pfgm() -> nn.Module:
    """Build a compact Poisson Flow Generative Model."""

    return PoissonFlowGenerativeModel()


def example_pfgm() -> torch.Tensor:
    """Create a small hemisphere-coordinate input for PFGM."""

    return F.normalize(torch.randn(1, 8), dim=-1)


MENAGERIE_ENTRIES = [
    (
        "Gated PixelCNN (conditional gated autoregressive image density model)",
        "build_gated_pixelcnn",
        "example_gated_pixelcnn",
        "2016",
        "DC",
    ),
    (
        "Wasserstein Auto-Encoder (WAE-MMD)",
        "build_wae",
        "example_autoencoder",
        "2018",
        "DC",
    ),
    (
        "VAE with VampPrior (variational mixture of posteriors prior)",
        "build_vampprior_vae",
        "example_autoencoder",
        "2018",
        "DC",
    ),
    (
        "FactorVAE (total-correlation discriminator VAE)",
        "build_factorvae",
        "example_autoencoder",
        "2018",
        "DC",
    ),
    (
        "VQ-Diffusion (mask-and-replace discrete latent diffusion transformer)",
        "build_vq_diffusion",
        "example_vq_diffusion",
        "2022",
        "DC",
    ),
    (
        "Score-SDE (predictor-corrector reverse-time SDE sampler)",
        "build_score_sde",
        "example_score_sde",
        "2021",
        "DC",
    ),
    (
        "Poisson Flow Generative Model (PFGM)",
        "build_pfgm",
        "example_pfgm",
        "2022",
        "DC",
    ),
]
