"""CompressAI-style learned image compression networks.

Paper: Balle et al. 2018, "Variational Image Compression with a Scale Hyperprior";
Minnen et al. 2018, "Joint Autoregressive and Hierarchical Priors"; Cheng et al.
2020, "Learned Image Compression with Discretized Gaussian Mixture Likelihoods and
Attention Modules".

This is a Torch-only random-init reimplementation of the architectural forward
paths used by the missing CompressAI catalog rows. It preserves the analysis
transform, synthesis transform, generalized divisive normalization, hyperprior,
autoregressive context, and Cheng residual-attention blocks at compact channel
counts suitable for base-environment TorchLens validation. Entropy coding and
probability mass table updates are intentionally omitted because they are not
part of the neural forward graph traced here.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GDN(nn.Module):
    """Generalized divisive normalization layer used by learned codecs."""

    def __init__(self, channels: int, inverse: bool = False) -> None:
        """Initialize per-channel normalization parameters.

        Parameters
        ----------
        channels:
            Number of feature channels.
        inverse:
            Whether to apply the inverse GDN transform used in decoders.
        """
        super().__init__()
        self.inverse = inverse
        self.beta = nn.Parameter(torch.ones(channels))
        self.gamma = nn.Parameter(torch.eye(channels) * 0.08)

    def forward(self, x: Tensor) -> Tensor:
        """Apply GDN or inverse GDN.

        Parameters
        ----------
        x:
            Feature map tensor.

        Returns
        -------
        Tensor
            Normalized feature map.
        """
        gamma = F.softplus(self.gamma).view(self.gamma.shape[0], self.gamma.shape[1], 1, 1)
        beta = F.softplus(self.beta).view(1, -1, 1, 1) + 1e-6
        norm = torch.sqrt(F.conv2d(x.pow(2), gamma, beta.flatten()))
        return x * norm if self.inverse else x / norm


class ResidualBlock(nn.Module):
    """Residual convolution block used in Cheng-style transforms."""

    def __init__(self, channels: int) -> None:
        """Initialize residual convolutions.

        Parameters
        ----------
        channels:
            Number of feature channels.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply residual convolution.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        Tensor
            Residual block output.
        """
        return x + self.net(x)


class AttentionBlock(nn.Module):
    """Lightweight non-local channel attention block."""

    def __init__(self, channels: int) -> None:
        """Initialize attention projections.

        Parameters
        ----------
        channels:
            Number of feature channels.
        """
        super().__init__()
        hidden = max(4, channels // 4)
        self.query = nn.Conv2d(channels, hidden, 1)
        self.key = nn.Conv2d(channels, hidden, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        """Apply spatial self-attention.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        Tensor
            Attention-refined feature map.
        """
        batch, channels, height, width = x.shape
        query = self.query(x).flatten(2).transpose(1, 2)
        key = self.key(x).flatten(2)
        value = self.value(x).flatten(2).transpose(1, 2)
        attn = torch.softmax(query.bmm(key) / (key.shape[1] ** 0.5), dim=-1)
        refined = attn.bmm(value).transpose(1, 2).view(batch, channels, height, width)
        return x + self.gate * refined


class AnalysisTransform(nn.Module):
    """Image-to-latent analysis transform."""

    def __init__(self, latent_channels: int, use_attention: bool, activation: str) -> None:
        """Initialize downsampling analysis layers.

        Parameters
        ----------
        latent_channels:
            Number of latent channels.
        use_attention:
            Whether to include Cheng-style attention.
        activation:
            Nonlinearity family, either ``"gdn"`` or ``"relu"``.
        """
        super().__init__()
        hidden = max(16, latent_channels // 2)
        act1: nn.Module = GDN(hidden) if activation == "gdn" else nn.ReLU(inplace=False)
        act2: nn.Module = GDN(hidden) if activation == "gdn" else nn.ReLU(inplace=False)
        blocks: list[nn.Module] = [
            nn.Conv2d(3, hidden, 5, stride=2, padding=2),
            act1,
            nn.Conv2d(hidden, hidden, 5, stride=2, padding=2),
            act2,
            ResidualBlock(hidden),
        ]
        if use_attention:
            blocks.append(AttentionBlock(hidden))
        blocks.extend(
            [
                nn.Conv2d(hidden, latent_channels, 5, stride=2, padding=2),
                ResidualBlock(latent_channels),
            ]
        )
        if use_attention:
            blocks.append(AttentionBlock(latent_channels))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        """Encode an image into latents.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        Tensor
            Latent tensor.
        """
        return self.net(x)


class SynthesisTransform(nn.Module):
    """Latent-to-image synthesis transform."""

    def __init__(self, latent_channels: int, use_attention: bool, activation: str) -> None:
        """Initialize upsampling synthesis layers.

        Parameters
        ----------
        latent_channels:
            Number of latent channels.
        use_attention:
            Whether to include Cheng-style attention.
        activation:
            Nonlinearity family, either ``"gdn"`` or ``"relu"``.
        """
        super().__init__()
        hidden = max(16, latent_channels // 2)
        act1: nn.Module = (
            GDN(hidden, inverse=True) if activation == "gdn" else nn.ReLU(inplace=False)
        )
        act2: nn.Module = (
            GDN(hidden, inverse=True) if activation == "gdn" else nn.ReLU(inplace=False)
        )
        blocks: list[nn.Module] = [ResidualBlock(latent_channels)]
        if use_attention:
            blocks.append(AttentionBlock(latent_channels))
        blocks.extend(
            [
                nn.ConvTranspose2d(
                    latent_channels, hidden, 5, stride=2, padding=2, output_padding=1
                ),
                act1,
                ResidualBlock(hidden),
            ]
        )
        if use_attention:
            blocks.append(AttentionBlock(hidden))
        blocks.extend(
            [
                nn.ConvTranspose2d(hidden, hidden, 5, stride=2, padding=2, output_padding=1),
                act2,
                nn.ConvTranspose2d(hidden, 3, 5, stride=2, padding=2, output_padding=1),
            ]
        )
        self.net = nn.Sequential(*blocks)

    def forward(self, y_hat: Tensor) -> Tensor:
        """Decode quantized latents into an image.

        Parameters
        ----------
        y_hat:
            Quantized latent tensor.

        Returns
        -------
        Tensor
            Reconstructed image tensor.
        """
        return torch.sigmoid(self.net(y_hat))


class Hyperprior(nn.Module):
    """Scale/mean hyperprior for latent entropy parameters."""

    def __init__(self, latent_channels: int, mean: bool) -> None:
        """Initialize hyper-analysis and hyper-synthesis networks.

        Parameters
        ----------
        latent_channels:
            Main latent channel count.
        mean:
            Whether to predict both means and scales.
        """
        super().__init__()
        hyper_channels = max(12, latent_channels // 3)
        out_channels = latent_channels * (2 if mean else 1)
        self.mean = mean
        self.h_a = nn.Sequential(
            nn.Conv2d(latent_channels, hyper_channels, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(hyper_channels, hyper_channels, 5, stride=2, padding=2),
            nn.ReLU(inplace=False),
            nn.Conv2d(hyper_channels, hyper_channels, 5, stride=2, padding=2),
        )
        self.h_s = nn.Sequential(
            nn.ConvTranspose2d(
                hyper_channels, hyper_channels, 5, stride=2, padding=2, output_padding=1
            ),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(
                hyper_channels, hyper_channels, 5, stride=2, padding=2, output_padding=1
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(hyper_channels, out_channels, 3, padding=1),
        )

    def forward(self, y: Tensor) -> tuple[Tensor, Tensor]:
        """Predict latent Gaussian parameters from hyperlatents.

        Parameters
        ----------
        y:
            Main latent tensor.

        Returns
        -------
        tuple[Tensor, Tensor]
            Mean and positive scale tensors.
        """
        z = self.h_a(torch.abs(y))
        z_hat = torch.round(z) + (z - z.detach())
        params = F.interpolate(self.h_s(z_hat), size=y.shape[-2:], mode="nearest")
        if self.mean:
            mean, scale = params.chunk(2, dim=1)
            return mean, F.softplus(scale) + 1e-4
        return torch.zeros_like(y), F.softplus(params) + 1e-4


class CompressAIReimplementation(nn.Module):
    """Compact learned image-compression architecture."""

    def __init__(
        self,
        latent_channels: int = 32,
        hyperprior: bool = False,
        mean_scale: bool = False,
        autoregressive: bool = False,
        attention: bool = False,
        activation: str = "gdn",
    ) -> None:
        """Initialize codec modules.

        Parameters
        ----------
        latent_channels:
            Number of main latent channels.
        hyperprior:
            Whether to include a hyperprior path.
        mean_scale:
            Whether the hyperprior predicts means as well as scales.
        autoregressive:
            Whether to include masked-context entropy parameters.
        attention:
            Whether to include Cheng attention blocks.
        activation:
            Analysis/synthesis nonlinearity family.
        """
        super().__init__()
        self.hyperprior_enabled = hyperprior
        self.autoregressive = autoregressive
        self.analysis = AnalysisTransform(latent_channels, attention, activation)
        self.synthesis = SynthesisTransform(latent_channels, attention, activation)
        self.hyperprior = Hyperprior(latent_channels, mean_scale) if hyperprior else None
        self.context_prediction = nn.Conv2d(latent_channels, latent_channels * 2, 5, padding=2)
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(latent_channels * 4, latent_channels * 2, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(latent_channels * 2, latent_channels * 2, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run analysis, quantization proxy, entropy-parameter path, and synthesis.

        Parameters
        ----------
        x:
            Image tensor with shape ``(batch, 3, height, width)``.

        Returns
        -------
        Tensor
            Reconstructed image tensor.
        """
        y = self.analysis(x)
        y_hat = torch.round(y) + (y - y.detach())
        if self.hyperprior is not None:
            mean, scale = self.hyperprior(y)
            if self.autoregressive:
                context = self.context_prediction(y_hat)
                hyper = torch.cat((mean, scale), dim=1)
                params = self.entropy_parameters(torch.cat((context, hyper), dim=1))
                mean, scale = params.chunk(2, dim=1)
                y_hat = y_hat - torch.tanh(mean) * torch.sigmoid(scale)
            else:
                y_hat = y_hat - torch.tanh(mean) * torch.sigmoid(scale)
        return self.synthesis(y_hat)


def build_factorized_prior_128() -> nn.Module:
    """Build Balle 2018 factorized-prior model with compact ``N=128`` scaling.

    Returns
    -------
    nn.Module
        Random-init codec.
    """
    return CompressAIReimplementation(latent_channels=24, hyperprior=False)


def build_factorized_prior_192() -> nn.Module:
    """Build Balle 2018 factorized-prior model with compact ``N=192`` scaling.

    Returns
    -------
    nn.Module
        Random-init codec.
    """
    return CompressAIReimplementation(latent_channels=32, hyperprior=False)


def build_factorized_prior_relu() -> nn.Module:
    """Build the ReLU analysis-transform variant of the factorized prior.

    Returns
    -------
    nn.Module
        Random-init codec.
    """
    return CompressAIReimplementation(latent_channels=24, hyperprior=False, activation="relu")


def build_scale_hyperprior_128() -> nn.Module:
    """Build Balle 2018 scale-hyperprior model with compact ``N=128`` scaling.

    Returns
    -------
    nn.Module
        Random-init codec.
    """
    return CompressAIReimplementation(latent_channels=24, hyperprior=True)


def build_scale_hyperprior_192() -> nn.Module:
    """Build Balle 2018 scale-hyperprior model with compact ``N=192`` scaling.

    Returns
    -------
    nn.Module
        Random-init codec.
    """
    return CompressAIReimplementation(latent_channels=32, hyperprior=True)


def build_mean_scale_hyperprior_192() -> nn.Module:
    """Build mean-scale hyperprior codec.

    Returns
    -------
    nn.Module
        Random-init codec.
    """
    return CompressAIReimplementation(latent_channels=32, hyperprior=True, mean_scale=True)


def build_joint_autoregressive_hyperprior_192() -> nn.Module:
    """Build Minnen joint autoregressive plus hierarchical-prior codec.

    Returns
    -------
    nn.Module
        Random-init codec.
    """
    return CompressAIReimplementation(
        latent_channels=32,
        hyperprior=True,
        mean_scale=True,
        autoregressive=True,
    )


def build_cheng2020_anchor_192() -> nn.Module:
    """Build Cheng 2020 residual codec without attention.

    Returns
    -------
    nn.Module
        Random-init codec.
    """
    return CompressAIReimplementation(
        latent_channels=32,
        hyperprior=True,
        mean_scale=True,
        autoregressive=True,
    )


def build_cheng2020_attention_192() -> nn.Module:
    """Build Cheng 2020 residual-attention codec.

    Returns
    -------
    nn.Module
        Random-init codec.
    """
    return CompressAIReimplementation(
        latent_channels=32,
        hyperprior=True,
        mean_scale=True,
        autoregressive=True,
        attention=True,
    )


def example_input() -> Tensor:
    """Return a compact natural-image tensor for codec tracing.

    Returns
    -------
    Tensor
        Image tensor.
    """
    return torch.randn(1, 3, 64, 64)
