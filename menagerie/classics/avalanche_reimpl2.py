"""Avalanche continual-learning model wrappers as compact classics.

Source: ContinualAI Avalanche model zoo and baseline strategies.

Avalanche often supplies architectural wrappers around standard PyTorch modules:
gated expert banks, feature extractors, LwF/iCaRL CNNs, progressive networks,
prompted ViTs, simple CNNs, VAEs, and PackNet masked models.  These compact
random-init versions preserve the forward graph patterns without training logic.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyCNN(nn.Module):
    """Small CNN feature extractor and classifier."""

    def __init__(self, out_dim: int = 10) -> None:
        """Initialize convolutional classifier."""

        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Linear(32, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify an image batch."""

        return self.head(self.features(x))


class AutoencoderExpert(nn.Module):
    """ExpertGate autoencoder/classifier pair."""

    def __init__(self, classes: int = 10) -> None:
        """Initialize expert encoder, decoder, and classifier."""

        super().__init__()
        self.encoder = TinyCNN(16)
        self.decoder = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 3 * 32 * 32))
        self.classifier = nn.Linear(16, classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return reconstruction error, logits, and feature vector."""

        feat = self.encoder(x)
        recon = self.decoder(feat).view_as(x)
        err = (recon - x).square().mean(dim=(1, 2, 3), keepdim=False)
        return err, self.classifier(feat), feat


class ExpertGate(nn.Module):
    """Gated bank of task autoencoder experts."""

    def __init__(self, n_experts: int = 3) -> None:
        """Initialize experts and gate."""

        super().__init__()
        self.experts = nn.ModuleList([AutoencoderExpert(10) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Select classifiers by autoencoder reconstruction error."""

        errors = []
        logits = []
        for expert in self.experts:
            err, cls, _ = expert(x)
            errors.append(err)
            logits.append(cls)
        gate = torch.softmax(-torch.stack(errors, dim=1), dim=1)
        stacked_logits = torch.stack(logits, dim=1)
        return (gate.unsqueeze(-1) * stacked_logits).sum(dim=1)


class LwFCNN(nn.Module):
    """Learning-without-Forgetting CNN with old/new heads and distillation path."""

    def __init__(self, old_classes: int = 5, new_classes: int = 5) -> None:
        """Initialize feature extractor plus frozen-old and new task heads."""

        super().__init__()
        self.features = TinyCNN(32).features
        self.old_head = nn.Linear(32, old_classes)
        self.new_head = nn.Linear(32, new_classes)
        self.temperature = 2.0
        for param in self.old_head.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return new logits, old logits, and softened old-task distillation targets."""

        feat = self.features(x)
        old_logits = self.old_head(feat)
        new_logits = self.new_head(feat)
        distill = torch.softmax(old_logits / self.temperature, dim=-1)
        return new_logits, old_logits, distill


class IcarlNet(nn.Module):
    """iCaRL feature extractor with nearest-exemplar class-mean classifier."""

    def __init__(self, classes: int = 5, exemplars: int = 3, dim: int = 32) -> None:
        """Initialize feature encoder and exemplar memory."""

        super().__init__()
        self.features = TinyCNN(dim).features
        self.register_buffer(
            "exemplar_features", F.normalize(torch.randn(classes, exemplars, dim), dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify by cosine distance to stored exemplar class means."""

        feat = F.normalize(self.features(x), dim=-1)
        class_means = F.normalize(self.exemplar_features.mean(dim=1), dim=-1)
        return torch.matmul(feat, class_means.t())


class ProgressiveNet(nn.Module):
    """Progressive neural network with lateral connections."""

    def __init__(self) -> None:
        """Initialize two columns and a lateral adapter."""

        super().__init__()
        self.old = TinyCNN(16)
        self.new = TinyCNN(16)
        self.lateral = nn.Linear(16, 16)
        self.head = nn.Linear(16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run frozen-column-style lateral fusion."""

        old = self.old(x)
        new = self.new(x)
        return self.head(new + torch.tanh(self.lateral(old)))


class PromptedViT(nn.Module):
    """Small ViT with learnable prompt tokens."""

    def __init__(self) -> None:
        """Initialize patch embedder, prompts, Transformer, and head."""

        super().__init__()
        self.patch = nn.Conv2d(3, 32, kernel_size=4, stride=4)
        self.prompts = nn.Parameter(torch.randn(4, 32) * 0.02)
        layer = nn.TransformerEncoderLayer(32, 4, dim_feedforward=64, batch_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify images with prepended prompt tokens."""

        tokens = self.patch(x).flatten(2).transpose(1, 2)
        prompts = self.prompts.unsqueeze(0).expand(x.shape[0], -1, -1)
        return self.head(self.blocks(torch.cat((prompts, tokens), dim=1))[:, 0])


class MlpVAE(nn.Module):
    """Avalanche-style MLP variational autoencoder."""

    def __init__(self, latent: int = 16) -> None:
        """Initialize encoder and decoder MLPs."""

        super().__init__()
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 64), nn.ReLU())
        self.mu = nn.Linear(64, latent)
        self.logvar = nn.Linear(64, latent)
        self.dec = nn.Sequential(nn.Linear(latent, 64), nn.ReLU(), nn.Linear(64, 3 * 32 * 32))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode and deterministically decode through the posterior mean."""

        h = self.enc(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        recon = self.dec(mu).view_as(x)
        return recon, mu, logvar


class PackNetWrapper(nn.Module):
    """PackNet-style masked classifier."""

    def __init__(self) -> None:
        """Initialize masked convolution weights."""

        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.register_buffer("mask", torch.ones_like(self.conv.weight))
        self.head = nn.Linear(16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify using a binary weight mask."""

        y = F.conv2d(x, self.conv.weight * self.mask, self.conv.bias, padding=1)
        y = F.relu(y).mean(dim=(2, 3))
        return self.head(y)


def build_expert_gate() -> nn.Module:
    """Build Avalanche ExpertGate."""

    return ExpertGate()


def build_icarl_feature_extractor() -> nn.Module:
    """Build Avalanche iCaRL feature extractor."""

    return IcarlNet()


def build_lwf_cnn() -> nn.Module:
    """Build Avalanche LwF CNN."""

    return LwFCNN()


def build_mtsimplecnn() -> nn.Module:
    """Build Avalanche multi-task simple CNN."""

    return TinyCNN(20)


def build_pnn() -> nn.Module:
    """Build Avalanche progressive neural network."""

    return ProgressiveNet()


def build_vit_prompt() -> nn.Module:
    """Build Avalanche prompted ViT."""

    return PromptedViT()


def build_icarl_net() -> nn.Module:
    """Build Avalanche IcarlNet-like classifier."""

    return IcarlNet()


def build_simplecnn() -> nn.Module:
    """Build Avalanche SimpleCNN."""

    return TinyCNN(10)


def build_mlpvae() -> nn.Module:
    """Build Avalanche MlpVAE."""

    return MlpVAE()


def build_packnet() -> nn.Module:
    """Build Avalanche PackNetModel-like wrapper."""

    return PackNetWrapper()


def example_image() -> torch.Tensor:
    """Return a small image batch."""

    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    ("Avalanche ExpertGate", "build_expert_gate", "example_image", "2021", "continual"),
    (
        "Avalanche-iCaRL-FeatureExtractor",
        "build_icarl_feature_extractor",
        "example_image",
        "2017",
        "continual",
    ),
    ("Avalanche-LwF-CNN", "build_lwf_cnn", "example_image", "2016", "continual"),
    ("Avalanche MTSimpleCNN", "build_mtsimplecnn", "example_image", "2021", "continual"),
    ("Avalanche PNN", "build_pnn", "example_image", "2016", "continual"),
    ("Avalanche ViTWithPrompt", "build_vit_prompt", "example_image", "2022", "continual"),
    ("Avalanche IcarlNet", "build_icarl_net", "example_image", "2017", "continual"),
    ("Avalanche SimpleCNN", "build_simplecnn", "example_image", "2021", "continual"),
    ("Avalanche MlpVAE", "build_mlpvae", "example_image", "2021", "continual"),
    ("Avalanche PackNetModel", "build_packnet", "example_image", "2018", "continual"),
]
