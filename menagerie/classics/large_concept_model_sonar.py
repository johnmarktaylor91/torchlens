"""Large Concept Model SONAR compact reconstruction.

Paper: Meta FAIR, 2024, "Large Concept Models: Language Modeling in a Sentence
Representation Space".

LCM predicts the next sentence-level concept in SONAR embedding space rather
than next tokens.  This compact model keeps SONAR pre/post nets, causal
sentence-concept transformer, and a diffusion-style denoising head over noisy
future concepts.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LargeConceptModelSONAR(nn.Module):
    """Compact LCM operating on SONAR-like sentence embeddings."""

    def __init__(self, sonar_dim: int = 32, dim: int = 64) -> None:
        """Initialize PreNet, decoder transformer, and PostNet."""

        super().__init__()
        self.prenet = nn.Sequential(nn.LayerNorm(sonar_dim), nn.Linear(sonar_dim, dim), nn.GELU())
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(layer, num_layers=2)
        self.time = nn.Linear(1, dim)
        self.postnet = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, sonar_dim))

    def forward(self, clean_concepts: Tensor, noisy_next: Tensor, noise_level: Tensor) -> Tensor:
        """Predict clean next SONAR concepts from context and noisy concepts."""

        context = self.prenet(clean_concepts)
        noisy = self.prenet(noisy_next) + self.time(noise_level[:, None, None])
        tokens = torch.cat([context, noisy], dim=1)
        length = tokens.shape[1]
        mask = torch.full((length, length), float("-inf"), device=tokens.device).triu(1)
        decoded = self.decoder(tokens, mask=mask)
        return self.postnet(decoded[:, -noisy_next.shape[1] :])


def build() -> nn.Module:
    """Build the compact Large Concept Model SONAR."""

    return LargeConceptModelSONAR().eval()


def example_input() -> tuple[Tensor, Tensor, Tensor]:
    """Return context concepts, noisy future concepts, and noise level."""

    return torch.randn(1, 5, 32), torch.randn(1, 2, 32), torch.tensor([0.4])


MENAGERIE_ENTRIES = [
    ("LargeConceptModel-SONAR", "build", "example_input", "2024", "NLP"),
]
