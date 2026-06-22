"""Compact dependency-gated robot, audio, depth, and materials models.

This module covers several install-hostile model packages with faithful compact
PyTorch reconstructions:

* OpenPI π0: VLM-conditioned flow-matching action head.
* OpenVoice: spectral encoder, speaker-conditioned normalizing-flow coupling,
  and transposed-convolution decoder.
* PackNet: symmetric packing/unpacking monocular depth encoder-decoder with 3D
  packing bias.
* Orb-v2/v3: atomistic graph neural potential with radial message passing and
  charge/spin conditioning for the newer v3-style interface.

Sources: π0 paper/OpenPI repository, OpenVoice paper, PackNet CVPR 2020, and
Orb/Orb-v3 release notes and paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Pi0Tiny(nn.Module):
    """Tiny π0-style VLA flow model with action expert tokens."""

    def __init__(self, action_dim: int = 7) -> None:
        super().__init__()
        self.vision = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU())
        self.text = nn.Embedding(64, 16)
        self.state = nn.Linear(8, 16)
        self.time = nn.Linear(1, 16)
        self.fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(16, 4, dim_feedforward=32, batch_first=True), num_layers=1
        )
        self.action_in = nn.Linear(action_dim, 16)
        self.action_expert = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(16, 4, dim_feedforward=32, batch_first=True), num_layers=1
        )
        self.action_head = nn.Linear(16, action_dim)

    def forward(
        self,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Predict a flow-matching action velocity.

        Parameters
        ----------
        inputs:
            Tuple of image batch, token IDs, robot state, noisy action tokens,
            and scalar diffusion/flow time. A legacy three-tuple omits state and
            action tokens and creates zeros.

        Returns
        -------
        torch.Tensor
            Action velocity vector.
        """

        if len(inputs) == 5:
            image, tokens, robot_state, noisy_actions, flow_time = inputs
        else:
            image, tokens, flow_time = inputs
            robot_state = torch.zeros(image.shape[0], 8, device=image.device)
            noisy_actions = torch.zeros(
                image.shape[0], 4, self.action_head.out_features, device=image.device
            )
        vis = F.adaptive_avg_pool2d(self.vision(image), 1).flatten(1).unsqueeze(1)
        txt = self.text(tokens).mean(dim=1, keepdim=True)
        state = self.state(robot_state).unsqueeze(1)
        time = self.time(flow_time).unsqueeze(1)
        context = self.fusion(torch.cat((vis, txt, state, time), dim=1)).mean(dim=1, keepdim=True)
        action_tokens = self.action_in(noisy_actions) + context
        expert_tokens = self.action_expert(action_tokens)
        return self.action_head(expert_tokens)


class OpenVoiceTiny(nn.Module):
    """Tiny OpenVoice tone-color converter."""

    def __init__(self, channels: int = 24) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(80, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, 3, padding=1),
        )
        self.speaker = nn.Linear(16, channels)
        self.coupling = nn.Conv1d(channels, channels * 2, 1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, 1, 3, padding=1),
        )

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Convert tone color from a mel spectrum and speaker vector.

        Parameters
        ----------
        inputs:
            Tuple of mel spectrum ``(batch, 80, time)`` and speaker embedding.

        Returns
        -------
        torch.Tensor
            Waveform-like output.
        """

        mel, speaker = inputs
        z = self.encoder(mel)
        shift = self.speaker(speaker).unsqueeze(-1)
        log_scale, bias = self.coupling(z + shift).chunk(2, dim=1)
        z = z * torch.exp(torch.tanh(log_scale)) + bias
        return self.decoder(z)


class PackNetTiny(nn.Module):
    """Tiny PackNet monocular depth network with packing/unpacking blocks."""

    def __init__(self) -> None:
        super().__init__()
        self.pack = nn.Conv3d(3, 12, (2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1))
        self.enc = nn.Conv2d(12, 24, 3, stride=2, padding=1)
        self.sparse = nn.Sequential(
            nn.Conv2d(2, 12, 3, padding=1), nn.ReLU(), nn.Conv2d(12, 12, 3, padding=1)
        )
        self.mid = nn.Conv2d(24, 24, 3, padding=1)
        self.dec = nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1)
        self.depth = nn.Conv2d(24, 1, 3, padding=1)

    def forward(self, inputs: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Predict inverse depth.

        Parameters
        ----------
        inputs:
            RGB image batch or ``(image, sparse_depth)`` tuple.

        Returns
        -------
        torch.Tensor
            Positive inverse-depth map.
        """

        if isinstance(inputs, tuple):
            image, sparse_depth = inputs
        else:
            image = inputs
            sparse_depth = torch.zeros(
                image.shape[0], 1, image.shape[-2], image.shape[-1], device=image.device
            )
        valid = (sparse_depth > 0).to(image.dtype)
        sparse_feat = self.sparse(torch.cat((sparse_depth, valid), dim=1))
        packed = image.unsqueeze(2).repeat(1, 1, 2, 1, 1)
        packed = F.relu(self.pack(packed)).squeeze(2)
        packed = packed + sparse_feat
        enc = F.relu(self.enc(packed))
        mid = F.relu(self.mid(enc))
        up = F.relu(self.dec(mid))
        fused = torch.cat((packed, up), dim=1)
        return F.softplus(self.depth(fused))


class OrbTiny(nn.Module):
    """Tiny Orb-style atomistic graph neural potential.

    Parameters
    ----------
    use_charge_spin:
        Whether to condition on total charge and spin multiplicity.
    """

    def __init__(self, use_charge_spin: bool = False) -> None:
        super().__init__()
        self.use_charge_spin = use_charge_spin
        self.atom = nn.Embedding(16, 24)
        self.radial = nn.Sequential(nn.Linear(1, 24), nn.SiLU(), nn.Linear(24, 24))
        self.cond = nn.Linear(2, 24)
        self.update = nn.GRUCell(24, 24)
        self.energy = nn.Linear(24, 1)
        self.force_head = nn.Linear(24, 3)
        self.confidence_head = nn.Linear(24, 5)

    def forward(
        self,
        inputs: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict total energy, direct forces, and confidence for v3.

        Parameters
        ----------
        inputs:
            ``(atomic_numbers, positions)`` or ``(atomic_numbers, positions,
            charge_spin)``.

        Returns
        -------
        torch.Tensor
            Scalar energy, or for v3 a tuple of energy, direct forces, and
            per-atom confidence logits.
        """

        if len(inputs) == 3:
            atomic_numbers, positions, charge_spin = inputs
        else:
            atomic_numbers, positions = inputs
            charge_spin = torch.zeros(atomic_numbers.shape[0], 2, device=positions.device)
        h = self.atom(atomic_numbers)
        dist = torch.cdist(positions, positions).unsqueeze(-1)
        messages = self.radial(dist).mean(dim=2)
        h = self.update(messages.reshape(-1, 24), h.reshape(-1, 24)).view_as(h)
        if self.use_charge_spin:
            h = h + self.cond(charge_spin).unsqueeze(1)
            forces = self.force_head(h)
            confidence = self.confidence_head(h)
            return self.energy(h).sum(dim=1), forces, confidence
        return self.energy(h).sum(dim=1)


def build_openpi_pi0_pytorch() -> Pi0Tiny:
    """Build OpenPI π0."""

    return Pi0Tiny()


def build_openvoice() -> OpenVoiceTiny:
    """Build OpenVoice tone-color converter."""

    return OpenVoiceTiny()


def build_packnet() -> PackNetTiny:
    """Build PackNet."""

    return PackNetTiny()


def build_orb_v2() -> OrbTiny:
    """Build Orb-v2-style potential."""

    return OrbTiny(use_charge_spin=False)


def build_orb_v3() -> OrbTiny:
    """Build Orb-v3-style charge/spin-conditioned potential."""

    return OrbTiny(use_charge_spin=True)


def example_pi0() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create compact π0 inputs."""

    return (
        torch.rand(2, 3, 32, 32),
        torch.randint(0, 64, (2, 6)),
        torch.rand(2, 8),
        torch.randn(2, 4, 7),
        torch.rand(2, 1),
    )


def example_openvoice() -> tuple[torch.Tensor, torch.Tensor]:
    """Create compact OpenVoice inputs."""

    return torch.rand(2, 80, 12), torch.rand(2, 16)


def example_packnet() -> tuple[torch.Tensor, torch.Tensor]:
    """Create compact PackNet input."""

    image = torch.rand(1, 3, 48, 64)
    sparse = torch.zeros(1, 1, 48, 64)
    sparse[:, :, ::8, ::8] = torch.rand(1, 1, 6, 8)
    return image, sparse


def example_orb() -> tuple[torch.Tensor, torch.Tensor]:
    """Create compact Orb-v2 inputs."""

    return torch.randint(1, 10, (2, 9)), torch.rand(2, 9, 3)


def example_orb_v3() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create compact Orb-v3 inputs."""

    return torch.randint(1, 10, (2, 9)), torch.rand(2, 9, 3), torch.rand(2, 2)


MENAGERIE_ENTRIES = [
    ("openpi_pi0_pytorch", "build_openpi_pi0_pytorch", "example_pi0", "2024", "robotics/vla"),
    ("OpenVoice", "build_openvoice", "example_openvoice", "2023", "audio/voice"),
    ("packnet01", "build_packnet", "example_packnet", "2020", "vision/depth"),
    ("packnet_san01", "build_packnet", "example_packnet", "2020", "vision/depth"),
    ("orb_v2", "build_orb_v2", "example_orb", "2024", "materials/gnn"),
    ("ORB / Orb-v3", "build_orb_v3", "example_orb_v3", "2025", "materials/gnn"),
]
