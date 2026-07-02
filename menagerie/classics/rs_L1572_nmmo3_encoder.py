# SOURCE: vendored from PufferAI/PufferLib @ 4.0 (tests/test_nmmo3_encoder.py)
#
# `NMMO3EncoderRef` is the real PyTorch reference observation encoder for PufferLib's
# Neural MMO 3 baseline (`ocean/nmmo3`), used in `tests/test_nmmo3_encoder.py` as the
# ground-truth numerical reference that PufferLib's fused CUDA `ocean.cu` kernel is
# checked against forward+backward -- i.e. this genuinely IS the actual forward pass
# of the shipped baseline policy's observation encoder, not an approximation. It
# multi-hot-encodes the packed per-tile categorical map observation (10 factored
# fields per tile: entity/terrain/item type codes etc.) into a one-hot channel stack
# via `scatter_`, runs it through a 2-layer CNN torso, embeds the per-player scalar
# features, and projects the concatenation (map CNN features + player embedding +
# raw player floats + per-population reward floats) through a final linear+ReLU into
# the policy trunk's 512-d representation. PufferLib is the actively maintained
# NeurIPS-competition-lineage baseline/training stack for Neural MMO (the original
# openai/neural-mmo `forge`/ANN code is 2019-era and inseparably coupled to a bespoke
# realm/entity/stimulus simulation framework that cannot be vendored as a standalone
# network). Vendored verbatim (module code only; the CUDA comparison harness and test
# functions around it are dropped).

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Per-tile factored categorical field cardinalities (terrain/entity/item codes etc.)
# and their cumulative one-hot channel offsets, from PufferLib's NMMO3 observation
# packing (tests/test_nmmo3_encoder.py).
FACTORS = [4, 4, 17, 5, 3, 5, 5, 5, 7, 4]
OFFSETS_NP = [0] + list(np.cumsum(FACTORS)[:-1])


class NMMO3EncoderRef(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        offsets = torch.tensor(OFFSETS_NP).view(1, -1, 1, 1)
        self.register_buffer("offsets", offsets)
        self.conv1 = nn.Conv2d(59, 128, 5, stride=3)
        self.conv2 = nn.Conv2d(128, 128, 3, stride=1)
        self.embed = nn.Embedding(128, 32)
        self.proj = nn.Linear(1817, 512)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        B = observations.shape[0]
        ob_map = observations[:, :1650].view(B, 11, 15, 10)
        mh = torch.zeros(B, 59, 11, 15, dtype=torch.float32, device=observations.device)
        codes = ob_map.long().permute(0, 3, 1, 2) + self.offsets
        mh.scatter_(1, codes, 1)
        x = F.relu(self.conv1(mh))
        x = self.conv2(x).flatten(1)
        ob_player = observations[:, 1650:-10]
        player_embed = self.embed(ob_player.int()).flatten(1)
        ob_reward = observations[:, -10:]
        cat = torch.cat([x, player_embed, ob_player.float(), ob_reward.float()], dim=1)
        return F.relu(self.proj(cat))


def build_nmmo3_encoder() -> nn.Module:
    model = NMMO3EncoderRef()
    model.eval()
    return model


def _generate_valid_obs(batch: int) -> torch.Tensor:
    """Real PufferLib generator for a validly-ranged packed observation vector
    (tests/test_nmmo3_encoder.py::generate_valid_obs), so every factored field is
    in-range for the `scatter_` one-hot and the player embedding table."""

    obs = torch.zeros(batch, 1707, dtype=torch.float32)
    for h in range(11):
        for w in range(15):
            for f in range(10):
                idx = (h * 15 + w) * 10 + f
                obs[:, idx] = torch.randint(0, FACTORS[f], (batch,)).float()
    obs[:, 1650:1697] = torch.randint(0, 128, (batch, 47)).float()
    obs[:, 1697:1707] = torch.randint(0, 256, (batch, 10)).float()
    return obs


def example_input_nmmo3_encoder() -> torch.Tensor:
    return _generate_valid_obs(2)


MENAGERIE_ZOO = "vendored-pytorch"

MENAGERIE_ENTRIES = [
    (
        "PufferLib NMMO3 observation encoder (multi-hot tile CNN + player embed)",
        "build_nmmo3_encoder",
        "example_input_nmmo3_encoder",
        "2024",
        "DC",
    ),
]
