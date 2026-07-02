# SOURCE: vendored from google-deepmind/open_spiel @ master
# (open_spiel/python/pytorch/dqn.py: MLP; open_spiel/python/pytorch/nfsp.py: NFSP)
#
# `MLP` below is the real, unmodified network class OpenSpiel's PyTorch DQN and NFSP
# (Neural Fictitious Self-Play) agents both use as their function approximator --
# `dqn.DQN` builds one as its Q-network and `nfsp.NFSP` builds a second one (with a
# different seed) as its `_avg_network` (open_spiel/python/pytorch/nfsp.py:194-210).
# NFSP's defining architectural trait is exactly this "two networks trained by two
# different learning rules over the same info-state representation" design (a
# best-response DQN network plus a supervised average-policy network, reconciled by
# anticipatory-parameter mixing at the *agent* level, not inside either network).
# `NFSPDualNetwork` below is a thin staging-only composite (NOT part of the original
# OpenSpiel source) that instantiates two real vendored `MLP`s exactly as
# `nfsp.NFSP.__init__` does, so a single TorchLens trace captures both networks that
# make up the real NFSP model pair. `MLP.__init__`/`forward` are vendored verbatim
# from `dqn.py`; only the `open_spiel.python.pytorch.dqn.set_seed` helper (itself
# trivial: `torch.manual_seed(seed)` plus numpy/cuda seeding not needed for a single
# forward trace) is inlined to avoid an `open_spiel` import.

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)


class MLP(nn.Module):
    """A simple network built from nn.linear layers."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Iterable[int],
        output_size: int,
        final_activation: nn.Module | None = None,
        seed: int = 42,
    ) -> None:
        """Create the MLP.

        Args:
          input_size: (int) number of inputs.
          hidden_sizes: (list) sizes (number of units) of each hidden layer.
          output_size: (int) number of outputs.
          final_activation: (nn.Module) final activation of the network. Defaults to
            None.
          seed: (int) seed for the random number generator.
        """
        super().__init__()
        _set_seed(seed)
        layers_ = []

        def _create_linear_block(in_features, out_features):
            return nn.Sequential(nn.Linear(in_features, out_features), nn.ReLU())

        # Input and Hidden layers
        for size in hidden_sizes:
            layers_.append(_create_linear_block(input_size, size))
            input_size = size
        # Output layer
        layers_.append(nn.Linear(input_size, output_size))
        if final_activation:
            layers_.append(final_activation)
        self.model = nn.Sequential(*layers_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class NFSPDualNetwork(nn.Module):
    """Staging composite (not original OpenSpiel code): the real best-response
    Q-network + real average-policy network that together define an NFSP agent,
    each a genuine vendored `MLP` instance built the same way
    `nfsp.NFSP.__init__` builds them."""

    def __init__(
        self,
        state_representation_size: int,
        hidden_layers_sizes: Iterable[int],
        num_actions: int,
        seed: int = 42,
    ) -> None:
        super().__init__()
        hidden_layers_sizes = list(hidden_layers_sizes)
        # Best-response network (mirrors dqn.DQN's internal Q-network).
        self.q_network = MLP(state_representation_size, hidden_layers_sizes, num_actions, seed=seed)
        # Average-policy network (nfsp.NFSP._avg_network, seed=seed + 1).
        self.avg_network = MLP(
            state_representation_size, hidden_layers_sizes, num_actions, seed=seed + 1
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q_network(x), self.avg_network(x)


def build_nfsp_dual_network() -> nn.Module:
    model = NFSPDualNetwork(
        state_representation_size=32,
        hidden_layers_sizes=[128, 128],
        num_actions=8,
    )
    model.eval()
    return model


def example_input_nfsp_dual_network() -> torch.Tensor:
    return torch.randn(4, 32)


MENAGERIE_ZOO = "vendored-pytorch"

MENAGERIE_ENTRIES = [
    (
        "OpenSpiel NFSP dual network (best-response DQN net + average-policy net)",
        "build_nfsp_dual_network",
        "example_input_nfsp_dual_network",
        "2016",
        "DC",
    ),
]
