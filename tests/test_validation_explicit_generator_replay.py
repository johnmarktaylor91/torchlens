"""Forward replay of random ops that draw from a ``torch.Generator`` argument.

``torch.randn(shape, generator=g)`` draws from ``g``, not the global stream. Capture
records ``g``'s state before the call and replay restores it for the call, so the replay
reproduces the saved value; the generator itself is left where the model left it.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens._utils.rng.explicit_generators import (
    EXPLICIT_GENERATORS_KEY,
    explicit_generators_at,
    log_op_rng_states,
)


class _SeededNoise(nn.Module):
    """Adds noise from a generator seeded inside the forward (a Brownian-motion pattern)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gen = torch.Generator().manual_seed(1234)
        noise = torch.randn(x.shape, generator=gen)
        more = torch.rand(x.shape, generator=gen)
        return self.lin(x) + noise * more


class _HeldGenerator(nn.Module):
    """Draws from a generator the module holds across calls."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 3)
        self.gen = torch.Generator().manual_seed(7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = torch.randperm(x.shape[1], generator=self.gen)
        return self.lin(x[:, idx]) * torch.randn(1, generator=self.gen)


def test_forward_validation_replays_draws_from_a_seeded_generator() -> None:
    """A generator seeded in the forward replays each draw from its own pre-call state."""

    torch.manual_seed(0)
    assert tl.validate(_SeededNoise(), torch.randn(2, 4), scope="forward")


def test_forward_validation_replays_draws_from_a_held_generator() -> None:
    """A generator the module holds replays positional and keyword draws."""

    torch.manual_seed(1)
    assert tl.validate(_HeldGenerator(), torch.randn(2, 3), scope="forward")


def test_capture_records_generator_state_and_replay_restores_it() -> None:
    """The record holds the pre-call state; the bracket leaves the generator as found."""

    gen = torch.Generator().manual_seed(5)
    states = log_op_rng_states((3,), {"generator": gen})
    expected = torch.randn(3, generator=gen)
    after = gen.get_state()
    assert set(states[EXPLICIT_GENERATORS_KEY]) == {"kwarg:generator"}
    assert "torch" in states

    with explicit_generators_at(states, (3,), {"generator": gen}):
        replayed = torch.randn(3, generator=gen)
    assert torch.equal(replayed, expected)
    assert torch.equal(gen.get_state(), after)

    no_generator = log_op_rng_states((torch.ones(2),), {})
    assert EXPLICIT_GENERATORS_KEY not in no_generator
