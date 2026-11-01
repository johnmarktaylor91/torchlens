"""Capture reproduces a sampling forward at the seed it discloses.

``tl.trace`` seeds torch's engines once per capture (``random_seed=`` or the
fresh auto seed disclosed on ``trace.random_seed``) and restores the host
stream afterwards, so a plain forward at that seed must produce the captured
output bit for bit. The sampler here is the Gumbel-softmax draw
(``torch.empty_like(logits).exponential_()``): an in-place seeded ATen
operator on a freshly allocated tensor, the shape the behaviour net's
DeepImmuno-GAN forward-match finding (F5) exercised. A difference would mean
capture changed the model's output, which it never may.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class GumbelHead(nn.Module):
    """Linear logits followed by a soft Gumbel-softmax sample."""

    def __init__(self) -> None:
        """Build the logits layer."""

        super().__init__()
        self.fc = nn.Linear(8, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a soft Gumbel-softmax sample of the logits."""

        return nn.functional.gumbel_softmax(self.fc(x), tau=0.75, hard=False)


def _build() -> tuple[GumbelHead, torch.Tensor]:
    torch.manual_seed(1)
    return GumbelHead(), torch.randn(2, 8)


def _plain_at(model: nn.Module, x: torch.Tensor, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    with torch.no_grad():
        return model(x).clone()


def test_explicit_seed_capture_matches_plain_forward_at_that_seed() -> None:
    """``random_seed=s`` reproduces ``torch.manual_seed(s); model(x)`` exactly."""

    model, x = _build()
    want = _plain_at(model, x, 7)
    trace = tl.trace(model, x.clone(), capture=tl.options.CaptureOptions(random_seed=7))
    assert trace.random_seed == 7
    assert torch.equal(trace.output_ops[0].out, want)
    assert torch.equal(trace["exponential_1_3"].out, trace["exponential_1_3"].out.clone())


def test_auto_seed_capture_matches_plain_forward_at_disclosed_seed() -> None:
    """An auto-seeded capture reproduces the plain forward at ``trace.random_seed``."""

    model, x = _build()
    torch.manual_seed(0)
    trace = tl.trace(model, x.clone())
    assert trace.random_seed is not None
    want = _plain_at(model, x, int(trace.random_seed))
    assert torch.equal(trace.output_ops[0].out, want)


def test_auto_seed_capture_leaves_the_host_stream_where_it_was() -> None:
    """The host's torch RNG state after a capture equals the state before it (R57)."""

    model, x = _build()
    torch.manual_seed(0)
    before = torch.get_rng_state().clone()
    tl.trace(model, x.clone())
    assert torch.equal(torch.get_rng_state(), before)
