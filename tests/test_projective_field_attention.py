"""Projective fields through attention: per-target status and the gradient fallback.

Attention mixes positions by data-dependent weights, so the field engine has no
per-unit index geometry through it and the geometric method refuses. The
gradient method is the documented fallback: it needs a capture with
``backward_ready=True`` and ``save_mode="reference"``, whose saved payloads stay
connected to the autograd graph (including payloads of non-mutating operator
dunders such as ``__mul__``).
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import SelectionError
from torchlens.receptive_field import ReceptiveFieldStatus
from torchlens.receptive_field._errors import ReceptiveFieldUnavailableError

_ARMED = {"backward_ready": True, "save_mode": "reference"}
_SEQ = 5
_SEED_POSITION = 3


class _ScaledMlp(nn.Module):
    """Linear, tanh, a scalar ``*`` (the ``__mul__`` dunder) and a second linear."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.tanh(self.a(x)) * 0.5)


class _CausalAttention(nn.Module):
    """One head of causal self-attention between two linear layers."""

    def __init__(self, width: int = 8, *, fused: bool = False) -> None:
        super().__init__()
        self.fused = fused
        self.pre = nn.Linear(4, width)
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.proj = nn.Linear(width, width)
        self.scale = 1.0 / math.sqrt(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre(x)
        if self.fused:
            heads = [part(h).unsqueeze(1) for part in (self.q, self.k, self.v)]
            mixed = nn.functional.scaled_dot_product_attention(*heads, is_causal=True)
            return self.proj(mixed.squeeze(1))
        scores = (self.q(h) @ self.k(h).transpose(-1, -2)) * self.scale
        length = scores.shape[-1]
        future = torch.ones(length, length, dtype=torch.bool).triu(1)
        weights = scores.masked_fill(future, float("-inf")).softmax(dim=-1)
        return self.proj(weights @ self.v(h))


def _row(selection, op):
    """Return the one entry of ``selection`` at ``op``."""

    rows = [entry for entry in selection if f"{entry.key[0]}:{entry.key[1]}" == op.label]
    assert len(rows) == 1, f"expected one row for {op.label}, got {len(rows)}"
    return rows[0]


def _only_output(trace, address: str):
    """Return the only output op of the module at ``address``."""

    module = trace.modules[address]
    (label,) = module.calls[0].output_ops
    return trace.ops[label]


def _positions_reached(mask: torch.Tensor) -> list[int]:
    """Sequence positions with at least one selected unit in a (batch, seq, width) mask."""

    return [pos for pos in range(mask.shape[1]) if bool(mask[:, pos].any())]


def test_reference_mode_dunder_payload_stays_grad_connected() -> None:
    """A ``__mul__`` payload saved by reference is the live tensor, not a detached copy."""

    torch.manual_seed(0)
    trace = tl.trace(
        _ScaledMlp().eval(), torch.randn(1, 4), capture=tl.options.CaptureOptions(**_ARMED)
    )
    try:
        mul = next(op for op in trace.ops if op.func_name == "__mul__")
        last = _only_output(trace, "b")
        (grad,) = torch.autograd.grad(last.out.sum(), mul.out, allow_unused=True)
        assert grad is not None
        assert grad.shape == mul.out.shape
    finally:
        trace.cleanup()


@pytest.mark.parametrize("fused", [False, True], ids=["explicit", "fused_kernel"])
def test_gradient_method_through_attention_respects_causality(fused: bool) -> None:
    """A unit at position 3 reaches the attention output at positions 3 and 4 only.

    The fused kernel has no second derivative, so the projective probe answers
    with one backward pass per target element instead of a double VJP.
    """

    torch.manual_seed(1)
    trace = tl.trace(
        _CausalAttention(fused=fused).eval(),
        torch.randn(1, _SEQ, 4),
        capture=tl.options.CaptureOptions(**_ARMED),
    )
    try:
        seed = _only_output(trace, "pre")
        target = _only_output(trace, "proj")
        units = [(0, _SEED_POSITION, j) for j in range(seed.out.shape[-1])]
        reached = seed.projective_field.of(units, within="proj", method="gradient")
        assert _positions_reached(_row(reached, target).mask) == list(range(_SEED_POSITION, _SEQ))
    finally:
        trace.cleanup()


def _gpt2(attention: str = "sdpa"):
    """A two-block GPT-2 with random weights, cache on, and a five-token input."""

    transformers = pytest.importorskip("transformers")
    config = transformers.GPT2Config(
        n_layer=2,
        n_head=4,
        n_embd=32,
        vocab_size=64,
        n_positions=16,
        use_cache=True,
        attn_implementation=attention,
    )
    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(config).eval()
    ids = torch.randint(0, 64, (1, _SEQ), generator=torch.Generator().manual_seed(0))
    return model, ids


@pytest.fixture(scope="module")
def gpt2():
    """The default-attention GPT-2 shared by the plain-capture tests."""

    return _gpt2()


@pytest.mark.heavy
def test_gpt2_projective_status_answers_per_target(gpt2) -> None:
    """Several reachable outputs give one status per output instead of an ambiguity error."""

    model, ids = gpt2
    trace = tl.trace(model, ids)
    try:
        view = _only_output(trace, "transformer.h.0.mlp.act").projective_field
        assert len(view.per_input) > 1
        statuses = view.status
        assert set(statuses) == set(view.per_input)
        assert all(isinstance(status, ReceptiveFieldStatus) for status in statuses.values())
    finally:
        trace.cleanup()


@pytest.mark.heavy
def test_gpt2_geometric_refusal_names_the_gradient_capture(gpt2) -> None:
    """The geometric method refuses through attention and names the capture the fallback needs."""

    model, ids = gpt2
    trace = tl.trace(model, ids)
    try:
        view = _only_output(trace, "transformer.h.0.mlp.act").projective_field
        units = [(0, _SEED_POSITION, j) for j in range(4)]
        with pytest.raises(SelectionError) as excinfo:
            view.of(units, within="transformer.h.1.attn.c_proj")
        assert excinfo.value.fields["code"] == "influence_geometry_unavailable"
        message = str(excinfo.value)
        assert "method='gradient'" in message
        assert "backward_ready=True" in message
        assert 'save_mode="reference"' in message
    finally:
        trace.cleanup()


@pytest.mark.heavy
def test_gpt2_gradient_on_plain_capture_says_how_to_recapture(gpt2) -> None:
    """A plain capture refuses the gradient method with a recipe usable on token ids."""

    model, ids = gpt2
    trace = tl.trace(model, ids)
    try:
        view = _only_output(trace, "transformer.h.0.mlp.act").projective_field
        with pytest.raises(ReceptiveFieldUnavailableError) as excinfo:
            view.of(
                [(0, _SEED_POSITION, 0)], within="transformer.h.1.attn.c_proj", method="gradient"
            )
        message = str(excinfo.value)
        assert "backward_ready=True" in message
        assert "requires_grad_" not in message
    finally:
        trace.cleanup()


@pytest.mark.heavy
@pytest.mark.parametrize("attention", ["sdpa", "eager"])
def test_gpt2_gradient_method_through_attention_respects_causality(attention: str) -> None:
    """On a backward-ready reference capture the gradient method answers through attention."""

    model, ids = _gpt2(attention)
    trace = tl.trace(model, ids, capture=tl.options.CaptureOptions(**_ARMED))
    try:
        seed = _only_output(trace, "transformer.h.0.mlp.act")
        target = _only_output(trace, "transformer.h.1.attn.c_proj")
        units = [(0, _SEED_POSITION, j) for j in range(4)]
        reached = seed.projective_field.of(
            units, within="transformer.h.1.attn.c_proj", method="gradient"
        )
        assert _positions_reached(_row(reached, target).mask) == list(range(_SEED_POSITION, _SEQ))
    finally:
        trace.cleanup()


def test_fused_attention_sweep_over_probe_budget_keeps_the_refusal() -> None:
    """A target site larger than probe_budget refuses rather than sweeping every element."""

    torch.manual_seed(1)
    trace = tl.trace(
        _CausalAttention(fused=True).eval(),
        torch.randn(1, _SEQ, 4),
        capture=tl.options.CaptureOptions(**_ARMED),
    )
    try:
        seed = _only_output(trace, "pre")
        target_numel = _only_output(trace, "proj").out.numel()
        with pytest.raises(ReceptiveFieldUnavailableError, match="probe_budget"):
            seed.projective_field.of(
                [(0, _SEED_POSITION, 0)],
                within="proj",
                method="gradient",
                probe_budget=target_numel - 1,
            )
    finally:
        trace.cleanup()
