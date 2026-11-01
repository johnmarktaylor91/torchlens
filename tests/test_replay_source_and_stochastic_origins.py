"""Replay edits at source records, on loaded traces, at random-source sites and past an index.

Pins four behaviour-net findings from the intervention cells (2026-11-01):

- an edit on a model-input leaf (``input_1``: no callable, no parents) fires on the captured
  value and propagates through its cone instead of refusing "has no func for replay";
- an edit on a trace loaded from a saved file refuses typed and NAMES the loaded-trace
  condition and its remedy (a save keeps no callables, so nothing can be replayed);
- an edit at a random-source site (``randn_like`` with no recorded RNG state) edits the
  captured draw instead of a fresh one the replay drew before the hook fired;
- an edit that drives a downstream index lookup out of range surfaces as a typed replay
  refusal naming the op that raised and the edit origin, with the torch error chained, and
  commits nothing.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.intervention.errors import ReplayPreconditionError

_READY = tl.options.CaptureOptions(layers_to_save="all", intervention_ready=True)


class _Affine(nn.Module):
    """Linear then relu: the input leaf feeds one replayable op."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 3, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))


class _NoisyAffine(nn.Module):
    """Adds a fresh ``randn_like`` draw to the input before the linear."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 3, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x + torch.randn_like(x))


class _IndexLookup(nn.Module):
    """A linear whose output, cast to long, indexes an embedding table of 59 rows."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.emb = nn.Embedding(59, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        idx = h.long().clamp(min=0)
        return self.emb(idx).sum(-1) + h


def _op_by_func(trace: tl.Trace, prefix: str) -> str:
    for op in trace.saved_ops:
        if str(getattr(op, "func_name", "") or "").lstrip("_").startswith(prefix):
            return op.label
    raise AssertionError(f"no saved op with func name {prefix!r}")


@pytest.mark.smoke
def test_fork_edit_on_input_leaf_fires_on_captured_value_and_propagates() -> None:
    torch.manual_seed(0)
    model = _Affine().eval()
    x = torch.randn(2, 4)
    trace = tl.trace(model, x, capture=_READY)
    captured_input = trace["input_1"].out.detach().clone()

    fork = trace.fork()
    fork.do(tl.label("input_1"), tl.scale(0.5))

    assert torch.equal(fork["input_1"].out, captured_input * 0.5)
    with torch.no_grad():
        expected = torch.relu(model.lin(captured_input * 0.5))
    assert torch.allclose(fork.reconstruct_output(), expected, atol=1e-6)
    assert torch.equal(trace["input_1"].out, captured_input), "the edit wrote through"
    assert torch.allclose(trace.reconstruct_output(), model(x).detach(), atol=1e-6)
    helpers = [getattr(record, "helper_name", None) for record in fork["input_1"].interventions]
    assert helpers == ["scale"]


@pytest.mark.smoke
def test_fork_edit_on_loaded_trace_refuses_naming_the_loaded_condition(tmp_path) -> None:
    torch.manual_seed(0)
    model = _Affine().eval()
    trace = tl.trace(model, torch.randn(2, 4), capture=_READY)
    trace.save(tmp_path / "trace.tlspec", level="portable")
    loaded = tl.load(tmp_path / "trace.tlspec")

    fork = loaded.fork()
    with pytest.raises(ReplayPreconditionError) as info:
        fork.do(tl.label("linear_1_1"), tl.scale(0.5))
    message = str(info.value)
    assert "linear_1_1" in message
    assert "loaded from a saved file" in message
    assert "intervention_ready=True" in message
    assert "trace.run(model, inputs)" in message


@pytest.mark.smoke
def test_fork_edit_at_random_source_site_edits_the_captured_draw() -> None:
    torch.manual_seed(0)
    model = _NoisyAffine().eval()
    x = torch.randn(2, 4)
    trace = tl.trace(model, x, capture=_READY)
    noise_label = _op_by_func(trace, "randn_like")
    captured_noise = trace[noise_label].out.detach().clone()
    add_label = _op_by_func(trace, "add")

    fork = trace.fork()
    fork.do(tl.label(noise_label), tl.scale(0.5))

    assert torch.equal(fork[noise_label].out, captured_noise * 0.5), (
        "the replay redrew the noise before the edit fired"
    )
    assert torch.allclose(fork[add_label].out, x + captured_noise * 0.5, atol=1e-6)
    with torch.no_grad():
        expected = model.lin(x + captured_noise * 0.5)
    assert torch.allclose(fork.reconstruct_output(), expected, atol=1e-6)
    assert torch.equal(trace[noise_label].out, captured_noise), "the edit wrote through"


@pytest.mark.smoke
def test_edit_past_an_index_lookup_refuses_typed_naming_site_and_origin() -> None:
    torch.manual_seed(0)
    model = _IndexLookup().eval()
    x = torch.rand(2, 4)
    trace = tl.trace(model, x, capture=_READY)
    captured_linear = trace["linear_1_1"].out.detach().clone()
    embedding_label = _op_by_func(trace, "embedding")

    fork = trace.fork()
    with pytest.raises(ReplayPreconditionError) as info:
        fork.do(tl.label("linear_1_1"), tl.scale(1000.0))
    message = str(info.value)
    assert embedding_label.split(":")[0] in message, message
    assert "'linear_1_1'" in message, message
    assert "IndexError" in message, message
    assert isinstance(info.value.__cause__, IndexError)
    assert torch.equal(fork["linear_1_1"].out, captured_linear), "a failed replay committed"
    assert torch.equal(trace["linear_1_1"].out, captured_linear)
