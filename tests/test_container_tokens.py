"""Write-time container tokens for the freeze's container pooling (per-site interning, part C).

``log_current_autocast_state`` registers each fresh snapshot with a content
token in the capturing trace's registry; the freeze keys a registered cell by
its token instead of walking it. These tests pin the three properties the
design's proof needs:

* the token debug check (proof step 4): with ``TOKEN_DEBUG`` on, every token
  hit re-walks the cell and raises if the walked key is not the token's key;
  real captures (grad, no-grad, inference mode, CPU autocast) pass it;
* the check fires on a token whose content changed;
* the registry belongs to one trace: it moves to that trace's
  ``TraceCore.pool`` when the capture workspaces drop, two traces share no
  registry and no registered object, and the freeze drops it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes import _compaction, _container_key
from torchlens._data_classes.op import _container_pool_key
from torchlens._trace_core.pools import ContainerTokens


class _Mlp(nn.Module):
    """Small model with enough ops for every snapshot token to repeat."""

    def __init__(self) -> None:
        """Build two linear layers."""

        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a short chain with an autocast-sensitive matmul."""

        h = torch.relu(self.fc1(x))
        h = h * 2 + 1
        return torch.tanh(self.fc2(h))


def _spy_tokens(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ContainerTokens | None]]:
    """Record each compaction's trace and the registry it took from the core's pool."""

    taken: list[tuple[Any, ContainerTokens | None]] = []
    original: Callable[[Any], ContainerTokens | None] = _compaction._take_container_tokens

    def spy(trace: Any) -> ContainerTokens | None:
        """Call the real take and keep what it returned."""

        tokens = original(trace)
        taken.append((trace, tokens))
        return tokens

    monkeypatch.setattr(_compaction, "_take_container_tokens", spy)
    return taken


def _capture(mode: str) -> Any:
    """Trace the small model under one execution mode."""

    model = _Mlp()
    x = torch.randn(3, 8)
    if mode == "no_grad":
        with torch.no_grad():
            return tl.trace(model, x)
    if mode == "inference":
        with torch.inference_mode():
            return tl.trace(model, x)
    if mode == "autocast":
        with torch.autocast("cpu", dtype=torch.bfloat16):
            return tl.trace(model, x)
    return tl.trace(model, x)


@pytest.mark.parametrize("mode", ["grad", "no_grad", "inference", "autocast"])
def test_token_debug_check_passes_on_real_captures(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Every token hit's walked key equals the token's cached key (proof step 4)."""

    monkeypatch.setattr(_container_key, "TOKEN_DEBUG", True)
    hits: list[int] = []
    original_check = _container_key._check_token_key

    def counting_check(value: Any, cached: Any, walked: Any) -> None:
        """Count the hit, then run the real check."""

        hits.append(1)
        original_check(value, cached, walked)

    monkeypatch.setattr(_container_key, "_check_token_key", counting_check)
    taken = _spy_tokens(monkeypatch)
    trace = _capture(mode)
    assert taken, "the freeze never took the registry"
    traced, tokens = taken[0]
    assert tokens is not None and len(tokens) > 0
    assert traced._trace_core.pool.tokens is None
    assert hits, "no autocast cell was keyed by its token"
    states = [op.call.autocast_state for op in trace._layer_list]
    assert all(isinstance(state, dict) for state in states)
    if mode == "autocast":
        assert any(state.get("cpu", {}).get("enabled") for state in states)


def test_token_debug_check_raises_on_changed_content() -> None:
    """A token whose content changed after registration fails the debug check."""

    first = {"__execution__": {"enabled": False, "dtype": None}}
    second = {"__execution__": {"enabled": True, "dtype": None}}
    token = (True, False)
    tokens = ContainerTokens({id(first): (first, token), id(second): (second, token)})
    id_counts = {id(first): 1, id(second): 1}
    id_counts.update({id(first["__execution__"]): 1, id(second["__execution__"]): 1})
    content_key = _container_key.make_content_key(_container_pool_key, tokens)
    assert content_key(first, id_counts) == _container_pool_key(first, 0, [])
    old = _container_key.TOKEN_DEBUG
    _container_key.TOKEN_DEBUG = True
    try:
        with pytest.raises(RuntimeError, match="container token key mismatch"):
            content_key(second, id_counts)
    finally:
        _container_key.TOKEN_DEBUG = old


def test_token_key_refuses_a_nested_alias() -> None:
    """A registered cell whose nested container is aliased does not pool, as the walk."""

    inner = {"enabled": False, "dtype": None}
    outer = {"__execution__": inner}
    tokens = ContainerTokens({id(outer): (outer, ("t",))})
    content_key = _container_key.make_content_key(_container_pool_key, tokens)
    assert content_key(outer, {id(outer): 1, id(inner): 2}) is None
    assert content_key(outer, {id(outer): 1, id(inner): 1}) is not None


def test_two_traces_share_no_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each trace owns its registry; no registered object is shared; the freeze drops it."""

    taken = _spy_tokens(monkeypatch)
    _capture("grad")
    _capture("grad")
    registries = [tokens for _, tokens in taken if tokens is not None]
    assert len(registries) == 2
    first, second = registries
    assert first is not second
    assert first.by_id is not second.by_id
    first_objects = {id(entry[0]) for entry in first.by_id.values()}
    second_objects = {id(entry[0]) for entry in second.by_id.values()}
    assert first_objects.isdisjoint(second_objects)
    assert all(traced._trace_core.pool.tokens is None for traced, _ in taken)
