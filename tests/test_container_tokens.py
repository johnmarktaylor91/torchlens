"""Write-time container tokens for the freeze's container pooling (per-site interning, part C).

``log_current_autocast_state`` registers each fresh snapshot with a content
token in the capturing trace's registry; the freeze keys a registered cell by
its token instead of walking it. These tests pin the three properties the
design's proof needs:

* the token debug check (proof step 4): with ``TOKEN_DEBUG`` on, every token
  hit re-walks the cell and raises if the walked key is not the token's key;
  real captures (grad, no-grad, inference mode, CPU autocast) pass it;
* the check fires on a token whose content changed;
* the registry belongs to one trace: two traces share no registry and no
  registered object, and the freeze drops it.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes import _compaction, _container_key
from torchlens._data_classes.op import _container_pool_key
from torchlens._postprocess._arg_position_tokens import register_arg_position_keys
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
    """Record each compaction's taken registry and the workspace's state after."""

    taken: list[tuple[Any, ContainerTokens | None]] = []
    original: Callable[[Any], ContainerTokens | None] = _compaction._take_container_tokens

    def spy(trace: Any) -> ContainerTokens | None:
        """Call the real take and keep what it returned."""

        tokens = original(trace)
        taken.append((trace._wrapper_runtime_ws, tokens))
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
    workspace, tokens = taken[0]
    assert tokens is not None and len(tokens) > 0
    assert workspace.container_tokens is False
    assert hits, "no autocast cell was keyed by its token"
    states = [op.func_autocast_state for op in trace._layer_list]
    assert all(isinstance(state, dict) for state in states)
    if mode == "autocast":
        assert any(state.get("cpu", {}).get("enabled") for state in states)


def test_token_debug_check_raises_on_changed_content() -> None:
    """A token whose content changed after registration fails the debug check."""

    first = {"__execution__": {"enabled": False, "dtype": None}}
    second = {"__execution__": {"enabled": True, "dtype": None}}
    token = (True, False)
    tokens = ContainerTokens(
        {id(first): (first, token, (id(first["__execution__"]),), 2)}
        | {id(second): (second, token, (id(second["__execution__"]),), 2)}
    )
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
    tokens = ContainerTokens({id(outer): (outer, ("t",), (id(inner),), 2)})
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
    assert all(workspace.container_tokens is False for workspace, _ in taken)


def _registered_by_field(trace: Any, tokens: ContainerTokens) -> dict[str, int]:
    """Count the registered cells of each op field in a frozen trace's store rows."""

    by_id = tokens.by_id
    counts: dict[str, int] = {}
    for op in trace._layer_list:
        for name in ("func_autocast_state", "transform_config", "parent_arg_positions"):
            value = getattr(op, name)
            entry = by_id.get(id(value))
            if entry is not None and entry[0] is value:
                counts[name] = counts.get(name, 0) + 1
    return counts


@pytest.mark.parametrize("mode", ["grad", "autocast"])
def test_exact_tokens_cover_transform_config_and_arg_positions(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """The emit and the rename pass register exact keys, and every one walks equal.

    ``TOKEN_DEBUG`` re-walks each registered cell at the freeze and raises on a
    difference; the registered entries carry ``TOKEN_EXACT`` and are keyed by the
    real content key, so the pooling result equals the full walk's.
    """

    monkeypatch.setattr(_container_key, "TOKEN_DEBUG", True)
    taken = _spy_tokens(monkeypatch)
    _capture(mode)
    _, tokens = taken[0]
    assert tokens is not None
    exact = [entry for entry in tokens.by_id.values() if entry[3] & 1]
    kinds = {next(iter(entry[0])) for entry in exact}
    assert "_tl_annotations" in kinds, "no transform_config cell registered"
    assert "args" in kinds, "no parent_arg_positions cell registered"
    for entry in exact:
        assert entry[1] == _container_pool_key(entry[0], 0, [])


def test_exact_token_mismatch_raises() -> None:
    """An exact token that is not the content key fails the debug check."""

    cell = {"args": {0: "relu_1_1"}, "kwargs": {}}
    wrong = ("d", ())
    nested = (id(cell["args"]), id(cell["kwargs"]))
    tokens = ContainerTokens({id(cell): (cell, wrong, nested, 1)})
    id_counts = {id(cell): 1, nested[0]: 1, nested[1]: 1}
    content_key = _container_key.make_content_key(_container_pool_key, tokens)
    old = _container_key.TOKEN_DEBUG
    _container_key.TOKEN_DEBUG = True
    try:
        with pytest.raises(RuntimeError, match="container token key mismatch"):
            content_key(cell, id_counts)
    finally:
        _container_key.TOKEN_DEBUG = old
    assert content_key(cell, {**id_counts, nested[0]: 2}) is None


def test_arg_position_keys_equal_the_walk() -> None:
    """The rename pass's stated key is the walk's key; unkeyable dicts stay unregistered."""

    keyed = [
        {"args": {0: "a_1_1", 2: "b_2_1"}, "kwargs": {"other": "c_3_1"}},
        {"args": {}, "kwargs": {}},
        {"kwargs": {"x": "a_1_1"}, "args": {1: "d_4_1"}},
    ]
    unkeyed = [
        {"args": {(0, 1): "a_1_1"}, "kwargs": {}},
        {"args": {0: None}, "kwargs": {}},
        {"args": {0: "a_1_1"}},
        {"args": [("a_1_1",)], "kwargs": {}},
    ]
    trace = SimpleNamespace(_wrapper_runtime_ws=SimpleNamespace(container_tokens=None))
    register_arg_position_keys(trace, [*keyed, *unkeyed, {}])
    table = trace._wrapper_runtime_ws.container_tokens
    assert set(table) == {id(cell) for cell in keyed}
    for cell in keyed:
        assert table[id(cell)][1] == _container_pool_key(cell, 0, [])
        assert table[id(cell)][2] == tuple(map(id, cell.values()))
