"""Deterministic counters for capture's per-op bookkeeping overhead.

The tensor search muted log records around every property read with
``logging.disable``, which clears every logger's level cache (costly when a
library registers many loggers): two clears per property read. The search now
mutes without clearing any cache (``_utils/_log_mute.py``). Capture also
entered ``@contextmanager`` generators on every op (phase timing, internal
scalar reads). These tests pin that the search mutes and restores while it
reads properties, that it clears no level cache however many properties it
reads, that capture's generator context-manager entries
do not grow with the op count, and the mute window's contract: the user's own
``logging.disable`` level survives a trace (also when a search raises), and a
record the model logs in its forward still reaches a handler.

Every op also derived its function's portable registry key afresh and
``copy.deepcopy``-ed its parent label map. The last two tests pin that, during
capture, neither the key derivation nor ``deepcopy`` runs once per op.
"""

from __future__ import annotations

import contextlib
import copy
import logging
from typing import Any

import pytest
import torch
from _hermetic_counters import quiet_gc, thread_counting
from torch import nn

import torchlens as tl
import torchlens._postprocess as postprocess_pkg
import torchlens._utils._quiet_property as quiet_property
import torchlens.intervention._resolver_registry as resolver_registry

_CACHE_LOGGER = "tl_counter_cache"
_MODEL_LOGGER = "tl_counter_model"


def _cache_class(n_props: int) -> type:
    """A generation-cache stand-in with ``n_props`` deprecated, logging properties."""

    def init(self: Any) -> None:
        self.keys = torch.ones(1, 8)

    namespace: dict[str, Any] = {"__init__": init}
    for index in range(n_props):

        def getter(self: Any, index: int = index) -> int:
            logging.getLogger(_CACHE_LOGGER).warning("size_%d is deprecated", index)
            return index

        namespace[f"size_{index}"] = property(getter)
    return type(f"_Cache{n_props}", (), namespace)


class _Block(nn.Module):
    def forward(self, x: torch.Tensor, cache: Any) -> torch.Tensor:
        return torch.relu(x + cache.keys)


class _Stack(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_Block() for _ in range(3))

    def forward(self, x: torch.Tensor, cache: Any) -> torch.Tensor:
        logging.getLogger(_MODEL_LOGGER).warning("forward ran")
        for block in self.blocks:
            x = block(x, cache)
        return x


def _mute_counts_during_trace(monkeypatch: pytest.MonkeyPatch, n_props: int) -> dict[str, int]:
    """Level-cache clears, mute window openings and closings during one trace."""

    counts: dict[str, int] = {}
    wrapped = (
        (logging.Manager, "_clear_cache", "clear"),
        (quiet_property, "mute", "mute"),
        (quiet_property, "unmute", "unmute"),
    )
    with monkeypatch.context() as patch:
        for owner, name, key in wrapped:
            patch.setattr(owner, name, thread_counting(getattr(owner, name), counts, key))
        cache = _cache_class(n_props)()
        with quiet_gc():
            tl.trace(_Stack(), torch.zeros(1, 8), input_kwargs={"cache": cache})
    return {key: counts.get(key, 0) for _owner, _name, key in wrapped}


def test_logger_cache_clears_do_not_grow_with_property_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    one = _mute_counts_during_trace(monkeypatch, 1)
    many = _mute_counts_during_trace(monkeypatch, 8)
    assert one["mute"] >= 1  # the search did mute,
    assert one["unmute"] == one["mute"]  # and restored
    assert one["clear"] == 0
    assert many == one, f"1 property: {one}; 8 properties: {many}"


def test_user_disable_level_survives_and_forward_records_reach_handlers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logging.disable(logging.DEBUG)
    try:
        with caplog.at_level(logging.WARNING):
            trace = tl.trace(_Stack(), torch.zeros(1, 8), input_kwargs={"cache": _cache_class(3)()})
            assert logging.root.manager.disable == logging.DEBUG
        names = [record.name for record in caplog.records]
        assert _CACHE_LOGGER not in names  # the search's property reads stay quiet
        assert _MODEL_LOGGER in names  # the model's own forward record still arrives
        assert torch.equal(trace.output, torch.full((1, 8), 3.0))
    finally:
        logging.disable(logging.NOTSET)


def test_mute_scope_restores_the_previous_level_when_the_search_raises() -> None:
    from torchlens._utils._quiet_property import _QuietPropertyLogs, read_attr_quietly

    cache = _cache_class(1)()
    logging.disable(logging.DEBUG)
    try:
        with pytest.raises(RuntimeError, match="search failed"):
            with _QuietPropertyLogs():
                assert read_attr_quietly(cache, "size_0") == 0
                assert logging.root.manager.disable == logging.WARNING
                raise RuntimeError("search failed")
        assert logging.root.manager.disable == logging.DEBUG
    finally:
        logging.disable(logging.NOTSET)


def _chain(depth: int) -> nn.Module:
    torch.manual_seed(0)
    layers: list[nn.Module] = []
    for _ in range(depth):
        layers.extend((nn.Linear(4, 4), nn.ReLU()))
    return nn.Sequential(*layers).eval()


def _capture_generator_cm_entries(monkeypatch: pytest.MonkeyPatch, depth: int) -> int:
    """``@contextmanager`` entries on this thread during capture (postprocess excluded)."""

    counts: dict[str, int] = {}
    state = {"capture": True}
    real_enter = contextlib._GeneratorContextManager.__enter__
    counting = thread_counting(real_enter, counts, "enter")
    real_postprocess = postprocess_pkg.postprocess

    def enter(self: Any) -> Any:
        return counting(self) if state["capture"] else real_enter(self)

    def postprocess(*args: Any, **kwargs: Any) -> Any:
        state["capture"] = False
        try:
            return real_postprocess(*args, **kwargs)
        finally:
            state["capture"] = True

    model = _chain(depth)
    with quiet_gc():
        monkeypatch.setattr(contextlib._GeneratorContextManager, "__enter__", enter)
        monkeypatch.setattr(postprocess_pkg, "postprocess", postprocess)
        try:
            tl.trace(model, torch.randn(2, 4))
        finally:
            monkeypatch.setattr(contextlib._GeneratorContextManager, "__enter__", real_enter)
            monkeypatch.setattr(postprocess_pkg, "postprocess", real_postprocess)
    return counts.get("enter", 0)


def test_capture_enters_no_generator_context_manager_per_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tl.trace(_chain(2), torch.randn(2, 4))  # warm one-time instrumentation
    small = _capture_generator_cm_entries(monkeypatch, 2)
    large = _capture_generator_cm_entries(monkeypatch, 12)
    assert large == small, f"4 ops: {small} entries; 24 ops: {large} entries"


class _Loop(nn.Module):
    """One module whose forward runs ``2 * n`` torch builtin ops."""

    def __init__(self, n: int) -> None:
        """Store the number of sin/cos rounds."""
        super().__init__()
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``cos(sin(x))`` ``n`` times."""
        for _ in range(self.n):
            x = torch.cos(torch.sin(x))
        return x


def _capture_calls(monkeypatch: pytest.MonkeyPatch, owner: Any, name: str, n: int) -> int:
    """Calls of ``owner.<name>`` on this thread during capture (postprocess excluded)."""

    counts: dict[str, int] = {}
    state = {"capture": True}
    real = getattr(owner, name)
    counting = thread_counting(real, counts, "calls")
    real_postprocess = postprocess_pkg.postprocess

    def patched(*args: Any, **kwargs: Any) -> Any:
        """Count the call while capture runs, else pass it through."""
        return counting(*args, **kwargs) if state["capture"] else real(*args, **kwargs)

    def postprocess(*args: Any, **kwargs: Any) -> Any:
        """Run the real postprocess with counting switched off."""
        state["capture"] = False
        try:
            return real_postprocess(*args, **kwargs)
        finally:
            state["capture"] = True

    with quiet_gc():
        monkeypatch.setattr(owner, name, patched)
        monkeypatch.setattr(postprocess_pkg, "postprocess", postprocess)
        try:
            tl.trace(_Loop(n), torch.randn(2, 4))
        finally:
            monkeypatch.setattr(owner, name, real)
            monkeypatch.setattr(postprocess_pkg, "postprocess", real_postprocess)
    return counts.get("calls", 0)


def test_capture_derives_each_function_registry_key_once_per_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registry-key derivations do not grow with the op count of one function."""
    tl.trace(_Loop(1), torch.randn(2, 4))  # warm one-time instrumentation
    name = "function_registry_key_from_callable"
    small = _capture_calls(monkeypatch, resolver_registry, name, 2)
    large = _capture_calls(monkeypatch, resolver_registry, name, 12)
    assert large == small, f"4 ops: {small} key derivations; 24 ops: {large}"


def test_capture_deepcopies_nothing_per_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """``copy.deepcopy`` calls do not grow with the op count."""
    tl.trace(_Loop(1), torch.randn(2, 4))  # warm one-time instrumentation
    small = _capture_calls(monkeypatch, copy, "deepcopy", 2)
    large = _capture_calls(monkeypatch, copy, "deepcopy", 12)
    assert large == small, f"4 ops: {small} deepcopy calls; 24 ops: {large}"
