"""Deterministic counters for the label-rename sweep (postprocess step 9).

* The sweep reads each dataclass's field list once per class for the whole
  trace: ``dataclasses.fields`` calls do not grow with the op count. It used
  to call ``fields`` (plus ``is_dataclass`` and ``hasattr`` per field) on
  every dataclass node of every op's replay templates.
* A ``ParentRef`` leaf is rebuilt with its constructor, not through
  ``dataclasses.replace`` (which re-reads the field list per leaf).
* The removal scrub (orphan pruning) likewise reads each dataclass's field
  list once per class per sweep, not once per surviving op's template.

``tests/test_label_rename_reference.py`` pins that the faster walks return
exactly what the plain ``isinstance`` ladders returned.
"""

from __future__ import annotations

import sys
from dataclasses import fields, replace
from types import FrameType
from typing import Any

import pytest
import torch
from _hermetic_counters import quiet_gc

import torchlens as tl
from torchlens._data_classes import cleanup
from torchlens._data_classes._template_walk import map_template_leaves
from torchlens._postprocess import labeling
from torchlens.intervention.types import ParentRef


class _Chain(torch.nn.Module):
    """``depth`` Linear/ReLU/add-constant triples: every op carries a replay template."""

    def __init__(self, depth: int) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.layers = torch.nn.ModuleList(torch.nn.Linear(4, 4) for _ in range(depth))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x)) + 0.5
        return x


class _OrphanChain(_Chain):
    """``_Chain`` plus a disconnected op chain, so orphan pruning scrubs every template."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        unused = torch.zeros(2) + 1  # reaches neither input nor output: pruned as orphans
        del unused
        return super().forward(x)


def _first_arg(frame: FrameType) -> Any:
    code = frame.f_code
    return frame.f_locals.get(code.co_varnames[0]) if code.co_argcount else None


def _count_in_step(
    monkeypatch: pytest.MonkeyPatch,
    depth: int,
    owner: Any = labeling,
    step: str = "_log_final_info_for_layers",
    model: type[_Chain] = _Chain,
) -> tuple[tl.Trace, dict[str, int], list[type]]:
    """Trace ``model(depth)``; count Python calls of interest inside ``owner.step`` only."""

    counts = {"fields": 0, "replace": 0}
    replaced: list[type] = []
    targets = {fields.__code__: "fields", replace.__code__: "replace"}
    real = getattr(owner, step)

    def profiler(frame: FrameType, event: str, _arg: Any) -> None:
        if event == "call":
            name = targets.get(frame.f_code)
            if name is not None:
                counts[name] += 1
                if name == "replace":
                    replaced.append(type(_first_arg(frame)))

    def counted(*args: Any) -> None:
        previous = sys.getprofile()
        sys.setprofile(profiler)
        try:
            real(*args)
        finally:
            sys.setprofile(previous)

    monkeypatch.setattr(owner, step, counted)
    with quiet_gc():
        trace = tl.trace(model(depth).eval(), torch.randn(2, 4))
    return trace, counts, replaced


def _parent_refs(value: Any) -> list[ParentRef]:
    found: list[ParentRef] = []

    def collect(leaf: Any) -> Any:
        if isinstance(leaf, ParentRef):
            found.append(leaf)
        return leaf

    map_template_leaves(value, collect)
    return found


def test_rename_sweep_reads_dataclass_fields_once_per_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Counter: ``dataclasses.fields`` calls in step 9 do not grow with the op count."""

    small, small_counts, _ = _count_in_step(monkeypatch, 3)
    large, large_counts, _ = _count_in_step(monkeypatch, 12)
    assert len(large.ops) >= len(small.ops) + 27
    assert small_counts["fields"] > 0, "the sweep must walk dataclass template nodes"
    assert large_counts["fields"] == small_counts["fields"]


def test_rename_builds_parent_refs_without_dataclasses_replace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Counter: no ``replace`` call on a ``ParentRef`` in step 9; refs carry final labels."""

    trace, counts, replaced = _count_in_step(monkeypatch, 4)
    assert counts["replace"] > 0, "the sweep must rebuild dataclass template nodes"
    assert ParentRef not in replaced
    final_labels = {op.label for op in trace.ops} | {op.layer_label for op in trace.ops}
    refs = [ref for op in trace.ops for ref in _parent_refs(op.args_template)]
    assert len(refs) >= len(trace.ops) // 2
    assert all(ref.parent_label in final_labels for ref in refs)


def test_removal_scrub_reads_dataclass_fields_once_per_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Counter: ``dataclasses.fields`` calls in the orphan-removal scrub do not grow with ops."""

    step = "_scrub_intervention_fields_after_removal"
    small, small_counts, _ = _count_in_step(monkeypatch, 3, cleanup, step, _OrphanChain)
    large, large_counts, _ = _count_in_step(monkeypatch, 12, cleanup, step, _OrphanChain)
    assert len(large.ops) >= len(small.ops) + 27
    assert small_counts["fields"] > 0, "the scrub must walk dataclass template nodes"
    assert large_counts["fields"] == small_counts["fields"]
