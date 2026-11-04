"""Deterministic counters for two postprocess passes that used to repeat work.

* Branch attribution (step 5) attributes each distinct frame-position stack
  once per pass instead of once per op: ops from one call site share a stack.
* The removal scrub rebuilds an op's replay template only when the template
  names a removed parent; it used to rebuild every surviving op's templates on
  every removal (orphan pruning removes a handful of ops from thousands).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from _hermetic_counters import quiet_gc

import torchlens as tl
from torchlens._data_classes import cleanup
from torchlens._postprocess import control_flow
from torchlens.intervention.types import ParentRef, Unsupported


class _RepeatedSites(torch.nn.Module):
    """Six Linear/ReLU pairs (two call sites) plus a disconnected orphan chain."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.blocks = torch.nn.Sequential(
            *[layer for _ in range(6) for layer in (torch.nn.Linear(4, 4), torch.nn.ReLU())]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        unused = torch.zeros(2) + 1  # reaches neither input nor output: pruned as orphans
        del unused
        return self.blocks(x)


def _count_calls(monkeypatch: pytest.MonkeyPatch, module: Any, name: str) -> list[Any]:
    calls: list[Any] = []
    real = getattr(module, name)

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, counting)
    return calls


def test_branch_attribution_runs_once_per_distinct_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    """Counter: one attribution per distinct frame-position stack, not per op."""

    calls = _count_calls(monkeypatch, control_flow, "_translate_conditional_stack")
    with quiet_gc():
        trace = tl.trace(_RepeatedSites().eval(), torch.randn(2, 4))
    distinct = {control_flow._code_context_position_key(op.code_context) for op in trace.ops}
    assert len(trace.ops) >= 12
    assert len(calls) == len(distinct)
    assert len(calls) < len(trace.ops)


def test_branch_stacks_are_private_per_op() -> None:
    """Memoized stacks are copied: no two ops share one list object."""

    trace = tl.trace(_RepeatedSites().eval(), torch.randn(2, 4))
    stacks = [op.conditional_branch_stack for op in trace.ops]
    assert len({id(stack) for stack in stacks}) == len(stacks)


def test_orphan_removal_rebuilds_no_untouched_template(monkeypatch: pytest.MonkeyPatch) -> None:
    """Counter: pruning orphans rebuilds zero templates that name no removed op."""

    scrubs = _count_calls(monkeypatch, cleanup, "_scrub_intervention_fields_after_removal")
    rebuilds = _count_calls(monkeypatch, cleanup, "_replace_removed_parent_refs")
    with quiet_gc():
        trace = tl.trace(_RepeatedSites().eval(), torch.randn(2, 4))
    removed = set().union(*(set(args[1]) for args in scrubs)) if scrubs else set()
    assert removed, "the model must exercise orphan removal"
    assert len(trace.ops) >= 12
    assert rebuilds == []


@dataclass(frozen=True)
class _Wrapper:
    inner: Any
    note: str = "n"


def _entry(args_template: Any, kwargs_template: Any) -> SimpleNamespace:
    return SimpleNamespace(args_template=args_template, kwargs_template=kwargs_template)


def test_scrub_still_replaces_removed_parent_refs_at_any_depth() -> None:
    """A template naming a removed parent is rebuilt with the stale ref replaced."""

    removed = ParentRef(parent_label="gone_1")
    kept = ParentRef(parent_label="kept_1")
    args = ([kept, (removed,)], 3)
    kwargs = {"a": _Wrapper(inner={"deep": removed}), "b": kept}
    entry = _entry(args, kwargs)
    owner = SimpleNamespace(state_history=[])
    cleanup._scrub_intervention_fields_after_removal(owner, {"gone_1"}, [entry])
    stale = Unsupported(reason="removed_parent_ref", value_type="ParentRef")
    assert entry.args_template == ([kept, (stale,)], 3)
    assert entry.kwargs_template == {"a": _Wrapper(inner={"deep": stale}), "b": kept}


def test_scrub_keeps_untouched_template_objects() -> None:
    """A template naming no removed parent keeps its object."""

    args = ([ParentRef(parent_label="kept_1")], {"x": 1})
    kwargs = {"w": _Wrapper(inner=[ParentRef(parent_label="kept_2")])}
    entry = _entry(args, kwargs)
    owner = SimpleNamespace(state_history=[])
    cleanup._scrub_intervention_fields_after_removal(owner, {"gone_1"}, [entry])
    assert entry.args_template is args
    assert entry.kwargs_template is kwargs
