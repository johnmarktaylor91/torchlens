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
from torchlens._utils._record_group_table import record_member
from torchlens.intervention._types import ParentRef, Unsupported


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


class _BranchedRepeatedSites(_RepeatedSites):
    """The same repeated call sites inside a data-dependent ``if`` arm, so step 5 attributes."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > -1e9:
            x = self.blocks(x)
        return x


def _frame_position_key(code_context: Any) -> tuple[tuple[Any, ...], ...]:
    """The six frame fields branch attribution reads, per frame."""

    return tuple(
        (
            frame.file,
            frame.code_firstlineno,
            frame.func_name,
            frame.func_qualname,
            frame.line_number,
            frame.col_offset,
        )
        for frame in code_context
    )


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
        trace = tl.trace(_BranchedRepeatedSites().eval(), torch.randn(2, 4))
    attributed = [_frame_position_key(args[0]) for args in calls]
    assert len(trace.ops) >= 12
    assert any(op.control_flow.branch_stack for op in trace.ops), "step 5 must attribute"
    assert len(attributed) == len(set(attributed)), "a stack was attributed twice"
    assert 0 < len(attributed) < len(trace.ops) // 2


def test_branch_stacks_are_private_per_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """Memoized stacks are copied: right after attribution no two ops share one list."""

    seen: dict[str, list[int]] = {"ids": [], "nonempty": []}
    real = control_flow._attribute_branches_forward

    def recording(self: Any, events_by_key: Any, plan: Any = None) -> None:
        real(self, events_by_key, plan)
        for label in self._raw_graph_ws.raw_layer_labels_list:
            layer = self[label]
            if record_member(layer, "is_orphan", False):
                continue
            stack = layer.control_flow.branch_stack
            seen["ids"].append(id(stack))
            seen["nonempty"].append(len(stack))

    monkeypatch.setattr(control_flow, "_attribute_branches_forward", recording)
    tl.trace(_BranchedRepeatedSites().eval(), torch.randn(2, 4))
    assert len(seen["ids"]) >= 12
    assert sum(1 for depth in seen["nonempty"] if depth) >= 12
    assert len(set(seen["ids"])) == len(seen["ids"])


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


class _Entry(SimpleNamespace):
    """Stand-in Op: the call-group templates live under their storage attributes."""

    @property
    def args_template(self) -> Any:
        return self._call_args_template

    @property
    def kwargs_template(self) -> Any:
        return self._call_kwargs_template


def _entry(args_template: Any, kwargs_template: Any) -> SimpleNamespace:
    return _Entry(_call_args_template=args_template, _call_kwargs_template=kwargs_template)


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
