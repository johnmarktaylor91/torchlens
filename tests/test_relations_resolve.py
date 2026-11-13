"""Ancestry resolved from the recorded parent records equals the call-time build.

An op with parents no longer builds ``input_ancestors``, ``internal_source_ancestors``,
``root_ancestors`` and ``has_internal_source_ancestor`` at the call: it records the
parent records the builder read and one bit, ``has_input_ancestor``, and the sets
resolve on the first read (``backends/torch/_relations_resolve.py``). These tests
run a shadow of the former builder beside the capture: at every deferral it unions
the parents' call-time sets (a deferred parent's from the shadow, any other
parent's as it answers then) exactly as the builder did. They assert

* the resolved sets of every deferred op equal the shadow's call-time sets;
* the bit equals ``bool(input_ancestors)`` for every live event at every commit
  (inputs, buffers, internal sources, in-place writes, module outputs included);
* a live-view read during the forward returns the call-time sets;
* the resolver run after the forward on a frozen copy of the recorded inputs
  returns the same sets.

The subjects are the field differential's small models and the plain-emit
soundness subjects, plus out-destination, internal-source, module-output and
layout-read cases.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from _forward_stash import stashed
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens._capture.projections import LiveOpView
from torchlens.backends.torch import (
    _ops_emission,
    _ops_predicate_events,
    _ops_row_journal,
    _relations_resolve as rr,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
from field_diff_lib.cases import SEED, build_model  # noqa: E402

_ANCESTRY_READS = (
    "root_ancestors",
    "has_input_ancestor",
    "input_ancestors",
    "has_internal_source_ancestor",
    "internal_source_ancestors",
)


class _Shadow:
    """The former call-time builder, run beside the capture, plus the bit sweep."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, sweep: bool = True) -> None:
        self.sweep = sweep
        self.by_parents: dict[int, tuple[frozenset[str], frozenset[str]]] = {}
        self.kept: list[Any] = []
        self.pending: list[tuple[Any, frozenset[str], frozenset[str], dict[str, Any]]] = []
        self.by_label: dict[str, tuple[frozenset[str], frozenset[str]]] = {}
        self.bits: list[tuple[Any, bool]] = []
        self.deferrals = 0
        self.witness_reads = 0
        self._has_input = rr.has_input_ancestor
        monkeypatch.setattr(_ops_predicate_events, "defer_ancestry", self._defer)
        monkeypatch.setattr(_ops_row_journal, "_fill_items", self._fill_items)
        monkeypatch.setattr(_ops_emission, "has_input_ancestor", self._witness_bit)

    def _parent_sets(self, record: Any) -> tuple[frozenset[str], frozenset[str]]:
        fields = rr._deferred_fields(record)
        if fields is not None:
            return self.by_parents[id(fields[rr.ANCESTRY_PARENTS])]
        return frozenset(record.input_ancestors), frozenset(record.internal_source_ancestors)

    def _defer(self, fields: dict[str, Any], parent_records: list[Any]) -> None:
        inputs: set[str] = set()
        internal: set[str] = set()
        for record in parent_records:
            parent_inputs, parent_internal = self._parent_sets(record)
            inputs.update(parent_inputs)
            internal.update(parent_internal)
        rr.defer_ancestry(fields, parent_records)
        key = fields[rr.ANCESTRY_PARENTS]
        self.kept.append(key)
        self.by_parents[id(key)] = (frozenset(inputs), frozenset(internal))
        self.deferrals += 1

    def _fill_items(self, pending: Any, fields: dict[str, Any]) -> None:
        _FILL_ITEMS(pending, fields)
        if rr.is_deferred(fields):
            inputs, internal = self.by_parents[id(fields[rr.ANCESTRY_PARENTS])]
            self.pending.append((pending, inputs, internal, dict(fields)))
            self.by_label[fields["_label_raw"]] = (inputs, internal)
        if not self.sweep:
            return
        trace = pending.trace_ref()
        for event in list(trace.capture_events.live_index.by_raw_label.values()):
            self.bits.append((event, self._has_input(event)))

    def _witness_bit(self, event: Any) -> bool:
        self.witness_reads += 1
        return self._has_input(event)

    def check(self) -> None:
        """Assert resolved sets, bits and a post-forward re-resolve match the call."""

        assert self.deferrals > 0
        for pending, inputs, internal, frozen in self.pending:
            pending.resolve_ancestry()
            fields = pending.fields
            assert not rr.is_deferred(fields)
            assert set(fields["input_ancestors"]) == inputs
            assert set(fields["internal_source_ancestors"]) == internal
            assert set(fields["root_ancestors"]) == inputs | internal
            assert fields["has_internal_source_ancestor"] is bool(internal)
            assert fields[rr.INPUT_ROOTED] is bool(inputs)
            assert set(pending["input_ancestors"]) == inputs
            again = dict(frozen)
            rr.resolve_fields(again)
            assert set(again["input_ancestors"]) == inputs
            assert set(again["internal_source_ancestors"]) == internal
            assert set(again["root_ancestors"]) == inputs | internal
            assert again["has_internal_source_ancestor"] is bool(internal)
        for event, bit in self.bits:
            assert bit is bool(event.input_ancestors), event.label_raw


_FILL_ITEMS = _ops_row_journal._fill_items


def _trace(build: Callable[[], nn.Module], x: Any, **kwargs: Any) -> Any:
    torch.manual_seed(SEED)
    model = build()
    capture = tl.options.CaptureOptions(seed=SEED)
    return tl.trace(model, x, capture=capture, **kwargs)


_DIFFERENTIAL = [
    "chain40",
    "varied8",
    "loopcell12",
    "conditional",
    "inplace_buffers",
    "sparse",
    "autocast",
    "transform",
    "two_relu",
    "dead_branches",
    "resnet18",
    "gpt2_tiny4",
]


@pytest.mark.parametrize("name", _DIFFERENTIAL)
def test_differential_models_resolve_to_the_call_time_sets(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    shadow = _Shadow(monkeypatch)
    model, x = build_model(name)
    tl.trace(model, x, capture=tl.options.CaptureOptions(seed=SEED))
    shadow.check()


class _Body(nn.Module):
    """A module with a Parameter, a buffer and a linear layer, whose forward is ``body``."""

    def __init__(self, body: Callable[[nn.Module, torch.Tensor], Any]) -> None:
        super().__init__()
        self.body = body
        self.weight = nn.Parameter(torch.ones(3))
        self.register_buffer("running", torch.zeros(3))
        self.lin = nn.Linear(3, 3)
        self.ident = nn.Identity()

    def forward(self, x: torch.Tensor) -> Any:
        return self.body(self, x)


def _x() -> torch.Tensor:
    return torch.arange(6.0).reshape(2, 3) - 2.0


def _inplace(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
    y = m.lin(x) * 2
    y.relu_()
    y.add_(1)
    y += x
    return y * m.weight


def _views(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
    y = x + 1
    v = y.view(-1)
    v.mul_(2)
    return y.t().sum() + v[1:4].sum() + y.detach().mean()


def _mutated(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
    a = x * 2
    b = a + 1
    a.add_(5)
    c = a * b
    a.zero_()
    return c + a


def _sources(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Internal sources (parentless ops), a buffer written in place, a module output."""

    k = torch.ones(3) * 2
    z = torch.zeros(2, 3)
    m.running.add_(x.mean(0))
    same = m.ident(x)
    return (same + z) * k + m.running + m.ident(m.running)


def _out_destination(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """``out=`` destinations are parents of the op that writes them."""

    dest = torch.empty(2, 3)
    torch.add(x, m.weight.detach(), out=dest)
    values = torch.empty(2, 3)
    indices = torch.empty(2, 3, dtype=torch.long)
    torch.sort(dest, dim=1, out=(values, indices))
    return values + indices.float()


def _layout_reads(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Layout reads of fresh activations reach the input-layout observer."""

    y = (x * 2).t()
    stride = y.stride()
    contiguous = y.is_contiguous()
    k = torch.ones(3, 2)
    k_stride = k.stride()
    return y + float(stride[0] + int(contiguous) + k_stride[0])


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner = nn.Sequential(nn.Linear(3, 3), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x) + x


class _Nested(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = _Block()
        self.other = _Block()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.other(self.block(self.block(x)))


class _Loop(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.zeros_like(x)
        for _ in range(6):
            h = torch.tanh(self.cell(h) + x)
        return h


class _Buffers(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("running", torch.zeros(3))
        self.norm = nn.BatchNorm1d(3)
        self.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.running.add_(x.mean(0))
        return self.norm(x) + self.running


_SUBJECTS: dict[str, Callable[[], nn.Module]] = {
    "inplace": lambda: _Body(_inplace),
    "views": lambda: _Body(_views),
    "mutated": lambda: _Body(_mutated),
    "sources": lambda: _Body(_sources),
    "out_destination": lambda: _Body(_out_destination),
    "layout_reads": lambda: _Body(_layout_reads),
    "nested": _Nested,
    "loop": _Loop,
    "buffers": _Buffers,
}


@pytest.mark.parametrize("name", sorted(_SUBJECTS))
def test_soundness_subjects_resolve_to_the_call_time_sets(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    shadow = _Shadow(monkeypatch)
    _trace(_SUBJECTS[name], _x())
    shadow.check()


def test_the_witness_reads_the_bit(monkeypatch: pytest.MonkeyPatch) -> None:
    """An in-place write into a data alias with an input-derived operand asks the bit."""

    def body(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
        y = x * 2
        y.data.add_(x)
        z = x + 1
        alias = z.data
        alias.mul_(y)
        return y + z

    shadow = _Shadow(monkeypatch)
    _trace(lambda: _Body(body), _x())
    shadow.check()
    assert shadow.witness_reads > 0


class _MidForwardAncestry(nn.Module):
    """Reads the ancestry of every live event through a live view, mid-forward."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 3)
        self.register_buffer("running", torch.ones(3))
        self.read: dict[str, tuple[Any, ...]] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = torch.ones(3)
        y = torch.relu(self.lin(x) + k) * self.running
        z = y.tanh() + x
        trace = _state._active_trace
        if trace is not None and getattr(trace, "capture_events", None) is not None:
            index = trace.capture_events.live_index
            for label in reversed(list(index.by_raw_label)):
                view = LiveOpView(trace, index.require_event(label))
                self.read[label] = tuple(getattr(view, name) for name in _ANCESTRY_READS)
        return z * 2 + y


def test_live_view_reads_mid_forward_return_the_call_time_sets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shadow = _Shadow(monkeypatch)
    trace = _trace(_MidForwardAncestry, _x())
    shadow.check()
    read = stashed(trace, "read")
    deferred = [label for label in read if label in shadow.by_label]
    assert deferred
    for label in deferred:
        inputs, internal = shadow.by_label[label]
        roots, has_input, got_inputs, has_internal, got_internal = read[label]
        assert roots == set(inputs | internal), label
        assert has_input is bool(inputs), label
        assert got_inputs == set(inputs), label
        assert has_internal is bool(internal), label
        assert got_internal == set(internal), label


def test_a_long_chain_resolves_without_recursion(monkeypatch: pytest.MonkeyPatch) -> None:
    """The last op of a long chain read first resolves every ancestor in a loop."""

    class _Chain(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for _ in range(3000):
                x = x + 1
            trace = _state._active_trace
            if trace is not None and getattr(trace, "capture_events", None) is not None:
                index = trace.capture_events.live_index
                label = next(reversed(index.by_raw_label))
                self.last = set(index.require_event(label).input_ancestors)
            return x

    shadow = _Shadow(monkeypatch, sweep=False)
    trace = _trace(_Chain, torch.zeros(2))
    shadow.check()
    assert len(stashed(trace, "last")) == 1
