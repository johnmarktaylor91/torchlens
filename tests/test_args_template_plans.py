"""Per-capture argument template plans and the validated-parameter memo change nothing.

A replay argument template is built from a plan kept per (callable, argument layout)
for the capture, and a parameter the session validated through its parameter log is
remembered for the rest of the capture. Each subject is traced with and without them
(the plan lookup answering ``None`` sends every op down the general classifier; the
memo is hidden behind a workspace view with an empty dict per read); the templates,
parents, argument positions and parameter edges must be equal op for op. The subjects
cover a repeated tensor-plus-int chain, linear layers with parameter literals, the
same callable at two argument layouts, keyword arguments, nested lists, non-contiguous
parents (the layout record) and an in-place parameter mutation mid-forward. Parent
argument positions read from the capture-time parent events must match the reads
through the parent op views.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.backends.torch import (
    _args_template_cache,
    _ops_arguments,
    _ops_predicate_events,
)
from torchlens.intervention._types import CapturedArgTemplate, LiteralTensor, ParentRef


class _Chain(nn.Module):
    """A tensor-plus-int chain: one callable at one layout, many times."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add one twenty times."""
        for _ in range(20):
            x = x + 1
        return x


class _Varied(nn.Module):
    """Linear and tanh pairs: parameter literals at every linear."""

    def __init__(self) -> None:
        """Build four linear layers."""
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(6, 6) for _ in range(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply each layer then tanh, twice over (parameters reused)."""
        for _ in range(2):
            for layer in self.layers:
                x = torch.tanh(layer(x))
        return x


class _Layouts(nn.Module):
    """One callable at several layouts, keyword args, nested lists, strided parents."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix tensor, scalar and keyword forms of the same ops."""
        a = torch.add(x, 2)
        b = torch.add(x, a)
        c = torch.add(x, a, alpha=0.5)
        t = b.t()
        d = t * 3
        e = t + c.t()
        f = torch.cat([d, e, t], dim=0)
        g = torch.cat((d, e), dim=1)
        h = f[..., :2] + 0.25
        return h.sum() + g.sum() + torch.where(a > 0, a, other=torch.zeros_like(a)).sum()


class _ParamMut(nn.Module):
    """A parameter multiplied in place mid-forward, then read again."""

    def __init__(self) -> None:
        """Build one linear layer."""
        super().__init__()
        self.lin = nn.Linear(5, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Use, mutate, reuse the weight."""
        h = self.lin(x)
        with torch.no_grad():
            self.lin.weight.mul_(0.5)
        return self.lin(h) + torch.nn.functional.linear(h, self.lin.weight)


_SUBJECTS: tuple[tuple[str, type[nn.Module], tuple[int, ...]], ...] = (
    ("chain", _Chain, (3, 4)),
    ("varied", _Varied, (2, 6)),
    ("layouts", _Layouts, (4, 4)),
    ("param_mutation", _ParamMut, (2, 5)),
)


def _norm_component(component: Any) -> Any:
    """Return a comparable form of one template component (tensors by value)."""

    if isinstance(component, ParentRef):
        layout = component._capture_layout
        return ("parent", component.parent_label, None if layout is None else repr(layout))
    if isinstance(component, LiteralTensor):
        value = component.value
        return (
            "tensor",
            tuple(value.shape),
            str(value.dtype),
            value.detach().flatten().tolist(),
            component.param_barcode is None,
            component._capture_param_fingerprint is None,
        )
    if isinstance(component, tuple):
        return tuple(_norm_component(item) for item in component)
    return (type(component).__name__, repr(component))


def _norm_template(template: Any) -> Any:
    """Return a comparable form of a captured argument template (or ``None``)."""

    if template is None:
        return None
    assert isinstance(template, CapturedArgTemplate)
    return (
        _norm_component(template.args),
        tuple((key, _norm_component(value)) for key, value in template.kwargs),
        template.notes,
        repr(template.func_id),
    )


def _fields(model_cls: type[nn.Module], shape: tuple[int, ...]) -> list[tuple[Any, ...]]:
    """Trace ``model_cls`` (fixed seed) and return each op's compared fields."""

    torch.manual_seed(0)
    model = model_cls()
    x = torch.randn(*shape)
    trace = tl.trace(model, x)
    return [
        (
            index,
            list(op.parents),
            {kind: dict(op.call.parent_arg_positions[kind]) for kind in ("args", "kwargs")},
            tuple(op.call.unattributed_tensor_args),
            _norm_template(op.call.args_template),
            _norm_template(op.call.kwargs_template),
            sorted(op.graph.parent_param_ops.values()),
            [tuple(param.shape) for param in op.graph.parent_params],
        )
        for index, op in enumerate(trace.ops)
    ]


class _MemoFreeWorkspace:
    """A raw-graph workspace view whose validated-parameter memo is always empty."""

    def __init__(self, workspace: Any) -> None:
        """Wrap ``workspace``."""
        self._workspace = workspace

    def __getattr__(self, name: str) -> Any:
        """Forward every read except the memo, which is a fresh dict."""
        if name == "_validated_params":
            return {}
        return getattr(self._workspace, name)


def _assert_same(name: str, left: list[tuple[Any, ...]], right: list[tuple[Any, ...]]) -> None:
    """Assert two per-op field lists are equal, naming the first differing fields."""

    assert len(left) == len(right), name
    diffs = [
        (a[0], index)
        for a, b in zip(left, right, strict=True)
        for index in range(1, len(a))
        if a[index] != b[index]
    ]
    assert diffs == [], f"{name}: fields differ (op, field index): {diffs[:6]}"


@pytest.mark.parametrize(("name", "model_cls", "shape"), _SUBJECTS, ids=[s[0] for s in _SUBJECTS])
def test_template_plans_equal_general_classifier(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    model_cls: type[nn.Module],
    shape: tuple[int, ...],
) -> None:
    """Planned templates match the general per-argument classifier op for op."""

    planned = _fields(model_cls, shape)
    with monkeypatch.context() as patch:
        patch.setattr(_ops_arguments, "planned_args_template", lambda *_args: None)
        # The capture observes templates through the plan lookup directly.
        patch.setattr(_args_template_cache, "_plan_of", lambda *_args: None)
        general = _fields(model_cls, shape)
    _assert_same(name, general, planned)


@pytest.mark.parametrize(("name", "model_cls", "shape"), _SUBJECTS, ids=[s[0] for s in _SUBJECTS])
def test_base_layout_reads_equal_wrapped_reads(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    model_cls: type[nn.Module],
    shape: tuple[int, ...],
) -> None:
    """The base-class layout pre-read gives the same layout records as the full read."""

    fast = _fields(model_cls, shape)
    with monkeypatch.context() as patch:
        patch.setattr(_args_template_cache, "_BASE_READS", None)
        full = _fields(model_cls, shape)
    _assert_same(name, full, fast)


@pytest.mark.parametrize(("name", "model_cls", "shape"), _SUBJECTS, ids=[s[0] for s in _SUBJECTS])
def test_validated_parameter_memo_changes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    model_cls: type[nn.Module],
    shape: tuple[int, ...],
) -> None:
    """Remembering validated parameters for the capture leaves every field unchanged."""

    remembered = _fields(model_cls, shape)
    real_peek = _ops_arguments._peek_raw_graph_workspace

    def memo_free(trace: Any) -> Any:
        """Return the trace's workspace behind a view with no memo."""
        workspace = real_peek(trace)
        return None if workspace is None else _MemoFreeWorkspace(workspace)

    with monkeypatch.context() as patch:
        patch.setattr(_ops_arguments, "_peek_raw_graph_workspace", memo_free)
        unremembered = _fields(model_cls, shape)
    _assert_same(name, unremembered, remembered)


@pytest.mark.parametrize(("name", "model_cls", "shape"), _SUBJECTS, ids=[s[0] for s in _SUBJECTS])
def test_parent_event_reads_equal_view_reads(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    model_cls: type[nn.Module],
    shape: tuple[int, ...],
) -> None:
    """Reading parent events directly gives the same positions and ancestors as the views."""

    direct = _fields(model_cls, shape)
    with monkeypatch.context() as patch:
        patch.setattr(_ops_predicate_events, "current_arg_slots", lambda *_args: None)
        through_views = _fields(model_cls, shape)
    _assert_same(name, through_views, direct)


def _has_layout(component: Any) -> bool:
    """Whether a normalized component holds a parent with a recorded layout."""

    if isinstance(component, tuple):
        if len(component) == 3 and component[0] == "parent":
            return component[2] is not None
        return any(_has_layout(item) for item in component)
    return False


def test_layouts_subject_records_a_strided_parent() -> None:
    """The layouts subject exercises the strided layout record, not only ``None``."""

    fields = _fields(_Layouts, (4, 4))
    assert any(_has_layout(row[4]) for row in fields)


def test_validated_parameter_memo_fills_once_per_parameter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each parameter takes the parameter-log rung once per capture; reuses hit the memo."""

    rungs: list[int] = []
    real_meta = _ops_arguments.get_param_meta

    def counting_meta(value: Any) -> Any:
        """Count one parameter-log validation (a meta read from the validator)."""
        if sys._getframe(1).f_code.co_name == "_session_validated_parameter":
            rungs.append(id(value))
        return real_meta(value)

    monkeypatch.setattr(_ops_arguments, "get_param_meta", counting_meta)
    torch.manual_seed(0)
    model = _Varied()
    tl.trace(model, torch.randn(2, 6))
    params = {id(param) for param in model.parameters()}
    validated = [rung for rung in rungs if rung in params]
    assert validated, "no parameter was validated through the parameter log"
    assert len(validated) == len(set(validated))
