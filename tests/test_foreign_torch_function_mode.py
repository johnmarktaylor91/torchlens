"""Capture must stay green while a foreign ``TorchFunctionMode`` is active.

Stage-0 safety-net fix: TorchLens used to DECORATE
``torch.nn.functional.handle_torch_function`` (it sits in torch's
``get_ignored_functions()`` list, which the decoration inventory re-adds).
``handle_torch_function`` is ``__torch_function__`` protocol plumbing, not a
tensor op: with any foreign ``TorchFunctionMode`` active (torch's own
``DeviceContext`` is one), every composite's
``if has_torch_function(...): return handle_torch_function(op, ...)``
preamble went live and routed through the TorchLens wrapper, which then fed
the FUNCTION first argument into tensor-arg capture and crashed with
``TorchLensTLCollisionError`` (observed: ``Expected TensorMeta on function,
found DecorationTag``). ``Tensor.__torch_function__`` carried the same
declared intent (it was never actually decorated — classmethod access binds
to ``method``, which the decoration type gate skips) and was removed from the
inventory alongside.

Repro provenance: BatchNorm + passthrough TFM crashed on base ``890a3cb5``
in BOTH nesting orders (mode armed around the trace call, and mode entered
inside ``forward``); green after un-decorating. The passthrough mode below is
observationally inert (it delegates unchanged), so the recorded op vocabulary
must ALSO match the modeless capture exactly — a silent-degradation guard on
top of the crash guard. (Models with mode-sensitive fused fast paths — eval
``MultiheadAttention`` — legitimately diverge under an armed mode; that
residual is out of scope here and covered by the safety-net corpus.)
"""

from __future__ import annotations

import torch
from torch import nn
from torch.overrides import TorchFunctionMode

import torchlens as tl


class _PassthroughMode(TorchFunctionMode):
    """Foreign user mode that observes every call and delegates unchanged."""

    def __init__(self) -> None:
        super().__init__()
        self.calls_seen = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.calls_seen += 1
        return func(*args, **(kwargs or {}))


class _BatchNormModel(nn.Module):
    """Minimal model whose composite preamble triggered the collision."""

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


class _ModeInsideForwardModel(nn.Module):
    """Enters the foreign mode INSIDE forward (the other nesting order)."""

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with _PassthroughMode():
            return self.bn(x)


class _MixedOpsModel(nn.Module):
    """Composite + leaf + factory mix to widen the preamble surface."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.linear(x))
        y = torch.nn.functional.softsign(y)
        return y + torch.ones_like(y)


def _op_names(trace: tl.Trace) -> list[str]:
    return [op.func_name for op in trace.ops]


def test_handle_torch_function_is_not_decorated() -> None:
    """The protocol plumbing entry must never be wrapped or tag-carrying."""
    from torchlens.backends.torch._tl import is_decorated_function
    from torchlens.backends.torch.wrappers import wrap_torch

    wrap_torch()
    assert not is_decorated_function(torch.nn.functional.handle_torch_function)
    assert not is_decorated_function(torch.overrides.handle_torch_function)
    inner = torch.Tensor.__torch_function__
    assert not is_decorated_function(getattr(inner, "__func__", inner))


def test_batchnorm_capture_green_under_foreign_mode_outside() -> None:
    """Nesting order A: foreign mode armed around the whole trace call."""
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    baseline = tl.trace(_BatchNormModel(), x)

    mode = _PassthroughMode()
    with mode:
        traced = tl.trace(_BatchNormModel(), x)

    assert mode.calls_seen > 0, "passthrough mode never engaged; test is vacuous"
    assert _op_names(traced) == _op_names(baseline)


def test_batchnorm_capture_green_under_foreign_mode_inside_forward() -> None:
    """Nesting order B: foreign mode entered inside the model's forward."""
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    baseline = tl.trace(_BatchNormModel(), x)

    traced = tl.trace(_ModeInsideForwardModel(), x)

    assert _op_names(traced) == _op_names(baseline)


def test_mixed_ops_capture_green_and_output_exact_under_foreign_mode() -> None:
    """Composite/leaf/factory mix: green, byte-equal output, pinned vocabulary.

    One NARROW, documented delta is expected (not masked): with a foreign mode
    active, TOP-LEVEL operator dunders route through the ``__torch_function__``
    protocol and arrive respelled — the user-level ``y + ones_like(y)`` records
    as ``add`` instead of ``__add__``. The ``__add__`` INSIDE the ``softsign``
    composite keeps its dunder spelling (the protocol pops the mode before the
    composite body runs, so interiors execute de-moded). Both facts are pinned
    positionally (exact single-site substitution, never a whole-column mask)
    so any WIDENING of the drift fails loudly.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    model = _MixedOpsModel()
    baseline = tl.trace(model, x)

    with _PassthroughMode():
        traced = tl.trace(model, x)

    base_names = _op_names(baseline)
    add_sites = [i for i, name in enumerate(base_names) if name == "__add__"]
    assert len(add_sites) == 2, "expected softsign-interior + top-level operator sites"
    expected = list(base_names)
    expected[add_sites[-1]] = "add"  # the top-level ``+`` is the moded call site
    assert _op_names(traced) == expected
    assert torch.equal(traced.output_ops[0].out, baseline.output_ops[0].out)
