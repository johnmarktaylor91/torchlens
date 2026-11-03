"""TorchScript call boundary: a ``torch.jit.ScriptFunction`` called from an eager forward.

Positive: a scripted helper (a module-level function, a scripted static method reached
through ``self``, one returning a tuple, one called inside ``autograd.Function.forward``)
is ONE ``jitscript`` boundary op that validates through replay, with the armed census
clean. ``PYTORCH_JIT=0`` (``torch.jit._state.disable()``, the proxy the variable sets)
gives the op-by-op graph. A scripted module is still refused. Armed negatives: a scripted
call that writes into its argument stays the census gap, and an opaque call that is not a
scripted function (a scripted module held outside ``named_modules()``) still fails completeness.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import CaptureGapWarning
from torchlens._record_type import record_type
from torchlens._utils._torch_compat import HAS_SCRIPT_FUNCTION_CALL_HOOK
from torchlens.backends.torch._script_boundary import (
    SCRIPT_FUNCTION_FUNC_NAME,
    SCRIPT_FUNCTION_TRANSFORM_KIND,
    is_script_function_recorder,
)
from torchlens.errors import TorchLensWarning
from torchlens.validation import last_validation_failure, validate_forward_pass

pytestmark = pytest.mark.skipif(
    not HAS_SCRIPT_FUNCTION_CALL_HOOK, reason="torch build without a ScriptFunction call hook"
)


@torch.jit.script
def scripted_gate(x: torch.Tensor, threshold: float) -> torch.Tensor:
    """DeBERTa-style module-level helper: its body runs in the TorchScript interpreter."""

    return (x >= threshold).to(x.dtype) * x.sin()


@torch.jit.script
def scripted_heaviside(x: torch.Tensor) -> torch.Tensor:
    """spikingjelly ``heaviside``: called from a surrogate Function's forward."""

    return (x >= 0).to(x)


@torch.jit.script
def scripted_write(v: torch.Tensor) -> torch.Tensor:
    """Writes into its argument in place: a side effect the boundary cannot attest."""

    v.add_(1.0)
    return v * 2


class _GateModel(nn.Module):
    """Linear, a scripted helper, relu."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(scripted_gate(self.fc(x), 0.1))


class _Cos3(nn.Module):
    """Body of the scripted module the negative holds outside ``named_modules()``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.cos() * 3


_UNREGISTERED_SCRIPTED = {"module": torch.jit.script(_Cos3())}


def _boundary_ops(trace: tl.Trace) -> list:
    """Return the ``jitscript`` ops of a trace."""

    return [op for op in trace._layer_list if op.func_name == SCRIPT_FUNCTION_FUNC_NAME]


def _trace_with_disclosure(model: nn.Module, x: torch.Tensor, **kwargs) -> tl.Trace:
    """Capture, asserting the once-per-function boundary disclosure fired."""

    with pytest.warns(TorchLensWarning, match="TorchScript function") as record:
        trace = tl.trace(model, x, **kwargs)
    codes = [getattr(w.message, "fields", {}).get("code") for w in record]
    assert "torchscript_boundary" in codes  # literal: the error-code coverage scan reads it
    return trace


def _assert_validates_clean(model: nn.Module, x: torch.Tensor) -> None:
    """Assert forward validation passes with no completeness-witness gap."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert validate_forward_pass(model, [x], input_kwargs={})
    assert last_validation_failure() is None
    assert not any(isinstance(item.message, CaptureGapWarning) for item in caught)


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/_completeness_dispatch_names.py, campaign mutant kills.
def test_scripted_helper_is_one_boundary_op_and_validates() -> None:
    """A module-level scripted helper is one replayable boundary op."""

    model = _GateModel()
    x = torch.randn(3, 4)
    trace = _trace_with_disclosure(model, x)

    (op,) = _boundary_ops(trace)
    assert op.is_transform is True
    assert op.transform_kind == SCRIPT_FUNCTION_TRANSFORM_KIND
    assert op.transform_config["function"] == scripted_gate.qualified_name
    assert op.transform_config["source"].startswith(__file__)
    assert op.transform_fn_qualname == "scripted_gate"
    assert is_script_function_recorder(op.func)
    assert [record_type(trace[label]) for label in op.parents] == ["linear"]
    assert [record_type(trace[label]) for label in op.children] == ["relu"]
    assert not any(other.type in {"sin", "ge", "__ge__"} for other in trace._layer_list)
    torch.testing.assert_close(op.out, scripted_gate(model.fc(x), 0.1))
    _assert_validates_clean(model, x)
