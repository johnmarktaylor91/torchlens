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
from torchlens import _state
from torchlens._errors import TorchLensCaptureGapWarning
from torchlens._utils._torch_compat import HAS_SCRIPT_FUNCTION_CALL_HOOK
from torchlens.backends.torch._script_boundary import (
    SCRIPT_FUNCTION_FUNC_NAME,
    SCRIPT_FUNCTION_TRANSFORM_KIND,
    SCRIPT_FUNCTION_WRAPPER_NAME,
    is_script_function_recorder,
)
from torchlens.backends.torch.completeness_witness import AUDITED_COMPLETENESS_BOUNDARIES
from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch
from torchlens.errors import TorchLensWarning
from torchlens.validation import last_validation_failure, validate_forward_pass
from torchlens.validation.diagnostics import CHECK_COMPLETENESS

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


class _NeuronModel(nn.Module):
    """spikingjelly neuron shape: a scripted static method through ``self`` returning a tuple."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    @staticmethod
    @torch.jit.script
    def jit_step_hard_reset(x: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float):
        v = v + x
        spike = (v >= v_threshold).to(x)
        v = v_reset * spike + (1.0 - spike) * v
        return spike, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        spike, v = self.jit_step_hard_reset(h, torch.zeros_like(h), 0.2, 0.0)
        return spike + torch.tanh(v)


class _Surrogate(torch.autograd.Function):
    """spikingjelly surrogate: the forward calls a scripted helper."""

    @staticmethod
    def forward(ctx, x):  # type: ignore[override]
        ctx.save_for_backward(x)
        return scripted_heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        (x,) = ctx.saved_tensors
        sig = torch.sigmoid(x)
        return grad_output * sig * (1 - sig)


class _SurrogateModel(nn.Module):
    """Linear, then the surrogate Function."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _Surrogate.apply(self.fc(x)) * 2


class _WriteModel(nn.Module):
    """A scripted call that writes into its argument."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(scripted_write(self.fc(x).clone()))


class _Cos3(nn.Module):
    """Body of the scripted module the negative holds outside ``named_modules()``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.cos() * 3


_UNREGISTERED_SCRIPTED = {"module": torch.jit.script(_Cos3())}


class _UnregisteredScriptModuleModel(nn.Module):
    """An opaque call that is not a ScriptFunction: a scripted module held in a dict.

    Its forward is a ``ScriptMethod`` of a ``ScriptModule`` the up-front refusal cannot
    see (it is not a registered submodule), so it is exactly as opaque as a scripted
    function but is not one.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(_UNREGISTERED_SCRIPTED["module"](self.fc(x)))


def _boundary_ops(trace: tl.Trace) -> list:
    """Return the ``jitscript`` ops of a trace."""

    return [op for op in trace.layer_list if op.func_name == SCRIPT_FUNCTION_FUNC_NAME]


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
    assert not any(isinstance(item.message, TorchLensCaptureGapWarning) for item in caught)


def _assert_fails_completeness(model: nn.Module, x: torch.Tensor) -> None:
    """Assert forward validation fails on the completeness check."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert validate_forward_pass(model, [x], input_kwargs={}) is False
    failure = last_validation_failure()
    assert failure is not None and failure.check == CHECK_COMPLETENESS


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
    assert [trace[label].layer_type for label in op.parents] == ["linear"]
    assert [trace[label].layer_type for label in op.children] == ["relu"]
    assert not any(other.layer_type in {"sin", "ge", "__ge__"} for other in trace.layer_list)
    torch.testing.assert_close(op.out, scripted_gate(model.fc(x), 0.1))
    _assert_validates_clean(model, x)


def test_scripted_helper_owns_its_interior_dispatches() -> None:
    """With the census armed, every dispatch inside the call belongs to the boundary."""

    saved_escape = _state._escape_detector_mode
    saved_witness = _state._completeness_witness_mode
    unwrap_torch()
    wrap_torch(completeness_witness=True)
    try:
        trace = _trace_with_disclosure(_GateModel(), torch.randn(3, 4))
    finally:
        unwrap_torch()
        wrap_torch(escape_detector=saved_escape, completeness_witness=saved_witness)

    assert trace.capture_verified is True
    assert trace.completeness_diagnostics == []
    rows = [
        entry
        for entry in trace.completeness_decompositions
        if entry["owner_wrapper"] == SCRIPT_FUNCTION_WRAPPER_NAME
    ]
    assert len(rows) == 1
    assert rows[0]["owner_func_call_id"] == _boundary_ops(trace)[0].func_call_id
    assert rows[0]["capture_accounted"] is True


def test_scripted_static_method_returning_a_tuple() -> None:
    """The spikingjelly neuron step: one call, two outputs, one func_call_id, validates."""

    model = _NeuronModel()
    x = torch.randn(3, 4)
    trace = _trace_with_disclosure(model, x)

    ops = _boundary_ops(trace)
    assert len(ops) == 2
    assert len({op.func_call_id for op in ops}) == 1
    assert ops[0].transform_config["function"] == _NeuronModel.jit_step_hard_reset.qualified_name
    _assert_validates_clean(model, x)


def test_scripted_helper_inside_an_autograd_function() -> None:
    """The surrogate case: the Function stays op by op, the scripted call is one op in it."""

    model = _SurrogateModel()
    x = torch.randn(3, 4)
    trace = _trace_with_disclosure(model, x)

    (op,) = _boundary_ops(trace)
    assert [trace[label].layer_type for label in op.parents] == ["linear"]
    assert not any(other.func_name == "autogradfunction" for other in trace.layer_list)
    _assert_validates_clean(model, x)


def test_pytorch_jit_disabled_gives_the_op_by_op_graph() -> None:
    """With TorchScript disabled the helper is plain Python, captured op by op as today."""

    import torch.jit._state as jit_state

    jit_state.disable()
    try:

        def plain_gate(x: torch.Tensor, threshold: float) -> torch.Tensor:
            return (x >= threshold).to(x.dtype) * x.sin()

        gate = torch.jit.script(plain_gate)
    finally:
        jit_state.enable()
    assert gate is plain_gate

    class _PlainGateModel(_GateModel):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(gate(self.fc(x), 0.1))

    model = _PlainGateModel()
    x = torch.randn(3, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("error", TorchLensWarning)
        trace = tl.trace(model, x)
    assert _boundary_ops(trace) == []
    assert "sin" in {op.layer_type for op in trace.layer_list}
    _assert_validates_clean(model, x)


def test_scripted_module_is_still_refused() -> None:
    """A scripted MODULE is refused up front; only scripted functions become boundaries."""

    with pytest.raises(RuntimeError, match="ScriptModule"):
        tl.trace(torch.jit.script(nn.Linear(4, 4)), torch.randn(2, 4))


def test_scripted_call_writing_its_argument_is_not_collapsed() -> None:
    """Armed negative: an in-place write into an argument keeps today's census gap."""

    model = _WriteModel()
    x = torch.randn(3, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(model, x)
    assert _boundary_ops(trace) == []
    _assert_fails_completeness(model, x)


def test_opaque_call_that_is_not_scripted_still_fails_completeness() -> None:
    """Armed negative: the boundary never reaches outside ``ScriptFunction`` calls."""

    model = _UnregisteredScriptModuleModel()
    x = torch.randn(3, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(model, x)
    assert _boundary_ops(trace) == []
    _assert_fails_completeness(model, x)


def test_boundary_selector_row_and_hook_lifecycle() -> None:
    """Selectable by transform kind, audited by the witness table, hook removed on unwrap."""

    trace = _trace_with_disclosure(
        _GateModel(), torch.randn(2, 4), save=tl.func_transform(SCRIPT_FUNCTION_TRANSFORM_KIND)
    )
    saved = [op for op in trace.layer_list if op.has_saved_activation]
    assert [op.func_name for op in saved] == [SCRIPT_FUNCTION_FUNC_NAME]
    rows = [
        row
        for row in AUDITED_COMPLETENESS_BOUNDARIES
        if row.wrapper_name == SCRIPT_FUNCTION_WRAPPER_NAME
    ]
    assert len(rows) == 1 and rows[0].operator is None

    wrapped_call = torch.jit.ScriptFunction.__dict__["__call__"]
    assert getattr(wrapped_call, "__name__", None) == "wrapped_call"
    unwrap_torch()
    try:
        restored = torch.jit.ScriptFunction.__dict__["__call__"]
        assert getattr(restored, "__name__", None) != "wrapped_call"
    finally:
        wrap_torch()
    assert torch.jit.ScriptFunction.__dict__["__call__"].__name__ == "wrapped_call"
