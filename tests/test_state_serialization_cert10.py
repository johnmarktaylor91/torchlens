"""Regression tests for round-10 (cert10) state-serialization fixes.

Round-9 (cert10 g1, commit 241bc170) hardened state serialization for Trace,
Op, Layer, and Buffer, but the fix was INSTANCE-deep rather than CLASS-deep:
it fixed the specific fields the audit happened to find, not the whole
defect class across every sibling record class. This module closes that
gap:

* ``GradFn.__getstate__`` never nulled its live ``_source_trace_ref``
  weakref (BLOCKER: crashes ``pickle.dumps(trace)`` for any trace with a
  captured backward pass).
* The base ``Accessor`` class had no ``__getstate__``/``__setstate__``, so
  ``pickle.dumps(trace.buffers)`` / ``pickle.dumps(trace.layers)`` crashed
  the same way.
* 7 of 11 record classes (``Layer``, ``ModuleCall``, ``Module``, ``Param``,
  ``GradFn``, ``GradFnCall``, ``BackwardPass``) never called
  ``coerce_container_typed_state`` in ``__setstate__``, so present-but-
  wrong-typed legacy container fields silently kept the wrong type.
* ``coerce_container_typed_state`` silently emptied present-but-wrong-typed
  fields when the conversion failed, turning real corruption into silent
  data loss instead of a loud failure -- defeating the
  validation-is-a-tripwire principle.
"""

from __future__ import annotations

import pickle
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import Trace, trace as trace_fn
from torchlens._io import TorchLensIOError
from torchlens.data_classes.backward_pass import BackwardPass
from torchlens.data_classes.grad_fn import GradFn
from torchlens.data_classes.grad_fn_call import GradFnCall
from torchlens.data_classes.module import Module, ModuleCall
from torchlens.data_classes.layer import Layer
from torchlens.data_classes.param import Param
from torchlens.options import CaptureOptions


class _TinyBackwardModel(nn.Module):
    """Small MLP used to exercise a real captured backward pass."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        return self.fc2(torch.relu(self.fc1(x)))


class _BufferModel(nn.Module):
    """Small model with a registered buffer, used for buffer-accessor tests."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("running_total", torch.zeros(4))
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass that mutates the registered buffer."""
        self.running_total = self.running_total + 1
        return self.linear(x + self.running_total)


def _output_loss(trace_obj: tl.Trace) -> torch.Tensor:
    """Return a scalar loss computed from the trace's final output layer."""

    return trace_obj[trace_obj.output_layers[0]].out.sum()


def _build_backward_trace() -> tl.Trace:
    """Build a Trace with a real captured backward pass."""

    torch.manual_seed(0)
    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    live_trace = tl.trace(model, x, capture=CaptureOptions(save_grads="all"))
    live_trace.log_backward(_output_loss(live_trace))
    return live_trace


def _build_buffer_trace() -> tl.Trace:
    """Build a Trace over a model with a registered, mutated buffer."""

    torch.manual_seed(0)
    model = _BufferModel()
    x = torch.randn(3, 4)
    return trace_fn(model, x, layers_to_save="all")


# ---------------------------------------------------------------------------
# (a) GradFn weakref: pickle.dumps/loads round-trip with a real backward pass
# ---------------------------------------------------------------------------


def test_gradfn_weakref_pickle_roundtrip_with_real_backward_pass() -> None:
    """A Trace with a captured backward pass must survive plain pickle.

    Regression guard for the round-10 BLOCKER: ``GradFn.__getstate__`` never
    nulled its live ``_source_trace_ref`` weakref, so ``pickle.dumps(trace)``
    crashed with ``TypeError: cannot pickle 'weakref.ReferenceType' object``
    for any trace with a real captured backward pass. ``GradFnCall`` already
    nulled this field correctly; ``GradFn`` did not.
    """

    live_trace = _build_backward_trace()
    assert live_trace.num_backward_passes >= 1
    assert len(live_trace.grad_fn_logs) > 0

    data = pickle.dumps(live_trace)
    restored = pickle.loads(data)

    assert restored.num_backward_passes == live_trace.num_backward_passes
    assert len(restored.grad_fn_logs) == len(live_trace.grad_fn_logs)
    # `Trace.__setstate__` deliberately reconnects each GradFn's
    # `source_trace` back-reference to the newly-restored (live) Trace, so
    # the weakref should point at `restored`, not at a stale intermediate
    # object from the pickle stream.
    for grad_fn_handle in restored.grad_fn_logs.values():
        assert grad_fn_handle.source_trace is restored


def test_gradfn_standalone_pickle_does_not_crash() -> None:
    """A standalone ``GradFn`` (outside its owning Trace) must pickle cleanly."""

    live_trace = _build_backward_trace()
    grad_fn_handle = next(iter(live_trace.grad_fn_logs.values()))

    data = pickle.dumps(grad_fn_handle)
    restored = pickle.loads(data)

    assert restored.label == grad_fn_handle.label
    assert restored._source_trace_ref is None


# ---------------------------------------------------------------------------
# (b) Accessor weakref: pickle.dumps(trace.buffers) / pickle.dumps(trace.layers)
# ---------------------------------------------------------------------------


def test_pickle_trace_buffers_accessor_roundtrip() -> None:
    """``pickle.dumps(trace.buffers)`` must not crash on the owning weakref.

    Regression guard: the base ``Accessor`` class had no
    ``__getstate__``/``__setstate__``, so subclasses that set a live
    ``_source_ref`` weakref (``BufferAccessor``, ``LayerAccessor``,
    ``GradFnCallAccessor``, ...) crashed on plain pickling with
    ``TypeError: cannot pickle 'weakref.ReferenceType' object``.
    """

    live_trace = _build_buffer_trace()
    assert len(live_trace.buffers) > 0

    data = pickle.dumps(live_trace.buffers)
    restored = pickle.loads(data)

    assert len(restored) == len(live_trace.buffers)
    assert set(restored.keys()) == set(live_trace.buffers.keys())


def test_pickle_trace_layers_accessor_roundtrip() -> None:
    """``pickle.dumps(trace.layers)`` must not crash on the owning weakref."""

    live_trace = _build_buffer_trace()
    assert len(live_trace.layers) > 0

    data = pickle.dumps(live_trace.layers)
    restored = pickle.loads(data)

    assert len(restored) == len(live_trace.layers)
    assert set(restored.keys()) == set(live_trace.layers.keys())


# ---------------------------------------------------------------------------
# (c) 7-class under-coercion: absent legacy container field -> typed default
# ---------------------------------------------------------------------------


def test_layer_setstate_absent_container_field_restores_typed() -> None:
    """An absent Layer container field must restore as its typed default."""

    live_trace = _build_buffer_trace()
    layer_log = next(iter(live_trace.layer_list))
    state = layer_log.__getstate__()
    state.pop("equivalent_ops", None)

    restored = Layer.__new__(Layer)
    restored.__setstate__(state)

    assert isinstance(restored.equivalent_ops, set)


def test_module_call_setstate_absent_container_field_restores_typed() -> None:
    """An absent ModuleCall container field must restore as its typed default.

    ``call_children`` had no entry at all in ``ModuleCall``'s pre-round-10
    ``default_fill_state`` defaults dict (unlike ``module_call_stack``, which
    was already covered), so an old pickle missing it crashed with
    ``AttributeError`` rather than restoring an empty list.
    """

    live_trace = _build_buffer_trace()
    module_call = next(iter(live_trace.module_calls))
    state = module_call.__getstate__()
    state.pop("call_children", None)

    restored = ModuleCall.__new__(ModuleCall)
    restored.__setstate__(state)

    assert isinstance(restored.call_children, list)


def test_module_setstate_absent_container_field_restores_typed() -> None:
    """An absent Module container field must restore as its typed default."""

    live_trace = _build_buffer_trace()
    module_log = live_trace.modules["self"]
    state = module_log.__getstate__()
    state.pop("custom_attributes", None)

    restored = Module.__new__(Module)
    restored.__setstate__(state)

    assert isinstance(restored.custom_attributes, dict)


def test_param_setstate_absent_container_field_restores_typed() -> None:
    """An absent Param container field must restore as its typed default.

    ``used_by_ops`` had no entry at all in ``Param``'s pre-round-10
    ``default_fill_state`` defaults dict (unlike ``co_parent_params``, which
    was already covered), so an old pickle missing it crashed with
    ``AttributeError`` rather than restoring an empty list.
    """

    live_trace = _build_buffer_trace()
    param_log = live_trace.params[0]
    state = param_log.__getstate__()
    state.pop("used_by_ops", None)

    restored = Param.__new__(Param)
    restored.__setstate__(state)

    assert isinstance(restored.used_by_ops, list)


def test_param_setstate_present_but_wrong_typed_co_parent_params_is_coerced() -> None:
    """A present-but-wrong-typed ``Param.co_parent_params`` must be coerced.

    This is the exact under-coercion gap named in the round-10 audit: without
    a ``coerce_container_typed_state`` call in ``Param.__setstate__``, a
    legacy ``set`` value for ``co_parent_params`` silently kept the wrong
    type instead of being repaired to the declared ``list``.
    """

    live_trace = _build_buffer_trace()
    param_log = live_trace.params[0]
    state = param_log.__getstate__()
    state["co_parent_params"] = {"legacy_as_set"}

    restored = Param.__new__(Param)
    restored.__setstate__(state)

    assert isinstance(restored.co_parent_params, list)
    assert restored.co_parent_params == ["legacy_as_set"]


def test_grad_fn_setstate_present_but_wrong_typed_co_parents_is_coerced() -> None:
    """A present-but-wrong-typed GradFn container field must be coerced.

    Every GradFn container field already had an absent-fill default before
    round-10 (unlike ``Param.co_parent_params``), so the missing piece here
    is specifically the ``coerce_container_typed_state`` call: without it, a
    legacy ``set`` value for ``co_parents`` silently kept the wrong type
    instead of being repaired to the declared ``list``.
    """

    live_trace = _build_backward_trace()
    grad_fn_handle = next(iter(live_trace.grad_fn_logs.values()))
    state = grad_fn_handle.__getstate__()
    state["co_parents"] = {"legacy_as_set"}

    restored = GradFn.__new__(GradFn)
    restored.__setstate__(state)

    assert isinstance(restored.co_parents, list)
    assert restored.co_parents == ["legacy_as_set"]


def test_grad_fn_call_setstate_survives_default_fill() -> None:
    """GradFnCall's (no-op) coerce call must not disturb normal restoration.

    GradFnCall declares no plain-container fields at all, so there is no
    present-but-wrong-typed defect to reproduce here; this is a supplementary
    consistency check (not a fail-before/pass-after regression guard) that
    adding the call for consistency with every sibling record class does not
    break ordinary restoration.
    """

    live_trace = _build_backward_trace()
    grad_fn_handle = next(iter(live_trace.grad_fn_logs.values()))
    call = next(iter(grad_fn_handle.calls.values()))
    state = call.__getstate__()

    restored = GradFnCall.__new__(GradFnCall)
    restored.__setstate__(state)

    assert restored.call_index == call.call_index


def test_backward_pass_setstate_present_but_wrong_typed_grad_fn_calls_is_coerced() -> None:
    """A present-but-wrong-typed BackwardPass container field must be coerced.

    Every BackwardPass container field already had an absent-fill default
    before round-10, so the missing piece here is specifically the
    ``coerce_container_typed_state`` call: without it, a legacy ``tuple``
    value for ``grad_fn_calls`` silently kept the wrong type instead of being
    repaired to the declared ``list``.
    """

    live_trace = _build_backward_trace()
    backward_pass_log = live_trace.backward_passes[0]
    state = backward_pass_log.__getstate__()
    state["grad_fn_calls"] = tuple(state["grad_fn_calls"])

    restored = BackwardPass.__new__(BackwardPass)
    restored.__setstate__(state)

    assert isinstance(restored.grad_fn_calls, list)


# ---------------------------------------------------------------------------
# (d) TRIPWIRE GUARD: corrupted, non-convertible container must not be
# silently swallowed into an empty container.
# ---------------------------------------------------------------------------


def test_coerce_container_typed_state_raises_on_incompatible_corruption() -> None:
    """A present-but-incompatible container field must raise, not go empty.

    Regression guard for the round-10 BLOCKER in
    ``coerce_container_typed_state``: previously, when a field was present
    with a value that could not be converted to its declared container type
    (e.g. an ``int`` where a ``dict`` is declared), the function silently
    substituted an empty typed container. That turned a loud pre-fix crash
    into silent, undetectable data loss -- exactly what the
    validation-is-a-tripwire principle forbids. The corrupted value must
    either raise, or otherwise stay detectably wrong; it must never come
    back as a plausible-looking empty dict.
    """

    live_trace = _build_buffer_trace()
    state = live_trace.__getstate__()
    # `layer_dict_main_keys` is declared a `dict`; an `int` cannot be
    # converted to a `dict` (`dict(12345)` raises `TypeError`), so this is
    # unambiguous corruption, not a legacy container-type migration.
    state["layer_dict_main_keys"] = 12345

    restored = Trace.__new__(Trace)
    with pytest.raises(TorchLensIOError):
        restored.__setstate__(state)


def test_coerce_container_typed_state_still_repairs_convertible_legacy_types() -> None:
    """A present-but-differently-typed (but convertible) field is still repaired.

    This is the companion case to the tripwire guard above: a legacy
    container-type migration (e.g. a `set` stored where a `list` is now
    declared) is NOT corruption -- `list(a_set)` succeeds and is lossless --
    so it must still be silently repaired, exactly as before this sprint's
    fix. Only genuinely incompatible values (where the conversion itself
    raises) must surface loudly.
    """

    from torchlens._io import coerce_container_typed_state

    state: dict[str, Any] = {"my_field": {"a", "b", "c"}}
    coerce_container_typed_state(state, {"my_field": []})

    assert isinstance(state["my_field"], list)
    assert set(state["my_field"]) == {"a", "b", "c"}


def test_coerce_container_typed_state_absent_field_untouched() -> None:
    """Absent fields are `default_fill_state`'s job, not `coerce`'s.

    `coerce_container_typed_state` only touches fields already present in
    `state` (per its docstring contract) -- absence is handled upstream by
    `default_fill_state`. This is a narrow unit check that the "only touch
    present fields" contract still holds after the round-10 fix.
    """

    from torchlens._io import coerce_container_typed_state

    state: dict[str, Any] = {}
    coerce_container_typed_state(state, {"my_field": []})

    assert "my_field" not in state
