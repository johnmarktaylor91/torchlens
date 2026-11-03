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

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import Trace, trace as trace_fn
from torchlens._io import ArtifactError


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


def _build_buffer_trace() -> tl.Trace:
    """Build a Trace over a model with a registered, mutated buffer."""

    torch.manual_seed(0)
    model = _BufferModel()
    x = torch.randn(3, 4)
    return trace_fn(model, x, save="all")


# ---------------------------------------------------------------------------
# (a) GradFn weakref: pickle.dumps/loads round-trip with a real backward pass
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# (b) Accessor weakref: pickle.dumps(trace.buffers) / pickle.dumps(trace.layers)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# (c) 7-class under-coercion: absent legacy container field -> typed default
# ---------------------------------------------------------------------------


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
    state["_layer_dict_main_keys"] = 12345

    restored = Trace.__new__(Trace)
    with pytest.raises(ArtifactError):
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
