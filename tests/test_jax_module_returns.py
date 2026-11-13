"""JAX module-return recording: what the backend-keyed cell does not cover.

The cell (``tests/test_module_return_cell.py``) checks recorded returns against each call's
actual return on Equinox and NNX. This file pins the JAX-only rules of
``torchlens/backends/jax/_module_return_probe.py``:

- a module called under ``jax.vmap`` returns tracers of an inner trace; the call stays
  unrecorded (no output op, no output path) and is disclosed with a ``CaptureGapWarning``
  instead of being paired with a guessed op;
- a module called inside a ``jax.lax.scan``, ``cond`` or ``while_loop`` body refuses (the
  strict module surface), so no such return is ever recorded;
- the extra jaxpr outputs that carry module returns never become trace outputs and are not
  in the stamped jaxpr;
- the trace still validates.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import torchlens as tl
from torchlens.backends._module_returns import ModuleReturnLedger, returned_op_labels
from torchlens.errors import CaptureGapWarning
from torchlens.intervention.errors import MultiOutputModuleError
from torchlens.validation import check_metadata_invariants

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
eqx = pytest.importorskip("equinox")

pytestmark = [pytest.mark.backend_jax, pytest.mark.optional, pytest.mark.rare]

_WIDTH = 4


class _Linear(eqx.Module):
    """Square linear layer on the last axis."""

    weight: Any
    bias: Any

    def __init__(self, seed: int) -> None:
        """Draw weights from ``seed``."""

        self.weight = jax.random.normal(jax.random.PRNGKey(seed), (_WIDTH, _WIDTH)) / _WIDTH
        self.bias = jnp.full((_WIDTH,), 0.1)

    def __call__(self, value: Any) -> Any:
        """Return ``value @ weight.T + bias``."""

        return jnp.matmul(value, self.weight.T) + self.bias


class _ControlFlowRoot(eqx.Module):
    """Root that calls ``block`` inside a ``lax.scan``, ``cond`` or ``while_loop`` body."""

    block: _Linear
    kind: str = eqx.field(static=True)

    def __call__(self, value: Any) -> Any:
        """Route ``block`` through the chosen control-flow primitive."""

        if self.kind == "scan":
            out, _ = jax.lax.scan(
                lambda carry, _x: (self.block(carry), None), value, None, length=3
            )
            return out
        if self.kind == "cond":
            return jax.lax.cond(jnp.sum(value) > 0, lambda v: self.block(v), lambda v: v, value)
        return jax.lax.while_loop(
            lambda state: state[0] < 3,
            lambda state: (state[0] + 1, self.block(state[1])),
            (0, value),
        )[1]


class _VmapBlock(eqx.Module):
    """Block that applies an unbatched ``eqx.nn.Linear`` under ``jax.vmap``."""

    lin: eqx.nn.Linear

    def __call__(self, value: Any) -> Any:
        """Return the rectified vmapped projection."""

        return jnp.maximum(jax.vmap(self.lin)(value), 0.0)


class _VmapRoot(eqx.Module):
    """Root holding one vmapping block."""

    block: _VmapBlock

    def __call__(self, value: Any) -> Any:
        """Return twice the block output."""

        return self.block(value) * 2.0


def _input() -> Any:
    """Return a deterministic input."""

    return jnp.linspace(-1.0, 1.0, 2 * _WIDTH).reshape(2, _WIDTH) + 0.25


def _assert_unrecorded(trace: Any, call_label: str) -> None:
    """Assert a module call has no recorded output at all."""

    call = trace.module_calls[call_label]
    assert list(call.output_ops) == []
    assert tuple(call.output_paths) == ()
    assert returned_op_labels(trace, call_label) == []
    with pytest.raises(MultiOutputModuleError):
        _ = call.out


def _assert_trace_sound(trace: Any, model: Any) -> None:
    """Assert invariants, validation, and that only the model output is a trace output."""

    assert check_metadata_invariants(trace) is True
    assert trace.validate_forward_pass([]) is True
    assert len(trace.output_layers) == 1
    assert len(trace._provenance_jax_closed_jaxpr.jaxpr.outvars) == 1
    np.testing.assert_allclose(
        np.asarray(trace.ops[trace.output_layers[0]].out), np.asarray(model(_input())), rtol=1e-6
    )


@pytest.mark.parametrize("kind", ["scan", "cond", "while"])
def test_module_inside_root_control_flow_is_captured(kind: str) -> None:
    """A module call inside a control-flow body is captured with its ops attributed.

    The call returns an inner-trace tracer of the scan, cond or while body; no single op is
    its output across loop iterations, so the call stays unrecorded and is disclosed.
    """

    model = _ControlFlowRoot(_Linear(1), kind)
    with pytest.warns(CaptureGapWarning, match=r"'block:1' returned arrays of an inner trace"):
        trace = tl.trace(model, _input(), backend="jax")

    _assert_trace_sound(trace, model)
    _assert_unrecorded(trace, "block:1")
    assert trace[tl.in_module("block")].labels


def test_vmapped_child_is_unrecorded_and_parent_recorded() -> None:
    """A child under ``jax.vmap`` is unrecorded and disclosed; its parent's return is recorded."""

    model = _VmapRoot(_VmapBlock(eqx.nn.Linear(_WIDTH, _WIDTH, key=jax.random.PRNGKey(4))))
    with pytest.warns(
        CaptureGapWarning, match=r"'block\.lin:1' returned arrays of an inner trace"
    ) as caught:
        trace = tl.trace(model, _input(), backend="jax")
    gaps = [w.message for w in caught if isinstance(w.message, CaptureGapWarning)]
    assert [gap.fields["code"] for gap in gaps] == ["module_return_unrecorded"]
    _assert_unrecorded(trace, "block.lin:1")
    block_call = trace.module_calls["block:1"]
    assert len(block_call.output_ops) == 1
    assert tuple(block_call.output_paths) == ((),)
    expected = model.block(_input())
    np.testing.assert_allclose(np.asarray(block_call.out), np.asarray(expected), rtol=1e-6)
    _assert_trace_sound(trace, model)


def test_recording_leaves_trace_outputs_and_jaxpr_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With recording on or off, the ops, trace outputs and stamped jaxpr are the same."""

    model = _VmapRoot(_VmapBlock(eqx.nn.Linear(_WIDTH, _WIDTH, key=jax.random.PRNGKey(5))))
    seen: dict[str, tuple[Any, ...]] = {}
    for arm in ("off", "on"):
        if arm == "off":
            monkeypatch.setattr(ModuleReturnLedger, "record", lambda *args, **kwargs: None)
        else:
            monkeypatch.undo()
        with pytest.warns(CaptureGapWarning, match=r"'block\.lin:1'"):
            trace = tl.trace(model, _input(), backend="jax")
        seen[arm] = (
            tuple(op.label for op in trace._layer_list),
            tuple(trace.output_layers),
            str(trace._provenance_jax_closed_jaxpr),
        )
    assert seen["on"] == seen["off"]
