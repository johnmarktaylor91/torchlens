"""JAX cone replay for gradient influence probes.

Each captured equation replays through the JAX validator's own
``replay_equation`` with its recorded inputs, except that a parent on the probed
cone reads the value the cone replay produced for it. ``jax.vjp`` and
``jax.jvp`` then differentiate the replay. JAX keeps no global random state
(keys are explicit equation inputs), so the state guard has nothing to restore.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np

from ...backends.jax import backend as jax_backend
from ...backends.jax._trace_graph_helpers import (
    _data_parent_arg_positions,
    _same_array,
    _saved_parent_input,
    _saved_single_output,
    _values_close,
)
from ...backends.jax.jaxpr import replay_equation
from . import ReplayRefusal

__all__: tuple[str, ...] = ()

_STRIPPED = (
    "This JAX trace holds no runtime equation records (a loaded trace keeps none), so its "
    "ops cannot be replayed for a gradient probe. Capture the model again in this process."
)


class _JaxReplayer:
    """Replay of single JAX equations from cone values and saved payloads."""

    def __init__(self, trace: Any) -> None:

        captures = tuple(getattr(trace, "jax_equation_captures", ()) or ())
        equation_ops = jax_backend._jax_equation_ops(trace)
        if not captures or len(captures) != len(equation_ops):
            raise ReplayRefusal(_STRIPPED)
        self._captures = {
            op.label: capture for capture, op in zip(captures, equation_ops, strict=True)
        }
        self._ops_by_label = jax_backend._jax_ops_by_label(trace)
        self._hidden = {
            label: value if isinstance(value, tuple) else (value,)
            for label, value in getattr(trace, "_selective_save_hidden_payloads", {}).items()
        }

    def saved(self, op: Any) -> Any:
        """Return the op's saved payload (hidden replay payloads included)."""

        try:
            return _saved_single_output(op, self._hidden)
        except (AttributeError, KeyError, RuntimeError, ValueError) as exc:
            raise ReplayRefusal(f"Op {op.label!r} has no saved payload to replay from.") from exc

    def _parent_value(self, parent_label: str, recorded: Any, env: Mapping[str, Any]) -> Any:
        """The value a child reads from one parent: the cone value or the saved one."""

        parent_op = self._ops_by_label[parent_label]
        if parent_label not in env:
            return _saved_parent_input(parent_op, self._hidden, recorded)
        value = env[parent_label]
        if not isinstance(value, tuple):
            return value
        saved = self.saved(parent_op)
        for index, element in enumerate(saved):
            if element is recorded or _same_array(element, recorded):
                return value[index]
        raise ReplayRefusal(
            f"Multi-output parent {parent_label!r} saved no output matching its child's input."
        )

    def run_op(self, op: Any, env: Mapping[str, Any]) -> Any:
        """Replay one equation op; an output marker passes its single parent through."""

        capture = self._captures.get(op.label)
        if capture is None:
            return _pass_through(op, env, self)
        inputs = list(capture.input_values)
        for position, parent_label in _data_parent_arg_positions(op).items():
            inputs[position] = self._parent_value(parent_label, inputs[position], env)
        try:
            outputs = replay_equation(capture, tuple(inputs))
        except (NotImplementedError, TypeError, ValueError) as exc:
            raise ReplayRefusal(
                f"JAX equation {op.label!r} ({capture.primitive}) cannot be replayed: {exc}"
            ) from exc
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def close(self, replayed: Any, saved: Any) -> bool:
        """The JAX validator's own value comparison over matching trees."""

        import jax  # deferred: optional dependency

        return bool(jax.tree.all(jax.tree.map(_values_close, replayed, saved)))


def _pass_through(op: Any, env: Mapping[str, Any], replayer: _JaxReplayer) -> Any:
    """An op with no equation record: a single-parent output marker reads its parent."""

    parents = tuple(getattr(op, "parents", ()) or ())
    if len(parents) != 1 or not bool(getattr(op, "is_output", False)):
        raise ReplayRefusal(
            f"Op {op.label!r} has no JAX equation record to replay (a control-flow region "
            "or an opaque call); a gradient probe cannot cross it."
        )
    (parent,) = parents
    if parent in env:
        return env[parent]
    return replayer.saved(replayer._ops_by_label[parent])


def replayer(trace: Any) -> _JaxReplayer:
    """Bind a JAX trace's equation records for cone replay."""

    return _JaxReplayer(trace)


def vjp(fn: Callable[..., Any], primals: tuple[Any, ...], cotangent: Any) -> tuple[Any, ...]:
    """``jax.vjp`` of ``fn`` at ``primals`` against one output cotangent."""

    import jax  # deferred: optional dependency

    _, pullback = jax.vjp(fn, *primals)
    return tuple(pullback(cotangent))


def jvp(
    fn: Callable[..., tuple[Any, ...]], primals: tuple[Any, ...], tangents: tuple[Any, ...]
) -> tuple[Any, ...]:
    """``jax.jvp`` of a tuple-valued ``fn``: one tangent per output."""

    import jax  # deferred: optional dependency

    _, tangent_out = jax.jvp(fn, primals, tangents)
    return tuple(tangent_out)


def to_numpy(value: Any) -> np.ndarray:
    """Host copy of a JAX array."""

    return np.asarray(value)


def from_numpy(array: np.ndarray, like: Any) -> Any:
    """A JAX array from ``array`` in the dtype of ``like``."""

    import jax.numpy as jnp  # deferred: optional dependency

    return jnp.asarray(array, dtype=jnp.asarray(like).dtype)


@contextmanager
def state_guard() -> Iterator[None]:
    """JAX has no global random state; nothing to restore."""

    yield
