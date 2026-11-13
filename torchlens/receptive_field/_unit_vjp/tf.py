"""TensorFlow cone replay for gradient influence probes.

Each captured raw op replays through the TensorFlow validator's own
``_replay_raw_op`` from inputs rebuilt by its ``_rebuild_inputs``, with cone
values substituted for the parents on the probed cone. ``tf.GradientTape`` and
``tf.autodiff.ForwardAccumulator`` then differentiate the replay. Identity
annotations forward their parent; shape queries (integer outputs that depend on
shapes only) read their saved value. The replayable raw ops are deterministic
kernels with no random state, so the state guard has nothing to restore.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np

from ...backends._validation_shared import ops_by_label as _ops_by_label
from ...backends.tf._raw_op_replay import _import_tensorflow, _raw_op_dispatch, _replay_raw_op
from ...backends.tf.validation import (
    _captures_by_label,
    _payloads_close,
    _rebuild_inputs,
    _tensor_to_numpy,
)
from . import ReplayRefusal, single_parent_value

__all__: tuple[str, ...] = ()

_STRIPPED = (
    "This TensorFlow trace holds no runtime op records (a loaded trace keeps none), so its "
    "ops cannot be replayed for a gradient probe. Capture the model again in this process."
)
_SHAPE_QUERIES = frozenset({"Shape", "Size", "Rank"})


class _TFReplayer:
    """Replay of single TensorFlow raw ops from cone values and saved payloads."""

    def __init__(self, trace: Any) -> None:
        self._captures = _captures_by_label(trace)
        if not self._captures:
            raise ReplayRefusal(_STRIPPED)
        self._ops_by_label = _ops_by_label(trace)
        self._dispatch = frozenset(_raw_op_dispatch())
        self._tf = _import_tensorflow()

    def saved(self, op: Any) -> Any:
        """Return the op's saved payload as a TensorFlow tensor."""

        if not bool(getattr(op, "has_saved_activation", False)):
            raise ReplayRefusal(f"Op {op.label!r} has no saved payload.")
        return self._tf.convert_to_tensor(op.out)

    def _saved_by_label(self, label: str) -> Any:
        return self.saved(self._ops_by_label[label])

    def run_op(self, op: Any, env: Mapping[str, Any]) -> Any:
        """Replay one raw op; annotations forward their parent, shape queries read saved."""

        op_type = str(getattr(op, "func_name", ""))
        capture = self._captures.get(str(getattr(op, "_label_raw", op.label)))
        if capture is not None and op_type in self._dispatch:
            try:
                inputs = _rebuild_inputs(capture, self._ops_by_label, env)
                return _replay_raw_op(capture, inputs)
            except (LookupError, TypeError, ValueError) as exc:
                raise ReplayRefusal(
                    f"TensorFlow op {op.label!r} ({op_type}) cannot be replayed: {exc}"
                ) from exc
        if op_type in _SHAPE_QUERIES:
            return self.saved(op)
        value = single_parent_value(op, env, self._saved_by_label)
        if op_type == "StopGradient":
            return self._tf.stop_gradient(value)
        if op_type == "Identity" or bool(getattr(op, "is_output", False)):
            return value
        raise ReplayRefusal(
            f"TensorFlow op {op.label!r} ({op_type}) has no raw-op replay, so a gradient "
            "probe cannot cross it."
        )

    def close(self, replayed: Any, saved: Any) -> bool:
        """The TensorFlow validator's own payload comparison."""

        return bool(_payloads_close(replayed, saved))


def replayer(trace: Any) -> _TFReplayer:
    """Bind a TensorFlow trace's raw-op records for cone replay."""

    return _TFReplayer(trace)


def vjp(fn: Callable[..., Any], primals: tuple[Any, ...], cotangent: Any) -> tuple[Any, ...]:
    """Reverse-mode derivative under ``tf.GradientTape``; unconnected sources get zeros."""

    tf = _import_tensorflow()
    sources = [tf.convert_to_tensor(value) for value in primals]
    with tf.GradientTape() as tape:
        for value in sources:
            tape.watch(value)
        output = fn(*sources)
    grads = tape.gradient(
        output,
        sources,
        output_gradients=cotangent,
        unconnected_gradients=tf.UnconnectedGradients.ZERO,
    )
    return tuple(grads)


def jvp(
    fn: Callable[..., tuple[Any, ...]], primals: tuple[Any, ...], tangents: tuple[Any, ...]
) -> tuple[Any, ...]:
    """Forward-mode derivative under ``tf.autodiff.ForwardAccumulator``."""

    tf = _import_tensorflow()
    sources = [tf.convert_to_tensor(value) for value in primals]
    with tf.autodiff.ForwardAccumulator(sources, list(tangents)) as accumulator:
        outputs = fn(*sources)
    return tuple(
        accumulator.jvp(output, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        for output in outputs
    )


def to_numpy(value: Any) -> np.ndarray:
    """Host copy of a TensorFlow tensor."""

    return _tensor_to_numpy(value)


def from_numpy(array: np.ndarray, like: Any) -> Any:
    """A TensorFlow tensor from ``array`` in the dtype of ``like``."""

    tf = _import_tensorflow()
    return tf.convert_to_tensor(array, dtype=tf.convert_to_tensor(like).dtype)


@contextmanager
def state_guard() -> Iterator[None]:
    """The replayable raw ops hold no random state; nothing to restore."""

    yield
