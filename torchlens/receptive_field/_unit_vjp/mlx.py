"""MLX cone replay for gradient influence probes.

Each captured call replays through the MLX validator's own ``_reconstruct_call``
and ``capture.func``, with cone values standing in for the saved payloads of the
parents on the probed cone (the validator reads parents by raw label, so the
cone values are offered under those labels). ``mx.vjp`` and ``mx.jvp`` then
differentiate the replay. The state guard restores ``mx.random.state``, which a
replayed random call would advance.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import numpy as np

from ...backends._validation_shared import ops_by_label as _ops_by_label
from ...backends.mlx._capture_records import _iter_output_arrays, _payloads_close
from ...backends.mlx._replay_args import _reconstruct_call, _saved_payload
from . import ReplayRefusal, single_parent_value

__all__: list[str] = []

_STRIPPED = (
    "This MLX trace holds no runtime call records (a loaded trace keeps none), so its "
    "ops cannot be replayed for a gradient probe. Capture the model again in this process."
)


class _MLXReplayer:
    """Replay of single captured MLX calls from cone values and saved payloads."""

    def __init__(self, trace: Any) -> None:
        captures = tuple(getattr(trace, "_mlx_op_captures", ()) or ())
        if not captures:
            raise ReplayRefusal(_STRIPPED)
        self._trace = trace
        self._ops_by_label = _ops_by_label(trace)
        self._outputs = {
            label: (capture, index)
            for capture in captures
            for index, label in enumerate(capture.labels_raw)
        }

    def saved(self, op: Any) -> Any:
        """Return the op's saved payload (hidden selective-save copies included)."""

        try:
            return _saved_payload(self._trace, self._ops_by_label, _raw(op))
        except ValueError as exc:
            raise ReplayRefusal(str(exc)) from exc

    def _saved_by_label(self, label: str) -> Any:
        return self.saved(self._ops_by_label[label])

    def replay(self, op: Any, env: Mapping[str, Any]) -> Any:
        """Replay the call that produced ``op``; a record-less op forwards its parent."""

        entry = self._outputs.get(_raw(op))
        if entry is None:
            return single_parent_value(op, env, self._saved_by_label)
        capture, index = entry
        if capture.func is None or capture.interventions:
            raise ReplayRefusal(
                f"MLX op {op.label!r} is a literal or carries an intervention, so a "
                "gradient probe cannot replay it."
            )
        view = {**self._ops_by_label, **{key: SimpleNamespace(out=val) for key, val in env.items()}}
        try:
            args, kwargs = _reconstruct_call(self._trace, view, capture)
            outputs = _iter_output_arrays(capture.func(*args, **kwargs))
        except (TypeError, ValueError) as exc:
            raise ReplayRefusal(
                f"MLX op {op.label!r} ({capture.op_name}) cannot be replayed: {exc}"
            ) from exc
        if len(outputs) != len(capture.labels_raw):
            raise ReplayRefusal(
                f"MLX op {op.label!r} replayed {len(outputs)} outputs for "
                f"{len(capture.labels_raw)} labels."
            )
        return outputs[index]

    def close(self, replayed: Any, saved: Any) -> bool:
        """The MLX validator's own payload comparison."""

        return bool(_payloads_close(replayed, saved))


def _raw(op: Any) -> str:
    return str(getattr(op, "_label_raw", None) or op.label)


def replayer(trace: Any) -> _MLXReplayer:
    """Bind an MLX trace's call records for cone replay."""

    return _MLXReplayer(trace)


def vjp(fn: Callable[..., Any], primals: tuple[Any, ...], cotangent: Any) -> tuple[Any, ...]:
    """``mx.vjp`` of ``fn`` at ``primals`` against one output cotangent."""

    import mlx.core as mx  # deferred: optional dependency

    _, grads = mx.vjp(lambda *values: [fn(*values)], list(primals), [cotangent])
    return tuple(grads)


def jvp(
    fn: Callable[..., tuple[Any, ...]], primals: tuple[Any, ...], tangents: tuple[Any, ...]
) -> tuple[Any, ...]:
    """``mx.jvp`` of a tuple-valued ``fn``: one tangent per output."""

    import mlx.core as mx  # deferred: optional dependency

    _, outputs = mx.jvp(lambda *values: list(fn(*values)), list(primals), list(tangents))
    return tuple(outputs)


def to_numpy(value: Any) -> np.ndarray:
    """Host copy of an MLX array."""

    return np.array(value)


def from_numpy(array: np.ndarray, like: Any) -> Any:
    """An MLX array from ``array`` in the dtype of ``like``."""

    import mlx.core as mx  # deferred: optional dependency

    return mx.array(array).astype(like.dtype)


@contextmanager
def state_guard() -> Iterator[None]:
    """Restore ``mx.random.state`` after the probe."""

    import mlx.core as mx  # deferred: optional dependency

    state = mx.random.state
    saved = list(state)
    try:
        yield
    finally:
        state[:] = saved
