"""Paddle cone replay for gradient influence probes.

Each captured call replays through the Paddle validator's own
``_rebuild_inputs`` and ``capture.func``, with cone values standing in for the
saved payloads of the parents on the probed cone. The validator replays under
``no_grad``; the probe replays the same call with gradients enabled, keeping
the validator's detached parameter copies as constants, and differentiates it
with ``paddle.grad``. Paddle has no forward-mode autodiff in dynamic mode, so
projective probes sweep rows of reverse probes. The state guard restores the
Paddle random state, which a replayed random call would advance.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import numpy as np

from ... import _state
from ...backends._validation_shared import ops_by_label as _ops_by_label
from ...backends.paddle.backend import _first_output_path
from ...backends.paddle.validation import (
    _payloads_close,
    _rebuild_inputs,
    _saved_payload,
    _value_at_path,
)
from . import ReplayRefusal, single_parent_value

__all__: list[str] = []

_STRIPPED = (
    "This Paddle trace holds no runtime call records (a loaded trace keeps none), so its "
    "ops cannot be replayed for a gradient probe. Capture the model again in this process."
)


class _PaddleReplayer:
    """Replay of single captured Paddle calls from cone values and saved payloads."""

    def __init__(self, trace: Any) -> None:
        captures = tuple(getattr(trace, "_paddle_op_captures", ()) or ())
        if not captures:
            raise ReplayRefusal(_STRIPPED)
        self._ops_by_label = _ops_by_label(trace)
        self._captures = {str(getattr(item, "label_raw", "")): item for item in captures}

    def saved(self, op: Any) -> Any:
        """Return the op's saved Paddle payload."""

        try:
            return _saved_payload(op)
        except ValueError as exc:
            raise ReplayRefusal(f"Op {op.label!r}: {exc}") from exc

    def _saved_by_label(self, label: str) -> Any:
        return self.saved(self._ops_by_label[label])

    def replay(self, op: Any, env: Mapping[str, Any]) -> Any:
        """Replay the call that produced ``op`` with gradients enabled."""

        import paddle  # deferred: optional dependency

        capture = self._captures.get(str(getattr(op, "_label_raw", None) or op.label))
        if capture is None:
            return single_parent_value(op, env, self._saved_by_label)
        if getattr(capture, "intervention", None) is not None or capture.func is None:
            raise ReplayRefusal(
                f"Paddle op {op.label!r} carries an intervention or has no replay "
                "function, so a gradient probe cannot replay it."
            )
        view = {
            **self._ops_by_label,
            **{
                key: SimpleNamespace(out=value, has_saved_activation=True)
                for key, value in env.items()
            },
        }
        rebuilt = _rebuild_inputs(capture, view)
        if not rebuilt.ok:
            raise ReplayRefusal(
                f"Paddle op {op.label!r} cannot be rebuilt for replay: {rebuilt.reason}"
            )
        with _state.pause_logging(), paddle.set_grad_enabled(True):
            replayed = capture.func(*rebuilt.args, **rebuilt.kwargs)
        return _value_at_path(replayed, _first_output_path(capture))

    def close(self, replayed: Any, saved: Any) -> bool:
        """The Paddle validator's own payload comparison."""

        return bool(_payloads_close(replayed.detach(), saved))


def replayer(trace: Any) -> _PaddleReplayer:
    """Bind a Paddle trace's call records for cone replay."""

    return _PaddleReplayer(trace)


def vjp(fn: Callable[..., Any], primals: tuple[Any, ...], cotangent: Any) -> tuple[Any, ...]:
    """``paddle.grad`` of ``fn`` at ``primals``; an unused source gets zeros."""

    import paddle  # deferred: optional dependency

    sources = []
    for value in primals:
        source = value.detach().clone()
        source.stop_gradient = False
        sources.append(source)
    with paddle.set_grad_enabled(True):
        output = fn(*sources)
        if output.stop_gradient:
            return tuple(paddle.zeros_like(source) for source in sources)
        grads = paddle.grad([output], sources, grad_outputs=[cotangent], allow_unused=True)
    return tuple(
        paddle.zeros_like(source) if grad is None else grad
        for source, grad in zip(sources, grads, strict=True)
    )


def jvp(
    fn: Callable[..., tuple[Any, ...]], primals: tuple[Any, ...], tangents: tuple[Any, ...]
) -> None:
    """Paddle dynamic mode has no forward-mode autodiff; the caller sweeps rows."""

    del fn, primals, tangents


def to_numpy(value: Any) -> np.ndarray:
    """Host copy of a Paddle tensor."""

    return np.asarray(value.detach().numpy())


def from_numpy(array: np.ndarray, like: Any) -> Any:
    """A Paddle tensor from ``array`` in the dtype and place of ``like``."""

    import paddle  # deferred: optional dependency

    return paddle.to_tensor(array, dtype=like.dtype, place=like.place)


@contextmanager
def state_guard() -> Iterator[None]:
    """Restore the Paddle random state after the probe."""

    import paddle  # deferred: optional dependency

    saved = paddle.get_rng_state()
    try:
        yield
    finally:
        paddle.set_rng_state(saved)
