"""tinygrad cone replay for gradient influence probes.

Each captured UOp replays through the tinygrad validator's own
``_replay_uop_from_trace_graph``, with cone values substituted at the source
positions of the parents on the probed cone (its perturbation hook). The
validator realizes and detaches each replay; the probe binds the method to a
replay host that keeps the lazy tensor instead, so ``Tensor.gradient`` can
differentiate the rebuilt UOp graph. tinygrad has no forward-mode autodiff, so
projective probes sweep rows of reverse probes. Random UOps carry their seeds
in the graph, so replay touches no global random state.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np

from ..._utils._record_group_table import record_member
from ...backends.tinygrad._raw_labels import _final_to_raw_labels
from ...backends.tinygrad.backend import (
    TinygradBackend,
    _payloads_close,
    _saved_single_output,
    _source_matches_payload,
)
from . import ReplayRefusal, single_parent_value

__all__: list[str] = []

_STRIPPED = (
    "This tinygrad trace holds no runtime UOp records (a loaded or audit-only trace keeps "
    "none), so its ops cannot be replayed for a gradient probe. Capture the model again "
    "in this process."
)


class _LazyReplayHost:
    """The two backend hooks the validator's replay calls, minus the detaching copy."""

    @staticmethod
    def _tensor_from_uop(uop: Any) -> Any:
        from tinygrad import Tensor  # deferred: optional dependency

        return Tensor(uop)

    @staticmethod
    def _realized_copy(tensor: Any) -> Any:
        return tensor


class _TinygradReplayer:
    """Replay of single captured UOps from cone values and saved payloads."""

    def __init__(self, trace: Any) -> None:
        captures = tuple(getattr(trace, "tinygrad_uop_captures", ()) or ())
        if not captures:
            raise ReplayRefusal(_STRIPPED)
        self._captures = {capture.label_raw: capture for capture in captures}
        self._ops_by_raw = {getattr(op, "_label_raw", ""): op for op in trace._layer_list}
        self._final_to_raw = _final_to_raw_labels(self._ops_by_raw)
        self._hidden = dict(getattr(trace, "_selective_save_hidden_payloads", {}))

    def saved(self, op: Any) -> Any:
        """Return the op's saved payload (hidden replay payloads included)."""

        try:
            return _saved_single_output(op, self._hidden)
        except ValueError as exc:
            raise ReplayRefusal(f"Op {op.label!r}: {exc}") from exc

    def _saved_by_label(self, label: str) -> Any:
        return self.saved(self._ops_by_raw[self._final_to_raw.get(label, label)])

    def _replacements(self, op: Any, capture: Any, env: Mapping[str, Any]) -> dict[int, Any]:
        """Cone values by UOp source position, where the validator would bind a parent."""

        src = tuple(getattr(capture.uop, "src", ()) or ())
        positions = record_member(op, "parent_arg_positions", {}).get("args", {})
        replacements: dict[int, Any] = {}
        for position, label in positions.items():
            raw = self._final_to_raw.get(label, label)
            if raw not in env or not 0 <= position < len(src):
                continue
            if _source_matches_payload(src[position], self._saved_by_label(raw)):
                replacements[position] = env[raw]
        return replacements

    def replay(self, op: Any, env: Mapping[str, Any]) -> Any:
        """Replay the UOp that produced ``op`` as a lazy, differentiable tensor."""

        raw = str(getattr(op, "_label_raw", None) or op.label)
        capture = self._captures.get(raw)
        if capture is None:
            return single_parent_value(op, env, self._saved_by_label)
        try:
            return TinygradBackend._replay_uop_from_trace_graph(
                _LazyReplayHost(),  # type: ignore[arg-type]
                capture,
                op,
                self._ops_by_raw,
                replacements=self._replacements(op, capture, env),
                hidden_outputs_by_label=self._hidden,
            )
        except (KeyError, ValueError) as exc:
            raise ReplayRefusal(f"tinygrad op {op.label!r} cannot be replayed: {exc}") from exc

    def close(self, replayed: Any, saved: Any) -> bool:
        """The tinygrad validator's own payload comparison."""

        return bool(_payloads_close(replayed, saved))


def replayer(trace: Any) -> _TinygradReplayer:
    """Bind a tinygrad trace's UOp records for cone replay."""

    return _TinygradReplayer(trace)


def vjp(fn: Callable[..., Any], primals: tuple[Any, ...], cotangent: Any) -> tuple[Any, ...]:
    """``Tensor.gradient`` of ``fn`` at fresh copies of ``primals`` (unconnected ones get zeros)."""

    sources = tuple(from_numpy(to_numpy(value), value) for value in primals)
    output = fn(*sources)
    grads = output.gradient(*sources, gradient=cotangent)
    return tuple(grads)


def jvp(
    fn: Callable[..., tuple[Any, ...]], primals: tuple[Any, ...], tangents: tuple[Any, ...]
) -> None:
    """tinygrad has no forward-mode autodiff; the caller sweeps rows."""

    del fn, primals, tangents


def to_numpy(value: Any) -> np.ndarray:
    """Host copy of a tinygrad tensor."""

    return np.asarray(value.numpy())


def from_numpy(array: np.ndarray, like: Any) -> Any:
    """A tinygrad tensor from ``array`` on the device and in the dtype of ``like``."""

    from tinygrad import Tensor  # deferred: optional dependency

    return Tensor(np.ascontiguousarray(array), device=like.device, dtype=like.dtype)


@contextmanager
def state_guard() -> Iterator[None]:
    """Replay touches no tinygrad global random state; nothing to restore."""

    yield
