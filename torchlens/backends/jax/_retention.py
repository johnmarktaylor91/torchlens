"""Payload retention decided at JAX capture time.

The JAX interpreter asks :func:`retention_policy` once per capture for a keep
test. The test takes the capture index of a primitive equation (its position
in the flattened capture list, ``len(captures)`` when its capture is
appended) and answers whether that equation's concrete input and output
values must stay in its capture. When it answers ``False`` the interpreter
may free the values after their last use in the jaxpr; the capture keeps only
abstract shape and dtype placeholders.

A session carries its policy as its ``jax_retention`` attribute; a session
without one keeps every value, which is the plain capture's behavior.
"""

from __future__ import annotations

import dataclasses
import weakref
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

KeepTest = Callable[[int], bool]


@dataclass(frozen=True)
class Retention:
    """Which equation captures keep concrete values.

    Parameters
    ----------
    keep
        Capture indexes whose values stay. ``None`` keeps every value.
    """

    keep: frozenset[int] | None

    def keeps(self, capture_index: int) -> bool:
        """Return whether the capture at ``capture_index`` keeps its values."""

        return self.keep is None or capture_index in self.keep


KEEP_ALL = Retention(keep=None)
KEEP_NONE = Retention(keep=frozenset())


def keep_only(indexes: Iterable[int]) -> Retention:
    """Return a retention keeping the values of the listed capture indexes."""

    return Retention(keep=frozenset(int(index) for index in indexes))


def retention_of(session: Any) -> Retention:
    """Return the retention a session carries as ``jax_retention`` (keep all when none)."""

    retention = getattr(session, "jax_retention", None)
    return retention if isinstance(retention, Retention) else KEEP_ALL


def retention_policy(session: Any) -> KeepTest:
    """Return the keep test the interpreter applies to each primitive equation.

    Parameters
    ----------
    session
        The capture's open equation session (``JaxEquationSession``).

    Returns
    -------
    Callable[[int], bool]
        ``keep(capture_index)``: ``True`` when that equation's concrete values
        must stay in its capture.
    """

    return retention_of(session).keeps


_PENDING: ContextVar[Retention | None] = ContextVar("torchlens_jax_pending_retention", default=None)


@contextmanager
def pending_retention(retention: Retention) -> Iterator[None]:
    """Make the next JAX capture in this context release values ``retention`` drops.

    Parameters
    ----------
    retention
        Capture indexes whose values stay (see :func:`keep_only`).

    Yields
    ------
    None
        Control while the capture runs.
    """

    token = _PENDING.set(retention)
    try:
        yield
    finally:
        _PENDING.reset(token)


def pending_release() -> Release | None:
    """Return a fresh :class:`Release` for the pending retention, if one is set."""

    retention = _PENDING.get()
    return None if retention is None else Release(retention)


def is_released(value: Any) -> bool:
    """Return whether ``value`` is a released placeholder (or a tuple of only those)."""

    import jax

    if isinstance(value, tuple):
        return bool(value) and all(isinstance(item, jax.ShapeDtypeStruct) for item in value)
    return isinstance(value, jax.ShapeDtypeStruct)


def _placeholder(value: Any) -> Any:
    """Return the shape and dtype of ``value`` without its data."""

    import jax

    return jax.ShapeDtypeStruct(getattr(value, "shape", ()), getattr(value, "dtype", type(value)))


def _weak(value: Any) -> Callable[[], Any] | None:
    """Return a weak reference to ``value``, or ``None`` when it cannot hold one."""

    try:
        return weakref.ref(value)
    except TypeError:
        return None


class Release:
    """Release the values of unkept primitive captures, one equation late.

    The interpreter stores every capture with concrete values (its own retention
    keeps all), so the equation hook still sees them. Before each later equation
    binds, this replaces the values of unkept primitive captures with shape and
    dtype placeholders. One placeholder object stands for one released value
    everywhere: the producer's output and every later consumer's input. Edges
    resolved by value identity therefore survive the release. A consumer that
    keeps its values (a kept primitive or a region capture) still reads a
    released producer's placeholder as its input, so its replay input is
    missing exactly as on a torch trace whose parent was not saved.

    Parameters
    ----------
    retention
        Capture indexes whose values stay.
    """

    def __init__(self, retention: Retention) -> None:
        """Start with nothing processed or released."""

        self.retention = retention
        self.released: set[int] = set()
        self._processed: list[Any] = []
        self._shared: dict[int, tuple[Callable[[], Any] | None, Any]] = {}

    def wrap(self, hook: Callable[..., Any]) -> Callable[..., Any]:
        """Return ``hook`` preceded by a release of every capture stored since the last call."""

        def released_hook(captures: list[Any], *args: Any) -> Any:
            self.release(captures)
            return hook(captures, *args)

        return released_hook

    def release(self, captures: list[Any]) -> None:
        """Replace unprocessed captures in ``captures`` by their released copies."""

        start = min(len(self._processed), len(captures))
        while start and captures[start - 1] is not self._processed[start - 1]:
            start -= 1
        del self._processed[start:]
        self.released = {index for index in self.released if index < start}
        for index in range(start, len(captures)):
            captures[index] = self._released_copy(index, captures[index])
            self._processed.append(captures[index])

    def finish(self, result: Any) -> Any:
        """Release the captures stored after the last hook call and repackage ``result``.

        A frame output stays alive with the result, so a released capture that
        produced one gets its value back and counts as kept; module-return
        probes and the output marks then still find its producer by identity.
        """

        from .jaxpr import JaxEquationCapture

        ordered = list(result.captures)
        self.release(ordered)
        restore = {}
        for value in result.outputs:
            placeholder = self._shared_placeholder(value)
            if placeholder is not None:
                restore[id(placeholder)] = value
        if restore:
            ordered = [self._restored(index, c, restore) for index, c in enumerate(ordered)]
        captures = tuple(ordered)
        return dataclasses.replace(
            result,
            captures=captures,
            equations=tuple(c for c in captures if isinstance(c, JaxEquationCapture)),
            released_capture_indices=frozenset(self.released),
        )

    def _released_copy(self, index: int, capture: Any) -> Any:
        """Return ``capture`` with shared placeholders in place of released values."""

        inputs = tuple(self._shared_or_value(value) for value in capture.input_values)
        outputs = capture.output_values
        if capture.kind == "primitive" and not self.retention.keeps(index):
            outputs = tuple(self._register(value) for value in outputs)
            self.released.add(index)
        else:
            for value in outputs:
                self._shared.pop(id(value), None)
        if _same(inputs, capture.input_values) and outputs is capture.output_values:
            return capture
        return dataclasses.replace(capture, input_values=inputs, output_values=outputs)

    def _register(self, value: Any) -> Any:
        """Return the placeholder standing for ``value`` from now on."""

        placeholder = _placeholder(value)
        self._shared[id(value)] = (_weak(value), placeholder)
        return placeholder

    def _shared_placeholder(self, value: Any) -> Any:
        """Return the placeholder registered for this exact ``value``, else ``None``."""

        entry = self._shared.get(id(value))
        if entry is None:
            return None
        ref, placeholder = entry
        if ref is not None and ref() is not value:
            del self._shared[id(value)]
            return None
        return placeholder

    def _shared_or_value(self, value: Any) -> Any:
        """Return the placeholder registered for ``value``, else ``value`` itself."""

        placeholder = self._shared_placeholder(value)
        return value if placeholder is None else placeholder

    def _restored(self, index: int, capture: Any, restore: dict[int, Any]) -> Any:
        """Return ``capture`` with restored frame outputs in place of their placeholders."""

        def swap(values: tuple[Any, ...]) -> tuple[Any, ...]:
            return tuple(restore.get(id(value), value) for value in values)

        inputs, outputs = swap(capture.input_values), swap(capture.output_values)
        if any(id(value) in restore for value in capture.output_values):
            self.released.discard(index)
        if _same(inputs, capture.input_values) and _same(outputs, capture.output_values):
            return capture
        return dataclasses.replace(capture, input_values=inputs, output_values=outputs)


def _same(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    """Return whether two value tuples hold the same objects (identity, never ``==``)."""

    return len(left) == len(right) and all(a is b for a, b in zip(left, right, strict=True))
