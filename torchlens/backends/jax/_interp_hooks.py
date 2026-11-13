"""Interpreter hooks for the JAX jaxpr interpreter (retention, halting, backward entry).

The interpreter in ``jaxpr.py`` appends every capture (primitive equations, region
boundaries and region projections, at any nesting depth) to one list. Hooks run at that
append, so one call site covers scan, cond, while and region captures as well as the main
loop:

* Retention. ``retention_policy(session)`` from ``_retention.py`` (owned by the capture
  lane; called once per run and only when that module exists) returns ``keep(index) ->
  bool`` over capture indexes. A primitive-equation capture it does not keep stores
  ``jax.ShapeDtypeStruct`` placeholders for its input and output values (same tuple
  lengths) and its index is recorded in ``released``; the interpreter then frees frame
  environment entries after their last use (:func:`dead_after`), so a released value is
  dropped once nothing later reads it. Region and control-flow decision captures always
  keep their values.
* Halting. ``halt(capture) -> bool`` is checked after the capture is stored (save, then
  halt, as torch does). A true result raises :class:`InterpreterHalt`, which unwinds every
  nested frame before any further primitive binds.

This module imports nothing from the backend's import cycle; captures are duck-typed.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

RetentionPolicy = Callable[[int], bool]
HaltCheck = Callable[[Any], bool]

COMBINER_JAXPR_PRIMITIVES = frozenset(
    {
        "reduce",
        "reduce_window",
        "scatter",
        "scatter-add",
        "scatter_add",
        "scatter-apply",
        "scatter_apply",
        "scatter-max",
        "scatter_max",
        "scatter-min",
        "scatter_min",
        "scatter-mul",
        "scatter_mul",
        "select_and_gather_add",
        "select_and_scatter",
    }
)
"""Primitives whose nested jaxpr parameter is an element combiner, not a call.

They bind as ordinary primitives (the combiner is part of the primitive's parameters).
Only interpreter runs with ``bind_combiners=True`` accept them; the forward capture keeps
its nested-jaxpr refusal."""


class InterpreterHalt(Exception):
    """Raised at a capture append when the halt check matches; unwinds the interpreter."""

    def __init__(self, capture: Any) -> None:
        """Remember the capture the stop matched (the halt frontier)."""

        super().__init__("interpreter halted at the stop frontier")
        self.capture = capture


@dataclass(frozen=True)
class InterpreterHooks:
    """Optional hooks for one interpreter run.

    Parameters
    ----------
    session
        Capture session handed to ``retention_policy``; ``None`` disables retention.
    halt
        ``halt(capture) -> bool``; a true result stops the run after that capture.
    bind_combiners
        Accept :data:`COMBINER_JAXPR_PRIMITIVES` (backward pullback programs use them).
    """

    session: Any = None
    halt: HaltCheck | None = None
    bind_combiners: bool = False


def _retention_module() -> Any | None:
    """Return the capture lane's ``_retention`` module when it is installed."""

    try:
        return importlib.import_module("torchlens.backends.jax._retention")
    except ModuleNotFoundError as exc:
        if exc.name != "torchlens.backends.jax._retention":
            raise
        return None


class HookedCaptures(list[Any]):
    """Capture list whose ``append`` applies retention, then the halt check."""

    def __init__(self, hooks: InterpreterHooks | None) -> None:
        """Resolve the retention policy once for this run."""

        super().__init__()
        self.released: set[int] = set()
        self._halt = None if hooks is None else hooks.halt
        self.keep: RetentionPolicy | None = None
        session = None if hooks is None else hooks.session
        module = None if session is None else _retention_module()
        if module is not None:
            self.keep = module.retention_policy(session)

    def append(self, capture: Any) -> None:
        """Store one capture (released when the policy drops it), then check the halt."""

        if self.keep is not None and capture.kind == "primitive" and not self.keep(len(self)):
            self.released.add(len(self))
            capture = replace(
                capture,
                input_values=_placeholders(capture.input_values),
                output_values=_placeholders(capture.output_values),
            )
        super().append(capture)
        if self._halt is not None and self._halt(capture):
            raise InterpreterHalt(capture)

    def __delitem__(self, key: Any) -> None:
        """Roll back captures (a region fallback drops a partial unroll) and their marks."""

        super().__delitem__(key)
        self.released = {index for index in self.released if index < len(self)}


def binds_as_primitive(primitive_name: str, hooks: InterpreterHooks | None) -> bool:
    """Return whether a nested-jaxpr equation may bind as a plain primitive."""

    return (
        hooks is not None and hooks.bind_combiners and (primitive_name in COMBINER_JAXPR_PRIMITIVES)
    )


def _placeholders(values: tuple[Any, ...]) -> tuple[Any, ...]:
    """Return one ``jax.ShapeDtypeStruct`` per value (shape and dtype kept, data dropped)."""

    import jax

    return tuple(
        jax.ShapeDtypeStruct(getattr(v, "shape", ()), getattr(v, "dtype", type(v))) for v in values
    )


def dead_after(jaxpr: Any, captures: HookedCaptures) -> Mapping[int, tuple[Any, ...]]:
    """Map each equation index to the frame variables last read there.

    Empty unless a retention policy is active. Frame outputs are never freed; literals
    and variables read by no equation are not listed.
    """

    if captures.keep is None:
        return {}
    keep = {id(var) for var in jaxpr.outvars}
    last: dict[int, tuple[int, Any]] = {}
    for index, eqn in enumerate(jaxpr.eqns):
        for var in eqn.invars:
            if type(var).__name__ != "Literal":
                last[id(var)] = (index, var)
    dead: dict[int, list[Any]] = {}
    for var_id, (index, var) in last.items():
        if var_id not in keep:
            dead.setdefault(index, []).append(var)
    return {index: tuple(vars_) for index, vars_ in dead.items()}
