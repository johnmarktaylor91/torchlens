"""Framework shape priors behind the neutral input-shape search.

The search controller (:mod:`torchlens.debug._shape_search`) owns the loop, the
probe budget, the bounds, verification with ``tl.trace`` and the
:class:`~torchlens.debug._infer_report.GuessInputShapeReport` it returns. A
prior module answers only framework questions. Torch keeps its own priors in
:mod:`torchlens.debug._infer_input_shape`; every other backend has one module
here, named in :data:`PRIOR_MODULES` and looked up by backend name.

A prior module exposes these module-level functions (framework imports happen
inside the functions, never at module import):

``handles(model) -> bool``
    Whether ``model`` is this framework's model object. Must not import the
    framework: check ``sys.modules`` first and return ``False`` when the
    framework was never imported.
``lazy_reason(model) -> str | None``
    Why the model cannot be probed without initializing lazy or unbuilt layers
    (``None`` when it can). The search never initializes them in the user's
    object.
``candidates(model, request) -> Sequence[ShapeCandidate]``
    First-layer shape priors in the framework's native layout, best first.
    Empty when the model has no readable first layer.
``make_input(shape, dtype, value_range, seed) -> Any``
    One native input tensor of ``shape`` and ``dtype`` filled per
    ``value_range`` (``("uniform", lo, hi)`` or ``("randint", lo, hi)``),
    deterministic for ``seed`` and without touching the global RNG beyond
    what :func:`state_scope` restores.
``forward(model, example) -> Any``
    One plain forward pass (no capture), realized where the framework is lazy.
``classify_error(exc) -> ProbeError``
    Sort a forward exception into shape, dtype, device or blocker, with the
    framework's suggested dtype and measured feature counts when the message
    names them.
``executed_constraint(trace) -> ExecutedConstraint | None``
    The first constraining op the verification trace actually executed.
``state_scope(model) -> ContextManager[None]``
    Save the model's mode flags and the framework RNG, switch to inference
    mode, and restore both on exit (also on error).

Search arguments arrive in :class:`SearchRequest`; ``input_dtype`` stays the
caller's native dtype object (or ``None``) and the prior interprets it.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import MappingProxyType, ModuleType
from typing import Any, Literal, Protocol

ValueRange = tuple[str, float, float]
"""Synthetic value recipe: ``("uniform", lo, hi)`` or ``("randint", lo, hi)``."""

ProbeErrorKind = Literal["shape", "dtype", "device", "blocker"]

#: Backend name to prior module (relative to this package). Torch is absent on
#: purpose: its priors stay in ``_infer_input_shape``.
PRIOR_MODULES: MappingProxyType[str, str] = MappingProxyType(
    {
        "paddle": "paddle",
        "tinygrad": "tinygrad",
        "tf": "keras",
        "mlx": "mlx",
        "jax": "jax",
    }
)


@dataclass(frozen=True)
class SearchRequest:
    """Caller arguments the search passes to a prior, already validated."""

    batch_size: int
    input_dtype: Any
    channels: int | None
    spatial_rank: int | Literal["auto"]
    seq_len: int | None
    min_size: int
    max_size: int
    preferred_sizes: tuple[int, ...]


@dataclass(frozen=True)
class ShapeCandidate:
    """One first-layer prior: a shape to probe and what pins it.

    Attributes
    ----------
    kind:
        ``"linear"``, ``"conv"``, ``"embedding"``, ``"sequence"`` or
        ``"unknown"``; the search only side-searches ``"conv"``.
    shape:
        Shape to probe first, batch dimension included.
    dtype:
        Native dtype object of the framework.
    value_range:
        Synthetic value recipe.
    layout:
        Native layout string, e.g. ``"NC"``, ``"NCHW"``, ``"NHWC"``,
        ``"NCL"``, ``"NLC"``, ``"NCDHW"``, ``"NDHWC"`` or ``"NL"`` (tokens).
    spatial_axes:
        Axes the search may resize (square, equal sides) when the first probe
        fails with a shape error.
    flexible_dims:
        Axes the prior believes tolerate other sizes (batch excluded); the
        search keeps only the ones a grown probe confirms.
    determined:
        ``True`` when a weight shape pins every non-flexible axis, so a
        working probe is the unique shape up to the flexible axes. ``False``
        when the prior guessed an axis (rank, spatial side, sequence length)
        and the working shape is one candidate among others.
    constraining_module:
        Module address that pinned the prior, when known.
    constraining_op:
        Layer or op type that pinned the prior, when known.
    strategy:
        Strategy name reported when this candidate wins.
    min_side:
        Smallest spatial side the side search may try.
    """

    kind: str
    shape: tuple[int, ...]
    dtype: Any
    value_range: ValueRange
    layout: str
    spatial_axes: tuple[int, ...] = ()
    flexible_dims: tuple[int, ...] = ()
    determined: bool = False
    constraining_module: str | None = None
    constraining_op: str | None = None
    strategy: str = "layer_prior"
    min_side: int = 1


@dataclass(frozen=True)
class ProbeError:
    """A classified forward-probe exception."""

    kind: ProbeErrorKind
    message: str
    suggested_dtype: Any = None
    got_features: int | None = None
    target_features: int | None = None


@dataclass(frozen=True)
class ExecutedConstraint:
    """The first constraining op a verification trace executed."""

    kind: str
    module: str | None
    op: str | None
    source_line: str | None
    input_shape: tuple[int, ...] | None


class ShapePriors(Protocol):
    """Typing view of a prior module (modules satisfy protocols in mypy)."""

    def handles(self, model: Any) -> bool:
        """Whether ``model`` belongs to this framework (no framework import)."""

    def lazy_reason(self, model: Any) -> str | None:
        """Why probing would initialize lazy layers, or ``None``."""

    def candidates(self, model: Any, request: SearchRequest) -> Sequence[ShapeCandidate]:
        """First-layer priors in native layout, best first."""

    def make_input(
        self, shape: tuple[int, ...], dtype: Any, value_range: ValueRange, seed: int
    ) -> Any:
        """One deterministic native input tensor."""

    def forward(self, model: Any, example: Any) -> Any:
        """One plain realized forward pass."""

    def classify_error(self, exc: Exception) -> ProbeError:
        """Sort a forward exception."""

    def executed_constraint(self, trace: Any) -> ExecutedConstraint | None:
        """The first constraining op the trace executed."""

    def state_scope(self, model: Any) -> AbstractContextManager[None]:
        """Save and restore model mode flags and framework RNG."""


def resolve_priors(model: Any) -> tuple[str, ShapePriors] | None:
    """Find the prior module whose ``handles`` accepts ``model``.

    A prior module that is not present yet (a sibling lane has not landed it)
    is skipped; any other import error propagates.

    Parameters
    ----------
    model:
        The caller's model object.

    Returns
    -------
    tuple[str, ShapePriors] | None
        ``(backend name, prior module)``, or ``None`` when no prior handles it.
    """

    for backend_name, module_name in PRIOR_MODULES.items():
        qualified = f"{__name__}.{module_name}"
        try:
            module: ModuleType = importlib.import_module(qualified)
        except ModuleNotFoundError as exc:
            if exc.name == qualified:
                continue
            raise
        priors: ShapePriors = module  # type: ignore[assignment]
        if priors.handles(model):
            return backend_name, priors
    return None


__all__ = [
    "PRIOR_MODULES",
    "ExecutedConstraint",
    "ProbeError",
    "ProbeErrorKind",
    "SearchRequest",
    "ShapeCandidate",
    "ShapePriors",
    "ValueRange",
    "resolve_priors",
]
