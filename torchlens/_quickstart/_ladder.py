"""The trace() entry door for the declared/inferred/kwargs-only rungs.

Extracted verbatim from ``torchlens._user_funcs``: the
capture-shaping-kwarg default table, its comparator, and the ladder capture
driver ``trace_via_ladder`` that ``tl.trace`` re-enters through. The gold rung
never comes through this module -- ``trace()`` handles a real input inline.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, cast

from torch import nn

from .._deprecations import MISSING
from .._errors import InvalidArgumentError
from .._utils._package_frames import external_warning_stacklevel
from ..errors._base import TorchLensWarning
from ._plan import RUNG_INFERRED, resolve_rung
from ._resolve import attach_provenance, resolve_inputs

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

__tl_layer__ = "FACADE"

#: trace() kwargs whose non-default values make the inference search's
#: verified trace unusable as-is (it was captured metadata-complete,
#: eval-mode, save='all', no interventions); any of these set means the
#: zero-argument rung RECAPTURES with the inferred concrete input instead of
#: reusing the verified trace (an incompatible requested
#: capture recaptures with disclosure, never silently serves the eval trace).
_LADDER_DEFAULTS: dict[str, Any] = {
    "capture": None,
    "save": MISSING,
    "intervene": None,
    "stop_after": None,
    "storage": None,
    "facets": MISSING,
    "episode": None,
    "chunk_paths": MISSING,
    "backend": None,
}


def _ladder_kwargs_are_default(trace_kwargs: dict[str, Any]) -> bool:
    """Return whether every capture-shaping trace() kwarg is at its default."""

    for name, default in _LADDER_DEFAULTS.items():
        value = trace_kwargs.get(name, default)
        if value is default:
            continue
        # int/str defaults compare by value; identity-compared sentinels
        # (MISSING/None) already matched above. == on arbitrary user objects
        # is deliberately avoided (tensors overload it).
        if isinstance(default, (int, str)) and isinstance(value, (int, str)) and value == default:
            continue
        return False
    return True


def _trace_via_ladder(
    model: nn.Module,
    input_args: Any,
    input_kwargs: dict[Any, Any] | None,
    input_size: Any,
    trace_kwargs: dict[str, Any],
) -> Trace:
    """Resolve the declared/inferred/kwargs-only rungs and capture.

    The declared rung synthesizes concrete tensors (disclosed) and re-enters
    ``trace()``. The inferred rung consumes the inference search's exact
    verified trace when every capture-shaping kwarg is at its default (in-call
    reuse: capture counter stays at 1); otherwise it recaptures
    from the inferred concrete input with the requested options, disclosed in
    the provenance record's execution policy.

    Parameters
    ----------
    model:
        The model to capture.
    input_args:
        The caller's real positional input, when supplied.
    input_kwargs:
        The caller's real keyword input, when supplied (kwargs-only calls
        normalize through this door).
    input_size:
        The declared-shape argument, or ``None``.
    trace_kwargs:
        Every other ``trace()`` keyword, forwarded verbatim on re-entry.

    Returns
    -------
    Trace
        The finished trace, carrying its input-provenance record.
    """

    from .._user_funcs import trace

    backend = trace_kwargs.get("backend")
    if (
        backend is not None
        and str(backend) not in ("torch",)
        and resolve_rung(input_args, input_kwargs, input_size) != 1
    ):
        raise InvalidArgumentError(
            f"input_shape= and zero-argument input resolution are torch-only; "
            f"backend={str(backend)!r} requires a real input.",
            code="input_ladder_backend_unsupported",
            remedy="pass a real input for non-torch backends",
        )
    resolved = resolve_inputs(model, input_args, input_kwargs, input_size, verb="trace")
    plan = resolved.plan
    if plan.verified_trace is not None and _ladder_kwargs_are_default(trace_kwargs):
        log = cast("Trace", plan.verified_trace)
        _attach_and_disclose(log, resolved.provenance)
        return log
    if len(plan.input_args) == 1:
        concrete: Any = plan.input_args[0]
    else:
        concrete = list(plan.input_args)
    log = trace(model, concrete, input_kwargs=dict(plan.input_kwargs) or None, **trace_kwargs)
    _attach_and_disclose(log, resolved.provenance)
    return log


def _attach_and_disclose(log: Trace, provenance: Any) -> None:
    """Attach the input provenance and warn when the input was inferred, not given.

    ``tl.trace(model)`` with no input infers a shape and fills it with random
    values. The provenance record discloses it on the trace; the warning makes
    a forgotten input argument visible at the call (code ``trace_input_inferred``).
    A declared ``input_shape=`` asked for random values, so it does not warn.
    """

    attach_provenance(log, provenance)
    if provenance.rung != RUNG_INFERRED:
        return
    shapes = ", ".join(str(tuple(facts.shape)) for facts in provenance.tensors) or "unknown"
    warnings.warn(
        TorchLensWarning(
            "tl.trace was called with no input, so TorchLens inferred the input shape "
            f"{shapes} and filled it with random values. The trace records the model "
            "running on that random input, not on your data. Remedy: pass a real input, "
            "tl.trace(model, x), or declare the shape with input_shape= to choose random "
            "values on purpose",
            code="trace_input_inferred",
            inferred_shapes=shapes,
        ),
        stacklevel=external_warning_stacklevel(),
    )
