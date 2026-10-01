"""The non-differentiable summary-transform role (explorer P1; lane F25).

A SUMMARY transform is a user reducer that is contractually OUTSIDE the
autograd graph: TorchLens applies it to a DETACHED VIEW of the saved tensor
after the save point, and its output is exempt from the train-mode
differentiability validation (``transform_not_differentiable``). Integer and
bool outputs are permitted -- a histogram's counts, a bin index vector, a
boolean mask are all legal summary products. The exemption is NARROW: it
applies only to transforms that explicitly declare the role; an undeclared
detached transform under ``backward_ready=True`` keeps refusing, exactly as
before (the validator stays a tripwire for accidental graph breaks).

Contract for summary callables:

- The callable receives a DETACHED VIEW (``tensor.detach()``): storage is
  shared with the live forward tensor, so the callable MUST NOT mutate its
  input in place. Reduce, never rewrite.
- The output never rides the autograd graph and is never validated for
  differentiability; any dtype is allowed.
- Streaming serialization requirements are unchanged: a streamed transformed
  payload must still be a strided ``torch.Tensor``.
- With ``save_raw_activations=False`` the capture is REDUCE-ONLY: the raw
  clone is skipped entirely (explorer P2) and only the summary output is
  retained.

Spellings here are DOCUMENTED-UNSTABLE pending naming-session ratification
(the memo defers ``summary_transform=`` vs ``tl.Summary(fn)`` to the UI
sprint); the mechanism -- a role DECLARED on the callable, checked
duck-typed at the validation seams -- is the decision.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__tl_layer__ = "L2"

#: Attribute name carrying the declared transform role. Duck-typed on
#: purpose: foreign wrappers may declare the role without importing this
#: module, and the validation seams never need an isinstance dependency.
TRANSFORM_ROLE_ATTR = "_tl_transform_role"

#: The one non-default role token this seam recognizes.
SUMMARY_ROLE = "summary"


class SummaryTransform:
    """A callable wrapper declaring the non-differentiable summary role.

    Wraps a user reducer ``fn`` and applies it to ``tensor.detach()`` --
    the detached-view application point is INSIDE the wrapper so every
    invocation site (trace save, grad hooks, lookback retention, fastlog
    storage resolution) inherits it without per-site plumbing.

    Parameters
    ----------
    fn:
        Reducer applied to a detached view of each saved tensor. Must not
        mutate its input in place (the view shares storage with the live
        forward tensor).
    name:
        Optional display name for diagnostics; defaults to the callable's
        ``__name__`` when available.
    """

    _tl_transform_role = SUMMARY_ROLE

    def __init__(self, fn: Callable[[Any], Any], *, name: str | None = None) -> None:
        """Wrap ``fn`` as a declared summary reducer."""

        if not callable(fn):
            raise TypeError(
                f"summary(...) needs a callable reducer, got {type(fn).__name__}. "
                "Remedy: pass the reducer function itself, e.g. "
                "summary(lambda t: torch.histc(t.float(), bins=64))."
            )
        self.fn = fn
        self.name = name or getattr(fn, "__name__", type(fn).__name__)

    def __call__(self, tensor: Any) -> Any:
        """Apply the reducer to a detached view of ``tensor``.

        Non-tensor inputs (already-reduced values on replay paths) pass
        through to the reducer unchanged.
        """

        detach = getattr(tensor, "detach", None)
        return self.fn(tensor if detach is None else detach())

    def __repr__(self) -> str:
        """Return a diagnostic repr naming the wrapped reducer."""

        return f"SummaryTransform({self.name})"


def summary(fn: Callable[[Any], Any], *, name: str | None = None) -> SummaryTransform:
    """Declare ``fn`` as a non-differentiable summary reducer (explorer P1).

    Usable anywhere an ``activation_transform`` / ``grad_transform`` is
    accepted::

        import torchlens as tl
        from torchlens.observability import summary

        log = tl.trace(
            model, x,
            save=tl.options.SaveOptions(
                activation_transform=summary(lambda t: torch.histc(t.float(), 64)),
                save_raw_activations=False,   # reduce-only: raw clone skipped (P2)
            ),
            capture=tl.options.CaptureOptions(backward_ready=True),
        )

    Parameters
    ----------
    fn:
        Reducer applied to a detached view of each saved tensor.
    name:
        Optional display name for diagnostics.

    Returns
    -------
    SummaryTransform
        The declared-role wrapper (itself a plain callable).
    """

    return SummaryTransform(fn, name=name)


def transform_role(transform: Any) -> str:
    """Return the declared role of a transform callable.

    ``"summary"`` when the callable declares the summary role (duck-typed on
    :data:`TRANSFORM_ROLE_ATTR`); ``"standard"`` otherwise, including for
    ``None``. Unknown role tokens read as ``"standard"`` -- an undeclared or
    misdeclared transform keeps the full tripwire validation.
    """

    role = getattr(transform, TRANSFORM_ROLE_ATTR, None)
    return SUMMARY_ROLE if role == SUMMARY_ROLE else "standard"


def is_summary_transform(transform: Any) -> bool:
    """Return whether ``transform`` declares the summary role."""

    return transform_role(transform) == SUMMARY_ROLE


def ensure_summary_output_owns_storage(output: Any, live: Any) -> Any:
    """Clone ``output`` when it aliases ``live``'s storage (reduce-only P2).

    On the reduce-only save path the reducer runs on a detached VIEW of the
    live forward tensor: a reducer that returns a view of its input (an
    identity, a slice) would otherwise retain storage the forward still
    owns -- pinned memory the skip-the-clone path promised to release, and
    payload bytes a later in-place op could rewrite. Aliasing that cannot be
    proven absent fails SAFE (clone). Only top-level tensor outputs are
    checked; views nested inside containers are the documented residual of
    the summary contract ("reduce, never alias").

    Parameters
    ----------
    output:
        The reducer's return value.
    live:
        The live forward tensor the reducer viewed.

    Returns
    -------
    Any
        ``output`` itself when it provably owns its storage; a clone when it
        aliases ``live`` or aliasing is unprovable; non-tensor values pass
        through unchanged.
    """

    output_storage = getattr(output, "untyped_storage", None)
    live_storage = getattr(live, "untyped_storage", None)
    if output_storage is None or live_storage is None:
        return output
    try:
        aliases = output_storage().data_ptr() == live_storage().data_ptr()
    except (AttributeError, NotImplementedError, RuntimeError):
        aliases = True
    if not aliases:
        return output
    # The defensive copy is a TorchLens-internal tensor op running while
    # capture logging is live; pause so the clone is never itself captured.
    from .._state import pause_logging

    with pause_logging():
        return output.clone()


def retained_activation_bytes(trace_like: Any, op_like: Any) -> int:
    """Bytes actually RETAINED for one saved op (explorer P4 memory truth).

    ``saved_activation_memory`` documents itself as "just the payloads
    ``save=`` retained". Under a reduce-only capture (an activation
    transform with raw retention off) the retained payload is the
    TRANSFORMED output, so its bytes are the truth -- the raw nominal size
    would overstate a reduced capture by the entire activation footprint
    (a reduced gpt2 step reads ~832 MB when ~2 KB is held). Raw-retaining
    captures keep the historical raw-bytes accounting.

    Parameters
    ----------
    trace_like:
        The owning trace (read for the capture-level retention decision).
    op_like:
        One op record with ``has_saved_activation`` truth.

    Returns
    -------
    int
        Retained bytes for this op under the capture's retention decision.
    """

    transform = getattr(trace_like, "activation_transform", None)
    store_raw = transform is None or getattr(trace_like, "save_raw_activations", True)
    if store_raw:
        return int(getattr(op_like, "activation_memory", 0) or 0)
    return int(getattr(op_like, "transformed_activation_memory", 0) or 0)


__all__ = [
    "SUMMARY_ROLE",
    "TRANSFORM_ROLE_ATTR",
    "SummaryTransform",
    "ensure_summary_output_owns_storage",
    "is_summary_transform",
    "retained_activation_bytes",
    "summary",
    "transform_role",
]
