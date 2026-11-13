"""Post-finalization selective-save helpers for non-torch backends."""

from __future__ import annotations

__tl_layer__ = "L3"

import warnings
from collections.abc import Callable
from typing import Any, cast

from .._capture.finalize.epilogue import _render_layer_visualizers
from .._capture.projections import _build_record_context
from .._ir.selector_eval import first_selector_kind_outside, split_followed_by_conjunction
from .._quantities import Bytes
from .._selection.selectors import BaseSelector
from .._utils._record_group_table import record_member
from ..backends.registry import BackendUnsupportedError
from ._post_plan import BackendPostPlan, split_post_options
from ._post_predicate import _STATIC_SELECTOR_KINDS, post_save_keep_labels
from ._post_shaping import apply_save_shaping, reshapes_payloads
from .saved_summary import refresh_pinned_autograd_memory, refresh_saved_module_call_count

#: Selector kinds a finished capture can evaluate after the fact: the static
#: kinds plus value-dependent ``tl.where`` and the lookback selectors.
POST_SAVE_SELECTOR_KINDS = _STATIC_SELECTOR_KINDS | frozenset(
    {"predicate", "preceded_by", "followed_by"}
)

# Intervention selectors resolve against ops that can be REWRITTEN mid-forward;
# the synthetic ``output`` pseudo-op only exists after the forward returns, so
# it is deliberately excluded here. Backends alias this set instead of
# re-spelling the 8 kinds (the drop of ``output`` is intent, not drift).
_STATIC_INTERVENTION_SELECTOR_KINDS = _STATIC_SELECTOR_KINDS - frozenset({"output"})


def reject_selector_outside_kinds(
    predicate: Callable[[Any], Any] | BaseSelector,
    *,
    allowed: frozenset[str],
    backend_name: str,
) -> None:
    """Reject selector trees containing kinds outside an explicit backend allowlist.

    Parameters
    ----------
    predicate
        Candidate public save predicate.
    allowed
        Selector kinds accepted by the backend, including any allowed boolean
        composite kinds.
    backend_name
        Backend name used in diagnostic errors.

    Returns
    -------
    None
        Returns when the whole predicate tree is allowed.
    """

    if not isinstance(predicate, BaseSelector):
        raise BackendUnsupportedError(
            f"{backend_name} backend supports trace(save=...) only for static-label selectors "
            "tl.func, tl.label, tl.label_contains and boolean composites (&, |, -, ~) of those. "
            "Value-dependent predicates need concrete activation values at predicate time. "
            "MLX lazy evaluation defers RecordContext.tensor_requires_grad, is_scalar_bool, "
            "and bool_value without per-op evaluation; use the PyTorch backend for "
            "value-dependent predicates."
        )
    unsupported = first_selector_kind_outside(predicate, allowed=allowed)
    if unsupported is None:
        return
    if unsupported in {"module", "in_module"}:
        raise BackendUnsupportedError(
            f"{backend_name} backend cannot use trace(save=...) selector kind "
            f"{unsupported!r}: module hierarchy required, not yet on {backend_name}."
        )
    if unsupported == "predicate":
        raise BackendUnsupportedError(
            f"{backend_name} backend does not support value-dependent trace(save=...) "
            "predicates from tl.where. Static-label save= on this backend is limited to "
            "tl.func, tl.label, tl.label_contains and boolean composites (&, |, -, ~) of those."
        )
    raise BackendUnsupportedError(
        f"{backend_name} backend supports trace(save=...) only for static-label selectors "
        "tl.func, tl.label, tl.label_contains and boolean composites (&, |, -, ~) of those; "
        f"unsupported selector kind {unsupported!r}."
    )


def apply_static_label_save_policy(
    trace: Any,
    predicate: Callable[[Any], Any] | BaseSelector | None,
    *,
    backend_name: str,
    lookback: int = 0,
    lookback_payload_policy: str = "metadata_only",
) -> None:
    """Filter public saved activations with a ``save=`` predicate on a finished trace.

    Static-label selectors are evaluated on each ``Op``. Value-dependent
    predicates (plain callables, ``tl.where``) and lookback selectors
    (``tl.preceded_by``, ``candidate & tl.followed_by``) are evaluated on the
    rebuilt torch ``RecordContext`` of each op, with the realized payload as
    ``tensor`` (:func:`._post_predicate.post_save_keep_labels`).

    Parameters
    ----------
    trace
        Finalized trace whose public saved activation surface should be filtered.
    predicate
        Predicate accepted by ``trace(save=...)``. ``None`` preserves full-save capture.
    backend_name
        Backend name used in diagnostic errors.
    lookback
        Predicate history size for lookback selectors (0 means torch's default).
    lookback_payload_policy
        Torch lookback payload policy.

    Returns
    -------
    None
        Public payload fields and saved-summary counters are updated in place.
    """

    if predicate is None:
        return
    _reject_non_static_save_predicate(predicate, backend_name=backend_name)
    if _is_static_label_selector(predicate):
        keep = {op.label for op in trace._layer_list if bool(predicate(op))}
    else:
        keep = post_save_keep_labels(
            trace,
            predicate,
            split=split_followed_by_conjunction(predicate),
            build_context=_build_record_context,
            lookback_options=(lookback, lookback_payload_policy),
        )
    apply_save_keep_set(trace, keep, backend_name=backend_name)


def apply_save_keep_set(trace: Any, keep: set[str], *, backend_name: str) -> None:
    """Hide every public payload whose op label is not in ``keep``.

    Parameters
    ----------
    trace
        Finalized full-save trace.
    keep
        Final op labels whose payloads stay public.
    backend_name
        Backend name used in the zero-match diagnostic.

    Returns
    -------
    None
        Hidden payloads stay replay-visible; saved-summary counters are refreshed.
    """

    hidden_payloads = _hidden_payloads_by_label(trace)
    for op in trace._layer_list:
        if op.label not in keep:
            _drop_public_activation_payload(op)
    if not keep:
        warn_zero_match_save_predicate(backend_name, stacklevel=4)
    trace._selective_save_hidden_payloads = hidden_payloads
    _refresh_saved_activation_summary(trace)


def _is_static_label_selector(predicate: Any) -> bool:
    """Return whether ``predicate`` is a selector of static-label kinds only."""

    return (
        isinstance(predicate, BaseSelector)
        and first_selector_kind_outside(predicate, allowed=_STATIC_SELECTOR_KINDS) is None
    )


def needs_post_save_predicate(save: Any) -> bool:
    """Return whether ``save`` is a value-dependent or lookback predicate.

    Parameters
    ----------
    save
        Public ``save=`` value after label routing.

    Returns
    -------
    bool
        ``True`` for a plain callable, or a selector tree that uses a kind beyond
        the static label kinds while staying inside :data:`POST_SAVE_SELECTOR_KINDS`.
    """

    if save is None or isinstance(save, (str, list, tuple)):
        return False
    if not isinstance(save, BaseSelector):
        return callable(save)
    if _is_static_label_selector(save):
        return False
    return first_selector_kind_outside(save, allowed=POST_SAVE_SELECTOR_KINDS) is None


def split_backend_post_options(kwargs: dict[str, Any], spec: Any) -> BackendPostPlan | None:
    """Move options a finished non-torch capture applies out of the dispatch bundle.

    Parameters
    ----------
    kwargs
        Mutable public dispatch bundle (see :func:`._post_plan.split_post_options`).
    spec
        Resolved backend spec; torch keeps every option at capture time.

    Returns
    -------
    BackendPostPlan | None
        Plan for :func:`apply_backend_post_plan`, or ``None`` when nothing moved.
    """

    save = kwargs.get("save")
    post_save = save if needs_post_save_predicate(save) else None
    return split_post_options(kwargs, str(spec.name), post_save=post_save)


def apply_backend_post_plan(trace: Any, plan: BackendPostPlan | None, spec: Any) -> Any:
    """Apply a post-processing plan to a finished full-save non-torch trace.

    Parameters
    ----------
    trace
        Finished trace returned by the backend.
    plan
        Plan from :func:`split_backend_post_options`, or ``None``.
    spec
        Resolved backend spec; its name labels diagnostics and picks the payload codec.

    Returns
    -------
    Any
        The same trace, filtered and shaped in place.
    """

    if plan is None:
        return trace
    backend_name = str(spec.name)
    apply_static_label_save_policy(
        trace,
        plan.save,
        backend_name=backend_name,
        lookback=plan.lookback,
        lookback_payload_policy=plan.lookback_payload_policy,
    )
    if reshapes_payloads(plan):
        trace._selective_save_hidden_payloads = {
            **_hidden_payloads_by_label(trace),
            **(getattr(trace, "_selective_save_hidden_payloads", None) or {}),
        }
    apply_save_shaping(trace, plan, backend_name=backend_name, set_field=_set_op_field)
    if plan.layer_visualizers:
        _render_layer_visualizers(trace, dict(plan.layer_visualizers))
    _refresh_saved_activation_summary(trace)
    return trace


def warn_zero_match_save_predicate(backend_name: str, *, stacklevel: int = 3) -> None:
    """Disclose a ``save=`` predicate that matched ZERO operations.

    A predicate matching no sites used to complete silently -- a typo'd label
    or function name produced a COMPLETE trace with no saved activations and
    no diagnostic. The trace stays usable (structure and metadata survive),
    so this warns rather than refusing; TF's intervene-side reachability
    check remains fail-closed separately. Shared by the post-finalization
    resolver (jax / paddle / tinygrad / mlx) and TF's per-op retention seams
    so no backend can drop out of the disclosure family unnoticed.

    Parameters
    ----------
    backend_name
        Backend name used in the diagnostic.
    stacklevel
        Forwarded to :func:`warnings.warn`.

    Returns
    -------
    None
        Emits exactly one warning.
    """

    warnings.warn(
        f"{backend_name} trace(save=...) predicate matched zero operations; "
        "the returned trace retains no saved activation payloads. Check the "
        "selector against the captured op labels (trace.ops) -- a "
        "mistyped label or function name silently saves nothing.",
        stacklevel=stacklevel,
    )


def _hidden_payloads_by_label(trace: Any) -> dict[str, Any]:
    """Return runtime-only payloads keyed by raw and final op labels.

    Parameters
    ----------
    trace
        Trace whose current public payloads should remain replay-visible after filtering.

    Returns
    -------
    dict[str, Any]
        Payloads keyed by available op labels.
    """

    hidden_payloads: dict[str, Any] = {}
    for op in trace._layer_list:
        output = getattr(op, "out", None)
        if output is None:
            continue
        for label in (getattr(op, "_label_raw", None), getattr(op, "label", None)):
            if isinstance(label, str):
                hidden_payloads[label] = output
    return hidden_payloads


def pop_static_label_save_predicate(
    kwargs: dict[str, Any],
    *,
    backend_name: str,
) -> Callable[[Any], Any] | BaseSelector | None:
    """Return the backend-neutral ``save=`` predicate from backend kwargs.

    Parameters
    ----------
    kwargs
        Extra backend kwargs from the public trace API. The ``save`` key is removed when present.
    backend_name
        Backend name used in diagnostic errors.

    Returns
    -------
    Callable[[Any], Any] | BaseSelector | None
        ``save=`` predicate evaluated after finalization by
        :func:`apply_static_label_save_policy`, or ``None`` for save-all
        (omitted or ``"all"``).
    """

    save_value = kwargs.pop("save", None)
    if save_value is None or (isinstance(save_value, str) and save_value == "all"):
        return None
    _reject_non_static_save_predicate(save_value, backend_name=backend_name)
    return cast("Callable[[Any], Any] | BaseSelector", save_value)


def _drop_public_activation_payload(op: Any) -> None:
    """Remove one op's public saved activation payload without erasing metadata.

    Parameters
    ----------
    op
        Operation whose public payload should be hidden.

    Returns
    -------
    None
        Payload-bearing fields are nulled and ``has_saved_activation`` is set false.
    """

    _set_op_field(op, "out", None)
    _set_op_field(op, "out_ref", None)
    _set_op_field(op, "transformed_out", None)
    _set_op_field(op, "transformed_out_shape", None)
    _set_op_field(op, "transformed_out_dtype", None)
    _set_op_field(op, "transformed_activation_memory", None)
    _set_op_field(op, "has_saved_activation", False)


def _set_op_field(op: Any, field_name: str, value: Any) -> None:
    """Set an op field without marking trace state dirty when supported.

    Parameters
    ----------
    op
        Operation whose field should be updated.
    field_name
        Field name to set.
    value
        Replacement value.

    Returns
    -------
    None
        Field is updated on ``op``.
    """

    setter = getattr(op, "_internal_set", None)
    if callable(setter):
        setter(field_name, value)
        return
    setattr(op, field_name, value)


def _refresh_saved_activation_summary(trace: Any) -> None:
    """Recompute trace-level saved-activation counters.

    Parameters
    ----------
    trace
        Trace whose saved summary should match current op flags.

    Returns
    -------
    None
        Aggregate counters are overwritten in place.
    """

    saved_ops = [
        op
        for op in trace._layer_list
        if getattr(op, "has_saved_activation", False) and not record_member(op, "is_orphan", False)
    ]
    trace._num_saved_ops = len(saved_ops)
    # Read the ONE byte model -- physically retained
    # alias-aware bytes -- never a sum of raw activation_memory fields. The
    # explorer memory truth (reduce-only captures count the retained
    # TRANSFORMED bytes) is served by this same aggregate.
    from .._save_budget import retained_activation_bytes

    trace._cost_saved_activation_memory = Bytes(retained_activation_bytes(saved_ops))
    refresh_saved_module_call_count(trace, {op.label for op in saved_ops})
    refresh_pinned_autograd_memory(trace, saved_ops)
    trace._layers_saved = trace._num_saved_ops == len(trace._layer_list)


def _reject_non_static_save_predicate(
    predicate: Any,
    *,
    backend_name: str,
) -> None:
    """Reject ``save=`` predicates a finished capture cannot evaluate.

    Parameters
    ----------
    predicate
        Candidate public save predicate.
    backend_name
        Backend name used in diagnostic errors.

    Returns
    -------
    None
        Returns for plain callables and selectors of :data:`POST_SAVE_SELECTOR_KINDS`.
    """

    if not isinstance(predicate, BaseSelector):
        if callable(predicate):
            return
        raise BackendUnsupportedError(
            f"{backend_name} backend trace(save=...) takes a selector, a predicate "
            f"callable, 'all' or None; received {type(predicate).__name__}."
        )
    unsupported = first_selector_kind_outside(predicate, allowed=POST_SAVE_SELECTOR_KINDS)
    if unsupported is None:
        return
    raise BackendUnsupportedError(
        f"{backend_name} backend cannot evaluate trace(save=...) selector kind "
        f"{unsupported!r} on a finished capture. Supported: static-label selectors, "
        "value-dependent predicates (callables, tl.where), tl.preceded_by and "
        "candidate & tl.followed_by. Use the PyTorch backend for this selector."
    )
