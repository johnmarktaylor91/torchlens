"""Post-finalization selective-save helpers for non-torch backends."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .registry import BackendUnsupportedError
from ..intervention.selectors import (
    BaseSelector,
    CompositeSelector,
    NotSelector,
    SelectorLike,
)
from ..quantities import Bytes

_STATIC_SELECTOR_KINDS = frozenset(
    {
        "label",
        "func",
        "module",
        "output",
        "contains",
        "in_module",
        "and",
        "or",
        "not",
    }
)


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
            "tl.func, tl.label, tl.contains and boolean composites (&, |, ~) of those. "
            "Value-dependent predicates need concrete activation values at predicate time. "
            "MLX lazy evaluation defers RecordContext.tensor_requires_grad, is_scalar_bool, "
            "and bool_value without per-op evaluation; use the PyTorch backend for "
            "value-dependent predicates."
        )
    unsupported = _first_selector_outside_kinds(predicate, allowed=allowed)
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
            "tl.func, tl.label, tl.contains and boolean composites (&, |, ~) of those."
        )
    raise BackendUnsupportedError(
        f"{backend_name} backend supports trace(save=...) only for static-label selectors "
        "tl.func, tl.label, tl.contains and boolean composites (&, |, ~) of those; "
        f"unsupported selector kind {unsupported!r}."
    )


def apply_static_label_save_policy(
    trace: Any,
    predicate: Callable[[Any], Any] | BaseSelector | None,
    *,
    backend_name: str,
) -> None:
    """Filter public saved activations with a static-label predicate.

    Parameters
    ----------
    trace
        Finalized trace whose public saved activation surface should be filtered.
    predicate
        Static selector accepted by ``trace(save=...)``. ``None`` preserves full-save capture.
    backend_name
        Backend name used in diagnostic errors.

    Returns
    -------
    None
        Public payload fields and saved-summary counters are updated in place.
    """

    if predicate is None:
        return
    _reject_non_static_save_predicate(predicate, backend_name=backend_name)
    hidden_payloads = _hidden_payloads_by_label(trace)
    for op in getattr(trace, "layer_list", ()):
        if not bool(predicate(op)):
            _drop_public_activation_payload(op)
    trace._selective_save_hidden_payloads = hidden_payloads
    _refresh_saved_activation_summary(trace)


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
    for op in getattr(trace, "layer_list", ()):
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
        Static-label save predicate, or ``None`` when no predicate was supplied.
    """

    save_value = kwargs.pop("save", None)
    if save_value is None:
        return None
    if not isinstance(save_value, BaseSelector):
        raise BackendUnsupportedError(
            f"{backend_name} backend supports trace(save=...) only for static-label selectors "
            "(for example tl.func, tl.label, tl.module, tl.in_module, or tl.contains). "
            "Value-dependent predicates need concrete activation values at predicate time, "
            "which traced/lazy non-torch captures do not expose without changing execution. "
            "Use static-label save= selectors or the PyTorch backend for value-dependent "
            "predicates, intervene=, and halt=."
        )
    _reject_non_static_save_predicate(save_value, backend_name=backend_name)
    return save_value


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
        for op in getattr(trace, "layer_list", ())
        if getattr(op, "has_saved_activation", False) and not getattr(op, "is_orphan", False)
    ]
    trace.num_saved_ops = len(saved_ops)
    trace.saved_activation_memory = Bytes(
        sum(int(getattr(op, "activation_memory", 0) or 0) for op in saved_ops)
    )
    trace.num_saved_layers = len({op.layer_label for op in saved_ops})
    saved_layer_labels = {op.layer_label for op in saved_ops}
    trace.num_saved_module_calls = sum(
        1
        for module_call in getattr(trace, "module_calls", ())
        if any(label in saved_layer_labels for label in getattr(module_call, "layers", ()))
    )
    trace._layers_saved = trace.num_saved_ops == len(getattr(trace, "layer_list", ()))


def _reject_non_static_save_predicate(
    predicate: Callable[[Any], Any] | BaseSelector,
    *,
    backend_name: str,
) -> None:
    """Reject non-static save predicates for non-torch post-filtering.

    Parameters
    ----------
    predicate
        Candidate public save predicate.
    backend_name
        Backend name used in diagnostic errors.

    Returns
    -------
    None
        Returns when the predicate can be evaluated from finalized graph labels.
    """

    if not isinstance(predicate, BaseSelector):
        raise BackendUnsupportedError(
            f"{backend_name} backend supports trace(save=...) only for static-label selectors. "
            "Value-dependent predicates need concrete activation values at predicate time, "
            "which traced/lazy non-torch captures do not expose without changing execution. "
            "Use static-label save= selectors or the PyTorch backend for value-dependent "
            "predicates, intervene=, and halt=."
        )
    unsupported = _first_non_static_selector(predicate)
    if unsupported is None:
        return
    raise BackendUnsupportedError(
        f"{backend_name} backend does not support value-dependent or runtime-mutation "
        f"trace(save=...) predicates ({unsupported!r}). Static-label save= selectors are "
        "supported, but value-dependent predicates need concrete activation values at "
        "predicate time, which traced/lazy non-torch captures do not expose without changing "
        "execution. Use the PyTorch backend for value-dependent predicates, intervene=, and "
        "halt=."
    )


def _first_non_static_selector(selector: SelectorLike) -> str | None:
    """Return the first non-static selector kind in a selector tree.

    Parameters
    ----------
    selector
        Selector or target spec to classify.

    Returns
    -------
    str | None
        Unsupported selector kind, or ``None`` if the whole tree is static-label only.
    """

    if not isinstance(selector, BaseSelector):
        return "target_spec"
    kind = selector.selector_kind
    if kind not in _STATIC_SELECTOR_KINDS:
        return kind
    if isinstance(selector, CompositeSelector):
        left, right = selector.selectors
        return _first_non_static_selector(left) or _first_non_static_selector(right)
    if isinstance(selector, NotSelector):
        return _first_non_static_selector(selector.selector)
    return None


def _first_selector_outside_kinds(
    selector: SelectorLike,
    *,
    allowed: frozenset[str],
) -> str | None:
    """Return the first selector kind outside ``allowed`` in a selector tree.

    Parameters
    ----------
    selector
        Selector or target spec to classify.
    allowed
        Selector kinds accepted by the caller.

    Returns
    -------
    str | None
        Unsupported selector kind, or ``None`` if the whole tree is allowed.
    """

    if not isinstance(selector, BaseSelector):
        return "target_spec"
    kind = selector.selector_kind
    if kind not in allowed:
        return kind
    if isinstance(selector, CompositeSelector):
        left, right = selector.selectors
        return _first_selector_outside_kinds(
            left,
            allowed=allowed,
        ) or _first_selector_outside_kinds(right, allowed=allowed)
    if isinstance(selector, NotSelector):
        return _first_selector_outside_kinds(selector.selector, allowed=allowed)
    return None
