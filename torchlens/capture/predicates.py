"""Predicate evaluation helpers for capture-time fastlog projections."""

from __future__ import annotations

import time
import warnings
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import torch

from ..fastlog.exceptions import PredicateError
from ..fastlog.types import CaptureSpec, ModuleStackFrame, RecordContext
from ..intervention.predicates import as_intervention_decision
from ..intervention.selectors import BaseSelector
from ..intervention.types import InterventionDecision
from ..ir.predicate import RetroactiveCaptureDecision
from ..ir.selector_eval import (
    contains_followed_by,
    module_union_addresses,
    module_union_matches,
    selector_contains_kind,
    split_followed_by_conjunction,
)

if TYPE_CHECKING:
    from ..fastlog.options import RecordingOptions


def _coerce_default_capture_spec(default: bool | CaptureSpec) -> CaptureSpec:
    """Normalize a default capture value to a CaptureSpec."""

    if isinstance(default, CaptureSpec):
        return default
    if default is True:
        return CaptureSpec(save_out=True, save_metadata=True)
    if default is False:
        return CaptureSpec(save_out=False, save_metadata=False)
    raise PredicateError(
        "default capture decision must be bool or CaptureSpec. "
        "Remedy: pass True, False, or a CaptureSpec as the default_op/default_module value.",
        code="predicate_default_invalid",
    )


def _normalize_capture_decision(
    result: bool | CaptureSpec | RetroactiveCaptureDecision | None,
    ctx: RecordContext,
    default: bool | CaptureSpec,
) -> CaptureSpec | RetroactiveCaptureDecision:
    """Normalize one predicate return value to a CaptureSpec.

    Parameters
    ----------
    result:
        Predicate return value.
    ctx:
        Event context supplied to the predicate.
    default:
        Slot default used when ``result`` is None.

    Returns
    -------
    CaptureSpec
        Normalized capture policy.

    Raises
    ------
    PredicateError
        If the predicate returned a value outside the supported contract.
    """

    default_spec = _coerce_default_capture_spec(default)
    if result is True:
        return CaptureSpec(
            save_out=True,
            save_metadata=True,
            keep_grad=default_spec.keep_grad,
            device=default_spec.device,
            dtype=default_spec.dtype,
            save_mode=default_spec.save_mode,
        )
    if result is False:
        return CaptureSpec(save_out=False, save_metadata=False)
    if result is None:
        return default_spec
    if isinstance(result, (CaptureSpec, RetroactiveCaptureDecision)):
        return result
    raise PredicateError(
        "predicate must return bool, CaptureSpec, RetroactiveCaptureDecision, or None. "
        "Remedy: return one of those values from the save predicate.",
        ctx=ctx,
        result=result,
        code="predicate_return_invalid",
    )


def _evaluate_keep_op(
    ctx: RecordContext,
    options: RecordingOptions,
) -> CaptureSpec | RetroactiveCaptureDecision:
    """Evaluate the operation/source predicate slot for one event."""

    result: bool | CaptureSpec | RetroactiveCaptureDecision | None
    module_union = _plain_module_union(options.keep_op)
    if options.keep_op is None:
        result = None
    elif module_union is not None:
        # A pure ``tl.module`` union has no followed_by term and no label-universe
        # kind, so the branch below would evaluate it once with no alias retry.
        result = module_union_matches(ctx, module_union)
    else:
        uses_supported_followed_by = _is_supported_followed_by_predicate(options.keep_op)
        result = _evaluate_retroactive_followed_by(ctx, options)
        if result is None:
            result = False if uses_supported_followed_by else options.keep_op(ctx)
        if (
            result is False
            and ctx.kind == "op"
            and ctx.layer_type is not None
            and ctx.type_index is not None
            and not uses_supported_followed_by
            and _keep_op_needs_alias_retry(options.keep_op)
        ):
            alias_ctx = replace(ctx, label=f"{ctx.layer_type}_{ctx.type_index}")
            result = options.keep_op(alias_ctx)
            if result is not False:
                ctx = alias_ctx
    return _normalize_capture_decision(result, ctx, options.default_op)


#: The last save selector seen and its plain-module address set (``None`` when it
#: is not a pure ``tl.module`` union). One slot, weakly held: a capture evaluates
#: one save selector for every event, and a finished capture's selector is never
#: kept alive by this cache.
_module_union_cache: list[Any] = [None, None]


def _plain_module_union(predicate: object | None) -> frozenset[str] | None:
    """Return the address set of a save selector built only from ``tl.module`` and ``|``.

    Parameters
    ----------
    predicate
        Configured keep-op predicate.

    Returns
    -------
    frozenset[str] | None
        The module addresses (optionally pass-qualified) when ``predicate`` is a
        pure ``tl.module`` union, else ``None``. Selectors are immutable, so the
        answer is computed once per selector object.
    """

    if not isinstance(predicate, BaseSelector):
        return None
    cached_ref, cached_addresses = _module_union_cache
    if cached_ref is not None and cached_ref() is predicate:
        return cast("frozenset[str] | None", cached_addresses)
    try:
        predicate_ref = weakref.ref(predicate)
    except TypeError:
        # An unreferenceable selector subclass cannot be cached; it keeps the walker.
        return None
    found = module_union_addresses(predicate)
    addresses = None if found is None else frozenset(found)
    _module_union_cache[:] = [predicate_ref, addresses]
    return addresses


def _save_selector_matches(selector: BaseSelector, ctx: RecordContext) -> bool:
    """Return whether a capture-time save selector matches one record context.

    Parameters
    ----------
    selector
        Save selector.
    ctx
        Capture-time record context.

    Returns
    -------
    bool
        ``selector(ctx)``, answered by one set lookup for a pure ``tl.module`` union.
    """

    module_union = _plain_module_union(selector)
    if module_union is not None:
        return module_union_matches(ctx, module_union)
    return bool(selector(ctx))


#: Capture-time selector kinds whose short/friendly ``{layer_type}_{type_index}``
#: label is only visible through the :func:`_evaluate_keep_op` alias retry. ``label``,
#: ``contains``, and ``regex`` all resolve through the capture label universe in
#: ``ir.selector_eval`` (which on the base context exposes only the raw label such as
#: ``"conv2d_2_4_raw"``); ``predicate`` trees read ``ctx.label`` directly.
_ALIAS_RETRY_SELECTOR_KINDS: tuple[str, ...] = ("predicate", "label", "contains", "regex")


def _keep_op_needs_alias_retry(predicate: object | None) -> bool:
    """Return whether a keep-op predicate still needs the alias compatibility retry.

    Parameters
    ----------
    predicate
        Configured keep-op predicate.

    Returns
    -------
    bool
        ``True`` when the predicate may still rely on the second evaluation with
        ``ctx.label`` rewritten to the short/friendly ``"{layer_type}_{type_index}"``
        label (e.g. ``"conv2d_2"``).

    Notes
    -----
    The base capture-time ``RecordContext`` only carries the RAW label (such as
    ``"conv2d_2_4_raw"``); the short/friendly label is synthesized ONLY by the alias
    retry in :func:`_evaluate_keep_op`. Every selector that resolves through the
    capture label universe (``label``, ``contains``, ``regex``) can therefore target a
    short label that is invisible on the first evaluation, so those kinds need the
    retry too -- not just bare-callable predicate trees whose inner callable observes
    ``ctx.label`` directly. Structured selectors that match non-label fields
    (``func``, ``module``, ``in_module``, ``output``, ...) already see everything they
    need on the base context and are intentionally excluded. The retry fires only after
    a first-call miss, so widening the set is purely additive: it can add a match for a
    short-label target, never remove an existing match.
    """

    if not isinstance(predicate, BaseSelector):
        return True
    return any(selector_contains_kind(predicate, kind) for kind in _ALIAS_RETRY_SELECTOR_KINDS)


def _evaluate_intervene_op(
    ctx: RecordContext,
    options: RecordingOptions,
) -> InterventionDecision | None:
    """Evaluate the active operation intervention predicate slot.

    Parameters
    ----------
    ctx:
        Operation context for the current candidate output.
    options:
        Unified predicate runtime options.

    Returns
    -------
    InterventionDecision | None
        Normalized current-op intervention decision, if any.
    """

    if options.intervene is None:
        return None
    result = options.intervene(ctx)
    try:
        return as_intervention_decision(result)
    except TypeError as exc:
        raise PredicateError(
            "intervene predicate must return InterventionDecision, HelperSpec, callable, "
            "or None. Remedy: return one of those values from the intervene predicate.",
            ctx=ctx,
            result=result,
            code="predicate_return_invalid",
        ) from exc


def _evaluate_halt(
    ctx: RecordContext,
    options: RecordingOptions,
    frontier_output: Any | None = None,
) -> None:
    """Evaluate the halt predicate slot and raise when it matches.

    Parameters
    ----------
    ctx:
        Event context for the current source, operation, or module boundary.
    options:
        Unified predicate runtime options.
    frontier_output:
        Live tensor or output structure to use as the partial-trace frontier if
        this halt predicate matches.

    Raises
    ------
    HaltSignal
        If ``options.halt`` returns ``True`` for ``ctx``.
    PredicateError
        If ``options.halt`` returns a non-bool value.
    """

    from .stop import StopDirective

    StopDirective(halt_options=options).evaluate_halt(ctx, frontier_output=frontier_output)


def _is_halt_only_capture(options: RecordingOptions) -> bool:
    """Return whether capture can evaluate only the halt predicate per event.

    The fast path is deliberately narrow: no save predicate, no default
    retention, no intervention, no gradient capture, and no armed echo
    narrator (narration needs the full per-event context build and the
    committed-event ordering, snoop D1). That preserves the save-then-halt
    ordering for every configuration that can retain payloads or metadata.
    """

    return (
        options.halt is not None
        and options.keep_op is None
        and options.default_op is False
        and options.default_module is False
        and options.intervene is None
        and options.save_grads in (None, False)
        and options.default_grad is False
        and getattr(options, "echo", None) is None
    )


def _evaluate_retroactive_followed_by(
    ctx: RecordContext,
    options: RecordingOptions,
) -> RetroactiveCaptureDecision | None:
    """Evaluate supported ``candidate & followed_by(successor)`` predicate sugar."""

    split = split_followed_by_conjunction(options.keep_op)
    if split is None:
        return None
    followed_selector, candidate_selector = split
    inner = followed_selector.inner
    if not callable(inner) or not bool(inner(ctx)):
        return None
    target_labels = _matching_recent_parent_labels(ctx, candidate_selector)
    if not target_labels:
        return None
    return RetroactiveCaptureDecision(
        target_raw_labels=target_labels,
        spec=CaptureSpec(save_out=True, save_metadata=True),
    )


def _is_supported_followed_by_predicate(predicate: Any) -> bool:
    """Return whether ``predicate`` is the supported retroactive selector shape.

    Parameters
    ----------
    predicate
        Candidate save predicate.

    Returns
    -------
    bool
        ``True`` for ``candidate & tl.followed_by(successor)``.
    """

    selector = getattr(predicate, "selector", None)
    if selector is not None:
        return _is_supported_followed_by_predicate(selector)
    return split_followed_by_conjunction(predicate) is not None


def validate_followed_by_capability(
    predicate: Any,
    *,
    api_name: str,
    supports_retroactive: bool,
) -> None:
    """Raise a typed error when ``followed_by`` cannot run on this surface.

    Parameters
    ----------
    predicate
        Public save predicate to inspect.
    api_name
        User-facing API name for the error message.
    supports_retroactive
        Whether the capture surface can replace prior candidate events.

    Returns
    -------
    None
        Raises only for unsupported ``followed_by`` usage.
    """

    if not contains_followed_by(predicate, unwrap=True):
        return
    if not _is_supported_followed_by_predicate(predicate):
        raise PredicateError(
            "tl.followed_by(...) only supports candidate & tl.followed_by(successor); "
            f"{api_name} received an unsupported followed_by predicate shape. "
            "Remedy: compose the predicate as candidate & tl.followed_by(successor).",
            code="followed_by_unsupported",
        )
    if not supports_retroactive:
        raise PredicateError(
            f"{api_name} does not support tl.followed_by(...) retroactive capture; "
            "use trace(save=...) with lookback and lookback_payload_policy instead. "
            "Remedy: use trace(save=...) with lookback= and lookback_payload_policy=.",
            code="followed_by_unsupported",
        )


def _matching_recent_parent_labels(
    ctx: RecordContext,
    candidate_selector: BaseSelector,
) -> tuple[str, ...]:
    """Return parent labels in the lookback window matching a candidate selector."""

    parent_labels = tuple(ctx.parent_labels_raw or ctx.parent_labels)
    if not parent_labels:
        return ()
    recent_by_label = {
        recent.raw_label or recent.label: recent
        for recent in ctx.recent_ops
        if recent.raw_label is not None or recent.label
    }
    matches: list[str] = []
    for parent_label in parent_labels:
        recent = recent_by_label.get(parent_label)
        if recent is None:
            warnings.warn(
                f"followed_by parent {parent_label!r} is outside the lookback window; "
                "increase lookback to make this dependency queryable.",
                RuntimeWarning,
                stacklevel=3,
            )
            continue
        if candidate_selector(recent):
            matches.append(parent_label)
    return tuple(matches)


def _module_capture_spec(options: RecordingOptions) -> CaptureSpec:
    """Return the capture policy for one module boundary event.

    Module events have no predicate slot (predicate-gated module-event
    selection was removed); ``default_module`` is the whole policy.
    """

    return _coerce_default_capture_spec(options.default_module)


def build_op_record_context(
    *,
    kind: str,
    label: str,
    raw_label: str,
    raw_index: int,
    layer_type: str,
    type_index: int,
    func_name: str | None,
    parent_labels: Sequence[str],
    tensor: torch.Tensor,
    output_index: int | None,
    is_bottom_level_func: bool | None,
    module_stack: Sequence[ModuleStackFrame | Mapping[str, Any]],
    history: Sequence[RecordContext],
    op_counts: Mapping[str, int],
    pass_index: int,
    event_index: int,
    step_index: int | None,
    capture_start_time: float,
    include_source_events: bool,
    sample_id: str | int | None,
    address: str | None = None,
    module_type: str | None = None,
    module_pass_index: int | None = None,
    is_transform: bool = False,
    transform_kind: str | None = None,
) -> RecordContext:
    """Build the unified operation ``RecordContext`` used by all capture paths.

    Parameters
    ----------
    kind:
        Event kind, usually ``"op"``.
    label, raw_label:
        Public-in-flight and raw labels for the operation.
    raw_index:
        Global raw operation index.
    layer_type, type_index:
        Normalized TorchLens operation type and per-type counter.
    func_name:
        Original function name, when known.
    parent_labels:
        Raw parent labels visible at forward time.
    tensor:
        Output tensor being considered.
    output_index:
        Index within a multi-output operation, when applicable.
    is_bottom_level_func:
        Whether the decorated call is a bottom-level function.
    module_stack:
        Active module-stack frames.
    history:
        Bounded recent ``RecordContext`` window.
    op_counts:
        Per-operation-type counts visible to predicate code.
    pass_index:
        Forward pass index for the active capture session.
    event_index:
        Chronological event index.
    step_index:
        Operation step index.
    capture_start_time:
        Wall-clock start time for the current capture.
    include_source_events:
        Whether source events should appear in ``recent_ops``.
    sample_id:
        Optional sample id for batched predicate runs.
    address, module_type, module_pass_index:
        Nearest module context fields, when known.

    Returns
    -------
    RecordContext
        Frozen predicate context.
    """

    from .projections import _build_record_context

    return _build_record_context(
        kind=kind,
        op_log_or_op_data={
            "label": label,
            "raw_label": raw_label,
            "_label_raw": raw_label,
            "raw_index": raw_index,
            "type": layer_type,
            "type_index": type_index,
            "func_name": func_name,
            "parent_labels": tuple(parent_labels),
            "tensor": tensor,
            "output_index": output_index,
            "is_bottom_level_func": is_bottom_level_func,
            "address": address,
            "module_type": module_type,
            "module_pass_index": module_pass_index,
            "is_transform": is_transform,
            "transform_kind": transform_kind,
        },
        module_stack=module_stack,
        history=tuple(history),
        op_counts=op_counts,
        pass_index=pass_index,
        event_index=event_index,
        step_index=step_index,
        time_since_pass_start=time.time() - capture_start_time,
        include_source_events=include_source_events,
        sample_id=sample_id,
    )
