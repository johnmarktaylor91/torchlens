"""Leaf plan nodes for the compiled selector evaluator.

:mod:`torchlens._ir.selector_eval` compiles a selector once per lane into a
plan; this module holds the per-kind node compilers and matchers that read a
subject through duck typing alone. A compiler maps ``(selector, lane)`` to a
matcher, closing over the payload and the lane-specific universe once, so
evaluation is a call with no branching on selector shape. Nodes that need the
package's own types (container paths, record contexts, backward ``GradFn``
pairing, episode sessions) stay beside :func:`evaluate`; this module and
:mod:`torchlens._ir._selector_universe` import nothing from the package, so
the selector split adds no module to the import cycle.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

from ._selector_universe import (
    _SUBSTRING_LABEL_ATTRS,
    _grad_fn_call_values,
    _grad_fn_type_matches,
    _string_labels,
    sanitize_transform_kind,
)

if TYPE_CHECKING:
    from ..intervention.selectors import BaseSelector

#: One compiled plan node: a callable closed over its selector's payload and
#: lane that answers ``matcher(subject)``. Composite nodes hold their children's
#: matchers; refusals are nodes that raise when evaluated, so a refusal behind a
#: short-circuiting sibling surfaces exactly where it did before compilation.
_Matcher = Callable[[Any], bool]

_Compiler = Callable[[BaseSelector, str], _Matcher]


def _compile_label(selector: BaseSelector, lane: str) -> _Matcher:
    """Compile an exact ``label`` literal.

    Parameters
    ----------
    selector:
        ``label`` node.
    lane:
        Lane; post hoc the universe also includes the site's ``lookup_keys``.

    Returns
    -------
    _Matcher
        Plan node.
    """

    target = str(selector.selector_value)
    matcher = partial(_match_label, target=target, lifecycle=lane, lookup_keys=lane == "site")
    return matcher


def _match_label(subject: Any, *, target: str, lifecycle: str, lookup_keys: bool) -> bool:
    """Return whether ``target`` is one of the subject's exact label spellings."""

    if target in _string_labels(subject, lifecycle):
        return True
    return lookup_keys and target in (getattr(subject, "lookup_keys", ()) or ())


def _compile_contains(selector: BaseSelector, lane: str) -> _Matcher:
    """Compile a case-insensitive ``contains`` substring search.

    Parameters
    ----------
    selector:
        ``contains`` node.
    lane:
        Lane; live module-boundary proxies never match, and a pass-qualified
        needle post hoc also searches the pass-qualified ``Op.label``.

    Returns
    -------
    _Matcher
        Plan node.
    """

    value = str(selector.selector_value)
    needle = value.lower()
    matcher = partial(
        _match_contains,
        needle=needle,
        lifecycle=lane,
        skip_live_boundary=lane == "live",
        pass_qualified=lane == "site" and ":" in needle,
    )
    return matcher


def _match_contains(
    subject: Any, *, needle: str, lifecycle: str, skip_live_boundary: bool, pass_qualified: bool
) -> bool:
    """Return whether ``needle`` occurs in one of the subject's search labels."""

    if skip_live_boundary and bool(getattr(subject, "_tl_module_boundary", False)):
        return False
    labels = _string_labels(subject, lifecycle, _SUBSTRING_LABEL_ATTRS)
    if pass_qualified:
        # Bare final labels never contain ':', so a pass-qualified needle (the
        # remedy the multipass_bare_label refusal teaches) can only address the
        # pass-qualified op spelling; without it, string addressing of one pass
        # of a multi-pass layer matches 0 sites.
        pass_label = getattr(subject, "label", None)
        if isinstance(pass_label, str) and pass_label not in labels:
            labels = (*labels, pass_label)
    return any(needle in label.lower() for label in labels)


def _compile_regex(selector: BaseSelector, lane: str) -> _Matcher:
    """Compile a case-sensitive ``regex`` search over the search labels.

    Parameters
    ----------
    selector:
        ``regex`` node.
    lane:
        Lane.

    Returns
    -------
    _Matcher
        Plan node.
    """

    pattern = str(selector.selector_value)
    return partial(_match_regex, pattern=pattern, lifecycle=lane)


def _match_regex(subject: Any, *, pattern: str, lifecycle: str) -> bool:
    """Return whether ``pattern`` searches into one of the subject's search labels."""

    return any(
        re.search(pattern, label) is not None
        for label in _string_labels(subject, lifecycle, _SUBSTRING_LABEL_ATTRS)
    )


def _compile_func_transform(selector: BaseSelector, lane: str) -> _Matcher:
    """Compile ``func_transform``: a transform boundary, optionally of one kind.

    Parameters
    ----------
    selector:
        ``func_transform`` node.
    lane:
        Lane (unused: the rule is lane-independent).

    Returns
    -------
    _Matcher
        Plan node.
    """

    value = selector.selector_value
    return partial(
        _match_func_transform, wanted=None if value is None else sanitize_transform_kind(value)
    )


def _match_func_transform(subject: Any, *, wanted: str | None) -> bool:
    """Return whether the subject is a transform boundary of the wanted kind."""

    if not bool(getattr(subject, "is_transform", False)):
        return False
    if wanted is None:
        return True
    transform_kind = getattr(subject, "transform_kind", None)
    return transform_kind is not None and sanitize_transform_kind(transform_kind) == wanted


def _match_predicate(subject: Any, *, predicate: Callable[[Any], Any]) -> bool:
    """Return the predicate's truth on ``subject``."""

    return bool(predicate(subject))


def _compile_grad_kind_capture(selector: BaseSelector, lane: str) -> _Matcher:
    """Compile ``grad_kind`` against a capture context's ``grad_kind`` stamp.

    Parameters
    ----------
    selector:
        ``grad_kind`` node.
    lane:
        Lane (capture).

    Returns
    -------
    _Matcher
        Plan node.
    """

    return partial(_match_attr_equals, attr="grad_kind", value=selector.selector_value)


def _match_attr_equals(subject: Any, *, attr: str, value: Any) -> bool:
    """Return whether ``subject.<attr>`` (default ``None``) equals ``value``."""

    return getattr(subject, attr, None) == value


def _compile_backward_pass_capture(selector: BaseSelector, lane: str) -> _Matcher:
    """Compile ``backward_pass`` against a capture context's pass index.

    Parameters
    ----------
    selector:
        ``backward_pass`` node.
    lane:
        Lane (capture).

    Returns
    -------
    _Matcher
        Plan node.
    """

    return partial(_match_backward_pass_capture, value=selector.selector_value)


def _match_backward_pass_capture(subject: Any, *, value: Any) -> bool:
    """Return whether the context's backward (or plain) pass index equals ``value``."""

    ctx_pass = getattr(subject, "backward_pass_index", None)
    if ctx_pass is None:
        ctx_pass = getattr(subject, "pass_index", None)
    return ctx_pass == value


def _compile_never(selector: BaseSelector, lane: str) -> _Matcher:
    """Compile a kind that silently never matches on this lane.

    Parameters
    ----------
    selector:
        Selector node (unused).
    lane:
        Lane (unused).

    Returns
    -------
    _Matcher
        :func:`_match_never`.
    """

    return _match_never


def _match_never(subject: Any) -> bool:
    """Never match."""

    return False


def _preceded_by_matches(ctx: Any, *, inner: Any) -> bool:
    """Return whether a retained lookback predecessor matches ``inner``.

    Parameters
    ----------
    ctx:
        Capture-time context with ``recent_ops`` populated.
    inner:
        The lookback selector's inner predicate (selector or callable).

    Returns
    -------
    bool
        Whether a retained (parent) predecessor matches the inner predicate.
    """

    parent_labels = set(getattr(ctx, "parent_labels_raw", ()) or getattr(ctx, "parent_labels", ()))
    recent_ops = tuple(getattr(ctx, "recent_ops", ()))
    if parent_labels:
        return any(
            (recent.raw_label or recent.label) in parent_labels and bool(inner(recent))
            for recent in recent_ops
        )
    return any(bool(inner(recent)) for recent in recent_ops)


def _compile_grad_fn_without_op(selector: BaseSelector, lane: str) -> _Matcher:
    """Compile ``intervening`` / ``without_op``: GradFns with no paired op.

    Parameters
    ----------
    selector:
        Selector node (unused).
    lane:
        Lane (unused).

    Returns
    -------
    _Matcher
        :func:`_match_grad_fn_without_op`.
    """

    return _match_grad_fn_without_op


def _match_grad_fn_without_op(site: Any) -> bool:
    """Return whether the GradFn has no paired forward op."""

    return not site.has_op


def _compile_grad_fn_label(selector: BaseSelector, lane: str) -> _Matcher:
    """Compile ``grad_fn_label``: the exact backward label.

    Parameters
    ----------
    selector:
        ``grad_fn_label`` node.
    lane:
        Lane (unused).

    Returns
    -------
    _Matcher
        Plan node.
    """

    return partial(_match_attr_equals, attr="label", value=str(selector.selector_value))


def _compile_grad_fn_payload(selector: BaseSelector, lane: str) -> _Matcher:
    """Compile ``grad_fn``: type, label pattern and custom-ness constraints.

    Parameters
    ----------
    selector:
        ``grad_fn`` node; a non-mapping payload constrains nothing.
    lane:
        Lane (unused).

    Returns
    -------
    _Matcher
        Plan node.
    """

    value = selector.selector_value
    payload = value if isinstance(value, dict) else {}
    return partial(
        _match_grad_fn_payload,
        grad_fn_type=payload.get("type"),
        label_pattern=payload.get("grad_fn_label_pattern"),
        is_custom=payload.get("is_custom"),
    )


def _match_grad_fn_payload(
    site: Any, *, grad_fn_type: Any, label_pattern: Any, is_custom: Any
) -> bool:
    """Return whether the GradFn satisfies every present ``grad_fn`` constraint."""

    if grad_fn_type is not None and not _grad_fn_type_matches(site, str(grad_fn_type)):
        return False
    if label_pattern is not None and str(label_pattern) not in site.label:
        return False
    return not (is_custom is not None and bool(site.is_custom) is not bool(is_custom))


def _compile_grad_fn_grad_kind(selector: BaseSelector, lane: str) -> _Matcher:
    """Compile ``grad_kind`` on a GradFn: any call carrying that gradient field.

    Parameters
    ----------
    selector:
        ``grad_kind`` node.
    lane:
        Lane (unused).

    Returns
    -------
    _Matcher
        Plan node.
    """

    grad_kind = str(selector.selector_value)
    field_name = "grad_inputs" if grad_kind == "grad_input" else "grad_outputs"
    return partial(_match_grad_fn_call_field, field_name=field_name)


def _match_grad_fn_call_field(site: Any, *, field_name: str) -> bool:
    """Return whether any backward call of the GradFn carries ``field_name``."""

    return any(getattr(call, field_name, None) is not None for call in _grad_fn_call_values(site))


def _compile_grad_fn_backward_pass(selector: BaseSelector, lane: str) -> _Matcher:
    """Compile ``backward_pass`` on a GradFn: any call at that pass index.

    Parameters
    ----------
    selector:
        ``backward_pass`` node.
    lane:
        Lane (unused).

    Returns
    -------
    _Matcher
        Plan node.
    """

    return partial(_match_grad_fn_backward_pass, pass_index=int(selector.selector_value))


def _match_grad_fn_backward_pass(site: Any, *, pass_index: int) -> bool:
    """Return whether any backward call of the GradFn ran at ``pass_index``."""

    return any(
        getattr(call, "backward_pass_index", None) == pass_index
        for call in _grad_fn_call_values(site)
    )
