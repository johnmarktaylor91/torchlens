"""Edit conditions that read predicate contexts, matched on a tinygrad capture.

A ``tl.when`` condition that is a plain callable or holds a ``tl.where`` leaf
reads the op's ``RecordContext``. On PyTorch's live capture that context
carries the op's labels, function, module, parents, shape and dtype and the
recent-event window; its value fields ``is_scalar_bool`` and ``bool_value``
stay unset during the forward. All of those facts are fixed by the forward's
structure, which an edit cannot change (edits keep shape and dtype), so the
context of each op is the same in the planning capture and in the edited
capture. The edited capture's planning run therefore matches the condition on
the rebuilt contexts of its finished ops, the same contexts ``trace(save=...)``
predicates see on a finished tinygrad capture, and the edited run edits the
calls that returned the matched ops. Nothing is realized mid-forward: the
edited capture still realizes once.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..._postprocess._post_predicate import (
    _DEFAULT_HISTORY,
    _STATIC_SELECTOR_KINDS,
    _matches,
    _raw_label,
    _rebuild_events,
)

__tl_layer__ = "L2"


def condition_matched_labels(
    trace: Any, condition: Any, build_context: Callable[..., Any]
) -> frozenset[str]:
    """Raw labels of the ops whose predicate context satisfies ``condition``.

    Parameters
    ----------
    trace
        A finished tinygrad capture.
    condition
        The ``tl.when`` condition: a selector tree (static leaves are
        evaluated on the op, the rest on its context) or a plain callable
        (called with the context).
    build_context
        Torch's ``RecordContext`` builder.

    Returns
    -------
    frozenset[str]
        Raw labels of the matching ops. Inputs, buffers and outputs are not
        candidates: torch evaluates an edit condition only at the ops the
        forward runs.
    """

    is_selector = getattr(condition, "selector_kind", None) is not None
    return frozenset(
        _raw_label(event.op)
        for event in _rebuild_events(trace, _DEFAULT_HISTORY, build_context)
        if event.kind == "op"
        and (
            _matches(condition, event, _STATIC_SELECTOR_KINDS)
            if is_selector
            else bool(condition(event.ctx))
        )
    )
