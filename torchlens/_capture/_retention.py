"""Selector classification for deferred capture retention.

:mod:`torchlens._capture.session` asks these questions of the ``save`` /
``save_grads`` selectors when it compiles a capture plan and resolves deferred
retention: which rolling tail windows a legacy selector declares, and whether a
gradient selector needs backward-fire context to evaluate.
"""

from __future__ import annotations

from typing import Any

#: Selector kinds decidable only with backward-fire context (pass number,
#: grad kind, grad_fn identity); site-lifecycle evaluation reads them as
#: False, so hook placement must not consult site resolution for them.
_BACKWARD_CONTEXT_SELECTOR_KINDS = frozenset(
    {"backward_pass", "grad_kind", "grad_fn", "grad_fn_handle", "grad_fn_label"}
)


def _selector_needs_backward_context(selector: Any) -> bool:
    """Return whether a selector tree contains a backward-context leaf.

    Parameters
    ----------
    selector
        Selector (possibly composite/negated) supplied as ``save_grads``.

    Returns
    -------
    bool
        Whether any leaf requires backward-fire context to evaluate.
    """

    stack: list[Any] = [selector]
    while stack:
        node = stack.pop()
        if getattr(node, "selector_kind", None) in _BACKWARD_CONTEXT_SELECTOR_KINDS:
            return True
        children = getattr(node, "selectors", None)
        if isinstance(children, (list, tuple)):
            stack.extend(children)
        wrapped = getattr(node, "selector", None)
        if wrapped is not None:
            stack.append(wrapped)
    return False


def _negative_selector_windows(selector: Any) -> tuple[int, ...]:
    """Return rolling-window bounds declared by negative integer selectors.

    Parameters
    ----------
    selector
        Legacy selector value or nested selector container.

    Returns
    -------
    tuple[int, ...]
        Positive tail-window sizes in encounter order.
    """

    if isinstance(selector, int) and selector < 0:
        return (-selector,)
    if isinstance(selector, (list, tuple, set, frozenset)):
        return tuple(window for item in selector for window in _negative_selector_windows(item))
    return ()
