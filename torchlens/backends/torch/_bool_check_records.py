"""One capture record per conditional test of a reused bool tensor.

``Op.terminal_conditional_id`` names the ONE conditional a terminal bool decides.
When user code tests the same labelled bool tensor in two different conditional
tests (``if flag: ... if flag: ...``, or a ternary and a later ``if`` on one
predicate), one op cannot carry both ids. Capture therefore gives every such
later test its own record: a ``clone`` of the bool, logged at the test, which
becomes that test's terminal bool. The first conditional test keeps the
original op, so a model that tests each bool once records exactly what it did
before.

The rule reads only the consumer location recorded for the ``__bool__`` call:

* the bool's first location, or a location already recorded for it (a loop
  re-running the same test), stays on the original op;
* a new location stays on the original op unless it is a conditional test AND
  the original already decides a conditional test at another location;
* otherwise a clone is logged and the location is recorded under its label.

If the clone cannot be logged with a label, the location falls back to the
original op (the old shape, which validation's invariant 8 still rejects): the
rule never hides a conditional it could not record.

It imports only two leaves (``_consumer_kind`` and ``_utils._tensor_label``), both
already imported by its one caller, ``_completeness_storage``.
"""

from __future__ import annotations

import weakref
from typing import Any

import torch

from ..._utils._tensor_label import get_tensor_label
from ._consumer_kind import _classify_consumer_kind

__all__ = ("bool_check_record_label",)

#: Kinds a step-5 conditional records (``_postprocess.control_flow._BRANCH_CONTEXT_KINDS``).
_CONDITIONAL_TEST_KINDS = frozenset({"if_test", "elif_test", "ifexp"})

_LAST_CLONED_CHECK: weakref.WeakKeyDictionary[Any, tuple[str, tuple[str, int], str, int]] = (
    weakref.WeakKeyDictionary()
)
"""Per-trace ``(source label, location, clone label, layer counter)`` of the newest clone.

Two scalar-escape belts can wrap ``Tensor.__bool__`` at once, so one ``__bool__``
call can reach the recorder twice. The second call reuses the clone only when no
op has been logged since it (the layer counter is unchanged), so a later test at
the same location (a later loop iteration, or the second pass of a two-pass
capture) gets its own clone.
"""


def _is_conditional_test(location: tuple[str, int]) -> bool:
    """Return whether a consumer location is a test step 5 records as a conditional.

    Parameters
    ----------
    location:
        ``(filename, line)`` of the ``__bool__`` call.

    Returns
    -------
    bool
        True for an ``if`` / ``elif`` test or a conditional expression.
    """

    return _classify_consumer_kind(location[0], location[1]) in _CONDITIONAL_TEST_KINDS


def _layer_counter(trace: Any) -> int | None:
    """Return the trace's raw layer counter, or None when it is not readable."""

    try:
        counter = trace._raw_graph_ws.layer_counter
    except AttributeError:
        return None
    return counter if isinstance(counter, int) else None


def _log_clone_label(trace: Any, source: torch.Tensor) -> str | None:
    """Log one ``clone`` of ``source`` as a captured op and return its label.

    Parameters
    ----------
    trace:
        Active capture trace.
    source:
        Labelled bool tensor being tested.

    Returns
    -------
    str | None
        The clone's raw label, or None when capture did not label it.
    """

    before = _layer_counter(trace)
    record = source.clone()
    label = get_tensor_label(record)
    after = _layer_counter(trace)
    if not isinstance(label, str) or label == get_tensor_label(source):
        return None
    if before is None or after is None or after <= before:
        return None
    return label


def bool_check_record_label(
    trace: Any,
    source: torch.Tensor,
    label: str,
    location: tuple[str, int],
    locations: dict[str, list[tuple[str, int]]],
) -> str:
    """Return the label of the record that decides this bool check.

    Parameters
    ----------
    trace:
        Active capture trace.
    source:
        Labelled 0-dim-or-larger bool tensor reaching ``__bool__``.
    label:
        ``source``'s raw label.
    location:
        ``(filename, line)`` of this ``__bool__`` call.
    locations:
        The trace's per-label consumer locations recorded so far.

    Returns
    -------
    str
        ``label`` itself, or the label of a clone logged for this check.
    """

    entries = locations.get(label, ())
    if not entries or location in entries or source.ndim != 0:
        return label
    counter = _layer_counter(trace)
    last = _LAST_CLONED_CHECK.get(trace)
    if last is not None and last[:2] == (label, location) and last[3] == counter:
        return last[2]
    if not _is_conditional_test(location):
        return label
    if not any(_is_conditional_test(entry) for entry in entries):
        return label
    clone_label = _log_clone_label(trace, source)
    if clone_label is None:
        return label
    after = _layer_counter(trace)
    if after is not None:
        _LAST_CLONED_CHECK[trace] = (label, location, clone_label, after)
    return clone_label
