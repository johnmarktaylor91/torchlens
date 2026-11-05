"""Read a module edit's replacement op at the site it replaced.

A ``tl.module(addr)`` edit, at capture time (``intervene=tl.when(...)``) or
post-capture (``fork.do(tl.module(addr), edit)``), leaves the module's last op
at the value it computed and records the edited value in its own
``interventionreplacement`` op between that op and its consumers. Comparison
reads pair ops across two traces (``tl.debug.compare`` and
``first_divergence``, ``Bundle.compare``, ``rank_changes`` and the bundle-diff
site deltas); the unedited trace has no replacement op, so a one-to-one
pairing skips exactly the edited value. These helpers let a read compare the
replacement at the site it replaced: :func:`replaced_site` maps a replacement
to that op, :func:`edited_stand_in` maps the op to its replacement.

A replacement is mapped only when it records its parent: exactly one parent,
which resolves to an op of the same trace that lists the replacement among
its children. Any other op is left alone, so a read still reports it on its
own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._errors import RecordBindingError
from .._utils._multipass_access import get_multipass_attr

if TYPE_CHECKING:
    from .op import Op
    from .trace import Trace

#: ``func_name`` of a module edit's replacement op (capture time and post-capture alike).
REPLACEMENT_FUNC_NAME = "intervention_replacement"
#: Layer type of that op: its labels start ``interventionreplacement_``.
_REPLACEMENT_LABEL_PREFIX = "interventionreplacement_"


def _owning_trace(op: Any) -> Trace | None:
    """The trace ``op`` belongs to, or ``None`` when it is detached."""

    try:
        return op.source_trace  # type: ignore[no-any-return]
    except (AttributeError, RecordBindingError):
        return None


def _spellings(op: Any) -> set[str]:
    """The labels a parent or child list may name ``op`` by."""

    names = {str(op.label)}
    if int(getattr(op, "num_passes", 1) or 1) <= 1:
        names.add(str(op.layer_label))
    return names


def _op_named(trace: Trace, label: str) -> Op | None:
    """The op of ``trace`` that ``label`` names, or ``None``."""

    for op in trace._layer_list:
        if label in _spellings(op):
            return op
    return None


def _replaced_parent(op: Any, trace: Trace) -> Op | None:
    """The op ``op`` replaced when ``op`` is a replacement recording its parent, else ``None``."""

    if getattr(op, "func_name", None) != REPLACEMENT_FUNC_NAME:
        return None
    parents = tuple(getattr(op, "parents", ()) or ())
    if len(parents) != 1:
        return None
    parent = _op_named(trace, str(parents[0]))
    if parent is None:
        return None
    children = {str(child) for child in (getattr(parent, "children", ()) or ())}
    return parent if children & _spellings(op) else None


def replaced_site(op: Any, trace: Trace | None = None) -> Any:
    """Return the op whose site a replacement op stands at; any other op is returned as is.

    Parameters
    ----------
    op:
        An op of ``trace``.
    trace:
        The trace ``op`` belongs to; defaults to ``op``'s owning trace.

    Returns
    -------
    Any
        The edited module's last op for a replacement that records it as its
        parent (followed through a replacement of a replacement), else ``op``.
    """

    owner = trace if trace is not None else _owning_trace(op)
    if owner is None:
        return op
    site = op
    for _ in range(len(owner._layer_list)):
        parent = _replaced_parent(site, owner)
        if parent is None:
            return site
        site = parent
    return site


def is_mapped_replacement(op: Any, trace: Trace | None = None) -> bool:
    """Whether ``op`` is a replacement a read compares at its parent's site."""

    return replaced_site(op, trace) is not op


def edited_stand_in(op: Any, trace: Trace) -> Any:
    """Return the op holding ``op``'s edited module-output value; ``op`` itself when unedited.

    Parameters
    ----------
    op:
        An op (or an op-like record carrying ``label`` and ``children``) of ``trace``.
    trace:
        The trace the stand-in is looked up in.

    Returns
    -------
    Any
        The last replacement op in the chain that records ``op`` as its
        parent, else ``op``.
    """

    current = op
    for _ in range(len(trace._layer_list)):
        follower = _replacement_child(current, trace)
        if follower is None:
            return current
        current = follower
    return current


def _replacement_child(op: Any, trace: Trace) -> Op | None:
    """The replacement op that records ``op`` as its parent, or ``None``."""

    # An aggregate multi-pass Layer has no single output to stand in for: it reads as itself.
    children = get_multipass_attr(op, "children", (), multipass=()) or ()
    labels = [str(c) for c in children if str(c).startswith(_REPLACEMENT_LABEL_PREFIX)]
    if not labels:
        return None  # the common case: no replacement child, no label scan
    names = _spellings(op)
    for child_label in labels:
        child = _op_named(trace, child_label)
        if child is None or getattr(child, "func_name", None) != REPLACEMENT_FUNC_NAME:
            continue
        parent = _replaced_parent(child, trace)
        if parent is not None and _spellings(parent) & names:
            return child
    return None
