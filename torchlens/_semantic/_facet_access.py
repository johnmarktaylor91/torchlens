"""Access-time facet checks: validated reconstructions and a live owning Trace.

A reconstructed facet (fused-SDPA ``scores``/``pattern``/``z``/``result``) is
validated when ``FacetView`` serves it or ``declared()`` reports it; a refusal
raises ``MissingFacetError`` at access, is reported ``needs_capture`` and carries
``EAGER_ATTENTION_REMEDY``. This module holds the parts that need no facet types.

A Module or ModuleCall holds its Trace weakly. Once the Trace is collected the
recipes see no ops, so every facet would read as structurally absent; the view
refuses with the package's collected-Trace ``RecordBindingError`` instead.
"""

from __future__ import annotations

import weakref
from collections.abc import Iterable
from typing import Any

from .._errors import RecordBindingError
from .._utils._record_group_table import record_member
from ._call_subject import CallBoundModule

#: Recapture hint for a refused reconstruction: eager attention records the
#: pattern as a real op, so nothing has to be reconstructed.
EAGER_ATTENTION_REMEDY = (
    "a capture of the model running eager (unfused) attention, which records the "
    "pattern as a real op (Hugging Face: attn_implementation='eager')"
)


def refuse_collected_trace(record: Any) -> None:
    """Raise when ``record``'s owning Trace has been garbage-collected.

    Parameters
    ----------
    record:
        Record a facet view reads (Module, ModuleCall, or a call-bound Module).

    Raises
    ------
    RecordBindingError
        ``trace_reference_collected`` when the weak Trace reference is dead.
        A record never bound to a Trace passes: its facets come from the record.
    """

    if isinstance(record, CallBoundModule):
        record, kind = record._call, "ModuleCall"
    else:
        kind = type(record).__name__
    ref = vars(record).get("_source_trace_ref") if hasattr(record, "__dict__") else None
    if isinstance(ref, weakref.ReferenceType) and ref() is None:
        raise RecordBindingError(
            f"Trace has been garbage-collected; a {kind} does not keep its Trace alive",
            code="trace_reference_collected",
            remedy="keep the Trace (t = tl.trace(...)) or fork (f = t.fork().do(...)) in a variable while reading its records",
        )


def absent_text(record: Any, available: Iterable[Any], name: Any) -> tuple[str, str]:
    """Return ``(remedy, detail)`` for a facet no registered recipe produces here.

    Parameters
    ----------
    record:
        Record the facet view reads.
    available:
        Facet keys the view does produce.
    name:
        Facet key requested.
    """

    owner = record_member(record, "address", None) or getattr(record, "label", None) or "record"
    class_name = getattr(record, "class_name", None) or _owning_module_class_name(record)
    keys = sorted(str(key) for key in available)
    detail = (
        f"no registered recipe for {owner!r} ({class_name}) produces it; available facets: "
        f"{keys if keys else 'none'}"
    )
    remedy = (
        f"register a recipe with @torchlens.facets.register(class_name={class_name!r}) "
        f"on a function returning {{{str(name)!r}: ...}}"
    )
    return remedy, detail


def _owning_module_class_name(record: Any) -> str:
    """Return the module class name behind a ModuleCall-like record, else its type name."""

    try:
        owner = getattr(record, "module", None)
    except (KeyError, RuntimeError, TypeError, ValueError):
        # A refusal message must not raise a second error.
        owner = None
    owner_class = getattr(owner, "class_name", None) if not isinstance(owner, str) else None
    return str(owner_class) if owner_class else type(record).__name__
