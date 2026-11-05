"""Refusal diagnostics for live state fields missing from a portable state spec."""

from __future__ import annotations

import functools
import inspect
from typing import Any

from . import ArtifactError


def _refuse_undeclared_field(owner: Any, field_name: str, owner_is_trace: bool) -> None:
    """Refuse a live field missing from the spec unless it is a cached_property cell."""

    if not _is_cached_property_cell(type(owner), field_name):
        raise _undeclared_field_error(owner, field_name, owner_is_trace)


def _is_cached_property_cell(owner_type: type, field_name: str) -> bool:
    """Return whether an undeclared field is a ``functools.cached_property`` cell.

    A ``cached_property`` read caches its value in the instance ``__dict__``
    under the property's own name (e.g. the public ``Trace.recipe`` accessor).
    Those cells are DERIVED state that rebuilds on access, never portable
    fields, and they appear only after a read -- refusing them made a
    read-only public property poison every later ``Trace.save``.
    The check runs only on the refusal path, so the hot field loop pays nothing.
    """

    return isinstance(
        inspect.getattr_static(owner_type, field_name, None), functools.cached_property
    )


def _undeclared_field_hint(value: Any, field_name: str, owner_is_trace: bool) -> str:
    """Name the writer that populated an undeclared field, when one is known.

    An undeclared ``_<name>_cache`` cell is almost always DERIVED state poked
    into ``__dict__`` by a lazily-caching public accessor read (the
    ModuleCall/Module ``.facets`` incident): name that accessor when it
    exists. A LEDGERED-but-undeclared Trace transient (the draw()
    ``_last_encoding_state`` incident) quotes its ledger row so the message
    names the writer. Either way the message points at the read that
    poisoned the save, not just an internal field the user never heard of.
    """

    if field_name.startswith("_") and field_name.endswith("_cache"):
        accessor_name = field_name[1 : -len("_cache")]
        if isinstance(inspect.getattr_static(type(value), accessor_name, None), property):
            return (
                f" This cell was populated by reading the public "
                f"`.{accessor_name}` accessor; a lazy derived cache "
                f"must be declared FieldPolicy.DROP in "
                f"{type(value).__name__}._PORTABLE_STATE_SPEC so a "
                f"read-only access cannot poison a later save."
            )
    if owner_is_trace:
        from .._data_classes._trace_components import TRACE_EXTERNAL_WRITE_EXEMPTIONS

        ledger_reason = TRACE_EXTERNAL_WRITE_EXEMPTIONS.get(field_name)
        if ledger_reason is not None:
            return (
                f" This field is ledgered in "
                f"TRACE_EXTERNAL_WRITE_EXEMPTIONS as: {ledger_reason!r}."
                f" The ledger documents the write; it is not a scrub "
                f"policy. Enroll the runtime-only transient in the "
                f"scrub's runtime-only set (or give it a "
                f"FieldPolicy.DROP row) so populating it cannot "
                f"poison a later save."
            )
    return ""


def _undeclared_field_error(value: Any, field_name: str, owner_is_trace: bool) -> ArtifactError:
    """Build the refusal for a live state field with no portability policy."""

    return ArtifactError(
        f"{type(value).__name__}.{field_name} is missing from "
        f"_PORTABLE_STATE_SPEC. Every live state field needs an "
        f"explicit portability policy before it can be saved."
        f"{_undeclared_field_hint(value, field_name, owner_is_trace)}"
    )
