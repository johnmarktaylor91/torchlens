"""Record schemas of the 3.0 saved format, derived from the field bindings.

One schema per record kind: the ``columns`` of ``records/<record>.json`` are the
record's persisted field names, each cell typed by its ``StorageKind`` and
annotation; nested families (timing provenance, checkpoint witnesses,
intervention audit rows, kernel launches, capture advisories, ...) are
declarative sub-schemas in ``FAMILIES`` that the one decoder applies. The
schema digest is stamped into the manifest so a binding change is visible
without a contract bump. Design: ``docs/maintainers/saved_format_3_design.md``,
sections 3, 4 and 9. Filled by the writer stream.
"""

from __future__ import annotations

from typing import Any, NoReturn

from .contract import refuse_writer_not_built

_LANE_MODULE = "torchlens/_io/format3/schema.py"

#: Nested-family sub-schemas, keyed by family name; filled by the writer stream.
FAMILIES: dict[str, dict[str, Any]] = {}


def record_schema(record: str) -> NoReturn:
    """Return the typed schema of one record kind (``op``, ``trace``, ...).

    Raises
    ------
    ArtifactError
        ``bundle_save_failed``: the schema table is a contract stub only.
    """
    refuse_writer_not_built(f"record_schema({record!r})", _LANE_MODULE)


def schema_digest() -> NoReturn:
    """Return the digest of every record schema, as stamped into the manifest."""
    refuse_writer_not_built("schema_digest", _LANE_MODULE)
