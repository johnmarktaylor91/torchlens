"""Cross-record invariants of the 3.0 saved format, run after every table is decoded.

Grammar (shape and vocabulary) is the decoder's job through the schemas; this
module holds the checks that need more than one cell: site keys name existing
ops, checkpoint windows order, structure-only coherence, edge carriers and
graph structure reference known rows, kernel relations, identity facts,
injection provenance and evidence facts. Codes are the existing
``artifact_*`` rows. Design: ``docs/maintainers/saved_format_3_design.md``,
section 9. Filled by the reader stream.
"""

from __future__ import annotations

from typing import Any, NoReturn

from .contract import refuse_reader_not_built

_LANE_MODULE = "torchlens/_io/format3/validate.py"


def validate_records(tables: dict[str, Any]) -> NoReturn:
    """Run every cross-record invariant over the decoded tables; refuse on the first failure."""
    refuse_reader_not_built("validate_records", _LANE_MODULE)
