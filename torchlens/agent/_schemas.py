"""The shipped schema registry: Draft 2020-12 files, fetchable at runtime.

Every agent-surface input/output/error schema ships as a JSON Schema file in
the wheel (``torchlens/schemas/agent_*.json``), validated in CI both
directions against live results (the error-contract lockstep precedent), and
served by the ``torchlens_schema`` tool so the contract stays fetchable
through the channel that emits results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .._errors import InvalidArgumentError

#: Result schema id of the schema tool itself.
SCHEMA_TOOL_SCHEMA = "torchlens.agent.schema.v1"

#: Directory holding the shipped schema documents.
_SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"

#: schema id -> shipped filename. The ONE naming authority.
SCHEMA_FILES: dict[str, str] = {
    "torchlens.agent.envelope.v1": "agent_envelope_v1.json",
    "torchlens.agent.error.v1": "agent_error_v1.json",
    "torchlens.agent.doctor.v1": "agent_doctor_v1.json",
    "torchlens.agent.api_map.v2": "agent_api_map_v2.json",
    "torchlens.agent.overview_manifest.v1": "agent_overview_manifest_v1.json",
    "torchlens.agent.overview_folded.v1": "agent_overview_folded_v1.json",
    "torchlens.agent.dump.v1": "agent_dump_v1.json",
    "torchlens.agent.explain.v1": "agent_explain_v1.json",
    "torchlens.agent.query_sites.v1": "agent_query_sites_v1.json",
    "torchlens.agent.payload_stats.v1": "agent_payload_stats_v1.json",
    "torchlens.agent.compare.v1": "agent_compare_v1.json",
    "torchlens.agent.schema.v1": "agent_schema_v1.json",
}


def schema_index() -> list[str]:
    """Return every served schema id (the machine-readable index)."""

    return sorted(SCHEMA_FILES)


def load_schema(schema_id: str) -> dict[str, Any]:
    """Load one shipped Draft 2020-12 schema document.

    Parameters
    ----------
    schema_id:
        Versioned schema id (``torchlens.agent.overview_folded.v1``).

    Returns
    -------
    dict[str, Any]
        Parsed schema document.

    Raises
    ------
    InvalidArgumentError
        ``agent_schema_unknown`` when the id is not served.
    """

    filename = SCHEMA_FILES.get(schema_id)
    if filename is None:
        raise InvalidArgumentError(
            f"{schema_id!r} is not a served schema id",
            code="agent_schema_unknown",
            remedy=f"fetch the index first; served ids: {', '.join(schema_index())}",
        )
    from ._artifacts import _loads_bounded

    return _loads_bounded((_SCHEMA_DIR / filename).read_text(encoding="utf-8"))
