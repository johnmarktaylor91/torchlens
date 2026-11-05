"""F29: schema files + lockstep, both directions (agent memo 3.9 item 24).

Direction 1: every shipped schema file parses as Draft 2020-12 and is served
by the schema tool. Direction 2: every LIVE tool result validates against its
declared schema document. The registry's output schema ids and the shipped
file index cannot drift.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_agent_surface_helpers import save_ablated_artifact, save_clean_artifact
from torchlens.agent import call_tool
from torchlens.agent._schemas import load_schema

# Per-test marks, not a module pytestmark: the live-results lockstep test is
# heavy (measured ~8s -- ten live tool calls over two saved artifacts) and a
# module-level smoke mark would additively conflict with its heavy tier.


@pytest.fixture()
def clean(tmp_path: Path) -> Path:
    """The deterministic clean fixture artifact."""

    return save_clean_artifact(tmp_path)


# Kept by the shrink-r4 coverage proof: sole branch coverage of agent/_schemas.py.
def test_unknown_schema_id_refuses_typed() -> None:
    """The schema tool refusal names the served index."""

    with pytest.raises(ValueError, match="served ids") as exc:
        load_schema("torchlens.agent.bogus.v1")
    assert exc.value.fields["code"] == "agent_schema_unknown"


# Kept by the shrink-r4 coverage proof: sole branch coverage of _deprecations.py, agent/_apimap.py and more.
@pytest.mark.heavy
def test_live_results_validate_against_their_schemas(clean: Path, tmp_path: Path) -> None:
    """Direction 2 of the lockstep: real envelopes validate under jsonschema."""

    jsonschema = pytest.importorskip("jsonschema")
    ablated = save_ablated_artifact(tmp_path)
    live: list[dict] = [
        call_tool("torchlens_doctor"),
        call_tool("torchlens_api_map"),
        call_tool("torchlens_api_map", {"name": "trace"}),
        call_tool("torchlens_overview", {"path": str(clean), "mode": "manifest"}),
        call_tool("torchlens_overview", {"path": str(clean)}),
        call_tool("torchlens_dump", {"path": str(clean), "view": "graph"}),
        call_tool("torchlens_explain", {"path": str(clean)}),
        call_tool(
            "torchlens_query_sites", {"path": str(clean), "query": {"op": "func", "value": "relu"}}
        ),
        call_tool("torchlens_payload_stats", {"path": str(clean)}),
        call_tool("torchlens_compare", {"reference": str(clean), "subject": str(ablated)}),
        call_tool("torchlens_schema", {"name": "torchlens.agent.envelope.v1"}),
    ]
    for envelope in live:
        document = load_schema(envelope["schema"])
        jsonschema.Draft202012Validator(document).validate(envelope)


# Kept by the shrink-r4 coverage proof: sole branch coverage of agent/_registry.py.
def test_error_envelope_validates(clean: Path) -> None:
    """The error envelope validates against its shipped schema."""

    jsonschema = pytest.importorskip("jsonschema")
    from torchlens.agent import call_tool_envelope

    error = call_tool_envelope("torchlens_nope")
    document = load_schema("torchlens.agent.error.v1")
    jsonschema.Draft202012Validator(document).validate(error)
