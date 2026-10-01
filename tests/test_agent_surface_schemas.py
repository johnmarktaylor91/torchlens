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
from torchlens.agent import call_tool, tool_specs
from torchlens.agent._schemas import SCHEMA_FILES, load_schema, schema_index

# Per-test marks, not a module pytestmark: the live-results lockstep test is
# heavy (measured ~8s -- ten live tool calls over two saved artifacts) and a
# module-level smoke mark would additively conflict with its heavy tier.


@pytest.fixture()
def clean(tmp_path: Path) -> Path:
    """The deterministic clean fixture artifact."""

    return save_clean_artifact(tmp_path)


@pytest.mark.smoke
def test_every_schema_file_ships_and_parses() -> None:
    """Index == files on disk; every document is Draft 2020-12 with an $id."""

    for schema_id in schema_index():
        document = load_schema(schema_id)
        assert document["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema_id in document["$id"]
        assert document["title"]


@pytest.mark.smoke
def test_registry_output_schemas_are_all_served() -> None:
    """Every registry output schema id has a shipped document; none dangle."""

    declared = {spec.output_schema for spec in tool_specs()}
    served = set(schema_index())
    assert declared <= served
    # The overview tool serves TWO schemas (distinct modes, by contract).
    assert "torchlens.agent.overview_manifest.v1" in served
    # The envelope + error schemas ship even though no tool declares them.
    assert {"torchlens.agent.envelope.v1", "torchlens.agent.error.v1"} <= served
    assert set(SCHEMA_FILES) == served


@pytest.mark.smoke
def test_unknown_schema_id_refuses_typed() -> None:
    """The schema tool refusal names the served index."""

    with pytest.raises(ValueError, match="served ids") as exc:
        load_schema("torchlens.agent.bogus.v1")
    assert exc.value.fields["code"] == "agent_schema_unknown"


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


@pytest.mark.smoke
def test_error_envelope_validates(clean: Path) -> None:
    """The error envelope validates against its shipped schema."""

    jsonschema = pytest.importorskip("jsonschema")
    from torchlens.agent import call_tool_envelope

    error = call_tool_envelope("torchlens_nope")
    document = load_schema("torchlens.agent.error.v1")
    jsonschema.Draft202012Validator(document).validate(error)
