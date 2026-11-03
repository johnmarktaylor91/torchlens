"""F29: query_sites -- closed AST, ceilings, listing semantics, FF2 closure.

Every refusal is typed and names the Python path; listing resolves the FULL
population (no fanout cap); every result carries a runnable handoff whose
verbatim execution reproduces the matched labels (the FF2 self-teaching
closure made mechanical).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import torchlens as tl
from tests.test_agent_surface_helpers import save_clean_artifact
from torchlens.agent import call_tool
from torchlens.agent._query import (
    MAX_QUERY_DEPTH,
    MAX_QUERY_NODES,
    MAX_QUERY_STRING,
    validate_query,
)


@pytest.fixture()
def artifact(tmp_path: Path) -> Path:
    """The deterministic clean fixture artifact."""

    return save_clean_artifact(tmp_path)


# Kept by the shrink-r4 coverage proof: sole branch coverage of agent/_query.py.
def test_refused_kinds_name_the_python_path() -> None:
    """regex/where/changed/callable refuse typed, teaching the Python spelling."""

    for kind, needle in [
        ("regex", "glob"),
        ("where", "Python"),
        ("changed", "tl.changed"),
        ("callable", "Python"),
    ]:
        with pytest.raises(ValueError, match=needle) as exc:
            validate_query({"op": kind, "value": "x"})
        assert exc.value.fields["code"] == "agent_query_invalid"


# Kept by the shrink-r4 coverage proof: sole branch coverage of agent/_query.py.
def test_decoder_ceilings_trip_before_resolution() -> None:
    """Depth, node-count, and string ceilings refuse with the DoS rationale."""

    deep: dict = {"op": "func", "value": "relu"}
    for _ in range(MAX_QUERY_DEPTH + 1):
        deep = {"op": "not", "item": deep}
    with pytest.raises(ValueError) as depth_exc:
        validate_query(deep)
    assert depth_exc.value.fields["code"] == "agent_query_ceiling"

    wide = {"op": "or", "items": [{"op": "func", "value": "relu"}] * (MAX_QUERY_NODES + 1)}
    with pytest.raises(ValueError) as node_exc:
        validate_query(wide)
    assert node_exc.value.fields["code"] == "agent_query_ceiling"

    with pytest.raises(ValueError) as string_exc:
        validate_query({"op": "contains", "value": "x" * (MAX_QUERY_STRING + 1)})
    assert string_exc.value.fields["code"] == "agent_query_ceiling"


# Kept by the shrink-r4 coverage proof: sole branch coverage of agent/_query.py.
def test_combinators_and_closure_semantics(artifact: Path) -> None:
    """and/not and the followed_by transitive closure match the graph truth."""

    conv_before_relu = {
        "op": "and",
        "items": [
            {"op": "func", "value": "linear"},
            {"op": "followed_by", "item": {"op": "func", "value": "relu"}},
        ],
    }
    envelope = call_tool(
        "torchlens_query_sites", {"path": str(artifact), "query": conv_before_relu}
    )
    labels = [row["label"] for row in envelope["data"]["rows"]]
    # Every block linear feeds a relu downstream; the head linear does not.
    assert labels == ["linear_1_1:1", "linear_2_3:1", "linear_3_5:1"]

    not_saved = {"op": "not", "item": {"op": "saved", "value": True}}
    envelope = call_tool("torchlens_query_sites", {"path": str(artifact), "query": not_saved})
    assert all(row["saved"] is False for row in envelope["data"]["rows"])


# Kept by the shrink-r4 coverage proof: sole branch coverage of agent/_query.py.
def test_handoff_runs_verbatim_and_reproduces_the_match(artifact: Path) -> None:
    """The handoff field executes against the loaded trace and agrees exactly."""

    for query in [
        {"op": "func", "value": "relu"},
        {"op": "and", "items": [{"op": "func", "value": "relu"}, {"op": "saved", "value": True}]},
        {"op": "in_module", "value": "blocks.1"},
        {"op": "followed_by", "item": {"op": "func", "value": "relu"}},
    ]:
        envelope = call_tool("torchlens_query_sites", {"path": str(artifact), "query": query})
        handoff = envelope["data"]["handoff"]
        served = [row["label"] for row in envelope["data"]["rows"]]
        scope: dict = {"log": tl.load(str(artifact))}
        if "\n" in handoff:
            exec(handoff, scope)  # noqa: S102 - the handoff contract IS verbatim execution
            local = scope["matched"]
        else:
            local = eval(handoff, scope)  # noqa: S307 - same contract, expression form
        assert local == served, (query, handoff)
