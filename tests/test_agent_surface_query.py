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

pytestmark = pytest.mark.smoke


@pytest.fixture()
def artifact(tmp_path: Path) -> Path:
    """The deterministic clean fixture artifact."""

    return save_clean_artifact(tmp_path)


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


def test_listing_resolves_the_full_population(artifact: Path) -> None:
    """No max_fanout on listing: 3 relus resolve where intervention would cap."""

    envelope = call_tool(
        "torchlens_query_sites", {"path": str(artifact), "query": {"op": "func", "value": "relu"}}
    )
    header = envelope["data"]["header"]
    assert header["matches_total"] == 3
    assert header["population_total"] == 9
    labels = [row["label"] for row in envelope["data"]["rows"]]
    assert labels == ["relu_1_2:1", "relu_2_4:1", "relu_3_6:1"]  # execution order


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


def test_ff2_match_reason_plus_row_fields_reproduce_locally(artifact: Path) -> None:
    """FF2: every row's disclosed reason is checkable from the row itself."""

    query = {"op": "func", "value": "relu"}
    envelope = call_tool("torchlens_query_sites", {"path": str(artifact), "query": query})
    for row in envelope["data"]["rows"]:
        assert row["matched"] == [{"op": "func", "value": "relu"}]
        assert row["func_name"] == "relu"  # the row field alone reproduces the reason


def test_ff1_module_containment_is_populated(artifact: Path) -> None:
    """FF1: every in-module op row carries a non-null containment address."""

    envelope = call_tool("torchlens_query_sites", {"path": str(artifact)})
    rows = envelope["data"]["rows"]
    in_module = [row for row in rows if row["module_call_stack"]]
    assert in_module
    assert all(row["module_containment"] for row in in_module)
    # Stratification: no single containment bucket swallows the population.
    buckets: dict[str, int] = {}
    for row in in_module:
        buckets[row["module_containment"]] = buckets.get(row["module_containment"], 0) + 1
    assert max(buckets.values()) <= max(1, len(in_module) // 2)


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
