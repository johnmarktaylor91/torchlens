"""F29: registry, envelope, canonical serializer, budgets, continuation.

The one-core/three-transports law (agent memo 3.1) plus the budget and
truncation discipline (3.8) and the envelope/determinism contract (3.9).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import torchlens as tl
from tests.test_agent_surface_helpers import save_clean_artifact
from torchlens.agent import (
    _budgets,
    build_error_envelope,
    call_tool,
    call_tool_envelope,
    canonical_dumps,
    json_safe,
    list_tools,
    tool_specs,
)

# Per-test marks, not a module pytestmark: the every-envelope contract test is
# heavy (measured ~7s cold -- it exercises every registry tool over the live
# artifact, paying the lazy stats/query/dump imports) and a module-level smoke
# mark would additively conflict with its heavy tier.


@pytest.fixture()
def artifact(tmp_path: Path) -> Path:
    """The deterministic clean fixture artifact."""

    return save_clean_artifact(tmp_path)


def test_canonical_serializer_is_strict_and_sorted() -> None:
    """Sorted keys, compact separators, ASCII, and NO bare non-finites."""

    text = canonical_dumps(json_safe({"b": 1, "a": {"z": [1.5, float("nan")]}}))
    assert text == '{"a":{"z":[1.5,{"nonfinite":"nan"}]},"b":1}'
    with pytest.raises(ValueError):
        canonical_dumps({"x": float("inf")})  # unsanitized non-finite is a crash here
    parsed = json.loads(text, parse_constant=pytest.fail)  # strict constant hook
    assert parsed["a"]["z"][1] == {"nonfinite": "nan"}


def test_json_safe_tags_every_nonfinite_direction() -> None:
    """nan/inf/-inf become tagged records; tuples become lists."""

    assert json_safe(float("nan")) == {"nonfinite": "nan"}
    assert json_safe(float("inf")) == {"nonfinite": "inf"}
    assert json_safe(float("-inf")) == {"nonfinite": "-inf"}
    assert json_safe((1, 2)) == [1, 2]
    assert math.isclose(json_safe(1.25), 1.25)


@pytest.mark.heavy
def test_every_envelope_carries_the_contract_keys(artifact: Path) -> None:
    """Schema, stability, version, status, echoes, truncation on EVERY payload."""

    for name, args in [
        ("torchlens_doctor", {}),
        ("torchlens_api_map", {}),
        ("torchlens_overview", {"path": str(artifact)}),
        ("torchlens_schema", {}),
    ]:
        envelope = call_tool(name, args)
        for key in (
            "schema",
            "schema_stability",
            "torchlens_version",
            "status",
            "artifact",
            "request",
            "limits",
            "data",
            "truncation",
            "warnings",
        ):
            assert key in envelope, (name, key)
        assert envelope["schema_stability"] == "documented-unstable"
        canonical_dumps(envelope)  # strict-serializable as-is


def test_no_absolute_path_in_machine_payloads(artifact: Path) -> None:
    """basename is a courtesy; no absolute/cwd-derived path appears outside the echo."""

    envelope = call_tool("torchlens_overview", {"path": str(artifact)})
    envelope["request"] = {}
    assert str(artifact.parent) not in canonical_dumps(envelope)
    assert envelope["artifact"]["basename"] == artifact.name


@pytest.mark.smoke
def test_error_envelope_exposes_typed_fields() -> None:
    """The error envelope carries class/code/remedy/severity, no traceback."""

    try:
        call_tool("torchlens_nope")
    except ValueError as exc:  # InvalidArgumentError subclasses ValueError
        envelope = build_error_envelope(exc)
    assert envelope["schema"] == "torchlens.agent.error.v1"
    assert envelope["status"] == "error"
    assert envelope["error"]["code"] == "agent_tool_unknown"
    assert envelope["error"]["remedy"]
    assert "Traceback" not in canonical_dumps(envelope)
    assert call_tool_envelope("torchlens_nope")["error"]["code"] == "agent_tool_unknown"


def test_arguments_validate_against_the_declared_schema(artifact: Path) -> None:
    """Unknown keys, wrong types, and closed-vocabulary misses refuse typed."""

    cases = [
        ({"path": str(artifact), "bogus": 1}, "does not take"),
        ({"path": 7}, "must be string"),
        ({"path": str(artifact), "mode": "prose"}, "closed vocabulary"),
    ]
    for args, needle in cases:
        with pytest.raises(ValueError, match=needle) as exc:
            call_tool("torchlens_overview", args)
        assert exc.value.fields["code"] == "agent_argument_invalid"


def test_limits_lower_never_raise(artifact: Path) -> None:
    """Requests lower ceilings; zero and above-ceiling refuse (no 0-as-infinity)."""

    for bad in (0, -1, _budgets.DEFAULT_MAX_ROWS + 1, True):
        with pytest.raises(ValueError) as exc:
            call_tool("torchlens_dump", {"path": str(artifact), "view": "graph", "max_rows": bad})
        assert exc.value.fields["code"] in ("agent_limit_invalid", "agent_argument_invalid")


def test_token_backstop_drops_rows_with_disclosure(artifact: Path) -> None:
    """A tight max_tokens truncates rows AFTER the row cap, disclosed."""

    envelope = call_tool("torchlens_query_sites", {"path": str(artifact), "max_tokens": 900})
    assert envelope["status"] == "ok"
    assert envelope["truncation"]["omitted"] > 0
    assert "how_to_get_more" in envelope["truncation"]
    assert _budgets.estimate_tokens(canonical_dumps(envelope)) <= 900


def test_budget_floor_is_never_a_plausible_fragment(artifact: Path) -> None:
    """When the floor alone exceeds the budget, the result SAYS so."""

    envelope = call_tool("torchlens_query_sites", {"path": str(artifact), "max_tokens": 1})
    assert envelope["status"] == "budget_floor_exceeded"
    assert envelope["data"] == {"budget_floor_exceeded": True}


def test_continuation_refuses_on_request_drift(artifact: Path) -> None:
    """A continuation minted under one query refuses under another."""

    page = call_tool("torchlens_dump", {"path": str(artifact), "view": "graph", "max_rows": 2})
    next_struct = page["data"]["next"]
    with pytest.raises(ValueError, match="continuation") as exc:
        call_tool(
            "torchlens_dump",
            {
                "path": str(artifact),
                "view": "graph",
                "max_rows": 3,  # different request -> different request_hash
                "continuation": next_struct,
            },
        )
    assert exc.value.fields["code"] == "agent_continuation_drift"


def test_paged_rows_reassemble_byte_exactly(artifact: Path) -> None:
    """Concatenated pages equal the unpaged ordered result."""

    unpaged = call_tool("torchlens_dump", {"path": str(artifact), "view": "graph"})
    rows: list[dict] = []
    continuation = None
    while True:
        args = {"path": str(artifact), "view": "graph", "max_rows": 2}
        if continuation is not None:
            args["continuation"] = continuation
        page = call_tool("torchlens_dump", args)
        rows.extend(page["data"]["rows"])
        continuation = page["data"]["next"]
        if continuation is None:
            break
    assert canonical_dumps(rows) == canonical_dumps(unpaged["data"]["rows"])


def test_registry_annotations_and_verbs_are_complete() -> None:
    """Nine tools, all read-only + idempotent, CLI verbs unique where present."""

    specs = tool_specs()
    assert len(specs) == 9
    assert all(spec.read_only and spec.idempotent for spec in specs)
    verbs = [spec.cli_verb for spec in specs if spec.cli_verb]
    assert len(verbs) == len(set(verbs))
    assert {declaration["name"] for declaration in list_tools()} == {spec.name for spec in specs}


@pytest.mark.smoke
def test_python_facade_reaches_the_agent_surface() -> None:
    """tl.agent resolves on a cold facade and serves the same call_tool."""

    assert tl.agent.call_tool is call_tool
    assert callable(tl.agent.guide)
    assert "TorchLens for AI agents" in tl.agent.guide()
