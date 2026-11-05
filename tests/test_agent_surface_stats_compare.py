"""F29: payload_stats and compare -- bounded numbers, coverage honesty.

Stats: deterministic float64 metrics against a torch reference, tagged
non-finites through the strict serializer, per-row typed statuses (never a
batch failure), byte refusal BEFORE materialization, one retained axis.
Compare: subject-minus-reference direction, site-key matching, the
non-droppable coverage header, and the intervention boundary recovered from
two real artifacts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import torchlens as tl
from tests.test_agent_surface_helpers import (
    save_ablated_artifact,
    save_clean_artifact,
    save_nan_artifact,
)
from torchlens.agent import _budgets, call_tool, canonical_dumps


@pytest.fixture()
def clean(tmp_path: Path) -> Path:
    """The deterministic clean fixture artifact."""

    return save_clean_artifact(tmp_path)


@pytest.fixture()
def ablated(tmp_path: Path) -> Path:
    """The middle-block zero-ablated sibling artifact."""

    return save_ablated_artifact(tmp_path)


# Kept by the shrink-r4 coverage proof: sole branch coverage of agent/_envelope.py, agent/_query.py and more.
def test_nonfinite_values_arrive_tagged_never_bare(tmp_path: Path) -> None:
    """NaN payload stats serialize under allow_nan=False with tagged records."""

    nan_artifact = save_nan_artifact(tmp_path)
    envelope = call_tool(
        "torchlens_payload_stats",
        {"path": str(nan_artifact), "query": {"op": "contains", "value": "truediv"}},
    )
    text = canonical_dumps(envelope)
    # The strict constant hook FAILS on any bare NaN/Infinity/-Infinity
    # literal ("NaN" as substring is fine -- the fixture model is NaNHead).
    parsed = json.loads(text, parse_constant=pytest.fail)
    rows = parsed["data"]["rows"]
    assert rows and rows[0]["nan_count"] + rows[0]["posinf_count"] + rows[0]["neginf_count"] > 0


# Kept by the shrink-r4 coverage proof: sole branch coverage of agent/_stats.py.
def test_byte_budget_refuses_before_materialization(
    clean: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A payload above the per-blob ceiling refuses per-row, pre-read."""

    monkeypatch.setattr(_budgets, "MAX_BLOB_BYTES", 1)
    envelope = call_tool("torchlens_payload_stats", {"path": str(clean), "labels": ["relu_1_2:1"]})
    row = envelope["data"]["rows"][0]
    assert row["status"] == "refused_blob_budget"
    assert row["declared_bytes"] > 1
    assert "trace['relu_1_2:1'].out" in row["remedy"]
    assert envelope["data"]["header"]["bytes_materialized"] == 0


# Kept by the shrink-r4 coverage proof: sole branch coverage of agent/_registry.py, agent/_stats.py.
def test_retained_axis_reduction_is_explicit_and_ranked(clean: Path) -> None:
    """One explicit dimension INDEX; over-extent requires a ranking metric."""

    envelope = call_tool(
        "torchlens_payload_stats",
        {"path": str(clean), "labels": ["relu_1_2:1"], "reduction": {"retain_dim": 1}},
    )
    reduction = envelope["data"]["rows"][0]["reduction"]
    assert reduction["retain_dim"] == 1
    assert reduction["extent"] == 8
    assert len(reduction["rows"]) == 8
    log = tl.load(str(clean))
    reference = log["relu_1_2"].out.double()
    assert math.isclose(reduction["rows"][0]["mean"], reference[:, 0].mean().item(), rel_tol=1e-12)
    with pytest.raises(ValueError) as exc:
        call_tool(
            "torchlens_payload_stats",
            {
                "path": str(clean),
                "labels": ["relu_1_2:1"],
                "reduction": {"retain_dim": 1},
                "max_rows": 4,
            },
        )
    assert exc.value.fields["code"] == "agent_reduction_invalid"


# Kept by the shrink-r4 coverage proof: sole branch coverage of agent/_stats.py.
def test_top_k_is_deterministic_with_full_coordinates(clean: Path) -> None:
    """top_k returns coordinate-addressed extremes, byte-stable across calls."""

    args = {"path": str(clean), "labels": ["relu_1_2:1"], "metrics": ["top_k"], "k": 3}
    first = call_tool("torchlens_payload_stats", args)
    second = call_tool("torchlens_payload_stats", args)
    assert canonical_dumps(first) == canonical_dumps(second)
    entries = first["data"]["rows"][0]["top_k"]
    log = tl.load(str(clean))
    payload = log["relu_1_2"].out
    top = entries[0]
    assert math.isclose(payload[tuple(top["coordinates"])].item(), top["value"], rel_tol=1e-6)


# Kept by the shrink-r4 coverage proof: a sole campaign mutation kill.
def test_compare_recovers_the_intervention_boundary(clean: Path, ablated: Path) -> None:
    """The flagship claim: boundary + downstream drift from two artifacts."""

    envelope = call_tool("torchlens_compare", {"reference": str(clean), "subject": str(ablated)})
    data = envelope["data"]
    assert data["direction"] == "subject_minus_reference"
    assert data["fingerprint_match"] is True
    assert data["match_basis"] == "site_key"
    by_label = {row["label"]: row for row in data["rows"] if "label" in row}
    assert by_label["relu_1_2:1"]["allclose"] is True  # upstream unchanged
    assert by_label["relu_2_4:1"]["allclose"] is False  # the intervention boundary
    assert by_label["relu_3_6:1"]["allclose"] is False  # downstream divergence
    coverage = data["coverage"]
    assert coverage["sites_value_compared"] == 3
    assert coverage["sites_changed"] == 2
    assert "never numerically compared" in coverage["note"]
