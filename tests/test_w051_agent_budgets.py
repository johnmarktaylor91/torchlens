"""W051-AGENT: budget/paging contract (AUD-CODE 2.7a/b) and the explain
default budget (AUD-CODE 4.8), exercised on the realistic fixture (4.9)."""

from __future__ import annotations

from pathlib import Path

import pytest

import torchlens as tl
from tests.test_agent_surface_helpers import save_clean_artifact
from tests.test_w051_agent_helpers import save_realistic_artifact
from torchlens.agent import _budgets, call_tool, canonical_dumps, spec_for


@pytest.fixture(scope="module")
def clean(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The deterministic clean fixture artifact (module-scoped)."""

    return save_clean_artifact(tmp_path_factory.mktemp("w051_budgets"))


@pytest.fixture(scope="module")
def realistic(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The ~108-op MiniTransformer artifact (module-scoped)."""

    return save_realistic_artifact(tmp_path_factory.mktemp("w051_budgets_real"))


def _page_all(tool: str, args: dict, row_key: str) -> list[dict]:
    """Follow ``data.next`` until exhausted, collecting rows."""

    rows: list[dict] = []
    continuation = None
    for _ in range(200):
        request = dict(args)
        if continuation is not None:
            request["continuation"] = continuation
        page = call_tool(tool, request)
        assert page["status"] == "ok", page["status"]
        rows.extend(page["data"][row_key])
        if "max_tokens" in args:
            assert _budgets.estimate_tokens(canonical_dumps(page)) <= args["max_tokens"]
        continuation = page["data"]["next"]
        if continuation is None:
            return rows
    raise AssertionError("paging did not terminate")


@pytest.mark.smoke
def test_token_backstop_reminted_next_reaches_every_row(clean: Path) -> None:
    """Rows the token backstop drops are reachable: next starts at the first dropped row."""

    unpaged = call_tool("torchlens_query_sites", {"path": str(clean)})
    first = call_tool("torchlens_query_sites", {"path": str(clean), "max_tokens": 900})
    assert first["truncation"]["omitted"] > 0
    assert first["data"]["next"] is not None
    assert (
        first["data"]["next"]["offset"]
        == len(first["data"]["rows"])
        == first["truncation"]["included"]
    )
    assert "data.next" in first["truncation"]["how_to_get_more"]
    paged = _page_all("torchlens_query_sites", {"path": str(clean), "max_tokens": 900}, "rows")
    assert canonical_dumps(paged) == canonical_dumps(unpaged["data"]["rows"])


@pytest.mark.heavy
def test_dump_full_pages_and_accepts_max_tokens(realistic: Path) -> None:
    """view=full pages its op rows like graph; every page keeps the honesty blocks (2.7a)."""

    log = tl.load(str(realistic))
    reference_ops = log.to_agent_json()["ops"]
    assert len(reference_ops) > 100
    first = call_tool("torchlens_dump", {"path": str(realistic), "view": "full", "max_rows": 40})
    assert first["status"] == "ok"
    assert first["data"]["rows_total"] == len(reference_ops)
    assert len(first["data"]["ops"]) == 40 and first["data"]["offset"] == 0
    assert first["data"]["next"]["offset"] == 40
    assert first["data"]["truncation"]["ops_omitted"] == len(reference_ops) - 40
    assert first["data"]["capture"]["capture_status"] == "complete"
    assert first["data"]["counts"]["operations"] == len(reference_ops)
    paged = _page_all(
        "torchlens_dump", {"path": str(realistic), "view": "full", "max_rows": 40}, "ops"
    )
    assert canonical_dumps(paged) == canonical_dumps(reference_ops)
    # The 2.7a remedy is now POSSIBLE: dump takes max_tokens (lowering only).
    budgeted = call_tool(
        "torchlens_dump", {"path": str(realistic), "view": "full", "max_tokens": 6_000}
    )
    assert budgeted["status"] == "ok" and budgeted["limits"]["max_tokens"] == 6_000
    assert budgeted["truncation"]["omitted"] > 0 and budgeted["data"]["next"] is not None
    tight = _page_all(
        "torchlens_dump", {"path": str(realistic), "view": "full", "max_tokens": 6_000}, "ops"
    )
    assert canonical_dumps(tight) == canonical_dumps(reference_ops)
    with pytest.raises(ValueError) as exc:
        call_tool(
            "torchlens_dump",
            {"path": str(realistic), "max_tokens": _budgets.ROW_TOOL_MAX_TOKENS + 1},
        )
    assert exc.value.fields["code"] == "agent_limit_invalid"


@pytest.mark.smoke
def test_budget_floor_keeps_the_capture_block(clean: Path) -> None:
    """The floor carries the honesty facts it claims to carry."""

    floor = call_tool("torchlens_overview", {"path": str(clean), "max_tokens": 1})
    assert floor["status"] == "budget_floor_exceeded"
    assert floor["data"]["budget_floor_exceeded"] is True
    assert floor["data"]["capture"]["capture_status"] == "complete"
    assert "max_tokens" in floor["truncation"]["how_to_get_more"]
    rows_floor = call_tool("torchlens_query_sites", {"path": str(clean), "max_tokens": 1})
    assert rows_floor["data"] == {"budget_floor_exceeded": True}  # no capture block to carry


@pytest.mark.smoke
def test_explain_defaults_to_the_declared_budget(clean: Path) -> None:
    """explain's served default IS its declared default; honesty blocks ride the record."""

    declared = spec_for("torchlens_explain").default_limits["max_tokens"]
    assert declared == _budgets.ORIENTATION_MAX_TOKENS
    envelope = call_tool("torchlens_explain", {"path": str(clean)})
    assert envelope["limits"]["max_tokens"] == declared
    assert envelope["data"]["capture"]["capture_status"] == "complete"
    assert envelope["data"]["audit"]["nonfinite"]["n_labels"] == 0
    tight = call_tool("torchlens_explain", {"path": str(clean), "max_tokens": 120})
    assert tight["limits"]["max_tokens"] == 120
    assert "Capture status" in tight["data"]["report"]
    assert "Truncation" in tight["data"]["report"]


@pytest.mark.smoke
def test_backstop_disclosure_is_honest_on_non_paging_tools(clean: Path) -> None:
    """payload_stats does not page; its trimmed disclosure must not promise a continuation."""

    full = call_tool("torchlens_payload_stats", {"path": str(clean)})
    budget = _budgets.estimate_tokens(canonical_dumps(full)) * 7 // 10
    trimmed = call_tool("torchlens_payload_stats", {"path": str(clean), "max_tokens": budget})
    if trimmed["status"] == "ok":
        assert "does not page" in trimmed["truncation"]["how_to_get_more"]
        assert "next" not in trimmed["data"]
    else:
        assert trimmed["status"] == "budget_floor_exceeded"
