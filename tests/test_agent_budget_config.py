"""The agent surface's budget configuration and its corpus measurement.

The numbers in ``torchlens/agent/_budgets.py`` are configuration set from the classics-corpus
measurement (``tools/agent_evals/measure_budgets.py``). These tests pin how each number is
served, that a trimmed overview names a next call it accepts, and that the measurement script
runs on one corpus entry.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.agent import _budgets, call_tool, canonical_dumps, spec_for

REPO_ROOT = Path(__file__).resolve().parents[1]


def _saved_trace(tmp_path: Path) -> Path:
    layers: list[nn.Module] = []
    for _ in range(6):
        layers += [nn.Linear(4, 4), nn.ReLU(), nn.Tanh()]
    model = nn.Sequential(*layers).eval()
    path = tmp_path / "chain.tlspec"
    tl.trace(model, torch.randn(2, 4), save="all").save(str(path))
    return path


def test_each_tool_serves_its_configured_defaults() -> None:
    assert spec_for("torchlens_overview").default_limits == {
        "max_tokens": _budgets.OVERVIEW_MAX_TOKENS
    }
    assert spec_for("torchlens_explain").default_limits == {
        "max_tokens": _budgets.ORIENTATION_MAX_TOKENS
    }
    for tool in ("torchlens_dump", "torchlens_query_sites", "torchlens_payload_stats"):
        limits = spec_for(tool).default_limits
        assert limits["max_rows"] == _budgets.DEFAULT_MAX_ROWS
        assert limits["max_tokens"] == _budgets.ROW_TOOL_MAX_TOKENS
        assert spec_for(tool).input_schema["properties"]["max_rows"]["maximum"] == (
            _budgets.DEFAULT_MAX_ROWS
        )


def test_overview_ceiling_is_its_own_number(tmp_path: Path) -> None:
    """The overview accepts up to its own ceiling and refuses one token more."""

    path = _saved_trace(tmp_path)
    served = call_tool(
        "torchlens_overview", {"path": str(path), "max_tokens": _budgets.OVERVIEW_MAX_TOKENS}
    )
    assert served["limits"]["max_tokens"] == _budgets.OVERVIEW_MAX_TOKENS
    with pytest.raises(tl.errors.TorchLensError) as refusal:
        call_tool(
            "torchlens_overview",
            {"path": str(path), "max_tokens": _budgets.OVERVIEW_MAX_TOKENS + 1},
        )
    assert refusal.value.fields["code"] == "agent_limit_invalid"


def test_trimmed_overview_discloses_and_names_a_call_it_accepts(tmp_path: Path) -> None:
    path = _saved_trace(tmp_path)
    whole = call_tool("torchlens_overview", {"path": str(path)})
    assert whole["truncation"] is None
    just_under = _budgets.estimate_tokens(canonical_dumps(whole)) - 40
    served = call_tool("torchlens_overview", {"path": str(path), "max_tokens": just_under})
    assert served["status"] != "budget_floor_exceeded"
    truncation = served["truncation"]
    assert truncation is not None and truncation["omitted"] != 0
    hint = truncation["how_to_get_more"]
    assert "torchlens_dump view='graph'" in hint
    assert "labels=" not in hint  # the overview takes no labels= or query=


def test_measurement_script_measures_one_corpus_entry(tmp_path: Path) -> None:
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    from agent_evals import measure_budgets
    from classics_corpus._loader import corpus_entries

    entry = next(e for e in corpus_entries("smoke") if e.id.startswith("Neural Gas"))
    record = measure_budgets.measure_entry(entry, tmp_path)
    assert "error" not in record
    assert record["dump_full_rows_total"] >= record["num_ops"]
    assert record["overview_folded_tokens"] > 0
    summary = measure_budgets.summarize([record])
    assert {row["constant"] for row in summary} >= {
        "ORIENTATION_MAX_TOKENS",
        "OVERVIEW_MAX_TOKENS",
        "ROW_TOOL_MAX_TOKENS",
        "DEFAULT_MAX_ROWS",
        "MAX_BLOB_BYTES",
        "MAX_CALL_BYTES",
    }
