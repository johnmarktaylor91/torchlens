"""W051-AGENT: closure handoff on multi-pass traces (AUD-CODE 3.11d), per-row
reduction status (3.11f), JSON-safe logged values (2.6), and the 4.8 boundary
refusals, plus realistic-fixture paging (4.9)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from tests.test_agent_surface_helpers import save_clean_artifact
from tests.test_w051_agent_helpers import save_loop_artifact, save_realistic_artifact
from torchlens.agent import call_tool, canonical_dumps


@pytest.fixture(scope="module")
def clean(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The deterministic clean fixture artifact (module-scoped)."""

    return save_clean_artifact(tmp_path_factory.mktemp("w051_qs"))


@pytest.fixture(scope="module")
def loop(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The multi-pass Loop artifact (module-scoped)."""

    return save_loop_artifact(tmp_path_factory.mktemp("w051_loop"))


@pytest.mark.smoke
def test_closure_handoff_agrees_with_the_tool_on_multipass(loop: Path) -> None:
    """followed_by/preceded_by handoffs reproduce the served rows on a recurrent trace."""

    log = tl.load(str(loop))
    assert any(op.num_passes > 1 for op in log.layer_list)
    for query in [
        {"op": "followed_by", "item": {"op": "func", "value": "tanh"}},
        {"op": "preceded_by", "item": {"op": "func", "value": "tanh"}},
        {"op": "followed_by", "item": {"op": "pass_index", "value": 3}},
        {"op": "preceded_by", "item": {"op": "label", "value": "linear_1_1:2"}},
    ]:
        envelope = call_tool("torchlens_query_sites", {"path": str(loop), "query": query})
        served = [row["label"] for row in envelope["data"]["rows"]]
        assert served, query
        scope: dict = {"log": log}
        exec(envelope["data"]["handoff"], scope)  # noqa: S102 - the handoff contract IS verbatim execution
        assert scope["matched"] == served, (query, envelope["data"]["handoff"])


class _ScalarTail(nn.Module):
    """Returns a rank-2 activation AND a rank-0 reduction of it."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = torch.relu(self.fc(x))
        return y, y.sum()


@pytest.mark.smoke
def test_reduction_out_of_range_is_a_row_status_never_a_batch_failure(tmp_path: Path) -> None:
    """One rank-0 payload among rank-2 sites: its row says so; the batch succeeds."""

    generator = torch.Generator().manual_seed(7)
    log = tl.trace(
        _ScalarTail().eval(),
        torch.randn(2, 4, generator=generator),
        capture=tl.options.CaptureOptions(layers_to_save="all"),
    )
    path = tmp_path / "scalar.tlspec"
    tl.save(log, str(path))
    envelope = call_tool(
        "torchlens_payload_stats", {"path": str(path), "reduction": {"retain_dim": 0}}
    )
    assert envelope["status"] == "ok"
    by_status: dict[str, list[str]] = {}
    for row in envelope["data"]["rows"]:
        by_status.setdefault(row["status"], []).append(row["label"])
    assert "ok" in by_status and "reduction_unsupported" in by_status
    scalar_rows = [
        row for row in envelope["data"]["rows"] if row["status"] == "reduction_unsupported"
    ]
    assert all(row["shape"] == [] for row in scalar_rows)
    assert all("no dimension to retain" in row["metric_note"] for row in scalar_rows)
    assert all(row["reduction"]["rows"] == [] for row in scalar_rows)
    ok_rows = [row for row in envelope["data"]["rows"] if row["status"] == "ok"]
    assert all(row["reduction"]["extent"] == 2 for row in ok_rows)
    # A malformed reduction SPEC is still a request refusal (site-independent).
    with pytest.raises(ValueError) as exc:
        call_tool("torchlens_payload_stats", {"path": str(path), "reduction": {"retain_dim": "0"}})
    assert exc.value.fields["code"] == "agent_reduction_invalid"


@pytest.mark.smoke
def test_to_agent_json_tags_nonfinite_logged_values() -> None:
    """The documented JSON-serializable promise holds under allow_nan=False."""

    class _Logs(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            torch.manual_seed(0)
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            tl.report.log_value("loss_nan", float("nan"))
            tl.report.log_value("scale_inf", float("inf"))
            tl.report.log_value("neg_inf", float("-inf"))
            tl.report.log_value("plain", 1.5)
            return self.fc(x)

    log = tl.trace(_Logs().eval(), torch.randn(2, 4, generator=torch.Generator().manual_seed(3)))
    dump = log.to_agent_json()
    text = json.dumps(dump, allow_nan=False)
    assert json.loads(text)["logged_values"] == {
        "loss_nan": {"nonfinite": "nan"},
        "scale_inf": {"nonfinite": "inf"},
        "neg_inf": {"nonfinite": "-inf"},
        "plain": 1.5,
    }


@pytest.mark.smoke
def test_unmatched_labels_are_disclosed(clean: Path) -> None:
    """A label that names no site is a warning, never a silent empty ok."""

    envelope = call_tool(
        "torchlens_payload_stats", {"path": str(clean), "labels": ["relu_1_2:1", "nope_9_9"]}
    )
    assert [row["label"] for row in envelope["data"]["rows"]] == ["relu_1_2:1"]
    assert any("nope_9_9" in warning for warning in envelope["warnings"])
    matched = call_tool("torchlens_payload_stats", {"path": str(clean), "labels": ["relu_1_2"]})
    assert matched["warnings"] == []  # bare layer labels match too


@pytest.mark.smoke
def test_compare_refuses_negative_tolerances_typed(clean: Path) -> None:
    """rtol/atol are magnitudes: a negative one refuses typed at the boundary."""

    for bad in ({"rtol": -1.0}, {"atol": -0.5}, {"rtol": float("nan")}):
        with pytest.raises(ValueError) as exc:
            call_tool("torchlens_compare", {"reference": str(clean), "subject": str(clean), **bad})
        assert exc.value.fields["code"] == "agent_argument_invalid"
    assert (
        call_tool(
            "torchlens_compare", {"reference": str(clean), "subject": str(clean), "rtol": 0.0}
        )["status"]
        == "ok"
    )


@pytest.mark.heavy
def test_realistic_fixture_pages_and_stats_the_full_population(tmp_path: Path) -> None:
    """108 ops: listing pages reassemble exactly; stats serve every retained site (4.9)."""

    artifact = save_realistic_artifact(tmp_path, save_all=True)
    unpaged = call_tool("torchlens_query_sites", {"path": str(artifact)})
    total = unpaged["data"]["header"]["population_total"]
    assert total > 100 and unpaged["truncation"] is None
    rows: list[dict] = []
    continuation = None
    pages = 0
    while True:
        args: dict = {"path": str(artifact), "max_rows": 50}
        if continuation is not None:
            args["continuation"] = continuation
        page = call_tool("torchlens_query_sites", args)
        rows.extend(page["data"]["rows"])
        pages += 1
        continuation = page["data"]["next"]
        if continuation is None:
            break
    assert pages == 3
    assert canonical_dumps(rows) == canonical_dumps(unpaged["data"]["rows"])
    stats = call_tool(
        "torchlens_payload_stats", {"path": str(artifact), "metrics": ["mean", "nan_count"]}
    )
    assert stats["status"] == "ok"
    assert len(stats["data"]["rows"]) == total
    assert {row["status"] for row in stats["data"]["rows"]} == {"ok"}
    assert stats["data"]["header"]["bytes_materialized"] > 0
