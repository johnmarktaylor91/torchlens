"""EPISODE export tests (memo D11-D14, B7) on a real tiny episode capture.

The step-membership join runs through the persisted module-call ops list
with the stack cross-check asserted equal; drivers window-assign exactly
once; boundary proxies mint reserved-prefix step-stable ids; the budget
binds per-step graphs with disclosure; the privacy profile owns tokens.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.export._model_explorer._ids import PROXY_ID_PREFIX


class _TinyLM(nn.Module):
    """Minimal stepped module."""

    def __init__(self) -> None:
        """Embedding + linear + head."""

        super().__init__()
        self.emb = nn.Embedding(16, 8)
        self.mix = nn.Linear(8, 8)
        self.head = nn.Linear(8, 16)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """One decode step."""

        hidden = self.emb(ids).mean(dim=1)
        hidden = torch.tanh(self.mix(hidden))
        return self.head(hidden)


class _Root(nn.Module):
    """Greedy three-step generation wrapper."""

    def __init__(self, model: nn.Module) -> None:
        """Hold the stepped module."""

        super().__init__()
        self.model = model

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Emit three greedy tokens."""

        tokens = []
        current = ids
        for _ in range(3):
            logits = self.model(current)
            next_token = logits.argmax(dim=-1, keepdim=True)
            tokens.append(next_token)
            current = torch.cat([current, next_token], dim=1)
        return torch.cat(tokens, dim=1)


@pytest.fixture(scope="module")
def episode_log() -> Any:
    """Capture one real three-step episode per module."""

    torch.manual_seed(0)
    stepped = _TinyLM().eval()
    root = _Root(stepped).eval()
    log = tl.trace(
        root,
        torch.randint(0, 16, (1, 2)),
        episode=tl.options.EpisodeOptions(stepped_module=stepped, num_steps=3),
    )
    try:
        yield log
    finally:
        log.cleanup()


@pytest.fixture(scope="module")
def episode_payload(episode_log: Any) -> dict[str, Any]:
    """Export the episode once per module."""

    return tl.export.to_model_explorer_dict(episode_log)


def test_full_graph_first_then_zero_padded_steps(episode_payload: dict[str, Any]) -> None:
    """ONE collection: full exact episode graph, then per-step graphs (D11)."""

    ids = [graph["id"] for graph in episode_payload["graphs"]]
    assert ids[0] == "00-episode"
    assert len(ids) == 4
    assert ids == sorted(ids), "physical order must match name_asc"
    report = tl.export.validate_model_explorer_payload(episode_payload)
    assert report.ok, report.failures


def test_boundary_proxies_false_drops_and_discloses(episode_log: Any) -> None:
    """boundary_proxies=False gives drop-plus-disclose (D12)."""

    payload = tl.export.to_model_explorer_dict(episode_log, boundary_proxies=False)
    for graph in payload["graphs"][1:]:
        assert not any(node["id"].startswith(PROXY_ID_PREFIX) for node in graph["nodes"])
    disclosures = [
        graph["groupNodeAttributes"][""].get("skipped_parents") for graph in payload["graphs"][2:]
    ]
    assert any(disclosures), "dropped cross-step edges must be counted"


def test_budget_omits_step_graphs_with_disclosure(episode_log: Any) -> None:
    """The bytes/count budget binds per-step graphs; omissions are listed."""

    payload = tl.export.to_model_explorer_dict(episode_log, max_step_graphs=1)
    ids = [graph["id"] for graph in payload["graphs"]]
    assert ids[0] == "00-episode"
    assert len(ids) == 2
    disclosure = payload["graphs"][0]["groupNodeAttributes"][""]["omitted_step_graphs"]
    assert "max_step_graphs" in disclosure or "budget" in disclosure


def test_step_output_rides_local_rows_and_public_drops_it(episode_log: Any) -> None:
    """Step output lands on local step rows and never on public exports (D14).

    Grammar v2 (C07X): the fact key is the generic ``step_output`` and the
    deleted arithmetic ``cache_len`` never appears -- no export fact may
    imply an unmeasured cache length.
    """

    local = tl.export.to_model_explorer_dict(episode_log)
    public = tl.export.to_model_explorer_dict(episode_log, privacy_profile="public")
    local_rows = [graph["groupNodeAttributes"][""] for graph in local["graphs"][1:]]
    public_rows = [graph["groupNodeAttributes"][""] for graph in public["graphs"][1:]]
    assert any("step_output" in row for row in local_rows)
    assert not any("step_output" in row for row in public_rows)
    for row in local_rows:
        assert row["status"] == "complete"
        assert "role" in row
        assert "cache_len" not in row and "tokens" not in row


def test_per_step_false_emits_full_graph_only(episode_log: Any) -> None:
    """per_step=False keeps the lossless authority alone."""

    payload = tl.export.to_model_explorer_dict(episode_log, per_step=False)
    assert [graph["id"] for graph in payload["graphs"]] == ["00-episode"]


def test_ledger_missing_stepped_address_refuses_typed(episode_log: Any) -> None:
    """A ledger without its stepped-module address refuses teaching (D11)."""

    ledger = episode_log.annotations["episode"]
    header = dict(ledger.get("header") or {})
    header.pop("stepped_module", None)
    episode_log.annotations["episode"] = {**ledger, "header": header}
    try:
        with pytest.raises(Exception) as excinfo:
            tl.export.to_model_explorer_dict(episode_log)
        assert excinfo.value.fields["code"] == "model_explorer_episode_ledger_unavailable"
        assert "per_step=False" in excinfo.value.fields["remedy"]
    finally:
        episode_log.annotations["episode"] = ledger


def test_membership_join_disagreement_refuses_typed(episode_log: Any) -> None:
    """The two step-membership spellings disagreeing is a typed failure (D11).

    Tampering the module-call ops list breaks its mandatory equality with the
    ``module_call_stack[0]`` cross-check; exporting either side silently would
    misattribute ops to steps.
    """

    ledger = episode_log.annotations["episode"]
    address = ledger["header"]["stepped_module"]
    call_label = f"{address}:{ledger['rows'][0]['coord']['member_call_index']}"
    module_call = episode_log.module_calls[call_label]
    original_ops = list(module_call.ops)
    module_call.ops = [*original_ops, "bogus_1_1:1"]
    try:
        with pytest.raises(Exception) as excinfo:
            tl.export.to_model_explorer_dict(episode_log)
        assert excinfo.value.fields["code"] == "model_explorer_episode_join_mismatch"
    finally:
        module_call.ops = original_ops
    healthy = tl.export.to_model_explorer_dict(episode_log)
    assert len(healthy["graphs"]) == 4, "the restored episode must export cleanly again"
