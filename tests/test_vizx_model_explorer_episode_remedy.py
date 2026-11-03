"""Model Explorer's stepless-ledger refusal remedy works as written (R2-11).

A trace whose episode ledger has no stepped-module address refuses the
per-step export with the remedy "export with per_step=False"; following that
remedy must export the full exact graph, disclose the missing step drivers,
and keep every capture-honesty fact on the root.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl


def _stepless_episode_trace() -> tl.Trace:
    """A capture carrying an episode header without a stepped-module address."""

    torch.manual_seed(0)
    trace = tl.trace(nn.Sequential(nn.Linear(3, 4), nn.ReLU()), torch.randn(2, 3))
    trace.annotations["episode"] = {
        "header": {"capture_kind": "episode", "n_steps_declared": 2, "token_feed": "forced"},
        "rows": [],
    }
    return trace


def _root_attrs(graph: dict) -> dict[str, str]:
    """The graph's root disclosure row (``groupNodeAttributes[""]``)."""

    return dict(graph["groupNodeAttributes"][""])


def test_default_export_still_refuses_typed() -> None:
    """Without per_step=False the stepless ledger keeps its teaching refusal."""

    with pytest.raises(Exception) as excinfo:
        tl.export.to_model_explorer_dict(_stepless_episode_trace())
    assert excinfo.value.fields["code"] == "model_explorer_episode_ledger_unavailable"
    assert "per_step=False" in excinfo.value.fields["remedy"]


def test_following_the_remedy_exports_the_full_graph() -> None:
    """per_step=False exports one full graph and discloses the missing drivers."""

    payload = tl.export.to_model_explorer_dict(_stepless_episode_trace(), per_step=False)
    assert [graph["id"] for graph in payload["graphs"]] == ["00-episode"]
    attrs = _root_attrs(payload["graphs"][0])
    assert attrs["step_drivers"].startswith("unavailable")
    assert attrs["view"] == "episode (full exact execution)"
    assert "episode" in attrs
