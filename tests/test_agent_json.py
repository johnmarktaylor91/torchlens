"""Tests for the agent-facing machine-readable dump ``Trace.to_agent_json``."""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

import torchlens as tl


class TinyRecurrentModel(nn.Module):
    """Model that reuses one module twice to exercise pass-qualified labels."""

    def __init__(self) -> None:
        """Initialize the shared projection."""

        super().__init__()
        self.proj = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the shared projection twice.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Twice-projected activations.
        """

        return self.proj(torch.relu(self.proj(x)))


def _captured_log() -> tl.Trace:
    """Return a small deterministic captured trace.

    Returns
    -------
    tl.Trace
        Capture of a three-op sequential model with one saved activation.
    """

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
    return tl.trace(model, torch.randn(2, 4), save=tl.func("relu"))


def test_to_agent_json_is_json_serializable_and_self_describing() -> None:
    """The dump round-trips through json and embeds its navigation guide."""

    log = _captured_log()
    dump = log.to_agent_json()

    assert dump["schema"] == "torchlens.agent_trace.v1"
    restored = json.loads(json.dumps(dump))
    assert restored == dump
    guide = dump["guide"]
    assert "purpose" in guide and "navigation" in guide and "next_steps" in guide
    # The guide points back at the live human surface, never a parallel API.
    assert "trace[<layer_label>].out" in guide["payloads"]
    assert guide["next_steps"]["plain_language_report"] == "tl.report.explain(trace)"


def test_to_agent_json_carries_capture_honesty_facts() -> None:
    """Outcome/verification facts ride the dump exactly like explain()."""

    log = _captured_log()
    dump = log.to_agent_json()
    capture = dump["capture"]

    assert capture["capture_status"] == "complete"
    assert capture["capture_verified"] is None
    assert capture["backend"] == "torch"
    assert capture["structure_only"] is False

    log.capture_verified = False
    log.capture_verification_reason = "dynamo_region_not_logged"
    ceilinged = log.to_agent_json()["capture"]
    assert ceilinged["capture_verified"] is False
    assert ceilinged["capture_verification_reason"] == "dynamo_region_not_logged"


def test_to_agent_json_graph_matches_the_live_trace() -> None:
    """Counts, labels, edges, and saved flags mirror the live object."""

    log = _captured_log()
    dump = log.to_agent_json()

    # F09 numbers core: "operations" counts every tracked tensor row; the
    # identity-partition split names compute rows separately.
    assert dump["counts"]["operations"] == len(log.layer_labels)
    assert dump["counts"]["compute_ops"] == log.num_ops
    assert dump["counts"]["tensors_saved"] == log.num_saved_ops
    assert dump["layer_labels"] == list(log.layer_labels)
    assert dump["truncation"] is None

    by_layer = {entry["layer_label"]: entry for entry in dump["ops"]}
    assert set(by_layer) == set(log.layer_labels)
    relu = by_layer["relu_1_2"]
    assert relu["saved"] is True
    assert relu["parents"] == ["linear_1_1"]
    assert relu["children"] == ["linear_2_3"]
    assert relu["dtype"] == "torch.float32"
    assert relu["shape"] == [2, 4]
    # Every layer_label in the dump addresses the live trace directly.
    assert log[relu["layer_label"]].out.shape == (2, 4)

    addresses = {row["address"] for row in dump["modules"]}
    assert "self" in addresses and "0" in addresses


def test_to_agent_json_pass_qualifies_multipass_ops() -> None:
    """Reused-module captures emit one row per pass with distinct labels."""

    log = tl.trace(TinyRecurrentModel().eval(), torch.randn(1, 3))
    dump = log.to_agent_json()

    linear_rows = [row for row in dump["ops"] if row["func_name"] == "linear"]
    assert len(linear_rows) == 2
    assert {row["pass_index"] for row in linear_rows} == {1, 2}
    assert all(row["num_passes"] == 2 for row in linear_rows)
    labels = {row["label"] for row in linear_rows}
    assert len(labels) == 2
    assert all(":" in label for label in labels)
    # Same structural site across the two passes of the shared module.
    site_keys = {row["site_key"] for row in linear_rows}
    assert len(site_keys) == 1


def test_to_agent_json_max_ops_truncation_is_disclosed_never_silent() -> None:
    """Capping op rows discloses exactly what was omitted."""

    log = _captured_log()
    dump = log.to_agent_json(max_ops=2)

    assert len(dump["ops"]) == 2
    truncation = dump["truncation"]
    assert truncation is not None
    assert truncation["ops_included"] == 2
    assert truncation["ops_omitted"] == 3
    # Full-capture counts stay the truth even when rows are dropped.
    assert dump["counts"]["operations"] == len(log.layer_labels)
    assert dump["counts"]["compute_ops"] == log.num_ops

    with pytest.raises(ValueError, match="positive integer"):
        log.to_agent_json(max_ops=0)
    with pytest.raises(ValueError, match="positive integer"):
        log.to_agent_json(max_ops=True)
