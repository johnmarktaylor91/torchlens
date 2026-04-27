"""Tests for opt-in fastlog recording enrichments."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.fastlog.exceptions import RecordingConfigError


def _mlp() -> nn.Module:
    """Return a small two-layer MLP."""

    return nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 2))


def test_enrich_module_path_strings_populates_metadata() -> None:
    """Module path enrichment adds expected metadata fields."""

    recording = tl.fastlog.record(_mlp(), torch.randn(1, 4), default_op=True)
    enriched = recording.enrich(["module_path_strings"])
    assert enriched is not recording
    assert enriched.records
    assert "module_path_strings" in enriched.records[0].metadata
    assert "module_path" in enriched.records[0].metadata


def test_enrich_param_addresses_requires_capture_time_field() -> None:
    """Param address enrichment raises clearly when capture data is absent."""

    recording = tl.fastlog.record(_mlp(), torch.randn(1, 4), default_op=True)
    with pytest.raises(RecordingConfigError, match="parent_param_addresses"):
        recording.enrich(["param_addresses"])


def test_enrich_all_feasible_matches_available_enrichments() -> None:
    """All-feasible applies all enrichments currently computable."""

    recording = tl.fastlog.record(_mlp(), torch.randn(1, 4), default_op=True)
    by_name = recording.enrich(["module_path_strings"])
    by_preset = recording.enrich("all-feasible")
    assert [record.metadata for record in by_preset.records] == [
        record.metadata for record in by_name.records
    ]
