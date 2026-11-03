"""Tests for opt-in fastlog recording enrichments."""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


def _mlp() -> nn.Module:
    """Return a small two-layer MLP."""

    return nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 2))


def test_enrich_module_path_strings_populates_metadata() -> None:
    """Module path enrichment adds expected metadata fields."""

    recording = tl.record(
        _mlp(), torch.randn(1, 4), recording=tl.options.RecordingOptions(default_op=True)
    )
    enriched = recording.enrich(["module_path_strings"])
    assert enriched is not recording
    assert enriched.records
    assert "module_path_strings" in enriched.records[0].metadata
    assert "module_path" in enriched.records[0].metadata


def test_enrich_all_feasible_matches_available_enrichments() -> None:
    """All-feasible applies all enrichments currently computable."""

    recording = tl.record(
        _mlp(), torch.randn(1, 4), recording=tl.options.RecordingOptions(default_op=True)
    )
    by_name = recording.enrich(["module_path_strings"])
    by_preset = recording.enrich("all-feasible")
    assert [record.metadata for record in by_preset.records] == [
        record.metadata for record in by_name.records
    ]
