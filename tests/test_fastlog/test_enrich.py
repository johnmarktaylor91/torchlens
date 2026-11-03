"""Tests for opt-in fastlog recording enrichments."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.recording.exceptions import RecordingConfigError


def _mlp() -> nn.Module:
    """Return a small two-layer MLP."""

    return nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 2))


def test_enrich_module_path_strings_populates_metadata() -> None:
    """Module path enrichment adds expected metadata fields."""

    recording = tl.record(_mlp(), torch.randn(1, 4), default_op=True)
    enriched = recording.enrich(["module_path_strings"])
    assert enriched is not recording
    assert enriched.records
    assert "module_path_strings" in enriched.records[0].metadata
    assert "module_path" in enriched.records[0].metadata


def test_enrich_all_feasible_matches_available_enrichments() -> None:
    """All-feasible applies all enrichments currently computable."""

    recording = tl.record(_mlp(), torch.randn(1, 4), default_op=True)
    by_name = recording.enrich(["module_path_strings"])
    by_preset = recording.enrich("all-feasible")
    assert [record.metadata for record in by_preset.records] == [
        record.metadata for record in by_name.records
    ]


def test_record_postprocess_accepts_only_the_live_enrich_steps() -> None:
    """``postprocess=`` names a live ``Recording.enrich`` step; a removed step refuses typed."""

    live = tl.record(
        _mlp(),
        torch.randn(1, 4),
        default_op=True,
        postprocess="module_path_strings",
    )
    assert "module_path" in live.records[0].metadata
    with pytest.raises(RecordingConfigError, match="module_path_strings") as info:
        tl.record(
            _mlp(),
            torch.randn(1, 4),
            default_op=True,
            postprocess="param_addresses",
        )
    assert "param_addresses" not in str(info.value).split("must be")[1]
