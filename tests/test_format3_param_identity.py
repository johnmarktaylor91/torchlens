"""Copied parameter edges retain their capture identity without merging unknown parameters."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io.format_errors import ArtifactError


def _trace_with_param_copy():
    """Reproduce the copied Param edge retained by a capture operation."""

    trace = tl.trace(nn.Linear(3, 2), torch.ones(1, 3))
    op = next(op for op in trace.ops if op._param_logs)
    original = op._param_logs[0]
    alias = copy.copy(original)
    op._param_logs = [alias, *op._param_logs[1:]]
    return trace, original, alias


def test_copied_capture_param_edge_uses_one_authoritative_row(tmp_path: Path) -> None:
    """One capture barcode stays one parameter, and saving leaves live barcodes untouched."""

    trace, original, alias = _trace_with_param_copy()
    barcode = original.barcode
    path = tmp_path / "aliased.tlspec"
    trace.save(path)
    table = json.loads((path / "records/param.json").read_text())
    assert table["rows"] == len(trace.params)
    loaded = tl.load(path)
    assert len(loaded.params) == len(trace.params)
    assert original.barcode == alias.barcode == barcode


def test_unknown_param_barcode_with_repeated_address_still_refuses(tmp_path: Path) -> None:
    """Only a known capture identity is an alias; an ambiguous address never last-wins."""

    trace, _, alias = _trace_with_param_copy()
    alias.barcode = "another_parameter"
    with pytest.raises(ArtifactError) as info:
        trace.save(tmp_path / "ambiguous.tlspec")
    assert info.value.fields["code"] == "record_schema_invalid"
