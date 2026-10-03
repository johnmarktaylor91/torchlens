"""R10-3: the op_log_fields invariant on LOADED torch artifacts.

``Op.func`` is ``FieldPolicy.DROP`` and ``__getstate__`` hard-nulls it, so
``callable(lpl.func)`` false-fired on EVERY loaded torch trace — the torch-only
``op_log_fields`` contract had zero loaded-torch coverage (the only loaded
callers were non-torch backends, which filter the contract out). The fix
splits the arm on ``_loaded_from_bundle``: the loaded arm asserts the known
load-time shape POSITIVELY (``func is None``; a non-None func on a loaded
artifact is tampering), while ``func_name`` enforcement and the live arm are
unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import MetadataInvariantError
from torchlens.validation.invariants import check_metadata_invariants


def _loaded_round_trip(tmp_path: Path) -> tl.Trace:
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
    trace = tl.trace(model, torch.randn(2, 4))
    bundle = tmp_path / "roundtrip"
    tl.save(trace, bundle, overwrite=True)
    trace.cleanup()
    return tl.load(bundle)


def _functionful_layer(trace: tl.Trace):
    """Return a layer the func block actually validates (linear/relu)."""

    for layer in trace.layer_list:
        if layer.func_name in {"linear", "relu"}:
            return layer
    raise AssertionError("fixture trace lost its linear/relu layers")


def test_loaded_torch_trace_passes_metadata_invariants(tmp_path: Path) -> None:
    """Fail-before: every loaded torch trace raised 'func is not callable'."""

    loaded = _loaded_round_trip(tmp_path)
    check_metadata_invariants(loaded)


def test_loaded_trace_with_blanked_func_name_still_fails(tmp_path: Path) -> None:
    """The loaded arm keeps the func_name half of the tripwire armed."""

    loaded = _loaded_round_trip(tmp_path)
    _functionful_layer(loaded).ops[0].func_name = ""
    with pytest.raises(MetadataInvariantError, match="func_name is empty"):
        check_metadata_invariants(loaded)


def test_loaded_trace_with_none_sentinel_func_name_still_fails(tmp_path: Path) -> None:
    """The 'none' sentinel stays name corruption on loaded artifacts too."""

    loaded = _loaded_round_trip(tmp_path)
    _functionful_layer(loaded).ops[0].func_name = "none"
    with pytest.raises(MetadataInvariantError):
        check_metadata_invariants(loaded)


def test_loaded_trace_with_nonnull_func_is_tampering(tmp_path: Path) -> None:
    """func never persists, so a non-None func on a loaded artifact fails."""

    loaded = _loaded_round_trip(tmp_path)
    _functionful_layer(loaded).ops[0].func = torch.relu
    with pytest.raises(MetadataInvariantError, match="non-None func"):
        check_metadata_invariants(loaded)


def test_live_trace_with_nulled_func_still_fails() -> None:
    """The live arm is untouched: a live op losing its callable still fails."""

    trace = tl.trace(nn.Sequential(nn.Linear(4, 3), nn.ReLU()), torch.randn(2, 4))
    try:
        _functionful_layer(trace).ops[0].func = None
        with pytest.raises(MetadataInvariantError, match="func is not callable"):
            check_metadata_invariants(trace)
    finally:
        trace.cleanup()
