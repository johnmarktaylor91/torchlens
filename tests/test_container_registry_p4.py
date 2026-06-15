"""P4 tests for container persistence, rename, and capability honesty."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.ir.container_registry import ContainerRecord, Role


class NestedRoundTripModel(nn.Module):
    """Return a nested tensor-bearing output and consume a nested input."""

    def forward(self, payload: dict[str, list[torch.Tensor]]) -> dict[str, object]:
        """Run a small deterministic nested-container forward pass."""

        x = payload["items"][0]
        y = payload["items"][1]
        return {"sum": x + y, "pair": (x.relu(), y + 2)}


def _payload() -> dict[str, list[torch.Tensor]]:
    """Return a deterministic nested input payload."""

    return {"items": [torch.tensor([1.0]), torch.tensor([2.0])]}


def _semantic_digest(trace: tl.Trace) -> str:
    """Return a stable semantic digest for flag-off byte-neutral comparisons."""

    projection = {
        "graph_shape_hash": trace.graph_shape_hash,
        "has_containers": "_containers" in trace.__dict__,
        "output_layers": list(trace.output_layers),
        "ops": [
            {
                "label": op.layer_label,
                "func_name": op.func_name,
                "parents": list(op.parents),
                "children": list(op.children),
                "container_path": repr(tuple(getattr(op, "container_path", ()) or ())),
                "container_spec": repr(getattr(op, "container_spec", None)),
            }
            for op in trace.layer_list
        ],
    }
    encoded = json.dumps(projection, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_no_portable_container_payloads(value: Any) -> None:
    """Assert portable container metadata contains no runtime payloads."""

    assert not isinstance(value, torch.Tensor)
    if isinstance(value, (Enum, type)):
        return
    assert not callable(value)
    if isinstance(value, dict):
        assert "id_to_entry" not in value
        for key, item in value.items():
            _assert_no_portable_container_payloads(key)
            _assert_no_portable_container_payloads(item)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _assert_no_portable_container_payloads(item)
    elif hasattr(value, "__dict__"):
        assert "id_to_entry" not in value.__dict__
        for item in value.__dict__.values():
            _assert_no_portable_container_payloads(item)
    elif hasattr(value, "__slots__"):
        for slot in value.__slots__:
            if hasattr(value, slot):
                _assert_no_portable_container_payloads(getattr(value, slot))


def _assert_structural_roundtrip(trace: tl.Trace) -> None:
    """Assert captured structures reconstruct and records are portable."""

    assert "_containers" in trace.__dict__
    assert any(Role.MODEL_INPUT in record.roles for record in trace._containers.values())
    assert any(Role.MODEL_OUTPUT in record.roles for record in trace._containers.values())
    assert trace.input_structure.kind == "dict"
    rebuilt_input = trace.reconstruct_container(role=Role.MODEL_INPUT)
    rebuilt_output = trace.reconstruct_output()

    assert torch.equal(rebuilt_input["items"][0], torch.tensor([1.0]))
    assert torch.equal(rebuilt_output["sum"], torch.tensor([3.0]))
    assert torch.equal(rebuilt_output["pair"][1], torch.tensor([4.0]))
    _assert_no_portable_container_payloads(trace._containers)
    for record in trace._containers.values():
        assert isinstance(record, ContainerRecord)
        for snapshot in record.snapshots:
            for occurrence in snapshot.leaf_occurrences:
                assert not isinstance(occurrence.tensor_identity, int)


@pytest.mark.parametrize("save_path", ["tlspec", "cache", "streaming"])
def test_flag_off_container_persistence_is_semantically_neutral(
    tmp_path: Path,
    save_path: str,
) -> None:
    """Flag-off traces stay container-free through every persistence path."""

    baseline = tl.trace(NestedRoundTripModel(), _payload(), random_seed=0)
    explicit_false = tl.trace(
        NestedRoundTripModel(),
        _payload(),
        random_seed=0,
        capture_container_structure=False,
    )

    if save_path == "tlspec":
        path = tmp_path / "off.tlspec"
        explicit_false.save(path)
        round_tripped = tl.load(path)
    elif save_path == "cache":
        first = tl.trace(
            NestedRoundTripModel(),
            _payload(),
            random_seed=0,
            cache=True,
            cache_dir=tmp_path / "cache",
            capture_container_structure=False,
        )
        round_tripped = tl.trace(
            NestedRoundTripModel(),
            _payload(),
            random_seed=0,
            cache=True,
            cache_dir=tmp_path / "cache",
            capture_container_structure=False,
        )
        assert first.capture_cache_key == round_tripped.capture_cache_key
        assert round_tripped.capture_cache_hit is True
    else:
        path = tmp_path / "streamed-off.tlspec"
        tl.trace(
            NestedRoundTripModel(),
            _payload(),
            random_seed=0,
            storage=tl.to_disk(path),
            capture_container_structure=False,
        )
        round_tripped = tl.load(path)

    assert "_containers" not in baseline.__dict__
    assert "_containers" not in explicit_false.__dict__
    assert "_containers" not in round_tripped.__dict__
    assert baseline.graph_shape_hash == explicit_false.graph_shape_hash
    assert _semantic_digest(baseline) == _semantic_digest(explicit_false)
    assert _semantic_digest(explicit_false) == _semantic_digest(round_tripped)


@pytest.mark.parametrize("save_path", ["tlspec", "cache", "streaming"])
def test_flag_on_container_structure_round_trips(
    tmp_path: Path,
    save_path: str,
) -> None:
    """Flag-on container records structurally round-trip through save paths."""

    if save_path == "tlspec":
        trace = tl.trace(
            NestedRoundTripModel(),
            _payload(),
            random_seed=0,
            capture_container_structure=True,
        )
        path = tmp_path / "on.tlspec"
        trace.save(path)
        round_tripped = tl.load(path)
    elif save_path == "cache":
        trace = tl.trace(
            NestedRoundTripModel(),
            _payload(),
            random_seed=0,
            cache=True,
            cache_dir=tmp_path / "cache",
            capture_container_structure=True,
        )
        round_tripped = tl.trace(
            NestedRoundTripModel(),
            _payload(),
            random_seed=0,
            cache=True,
            cache_dir=tmp_path / "cache",
            capture_container_structure=True,
        )
        assert round_tripped.capture_cache_hit is True
        assert trace.capture_cache_key == round_tripped.capture_cache_key
    else:
        path = tmp_path / "streamed-on.tlspec"
        streamed_trace = tl.trace(
            NestedRoundTripModel(),
            _payload(),
            random_seed=0,
            storage=tl.to_disk(path),
            capture_container_structure=True,
        )
        assert "_containers" in streamed_trace.__dict__
        round_tripped = tl.load(path)

    _assert_structural_roundtrip(round_tripped)


def test_capture_container_structure_rename_and_cache_key(tmp_path: Path) -> None:
    """Canonical and deprecated flags resolve alike, but cache on/off splits."""

    canonical = tl.trace(
        NestedRoundTripModel(),
        _payload(),
        random_seed=0,
        capture_container_structure=True,
    )
    with pytest.warns(DeprecationWarning, match="capture_output_structure"):
        alias = tl.trace(
            NestedRoundTripModel(),
            _payload(),
            random_seed=0,
            capture_output_structure=True,
        )

    _assert_structural_roundtrip(canonical)
    _assert_structural_roundtrip(alias)
    assert _semantic_digest(canonical) == _semantic_digest(alias)

    off = tl.trace(
        NestedRoundTripModel(),
        _payload(),
        random_seed=0,
        cache=True,
        cache_dir=tmp_path / "cache",
        capture_container_structure=False,
    )
    on = tl.trace(
        NestedRoundTripModel(),
        _payload(),
        random_seed=0,
        cache=True,
        cache_dir=tmp_path / "cache",
        capture_container_structure=True,
    )
    assert off.capture_cache_key != on.capture_cache_key


def test_after_save_tensor_fill_is_best_effort(tmp_path: Path) -> None:
    """Structure persists even when the in-memory streamed trace drops payloads."""

    path = tmp_path / "streamed-best-effort.tlspec"
    streamed_trace = tl.trace(
        NestedRoundTripModel(),
        _payload(),
        random_seed=0,
        storage=tl.to_disk(path),
        capture_container_structure=True,
    )
    output_container = streamed_trace.ops[streamed_trace.output_layers[0]].container
    assert isinstance(output_container, tl.Container)
    assert output_container.reconstructable is False

    loaded = tl.load(path)
    _assert_structural_roundtrip(loaded)
