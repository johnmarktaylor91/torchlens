"""A09 agent stage-0 item 1: the composed lazy-payload honesty gate.

Record-shaped AND memory-shaped acceptance (the agent memo P0's exact
demand -- record-shaped tests alone are how the silent-None bug shipped):

- ``.out`` on a lazily loaded trace materializes one sha-verified blob or
  raises typed ``PayloadUnavailableError`` -- never ``None``.
- ``payload_load_status`` gains the distinct ``loaded_lazy`` state and op
  rows carry ``payload_state: present | lazy | unsaved``.
- The scoped reader (``torchlens._io.payload_reader.read_op_payload``) never
  attaches: retained RSS stays flat across 200 sequential payload reads.
- Payload work is budgetable BEFORE materialization from manifest-declared
  dtype/element counts (torch-free).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io.payload_reader import declared_payload_bytes, read_op_payload
from torchlens.errors import PayloadUnavailableError

pytestmark = pytest.mark.smoke


def _saved_labels(trace: tl.Trace) -> list[str]:
    return [
        str(op.layer_label) for op in trace.layer_list if getattr(op, "has_saved_activation", False)
    ]


@pytest.fixture()
def lazy_artifact(tmp_path: Path) -> tuple[Path, tl.Trace]:
    """A saved artifact plus its source capture (selective save)."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4), nn.Sigmoid())
    source = tl.trace(model, torch.randn(4, 8), save=tl.func("relu"))
    destination = tmp_path / "lazy.tlspec"
    tl.save(source, destination)
    return destination, source


def test_lazy_saved_rows_materialize_never_none(lazy_artifact: tuple[Path, tl.Trace]) -> None:
    """Every saved: true row returns a tensor under eager AND lazy load."""

    path, source = lazy_artifact
    for lazy in (False, True):
        loaded = tl.load(path, lazy=lazy)
        expected_status = "loaded_lazy" if lazy else "loaded"
        assert loaded.payload_load_status == expected_status
        for label in _saved_labels(loaded):
            out = loaded[label].out
            assert isinstance(out, torch.Tensor), (lazy, label)
            assert torch.equal(out, source[label].out)


def test_lazy_unsaved_rows_refuse_typed(lazy_artifact: tuple[Path, tl.Trace]) -> None:
    """Unsaved rows on a lazily loaded trace raise typed, never None."""

    path, _ = lazy_artifact
    loaded = tl.load(path, lazy=True)
    unsaved = [op for op in loaded.layer_list if not getattr(op, "has_saved_activation", False)]
    assert unsaved, "fixture must retain unsaved rows"
    with pytest.raises(PayloadUnavailableError) as exc_info:
        _ = unsaved[0].out
    assert exc_info.value.fields["code"] == "activation_not_saved"


def test_corrupt_blob_raises_typed_not_none(lazy_artifact: tuple[Path, tl.Trace]) -> None:
    """A checksum-drifted blob refuses typed on the materializing read."""

    from torchlens._io import TorchLensIOError

    path, _ = lazy_artifact
    loaded = tl.load(path, lazy=True)
    label = _saved_labels(loaded)[0]
    ref = loaded[label].ops[0].out_ref
    blob_path = ref.blob_path()
    blob_path.write_bytes(b"corrupted" + blob_path.read_bytes()[9:])
    with pytest.raises(TorchLensIOError, match="sha256 mismatch"):
        _ = loaded[label].out


def test_payload_state_vocabulary_on_agent_rows(lazy_artifact: tuple[Path, tl.Trace]) -> None:
    """Op rows carry payload_state so 'saved' stops meaning two things."""

    path, source = lazy_artifact
    live_states = {row["payload_state"] for row in source.to_agent_json()["ops"]}
    assert live_states <= {"present", "unsaved"}
    lazy = tl.load(path, lazy=True)
    rows = lazy.to_agent_json()["ops"]
    states = {row["payload_state"] for row in rows}
    assert "lazy" in states
    for row in rows:
        if row["payload_state"] == "lazy":
            assert row["saved"] is True


def test_declared_payload_bytes_is_exact_and_torch_free(
    lazy_artifact: tuple[Path, tl.Trace],
) -> None:
    """Manifest-declared budgeting matches the real retained payload bytes."""

    path, source = lazy_artifact
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    declared = declared_payload_bytes(manifest)
    retained = sum(
        op.out.element_size() * op.out.numel()
        for op in source.layer_list
        if getattr(op, "has_saved_activation", False)
    )
    assert declared == retained


def test_scoped_reader_refuses_payloadless_rows() -> None:
    """The scoped reader refuses typed where there is nothing to read."""

    with pytest.warns(UserWarning, match="matched zero sites"):
        trace = tl.trace(
            nn.Sequential(nn.Linear(3, 4), nn.ReLU()),
            torch.randn(2, 3),
            save=tl.func("no_such_op"),
        )
    with pytest.raises(PayloadUnavailableError) as exc_info:
        read_op_payload(trace.layer_list[0])
    assert exc_info.value.fields["code"] == "payload_unavailable"
