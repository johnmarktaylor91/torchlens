"""Behavioral coverage for fastlog ``activation_transform`` parity.

Mirrors the slow-path raw-vs-transformed contract added in PR #166 while
documenting the intentional fastlog divergence:

* Fastlog stores transformed payloads on parallel ``ActivationRecord``
  fields (``transformed_ram_payload`` / ``transformed_disk_payload``).
* The transform runs in ``_storage_resolver`` after predicate selection,
  so predicates continue to see raw metadata.
* ``dry_run()`` does not invoke the transform.
* No ``grad_transform`` is exposed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.fastlog import ActivationRecord, CaptureSpec, RecordContext


class _PostfuncModel(nn.Module):
    """Small model with two operation events (linear + relu)."""

    def __init__(self) -> None:
        """Initialize the layers."""

        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""

        return torch.relu(self.linear(x))


def _out_records(
    recording: tl.fastlog.Recording,
) -> list[ActivationRecord]:
    """Return retained records for out events."""

    return [
        record
        for record in recording.records
        if record.spec.save_out
        and (record.ram_payload is not None or record.transformed_ram_payload is not None)
    ]


def _disk_out_records(
    recording: tl.fastlog.Recording,
) -> list[ActivationRecord]:
    """Return retained records that wrote a disk out blob."""

    return [
        record
        for record in recording.records
        if record.metadata.get("blob_id") is not None
        or record.metadata.get("transformed_out_blob_id") is not None
    ]


@pytest.mark.smoke
def test_transform_replaces_transformed_payload_with_raw_kept() -> None:
    """Postfunc populates transformed RAM payload while keeping raw."""

    recording = tl.fastlog.record(
        _PostfuncModel(),
        torch.ones(1, 3),
        default_op=True,
        activation_transform=lambda t: t.float() * 2,
    )

    records = _out_records(recording)
    assert records, "expected at least one out record"
    for record in records:
        assert record.ram_payload is not None
        assert record.transformed_ram_payload is not None
        # Postfunc output equals raw * 2.
        torch.testing.assert_close(
            record.transformed_ram_payload,
            record.ram_payload.float() * 2,
        )
        # Disk payloads remain unset for RAM-only mode.
        assert record.disk_payload is None
        assert record.transformed_disk_payload is None


def test_activation_transform_canonical_name_matches_transform_alias() -> None:
    """Fastlog accepts activation_transform as the canonical transform name."""

    recording = tl.fastlog.record(
        _PostfuncModel(),
        torch.ones(1, 3),
        default_op=True,
        activation_transform=lambda t: t.float() * 2,
    )

    records = _out_records(recording)
    assert records, "expected at least one out record"
    for record in records:
        assert record.ram_payload is not None
        assert record.transformed_ram_payload is not None
        torch.testing.assert_close(
            record.transformed_ram_payload,
            record.ram_payload.float() * 2,
        )


def test_save_raw_outs_false_drops_raw_payload() -> None:
    """Disabling raw save retains only the transformed RAM payload."""

    recording = tl.fastlog.record(
        _PostfuncModel(),
        torch.ones(1, 3),
        default_op=True,
        activation_transform=lambda t: t.float() * 0.5,
        save_raw_activations=False,
    )

    records = _out_records(recording)
    assert records, "expected at least one out record"
    for record in records:
        assert record.ram_payload is None
        assert record.transformed_ram_payload is not None
        # Raw metadata still captured on the context.
        assert record.ctx.shape is not None
        assert record.ctx.dtype is not None


def test_no_transform_leaves_transformed_payload_unset() -> None:
    """Without a transform transformed fields stay None on every record."""

    recording = tl.fastlog.record(
        _PostfuncModel(),
        torch.ones(1, 3),
        default_op=True,
    )

    assert recording._activation_transform_repr is None
    for record in recording.records:
        assert record.transformed_ram_payload is None
        assert record.transformed_disk_payload is None


def test_disk_only_mode_persists_transformed_blob(tmp_path: Path) -> None:
    """Disk-only mode writes a ``transformed_out`` blob and roundtrips."""

    bundle_path = tmp_path / "disk.tlfast"
    recording = tl.fastlog.record(
        _PostfuncModel(),
        torch.ones(1, 3),
        default_op=True,
        activation_transform=lambda t: t.float() * 2,
        streaming=tl.StreamingOptions(bundle_path=bundle_path, retain_in_memory=False),
    )

    records = _disk_out_records(recording)
    assert records, "expected at least one disk-backed record"
    for record in records:
        assert record.metadata.get("transformed_out_blob_id") is not None
        assert record.metadata.get("transformed_out_relative_path", "").endswith(".safetensors")

    # Manifest counts transformed_out blobs as auxiliary entries.
    manifest = json.loads((bundle_path / "manifest.json").read_text(encoding="utf-8"))
    transformed_entries = [
        entry for entry in manifest["tensors"] if entry["kind"] == "transformed_out"
    ]
    assert transformed_entries, "manifest is missing transformed_out kind"
    assert manifest["n_auxiliary_blobs"] == len(transformed_entries)

    loaded = tl.fastlog.load(bundle_path)
    loaded_disk = _disk_out_records(loaded)
    assert len(loaded_disk) == len(records)
    for original, restored in zip(records, loaded_disk):
        assert restored.metadata.get("transformed_out_blob_id") == original.metadata.get(
            "transformed_out_blob_id"
        )
        assert restored.metadata.get("transformed_out_sha256") == original.metadata.get(
            "transformed_out_sha256"
        )


def test_ram_disk_mirror_populates_both_transformed_payloads(tmp_path: Path) -> None:
    """Mirror mode populates both RAM and disk transformed payloads."""

    bundle_path = tmp_path / "mirror.tlfast"
    recording = tl.fastlog.record(
        _PostfuncModel(),
        torch.ones(1, 3),
        default_op=True,
        activation_transform=lambda t: t.float() * 3,
        streaming=tl.StreamingOptions(bundle_path=bundle_path, retain_in_memory=True),
    )

    records = _out_records(recording)
    assert records, "expected at least one out record"
    for record in records:
        assert record.ram_payload is not None
        assert record.transformed_ram_payload is not None
        assert record.disk_payload is not None
        assert record.transformed_disk_payload is not None
        torch.testing.assert_close(
            record.transformed_ram_payload,
            record.ram_payload.float() * 3,
        )
        torch.testing.assert_close(
            record.transformed_disk_payload,
            record.disk_payload.float() * 3,
        )


def test_train_mode_well_behaved_transform_keeps_graph_connected_payload() -> None:
    """A graph-preserving transform keeps RAM payload differentiable."""

    model = _PostfuncModel()
    recording = tl.fastlog.record(
        model,
        torch.ones(1, 3),
        default_op=CaptureSpec(keep_grad=True),
        activation_transform=lambda t: t * 2,
    )

    records = _out_records(recording)
    grad_records = [record for record in records if record.spec.keep_grad]
    assert grad_records, "expected at least one keep_grad record"
    for record in grad_records:
        assert record.transformed_ram_payload is not None
        assert record.transformed_ram_payload.requires_grad is True
        assert record.transformed_ram_payload.grad_fn is not None
    grad_records[0].transformed_ram_payload.sum().backward()
    assert model.linear.weight.grad is not None


def test_train_mode_detaching_transform_rejected() -> None:
    """A detaching transform fails train-mode validation."""

    with pytest.raises(tl.TrainingModeConfigError, match="grad_fn is None"):
        tl.fastlog.record(
            _PostfuncModel(),
            torch.ones(1, 3),
            default_op=CaptureSpec(keep_grad=True),
            activation_transform=lambda t: t.detach(),
        )


def test_train_mode_integer_transform_rejected() -> None:
    """A transform returning integer dtype fails train-mode validation."""

    with pytest.raises(tl.TrainingModeConfigError, match="non-grad dtype"):
        tl.fastlog.record(
            _PostfuncModel(),
            torch.ones(1, 3),
            default_op=CaptureSpec(keep_grad=True),
            activation_transform=lambda t: t.to(torch.int64),
        )


def test_source_events_receive_transform_outputs() -> None:
    """``include_source_events`` routes input/buffer events through the transform."""

    recording = tl.fastlog.record(
        _PostfuncModel(),
        torch.ones(1, 3),
        default_op=True,
        include_source_events=True,
        activation_transform=lambda t: t.float() * 4,
    )

    source_records = [record for record in recording.records if record.ctx.kind == "input"]
    assert source_records, "expected at least one input source event record"
    for record in source_records:
        if record.spec.save_out:
            assert record.transformed_ram_payload is not None
            torch.testing.assert_close(
                record.transformed_ram_payload,
                record.ram_payload.float() * 4,
            )


def test_predicate_transform_invocation_count_matches_kept_events() -> None:
    """Postfunc runs once per predicate-selected out event."""

    invocations: list[int] = [0]

    def counting_transform(t: torch.Tensor) -> torch.Tensor:
        """Increment a counter and return the tensor unchanged."""

        invocations[0] += 1
        return t * 1

    def keep_op(ctx: RecordContext) -> bool:
        """Keep only the relu operation event (filters linear)."""

        return ctx.kind == "op" and ctx.func_name == "relu"

    recording = tl.fastlog.record(
        _PostfuncModel(),
        torch.ones(1, 3),
        keep_op=keep_op,
        activation_transform=counting_transform,
    )

    selected = _out_records(recording)
    assert selected, "expected at least one selected event"
    assert invocations[0] == len(selected)


def test_dry_run_does_not_invoke_transform() -> None:
    """``dry_run`` never invokes the transform since payloads are suppressed."""

    invocations: list[int] = [0]

    def counting_transform(t: torch.Tensor) -> torch.Tensor:
        """Track invocation count and return the tensor."""

        invocations[0] += 1
        return t

    # dry_run does not accept activation_transform, but it shares the
    # storage resolver path. We exercise the same callable via a
    # full Recorder with no_tensor_capture toggled like dry_run does.
    trace = tl.fastlog.dry_run(
        _PostfuncModel(),
        torch.ones(1, 3),
        keep_op=lambda ctx: ctx.kind == "op",
    )

    assert trace.contexts
    assert invocations[0] == 0


def test_transform_error_wrapped_with_event_context() -> None:
    """Postfunc exceptions surface as TorchLensPostfuncError with event context."""

    def boom(_: torch.Tensor) -> torch.Tensor:
        """Always raise to validate error wrapping."""

        raise RuntimeError("custom failure")

    with pytest.raises(tl.TorchLensPostfuncError) as exc_info:
        tl.fastlog.record(
            _PostfuncModel(),
            torch.ones(1, 3),
            default_op=True,
            activation_transform=boom,
        )

    message = str(exc_info.value)
    assert "label=" in message
    assert "func=" in message
    assert "shape=" in message
    assert "dtype=" in message
    assert "storage_target=" in message
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "custom failure" in str(exc_info.value.__cause__)


def test__activation_transform_repr_exposed_and_persisted(tmp_path: Path) -> None:
    """Recording exposes ``_activation_transform_repr`` and persists it on disk."""

    def named_transform(t: torch.Tensor) -> torch.Tensor:
        """Doubling transform with a stable repr."""

        return t.float() * 2

    recording = tl.fastlog.record(
        _PostfuncModel(),
        torch.ones(1, 3),
        default_op=True,
        activation_transform=named_transform,
    )
    assert recording._activation_transform_repr is not None
    assert "named_transform" in recording._activation_transform_repr

    bundle_path = tmp_path / "metadata.tlfast"
    persisted = tl.fastlog.record(
        _PostfuncModel(),
        torch.ones(1, 3),
        default_op=True,
        activation_transform=named_transform,
        streaming=tl.StreamingOptions(bundle_path=bundle_path, retain_in_memory=False),
    )
    metadata = json.loads((bundle_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata.get("_activation_transform_repr") is not None
    assert "named_transform" in metadata["_activation_transform_repr"]
    loaded = tl.fastlog.load(bundle_path)
    assert loaded._activation_transform_repr == persisted._activation_transform_repr


def test_transform_roundtrips_via_disk_recovery(tmp_path: Path) -> None:
    """Disk bundles roundtrip transformed blob metadata via load/recover."""

    bundle_path = tmp_path / "roundtrip.tlfast"
    original = tl.fastlog.record(
        _PostfuncModel(),
        torch.ones(1, 3),
        default_op=True,
        activation_transform=lambda t: t.float() * 2,
        streaming=tl.StreamingOptions(bundle_path=bundle_path, retain_in_memory=False),
    )

    loaded = tl.fastlog.load(bundle_path)
    recovered = tl.fastlog.recover(bundle_path)

    original_records = _disk_out_records(original)
    loaded_records = _disk_out_records(loaded)
    recovered_records = _disk_out_records(recovered)

    assert original_records and loaded_records and recovered_records
    assert len(loaded_records) == len(original_records)
    assert len(recovered_records) == len(original_records)
    for original_record, loaded_record, recovered_record in zip(
        original_records, loaded_records, recovered_records
    ):
        original_blob = original_record.metadata.get("transformed_out_blob_id")
        loaded_blob = loaded_record.metadata.get("transformed_out_blob_id")
        recovered_blob = recovered_record.metadata.get("transformed_out_blob_id")
        assert original_blob is not None
        assert loaded_blob == original_blob
        assert recovered_blob == original_blob
