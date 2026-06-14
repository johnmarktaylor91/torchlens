"""Tests for Phase 11 unified ``.tlspec`` writer and reader behavior."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends import BackendPayloadUnsupportedError
from torchlens.intervention.types import InterventionSpec
from torchlens.options import CaptureOptions
from torchlens.validation import validate_tlspec


class UnifiedTinyModel(nn.Module):
    """Tiny model with one intervention-friendly operation."""

    def __init__(self) -> None:
        """Initialize the tiny model."""

        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return torch.relu(self.linear(x))


class UnifiedSavedOrphanModel(nn.Module):
    """Tiny model with a saved factory tensor outside the output graph."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create an orphan tensor and return an unrelated output.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output unrelated to the orphan tensor.
        """

        _unused = torch.randn(x.shape)
        return x + 1


def _captured_log(*, intervention_ready: bool = False) -> tl.Trace:
    """Create a deterministic captured model log.

    Parameters
    ----------
    intervention_ready:
        Whether to include intervention replay metadata.

    Returns
    -------
    tl.Trace
        Captured log.
    """

    torch.manual_seed(2100)
    return tl.trace(
        UnifiedTinyModel().eval(),
        torch.randn(2, 3),
        capture=CaptureOptions(
            intervention_ready=intervention_ready,
            layers_to_save="all",
            random_seed=0,
        ),
    )


def _read_manifest(path: Path) -> dict[str, Any]:
    """Read one manifest JSON object.

    Parameters
    ----------
    path:
        ``.tlspec`` path.

    Returns
    -------
    dict[str, Any]
        Decoded manifest.
    """

    data = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write one manifest JSON object.

    Parameters
    ----------
    path:
        ``.tlspec`` path.
    manifest:
        Manifest object to persist.
    """

    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def _mlx_schema_v2_manifest(path: Path) -> dict[str, Any]:
    """Return a schema v2 non-torch audit manifest based on a torch fixture.

    Parameters
    ----------
    path:
        Existing ``.tlspec`` path whose v1 manifest supplies common fields.

    Returns
    -------
    dict[str, Any]
        Mutable schema v2 manifest.
    """

    manifest = _read_manifest(path)
    manifest.update(
        {
            "schema_version": 2,
            "backend": "mlx",
            "backend_runtime": {
                "name": "mlx",
                "version": "0.0.test",
                "runtime_config": {},
                "device_summary": {},
                "compat_policy": {"policy": "audit-only-test"},
            },
            "torch_version": None,
            "model_fingerprint": {
                "backend_fingerprint": {
                    "callable_identity": "tests.test_tlspec_unified.mlx_fixture",
                    "graph_digest": "mlx-audit-fixture",
                }
            },
            "body_format": "audit_only",
            "body_index": [
                {
                    "filename": "audit.json",
                    "dtype": "mlx.float32",
                    "shape": [2, 3],
                    "num_elements": 6,
                    "intended_use": "audit_record",
                    "sha256": "0" * 64,
                }
            ],
            "backward_summary": None,
            "derived_gradient_summary": None,
            "payload_policy": {
                "policy": "audit_only",
                "materialization_supported": False,
                "payload_kinds": ["audit_record"],
            },
        }
    )
    return manifest


def _mlx_materialized_schema_v2_manifest(path: Path) -> dict[str, Any]:
    """Return a schema v2 MLX materialized manifest backed by real blobs.

    Parameters
    ----------
    path:
        Existing ``.tlspec`` path whose saved blobs supply the body files.

    Returns
    -------
    dict[str, Any]
        Mutable schema v2 manifest with MLX codec body entries.
    """

    manifest = _read_manifest(path)
    body_index = manifest["body_index"]
    tensor_entries = manifest["tensors"]
    for entry in [*body_index, *tensor_entries]:
        logical_dtype = f"mlx.core.{entry['dtype']}"
        entry.update(
            {
                "logical_backend": "mlx",
                "codec": "numpy_safetensors_v1",
                "logical_dtype": logical_dtype,
                "logical_device": "unknown",
                "transport_backend": "safetensors.torch",
                "transport_dtype": entry["dtype"],
                "codec_metadata": {"logical_shape": entry["shape"]},
            }
        )
    manifest.update(
        {
            "schema_version": 2,
            "backend": "mlx",
            "backend_runtime": {
                "name": "mlx",
                "version": "0.0.test",
                "runtime_config": {},
                "device_summary": {},
                "compat_policy": {"policy": "materialized-test"},
            },
            "torch_version": None,
            "model_fingerprint": {
                "backend_fingerprint": {
                    "callable_identity": "tests.test_tlspec_unified.mlx_fixture",
                    "graph_digest": "mlx-materialized-fixture",
                }
            },
            "body_format": "safetensors",
            "backward_summary": None,
            "derived_gradient_summary": None,
            "payload_policy": {
                "policy": "array_payloads",
                "materialization_supported": True,
                "payload_kinds": sorted({entry["intended_use"] for entry in body_index}),
            },
        }
    )
    return manifest


def test_schema_v2_accepts_array_payload_policy_and_codec_body_fields(tmp_path: Path) -> None:
    """Schema v2 admits the MLX materialized array-payload vocabulary."""

    path = tmp_path / "array_payload_vocab.tlspec"
    _captured_log().save(path)
    manifest = _mlx_schema_v2_manifest(path)
    manifest["body_format"] = "safetensors"
    manifest["payload_policy"] = {
        "policy": "array_payloads",
        "materialization_supported": True,
        "payload_kinds": ["out"],
    }
    manifest["body_index"][0].update(
        {
            "logical_backend": "mlx",
            "codec": "numpy_safetensors_v1",
            "logical_dtype": "mlx.core.float32",
            "logical_device": "unknown",
            "transport_backend": "safetensors.torch",
            "transport_dtype": "float32",
            "codec_metadata": {"logical_shape": [2, 3]},
        }
    )
    _write_manifest(path, manifest)

    validate_tlspec(path)


@pytest.mark.smoke
@pytest.mark.parametrize("level", ["audit", "executable_with_callables", "portable"])
def test_unified_modellog_round_trips_per_save_level(tmp_path: Path, level: str) -> None:
    """Trace.save writes unified manifests that load polymorphically."""

    log = _captured_log()
    path = tmp_path / f"model_{level}.tlspec"

    log.save(path, level=level)
    validate_tlspec(path)
    loaded = tl.load(path)
    manifest = _read_manifest(path)

    assert isinstance(loaded, tl.Trace)
    assert manifest["kind"] == "trace"
    assert manifest["tlspec_version"] == 1
    assert manifest["save_level"] == level
    assert [layer.layer_label for layer in loaded.layer_list] == [
        layer.layer_label for layer in log.layer_list
    ]


@pytest.mark.smoke
def test_unified_portable_round_trips_orphan_records(tmp_path: Path) -> None:
    """Portable saves preserve orphan record payload tensors."""

    x = torch.ones(2, 2)
    log = tl.trace(UnifiedSavedOrphanModel(), x, save=tl.func("randn"), random_seed=1)
    path = tmp_path / "orphan_records.tlspec"
    expected_payload = log.orphan_records[0]["payload_ref"].detach().clone()

    log.save(path, level="portable")
    validate_tlspec(path)
    loaded = tl.load(path)
    manifest = _read_manifest(path)

    assert isinstance(loaded, tl.Trace)
    assert loaded.orphan_records
    assert loaded.orphan_records[0]["raw_label"].startswith("randn")
    assert isinstance(loaded.orphan_records[0]["payload_ref"], torch.Tensor)
    assert torch.equal(loaded.orphan_records[0]["payload_ref"], expected_payload)
    assert any(entry["intended_use"] == "orphan_payload" for entry in manifest["body_index"])


def test_full_resnet18_default_trace_bundle_validates_and_loads(tmp_path: Path) -> None:
    """Full default ResNet18 traces with buffers write schema-valid bundles."""

    torchvision_models = pytest.importorskip("torchvision.models")
    torch.manual_seed(2101)
    model = torchvision_models.resnet18(weights=None).eval()
    x = torch.randn(1, 3, 64, 64)
    log = tl.trace(model, x, capture=CaptureOptions(random_seed=2101))
    path = tmp_path / "resnet18_full_default.tlspec"

    log.save(path, level="portable")
    validate_tlspec(path)
    loaded = tl.load(path)
    manifest = _read_manifest(path)

    assert isinstance(loaded, tl.Trace)
    assert loaded.num_ops == log.num_ops
    assert [layer.label for layer in loaded.layer_list] == [layer.label for layer in log.layer_list]
    assert any(entry["intended_use"] == "buffer_initial_value" for entry in manifest["body_index"])


@pytest.mark.smoke
@pytest.mark.parametrize("level", ["audit", "executable_with_callables", "portable"])
def test_unified_bundle_round_trips_per_save_level(tmp_path: Path, level: str) -> None:
    """Bundle.save writes unified manifests that load as Bundle objects."""

    log = _captured_log()
    bundle = tl.bundle({"baseline": log}, baseline="baseline")
    path = tmp_path / f"bundle_{level}.tlspec"

    bundle.save(path, level=level)
    validate_tlspec(path)
    loaded = tl.load(path)
    manifest = _read_manifest(path)

    assert isinstance(loaded, tl.Bundle)
    assert manifest["kind"] == "bundle"
    assert manifest["save_level"] == level
    assert loaded.names == ["baseline"]
    assert isinstance(loaded["baseline"], tl.Trace)


@pytest.mark.smoke
@pytest.mark.parametrize("level", ["audit", "executable_with_callables", "portable"])
def test_unified_intervention_round_trips_per_save_level(tmp_path: Path, level: str) -> None:
    """Intervention saves now emit the full unified manifest field set."""

    log = _captured_log(intervention_ready=True)
    log.set(tl.func("relu"), tl.zero_ablate(), confirm_mutation=True)
    path = tmp_path / f"intervention_{level}.tlspec"

    log.save_intervention(path, level=level)
    validate_tlspec(path)
    loaded = tl.load(path)
    manifest = _read_manifest(path)

    assert isinstance(loaded, InterventionSpec)
    assert manifest["kind"] == "intervention"
    assert manifest["tlspec_version"] == 1
    assert manifest["format_version"] == "1"
    assert manifest["save_level"] == level
    assert loaded.metadata["save_level"] == level


@pytest.mark.smoke
def test_inspect_tlspec_returns_unified_manifest(tmp_path: Path) -> None:
    """Manifest inspection returns parsed unified metadata."""

    path = tmp_path / "inspect.tlspec"
    _captured_log().save(path)

    manifest = tl.io.inspect_tlspec(path)

    assert manifest["kind"] == "trace"
    assert manifest["body_format"] == "safetensors"
    assert isinstance(manifest["sites"], list)


@pytest.mark.smoke
def test_validate_tlspec_rejects_missing_unified_field(tmp_path: Path) -> None:
    """Schema validation fails closed for malformed unified manifests."""

    path = tmp_path / "invalid.tlspec"
    _captured_log().save(path)
    manifest = _read_manifest(path)
    manifest.pop("model_fingerprint")
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="model_fingerprint"):
        validate_tlspec(path)


@pytest.mark.smoke
def test_unified_manifest_records_backward_summary(tmp_path: Path) -> None:
    """Trace.save records backward fields and gradient blob kinds in the manifest."""

    torch.manual_seed(2101)
    model = UnifiedTinyModel().eval()
    x = torch.randn(2, 3, requires_grad=True)
    log = tl.trace(model, x, save_grads=True)
    log.log_backward(log[log.output_layers[0]].out.sum())
    path = tmp_path / "backward.tlspec"

    log.save(path)
    validate_tlspec(path)
    loaded = tl.load(path)
    manifest = _read_manifest(path)

    assert isinstance(loaded, tl.Trace)
    assert loaded.has_backward_pass is True
    assert loaded.backward_passes.for_pass(1).pass_index == 1
    assert manifest["backward_summary"]["has_backward_pass"] is True
    assert manifest["backward_summary"]["num_backward_passes"] == log.num_backward_passes
    assert manifest["backward_summary"]["num_grad_fns"] == log.num_grad_fns
    assert manifest["backward_summary"]["gradient_blob_count"] >= 1
    assert "grad" in manifest["backward_summary"]["gradient_blob_kinds"]
    assert any(entry["intended_use"] == "grad" for entry in manifest["body_index"])
    assert (
        "removed backward gradient configuration fields"
        in (manifest["backward_summary"]["old_bundle_policy"])
    )


@pytest.mark.smoke
def test_validate_tlspec_rejects_bad_backward_blob_kind(tmp_path: Path) -> None:
    """Schema validation rejects malformed backward blob kinds."""

    torch.manual_seed(2102)
    model = UnifiedTinyModel().eval()
    x = torch.randn(2, 3, requires_grad=True)
    log = tl.trace(model, x, save_grads=True)
    log.log_backward(log[log.output_layers[0]].out.sum())
    path = tmp_path / "bad_backward_kind.tlspec"
    log.save(path)
    manifest = _read_manifest(path)
    manifest["backward_summary"]["gradient_blob_kinds"] = ["grad", "legacy_grad"]
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="gradient_blob_kinds"):
        validate_tlspec(path)


@pytest.mark.smoke
def test_schema_v2_mlx_materialized_manifest_validates(tmp_path: Path) -> None:
    """Schema v2 accepts MLX materialized backend/runtime and body fields."""

    path = tmp_path / "mlx_materialized.tlspec"
    _captured_log().save(path)
    manifest = _mlx_materialized_schema_v2_manifest(path)
    _write_manifest(path, manifest)

    validate_tlspec(path)


@pytest.mark.smoke
def test_schema_v2_mlx_materialized_loads_payloads(tmp_path: Path) -> None:
    """MLX schema v2 materialized traces load body entries as MLX arrays."""

    pytest.importorskip("mlx")
    import mlx.core as mx

    path = tmp_path / "mlx_materialized_load.tlspec"
    _captured_log().save(path)
    _write_manifest(path, _mlx_materialized_schema_v2_manifest(path))

    loaded = tl.load(path)

    assert isinstance(loaded, tl.Trace)
    assert loaded.backend == "mlx"
    assert getattr(loaded, "payload_load_status") == "loaded_device_best_effort"
    saved_ops = [op for op in loaded.layer_list if op.has_saved_activation]
    assert saved_ops
    assert all(isinstance(op.out, mx.array) for op in saved_ops)
    assert loaded.validation_replay_status.state == "unavailable"
    assert loaded.validation_replay_status.reason == "loaded_trace_runtime_capture_stripped"


@pytest.mark.smoke
def test_schema_v2_mlx_old_audit_only_fixture_loads_metadata_only(tmp_path: Path) -> None:
    """Old MLX audit-only schema-v2 bundles should still load metadata-only."""

    path = tmp_path / "mlx_old_audit_only.tlspec"
    _captured_log().save(path)
    manifest = _mlx_schema_v2_manifest(path)
    manifest["unsupported_tensors"].append(
        {
            "label": "relu_1_2_raw:1",
            "kind": "out",
            "reason": "mlx_array_audit_null",
        }
    )
    _write_manifest(path, manifest)

    validate_tlspec(path)
    loaded = tl.load(path)
    saved_ops = [op for op in loaded.layer_list if op.has_saved_activation]

    assert isinstance(loaded, tl.Trace)
    assert loaded.backend == "mlx"
    assert getattr(loaded, "payload_load_status") == "audit_only"
    assert saved_ops
    assert all(op.out is None and op.out_ref is None for op in saved_ops)
    assert all(op.shape is not None and op.dtype_ref is not None for op in saved_ops)
    assert any(
        record["reason"] == "mlx_array_audit_null"
        for record in _read_manifest(path)["unsupported_tensors"]
    )


@pytest.mark.optional
def test_mlx_public_save_writes_materialized_manifest_body(tmp_path: Path) -> None:
    """Public MLX portable saves write codec-backed safetensors body entries."""

    pytest.importorskip("mlx")
    import mlx.core as mx
    import mlx.nn as mlx_nn

    class MlxTinyModel(mlx_nn.Module):
        """Small MLX model used for public tlspec save assertions."""

        def __init__(self) -> None:
            """Initialize the projection layer."""

            super().__init__()
            self.proj = mlx_nn.Linear(3, 2)

        def __call__(self, x: mx.array) -> mx.array:
            """Run the projection."""

            return self.proj(x)

    trace = tl.trace(MlxTinyModel(), mx.ones((1, 3)))
    path = tmp_path / "mlx_public_materialized.tlspec"

    tl.save(trace, path, level="portable")
    validate_tlspec(path)
    manifest = _read_manifest(path)

    assert manifest["backend"] == "mlx"
    assert manifest["body_format"] == "safetensors"
    assert manifest["payload_policy"]["policy"] == "array_payloads"
    assert manifest["payload_policy"]["materialization_supported"] is True
    assert manifest["payload_policy"]["payload_kinds"]
    assert manifest["body_index"]
    assert all(entry["logical_backend"] == "mlx" for entry in manifest["body_index"])
    assert all(entry["codec"] == "numpy_safetensors_v1" for entry in manifest["body_index"])


@pytest.mark.smoke
def test_schema_v2_materialized_unknown_codec_fails_closed(tmp_path: Path) -> None:
    """Materialized schema-v2 manifests with unknown codecs fail before loading."""

    path = tmp_path / "jax_unknown_codec.tlspec"
    _captured_log().save(path)
    manifest = _mlx_schema_v2_manifest(path)
    manifest["backend"] = "jax"
    manifest["backend_runtime"]["name"] = "jax"
    manifest["payload_policy"] = {
        "policy": "array_payloads",
        "materialization_supported": True,
        "payload_kinds": ["out"],
    }
    manifest["body_index"][0].update(
        {
            "logical_backend": "jax",
            "codec": "unknown_codec_v1",
            "logical_dtype": "float32",
            "logical_device": "cpu",
        }
    )
    _write_manifest(path, manifest)

    with pytest.raises(BackendPayloadUnsupportedError, match="unknown_codec_v1"):
        tl.load(path)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda manifest: manifest.pop("backend"), "backend"),
        (lambda manifest: manifest.__setitem__("backend", "unknown-backend"), "unknown backend"),
        (lambda manifest: manifest.pop("backend_runtime"), "backend_runtime"),
        (
            lambda manifest: manifest.__setitem__("backward_summary", {"has_backward_pass": True}),
            "backward_summary",
        ),
        (lambda manifest: manifest.__setitem__("torch_version", "2.0.0"), "torch_version=null"),
    ],
)
def test_schema_v2_non_torch_corruptions_fail_closed(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], Any],
    match: str,
) -> None:
    """Schema v2 backend-conditional fields reject malformed non-torch manifests."""

    path = tmp_path / "bad_v2.tlspec"
    _captured_log().save(path)
    manifest = _mlx_schema_v2_manifest(path)
    mutation(manifest)
    _write_manifest(path, manifest)

    with pytest.raises(ValueError, match=match):
        validate_tlspec(path)


@pytest.mark.smoke
def test_schema_v1_rejects_schema_v2_body_uses(tmp_path: Path) -> None:
    """Schema v1 fixtures stay torch-only and reject v2 intended-use literals."""

    path = tmp_path / "v1_with_v2_body_use.tlspec"
    _captured_log().save(path)
    manifest = _read_manifest(path)
    manifest["body_index"] = [
        {
            "filename": "audit.json",
            "dtype": "mlx.float32",
            "shape": [1],
            "num_elements": 1,
            "intended_use": "audit_record",
        }
    ]
    _write_manifest(path, manifest)

    with pytest.raises(ValueError, match="intended_use"):
        validate_tlspec(path)
