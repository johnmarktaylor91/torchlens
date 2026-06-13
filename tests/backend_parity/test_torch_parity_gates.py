"""Golden parity gates protecting torch before backend substrate work."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.constants import (
    LAYER_PASS_LOG_FIELD_ORDER,
    MODEL_LOG_FIELD_ORDER,
)
from torchlens.data_classes.op import Op
from torchlens.data_classes.trace import Trace


pytestmark = pytest.mark.backend_parity

_GOLDEN_DIR = Path(__file__).with_name("goldens")
_UPDATE_ENV = "TORCHLENS_UPDATE_BACKEND_PARITY"


class _ParityMLP(nn.Module):
    """Small deterministic model for torch parity fixtures."""

    def __init__(self) -> None:
        """Initialize fixed-shape layers."""

        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a linear-ReLU-linear forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        hidden = torch.relu(self.fc1(x))
        return self.fc2(hidden)


class _BranchyTinyModel(nn.Module):
    """Small model with module containment and repeated public surfaces."""

    def __init__(self) -> None:
        """Initialize two child modules."""

        super().__init__()
        self.proj = nn.Linear(3, 3)
        self.out = nn.Linear(3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a tanh projection plus residual arithmetic.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        projected = torch.tanh(self.proj(x))
        return self.out(projected + x)


def _seeded_mlp_input(*, requires_grad: bool = False) -> tuple[_ParityMLP, torch.Tensor]:
    """Create a deterministic MLP fixture.

    Parameters
    ----------
    requires_grad:
        Whether the input should require gradients.

    Returns
    -------
    tuple[_ParityMLP, torch.Tensor]
        Initialized model and input tensor.
    """

    torch.manual_seed(1701)
    model = _ParityMLP()
    x = torch.randn(2, 3, requires_grad=requires_grad)
    return model, x


def _seeded_branchy_input() -> tuple[_BranchyTinyModel, torch.Tensor]:
    """Create a deterministic branchy fixture.

    Returns
    -------
    tuple[_BranchyTinyModel, torch.Tensor]
        Initialized model and input tensor.
    """

    torch.manual_seed(1702)
    model = _BranchyTinyModel()
    x = torch.randn(2, 3)
    return model, x


def _default_trace() -> Trace:
    """Build the default full-save golden trace.

    Returns
    -------
    Trace
        Current torch default parity trace.
    """

    model, x = _seeded_mlp_input()
    return tl.trace(model, x, layers_to_save="all", random_seed=1701, save_arg_values=True)


def _selective_trace() -> Trace:
    """Build the selective-save golden trace.

    Returns
    -------
    Trace
        Current torch selective parity trace.
    """

    model, x = _seeded_branchy_input()
    return tl.trace(model, x, save=tl.func("tanh"), random_seed=1702)


def _backward_ready_trace() -> Trace:
    """Build the backward-ready golden trace with one logged backward pass.

    Returns
    -------
    Trace
        Current torch backward parity trace.
    """

    model, x = _seeded_mlp_input(requires_grad=True)
    trace = tl.trace(
        model,
        x,
        layers_to_save="all",
        save_grads="all",
        backward_ready=True,
        random_seed=1701,
    )
    trace.log_backward(trace[trace.output_layers[0]].out.sum())
    return trace


def _normalize(value: Any) -> Any:
    """Normalize projection values into deterministic JSON-compatible data.

    Parameters
    ----------
    value:
        Arbitrary projected value.

    Returns
    -------
    Any
        JSON-compatible normalized value.
    """

    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, torch.Size):
        return list(value)
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _digest_payload(payload: dict[str, Any]) -> str:
    """Return a stable SHA256 digest for a projection.

    Parameters
    ----------
    payload:
        Projection payload.

    Returns
    -------
    str
        Hex digest over canonical JSON bytes.
    """

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _digest_chunks(digest: str) -> list[str]:
    """Split a digest into short chunks for committed golden storage.

    Parameters
    ----------
    digest:
        Hex digest string.

    Returns
    -------
    list[str]
        Digest chunks short enough to avoid secret-scanner false positives.
    """

    return [digest[index : index + 8] for index in range(0, len(digest), 8)]


def _digest_from_golden(golden: dict[str, Any]) -> str:
    """Read the digest from a committed golden payload.

    Parameters
    ----------
    golden:
        Decoded golden payload.

    Returns
    -------
    str
        Reconstructed digest.
    """

    return "".join(golden["sha256_chunks"])


def _golden_payload(projection: dict[str, Any]) -> dict[str, Any]:
    """Wrap a projection with its digest.

    Parameters
    ----------
    projection:
        Stable parity projection.

    Returns
    -------
    dict[str, Any]
        Golden file payload.
    """

    return {"sha256_chunks": _digest_chunks(_digest_payload(projection)), "projection": projection}


def _read_or_update_golden(path: Path, projection: dict[str, Any]) -> dict[str, Any]:
    """Read a golden payload, optionally updating it under the update env flag.

    Parameters
    ----------
    path:
        Golden JSON path.
    projection:
        Newly computed projection.

    Returns
    -------
    dict[str, Any]
        Golden payload.
    """

    payload = _golden_payload(projection)
    if os.environ.get(_UPDATE_ENV) == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_projection_matches_golden(name: str, projection: dict[str, Any]) -> None:
    """Assert a projection matches its committed golden digest file.

    Parameters
    ----------
    name:
        Golden basename without suffix.
    projection:
        Newly computed projection.
    """

    golden = _read_or_update_golden(_GOLDEN_DIR / f"{name}.json", projection)
    assert _digest_payload(projection) == _digest_from_golden(golden)
    assert projection == golden["projection"]


def _skip_during_golden_update() -> None:
    """Skip meta-tests while committed goldens are being regenerated."""

    if os.environ.get(_UPDATE_ENV) == "1":
        pytest.skip("can-fail meta-tests are skipped during golden regeneration")


def _op_projection(op: Any) -> dict[str, Any]:
    """Project stable op graph metadata.

    Parameters
    ----------
    op:
        TorchLens op-like object.

    Returns
    -------
    dict[str, Any]
        Stable op projection.
    """

    return {
        "label": op.label,
        "layer_label": op.layer_label,
        "func_name": op.func_name,
        "parents": list(op.parents),
        "children": list(op.children),
        "parent_arg_positions": _normalize(op.parent_arg_positions),
        "has_saved_activation": bool(op.has_saved_activation),
        "has_saved_args": bool(op.has_saved_args),
        "has_grad": bool(op.has_grad),
        "num_saved_grads": len(op.grads),
        "shape": _normalize(op.shape),
        "dtype": _normalize(op.dtype),
        "dtype_ref": None if op.dtype_ref is None else str(op.dtype_ref),
        "device_ref": None if op.device_ref is None else str(op.device_ref),
        "backend_address": _normalize(op.backend_address),
        "resolver_status": op.resolver_status,
        "module_address": _normalize(getattr(op, "module_address", None)),
        "module_pass": _normalize(getattr(op, "module_pass", None)),
        "containing_modules": _normalize(getattr(op, "containing_modules_origin_nested", None)),
        "lookup_keys": _normalize(op.lookup_keys),
    }


def _trace_projection(trace: Trace) -> dict[str, Any]:
    """Project stable trace metadata for digest comparison.

    Parameters
    ----------
    trace:
        Trace to project.

    Returns
    -------
    dict[str, Any]
        Stable trace projection.
    """

    return {
        "backend": trace.backend,
        "module_identity_mode": trace.module_identity_mode,
        "param_source": trace.param_source,
        "num_ops": trace.num_ops,
        "num_saved_ops": trace.num_saved_ops,
        "has_backward_pass": trace.has_backward_pass,
        "num_backward_passes": trace.num_backward_passes,
        "output_layers": list(trace.output_layers),
        "ops": [_op_projection(op) for op in trace.layer_list],
        "lookup_keys": {
            "ops": list(trace.ops.keys()),
            "layers": list(trace.layers.keys()),
            "saved_ops": list(trace.saved_ops.keys()),
            "params": list(trace.params.keys()),
            "modules": list(trace.modules.keys()),
            "grad_fns": list(trace.grad_fns.keys()),
        },
    }


def _manifest_projection(bundle_path: Path) -> dict[str, Any]:
    """Project stable public manifest fields from a saved bundle.

    Parameters
    ----------
    bundle_path:
        Saved ``.tlspec`` directory.

    Returns
    -------
    dict[str, Any]
        Stable manifest projection.
    """

    manifest = json.loads((bundle_path / "manifest.json").read_text(encoding="utf-8"))
    tensors = manifest.get("body_index", manifest.get("tensors", []))
    projected_tensors = [
        {
            "dtype": tensor.get("dtype"),
            "intended_use": tensor.get("intended_use"),
            "shape": tensor.get("shape"),
            "storage_backend": tensor.get("storage_backend", tensor.get("backend")),
        }
        for tensor in tensors
    ]
    return {
        "kind": manifest.get("kind"),
        "schema_version": manifest.get("schema_version"),
        "tlspec_version": manifest.get("tlspec_version"),
        "body_format": manifest.get("body_format"),
        "save_level": manifest.get("save_level"),
        "torch_version_present": bool(manifest.get("torch_version")),
        "tensor_count": len(projected_tensors),
        "tensors": projected_tensors,
    }


def _dataframe_projection(trace: Trace) -> dict[str, Any]:
    """Project the FIELD_ORDER-derived dataframe surface.

    Parameters
    ----------
    trace:
        Trace to project.

    Returns
    -------
    dict[str, Any]
        Stable dataframe projection.
    """

    frame = trace.to_pandas()
    stable_columns = [
        "layer_label",
        "label",
        "func_name",
        "parents",
        "parent_arg_positions",
        "has_saved_activation",
        "has_saved_args",
        "shape",
        "dtype",
        "is_module_input",
        "is_module_output",
    ]
    rows = [
        {column: _normalize(row[column]) for column in stable_columns}
        for row in frame[stable_columns].to_dict(orient="records")
    ]
    return {
        "model_field_order": list(MODEL_LOG_FIELD_ORDER),
        "op_field_order": list(LAYER_PASS_LOG_FIELD_ORDER),
        "trace_portable_fields": sorted(Trace.PORTABLE_STATE_SPEC),
        "op_portable_fields": sorted(Op.PORTABLE_STATE_SPEC),
        "dataframe_columns": list(frame.columns),
        "stable_rows": rows,
    }


def _public_accessor_projection(trace: Trace) -> dict[str, Any]:
    """Project public accessor and grad-adjacent surfaces.

    Parameters
    ----------
    trace:
        Trace to project.

    Returns
    -------
    dict[str, Any]
        Stable public accessor projection.
    """

    first_compute = next(op for op in trace.layer_list if op.func_name not in {"none", None})
    return {
        "ops": {
            "keys": list(trace.ops.keys()),
            "first_label": trace.ops[0].label,
            "first_key_label": trace.ops[trace.ops.keys()[0]].label,
        },
        "layers": {
            "keys": list(trace.layers.keys()),
            "first_label": trace.layers[0].layer_label,
            "first_key_label": trace.layers[trace.layers.keys()[0]].layer_label,
        },
        "params": {
            key: {
                "module_address": param.module_address,
                "shape": _normalize(param.shape),
                "dtype": _normalize(param.dtype),
                "dtype_ref": None if param.dtype_ref is None else str(param.dtype_ref),
                "device_ref": None if param.device_ref is None else str(param.device_ref),
                "backend_address": _normalize(param.backend_address),
                "resolver_status": param.resolver_status,
                "has_grad": bool(param.has_grad),
            }
            for key, param in trace.params.items()
        },
        "modules": {
            key: {
                "num_calls": module.num_calls,
                "num_params": module.num_params,
            }
            for key, module in trace.modules.items()
        },
        "grad_adjacent": {
            "op_label": first_compute.label,
            "has_grad": bool(first_compute.has_grad),
            "num_saved_grads": len(first_compute.grads),
            "grad_is_none": first_compute.grad is None,
            "grad_for_bwd1_shape": _normalize(
                first_compute.grad_for(bwd=1).shape if first_compute.has_grad else None
            ),
        },
    }


@pytest.mark.parametrize(
    ("name", "builder"),
    [
        ("default_trace_digest", _default_trace),
        ("selective_trace_digest", _selective_trace),
        ("backward_ready_trace_digest", _backward_ready_trace),
    ],
)
def test_trace_golden_digest(name: str, builder: Any) -> None:
    """Trace projections match committed digest goldens."""

    _assert_projection_matches_golden(name, _trace_projection(builder()))


def test_tlspec_roundtrip_and_manifest_goldens(tmp_path: Path) -> None:
    """Portable bundle round-trip and manifest projections match goldens."""

    trace = _default_trace()
    bundle_path = tmp_path / "default_trace.tlspec"
    tl.save(trace, bundle_path)
    loaded = cast(Trace, tl.load(bundle_path))

    _assert_projection_matches_golden(
        "tlspec_roundtrip_digest",
        {
            "source": _trace_projection(trace),
            "loaded": _trace_projection(loaded),
        },
    )
    _assert_projection_matches_golden(
        "tlspec_manifest_projection",
        _manifest_projection(bundle_path),
    )


def test_field_order_and_dataframe_golden() -> None:
    """FIELD_ORDER, portable state, and dataframe projections match goldens."""

    _assert_projection_matches_golden(
        "field_order_dataframe_digest",
        _dataframe_projection(_default_trace()),
    )


def test_public_accessor_golden() -> None:
    """Public accessors and grad-adjacent op surfaces match goldens."""

    _assert_projection_matches_golden(
        "public_accessors_digest",
        _public_accessor_projection(_backward_ready_trace()),
    )


def test_parent_edge_mutation_fails_gate() -> None:
    """A deliberate parent-edge mutation changes the trace digest."""

    _skip_during_golden_update()
    projection = _trace_projection(_default_trace())
    mutated = copy.deepcopy(projection)
    mutated["ops"][2]["parents"] = []
    with pytest.raises(AssertionError):
        _assert_projection_matches_golden("default_trace_digest", mutated)


def test_save_policy_mutation_fails_gate() -> None:
    """A deliberate saved-flag mutation changes the trace digest."""

    _skip_during_golden_update()
    projection = _trace_projection(_selective_trace())
    mutated = copy.deepcopy(projection)
    mutated["ops"][1]["has_saved_activation"] = not mutated["ops"][1]["has_saved_activation"]
    with pytest.raises(AssertionError):
        _assert_projection_matches_golden("selective_trace_digest", mutated)


def test_serialization_mutation_fails_gate(tmp_path: Path) -> None:
    """A deliberate manifest mutation changes the manifest digest."""

    _skip_during_golden_update()
    trace = _default_trace()
    bundle_path = tmp_path / "serialization_mutation.tlspec"
    tl.save(trace, bundle_path)
    projection = _manifest_projection(bundle_path)
    projection["schema_version"] = 999
    with pytest.raises(AssertionError):
        _assert_projection_matches_golden("tlspec_manifest_projection", projection)


def test_saved_payload_deletion_fails_gate(tmp_path: Path) -> None:
    """Deleting a saved payload makes bundle loading fail."""

    _skip_during_golden_update()
    trace = _default_trace()
    bundle_path = tmp_path / "payload_deletion.tlspec"
    tl.save(trace, bundle_path)
    blob_dir = bundle_path / "blobs"
    first_blob = next(blob_dir.iterdir())
    first_blob.unlink()

    with pytest.raises(Exception, match="No such file|not found|missing|does not exist"):
        tl.load(bundle_path)


def test_function_param_deletion_fails_gate() -> None:
    """Deleting a public parameter surface changes the accessor digest."""

    _skip_during_golden_update()
    projection = _public_accessor_projection(_backward_ready_trace())
    mutated = copy.deepcopy(projection)
    del mutated["params"]["fc1.weight"]
    with pytest.raises(AssertionError):
        _assert_projection_matches_golden("public_accessors_digest", mutated)


def test_bundle_copy_preserves_manifest_gate(tmp_path: Path) -> None:
    """A copied bundle still projects to the same manifest golden."""

    trace = _default_trace()
    source_path = tmp_path / "source.tlspec"
    copied_path = tmp_path / "copied.tlspec"
    tl.save(trace, source_path)
    shutil.copytree(source_path, copied_path)
    _assert_projection_matches_golden(
        "tlspec_manifest_projection", _manifest_projection(copied_path)
    )
