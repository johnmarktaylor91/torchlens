"""Hardening tests for the technical-preview MLX backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

mlx = pytest.importorskip("mlx")
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

import torchlens as tl  # noqa: E402
from torchlens.backends import BackendUnsupportedError, get_backend_spec  # noqa: E402
import torchlens.backends.mlx.capabilities as capabilities  # noqa: E402


class TinyMLP(nn.Module):
    """Small MLX MLP used by backend hardening tests."""

    def __init__(self) -> None:
        """Initialize the MLP layers."""

        super().__init__()
        self.l1 = nn.Linear(4, 8)
        self.l2 = nn.Linear(8, 2)

    def __call__(self, x: mx.array) -> mx.array:
        """Run the MLP forward pass."""

        hidden = nn.relu(self.l1(x))
        return self.l2(hidden)


class TinyCNN(nn.Module):
    """Small MLX CNN used to verify Conv2d capture."""

    def __init__(self) -> None:
        """Initialize convolution and projection layers."""

        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.proj = nn.Linear(4, 2)

    def __call__(self, x: mx.array) -> mx.array:
        """Run a tiny CNN forward pass."""

        hidden = nn.relu(self.conv(x))
        pooled = mx.mean(hidden, axis=(1, 2))
        return self.proj(pooled)


class TinyNorm(nn.Module):
    """Small MLX model used to verify normalization capture."""

    def __init__(self) -> None:
        """Initialize normalization and projection layers."""

        super().__init__()
        self.norm = nn.LayerNorm(4)
        self.proj = nn.Linear(4, 2)

    def __call__(self, x: mx.array) -> mx.array:
        """Run a normalized forward pass."""

        return self.proj(self.norm(x))


class TinyAttention(nn.Module):
    """Small MLX model used to verify attention capture."""

    def __init__(self) -> None:
        """Initialize attention and projection layers."""

        super().__init__()
        self.attn = nn.MultiHeadAttention(dims=8, num_heads=2)
        self.proj = nn.Linear(8, 4)

    def __call__(self, x: mx.array) -> mx.array:
        """Run a self-attention forward pass."""

        return self.proj(self.attn(x, x, x))


def _tiny_mlp_input() -> mx.array:
    """Return a deterministic-shape MLX input array for hardening tests."""

    return mx.random.normal((2, 4))


def _assert_captured(trace: tl.Trace, expected_fragment: str) -> None:
    """Assert that an MLX trace captured an operation label fragment."""

    assert any(expected_fragment in label.lower() for label in trace.layer_labels)


def _assert_static_save_summary(trace: tl.Trace, expected_func_names: set[str]) -> None:
    """Assert that MLX static save filtering preserves metadata and refreshes counters."""

    saved_ops = [op for op in trace.layer_list if op.has_saved_activation]
    unsaved_ops = [op for op in trace.layer_list if not op.has_saved_activation]

    assert saved_ops
    assert {op.func_name for op in saved_ops} == expected_func_names
    assert unsaved_ops
    assert all(op.out is not None for op in saved_ops)
    assert all(op.out is None and op.out_ref is None for op in unsaved_ops)
    assert all(op.shape is not None and op.dtype_ref is not None for op in unsaved_ops)
    assert trace.num_saved_ops == len(saved_ops)
    assert len(trace.saved_ops) == len(saved_ops)
    assert int(trace.saved_activation_memory) == sum(
        int(op.activation_memory or 0) for op in saved_ops
    )


def _assert_mlx_materialized_manifest(manifest: dict[str, Any], expected_kinds: set[str]) -> None:
    """Assert that a public MLX save wrote materialized codec body entries."""

    assert manifest["body_format"] == "safetensors"
    assert manifest["payload_policy"] == {
        "policy": "array_payloads",
        "materialization_supported": True,
        "payload_kinds": sorted(expected_kinds),
    }
    assert manifest["body_index"]
    assert {entry["intended_use"] for entry in manifest["body_index"]} == expected_kinds
    assert all(entry["logical_backend"] == "mlx" for entry in manifest["body_index"])
    assert all(entry["codec"] == "numpy_safetensors_v1" for entry in manifest["body_index"])
    assert all(entry["filename"].endswith(".safetensors") for entry in manifest["body_index"])
    assert not any(
        record["reason"] == "mlx_array_audit_null" for record in manifest["unsupported_tensors"]
    )


def _always_true(_op: Any) -> bool:
    """Return true for ``tl.where`` rejection tests."""

    return True


@pytest.mark.optional
def test_mlx_intervention_ready_raises() -> None:
    """MLX capture rejects intervention metadata requests explicitly."""

    with pytest.raises(BackendUnsupportedError, match="intervention_ready"):
        tl.trace(TinyMLP(), _tiny_mlp_input(), intervention_ready=True)


@pytest.mark.optional
def test_mlx_save_grads_raises() -> None:
    """MLX capture rejects backward-gradient capture explicitly."""

    with pytest.raises(BackendUnsupportedError, match="backward capture"):
        tl.trace(TinyMLP(), _tiny_mlp_input(), save_grads=True)


@pytest.mark.optional
def test_mlx_hooks_raise() -> None:
    """MLX capture rejects pre-attached live hook plans explicitly."""

    hooks: list[dict[str, Any]] = [{"target": "linear_1_1", "action": lambda x: x}]
    with pytest.raises(BackendUnsupportedError, match="hooks"):
        tl.trace(TinyMLP(), _tiny_mlp_input(), hooks=hooks)


@pytest.mark.optional
def test_mlx_trace_draw_smokes(tmp_path: Path) -> None:
    """MLX traces draw without dangling synthetic input-parent labels."""

    trace = tl.trace(TinyMLP(), _tiny_mlp_input())
    rendered = trace.draw(
        vis_outpath=str(tmp_path / "mlx_trace"),
        vis_fileformat="pdf",
        vis_save_only=True,
    )

    assert rendered is not None
    assert (tmp_path / "mlx_trace.pdf").exists()


@pytest.mark.optional
def test_mlx_parents_resolve() -> None:
    """Every MLX parent label resolves to a captured layer."""

    trace = tl.trace(TinyMLP(), _tiny_mlp_input())

    for op in trace.layer_list:
        for parent in op.parents:
            assert parent in trace.layer_dict_all_keys


@pytest.mark.optional
def test_mlx_repeated_trace_no_wrapper_leak() -> None:
    """Repeated MLX captures install fresh wrappers and clean them up."""

    first = tl.trace(TinyMLP(), _tiny_mlp_input())
    second = tl.trace(TinyMLP(), _tiny_mlp_input())

    assert first.num_ops > 0
    assert second.num_ops > 0


@pytest.mark.optional
def test_mlx_save_load_materializes_public_payloads(tmp_path: Path) -> None:
    """MLX public saves now materialize array payloads through the MLX codec."""

    trace = tl.trace(TinyMLP(), _tiny_mlp_input())
    expected_outs = {
        op.layer_label: np.asarray(op.out)
        for op in trace.layer_list
        if op.has_saved_activation and op.out is not None
    }
    bundle_path = tmp_path / "mlx-materialized.tlspec"

    tl.save(trace, bundle_path)
    loaded = tl.load(bundle_path)
    manifest = tl.io.inspect_tlspec(bundle_path)

    assert loaded.backend == "mlx"
    assert getattr(loaded, "payload_load_status") == "loaded_device_best_effort"
    _assert_mlx_materialized_manifest(manifest, {"out"})
    loaded_saved_ops = [op for op in loaded.layer_list if op.has_saved_activation]
    assert loaded_saved_ops
    assert all(isinstance(op.out, mx.array) for op in loaded_saved_ops)
    for op in loaded_saved_ops:
        np.testing.assert_allclose(np.asarray(op.out), expected_outs[op.layer_label])

    status = loaded.validate_forward_pass([])
    assert status is loaded.validation_replay_status
    assert status.state == "unavailable"
    assert status.available is False
    assert status.reason == "loaded_trace_runtime_capture_stripped"
    assert status.payload_load_status == "loaded_device_best_effort"
    with pytest.raises(TypeError, match="not a boolean"):
        bool(status)


@pytest.mark.optional
def test_mlx_static_func_save_filters_public_payloads(tmp_path: Path) -> None:
    """MLX static ``tl.func`` save filters and materializes public payloads."""

    trace = tl.trace(TinyMLP(), _tiny_mlp_input(), save=tl.func("relu"))

    _assert_static_save_summary(trace, {"relu"})

    bundle_path = tmp_path / "mlx-selective-materialized.tlspec"
    tl.save(trace, bundle_path)
    loaded = tl.load(bundle_path)
    manifest = tl.io.inspect_tlspec(bundle_path)
    loaded_saved_ops = [op for op in loaded.layer_list if op.has_saved_activation]
    loaded_unsaved_ops = [op for op in loaded.layer_list if not op.has_saved_activation]

    _assert_mlx_materialized_manifest(manifest, {"out"})
    assert len(manifest["body_index"]) == trace.num_saved_ops
    assert loaded_saved_ops
    assert all(isinstance(op.out, mx.array) for op in loaded_saved_ops)
    assert all(op.out is None for op in loaded_unsaved_ops)


@pytest.mark.optional
def test_mlx_static_label_contains_and_composite_save_work() -> None:
    """MLX allows label, contains, and safe boolean composites for static save."""

    label_trace = tl.trace(TinyMLP(), _tiny_mlp_input(), save=tl.label("relu_1_3_raw:1"))
    contains_trace = tl.trace(TinyMLP(), _tiny_mlp_input(), save=tl.contains("relu"))
    composite_trace = tl.trace(
        TinyMLP(),
        _tiny_mlp_input(),
        save=tl.func("linear") & ~tl.contains("linear_2"),
    )

    _assert_static_save_summary(label_trace, {"relu"})
    _assert_static_save_summary(contains_trace, {"relu"})
    _assert_static_save_summary(composite_trace, {"linear"})
    assert list(composite_trace.saved_ops.keys()) == ["linear_1_2_raw:1"]


@pytest.mark.parametrize(
    ("selector", "pattern"),
    [
        (tl.module("l1"), "module hierarchy required, not yet on MLX"),
        (tl.in_module("l1"), "module hierarchy required, not yet on MLX"),
        (tl.output(0), "unsupported selector kind 'output'"),
        (tl.where(_always_true, name_hint="always"), "tl.where"),
        (tl.func("relu") & tl.in_module("l1"), "module hierarchy required, not yet on MLX"),
    ],
)
@pytest.mark.optional
def test_mlx_static_save_rejects_unsupported_selector_kinds(
    selector: Any,
    pattern: str,
) -> None:
    """MLX rejects selectors outside the static-label phase-A allowlist."""

    with pytest.raises(BackendUnsupportedError, match=pattern):
        tl.trace(TinyMLP(), _tiny_mlp_input(), save=selector)


@pytest.mark.optional
def test_mlx_capability_flags_match_materialized_contract() -> None:
    """MLX capability exports should advertise materialized array payloads."""

    spec = get_backend_spec("mlx")

    assert spec.capabilities.payload_materialization is True
    assert spec.serialization_policy.payload_policy == "array_payloads"
    assert spec.serialization_policy.body_format == "safetensors"
    assert capabilities.supports_payload_materialization is True
    assert capabilities.payload_policy == "array_payloads"


@pytest.mark.optional
def test_mlx_cnn_captures_conv2d() -> None:
    """MLX Conv2d calls are wrapped and appear in the captured trace."""

    trace = tl.trace(TinyCNN(), mx.random.normal((1, 8, 8, 3)))

    _assert_captured(trace, "conv2d")


@pytest.mark.optional
def test_mlx_layernorm_wraps_and_captures() -> None:
    """MLX LayerNorm calls are wrapped and appear in the captured trace."""

    trace = tl.trace(TinyNorm(), _tiny_mlp_input())

    _assert_captured(trace, "layernorm")


@pytest.mark.optional
def test_mlx_attention_wraps_and_captures() -> None:
    """MLX attention calls are wrapped and appear in the captured trace."""

    trace = tl.trace(TinyAttention(), mx.random.normal((1, 3, 8)))

    _assert_captured(trace, "multiheadattention")
