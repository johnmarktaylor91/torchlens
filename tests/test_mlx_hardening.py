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
from torchlens.backends import (  # noqa: E402
    BackendRuntimeCompatibilityError,
    BackendUnsupportedError,
    get_backend_spec,
)
from torchlens.backends.mlx import GradOptions  # noqa: E402
import torchlens.backends.mlx.backend as mlx_backend  # noqa: E402
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


class TinyGradLinear(nn.Module):
    """Small MLX module used by derived-gradient tests."""

    def __init__(self) -> None:
        """Initialize the projection layer."""

        super().__init__()
        self.proj = nn.Linear(3, 2)

    def __call__(self, x: mx.array) -> mx.array:
        """Run a single linear projection."""

        return self.proj(x)


class DivergentGradLinear(nn.Module):
    """MLX module whose second raw output intentionally diverges."""

    def __init__(self) -> None:
        """Initialize state used to trip the derived-gradient honesty guard."""

        super().__init__()
        self.proj = nn.Linear(3, 2)
        self.calls = 0

    def __call__(self, x: mx.array) -> mx.array:
        """Return a call-count-dependent output."""

        self.calls += 1
        return self.proj(x) + self.calls


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


def _assert_mlx_array_equal(
    actual: mx.array,
    expected: tuple[np.ndarray, tuple[int, ...], str],
) -> None:
    """Assert that an MLX array matches expected value, shape, and dtype metadata."""

    expected_value, expected_shape, expected_dtype = expected
    assert isinstance(actual, mx.array)
    assert tuple(actual.shape) == expected_shape
    assert str(actual.dtype) == expected_dtype
    np.testing.assert_allclose(np.asarray(actual), expected_value)


def _always_true(_op: Any) -> bool:
    """Return true for ``tl.where`` rejection tests."""

    return True


def _grad_loss(output: mx.array) -> mx.array:
    """Return a scalar loss for MLX derived-gradient tests."""

    return mx.sum(output * output)


def _mlx_intermediate_loss(x: mx.array) -> mx.array:
    """Return a scalar loss with MLX eager wrapper-visible intermediates."""

    hidden = mx.multiply(x, x)
    shifted = mx.add(hidden, 3.0)
    return mx.sum(mx.multiply(shifted, shifted))


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
def test_mlx_derived_grads_match_value_and_grad_oracle() -> None:
    """MLX leaf derived gradients should match a direct MLX AD oracle."""

    model = TinyGradLinear()
    x = mx.array([[1.0, -2.0, 0.5], [0.25, 0.75, -1.5]], dtype=mx.float32)
    params = model.parameters()
    trace = tl.trace(
        model,
        x,
        backend="mlx",
        grad_options=GradOptions(params=params, loss_fn=_grad_loss, input_grad_argnums=(0,)),
    )

    def value_fn(test_params: dict[str, Any], test_x: mx.array) -> mx.array:
        """Return direct oracle loss."""

        model.update(test_params)
        return _grad_loss(model(test_x))

    expected_param_grads, expected_x_grad = mx.grad(value_fn, argnums=(0, 1))(params, x)
    mx.eval(
        expected_x_grad,
        expected_param_grads["proj"]["weight"],
        expected_param_grads["proj"]["bias"],
    )

    assert set(trace.derived_grads.keys()) == {
        "params.proj.bias",
        "params.proj.weight",
        "inputs.0",
    }
    np.testing.assert_allclose(
        np.asarray(trace.derived_grads["params.proj.weight"].grad),
        np.asarray(expected_param_grads["proj"]["weight"]),
    )
    np.testing.assert_allclose(
        np.asarray(trace.derived_grads["params.proj.bias"].grad),
        np.asarray(expected_param_grads["proj"]["bias"]),
    )
    np.testing.assert_allclose(
        np.asarray(trace.derived_grads["inputs.0"].grad),
        np.asarray(expected_x_grad),
    )
    assert trace.derived_grads["inputs.0"].provenance["mechanism"] == "mlx_value_and_grad"
    assert trace.params["proj.weight"].grad is trace.derived_grads["params.proj.weight"].grad
    assert trace.params["proj.bias"].grad is trace.derived_grads["params.proj.bias"].grad
    assert len(trace.intermediate_derived_grads) == 0


@pytest.mark.optional
def test_mlx_intermediate_derived_grads_match_direct_reference() -> None:
    """MLX T1 should expose exact saved-op gradients confirmed by a direct reference."""

    x = mx.array([1.5, -2.0], dtype=mx.float32)
    trace = tl.trace(
        _mlx_intermediate_loss,
        x,
        backend="mlx",
        grad_options=GradOptions(input_grad_argnums=(0,), intermediate_grads=True),
    )
    hidden_op = next(
        op for op in trace.layer_list if op.func_name == "multiply" and op.shape == (2,)
    )

    def suffix(boundary: mx.array) -> mx.array:
        """Return the scalar suffix loss from the hidden boundary."""

        shifted = mx.add(boundary, 3.0)
        return mx.sum(mx.multiply(shifted, shifted))

    expected = mx.grad(suffix)(hidden_op.out)
    mx.eval(expected, hidden_op.derived_grad)

    assert set(trace.derived_grads.keys()) == {"inputs.0"}
    assert hidden_op.label in trace.intermediate_derived_grads
    np.testing.assert_allclose(np.asarray(hidden_op.derived_grad), np.asarray(expected))
    record = trace.intermediate_derived_grads[hidden_op.label]
    assert record.provenance["mechanism"] == "mlx_custom_vjp_tap_value_and_grad"
    assert record.provenance["status"] == "exact"
    assert record.provenance["oracle"] == (
        "producer_custom_vjp+boundary_replacement_grad+perturbation"
    )


@pytest.mark.optional
def test_mlx_intermediate_derived_grads_perturbation_reference_oracle() -> None:
    """The MLX T1 oracle should catch boundary-sensitive gradients."""

    x = mx.array([1.5, -2.0], dtype=mx.float32)
    hidden = mx.multiply(x, x)

    def suffix(boundary: mx.array) -> mx.array:
        """Return a nonlinear scalar suffix from the boundary."""

        shifted = mx.add(boundary, 3.0)
        return mx.sum(mx.multiply(shifted, shifted))

    base_grad = mx.grad(suffix)(hidden)
    perturbed_grad = mx.grad(suffix)(hidden + 0.25)
    mx.eval(base_grad, perturbed_grad)
    assert not np.allclose(np.asarray(base_grad), np.asarray(perturbed_grad))

    trace = tl.trace(
        _mlx_intermediate_loss,
        x,
        backend="mlx",
        grad_options=GradOptions(input_grad_argnums=(0,), intermediate_grads=True),
    )
    hidden_op = next(
        op for op in trace.layer_list if op.func_name == "multiply" and op.shape == (2,)
    )

    np.testing.assert_allclose(np.asarray(hidden_op.derived_grad), np.asarray(base_grad))


@pytest.mark.optional
def test_mlx_intermediate_derived_grads_duplicate_signature_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicate grouped signatures should be skipped rather than guessed."""

    original_grouping = mlx_backend._mlx_trace_intermediate_signatures

    def duplicate_first_signature(ops: Any) -> dict[Any, list[Any]]:
        """Return grouped signatures with one deliberate duplicate."""

        grouped = original_grouping(ops)
        for matches in grouped.values():
            if matches:
                matches.append(matches[0])
                break
        return grouped

    monkeypatch.setattr(
        mlx_backend, "_mlx_trace_intermediate_signatures", duplicate_first_signature
    )
    trace = tl.trace(
        _mlx_intermediate_loss,
        mx.array([1.5, -2.0], dtype=mx.float32),
        backend="mlx",
        grad_options=GradOptions(input_grad_argnums=(0,), intermediate_grads=True),
    )

    first_multiply = next(op for op in trace.layer_list if op.func_name == "multiply")

    assert first_multiply.label not in trace.intermediate_derived_grads
    assert first_multiply.derived_grad is None


@pytest.mark.optional
def test_mlx_intermediate_derived_grads_do_not_pollute_module_call_counts() -> None:
    """The MLX tap observer should not double-count nested module calls."""

    model = TinyGradLinear()
    x = mx.array([[1.0, -2.0, 0.5], [0.25, 0.75, -1.5]], dtype=mx.float32)
    trace = tl.trace(
        model,
        x,
        backend="mlx",
        grad_options=GradOptions(
            loss_fn=_grad_loss, input_grad_argnums=(0,), intermediate_grads=True
        ),
    )

    assert trace.modules["proj"].num_calls == 1
    assert set(trace.module_calls.keys()) == {"self:1", "proj:1"}
    assert trace.module_calls["proj:1"].num_ops >= 1


@pytest.mark.optional
def test_mlx_intermediate_derived_grads_cap_hard_fails() -> None:
    """Explicit MLX T1 should hard-fail when saved boundaries exceed the cap."""

    with pytest.raises(BackendUnsupportedError, match="capped"):
        tl.trace(
            _mlx_intermediate_loss,
            mx.array([1.5, -2.0], dtype=mx.float32),
            backend="mlx",
            grad_options=GradOptions(
                input_grad_argnums=(0,),
                intermediate_grads=True,
                max_intermediate_grads=1,
            ),
        )


@pytest.mark.optional
def test_mlx_intermediate_derived_grads_oracle_fail_skips_public_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Oracle-failed MLX T1 candidates should not reach ``op.derived_grad``."""

    def always_reject(*args: Any, **kwargs: Any) -> bool:
        """Reject every candidate in the patched oracle."""

        return False

    monkeypatch.setattr(mlx_backend, "_mlx_intermediate_oracle_passes", always_reject)
    trace = tl.trace(
        _mlx_intermediate_loss,
        mx.array([1.5, -2.0], dtype=mx.float32),
        backend="mlx",
        grad_options=GradOptions(input_grad_argnums=(0,), intermediate_grads=True),
    )

    assert set(trace.derived_grads.keys()) == {"inputs.0"}
    assert len(trace.intermediate_derived_grads) == 0
    assert all(op.derived_grad is None for op in trace.layer_list)


@pytest.mark.optional
def test_mlx_intermediate_derived_grads_are_not_backward_capture() -> None:
    """MLX T1 should still keep true backward op-gradient surfaces closed."""

    trace = tl.trace(
        _mlx_intermediate_loss,
        mx.array([1.5, -2.0], dtype=mx.float32),
        backend="mlx",
        grad_options=GradOptions(input_grad_argnums=(0,), intermediate_grads=True),
    )

    with pytest.raises(ValueError, match="trace\\.derived_grads"):
        _ = trace[0].grads


@pytest.mark.optional
def test_mlx_derived_grads_allow_scalar_output_without_loss_fn() -> None:
    """MLX derived gradients should allow scalar raw outputs without a loss function."""

    def scalar_model(x: mx.array) -> mx.array:
        """Return a scalar MLX output."""

        return mx.sum(x * x)

    x = mx.array([1.0, -2.0, 3.0], dtype=mx.float32)
    trace = tl.trace(
        scalar_model,
        x,
        backend="mlx",
        grad_options=GradOptions(input_grad_argnums=(0,)),
    )

    np.testing.assert_allclose(np.asarray(trace.derived_grads["inputs.0"].grad), [2.0, -4.0, 6.0])


@pytest.mark.optional
def test_mlx_derived_grads_reject_raw_output_divergence() -> None:
    """MLX derived gradients should refuse records when the AD rerun output diverges."""

    with pytest.raises(ValueError, match="raw output diverged"):
        tl.trace(
            DivergentGradLinear(),
            mx.ones((1, 3), dtype=mx.float32),
            backend="mlx",
            grad_options=GradOptions(loss_fn=_grad_loss, input_grad_argnums=(0,)),
        )


@pytest.mark.optional
def test_mlx_derived_grads_are_not_backward_capture() -> None:
    """MLX derived gradients should not unlock true backward surfaces."""

    trace = tl.trace(
        TinyGradLinear(),
        mx.ones((1, 3), dtype=mx.float32),
        backend="mlx",
        grad_options=GradOptions(loss_fn=_grad_loss, input_grad_argnums=(0,)),
    )

    with pytest.raises(BackendUnsupportedError, match="trace\\.derived_grads"):
        trace.log_backward(trace[trace.output_layers[0]].out)
    with pytest.raises(ValueError, match="trace\\.derived_grads"):
        _ = trace.backward_passes
    with pytest.raises(ValueError, match="trace\\.derived_grads"):
        _ = trace.saved_grad_ops
    with pytest.raises(ValueError, match="trace\\.derived_grads"):
        _ = trace[0].grads


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
        op.layer_label: (np.asarray(op.out), tuple(op.out.shape), str(op.out.dtype))
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
        _assert_mlx_array_equal(op.out, expected_outs[op.layer_label])

    status = loaded.validate_forward_pass([])
    assert status is loaded.validation_replay_status
    assert status.state == "unavailable"
    assert status.available is False
    assert status.reason == "loaded_trace_runtime_capture_stripped"
    assert status.payload_load_status == "loaded_device_best_effort"
    with pytest.raises(TypeError, match="not a boolean"):
        bool(status)


@pytest.mark.optional
def test_mlx_save_load_lazy_ref_materializes_public_payloads(tmp_path: Path) -> None:
    """MLX lazy public loads should defer array decode until ``out_ref.materialize``."""

    trace = tl.trace(TinyMLP(), _tiny_mlp_input())
    expected_outs = {
        op.layer_label: (np.asarray(op.out), tuple(op.out.shape), str(op.out.dtype))
        for op in trace.layer_list
        if op.has_saved_activation and op.out is not None
    }
    bundle_path = tmp_path / "mlx-lazy-materialized.tlspec"

    tl.save(trace, bundle_path)
    loaded = tl.load(bundle_path, lazy=True)
    manifest = tl.io.inspect_tlspec(bundle_path)

    assert loaded.backend == "mlx"
    assert getattr(loaded, "payload_load_status") == "loaded_device_best_effort"
    _assert_mlx_materialized_manifest(manifest, {"out"})
    loaded_saved_ops = [op for op in loaded.layer_list if op.has_saved_activation]
    assert loaded_saved_ops
    for op in loaded_saved_ops:
        assert op.out is None
        assert op.out_ref is not None
        assert op.out_ref.logical_backend == "mlx"
        materialized = op.out_ref.materialize()
        _assert_mlx_array_equal(materialized, expected_outs[op.layer_label])


@pytest.mark.optional
def test_mlx_save_load_missing_runtime_falls_back_to_audit_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing MLX runtime during load should preserve metadata and lazy refs."""

    trace = tl.trace(TinyMLP(), _tiny_mlp_input())
    bundle_path = tmp_path / "mlx-missing-runtime.tlspec"

    tl.save(trace, bundle_path)

    def _missing_mlx_runtime() -> Any:
        """Pretend ``mlx.core`` is unavailable during payload decode."""

        raise BackendRuntimeCompatibilityError("mlx-runtime-missing: mlx runtime unavailable")

    monkeypatch.setattr("torchlens._io.payload_codec._import_mlx_core", _missing_mlx_runtime)

    loaded = tl.load(bundle_path)
    loaded_saved_ops = [op for op in loaded.layer_list if op.has_saved_activation]

    assert loaded.backend == "mlx"
    assert getattr(loaded, "payload_load_status") == "audit_only_missing_runtime"
    assert loaded_saved_ops
    for op in loaded_saved_ops:
        assert op.out is None
        assert op.out_ref is not None
        assert op.shape is not None
        assert op.dtype_ref is not None
        with pytest.raises(
            BackendRuntimeCompatibilityError,
            match="mlx-runtime-missing",
        ):
            op.out_ref.materialize()


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
        (tl.output(0), "unsupported selector kind 'output'"),
        (tl.where(_always_true, name_hint="always"), "tl.where"),
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
    assert spec.capabilities.intermediate_derived_grads is True
    assert spec.capabilities.module_identity_modes == ("function_root", "object_module")
    assert spec.capabilities.trace_options == ("module_identity_mode", "grad_options")
    assert spec.serialization_policy.payload_policy == "array_payloads"
    assert spec.serialization_policy.body_format == "safetensors"
    assert capabilities.supports_payload_materialization is True
    assert capabilities.supports_intermediate_derived_grads is True
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
