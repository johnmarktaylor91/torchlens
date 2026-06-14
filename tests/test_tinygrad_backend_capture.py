"""tinygrad backend capture tests."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import torchlens as tl
from torchlens._io.bundle import _build_manifest, _write_payload_blob
from torchlens._io.payload_codec import get_payload_codec
from torchlens._io.scrub import scrub_for_save
from torchlens._io.tensor_policy import FailReason, Ok
from torchlens._io.tlspec import _TlSpecWriter
from torchlens.backends import (
    BackendRuntimeCompatibilityError,
    BackendUnsupportedError,
    get_backend_spec,
)
from torchlens.backends.tinygrad import GradOptions, TinygradBackend, capabilities

tinygrad = pytest.importorskip("tinygrad")
Tensor = pytest.importorskip("tinygrad").Tensor


pytestmark = pytest.mark.backend_tinygrad


def _tiny_block(x: Any) -> Any:
    """Return a small tinygrad expression.

    Parameters
    ----------
    x
        tinygrad Tensor input.

    Returns
    -------
    Any
        tinygrad Tensor output.
    """

    return ((x + 1.0).relu() * 2.0).sum()


def _tiny_square_loss(x: Any) -> Any:
    """Return a scalar tinygrad square loss.

    Parameters
    ----------
    x
        tinygrad Tensor input.

    Returns
    -------
    Any
        Scalar tinygrad Tensor output.
    """

    return (x * x).sum()


def _tiny_vector_output(x: Any) -> Any:
    """Return a non-scalar tinygrad output.

    Parameters
    ----------
    x
        tinygrad Tensor input.

    Returns
    -------
    Any
        Vector tinygrad Tensor output.
    """

    return x * x


def _tiny_vector_loss(output: Any) -> Any:
    """Return a scalar loss for a tinygrad vector output.

    Parameters
    ----------
    output
        tinygrad vector output.

    Returns
    -------
    Any
        Scalar tinygrad Tensor loss.
    """

    return output.sum()


def _tiny_intermediate_loss(x: Any) -> Any:
    """Return a scalar tinygrad loss with retained intermediates.

    Parameters
    ----------
    x
        tinygrad Tensor input.

    Returns
    -------
    Any
        Scalar tinygrad Tensor loss.
    """

    hidden = x * x
    activated = (hidden + 1.0).relu()
    scaled = activated * 3.0
    return scaled.sum()


def test_tinygrad_payload_codec_encodes_numpy_manifest_fields() -> None:
    """tinygrad payload codec should expose logical and transport metadata."""

    codec = get_payload_codec("tinygrad")
    value = Tensor([1.0, 2.0, 3.0]).realize()

    assert codec.can_encode(value)
    assert isinstance(codec.validate_for_save(value, strict=True), Ok)
    encoded = codec.to_numpy(value)
    np.testing.assert_array_equal(encoded.array, value.numpy())
    assert "float" in encoded.logical_dtype
    assert encoded.logical_device

    fields = codec.manifest_fields(value, encoded)
    assert fields["logical_backend"] == "tinygrad"
    assert fields["codec"] == "numpy_safetensors_v1"
    assert fields["logical_dtype"] == encoded.logical_dtype
    assert fields["logical_device"] == encoded.logical_device
    assert fields["transport_backend"] == "safetensors.torch"
    assert fields["transport_dtype"] == "float32"
    assert isinstance(fields["codec_metadata"], dict)

    restored = codec.from_numpy(encoded.array, fields, map_location=None, strict_runtime=True)
    np.testing.assert_array_equal(restored.numpy(), encoded.array)


def test_tinygrad_codec_fixture_loads_eager_payloads_as_tensors(tmp_path: Path) -> None:
    """Direct-writer tinygrad codec fixtures should load eager outs as tinygrad tensors."""

    path = tmp_path / "tinygrad_codec_roundtrip.tlspec"
    _, expected, expected_dtype, expected_device = _write_tinygrad_codec_fixture(path)

    loaded = tl.load(path)
    loaded_op = _first_saved_op(loaded)

    assert loaded.backend == "tinygrad"
    assert isinstance(loaded_op.out, Tensor)
    assert str(loaded_op.out.dtype) == expected_dtype
    assert str(loaded_op.out.device) == expected_device
    assert getattr(loaded, "payload_load_status") == "loaded_device_best_effort"
    np.testing.assert_allclose(loaded_op.out.numpy(), expected)


def test_tinygrad_codec_fixture_loads_lazy_refs_as_tensors(tmp_path: Path) -> None:
    """Lazy tinygrad codec fixtures should defer payload decode until materialization."""

    path = tmp_path / "tinygrad_codec_lazy.tlspec"
    _, expected, expected_dtype, expected_device = _write_tinygrad_codec_fixture(path)

    loaded = tl.load(path, lazy=True)
    loaded_op = _first_saved_op(loaded)

    assert loaded_op.out is None
    assert loaded_op.out_ref is not None
    assert loaded_op.out_ref.logical_backend == "tinygrad"
    materialized = loaded_op.out_ref.materialize()
    assert isinstance(materialized, Tensor)
    assert str(materialized.dtype) == expected_dtype
    assert str(materialized.device) == expected_device
    np.testing.assert_allclose(materialized.numpy(), expected)


def test_tinygrad_payload_codec_rejects_unresolved_load_dtype() -> None:
    """tinygrad load should fail closed for unresolvable dtype strings."""

    codec = get_payload_codec("tinygrad")
    with pytest.raises(BackendRuntimeCompatibilityError, match="not-a-real-dtype"):
        codec.from_numpy(
            np.asarray([1.0], dtype=np.float32),
            {"logical_dtype": "not-a-real-dtype", "logical_device": "CPU"},
            map_location=None,
            strict_runtime=True,
        )


def test_tinygrad_payload_codec_strict_rejects_object_dtype_arrays() -> None:
    """tinygrad codec strict validation should reject object dtype payloads."""

    class _FakeObjectTinygradTensor:
        """Minimal tinygrad-shaped value for object dtype rejection."""

        __module__ = "tinygrad.tensor"
        dtype = "object"
        device = "CPU"

        def numpy(self) -> np.ndarray:
            """Return an unsupported object dtype array."""

            return np.asarray([object()], dtype=object)

    decision = get_payload_codec("tinygrad").validate_for_save(
        _FakeObjectTinygradTensor(),
        strict=True,
    )

    assert isinstance(decision, FailReason)
    assert "object" in decision.text


def _multi_output(x: Any) -> tuple[Any, Any]:
    """Return two tinygrad outputs.

    Parameters
    ----------
    x
        tinygrad Tensor input.

    Returns
    -------
    tuple[Any, Any]
        Pair of tinygrad Tensor outputs.
    """

    return x + 1.0, x * 3.0


def _wide_corpus(x: Any) -> tuple[Any, Any]:
    """Return a mixed tinygrad corpus with conv, matmul, reduce, and elementwise ops.

    Parameters
    ----------
    x
        tinygrad Tensor input.

    Returns
    -------
    tuple[Any, Any]
        Scalar and vector tinygrad outputs.
    """

    image = x.reshape(1, 1, 3, 3)
    kernel = Tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    conv = image.conv2d(kernel).relu().sum()
    matrix = x[:4].reshape(2, 2)
    matmul = matrix @ Tensor([[1.0], [2.0]])
    reduced = (matmul + 1.0).sum(axis=0)
    return conv + reduced.sum(), (x.relu() * 0.5) + 2.0


def _realize_inside(x: Any) -> Any:
    """Return an expression that explicitly realizes mid-capture.

    Parameters
    ----------
    x
        tinygrad Tensor input.

    Returns
    -------
    Any
        tinygrad Tensor output.
    """

    return (x + 1.0).realize() * 2.0


def _assign_inside(x: Any) -> Any:
    """Return an expression after mutating an input tensor.

    Parameters
    ----------
    x
        tinygrad Tensor input.

    Returns
    -------
    Any
        tinygrad Tensor output.
    """

    x.assign(x + 1.0)
    return x * 2.0


def _replace_inside(x: Any) -> Any:
    """Return an expression after replacing an input tensor's backing UOp.

    Parameters
    ----------
    x
        tinygrad Tensor input.

    Returns
    -------
    Any
        tinygrad Tensor output.
    """

    x.replace(x + 1.0)
    return x * 2.0


def _setitem_inside(x: Any) -> Any:
    """Return an expression after mutating an input tensor through setitem.

    Parameters
    ----------
    x
        tinygrad Tensor input.

    Returns
    -------
    Any
        tinygrad Tensor output.
    """

    x[0] = x[0] + 1.0
    return x * 2.0


def _trace(**kwargs: Any) -> Any:
    """Trace the shared tinygrad scalar block.

    Parameters
    ----------
    **kwargs
        Public trace keyword overrides.

    Returns
    -------
    Any
        Captured tinygrad trace.
    """

    return tl.trace(_tiny_block, Tensor([1.0, -2.0, 3.0]), backend="tinygrad", **kwargs)


def _write_tinygrad_codec_fixture(path: Path) -> tuple[Any, np.ndarray, str, str]:
    """Write a materialized tinygrad bundle through the private codec writer path.

    Parameters
    ----------
    path
        Destination tlspec directory.

    Returns
    -------
    tuple[Any, np.ndarray, str, str]
        Source trace, first saved payload values, dtype string, and device string.
    """

    trace = _trace()
    scrubbed_state, blob_specs, unsupported_tensors = scrub_for_save(
        trace,
        include_outs=True,
        include_grads=False,
        include_saved_args=False,
        include_rng_states=False,
        backend_name="tinygrad",
        payload_materialization=True,
    )
    if not blob_specs:
        raise AssertionError("tinygrad codec fixture expected at least one saved payload.")

    path.mkdir()
    (path / "blobs").mkdir()
    codec = get_payload_codec("tinygrad")
    tensor_entries = [
        _write_payload_blob(tmp_path=path, blob_spec=blob_spec, codec=codec)
        for blob_spec in blob_specs
    ]
    manifest = _build_manifest(
        trace=trace,
        tensor_entries=tensor_entries,
        unsupported_tensors=unsupported_tensors,
    )
    _TlSpecWriter.write_trace_manifest(
        path=path / "manifest.json",
        trace=trace,
        legacy_manifest=manifest,
        save_level="portable",
    )
    with (path / "metadata.pkl").open("wb") as handle:
        pickle.dump(scrubbed_state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    _mark_fixture_manifest_materialized(path)
    first_payload = blob_specs[0].value
    return (
        trace,
        np.asarray(first_payload.numpy()),
        str(first_payload.dtype),
        str(first_payload.device),
    )


def _mark_fixture_manifest_materialized(path: Path) -> None:
    """Patch a direct-writer fixture manifest to declare materialized payloads."""

    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["body_format"] = "safetensors"
    manifest["payload_policy"] = {
        "policy": "array_payloads",
        "materialization_supported": True,
        "payload_kinds": sorted({entry["kind"] for entry in manifest["tensors"]}),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _first_saved_op(trace: Any) -> Any:
    """Return the first op with a saved activation."""

    for op in trace.layer_list:
        if op.has_saved_activation:
            return op
    raise AssertionError("trace has no saved activation op")


def test_tinygrad_spec_registered() -> None:
    """Assert the built-in tinygrad spec exposes the G1 capability split."""

    spec = get_backend_spec("tinygrad")
    assert spec.capabilities.backward_capture is False
    assert spec.capabilities.validation_replay is True
    assert spec.capabilities.fastlog is False
    assert spec.capabilities.interventions is False
    assert spec.capabilities.payload_materialization is True
    assert spec.capabilities.module_identity_modes == ("function_root", "object_module")
    assert spec.serialization_policy.payload_policy == "array_payloads"
    assert spec.serialization_policy.body_format == "safetensors"
    assert capabilities.supports_backward_capture is False
    assert capabilities.supports_validation_replay is True
    assert capabilities.supports_fastlog is False
    assert capabilities.supports_intervention is False
    assert capabilities.supports_payload_materialization is True
    assert capabilities.module_identity_modes == ("function_root", "object_module")
    assert capabilities.payload_policy == "array_payloads"
    assert capabilities.live_payload_policy == "dev_python_realized_copy"
    assert capabilities.trace_options == ("module_identity_mode", "grad_options")


def test_tinygrad_forward_capture_from_uop_snapshots() -> None:
    """Capture a tinygrad forward graph from output UOp snapshots."""

    x = Tensor([1.0, -2.0, 3.0])
    trace = tl.trace(_tiny_block, x, backend="tinygrad")

    layer_types = {op.layer_type for op in trace.layer_list}
    assert trace.backend == "tinygrad"
    assert trace.module_identity_mode == "function_root"
    assert trace.param_source == "none"
    assert "input" in layer_types
    assert "add" in layer_types
    assert "where" in layer_types
    assert "mul" in layer_types
    assert {"add", "where", "mul"} <= layer_types
    assert trace.validate_forward_pass([_tiny_block(x)]) is True
    assert getattr(trace, "tinygrad_payload_policy") == "dev_python_realized_copy"
    assert trace.output_layers


def test_tinygrad_multi_output_marks_outputs() -> None:
    """Capture a multi-output tinygrad function and mark both output parents."""

    x = Tensor([1.0, 2.0])
    trace = tl.trace(_multi_output, x, backend="tinygrad")

    assert len(trace.output_layers) == 2
    assert all(trace.layer_dict_main_keys[label].is_output_parent for label in trace.output_layers)
    assert trace.validate_forward_pass(list(_multi_output(x))) is True


def test_tinygrad_wide_corpus_capture_and_validation() -> None:
    """Capture representative tinygrad conv, matmul, reduce, and elementwise ops."""

    x = Tensor([1.0, -2.0, 3.0, 4.0, -5.0, 6.0, 7.0, -8.0, 9.0])
    trace = tl.trace(_wide_corpus, x, backend="tinygrad")

    layer_types = {op.layer_type for op in trace.layer_list}
    assert {"mul", "reduce", "where"} <= layer_types
    assert {"permute", "shrink"} & layer_types
    assert len(trace.output_layers) == 2
    assert trace.validate_forward_pass(list(_wide_corpus(x))) is True


@pytest.mark.parametrize(
    ("fn", "pattern"),
    (
        (_realize_inside, "Tensor\\.realize|lazy tinygrad expression"),
        (_assign_inside, "Tensor\\.assign|lazy tinygrad expression"),
        (_replace_inside, "Tensor\\.replace|pure lazy tinygrad expression"),
        (_setitem_inside, "setitem input mutation|pure lazy tinygrad expression"),
    ),
)
def test_tinygrad_rejects_mid_capture_realize_and_mutation(
    fn: Any,
    pattern: str,
) -> None:
    """Reject tinygrad execution that can truncate lazy lineage during capture."""

    with pytest.raises(BackendUnsupportedError, match=pattern):
        tl.trace(fn, Tensor([1.0, 2.0]), backend="tinygrad")


def test_tinygrad_rejects_tinyjit_callable() -> None:
    """Reject TinyJit execution until the backend has a versioned JIT identity model."""

    tiny_jit = pytest.importorskip("tinygrad").TinyJit

    @tiny_jit
    def jit_block(x: Any) -> Any:
        """Return a tiny JIT expression.

        Parameters
        ----------
        x
            tinygrad Tensor input.

        Returns
        -------
        Any
            tinygrad Tensor output.
        """

        return (x + 1.0).realize()

    x = Tensor([1.0, 2.0])
    with pytest.raises(BackendUnsupportedError, match="realize|assign|TinyJit"):
        tl.trace(jit_block, x, backend="tinygrad")


@pytest.mark.parametrize(
    ("kwargs", "pattern"),
    (
        ({"save": tl.func("relu")}, "full-save only.*save-shaping"),
        ({"layers_to_save": ["relu"]}, "full-save only.*save shaping"),
        ({"lookback": 1}, "full-save only.*save-window"),
        ({"intervene": tl.when(tl.func("relu"), tl.zero_ablate())}, "full-save only"),
        ({"halt": tl.func("relu")}, "full-save only"),
        ({"save_grads": True}, "save_grads.*full-save forward capture"),
    ),
)
def test_tinygrad_save_shaping_rejected(kwargs: dict[str, Any], pattern: str) -> None:
    """Reject save shaping because tinygrad preview is full-save only."""

    with pytest.raises(BackendUnsupportedError, match=pattern):
        _trace(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "pattern"),
    (
        ({"jax_control_flow": "unroll"}, "control-flow unrolling.*not implemented"),
        ({"jax_max_control_flow_unroll": 4}, "control-flow unrolling.*not implemented"),
        ({"module_identity_mode": "pytree_module"}, "module_identity_mode must be"),
        ({"payload_policy": "full"}, "payload_policy.*not implemented"),
        ({"save_preview": True}, "save_preview.*not implemented"),
    ),
)
def test_tinygrad_rejects_declared_future_public_options(
    kwargs: dict[str, Any],
    pattern: str,
) -> None:
    """Declared public-option spine knobs should reject until tinygrad phases implement them."""

    with pytest.raises(BackendUnsupportedError, match=pattern):
        _trace(**kwargs)


def test_tinygrad_record_backend_rejected() -> None:
    """Sparse ``tl.record`` should stay torch-only for backend v1."""

    with pytest.raises(BackendUnsupportedError, match="torch-only.*tl\\.trace"):
        tl.record(_tiny_block, Tensor([1.0, 2.0]), backend="tinygrad")


def test_tinygrad_module_predicate_surfaces_rejected_for_function_root() -> None:
    """Module predicate capture should fail because tinygrad only has function_root."""

    with pytest.raises(BackendUnsupportedError, match="full-save only.*save-shaping"):
        _trace(save=tl.in_module("self"))


def test_tinygrad_backward_ready_rejected() -> None:
    """Reject true backward capture surfaces for tinygrad G1."""

    x = Tensor([1.0, 2.0])
    with pytest.raises(BackendUnsupportedError, match="backward_ready"):
        tl.trace(_tiny_block, x, backend="tinygrad", backward_ready=True)


def test_tinygrad_derived_grads_match_backward_oracle() -> None:
    """Derived input gradients should match direct tinygrad backward and .grad."""

    x = Tensor([1.0, -2.0, 3.0])
    trace = tl.trace(
        _tiny_square_loss,
        x,
        backend="tinygrad",
        grad_options=GradOptions(input_grad_argnums=(0,)),
    )
    oracle_x = Tensor([1.0, -2.0, 3.0])
    _tiny_square_loss(oracle_x).backward()

    assert set(trace.derived_grads.keys()) == {"inputs.0"}
    assert trace.derived_grads["inputs.0"].grad.tolist() == pytest.approx(oracle_x.grad.tolist())
    assert x.grad is None


def test_tinygrad_derived_grads_use_loss_fn_for_vector_output() -> None:
    """Derived gradients can use an explicit scalar loss function."""

    x = Tensor([1.0, -2.0, 3.0])
    trace = tl.trace(
        _tiny_vector_output,
        x,
        backend="tinygrad",
        grad_options=GradOptions(loss_fn=_tiny_vector_loss, input_grad_argnums=(0,)),
    )

    assert trace.derived_grads["inputs.0"].grad.tolist() == pytest.approx([2.0, -4.0, 6.0])
    assert x.grad is None


def test_tinygrad_derived_grads_reject_non_scalar_without_loss_fn() -> None:
    """Non-scalar raw outputs require a loss function for derived gradients."""

    with pytest.raises(ValueError, match="loss_fn"):
        tl.trace(
            _tiny_vector_output,
            Tensor([1.0, -2.0, 3.0]),
            backend="tinygrad",
            grad_options=GradOptions(input_grad_argnums=(0,)),
        )


def test_tinygrad_derived_grads_restore_preexisting_user_grad() -> None:
    """Bracketed backward should preserve pre-existing user grads and return increments."""

    x = Tensor([1.0, -2.0, 3.0])
    x.grad = Tensor([10.0, 20.0, 30.0]).realize()
    original_grad = x.grad

    trace = tl.trace(
        _tiny_square_loss,
        x,
        backend="tinygrad",
        grad_options=GradOptions(input_grad_argnums=(0,)),
    )

    assert x.grad is original_grad
    assert x.grad.tolist() == pytest.approx([10.0, 20.0, 30.0])
    assert trace.derived_grads["inputs.0"].grad.tolist() == pytest.approx([2.0, -4.0, 6.0])


def test_tinygrad_derived_grads_repeat_without_accumulating_user_grad() -> None:
    """Repeated bracketed gradient runs should not leave accumulated .grad state."""

    x = Tensor([1.0, -2.0, 3.0])
    first = tl.trace(
        _tiny_square_loss,
        x,
        backend="tinygrad",
        grad_options=GradOptions(input_grad_argnums=(0,)),
    )
    second = tl.trace(
        _tiny_square_loss,
        x,
        backend="tinygrad",
        grad_options=GradOptions(input_grad_argnums=(0,)),
    )

    assert first.derived_grads["inputs.0"].grad.tolist() == pytest.approx([2.0, -4.0, 6.0])
    assert second.derived_grads["inputs.0"].grad.tolist() == pytest.approx([2.0, -4.0, 6.0])
    assert x.grad is None


def test_tinygrad_intermediate_derived_grads_match_retained_backward_oracle() -> None:
    """Opt-in intermediate derived grads should match retained tinygrad tensors."""

    x = Tensor([1.0, -2.0, 3.0])
    trace = tl.trace(
        _tiny_intermediate_loss,
        x,
        backend="tinygrad",
        grad_options=GradOptions(input_grad_argnums=(0,), intermediate_grads=True),
    )
    oracle_x = Tensor([1.0, -2.0, 3.0])
    oracle_hidden = oracle_x * oracle_x
    oracle_activated = (oracle_hidden + 1.0).relu()
    oracle_scaled = oracle_activated * 3.0
    oracle_loss = oracle_scaled.sum()
    oracle_loss.backward()

    hidden_op = next(op for op in trace.layer_list if op.layer_type == "mul" and op.shape == (3,))
    scaled_op = [op for op in trace.layer_list if op.layer_type == "mul" and op.shape == (3,)][-1]

    assert set(trace.derived_grads.keys()) == {"inputs.0"}
    assert x.grad is None
    assert len(trace.intermediate_derived_grads) > 0
    assert hidden_op.derived_grad.tolist() == pytest.approx(oracle_hidden.grad.tolist())
    assert scaled_op.derived_grad.tolist() == pytest.approx(oracle_scaled.grad.tolist())
    assert hidden_op.label in trace.intermediate_derived_grads
    assert trace.intermediate_derived_grads[hidden_op.label].provenance["mechanism"] == (
        "tinygrad_retained_tensor_backward_no_realize"
    )
    assert trace.intermediate_derived_grads[hidden_op.label].provenance["status"] == "exact"
    assert trace.intermediate_derived_grads[hidden_op.label].provenance["save_predicate_id"] == (
        "trace.saved_ops"
    )


def test_tinygrad_derived_grads_are_not_backward_capture() -> None:
    """tinygrad traces should reject true backward surfaces with derived-grad guidance."""

    trace = tl.trace(
        _tiny_square_loss,
        Tensor([1.0, -2.0, 3.0]),
        backend="tinygrad",
        grad_options=GradOptions(input_grad_argnums=(0,)),
    )

    with pytest.raises(BackendUnsupportedError, match="trace\\.derived_grads"):
        trace.log_backward(trace[trace.output_layers[0]].out)
    with pytest.raises(ValueError, match="trace\\.derived_grads"):
        _ = trace.backward_passes
    with pytest.raises(ValueError, match="trace\\.derived_grads"):
        _ = trace.saved_grad_ops
    with pytest.raises(ValueError, match="trace\\.derived_grads"):
        _ = trace[0].grads


def test_tinygrad_derived_grads_reject_audit_only_payload_envelope() -> None:
    """Refuse derived grads when the tinygrad payload envelope is audit-only."""

    backend = TinygradBackend()
    x = Tensor([1.0, -2.0, 3.0])
    trace = tl.trace(_tiny_square_loss, x, backend="tinygrad")
    trace.tinygrad_payload_policy = "audit_only"

    with pytest.raises(BackendUnsupportedError, match="audit-only|derived_grads"):
        backend._attach_derived_grads(
            trace=trace,
            model=_tiny_square_loss,
            args=[x],
            captured_output=_tiny_square_loss(x),
            grad_options=GradOptions(input_grad_argnums=(0,)),
        )


def test_tinygrad_public_surface_matrix(tmp_path: Path) -> None:
    """Assert supported and unsupported public surfaces on a real tinygrad trace."""

    trace = tl.trace(
        _tiny_square_loss,
        Tensor([1.0, -2.0, 3.0]),
        backend="tinygrad",
        grad_options=GradOptions(input_grad_argnums=(0,)),
    )

    assert trace.backend == "tinygrad"
    assert trace.validate_forward_pass([_tiny_square_loss(Tensor([1.0, -2.0, 3.0]))]) is True
    assert trace.validation_replay_status.state == "passed"
    assert trace.validation_replay_status.source == "live"
    assert isinstance(trace.summary(), str)
    assert len(trace.to_pandas()) == len(trace.layer_list)
    assert trace.modules["self"].address == "self"
    assert trace.module_identity_mode == "function_root"
    assert trace.param_source == "none"
    assert list(trace.params.keys()) == []
    assert trace.has_backward_pass is False
    assert trace.num_backward_passes == 0
    assert trace.num_backward_edges is None
    assert set(trace.derived_grads.keys()) == {"inputs.0"}
    assert trace.derived_grads["inputs.0"].grad.tolist() == pytest.approx([2.0, -4.0, 6.0])

    outpath = tmp_path / "tinygrad_graph"
    dot = trace.draw(vis_outpath=str(outpath), vis_save_only=True, vis_fileformat="dot")
    assert isinstance(dot, str)

    audit_path = tmp_path / "tinygrad_audit.tlspec"
    trace.save(audit_path, level="audit")
    loaded = tl.load(audit_path)
    assert loaded.backend == "tinygrad"
    assert loaded.param_source == "none"
    assert all(op.out is None for op in loaded.layer_list)

    portable_path = tmp_path / "tinygrad_portable.tlspec"
    source_out = trace[trace.output_layers[0]].out
    expected = source_out.numpy()
    trace.save(portable_path)
    loaded_portable = tl.load(portable_path)
    loaded_out = loaded_portable[loaded_portable.output_layers[0]].out
    assert loaded_portable.backend == "tinygrad"
    assert getattr(loaded_portable, "payload_load_status") == "loaded_device_best_effort"
    assert isinstance(loaded_out, Tensor)
    assert loaded_out.shape == expected.shape
    assert str(loaded_out.dtype) == str(source_out.dtype)
    np.testing.assert_allclose(loaded_out.numpy(), expected)

    loaded_status = loaded_portable.validate_forward_pass(
        [_tiny_square_loss(Tensor([1.0, -2.0, 3.0]))]
    )
    assert loaded_status is loaded_portable.validation_replay_status
    assert loaded_status.state == "unavailable"
    assert loaded_status.available is False
    assert loaded_status.reason == "loaded_trace_runtime_capture_stripped"
    assert loaded_status.payload_load_status == "loaded_device_best_effort"
    with pytest.raises(TypeError, match="not a boolean"):
        bool(loaded_status)
    with pytest.raises(BackendUnsupportedError, match="trace\\.derived_grads"):
        trace.log_backward(trace[trace.output_layers[0]].out)
    with pytest.raises(ValueError, match="trace\\.derived_grads"):
        _ = trace.backward_passes
    with pytest.raises(ValueError, match="trace\\.derived_grads"):
        _ = trace[0].grads
