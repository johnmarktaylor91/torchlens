"""tinygrad backend capture tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import torchlens as tl
from torchlens.backends import (
    BackendPayloadUnsupportedError,
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


def test_tinygrad_spec_registered() -> None:
    """Assert the built-in tinygrad spec exposes the G1 capability split."""

    spec = get_backend_spec("tinygrad")
    assert spec.capabilities.backward_capture is False
    assert spec.capabilities.validation_replay is True
    assert spec.capabilities.fastlog is False
    assert spec.capabilities.interventions is False
    assert spec.capabilities.payload_materialization is False
    assert spec.capabilities.module_identity_modes == ("function_root",)
    assert spec.serialization_policy.payload_policy == "audit_only"
    assert capabilities.supports_backward_capture is False
    assert capabilities.supports_validation_replay is True
    assert capabilities.supports_fastlog is False
    assert capabilities.supports_intervention is False
    assert capabilities.supports_payload_materialization is False
    assert capabilities.module_identity_modes == ("function_root",)
    assert capabilities.payload_policy == "audit_only"
    assert capabilities.live_payload_policy == "dev_python_realized_copy"
    assert capabilities.trace_options == ("grad_options",)


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
        ({"module_identity_mode": "object_module"}, "module_identity_mode selection"),
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
    with pytest.raises(BackendUnsupportedError, match="audit-only|realized-copy"):
        loaded.validate_forward_pass([_tiny_square_loss(Tensor([1.0, -2.0, 3.0]))])

    with pytest.raises(
        BackendPayloadUnsupportedError, match="audit-only|materialized payloads"
    ) as exc_info:
        trace.save(tmp_path / "tinygrad_portable.tlspec")
    assert "expected a tensor for portable blobification" not in str(exc_info.value)
    with pytest.raises(BackendUnsupportedError, match="trace\\.derived_grads"):
        trace.log_backward(trace[trace.output_layers[0]].out)
    with pytest.raises(ValueError, match="trace\\.derived_grads"):
        _ = trace.backward_passes
    with pytest.raises(ValueError, match="trace\\.derived_grads"):
        _ = trace[0].grads
