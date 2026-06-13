"""tinygrad backend capture tests."""

from __future__ import annotations

from typing import Any

import pytest

import torchlens as tl
from torchlens.backends import BackendUnsupportedError, get_backend_spec

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


def test_tinygrad_rejects_mid_capture_realize_and_mutation() -> None:
    """Reject tinygrad execution that can truncate lazy lineage during capture."""

    x = Tensor([1.0, 2.0])
    with pytest.raises(BackendUnsupportedError, match="realize|assign|TinyJit"):
        tl.trace(_realize_inside, x, backend="tinygrad")

    y = Tensor([1.0, 2.0])
    with pytest.raises(BackendUnsupportedError, match="realize|assign|TinyJit"):
        tl.trace(_assign_inside, y, backend="tinygrad")


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


def test_tinygrad_save_shaping_rejected() -> None:
    """Reject save shaping because tinygrad preview is full-save only."""

    x = Tensor([1.0, 2.0])
    with pytest.raises(BackendUnsupportedError, match="full-save only|save-shaping"):
        tl.trace(_tiny_block, x, backend="tinygrad", save=tl.func("relu"))


def test_tinygrad_backward_ready_rejected() -> None:
    """Reject true backward capture surfaces for tinygrad G1."""

    x = Tensor([1.0, 2.0])
    with pytest.raises(BackendUnsupportedError, match="backward_ready"):
        tl.trace(_tiny_block, x, backend="tinygrad", backward_ready=True)
