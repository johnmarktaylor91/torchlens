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
