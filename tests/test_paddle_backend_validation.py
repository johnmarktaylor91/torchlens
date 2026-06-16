"""Paddle backend validation tripwire adversary tests."""

from __future__ import annotations

from typing import Any

import pytest

paddle = pytest.importorskip("paddle")

import torchlens as tl  # noqa: E402
from torchlens.backends import BackendUnsupportedError  # noqa: E402
from torchlens.backends.paddle import PaddleBackend  # noqa: E402
from torchlens.backends.paddle import wrappers as paddle_wrappers  # noqa: E402
from torchlens.validation.invariants import check_metadata_invariants  # noqa: E402
from torchlens.validation.status import ValidationReplayStatus  # noqa: E402

pytestmark = pytest.mark.backend_paddle


def _inputs() -> tuple[Any, Any, Any, Any, Any]:
    """Return deterministic explicit-parameter MLP inputs.

    Returns
    -------
    tuple[Any, Any, Any, Any, Any]
        Input, first weight, first bias, second weight, second bias tensors.
    """

    paddle.seed(0)
    x = paddle.arange(8, dtype="float32").reshape([2, 4]) / 8.0
    w1 = paddle.arange(32, dtype="float32").reshape([4, 8]) / 16.0
    b1 = paddle.arange(8, dtype="float32") / 10.0
    w2 = paddle.arange(16, dtype="float32").reshape([8, 2]) / 12.0
    b2 = paddle.arange(2, dtype="float32") / 7.0
    return x, w1, b1, w2, b2


def _functional_mlp(x: Any, w1: Any, b1: Any, w2: Any, b2: Any) -> Any:
    """Run a two-layer MLP with explicit parameter tensors.

    Parameters
    ----------
    x
        Input tensor.
    w1
        First layer weight.
    b1
        First layer bias.
    w2
        Second layer weight.
    b2
        Second layer bias.

    Returns
    -------
    Any
        MLP output tensor.
    """

    hidden = paddle.nn.functional.linear(x, w1, b1)
    hidden = paddle.nn.functional.relu(hidden)
    return paddle.nn.functional.linear(hidden, w2, b2)


def _healthy_trace() -> Any:
    """Return a healthy Paddle validation trace.

    Returns
    -------
    Any
        Captured Paddle trace.
    """

    return tl.trace(_functional_mlp, _inputs(), backend="paddle")


def test_paddle_validation_healthy_two_layer_mlp_passes() -> None:
    """Validate a healthy two-layer MLP with replay and perturbation."""

    backend = PaddleBackend()
    trace = _healthy_trace()

    assert backend.validate_trace(trace) is True
    status = trace.validation_replay_status
    assert status.state == "passed"
    assert status.replayed_node_count >= 1
    assert PaddleBackend().validate_entry(_functional_mlp, _inputs()) is True


def test_paddle_validation_fails_unwrapped_intermediate_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail validation when a real intermediate op is omitted from wrapping."""

    monkeypatch.setattr(
        paddle_wrappers,
        "_TOP_LEVEL_CORE_OPS",
        paddle_wrappers._TOP_LEVEL_CORE_OPS - {"add"},
    )

    def add_then_relu(x: Any, y: Any) -> Any:
        """Apply an unwrapped add followed by a wrapped relu."""

        return paddle.nn.functional.relu(paddle.add(x, y))

    args = (paddle.ones([2, 3], dtype="float32"), paddle.ones([2, 3], dtype="float32"))
    trace = tl.trace(add_then_relu, args, backend="paddle")

    assert PaddleBackend().validate_trace(trace) is False
    assert PaddleBackend().validate_entry(add_then_relu, args) is False


def test_paddle_validation_fails_dropped_parent_edge() -> None:
    """Fail validation when materialized graph parents lose a captured edge."""

    trace = _healthy_trace()
    relu = next(op for op in trace.layer_list if op.layer_type == "functional.relu")
    relu.parents = []

    assert PaddleBackend().validate_trace(trace) is False


def test_paddle_validation_fails_corrupted_saved_output() -> None:
    """Fail validation when a saved Paddle op output payload is corrupted."""

    trace = _healthy_trace()
    relu = next(op for op in trace.layer_list if op.layer_type == "functional.relu")
    relu.out = paddle.zeros_like(relu.out) - 99.0

    assert PaddleBackend().validate_trace(trace) is False


@pytest.mark.parametrize(
    "func",
    [
        lambda x: paddle.full(x.shape, float(x.sum()), dtype=x.dtype),
        lambda x: x * float(x.sum()),
    ],
)
def test_paddle_validation_scalar_escape_raises_at_capture(func: Any) -> None:
    """Raise at capture for depth-0 tensor-derived Python scalar escapes."""

    with pytest.raises(BackendUnsupportedError, match="scalar/control escape"):
        tl.trace(func, paddle.ones([2, 2], dtype="float32"), backend="paddle")


def test_paddle_validation_loaded_payload_stripped_trace_is_unavailable() -> None:
    """Return unavailable status for loaded traces stripped of replay payloads."""

    trace = _healthy_trace()
    trace._loaded_from_bundle = True
    trace._paddle_op_captures = ()

    result = PaddleBackend().validate_trace(trace)

    assert isinstance(result, ValidationReplayStatus)
    assert result.state == "unavailable"
    assert result.reason == "loaded_trace_runtime_capture_stripped"
    assert result.passed is False


def test_paddle_validation_metadata_invariants_pass_on_valid_trace() -> None:
    """Run backend-neutral metadata invariants on a valid Paddle trace."""

    trace = _healthy_trace()

    assert check_metadata_invariants(trace) is True


def test_paddle_validation_same_object_static_snapshot_guard_is_p6_todo() -> None:
    """Document that same-object no-op inventory coverage belongs to P6."""

    pytest.skip("TODO(P6): static inventory snapshot guards same-object no-op coverage.")
