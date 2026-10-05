"""Paddle backend validation tripwire adversary tests."""

from __future__ import annotations

from typing import Any

import pytest

paddle = pytest.importorskip("paddle")

import torchlens as tl  # noqa: E402
from torchlens.backends import BackendUnsupportedError  # noqa: E402
from torchlens.backends.paddle import (  # noqa: E402
    PaddleBackend,
    wrappers as paddle_wrappers,
)
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


def test_paddle_validation_same_object_static_snapshot_guard_fires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate a same-object no-op inventory gap and require static coverage to fail."""

    patched_tensor_methods = paddle_wrappers._TENSOR_CORE_METHODS - {"astype", "reshape"}
    monkeypatch.setattr(paddle_wrappers, "_TENSOR_CORE_METHODS", patched_tensor_methods)
    registry = paddle_wrappers._PaddleWrapperRegistry()
    registry.wrap(object())
    try:
        inventory = registry.inventory()
    finally:
        registry.unwrap()

    with pytest.raises(AssertionError):
        assert {"tensor.astype", "tensor.reshape"} <= set(inventory.wrapped)


class _LinearRelu(paddle.nn.Layer):
    """Layer whose ops read its own registered parameters."""

    def __init__(self) -> None:
        """Build one linear layer with deterministic weights."""

        super().__init__()
        self.linear = paddle.nn.Linear(4, 3)
        self.linear.weight.set_value(paddle.arange(12, dtype="float32").reshape([4, 3]) / 10.0)
        self.linear.bias.set_value(paddle.to_tensor([0.1, -0.2, 0.3], dtype="float32"))

    def forward(self, x: Any) -> Any:
        """Run linear then relu."""

        return paddle.nn.functional.relu(self.linear(x))


class _ClosureTensorLayer(paddle.nn.Layer):
    """Layer that reads an UNREGISTERED tensor (not a parameter, not an input)."""

    def __init__(self) -> None:
        """Hold a plain tensor attribute outside the parameter registry."""

        super().__init__()
        self.linear = paddle.nn.Linear(4, 3)
        self.__dict__["hidden"] = paddle.ones([3], dtype="float32")

    def forward(self, x: Any) -> Any:
        """Add the unregistered tensor through an explicit functional op."""

        return paddle.add(self.linear(x), self.__dict__["hidden"])


def _layer_input() -> Any:
    """Return a deterministic Layer input."""

    return paddle.arange(8, dtype="float32").reshape([2, 4]) / 8.0 - 0.25


def test_paddle_validation_layer_parameters_are_known_sources() -> None:
    """Ops reading a Layer's own parameters pass the coverage oracle and replay."""

    trace = tl.trace(_LinearRelu(), _layer_input(), backend="paddle")

    assert trace.module_identity_mode == "object_module"
    assert PaddleBackend().validate_trace(trace) is True
    assert trace.validation_replay_status.replayed_node_count == 2
    linear = next(c for c in trace._paddle_op_captures if c.op_name == "functional.linear")
    assert sorted(leaf.param_address for leaf in linear.tensor_inputs if leaf.label is None) == [
        "linear.bias",
        "linear.weight",
    ]
    assert linear.capture_gap_markers == ()
    assert tl.validate(_LinearRelu(), _layer_input(), scope="forward", backend="paddle") is True


def test_paddle_validation_unregistered_tensor_leaf_still_fails() -> None:
    """An unlabeled tensor that is NOT a registered parameter stays a coverage gap."""

    trace = tl.trace(_ClosureTensorLayer(), _layer_input(), backend="paddle")

    add = next(c for c in trace._paddle_op_captures if c.op_name.endswith("add"))
    assert any("unlabeled tensor input" in marker for marker in add.capture_gap_markers)
    assert PaddleBackend().validate_trace(trace) is False


def test_paddle_validation_parameter_replay_uses_the_parameter_value() -> None:
    """A parameter changed after capture makes replay disagree: it fails, never passes."""

    model = _LinearRelu()
    trace = tl.trace(model, _layer_input(), backend="paddle")
    model.linear.bias.set_value(paddle.to_tensor([5.0, 5.0, 5.0], dtype="float32"))

    assert PaddleBackend().validate_trace(trace) is False


def test_paddle_validation_parameter_replay_leaves_the_model_unchanged() -> None:
    """Validation replays on copies; the user's parameters are untouched."""

    model = _LinearRelu()
    before = model.linear.weight.numpy().copy()
    trace = tl.trace(model, _layer_input(), backend="paddle")

    assert PaddleBackend().validate_trace(trace) is True
    assert (model.linear.weight.numpy() == before).all()


def test_paddle_validation_layer_corrupted_output_still_fails() -> None:
    """Corrupting the parameter-reading op's saved output still fails replay."""

    trace = tl.trace(_LinearRelu(), _layer_input(), backend="paddle")
    linear = next(op for op in trace.layer_list if op.layer_type == "functional.linear")
    linear.out = linear.out + 1.0

    assert PaddleBackend().validate_trace(trace) is False


def test_paddle_function_root_trace_has_buffers_and_graph_shape_hash() -> None:
    """A function-root preview trace carries ``buffers`` and a graph hash."""

    first = tl.trace(_functional_mlp, _inputs(), backend="paddle")
    second = tl.trace(_functional_mlp, _inputs(), backend="paddle")
    different = tl.trace(
        lambda x: paddle.nn.functional.relu(x + 1.0), _layer_input(), backend="paddle"
    )

    assert first.module_identity_mode == "function_root"
    assert first.buffers is not None and len(first.buffers) == 0
    assert isinstance(first.graph_shape_hash, str)
    assert first.graph_shape_hash == second.graph_shape_hash
    assert different.graph_shape_hash != first.graph_shape_hash


class _ConvBnRelu(paddle.nn.Layer):
    """ResNet stem shape: conv, BatchNorm2D (eval), relu."""

    def __init__(self) -> None:
        """Build the stem."""

        super().__init__()
        self.conv = paddle.nn.Conv2D(2, 3, 3, padding=1)
        self.bn = paddle.nn.BatchNorm2D(3)

    def forward(self, x: Any) -> Any:
        """Run conv, batch norm, relu."""

        return paddle.nn.functional.relu(self.bn(self.conv(x)))


_ORIGINAL_RELU = paddle.nn.functional.relu


def _stale_alias_relu(x: Any) -> Any:
    """Call relu through a reference bound before any wrap (a user stale alias)."""

    return _ORIGINAL_RELU(x * 2.0)


def test_paddle_validation_batchnorm_layer_import_alias_is_captured() -> None:
    """``nn.BatchNorm2D`` calls its module's import-time ``batch_norm`` alias; it is captured."""

    paddle.seed(0)
    model = _ConvBnRelu()
    model.eval()
    x = paddle.arange(32, dtype="float32").reshape([1, 2, 4, 4]) / 16.0
    trace = tl.trace(model, x, backend="paddle")

    assert "functional.batch_norm" in [op.layer_type for op in trace.layer_list]
    assert PaddleBackend().validate_trace(trace) is True
    from paddle.nn.layer import norm as paddle_norm

    assert paddle_norm.batch_norm is paddle.nn.functional.batch_norm


def test_paddle_validation_user_stale_alias_still_fails_closed() -> None:
    """A user's own pre-wrap alias is not patched: the gap still fails validation."""

    trace = tl.trace(_stale_alias_relu, paddle.ones([2, 2], dtype="float32"), backend="paddle")

    assert PaddleBackend().validate_trace(trace) is False
