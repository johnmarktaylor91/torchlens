"""Optional smoke coverage for the technical-preview Paddle backend."""

from __future__ import annotations

import pytest

paddle = pytest.importorskip("paddle")

import torchlens as tl  # noqa: E402
from torchlens.backends import BackendUnsupportedError  # noqa: E402

pytestmark = pytest.mark.backend_paddle


def _input(shape: tuple[int, ...] = (2, 4)) -> object:
    """Return a deterministic Paddle input.

    Parameters
    ----------
    shape
        Tensor shape.

    Returns
    -------
    object
        Paddle tensor.
    """

    paddle.seed(0)
    return paddle.ones(shape, dtype="float32")


def test_paddle_relu_single_op_smoke() -> None:
    """Capture a single Paddle relu operation."""

    x = _input()
    trace = tl.trace(lambda value: paddle.nn.functional.relu(value), x, backend="paddle")

    assert trace.backend == "paddle"
    assert trace.has_backward_pass is False
    assert trace.module_identity_mode == "function_root"
    assert trace.num_ops == 1
    assert any("relu" in label for label in trace.layer_labels)
    assert trace.output_layers == ["functional.relu_1_2_raw"]
    capture = trace._paddle_op_captures[0]
    assert capture.op_name == "functional.relu"
    assert capture.tensor_inputs[0].label == "input.arg_0"


def test_paddle_two_layer_mlp_parents_and_labels() -> None:
    """Capture a two-layer Paddle MLP with correct structural parents."""

    class MLP(paddle.nn.Layer):
        """Small Paddle MLP used for graph smoke testing."""

        def __init__(self) -> None:
            """Initialize linear children."""

            super().__init__()
            self.l1 = paddle.nn.Linear(4, 8)
            self.l2 = paddle.nn.Linear(8, 2)

        def forward(self, x: object) -> object:
            """Run the MLP forward pass."""

            hidden = self.l1(x)
            hidden = paddle.nn.functional.relu(hidden)
            return self.l2(hidden)

    trace = tl.trace(MLP(), _input(), backend="paddle")

    assert trace.backend == "paddle"
    assert "input.arg_0" in trace.layer_labels
    relu_label = next(label for label in trace.op_labels if "relu" in label)
    final_label = trace.output_layers[0] + ":1"
    assert trace[relu_label].parents == ["functional.linear_1_2_raw"]
    assert trace[final_label].parents == ["functional.relu_1_3_raw"]
    assert all(capture.tensor_inputs for capture in trace._paddle_op_captures)


def test_paddle_source_input_labels_and_function_root() -> None:
    """Raw callables should use function-root mode and source input labels."""

    def raw_callable(x: object, y: object) -> object:
        """Add two Paddle tensors."""

        return x + y

    trace = tl.trace(raw_callable, (_input(), _input()), backend="paddle")

    assert trace.module_identity_mode == "function_root"
    assert {"input.arg_0", "input.arg_1"} <= set(trace.layer_labels)
    add_label = next(label for label in trace.op_labels if "__add__" in label)
    assert trace[add_label].parents == ["input.arg_0", "input.arg_1"]


def test_paddle_recursion_guard_records_one_composite_op() -> None:
    """A wrapped composite should not also record its internal wrapped operations."""

    layer = paddle.nn.Linear(4, 3)
    trace = tl.trace(
        lambda x: paddle.nn.functional.linear(x, layer.weight, layer.bias),
        _input(),
        backend="paddle",
    )

    op_labels = [label for label in trace.op_labels if not label.startswith("input.")]
    assert len(op_labels) == 1
    assert "linear" in op_labels[0]


def test_paddle_dygraph_guard_rejects_static_or_pir(monkeypatch: pytest.MonkeyPatch) -> None:
    """Paddle capture should reject non-dygraph runtimes."""

    monkeypatch.setattr(paddle, "in_dynamic_mode", lambda: False)

    with pytest.raises(BackendUnsupportedError, match="dygraph"):
        tl.trace(lambda x: x + 1, _input(), backend="paddle")


@pytest.mark.parametrize(
    ("name", "func", "message"),
    [
        ("rng", lambda x: paddle.rand([2, 2]), "stochastic"),
        ("inplace", lambda x: x.__setitem__((0, 0), 1.0) or x, "in-place"),
        ("scalar", lambda x: x * float(x.sum()), "scalar/control escape"),
    ],
)
def test_paddle_denied_ops_raise_typed_errors(
    name: str,
    func: object,
    message: str,
) -> None:
    """Denied Paddle operation classes should wrap-and-raise."""

    del name
    with pytest.raises(BackendUnsupportedError, match=message):
        tl.trace(func, paddle.ones([2, 2], dtype="float32"), backend="paddle")


def test_paddle_std_internal_escape_is_depth_exempt() -> None:
    """Wrapped std should capture despite internal Paddle scalar/control escapes."""

    trace = tl.trace(
        lambda x: paddle.std(x), paddle.ones([2, 2], dtype="float32"), backend="paddle"
    )

    assert trace.num_ops == 1
    assert any("std" in label for label in trace.layer_labels)


def test_paddle_same_dtype_astype_preserves_parent_label() -> None:
    """Same-object astype aliases should not mis-parent downstream ops."""

    x = paddle.ones([2, 3], dtype="float32")
    weight = paddle.ones([3, 3], dtype="float32")

    def model(x_arg: object, weight_arg: object) -> object:
        """Run producer, same-object alias, then consumer."""

        produced = x_arg + 1
        alias = produced.astype(produced.dtype)
        return paddle.matmul(alias, weight_arg)

    trace = tl.trace(model, (x, weight), backend="paddle")

    matmul_label = next(label for label in trace.op_labels if "matmul" in label)
    assert trace[matmul_label].parents == ["tensor.__add___1_3_raw", "input.arg_1"]
    assert not any("astype" in label for label in trace.layer_labels)
    assert trace._paddle_alias_annotations[0]["preserved_label"] == "tensor.__add___1_3_raw"
