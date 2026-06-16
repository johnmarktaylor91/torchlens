"""Paddle object-module hierarchy tests."""

from __future__ import annotations

import pytest

paddle = pytest.importorskip("paddle")

import torchlens as tl  # noqa: E402

pytestmark = pytest.mark.backend_paddle


class PaddleScale(paddle.nn.Layer):
    """Parameterless Paddle layer with a keyword argument."""

    def forward(self, x: object, scale: float = 1.0) -> object:
        """Scale an input tensor.

        Parameters
        ----------
        x
            Input tensor.
        scale
            Scalar scale value.

        Returns
        -------
        object
            Scaled tensor.
        """

        return x * scale


class PaddleNested(paddle.nn.Layer):
    """Nested Paddle module with parameterized children."""

    def __init__(self) -> None:
        """Initialize child layers."""

        super().__init__()
        self.seq = paddle.nn.Sequential(paddle.nn.Linear(4, 4), paddle.nn.ReLU())
        self.scale = PaddleScale()
        self.head = paddle.nn.Linear(4, 2)

    def forward(self, x: object, scale: float = 1.0) -> object:
        """Run nested forward."""

        hidden = self.seq(x)
        hidden = self.scale(hidden, scale=scale)
        return self.head(hidden)


def _input() -> object:
    """Return a deterministic Paddle input.

    Returns
    -------
    object
        Paddle tensor.
    """

    paddle.seed(0)
    return paddle.ones([2, 4], dtype="float32")


def test_paddle_nested_layer_addresses_and_kwargs() -> None:
    """Nested Paddle layers should populate module logs and forward kwargs."""

    trace = tl.trace(PaddleNested(), _input(), {"scale": 2.0}, backend="paddle")

    assert trace.module_identity_mode == "object_module"
    addresses = {module.address for module in trace.modules}
    assert {"self", "seq", "seq.0", "seq.1", "scale", "head"} <= addresses
    assert trace.module_calls["scale:1"].forward_kwargs == {"scale": 2.0}
    scale_labels = trace.resolve_sites(tl.in_module("scale"), max_fanout=16).labels()
    assert scale_labels
    assert all("scale:1" in trace[label].modules for label in scale_labels)


def test_paddle_param_logs_from_named_parameters() -> None:
    """Paddle named parameters should become Trace param logs."""

    trace = tl.trace(PaddleNested(), _input(), backend="paddle")

    addresses = {param.address for param in trace.param_logs}
    assert {"seq.0.weight", "seq.0.bias", "head.weight", "head.bias"} <= addresses
    assert trace.num_param_tensors == 4
    assert trace.modules["seq.0"].params
    assert trace.modules["head"].params


def test_paddle_call_index_counting_for_reused_layer() -> None:
    """Repeated calls to the same Paddle layer should increment call indexes."""

    class Reused(paddle.nn.Layer):
        """Layer that calls one child twice."""

        def __init__(self) -> None:
            """Initialize shared child."""

            super().__init__()
            self.scale = PaddleScale()

        def forward(self, x: object) -> object:
            """Run the shared child twice."""

            first = self.scale(x, scale=2.0)
            return self.scale(first, scale=3.0)

    trace = tl.trace(Reused(), _input(), backend="paddle")

    assert trace.modules["scale"].num_calls == 2
    assert {"scale:1", "scale:2"} <= set(trace.module_calls.keys())
    assert trace.module_calls["scale:1"].forward_kwargs == {"scale": 2.0}
    assert trace.module_calls["scale:2"].forward_kwargs == {"scale": 3.0}


def test_paddle_object_module_vs_function_root() -> None:
    """Paddle layers use object-module mode while raw callables use function-root."""

    object_trace = tl.trace(PaddleNested(), _input(), backend="paddle")
    function_trace = tl.trace(lambda x: paddle.nn.functional.relu(x), _input(), backend="paddle")

    assert object_trace.module_identity_mode == "object_module"
    assert function_trace.module_identity_mode == "function_root"
    assert [module.address for module in function_trace.modules] == ["self"]
