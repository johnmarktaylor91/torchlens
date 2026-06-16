"""Paddle backend derived-gradient preview tests."""

from __future__ import annotations

from typing import Any

import pytest

paddle = pytest.importorskip("paddle")

import torchlens as tl  # noqa: E402
from torchlens.backends import BackendUnsupportedError  # noqa: E402
from torchlens.backends.paddle import GradOptions  # noqa: E402
from torchlens.backends.paddle.backend import (  # noqa: E402
    PaddleIntermediateSignature,
    _paddle_trace_intermediate_signatures,
)

pytestmark = pytest.mark.backend_paddle


def _assert_close(actual: Any, expected: Any) -> None:
    """Assert two Paddle tensors are numerically close.

    Parameters
    ----------
    actual
        Actual Paddle tensor.
    expected
        Expected Paddle tensor.
    """

    assert paddle.allclose(actual, expected, rtol=1e-5, atol=1e-6)


def _loss(output: Any) -> Any:
    """Return scalar sum loss.

    Parameters
    ----------
    output
        Model output.

    Returns
    -------
    Any
        Scalar Paddle loss.
    """

    return output.sum()


class LinearRelu(paddle.nn.Layer):
    """Deterministic linear-relu fixture."""

    def __init__(self) -> None:
        """Initialize deterministic parameters."""

        super().__init__()
        self.linear = paddle.nn.Linear(3, 2)
        self.linear.weight.set_value(
            paddle.to_tensor([[0.2, -0.4], [0.7, 0.3], [-0.5, 0.1]], dtype="float32")
        )
        self.linear.bias.set_value(paddle.to_tensor([0.05, -0.1], dtype="float32"))

    def forward(self, x: Any) -> Any:
        """Run ``linear -> relu``.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        Any
            Model output.
        """

        return paddle.nn.functional.relu(self.linear(x))


def test_paddle_leaf_input_and_param_grads_match_direct_reference() -> None:
    """Leaf input and parameter derived grads should match direct ``paddle.grad``."""

    paddle.seed(0)
    model = LinearRelu()
    x = paddle.to_tensor([[1.0, -2.0, 0.5], [0.3, 0.1, -0.8]], dtype="float32")
    trace = tl.trace(
        model,
        x,
        backend="paddle",
        grad_options=GradOptions(loss_fn=_loss),
    )

    x.stop_gradient = False
    for param in model.parameters():
        param.stop_gradient = False
    direct_output = model(x)
    expected = paddle.grad(
        _loss(direct_output),
        [x, model.linear.weight, model.linear.bias],
        allow_unused=True,
    )

    assert set(trace.derived_grads.keys()) == {
        "inputs.0",
        "params.linear.weight",
        "params.linear.bias",
    }
    _assert_close(trace.derived_grads["inputs.0"].grad, expected[0])
    _assert_close(trace.derived_grads["params.linear.weight"].grad, expected[1])
    _assert_close(trace.derived_grads["params.linear.bias"].grad, expected[2])
    assert trace.params["linear.weight"].grad is trace.derived_grads["params.linear.weight"].grad


def test_paddle_derived_grads_do_not_create_backward_logs() -> None:
    """Derived gradients should not masquerade as captured backward-pass logs."""

    paddle.seed(0)
    trace = tl.trace(
        LinearRelu(),
        paddle.ones([2, 3], dtype="float32"),
        backend="paddle",
        grad_options=GradOptions(loss_fn=_loss),
    )

    assert trace.has_backward_pass is False
    assert not trace.backward_pass_logs
    assert not trace.grad_fn_logs


def test_paddle_intermediate_grads_match_direct_oracle_and_skip_unused() -> None:
    """Intermediate grads should match reachable direct AD values and skip ``None``."""

    paddle.seed(0)

    def model(x: Any) -> Any:
        """Run a reachable two-op path and one unused op."""

        _unused = x * 3.0
        hidden = x * 2.0
        return paddle.nn.functional.relu(hidden).sum()

    x = paddle.ones([2, 2], dtype="float32")
    trace = tl.trace(
        model,
        x,
        backend="paddle",
        keep_orphans=True,
        grad_options=GradOptions(intermediate_grads=True, max_intermediate_grads=8),
    )

    records = trace.intermediate_derived_grads
    unused_label = next(op.label for op in trace.layer_list if op.func_name == "tensor.__mul__")
    relu_label = next(op.label for op in trace.layer_list if op.func_name == "functional.relu")
    sum_label = next(op.label for op in trace.layer_list if op.func_name == "tensor.sum")

    assert unused_label not in records
    _assert_close(records[relu_label].grad, paddle.ones([2, 2], dtype="float32"))
    _assert_close(records[sum_label].grad, paddle.ones([], dtype="float32"))


def test_paddle_same_shape_relus_are_disambiguated_by_ordinal() -> None:
    """Two same-shape relus should receive their own cotangents."""

    paddle.seed(0)

    def model(x: Any) -> Any:
        """Run two same-shape relus with different downstream weights."""

        first = paddle.nn.functional.relu(x)
        second = paddle.nn.functional.relu(x + 1.0)
        return (first * 2.0 + second * 3.0).sum()

    trace = tl.trace(
        model,
        paddle.ones([2, 2], dtype="float32"),
        backend="paddle",
        grad_options=GradOptions(intermediate_grads=True, max_intermediate_grads=16),
    )

    relu_ops = [op for op in trace.layer_list if op.func_name == "functional.relu"]
    assert len(relu_ops) == 2
    _assert_close(
        trace.intermediate_derived_grads[relu_ops[0].label].grad, paddle.full([2, 2], 2.0)
    )
    _assert_close(
        trace.intermediate_derived_grads[relu_ops[1].label].grad, paddle.full([2, 2], 3.0)
    )


def test_paddle_duplicate_trace_signature_group_is_ambiguous() -> None:
    """A duplicate signature group should be detectable and skipped by attach logic."""

    paddle.seed(0)

    def model(x: Any) -> Any:
        """Run two relus."""

        return paddle.nn.functional.relu(x) + paddle.nn.functional.relu(x)

    trace = tl.trace(model, paddle.ones([2, 2], dtype="float32"), backend="paddle")
    relu_ops = [op for op in trace.layer_list if op.func_name == "functional.relu"]
    relu_ops[1].func_call_id = relu_ops[0].func_call_id
    groups = _paddle_trace_intermediate_signatures(trace)
    signature = PaddleIntermediateSignature(
        func_call_id=relu_ops[0].func_call_id,
        op_name=relu_ops[0].func_name,
        parent_labels=tuple(relu_ops[0].parents),
        module_stack=tuple(relu_ops[0].modules),
    )

    assert len(groups[signature]) == 2


def test_paddle_max_intermediate_grads_cap_raises() -> None:
    """Intermediate cap should raise when exact attached records exceed it."""

    paddle.seed(0)

    with pytest.raises(BackendUnsupportedError, match="capped"):
        tl.trace(
            lambda x: paddle.nn.functional.relu(x + 1.0).sum(),
            paddle.ones([2, 2], dtype="float32"),
            backend="paddle",
            grad_options=GradOptions(intermediate_grads=True, max_intermediate_grads=1),
        )


def test_paddle_non_scalar_output_without_loss_fn_raises() -> None:
    """Non-scalar raw output should require ``loss_fn``."""

    paddle.seed(0)

    with pytest.raises(ValueError, match="scalar"):
        tl.trace(
            lambda x: x + 1.0,
            paddle.ones([2, 2], dtype="float32"),
            backend="paddle",
            grad_options=GradOptions(),
        )


def test_paddle_replay_output_divergence_refuses_grads() -> None:
    """Divergent AD replay output should refuse derived gradients."""

    paddle.seed(0)

    class Diverges:
        """Callable that changes output between capture and replay."""

        def __init__(self) -> None:
            """Initialize call counter."""

            self.calls = 0

        def __call__(self, x: Any) -> Any:
            """Return a call-count-dependent scalar."""

            self.calls += 1
            return (x + float(self.calls)).sum()

    with pytest.raises(ValueError, match="diverged"):
        tl.trace(
            Diverges(),
            paddle.ones([2, 2], dtype="float32"),
            backend="paddle",
            grad_options=GradOptions(),
        )


def test_paddle_stop_gradient_and_grad_restore_after_success_and_exception() -> None:
    """Replay should restore input and parameter gradient state in all exits."""

    paddle.seed(0)
    model = LinearRelu()
    x = paddle.ones([2, 3], dtype="float32")
    x_prior_grad = paddle.full([2, 3], 7.0, dtype="float32")
    param_prior_grad = paddle.full(model.linear.weight.shape, 5.0, dtype="float32")
    x.stop_gradient = True
    x.grad = x_prior_grad
    model.linear.weight.stop_gradient = True
    model.linear.weight.grad = param_prior_grad

    tl.trace(model, x, backend="paddle", grad_options=GradOptions(loss_fn=_loss))

    assert x.stop_gradient is True
    assert model.linear.weight.stop_gradient is True
    _assert_close(x.grad, x_prior_grad)
    _assert_close(model.linear.weight.grad, param_prior_grad)

    with pytest.raises(ValueError, match="scalar"):
        tl.trace(model, x, backend="paddle", grad_options=GradOptions())

    assert x.stop_gradient is True
    assert model.linear.weight.stop_gradient is True
    _assert_close(x.grad, x_prior_grad)
    _assert_close(model.linear.weight.grad, param_prior_grad)
