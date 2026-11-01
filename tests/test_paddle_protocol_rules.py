"""Paddle preview under the backend protocol: conv layers, constants, reasons, refusals."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

paddle = pytest.importorskip("paddle")

import torchlens as tl  # noqa: E402
from torchlens.backends import REFUSAL_CODES  # noqa: E402
from torchlens.backends._backend_protocol import VALIDATION_REASONS  # noqa: E402
from torchlens.backends._protocols.paddle import (  # noqa: E402
    PADDLE_CODES_PENDING_REGISTRATION,
    paddle_package_refusal_codes,
)
from torchlens.backends.paddle import PaddleBackend  # noqa: E402
from torchlens.backends.paddle.validation_decisions import (  # noqa: E402
    PADDLE_VALIDATION_REASONS,
    REASON_METADATA_INVARIANT,
)
from torchlens.backends.registry import get_backend_spec  # noqa: E402
from torchlens.errors import BackendUnsupportedError  # noqa: E402

pytestmark = pytest.mark.backend_paddle

_F = paddle.nn.functional


def _protocol() -> Any:
    """The registered Paddle protocol object."""

    factory = get_backend_spec("paddle").protocol
    assert factory is not None
    return factory()


def _x24() -> Any:
    """A deterministic (2, 4) float32 input."""

    return paddle.to_tensor(np.linspace(-1.0, 1.0, 8, dtype="float32").reshape(2, 4))


def _ops(trace: Any, layer_type: str) -> list[Any]:
    """Ops of ``trace`` with the given layer type."""

    return [op for op in trace._layer_list if op.type == layer_type]


def _parent_types(trace: Any, op: Any) -> list[str]:
    """Layer types of an op's parents."""

    by_label = {other.layer_label: other for other in trace._layer_list}
    return [by_label[str(parent)].type for parent in op.parents if str(parent) in by_label]


class _ConvNet(paddle.nn.Layer):
    """One convolution layer of the requested rank followed by a relu."""

    def __init__(self, rank: int, padding_mode: str = "zeros") -> None:
        """Build the conv layer."""

        super().__init__()
        layer_cls = {1: paddle.nn.Conv1D, 2: paddle.nn.Conv2D, 3: paddle.nn.Conv3D}[rank]
        self.conv = layer_cls(2, 3, 3, padding=1, padding_mode=padding_mode)

    def forward(self, x: Any) -> Any:
        """Convolve then rectify."""

        return _F.relu(self.conv(x))


def _conv_input(rank: int) -> Any:
    """A deterministic NC* input for a rank-``rank`` convolution."""

    shape = (1, 2) + (5,) * rank
    count = int(np.prod(shape))
    return paddle.to_tensor(np.linspace(-1.0, 1.0, count, dtype="float32").reshape(shape))


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_conv_layers_are_layer_level_ops_with_parents(rank: int) -> None:
    """Conv1D/2D/3D appear as one ``conv<n>d`` op whose parent is the input (F4a)."""

    paddle.seed(0)
    trace = tl.trace(_ConvNet(rank), _conv_input(rank), backend="paddle")

    convs = _ops(trace, f"conv{rank}d")
    assert len(convs) == 1
    assert _parent_types(trace, convs[0]) == ["input"]
    relu = _ops(trace, "functional.relu")[0]
    assert _parent_types(trace, relu) == [f"conv{rank}d"]
    assert not [op for op in trace._layer_list if "_conv_nd" in op.type]
    assert PaddleBackend().validate_trace(trace) is True


def test_conv_layer_reflect_padding_replays() -> None:
    """A non-zeros padding mode is captured as one op and replays through the pad."""

    paddle.seed(0)
    trace = tl.trace(_ConvNet(2, padding_mode="reflect"), _conv_input(2), backend="paddle")

    assert len(_ops(trace, "conv2d")) == 1
    assert not _ops(trace, "functional.pad")
    assert PaddleBackend().validate_trace(trace) is True


def test_conv_layer_corrupted_output_fails_naming_the_op() -> None:
    """A tampered conv payload fails validation with the op label and replay_mismatch."""

    paddle.seed(0)
    trace = tl.trace(_ConvNet(2), _conv_input(2), backend="paddle")
    conv = _ops(trace, "conv2d")[0]
    conv.out = conv.out + 1.0

    status = _protocol().validate(trace, None)
    assert status.state == "failed"
    failed = [item for item in status.decisions if item["decision"] == "failed"]
    assert failed[0]["op_label"] == conv.layer_label
    assert failed[0]["reason"] == "replay_mismatch"
    assert conv.layer_label in status.message


def test_conv_wrappers_are_removed_after_capture() -> None:
    """The layer ``forward`` slots are restored once the capture ends."""

    originals = {cls: cls.forward for cls in (paddle.nn.Conv1D, paddle.nn.Conv2D)}
    tl.trace(_ConvNet(1), _conv_input(1), backend="paddle")

    assert {cls: cls.forward for cls in originals} == originals


_W44 = np.linspace(-0.5, 0.5, 16, dtype="float32").reshape(4, 4)
_B4 = np.array([0.1, -0.2, 0.3, 0.0], dtype="float32")


def _closure_loop(steps: int) -> Any:
    """``v = relu(v @ W + b)`` repeated with closure ``W`` and ``b`` (the F4b models)."""

    weight = paddle.to_tensor(_W44)
    bias = paddle.to_tensor(_B4)

    def loop(v: Any) -> Any:
        for _ in range(steps):
            v = _F.relu(paddle.matmul(v, weight) + bias)
        return v

    return loop


@pytest.mark.parametrize("steps", [1, 3])
def test_closure_constants_are_recorded_and_validate(steps: int) -> None:
    """Closure tensors become parentless constant ops with payloads; validation passes."""

    trace = tl.trace(_closure_loop(steps), _x24(), backend="paddle")

    constants = _ops(trace, "constant")
    assert len(constants) == 2
    assert all(list(op.parents) == [] for op in constants)
    payloads = sorted((op.out.numpy() for op in constants), key=lambda array: array.ndim)
    np.testing.assert_array_equal(payloads[0], _B4)
    np.testing.assert_array_equal(payloads[1], _W44)
    matmul = next(op for op in trace._layer_list if op.type == "matmul")
    assert "constant" in _parent_types(trace, matmul)
    status = _protocol().validate(trace, None)
    assert status.state == "passed", status.message
    assert status.exempted_reason_counts == {"constant_producer_leaf": 2}
    assert not set(status.unverified_reason_counts) - VALIDATION_REASONS


def test_literal_producer_is_replayed() -> None:
    """``paddle.ones`` is a parentless literal producer that validation replays (F5 shape)."""

    def reshape_t(v: Any) -> Any:
        return paddle.matmul(
            paddle.transpose(paddle.reshape(v, [4, 2]), [1, 0]), paddle.ones([4, 1])
        )

    trace = tl.trace(reshape_t, _x24(), backend="paddle")

    ones = _ops(trace, "ones")[0]
    assert list(ones.parents) == []
    status = _protocol().validate(trace, None)
    assert status.state == "passed", status.message
    replayed = {item["op_label"] for item in status.decisions if item["decision"] == "validated"}
    assert ones.layer_label in replayed


def test_untraced_intermediate_fails_naming_consumer_and_reason() -> None:
    """A tensor built by an unwrapped op during the forward is never a constant."""

    def erf_relu(v: Any) -> Any:
        return _F.relu(paddle.erf(v))

    trace = tl.trace(erf_relu, _x24(), backend="paddle")

    assert not _ops(trace, "constant")
    status = _protocol().validate(trace, None)
    relu = _ops(trace, "functional.relu")[0]
    assert status.state == "failed"
    assert status.decisions[0]["op_label"] == relu.layer_label
    assert status.decisions[0]["reason"] == "parent_payload_missing"
    assert relu.layer_label in status.message


def test_constant_payload_tamper_fails_the_consumer() -> None:
    """Corrupting a constant's saved payload makes its consumer's replay disagree."""

    trace = tl.trace(_closure_loop(1), _x24(), backend="paddle")
    weight = next(op for op in _ops(trace, "constant") if op.out.ndim == 2)
    weight.out = weight.out + 1.0

    status = _protocol().validate(trace, None)
    assert status.state == "failed"
    assert status.decisions[0]["decision"] in {"exempted", "failed"}
    first_failed = next(item for item in status.decisions if item["decision"] == "failed")
    assert first_failed["reason"] == "replay_mismatch"


def test_protocol_validate_returns_status_not_bool() -> None:
    """``protocol.validate`` returns the status object; ``tl.validate`` keeps its bool."""

    trace = tl.trace(_closure_loop(1), _x24(), backend="paddle")

    status = _protocol().validate(trace, None)
    assert not isinstance(status, bool)
    assert status.state == "passed"


def test_paddle_reasons_are_registered_or_pending() -> None:
    """Every Paddle reason is in VALIDATION_REASONS except the one awaiting registration."""

    assert {REASON_METADATA_INVARIANT} >= PADDLE_VALIDATION_REASONS - VALIDATION_REASONS


def _declared_code(exc: BackendUnsupportedError) -> object:
    """The code a raise site passed as ``code=``.

    ``BackendRegistryError`` keeps a per-raise ``code=`` in ``exc.fields`` while
    ``exc.code`` still reads the class default (registry fix requested from the
    protocol owner); this reads the declared value the raise site wrote.
    """

    return exc.fields.get("code")


def test_refusal_codes_registered_and_raised() -> None:
    """The shell's codes are registered; the seed and identity-mode raises carry them."""

    codes = _protocol().refusal_codes()
    assert codes <= REFUSAL_CODES
    assert paddle_package_refusal_codes() <= codes
    assert not (PADDLE_CODES_PENDING_REGISTRATION & REFUSAL_CODES) or (
        codes >= PADDLE_CODES_PENDING_REGISTRATION & REFUSAL_CODES
    )
    assert {"backward_capture_unsupported", "rng_replay_unsupported"} <= codes
    backend = PaddleBackend()
    with pytest.raises(BackendUnsupportedError) as seed_error:
        backend.capture_trace(_closure_loop(1), (_x24(),), random_seed=3)
    assert _declared_code(seed_error.value) == "explicit_seed_unsupported"
    with pytest.raises(BackendUnsupportedError) as mode_error:
        backend.capture_trace(_closure_loop(1), (_x24(),), module_identity_mode="object_module")
    assert _declared_code(mode_error.value) == "module_identity_mode_unsupported"


def test_host_copy_keeps_zero_dim_shape() -> None:
    """``host_copy`` of a 0-d tensor stays 0-d (F3)."""

    value = paddle.to_tensor(np.float32(2.5))
    copy = _protocol().host_copy(paddle.sum(value))
    assert copy.shape == ()
    assert float(copy) == 2.5
