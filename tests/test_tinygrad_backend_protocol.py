"""tinygrad adoption of the preview-backend protocol: refusals, verdicts, constants."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import torchlens as tl
from torchlens._validation_status import ValidationStatus
from torchlens.backends._backend_protocol import REFUSAL_CODES, VALIDATION_REASONS, BackendProtocol
from torchlens.backends._protocols.tinygrad import TinygradProtocol
from torchlens.backends.tinygrad._rules import (
    CONSTANT_KIND_ANNOTATION,
    ReplayVerdict,
    constant_kind_for_uop,
    describe_uop,
    tinygrad_constant_ops,
)
from torchlens.errors import BackendUnsupportedError

tinygrad = pytest.importorskip("tinygrad")
Tensor = pytest.importorskip("tinygrad").Tensor

pytestmark = pytest.mark.backend_tinygrad


def _block(x: Any) -> Any:
    """A small pure tinygrad expression with an add, a compare and a reduce."""

    return ((x + 1.0).relu() * 2.0).sum()


def _trace() -> Any:
    """Capture ``_block`` on a fixed input."""

    return tl.trace(_block, Tensor([1.0, -2.0, 3.0]), backend="tinygrad")


def _first_op(trace: Any, layer_type: str) -> Any:
    """The first op of ``layer_type`` in ``trace``."""

    return next(op for op in trace._layer_list if op.type == layer_type)


def test_protocol_object_satisfies_the_contract() -> None:
    """The registered factory returns a protocol object whose codes are registered."""

    protocol = tl.backends.spec("tinygrad").protocol()
    assert isinstance(protocol, BackendProtocol)
    assert isinstance(protocol, TinygradProtocol)
    assert protocol.materialization == "lazy"
    assert protocol.refusal_codes() <= REFUSAL_CODES
    assert protocol.runtime_available() is None


def test_protocol_validate_returns_status_never_bool() -> None:
    """``validate`` hands back the stored status object on a passing trace."""

    trace = _trace()
    status = TinygradProtocol().validate(trace, None)

    assert isinstance(status, ValidationStatus)
    assert status.state == "passed"
    assert status.decisions == ()
    assert "replayed" in status.message


def test_protocol_validate_names_op_and_reason_on_tamper() -> None:
    """A tampered payload yields a failed status naming the op and a registered reason."""

    trace = _trace()
    op = _first_op(trace, "add")
    op.out = Tensor([99.0, 99.0, 99.0]).realize()

    status = TinygradProtocol().validate(trace, None)

    assert status.state == "failed"
    assert len(status.decisions) == 1
    decision = status.decisions[0]
    assert decision["label"] == op._label_raw
    assert decision["reason"] in VALIDATION_REASONS
    assert op._label_raw in status.message
    # the public boolean contract of the spec entry is unchanged
    assert trace.validate_forward_pass([_block(Tensor([1.0, -2.0, 3.0]))]) is False


def test_protocol_validate_names_dropped_payload() -> None:
    """A dropped payload is reported as ``parent_payload_missing`` on that op."""

    trace = _trace()
    op = _first_op(trace, "add")
    op.out = None

    status = TinygradProtocol().validate(trace, None)

    assert status.state == "failed"
    assert status.decisions[0]["reason"] == "parent_payload_missing"
    assert status.decisions[0]["label"] == op._label_raw


def test_replay_verdict_decisions_and_message() -> None:
    """The verdict renders its decision row and message deterministically."""

    passed = ReplayVerdict.ok(3)
    assert passed.decisions() == ()
    assert "3 tinygrad ops" in passed.message()

    failed = ReplayVerdict.fail("add_1", "replay_mismatch", "differs", 1)
    (decision,) = failed.decisions()
    assert decision == {
        "label": "add_1",
        "verdict": "failed",
        "reason": "replay_mismatch",
        "detail": "differs",
    }
    assert "add_1" in failed.message() and "replay_mismatch" in failed.message()


def test_explicit_seed_refuses_with_registered_code() -> None:
    """An explicit seed on the preview refuses with a registered code (SPEC 1.3)."""

    with pytest.raises(BackendUnsupportedError) as info:
        tl.trace(
            _block,
            Tensor([1.0, 2.0]),
            backend="tinygrad",
            capture=tl.options.CaptureOptions(seed=3),
        )
    assert info.value.code in {"explicit_seed_unsupported", "rng_replay_unsupported"}
    assert info.value.code in REFUSAL_CODES


def test_module_identity_mode_refuses_with_registered_code() -> None:
    """An unknown ``module_identity_mode`` refuses ``module_identity_mode_unsupported``."""

    with pytest.raises(BackendUnsupportedError) as info:
        tl.trace(
            _block,
            Tensor([1.0, 2.0]),
            backend="tinygrad",
            capture=tl.options.CaptureOptions(module_identity_mode="nope"),
        )
    assert info.value.code == "module_identity_mode_unsupported"


def test_host_copy_keeps_shape_and_dtype() -> None:
    """``host_copy`` returns a NumPy copy with shape kept; 0-d stays 0-d."""

    protocol = TinygradProtocol()
    matrix = protocol.host_copy(Tensor([[1.0, 2.0], [3.0, 4.0]]).realize())
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (2, 2)
    np.testing.assert_allclose(matrix, [[1.0, 2.0], [3.0, 4.0]])

    scalar = protocol.host_copy(Tensor(2.5).realize())
    assert scalar.shape == ()
    assert float(scalar) == 2.5


def test_realize_once_realizes_every_listed_value() -> None:
    """``realize_once`` evaluates every listed lazy tensor in one call."""

    protocol = TinygradProtocol()
    a = Tensor([1.0, 2.0]) + 1.0
    b = (Tensor([3.0, 4.0]) * 2.0).sum()
    protocol.realize_once([a, b, "not a tensor"])
    np.testing.assert_allclose(a.numpy(), [2.0, 3.0])
    assert float(b.numpy()) == 14.0


def test_constant_kinds_are_annotated() -> None:
    """A list-built closure weight is a closure constant; ``Tensor.ones`` a literal producer."""

    weight = Tensor([[1.0, 0.0], [0.0, 1.0]])

    def forward(value: Any) -> Any:
        return value.matmul(weight) + Tensor.ones(1, 2)

    trace = tl.trace(forward, Tensor([[2.0, 3.0]]), backend="tinygrad")
    kinds = tinygrad_constant_ops(trace)

    assert "closure" in kinds.values()
    assert "literal" in kinds.values()
    for op in trace._layer_list:
        kind = (op.annotations or {}).get(CONSTANT_KIND_ANNOTATION)
        if kind is not None:
            assert op.parents == ()
            assert kind in ("closure", "literal")
            assert op.out is not None
    assert trace.validate_forward_pass([forward(Tensor([[2.0, 3.0]]))]) is True


def test_constant_kind_classifier_rules() -> None:
    """The classifier ignores ops with parents, inputs and parameters."""

    buffer_uop = Tensor([1.0, 2.0]).realize().uop
    assert constant_kind_for_uop(
        buffer_uop, parent_labels=(), is_input=False, is_parameter=False
    ) in ("closure", None)
    assert (
        constant_kind_for_uop(buffer_uop, parent_labels=("x",), is_input=False, is_parameter=False)
        is None
    )
    assert (
        constant_kind_for_uop(buffer_uop, parent_labels=(), is_input=True, is_parameter=False)
        is None
    )
    assert (
        constant_kind_for_uop(buffer_uop, parent_labels=(), is_input=False, is_parameter=True)
        is None
    )
    assert "shape=" in describe_uop(buffer_uop)
