"""tinygrad backend validation tripwire tests."""

from __future__ import annotations

from typing import Any

import pytest

import torchlens as tl
from torchlens.backends import BackendUnsupportedError
from torchlens.backends.tinygrad import TinygradBackend

tinygrad = pytest.importorskip("tinygrad")
Tensor = pytest.importorskip("tinygrad").Tensor


pytestmark = pytest.mark.backend_tinygrad


def _validation_block(x: Any) -> Any:
    """Return a tinygrad expression with non-commutative parent dependencies.

    Parameters
    ----------
    x
        tinygrad Tensor input.

    Returns
    -------
    Any
        tinygrad Tensor output.
    """

    shifted = x + 1.0
    return shifted.where(shifted > 0.0, x - 1.0).sum()


def _trace() -> Any:
    """Return a fresh tinygrad validation trace.

    Returns
    -------
    Any
        TorchLens trace captured with tinygrad.
    """

    return tl.trace(_validation_block, Tensor([1.0, -2.0, 3.0]), backend="tinygrad")


def _first_op(trace: Any, layer_type: str) -> Any:
    """Return the first operation with ``layer_type``.

    Parameters
    ----------
    trace
        TorchLens trace.
    layer_type
        Layer type to find.

    Returns
    -------
    Any
        Matching operation.
    """

    return next(op for op in trace.layer_list if op.layer_type == layer_type)


def test_tinygrad_validate_entry_and_trace_return_honest_bool() -> None:
    """Validate tinygrad through public entry and trace registry dispatch."""

    x = Tensor([1.0, -2.0, 3.0])
    assert tl.validate(_validation_block, x, scope="forward", backend="tinygrad") is True

    trace = tl.trace(_validation_block, x, backend="tinygrad")
    assert trace.validate_forward_pass([_validation_block(x)]) is True
    assert TinygradBackend().validate_trace(trace) is True


def test_tinygrad_validation_fails_corrupted_saved_output() -> None:
    """Fail validation when a materialized op output payload is corrupted."""

    trace = _trace()
    op = _first_op(trace, "add")
    op.out = Tensor([99.0, 99.0, 99.0]).realize()

    assert trace.validate_forward_pass([_validation_block(Tensor([1.0, -2.0, 3.0]))]) is False


def test_tinygrad_validation_fails_wrong_parent_wiring() -> None:
    """Fail validation when a child is rewired to the wrong saved parent."""

    trace = _trace()
    cmplt = _first_op(trace, "cmplt")
    args = cmplt.parent_arg_positions["args"]
    first_parent = args[0]
    args[1] = first_parent
    cmplt.parents = (first_parent,)

    assert trace.validate_forward_pass([_validation_block(Tensor([1.0, -2.0, 3.0]))]) is False


def test_tinygrad_validation_fails_dropped_payload() -> None:
    """Fail validation when a required parent payload has been dropped."""

    trace = _trace()
    op = _first_op(trace, "add")
    op.out = None

    assert trace.validate_forward_pass([_validation_block(Tensor([1.0, -2.0, 3.0]))]) is False


def test_tinygrad_audit_only_payload_validation_raises() -> None:
    """Raise unsupported when tinygrad payloads are audit-only instead of live."""

    trace = _trace()
    trace.tinygrad_uop_captures = ()
    trace.tinygrad_payload_policy = "audit_only"

    with pytest.raises(BackendUnsupportedError, match="audit-only|realized-copy"):
        trace.validate_forward_pass([_validation_block(Tensor([1.0, -2.0, 3.0]))])
