"""tinygrad backend validation tripwire tests."""

from __future__ import annotations

from typing import Any

import pytest

import torchlens as tl
from torchlens.backends.tinygrad import TinygradBackend
from torchlens.errors import BackendUnsupportedError

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

    return next(op for op in trace._layer_list if op.type == layer_type)


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

    with pytest.raises(BackendUnsupportedError, match="audit-only|realized-copy") as info:
        trace.validate_forward_pass([_validation_block(Tensor([1.0, -2.0, 3.0]))])
    assert info.value.code == "live_payloads_required"


def _failed_reasons(trace: Any) -> set[str]:
    """Return the reasons of every failed decision on the trace's replay status.

    Parameters
    ----------
    trace
        Validated tinygrad trace.

    Returns
    -------
    set[str]
        Closed-list reasons recorded with ``decision == "failed"``.
    """

    status = trace.validation_replay_status
    return {
        str(record["reason"])
        for record in status.decisions
        if record["verdict"] == "failed" and record.get("reason") is not None
    }


def _input_op(trace: Any) -> Any:
    """Return the trace's input operation.

    Parameters
    ----------
    trace
        TorchLens trace.

    Returns
    -------
    Any
        The input op.
    """

    return next(op for op in trace._layer_list if op.is_input)


def test_tinygrad_validation_fails_slightly_corrupted_saved_output() -> None:
    """A one percent drift in a saved payload is outside the float replay band."""

    trace = _trace()
    op = _first_op(trace, "add")
    op.out = (op.out * 1.01).realize()

    assert trace.validate_forward_pass([_validation_block(Tensor([1.0, -2.0, 3.0]))]) is False
    assert "replay_mismatch" in _failed_reasons(trace)


def test_tinygrad_validation_fails_saved_output_dtype_drift_with_equal_values() -> None:
    """A saved payload whose dtype drifted fails replay even when its values are exact."""

    from tinygrad import dtypes

    trace = _trace()
    op = _first_op(trace, "add")
    narrowed = op.out.cast(dtypes.half).realize()
    assert narrowed.tolist() == op.out.tolist()
    op.out = narrowed

    assert (
        trace.validate_forward_pass(
            [_validation_block(Tensor([1.0, -2.0, 3.0]))], validate_metadata=False
        )
        is False
    )
    assert "replay_mismatch" in _failed_reasons(trace)


def test_tinygrad_validation_fails_corrupted_input_payload() -> None:
    """Replay reads parent values from the trace, so a corrupted input payload fails."""

    trace = _trace()
    _input_op(trace).out = Tensor([5.0, 5.0, 5.0]).realize()

    assert trace.validate_forward_pass([_validation_block(Tensor([1.0, -2.0, 3.0]))]) is False
    assert "replay_mismatch" in _failed_reasons(trace)


def test_tinygrad_validation_fails_when_output_ignores_its_parent() -> None:
    """An op whose output does not depend on its parent fails the perturbation check."""

    def vanishing(x: Any) -> Any:
        """Return an output that is False for every input (one parent, both positions)."""

        return x != x

    trace = tl.trace(vanishing, Tensor([1.0, -2.0, 3.0]), backend="tinygrad")

    assert trace.validate_forward_pass([vanishing(Tensor([1.0, -2.0, 3.0]))]) is False
    assert "perturbation_insensitive" in _failed_reasons(trace)


def test_tinygrad_validation_fails_when_arg_positions_are_dropped() -> None:
    """Parents without argument positions cannot replay and fail closed by name."""

    trace = _trace()
    where_op = _first_op(trace, "where")
    assert where_op.parents
    where_op.parent_arg_positions["args"] = {}

    assert (
        trace.validate_forward_pass(
            [_validation_block(Tensor([1.0, -2.0, 3.0]))], validate_metadata=False
        )
        is False
    )
    assert "replay_mismatch" in _failed_reasons(trace)


def test_tinygrad_validation_fails_when_commutative_arg_positions_swap() -> None:
    """Swapping the positions of a commutative op's parents is a capture drift, not a match."""

    def two_parent_add(x: Any) -> Any:
        """Return a sum of two distinct graph parents."""

        return (x + 1.0) + (x * 2.0)

    trace = tl.trace(two_parent_add, Tensor([1.0, -2.0, 3.0]), backend="tinygrad")
    merge = next(op for op in trace._layer_list if op.type == "add" and len(set(op.parents)) == 2)
    args = merge.parent_arg_positions["args"]
    assert sorted(args) == [0, 1] and args[0] != args[1]
    args[0], args[1] = args[1], args[0]

    assert (
        trace.validate_forward_pass(
            [two_parent_add(Tensor([1.0, -2.0, 3.0]))], validate_metadata=False
        )
        is False
    )
    assert "replay_mismatch" in _failed_reasons(trace)
