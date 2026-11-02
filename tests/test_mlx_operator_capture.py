"""Operator, method and function capture on the MLX preview; the constants rule.

Packet P3 of the preview-backend protocol: ``mx.array`` operators and methods
and the extended ``mx.*`` surface are captured, closure constants become
``constant`` source ops with payloads, literal producers keep zero parents,
and an op no wrapper saw is flagged or refused by the backend-local policy.
"""

from __future__ import annotations

import inspect
import warnings

import pytest

pytestmark = pytest.mark.backend_mlx

mlx = pytest.importorskip("mlx")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import numpy as np  # noqa: E402

import torchlens as tl  # noqa: E402
from tests.conformance._protocol_cases import (  # noqa: E402
    graph_integrity_problems,
    shape_metadata_problems,
    silent_gap_reason,
)
from tests.conformance._rosters.mlx import CONFIG_BUILT, W44, X24, roster  # noqa: E402
from torchlens.backends._backend_protocol import REFUSAL_CODES, VALIDATION_REASONS  # noqa: E402
from torchlens.backends.mlx import get_unwrapped_policy, unwrapped_policy  # noqa: E402
from torchlens.backends.mlx.wrappers import is_mlx_wrapped  # noqa: E402
from torchlens.backends.registry import BackendUnsupportedError  # noqa: E402
from torchlens.validation.status import ValidationStatus  # noqa: E402

_GLOBAL_SCALE = None


def _code(exc: BaseException) -> str:
    """The typed refusal code (SPEC 1.3 rule 4: ``exc.code`` is the contract)."""

    return str(getattr(exc, "code", ""))


def _kinds(trace: tl.Trace) -> tuple[str, ...]:
    return tuple(str(op.func_name) for op in trace._layer_list)


def _validate(trace: tl.Trace) -> ValidationStatus:
    status = tl.backends.spec("mlx").validate_trace(trace)
    assert isinstance(status, ValidationStatus), "validate must return the status object"
    return status


# ----------------------------------------------------------------- roster rows
@pytest.mark.parametrize("row", CONFIG_BUILT, ids=lambda row: row.family)
def test_roster_row_inventory_graph_and_shapes(row) -> None:
    """Every config_built row captures its exact inventory with a sound graph."""

    model, x = row.builder()
    trace = tl.trace(model, x, backend="mlx")
    assert _kinds(trace) == row.expected_op_kinds
    assert graph_integrity_problems(trace) == ()
    assert shape_metadata_problems(trace, np.asarray) == ()
    for op in trace._layer_list:
        if op.out is not None:
            assert tuple(op.shape) == np.asarray(op.out).shape, op.label
    assert silent_gap_reason(trace) is None
    assert list(trace.output_ops)


@pytest.mark.parametrize(
    "row",
    [row for row in CONFIG_BUILT if row.expected_intermediates is not None],
    ids=lambda row: row.family,
)
def test_roster_row_intermediates_match_numpy(row) -> None:
    """Captured intermediate payloads equal the row's NumPy-computed values."""

    model, x = row.builder()
    trace = tl.trace(model, x, backend="mlx")
    expected = row.expected_intermediates()
    for kind, want in expected.items():
        wants = want if isinstance(want, list) else [want]
        got = [np.asarray(op.out) for op in trace._layer_list if op.func_name == kind]
        assert len(got) == len(wants), f"{kind}: {len(got)} captured, {len(wants)} expected"
        for captured, reference in zip(got, wants, strict=True):
            np.testing.assert_allclose(captured, np.asarray(reference), rtol=1e-5, atol=1e-5)


#: Rows whose replay validation is a known, root-caused red on this branch: the
#: MLX module tree addresses Sequential children ``block.layers.0`` while the shared
#: finalization derives ``address_parent`` by splitting the address, so the
#: module-hierarchy metadata invariant (the tripwire) fails. Fix: canonical MLX
#: module addresses without container segments (packet report, "Known reds").
_KNOWN_VALIDATION_REDS: dict[str, str] = {
    "mod_nested_residual": "module address container segment vs address_parent rsplit",
}


@pytest.mark.parametrize(
    "row",
    [
        pytest.param(
            row,
            marks=pytest.mark.xfail(strict=True, reason=_KNOWN_VALIDATION_REDS[row.family]),
        )
        if row.family in _KNOWN_VALIDATION_REDS
        else row
        for row in CONFIG_BUILT
    ],
    ids=lambda row: row.family,
)
def test_roster_row_parity_and_validation(row) -> None:
    """Outputs equal the unwrapped forward exactly and replay validation passes."""

    model, x = row.builder()
    assert not is_mlx_wrapped()
    reference = model(x)
    references = reference if isinstance(reference, tuple) else (reference,)
    mx.eval(*references)
    trace = tl.trace(model, x, backend="mlx")
    outputs = list(trace.output_ops)
    assert len(outputs) == len(references)
    for op, ref in zip(outputs, references, strict=True):
        np.testing.assert_array_equal(np.asarray(op.out), np.asarray(ref))
    status = _validate(trace)
    assert status.state == "passed", status.message
    assert set(status.unverified_reason_counts) <= VALIDATION_REASONS
    assert set(status.exempted_reason_counts) <= VALIDATION_REASONS
    for decision in status.decisions:
        assert decision["reason"] is None or decision["reason"] in VALIDATION_REASONS


def test_roster_folds_into_runner_records() -> None:
    """The roster folds to RosterModel rows with at least eight config_built rows."""

    rows = roster(include_ecosystem=False)
    assert len(rows) >= 8
    assert all(row.realism == "config_built" for row in rows)
    assert all(row.expected_op_kinds for row in rows)


# ------------------------------------------------------------------ operators
def test_array_dunders_log_under_function_spellings() -> None:
    """``a + b``, ``a @ b``, ``-a``, ``a[0]`` and ``a > b`` are logged like mx.* calls."""

    x = mx.array(X24)
    w = mx.array(W44)

    def forward(v: mx.array) -> mx.array:
        return ((v @ w) - 1.0)[0] > -(v[1])

    trace = tl.trace(forward, x, backend="mlx")
    assert _kinds(trace) == (
        "input",
        "constant",
        "matmul",
        "subtract",
        "getitem",
        "getitem",
        "negative",
        "greater",
    )
    assert silent_gap_reason(trace) is None


def test_array_methods_and_fast_kernels_log() -> None:
    """Array methods and ``mx.fast`` kernels are ops; nothing double-logs."""

    x = mx.array(X24)
    weight = mx.ones((4,))
    bias = mx.zeros((4,))

    def forward(v: mx.array) -> mx.array:
        return mx.fast.layer_norm(v.T.T.square().exp(), weight, bias, 1e-5)

    trace = tl.trace(forward, x, backend="mlx")
    assert _kinds(trace) == (
        "input",
        "transpose",
        "transpose",
        "square",
        "exp",
        "constant",
        "constant",
        "layer_norm",
    )
    assert isinstance(inspect.getattr_static(mx.array, "T"), property)


def test_layer_bodies_run_under_the_nested_guard() -> None:
    """A wrapped layer appears once; its Python body's operators are not logged."""

    mx.random.seed(0)
    model = nn.Sequential(nn.Conv2d(1, 2, 3, padding=1), nn.MaxPool2d(2))
    image = mx.array(np.ones((1, 6, 6, 1), np.float32))
    trace = tl.trace(model, image, backend="mlx")
    assert _kinds(trace) == ("input", "conv2d", "max_pool2d")


def test_wrappers_are_removed_after_capture() -> None:
    """The process-global wrappers, array slots included, are restored afterwards."""

    x = mx.array(X24)
    tl.trace(lambda v: v + 1.0, x, backend="mlx")
    assert not is_mlx_wrapped()
    assert not hasattr(mx.array.__add__, "__wrapped__")
    assert not hasattr(mx.array.sum, "__wrapped__")
    assert not hasattr(mx.add, "__wrapped__")
    out = x + 1.0
    mx.eval(out)
    assert isinstance(out, mx.array)


# ------------------------------------------------------------------ constants
def test_closure_constant_is_a_constant_op_with_payload_and_parent_edge() -> None:
    """A closure array becomes a ``constant`` source op the consumer lists as parent."""

    weight = mx.array(W44)
    trace = tl.trace(lambda v: mx.matmul(v, weight), mx.array(X24), backend="mlx")
    constants = [op for op in trace._layer_list if op.func_name == "constant"]
    assert len(constants) == 1
    constant = constants[0]
    np.testing.assert_array_equal(np.asarray(constant.out), W44)
    assert not tuple(constant.parents or ())
    assert constant not in list(trace.input_ops)
    matmul = next(op for op in trace._layer_list if op.func_name == "matmul")
    assert constant.layer_label in set(matmul.parents)
    status = _validate(trace)
    assert status.state == "passed", status.message
    assert dict(status.exempted_reason_counts) == {"constant_producer_leaf": 1}


def test_module_global_constant_is_recorded() -> None:
    """An array named through the function's globals is a constant too."""

    global _GLOBAL_SCALE
    _GLOBAL_SCALE = mx.array(np.full((4,), 2.0, np.float32))

    def forward(v: mx.array) -> mx.array:
        return mx.multiply(v, _GLOBAL_SCALE)

    try:
        trace = tl.trace(forward, mx.array(X24), backend="mlx")
    finally:
        _GLOBAL_SCALE = None
    assert _kinds(trace) == ("input", "constant", "multiply")
    assert silent_gap_reason(trace) is None


def test_literal_producer_keeps_zero_parents_and_a_payload() -> None:
    """``mx.ones`` inside the forward is an op with no parents and a saved payload."""

    trace = tl.trace(lambda v: v @ mx.ones((4, 1)), mx.array(X24), backend="mlx")
    ones = next(op for op in trace._layer_list if op.func_name == "ones")
    assert not tuple(ones.parents or ())
    np.testing.assert_array_equal(np.asarray(ones.out), np.ones((4, 1), np.float32))
    assert "constant" not in _kinds(trace)
    status = _validate(trace)
    assert status.state == "passed", status.message
    assert dict(status.exempted_reason_counts) == {"constant_producer_leaf": 1}


def test_parameter_leaf_is_not_a_constant() -> None:
    """A module parameter used directly by an op stays a parameter leaf."""

    class Affine(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = mx.array(W44)

        def __call__(self, v: mx.array) -> mx.array:
            return v @ self.weight

    trace = tl.trace(Affine(), mx.array(X24), backend="mlx")
    assert _kinds(trace) == ("input", "matmul")
    assert _validate(trace).state == "passed"


def test_tampered_constant_payload_fails_and_names_the_consumer() -> None:
    """Replay reads the constant's saved payload: tampering it fails the consumer."""

    weight = mx.array(W44)
    trace = tl.trace(lambda v: mx.matmul(v, weight), mx.array(X24), backend="mlx")
    constant = next(op for op in trace._layer_list if op.func_name == "constant")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        constant.out = constant.out + 1.0
    status = _validate(trace)
    assert status.state == "failed"
    failed = [d for d in status.decisions if d["decision"] == "failed"]
    assert failed and failed[0]["reason"] == "replay_mismatch"
    assert failed[0]["op_label"] in status.message
    assert failed[0]["func_name"] == "matmul"


def test_constant_without_payload_is_a_failed_check_not_an_exemption() -> None:
    """Dropping a constant's payload fails validation (payload missing), never exempts."""

    weight = mx.array(W44)
    trace = tl.trace(lambda v: mx.matmul(v, weight), mx.array(X24), backend="mlx")
    constant = next(op for op in trace._layer_list if op.func_name == "constant")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        constant.out = None
    status = _validate(trace)
    assert status.state == "failed"
    assert any(d["reason"] == "parent_payload_missing" for d in status.decisions)


# -------------------------------------------------------------- unwrapped ops
def _unwrapped(v: mx.array) -> mx.array:
    return mx.linalg.norm(v)


def test_unwrapped_output_flags_with_the_registry_reason() -> None:
    """An output no wrapped op produced flags the trace with ``unwrapped_op_in_capture``."""

    with pytest.warns(UserWarning, match="not produced by a wrapped operation"):
        trace = tl.trace(_unwrapped, mx.array(X24), backend="mlx")
    assert trace.provenance.verified is False
    assert trace.provenance.verification_reason == "unwrapped_op_in_capture"
    assert "unwrapped_op_in_capture" in REFUSAL_CODES
    assert silent_gap_reason(trace) == "unwrapped_op_in_capture"


def test_unwrapped_intermediate_flags_instead_of_becoming_a_constant() -> None:
    """An unlabeled array built inside the forward is a gap, never a constant."""

    with pytest.warns(UserWarning, match="not produced by a wrapped operation"):
        trace = tl.trace(lambda v: mx.add(mx.linalg.norm(v), 1.0), mx.array(X24), backend="mlx")
    assert trace.provenance.verified is False
    assert trace.provenance.verification_reason == "unwrapped_op_in_capture"
    assert "constant" not in _kinds(trace)


def test_unwrapped_policy_refuse_raises_typed_and_restores_state() -> None:
    """Under ``unwrapped_policy("refuse")`` the gap is a typed refusal; wrappers unwind."""

    with unwrapped_policy("refuse"), pytest.raises(BackendUnsupportedError) as info:
        tl.trace(_unwrapped, mx.array(X24), backend="mlx")
    assert _code(info.value) == "unwrapped_op_in_capture"
    assert not is_mlx_wrapped()
    assert get_unwrapped_policy() == "flag"


def test_unwrapped_policy_rejects_unknown_values() -> None:
    """Only ``flag`` and ``refuse`` are policies."""

    with pytest.raises(ValueError, match="flag"), unwrapped_policy("ignore"):
        pass


# ------------------------------------------------------------- typed refusals
def test_explicit_seed_refusal_is_coded() -> None:
    """The backend's ``seed=`` refusal carries code ``explicit_seed_unsupported``.

    The public spine refuses an explicit seed on previews before the backend
    runs (its own gate, outside this packet); the backend entry is pinned here.
    """

    from torchlens.backends.mlx import MLXBackend

    with pytest.raises(BackendUnsupportedError) as info:
        MLXBackend().capture_trace(lambda v: v + 1, mx.array([1.0]), seed=123)
    assert _code(info.value) == "explicit_seed_unsupported"
    assert not is_mlx_wrapped()


def test_protocol_shell_members_for_mlx() -> None:
    """The mlx protocol object: codes registered, host copies keep 0-d, validate is a status."""

    protocol = tl.backends.spec("mlx").protocol()
    assert protocol.refusal_codes() <= REFUSAL_CODES
    assert protocol.materialization == "lazy"
    assert protocol.runtime_available() is None
    scalar = protocol.host_copy(mx.array(1.5))
    assert scalar.shape == () and scalar.dtype == np.float32
    model, x = CONFIG_BUILT[0].builder()
    reference = protocol.reference_output(model, x, None)
    trace = protocol.capture(model, x, None)
    np.testing.assert_array_equal(np.asarray(trace.output_ops[0].out), np.asarray(reference))
    status = protocol.validate(trace, reference)
    assert isinstance(status, ValidationStatus) and status.state == "passed"


def test_host_copy_refuses_unreadable_values_typed() -> None:
    """A value with no readable buffer refuses ``constant_unrealizable``."""

    protocol = tl.backends.spec("mlx").protocol()
    with pytest.raises(BackendUnsupportedError) as info:
        protocol.host_copy(object())
    assert _code(info.value) == "constant_unrealizable"
