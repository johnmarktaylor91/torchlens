"""The ``non_torch_backward_inert`` contract for backends that declare backward capture.

A non-torch backend whose spec declares ``backward_capture`` records backward
through the neutral builder; ``check_metadata_invariants`` then requires its
records to be consistent instead of absent. Each arm is driven alone on a stub
trace so a disarmed arm cannot hide behind an earlier one, and the dispatch is
checked both ways: a backend that does not declare backward capture (or that
the registry does not know) still runs every inert arm.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from torchlens.errors import MetadataInvariantError
from torchlens.validation import _invariants_entry
from torchlens.validation._invariants_entry import _check_non_torch_backward_inert
from torchlens.validation._invariants_native_backward import check_declared_backward_records

_NAME = "non_torch_backward_inert"


def _call(pass_number: int) -> SimpleNamespace:
    """A recorded grad-fn call in backward pass ``pass_number``."""

    return SimpleNamespace(backward_pass_number=pass_number)


def _grad_fn(label: str, *pass_numbers: int) -> SimpleNamespace:
    """A grad-fn record with one call per pass number given."""

    calls = {index: _call(number) for index, number in enumerate(pass_numbers, start=1)}
    return SimpleNamespace(label=label, calls=calls)


def _pass(*root_ids: int) -> SimpleNamespace:
    """A recorded backward pass started from ``root_ids``."""

    return SimpleNamespace(root_grad_fn_ids=list(root_ids))


def _consistent(**overrides: Any) -> SimpleNamespace:
    """A declaring backend's trace after one consistent backward pass."""

    fields: dict[str, Any] = {
        "_provenance_backend": "declaring",
        "_autograd_has_backward_pass": True,
        "_grad_fn_logs": {11: _grad_fn("mul_back_1_1", 1), 12: _grad_fn("sum_back_1_2", 1)},
        "_autograd_grad_fn_order": [11, 12],
        "_backward_pass_logs": {1: _pass(12)},
        "_autograd_backward_root_grad_fn_object_ids": [12],
        "_num_backward_passes": 1,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.fixture
def declaring(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every registry lookup report a backend that declares backward capture."""

    spec = SimpleNamespace(capabilities=SimpleNamespace(backward_capture=True))
    monkeypatch.setattr(_invariants_entry, "get_backend_spec", lambda _name: spec)


def test_consistent_records_pass(declaring: None) -> None:
    """One pass whose grad fns all fired in it satisfies the declared contract."""

    _check_non_torch_backward_inert(_consistent())  # type: ignore[arg-type]


def test_a_declaring_trace_with_no_backward_passes(declaring: None) -> None:
    """Before any backward call a declaring backend's trace carries no records."""

    trace = _consistent(
        _autograd_has_backward_pass=False,
        _grad_fn_logs={},
        _autograd_grad_fn_order=[],
        _backward_pass_logs={},
        _autograd_backward_root_grad_fn_object_ids=[],
        _num_backward_passes=0,
    )
    _check_non_torch_backward_inert(trace)  # type: ignore[arg-type]


def test_a_root_grad_fn_without_a_call_is_owned_by_its_pass(declaring: None) -> None:
    """A pass's root is owned by that pass even when it recorded no call."""

    trace = _consistent(_grad_fn_logs={11: _grad_fn("mul_back_1_1", 1), 12: _grad_fn("sum")})
    _check_non_torch_backward_inert(trace)  # type: ignore[arg-type]


def test_fires_on_a_grad_fn_no_recorded_pass_owns(declaring: None) -> None:
    """A grad fn with no call that is no pass's root is not owned by a recorded pass."""

    logs = {11: _grad_fn("mul_back_1_1", 1), 12: _grad_fn("sum_back_1_2", 1), 13: _grad_fn("x")}
    with pytest.raises(MetadataInvariantError, match="not owned by a recorded backward pass"):
        _check_non_torch_backward_inert(_consistent(_grad_fn_logs=logs))  # type: ignore[arg-type]


def test_fires_on_a_call_in_an_unrecorded_pass(declaring: None) -> None:
    """A grad-fn call naming a pass that has no record fails."""

    logs = {11: _grad_fn("mul_back_1_1", 1, 2), 12: _grad_fn("sum_back_1_2", 1)}
    with pytest.raises(MetadataInvariantError, match=r"calls in backward pass\(es\) \['2'\]"):
        _check_non_torch_backward_inert(_consistent(_grad_fn_logs=logs))  # type: ignore[arg-type]


def test_fires_when_the_pass_count_exceeds_the_records(declaring: None) -> None:
    """``num_backward_passes`` larger than the recorded passes fails."""

    with pytest.raises(MetadataInvariantError, match="num_backward_passes=2 disagrees"):
        _check_non_torch_backward_inert(_consistent(_num_backward_passes=2))  # type: ignore[arg-type]


def test_fires_when_the_pass_count_is_below_the_records(declaring: None) -> None:
    """A recorded pass beyond ``num_backward_passes`` fails."""

    trace = _consistent(_backward_pass_logs={1: _pass(12), 2: _pass()})
    with pytest.raises(MetadataInvariantError, match="num_backward_passes=1 disagrees"):
        _check_non_torch_backward_inert(trace)  # type: ignore[arg-type]


def test_fires_on_pass_numbers_that_are_not_dense(declaring: None) -> None:
    """Pass records numbered 2 alone with a count of 1 fail, as PyTorch's density check does."""

    logs = {11: _grad_fn("mul_back_1_1", 2), 12: _grad_fn("sum_back_1_2", 2)}
    trace = _consistent(_grad_fn_logs=logs, _backward_pass_logs={2: _pass(12)})
    with pytest.raises(MetadataInvariantError, match=r"numbers \[2\]"):
        _check_non_torch_backward_inert(trace)  # type: ignore[arg-type]


def test_fires_when_records_exist_but_has_backward_pass_is_false(declaring: None) -> None:
    """Pass records with ``has_backward_pass`` false fail."""

    trace = _consistent(_autograd_has_backward_pass=False)
    with pytest.raises(MetadataInvariantError, match="has_backward_pass is false"):
        _check_non_torch_backward_inert(trace)  # type: ignore[arg-type]


def test_fires_when_has_backward_pass_is_true_without_records(declaring: None) -> None:
    """``has_backward_pass`` true with no pass record fails."""

    trace = _consistent(_grad_fn_logs={}, _backward_pass_logs={}, _num_backward_passes=0)
    with pytest.raises(MetadataInvariantError, match="has_backward_pass is true"):
        _check_non_torch_backward_inert(trace)  # type: ignore[arg-type]


def test_fires_on_grad_fn_records_without_any_pass_record(declaring: None) -> None:
    """Grad-fn records beside no pass record fail even with the flag and count at rest."""

    trace = _consistent(
        _autograd_has_backward_pass=False, _backward_pass_logs={}, _num_backward_passes=0
    )
    with pytest.raises(MetadataInvariantError, match="no backward pass record is present"):
        _check_non_torch_backward_inert(trace)  # type: ignore[arg-type]


def test_the_declared_arms_report_under_the_contract_name() -> None:
    """Every declared-arm failure names ``non_torch_backward_inert``."""

    with pytest.raises(MetadataInvariantError, match=_NAME):
        check_declared_backward_records(_consistent(_num_backward_passes=3), _NAME)


def test_a_backend_that_does_not_declare_backward_stays_inert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Consistent records on a backend without ``backward_capture`` still fail inert."""

    spec = SimpleNamespace(capabilities=SimpleNamespace(backward_capture=False))
    monkeypatch.setattr(_invariants_entry, "get_backend_spec", lambda _name: spec)
    with pytest.raises(MetadataInvariantError, match="must not set has_backward_pass"):
        _check_non_torch_backward_inert(_consistent())  # type: ignore[arg-type]


@pytest.mark.parametrize("backend", ["mlx", "not-a-registered-backend", "", None])
def test_undeclared_unknown_or_missing_backends_run_the_inert_arms(backend: Any) -> None:
    """MLX (no backward capture), an unregistered name and no name all fail closed."""

    trace = _consistent(_provenance_backend=backend, _autograd_has_backward_pass=False)
    with pytest.raises(MetadataInvariantError, match="must not populate _grad_fn_logs"):
        _check_non_torch_backward_inert(trace)  # type: ignore[arg-type]
