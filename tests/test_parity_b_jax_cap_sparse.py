"""JAX selective ``save=`` decided at capture: sparse payloads equal a full capture's.

A static-label selector makes the JAX capture keep only the selected equations'
values. The sparse trace must have the same ops and edges as a full capture,
equal payloads on the selected ops, no retained value for an unselected
equation, and the validation state torch gives the same selective save.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import torchlens as tl  # noqa: E402
from torchlens.backends.jax import (
    _sparse_capture,  # noqa: E402
    backend as jax_backend,  # noqa: E402
)
from torchlens.backends.jax._trace_graph_helpers import _data_parent_labels  # noqa: E402


def _fn(x: Any) -> Any:
    return jnp.tanh(jnp.sin(x) * 2.0 + 1.0)


def _two_tanh(x: Any) -> Any:
    hidden = jnp.tanh(x * 3.0)
    return jnp.tanh(jnp.cos(hidden) - 0.5)


class _TorchTwoTanh(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(x * 3.0)
        return torch.tanh(torch.cos(hidden) - 0.5)


def _x() -> Any:
    return jnp.asarray(np.linspace(-1.0, 1.0, 12, dtype=np.float32).reshape(3, 4))


def _ops_by_raw(trace: Any) -> dict[str, Any]:
    return {
        str(op._label_raw): op
        for op in trace._layer_list
        if getattr(op, "_label_raw", None) is not None
    }


def _spy_results(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    results: list[Any] = []
    interpret = jax_backend.interpret_closed_jaxpr_with_inlining

    def spying_interpret(*args: Any, **kwargs: Any) -> Any:
        result = interpret(*args, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(jax_backend, "interpret_closed_jaxpr_with_inlining", spying_interpret)
    return results


@pytest.mark.parametrize("fn", [_fn, _two_tanh], ids=["one_tanh", "two_tanh"])
def test_sparse_payloads_equal_full_on_selected_ops(fn: Any) -> None:
    """Same ops and edges as a full capture; every selected payload is equal."""

    x = _x()
    full = tl.trace(fn, x, backend="jax")
    sparse = tl.trace(fn, x, save=tl.func("tanh"), backend="jax")

    full_ops, sparse_ops = _ops_by_raw(full), _ops_by_raw(sparse)
    assert list(sparse_ops) == list(full_ops)
    assert [op.label for op in sparse._layer_list] == [op.label for op in full._layer_list]
    selected = [raw for raw, op in sparse_ops.items() if op.func_name == "tanh"]
    assert selected
    for raw, op in sparse_ops.items():
        assert sorted(_data_parent_labels(op)) == sorted(_data_parent_labels(full_ops[raw]))
    for raw in selected:
        op = sparse_ops[raw]
        assert op.has_saved_activation and op.out is not None, raw
        np.testing.assert_array_equal(np.asarray(op.out), np.asarray(full_ops[raw].out))
    for raw, op in sparse_ops.items():
        if op.func_name not in ("tanh", "input") and not raw.startswith(("input", "output")):
            assert op.out is None, raw


def test_sparse_capture_never_retains_unselected_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """The value pass releases exactly the unselected primitive equations."""

    results = _spy_results(monkeypatch)
    reasons: list[Any] = []
    reason_of = _sparse_capture.ineligible_reason

    def recording_reason(*args: Any) -> Any:
        reasons.append((reason_of(*args), sorted(args[1])))
        return reasons[-1][0]

    monkeypatch.setattr(_sparse_capture, "ineligible_reason", recording_reason)
    sparse = tl.trace(_fn, _x(), save=tl.func("tanh"), backend="jax")

    assert len(results) == 2, f"a structure pass and a value pass; eligibility {reasons}"
    structure, values = results
    assert {i for i, c in enumerate(structure.captures) if c.kind == "primitive"} <= set(
        structure.released_capture_indices
    ) | _frame_output_producers(structure)
    raw_by_index = sparse._jax_capture_index_to_raw_op_label
    ops = _ops_by_raw(sparse)
    unselected = {
        i
        for i, c in enumerate(values.captures)
        if c.kind == "primitive" and ops[raw_by_index[i]].func_name != "tanh"
    }
    assert unselected
    assert set(values.released_capture_indices) == unselected
    for index in values.released_capture_indices:
        for value in values.captures[index].output_values:
            assert type(value).__name__ == "ShapeDtypeStruct"


def _frame_output_producers(result: Any) -> set[int]:
    output_ids = {id(value) for value in result.outputs}
    return {
        i
        for i, c in enumerate(result.captures)
        if any(id(value) in output_ids for value in c.output_values)
    }


def test_sparse_validation_matches_torch_selective_save() -> None:
    """The sparse trace's replay state and reasons equal torch's for the same selector."""

    selector = tl.func("tanh")
    model, x = _TorchTwoTanh(), torch.from_numpy(np.asarray(_x()))
    jax_trace = tl.trace(_two_tanh, _x(), save=selector, backend="jax")
    torch_trace = tl.trace(model, x, save=selector)

    jax_trace.validate_forward_pass([], validate_metadata=False)
    torch_trace.validate_forward_pass([model(x).detach().clone()], validate_metadata=False)
    jax_status = jax_trace.validation_replay_status
    torch_status = torch_trace.validation_replay_status
    detail = (
        f"jax {jax_status.state} {dict(jax_status.unverified_reason_counts)} "
        f"{dict(jax_status.exempted_reason_counts)}; torch {torch_status.state} "
        f"{dict(torch_status.unverified_reason_counts)} {dict(torch_status.exempted_reason_counts)}"
    )

    assert torch_status.state == "unverified", detail
    assert jax_status.state == torch_status.state, detail
    assert jax_status.unverified_reason_counts.get(_sparse_capture.MISSING_SAVED_ARGS), detail
    assert set(jax_status.unverified_reason_counts) <= {
        "missing_saved_args",
        "parent_payload_missing",
    }, detail
    assert set(torch_status.unverified_reason_counts) <= {
        "missing_saved_args",
        "parent_payload_missing",
    }, detail


def test_selecting_every_equation_validates_like_a_plain_capture() -> None:
    """When the selector keeps every equation the sparse trace passes replay validation."""

    every = tl.func("tanh") | tl.func("sin") | tl.func("mul") | tl.func("add")
    sparse = tl.trace(_fn, _x(), save=every, backend="jax")
    plain = tl.trace(_fn, _x(), backend="jax")

    evidence = sparse.annotations[_sparse_capture.EVIDENCE_KEY]
    excluded = set(evidence["excluded_raw_labels"])
    equation_raws = set(sparse._jax_capture_index_to_raw_op_label.values())
    assert not equation_raws & excluded
    assert plain.validate_forward_pass([]) is True
    assert sparse.validate_forward_pass([]) is True


def test_plain_capture_carries_no_selective_save_evidence() -> None:
    """A capture without ``save=`` runs once and records no selective-save evidence."""

    trace = tl.trace(_fn, _x(), backend="jax")

    assert _sparse_capture.EVIDENCE_KEY not in trace.annotations
    assert trace.validate_forward_pass([]) is True


def test_non_static_selector_keeps_the_full_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    """A callable ``save=`` predicate runs one full capture and hides payloads afterwards."""

    results = _spy_results(monkeypatch)
    trace = tl.trace(_fn, _x(), save=lambda op: op.func_name == "tanh", backend="jax")

    assert len(results) == 1
    assert not results[0].released_capture_indices
    assert _sparse_capture.EVIDENCE_KEY not in trace.annotations


@pytest.mark.parametrize(
    "options",
    [{"backward_ready": True}, {"save_grads": True}, {"intervene": object()}, {"unknown": 1}],
    ids=["backward_ready", "save_grads", "intervene", "unknown_option"],
)
def test_options_that_read_every_value_are_ineligible(options: dict[str, Any]) -> None:
    """Features that read every equation value, or options the capture does not know, fall back."""

    accepted = frozenset({"save", "backward_ready", "save_grads", "intervene"})
    selector = tl.func("tanh")

    assert _sparse_capture._eligible(selector, {"save": selector}, accepted, lambda _s: True)
    assert not _sparse_capture._eligible(
        selector, {"save": selector, **options}, accepted, lambda _s: True
    )
