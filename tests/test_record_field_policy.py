"""Structural field-policy audits for TorchLens record classes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.constants import (
    BACKWARD_PASS_FIELD_ORDER,
    BUFFER_LOG_FIELD_ORDER,
    GRAD_FN_LOG_FIELD_ORDER,
    GRAD_FN_PASS_LOG_FIELD_ORDER,
    LAYER_LOG_FIELD_ORDER,
    LAYER_PASS_LOG_FIELD_ORDER,
    MODEL_LOG_FIELD_ORDER,
    MODULE_LOG_FIELD_ORDER,
    MODULE_PASS_LOG_FIELD_ORDER,
    PARAM_LOG_FIELD_ORDER,
)
from torchlens.data_classes.backward_pass import BackwardPass
from torchlens.data_classes.buffer import Buffer
from torchlens.data_classes.field_policy import (
    field_order_from_policy,
    fork_policy_from_policy,
    portable_state_spec_from_policy,
)
from torchlens.data_classes.grad_fn import GradFn
from torchlens.data_classes.grad_fn_call import GradFnCall
from torchlens.data_classes.layer import Layer
from torchlens.data_classes.module import Module, ModuleCall
from torchlens.data_classes.op import Op
from torchlens.data_classes.param import Param
from torchlens.data_classes.trace import Trace
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness, ReadinessStatus


@dataclass(frozen=True)
class RecordCase:
    """One record class and its canonical user-facing field order."""

    cls: type[Any]
    field_order: list[str]


RECORD_CASES = (
    RecordCase(Trace, MODEL_LOG_FIELD_ORDER),
    RecordCase(Op, LAYER_PASS_LOG_FIELD_ORDER),
    RecordCase(Layer, LAYER_LOG_FIELD_ORDER),
    RecordCase(Param, PARAM_LOG_FIELD_ORDER),
    RecordCase(Buffer, BUFFER_LOG_FIELD_ORDER),
    RecordCase(GradFn, GRAD_FN_LOG_FIELD_ORDER),
    RecordCase(GradFnCall, GRAD_FN_PASS_LOG_FIELD_ORDER),
    RecordCase(ModuleCall, MODULE_PASS_LOG_FIELD_ORDER),
    RecordCase(Module, MODULE_LOG_FIELD_ORDER),
    RecordCase(BackwardPass, BACKWARD_PASS_FIELD_ORDER),
)
OPTIONAL_LIVE_FIELDS: dict[type[Any], set[str]] = {
    Trace: {"input_structure", "_containers"},
}


class _PolicyModel(nn.Module):
    """Small model covering params, buffers, modules, and backward grad_fns."""

    def __init__(self) -> None:
        """Initialize test layers."""

        super().__init__()
        self.linear = nn.Linear(3, 3)
        self.bn = nn.BatchNorm1d(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Scalar output.
        """

        return torch.relu(self.bn(self.linear(x))).sum()


class _RunnablePolicyModel(nn.Module):
    """Minimal model that satisfies sparse runnable capture prerequisites."""

    def __init__(self) -> None:
        """Initialize the single deterministic layer."""

        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the recorded static path.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Activated output tensor.
        """

        return torch.relu(self.linear(x))


def _policy_trace() -> Trace:
    """Return a trace with all record families populated.

    Returns
    -------
    Trace
        Trace after one backward pass.
    """

    model = _PolicyModel().eval()
    trace = tl.trace(
        model,
        torch.randn(2, 3, requires_grad=True),
        capture=tl.options.CaptureOptions(save_grads="all"),
    )
    trace.log_backward(trace[trace.output_layers[0]].out)
    return trace


def _record_instances(trace: Trace, cls: type[Any]) -> Iterable[Any]:
    """Yield representative live instances for one record class.

    Parameters
    ----------
    trace:
        Trace containing populated record objects.
    cls:
        Record class to sample.

    Yields
    ------
    Any
        Representative instances for field access checks.
    """

    if cls is Trace:
        yield trace
    elif cls is Op:
        yield next(iter(trace.ops))
    elif cls is Layer:
        yield next(iter(trace.layers))
    elif cls is Param:
        yield next(iter(trace.params))
    elif cls is Buffer:
        yield next(iter(trace.buffers))
    elif cls is GradFn:
        yield next(iter(trace.grad_fns))
    elif cls is GradFnCall:
        grad_fn = next(record for record in trace.grad_fns if record.calls)
        yield next(iter(grad_fn.calls.values()))
    elif cls is ModuleCall:
        yield next(iter(trace.module_calls))
    elif cls is Module:
        yield next(iter(trace.modules))
    elif cls is BackwardPass:
        yield next(iter(trace.backward_passes))


def test_layer_field_order_includes_live_aggregate_fields() -> None:
    """Layer FIELD_ORDER includes aggregate fields kept by FIELD_POLICY."""

    expected_fields = {
        "io_role",
        "is_atomic_module",
        "output_of_modules",
        "output_of_module_calls",
        "has_input_ancestor",
        "buffer_write_kind",
        "buffer_pass",
    }

    assert expected_fields.issubset(LAYER_LOG_FIELD_ORDER)


@pytest.mark.parametrize("case", RECORD_CASES, ids=lambda case: case.cls.__name__)
def test_record_field_policy_is_field_order_source(case: RecordCase) -> None:
    """Every record's user-facing policy entries exactly match FIELD_ORDER."""

    policy = case.cls.FIELD_POLICY
    assert field_order_from_policy(policy) == case.field_order
    assert [name for name, item in policy.items() if item.user_facing] == case.field_order
    assert len(case.field_order) == len(set(case.field_order))


@pytest.mark.parametrize("case", RECORD_CASES, ids=lambda case: case.cls.__name__)
def test_record_portable_spec_is_generated_from_policy(case: RecordCase) -> None:
    """Class portable specs are generated views over FIELD_POLICY."""

    assert portable_state_spec_from_policy(case.cls.FIELD_POLICY) == case.cls.PORTABLE_STATE_SPEC


def test_trace_and_op_generated_policy_views_match_old_names() -> None:
    """Trace/Op fork-policy views are generated from their field policy tables."""

    assert fork_policy_from_policy(Trace.FIELD_POLICY) == Trace.FIELD_FORK_POLICY
    assert fork_policy_from_policy(Op.FIELD_POLICY) == Op.FIELD_FORK_POLICY


@pytest.mark.parametrize("case", RECORD_CASES, ids=lambda case: case.cls.__name__)
def test_record_field_order_attributes_are_live(case: RecordCase) -> None:
    """Captured records expose every ordered user-facing field."""

    trace = _policy_trace()
    try:
        for record in _record_instances(trace, case.cls):
            optional = OPTIONAL_LIVE_FIELDS.get(case.cls, set())
            missing = [
                field
                for field in case.field_order
                if field not in optional and not hasattr(record, field)
            ]
            assert not missing, f"{case.cls.__name__} missing FIELD_ORDER fields: {missing}"
    finally:
        trace.cleanup()


def test_trace_runnable_field_order_slots_are_live_and_loaded_values_override(
    tmp_path: Path,
) -> None:
    """Fresh Trace runnable slots are inert and runnable loads replace their defaults."""

    trace = tl.trace(
        _RunnablePolicyModel().eval(),
        torch.randn(2, 3),
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    try:
        assert trace._runnable.descriptor is None
        assert trace._runnable.readiness is None
        assert trace._runnable.staged_user_state is None
        assert trace._runnable.embedded_state is None
        assert trace._runnable.archived_activations is None
        assert trace._runnable.path_faithfulness is None
        assert trace._runnable.first_mismatch is None
        assert trace._runnable.poisoned is False

        path = tmp_path / "trace.tlspec"
        trace.save(path, level="runnable")
        loaded = tl.load(path)
        try:
            assert loaded.runnable_descriptor is not None
            assert loaded.readiness is not None
            assert loaded.readiness.status is ReadinessStatus.READY
            assert loaded._runnable.path_faithfulness is None
            assert loaded._runnable.poisoned is False
            result = loaded.run(inputs=torch.randn(2, 3), seed=11)
            try:
                assert result.trace._runnable.path_faithfulness is PathFaithfulness.VERIFIED
                assert result.trace._runnable.poisoned is False
            finally:
                result.trace.cleanup()
        finally:
            loaded.cleanup()
    finally:
        trace.cleanup()


@pytest.mark.requires_assertions
def test_postprocess_contract_assertions_run_over_standard_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The standard torch capture path passes debug postprocess boundary assertions."""

    monkeypatch.setenv("TORCHLENS_POSTPROCESS_ASSERTIONS", "1")
    trace = tl.trace(
        _PolicyModel().eval(),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(save_grads=False),
    )
    try:
        assert trace.graph_shape_hash is not None
        assert trace.layer_logs
        assert trace.modules
    finally:
        trace.cleanup()


@pytest.mark.requires_assertions
def test_postprocess_write_audit_enforces_declared_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each postprocess step writes only its declared op-store columns (M10).

    The declared ``PostprocessStepContract.writes`` sets were recorded over
    the six surface-oracle model axes; the audit re-runs one representative
    conditional+recurrent-free capture here so a step growing an undeclared
    column write fails CI, not just the opt-in debug env.
    """

    monkeypatch.setenv("TORCHLENS_POSTPROCESS_ASSERTIONS", "1")
    trace = tl.trace(
        _PolicyModel().eval(),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(save_grads=False),
    )
    trace.cleanup()


@pytest.mark.requires_assertions
def test_postprocess_write_audit_covers_save_code_context_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Step 11.5's var_names writes pass enforcement under save_code_context.

    Design-ppdag-v3 defect 3: step 11.5 declared an EMPTY write set, silently
    wrong under ``save_code_context=True`` (it assigns ``op.var_names`` on
    every op). No recorded enforcement axis enabled the flag, so the audit
    never tripped. This axis pins the repaired declaration.
    """

    class _AssigningModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            hidden = self.linear(x)
            activated = torch.relu(hidden)
            return activated

    monkeypatch.setenv("TORCHLENS_POSTPROCESS_ASSERTIONS", "1")
    trace = tl.trace(
        _AssigningModel().eval(),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(save_code_context=True, save_grads=False),
    )
    try:
        assert any(op.var_names for op in trace.layer_list if op.type != "output")
    finally:
        trace.cleanup()


@pytest.mark.requires_assertions
def test_postprocess_write_audit_covers_streaming_axis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Steps 18/19 pass enforcement with their hand-derived write sets.

    Design-ppdag-v3 defect 1: steps 18/19 declared ``writes=None``
    (wildcard), so the streaming finalization/eviction windows were never
    audited. This axis runs a disk-streamed capture under enforcement and
    asserts the streamed refs landed (step 18) and outs were evicted
    (step 19).
    """

    monkeypatch.setenv("TORCHLENS_POSTPROCESS_ASSERTIONS", "1")
    trace = tl.trace(
        _PolicyModel().eval(),
        torch.randn(2, 3),
        storage=tl.to_disk(tmp_path / "run.tlspec"),
        capture=tl.options.CaptureOptions(save_grads=False),
    )
    try:
        streamed = [op for op in trace.layer_list if getattr(op, "out_ref", None) is not None]
        assert streamed, "step 18 must attach streamed out refs"
        assert all(op._slot("out") is None for op in streamed), "step 19 must evict outs"
    finally:
        trace.cleanup()


class _OrphanEquivalenceModel(nn.Module):
    """Orphan island sharing an equivalence class with surviving ops.

    The ``z``-side ops are a disconnected component (orphaned by step 3's
    undirected flood) while ``z + 1``/``z ** 2`` are equivalence-classmates
    of the surviving ``x + 1``/``x ** 2``, so the removal scrub must rebind
    the survivors' ``equivalent_ops``.
    """

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        x = x + 1
        z = torch.ones(5, 5)
        z = z + 1
        _dead = z**2
        return x**2


@pytest.mark.requires_assertions
def test_postprocess_write_audit_covers_orphan_keep_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Step 3 passes enforcement writing is_orphan under keep_orphans=True.

    Design-ppdag-v3 defect 4a: with ``keep_orphans=True`` on an
    orphan-bearing model, step 3 writes ``is_orphan`` on every retained
    orphan and returns BEFORE the batch removal — a write set no recorded
    enforcement axis exercised.
    """

    monkeypatch.setenv("TORCHLENS_POSTPROCESS_ASSERTIONS", "1")
    trace = tl.trace(
        _OrphanEquivalenceModel(),
        torch.ones(5, 5),
        capture=tl.options.CaptureOptions(keep_orphans=True),
    )
    try:
        assert trace.orphans, "the island must be retained as orphans"
        assert all(op.is_orphan for op in trace.orphans)
    finally:
        trace.cleanup()


@pytest.mark.requires_assertions
def test_postprocess_write_audit_covers_orphan_removal_scrub_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Step 3 passes enforcement scrubbing survivors' equivalent_ops.

    Design-ppdag-v3 defect 4b: default orphan REMOVAL rebinds surviving
    rows' ``equivalent_ops`` when an orphan shared an equivalence class —
    an undeclared write that tripped the audit the day this axis landed.
    """

    monkeypatch.setenv("TORCHLENS_POSTPROCESS_ASSERTIONS", "1")
    trace = tl.trace(_OrphanEquivalenceModel(), torch.ones(5, 5))
    try:
        labels = {op.label for op in trace.layer_list}
        for op in trace.layer_list:
            assert set(op.equivalent_ops) <= labels, "scrub left a dead equivalence label"
    finally:
        trace.cleanup()


@pytest.mark.requires_assertions
def test_postprocess_write_audit_trips_on_undeclared_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An undeclared op-store column write fails the step-contract tripwire."""

    from torchlens.postprocess import POSTPROCESS_STEP_CONTRACTS, PostprocessStepContract

    original = POSTPROCESS_STEP_CONTRACTS["4"]
    assert original.writes, "step 4 must declare a non-empty write set"
    narrowed = PostprocessStepContract(
        original.step,
        original.name,
        original.contract,
        writes=frozenset(),
        reads=original.reads,
        trace_state=original.trace_state,
    )
    monkeypatch.setenv("TORCHLENS_POSTPROCESS_ASSERTIONS", "1")
    monkeypatch.setitem(POSTPROCESS_STEP_CONTRACTS, "4", narrowed)
    with pytest.raises(AssertionError, match="undeclared op-store columns"):
        tl.trace(
            _PolicyModel().eval(),
            torch.randn(2, 3),
            capture=tl.options.CaptureOptions(save_grads=False),
        )


@pytest.mark.requires_assertions
def test_postprocess_write_audit_catches_in_place_container_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An in-place container mutation inside a step trips the write audit.

    Sol review finding 6: the audit intercepted only cell assignment and
    deletion, so an in-place ``annotations[...] = ...`` mutation smuggled an
    undeclared write through a step with an EMPTY declared write set
    (step 10). The content-fingerprint diff now surfaces it.
    """

    import torchlens.postprocess as postprocess_mod

    real_rename = postprocess_mod._rename_model_history_layer_names

    def smuggling_rename(trace: object) -> None:
        real_rename(trace)
        raw_layer_dict = trace._raw_graph_ws.raw_layer_dict
        first_op = next(iter(raw_layer_dict.values()))
        first_op.annotations["smuggled_in_place"] = 1

    monkeypatch.setenv("TORCHLENS_POSTPROCESS_ASSERTIONS", "1")
    monkeypatch.setattr(postprocess_mod, "_rename_model_history_layer_names", smuggling_rename)
    with pytest.raises(
        AssertionError, match=r"Step 10 .* undeclared op-store columns.*annotations"
    ):
        tl.trace(
            _PolicyModel().eval(),
            torch.randn(2, 3),
            capture=tl.options.CaptureOptions(save_grads=False),
        )


@pytest.mark.requires_assertions
def test_postprocess_write_audit_allows_sanctioned_row_removal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fastlog cook passes enforcement despite step-3 orphan removal.

    Pre-existing false positive (reproduced at 684f8860): removing an
    orphan op husks the row cell-by-cell, and the audit recorded every
    per-cell delete as a column write — step 3 tripped with essentially
    the whole layout on removal-heavy paths such as ``Recording.to_trace``.
    Whole-row release is now a row-lifecycle event accounted against the
    step contract's explicit ``removes_rows`` sanction instead.
    """

    monkeypatch.setenv("TORCHLENS_POSTPROCESS_ASSERTIONS", "1")
    model = _PolicyModel().eval()
    recording = tl.record(model, torch.randn(2, 3), save=tl.func("linear"))
    trace = recording.to_trace()
    assert trace.layer_logs
    trace.cleanup()


@pytest.mark.requires_assertions
def test_postprocess_write_audit_trips_on_unsanctioned_row_removal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing rows in a step without a removes_rows sanction still trips."""

    from torchlens.postprocess import POSTPROCESS_STEP_CONTRACTS, PostprocessStepContract

    original = POSTPROCESS_STEP_CONTRACTS["3"]
    assert "deletes" in original.row_effects, "step 3 must sanction orphan-row removal"
    unsanctioned = PostprocessStepContract(
        original.step,
        original.name,
        original.contract,
        writes=original.writes,
        reads=original.reads,
        row_effects=original.row_effects - {"deletes"},
        trace_state=original.trace_state,
    )
    monkeypatch.setenv("TORCHLENS_POSTPROCESS_ASSERTIONS", "1")
    monkeypatch.setitem(POSTPROCESS_STEP_CONTRACTS, "3", unsanctioned)
    model = _PolicyModel().eval()
    recording = tl.record(model, torch.randn(2, 3), save=tl.func("linear"))
    with pytest.raises(AssertionError, match="without a 'deletes' row_effects sanction"):
        recording.to_trace()


def test_write_audit_fingerprint_is_order_canonical() -> None:
    """Equal container content fingerprints equal; real mutation differs.

    ``hash(repr(value))`` on set cells was iteration-order sensitive: 8 and
    16 collide in a small set table, so equal sets built in different
    insertion orders repr differently, and a mutate-and-revert inside one
    step could register as a write of an undeclared column (closure
    review, F6 fragility). The fingerprint now sorts element fingerprints
    for unordered containers.
    """

    from torchlens._trace_core.op_store import _cell_content_fingerprint

    a = {8, 16}
    b = {16, 8}
    assert list(a) != list(b) or repr(a) != repr(b) or a == b  # equal content
    assert _cell_content_fingerprint(a) == _cell_content_fingerprint(b)

    # Mutate-and-revert keeps the fingerprint stable even when the revert
    # changes iteration order (table resize).
    grown = {8, 16}
    before = _cell_content_fingerprint(grown)
    for value in range(100, 200):
        grown.add(value)
    for value in range(100, 200):
        grown.remove(value)
    assert _cell_content_fingerprint(grown) == before

    # Equal dicts with reordered keys fingerprint equal; nested unordered
    # containers canonicalize recursively.
    assert _cell_content_fingerprint({"a": {8, 16}, "b": 1}) == (
        _cell_content_fingerprint({"b": 1, "a": {16, 8}})
    )

    # Real content changes are still caught, including nested ones.
    assert _cell_content_fingerprint({8, 16}) != _cell_content_fingerprint({8, 17})
    assert _cell_content_fingerprint([1, [2, 3]]) != _cell_content_fingerprint([1, [2, 4]])
    # Lists stay ORDER-SENSITIVE (list equality is positional).
    assert _cell_content_fingerprint([1, 2]) != _cell_content_fingerprint([2, 1])
