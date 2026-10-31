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
from torchlens._constants import (
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
from torchlens._data_classes.backward_pass import BackwardPass
from torchlens._data_classes.buffer import Buffer
from torchlens._data_classes.field_policy import (
    field_order_from_policy,
    fork_policy_from_policy,
    portable_state_spec_from_policy,
)
from torchlens._data_classes.grad_fn import GradFn
from torchlens._data_classes.grad_fn_call import GradFnCall
from torchlens._data_classes.layer import Layer
from torchlens._data_classes.module import Module, ModuleCall
from torchlens._data_classes.op import Op
from torchlens._data_classes.param import Param
from torchlens._data_classes.trace import Trace
from torchlens._runnable import PathFaithfulness, ReadinessStatus
from torchlens.options import CaptureOptions


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
        yield next(iter(trace.autograd.grad_fns))
    elif cls is GradFnCall:
        grad_fn = next(record for record in trace.autograd.grad_fns if record.calls)
        yield next(iter(grad_fn.calls.values()))
    elif cls is ModuleCall:
        yield next(iter(trace.module_calls))
    elif cls is Module:
        yield next(iter(trace.modules))
    elif cls is BackwardPass:
        yield next(iter(trace.autograd.backward_passes))


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

    assert portable_state_spec_from_policy(case.cls.FIELD_POLICY) == case.cls._PORTABLE_STATE_SPEC


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


@pytest.mark.filterwarnings(
    # A08 random-role-init disclosure (runnable_random_init_run): this test
    # exercises a weight-free runnable run DELIBERATELY; the warning is the
    # feature under test elsewhere (tests/test_persist_honesty_runnable_random.py).
    "default:This runnable artifact carries no model weights"
)
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
