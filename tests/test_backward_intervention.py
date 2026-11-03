"""Backward intervention selector, helper, and grad_fn_handle dispatch tests."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens._data_classes.grad_fn import GradFn
from torchlens._ir.capture_events import CaptureEvents
from torchlens._ir.selector_eval import selector_from_spec
from torchlens._selection.selectors import classify_selector_universe
from torchlens.backends.torch.backward import (
    _clear_pending_accumulate_grad_records,
    _make_grad_fn_hook,
    _make_grad_fn_prehook,
)
from torchlens.intervention.errors import HelperMountError, HookValueError, SelectorCompositionError
from torchlens.intervention.helpers import _helper_spec
from torchlens.intervention.hooks import normalize_hook_plan
from torchlens.intervention.types import FireRecord, InterventionSpec, TargetSpec


class _EncoderModel(nn.Module):
    """Small model with module-scoped ReLU and intervening backward nodes."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        hidden = torch.relu(self.encoder(x))
        viewed = hidden.view(hidden.shape[0], 4)
        return self.head(viewed).sum()


class _PowModel(nn.Module):
    """Model with a differentiable PowBackward grad input."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a scalar cubic loss."""

        return (x.pow(3)).sum()


class _IdentityReluModel(nn.Module):
    """Model with one deterministic gradient path through a ReLU."""

    def __init__(self) -> None:
        """Initialize an identity linear layer."""

        super().__init__()
        self.linear = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a scalar loss with unit baseline input gradient."""

        return torch.relu(self.linear(x)).sum()


@dataclass
class _TraceStub:
    """Weakref-able trace stub for direct hook-factory tests."""

    _grad_fn_logs: dict[int, GradFn]
    _grad_op_nums_to_save: str = "all"
    last_run: dict[str, Any] | None = None
    _num_backward_passes: int = 0
    # A live trace always owns its capture event stream; hooks refuse typed
    # when it is absent, so the stub must model ownership explicitly.
    _capture_events: CaptureEvents = field(default_factory=CaptureEvents)


@pytest.fixture(autouse=True)
def _clear_active_hook_plan() -> None:
    """Reset global hook state around tests."""

    previous_plan = _state._active_hook_plan
    previous_trace = _state._active_trace
    previous_spec = _state._active_intervention_spec
    try:
        yield
    finally:
        _state._active_hook_plan = previous_plan
        _state._active_trace = previous_trace
        _state._active_intervention_spec = previous_spec


def _trace() -> tuple[_EncoderModel, torch.Tensor, tl.Trace]:
    """Return a traced model with a scalar output."""

    model = _EncoderModel()
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads="all"))
    return model, x, trace


def _loss(trace: tl.Trace) -> torch.Tensor:
    """Return the traced scalar output."""

    return trace[trace.output_layers[0]].out


def _logged_backward_trace() -> tl.Trace:
    """Return a trace after backward capture."""

    _model, _x, trace = _trace()
    trace.log_backward(_loss(trace), retain_graph=True)
    return trace


def _first_accumulate_grad(loss: torch.Tensor) -> Any:
    """Return the first AccumulateGrad node below a loss."""

    queue = [loss.grad_fn]
    seen: set[int] = set()
    strong_refs: list[Any] = []
    while queue:
        grad_fn_handle = queue.pop(0)
        if grad_fn_handle is None or id(grad_fn_handle) in seen:
            continue
        seen.add(id(grad_fn_handle))
        strong_refs.append(grad_fn_handle)
        if type(grad_fn_handle).__name__ == "AccumulateGrad":
            return grad_fn_handle
        next_fns = [next_fn for next_fn, _idx in getattr(grad_fn_handle, "next_functions", ())]
        strong_refs.extend(next_fn for next_fn in next_fns if next_fn is not None)
        queue.extend(next_fns)
    raise AssertionError("AccumulateGrad not found")


def _hook_trace() -> tuple[_TraceStub, GradFn]:
    """Return a stub trace and grad_fn_handle log for hook-factory tests."""

    grad_fn_handle = GradFn(
        grad_fn_object_id=1,
        class_name="ReluBackward0",
        class_qualname="torch.autograd.ReluBackward0",
        is_custom=False,
        label="relu_back_1_1",
        type="relu",
        type_index=1,
        ordinal_index=1,
        step_index=1,
        has_op=False,
    )
    return _TraceStub({1: grad_fn_handle}), grad_fn_handle


def test_accumulategrad_post_hook_logs_grad_fn_call() -> None:
    """AccumulateGrad post-hooks still record call logs."""

    trace = _logged_backward_trace()
    accumulate_logs = [
        gfl for gfl in trace._grad_fn_logs.values() if gfl.class_name == "AccumulateGrad"
    ]
    assert accumulate_logs
    assert all(gfl.num_calls >= 1 for gfl in accumulate_logs)


def test_accumulategrad_post_hook_skips_helper_dispatch() -> None:
    """AccumulateGrad helpers fire through the prehook only."""

    trace_stub, grad_fn_handle = _hook_trace()
    counter = {"fires": 0}

    def factory() -> Any:
        """Return a tuple-shaped counting helper."""

        def helper(
            grad_input: tuple[torch.Tensor | None, ...],
            *,
            grad_output: tuple[torch.Tensor | None, ...] | None,
            grad_fn_handle: GradFn,
            call_index: int,
            run_ctx: dict[str, Any],
        ) -> tuple[torch.Tensor | None, ...]:
            """Count helper dispatches."""

            del grad_output, grad_fn_handle, call_index, run_ctx
            counter["fires"] += 1
            return grad_input

        return helper

    helper_spec = _helper_spec(
        "count_grad",
        signal="backward",
        factory=factory,
        metadata={"mount_shape": "tuple"},
    )
    _state._active_hook_plan = normalize_hook_plan(tl.grad_fn(name="relu"), helper_spec)
    post_hook = _make_grad_fn_hook(trace_stub, 1, is_accumulate_grad=True)
    pre_hook = _make_grad_fn_prehook(trace_stub, 1)
    grad = (torch.ones(1),)
    assert pre_hook(grad) == grad
    assert post_hook((), grad) is None
    assert grad_fn_handle.num_calls == 1
    assert counter["fires"] == 1


def test_grad_fn_hook_returns_none_when_no_active_plan() -> None:
    """Grad_fn hook returns None when no hook plan is active."""

    trace_stub, _grad_fn_log = _hook_trace()
    hook = _make_grad_fn_hook(trace_stub, 1)
    assert hook((torch.ones(1),), (torch.ones(1),)) is None


def test_grad_fn_hook_returns_none_when_no_match() -> None:
    """Grad_fn hook returns None when active hooks do not match."""

    trace_stub, _grad_fn_log = _hook_trace()
    _state._active_hook_plan = normalize_hook_plan(
        tl.grad_fn(name="missing"), tl.clamp(low=0, high=1)
    )
    hook = _make_grad_fn_hook(trace_stub, 1)
    assert hook((torch.ones(1),), (torch.ones(1),)) is None


def test_grad_fn_hook_returns_tuple_when_mutating() -> None:
    """Grad_fn hook returns a tuple when a matching helper mutates gradients."""

    trace_stub, _grad_fn_log = _hook_trace()
    _state._active_hook_plan = normalize_hook_plan(tl.grad_fn(name="relu"), tl.clamp(low=0, high=0))
    hook = _make_grad_fn_hook(trace_stub, 1)
    result = hook((torch.ones(1),), (torch.ones(1),))
    assert isinstance(result, tuple)
    assert torch.equal(result[0], torch.zeros(1))


def test_grad_fn_hook_records_backward_fire() -> None:
    """A matching grad_fn hook attaches a backward fire record to its call."""

    trace_stub, grad_fn_handle = _hook_trace()
    _state._active_hook_plan = normalize_hook_plan(tl.grad_fn(name="relu"), tl.clamp(low=0, high=0))
    hook = _make_grad_fn_hook(trace_stub, 1)

    result = hook((torch.ones(1),), (torch.ones(1),))

    assert isinstance(result, tuple)
    record = grad_fn_handle.calls[0].fire_record
    assert isinstance(record, FireRecord)
    assert record.signal == "backward"
    assert record.call_index == 1
    assert record.grad_kind == "grad_input"
    assert record.tuple_index == 0
    assert record.replaced is True


def test_grad_fn_hook_inplace_mutation_records_replaced() -> None:
    """A backward hook mutating a grad slot in place must not record replaced=False.

    ``replaced`` derived from tuple-slot identity alone, so a hook editing
    ``grad_input[0]`` in place and returning ``None`` was a genuine gradient
    change recorded with zero replacement evidence.
    """

    trace_stub, grad_fn_handle = _hook_trace()

    def mutate_in_place(
        grad_input: tuple[torch.Tensor, ...],
        *,
        grad_output: tuple[torch.Tensor, ...],
        grad_fn_handle: object,
        call_index: int,
        run_ctx: object,
    ) -> None:
        """Zero the first grad slot in place, returning nothing."""

        del grad_output, grad_fn_handle, call_index, run_ctx
        grad_input[0].mul_(0)
        return None

    mutate_in_place.signal = "backward"
    _state._active_hook_plan = normalize_hook_plan(tl.grad_fn(name="relu"), mutate_in_place)
    hook = _make_grad_fn_hook(trace_stub, 1)

    hook((torch.ones(1),), (torch.ones(1),))

    record = grad_fn_handle.calls[0].fire_record
    assert isinstance(record, FireRecord)
    assert record.replaced is True


def test_grad_fn_hook_data_alias_mutation_records_replaced() -> None:
    """A backward hook mutating a grad slot through ``.data`` records replaced.

    grind-p5 rollup (incomplete 2289e56c): ``.data`` mints a storage-sharing
    alias with a FRESH version counter, so the tuple-version witness read a
    ``grad_input[0].data.mul_(0)`` edit as "no mutation" -- the same false
    no-replacement claim the counter fix closed for direct in-place edits.
    """

    trace_stub, grad_fn_handle = _hook_trace()

    def mutate_via_data_alias(
        grad_input: tuple[torch.Tensor, ...],
        *,
        grad_output: tuple[torch.Tensor, ...],
        grad_fn_handle: object,
        call_index: int,
        run_ctx: object,
    ) -> None:
        """Zero the first grad slot through the ``.data`` channel."""

        del grad_output, grad_fn_handle, call_index, run_ctx
        grad_input[0].data.mul_(0)
        return None

    mutate_via_data_alias.signal = "backward"
    _state._active_hook_plan = normalize_hook_plan(tl.grad_fn(name="relu"), mutate_via_data_alias)
    hook = _make_grad_fn_hook(trace_stub, 1)

    hook((torch.ones(1),), (torch.ones(1),))

    record = grad_fn_handle.calls[0].fire_record
    assert isinstance(record, FireRecord)
    assert record.replaced is True


def test_composite_backward_target_specs_match_live_hooks() -> None:
    """Composite selector target specs reconstruct recursively for live hooks."""

    selectors = [
        tl.grad_fn(name="relu") & tl.in_backward_pass(1),
        tl.grad_fn(name="missing") | tl.grad_fn(name="relu"),
        (tl.grad_fn(name="missing") | tl.grad_fn(name="relu")) & tl.in_backward_pass(1),
    ]
    for selector in selectors:
        trace_stub, grad_fn_handle = _hook_trace()
        trace_stub._active_backward_pass_index = 1  # type: ignore[attr-defined]
        grad_fn_handle.trace = trace_stub  # type: ignore[attr-defined]
        _state._active_hook_plan = normalize_hook_plan(selector, tl.clamp(low=0, high=0))
        hook = _make_grad_fn_hook(trace_stub, 1)

        result = hook((torch.ones(1),), (torch.ones(1),))

        assert isinstance(result, tuple), repr(selector)
        assert torch.equal(result[0], torch.zeros(1))
        assert isinstance(grad_fn_handle.calls[0].fire_record, FireRecord)


def test_composite_grad_output_selector_matches_real_live_backward_hook() -> None:
    """A grad_output composite selector matches during real live backward capture."""

    fires = {"count": 0}

    def factory() -> Any:
        """Return a tuple helper that counts grad-output composite matches."""

        def helper(
            grad_input: tuple[torch.Tensor | None, ...],
            *,
            grad_output: tuple[torch.Tensor | None, ...] | None,
            grad_fn_handle: GradFn,
            call_index: int,
            run_ctx: dict[str, Any],
        ) -> tuple[torch.Tensor | None, ...]:
            """Count a live backward helper fire."""

            del grad_output, grad_fn_handle, call_index, run_ctx
            fires["count"] += 1
            return grad_input

        return helper

    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(
        _EncoderModel(),
        x,
        capture=tl.options.CaptureOptions(save_grads="all", backward_ready=True),
    )
    trace._attach_hooks(
        tl.grad_output() & tl.grad_fn(name="relu"),
        _helper_spec(
            "count_grad_output",
            signal="backward",
            factory=factory,
            metadata={"mount_shape": "tuple"},
        ),
        confirm_mutation=True,
    )

    trace.log_backward(trace[trace.output_layers[0]].out.sum(), retain_graph=True)

    assert fires["count"] == 1


def test_live_backward_pass_selector_targets_second_retain_graph_pass() -> None:
    """Live in_backward_pass(2) targeting fires only during the second backward pass."""

    fires = {"count": 0}

    def factory() -> Any:
        """Return a tuple helper that counts live fires."""

        def helper(
            grad_input: tuple[torch.Tensor | None, ...],
            *,
            grad_output: tuple[torch.Tensor | None, ...] | None,
            grad_fn_handle: GradFn,
            call_index: int,
            run_ctx: dict[str, Any],
        ) -> tuple[torch.Tensor | None, ...]:
            """Count the matching backward helper fire."""

            del grad_output, grad_fn_handle, call_index, run_ctx
            fires["count"] += 1
            return grad_input

        return helper

    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(
        _EncoderModel(),
        x,
        capture=tl.options.CaptureOptions(save_grads="all", backward_ready=True),
    )
    trace._attach_hooks(
        tl.grad_fn(name="relu") & tl.in_backward_pass(2),
        _helper_spec(
            "count_second_pass",
            signal="backward",
            factory=factory,
            metadata={"mount_shape": "tuple"},
        ),
        confirm_mutation=True,
    )

    trace.log_backward(trace[trace.output_layers[0]].out.sum(), retain_graph=True)
    assert fires["count"] == 0
    trace.log_backward(trace[trace.output_layers[0]].out.sum(), retain_graph=True)

    assert fires["count"] == 1
    materialized_sites = trace.find_sites(
        tl.grad_fn(name="relu") & tl.in_backward_pass(2), max_fanout=100
    )
    assert materialized_sites
    assert all(
        any(call.backward_pass_index == 2 for call in site.calls._list)
        for site in materialized_sites
        if isinstance(site, GradFn)
    )


def test_backward_none_return_helper_records_non_replacing_fire() -> None:
    """Matching backward helpers returning None still produce a FireRecord."""

    trace_stub, grad_fn_handle = _hook_trace()
    spec = InterventionSpec()
    counter = {"fires": 0}

    def factory() -> Any:
        """Return a helper that counts and returns None."""

        def helper(
            grad_input: tuple[torch.Tensor | None, ...],
            *,
            grad_output: tuple[torch.Tensor | None, ...] | None,
            grad_fn_handle: GradFn,
            call_index: int,
            run_ctx: dict[str, Any],
        ) -> None:
            """Count a matched helper call without replacing gradients."""

            del grad_input, grad_output, grad_fn_handle, call_index, run_ctx
            counter["fires"] += 1
            return None

        return helper

    _state._active_intervention_spec = spec
    _state._active_hook_plan = normalize_hook_plan(
        tl.grad_fn(name="relu"),
        _helper_spec(
            "count_none", signal="backward", factory=factory, metadata={"mount_shape": "tuple"}
        ),
    )
    hook = _make_grad_fn_hook(trace_stub, 1)

    result = hook((torch.ones(1),), (torch.ones(1),))

    assert result is None
    assert counter["fires"] == 1
    record = grad_fn_handle.calls[0].fire_record
    assert isinstance(record, FireRecord)
    assert record.replaced is False
    assert spec.records == [record]


def test_backward_invalid_replacement_tuple_raises_before_recording_success() -> None:
    """Invalid backward replacement tuples fail before a success record is attached."""

    trace_stub, grad_fn_handle = _hook_trace()
    spec = InterventionSpec()

    def factory() -> Any:
        """Return a helper with a shape-invalid replacement tuple."""

        def helper(
            grad_input: tuple[torch.Tensor | None, ...],
            *,
            grad_output: tuple[torch.Tensor | None, ...] | None,
            grad_fn_handle: GradFn,
            call_index: int,
            run_ctx: dict[str, Any],
        ) -> tuple[torch.Tensor | None, ...]:
            """Return a same-arity tuple with an invalid tensor shape."""

            del grad_output, grad_fn_handle, call_index, run_ctx
            return (torch.ones(2),)

        return helper

    _state._active_intervention_spec = spec
    _state._active_hook_plan = normalize_hook_plan(
        tl.grad_fn(name="relu"),
        _helper_spec(
            "bad_shape", signal="backward", factory=factory, metadata={"mount_shape": "tuple"}
        ),
    )
    hook = _make_grad_fn_hook(trace_stub, 1)

    with pytest.raises(HookValueError, match="shape"):
        hook((torch.ones(1),), (torch.ones(1),))

    assert grad_fn_handle.calls[0].fire_record is None
    assert spec.records == []


def test_backward_in_place_none_return_helper_records_gradient_effect() -> None:
    """In-place gradient mutation returning None is audited and affects gradients."""

    def zero_in_place(
        grad_input: tuple[torch.Tensor | None, ...],
        *,
        grad_output: tuple[torch.Tensor | None, ...] | None,
        grad_fn_handle: GradFn,
        call_index: int,
        run_ctx: dict[str, Any],
    ) -> None:
        """Zero the first gradient tensor in place and return None."""

        del grad_output, grad_fn_handle, call_index, run_ctx
        if grad_input[0] is not None:
            grad_input[0].mul_(0)
        return None

    zero_in_place.signal = "backward"  # type: ignore[attr-defined]
    x = torch.ones(2, 3, requires_grad=True)
    trace = tl.trace(
        _EncoderModel(),
        x,
        capture=tl.options.CaptureOptions(save_grads="all", backward_ready=True),
    )
    trace._attach_hooks(tl.grad_fn(name="relu"), zero_in_place, confirm_mutation=True)

    trace.log_backward(trace[trace.output_layers[0]].out.sum(), retain_graph=True)

    assert x.grad is not None
    assert torch.equal(x.grad, torch.zeros_like(x.grad))
    records = [
        call.fire_record
        for grad_fn in trace._grad_fn_logs.values()
        for call in grad_fn.calls._list
        if call.fire_record is not None
    ]
    assert records
    assert isinstance(records[0], FireRecord)
    # Round-3 semantics: an in-place mutation IS a genuine gradient change and
    # must carry replacement evidence (version-counter witnessed). The old
    # identity-only derivation recorded replaced=False here -- a real value
    # change with zero replacement evidence.
    assert records[0].replaced is True


def test_grad_fn_hook_call_index_targeting() -> None:
    """Backward live matching honors an explicit target-spec call index."""

    trace_stub, grad_fn_handle = _hook_trace()
    target = TargetSpec(
        "grad_fn",
        {"type": "relu"},
        metadata={"call_index": 2},
    )
    _state._active_hook_plan = normalize_hook_plan(target, tl.clamp(low=0, high=0))
    hook = _make_grad_fn_hook(trace_stub, 1)

    first = hook((torch.ones(1),), (torch.ones(1),))
    second = hook((torch.ones(1),), (torch.ones(1),))

    assert first is None
    assert isinstance(second, tuple)
    assert grad_fn_handle.calls[0].fire_record is None
    assert isinstance(grad_fn_handle.calls[1].fire_record, FireRecord)


def test_helper_requires_grad_output_at_accumulategrad_raises_helpermounterror() -> None:
    """Helpers that require grad_output cannot mount on backward prehook sites."""

    helper = _helper_spec(
        "needs_output",
        signal="backward",
        factory=lambda: lambda grad_input, **kwargs: grad_input,
        metadata={"mount_shape": "tuple", "requires_grad_output": True},
    )
    with pytest.raises(HelperMountError):
        normalize_hook_plan(tl.grad_fn(name="AccumulateGrad"), helper)


def test_grad_fn_prehook_posthook_call_index_alignment() -> None:
    """Prehook and post-hook agree on the first one-based call index."""

    seen: list[int] = []

    def factory() -> Any:
        """Return a helper that records call indexes."""

        def helper(
            grad_input: tuple[torch.Tensor | None, ...],
            *,
            grad_output: tuple[torch.Tensor | None, ...] | None,
            grad_fn_handle: GradFn,
            call_index: int,
            run_ctx: dict[str, Any],
        ) -> tuple[torch.Tensor | None, ...]:
            """Record prehook call index."""

            del grad_output, grad_fn_handle, run_ctx
            seen.append(call_index)
            return grad_input

        return helper

    trace_stub, grad_fn_handle = _hook_trace()
    _state._active_hook_plan = normalize_hook_plan(
        tl.grad_fn(name="relu"),
        _helper_spec(
            "record_index", signal="backward", factory=factory, metadata={"mount_shape": "tuple"}
        ),
    )
    pre_hook = _make_grad_fn_prehook(trace_stub, 1)
    post_hook = _make_grad_fn_hook(trace_stub, 1, is_accumulate_grad=True)
    pre_hook((torch.ones(1),))
    post_hook((), (torch.ones(1),))
    assert seen == [1]
    assert grad_fn_handle.calls[0].call_index == 1


def test_grad_fn_prehook_posthook_call_index_alignment_multi_fire() -> None:
    """Prehook indexes advance one-based across repeated fires."""

    seen: list[int] = []

    def factory() -> Any:
        """Return a helper that records call indexes."""

        def helper(
            grad_input: tuple[torch.Tensor | None, ...], **kwargs: Any
        ) -> tuple[torch.Tensor | None, ...]:
            """Record prehook call index."""

            seen.append(int(kwargs["call_index"]))
            return grad_input

        return helper

    trace_stub, grad_fn_handle = _hook_trace()
    _state._active_hook_plan = normalize_hook_plan(
        tl.grad_fn(name="relu"),
        _helper_spec(
            "record_index", signal="backward", factory=factory, metadata={"mount_shape": "tuple"}
        ),
    )
    pre_hook = _make_grad_fn_prehook(trace_stub, 1)
    post_hook = _make_grad_fn_hook(trace_stub, 1, is_accumulate_grad=True)
    for _idx in range(3):
        pre_hook((torch.ones(1),))
        post_hook((), (torch.ones(1),))
    assert seen == [1, 2, 3]
    assert tuple(grad_fn_handle.calls) == (1, 2, 3)


def test_accumulategrad_prehook_records_backward_fire() -> None:
    """AccumulateGrad prehook records are attached to the paired post-hook call."""

    trace_stub, grad_fn_handle = _hook_trace()
    _state._active_hook_plan = normalize_hook_plan(tl.grad_fn(name="relu"), tl.clamp(low=0, high=0))
    pre_hook = _make_grad_fn_prehook(trace_stub, 1)
    post_hook = _make_grad_fn_hook(trace_stub, 1, is_accumulate_grad=True)

    result = pre_hook((torch.ones(1),))
    post_hook((), (torch.ones(1),))

    assert isinstance(result, tuple)
    record = grad_fn_handle.calls[0].fire_record
    assert isinstance(record, FireRecord)
    assert record.timing == "pre"
    assert record.signal == "backward"
    assert record.call_index == 1


def test_accumulategrad_none_return_prehook_records_backward_fire() -> None:
    """AccumulateGrad prehooks returning None still attach a non-replacing record."""

    trace_stub, grad_fn_handle = _hook_trace()

    def factory() -> Any:
        """Return a helper that fires without replacing gradients."""

        def helper(grad_input: tuple[torch.Tensor | None, ...], **kwargs: Any) -> None:
            """Return None after matching an AccumulateGrad prehook."""

            del grad_input, kwargs
            return None

        return helper

    _state._active_hook_plan = normalize_hook_plan(
        tl.grad_fn(name="relu"),
        _helper_spec(
            "none_prehook", signal="backward", factory=factory, metadata={"mount_shape": "tuple"}
        ),
    )
    pre_hook = _make_grad_fn_prehook(trace_stub, 1)
    post_hook = _make_grad_fn_hook(trace_stub, 1, is_accumulate_grad=True)

    assert pre_hook((torch.ones(1),)) is None
    post_hook((), (torch.ones(1),))

    record = grad_fn_handle.calls[0].fire_record
    assert isinstance(record, FireRecord)
    assert record.timing == "pre"
    assert record.replaced is False


def test_accumulategrad_pending_records_clear_between_passes() -> None:
    """Unpaired AccumulateGrad prehook records are cleared at pass teardown."""

    trace_stub, grad_fn_handle = _hook_trace()
    _state._active_hook_plan = normalize_hook_plan(tl.grad_fn(name="relu"), tl.clamp(low=0, high=0))
    pre_hook = _make_grad_fn_prehook(trace_stub, 1)
    post_hook = _make_grad_fn_hook(trace_stub, 1, is_accumulate_grad=True)

    assert isinstance(pre_hook((torch.ones(1),)), tuple)
    assert getattr(trace_stub, "_tl_pending_accumulate_grad_fire_records")
    _clear_pending_accumulate_grad_records(trace_stub)
    post_hook((), (torch.ones(1),))

    assert not hasattr(trace_stub, "_tl_pending_accumulate_grad_fire_records")
    assert grad_fn_handle.calls[0].fire_record is None


def test_accumulategrad_post_hook_crashes_on_non_none_return() -> None:
    """PyTorch rejects non-None AccumulateGrad post-hook returns."""

    x = torch.ones(1, requires_grad=True)
    loss = (x * 2).sum()
    accumulate_grad = _first_accumulate_grad(loss)
    accumulate_grad.register_hook(lambda *args: (torch.ones_like(x),))
    with pytest.raises(RuntimeError, match="incorrect number of values"):
        loss.backward()


def test_site_decides_the_signal_of_a_neutral_edit() -> None:
    """A neutral edit mounts backward on tl.grad(site) and forward on the bare site."""

    backward = normalize_hook_plan(tl.grad(tl.func("relu")), tl.clip_norm(0.5))
    forward = normalize_hook_plan(tl.func("relu"), tl.clip_norm(0.5))

    assert len(backward) == 1 and len(forward) == 1
    assert backward[0].metadata["signal"] == "backward"
    assert forward[0].metadata["signal"] == "forward"


def _composition_code(excinfo: pytest.ExceptionInfo[SelectorCompositionError]) -> str | None:
    fields = getattr(excinfo.value, "fields", None) or {}
    return fields.get("code", getattr(excinfo.value, "code", None))


@pytest.mark.parametrize(
    "build",
    [
        lambda: tl.grad_fn(name="relu") & tl.in_module("encoder"),
        lambda: tl.func("relu") & tl.grad_fn(name="relu"),
        lambda: tl.grad_fn(name="relu") & tl.label("relu_1"),
        lambda: tl.without_op() & tl.in_module("encoder"),
        lambda: tl.grad(tl.func("relu")) & tl.grad_fn(name="relu"),
        lambda: tl.grad(tl.func("relu")) & tl.func("relu"),
    ],
    ids=[
        "node_and_module",
        "activation_and_node",
        "node_and_label",
        "without_op_and_module",
        "gradient_and_node",
        "gradient_and_activation",
    ],
)
def test_cross_universe_and_refuses_at_construction(build: Any) -> None:
    """A bare selector never bridges into the node universe; & across universes refuses."""

    with pytest.raises(SelectorCompositionError) as excinfo:
        build()
    assert _composition_code(excinfo) == "selector_cross_universe_and"


def test_pass_qualifier_on_an_activation_selector_refuses() -> None:
    """A pass qualifier needs a gradient or node address."""

    with pytest.raises(SelectorCompositionError) as excinfo:
        tl.func("relu") & tl.in_backward_pass(2)
    assert _composition_code(excinfo) == "selector_signal_mismatch"
    qualified = tl.grad(tl.func("relu")) & tl.in_backward_pass(2)
    assert classify_selector_universe(qualified) == "gradient"


def test_find_sites_gradient_address_returns_forward_sites() -> None:
    """tl.grad(label) resolves to the forward op whose output gradient it names."""

    trace = _logged_backward_trace()
    relu_op = trace.find_sites(tl.func("relu")).first()
    sites = trace.find_sites(tl.grad(tl.label(relu_op.layer_label)))
    assert sites
    assert all(not isinstance(site, GradFn) for site in sites)
    assert {site.layer_label for site in sites} == {relu_op.layer_label}


def test_find_sites_label_alone_returns_op_sites() -> None:
    """A lone label selector keeps the forward Op universe."""

    trace = _logged_backward_trace()
    relu_op = trace.find_sites(tl.func("relu")).first()
    sites = trace.find_sites(tl.label(relu_op.layer_label))
    assert sites
    assert all(not isinstance(site, GradFn) for site in sites)


def test_find_sites_gradient_address_filters_by_module() -> None:
    """Module filters apply inside tl.grad, in the forward universe."""

    trace = _logged_backward_trace()
    sites = trace.find_sites(tl.grad(tl.func("linear") & tl.in_module("encoder")))
    assert sites
    assert all(not isinstance(site, GradFn) for site in sites)
    assert len(sites) == len(trace.find_sites(tl.func("linear") & tl.in_module("encoder")))


def test_backward_grad_kind_and_pass_selectors_resolve_real_sites() -> None:
    """grad_input, grad_output, and in_backward_pass resolve over GradFn calls."""

    trace = _logged_backward_trace()

    grad_input_sites = trace.find_sites(tl.grad_input(), max_fanout=100)
    grad_output_sites = trace.find_sites(tl.grad_output(), max_fanout=100)
    pass_sites = trace.find_sites(tl.in_backward_pass(1), max_fanout=100)

    assert grad_input_sites
    assert grad_output_sites
    assert pass_sites
    assert all(isinstance(site, GradFn) for site in grad_input_sites)
    assert all(isinstance(site, GradFn) for site in grad_output_sites)
    assert all(isinstance(site, GradFn) for site in pass_sites)


def test_backward_intervention_replacement_registers_higher_order_terminal() -> None:
    """Double-backward terminal discovery scans post-intervention gradients."""

    def double_grad_input(
        grad_input: tuple[torch.Tensor | None, ...],
        *,
        grad_output: tuple[torch.Tensor | None, ...] | None,
        grad_fn_handle: GradFn,
        call_index: int,
        run_ctx: dict[str, Any],
    ) -> tuple[torch.Tensor | None, ...]:
        """Return a differentiable replacement grad tuple."""

        del grad_output, grad_fn_handle, call_index, run_ctx
        return tuple(None if grad is None else grad * 2 for grad in grad_input)

    double_grad_input.signal = "backward"  # type: ignore[attr-defined]
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(
        _PowModel(),
        x,
        capture=tl.options.CaptureOptions(save_grads="all", backward_ready=True),
    )
    trace._attach_hooks(tl.grad_fn(name="pow"), double_grad_input, confirm_mutation=True)
    trace.log_backward(trace[trace.output_layers[0]].out, retain_graph=True, create_graph=True)
    pow_sites = [site for site in trace._grad_fn_logs.values() if site.type == "pow"]

    assert pow_sites
    assert any(
        site.creator_object_id == pow_sites[0].grad_fn_object_id
        for site in trace._grad_fn_logs.values()
        if site is not pow_sites[0]
    )


def test_selector_universe_classification() -> None:
    """Bare selectors are activations, tl.grad is a gradient, grad_fn selectors are nodes."""

    assert classify_selector_universe(tl.func("relu")) == "activation"
    assert classify_selector_universe(tl.grad(tl.label("relu_1"))) == "gradient"
    assert classify_selector_universe(tl.grad_fn(name="relu")) == "node"
    assert classify_selector_universe(tl.grad_fn(name="relu") & tl.in_backward_pass(1)) == "node"


def _run_identity_relu_intervention(selector: Any, helper: Any) -> tuple[torch.Tensor, list[Any]]:
    """Run the deterministic ReLU model with one backward intervention.

    Parameters
    ----------
    selector:
        Gradient (``tl.grad``) or node selector used in ``tl.when``.
    helper:
        Intervention helper to apply.

    Returns
    -------
    tuple[torch.Tensor, list[Any]]
        Input-layer gradient and backward fire records.
    """

    x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    trace = tl.trace(
        _IdentityReluModel(),
        x,
        capture=tl.options.CaptureOptions(backward_ready=True, save_grads=True),
        intervene=tl.when(selector, helper),
    )
    trace.backward(trace[trace.output_layers[0]].out, retain_graph=True)
    records = [record for record in trace._intervention_spec.records if record.signal == "backward"]
    return trace[trace.input_layers[0]].grad, records


@pytest.mark.parametrize("selector", [tl.grad(tl.func("relu")), tl.grad_fn(name="relu")])
def test_gradient_action_family_changes_only_gradient_path(selector: Any) -> None:
    """Neutral edits change the gradient through the gradient and node universes."""

    zeroed, zero_records = _run_identity_relu_intervention(selector, tl.zero_ablate())
    scaled_grad, scaled_records = _run_identity_relu_intervention(selector, tl.scale(2.0))
    clamped_grad, clamped_records = _run_identity_relu_intervention(
        selector, tl.clamp(low=0, high=0)
    )
    clipped_grad, clipped_records = _run_identity_relu_intervention(selector, tl.clip_norm(0.1))
    noisy_grad, noisy_records = _run_identity_relu_intervention(selector, tl.add_noise(0.5, seed=7))

    assert torch.equal(zeroed, torch.zeros_like(zeroed))
    assert torch.equal(scaled_grad, torch.full_like(scaled_grad, 2.0))
    assert torch.equal(clamped_grad, torch.zeros_like(clamped_grad))
    assert (
        0
        < torch.linalg.vector_norm(clipped_grad)
        < torch.linalg.vector_norm(torch.ones_like(clipped_grad))
    )
    assert not torch.equal(noisy_grad, torch.ones_like(noisy_grad))
    assert all(
        records
        for records in (
            zero_records,
            scaled_records,
            clamped_records,
            clipped_records,
            noisy_records,
        )
    )


def test_activation_only_edit_on_gradient_site_refuses_at_capture() -> None:
    """An edit that reads recorded activations refuses typed on tl.grad(...), never warns."""

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(Exception) as excinfo:
            tl.trace(
                _IdentityReluModel(),
                torch.ones(1, 3, requires_grad=True),
                capture=tl.options.CaptureOptions(backward_ready=True, save_grads=True),
                intervene=tl.when(
                    tl.grad(tl.func("relu")), tl.mean_ablate(torch.ones(3), dim=None)
                ),
            )
    fields = getattr(excinfo.value, "fields", None) or {}
    assert fields.get("code", getattr(excinfo.value, "code", None)) == "edit_signal_mismatch"


def test_backward_zero_match_selector_warns_after_backward() -> None:
    """Unmatched backward selectors report instead of silently no-oping."""
    x = torch.ones(1, 3, requires_grad=True)
    trace = tl.trace(
        _IdentityReluModel(),
        x,
        capture=tl.options.CaptureOptions(backward_ready=True, save_grads=True),
        intervene=tl.when(tl.grad_fn(name="NoSuchGradFnClassZZZ"), tl.zero_ablate()),
    )

    with pytest.warns(UserWarning, match="matched zero sites; no intervention fired"):
        trace.log_backward(trace[trace.output_layers[0]].out)


def test_matching_backward_selector_does_not_warn_zero_match() -> None:
    """The deferred selector ledger records a real GradFn intervention fire."""
    x = torch.ones(1, 3, requires_grad=True)
    trace = tl.trace(
        _IdentityReluModel(),
        x,
        capture=tl.options.CaptureOptions(backward_ready=True, save_grads=True),
        intervene=tl.when(tl.grad_fn(name="relu"), tl.zero_ablate()),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace.log_backward(trace[trace.output_layers[0]].out)

    assert not any("matched zero sites" in str(item.message) for item in caught)


@pytest.mark.parametrize("selector", [tl.grad(tl.func("relu")), tl.grad_fn(name="relu")])
def test_callable_fires_and_can_replace_gradient(selector: Any) -> None:
    """A plain callable on a gradient or node site replaces the gradient."""

    calls: list[torch.Tensor] = []

    def _zero_hook(grad: torch.Tensor, *, ctx: Any) -> torch.Tensor:
        """Record and zero a tensor gradient."""

        del ctx
        calls.append(grad.detach().clone())
        return torch.zeros_like(grad)

    grad, records = _run_identity_relu_intervention(selector, _zero_hook)

    assert calls
    assert torch.equal(grad, torch.zeros_like(grad))
    assert records


@pytest.mark.parametrize(
    "selector",
    [
        tl.grad_fn(name="ReluBackward0"),
        tl.without_op(),
        tl.label("relu_back_1_1"),
        tl.grad(tl.func("relu")),
        tl.grad(tl.label("relu_1_2")) & tl.in_backward_pass(2),
    ],
)
def test_backward_selector_target_spec_round_trip(selector: Any) -> None:
    """Backward selectors round-trip through resolver and hook target specs."""

    target = selector.to_target_spec()
    frozen = target.freeze()
    rebuilt_resolver = selector_from_spec(
        frozen.selector_kind,
        frozen.selector_value,
        dict(frozen.metadata),
        lifecycle="site",
    )
    rebuilt_hook = selector_from_spec(
        target.selector_kind,
        target.selector_value,
        target.metadata,
        lifecycle="live",
    )
    assert rebuilt_resolver.selector_kind == selector.selector_kind
    assert rebuilt_resolver.selector_value == selector.selector_value
    assert rebuilt_hook.selector_kind == selector.selector_kind
    assert rebuilt_hook.selector_value == selector.selector_value


def test_clip_norm_on_gradient_site_clips_the_output_gradient() -> None:
    """clip_norm rescales the site's output gradient (norm sqrt(3)) down to max_norm."""

    grad, records = _run_identity_relu_intervention(tl.grad(tl.func("relu")), tl.clip_norm(0.5))
    assert records
    torch.testing.assert_close(torch.linalg.vector_norm(grad), torch.tensor(0.5))


def test_add_noise_on_gradient_site_is_seeded_reproducible() -> None:
    """Seeded add_noise on a gradient site gives the same gradient twice."""

    first, _ = _run_identity_relu_intervention(tl.grad(tl.func("relu")), tl.add_noise(0.1, seed=7))
    second, _ = _run_identity_relu_intervention(tl.grad(tl.func("relu")), tl.add_noise(0.1, seed=7))
    assert torch.equal(first, second)
    assert not torch.equal(first, torch.ones_like(first))


def test_clamp_on_gradient_site_clamps_elementwise() -> None:
    """clamp on tl.grad(relu) clamps each element of the unit output gradient."""

    grad, _records = _run_identity_relu_intervention(
        tl.grad(tl.func("relu")), tl.clamp(low=-0.5, high=0.5)
    )
    assert torch.equal(grad, torch.full_like(grad, 0.5))


class _GuidedCNN(nn.Module):
    """Two-ReLU CNN used by the guided-backprop recipe acceptance tests."""

    def __init__(self, inplace: bool) -> None:
        """Initialize the model.

        Parameters
        ----------
        inplace:
            Whether the ReLU activations run in-place (the ResNet idiom).
        """

        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, padding=1)
        self.r1 = nn.ReLU(inplace=inplace)
        self.c2 = nn.Conv2d(8, 8, 3, padding=1)
        self.r2 = nn.ReLU(inplace=inplace)
        self.pool = nn.AdaptiveAvgPool2d(2)
        self.fc = nn.Linear(32, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""

        h = self.r1(self.c1(x))
        h = self.r2(self.c2(h))
        return self.fc(torch.flatten(self.pool(h), 1))


def _guided_backprop_via_torchlens(model: nn.Module, x: torch.Tensor, target: int) -> torch.Tensor:
    """Return the guided-backprop input gradient using the one-off recipe."""

    # Clamping the ReLU's output gradient then masking equals masking then clamping (the
    # classic grad_input recipe), because the mask is 0 or 1.
    guided = tl.when(tl.grad(tl.func("relu")), lambda g, *, ctx: g.clamp(min=0))
    trace = tl.trace(
        model,
        x.clone().requires_grad_(True),
        capture=tl.options.CaptureOptions(
            backward_ready=True, save_grads=True, save_mode="reference"
        ),
        intervene=guided,
    )
    out = trace[trace.output_layers[0]].out
    trace.backward(out[:, target].sum(), retain_graph=True)
    return trace[trace.input_layers[0]].grad


def _guided_backprop_reference(model: nn.Module, x: torch.Tensor, target: int) -> torch.Tensor:
    """Return the classic module-hook guided-backprop gradient (non-inplace only)."""

    handles = [
        module.register_full_backward_hook(
            lambda mod, gin, gout: tuple(g.clamp(min=0) for g in gin)
        )
        for module in model.modules()
        if isinstance(module, nn.ReLU)
    ]
    leaf = x.clone().requires_grad_(True)
    model(leaf)[:, target].sum().backward()
    for handle in handles:
        handle.remove()
    return leaf.grad


def test_guided_backprop_recipe_matches_module_hook_reference() -> None:
    """The one-off guided-backprop recipe reproduces the classic hook recipe exactly."""

    torch.manual_seed(0)
    model = _GuidedCNN(inplace=False).eval()
    x = torch.randn(2, 3, 8, 8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grad = _guided_backprop_via_torchlens(model, x, target=3)
    reference = _guided_backprop_reference(model, x, target=3)
    assert torch.equal(grad, reference)
    plain_leaf = x.clone().requires_grad_(True)
    model(plain_leaf)[:, 3].sum().backward()
    assert not torch.allclose(grad, plain_leaf.grad)


def test_guided_backprop_recipe_works_on_inplace_relu() -> None:
    """The recipe handles in-place ReLU models the classic hook recipe cannot.

    ``register_full_backward_hook`` + in-place ReLU raises torch's
    view-inplace ``BackwardHookFunction`` error, so the reference gradient is
    computed on a non-inplace twin sharing the same parameters.
    """

    torch.manual_seed(0)
    model = _GuidedCNN(inplace=True).eval()
    x = torch.randn(2, 3, 8, 8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grad = _guided_backprop_via_torchlens(model, x, target=1)
    twin = _GuidedCNN(inplace=False).eval()
    twin.load_state_dict(model.state_dict())
    reference = _guided_backprop_reference(twin, x, target=1)
    assert torch.equal(grad, reference)
