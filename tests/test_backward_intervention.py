"""Backward intervention selector, helper, and grad_fn_handle dispatch tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens.backends.torch.backward import _make_grad_fn_hook, _make_grad_fn_prehook
from torchlens.data_classes.grad_fn_log import GradFn
from torchlens.intervention.errors import HelperMountError, SelectorCompositionError
from torchlens.intervention.helpers import _helper_spec
from torchlens.intervention.hooks import _selector_from_target_spec, normalize_hook_plan
from torchlens.intervention.resolver import _selector_from_spec, _selector_resolution_direction


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


@dataclass
class _TraceStub:
    """Weakref-able trace stub for direct hook-factory tests."""

    grad_fn_logs: dict[int, GradFn]
    _grad_layer_nums_to_save: str = "all"
    last_run: dict[str, Any] | None = None


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
    trace = tl.trace(model, x, gradients_to_save="all")
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
        grad_fn_type="relu",
        grad_fn_type_num=1,
        grad_fn_total_num=1,
        is_intervening=True,
    )
    return _TraceStub({1: grad_fn_handle}), grad_fn_handle


def test_accumulategrad_post_hook_logs_grad_fn_call() -> None:
    """AccumulateGrad post-hooks still record call logs."""

    trace = _logged_backward_trace()
    accumulate_logs = [
        gfl for gfl in trace.grad_fn_logs.values() if gfl.class_name == "AccumulateGrad"
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
        kind="backward",
        factory=factory,
        metadata={"mount_shape": "tuple"},
    )
    _state._active_hook_plan = normalize_hook_plan(tl.grad_fn(type="relu"), helper_spec)
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
    _state._active_hook_plan = normalize_hook_plan(tl.grad_fn_label("missing"), tl.grad_clamp(0, 1))
    hook = _make_grad_fn_hook(trace_stub, 1)
    assert hook((torch.ones(1),), (torch.ones(1),)) is None


def test_grad_fn_hook_returns_tuple_when_mutating() -> None:
    """Grad_fn hook returns a tuple when a matching helper mutates gradients."""

    trace_stub, _grad_fn_log = _hook_trace()
    _state._active_hook_plan = normalize_hook_plan(tl.grad_fn(type="relu"), tl.grad_clamp(0, 0))
    hook = _make_grad_fn_hook(trace_stub, 1)
    result = hook((torch.ones(1),), (torch.ones(1),))
    assert isinstance(result, tuple)
    assert torch.equal(result[0], torch.zeros(1))


def test_helper_requires_grad_output_at_accumulategrad_raises_helpermounterror() -> None:
    """Helpers that require grad_output cannot mount on backward prehook sites."""

    helper = _helper_spec(
        "needs_output",
        kind="backward",
        factory=lambda: lambda grad_input, **kwargs: grad_input,
        metadata={"mount_shape": "tuple", "requires_grad_output": True},
    )
    with pytest.raises(HelperMountError):
        normalize_hook_plan(tl.grad_fn(type="AccumulateGrad"), helper)


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
        tl.grad_fn(type="relu"),
        _helper_spec(
            "record_index", kind="backward", factory=factory, metadata={"mount_shape": "tuple"}
        ),
    )
    pre_hook = _make_grad_fn_prehook(trace_stub, 1)
    post_hook = _make_grad_fn_hook(trace_stub, 1, is_accumulate_grad=True)
    pre_hook((torch.ones(1),))
    post_hook((), (torch.ones(1),))
    assert seen == [1]
    assert grad_fn_handle.ops[1].call_index == 1


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
        tl.grad_fn(type="relu"),
        _helper_spec(
            "record_index", kind="backward", factory=factory, metadata={"mount_shape": "tuple"}
        ),
    )
    pre_hook = _make_grad_fn_prehook(trace_stub, 1)
    post_hook = _make_grad_fn_hook(trace_stub, 1, is_accumulate_grad=True)
    for _idx in range(3):
        pre_hook((torch.ones(1),))
        post_hook((), (torch.ones(1),))
    assert seen == [1, 2, 3]
    assert tuple(grad_fn_handle.ops) == (1, 2, 3)


def test_accumulategrad_post_hook_crashes_on_non_none_return() -> None:
    """PyTorch rejects non-None AccumulateGrad post-hook returns."""

    x = torch.ones(1, requires_grad=True)
    loss = (x * 2).sum()
    accumulate_grad = _first_accumulate_grad(loss)
    accumulate_grad.register_hook(lambda *args: (torch.ones_like(x),))
    with pytest.raises(RuntimeError, match="incorrect number of values"):
        loss.backward()


def test_grad_clip_on_forward_selector_raises_helpermounterror() -> None:
    """Tuple-shaped grad_fn_handle helpers cannot mount on forward selectors."""

    with pytest.raises(HelperMountError):
        normalize_hook_plan(tl.label("relu_1"), tl.grad_clip(0.5))


def test_selector_compose_grad_fn_and_in_module_legal_at_construction() -> None:
    """Backward selectors compose with direction-agnostic module filters."""

    selector = tl.grad_fn(type="relu") & tl.in_module("encoder")
    assert selector.selector_kind == "and"


def test_selector_compose_func_and_grad_fn_rejected_at_construction() -> None:
    """Forward-only and backward-only selectors cannot compose."""

    with pytest.raises(SelectorCompositionError, match="forward"):
        tl.func("relu") & tl.grad_fn(type="relu")


def test_selector_compose_grad_fn_and_label_legal_at_construction() -> None:
    """Backward selectors compose with direction-agnostic labels."""

    selector = tl.grad_fn(type="relu") & tl.label("relu_1")
    assert selector.selector_kind == "and"


def test_selector_resolve_intervening_grad_fn_filtered_with_in_module() -> None:
    """Intervening grad_fns are filtered by module predicates."""

    trace = _logged_backward_trace()
    sites = trace.find_sites(tl.intervening() & tl.in_module("encoder"), max_fanout=100)
    assert len(sites) == 0


def test_find_sites_grad_fn_and_label_returns_grad_fn_sites() -> None:
    """Grad_fn plus label resolves over GradFn sites."""

    trace = _logged_backward_trace()
    relu_op = trace.find_sites(tl.func("relu")).first()
    sites = trace.find_sites(tl.grad_fn(type="relu") & tl.label(relu_op.layer_label))
    assert sites
    assert all(isinstance(site, GradFn) for site in sites)
    assert all(site.op is not None and site.op.layer_label == relu_op.layer_label for site in sites)


def test_find_sites_label_alone_returns_op_sites() -> None:
    """A lone label selector keeps the forward Op universe."""

    trace = _logged_backward_trace()
    relu_op = trace.find_sites(tl.func("relu")).first()
    sites = trace.find_sites(tl.label(relu_op.layer_label))
    assert sites
    assert all(not isinstance(site, GradFn) for site in sites)


def test_find_sites_grad_fn_and_in_module_filters_intervening() -> None:
    """Grad_fn plus module filter returns paired grad_fns only."""

    trace = _logged_backward_trace()
    sites = trace.find_sites(tl.grad_fn(type="addmm") & tl.in_module("encoder"))
    assert sites
    assert all(isinstance(site, GradFn) for site in sites)
    assert all(site.op is not None for site in sites)


def test_selector_resolution_direction_grad_fn_and_label() -> None:
    """Composite traversal uses selectors and returns backward direction."""

    assert (
        _selector_resolution_direction(tl.grad_fn(type="relu") & tl.label("relu_1")) == "backward"
    )


@pytest.mark.parametrize(
    "selector",
    [tl.grad_fn(type="ReluBackward0"), tl.intervening(), tl.grad_fn_label("relu_back_1_1")],
)
def test_backward_selector_target_spec_round_trip(selector: Any) -> None:
    """Backward selectors round-trip through resolver and hook target specs."""

    target = selector.to_target_spec()
    frozen = target.freeze()
    rebuilt_resolver = _selector_from_spec(
        frozen.selector_kind,
        frozen.selector_value,
        dict(frozen.metadata),
    )
    rebuilt_hook = _selector_from_target_spec(target)
    assert rebuilt_resolver.selector_kind == selector.selector_kind
    assert rebuilt_resolver.selector_value == selector.selector_value
    assert rebuilt_hook.selector_kind == selector.selector_kind
    assert rebuilt_hook.selector_value == selector.selector_value


def test_grad_clip_helper_norm_per_node() -> None:
    """grad_clip clips per tensor in the grad_input tuple."""

    helper = tl.grad_clip(0.5)()
    result = helper(
        (torch.ones(4),), grad_output=None, grad_fn_handle=None, call_index=1, run_ctx={}
    )
    assert result is not None
    assert torch.linalg.vector_norm(result[0]) <= 0.5001


def test_grad_noise_helper_seeded_reproducible() -> None:
    """Seeded grad_noise helpers are reproducible."""

    helper_a = tl.grad_noise(0.1, seed=7)()
    helper_b = tl.grad_noise(0.1, seed=7)()
    kwargs = {"grad_output": None, "grad_fn_handle": None, "call_index": 1, "run_ctx": {}}
    result_a = helper_a((torch.zeros(3),), **kwargs)
    result_b = helper_b((torch.zeros(3),), **kwargs)
    assert torch.equal(result_a[0], result_b[0])


def test_grad_clamp_helper_elementwise() -> None:
    """grad_clamp clamps each tensor element."""

    helper = tl.grad_clamp(-0.5, 0.5)()
    result = helper(
        (torch.tensor([-2.0, 0.0, 2.0]),),
        grad_output=None,
        grad_fn_handle=None,
        call_index=1,
        run_ctx={},
    )
    assert torch.equal(result[0], torch.tensor([-0.5, 0.0, 0.5]))
