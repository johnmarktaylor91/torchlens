"""The ruled member moves: collapse diagnostics on ``Drawing``, sticky hooks staged through ``do``.

``Trace.collapse_plan`` / ``collapse_schedule`` / ``collapse_order`` left the record
for the ``Drawing`` a forward draw returns; ``Trace.attach_hooks`` and
``clear_hooks`` folded into ``do`` and ``clear_edits``. These tests pin the live
replacements and the typed refusals of the moved spellings.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from example_models import TinyLinearRelu as LinearRelu, TinyReluAdd as ReluAdd

import torchlens as tl
from torchlens._errors import FacadeTeachingError
from torchlens._visualization._trace_collapse import (
    collapse_plan as _collapse_plan,
    collapse_schedule as _collapse_schedule,
)
from torchlens.errors import TorchLensError
from torchlens.io import TraceState
from torchlens.options import CaptureOptions, InterventionOptions


def _identity_hook(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
    del ctx
    return out


def _double_hook(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
    del ctx
    return out * 2


def _relu_label(trace: Any) -> str:
    """Return the label of the relu op (the first internal layer of ``ReluAdd``)."""

    for layer in trace._layer_list:
        if not (layer.is_input or layer.is_output or layer.is_buffer):
            return str(layer.layer_label)
    raise AssertionError("expected an internal layer")


@pytest.fixture
def log() -> Any:
    trace = tl.trace(ReluAdd(), torch.randn(2, 3), capture=CaptureOptions(replayable=True))
    try:
        yield trace
    finally:
        trace.cleanup()


# ---------------------------------------------------------------- Drawing members


def test_forward_drawing_carries_the_plan_it_rendered_from(log: Any, tmp_path: Any) -> None:
    """``Drawing.collapse_plan`` is the plan of the drawing's own mode and context."""

    drawing = log.draw(path=str(tmp_path / "g"), collapse="auto", file_format="dot")
    plan = drawing.collapse_plan
    direct = _collapse_plan(log, "auto", drawing._collapse_context)
    assert plan.total == direct.total
    assert len(plan) == len(direct)
    assert drawing._collapse_context.collapse == "auto"
    assert drawing._collapse_context.vis_mode == "unrolled"


def test_forward_drawing_schedule_runs_from_full_graph_to_max(log: Any, tmp_path: Any) -> None:
    """``Drawing.collapse_schedule`` is the monotone schedule under the drawing's context."""

    drawing = log.draw(path=str(tmp_path / "g"), file_format="dot")
    schedule = drawing.collapse_schedule
    direct = _collapse_schedule(log, drawing._collapse_context)
    assert schedule.steps[0].t == 0.0
    assert schedule.steps[-1].t == 1.0
    assert [step.visible_count for step in schedule.steps] == [
        step.visible_count for step in direct.steps
    ]


def test_rolled_drawing_context_follows_the_draw_call(log: Any, tmp_path: Any) -> None:
    """The stored context is the request the draw resolved, rolled mode included."""

    drawing = log.draw(path=str(tmp_path / "g"), rolled=True, collapse="max", file_format="dot")
    assert drawing._collapse_context.vis_mode == "rolled"
    assert (
        drawing.collapse_plan.nodes == _collapse_plan(log, "max", drawing._collapse_context).nodes
    )


def test_collapse_none_drawing_refuses_a_plan_but_keeps_the_schedule(
    log: Any, tmp_path: Any
) -> None:
    """A ``collapse="none"`` drawing has no plan to show; its schedule still exists."""

    drawing = log.draw(path=str(tmp_path / "g"), file_format="dot")
    with pytest.raises(TorchLensError) as excinfo:
        _ = drawing.collapse_plan
    assert excinfo.value.fields["code"] == "collapse_plan_unavailable"
    assert "collapse='none'" in str(excinfo.value)
    assert drawing.collapse_schedule.steps


def test_backward_drawing_refuses_collapse_diagnostics(tmp_path: Any) -> None:
    """Only a forward drawing carries the collapse members."""

    model = LinearRelu()
    trace = tl.trace(model, torch.randn(2, 4), capture=CaptureOptions(save_grads="all"))
    try:
        trace.backward(trace[trace.output_layers[0]].out.sum())
        drawing = trace.draw(path=str(tmp_path / "g"), graph="backward", file_format="dot")
        for member in ("collapse_plan", "collapse_schedule"):
            with pytest.raises(TorchLensError) as excinfo:
                getattr(drawing, member)
            assert excinfo.value.fields["code"] == "collapse_plan_unavailable"
    finally:
        trace.cleanup()


def test_released_trace_refuses_collapse_diagnostics(tmp_path: Any) -> None:
    """Once the source trace is gone the members refuse typed instead of resurrecting it."""

    trace = tl.trace(ReluAdd(), torch.randn(2, 3), capture=CaptureOptions(replayable=True))
    drawing = trace.draw(path=str(tmp_path / "g"), collapse="auto", file_format="dot")
    trace.cleanup()
    del trace
    import gc

    gc.collect()
    with pytest.raises(TorchLensError) as excinfo:
        _ = drawing.collapse_schedule
    assert excinfo.value.fields["code"] == "collapse_plan_unavailable"
    assert "released" in str(excinfo.value)


def test_trace_collapse_spellings_redirect_to_draw(log: Any) -> None:
    """The removed ``Trace`` spellings name the ``Drawing`` members."""

    for removed in ("collapse_plan", "collapse_schedule", "collapse_order"):
        with pytest.raises(FacadeTeachingError) as excinfo:
            getattr(log, removed)
        assert "draw(" in str(excinfo.value)
    assert log.module_collapse_order  # the per-module ranking stays on the record


# ------------------------------------------------------------- sticky hooks via do


def test_recipe_with_set_only_stages_one_sticky_hook_without_running(log: Any) -> None:
    """``do(tl.when(...), engine="set_only")`` stages the hook and propagates nothing."""

    revision = log._spec_revision
    before = log[_relu_label(log)].out.clone()
    log.do(
        tl.when(tl.func("relu"), _double_hook),
        intervention=InterventionOptions(engine="set_only", confirm_mutation=True),
    )
    assert len(log._intervention_spec.hook_specs) == 1
    assert log._spec_revision == revision + 1
    assert log.state is TraceState.SPEC_STALE
    torch.testing.assert_close(log[_relu_label(log)].out, before)


def test_do_with_a_hook_and_set_only_stages_it_sticky(log: Any) -> None:
    """``do(site, hook, engine="set_only")`` stages a sticky hook, not a set value."""

    before = log[_relu_label(log)].out.clone()
    log.do(
        tl.func("relu"),
        tl.scale(2.0),
        intervention=InterventionOptions(engine="set_only", confirm_mutation=True),
    )
    assert len(log._intervention_spec.hook_specs) == 1
    assert not log._intervention_spec.target_value_specs
    torch.testing.assert_close(log[_relu_label(log)].out, before)
    # The next propagating do fires the staged hook (push folded into do).
    log.do(
        tl.func("relu"),
        tl.scale(1.0),
        intervention=InterventionOptions(confirm_mutation=True),
    )
    torch.testing.assert_close(log[_relu_label(log)].out, before * 2)


def test_do_with_a_hook_propagates_at_once(log: Any) -> None:
    """``do(site, hook)`` attaches the sticky hook and replays in the same call."""

    before = log[_relu_label(log)].out.clone()
    log.do(tl.func("relu"), _double_hook, intervention=InterventionOptions(confirm_mutation=True))
    assert len(log._intervention_spec.hook_specs) == 1
    torch.testing.assert_close(log[_relu_label(log)].out, before * 2)


def test_clear_edits_accepts_the_handle_tuple_do_staged(log: Any) -> None:
    """``clear_edits(handle=)`` takes one id or the tuple of ids one staging call made."""

    first = log._attach_hooks(tl.func("relu"), _identity_hook, confirm_mutation=True)
    second = log._attach_hooks(tl.func("__add__"), _identity_hook, confirm_mutation=True)
    assert isinstance(first, tuple) and len(first) == 1
    assert len(log._intervention_spec.hook_specs) == 2

    log.clear_edits(handle=first, confirm_mutation=True)
    remaining = log._intervention_spec.hook_specs
    assert len(remaining) == 1
    assert remaining[0].handle in second

    log.clear_edits(handle=list(second), confirm_mutation=True)
    assert log._intervention_spec.hook_specs == []


def test_hook_spellings_redirect_to_do_and_clear_edits(log: Any) -> None:
    """``attach_hooks`` and ``clear_hooks`` refuse typed and name the live doors."""

    with pytest.raises(FacadeTeachingError) as excinfo:
        _ = log.attach_hooks
    assert "do(" in str(excinfo.value)
    with pytest.raises(FacadeTeachingError) as excinfo:
        _ = log.clear_hooks
    assert "clear_edits" in str(excinfo.value)
    with pytest.raises(FacadeTeachingError) as excinfo:
        _ = log.save_new_outs
    assert "run(" in str(excinfo.value)


class _ShapeBranch(torch.nn.Module):
    """Model whose graph gains a relu only when the feature dimension is 4."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + 1
        if x.shape[1] == 4:
            out = torch.relu(out)
        return out * 2


class _SignBranch(torch.nn.Module):
    """Model whose graph gains a relu only when the input sums positive (same shape, new graph)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + 1
        if bool(x.sum() > 0):
            out = torch.relu(out)
        return out * 2


def test_save_new_outs_redirect_names_the_graph_checked_refresh() -> None:
    """The ``save_new_outs`` refusal names ``run(inputs=)``, and that door refuses a changed graph.

    ``save_new_outs`` refreshed payloads against the recorded graph and raised a
    ``ValueError`` when an input changed it. The legacy ``run(model, x)`` re-captures
    and only warns on divergence, so a redirect to it would drop the mismatch raise.
    ``run(inputs=)`` refuses the training tutorial's shape-diverging input at the input
    contract and a same-shape input that changes the graph at the refresh compare.
    """

    shape_model = _ShapeBranch()  # run(inputs=) needs the live model; the trace holds it weakly
    trace = tl.trace(
        shape_model,
        torch.ones(2, 4, requires_grad=True),
        capture=CaptureOptions(detach_saved_activations=True, seed=7),
    )
    try:
        with pytest.raises(FacadeTeachingError) as excinfo:
            _ = trace.save_new_outs
        assert "run(inputs=" in str(excinfo.value)
        with pytest.raises(ValueError, match="does not match"):
            trace.run(inputs=torch.ones(2, 5, requires_grad=True), seed=7)
        assert trace.options.detach_saved_activations is True
    finally:
        trace.cleanup()

    sign_model = _SignBranch()
    trace = tl.trace(sign_model, torch.ones(2, 4), capture=CaptureOptions(seed=7))
    try:
        with pytest.raises(ValueError) as mismatch:
            trace.run(inputs=-torch.ones(2, 4), seed=7)
        refusal: Any = mismatch.value
        assert refusal.fields["code"] == "call_structure_mismatch", refusal
    finally:
        trace.cleanup()
