"""Gradient sites on the hook runtime: ``tl.grad(site)`` edits the output gradient.

Covers the packet-P2 criteria of the gradient-sites spec: output-gradient semantics,
connectedness, read/write agreement, neutral edits, the activation-only refusal,
``HookContext.signal`` / ``state`` / ``run_state``, mixed sites and capture-time arming.
Every expected value is derived by hand from the model's closed form, never from a
second TorchLens door.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.intervention.hooks import HookContext

_CAPTURE = tl.options.CaptureOptions(backward_ready=True, save_grads="all")


class _Double(nn.Module):
    """``h = fc(x)``, ``y = 2 * h``; the loss is ``y.sum()``."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        return h * 2


class _TwoLayer(nn.Module):
    """``fc2(relu(fc1(x)))``: fc1's parameters reach the loss only through the relu."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(3, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _label(trace: object, prefix: str) -> str:
    """Return the first layer label starting with ``prefix``."""

    return next(lab for lab in trace.layer_labels if lab.startswith(prefix))  # type: ignore[attr-defined]


def _double() -> tuple[_Double, torch.Tensor]:
    torch.manual_seed(0)
    return _Double(), torch.randn(4, 4)


def _backward(trace: object) -> None:
    out = trace[trace.output_layers[0]].out  # type: ignore[attr-defined,index]
    trace.backward(out.sum())  # type: ignore[attr-defined]


def _backward_records(trace: object) -> list[object]:
    spec = getattr(trace, "_intervention_spec", None)
    return [r for r in (getattr(spec, "records", None) or ()) if r.signal == "backward"]


def test_clamp_edits_the_output_gradient_not_grad_input() -> None:
    model, x = _double()
    mul = _label(tl.trace(model, x, capture=_CAPTURE), "mul")
    lin = _label(tl.trace(model, x, capture=_CAPTURE), "linear")
    trace = tl.trace(
        model, x, capture=_CAPTURE, intervene=tl.when(tl.grad(mul), tl.clamp(low=-0.5, high=0.5))
    )
    _backward(trace)
    # d loss / d y = 1 everywhere; clamped to 0.5; d y / d h = 2, so h.grad = 1.0.
    # The grad_input reading would clamp 2.0 to 0.5 instead.
    assert torch.equal(trace[lin].grad, torch.ones(4, 4))
    records = _backward_records(trace)
    assert records and {r.grad_kind for r in records} == {"grad_output"}
    assert {r.site_label for r in records} == {mul}


def test_scale_zero_disconnects_only_the_upstream_parameters() -> None:
    torch.manual_seed(1)
    model, x = _TwoLayer(), torch.randn(6, 3)
    plain = tl.trace(model, x, capture=_CAPTURE)
    relu = _label(plain, "relu")
    model.zero_grad(set_to_none=True)
    _backward(plain)
    fc2_plain = model.fc2.weight.grad.clone()
    model.zero_grad(set_to_none=True)
    edited = tl.trace(model, x, capture=_CAPTURE, intervene=tl.when(tl.grad(relu), tl.scale(0)))
    _backward(edited)
    assert torch.equal(model.fc1.weight.grad, torch.zeros_like(model.fc1.weight))
    assert torch.equal(model.fc1.bias.grad, torch.zeros_like(model.fc1.bias))
    assert torch.equal(model.fc2.weight.grad, fc2_plain)
    for label in plain.layer_labels:
        assert torch.equal(edited[label].out, plain[label].out), label


def test_saved_grad_is_the_edit_chains_final_value() -> None:
    torch.manual_seed(2)
    model, x = _TwoLayer(), torch.randn(6, 3)
    plain = tl.trace(model, x, capture=_CAPTURE)
    relu = _label(plain, "relu")
    _backward(plain)
    edited = tl.trace(model, x, capture=_CAPTURE, intervene=tl.when(tl.grad(relu), tl.scale(0.25)))
    _backward(edited)
    assert torch.allclose(edited[relu].grad, 0.25 * plain[relu].grad)


@pytest.mark.parametrize(
    ("name", "edit", "expected_h_grad"),
    [
        ("scale", tl.scale(0.0), 0.0),
        ("zero_ablate", tl.zero_ablate(), 0.0),
        ("clamp", tl.clamp(low=-0.1, high=0.1), 0.2),
        ("callable", lambda g, *, ctx: g * 3, 6.0),
        # ones(4, 4) has 2-norm 4.0; clipped to norm 1.0 it is 0.25 everywhere.
        ("clip_norm", tl.clip_norm(1.0), 0.5),
    ],
)
def test_neutral_edits_apply_on_gradient_sites(
    name: str, edit: object, expected_h_grad: float
) -> None:
    model, x = _double()
    base = tl.trace(model, x, capture=_CAPTURE)
    mul, lin = _label(base, "mul"), _label(base, "linear")
    trace = tl.trace(model, x, capture=_CAPTURE, intervene=tl.when(tl.grad(mul), edit))
    _backward(trace)
    assert torch.allclose(trace[lin].grad, torch.full((4, 4), expected_h_grad)), name


def test_clip_norm_reaches_norm_one_on_gradient_and_activation_sites() -> None:
    model, x = _double()
    mul = _label(tl.trace(model, x, capture=_CAPTURE), "mul")
    on_grad = tl.trace(
        model, x, capture=_CAPTURE, intervene=tl.when(tl.grad(mul), tl.clip_norm(1.0))
    )
    _backward(on_grad)
    assert torch.linalg.vector_norm(on_grad[mul].grad).item() == pytest.approx(1.0)
    on_act = tl.trace(
        model, x, capture=_CAPTURE, intervene=tl.when(tl.func("mul"), tl.clip_norm(1.0))
    )
    assert torch.linalg.vector_norm(on_act[mul].out).item() == pytest.approx(1.0)


def test_add_noise_and_steer_change_the_upstream_gradient_deterministically() -> None:
    model, x = _double()
    base = tl.trace(model, x, capture=_CAPTURE)
    mul, lin = _label(base, "mul"), _label(base, "linear")
    grads = []
    for _ in range(2):
        trace = tl.trace(
            model, x, capture=_CAPTURE, intervene=tl.when(tl.grad(mul), tl.add_noise(0.1, seed=0))
        )
        _backward(trace)
        grads.append(trace[lin].grad)
    assert torch.equal(grads[0], grads[1])
    assert not torch.equal(grads[0], torch.full((4, 4), 2.0))
    direction = torch.ones(4, 4)
    steered = tl.trace(
        model, x, capture=_CAPTURE, intervene=tl.when(tl.grad(mul), tl.steer(direction, 0.5))
    )
    _backward(steered)
    # d loss / d y = 1, plus 0.5 * ones = 1.5; times dy/dh = 2 gives 3.0.
    assert torch.allclose(steered[lin].grad, torch.full((4, 4), 3.0))


@pytest.mark.parametrize("edit_name", ["mean_ablate", "splice_module"])
def test_activation_only_edit_on_gradient_site_refuses_at_capture(edit_name: str) -> None:
    model, x = _double()
    mul = _label(tl.trace(model, x, capture=_CAPTURE), "mul")
    edit = tl.mean_ablate() if edit_name == "mean_ablate" else tl.splice_module(nn.Identity())
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(InvalidArgumentError) as exc_info:
            tl.trace(model, x, capture=_CAPTURE, intervene=tl.when(tl.grad(mul), edit))
    assert exc_info.value.fields["code"] == "edit_signal_mismatch"


def test_patch_from_on_gradient_site_refuses_before_attaching() -> None:
    model, x = _double()
    trace = tl.trace(model, x, capture=_CAPTURE)
    donor = tl.trace(model, x * 2, capture=_CAPTURE)
    mul = _label(trace, "mul")
    hook_specs_before = len(
        getattr(getattr(trace, "_intervention_spec", None), "hook_specs", ()) or ()
    )
    audit_before = list(trace.intervention_audit)
    with pytest.raises(InvalidArgumentError) as exc_info:
        trace.attach_hooks(tl.grad(mul), tl.patch_from(donor))
    assert exc_info.value.fields["code"] == "edit_signal_mismatch"
    hook_specs_after = len(
        getattr(getattr(trace, "_intervention_spec", None), "hook_specs", ()) or ()
    )
    assert hook_specs_after == hook_specs_before
    assert list(trace.intervention_audit) == audit_before


def test_hook_context_signal_state_and_run_state() -> None:
    model, x = _double()
    seen: list[tuple[str, object, int, object, object]] = []

    def observe(tensor: torch.Tensor, *, ctx: HookContext) -> None:
        ctx.state["fires"] = ctx.state.get("fires", 0) + 1
        ctx.run_state.setdefault("signals", []).append(ctx.signal)
        kwargs = dict(ctx.kwargs)
        seen.append(
            (
                ctx.signal,
                kwargs.get("grad_kind"),
                ctx.state["fires"],
                kwargs.get("backward_pass_index"),
                kwargs.get("tuple_index"),
            )
        )
        return None

    # Live capture cannot match a finalized label, so the activation half is
    # spelled by function; the gradient half resolves after the forward.
    site = tl.func("mul")
    trace = tl.trace(model, x, capture=_CAPTURE, intervene=tl.when(tl.grad(site) | site, observe))
    out = trace[trace.output_layers[0]].out
    trace.backward(out.sum(), retain_graph=True)
    trace.backward(out.sum())
    forward = [row for row in seen if row[0] == "forward"]
    backward = [row for row in seen if row[0] == "backward"]
    assert len(forward) == 1 and forward[0][1] is None
    assert [row[1] for row in backward] == ["grad_output", "grad_output"]
    # ctx.state persists across fires of the same (site, hook): 1 then 2 per pass.
    assert [row[2] for row in backward] == [1, 2]
    assert [row[3] for row in backward] == [1, 2]
    assert all(row[4] is None for row in backward)
    for name in ("ctx", "run_ctx", "direction"):
        assert name not in HookContext.__dataclass_fields__
    for name in ("signal", "state", "run_state"):
        assert name in HookContext.__dataclass_fields__


def test_capture_time_gradient_edit_arms_then_fires_on_backward() -> None:
    model, x = _double()
    base = tl.trace(model, x, capture=_CAPTURE)
    mul, lin = _label(base, "mul"), _label(base, "linear")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        trace = tl.trace(model, x, capture=_CAPTURE, intervene=tl.when(tl.grad(mul), tl.scale(0)))
    events = [
        row
        for row in trace.state_history
        if isinstance(row, dict) and row.get("op") == "intervention_event"
    ]
    assert events[-1]["status"] == "armed"
    assert events[-1]["armed_edits"] == 1 and events[-1]["awaiting"] == "backward"
    assert not _backward_records(trace)
    _backward(trace)
    assert torch.equal(trace[lin].grad, torch.zeros(4, 4))
    assert len(_backward_records(trace)) == 1


def test_gradient_edit_fires_at_a_site_save_grads_does_not_select() -> None:
    model, x = _double()
    base = tl.trace(model, x, capture=_CAPTURE)
    mul, lin = _label(base, "mul"), _label(base, "linear")
    selective = tl.options.CaptureOptions(backward_ready=True, save_grads=[lin])
    trace = tl.trace(model, x, capture=selective, intervene=tl.when(tl.grad(mul), tl.scale(0)))
    _backward(trace)
    # The edit fired at mul (not selected for saving), so lin's saved gradient is zero.
    assert torch.equal(trace[lin].grad, torch.zeros(4, 4))
    assert len(_backward_records(trace)) == 1


def test_gradient_rule_never_edits_the_activation_at_capture() -> None:
    model, x = _double()
    plain = tl.trace(model, x, capture=_CAPTURE)
    mul, lin = _label(plain, "mul"), _label(plain, "linear")
    # tl.func("mul") is live-matchable: the forward predicate must still skip it,
    # because the rule addresses the gradient, not the activation.
    trace = tl.trace(
        model, x, capture=_CAPTURE, intervene=tl.when(tl.grad(tl.func("mul")), tl.scale(0))
    )
    assert torch.equal(trace[mul].out, plain[mul].out)
    _backward(trace)
    assert torch.equal(trace[lin].grad, torch.zeros(4, 4))
