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


@pytest.mark.parametrize("edit_name", ["splice_module"])
def test_activation_only_edit_on_gradient_site_refuses_at_capture(edit_name: str) -> None:
    model, x = _double()
    mul = _label(tl.trace(model, x, capture=_CAPTURE), "mul")
    edit = (
        tl.mean_ablate(dim=None) if edit_name == "mean_ablate" else tl.splice_module(nn.Identity())
    )
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
        trace._attach_hooks(tl.grad(mul), tl.patch_from(donor))
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
