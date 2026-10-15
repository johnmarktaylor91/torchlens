"""The gradient-transparency matrix: every edit verb under every intervention door.

One cell per (door, verb, keep_grad off/on). Each cell runs the door on a tiny
float64 ``lin1 -> lin2`` model with the edit at ``lin1``'s output, takes
``output.sum().backward()`` and records whether a gradient reaches
``lin1.weight`` (the parameter upstream of the edit). The same edit applied
by hand in a plain torch forward hook gives the eager gradient, and the two
must agree (both absent, or equal). The recorded outcome must equal the row of
``docs/gradient_contract.md``, so the page is the contract and this test keeps
it true. Outcomes: ``reaches``, ``cut`` (no gradient reaches the parameter),
or ``raises:<code>`` / ``error:<type>`` for a door that refuses the verb.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention import (
    InterventionDecision,
    add,
    clamp,
    compose,
    keep_grad,
    mean_ablate,
    mean_fill,
    mean_from,
    noise,
    patch_from,
    permute_batch,
    project_off,
    project_onto,
    reference,
    replace_with,
    resample_rows_from,
    scale,
    scramble_elements,
    set_direction_mean,
    splice_module,
    steer,
    subspace_ablate,
    subspace_interchange,
    swap_with,
    zero_ablate,
)
from torchlens.intervention._position_edits import at_positions

_PAGE = Path(__file__).resolve().parent.parent / "docs" / "gradient_contract.md"
_SITE = "linear_1_1"
_SHAPE = (2, 3, 4)
_X = torch.randn(*_SHAPE, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
_X2 = torch.randn(*_SHAPE, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
_CONST = torch.randn(*_SHAPE, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
_V = torch.tensor([1.0, 2.0, -1.0, 0.5], dtype=torch.float64)
_V_HAT = _V / _V.norm()
# set_direction_mean builds its unit direction in float32 (its own canonical form).
_V_HAT32 = (_V.float() / _V.float().norm()).double()
_Q = 0.5 * torch.tensor([[1, 1, 1, 1], [1, -1, 1, -1]], dtype=torch.float64)
_P = _Q.T @ _Q
_UNITS = torch.zeros(*_SHAPE, dtype=torch.bool)
_UNITS[:, 0, :] = True


class _Net(nn.Module):
    """``lin2(lin1(x))``: the edit sits on ``lin1``'s output."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lin1 = nn.Linear(4, 4).double()
        self.lin2 = nn.Linear(4, 3).double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.lin1(x))


_BASE = _Net().eval()
_SPLICE = nn.Tanh()


def _fresh() -> _Net:
    model = copy.deepcopy(_BASE)
    model.zero_grad(set_to_none=True)
    return model


_SOURCE: dict[str, Any] = {}


def _source_trace() -> Any:
    if "trace" not in _SOURCE:
        _SOURCE["trace"] = tl.trace(
            _fresh(), _X2, capture=tl.options.CaptureOptions(intervention_ready=True)
        )
    return _SOURCE["trace"]


def _site_donors() -> Any:
    return reference(torch.stack([_CONST, _CONST * 2.0, _CONST - 1.0]), origin="site donors")


def _row_donors() -> Any:
    rows = torch.randn(3, 3, 4, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    return reference(rows, origin="row donors")


def _source_value() -> torch.Tensor:
    with torch.no_grad():
        return _BASE.lin1(_X2)


def _scramble_hand(out: torch.Tensor) -> torch.Tensor:
    indices = torch.randint(out.numel(), out.shape, generator=torch.Generator().manual_seed(0))
    return out.reshape(-1)[indices]


def _cut(value: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
    """A hand edit that writes a value computed without the site's graph."""

    return lambda out: value(out).detach()


# name -> (edit factory, hand edit on the site value, meaning under a subspace door)
# Meanings: "coord" acts on the coordinates, "value" takes a value's component,
# "refused" refuses subspace_edit_unsupported.
_VERBS: dict[str, tuple[Callable[[], Any], Callable[[torch.Tensor], torch.Tensor], str]] = {
    "zero_ablate": (zero_ablate, _cut(torch.zeros_like), "coord"),
    "mean_ablate": (mean_ablate, lambda out: torch.zeros_like(out) + out.mean(), "coord"),
    "mean_ablate(source)": (
        lambda: mean_ablate(_CONST),
        _cut(lambda out: torch.zeros_like(out) + _CONST.mean()),
        "refused",
    ),
    "scale": (lambda: scale(3.0), lambda out: 3.0 * out, "coord"),
    "clamp": (lambda: clamp(min=-0.3, max=0.3), lambda out: out.clamp(-0.3, 0.3), "coord"),
    # A linear tail and a sum loss make the gradient independent of the draw.
    "noise": (lambda: noise(0.1, seed=0), lambda out: out + 0.0, "coord"),
    "steer": (lambda: steer(_V, 2.0, feature_axis=-1), lambda out: out + 2.0 * _V, "value"),
    "add(scalar)": (lambda: add(2.0), lambda out: out + 2.0, "coord"),
    "add(tensor)": (lambda: add(_CONST), lambda out: out + _CONST, "value"),
    "replace_with": (lambda: replace_with(_CONST), _cut(lambda out: _CONST.clone()), "value"),
    "swap_with": (lambda: swap_with(_CONST), _cut(lambda out: _CONST.clone()), "value"),
    "project_onto": (
        lambda: project_onto(_V, feature_axis=-1),
        lambda out: (out @ _V_HAT)[..., None] * _V_HAT,
        "refused",
    ),
    "project_off": (
        lambda: project_off(_V, feature_axis=-1),
        lambda out: out - (out @ _V_HAT)[..., None] * _V_HAT,
        "refused",
    ),
    "scramble_elements": (lambda: scramble_elements(seed=0), _scramble_hand, "coord"),
    "scramble_elements(source)": (
        lambda: scramble_elements(_CONST, seed=0),
        _cut(torch.zeros_like),
        "refused",
    ),
    "splice_module": (lambda: splice_module(_SPLICE, input="out"), _SPLICE, "coord"),
    "patch_from": (
        lambda: patch_from(_source_trace()),
        _cut(lambda out: _source_value()),
        "value",
    ),
    # A sum loss makes the gradient independent of the permutation.
    "permute_batch": (lambda: permute_batch(seed=0, axis=0), lambda out: out + 0.0, "value"),
    "resample_rows_from": (
        lambda: resample_rows_from(_row_donors(), seed=0, axis=0),
        _cut(torch.zeros_like),
        "value",
    ),
    "mean_from": (lambda: mean_from(_site_donors()), _cut(torch.zeros_like), "value"),
    "mean_fill": (
        lambda: mean_fill(over="all"),
        lambda out: torch.zeros_like(out) + out.mean(),
        "coord",
    ),
    "mean_fill(ref)": (
        lambda: mean_fill(_site_donors(), over="all"),
        _cut(torch.zeros_like),
        "value",
    ),
    "set_direction_mean": (
        lambda: set_direction_mean(_V, _site_donors(), feature_axis=-1),
        lambda out: out - (out @ _V_HAT32)[..., None] * _V_HAT32,
        "refused",
    ),
    "subspace_ablate": (lambda: subspace_ablate(_Q), lambda out: out - out @ _P, "refused"),
    "subspace_ablate(resample)": (
        lambda: subspace_ablate(_Q, mode="resample", seed=0),
        lambda out: out - out @ _P,
        "refused",
    ),
    "subspace_interchange": (
        lambda: subspace_interchange(_Q, _source_trace()),
        lambda out: out - out @ _P,
        "refused",
    ),
    "compose": (lambda: compose(scale(2.0), scale(3.0)), lambda out: 6.0 * out, "refused"),
    "at_positions": (
        lambda: at_positions(scale(3.0), [0], axis=1),
        lambda out: torch.where(_UNITS, 3.0 * out, out),
        "refused",
    ),
    "hook callable": (lambda: lambda out, *, hook: out * 2.0, lambda out: out * 2.0, "coord"),
}


def _out(trace: Any) -> torch.Tensor:
    return trace[trace.output_layers[0]].out


def _ready(model: nn.Module) -> Any:
    return tl.trace(model, _X, capture=tl.options.CaptureOptions(intervention_ready=True))


def _door_capture(model: nn.Module, edit: Any) -> torch.Tensor:
    return _out(tl.trace(model, _X, intervene=tl.when(tl.module("lin1"), edit)))


def _door_hooks(model: nn.Module, edit: Any) -> torch.Tensor:
    options = tl.options.CaptureOptions(hooks={tl.module("lin1"): edit})
    return _out(tl.trace(model, _X, capture=options))


def _door_replay(model: nn.Module, edit: Any) -> torch.Tensor:
    fork = _ready(model).fork()
    fork.do(_SITE, edit)
    return _out(fork)


def _door_rerun(model: nn.Module, edit: Any) -> torch.Tensor:
    trace = tl.trace(model, _X)
    trace.do(tl.module("lin1"), edit, model=model, x=_X)
    return _out(trace)


def _door_attach_run(model: nn.Module, edit: Any) -> torch.Tensor:
    trace = tl.trace(model, _X)
    trace.attach_hooks(tl.module("lin1"), edit, confirm_mutation=True)
    trace.run(model, _X)
    return _out(trace)


def _door_bind(model: nn.Module, edit: Any) -> torch.Tensor:
    return tl.when(tl.module("lin1"), edit).bind(model)(_X)


def _door_push_differentiable(model: nn.Module, edit: Any) -> torch.Tensor:
    replay = tl.options.ReplayOptions(hooks={tl.label(_SITE): edit}, differentiable=True)
    return _out(_ready(model).push(replay=replay))


def _door_units(model: nn.Module, edit: Any) -> torch.Tensor:
    fork = _ready(model).fork()
    fork.do(tl.units(_SITE, _UNITS), edit)
    return _out(fork)


def _door_subspace(model: nn.Module, edit: Any) -> torch.Tensor:
    fork = _ready(model).fork()
    fork.do(tl.subspace(_SITE, _Q, origin="hadamard rows"), edit)
    return _out(fork)


def _door_decision(model: nn.Module, edit: Any) -> torch.Tensor:
    """Capture-time door with the edit inside an ``InterventionDecision``."""

    keep = bool(getattr(edit, "_matrix_keep", False))
    inner = getattr(edit, "_matrix_inner", edit)
    decision = InterventionDecision(action="add_hook", hook=inner, keep_grad=keep)
    return _out(tl.trace(model, _X, intervene=tl.when(tl.module("lin1"), decision)))


def _plain(f: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
    return f


def _units_hand(
    f: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    return lambda out: torch.where(_UNITS, f(out), out)


def _frontier_hand(
    f: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Differentiable replay re-leafs the site value: nothing reaches upstream."""

    return lambda out: f(out.detach().requires_grad_())


# door -> (runner, hand transform of the verb's hand edit)
_DOORS: dict[str, tuple[Callable[[nn.Module, Any], torch.Tensor], Callable[..., Any]]] = {
    "capture intervene=": (_door_capture, _plain),
    "capture hooks=": (_door_hooks, _plain),
    "capture InterventionDecision": (_door_decision, _plain),
    "fork.do (replay)": (_door_replay, _plain),
    "do(model=, x=) (rerun)": (_door_rerun, _plain),
    "attach_hooks + run": (_door_attach_run, _plain),
    "spec.bind(model)": (_door_bind, _plain),
    "push(differentiable=True)": (_door_push_differentiable, _frontier_hand),
    "do(tl.units(...))": (_door_units, _units_hand),
    "do(tl.subspace(...))": (_door_subspace, None),
}


def _subspace_hand(verb: str) -> Callable[[torch.Tensor], torch.Tensor] | None:
    _factory, f, meaning = _VERBS[verb]
    if meaning == "coord":
        return lambda out: out + ((f(out @ _Q.T) - out @ _Q.T) @ _Q)
    if meaning == "value":
        return lambda out: out + (f(out) - out) @ _P
    return None


def _hand_gradient(door: str, verb: str) -> torch.Tensor | None:
    """Return ``lin1.weight.grad`` with the edit applied by hand in a forward hook."""

    _factory, f, _meaning = _VERBS[verb]
    transform = _DOORS[door][1]
    hand = _subspace_hand(verb) if transform is None else transform(f)
    assert hand is not None
    model = _fresh()
    handle = model.lin1.register_forward_hook(lambda _m, _a, out: hand(out))
    try:
        output = model(_X)
    finally:
        handle.remove()
    if output.requires_grad:
        output.sum().backward()
    return model.lin1.weight.grad


def _make_edit(door: str, verb: str, keep: bool) -> Any:
    edit = _VERBS[verb][0]()
    if door == "capture InterventionDecision":
        holder = _DecisionEdit(edit, keep)
        return holder
    return keep_grad(edit) if keep else edit


class _DecisionEdit:
    """Carry (edit, keep_grad) into the decision door without wrapping it."""

    def __init__(self, edit: Any, keep: bool) -> None:
        self._matrix_inner = edit
        self._matrix_keep = keep


def _measure(door: str, verb: str, keep: bool) -> tuple[str, torch.Tensor | None]:
    model = _fresh()
    try:
        output = _DOORS[door][0](model, _make_edit(door, verb, keep))
    except Exception as error:  # noqa: BLE001 -- the outcome IS the refusal
        code = getattr(error, "fields", {}).get("code") if hasattr(error, "fields") else None
        return (f"raises:{code}" if code else f"error:{type(error).__name__}"), None
    if not output.requires_grad:
        return "cut", None
    output.sum().backward()
    grad = model.lin1.weight.grad
    return ("cut" if grad is None else "reaches"), grad


def _page_rows() -> dict[tuple[str, str], tuple[str, str]]:
    """Parse the matrix table of the gradient-contract page."""

    rows: dict[tuple[str, str], tuple[str, str]] = {}
    pattern = re.compile(r"^\| (.+?) \| `(.+?)` \| (.+?) \| (.+?) \|$")
    for line in _PAGE.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            door, verb, off, on = (part.strip().strip("`") for part in match.groups())
            rows[(door, verb)] = (off, on)
    return rows


_CELLS = [(door, verb) for door in _DOORS for verb in _VERBS]


def test_page_lists_exactly_the_matrix_cells() -> None:
    assert sorted(_page_rows()) == sorted(_CELLS)


@pytest.mark.parametrize("keep", [False, True], ids=["keep_grad_off", "keep_grad_on"])
@pytest.mark.parametrize(("door", "verb"), _CELLS)
def test_gradient_cell_matches_page_and_eager_hand(door: str, verb: str, keep: bool) -> None:
    outcome, grad = _measure(door, verb, keep)
    expected = _page_rows()[(door, verb)][1 if keep else 0]
    assert outcome == expected, (door, verb, keep, outcome)
    if outcome in {"reaches", "cut"}:
        want = _hand_gradient(door, verb)
        assert (grad is None) == (want is None), (door, verb, keep)
        if grad is not None and want is not None:
            torch.testing.assert_close(grad, want, rtol=1e-10, atol=1e-12)
    if keep and outcome == "raises:edit_detached_from_graph":
        hand = _VERBS[verb][1]
        assert not hand(_X.clone().requires_grad_()).requires_grad, (door, verb)
