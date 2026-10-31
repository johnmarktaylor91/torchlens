"""Projection-valued subspace edits: ``do(subspace(...), edit)`` edits the component.

The fixture keeps every number exact in float64: the edited site is a linear
layer whose input passes through unchanged, the site values are built as
``coords @ H`` for dyadic coordinates, and the bases are rows of the 4x4
Hadamard matrix scaled by 1/2 (dense and exactly orthonormal). So the
orthogonal complement can be checked bit for bit and the component against
the hand computation exactly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.intervention import (
    compose,
    grad_zero,
    mean_ablate,
    mean_from,
    project_off,
    project_onto,
    reference,
    scramble_elements,
    set_direction_mean,
    splice_module,
    subspace_ablate,
    subspace_interchange,
)
from torchlens.intervention._position_edits import at_positions
from torchlens.intervention._subspace_projection import PROJECTION_NOTE

_H = 0.5 * torch.tensor(
    [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], dtype=torch.float64
)
_BASES = {"rank1": _H[:1], "rank3": _H[:3]}
_COORDS = torch.tensor(
    [[2.0, -4.0, 6.0, 1.0], [6.0, 2.0, -2.0, 3.0], [-2.0, 4.0, 2.0, -1.0], [2.0, 6.0, 8.0, 5.0]],
    dtype=torch.float64,
)
_SITE = "linear_1_1"


class _Mlp(nn.Module):
    """``x -> lin1 -> relu -> lin2``; ``lin1`` is the identity, so its output is ``x``."""

    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(4, 4).double()
        self.lin2 = nn.Linear(4, 3).double()
        with torch.no_grad():
            self.lin1.weight.copy_(torch.eye(4, dtype=torch.float64))
            self.lin1.bias.zero_()
            self.lin2.weight.copy_(torch.arange(12, dtype=torch.float64).reshape(3, 4) - 5)
            self.lin2.bias.copy_(torch.tensor([1.0, -1.0, 0.5], dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(torch.relu(self.lin1(x)))


def _trace(coords: torch.Tensor = _COORDS) -> tl.Trace:
    return tl.trace(
        _Mlp().eval(),
        coords @ _H,
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )


def _split(value: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (component coordinates, complement coordinates) in the Hadamard frame."""

    frame = value.detach() @ _H.T
    return frame[:, :k], frame[:, k:]


def _edited(basis: torch.Tensor, edit: Any) -> tuple[torch.Tensor, tl.Trace]:
    log = _trace()
    fork = log.fork()
    fork.do(tl.subspace(_SITE, basis, origin="hadamard rows", method="manual"), edit)
    return fork[_SITE].out.detach(), fork


_HAND: dict[str, tuple[Callable[[], Any], Callable[[torch.Tensor], torch.Tensor]]] = {
    "zero": (tl.zero_ablate, torch.zeros_like),
    "mean": (mean_ablate, lambda c: torch.zeros_like(c) + c.mean()),
    "scale": (lambda: tl.scale(3.0), lambda c: 3.0 * c),
}


@pytest.mark.parametrize("rank", sorted(_BASES))
@pytest.mark.parametrize("verb", sorted(_HAND))
def test_component_matches_hand_and_complement_is_bit_unchanged(rank: str, verb: str) -> None:
    basis = _BASES[rank]
    k = int(basis.shape[0])
    make_edit, hand = _HAND[verb]
    out, _fork = _edited(basis, make_edit())
    component, complement = _split(out, k)
    base_component, base_complement = _split(_COORDS @ _H, k)
    assert torch.equal(complement, base_complement)
    assert torch.equal(component, hand(base_component))
    assert not torch.equal(component, base_component)


def test_audit_record_states_the_projection_and_keeps_the_basis_provenance() -> None:
    _out, fork = _edited(_BASES["rank1"], tl.zero_ablate())
    (site,) = fork.intervention_audit[-1]["sites"]
    assert PROJECTION_NOTE in site["source"]
    assert "origin='hadamard rows'" in site["source"]
    assert "basis_sha256=" in site["source"]


def test_non_orthonormal_basis_is_orthonormalised_keeping_sign() -> None:
    """A scaled direction edits the component along its unit vector, sign kept."""

    log = _trace()
    fork = log.fork()
    fork.do(tl.subspace(_SITE, 4.0 * _H[0], origin="scaled row"), tl.clamp(max=0.0))
    component, complement = _split(fork[_SITE].out, 1)
    base_component, base_complement = _split(_COORDS @ _H, 1)
    torch.testing.assert_close(complement, base_complement, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(component, base_component.clamp(max=0.0), rtol=0.0, atol=1e-12)
    assert "basis orthonormalised" in fork.intervention_audit[-1]["sites"][0]["source"]


def test_callable_edit_receives_the_coordinates() -> None:
    seen: list[tuple[int, ...]] = []

    def _hook(coords: torch.Tensor, *, hook: Any) -> torch.Tensor:
        del hook
        seen.append(tuple(coords.shape))
        return coords + 1.0

    out, _fork = _edited(_BASES["rank3"], _hook)
    assert seen == [(4, 3)]
    component, _complement = _split(out, 3)
    assert torch.equal(component, _COORDS[:, :3] + 1.0)


def _das_interchange(model: _Mlp, base: torch.Tensor, source: torch.Tensor, basis: torch.Tensor):
    """Hand DAS interchange: ``x_b - Q^T Q x_b + Q^T Q x_s`` at lin1's output, eagerly."""

    with torch.no_grad():
        source_hidden = model.lin1(source)

        def _swap(_module: Any, _args: Any, out: torch.Tensor) -> torch.Tensor:
            return out - (out @ basis.T) @ basis + (source_hidden @ basis.T) @ basis

        handle = model.lin1.register_forward_hook(_swap)
        try:
            return model(base)
        finally:
            handle.remove()


@pytest.mark.parametrize("rank", sorted(_BASES))
def test_patch_from_is_the_das_interchange(rank: str) -> None:
    basis = _BASES[rank]
    torch.manual_seed(0)
    model = _Mlp().eval()
    with torch.no_grad():
        model.lin1.weight.copy_(torch.randn(4, 4, dtype=torch.float64))
        model.lin2.weight.copy_(torch.randn(3, 4, dtype=torch.float64))
    base = torch.randn(5, 4, dtype=torch.float64)
    source = torch.randn(5, 4, dtype=torch.float64)
    options = tl.options.CaptureOptions(intervention_ready=True)
    target_log = tl.trace(model, base, capture=options)
    source_log = tl.trace(model, source, capture=options)
    fork = target_log.fork()
    fork.do(tl.subspace(_SITE, basis, origin="das rows"), tl.patch_from(source_log))
    got = fork[fork.output_layers[0]].out.detach()
    want = _das_interchange(model, base, source, basis)
    torch.testing.assert_close(got, want, rtol=0.0, atol=1e-12)
    component, complement = _split(fork[_SITE].out, int(basis.shape[0]))
    source_component, _ = _split(source_log[_SITE].out, int(basis.shape[0]))
    _, base_complement = _split(target_log[_SITE].out, int(basis.shape[0]))
    torch.testing.assert_close(component, source_component, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(complement, base_complement, rtol=0.0, atol=1e-12)


def _site_units_mask(rows: list[int]) -> torch.Tensor:
    mask = torch.zeros(4, 4, dtype=torch.bool)
    mask[rows] = True
    return mask


def _donors() -> Any:
    return reference(torch.arange(32, dtype=torch.float64).reshape(2, 4, 4), origin="donors")


def _subspace_sel(basis: torch.Tensor = _BASES["rank1"]) -> Any:
    return tl.subspace(_SITE, basis, origin="hadamard rows")


_REFUSED: dict[str, tuple[Callable[[], tuple[Any, Any]], Callable[[], tuple[Any, Any]]]] = {
    "project_off": (
        lambda: (_subspace_sel(), project_off(_H[0], feature_axis=-1)),
        lambda: (_subspace_sel(), tl.zero_ablate()),
    ),
    "project_onto": (
        lambda: (_subspace_sel(), project_onto(_H[0], feature_axis=-1)),
        lambda: (_SITE, project_onto(_H[0], feature_axis=-1)),
    ),
    "set_direction_mean": (
        lambda: (_subspace_sel(), set_direction_mean(_H[0], _donors(), feature_axis=-1)),
        lambda: (_subspace_sel(), mean_from(_donors())),
    ),
    "subspace_ablate": (
        lambda: (_subspace_sel(), subspace_ablate(_H[:1])),
        lambda: (_SITE, subspace_ablate(_H[:1])),
    ),
    "subspace_interchange": (
        lambda: (_subspace_sel(), subspace_interchange(_H[:1], _trace())),
        lambda: (_subspace_sel(), tl.patch_from(_trace(_COORDS * 2))),
    ),
    "compose": (
        lambda: (_subspace_sel(), compose(tl.scale(2.0), tl.scale(3.0))),
        lambda: (_subspace_sel(), tl.scale(6.0)),
    ),
    "at_positions": (
        lambda: (_subspace_sel(), at_positions(tl.zero_ablate(), [0])),
        lambda: (
            tl.subspace(tl.units(_SITE, _site_units_mask([0])), _H[:1], origin="rows"),
            tl.zero_ablate(),
        ),
    ),
    "mean_ablate_source": (
        lambda: (_subspace_sel(), mean_ablate(torch.ones(4, 4, dtype=torch.float64))),
        lambda: (_subspace_sel(), mean_ablate()),
    ),
    "scramble_elements_source": (
        lambda: (_subspace_sel(), scramble_elements(torch.ones(4, 4), seed=0)),
        lambda: (_subspace_sel(), scramble_elements(seed=0)),
    ),
    "splice_module_in": (
        lambda: (_subspace_sel(), splice_module(nn.Identity(), input_route="in")),
        lambda: (_subspace_sel(), splice_module(nn.Identity(), input_route="out")),
    ),
    "grad_zero": (
        lambda: (_subspace_sel(), grad_zero()),
        lambda: ("capture", grad_zero()),
    ),
}


def _run_remedy(target: Any, edit: Any) -> None:
    """Run one remedy spelling; backward verbs attach at capture time."""

    if target != "capture":
        _trace().fork().do(target, edit)
        return
    trace = tl.trace(_Mlp(), _COORDS @ _H, intervene=tl.when(tl.module("lin1"), edit))
    trace[trace.output_layers[0]].out.sum().backward()


@pytest.mark.parametrize("name", sorted(_REFUSED))
def test_refused_verb_carries_its_code_and_the_remedy_runs(name: str) -> None:
    refused, remedy = _REFUSED[name]
    fork = _trace().fork()
    with pytest.raises(InvalidArgumentError) as caught:
        fork.do(*refused())
    assert caught.value.fields["code"] == "subspace_edit_unsupported"
    assert caught.value.fields["remedy"]
    _run_remedy(*remedy())


def test_composed_target_refuses_and_within_runs() -> None:
    fork = _trace().fork()
    composed = _subspace_sel() & tl.units(_SITE, _site_units_mask([0, 1]))
    with pytest.raises(InvalidArgumentError) as caught:
        fork.do(composed, tl.zero_ablate())
    assert caught.value.fields["code"] == "subspace_projection_composed"
    restricted = tl.subspace(tl.units(_SITE, _site_units_mask([0, 1])), _H[:1], origin="rows")
    fork.do(restricted, tl.zero_ablate())
    component, complement = _split(fork[_SITE].out, 1)
    assert torch.equal(component[:2], torch.zeros(2, 1, dtype=torch.float64))
    assert torch.equal(component[2:], _COORDS[2:, :1])
    assert torch.equal(complement, _COORDS[:, 1:])


def test_resolved_target_refuses_and_units_mask_edits_the_support_set() -> None:
    log = _trace()
    fork = log.fork()
    resolved = _subspace_sel().resolve(fork)
    with pytest.raises(InvalidArgumentError) as caught:
        fork.do(resolved, tl.zero_ablate())
    assert caught.value.fields["code"] == "subspace_projection_composed"
    fork.do(tl.units(_SITE, resolved[0].mask), tl.zero_ablate())
    assert bool((fork[_SITE].out == 0).all())


def test_partial_axis_population_refuses_and_full_rows_run() -> None:
    fork = _trace().fork()
    partial = torch.zeros(4, 4, dtype=torch.bool)
    partial[0, :2] = True
    with pytest.raises(InvalidArgumentError) as caught:
        fork.do(tl.subspace(tl.units(_SITE, partial), _H[:1], origin="rows"), tl.zero_ablate())
    assert caught.value.fields["code"] == "subspace_projection_partial_axis"
    fork.do(
        tl.subspace(tl.units(_SITE, _site_units_mask([0])), _H[:1], origin="rows"), tl.zero_ablate()
    )


def test_shape_changing_callable_refuses() -> None:
    fork = _trace().fork()
    with pytest.raises(InvalidArgumentError) as caught:
        fork.do(_subspace_sel(), lambda c, *, hook: c.sum(dim=-1))
    assert caught.value.fields["code"] == "subspace_edit_unsupported"


def _random_mlp() -> _Mlp:
    torch.manual_seed(0)
    model = _Mlp().eval()
    with torch.no_grad():
        model.lin1.weight.copy_(torch.randn(4, 4, dtype=torch.float64))
        model.lin2.weight.copy_(torch.randn(3, 4, dtype=torch.float64))
    return model


@pytest.mark.parametrize("rank", sorted(_BASES))
@pytest.mark.parametrize("verb", ["scale", "patch_from"])
def test_gradient_reaches_upstream_through_the_component_edit(rank: str, verb: str) -> None:
    """``lin1.weight.grad`` through the subspace edit equals the eager hand version."""

    basis = _BASES[rank]
    x = torch.randn(5, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(7))
    source_x = torch.randn(5, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(8))
    options = tl.options.CaptureOptions(intervention_ready=True)
    model = _random_mlp()
    source = tl.trace(_random_mlp(), source_x, capture=options)
    edit = tl.scale(2.0) if verb == "scale" else tl.patch_from(source)
    fork = tl.trace(model, x, capture=options).fork()
    fork.do(tl.subspace(_SITE, basis, origin="rows"), edit)
    fork[fork.output_layers[0]].out.sum().backward()
    got = model.lin1.weight.grad

    eager = _random_mlp()
    with torch.no_grad():
        donor = eager.lin1(source_x)

    def _hand(_module: Any, _args: Any, out: torch.Tensor) -> torch.Tensor:
        coords = out @ basis.T
        new = 2.0 * coords if verb == "scale" else donor @ basis.T
        return out + (new - coords) @ basis

    handle = eager.lin1.register_forward_hook(_hand)
    try:
        eager(x).sum().backward()
    finally:
        handle.remove()
    assert got is not None
    torch.testing.assert_close(got, eager.lin1.weight.grad, rtol=1e-12, atol=1e-12)
