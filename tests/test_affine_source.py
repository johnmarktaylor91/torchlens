"""``affine_source``: under a subspace, the new coordinates are ``weight @ x + bias``.

``fork.do(tl.subspace(site, R.T, origin=...), tl.intervention.set_affine_coordinates(W, bias=b))``
is LoReFT, ``h + R (W h + b - R^T h)``. These tests check it against that hand formula in
float64 (rank 1 and 4, every position and one position), check that a saved spec applies
bit-identically, and check the refusals (outside a subspace, malformed arguments), each
with its remedy run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.intervention.helpers import rebuild_builtin_helper

_SITE = "linear_1_1"
_ORIGIN = "LoReFT trained elsewhere"
_D = 6
_POSITION = 3


class _SeqNet(nn.Module):
    """``lin2(lin1(x))`` over ``[batch, seq, 6]`` in float64."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lin1 = nn.Linear(_D, _D).double()
        self.lin2 = nn.Linear(_D, 3).double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.lin1(x))


_MODEL = _SeqNet().eval()
_X = torch.randn(2, 5, _D, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
_R = torch.linalg.qr(
    torch.randn(_D, _D, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
)[0]
_W = torch.randn(4, _D, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
_B = torch.randn(4, generator=torch.Generator().manual_seed(5), dtype=torch.float64)


def _ready() -> Any:
    return tl.trace(_MODEL, _X, capture=tl.options.CaptureOptions(replayable=True))


def _one_position_mask() -> torch.Tensor:
    mask = torch.zeros(2, 5, _D, dtype=torch.bool)
    mask[:, _POSITION, :] = True
    return mask


def _target(rank: int, one_position: bool) -> Any:
    within: Any = tl.units(_SITE, _one_position_mask()) if one_position else _SITE
    return tl.subspace(within, _R[:, :rank].T, origin=_ORIGIN, method="loreft")


def _edit(rank: int) -> Any:
    return tl.intervention.set_affine_coordinates(_W[:rank], bias=_B[:rank])


def _hand(rank: int, one_position: bool) -> torch.Tensor:
    """``h + R (W h + b - R^T h)`` at the edited positions, ``h`` elsewhere."""

    rotation, weight, bias = _R[:, :rank], _W[:rank], _B[:rank]
    with torch.no_grad():
        h = _MODEL.lin1(_X)
        edited = h + (h @ weight.T + bias - h @ rotation) @ rotation.T
    if not one_position:
        return edited
    return torch.where(_one_position_mask(), edited, h)


@pytest.mark.parametrize("one_position", [False, True], ids=["every-position", "one-position"])
@pytest.mark.parametrize("rank", [1, 4])
def test_affine_source_equals_the_hand_loreft_formula(rank: int, one_position: bool) -> None:
    fork = _ready().fork()
    fork.do(_target(rank, one_position), _edit(rank))
    result = fork.push()[_SITE].out
    assert torch.allclose(result, _hand(rank, one_position), rtol=0.0, atol=1e-12)
    if one_position:
        untouched = ~_one_position_mask()
        assert torch.equal(result[untouched], _ready()[_SITE].out[untouched])


def _apply(spec: Any) -> Any:
    fork = _ready().fork()
    for hook_spec in spec.hook_specs:
        fork.attach_hooks(hook_spec.site_target, hook_spec.helper, confirm_mutation=True)
    return fork.push()


@pytest.mark.parametrize("rank", [1, 4])
def test_saved_affine_source_applies_bit_identically(tmp_path: Path, rank: int) -> None:
    fork = _ready().fork()
    fork.do(_target(rank, False), _edit(rank))
    session = fork.push()
    path = tmp_path / "loreft.tlspec"
    tl.io.save_intervention(fork, path, level="portable")
    loaded = tl.io.load_intervention_spec(path)
    helper = loaded.hook_specs[0].helper
    assert helper.name == "subspace_edit" and helper.portability == "builtin"
    inner = helper.args[0]
    assert inner.name == "set_affine_coordinates"
    assert torch.equal(inner.args[0], _W[:rank]) and torch.equal(
        dict(inner.kwargs)["bias"], _B[:rank]
    )
    assert torch.equal(_apply(loaded)[_SITE].out, session[_SITE].out)


def test_bias_absent_default_fills_and_applies_without_a_bias() -> None:
    """An older payload with no ``bias`` keyword rebuilds with ``bias=None``."""

    rebuilt = rebuild_builtin_helper("set_affine_coordinates", (_W[:2],), {})
    assert dict(rebuilt.kwargs)["bias"] is None
    fork = _ready().fork()
    fork.do(tl.subspace(_SITE, _R[:, :2].T, origin=_ORIGIN), rebuilt)
    rotation, weight = _R[:, :2], _W[:2]
    with torch.no_grad():
        h = _MODEL.lin1(_X)
        hand = h + (h @ weight.T - h @ rotation) @ rotation.T
    assert torch.allclose(fork.push()[_SITE].out, hand, rtol=0.0, atol=1e-12)


def _coded(error: BaseException) -> Any:
    current: BaseException | None = error
    while current is not None:
        if getattr(current, "fields", {}).get("code"):
            return current
        current = current.__cause__ or current.__context__
    raise AssertionError(f"no coded error in the chain of {error!r}")


def _remedy_runs(error: Any) -> None:
    """Run the refusal's spelling: the verb under a subspace on the same site."""

    remedy = error.fields["remedy"]
    assert "tl.subspace(site, rotation.T, origin=...)" in remedy
    assert "tl.intervention.set_affine_coordinates(weight, bias=bias)" in remedy
    rotation, weight, bias = _R[:, :2], _W[:2], _B[:2]
    fork = _ready().fork()
    fork.do(
        tl.subspace(_SITE, rotation.T, origin=_ORIGIN),
        tl.intervention.set_affine_coordinates(weight, bias=bias),
    )
    assert not torch.equal(fork.push()[_SITE].out, _ready()[_SITE].out)


def test_outside_a_subspace_refuses_at_do_and_the_remedy_runs() -> None:
    fork = _ready().fork()
    with pytest.raises(InvalidArgumentError) as excinfo:
        fork.do(tl.units(_SITE, torch.ones(2, 5, _D, dtype=torch.bool)), _edit(2))
    assert excinfo.value.fields["code"] == "affine_source_needs_subspace"
    _remedy_runs(excinfo.value)


def test_outside_a_subspace_refuses_on_a_site_label() -> None:
    """The string-label door has no selection to check, so the hook refuses when it fires."""

    fork = _ready().fork()
    with pytest.raises(Exception) as excinfo:
        fork.do(_SITE, _edit(2))
        fork.push()
    error = _coded(excinfo.value)
    assert error.fields["code"] == "affine_source_needs_subspace"
    assert _SITE in str(error)
    _remedy_runs(error)


_MALFORMED = {
    "weight-not-tensor": ("weight", lambda: ([[1.0] * _D],), {}),
    "weight-1d": ("weight", lambda: (_W[0],), {}),
    "weight-int": ("weight", lambda: (torch.ones(2, _D, dtype=torch.int64),), {}),
    "weight-nan": ("weight", lambda: (torch.full((2, _D), float("nan"), dtype=torch.float64),), {}),
    "bias-rows": ("bias", lambda: (_W[:2],), {"bias": _B[:3]}),
}


@pytest.mark.parametrize("case", sorted(_MALFORMED))
def test_malformed_argument_refuses_at_construction_and_at_load(case: str) -> None:
    argument, args, kwargs = _MALFORMED[case]
    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.intervention.set_affine_coordinates(*args(), **kwargs)
    assert excinfo.value.fields["code"] == "affine_source_invalid"
    assert excinfo.value.fields["argument"] == argument
    with pytest.raises(InvalidArgumentError) as loaded:
        rebuild_builtin_helper("set_affine_coordinates", args(), kwargs)
    assert loaded.value.fields["code"] == "affine_source_invalid"
    _remedy_runs(excinfo.value)


def test_weight_shaped_unlike_the_basis_refuses() -> None:
    fork = _ready().fork()
    with pytest.raises(InvalidArgumentError) as excinfo:
        fork.do(
            tl.subspace(_SITE, _R[:, :2].T, origin=_ORIGIN),
            tl.intervention.set_affine_coordinates(_W[:3]),
        )
    assert excinfo.value.fields["code"] == "affine_source_invalid"
    assert excinfo.value.fields["argument"] == "weight"
    _remedy_runs(excinfo.value)
