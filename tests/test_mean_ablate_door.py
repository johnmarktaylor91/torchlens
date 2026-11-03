"""``tl.mean_ablate`` is the one mean door (round 2, decided rows D.11, D.4, D.17, D.19).

It absorbs the former ``mean_fill`` (axis-aware self or population mean) and
``mean_from`` (elementwise population mean). The equality section pins the door
to those helpers' bodies, written out as hand oracles (commit 3d9ed5c06 ran the
old helpers and the door side by side on the same captured values and required
equal bytes before the old names left). The contract section pins the door's
vocabulary and derived flags.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.intervention import population


class _Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))


def _toy_trace() -> tl.Trace:
    torch.manual_seed(0)
    model = _Toy()
    x = torch.randn(5, 4)
    return tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))


_SITE = "relu_1_2"


def _apply(log: tl.Trace, edit: object) -> torch.Tensor:
    fork = log.fork()
    fork.do(tl.label(_SITE), edit)
    return fork[_SITE].out.detach().clone()


def test_door_equals_mean_fill_population_axis() -> None:
    log = _toy_trace()
    baseline = log[_SITE].out
    donors = population(torch.randn(4, *baseline.shape), origin="pop")
    assert torch.equal(
        _apply(log, tl.mean_ablate(donors, dim=0)), donors.mean(0).expand(baseline.shape).clone()
    )
    assert torch.equal(
        _apply(log, tl.mean_ablate(donors, dim=(0, 1))),
        donors.mean((0, 1)).expand(baseline.shape).clone(),
    )


@pytest.mark.parametrize("token", ["batch", "position", "feature"])
def test_named_axis_refuses_until_evidence(token: str) -> None:
    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.mean_ablate(dim=token)  # type: ignore[arg-type]
    assert excinfo.value.fields["code"] == "axis_semantics_unknown"


def test_tensor_source_with_axes_broadcasts_or_refuses() -> None:
    log = _toy_trace()
    out = log[_SITE].out
    source = torch.randn(5, 4)
    filled = _apply(log, tl.mean_ablate(source, dim=0))
    assert torch.equal(filled, (torch.zeros_like(out) + source.mean(dim=[0], keepdim=True)))
    with pytest.raises(Exception) as excinfo:
        log.fork().do(tl.label(_SITE), tl.mean_ablate(torch.randn(7, 3), dim=0))
    assert excinfo.value.fields["code"] == "sampling_geometry_mismatch"


def test_saved_mean_fill_spec_refuses_as_renamed() -> None:
    from torchlens.intervention.helpers import rebuild_builtin_helper

    with pytest.raises(InvalidArgumentError) as excinfo:
        rebuild_builtin_helper("mean_fill", (), {"dim": "all", "allow_shape_change": False})
    assert excinfo.value.fields["code"] == "intervention_helper_renamed"
    assert "torchlens.mean_ablate" in str(excinfo.value)


def test_saved_self_token_spec_refuses_typed() -> None:
    from torchlens.intervention.helpers import rebuild_builtin_helper

    with pytest.raises(InvalidArgumentError) as excinfo:
        rebuild_builtin_helper("mean_ablate", (None,), {"dim": "self", "allow_shape_change": False})
    assert excinfo.value.fields["code"] == "intervention_over_invalid"
