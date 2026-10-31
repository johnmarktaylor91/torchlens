"""``tl.mean_ablate`` is the one mean door (round 2, decided rows D.11, D.4, D.17, D.19).

It absorbs the former ``mean_fill`` (axis-aware self or population mean) and
``mean_from`` (elementwise population mean). The side-by-side section runs the
old helpers and the new door on the same captured values and requires equal
bytes; it exists only while both spellings are importable. The contract section
pins the door's vocabulary and derived flags.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.intervention import population
from torchlens.intervention.stochastic import mean_fill, mean_from, sampling_records

pytestmark = pytest.mark.smoke


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
    return tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))


_SITE = "relu_1_2"


def _apply(log: tl.Trace, edit: object) -> torch.Tensor:
    fork = log.fork()
    fork.do(tl.label(_SITE), edit)
    return fork[_SITE].out.detach().clone()


# ----------------------------------------------------------------------------
# side by side: old helper and new door on the existing tests' inputs
# ----------------------------------------------------------------------------
def test_door_equals_mean_fill_self_global() -> None:
    log = _toy_trace()
    assert torch.equal(_apply(log, mean_fill(dim="all")), _apply(log, tl.mean_ablate(dim=None)))


def test_door_equals_mean_fill_self_axis() -> None:
    log = _toy_trace()
    assert torch.equal(_apply(log, mean_fill(dim=0)), _apply(log, tl.mean_ablate(dim=0)))
    assert torch.equal(_apply(log, mean_fill(dim=(0, 1))), _apply(log, tl.mean_ablate(dim=(0, 1))))


def test_door_equals_mean_from_population_elementwise() -> None:
    log = _toy_trace()
    baseline = log[_SITE].out
    donors = population(torch.stack([baseline, baseline + 2.0]), origin="evidence")
    old = _apply(log, mean_from(donors))
    new = _apply(log, tl.mean_ablate(donors, dim=None))
    assert torch.equal(old, new)
    assert torch.allclose(new, baseline + 1.0)


def test_door_equals_mean_fill_population_axis() -> None:
    log = _toy_trace()
    baseline = log[_SITE].out
    donors = population(torch.randn(4, *baseline.shape), origin="pop")
    assert torch.equal(
        _apply(log, mean_fill(donors, dim=0)), _apply(log, tl.mean_ablate(donors, dim=0))
    )
    assert torch.equal(
        _apply(log, mean_fill(donors, dim=(0, 1))), _apply(log, tl.mean_ablate(donors, dim=(0, 1)))
    )


def test_door_equals_previous_mean_ablate_tensor_source_formula() -> None:
    """The former ``mean_ablate(source)`` body, verbatim, is the hand oracle."""

    log = _toy_trace()
    out = log[_SITE].out
    source = torch.full((5,), 7.0, dtype=torch.float64)
    hand = torch.zeros_like(out) + source.to(device=out.device, dtype=out.dtype).mean()
    assert torch.equal(_apply(log, tl.mean_ablate(source=source, dim=None)), hand)


def test_door_flags_match_the_old_helpers() -> None:
    log = _toy_trace()
    baseline = log[_SITE].out
    donors = population(torch.randn(4, *baseline.shape), origin="pop")
    pairs = [
        (mean_fill(dim="all"), tl.mean_ablate(dim=None)),
        (mean_fill(dim=0), tl.mean_ablate(dim=0)),
        (mean_from(donors), tl.mean_ablate(donors, dim=None)),
        (mean_fill(donors, dim=0), tl.mean_ablate(donors, dim=0)),
    ]
    for old, new in pairs:
        assert old.batch_independent == new.batch_independent, (old, new)
        assert old.portability == new.portability, (old, new)
        assert dict(old.metadata).get("batch_coherent") == dict(new.metadata).get("batch_coherent")
        assert new.helper_name == "mean_ablate"
    assert dict(tl.mean_ablate(dim=None).kwargs)["dim"] is None
    assert dict(tl.mean_ablate(dim=0).kwargs)["dim"] == dict(mean_fill(dim=0).kwargs)["dim"]


# ----------------------------------------------------------------------------
# the door's own contract
# ----------------------------------------------------------------------------
def test_dim_is_required() -> None:
    with pytest.raises(TypeError):
        tl.mean_ablate()  # type: ignore[call-arg]


@pytest.mark.parametrize("token", ["self", "all", "btach", "", 1.5, True, (), (0, "x")])
def test_vocabulary_refuses_typed(token: object) -> None:
    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.mean_ablate(dim=token)  # type: ignore[arg-type]
    assert excinfo.value.fields["code"] == "intervention_over_invalid"
    assert "dim=None" in str(excinfo.value)


@pytest.mark.parametrize("token", ["batch", "position", "feature"])
def test_named_axis_refuses_until_evidence(token: str) -> None:
    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.mean_ablate(dim=token)  # type: ignore[arg-type]
    assert excinfo.value.fields["code"] == "axis_semantics_unknown"


def test_derived_flags() -> None:
    donors = population(torch.randn(3, 5, 4), origin="pop")
    assert tl.mean_ablate(dim=None).batch_independent is False
    assert tl.mean_ablate(dim=0).batch_independent is False
    assert dict(tl.mean_ablate(dim=None).metadata)["batch_coherent"] is True
    assert tl.mean_ablate(source=torch.ones(4), dim=None).batch_independent is True
    assert tl.mean_ablate(donors, dim=None).batch_independent is True
    assert tl.mean_ablate(donors, dim=None).portability == "opaque_audit"
    assert dict(tl.mean_ablate(dim=(1, 0)).kwargs)["dim"] == (1, 0)
    assert dict(tl.mean_ablate(dim=1).kwargs)["dim"] == (1,)


def test_tensor_source_with_axes_broadcasts_or_refuses() -> None:
    log = _toy_trace()
    out = log[_SITE].out
    source = torch.randn(5, 4)
    filled = _apply(log, tl.mean_ablate(source, dim=0))
    assert torch.equal(filled, (torch.zeros_like(out) + source.mean(dim=[0], keepdim=True)))
    with pytest.raises(Exception) as excinfo:
        log.fork().do(tl.label(_SITE), tl.mean_ablate(torch.randn(7, 3), dim=0))
    assert excinfo.value.fields["code"] == "sampling_geometry_mismatch"


def test_population_mean_records_a_deterministic_note() -> None:
    log = _toy_trace()
    baseline = log[_SITE].out
    donors = population(torch.stack([baseline, baseline + 2.0]), origin="evidence")
    fork = log.fork()
    fork.do(tl.label(_SITE), tl.mean_ablate(donors, dim=None))
    assert torch.allclose(fork[_SITE].out, baseline + 1.0)
    record = sampling_records(fork)[-1]
    assert record["stochastic"] is False
    assert record["base_seed"] is None
    assert record["replacement_mode"] == "evidence_mean"


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
