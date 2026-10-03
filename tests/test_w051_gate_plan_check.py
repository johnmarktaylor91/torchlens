"""AUD-CODE 4.2 regression: check_plan multiplicity and index bounds.

The multiplicity check was ``len(labels) > 0``: a selector fanning ONE
declared replacement out over many sites read ``multiplicity_ok``, and a
pass-qualified ``label:k`` outside ``1..num_passes`` surfaced only as the
resolver's opaque "matched 0 sites" refusal despite the docstring's
index-bounds promise.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import MultiMatchWarning
from torchlens.options import CaptureOptions


class _TwoPass(nn.Module):
    """One Linear+ReLU pair called twice: relu_1_2 and linear_1_1 run 2 passes."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.head = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(2):
            x = torch.relu(self.fc(x))
        return self.head(x)


@pytest.fixture
def meta_trace() -> Any:
    with torch.device("meta"):
        model = _TwoPass()
    model.eval()
    trace = tl.trace(
        model, torch.empty(2, 4, device="meta"), capture=CaptureOptions(structure_only=True)
    )
    yield trace
    trace.cleanup()


def test_one_replacement_over_many_sites_is_a_multiplicity_finding(meta_trace: Any) -> None:
    with pytest.warns(MultiMatchWarning):  # the resolver's fan-out notice is unchanged
        report = meta_trace.check_plan([(tl.func("relu"), torch.empty(2, 4, device="meta"))])
    site = report.sites[0]
    assert len(site.resolved_labels) == 2
    assert site.multiplicity_ok is False
    assert site.replacement_verdict.startswith("multiplicity_mismatch: 2 sites")
    assert "label:pass" in site.replacement_verdict
    assert not report.ok
    assert report.executable is False


def test_bare_selector_over_many_sites_stays_ok(meta_trace: Any) -> None:
    """Existence check only: a bare selector may fan out."""

    with pytest.warns(MultiMatchWarning):
        report = meta_trace.check_plan([tl.func("relu")])
    assert report.sites[0].multiplicity_ok is True
    assert report.sites[0].replacement_verdict == "not_declared"
    assert report.ok


def test_single_site_replacement_still_passes_geometry(meta_trace: Any) -> None:
    report = meta_trace.check_plan([("relu_1_2:1", torch.empty(2, 4, device="meta"))])
    assert report.sites[0].multiplicity_ok is True
    assert report.sites[0].replacement_verdict == "geometry_ok"
    assert report.ok


@pytest.mark.parametrize("bad_pass", [0, 3, 99])
def test_pass_index_out_of_bounds_is_a_finding_not_an_opaque_refusal(
    meta_trace: Any, bad_pass: int
) -> None:
    report = meta_trace.check_plan([f"relu_1_2:{bad_pass}"])
    site = report.sites[0]
    assert site.resolved_labels == ()
    assert site.multiplicity_ok is False
    assert site.replacement_verdict.startswith("index_out_of_bounds")
    assert "ran 2 passes" in site.replacement_verdict
    assert "1..2" in site.replacement_verdict
    assert not report.ok
    assert "index_out_of_bounds" in report.report()


def test_unknown_base_label_is_still_the_resolvers_refusal(meta_trace: Any) -> None:
    """Index bounds cover EXISTING layers only; typos stay the resolver's call."""

    with pytest.raises(Exception):  # noqa: B017 - resolver refusal type is its own contract
        meta_trace.check_plan(["relu_9_9:1"])
