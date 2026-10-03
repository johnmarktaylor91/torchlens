"""Plan-check verb (weightsfree memo D14 / face 4 / composition rows).

Audit-only, forever separate from the runnable gate: pass, shape mismatch,
selector typo, value-callable refusal, and REFUTED-source refusal — plus the
never-arms pin.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions


class Toy(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(4, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


def _meta_trace():
    with torch.device("meta"):
        model = Toy()
    model.eval()
    return tl.trace(
        model, torch.empty(2, 4, device="meta"), capture=CaptureOptions(structure_only=True)
    )


@pytest.mark.smoke
def test_plan_check_pass() -> None:
    trace = _meta_trace()
    plan = [
        ("relu_1_2", torch.empty(2, 4, device="meta")),
        (tl.func("linear"), None),
    ]
    report = trace.check_plan(plan)
    assert report.ok
    assert report.executable is False
    assert report.evidence_status == "hypothesis"
    assert report.sites[0].replacement_verdict == "geometry_ok"
    assert report.sites[1].replacement_verdict == "not_declared"
    assert "executable=false" in report.report()


def test_plan_check_shape_mismatch_is_a_finding() -> None:
    trace = _meta_trace()
    report = trace.check_plan([("relu_1_2", torch.empty(2, 8, device="meta"))])
    assert not report.ok
    assert report.sites[0].replacement_verdict.startswith("geometry_mismatch")
    assert "expects shape (2, 4)" in report.sites[0].replacement_verdict


def test_plan_check_selector_typo_is_a_finding_or_refusal() -> None:
    trace = _meta_trace()
    with pytest.raises(Exception):  # noqa: B017 — resolver refusal type is its own contract
        trace.check_plan([("relu_9_9", None)])


def test_plan_check_value_callable_refuses_typed() -> None:
    trace = _meta_trace()
    with pytest.raises(Exception) as excinfo:
        trace.check_plan([("relu_1_2", lambda t: t * 2)])
    assert excinfo.value.fields["code"] == "plan_check_unsupported"


@pytest.mark.smoke
def test_plan_check_refuted_source_refuses() -> None:
    from torchlens.capture.structure_only import StructureOnlyCapabilityError

    trace = _meta_trace()
    real = tl.trace(Toy(width=6), torch.randn(2, 4))
    assert trace.discharge_against(real).verdict.value == "refuted"
    with pytest.raises(StructureOnlyCapabilityError) as excinfo:
        trace.check_plan([("relu_1_2", None)])
    assert excinfo.value.fields["code"] == "structure_only_refuted_hypothesis"


def test_plan_check_never_arms_the_capture() -> None:
    trace = _meta_trace()
    trace.check_plan([("relu_1_2", torch.empty(2, 4, device="meta"))])
    assert not bool(getattr(trace, "intervention_ready", False))
    # The runnable refusal is untouched by a successful plan check.
    with pytest.raises(Exception) as excinfo:
        tl.save(trace, "/tmp/never-written.tlspec", level="runnable")
    assert "structure_only" in str(getattr(excinfo.value, "fields", {}).get("code", "")) or (
        "runnable" in str(excinfo.value)
    )


def test_plan_check_works_on_ordinary_traces_too() -> None:
    trace = tl.trace(Toy(), torch.randn(2, 4))
    report = trace.check_plan([("relu_1_2", torch.empty(2, 4))])
    assert report.ok
    assert report.evidence_status == "measured"
    assert report.executable is False
