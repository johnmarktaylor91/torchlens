"""F10 lovely item 7: container/query/result conversion laws.

Containers point, records show (voice rule 1); honesty tokens are shared
vocabulary and cannot be laundered by any presenter (rule 4); no public
dataclass on the swept surfaces inherits an auto repr (D31, extended
sweep); no ANSI/OSC-8 bytes in any returned string.
"""

from __future__ import annotations

import dataclasses
import re

import pytest
import torch
from torch import nn

import torchlens as tl


class _Deep(nn.Module):
    """Deep sequential + a buffer so print(trace) exercises every rung."""

    def __init__(self) -> None:
        super().__init__()
        self.stack = nn.Sequential(*[nn.Linear(4, 4) for _ in range(20)])
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.stack:
            x = torch.relu(layer(x))
        return self.bn(x)


@pytest.fixture(scope="module")
def deep_trace():
    """One finished capture of the deep model."""

    trace = tl.trace(_Deep().eval(), torch.randn(4, 4))
    yield trace
    trace.cleanup()


def test_trace_repr_carries_health_proof(deep_trace) -> None:
    """The identity repr gains the whole-trace proof line when earned."""

    line = repr(deep_trace)
    assert "\n" not in line
    assert "Trace(" in line
    # Default full-save captures derive the verdict from saved payloads.
    if deep_trace.nonfinite_verdict == "checked_and_clean":
        assert "no NaN/Inf (exact)" in line


def test_trace_str_is_bounded_with_folds(deep_trace) -> None:
    """print(trace): buffer rows fold; roster elides with exact remainder."""

    text = str(deep_trace)
    lines = text.splitlines()
    assert len(lines) < 80, f"bounded execution log regressed to {len(lines)} lines"
    assert any("buffer rows folded" in line for line in lines)
    assert any("more layers (see .layer_list)" in line for line in lines)


@pytest.mark.smoke
def test_trace_str_honesty_header_on_not_checked() -> None:
    """State precedes detail: a NOT-CHECKED capture says so up front."""

    trace = tl.trace(
        nn.Identity(),
        torch.randn(2, 2),
        capture=tl.options.CaptureOptions(layers_to_save=None),
    )
    text = str(trace)
    assert trace.nonfinite_verdict == "not_checked"
    assert "Health: NOT-CHECKED" in text
    # The header sits ABOVE structure/tensor detail (state precedes detail).
    assert text.index("Health: NOT-CHECKED") < text.index("Structure:")
    trace.cleanup()


def test_accessors_point_never_dump(deep_trace) -> None:
    """Collection reprs are one-line composition cards with exits."""

    for accessor in (deep_trace.layers, deep_trace.params, deep_trace.modules, deep_trace.buffers):
        line = repr(accessor)
        assert "\n" not in line
        assert ".head()" in line and ".find()" in line
    assert "buffers" in repr(deep_trace.layers)  # composition note names buffers
    view = str(deep_trace.layers)
    assert len(view.splitlines()) <= 24  # voice rule 9 collection budget
    assert "... " in view  # exact remainder, never a silent cut


@pytest.mark.smoke
def test_accessor_head_and_find(deep_trace) -> None:
    """head()/find() are the named exits and return records."""

    head = deep_trace.layers.head(3)
    assert len(head) == 3
    found = deep_trace.params.find("weight")
    assert found and all("weight" in p.address for p in found)


@pytest.mark.smoke
def test_partial_trace_badge_first() -> None:
    """PartialTrace: the badge leads; the card names the failure."""

    class _Boom(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.relu(x)
            raise RuntimeError("mid-forward failure")

    with pytest.raises(Exception) as exc_info:
        tl.trace(_Boom(), torch.randn(2, 2))
    partial = tl.partial.from_failed_capture(exc_info.value)
    line = repr(partial)
    assert line.startswith("PartialTrace [partial]")
    card = str(partial)
    assert card.splitlines()[0] == line
    assert "error:" in card and "More:" in card


@pytest.mark.smoke
def test_bundle_repr_distribution_and_member_lines() -> None:
    """Bundle: outcome distribution above the fold; members one-line each."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    a = tl.trace(model, torch.randn(2, 4))
    b = tl.trace(model, torch.randn(2, 4))
    bundle = tl.Bundle({"a": a, "b": b}, baseline="a")
    line = repr(bundle)
    assert "baseline='a'" in line and "n_members=2" in line
    view = str(bundle)
    assert view.splitlines()[0] == line
    assert "a (baseline):" in view
    a.cleanup()
    b.cleanup()


def test_edge_record_repr_designed() -> None:
    """Edge records render source -> target lines, sentinel-silent."""

    class _Viewy(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.view(-1).relu()

    trace = tl.trace(
        _Viewy(),
        torch.randn(2, 2),
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )
    records = [e for op in trace.layer_list for e in (getattr(op, "edge_uses", ()) or ())]
    assert records
    texts = [repr(e) for e in records]
    assert all(text.startswith("edge ") and " -> " in text for text in texts)
    assert any("[view]" in text for text in texts)
    assert all("unknown" not in text for text in texts)  # sentinel never prints
    trace.cleanup()


@pytest.mark.smoke
def test_report_protocol_card_shape() -> None:
    """The common report lead + bounded findings + exact remainder."""

    from torchlens.report._report_protocol import ReportLead, render_report_card

    lead = ReportLead(
        report_type="DemoAudit",
        subject="trace_1",
        status="findings",
        severity_counts={"high": 2, "low": 5},
        coverage="7/7 ops examined",
    )
    card = render_report_card(lead, [f"finding {i}" for i in range(7)])
    lines = card.splitlines()
    assert lines[0].startswith("DemoAudit(trace_1) status=findings")
    assert "... 2 more findings" in card  # exact remainder
    assert "More:" in lines[-1]


_ANSI_OSC = re.compile(r"\x1b|\x9b|\x07")


@pytest.mark.smoke
def test_no_terminal_bytes_in_container_strings(deep_trace) -> None:
    """No ANSI/OSC-8/terminal bytes in ANY returned container string."""

    for text in (
        repr(deep_trace),
        str(deep_trace),
        repr(deep_trace.layers),
        str(deep_trace.layers),
        repr(deep_trace.params),
        repr(deep_trace.modules),
        repr(deep_trace.buffers),
    ):
        assert not _ANSI_OSC.search(text)


def _has_auto_repr(cls: type) -> bool:
    """Whether a dataclass carries the GENERATED dataclass __repr__.

    The D31 defect class is the generated field-dump repr (it recurses and
    OOMs); ``repr=False`` classes that fall back to ``object.__repr__``
    are bounded by construction and pass this sweep.
    """

    repr_fn = getattr(cls, "__repr__", None)
    qualname = getattr(repr_fn, "__qualname__", "")
    wrapped = getattr(repr_fn, "__wrapped__", None)
    return "__create_fn__" in qualname or (
        wrapped is not None and "__create_fn__" in getattr(wrapped, "__qualname__", "")
    )


#: Small frozen FACT records whose auto reprs are bounded by construction
#: (a handful of scalar fields); the D31 law targets aggregate/result/
#: report/context classes whose auto reprs recurse or dump. The three
#: observability kernels re-exported through tl.stats are C06-owned
#: records (bounded log2-grid descriptors) -- their repr design belongs to
#: that owner, so they are ledgered here rather than redesigned cross-lane.
_AUTO_REPR_ALLOWED = {
    "FamilyEvidence",
    "EdgeDistributionRelation",
    "ReportLead",
    "StatsTableRow",
    # C06-owned re-exports (bounded descriptors; ownership seam):
    "HistogramDescriptor",
    "HistogramResult",
    "SpineResult",
    # C02/F08/F09-owned bounded FACT rows (measured <=320 chars on a real
    # capture; the unbounded containers FactCore / ComputeAggregation /
    # IdentityIndex got designed reprs in F10):
    "CaptureFacts",
    "ComputeRow",
    "CountsRecord",
    "HealthFacts",
    "MemoryFacts",
    "ParamFacts",
    "RatioFact",
    "SummaryRow",
    "SummaryTotals",
    # F09-owned bounded FACT rows landed at the T66 reconcile (all-scalar
    # per-row records and static registry/catalog rows; measured 175-465
    # chars on a real capture -- the F09 aggregates that scale with the
    # model, CostTree / FlopsReport / RooflineResult / CapabilityCard /
    # BackwardEstimate / PaddingWaste / MfuResult / DevicePeaks /
    # InstrumentedRateTable / CostReportResult, got designed reprs):
    "BackwardStatus",
    "CapabilityRow",
    "CleanStepTime",
    "CostTierMeasurement",
    "CostTreeRow",
    "InstrumentedRateRow",
    "MachineBalance",
    "MfuProvenance",
    "NextStep",
    "PeakRow",
    "RateCalibration",
    "RooflineRow",
    "SixNDFacts",
    "SurfaceEntry",
}


def test_no_auto_repr_on_f10_surfaces() -> None:
    """D31 sweep over the F10-owned surfaces (stats + report protocol)."""

    import torchlens.report as report
    import torchlens.stats as stats

    for module, names in ((stats, stats.__all__), (report, report.__all__)):
        for name in names:
            obj = getattr(module, name)
            if not isinstance(obj, type) or not dataclasses.is_dataclass(obj):
                continue
            if obj.__name__ in _AUTO_REPR_ALLOWED:
                continue
            assert not _has_auto_repr(obj), (
                f"{module.__name__}.{name} inherits the dataclass auto-repr; "
                "aggregate/result/report records must design theirs (D31)"
            )
