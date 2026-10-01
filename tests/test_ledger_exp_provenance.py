"""F03 ledger memo item 4: the provenance join (why / provenance).

Pins the four honesty axes and the wording law: additive phrasing licensed
only when the reference-side suffix is empty; unrelated histories are never
joined by guess; emptiness is disclosure; ordered multiplicity (A vs
A-then-B) stays distinct; the value residual flags identical recorded chains
whose outputs disagree (the unrecorded-write detector).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import BundleMemberError

pytestmark = pytest.mark.smoke


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x))


def _ready() -> tl.Trace:
    torch.manual_seed(5)
    return tl.trace(
        _Tiny(), torch.randn(2, 3), capture=tl.options.CaptureOptions(intervention_ready=True)
    )


def test_why_exact_on_event_lineage_fork_chain() -> None:
    base = _ready().fork()
    base.do(tl.when(tl.func("relu"), tl.scale(0.5)))
    child = base.fork()
    child.do(tl.when(tl.func("relu"), tl.scale(2.0)))
    bundle = tl.Bundle({"base": base, "child": child}, baseline="base")

    report = bundle.why("child")
    assert report.lineage_status == "exact"
    assert report.lineage_basis == "event_lineage"
    assert report.common_prefix_len == 1
    assert len(report.member_suffix) == 1
    assert report.reference_suffix == ()
    assert report.payload_fidelity == "declared"
    assert "scale" in report.member_suffix[0].edit_names[0]
    text = report.describe()
    assert "'child' = 'base' + [" in text


def test_why_diverged_refuses_additive_wording() -> None:
    base = _ready().fork()
    base.do(tl.when(tl.func("relu"), tl.scale(0.5)))
    child = base.fork()
    child.do(tl.when(tl.func("relu"), tl.scale(2.0)))
    base.do(tl.when(tl.func("relu"), tl.add(1.0)))
    bundle = tl.Bundle({"base": base, "child": child}, baseline="base")

    report = bundle.why("child")
    assert report.lineage_status == "diverged"
    assert report.member_suffix and report.reference_suffix
    text = report.describe()
    assert "diverged" in text and "additive wording refused" in text
    assert "= 'base' +" not in text


def test_why_ordered_multiplicity_stays_distinct() -> None:
    base = _ready().fork()
    base.do(tl.when(tl.func("relu"), tl.scale(0.5)))
    once = base.fork()
    twice = base.fork()
    twice.do(tl.when(tl.func("relu"), tl.scale(0.5)))
    bundle = tl.Bundle({"base": base, "once": once, "twice": twice}, baseline="base")

    report_once = bundle.why("once")
    report_twice = bundle.why("twice")
    assert report_once.lineage_status == "exact"
    assert report_once.member_suffix == ()
    assert report_twice.lineage_status == "exact"
    assert len(report_twice.member_suffix) == 1  # A-then-A != A


def test_why_container_basis_on_sweep_bundle() -> None:
    torch.manual_seed(0)
    model = _Tiny()
    x = torch.randn(2, 3)
    bundle = tl.sweep(model, x, at="relu", values=[0.25, 0.75], include_baseline=True)
    report = bundle.why("sweep_1")
    assert report.lineage_status == "exact"
    assert report.lineage_basis == "container_operation"
    assert report.payload_fidelity == "declared"
    assert any("0.75" in name for name in report.member_suffix[0].edit_names)


def test_why_unrelated_and_unattested_disclose_never_fabricate() -> None:
    first = _ready().fork()
    first.do(tl.when(tl.func("relu"), tl.scale(0.5)))
    second = _ready().fork()
    second.do(tl.when(tl.func("relu"), tl.scale(0.5)))
    bundle = tl.Bundle({"first": first, "second": second}, baseline="first")
    report = bundle.why("second")
    assert report.lineage_status == "unrelated"
    assert "relate()" in report.describe()

    pristine = tl.Bundle({"a": _ready(), "b": _ready()}, baseline="a")
    unattested = pristine.why("b")
    assert unattested.lineage_status == "unattested"
    assert "no recorded construction evidence" in unattested.describe()


def test_why_opaque_payload_fidelity_names_fact_not_content() -> None:
    base = _ready().fork()
    child = base.fork()

    def user_hook(out: torch.Tensor, *, hook: object) -> torch.Tensor:
        return out * 3.0

    child.do("relu_1_2", user_hook)
    bundle = tl.Bundle({"base": base, "child": child}, baseline="base")
    report = bundle.why("child")
    # A bare user callable names the site and the fact, never the content.
    assert report.payload_fidelity == "opaque"


def test_value_residual_flags_unrecorded_write() -> None:
    base = _ready().fork()
    base.do(tl.when(tl.func("relu"), tl.scale(0.5)))
    clone = base.fork()  # identical recorded chain, shared payloads
    bundle = tl.Bundle({"base": base, "clone": clone}, baseline="base")
    clean = bundle.why("clone")
    assert clean.lineage_status == "exact"
    assert clean.value_residual == "explained"

    # An unrecorded write: replace the clone's output value without any event
    # (assignment on the Op record; no envelope row is minted).
    out_label = clone.output_layers[0]
    output_op = clone[out_label].ops[0]
    output_op.out = output_op.out + 1.0
    dirty = bundle.why("clone")
    assert dirty.value_residual == "unexplained"


def test_why_refusals_are_typed() -> None:
    bundle = tl.Bundle({"a": _ready(), "b": _ready()})
    with pytest.raises(BundleMemberError) as excinfo:
        bundle.why("a")
    assert excinfo.value.fields["code"] == "provenance_reference_missing"
    with pytest.raises(BundleMemberError) as excinfo:
        bundle.why("a", relative_to="a")
    assert excinfo.value.fields["code"] == "provenance_self_comparison"


def test_provenance_rows_one_per_member() -> None:
    torch.manual_seed(0)
    model = _Tiny()
    x = torch.randn(2, 3)
    bundle = tl.sweep(model, x, at="relu", values=[0.0], include_baseline=True)
    rows = bundle.provenance()
    by_member = {row["member"]: row for row in rows}
    assert by_member["baseline"]["lineage_status"] == "baseline"
    assert by_member["sweep_0"]["lineage_status"] == "exact"
    assert by_member["sweep_0"]["origin"]["origin"] == "swept"
    assert by_member["sweep_0"]["n_events"] == 1
    # No baseline -> disclosure, not a default.
    plain = tl.Bundle({"a": _ready(), "b": _ready()})
    assert {row["lineage_status"] for row in plain.provenance()} == {"no_baseline"}
