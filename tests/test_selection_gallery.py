"""L6 stage 4a: the FULL ACCEPTANCE GALLERY (design memo sec 3.4).

The flagship composition demo of the selection algebra: ablate the
INTERSECTION of two units' receptive fields vs their UNION vs a size-matched
seeded RANDOM CONTROL, using the exact METAPLAN producer-to-producer
spelling (``u.receptive_field.at(p) & u.receptive_field.at(q)`` — no
explicit conversion calls).

THE COMMITTED ARTIFACT: seed 0, float64 CPU convs (outside the oneDNN
Winograd selection domain — the memo's pinned arithmetic assumption for
A1's exact zero), two stacked 3x3 convs on a 1x1x12x12 input; u1/u2 are the
late-conv units at channel 0 position (3, 3) and channel 1 position (5, 5),
whose input receptive fields are the 5x5 boxes [3:8]^2 and [5:10]^2
(overlap [5:8]^2 = 9 elements). Build-time verification measured (and this
suite re-asserts):

  inter ablation: d_u1 = 7.742e-3, d_u2 = 4.617e-2, d_out = 6.82e-2
  union ablation: d_u1 = 1.724e-1, d_u2 = 3.122e-2, d_out = 2.22e-1
  ctrl  ablation: d_u1 = 0.0 EXACTLY, d_u2 = 0.0 EXACTLY, d_out = 9.51e-2

The assertion honesty split (memo 3.4):
  A1 [THEOREM under exact geometry + direct convolution arithmetic]: the
     control ablation changes u1 and u2 by EXACTLY ZERO — zero tolerance;
     doubles as an RF-containment tripwire.
  A2 [PINNED-ARTIFACT REGRESSION]: the intersection ablation moves BOTH
     units above the stated floor (1e-3; measured min 7.7e-3). Not a
     theorem — signed weights permit cancellation; the committed artifact
     was verified at build to clear the floor. A red here means the
     artifact changed: that is the signal.
  A3 [REPORTED + cardinality THEOREM]: union-vs-inter per-unit magnitudes
     are REPORTED, never asserted (this very artifact shows the legal
     inversion: union's d_u2 < inter's d_u2 — superset ablation with a
     SMALLER delta via cancellation). Asserted: |inter| <= |union| and
     inter is element-nonempty — exact by construction.
  A4 [CONSTRUCTION]: |ctrl| == |inter| (size-matched by construction).

The gallery doubles as documentation of the algebra (three compositions +
one kind refusal) and closes with the stage-4a cross-run patching rows.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._selection import SelectionError

#: The two late-conv unit coordinates (channel, spatial) of the artifact.
P = (3, 3)
Q = (5, 5)
U1 = (0, 0, 3, 3)
U2 = (0, 1, 5, 5)

#: A2's pinned floor: build-time verified (measured min inter delta 7.7e-3).
A2_FLOOR = 1e-3

#: The seeded control's committed seed.
CTRL_SEED = 7


class _GalleryNet(nn.Module):
    """The committed gallery convnet: two stacked 3x3 convs, relu between."""

    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(1, 2, 3)
        self.c2 = nn.Conv2d(2, 2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run conv -> relu -> conv -> relu."""

        return torch.relu(self.c2(torch.relu(self.c1(x))))


def _gallery_capture() -> tl.Trace:
    """Capture the committed artifact: seed 0, float64 CPU, 1x1x12x12."""

    torch.manual_seed(0)
    model = _GalleryNet().double()
    x = torch.randn(1, 1, 12, 12, dtype=torch.float64)
    return tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(intervention_ready=True, save_args=True),
    )


@pytest.fixture(scope="module")
def gallery():
    """The baseline capture plus the three composed selections."""

    trace = _gallery_capture()
    late = trace["conv2d_2_3"]
    # The exact METAPLAN flagship spelling: producer-to-producer dispatch,
    # no explicit conversion calls.
    inter = late.receptive_field.at(P) & late.receptive_field.at(Q)
    union = late.receptive_field.at(P) | late.receptive_field.at(Q)
    ctrl = tl.random_like(like=inter, within=~union, seed=CTRL_SEED)
    try:
        yield trace, inter, union, ctrl
    finally:
        trace.cleanup()


def _ablation_deltas(trace: tl.Trace, selection: tl.Selector) -> tuple[float, float, float]:
    """Zero-ablate one selection on a fork; return (d_u1, d_u2, d_out)."""

    fork = trace.fork()
    fork.do(selection.resolve(fork), tl.zero_ablate())
    late_delta = fork["conv2d_2_3"].out - trace["conv2d_2_3"].out
    out_delta = fork["relu_2_4"].out - trace["relu_2_4"].out
    return (
        abs(float(late_delta[U1])),
        abs(float(late_delta[U2])),
        float(out_delta.abs().max()),
    )


def test_flagship_compositions_are_selections(gallery):
    """The producer expressions ARE Selections; resolution is explicit."""

    trace, inter, union, ctrl = gallery
    for composed in (inter, union, ctrl):
        assert isinstance(composed, tl.Selector)
    resolved_inter = inter.resolve(trace)
    resolved_union = union.resolve(trace)
    # Both land on the trace's input site with exact-relation provenance
    # (exact RF geometry ops only).
    for resolved in (resolved_inter, resolved_union):
        assert len(resolved) == 1
        assert resolved[0].key == ("input_1", 1)
        assert resolved[0].provenance.relation == "exact"


def test_a1_control_ablation_is_exactly_zero_on_both_units(gallery):
    """A1 [THEOREM]: out-of-RF ablation moves u1/u2 by EXACTLY zero.

    Zero tolerance under the pinned float64 CPU direct-convolution
    arithmetic; doubles as the RF-containment tripwire. A red here means
    containment (or the arithmetic pin) broke — never loosen to a tolerance.
    """

    trace, _, _, ctrl = gallery
    d_u1, d_u2, d_out = _ablation_deltas(trace, ctrl)
    assert d_u1 == 0.0
    assert d_u2 == 0.0
    # The control DOES move the network elsewhere (it is not a no-op).
    assert d_out > 0.0


def test_a2_intersection_ablation_moves_both_units_above_floor(gallery):
    """A2 [PINNED-ARTIFACT]: inter ablation clears the committed floor.

    Build-time verified on the committed seed-0 artifact (measured
    d_u1 = 7.7e-3, d_u2 = 4.6e-2 against the 1e-3 floor). A red means the
    artifact changed — that is the signal (memo descope rule applies).
    """

    trace, inter, _, _ = gallery
    d_u1, d_u2, _ = _ablation_deltas(trace, inter)
    assert d_u1 > A2_FLOOR
    assert d_u2 > A2_FLOOR


def test_a3_union_magnitudes_reported_and_cardinality_theorem(gallery):
    """A3: report union-vs-inter magnitudes; assert only the exact parts.

    The per-unit magnitudes are REPORTED (printed for the gallery record),
    never asserted — this artifact itself exhibits the legal inversion
    (union's d_u2 < inter's d_u2, cancellation under signed weights). The
    asserted parts are exact by construction: |inter| <= |union| and the
    intersection is element-nonempty.
    """

    trace, inter, union, _ = gallery
    resolved_inter = inter.resolve(trace)
    resolved_union = union.resolve(trace)
    inter_deltas = _ablation_deltas(trace, inter)
    union_deltas = _ablation_deltas(trace, union)
    report = {
        "inter": {"elements": sum(e.num_selected for e in resolved_inter)},
        "union": {"elements": sum(e.num_selected for e in resolved_union)},
    }
    for name, deltas in (("inter", inter_deltas), ("union", union_deltas)):
        report[name].update({"d_u1": deltas[0], "d_u2": deltas[1], "d_out": deltas[2]})
    print(f"GALLERY A3 REPORT: {report}")
    assert report["inter"]["elements"] <= report["union"]["elements"]
    assert resolved_inter.is_empty is False


def test_a4_control_is_size_matched_by_construction(gallery):
    """A4 [CONSTRUCTION]: |ctrl| == |inter|, sampled inside ~union."""

    trace, inter, union, ctrl = gallery
    resolved_inter = inter.resolve(trace)
    resolved_ctrl = ctrl.resolve(trace)
    assert sum(e.num_selected for e in resolved_ctrl) == sum(e.num_selected for e in resolved_inter)
    # The control is disjoint from the union (sampled from its complement).
    union_mask = union.resolve(trace)[0].mask
    ctrl_mask = resolved_ctrl[0].mask
    assert not bool((union_mask & ctrl_mask).any())
    # Seeded determinism: same seed, same sample.
    again = tl.random_like(like=inter, within=~union, seed=CTRL_SEED).resolve(trace)
    assert again == resolved_ctrl


def test_algebra_refusal_row_kind_incompatible(gallery):
    """The gallery's one refusal example: ACT & PARAM refuses typed."""

    trace, inter, _, _ = gallery
    with pytest.raises(SelectionError) as excinfo:
        _ = inter & tl.param("c2.weight")
    assert excinfo.value.fields["code"] == "selection_kind_incompatible"


def test_cross_run_patch_gallery_row():
    """Stage-4a closing row: RF-composed selection patched ACROSS RUNS.

    Resolve the intersection on run A, align it onto a fork of run B on the
    L1 site keys, patch A's input values in, and verify the fork's masked
    input elements equal run A's exactly while unmasked elements keep run
    B's — the full cross-run flow on the flagship composition, not just on
    a hand-built unit set.
    """

    torch.manual_seed(0)
    model = _GalleryNet().double()
    x_a = torch.randn(1, 1, 12, 12, dtype=torch.float64)
    x_b = torch.randn(1, 1, 12, 12, dtype=torch.float64)
    options = tl.options.CaptureOptions(intervention_ready=True, save_args=True)
    trace_a = tl.trace(model, x_a, capture=options)
    trace_b = tl.trace(model, x_b, capture=options)
    try:
        late = trace_a["conv2d_2_3"]
        inter = late.receptive_field.at(P) & late.receptive_field.at(Q)
        resolved_a = inter.resolve(trace_a)
        fork = trace_b.fork()
        aligned = resolved_a.align_to(fork)
        fork.do(aligned, tl.patch_from(trace_a))
        mask = resolved_a[0].mask
        patched_input = fork["input_1"].out
        assert torch.equal(patched_input[mask], trace_a["input_1"].out[mask])
        assert torch.equal(patched_input[~mask], trace_b["input_1"].out[~mask])
        downstream = (fork["relu_2_4"].out - trace_b["relu_2_4"].out).abs().max()
        assert float(downstream) > 0.0
    finally:
        trace_a.cleanup()
        trace_b.cleanup()
