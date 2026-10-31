"""L6 stage 4a: cross-run selection alignment (``Selection.align_to``).

Pins the cross-run door: alignment keys on the L1 structural site keys each
ACT entry records (the bridging relation), SAME-POLICY captures only per the
L1 cross-stamp rule (healthy, agreeing grouping stamps on both sides), masks
travel unchanged onto an identical index space, and ``do()`` keeps refusing
foreign resolved selections typed — ``align_to`` is the only rebind, and it
is explicit. The refusal matrix covers every reason in the closed
``selection_alignment_invalid`` set: ``kind_unsupported`` /
``grouping_stamp_degraded`` / ``grouping_stamp_mismatch`` /
``site_key_unavailable`` / ``site_not_in_target`` / ``index_space_mismatch``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.errors import SelectionError

#: Committed index set for the aligned patch rows.
IDX = ((0, 0, 1, 1), (0, 1, 2, 2))


class _TwoConv(nn.Module):
    """Tiny deterministic CPU convnet (exact-geometry ops only)."""

    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(1, 2, 3)
        self.c2 = nn.Conv2d(2, 2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run two convs with interleaved relus."""

        return torch.relu(self.c2(torch.relu(self.c1(x))))


def _capture(model: nn.Module, x: torch.Tensor) -> tl.Trace:
    """Capture one intervention-ready trace of the shared model."""

    return tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(intervention_ready=True, save_args=True),
    )


@pytest.fixture(scope="module")
def pair():
    """Two same-policy captures of ONE model on different inputs."""

    torch.manual_seed(0)
    model = _TwoConv().double()
    x_a = torch.randn(1, 1, 12, 12, dtype=torch.float64)
    x_b = torch.randn(1, 1, 12, 12, dtype=torch.float64)
    trace_a = _capture(model, x_a)
    trace_b = _capture(model, x_b)
    try:
        yield trace_a, trace_b
    finally:
        trace_a.cleanup()
        trace_b.cleanup()


def _alignment_reason(excinfo: pytest.ExceptionInfo) -> str:
    """Return the closed alignment refusal reason from a SelectionError."""

    assert excinfo.value.fields["code"] == "selection_alignment_invalid"
    return excinfo.value.fields["reason"]


def test_entries_record_the_l1_structural_site_key(pair):
    """ACT entries carry the op's site_key_v1 string (the bridging relation)."""

    trace_a, _ = pair
    resolved = tl.units("relu_1_2", IDX).resolve(trace_a)
    entry = resolved[0]
    assert isinstance(entry.structural_site_key, str)
    assert entry.structural_site_key.startswith("s1|")


def test_align_to_same_trace_is_identity(pair):
    """Alignment onto the selection's own trace returns the selection itself."""

    trace_a, _ = pair
    resolved = tl.units("relu_1_2", IDX).resolve(trace_a)
    assert resolved.align_to(trace_a) is resolved


def test_align_to_rebinds_with_masks_and_provenance(pair):
    """Happy path: target binding, unchanged masks, cross-run disclosure."""

    trace_a, trace_b = pair
    resolved = tl.units("relu_1_2", IDX).resolve(trace_a)
    aligned = resolved.align_to(trace_b)
    assert aligned is not resolved
    assert aligned._trace is trace_b
    assert len(aligned) == len(resolved)
    for source_entry, aligned_entry in zip(resolved, aligned, strict=True):
        assert aligned_entry.key == source_entry.key
        assert aligned_entry.structural_site_key == source_entry.structural_site_key
        assert torch.equal(aligned_entry.mask, source_entry.mask)
        assert aligned_entry.provenance.relation == source_entry.provenance.relation
        assert "cross-run aligned from" in aligned_entry.provenance.source
    # Content digest is mask/site-based, so a label-identical alignment agrees.
    assert aligned.digest == resolved.digest


def test_do_still_refuses_foreign_resolved_selection(pair):
    """The trace-mismatch tripwire stays armed; its remedy teaches align_to."""

    trace_a, trace_b = pair
    resolved = tl.units("relu_1_2", IDX).resolve(trace_a)
    fork = trace_b.fork()
    with pytest.raises(SelectionError) as excinfo:
        fork.do(resolved, tl.zero_ablate())
    assert excinfo.value.fields["code"] == "selection_trace_mismatch"
    assert "align_to" in str(excinfo.value)


def test_cross_run_patch_lands_source_values(pair):
    """Flagship flow: resolve on A, align to a fork of B, patch from A.

    Masked elements take trace A's recorded values, unmasked elements keep
    trace B's, and the patch propagates a real downstream delta.
    """

    trace_a, trace_b = pair
    resolved = tl.units("relu_1_2", IDX).resolve(trace_a)
    fork = trace_b.fork()
    aligned = resolved.align_to(fork)
    fork.do(aligned, tl.patch_from(trace_a))
    value_a = trace_a["relu_1_2"].out
    value_b = trace_b["relu_1_2"].out
    patched = fork["relu_1_2"].out
    for index in IDX:
        assert patched[index] == value_a[index]
    untouched = (0, 0, 0, 0)
    assert untouched not in IDX
    assert patched[untouched] == value_b[untouched]
    # The values differ across runs, so the patch must move downstream ops.
    assert any(value_a[index] != value_b[index] for index in IDX)
    downstream_delta = (fork["relu_2_4"].out - trace_b["relu_2_4"].out).abs().max()
    assert float(downstream_delta) > 0.0
    audit = fork.intervention_audit[-1]
    assert audit["resolve_digest"] == aligned.digest
    assert audit["patch_source"]["source_object_id"] == str(id(trace_a))


def test_alignment_mask_stays_immutable(pair):
    """Mutating a mask returned by an aligned entry cannot alter the selection."""

    trace_a, trace_b = pair
    resolved = tl.units("relu_1_2", IDX).resolve(trace_a)
    aligned = resolved.align_to(trace_b)
    digest = aligned.digest
    mask = aligned[0].mask
    mask.zero_()
    assert aligned.digest == digest
    assert aligned[0].num_selected == len(IDX)


def test_refusal_kind_unsupported(pair):
    """PARAM selections refuse alignment (addresses not run-stable)."""

    trace_a, trace_b = pair
    resolved = tl.param("c1.weight").resolve(trace_a)
    with pytest.raises(SelectionError) as excinfo:
        resolved.align_to(trace_b)
    assert _alignment_reason(excinfo) == "kind_unsupported"


def test_refusal_grouping_stamp_degraded(pair):
    """A degraded stamp on either side refuses (L1 consumer-matrix row)."""

    from torchlens._postprocess._grouping_stamp import degraded_grouping_policy_stamp

    trace_a, trace_b = pair
    resolved = tl.units("relu_1_2", IDX).resolve(trace_a)
    healthy = trace_b.grouping_policy
    trace_b.grouping_policy = degraded_grouping_policy_stamp("parse_failure")
    try:
        with pytest.raises(SelectionError) as excinfo:
            resolved.align_to(trace_b)
        assert _alignment_reason(excinfo) == "grouping_stamp_degraded"
        assert excinfo.value.fields["role"] == "target"
    finally:
        trace_b.grouping_policy = healthy
    missing = dict(trace_a.grouping_policy or {})
    trace_a.grouping_policy = None
    try:
        with pytest.raises(SelectionError) as excinfo:
            resolved.align_to(trace_b)
        assert _alignment_reason(excinfo) == "grouping_stamp_degraded"
        assert excinfo.value.fields["role"] == "source"
    finally:
        trace_a.grouping_policy = missing


def test_refusal_grouping_stamp_mismatch(pair):
    """Cross-stamp captures refuse typed (no relaxation minted here)."""

    trace_a, trace_b = pair
    resolved = tl.units("relu_1_2", IDX).resolve(trace_a)
    healthy = trace_b.grouping_policy
    crossed = dict(healthy)
    crossed["policy"] = "params_only"
    trace_b.grouping_policy = crossed
    try:
        with pytest.raises(SelectionError) as excinfo:
            resolved.align_to(trace_b)
        assert _alignment_reason(excinfo) == "grouping_stamp_mismatch"
        assert excinfo.value.fields["axis"] == "policy"
    finally:
        trace_b.grouping_policy = healthy


def test_refusal_site_key_unavailable(pair):
    """A keyless entry (legacy capture) cannot bridge."""

    import dataclasses

    from torchlens._selection import Selection

    trace_a, trace_b = pair
    resolved = tl.units("relu_1_2", IDX).resolve(trace_a)
    keyless = Selection(
        trace_a,
        "ACT",
        [dataclasses.replace(entry, structural_site_key=None) for entry in resolved],
    )
    with pytest.raises(SelectionError) as excinfo:
        keyless.align_to(trace_b)
    assert _alignment_reason(excinfo) == "site_key_unavailable"


def test_refusal_site_not_in_target(pair):
    """A different architecture has no matching (address, site key) pair."""

    trace_a, _ = pair

    class _OneConv(nn.Module):
        """Single-conv control architecture (no relu_1_2 twin geometry)."""

        def __init__(self) -> None:
            super().__init__()
            self.c1 = nn.Conv2d(1, 2, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run one conv then a tanh (no relu sites at all)."""

            return torch.tanh(self.c1(x))

    torch.manual_seed(1)
    other = _capture(_OneConv().double(), torch.randn(1, 1, 12, 12, dtype=torch.float64))
    try:
        resolved = tl.units("relu_1_2", IDX).resolve(trace_a)
        with pytest.raises(SelectionError) as excinfo:
            resolved.align_to(other)
        assert _alignment_reason(excinfo) == "site_not_in_target"
    finally:
        other.cleanup()


def test_refusal_index_space_mismatch(pair):
    """Same site, different input geometry: masks cannot re-index."""

    trace_a, _ = pair
    torch.manual_seed(2)
    smaller = _capture(_TwoConv().double(), torch.randn(1, 1, 10, 10, dtype=torch.float64))
    try:
        resolved = tl.units("relu_1_2", IDX).resolve(trace_a)
        with pytest.raises(SelectionError) as excinfo:
            resolved.align_to(smaller)
        assert _alignment_reason(excinfo) == "index_space_mismatch"
    finally:
        smaller.cleanup()
