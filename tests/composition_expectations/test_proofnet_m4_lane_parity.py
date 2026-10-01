"""M4 equivalent-lane parity (compo memo stratum 1; F36 wave A).

The same semantic request down different lanes (string label, Selection,
predicate, accessor) must accept/refuse IDENTICALLY -- the memo's founding
counterexamples were zero-match asymmetries (a selector lane refuses typed
while the plan lane silently warns). This suite pins the CURRENT parity
truth cell by cell: parity holds where it holds, and every measured
asymmetry is a ledgered cell whose silent arm cannot drift further without
going red.

Ground truth probed live on the merged tree, 2026-08-30.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.heavy, pytest.mark.compo]


#: Memo D12 fixture-economics license (test_galleries.py lint):
COMPO_CONSTRUCTION_LICENSE = "lane-parity cells compare CONSTRUCTION-time lanes (save= predicates vs layers_to_save) and need private captures per lane arm"

MISSING_SITE = "sigmoid_1_1"  # no sigmoid anywhere in the fixture model
REAL_SITE = "relu_1_2"


@pytest.fixture(scope="module")
def parity_trace() -> Any:
    """One intervention-ready capture every lane cell reads."""

    import torchlens as tl

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)
    ).eval()
    trace = tl.trace(
        model,
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )
    yield trace
    trace.cleanup()


def _code(exc: BaseException) -> Any:
    return getattr(exc, "fields", {}).get("code") if hasattr(exc, "fields") else None


def test_zero_match_getitem_and_label_list_refuse_with_one_code(parity_trace: Any) -> None:
    """The two label-shaped read lanes agree: typed op_lookup_not_found."""

    import torchlens as tl

    with pytest.raises(Exception) as getitem_exc:
        parity_trace[MISSING_SITE]
    assert _code(getitem_exc.value) == "op_lookup_not_found"

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)
    ).eval()
    with pytest.raises(Exception) as list_exc:
        tl.trace(
            model,
            torch.randn(2, 4),
            capture=tl.options.CaptureOptions(layers_to_save=[MISSING_SITE]),
        )
    assert _code(list_exc.value) == "op_lookup_not_found", (
        "layers_to_save zero-match diverged from the getitem lane"
    )


def test_zero_match_selection_lane_refuses_typed(parity_trace: Any) -> None:
    """tl.units on a missing site refuses selection_unresolvable."""

    import torchlens as tl

    with pytest.raises(Exception) as excinfo:
        tl.units(MISSING_SITE, [(0, 0)]).resolve(parity_trace)
    assert _code(excinfo.value) == "selection_unresolvable"


def test_zero_match_do_lane_is_the_ledgered_codeless_cell(parity_trace: Any) -> None:
    """The do() string lane refuses SiteResolutionError with code=None.

    LEDGERED ASYMMETRY (compo memo named first-wave cell): typed and
    teaching in prose, code=None -- the fields-clobber class. When the
    intervention-errors fix lands a code here, this cell must be promoted
    deliberately.
    """

    fork = parity_trace.fork()
    try:
        with pytest.raises(Exception) as excinfo:
            fork.do(MISSING_SITE, __import__("torchlens").zero_ablate())
        exc = excinfo.value
        assert type(exc).__name__ == "SiteResolutionError"
        assert _code(exc) is None, (
            "the do() zero-match refusal now carries a code -- promote this"
            " ledgered asymmetry cell to a typed-parity assertion"
        )
    finally:
        fork.cleanup()


def test_zero_match_save_predicate_lane_warns_instead_of_refusing() -> None:
    """The predicate save lane's zero-match is a WARNING, not a refusal.

    LEDGERED ASYMMETRY (M4's founding counterexample, still live): the
    selector lanes refuse typed while the capture-plan lane completes with
    a disclosure warning. Pinned so the asymmetry cannot silently widen; a
    lane that flips this to a refusal must update the cell.
    """

    import torchlens as tl

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)
    ).eval()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.trace(model, torch.randn(2, 4), save=tl.func("sigmoid"))
    try:
        assert trace.outcome.status.name == "COMPLETE"
        zero_match_warnings = [
            str(entry.message) for entry in caught if "save selector" in str(entry.message)
        ]
        assert zero_match_warnings, (
            "the predicate lane's zero-match disclosure warning disappeared:"
            " either the lane now refuses (update this cell to parity) or the"
            " zero-match became fully silent (a regression -- SG-class)"
        )
    finally:
        trace.cleanup()


def test_accept_parity_string_and_selection_lanes_edit_identically(
    parity_trace: Any,
) -> None:
    """The SAME zero-ablation through the string lane, the tl.units lane,
    and the Layer __selection__ lane produces IDENTICAL edited payloads."""

    import torchlens as tl

    results = []
    for lane in ("string", "units", "layer_selection"):
        fork = parity_trace.fork()
        try:
            if lane == "string":
                fork.do(REAL_SITE, tl.zero_ablate())
            elif lane == "units":
                shape = tuple(parity_trace[REAL_SITE].out.shape)
                all_indices = [
                    tuple(idx)
                    for idx in torch.cartesian_prod(*[torch.arange(dim) for dim in shape]).tolist()
                ]
                fork.do(tl.units(REAL_SITE, all_indices).resolve(fork), tl.zero_ablate())
            else:
                fork.do(fork[REAL_SITE].__selection__().resolve(fork), tl.zero_ablate())
            results.append((lane, fork[REAL_SITE].out.clone(), fork.output_ops[0].out.clone()))
        finally:
            fork.cleanup()
    baseline_site, baseline_output = results[0][1], results[0][2]
    assert torch.count_nonzero(baseline_site) == 0, "the ablation did not land"
    for lane, site_payload, output_payload in results[1:]:
        assert torch.equal(site_payload, baseline_site), f"lane {lane} edited payload differs"
        assert torch.equal(output_payload, baseline_output), (
            f"lane {lane} propagated a DIFFERENT downstream output than the"
            " string lane -- accept parity broken"
        )


def test_engine_parity_downstream_recompute_matches_manual_forward(
    parity_trace: Any,
) -> None:
    """Push-engine truth against an independent oracle: zero-ablating relu
    equals hand-running the tail linear on a zeroed activation."""

    import torchlens as tl

    fork = parity_trace.fork()
    try:
        fork.do(REAL_SITE, tl.zero_ablate())
        edited_output = fork.output_ops[0].out
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)
        ).eval()
        with torch.no_grad():
            manual = model[2](torch.zeros(2, 8))
        assert torch.allclose(edited_output, manual, atol=1e-6), (
            "push-engine recompute disagrees with the hand-coded frozen-"
            "complement forward (independent oracle, assertion-ladder rung 1)"
        )
    finally:
        fork.cleanup()
