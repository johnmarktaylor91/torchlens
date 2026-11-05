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
        capture=tl.options.CaptureOptions(replayable=True),
    )
    yield trace
    trace.cleanup()


def _code(exc: BaseException) -> Any:
    return getattr(exc, "fields", {}).get("code") if hasattr(exc, "fields") else None


def test_zero_match_selection_lane_refuses_typed(parity_trace: Any) -> None:
    """tl.units on a missing site refuses selection_unresolvable."""

    import torchlens as tl

    with pytest.raises(Exception) as excinfo:
        parity_trace[tl.units(MISSING_SITE, [(0, 0)])]
    assert _code(excinfo.value) == "selection_unresolvable"


def test_zero_match_do_lane_refuses_typed(parity_trace: Any) -> None:
    """The do() string lane refuses SiteResolutionError with site_selector_no_match.

    Promoted from the ledgered code=None asymmetry cell (compo memo named
    first-wave cell) once the site-resolution refusals gained stable codes.
    """

    fork = parity_trace.fork()
    try:
        with pytest.raises(Exception) as excinfo:
            fork.do(MISSING_SITE, __import__("torchlens").zero_ablate())
        exc = excinfo.value
        assert type(exc).__name__ == "SiteResolutionError"
        assert _code(exc) == "site_selector_no_match"
    finally:
        fork.cleanup()
