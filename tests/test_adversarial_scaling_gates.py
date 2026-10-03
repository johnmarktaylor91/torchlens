"""Adversarial scaling gates: pathological-but-legal shapes with exact pass/fail bars.

Each gate runs a REAL capture at a size where a complexity regression is
catastrophic but current behavior is comfortable (sizes calibrated 2026-08-19,
torch 2.13, the benchmark host: wide-equivalence 300 ~0.9s, 300-pass recurrence ~1.9s,
1400-op diameter ~3.9s, fan-out 300 ~1.0s). The wall-clock bar is the tier
duration tripwire itself (smoke <5s / heavy <20s, load-scaled, min(wall, cpu)):
a regression from linear to quadratic work blows the budget and fails the
session, while the assertions below pin CORRECTNESS at scale — exact layer
counts, exact relation tuples, O(1) identity-shared group views, and silent
(disclosure-free) grouping below the real ceilings. No ceiling constant is
monkeypatched here; the tiny-ceiling behavior gates live in
test_scaling_ladder.py.

Axes and why they are load-bearing:

- GIANT EQUIVALENCE CLASS (300 mutually-unreachable equal-signature ops): the
  pair-probe triangle is O(cohort^2) bounded by _PF_COHORT_PAIR_PROBE_CEILING,
  and equivalent_ops/recurrent_ops are live views that must stay ONE shared
  cached object per group — a fresh-copy-per-read regression turns every sweep
  over members into O(n^2) memory.
- DEEP RECURRENCE (one module, 300 passes): pass grouping, pass-qualified
  addressing at the extreme pass, and the shared site_key must all hold at
  depths far beyond the toy sizes the semantic suites use.
- GRAPH DIAMETER (1400 sequential ops under the DEFAULT 1000-frame recursion
  limit): proves every postprocess traversal (ancestor closures, distance
  flood, grouping) stays iterative — a regression to recursion fails
  deterministically, machine speed notwithstanding.
- DEPTH OVERFLOW RECOVERY (module nesting past the recursion limit): TorchLens
  adds one decorated_forward frame per module call, so a deep-but-legal model
  can overflow ONLY under capture; the failure must stay clean — typed
  warning, original RecursionError, model and torch environment restored.
- WIDE FAN-OUT / FAN-IN (one tensor consumed by 300 children, one op with 300
  parents): relation tuples and their ordering must stay exact at widths where
  quadratic bookkeeping would show.
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Callable

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import CaptureAttemptFailedWarning

N_STREAMS = 300
N_PASSES = 300
N_CHAIN = 1400
N_FANOUT = 300
NEST_DEPTH_OVERFLOW = 400


class _ParallelStreams(nn.Module):
    """N mutually-unreachable equal-signature mults behind one anchored parent."""

    def __init__(self, streams: int) -> None:
        super().__init__()
        self.streams = streams
        self.lin = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        ys = [h * 2.0 for _ in range(self.streams)]
        return torch.stack(ys).sum(dim=0)


class _DeepRecurrence(nn.Module):
    """One cell applied ``steps`` times: one layer with ``steps`` passes."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor, steps: int) -> torch.Tensor:
        for _ in range(steps):
            x = torch.tanh(self.cell(x))
        return x


class _Chain(nn.Module):
    """A functional chain of ``steps`` adds: graph diameter without call depth."""

    def forward(self, x: torch.Tensor, steps: int) -> torch.Tensor:
        for _ in range(steps):
            x = x + 1.0
        return x


class _FanOut(nn.Module):
    """One input consumed by ``n`` children, re-joined by one ``n``-parent stack."""

    def forward(self, x: torch.Tensor, n: int) -> torch.Tensor:
        return torch.stack([x * float(i + 1) for i in range(n)]).sum(dim=0)


def _nested_sequential(depth: int) -> nn.Module:
    module: nn.Module = nn.Linear(4, 4)
    for _ in range(depth):
        module = nn.Sequential(module)
    return module


@pytest.fixture(scope="module")
def wide_capture():
    """The giant-equivalence-class trace plus every warning capture emitted."""

    torch.manual_seed(0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        log = tl.trace(_ParallelStreams(N_STREAMS), torch.randn(3))
    try:
        yield log, caught
    finally:
        log.cleanup()


@pytest.fixture(scope="module")
def deep_capture():
    """The 300-pass recurrence trace."""

    torch.manual_seed(0)
    log = tl.trace(_DeepRecurrence(), (torch.randn(2, 8), N_PASSES))
    try:
        yield log
    finally:
        log.cleanup()


@pytest.fixture(scope="module")
def chain_capture():
    """The 1400-op diameter trace, captured under the ambient recursion limit."""

    torch.manual_seed(0)
    limit_before = sys.getrecursionlimit()
    log = tl.trace(_Chain(), (torch.randn(4), N_CHAIN))
    try:
        yield log, limit_before, sys.getrecursionlimit()
    finally:
        log.cleanup()


def _mult_labels(log) -> list[str]:
    return [layer.label for layer in log.layer_list if layer.label.startswith("mul")]


def test_giant_equivalence_class_is_exact(wide_capture) -> None:
    """All 300 streams land in ONE equivalence class with no member missing."""

    log, _ = wide_capture
    mults = _mult_labels(log)
    assert len(mults) == N_STREAMS
    assert len(log.layer_list) == N_STREAMS + 5  # input, linear, stack, sum, output
    members = log[mults[0]].equivalent_ops
    assert len(members) == N_STREAMS
    assert all(label in members for label in mults)


def test_giant_equivalence_view_is_one_shared_object(wide_capture) -> None:
    """Every member reads THE one cached immutable view — O(1), identity-stable."""

    log, _ = wide_capture
    mults = _mult_labels(log)
    view_ids = {id(log[label].equivalent_ops) for label in mults}
    assert len(view_ids) == 1
    assert log[mults[0]].equivalent_ops is log[mults[-1]].equivalent_ops


def test_giant_cohort_groups_silently_below_real_ceiling(wide_capture) -> None:
    """At 300 members (~45k pairs) the UNPATCHED sweep finishes with no disclosure.

    A pair-probe-ceiling or fixpoint-budget warning here means the real
    constants degraded on an in-band size — a scaling regression, not tuning.
    The mutually-unreachable streams must also stay ungrouped: equivalence
    never silently promotes to recurrence at scale.
    """

    log, caught = wide_capture
    scaling_warnings = [
        w
        for w in caught
        if "pair-probe ceiling" in str(w.message)
        or "refinement" in str(w.message)
        or "recorded" in str(w.message)
    ]
    assert scaling_warnings == []
    mults = _mult_labels(log)
    assert len(log[mults[0]].recurrent_ops) == 1
    assert len(log[mults[-1]].recurrent_ops) == 1


def test_deep_recurrence_rolls_to_one_layer_per_site(deep_capture) -> None:
    """300 calls of one cell group into single layers with exactly 300 passes."""

    log = deep_capture
    assert len(log.layer_list) == 2 * N_PASSES + 2  # linear+tanh per step, input, output
    for label in ("linear_1_1", "tanh_1_2"):
        layer = log[label]
        assert layer.num_passes == N_PASSES, label


def test_deep_recurrence_pass_addressing_at_the_extremes(deep_capture) -> None:
    """Pass-qualified addressing stays exact at pass 1 and pass 300."""

    log = deep_capture
    first = log[f"linear_1_1:{1}"]
    last = log[f"linear_1_1:{N_PASSES}"]
    assert first.pass_index == 1
    assert last.pass_index == N_PASSES
    assert first.num_passes == N_PASSES
    assert last.num_passes == N_PASSES


def test_deep_recurrence_group_view_shared_across_all_passes(deep_capture) -> None:
    """All 300 passes read the ONE cached recurrent_ops view (O(1) live view)."""

    log = deep_capture
    views = {id(log[f"linear_1_1:{p}"].recurrent_ops) for p in range(1, N_PASSES + 1)}
    assert len(views) == 1
    assert len(log["linear_1_1:1"].recurrent_ops) == N_PASSES


def test_deep_recurrence_site_key_stable_across_passes(deep_capture) -> None:
    """Every pass carries the same structural site key; the Layer serves it."""

    log = deep_capture
    keys = {log[f"linear_1_1:{p}"].site_key for p in range(1, N_PASSES + 1)}
    assert len(keys) == 1
    assert log["linear_1_1"].site_key == keys.pop()


@pytest.mark.heavy
def test_graph_diameter_capture_is_iterative(chain_capture) -> None:
    """A 1400-op diameter completes under the ambient limit, which stays put.

    The pytest session runs at CPython's default 1000-frame limit; any
    recursive per-op traversal in postprocess would overflow here. The
    before/after equality also trips a "fix" that bumps the global limit.
    """

    log, limit_before, limit_after = chain_capture
    assert limit_before == limit_after
    assert len(log.layer_list) == N_CHAIN + 2


@pytest.mark.heavy
def test_graph_diameter_relations_exact_at_midpoint(chain_capture) -> None:
    """Parent/child tuples at op 700 of 1400 are exactly the neighbors."""

    log, _, _ = chain_capture
    mid = log["add_700_700:1"]
    assert mid.parents == ("add_699_699",)
    assert mid.children == ("add_701_701",)


@pytest.mark.heavy
def test_graph_diameter_closures_exact_at_endpoints(chain_capture) -> None:
    """Input/output closures stay exact across the full 1400-op span."""

    log, _, _ = chain_capture
    first = log["add_1_1:1"]
    last = log[f"add_{N_CHAIN}_{N_CHAIN}:1"]
    assert first.output_descendants == frozenset({"output_1"})
    assert last.input_ancestors == frozenset({"input_1"})
    assert log["output_1:1"].parents == (f"add_{N_CHAIN}_{N_CHAIN}",)


def _stack_depth() -> int:
    """Return the current Python stack depth, frames from here to the root."""

    depth = 0
    frame = sys._getframe()
    while frame is not None:
        depth += 1
        frame = frame.f_back
    return depth


#: Safety margin ADDED to the measured minimal-working untraced recursion limit
#: (see ``_minimal_recursion_limit`` below) to pick the gate's pinned test limit.
#: TorchLens adds one ``decorated_forward`` frame per module call, so capturing
#: ``NEST_DEPTH_OVERFLOW`` nested modules needs roughly ``NEST_DEPTH_OVERFLOW``
#: more frames than running them untraced -- a window of several hundred frames
#: at the current ``NEST_DEPTH_OVERFLOW == 400``. This margin must stay small
#: relative to that window so the chosen limit always lands strictly between
#: the untraced and captured requirements; it must NOT be a fixed ABSOLUTE
#: frame count for either requirement, because those shift with interpreter
#: internals (grind-pyver R9: Python 3.12 shrank the untraced requirement from
#: 1608 to 1210 measured frames at this depth, a fixed 1800-frame headroom off
#: the old calibration then exceeded the new ~1666-frame captured requirement
#: too, and the gate stopped raising RecursionError at all). Measuring the
#: untraced minimum live, every run, is interpreter-version-agnostic by
#: construction.
NEST_RECURSION_SAFETY_MARGIN = 150


def _minimal_recursion_limit(fn: Callable[[], None], lo: int, hi: int) -> int:
    """Binary-search the minimal ``sys.setrecursionlimit()`` value at which
    ``fn()`` completes without ``RecursionError``.

    ``hi`` must already be a known-working ceiling (asserted here so a too-low
    caller-supplied bracket fails loudly instead of silently returning a wrong
    answer). CPython additionally refuses to LOWER the limit below the
    caller's current actual stack depth, itself raising ``RecursionError`` at
    the ``sys.setrecursionlimit()`` call rather than at ``fn()`` -- both cases
    mean "this limit does not work" and are treated identically here.
    """

    original_limit = sys.getrecursionlimit()

    def _works(limit: int) -> bool:
        try:
            sys.setrecursionlimit(limit)
            fn()
            return True
        except RecursionError:
            return False
        finally:
            sys.setrecursionlimit(original_limit)

    assert _works(hi), f"search ceiling {hi} does not work; widen the bracket"
    while lo < hi:
        mid = (lo + hi) // 2
        if _works(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def test_depth_overflow_fails_clean_and_restores() -> None:
    """Nesting past the recursion limit fails typed and leaves torch capturable.

    TorchLens adds one decorated_forward frame per module call, so this model is
    legal untraced and overflows only under capture. The contract: the original
    RecursionError propagates, the typed CaptureAttemptFailedWarning discloses
    the restore, and an immediate follow-up capture is fully sane.

    The limit is MEASURED rather than inherited or pinned to a fixed absolute
    frame count, and the untraced premise is ASSERTED, because relying on the
    ambient limit made this gate both fragile and tautological (2026-08-19):

    * At the default 1000 the untraced forward ALSO overflows, so the test
      passed while proving only "400 nesting levels overflow", never "capture
      is what tips it".
    * ``jedi/api/__init__.py`` calls ``sys.setrecursionlimit(3000)`` at import
      time and never restores it. Any earlier test importing IPython (jedi is a
      dependency) silently raised the ceiling above the captured requirement,
      so the gate could not fire at all -- it passed alone and failed in a
      full session, which is the wrong way round for an adversarial gate.

    A fixed PINNED absolute frame count has the same failure mode one level up
    (grind-pyver R9): it is calibrated against whichever interpreter happened
    to run the calibration, and a later interpreter that needs a different
    number of frames per call silently drifts the gate out of its window
    (Python 3.12 did exactly this). Measuring the untraced minimum fresh, every
    run, is immune to that by construction.
    """

    model = _nested_sequential(NEST_DEPTH_OVERFLOW)
    inputs = torch.randn(1, 4)
    original_limit = sys.getrecursionlimit()
    search_floor = _stack_depth() + 50
    search_ceiling = _stack_depth() + 6000
    try:
        untraced_min = _minimal_recursion_limit(lambda: model(inputs), search_floor, search_ceiling)
        sys.setrecursionlimit(untraced_min + NEST_RECURSION_SAFETY_MARGIN)
        # The premise, asserted: without capture this model runs fine at this
        # limit. If this ever raises, the gate below proves nothing.
        model(inputs)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(RecursionError):
                tl.trace(model, inputs)
    finally:
        sys.setrecursionlimit(original_limit)
    assert any(isinstance(w.message, CaptureAttemptFailedWarning) for w in caught), (
        "expected the capture-failed restore disclosure"
    )
    follow_up = tl.trace(nn.Linear(2, 2), torch.randn(1, 2))
    assert [layer.label for layer in follow_up.layer_list] == [
        "input_1:1",
        "linear_1_1:1",
        "output_1:1",
    ]


def test_wide_fanout_and_fanin_relations_exact() -> None:
    """300-way fan-out children and 300-parent fan-in stay exact and ordered."""

    torch.manual_seed(0)
    log = tl.trace(_FanOut(), (torch.randn(4), N_FANOUT))
    mults = _mult_labels(log)
    assert len(mults) == N_FANOUT
    bare_mults = [label.split(":")[0] for label in mults]
    input_children = log["input_1:1"].children
    assert len(input_children) == N_FANOUT
    assert set(input_children) == set(bare_mults)
    stack_labels = [layer.label for layer in log.layer_list if layer.label.startswith("stack")]
    assert len(stack_labels) == 1
    stack_parents = log[stack_labels[0]].parents
    assert list(stack_parents) == bare_mults  # arg order preserved
