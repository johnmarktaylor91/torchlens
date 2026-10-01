"""Deterministic collapse work estimator and admission policy (memo D5/F5).

Every spelling here is DOCUMENTED-UNSTABLE pending the naming sprint.

The compute gate for smart collapse used to be one raw-op-count ceiling
(``COLLAPSE_OPTIMIZER_MAX_OPS`` over ``len(trace.ops)``), which fails in both
directions on real models (vit_l_16: 852 ops, 4.1 s; maxvit_t: 1,307 ops,
0.48 s) and is not even mode-stable (train-mode densenet201 was REFUSED while
eval mode was admitted at the same rendered universe). This module implements
the panel's replacement (collapse memo D5):

- the gate variable becomes ``U`` — the rendered-universe node count of the
  FULL plan (what would actually be drawn after focus/depth/rolled
  reductions), so ``module=`` focus, ``vis_call_depth``, and rolled mode are
  real remedies for the first time;
- a measured two-variable work estimator ``0.0095 * U**1.43 * W**1.02``
  (held-out-validated within 2.05x on 7 unseen checkpoints; W is the widest
  rendered sibling-group width) tiers requests deterministically — never a
  function of machine speed;
- frontier records become the ALLOCATION cap they actually measure (memory
  defence, not a clock);
- a generous wall-clock watchdog may abandon the quality planner TO THE
  DETERMINISTIC FALLBACK only (never a partial best-so-far frontier), is OFF
  in CI/deterministic mode, and every firing is disclosed as an estimator
  bug with ``(U, W, predicted_ms, actual_ms)``;
- admission numbers are published only for the measured envelope
  (``U <= ~1,500``); beyond it the fitted power law is an extrapolation that
  over-predicts (wrongly conservative, never a hang — measured 17.6x at
  18,644 ops), and the large-U refit is a named calibration item.

Over-budget requests degrade to the deterministic linear fallback planner
(:mod:`.collapse_fallback`), NEVER to an uncollapsed wall. The fallback is
built from deterministic inputs only (trace revision + context), never from
partial optimizer state, which is the property "always precomputed" bought
in the memo's design.
"""

from __future__ import annotations

import os
import time
from contextvars import ContextVar
from dataclasses import dataclass, field

#: Version tag stamped on estimator diagnostics so field refits are
#: attributable to the formula that produced each record (memo item 14).
ESTIMATOR_FORMULA_VERSION = "uw_power_v1"

#: Fitted coefficients of the measured work model (collapse memo D5(ii)):
#: ``predicted_ms = COEFF * U**U_EXP * W**W_EXP``, fitted at max-mode on the
#: 13-model corpus and held-out-validated within 2.05x on 7 unseen
#: checkpoints. The W exponent is the B2 de-quadratic proof handle: after
#: item 7 the refit must drive it toward zero.
ESTIMATOR_COEFFICIENT = 0.0095
ESTIMATOR_U_EXPONENT = 1.43
ESTIMATOR_W_EXPONENT = 1.02

#: Predicted-cost admission budget for the quality (frontier) planner, in
#: estimator milliseconds. The 2-3x measurement margin of D5(ii) is baked
#: into the constant: densenet201 (U=713, W=48) predicts ~5.9 s and must
#: stay admitted; gpt2-large bare cache=True (U~1,669, W~81) predicts ~34 s
#: and degrades to the deterministic fallback; Qwen3.5-4B predicts ~475 s.
QUALITY_PLANNER_BUDGET_MS = 30_000.0

#: Measured envelope bound for published admission numbers (memo D5(ii)):
#: above this U the power law is extrapolation and only its conservative
#: direction is trusted.
MEASURED_ENVELOPE_MAX_U = 1_500

#: Pathological-input pre-gate multiplier over the retained defensive
#: constant (memo D5(i)): traces above ``20 x COLLAPSE_OPTIMIZER_MAX_OPS``
#: raw ops decline smart collapse outright before any universe build.
PATHOLOGICAL_OP_MULTIPLIER = 20

#: Frontier-record allocation cap (memo D5(iii)): the budget unit Sol
#: proposed as a clock is measured to be an ALLOCATION meter (98x per-record
#: cost spread, tiering backwards on exactly the two models the ceiling
#: exists for), so it survives as memory defence only. The constant is
#: generous: maxvit_t, the measured record-heavy admitted model, allocates
#: 8,788 records.
FRONTIER_ALLOCATION_CAP = 250_000

#: Watchdog slack multiplier over the predicted cost, and its floor. The
#: watchdog is subordinate and generous by design: it exists to catch
#: estimator bugs, not to tune performance.
WATCHDOG_SLACK = 8.0
WATCHDOG_FLOOR_MS = 15_000.0


def predicted_select_ms(universe_count: int, max_sibling_width: int) -> float:
    """Return the estimator's predicted quality-planner cost in ms.

    Parameters
    ----------
    universe_count:
        ``U`` — rendered-universe node count of the full plan.
    max_sibling_width:
        ``W`` — widest rendered sibling group (direct child units plus
        parent-owned rendered units of the widest parent).
    """

    safe_u = max(int(universe_count), 1)
    safe_w = max(int(max_sibling_width), 1)
    return ESTIMATOR_COEFFICIENT * safe_u**ESTIMATOR_U_EXPONENT * safe_w**ESTIMATOR_W_EXPONENT


def watchdog_enabled() -> bool:
    """Return whether the wall-clock watchdog may arm for this process.

    OFF in CI and in explicitly requested deterministic runs (memo D5(v)):
    a wall clock must never change results where reproducibility is the
    contract, so both ``CI`` and ``TORCHLENS_DETERMINISTIC`` disable it, as
    does ``TORCHLENS_COLLAPSE_WATCHDOG=0``.
    """

    from ..utils.env_flags import closed_bool_env

    if os.environ.get("CI"):
        return False
    if closed_bool_env("TORCHLENS_DETERMINISTIC"):
        return False
    return closed_bool_env("TORCHLENS_COLLAPSE_WATCHDOG", default=True)


class FallbackDegrade(Exception):
    """Internal control-flow signal: abandon the quality planner.

    Raised by budget checkpoints (allocation cap, watchdog) inside the
    frontier selection; caught at the selection entry, which degrades to the
    deterministic fallback planner. Never user-visible.
    """

    def __init__(self, cause: str) -> None:
        """Record the degrade cause (``"watchdog"`` or ``"allocation"``)."""

        super().__init__(cause)
        self.cause = cause


@dataclass
class SelectionBudget:
    """Mutable per-selection budget state (allocation meter + watchdog).

    One instance is installed in :data:`ACTIVE_BUDGET` for the duration of a
    quality-planner run; the hot frontier chokepoints call :meth:`charge`,
    which trips :class:`FallbackDegrade` when a bound is exceeded.
    """

    predicted_ms: float
    universe_count: int
    max_sibling_width: int
    #: perf_counter at admission, so degrade paths compute actual_ms without
    #: threading a separate wall-start argument.
    started_at: float = 0.0
    deadline: float | None = None
    allocated_records: int = 0
    allocation_cap: int = FRONTIER_ALLOCATION_CAP
    fired: str | None = None
    #: Peak allocation observed (calibration instrument, memo item 14:
    #: FRONTIER_CAP saturation was never instrumented).
    peak_allocated: int = field(default=0)

    def charge(self, records: int) -> None:
        """Charge ``records`` frontier allocations and run the checkpoints."""

        self.allocated_records += records
        if self.allocated_records > self.peak_allocated:
            self.peak_allocated = self.allocated_records
        if self.allocated_records > self.allocation_cap:
            self.fired = "allocation"
            raise FallbackDegrade("allocation")
        if self.deadline is not None and time.perf_counter() > self.deadline:
            self.fired = "watchdog"
            raise FallbackDegrade("watchdog")


#: The active selection budget, or ``None`` outside a quality-planner run.
#: A context variable (not optimizer-state plumbing) because the repo is
#: single-threaded by design and the frontier chokepoints are pure helpers.
ACTIVE_BUDGET: ContextVar[SelectionBudget | None] = ContextVar(
    "torchlens_collapse_selection_budget", default=None
)


def charge_frontier_allocation(records: int) -> None:
    """Charge the active selection budget, if one is installed."""

    budget = ACTIVE_BUDGET.get()
    if budget is not None:
        budget.charge(records)


@dataclass(frozen=True)
class EstimatorDiagnostics:
    """Per-plan estimator disclosure for field refits (memo item 14).

    Stored on ``OptimizerResult.estimator`` so every plan carries
    ``(U, W, predicted_ms, actual_ms, tier, formula version)`` — the exact
    tuple the memo requires for refitting the power law from field data.
    """

    universe_count: int
    max_sibling_width: int
    predicted_ms: float
    actual_ms: float
    tier: str
    formula_version: str = ESTIMATOR_FORMULA_VERSION
    peak_frontier_records: int = 0
    budget_fired: str | None = None
