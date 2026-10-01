"""Skip/scale, clip, and watchdog ledgers (checks memo D10 / D11 / D12).

The AMP skip ledger counts GradScaler scale DECREMENTS -- ``update()``
multiplies by ``backoff_factor`` exactly once per skipped attempt, so
reading ``get_scale()`` once per backward yields the exact skip and attempt
counts with ZERO loop edits (memo D10; E31 measured 4/4 configurations).
Fire-count subtraction is WRONG under gradient accumulation by construction
(19 reported vs 1 true skip at K=4) and is locked out by test so it cannot
return as an optimization.

Preconditions ship as first-class disclosures, never assumptions: a
``scaler=`` handle, canonical ``update()`` cadence, and static accumulation.
Hook-only evidence labels attempt GROUPING ``inferred``; caller
``global_step`` and exact grouping remain the explicit boundary's unique
property (memo D18, Sol's standing distinction in DR-4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__tl_layer__ = "L5"

#: Relative tolerance for the tier-(c) post-clip constancy signature and the
#: tier-(b) saturation comparison (memo D12: a hint, never a pass).
_CLIP_CONSTANCY_RTOL = 1e-6

#: Window length of accepted steps the constancy signature needs before the
#: INFO hint fires.
_CLIP_CONSTANCY_WINDOW = 6


@dataclass(frozen=True)
class ScaleLedgerSnapshot:
    """Frozen skip/scale ledger facts for reports and events."""

    scaler_present: bool
    enabled: bool
    backward_reads: int
    current_scale: float | None
    initial_scale: float | None
    skipped_attempts: int
    growth_events: int
    accepted_steps: int
    attempts: int
    inferred_k: float | None
    preconditions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable ledger payload."""

        return {
            "scaler_present": self.scaler_present,
            "enabled": self.enabled,
            "backward_reads": self.backward_reads,
            "current_scale": self.current_scale,
            "initial_scale": self.initial_scale,
            "skipped_attempts": self.skipped_attempts,
            "growth_events": self.growth_events,
            "accepted_steps": self.accepted_steps,
            "attempts": self.attempts,
            "inferred_k": self.inferred_k,
            "preconditions": list(self.preconditions),
        }


class ScaleLedger:
    """The decrement-counting AMP skip ledger (memo D10).

    ``read()`` is called once per observed backward (the S-A counting tier
    supplies the cadence); ``accepted()`` once per optimizer post-step. The
    ledger performs no device work: ``GradScaler.get_scale()`` reads host
    state.
    """

    def __init__(self, scaler: Any | None) -> None:
        self._scaler = scaler
        self._last_scale: float | None = None
        self._initial_scale: float | None = None
        self.backward_reads = 0
        self.skipped_attempts = 0
        self.growth_events = 0
        self.accepted_steps = 0
        self._dynamic_k_disclosed = False

    @property
    def enabled(self) -> bool:
        """Whether a live, enabled GradScaler is attached."""

        return self._scaler is not None and bool(getattr(self._scaler, "is_enabled", bool)())

    def read(self) -> None:
        """Record one backward's scale reading; count decrements exactly."""

        self.backward_reads += 1
        self.observe_scale()

    def observe_scale(self) -> None:
        """Fold any unobserved scale movement in WITHOUT counting a backward.

        The decrement lands on the NEXT read after ``update()``; a skip on
        the final attempt of a run would otherwise stay invisible, so the
        report snapshot calls this once more. Idempotent between scale
        changes.
        """

        scaler = self._scaler
        if not self.enabled or scaler is None:
            return
        scale = float(scaler.get_scale())
        if self._initial_scale is None:
            self._initial_scale = scale
        last = self._last_scale
        self._last_scale = scale
        if last is None or scale == last:
            return
        if scale < last:
            backoff = float(getattr(self._scaler, "get_backoff_factor", lambda: 0.5)())
            self.skipped_attempts += _exact_event_count(scale, last, backoff)
        else:
            growth = float(getattr(self._scaler, "get_growth_factor", lambda: 2.0)())
            self.growth_events += _exact_event_count(scale, last, growth)

    def accepted(self) -> None:
        """Record one accepted optimizer step."""

        self.accepted_steps += 1

    def snapshot(self) -> ScaleLedgerSnapshot:
        """Freeze the ledger with its precondition disclosures."""

        self.observe_scale()
        attempts = self.accepted_steps + self.skipped_attempts
        inferred_k: float | None = None
        if attempts > 0 and self.backward_reads > 0:
            inferred_k = self.backward_reads / attempts
        preconditions = [
            "scale decrements are read once per observed backward",
            "canonical GradScaler.update() cadence (once per attempt) assumed",
        ]
        if inferred_k is not None and abs(inferred_k - round(inferred_k)) > 1e-9:
            preconditions.append(
                "accumulation factor K is not an exact integer over this window: "
                "dynamic K is ambiguous in hook-only mode; use the explicit step "
                "boundary for exact attempt grouping"
            )
        return ScaleLedgerSnapshot(
            scaler_present=self._scaler is not None,
            enabled=self.enabled,
            backward_reads=self.backward_reads,
            current_scale=self._last_scale,
            initial_scale=self._initial_scale,
            skipped_attempts=self.skipped_attempts,
            growth_events=self.growth_events,
            accepted_steps=self.accepted_steps,
            attempts=self.accepted_steps + self.skipped_attempts,
            inferred_k=inferred_k,
            preconditions=tuple(preconditions),
        )


def _exact_event_count(scale: float, last: float, factor: float) -> int:
    """Count how many multiplications by ``factor`` map ``last`` to ``scale``.

    Consecutive skipped attempts between two reads appear as one combined
    ratio (65536 -> 8192 is three backoffs at 0.5); the count recovers each
    event exactly because GradScaler scales move only by integer powers of
    its factors.
    """

    if factor <= 0 or factor == 1.0 or last <= 0 or scale <= 0:
        return 1
    return max(1, round(math.log(scale / last) / math.log(factor)))


@dataclass(frozen=True)
class WatchdogSnapshot:
    """Frozen death-spiral watchdog state (memo D11)."""

    backwards_since_accepted: int
    threshold: int
    tripped: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable watchdog payload."""

        return {
            "backwards_since_accepted": self.backwards_since_accepted,
            "threshold": self.threshold,
            "tripped": self.tripped,
        }


class Watchdog:
    """Death-spiral watchdog: warns when no step is ever accepted (D11).

    S-A fires on skipped attempts and S-B does not, so "N backwards since
    the last accepted step exceeding K x factor" detects scale collapse even
    when every retrospective method stays silent forever.
    """

    def __init__(self, accumulation_steps: int | None, factor: int) -> None:
        self._k = max(1, accumulation_steps or 1)
        self._factor = factor
        self.backwards_since_accepted = 0
        self._warned = False

    @property
    def threshold(self) -> int:
        """Backwards without an accepted step that trip the watchdog."""

        return self._k * self._factor

    def on_backward(self) -> bool:
        """Record one backward; return True exactly once when tripping."""

        self.backwards_since_accepted += 1
        if not self._warned and self.backwards_since_accepted > self.threshold:
            self._warned = True
            return True
        return False

    def on_accepted(self) -> None:
        """Reset on an accepted step; re-arm the single warning."""

        self.backwards_since_accepted = 0
        self._warned = False

    def snapshot(self) -> WatchdogSnapshot:
        """Freeze the watchdog state for reports and events."""

        return WatchdogSnapshot(
            backwards_since_accepted=self.backwards_since_accepted,
            threshold=self.threshold,
            tripped=self._warned,
        )


@dataclass(frozen=True)
class ClipLedgerSnapshot:
    """Frozen clip-saturation facts (memo D12), tiered and honest."""

    tier: str
    declared_clip_norm: float | None
    accepted_steps_observed: int
    saturated_steps: int
    saturation_rate: float | None
    applied_clip_factors: tuple[float, ...]
    constancy_hint: bool
    constant_total: float | None
    nonfinite_totals: int

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable clip payload."""

        return {
            "tier": self.tier,
            "declared_clip_norm": self.declared_clip_norm,
            "accepted_steps_observed": self.accepted_steps_observed,
            "saturated_steps": self.saturated_steps,
            "saturation_rate": self.saturation_rate,
            "applied_clip_factors": list(self.applied_clip_factors),
            "constancy_hint": self.constancy_hint,
            "constant_total": self.constant_total,
            "nonfinite_totals": self.nonfinite_totals,
        }


class ClipLedger:
    """Clip-saturation evidence in three tiers, never a silent guess (D12).

    (a) with the magnitude tier armed, the exact applied-clip-factor series
    (pre-clip vs post-clip totals); (b) with ``clip_norm=`` declared, the
    exact saturation rate plus the censorship finding armed; (c) neither:
    the constancy signature (post-clip total invariant to ~1e-6 across a
    window) fires an INFO-level hint naming ``clip_norm=`` -- a hint, never
    a pass or a warning. A nonfinite total records the collateral-damage
    signal, not a meaningless clip ratio.
    """

    def __init__(self, clip_norm: float | None) -> None:
        self.clip_norm = clip_norm
        self._post_totals: list[float] = []
        self._factors: list[float] = []
        self.saturated_steps = 0
        self.nonfinite_totals = 0

    def observe_step(self, post_clip_total: float, pre_clip_total: float | None) -> None:
        """Record one accepted step's total norms (post-clip; pre when armed)."""

        if not math.isfinite(post_clip_total):
            self.nonfinite_totals += 1
            return
        self._post_totals.append(post_clip_total)
        if pre_clip_total is not None and pre_clip_total > 0:
            self._factors.append(post_clip_total / pre_clip_total)
        if self.clip_norm is not None and post_clip_total >= self.clip_norm * (
            1 - _CLIP_CONSTANCY_RTOL
        ):
            self.saturated_steps += 1

    def _constancy(self) -> tuple[bool, float | None]:
        """Detect the tier-(c) constant post-clip total signature."""

        window = self._post_totals[-_CLIP_CONSTANCY_WINDOW:]
        if len(window) < _CLIP_CONSTANCY_WINDOW:
            return False, None
        center = window[0]
        if center == 0:
            return False, None
        constant = all(
            abs(total - center) <= abs(center) * _CLIP_CONSTANCY_RTOL for total in window
        )
        return constant, (center if constant else None)

    def snapshot(self) -> ClipLedgerSnapshot:
        """Freeze the tiered clip evidence."""

        if self._factors:
            tier = "a"
        elif self.clip_norm is not None:
            tier = "b"
        else:
            tier = "c"
        constancy_hint, constant_total = (False, None) if tier != "c" else self._constancy()
        observed = len(self._post_totals)
        return ClipLedgerSnapshot(
            tier=tier,
            declared_clip_norm=self.clip_norm,
            accepted_steps_observed=observed,
            saturated_steps=self.saturated_steps,
            saturation_rate=(
                self.saturated_steps / observed if self.clip_norm is not None and observed else None
            ),
            applied_clip_factors=tuple(self._factors),
            constancy_hint=constancy_hint,
            constant_total=constant_total,
            nonfinite_totals=self.nonfinite_totals,
        )


__all__ = [
    "ClipLedger",
    "ClipLedgerSnapshot",
    "ScaleLedger",
    "ScaleLedgerSnapshot",
    "Watchdog",
    "WatchdogSnapshot",
]
