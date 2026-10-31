"""Parameter-check registrations and finding builders (checks memo item 4).

The change check ships as a FACT vocabulary whose primary detector is the
GRADIENT FACT (``grad is None`` / ``grad_norm == 0``), never a movement
statistic: under torch's default AdamW a DEAD network "changes" every step
(decoupled weight decay is a rescale applied regardless of the gradient) and
the decay-band rescue lags a real death by 22 to >32 steps of Adam momentum
bleed-out -- both measured (memo D2). Movement statistics are recorded as
corroborating facts carrying their measured latency; no finding ever
contains the word "learning" or "dead" (``tl.dead`` keeps the only
dead-unit verdict, memo D14).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ._constants import WARN_WINDOW_DEFAULT
from ._errors import CheckConfigError
from ._records import ACTIONS, CheckFinding

__tl_layer__ = "L5"

#: Follow-up spelling every gradient-cone finding must point at (memo D3):
#: the finding names a cone; the repaired frontier names the culprit.
FRONTIER_FOLLOW_UP = "tl.debug.gradient_flow_audit(trace)  # module + is_frontier columns"


def validate_action(action: str) -> str:
    """Validate one action token, refusing typed on violation."""

    if action not in ACTIONS:
        raise CheckConfigError(
            f"action={action!r} is not one of {ACTIONS}. Severity describes "
            "evidence; action describes control flow, overridable per "
            "registration (memo D16).",
            code="check_vocab_invalid",
            field="action",
            value=action,
            remedy=f"Use one of {ACTIONS}.",
        )
    return action


def validate_window(window: tuple[int, int] | None) -> tuple[int, int]:
    """Validate an M-of-N warn window, refusing typed on violation."""

    if window is None:
        return WARN_WINDOW_DEFAULT
    try:
        m, n = int(window[0]), int(window[1])
    except (TypeError, ValueError, IndexError) as exc:
        raise CheckConfigError(
            f"window must be an (m, n) pair, got {window!r}.",
            code="check_window_invalid",
            remedy="Pass window=(5, 8)-style integers with 1 <= m <= n.",
        ) from exc
    if not 1 <= m <= n:
        raise CheckConfigError(
            f"window=({m}, {n}) needs 1 <= m <= n: warns fire only on "
            "sustained evidence, never a single observation (memo 4.5).",
            code="check_window_invalid",
            remedy="Pass window=(m, n) with 1 <= m <= n.",
        )
    return (m, n)


class MofNWindow:
    """One M-of-N sustained-evidence window with dedup and re-arm.

    The registry gates emission itself: one warn per (check, site), re-armed
    only after a FULL healthy window, with a warmup grace of one window
    (memo D16) -- the flood is impossible by construction, not by Python's
    warning filters.
    """

    def __init__(self, m: int, n: int) -> None:
        self.m = m
        self.n = n
        self._observations: deque[bool] = deque(maxlen=n)
        self._warned = False

    def observe(self, bad: bool) -> bool:
        """Record one observation; return True exactly when a warn fires."""

        self._observations.append(bad)
        if len(self._observations) < self.n:
            return False  # warmup grace of one full window
        bad_count = sum(self._observations)
        if self._warned:
            if bad_count == 0:
                self._warned = False  # re-arm only after a fully healthy window
            return False
        if bad_count >= self.m:
            self._warned = True
            return True
        return False

    @property
    def warned(self) -> bool:
        """Whether this window is currently in its warned (deduped) state."""

        return self._warned


@dataclass
class ChangeCheck:
    """The params-change fact vocabulary registration (memo D2/D3)."""

    within: tuple[str, ...] | None
    window: tuple[int, int]
    action: str
    windows: dict[str, MofNWindow] = field(default_factory=dict)

    def window_for(self, name: str) -> MofNWindow:
        """Return (creating on first use) the per-parameter window."""

        if name not in self.windows:
            self.windows[name] = MofNWindow(*self.window)
        return self.windows[name]


@dataclass
class FrozenCheck:
    """The declared-frozen invariant registration (memo D5).

    ``evidence="clone"`` is the default (exact, can SHOW the offending
    delta); ``evidence="digest"`` is the labeled opt-in whose findings
    disclose probabilistic evidence and may NEVER emit an exact pass.
    """

    names: tuple[str, ...]
    evidence: str
    action: str
    baselines: dict[str, object] = field(default_factory=dict)


@dataclass
class UpdateRatioCheck:
    """The Karpathy update-ratio registration (memo D13).

    Reports the RAW ratio and the lr-normalized companion; NO default band
    ships before the calibration matrix (a single healthy model measured a
    38x spread across parameters) -- bounds are explicit per registration.
    """

    within: tuple[str, ...] | None
    bounds: tuple[float | None, float | None] | None
    action: str
    windows: dict[str, MofNWindow] = field(default_factory=dict)

    def window_for(self, name: str, window: tuple[int, int]) -> MofNWindow:
        """Return (creating on first use) the per-parameter window."""

        if name not in self.windows:
            self.windows[name] = MofNWindow(*window)
        return self.windows[name]


@dataclass
class NonfiniteGradientCheck:
    """Site-conditional live parameter-gradient nonfinite check (memo D6)."""

    action: str


@dataclass
class NonfiniteParamCheck:
    """Scheduled parameter/buffer-space nonfinite scan (memo D7)."""

    every: int
    include_buffers: bool
    action: str


@dataclass
class MagnitudeCheck:
    """The S-A pre-clip magnitude registration (memo D8/D9).

    Registering it IS the explicit arming: the pass stays OFF by default
    until the canonical gate set (idle-box + CUDA + multi-rank DDP +
    at-scale compile) passes -- panel law: no default without the number.
    """

    vanishing_threshold: float
    exploding_threshold: float
    action: str


def no_gradient_finding(  # noqa: PLR0913 -- the args ARE the finding's evidence fields (D2); a bundling object would duplicate CheckFinding
    name: str,
    *,
    kind: str,
    window: tuple[int, int],
    accepted_step_id: int,
    global_step: int | None,
    step_provenance: str,
    action: str,
) -> CheckFinding:
    """Build the primary gradient-fact finding (memo D2).

    ``kind`` distinguishes ``grad is None`` (disconnection, also the
    reentrant-checkpoint detached-input footgun) from an exact zero tensor;
    the two are different evidence (memo test row 1).
    """

    return CheckFinding(
        check="param_received_no_gradient",
        code="param_received_no_gradient",
        severity="warning",
        action=action,
        message=(
            f"{name} received no gradient ({kind}) in at least "
            f"{window[0]} of the last {window[1]} accepted steps. This is the "
            "zero-latency detector; movement statistics lag a real death by "
            "22 to >32 steps of optimizer momentum (measured). Legitimate "
            "causes exist (MoE routing, untaken branches, embedding rows "
            "outside the batch) -- the window is why this warns, not raises."
        ),
        names=(name,),
        accepted_step_id=accepted_step_id,
        global_step=global_step,
        step_provenance=step_provenance,
        stage="post_clip_applied",
        scale_provenance="unscaled",
        evidence="step_hook",
        values={"window_m": float(window[0]), "window_n": float(window[1])},
        remedy=(
            "Check requires_grad, optimizer membership (report.disclosures), and "
            "the graph connection at the named parameter"
        ),
        follow_up=FRONTIER_FOLLOW_UP,
    )


def decay_only_movement_fact(  # noqa: PLR0913 -- the args ARE the corroborating fact's evidence fields (D2); a bundling object would duplicate CheckFinding
    name: str,
    *,
    ratio: float,
    lr: float | None,
    weight_decay: float | None,
    beta1: float | None,
    beta2: float | None,
    accepted_step_id: int,
) -> CheckFinding:
    """Build the corroborating decay-band movement fact (memo D2).

    A CORROBORATING fact, never a detector: it carries its measured latency
    (momentum bleed-out goldens 22 / >32 / >32 steps) and the adapter's
    betas so the predicted lag horizon can be rendered later ([UI-SPRINT]).
    """

    values: dict[str, float | None] = {
        "update_ratio": ratio,
        "lr": lr,
        "weight_decay": weight_decay,
        "beta1": beta1,
        "beta2": beta2,
    }
    return CheckFinding(
        check="param_movement_decay_band",
        code="param_movement_decay_band",
        severity="info",
        action="collect",
        message=(
            f"{name} moved by approximately its decoupled weight-decay rescale "
            "alone this step -- consistent with (not proof of) a parameter "
            "receiving no useful gradient. Corroborating fact only: this "
            "statistic lags a real death by 22 to >32 steps (measured); the "
            "gradient fact is the primary detector."
        ),
        names=(name,),
        accepted_step_id=accepted_step_id,
        evidence="clone",
        values=values,
        remedy="Read param_received_no_gradient for the zero-latency evidence",
        follow_up=FRONTIER_FOLLOW_UP,
    )


__all__ = [
    "FRONTIER_FOLLOW_UP",
    "ChangeCheck",
    "FrozenCheck",
    "MagnitudeCheck",
    "MofNWindow",
    "NonfiniteGradientCheck",
    "NonfiniteParamCheck",
    "UpdateRatioCheck",
    "decay_only_movement_fact",
    "no_gradient_finding",
    "validate_action",
    "validate_window",
]
