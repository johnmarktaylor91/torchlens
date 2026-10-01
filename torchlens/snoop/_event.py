"""The frozen ``NarrationEvent`` record behind every narration line (snoop D2).

Narration is data first: every rendered line is backed by one immutable,
backend-neutral record. The record -- not the text -- is the parser channel,
which is what lets the line grammar optimize for the human scanning 300
repetitions. All spellings are DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__tl_layer__ = "L5"

#: Closed narration event kinds. ``attempted`` is the in-flight marker line;
#: ``note`` carries suppression markers and footer disclosures.
NARRATION_KINDS = (
    "op",
    "input",
    "buffer",
    "module_enter",
    "module_exit",
    "attempted",
    "note",
)


@dataclass(frozen=True, slots=True)
class NarrationStats:
    """Structured stats segment for one narrated tensor (snoop D4).

    Exactly one of the three evidence shapes is populated:

    - ``policy="sampled"``: moments from a bounded seeded sample, ``~``-marked,
      with ``sample_size``/``population`` evidence. NEVER carries finiteness
      fields -- a planted NaN in 12.6M elements is invisible to an 8k
      subsample (measured false negative), so a sampled line omits the
      finiteness family entirely rather than printing a clearance.
    - ``policy="exact"``: exact moments and the exact nonfinite census.
    - ``policy="reuse"``: only facts another armed feature already paid for
      (``track_nonfinite`` / ``raise_on_nan`` / saved payloads); every field
      another feature did not pay for stays ``None`` and is omitted.
    """

    policy: str
    population: int
    sample_size: int | None = None
    mean: float | None = None
    sd: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    #: Exact nonfinite census fields; ``None`` = no claim (ABSENT on lines).
    nan_count: int | None = None
    posinf_count: int | None = None
    neginf_count: int | None = None
    #: track_nonfinite reuse verdict: True = has nonfinite, False = finite,
    #: ``None`` = no synchronous evidence (deferred device flag -- omitted,
    #: never printed as though known live).
    has_nonfinite: bool | None = None


@dataclass(frozen=True, slots=True)
class NarrationEvent:
    """One immutable narration record (snoop D2; the structured channel).

    ``label`` is TIER-HONEST: the spelling THE OBJECT RETURNED BY THIS CALL
    will accept for lookup (the raw in-flight label on both live tiers;
    post-hoc rendering re-derives final labels natively). ``direction`` is
    reserved for backward narration and is always ``"forward"`` in V1.
    """

    kind: str
    ordinal: int
    label: str
    pass_index: int = 1
    step_index: int | None = None
    layer_type: str | None = None
    func_name: str | None = None
    address: str | None = None
    module_type: str | None = None
    module_depth: int = 0
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    device: str | None = None
    output_index: int | None = None
    source_location: str | None = None
    intervened: bool = False
    #: Ordinal distance disclosed on buffered ``followed_by`` releases.
    late: int | None = None
    stats: NarrationStats | None = None
    #: Free-form single-line body for ``attempted`` / ``note`` kinds.
    text: str | None = None
    direction: str = "forward"
    tier: str | None = None
    extra_labels: tuple[tuple[str, str], ...] = field(default=())


__all__ = ["NARRATION_KINDS", "NarrationEvent", "NarrationStats"]
