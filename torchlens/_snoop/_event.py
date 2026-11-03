"""The frozen ``NarrationEvent`` record behind every narration line (snoop D2).

Narration is data first: every rendered line is backed by one immutable,
backend-neutral record. The record -- not the text -- is the parser channel,
which is what lets the line grammar optimize for the human scanning 300
repetitions. All spellings are DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..stats._tensor_stats import TensorStats

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
class NarrationEvent:
    """One immutable narration record (snoop D2; the structured channel).

    ``label`` is TIER-HONEST: the spelling THE OBJECT RETURNED BY THIS CALL
    will accept for lookup (the raw in-flight label on both live tiers;
    post-hoc rendering re-derives final labels natively). ``signal`` is
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
    has_intervention: bool = False
    #: Ordinal distance disclosed on buffered ``followed_by`` releases.
    late: int | None = None
    stats: TensorStats | None = None
    #: Free-form single-line body for ``attempted`` / ``note`` kinds.
    text: str | None = None
    signal: str = "forward"
    tier: str | None = None
    extra_labels: tuple[tuple[str, str], ...] = field(default=())


__all__ = ["NARRATION_KINDS", "NarrationEvent"]
