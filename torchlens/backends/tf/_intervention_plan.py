"""TensorFlow intervention site and plan records (leaf: no capture-core imports)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from ..registry import BackendUnsupportedError

if TYPE_CHECKING:
    from ..._selection.selectors import BaseSelector
    from ...intervention._types import InterventionDecision


class TFInterventionSiteUnreachableError(BackendUnsupportedError):
    """A selector matched captured ops the writable wrap layer never saw."""


@dataclass(frozen=True)
class TFInterventionSite:
    """One normalized TensorFlow intervention site.

    Parameters
    ----------
    plan_id
        Stable site identifier within the plan.
    predicate
        ``tl.when``-built predicate (or user callable) returning a decision on match.
    selector
        Static selector, or ``None`` for a value-dependent site.
    decision
        Normalized decision; ``None`` on a value-dependent site until it fires.
    hook
        Resolved TensorFlow replacement callable (``None`` like ``decision``).
    level
        Dispatch level, ``"module"`` or ``"op"``.
    """

    plan_id: str
    predicate: Callable[[Any], Any]
    selector: BaseSelector | None
    decision: InterventionDecision | None
    hook: Callable[[Any], Any] | None
    level: Literal["module", "op"]


@dataclass
class TFInterventionPlan:
    """Mutable per-capture TensorFlow intervention state.

    Parameters
    ----------
    sites
        All normalized sites.
    op_sites
        Op-level sites dispatched by the functional wrap layer.
    module_sites
        Module-boundary sites dispatched at module exits.
    pending_decisions
        Op label to the sites that fired on it in the callback, awaiting the wrap layer.
    on_presented
        Called with ``(label, value)`` after the wrap layer handled an op (halt release).
    """

    sites: tuple[TFInterventionSite, ...]
    op_sites: tuple[TFInterventionSite, ...]
    module_sites: tuple[TFInterventionSite, ...]
    presented_labels: set[str] = field(default_factory=set)
    hook_generated_labels: set[str] = field(default_factory=set)
    fired_site_labels: list[tuple[str, str]] = field(default_factory=list)
    pending_decisions: dict[str, list[TFInterventionSite]] = field(default_factory=dict)
    on_presented: Callable[[str, Any], None] | None = None
    _in_hook: bool = False
