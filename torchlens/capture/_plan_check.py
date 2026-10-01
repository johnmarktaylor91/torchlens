"""The audit-only plan checker (weightsfree memo D14, face 4).

nnsight-scan parity: resolve selectors against a trace, check multiplicity
and index bounds, and compare declared replacement geometry (a meta/real
tensor or any shape/dtype-bearing spec) against the trace's hypothesis
shapes/dtypes. The report is a PERSISTABLE artifact whose claims carry typed
status until discharged — where nnsight's scan evaporates with its trace
context.

FOREVER SEPARATE from the runnable gate: checking a plan never sets
``intervention_ready``, arms, replays, or lifts the L7b late-bind refusal —
scanning a plan is not arming a capture. ``executable`` is ``False`` on
every report by construction. Refused typed: arbitrary callables as
replacements, value-derived selection, and REFUTED sources (through the one
capability chokepoint — a refuted hypothesis licenses no plan claims).

Every spelling is DOCUMENTED-UNSTABLE; the public verb name belongs to the
UI sprint (memo sec 12) — until then the spelling is ``Trace.check_plan``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["PlanCheckReport", "PlanSiteCheck", "check_plan"]


@dataclass(frozen=True)
class PlanSiteCheck:
    """One checked plan entry: the resolved sites and the geometry verdict."""

    selector: str
    resolved_labels: tuple[str, ...]
    multiplicity_ok: bool
    replacement_verdict: str  # "not_declared" | "geometry_ok" | "geometry_mismatch: ..."


@dataclass(frozen=True)
class PlanCheckReport:
    """Frozen audit result for one plan against one trace.

    ``executable`` is ALWAYS ``False``: this report licenses no replay and no
    arming; ``evidence_status`` is the trace's claim status at check time
    (hypothesis / corroborated) and every geometry verdict inherits it.
    """

    sites: tuple[PlanSiteCheck, ...]
    source_digest: str
    evidence_status: str
    executable: bool = False

    @property
    def ok(self) -> bool:
        """Whether every entry resolved and every declared geometry matched."""

        return all(
            site.multiplicity_ok and not site.replacement_verdict.startswith("geometry_mismatch")
            for site in self.sites
        )

    def report(self) -> str:
        """One-screen human projection of the checked plan."""

        lines = [
            f"plan check: {'OK' if self.ok else 'FINDINGS'} "
            f"({len(self.sites)} entr{'y' if len(self.sites) == 1 else 'ies'}); "
            f"executable=false (audit-only, never arms a capture)",
            f"evidence status: {self.evidence_status} (source digest {self.source_digest[:16]}...)",
        ]
        for site in self.sites:
            lines.append(
                f"  {site.selector}: {len(site.resolved_labels)} site(s) "
                f"[{', '.join(site.resolved_labels[:4])}"
                f"{', ...' if len(site.resolved_labels) > 4 else ''}] "
                f"replacement={site.replacement_verdict}"
            )
        return "\n".join(lines)


def _refuse(problem: str, remedy: str, **fields: Any) -> None:
    """Typed plan-check refusal (single code, reason-carrying)."""

    from .._errors import InvalidArgumentError

    raise InvalidArgumentError(
        problem,
        code="plan_check_unsupported",
        remedy=remedy,
        argument="plan",
        **fields,
    )


def _geometry_verdict(trace: Any, labels: tuple[str, ...], replacement: Any) -> str:
    """Compare a declared replacement's geometry against site hypotheses."""

    shape = getattr(replacement, "shape", None)
    dtype = getattr(replacement, "dtype", None)
    if shape is None and dtype is None:
        _refuse(
            f"plan replacement {replacement!r} declares neither shape nor "
            "dtype; a plan check can only audit DECLARED geometry",
            "pass a meta tensor (torch.empty(shape, device='meta', dtype=...)) "
            "or any object with shape/dtype attributes as the replacement",
        )
    for label in labels:
        layer = trace[label]
        site_shape = getattr(layer, "shape", None)
        site_dtype = getattr(layer, "dtype", None)
        if shape is not None and site_shape is not None and tuple(shape) != tuple(site_shape):
            return (
                f"geometry_mismatch: {label} expects shape {tuple(site_shape)}, "
                f"plan declares {tuple(shape)}"
            )
        if dtype is not None and site_dtype is not None and str(dtype) != str(site_dtype):
            return f"geometry_mismatch: {label} expects dtype {site_dtype}, plan declares {dtype}"
    return "geometry_ok"


def check_plan(trace: Any, plan: Any) -> PlanCheckReport:
    """Audit a plan of (selector[, replacement]) entries against a trace.

    Parameters
    ----------
    trace:
        Any finished trace (structure-only traces are the design center; the
        REFUTED state refuses through the capability chokepoint).
    plan:
        Iterable of entries: a bare selector (site existence/multiplicity
        check only) or a ``(selector, replacement)`` pair where the
        replacement declares ``shape``/``dtype`` (a meta tensor is the
        canonical spelling). Callables refuse typed — a callable's output
        geometry is not declarable without running it.

    Returns
    -------
    PlanCheckReport
        Frozen audit report; ``executable`` is always ``False``.
    """

    from .structure_only import claim_status_for, require_structure_only_capability

    # REFUTED sources refuse via the chokepoint (supported_hypothesis row).
    require_structure_only_capability(trace, "plan_shape_check")
    checks: list[PlanSiteCheck] = []
    for entry in plan:
        if isinstance(entry, tuple) and len(entry) == 2:
            selector, replacement = entry
        else:
            selector, replacement = entry, None
        if callable(replacement) and not hasattr(replacement, "shape"):
            _refuse(
                f"plan replacement for {selector!r} is a callable; its output "
                "geometry cannot be audited without executing it, and a plan "
                "check never executes",
                "declare the replacement geometry (a meta tensor or a "
                "shape/dtype-bearing spec) instead of a callable",
            )
        if callable(selector) and not _is_static_selector(selector):
            _refuse(
                f"plan selector {selector!r} is a bare callable or "
                "value-dependent predicate; plan checks audit STRUCTURAL "
                "selection only",
                "use label strings or structural selectors (tl.func, "
                "tl.in_module, label selectors and their compositions)",
            )
        table = trace.resolve_sites(selector)
        labels_attr = getattr(table, "labels", None)
        raw_labels = labels_attr() if callable(labels_attr) else labels_attr
        labels = tuple(str(site) for site in (raw_labels or _labels_of(table)))
        multiplicity_ok = len(labels) > 0
        verdict = "not_declared"
        if replacement is not None and multiplicity_ok:
            verdict = _geometry_verdict(trace, labels, replacement)
        checks.append(
            PlanSiteCheck(
                selector=repr(selector),
                resolved_labels=labels,
                multiplicity_ok=multiplicity_ok,
                replacement_verdict=verdict,
            )
        )
    from .. import hash as tl_hash

    return PlanCheckReport(
        sites=tuple(checks),
        source_digest=tl_hash.trace(trace),
        evidence_status=(
            claim_status_for(trace).value
            if bool(getattr(trace, "structure_only", False))
            else "measured"
        ),
    )


def _is_static_selector(selector: Any) -> bool:
    """Whether a callable selector is a provably value-free structural one."""

    from ..intervention.selectors import BaseSelector
    from ..ir.selector_eval import first_selector_kind_outside
    from ..postprocess._selective_save import _STATIC_SELECTOR_KINDS

    if not isinstance(selector, BaseSelector):
        return False
    return first_selector_kind_outside(selector, allowed=_STATIC_SELECTOR_KINDS) is None


def _labels_of(table: Any) -> tuple[str, ...]:
    """Best-effort site labels from a resolver SiteTable."""

    for attribute in ("site_labels", "resolved_labels", "sites"):
        value = getattr(table, attribute, None)
        if value:
            return tuple(str(item) for item in value)
    try:
        return tuple(str(item) for item in table)
    except TypeError:
        return ()
