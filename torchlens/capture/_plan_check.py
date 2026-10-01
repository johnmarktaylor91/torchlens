"""The audit-only plan checker (weightsfree memo D14, face 4).

nnsight-scan parity: resolve selectors against a trace, check multiplicity
(a bare selector must name at least one site; a declared replacement is ONE
geometry for ONE site) and index bounds (a pass-qualified ``label:k`` must
name an existing pass), and compare declared replacement geometry (a meta/real
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
        bounds_finding = _index_bounds_finding(trace, selector)
        if bounds_finding is not None:
            checks.append(
                PlanSiteCheck(
                    selector=repr(selector),
                    resolved_labels=(),
                    multiplicity_ok=False,
                    replacement_verdict=bounds_finding,
                )
            )
            continue
        table = trace.resolve_sites(selector)
        labels_attr = getattr(table, "labels", None)
        raw_labels = labels_attr() if callable(labels_attr) else labels_attr
        labels = tuple(str(site) for site in (raw_labels or _labels_of(table)))
        multiplicity_ok, verdict = _multiplicity_verdict(labels, replacement)
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


def _multiplicity_verdict(labels: tuple[str, ...], replacement: Any) -> tuple[bool, str]:
    """Return ``(multiplicity_ok, verdict)`` for one resolved plan entry.

    A bare selector audits site EXISTENCE (at least one site). A declared
    replacement is ONE geometry destined for ONE site (nnsight-scan parity:
    ``module.output = value`` names one call), so an entry that resolves to
    several sites is a multiplicity finding, never ``multiplicity_ok`` --
    the historical ``len(labels) > 0`` blessed a selector fanning one
    replacement out over 50 sites (AUD-CODE 4.2).
    """

    count = len(labels)
    if count == 0:
        return False, "site_unresolved: selector matched no site on this trace"
    if replacement is not None and count > 1:
        return False, (
            f"multiplicity_mismatch: {count} sites resolved for ONE declared "
            "replacement; qualify the selector to a single site "
            "(a pass-qualified label such as 'label:pass')"
        )
    return True, "not_declared"


def _index_bounds_finding(trace: Any, selector: Any) -> str | None:
    """Return an index-bounds finding for a pass-qualified label, or ``None``.

    ``label:k`` names pass ``k`` (1-based) of a layer that exists on the
    trace; ``k`` outside ``1..num_passes`` is a plan defect the resolver
    would otherwise surface as an opaque "matched 0 sites" refusal. Unknown
    base labels are NOT bounds findings (the resolver's typo refusal owns
    them); non-string selectors have no index to bound.
    """

    if not isinstance(selector, str) or ":" not in selector:
        return None
    base, _, pass_text = selector.rpartition(":")
    try:
        pass_index = int(pass_text)
    except ValueError:
        return None
    layer_labels = {
        str(getattr(layer, "layer_label", "")) for layer in getattr(trace, "layer_list", ())
    }
    if base not in layer_labels:
        return None
    layer = trace[base]
    num_passes = int(getattr(layer, "num_passes", 1) or 1)
    if 1 <= pass_index <= num_passes:
        return None
    return (
        f"index_out_of_bounds: pass {pass_index} of {base!r} does not exist "
        f"(the layer ran {num_passes} pass{'es' if num_passes != 1 else ''}; "
        f"valid pass indices are 1..{num_passes})"
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
