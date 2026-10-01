"""Saved intervention-spec compatibility checking (site-key-first).

Split out of ``save.py`` (C03 fix cycle): the compat preview owns the
site-key-first join (surgery Build 0b) -- the structural key is the join
identity, labels stay display-only disclosure. ``save.py`` re-exports the
public names, so ``torchlens.intervention.save.check_spec_compat`` and
``torchlens.validation.check_spec_compat`` keep resolving.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple

from .errors import GraphShapeMismatchError, SiteResolutionError
from .resolver import resolve_sites
from .types import InterventionSpec

__all__ = ["SpecCompat", "TargetManifestDiff", "check_spec_compat"]


@dataclass(frozen=True)
class TargetManifestDiff:
    """Diff between saved target manifest and a new model log.

    Site-key rows (C03, surgery Build 0b; DOCUMENTED-UNSTABLE) are the join
    identity when both sides carry keys; the label rows stay as display-only
    disclosure (labels renumber under edits, keys do not).
    """

    matched: list[str]
    new_labels: list[str]
    missing_labels: list[str]
    selector_resolution_diffs: dict[str, dict[str, Any]]
    matched_site_keys: list[str] = field(default_factory=list)
    new_site_keys: list[str] = field(default_factory=list)
    missing_site_keys: list[str] = field(default_factory=list)

    def site_diff_lines(self) -> list[str]:
        """Render the printed site-level diff (one line per changed site)."""

        lines = [f"  matched   {key}" for key in self.matched_site_keys]
        lines.extend(f"  + new     {key}" for key in self.new_site_keys)
        lines.extend(f"  - missing {key}" for key in self.missing_site_keys)
        return lines


@dataclass(frozen=True)
class SpecCompat:
    """Compatibility result for applying a saved spec to a model log."""

    outcome: Literal["EXACT", "COMPATIBLE_WITH_CONFIRMATION", "FAIL"]
    diff: TargetManifestDiff
    targets_resolve_identically: bool
    #: Human-readable site-level diff (C03; empty when no keys on either side).
    site_diff: str = ""


def _site_key_for_label(log: Any, label: str) -> str | None:
    """Return the op's structural site key, or None for keyless rows.

    Backward selectors resolve to grad-fn site labels that are not in the
    forward op universe; those rows are honestly keyless (site keys are a
    forward structural identity).
    """

    try:
        op = log.ops[label]
    except (KeyError, AttributeError, TypeError):
        # The honest lookup-failure set: missing label (accessor KeyError),
        # a trace-like without .ops, or a husked/None accessor.
        return None
    key = getattr(op, "site_key", None)
    return key if isinstance(key, str) else None


def _resolution_fanout_bound(log: Any, *, min_required: int = 1) -> int:
    """Return the strict resolver fanout bound for persistence workflows.

    Parameters
    ----------
    log:
        Trace-like object used for resolution.
    min_required:
        Minimum bound required by already-validated saved labels.

    Returns
    -------
    int
        Explicit resolver fanout bound.
    """

    layer_list = getattr(log, "layer_list", None)
    layer_count = len(layer_list) if layer_list is not None else len(getattr(log, "layer_logs", {}))
    return max(1, int(min_required), int(layer_count))


@dataclass
class _ManifestResolution:
    """Accumulated per-selector resolution facts for one compat check."""

    all_saved: set[str] = field(default_factory=set)
    all_resolved: set[str] = field(default_factory=set)
    all_saved_keys: set[str] = field(default_factory=set)
    all_resolved_keys: set[str] = field(default_factory=set)
    selector_diffs: dict[str, dict[str, Any]] = field(default_factory=dict)
    unresolved: bool = False
    graph_matches: bool = True
    label_only_mismatch: bool = False


class _EntryFacts(NamedTuple):
    """Resolution facts for ONE saved selector entry on the new log."""

    saved_labels: list[str]
    saved_keys: list[str]
    keys_complete: bool
    hash_matches: bool
    resolved_labels: list[str] | None  # None = the selector failed to resolve
    resolved_keys: list[str]
    error: str | None


def _resolve_entry(new_log: Any, entry: dict[str, Any], graph_hash: Any) -> _EntryFacts:
    """Resolve one manifest entry's selector on the new log (pure lookup)."""

    from .save import _target_spec_from_json

    saved_labels = list(entry.get("resolved_labels", []))
    saved_keys_raw = entry.get("resolved_site_keys")
    saved_keys = [key for key in (saved_keys_raw or []) if isinstance(key, str)]
    keys_complete = bool(saved_keys_raw) and len(saved_keys) == len(saved_labels)
    hash_matches = entry.get("graph_shape_hash") == graph_hash
    selector = _target_spec_from_json(entry["selector"])
    try:
        resolved_labels = list(
            resolve_sites(
                new_log,
                selector,
                strict=True,
                max_fanout=_resolution_fanout_bound(new_log, min_required=len(saved_labels)),
            ).labels()
        )
    except SiteResolutionError as exc:
        return _EntryFacts(
            saved_labels, saved_keys, keys_complete, hash_matches, None, [], str(exc)
        )
    resolved_keys = [
        key
        for key in (_site_key_for_label(new_log, label) for label in resolved_labels)
        if isinstance(key, str)
    ]
    return _EntryFacts(
        saved_labels, saved_keys, keys_complete, hash_matches, resolved_labels, resolved_keys, None
    )


def _fold_entry(
    res: _ManifestResolution, entry: dict[str, Any], index: int, facts: _EntryFacts
) -> None:
    """Fold one entry's resolution facts into the running accumulator."""

    selector_key = f"selector_{index}"
    res.all_saved.update(facts.saved_labels)
    if not facts.hash_matches:
        res.graph_matches = False
    diff_row: dict[str, Any] = {
        "selector": entry["selector"],
        "saved_labels": facts.saved_labels,
        "resolved_labels": facts.resolved_labels if facts.resolved_labels is not None else [],
        "saved_site_keys": facts.saved_keys,
        "resolved_site_keys": facts.resolved_keys,
    }
    if facts.resolved_labels is None:
        diff_row["error"] = facts.error
        res.selector_diffs[selector_key] = diff_row
        res.unresolved = True
        return
    res.all_resolved.update(facts.resolved_labels)
    keys_comparable = facts.keys_complete and len(facts.resolved_keys) == len(facts.resolved_labels)
    if keys_comparable:
        res.all_saved_keys.update(facts.saved_keys)
        res.all_resolved_keys.update(facts.resolved_keys)
    if facts.resolved_labels == facts.saved_labels:
        return
    # SITE-KEY-FIRST (C03, surgery Build 0b): the structural key is the join
    # identity; a label drift with IDENTICAL key multisets is disclosure,
    # never incompatibility -- one live edit renumbers 31-62 labels on real
    # resnet18 without moving a single site.
    if keys_comparable and sorted(facts.saved_keys) == sorted(facts.resolved_keys):
        diff_row["label_drift_only"] = True
        res.selector_diffs.setdefault(selector_key, diff_row)
        res.label_only_mismatch = True
    else:
        res.selector_diffs[selector_key] = diff_row


def _derive_outcome(
    res: _ManifestResolution,
) -> tuple[Literal["EXACT", "COMPATIBLE_WITH_CONFIRMATION", "FAIL"], bool, TargetManifestDiff]:
    """Derive the compat verdict and diff from the accumulated facts."""

    diff = TargetManifestDiff(
        matched=sorted(res.all_saved & res.all_resolved),
        new_labels=sorted(res.all_resolved - res.all_saved),
        missing_labels=sorted(res.all_saved - res.all_resolved),
        selector_resolution_diffs=res.selector_diffs,
        matched_site_keys=sorted(res.all_saved_keys & res.all_resolved_keys),
        new_site_keys=sorted(res.all_resolved_keys - res.all_saved_keys),
        missing_site_keys=sorted(res.all_saved_keys - res.all_resolved_keys),
    )
    substantive_diffs = {
        key: row
        for key, row in res.selector_diffs.items()
        if not row.get("label_drift_only", False)
    }
    if res.all_saved_keys:
        targets_identical = (
            not substantive_diffs and not diff.new_site_keys and not diff.missing_site_keys
        )
        missing_side = bool(diff.missing_site_keys)
        superset_side = res.all_saved_keys.issubset(res.all_resolved_keys)
    else:
        targets_identical = (
            not res.selector_diffs and not diff.new_labels and not diff.missing_labels
        )
        missing_side = bool(diff.missing_labels)
        superset_side = res.all_saved.issubset(res.all_resolved)

    outcome: Literal["EXACT", "COMPATIBLE_WITH_CONFIRMATION", "FAIL"]
    if res.unresolved or missing_side:
        outcome = "FAIL"
    elif targets_identical and res.graph_matches:
        outcome = "EXACT"
    elif targets_identical and res.label_only_mismatch:
        # Identical structural targets on a drifted-label graph: compatible,
        # confirmation discloses the drift (never a silent EXACT -- the graph
        # hash differs by construction when labels renumber).
        outcome = "COMPATIBLE_WITH_CONFIRMATION"
    elif superset_side or not res.graph_matches:
        outcome = "COMPATIBLE_WITH_CONFIRMATION"
    else:
        outcome = "FAIL"
    return outcome, targets_identical, diff


def check_spec_compat(spec: InterventionSpec, new_log: Any) -> SpecCompat:
    """Check whether a loaded intervention spec targets a new model log.

    Parameters
    ----------
    spec:
        Loaded or in-memory intervention spec.
    new_log:
        Model log to check.

    Returns
    -------
    SpecCompat
        Compatibility classification and target diff.
    """

    graph_hash = getattr(new_log, "graph_shape_hash", None)
    res = _ManifestResolution()
    for index, entry in enumerate(spec.metadata.get("target_manifest", [])):
        _fold_entry(res, entry, index, _resolve_entry(new_log, entry, graph_hash))
    outcome, targets_identical, diff = _derive_outcome(res)

    # A graph_shape_hash mismatch alone cannot distinguish a genuinely different
    # target graph from mere cross-version hash drift on the SAME graph (an older
    # torchlens computes a different hash for identical topology; the v2.16 backcompat
    # fixtures encode exactly this and resolve to identical labels). Refusing at
    # compat-preview time on any mismatch would break every cross-version executable
    # spec reuse. ``COMPATIBLE_WITH_CONFIRMATION`` is the honest preview verdict here --
    # it flags the shape difference and defers to explicit confirmation. The genuine
    # "wrong graph" tripwire lives at REPLAY time (see torchlens/intervention/replay.py
    # _warn_if_unexpected_parent / _check_edge_expectations), which compares actual
    # parent/edge topology and raises ControlFlowDivergenceError under strict replay --
    # a version-stable structural check, not a coarse hash string. The narrow existing
    # refusal below stays: an executable spec whose targets cannot even resolve on a
    # mismatched graph is a hard GraphShapeMismatchError.
    if outcome == "FAIL" and bool(spec.metadata.get("executable", False)) and not res.graph_matches:
        raise GraphShapeMismatchError(
            "Saved spec's graph_shape_hash doesn't match target log; refusing to apply at "
            "executable level."
        )
    site_diff_lines = diff.site_diff_lines()
    site_diff = "site-level diff:\n" + "\n".join(site_diff_lines) if site_diff_lines else ""
    return SpecCompat(outcome, diff, targets_identical, site_diff=site_diff)
