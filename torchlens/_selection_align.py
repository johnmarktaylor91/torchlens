"""Cross-run selection alignment (L6 stage 4a).

``ResolvedSelection.align_to(target)`` re-binds a selection resolved on
trace A onto trace B, keyed on the L1 STRUCTURAL SITE KEYS (``op.site_key``,
``site_key_v1``) that every ACT :class:`~torchlens.selection.SiteEntry`
records as its bridging relation. This is the sanctioned cross-run door:
``do()`` keeps refusing foreign resolved selections typed
(``selection_trace_mismatch``) — alignment is EXPLICIT, never a silent
rebind.

SAME-POLICY CAPTURES ONLY (the L1 cross-stamp rule, design-L1-memo consumer
matrix): both traces must carry a HEALTHY ``grouping_policy_v1`` stamp and
the stamps must agree on ``policy`` / ``folded_sites`` / ``site_join``.
Degraded or missing stamps (every loaded artifact today — the stamp is
``FieldPolicy.DROP``) refuse typed; cross-stamp alignment is deliberately
NOT relaxed here (L1 owns that door and it opens only behind L1's exit-gate
evidence).

Alignment is CONSERVATIVE (the memo's "conservative stance held anyway"):
a site aligns only when the target trace has an op matching BOTH the
structural site key (proves position) AND the ``(layer_label, pass_index)``
address (same-policy captures of one architecture agree on labels; a
key/label disagreement means the pair is not the same-policy pair the gate
promises, and the join refuses rather than guesses). Masks travel unchanged
and must land on an identical index space.

Every spelling here ships DOCUMENTED-UNSTABLE pending naming-session
ratification. Refusals ride ``SelectionError`` with the single new code
``selection_alignment_invalid`` and a CLOSED reason set (S2-routed,
provisional): ``kind_unsupported | grouping_stamp_degraded |
grouping_stamp_mismatch | site_key_unavailable | site_not_in_target |
index_space_mismatch``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .selection import ResolvedSelection

__all__ = ["ALIGNMENT_REASONS", "align_resolved_selection"]

#: Closed reason vocabulary for ``selection_alignment_invalid`` (S2-routed,
#: provisional pending the amendment that also covers the stage-1 codes).
ALIGNMENT_REASONS: tuple[str, ...] = (
    "kind_unsupported",
    "grouping_stamp_degraded",
    "grouping_stamp_mismatch",
    "site_key_unavailable",
    "site_not_in_target",
    "index_space_mismatch",
)

#: The stamp axes that must agree for two captures to count as same-policy.
_SAME_POLICY_AXES: tuple[str, ...] = ("policy", "folded_sites", "site_join")


def _alignment_invalid(reason: str, message: str, **fields: Any) -> Exception:
    """Build the closed-reason cross-run alignment refusal.

    Raises
    ------
    ValueError
        If ``reason`` is outside the closed alignment-reason vocabulary.
    """

    from .selection import SelectionError

    if reason not in ALIGNMENT_REASONS:
        raise ValueError(f"unknown selection alignment reason: {reason!r}")
    return SelectionError(
        message,
        code="selection_alignment_invalid",
        reason=reason,
        **fields,
    )


def _healthy_stamp(trace: Any, role: str) -> dict[str, Any]:
    """Return one trace's healthy grouping stamp, or refuse typed.

    A healthy stamp is a present ``grouping_policy_v1`` payload with
    ``effective=True``, a known policy, and no settlement note — exactly the
    live wave-0 writer output. Loaded artifacts settle degraded (the stamp
    is session-time), so alignment is live-capture territory today; that is
    the L1 consumer matrix's "degraded stamp -> typed refusal" row, not a
    gap to relax here.
    """

    stamp = getattr(trace, "grouping_policy", None)
    if (
        not isinstance(stamp, dict)
        or stamp.get("effective") is not True
        or stamp.get("policy") in (None, "unknown")
        or stamp.get("settlement_note") is not None
    ):
        raise _alignment_invalid(
            "grouping_stamp_degraded",
            f"the {role} trace does not carry a healthy grouping-policy stamp "
            "(degraded, legacy, or loaded artifact); cross-run alignment is "
            "same-policy-only and requires healthy stamps on both sides.",
            role=role,
        )
    return stamp


def _require_same_policy(resolved: ResolvedSelection, target: Any) -> None:
    """Enforce the L1 same-policy cross-stamp rule across both traces."""

    source_stamp = _healthy_stamp(resolved._trace, "source")
    target_stamp = _healthy_stamp(target, "target")
    for axis in _SAME_POLICY_AXES:
        if source_stamp.get(axis) != target_stamp.get(axis):
            raise _alignment_invalid(
                "grouping_stamp_mismatch",
                f"cross-run alignment requires same-policy captures: stamp axis "
                f"{axis!r} differs (source {source_stamp.get(axis)!r} vs target "
                f"{target_stamp.get(axis)!r}). Cross-stamp alignment is typed-"
                "refused until L1's exit gate proves it.",
                axis=axis,
                source_value=source_stamp.get(axis),
                target_value=target_stamp.get(axis),
            )


def _target_op_for_entry(entry: Any, target: Any, source_trace: Any = None) -> Any:
    """Find the ONE target op for an entry, SITE-KEY-FIRST (C03 / leverage B4).

    The fast path keeps the shipped behavior: the same-policy
    ``(layer_label, pass_index)`` address resolves and its structural key
    agrees. When the LABEL misses or disagrees -- one live edit renumbers
    31-62 labels on real resnet18, so a structurally-intact site used to
    refuse here -- the label leaves the join key: the target is looked up by
    structural site key alone, and the SHIPPED guarded join must return a
    ``corroborated`` verdict for the key before the rebind is authorized
    (``positional`` is display-only, per the verdict policy; a multi-op key
    family stays ambiguous and refuses).
    """

    from .selection import _find_act_ops

    structural_key = entry.structural_site_key
    if structural_key is None:
        raise _alignment_invalid(
            "site_key_unavailable",
            f"resolved site {entry.site_key!r} carries no structural site key "
            "(legacy keyless capture); cross-run alignment keys on L1 site "
            "keys and cannot bridge without one.",
            site=repr(entry.site_key),
        )
    layer_label, pass_index = entry.site_key
    candidates = [
        op
        for op in _find_act_ops(target, layer_label)
        if (getattr(op, "pass_index", 1) or 1) == pass_index
    ]
    if candidates and getattr(candidates[0], "site_key", None) == structural_key:
        return candidates[0]
    key_family = [
        target.ops[label]
        for label in target.op_labels
        if getattr(target.ops[label], "site_key", None) == structural_key
    ]
    if len(key_family) == 1:
        verdict = _join_verdict_for_key(source_trace, target, structural_key)
        if verdict == "corroborated":
            return key_family[0]
        raise _alignment_invalid(
            "site_not_in_target",
            f"the target trace holds structural site key {structural_key!r} "
            f"but the guarded site join verdict is {verdict!r}, not "
            "'corroborated' -- positional agreement is display-only and a "
            "refused join names a structural change; cross-run action is "
            "authorized only by a corroborated join (verdict policy D-2).",
            site=repr(entry.site_key),
            structural_site_key=structural_key,
            join_verdict=verdict,
        )
    raise _alignment_invalid(
        "site_not_in_target",
        f"the target trace has no op at address {entry.site_key!r} with "
        f"structural site key {structural_key!r}"
        + (
            f" ({len(key_family)} target ops share that key -- a reuse family "
            "is ambiguous without the same-policy address)"
            if key_family
            else ""
        )
        + "; cross-run alignment requires the position proof (site key) and, "
        "when the label survives, the same-policy address to agree.",
        site=repr(entry.site_key),
        structural_site_key=structural_key,
    )


def _join_verdict_for_key(source_trace: Any, target: Any, structural_key: str) -> str:
    """Run the SHIPPED guarded join for one key and return its verdict value."""

    from .postprocess._site_join import join_site_profiles, site_profile

    if source_trace is None:
        return "unjoined"
    rows = join_site_profiles(site_profile(source_trace), site_profile(target))
    row = rows.get(structural_key)
    if row is None:
        return "unjoined"
    return str(row.verdict.value)


def align_resolved_selection(resolved: ResolvedSelection, target: Any) -> ResolvedSelection:
    """Re-bind a resolved selection onto ``target``, keyed on L1 site keys.

    Implementation home of ``ResolvedSelection.align_to``. Returns a NEW
    frozen ``ResolvedSelection`` bound to ``target`` whose entries carry the
    source masks unchanged, with per-entry provenance disclosing the
    cross-run origin. ``target is source`` returns ``resolved`` itself
    (alignment is the identity there).

    Raises
    ------
    SelectionError
        ``selection_alignment_invalid`` with the closed reason set
        documented in the module docstring.
    """

    import dataclasses

    from .selection import ResolvedSelection, SelectionProvenance, _site_shape

    if target is resolved._trace:
        return resolved
    if resolved.kind != "ACT":
        raise _alignment_invalid(
            "kind_unsupported",
            f"cross-run alignment supports ACT selections only; {resolved.kind} "
            "site addresses are not stable across runs (PARAM edits are typed-"
            "refused pending D3; EDGE occurrence addresses are per-capture).",
            kind=resolved.kind,
        )
    _require_same_policy(resolved, target)
    source_label = str(getattr(resolved._trace, "trace_label", "") or "source trace")
    entries = []
    for entry in resolved:
        op = _target_op_for_entry(entry, target, resolved._trace)
        target_shape = _site_shape(op)
        if target_shape != entry.shape:
            raise _alignment_invalid(
                "index_space_mismatch",
                f"target site {entry.site_key!r} output space {target_shape!r} "
                f"does not match the source index space {entry.shape!r}; masks "
                "travel unchanged and cannot re-index.",
                site=repr(entry.site_key),
                source_shape=entry.shape,
                target_shape=target_shape,
            )
        provenance = SelectionProvenance(
            relation=entry.provenance.relation,
            source=f"{entry.provenance.source} [cross-run aligned from {source_label!r}]",
        )
        target_address = (op.layer_label, getattr(op, "pass_index", 1) or 1)
        if target_address != entry.site_key:
            # Site-key-first rebind under label drift (C03/B4): the aligned
            # entry addresses the TARGET op's same-policy address; the
            # provenance line above already discloses the cross-run origin.
            entries.append(
                dataclasses.replace(entry, site_key=target_address, provenance=provenance)
            )
        else:
            entries.append(dataclasses.replace(entry, provenance=provenance))
    return ResolvedSelection(target, "ACT", entries)
