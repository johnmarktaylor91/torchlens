"""Shared capture-honesty facts and preamble for report/export surfaces.

ONE source of truth for the honesty facts every human report, machine dump,
tabular export, and file export must carry: the settled capture outcome, the
verification ceiling, the rescue disclosure, structure-only hypothesis status,
sparse-run poisoning, episode capture (including the forced-tokens
non-verifying basis), and persisted capture-time advisories.

``tl.report.explain`` and ``Trace.to_agent_json`` read their honesty facts
through this module, and every exporter in ``torchlens.export``, every debug
DataFrame, ``Trace.to_pandas``, and the fastlog table attach the same facts
through :func:`attach_dataframe_honesty` / :func:`honesty_preamble_lines` --
an export without these facts presents a possibly-unverified, possibly-poisoned
capture as clean data.

Everything here is JSON-primitive and plain ASCII: no ANSI/OSC escapes, no
resolved absolute-path URIs (returned text is publishable data).
"""

from __future__ import annotations

import contextlib
from typing import Any

#: Schema identifier stamped on the shared honesty fact block.
CAPTURE_HONESTY_SCHEMA = "torchlens.capture_honesty.v1"

#: Key under which DataFrame exports carry the fact block (``DataFrame.attrs``).
DATAFRAME_ATTRS_KEY = "torchlens_capture_honesty"

#: Key under which capture-time advisories live in ``Trace.annotations``.
ADVISORIES_ANNOTATIONS_KEY = "capture_advisories"

#: ``Trace.annotations["capture_advisories"]`` kind recorded when a module boundary
#: adopts an untagged tensor as an internal source (postprocess
#: ``_warn_unattributed_tensor_args``).
ADVISORY_MODULE_BOUNDARY_ADOPTION = "module_boundary_adoption"
#: Advisory kind recorded when the module-held plain-tensor scan was cut
#: (``backends/torch/buffer_writes.warn_held_scan_truncated``).
ADVISORY_HELD_SCAN_TRUNCATED = "held_tensor_scan_truncated"


def append_capture_advisory(trace: Any, kind: str, entries: list[str]) -> None:
    """Persist one source-provenance gap advisory row on a capture trace.

    The row is written BEFORE any warning is raised (warning filters may raise) so
    the gap survives on the trace for forward validation's source-provenance check
    (``validation/_source_provenance.py``) and the report honesty preambles, never
    evaporating with a process-transient warning.

    Parameters
    ----------
    trace:
        Capture trace whose ``annotations`` carry the advisory family.
    kind:
        One of the gap advisory kinds defined in this module.
    entries:
        Human-readable gap descriptions; at least one.
    """

    annotations = getattr(trace, "annotations", None)
    if not isinstance(annotations, dict) or not entries:
        return
    rows = annotations.setdefault(ADVISORIES_ANNOTATIONS_KEY, [])
    if isinstance(rows, list):
        rows.append(
            {
                "kind": kind,
                "count": len(entries),
                "first_location": None,
                "message": "; ".join(entries),
            }
        )


def capture_verification(log: Any) -> dict[str, Any]:
    """Return the capture's verification/outcome facts, never inferred.

    The report layer's honesty contract (report/AGENTS.md) requires a rescued
    or ceilinged capture (``capture_verified=False``) to stay visible in every
    report surface.

    Parameters
    ----------
    log:
        Capture object (``Trace``-like; tolerant of partials and presenters).

    Returns
    -------
    dict[str, Any]
        ``capture_status`` (the settled ``CaptureOutcome`` status value, or
        ``"unknown"`` when the log carries none), tri-state
        ``capture_verified`` (``None`` = NOT RECORDED: the default capture does
        not arm the completeness witness, so the absence of a ceiling is not a
        clean bill -- ``True`` requires ``wrap_torch(completeness_witness=True)``
        or the ``tl.validate`` paths; load also degrades a claimed ``True`` to
        ``None``), ``capture_verification_reason``, and ``rescue_rerun``.
    """

    outcome = getattr(log, "outcome", None)
    status = getattr(outcome, "status", None)
    status_value = getattr(status, "value", None)
    return {
        "capture_status": str(status_value) if status_value is not None else "unknown",
        "capture_verified": getattr(log, "capture_verified", None),
        "capture_verification_reason": getattr(log, "capture_verification_reason", None),
        "rescue_rerun": bool(getattr(log, "rescue_rerun", None) or False),
    }


def poison_facts(log: Any) -> dict[str, Any]:
    """Return the sparse-run poison disclosure for a log.

    A poisoned Trace (``run(..., return_diverged=True)``) is returned
    specifically for inspection, so inspection surfaces must render it with
    the poison visible rather than refusing -- but a clean-looking render of a
    poisoned trace is silent wrongness.

    Parameters
    ----------
    log:
        Capture object.

    Returns
    -------
    dict[str, Any]
        ``poisoned`` plus, when poisoned, ``path_faithfulness`` and
        ``first_mismatch`` (stringified, bounded).
    """

    try:
        runnable = log._runnable
    except AttributeError:
        runnable = None
    poisoned = bool(getattr(runnable, "poisoned", False))
    facts: dict[str, Any] = {"poisoned": poisoned}
    if poisoned:
        status = getattr(runnable, "path_faithfulness", None)
        status_value = getattr(status, "value", status)
        facts["path_faithfulness"] = str(status_value) if status_value is not None else "unknown"
        mismatch = getattr(runnable, "first_mismatch", None)
        facts["first_mismatch"] = _bounded_text(mismatch) if mismatch is not None else None
    return facts


def episode_facts(log: Any) -> dict[str, Any] | None:
    """Return the episode-capture disclosure, or ``None`` for plain captures.

    The per-step ledger at ``annotations["episode"]`` is a DISCLOSURE, never a
    settlement authority; this projects its header facts (declared steps,
    token feed, fidelity basis -- ``"forced"`` marks the non-verifying
    teacher-forcing mode -- and any escalation) so report surfaces stop
    rendering an episode capture as an ordinary forward.

    Parameters
    ----------
    log:
        Capture object.

    Returns
    -------
    dict[str, Any] | None
        Header disclosure, or ``None`` when the log carries no episode ledger.
    """

    annotations = getattr(log, "annotations", None)
    if not isinstance(annotations, dict):
        return None
    payload = annotations.get("episode")
    if not isinstance(payload, dict):
        return None
    header = payload.get("header")
    if not isinstance(header, dict):
        return None
    facts = {
        "capture_kind": str(header.get("capture_kind", "episode")),
        "n_steps_declared": header.get("n_steps_declared"),
        "token_feed": header.get("token_feed"),
        "step_join": _step_join_fact(header),
        "step_output_kind": header.get("step_output_kind"),
        "step_output_from": header.get("step_output_from"),
        "fidelity_basis": header.get("fidelity_basis"),
        "escalated_from": header.get("escalated_from"),
        "escalation_reason": header.get("reason"),
    }
    rows = payload.get("rows")
    if isinstance(rows, list):
        facts["n_step_rows"] = len(rows)
    return facts


def _step_join_fact(header: dict[str, Any]) -> str:
    """Summarize THIS artifact's cross-step join evidence (lane F40c).

    Per-artifact, never the build switch: an absent ``step_join`` envelope
    (pre-measurement artifact, failed live measurement) reads
    ``"unmeasured"`` -- an unmeasured join can never present as measured.
    """

    envelope = header.get("step_join")
    if not isinstance(envelope, dict):
        return "unmeasured"
    break_step = envelope.get("break_step")
    grades = envelope.get("grades")
    if isinstance(break_step, int) and isinstance(grades, list):
        grade = grades[break_step] if break_step < len(grades) else None
        if grade == "declared":
            return f"declared_crossing_at_step_{break_step}"
        return f"broken_at_step_{break_step}"
    if isinstance(grades, list) and any(grade == "unchecked" for grade in grades):
        return "measured_with_unchecked_joins"
    return "continuous"


def capture_advisories(log: Any) -> list[dict[str, Any]]:
    """Return persisted capture-time advisories recorded on this log.

    Parameters
    ----------
    log:
        Capture object.

    Returns
    -------
    list[dict[str, Any]]
        Advisory records (``kind``, ``count``, ``first_location``,
        ``message``); empty when none were recorded.
    """

    annotations = getattr(log, "annotations", None)
    if not isinstance(annotations, dict):
        return []
    advisories = annotations.get(ADVISORIES_ANNOTATIONS_KEY)
    if not isinstance(advisories, list):
        return []
    return [dict(entry) for entry in advisories if isinstance(entry, dict)]


def intervention_facts(log: Any) -> dict[str, Any] | None:
    """Return the intervention disclosure for a log, or ``None`` when untouched.

    AUD-HONESTY M5: a capture-time ``intervene=`` (or a fork-side ``do()``) leaves
    the outcome ``complete`` while the retained values at the fired sites are
    COUNTERFACTUAL. Every honesty surface that reads
    :func:`capture_honesty_facts` must therefore carry the intervention evidence,
    or an agent reading a zero-ablated dump has no way to know the numbers are
    not the model's. Evidence is read from the persisted record, never inferred:
    per-op ``intervention_replaced`` flags and ``FireRecord`` lists, plus the
    ``intervention_audit`` rows (fork-side ``do()`` / ``PARAM`` / ``EVENT`` kinds).

    Parameters
    ----------
    log:
        Capture object.

    Returns
    -------
    dict[str, Any] | None
        ``{"intervened": True, "replaced_ops": [...], "replaced_op_count": n,
        "fire_count": n, "audit_row_count": n, "audit_kinds": [...]}`` when any
        intervention evidence exists; ``None`` for an untouched log.
    """

    replaced: list[str] = []
    fires = 0
    for layer in getattr(log, "layer_list", None) or ():
        records = getattr(layer, "interventions", None) or ()
        fires += len(records)
        if bool(getattr(layer, "intervention_replaced", False)):
            label = getattr(layer, "layer_label", None)
            if isinstance(label, str):
                replaced.append(label)
    audit = getattr(log, "intervention_audit", None)
    audit_rows = [row for row in (audit or ()) if isinstance(row, dict)]
    kinds = sorted({str(row.get("kind")) for row in audit_rows if row.get("kind") is not None})
    if not replaced and not fires and not audit_rows:
        return None
    return {
        "intervened": True,
        "replaced_ops": replaced,
        "replaced_op_count": len(replaced),
        "fire_count": int(fires),
        "audit_row_count": len(audit_rows),
        "audit_kinds": kinds,
    }


def capture_honesty_facts(log: Any) -> dict[str, Any]:
    """Return the full JSON-primitive honesty fact block for one log.

    Parameters
    ----------
    log:
        Capture object.

    Returns
    -------
    dict[str, Any]
        Schema-stamped facts: verification quartet, ``structure_only``,
        poison disclosure, episode disclosure (when present), intervention
        disclosure (when present, M5), and advisories (when present).
    """

    facts: dict[str, Any] = {
        "schema": CAPTURE_HONESTY_SCHEMA,
        **capture_verification(log),
        "structure_only": bool(getattr(log, "structure_only", False)),
        **poison_facts(log),
    }
    if facts["structure_only"]:
        facts.update(_structure_evidence_facts(log))
    episode = episode_facts(log)
    if episode is not None:
        facts["episode"] = episode
    interventions = intervention_facts(log)
    if interventions is not None:
        facts["interventions"] = interventions
    advisories = capture_advisories(log)
    if advisories:
        facts["advisories"] = advisories
    return facts


def _structure_evidence_facts(log: Any) -> dict[str, Any]:
    """Evidence-envelope + session claim-status facts for a structure-only log.

    ONE envelope, consumed by every surface (weightsfree memo sec 5): the
    persisted envelope supplies substrate/values_available/factory-policy
    facts; the SESSION discharge registry supplies the claim-status ladder
    (HYPOTHESIS / CORROBORATED / REFUTED) and the discharge projection — an
    agent that reads ``structure_only: true`` without ``claim_status`` cannot
    learn that a discharge already REFUTED these shapes, precisely the fact
    that should stop it.
    """

    from .capture.structure_only import claim_status_for, registered_discharge

    facts: dict[str, Any] = {
        "claim_status": claim_status_for(log).value,
        "values_available": False,
    }
    envelope = getattr(log, "structure_evidence", None)
    if isinstance(envelope, dict):
        facts["substrate"] = envelope.get("substrate")
        facts["structure_evidence"] = envelope
    discharge = registered_discharge(log)
    if discharge is not None:
        facts["discharge"] = {
            "verdict": discharge.verdict.value,
            "comparison_vocabulary": discharge.comparison_vocabulary,
            "structure_digest": discharge.structure_digest,
            "real_digest": discharge.real_digest,
            "claim_counts": dict(discharge.claim_counts),
            "first_contradiction": discharge.first_contradiction,
        }
    return facts


def _structure_only_status_lines(facts: dict[str, Any]) -> list[str]:
    """The HYPOTHESIS / CORROBORATED / REFUTED strength ladder (memo sec 5).

    HYPOTHESIS says shapes/dtypes and derived estimates are unproven;
    CORROBORATED names the discharge and STILL says no payloads exist;
    REFUTED is visually stronger (consumers refuse through the chokepoint).
    """

    status = facts.get("claim_status", "hypothesis")
    substrate = facts.get("substrate")
    substrate_note = f" ({substrate} substrate)" if substrate else ""
    if status == "refuted":
        discharge = facts.get("discharge") or {}
        first = discharge.get("first_contradiction") or "unrecorded contradiction"
        return [
            f"!! REFUTED structure-only capture{substrate_note}: a registered real-run "
            f"discharge contradicted these hypotheses -- {first}"
        ]
    if status == "corroborated":
        discharge = facts.get("discharge") or {}
        digest = str(discharge.get("real_digest") or "")[:16]
        return [
            f"structure-only capture{substrate_note}: hypotheses CORROBORATED by a real-run "
            f"discharge (oracle digest {digest}...); no tensor payloads exist"
        ]
    return [
        f"structure-only capture{substrate_note}: shapes/dtypes are hypotheses, not measurements"
    ]


def _verification_verdict_lines(facts: dict[str, Any]) -> list[str]:
    """Return the preamble's verification-verdict lines (header + ceiling/not-recorded).

    The ``capture_verified`` flag is tri-state: ``False`` names the ceiling reason,
    ``True`` reads ``true``, and ``None`` reads ``not_recorded`` with the arming hint
    (or, on a structure-only capture, the note that value-free captures are never verified).
    """

    verified = facts["capture_verified"]
    if verified is False:
        verified_text = "unverified"
    elif verified is True:
        verified_text = "true"
    else:
        # L10: "checked and clean" and "never checked" are indistinguishable once
        # the witness verdict is absent (default capture, or a load-degraded True),
        # so the preamble says so instead of printing a ``none`` that reads as a
        # recorded verdict.
        verified_text = "not_recorded"
    lines = [
        f"torchlens capture honesty: status={facts['capture_status']} verified={verified_text}"
    ]
    if verified is False:
        reason = facts["capture_verification_reason"] or "unrecorded reason"
        lines.append(f"verification ceiling: {reason}")
    elif verified is None and facts["structure_only"]:
        lines.append(
            "capture verification not recorded: a structure-only (value-free) capture is "
            "never verified, even with the completeness witness armed"
        )
    elif verified is None:
        lines.append(
            "capture verification not recorded: the completeness witness is not armed on "
            "the default capture (wrap_torch(completeness_witness=True) or tl.validate arm it)"
        )
    return lines


def honesty_preamble_lines(log: Any) -> list[str]:
    """Return the plain-text capture-honesty preamble for file exports.

    One shared preamble for every text-bearing export format (SVG/HTML
    comments, CSV comment lines, flamegraph zero-weight frame). Plain ASCII,
    bounded, no escape bytes.

    Parameters
    ----------
    log:
        Capture object.

    Returns
    -------
    list[str]
        Preamble lines (without comment markers; the exporter supplies its
        format's comment syntax).
    """

    facts = capture_honesty_facts(log)
    lines = _verification_verdict_lines(facts)
    if facts["rescue_rerun"]:
        lines.append("result from disclosed rescue re-run (mode_rescue_rerun)")
    if facts["structure_only"]:
        lines.extend(_structure_only_status_lines(facts))
    if facts["poisoned"]:
        lines.append(
            "POISONED sparse run: path_faithfulness="
            f"{facts.get('path_faithfulness', 'unknown')} -- values below are "
            "NOT model-faithful"
        )
    episode = facts.get("episode")
    if episode is not None:
        basis = episode.get("fidelity_basis")
        lines.append(
            f"episode capture: {episode.get('n_steps_declared')} declared step(s), "
            f"token_feed={episode.get('token_feed')}, fidelity_basis={basis}, "
            f"step_join={episode.get('step_join')}"
        )
        if basis == "forced":
            lines.append("forced-tokens episode: a disclosed NON-VERIFYING mode")
    interventions = facts.get("interventions")
    if interventions is not None:
        lines.append(
            "INTERVENED capture: "
            f"{interventions.get('fire_count')} fire(s) replaced "
            f"{interventions.get('replaced_op_count')} op(s) "
            f"{interventions.get('replaced_ops')}; audit rows "
            f"{interventions.get('audit_row_count')} {interventions.get('audit_kinds')} -- "
            "values at fired sites are COUNTERFACTUAL, not the model's"
        )
    for advisory in facts.get("advisories", ()):
        lines.append(
            f"capture advisory: {advisory.get('kind')} x{advisory.get('count')} "
            f"(first at {advisory.get('first_location') or 'unknown location'})"
        )
    return lines


def honesty_banner_lines(log: Any) -> list[str]:
    """Return honesty banner lines for NON-CLEAN captures only.

    A surface "opens with the honesty banner when the capture is non-clean":
    unverified/rescued/structure-only/poisoned/episode/advisory facts render;
    a clean verified plain capture returns no lines. Use this on presenter
    surfaces (``TraceSlice.summary``, ``draw`` captions) where a full status
    block would be noise but laundering the facts away is silent wrongness.

    Parameters
    ----------
    log:
        Capture object (or presenter's source trace).

    Returns
    -------
    list[str]
        Banner lines; empty when there is nothing non-clean to disclose.
    """

    facts = capture_honesty_facts(log)
    lines: list[str] = []
    if facts["capture_status"] not in ("complete", "unknown"):
        lines.append(f"capture outcome: {facts['capture_status']}")
    if facts["capture_verified"] is False:
        reason = facts["capture_verification_reason"] or "unrecorded reason"
        lines.append(f"capture UNVERIFIED ({reason}); treat contents as a lower bound")
    if facts["rescue_rerun"]:
        lines.append("result from disclosed rescue re-run (mode_rescue_rerun)")
    if facts["structure_only"]:
        lines.extend(_structure_only_status_lines(facts))
    if facts["poisoned"]:
        lines.append(
            "POISONED sparse run (path_faithfulness="
            f"{facts.get('path_faithfulness', 'unknown')}): values are NOT model-faithful"
        )
    episode = facts.get("episode")
    if episode is not None:
        basis = episode.get("fidelity_basis")
        suffix = "; forced-tokens NON-VERIFYING basis" if basis == "forced" else ""
        lines.append(
            f"episode capture ({episode.get('n_steps_declared')} declared step(s), "
            f"fidelity_basis={basis}, step_join={episode.get('step_join')}{suffix})"
        )
    for advisory in facts.get("advisories", ()):
        lines.append(
            f"capture advisory: {advisory.get('kind')} x{advisory.get('count')} "
            f"(first at {advisory.get('first_location') or 'unknown location'})"
        )
    return lines


def attach_dataframe_honesty(dataframe: Any, log: Any) -> Any:
    """Attach the shared honesty fact block to a DataFrame's ``attrs``.

    Parameters
    ----------
    dataframe:
        ``pandas.DataFrame`` (or anything exposing ``attrs``).
    log:
        Capture object the table was projected from.

    Returns
    -------
    Any
        The same DataFrame, with ``attrs["torchlens_capture_honesty"]`` set.
    """

    with contextlib.suppress(AttributeError, TypeError):
        dataframe.attrs[DATAFRAME_ATTRS_KEY] = capture_honesty_facts(log)
    return dataframe


#: Known non-Trace presenter types -> the executable spelling that answers the
#: caller's actual question. Keyed by class name; module-verified before use.
_PRESENTER_REMEDIES: dict[str, str] = {
    "MergedTrace": (
        "use the merged presenter's own surfaces (merged.to_markdown(), "
        "merged.to_pandas()) or report on one member rank Trace"
    ),
    "TraceSlice": (
        "report on the slice's source trace (tl.report.explain("
        "slice.source_trace)); the slice itself discloses via slice.summary()"
    ),
    "Bundle": "report on one member Trace (e.g. tl.report.explain(bundle[name]))",
    "Recording": ("cook the recording into a Trace first: tl.report.explain(recording.to_trace())"),
}


def refuse_presenter_subject(log: Any, surface: str) -> None:
    """Refuse known non-Trace presenters typed instead of reporting hollow facts.

    ``explain()``/``to_agent_json()`` read trace fields with permissive
    defaults, so a presenter (MergedTrace, TraceSlice, Bundle, Recording)
    used to produce a confidently WRONG report -- zero ops, zero params,
    status unknown -- instead of an answer or an error.

    Parameters
    ----------
    log:
        Report subject.
    surface:
        Human-readable calling surface name for the refusal message.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``report_subject_unsupported`` when the subject is a known
        torchlens presenter type rather than a Trace/PartialTrace.
    """

    type_name = type(log).__name__
    remedy = _PRESENTER_REMEDIES.get(type_name)
    if remedy is None:
        return
    module = type(log).__module__ or ""
    if not module.startswith("torchlens"):
        return
    from ._errors import InvalidArgumentError

    raise InvalidArgumentError(
        f"{surface} cannot report on a {type_name}: it is a presenter, not a "
        "captured Trace, and reading it as one produces a hollow wrong report "
        "(zero ops, zero params, unknown status).",
        code="report_subject_unsupported",
        remedy=remedy,
        argument="log",
    )


def _bounded_text(value: Any, limit: int = 500) -> str:
    """Return ``str(value)`` truncated to ``limit`` characters with disclosure."""

    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


__all__ = [
    "ADVISORIES_ANNOTATIONS_KEY",
    "CAPTURE_HONESTY_SCHEMA",
    "DATAFRAME_ATTRS_KEY",
    "attach_dataframe_honesty",
    "capture_advisories",
    "capture_honesty_facts",
    "capture_verification",
    "episode_facts",
    "honesty_banner_lines",
    "honesty_preamble_lines",
    "poison_facts",
    "refuse_presenter_subject",
]
