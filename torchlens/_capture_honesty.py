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
        ``capture_verified`` (``None`` = no ceiling recorded),
        ``capture_verification_reason``, and ``rescue_rerun``.
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

    runnable = getattr(log, "_runnable", None)
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
        "fidelity_basis": header.get("fidelity_basis"),
        "escalated_from": header.get("escalated_from"),
        "escalation_reason": header.get("reason"),
    }
    rows = payload.get("rows")
    if isinstance(rows, list):
        facts["n_step_rows"] = len(rows)
    return facts


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
        poison disclosure, episode disclosure (when present), and advisories
        (when present).
    """

    facts: dict[str, Any] = {
        "schema": CAPTURE_HONESTY_SCHEMA,
        **capture_verification(log),
        "structure_only": bool(getattr(log, "structure_only", False)),
        **poison_facts(log),
    }
    episode = episode_facts(log)
    if episode is not None:
        facts["episode"] = episode
    advisories = capture_advisories(log)
    if advisories:
        facts["advisories"] = advisories
    return facts


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
    verified = facts["capture_verified"]
    verified_text = "unverified" if verified is False else str(verified).lower()
    lines = [
        f"torchlens capture honesty: status={facts['capture_status']} verified={verified_text}"
    ]
    if verified is False:
        reason = facts["capture_verification_reason"] or "unrecorded reason"
        lines.append(f"verification ceiling: {reason}")
    if facts["rescue_rerun"]:
        lines.append("result from disclosed rescue re-run (mode_rescue_rerun)")
    if facts["structure_only"]:
        lines.append("structure-only capture: shapes/dtypes are hypotheses, not measurements")
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
            f"token_feed={episode.get('token_feed')}, fidelity_basis={basis}"
        )
        if basis == "forced":
            lines.append("forced-tokens episode: a disclosed NON-VERIFYING mode")
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
        lines.append("structure-only capture: shapes/dtypes are hypotheses")
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
            f"fidelity_basis={basis}{suffix})"
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
