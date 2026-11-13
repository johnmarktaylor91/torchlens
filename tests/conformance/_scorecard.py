"""Backend scorecards (ecosystem MEMO section 4, build item B4; SPEC 4.5 v3).

Every backend publishes the highest tier, profile and scope it actually
earns, with unsupported cases in the denominator. A scorecard is one canonical record per
backend, folded from a full C0-C2 conformance run over that backend's roster.
It lists every case the runner defines for that roster, each in exactly one
state:

- ``passed``: the case executed and its independent oracle matched;
- ``failed``: the case executed and its oracle did not match, or it raised;
- ``unsupported``: the backend refused with a typed ``BackendUnsupportedError``;
- ``not_run``: the framework cannot run here (named reason), or an earlier
  failure for the same model stopped the case from being reached.

Every cell is in the denominator; only ``passed`` counts toward it. The
headline is the highest tier whose every cell passed. The record also carries
the framework requirement, the capability summary (``supported`` /
``unsupported`` / ``untested`` per flag), the roster group counts and the six
graduation items, kept as evidence; no record carries a maturity label
(version 3 retired it). A scorecard is never a claim:
its inputs are toys, it carries no claim string, and the word the claim
grammar reserves is redacted from any provider detail it records.

The shipped record is ``torchlens/backends/<backend>/scorecard.json``, written
and checked by ``python -m tests.conformance.scorecard`` on the pinned CI
leg; ``tl.report.doctor()`` reads it without running any framework.

Every spelling is DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, fields as dataclass_fields, replace
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from torchlens.backends._scorecard_rows import (
    read_shipped_scorecard,
    shipped_scorecard_path as product_shipped_scorecard_path,
)
from torchlens.backends.registry import registered_backend_specs

from ._adapters import RosterModel, resolve_adapter
from ._cases import TIER_C0, TIER_C1, TIER_C2, UNSUPPORTED_DETAIL_PREFIX
from ._graduation import GraduationItem, graduation_items
from ._protocol_cases import (
    C1_REFUSAL_PREFIX,
    adapter_protocol,
    c1_row_case_ids,
    c1_tail_case_ids,
    c2_row_case_ids,
    false_capability_flags,
)

if TYPE_CHECKING:
    from . import ConformanceReport

__tl_layer__ = "L9"

SCORECARD_VERSION = "torchlens.conformance.scorecard.v3"

#: The closed cell-state vocabulary, in canonical count order.
CELL_STATES: tuple[str, ...] = ("passed", "failed", "unsupported", "not_run")

#: The closed capability-summary vocabulary (FORKS F-P).
CAPABILITY_STATES: tuple[str, ...] = ("supported", "unsupported", "untested")

#: File name of the shipped record inside ``torchlens/backends/<backend>/``.
SCORECARD_FILENAME = "scorecard.json"

#: The stable backend whose scorecard is the graduation reference (SPEC 5, G1).
REFERENCE_BACKEND = "torch"

_TIERS: tuple[str, ...] = (TIER_C0, TIER_C1, TIER_C2)
_CLAIM_WORD = re.compile("conformant", flags=re.IGNORECASE)

#: Installed distribution that versions each backend's framework.
_FRAMEWORK_DISTRIBUTIONS: dict[str, tuple[str, ...]] = {
    "torch": ("torch",),
    "fake": ("torch",),
    "jax": ("jax",),
    # tensorflow-cpu (what the nightly tf leg installs) is its own distribution.
    "tf": ("tensorflow", "tensorflow-cpu"),
    "mlx": ("mlx",),
    "tinygrad": ("tinygrad",),
    "paddle": ("paddlepaddle",),
}


@dataclass(frozen=True)
class ScorecardCell:
    """One scorecard cell: a case for one roster family (``""`` for C0).

    Parameters
    ----------
    tier:
        ``"C0"``, ``"C1"`` or ``"C2"``.
    case_id:
        The runner's stable case identifier.
    family:
        Roster family the case ran against (``""`` for model-free cases).
    state:
        One of :data:`CELL_STATES`.
    detail:
        Failure, refusal or not-run reason (``""`` when passed).
    group:
        Roster group of the family (``"config_built"`` or ``"ecosystem"``;
        ``""`` for model-free cells).
    """

    tier: str
    case_id: str
    family: str
    state: str
    detail: str = ""
    group: str = ""


@dataclass(frozen=True)
class PreviewScorecard:
    """Canonical per-backend scorecard (record version 3; version 2 also carried a
    ``maturity`` label, retired 2026-11-12, which a reader ignores).

    Parameters
    ----------
    scorecard_version:
        :data:`SCORECARD_VERSION`.
    backend:
        Backend scored.
    framework_version:
        Installed framework distribution version, ``""`` when absent.
    suite_version:
        Conformance suite the cells came from.
    torchlens_version:
        TorchLens version that produced the run.
    cells:
        Every cell, in tier, family and case order.
    counts:
        Cells per state plus ``denominator`` (all cells).
    highest_tier:
        Highest tier whose every cell passed, cumulative from C0, else ``""``.
    framework_requirement:
        ``minimum``, ``below``, ``tested_through``, ``pinned`` and ``extra`` as
        strings (``""`` for an undeclared value); empty without a protocol.
    capabilities:
        Refusable capability flag -> one of :data:`CAPABILITY_STATES`.
    roster_groups:
        Roster rows per group.
    graduation:
        The six graduation items (SPEC 5): evidence, never a label.
    sha256:
        SHA-256 over the canonical bytes with this field excluded.
    """

    scorecard_version: str
    backend: str
    framework_version: str
    suite_version: str
    torchlens_version: str
    cells: tuple[ScorecardCell, ...]
    counts: dict[str, int]
    highest_tier: str
    framework_requirement: dict[str, str] = field(default_factory=dict)
    capabilities: dict[str, str] = field(default_factory=dict)
    roster_groups: dict[str, int] = field(default_factory=dict)
    graduation: tuple[GraduationItem, ...] = ()
    sha256: str = ""

    def to_json(self) -> str:
        """Render the canonical byte form (sorted keys, compact, ASCII)."""

        return _canonical_json(self, include_digest=True)


def _canonical_payload(card: PreviewScorecard) -> dict[str, object]:
    """The digest-free canonical payload of one scorecard."""

    return {
        "scorecard_version": card.scorecard_version,
        "backend": card.backend,
        "framework_version": card.framework_version,
        "suite_version": card.suite_version,
        "torchlens_version": card.torchlens_version,
        "cells": [
            {
                "tier": cell.tier,
                "case_id": cell.case_id,
                "family": cell.family,
                "state": cell.state,
                "detail": cell.detail,
                "group": cell.group,
            }
            for cell in card.cells
        ],
        "counts": dict(sorted(card.counts.items())),
        "highest_tier": card.highest_tier,
        "framework_requirement": dict(sorted(card.framework_requirement.items())),
        "capabilities": dict(sorted(card.capabilities.items())),
        "roster_groups": dict(sorted(card.roster_groups.items())),
        "graduation": [
            {"id": item.id, "met": item.met, "detail": item.detail} for item in card.graduation
        ],
    }


def _canonical_json(card: PreviewScorecard, *, include_digest: bool) -> str:
    """Serialize one scorecard canonically; the digest hashes the digest-free form."""

    payload = _canonical_payload(card)
    if include_digest:
        payload["sha256"] = card.sha256
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _with_digest(card: PreviewScorecard) -> PreviewScorecard:
    """The card with its digest computed over the digest-free canonical bytes."""

    digest = hashlib.sha256(_canonical_json(card, include_digest=False).encode("utf-8"))
    return replace(card, sha256=digest.hexdigest())


def scorecard_from_json(text: str) -> PreviewScorecard:
    """Parse one shipped record back into a :class:`PreviewScorecard`.

    Parameters
    ----------
    text:
        The canonical JSON text written by :meth:`PreviewScorecard.to_json`.

    Returns
    -------
    PreviewScorecard
        The record; its ``sha256`` is the stored digest, not recomputed.
    """

    from torchlens._io import _json

    data = _json.loads_bounded(text)
    cells = tuple(
        ScorecardCell(
            tier=str(cell["tier"]),
            case_id=str(cell["case_id"]),
            family=str(cell.get("family", "")),
            state=str(cell["state"]),
            detail=str(cell.get("detail", "")),
            group=str(cell.get("group", "")),
        )
        for cell in data.get("cells", ())
    )
    graduation = tuple(
        GraduationItem(
            id=str(item["id"]), met=bool(item["met"]), detail=str(item.get("detail", ""))
        )
        for item in data.get("graduation", ())
    )
    return PreviewScorecard(
        scorecard_version=str(data.get("scorecard_version", "")),
        backend=str(data.get("backend", "")),
        framework_version=str(data.get("framework_version", "")),
        suite_version=str(data.get("suite_version", "")),
        torchlens_version=str(data.get("torchlens_version", "")),
        cells=cells,
        counts={str(key): int(value) for key, value in data.get("counts", {}).items()},
        highest_tier=str(data.get("highest_tier", "")),
        framework_requirement={
            str(key): str(value) for key, value in data.get("framework_requirement", {}).items()
        },
        capabilities={str(key): str(value) for key, value in data.get("capabilities", {}).items()},
        roster_groups={
            str(key): int(value) for key, value in data.get("roster_groups", {}).items()
        },
        graduation=graduation,
        sha256=str(data.get("sha256", "")),
    )


def shipped_scorecard_path(backend: str) -> Path:
    """Where the backend's shipped record lives inside the installed package."""

    return product_shipped_scorecard_path(backend)


def shipped_scorecard(backend: str) -> PreviewScorecard | None:
    """The shipped record for ``backend``, or ``None`` when none ships.

    Parameters
    ----------
    backend:
        Registered backend name.

    Returns
    -------
    PreviewScorecard | None
        Parsed from ``torchlens/backends/<backend>/scorecard.json``; reading
        it runs no framework code.
    """

    path = shipped_scorecard_path(backend)
    if not path.is_file():
        return None
    return scorecard_from_json(path.read_text(encoding="utf-8"))


def reference_scorecard_for(backend: str, card: PreviewScorecard) -> PreviewScorecard | None:
    """The torch reference card G1 compares ``backend`` against.

    Parameters
    ----------
    backend:
        The backend being scored.
    card:
        Its freshly computed or shipped scorecard.

    Returns
    -------
    PreviewScorecard | None
        ``card`` itself when ``backend`` is the reference backend (torch is
        its own baseline), else the shipped torch record, or ``None`` when
        none ships.
    """

    if backend == REFERENCE_BACKEND:
        return card
    return shipped_scorecard(REFERENCE_BACKEND)


def _adapter_for(report: ConformanceReport, adapter: Any | None) -> Any | None:
    """The adapter the slots are enumerated for (resolved from the report when not given)."""

    if adapter is not None:
        return adapter
    try:
        return resolve_adapter(report.backend)
    except Exception:  # noqa: BLE001 - an unknown provider enumerates protocol-free slots
        return None


def _expected_slots(
    report: ConformanceReport, roster: tuple[RosterModel, ...], adapter: Any | None
) -> list[tuple[str, str, str, str]]:
    """Every (tier, case_id, family, group) slot a full run over ``roster`` defines."""

    slots = [("C0", row.case_id, "", "") for row in report.results if row.tier == "C0"]
    shell = adapter if adapter is not None else _ProtocolFreeAdapter()
    slots += [
        ("C1", case_id, model.family, model.group)
        for model in roster
        for case_id in c1_row_case_ids(shell, model)
    ]
    slots += [("C1", case_id, "", "") for case_id in c1_tail_case_ids(shell)]
    slots += [
        ("C2", case_id, model.family, model.group)
        for model in roster
        for case_id in c2_row_case_ids(shell, model)
    ]
    return slots


class _ProtocolFreeAdapter:
    """Slot enumeration stand-in for a provider without a protocol view."""

    name = ""


def _cell_for(
    slot: tuple[str, str, str, str],
    report: ConformanceReport,
    stopped_at: dict[tuple[str, str], str],
) -> ScorecardCell:
    """Fold the report row for one slot (or its absence) into a cell."""

    tier, case_id, family, group = slot
    for row in report.results:
        if (row.tier, row.case_id, row.family) != (tier, case_id, family):
            continue
        detail = _CLAIM_WORD.sub("[claim word]", row.detail)
        if row.outcome == "passed":
            return ScorecardCell(tier, case_id, family, "passed", group=group)
        if row.outcome == "skipped":
            return ScorecardCell(tier, case_id, family, "not_run", detail, group)
        if row.detail.startswith(UNSUPPORTED_DETAIL_PREFIX + "["):
            return ScorecardCell(tier, case_id, family, "unsupported", detail, group)
        return ScorecardCell(tier, case_id, family, "failed", detail, group)
    blocker = stopped_at.get((tier, family), "an earlier case")
    return ScorecardCell(
        tier, case_id, family, "not_run", f"not reached: {blocker} did not pass", group
    )


def _framework_requirement_strings(protocol: Any | None) -> dict[str, str]:
    """The protocol's framework requirement as record strings (``""`` for undeclared)."""

    requirement = getattr(protocol, "framework", None)
    if requirement is None:
        return {}
    return {
        key: "" if getattr(requirement, key) is None else str(getattr(requirement, key))
        for key in ("minimum", "below", "tested_through", "pinned", "extra")
    }


def capability_summary(protocol: Any | None, cells: tuple[ScorecardCell, ...]) -> dict[str, str]:
    """Refusable flag -> ``supported`` / ``unsupported`` / ``untested`` (SPEC 4.5).

    ``supported``: the flag is True and the consumption cell passed;
    ``unsupported``: the flag is False and its refusal cell passed;
    ``untested``: anything else (a declared flag whose evidence is missing).

    Parameters
    ----------
    protocol:
        The backend's protocol view, or ``None`` (empty summary).
    cells:
        The folded cells the evidence is read from.

    Returns
    -------
    dict[str, str]
        One entry per refusable capability flag.
    """

    from torchlens.backends._backend_protocol import REFUSABLE_CAPABILITY_FLAGS

    if protocol is None:
        return {}
    states = {(cell.tier, cell.case_id): cell.state for cell in cells if not cell.family}
    consumed = states.get(("C1", "c1_capabilities_consumed")) == "passed"
    false_flags = set(false_capability_flags(protocol))
    summary: dict[str, str] = {}
    for flag in REFUSABLE_CAPABILITY_FLAGS:
        if flag in false_flags:
            refused = states.get(("C1", f"{C1_REFUSAL_PREFIX}{flag}")) == "passed"
            summary[flag] = "unsupported" if refused else "untested"
        else:
            summary[flag] = "supported" if consumed else "untested"
    return summary


@dataclass(frozen=True)
class GraduationInputs:
    """What the graduation items of :func:`scorecard_from_report` compare against.

    Parameters
    ----------
    reference:
        The torch scorecard the graduation check G1 compares against; the
        shipped torch record when omitted, the card itself for torch.
    repo_root:
        Repository root for the file-based graduation items; auto-detected
        from the package location when omitted.
    """

    reference: PreviewScorecard | None = None
    repo_root: Path | None = None


def scorecard_from_report(
    report: ConformanceReport,
    roster: tuple[RosterModel, ...],
    *,
    framework_version: str,
    adapter: Any | None = None,
    graduation: GraduationInputs | None = None,
) -> PreviewScorecard:
    """Fold one full C0-C2 conformance report into a scorecard.

    Parameters
    ----------
    report:
        A report from ``run_conformance(..., tiers=("C0", "C1", "C2"))``.
    roster:
        The roster the run covered, in roster order.
    framework_version:
        Installed framework version (``""`` when absent).
    adapter:
        The adapter the run used; resolved from ``report.backend`` when omitted
        (a provider the runner cannot resolve enumerates protocol-free slots).
    graduation:
        The reference scorecard and repository root the graduation items use;
        both auto-detected when omitted.

    Returns
    -------
    PreviewScorecard
        The scorecard with its digest.
    """

    resolved = _adapter_for(report, adapter)
    protocol = adapter_protocol(resolved) if resolved is not None else None
    cells = _report_cells(report, roster, resolved)
    counts, highest = _cell_counts_and_highest_tier(cells)
    groups: dict[str, int] = {}
    for model in roster:
        groups[model.group] = groups.get(model.group, 0) + 1
    card = PreviewScorecard(
        scorecard_version=SCORECARD_VERSION,
        backend=report.backend,
        framework_version=framework_version,
        suite_version=report.suite_version,
        torchlens_version=report.environment.get("torchlens_version", ""),
        cells=cells,
        counts=counts,
        highest_tier=highest,
        framework_requirement=_framework_requirement_strings(protocol),
        capabilities=capability_summary(protocol, cells),
        roster_groups=groups,
    )
    if protocol is not None:
        card = _with_graduation(card, graduation)
    return _with_digest(card)


def _report_cells(
    report: ConformanceReport, roster: tuple[RosterModel, ...], resolved: Any | None
) -> tuple[ScorecardCell, ...]:
    """One cell per expected slot, each stopped at the row's first non-passing case."""

    stopped_at: dict[tuple[str, str], str] = {}
    for row in report.results:
        if row.outcome != "passed":
            stopped_at.setdefault((row.tier, row.family), row.case_id)
    return tuple(
        _cell_for(slot, report, stopped_at) for slot in _expected_slots(report, roster, resolved)
    )


def _cell_counts_and_highest_tier(cells: tuple[ScorecardCell, ...]) -> tuple[dict[str, int], str]:
    """Per-state cell counts (plus the denominator) and the highest fully-passed tier."""

    counts = {state: sum(1 for cell in cells if cell.state == state) for state in CELL_STATES}
    counts["denominator"] = len(cells)
    highest = ""
    for tier in _TIERS:
        tier_cells = [cell for cell in cells if cell.tier == tier]
        if not tier_cells or any(cell.state != "passed" for cell in tier_cells):
            break
        highest = tier
    return counts, highest


def _with_graduation(
    card: PreviewScorecard, graduation: GraduationInputs | None
) -> PreviewScorecard:
    """Attach the graduation items computed against the reference scorecard."""

    inputs = graduation or GraduationInputs()
    torch_card = inputs.reference
    if torch_card is None:
        torch_card = reference_scorecard_for(card.backend, card)
    return replace(
        card,
        graduation=graduation_items(card, reference=torch_card, repo_root=inputs.repo_root),
    )


def _framework_version(backend: str) -> str:
    """Installed version of the backend's framework distribution, else ``""``."""

    for distribution in _FRAMEWORK_DISTRIBUTIONS.get(backend, ()):
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
    return ""


def preview_scorecard(backend: str, *, workdir: str | Path | None = None) -> PreviewScorecard:
    """Run the full C0-C2 conformance packs for one backend and score them.

    Parameters
    ----------
    backend:
        A preview backend (``"jax"``, ``"tf"``, ``"mlx"``, ``"tinygrad"``,
        ``"paddle"``); ``"torch"`` scores the reference as a baseline.
    workdir:
        Scratch directory for C2 artifacts; a temp dir when omitted.

    Returns
    -------
    PreviewScorecard
        The canonical scorecard. An absent framework yields ``not_run``
        cells with the install remedy, never an error.

    Raises
    ------
    ConfigurationError
        ``conformance_backend_unknown`` for a name the runner cannot resolve.
    """

    return _preview_scorecard(backend, workdir=workdir)


def _preview_scorecard(
    backend: str, *, workdir: str | Path | None = None, repo_root: Path | None = None
) -> PreviewScorecard:
    """:func:`preview_scorecard` with an explicit repository root for the
    file-based graduation items (the CLI's ``--repo-root``)."""

    from . import _roster_for, run_conformance

    adapter = resolve_adapter(backend)
    roster = _roster_for(adapter, None)
    report = run_conformance(backend, tiers=_TIERS, roster=roster, workdir=workdir)
    return scorecard_from_report(
        report,
        roster,
        framework_version=_framework_version(backend),
        adapter=adapter,
        graduation=GraduationInputs(repo_root=repo_root),
    )


def scorecards_differ(shipped: PreviewScorecard, computed: PreviewScorecard) -> list[str]:
    """Where a recomputed scorecard disagrees with the shipped record.

    The TorchLens version the record was written under is not a measured
    fact of the backend, so it is excluded: the comparison is over the
    canonical payload with ``torchlens_version`` and the digest removed.

    Parameters
    ----------
    shipped:
        The committed record.
    computed:
        The freshly computed scorecard.

    Returns
    -------
    list[str]
        One line per differing top-level key or cell; empty when equal.
    """

    left, right = _canonical_payload(shipped), _canonical_payload(computed)
    left.pop("torchlens_version", None)
    right.pop("torchlens_version", None)
    lines: list[str] = []
    for key in sorted(set(left) | set(right)):
        if key == "cells":
            continue
        if left.get(key) != right.get(key):
            lines.append(f"{key}: shipped {left.get(key)!r} != computed {right.get(key)!r}")
    left_cells, right_cells = _cells_by_key(left), _cells_by_key(right)
    for cell_key in sorted(set(left_cells) | set(right_cells)):
        if left_cells.get(cell_key) != right_cells.get(cell_key):
            lines.append(
                f"cell {cell_key}: shipped {left_cells.get(cell_key, {}).get('state', 'absent')} != "
                f"computed {right_cells.get(cell_key, {}).get('state', 'absent')}"
            )
    return lines


def _cells_by_key(payload: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    """A canonical payload's cells keyed on (tier, case_id, family)."""

    cells = cast("list[dict[str, Any]]", payload.get("cells", []))
    return {(c["tier"], c["case_id"], c["family"]): c for c in cells}


def scorecard_from_record(record: dict[str, Any]) -> PreviewScorecard:
    """Rebuild a ``PreviewScorecard`` from a shipped record.

    Only fields the dataclass declares are passed, so a record written by a
    later scorecard version (extra keys) still loads; cells become
    ``ScorecardCell`` rows.
    """

    known = {field.name for field in dataclass_fields(PreviewScorecard)}
    payload = {key: value for key, value in record.items() if key in known}
    cell_fields = {field.name for field in dataclass_fields(ScorecardCell)}
    payload["cells"] = tuple(
        ScorecardCell(**{key: value for key, value in cell.items() if key in cell_fields})
        for cell in record.get("cells", ())
        if isinstance(cell, dict)
    )
    payload.setdefault("counts", {})
    payload.setdefault("highest_tier", "")
    for name in ("scorecard_version", "backend", "framework_version", "suite_version"):
        payload.setdefault(name, "")
    payload.setdefault("torchlens_version", "")
    return PreviewScorecard(**payload)


def backend_scorecards() -> tuple[PreviewScorecard, ...]:
    """The shipped records of every registered backend that has one."""

    cards = []
    for spec in registered_backend_specs():
        record = read_shipped_scorecard(spec.name)
        if record is not None:
            cards.append(scorecard_from_record(record))
    return tuple(cards)
