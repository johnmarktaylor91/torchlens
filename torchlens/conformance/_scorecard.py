"""Preview scorecards (ecosystem MEMO section 4, build item B4).

"Preview backends publish the highest tier/profile/scope they actually earn,
unsupported cases in the denominator." A scorecard is one canonical record per
backend, folded from a full C0-C2 conformance run over that backend's
config-built roster. It lists every case the runner defines for that roster,
each in exactly one state:

- ``passed``: the case executed and its independent oracle matched;
- ``failed``: the case executed and its oracle did not match, or it raised;
- ``unsupported``: the backend refused with a typed ``BackendUnsupportedError``;
- ``not_run``: the framework cannot run here (named reason), or an earlier
  failure for the same model stopped the case from being reached.

Every cell is in the denominator; only ``passed`` counts toward it. The
headline is the highest tier whose every cell passed. A scorecard is never a
claim: its inputs are toys, it carries no claim string, and the word the claim
grammar reserves is redacted from any provider detail it records.

Every spelling is DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

from ._adapters import RosterModel, resolve_adapter
from ._cases import C1_MODEL_CASES, C2_MODEL_CASES, UNSUPPORTED_DETAIL_PREFIX, model_case_ids

if TYPE_CHECKING:
    from . import ConformanceReport

__tl_layer__ = "L9"

SCORECARD_VERSION = "torchlens.conformance.preview_scorecard.v1"

#: The closed cell-state vocabulary, in canonical count order.
CELL_STATES: tuple[str, ...] = ("passed", "failed", "unsupported", "not_run")

_TIERS: tuple[str, ...] = ("C0", "C1", "C2")
_CLAIM_WORD = re.compile("conformant", flags=re.IGNORECASE)

#: Installed distribution that versions each backend's framework.
_FRAMEWORK_DISTRIBUTIONS: dict[str, str] = {
    "torch": "torch",
    "fake": "torch",
    "jax": "jax",
    "tf": "tensorflow",
    "mlx": "mlx",
    "tinygrad": "tinygrad",
    "paddle": "paddlepaddle",
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
    """

    tier: str
    case_id: str
    family: str
    state: str
    detail: str = ""


@dataclass(frozen=True)
class PreviewScorecard:
    """Canonical per-backend scorecard.

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
    sha256: str = ""

    def to_json(self) -> str:
        """Render the canonical byte form (sorted keys, compact, ASCII)."""

        return _canonical_json(self, include_digest=True)


def _canonical_json(card: PreviewScorecard, *, include_digest: bool) -> str:
    """Serialize one scorecard canonically; the digest hashes the digest-free form."""

    payload: dict[str, object] = {
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
            }
            for cell in card.cells
        ],
        "counts": dict(sorted(card.counts.items())),
        "highest_tier": card.highest_tier,
    }
    if include_digest:
        payload["sha256"] = card.sha256
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _expected_slots(
    report: ConformanceReport, roster: tuple[RosterModel, ...]
) -> list[tuple[str, str, str]]:
    """Every (tier, case_id, family) slot a full run over ``roster`` defines."""

    slots = [("C0", row.case_id, "") for row in report.results if row.tier == "C0"]
    slots += [
        ("C1", case_id, model.family)
        for model in roster
        for case_id in model_case_ids(model, "C1", C1_MODEL_CASES)
    ]
    slots.append(("C1", "c1_capabilities_consumed", ""))
    slots += [("C2", case_id, model.family) for model in roster for case_id in C2_MODEL_CASES]
    return slots


def _cell_for(
    slot: tuple[str, str, str], report: ConformanceReport, stopped_at: dict[tuple[str, str], str]
) -> ScorecardCell:
    """Fold the report row for one slot (or its absence) into a cell."""

    tier, case_id, family = slot
    for row in report.results:
        if (row.tier, row.case_id, row.family) != slot:
            continue
        detail = _CLAIM_WORD.sub("[claim word]", row.detail)
        if row.outcome == "passed":
            return ScorecardCell(tier, case_id, family, "passed")
        if row.outcome == "skipped":
            return ScorecardCell(tier, case_id, family, "not_run", detail)
        if row.detail.startswith(UNSUPPORTED_DETAIL_PREFIX + "["):
            return ScorecardCell(tier, case_id, family, "unsupported", detail)
        return ScorecardCell(tier, case_id, family, "failed", detail)
    blocker = stopped_at.get((tier, family), "an earlier case")
    return ScorecardCell(tier, case_id, family, "not_run", f"not reached: {blocker} did not pass")


def scorecard_from_report(
    report: ConformanceReport, roster: tuple[RosterModel, ...], *, framework_version: str
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

    Returns
    -------
    PreviewScorecard
        The scorecard with its digest.
    """

    stopped_at: dict[tuple[str, str], str] = {}
    for row in report.results:
        if row.outcome != "passed":
            stopped_at.setdefault((row.tier, row.family), row.case_id)
    cells = tuple(_cell_for(slot, report, stopped_at) for slot in _expected_slots(report, roster))
    counts = {state: sum(1 for cell in cells if cell.state == state) for state in CELL_STATES}
    counts["denominator"] = len(cells)
    highest = ""
    for tier in _TIERS:
        tier_cells = [cell for cell in cells if cell.tier == tier]
        if not tier_cells or any(cell.state != "passed" for cell in tier_cells):
            break
        highest = tier
    card = PreviewScorecard(
        scorecard_version=SCORECARD_VERSION,
        backend=report.backend,
        framework_version=framework_version,
        suite_version=report.suite_version,
        torchlens_version=report.environment.get("torchlens_version", ""),
        cells=cells,
        counts=counts,
        highest_tier=highest,
    )
    digest = hashlib.sha256(_canonical_json(card, include_digest=False).encode("utf-8"))
    return replace(card, sha256=digest.hexdigest())


def _framework_version(backend: str) -> str:
    """Installed version of the backend's framework distribution, else ``""``."""

    distribution = _FRAMEWORK_DISTRIBUTIONS.get(backend)
    if distribution is None:
        return ""
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
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

    from . import _roster_for, run_conformance

    roster = _roster_for(resolve_adapter(backend), None)
    report = run_conformance(backend, tiers=_TIERS, roster=roster, workdir=workdir)
    return scorecard_from_report(report, roster, framework_version=_framework_version(backend))
