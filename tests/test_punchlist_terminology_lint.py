"""Terminology lint -- the glossary's first mechanical enforcement (SPEC, normative).

The completeness audit found the glossary LOCKED governance-side with zero
mechanical enforcement (DIGEST-AUDIT 3b, miss r2-MISS-20). This module is the
structural fix: the terminology rules that are already ruled and mechanically
checkable are ENFORCED here; the rules that await the naming sprint are
SPECIFIED here (so the toolkit exists the day the canon is ruled) but not yet
armed. The lint runs over the SHIPPED tree -- docs, audit-text sources,
caption sources -- never over a fixture.

NORMATIVE RULES (armed now)
===========================

R1. BANNED EXECUTION VERBS. TorchLens intervention engines substitute values;
    they never remove execution. No shipped doc page, audit-record text, or
    render caption may claim a block/module/layer/op was "skipped",
    "deleted", or "removed" by an intervention (honesty-vocabulary ownership:
    F01 writes the closed ``execution_effect`` value, THIS lint polices the
    wording, F43 renders it). Graph-RECORD removal (orphan scrub, collapse)
    is a different, legitimate sense and is either out of pattern reach or
    carried in the audited allowlist ledger below.

R2. THREE AXES STAY APART. Member settlement/capability (``CaptureStatus``:
    complete/halted/aborted_nonfinite/failed/unattested/unknown), relation
    claim grade (``RELATION_CLAIM_GRADES``: verified/consistent/disclosed/
    unchecked/divergent), and protocol scope (VERIFIED/ATTESTED/LINKED/
    REFUSE; not yet shipped) are three orthogonal vocabularies that never
    map onto each other and never trade members. MergedTrace's frozen enums
    stay domain-specific: they are pinned exactly and are NOT re-graded into
    any of the three axes.

R3. EXECUTION-EFFECT VOCABULARY CLOSURE. When the surgery engines land the
    persisted ``execution_effect`` disclosure (lane F01), its value set is
    exactly the two ruled members: ``values_replaced_after_execution`` and
    ``exits_substituted_interior_not_replayed``. Until then no partial or
    renamed spelling of the vocabulary may ship (the probe below fails on
    any near-miss attribute).

SPECIFIED, NOT YET ARMED (naming-sprint slate; arm each rule the day its
canon is ruled, as a new group in the R1 scanner)
=================================================

- trace-vs-log product noun (canon = Trace; printed header + README bindings).
- pass/call/op unit words on output surfaces (lookup errors, print(trace),
  explain -- the audit's fix-now rows are owned by their surface lanes; the
  LINT rule arms once the canon noun per unit is ruled).
- layer's three referents; op/operation/node; label/name/address/key;
  Bundle-vs-artifact; hook/edit/helper/spec/recipe; the save= four senses.
- retired-verb teaching (replay/rerun in prose) is already enforced
  repo-wide by tests/test_removed_spelling_lint.py and is deliberately not
  duplicated here.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

_SCAN_SUFFIXES = (".py", ".md", ".ipynb", ".rst", ".txt")

#: R1 scan scope: the shipped docs tree plus the audit-text and caption
#: sources (intervention audit strings live in torchlens/intervention and
#: torchlens/bundle; render captions in torchlens/visualization).
_R1_SCOPES = (
    "docs/",
    "README.md",
    "torchlens/intervention/",
    "torchlens/bundle/",
    "torchlens/visualization/",
)

#: R1 patterns: an execution subject claimed skipped/deleted/removed.
_R1_FORBIDDEN: dict[str, re.Pattern[str]] = {
    # "the block was skipped", "modules are removed", ...
    "execution_claimed_absent": re.compile(
        r"\b(?:block|module|layer|op|node|branch|execution|call|forward)s?\s+"
        r"(?:was|were|is|are|be|been|being|gets?|got)\s+"
        r"(?:completely\s+|entirely\s+|silently\s+)?(?:skipped|deleted|removed)\b"
    ),
    # "skips the block", "deletes this module", "removed the layer", ...
    "verb_targets_execution": re.compile(
        r"\b(?:skips?|skipped|deletes?|deleted|removes?|removed)\s+"
        r"(?:the|this|that|its|a|an)\s+"
        r"(?:block|module|layer|op|node|branch|execution|call)s?\b"
    ),
}

#: Plain-substring prefilter needles (C-level ``in`` scans keep the lint
#: inside the smoke budget; the red-capability test routes through the same
#: filter so needle/regex drift fails loudly).
_R1_NEEDLES: dict[str, tuple[str, ...]] = {
    "execution_claimed_absent": ("skipped", "deleted", "removed"),
    "verb_targets_execution": ("skip", "delete", "remove"),
}

#: Audited-legitimate ledger: path -> (groups, reason). Every row must still
#: match (a stale row fails the lint until pruned).
_R1_ALLOWED: dict[str, tuple[frozenset[str], str]] = {
    "docs/reference/capture_outcomes.md": (
        frozenset({"execution_claimed_absent"}),
        "graph-RECORD sense: 'events whose op was removed by' describes "
        "trace-record scrub (orphan/dedup removal), not an intervention "
        "claiming execution did not happen",
    ),
    "tests/test_punchlist_terminology_lint.py": (
        frozenset({"ALL"}),
        "this lint's own spec and ledger",
    ),
}

#: R2/R3 closed vocabularies (value strings, case-sensitive).
_CAPTURE_STATUS_VALUES = frozenset(
    {"complete", "halted", "aborted_nonfinite", "failed", "unattested", "unknown"}
)
_RELATION_CLAIM_GRADE_VALUES = frozenset(
    {"verified", "consistent", "disclosed", "unchecked", "divergent"}
)
#: Protocol-scope labels (foldB D12); the vocabulary has no shipped home yet.
_PROTOCOL_SCOPE_VALUES = frozenset({"VERIFIED", "ATTESTED", "LINKED", "REFUSE"})

_EXECUTION_EFFECT_VALUES = frozenset(
    {"values_replaced_after_execution", "exits_substituted_interior_not_replayed"}
)


def _in_r1_scope(rel: str) -> bool:
    """Return whether a tracked path is inside the R1 sweep boundary."""

    if not rel.endswith(_SCAN_SUFFIXES):
        return False
    return (
        any(rel == scope or rel.startswith(scope) for scope in _R1_SCOPES)
        or rel == "tests/test_punchlist_terminology_lint.py"
    )


def _tracked_r1_files() -> list[Path]:
    """Return the tracked text files in R1 scope."""

    try:
        listing = subprocess.run(
            ["git", "ls-files"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - non-checkout run
        pytest.skip("terminology lint requires a git checkout")
    return [_REPO_ROOT / rel for rel in listing.splitlines() if _in_r1_scope(rel)]


def _scan_text(text: str) -> list[tuple[str, int, str]]:
    """Return ``(group, line_number, line)`` R1 violations for one text."""

    violations: list[tuple[str, int, str]] = []
    for group, pattern in _R1_FORBIDDEN.items():
        if not any(needle in text for needle in _R1_NEEDLES[group]):
            continue
        for match in pattern.finditer(text):
            line_start = text.rfind("\n", 0, match.start()) + 1
            line_end = text.find("\n", match.start())
            line = text[line_start : line_end if line_end != -1 else len(text)]
            line_number = text.count("\n", 0, match.start()) + 1
            violations.append((group, line_number, line.strip()))
    return violations


@pytest.mark.smoke
def test_no_banned_execution_verbs() -> None:
    """R1: no shipped doc/audit/caption text claims execution was removed."""

    offenders: list[str] = []
    matched: dict[str, set[str]] = {}
    for path in _tracked_r1_files():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except (OSError, IsADirectoryError):  # pragma: no cover - racing deletion
            continue
        hits = _scan_text(text)
        if not hits:
            continue
        rel = path.relative_to(_REPO_ROOT).as_posix()
        allowed_groups, _reason = _R1_ALLOWED.get(rel, (frozenset(), ""))
        for group, line_number, line in hits:
            if "ALL" in allowed_groups or group in allowed_groups:
                matched.setdefault(rel, set()).add(group)
                continue
            offenders.append(f"{rel}:{line_number} [{group}] {line[:160]}")
    assert not offenders, (
        "banned execution verb: interventions substitute values, they never "
        "skip/delete/remove execution (say what was substituted instead, or "
        "add an audited ledger row here for a legitimate graph-RECORD sense):"
        "\n    " + "\n    ".join(offenders)
    )

    stale = [
        f"{rel} (reason: {_R1_ALLOWED[rel][1]})"
        for rel in _R1_ALLOWED
        if rel not in matched and rel != "tests/test_punchlist_terminology_lint.py"
    ]
    assert not stale, (
        "stale terminology-lint allowlist rows; prune them so the ledger "
        "stays honest:\n    " + "\n    ".join(stale)
    )


@pytest.mark.smoke
def test_banned_execution_verb_scanner_is_red_capable(tmp_path: Path) -> None:
    """R1 scanner actually detects a planted violation per group shape."""

    planted = {
        "execution_claimed_absent": "After the edit, the block was skipped entirely.\n",
        "verb_targets_execution": "tl.bypass deletes the module from the forward pass.\n",
    }
    assert set(planted) == set(_R1_FORBIDDEN)
    assert set(_R1_NEEDLES) == set(_R1_FORBIDDEN), "every group needs a needle row"
    for group, snippet in planted.items():
        hits = _scan_text(snippet)
        assert any(hit_group == group for hit_group, _, _ in hits), (
            f"scanner failed to flag the planted {group} violation: {snippet!r}"
        )

    clean = "The edit substituted the boundary value; downstream ops re-executed.\n"
    assert not _scan_text(clean), "substitution vocabulary must not false-positive"


@pytest.mark.smoke
def test_three_axes_stay_apart() -> None:
    """R2: settlement, relation grade, and protocol scope never trade members."""

    from torchlens.bundle._relations import RELATION_CLAIM_GRADES
    from torchlens.capture.outcome import CaptureStatus

    capture_values = {member.value for member in CaptureStatus}
    assert capture_values == _CAPTURE_STATUS_VALUES, (
        "CaptureStatus grew or lost a member without re-adjudicating the "
        "three-axes rule (foldB D12): settlement vocabulary is closed"
    )
    assert frozenset(RELATION_CLAIM_GRADES) == _RELATION_CLAIM_GRADE_VALUES, (
        "RELATION_CLAIM_GRADES drifted from the closed C07X vocabulary"
    )

    # No axis may absorb a member of another (users extend the RELATION
    # vocabulary, never the TRUST vocabulary; nothing re-grades settlement).
    assert not capture_values & _RELATION_CLAIM_GRADE_VALUES
    assert not capture_values & _PROTOCOL_SCOPE_VALUES
    assert not _RELATION_CLAIM_GRADE_VALUES & _PROTOCOL_SCOPE_VALUES


@pytest.mark.smoke
def test_merged_trace_vocabulary_stays_domain_specific() -> None:
    """R2: MergedTrace's frozen enums are pinned, never re-graded."""

    from torchlens.merged import (
        BoundaryConsistency,
        MergeAlignment,
        MergedErrorCode,
        MergeValueStatus,
    )

    assert {m.value for m in MergeAlignment} == {"aligned", "partial", "conflicted"}
    assert {m.value for m in BoundaryConsistency} == {
        "attested",
        "mismatched",
        "not_applicable",
        "not_present",
    }
    assert {m.value for m in MergeValueStatus} == {
        "divergent",
        "attested_complete",
        "attested_partial",
        "unwitnessed",
    }
    # MergedTrace never absorbs the settlement or grade axes wholesale: its
    # vocabularies answer merge-domain questions ("did the join witness
    # attest?"), not member settlement or relation-claim strength. Word
    # overlap on a single member (e.g. "divergent") is domain reuse, not a
    # mapping; the pin above is what prevents silent re-grading.
    merged_values = (
        {m.value for m in MergeAlignment}
        | {m.value for m in BoundaryConsistency}
        | {m.value for m in MergeValueStatus}
        | {m.value for m in MergedErrorCode}
    )
    assert not merged_values >= _CAPTURE_STATUS_VALUES
    assert not merged_values >= _RELATION_CLAIM_GRADE_VALUES


@pytest.mark.smoke
def test_execution_effect_vocabulary_closed_when_shipped() -> None:
    """R3: the execution-effect disclosure ships whole or not at all."""

    import torchlens.intervention as intervention

    candidates = [
        attr
        for attr in dir(intervention)
        if "execution_effect" in attr.lower() or attr == "EXECUTION_EFFECT_VALUES"
    ]
    if not candidates:
        # Lane F01 has not landed the writer yet; the ruled vocabulary is
        # specified above and this probe arms the day any spelling appears.
        return
    for attr in candidates:
        vocab = getattr(intervention, attr)
        values = {getattr(v, "value", v) for v in vocab}
        assert values == _EXECUTION_EFFECT_VALUES, (
            f"torchlens.intervention.{attr} must carry exactly the two ruled "
            f"execution_effect members, got {sorted(values)!r}"
        )
