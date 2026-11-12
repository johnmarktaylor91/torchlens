"""Graduation criteria as checkable items (SPEC section 5).

Six ``GraduationItem`` rows turn the bar for a complete backend into
predicates over the backend's scorecard, the torch reference scorecard and
the repository. The report is evidence, not a label: no backend carries a
maturity label (retired 2026-11-12), the report is embedded in the scorecard
record, and ``python -m tests.conformance.scorecard --backend <b>
--graduation`` prints it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._cases import TIER_C1, TIER_C2
from ._protocol_cases import C1_REFUSAL_OPTION_PREFIX, C1_REFUSAL_PREFIX

if TYPE_CHECKING:
    from ._scorecard import PreviewScorecard

__tl_layer__ = "L9"

#: The six criteria, in report order.
GRADUATION_IDS: tuple[str, ...] = ("G1", "G2", "G3", "G4", "G5", "G6")

#: Validation cells G2 requires passed on every per-change row.
_G2_CELLS: tuple[str, ...] = (
    "c1_validation_passes",
    "c1_validation_names_reason",
    "c2_validation_after_load",
)
#: Ecosystem families G3 requires with every c1 and c2 cell passed.
_G3_MIN_FAMILIES = 2
#: Per-change roster group (the graduation evidence group).
_PER_CHANGE_GROUP = "config_built"
_ECOSYSTEM_GROUP = "ecosystem"
_UNSTABLE_MARKER = "DOCUMENTED-UNSTABLE"


@dataclass(frozen=True)
class GraduationItem:
    """One graduation criterion and whether the evidence meets it.

    Parameters
    ----------
    id:
        ``"G1"`` to ``"G6"``.
    met:
        Whether the check holds on the evidence given.
    detail:
        What was checked, and what is missing when ``met`` is False.
    """

    id: str
    met: bool
    detail: str


@dataclass(frozen=True)
class GraduationReport:
    """The six items for one backend.

    Parameters
    ----------
    backend:
        Backend the items were computed for.
    items:
        The items in :data:`GRADUATION_IDS` order.
    """

    backend: str
    items: tuple[GraduationItem, ...]

    @property
    def met(self) -> bool:
        """Whether every item is met."""

        return all(item.met for item in self.items)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""

        return {
            "backend": self.backend,
            "met": self.met,
            "items": [
                {"id": item.id, "met": item.met, "detail": item.detail} for item in self.items
            ],
        }

    def render(self) -> str:
        """One line per item for the CLI."""

        lines = [f"graduation for {self.backend}: {'met' if self.met else 'not met'}"]
        lines += [
            f"  {item.id} {'met    ' if item.met else 'not met'} {item.detail}"
            for item in self.items
        ]
        return "\n".join(lines)


def repository_root(start: Path | None = None) -> Path | None:
    """The TorchLens repository root above ``start`` (the package by default), else ``None``.

    An installed wheel has no repository, so the file-based items read
    "repository files unavailable" there instead of guessing.
    """

    here = (start or Path(__file__)).resolve()
    for candidate in (here, *here.parents):
        marker = candidate / "pyproject.toml"
        if marker.is_file() and 'name = "torchlens"' in marker.read_text(encoding="utf-8"):
            return candidate
    return None


def _per_change_cells(card: PreviewScorecard) -> list[Any]:
    """Cells of the per-change group plus every model-free cell."""

    return [cell for cell in card.cells if cell.group in (_PER_CHANGE_GROUP, "")]


def _g1(card: PreviewScorecard, reference: PreviewScorecard | None) -> GraduationItem:
    """Passes the shared suite torch passes; every unsupported capability is a typed refusal."""

    cells = _per_change_cells(card)
    failed = [cell.case_id for cell in cells if cell.state == "failed"]
    not_run = [cell.case_id for cell in cells if cell.state == "not_run"]
    problems: list[str] = []
    if failed:
        problems.append(f"{len(failed)} failed cells (first {failed[0]})")
    if not_run:
        problems.append(f"{len(not_run)} not_run cells (first {not_run[0]})")
    if reference is None:
        problems.append("no torch reference scorecard to compare against")
    else:
        own: dict[str, set[str]] = {}
        for cell in cells:
            own.setdefault(cell.case_id, set()).add(cell.state)
        torch_passed = {cell.case_id for cell in reference.cells if cell.state == "passed"}
        missing = sorted(
            case_id for case_id in torch_passed if not _covered(case_id, own, card.capabilities)
        )
        if missing:
            problems.append(f"torch-passed cells not passed or unsupported here: {missing[:4]}")
    untested = sorted(flag for flag, state in card.capabilities.items() if state == "untested")
    if untested:
        problems.append(f"capabilities untested: {untested}")
    if problems:
        return GraduationItem("G1", False, "; ".join(problems))
    return GraduationItem(
        "G1",
        True,
        f"{len(cells)} per-change cells passed or typed-unsupported; torch's cells covered",
    )


def _covered(case_id: str, own: dict[str, set[str]], capabilities: dict[str, str]) -> bool:
    """Whether the backend covers one cell torch passed.

    The backend's own cell must be passed or typed-unsupported. A torch refusal
    cell (``c1_refusal_<flag>``, torch refusing a capability it lacks) is also
    covered when the backend supports that capability: a backend with the
    capability has no refusal to test, so the cell is absent, not missing.
    """

    states = own.get(case_id)
    if states is not None:
        return states <= {"passed", "unsupported"}
    if case_id.startswith(C1_REFUSAL_PREFIX) and not case_id.startswith(C1_REFUSAL_OPTION_PREFIX):
        return capabilities.get(case_id[len(C1_REFUSAL_PREFIX) :]) == "supported"
    return False


def _g2(card: PreviewScorecard) -> GraduationItem:
    """Replay validation works: the flag is supported and every validation cell passed."""

    problems: list[str] = []
    if card.capabilities.get("has_validation_replay") != "supported":
        problems.append("has_validation_replay is not supported")
    rows = [cell for cell in card.cells if cell.group == _PER_CHANGE_GROUP]
    for case_id in _G2_CELLS:
        states = [cell.state for cell in rows if cell.case_id == case_id]
        if not states:
            problems.append(f"{case_id} has no per-change cell")
        elif any(state != "passed" for state in states):
            problems.append(f"{case_id} not passed on every per-change row")
    if problems:
        return GraduationItem("G2", False, "; ".join(problems))
    return GraduationItem("G2", True, "validation passes, names a tampered op and survives reload")


def _g3(card: PreviewScorecard) -> GraduationItem:
    """Real models from the backend's ecosystem are captured and validated."""

    families: dict[str, list[str]] = {}
    for cell in card.cells:
        if cell.group == _ECOSYSTEM_GROUP and cell.tier in (TIER_C1, TIER_C2):
            families.setdefault(cell.family, []).append(cell.state)
    clean = sorted(name for name, states in families.items() if all(s == "passed" for s in states))
    if len(clean) >= _G3_MIN_FAMILIES:
        return GraduationItem("G3", True, f"ecosystem families fully passed: {clean}")
    return GraduationItem(
        "G3",
        False,
        f"{len(clean)} ecosystem families with every c1 and c2 cell passed "
        f"(need {_G3_MIN_FAMILIES}); roster declares {card.roster_groups.get(_ECOSYSTEM_GROUP, 0)}",
    )


def _g4(card: PreviewScorecard, repo_root: Path | None) -> GraduationItem:
    """Per-change CI against the pinned version, with a stated supported range."""

    requirement = card.framework_requirement
    problems: list[str] = []
    consistent = any(
        cell.case_id == "c0_framework_requirement_consistent" and cell.state == "passed"
        for cell in card.cells
    )
    if not consistent:
        problems.append("c0_framework_requirement_consistent did not pass")
    if not requirement.get("below") and not requirement.get("tested_through"):
        problems.append("neither below nor tested_through is stated")
    pinned = requirement.get("pinned", "")
    if not pinned:
        problems.append("no pinned version")
    workflows = _workflow_files(repo_root)
    hits: list[str] = []
    if workflows is None:
        problems.append("repository files unavailable")
    elif pinned:
        hits = [
            path.name
            for path in workflows
            if _workflow_installs(path.read_text(encoding="utf-8"), card.backend, pinned)
        ]
        if not hits:
            problems.append(f"no pull_request workflow installs {card.backend} at {pinned}")
    if problems:
        return GraduationItem("G4", False, "; ".join(problems))
    return GraduationItem("G4", True, f"{hits[0]} runs {card.backend} at {pinned} on pull_request")


def _workflow_files(repo_root: Path | None) -> list[Path] | None:
    """Workflow files under the repository, or ``None`` without a repository."""

    if repo_root is None or not repo_root.is_dir():
        return None
    folder = repo_root / ".github" / "workflows"
    if not folder.is_dir():
        return []
    return sorted(path for path in folder.iterdir() if path.suffix in (".yml", ".yaml"))


def _workflow_installs(text: str, backend: str, pinned: str) -> bool:
    """Whether a workflow runs on pull_request and installs the backend's pinned version."""

    return "pull_request" in text and backend in text and pinned in text


def _g5(card: PreviewScorecard, repo_root: Path | None) -> GraduationItem:
    """Docs: a tutorial and a can-and-cannot table."""

    if repo_root is None:
        return GraduationItem("G5", False, "repository files unavailable")
    page = repo_root / "docs" / "backends" / f"{card.backend}.md"
    if not page.is_file():
        return GraduationItem("G5", False, f"docs/backends/{card.backend}.md is missing")
    text = page.read_text(encoding="utf-8")
    missing = [heading for heading in ("## Tutorial", "## Can and cannot") if heading not in text]
    if missing:
        return GraduationItem("G5", False, f"docs/backends/{card.backend}.md lacks {missing}")
    return GraduationItem("G5", True, f"docs/backends/{card.backend}.md has both sections")


def _g6(card: PreviewScorecard, repo_root: Path | None) -> GraduationItem:
    """Frozen spellings: no unstable marker in the package; glossary names the backend."""

    if repo_root is None:
        return GraduationItem("G6", False, "repository files unavailable")
    package = repo_root / "torchlens" / "backends" / card.backend
    marked = (
        sorted(
            str(path.relative_to(repo_root))
            for path in package.rglob("*.py")
            if _UNSTABLE_MARKER in path.read_text(encoding="utf-8")
        )
        if package.is_dir()
        else []
    )
    problems: list[str] = []
    if marked:
        problems.append(f"{_UNSTABLE_MARKER} in {marked[:3]}")
    glossary = repo_root / "docs" / "reference" / "glossary.md"
    if not glossary.is_file():
        problems.append("docs/reference/glossary.md is missing")
    else:
        text = glossary.read_text(encoding="utf-8")
        if f"backends.{card.backend}" not in text:
            problems.append(f"glossary has no entry naming backends.{card.backend}")
    if problems:
        return GraduationItem("G6", False, "; ".join(problems))
    return GraduationItem(
        "G6", True, "no unstable marker in the package; glossary names the backend"
    )


def graduation_items(
    card: PreviewScorecard,
    *,
    reference: PreviewScorecard | None,
    repo_root: Path | None = None,
) -> tuple[GraduationItem, ...]:
    """Compute the six items for one scorecard.

    Parameters
    ----------
    card:
        The backend's scorecard (its own ``graduation`` field is ignored).
    reference:
        Torch's scorecard, the suite's reference (``None`` leaves G1 unmet).
    repo_root:
        Repository root for the file-based items; auto-detected when omitted.

    Returns
    -------
    tuple[GraduationItem, ...]
        The items in :data:`GRADUATION_IDS` order.
    """

    root = repo_root if repo_root is not None else repository_root()
    return (
        _g1(card, reference),
        _g2(card),
        _g3(card),
        _g4(card, root),
        _g5(card, root),
        _g6(card, root),
    )


def graduation_report(backend: str) -> GraduationReport:
    """The graduation report for one backend from its shipped scorecard.

    Parameters
    ----------
    backend:
        Registered backend name.

    Returns
    -------
    GraduationReport
        Six items. Without a shipped record every item is unmet with the
        detail ``no shipped scorecard``; the report never writes ``maturity``.
    """

    return _graduation_report(backend)


def _graduation_report(backend: str, *, repo_root: Path | None = None) -> GraduationReport:
    """:func:`graduation_report` with an explicit repository root (the CLI's ``--repo-root``)."""

    from ._scorecard import reference_scorecard_for, shipped_scorecard

    card = shipped_scorecard(backend)
    if card is None:
        items = tuple(GraduationItem(id, False, "no shipped scorecard") for id in GRADUATION_IDS)
        return GraduationReport(backend, items)
    reference = reference_scorecard_for(backend, card)
    return GraduationReport(
        backend, graduation_items(card, reference=reference, repo_root=repo_root)
    )
