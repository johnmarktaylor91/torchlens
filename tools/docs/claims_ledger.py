"""The public claims ledger: one row per claim TorchLens makes in public copy.

The ledger (``docs/migration/_claims_ledger.json``) records, for every public
claim, its wording, where it appears, the evidence behind it and its status. It
sits beside ``docs/migration/_claims.json`` rather than inside it: that registry
holds wording rules per topic (allowed, narrowed and limitation wording, and the
lint patterns that enforce them), while the ledger is an inventory of claim
occurrences with release evidence. A ledger row points at the registry rows that
govern its wording through ``registry_ids``.

Statuses
--------
``release-supported``
    Evidence measured on a release; needs ``release`` (``v2.36.1@550b96a6``: the release
    tag and the first eight hex digits of its commit) and evidence.
``train-only``
    Evidence exists on the development line but not yet on a release.
``historical``
    True of an earlier release; kept only in historical copy.
``preview-or-partial``
    Holds for part of the stated scope (a preview backend, a subset of models).
``unmeasured``
    No evidence on file. Planned claims carry this status until measured.

``python tools/docs/claims_ledger.py check`` validates the ledger;
``python tools/docs/claims_ledger.py scan FILE ...`` lists claim-like paragraphs
that no ledger row covers; ``python tools/docs/claims_ledger.py release`` lists
the rows a release would block on (README and badge numbers need release
evidence).
"""

from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER = REPO_ROOT / "docs" / "migration" / "_claims_ledger.json"
REGISTRY = REPO_ROOT / "docs" / "migration" / "_claims.json"
SCHEMA = "torchlens.claims_ledger.v1"

STATUSES = ("release-supported", "train-only", "historical", "preview-or-partial", "unmeasured")
CATEGORIES = (
    "any-model",
    "every-operation",
    "validated",
    "backends",
    "speed",
    "agents",
    "science",
    "capability",
    "comparison",
    "support",
)
ROW_FIELDS = {
    "id": str,
    "category": str,
    "wording": str,
    "locations": list,
    "release": (str, type(None)),
    "runtime_backend": str,
    "models": str,
    "evidence": list,
    "evidence_note": str,
    "exclusions": str,
    "date": str,
    "status": str,
    "registry_ids": list,
}
OPTIONAL_ROW_FIELDS = {"say_today": str, "before_strengthening": str}
#: A location is a repo file, or an outside surface (no file to check).
LOCATION_KINDS = ("file", "external")
_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_RELEASE = re.compile(r"^v\d+\.\d+\.\d+@[0-9a-f]{8}$")

#: Paragraph patterns that mark a claim in public copy. A paragraph that matches
#: one of these must contain the wording of some ledger row (coverage check).
CLAIM_PATTERNS = (
    r"\b\d[\d,.]*\s*(?:x|×)(?!\w)",
    r"\b\d[\d,]*\+",
    r"\b(?:every|any)\s+(?:activation|operation|op|model|architecture|pytorch model)s?\b",
    r"\b(?:faster|slower)\b",
    r"\b(?:verified|validated|proven)\b",
    r"\b(?:first tool|no other|only tool|nobody)\b",
    r"\bthousands\b",
)
_CLAIM = re.compile("|".join(CLAIM_PATTERNS), re.IGNORECASE)
_FENCE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)
_LIST_ITEM = re.compile(r"(?:[-*+]|\d+[.)])\s")


def normalize(text: str) -> str:
    """Normalize Markdown for wording comparison.

    Removes blockquote markers, backticks and emphasis, and collapses whitespace,
    so wording matches however the copy is wrapped or styled.
    """

    text = re.sub(r"^\s*(?:>\s?)+", "", text, flags=re.MULTILINE)
    text = text.replace("`", "").replace("**", "")
    text = re.sub(r"(?<!\w)\*(?=\S)|(?<=\S)\*(?!\w)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def paragraphs(text: str) -> list[str]:
    """Split Markdown prose (code blocks removed) into normalized paragraphs."""

    out: list[str] = []
    block: list[str] = []
    for line in _FENCE.sub("", text).splitlines():
        stripped = re.sub(r"^(?:>\s?)+", "", line.strip()).strip()
        if block and _LIST_ITEM.match(stripped):
            out.append(normalize(" ".join(block)))
            block = []
        if not stripped or stripped.startswith(("|", "#", "<img", "[![")):
            if block:
                out.append(normalize(" ".join(block)))
                block = []
            if stripped.startswith(("|", "#")):
                out.append(normalize(stripped))
            continue
        block.append(stripped)
    if block:
        out.append(normalize(" ".join(block)))
    return out


def load(path: Path = LEDGER) -> dict[str, Any]:
    """Read the ledger file."""

    return json.loads(path.read_text(encoding="utf-8"))


def _registry_ids(path: Path = REGISTRY) -> set[str]:
    """Ids of the claim and lint rows in the wording registry."""

    data = json.loads(path.read_text(encoding="utf-8"))
    return {row["id"] for key in ("claims", "lints") for row in data[key]}


def _location_problems(row_id: str, location: Any, root: Path, wording: str) -> list[str]:
    """Problems with one location of one row."""

    if (
        not isinstance(location, dict)
        or len(location) != 1
        or next(iter(location)) not in LOCATION_KINDS
    ):
        return [f"{row_id}: location {location!r} must be {{'file': path}} or {{'external': name}}"]
    kind, value = next(iter(location.items()))
    if kind == "external":
        return (
            []
            if isinstance(value, str) and value.strip()
            else [f"{row_id}: empty external location"]
        )
    if not isinstance(value, str) or value.startswith("/") or ".." in Path(value).parts:
        return [f"{row_id}: location {value!r} must be a repo-relative path"]
    file = root / value
    if not file.is_file():
        return [f"{row_id}: location {value} does not exist"]
    if normalize(wording) not in normalize(file.read_text(encoding="utf-8")):
        return [f"{row_id}: wording not found in {value} (update the row with the copy)"]
    return []


def row_problems(row: Any, root: Path = REPO_ROOT, registry: set[str] | None = None) -> list[str]:
    """List every problem with one ledger row."""

    if not isinstance(row, dict):
        return [f"row must be a mapping, got {type(row).__name__}"]
    row_id = row.get("id", "<no id>")
    problems = [
        f"{row_id}: unknown field {key!r}"
        for key in sorted(set(row) - set(ROW_FIELDS) - set(OPTIONAL_ROW_FIELDS))
    ]
    for name, kind in {**ROW_FIELDS, **OPTIONAL_ROW_FIELDS}.items():
        if name not in row:
            if name in ROW_FIELDS:
                problems.append(f"{row_id}: missing field {name!r}")
            continue
        if not isinstance(row[name], kind):
            problems.append(f"{row_id}: {name} has the wrong type")
    if problems:
        return problems
    if not _ID.match(row_id):
        problems.append(f"{row_id}: id must be lowercase words joined by hyphens")
    if row["status"] not in STATUSES:
        problems.append(f"{row_id}: status {row['status']!r} is not one of {list(STATUSES)}")
    if row["category"] not in CATEGORIES:
        problems.append(f"{row_id}: category {row['category']!r} is not one of {list(CATEGORIES)}")
    if not row["wording"].strip():
        problems.append(f"{row_id}: empty wording")
    try:
        date.fromisoformat(row["date"])
    except ValueError:
        problems.append(f"{row_id}: date {row['date']!r} is not YYYY-MM-DD")
    if row["release"] is not None and not _RELEASE.match(row["release"]):
        problems.append(f"{row_id}: release must be <tag>@<first 8 hex of its commit>")
    if row["status"] == "release-supported" and (row["release"] is None or not row["evidence"]):
        problems.append(f"{row_id}: release-supported needs release and evidence")
    if row["status"] != "unmeasured" and not (row["evidence"] or row["evidence_note"].strip()):
        problems.append(f"{row_id}: a measured status needs evidence or an evidence note")
    for location in row["locations"]:
        problems += _location_problems(row_id, location, root, row["wording"])
    for pointer in row["evidence"]:
        path = str(pointer).split("::", 1)[0]
        if path.startswith("/") or ".." in Path(path).parts or not (root / path).exists():
            problems.append(f"{row_id}: evidence {pointer!r} is not an existing repo-relative path")
    if registry is not None:
        unknown = sorted(set(row["registry_ids"]) - registry)
        if unknown:
            problems.append(f"{row_id}: registry_ids {unknown} are not rows of _claims.json")
    return problems


def ledger_problems(
    ledger: dict[str, Any], root: Path = REPO_ROOT, registry: set[str] | None = None
) -> list[str]:
    """List every problem with the whole ledger."""

    problems = [] if ledger.get("schema") == SCHEMA else [f"schema must be {SCHEMA!r}"]
    rows = ledger.get("rows")
    if not isinstance(rows, list) or not rows:
        return problems + ["rows must be a non-empty list"]
    seen: set[str] = set()
    for row in rows:
        problems += row_problems(row, root, registry)
        row_id = row.get("id") if isinstance(row, dict) else None
        if row_id in seen:
            problems.append(f"{row_id}: duplicate id")
        seen.add(str(row_id))
    return problems


def uncovered(text: str, ledger: dict[str, Any]) -> list[str]:
    """Claim-like paragraphs of ``text`` that contain no ledger row's wording."""

    wordings = [normalize(row["wording"]) for row in ledger["rows"]]
    return [p for p in paragraphs(text) if _CLAIM.search(p) and not any(w in p for w in wordings)]


def release_blockers(ledger: dict[str, Any]) -> list[str]:
    """Rows a release would block on: README or badge numbers without release evidence."""

    blockers = []
    for row in ledger["rows"]:
        in_readme = any(
            loc.get("file") == "README.md" or "badge" in str(loc.get("external", "")).lower()
            for loc in row["locations"]
        )
        if in_readme and re.search(r"\d", row["wording"]) and row["status"] != "release-supported":
            blockers.append(f"{row['id']}: {row['status']}")
    return blockers


def main(argv: list[str]) -> int:
    """Command-line entry point (see the module docstring)."""

    command, *paths = argv or ["check"]
    ledger = load()
    if command == "check":
        problems = ledger_problems(ledger, registry=_registry_ids())
        print("\n".join(problems) or f"{len(ledger['rows'])} rows, no problems")
        return 1 if problems else 0
    if command == "scan":
        for path in paths:
            for paragraph in uncovered(Path(path).read_text(encoding="utf-8"), ledger):
                print(f"{path}: {paragraph[:200]}")
        return 0
    if command == "release":
        blockers = release_blockers(ledger)
        print("\n".join(blockers) or "no release blockers")
        return 1 if blockers else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
