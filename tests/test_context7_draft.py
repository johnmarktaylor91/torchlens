"""The draft ``context7.json`` stays valid against the fields Context7 documents.

The file is a draft; registering it with Context7 is a separate maintainer step. This test keeps
it parseable, inside Context7's documented field limits, and pointed at paths that exist.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DRAFT = REPO_ROOT / "context7.json"
#: Field names Context7 documents for ``context7.json``.
KNOWN_FIELDS = {
    "$schema",
    "projectTitle",
    "description",
    "branch",
    "folders",
    "excludeFolders",
    "excludeFiles",
    "rules",
    "previousVersions",
    "branchVersions",
}


def _draft() -> dict:
    return json.loads(DRAFT.read_text(encoding="utf-8"))


def test_draft_uses_only_documented_fields_within_limits() -> None:
    draft = _draft()
    assert set(draft) <= KNOWN_FIELDS, set(draft) - KNOWN_FIELDS
    assert draft["$schema"] == "https://context7.com/schema/context7.json"
    assert 0 < len(draft["projectTitle"]) <= 100
    assert 0 < len(draft["description"]) <= 500
    assert draft["rules"] and all(isinstance(rule, str) and rule for rule in draft["rules"])


def test_draft_paths_exist() -> None:
    draft = _draft()
    for folder in draft["folders"] + draft["excludeFolders"]:
        assert (REPO_ROOT / folder).is_dir(), folder
    for name in draft["excludeFiles"]:
        assert "/" not in name, "excludeFiles takes bare file names"
        assert (REPO_ROOT / name).is_file(), name


def test_draft_excludes_package_source_and_contributor_notes() -> None:
    draft = _draft()
    assert "torchlens" in draft["excludeFolders"]
    assert {"AGENTS.md", "CLAUDE.md"} <= set(draft["excludeFiles"])
