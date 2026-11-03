"""Pin ``docs/maintainers/persisted_fields.md`` to its generator and the writer contract.

The page is generated from the writer contract of record, the frozen previous contract,
the storage bindings and the record-class modules by ``tools/docs/persisted_fields.py``.
These tests fail when the committed page differs from what those inputs produce, so a
contract change without a regenerated page cannot land, and they prove the generator
notices a contract change at all.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from torchlens._io import TLSPEC_VERSION

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools" / "docs"))

import persisted_fields  # noqa: E402

PAGE = REPO_ROOT / "docs" / "maintainers" / "persisted_fields.md"


def test_generator_selects_the_active_record_contract() -> None:
    """A record-version bump must move both generator contract inputs."""

    assert persisted_fields.CONTRACT.name == f"writer_contract_v{TLSPEC_VERSION}.json"
    assert persisted_fields.PREVIOUS_CONTRACT.name == f"writer_contract_v{TLSPEC_VERSION - 1}.json"


def test_committed_page_equals_the_generator_output() -> None:
    """The page on disk is byte for byte what the generator writes today."""

    assert PAGE.exists(), (
        f"{PAGE.relative_to(REPO_ROOT)} is missing; run {persisted_fields.COMMAND}"
    )
    expected = persisted_fields.render()
    assert PAGE.read_text(encoding="utf-8") == expected, (
        f"{PAGE.relative_to(REPO_ROOT)} is stale; run {persisted_fields.COMMAND}"
    )


def test_check_flag_accepts_the_committed_page(capsys: pytest.CaptureFixture[str]) -> None:
    """``--check`` exits 0 on the committed page and says so."""

    assert persisted_fields.main(["--check"]) == 0
    assert "is current" in capsys.readouterr().out


def test_a_changed_contract_changes_the_page(tmp_path: Path) -> None:
    """A contract with one more persisted key renders a different page."""

    document = json.loads(persisted_fields.CONTRACT.read_text(encoding="utf-8"))
    record = document["contract"]["record_contracts"]["layer"]
    record["persisted_field_names"].append("zz_added_by_test")
    record["field_policies"]["zz_added_by_test"] = "keep"
    record["field_tiers"]["zz_added_by_test"] = "fact"
    changed = tmp_path / persisted_fields.CONTRACT.name
    changed.write_text(json.dumps(document), encoding="utf-8")
    rendered = persisted_fields.render(contract_path=changed)
    assert rendered != PAGE.read_text(encoding="utf-8")
    assert "| `zz_added_by_test` |" in rendered
    assert "| `layer` | `zz_added_by_test` | added |" in rendered


def test_page_states_the_tlspec_version_and_no_torchlens_version() -> None:
    """The page cites the contract's tlspec version and digest and no package version."""

    document = json.loads(persisted_fields.CONTRACT.read_text(encoding="utf-8"))
    text = PAGE.read_text(encoding="utf-8")
    assert text.startswith(
        f"# Persisted fields of the saved format (tlspec {document['contract']['tlspec_version']})"
    )
    assert document["writer_contract_digest"] in text
    assert document["contract"]["min_torchlens_version"] not in text
    assert "torchlens 3" not in text and "torchlens 2" not in text


def test_every_persisted_key_has_a_row() -> None:
    """Each record contract's persisted keys all appear as rows in the page."""

    document = json.loads(persisted_fields.CONTRACT.read_text(encoding="utf-8"))
    text = PAGE.read_text(encoding="utf-8")
    for key, record in document["contract"]["record_contracts"].items():
        section = text.split(f"## `{key}` (", 1)[1].split("\n## ", 1)[0]
        for name in record["persisted_field_names"]:
            assert f"| `{name}` |" in section, f"{key}.{name} has no row"
