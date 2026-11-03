"""Pin the 3.0 saved-format constants to the generated persisted-field page.

``torchlens/_io/format3/contract.py`` is the data every format lane builds
against. These tests hold it to ``docs/maintainers/persisted_fields.md`` (record
index, key count, contract version), to the storage-kind enum (the JSON type
map covers every kind), to the refusal contract (every code named is a
documented row) and to the writer contract (identity fields, codecs, levels),
and prove each stub door refuses typed with its code and a remedy.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from torchlens._io.format3 import contract, reader, runnable, writer
from torchlens._io.format_errors import ArtifactError

REPO_ROOT = Path(__file__).resolve().parent.parent
PAGE = REPO_ROOT / "docs" / "maintainers" / "persisted_fields.md"
REFUSAL_DOC = REPO_ROOT / "docs" / "reference" / "error_refusal_contract.md"
CONTRACT_PAGE = REPO_ROOT / "docs" / "maintainers" / "saved_format_3.md"

_RECORD_ROW = re.compile(r"^\| `([a-z_]+)` \| `([A-Za-z0-9_.]+)` \| (\d+) \| (\d+) \|$", re.M)
_KEY_ROW = re.compile(r"^\| `([A-Za-z0-9_]+)` \| ", re.M)


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


def _record_section(record: str) -> str:
    text = _page()
    start = text.index(f"\n## `{record}` (")
    end = text.find("\n## ", start + 1)
    return text[start : end if end > 0 else None]


def test_every_refusal_code_is_a_documented_row() -> None:
    """Each code the format names has a row in the public refusal contract."""

    documented = set(
        re.findall(r"^\| `([a-z0-9_]+)` \| ", REFUSAL_DOC.read_text(encoding="utf-8"), re.M)
    )
    named = set(contract.REFUSAL_CODES.values()) | set(contract.PROVISIONAL_REFUSALS.values())
    assert named <= documented, sorted(named - documented)
    assert not set(contract.PROVISIONAL_REFUSALS) & documented, (
        "a provisional spelling became a documented code; move it to REFUSAL_CODES"
    )


def test_ruled_spellings_are_on_the_contract_page() -> None:
    """Every ruled name the contract module carries is spelled on the public contract page."""

    text = CONTRACT_PAGE.read_text(encoding="utf-8")
    ruled = (
        contract.UNENCODABLE_VALUE_TYPE,
        contract.UNENCODABLE_LOAD_ADVISORY,
        *(f'"{label}"' for label in contract.FORMAT_DETECTION_LABELS),
        *contract.PROVISIONAL_REFUSALS,
        "intervention_spec_not_portable",
    )
    missing = [name for name in ruled if name not in text]
    assert not missing, missing


@pytest.mark.parametrize(
    ("door", "code"),
    [
        (lambda: writer.write_trace(None, "x"), "bundle_save_failed"),
        (lambda: writer.write_bundle(None, "x"), "bundle_save_failed"),
        (lambda: writer.write_intervention_recipe(None, "x"), "bundle_save_failed"),
        (lambda: runnable.write_runnable(None, "x"), "bundle_save_failed"),
        (lambda: reader.read_manifest("x"), "bundle_load_failed"),
        (lambda: reader.validate_artifact("x"), "bundle_load_failed"),
        (lambda: reader.load_trace("x"), "bundle_load_failed"),
        (lambda: reader.load_bundle("x"), "bundle_load_failed"),
        (lambda: reader.load_intervention_recipe("x"), "bundle_load_failed"),
        (lambda: runnable.load_runnable("x"), "bundle_load_failed"),
    ],
    ids=lambda value: value if isinstance(value, str) else "door",
)
def test_stub_doors_refuse_typed_naming_their_module(door, code) -> None:  # type: ignore[no-untyped-def]
    """Every unbuilt door refuses with its code, a remedy and the module that fills it."""

    with pytest.raises(ArtifactError) as info:
        door()
    assert info.value.fields["code"] == code
    assert info.value.fields["remedy"]
    assert "torchlens/_io/format3/" in str(info.value)
