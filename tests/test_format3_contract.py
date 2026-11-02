"""Pin the 3.0 saved-format constants to the generated persisted-field page.

``torchlens/_io/format3/contract.py`` is the data every format lane builds
against. These tests hold it to ``docs/maintainers/persisted_fields.md`` (record
index, key count, contract version), to the storage-kind enum (the JSON type
map covers every kind), to the refusal contract (every code named is a
documented row) and to the writer contract (identity fields, codecs, levels),
and prove each stub door refuses typed with its code and a remedy.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

from torchlens._data_classes.field_policy import StorageKind
from torchlens._io import save_policy, writer_contract
from torchlens._io.format3 import contract, reader, runnable, writer
from torchlens._io.format_errors import ArtifactError

REPO_ROOT = Path(__file__).resolve().parent.parent
PAGE = REPO_ROOT / "docs" / "maintainers" / "persisted_fields.md"
REFUSAL_DOC = REPO_ROOT / "docs" / "reference" / "error_refusal_contract.md"

_RECORD_ROW = re.compile(r"^\| `([a-z_]+)` \| `([A-Za-z0-9_.]+)` \| (\d+) \| (\d+) \|$", re.M)
_KEY_ROW = re.compile(r"^\| `([A-Za-z0-9_]+)` \| ", re.M)


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


def test_record_contract_version_matches_the_page() -> None:
    """The pinned record contract version is the page's tlspec and reader floor."""

    text = _page()
    header = re.search(r"^# Persisted fields of the saved format \(tlspec (\d+)\)", text, re.M)
    floor = re.search(r"^Reader floor: tlspec (\d+)\.", text, re.M)
    assert header is not None and floor is not None
    assert int(header.group(1)) == contract.RECORD_CONTRACT_VERSION
    assert int(floor.group(1)) == contract.RECORD_CONTRACT_FLOOR


def test_record_index_matches_the_page_table() -> None:
    """Record names, class paths and key counts equal the page's Records table."""

    rows = [(name, cls, int(keys)) for name, cls, keys, _ in _RECORD_ROW.findall(_page())]
    assert rows == list(contract.RECORD_INDEX)


def test_persisted_key_count_matches_the_page_header() -> None:
    """The key total is the page's headline count and the sum of its record tables."""

    text = _page()
    headline = re.search(r"^(\d+) record contracts, (\d+) persisted keys\.", text, re.M)
    assert headline is not None
    assert int(headline.group(1)) == len(contract.RECORD_INDEX)
    assert int(headline.group(2)) == contract.PERSISTED_KEY_COUNT


def test_planned_keys_are_not_on_the_page_yet() -> None:
    """A planned key is one the writer lane still adds; once added it leaves the plan."""

    for record, keys in contract.PLANNED_KEYS.items():
        section = _record_section(record)
        present = set(_KEY_ROW.findall(section))
        assert not present & set(keys), (
            f"{record}: {sorted(present & set(keys))} are on the page; remove them from "
            "PLANNED_KEYS in the same change"
        )


def _record_section(record: str) -> str:
    text = _page()
    start = text.index(f"\n## `{record}` (")
    end = text.find("\n## ", start + 1)
    return text[start : end if end > 0 else None]


def test_json_type_map_covers_every_storage_kind() -> None:
    """Every storage kind has one JSON encoding class, and the classes are the four named."""

    assert set(contract.JSON_TYPE_BY_STORAGE_KIND) == set(StorageKind)
    assert set(contract.JSON_TYPE_BY_STORAGE_KIND.values()) == {
        "value",
        "id_array",
        "tensor_ref",
        "never",
    }
    assert contract.JSON_TYPE_BY_STORAGE_KIND[StorageKind.RUNTIME] == "never"
    assert contract.JSON_TYPE_BY_STORAGE_KIND[StorageKind.PAYLOAD] == "tensor_ref"


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


def test_envelope_constants_match_the_writer_contract() -> None:
    """Identity fields, codecs and save levels are the writer contract's own."""

    assert contract.WRITER_IDENTITY_FIELDS == writer_contract.WRITER_IDENTITY_FIELDS
    assert contract.PAYLOAD_CODEC_IDS == writer_contract.PAYLOAD_CODEC_IDS
    assert set(contract.SAVE_LEVELS) == set(save_policy._LEVEL_INCLUDE_FLAG_DEFAULTS)
    assert contract.FORMAT_MAJOR == 3
    assert set(contract.WRITER_IDENTITY_FIELDS) <= set(contract.MANIFEST_REQUIRED_FIELDS)
    assert len(set(contract.TYPED_VALUE_TAGS)) == len(contract.TYPED_VALUE_TAGS)


def test_phase_timing_keys_are_the_pipeline_step_names() -> None:
    """The timing key source resolves to the pipeline, whose step names are unique."""

    module_path, _, attribute = contract.PHASE_TIMING_KEY_SOURCE.rpartition(".")
    pipeline = getattr(importlib.import_module(module_path), attribute)
    names = [step.name for step in pipeline]
    assert names and len(set(names)) == len(names)
    assert all(re.fullmatch(r"[a-z][a-z0-9_]*", name) for name in names)


def test_recipe_door_refuses_a_non_spec_typed() -> None:
    """The built recipe door refuses anything that is not a staged spec, typed."""

    with pytest.raises(ArtifactError) as info:
        writer.write_intervention_recipe(None, "x")
    assert info.value.fields["code"] == "bundle_save_failed"
    assert info.value.fields["remedy"]


@pytest.mark.parametrize("door", [writer.write_trace, writer.write_bundle])
def test_built_save_doors_refuse_a_non_saveable_typed(door) -> None:  # type: ignore[no-untyped-def]
    """The built trace and bundle doors refuse a value with no save door, typed."""

    with pytest.raises(ArtifactError) as info:
        door(None, "x")
    assert info.value.fields["code"] == "bundle_save_failed"
    assert info.value.fields["remedy"]
    assert "NoneType" in str(info.value)


@pytest.mark.parametrize(
    ("door", "code"),
    [
        (lambda: runnable.write_runnable(None, "x"), "bundle_save_failed"),
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


@pytest.mark.parametrize(
    "door",
    [
        reader.read_manifest,
        reader.validate_artifact,
        reader.load_trace,
        reader.load_bundle,
        reader.load_intervention_recipe,
    ],
    ids=lambda door: door.__name__,
)
def test_reader_doors_refuse_a_missing_artifact_typed(door, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Every reader door refuses a path that is not an artifact with ``manifest_missing``."""

    with pytest.raises(ArtifactError) as info:
        door(tmp_path / "absent")
    assert info.value.fields["code"] == contract.REFUSAL_CODES["manifest_missing"]
    assert info.value.fields["remedy"]
