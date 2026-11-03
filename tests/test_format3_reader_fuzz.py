"""Fuzz the 3.0 reader with mutated writer envelopes: every mutation refuses typed or round-trips exactly.

Byte flips (records and blobs), dropped manifest keys and swapped cell and
field types are applied one at a time to copies of a saved tiny CNN. A
mutated artifact either loads back and resaves equal to the original (the mutation
changed nothing the format reads) or refuses with an ``ArtifactError`` whose
code is in the contract's refusal table, with no record object built.
"""

from __future__ import annotations

import itertools
import json

import pytest
from _format3_envelope import (
    copy_artifact,
    first_column,
    flip_byte,
    read_manifest,
    read_record,
    record_table_diffs,
    redigest,
    resave,
    save_tiny,
    set_cell,
    write_manifest,
    write_record,
)

from torchlens._io.format3 import contract
from torchlens._io.format_errors import ArtifactError

pytestmark = pytest.mark.heavy

KNOWN_CODES = frozenset(contract.REFUSAL_CODES.values()) | frozenset(
    contract.PROVISIONAL_REFUSALS.values()
)


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    root = tmp_path_factory.mktemp("f3-fuzz") / "trace"
    trace = save_tiny(root, "portable")
    return root, trace


@pytest.fixture
def built_counter(monkeypatch):
    """Count record objects allocated by the reader (none may exist after a refusal)."""

    import torchlens._data_classes._state_adapter as adapter

    calls = []
    original = adapter.state_new

    def counting(cls):
        calls.append(cls)
        return original(cls)

    monkeypatch.setattr(adapter, "state_new", counting)
    return calls


def _load(root):
    import torchlens as tl

    return tl.load(root)


def _expect_refusal_or_roundtrip(root, original, built_counter):
    """Load the mutated ``root``: a typed refusal, or a load whose resave equals what it loaded.

    ``original`` names the save level. A mutation the reader accepts is a value change (a
    free-text cell swapped for another string), so fidelity is measured against ``root``, the
    artifact that was loaded, never against the pristine copy.
    """

    try:
        loaded = _load(root)
    except ArtifactError as exc:
        assert exc.fields.get("code") in KNOWN_CODES, exc
        assert not any(cls.__name__ == "Trace" for cls in built_counter), "a Trace was built"
        return "refused"
    again = resave(
        loaded, root.parent / (root.name + "-again"), read_manifest(original)["save_level"]
    )
    assert record_table_diffs(root, again) == []
    return "roundtrip"


def _mutable_files(root):
    manifest = read_manifest(root)
    return sorted(manifest["files"])


def test_unmodified_copy_round_trips(saved, tmp_path, built_counter):
    root, _ = saved
    copy = copy_artifact(root, tmp_path / "copy")
    assert _expect_refusal_or_roundtrip(copy, root, built_counter) == "roundtrip"


@pytest.mark.parametrize("offset", [0, 7, 64, 257, 1031, -3])
def test_byte_flip_in_each_listed_file(saved, tmp_path, built_counter, offset):
    root, _ = saved
    for index, relpath in enumerate(_mutable_files(root)):
        copy = copy_artifact(root, tmp_path / f"flip-{index}")
        flip_byte(copy / relpath, offset)
        verdict = _expect_refusal_or_roundtrip(copy, root, built_counter)
        assert verdict == "refused", relpath


@pytest.mark.parametrize("offset", [3, 99, 1024])
def test_byte_flip_in_the_manifest(saved, tmp_path, built_counter, offset):
    root, _ = saved
    copy = copy_artifact(root, tmp_path / "manifest-flip")
    flip_byte(copy / contract.MANIFEST_FILE, offset)
    _expect_refusal_or_roundtrip(copy, root, built_counter)


def test_each_dropped_manifest_key_refuses(saved, tmp_path, built_counter):
    root, _ = saved
    manifest = read_manifest(root)
    for key in sorted(manifest):
        copy = copy_artifact(root, tmp_path / f"drop-{key}")
        trimmed = dict(manifest)
        del trimmed[key]
        write_manifest(copy, trimmed)
        verdict = _expect_refusal_or_roundtrip(copy, root, built_counter)
        if key in contract.MANIFEST_REQUIRED_FIELDS:
            assert verdict == "refused", key


SWAPS = [None, True, 0, -1, 2**53 + 1, 1.5, "", "x", [], {}, {"$tl": "pickle"}]
#: Manifest version stamps; a non-empty string that is not PEP 440 refuses by field and code.
VERSION_STAMPS = ("torchlens_version", "python_version", "torch_version")


@pytest.mark.parametrize("value", SWAPS, ids=lambda v: json.dumps(v))
def test_swapped_manifest_field_types_refuse(saved, tmp_path, built_counter, value):
    root, _ = saved
    manifest = read_manifest(root)
    for key in contract.MANIFEST_REQUIRED_FIELDS:
        if manifest.get(key) == value:
            continue
        copy = copy_artifact(root, tmp_path / f"swap-{key}")
        swapped = dict(manifest)
        swapped[key] = value
        write_manifest(copy, swapped)
        if key in VERSION_STAMPS and isinstance(value, str) and value:
            # A version stamp that does not parse under PEP 440 is malformed manifest data:
            # the schema check names the field, before any runtime identity policy runs.
            with pytest.raises(ArtifactError) as info:
                _load(copy)
            assert info.value.fields.get("code") == "manifest_schema_invalid", key
            assert f"{key} must be a PEP 440 version string" in str(info.value), key
            assert not any(cls.__name__ == "Trace" for cls in built_counter), "a Trace was built"
            continue
        assert _expect_refusal_or_roundtrip(copy, root, built_counter) == "refused", key


@pytest.mark.parametrize("value", SWAPS, ids=lambda v: json.dumps(v))
def test_swapped_record_cell_types_refuse_or_roundtrip(saved, tmp_path, built_counter, value):
    root, _ = saved
    for kind in read_manifest(root)["records"]:
        copy = copy_artifact(root, tmp_path / f"cell-{kind}")
        set_cell(copy, kind, first_column(copy, kind), 0, value)
        _expect_refusal_or_roundtrip(copy, root, built_counter)


def test_dropped_and_added_columns_refuse(saved, tmp_path, built_counter):
    root, _ = saved
    for kind in read_manifest(root)["records"]:
        payload = read_record(root, kind)
        table = payload["fields"] if kind == "trace" else payload["columns"]
        dropped = copy_artifact(root, tmp_path / f"dropcol-{kind}")
        trimmed = json.loads(json.dumps(payload))
        del (trimmed["fields"] if kind == "trace" else trimmed["columns"])[next(iter(table))]
        write_record(dropped, kind, trimmed)
        assert _expect_refusal_or_roundtrip(dropped, root, built_counter) == "refused", kind
        added = copy_artifact(root, tmp_path / f"addcol-{kind}")
        extra = json.loads(json.dumps(payload))
        rows = 1 if kind == "trace" else extra["rows"]
        (extra["fields"] if kind == "trace" else extra["columns"])["__class__"] = (
            "os.system" if kind == "trace" else ["os.system"] * rows
        )
        write_record(added, kind, extra)
        assert _expect_refusal_or_roundtrip(added, root, built_counter) == "refused", kind


def test_row_count_mismatch_refuses(saved, tmp_path, built_counter):
    root, _ = saved
    for kind, delta in itertools.product(read_manifest(root)["records"], (-1, 1)):
        if kind == "trace":
            continue
        copy = copy_artifact(root, tmp_path / f"rows-{kind}-{delta}")
        payload = read_record(copy, kind)
        payload["rows"] = max(0, payload["rows"] + delta)
        write_record(copy, kind, payload)
        assert _expect_refusal_or_roundtrip(copy, root, built_counter) == "refused", kind


def test_blob_digest_mismatch_refuses(saved, tmp_path, built_counter):
    root, _ = saved
    blobs = [p for p in _mutable_files(root) if p.startswith(contract.BLOBS_DIR + "/")]
    for index, relpath in enumerate(blobs):
        copy = copy_artifact(root, tmp_path / f"blob-{index}")
        manifest = read_manifest(copy)
        manifest["files"][relpath]["sha256"] = "0" * 64
        write_manifest(copy, manifest)
        assert _expect_refusal_or_roundtrip(copy, root, built_counter) == "refused", relpath
    if blobs:
        copy = copy_artifact(root, tmp_path / "blob-rewrite")
        flip_byte(copy / blobs[0], 5)
        redigest(copy, blobs[0])
        _expect_refusal_or_roundtrip(copy, root, built_counter)
