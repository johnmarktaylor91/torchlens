"""Load then save is lossless, and validation refuses what load refuses.

Round trip: the three fields a resave used to change (``module.forward_args_template``
to null, ``module_call.has_saved_forward_args`` False to null, ``trace._buffer_persistence``
null to ``{}``) keep their saved cells and their live values. Parity: a forged cell, a
dangling reference and a repeated row refuse with one code through ``tl.load``,
``reader.validate_artifact`` and ``validate_tlspec``. Runtime identity: the shared policy
runs on a 3.0 manifest after the floor; the optional summaries meet their closed shapes;
an unlisted file is named in a warning and never read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from _format3_envelope import (
    read_manifest,
    read_record,
    resave,
    save_tiny,
    set_cell,
    write_manifest,
)

import torchlens as tl
from torchlens._io.format3 import contract, reader
from torchlens._io.format_errors import ArtifactError
from torchlens.errors._base import TorchLensWarning
from torchlens.validation import validate_tlspec

pytestmark = pytest.mark.heavy


def _column(root: Path, kind: str, column: str) -> list[Any]:
    payload = read_record(root, kind)
    if kind == "trace":
        return [payload["fields"].get(column)]
    return payload["columns"].get(column, [])


# ---- round trip -------------------------------------------------------------


def test_has_saved_forward_args_survives_a_resave(tmp_path: Path) -> None:
    first = tmp_path / "first"
    live = save_tiny(first, "portable")
    loaded = tl.load(first)
    live_flags = [call.has_saved_forward_args for module in live.modules for call in module.ops]
    loaded_flags = [call.has_saved_forward_args for module in loaded.modules for call in module.ops]
    assert loaded_flags == live_flags
    assert all(flag is False for flag in live_flags)
    second = resave(loaded, tmp_path / "second", "portable")
    assert _column(second, "module_call", "has_saved_forward_args") == _column(
        first, "module_call", "has_saved_forward_args"
    )
    assert None not in _column(second, "module_call", "has_saved_forward_args")


def test_module_templates_survive_a_resave(tmp_path: Path) -> None:
    first = tmp_path / "first"
    save_tiny(first, "portable")
    loaded = tl.load(first)
    second = resave(loaded, tmp_path / "second", "portable")
    for column in ("forward_args_template", "forward_kwargs_template"):
        assert _column(second, "module", column) == _column(first, "module", column), column


def test_null_buffer_persistence_stays_none(tmp_path: Path) -> None:
    first = tmp_path / "first"
    live = save_tiny(first, "portable")
    assert _column(first, "trace", "_buffer_persistence") == [live._buffer_persistence]
    loaded = tl.load(first)
    assert loaded._buffer_persistence == live._buffer_persistence
    second = resave(loaded, tmp_path / "second", "portable")
    assert _column(second, "trace", "_buffer_persistence") == _column(
        first, "trace", "_buffer_persistence"
    )


def test_forced_null_scalar_round_trips_as_null(tmp_path: Path) -> None:
    """A trace whose ``_buffer_persistence`` is None saves null, loads None, resaves null."""

    first = tmp_path / "first"
    live = save_tiny(tmp_path / "scratch", "portable")
    live._buffer_persistence = None
    live.save(first, level="portable")
    assert _column(first, "trace", "_buffer_persistence") == [None]
    loaded = tl.load(first)
    assert loaded._buffer_persistence is None
    second = resave(loaded, tmp_path / "second", "portable")
    assert _column(second, "trace", "_buffer_persistence") == [None]


# ---- validation parity ------------------------------------------------------


def _codes(root: Path) -> set[str]:
    codes = set()
    for api in (tl.load, reader.validate_artifact, validate_tlspec):
        with pytest.raises(ArtifactError) as info:
            api(root)
        codes.add(info.value.fields.get("code"))
    return codes


def _forge_typed_cell(root: Path) -> None:
    set_cell(root, "op", "label", 0, 7)


def _forge_dangling_reference(root: Path) -> None:
    layer_list = read_record(root, "trace")["fields"]["_layer_list"]
    layer_list[0] = {"$tl": "ref", "record": "op", "row": 999}
    set_cell(root, "trace", "_layer_list", 0, layer_list)


def _forge_repeated_row(root: Path) -> None:
    labels = read_record(root, "op")["columns"]["label"]
    set_cell(root, "op", "label", 1, labels[0])


@pytest.mark.parametrize(
    "forge", [_forge_typed_cell, _forge_dangling_reference, _forge_repeated_row]
)
def test_validation_refuses_what_load_refuses(tmp_path: Path, forge) -> None:
    root = tmp_path / "forged"
    save_tiny(root, "portable")
    forge(root)
    codes = _codes(root)
    assert len(codes) == 1, codes
    assert codes <= set(contract.REFUSAL_CODES.values())


def test_validation_accepts_a_real_save(tmp_path: Path) -> None:
    root = tmp_path / "ok"
    save_tiny(root, "portable")
    assert reader.validate_artifact(root)["kind"] == "trace"
    validate_tlspec(root)


# ---- runtime identity, summaries, unlisted files -----------------------------


def _stamp(root: Path, **fields: Any) -> None:
    manifest = read_manifest(root)
    manifest.update(fields)
    write_manifest(root, manifest)


def test_torch_major_mismatch_refuses_after_the_floor(tmp_path: Path) -> None:
    root = tmp_path / "torch"
    save_tiny(root, "portable")
    _stamp(root, torch_version="999.0.0")
    with pytest.raises(ArtifactError) as info:
        tl.load(root)
    assert info.value.fields.get("code") == "bundle_torch_incompatible"
    _stamp(root, torch_version="999.0.0", tlspec_version=0)
    with pytest.raises(ArtifactError) as info:
        tl.load(root)
    assert info.value.fields.get("code") == "artifact_version_below_floor"


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("torchlens_version", "999.0.0", "newer than runtime"),
        ("python_version", "999.0.0", "python_version=999.0.0"),
    ],
)
def test_identity_warnings_load(tmp_path: Path, field: str, value: str, match: str) -> None:
    root = tmp_path / field
    save_tiny(root, "portable")
    _stamp(root, **{field: value})
    with pytest.warns(TorchLensWarning, match=match):
        trace = tl.load(root)
    assert trace is not None


def test_identity_grammar_is_the_code_owned_table(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "grammar"
    save_tiny(root, "portable")
    monkeypatch.setitem(contract.IDENTITY_FIELD_PATTERNS, "platform", r"never-matches-\d+")
    with pytest.raises(ArtifactError) as info:
        reader.validate_artifact(root)
    assert info.value.fields.get("code") == "manifest_schema_invalid"
    assert "platform" in str(info.value)


def test_summaries_are_checked_when_present(tmp_path: Path) -> None:
    root = tmp_path / "summaries"
    save_tiny(root, "portable")
    _stamp(
        root,
        model_fingerprint={
            "parameter_meta_hash": "0" * 64,
            "buffer_meta_hash": "1" * 64,
            "class_qualname": "Tiny",
            "extra_metadata": {},
        },
        backward_summary={
            "has_backward_pass": False,
            **dict.fromkeys(contract.BACKWARD_SUMMARY_COUNT_FIELDS, 0),
            "gradient_blob_kinds": [],
            "old_bundle_policy": "none",
        },
    )
    reader.validate_artifact(root)
    _stamp(root, backward_summary={"has_backward_pass": "no"})
    with pytest.raises(ArtifactError) as info:
        reader.validate_artifact(root)
    assert info.value.fields.get("code") == "manifest_schema_invalid"
    _stamp(root, backward_summary=None, model_fingerprint={"class_qualname": ""})
    with pytest.raises(ArtifactError) as info:
        tl.load(root)
    assert info.value.fields.get("code") == "manifest_schema_invalid"


def test_unlisted_files_warn_and_are_never_read(tmp_path: Path) -> None:
    root = tmp_path / "unlisted"
    live = save_tiny(root, "portable")
    (root / "README.txt").write_text("notes", encoding="utf-8")
    (root / "records" / "extra.json").write_bytes(b"{not json")
    (root / "blobs").mkdir(exist_ok=True)
    (root / "blobs" / "planted.safetensors").write_bytes(b"\x00\x01")
    with pytest.warns(TorchLensWarning, match="unreferenced extra files") as caught:
        loaded = tl.load(root)
    message = str(caught[0].message)
    assert "records/extra.json" in message and "blobs/planted.safetensors" in message
    assert loaded.source.class_name == live.source.class_name
    assert len(loaded.layers) == len(live.layers)
