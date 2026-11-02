"""The bundle container references member envelopes and keeps relations as JSON.

``torchlens/_io/format3/container.py`` writes each member trace as an artifact
of its own and records, in ``bundle.json``, the member's path plus the sha256 of
its manifest. These tests prove that a bundle of two traces joined by a
``forked_from`` relation round trips through ``Bundle.save`` and ``tl.load``,
that nothing in the container is a pickle, and that a member replaced, edited or
referenced with an unknown key refuses with the contract's typed codes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io.format3 import container
from torchlens._io.format_errors import ArtifactError


class _Shift(nn.Module):
    """A linear map plus a fixed offset, so two members differ numerically."""

    def __init__(self, offset: float) -> None:
        """Build the map with a fixed seed and keep the offset."""

        super().__init__()
        torch.manual_seed(0)
        self.fc = nn.Linear(3, 2)
        self.offset = offset

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Map, offset and rectify."""

        return torch.relu(self.fc(value) + self.offset)


def _forked_bundle() -> tl.Bundle:
    """Two traces of the same input, the second forked from the first."""

    value = torch.randn(2, 3, generator=torch.Generator().manual_seed(1))
    clean = tl.trace(_Shift(0.0), value)
    changed = tl.trace(_Shift(0.5), value)
    bundle = tl.Bundle({"clean": clean, "changed": changed}, baseline="clean")
    bundle.relate("forked_from", "changed", "clean", at_step=0)
    return bundle


def _saved(tmp_path: Path) -> tuple[tl.Bundle, Path]:
    """Save the forked bundle and return it with its artifact path."""

    bundle = _forked_bundle()
    target = tmp_path / "forked.tlspec"
    bundle.save(target)
    return bundle, target


def _index(target: Path) -> dict[str, Any]:
    """Read the bundle index."""

    return json.loads((target / container.BUNDLE_INDEX_FILE).read_text(encoding="utf-8"))


def _write_index(target: Path, index: dict[str, Any]) -> None:
    """Rewrite the bundle index."""

    (target / container.BUNDLE_INDEX_FILE).write_text(json.dumps(index), encoding="utf-8")


def _load_code(target: Path) -> str:
    """Load ``target`` expecting a typed artifact refusal; return its code."""

    with pytest.raises(ArtifactError) as info:
        tl.load(target)
    return str(info.value.fields["code"])


def test_forked_bundle_round_trips(tmp_path: Path) -> None:
    """Members, their order, the relation and the member records survive the round trip."""

    bundle, target = _saved(tmp_path)
    index = _index(target)
    references = index["members"]
    assert [entry["name"] for entry in references] == ["clean", "changed"]
    for position, entry in enumerate(references):
        assert set(entry) == set(container.MEMBER_REFERENCE_KEYS)
        assert entry["path"] == container.member_relative_path(position, entry["name"])
        assert (target / entry["path"] / "manifest.json").is_file()

    loaded = tl.load(target)
    assert isinstance(loaded, tl.Bundle)
    assert list(loaded.members) == ["clean", "changed"]
    assert [row.to_payload() for row in loaded.relations] == [
        row.to_payload() for row in bundle.relations
    ]
    assert loaded.upstream("changed") == ("clean",)
    for name in ("clean", "changed"):
        live_layers = [layer.label for layer in bundle.members[name].layers]
        assert [layer.label for layer in loaded.members[name].layers] == live_layers


def _writer_built() -> bool:
    """Return whether this checkout's 3.0 trace writer door is built."""

    from torchlens._io.format3 import writer as writer3

    try:
        writer3.write_trace(None, "unused")
    except ArtifactError as exc:
        return "is not built in this checkout" not in str(exc)
    except Exception:  # noqa: BLE001 - a built door refuses ``None`` in its own way
        return True
    return True


def test_bundle_container_holds_no_pickle(tmp_path: Path) -> None:
    """No file in a 3.0 bundle artifact is a pickle (tlspec 11 members still pickle records)."""

    if not _writer_built():
        pytest.skip("the 3.0 trace writer is not built in this checkout")
    _bundle, target = _saved(tmp_path)
    for path in target.rglob("*"):
        assert path.suffix not in {".pkl", ".pickle", ".pt"}, path


def test_edited_member_manifest_refuses(tmp_path: Path) -> None:
    """A member manifest edited after the bundle save refuses as an integrity failure."""

    _bundle, target = _saved(tmp_path)
    manifest = target / _index(target)["members"][1]["path"] / "manifest.json"
    manifest.write_text(manifest.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert _load_code(target) == "bundle_metadata_integrity_refused"


def test_swapped_members_refuse(tmp_path: Path) -> None:
    """Two member paths swapped in the index no longer match their recorded envelopes."""

    _bundle, target = _saved(tmp_path)
    index = _index(target)
    first, second = index["members"]
    first["path"], second["path"] = second["path"], first["path"]
    _write_index(target, index)
    assert _load_code(target) == "bundle_metadata_integrity_refused"


def test_unknown_member_reference_key_refuses(tmp_path: Path) -> None:
    """A member reference carrying a key outside the contract refuses as schema invalid."""

    _bundle, target = _saved(tmp_path)
    index = _index(target)
    index["members"][0]["loader"] = "custom"
    _write_index(target, index)
    assert _load_code(target) == "manifest_schema_invalid"


def test_mistyped_member_hash_refuses(tmp_path: Path) -> None:
    """A member hash that is not a string refuses as schema invalid."""

    _bundle, target = _saved(tmp_path)
    index = _index(target)
    index["members"][0]["manifest_sha256"] = 7
    _write_index(target, index)
    assert _load_code(target) == "manifest_schema_invalid"


def test_required_reference_without_hash_refuses(tmp_path: Path) -> None:
    """When the hash is required, a reference without one refuses."""

    with pytest.raises(ArtifactError) as info:
        container.verify_member_reference(tmp_path, {"name": "a", "path": "m"}, required=True)
    assert info.value.fields["code"] == "manifest_schema_invalid"


def test_member_writer_that_writes_nothing_refuses(tmp_path: Path) -> None:
    """A member write that leaves no manifest refuses at the bundle save."""

    bundle = _forked_bundle()
    with pytest.raises(ArtifactError) as info:
        container.write_members(
            bundle, tmp_path, level="portable", write_member=lambda trace, path, level: None
        )
    assert info.value.fields["code"] == "bundle_save_failed"


def test_member_directory_names_are_contained() -> None:
    """A member name with separators or dots stays one directory under ``members/``."""

    assert container.member_relative_path(0, "clean") == "members/0000-clean"
    assert container.member_relative_path(3, "a/../b c") == "members/0003-a_.._b_c"
    assert container.member_relative_path(1, "..") == "members/0001-.."
    long_name = container.member_relative_path(2, "x" * 200)
    assert long_name == "members/0002-" + "x" * container.MEMBER_NAME_CHARS


def test_relations_file_round_trips(tmp_path: Path) -> None:
    """relations.json holds the relation rows and reads back through the closed parser."""

    bundle = _forked_bundle()
    row = container.write_relations(bundle, tmp_path)
    assert row is not None
    files = {container.RELATIONS_FILE: row}
    rows = container.read_relations(tmp_path, files)
    assert rows is not None
    assert [r.to_payload() for r in rows] == [r.to_payload() for r in bundle.relations]


def test_relation_free_bundle_writes_no_relations_file(tmp_path: Path) -> None:
    """A bundle without relation rows writes no file and reads back as none."""

    value = torch.randn(2, 3)
    bundle = tl.Bundle({"a": tl.trace(_Shift(0.0), value)})
    assert container.write_relations(bundle, tmp_path) is None
    assert not (tmp_path / container.RELATIONS_FILE).exists()
    assert container.read_relations(tmp_path, {}) is None


def test_edited_relations_file_refuses(tmp_path: Path) -> None:
    """A relation table edited after the write refuses as an integrity failure."""

    row = container.write_relations(_forked_bundle(), tmp_path)
    target = tmp_path / container.RELATIONS_FILE
    target.write_text(target.read_text("utf-8").replace("changed", "ghost"), "utf-8")
    with pytest.raises(ArtifactError) as info:
        container.read_relations(tmp_path, {container.RELATIONS_FILE: row})
    assert info.value.fields["code"] == "bundle_metadata_integrity_refused"


def test_unlisted_relations_file_refuses(tmp_path: Path) -> None:
    """A relations file the manifest does not list refuses."""

    container.write_relations(_forked_bundle(), tmp_path)
    with pytest.raises(ArtifactError) as info:
        container.read_relations(tmp_path, {})
    assert info.value.fields["code"] == "manifest_schema_invalid"


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_symlinked_member_manifest_refuses(tmp_path: Path) -> None:
    """A member manifest replaced by a symlink refuses before it is hashed."""

    _bundle, target = _saved(tmp_path)
    manifest = target / _index(target)["members"][0]["path"] / "manifest.json"
    moved = tmp_path / "elsewhere.json"
    manifest.rename(moved)
    manifest.symlink_to(moved)
    assert _load_code(target) == "load_path_symlink_rejected"
