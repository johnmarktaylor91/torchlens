"""An interrupted copy of a saved artifact refuses to load; only the complete copy loads.

A save writes ``manifest.json`` last, and the manifest lists the sha256 and byte count of
every other file it wrote: record tables, the recipe (``intervention.json``), the runnable
descriptor and every tensor chunk. Every load door, eager or lazy, checks every listed
file before it parses any of them. A copy that stopped part way (an ``rsync`` killed
mid-transfer, a ``cp`` that ran out of disk) therefore lacks a file, holds a short file or
lacks the manifest, and each of those refuses with a typed code instead of loading what
is there as complete.
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch
from _format3_envelope import read_manifest, save_tiny, write_manifest
from torch import nn

import torchlens as tl
from torchlens._io.format3 import contract
from torchlens.errors import ArtifactError, TorchLensWarning

pytestmark = pytest.mark.heavy

MANIFEST = contract.MANIFEST_FILE
CORRUPT = contract.REFUSAL_CODES["record_file_corrupt"]
#: Every code an interrupted or stripped copy may refuse with: all are integrity or
#: manifest refusals that already exist; none is a version or argument refusal.
INTEGRITY_CODES = frozenset(
    {
        contract.REFUSAL_CODES["manifest_missing"],
        contract.REFUSAL_CODES["manifest_unreadable"],
        contract.REFUSAL_CODES["manifest_invalid"],
        CORRUPT,
        contract.REFUSAL_CODES["record_schema_invalid"],
        contract.REFUSAL_CODES["blob_missing"],
        "bundle_blob_missing",
    }
)


def _relu_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 3))


def _replayable_trace() -> Any:
    return tl.trace(
        _relu_model(),
        torch.ones(2, 4),
        capture=tl.options.CaptureOptions(replayable=True),
    )


def _save_trace_with_recipe(root: Path) -> None:
    fork = _replayable_trace().fork()
    fork.do(tl.func("relu"), tl.zero_ablate())
    fork.save(root)


def _save_intervention(root: Path) -> None:
    _replayable_trace().save_intervention(root, level="audit")


def _save_bundle(root: Path) -> None:
    tl.Bundle({"a": _replayable_trace()}).save(root)


#: (name, writer, whether ``lazy=True`` applies, a file the artifact must hold).
ARTIFACTS: dict[str, tuple[Any, bool, str | None]] = {
    "portable": (lambda root: save_tiny(root, "portable"), True, None),
    "runnable": (lambda root: save_tiny(root, "runnable"), True, None),
    "trace_with_recipe": (_save_trace_with_recipe, True, contract.INTERVENTION_FILE),
    "intervention": (_save_intervention, False, contract.INTERVENTION_FILE),
    "bundle": (_save_bundle, False, None),
}


@pytest.fixture(scope="module")
def saved(tmp_path_factory) -> dict[str, Path]:
    """One saved artifact of each kind, written once for the module."""

    base = tmp_path_factory.mktemp("interrupted-copy")
    roots: dict[str, Path] = {}
    for name, (writer, _lazy, must_hold) in ARTIFACTS.items():
        root = base / f"{name}.tlspec"
        writer(root)
        if must_hold is not None:
            assert must_hold in read_manifest(root)["files"], (name, must_hold)
        roots[name] = root
    return roots


def _files(root: Path) -> list[str]:
    """Every file of the artifact, in the order ``rsync`` sends them (sorted paths)."""

    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def _load(root: Path, *, lazy: bool) -> Any:
    return tl.load(root, lazy=True) if lazy else tl.load(root)


def _refusal_code(root: Path, *, lazy: bool, allowed: frozenset[str] = INTEGRITY_CODES) -> str:
    with pytest.raises(ArtifactError) as info:
        _load(root, lazy=lazy)
    code = info.value.fields.get("code")
    assert code in allowed, f"{root}: {code!r}: {info.value}"
    return str(code)


def _cut_tail(path: Path) -> None:
    """Keep the first half of the file, as a transfer stopped part way through it leaves it."""

    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])


def _lazy_modes(name: str) -> list[bool]:
    return [False, True] if ARTIFACTS[name][1] else [False]


def _expected_code(relpath: str, *, removed: bool) -> str | None:
    """The code a single missing or short file refuses with (``None``: any integrity code)."""

    if relpath == MANIFEST:
        key = "manifest_missing" if removed else "manifest_unreadable"
        return contract.REFUSAL_CODES[key]
    if relpath.startswith(f"{contract.BLOBS_DIR}/"):
        return "bundle_blob_missing" if removed else CORRUPT
    top = relpath.split("/", 1)[0]
    if "/" in relpath and top not in (contract.RECORDS_DIR, contract.RUNNABLE_DIR):
        return None  # a bundle member's file: the member's own door names the cause
    return CORRUPT


@pytest.mark.parametrize("name", list(ARTIFACTS))
def test_a_copy_missing_or_cutting_short_any_one_file_refuses(
    saved: dict[str, Path], tmp_path: Path, name: str
) -> None:
    """Every file of the artifact, removed or cut to half its bytes, makes the copy refuse."""

    source = saved[name]
    relpaths = _files(source)
    assert MANIFEST in relpaths and len(relpaths) >= 2
    seen: list[str] = []
    for index, relpath in enumerate(relpaths):
        for removed in (True, False):
            if not removed and (source / relpath).stat().st_size == 0:
                continue  # an empty file has no tail to lose
            copy = tmp_path / f"{index}-{'gone' if removed else 'short'}"
            shutil.copytree(source, copy)
            if removed:
                (copy / relpath).unlink()
            else:
                _cut_tail(copy / relpath)
            for lazy in _lazy_modes(name):
                code = _refusal_code(copy, lazy=lazy)
                expected = _expected_code(relpath, removed=removed)
                assert expected is None or code == expected, (relpath, removed, lazy, code)
                seen.append(code)
            shutil.rmtree(copy)
    assert seen


@pytest.mark.parametrize("name", list(ARTIFACTS))
def test_every_cut_of_an_in_order_copy_refuses_until_it_completes(
    saved: dict[str, Path], tmp_path: Path, name: str
) -> None:
    """Copy the files one by one in ``rsync`` order and stop inside each: only the end loads."""

    source = saved[name]
    relpaths = _files(source)
    for stop, relpath in enumerate(relpaths):
        copy = tmp_path / f"cut-{stop}"
        copy.mkdir()
        for done in relpaths[:stop]:
            (copy / done).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / done, copy / done)
        target = copy / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        data = (source / relpath).read_bytes()
        target.write_bytes(data[: len(data) // 2])  # an empty file stays missing its peers
        for lazy in _lazy_modes(name):
            _refusal_code(copy, lazy=lazy)
        shutil.rmtree(copy)
    complete = tmp_path / "complete"
    shutil.copytree(source, complete)
    for lazy in _lazy_modes(name):
        _load(complete, lazy=lazy)


@pytest.mark.parametrize("name", list(ARTIFACTS))
def test_a_manifest_stripped_of_any_file_entry_refuses(
    saved: dict[str, Path], tmp_path: Path, name: str
) -> None:
    """A manifest that no longer lists a file the reader opens refuses: listing is required."""

    source = saved[name]
    # A bundle member's files are also listed, with digests, in the member's own manifest,
    # which its load checks (the coverage test below asserts that listing).
    listed = sorted(path for path in read_manifest(source)["files"] if not _in_member(path))
    assert listed
    for index, relpath in enumerate(listed):
        copy = tmp_path / f"strip-{index}"
        shutil.copytree(source, copy)
        manifest = read_manifest(copy)
        del manifest["files"][relpath]
        write_manifest(copy, manifest)
        for lazy in _lazy_modes(name):
            with warnings.catch_warnings():
                # The file is now unlisted, so the load may first name it in the
                # advisory; the refusal is what this test checks.
                warnings.filterwarnings("ignore", "Artifact .* unreferenced", TorchLensWarning)
                _refusal_code(copy, lazy=lazy, allowed=_strip_codes(relpath))
        shutil.rmtree(copy)


def _strip_codes(relpath: str) -> frozenset[str]:
    """The codes an unlisted file refuses with; a recipe the manifest names but does not list
    refuses ``manifest_schema_invalid``, and an intervention artifact without a listed recipe
    refuses at the recipe door as carrying none (``artifact_kind_mismatch``)."""

    if relpath == contract.INTERVENTION_FILE:
        return frozenset(
            {contract.REFUSAL_CODES["manifest_invalid"], contract.REFUSAL_CODES["kind_mismatch"]}
        )
    return INTEGRITY_CODES


def _in_member(relpath: str) -> bool:
    return relpath.startswith("members/")


def test_a_trace_whose_manifest_drops_its_recipe_listing_refuses_before_reading_it(
    saved: dict[str, Path], tmp_path: Path
) -> None:
    """The recipe file is read only when listed and digest-checked, eager and lazy."""

    copy = tmp_path / "unlisted-recipe"
    shutil.copytree(saved["trace_with_recipe"], copy)
    manifest = read_manifest(copy)
    assert manifest.get("intervention") is not None
    del manifest["files"][contract.INTERVENTION_FILE]
    write_manifest(copy, manifest)
    (copy / contract.INTERVENTION_FILE).write_text("{", encoding="utf-8")  # would not parse
    for lazy in (False, True):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Artifact .* unreferenced", TorchLensWarning)
            code = _refusal_code(copy, lazy=lazy)
        assert code == contract.REFUSAL_CODES["manifest_invalid"], code


def test_the_manifest_lists_every_other_file_the_save_wrote(saved: dict[str, Path]) -> None:
    """Coverage is the whole tree: no file of a saved artifact is left out of ``files``."""

    for name, root in saved.items():
        listed = set(read_manifest(root)["files"])
        on_disk = set(_files(root)) - {MANIFEST}
        assert on_disk == listed, (name, sorted(on_disk ^ listed))
        members = {path.split("/", 2)[1] for path in on_disk if _in_member(path)}
        assert (name == "bundle") == bool(members), name
        for member in members:
            member_root = root / "members" / member
            member_files = set(_files(member_root)) - {MANIFEST}
            assert member_files == set(read_manifest(member_root)["files"]), (name, member)


@pytest.mark.parametrize("lazy", [False, True])
def test_a_complete_copy_loads_with_the_saved_values(
    saved: dict[str, Path], tmp_path: Path, lazy: bool
) -> None:
    """The control: a full copy with fresh mtimes loads and matches the original."""

    copy = tmp_path / "full"
    shutil.copytree(saved["portable"], copy, copy_function=shutil.copyfile)
    original, copied = tl.load(saved["portable"]), _load(copy, lazy=lazy)
    assert len(copied.layers) == len(original.layers)
    final = original.layers[-1].out
    assert torch.equal(copied.layers[-1].out, final)
