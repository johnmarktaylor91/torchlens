"""Generative tlspec parse-robustness sweep (R73, round 3).

The historical parse-robustness tests ENUMERATED hand-picked corruptions; this
module GENERATES them from the artifact itself, so the coverage tracks the
manifest schema instead of a hand-list:

* the key sweeps parametrize over whatever top-level keys a freshly saved
  manifest actually carries — a NEW manifest key automatically enters the
  required-key contract and forces a conscious ledger decision;
* the seeded sweeps (truncations, byte flips) draw offsets from a seeded
  RNG — deterministic by default, and ``TORCHLENS_FUZZ_SEED`` re-seeds them
  for the nightly fresh-seed leg (the R73 "exploration, not just pinning"
  half; failures name the seed for local reproduction).

Contract, pinned from the probed loader surface (2026-08-15): every
structural corruption must be refused with a torchlens-typed
``ArtifactError`` — never a silent success, never an untyped stack trace.
The one ledgered tolerance: ``_OPTIONAL_KEYS`` may be ABSENT
(legacy-manifest compatibility) but still type-check when present. The
former ``tlspec_version``-deletion wart (untyped ``FileNotFoundError`` out
of the legacy-format dispatch) is FIXED — incoherent format markers now
refuse typed at the dispatch (``tlspec_format_markers_incoherent``).
"""

from __future__ import annotations

import json
import os
import random
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.errors import ArtifactError

pytestmark = pytest.mark.heavy

#: Deterministic defaults; the nightly fresh-seed leg overrides via env. The
#: two sweeps use distinct streams (base and base+1) so a shared seed never
#: replays identical offsets across them.
_FUZZ_SEED_ENV = os.environ.get("TORCHLENS_FUZZ_SEED")
_TRUNCATION_SEED = 1973 if _FUZZ_SEED_ENV is None else int(_FUZZ_SEED_ENV)
_BYTE_FLIP_SEED = 2026 if _FUZZ_SEED_ENV is None else int(_FUZZ_SEED_ENV) + 1

#: Top-level manifest keys that may legitimately be ABSENT (older manifests
#: predate them; the loader defaults them). Deleting any OTHER key must be a
#: typed refusal — a new manifest key lands in the required contract until
#: consciously ledgered here.
_OPTIONAL_KEYS = frozenset(
    {
        "unsupported_tensors",
        "provenance",
        "custom_attributes_disclosure",
        "buffer_values_disclosure",
        "kind",
    }
)


@pytest.fixture(scope="module")
def seed_artifact(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Save one pristine tiny-trace tlspec directory for the module."""

    path = tmp_path_factory.mktemp("tlspec_fuzz") / "seed.tlspec"
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
    log = tl.trace(model, torch.rand(2, 4))
    try:
        log.save(str(path))
        yield path
    finally:
        log.cleanup()


def _corrupt_copy(seed: Path, tmp_path: Path) -> Path:
    """Return a fresh writable copy of the pristine artifact."""

    target = tmp_path / "mutant.tlspec"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(seed, target)
    return target


def _manifest_keys(seed: Path) -> list[str]:
    """Discover the artifact's top-level manifest keys."""

    return list(json.loads((seed / "manifest.json").read_text(encoding="utf-8")))


def _rewrite_manifest(artifact: Path, mutate) -> None:
    """Apply one structural mutation to the copied manifest."""

    manifest_path = artifact / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(data)
    manifest_path.write_text(json.dumps(data), encoding="utf-8")


def test_every_toplevel_key_deletion_is_adjudicated(seed_artifact: Path, tmp_path: Path) -> None:
    """Deleting each discovered key: typed refusal or ledgered tolerance.

    ``tlspec_version`` deletion needs no special arm anymore: an artifact
    that classifies as a v2.16 intervention bundle but carries no
    ``spec.json`` refuses typed at the dispatch
    (``tlspec_format_markers_incoherent``; formerly an untyped
    ``FileNotFoundError`` wart, R73).
    """

    outcomes: list[str] = []
    for key in _manifest_keys(seed_artifact):
        artifact = _corrupt_copy(seed_artifact, tmp_path)
        _rewrite_manifest(artifact, lambda data, key=key: data.pop(key))
        if key in _OPTIONAL_KEYS:
            loaded = tl.load(str(artifact))
            assert type(loaded).__name__ == "Trace"
            loaded.cleanup()
        else:
            try:
                tl.load(str(artifact))
            except ArtifactError:
                pass
            else:
                outcomes.append(key)
    assert not outcomes, (
        "deleting these manifest keys loaded SILENTLY despite not being in the "
        f"optional ledger — either a lost required-field check or a key that "
        f"needs a conscious _OPTIONAL_KEYS entry: {outcomes}"
    )


def test_every_toplevel_key_type_swap_is_typed_refusal(seed_artifact: Path, tmp_path: Path) -> None:
    """Swapping each discovered key's type must raise ArtifactError."""

    survivors: list[str] = []
    for key in _manifest_keys(seed_artifact):
        artifact = _corrupt_copy(seed_artifact, tmp_path)

        def swap(data: dict, key: str = key) -> None:
            data[key] = [123] if not isinstance(data[key], list) else "oops"

        _rewrite_manifest(artifact, swap)
        try:
            tl.load(str(artifact))
        except ArtifactError:
            pass
        except Exception as exc:  # noqa: BLE001 - adjudicating the full surface
            survivors.append(f"{key} -> untyped {type(exc).__name__}")
        else:
            survivors.append(f"{key} -> silent success")
    assert not survivors, f"type-swapped manifest keys escaped the typed refusal: {survivors}"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("blob_id", "9999999999"),
        ("dtype", "float999"),
        ("shape", "bogus"),
        ("relative_path", "../../outside.safetensors"),
        ("relative_path", "/etc/passwd"),
    ],
)
def test_tensor_entry_mutations_are_typed_refusals(
    seed_artifact: Path, tmp_path: Path, field: str, value: str
) -> None:
    """Tensor-entry corruption (including path escapes) must refuse typed."""

    artifact = _corrupt_copy(seed_artifact, tmp_path)
    _rewrite_manifest(artifact, lambda data: data["tensors"][0].__setitem__(field, value))
    with pytest.raises(ArtifactError):
        tl.load(str(artifact))


def test_seeded_truncations_are_typed_refusals(seed_artifact: Path, tmp_path: Path) -> None:
    """Seeded truncations of manifest and metadata must refuse typed."""

    rng = random.Random(_TRUNCATION_SEED)
    manifest_bytes = (seed_artifact / "manifest.json").read_bytes()
    metadata_bytes = (seed_artifact / "metadata.pkl").read_bytes()
    for _ in range(8):
        artifact = _corrupt_copy(seed_artifact, tmp_path)
        cut = rng.randrange(1, len(manifest_bytes))
        (artifact / "manifest.json").write_bytes(manifest_bytes[:cut])
        try:
            tl.load(str(artifact))
        except ArtifactError:
            continue
        pytest.fail(
            f"manifest truncated at byte {cut} loaded silently (seed {_TRUNCATION_SEED}; reproduce with TORCHLENS_FUZZ_SEED)"
        )
    for _ in range(4):
        artifact = _corrupt_copy(seed_artifact, tmp_path)
        cut = rng.randrange(1, len(metadata_bytes))
        (artifact / "metadata.pkl").write_bytes(metadata_bytes[:cut])
        try:
            tl.load(str(artifact))
        except ArtifactError:
            continue
        pytest.fail(
            f"metadata truncated at byte {cut} loaded silently (seed {_TRUNCATION_SEED}; reproduce with TORCHLENS_FUZZ_SEED)"
        )


def test_seeded_manifest_byte_flips_never_escape_untyped(
    seed_artifact: Path, tmp_path: Path
) -> None:
    """Random byte flips: typed refusal or clean load, never stack soup.

    A flip landing in an insignificant JSON position can legitimately load;
    the tripwire is the third outcome — an UNTYPED exception leaking from the
    parse path.
    """

    rng = random.Random(_BYTE_FLIP_SEED)
    manifest_bytes = (seed_artifact / "manifest.json").read_bytes()
    escapes: list[str] = []
    for index in range(12):
        artifact = _corrupt_copy(seed_artifact, tmp_path)
        flipped = bytearray(manifest_bytes)
        offset = rng.randrange(len(flipped))
        flipped[offset] ^= 0xFF
        (artifact / "manifest.json").write_bytes(bytes(flipped))
        try:
            loaded = tl.load(str(artifact))
        except ArtifactError:
            continue
        except Exception as exc:  # noqa: BLE001 - adjudicating the full surface
            escapes.append(f"case {index} offset {offset}: {type(exc).__name__}: {exc}")
        else:
            loaded.cleanup()
    assert not escapes, (
        f"manifest byte flips escaped the typed surface (seed {_BYTE_FLIP_SEED}; reproduce "
        f"with TORCHLENS_FUZZ_SEED): {escapes}"
    )


def test_non_utf8_manifest_byte_is_typed_refusal(seed_artifact: Path, tmp_path: Path) -> None:
    """R73: a non-UTF8 manifest byte refuses typed, never a raw decode error.

    Fail-before: ``_json.load_bounded``/``read_bounded`` decoded the bounded
    bytes unguarded, so the raw ``UnicodeDecodeError`` leaked through every
    JSON-boundary handler untyped (this pin was a strict xfail and the sweep
    above carried a matching allowance until the chokepoint fix landed).
    """

    artifact = _corrupt_copy(seed_artifact, tmp_path)
    corrupted = bytearray((seed_artifact / "manifest.json").read_bytes())
    corrupted[len(corrupted) // 2] = 0xFF
    (artifact / "manifest.json").write_bytes(bytes(corrupted))
    with pytest.raises(ArtifactError):
        tl.load(str(artifact))


def test_blob_corruption_is_typed_refusal(seed_artifact: Path, tmp_path: Path) -> None:
    """Blob deletion, truncation, and byte flips must all refuse typed."""

    blob_names = sorted(p.name for p in (seed_artifact / "blobs").iterdir())
    assert blob_names, "seed artifact saved no blobs"
    first = blob_names[0]

    artifact = _corrupt_copy(seed_artifact, tmp_path)
    (artifact / "blobs" / first).unlink()
    with pytest.raises(ArtifactError):
        tl.load(str(artifact))

    artifact = _corrupt_copy(seed_artifact, tmp_path)
    blob_path = artifact / "blobs" / first
    blob_bytes = blob_path.read_bytes()
    blob_path.write_bytes(blob_bytes[: len(blob_bytes) // 2])
    with pytest.raises(ArtifactError):
        tl.load(str(artifact))

    artifact = _corrupt_copy(seed_artifact, tmp_path)
    blob_path = artifact / "blobs" / first
    flipped = bytearray(blob_path.read_bytes())
    flipped[len(flipped) // 2] ^= 0xFF
    blob_path.write_bytes(bytes(flipped))
    with pytest.raises(ArtifactError):
        tl.load(str(artifact))
