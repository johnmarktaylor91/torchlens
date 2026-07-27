"""Round-34 security + IO-robustness regressions on the sparse ``.tlspec`` surface.

Five confirmed findings, all on files disjoint from the witness/execution lane:

secE-1 (HIGH, trust-gated RCE) -- ``intervention/resolver.py``: the
    ``path_claims_torchlens`` -> ``not _is_torchlens_owned(obj)`` leg re-enforced
    trust on the RESOLVED-owner STRING but omitted the ``is_pure_forward_callable``
    torch-purity gate the sibling genuinely-foreign branch applies. Under broad
    ``trust_custom_callables=True`` an attacker key walking a side-effecting torch
    callable off a torchlens module (``torchlens.backends.torch.ops:torch.from_file``,
    ``Tensor.apply_`` / ``resize_`` / ``set_``) RESOLVED where the sibling spelling
    DENIED. The fix routes the walked object through ``is_pure_forward_callable`` on
    its REAL identity, closing the ``__module__ is None`` / bare-``"torch"`` fallback
    loophole while preserving the r33 operator carve-out and default-victim deny.

secF-1 (MED, default-victim DoS) -- ``_io/rehydrate.py``: ``_rehydrate_small_raw_images``
    handed attacker bytes to ``PIL.Image.open(...).load()`` with no byte cap, no
    dimension bound, and no try/except -> decompression-bomb allocation + uncaught
    crash on a plain ``tl.load()``. The fix imposes the canonical save-time byte cap,
    a header-only dimension check before decode, and fails closed to the inert dict.

secF-2 (LOW, default-victim DoS) -- ``_io/bundle.py``: nested-bundle member recursion
    had no cycle guard and ``_resolve_bundle_member_path`` accepted self-reference
    (``"."`` / ``""``) -> unbounded recursion. The fix rejects self-referential member
    paths and threads a resolved-path visited set + a nesting-depth cap.

secC (LOW robustness) -- ``_io/runnable_load.py``: ``_parse_literal`` built
    ``LiteralAtom(kind=..., value=...)`` without checking the JSON value type matched
    the declared ``kind`` (a ``kind=int`` atom could carry a list/dict/str). The fix
    enforces value-type-per-kind at parse.

fastlog (LOW defense-in-depth) -- ``fastlog/storage_disk.py``: ``_dtype_from_name``
    did ``getattr(torch, name)`` with no ``isinstance(_, torch.dtype)`` guard (the
    sibling dtype resolvers have one). The fix adds the guard for parity.
"""

from __future__ import annotations

import io
import json
import struct
import zlib
from pathlib import Path

import pytest
import torch

import torchlens as tl
from torchlens._io import TorchLensIOError
from torchlens._io.bundle import (
    _MAX_BUNDLE_NESTING_DEPTH,
    _load_unified_bundle_directory,
    _resolve_bundle_member_path,
)
from torchlens._io.rehydrate import _rehydrate_small_raw_images
from torchlens._io.runnable_load import _parse_literal
from torchlens._io.scrub import (
    _RAW_IMAGE_SENTINEL,
    _RAW_INPUT_IMAGE_BYTES_LIMIT,
    _RAW_INPUT_IMAGE_MAX_EDGE,
)
from torchlens.fastlog.storage_disk import _dtype_from_name
from torchlens.intervention.errors import UntrustedCallableError
from torchlens.intervention.resolver import resolve_import_ref
from torchlens.runnable import LiteralAtom, LiteralAtomKind
from torchlens.utils._callable_safety import is_pure_forward_callable

# --------------------------------------------------------------------------- #
# secE-1 -- resolver torchlens-walk branch now applies the torch-purity gate.
# --------------------------------------------------------------------------- #

# Attacker keys that WALK a side-effecting torch/Tensor callable off a torchlens
# module. Every one is exactly what ``is_pure_forward_callable`` exists to deny.
_WALK = "torchlens.backends.torch.ops:{}"
_SIDE_EFFECTING_WALKED = [
    "torch.from_file",  # create/truncate a file, or read arbitrary bytes into a tensor
    "torch.compile",  # compile/invoke an arbitrary callable + mutate dynamo globals
    "torch.Tensor.apply_",  # invoke an attacker callable per element
    "torch.Tensor.map_",  # invoke an attacker callable per element
    "torch.Tensor.resize_",  # expose uninitialized heap memory (info leak)
    "torch.Tensor.set_",  # repoint tensor storage
]


@pytest.mark.smoke
@pytest.mark.parametrize("terminal", _SIDE_EFFECTING_WALKED)
def test_secE1_torchlens_walk_denies_side_effecting_even_under_trust(terminal: str) -> None:
    """A side-effecting torch callable walked off a torchlens module never resolves."""

    path = _WALK.format(terminal)
    # Default (untrusting) victim: denied.
    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(path)
    # Broad trust: STILL denied -- purity parity with the sibling foreign branch.
    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(path, trust_custom_callables=True)
    # An explicit torch allowlist entry does NOT re-admit it either.
    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(path, allowed_custom_callable_modules={"torch"})


@pytest.mark.smoke
def test_secE1_pure_torch_op_walk_still_resolves_under_trust() -> None:
    """A PURE torch op walked off a torchlens module still resolves under trust.

    The purity gate must not over-deny: it denies only the side-effecting surface.
    """

    fn = resolve_import_ref(_WALK.format("torch.relu"), trust_custom_callables=True)
    assert fn is torch.relu


@pytest.mark.smoke
def test_secE1_purity_gate_denies_the_walked_surface_directly() -> None:
    """The purity gate itself refuses every side-effecting walked callable."""

    assert not is_pure_forward_callable(torch.from_file)
    assert not is_pure_forward_callable(torch.compile)
    assert not is_pure_forward_callable(torch.Tensor.apply_)
    assert not is_pure_forward_callable(torch.Tensor.resize_)
    assert not is_pure_forward_callable(torch.Tensor.set_)


@pytest.mark.smoke
def test_secE1_default_victim_denies_torchlens_walk() -> None:
    """The default untrusting victim denies a torchlens-path walk to a foreign callable."""

    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(_WALK.format("torch.from_file"))


# --------------------------------------------------------------------------- #
# secF-1 -- raw-image rehydration is bounded and fails closed.
# --------------------------------------------------------------------------- #


def _png_bytes(width: int, height: int) -> bytes:
    """Return a minimal (bomb-shaped) PNG declaring ``width`` x ``height``."""

    sig = b"\x89PNG\r\n\x1a\n"

    def chunk(typ: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + typ
            + data
            + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00" * 16)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


@pytest.mark.smoke
def test_secF1_malformed_raw_image_degrades_without_raising() -> None:
    """Malformed sentinel bytes degrade to the inert dict instead of crashing load."""

    pytest.importorskip("PIL.Image")
    bad = {_RAW_IMAGE_SENTINEL: True, "data": b"\x89PNG\r\n\x1a\n" + b"\x00" * 40}
    assert _rehydrate_small_raw_images(bad) is bad


@pytest.mark.smoke
def test_secF1_decompression_bomb_rejected() -> None:
    """A tiny blob declaring huge dimensions is refused before ``.load()`` allocates."""

    pytest.importorskip("PIL.Image")
    bomb = {_RAW_IMAGE_SENTINEL: True, "data": _png_bytes(60000, 60000)}
    assert _rehydrate_small_raw_images(bomb) is bomb


@pytest.mark.smoke
def test_secF1_oversized_bytes_rejected() -> None:
    """Bytes over the canonical save-time cap never reach the decoder."""

    pytest.importorskip("PIL.Image")
    huge = {
        _RAW_IMAGE_SENTINEL: True,
        "data": b"\x89PNG\r\n\x1a\n" + b"\x00" * (_RAW_INPUT_IMAGE_BYTES_LIMIT + 1),
    }
    assert _rehydrate_small_raw_images(huge) is huge


@pytest.mark.smoke
def test_secF1_oversized_declared_dimensions_rejected() -> None:
    """A declared edge over the save-time max is refused (even within the byte cap)."""

    pytest.importorskip("PIL.Image")
    oversized = {
        _RAW_IMAGE_SENTINEL: True,
        "data": _png_bytes(_RAW_INPUT_IMAGE_MAX_EDGE + 1, 8),
    }
    assert _rehydrate_small_raw_images(oversized) is oversized


@pytest.mark.smoke
def test_secF1_legit_small_image_still_decodes() -> None:
    """A genuine small image within bounds still decodes to a PIL image."""

    image_mod = pytest.importorskip("PIL.Image")
    buf = io.BytesIO()
    image_mod.new("RGB", (16, 16), color=(1, 2, 3)).save(buf, format="PNG")
    record = {_RAW_IMAGE_SENTINEL: True, "data": buf.getvalue()}
    out = _rehydrate_small_raw_images(record)
    assert not isinstance(out, dict)
    assert tuple(out.size) == (16, 16)


# --------------------------------------------------------------------------- #
# secF-2 -- nested-bundle recursion is bounded (self-reference + depth cap).
# --------------------------------------------------------------------------- #


def _write_unified_bundle(directory: Path, members: list[dict[str, str]]) -> None:
    """Write a minimal unified ``kind=bundle`` directory with the given members."""

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(
        json.dumps({"kind": "bundle", "tlspec_version": "2.0"})
    )
    (directory / "bundle.json").write_text(json.dumps({"members": members}))


@pytest.mark.smoke
def test_secF2_self_referential_bundle_member_rejected(tmp_path: Path) -> None:
    """A ``path='.'`` self-referential member raises instead of recursing forever."""

    bundle = tmp_path / "loop.tlspec"
    _write_unified_bundle(bundle, [{"name": "loop", "path": "."}])
    with pytest.raises(TorchLensIOError):
        tl.load(str(bundle))


@pytest.mark.smoke
@pytest.mark.parametrize("selfref", [".", "", "./"])
def test_secF2_member_path_resolving_to_root_rejected(tmp_path: Path, selfref: str) -> None:
    """The member-path resolver rejects any path collapsing onto the bundle root."""

    with pytest.raises(TorchLensIOError):
        _resolve_bundle_member_path(tmp_path, selfref)


@pytest.mark.smoke
def test_secF2_nesting_depth_cap_enforced(tmp_path: Path) -> None:
    """A member chain deeper than the cap is refused before the stack is exhausted."""

    _write_unified_bundle(tmp_path, [])
    saturated = frozenset(
        Path(f"/nonexistent/bundle/{index}") for index in range(_MAX_BUNDLE_NESTING_DEPTH)
    )
    with pytest.raises(TorchLensIOError):
        _load_unified_bundle_directory(tmp_path, tmp_path / "bundle.json", bundle_visited=saturated)


# --------------------------------------------------------------------------- #
# secC -- literal-atom value type must match its declared kind.
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
@pytest.mark.parametrize(
    "kind,bad_value",
    [
        ("int", [1, 2, 3]),
        ("int", {"a": 1}),
        ("int", "5"),
        ("int", True),  # bool is not an int here (encoder emits BOOL for bools)
        ("bool", 5),
        ("bool", "true"),
        ("str", 5),
        ("str", None),
        ("float", 1),  # JSON int is not a float atom
        ("float", "1.0"),
        ("none", 7),
        ("ellipsis", "x"),
        ("nonfinite_float", 5),
    ],
)
def test_secC_atom_value_type_mismatch_rejected(kind: str, bad_value: object) -> None:
    """A tagged atom whose JSON value contradicts its kind is refused at parse."""

    with pytest.raises(ValueError):
        _parse_literal({"kind": kind, "value": bad_value})


@pytest.mark.smoke
@pytest.mark.parametrize(
    "kind,good_value",
    [
        ("none", None),
        ("ellipsis", None),
        ("bool", True),
        ("int", 5),
        ("float", 1.5),
        ("str", "hi"),
        ("nonfinite_float", "inf"),
    ],
)
def test_secC_legit_atoms_still_parse(kind: str, good_value: object) -> None:
    """Every legitimate encoder-shaped atom still round-trips through the parser."""

    atom = _parse_literal({"kind": kind, "value": good_value})
    assert isinstance(atom, LiteralAtom)
    assert atom.kind is LiteralAtomKind(kind)
    assert atom.value == good_value or (good_value is None and atom.value is None)


# --------------------------------------------------------------------------- #
# fastlog -- dtype-name resolution is isinstance-guarded (parity).
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_fastlog_dtype_from_name_resolves_real_dtypes() -> None:
    """A real dtype name (or ``None``) resolves as before."""

    assert _dtype_from_name(None) is None
    assert _dtype_from_name("float32") is torch.float32
    assert _dtype_from_name("int64") is torch.int64


@pytest.mark.smoke
@pytest.mark.parametrize(
    "bad_name",
    ["load", "save", "compile", "nn", "__class__", "os", "system"],
)
def test_fastlog_dtype_from_name_rejects_non_dtype(bad_name: str) -> None:
    """A non-dtype torch attribute name is refused instead of silently resolving."""

    with pytest.raises(TorchLensIOError):
        _dtype_from_name(bad_name)
