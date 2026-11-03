"""Round-28 security regressions: torch._utils._rebuild dotted-walk RCE + sweep.

Four confirmed findings on the sparse ``.tlspec`` load surface:

A-R28-1 (CRITICAL, default-victim RCE) -- the ``SafeBundleUnpickler`` /
    ``_RenameAwareUnpickler`` ``torch._utils`` ``_rebuild*`` fast-path returned
    ``super().find_class`` for ANY ``name`` starting with ``_rebuild`` with NO gate.
    Under pickle protocol >= 4 the pickled ``name`` is attribute-walked, so a DOTTED
    name (``_rebuild_tensor_v2.__globals__.get``) walks OFF the reconstructor into
    ``torch._utils`` module globals and returns a REDUCE-invocable bound ``dict.get``
    -> ``_import_dotted_name`` -> ``os.system`` (fires at metadata unpickle, no
    ``.run()`` needed). This is the SAME dotted-walk escape class the r27 torch-type /
    preview branches close; the ``_rebuild`` branch was MISSED. The fix refuses a
    dotted ``_rebuild`` name BEFORE resolving and positively requires a torch-owned,
    non-type callable after.

DiD-1 (trust-gated, defense-in-depth) -- under trust, a dotted
    ``<trusted_mod>.<func>.__globals__`` returns a module-globals ``dict`` whose
    ``resolved_owner`` falls back to the non-denied pickled module (dicts have no
    ``__module__``), sidestepping the E2 resolved-real-module denylist recheck. The
    fix refuses dunder attribute-walks (and any resolved mapping) on the trusted
    branch too.

E-r28-1 (MEDIUM, "denied even under trust" contract violation) -- ``_frozen_importlib``
    / ``_frozen_importlib_external`` were absent from both denylists, so under trust
    ``resolve_import_ref`` reached the real import machinery
    (``_frozen_importlib:__import__``, ``_call_with_frames_removed`` universal call
    gadget). The fix adds both to ``_DENIED_MODULES`` and ``_DENIED_FOREIGN_MODULES``.

F-r28-1 (LOW, defense-in-depth) -- ``detect_tlspec_format`` / ``inspect_tlspec`` read
    ``manifest.json`` / ``spec.json`` for format classification BEFORE the loader
    symlink guards fire, following a symlinked child JSON (or root) out of the bundle.
    The fix mirrors the loader ``_reject_symlink_path`` guards at classification time.
"""

from __future__ import annotations

import importlib
import io
import json
import pickle
import sys
from pathlib import Path

import pytest

from torchlens._io._safe_unpickle import (
    SafeBundleUnpickler,
)
from torchlens._io.bundle import _RenameAwareUnpickler
from torchlens.io import inspect_tlspec

_UNPICKLERS = [SafeBundleUnpickler, _RenameAwareUnpickler]


# ---------------------------------------------------------------------------
# Raw pickle-opcode builders (protocol 4). Hand-assembled so the pickled
# (module, name) can be an arbitrary DOTTED name -- the STACK_GLOBAL dotted
# attribute-walk is only performed at protocol >= 4, so every escape vector must
# go through a real ``.load()`` of a proto-4 stream (a bare ``find_class`` call
# runs at proto 0 = a single getattr, which never walks). Mirrors
# tests/test_r27_dotted_walk_rce.py.
# ---------------------------------------------------------------------------


def _short_binunicode(text: str) -> bytes:
    """Encode ``text`` as a SHORT_BINUNICODE opcode (length < 256)."""

    raw = text.encode("utf-8")
    assert len(raw) < 256
    return b"\x8c" + bytes([len(raw)]) + raw


def _stack_global(module: str, name: str) -> bytes:
    """Push a global resolved by ``STACK_GLOBAL`` (pops name, module off the stack)."""

    return _short_binunicode(module) + _short_binunicode(name) + b"\x93"


def _stop_global_pickle(module: str, name: str) -> bytes:
    """Proto-4 pickle that resolves ``module.name`` via STACK_GLOBAL then STOPs."""

    return b"\x80\x04" + _stack_global(module, name) + b"."


def _reduce_global_pickle(module: str, name: str, str_args: list[str]) -> bytes:
    """Proto-4 pickle that STACK_GLOBAL-resolves ``module.name`` then REDUCE-calls it.

    The REDUCE invokes ``callable([*str_args])`` -- e.g. a bound ``dict.get`` off a
    module-globals mapping -- which would fire a gadget if the resolved global were
    admitted.
    """

    body = b"\x80\x04"  # PROTO 4
    body += _stack_global(module, name)
    body += b"]"  # EMPTY_LIST
    body += b"("  # MARK
    for arg in str_args:
        body += _short_binunicode(arg)
    body += b"e"  # APPENDS  -> list is [*str_args]
    body += b"\x85"  # TUPLE1   -> (list,)
    body += b"R"  # REDUCE   -> callable(list)
    body += b"."  # STOP
    return body


# ---------------------------------------------------------------------------
# A-R28-1 -- torch._utils._rebuild dotted-name attribute-walk escape.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("unpickler_cls", _UNPICKLERS)
def test_rebuild_dotted_walk_to_globals_get_denied_no_exec(
    unpickler_cls: type[pickle.Unpickler],
    tmp_path: Path,
) -> None:
    """The proven vector ``_rebuild_tensor_v2.__globals__.get`` is denied, nothing execs.

    A proto-4 stream that STACK_GLOBAL-resolves the dotted ``_rebuild`` name and then
    REDUCE-invokes the result MUST raise at ``find_class`` (before REDUCE runs) and
    execute nothing.
    """

    sentinel = tmp_path / "r28_rebuild_pwned"
    payload = _reduce_global_pickle(
        "torch._utils",
        "_rebuild_tensor_v2.__globals__.get",
        ["/bin/sh", "-c", f"touch {sentinel}"],
    )
    with pytest.raises(pickle.UnpicklingError, match="dotted"):
        unpickler_cls(io.BytesIO(payload)).load()
    assert not sentinel.exists()


# ---------------------------------------------------------------------------
# DiD-1 -- trusted-foreign branch dunder-walk into module globals.
# ---------------------------------------------------------------------------


@pytest.fixture
def _trusted_os_module(tmp_path: Path) -> str:
    """A foreign module that (like ~75 real ones) does ``import os`` at top level."""

    mod_name = "r28_unpickler_trusted_mod"
    (tmp_path / f"{mod_name}.py").write_text("import os\n\ndef hook(x):\n    return x\n")
    sys.path.insert(0, str(tmp_path))
    importlib.import_module(mod_name)
    try:
        yield mod_name
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop(mod_name, None)


def test_trusted_globals_mapping_return_denied(_trusted_os_module: str) -> None:
    """A trusted dotted name resolving to the bare ``__globals__`` mapping is denied."""

    payload = _stop_global_pickle(_trusted_os_module, "hook.__globals__")
    with pytest.raises(pickle.UnpicklingError, match="dunder"):
        SafeBundleUnpickler(io.BytesIO(payload), trust_custom_callables=True).load()


# ---------------------------------------------------------------------------
# E-r28-1 -- frozen import machinery denied under trust.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("unpickler_cls", _UNPICKLERS)
def test_frozen_importlib_unpickle_denied_under_trust(
    unpickler_cls: type[pickle.Unpickler],
) -> None:
    """The unpickler hard-denies ``_frozen_importlib`` even under broad trust."""

    unpickler = unpickler_cls(io.BytesIO(b""), trust_custom_callables=True)
    with pytest.raises(pickle.UnpicklingError, match="dangerous"):
        unpickler.find_class("_frozen_importlib", "__import__")


# ---------------------------------------------------------------------------
# F-r28-1 -- symlinked format-detection metadata path.
# ---------------------------------------------------------------------------


def test_inspect_tlspec_rejects_symlinked_spec(tmp_path: Path) -> None:
    """A symlinked ``spec.json`` child is refused by ``inspect_tlspec``."""

    from torchlens._io import ArtifactError

    outside = tmp_path / "outside_spec.json"
    outside.write_text(json.dumps({"format_version": 1}), encoding="utf-8")
    bundle = tmp_path / "spec_bundle.tlspec"
    bundle.mkdir()
    (bundle / "spec.json").symlink_to(outside)
    with pytest.raises(ArtifactError, match="symlink"):
        inspect_tlspec(bundle)
