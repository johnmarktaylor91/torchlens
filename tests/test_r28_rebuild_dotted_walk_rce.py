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
import torch

from torchlens._io._safe_unpickle import (
    SafeBundleUnpickler,
    _DENIED_FOREIGN_MODULES,
    _module_denied,
    _name_has_dunder_walk,
)
from torchlens._io.bundle import _RenameAwareUnpickler
from torchlens.io import detect_tlspec_format, inspect_tlspec
from torchlens.utils._callable_safety import _DENIED_MODULES

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


@pytest.mark.smoke
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


@pytest.mark.smoke
@pytest.mark.parametrize("unpickler_cls", _UNPICKLERS)
def test_rebuild_dotted_name_refused_before_walk(
    unpickler_cls: type[pickle.Unpickler],
) -> None:
    """ANY dotted ``_rebuild`` name is refused up front (never triggers the walk)."""

    payload = _stop_global_pickle("torch._utils", "_rebuild_tensor_v2.__globals__")
    with pytest.raises(pickle.UnpicklingError, match="dotted"):
        unpickler_cls(io.BytesIO(payload)).load()


@pytest.mark.smoke
def test_legit_rebuild_reconstructors_still_admit() -> None:
    """Genuine ``_rebuild_*`` reconstructors (single-segment names) still resolve."""

    unpickler = SafeBundleUnpickler(io.BytesIO(b""))
    for name in ("_rebuild_tensor_v2", "_rebuild_parameter"):
        resolved = unpickler.find_class("torch._utils", name)
        assert resolved is getattr(torch._utils, name)
        assert callable(resolved) and not isinstance(resolved, type)


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


@pytest.mark.smoke
def test_trusted_globals_walk_to_os_system_denied(_trusted_os_module: str, tmp_path: Path) -> None:
    """Under trust, ``<trusted>.hook.__globals__.get`` -> module globals is denied.

    Without the DiD-1 guard this walk returns the module-globals ``dict`` (no
    ``__module__``), the resolved-owner denylist recheck falls back to the trusted
    (non-denied) pickled module, and a REDUCE-invocable ``dict.get`` of ``os`` is
    handed back. The dunder-walk guard must refuse it BEFORE the walk.
    """

    sentinel = tmp_path / "r28_trusted_globals_pwned"
    payload = _reduce_global_pickle(
        _trusted_os_module,
        "hook.__globals__.get",
        [str(sentinel)],
    )
    with pytest.raises(pickle.UnpicklingError, match="dunder"):
        SafeBundleUnpickler(io.BytesIO(payload), trust_custom_callables=True).load()
    assert not sentinel.exists()
    # A narrow allowlist keyed on the trusted module is likewise not fooled.
    with pytest.raises(pickle.UnpicklingError, match="dunder"):
        SafeBundleUnpickler(
            io.BytesIO(payload), allowed_custom_callable_modules={_trusted_os_module}
        ).load()


@pytest.mark.smoke
def test_trusted_globals_mapping_return_denied(_trusted_os_module: str) -> None:
    """A trusted dotted name resolving to the bare ``__globals__`` mapping is denied."""

    payload = _stop_global_pickle(_trusted_os_module, "hook.__globals__")
    with pytest.raises(pickle.UnpicklingError, match="dunder"):
        SafeBundleUnpickler(io.BytesIO(payload), trust_custom_callables=True).load()


@pytest.mark.smoke
def test_trusted_legit_single_name_still_resolves(_trusted_os_module: str) -> None:
    """A genuine (single-name) trusted callable still resolves (no regression)."""

    unpickler = SafeBundleUnpickler(io.BytesIO(b""), trust_custom_callables=True)
    resolved = unpickler.find_class(_trusted_os_module, "hook")
    assert getattr(resolved, "__name__", None) == "hook"


@pytest.mark.smoke
def test_name_has_dunder_walk_helper() -> None:
    """The dunder-walk predicate flags every walk vector and passes legit qualnames."""

    for bad in (
        "hook.__globals__.get",
        "f.__globals__",
        "x.__builtins__",
        "T.__reduce__",
        "obj.__class__.__mro__",
        "__dict__",
    ):
        assert _name_has_dunder_walk(bad)
    for good in ("hook", "Outer.Inner", "_rebuild_tensor_v2", "recipe_fn"):
        assert not _name_has_dunder_walk(good)


# ---------------------------------------------------------------------------
# E-r28-1 -- frozen import machinery denied under trust.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize("module", ["_frozen_importlib", "_frozen_importlib_external"])
def test_frozen_importlib_in_both_denylists(module: str) -> None:
    """Both frozen import modules are hard-denied in both denylists (incl. submodules)."""

    assert module in _DENIED_MODULES
    assert module in _DENIED_FOREIGN_MODULES
    assert _module_denied(module)
    assert _module_denied(module + ".SourceFileLoader")


@pytest.mark.smoke
def test_resolve_import_ref_frozen_importlib_denied_under_trust() -> None:
    """``resolve_import_ref('_frozen_importlib:__import__', trust=True)`` is DENIED."""

    from torchlens.intervention.errors import UntrustedCallableError
    from torchlens.intervention.resolver import resolve_import_ref

    with pytest.raises(UntrustedCallableError):
        resolve_import_ref("_frozen_importlib:__import__")
    with pytest.raises(UntrustedCallableError):
        resolve_import_ref("_frozen_importlib:__import__", trust_custom_callables=True)
    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(
            "_frozen_importlib_external:_NamespaceLoader", trust_custom_callables=True
        )


@pytest.mark.smoke
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


def _write_bundle_manifest(bundle: Path, obj: dict) -> None:
    """Write a minimal ``manifest.json`` inside ``bundle``."""

    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "manifest.json").write_text(json.dumps(obj), encoding="utf-8")


@pytest.mark.smoke
def test_detect_tlspec_format_rejects_symlinked_manifest(tmp_path: Path) -> None:
    """A symlinked ``manifest.json`` child is refused during format detection."""

    from torchlens._io import TorchLensIOError

    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"kind": "x", "tlspec_version": 1}), encoding="utf-8")
    bundle = tmp_path / "bundle.tlspec"
    bundle.mkdir()
    (bundle / "manifest.json").symlink_to(outside)
    with pytest.raises(TorchLensIOError, match="symlink"):
        detect_tlspec_format(bundle)


@pytest.mark.smoke
def test_detect_tlspec_format_rejects_symlinked_root(tmp_path: Path) -> None:
    """A symlinked bundle ROOT directory is refused during format detection."""

    from torchlens._io import TorchLensIOError

    real_bundle = tmp_path / "real.tlspec"
    _write_bundle_manifest(real_bundle, {"kind": "x", "tlspec_version": 1})
    link = tmp_path / "link.tlspec"
    link.symlink_to(real_bundle, target_is_directory=True)
    with pytest.raises(TorchLensIOError, match="symlink"):
        detect_tlspec_format(link)


@pytest.mark.smoke
def test_inspect_tlspec_rejects_symlinked_spec(tmp_path: Path) -> None:
    """A symlinked ``spec.json`` child is refused by ``inspect_tlspec``."""

    from torchlens._io import TorchLensIOError

    outside = tmp_path / "outside_spec.json"
    outside.write_text(json.dumps({"format_version": 1}), encoding="utf-8")
    bundle = tmp_path / "spec_bundle.tlspec"
    bundle.mkdir()
    (bundle / "spec.json").symlink_to(outside)
    with pytest.raises(TorchLensIOError, match="symlink"):
        inspect_tlspec(bundle)


@pytest.mark.smoke
def test_detect_tlspec_format_regular_bundle_unaffected(tmp_path: Path) -> None:
    """A regular (non-symlinked) bundle still classifies normally (no regression)."""

    bundle = tmp_path / "regular.tlspec"
    _write_bundle_manifest(bundle, {"kind": "x", "tlspec_version": 1})
    assert detect_tlspec_format(bundle) == "v2.0_unified"
    assert inspect_tlspec(bundle)["kind"] == "x"
