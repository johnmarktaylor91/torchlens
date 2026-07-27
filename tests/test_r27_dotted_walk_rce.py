"""Round-27 security regressions: dotted-name attribute-walk RCE + CVE-2025-32434.

Three confirmed findings on the DEFAULT (untrusting) ``tl.load(path)`` + ``.run()``
surface for an attacker-controlled ``.tlspec``:

E1 (CRITICAL, default-victim RCE) -- the ``SafeBundleUnpickler`` torch branch admitted
    any resolved ``torch.*`` ``type`` after a name/module denylist, but never asserted
    the resolved type is genuinely torch-owned. A DOTTED pickled name
    (``module="torch"``, ``name="utils.collect_env.subprocess.Popen"``) makes pickle's
    ``STACK_GLOBAL`` attribute walk (active at pickle protocol >= 4) ESCAPE torch into
    stdlib ``subprocess`` and return ``subprocess.Popen``; a following ``REDUCE`` spawns
    a process. The fix positively requires the resolved type's real ``__module__`` to be
    within ``torch`` (and the same for the preview-backend branch).

E2 (HIGH, trust-gated but violating the "denied even under trust" contract) -- both the
    intervention ``resolve_import_ref`` foreign branch and the unpickler foreign-global
    branch keyed the denylist on the pickled import-PATH STRING, then attribute-walked a
    dotted qualname with no real-module recheck: ``resolve_import_ref("torch:os.system",
    trust=True)`` returned live ``os.system``. The fix re-enforces the denylist / trust
    gate on the RESOLVED object's real module (plus the ``is_pure_forward_callable`` gate
    for anything that walked back into the torch namespace).

A-1 (HIGH; CRITICAL on torch <= 2.5.1) -- CVE-2025-32434: the embedded-tensor
    reconstruction wrapper forwards attacker bytes into ``torch.load(BytesIO,
    weights_only=True)``, which is itself a working RCE on torch <= 2.5.1. The fix adds
    a named ``HAS_SAFE_WEIGHTS_ONLY_LOAD`` capability flag (feature-detected, surfaced in
    the compat snapshot / doctor) and makes the wrapper fail closed on an affected runtime.
"""

from __future__ import annotations

import importlib
import io
import pickle
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from torchlens._io._safe_unpickle import SafeBundleUnpickler, _safe_load_from_bytes
from torchlens._io.bundle import _RenameAwareUnpickler
from torchlens.intervention.errors import UntrustedCallableError
from torchlens.intervention.save import _resolve_import_ref

_UNPICKLERS = [SafeBundleUnpickler, _RenameAwareUnpickler]

# ---------------------------------------------------------------------------
# Raw pickle-opcode builders (protocol 4). We hand-assemble the byte stream so
# the pickled (module, name) can be an arbitrary DOTTED name -- a normal Pickler
# would only ever emit an object's own real (__module__, __qualname__). The
# STACK_GLOBAL dotted attribute-walk is only performed at protocol >= 4, so every
# escape vector must go through a real ``.load()`` of a proto-4 stream (a bare
# ``find_class`` call runs at proto 0 = a single getattr, which never walks).
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

    The REDUCE invokes ``callable([*str_args])`` -- e.g. ``subprocess.Popen(cmd_list)``
    -- which would spawn a process if the resolved global were admitted.
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
# E1 -- dotted-name attribute-walk escape in the unpickler torch branch.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize("unpickler_cls", _UNPICKLERS)
def test_torch_dotted_walk_to_nontorch_type_denied_no_spawn(
    unpickler_cls: type[pickle.Unpickler],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dotted torch name resolving to a NON-torch-owned type is denied; nothing spawns.

    A benign re-export (``torch._r27_escape = subprocess``) is injected so the walk
    resolves DETERMINISTICALLY to ``subprocess.Popen`` (real module ``subprocess``) --
    the exact condition a real bundle reaches through an existing torch re-export chain
    (``torch.utils.collect_env.subprocess``). The ownership gate must refuse it.
    """

    monkeypatch.setattr(torch, "_r27_escape", subprocess, raising=False)
    sentinel = tmp_path / "r27_e1_pwned"
    payload = _reduce_global_pickle(
        "torch", "_r27_escape.Popen", ["/bin/sh", "-c", f"touch {sentinel}"]
    )
    with pytest.raises(pickle.UnpicklingError, match="non-torch-owned"):
        unpickler_cls(io.BytesIO(payload)).load()
    # The REDUCE never ran -- find_class refused the escaped type at STACK_GLOBAL.
    assert not sentinel.exists()


@pytest.mark.smoke
@pytest.mark.parametrize("unpickler_cls", _UNPICKLERS)
def test_reported_collect_env_popen_vector_inert(
    unpickler_cls: type[pickle.Unpickler], tmp_path: Path
) -> None:
    """The exact reported payload (``torch`` / ``utils.collect_env.subprocess.Popen``) is inert.

    It must RAISE and spawn NOTHING. Depending on which torch submodules a runtime has
    imported, the escape either resolves (denied by the ownership gate ->
    ``UnpicklingError``) or the walk itself fails (``AttributeError``); both are denials
    with no process spawned.
    """

    sentinel = tmp_path / "r27_e1_reported_pwned"
    payload = _reduce_global_pickle(
        "torch",
        "utils.collect_env.subprocess.Popen",
        ["/bin/sh", "-c", f"touch {sentinel}"],
    )
    with pytest.raises((pickle.UnpicklingError, AttributeError)):
        unpickler_cls(io.BytesIO(payload)).load()
    assert not sentinel.exists()


@pytest.mark.smoke
def test_legit_torch_types_still_admit() -> None:
    """Genuine torch data types (real torch module) still resolve (no regression)."""

    unpickler = SafeBundleUnpickler(io.BytesIO(b""))
    assert unpickler.find_class("torch", "Size") is torch.Size
    assert unpickler.find_class("torch.nn.modules.linear", "Identity") is torch.nn.Identity


@pytest.mark.smoke
def test_preview_backend_dotted_walk_escape_denied() -> None:
    """The preview-backend branch also requires the resolved type to be preview-owned.

    A fake ``mlx.core`` is injected so the branch is reachable without the real optional
    dependency; it re-exports a NON-preview type (``subprocess.Popen``) under a dotted
    name, which the ownership gate must refuse.
    """

    fake_mlx = types.ModuleType("mlx")
    fake_core = types.ModuleType("mlx.core")
    fake_mlx.core = fake_core  # type: ignore[attr-defined]
    fake_core.subprocess = subprocess  # type: ignore[attr-defined]
    sys.modules["mlx"] = fake_mlx
    sys.modules["mlx.core"] = fake_core
    try:
        payload = _stop_global_pickle("mlx.core", "subprocess.Popen")
        with pytest.raises(pickle.UnpicklingError, match="non-preview-owned"):
            SafeBundleUnpickler(io.BytesIO(payload)).load()
    finally:
        sys.modules.pop("mlx.core", None)
        sys.modules.pop("mlx", None)


# ---------------------------------------------------------------------------
# E2(a) -- intervention resolver: denylist keyed on resolved real module.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize(
    "import_path",
    ["torch:os.system", "torch:serialization.load", "torch:serialization.save"],
)
def test_resolver_dotted_walk_to_denied_module_denied_under_trust(import_path: str) -> None:
    """A dotted qualname off ``torch`` that walks into a denied module is refused."""

    # Denied by default...
    with pytest.raises(UntrustedCallableError):
        _resolve_import_ref(import_path)
    # ...and STILL denied under broad trust (trust never authorizes os / serialization)...
    with pytest.raises(UntrustedCallableError):
        _resolve_import_ref(import_path, trust_custom_callables=True)
    # ...and denied even when the CLAIMED root "torch" is explicitly allowlisted: trust
    # keys on the RESOLVED real module (posix / torch.serialization), never the path.
    with pytest.raises(UntrustedCallableError):
        _resolve_import_ref(import_path, allowed_custom_callable_modules={"torch"})


@pytest.mark.smoke
def test_resolver_from_file_via_dotted_torch_denied_by_purity() -> None:
    """A dotted torch qualname reaching ``from_file`` is refused by the purity gate.

    ``torch.from_file``'s real module is the bare, non-denied ``"torch"``, so the
    denylist alone would admit it; the ``is_pure_forward_callable`` parity gate denies it.
    """

    with pytest.raises(UntrustedCallableError):
        _resolve_import_ref("torch:_C._VariableFunctions.from_file", trust_custom_callables=True)


@pytest.mark.smoke
def test_resolver_legit_foreign_trusted_callable_still_resolves(tmp_path: Path) -> None:
    """A genuinely foreign (non-torch) trusted callable is NOT purity-gated (no regression)."""

    mod_name = "r27_resolver_trusted_mod"
    (tmp_path / f"{mod_name}.py").write_text("def hook(out, *, hook):\n    return out * 2\n")
    sys.path.insert(0, str(tmp_path))
    try:
        mod = importlib.import_module(mod_name)
        assert _resolve_import_ref(f"{mod_name}:hook", trust_custom_callables=True) is mod.hook
        assert (
            _resolve_import_ref(f"{mod_name}:hook", allowed_custom_callable_modules={mod_name})
            is mod.hook
        )
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop(mod_name, None)


# ---------------------------------------------------------------------------
# E2(b) -- unpickler foreign-global trusted branch: re-enforce on resolved module.
# ---------------------------------------------------------------------------


@pytest.fixture
def _trusted_os_module(tmp_path: Path) -> str:
    """A foreign module that (like ~75 real ones) does ``import os`` at top level."""

    mod_name = "r27_unpickler_trusted_mod"
    (tmp_path / f"{mod_name}.py").write_text("import os\n\ndef hook(x):\n    return x\n")
    sys.path.insert(0, str(tmp_path))
    importlib.import_module(mod_name)
    try:
        yield mod_name
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop(mod_name, None)


@pytest.mark.smoke
def test_unpickler_trusted_dotted_walk_to_os_denied(_trusted_os_module: str) -> None:
    """Under trust, a dotted name walking off a trusted module to ``os.system`` is denied."""

    payload = _stop_global_pickle(_trusted_os_module, "os.system")
    with pytest.raises(pickle.UnpicklingError, match="posix"):
        SafeBundleUnpickler(io.BytesIO(payload), trust_custom_callables=True).load()
    # A narrow allowlist keyed on the trusted module is likewise not fooled.
    with pytest.raises(pickle.UnpicklingError):
        SafeBundleUnpickler(
            io.BytesIO(payload), allowed_custom_callable_modules={_trusted_os_module}
        ).load()


@pytest.mark.smoke
def test_unpickler_legit_trusted_callable_still_resolves(_trusted_os_module: str) -> None:
    """A genuine (single-name) callable in the trusted module still resolves (no regression)."""

    unpickler = SafeBundleUnpickler(io.BytesIO(b""), trust_custom_callables=True)
    resolved = unpickler.find_class(_trusted_os_module, "hook")
    assert getattr(resolved, "__name__", None) == "hook"


# ---------------------------------------------------------------------------
# A-1 -- CVE-2025-32434 embedded-tensor load gate.
# ---------------------------------------------------------------------------


def _embedded_tensor_bytes() -> bytes:
    """Bytes accepted by ``torch.storage._load_from_bytes`` (a ``torch.save`` payload)."""

    buf = io.BytesIO()
    torch.save(torch.tensor([1.0, 2.0, 3.0]), buf)
    return buf.getvalue()


@pytest.mark.smoke
def test_embedded_load_refused_on_cve_vulnerable_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the CVE-2025-32434 fix is absent, the embedded load fails closed."""

    monkeypatch.setattr(
        "torchlens.utils._torch_compat.HAS_SAFE_WEIGHTS_ONLY_LOAD", False, raising=True
    )
    with pytest.raises(pickle.UnpicklingError, match="CVE-2025-32434"):
        _safe_load_from_bytes(_embedded_tensor_bytes())


@pytest.mark.smoke
def test_embedded_load_roundtrips_on_fixed_torch() -> None:
    """On a CVE-fixed torch (>= 2.6), a benign embedded tensor still round-trips."""

    from torchlens.utils._torch_compat import HAS_SAFE_WEIGHTS_ONLY_LOAD

    if not HAS_SAFE_WEIGHTS_ONLY_LOAD:  # pragma: no cover - depends on runtime torch
        pytest.skip("running torch predates the CVE-2025-32434 fix")
    result = _safe_load_from_bytes(_embedded_tensor_bytes())
    assert torch.equal(result, torch.tensor([1.0, 2.0, 3.0]))


@pytest.mark.smoke
def test_weights_only_load_capability_in_snapshot() -> None:
    """The named capability flag surfaces in the torch capability snapshot / doctor report."""

    import torchlens.utils as tl_utils
    from torchlens.utils._torch_compat import get_torch_capability_snapshot

    snapshot = get_torch_capability_snapshot()
    assert "HAS_SAFE_WEIGHTS_ONLY_LOAD" in snapshot
    assert isinstance(snapshot["HAS_SAFE_WEIGHTS_ONLY_LOAD"], bool)
    # Surfaced in the model-free doctor diagnostic (tl.compat.report needs a model).
    assert "HAS_SAFE_WEIGHTS_ONLY_LOAD" in str(tl_utils.doctor())
