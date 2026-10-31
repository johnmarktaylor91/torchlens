"""Round-26 security regression: exec/spawn/install denylist completion.

A comprehensive audit found two defense-in-depth gaps in the default-deny
callable/import resolution for attacker-controllable ``.tlspec`` bundles:

* E1 (MEDIUM): the hard module denylists (``_DENIED_MODULES`` in
  ``torchlens._utils._callable_safety`` and its mirrored
  ``_DENIED_FOREIGN_MODULES`` in ``torchlens._io._safe_unpickle``) omitted the
  stdlib exec/spawn/install class, so under ``trust_custom_callables=True`` the
  resolver handed back live callables for e.g. ``pdb:run`` / ``timeit:timeit``
  / ``webbrowser:open`` / ``pip:main`` -- violating the contract that dangerous
  modules are denied EVEN under trust. Both lists now deny: pdb, bdb, timeit,
  trace, cProfile, profile, pydoc, webbrowser, antigravity, platform, asyncio,
  pip, setuptools, venv (runpy and code were already denied).
* E2 (LOW): ``_DENIED_CALLABLE_NAMES`` now also denies the code-execution
  vectors reachable in the allowlisted torch surface: ``torch.compile`` and the
  elementwise Python-callable runners ``Tensor.apply_`` / ``Tensor.map_``.

These tests mirror the r23 LAYER-3 patterns: denial must hold on the
trust-satisfied path AND under an explicit module allowlist.

Every name here is also refused by a structural rule (the r31 stdlib/builtin
module gate, the r39 invoker-name shape), so the layer-independence tests at the
end switch that rule off and require the explicit entry to refuse on its own,
directly and through ``tl.load`` of a forged bundle.
"""

from __future__ import annotations

import io
import pickle
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens._utils._callable_safety as _callable_safety
import torchlens.intervention.resolver as _resolver
from torchlens._io import TorchLensIOError, _safe_unpickle
from torchlens._io._safe_unpickle import SafeBundleUnpickler
from torchlens._utils._callable_safety import is_pure_forward_callable
from torchlens.intervention.errors import UntrustedCallableError
from torchlens.intervention.resolver import resolve_import_ref

# The exec/spawn/install stdlib refs the audit proved resolvable under trust.
_EXEC_SPAWN_INSTALL_REFS = [
    "pdb:run",
    "timeit:timeit",
    "webbrowser:open",
    "pip:main",
    "pydoc:cli",
    "cProfile:run",
    "profile:run",
    "trace:main",
    "bdb:Bdb",
    "venv:create",
    "setuptools:setup",
    "platform:system",
    "asyncio:run",
    # Already denied pre-r26; pinned here so they can never regress out.
    "runpy:run_module",
    "code:interact",
]


def _global_ref_pickle(module: str, name: str) -> bytes:
    """Build a minimal pickle that is a bare GLOBAL reference to ``module.name``."""

    return b"c" + module.encode() + b"\n" + name.encode() + b"\n."


# --------------------------------------------------------------------------- #
# E1 -- resolver: exec/spawn/install modules denied EVEN under trust.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("import_path", _EXEC_SPAWN_INSTALL_REFS)
def test_exec_spawn_install_modules_denied_even_under_trust(import_path: str) -> None:
    """An exec/spawn/install stdlib module never resolves, even with trust satisfied."""

    # Default: denied.
    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(import_path)
    # Broad trust: STILL denied -- trust never authorizes these modules.
    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(import_path, trust_custom_callables=True)
    # Explicit allowlist naming the module: STILL denied.
    module_name = import_path.split(":", 1)[0]
    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(import_path, allowed_custom_callable_modules={module_name})


# --------------------------------------------------------------------------- #
# E1 -- safe unpickler: mirrored denylist blocks the same class under trust.
# --------------------------------------------------------------------------- #


@pytest.mark.smoke_cells(
    "test_unpickler_denies_exec_spawn_install_globals_even_under_trust[pdb:run]"
)
@pytest.mark.parametrize("import_path", _EXEC_SPAWN_INSTALL_REFS)
def test_unpickler_denies_exec_spawn_install_globals_even_under_trust(
    import_path: str,
) -> None:
    """The bundle unpickler hard-denies exec/spawn/install globals, even with trust."""

    module_name, name = import_path.split(":", 1)
    blob = _global_ref_pickle(module_name, name)
    with pytest.raises(pickle.UnpicklingError):
        SafeBundleUnpickler(io.BytesIO(blob)).load()
    with pytest.raises(pickle.UnpicklingError):
        SafeBundleUnpickler(io.BytesIO(blob), trust_custom_callables=True).load()
    with pytest.raises(pickle.UnpicklingError):
        SafeBundleUnpickler(io.BytesIO(blob), allowed_custom_callable_modules={module_name}).load()


# --------------------------------------------------------------------------- #
# E2 -- purity gate: torch.compile / Tensor.apply_ / Tensor.map_ denied.
# --------------------------------------------------------------------------- #


@pytest.mark.smoke_cells(
    "test_code_exec_torch_callables_denied_even_under_trust[torch.Tensor:apply_]"
)
@pytest.mark.parametrize(
    "import_path",
    ["torch:compile", "torch.Tensor:apply_", "torch.Tensor:map_"],
)
def test_code_exec_torch_callables_denied_even_under_trust(import_path: str) -> None:
    """torch.compile / Tensor.apply_ / Tensor.map_ never resolve, even under trust."""

    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(import_path)
    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(import_path, trust_custom_callables=True)


def test_code_exec_torch_callables_fail_purity_gate() -> None:
    """The purity gate itself refuses the three r26 code-exec vectors."""

    assert not is_pure_forward_callable(torch.compile)
    assert not is_pure_forward_callable(torch.Tensor.apply_)
    assert not is_pure_forward_callable(torch.Tensor.map_)


def test_pure_forward_surface_not_over_denied() -> None:
    """The r26 additions do not over-deny the legitimate pure forward surface."""

    # Ordinary pure / in-place elementwise ops stay resolvable.
    assert resolve_import_ref("torch:relu") is torch.relu
    assert is_pure_forward_callable(torch.Tensor.add_)
    assert is_pure_forward_callable(torch.Tensor.mul_)
    # ``apply_``/``map_``/``compile`` denials are EXACT: sibling pure names with
    # similar shapes (e.g. masked_fill_'s trailing underscore) are unaffected.
    assert is_pure_forward_callable(torch.Tensor.masked_fill_)


# --------------------------------------------------------------------------- #
# Layer independence: the explicit lists must refuse on their own.
#
# Every name above is ALSO refused by a structural rule: the stdlib/builtin
# module gate (r31) for the modules, and the ``(map|apply)\d*_`` invoker-name
# shape (r39) for the elementwise runners. With both layers on, removing a name
# from the explicit list changes nothing observable, so the tests above cannot
# notice it. Defense in depth means each layer holds alone: these tests switch
# the structural rule off and require the explicit entry to refuse by itself.
# --------------------------------------------------------------------------- #

# Every entry of the explicit module list, by class. The list is a declared
# floor: entries may be added, never dropped.
_EXPLICIT_MODULE_FLOOR = frozenset(
    {
        # Serialization.
        "torch.serialization",
        "torch.storage",
        "torch._utils_internal",
        "pickle",
        "_pickle",
        "marshal",
        # Torch entry points that run user code, import packages or spawn.
        "torch.jit",
        "torch.package",
        "torch.hub",
        "torch.multiprocessing",
        "torch.distributed",
        # Code execution, imports, debuggers, profilers.
        "builtins",
        "importlib",
        "_frozen_importlib",
        "_frozen_importlib_external",
        "runpy",
        "code",
        "codeop",
        "ctypes",
        "pdb",
        "bdb",
        "timeit",
        "trace",
        "cProfile",
        "profile",
        # Spawn processes or browsers; install packages.
        "pydoc",
        "webbrowser",
        "antigravity",
        "platform",
        "asyncio",
        "pip",
        "setuptools",
        "venv",
        # Process, OS and filesystem I/O.
        "os",
        "posix",
        "nt",
        "sys",
        "subprocess",
        "shutil",
        "socket",
        "pty",
        "signal",
        "threading",
        "multiprocessing",
        "glob",
        "tempfile",
        "pathlib",
    }
)

# The torch callables on the explicit name list that run a caller-supplied
# Python callable or compile one: ``(owner, name)``.
_CODE_RUNNING_TORCH_CALLABLES = (
    (torch, "compile"),
    (torch, "vmap"),
    (torch.Tensor, "apply_"),
    (torch.Tensor, "map_"),
    (torch.Tensor, "map2_"),
    (torch.Tensor, "register_hook"),
    (torch.Tensor, "register_post_accumulate_grad_hook"),
)

# Every entry of the explicit callable-name list. Like the module list it is a
# declared floor; most entries also match a structural rule, so only this pin
# notices one being dropped.
_EXPLICIT_NAME_FLOOR = frozenset(
    {
        # File I/O, (de)serialization and module import (round 5).
        "from_file",
        "PyTorchFileReader",
        "PyTorchFileWriter",
        "FileCheck",
        "ScriptModuleSerializer",
        "SerializationStorageContext",
        "DeserializationStorageContext",
        "get_file_path",
        "import_ir_module",
        "import_ir_module_from_buffer",
        "save",
        "load",
        # Arbitrary import and dynamic-library load.
        "_import_device_backends",
        "_import_dotted_name",
        "_load_global_deps",
        "_preload_cuda_deps",
        # Run or compile a caller-supplied callable (r26, r39).
        "compile",
        "apply_",
        "map_",
        "map2_",
        "vmap",
        "register_hook",
        "register_post_accumulate_grad_hook",
    }
)

# The elementwise runners, reachable through the unpickler's restricted getattr.
_CALLABLE_RUNNER_NAMES = ("apply_", "map_", "map2_")

# Explicit entries the stdlib gate does NOT cover (not standard library), so
# the explicit entry is the only module-level refusal on the trusted resolver
# path. ``pip`` / ``setuptools`` are third-party; the torch submodules are not
# stdlib. One real callable each; resolution only, nothing is called.
_SOLE_LAYER_REFS = [
    "pip:main",
    "setuptools:setup",
    "torch.storage:_load_from_bytes",
    "torch.distributed:init_process_group",
]


@pytest.fixture
def stdlib_gate_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable the structural stdlib/builtin module gate on every path that uses it."""

    def _never(_module: str) -> bool:
        return False

    monkeypatch.setattr(_callable_safety, "is_denied_stdlib_or_builtin_module", _never)
    monkeypatch.setattr(_resolver, "is_denied_stdlib_or_builtin_module", _never)
    monkeypatch.setattr(_safe_unpickle, "_stdlib_or_builtin_denied", _never)


@pytest.fixture
def structural_name_rules_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable the invoker-name shape rule and the callable-parameter signature rule."""

    monkeypatch.setattr(_callable_safety, "_is_callable_invoker_name", lambda _name: False)
    monkeypatch.setattr(_callable_safety, "_signature_invokes_callable", lambda _func: False)


@pytest.mark.parametrize("module_name", sorted(_EXPLICIT_MODULE_FLOOR))
def test_explicit_module_list_never_drops_an_entry(module_name: str) -> None:
    """Each floor module stays on the shared explicit denylist (unpickler alias too)."""

    assert module_name in _callable_safety._DENIED_MODULES
    assert _safe_unpickle._module_denied(module_name)
    assert _safe_unpickle._module_denied(module_name + ".submodule")


@pytest.mark.parametrize("name", sorted(_EXPLICIT_NAME_FLOOR))
def test_explicit_name_list_never_drops_an_entry(name: str) -> None:
    """Each floor name stays on the explicit callable-name list."""

    assert name in _callable_safety._DENIED_CALLABLE_NAMES


@pytest.mark.parametrize(("owner", "name"), _CODE_RUNNING_TORCH_CALLABLES)
def test_code_running_torch_callables_exist(owner: object, name: str) -> None:
    """The pinned code-running callables still exist, so the pins test something real."""

    assert callable(getattr(owner, name))


@pytest.mark.parametrize("import_path", _SOLE_LAYER_REFS)
def test_non_stdlib_explicit_entries_deny_under_trust(import_path: str) -> None:
    """Explicit entries with no stdlib-gate twin refuse on the trusted resolver path."""

    module_name, name = import_path.split(":", 1)
    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(import_path, trust_custom_callables=True)
    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(import_path, allowed_custom_callable_modules={module_name})
    if module_name in {"pip", "setuptools"}:
        blob = _global_ref_pickle(module_name, name)
        with pytest.raises(pickle.UnpicklingError, match="dangerous global"):
            SafeBundleUnpickler(io.BytesIO(blob), trust_custom_callables=True).load()


@pytest.mark.usefixtures("stdlib_gate_off")
@pytest.mark.parametrize("import_path", _EXEC_SPAWN_INSTALL_REFS)
def test_explicit_module_list_alone_denies_exec_spawn_install(import_path: str) -> None:
    """With the stdlib gate off, the explicit list alone still refuses every ref."""

    module_name, name = import_path.split(":", 1)
    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(import_path, trust_custom_callables=True)
    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(import_path, allowed_custom_callable_modules={module_name})
    blob = _global_ref_pickle(module_name, name)
    with pytest.raises(pickle.UnpicklingError, match="dangerous global"):
        SafeBundleUnpickler(io.BytesIO(blob), trust_custom_callables=True).load()
    with pytest.raises(pickle.UnpicklingError, match="dangerous global"):
        SafeBundleUnpickler(io.BytesIO(blob), allowed_custom_callable_modules={module_name}).load()


@pytest.mark.usefixtures("structural_name_rules_off")
@pytest.mark.parametrize(("owner", "name"), _CODE_RUNNING_TORCH_CALLABLES)
def test_explicit_name_list_alone_denies_code_running_callables(owner: object, name: str) -> None:
    """With the shape and signature rules off, the explicit name list alone refuses each."""

    assert not is_pure_forward_callable(getattr(owner, name))
    owner_path = "torch" if owner is torch else "torch.Tensor"
    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(f"{owner_path}:{name}", trust_custom_callables=True)
    if name in _CALLABLE_RUNNER_NAMES:
        with pytest.raises(pickle.UnpicklingError, match="side-effecting"):
            _safe_unpickle._safe_getattr(torch._C.TensorBase, name)


# --------------------------------------------------------------------------- #
# The same denials through the public load path (``tl.load`` of a forged bundle).
# --------------------------------------------------------------------------- #


class _TinyNet(nn.Module):
    """Smallest model whose portable save writes a ``metadata.pkl``."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(value) + 1


class _TimeitGadget:
    """``timeit.timeit`` payload that would create ``marker`` if it ever ran."""

    def __init__(self, marker: Path) -> None:
        self.marker = str(marker)

    def __reduce__(self):  # type: ignore[no-untyped-def]
        import timeit

        return (timeit.timeit, (f"open({self.marker!r}, 'w').close()",))


class _RunnerPivot:
    """``getattr(TensorBase, name)`` pivot toward an elementwise callable runner."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __reduce__(self):  # type: ignore[no-untyped-def]
        return (getattr, (torch._C.TensorBase, self.name))


def _forged_bundle(tmp_path: Path, payload: bytes) -> Path:
    """Save a real portable bundle, then replace only its ``metadata.pkl``."""

    bundle = tmp_path / "victim"
    tl.trace(_TinyNet(), torch.randn(2, 3)).save(bundle, level="portable")
    (bundle / "metadata.pkl").write_bytes(payload)
    return bundle


def _error_chain_text(exc: BaseException) -> str:
    """Join the messages of an exception and its causes."""

    parts: list[str] = []
    current: BaseException | None = exc
    while current is not None and len(parts) < 8:
        parts.append(str(current))
        current = current.__cause__ or current.__context__
    return " | ".join(parts)


@pytest.mark.parametrize("layers", ["all_layers", "explicit_only"])
def test_tl_load_refuses_pdb_global_under_trust(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layers: str
) -> None:
    """A forged bundle naming ``pdb.run`` is refused by ``tl.load`` even under trust."""

    if layers == "explicit_only":
        monkeypatch.setattr(_safe_unpickle, "_stdlib_or_builtin_denied", lambda _m: False)
    bundle = _forged_bundle(tmp_path, _global_ref_pickle("pdb", "run"))
    for kwargs in (
        {"trust_custom_callables": True},
        {"allowed_custom_callable_modules": {"pdb"}},
    ):
        with pytest.raises((TorchLensIOError, pickle.UnpicklingError)) as raised:
            tl.load(bundle, **kwargs)
        assert "pdb.run" in _error_chain_text(raised.value)


@pytest.mark.parametrize("layers", ["all_layers", "explicit_only"])
def test_tl_load_timeit_payload_executes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layers: str
) -> None:
    """A forged ``timeit`` payload is refused under trust and never runs."""

    if layers == "explicit_only":
        monkeypatch.setattr(_safe_unpickle, "_stdlib_or_builtin_denied", lambda _m: False)
    marker = tmp_path / "RAN"
    bundle = _forged_bundle(tmp_path, pickle.dumps(_TimeitGadget(marker), protocol=4))
    with pytest.raises((TorchLensIOError, pickle.UnpicklingError)) as raised:
        tl.load(bundle, trust_custom_callables=True, allowed_custom_callable_modules={"timeit"})
    assert "timeit" in _error_chain_text(raised.value)
    assert not marker.exists(), "load-time payload executed"


@pytest.mark.parametrize("layers", ["all_layers", "explicit_only"])
@pytest.mark.parametrize("name", _CALLABLE_RUNNER_NAMES)
def test_tl_load_refuses_callable_runner_pivot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layers: str, name: str
) -> None:
    """A forged ``getattr(TensorBase, "map_")`` pivot is refused by ``tl.load``."""

    if layers == "explicit_only":
        monkeypatch.setattr(_callable_safety, "_is_callable_invoker_name", lambda _name: False)
    bundle = _forged_bundle(tmp_path, pickle.dumps(_RunnerPivot(name), protocol=4))
    with pytest.raises((TorchLensIOError, pickle.UnpicklingError)) as raised:
        tl.load(bundle, trust_custom_callables=True)
    assert repr(name) in _error_chain_text(raised.value)
