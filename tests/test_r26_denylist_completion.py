"""Round-26 security regression: exec/spawn/install denylist completion.

A comprehensive audit found two defense-in-depth gaps in the default-deny
callable/import resolution for attacker-controllable ``.tlspec`` bundles:

* E1 (MEDIUM): the hard module denylists (``_DENIED_MODULES`` in
  ``torchlens.utils._callable_safety`` and its mirrored
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
"""

from __future__ import annotations

import io
import pickle

import pytest
import torch

from torchlens._io._safe_unpickle import SafeBundleUnpickler
from torchlens.intervention.errors import UntrustedCallableError
from torchlens.intervention.resolver import resolve_import_ref
from torchlens.utils._callable_safety import is_pure_forward_callable

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


@pytest.mark.smoke
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


@pytest.mark.smoke
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


@pytest.mark.smoke
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


@pytest.mark.smoke
def test_code_exec_torch_callables_fail_purity_gate() -> None:
    """The purity gate itself refuses the three r26 code-exec vectors."""

    assert not is_pure_forward_callable(torch.compile)
    assert not is_pure_forward_callable(torch.Tensor.apply_)
    assert not is_pure_forward_callable(torch.Tensor.map_)


@pytest.mark.smoke
def test_pure_forward_surface_not_over_denied() -> None:
    """The r26 additions do not over-deny the legitimate pure forward surface."""

    # Ordinary pure / in-place elementwise ops stay resolvable.
    assert resolve_import_ref("torch:relu") is torch.relu
    assert is_pure_forward_callable(torch.Tensor.add_)
    assert is_pure_forward_callable(torch.Tensor.mul_)
    # ``apply_``/``map_``/``compile`` denials are EXACT: sibling pure names with
    # similar shapes (e.g. masked_fill_'s trailing underscore) are unaffected.
    assert is_pure_forward_callable(torch.Tensor.masked_fill_)
