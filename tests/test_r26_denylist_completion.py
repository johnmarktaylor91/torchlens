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

import pytest
import torch
from torch import nn

import torchlens._utils._callable_safety as _callable_safety
import torchlens.intervention._resolver_registry as _resolver
from torchlens._io import _safe_unpickle
from torchlens._io._safe_unpickle import SafeBundleUnpickler
from torchlens._utils._callable_safety import is_pure_forward_callable
from torchlens.intervention._resolver_registry import resolve_import_ref
from torchlens.intervention.errors import UntrustedCallableError

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


# --------------------------------------------------------------------------- #
# E2 -- purity gate: torch.compile / Tensor.apply_ / Tensor.map_ denied.
# --------------------------------------------------------------------------- #


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


# The elementwise runners, reachable through the unpickler's restricted getattr.
_CALLABLE_RUNNER_NAMES = ("apply_", "map_", "map2_")


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
