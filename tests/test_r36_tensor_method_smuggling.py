"""Round-36 security regression: module-less tensor-method smuggling under trust.

r35 fixed the ``__module__``-missing / ``or <trusted-module>`` fallback loophole on
the intervention resolver's **torchlens-walk** foreign branch (by routing the
resolved object through ``is_pure_forward_callable`` on its REAL identity). r36
(secE-r36-1, HIGH, trust-gated) closes the SAME loophole on the two SIBLING foreign
branches that r35 left untouched:

* **Sink 1** -- ``intervention/resolver.py`` genuinely-foreign branch: it re-enforced
  the denylist / stdlib / torch-purity rechecks on the ``resolved_owner`` STRING,
  which falls back to the trusted ``module_name`` when the resolved object has no
  ``__module__``.
* **Sink 2** -- ``_io/_safe_unpickle.py`` trusted-foreign tail: identical
  ``resolved_owner = getattr(obj, "__module__", "") or module`` fallback.

C-level tensor methods (``resize_`` / ``set_`` / ``apply_`` / ``map_``) report NO
``__module__``, so the fallback string became the (non-torch) trusted module name and
the torch-purity gate was SKIPPED -- resolving a side-effecting method under trust
(``resize_`` leaks uninitialized heap memory; ``set_`` repoints storage; ``apply_`` /
``map_`` invoke an arbitrary callable per element). ``torch.from_file`` (real
``__module__ == "torch"``) was already denied by the string gate -- it is specifically
the ``__module__``-missing tensor-method family that slipped.

Each branch now holds ANY resolved object whose real ``__module__`` is missing/None to
the pure-forward contract on the REAL object identity, so the fallback can never skip
the gate. Legit trusted user-recipe functions (real ``__module__``), pure torch ops,
module-less pure in-place ELEMENTWISE tensor methods (``add_`` / ``mul_`` / ...), and
the ``operator`` carve-out all still resolve.

These tests exercise BOTH trust-gated paths (``resolve_import_ref`` and
``SafeBundleUnpickler``) in lockstep, mirroring test_r31_stdlib_deny.py.
"""

from __future__ import annotations

import importlib
import io
import operator
import pickle
import sys
from pathlib import Path
from typing import Iterator

import pytest
import torch

from torchlens._io._safe_unpickle import SafeBundleUnpickler
from torchlens.intervention.errors import UntrustedCallableError
from torchlens.intervention.resolver import resolve_import_ref
from torchlens.utils._callable_safety import is_pure_forward_callable

# The ``__module__``-missing C tensor-method family the fallback loophole smuggled.
# Every one is exactly what ``is_pure_forward_callable`` exists to deny.
_STORAGE_UNSAFE_METHODS = [
    "torch.Tensor.resize_",  # exposes uninitialized heap memory (info leak)
    "torch.Tensor.set_",  # repoints tensor storage at attacker-chosen bytes
    "torch.Tensor.apply_",  # invokes an attacker callable per element
    "torch.Tensor.map_",  # invokes an attacker callable per element
]


# --------------------------------------------------------------------------- #
# Raw pickle-opcode builders (protocol 4). The STACK_GLOBAL dotted attribute-walk
# only runs at proto >= 4, so the smuggling qualname must go through a real
# ``.load()`` of a proto-4 stream. Mirrors test_r28_rebuild_dotted_walk_rce.py.
# --------------------------------------------------------------------------- #


def _short_binunicode(text: str) -> bytes:
    """Encode ``text`` as a SHORT_BINUNICODE opcode (length < 256)."""

    raw = text.encode("utf-8")
    assert len(raw) < 256
    return b"\x8c" + bytes([len(raw)]) + raw


def _stop_global_pickle(module: str, name: str) -> bytes:
    """Proto-4 pickle that STACK_GLOBAL-resolves ``module.name`` then STOPs."""

    return b"\x80\x04" + _short_binunicode(module) + _short_binunicode(name) + b"\x93" + b"."


@pytest.fixture
def trusted_torch_module(tmp_path: Path) -> Iterator[str]:
    """A foreign module that (like countless real ones) does ``import torch``.

    ``import torch`` makes ``torch`` reachable as an attribute of the module, so a
    dotted qualname such as ``torch.Tensor.resize_`` attribute-walks off it -- the
    exact smuggling surface. The module also exports a benign recipe function so we
    can assert legit resolution under the same trust is preserved.
    """

    mod_name = "r36_trusted_torch_mod"
    (tmp_path / f"{mod_name}.py").write_text("import torch\n\n\ndef recipe(t):\n    return t + 1\n")
    sys.path.insert(0, str(tmp_path))
    importlib.import_module(mod_name)
    try:
        yield mod_name
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop(mod_name, None)


# --------------------------------------------------------------------------- #
# Sink 1 -- intervention RESOLVER genuinely-foreign branch.
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
@pytest.mark.parametrize("method", _STORAGE_UNSAFE_METHODS)
def test_secE_r36_resolver_denies_module_less_tensor_method_under_trust(
    trusted_torch_module: str, method: str
) -> None:
    """The RESOLVER denies a module-less tensor method walked off a trusted module.

    Denied under BROAD trust and under an EXPLICIT module allowlist naming the very
    trusted module -- the ``or module_name`` fallback can no longer skip the gate.
    """

    ref = f"{trusted_torch_module}:{method}"
    allow = {trusted_torch_module}
    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(ref, allowed_custom_callable_modules=allow)
    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(ref, trust_custom_callables=True)


@pytest.mark.smoke
def test_secE_r36_resolver_control_from_file_still_denied(trusted_torch_module: str) -> None:
    """CONTROL: ``torch.from_file`` (real ``__module__ == 'torch'``) stays denied.

    It was already covered by the torch-prefixed string gate; the new module-less
    gate must not be the only thing standing between it and resolution.
    """

    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(
            f"{trusted_torch_module}:torch.from_file",
            allowed_custom_callable_modules={trusted_torch_module},
        )


@pytest.mark.smoke
def test_secE_r36_resolver_preserves_legit_resolution_under_trust(
    trusted_torch_module: str,
) -> None:
    """The gate must NOT over-deny legitimate trusted resolutions."""

    allow = {trusted_torch_module}
    # A genuine trusted user-recipe function (real ``__module__`` == the module).
    recipe = resolve_import_ref(
        f"{trusted_torch_module}:recipe", allowed_custom_callable_modules=allow
    )
    assert recipe(torch.zeros(1)).item() == 1.0
    # A PURE torch op walked off the trusted module (real ``__module__`` == "torch").
    relu = resolve_import_ref(
        f"{trusted_torch_module}:torch.relu", allowed_custom_callable_modules=allow
    )
    assert relu is torch.relu
    # A MODULE-LESS pure in-place ELEMENTWISE tensor method still resolves -- the new
    # gate denies only the storage-unsafe / callable-invoking family, not ``add_``.
    add_ = resolve_import_ref(
        f"{trusted_torch_module}:torch.Tensor.add_", allowed_custom_callable_modules=allow
    )
    assert add_ is torch.Tensor.add_
    # The ``operator`` carve-out is unaffected.
    neg = resolve_import_ref("operator:neg", trust_custom_callables=True)
    assert neg is operator.neg


# --------------------------------------------------------------------------- #
# Sink 2 -- metadata UNPICKLER trusted-foreign tail.
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
@pytest.mark.parametrize("method", _STORAGE_UNSAFE_METHODS)
def test_secE_r36_unpickler_denies_module_less_tensor_method_under_trust(
    trusted_torch_module: str, method: str
) -> None:
    """The UNPICKLER denies a module-less tensor method walked off a trusted module."""

    payload = _stop_global_pickle(trusted_torch_module, method)
    with pytest.raises(pickle.UnpicklingError):
        SafeBundleUnpickler(
            io.BytesIO(payload), allowed_custom_callable_modules={trusted_torch_module}
        ).load()


@pytest.mark.smoke
def test_secE_r36_unpickler_control_from_file_still_denied(trusted_torch_module: str) -> None:
    """CONTROL: ``torch.from_file`` stays denied on the unpickler tail as well."""

    payload = _stop_global_pickle(trusted_torch_module, "torch.from_file")
    with pytest.raises(pickle.UnpicklingError):
        SafeBundleUnpickler(
            io.BytesIO(payload), allowed_custom_callable_modules={trusted_torch_module}
        ).load()


@pytest.mark.smoke
def test_secE_r36_unpickler_default_victim_never_resolves_live_method(
    trusted_torch_module: str,
) -> None:
    """Default (untrusting) victim never resolves the LIVE method (inert deferral)."""

    payload = _stop_global_pickle(trusted_torch_module, "torch.Tensor.resize_")
    resolved = SafeBundleUnpickler(io.BytesIO(payload)).load()
    assert resolved is not torch.Tensor.resize_


@pytest.mark.smoke
def test_secE_r36_unpickler_preserves_pure_torch_under_trust(trusted_torch_module: str) -> None:
    """A PURE torch op walked off a trusted module still resolves through the unpickler."""

    payload = _stop_global_pickle(trusted_torch_module, "torch.relu")
    resolved = SafeBundleUnpickler(
        io.BytesIO(payload), allowed_custom_callable_modules={trusted_torch_module}
    ).load()
    assert resolved is torch.relu


# --------------------------------------------------------------------------- #
# The purity gate itself refuses the smuggled surface but admits the legit one.
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_secE_r36_purity_gate_denies_storage_unsafe_but_admits_elementwise() -> None:
    """``is_pure_forward_callable`` denies the module-less storage-unsafe family only."""

    assert not is_pure_forward_callable(torch.Tensor.resize_)
    assert not is_pure_forward_callable(torch.Tensor.set_)
    assert not is_pure_forward_callable(torch.Tensor.apply_)
    assert not is_pure_forward_callable(torch.Tensor.map_)
    # Legit module-less pure in-place ELEMENTWISE ops stay admitted.
    assert is_pure_forward_callable(torch.Tensor.add_)
    assert is_pure_forward_callable(torch.Tensor.mul_)


# --------------------------------------------------------------------------- #
# WRAPPED-torch state: once TorchLens capture-wraps torch (the near-universal
# live state), tensor methods carry ``__module__ == 'torchlens.backends.torch.
# wrappers'`` -- a truthy, non-torch string that spoofs BOTH a __module__-missing
# check AND a raw torch-prefix check. The gate must key on the REAL (unwrapped)
# identity, so denial must hold whether or not torch is wrapped.
# --------------------------------------------------------------------------- #


@pytest.fixture
def torch_capture_wrapped() -> Iterator[None]:
    """Ensure torch is capture-wrapped so tensor methods carry the wrappers module."""

    import torchlens as tl

    tl.trace(torch.nn.Linear(4, 4), torch.randn(1, 4))
    # Sanity: the wrapped surface is exactly what spoofs a raw ``__module__`` check.
    assert getattr(torch.Tensor.resize_, "__module__", None) == "torchlens.backends.torch.wrappers"
    yield


@pytest.mark.smoke
@pytest.mark.parametrize("method", _STORAGE_UNSAFE_METHODS)
def test_secE_r36_resolver_denies_wrapped_tensor_method(
    torch_capture_wrapped: None, trusted_torch_module: str, method: str
) -> None:
    """The RESOLVER denies a WRAPPED tensor method (real-identity gate, not the string)."""

    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(
            f"{trusted_torch_module}:{method}",
            allowed_custom_callable_modules={trusted_torch_module},
        )


@pytest.mark.smoke
@pytest.mark.parametrize("method", _STORAGE_UNSAFE_METHODS)
def test_secE_r36_unpickler_denies_wrapped_tensor_method(
    torch_capture_wrapped: None, trusted_torch_module: str, method: str
) -> None:
    """The UNPICKLER denies a WRAPPED tensor method (real-identity gate, not the string)."""

    payload = _stop_global_pickle(trusted_torch_module, method)
    with pytest.raises(pickle.UnpicklingError):
        SafeBundleUnpickler(
            io.BytesIO(payload), allowed_custom_callable_modules={trusted_torch_module}
        ).load()


@pytest.mark.smoke
def test_secE_r36_wrapped_pure_ops_still_resolve(
    torch_capture_wrapped: None, trusted_torch_module: str
) -> None:
    """Legit pure ops still resolve even when torch is wrapped (no over-denial)."""

    allow = {trusted_torch_module}
    relu = resolve_import_ref(
        f"{trusted_torch_module}:torch.relu", allowed_custom_callable_modules=allow
    )
    assert relu is torch.relu
    add_ = resolve_import_ref(
        f"{trusted_torch_module}:torch.Tensor.add_", allowed_custom_callable_modules=allow
    )
    assert add_ is torch.Tensor.add_
