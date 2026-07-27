"""Round-31 security regression: structural stdlib/builtin deny under trust.

The "denied even under trust_custom_callables" contract for attacker-controllable
``.tlspec`` bundles was historically enforced by a hand-maintained DENYLIST of
dangerous stdlib/builtin modules (``_DENIED_MODULES`` /
``_DENIED_FOREIGN_MODULES``). That denylist is inherently INCOMPLETE: successive
audits kept finding more resolvable gadgets -- r26 (exec/spawn/install), r28
(``_frozen_importlib``), r30 (``_imp`` / ``zipimport`` / ``io`` / ``_io`` /
``linecache`` / ``fileinput`` / ``py_compile`` / ``compileall`` / ``gc`` / ``mmap``
/ ``fcntl``). Chasing modules one at a time never terminates.

r31 closes the CLASS with a POSITIVE structural rule: on the TRUSTED foreign
resolution path, DENY any resolved callable whose real top-level module is a Python
STANDARD-LIBRARY or BUILTIN module (``sys.stdlib_module_names`` UNION
``sys.builtin_module_names``), keyed on the RESOLVED object's real ``__module__``.
The legitimate pure-forward ``operator`` root, the torch namespaces, first-party
``torchlens.*`` (incl. appliance packages under explicit trust), and NON-stdlib user
packages are all preserved. A user CANNOT re-allow a stdlib module via
``allowed_custom_callable_modules`` -- that is the whole point of the contract.

These tests exercise BOTH trust-gated resolution paths in lockstep: the intervention
resolver (``resolve_import_ref``) and the metadata unpickler
(``SafeBundleUnpickler`` GLOBAL and STACK_GLOBAL).
"""

from __future__ import annotations

import io
import operator
import pickle

import pytest
import torch

from torchlens._io._safe_unpickle import SafeBundleUnpickler
from torchlens.intervention.errors import UntrustedCallableError
from torchlens.intervention.resolver import resolve_function_registry_key, resolve_import_ref
from torchlens.intervention.types import FunctionRegistryKey
from torchlens.utils._callable_safety import is_denied_stdlib_or_builtin_module

# The stdlib/builtin refs the r30 (and prior) audits proved resolvable under trust.
# Every one must be DENIED on the default path, under broad trust, AND under an
# explicit module allowlist that names the very module -- stdlib stays denied.
_STDLIB_REFS = [
    # r30 finds.
    "_imp:create_dynamic",
    "zipimport:zipimporter",
    "io:open",
    "_io:open",
    "linecache:getline",
    "fileinput:input",
    "py_compile:compile",
    "compileall:compile_dir",
    "gc:get_referrers",
    "mmap:mmap",
    "fcntl:fcntl",
    # Prior denylist entries, pinned so they can never regress out.
    "os:system",
    "pdb:run",
    "_frozen_importlib:__import__",
]


def _global_ref_pickle(module: str, name: str) -> bytes:
    """Build a protocol-2 GLOBAL bare-reference pickle to ``module.name``."""

    return b"c" + module.encode() + b"\n" + name.encode() + b"\n."


def _stack_global_ref_pickle(module: str, name: str) -> bytes:
    """Build a protocol-4 STACK_GLOBAL bare-reference pickle to ``module.name``."""

    def _short_binunicode(text: str) -> bytes:
        encoded = text.encode()
        assert len(encoded) < 256
        return b"\x8c" + bytes([len(encoded)]) + encoded

    return b"\x80\x04" + _short_binunicode(module) + _short_binunicode(name) + b"\x93."


# --------------------------------------------------------------------------- #
# Resolver: stdlib/builtin modules denied EVEN under trust / allowlist.
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
@pytest.mark.parametrize("import_path", _STDLIB_REFS)
def test_resolver_denies_stdlib_even_under_trust(import_path: str) -> None:
    """A stdlib/builtin ref never resolves, default / trust / allowlist alike."""

    module_name = import_path.split(":", 1)[0]
    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(import_path)
    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(import_path, trust_custom_callables=True)
    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(import_path, allowed_custom_callable_modules={module_name})


@pytest.mark.smoke
def test_resolver_allowlist_cannot_reenable_stdlib_module() -> None:
    """Naming a stdlib module in the allowlist does not re-enable it (the contract)."""

    with pytest.raises(UntrustedCallableError):
        resolve_import_ref("io:open", allowed_custom_callable_modules={"io"})
    with pytest.raises(UntrustedCallableError):
        resolve_import_ref("gc:get_referrers", allowed_custom_callable_modules={"gc"})


@pytest.mark.smoke
def test_resolver_stdlib_reached_by_dotted_walk_denied_under_trust() -> None:
    """A dotted qualname walking off a trusted torch root into stdlib is denied."""

    # ``torch:os.system`` walks torch.os.system -> os.system (real module ``posix``);
    # this is denied by the resolved-owner recheck (posix is builtin + denylisted).
    key = FunctionRegistryKey(
        namespace="custom",
        qualname="os.system",
        dispatch_kind="function",
        import_path="torch:os.system",
    )
    with pytest.raises(UntrustedCallableError):
        resolve_function_registry_key(key, trust_custom_callables=True)


# --------------------------------------------------------------------------- #
# Unpickler: mirrored stdlib/builtin deny (GLOBAL + STACK_GLOBAL) under trust.
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
@pytest.mark.parametrize("import_path", _STDLIB_REFS)
@pytest.mark.parametrize("builder", [_global_ref_pickle, _stack_global_ref_pickle])
def test_unpickler_denies_stdlib_even_under_trust(import_path: str, builder) -> None:
    """The bundle unpickler hard-denies stdlib/builtin globals, GLOBAL + STACK_GLOBAL."""

    module_name, name = import_path.split(":", 1)
    blob = builder(module_name, name)
    with pytest.raises(pickle.UnpicklingError):
        SafeBundleUnpickler(io.BytesIO(blob)).load()
    with pytest.raises(pickle.UnpicklingError):
        SafeBundleUnpickler(io.BytesIO(blob), trust_custom_callables=True).load()
    with pytest.raises(pickle.UnpicklingError):
        SafeBundleUnpickler(io.BytesIO(blob), allowed_custom_callable_modules={module_name}).load()


# --------------------------------------------------------------------------- #
# Preserved legitimate paths: operator / torch / first-party STILL resolve.
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_operator_root_still_resolves() -> None:
    """``operator:neg`` (real module ``_operator``) is carved out and resolves."""

    assert resolve_import_ref("operator:neg") is operator.neg
    # Also via an explicit custom key routed through the foreign trust path.
    key = FunctionRegistryKey(
        namespace="custom",
        qualname="neg",
        dispatch_kind="function",
        import_path="operator:neg",
    )
    assert resolve_function_registry_key(key, trust_custom_callables=True) is operator.neg
    assert (
        resolve_function_registry_key(key, allowed_custom_callable_modules={"operator"})
        is operator.neg
    )


@pytest.mark.smoke
@pytest.mark.parametrize(
    "import_path",
    ["torch:relu", "torch.nn.functional:relu"],
)
def test_torch_ops_still_resolve(import_path: str) -> None:
    """Pure torch ops (non-stdlib namespace) resolve without trust."""

    resolved = resolve_import_ref(import_path)
    assert callable(resolved)


@pytest.mark.smoke
def test_first_party_inert_callable_still_resolves() -> None:
    """A vetted-inert first-party callable (``identity``) resolves without trust."""

    resolved = resolve_import_ref("torchlens.utils.display:identity")
    assert callable(resolved)
    assert resolved(7) == 7


@pytest.mark.smoke
def test_operator_neg_via_unpickler_under_trust_resolves() -> None:
    """Unpickling ``operator.neg`` under trust yields the real callable (carve-out)."""

    blob = pickle.dumps(operator.neg)
    resolved = SafeBundleUnpickler(io.BytesIO(blob), trust_custom_callables=True).load()
    assert resolved is operator.neg


# --------------------------------------------------------------------------- #
# Detector unit checks.
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
@pytest.mark.parametrize(
    "module",
    ["io", "_io", "os", "os.path", "gc", "mmap", "importlib.util", "zipimport", "_imp"],
)
def test_detector_flags_stdlib(module: str) -> None:
    """The structural detector flags stdlib/builtin modules (incl. submodules)."""

    assert is_denied_stdlib_or_builtin_module(module) is True


@pytest.mark.smoke
@pytest.mark.parametrize(
    "module",
    ["operator", "_operator", "torch", "torch.nn.functional", "torchlens.neuro", "numpy", "mymod"],
)
def test_detector_allows_carveouts_and_non_stdlib(module: str) -> None:
    """The detector allows operator carve-out, torch, first-party, and user packages."""

    assert is_denied_stdlib_or_builtin_module(module) is False


def test_torch_module_is_not_stdlib() -> None:
    """Sanity: torch's real top-level package is not in the stdlib detector set."""

    assert is_denied_stdlib_or_builtin_module(str(torch.relu.__module__)) is False
