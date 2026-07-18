"""Round-33 security regression: name-scope the operator/_operator carve-out.

The r31 structural stdlib deny carves the WHOLE ``operator`` / ``_operator`` module
out of the stdlib/builtin denial (``_ALLOWED_STDLIB_ROOTS``) so the legitimate pure
operators (``operator:neg`` et al.) still resolve on the FOREIGN resolution paths.
Before r33, that carve-out applied NO positive name filter on those paths, so under
trust (even a narrow ``allowed_custom_callable_modules={"operator"}``) the generic
operator GADGETS -- ``attrgetter`` / ``methodcaller`` / ``call`` / ``getitem`` /
``setitem`` / ``delitem`` / the in-place ``iadd`` / ``imul`` / ... mutators --
resolved, enabling an RCE chain
(``attrgetter('__globals__')(identity)`` -> ``__builtins__`` -> ``__import__`` ->
``os.system``).

r33 (A-R32-1 / E-r32-1) requires the terminal name in the pure-forward
``_ALLOWED_OPERATOR_NAMES`` allowlist whenever a RESOLVED callable's real module is
``operator`` / ``_operator``, on BOTH foreign paths: the intervention resolver
(custom / foreign tail) and the metadata unpickler (trusted foreign tail). It also
routes ``_operator:<name>`` onto the name-allowlisted fixed operator root so
``_operator:setitem`` cannot fall through to the (previously unfiltered) custom tail.

r33 (A-R32-2) additionally rejects a resolved bare MODULE object in the trusted
unpickler tail: a module has no ``__module__``, so the resolved-real-module recheck
fell back to the (trusted, non-stdlib) pickled module name and admitted it.

The victim is already safe by default (these paths require explicit trust), but the
findings violate the "denied even under trust" contract that gates the operator
carve-out.
"""

from __future__ import annotations

import io
import operator
import pickle
import sys
from types import ModuleType

import pytest

from torchlens._io._safe_unpickle import SafeBundleUnpickler
from torchlens.intervention.errors import UntrustedCallableError
from torchlens.intervention.resolver import resolve_function_registry_key, resolve_import_ref
from torchlens.intervention.types import FunctionRegistryKey
from torchlens.utils._callable_safety import is_denied_operator_gadget

# Generic operator gadgets / mutators that MUST stay denied even under operator trust.
_GADGET_NAMES = [
    "attrgetter",
    "methodcaller",
    "itemgetter",
    "call",
    "setitem",
    "delitem",
    "iadd",
    "imul",
]

# Pure forward operators that MUST still resolve.
_PURE_NAMES = ["neg", "add", "mul", "getitem", "eq", "and_", "invert"]


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


@pytest.fixture()
def trusted_evilmod():
    """Register a synthetic trusted foreign module that re-exports operator / io.

    Simulates a legitimately allowlisted user recipe module that happens to have
    ``import operator`` / ``import io`` at top level, so a dotted qualname can
    attribute-walk off it into the operator gadgets or a bare module object.
    """

    mod = ModuleType("evilmod_r33")
    mod.operator = operator  # type: ignore[attr-defined]
    mod.io = io  # type: ignore[attr-defined]
    sys.modules["evilmod_r33"] = mod
    try:
        yield "evilmod_r33"
    finally:
        sys.modules.pop("evilmod_r33", None)


# --------------------------------------------------------------------------- #
# Detector unit checks.
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
@pytest.mark.parametrize("name", _GADGET_NAMES)
def test_detector_flags_operator_gadget(name: str) -> None:
    """The operator-gadget detector flags generic gadgets / mutators (DENY)."""

    gadget = getattr(operator, name)
    assert is_denied_operator_gadget(gadget) is True


@pytest.mark.smoke
@pytest.mark.parametrize("name", _PURE_NAMES)
def test_detector_allows_pure_operators(name: str) -> None:
    """The operator-gadget detector never flags a pure forward operator (ALLOW)."""

    assert is_denied_operator_gadget(getattr(operator, name)) is False


@pytest.mark.smoke
def test_detector_ignores_non_operator_callables() -> None:
    """Non-operator callables are not this gate's concern (returns False)."""

    import torch

    assert is_denied_operator_gadget(torch.relu) is False
    assert is_denied_operator_gadget(len) is False


# --------------------------------------------------------------------------- #
# Resolver: operator gadgets denied on the foreign path (dotted walk), even trust.
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
@pytest.mark.parametrize("name", _GADGET_NAMES)
def test_resolver_denies_operator_gadget_via_dotted_walk(trusted_evilmod, name: str) -> None:
    """A dotted qualname walking off a trusted module into an operator gadget is denied."""

    key = FunctionRegistryKey(
        namespace="custom",
        qualname=f"operator.{name}",
        dispatch_kind="function",
        import_path=f"{trusted_evilmod}:operator.{name}",
    )
    with pytest.raises(UntrustedCallableError):
        resolve_function_registry_key(key, trust_custom_callables=True)
    with pytest.raises(UntrustedCallableError):
        resolve_function_registry_key(key, allowed_custom_callable_modules={trusted_evilmod})


@pytest.mark.smoke
def test_resolver_allows_pure_operator_via_dotted_walk(trusted_evilmod) -> None:
    """A pure operator reached by a dotted walk off a trusted module still resolves."""

    key = FunctionRegistryKey(
        namespace="custom",
        qualname="operator.neg",
        dispatch_kind="function",
        import_path=f"{trusted_evilmod}:operator.neg",
    )
    assert resolve_function_registry_key(key, trust_custom_callables=True) is operator.neg


@pytest.mark.smoke
@pytest.mark.parametrize("name", _GADGET_NAMES)
def test_resolver_denies_operator_gadget_direct_ref(name: str) -> None:
    """``operator:<gadget>`` (routed to the fixed root) is denied under any trust."""

    path = f"operator:{name}"
    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(path)
    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(path, trust_custom_callables=True)
    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(path, allowed_custom_callable_modules={"operator"})


@pytest.mark.smoke
@pytest.mark.parametrize("name", _GADGET_NAMES)
def test_resolver_denies_underscore_operator_gadget(name: str) -> None:
    """``_operator:<gadget>`` now routes to the fixed root and is denied under trust."""

    path = f"_operator:{name}"
    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(path, trust_custom_callables=True)
    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(path, allowed_custom_callable_modules={"_operator"})


@pytest.mark.smoke
@pytest.mark.parametrize("name", _PURE_NAMES)
def test_resolver_allows_underscore_operator_pure(name: str) -> None:
    """``_operator:<pure>`` routes to the fixed operator root and resolves."""

    assert resolve_import_ref(f"_operator:{name}") is getattr(operator, name)


# --------------------------------------------------------------------------- #
# Unpickler: operator gadgets denied on the trusted foreign tail.
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
@pytest.mark.parametrize("name", _GADGET_NAMES)
@pytest.mark.parametrize("builder", [_global_ref_pickle, _stack_global_ref_pickle])
def test_unpickler_denies_operator_gadget(name: str, builder) -> None:
    """The bundle unpickler denies operator gadgets even under operator trust."""

    for mod in ("operator", "_operator"):
        blob = builder(mod, name)
        with pytest.raises(pickle.UnpicklingError):
            SafeBundleUnpickler(io.BytesIO(blob), trust_custom_callables=True).load()
        with pytest.raises(pickle.UnpicklingError):
            SafeBundleUnpickler(io.BytesIO(blob), allowed_custom_callable_modules={mod}).load()


@pytest.mark.smoke
@pytest.mark.parametrize("name", _PURE_NAMES)
def test_unpickler_allows_pure_operator_under_trust(name: str) -> None:
    """Pure operators still resolve through the unpickler under trust (carve-out)."""

    blob = _global_ref_pickle("operator", name)
    resolved = SafeBundleUnpickler(io.BytesIO(blob), trust_custom_callables=True).load()
    assert resolved is getattr(operator, name)


@pytest.mark.smoke
def test_unpickler_attrgetter_globals_rce_chain_blocked() -> None:
    """A REDUCE chain driving ``attrgetter('__globals__')`` must RAISE, not resolve.

    The classic operator-gadget RCE bootstraps ``attrgetter('__globals__')`` to reach
    ``__builtins__`` -> ``__import__`` -> ``os.system``. Blocking ``attrgetter`` at
    resolution kills the chain: no sentinel / callable is ever produced.
    """

    class _Evil:
        def __reduce__(self):
            return (operator.attrgetter, ("__globals__",))

    blob = pickle.dumps(_Evil(), protocol=4)
    with pytest.raises(pickle.UnpicklingError):
        SafeBundleUnpickler(io.BytesIO(blob), trust_custom_callables=True).load()
    with pytest.raises(pickle.UnpicklingError):
        SafeBundleUnpickler(io.BytesIO(blob), allowed_custom_callable_modules={"operator"}).load()


# --------------------------------------------------------------------------- #
# A-R32-2: a resolved bare MODULE object is denied in the trusted foreign tail.
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_unpickler_denies_bare_module_object(trusted_evilmod) -> None:
    """A dotted name resolving a bare module object is refused (mirrors dict refusal)."""

    blob = _stack_global_ref_pickle(trusted_evilmod, "io")
    with pytest.raises(pickle.UnpicklingError):
        SafeBundleUnpickler(io.BytesIO(blob), trust_custom_callables=True).load()
    with pytest.raises(pickle.UnpicklingError):
        SafeBundleUnpickler(
            io.BytesIO(blob), allowed_custom_callable_modules={trusted_evilmod}
        ).load()
