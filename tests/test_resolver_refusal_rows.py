"""Row table for callable refusals at the intervention resolver and the bundle unpickler.

Every row names the round-named test file it replaces (its ``reason`` starts with that
file's stem) so the history of each attack string stays findable. A row's ``checks``
letters say which doors must refuse it:

* ``R`` the intervention resolver (``resolve_import_ref``), default path and under
  ``trust_custom_callables=True``;
* ``L`` the resolver also refuses when ``allowed_custom_callable_modules`` names the
  ref's own module (stdlib stays denied under an allowlist: the r31 contract);
* ``U`` the bundle unpickler (``SafeBundleUnpickler``) refuses a bare GLOBAL and a bare
  STACK_GLOBAL reference, default, trusted and allowlisted alike;
* ``P`` the pure-forward purity gate refuses the live callable;
* ``E`` the sparse run-path reattach ladder resolves the key to no callable.

The allowed rows pin the legitimate surface so a fault that denies everything is red too.
"""

from __future__ import annotations

import io
import operator
import pickle
from collections.abc import Callable
from typing import Any

import pytest
import torch

from torchlens._io._safe_unpickle import SafeBundleUnpickler
from torchlens._utils._callable_safety import (
    _is_callable_invoker_name,
    _is_side_effecting_callable_name,
    is_denied_stdlib_or_builtin_module,
    is_pure_forward_callable,
)
from torchlens.backends.torch.wrappers import wrap_torch
from torchlens.intervention._resolver_registry import (
    resolve_function_registry_key,
    resolve_import_ref,
)
from torchlens.intervention.errors import UntrustedCallableError
from torchlens.intervention.types import FunctionRegistryKey


@pytest.fixture(autouse=True, scope="module")
def _torch_wrapped() -> None:
    # The purity gate must unwrap captured descriptors before deciding; install the
    # wrappers so the resolved objects mirror the live state (test_r39 did this at import).
    wrap_torch()


# --------------------------------------------------------------------------- #
# Rows: (ref, checks, reason). The reason names the removed file's stem.
# --------------------------------------------------------------------------- #

_R31 = "test_r31_stdlib_deny: stdlib/builtin module denied even under trust"
_R39 = "test_r39_callable_invoke_denylist"

REFUSED_REFS: tuple[tuple[str, str, str], ...] = (
    # r31: the stdlib/builtin refs the r30 and earlier audits proved resolvable under trust.
    ("_imp:create_dynamic", "RLU", _R31 + " (r30 find)"),
    ("zipimport:zipimporter", "RLU", _R31 + " (r30 find)"),
    ("io:open", "RLU", _R31 + " (r30 find)"),
    ("_io:open", "RLU", _R31 + " (r30 find)"),
    ("linecache:getline", "RLU", _R31 + " (r30 find)"),
    ("fileinput:input", "RLU", _R31 + " (r30 find)"),
    ("py_compile:compile", "RLU", _R31 + " (r30 find)"),
    ("compileall:compile_dir", "RLU", _R31 + " (r30 find)"),
    ("gc:get_referrers", "RLU", _R31 + " (r30 find)"),
    ("mmap:mmap", "RLU", _R31 + " (r30 find)"),
    ("fcntl:fcntl", "RLU", _R31 + " (r30 find)"),
    ("os:system", "RLU", _R31 + " (prior denylist entry)"),
    ("pdb:run", "RLU", _R31 + " (prior denylist entry)"),
    ("_frozen_importlib:__import__", "RLU", _R31 + " (r28 find)"),
    # r39: callable-invoke, storage-realloc and global-mutator misses of the r26 enumeration.
    ("torch.Tensor:map2_", "RPE", _R39 + ": 3-tensor elementwise Python-callable runner"),
    ("torch:vmap", "RPE", _R39 + ": transform that invokes an attacker fn"),
    ("torch.Tensor:register_hook", "RPE", _R39 + ": registers a callback fired on backward"),
    ("torch.Tensor:register_post_accumulate_grad_hook", "RP", _R39 + ": backward callback"),
    ("torch:_resize_output_", "RPE", _R39 + ": storage reallocator (r6 resize_ class)"),
    ("torch:_copy_from_and_resize", "RP", _R39 + ": storage reallocator (r6 resize_ class)"),
    ("torch.Tensor:resize", "RP", _R39 + ": storage reallocator (r6 resize_ class)"),
    ("torch.Tensor:resize_as", "RP", _R39 + ": storage reallocator (r6 resize_ class)"),
    ("torch:_register_device_module", "RPE", _R39 + ": global-state mutator (r6 set_ class)"),
)

ALLOWED_REFS: tuple[tuple[str, str, str], ...] = (
    ("operator:neg", "RLU", "test_r31_stdlib_deny: the operator carve-out still resolves"),
    ("torch:relu", "R", "test_r31_stdlib_deny, test_r39: pure torch ops resolve without trust"),
    ("torch.nn.functional:relu", "R", "test_r31_stdlib_deny: pure torch ops resolve without trust"),
    (
        "torchlens._utils.display:identity",
        "R",
        "test_r31_stdlib_deny: a vetted-inert first-party callable resolves without trust",
    ),
)

# (module, denied) for the structural stdlib detector.
STDLIB_DETECTOR_ROWS: tuple[tuple[str, bool, str], ...] = (
    ("io", True, _R31),
    ("_io", True, _R31),
    ("os", True, _R31),
    ("os.path", True, _R31 + " (submodule keyed on its top level)"),
    ("gc", True, _R31),
    ("mmap", True, _R31),
    ("importlib.util", True, _R31 + " (submodule keyed on its top level)"),
    ("zipimport", True, _R31),
    ("_imp", True, _R31),
    ("operator", False, _R31 + " (carve-out)"),
    ("_operator", False, _R31 + " (carve-out)"),
    ("torch", False, _R31 + " (third party)"),
    ("torch.nn.functional", False, _R31 + " (third party)"),
    ("torchlens.neuro", False, _R31 + " (first party)"),
    ("numpy", False, _R31 + " (third party)"),
    ("mymod", False, _R31 + " (user package)"),
)

# (terminal name, invoker by shape, side-effecting by name) for the r39 structural guards.
STRUCTURAL_NAME_ROWS: tuple[tuple[str, bool, bool, str], ...] = (
    ("map_", True, True, _R39 + ": (map|apply)\\d*_ runner pattern"),
    ("map2_", True, True, _R39 + ": (map|apply)\\d*_ runner pattern"),
    ("map3_", True, True, _R39 + ": future sibling denied by shape"),
    ("map10_", True, True, _R39 + ": future sibling denied by shape"),
    ("apply_", True, True, _R39 + ": (map|apply)\\d*_ runner pattern"),
    ("apply2_", True, True, _R39 + ": future sibling denied by shape"),
    ("register_hook", True, True, _R39 + ": register family"),
    ("register_post_accumulate_grad_hook", True, True, _R39 + ": register family"),
    ("register_forward_hook", True, True, _R39 + ": future nn-shaped sibling"),
    ("_register_device_module", True, True, _R39 + ": register family after underscore strip"),
    ("_register_something_new", True, True, _R39 + ": future sibling denied by shape"),
    ("resize", False, True, _R39 + ": resize substring realloc pattern"),
    ("resize_", False, True, _R39 + ": resize substring realloc pattern"),
    ("resize_as", False, True, _R39 + ": resize substring realloc pattern"),
    ("_resize_output_", False, True, _R39 + ": resize substring realloc pattern"),
    ("_copy_from_and_resize", False, True, _R39 + ": resize substring realloc pattern"),
    ("some_future_resize_op", False, True, _R39 + ": future reallocator denied by shape"),
    ("matmul", False, False, _R39 + ": lookalike, starts with ma not map"),
    ("maximum", False, False, _R39 + ": lookalike, starts with ma not map"),
    ("_sparse_semi_structured_apply", False, False, _R39 + ": aten op, not a runner"),
    ("_sparse_semi_structured_apply_dense", False, False, _R39 + ": aten op, not a runner"),
    ("is_set_to", False, False, _R39 + ": pure read with set_ mid-name"),
    ("masked_fill_", False, False, _R39 + ": trailing-underscore in-place elementwise op"),
)

# (ref, pure) for the purity gate over the live torch surface.
PURITY_ROWS: tuple[tuple[str, bool, str], ...] = (
    ("torch:add", True, _R39 + ": pure forward surface not over-denied"),
    ("torch:matmul", True, _R39 + ": pure forward surface not over-denied"),
    ("torch.nn.functional:conv2d", True, _R39 + ": pure forward surface not over-denied"),
    ("torch.Tensor:add_", True, _R39 + ": in-place elementwise op stays pure"),
    ("torch.Tensor:mul_", True, _R39 + ": in-place elementwise op stays pure"),
    ("torch.Tensor:masked_fill_", True, _R39 + ": in-place elementwise op stays pure"),
    ("torch.Tensor:copy_", True, _R39 + ": in-place elementwise op stays pure"),
    ("torch.Tensor:is_set_to", True, _R39 + ": pure read stays resolvable"),
    ("torch:get_rng_state", True, _R39 + ": pure read stays resolvable"),
    ("torch:initial_seed", True, _R39 + ": pure read stays resolvable"),
    ("torch.Tensor:apply_", False, _R39 + ": r26 denial still holds"),
    ("torch.Tensor:map_", False, _R39 + ": r26 denial still holds"),
    ("torch:compile", False, _R39 + ": r26 denial still holds"),
)


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #


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


_PICKLE_BUILDERS = (_global_ref_pickle, _stack_global_ref_pickle)


def _live(ref: str) -> Callable[..., Any]:
    """Resolve a ``module:qualname`` ref against the live torch namespaces."""

    module, _, qualname = ref.partition(":")
    root: Any = {
        "torch": torch,
        "torch.Tensor": torch.Tensor,
        "torch.nn.functional": torch.nn.functional,
    }[module]
    return getattr(root, qualname)


def _rows(table: tuple[tuple[str, str, str], ...], letter: str) -> list[str]:
    return [ref for ref, checks, _reason in table if letter in checks]


# --------------------------------------------------------------------------- #
# Refused rows.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("ref", _rows(REFUSED_REFS, "R"))
def test_refused_at_resolver_default_and_trusted(ref: str) -> None:
    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(ref)
    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(ref, trust_custom_callables=True)


@pytest.mark.parametrize("ref", _rows(REFUSED_REFS, "L"))
def test_refused_at_resolver_under_own_module_allowlist(ref: str) -> None:
    module_name = ref.split(":", 1)[0]
    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(ref, allowed_custom_callable_modules={module_name})


@pytest.mark.parametrize("builder", _PICKLE_BUILDERS, ids=["GLOBAL", "STACK_GLOBAL"])
@pytest.mark.parametrize("ref", _rows(REFUSED_REFS, "U"))
def test_refused_at_unpickler_default_trusted_and_allowlisted(ref: str, builder) -> None:
    module_name, name = ref.split(":", 1)
    blob = builder(module_name, name)
    with pytest.raises(pickle.UnpicklingError):
        SafeBundleUnpickler(io.BytesIO(blob)).load()
    with pytest.raises(pickle.UnpicklingError):
        SafeBundleUnpickler(io.BytesIO(blob), trust_custom_callables=True).load()
    with pytest.raises(pickle.UnpicklingError):
        SafeBundleUnpickler(io.BytesIO(blob), allowed_custom_callable_modules={module_name}).load()


@pytest.mark.parametrize("ref", _rows(REFUSED_REFS, "P"))
def test_refused_at_purity_gate(ref: str) -> None:
    assert not is_pure_forward_callable(_live(ref))


@pytest.mark.parametrize("ref", _rows(REFUSED_REFS, "E"))
def test_refused_at_run_path_reattach(ref: str) -> None:
    """The sparse run-path reattach ladder resolves the key to no callable."""

    from torchlens._io.runnable_load import CallableRegistryEntry, _resolve_registry_entry

    namespace, _, qualname = ref.partition(":")

    class _Compat:
        backend_version = str(torch.__version__)

    class _Descriptor:
        compatibility = _Compat()
        calls: tuple[Any, ...] = ()
        callable_registry: tuple[Any, ...] = ()
        backend = "torch"

    dispatch = "method" if namespace == "torch.Tensor" else "function"
    entry = CallableRegistryEntry("rows", FunctionRegistryKey(namespace, qualname, dispatch))
    resolution = _resolve_registry_entry(entry, descriptor=_Descriptor(), affected_ops=(), calls=())
    assert resolution.func is None


def test_dotted_walk_off_torch_root_into_stdlib_denied_under_trust() -> None:
    """test_r31_stdlib_deny: ``torch:os.system`` walks to ``posix.system`` and is denied."""

    key = FunctionRegistryKey(
        namespace="custom",
        qualname="os.system",
        dispatch_kind="function",
        import_path="torch:os.system",
    )
    with pytest.raises(UntrustedCallableError):
        resolve_function_registry_key(key, trust_custom_callables=True)


def test_custom_key_missing_import_path_refuses_typed() -> None:
    """test_r31_stdlib_deny (SF-07): a custom key without ``import_path`` refuses with a code."""

    from torchlens._errors import InvalidArgumentError

    key = FunctionRegistryKey(namespace="custom", qualname="foo", dispatch_kind="function")
    with pytest.raises(InvalidArgumentError) as exc_info:
        resolve_function_registry_key(key, trust_custom_callables=True)
    assert exc_info.value.fields["code"] == "custom_callable_import_path_missing"
    assert "module:qualname" in str(exc_info.value.fields["remedy"])


# --------------------------------------------------------------------------- #
# Allowed rows: the legitimate surface still resolves.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("ref", _rows(ALLOWED_REFS, "R"))
def test_allowed_resolves_without_trust(ref: str) -> None:
    resolved = resolve_import_ref(ref)
    assert callable(resolved)
    if ref == "torch:relu":
        assert resolved is torch.relu
    if ref.endswith(":identity"):
        assert resolved(7) == 7


@pytest.mark.parametrize("ref", _rows(ALLOWED_REFS, "L"))
def test_allowed_resolves_through_custom_key_under_trust_and_allowlist(ref: str) -> None:
    module_name, qualname = ref.split(":", 1)
    expected = _live_operator(qualname)
    key = FunctionRegistryKey(
        namespace="custom", qualname=qualname, dispatch_kind="function", import_path=ref
    )
    assert resolve_import_ref(ref) is expected
    assert resolve_function_registry_key(key, trust_custom_callables=True) is expected
    assert (
        resolve_function_registry_key(key, allowed_custom_callable_modules={module_name})
        is expected
    )


@pytest.mark.parametrize("ref", _rows(ALLOWED_REFS, "U"))
def test_allowed_unpickles_under_trust(ref: str) -> None:
    expected = _live_operator(ref.split(":", 1)[1])
    blob = pickle.dumps(expected)
    resolved = SafeBundleUnpickler(io.BytesIO(blob), trust_custom_callables=True).load()
    assert resolved is expected


def _live_operator(qualname: str) -> Callable[..., Any]:
    return getattr(operator, qualname)


# --------------------------------------------------------------------------- #
# Detector and structural-guard rows.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("module", "denied"), [(m, d) for m, d, _ in STDLIB_DETECTOR_ROWS])
def test_stdlib_detector_rows(module: str, denied: bool) -> None:
    assert is_denied_stdlib_or_builtin_module(module) is denied


def test_torch_real_module_is_not_stdlib() -> None:
    """test_r31_stdlib_deny: torch's real top-level package is outside the detector set."""

    assert is_denied_stdlib_or_builtin_module(str(torch.relu.__module__)) is False


def _named(name: str) -> Callable[..., Any]:
    """Return a throwaway callable carrying a synthetic terminal ``__name__``."""

    def _f() -> None:  # pragma: no cover - never invoked; only its name is read.
        return None

    _f.__name__ = name
    _f.__qualname__ = name
    return _f


@pytest.mark.parametrize(
    ("name", "invoker", "side_effecting"), [(n, i, s) for n, i, s, _ in STRUCTURAL_NAME_ROWS]
)
def test_structural_name_guard_rows(name: str, invoker: bool, side_effecting: bool) -> None:
    assert bool(_is_callable_invoker_name(name)) is invoker
    assert bool(_is_side_effecting_callable_name(_named(name))) is side_effecting


@pytest.mark.parametrize(("ref", "pure"), [(r, p) for r, p, _ in PURITY_ROWS])
def test_purity_gate_rows(ref: str, pure: bool) -> None:
    assert bool(is_pure_forward_callable(_live(ref))) is pure


def test_module_load_stays_pure_when_present() -> None:
    """test_r39_callable_invoke_denylist: ``Tensor.module_load`` is a pure read where torch has it."""

    if hasattr(torch.Tensor, "module_load"):
        assert is_pure_forward_callable(torch.Tensor.module_load)
