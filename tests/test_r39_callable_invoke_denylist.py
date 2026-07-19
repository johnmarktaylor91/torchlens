"""Round-39 security regression: close the arbitrary-callable-INVOKE class in the
``is_pure_forward_callable`` denylist (finding secE-r38-1).

The central runnable-``.tlspec`` purity gate ``is_pure_forward_callable`` denies
side-effecting torch/tensor callables resolved from an UNTRUSTED bundle. r26 denied
the elementwise Python-callable runners ``Tensor.apply_`` / ``Tensor.map_`` because
they INVOKE an attacker-supplied callable per element -- but the enumeration MISSED the
exact sibling class:

* ``Tensor.map2_`` -- the 3-tensor elementwise Python-callable runner (identical
  primitive to ``map_``);
* ``torch.vmap`` -- a transform (real module ``torch.func``) that INVOKES an attacker fn;
* ``Tensor.register_hook`` / ``Tensor.register_post_accumulate_grad_hook`` -- register an
  arbitrary callback fired on backward.

Adjacent denied-class misses were also admitted: the storage-REALLOCATORS
``torch._resize_output_`` / ``torch._copy_from_and_resize`` (r6 ``resize_`` class) and the
global-state mutator ``torch._register_device_module`` (r6 ``set_*`` class).

The fix adds the named misses AND closes each CLASS structurally so a FUTURE sibling
(``map3_`` / a new ``register_*hook`` / a new ``*_resize_*`` reallocator) is denied by
SHAPE even if never enumerated:
* ``(map|apply)\\d*_`` elementwise-runner pattern;
* a leading ``register`` (after underscore-strip) registration-family pattern;
* a ``resize`` substring storage-realloc pattern.

These tests mirror the r26 patterns: denial holds at the purity gate, at the intervention
resolver boundary, and at the sparse run-path reattach boundary -- on the default path AND
under explicit trust -- while the legitimate pure forward surface is NOT over-denied.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest
import torch

from torchlens.backends.torch.wrappers import wrap_torch
from torchlens.intervention.errors import UntrustedCallableError
from torchlens.intervention.resolver import resolve_import_ref
from torchlens.utils._callable_safety import (
    _is_callable_invoker_name,
    _is_side_effecting_callable_name,
    is_pure_forward_callable,
)

# Ensure the torch wrappers are installed so the resolved descriptors mirror the live
# state (wrapped ops report ``__module__ == 'torchlens.backends.torch.wrappers'``; the
# gate must unwrap them before deciding).
wrap_torch()


# --------------------------------------------------------------------------- #
# The confirmed named misses, grouped by class.
# --------------------------------------------------------------------------- #
_CALLABLE_INVOKE_REFS = [
    "torch.Tensor:map2_",
    "torch:vmap",
    "torch.Tensor:register_hook",
    "torch.Tensor:register_post_accumulate_grad_hook",
]
_STORAGE_REALLOC_REFS = [
    "torch:_resize_output_",
    "torch:_copy_from_and_resize",
    "torch.Tensor:resize",
    "torch.Tensor:resize_as",
]
_GLOBAL_MUTATOR_REFS = [
    "torch:_register_device_module",
]
_ALL_R39_REFS = _CALLABLE_INVOKE_REFS + _STORAGE_REALLOC_REFS + _GLOBAL_MUTATOR_REFS


def _resolve(ref: str) -> Callable[..., Any]:
    """Resolve a ``module:qualname`` ref against the live torch namespaces."""

    module, _, qualname = ref.partition(":")
    root: Any = torch
    if module == "torch.Tensor":
        root = torch.Tensor
    elif module == "torch.nn.functional":
        root = torch.nn.functional
    return getattr(root, qualname)


# --------------------------------------------------------------------------- #
# Purity gate: every r39 miss is refused.
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
@pytest.mark.parametrize("ref", _ALL_R39_REFS)
def test_r39_misses_fail_purity_gate(ref: str) -> None:
    """The purity gate refuses every callable-invoke / realloc / mutator miss."""

    assert not is_pure_forward_callable(_resolve(ref))


# --------------------------------------------------------------------------- #
# Resolver boundary: refused on the default path AND under trust.
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
@pytest.mark.parametrize("ref", _ALL_R39_REFS)
def test_r39_misses_denied_at_resolver_even_under_trust(ref: str) -> None:
    """Fixed-root refs for the r39 misses never resolve, even with trust satisfied."""

    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(ref)
    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(ref, trust_custom_callables=True)


# --------------------------------------------------------------------------- #
# Structural guards: FUTURE siblings are denied by SHAPE, not just enumeration.
# --------------------------------------------------------------------------- #


def _named(name: str) -> Callable[..., Any]:
    """Return a throwaway callable carrying a synthetic terminal ``__name__``."""

    def _f() -> None:  # pragma: no cover - never invoked; only its name is read.
        return None

    _f.__name__ = name
    _f.__qualname__ = name
    return _f


# map/apply elementwise-runner pattern (future map3_ / apply2_) and register family.
_STRUCTURAL_INVOKER_NAMES = [
    "map_",
    "map2_",
    "map3_",
    "map10_",
    "apply_",
    "apply2_",
    "register_hook",
    "register_post_accumulate_grad_hook",
    "register_forward_hook",  # future/nn-shaped sibling
    "_register_device_module",
    "_register_something_new",
]
# storage-realloc substring pattern (future *_resize_* reallocators).
_STRUCTURAL_RESIZE_NAMES = [
    "resize",
    "resize_",
    "resize_as",
    "_resize_output_",
    "_copy_from_and_resize",
    "some_future_resize_op",
]


@pytest.mark.smoke
@pytest.mark.parametrize("name", _STRUCTURAL_INVOKER_NAMES)
def test_r39_structural_invoker_pattern_denies_future_siblings(name: str) -> None:
    """The (map|apply)\\d*_ / register structural guard denies future siblings by name."""

    assert _is_callable_invoker_name(name)
    assert _is_side_effecting_callable_name(_named(name))


@pytest.mark.smoke
@pytest.mark.parametrize("name", _STRUCTURAL_RESIZE_NAMES)
def test_r39_structural_resize_pattern_denies_future_siblings(name: str) -> None:
    """The ``resize`` substring guard denies future reallocators by name."""

    assert _is_side_effecting_callable_name(_named(name))


# --------------------------------------------------------------------------- #
# The structural guards must NOT over-deny lookalikes that are pure ops.
# --------------------------------------------------------------------------- #

# Pure ops whose names merely resemble a denied pattern but are NOT invokers/reallocators.
_PATTERN_LOOKALIKE_PURE_NAMES = [
    "matmul",  # starts with "ma", not "map"
    "maximum",  # starts with "ma", not "map"
    "_sparse_semi_structured_apply",  # aten sparse op, NOT a Python-callable runner
    "_sparse_semi_structured_apply_dense",
    "is_set_to",  # pure read; contains "set_" mid-name (leading-only set_ guard preserves it)
    "masked_fill_",  # trailing underscore in-place elementwise op
]


@pytest.mark.smoke
@pytest.mark.parametrize("name", _PATTERN_LOOKALIKE_PURE_NAMES)
def test_r39_structural_guards_do_not_over_deny_lookalikes(name: str) -> None:
    """Anchored patterns exclude pure ops that merely resemble a denied shape."""

    assert not _is_callable_invoker_name(name)
    assert not _is_side_effecting_callable_name(_named(name))


@pytest.mark.smoke
def test_r39_pure_forward_surface_not_over_denied() -> None:
    """The r39 additions do not over-deny the legitimate pure forward surface."""

    # Ordinary pure / in-place elementwise ops and pure reads stay resolvable.
    assert resolve_import_ref("torch:relu") is torch.relu
    assert is_pure_forward_callable(torch.add)
    assert is_pure_forward_callable(torch.matmul)
    assert is_pure_forward_callable(torch.nn.functional.conv2d)
    assert is_pure_forward_callable(torch.Tensor.add_)
    assert is_pure_forward_callable(torch.Tensor.mul_)
    assert is_pure_forward_callable(torch.Tensor.masked_fill_)
    assert is_pure_forward_callable(torch.Tensor.copy_)
    # Pure READS/getters (finding: getters stay resolvable) are unaffected.
    assert is_pure_forward_callable(torch.Tensor.is_set_to)
    assert is_pure_forward_callable(torch.get_rng_state)
    assert is_pure_forward_callable(torch.initial_seed)
    assert is_pure_forward_callable(torch.Tensor.module_load)
    # The r26 denials still hold (no regression).
    assert not is_pure_forward_callable(torch.Tensor.apply_)
    assert not is_pure_forward_callable(torch.Tensor.map_)
    assert not is_pure_forward_callable(torch.compile)


# --------------------------------------------------------------------------- #
# Sparse run-path reattach boundary (the finding reaches this too).
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
@pytest.mark.parametrize(
    "namespace,qualname",
    [
        ("torch.Tensor", "map2_"),
        ("torch.Tensor", "register_hook"),
        ("torch", "vmap"),
        ("torch", "_resize_output_"),
        ("torch", "_register_device_module"),
    ],
)
def test_r39_misses_denied_at_run_path_reattach(namespace: str, qualname: str) -> None:
    """The sparse run-path reattach ladder refuses the r39 misses (no ``.func``)."""

    from torchlens._io.runnable_load import (
        CallableRegistryEntry,
        _resolve_registry_entry,
    )
    from torchlens.intervention.types import FunctionRegistryKey

    class _Compat:
        backend_version = str(torch.__version__)

    class _Descriptor:
        compatibility = _Compat()
        calls: tuple[Any, ...] = ()
        callable_registry: tuple[Any, ...] = ()
        backend = "torch"

    dispatch = "method" if namespace == "torch.Tensor" else "function"
    entry = CallableRegistryEntry("r39", FunctionRegistryKey(namespace, qualname, dispatch))
    resolution = _resolve_registry_entry(entry, descriptor=_Descriptor(), affected_ops=(), calls=())
    assert resolution.func is None
