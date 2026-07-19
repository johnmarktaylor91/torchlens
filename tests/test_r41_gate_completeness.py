"""Round-41 security regression: make r39's "denied by SHAPE even when never
enumerated" claim TRUE for the callable-INVOKE and global-MUTATOR classes that
``is_pure_forward_callable`` still admitted (finding secE-r40-1).

r39 asserted the higher-order / callback-taking ops and the non-``set_`` process-global
mutators were structurally denied. They were NOT: under DEFAULT untrust the central
runnable-``.tlspec`` purity gate ``is_pure_forward_callable`` still returned ``True``
(ADMITTED) for

* GLOBAL-STATE mutators the r6 ``set_*`` audit missed because none leads with ``set_`` --
  ``torch.autocast_increment_nesting`` / ``autocast_decrement_nesting`` (bump the
  process-global autocast nesting counter that OUTLIVES ``Trace.run``),
  ``torch.clear_autocast_cache`` / ``torch._cufft_clear_plan_cache`` (flush process-global
  caches), and ``torch._cufft_set_plan_cache_max_size`` (resize the global cuFFT plan
  cache). Zero/low-arg, take no callable -> directly reachable via the run-path;
* callable-INVOKE / higher-order ops in the SAME class as the r39-denied ``vmap`` but
  carrying no ``map`` / ``register`` name marker -- ``torch.cond`` / ``torch.while_loop``
  (higher-order control flow that invokes the branch callables and routes through the
  DENIED ``torch.compile``), ``torch.nn.functional.handle_torch_function`` (its
  ``public_api``), ``torch.nn.functional.triplet_margin_with_distance_loss`` (its
  ``distance_function``), and ``torch._check_with`` (its ``message``).

The fix (``torchlens/utils/_callable_safety.py``) closes each CLASS structurally so a
FUTURE sibling is denied by SHAPE even if never enumerated:

* callable-INVOKE by SIGNATURE -- ``_signature_invokes_callable`` denies any op whose
  ``inspect.signature`` exposes a ``Callable``-annotated OR callable-named parameter
  (``fn`` / ``func`` / ``callback`` / ``hook`` / ``if_true`` / ``if_false`` / any
  ``*_fn`` / ``*_func``). This complements r39's ``(map|apply)\\d*_`` / leading-``register``
  NAME guard, which those higher-order ops evade;
* global-MUTATE by non-``set_`` VERB -- ``_is_global_state_mutator_name`` denies the
  ``nesting`` counter, the ``clear`` + ``cache`` flushers, and the ``set_plan_cache``
  sizer, complementing r6's leading-``set_`` prefix guard.

THIS IS THE MACHINE-CHECKED IMMUNIZER. The enumeration tests below FAIL if ANY
``torch`` / ``torch.Tensor`` / ``torch.nn.functional`` callable whose signature exposes a
``Callable`` parameter, OR whose name matches a global-mutator verb, ever passes
``is_pure_forward_callable`` under default untrust -- so "denied by shape" is ENFORCED
against the LIVE torch surface (incl. future torch versions), not merely asserted. The
detectors here are INDEPENDENT re-implementations (not imports of the gate internals) so a
regression that weakens the gate is caught. The pure forward surface -- including the
deliberately tricky lookalikes ``nuclear_norm`` (``clear`` without ``cache``),
``_fake_quantize_..._cachemask_...`` (``cache`` without ``clear``),
``_autocast_to_full_precision``, and every autocast / cuFFT-plan-cache GETTER -- must stay
resolvable.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

import pytest
import torch

from torchlens.backends.torch.wrappers import wrap_torch
from torchlens.intervention.errors import UntrustedCallableError
from torchlens.intervention.resolver import resolve_import_ref
from torchlens.utils._callable_safety import is_pure_forward_callable

# Install the torch wrappers so resolved descriptors mirror the live (wrapped) state:
# the gate must unwrap ``torchlens.backends.torch.wrappers`` before deciding.
wrap_torch()


_FIXED_ROOTS: dict[str, Any] = {
    "torch": torch,
    "torch.Tensor": torch.Tensor,
    "torch.nn.functional": torch.nn.functional,
}


# --------------------------------------------------------------------------- #
# INDEPENDENT detectors (deliberately NOT imported from the gate) -- these are the
# tripwire's own definition of the denied classes.
# --------------------------------------------------------------------------- #

# Conventional callable-parameter names for higher-order ops whose callable arg is
# UNANNOTATED (``torch.while_loop(cond_fn, body_fn, ...)`` / ``classproperty(func)``).
_CALLABLE_PARAM_NAMES = frozenset(
    {"fn", "func", "callback", "hook", "closure", "body", "branch", "if_true", "if_false"}
)


def _has_callable_annotation(func: Callable[..., Any]) -> bool:
    """Return whether any parameter is annotated ``Callable`` (independent check)."""

    try:
        signature = inspect.signature(func)
    except (ValueError, TypeError):
        return False
    for parameter in signature.parameters.values():
        if parameter.annotation is inspect.Parameter.empty:
            continue
        if "callable" in str(parameter.annotation).lower():
            return True
    return False


def _has_callable_named_param(func: Callable[..., Any]) -> bool:
    """Return whether any parameter is a conventionally-named callable (independent)."""

    try:
        signature = inspect.signature(func)
    except (ValueError, TypeError):
        return False
    for parameter in signature.parameters.values():
        low = parameter.name.lower()
        if low in _CALLABLE_PARAM_NAMES or low.endswith("_fn") or low.endswith("_func"):
            return True
    return False


def _is_global_mutator_verb(name: str) -> bool:
    """Return whether a terminal name matches a NON-``set_`` global-mutator verb."""

    low = name.lower()
    if "nesting" in low:
        return True
    if "clear" in low and "cache" in low:
        return True
    if "set_plan_cache" in low:
        return True
    return False


def _iter_fixed_root_callables() -> list[tuple[str, str, Callable[..., Any]]]:
    """Yield ``(namespace, qualname, callable)`` for every fixed-root attribute."""

    out: list[tuple[str, str, Callable[..., Any]]] = []
    for namespace, root in _FIXED_ROOTS.items():
        for name in dir(root):
            try:
                attr = getattr(root, name)
            except Exception:  # pragma: no cover - some torch attrs raise on access.
                continue
            if callable(attr):
                out.append((namespace, name, attr))
    return out


# --------------------------------------------------------------------------- #
# MACHINE-CHECKED IMMUNIZERS: enforce "denied by shape" over the LIVE torch surface.
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_immunizer_no_callable_annotation_op_passes_gate() -> None:
    """No ``Callable``-annotated op may pass the purity gate (enforced, not asserted)."""

    leaks = [
        f"{ns}:{q}"
        for ns, q, func in _iter_fixed_root_callables()
        if _has_callable_annotation(func) and is_pure_forward_callable(func)
    ]
    assert not leaks, f"Callable-annotated ops still admitted by the gate: {sorted(set(leaks))}"


@pytest.mark.smoke
def test_immunizer_no_callable_named_param_op_passes_gate() -> None:
    """No higher-order op with a callable-named parameter may pass the gate.

    Catches the UNANNOTATED higher-order ops (``torch.while_loop`` /
    ``torch.nn.functional.boolean_dispatch`` / ``torch.Tensor.__torch_function__``) that
    the annotation sweep alone would miss.
    """

    leaks = [
        f"{ns}:{q}"
        for ns, q, func in _iter_fixed_root_callables()
        if _has_callable_named_param(func) and is_pure_forward_callable(func)
    ]
    assert not leaks, f"Callable-named-param ops still admitted by the gate: {sorted(set(leaks))}"


@pytest.mark.smoke
def test_immunizer_no_global_mutator_verb_passes_gate() -> None:
    """No callable whose name matches a global-mutator verb may pass the gate."""

    leaks = [
        f"{ns}:{q}"
        for ns, q, func in _iter_fixed_root_callables()
        if _is_global_mutator_verb(q) and is_pure_forward_callable(func)
    ]
    assert not leaks, f"Global-mutator-verb ops still admitted by the gate: {sorted(set(leaks))}"


# --------------------------------------------------------------------------- #
# The specific secE-r40-1 ops, as explicit assertions (gate + resolver, both trust modes).
# --------------------------------------------------------------------------- #

_GLOBAL_MUTATOR_REFS = [
    "torch:autocast_increment_nesting",
    "torch:autocast_decrement_nesting",
    "torch:clear_autocast_cache",
    "torch:_cufft_clear_plan_cache",
    "torch:_cufft_set_plan_cache_max_size",
]
_CALLABLE_INVOKE_REFS = [
    "torch:cond",
    "torch:while_loop",
    "torch.nn.functional:handle_torch_function",
    "torch.nn.functional:triplet_margin_with_distance_loss",
    "torch:_check_with",
]
_ALL_SECE_R40_REFS = _GLOBAL_MUTATOR_REFS + _CALLABLE_INVOKE_REFS


def _resolve_attr(ref: str) -> Callable[..., Any]:
    namespace, _, qualname = ref.partition(":")
    return getattr(_FIXED_ROOTS[namespace], qualname)


@pytest.mark.smoke
@pytest.mark.parametrize("ref", _ALL_SECE_R40_REFS)
def test_secE_r40_targets_fail_purity_gate(ref: str) -> None:
    """Every secE-r40-1 op is refused by the purity gate."""

    assert not is_pure_forward_callable(_resolve_attr(ref))


@pytest.mark.smoke
@pytest.mark.parametrize("ref", _ALL_SECE_R40_REFS)
def test_secE_r40_targets_denied_at_resolver_even_under_trust(ref: str) -> None:
    """Fixed-root refs for the secE-r40-1 ops never resolve, even with trust satisfied."""

    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(ref)
    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(ref, trust_custom_callables=True)


# --------------------------------------------------------------------------- #
# Structural guards deny FUTURE siblings by SHAPE (synthetic names / ops).
# --------------------------------------------------------------------------- #


def _named(name: str) -> Callable[..., Any]:
    def _f() -> None:  # pragma: no cover - never invoked; only its name is read.
        return None

    _f.__name__ = name
    _f.__qualname__ = name
    return _f


@pytest.mark.smoke
@pytest.mark.parametrize(
    "name",
    [
        "autocast_increment_nesting",
        "autocast_decrement_nesting",
        "some_future_increment_nesting",  # future counter sibling
        "clear_autocast_cache",
        "_cufft_clear_plan_cache",
        "flush_some_future_cache_clear",  # future flusher sibling (clear + cache)
        "_cufft_set_plan_cache_max_size",
        "_some_future_set_plan_cache_size",  # future plan-cache sizer sibling
    ],
)
def test_secE_r40_future_global_mutator_siblings_denied(name: str) -> None:
    """The ``nesting`` / ``clear``+``cache`` / ``set_plan_cache`` verb close is by shape."""

    assert not is_pure_forward_callable(_named(name))


def _higher_order(param_annotation: Any, param_name: str = "fn") -> Callable[..., Any]:
    """Return a synthetic op with a single callable-shaped parameter."""

    def _f(x: Any) -> Any:  # pragma: no cover - never invoked.
        return x

    _f.__name__ = "future_higher_order_op"
    _f.__qualname__ = "future_higher_order_op"
    parameter = inspect.Parameter(
        param_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=param_annotation
    )
    _f.__signature__ = inspect.Signature([parameter])  # type: ignore[attr-defined]
    _f.__module__ = "torch"  # would otherwise pass the module allowlist
    return _f


@pytest.mark.smoke
def test_secE_r40_future_higher_order_ops_denied_by_signature() -> None:
    """A future callable-taking op is denied by SIGNATURE shape (annotation OR name)."""

    # Callable-annotated parameter (even with a benign parameter name).
    assert not is_pure_forward_callable(_higher_order(Callable[..., Any], "benign"))
    assert not is_pure_forward_callable(_higher_order("Callable[[Tensor], Tensor]", "benign"))
    # UNANNOTATED but callable-named parameters (mirrors ``while_loop(cond_fn, body_fn)``).
    assert not is_pure_forward_callable(_higher_order(inspect.Parameter.empty, "cond_fn"))
    assert not is_pure_forward_callable(_higher_order(inspect.Parameter.empty, "callback"))
    assert not is_pure_forward_callable(_higher_order(inspect.Parameter.empty, "if_true"))


# --------------------------------------------------------------------------- #
# The new guards must NOT over-deny pure ops that merely resemble a denied shape.
# --------------------------------------------------------------------------- #

# Pure forward ops whose names contain a mutator token but are NOT global mutators.
_MUTATOR_LOOKALIKE_PURE_REFS = [
    "torch:nuclear_norm",  # "clear" via nu-CLEAR-norm, no "cache"
    "torch:_fake_quantize_per_tensor_affine_cachemask_tensor_qparams",  # "cache", no "clear"
    "torch.Tensor:_autocast_to_full_precision",  # a casting op, not a mutator
    "torch.Tensor:_autocast_to_reduced_precision",
    # Autocast / cuFFT-plan-cache GETTERS (pure reads) stay resolvable.
    "torch:is_autocast_enabled",
    "torch:get_autocast_dtype",
    "torch:_cufft_get_plan_cache_size",
    "torch:_cufft_get_plan_cache_max_size",
]


@pytest.mark.smoke
@pytest.mark.parametrize("ref", _MUTATOR_LOOKALIKE_PURE_REFS)
def test_secE_r40_mutator_lookalikes_not_over_denied(ref: str) -> None:
    """The verb close preserves pure ops that merely contain a mutator token."""

    assert is_pure_forward_callable(_resolve_attr(ref))


@pytest.mark.smoke
def test_secE_r40_pure_forward_surface_resolves() -> None:
    """The r41 additions do not over-deny the legitimate pure forward surface."""

    # Ordinary pure ops, in-place elementwise ops, and view ops stay resolvable.
    assert resolve_import_ref("torch:relu") is torch.relu
    assert is_pure_forward_callable(torch.add)
    assert is_pure_forward_callable(torch.matmul)
    assert is_pure_forward_callable(torch.nn.functional.conv2d)
    assert is_pure_forward_callable(torch.Tensor.add)
    assert is_pure_forward_callable(torch.Tensor.add_)
    assert is_pure_forward_callable(torch.Tensor.mul_)
    assert is_pure_forward_callable(torch.Tensor.masked_fill_)
    assert is_pure_forward_callable(torch.Tensor.copy_)
    # View / stride ops (explicitly required to keep resolving).
    assert is_pure_forward_callable(torch.Tensor.as_strided)
    assert is_pure_forward_callable(torch.Tensor.narrow)
    assert is_pure_forward_callable(torch.Tensor.view)
    assert is_pure_forward_callable(torch.Tensor.expand)
    # Pure READS / getters are unaffected.
    assert is_pure_forward_callable(torch.Tensor.is_set_to)
    assert is_pure_forward_callable(torch.get_rng_state)
    assert is_pure_forward_callable(torch.initial_seed)
    assert is_pure_forward_callable(torch.Tensor.module_load)


# --------------------------------------------------------------------------- #
# No regression: the r39 / r6 controls stay DENIED.
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_secE_r40_prior_controls_still_denied() -> None:
    """The r39 (invoke) and r6 (mutator) controls remain refused (no regression)."""

    assert not is_pure_forward_callable(torch.vmap)
    assert not is_pure_forward_callable(torch.Tensor.apply_)
    assert not is_pure_forward_callable(torch.Tensor.map_)
    assert not is_pure_forward_callable(torch.Tensor.map2_)
    assert not is_pure_forward_callable(torch.Tensor.register_hook)
    assert not is_pure_forward_callable(torch.compile)
    assert not is_pure_forward_callable(torch.set_grad_enabled)
    assert not is_pure_forward_callable(torch.manual_seed)
    assert not is_pure_forward_callable(torch.set_default_dtype)
    assert not is_pure_forward_callable(torch.Tensor.resize_)
    assert not is_pure_forward_callable(torch.Tensor.set_)
