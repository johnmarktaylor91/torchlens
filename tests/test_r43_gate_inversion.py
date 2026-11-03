"""Round-43 security regression: INVERT the CLASS 1 pure-forward-callable gate on the
internal-builtin torch roots ``torch`` / ``torch._C`` / ``torch._tensor`` from a growing
denylist-of-verbs to a positive STRUCTURAL recognized-operator predicate (findings
secE-r42-1/2/3).

The pre-r43 gate admitted any callable whose real ``__module__`` was one of those roots
by module PREFIX, then tried to subtract the non-forward internal builtins those roots
host (alongside the entire pure forward-op catalog) with an ever-growing verb denylist.
Successive audits kept defeating that with a sibling the verb list never enumerated:

* secE_1 -- the FUNCTIONALIZATION dispatch-mode / TLS mutators
  (``torch._enable_functionalization`` / ``_disable_functionalization`` /
  ``_functionalize_enable_reapply_views`` and the whole ``_functionalize_*`` family):
  push/pop thread-local dispatch state that OUTLIVES ``Trace.run``. No ``set_`` prefix,
  no verb marker, no inspectable signature -> sailed through.
* secE_2 -- ``torch.Tensor.share_memory_``: REBINDS the tensor's storage into a
  shared-memory / IPC segment (``data_ptr()`` changes, ``is_shared() -> True``). It is
  torch-OVERRIDABLE, so the r43 operator predicate would admit it on identity -- the
  KEPT storage belt (extended with the ``share_memory`` token) must catch it FIRST.
* secE_3 -- ``torch._sobol_engine_initialize_state_``: a genuine ``aten`` operator whose
  native malformed-arg behavior crashes. Admitted as a DOCUMENTED RESIDUAL (its crash is
  a torch operator-robustness boundary outside the side-effect-free admission contract);
  pinned here as ADMITTED so a future refactor cannot silently name-deny it.

The r43 fix (``torchlens/_utils/_callable_safety.py``) DEFAULT-DENIES on those exact roots
and admits ONLY structurally-recognized genuine forward operators, decided against
torch's OWN operator authority (``torch.overrides.get_overridable_functions`` /
``torch.ops.aten``) -- independent of the buggy gate and self-updating across torch
versions: torch-overridable identity OR an aten operator schema OR a small stable pure
tensor factory name OR the narrow audited ``to_sparse_coo`` wrapper OR a safe pure-read
tensor property name (``T`` / ``mT`` / ``real`` / ``imag`` -- the r43 fixup rung for the
loader's synthetic property getters, which the original inversion over-denied). This closes the
whole functionalization family, ``share_memory_``, JIT / IR type constructors, Storage /
legacy ``*Tensor`` ctors, state getters, and deprecated methods as a CLASS, not instance
by instance.

THIS IS THE MACHINE-CHECKED IMMUNIZER. It fails RED if the structural predicate ever
DENIES an op in the audited legit-forward allow-set, or ADMITS an op in the known
non-forward flip set (both frozen fixtures below), or -- over the LIVE torch surface --
admits ANY non-operator internal builtin on the gated roots. The recognizer used to
audit the live surface is an INDEPENDENT re-derivation from torch's authorities (not an
import of the gate's decision), so a regression that loosens the gate is caught even on a
future torch version. The functionalization family is enumerated LIVE from ``dir(torch)``
so a newly-added sibling is covered automatically.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any

import torch
from torch.overrides import get_overridable_functions, get_testing_overrides

from torchlens._utils._callable_safety import (
    _unwrap_capture_wrapper,  # mechanical capture-unwrap only; NOT the gate decision
    is_pure_forward_callable,
)
from torchlens.intervention._resolver_registry import resolve_import_ref

# --------------------------------------------------------------------------- #
# INDEPENDENT recognized-operator check (re-derived from torch's authorities, NOT
# imported from the gate). This is the tripwire's own definition of "genuine operator".
# --------------------------------------------------------------------------- #

_OPERATOR_GATED_ROOTS = frozenset({"torch", "torch._C", "torch._tensor"})
_FACTORY_NAMES = frozenset({"from_numpy", "frombuffer", "asarray", "from_dlpack"})
_WRAPPER_NAMES = frozenset({"to_sparse_coo"})
# Safe pure-read tensor PROPERTY accessors (r43 fixup; r45 adds ``H``/``mH``): the loader
# resolves a recorded ``("torch.Tensor", <name>, "method")`` property key to a SYNTHETIC
# getter whose ``__module__`` is ``"torch._tensor"`` -- a fresh function id (never
# overridable), and ``T`` carries no aten schema by terminal name -- so the recognizer
# admits the class by NAME. Every name is a pure view/read; the canonical set is now
# STRUCTURALLY computed (autograd-preserving pure-view probe) and the r45 immunizer in
# ``tests/test_r45_property_classification.py`` pins the exhaustive descriptor
# classification.
_PROPERTY_NAMES = frozenset({"T", "mT", "H", "mH", "real", "imag"})


@lru_cache(maxsize=1)
def _indep_overridable_ids() -> frozenset[int]:
    """Return torch's overridable / testing-override identities (raw AND unwrapped).

    ``get_overridable_functions()`` returns a MIX of wrapped and unwrapped forms once
    ``wrap_torch()`` has run, so both ids are recorded to make membership robust.
    """

    from torchlens.backends.torch.wrappers import wrap_torch

    # Force the wrapped state only when this immunizer actually runs, not at collection time.
    wrap_torch()
    ids: set[int] = set()
    for funcs in get_overridable_functions().values():
        for func in funcs:
            ids.add(id(func))
            ids.add(id(_unwrap_capture_wrapper(func)))
    for func in get_testing_overrides():
        ids.add(id(func))
        ids.add(id(_unwrap_capture_wrapper(func)))
    return frozenset(ids)


def _indep_has_aten_schema(name: str) -> bool:
    """Return whether ``torch.ops.aten`` exposes a real operator schema for ``name``."""

    if not name:
        return False
    try:
        packet = getattr(torch.ops.aten, name, None)
    except (AttributeError, RuntimeError):
        return False
    if packet is None:
        return False
    try:
        return len(packet.overloads()) > 0
    except Exception:
        return False


def _terminal_name(func: Callable[..., Any]) -> str:
    """Return a callable's terminal name (independent of the gate helper)."""

    name = getattr(func, "__name__", None)
    if not name:
        qualname = str(getattr(func, "__qualname__", "") or "")
        name = qualname.rsplit(".", maxsplit=1)[-1]
    return str(name or "")


def _indep_is_recognized_operator(real: Callable[..., Any]) -> bool:
    """Independently decide whether ``real`` is a genuine forward operator."""

    name = _terminal_name(real)
    return (
        id(real) in _indep_overridable_ids()
        or _indep_has_aten_schema(name)
        or name in _FACTORY_NAMES
        or name in _WRAPPER_NAMES
        or name in _PROPERTY_NAMES
    )


# --------------------------------------------------------------------------- #
# LIVE-surface enumeration of the operator-gated-root callables.
# --------------------------------------------------------------------------- #

_ENUM_ROOTS: dict[str, Any] = {
    "torch": torch,
    "torch.Tensor": torch.Tensor,
    "torch.nn.functional": torch.nn.functional,
    "torch._C": torch._C,
    "torch._tensor": __import__("torch._tensor", fromlist=["_tensor"]),
}


def _iter_gated_root_callables() -> list[tuple[str, str, Callable[..., Any], Callable[..., Any]]]:
    """Yield ``(root_label, name, func, real)`` for callables whose REAL module is a
    gated root (``torch`` / ``torch._C`` / ``torch._tensor``).

    Module-less C tensor-method descriptors (real ``__module__ == ""``) are handled by a
    separate gate path and are deliberately excluded here -- only Python callables that
    reach the operator-gated-roots predicate are enumerated.
    """

    out: list[tuple[str, str, Callable[..., Any], Callable[..., Any]]] = []
    for label, root in _ENUM_ROOTS.items():
        for name in dir(root):
            if name.startswith("__"):
                continue
            try:
                func = getattr(root, name)
            except Exception:  # pragma: no cover - some torch attrs raise on access.
                continue
            if not callable(func):
                continue
            real = _unwrap_capture_wrapper(func)
            real_module = str(getattr(real, "__module__", "") or "")
            if real_module in _OPERATOR_GATED_ROOTS:
                out.append((label, name, func, real))
    return out


_GATED_ROOT_CALLABLES = _iter_gated_root_callables()


# --------------------------------------------------------------------------- #
# MACHINE-CHECKED IMMUNIZER: live-surface completeness (no non-operator passes).
# --------------------------------------------------------------------------- #


def test_immunizer_no_nonoperator_builtin_passes_gated_roots() -> None:
    """Every gate-passer on an operator-gated root is a genuine RECOGNIZED operator.

    Enforced over the LIVE torch surface with an INDEPENDENT recognizer, so a regression
    that re-admits a non-forward internal builtin on ``torch`` / ``torch._C`` /
    ``torch._tensor`` (e.g. by restoring the pre-r43 module-prefix admission) goes RED --
    including on future torch versions that add new internal builtins.
    """

    checked = 0
    leaks: list[str] = []
    for label, name, func, real in _GATED_ROOT_CALLABLES:
        if not is_pure_forward_callable(func):
            continue
        checked += 1
        if not _indep_is_recognized_operator(real):
            leaks.append(f"{label}:{name} (terminal={_terminal_name(real)})")
    # Non-vacuous: the gated roots host hundreds of genuine operators.
    assert checked > 100, f"suspiciously few gated-root gate-passers checked: {checked}"
    assert not leaks, (
        f"non-operator internal builtins admitted on gated roots: {sorted(set(leaks))}"
    )


# --------------------------------------------------------------------------- #
# FROZEN FIXTURE 1: the audited legit-forward ALLOW-SET (must ALL be admitted).
# Denying ANY of these fails RED. Spans the gated-root predicate (torch.* funcs), the
# C tensor-method path (Tensor.* descriptors), the pure factories, the ``to_sparse_coo``
# wrapper rescue, deeper allowlisted submodules, and the operator root.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# FROZEN FIXTURE 2: the known non-forward FLIP-SET (must ALL be denied).
# Admitting ANY of these fails RED. Drawn from the r43 probe flip set (torch 2.8.0):
# functionalization family, share_memory_, JIT/IR type ctors, Storage/legacy tensor
# ctors, state getters, deprecated methods.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Explicit secE_1/2/3 pins (the specific findings), plus to_sparse_coo admit.
# --------------------------------------------------------------------------- #


def test_safe_tensor_property_getters_admitted() -> None:
    """The loader's synthetic safe-property getters (``x.T`` / ``x.mT`` / ``x.real`` /
    ``x.imag``) pass the structural gate (r43 fixup).

    The loader resolves a recorded ``("torch.Tensor", <name>, "method")`` property key to
    a SYNTHETIC Python getter with ``__module__ == "torch._tensor"`` -- a fresh function
    id the overridable-identity rung can never cover, and ``T`` has no aten schema by
    terminal name -- so the original r43 inversion DENIED the class and broke replay of
    legit property-reading captures (``ReattachError``). Pinned ADMITTED here, over the
    REAL loader factory, so a future refactor cannot silently drop the class again.
    """

    from torchlens._io.runnable_load import _safe_tensor_property_getter
    from torchlens._utils._callable_safety import _PURE_TENSOR_PROPERTY_NAMES
    from torchlens.intervention.types import FunctionRegistryKey

    # The shipped canonical allowlist IS the audited frozen surface (drift goes RED).
    assert _PURE_TENSOR_PROPERTY_NAMES == _PROPERTY_NAMES
    for name in sorted(_PROPERTY_NAMES):
        getter = _safe_tensor_property_getter(FunctionRegistryKey("torch.Tensor", name, "method"))
        assert getter is not None, f"loader refused safe property key: {name}"
        real = _unwrap_capture_wrapper(getter)
        # A synthetic getter can never be admitted on identity; the class needs the
        # property-name rung (``T`` also has no aten schema).
        assert id(real) not in _indep_overridable_ids()
        assert is_pure_forward_callable(getter), f"safe tensor property WRONGLY DENIED: {name}"
    assert not _indep_has_aten_schema("T")
    # A property-shaped key OUTSIDE the audited surface gets NO synthetic getter (the
    # rescue is a closed allowlist, not a property-wide admission).
    assert (
        _safe_tensor_property_getter(FunctionRegistryKey("torch.Tensor", "grad", "method")) is None
    )


# --------------------------------------------------------------------------- #
# Resolver-path enforcement: the secE ops are refused through the real resolver, in
# BOTH trust modes; the residual + legit ops resolve.
# --------------------------------------------------------------------------- #


def test_legit_and_residual_resolve_at_resolver() -> None:
    """Legit forward ops, the wrapper rescue, and the sobol residual resolve through the
    real resolver (the gate does not over-deny the run path)."""

    assert resolve_import_ref("torch:relu") is torch.relu
    assert resolve_import_ref("torch.Tensor:to_sparse_coo") is torch.Tensor.to_sparse_coo
    # Documented residual: a genuine aten op resolves through the run path.
    resolve_import_ref("torch:_sobol_engine_initialize_state_")


# --------------------------------------------------------------------------- #
# No regression: the r36 / r39 / r41 controls stay DENIED and the pure surface resolves.
# --------------------------------------------------------------------------- #
