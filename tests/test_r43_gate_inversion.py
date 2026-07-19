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

The r43 fix (``torchlens/utils/_callable_safety.py``) DEFAULT-DENIES on those exact roots
and admits ONLY structurally-recognized genuine forward operators, decided against
torch's OWN operator authority (``torch.overrides.get_overridable_functions`` /
``torch.ops.aten``) -- independent of the buggy gate and self-updating across torch
versions: torch-overridable identity OR an aten operator schema OR a small stable pure
tensor factory name OR the narrow audited ``to_sparse_coo`` wrapper. This closes the
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

from typing import Any, Callable

import pytest
import torch
from torch.overrides import get_overridable_functions, get_testing_overrides

from torchlens.backends.torch.wrappers import wrap_torch
from torchlens.intervention.errors import UntrustedCallableError
from torchlens.intervention.resolver import resolve_import_ref
from torchlens.utils._callable_safety import (
    _unwrap_capture_wrapper,  # mechanical capture-unwrap only; NOT the gate decision
    is_pure_forward_callable,
)

# Install the torch wrappers so resolved callables mirror the live (wrapped) state: the
# gate -- and this immunizer's independent recognizer -- must unwrap before deciding.
wrap_torch()


# --------------------------------------------------------------------------- #
# INDEPENDENT recognized-operator check (re-derived from torch's authorities, NOT
# imported from the gate). This is the tripwire's own definition of "genuine operator".
# --------------------------------------------------------------------------- #

_OPERATOR_GATED_ROOTS = frozenset({"torch", "torch._C", "torch._tensor"})
_FACTORY_NAMES = frozenset({"from_numpy", "frombuffer", "asarray", "from_dlpack"})
_WRAPPER_NAMES = frozenset({"to_sparse_coo"})


def _indep_overridable_ids() -> frozenset[int]:
    """Return torch's overridable / testing-override identities (raw AND unwrapped).

    ``get_overridable_functions()`` returns a MIX of wrapped and unwrapped forms once
    ``wrap_torch()`` has run, so both ids are recorded to make membership robust.
    """

    ids: set[int] = set()
    for funcs in get_overridable_functions().values():
        for func in funcs:
            ids.add(id(func))
            ids.add(id(_unwrap_capture_wrapper(func)))
    for func in get_testing_overrides():
        ids.add(id(func))
        ids.add(id(_unwrap_capture_wrapper(func)))
    return frozenset(ids)


_OVERRIDABLE_IDS = _indep_overridable_ids()


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
        id(real) in _OVERRIDABLE_IDS
        or _indep_has_aten_schema(name)
        or name in _FACTORY_NAMES
        or name in _WRAPPER_NAMES
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


@pytest.mark.smoke
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

_AUDITED_LEGIT_FORWARD_ALLOW_SET: tuple[tuple[str, str], ...] = (
    # torch root forward ops / factories (gated-root recognized-operator predicate).
    ("torch", "relu"),
    ("torch", "add"),
    ("torch", "sub"),
    ("torch", "mul"),
    ("torch", "div"),
    ("torch", "matmul"),
    ("torch", "bmm"),
    ("torch", "mm"),
    ("torch", "einsum"),
    ("torch", "cat"),
    ("torch", "stack"),
    ("torch", "mean"),
    ("torch", "sum"),
    ("torch", "softmax"),
    ("torch", "sigmoid"),
    ("torch", "tanh"),
    ("torch", "zeros"),
    ("torch", "ones"),
    ("torch", "empty"),
    ("torch", "arange"),
    ("torch", "as_tensor"),
    ("torch", "sparse_coo_tensor"),
    ("torch", "nuclear_norm"),
    ("torch", "_conj_physical"),
    # Pure tensor factories (neither overridable nor aten by terminal name).
    ("torch", "from_numpy"),
    ("torch", "frombuffer"),
    ("torch", "asarray"),
    ("torch", "from_dlpack"),
    # C tensor-method descriptors (module-less tmd path) + in-place / view ops.
    ("torch.Tensor", "view"),
    ("torch.Tensor", "reshape"),
    ("torch.Tensor", "expand"),
    ("torch.Tensor", "narrow"),
    ("torch.Tensor", "as_strided"),
    ("torch.Tensor", "add_"),
    ("torch.Tensor", "mul_"),
    ("torch.Tensor", "relu_"),
    ("torch.Tensor", "copy_"),
    ("torch.Tensor", "masked_fill_"),
    ("torch.Tensor", "scatter_add_"),
    ("torch.Tensor", "to_sparse"),
    # The narrow audited pure Tensor WRAPPER rescue (the one live method the raw
    # predicate dropped).
    ("torch.Tensor", "to_sparse_coo"),
    # Deeper allowlisted operator submodules keep module-prefix admission.
    ("torch.nn.functional", "linear"),
    ("torch.nn.functional", "conv2d"),
    ("torch.nn.functional", "relu"),
    ("torch.nn.functional", "softmax"),
    ("torch.linalg", "norm"),
    ("torch.fft", "fft"),
    # Operator root positive allowlist.
    ("operator", "add"),
    ("operator", "getitem"),
)


def _resolve_attr(namespace: str, qualname: str) -> Any:
    if namespace == "operator":
        import operator

        return getattr(operator, qualname, None)
    root: Any = torch
    for part in namespace.removeprefix("torch").lstrip(".").split("."):
        if not part:
            continue
        root = getattr(root, part, None)
        if root is None:
            return None
    return getattr(root, qualname, None)


@pytest.mark.smoke
@pytest.mark.parametrize("namespace,qualname", _AUDITED_LEGIT_FORWARD_ALLOW_SET)
def test_immunizer_audited_allow_set_admitted(namespace: str, qualname: str) -> None:
    """No op in the audited legit-forward allow-set may be denied by the structural gate."""

    obj = _resolve_attr(namespace, qualname)
    if obj is None:  # pragma: no cover - version drift; a floor test guards vacuity.
        pytest.skip(f"{namespace}:{qualname} absent on torch {torch.__version__}")
    assert is_pure_forward_callable(obj), (
        f"audited legit forward op WRONGLY DENIED: {namespace}:{qualname}"
    )


@pytest.mark.smoke
def test_immunizer_allow_set_is_non_vacuous() -> None:
    """The frozen allow-set overwhelmingly resolves on this torch (guards vacuity)."""

    present = sum(
        1 for ns, q in _AUDITED_LEGIT_FORWARD_ALLOW_SET if _resolve_attr(ns, q) is not None
    )
    assert present >= len(_AUDITED_LEGIT_FORWARD_ALLOW_SET) - 3


# --------------------------------------------------------------------------- #
# FROZEN FIXTURE 2: the known non-forward FLIP-SET (must ALL be denied).
# Admitting ANY of these fails RED. Drawn from the r43 probe flip set (torch 2.8.0):
# functionalization family, share_memory_, JIT/IR type ctors, Storage/legacy tensor
# ctors, state getters, deprecated methods.
# --------------------------------------------------------------------------- #

_KNOWN_NON_FORWARD_FLIP_SET: tuple[tuple[str, str], ...] = (
    # secE_1: functionalization dispatch-mode / TLS mutators.
    ("torch", "_enable_functionalization"),
    ("torch", "_disable_functionalization"),
    ("torch", "_functionalize_enable_reapply_views"),
    ("torch", "_functionalize_sync"),
    ("torch", "_functionalize_replace"),
    ("torch", "_functionalize_commit_update"),
    ("torch", "_functionalize_unsafe_set"),
    ("torch", "_functionalize_mark_mutation_hidden_from_autograd"),
    # secE_2: storage-rebind (overridable but belt-denied).
    ("torch.Tensor", "share_memory_"),
    # JIT / IR type constructors reachable at the torch root.
    ("torch", "AnyType"),
    ("torch", "BoolType"),
    ("torch", "ClassType"),
    ("torch", "Argument"),
    ("torch", "Block"),
    ("torch", "Code"),
    # Storage / legacy ``*Tensor`` type constructors.
    ("torch", "BoolStorage"),
    ("torch", "ByteStorage"),
    ("torch", "BoolTensor"),
    ("torch", "ByteTensor"),
    # State getters / internal plumbing.
    ("torch.Tensor", "_typed_storage"),
    ("torch.Tensor", "_reduce_ex_internal"),
    # Deprecated / removed Tensor methods that are NOT in torch's overridable registry
    # (a captured DAG resolves to torch.linalg.* / aten.*, never these). NOTE: the
    # tombstones ``solve`` / ``reinforce`` are deliberately EXCLUDED from this flip set --
    # torch's own operator authority still lists them as overridable, so the structural
    # predicate admits them by design (defer-to-torch). That is harmless: they are pure
    # RuntimeError raises with zero side effects and never appear in a real captured DAG.
    # ``eig`` / ``lstsq`` / ``symeig`` are absent from the overridable registry and are
    # genuinely denied.
    ("torch.Tensor", "eig"),
    ("torch.Tensor", "lstsq"),
    ("torch.Tensor", "symeig"),
)


@pytest.mark.smoke
@pytest.mark.parametrize("namespace,qualname", _KNOWN_NON_FORWARD_FLIP_SET)
def test_immunizer_known_flip_set_denied(namespace: str, qualname: str) -> None:
    """No op in the known non-forward flip set may be admitted by the structural gate."""

    obj = _resolve_attr(namespace, qualname)
    if obj is None:  # pragma: no cover - version drift; a floor test guards vacuity.
        pytest.skip(f"{namespace}:{qualname} absent on torch {torch.__version__}")
    assert not is_pure_forward_callable(obj), (
        f"non-forward internal builtin WRONGLY ADMITTED: {namespace}:{qualname}"
    )


@pytest.mark.smoke
def test_immunizer_flip_set_is_non_vacuous() -> None:
    """The frozen flip-set overwhelmingly resolves on this torch (guards vacuity)."""

    present = sum(1 for ns, q in _KNOWN_NON_FORWARD_FLIP_SET if _resolve_attr(ns, q) is not None)
    assert present >= len(_KNOWN_NON_FORWARD_FLIP_SET) - 3


# --------------------------------------------------------------------------- #
# Explicit secE_1/2/3 pins (the specific findings), plus to_sparse_coo admit.
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_secE_1_whole_functionalization_family_denied() -> None:
    """The ENTIRE functionalization family is denied -- enumerated LIVE from dir(torch).

    Covers any newly-added ``_functionalize_*`` / ``*functionaliz*`` sibling automatically.
    """

    family = [
        name
        for name in dir(torch)
        if "functionaliz" in name.lower() and callable(getattr(torch, name, None))
    ]
    # Non-vacuous: the family exists on every supported torch.
    assert len(family) >= 3, f"functionalization family unexpectedly small: {family}"
    admitted = [name for name in family if is_pure_forward_callable(getattr(torch, name))]
    assert not admitted, f"functionalization ops WRONGLY ADMITTED: {sorted(admitted)}"


@pytest.mark.smoke
def test_secE_2_share_memory_denied_reader_resolves() -> None:
    """``share_memory_`` (storage rebind) is denied though it is torch-overridable; the
    pure ``is_shared`` reader stays resolvable."""

    # Sanity: the belt must run BEFORE the operator predicate, since share_memory_ IS a
    # recognized (overridable) operator -- absent the belt it would be admitted.
    assert _indep_is_recognized_operator(_unwrap_capture_wrapper(torch.Tensor.share_memory_))
    assert not is_pure_forward_callable(torch.Tensor.share_memory_)
    assert is_pure_forward_callable(torch.Tensor.is_shared)


@pytest.mark.smoke
def test_secE_3_sobol_admitted_as_documented_residual() -> None:
    """``torch._sobol_engine_initialize_state_`` is a genuine aten op -> ADMITTED residual.

    Pinned as admitted so a future refactor cannot silently name-deny it (it is a
    documented residual, not a denied op).
    """

    obj = torch._sobol_engine_initialize_state_
    assert _indep_has_aten_schema("_sobol_engine_initialize_state_")
    assert is_pure_forward_callable(obj)


@pytest.mark.smoke
def test_to_sparse_coo_wrapper_rescue_admitted() -> None:
    """The narrow ``to_sparse_coo`` wrapper (neither overridable nor aten by name) admits."""

    real = _unwrap_capture_wrapper(torch.Tensor.to_sparse_coo)
    # It genuinely needs the wrapper rescue (not caught by the operator authorities).
    assert id(real) not in _OVERRIDABLE_IDS
    assert not _indep_has_aten_schema("to_sparse_coo")
    assert is_pure_forward_callable(torch.Tensor.to_sparse_coo)


# --------------------------------------------------------------------------- #
# Resolver-path enforcement: the secE ops are refused through the real resolver, in
# BOTH trust modes; the residual + legit ops resolve.
# --------------------------------------------------------------------------- #

_SECE_DENIED_REFS = [
    "torch:_enable_functionalization",
    "torch:_disable_functionalization",
    "torch:_functionalize_enable_reapply_views",
    "torch.Tensor:share_memory_",
]


@pytest.mark.smoke
@pytest.mark.parametrize("ref", _SECE_DENIED_REFS)
def test_secE_ops_denied_at_resolver_even_under_trust(ref: str) -> None:
    """The secE gate closures hold through the resolver call site, even under trust."""

    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(ref)
    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(ref, trust_custom_callables=True)


@pytest.mark.smoke
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


@pytest.mark.smoke
def test_prior_round_controls_still_denied() -> None:
    """All prior-round denials remain refused (no regression from the r43 inversion)."""

    for obj in (
        torch.load,
        torch.save,
        torch.from_file,
        torch.Tensor.apply_,
        torch.Tensor.map_,
        torch.Tensor.map2_,
        torch.vmap,
        torch.Tensor.register_hook,
        torch.Tensor.resize_,
        torch.Tensor.set_,
        torch.set_default_dtype,
        torch.manual_seed,
        torch.set_num_threads,
        torch.cond,
        torch.while_loop,
        torch.compile,
        torch.autocast_increment_nesting,
        torch.clear_autocast_cache,
        torch.nn.functional.handle_torch_function,
    ):
        assert not is_pure_forward_callable(obj)


@pytest.mark.smoke
def test_pure_forward_surface_still_resolves() -> None:
    """The r43 inversion does not over-deny the legitimate pure forward surface."""

    assert is_pure_forward_callable(torch.add)
    assert is_pure_forward_callable(torch.matmul)
    assert is_pure_forward_callable(torch.nn.functional.conv2d)
    assert is_pure_forward_callable(torch.Tensor.add_)
    assert is_pure_forward_callable(torch.Tensor.relu_)
    assert is_pure_forward_callable(torch.Tensor.as_strided)
    assert is_pure_forward_callable(torch.Tensor.narrow)
    # Pure reads / getters unaffected.
    assert is_pure_forward_callable(torch.get_rng_state)
    assert is_pure_forward_callable(torch.initial_seed)
    assert is_pure_forward_callable(torch.Tensor.is_shared)
