"""Round-47 security regression (finding secE_1): a POSITIVE forward-dunder allowlist on
the run-path callable-reattach gate, closing the ``torch.Tensor.__setstate__`` storage-
REBIND class (and its non-forward dunder siblings) the r43/r45 belts miss.

secE_1: ``torch.Tensor.__setstate__`` passed ``is_pure_forward_callable`` under default
untrust. It is NOT a forward op -- it is the pickle-protocol state restorer, whose legacy
tuple form ``(storage, offset, size, stride)`` REBINDS the target tensor's storage onto an
attacker donor with a fabricated size/stride, reaching the SAME uninitialized / out-of-bounds
heap-read the storage belt DENIES ``set_`` / ``resize_`` / ``share_memory_`` for. It slipped
every belt: its leading-``__`` carries no ``set_`` prefix / ``resize`` / ``share_memory``
token, and it IS torch-OVERRIDABLE, so the r43 recognized-operator predicate ADMITTED it on
identity -- exactly the gap ``share_memory_`` occupied before r43 pinned it.

A blanket dunder-DENY is WRONG: an exhaustive live sweep found genuine FORWARD dunders on
the run path -- the 14 Python-level forward dunders on the gated root ``torch._tensor``
(``__pow__`` / ``__floordiv__`` / reflected ops / ``__ipow__`` / ``__len__`` /
``__contains__``) plus the arithmetic ``__add__`` / ``__mul__`` / ``__matmul__`` /
``__getitem__`` and their in-place / comparison / bitwise siblings on the module-less
descriptor path. Denying those would break replay (the LOCKED zero-forward-regression +
validation-tripwire rules).

The fix (``torchlens/utils/_callable_safety.py``) is a POSITIVE ``_ALLOWED_FORWARD_DUNDERS``
allowlist checked by a DEDICATED helper (``_is_denied_forward_dunder_name``) wired into
``is_pure_forward_callable`` BEFORE every module branch -- so it covers BOTH the module-less
``_is_tensor_method_descriptor`` path AND the gated-root ``_is_recognized_operator`` path.
The helper is deliberately NOT folded into the shared ``_is_side_effecting_callable_name``
belt, whose second caller (``is_inert_first_party_callable``) gates ``torchlens.*`` facet
recipes.

THIS IS THE MACHINE-CHECKED NEXT-SIBLING IMMUNIZER. It enumerates EVERY callable ``__x__``
on ``torch.Tensor`` (and the fixed roots ``torch`` / ``torch._C`` / ``torch._tensor``) and
asserts the gate decision is TRUE iff the name is in the forward allowlist. It goes RED in
BOTH directions:

* a future DANGEROUS dunder that passes the gate but is NOT in ``_ALLOWED_FORWARD_DUNDERS``
  (a new ``__setstate__``-class sibling), and
* a FORWARD regression -- an allowlisted forward dunder the gate wrongly denies.

Plus explicit pins: ``__setstate__`` + its 9 dangerous siblings DENIED (and proven to slip
the name/verb belt, so the dunder gate is load-bearing), the forward dunders ADMITTED, the
resolver refuses ``__setstate__`` even under trust, and a forward-dunder model still saves /
loads / runs VERIFIED end-to-end (behavioral zero-forward-regression proof).
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch.wrappers import wrap_torch
from torchlens.intervention.errors import UntrustedCallableError
from torchlens.intervention.resolver import resolve_import_ref
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness
from torchlens.utils._callable_safety import (
    _ALLOWED_FORWARD_DUNDERS,
    _is_denied_forward_dunder_name,
    _is_side_effecting_callable_name,
    _terminal_callable_name,
    _unwrap_capture_wrapper,  # mechanical capture-unwrap only; NOT the gate decision
    is_pure_forward_callable,
)

# Install the torch wrappers so resolved callables mirror the live (wrapped) state: the
# gate must unwrap before deciding, exactly as the run path does.
wrap_torch()


_FIXED_ROOTS: dict[str, Any] = {
    "torch.Tensor": torch.Tensor,
    "torch": torch,
    "torch._C": torch._C,
    "torch._tensor": importlib.import_module("torch._tensor"),
}


def _iter_callable_dunders() -> list[tuple[str, str, Callable[..., Any]]]:
    """Yield ``(root_label, name, obj)`` for every callable ``__x__`` on the fixed roots."""

    out: list[tuple[str, str, Callable[..., Any]]] = []
    for label, root in _FIXED_ROOTS.items():
        for name in dir(root):
            if not (name.startswith("__") and name.endswith("__")):
                continue
            try:
                obj = getattr(root, name)
            except Exception:  # pragma: no cover - some torch attrs raise on access.
                continue
            if callable(obj):
                out.append((label, name, obj))
    return out


_CALLABLE_DUNDERS = _iter_callable_dunders()


# --------------------------------------------------------------------------- #
# WHOLE-CLASS IFF SWEEP: gate decision is TRUE iff the name is allowlisted.
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_immunizer_dunder_gate_iff_allowlist() -> None:
    """Every callable dunder on the fixed roots: admitted IFF in ``_ALLOWED_FORWARD_DUNDERS``.

    RED in BOTH directions: a gate-passer not in the allowlist (a future ``__setstate__``
    sibling) OR an allowlisted forward dunder wrongly denied (a forward regression).
    """

    # Non-vacuous: torch.Tensor alone exposes dozens of callable dunders.
    assert len(_CALLABLE_DUNDERS) > 40, (
        f"suspiciously few callable dunders: {len(_CALLABLE_DUNDERS)}"
    )
    passer_not_allowlisted: list[str] = []
    allowlisted_denied: list[str] = []
    for label, name, obj in _CALLABLE_DUNDERS:
        admitted = is_pure_forward_callable(obj)
        allowlisted = name in _ALLOWED_FORWARD_DUNDERS
        if admitted and not allowlisted:
            passer_not_allowlisted.append(f"{label}:{name}")
        if allowlisted and not admitted:
            allowlisted_denied.append(f"{label}:{name}")
    assert not passer_not_allowlisted, (
        "dunder gate-passers NOT in the forward allowlist (possible dangerous sibling): "
        f"{sorted(set(passer_not_allowlisted))}"
    )
    assert not allowlisted_denied, (
        f"allowlisted forward dunders WRONGLY DENIED (forward regression): "
        f"{sorted(set(allowlisted_denied))}"
    )


@pytest.mark.smoke
def test_immunizer_sweep_is_non_vacuous_on_tensor() -> None:
    """torch.Tensor exposes BOTH allowlisted forward dunders AND denied non-forward dunders.

    Guards against a vacuous sweep (e.g. an enumeration that stopped finding dunders).
    """

    tensor_dunders = {name for label, name, _ in _CALLABLE_DUNDERS if label == "torch.Tensor"}
    forward_present = tensor_dunders & _ALLOWED_FORWARD_DUNDERS
    nonforward_present = tensor_dunders - _ALLOWED_FORWARD_DUNDERS
    assert len(forward_present) >= 40, sorted(forward_present)
    assert len(nonforward_present) >= 10, sorted(nonforward_present)


# --------------------------------------------------------------------------- #
# FROZEN FIXTURE 1: the DANGEROUS non-forward dunder set (secE_1 + 9 siblings).
# Each must be DENIED, must slip the name/verb belt (so the dunder gate is load-bearing),
# and must NOT be in the allowlist.
# --------------------------------------------------------------------------- #

_DANGEROUS_DUNDERS: tuple[str, ...] = (
    "__setstate__",  # secE_1: storage REBIND onto attacker donor (OOB heap read).
    "__reduce_ex__",  # pickle reducer.
    "__reduce__",  # pickle reducer.
    "__array__",  # numpy array export.
    "__array_wrap__",  # numpy array protocol.
    "__deepcopy__",  # copy protocol.
    "__dlpack__",  # zero-copy export.
    "__dlpack_device__",  # zero-copy export.
    "__format__",  # stringification.
    "__repr__",  # stringification.
    "__reversed__",  # container reversal.
)


@pytest.mark.smoke
@pytest.mark.parametrize("name", _DANGEROUS_DUNDERS)
def test_immunizer_dangerous_dunder_denied(name: str) -> None:
    """A non-forward dunder is DENIED and is not in the forward allowlist."""

    obj = getattr(torch.Tensor, name, None)
    if obj is None:  # pragma: no cover - version drift; a floor test guards vacuity.
        pytest.skip(f"torch.Tensor.{name} absent on torch {torch.__version__}")
    assert name not in _ALLOWED_FORWARD_DUNDERS
    assert not is_pure_forward_callable(obj), f"dangerous dunder WRONGLY ADMITTED: {name}"


@pytest.mark.smoke
def test_immunizer_setstate_and_siblings_are_dunder_gate_load_bearing() -> None:
    """The dangerous dunders slip the name/verb belt; ONLY the r47 dunder gate closes them.

    Mirrors the finding: ``__setstate__`` "slips every belt and is admitted on overridable
    identity". If a future refactor dropped the dunder gate expecting the storage belt to
    cover these, this goes RED (belt does NOT catch them).
    """

    for name in _DANGEROUS_DUNDERS:
        obj = getattr(torch.Tensor, name, None)
        if obj is None:  # pragma: no cover - version drift.
            continue
        real = _unwrap_capture_wrapper(obj)
        # The name/verb/storage belt does NOT catch these (leading ``__``, no set_/resize/
        # share_memory token) -- so the dunder gate is the load-bearing closure.
        assert not _is_side_effecting_callable_name(real), (
            f"{name} unexpectedly caught by the name/verb belt -- re-audit load-bearing claim"
        )
        assert _is_denied_forward_dunder_name(_terminal_callable_name(real)), name
        assert not is_pure_forward_callable(obj), name


# --------------------------------------------------------------------------- #
# FROZEN FIXTURE 2: the forward dunders that MUST resolve (Sol would have regressed the
# ``torch._tensor`` ones). Denying ANY fails RED.
# --------------------------------------------------------------------------- #

_FORWARD_DUNDERS_ADMITTED: tuple[str, ...] = (
    "__add__",
    "__mul__",
    "__matmul__",
    "__getitem__",
    "__pow__",
    "__len__",  # brief-pinned; a live forward dunder on torch._tensor.
    "__contains__",  # a live forward dunder on torch._tensor.
    "__rmatmul__",  # reflected op on torch._tensor.
    "__floordiv__",  # a live forward dunder on torch._tensor.
    "__rsub__",  # reflected op on torch._tensor.
    "__setitem__",
    "__eq__",
    "__neg__",
    "__ipow__",  # in-place op on torch._tensor.
)


@pytest.mark.smoke
@pytest.mark.parametrize("name", _FORWARD_DUNDERS_ADMITTED)
def test_immunizer_forward_dunder_admitted(name: str) -> None:
    """A genuine forward-operator dunder resolves through the gate (zero forward regression)."""

    obj = getattr(torch.Tensor, name, None)
    if obj is None:  # pragma: no cover - version drift; a floor test guards vacuity.
        pytest.skip(f"torch.Tensor.{name} absent on torch {torch.__version__}")
    assert name in _ALLOWED_FORWARD_DUNDERS
    assert is_pure_forward_callable(obj), f"forward dunder WRONGLY DENIED: {name}"


# --------------------------------------------------------------------------- #
# Helper unit: shape gate denies non-allowlisted dunders, leaves non-dunder names alone.
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_denied_forward_dunder_name_helper() -> None:
    """``_is_denied_forward_dunder_name`` denies non-allowlisted dunders, spares the rest."""

    assert _is_denied_forward_dunder_name("__setstate__")
    assert _is_denied_forward_dunder_name("__reduce_ex__")
    assert _is_denied_forward_dunder_name("__init_subclass__")
    assert not _is_denied_forward_dunder_name("__add__")
    assert not _is_denied_forward_dunder_name("__getitem__")
    assert not _is_denied_forward_dunder_name("__len__")
    # Non-dunder names are untouched -- the operator / torch-function surface is unaffected.
    assert not _is_denied_forward_dunder_name("add")
    assert not _is_denied_forward_dunder_name("relu_")
    assert not _is_denied_forward_dunder_name("matmul")
    assert not _is_denied_forward_dunder_name("_private")


# --------------------------------------------------------------------------- #
# Behavioral: resolver refuses __setstate__ even under trust (mirrors r43 enforcement).
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
@pytest.mark.parametrize("name", ("__setstate__", "__reduce_ex__", "__array__", "__deepcopy__"))
def test_setstate_class_denied_at_resolver_even_under_trust(name: str) -> None:
    """The dangerous dunders are refused through the real resolver, in BOTH trust modes."""

    ref = f"torch.Tensor:{name}"
    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(ref)
    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(ref, trust_custom_callables=True)


# --------------------------------------------------------------------------- #
# Behavioral: a forward-dunder model still saves / loads / runs VERIFIED end-to-end.
# --------------------------------------------------------------------------- #


class _ForwardDunderModel(nn.Module):
    """A graph exercising the run-path forward dunders (add / mul / matmul / getitem / pow)."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute through ``+`` / ``*`` / ``@`` / ``[...]`` / ``**`` / ``-``."""

        a = x + y  # __add__
        a = a * 2  # __mul__
        a = a @ y.T  # __matmul__
        a = a[0]  # __getitem__
        a = a**2  # __pow__
        a = a - x[0]  # __sub__ / __getitem__
        return a


@pytest.mark.smoke
def test_forward_dunder_model_runnable_round_trip_verified(tmp_path: Path) -> None:
    """A forward-dunder model saves, loads, and runs VERIFIED (behavioral zero-regression)."""

    model = _ForwardDunderModel()
    inputs = (torch.randn(3, 3), torch.randn(3, 3))
    trace = tl.trace(
        model,
        inputs,
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    path = tmp_path / "forward_dunder.tlspec"
    trace.save(path, level="runnable")

    result = tl.load(path).run(inputs=inputs)
    torch.testing.assert_close(result.output, model(*inputs))
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
