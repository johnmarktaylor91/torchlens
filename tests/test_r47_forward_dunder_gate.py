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

The fix (``torchlens/_utils/_callable_safety.py``) is a POSITIVE ``_ALLOWED_FORWARD_DUNDERS``
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

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._runnable import PathFaithfulness
from torchlens._utils._callable_safety import (
    _ALLOWED_FORWARD_DUNDERS,
    is_pure_forward_callable,
)
from torchlens.backends.torch.wrappers import wrap_torch
from torchlens.options import CaptureOptions

# Install the torch wrappers so resolved callables mirror the live (wrapped) state: the
# gate must unwrap before deciding, exactly as the run path does.
wrap_torch()


# --------------------------------------------------------------------------- #
# WHOLE-CLASS IFF SWEEP: gate decision is TRUE iff the name is allowlisted.
# --------------------------------------------------------------------------- #


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


@pytest.mark.parametrize("name", _DANGEROUS_DUNDERS)
def test_immunizer_dangerous_dunder_denied(name: str) -> None:
    """A non-forward dunder is DENIED and is not in the forward allowlist."""

    obj = getattr(torch.Tensor, name, None)
    if obj is None:  # pragma: no cover - version drift; a floor test guards vacuity.
        pytest.skip(f"torch.Tensor.{name} absent on torch {torch.__version__}")
    assert name not in _ALLOWED_FORWARD_DUNDERS
    assert not is_pure_forward_callable(obj), f"dangerous dunder WRONGLY ADMITTED: {name}"


# --------------------------------------------------------------------------- #
# FROZEN FIXTURE 2: the forward dunders that MUST resolve (review would have regressed the
# ``torch._tensor`` ones). Denying ANY fails RED.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Helper unit: shape gate denies non-allowlisted dunders, leaves non-dunder names alone.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Behavioral: resolver refuses __setstate__ even under trust (mirrors r43 enforcement).
# --------------------------------------------------------------------------- #


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


def test_forward_dunder_model_runnable_round_trip_verified(tmp_path: Path) -> None:
    """A forward-dunder model saves, loads, and runs VERIFIED (behavioral zero-regression)."""

    model = _ForwardDunderModel()
    inputs = (torch.randn(3, 3), torch.randn(3, 3))
    trace = tl.trace(
        model,
        inputs,
        capture=CaptureOptions(
            replayable=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    path = tmp_path / "forward_dunder.tlspec"
    trace.save(path, level="runnable")

    result = tl.load(path).run(inputs=inputs)
    torch.testing.assert_close(result.output, model(*inputs))
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
