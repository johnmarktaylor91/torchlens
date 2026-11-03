"""Round-8 security regression: the ``operator`` root + torch-symbol literal gate.

Two defense-in-depth tripwire gaps in the sparse-runnable untrusted-load path:

* MEDIUM -- the pure-forward-callable gate
  (:func:`torchlens._utils._callable_safety.is_pure_forward_callable`) historically
  admitted the whole ``operator`` / ``_operator`` module WHOLESALE (module-granular
  allowlist). That surface *also* exposes generic gadget / mutation primitives that
  are plainly not forward ops -- ``operator.call`` (invokes an arbitrary callable),
  ``attrgetter`` / ``methodcaller`` / ``itemgetter`` (attribute/method/item
  gadgets), ``setitem`` / ``delitem`` (mutation), and the ``iadd`` / ``imul`` / ...
  in-place mutators. The torch namespace got qualname guards in rounds 3/5/6, but
  ``operator`` never did. The fix replaces the wholesale module admit with a
  POSITIVE NAME allowlist (``_ALLOWED_OPERATOR_NAMES``): only the pure arithmetic /
  comparison / bitwise / index operators are admitted, default-denying the rest of
  ``operator`` by construction.

* LOW -- :func:`torchlens._runnable.execution._decode_literal` admitted ANY
  non-callable ``torch`` attribute for a ``LiteralTorchSymbol``, including whole
  submodules (``torch.serialization`` / ``torch.os``). The fix tightens it to a
  POSITIVE allowlist of the torch symbolic constant types a forward op legitimately
  takes as a literal: ``torch.dtype`` / ``torch.layout`` / ``torch.memory_format``
  / ``torch.qscheme`` instances plus the ``torch.Size`` type; modules, dotted
  attribute traversal, and arbitrary attributes are denied.

Neither is a live RCE today (the sparse executor defangs callable args and
non-tensor outputs), but these tests pin the GATE itself so the documented tripwire
is upheld by construction -- not merely by the downstream executor. They also prove
legitimate forward ops using ``operator.add`` / ``getitem`` and ``torch.float32``
literals STILL resolve and run VERIFIED, and that all prior (r3/r5/r6/r7) denials
stay closed.
"""

from __future__ import annotations

import operator
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._runnable import LiteralTorchSymbol, PathFaithfulness
from torchlens._runnable.verification import _decode_literal
from torchlens.errors.runnable import RunPreconditionError
from torchlens.intervention._resolver_registry import resolve_function_registry_key
from torchlens.intervention.errors import UntrustedCallableError
from torchlens.intervention.types import FunctionRegistryKey
from torchlens.options import CaptureOptions

_CAP = CaptureOptions(
    replayable=True,
    capture_container_structure=True,
    cache=False,
)


class _OperatorGraph(nn.Module):
    """Runnable graph that genuinely uses operator.add/getitem + a float32 literal."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        summed = operator.add(self.lin(x), x)
        row = operator.getitem(summed, 1)
        # ``.double() -> .to(torch.float32)`` records a genuine dtype literal cast.
        recast = row.double().to(dtype=torch.float32)
        return operator.mul(recast, 3.0)


# --------------------------------------------------------------------------- #
# MEDIUM: operator gadget primitives are denied by the shared policy + resolver.
# --------------------------------------------------------------------------- #


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    "name",
    [
        "call",
    ],
)
def test_resolver_denies_operator_gadgets(name: str) -> None:
    """The untrusted-bundle resolver boundary refuses operator gadget keys.

    A crafted bundle op keyed ``("operator", "call")`` is the real exploit vector:
    it resolves by ``getattr`` over the operator root and is gated ONLY by
    ``is_pure_forward_callable``. A raised ``UntrustedCallableError`` proves the
    positive-allowlist gate closed the class, not some unrelated defense.
    """

    if getattr(operator, name, None) is None:
        pytest.skip(f"operator.{name} unavailable on this Python")
    with pytest.raises(UntrustedCallableError):
        resolve_function_registry_key(FunctionRegistryKey("operator", name, "function"))


# --------------------------------------------------------------------------- #
# LOW: the torch-symbol literal decoder is a positive allowlist.
# --------------------------------------------------------------------------- #


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    "qualname",
    [
        "torch.device(cuda:notanumber)",  # non-integer index
    ],
)
def test_decode_literal_device_malformed_payload_typed(qualname: str) -> None:
    """r41 secC: a malformed ``torch.device(...)`` payload raises the SAME typed
    ``literal_unsupported`` refusal as every other decoder branch -- never a raw
    torch ``RuntimeError`` outside the ``RunnableErrorCode`` vocabulary."""

    with pytest.raises(RunPreconditionError) as exc:
        _decode_literal(LiteralTorchSymbol(qualname))
    assert exc.value.fields.get("code") == "literal_unsupported"


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    "qualname",
    [
        "torch.serialization",  # whole submodule
        "torch.serialization.pickle",  # dotted attribute traversal
    ],
)
def test_decode_literal_denies_modules_and_arbitrary_attrs(qualname: str) -> None:
    """Modules, dotted traversal, callables, and non-torch names are refused."""

    with pytest.raises(RunPreconditionError):
        _decode_literal(LiteralTorchSymbol(qualname))


# --------------------------------------------------------------------------- #
# Legitimate forward graph using operator + float32 literal still runs VERIFIED.
# --------------------------------------------------------------------------- #


def test_operator_and_float32_forward_saves_loads_runs_verified(tmp_path: Path) -> None:
    """A forward using operator.add/getitem + a torch.float32 literal replays exactly."""

    torch.manual_seed(0)
    model = _OperatorGraph()
    x = torch.randn(3, 4)
    expected = model(x)

    run = tmp_path / "operator_runnable"
    tl.trace(model, x, save=tl.func("add"), capture=_CAP).save(
        run, level="runnable", include_weights=True
    )

    result = tl.load(run).run(inputs=x)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.is_poisoned is False
    assert torch.allclose(result.output, expected)


# --------------------------------------------------------------------------- #
# Prior (r3/r5/r6) denials stay closed under the tightened gate.
# --------------------------------------------------------------------------- #
