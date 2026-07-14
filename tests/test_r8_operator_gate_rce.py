"""Round-8 security regression: the ``operator`` root + torch-symbol literal gate.

Two defense-in-depth tripwire gaps in the sparse-runnable untrusted-load path:

* MEDIUM -- the pure-forward-callable gate
  (:func:`torchlens.utils._callable_safety.is_pure_forward_callable`) historically
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

* LOW -- :func:`torchlens._runnable_execution._decode_literal` admitted ANY
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
from torchlens._runnable_execution import _decode_literal
from torchlens.errors.runnable import RunPreconditionError
from torchlens.intervention.errors import UntrustedCallableError
from torchlens.intervention.resolver import resolve_function_registry_key
from torchlens.intervention.types import FunctionRegistryKey
from torchlens.options import CaptureOptions
from torchlens.runnable import LiteralTorchSymbol, PathFaithfulness
from torchlens.utils._callable_safety import is_pure_forward_callable

_CAP = CaptureOptions(
    intervention_ready=True,
    capture_container_structure=True,
    cache=False,
)

# Generic gadget / mutation primitives that MUST be denied. Their real module is
# ``operator`` / ``_operator`` (on the historical wholesale allowlist), so only the
# new positive name allowlist keeps them out.
_GADGET_NAMES: tuple[str, ...] = (
    "call",
    "attrgetter",
    "methodcaller",
    "itemgetter",
    "setitem",
    "delitem",
    "iadd",
    "isub",
    "imul",
    "iconcat",
    "ipow",
    "ilshift",
    "irshift",
    "iand",
    "ior",
    "ixor",
)

# Pure operators that legitimately appear in a captured forward graph and MUST
# stay resolvable (mirrors ``_ALLOWED_OPERATOR_NAMES``).
_PURE_OPERATOR_NAMES: tuple[str, ...] = (
    "add",
    "sub",
    "mul",
    "truediv",
    "floordiv",
    "mod",
    "pow",
    "neg",
    "pos",
    "abs",
    "matmul",
    "and_",
    "or_",
    "xor",
    "invert",
    "lshift",
    "rshift",
    "lt",
    "le",
    "eq",
    "ne",
    "gt",
    "ge",
    "getitem",
    "index",
    "concat",
    "contains",
    "not_",
    "is_",
    "is_not",
    "length_hint",
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


def test_operator_gadget_primitives_denied_by_policy() -> None:
    """Every generic gadget / mutation ``operator`` primitive is refused."""

    for name in _GADGET_NAMES:
        func = getattr(operator, name, None)
        if func is None:  # e.g. ``operator.call`` predates 3.11; skip if absent.
            continue
        assert is_pure_forward_callable(func) is False, name


def test_pure_operators_admitted_by_policy() -> None:
    """Every pure arithmetic/comparison/bitwise/index operator stays admitted."""

    for name in _PURE_OPERATOR_NAMES:
        assert is_pure_forward_callable(getattr(operator, name)) is True, name


@pytest.mark.smoke
@pytest.mark.parametrize(
    "name",
    ["call", "attrgetter", "methodcaller", "itemgetter", "setitem", "delitem", "iadd"],
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


@pytest.mark.parametrize("name", ["add", "sub", "mul", "getitem", "gt", "concat", "index", "not_"])
def test_resolver_admits_pure_operators(name: str) -> None:
    """The SAME operator-root path still resolves pure forward operators."""

    resolved = resolve_function_registry_key(FunctionRegistryKey("operator", name, "function"))
    assert resolved is getattr(operator, name)


def test_operator_module_not_wholesale_admitted() -> None:
    """The operator root is no longer a wholesale module admit.

    A wholesale admit would pass ANY operator callable; the gadget set proves the
    module is now default-deny with a small positive name allowlist.
    """

    admitted = {
        n
        for n in _GADGET_NAMES
        if getattr(operator, n, None) is not None and is_pure_forward_callable(getattr(operator, n))
    }
    assert admitted == set(), admitted


# --------------------------------------------------------------------------- #
# LOW: the torch-symbol literal decoder is a positive allowlist.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "qualname",
    [
        "torch.float32",
        "torch.int64",
        "torch.float64",
        "torch.strided",
        "torch.sparse_coo",
        "torch.contiguous_format",
        "torch.channels_last",
        "torch.per_tensor_affine",
        "torch.Size",
    ],
)
def test_decode_literal_admits_torch_symbol_constants(qualname: str) -> None:
    """dtype / layout / memory_format / qscheme instances + Size still decode."""

    decoded = _decode_literal(LiteralTorchSymbol(qualname))
    assert decoded is getattr(torch, qualname.removeprefix("torch."))


def test_decode_literal_admits_torch_device() -> None:
    """The ``torch.device(...)`` round-trip is preserved."""

    assert _decode_literal(LiteralTorchSymbol("torch.device(cpu)")) == torch.device("cpu")


@pytest.mark.parametrize(
    "qualname",
    [
        "torch.serialization",  # whole submodule
        "torch.os",  # re-exported stdlib module
        "torch.nn",  # subpackage
        "torch.load",  # callable
        "torch.save",  # callable
        "torch.from_file",  # callable factory
        "torch.serialization.pickle",  # dotted attribute traversal
        "torch.__dict__",  # dunder attribute
        "os.system",  # non-torch qualname
    ],
)
def test_decode_literal_denies_modules_and_arbitrary_attrs(qualname: str) -> None:
    """Modules, dotted traversal, callables, and non-torch names are refused."""

    with pytest.raises(RunPreconditionError):
        _decode_literal(LiteralTorchSymbol(qualname))


# --------------------------------------------------------------------------- #
# Legitimate forward graph using operator + float32 literal still runs VERIFIED.
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
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
    assert result.report.poisoned is False
    assert torch.allclose(result.output, expected)


# --------------------------------------------------------------------------- #
# Prior (r3/r5/r6) denials stay closed under the tightened gate.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "key",
    [
        FunctionRegistryKey("torch", "load", "function"),
        FunctionRegistryKey("torch", "save", "function"),
        FunctionRegistryKey("torch", "from_file", "function"),
        FunctionRegistryKey("torch", "set_default_dtype", "function"),
        FunctionRegistryKey("torch", "manual_seed", "function"),
        FunctionRegistryKey("torch.Tensor", "resize_", "method"),
    ],
    ids=["load", "save", "from_file", "set_default_dtype", "manual_seed", "resize_"],
)
def test_prior_torch_denials_stay_closed(key: FunctionRegistryKey) -> None:
    """The operator + literal tightening leaves all prior torch denials intact."""

    with pytest.raises(UntrustedCallableError):
        resolve_function_registry_key(key)
