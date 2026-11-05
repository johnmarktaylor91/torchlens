"""The raw-pointer constructor family is denied as a CLASS on every untrusted path.

Torch's legacy tensor constructors carry a hidden keyword-only ``cdata`` overload
that builds a live tensor over a RAW PROCESS ADDRESS (``torch.FloatTensor(cdata=0)``
/ ``torch.Tensor.new(cdata=0)`` segfault the interpreter on NULL; a non-NULL address
is an arbitrary-memory primitive). The families are regenerated per dtype across
``torch`` / ``torch.cuda`` / ``torch.sparse`` / ``torch.cuda.sparse``, so denying
them one name at a time never terminates.

Two structural rules close the class (``torchlens/_utils/_raw_pointer_denial.py``):
every torch tensor / storage CLASS is refused as a resolvable callable, and any
bundle-supplied keyword naming a pointer / address / foreign handle is refused at the
loader's parse boundary and again at the replay argument boundary. These tests pin
both, enumerate the family at RUNTIME so a missing deny fails as torch grows, and
pin the no-regression side: the genuinely captured ``Tensor.new(*sizes)`` allocator
and the ordinary forward ops keep resolving.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from _format3_envelope import mutate_runnable_descriptor

import torchlens as tl
from torchlens._io.runnable_load import DescriptorStructuralBoundError, _validate_call_arity
from torchlens._runnable import (
    LiteralArgumentRef,
    LiteralAtom,
    LiteralAtomKind,
    ReadinessStatus,
    RunnableErrorCode,
)
from torchlens._utils._callable_safety import is_pure_forward_callable
from torchlens._utils._raw_pointer_denial import (
    RAW_POINTER_PARAMETER_NAMES,
    is_raw_pointer_constructor_type,
    is_raw_pointer_parameter_name,
    raw_pointer_constructor_family,
)
from torchlens._utils._torch_symbols import (
    ALLOWED_EXACT_ROOTS,
    ENUMERATED_TORCH_NAMESPACES,
    getattr_allowlisted,
    torch_namespace,
)
from torchlens.errors import RunCapabilityUnavailableError, UntrustedCallableError
from torchlens.intervention._resolver_registry import resolve_function_registry_key
from torchlens.intervention.types import FunctionRegistryKey


class _Tiny(nn.Module):
    """One linear plus a softmax: the smallest replayable capture."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a softmax over the linear output."""

        return torch.softmax(self.fc(x), dim=1)


def _bundle(tmp_path: Path, name: str) -> Path:
    """Save one runnable bundle for tampering."""

    trace = tl.trace(
        _Tiny().eval(), torch.randn(2, 4), capture=tl.options.CaptureOptions(replayable=True)
    )
    path = tmp_path / name
    trace.save(str(path), level="runnable", include_weights=True)
    return path


# --------------------------------------------------------------------------- #
# (1) the constructor family, enumerated at runtime                           #
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_every_live_tensor_and_storage_class_is_refused() -> None:
    """RUNTIME enumeration: no torch tensor / storage class resolves from a bundle.

    The guard enumerates ``torch._tensor_classes`` / ``torch._storage_classes`` on the
    RUNNING torch, so a dtype or device sibling added by a future torch fails here
    unless the structural rule already denies it.
    """

    family = raw_pointer_constructor_family()
    assert len(family) >= 38, "torch's legacy tensor/storage registries look empty"
    admitted = [
        f"{cls.__module__}.{cls.__qualname__}" for cls in family if is_pure_forward_callable(cls)
    ]
    assert not admitted, f"raw-pointer constructor classes admitted by the gate: {admitted}"


@pytest.mark.smoke
def test_the_four_legacy_namespaces_are_each_denied() -> None:
    """One refusal per family: torch / torch.cuda / torch.sparse / torch.cuda.sparse."""

    for namespace in ("torch", "torch.cuda", "torch.sparse", "torch.cuda.sparse"):
        root: Any = torch
        for part in namespace.split(".")[1:]:
            root = getattr(root, part)
        cls = getattr(root, "FloatTensor")
        assert is_raw_pointer_constructor_type(cls), namespace
        assert not is_pure_forward_callable(cls), f"{namespace}.FloatTensor resolves"


def test_storage_classes_and_tensor_subclasses_are_denied() -> None:
    """The storage family and ``torch.Tensor`` subclasses are refused too."""

    for cls in (torch.Tensor, torch.nn.Parameter, torch.UntypedStorage, torch.cuda.FloatStorage):
        assert not is_pure_forward_callable(cls), f"{cls!r} resolves"


# --------------------------------------------------------------------------- #
# (2) no admitted callable on the untrusted roots takes a raw pointer         #
# --------------------------------------------------------------------------- #


def _reachable_callables() -> list[tuple[str, Any]]:
    """Return every callable a bundle key can name on the in-memory roots."""

    found: list[tuple[str, Any]] = []
    for namespace in (*ALLOWED_EXACT_ROOTS, *ENUMERATED_TORCH_NAMESPACES):
        root = ALLOWED_EXACT_ROOTS.get(namespace) or torch_namespace(namespace)
        if root is None:
            continue
        for name in dir(root):
            value = getattr_allowlisted(namespace, name)
            if value is not None:
                found.append((f"{namespace}.{name}", value))
    return found


@pytest.mark.smoke
def test_no_admitted_callable_accepts_a_raw_pointer_parameter() -> None:
    """RUNTIME enumeration of the loader's reachable roots: no pointer sink is admitted."""

    offenders: list[str] = []
    for path, value in _reachable_callables():
        if not is_pure_forward_callable(value):
            continue
        try:
            signature = inspect.signature(value)
        except (TypeError, ValueError):
            continue
        pointer_params = [
            name for name in signature.parameters if is_raw_pointer_parameter_name(name)
        ]
        if pointer_params:
            offenders.append(f"{path}{pointer_params}")
    assert not offenders, f"admitted callables taking a raw pointer: {offenders}"


def test_pointer_name_rule_covers_the_suffix_shapes() -> None:
    """A future sibling spelling is denied by shape, not by enumeration."""

    for name in (
        "cdata",
        "_cdata",
        "stream_ptr",
        "some_future_ptr",
        "event_handle",
        "base_address",
    ):
        assert is_raw_pointer_parameter_name(name), name
    for name in ("dim", "keepdim", "input", "weight", "start", "handle_count"):
        assert not is_raw_pointer_parameter_name(name), name
    assert "cdata" in RAW_POINTER_PARAMETER_NAMES


# --------------------------------------------------------------------------- #
# (3) a crafted bundle is refused BEFORE anything is called                   #
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_parse_refuses_a_pointer_keyword_before_resolution() -> None:
    """``_validate_call_arity`` refuses a ``cdata`` keyword leaf at parse."""

    literal = LiteralArgumentRef(
        argument_path=("kwargs", "cdata"),
        value=LiteralAtom(kind=LiteralAtomKind.INT, value=140737488355328),
    )
    with pytest.raises(DescriptorStructuralBoundError) as refusal:
        _validate_call_arity("call-0", 0, 1, (), (literal,))
    assert refusal.value.code == RunnableErrorCode.CALL_ARITY_MISMATCH.value
    assert "cdata" in str(refusal.value)
    # The same shape with an ordinary keyword still parses.
    ordinary = LiteralArgumentRef(
        argument_path=("kwargs", "dim"), value=LiteralAtom(kind=LiteralAtomKind.INT, value=1)
    )
    _validate_call_arity("call-0", 0, 1, (), (ordinary,))


@pytest.mark.smoke
def test_crafted_bundle_with_a_cdata_keyword_refuses_at_load(tmp_path: Path) -> None:
    """End to end: a tampered bundle naming ``cdata`` never reaches a call.

    The descriptor is rewritten so one captured call carries a keyword-only
    ``cdata`` literal holding a raw address. The load refuses at
    ``descriptor_parse`` and ``run_transaction`` refuses typed, so torch is never
    handed the address (an unrefused call would dereference it).
    """

    bundle = _bundle(tmp_path, "cdata.tlspec")

    def _craft(descriptor: dict[str, Any]) -> None:
        """Rewrite the first call's keyword leaf to a raw-pointer keyword."""

        call = descriptor["calls"][0]
        call["literal_arguments"].append(
            {"argument_path": ["kwargs", "cdata"], "value": {"kind": "int", "value": 0}}
        )
        call["num_keyword_args"] = len(
            {
                tuple(argument["argument_path"])
                for argument in call["literal_arguments"]
                if argument["argument_path"][0] == "kwargs"
            }
        )

    mutate_runnable_descriptor(bundle, _craft)
    loaded = tl.load(str(bundle))
    readiness = loaded._runnable.readiness
    assert readiness.status is ReadinessStatus.UNAVAILABLE
    assert RunnableErrorCode.CALL_ARITY_MISMATCH in {d.code for d in readiness.diagnostics}
    assert "descriptor_parse" in {d.detection_stage for d in readiness.diagnostics}
    assert any("cdata" in (d.message or "") for d in readiness.diagnostics), (
        "the refusal must name the offending keyword"
    )
    with pytest.raises(RunCapabilityUnavailableError):
        loaded.run_transaction(inputs=torch.randn(2, 4))


def test_execution_boundary_refuses_a_pointer_keyword() -> None:
    """Defense in depth: an in-memory descriptor cannot write a pointer keyword either."""

    from torchlens._runnable.call_arguments import _write_argument
    from torchlens.errors import RunPreconditionError

    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    with pytest.raises(RunPreconditionError) as refusal:
        _write_argument(args, kwargs, ("kwargs", "cdata"), 140737488355328)
    assert "cdata" in str(refusal.value)
    assert kwargs == {}
    _write_argument(args, kwargs, ("kwargs", "dim"), 1)
    assert kwargs == {"dim": 1}


# --------------------------------------------------------------------------- #
# (4) the resolvers refuse the family, even under trust                       #
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
@pytest.mark.parametrize(
    "import_path",
    [
        "torch.cuda:FloatTensor",
        "torch.sparse:FloatTensor",
        "torch.cuda.sparse:FloatTensor",
        "torch:FloatTensor",
    ],
)
def test_intervention_resolver_refuses_legacy_constructors_under_trust(import_path: str) -> None:
    """A trusted bundle cannot resolve a legacy constructor: trust never grants a pointer sink."""

    key = FunctionRegistryKey(
        "custom", import_path.split(":")[1], "function", import_path=import_path
    )
    with pytest.raises(UntrustedCallableError):
        resolve_function_registry_key(key, trust_custom_callables=True)


def test_pointer_adopting_free_functions_are_refused() -> None:
    """Pointer-adopting free functions are denied by signature shape where they exist."""

    candidates = [
        ("torch.xpu", "get_stream_from_external"),
        ("torch.cuda.streams", "ExternalStream"),
        ("torch.cuda.memory", "caching_allocator_delete"),
    ]
    checked = 0
    for module_name, name in candidates:
        module: Any = torch
        for part in module_name.split(".")[1:]:
            module = getattr(module, part, None)
            if module is None:
                break
        candidate = getattr(module, name, None) if module is not None else None
        if candidate is None:
            continue
        checked += 1
        assert not is_pure_forward_callable(candidate), f"{module_name}.{name} resolves"
    assert checked, "no pointer-adopting free function was available to check"


# --------------------------------------------------------------------------- #
# (5) no regression: the legitimate spellings keep resolving                  #
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_legitimate_forward_ops_still_resolve() -> None:
    """The captured allocator ``Tensor.new`` and ordinary forward ops are unaffected.

    ``Tensor.new(*sizes)`` is a decorated, genuinely captured op (an ``IGNORED_FUNCS``
    roster row with uninitialized-memory semantics tracked in ``_utils/_uninit_alloc``),
    so only its keyword-only ``cdata`` spelling is refused -- at the argument boundary,
    never by denying the callable.
    """

    for func in (
        torch.Tensor.new,
        torch.relu,
        torch.nn.functional.linear,
        torch.Tensor.add,
        torch.softmax,
    ):
        assert is_pure_forward_callable(func), f"{func!r} was denied"


def test_untampered_bundle_still_runs(tmp_path: Path) -> None:
    """Over-trigger guard: an honest bundle loads and runs unchanged."""

    loaded = tl.load(str(_bundle(tmp_path, "clean.tlspec")))
    assert loaded._runnable.readiness.status is not ReadinessStatus.UNAVAILABLE
    loaded.run_transaction(inputs=torch.randn(2, 4))
