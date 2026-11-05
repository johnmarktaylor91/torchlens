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

import importlib
import inspect
import sys
import warnings
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
from _format3_envelope import mutate_runnable_descriptor

import torchlens as tl
from torchlens._constants import ORIG_TORCH_FUNCS
from torchlens._io import ArtifactError
from torchlens._io.runnable_load import DescriptorStructuralBoundError, _validate_call_arity
from torchlens._runnable import (
    LiteralArgumentRef,
    LiteralAtom,
    LiteralAtomKind,
    ReadinessStatus,
    RunnableErrorCode,
)
from torchlens._utils._callable_safety import (
    _ALLOWED_FORWARD_OP_MODULES,
    _OPERATOR_MODULES,
    _unwrap_capture_wrapper,
    is_pure_forward_callable,
)
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
from torchlens.errors import (
    RunCapabilityUnavailableError,
    RunPreconditionError,
    UntrustedCallableError,
)
from torchlens.intervention._resolver_registry import resolve_function_registry_key
from torchlens.intervention._types import FunctionRegistryKey
from torchlens.validation._autograd_grad_replay import execute_replay_func
from torchlens.validation._influence_probes import _ProbeCall, _substitutes_move_output
from torchlens.validation._loaded_callables import resolve_loaded_callables


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


def _pointer_verdicts(found: list[tuple[str, Any]]) -> tuple[list[str], list[str], int]:
    """Classify enumerated callables, failing CLOSED on the signature-less ones.

    Returns admitted pointer-takers, admitted signature-less callables whose real
    module is not EXACTLY listed (a C builtin the signature rule cannot read), and
    the count of pointer-takers the gate refused (proof the guard is not vacuous).
    """

    listed = _ALLOWED_FORWARD_OP_MODULES | _OPERATOR_MODULES
    offenders: list[str] = []
    unlisted: list[str] = []
    refused = 0
    for path, value in found:
        admitted = is_pure_forward_callable(value)
        real = _unwrap_capture_wrapper(value)
        try:
            parameters: Any = inspect.signature(real).parameters
        except (TypeError, ValueError):
            module = str(getattr(real, "__module__", "") or "")
            if admitted and module and module not in listed:
                unlisted.append(f"{path} ({module})")
            continue
        pointer_params = [name for name in parameters if is_raw_pointer_parameter_name(name)]
        if pointer_params and admitted:
            offenders.append(f"{path}{pointer_params}")
        refused += bool(pointer_params and not admitted)
    return offenders, unlisted, refused


@pytest.mark.smoke
def test_no_admitted_callable_accepts_a_raw_pointer_parameter() -> None:
    """RUNTIME enumeration of the loader's reachable roots: no pointer sink is admitted.

    Fails CLOSED: an admitted callable with no inspectable signature must live in an
    EXACTLY listed module, so a new pointer-taking C builtin cannot pass unseen.
    """

    offenders, unlisted, _refused = _pointer_verdicts(_reachable_callables())
    assert not offenders, f"admitted callables taking a raw pointer: {offenders}"
    assert not unlisted, f"admitted signature-less callables in unlisted modules: {unlisted}"


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


# --------------------------------------------------------------------------- #
# (6) replay validation of a loaded bundle (the second execution boundary)    #
# --------------------------------------------------------------------------- #


class _Relu(nn.Module):
    """A linear and a relu: the relu op is rewritten into ``Tensor.new``."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return relu of the linear output plus a residual."""

        return torch.relu(self.fc(x)) + x * 2.0


def _replay_capture() -> tuple[Any, torch.Tensor]:
    """Capture with re-execution payloads; return the trace and the reference output."""

    torch.manual_seed(0)
    model = _Relu().eval()
    x = torch.randn(3, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_args=True, save_rng=True))
    return trace, model(x)


def _crafted_new_bundle(tmp_path: Path, kwargs: dict[str, Any]) -> tuple[Path, torch.Tensor]:
    """Save a replay-level bundle whose relu op claims to be ``Tensor.new(**kwargs)``.

    Both keyword records are rewritten, as a crafted file can: ``saved_kwargs`` (the
    replay call) and ``non_tensor_kwargs`` (which keeps the op out of the
    uninitialized-allocation exemption, so replay would really execute it).
    """

    trace, reference = _replay_capture()
    target = next(op for op in trace._layer_list if op.func_name == "relu")
    target._internal_set("func_name", "new")
    target._internal_set("func_id", FunctionRegistryKey("torch.Tensor", "new", "method"))
    target._internal_set("saved_kwargs", dict(kwargs))
    target._internal_set("non_tensor_kwargs", dict(kwargs))
    path = tmp_path / "crafted.tlz"
    trace.save(path, level="executable_with_callables")
    return path, reference


def test_loaded_validation_refuses_a_pointer_keyword_before_resolution(tmp_path: Path) -> None:
    """``validate_forward_pass`` on a crafted bundle refuses by name; nothing is called.

    Never dereferences: the op stays unresolved, so the entry check raises before replay.
    """

    path, reference = _crafted_new_bundle(tmp_path, {"cdata": 0})
    loaded = tl.load(path)
    crafted = next(op for op in loaded._layer_list if op.func_name == "new")
    resolved = resolve_loaded_callables(loaded)
    try:
        assert all(op is not crafted for op in resolved)
        assert crafted.func is None
    finally:
        for op in resolved:
            op.func = None
    with pytest.raises(ArtifactError, match="requires resolved func callables"):
        loaded.validate_forward_pass(reference)


def test_loaded_validation_never_runs_or_verifies_a_pointer_call(tmp_path: Path) -> None:
    """With resolution bypassed, replay still never calls the op and never reports a pass.

    A spy stands in for a callable that slipped past resolution: the replay belt must
    refuse before the call, the refusal surfaces as a failed replay the user is warned
    about, and no broad handler turns it into a verdict of verified.
    """

    path, reference = _crafted_new_bundle(tmp_path, {"cdata": 0})
    loaded = tl.load(path)
    crafted = next(op for op in loaded._layer_list if op.func_name == "new")
    calls: list[Any] = []
    crafted.func = lambda *args, **kwargs: calls.append(kwargs) or torch.zeros(3, 4)
    try:
        with pytest.warns(UserWarning, match="treating as failed validation"):
            outcome = loaded.validate_forward_pass(reference)
    finally:
        crafted.func = None
    assert calls == [], f"the pointer call executed with {calls}"
    assert outcome is not True
    status = getattr(loaded, "validation_replay_status", None)
    assert getattr(status, "state", None) != "passed", f"reported verified: {status}"


def test_loaded_tensor_new_without_a_pointer_keyword_still_resolves(tmp_path: Path) -> None:
    """The refusal keys on the pointer keyword, never on ``Tensor.new`` itself."""

    path, _reference = _crafted_new_bundle(tmp_path, {})
    loaded = tl.load(path)
    crafted = next(op for op in loaded._layer_list if op.func_name == "new")
    resolved = resolve_loaded_callables(loaded)
    assert any(op is crafted for op in resolved)
    for op in resolved:
        op.func = None


def test_replay_execution_refuses_a_pointer_keyword() -> None:
    """The replay belt refuses before the op's function is called (in-memory traces too)."""

    trace, _reference = _replay_capture()
    op = next(op for op in trace._layer_list if op.func_name == "relu")
    calls: list[Any] = []
    op.func = lambda *args, **kwargs: calls.append(kwargs) or torch.zeros(3, 4)
    with pytest.raises(RunPreconditionError) as excinfo:
        execute_replay_func(op, {"args": list(op.call.saved_args), "kwargs": {"cdata": 0}}, [])
    assert excinfo.value.fields["code"] == RunnableErrorCode.CALL_STRUCTURE_MISMATCH.value
    assert calls == []


def test_influence_probe_refuses_a_pointer_keyword() -> None:
    """A probe refusal must RAISE: a probe that cannot run would keep an exemption."""

    calls: list[Any] = []
    layer = SimpleNamespace(
        func=lambda *args, **kwargs: calls.append(kwargs) or torch.zeros(2),
        out=torch.zeros(2),
        layer_label="crafted",
        func_name="new",
    )
    call = _ProbeCall((torch.zeros(2),), {"cdata": 0}, 0)
    with pytest.raises(RunPreconditionError):
        _substitutes_move_output(layer, call, lambda: (torch.ones(2),))  # type: ignore[arg-type]
    assert calls == []


# --------------------------------------------------------------------------- #
# (7) signature-less private-C builtins fail closed unless exactly listed     #
# --------------------------------------------------------------------------- #

_PRIVATE_C_POINTER_ADOPTERS = (
    ("torch._C._aoti", "alloc_tensor_by_stealing_from_void_ptr"),
    ("torch._C._aoti", "AOTIModelContainerRunnerCpu"),
    ("torch._C._cudart", "cudaHostRegister"),
    ("torch._C._cudart", "cudaHostUnregister"),
)


def test_private_c_pointer_adopters_are_refused_even_under_trust() -> None:
    """pybind builtins with no signature (AOTI, cudart) never resolve, trusted or not."""

    checked = 0
    for module_name, name in _PRIVATE_C_POINTER_ADOPTERS:
        try:
            candidate = getattr(importlib.import_module(module_name), name, None)
        except ImportError:
            candidate = None
        if candidate is None:
            continue
        checked += 1
        assert not is_pure_forward_callable(candidate), f"{module_name}.{name} admitted"
        key = FunctionRegistryKey("custom", name, "function", import_path=f"{module_name}:{name}")
        with pytest.raises(UntrustedCallableError):
            resolve_function_registry_key(key, trust_custom_callables=True)
    assert checked, "no private-C pointer adopter was available to check"


def test_uninspectable_builtin_with_a_public_module_name_is_refused() -> None:
    """A pybind class reporting a public ``__module__`` cannot pass on the bare prefix."""

    candidate = torch._C._SDPAParams
    module = str(getattr(candidate, "__module__", ""))
    assert module.startswith("torch.") and not module.startswith("torch._C")
    assert module not in _ALLOWED_FORWARD_OP_MODULES
    assert not is_pure_forward_callable(candidate), f"torch._C._SDPAParams ({module}) admitted"
    key = FunctionRegistryKey(
        "custom", "_SDPAParams", "function", import_path="torch._C:_SDPAParams"
    )
    with pytest.raises(UntrustedCallableError):
        resolve_function_registry_key(key, trust_custom_callables=True)


def _torch_module_callables() -> list[tuple[str, Any]]:
    """Every callable defined in a loaded ``torch.*`` module or ``torch._C`` submodule."""

    modules: dict[str, ModuleType] = {
        name: module
        for name, module in list(sys.modules.items())
        if isinstance(module, ModuleType) and (name == "torch" or name.startswith("torch."))
    }
    stack = [torch._C]
    while stack:
        current = stack.pop()
        for value in list(vars(current).values()):
            is_c_submodule = isinstance(value, ModuleType) and value.__name__.startswith(
                "torch._C."
            )
            if is_c_submodule and value.__name__ not in modules:
                modules[value.__name__] = value
                stack.append(value)
    found: list[tuple[str, Any]] = []
    for module_name, module in sorted(modules.items()):
        for name, value in list(vars(module).items()):
            real_module = str(getattr(value, "__module__", "") or "")
            if callable(value) and real_module.startswith("torch"):
                found.append((f"{module_name}.{name}", value))
    return found


def test_trusted_resolution_fails_closed_for_unlisted_signature_less_callables() -> None:
    """Whole-torch census: what a trusted key can name is listed, refused or inspectable.

    A pointer-taking callable with a signature is refused by the signature rule (and
    the census must SEE at least one, or the check is vacuous); one without a
    signature must live in an EXACTLY listed module.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        offenders, unlisted, refused = _pointer_verdicts(_torch_module_callables())
    assert not offenders, f"admitted callables taking a raw pointer: {offenders}"
    assert not unlisted, f"admitted signature-less callables in unlisted modules: {unlisted}"
    assert refused, "the census saw no pointer-taking callable: the guard would be vacuous"


def test_capture_roster_needs_no_unlisted_private_c_module() -> None:
    """Every decorated torch function still clears the private-C rule (no capture lost)."""

    listed = _ALLOWED_FORWARD_OP_MODULES
    lost: list[str] = []
    for namespace, name in ORIG_TORCH_FUNCS:
        owner: Any = torch
        for part in namespace.split(".")[1:]:
            owner = getattr(owner, part, None)
        value = getattr(owner, name, None) if owner is not None else None
        module = str(getattr(_unwrap_capture_wrapper(value), "__module__", "") or "")
        if module.startswith("torch._C.") and module not in listed:
            lost.append(f"{namespace}.{name} ({module})")
    assert not lost, f"decorated functions in unlisted private-C modules: {lost}"
