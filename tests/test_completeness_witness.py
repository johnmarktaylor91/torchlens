"""Adversarial coverage for the opt-in aten completeness witness."""

from __future__ import annotations

import ast
import inspect
import textwrap
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens._errors import TorchLensCaptureGapWarning
from torchlens.backends.torch import completeness_witness as cw
from torchlens.backends.torch._modes import SubclassConstructionUnderDispatchModeError
from torchlens.backends.torch.completeness_witness import (
    AUDITED_COMPLETENESS_BOUNDARIES,
    MAX_AUDITED_COMPLETENESS_BOUNDARIES,
)
from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch
from torchlens.utils._torch_compat import (
    HAS_FUNCTORCH_LEVEL_API,
    HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE,
)
from torchlens.utils.introspection import INPUT_SEARCH_DEPTH_LIMIT


def _observer_patch_ast() -> ast.FunctionDef:
    """Return the parsed AST for ``_observe_invisible_host_escapes``.

    Returns
    -------
    ast.FunctionDef
        Parsed function definition for the observer-install context manager.
    """

    source = textwrap.dedent(inspect.getsource(cw._observe_invisible_host_escapes))
    function = ast.parse(source).body[0]
    assert isinstance(function, ast.FunctionDef)
    return function


def _loop_by_iter(function: ast.FunctionDef, iter_expr: str) -> ast.For:
    """Return the loop matching one iterator expression.

    Parameters
    ----------
    function:
        Parsed observer-install function.
    iter_expr:
        ``ast.unparse`` string for the target loop iterator.

    Returns
    -------
    ast.For
        Matching loop node.
    """

    for statement in ast.walk(function):
        if isinstance(statement, ast.For) and ast.unparse(statement.iter) == iter_expr:
            return statement
    pytest.fail(f"missing observer loop for {iter_expr!r}")


def _statement_blocks(node: ast.AST) -> list[list[ast.stmt]]:
    """Collect nested statement blocks under ``node``.

    Parameters
    ----------
    node:
        AST node to inspect.

    Returns
    -------
    list[list[ast.stmt]]
        Statement lists from bodies, orelse blocks, final blocks, and handlers.
    """

    blocks: list[list[ast.stmt]] = []
    for field_name in ("body", "orelse", "finalbody"):
        field = getattr(node, field_name, None)
        if isinstance(field, list) and field and all(isinstance(stmt, ast.stmt) for stmt in field):
            blocks.append(field)
            for statement in field:
                blocks.extend(_statement_blocks(statement))
    handlers = getattr(node, "handlers", None)
    if isinstance(handlers, list):
        for handler in handlers:
            if isinstance(handler, ast.ExceptHandler):
                blocks.append(handler.body)
                for statement in handler.body:
                    blocks.extend(_statement_blocks(statement))
    return blocks


def _is_observer_failed_add(statement: ast.stmt) -> bool:
    """Return whether ``statement`` records ``_HOST_ESCAPE_OBSERVER_FAILED``.

    Parameters
    ----------
    statement:
        AST statement to classify.

    Returns
    -------
    bool
        ``True`` when the statement is ``_HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)``.
    """

    if not isinstance(statement, ast.Expr):
        return False
    call = statement.value
    if not isinstance(call, ast.Call):
        return False
    func = call.func
    if not isinstance(func, ast.Attribute) or func.attr != "add":
        return False
    owner = func.value
    if not isinstance(owner, ast.Name) or owner.id != "_HOST_ESCAPE_OBSERVER_FAILED":
        return False
    return ast.unparse(call) == "_HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)"


@pytest.fixture(autouse=True)
def _isolated_witness_epoch() -> Iterator[None]:
    """Give each witness test a clean process-level wrapper configuration.

    Teardown restores the PRE-TEST diagnostic modes rather than hardcoding
    them off: a fixed ``escape_detector="off", completeness_witness=False``
    re-wrap silently disarmed diagnostics a surrounding session had
    deliberately armed (R77 fixture-health finding 3).
    """

    saved_escape_detector = _state._escape_detector_mode
    saved_completeness_witness = _state._completeness_witness_mode
    unwrap_torch()
    yield
    unwrap_torch()
    wrap_torch(
        escape_detector=saved_escape_detector,
        completeness_witness=saved_completeness_witness,
    )


class _WrappedOpsModel(nn.Module):
    """Use only ordinary wrapped torch namespace calls."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run two independently wrapped tensor operations."""

        return torch.sigmoid(torch.relu(x)).add(1)


class _AliasedInputMutationModel(nn.Module):
    """Mutate one input site before reading a second aliased site."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return a value that proves whether ``a`` and ``b`` stayed identical."""

        a.add_(10.0)
        return b + 2.0


class _StorageOffsetBranchModel(nn.Module):
    """Branch on physical input-view metadata preserved by capture copying."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return different values for base tensors and nonzero-offset views."""

        adjustment = 100.0 if x.storage_offset() == 0 else -100.0
        return x + adjustment


class _DeepInputModel(nn.Module):
    """Consume a tensor nested below the historical five-level boundary."""

    def __init__(self, depth: int) -> None:
        """Store the number of list levels to unwrap."""

        super().__init__()
        self.depth = depth

    def forward(self, nested: object) -> torch.Tensor:
        """Unwrap ``nested`` and add one to its tensor leaf."""

        value = nested
        for _ in range(self.depth):
            value = value[0]  # type: ignore[index]
        return value + 1.0  # type: ignore[operator, no-any-return]


class _AttrWrap:
    """Plain attribute wrapper the container-boundary walkers do not descend into."""

    __slots__ = ("inner",)

    def __init__(self, inner: object) -> None:
        """Store the wrapped value."""

        self.inner = inner


class _DeepAttrInputModel(nn.Module):
    """Consume a tensor nested below an attribute-wrapper chain."""

    def __init__(self, depth: int) -> None:
        """Store the number of attribute levels to unwrap."""

        super().__init__()
        self.depth = depth

    def forward(self, nested: object) -> torch.Tensor:
        """Unwrap ``nested`` and add one to its tensor leaf."""

        value = nested
        for _ in range(self.depth):
            value = value.inner  # type: ignore[attr-defined]
        return value + 1.0  # type: ignore[operator, no-any-return]


class _DirectAtenGapModel(nn.Module):
    """Run one deliberately unwrapped aten op before a represented sink."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Bypass the Python wrapper namespace for the relu call."""

        escaped = torch.ops.aten.relu.default(x)
        return torch.sigmoid(escaped)


class _DirectAtenChild(nn.Module):
    """Run an unwrapped aten operation inside a child module."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call aten directly.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Direct aten result.
        """

        return torch.ops.aten.relu.default(x)


class _DirectAtenSubmoduleGapModel(nn.Module):
    """Consume a direct-aten child result in a represented sink."""

    def __init__(self) -> None:
        """Create the child module."""

        super().__init__()
        self.child = _DirectAtenChild()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the child and a wrapped sink.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Sigmoid of the escaped child result.
        """

        return torch.sigmoid(self.child(x))


class _DirectAtenIntermediateChild(nn.Module):
    """Use an unwrapped intermediate but return a separately traced output."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a traced sigmoid of an unwrapped relu intermediate."""

        escaped = torch.ops.aten.relu.default(x)
        return torch.sigmoid(escaped)


class _DirectAtenIntermediateSubmoduleGapModel(nn.Module):
    """Expose a child-level raw dispatch not owned by an output boundary."""

    def __init__(self) -> None:
        """Create the child module."""

        super().__init__()
        self.child = _DirectAtenIntermediateChild()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the child whose direct aten intermediate remains unaccounted."""

        return self.child(x).add(1)


class _MutatingDirectAtenOutputChild(nn.Module):
    """Mutate a traced value before returning a separate untraceable raw output."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw relu output after an observable unwrapped in-place mutation."""

        y = x + 1
        torch.ops.aten.mul_.Tensor(y, 2)
        return torch.ops.aten.relu.default(y)


class _MutatingDirectAtenOutputSubmoduleModel(nn.Module):
    """Consume the ATTACK4 child output in a represented parent operation."""

    def __init__(self) -> None:
        """Create the mutating child module."""

        super().__init__()
        self.child = _MutatingDirectAtenOutputChild()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the child and consume its boundary output."""

        return torch.sigmoid(self.child(x))


class _LinearCompositeModel(nn.Module):
    """Exercise a Python-level linear call with a multi-aten decomposition."""

    def __init__(self) -> None:
        """Create stable linear parameters outside the witnessed forward."""

        super().__init__()
        self.weight = nn.Parameter(torch.randn(3, 4))
        self.bias = nn.Parameter(torch.randn(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the functional linear composite."""

        return F.linear(x, self.weight, self.bias)


class _DynamicParameterInitializationModel(nn.Module):
    """Create and initialize a temporary module during the captured forward."""

    def __init__(self) -> None:
        """Create stable prepared state for the represented output path."""

        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Use stable inputs after constructing a dead temporary Linear module."""

        nn.Linear(4, 4)
        return x @ self.weight


class _RegisteredParameterMutationModel(nn.Module):
    """Mutate prepared model state during the captured forward."""

    def __init__(self) -> None:
        """Create the registered parameter that must remain a witnessed gap."""

        super().__init__()
        self.weight = nn.Parameter(torch.ones(4, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mutate the registered parameter before consuming it."""

        with torch.no_grad():
            self.weight.uniform_()
        return x @ self.weight


class _MidForwardAutogradGradModel(nn.Module):
    """Execute a separately captured autograd pass inside the forward."""

    def __init__(self) -> None:
        """Create the parameter differentiated by the inner pass."""

        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiate an inner loss and consume the resulting gradient."""

        inner = (x @ self.weight).square().mean()
        (gradient,) = torch.autograd.grad(inner, self.weight, create_graph=True)
        return x @ (self.weight - 0.01 * gradient)


class _DataPropertyModel(nn.Module):
    """Read Tensor.data before a represented tensor operation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a sigmoid of the captured detach-like property result."""

        return torch.sigmoid(x.data)


class _VmapBoundaryModel(nn.Module):
    """Exercise a documented torch.func transform boundary."""

    def __init__(self) -> None:
        """Build a wrapped vmap boundary callable."""

        super().__init__()
        self.vectorized = torch.vmap(lambda row: torch.sin(row).add(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run opaque transform work followed by a represented sink."""

        return torch.sigmoid(self.vectorized(x))


class _NestedPoolModel(nn.Module):
    """Exercise nested functional pooling wrappers that emit represented ops."""

    def __init__(self, pool: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """Store a pooling callable.

        Parameters
        ----------
        pool:
            Functional pooling call used during forward.
        """

        super().__init__()
        self.pool = pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the configured nested pooling wrapper.

        Parameters
        ----------
        x:
            Four-dimensional image tensor.

        Returns
        -------
        torch.Tensor
            Pooled tensor.
        """

        return self.pool(x)


class _ScalarExtractionModel(nn.Module):
    """Use intentional scalar extraction for data-dependent control flow."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract one item and branch on another scalar tensor.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Input shifted according to its scalar values.
        """

        shift = x.sum().item() + float(x.sum()) + int(x.sum())
        if x.sum() > 0:
            return x + shift
        return x - shift


class _PreWrapVmapModel(nn.Module):
    """Invoke a vmap callable constructed before torch wrapping."""

    def __init__(self, vectorized: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """Store the pre-wrap transform callable.

        Parameters
        ----------
        vectorized:
            Raw vmap callable built before :func:`wrap_torch`.
        """

        super().__init__()
        self.vectorized = vectorized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run only the raw transform route.

        Parameters
        ----------
        x:
            Batched input tensor.

        Returns
        -------
        torch.Tensor
            Vectorized output.
        """

        return self.vectorized(x)


@pytest.mark.smoke
def test_wrapped_ops_have_zero_unaccounted_dispatches() -> None:
    """Every ordinarily wrapped operation is owned by a captured leaf token."""

    wrap_torch(completeness_witness=True)
    trace = tl.trace(_WrappedOpsModel(), torch.randn(4))

    assert trace.completeness_witness_mode == "shadow"
    assert trace.completeness_witness_verified is True
    assert trace.completeness_witness_event_count >= 3
    assert trace.completeness_witness_unaccounted_count == 0
    assert trace.completeness_diagnostics == []
    assert trace.capture_verified is True
    assert trace.capture_verification_reason == "dispatch_witness_verified"


@pytest.mark.smoke
def test_input_copy_preserves_alias_mutation_semantics_and_validation() -> None:
    """Caller protection preserves repeated tensor identity across model sites."""

    wrap_torch(completeness_witness=True)
    plain_input = torch.tensor([1.0])
    plain_output = _AliasedInputMutationModel()(plain_input, plain_input)
    capture_input = torch.tensor([1.0])
    trace = tl.trace(_AliasedInputMutationModel(), [capture_input, capture_input])
    captured_output = trace[trace.output_layers[0]].out

    assert plain_output.tolist() == [13.0]
    assert captured_output.tolist() == [13.0]
    assert capture_input.tolist() == [1.0]
    assert len(trace.input_layers) == 1
    assert trace.capture_verified is True
    assert trace.completeness_witness_verified is True
    validation_input = torch.tensor([1.0])
    assert (
        tl.validation.validate_forward_pass(
            _AliasedInputMutationModel(),
            [validation_input, validation_input],
        )
        is True
    )


@pytest.mark.smoke
def test_input_copy_preserves_nonzero_storage_offset_semantics() -> None:
    """A copied tensor view retains its physical storage offset."""

    wrap_torch(completeness_witness=True)
    plain_input = torch.arange(6.0)[2:5]
    plain_output = _StorageOffsetBranchModel()(plain_input)
    capture_input = torch.arange(6.0)[2:5]
    trace = tl.trace(_StorageOffsetBranchModel(), capture_input)

    assert plain_output.tolist() == [-98.0, -97.0, -96.0]
    assert trace[trace.output_layers[0]].out.tolist() == [-98.0, -97.0, -96.0]
    assert trace.capture_verified is True
    assert trace.completeness_witness_verified is True


@pytest.mark.smoke
def test_deep_input_tensor_is_captured_and_witnessed() -> None:
    """A seven-level tensor input remains a represented graph source."""

    wrap_torch(completeness_witness=True)
    nested: object = torch.tensor([5.0])
    for _ in range(7):
        nested = [nested]
    trace = tl.trace(_DeepInputModel(7), nested)

    assert len(trace.input_layers) == 1
    assert not any(op.unattributed_tensor_args for op in trace.ops)
    assert trace.capture_verified is True
    assert trace.completeness_witness_verified is True
    assert trace.capture_verification_reason == "dispatch_witness_verified"


@pytest.mark.smoke
def test_mid_band_container_depth_is_captured_and_witnessed() -> None:
    """Container nesting in the once-dropped 65-200 band is fully captured.

    The witness walker's private ``64`` ceiling used to silently drop tensor
    leaves for legal inputs in this band (grind-p3 T11.4); after the unified
    ``INPUT_TREE_MAX_DEPTH`` ceiling, a 70-level list input is a represented,
    verified graph source.
    """

    wrap_torch(completeness_witness=True)
    nested: object = torch.tensor([5.0])
    for _ in range(70):
        nested = [nested]
    trace = tl.trace(_DeepInputModel(70), nested)

    assert len(trace.input_layers) == 1
    assert trace.capture_verified is True
    assert trace.completeness_witness_verified is True
    assert trace.capture_verification_reason == "dispatch_witness_verified"


@pytest.mark.smoke
def test_input_depth_limit_fails_closed_with_unresolved_path() -> None:
    """The retained safety ceiling names its frontier and forbids verification.

    Container nesting past ``INPUT_TREE_MAX_DEPTH`` refuses TYPED at capture
    entry (``input_tree_depth_exceeded``; pinned in
    ``test_input_boundary_guards.py``), so the retained witness ceiling is
    exercised through ATTRIBUTE nesting, which the container-boundary entry
    walkers deliberately do not descend into.
    """

    wrap_torch(completeness_witness=True)
    depth = INPUT_SEARCH_DEPTH_LIMIT + 50
    nested: object = torch.tensor([5.0])
    for _ in range(depth):
        nested = _AttrWrap(nested)
    with pytest.warns(TorchLensCaptureGapWarning, match="input_traversal_depth_exceeded"):
        trace = tl.trace(_DeepAttrInputModel(depth), nested)

    assert trace.input_layers == []
    assert trace.capture_verified is False
    assert trace.completeness_witness_verified is False
    assert trace.capture_verification_reason == "input_boundary_unverifiable"
    input_gap = next(
        report
        for report in trace.completeness_diagnostics
        if report["reason"] == "input_traversal_depth_exceeded"
    )
    assert input_gap["input_path"].startswith("input.nested.inner.inner")


@pytest.mark.smoke
def test_direct_aten_call_trips_non_vacuous_witness() -> None:
    """A direct aten call is loudly and machine-readably unaccounted."""

    wrap_torch(completeness_witness=True)
    with pytest.warns(TorchLensCaptureGapWarning, match="unaccounted aten dispatch"):
        trace = tl.trace(_DirectAtenGapModel(), torch.randn(4))

    assert trace.completeness_witness_verified is False
    assert trace.completeness_witness_unaccounted_count == 1
    assert trace.capture_verified is False
    assert trace.capture_verification_reason == "dispatch_witness_unaccounted_ops"
    assert len(trace.completeness_diagnostics) == 1
    report = trace.completeness_diagnostics[0]
    assert report["operator"] == "aten.relu.default"
    assert report["reason"] == "unowned_dispatch"
    assert report["function"] == "forward"
    assert report["owner_wrapper"] is None
    # A forward-body drop did NOT fire inside a replacement hook, so it is a real
    # silent drop that the validation census must fail on (not excused).
    assert report["in_replacement_hook"] is False


@pytest.mark.smoke
def test_genuine_replacement_hook_dispatch_is_tagged_in_replacement_hook() -> None:
    """A raw replacement hook's dispatches belong to its explicit boundary Op.

    A raw ``register_forward_hook`` output replacement runs inside the torchlens
    ``wrapped_hook`` frame. Its raw-aten construction must be owned by the hook
    token and accounted by the synthesized ``intervention_replacement`` boundary,
    while nested Python-wrapped construction remains independently accounted.
    """

    class _Mlp(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(4, 4)
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.relu(self.fc1(x))

    def _replacement_hook(module, inputs, output):  # type: ignore[no-untyped-def]
        return torch.ops.aten.mul.Tensor(output, torch.tensor(0.5))

    wrap_torch(completeness_witness=True)
    model = _Mlp().eval()
    model.relu.register_forward_hook(_replacement_hook)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.trace(model, torch.randn(3, 4))

    assert not any(isinstance(item.message, TorchLensCaptureGapWarning) for item in caught)
    assert trace.capture_verified is True
    assert trace.completeness_witness_verified is True
    assert trace.completeness_witness_unaccounted_count == 0
    assert trace.completeness_diagnostics == []
    hook_owners = [
        row
        for row in trace.completeness_decompositions
        if row["owner_wrapper"] == "module_forward_hook:user"
    ]
    assert len(hook_owners) == 1
    assert hook_owners[0]["capture_accounted"] is True
    assert hook_owners[0]["in_replacement_hook"] is True
    assert "aten.mul.Tensor" in hook_owners[0]["aten_ops"]
    assert any(
        op.func_name == "intervention_replacement" and op.intervention_replaced for op in trace.ops
    )


@pytest.mark.smoke
def test_direct_aten_submodule_output_is_owned_by_internal_source() -> None:
    """A child module's untraceable output is owned by its internal-source boundary."""

    wrap_torch(completeness_witness=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.trace(_DirectAtenSubmoduleGapModel(), torch.randn(4))

    assert not any(isinstance(item.message, TorchLensCaptureGapWarning) for item in caught)
    assert trace.capture_verified is True
    assert trace.completeness_witness_verified is True
    assert trace.completeness_witness_unaccounted_count == 0
    assert trace.completeness_diagnostics == []
    module_owners = [
        row
        for row in trace.completeness_decompositions
        if row["owner_wrapper"] == "module_forward:exhaustive"
        and "aten.relu.default" in row["aten_ops"]
    ]
    assert len(module_owners) == 1
    assert module_owners[0]["capture_accounted"] is True
    assert len(module_owners[0]["capture_accounted_boundary_labels"]) == 1
    assert any(op.func_name == "none" and op.is_internal_source for op in trace.ops)


@pytest.mark.smoke
def test_direct_aten_child_intermediate_still_trips_witness() -> None:
    """A child raw dispatch not represented by its output boundary fails closed."""

    wrap_torch(completeness_witness=True)
    with pytest.warns(TorchLensCaptureGapWarning, match="unaccounted aten dispatch"):
        trace = tl.trace(_DirectAtenIntermediateSubmoduleGapModel(), torch.randn(4))

    assert trace.capture_verified is False
    assert trace.completeness_witness_verified is False
    assert trace.completeness_witness_unaccounted_count == 1
    report = trace.completeness_diagnostics[0]
    assert report["operator"] == "aten.relu.default"
    assert report["reason"] == "owner_not_captured"
    assert report["mutates"] is False
    assert report["file"] == __file__
    assert isinstance(report["line"], int)
    assert report["line"] > 0
    assert report["function"] == "forward"


@pytest.mark.smoke
def test_untraceable_child_output_does_not_mask_observable_mutation() -> None:
    """ATTACK4 mutation remains unaccounted beside an owned output boundary."""

    wrap_torch(completeness_witness=True)
    with pytest.warns(TorchLensCaptureGapWarning, match="unaccounted aten dispatch"):
        trace = tl.trace(_MutatingDirectAtenOutputSubmoduleModel(), torch.randn(4))

    assert trace.capture_verified is False
    assert trace.completeness_witness_verified is False
    assert trace.completeness_witness_unaccounted_count == 1
    mutating = [
        report
        for report in trace.completeness_diagnostics
        if report["operator"] == "aten.mul_.Tensor"
    ]
    assert len(mutating) == 1
    assert mutating[0]["reason"] == "owner_not_captured"
    assert mutating[0]["mutates"] is True
    assert mutating[0]["in_replacement_hook"] is False
    child_owners = [
        row
        for row in trace.completeness_decompositions
        if row["owner_wrapper"] == "module_forward:exhaustive"
        and "aten.mul_.Tensor" in row["aten_ops"]
        and "aten.relu.default" in row["aten_ops"]
    ]
    assert len(child_owners) == 1
    assert child_owners[0]["capture_accounted"] is True
    assert len(child_owners[0]["capture_accounted_boundary_labels"]) == 1
    assert any(op.func_name == "none" and op.is_internal_source for op in trace.ops)


@pytest.mark.smoke
def test_record_wrapped_ops_have_zero_unaccounted_dispatches() -> None:
    """Fastlog accounting uses capture emission rather than Trace-only events."""

    wrap_torch(completeness_witness=True)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        recording = tl.record(model, torch.randn(2, 4), save=tl.func("relu"))

    assert recording.completeness_witness_verified is True
    assert recording.completeness_witness_event_count >= 4
    assert recording.completeness_witness_unaccounted_count == 0
    assert recording.completeness_diagnostics == []
    assert recording.capture_verified is True
    assert not any(isinstance(item.message, TorchLensCaptureGapWarning) for item in caught)


@pytest.mark.smoke
def test_record_direct_aten_call_trips_non_vacuous_witness() -> None:
    """A direct aten gap remains loud on the fastlog capture path."""

    wrap_torch(completeness_witness=True)
    with pytest.warns(TorchLensCaptureGapWarning, match="unaccounted aten dispatch"):
        recording = tl.record(
            _DirectAtenGapModel(),
            torch.randn(4),
            save=tl.func("sigmoid"),
        )

    assert recording.completeness_witness_verified is False
    assert recording.completeness_witness_unaccounted_count == 1
    assert recording.capture_verified is False
    report = recording.completeness_diagnostics[0]
    assert report["operator"] == "aten.relu.default"
    assert report["reason"] == "unowned_dispatch"


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("owner_name", "pool"),
    [
        (
            "adaptive_max_pool2d_with_indices",
            lambda x: F.adaptive_max_pool2d(x, (2, 2)),
        ),
        ("max_pool2d", lambda x: F.max_pool2d(x, 2)),
    ],
)
def test_logged_nested_wrapper_calls_are_accounted(
    owner_name: str,
    pool: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """A non-leaf wrapper is accounted when its func-call id emitted an op.

    Parameters
    ----------
    owner_name:
        Expected inner pooling wrapper name.
    pool:
        Nested functional pooling pair under test.
    """

    wrap_torch(completeness_witness=True)
    trace = tl.trace(_NestedPoolModel(pool), torch.randn(1, 2, 4, 4))

    assert trace.completeness_witness_unaccounted_count == 0
    assert trace.completeness_diagnostics == []
    row = next(
        item for item in trace.completeness_decompositions if item["owner_func_name"] == owner_name
    )
    assert row["capture_accounted"] is True
    assert any(op.func_call_id == row["owner_func_call_id"] for op in trace.ops)


@pytest.mark.smoke
def test_scalar_extraction_boundaries_are_narrowly_accounted() -> None:
    """Python scalar conversions remain intentional scalar-output boundaries."""

    wrap_torch(completeness_witness=True)
    trace = tl.trace(_ScalarExtractionModel(), torch.ones(4))

    assert trace.completeness_witness_unaccounted_count == 0
    assert trace.completeness_diagnostics == []
    scalar_rows = [
        row
        for row in trace.completeness_decompositions
        if row["owner_func_name"] in {"item", "__bool__", "__float__", "__int__"}
    ]
    assert {row["owner_func_name"] for row in scalar_rows} == {
        "item",
        "__bool__",
        "__float__",
        "__int__",
    }
    assert all(row["scope"] == "expected_opaque" for row in scalar_rows)
    assert all(row["aten_ops"] == ("aten._local_scalar_dense.default",) for row in scalar_rows)


@pytest.mark.smoke
def test_linear_decomposition_is_owned_by_one_captured_call() -> None:
    """Multiple aten events owned by one linear call do not false-alarm."""

    wrap_torch(completeness_witness=True)
    trace = tl.trace(_LinearCompositeModel(), torch.randn(2, 4))

    assert trace.completeness_diagnostics == []
    assert trace.completeness_witness_verified is True
    linear_rows = [
        row for row in trace.completeness_decompositions if row["owner_func_name"] == "linear"
    ]
    assert len(linear_rows) == 1
    linear_row = linear_rows[0]
    assert linear_row["capture_accounted"] is True
    assert linear_row["aten_ops"] == ("aten.t.default", "aten.addmm.default")
    assert any(op.func_call_id == linear_row["owner_func_call_id"] for op in trace.ops)


@pytest.mark.smoke
def test_vmap_interior_is_expected_opaque() -> None:
    """Documented vmap interiors remain outside the active dispatch census."""

    wrap_torch(completeness_witness=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.trace(_VmapBoundaryModel(), torch.randn(3, 4))

    assert any("captured a vmap transform as a boundary op" in str(item.message) for item in caught)
    assert trace.completeness_diagnostics == []
    assert trace.completeness_witness_verified is True
    assert trace.capture_verified is True
    assert "vmap" in {op.func_name for op in trace.ops}


@pytest.mark.smoke
@pytest.mark.skipif(
    not HAS_FUNCTORCH_LEVEL_API,
    reason=(
        "Graceful pre-wrap-vmap transform-escape handling (witness-only, "
        "capture_verified=False) needs torch._C._functorch.maybe_current_level to detect "
        "the escape (_is_inside_functorch_transform); absent it (the whole 2.1/2.2 floor, "
        "not just 2.1 -- the capability postdates 2.2 too), the escape goes undetected and "
        "the unattributable output honestly raises an output-attribution error instead."
    ),
)
def test_pre_wrap_vmap_is_witness_only_not_capture_verified() -> None:
    """A clean dispatch census does not verify an escaped raw transform call route."""

    vectorized = torch.vmap(lambda row: row * 2.0)
    model = _PreWrapVmapModel(vectorized)
    wrap_torch(completeness_witness=True)
    with pytest.warns(UserWarning, match="functorch"):
        trace = tl.trace(model, torch.randn(3, 4))

    assert trace._raw_transform_escape_detected is True
    assert trace.completeness_witness_verified is True
    assert trace.completeness_witness_unaccounted_count == 0
    assert trace.capture_verified is False
    assert trace.capture_verification_reason == "transform_call_route_unverified"
    assert "vmap" not in {op.func_name for op in trace.ops}


@pytest.mark.smoke
def test_escape_detector_and_witness_compose_on_shared_tokens() -> None:
    """Both diagnostics can run together and independently verify a clean call."""

    wrap_torch(
        escape_detector="shadow",
        completeness_witness=True,
    )
    trace = tl.trace(_WrappedOpsModel(), torch.randn(4))

    assert trace.escape_detector_verified is True
    assert trace.escape_diagnostics == []
    assert trace.completeness_witness_verified is True
    assert trace.completeness_diagnostics == []
    assert trace.capture_verified is True
    assert trace.capture_verification_reason == "dispatch_witness_and_detector_verified"


def test_expected_opaque_boundary_table_is_exact_and_budgeted() -> None:
    """Metadata-only exclusions remain a small reviewable exact-name table."""

    assert {row.wrapper_name for row in AUDITED_COMPLETENESS_BOUNDARIES} == {
        "torch_func:numpy:not_logged",
        "torch_func:__array__:not_logged",
        "torch_func:size:not_logged",
        "torch_func:dim:not_logged",
        "torch_func:item:logged",
        "torch_func:__bool__:logged",
        "torch_func:__float__:logged",
        "torch_func:__int__:logged",
        "autograd:grad",
    }
    scalar_rows = [row for row in AUDITED_COMPLETENESS_BOUNDARIES if row.operator is not None]
    assert {row.wrapper_name for row in scalar_rows} == {
        "torch_func:item:logged",
        "torch_func:__bool__:logged",
        "torch_func:__float__:logged",
        "torch_func:__int__:logged",
    }
    assert {row.operator for row in scalar_rows} == {"aten._local_scalar_dense.default"}
    assert all(row.reason for row in AUDITED_COMPLETENESS_BOUNDARIES)


def test_observer_install_loops_fail_closed_before_each_continue() -> None:
    """Every required observer-install loop must flag failure before continuing.

    Returns
    -------
    None
        Asserts that every structural ``continue`` in the required install loops is
        preceded in-branch by ``_HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)``.
    """

    function = _observer_patch_ast()
    install_iters = (
        "INVISIBLE_HOST_ESCAPE_FUNCS | STORAGE_BRIDGE_ESCAPE_FUNCS",
        "_MODULE_ESCAPE_TARGETS()",
        "_STORAGE_RAW_POINTER_TARGETS()",
        "INVISIBLE_HOST_ESCAPE_PROPERTIES",
        "INPUT_METADATA_PREDICATE_FUNCS",
        "INPUT_METADATA_BOOL_METHODS",
        "INPUT_METADATA_PROPERTY_NAMES",
    )
    for iter_expr in install_iters:
        loop = _loop_by_iter(function, iter_expr)
        continue_blocks = [
            block
            for block in _statement_blocks(loop)
            if any(isinstance(stmt, ast.Continue) for stmt in block)
        ]
        assert continue_blocks, f"{iter_expr} no longer has structural fail-closed coverage"
        for block in continue_blocks:
            for index, statement in enumerate(block):
                if isinstance(statement, ast.Continue):
                    assert index > 0, f"{iter_expr} has a bare continue with no guard"
                    assert _is_observer_failed_add(block[index - 1]), (
                        f"{iter_expr} continue is not fail-closed"
                    )


def test_observer_restore_loops_fail_closed_on_restore_error() -> None:
    """Every targeted observer-restore loop must downgrade on restore failure.

    Returns
    -------
    None
        Asserts the restore loops no longer swallow ``TypeError``/``AttributeError``
        with ``pass``.
    """

    function = _observer_patch_ast()
    restore_iters = (
        "originals.items()",
        "module_originals",
        "storage_originals",
        "property_originals.items()",
        "metadata_originals.items()",
        "bool_method_originals.items()",
        "grad_property_restore.items()",
    )
    for iter_expr in restore_iters:
        loop = _loop_by_iter(function, iter_expr)
        handlers = [node for node in ast.walk(loop) if isinstance(node, ast.ExceptHandler)]
        assert handlers, f"{iter_expr} restore loop lost its guarded restore path"
        for handler in handlers:
            assert not any(isinstance(statement, ast.Pass) for statement in handler.body), (
                f"{iter_expr} restore handler still swallows failure"
            )
            assert any(_is_observer_failed_add(statement) for statement in handler.body), (
                f"{iter_expr} restore handler is not fail-closed"
            )


def test_invisible_host_escape_property_lookup_uses_tensor_mro() -> None:
    """The property observer must resolve descriptors through ``torch.Tensor``'s MRO.

    Returns
    -------
    None
        Asserts the property install loop uses ``inspect.getattr_static`` on
        ``torch.Tensor`` rather than metaclass or ``__dict__`` probing.
    """

    function = _observer_patch_ast()
    loop = _loop_by_iter(function, "INVISIBLE_HOST_ESCAPE_PROPERTIES")
    descriptor_assignments = [
        statement
        for statement in loop.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "descriptor"
            for target in statement.targets
        )
    ]
    assert descriptor_assignments, "property observer loop no longer binds a descriptor"
    descriptor_call = descriptor_assignments[0].value
    assert ast.unparse(descriptor_call) == "inspect.getattr_static(torch.Tensor, name, None)"


def test_completeness_witness_functions_have_docstrings() -> None:
    """The verification module must keep docstrings on every function.

    Returns
    -------
    None
        Fails with the sorted ``lineno:name`` list for any undocumented function in
        ``completeness_witness.py``.
    """

    source_paths = [
        Path(cw.__file__),
        *sorted(Path(cw.__file__).parent.glob("_completeness_*.py")),
    ]
    module_asts = [ast.parse(path.read_text()) for path in source_paths]
    missing = sorted(
        f"{node.lineno}:{node.name}"
        for module_ast in module_asts
        for node in ast.walk(module_ast)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and ast.get_docstring(node) is None
    )
    assert not missing, f"missing docstrings: {missing}"


def test_writeback_sampling_outer_failures_are_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outer sampling failures must trip the mutable-writeback ceiling.

    Parameters
    ----------
    monkeypatch:
        Pytest monkeypatch fixture.

    Returns
    -------
    None
        Forces an outer sampling failure and asserts it marks the trace
        ``_HOST_ESCAPE_MUTABLE_WRITEBACK`` set.
    """

    class _Trace:
        """Weakrefable trace stand-in for the fail-closed weak set."""

    trace = _Trace()
    state = cw._WitnessState(trace=trace, owner_thread_id=0, guard_pass_index=1)
    state.writeback_watch.append((torch.tensor([1.0]), None, torch.tensor([1], dtype=torch.uint8)))
    monkeypatch.setattr(
        cw,
        "_iter_dispatch_tensors",
        lambda args, kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(cw, "_has_state_toctou_watch", lambda _trace: False)

    assert trace not in cw._HOST_ESCAPE_MUTABLE_WRITEBACK
    cw._sample_writeback_at_consumption(state, (torch.tensor([2.0]),), None)
    assert trace in cw._HOST_ESCAPE_MUTABLE_WRITEBACK


def test_finalize_census_keeps_dispatch_reason_across_later_guard_passes() -> None:
    """Later clean passes must not relabel earlier dispatch gaps as input-boundary gaps."""

    trace = SimpleNamespace(
        completeness_diagnostics=[
            {
                "scope": "active_logging",
                "reason": "owner_not_captured",
                "operator": "aten.relu.default",
            }
        ],
        completeness_decompositions=[],
    )
    state = cw._WitnessState(trace=trace, owner_thread_id=0, guard_pass_index=2)

    cw._finalize_census(state)

    assert trace.capture_verified is False
    assert trace.capture_verification_reason == "dispatch_witness_unaccounted_ops"
    assert len(trace.completeness_diagnostics) == 1


@pytest.mark.smoke
def test_dynamic_parameter_initializers_are_captured_without_hiding_state_mutations() -> None:
    """Capture temporary Parameter initialization while registered-state writes fail closed."""

    wrap_torch(completeness_witness=True)
    temporary_trace = tl.trace(_DynamicParameterInitializationModel(), torch.randn(2, 4))

    initializer_rows = [
        row
        for row in temporary_trace.completeness_decompositions
        if row["owner_func_name"] == "uniform_"
    ]
    assert len(initializer_rows) == 2
    assert all(row["capture_accounted"] is True for row in initializer_rows)
    assert temporary_trace.completeness_diagnostics == []
    assert temporary_trace.capture_verified is True

    with pytest.warns(TorchLensCaptureGapWarning, match="aten.uniform_"):
        state_trace = tl.trace(_RegisteredParameterMutationModel(), torch.randn(2, 4))

    assert state_trace.capture_verified is False
    assert [
        (row["operator"], row["reason"], row["mutates"])
        for row in state_trace.completeness_diagnostics
    ] == [("aten.uniform_.default", "owner_not_captured", True)]


@pytest.mark.smoke
def test_mid_forward_autograd_grad_is_an_exact_backward_boundary() -> None:
    """Exclude only engine dispatches represented by the captured backward pass."""

    wrap_torch(completeness_witness=True)
    with pytest.warns(UserWarning, match="no graph/source provenance"):
        trace = tl.trace(_MidForwardAutogradGradModel(), torch.randn(2, 4))

    boundary_rows = [
        row for row in trace.completeness_decompositions if row["owner_wrapper"] == "autograd:grad"
    ]
    assert len(boundary_rows) == 1
    assert boundary_rows[0]["scope"] == "expected_opaque"
    assert boundary_rows[0]["aten_ops"]
    assert trace.num_backward_passes == 1
    assert trace.completeness_diagnostics == []
    assert trace.capture_verified is True


@pytest.mark.smoke
def test_tensor_data_getter_dispatch_is_captured() -> None:
    """Represent the C-level data getter's detach dispatch as an ordinary op."""

    wrap_torch(completeness_witness=True)
    trace = tl.trace(_DataPropertyModel(), torch.randn(4))

    detach_ops = [op for op in trace.ops if op.func_name == "detach"]
    assert len(detach_ops) == 1
    assert detach_ops[0].parents == ("input_1",)
    assert trace.completeness_diagnostics == []
    assert trace.capture_verified is True


def test_is_mutating_operator_reads_schema_not_name() -> None:
    """Mutation detection uses the operator schema, covering out= overloads too."""

    from torchlens.backends.torch.completeness_witness import _is_mutating_operator

    assert _is_mutating_operator(torch.ops.aten.mul_.Tensor) is True
    assert _is_mutating_operator(torch.ops.aten.add_.Tensor) is True
    assert _is_mutating_operator(torch.ops.aten.copy_.default) is True
    assert _is_mutating_operator(torch.ops.aten.zero_.default) is True
    # out= overload mutates yet its name does NOT end in an underscore.
    assert _is_mutating_operator(torch.ops.aten.add.out) is True
    # Pure reads -- exactly the benign owner_not_captured control-flow comparisons.
    assert _is_mutating_operator(torch.ops.aten.equal.default) is False
    assert _is_mutating_operator(torch.ops.aten.allclose.default) is False
    assert _is_mutating_operator(torch.ops.aten.add.Tensor) is False
    assert _is_mutating_operator(torch.ops.aten.sigmoid.default) is False
    # A callable with no schema is treated as non-mutating (fail-safe).
    assert _is_mutating_operator(object()) is False


class _DirectMutatingAtenModel(nn.Module):
    """Perform an OBSERVABLE in-place aten mutation via an unwrapped route."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mutate a tensor in place through a direct (unwrapped) aten call."""

        y = x + 1
        torch.ops.aten.mul_.Tensor(y, 2)
        return torch.sigmoid(y)


@pytest.mark.smoke
def test_observable_uncaptured_mutation_is_flagged_mutates() -> None:
    """An observable uncaptured in-place aten op is tripped AND tagged mutates.

    A direct ``aten.mul_`` call is unowned (a real silent drop) and, being an
    in-place op, carries ``mutates=True``. This proves the witness observes and
    tags value-affecting drops it can actually see, distinct from a pure read.
    """

    wrap_torch(completeness_witness=True)
    with pytest.warns(TorchLensCaptureGapWarning, match="unaccounted aten dispatch"):
        trace = tl.trace(_DirectMutatingAtenModel(), torch.randn(4))

    assert trace.completeness_witness_verified is False
    mutating = [d for d in trace.completeness_diagnostics if d["mutates"] is True]
    assert mutating, "expected the in-place mul_ drop to be tagged mutates=True"
    assert any(d["operator"].startswith("aten.mul_") for d in mutating)
    # A pure-read op that also dispatched (e.g. add) is never tagged mutates.
    assert all(
        d["mutates"] is False
        for d in trace.completeness_diagnostics
        if d["operator"].startswith("aten.add.")
    )


class _SubclassHiddenMutationTensor(torch.Tensor):
    """Hide an in-place mutation inside ``aten.equal`` under disabled dispatch."""

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        """Zero the first equal operand while the dispatcher is suppressed."""

        del types
        if func is torch.ops.aten.equal.default:
            with torch._C._DisableTorchDispatch():
                torch.ops.aten.mul_.Tensor(args[0], 0)
            return True
        with torch._C._DisableTorchDispatch():
            return func(*args, **(kwargs or {}))


class _SubclassHiddenMutationModel(nn.Module):
    """Mutate a tensor through a subclass ``equal`` the witness cannot observe."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Alias the input as the adversarial subclass and mutate via equal."""

        subclass = torch.Tensor._make_subclass(
            _SubclassHiddenMutationTensor, value, require_grad=False
        )
        torch.equal(subclass, subclass)
        with torch._C._DisableTorchDispatch():
            base = subclass.as_subclass(torch.Tensor)
        return torch.sigmoid(base + 1)


def test_subclass_disabled_dispatch_mutation_is_outside_observational_reach() -> None:
    """DOCUMENTED BOUNDARY: a mutation hidden under _DisableTorchDispatch is invisible.

    PyTorch suppresses dispatcher re-entry while a tensor subclass handles an op,
    so an aten mutation the subclass performs under
    ``torch._C._DisableTorchDispatch()`` never reaches the witness. This pins the
    exact observational boundary: the witness sees only the OUTER pure-read
    ``aten.equal.default`` (benign ``owner_not_captured``) and cannot see the
    nested ``aten.mul_``. This is honestly recorded here rather than silently
    claimed as caught -- the strengthening closes every OBSERVABLE mutation, but a
    subclass that deliberately disables dispatch is a cooperative-model boundary.
    """

    wrap_torch(completeness_witness=True)
    if not HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE:
        # Floor-torch capability gap (torch 2.1/2.2): constructing the
        # _SubclassHiddenMutationTensor subclass while the completeness
        # witness mode is active refuses typed instead of crashing on
        # torch's own "already associated to a python object" RuntimeError.
        # See SubclassConstructionUnderDispatchModeError.
        with (
            warnings.catch_warnings(),
            pytest.raises(SubclassConstructionUnderDispatchModeError),
        ):
            warnings.simplefilter("ignore")
            tl.trace(
                _SubclassHiddenMutationModel(),
                torch.tensor([1.0, 2.0]),
                capture=tl.options.CaptureOptions(save_arg_values=True),
            )
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(
            _SubclassHiddenMutationModel(),
            torch.tensor([1.0, 2.0]),
            capture=tl.options.CaptureOptions(save_arg_values=True),
        )

    operators = [d["operator"] for d in trace.completeness_diagnostics]
    # The nested mutating op is fundamentally invisible to the witness.
    assert not any(op.startswith("aten.mul_") for op in operators)
    equal_diags = [
        d for d in trace.completeness_diagnostics if d["operator"] == "aten.equal.default"
    ]
    assert equal_diags, "the outer equal is the only observable dispatch"
    # What IS observed is a pure read (mutates=False) -> correctly NOT counted as a
    # value-affecting drop. The mutation is out of reach, by design of dispatch.
    assert all(d["reason"] == "owner_not_captured" for d in equal_diags)
    assert all(d["mutates"] is False for d in equal_diags)
    assert len(AUDITED_COMPLETENESS_BOUNDARIES) <= MAX_AUDITED_COMPLETENESS_BOUNDARIES


def test_witness_arm_install_failure_strands_no_patches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A BaseException escaping the observer INSTALL phase must unwind cleanly.

    R07 regression: ~240 lines of process-global ``setattr``s ran BEFORE the
    ``try`` owning the yield, and that try had no except arm -- Python never
    calls ``__exit__`` when ``__enter__`` raises, so one Ctrl-C during an
    ``intervention_ready`` capture stranded ~124 torch surfaces (the
    ``torch._ops.*.__call__`` dispatch, the storage accessor surface, the
    scalar/predicate belt) permanently and monotonically: the next armed
    capture read the leaked wrapper as its "original" and re-installed over
    it. The whole install phase is now fenced by the L4 unwind standard.
    """

    from torchlens.backends.torch.wrappers import wrap_torch

    wrap_torch()

    def _surface_snapshot() -> dict[str, object]:
        import torch._ops as torch_ops

        return {
            "Tensor.item": inspect.getattr_static(torch.Tensor, "item"),
            "Tensor.tolist": inspect.getattr_static(torch.Tensor, "tolist"),
            "Tensor.numpy": inspect.getattr_static(torch.Tensor, "numpy"),
            "torch.equal": torch.equal,
            "UntypedStorage.data_ptr": inspect.getattr_static(torch.UntypedStorage, "data_ptr"),
            "OpOverloadPacket.__call__": torch_ops.OpOverloadPacket.__dict__.get("__call__"),
        }

    baseline = _surface_snapshot()

    # Inject the interrupt at the LAST install step, so every earlier patch
    # has already landed and must be unwound.
    def _interrupted() -> object:
        raise KeyboardInterrupt

    # The runtime observer is the cw-rebound copy, so patch cw's globals.
    monkeypatch.setattr(cw, "_private_c_module_callables", _interrupted)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.lin(x))

    with pytest.raises(KeyboardInterrupt):
        tl.trace(
            Model(),
            torch.randn(2, 4),
            capture=tl.options.CaptureOptions(intervention_ready=True, cache=False),
        )

    leaked = {name for name, obj in _surface_snapshot().items() if obj is not baseline[name]}
    assert not leaked, f"stranded witness patches: {sorted(leaked)}"

    # No monotonic stacking either: a later armed capture completes and the
    # surfaces are still pristine afterwards.
    monkeypatch.undo()
    trace = tl.trace(
        Model(),
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(intervention_ready=True, cache=False),
    )
    assert trace is not None
    leaked_after = {name for name, obj in _surface_snapshot().items() if obj is not baseline[name]}
    assert not leaked_after, f"stranded witness patches after clean capture: {sorted(leaked_after)}"


def test_observer_restore_leaves_no_shadow_entries() -> None:
    """Arm/disarm must leave ``torch.Tensor`` and storage class dicts unchanged.

    Most belt names (``tolist``/``numpy``/``data_ptr``/``stride``/
    ``is_contiguous``/the bool-method belt/``UntypedStorage.data_ptr``) are C
    slots on the BASE class and not in the patched class ``__dict__``; the
    pre-fix plain-setattr restore left a permanent shadow entry behind after
    every armed capture, so ``unwrap_torch``'s "torch is clean again" claim
    was false at the class-dict level.
    """

    class _Trace:
        """Weakrefable trace stand-in."""

    trace = _Trace()
    state = cw._WitnessState(trace=trace, owner_thread_id=0, guard_pass_index=1)
    tensor_keys_before = set(vars(torch.Tensor))
    storage_classes = (torch.UntypedStorage, torch.TypedStorage)
    storage_keys_before = {cls: set(vars(cls)) for cls in storage_classes}

    with cw._observe_invisible_host_escapes(state):
        # The belt is armed: the census-invisible conversions are shadowed in.
        assert "tolist" in vars(torch.Tensor)
        assert "stride" in vars(torch.Tensor)

    assert set(vars(torch.Tensor)) == tensor_keys_before
    for cls in storage_classes:
        assert set(vars(cls)) == storage_keys_before[cls]
