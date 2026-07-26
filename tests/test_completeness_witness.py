"""Adversarial coverage for the opt-in aten completeness witness."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import torchlens as tl
from torchlens._errors import TorchLensCaptureGapWarning
from torchlens.backends.torch.completeness_witness import (
    AUDITED_COMPLETENESS_BOUNDARIES,
    MAX_AUDITED_COMPLETENESS_BOUNDARIES,
)
from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

# (major, minor) of the running torch, dependency-free (e.g. "2.8.0+cpu" -> (2, 8)).
_TORCH_XY = tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2])


@pytest.fixture(autouse=True)
def _isolated_witness_epoch() -> Iterator[None]:
    """Give each witness test a clean process-level wrapper configuration."""

    unwrap_torch()
    yield
    unwrap_torch()
    wrap_torch(
        patch_policy="legacy",
        escape_detector="off",
        completeness_witness=False,
    )


class _WrappedOpsModel(nn.Module):
    """Use only ordinary wrapped torch namespace calls."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run two independently wrapped tensor operations."""

        return torch.sigmoid(torch.relu(x)).add(1)


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

    wrap_torch(patch_policy="scoped", completeness_witness=True)
    trace = tl.trace(_WrappedOpsModel(), torch.randn(4))

    assert trace.completeness_witness_mode == "shadow"
    assert trace.completeness_witness_verified is True
    assert trace.completeness_witness_event_count >= 3
    assert trace.completeness_witness_unaccounted_count == 0
    assert trace.completeness_diagnostics == []
    assert trace.capture_verified is True
    assert trace.capture_verification_reason == "dispatch_witness_verified"


@pytest.mark.smoke
def test_direct_aten_call_trips_non_vacuous_witness() -> None:
    """A direct aten call is loudly and machine-readably unaccounted."""

    wrap_torch(patch_policy="scoped", completeness_witness=True)
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

    wrap_torch(patch_policy="scoped", completeness_witness=True)
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

    wrap_torch(patch_policy="scoped", completeness_witness=True)
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
    assert any(op.func_name == "none" and op.is_internal_source for op in trace.ops)


@pytest.mark.smoke
def test_record_wrapped_ops_have_zero_unaccounted_dispatches() -> None:
    """Fastlog accounting uses capture emission rather than Trace-only events."""

    wrap_torch(patch_policy="scoped", completeness_witness=True)
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

    wrap_torch(patch_policy="scoped", completeness_witness=True)
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

    wrap_torch(patch_policy="scoped", completeness_witness=True)
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

    wrap_torch(patch_policy="scoped", completeness_witness=True)
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

    wrap_torch(patch_policy="scoped", completeness_witness=True)
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

    wrap_torch(patch_policy="scoped", completeness_witness=True)
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
    _TORCH_XY < (2, 2),
    reason=(
        "Graceful pre-wrap-vmap transform-escape handling (witness-only, "
        "capture_verified=False) requires torch>=2.2; on the 2.1 best-effort floor the "
        "escaped vmap output honestly raises an output-attribution error instead."
    ),
)
def test_pre_wrap_vmap_is_witness_only_not_capture_verified() -> None:
    """A clean dispatch census does not verify an escaped raw transform call route."""

    vectorized = torch.vmap(lambda row: row * 2.0)
    model = _PreWrapVmapModel(vectorized)
    wrap_torch(patch_policy="scoped", completeness_witness=True)
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
        patch_policy="scoped",
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

    wrap_torch(patch_policy="scoped", completeness_witness=True)
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

    wrap_torch(patch_policy="scoped", completeness_witness=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(
            _SubclassHiddenMutationModel(), torch.tensor([1.0, 2.0]), save_arg_values=True
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
