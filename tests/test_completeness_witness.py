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


@pytest.mark.smoke
def test_direct_aten_call_in_submodule_still_trips_witness() -> None:
    """Accounting fixes do not dull a direct-aten tripwire in a child module."""

    wrap_torch(patch_policy="scoped", completeness_witness=True)
    with pytest.warns(TorchLensCaptureGapWarning, match="unaccounted aten dispatch"):
        trace = tl.trace(_DirectAtenSubmoduleGapModel(), torch.randn(4))

    assert trace.completeness_witness_verified is False
    assert trace.completeness_witness_unaccounted_count == 1
    report = trace.completeness_diagnostics[0]
    assert report["operator"] == "aten.relu.default"
    assert report["reason"] == "owner_not_captured"
    assert report["file"] == __file__
    assert isinstance(report["line"], int)
    assert report["line"] > 0
    assert report["function"] == "forward"


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
    assert len(AUDITED_COMPLETENESS_BOUNDARIES) <= MAX_AUDITED_COMPLETENESS_BOUNDARIES
