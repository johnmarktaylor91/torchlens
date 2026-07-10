"""Adversarial coverage for the opt-in aten completeness witness."""

from __future__ import annotations

import warnings
from collections.abc import Iterator

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
    }
    assert all(row.reason for row in AUDITED_COMPLETENESS_BOUNDARIES)
    assert len(AUDITED_COMPLETENESS_BOUNDARIES) <= MAX_AUDITED_COMPLETENESS_BOUNDARIES
