"""Tests for plain-capture tensor-to-Python scalar escape warnings."""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import ScalarEscapeWarning
from torchlens.options import CaptureOptions


class _ItemScale(nn.Module):
    """Feed an item-derived scalar into a later tensor operation."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Scale inputs by a tensor-derived Python scalar."""
        scalar = (inputs + 1).sum().item()
        return inputs * scalar


class _BoolBranch(nn.Module):
    """Use a captured scalar tensor in Python control flow."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Select one branch via Tensor.__bool__."""
        if inputs.sum() > 0:
            return inputs + 1
        return inputs - 1


class _Clean(nn.Module):
    """Avoid every tensor-to-Python scalar conversion."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a clean tensor-only computation."""
        return torch.relu(inputs + 1)


class _ManyEscapes(nn.Module):
    """Perform several captured tensor scalar conversions."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Read three Python scalars and preserve ordinary forward behavior."""
        first = inputs.sum().item()
        second = float(inputs.mean())
        third = int(inputs.max())
        return inputs + first + second + third


def test_item_escape_warns_with_count_location_and_remediation() -> None:
    """An item-derived downstream literal produces the dedicated warning."""
    with pytest.warns(ScalarEscapeWarning) as warning_records:
        tl.trace(_ItemScale(), torch.ones(2))

    # Key on the warning's own category, not bare record count: a floor-torch
    # install may also fire a one-time TorchCapabilityWarning from an
    # unrelated capability probe tripped by this capture, which must not be
    # mistaken for a second scalar-escape disclosure.
    scalar_warnings = [w for w in warning_records if issubclass(w.category, ScalarEscapeWarning)]
    assert len(scalar_warnings) == 1
    message = str(scalar_warnings[0].message)
    assert "1 tensor-to-Python scalar escape(s)" in message
    assert f"{__file__}:" in message
    assert "keep it as a tensor or pass the value as an explicit input" in message.lower()
    assert "the dependence is not captured" in message


def test_bool_control_flow_escape_warns() -> None:
    """A Tensor.__bool__ control-flow conversion is observed."""
    with pytest.warns(ScalarEscapeWarning, match="1 tensor-to-Python"):
        tl.trace(_BoolBranch(), torch.ones(2))


def test_clean_model_and_torchlens_internal_reads_do_not_warn() -> None:
    """Tensor-only user code and internal scalar bookkeeping remain quiet."""
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always", ScalarEscapeWarning)
        tl.trace(_Clean(), torch.tensor([1.0]))

    assert not [record for record in warning_records if record.category is ScalarEscapeWarning]


def test_many_escapes_emit_one_aggregate_warning() -> None:
    """One Trace emits at most one warning carrying the aggregate count."""
    with pytest.warns(ScalarEscapeWarning) as warning_records:
        tl.trace(_ManyEscapes(), torch.tensor([1.0, 2.0]))

    # See test_item_escape_warns_with_count_location_and_remediation: narrow
    # to this warning's own category so an incidental floor-torch capability
    # notice cannot masquerade as a second aggregate-count disclosure.
    scalar_warnings = [w for w in warning_records if issubclass(w.category, ScalarEscapeWarning)]
    assert len(scalar_warnings) == 1
    assert "3 tensor-to-Python scalar escape(s)" in str(scalar_warnings[0].message)


def test_warning_class_is_filterable() -> None:
    """Users can silence scalar escape diagnostics by their dedicated category."""
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("ignore", ScalarEscapeWarning)
        tl.trace(_ItemScale(), torch.ones(2))

    # Narrow to this category: a floor-torch install may still record an
    # unrelated one-time TorchCapabilityWarning here (this filter only
    # ignores ScalarEscapeWarning), which is not what this test checks.
    assert not [w for w in warning_records if issubclass(w.category, ScalarEscapeWarning)]


def test_runnable_capture_uses_existing_witness_without_plain_warning() -> None:
    """Runnable eligibility retains its existing witness path and verdict behavior."""
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always", ScalarEscapeWarning)
        trace = tl.trace(
            _ItemScale(),
            torch.ones(2),
            capture=CaptureOptions(intervention_ready=True),
        )

    assert not [record for record in warning_records if record.category is ScalarEscapeWarning]
    assert trace.completeness_witness_mode == "off"
    assert not hasattr(trace, "scalar_escape_count")


class _EscapeThenBoom(nn.Module):
    """Read a scalar escape, then fail the forward."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Escape one Python scalar and raise."""
        scalar = inputs.sum().item()
        raise ValueError(f"boom after escape {scalar}")


def test_escape_advisory_never_replaces_inflight_capture_exception() -> None:
    """b3-sol rollup of R07-1: the aggregate advisory used to fire from an
    unconditional ``finally``, so a warnings-as-error filter raised it during
    unwind and REPLACED the real in-flight capture failure (cascading with the
    terminal capture-failed advisory). The user's exception must propagate."""
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.filterwarnings("error", category=ScalarEscapeWarning)
        with pytest.raises(ValueError, match="boom after escape"):
            tl.trace(_EscapeThenBoom(), torch.ones(2))


def test_escape_advisory_still_raises_on_success_path_under_error_filter() -> None:
    """With no in-flight exception, an as-error filter legitimately surfaces
    the advisory as the raised error -- nothing is being masked."""
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.filterwarnings("error", category=ScalarEscapeWarning)
        with pytest.raises(ScalarEscapeWarning):
            tl.trace(_ItemScale(), torch.ones(2))
