"""Tests for saved-activation dtype range diagnostics."""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class _DowncastModel(nn.Module):
    """Model with a visible float32-to-float16 boundary."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downcast the input tensor.

        Parameters
        ----------
        x:
            Float32 input.

        Returns
        -------
        torch.Tensor
            Float16 output.
        """

        return x.to(torch.float16)


def test_dtype_range_audit_flags_fp16_near_max() -> None:
    """Finite fp16 values near 65504 produce a measured range finding."""

    dtype_max = torch.finfo(torch.float16).max
    trace = tl.trace(nn.ReLU(), torch.full((2, 3), dtype_max * 0.95, dtype=torch.float16))

    audit = tl.debug.dtype_range_audit(trace)

    finding = next(finding for finding in audit.findings if finding.check == "dtype_near_max")
    assert "torch.float16" in finding.message
    assert "dtype_max=65504" in finding.message
    assert "max_abs=" in finding.message


def test_dtype_range_audit_clean_fp32_has_full_coverage() -> None:
    """A clean fully saved fp32 trace has no findings and honest coverage."""

    trace = tl.trace(nn.ReLU(), torch.full((2, 3), 1.5, dtype=torch.float32))

    audit = tl.debug.dtype_range_audit(trace)

    assert audit.findings == ()
    assert audit.n_ops_audited == audit.n_ops_total
    assert audit.coverage == 1.0


def test_dtype_range_audit_partial_save_reports_incomplete_coverage() -> None:
    """Unsaved operations reduce coverage and are never claimed clean."""

    model = nn.Sequential(nn.Linear(3, 3), nn.ReLU()).eval()
    trace = tl.trace(model, torch.ones(2, 3), save=tl.func("relu"))
    unsaved_labels = {
        op.label
        for op in trace.layer_list
        if int(op.step_index or 0) > 0 and not op.has_saved_activation
    }

    audit = tl.debug.dtype_range_audit(trace)

    assert audit.n_ops_audited < audit.n_ops_total
    assert audit.coverage < 1.0
    assert unsaved_labels
    assert all(not unsaved_labels.intersection(finding.ops) for finding in audit.findings)
    assert f"coverage={audit.n_ops_audited}/{audit.n_ops_total}" in repr(audit)


def test_dtype_range_audit_reports_recorded_downcast_boundary() -> None:
    """Wider recorded input dtype and narrower saved output are reported."""

    trace = tl.trace(_DowncastModel(), torch.ones(2, 3, dtype=torch.float32))

    audit = tl.debug.dtype_range_audit(trace)

    finding = next(finding for finding in audit.findings if finding.check == "dtype_downcast")
    assert "torch.float32" in finding.message
    assert "torch.float16" in finding.message


def test_dtype_range_audit_reports_large_subnormal_fraction() -> None:
    """A tensor dominated by subnormal values carries the measured fraction."""

    subnormal = torch.finfo(torch.float32).tiny / 2
    trace = tl.trace(nn.ReLU(), torch.full((2, 3), subnormal, dtype=torch.float32))

    audit = tl.debug.dtype_range_audit(trace)

    finding = next(finding for finding in audit.findings if finding.check == "dtype_subnormal")
    assert "subnormal_fraction=1" in finding.message
