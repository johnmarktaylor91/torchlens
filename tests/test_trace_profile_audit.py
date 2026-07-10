"""Tests for Trace-level profile and audit convenience reports."""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class ProfileModel(nn.Module):
    """Small nested model with two parameterized child modules."""

    def __init__(self) -> None:
        """Initialize the fixture modules."""

        super().__init__()
        self.features = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
        self.head = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the fixture forward pass.

        Parameters
        ----------
        x:
            Input batch.

        Returns
        -------
        torch.Tensor
            Output logits.
        """

        return self.head(self.features(x))


class NanModel(nn.Module):
    """Fixture that emits NaNs at one known operation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produce a non-finite output.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Tensor containing NaNs.
        """

        zeros = x - x
        return zeros / zeros


def test_trace_profile_levels_columns_aggregation_and_sorting() -> None:
    """profile exposes complete rows at op, module, and call granularities."""

    trace = tl.trace(ProfileModel().eval(), torch.randn(2, 4))
    op_frame = trace.profile().to_pandas()
    module_frame = trace.profile("module", sort_by="flops").to_pandas()
    call_frame = trace.profile("call").to_pandas()

    required = {
        "name",
        "kind",
        "op_count",
        "time",
        "flops",
        "activation_memory",
        "saved_activation",
        "param_count",
        "dtype",
        "device",
    }
    assert required.issubset(op_frame.columns)
    assert len(op_frame) == len(trace.layer_list)
    assert len(module_frame) == len(trace.modules)
    assert len(call_frame) == len(trace.module_calls)
    assert module_frame["flops"].dropna().tolist() == sorted(
        module_frame["flops"].dropna().tolist(), reverse=True
    )
    features_row = module_frame.loc[module_frame["name"] == "features"].iloc[0]
    features_ops = [
        trace[label] for call in trace.modules["features"].calls.values() for label in call.ops
    ]
    assert features_row["op_count"] == len(features_ops)
    assert features_row["flops"] == sum(int(op.flops_forward or 0) for op in features_ops)
    assert "TraceProfile" not in repr(trace.profile())


def test_trace_profile_preserves_subsecond_time_for_hotspot_sorting() -> None:
    """profile retains float durations rather than truncating them to integer seconds."""

    trace = tl.trace(ProfileModel().eval(), torch.randn(2, 4))
    frame = trace.profile().to_pandas()

    assert frame["time"].notna().all()
    assert (frame["time"] > 0).any()
    assert frame["time"].tolist() == sorted(frame["time"].tolist(), reverse=True)
    assert any(unit in repr(trace.profile()) for unit in (" us", " ms", " s"))


def test_trace_profile_sparse_save_preserves_honest_availability() -> None:
    """profile retains metadata while making unavailable activation payloads visible."""

    trace = tl.trace(ProfileModel().eval(), torch.randn(2, 4), save=tl.func("linear"))
    frame = trace.profile().to_pandas()

    assert "activation_memory" in frame
    assert "saved_activation" in frame
    assert (~frame["saved_activation"].astype(bool)).any()
    assert frame["flops"].notna().any()


def test_trace_audit_clean_model_reports_run_and_skipped_scope() -> None:
    """audit gives a clean result while retaining unsupported-check accounting."""

    audit = tl.trace(ProfileModel().eval(), torch.randn(2, 4)).audit()

    assert audit.findings == ()
    assert "find_nan" in audit.checks_run
    assert "gradient_flow_audit" not in audit.checks_run
    assert any(check == "gradient_flow_audit" for check, _ in audit.skipped)
    assert "no issues found" in repr(audit)


def test_trace_audit_nan_finding_names_op_and_follow_up() -> None:
    """audit reports a non-finite output with its direct diagnostic pointer."""

    audit = tl.trace(NanModel(), torch.ones(1, 2)).audit()

    finding = next(finding for finding in audit.findings if finding.check == "find_nan")
    assert finding.ops and "truediv" in finding.ops[0]
    assert finding.follow_up == "trace.find_nan()"


def test_trace_audit_sparse_save_runs_find_nan_with_honest_scope() -> None:
    """audit scans saved sparse payloads while skipping full-coverage diagnostics."""

    audit = tl.trace(ProfileModel().eval(), torch.randn(2, 4), save=tl.func("linear")).audit()

    skipped = dict(audit.skipped)
    assert "find_nan" in audit.checks_run
    assert "find_nan" not in skipped
    assert "dead_neurons" in skipped


def test_trace_audit_sparse_save_reports_saved_nan_with_uncertainty() -> None:
    """audit reports a NaN retained by a sparse save predicate with its uncertainty zone."""

    audit = tl.trace(NanModel(), torch.ones(1, 2), save=tl.func("truediv")).audit()

    finding = next(finding for finding in audit.findings if finding.check == "find_nan")
    assert "First among saved tensors" in finding.message
    assert "Unsaved upstream uncertainty zone" in finding.message
