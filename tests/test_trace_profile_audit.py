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

    # Boundary pseudo-rows own no time (identity partition, A1): their cells
    # are NaN and sort last; every REAL op row keeps its float duration.
    op_rows = frame[frame["kind"] == "op"]
    boundary_rows = frame[frame["kind"] == "boundary"]
    assert not boundary_rows.empty
    assert boundary_rows["time"].isna().all()
    assert op_rows["time"].notna().all()
    assert (op_rows["time"] > 0).any()
    op_times = op_rows["time"].tolist()
    assert op_times == sorted(op_times, reverse=True)
    assert frame["time"].notna().tolist() == sorted(frame["time"].notna().tolist(), reverse=True), (
        "NaN boundary rows must sort after every timed row"
    )
    assert any(unit in repr(trace.profile()) for unit in (" us", " ms", " s"))


def test_trace_profile_sparse_save_preserves_honest_availability() -> None:
    """profile retains metadata while making unavailable activation payloads visible."""

    trace = tl.trace(ProfileModel().eval(), torch.randn(2, 4), save=tl.func("linear"))
    frame = trace.profile().to_pandas()

    assert "activation_memory" in frame
    assert "saved_activation" in frame
    assert (~frame["saved_activation"].astype(bool)).any()
    assert frame["flops"].notna().any()


def test_trace_profile_top_k_truncates_after_sorting() -> None:
    """top_k retains only the leading rows after the requested stable sort."""

    trace = tl.trace(ProfileModel().eval(), torch.randn(2, 4))
    full = trace.profile(sort_by="activation_memory").to_pandas()
    bottlenecks = trace.profile(sort_by="activation_memory", top_k=3).to_pandas()

    assert len(bottlenecks) == 3
    assert bottlenecks["name"].tolist() == full["name"].tolist()[:3]


def test_trace_profile_tree_follows_module_call_nesting() -> None:
    """Tree indentation follows invocation parents rather than address parsing."""

    profile = tl.trace(ProfileModel().eval(), torch.randn(2, 4)).profile()
    lines = profile.tree().splitlines()

    features_line = next(line for line in lines if line.endswith("features:1"))
    linear_line = next(line for line in lines if line.endswith("features.0:1"))
    # ASCII rails (C02 safety tranche, lovely bug 10): returned report
    # strings are ASCII-canonical.
    assert features_line.startswith("|-- ")
    assert linear_line.startswith("|   ")
    assert lines.index(linear_line) > lines.index(features_line)


def test_trace_profile_honesty_labels_missing_timing_as_unknown() -> None:
    """Resource provenance distinguishes measured, estimated, and absent values."""

    trace = tl.trace(ProfileModel().eval(), torch.randn(2, 4))
    missing_timing_op = next(
        op for op in trace.layer_list if not (op.is_input or op.is_output or op.is_buffer)
    )
    missing_timing_op._internal_set("func_duration", None)
    profile = trace.profile(sort_by="activation_memory")
    honesty = profile.honesty()
    row = honesty.loc[honesty["name"] == missing_timing_op.label].iloc[0]

    assert row["time"] == "unknown"
    # Boundary pseudo-rows carry not_applicable, never a fabricated evidence
    # label and never "unknown" (costreport D2). F09 (D9): host time carries
    # the instrumentation qualifier, never bare "measured"; FLOPs carry the
    # compute-record evidence (formula_exact here); shape-derived memory is
    # a formula fact, not an estimate.
    assert set(honesty["time"]).issubset(
        {"measured+instrumentation_inclusive", "unknown", "not_applicable"}
    )
    assert set(honesty["activation_memory"]).issubset(
        {"formula_exact+shape_derived", "unknown", "not_applicable"}
    )
    assert "formula_exact" in set(honesty["flops"])
    assert "not_applicable" in set(honesty["flops"])


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
