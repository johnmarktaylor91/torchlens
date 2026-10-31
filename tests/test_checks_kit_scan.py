"""Checks kit item 1: scan kernel, audit_params, ParamAudit, JSON export.

Pins the checks memo 4.4 contract: mapping-only door, tied-alias dedup,
skip-with-reason, unknown-name refusals, severity mapping, the versioned
export, knob parity with dtype_range_audit, and the fp16 peak-allocated-
bytes regression (NO whole-tensor float64/float32 widening -- the known bad
widen-everything helper is in the tree and will be tempting).
"""

from __future__ import annotations

import json
from inspect import signature

import pytest
import torch
import torch.nn as nn

import torchlens.checks as tc


class _TiedNet(nn.Module):
    """Two names bound to one Linear (tied-weight fixture)."""

    def __init__(self) -> None:
        super().__init__()
        shared = nn.Linear(4, 4, bias=False)
        self.a = shared
        self.b = shared

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the shared layer twice."""

        return self.b(self.a(x))


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.BatchNorm1d(8), nn.Linear(8, 2))


@pytest.mark.smoke
def test_audit_params_clean_model_rows_and_export() -> None:
    """A fresh model audits with rows, coverage, and a versioned export."""

    audit = tc.audit_params(_model())

    assert audit.schema_version == tc.CHECK_REPORT_SCHEMA_VERSION == 1
    assert len(audit.rows) == 9  # 6 params + 3 batchnorm buffers
    assert all(row.audited for row in audit.rows)
    assert audit.coverage["scope"] == "process_local"
    assert audit.coverage["global_reduce"] == "not_applied"
    payload = json.loads(audit.to_json())
    assert payload["schema_version"] == 1
    assert len(payload["rows"]) == 9
    # No nonfinite findings on a fresh model.
    assert not [f for f in audit.findings if f.severity == "critical"]


def test_audit_params_nonfinite_named_critical() -> None:
    """Seeded NaN/Inf in a param and a buffer are named critical findings."""

    model = _model()
    with torch.no_grad():
        model[0].weight[0, 0] = float("nan")
        model[2].running_mean[0] = float("inf")

    audit = tc.audit_params(model)

    critical = {f.names[0]: f for f in audit.findings if f.severity == "critical"}
    assert set(critical) == {"0.weight", "2.running_mean"}
    weight_finding = critical["0.weight"]
    assert weight_finding.code == "param_value_nonfinite"
    assert weight_finding.values["n_nan"] == 1.0
    assert weight_finding.remedy


def test_audit_params_severity_mapping() -> None:
    """Bounds / headroom / subnormal / all-same warn; all-zero is info."""

    mapping = {
        "near_max": torch.full((8,), 0.95 * torch.finfo(torch.float16).max, dtype=torch.float16),
        "subnormal_heavy": torch.full((8,), 1e-42),
        "constant": torch.full((8,), 3.0),
        "all_zero": torch.zeros(8),
        "bounded": torch.linspace(-2.0, 2.0, 8),
    }

    audit = tc.audit_params(mapping, bounds={"bounded": (-1.0, 1.0)})

    by_check = {(f.check, f.names[0]): f for f in audit.findings}
    assert by_check[("dtype_headroom", "near_max")].severity == "warning"
    assert by_check[("param_subnormal", "subnormal_heavy")].severity == "warning"
    assert by_check[("param_all_same", "constant")].severity == "warning"
    assert by_check[("param_all_zero", "all_zero")].severity == "info"
    bounds_finding = by_check[("param_bounds", "bounded")]
    assert bounds_finding.severity == "warning"
    assert bounds_finding.values["below"] > 0 and bounds_finding.values["above"] > 0


def test_audit_params_tied_dedup_keeps_aliases() -> None:
    """Tied tensors are ONE row with every alias preserved (memo 4.4)."""

    audit = tc.audit_params(_TiedNet())

    assert len(audit.rows) == 1
    row = audit.rows[0]
    assert row.name == "a.weight"
    assert row.aliases == ("b.weight",)


@pytest.mark.smoke
def test_audit_params_mapping_door_and_skip_reasons() -> None:
    """state_dict mapping enters the same door; meta/sparse skip with reason."""

    model = _model()
    state_audit = tc.audit_params(dict(model.state_dict()))
    assert len(state_audit.rows) == 9

    mapping = {
        "meta_param": torch.empty(4, device="meta"),
        "sparse": torch.eye(3).to_sparse(),
        "not_a_tensor": 3.14,
        "fine": torch.ones(4),
    }
    audit = tc.audit_params(mapping)
    reasons = dict(audit.skipped)
    assert "meta" in reasons["meta_param"]
    assert "layout" in reasons["sparse"]
    assert "not a tensor" in reasons["not_a_tensor"]
    assert audit.coverage["audited_tensors"] == 1


@pytest.mark.smoke
def test_audit_params_refusals_are_typed() -> None:
    """Unknown names, malformed bounds, junk fractions refuse typed."""

    model = _model()
    with pytest.raises(tc.CheckConfigError) as bounds_exc:
        tc.audit_params(model, bounds={"nope": (0.0, 1.0)})
    assert bounds_exc.value.fields["code"] == "audit_bounds_unknown_name"
    assert bounds_exc.value.fields["remedy"]

    with pytest.raises(tc.CheckConfigError) as within_exc:
        tc.audit_params(model, within=["ghost.weight"])
    assert within_exc.value.fields["code"] == "audit_within_unknown_name"

    with pytest.raises(tc.CheckConfigError) as pair_exc:
        tc.audit_params(model, bounds={"0.weight": (1.0, -1.0)})
    assert pair_exc.value.fields["code"] == "check_bounds_invalid"

    with pytest.raises(tc.CheckConfigError) as fraction_exc:
        tc.audit_params(model, max_fraction=1.5)
    assert fraction_exc.value.fields["code"] == "check_fraction_invalid"

    with pytest.raises(tc.CheckConfigError) as target_exc:
        tc.audit_params([1, 2, 3])
    assert target_exc.value.fields["code"] == "check_target_invalid"


def test_audit_params_within_filters_exact_names() -> None:
    """within= audits exactly the named tensors, day-1 exact spelling."""

    model = _model()
    audit = tc.audit_params(model, within=["0.weight", "3.bias"])
    assert sorted(row.name for row in audit.rows) == ["0.weight", "3.bias"]


def test_debug_door_and_knob_parity() -> None:
    """tl.debug.audit_params is the memo door; knobs match dtype_range_audit."""

    import torchlens as tl

    assert tl.debug.audit_params is tc.audit_params
    assert tl.debug.ParamAudit is tc.ParamAudit

    audit_defaults = signature(tc.audit_params).parameters
    range_defaults = signature(tl.debug.dtype_range_audit).parameters
    # Knob parity through the ONE constants module (memo 4.4).
    assert audit_defaults["max_fraction"].default == range_defaults["max_fraction"].default
    assert (
        audit_defaults["subnormal_fraction_threshold"].default
        == range_defaults["subnormal_fraction_threshold"].default
    )
    assert audit_defaults["max_fraction"].default == tc.MAX_FRACTION_DEFAULT
    assert (
        audit_defaults["subnormal_fraction_threshold"].default
        == tc.SUBNORMAL_FRACTION_THRESHOLD_DEFAULT
    )


class _AllocationRecorder(torch.utils._python_dispatch.TorchDispatchMode):
    """Record the byte size of every tensor an op produces during the scan."""

    def __init__(self, input_numel: int) -> None:
        super().__init__()
        self._input_numel = input_numel
        self.max_intermediate_bytes = 0
        self.saw_widened_float = False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: ANN001, ANN204
        """Track output tensor sizes for every dispatched op."""

        out = func(*args, **(kwargs or {}))
        for value in out if isinstance(out, (list, tuple)) else (out,):
            if isinstance(value, torch.Tensor):
                nbytes = value.numel() * value.element_size()
                # The sanctioned float64 objects are 0-dim stats and the tiny
                # stacked stats vector (a handful of scalars per tensor); a
                # WHOLE-TENSOR widening has input-sized numel.
                if (
                    value.dtype in (torch.float64, torch.float32)
                    and value.numel() >= self._input_numel
                ):
                    self.saw_widened_float = True
                if value.numel() > 1:
                    self.max_intermediate_bytes = max(self.max_intermediate_bytes, nbytes)
        return out


def test_fp16_scan_never_widens_whole_tensors() -> None:
    """The fp16 peak-allocated-bytes regression (memo item 1, the gate).

    Every intermediate the kernel allocates stays <= the input's own byte
    size (fp16 abs/where at 1.0x, bool masks at 0.5x); a whole-tensor
    float32 widening would be 2.0x and float64 4.0x. The known bad helper
    (widen-to-float for numeric ops) must never enter this path for fp16.
    """

    tensor = torch.randn(512, 257, dtype=torch.float16)
    input_bytes = tensor.numel() * tensor.element_size()

    recorder = _AllocationRecorder(tensor.numel())
    with recorder:
        rows = tc.scan_named_tensors([("t", "parameter", tensor)])

    assert rows[0].audited
    assert not recorder.saw_widened_float, "whole-tensor float widening is forbidden"
    assert recorder.max_intermediate_bytes <= input_bytes, (
        f"an intermediate allocated {recorder.max_intermediate_bytes} bytes > "
        f"input {input_bytes}: the scan widened a whole tensor"
    )


def test_scan_counts_are_exact_on_fp16() -> None:
    """Counts and extrema survive the scalar-sync path exactly."""

    tensor = torch.zeros(1000, dtype=torch.float16)
    tensor[:3] = float("nan")
    tensor[3] = float("inf")
    tensor[4] = float("-inf")
    tensor[5] = 2.5
    tensor[6] = -1.25

    (row,) = tc.scan_named_tensors([("t", "parameter", tensor)])

    assert (row.n_nan, row.n_posinf, row.n_neginf) == (3, 1, 1)
    assert row.finite_min == -1.25 and row.finite_max == 2.5 and row.abs_max == 2.5
    assert row.zero_fraction == pytest.approx(993 / 1000)
    assert row.dtype_max == pytest.approx(65504.0)


def test_tensor_digest_proves_change() -> None:
    """Digest differs on any change; equality never claims exactness."""

    tensor = torch.randn(16, 16)
    baseline = tc.tensor_digest(tensor)
    assert tc.tensor_digest(tensor) == baseline
    with torch.no_grad():
        tensor[3, 3] += 1e-3
    assert tc.tensor_digest(tensor) != baseline
