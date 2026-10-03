"""W051-MECH regression pins for AUD-CODE 2.11: roofline parameter traffic.

The roofline's read-once/write-once traffic model read the parameter term
through ``getattr(op, "params_memory", None)`` -- a misspelling of the Op
field ``param_memory`` -- so every parameter byte silently vanished and a
Linear(256, 512) reported 85 FLOP/B instead of 1.96 FLOP/B. These tests pin
the traffic arithmetic against the Op record AND pin the field names the
model reads against the Op field vocabulary, so a rename cannot zero a
traffic term again without failing loudly.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.constants import OP_LOG_FIELD_ORDER
from torchlens.report import _cost_perf
from torchlens.report._cost_perf import ROOFLINE_TRAFFIC_FIELDS, roofline


def test_roofline_traffic_fields_are_in_the_op_vocabulary() -> None:
    """Every field the traffic model reads is a real Op record field."""

    assert "param_memory" in ROOFLINE_TRAFFIC_FIELDS
    assert "activation_memory" in ROOFLINE_TRAFFIC_FIELDS
    for name in ROOFLINE_TRAFFIC_FIELDS:
        assert name in OP_LOG_FIELD_ORDER, name
    # The historical misspelling must never be a vocabulary member.
    assert "params_memory" not in OP_LOG_FIELD_ORDER


def test_traffic_field_pin_refuses_a_name_outside_the_vocabulary(monkeypatch) -> None:
    """The import-time pin fires on a misspelled/renamed traffic field."""

    monkeypatch.setattr(
        _cost_perf, "ROOFLINE_TRAFFIC_FIELDS", ("activation_memory", "params_memory")
    )
    with pytest.raises(RuntimeError, match="params_memory"):
        _cost_perf._pin_traffic_fields()


def _linear_traffic_case(bias: bool) -> None:
    lin = nn.Linear(256, 512, bias=bias)
    x = torch.randn(4, 256)
    trace = tl.trace(lin, x)
    try:
        # The roofline reads Op records (trace.layer_list); the Layer facade
        # deliberately serves only the per-layer total_param_memory.
        op = next(o for o in trace.layer_list if o.layer_label == "linear_1_1")
        param_bytes = int(op.param_memory)
        expected_param_bytes = (256 * 512 + (512 if bias else 0)) * 4
        assert param_bytes == expected_param_bytes
        result = roofline(trace)
        row = next(r for r in result.rows if r.label.startswith("linear"))
        activation_bytes = (4 * 256 + 4 * 512) * 4
        assert row.ideal_traffic_bytes == activation_bytes + param_bytes
        # The defective figure (activations only) must not reappear.
        assert row.ideal_traffic_bytes != activation_bytes
        assert row.intensity is not None
        assert row.intensity == pytest.approx(row.flops / row.ideal_traffic_bytes)
        assert row.intensity < 3.0  # 1.96 FLOP/B on the corrected traffic
    finally:
        trace.cleanup()


def test_roofline_linear_traffic_counts_parameter_bytes() -> None:
    """Linear(256, 512) traffic = activations + the weight bytes the op reads."""

    _linear_traffic_case(bias=False)


def test_roofline_linear_traffic_counts_bias_bytes_too() -> None:
    """The bias read rides the same param_memory term."""

    _linear_traffic_case(bias=True)
