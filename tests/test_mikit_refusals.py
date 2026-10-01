"""Provocation rows for the mech-interp kit's refusal vocabulary (S-17).

Every cheaply-provokable ``mi_*`` / ``norm_*`` code gets a direct provoking
test here (pure logic, smoke tier); the model-bearing provocations live in
``test_mikit_core.py``; the handful needing a broken substrate are
consciously baselined in ``test_error_code_coverage.py`` with reasons.
"""

from __future__ import annotations

import sys

import pytest
import torch

import torchlens.mechinterp as mi
from torchlens.mechinterp._errors import MechInterpError

pytestmark = pytest.mark.smoke


def _code(excinfo: pytest.ExceptionInfo) -> str:
    """Return the refusal code from a caught MechInterpError."""

    return str(excinfo.value.fields["code"])


def test_stack_empty_refuses():
    """mi_stack_empty: accumulating a rowless stack refuses."""

    stack = mi.ComponentStack(
        (),
        grading="complete",
        target_coordinate=mi.Coordinate(label="t"),
        target_value=None,
        identity_receipt={},
    )
    with pytest.raises(MechInterpError) as excinfo:
        stack.sum()
    assert _code(excinfo) == "mi_stack_empty"


def test_stack_shape_mixed_refuses():
    """mi_stack_shape_mixed: mixed row shapes with no broadcast target."""

    rows = (
        mi.ComponentRow(coordinate=mi.Coordinate(label="a"), value=torch.zeros(1, 2, 3)),
        mi.ComponentRow(coordinate=mi.Coordinate(label="b"), value=torch.zeros(1, 1, 3)),
    )
    stack = mi.ComponentStack(
        rows,
        grading="complete",
        target_coordinate=mi.Coordinate(label="t"),
        target_value=None,
        identity_receipt={},
    )
    with pytest.raises(MechInterpError) as excinfo:
        stack.stack()
    assert _code(excinfo) == "mi_stack_shape_mixed"


def test_pandas_unavailable_refuses(monkeypatch):
    """mi_pandas_unavailable: to_pandas without pandas refuses typed."""

    monkeypatch.setitem(sys.modules, "pandas", None)
    rows = (mi.ComponentRow(coordinate=mi.Coordinate(label="a"), value=torch.zeros(2)),)
    stack = mi.ComponentStack(
        rows,
        grading="complete",
        target_coordinate=mi.Coordinate(label="t"),
        target_value=None,
        identity_receipt={},
    )
    with pytest.raises(MechInterpError) as excinfo:
        stack.to_pandas()
    assert _code(excinfo) == "mi_pandas_unavailable"


def test_analysis_unknown_refuses_before_trace_use():
    """mi_analysis_unknown: an unknown analysis refuses before any trace read."""

    with pytest.raises(MechInterpError) as excinfo:
        mi.retention_plan(None, analyses=("bogus",))
    assert _code(excinfo) == "mi_analysis_unknown"


def test_head_score_kind_invalid_refuses_before_trace_use():
    """mi_head_score_kind_invalid: unknown kind and measure both land here."""

    with pytest.raises(MechInterpError) as excinfo:
        mi.head_scores(None, kind="bogus")
    assert _code(excinfo) == "mi_head_score_kind_invalid"
    with pytest.raises(MechInterpError) as excinfo:
        mi.head_scores(None, kind="induction", measure="bogus")
    assert _code(excinfo) == "mi_head_score_kind_invalid"


def test_norm_scale_mode_invalid_refuses_before_trace_use():
    """mi_norm_scale_mode_invalid: unknown apply_norm_scale mode."""

    with pytest.raises(MechInterpError) as excinfo:
        mi.apply_norm_scale(None, None, scale="bogus")
    assert _code(excinfo) == "mi_norm_scale_mode_invalid"


def test_translation_generation_unknown_refuses():
    """mi_translation_generation_unknown: unknown TLens generation."""

    with pytest.raises(MechInterpError) as excinfo:
        mi.alias_report(None, generation="9.x")
    assert _code(excinfo) == "mi_translation_generation_unknown"


def test_tokenizer_required_refuses():
    """mi_tokenizer_required: a text prompt with no tokenizer."""

    with pytest.raises(MechInterpError) as excinfo:
        mi.test_prompt(None, "a text prompt", answer=1)
    assert _code(excinfo) == "mi_tokenizer_required"


def test_token_ids_unavailable_refuses():
    """mi_token_ids_unavailable: no captured integer input for score masks."""

    from torchlens.mechinterp._scores import _token_ids

    class _StubTrace:
        input_ops = ()

    with pytest.raises(MechInterpError) as excinfo:
        _token_ids(_StubTrace(), 8)
    assert _code(excinfo) == "mi_token_ids_unavailable"


def test_orientation_unknown_refuses():
    """mi_orientation_unknown: unknown class + square weight never guesses."""

    from torchlens.mechinterp._params import _orientation

    assert _orientation("Conv1D", (64, 192), 64) == "in_out"
    assert _orientation("Linear", (192, 64), 64) == "out_in"
    with pytest.raises(MechInterpError) as excinfo:
        _orientation("MysteryProj", (64, 64), 64)
    assert _code(excinfo) == "mi_orientation_unknown"


def test_projection_unresolvable_refuses():
    """mi_projection_unresolvable: no candidate weight in the subtree."""

    from torchlens.mechinterp._params import _projection_for

    class _StubTrace:
        class _Ops:
            def __getitem__(self, key):
                raise KeyError(key)

        ops = _Ops()

    with pytest.raises(MechInterpError) as excinfo:
        _projection_for(_StubTrace(), object(), "model.attn", [])
    assert _code(excinfo) == "mi_projection_unresolvable"


def test_norm_reconstruction_refusals():
    """norm_eps_unavailable / norm_geometry_mismatch / norm_convention_unmatched."""

    from torchlens.semantic._norm_reconstruction import (
        NormReconstructionError,
        reconstruct_norm,
    )

    x = torch.randn(1, 4, 8)
    with pytest.raises(NormReconstructionError) as excinfo:
        reconstruct_norm(input=x, output=x, gamma=None, beta=None, eps=None)
    assert excinfo.value.fields["code"] == "norm_eps_unavailable"

    with pytest.raises(NormReconstructionError) as excinfo:
        reconstruct_norm(input=x, output=torch.randn(1, 4, 9), gamma=None, beta=None, eps=1e-5)
    assert excinfo.value.fields["code"] == "norm_geometry_mismatch"

    with pytest.raises(NormReconstructionError) as excinfo:
        reconstruct_norm(input=x, output=torch.randn(1, 4, 8), gamma=None, beta=None, eps=1e-5)
    assert excinfo.value.fields["code"] == "norm_convention_unmatched"

    # And the positive control: a genuine LayerNorm application validates.
    ln = torch.nn.LayerNorm(8)
    with torch.no_grad():
        ln.weight.mul_(1.7).add_(0.1)
        ln.bias.add_(0.3)
        record = reconstruct_norm(
            input=x, output=ln(x), gamma=ln.weight, beta=ln.bias, eps=float(ln.eps)
        )
    assert record.kind == "layernorm_affine"
    assert record.validation_receipt["result"] == "validated"


def test_component_stack_records_smoke():
    """Pure-logic record protocol: order, tuple unpacking, top(), refusals."""

    rows = tuple(
        mi.ComponentRow(
            coordinate=mi.Coordinate(label=f"w{i}", kind="writer"),
            value=torch.full((1, 2, 3), float(i)),
        )
        for i in range(3)
    )
    target = rows[0].value + rows[1].value + rows[2].value
    stack = mi.ComponentStack(
        rows,
        grading="complete",
        target_coordinate=mi.Coordinate(label="t"),
        target_value=target,
        identity_receipt={"result": "bitwise_equal"},
    )
    values, labels = stack
    assert labels == ("w0", "w1", "w2") and values.shape == (3, 1, 2, 3)
    assert torch.equal(stack.sum(), target)
    assert stack.top(1)[0][0] == "w2"
    with pytest.raises(MechInterpError) as excinfo:
        stack.top(1, by="max")
    assert excinfo.value.fields["code"] == "mi_top_by_invalid"


def test_translation_table_rows_well_formed():
    """Every translation row carries (tlens, generation, facet) -- the CI row."""

    rows = mi.translation_table()
    assert len(rows) >= 18
    for row in rows:
        assert row["tlens"] and row["generation"] in ("2.x", "3.x-bridge")
        assert row["facet"]
