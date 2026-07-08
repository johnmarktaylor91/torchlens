"""Tests for structured replay-failure diagnostics (sprint bucket B1).

These cover the ADD-ONLY diagnostics side-channel that lets a forward-validation
mismatch carry the ACTUAL reason (divergent op, shapes/dtypes, max abs/rel diff,
which check fired) instead of the bare ``repr(False) == "False"`` that previously
reached the menagerie ledger.

LOCKED INVARIANT under test: the diagnostics never change the pass/fail decision.
A clean model still validates; a corrupted capture still fails -- only the error
*message* is richer. The regression gate (``test_regression_replay_diagnostic_*``)
keeps this class from silently degrading back to ``"False"``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from torchlens.user_funcs import _run_model_and_save_specified_outs
from torchlens.validation import ValidationFailure, get_validation_failure
from torchlens.validation.core import validate_saved_outs
from torchlens.validation.diagnostics import (
    CHECK_GROUND_TRUTH,
    CHECK_REPLAY,
    describe_tensor_mismatch,
    record_validation_failure,
    reset_validation_failure,
)


def _capture_full_trace(model: nn.Module, x: torch.Tensor):
    """Capture a full save_arg_values trace the way forward validation does."""

    return _run_model_and_save_specified_outs(
        model=model,
        input_args=(x,),
        input_kwargs={},
        layers_to_save="all",
        activation_transform=None,
        mark_layer_depths=False,
        detach_saved_activations=False,
        save_grads=False,
        save_arg_values=True,
        random_seed=0,
        save_rng_states=True,
    )


def _patch_op_func_to_diverge(trace: Any, func_name: str, delta: float = 1.0) -> str:
    """Replace one op's ``func`` so isolated replay diverges from the saved out.

    The saved ``out`` is left intact (arg-logging still matches), so the FIRST
    failure surfaced is the forward-replay value mismatch -- the exact class the
    video-transformer / Wan* failures fall into.

    Returns the patched op's layer label.
    """

    for layer in trace.layer_list:
        op = layer.ops[1] if hasattr(layer.ops, "_list") else layer
        if getattr(op, "func_name", None) == func_name:
            original = op.func

            def diverging(*args: Any, _orig=original, **kwargs: Any):
                return _orig(*args, **kwargs) + delta

            op.func = diverging
            return layer.layer_label
    raise AssertionError(f"no op with func_name={func_name!r} in trace")


# ---------------------------------------------------------------------------
# PART 1 -- diagnostics unit tests
# ---------------------------------------------------------------------------


def test_describe_tensor_mismatch_computes_metrics() -> None:
    """A saved-vs-recomputed tensor pair yields shapes, dtypes, and diffs."""

    saved = torch.zeros(3, 4)
    recomputed = torch.zeros(3, 4)
    recomputed[0, 0] = 0.5

    failure = describe_tensor_mismatch(
        saved, recomputed, check=CHECK_REPLAY, op_label="op_1", func_name="linear"
    )

    assert failure.check == CHECK_REPLAY
    assert failure.op_label == "op_1"
    assert failure.func_name == "linear"
    assert failure.saved_shape == (3, 4)
    assert failure.recomputed_shape == (3, 4)
    assert failure.saved_dtype == "f32"
    assert failure.max_abs_diff == 0.5
    assert failure.max_rel_diff is not None and failure.max_rel_diff > 0
    # The compact summary must mention the location and the magnitude.
    summary = failure.summary()
    assert "op_1" in summary
    assert "linear" in summary
    assert "max_abs_diff" in summary


def test_describe_tensor_mismatch_reports_shape_divergence() -> None:
    """Differing shapes are reported without attempting an element-wise diff."""

    failure = describe_tensor_mismatch(
        torch.zeros(3, 4), torch.zeros(3, 5), check=CHECK_REPLAY, op_label="op_2"
    )
    assert failure.saved_shape == (3, 4)
    assert failure.recomputed_shape == (3, 5)
    # No element-wise diff is defined across mismatched shapes.
    assert failure.max_abs_diff is None
    assert "(3, 4)" in failure.summary()
    assert "(3, 5)" in failure.summary()


def test_describe_tensor_mismatch_flags_nan_pattern() -> None:
    """A divergent NaN pattern is flagged and the magnitude stays finite."""

    saved = torch.zeros(4)
    recomputed = torch.zeros(4)
    recomputed[1] = float("nan")
    failure = describe_tensor_mismatch(saved, recomputed, check=CHECK_REPLAY)
    assert failure.nan_mismatch is True
    # nan replaced by sentinel -> diff is finite, not NaN.
    assert failure.max_abs_diff is not None
    assert failure.max_abs_diff == failure.max_abs_diff  # not NaN


def test_describe_tensor_mismatch_never_raises_on_non_tensor() -> None:
    """A non-tensor operand degrades to a partial diagnostic, never an error."""

    failure = describe_tensor_mismatch(
        torch.zeros(2), "not-a-tensor", check=CHECK_REPLAY, op_label="op_x"
    )
    assert isinstance(failure, ValidationFailure)
    assert failure.recomputed_shape is None
    assert "recomputed_type" in failure.extra


def test_record_validation_failure_is_first_write_wins() -> None:
    """Only the FIRST recorded failure is kept (matches BFS return ordering)."""

    trace = type("T", (), {})()
    reset_validation_failure(trace)
    record_validation_failure(trace, ValidationFailure(check="first"))
    record_validation_failure(trace, ValidationFailure(check="second"))
    failure = get_validation_failure(trace)
    assert failure is not None
    assert failure.check == "first"


def test_to_dict_roundtrips_fields() -> None:
    """``to_dict`` exposes every field for ledger/JSON serialization."""

    failure = ValidationFailure(
        check=CHECK_REPLAY,
        op_label="conv2d_3_4",
        func_name="conv2d",
        max_abs_diff=0.1,
        saved_shape=(1, 64, 7, 7),
    )
    d = failure.to_dict()
    assert d["check"] == CHECK_REPLAY
    assert d["op_label"] == "conv2d_3_4"
    assert d["saved_shape"] == [1, 64, 7, 7]
    assert d["max_abs_diff"] == 0.1


# ---------------------------------------------------------------------------
# PART 1 -- decision-invariance: diagnostics NEVER change pass/fail
# ---------------------------------------------------------------------------


def test_clean_validation_passes_and_records_no_failure() -> None:
    """A correct capture validates True and leaves the side-channel empty."""

    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    x = torch.randn(2, 8)
    gt = model(x).detach().clone()
    trace = _capture_full_trace(model, x)
    try:
        result = validate_saved_outs(trace, [gt], validate_metadata=True)
        # validate_saved_outs returns a ValidationReplayStatus (da60a76e); a
        # clean capture must be an explicit "passed", not merely truthy.
        assert result.state == "passed"
        assert get_validation_failure(trace) is None
    finally:
        trace.cleanup()


# ---------------------------------------------------------------------------
# PART 3 -- regression gate: the replay-mismatch class cannot silently recur
# ---------------------------------------------------------------------------


def test_regression_replay_diagnostic_carries_real_mismatch() -> None:
    """A synthesized forward-replay mismatch surfaces the ACTUAL divergent op.

    This is the regression gate for the video-transformer ``failed:replay`` class
    (Wan*/SanaVideo/SkyReels/ChronoEdit/Helios, etc.). It locks two things:

    1. The validation still FAILS on a genuinely-wrong replay (tripwire armed).
    2. The failure carries a structured diagnostic -- divergent op label, func
       name, shapes/dtypes, and a non-trivial ``max_abs_diff`` -- NOT ``"False"``.

    If a future change reverts the diagnostics to a bare bool, the assertions on
    ``failure`` go None and this test goes red.
    """

    class TinyNet(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(x) * 2.0

    model = TinyNet()
    x = torch.randn(2, 8)
    gt = model(x).detach().clone()
    trace = _capture_full_trace(model, x)
    try:
        patched_label = _patch_op_func_to_diverge(trace, "__mul__", delta=1.0)
        result = validate_saved_outs(trace, [gt], validate_metadata=False)

        # (1) tripwire still fires -- a wrong replay is still a FAILURE.
        assert result.state == "failed"

        # (2) the failure carries the real mismatch, not repr(False).
        failure = get_validation_failure(trace)
        assert failure is not None
        assert failure.check == CHECK_REPLAY
        assert failure.op_label is not None and patched_label.split(":")[0] in failure.op_label
        assert failure.func_name == "__mul__"
        assert failure.saved_shape == (2, 8)
        assert failure.recomputed_shape == (2, 8)
        assert failure.max_abs_diff is not None and failure.max_abs_diff >= 1.0 - 1e-6
        summary = failure.summary()
        assert "False" != summary
        assert "max_abs_diff" in summary
    finally:
        trace.cleanup()


def test_regression_ground_truth_diagnostic_carries_real_mismatch() -> None:
    """A wrong ground-truth output surfaces a structured ground-truth diagnostic.

    Locks the output-vs-ground-truth path (phase 0 of ``validate_saved_outs``):
    a mismatched expected output must fail AND carry shapes + max_abs_diff, not
    a bare bool.
    """

    model = nn.Sequential(nn.Linear(8, 4))
    x = torch.randn(2, 8)
    wrong_gt = torch.zeros(2, 4)  # deliberately not the model's real output
    trace = _capture_full_trace(model, x)
    try:
        result = validate_saved_outs(trace, [wrong_gt], validate_metadata=False)
        assert result.state == "failed"
        failure = get_validation_failure(trace)
        assert failure is not None
        assert failure.check == CHECK_GROUND_TRUTH
        assert failure.saved_shape == (2, 4)
        assert failure.recomputed_shape == (2, 4)
        assert failure.max_abs_diff is not None
    finally:
        trace.cleanup()
