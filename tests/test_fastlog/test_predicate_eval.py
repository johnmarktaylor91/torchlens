"""Tests for fastlog predicate normalization and RecordContext construction."""

from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

from torchlens.fastlog._predicate import (
    _evaluate_keep_module,
    _evaluate_keep_op,
    _normalize_capture_decision,
)
from torchlens.fastlog._record_context import _build_record_context
from torchlens.fastlog.exceptions import PredicateError
from torchlens.fastlog.options import RecordingOptions
from torchlens.fastlog.types import CaptureSpec, ModuleStackFrame


def _ctx() -> object:
    """Build a minimal operation context for predicate tests."""

    return _build_record_context(
        kind="op",
        layer_pass_log_or_op_data={
            "label": "linear_1_1_raw",
            "func_name": "linear",
            "tensor": torch.ones(2, 3),
        },
        event_index=1,
        op_index=1,
    )


def test_normalize_capture_decision_bool_and_none_rules() -> None:
    """Boolean and None returns normalize according to the slot default."""

    ctx = _ctx()
    default = CaptureSpec(save_activation=True, save_metadata=False, keep_grad=True)

    keep = _normalize_capture_decision(True, ctx, default)
    skip = _normalize_capture_decision(False, ctx, default)
    inherited = _normalize_capture_decision(None, ctx, default)

    assert keep == CaptureSpec(save_activation=True, save_metadata=True, keep_grad=True)
    assert skip == CaptureSpec(save_activation=False, save_metadata=False)
    assert inherited is default


def test_normalize_capture_decision_accepts_capture_spec() -> None:
    """CaptureSpec returns pass through unchanged."""

    ctx = _ctx()
    spec = CaptureSpec(save_activation=False, save_metadata=True)

    assert _normalize_capture_decision(spec, ctx, False) is spec


@pytest.mark.parametrize("bad_result", [1, "yes", torch.tensor(True)])
def test_normalize_capture_decision_rejects_invalid_returns(bad_result: object) -> None:
    """Invalid predicate return values raise PredicateError with context."""

    ctx = _ctx()

    with pytest.raises(PredicateError) as exc_info:
        _normalize_capture_decision(bad_result, ctx, False)  # type: ignore[arg-type]

    assert exc_info.value.ctx is ctx
    assert exc_info.value.result is bad_result


def test_evaluate_keep_op_and_module_use_predicates_and_defaults() -> None:
    """Slot evaluators call the right predicate and default."""

    ctx = _ctx()
    options = RecordingOptions(
        keep_op=lambda event: event.func_name == "linear",
        keep_module=lambda event: None,
        default_module=True,
    )

    assert _evaluate_keep_op(ctx, options).save_activation is True
    assert _evaluate_keep_module(ctx, options) == CaptureSpec(
        save_activation=True,
        save_metadata=True,
    )


def test_record_context_constructor_is_schema_source_of_truth() -> None:
    """Equivalent inputs from real and synthesized data produce identical contexts."""

    tensor = torch.ones(2, 3)
    module_stack = (
        ModuleStackFrame(
            module_address="encoder",
            module_type="Linear",
            module_id=123,
            pass_index=1,
        ),
    )
    op_data = {
        "label": "relu_1_2_raw",
        "raw_label": "relu_1_2_raw",
        "creation_order": 2,
        "func_name": "relu",
        "module_address": "encoder",
        "module_type": "Linear",
        "module_pass_index": 1,
        "parent_labels": ("input_1_raw",),
        "tensor": tensor,
        "output_index": 0,
        "is_bottom_level_func": True,
    }

    real_ctx = _build_record_context(
        kind="op",
        layer_pass_log_or_op_data=op_data,
        module_stack=module_stack,
        event_index=2,
        op_index=1,
        time_since_pass_start=0.25,
    )
    synth_ctx = _build_record_context(
        kind="op",
        layer_pass_log_or_op_data=op_data,
        module_stack=tuple(frame for frame in module_stack),
        event_index=2,
        op_index=1,
        time_since_pass_start=0.25,
    )

    assert asdict(real_ctx) == asdict(synth_ctx)
