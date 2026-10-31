"""Tests for fastlog predicate normalization and RecordContext construction."""

from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

from torchlens.capture.predicates import (
    _evaluate_keep_op,
    _module_capture_spec,
    _normalize_capture_decision,
)
from torchlens.capture.projections import _build_record_context
from torchlens.fastlog.exceptions import PredicateError
from torchlens.fastlog.options import RecordingOptions
from torchlens.fastlog.types import CaptureSpec, ModuleStackFrame
from torchlens.intervention.selectors import BaseSelector


def _ctx() -> object:
    """Build a minimal operation context for predicate tests."""

    return _build_record_context(
        kind="op",
        op_log_or_op_data={
            "label": "linear_1_1_raw",
            "func_name": "linear",
            "tensor": torch.ones(2, 3),
        },
        event_index=1,
        step_index=1,
    )


def test_normalize_capture_decision_bool_and_none_rules() -> None:
    """Boolean and None returns normalize according to the slot default."""

    ctx = _ctx()
    default = CaptureSpec(
        save_out=True,
        save_metadata=False,
        keep_grad=True,
        save_mode="reference",
    )

    keep = _normalize_capture_decision(True, ctx, default)
    skip = _normalize_capture_decision(False, ctx, default)
    inherited = _normalize_capture_decision(None, ctx, default)

    assert keep == CaptureSpec(
        save_out=True,
        save_metadata=True,
        keep_grad=True,
        save_mode="reference",
    )
    assert skip == CaptureSpec(save_out=False, save_metadata=False)
    assert inherited is default


def test_normalize_capture_decision_accepts_capture_spec() -> None:
    """CaptureSpec returns pass through unchanged."""

    ctx = _ctx()
    spec = CaptureSpec(save_out=False, save_metadata=True)

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
    """The op slot calls its predicate; module events follow default_module."""

    ctx = _ctx()
    options = RecordingOptions(
        keep_op=lambda event: event.func_name == "linear",
        default_module=True,
    )

    assert _evaluate_keep_op(ctx, options).save_out is True
    assert _module_capture_spec(options) == CaptureSpec(
        save_out=True,
        save_metadata=True,
    )


class _CountingMissingLabelSelector(BaseSelector):
    """Structured selector that records every capture-time invocation."""

    calls: list[str]

    def __init__(self, calls: list[str]) -> None:
        """Initialize a label selector that intentionally never matches.

        Parameters
        ----------
        calls
            Mutable list receiving the observed ``ctx.label`` spellings.
        """

        object.__setattr__(self, "selector_kind", "label")
        object.__setattr__(self, "selector_value", "missing_label")
        object.__setattr__(self, "calls", calls)

    def __call__(self, ctx: object) -> bool:
        """Record the invocation and delegate to selector matching.

        Parameters
        ----------
        ctx
            Capture-time predicate context.

        Returns
        -------
        bool
            Whether the missing label matches ``ctx``.
        """

        self.calls.append(getattr(ctx, "label"))
        return super().__call__(ctx)


def test_evaluate_keep_op_skips_alias_retry_for_structured_selectors() -> None:
    """Structured selectors evaluate once even when they miss the current op."""

    calls: list[str] = []
    selector = _CountingMissingLabelSelector(calls)
    ctx = _ctx()
    options = RecordingOptions(keep_op=selector)

    assert _evaluate_keep_op(ctx, options) == CaptureSpec(save_out=False, save_metadata=False)
    assert calls == ["linear_1_1_raw"]


def test_record_context_constructor_is_schema_source_of_truth() -> None:
    """Equivalent inputs from real and synthesized data produce identical contexts."""

    tensor = torch.ones(2, 3)
    module_stack = (
        ModuleStackFrame(
            address="encoder",
            module_type="Linear",
            module_id=123,
            pass_index=1,
        ),
    )
    op_data = {
        "label": "relu_1_2_raw",
        "raw_label": "relu_1_2_raw",
        "raw_index": 2,
        "func_name": "relu",
        "address": "encoder",
        "module_type": "Linear",
        "module_pass_index": 1,
        "parent_labels": ("input_1_raw",),
        "tensor": tensor,
        "output_index": 0,
        "is_bottom_level_func": True,
    }

    real_ctx = _build_record_context(
        kind="op",
        op_log_or_op_data=op_data,
        module_stack=module_stack,
        event_index=2,
        step_index=1,
        time_since_pass_start=0.25,
    )
    synth_ctx = _build_record_context(
        kind="op",
        op_log_or_op_data=op_data,
        module_stack=tuple(frame for frame in module_stack),
        event_index=2,
        step_index=1,
        time_since_pass_start=0.25,
    )

    assert asdict(real_ctx) == asdict(synth_ctx)
