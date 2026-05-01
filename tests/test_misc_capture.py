"""Phase 4 miscellaneous capture tests."""

from __future__ import annotations

import json
from pathlib import Path

import torch

import torchlens as tl


def test_register_op_rule_affects_flops() -> None:
    """Custom op rules are accepted by the utility registry."""

    tl.utils.register_op_rule("custom_test_op", lambda *_args: 123, None)
    from torchlens.capture.flops import compute_forward_flops

    assert compute_forward_flops("custom_test_op", (1,), [], (), {}) == 123


def test_decide_recording_of_batch_marks_discarded() -> None:
    """Retroactive batch decision can discard a log."""

    log = tl.log_forward_pass(torch.nn.ReLU(), torch.ones(1, 2))
    kept = tl.decide_recording_of_batch(log, lambda _log: False)
    assert kept is False
    assert log.recording_kept is False


def test_profiler_execution_trace_bridge(tmp_path: Path) -> None:
    """Write a lightweight profiler execution trace."""

    trace_path = tmp_path / "trace.json"
    log = tl.log_forward_pass(torch.nn.ReLU(), torch.ones(1, 2))
    payload = tl.bridge.profiler.execution_trace(log, trace_path)
    assert payload["nodes"]
    assert json.loads(trace_path.read_text(encoding="utf-8"))["nodes"]


def test_fastlog_gradient_capture_spec() -> None:
    """Fastlog capture specs expose the gradient retention flag."""

    spec = tl.fastlog.CaptureSpec(keep_grad=True)
    assert spec.keep_grad is True
