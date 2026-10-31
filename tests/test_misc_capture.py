"""Phase 4 miscellaneous capture tests."""

from __future__ import annotations

import torchlens as tl


def test_register_op_rule_affects_flops() -> None:
    """Custom op rules are accepted by the utility registry."""

    from torchlens.capture.flops import register_op_rule

    register_op_rule("custom_test_op", lambda *_args: 123, None)
    from torchlens.capture.flops import compute_forward_flops

    assert compute_forward_flops("custom_test_op", (1,), [], (), {}) == 123


def test_fastlog_grad_capture_spec() -> None:
    """Fastlog capture specs expose the grad retention flag."""

    spec = tl.fastlog.CaptureSpec(keep_grad=True)
    assert spec.save_grad is True
