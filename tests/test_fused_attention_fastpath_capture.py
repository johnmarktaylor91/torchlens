"""Capture, replay and ground truth run one attention program (wave-3 stress F4).

``nn.MultiheadAttention`` takes its fused ``_native_multi_head_attention`` fast path
in eval mode only while no parameter requires a gradient under grad mode. Capture
forces ``requires_grad`` on every parameter for provenance, which alone flipped that
gate: an eval, frozen, batch-first attention module ran the fused kernel eagerly and
the unfused program under capture, so forward validation's pristine ground truth
differed from the replay by one float rounding and ``ground_truth`` failed. Capture
now holds torch's public fast-path switch off for its forward, validation holds it
for the ground truth and the fast live run for its native forward, and the trace
records the hold as a capture advisory.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._capture_honesty import (
    ADVISORY_FUSED_ATTENTION_FASTPATH_HELD,
    capture_advisories,
)
from torchlens._capture_state_helpers import hold_fused_attention_fastpath_off
from torchlens._utils._torch_compat import (
    get_mha_fastpath_enabled,
    get_mha_fastpath_switch_support,
)
from torchlens.validation import last_validation_failure

WIDTH = 8
BATCH = 3


class FastPathAttention(nn.Module):
    """``nn.MultiheadAttention`` set up for its inference fast path (eval, frozen, batch-first)."""

    def __init__(self) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(WIDTH, 2, batch_first=True)
        self.attention.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attention(x, x, x, need_weights=False)
        return out


def _model_and_input() -> tuple[nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    return FastPathAttention().eval(), torch.randn(BATCH, 3, WIDTH)


def _needs_switch() -> None:
    if not get_mha_fastpath_switch_support():
        pytest.skip("torch build without the public torch.backends.mha fast-path switch")


@pytest.mark.smoke
def test_eager_takes_the_fused_path_and_capture_records_the_unfused_program() -> None:
    _needs_switch()
    model, x = _model_and_input()
    native_calls = 0
    original = torch._native_multi_head_attention

    def counting(*args: object, **kwargs: object) -> object:
        nonlocal native_calls
        native_calls += 1
        return original(*args, **kwargs)

    torch._native_multi_head_attention = counting  # type: ignore[assignment]
    try:
        with torch.no_grad():
            model(x)
        assert native_calls == 1, "the row no longer exercises the eager fast path"
        trace = tl.trace(model, [x])
        assert native_calls == 1, "capture must not run the fused kernel"
    finally:
        torch._native_multi_head_attention = original  # type: ignore[assignment]
    names = [op.func_name for op in trace.ops]
    assert "_native_multi_head_attention" not in names
    assert "scaled_dot_product_attention" in names
    rows = [
        row
        for row in capture_advisories(trace)
        if row["kind"] == ADVISORY_FUSED_ATTENTION_FASTPATH_HELD
    ]
    assert len(rows) == 1
    assert "attention" in rows[0]["message"]
    assert rows[0]["count"] == 1


def test_capture_runs_the_unfused_program_under_no_grad_too() -> None:
    """The recorded program no longer depends on the ambient grad mode."""

    _needs_switch()
    model, x = _model_and_input()
    with torch.no_grad():
        trace = tl.trace(model, [x])
    names = [op.func_name for op in trace.ops]
    assert "_native_multi_head_attention" not in names
    assert "scaled_dot_product_attention" in names


def test_forward_validation_passes_ground_truth_on_the_fast_path_row() -> None:
    _needs_switch()
    model, x = _model_and_input()
    assert tl.validate(model, [x], scope="forward"), last_validation_failure()


def test_switch_is_restored_after_capture_and_after_a_failing_forward() -> None:
    _needs_switch()
    model, x = _model_and_input()
    before = get_mha_fastpath_enabled()
    tl.trace(model, [x])
    assert get_mha_fastpath_enabled() == before

    class Boom(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention = nn.MultiheadAttention(WIDTH, 2, batch_first=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        tl.trace(Boom().eval(), [x])
    assert get_mha_fastpath_enabled() == before


def test_models_without_fused_attention_record_no_advisory_and_hold_nothing() -> None:
    model = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.ReLU())
    with hold_fused_attention_fastpath_off(model) as held:
        assert held == ()
    trace = tl.trace(model, [torch.randn(BATCH, WIDTH)])
    assert not [
        row
        for row in capture_advisories(trace)
        if row["kind"] == ADVISORY_FUSED_ATTENTION_FASTPATH_HELD
    ]


def test_hold_names_every_fused_fast_path_module() -> None:
    _needs_switch()
    model = nn.Sequential(
        nn.TransformerEncoderLayer(WIDTH, 2, dim_feedforward=16, batch_first=True),
        FastPathAttention(),
    ).eval()
    with hold_fused_attention_fastpath_off(model) as held:
        assert held == ("0", "0.self_attn", "1.attention")
        assert get_mha_fastpath_enabled() is False
