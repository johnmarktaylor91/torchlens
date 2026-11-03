"""Buffer threading for SINGLE-pass producers (AUD-CODE 2.8).

The replay overlay is keyed by the pass-qualified replay key (``copy_1_4:1``)
while a single-pass buffer record spells its writing parent BARE
(``copy_1_4``); the overlay miss fell into the "producer outside the cone"
branch, so a written buffer kept its captured value, downstream reads went
stale, and NO ``BufferThreadGapWarning`` fired. The sprint's threading
fixture was 3-pass (pass-qualified parents), which is why it passed.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.intervention.errors import BufferThreadGapWarning


class _Buf(nn.Module):
    """Write a buffer once from an activation, then read it where x is untouched."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.register_buffer("buf", torch.zeros(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.fc(x))
        self.buf.copy_(hidden.sum(0))
        return x + self.buf


@pytest.fixture(scope="module")
def buffered():
    torch.manual_seed(0)
    model = _Buf().eval()
    x = torch.randn(2, 4)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))
    try:
        yield model, x, trace
    finally:
        trace.cleanup()


def _written_buffer_op(trace: tl.Trace):
    return next(
        op
        for op in trace._layer_list
        if getattr(op, "is_buffer", False) and getattr(op, "parents", ())
    )


def test_single_pass_buffer_write_threads_zero_ablation(buffered) -> None:
    model, x, trace = buffered
    written = _written_buffer_op(trace)
    assert written.num_passes == 1
    captured_buffer = trace[written.label].out.clone()
    assert not torch.equal(captured_buffer, torch.zeros(4))
    fork = trace.fork()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fork.do(tl.label("relu_1_2"), tl.zero_ablate())
    assert not [w for w in caught if issubclass(w.category, BufferThreadGapWarning)]
    # hand truth: relu -> 0, buffer <- sum(0) of zeros = 0, output = x + 0
    assert torch.equal(fork[written.label].out, torch.zeros(4))
    assert torch.allclose(fork["output_1"].out, x, atol=1e-7)
    # capture truth untouched on the source
    assert torch.equal(trace[written.label].out, captured_buffer)


@pytest.mark.smoke
def test_single_pass_buffer_write_threads_scaled_value(buffered) -> None:
    model, x, trace = buffered
    written = _written_buffer_op(trace)
    fork = trace.fork()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fork.do(tl.label("relu_1_2"), tl.scale(0.5))
    assert not [w for w in caught if issubclass(w.category, BufferThreadGapWarning)]
    with torch.no_grad():
        expected_buffer = (0.5 * torch.relu(model.fc(x))).sum(0)
    assert torch.allclose(fork[written.label].out, expected_buffer, atol=1e-6)
    assert torch.allclose(fork["output_1"].out, x + expected_buffer, atol=1e-6)
    assert (
        written.label.split(":")[0] in fork.last_run["cone"]
        or written.label in fork.last_run["cone"]
    )


class _EvalNorm(nn.Module):
    """Conv -> BatchNorm; in eval mode the norm reads its running statistics only."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 3, 1)
        self.bn = nn.BatchNorm2d(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.bn(self.conv(x)))


class _UnchangedNormWrite(nn.Module):
    """Conv -> a training-mode ``batch_norm`` with momentum 0.

    The call DECLARES a fused write of its running statistics (``training`` is
    true), and momentum 0 keeps their bytes exactly unchanged.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 3, 1)
        self.register_buffer("running_mean", torch.zeros(3))
        self.register_buffer("running_var", torch.ones(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self._norm(self.conv(x)))

    def _norm(self, h: torch.Tensor) -> torch.Tensor:
        return nn.functional.batch_norm(
            h, self.running_mean, self.running_var, training=True, momentum=0.0
        )


def test_eval_mode_norm_journals_no_buffer_write() -> None:
    """An eval-mode BatchNorm reads its running statistics and writes no buffer version."""

    model = _EvalNorm().eval()
    trace = tl.trace(model, torch.randn(2, 2, 4, 4))
    assert not [op.label for op in trace._layer_list if getattr(op, "buffer_write_kind", None)]
    assert all(not op.parents for op in trace._layer_list if getattr(op, "is_buffer", False))


def test_proven_unchanged_fused_buffer_write_in_cone_is_silent() -> None:
    """A fused norm write whose bytes never change stays silent inside the cone.

    A declared fused write with ``buffer_value_changed=False`` (bytes provably
    unchanged) resolves its single-pass producer inside the cone (2.8); a
    proven-unchanged write is capture truth, not a threading gap, and the
    replayed output must match an eager rerun of the ablated model bit-exactly.
    """

    torch.manual_seed(0)
    model = _UnchangedNormWrite().eval()
    with torch.no_grad():
        model.running_mean.uniform_(-1, 1)
        model.running_var.uniform_(0.5, 2)
    x = torch.randn(2, 2, 4, 4)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))
    try:
        written = _written_buffer_op(trace)
        assert written.buffer_write_kind == "fused"
        assert written.buffer_value_changed is False
        fork = trace.fork()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fork.do(tl.label("conv2d_1_1"), tl.zero_ablate())
        assert not [w for w in caught if issubclass(w.category, BufferThreadGapWarning)]
        with torch.no_grad():
            expected = torch.relu(model._norm(torch.zeros_like(model.conv(x))))
        assert torch.equal(fork["output_1"].out, expected)
        assert torch.equal(fork[written.label].out, trace[written.label].out)
    finally:
        trace.cleanup()


@pytest.mark.parametrize("value_changed", [True, None])
def test_unproven_fused_buffer_write_in_cone_still_discloses(value_changed) -> None:
    """The fail-closed disclosure survives: fused + changed/unknown bytes still warns.

    Only ``buffer_value_changed is False`` (bytes provably unchanged) is
    silent; ``True`` and ``None`` (unknown) keep the gap disclosure.
    """

    torch.manual_seed(0)
    model = _UnchangedNormWrite().eval()
    x = torch.randn(2, 2, 4, 4)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))
    try:
        written = _written_buffer_op(trace)
        assert written.buffer_write_kind == "fused"
        fork = trace.fork()
        fork._layer_dict_all_keys[written.label]._internal_set(
            "buffer_value_changed", value_changed
        )
        with pytest.warns(BufferThreadGapWarning, match="does not prove"):
            fork.do(tl.label("conv2d_1_1"), tl.zero_ablate())
    finally:
        trace.cleanup()
