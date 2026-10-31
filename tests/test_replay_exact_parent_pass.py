"""Replay resolves a template's parent to the exact op the capture recorded.

Labeling renames each template parent ref to a final lookup label. When grouping fuses two
ops of one module call into a two-pass layer (a loop body, grouping rule row 2), later step indices shift, so the fused
layer's second pass keeps a public raw label (``add_2_5``) that is also the final label of
the next op. Resolving the ref as a lookup key then matched both ops and replay refused as
if they were two passes of one layer. The pass-exact label maps (``_final_to_raw_layer_labels``
then ``_raw_to_final_op_labels``) name one op; a trace without them still refuses.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import ReplayPreconditionError


class _ResidualPair(nn.Module):
    """One residual add run twice by a loop in one call: one two-pass layer.

    Grouping rule row 2: a loop in a submodule called once gives its body ops one
    pass per iteration. (Two residual adds written on two lines are two layers,
    row 10, and no longer fuse.)
    """

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for lin in (self.a, self.b):
            h = h + lin(h)
        return h


class _FusedThenAdd(nn.Module):
    """A fused residual pair, then an add whose final label is the second pass's raw label."""

    def __init__(self) -> None:
        super().__init__()
        self.block = _ResidualPair()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block(x) + 1.0
        return torch.rsqrt(h * h + 1.0)


def _capture() -> tl.Trace:
    torch.manual_seed(0)
    return tl.trace(
        _FusedThenAdd(),
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(replayable=True),
    )


def _colliding_label(trace: tl.Trace) -> str:
    """Return a final layer label that is also another op's public raw label."""

    final = {str(op.layer_label): str(op.label) for op in trace.layer_list}
    for op in trace.layer_list:
        other = final.get(str(op.raw_label))
        if other is not None and other != str(op.label):
            return str(op.raw_label)
    raise AssertionError("the toy no longer produces a raw/final label collision")


def _hold_rsqrt(trace: tl.Trace) -> tl.Trace:
    held = [str(op.label) for op in trace.layer_list if op.func_name == "rsqrt"]
    fork = trace.fork()
    fork.do(
        {label: tl.patch_from(trace) for label in held},
        intervention=tl.options.InterventionOptions(engine="replay"),
    )
    return fork


def test_replay_names_the_recorded_parent_when_labels_collide() -> None:
    trace = _capture()
    assert any(int(op.num_passes) > 1 for op in trace.layer_list)
    _colliding_label(trace)
    out = str(trace.output_layers[0])
    fork = _hold_rsqrt(trace)
    assert torch.equal(fork[out].out, trace[out].out)


def test_op_lookup_prefers_the_final_label_over_a_colliding_raw_label() -> None:
    """``trace.ops[label]`` names the op whose final label it is, not the op it was raw."""

    trace = _capture()
    key = _colliding_label(trace)
    assert trace.ops[key].layer_label == key


def test_trace_without_pass_exact_maps_still_refuses() -> None:
    """An older trace (no pass-exact maps) refuses rather than guess."""

    trace = _capture()
    _colliding_label(trace)
    trace._raw_to_final_op_labels = {}
    trace._final_to_raw_layer_labels = {}
    with pytest.raises(ReplayPreconditionError, match="refusing to guess a pass"):
        _hold_rsqrt(trace)


class TestTinyGemma2:
    """AG-3's tiny random Gemma-2: holding every norm and pattern replays exactly."""

    @pytest.fixture(scope="class")
    def setup(self):
        pytest.importorskip("transformers")
        from tests import _ag3_fixtures as fx

        model = fx.build_model(torch.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace = tl.trace(
                model,
                torch.tensor([fx.TOKENS]),
                capture=tl.options.CaptureOptions(replayable=True),
            )
        try:
            yield fx, model, trace
        finally:
            trace.cleanup()

    def test_replay_engine_holds_every_norm_and_pattern(self, setup) -> None:
        """Holding every rsqrt and softmax at its recorded value replays exactly."""

        _, _, trace = setup
        held = [str(op.label) for op in trace if op.func_name in ("rsqrt", "softmax")]
        assert held
        lm = str(trace.modules["lm_head"].output_ops[0])
        fork = trace.fork()
        fork.do(
            {label: tl.patch_from(trace) for label in held},
            intervention=tl.options.InterventionOptions(engine="replay"),
        )
        assert torch.equal(fork[lm].out, trace[lm].out)
