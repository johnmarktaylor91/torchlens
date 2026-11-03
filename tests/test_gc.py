"""GC and memory leak tests for TorchLens.

Verifies that Trace, Op, Param, and model parameters
are garbage-collectible after use / cleanup.

These are marked ``smoke``: a lifetime regression is invisible to call-count,
``tracemalloc``, and wall-clock gates, so this file is the only per-step gate
that can see one. It is fast (~5 s) and it has caught the same module-global
cache root twice now.
"""

import gc
import tracemalloc
import weakref

import pytest
import torch
from torch import nn

import torchlens as tl

# Marks are PER-TEST: four session-state-scaling gc tests are re-tiered
# `heavy` below (charged 9.4-19.3s min(wall, cpu) in the merged full
# not-slow session -- their explicit gc.collect() pays for the whole
# session's accumulated cycles -- while running 0.6-1.5s isolated). Marks
# are additive, so a file-level smoke pytestmark would drag them back
# into the smoke tier; the remaining leak gates stay smoke-marked.


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


class _SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 3)

    def forward(self, x):
        return self.fc(x)


class _TwoLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 4)
        self.fc2 = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTraceGC:
    def test_fast_live_hooks_finalize_when_trace_is_deleted(self):
        """Deleting a fast Trace removes its hooks and leaves parameters collectible."""

        model = _TwoLayerNet().eval()
        trace = tl.trace(model, torch.randn(1, 5), save=tl.module("fc1"))
        trace.run(inputs=torch.randn(1, 5), fast=True)
        trace_ref = weakref.ref(trace)
        param_ref = weakref.ref(next(model.parameters()))
        assert model.fc1._forward_hooks

        del trace
        gc.collect()

        assert trace_ref() is None
        assert not model.fc1._forward_hooks
        del model
        gc.collect()
        assert param_ref() is None

    @pytest.mark.heavy
    def test_save_new_outs_no_leak(self):
        """5x save_new_outs should not leak memory."""
        model = _TwoLayerNet()
        x = torch.randn(1, 5)
        # Two-pass path: exhaustive first, then fast via save_new_outs
        trace = tl.trace(model, x, save=None)

        # Warm up
        trace._save_new_outs(model, torch.randn(1, 5), save="all")
        gc.collect()

        tracemalloc.start()
        baseline = tracemalloc.take_snapshot()

        for _ in range(5):
            trace._save_new_outs(model, torch.randn(1, 5), save="all")
            gc.collect()

        after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        stats = after.compare_to(baseline, "lineno")
        tl_growth = sum(s.size_diff for s in stats if "torchlens" in str(s.traceback))
        assert tl_growth < 256 * 1024, f"Memory grew by {tl_growth} bytes across 5 save_new_outs"
        trace.cleanup()

    def test_transient_data_cleared(self):
        """Verify module build scratch is removed after postprocess."""
        model = _TwoLayerNet()
        trace = tl.trace(model, torch.randn(1, 5))
        assert "_build_state" not in trace.__dict__
        assert "_raw_graph_ws" not in trace.__dict__
        assert "_module_capture_ws" not in trace.__dict__
        assert "_wrapper_runtime_ws" not in trace.__dict__
        assert not hasattr(trace, "_module_build_data")
        assert not hasattr(trace, "_module_metadata")
        assert not hasattr(trace, "_module_forward_args")
        trace.cleanup()

    def test_raw_layer_dict_cleared_after_cleanup(self):
        """Verify raw layer scratch is absent after postprocess and cleanup."""
        model = _TwoLayerNet()
        trace = tl.trace(model, torch.randn(1, 5))
        assert "_build_state" not in trace.__dict__
        assert "_raw_graph_ws" not in trace.__dict__
        assert "_module_capture_ws" not in trace.__dict__
        assert "_wrapper_runtime_ws" not in trace.__dict__
        assert not hasattr(trace, "_raw_layer_dict")
        trace.cleanup()
        assert not hasattr(trace, "_raw_layer_dict")

    def test_live_run_after_model_collection_refuses_typed_and_names_the_weak_ref(self):
        """A collected source model refuses run() typed AND discloses why (R37).

        The refusal itself is correct by design (the trace holds its model
        weakly), but it used to be undocumented and its message named neither
        the weak reference nor a remedy -- the plainest documented idiom
        ``tl.trace(Model(), x)`` then failed gc-timing-dependently with no
        actionable explanation.
        """

        from torchlens.errors import RunCapabilityUnavailableError

        trace = tl.trace(_SimpleLinear(), torch.randn(1, 5))
        gc.collect()
        assert trace._source_model_ref() is None, "inline model should be collected"
        with pytest.raises(RunCapabilityUnavailableError) as excinfo:
            trace.run(inputs=torch.randn(1, 5))
        message = str(excinfo.value)
        assert "weakly" in message and "strong reference" in message, (
            "the collected-model refusal must disclose the weak-reference "
            f"dependency and its remedy; got: {message}"
        )
        with pytest.raises(RunCapabilityUnavailableError) as fast_excinfo:
            trace.run(inputs=torch.randn(1, 5), fast=True)
        assert "weakly" in str(fast_excinfo.value)


def test_cleaned_trace_refuses_typed_and_settles_unknown():
    """A husked Trace refuses public reads TYPED and settles outcome UNKNOWN.

    b6-opus R25: seven public reads raised raw AttributeError naming
    whichever private field they touched first (``_tracing_finished``,
    ``_layers_logged``, ``layer_list``), ``Trace.outcome`` returned ``None``
    (outside the frozen vocabulary), and a second ``cleanup()`` crashed on
    the first cleanup's own output.
    """

    from torchlens._capture.outcome import CaptureStatus
    from torchlens._errors import TraceCleanedUpError

    trace = tl.trace(_TwoLayerNet(), torch.randn(2, 5))
    assert trace.outcome is not None and trace.outcome.status is CaptureStatus.COMPLETE
    trace.cleanup()

    # One typed code for every public reader.
    readers = {
        "summary": lambda: trace.summary(),
        "iteration": lambda: list(trace),
        "getitem": lambda: trace["relu_1_2"],
        "draw": lambda: trace.draw(),
        "receptive_fields": lambda: trace.receptive_fields(),
    }
    for name, reader in readers.items():
        with pytest.raises(TraceCleanedUpError) as exc_info:
            reader()
        assert exc_info.value.fields["code"] == "trace_cleaned_up", name
        assert exc_info.value.fields["remedy"], name

    # AttributeError lineage keeps hasattr/getattr-default degrade paths.
    assert getattr(trace, "_layer_list", None) is None
    assert not hasattr(trace, "_tracing_finished")

    # The settled outcome is UNKNOWN (most restrictive), never None.
    assert trace.outcome is not None
    assert trace.outcome.status is CaptureStatus.UNKNOWN

    # Idempotent teardown and surviving diagnostics.
    trace.cleanup()
    assert repr(trace)
    assert trace.explain()


def test_merged_trace_release_drops_member_traces():
    """MergedTrace.cleanup() unpins the rank traces it held strongly (b2:B20).

    Regression guard for the presenter's lifetime contract: without release(),
    a presenter over live traces transitively pinned every member (and its
    activations) with no counterpart to Trace.cleanup().
    """

    from torchlens._merged._presenter import MergedTrace, _RankHandle

    trace_a = tl.trace(_SimpleLinear(), torch.randn(2, 5))
    trace_b = tl.trace(_SimpleLinear(), torch.randn(2, 5))
    refs = [weakref.ref(trace_a), weakref.ref(trace_b)]
    merged = MergedTrace(
        derivation=None,  # placeholder: release() must not need the derivation
        rank_handles={0: _RankHandle(0, trace=trace_a), 1: _RankHandle(1, trace=trace_b)},
    )
    del trace_a, trace_b
    gc.collect()
    assert all(ref() is not None for ref in refs), "presenter should pin members"

    merged.cleanup()
    gc.collect()
    gc.collect()
    assert all(ref() is None for ref in refs), "release() left a member pinned"
    merged.cleanup()  # idempotent
