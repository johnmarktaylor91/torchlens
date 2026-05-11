"""GC and memory leak tests for TorchLens.

Verifies that Trace, OpLog, ParamLog, and model parameters
are garbage-collectible after use / cleanup.
"""

import gc
import tracemalloc
import weakref

import pytest
import torch
import torchlens as tl
from torch import nn

from torchlens import trace as trace_fn


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
    def test_trace_gc_without_cleanup(self):
        """del trace; gc.collect() should release the Trace."""
        model = _SimpleLinear()
        trace = tl.trace(model, torch.randn(1, 5))
        ref = weakref.ref(trace)
        del trace
        gc.collect()
        assert ref() is None

    def test_trace_gc_with_cleanup(self):
        """cleanup() + del + gc.collect() should release the Trace."""
        model = _SimpleLinear()
        trace = tl.trace(model, torch.randn(1, 5))
        ref = weakref.ref(trace)
        trace.cleanup()
        del trace
        gc.collect()
        assert ref() is None

    def test_model_params_not_pinned_after_cleanup(self):
        """After cleanup + del trace, model params should be GC-able."""
        model = _SimpleLinear()
        param_ref = weakref.ref(list(model.parameters())[0])
        trace = tl.trace(model, torch.randn(1, 5))
        trace.cleanup()
        del trace
        del model
        gc.collect()
        assert param_ref() is None

    def test_model_gc_after_release_param_refs(self):
        """release_param_refs() then del model -> model GC'd while log alive."""
        model = _TwoLayerNet()
        model_ref = weakref.ref(model)
        trace = tl.trace(model, torch.randn(1, 5))
        trace.release_param_refs()
        del model
        gc.collect()
        assert model_ref() is None
        # trace is still usable
        assert len(trace) > 0
        trace.cleanup()

    def test_no_memory_growth_across_sessions(self):
        """5x trace + del should not leak memory."""
        model = _TwoLayerNet()
        x = torch.randn(1, 5)
        # Warm up
        ml = trace_fn(model, x)
        ml.cleanup()
        del ml
        gc.collect()

        tracemalloc.start()
        baseline = tracemalloc.take_snapshot()

        for _ in range(5):
            ml = trace_fn(model, x)
            ml.cleanup()
            del ml
            gc.collect()

        after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Compare: filter to torchlens allocations
        stats = after.compare_to(baseline, "lineno")
        tl_growth = sum(s.size_diff for s in stats if "torchlens" in str(s.traceback))
        # Allow up to 256KB of noise (caches, interned strings, etc.)
        assert tl_growth < 256 * 1024, f"Memory grew by {tl_growth} bytes across 5 sessions"

    def test_save_new_outs_no_leak(self):
        """5x save_new_outs should not leak memory."""
        model = _TwoLayerNet()
        x = torch.randn(1, 5)
        # Two-pass path: exhaustive first, then fast via save_new_outs
        trace = tl.trace(model, x, layers_to_save=None)

        # Warm up
        trace.save_new_outs(model, torch.randn(1, 5), layers_to_save="all")
        gc.collect()

        tracemalloc.start()
        baseline = tracemalloc.take_snapshot()

        for _ in range(5):
            trace.save_new_outs(model, torch.randn(1, 5), layers_to_save="all")
            gc.collect()

        after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        stats = after.compare_to(baseline, "lineno")
        tl_growth = sum(s.size_diff for s in stats if "torchlens" in str(s.traceback))
        assert tl_growth < 256 * 1024, f"Memory grew by {tl_growth} bytes across 5 save_new_outs"
        trace.cleanup()

    def test_cleanup_breaks_param_ref(self):
        """After cleanup, all ParamLog._param_ref should be None."""
        model = _TwoLayerNet()
        trace = tl.trace(model, torch.randn(1, 5))
        param_logs = list(trace.param_logs)
        trace.cleanup()
        for pl in param_logs:
            assert pl._param_ref is None

    def test_release_param_refs_preserves_grad_metadata(self):
        """backward(), release_param_refs(), verify grad info is cached."""
        model = _TwoLayerNet()
        x = torch.randn(1, 5)
        trace = tl.trace(model, x, save_grads=True)
        # Run backward to populate grads
        out = model(x)
        out.sum().backward()
        # Access grad metadata to cache it
        for pl in trace.param_logs:
            pl._check_param_grad()
        # Now release
        trace.release_param_refs()
        # Grad metadata should still be accessible
        has_any_grad = False
        for pl in trace.param_logs:
            assert pl._param_ref is None
            if pl._has_grad:
                has_any_grad = True
                assert pl._grad_shape is not None
                assert pl._grad_dtype is not None
                assert pl._grad_memory > 0
        assert has_any_grad, "Expected at least one param to have grad metadata cached"
        trace.cleanup()

    def test_transient_data_cleared(self):
        """Verify module build scratch is removed after postprocess."""
        model = _TwoLayerNet()
        trace = tl.trace(model, torch.randn(1, 5))
        assert "_build_state" not in trace.__dict__
        assert not hasattr(trace, "_module_build_data")
        assert not hasattr(trace, "_module_metadata")
        assert not hasattr(trace, "_module_forward_args")
        trace.cleanup()

    def test_raw_layer_dict_cleared_after_cleanup(self):
        """Verify raw layer scratch is absent after postprocess and cleanup."""
        model = _TwoLayerNet()
        trace = tl.trace(model, torch.randn(1, 5))
        assert "_build_state" not in trace.__dict__
        assert not hasattr(trace, "_raw_layer_dict")
        trace.cleanup()
        assert not hasattr(trace, "_raw_layer_dict")
