"""GC and memory leak tests for TorchLens.

Verifies that ModelLog, LayerPassLog, ParamLog, and model parameters
are garbage-collectible after use / cleanup.
"""

import gc
import tracemalloc
import weakref

import pytest
import torch
from torch import nn

from torchlens import log_forward_pass


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


class TestModelLogGC:
    def test_model_log_gc_without_cleanup(self):
        """del model_log; gc.collect() should release the ModelLog."""
        model = _SimpleLinear()
        model_log = log_forward_pass(model, torch.randn(1, 5))
        ref = weakref.ref(model_log)
        del model_log
        gc.collect()
        assert ref() is None

    def test_model_log_gc_with_cleanup(self):
        """cleanup() + del + gc.collect() should release the ModelLog."""
        model = _SimpleLinear()
        model_log = log_forward_pass(model, torch.randn(1, 5))
        ref = weakref.ref(model_log)
        model_log.cleanup()
        del model_log
        gc.collect()
        assert ref() is None

    def test_model_params_not_pinned_after_cleanup(self):
        """After cleanup + del model_log, model params should be GC-able."""
        model = _SimpleLinear()
        param_ref = weakref.ref(list(model.parameters())[0])
        model_log = log_forward_pass(model, torch.randn(1, 5))
        model_log.cleanup()
        del model_log
        del model
        gc.collect()
        assert param_ref() is None

    def test_model_gc_after_release_param_refs(self):
        """release_param_refs() then del model -> model GC'd while log alive."""
        model = _TwoLayerNet()
        model_ref = weakref.ref(model)
        model_log = log_forward_pass(model, torch.randn(1, 5))
        model_log.release_param_refs()
        del model
        gc.collect()
        assert model_ref() is None
        # model_log is still usable
        assert len(model_log) > 0
        model_log.cleanup()

    def test_no_memory_growth_across_sessions(self):
        """5x log_forward_pass + del should not leak memory."""
        model = _TwoLayerNet()
        x = torch.randn(1, 5)
        # Warm up
        ml = log_forward_pass(model, x)
        ml.cleanup()
        del ml
        gc.collect()

        tracemalloc.start()
        baseline = tracemalloc.take_snapshot()

        for _ in range(5):
            ml = log_forward_pass(model, x)
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

    def test_save_new_activations_no_leak(self):
        """5x save_new_activations should not leak memory."""
        model = _TwoLayerNet()
        x = torch.randn(1, 5)
        # Two-pass path: exhaustive first, then fast via save_new_activations
        model_log = log_forward_pass(model, x, layers_to_save=None)

        # Warm up
        model_log.save_new_activations(model, torch.randn(1, 5), layers_to_save="all")
        gc.collect()

        tracemalloc.start()
        baseline = tracemalloc.take_snapshot()

        for _ in range(5):
            model_log.save_new_activations(model, torch.randn(1, 5), layers_to_save="all")
            gc.collect()

        after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        stats = after.compare_to(baseline, "lineno")
        tl_growth = sum(s.size_diff for s in stats if "torchlens" in str(s.traceback))
        assert tl_growth < 256 * 1024, (
            f"Memory grew by {tl_growth} bytes across 5 save_new_activations"
        )
        model_log.cleanup()

    def test_cleanup_breaks_param_ref(self):
        """After cleanup, all ParamLog._param_ref should be None."""
        model = _TwoLayerNet()
        model_log = log_forward_pass(model, torch.randn(1, 5))
        param_logs = list(model_log.param_logs)
        model_log.cleanup()
        for pl in param_logs:
            assert pl._param_ref is None

    def test_release_param_refs_preserves_grad_metadata(self):
        """backward(), release_param_refs(), verify grad info is cached."""
        model = _TwoLayerNet()
        x = torch.randn(1, 5)
        model_log = log_forward_pass(model, x, save_gradients=True)
        # Run backward to populate gradients
        out = model(x)
        out.sum().backward()
        # Access grad metadata to cache it
        for pl in model_log.param_logs:
            pl._check_param_grad()
        # Now release
        model_log.release_param_refs()
        # Grad metadata should still be accessible
        has_any_grad = False
        for pl in model_log.param_logs:
            assert pl._param_ref is None
            if pl._has_grad:
                has_any_grad = True
                assert pl._grad_shape is not None
                assert pl._grad_dtype is not None
                assert pl._grad_fsize > 0
        assert has_any_grad, "Expected at least one param to have grad metadata cached"
        model_log.cleanup()

    def test_transient_data_cleared(self):
        """Verify _module_build_data/metadata/forward_args are cleared after postprocess."""
        model = _TwoLayerNet()
        model_log = log_forward_pass(model, torch.randn(1, 5))
        # _module_build_data should be empty after postprocessing
        assert len(model_log._module_build_data) == 0 or all(
            (len(v) == 0 if hasattr(v, "__len__") else True)
            for v in model_log._module_build_data.values()
        )
        # _module_metadata and _module_forward_args consumed by _build_module_logs
        assert len(model_log._module_metadata) == 0
        assert len(model_log._module_forward_args) == 0
        model_log.cleanup()

    def test_raw_layer_dict_cleared_after_cleanup(self):
        """Verify _raw_layer_dict is cleared after cleanup()."""
        model = _TwoLayerNet()
        model_log = log_forward_pass(model, torch.randn(1, 5))
        # _raw_layer_dict is populated during capture; still present after log_forward_pass
        assert len(model_log._raw_layer_dict) > 0
        model_log.cleanup()
        assert not hasattr(model_log, "_raw_layer_dict")
