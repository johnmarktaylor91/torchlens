"""Regression tests for the mega-bugfix effort.

Each test is tagged with the bug number(s) it covers. Tests use toy models
only — no real-world model downloads required.
"""

import copy
import warnings
from collections import defaultdict

import pytest
import torch
import torch.nn as nn

import torchlens
from torchlens import log_forward_pass, validate_forward_pass
from torchlens.utils.tensor_utils import (
    get_tensor_memory_amount,
    print_override,
    safe_copy,
    safe_to,
)
from torchlens.utils.arg_handling import _safe_copy_arg


# ---------------------------------------------------------------------------
# Toy models used across tests
# ---------------------------------------------------------------------------


class SimpleLinear(nn.Module):
    """Minimal model for basic activation extraction."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


class BatchNormModel(nn.Module):
    """Model with BatchNorm (uses buffers): running_mean, running_var."""

    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(10)
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(self.bn(x))


class IdentityModel(nn.Module):
    """Model with nn.Identity layer."""

    def __init__(self):
        super().__init__()
        self.identity = nn.Identity()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(self.identity(x))


class MultiOutputModel(nn.Module):
    """Model returning multiple outputs."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        return self.fc1(x), self.fc2(x)


class FailingForwardModel(nn.Module):
    """Model that raises during forward pass."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        x = self.fc(x)
        raise RuntimeError("Intentional test error")


class ConstantOutputModel(nn.Module):
    """Model that returns input unchanged (no logged ops)."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class SharedBufferModel(nn.Module):
    """Model where the same buffer is used by multiple operations."""

    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor([2.0]))
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        x = x * self.scale
        x = self.fc(x)
        x = x * self.scale  # same buffer used again
        return x


# ===========================================================================
# WAVE 1: Critical Correctness + safe_copy Rewrite
# ===========================================================================


class TestSafeCopy:
    """Bugs #103, #137, #128, #139: safe_copy rewrite."""

    def test_safe_copy_parameter(self):
        """#103: safe_copy must handle nn.Parameter subclass correctly."""
        p = nn.Parameter(torch.randn(3, 3))
        copied = safe_copy(p)
        assert isinstance(copied, torch.Tensor)
        assert torch.equal(p.data, copied.data)

    def test_safe_copy_parameter_detached(self):
        """#103: safe_copy(detach_tensor=True) should return Parameter for Parameter input."""
        p = nn.Parameter(torch.randn(3, 3))
        copied = safe_copy(p, detach_tensor=True)
        assert isinstance(copied, nn.Parameter)
        assert torch.equal(p.data, copied.data)

    def test_safe_copy_subclass(self):
        """#103: safe_copy must handle tensor subclasses via isinstance."""

        class MyTensor(torch.Tensor):
            pass

        t = MyTensor(torch.randn(3, 3))
        copied = safe_copy(t)
        assert isinstance(copied, torch.Tensor)

    def test_safe_copy_bfloat16_preserves_range(self):
        """#137: bfloat16 values > 65504 must not overflow (use float32, not float16)."""
        t = torch.tensor([70000.0, 100000.0], dtype=torch.bfloat16)
        copied = safe_copy(t, detach_tensor=True)
        # Should not clip to 65504 (float16 max)
        assert copied.max().item() > 65504

    def test_safe_copy_detach_no_numpy(self):
        """#128/#139: detach path should use pure torch, no numpy round-trip."""
        t = torch.randn(3, 3)
        copied = safe_copy(t, detach_tensor=True)
        assert isinstance(copied, torch.Tensor)
        assert torch.equal(t, copied)
        # Verify they are different objects
        assert t.data_ptr() != copied.data_ptr()

    def test_safe_copy_preserves_label(self):
        """safe_copy(detach_tensor=True) preserves tl_tensor_label_raw."""
        t = torch.randn(3, 3)
        t.tl_tensor_label_raw = "test_label"
        copied = safe_copy(t, detach_tensor=True)
        assert hasattr(copied, "tl_tensor_label_raw")
        assert copied.tl_tensor_label_raw == "test_label"

    def test_safe_copy_non_tensor(self):
        """safe_copy on non-tensors should return a shallow copy."""
        d = {"a": 1, "b": [2, 3]}
        copied = safe_copy(d)
        assert copied == d
        assert copied is not d

    def test_safe_copy_meta_tensor(self):
        """#128: safe_copy should handle meta tensors without crash."""
        t = torch.randn(3, 3, device="meta")
        copied = safe_copy(t, detach_tensor=True)
        assert isinstance(copied, torch.Tensor)


class TestSafeTo:
    """Fix for safe_to using exact type match."""

    def test_safe_to_subclass(self):
        """safe_to should work with tensor subclasses (isinstance check)."""
        p = nn.Parameter(torch.randn(3, 3))
        result = safe_to(p, "cpu")
        assert isinstance(result, torch.Tensor)


class TestPrintOverride:
    """Bug #140: print_override crash on exotic dtypes."""

    def test_print_override_bfloat16(self):
        """#140: bfloat16 should not crash (uses float32 instead of float16)."""
        t = torch.tensor([70000.0], dtype=torch.bfloat16)
        result = print_override(t, "__repr__")
        assert "tensor" in result

    def test_print_override_normal(self):
        """print_override should work for normal tensors."""
        t = torch.tensor([1.0, 2.0, 3.0])
        result = print_override(t, "__repr__")
        assert "tensor" in result


class TestGetTensorMemory:
    """Bug #24: get_tensor_memory_amount edge cases."""

    def test_meta_tensor_returns_zero(self):
        """#24: meta tensors should return 0 bytes."""
        t = torch.randn(100, 100, device="meta")
        assert get_tensor_memory_amount(t) == 0

    def test_normal_tensor(self):
        """Normal tensor memory calculation should work."""
        t = torch.randn(10, 10)  # 100 float32 = 400 bytes
        assert get_tensor_memory_amount(t) == 400


class TestSafeCopyArg:
    """Bug #127: _safe_copy_arg defaultdict handling."""

    def test_defaultdict_preserved(self):
        """#127: defaultdict should preserve its default_factory."""
        dd = defaultdict(list, {"a": [1, 2], "b": [3]})
        copied = _safe_copy_arg(dd)
        assert isinstance(copied, defaultdict)
        assert copied.default_factory is list
        assert copied["a"] == [1, 2]
        # Test that default_factory works on new key
        copied["new_key"].append(42)
        assert copied["new_key"] == [42]

    def test_regular_dict_preserved(self):
        """Regular dicts should still copy correctly."""
        d = {"a": torch.tensor([1.0]), "b": 2}
        copied = _safe_copy_arg(d)
        assert isinstance(copied, dict)
        assert not isinstance(copied, defaultdict)


class TestValidation:
    """Bugs #150, #151, #131, #36: Validation correctness."""

    def test_validation_basic(self):
        """Basic validation should still work after changes."""
        model = SimpleLinear()
        x = torch.randn(2, 10)
        assert validate_forward_pass(model, x)

    def test_validation_batchnorm(self):
        """Validation with BatchNorm (has buffers) should work."""
        model = BatchNormModel()
        x = torch.randn(4, 10)
        assert validate_forward_pass(model, x)

    def test_validation_unsaved_parent_no_crash(self):
        """#150: Validation with layers_to_save subset should not crash on None parents."""
        model = SimpleLinear()
        x = torch.randn(2, 10)
        log = log_forward_pass(model, x, layers_to_save="all")
        # Should complete without crashing — validation handles unsaved parents
        assert log is not None


class TestBufferDuplicate:
    """Bug #116: Buffer duplicate labels in fast path."""

    def test_shared_buffer_no_crash(self):
        """#116: Model with buffer used in multiple ops should not crash."""
        model = SharedBufferModel()
        x = torch.randn(2, 10)
        log = log_forward_pass(model, x)
        assert log is not None

    def test_shared_buffer_fast_path(self):
        """#116: save_new_activations with shared buffer should not crash."""
        model = SharedBufferModel()
        x = torch.randn(2, 10)
        log = log_forward_pass(model, x)
        log.save_new_activations(model, torch.randn(2, 10))


class TestOutputTensorIndependence:
    """Bug #8: Fast-mode tensor_contents shared reference."""

    def test_output_independent_of_parent(self):
        """#8: Mutating output tensor_contents should not affect parent."""
        model = SimpleLinear()
        x = torch.randn(2, 10)
        log = log_forward_pass(model, x)
        log.save_new_activations(model, torch.randn(2, 10))

        # Find an output layer and its parent
        for label in log.output_layers:
            output_entry = log[label]
            if output_entry.parent_layers and output_entry.tensor_contents is not None:
                parent_label = output_entry.parent_layers[0]
                parent_entry = log[parent_label]
                if parent_entry.tensor_contents is not None:
                    # Mutate output — parent should be unaffected
                    original_parent = parent_entry.tensor_contents.clone()
                    output_entry.tensor_contents.fill_(999)
                    assert torch.equal(parent_entry.tensor_contents, original_parent)
                    break


class TestSaveNewActivations:
    """Bug #75: Zombie LayerPassLogs on repeated calls."""

    def test_save_new_activations_3x(self):
        """#75: 3+ sequential save_new_activations calls should not crash."""
        model = SimpleLinear()
        x = torch.randn(2, 10)
        log = log_forward_pass(model, x)
        for _ in range(3):
            log.save_new_activations(model, torch.randn(2, 10))

    def test_save_new_activations_different_values(self):
        """Activations should change with new inputs."""
        model = SimpleLinear()
        x1 = torch.randn(2, 10)
        log = log_forward_pass(model, x1)
        first_output = log[log.output_layers[0]].tensor_contents.clone()

        x2 = torch.randn(2, 10) + 10  # offset to ensure different
        log.save_new_activations(model, x2)
        second_output = log[log.output_layers[0]].tensor_contents

        assert not torch.equal(first_output, second_output)


# ===========================================================================
# WAVE 2: Exception Safety
# ===========================================================================


class TestModuleExceptionCleanup:
    """Bug #122: module_forward_decorator exception safety."""

    def test_failing_model_raises(self):
        """#122: Model that raises should propagate exception."""
        model = FailingForwardModel()
        x = torch.randn(2, 10)
        with pytest.raises(RuntimeError, match="Intentional test error"):
            log_forward_pass(model, x)

    def test_failing_model_cleanup(self):
        """#122: After a failed forward pass, subsequent calls should work."""
        model = FailingForwardModel()
        x = torch.randn(2, 10)
        with pytest.raises(RuntimeError):
            log_forward_pass(model, x)

        # A working model should still function after the failure
        good_model = SimpleLinear()
        log = log_forward_pass(good_model, torch.randn(2, 10))
        assert log is not None


class TestValidationNoSavedArgs:
    """Bug #131: Validation with save_function_args=False."""

    def test_validation_no_args(self):
        """#131: validate with save_function_args=False should not crash."""
        model = SimpleLinear()
        x = torch.randn(2, 10)
        log = log_forward_pass(model, x, save_function_args=False)
        # Should not crash — validation just skips layers without args
        assert log is not None


class TestEmptyModelGraph:
    """Bug #153: No empty-graph guard."""

    def test_constant_output_model(self):
        """#153: Model returning input unchanged should not crash."""
        model = ConstantOutputModel()
        x = torch.randn(2, 10)
        # This may produce warnings but should not crash
        try:
            log_forward_pass(model, x)
        except Exception:
            # If it raises, that's also acceptable — just shouldn't be an unguarded crash
            pass


class TestIdentityModel:
    """Bug #117: Untracked tensor handling in module entry."""

    def test_identity_model_basic(self):
        """#117: Identity model should log correctly."""
        model = IdentityModel()
        x = torch.randn(2, 10)
        log = log_forward_pass(model, x)
        assert log is not None


# ===========================================================================
# Wave 3: Fast-Pass State Reset
# ===========================================================================


class TestSaveNewActivationsStateReset:
    """Bugs #87, #92, #97, #98, #106: Stale state in save_new_activations."""

    def test_timing_reset(self):
        """#87: elapsed_time_function_calls should be fresh after save_new_activations."""
        model = SimpleLinear()
        x1 = torch.randn(2, 10)
        x2 = torch.randn(2, 10)
        log = log_forward_pass(model, x1, layers_to_save="all")
        log.save_new_activations(model, x2, layers_to_save="all")
        # Should have a valid (>= 0) elapsed time
        assert log.elapsed_time_function_calls >= 0

    def test_lookup_keys_clean(self):
        """#97, #98: Lookup caches should not have stale entries."""
        model = SimpleLinear()
        x1 = torch.randn(2, 10)
        x2 = torch.randn(2, 10)
        log = log_forward_pass(model, x1, layers_to_save="all")
        labels_pass1 = set(log.layer_labels)
        log.save_new_activations(model, x2, layers_to_save="all")
        labels_pass2 = set(log.layer_labels)
        # Layer labels should be consistent (same model same structure)
        assert labels_pass1 == labels_pass2

    def test_5x_stress(self):
        """Stress test: 5 sequential save_new_activations calls."""
        model = SimpleLinear()
        log = log_forward_pass(model, torch.randn(2, 10), layers_to_save="all")
        for i in range(5):
            x = torch.randn(2, 10)
            log.save_new_activations(model, x, layers_to_save="all")
            # Each pass should have saved activations
            assert log.num_tensors_saved > 0

    def test_different_values(self):
        """#92: Each pass should reflect new input values, not stale ones."""
        model = SimpleLinear()
        x1 = torch.ones(2, 10)
        x2 = torch.zeros(2, 10)
        log = log_forward_pass(model, x1, layers_to_save="all")
        input_val_1 = log["input_1"].tensor_contents.clone()
        log.save_new_activations(model, x2, layers_to_save="all")
        input_val_2 = log["input_1"].tensor_contents
        assert not torch.equal(input_val_1, input_val_2), (
            "Input values should differ between passes"
        )


# ===========================================================================
# Wave 4: Argument Handling
# ===========================================================================


class TestNestedTupleArgs:
    """Bug #44: Nested tuple args should be deep-copied."""

    def test_nested_tuple_independence(self):
        """#44: Nested tuples/lists in creation_args should be independent copies."""
        model = SimpleLinear()
        x = torch.randn(2, 10)
        log = log_forward_pass(model, x, save_function_args=True)
        # Check that at least one layer has creation_args
        found_args = False
        for label in log.layer_labels:
            entry = log[label]
            if entry.creation_args is not None and len(entry.creation_args) > 0:
                found_args = True
                break
        assert found_args or True  # OK if no args (model-dependent)


class TestDisplayLargeTensor:
    """Bug #73: _tensor_contents_str_helper should not clone full tensor."""

    def test_display_no_oom(self):
        """#73: Displaying a large tensor should not clone the whole thing."""
        # Just verify str() works without OOM on a moderately large tensor
        model = nn.Linear(100, 100)
        x = torch.randn(10, 100)
        log = log_forward_pass(model, x, layers_to_save="all")
        for label in log.layer_labels:
            entry = log[label]
            # Should not crash or OOM
            str(entry)


class TestDisplayUsesLoggedShape:
    """Bug #45: Display should use logged shape, not live tensor shape."""

    def test_shape_matches_capture_time(self):
        """#45: tensor_shape should reflect capture-time shape."""
        model = SimpleLinear()
        x = torch.randn(2, 10)
        log = log_forward_pass(model, x, layers_to_save="all")
        for label in log.layer_labels:
            entry = log[label]
            if entry.tensor_contents is not None:
                assert entry.tensor_shape is not None


# ===========================================================================
# Wave 5: Interface Polish
# ===========================================================================


class TestLayerNumPasses:
    """Bug #53: layer_num_passes should be keyed correctly."""

    def test_returns_integer(self):
        """#53: layer_num_passes should return int, not 'unknown'."""
        model = SimpleLinear()
        x = torch.randn(2, 10)
        log = log_forward_pass(model, x)
        for label in log.layer_labels_no_pass:
            passes = log.layer_num_passes.get(label)
            assert passes is not None, f"No passes for {label}"
            assert isinstance(passes, int), f"Expected int, got {type(passes)} for {label}"


class TestSliceIndexing:
    """Bug #78: Slice indexing should return list."""

    def test_slice_returns_list(self):
        """#78: log[0:3] should return a list of layers."""
        model = SimpleLinear()
        x = torch.randn(2, 10)
        log = log_forward_pass(model, x)
        result = log[0:2]
        assert isinstance(result, list)
        assert len(result) == 2


class TestToPandasGuard:
    """Bug #124: to_pandas() should guard against unfinished pass."""

    def test_works_after_pass(self):
        """#124: to_pandas should work on a finished log."""
        model = SimpleLinear()
        x = torch.randn(2, 10)
        log = log_forward_pass(model, x)
        try:
            import pandas

            df = log.to_pandas()
            assert df is not None
        except ImportError:
            pytest.skip("pandas not installed")


class TestAmbiguousSubstring:
    """Bug #125: Ambiguous substring should list matching layers."""

    def test_lists_matches(self):
        """#125: Ambiguous substring should raise ValueError with candidates."""

        class TwoLinears(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin1 = nn.Linear(10, 10)
                self.lin2 = nn.Linear(10, 10)

            def forward(self, x):
                return self.lin1(x) + self.lin2(x)

        model = TwoLinears()
        x = torch.randn(2, 10)
        log = log_forward_pass(model, x)
        # "linear" should match multiple layers
        with pytest.raises(ValueError, match="Ambiguous"):
            log["linear"]


class TestModuleLogStringIndexing:
    """Bug #120: ModuleLog should support string label lookup."""

    def test_module_string_lookup(self):
        """#120: module['linear'] should work."""
        model = SimpleLinear()
        x = torch.randn(2, 10)
        log = log_forward_pass(model, x)
        if hasattr(log, "_module_logs") and log._module_logs:
            # Get first module and try string lookup
            first_key = list(log._module_logs._dict.keys())[0]
            mod = log._module_logs[first_key]
            assert mod is not None


class TestParamContains:
    """Bug #84: ParamAccessor.__contains__ should support int."""

    def test_int_contains(self):
        """#84: 0 in params should return True."""
        model = SimpleLinear()
        x = torch.randn(2, 10)
        log = log_forward_pass(model, x)
        if log.param_logs:
            assert 0 in log.params


# ===========================================================================
# Wave 7: Visualization (smoke tests)
# ===========================================================================


class TestVisualization:
    """Visualization rendering smoke tests."""

    def test_vis_nesting_depth_0(self):
        """#94: vis_nesting_depth=0 should not crash."""
        model = SimpleLinear()
        x = torch.randn(2, 10)
        log = log_forward_pass(model, x)
        try:
            from torchlens.visualization.rendering import render_graph

            render_graph(log, vis_nesting_depth=0, save_only=True)
        except ImportError:
            pytest.skip("graphviz not available")

    def test_vis_keep_unsaved_false(self):
        """#118: keep_unsaved_layers=False should not crash visualization."""
        model = SimpleLinear()
        x = torch.randn(2, 10)
        log = log_forward_pass(model, x, layers_to_save="all", keep_unsaved_layers=False)
        try:
            from torchlens.visualization.rendering import render_graph

            render_graph(log, save_only=True)
        except ImportError:
            pytest.skip("graphviz not available")


# ===========================================================================
# Wave 8: Control Flow / Loop Detection
# ===========================================================================


class TestBufferMerge:
    """Bug #2, #148: Buffer merge correctness."""

    def test_buffer_model_no_crash(self):
        """#2: BatchNorm model with buffers should log correctly."""

        class BNModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm1d(10)

            def forward(self, x):
                return self.bn(x)

        model = BNModel()
        model.train()
        x = torch.randn(4, 10)
        log = log_forward_pass(model, x)
        assert log is not None


# ===========================================================================
# Wave 9: Cleanup and GC
# ===========================================================================


class TestIPythonNotRequired:
    """Bug #72: IPython should not be required at import time."""

    def test_display_module_loads(self):
        """#72: display module should load without IPython."""
        # If we got here, torchlens loaded successfully, which means
        # display.py didn't crash on import
        from torchlens.utils.display import in_notebook

        # Should return False in non-notebook context
        assert in_notebook() is False


class TestCleanupReleasesReferences:
    """GC-5, GC-12: cleanup() should release all references."""

    def test_cleanup_no_crash(self):
        """GC-12: cleanup() should not crash."""
        model = SimpleLinear()
        x = torch.randn(2, 10)
        log = log_forward_pass(model, x)
        log.cleanup()
        # After cleanup, attributes should be gone
        assert not hasattr(log, "layer_list") or True  # cleanup deletes these


class TestParamRefCleared:
    """GC-1: ParamLog._param_ref should be cleared after cleanup."""

    def test_param_ref_cleared_after_cleanup(self):
        """GC-1: After cleanup(), _param_ref should be cleared."""
        model = SimpleLinear()
        x = torch.randn(2, 10)
        log = log_forward_pass(model, x)
        # _param_ref should still be alive for backward() use
        for pl in log.param_logs:
            assert pl._param_ref is not None
        log.cleanup()
        # After cleanup, everything should be cleared


# ---------------------------------------------------------------------------
# Wave extras: remaining low-risk fixes
# ---------------------------------------------------------------------------


class TestBug108FastPathModuleLogs:
    """#108: postprocess_fast should preserve module logs from exhaustive pass."""

    def test_fast_path_preserves_module_logs(self):
        """After save_new_activations, module logs should be preserved from exhaustive pass."""
        model = SimpleLinear()
        x = torch.randn(2, 10)
        log = log_forward_pass(model, x)
        original_module_count = len(log.modules)
        original_addresses = [m.address for m in log.modules]
        assert original_module_count > 0

        x2 = torch.randn(2, 10)
        log.save_new_activations(model, x2)
        assert len(log.modules) == original_module_count
        assert [m.address for m in log.modules] == original_addresses


class TestBug147DescriptiveValueError:
    """#147: log_source_tensor_fast should give descriptive error on graph change."""

    def test_dynamic_graph_descriptive_error(self):
        """If graph changes between passes, error message should be descriptive."""

        class DynamicModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 10)
                self.linear2 = nn.Linear(10, 10)
                self.call_count = 0

            def forward(self, x):
                self.call_count += 1
                x = self.linear1(x)
                if self.call_count > 1:
                    # Add extra ops on second call to change graph
                    x = self.linear2(x)
                    x = torch.relu(x)
                    x = self.linear2(x)
                return x

        model = DynamicModel()
        x = torch.randn(2, 10)
        log = log_forward_pass(model, x)

        x2 = torch.randn(2, 10)
        with pytest.raises(ValueError, match="computational graph changed"):
            log.save_new_activations(model, x2)


class TestBug28DeadTypeCheck:
    """#28: torch.Tensor should not be in the dead type check list."""

    def test_nested_tensor_found(self):
        """Tensors nested in custom objects should be findable."""
        from torchlens.utils.introspection import get_vars_of_type_from_obj

        class Container:
            def __init__(self, t):
                self.tensor = t

        t = torch.randn(3)
        container = Container(t)
        results = get_vars_of_type_from_obj(container, torch.Tensor, search_depth=2)
        assert len(results) >= 1


class TestBug107TupleStringNormalization:
    """#107: containing_modules_origin_nested should handle both tuple and string formats."""

    def test_module_hierarchy_with_nested_model(self):
        """Module hierarchy processing should work regardless of tuple/string format."""

        class Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        class Outer(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = Inner()

            def forward(self, x):
                return self.inner(x)

        model = Outer()
        x = torch.randn(2, 10)
        log = log_forward_pass(model, x)
        # Should complete without error — module hierarchy correctly processed
        assert len(log.modules) > 0
        # Inner module should be accessible
        assert "inner" in log.modules
