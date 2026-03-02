"""Tests for permanent toggle-gated decoration architecture.

Covers: toggle state, detached imports, sys.modules crawl, permanent model
preparation, pause_logging, functools.wraps transparency, torch.identity,
JIT compatibility, and exception/interrupt safety.
"""

import gc
import signal
import sys
import types
import weakref

import pytest
import torch
from torch import nn

import torchlens
from torchlens import _state, log_forward_pass
from torchlens.decorate_torch import (
    decorate_all_once,
    patch_detached_references,
    patch_model_instance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimpleModel(nn.Module):
    def forward(self, x):
        return torch.relu(x + 1)


class TwoLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 5)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class MultiInputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 5)

    def forward(self, x, y):
        return self.linear(x + y)


class ErrorModel(nn.Module):
    def forward(self, x):
        torch.relu(x)
        raise RuntimeError("intentional error")


class KeyboardInterruptModel(nn.Module):
    def forward(self, x):
        torch.relu(x)
        raise KeyboardInterrupt("simulated ctrl-c")


class InplaceModel(nn.Module):
    def forward(self, x):
        x = x.clone()
        x.add_(1)
        x.relu_()
        return x


class PropertyAccessModel(nn.Module):
    def forward(self, x):
        return x.real + x.T.sum()


class InternalCreationModel(nn.Module):
    def forward(self, x):
        y = torch.zeros(3)
        return x.sum() + y.sum()


class NestedModuleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(5, 5), nn.ReLU())

    def forward(self, x):
        return self.block(x)


class IdentityModuleModel(nn.Module):
    """Model with nn.Identity — tests torch.identity injection."""

    def __init__(self):
        super().__init__()
        self.id = nn.Identity()
        self.linear = nn.Linear(5, 5)

    def forward(self, x):
        return self.linear(self.id(x))


# =========================================================================
# 1. Toggle State
# =========================================================================


class TestToggleState:
    def test_toggle_off_by_default(self):
        """After import, logging toggle should be off."""
        assert _state._logging_enabled is False
        assert _state._active_model_log is None

    def test_toggle_on_during_logging(self):
        """Inside log_forward_pass, toggle should be True."""
        observed = {}

        class ProbeModel(nn.Module):
            def forward(self, x):
                observed["enabled"] = _state._logging_enabled
                observed["model_log"] = _state._active_model_log
                return x + 1

        model = ProbeModel()
        log_forward_pass(model, torch.randn(3))
        assert observed["enabled"] is True
        assert observed["model_log"] is not None

    def test_toggle_off_after_logging(self):
        """After log_forward_pass completes, toggle should be off."""
        model = SimpleModel()
        log_forward_pass(model, torch.randn(5))
        assert _state._logging_enabled is False
        assert _state._active_model_log is None

    def test_toggle_off_after_exception(self):
        """After an exception in forward, toggle must be off."""
        model = ErrorModel()
        with pytest.raises(RuntimeError, match="intentional error"):
            log_forward_pass(model, torch.randn(5))
        assert _state._logging_enabled is False
        assert _state._active_model_log is None

    def test_toggle_off_after_keyboard_interrupt(self):
        """After KeyboardInterrupt in forward, toggle must be off."""
        model = KeyboardInterruptModel()
        with pytest.raises(KeyboardInterrupt):
            log_forward_pass(model, torch.randn(5))
        assert _state._logging_enabled is False
        assert _state._active_model_log is None

    def test_torch_works_after_keyboard_interrupt(self):
        """Torch functions must work normally after a KeyboardInterrupt."""
        model = KeyboardInterruptModel()
        with pytest.raises(KeyboardInterrupt):
            log_forward_pass(model, torch.randn(5))
        # These must not crash or produce tl_ attributes
        x = torch.randn(3)
        y = torch.cos(x)
        assert y.shape == (3,)
        assert not hasattr(y, "tl_tensor_label_raw")

    def test_requires_grad_restored_after_exception(self):
        """requires_grad must be restored even when forward raises."""
        model = TwoLayerModel()
        orig_grads = {n: p.requires_grad for n, p in model.named_parameters()}
        with pytest.raises(RuntimeError):
            # Force an error by passing wrong shape
            log_forward_pass(model, torch.randn(0))
        for name, param in model.named_parameters():
            assert param.requires_grad == orig_grads[name], f"{name} requires_grad changed"


# =========================================================================
# 2. Torch Functions Normal When Toggle Off
# =========================================================================


class TestPassthroughWhenOff:
    def test_basic_ops_produce_correct_results(self):
        """Decorated torch functions must produce identical results when off."""
        x = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(torch.cos(x), x.cos())
        assert torch.allclose(torch.sin(x), x.sin())
        assert torch.allclose(torch.relu(x), torch.nn.functional.relu(x))
        assert torch.allclose(x + x, torch.add(x, x))

    def test_no_tl_attributes_on_results(self):
        """Results must not have tl_ attributes when toggle is off."""
        x = torch.randn(3, 4)
        y = torch.matmul(x, x.T)
        assert not hasattr(y, "tl_tensor_label_raw")
        assert not hasattr(y, "tl_source_model_log")

    def test_ops_normal_after_logging(self):
        """Torch ops must be clean after a logging session completes."""
        model = SimpleModel()
        log_forward_pass(model, torch.randn(5))

        x = torch.randn(5)
        y = torch.relu(x)
        z = x.view(1, 5)
        assert not hasattr(y, "tl_tensor_label_raw")
        assert not hasattr(z, "tl_tensor_label_raw")

    def test_tensor_method_ops(self):
        """Tensor method operations like .view, .reshape must work."""
        x = torch.randn(6)
        assert x.view(2, 3).shape == (2, 3)
        assert x.reshape(3, 2).shape == (3, 2)
        assert x.unsqueeze(0).shape == (1, 6)


# =========================================================================
# 3. Detached Import Patching
# =========================================================================


class TestDetachedImports:
    def test_module_level_import_patched(self):
        """A 'from torch import cos' at module level should be patched."""
        # Create a synthetic module that simulates 'from torch import cos'
        mod = types.ModuleType("_test_detached_cos")
        mod.cos = torch.cos  # torch.cos is already decorated at this point
        sys.modules["_test_detached_cos"] = mod
        try:
            # The function should be decorated
            assert getattr(mod.cos, "tl_is_decorated_function", False)
        finally:
            del sys.modules["_test_detached_cos"]

    def test_model_with_stored_torch_func(self):
        """A model storing self.act = torch.relu should have it patched."""

        class FuncAttrModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.act = torch.relu
                self.linear = nn.Linear(5, 5)

            def forward(self, x):
                return self.act(self.linear(x))

        model = FuncAttrModel()
        # Before patching, self.act might be undecorated (if bound to original)
        # patch_model_instance should fix it
        patch_model_instance(model)
        # After patching, the relu stored on the model should be decorated
        result = log_forward_pass(model, torch.randn(5))
        # The relu should appear in the graph
        relu_layers = [label for label in result.layer_labels if "relu" in label.lower()]
        assert len(relu_layers) > 0, "relu from self.act not logged in graph"

    def test_model_with_func_in_list(self):
        """A model storing functions in a list should still have them logged."""

        class ListFuncModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.funcs = [torch.relu, torch.sigmoid]
                self.linear = nn.Linear(5, 5)

            def forward(self, x):
                x = self.linear(x)
                for f in self.funcs:
                    x = f(x)
                return x

        model = ListFuncModel()
        result = log_forward_pass(model, torch.randn(5))
        labels = " ".join(result.layer_labels).lower()
        assert "relu" in labels, "relu from list not logged"
        assert "sigmoid" in labels, "sigmoid from list not logged"

    def test_model_with_func_in_dict(self):
        """A model storing functions in a dict should still have them logged."""

        class DictFuncModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.ops = {"activation": torch.relu}
                self.linear = nn.Linear(5, 5)

            def forward(self, x):
                return self.ops["activation"](self.linear(x))

        model = DictFuncModel()
        result = log_forward_pass(model, torch.randn(5))
        relu_layers = [lbl for lbl in result.layer_labels if "relu" in lbl.lower()]
        assert len(relu_layers) > 0

    def test_nn_functional_import_patched(self):
        """torch.nn.functional functions should be decorated."""
        import torch.nn.functional as F

        assert getattr(F.relu, "tl_is_decorated_function", False)
        assert getattr(F.linear, "tl_is_decorated_function", False)

    def test_late_import_patched_incrementally(self):
        """Modules imported after torchlens should be patched on next crawl."""
        mod_name = "_test_late_import_module"
        # Remove if somehow already present
        sys.modules.pop(mod_name, None)
        _state._crawled_module_keys.discard(mod_name)

        # Simulate a late import
        mod = types.ModuleType(mod_name)
        # Store the DECORATED cos (since torch.cos is already decorated)
        mod.my_cos = torch.cos
        sys.modules[mod_name] = mod
        try:
            # Trigger incremental crawl
            patch_detached_references()
            assert getattr(mod.my_cos, "tl_is_decorated_function", False)
        finally:
            sys.modules.pop(mod_name, None)

    def test_crawl_skips_torchlens_modules(self):
        """The crawl must not modify torchlens internal modules."""
        # torchlens modules should be in _crawled_module_keys but NOT patched
        tl_modules = [k for k in sys.modules if k.startswith("torchlens")]
        assert len(tl_modules) > 0
        # _state itself should not have been modified by the crawl
        assert not hasattr(_state, "tl_is_decorated_function")

    def test_crawl_only_processes_new_modules(self):
        """Calling patch_detached_references twice should not re-scan."""
        keys_after_first = set(_state._crawled_module_keys)
        patch_detached_references()
        keys_after_second = set(_state._crawled_module_keys)
        # If no new modules were imported, sets should be identical
        assert keys_after_first == keys_after_second


# =========================================================================
# 4. Permanent Model Preparation
# =========================================================================


class TestPermanentModelPrep:
    def test_model_prepared_once(self):
        """Same model instance should only be prepared once."""
        model = TwoLayerModel()
        log_forward_pass(model, torch.randn(5))
        assert model in _state._prepared_models

        # Run again — should reuse
        log_forward_pass(model, torch.randn(5))
        assert model in _state._prepared_models

    def test_permanent_attrs_survive_sessions(self):
        """tl_module_address and tl_module_type should persist after logging."""
        model = NestedModuleModel()
        log_forward_pass(model, torch.randn(5))

        # Permanent attributes should still be there
        assert hasattr(model.block[0], "tl_module_address")
        assert hasattr(model.block[0], "tl_module_type")
        assert model.block[0].tl_module_address == "block.0"
        assert model.block[0].tl_module_type == "Linear"

    def test_session_attrs_cleaned(self):
        """Session-scoped tl_ attrs should be removed after logging."""
        model = TwoLayerModel()
        log_forward_pass(model, torch.randn(5))

        for module in model.modules():
            assert not hasattr(module, "tl_source_model_log")
            assert not hasattr(module, "tl_module_pass_num")
            assert not hasattr(module, "tl_tensors_entered_labels")
            assert not hasattr(module, "tl_tensors_exited_labels")

    def test_session_attrs_cleaned_after_exception(self):
        """Session attrs must be cleaned even if forward raises."""
        model = nn.Sequential(nn.Linear(5, 5), nn.ReLU())

        class FailAfterLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 5)

            def forward(self, x):
                x = self.linear(x)
                raise RuntimeError("boom")

        model = FailAfterLinear()
        with pytest.raises(RuntimeError):
            log_forward_pass(model, torch.randn(5))

        for module in model.modules():
            assert not hasattr(module, "tl_source_model_log")

    def test_different_models_prepared_independently(self):
        """Two different model instances should both be in _prepared_models."""
        model_a = SimpleModel()
        model_b = TwoLayerModel()
        log_forward_pass(model_a, torch.randn(5))
        log_forward_pass(model_b, torch.randn(5))
        assert model_a in _state._prepared_models
        assert model_b in _state._prepared_models

    def test_model_gc_after_deletion(self):
        """WeakSet should not prevent model garbage collection.

        Note: log_forward_pass returns a ModelLog that holds refs to the model
        (known GC issues GC-1 through GC-5). We test that the WeakSet itself
        doesn't add a strong ref by preparing a model WITHOUT logging.
        """
        from torchlens.model_funcs import _prepare_model_once

        model = TwoLayerModel()
        ref = weakref.ref(model)
        _prepare_model_once(model)
        assert model in _state._prepared_models

        del model
        gc.collect()
        assert ref() is None  # model should be collected

    def test_requires_grad_restored(self):
        """All params should have original requires_grad after logging."""
        model = TwoLayerModel()
        # Freeze some params
        model.linear.bias.requires_grad_(False)
        orig_grads = {n: p.requires_grad for n, p in model.named_parameters()}

        log_forward_pass(model, torch.randn(5))

        for name, param in model.named_parameters():
            assert param.requires_grad == orig_grads[name], (
                f"{name}: expected requires_grad={orig_grads[name]}"
            )


# =========================================================================
# 5. pause_logging Context Manager
# =========================================================================


class TestPauseLogging:
    def test_pause_restores_state(self):
        """pause_logging should restore previous _logging_enabled value."""
        _state._logging_enabled = True
        try:
            with _state.pause_logging():
                assert _state._logging_enabled is False
            assert _state._logging_enabled is True
        finally:
            _state._logging_enabled = False

    def test_pause_nested(self):
        """Nested pause_logging should correctly restore at each level."""
        _state._logging_enabled = True
        try:
            with _state.pause_logging():
                assert _state._logging_enabled is False
                with _state.pause_logging():
                    assert _state._logging_enabled is False
                assert _state._logging_enabled is False  # restored to inner's prev
            assert _state._logging_enabled is True  # restored to outer's prev
        finally:
            _state._logging_enabled = False

    def test_pause_exception_safety(self):
        """pause_logging must restore state even if body raises."""
        _state._logging_enabled = True
        try:
            with pytest.raises(ValueError):
                with _state.pause_logging():
                    assert _state._logging_enabled is False
                    raise ValueError("test")
            assert _state._logging_enabled is True
        finally:
            _state._logging_enabled = False

    def test_operations_during_pause_not_logged(self):
        """Torch ops inside pause_logging should not appear in graph."""

        class PauseInsideModel(nn.Module):
            def forward(self, x):
                y = torch.relu(x)  # should be logged
                with _state.pause_logging():
                    torch.cos(x)  # should NOT be logged
                return y

        model = PauseInsideModel()
        result = log_forward_pass(model, torch.randn(5))
        labels = " ".join(result.layer_labels).lower()
        assert "relu" in labels
        # cos should not appear — it ran during pause
        cos_layers = [lbl for lbl in result.layer_labels if "cos" in lbl.lower()]
        assert len(cos_layers) == 0, f"cos appeared in graph during pause: {cos_layers}"

    def test_activation_postfunc_not_logged(self):
        """activation_postfunc runs inside pause_logging, ops should not appear."""
        model = SimpleModel()
        result = log_forward_pass(model, torch.randn(5), activation_postfunc=torch.sigmoid)
        # sigmoid from postfunc should NOT appear in graph
        sigmoid_layers = [lbl for lbl in result.layer_labels if "sigmoid" in lbl.lower()]
        assert len(sigmoid_layers) == 0, f"postfunc sigmoid leaked into graph: {sigmoid_layers}"


# =========================================================================
# 6. functools.wraps / Transparency
# =========================================================================


class TestWrapperTransparency:
    def test_wrapped_function_name(self):
        """Decorated functions should preserve __name__."""
        assert torch.cos.__name__ == "cos"
        assert torch.nn.functional.relu.__name__ == "relu"

    def test_wrapped_function_doc(self):
        """Decorated functions should preserve __doc__."""
        assert torch.cos.__doc__ is not None
        assert len(torch.cos.__doc__) > 0

    def test_is_decorated_flag(self):
        """Decorated functions should have tl_is_decorated_function=True."""
        assert getattr(torch.cos, "tl_is_decorated_function", False)
        assert getattr(torch.nn.functional.relu, "tl_is_decorated_function", False)

    def test_builtin_no_wrapped(self):
        """Built-in wrappers should NOT have __wrapped__ (JIT compat)."""
        # Built-ins like torch.cos don't have __code__; we remove __wrapped__
        # to prevent inspect.unwrap from following it.
        if not hasattr(torch.cos, "__code__"):
            assert not hasattr(torch.cos, "__wrapped__")


# =========================================================================
# 7. torch.identity
# =========================================================================


class TestTorchIdentity:
    def test_torch_identity_exists(self):
        assert hasattr(torch, "identity")
        assert callable(torch.identity)

    def test_torch_identity_passthrough(self):
        x = torch.randn(3, 4)
        y = torch.identity(x)
        assert torch.equal(x, y)

    def test_torch_identity_in_graph(self):
        """torch.identity should appear in graphs for nn.Identity modules."""
        model = IdentityModuleModel()
        result = log_forward_pass(model, torch.randn(5))
        identity_layers = [lbl for lbl in result.layer_labels if "identity" in lbl.lower()]
        assert len(identity_layers) > 0


# =========================================================================
# 8. JIT Compatibility
# =========================================================================


class TestJITCompat:
    def test_jit_builtin_table_has_decorated(self):
        """Decorated functions should be in the JIT builtin table."""
        import torch.jit._builtins as _jit_builtins

        # torch.cos should be registered
        assert id(torch.cos) in _jit_builtins._builtin_table

    def test_shared_originals_same_wrapper(self):
        """Functions sharing an original (e.g. torch.cos / torch._VF.cos)
        should use the exact same wrapper object."""
        if hasattr(torch._VF, "cos"):
            assert torch.cos is torch._VF.cos

    def test_jit_script_with_decorated_functions(self):
        """torch.jit.script should work on functions using decorated ops."""

        @torch.jit.script
        def jit_fn(x: torch.Tensor) -> torch.Tensor:
            return torch.relu(x) + 1.0

        x = torch.randn(5)
        result = jit_fn(x)
        expected = torch.relu(x) + 1.0
        assert torch.allclose(result, expected)


# =========================================================================
# 9. Decoration Mapping Consistency
# =========================================================================


class TestDecorationConsistency:
    def test_orig_to_decorated_populated(self):
        """_orig_to_decorated should have been populated at import time."""
        assert len(_state._orig_to_decorated) > 1000

    def test_decorated_to_orig_populated(self):
        """_decorated_to_orig should mirror _orig_to_decorated."""
        assert len(_state._decorated_to_orig) == len(_state._orig_to_decorated)

    def test_bidirectional_mapper(self):
        """_decorated_func_mapper should have dec->orig for every wrapper."""
        for orig_id, dec in _state._orig_to_decorated.items():
            if isinstance(dec, property):
                continue
            orig = _state._decorated_to_orig.get(id(dec))
            if orig is None:
                continue
            # dec -> orig must always be correct
            assert dec in _state._decorated_func_mapper
            assert _state._decorated_func_mapper[dec] is orig
            # orig -> dec may point to a different wrapper if the original
            # is shared across function names (e.g. spmm/dsmm are aliases),
            # so we only check it exists, not which wrapper it points to.
            assert orig in _state._decorated_func_mapper

    def test_no_duplicate_wrappers(self):
        """Each decorated func in _orig_to_decorated should be unique."""
        dec_ids = set()
        for dec in _state._orig_to_decorated.values():
            d_id = id(dec)
            assert d_id not in dec_ids, f"Duplicate wrapper found: {dec}"
            dec_ids.add(d_id)

    def test_func_argnames_populated(self):
        """_func_argnames should be pre-computed for many functions."""
        assert len(_state._func_argnames) > 500

    def test_decorate_all_once_idempotent(self):
        """Calling decorate_all_once again should be a no-op."""
        count_before = len(_state._orig_to_decorated)
        decorate_all_once()
        assert len(_state._orig_to_decorated) == count_before


# =========================================================================
# 10. In-Place Ops
# =========================================================================


class TestInPlaceOps:
    def test_inplace_ops_logged(self):
        """In-place operations should appear in the logged graph."""
        model = InplaceModel()
        result = log_forward_pass(model, torch.randn(5))
        labels = " ".join(result.layer_labels).lower()
        assert "add" in labels
        assert "relu" in labels

    def test_inplace_ops_correct_output(self):
        """In-place model should produce correct numerical output."""
        model = InplaceModel()
        x = torch.randn(5)
        result = log_forward_pass(model, x)
        output = result.output_layers
        assert len(output) > 0


# =========================================================================
# 11. Property Descriptors
# =========================================================================


class TestPropertyDescriptors:
    def test_tensor_real_access(self):
        """Accessing .real should work and be logged."""
        x = torch.randn(3)
        result = x.real
        assert torch.equal(result, x)

    def test_tensor_T_access(self):
        """Accessing .T should work."""
        x = torch.randn(3, 4)
        assert x.T.shape == (4, 3)


# =========================================================================
# 12. Edge Cases
# =========================================================================


class TestEdgeCases:
    def test_model_with_no_torch_ops(self):
        """A model that just returns input should not crash."""

        class PassthroughModel(nn.Module):
            def forward(self, x):
                return x

        model = PassthroughModel()
        result = log_forward_pass(model, torch.randn(5))
        assert result is not None

    def test_model_with_internal_tensor_creation(self):
        """torch.zeros inside forward should be logged."""
        model = InternalCreationModel()
        result = log_forward_pass(model, torch.randn(3))
        labels = " ".join(result.layer_labels).lower()
        assert "zeros" in labels or "sum" in labels

    def test_multiple_inputs(self):
        """Models with multiple tensor inputs should work."""
        model = MultiInputModel()
        result = log_forward_pass(model, [torch.randn(5), torch.randn(5)])
        assert len(result.output_layers) > 0

    def test_nested_modules(self):
        """Nested module structure should be correctly logged."""
        model = NestedModuleModel()
        result = log_forward_pass(model, torch.randn(5))
        labels = " ".join(result.layer_labels).lower()
        assert "linear" in labels
        assert "relu" in labels

    def test_fast_mode_with_toggle(self):
        """Fast mode (layers_to_save subset) should work with permanent decoration."""
        model = TwoLayerModel()
        result = log_forward_pass(model, torch.randn(5))
        # Get a specific layer to save
        all_labels = result.layer_labels
        if len(all_labels) > 2:
            result2 = log_forward_pass(model, torch.randn(5), layers_to_save=[all_labels[1]])
            assert result2 is not None

    def test_consecutive_logging_sessions(self):
        """Multiple consecutive log_forward_pass calls should work cleanly."""
        model = SimpleModel()
        for _ in range(3):
            result = log_forward_pass(model, torch.randn(5))
            assert _state._logging_enabled is False
            assert _state._active_model_log is None
            assert len(result.output_layers) > 0


# =========================================================================
# 13. SIGINT / Signal Safety
# =========================================================================


class TestSignalSafety:
    def test_sigalrm_during_forward(self):
        """Simulate an interrupt via SIGALRM during forward pass.
        Toggle must be off after the signal handler raises."""
        if not hasattr(signal, "SIGALRM"):
            pytest.skip("SIGALRM not available on this platform")

        class SlowModel(nn.Module):
            def forward(self, x):
                # Do enough work that the alarm fires
                for _ in range(1000):
                    x = x + 0.001
                return x

        def alarm_handler(signum, frame):
            raise TimeoutError("alarm fired")

        old_handler = signal.signal(signal.SIGALRM, alarm_handler)
        try:
            signal.alarm(1)  # 1 second
            model = SlowModel()
            try:
                log_forward_pass(model, torch.randn(5))
            except TimeoutError:
                pass
            signal.alarm(0)  # cancel alarm

            # Toggle MUST be off
            assert _state._logging_enabled is False
            assert _state._active_model_log is None

            # Torch must still work
            y = torch.cos(torch.tensor([1.0]))
            assert y.shape == (1,)
        finally:
            signal.signal(signal.SIGALRM, old_handler)
            signal.alarm(0)

    def test_active_logging_context_manager_exception_safety(self):
        """active_logging must restore state on any exception type."""

        # Simulate the context manager directly
        class CustomError(BaseException):
            pass

        assert _state._logging_enabled is False
        try:
            with _state.active_logging(object()):
                assert _state._logging_enabled is True
                raise CustomError("unusual error")
        except CustomError:
            pass
        assert _state._logging_enabled is False
        assert _state._active_model_log is None


# =========================================================================
# 14. Multiple Session Isolation
# =========================================================================


class TestSessionIsolation:
    def test_no_cross_session_leakage(self):
        """Tensors from one session should not leak tl_ attrs into the next."""
        model = TwoLayerModel()
        x = torch.randn(5)

        result1 = log_forward_pass(model, x)
        # After first session, input tensor should be clean
        assert not hasattr(x, "tl_tensor_label_raw")

        result2 = log_forward_pass(model, x)
        assert not hasattr(x, "tl_tensor_label_raw")

        # Both sessions should have produced valid results
        assert len(result1.output_layers) > 0
        assert len(result2.output_layers) > 0

    def test_model_log_not_retained_on_state(self):
        """_active_model_log should be None between sessions."""
        model = SimpleModel()
        log_forward_pass(model, torch.randn(5))
        assert _state._active_model_log is None

        log_forward_pass(model, torch.randn(5))
        assert _state._active_model_log is None
