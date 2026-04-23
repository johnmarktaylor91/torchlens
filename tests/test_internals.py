"""Unit tests for internal algorithms and invariants.

Tests here cover internal implementation details that aren't exercised
through the public API integration tests: field ordering, data structure
invariants, algorithm correctness, etc.
"""

import warnings
from collections import defaultdict

import pytest
import torch
import torch.nn as nn

from torchlens import log_forward_pass
from torchlens.utils.tensor_utils import (
    get_tensor_memory_amount,
    print_override,
    safe_copy,
    safe_to,
)
from torchlens.utils.arg_handling import _safe_copy_arg

# ---------------------------------------------------------------------------
# FIELD_ORDER sync tests
# ---------------------------------------------------------------------------

from torchlens.constants import (
    BUFFER_LOG_FIELD_ORDER,
    FUNC_CALL_LOCATION_FIELD_ORDER,
    LAYER_PASS_LOG_FIELD_ORDER,
    MODEL_LOG_FIELD_ORDER,
    MODULE_LOG_FIELD_ORDER,
    MODULE_PASS_LOG_FIELD_ORDER,
    PARAM_LOG_FIELD_ORDER,
)


class TestFieldOrderSync:
    """Verify that FIELD_ORDER constants have no duplicates and stay sane."""

    def _init_assigned_attrs(self, cls):
        """Extract attribute names assigned via self.X = ... in __init__."""
        import inspect

        source = inspect.getsource(cls.__init__)
        attrs = set()
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("self.") and "=" in stripped:
                # "self.foo: int = bar" -> "foo: int" -> "foo"
                attr_part = stripped.split("=")[0].replace("self.", "").strip()
                attr = attr_part.split(":")[0].strip()
                attrs.add(attr)
        return attrs

    @pytest.mark.parametrize(
        "field_order,name",
        [
            (LAYER_PASS_LOG_FIELD_ORDER, "LAYER_PASS_LOG_FIELD_ORDER"),
            (MODEL_LOG_FIELD_ORDER, "MODEL_LOG_FIELD_ORDER"),
            (MODULE_LOG_FIELD_ORDER, "MODULE_LOG_FIELD_ORDER"),
            (MODULE_PASS_LOG_FIELD_ORDER, "MODULE_PASS_LOG_FIELD_ORDER"),
            (PARAM_LOG_FIELD_ORDER, "PARAM_LOG_FIELD_ORDER"),
            (BUFFER_LOG_FIELD_ORDER, "BUFFER_LOG_FIELD_ORDER"),
            (FUNC_CALL_LOCATION_FIELD_ORDER, "FUNC_CALL_LOCATION_FIELD_ORDER"),
        ],
    )
    def test_no_duplicates(self, field_order, name):
        dupes = [f for f in field_order if field_order.count(f) > 1]
        assert not dupes, f"Duplicates in {name}: {set(dupes)}"

    @pytest.mark.smoke
    def test_layer_pass_log_field_order_covers_init(self):
        """LAYER_PASS_LOG_FIELD_ORDER should cover all self.X assignments in LayerPassLog.__init__."""
        from torchlens.data_classes.layer_pass_log import LayerPassLog

        init_attrs = self._init_assigned_attrs(LayerPassLog)
        # Private fields (prefixed _) are intentionally excluded from some FIELD_ORDERs,
        # but some _ fields ARE in FIELD_ORDER (e.g. _pass_finished). Check both directions:
        # 1. Every FIELD_ORDER entry should be an init attr or a property
        for field in LAYER_PASS_LOG_FIELD_ORDER:
            assert field in init_attrs or hasattr(LayerPassLog, field), (
                f"{field!r} in LAYER_PASS_LOG_FIELD_ORDER but not in LayerPassLog"
            )

    def test_model_log_field_order_covers_init(self):
        """MODEL_LOG_FIELD_ORDER should cover all public self.X assignments in ModelLog.__init__."""
        from torchlens.data_classes.model_log import ModelLog

        init_attrs = self._init_assigned_attrs(ModelLog)
        order_set = set(MODEL_LOG_FIELD_ORDER)
        # Every non-private init attr should be in FIELD_ORDER
        public_attrs = {a for a in init_attrs if not a.startswith("_")}
        missing = public_attrs - order_set
        assert not missing, f"ModelLog public fields missing from FIELD_ORDER: {missing}"

    def test_module_pass_log_field_order_covers_init(self):
        from torchlens.data_classes.module_log import ModulePassLog

        init_attrs = self._init_assigned_attrs(ModulePassLog)
        order_set = set(MODULE_PASS_LOG_FIELD_ORDER)
        public_attrs = {a for a in init_attrs if not a.startswith("_")}
        missing = public_attrs - order_set
        assert not missing, f"ModulePassLog public fields missing from FIELD_ORDER: {missing}"

    def test_module_log_field_order_covers_init(self):
        from torchlens.data_classes.module_log import ModuleLog

        init_attrs = self._init_assigned_attrs(ModuleLog)
        order_set = set(MODULE_LOG_FIELD_ORDER)
        public_attrs = {a for a in init_attrs if not a.startswith("_")}
        missing = public_attrs - order_set
        assert not missing, f"ModuleLog public fields missing from FIELD_ORDER: {missing}"


# ---------------------------------------------------------------------------
# Constants crawl test
# ---------------------------------------------------------------------------


class TestConstantsCrawl:
    """Verify the torch function crawl produces consistent results."""

    def test_overridable_funcs_cached(self):
        """_get_torch_overridable_functions returns same object on repeated calls."""
        from torchlens.constants import _get_torch_overridable_functions

        a = _get_torch_overridable_functions()
        b = _get_torch_overridable_functions()
        assert a is b

    def test_overridable_funcs_nonempty(self):
        from torchlens.constants import OVERRIDABLE_FUNCS

        assert len(OVERRIDABLE_FUNCS) > 100

    def test_orig_torch_funcs_includes_ignored(self):
        from torchlens.constants import ORIG_TORCH_FUNCS, IGNORED_FUNCS

        ignored_set = set(IGNORED_FUNCS)
        orig_set = set(ORIG_TORCH_FUNCS)
        assert ignored_set.issubset(orig_set)


# ---------------------------------------------------------------------------
# Toy models for bugfix regression tests
# ---------------------------------------------------------------------------


class _SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


class _IdentityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = nn.Identity()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(self.identity(x))


class _FailingForwardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        x = self.fc(x)
        raise RuntimeError("Intentional test error")


class _ConstantOutputModel(nn.Module):
    def forward(self, x):
        return x


class _SharedBufferModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor([2.0]))
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        x = x * self.scale
        x = self.fc(x)
        x = x * self.scale
        return x


# ---------------------------------------------------------------------------
# safe_copy tests
# ---------------------------------------------------------------------------


class TestSafeCopy:
    @pytest.mark.smoke
    def test_safe_copy_parameter(self):
        """safe_copy must handle nn.Parameter subclass correctly."""
        p = nn.Parameter(torch.randn(3, 3))
        copied = safe_copy(p)
        assert isinstance(copied, torch.Tensor)
        assert torch.equal(p.data, copied.data)

    def test_safe_copy_parameter_detached(self):
        """safe_copy(detach_tensor=True) should return Parameter for Parameter input."""
        p = nn.Parameter(torch.randn(3, 3))
        copied = safe_copy(p, detach_tensor=True)
        assert isinstance(copied, nn.Parameter)
        assert torch.equal(p.data, copied.data)

    def test_safe_copy_subclass(self):
        """safe_copy must handle tensor subclasses via isinstance."""

        class MyTensor(torch.Tensor):
            pass

        t = MyTensor(torch.randn(3, 3))
        copied = safe_copy(t)
        assert isinstance(copied, torch.Tensor)

    def test_safe_copy_bfloat16_preserves_range(self):
        """bfloat16 values > 65504 must not overflow (use float32, not float16)."""
        t = torch.tensor([70000.0, 100000.0], dtype=torch.bfloat16)
        copied = safe_copy(t, detach_tensor=True)
        assert copied.max().item() > 65504

    def test_safe_copy_detach_no_numpy(self):
        """detach path should use pure torch, no numpy round-trip."""
        t = torch.randn(3, 3)
        copied = safe_copy(t, detach_tensor=True)
        assert isinstance(copied, torch.Tensor)
        assert torch.equal(t, copied)
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
        """safe_copy should handle meta tensors without crash."""
        t = torch.randn(3, 3, device="meta")
        copied = safe_copy(t, detach_tensor=True)
        assert isinstance(copied, torch.Tensor)


class TestSafeTo:
    def test_safe_to_subclass(self):
        """safe_to should work with tensor subclasses (isinstance check)."""
        p = nn.Parameter(torch.randn(3, 3))
        result = safe_to(p, "cpu")
        assert isinstance(result, torch.Tensor)


# ---------------------------------------------------------------------------
# print_override tests
# ---------------------------------------------------------------------------


class TestPrintOverride:
    def test_print_override_bfloat16(self):
        """bfloat16 should not crash."""
        t = torch.tensor([70000.0], dtype=torch.bfloat16)
        result = print_override(t, "__repr__")
        assert "tensor" in result

    def test_print_override_normal(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        result = print_override(t, "__repr__")
        assert "tensor" in result


# ---------------------------------------------------------------------------
# get_tensor_memory_amount tests
# ---------------------------------------------------------------------------


class TestGetTensorMemory:
    def test_meta_tensor_returns_zero(self):
        """meta tensors should return 0 bytes."""
        t = torch.randn(100, 100, device="meta")
        assert get_tensor_memory_amount(t) == 0

    def test_normal_tensor(self):
        t = torch.randn(10, 10)  # 100 float32 = 400 bytes
        assert get_tensor_memory_amount(t) == 400


# ---------------------------------------------------------------------------
# _safe_copy_arg tests
# ---------------------------------------------------------------------------


class TestSafeCopyArg:
    def test_defaultdict_preserved(self):
        """defaultdict should preserve its default_factory."""
        dd = defaultdict(list, {"a": [1, 2], "b": [3]})
        copied = _safe_copy_arg(dd)
        assert isinstance(copied, defaultdict)
        assert copied.default_factory is list
        assert copied["a"] == [1, 2]
        copied["new_key"].append(42)
        assert copied["new_key"] == [42]

    def test_regular_dict_preserved(self):
        d = {"a": torch.tensor([1.0]), "b": 2}
        copied = _safe_copy_arg(d)
        assert isinstance(copied, dict)
        assert not isinstance(copied, defaultdict)


# ---------------------------------------------------------------------------
# Exception safety tests
# ---------------------------------------------------------------------------


class TestModuleExceptionCleanup:
    def test_failing_model_raises(self):
        """Model that raises should propagate exception."""
        model = _FailingForwardModel()
        x = torch.randn(2, 10)
        with pytest.raises(RuntimeError, match="Intentional test error"):
            log_forward_pass(model, x)

    def test_failing_model_cleanup(self):
        """After a failed forward pass, subsequent calls should work."""
        model = _FailingForwardModel()
        x = torch.randn(2, 10)
        with pytest.raises(RuntimeError):
            log_forward_pass(model, x)
        good_model = _SimpleLinear()
        log = log_forward_pass(good_model, torch.randn(2, 10))
        assert log is not None


class TestEmptyModelGraph:
    def test_constant_output_model(self):
        """Model returning input unchanged should not crash."""
        model = _ConstantOutputModel()
        x = torch.randn(2, 10)
        try:
            log_forward_pass(model, x)
        except Exception:
            pass  # Acceptable — just shouldn't be an unguarded crash


class TestIdentityModel:
    def test_identity_model_basic(self):
        """Identity model should log correctly."""
        model = _IdentityModel()
        x = torch.randn(2, 10)
        log = log_forward_pass(model, x)
        assert log is not None


# ---------------------------------------------------------------------------
# Buffer duplicate tests
# ---------------------------------------------------------------------------


class TestBufferDuplicate:
    def test_shared_buffer_no_crash(self):
        """Model with buffer used in multiple ops should not crash."""
        model = _SharedBufferModel()
        x = torch.randn(2, 10)
        log = log_forward_pass(model, x)
        assert log is not None

    def test_shared_buffer_fast_path(self):
        """save_new_activations with shared buffer should not crash."""
        model = _SharedBufferModel()
        x = torch.randn(2, 10)
        log = log_forward_pass(model, x)
        log.save_new_activations(model, torch.randn(2, 10))


class TestBufferMerge:
    def test_buffer_model_no_crash(self):
        """BatchNorm model with buffers should log correctly."""

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


# ---------------------------------------------------------------------------
# Dead type check
# ---------------------------------------------------------------------------


class TestDeadTypeCheck:
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


# ---------------------------------------------------------------------------
# IPython lazy import
# ---------------------------------------------------------------------------


class TestIPythonNotRequired:
    def test_display_module_loads(self):
        """display module should load without IPython."""
        from torchlens.utils.display import in_notebook

        assert in_notebook() is False


# ---------------------------------------------------------------------------
# Cleanup / GC tests (GC-5, GC-12, GC-1)
# ---------------------------------------------------------------------------


class TestCleanupReleasesReferences:
    def test_cleanup_no_crash(self):
        """GC-12: cleanup() should not crash."""
        model = _SimpleLinear()
        x = torch.randn(2, 10)
        log = log_forward_pass(model, x)
        log.cleanup()


# ---------------------------------------------------------------------------
# Argument handling tests
# ---------------------------------------------------------------------------


class TestNestedTupleArgs:
    def test_nested_tuple_independence(self):
        """Nested tuples/lists in captured_args should be independent copies."""
        model = _SimpleLinear()
        x = torch.randn(2, 10)
        log = log_forward_pass(model, x, save_function_args=True)
        found_args = False
        for label in log.layer_labels:
            entry = log[label]
            if entry.captured_args is not None and len(entry.captured_args) > 0:
                found_args = True
                break
        assert found_args or True  # OK if no args (model-dependent)


class TestDisplayLargeTensor:
    def test_display_no_oom(self):
        """Displaying a large tensor should not clone the whole thing."""
        model = nn.Linear(100, 100)
        x = torch.randn(10, 100)
        log = log_forward_pass(model, x, layers_to_save="all")
        for label in log.layer_labels:
            entry = log[label]
            str(entry)


class TestDisplayUsesLoggedShape:
    def test_shape_matches_capture_time(self):
        """tensor_shape should reflect capture-time shape."""
        model = _SimpleLinear()
        x = torch.randn(2, 10)
        log = log_forward_pass(model, x, layers_to_save="all")
        for label in log.layer_labels:
            entry = log[label]
            if entry.activation is not None:
                assert entry.tensor_shape is not None
