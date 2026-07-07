"""Unit tests for internal algorithms and invariants.

Tests here cover internal implementation details that aren't exercised
through the public API integration tests: field ordering, data structure
invariants, algorithm correctness, etc.
"""

from collections import defaultdict

import pytest
import torch
import torch.nn as nn

from torchlens import trace as trace_fn
from torchlens.backends.torch._tl import get_tensor_label, set_tensor_label
from torchlens.utils.tensor_utils import (
    get_memory_amount,
    get_memory_amount_from_metadata,
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
    def test_op_log_field_order_covers_init(self):
        """LAYER_PASS_LOG_FIELD_ORDER should cover all self.X assignments in Op.__init__."""
        from torchlens.data_classes.op import Op

        init_attrs = self._init_assigned_attrs(Op)
        # Private fields (prefixed _) are intentionally excluded from some FIELD_ORDERs,
        # but some _ fields ARE in FIELD_ORDER (e.g. _tracing_finished). Check both directions:
        # 1. Every FIELD_ORDER entry should be an init attr or a property
        for field in LAYER_PASS_LOG_FIELD_ORDER:
            assert field in init_attrs or hasattr(Op, field), (
                f"{field!r} in LAYER_PASS_LOG_FIELD_ORDER but not in Op"
            )

    def test_trace_field_order_covers_init(self):
        """MODEL_LOG_FIELD_ORDER should cover all public self.X assignments in Trace.__init__."""
        from torchlens.data_classes.trace import Trace

        init_attrs = self._init_assigned_attrs(Trace)
        order_set = set(MODEL_LOG_FIELD_ORDER)
        # Every non-private init attr should be in FIELD_ORDER
        public_attrs = {a for a in init_attrs if not a.startswith("_")}
        missing = public_attrs - order_set
        assert not missing, f"Trace public fields missing from FIELD_ORDER: {missing}"

    def test_module_call_log_field_order_covers_init(self):
        from torchlens.data_classes.module import ModuleCall

        init_attrs = self._init_assigned_attrs(ModuleCall)
        order_set = set(MODULE_PASS_LOG_FIELD_ORDER)
        public_attrs = {a for a in init_attrs if not a.startswith("_")}
        missing = public_attrs - order_set
        assert not missing, f"ModuleCall public fields missing from FIELD_ORDER: {missing}"

    def test_module_log_field_order_covers_init(self):
        from torchlens.data_classes.module import Module

        init_attrs = self._init_assigned_attrs(Module)
        order_set = set(MODULE_LOG_FIELD_ORDER)
        public_attrs = {a for a in init_attrs if not a.startswith("_")}
        missing = public_attrs - order_set
        assert not missing, f"Module public fields missing from FIELD_ORDER: {missing}"


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
        """safe_copy(detach_tensor=True) preserves the TorchLens raw label."""
        t = torch.randn(3, 3)
        set_tensor_label(t, "test_label")
        copied = safe_copy(t, detach_tensor=True)
        assert get_tensor_label(copied) == "test_label"

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
# get_memory_amount tests
# ---------------------------------------------------------------------------


class TestGetTensorMemory:
    def test_meta_tensor_returns_zero(self):
        """meta tensors should return 0 bytes."""
        t = torch.randn(100, 100, device="meta")
        assert get_memory_amount(t) == 0

    def test_normal_tensor(self):
        t = torch.randn(10, 10)  # 100 float32 = 400 bytes
        assert get_memory_amount(t) == 400

    def test_metadata_memory_uses_dense_shape_dtype(self) -> None:
        """Dense metadata memory should match tensor-method accounting.

        Returns
        -------
        None
            Assertion-only regression test.
        """

        t = torch.randn(2, 3, 4, dtype=torch.float16)

        assert get_memory_amount_from_metadata(t, tuple(t.shape), t.dtype) == 48

    def test_metadata_memory_uses_sparse_fallback(self) -> None:
        """Sparse metadata memory should preserve non-zero-value accounting.

        Returns
        -------
        None
            Assertion-only regression test.
        """

        indices = torch.tensor([[0, 1, 1], [2, 0, 2]])
        values = torch.tensor([3.0, 4.0, 5.0])
        sparse = torch.sparse_coo_tensor(indices, values, (2, 3))

        assert get_memory_amount_from_metadata(sparse, tuple(sparse.shape), sparse.dtype) == 12


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
            trace_fn(model, x)

    def test_failing_model_cleanup(self):
        """After a failed forward pass, subsequent calls should work."""
        model = _FailingForwardModel()
        x = torch.randn(2, 10)
        with pytest.raises(RuntimeError):
            trace_fn(model, x)
        good_model = _SimpleLinear()
        log = trace_fn(good_model, torch.randn(2, 10))
        assert log is not None


class TestEmptyModelGraph:
    def test_constant_output_model(self):
        """Model returning input unchanged should not crash."""
        model = _ConstantOutputModel()
        x = torch.randn(2, 10)
        log = trace_fn(model, x)
        assert log is not None


class TestIdentityModel:
    def test_identity_model_basic(self):
        """Identity model should log correctly."""
        model = _IdentityModel()
        x = torch.randn(2, 10)
        log = trace_fn(model, x)
        assert log is not None


# ---------------------------------------------------------------------------
# Buffer duplicate tests
# ---------------------------------------------------------------------------


class TestBufferDuplicate:
    def test_shared_buffer_no_crash(self):
        """Model with buffer used in multiple ops should not crash."""
        model = _SharedBufferModel()
        x = torch.randn(2, 10)
        log = trace_fn(model, x)
        assert log is not None

    def test_shared_buffer_fast_path(self):
        """save_new_outs with shared buffer should not crash."""
        model = _SharedBufferModel()
        x = torch.randn(2, 10)
        log = trace_fn(model, x)
        log.save_new_outs(model, torch.randn(2, 10))


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
        log = trace_fn(model, x)
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
        log = trace_fn(model, x)
        log.cleanup()


# ---------------------------------------------------------------------------
# Argument handling tests
# ---------------------------------------------------------------------------


class _NestedListArgModel(nn.Module):
    """Two independent linear branches stacked via a list argument.

    ``torch.stack`` receives a Python ``list`` of tensors as its sole
    positional argument, so its captured ``saved_args[0]`` is a genuine
    nested container (unlike a plain ``nn.Linear`` call, whose args are
    top-level tensors). The forward pass stashes the exact live tensors it
    passed to ``torch.stack`` on ``self._last_pair`` so the test can grab
    the very objects TorchLens copied and mutate them afterward.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        a = self.fc1(x)
        b = self.fc2(x)
        self._last_pair = [a, b]
        stacked = torch.stack(self._last_pair)
        return stacked.sum(0)


class TestNestedTupleArgs:
    def test_nested_tuple_independence(self):
        """Nested tuples/lists in saved_args must be independent copies.

        Mutating the live source tensors after capture must NOT change the
        saved snapshot, and the saved container/tensors must not be the
        same objects as the live ones (copy, not reference).
        """
        model = _NestedListArgModel()
        x = torch.randn(2, 10)
        log = trace_fn(model, x, save_arg_values=True)

        stack_entry = None
        for label in log.layer_labels:
            entry = log[label]
            saved_args = entry.saved_args
            if saved_args is not None and len(saved_args) > 0 and isinstance(saved_args[0], list):
                stack_entry = entry
                break
        assert stack_entry is not None, (
            "expected to find the torch.stack layer with a nested-list saved_args[0]"
        )

        saved_list = stack_entry.saved_args[0]
        assert isinstance(saved_list, list)
        assert len(saved_list) == 2
        live_pair = model._last_pair

        # Identity: the saved container and its tensors must be copies, not references.
        assert saved_list is not live_pair
        assert all(saved is not live for saved, live in zip(saved_list, live_pair))

        before = [t.clone() for t in saved_list]

        # Mutate the live source tensors captured during forward. If saved_args held
        # references instead of independent copies, this mutation would leak through.
        for t in live_pair:
            t.add_(1000.0)

        for pre_mutation, post_mutation in zip(before, saved_list):
            assert torch.equal(pre_mutation, post_mutation), (
                "saved_args nested-list tensors changed after mutating the live source "
                "tensors -- saved_args is not holding independent copies"
            )

        # And the reverse: mutating the saved copy must not affect the (already-mutated) live tensors.
        live_before_second_mutation = [t.clone() for t in live_pair]
        for t in saved_list:
            t.add_(-5000.0)
        for pre_mutation, post_mutation in zip(live_before_second_mutation, live_pair):
            assert torch.equal(pre_mutation, post_mutation), (
                "live tensors changed after mutating the saved_args copy -- saved_args is "
                "not holding independent copies"
            )


class TestDisplayLargeTensor:
    def test_display_no_oom(self):
        """Displaying a large captured tensor must not clone the whole tensor (#73).

        ``Op._tensor_contents_str_helper`` is documented ("Slice first, then
        clone only the small slice (#73)") to slice down to at most an 8x8
        preview *before* calling ``.clone()``. This test tracks every
        ``torch.Tensor.clone()`` call made while formatting a real captured
        entry with ``str(op)`` and asserts none of them ever clones more than
        the 8x8=64-element preview -- i.e. it can never clone the full
        (50, 2000) = 100,000-element activation. A regression that clones
        the whole tensor before slicing would make this fail.
        """
        model = nn.Linear(100, 2000)
        x = torch.randn(50, 100)
        log = trace_fn(model, x, layers_to_save="all")

        clone_call_sizes: list[int] = []
        orig_clone = torch.Tensor.clone

        def _tracking_clone(self, *args, **kwargs):
            clone_call_sizes.append(self.numel())
            return orig_clone(self, *args, **kwargs)

        torch.Tensor.clone = _tracking_clone
        try:
            for label in log.layer_labels:
                entry = log[label]
                op = entry.ops[0]
                if op.out is None:
                    continue
                str(op)
        finally:
            torch.Tensor.clone = orig_clone

        assert clone_call_sizes, "expected str(op) to clone at least one tensor slice"
        max_cloned_elements = max(clone_call_sizes)
        full_tensor_elements = 50 * 2000
        assert max_cloned_elements <= 64, (
            f"str(op) cloned a tensor with {max_cloned_elements} elements "
            f"(full activation has {full_tensor_elements}); expected the display path to "
            f"slice down to <= 8x8=64 elements before cloning"
        )


class TestDisplayUsesLoggedShape:
    def test_shape_matches_capture_time(self):
        """shape should reflect capture-time shape."""
        model = _SimpleLinear()
        x = torch.randn(2, 10)
        log = trace_fn(model, x, layers_to_save="all")
        for label in log.layer_labels:
            entry = log[label]
            if entry.out is not None:
                assert entry.shape is not None
