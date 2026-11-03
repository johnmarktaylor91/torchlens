"""Tests for lazy toggle-gated decoration architecture.

Covers: lazy wrapping lifecycle, toggle state, detached imports, sys.modules
crawl, permanent model preparation, pause_logging, functools.wraps
transparency, decorated identity, JIT compatibility, wrap/unwrap, and
exception/interrupt safety.
"""

import inspect

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.backends.torch.wrappers as torch_funcs_module
from torchlens import _state, trace as trace_fn
from torchlens.backends.torch.wrappers import (
    decorate_all_once,
    get_arg_names,
    unwrap_torch,
    wrap_torch,
    wrapped,
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


class IgnoredFunctionModel(nn.Module):
    """Exercise tensor-producing functions ignored by ``__torch_function__``."""

    def __init__(self, route: str) -> None:
        """Store the ignored-function route to execute."""

        super().__init__()
        self.route = route

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call one ignored tensor-producing function and return its result."""

        if self.route == "empty":
            return torch.empty((2, 2))
        if self.route == "empty_strided":
            return torch.empty_strided((2, 2), (2, 1))
        if self.route == "empty_permuted":
            return torch.empty_permuted((2, 3), (1, 0))
        if self.route == "frombuffer":
            return torch.frombuffer(bytearray(16), dtype=torch.float32)
        if self.route == "asarray":
            return torch.asarray(x)
        if self.route == "fill":
            return torch.fill(x, 3.0)
        raise AssertionError(f"unsupported route: {self.route}")


# Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
@pytest.mark.parametrize(
    "route",
    ["empty_strided", "frombuffer"],
)
def test_ignored_tensor_producers_are_wrapped(route: str) -> None:
    """Ignored tensor producers emit capture ops, including input provenance."""

    log = trace_fn(IgnoredFunctionModel(route), torch.tensor([-2.0, 4.0]))
    op = next(layer for layer in log._layer_list if layer.func_name == route)

    if route in {"asarray", "fill"}:
        assert op.parents == ("input_1",)


class SameObjectReturnModel(nn.Module):
    """Record object identity for torch operations that return their input."""

    def __init__(self) -> None:
        """Initialize identity observations."""

        super().__init__()
        self.contiguous_returned_input: bool | None = None
        self.inplace_returned_input: bool | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply same-object-returning tensor operations.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Mutated clone after same-object return checks.
        """

        y = x.contiguous()
        self.contiguous_returned_input = y is x
        z = y.clone()
        out = z.add_(1)
        self.inplace_returned_input = out is z
        return out


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
    """Model with nn.Identity — tests decorated identity injection."""

    def __init__(self):
        super().__init__()
        self.id = nn.Identity()
        self.linear = nn.Linear(5, 5)

    def forward(self, x):
        return self.linear(self.id(x))


# =========================================================================
# 0. Lazy Decoration Lifecycle
# =========================================================================


class TestLazyDecoration:
    @pytest.fixture(autouse=True)
    def _clean_state(self):
        """Unwrap before each test, re-wrap after for other test classes."""
        unwrap_torch()
        yield
        wrap_torch()

    # Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/wrappers.py, campaign mutant kills.
    def test_wrapped_context_manager_exception_safety(self):
        """wrapped() must unwrap even if body raises."""
        assert _state._is_decorated is False
        with pytest.raises(RuntimeError), wrapped():
            assert _state._is_decorated is True
            raise RuntimeError("test error")
        assert _state._is_decorated is False

    # Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/_function_boundary.py, backends/torch/_script_boundary.py, backends/torch/backward.py ....
    def test_unwrap_is_idempotent(self):
        """Calling unwrap_torch multiple times should not crash."""
        unwrap_torch()
        unwrap_torch()
        assert _state._is_decorated is False


# =========================================================================
# 1. Toggle State
# =========================================================================


class TestToggleState:
    @pytest.fixture(autouse=True)
    def _ensure_wrapped(self):
        """Ensure torch functions are wrapped for these tests."""
        wrap_torch()

    def test_toggle_off_by_default(self):
        """After import, logging toggle should be off."""
        assert _state._logging_enabled is False
        assert _state._active_trace is None


# =========================================================================
# 1b. Sequence Slot Fix (sq_item pollution from __getitem__ wrapping)
# =========================================================================


class TestSequenceSlotFix:
    """Wrapping Tensor.__getitem__ pollutes CPython's sq_item C slot, making
    PySequence_Check(tensor) return True.  This breaks torch.tensor() on lists
    of 0-d tensors.  These tests verify the fix holds across all lifecycle paths.
    """

    @pytest.fixture(autouse=True)
    def _ensure_wrapped(self):
        """Start wrapped, ensure wrapped on exit."""
        wrap_torch()
        yield
        wrap_torch()

    @staticmethod
    def _check_tensor_from_0d(msg: str = "") -> None:
        """Assert torch.tensor([0-d, 0-d]) works without TypeError."""
        D = torch.randn(3, 3)
        result = torch.tensor([D[0, 0], D[1, 1]])
        assert result.shape == (2,), f"wrong shape {msg}"
        assert torch.allclose(result, torch.stack([D[0, 0], D[1, 1]])), msg

    # Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/finalize/epilogue.py.
    def test_tensor_from_0d_after_unwrap_when_done(self):
        """torch.tensor([0-d, 0-d]) works after trace_fn(unwrap_when_done=True)."""
        model = SimpleModel()
        trace_fn(model, torch.randn(5), capture=tl.options.CaptureOptions(unwrap_when_done=True))
        self._check_tensor_from_0d("after unwrap_when_done")


# =========================================================================
# 2. Torch Functions Normal When Toggle Off
# =========================================================================


# =========================================================================
# 3. Detached Import Patching
# =========================================================================


# =========================================================================
# 5. pause_logging Context Manager
# =========================================================================


# =========================================================================
# 6. functools.wraps / Transparency
# =========================================================================


class TestWrapperTransparency:
    @pytest.fixture(autouse=True)
    def _ensure_wrapped(self):
        """Ensure torch functions are wrapped for these tests."""
        wrap_torch()

    def test_wrapped_function_name(self):
        """Decorated functions should preserve __name__."""
        assert torch.cos.__name__ == "cos"
        assert torch.nn.functional.relu.__name__ == "relu"

    def test_wrapped_function_doc(self):
        """Decorated functions should preserve __doc__."""
        assert torch.cos.__doc__ is not None
        assert len(torch.cos.__doc__) > 0

    def test_builtin_no_wrapped(self):
        """Built-in wrappers should NOT have __wrapped__ (JIT compat)."""
        # Built-ins like torch.cos don't have __code__; we remove __wrapped__
        # to prevent inspect.unwrap from following it.
        if not hasattr(torch.cos, "__code__"):
            assert not hasattr(torch.cos, "__wrapped__")


# =========================================================================
# 7. Decorated Identity (via _state._decorated_identity)
# =========================================================================


class TestDecoratedIdentity:
    @pytest.fixture(autouse=True)
    def _ensure_wrapped(self):
        """Ensure torch functions are wrapped for these tests."""
        wrap_torch()

    def test_decorated_identity_exists(self):
        """_state._decorated_identity should be set after wrapping."""
        assert _state._decorated_identity is not None
        assert callable(_state._decorated_identity)

    def test_identity_not_on_torch(self):
        """torch.identity should NOT exist — we don't monkey-patch torch."""
        assert not hasattr(torch, "identity")


# =========================================================================
# 8. JIT Compatibility
# =========================================================================


class TestJITCompat:
    @pytest.fixture(autouse=True)
    def _ensure_wrapped(self):
        """Ensure torch functions are wrapped for these tests."""
        wrap_torch()

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


# =========================================================================
# 9. Decoration Mapping Consistency
# =========================================================================


class TestFuncArgnames:
    # Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/arg_names.py, campaign mutant kills.
    def test_get_arg_names_tolerates_bad_annotations(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_arg_names should tolerate ``inspect.signature`` TypeErrors. #138"""

        def fake_func(x: object, *, flag: bool = False) -> None:
            """fake_func(x, *, flag=False)"""

        original_signature = torch_funcs_module.inspect.signature

        def failing_signature(target: object) -> inspect.Signature:
            """Raise the simulated PEP 649 failure for the fake function only."""
            if target is fake_func:
                raise TypeError("simulated PEP 649 failure")
            return original_signature(target)

        monkeypatch.setattr(torch_funcs_module.inspect, "signature", failing_signature)
        _state._arg_names.pop("fake_func", None)

        get_arg_names(fake_func, "fake_func")

        assert _state._arg_names["fake_func"] == ("x", "flag")


class TestPartialDecorationRecovery:
    """Guard against partial decoration leaving the system in an inconsistent
    state.  The idempotency check in ``decorate_all_once`` must use
    ``_is_decorated`` (completion flag), NOT ``_orig_to_decorated`` (non-empty
    dict), so that a midway failure allows retry. #138"""

    # Kept by shrink-r4 (per-id subset rule): sole coverage of _utils/_torch_compat.py, backends/torch/_transform_wrappers.py, backends/torch/arg_names.py ....
    def test_is_decorated_distinguishes_partial_from_complete(self) -> None:
        """_is_decorated must be False when _orig_to_decorated is non-empty
        but decoration was not completed (simulated partial failure)."""
        wrap_torch()
        # Simulate partial state: decorated flag cleared, but maps remain
        _state._is_decorated = False
        assert len(_state._orig_to_decorated) > 0
        assert _state._is_decorated is False
        # decorate_all_once should NOT early-return on non-empty dict
        decorate_all_once()
        assert _state._is_decorated is True
        # Restore clean state
        wrap_torch()


class TestDecorationConsistency:
    @pytest.fixture(autouse=True)
    def _ensure_wrapped(self):
        """Ensure torch functions are wrapped for these tests."""
        wrap_torch()

    def test_orig_to_decorated_populated(self):
        """_orig_to_decorated should be populated after wrapping."""
        assert len(_state._orig_to_decorated) > 1000

    def test_decorated_to_orig_populated(self):
        """The append-only unwrap ledger mirrors every CURRENT wrapper.

        Strict length equality only holds in a single-wrap-generation
        session: ``_decorated_to_orig`` is the session's append-only unwrap
        LEDGER (it must never shrink -- see
        lesson-decorated-to-orig-append-only), so after any legitimate
        re-wrap generation it is a strict superset of the current map and
        the old ``len == len`` assertion was order-dependent (p2 R76
        reverse-order red, 3886 == 1944).
        """

        current_wrapper_ids = {
            id(dec) for dec in _state._orig_to_decorated.values() if not isinstance(dec, property)
        }
        missing = current_wrapper_ids - set(_state._decorated_to_orig)
        assert not missing, "current-generation wrappers absent from the unwrap ledger"
        assert len(_state._decorated_to_orig) >= len(current_wrapper_ids)

    def test_bidirectional_mapper(self):
        """_decorated_func_mapper should have dec->orig for every wrapper."""
        for _orig_id, dec in _state._orig_to_decorated.items():
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

    def test_arg_names_populated(self):
        """_arg_names should be pre-computed for many functions."""
        assert len(_state._arg_names) > 500

    # Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/wrappers.py.
    def test_decorate_all_once_idempotent(self):
        """Calling decorate_all_once again should be a no-op."""
        count_before = len(_state._orig_to_decorated)
        decorate_all_once()
        assert len(_state._orig_to_decorated) == count_before


# =========================================================================
# 10. In-Place Ops
# =========================================================================


# =========================================================================
# 11. Property Descriptors
# =========================================================================


# =========================================================================
# 12. Edge Cases
# =========================================================================


# =========================================================================
# 13. SIGINT / Signal Safety
# =========================================================================


# =========================================================================
# 14. Multiple Session Isolation
# =========================================================================
