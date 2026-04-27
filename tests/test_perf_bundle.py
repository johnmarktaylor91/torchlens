"""Behavioral tests for the 2026-04-27 performance bundle.

The bundle ships three independent performance fixes uncovered by the
profiling audit at ``.project-context/research/profiling_audit_2026-04-27.md``:

1. **Bytecode column-offset cache** -- ``_get_col_offset`` no longer
   re-disassembles the same code object on every captured frame.
2. **Step 5 fast-skip** -- branch attribution skips the per-op AST work
   when the model has no captured terminal scalar bools (i.e. no
   conditional branches that the pipeline can attribute).
3. **CUDA probe gating** -- ``torch.cuda.empty_cache()`` calls in the
   postprocess and capture paths are gated behind the cached
   ``torch.cuda.is_available()`` so CPU-only runs skip CUDA driver
   probes entirely.

Each fix is verified behaviorally rather than via wall-clock timing
because timings are flaky in CI. A single optional benchmark probe is
provided behind ``@pytest.mark.slow`` for local sanity checking.
"""

from __future__ import annotations

import sys
import types
from unittest import mock

import pytest
import torch
import torch.nn as nn

import torchlens
from torchlens.postprocess import control_flow
from torchlens.utils import introspection


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


class _NoConditionalModel(nn.Module):
    """nn.Sequential-like model with no Python-level conditional branches."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class _ConditionalModel(nn.Module):
    """Tiny model with a real Python ``if`` branch driven by tensor data."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(x)
        if y.sum() > 0:
            y = torch.relu(y)
        else:
            y = torch.sigmoid(y)
        return y


# ---------------------------------------------------------------------------
# Fix 1 -- column-offset cache
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="_get_col_offset is a no-op on Python < 3.11",
)
class TestColOffsetCache:
    """Cache should reuse the disassembled offset map per code object."""

    def setup_method(self) -> None:
        introspection._clear_col_offset_cache()

    def teardown_method(self) -> None:
        introspection._clear_col_offset_cache()

    def _make_frame_at_offset(self, code: types.CodeType, lasti: int):
        """Return a duck-typed frame exposing the bits ``_get_col_offset`` reads."""

        return types.SimpleNamespace(f_code=code, f_lasti=lasti)

    def test_repeated_calls_reuse_disassembly(self) -> None:
        """Two lookups for the same code object disassemble it only once."""

        code = (lambda x, y: x + y).__code__

        with mock.patch.object(
            introspection.dis,
            "get_instructions",
            wraps=introspection.dis.get_instructions,
        ) as wrapped:
            introspection._get_col_offset(self._make_frame_at_offset(code, 0))
            introspection._get_col_offset(self._make_frame_at_offset(code, 0))
            introspection._get_col_offset(self._make_frame_at_offset(code, 2))

        assert wrapped.call_count == 1, (
            "Expected dis.get_instructions to run once per code object, "
            f"observed {wrapped.call_count} calls."
        )

    def test_different_code_objects_each_disassembled_once(self) -> None:
        """Each unique code object gets its own cache entry."""

        code_a = (lambda: 1).__code__
        code_b = (lambda: 2).__code__

        with mock.patch.object(
            introspection.dis,
            "get_instructions",
            wraps=introspection.dis.get_instructions,
        ) as wrapped:
            introspection._get_col_offset(self._make_frame_at_offset(code_a, 0))
            introspection._get_col_offset(self._make_frame_at_offset(code_b, 0))
            introspection._get_col_offset(self._make_frame_at_offset(code_a, 0))
            introspection._get_col_offset(self._make_frame_at_offset(code_b, 0))

        assert wrapped.call_count == 2, (
            "Expected dis.get_instructions to run once per unique code object, "
            f"observed {wrapped.call_count} calls."
        )

    def test_cache_returns_consistent_offsets(self) -> None:
        """Cached lookups agree with a fresh computation on the same instruction."""

        def sample(value: int) -> int:
            return value + 1

        code = sample.__code__

        # Pick a real instruction offset by walking the bytecode directly.
        instructions = list(introspection.dis.get_instructions(code))
        assert instructions, "Sample function must compile to at least one instruction."
        first_offset = instructions[0].offset
        expected = (
            instructions[0].positions.col_offset if instructions[0].positions is not None else None
        )

        cached = introspection._get_col_offset(self._make_frame_at_offset(code, first_offset))
        assert cached == expected

        # Hit again to ensure the cached path returns the same value.
        cached_again = introspection._get_col_offset(self._make_frame_at_offset(code, first_offset))
        assert cached_again == expected

    def test_unknown_offset_returns_none(self) -> None:
        """An offset that does not match any indexed instruction yields ``None``."""

        code = (lambda x: x).__code__
        result = introspection._get_col_offset(self._make_frame_at_offset(code, 9999))
        assert result is None


# ---------------------------------------------------------------------------
# Fix 2 -- branch attribution fast-skip
# ---------------------------------------------------------------------------


class TestBranchFastSkip:
    """When no conditional bools exist, Step 5 short-circuits the slow path."""

    def test_fast_skip_triggers_for_non_conditional_model(self) -> None:
        """A model without if-branches should never call ``attribute_op``."""

        model = _NoConditionalModel()
        x = torch.randn(2, 8)

        with (
            mock.patch.object(
                control_flow.ast_branches,
                "attribute_op",
                wraps=control_flow.ast_branches.attribute_op,
            ) as wrapped_attr,
            mock.patch.object(
                control_flow.ast_branches,
                "classify_bool",
                wraps=control_flow.ast_branches.classify_bool,
            ) as wrapped_classify,
        ):
            log = torchlens.log_forward_pass(model, x)

        assert wrapped_attr.call_count == 0, (
            "attribute_op was invoked for a model with no terminal scalar bools; "
            "the fast-skip precondition is no longer holding."
        )
        assert wrapped_classify.call_count == 0, (
            "classify_bool was invoked for a model with no terminal scalar bools."
        )

        # And the conditional collections must remain at their empty defaults.
        assert log.internally_terminated_bool_layers == []
        assert log.conditional_events == []
        assert log.conditional_branch_edges == []
        assert log.conditional_then_edges == []
        assert log.conditional_elif_edges == []
        assert log.conditional_else_edges == []
        assert log.conditional_arm_edges == {}
        assert log.conditional_edge_passes == {}

    def test_slow_path_runs_when_conditional_present(self) -> None:
        """A model with an if-branch must still run the full slow path."""

        torch.manual_seed(0)
        model = _ConditionalModel()
        x = torch.ones(1, 4)

        with mock.patch.object(
            control_flow.ast_branches,
            "attribute_op",
            wraps=control_flow.ast_branches.attribute_op,
        ) as wrapped_attr:
            log = torchlens.log_forward_pass(model, x)

        # The slow path must touch attribute_op at least once, and the
        # ConditionalEvent table must be populated.
        assert wrapped_attr.call_count > 0, (
            "Expected attribute_op to run for a model with a real conditional "
            "branch; the fast-skip is incorrectly engaging."
        )
        assert len(log.conditional_events) >= 1
        assert len(log.internally_terminated_bool_layers) >= 1

    def test_empty_model_does_not_crash(self) -> None:
        """A model that produces zero ops must not crash inside Step 5."""

        class _EmptyForward(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        # The postprocess pipeline already guards a zero-layer log
        # (``len(self._raw_layer_labels_list) == 0``) and returns early.
        # We still exercise the path to confirm the fast-skip change does
        # not introduce a regression upstream of that guard.
        model = _EmptyForward()
        x = torch.zeros(2, 2)
        log = torchlens.log_forward_pass(model, x)
        assert log.conditional_events == []


# ---------------------------------------------------------------------------
# Fix 3 -- CUDA probe gating
# ---------------------------------------------------------------------------


class TestCudaProbeGating:
    """``torch.cuda.empty_cache`` is gated behind cached ``cuda.is_available``."""

    def test_no_cuda_calls_when_unavailable(self) -> None:
        """On a CPU-only run, ``empty_cache`` and ``is_available`` see at most one call."""

        from torchlens.utils import tensor_utils

        # Reset the module-level cache so we observe the first call this test
        # makes (the conftest may have warmed it up earlier in the session).
        tensor_utils._cuda_available = None

        with (
            mock.patch.object(torch.cuda, "is_available", return_value=False) as fake_avail,
            mock.patch.object(torch.cuda, "empty_cache") as fake_empty,
        ):
            model = _NoConditionalModel()
            x = torch.randn(2, 8)
            torchlens.log_forward_pass(model, x)

        # The cached helper consults torch.cuda.is_available at most once
        # for this process; subsequent tensor_utils._is_cuda_available()
        # calls reuse the cached False value.
        assert fake_avail.call_count <= 1, (
            "torch.cuda.is_available was queried "
            f"{fake_avail.call_count} times; the cache is not being honored."
        )
        assert fake_empty.call_count == 0, (
            "torch.cuda.empty_cache must not be invoked on CPU-only runs; "
            f"observed {fake_empty.call_count} calls."
        )

        # The cache must have been populated to the False value we forced.
        assert tensor_utils._cuda_available is False

    def test_cached_value_short_circuits_subsequent_lookups(self) -> None:
        """Once cached, ``_is_cuda_available`` does not reach the torch API again."""

        from torchlens.utils import tensor_utils

        tensor_utils._cuda_available = None
        with mock.patch.object(torch.cuda, "is_available", return_value=False) as fake_avail:
            assert tensor_utils._is_cuda_available() is False
            assert tensor_utils._is_cuda_available() is False
            assert tensor_utils._is_cuda_available() is False

        assert fake_avail.call_count == 1, (
            "Repeated ``_is_cuda_available`` calls must reuse the cached value; "
            f"observed {fake_avail.call_count} torch.cuda.is_available calls."
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA-only sanity check; skipped without a real CUDA device.",
    )
    def test_cuda_path_still_runs_when_available(self) -> None:
        """On a CUDA host, ``empty_cache`` must still be invoked."""

        from torchlens.utils import tensor_utils

        tensor_utils._cuda_available = None
        # Don't mock torch.cuda.is_available so the genuine probe is used.
        with mock.patch.object(
            torch.cuda, "empty_cache", wraps=torch.cuda.empty_cache
        ) as wrapped_empty:
            model = _NoConditionalModel()
            x = torch.randn(2, 8)
            torchlens.log_forward_pass(model, x)

        assert wrapped_empty.call_count >= 1


# ---------------------------------------------------------------------------
# Optional benchmark probe (skipped in default CI)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_perf_bundle_benchmark_probe() -> None:
    """End-to-end smoke benchmark to sanity-check the bundle locally.

    Not a regression gate -- timings vary with system load. Simply confirm
    that the standard log_forward_pass still completes for a small model
    after the bundle is applied.
    """

    import time

    model = _NoConditionalModel()
    x = torch.randn(2, 8)

    # Warm-up to absorb first-call decoration cost.
    torchlens.log_forward_pass(model, x)

    start = time.perf_counter()
    for _ in range(3):
        torchlens.log_forward_pass(model, x)
    elapsed = time.perf_counter() - start

    assert elapsed < 60.0, f"Bundle benchmark unexpectedly slow: {elapsed:.2f}s"
